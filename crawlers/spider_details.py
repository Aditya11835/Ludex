import time
import re
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from threading import Lock

import requests
from bs4 import BeautifulSoup
import undetected_chromedriver as uc


# ==========================================
# CONFIG

DATA_DIR = Path("data/raw")
INPUT_CSV = DATA_DIR / "search_basic_clean.csv"
OUTPUT_CSV = DATA_DIR / "game_details.csv"

NUM_WORKERS = 6          
REQUEST_DELAY = 0.8      # per-thread polite delay

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ==========================================
# THREAD GLOBALS

progress = 0
progress_lock = Lock()


# ==========================================
# HELPERS

def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] Ensured folder: {DATA_DIR}")


def get_steam_cookies():
    """Use undetected Chrome once to bypass Cloudflare & age checks."""
    print("[init] Launching browser to bypass Cloudflare…")

    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")

    browser = uc.Chrome(options=options)
    cookies = {}
    try:
        browser.get("https://store.steampowered.com/app/730")
        time.sleep(7)
        cookies = {c["name"]: c["value"] for c in browser.get_cookies()}
    finally:
        browser.quit()

    cookies.update({
        "birthtime": "189345601",
        "mature_content": "1",
        "wants_mature_content": "1",
        "lastagecheckage": "1-January-1980",
    })

    print(f"[init] Cookies obtained: {len(cookies)} entries.")
    return cookies


def make_session(cookies_dict):
    """Each thread gets its own session."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Referer": "https://store.steampowered.com/",
    })
    s.cookies.update(cookies_dict)
    return s


def clean_url(url):
    """Strip query params."""
    m = re.match(r"(https://store\.steampowered\.com/app/\d+)", url)
    if m:
        return m.group(1)
    return url.split("?")[0]


def fetch_html(session: requests.Session, url: str) -> str:
    full = url + "?l=english&cc=US"
    r = session.get(full, timeout=25, allow_redirects=True)

    if "agecheck" in r.url:
        print("  [warn] Agecheck redirect")

    if "problem fulfilling your request" in r.text:
        print("  [warn] Steam error page")

    r.raise_for_status()
    return r.text


def fetch_with_retry(session, url, retries=3):
    for attempt in range(retries):
        try:
            return fetch_html(session, url)
        except Exception as e:
            print(f"   [retry {attempt+1}/{retries}] {url} — {e}")
            time.sleep(1 + random.random() * 2)
    raise RuntimeError(f"Failed {url} after {retries} retries.")


def parse_detail_html(appid: int, title: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # Developers
    developers = sorted(set(
        a.get_text(strip=True)
        for a in soup.select("div.dev_row a")
        if a.get_text(strip=True)
    ))

    publishers = []
    genres = []

    for block in soup.select("div.details_block"):
        for b in block.find_all("b"):
            label = b.get_text(strip=True)

            if "Publisher" in label:
                pubs = [a.get_text(strip=True) for a in b.parent.find_all("a")]
                publishers.extend(pubs)

            if "Genre" in label:
                gens = [a.get_text(strip=True) for a in b.parent.find_all("a")]
                genres.extend(gens)

    publishers = sorted(set(publishers))
    genres = sorted(set(genres))

    # Tags
    tags = sorted(set(
        a.get_text(strip=True).rstrip(",")
        for a in soup.select("a.app_tag, a.app_tag_trends")
    ))

    # Description
    desc_el = soup.select_one("#game_area_description")
    if desc_el:
        description = desc_el.get_text(separator=" ", strip=True)
    else:
        snip = soup.select_one("div.game_description_snippet")
        description = snip.get_text(strip=True) if snip else ""

    return {
        "appid": appid,
        "title": title,
        "developers": "; ".join(developers),
        "publishers": "; ".join(publishers),
        "genres": "; ".join(genres),
        "tags": "; ".join(tags),
        "description": description,
    }


# ==========================================
# WORKER (PARALLEL)

def worker(game_row, cookies, TOTAL_GAMES):
    """One parallel task: fetch + parse a game."""
    global progress

    session = make_session(cookies)

    appid = int(game_row["appid"])
    title = game_row["title"][:40]
    url = clean_url(game_row["detail_link"])

    # Thread-safe progress counter
    with progress_lock:
        progress += 1
        p = progress

    print(f"[worker] ({p}/{TOTAL_GAMES}) appid={appid} title='{title}'")

    time.sleep(REQUEST_DELAY + random.random() * 0.3)

    html = fetch_with_retry(session, url)
    return parse_detail_html(appid, title, html)


# ==========================================
# MAIN

def crawl_details_parallel():
    ensure_directories()

    if not INPUT_CSV.exists():
        print("❌ Missing:", INPUT_CSV)
        return

    games = list(csv.DictReader(open(INPUT_CSV, encoding="utf-8-sig")))
    TOTAL_GAMES = len(games)

    print(f"[load] Loaded {TOTAL_GAMES} games")
    cookies = get_steam_cookies()

    results = []
    errors = 0

    print(f"[thread] Starting ThreadPool with {NUM_WORKERS} workers…")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {
            ex.submit(worker, g, cookies, TOTAL_GAMES): g
            for g in games
        }

        for fut in as_completed(futures):
            game = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"❌ Error appid={game['appid']} — {e}")
                errors += 1

    print(f"[done] Completed. {len(results)} results, {errors} errors.")
    print(f"[save] Saving → {OUTPUT_CSV}")

    fieldnames = ["appid", "title", "developers", "publishers",
                  "genres", "tags", "description"]

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("[finish] Done.")


if __name__ == "__main__":
    crawl_details_parallel()
