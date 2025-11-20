import time
import re
import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# ==========================================
# CONFIG

DATA_DIR = Path("data/raw")
OUTPUT_CSV = DATA_DIR / "search_basic.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Charts pages (Top Selling / Most Played)
CHART_URLS = [
    "https://store.steampowered.com/charts/topselling/global",
    "https://store.steampowered.com/charts/mostplayed",
]

# "New & Trending" page
NEW_URL = "https://store.steampowered.com/explore/new/"

# Global search (ignore preferences) infinite scroll endpoint
SEARCH_RESULTS_TEMPLATE = (
    "https://store.steampowered.com/search/results/"
    "?term=&ignore_preferences=1"
    "&start={start}&count={count}&infinite=1&l=english&cc=US"
)

# New global target: about 12k unique games after dedup
DESIRED_UNIQUE_TOTAL = 20_000

# How many Top Rated games per category page
PER_CATEGORY_LIMIT = 200

# Category URLs whose "Top Rated" section we want to crawl
CATEGORY_URLS = [
    "https://store.steampowered.com/category/action_fps?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/action_tps?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/hack_and_slash?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/arcade_rhythm?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/action_run_jump?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/fighting_martial_arts?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/hidden_object?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/casual?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/metroidvania?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/puzzle_matching?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/adventure_rpg?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/visual_novel?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/story_rich?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/rpg_action?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/rpg_strategy_tactics?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/rpg_jrpg?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/rogue_like_rogue_lite?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/rpg_turn_based?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/rpg_party_based?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sim_building_automation?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sim_hobby_sim?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sim_dating?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sim_farming_crafting?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sim_space_flight?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sim_life?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sim_physics_sandbox?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/strategy_turn_based?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/strategy_real_time?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/tower_defense?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/strategy_card_board?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/strategy_cities_settlements?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/strategy_grand_4x?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/strategy_military?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sports_sim?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/racing?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/racing_sim?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/sports?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/horror?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/science_fiction?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/exploration_open_world?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/anime?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/survival?flavor=contenthub_toprated",
    "https://store.steampowered.com/category/mystery_detective?flavor=contenthub_toprated",
    "https://store.steampowered.com/adultonly?flavor=contenthub_toprated"
]


# --- Steam "appdetails" API ---
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"

# Non-game types we want to exclude (ONLY from global search)
NON_GAME_TYPES = {
    "dlc",
    "music",
    "demo",
    "mod",
    "video",
    "advertising",
    "hardware",
    "guide",
    "application"
}

# polite delay between appdetails calls (single appid)
APPDETAILS_SLEEP_SECONDS = 0.1  


# ==========================================
# BASIC HELPERS

def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] Ensured folder: {DATA_DIR}")


def get_steam_cookies():
    print("[init] Launching browser to bypass Cloudflare...")

    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")

    browser = uc.Chrome(options=options)
    cookies = {}
    try:
        browser.get("https://store.steampowered.com/")
        time.sleep(6)
        cookies = {c["name"]: c["value"] for c in browser.get_cookies()}
    finally:
        browser.quit()

    return cookies


def normalize_detail_link(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://store.steampowered.com" + href
    return "https://store.steampowered.com" + href


def looks_like_price(text: str) -> bool:
    """
    Heuristic: detect price-like strings (₹ 480, $19.99, -50%, etc.)
    """
    if not text:
        return False

    t = text.strip()
    lower = t.lower()

    # obvious currency or discount markers
    if any(sym in t for sym in ["₹", "$", "€", "£", "¥"]):
        return True
    if "%" in t or "off" in lower:
        return True

    # mostly digits, dots, commas, spaces, minus
    if re.fullmatch(r"[\d\s.,-]+", t):
        return True

    return False


def extract_best_title_from_anchor(a) -> str:
    """
    Try hard to get the game title, not the price.

    1. Prefer known title classes:
       - .tab_item_name          (New & Trending)
       - span.title / .title     (Search results)
       - .game_name              (some legacy layouts)
       - div._1n_4-zvf0n4aqGEksbgW9N  (current Top Sellers layout)
    2. Otherwise, scan all text inside <a> and pick the longest non-price string.
    """

    # 1) Common/known title locations
    name_el = (
        a.select_one(".tab_item_name")
        or a.select_one("span.title")
        or a.select_one(".title")
        or a.select_one(".game_name")
        # Top Sellers: hashed class Steam currently uses for the title
        or a.select_one("div._1n_4-zvf0n4aqGEksbgW9N")
    )
    if name_el:
        txt = name_el.get_text(strip=True)
        if txt and not looks_like_price(txt):
            return txt

    # 2) Fallback: scan all text segments inside anchor and skip price-like strings
    candidates = []
    for s in a.stripped_strings:
        s = s.strip()
        if not s:
            continue
        if looks_like_price(s):
            continue
        candidates.append(s)

    if candidates:
        # choose the longest candidate (usually the proper game name)
        return max(candidates, key=len)

    # 3) Ultimate fallback: raw anchor text
    return a.get_text(strip=True)


def parse_generic_page(html: str, source_url: str):
    """
    Used for:
      - CHART_URLS pages (Top selling / Most played)
      - NEW_URL page (New & Trending)
    """
    soup = BeautifulSoup(html, "html.parser")
    anchor_tags = soup.find_all("a", href=re.compile(r"/app/\d+"))
    results = []

    for a in anchor_tags:
        try:
            href = a.get("href", "")
            if not href:
                continue

            detail_link = normalize_detail_link(href)
            m = re.search(r"/app/(\d+)", detail_link)
            if not m:
                continue

            appid = int(m.group(1))

            # robust title extraction
            title = extract_best_title_from_anchor(a)

            # fallback: slug from URL if still empty or looks like price
            if not title or looks_like_price(title):
                slug_match = re.search(r"/app/\d+/([^/?#]+)/?", detail_link)
                if slug_match:
                    title = slug_match.group(1).replace("_", " ")

            results.append({
                "appid": appid,
                "title": title,
                "detail_link": detail_link,
            })

        except Exception:
            pass

    return results


def fetch_html(url, cookies):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html",
    }
    resp = requests.get(url, headers=headers, cookies=cookies, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_search_chunk(start, count, cookies):
    """Call the infinite-scroll search JSON endpoint."""
    url = SEARCH_RESULTS_TEMPLATE.format(start=start, count=count)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://store.steampowered.com/search/?term=&ignore_preferences=1",
    }
    resp = requests.get(url, headers=headers, cookies=cookies, timeout=30)

    try:
        return resp.json()
    except Exception:
        return {"results_html": "", "total_count": 0}


def parse_search_results_html(html: str):
    """Parse the HTML fragment returned in search JSON."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("a.search_result_row")
    results = []

    for row in rows:
        try:
            appid_attr = row.get("data-ds-appid")
            if not appid_attr:
                continue
            appid = int(appid_attr)

            # Titles are inside <span class="title">...</span>
            title_el = row.select_one("span.title")
            title = title_el.get_text(strip=True) if title_el else ""

            detail_link = normalize_detail_link(row.get("href", ""))

            # If somehow we got a price-like string or empty, fall back to slug
            if not title or looks_like_price(title):
                slug_match = re.search(r"/app/\d+/([^/?#]+)/?", detail_link)
                if slug_match:
                    title = slug_match.group(1).replace("_", " ")

            results.append({
                "appid": appid,
                "title": title,
                "detail_link": detail_link,
            })
        except Exception:
            pass

    return results


def crawl_charts_with_browser():
    print("[charts] Launching browser for charts pages...")
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")

    browser = uc.Chrome(options=options)
    all_games = []

    try:
        for url in CHART_URLS:
            print(f"[charts] Visiting {url}")
            browser.get(url)
            time.sleep(6)
            html = browser.page_source
            games = parse_generic_page(html, url)
            print(f"[charts] Parsed {len(games)}")
            all_games.extend(games)
    finally:
        browser.quit()

    return all_games


# ==========================================
# CATEGORY "TOP RATED" PARSING

def parse_category_top_rated(html: str, source_url: str):
    """
    Parse Top Rated category pages, handling:
      - Compact widgets:  div.StoreSaleWidgetOuterContainer + .StoreSaleWidgetTitle
      - Expanded cards:   div.LibraryAssetExpandedDisplay + hero img alt
      - (Optional) old large_cap layout
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    used = set()

    # ---------- LAYOUT A: compact content-hub widgets ----------
    for widget in soup.select("div.StoreSaleWidgetOuterContainer"):
        # main app link (avoid the #app_reviews_hash links)
        main_link = None
        for a in widget.find_all("a", href=True):
            if re.search(r"/app/\d+", a["href"]) and "#app_reviews_hash" not in a["href"]:
                main_link = a
                break
        if not main_link:
            continue

        href = main_link["href"]
        detail_link = normalize_detail_link(href)
        m = re.search(r"/app/(\d+)", detail_link)
        if not m:
            continue
        appid = int(m.group(1))
        if appid in used:
            continue

        title_el = widget.select_one(".StoreSaleWidgetTitle")
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            # fallback: hero img alt or slug
            hero_img = widget.select_one("img[alt]")
            if hero_img:
                title = hero_img["alt"].strip()
            else:
                slug = re.search(r"/app/\d+/([^/?#]+)/?", detail_link)
                title = slug.group(1).replace("_", " ") if slug else ""

        used.add(appid)
        results.append({"appid": appid, "title": title, "detail_link": detail_link})

    # ---------- LAYOUT B: expanded content-hub cards ----------
    for card in soup.select("div.LibraryAssetExpandedDisplay"):
        a = card.find("a", href=re.compile(r"/app/\d+"))
        if not a:
            continue
        href = a.get("href", "")
        if not href:
            continue
        detail_link = normalize_detail_link(href)
        m = re.search(r"/app/(\d+)", detail_link)
        if not m:
            continue
        appid = int(m.group(1))
        if appid in used:
            continue

        hero_img = card.select_one("img[alt]")
        if hero_img:
            title = hero_img["alt"].strip()
        else:
            slug = re.search(r"/app/\d+/([^/?#]+)/?", detail_link)
            title = slug.group(1).replace("_", " ") if slug else ""

        used.add(appid)
        results.append({"appid": appid, "title": title, "detail_link": detail_link})

    # ---------- Optional: very old large_cap layout ----------
    for cap in soup.select("div.large_cap"):
        a = cap.find("a", href=re.compile(r"/app/\d+"))
        if not a:
            continue
        href = a.get("href", "")
        if not href:
            continue

        detail_link = normalize_detail_link(href)
        m = re.search(r"/app/(\d+)", detail_link)
        if not m:
            continue
        appid = int(m.group(1))
        if appid in used:
            continue

        t = cap.select_one("div.large_cap_title")
        if t:
            title = t.get_text(strip=True)
        else:
            slug = re.search(r"/app/\d+/([^/?#]+)/?", detail_link)
            title = slug.group(1).replace("_", " ") if slug else ""

        used.add(appid)
        results.append({"appid": appid, "title": title, "detail_link": detail_link})

    return results


def crawl_categories_top_rated():
    print("[categories] Launching browser for category pages...")

    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    browser = uc.Chrome(options=options)

    all_cat_games = []
    try:
        for url in CATEGORY_URLS:
            print(f"[categories] Visiting {url}")
            browser.get(url)
            time.sleep(6)

            seen_appids = set()
            collected = []
            stagnant_rounds = 0
            MAX_STAGNANT = 10  # how many loops with no new games before giving up

            while len(collected) < PER_CATEGORY_LIMIT and stagnant_rounds < MAX_STAGNANT:
                html = browser.page_source
                games = parse_category_top_rated(html, url)

                before = len(seen_appids)
                for g in games:
                    appid = g["appid"]
                    if appid in seen_appids:
                        continue
                    seen_appids.add(appid)
                    collected.append(g)
                    if len(collected) >= PER_CATEGORY_LIMIT:
                        break

                after = len(seen_appids)
                if after == before:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0

                if len(collected) >= PER_CATEGORY_LIMIT:
                    break

                # Try clicking "Show more" (this is what actually loads more entries)
                try:
                    show_more = browser.find_element(
                        By.XPATH, "//button[contains(., 'Show more')]"
                    )
                    browser.execute_script("arguments[0].click();", show_more)
                    time.sleep(2)
                except NoSuchElementException:
                    # no show-more button -> nothing else to load; bump stagnant counter
                    stagnant_rounds += 1
                    time.sleep(1)

            print(
                f"[categories] {url} → collected {len(collected)} (limit {PER_CATEGORY_LIMIT})"
            )
            all_cat_games.extend(collected)

    finally:
        browser.quit()

    print(f"[categories] Total raw category games: {len(all_cat_games)}")
    return all_cat_games





# ==========================================
# DEDUP HELPER

def upsert_games(master: dict, games, label: str = ""):
    """
    Dedup insert into master dict keyed by appid.
    If appid already exists, we keep the first occurrence by default.
    `games` can be any iterable of dicts.
    """
    before = len(master)
    for g in games:
        if not g:
            continue
        appid = g.get("appid")
        if not appid:
            continue
        if appid not in master:
            master[appid] = g
    after = len(master)
    if label:
        print(f"[dedup] After {label}: {after} unique (added {after - before})")


# ==========================================
# STEAM APPDETAILS FILTERING (SEARCH-ONLY)

def fetch_appdetails_single(appid: int):
    """
    Call Steam's appdetails API for a single appid.

    Returns the inner object for this appid:
      {"success": true, "data": {...}}
    or {} on failure.
    """
    params = {
        "appids": str(appid),
        "cc": "US",
        "l": "english",
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    try:
        resp = requests.get(APPDETAILS_URL, params=params, headers=headers, timeout=8)
    except Exception as e:
        print(f"[steamapi] Request error for appid {appid}: {e}")
        return {}

    try:
        data = resp.json()
    except Exception as e:
        print(f"[steamapi] JSON decode error for appid {appid}: {e}")
        return {}

    # Sometimes Steam returns JSON null or something odd
    if data is None or not isinstance(data, dict):
        return {}

    return data.get(str(appid), {}) or {}


def filter_non_games_via_steam_api(all_games: dict) -> dict:
    """
    Use Steam's appdetails API to remove DLC, music, demos, mods, etc.
    We call appdetails for *every* appid here, and drop only when
    appdetails says the type is clearly non-game.

    all_games: {appid: {"appid": ..., "title": ..., "detail_link": ...}, ...}

    NOTE: In this script, we apply this ONLY to the global search results.
    Charts / categories / 'new' page are left as-is.
    """
    print("\n[steamapi] Filtering via Steam appdetails (type != 'game' removed)...")

    filtered = {}
    dropped = 0

    appids = list(all_games.keys())
    total = len(appids)

    for idx, appid in enumerate(appids, start=1):
        game = all_games.get(appid)
        if not game:
            continue

        info = fetch_appdetails_single(appid)
        success = info.get("success", False)

        if not success:
            # If appdetails fails or returns nothing, keep the game (conservative)
            filtered[appid] = game
        else:
            app_data = info.get("data", {}) or {}
            app_type = (app_data.get("type") or "").lower().strip()

            if not app_type or app_type == "game":
                filtered[appid] = game
            else:
                # Drop only if it's in our explicit non-game list
                if app_type in NON_GAME_TYPES:
                    dropped += 1
                else:
                    # e.g. some weird type; keep it by default
                    filtered[appid] = game

        # progress logging
        if idx % 200 == 0 or idx == total:
            print(
                f"[steamapi] Processed {idx}/{total} appids "
                f"→ kept {len(filtered)}, dropped {dropped}"
            )

        time.sleep(APPDETAILS_SLEEP_SECONDS)

    print(
        f"[steamapi] Done. Original: {total}, kept: {len(filtered)}, dropped: {dropped}"
    )
    return filtered


# ==========================================
# MAIN CRAWLER

def crawl():
    ensure_directories()

    # master dict: all unique games across every source
    master_games = {}

    # -----------------
    # 1) Charts (Top selling / Most played) via browser
    # -----------------
    chart_games = crawl_charts_with_browser()
    upsert_games(master_games, chart_games, label="charts")

    # -----------------
    # 2) Category Top Rated pages via browser
    # -----------------
    category_games = crawl_categories_top_rated()
    upsert_games(master_games, category_games, label="categories")

    # -----------------
    # 3) Cookies for requests-based endpoints
    # -----------------
    cookies = get_steam_cookies()

    # -----------------
    # 4) 'New & Trending' page
    # -----------------
    print(f"\n[crawl] Fetching: {NEW_URL}")
    try:
        new_html = fetch_html(NEW_URL, cookies)
        new_games = parse_generic_page(new_html, NEW_URL)
        print(f"[crawl] Parsed {len(new_games)} new-games")
        upsert_games(master_games, new_games, label="new_page")
    except Exception as e:
        print(f"[crawl] Error fetching NEW_URL: {e}")

    # -----------------
    # 5) Global search JSON paginator (infinite scroll)
    #     -> DLC filter via appdetails applies ONLY here
    # -----------------
    print("\n[crawl] Fetching global search via JSON pagination...")

    search_games_raw = {}  # appid -> game dict
    start = 0
    page_size = 200

    while True:
        # Stop if we already have enough unique candidates overall
        approx_total = len(master_games) + len(search_games_raw)
        if approx_total >= DESIRED_UNIQUE_TOTAL:
            print(
                f"[search] Reached desired total ~{DESIRED_UNIQUE_TOTAL} "
                f"(master+search_raw={approx_total})."
            )
            break

        data = fetch_search_chunk(start, page_size, cookies)
        html = data.get("results_html", "")

        if not html:
            print("[search] No HTML returned, stopping.")
            break

        games = parse_search_results_html(html)
        print(f"[search] Parsed {len(games)} games at start={start}")

        if not games:
            print("[search] No games parsed from chunk, stopping.")
            break

        for g in games:
            appid = g["appid"]
            # store in search-only pool (we will DLC-filter this later)
            if appid not in search_games_raw:
                search_games_raw[appid] = g

        total_count = data.get("total_count", 0)
        start += page_size

        print(
            f"[search] search_raw size={len(search_games_raw)}, "
            f"master={len(master_games)}, approx_total={approx_total}"
        )

        if total_count and start >= total_count:
            print("[search] Reached total_count from API, stopping.")
            break

        time.sleep(1)  # be polite

    # -----------------
    # 6) DLC / non-game filter ONLY on search results
    if search_games_raw:
        filtered_search = filter_non_games_via_steam_api(search_games_raw)
        # Merge filtered search games into master dict
        upsert_games(master_games, filtered_search.values(), label="search_filtered")
    else:
        print("[search] No search games to filter via appdetails.")

    # -----------------
    # 7) Save final deduped dataset
    final_games = list(master_games.values())
    print(f"\n[save] Saving {len(final_games)} unique games to CSV...")

    fieldnames = ["appid", "title", "detail_link"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_games)

    print(f"[save] Saved {len(final_games)} → {OUTPUT_CSV}")


if __name__ == "__main__":
    crawl()
