# recommender/catalogue_update.py

import time
import random
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
import scipy.sparse as sp
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

from scipy.sparse import save_npz, load_npz

from .cbf_model import (
    INPUT_CSV,            # data/raw/game_details.csv
    FEATURE_MATRIX_NPZ,   # data/processed/recommender_matrix.npz
    build_feature_matrix,
)

# ======================================================
# CONFIG (mirrors crawler)
# ======================================================

DATA_DIR = INPUT_CSV.parent  # data/raw

REQUEST_DELAY = 0.8  # polite delay per request
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ======================================================
# COOKIE + SESSION HELPERS (adapted from crawler)
# ======================================================

def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[catalogue_update] Ensured folder: {DATA_DIR}")


def get_steam_cookies():
    """
    Use undetected Chrome once to bypass Cloudflare & age checks.

    This is the same trick used in the main crawler, but here we'll
    typically fetch only a handful of missing games for a specific user.
    """
    print("[catalogue_update] Launching browser to bypass Cloudflare…")

    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")

    browser = uc.Chrome(options=options)
    cookies = {}
    try:
        # Any popular app page works; 730 = CS:GO/CS2
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

    print(f"[catalogue_update] Cookies obtained: {len(cookies)} entries.")
    return cookies


def make_session(cookies_dict):
    """Create a requests.Session with Steam cookies + headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Referer": "https://store.steampowered.com/",
    })
    s.cookies.update(cookies_dict)
    return s


def clean_url(url: str) -> str:
    """Strip query params and normalise app URL."""
    m = re.match(r"(https://store\.steampowered\.com/app/\d+)", url)
    if m:
        return m.group(1)
    return url.split("?")[0]


def fetch_html(session: requests.Session, url: str) -> str:
    """Fetch a game page in English with basic error hints."""
    full = url + "?l=english&cc=US"
    r = session.get(full, timeout=25, allow_redirects=True)

    if "agecheck" in r.url:
        print("  [warn] Agecheck redirect during catalogue update")

    if "problem fulfilling your request" in r.text:
        print("  [warn] Steam error page during catalogue update")

    r.raise_for_status()
    return r.text


def fetch_with_retry(session, url, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            return fetch_html(session, url)
        except Exception as e:
            print(f"   [retry {attempt+1}/{retries}] {url} — {e}")
            time.sleep(1 + random.random() * 2)
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries.")


# ======================================================
# PARSING (copied from crawler)
# ======================================================

def parse_detail_html(appid: int, title: str, html: str) -> dict:
    """
    Parse Steam game detail HTML into the same schema as game_details.csv.

    Returns a dict with keys:
        appid, title, developers, publishers, genres, tags, description
    """
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


# ======================================================
# CATALOGUE EXTENSION LOGIC
# ======================================================

def extend_catalogue_with_missing_games(
    owned_df: pd.DataFrame,
    catalogue_df: pd.DataFrame,
    max_new: int = 50,
) -> pd.DataFrame:
    """
    Ensure that all (or at most `max_new`) games in owned_df (by appid) exist in catalogue_df.

    Logic:
      - Find games the user owns that are NOT in the catalogue.
      - Sort those missing games by playtime_min descending.
      - Take the top `max_new` of them (safety cap).
      - Crawl & parse details for just those appids.
      - Append new rows to the catalogue_df.

    Returns:
        updated_df: updated catalogue DataFrame (in-memory only).
    """
    ensure_directories()

    # Existing appids in catalogue
    catalog_appids = set(int(a) for a in catalogue_df["appid"])

    # Owned games that are NOT in the catalogue yet
    missing_owned = owned_df[~owned_df["appid"].isin(catalog_appids)].copy()

    if missing_owned.empty:
        print("[catalogue_update] No missing games – catalogue already covers user library.")
        return catalogue_df

    # Sort missing games by playtime (most played first)
    if "playtime_min" in missing_owned.columns:
        missing_owned = missing_owned.sort_values("playtime_min", ascending=False)
    else:
        # Fallback: no playtime column? Just keep as-is.
        print("[catalogue_update] Warning: 'playtime_min' not found; cannot rank by playtime.")
    
    # Apply safety cap: only crawl top `max_new` missing games
    if len(missing_owned) > max_new:
        print(
            f"[catalogue_update] Found {len(missing_owned)} missing games; "
            f"capping crawl to top {max_new} by playtime."
        )
        missing_owned = missing_owned.head(max_new)
    else:
        print(f"[catalogue_update] Found {len(missing_owned)} missing games to crawl.")

    missing_appids = [int(a) for a in missing_owned["appid"].tolist()]

    # Map appid -> title from owned_df so we have a reasonable title fallback.
    title_map = {
        int(row["appid"]): str(row.get("title", "")).strip()
        for _, row in owned_df.iterrows()
    }

    cookies = get_steam_cookies()
    session = make_session(cookies)

    new_rows = []

    for idx, appid in enumerate(missing_appids, start=1):
        title_guess = title_map.get(appid, f"appid_{appid}")
        url = clean_url(f"https://store.steampowered.com/app/{appid}")

        print(
            f"[catalogue_update] ({idx}/{len(missing_appids)}) "
            f"Fetching metadata for missing appid={appid} title='{title_guess[:40]}'…"
        )

        try:
            time.sleep(REQUEST_DELAY + random.random() * 0.3)
            html = fetch_with_retry(session, url)
            row = parse_detail_html(appid, title_guess, html)
            new_rows.append(row)
        except Exception as e:
            print(f"[catalogue_update] ❌ Error fetching appid={appid} — {e}")

    if not new_rows:
        print("[catalogue_update] No new rows could be fetched; catalogue unchanged.")
        return catalogue_df

    new_df = pd.DataFrame(new_rows)

    # Align columns with existing catalogue where possible.
    merged_df = pd.concat(
        [catalogue_df, new_df.reindex(columns=catalogue_df.columns, fill_value=pd.NA)],
        ignore_index=True,
    )

    print(f"[catalogue_update] Catalogue extended with {len(new_df)} new games.")
    return merged_df


def rebuild_feature_matrix_and_cache(
    updated_df: pd.DataFrame,
) -> sp.csr_matrix:
    """
    Rebuild the full feature matrix from scratch and cache it to FEATURE_MATRIX_NPZ.

    This is the simplest, most robust approach for now.
    """
    print("[catalogue_update] Rebuilding feature matrix from updated catalogue…")
    full_matrix_norm = build_feature_matrix(updated_df)
    save_npz(FEATURE_MATRIX_NPZ, full_matrix_norm)
    print(f"[catalogue_update] Saved updated feature matrix to: {FEATURE_MATRIX_NPZ}")

    # Also overwrite the CSV to keep everything consistent.
    updated_df.to_csv(INPUT_CSV, index=False)
    print(f"[catalogue_update] Saved updated catalogue CSV to: {INPUT_CSV}")

    return full_matrix_norm


def ensure_user_games_in_catalogue_and_refresh(
    owned_df: pd.DataFrame,
    catalogue_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, sp.csr_matrix]:
    """
    High-level helper used by main.py:

      1. Extend the catalogue with any games from owned_df that are missing.
      2. If anything was added, rebuild the feature matrix and cache.
      3. If nothing was added, simply load the existing matrix from FEATURE_MATRIX_NPZ.

    Returns:
        new_df, new_full_matrix_norm
    """
    original_count = len(catalogue_df)
    updated_df = extend_catalogue_with_missing_games(owned_df, catalogue_df)

    if len(updated_df) == original_count:
        # No changes → just load the existing matrix (assumed up-to-date).
        print("[catalogue_update] No new games added; using existing feature matrix.")
        full_matrix_norm = load_npz(FEATURE_MATRIX_NPZ)
        return updated_df, full_matrix_norm

    # Rebuild matrix because catalogue changed
    full_matrix_norm = rebuild_feature_matrix_and_cache(updated_df)
    return updated_df, full_matrix_norm
