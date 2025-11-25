"""
Ludex Steam Crawler
-------------------
Crawls the Steam friend graph starting from seed SteamIDs, collects
playtime data (top-N or all >0 minutes), and writes:

    data/raw/user_game_playtime_top20.csv

Also caches all Steam API responses under:
    data/raw/players/

Author: Ludex Project
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import deque

import requests
import pandas as pd

# ======================================================
# PATHS

BASE = Path(__file__).resolve().parent.parent

DATA_RAW = BASE / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

RAW_PLAYERS = DATA_RAW / "players"
RAW_PLAYERS.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_RAW / "user_game_playtime_top20.csv"


# ======================================================
# STEAM API

STEAM_API_KEY = os.environ.get("STEAM_API_KEY")
if not STEAM_API_KEY:
    print("Ludex Error: STEAM_API_KEY is not set in environment.")
    sys.exit(1)

FRIENDS_URL = "https://api.steampowered.com/ISteamUser/GetFriendList/v1/"
OWNED_URL   = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"

REQUEST_DELAY = 0.4  # polite delay between calls


# ======================================================
# HELPERS

def safe_get(url: str, params: dict, max_retries: int = 3):
    """GET with simple retries & exponential backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            else:
                print(f"[HTTP {r.status_code}] {url} params={params}")
        except Exception as e:
            print(f"[Error] {url}: {e}")

        time.sleep(REQUEST_DELAY * (attempt + 1))

    return None


def fetch_friends(steamid: str):
    """Return friend's SteamIDs. Uses cache at data/raw/players/<id>_friends.json."""
    out = RAW_PLAYERS / f"{steamid}_friends.json"

    if out.exists():
        try:
            js = json.loads(out.read_text())
            friends = js.get("friendslist", {}).get("friends", [])
            return [f["steamid"] for f in friends]
        except Exception:
            pass

    params = {"key": STEAM_API_KEY, "steamid": steamid, "relationship": "friend"}
    r = safe_get(FRIENDS_URL, params)
    if not r:
        return []

    js = r.json()
    out.write_text(json.dumps(js, indent=2))
    time.sleep(REQUEST_DELAY)

    friends = js.get("friendslist", {}).get("friends", [])
    return [f["steamid"] for f in friends]


def fetch_owned_games(steamid: str):
    """Return owned-games JSON. Uses cache at data/raw/players/<id>_owned.json."""
    out = RAW_PLAYERS / f"{steamid}_owned.json"

    if out.exists():
        try:
            return json.loads(out.read_text())
        except Exception:
            pass

    params = {
        "key": STEAM_API_KEY,
        "steamid": steamid,
        "include_played_free_games": 1,
        "include_appinfo": 0,
    }
    r = safe_get(OWNED_URL, params)
    if not r:
        return None

    js = r.json()
    out.write_text(json.dumps(js, indent=2))
    time.sleep(REQUEST_DELAY)
    return js


def has_public_games(owned_json) -> bool:
    """Return True if user has any public games in API response."""
    if not owned_json:
        return False
    resp = owned_json.get("response", {})
    games = resp.get("games", [])
    return len(games) > 0


def load_seeds(path: Path):
    """Load seeds from file."""
    if not path.exists():
        print(f"Ludex Error: Seed file not found: {path}")
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def select_games(games, top_n: int):
    """Return list of {appid, playtime_forever} for played games."""
    played = [
        g for g in games
        if int(g.get("playtime_forever", 0)) > 0
    ]

    if top_n > 0:
        played = sorted(
            played,
            key=lambda g: int(g.get("playtime_forever", 0)),
            reverse=True,
        )[:top_n]

    return [
        {
            "appid": int(g["appid"]),
            "playtime_forever": int(g["playtime_forever"]),
        }
        for g in played
    ]


# ======================================================
# BFS CRAWLER

def crawl_topN_to_csv(max_users=1000, top_n=0, seeds_file=None):
    """
    Crawl Steam friend graph:
    - Visit users
    - Collect public games with playtime > 0
    - BFS through friends until max_users reached
    """
    seeds = load_seeds(seeds_file) if seeds_file else []
    if not seeds:
        print(f"Ludex Error: No seeds found in {seeds_file}")
        return

    print(f"[Ludex] Seed users: {seeds}")

    visited = set()
    valid_users = set()
    user_game_rows = []

    # Resume mode: pre-load existing data
    for f in RAW_PLAYERS.glob("*_owned.json"):
        sid = f.name.split("_owned.json")[0]
        visited.add(sid)
        try:
            js = json.loads(f.read_text())
            if has_public_games(js):
                games = js.get("response", {}).get("games", [])
                selected = select_games(games, top_n)
                if selected:
                    valid_users.add(sid)
                    for g in selected:
                        user_game_rows.append(
                            {"steamid": sid, "appid": g["appid"], "playtime_forever": g["playtime_forever"]}
                        )
        except Exception:
            pass

    print(f"[Ludex] Resume: visited={len(visited)}, valid={len(valid_users)}")

    queue = deque(seeds)

    # -------------------------
    # BFS graph crawl
    while queue and len(valid_users) < max_users:
        sid = queue.popleft()
        if sid in visited:
            continue

        print(f"\n[Ludex] User {sid} — valid={len(valid_users)}/{max_users}")
        visited.add(sid)

        owned_json = fetch_owned_games(sid)
        if has_public_games(owned_json):
            games = owned_json.get("response", {}).get("games", [])
            print(f"  Public games: {len(games)}")

            selected = select_games(games, top_n)
            if selected:
                valid_users.add(sid)
                for g in selected:
                    user_game_rows.append(
                        {"steamid": sid, "appid": g["appid"], "playtime_forever": g["playtime_forever"]}
                    )
        else:
            print("  No public games.")

        friends = fetch_friends(sid)
        print(f"  Friends found: {len(friends)}")

        for fid in friends:
            if fid not in visited:
                queue.append(fid)

        print(f"  Queue={len(queue)} | Visited={len(visited)} | Valid={len(valid_users)}")

    print("\n[Ludex] Crawl completed.")
    print(f"Visited: {len(visited)} | Valid users: {len(valid_users)}")

    # -------------------------
    # Save results

    df = pd.DataFrame(user_game_rows)
    if not df.empty:
        df.drop_duplicates(subset=["steamid", "appid"], inplace=True)
        num_users = df["steamid"].nunique()
    else:
        num_users = 0

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[Ludex] Saved {len(df)} rows for {num_users} users → {OUTPUT_CSV}")


# ======================================================
# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-users", type=int, default=1000,
                        help="Maximum number of users with public game data.")

    parser.add_argument("--top-n", type=int, default=0,
                        help="Top N games per user by playtime. 0 = all >0 playtime.")

    parser.add_argument("--seeds-file", type=str,
                        default="data/raw/seed_steamids.txt",
                        help="Path to file containing seed SteamIDs (one per line).")

    args = parser.parse_args()

    seeds_path = BASE / args.seeds_file

    crawl_topN_to_csv(
        max_users=args.max_users,
        top_n=args.top_n,
        seeds_file=seeds_path
    )
