import os
import sys
import json
import time
import argparse
from collections import deque
from pathlib import Path

import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PLAYERS = BASE_DIR / "steam_data" / "raw" / "players"
RAW_PLAYERS.mkdir(parents=True, exist_ok=True)
PROCESSED = BASE_DIR / "steam_data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

STEAM_API_KEY = os.environ.get("STEAM_API_KEY")
if not STEAM_API_KEY:
    print("ERROR: STEAM_API_KEY environment variable is not set.")
    sys.exit(1)

FRIENDS_URL = "https://api.steampowered.com/ISteamUser/GetFriendList/v1/"
OWNED_URL   = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"

REQUEST_DELAY = 0.4  # seconds between API calls (tweak if you want)


def safe_get(url, params, max_retries=3):
    """GET with simple retries + polite backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            else:
                print(f"HTTP {r.status_code} for {url} (params={params})")
        except Exception as e:
            print(f"Error on GET {url}: {e}")
        time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


def fetch_friends(steamid):
    """Fetch friends list; cache to <steamid>_friends.json; return list of friend steamids."""
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


def fetch_owned_games(steamid):
    """Fetch owned-games; cache to <steamid>_owned.json; return parsed JSON (or None)."""
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


def has_public_games(owned_json):
    """True if owned-games JSON has a non-empty games list."""
    if not owned_json:
        return False
    resp = owned_json.get("response", {})
    games = resp.get("games", [])
    return len(games) > 0


def load_seeds(seeds_file: Path):
    if not seeds_file.exists():
        print("Seed file not found:", seeds_file)
        return []
    return [l.strip() for l in seeds_file.read_text().splitlines() if l.strip()]


def select_games(games, top_n):
    """
    From the full games list, return a list of dicts with appid + playtime.
    - Always drop games with 0 playtime (never played).
    - If top_n > 0, keep only the top_n by playtime.
    - If top_n <= 0, keep ALL played games.
    """
    # drop never-played games early
    played = [
        g for g in games
        if int(g.get("playtime_forever", 0)) > 0
    ]

    if top_n and top_n > 0:
        played = sorted(
            played,
            key=lambda g: int(g.get("playtime_forever", 0)),
            reverse=True,
        )[:top_n]

    return [
        {
            "appid": int(g.get("appid")),
            "playtime_forever": int(g.get("playtime_forever", 0)),
        }
        for g in played
    ]


def crawl_topN_to_csv(max_users=1000, top_n=0, seeds_file=None):
    """
    Crawl the friend graph starting from seed SteamIDs.

    - For each user with public games:
        * collect all games with playtime > 0 (or top_n most-played if top_n > 0)
        * store (steamid, appid, playtime_forever)
    - Continue BFS through their friends until we have max_users users with data.
    """
    seeds = load_seeds(seeds_file) if seeds_file else []
    if not seeds:
        print("No seeds in", seeds_file, "- add some steamids first.")
        return

    print("Seed users:", seeds)

    visited = set()
    valid_users = set()
    user_game_rows = []

    # resume support: pre-load any existing data if you re-run
    for f in RAW_PLAYERS.glob("*_owned.json"):
        sid = f.name.split("_owned.json")[0]
        visited.add(sid)
        try:
            js = json.loads(f.read_text())
            if has_public_games(js):
                resp = js.get("response", {})
                games = resp.get("games", [])
                selected = select_games(games, top_n)
                if selected:
                    valid_users.add(sid)
                    for g in selected:
                        user_game_rows.append(
                            {
                                "steamid": sid,
                                "appid": g["appid"],
                                "playtime_forever": g["playtime_forever"],
                            }
                        )
        except Exception:
            pass

    print(
        f"Starting crawl. Already visited: {len(visited)}, "
        f"valid with games: {len(valid_users)}"
    )

    queue = deque(seeds)

    while queue and len(valid_users) < max_users:
        sid = queue.popleft()
        if sid in visited:
            continue

        print(f"\nProcessing user {sid} (valid so far: {len(valid_users)}/{max_users})")
        visited.add(sid)

        owned = fetch_owned_games(sid)
        if has_public_games(owned):
            resp = owned.get("response", {})
            games = resp.get("games", [])
            print("  Public games:", len(games))

            selected = select_games(games, top_n)
            if selected:
                valid_users.add(sid)
                for g in selected:
                    user_game_rows.append(
                        {
                            "steamid": sid,
                            "appid": g["appid"],
                            "playtime_forever": g["playtime_forever"],
                        }
                    )
            else:
                print("  All games have 0 playtime, skipping rows.")
        else:
            print("  No public games (private or empty).")

        friends = fetch_friends(sid)
        print("  Friends discovered:", len(friends))
        for fid in friends:
            if fid not in visited:
                queue.append(fid)

        print(
            f"  Queue size: {len(queue)} | "
            f"Visited: {len(visited)} | Valid users: {len(valid_users)}"
        )

    print("\nCrawl finished.")
    print("Total visited:", len(visited))
    print("Total users with public games and >0 playtime:", len(valid_users))

    # create CSV
    df = pd.DataFrame(user_game_rows)
    if not df.empty:
        df.drop_duplicates(subset=["steamid", "appid"], inplace=True)
        n_users = df["steamid"].nunique()
    else:
        n_users = 0

    out_csv = PROCESSED / "user_game_playtime_top20.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows for {n_users} users to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-users",
        type=int,
        default=1000,
        help="Number of users with public game data to target.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help=(
            "Max number of games per user (most-played). "
            "Use 0 or negative to keep ALL games with playtime > 0."
        ),
    )
    parser.add_argument(
        "--seeds-file",
        type=str,
        default="steam_data/seed_steamids.txt",
        help="File with initial seed steamids (one per line).",
    )
    args = parser.parse_args()

    seeds_path = BASE_DIR / args.seeds_file
    crawl_topN_to_csv(max_users=args.max_users, top_n=args.top_n, seeds_file=seeds_path)
