from pathlib import Path
import os
import sys
import json
import time
import random
import pickle
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------- PATHS / CONSTANTS ----------------

# CF/ -> Ludex/ (project root)
BASE = Path(__file__).resolve().parent.parent

# Load .env from project root (Ludex/.env)
ENV_PATH = BASE / ".env"
load_dotenv(ENV_PATH)

# processed data (Ludex/data/processed) – for model + index only
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# raw cache (Ludex/data/raw/players)
RAW_PLAYERS = BASE / "data" / "raw" / "players"
RAW_PLAYERS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROC / "cf_als_model.pkl"
INDEX_PATH = PROC / "cf_als_index.pkl"

INTERACTIONS_CSV = BASE / "data" / "raw" / "user_game_playtime_top20.csv"
GAMES_CSV        = BASE / "data" / "raw" / "game_details.csv"
NAMES_CSV        = BASE / "data" / "raw" / "user_game_playtime_top20.csv"

# Steam API Key loaded from .env
STEAM_API_KEY = os.getenv("STEAM_API_KEY")

FRIENDS_URL = "https://api.steampowered.com/ISteamUser/GetFriendList/v1/"
OWNED_URL   = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
REQUEST_DELAY = 0.5  # seconds between live API calls (polite)
STORE_URL   = "https://store.steampowered.com/api/appdetails"


# ---------------- SMALL UTILITIES ----------------

def safe_get(url: str, params: dict, max_retries: int = 3):
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


def fetch_friends(steamid: str) -> List[str]:
    """Fetch friend list via Steam API (with simple JSON cache)."""
    if not STEAM_API_KEY:
        print("WARNING: STEAM_API_KEY not set, skipping friend fetch.")
        return []

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
    friends = js.get("friendslist", {}).get("friends", [])
    return [f["steamid"] for f in friends]


def fetch_owned_games(steamid: str):
    """
    Fetch owned games and ALWAYS use cached JSON if available.
    Only fall back to Steam API if no cache exists AND API key is available.
    This ensures we always know what games the user owns, even without API key.
    """
    # 1) Try quick recommender cache
    out_quick = RAW_PLAYERS / f"{steamid}_owned_quick.json"
    if out_quick.exists():
        try:
            return json.loads(out_quick.read_text())
        except Exception:
            pass

    # 2) Try old crawler cache (<steamid>_owned.json)
    out_old = RAW_PLAYERS / f"{steamid}_owned.json"
    if out_old.exists():
        try:
            return json.loads(out_old.read_text())
        except Exception:
            pass

    # 3) Only if NO cache, attempt Steam API fetch
    if not STEAM_API_KEY:
        print("WARNING: STEAM_API_KEY not set and no cached owned-games JSON.")
        return None

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
    out_quick.write_text(json.dumps(js, indent=2))
    return js


def has_public_games(owned_json) -> bool:
    if not owned_json:
        return False
    resp = owned_json.get("response", {})
    games = resp.get("games", [])
    return len(games) > 0


def select_top_games_from_json(owned_json, top_n: int = 20):
    """Return list of {'appid', 'playtime_forever'} from owned_json."""
    if not has_public_games(owned_json):
        return []

    resp = owned_json.get("response", {})
    games = resp.get("games", [])
    if not games:
        return []

    games_sorted = sorted(
        games,
        key=lambda g: int(g.get("playtime_forever", 0)),
        reverse=True,
    )
    selected = games_sorted[:top_n]
    rows = []
    for g in selected:
        playtime = int(g.get("playtime_forever", 0))
        if playtime <= 0:
            continue
        rows.append(
            {
                "steamid": str(resp.get("steamid", "")) or str(g.get("steamid", "")),
                "appid": int(g.get("appid")),
                "playtime_forever": playtime,
            }
        )
    return rows


# ---------------- DATA / MODEL HELPERS ----------------

def build_appid_to_name(appids: List[int]) -> Dict[int, str]:
    """
    Fast mapping from appid -> name/title.

    1) Use local CSVs (game_details.csv, then interactions CSV).
    2) For any remaining 'unknown' appids, query the Steam Store API
       once per app (only for this small set of recommended games).
    """
    appids = [int(a) for a in appids]
    appid_to_name: Dict[int, str] = {}

    # 1) game_details.csv if available
    if GAMES_CSV.exists():
        try:
            # game_details.csv: appid,title,developers,publishers,genres,tags,description
            games = pd.read_csv(GAMES_CSV, usecols=["appid", "title"])
            games["appid"] = games["appid"].astype(int)
            sub = games[games["appid"].isin(appids)]
            for _, row in sub.iterrows():
                appid = int(row["appid"])
                nm = str(row["title"]) if pd.notna(row["title"]) else ""
                if nm:
                    appid_to_name[appid] = nm
        except Exception as e:
            print(f"WARNING: failed to read {GAMES_CSV}: {e}")

    # 2) fallback: names from interactions CSV (if it has game_name)
    missing = [a for a in appids if a not in appid_to_name]
    if missing and NAMES_CSV.exists():
        try:
            df = pd.read_csv(NAMES_CSV)
            if "game_name" in df.columns:
                df["appid"] = df["appid"].astype(int)
                sub = df[df["appid"].isin(missing)]
                for _, row in sub.drop_duplicates("appid").iterrows():
                    appid = int(row["appid"])
                    nm = str(row["game_name"]) if pd.notna(row["game_name"]) else ""
                    if nm:
                        appid_to_name[appid] = nm
        except Exception as e:
            print(f"WARNING: failed to read NAMES_CSV: {e}")

    # 3) final fallback: Steam Store API for remaining ids
    still_missing = [a for a in appids if a not in appid_to_name]
    for appid in still_missing:
        try:
            r = requests.get(
                STORE_URL,
                params={"appids": appid, "cc": "us", "l": "english"},
                timeout=10,
            )
            r.raise_for_status()
            js = r.json()
            entry = js.get(str(appid), {})
            if entry.get("success") and entry.get("data"):
                nm = entry["data"].get("name", "")
                if nm:
                    appid_to_name[appid] = nm
        except Exception:
            continue

    return appid_to_name


def load_interactions() -> pd.DataFrame:
    if not INTERACTIONS_CSV.exists():
        raise SystemExit(f"Missing interactions CSV: {INTERACTIONS_CSV}")
    df = pd.read_csv(INTERACTIONS_CSV)
    if df.empty:
        raise SystemExit("ERROR: interactions CSV is empty.")
    df["steamid"] = df["steamid"].astype(str)
    df["appid"] = df["appid"].astype(int)
    return df


def load_user_owned_and_popularity(df: pd.DataFrame) -> Tuple[Dict[str, Set[int]], Dict[int, float]]:
    """From df, build owned_map + popularity scores."""
    owned_map: Dict[str, Set[int]] = {}
    for sid, group in df.groupby("steamid"):
        owned_map[str(sid)] = set(group["appid"].tolist())

    pop = df.groupby("appid")["steamid"].nunique().astype(float)
    pop_min, pop_max = pop.min(), pop.max()
    if pop_max > pop_min:
        pop_norm = (pop - pop_min) / (pop_max - pop_min)
    elif pop_max > 0:
        pop_norm = pop / pop_max
    else:
        pop_norm = pop
    return owned_map, pop_norm.to_dict()


def ensure_users_in_data_and_retrain(steamids: List[str]) -> None:
    """
    For each steamid not present in INTERACTIONS_CSV:
      - fetch owned games (top 20, public only)
      - append to CSV
    If any rows were added, retrain ALS once.

    This function assumes the base CF model/index already exist.
    """
    if not STEAM_API_KEY:
        print("STEAM_API_KEY not set; cannot auto-add new users.")
        return

    steamids = [str(s) for s in steamids]
    df = load_interactions()
    existing_users = set(df["steamid"].unique())

    new_rows = []
    for sid in steamids:
        if sid in existing_users:
            continue
        print(f"Adding new user {sid} to interactions via API...")
        owned_json = fetch_owned_games(sid)
        rows = select_top_games_from_json(owned_json, top_n=20)
        for r in rows:
            r["steamid"] = sid
        new_rows.extend(rows)

    if not new_rows:
        print("No new playable games found for missing users; no retrain needed.")
        return

    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_new.drop_duplicates(subset=["steamid", "appid"], inplace=True)
    # writes back to data/raw/user_game_playtime_top20.csv
    df_new.to_csv(INTERACTIONS_CSV, index=False)
    print(f"Appended {len(new_rows)} new interaction rows. Retraining model...")

    # use modular trainer: force_retrain=True to overwrite old .pkl files
    from train_cf_als import train_and_save_model
    train_and_save_model(force_retrain=True)
    print("Retrain complete.")


# ---------------- CORE RECOMMENDATION LOGIC ----------------

def recommend_with_model(
    model,
    item_ids: List[int],
    pop_norm: Dict[int, float],
    owned_appids_all: Set[int],
    user_idx: int,
    num_recs: int,
    friends_owned: Set[int],
) -> List[Tuple[int, float]]:

    total_items = len(item_ids)
    raw_N = num_recs + len(owned_appids_all) + 200
    raw_N = min(raw_N, total_items)

    item_indices, scores = model.recommend(
        userid=user_idx,
        user_items=None,
        N=raw_N,
        filter_already_liked_items=False,
    )

    LAMBDA = 0.5
    friend_candidates: List[Tuple[int, float]] = []
    other_candidates: List[Tuple[int, float]] = []

    for item_idx, s in zip(item_indices, scores):
        if item_idx < 0 or item_idx >= len(item_ids):
            continue
        appid = int(item_ids[item_idx])

        if appid in owned_appids_all:
            continue

        base_score = float(s)
        p = pop_norm.get(appid, 0.0)
        adj_score = base_score * (1.0 - LAMBDA * p)

        if appid in friends_owned:
            friend_candidates.append((appid, adj_score))
        else:
            other_candidates.append((appid, adj_score))

    friend_candidates.sort(key=lambda x: x[1], reverse=True)
    other_candidates.sort(key=lambda x: x[1], reverse=True)

    n_from_friends = max(1, int(round(num_recs * 0.75)))
    n_from_friends = min(n_from_friends, num_recs)
    n_from_random = num_recs - n_from_friends

    chosen: List[Tuple[int, float]] = friend_candidates[:n_from_friends]

    if n_from_random > 0 and other_candidates:
        top_k = min(100, len(other_candidates))
        pool = other_candidates[:top_k]
        if len(pool) <= n_from_random:
            chosen.extend(pool)
        else:
            chosen.extend(random.sample(pool, n_from_random))

    if len(chosen) < num_recs:
        remaining = num_recs - len(chosen)
        chosen_ids = {a for a, _ in chosen}
        merged = friend_candidates[n_from_friends:] + other_candidates
        filler = [(a, s) for (a, s) in merged if a not in chosen_ids]
        filler.sort(key=lambda x: x[1], reverse=True)
        chosen.extend(filler[:remaining])

    return chosen[:num_recs]


def cold_start_random_from_users(
    df: pd.DataFrame,
    num_recs: int,
    seed_users_count: int = 60,
) -> List[int]:

    all_users = df["steamid"].unique().tolist()
    if not all_users:
        return []

    k = min(seed_users_count, len(all_users))
    seeds = random.sample(all_users, k)

    sub = df[df["steamid"].isin(seeds)]
    appids = list(set(sub["appid"].tolist()))
    if len(appids) <= num_recs:
        return appids
    return random.sample(appids, num_recs)


# ---------------- MAIN ENTRY ----------------

def main(steamid_str: str, num_recs: int = 10) -> None:
    steamid_str = str(steamid_str)

    # friends from API
    friends = fetch_friends(steamid_str)
    has_friends = len(friends) > 0

    # Ensure interactions CSV exists (crawler must have run)
    if not INTERACTIONS_CSV.exists():
        raise SystemExit(
            f"Interactions file not found: {INTERACTIONS_CSV}\n"
            f"Run the crawler first to create it."
        )

    # --- Initial training if model/index missing ---
    if (not MODEL_PATH.exists()) or (not INDEX_PATH.exists()):
        print("CF model/index not found. Training ALS from scratch via train_cf_als.train_and_save_model() ...")
        from train_cf_als import train_and_save_model
        # force_retrain=False is enough here; files don't exist anyway
        train_and_save_model(force_retrain=False)
        print("Initial CF model trained.\n")

    # Enrich interactions with this user + friends and retrain only if needed
    ensure_users_in_data_and_retrain([steamid_str] + friends)

    # Reload interactions after possible enrichment
    df = load_interactions()
    owned_map, pop_norm = load_user_owned_and_popularity(df)

    # Safety: model/index MUST exist now
    if not MODEL_PATH.exists() or not INDEX_PATH.exists():
        raise SystemExit(
            "Model or index file not found even after training.\n"
            "Check train_cf_als.py and your data setup."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(INDEX_PATH, "rb") as f:
        idx = pickle.load(f)

    user_ids = [str(sid) for sid in idx["user_ids"]]
    item_ids = list(idx["item_ids"])

    steamid_to_idx = {sid: i for i, sid in enumerate(user_ids)}
    in_model = steamid_str in steamid_to_idx

    owned_appids_training = owned_map.get(steamid_str, set())

    owned_appids_api: Set[int] = set()
    owned_json = fetch_owned_games(steamid_str)
    if has_public_games(owned_json):
        games = owned_json.get("response", {}).get("games", []) or []
        for g in games:
            try:
                owned_appids_api.add(int(g.get("appid")))
            except Exception:
                continue

    owned_appids_all = owned_appids_training | owned_appids_api
    has_games = len(owned_appids_all) > 0

    if not has_games and not has_friends:
        print(f"User {steamid_str} has no games and no friends in Steam (fresh account).")
        appids = cold_start_random_from_users(df, num_recs=num_recs, seed_users_count=60)
        appid_to_name = build_appid_to_name(appids)
        if not appids:
            print("Could not build cold-start recommendations (no data).")
            return
        for rank, appid in enumerate(appids, start=1):
            name = appid_to_name.get(appid, f"Unknown app {appid}")
            print(f"{rank:2d}. {name} (appid {appid}) — [cold start random]")
        return

    if not in_model:
        print(f"User {steamid_str} is not in ALS model even after enrichment; "
              f"falling back to cold-start from users.")
        appids = cold_start_random_from_users(df, num_recs=num_recs, seed_users_count=60)
        appid_to_name = build_appid_to_name(appids)
        for rank, appid in enumerate(appids, start=1):
            name = appid_to_name.get(appid, f"Unknown app {appid}")
            print(f"{rank:2d}. {name} (appid {appid}) — [cold start fallback]")
        return

    user_idx = steamid_to_idx[steamid_str]
    print(f"Recommendations for user {steamid_str} (index {user_idx}).")

    if has_games:
        print(f"- User has {len(owned_appids_all)} games (API + training).")
    else:
        print("- User has no games in API/training data (but has friends).")

    if has_friends:
        print(f"- User has {len(friends)} Steam friends (API).")
    else:
        print("- User has no friends (or friend list is private).")

    friends_owned: Set[int] = set()
    for fid in friends:
        games = owned_map.get(str(fid))
        if games:
            friends_owned.update(games)

    if not friends_owned:
        print("No friends with games in training data; falling back to plain CF.")
        total_items = len(item_ids)
        raw_N = num_recs + len(owned_appids_all) + 200
        raw_N = min(raw_N, total_items)

        item_indices, scores = model.recommend(
            userid=user_idx,
            user_items=None,
            N=raw_N,
            filter_already_liked_items=False,
        )

        LAMBDA = 0.5
        candidates: List[Tuple[int, float]] = []
        for item_idx, s in zip(item_indices, scores):
            if item_idx < 0 or item_idx >= len(item_ids):
                continue
            appid = int(item_ids[item_idx])
            if appid in owned_appids_all:
                continue
            base_score = float(s)
            p = pop_norm.get(appid, 0.0)
            adj_score = base_score * (1.0 - LAMBDA * p)
            candidates.append((appid, adj_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:num_recs]
        top_appids = [appid for appid, _ in top]
    else:
        top = recommend_with_model(
            model=model,
            item_ids=item_ids,
            pop_norm=pop_norm,
            owned_appids_all=owned_appids_all,
            user_idx=user_idx,
            num_recs=num_recs,
            friends_owned=friends_owned,
        )
        top_appids = [appid for appid, _ in top]

    appid_to_name = build_appid_to_name(top_appids)
    for rank, (appid, score) in enumerate(top, start=1):
        name = appid_to_name.get(appid, f"Unknown app {appid}")
        print(f"{rank:2d}. {name} (appid {appid}) — score {score:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recommend_for_user.py <steamid> [num_recs]")
        raise SystemExit(1)

    steamid = sys.argv[1]
    num_recs = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
    main(steamid, num_recs)
