# scripts/fetch_storefront.py
import requests, time, json, argparse, sys
from pathlib import Path
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "steam_data" / "raw"
GAMES_DIR = DATA / "games"
GAMES_DIR.mkdir(parents=True, exist_ok=True)
APPLIST = DATA / "apps_list.json"
STORE_URL = "https://store.steampowered.com/api/appdetails"

HEADERS = {"User-Agent": "steam-recommender-bot/1.0 (+you)"}
REQUEST_DELAY = 0.25  # polite: 250ms between requests

def load_appids():
    if not APPLIST.exists():
        print("apps_list.json not found. Run get_all_apps.py first.")
        sys.exit(1)
    data = json.loads(APPLIST.read_text())
    apps = data.get("applist", {}).get("apps", [])
    return [str(a["appid"]) for a in apps]

def fetch_one(appid):
    out = GAMES_DIR / f"{appid}.json"
    if out.exists():
        return
    try:
        r = requests.get(STORE_URL, params={"appids": appid, "l": "english"}, headers=HEADERS, timeout=30)
        r.raise_for_status()
        out.write_text(r.text)
    except Exception as e:
        # write a small error wrapper so we can retry later if needed
        out.write_text(json.dumps({"success": False, "error": str(e)}))
    time.sleep(REQUEST_DELAY)

def main(limit=None, offset=0):
    ids = load_appids()
    if offset:
        ids = ids[offset:]
    if limit:
        ids = ids[:limit]
    print(f"Fetching {len(ids)} app storefront entries (delay {REQUEST_DELAY}s)...")
    for aid in tqdm(ids):
        fetch_one(aid)
    print("Done. Cached in:", GAMES_DIR)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="max number of apps to fetch (for testing)")
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()
    main(limit=args.limit, offset=args.offset)
