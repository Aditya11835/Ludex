import requests, time, json, argparse, sys, traceback
from pathlib import Path
from tqdm import tqdm
import csv
import math

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "steam_data" / "raw"
GAMES_DIR = RAW / "games"
PROCESSED = BASE / "steam_data" / "processed"
GAMES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

STORE_URL = "https://store.steampowered.com/api/appdetails"
STEAM_APPLIST = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
STEAMSPY_ALL = "https://steamspy.com/api.php?request=all"
HEADERS = {"User-Agent": "steam-recommender-bot/1.0 (+you)"}

DEFAULT_DELAY = 0.25  # seconds between storefront requests
RETRY_DELAY = 5

def fetch_official_applist():
    try:
        r = requests.get(STEAM_APPLIST, timeout=20)
        if r.status_code == 200:
            data = r.json()
            apps = data.get("applist", {}).get("apps", [])
            appids = [str(a["appid"]) for a in apps if "appid" in a]
            print("Official app list fetched. Total apps:", len(appids))
            return appids
        else:
            print("Official applist returned status", r.status_code)
            return None
    except Exception as e:
        print("Error fetching official applist:", e)
        return None

def fetch_steamspy_all():
    try:
        print("Fetching SteamSpy 'all' fallback (this returns a big JSON)...")
        r = requests.get(STEAMSPY_ALL, timeout=60)
        r.raise_for_status()
        data = r.json()
        appids = list({str(int(k)): True for k in data.keys()}.keys())  # keys are appids (strings)
        print("SteamSpy all returned entries:", len(appids))
        return appids
    except Exception as e:
        print("SteamSpy fallback failed:", e)
        return None

def load_appids(limit=None, offset=0):
    print("Attempting official Steam applist...")
    appids = fetch_official_applist()
    if not appids:
        print("Official applist failed — trying SteamSpy fallback...")
        appids = fetch_steamspy_all()
    if not appids:
        print("Both sources failed. Exiting.")
        sys.exit(1)
    if offset:
        appids = appids[offset:]
    if limit:
        appids = appids[:limit]
    return appids

def fetch_storefront_for_appid(appid, delay=DEFAULT_DELAY):
    out = GAMES_DIR / f"{appid}.json"
    if out.exists():
        return True
    try:
        r = requests.get(STORE_URL, params={"appids": appid, "l": "english"}, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            out.write_text(json.dumps({"success": False, "http_status": r.status_code}))
            return False
        out.write_bytes(r.content)
        # small polite delay
        time.sleep(delay)
        return True
    except Exception as e:
        print("Error fetching app", appid, "->", e)
        # write an error file so we can retry selectively later
        try:
            out.write_text(json.dumps({"success": False, "error": str(e)}), encoding="utf-8")
        except Exception:
            pass
        time.sleep(RETRY_DELAY)
        return False

def parse_storefront_json(appid):
    f = GAMES_DIR / f"{appid}.json"
    if not f.exists():
        return None
    try:
        js = json.loads(f.read_text())
        payload = js.get(str(appid), {})
        if not payload.get("success"):
            return None
        d = payload.get("data", {})
        # basic fields to extract
        name = d.get("name","")
        type_ = d.get("type","")
        short_desc = d.get("short_description","")
        detailed = d.get("detailed_description","")
        developers = "|".join(d.get("developers",[])) if d.get("developers") else ""
        publishers = "|".join(d.get("publishers",[])) if d.get("publishers") else ""
        genres = "|".join([g.get("description","") for g in d.get("genres",[])]) if d.get("genres") else ""
        categories = "|".join([c.get("description","") for c in d.get("categories",[])]) if d.get("categories") else ""
        release_date = d.get("release_date",{}).get("date","")
        metacritic = d.get("metacritic",{}).get("score")
        price_overview = d.get("price_overview",{})
        initial_price = price_overview.get("initial")
        final_price = price_overview.get("final")
        return {
            "appid": int(appid),
            "name": name,
            "type": type_,
            "short_description": short_desc,
            "detailed_description": detailed,
            "developers": developers,
            "publishers": publishers,
            "genres": genres,
            "categories": categories,
            "release_date": release_date,
            "metacritic_score": metacritic,
            "initial_price": initial_price,
            "final_price": final_price,
        }
    except Exception:
        # corrupted JSON or parse error
        # print(traceback.format_exc())
        return None

def write_csv(rows, outpath):
    if not rows:
        print("No rows to write.")
        return
    keys = ["appid","name","type","short_description","detailed_description","developers","publishers","genres","categories","release_date","metacritic_score","initial_price","final_price"]
    with open(outpath, "w", encoding="utf-8", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            # ensure presence of keys
            row = {k: r.get(k,"") for k in keys}
            writer.writerow(row)
    print("Wrote CSV:", outpath, "rows:", len(rows))

def main(limit=None, offset=0, delay=DEFAULT_DELAY, only_cached=False):
    appids = load_appids(limit=limit, offset=offset)
    print("Total appids to process:", len(appids))

    # fetch storefront entries (skip already-cached)
    if not only_cached:
        print("Fetching storefront metadata (skipping already cached files)...")
        for aid in tqdm(appids):
            fetch_storefront_for_appid(aid, delay=delay)

    # parse cached JSONs into rows
    print("Parsing cached storefront JSONs into rows...")
    rows = []
    missing = 0
    for aid in tqdm(appids):
        parsed = parse_storefront_json(aid)
        if parsed:
            rows.append(parsed)
        else:
            missing += 1
    print("Parsed rows:", len(rows), "missing/unparsed:", missing)

    outcsv = PROCESSED / "games_all.csv"
    write_csv(rows, outcsv)

    # companion info
    sz = outcsv.stat().st_size if outcsv.exists() else 0
    print("Output CSV size (bytes):", sz)
    mb = math.floor(sz / (1024*1024))
    print("Approx size (MB):", mb)
    print("Done. You can open:", outcsv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="Limit number of appids to fetch (for testing)")
    p.add_argument("--offset", type=int, default=0, help="Offset into app list")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between storefront requests (seconds)")
    p.add_argument("--only-cached", action="store_true", help="Do not fetch; only parse already-cached JSONs")
    args = p.parse_args()
    main(limit=args.limit, offset=args.offset, delay=args.delay, only_cached=args.only_cached)