# scripts/get_all_apps.py
import requests, json, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "steam_data" / "raw"
DATA.mkdir(parents=True, exist_ok=True)
OUT = DATA / "apps_list.json"

URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"

def fetch():
    print("Fetching app list...")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    OUT.write_text(json.dumps(data))
    print("Saved:", OUT, " Total apps:", len(data.get("applist",{}).get("apps",[])))

if __name__ == "__main__":
    fetch()
