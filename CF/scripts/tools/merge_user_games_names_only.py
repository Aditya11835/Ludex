from pathlib import Path
import pandas as pd
import json

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "steam_data" / "processed"
RAW_GAMES = BASE / "steam_data" / "raw" / "games"

USER_CSV = PROC / "user_game_playtime_top20.csv"
GAMES_ALL_CSV = PROC / "games_all.csv"
GAMES_SMALL_CSV = PROC / "games.csv"

if not USER_CSV.exists():
    raise SystemExit(f"Missing {USER_CSV}. Run the top20 crawler first.")

# Load user-game interactions
ug = pd.read_csv(USER_CSV)

# Step 1: build mapping appid -> name from games_all or games.csv
name_map = {}

if GAMES_ALL_CSV.exists():
    games = pd.read_csv(GAMES_ALL_CSV, usecols=["appid", "name"])
    print("Using games_all.csv for names")
elif GAMES_SMALL_CSV.exists():
    games = pd.read_csv(GAMES_SMALL_CSV, usecols=["appid", "name"])
    print("Using games.csv for names")
else:
    print("No game metadata CSV found, will try raw JSON only.")
    games = pd.DataFrame(columns=["appid", "name"])

if not games.empty:
    games["appid"] = games["appid"].astype(int)
    for _, row in games.iterrows():
        appid = int(row["appid"])
        nm = str(row["name"]) if pd.notna(row["name"]) else ""
        if nm:
            name_map[appid] = nm

# Step 2: fill missing names from raw storefront JSON if possible
missing_appids = set(ug["appid"].astype(int)) - set(name_map.keys())
if missing_appids:
    print("Trying to fill names from raw storefront JSON for", len(missing_appids), "appids...")
    for appid in missing_appids:
        f = RAW_GAMES / f"{appid}.json"
        if not f.exists():
            continue
        try:
            js = json.loads(f.read_bytes().decode("utf-8", errors="ignore"))
            entry = js.get(str(appid), {})
            if entry.get("success") and entry.get("data"):
                nm = entry["data"].get("name", "")
                if nm:
                    name_map[appid] = nm
        except Exception:
            continue

# Step 3: build final DataFrame with clean column 'game_name'
ug["appid"] = ug["appid"].astype(int)

def lookup_name(a):
    return name_map.get(int(a), f"UNKNOWN_APP_{int(a)}")

ug["game_name"] = ug["appid"].apply(lookup_name)

# Keep only the columns we care about
clean = ug[["steamid", "appid", "playtime_forever", "game_name"]]

out_path = PROC / "user_game_playtime_top20_with_names_clean.csv"
clean.to_csv(out_path, index=False, encoding="utf-8")
print(f"Wrote {len(clean)} rows for {clean['steamid'].nunique()} users to {out_path}")