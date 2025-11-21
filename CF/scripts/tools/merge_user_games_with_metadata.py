import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "steam_data" / "processed"

USER_CSV = PROC / "user_game_playtime_top20.csv"
GAMES_CSV = PROC / "games_all.csv"   # or games.csv if you prefer smaller metadata file

if not USER_CSV.exists():
    raise SystemExit(f"Missing {USER_CSV}. Run the crawler first.")

if not GAMES_CSV.exists():
    raise SystemExit(f"Missing {GAMES_CSV}. Run export_games_to_csv.py first.")

# Load data
users = pd.read_csv(USER_CSV)
games = pd.read_csv(GAMES_CSV)

# Ensure types
users["appid"] = users["appid"].astype(int)
games["appid"] = games["appid"].astype(int)

# Merge
merged = users.merge(games, on="appid", how="left")

# Output
OUT = PROC / "user_game_playtime_top20_with_names.csv"
merged.to_csv(OUT, index=False)

print(f"Wrote merged CSV with {len(merged)} rows to {OUT}")
print(f"Unique users: {merged['steamid'].nunique()}")
print(f"Unique games: {merged['appid'].nunique()}")