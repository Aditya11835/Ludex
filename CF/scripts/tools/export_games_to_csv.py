from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW_GAMES = BASE / "steam_data" / "raw" / "games"
PROC = BASE / "steam_data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

OUT_CSV = PROC / "games_all.csv"

def extract_row(appid: int, js: dict):
    """Extract a compact row from a single storefront response."""
    entry = js.get(str(appid), {})
    if not isinstance(entry, dict):
        return None
    if not entry.get("success"):
        return None
    data = entry.get("data")
    if not isinstance(data, dict):
        return None

    # basic fields
    name = data.get("name", "")
    app_type = data.get("type", "")
    is_free = bool(data.get("is_free", False))
    required_age = data.get("required_age", "")
    release = data.get("release_date", {}) or {}
    release_date = release.get("date", "")
    # price
    price = data.get("price_overview") or {}
    price_initial = price.get("initial")
    price_final = price.get("final")
    currency = price.get("currency")

    # list-ish fields
    def join_list(lst, key="description"):
        if not isinstance(lst, list):
            return ""
        parts = []
        for item in lst:
            if isinstance(item, dict):
                val = item.get(key)
            else:
                val = str(item)
            if val:
                parts.append(str(val))
        return "|".join(parts)

    genres = join_list(data.get("genres", []))
    categories = join_list(data.get("categories", []))
    developers = "|".join(data.get("developers", [])) if data.get("developers") else ""
    publishers = "|".join(data.get("publishers", [])) if data.get("publishers") else ""

    return {
        "appid": appid,
        "name": name,
        "type": app_type,
        "is_free": is_free,
        "required_age": required_age,
        "release_date": release_date,
        "developers": developers,
        "publishers": publishers,
        "genres": genres,
        "categories": categories,
        "price_initial": price_initial,
        "price_final": price_final,
        "currency": currency,
    }


def main():
    if not RAW_GAMES.exists():
        raise SystemExit(f"RAW games folder not found: {RAW_GAMES}")

    rows = []
    files = sorted(RAW_GAMES.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {RAW_GAMES}. Run the fetch scripts first.")

    print(f"Reading {len(files)} storefront JSON files from {RAW_GAMES}...")

    for i, f in enumerate(files, start=1):
        try:
            appid = int(f.stem)
        except ValueError:
            # file not named as appid.json
            continue
        try:
            js = json.loads(f.read_bytes().decode("utf-8", errors="ignore"))
        except Exception as e:
            print(f"[{i}/{len(files)}] Error reading {f.name}: {e}")
            continue

        row = extract_row(appid, js)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit("No valid rows extracted from storefront JSONs.")

    df = pd.DataFrame(rows)
    df.sort_values("appid", inplace=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} games to {OUT_CSV}")


if __name__ == "__main__":
    main()