"""
Ludex CF Module
----------------
Trains an ALS (implicit feedback) collaborative filtering model using
Steam user playtime data stored in:
    data/raw/user_game_playtime_top20.csv

This script:
1. Loads & filters raw interactions
2. Builds user–item implicit matrix
3. Trains ALS
4. Saves:
       - cf_als_model.pkl
       - cf_als_index.pkl

Author: Ludex Project
"""

from pathlib import Path
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import implicit


# ======================================================
# PATHS

BASE = Path(__file__).resolve().parent.parent

# Models are always stored in processed/
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

INTERACTIONS_CSV = BASE / "data" / "raw" / "user_game_playtime_top20.csv"


# ======================================================
# CONFIG

MIN_PLAYTIME = 10          # drop interactions with tiny playtime
MIN_USER_SUPPORT = 2       # game must be played by ≥2 users
ALS_FACTORS = 64
ALS_REG = 0.15
ALS_ITERS = 25
RANDOM_STATE = 42


# ======================================================
# LOAD + FILTER

def load_and_filter() -> Tuple[pd.DataFrame, pd.Index, pd.Index]:
    """
    Load the raw interactions CSV and apply all filters.
    Then factorize user and item into contiguous integer indices.
    """
    print(f"\n[Ludex] Loading interactions from: {INTERACTIONS_CSV}")

    try:
        df = pd.read_csv(
            INTERACTIONS_CSV,
            usecols=["steamid", "appid", "playtime_forever"],
        )
    except FileNotFoundError:
        raise SystemExit(
            f"Ludex Error: Interactions CSV not found:\n{INTERACTIONS_CSV}"
        )

    if df.empty:
        raise SystemExit("Ludex Error: interactions CSV is empty.")

    # Combine duplicate (user, item)
    df = (
        df.groupby(["steamid", "appid"], as_index=False)["playtime_forever"]
          .sum()
    )

    # Filter: playtime threshold
    df = df[df["playtime_forever"] >= MIN_PLAYTIME]
    print(f"[Ludex] After MIN_PLAYTIME={MIN_PLAYTIME}: {len(df)} rows")

    if df.empty:
        raise SystemExit(
            f"Ludex Error: No interactions remain after MIN_PLAYTIME.\n"
            "Check crawler output or reduce filter thresholds."
        )

    # Filter: item must have enough users
    item_counts = df.groupby("appid")["steamid"].nunique()
    valid_items = item_counts[item_counts >= MIN_USER_SUPPORT].index
    df = df[df["appid"].isin(valid_items)]
    print(f"[Ludex] After MIN_USER_SUPPORT={MIN_USER_SUPPORT}: {len(df)} rows")

    if df.empty:
        raise SystemExit(
            f"Ludex Error: All items removed after MIN_USER_SUPPORT filter.\n"
            "Reduce MIN_USER_SUPPORT or collect more crawl data."
        )

    df = df.reset_index(drop=True)

    # Factorize IDs
    df["user_idx"], user_ids = pd.factorize(df["steamid"])
    df["item_idx"], item_ids = pd.factorize(df["appid"])

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    print(f"[Ludex] Users={n_users}, Items={n_items}, Interactions={len(df)}")

    if n_users < 2 or n_items < 2:
        raise SystemExit(
            f"Ludex Error: Not enough users/items for ALS "
            f"(users={n_users}, items={n_items})."
        )

    return df, user_ids, item_ids


# ======================================================
# MATRIX BUILDING

def build_user_item_matrix(df: pd.DataFrame) -> coo_matrix:
    """
    Construct a user–item implicit matrix using normalized playtime.
    """
    # Normalize per-user playtime
    df["norm_playtime"] = df.groupby("user_idx")["playtime_forever"].transform(
        lambda x: x / x.max()
    )

    # Convert to implicit confidence
    confidence = np.log1p(df["norm_playtime"] * 40).astype(np.float32)

    rows = df["user_idx"].astype(np.int32).values
    cols = df["item_idx"].astype(np.int32).values

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    matrix = coo_matrix(
        (confidence, (rows, cols)),
        shape=(n_users, n_items),
    ).tocsr()

    return matrix


# ======================================================
# ALS TRAINING

def train_als(user_items: coo_matrix) -> implicit.als.AlternatingLeastSquares:
    print("\n[Ludex] Training implicit ALS model…")

    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REG,
        iterations=ALS_ITERS,
        random_state=RANDOM_STATE,
    )

    model.fit(user_items)
    return model


# ======================================================
# MAIN

def main() -> None:
    df, user_ids, item_ids = load_and_filter()
    user_items = build_user_item_matrix(df)

    model = train_als(user_items)

    # Save model + index
    with open(PROC / "cf_als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(PROC / "cf_als_index.pkl", "wb") as f:
        pickle.dump(
            {"user_ids": list(user_ids),
             "item_ids": list(item_ids)},
            f,
        )

    print("\n[Ludex] ✓ Model trained and saved successfully.\n")


if __name__ == "__main__":
    main()
