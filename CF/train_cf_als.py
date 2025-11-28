# CF/train_cf_als.py

from pathlib import Path
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import implicit  # ✅ required for ALS


BASE = Path(__file__).resolve().parent.parent

# processed dir (Ludex/data/processed)
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

INTERACTIONS_CSV = PROC / "user_game_playtime_top20.csv"

# ---------------- CONFIG ----------------
MIN_PLAYTIME = 10         # minimum minutes of playtime to consider a (user, game) interaction
MIN_USER_SUPPORT = 2      # game must be played by at least this many users
ALS_FACTORS = 64
ALS_REG = 0.15
ALS_ITERS = 25
RANDOM_STATE = 42


def load_and_filter() -> Tuple[pd.DataFrame, pd.Index, pd.Index]:
    """
    Load interactions and apply all filters, then factorize user & item IDs.
    """
    print("Loading:", INTERACTIONS_CSV)
    try:
        df = pd.read_csv(
            INTERACTIONS_CSV,
            usecols=["steamid", "appid", "playtime_forever"],
        )
    except FileNotFoundError:
        raise SystemExit(f"ERROR: interactions CSV not found: {INTERACTIONS_CSV}")

    if df.empty:
        raise SystemExit("ERROR: interactions CSV is empty. Did the crawler run?")

    df = (
        df.groupby(["steamid", "appid"], as_index=False)["playtime_forever"]
        .sum()
    )

    df = df[df["playtime_forever"] >= MIN_PLAYTIME]
    print("After MIN_PLAYTIME filter:", len(df))
    if df.empty:
        raise SystemExit(
            f"ERROR: no interactions left after MIN_PLAYTIME={MIN_PLAYTIME}. "
            "Consider lowering MIN_PLAYTIME or checking the crawl."
        )

    item_user_counts = df.groupby("appid")["steamid"].nunique()
    valid_items = item_user_counts[item_user_counts >= MIN_USER_SUPPORT].index
    df = df[df["appid"].isin(valid_items)]
    print("After MIN_USER_SUPPORT filter:", len(df))
    if df.empty:
        raise SystemExit(
            f"ERROR: no interactions left after MIN_USER_SUPPORT={MIN_USER_SUPPORT}. "
            "Dataset is too sparse; consider lowering MIN_USER_SUPPORT."
        )

    df = df.reset_index(drop=True)

    df["user_idx"], user_ids = pd.factorize(df["steamid"])
    df["item_idx"], item_ids = pd.factorize(df["appid"])

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")
    print("Max user_idx =", df["user_idx"].max())
    print("Max item_idx =", df["item_idx"].max())

    if n_users < 2 or n_items < 2:
        raise SystemExit(
            f"ERROR: Not enough users/items to train ALS "
            f"(users={n_users}, items={n_items})."
        )

    assert df["user_idx"].max() == n_users - 1
    assert df["item_idx"].max() == n_items - 1

    return df, user_ids, item_ids


def build_user_item_matrix(df: pd.DataFrame) -> coo_matrix:
    """
    Build user x item implicit-feedback matrix for ALS.
    """
    df["norm_playtime"] = df.groupby("user_idx")["playtime_forever"].transform(
        lambda x: x / x.max()
    )

    confidence = np.log1p(df["norm_playtime"] * 40).astype(np.float32)

    rows = df["user_idx"].astype(np.int32).values
    cols = df["item_idx"].astype(np.int32).values

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    user_items = coo_matrix(
        (confidence, (rows, cols)),
        shape=(n_users, n_items),
    ).tocsr()

    assert user_items.shape == (n_users, n_items)

    return user_items


def train_als(user_items: coo_matrix) -> implicit.als.AlternatingLeastSquares:
    """
    Train implicit ALS collaborative filtering model.
    """
    print("Training ALS...")
    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REG,
        iterations=ALS_ITERS,
        random_state=RANDOM_STATE,
    )

    model.fit(user_items)
    return model


def main() -> None:
    df, user_ids, item_ids = load_and_filter()
    user_items = build_user_item_matrix(df)

    n_users = user_items.shape[0]
    n_items = user_items.shape[1]

    model = train_als(user_items)

    assert model.user_factors.shape[0] == n_users
    assert model.item_factors.shape[0] == n_items

    with open(PROC / "cf_als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(PROC / "cf_als_index.pkl", "wb") as f:
        pickle.dump(
            {
                "user_ids": list(user_ids),
                "item_ids": list(item_ids),
            },
            f,
        )

    print("Saved model + index successfully.")


if __name__ == "__main__":
    main()
