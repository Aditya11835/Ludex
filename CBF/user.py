import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from typing import Optional

from sklearn.preprocessing import normalize


# ======================================================
# STEAM FETCH + MAPPING
# ======================================================

def fetch_owned_games(steamid64: str, api_key: str) -> pd.DataFrame:
    """
    Call the Steam Web API to fetch owned games + playtime.

    Returns a DataFrame with columns:
        ['appid', 'title', 'playtime_min']
    """
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": api_key,
        "steamid": steamid64,
        "include_appinfo": 1,
        "include_played_free_games": 1,
        "format": "json",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()

    games = resp.json().get("response", {}).get("games", [])
    if not games:
        print("No games returned for this user.")
        return pd.DataFrame(columns=["appid", "title", "playtime_min"])

    df = pd.DataFrame(games)
    df = df.rename(
        columns={
            "appid": "appid",
            "playtime_forever": "playtime_min",
            "name": "title",
        }
    )
    df = df[["appid", "title", "playtime_min"]]
    df = df.sort_values("playtime_min", ascending=False)
    return df


def map_owned_to_indices(owned_df: pd.DataFrame, catalogue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map owned games (by appid) to row indices in the catalogue DataFrame.

    Adds a 'row_idx' column to owned_df (dropping any games not in the catalogue).
    """
    appid_to_idx = {int(a): i for i, a in enumerate(catalogue_df["appid"])}

    owned_df = owned_df.copy()
    owned_df["row_idx"] = owned_df["appid"].map(appid_to_idx)

    owned_df = owned_df.dropna(subset=["row_idx"])
    owned_df["row_idx"] = owned_df["row_idx"].astype(int)
    return owned_df


# ======================================================
# USER CONTENT PROFILE (SINGLE VECTOR, WITH COLD-START HANDLING)
# ======================================================

def build_user_content_profile(
    owned_mapped: pd.DataFrame,
    full_matrix_norm: sp.csr_matrix,
    min_playtime: int = 60,
    min_games_for_strict: int = 5,
    fallback_top_k: int = 5,
    max_strict_anchors: int = 10,
) -> Optional[np.ndarray]:
    """
    Build a single user content profile vector v_u in the same space as the games.

    Logic:
      - Prefer games with playtime >= min_playtime.
      - If there are at least `min_games_for_strict` such games:
            * sort them by playtime descending
            * take the top `max_strict_anchors` as anchors
      - Otherwise, fall back to the user's top `fallback_top_k` games by playtime
        (playtime_min > 0), even if they are below min_playtime.
      - If the user has no games with playtime > 0, return None (true cold-start).

    Steps (once anchors are chosen):
        1) Turn playtime into log-scaled weights w_i = log(1 + p_i).
        2) Normalise weights to sum to 1.
        3) Compute weighted average of their L2-normalised feature vectors.
        4) L2-normalise the resulting user vector.

    Returns:
        user_vec: np.ndarray of shape (d,) with ||user_vec||_2 = 1,
                  or None if we cannot build a profile.
    """
    owned_mapped = owned_mapped.copy()

    # ----- 1) Strict anchors: games above min_playtime -----
    strict_anchors = owned_mapped[owned_mapped["playtime_min"] >= min_playtime].copy()

    if len(strict_anchors) >= min_games_for_strict:
        # Use only the top `max_strict_anchors` by playtime to avoid noise
        strict_anchors = strict_anchors.sort_values("playtime_min", ascending=False)
        anchors_df = strict_anchors.head(max_strict_anchors)
        print(
            f"Using top {len(anchors_df)} games with playtime >= {min_playtime} minutes "
            "to build user content profile."
        )
    else:
        # ----- 2) Fallback: few or zero strict anchors -----
        print(
            f"User has only {len(strict_anchors)} games with playtime >= {min_playtime} minutes. "
            "Falling back to top-played games (cold-start contingency)."
        )

        fallback_candidates = owned_mapped[owned_mapped["playtime_min"] > 0].copy()
        if fallback_candidates.empty:
            # True cold-start: no meaningful playtime at all
            print(
                "User has no games with non-zero playtime. "
                "Cannot build a content-based profile (pure cold-start)."
            )
            return None

        fallback_candidates = fallback_candidates.sort_values("playtime_min", ascending=False)
        anchors_df = fallback_candidates.head(fallback_top_k)
        print(
            f"Using top {len(anchors_df)} games by playtime (> 0 min) "
            "to approximate user content profile."
        )

    # If somehow we still have no anchors, bail out.
    if anchors_df.empty:
        print("No anchors available to build user profile.")
        return None

    # Row indices into full_matrix_norm
    row_idxs = anchors_df["row_idx"].to_numpy()
    playtimes = anchors_df["playtime_min"].to_numpy(dtype=float)

    # ----- 3) log-scaled playtime weights, normalised to sum=1 -----
    w_tilde = np.log1p(playtimes)  # log(1 + p)
    weight_sum = w_tilde.sum()
    if weight_sum <= 0:
        print("Non-positive weight sum – cannot build user profile.")
        return None

    weights = w_tilde / weight_sum  # shape (k,)

    # ----- 4) Weighted average of game vectors -----
    subset = full_matrix_norm[row_idxs]           # (k, d) sparse CSR
    weighted = subset.multiply(weights.reshape(-1, 1))
    user_vec = weighted.sum(axis=0)              # (1, d)

    # Convert to dense 1D array
    user_vec = np.asarray(user_vec).ravel()      # (d,)

    # ----- 5) L2-normalise the user vector -----
    norm = np.linalg.norm(user_vec)
    if norm == 0.0:
        print("User vector has zero norm – cannot normalise.")
        return None

    user_vec = user_vec / norm
    return user_vec


# ======================================================
# GLOBAL CBF SCORING
# ======================================================

def score_games_cbf(
    user_vec: np.ndarray,
    full_matrix_norm: sp.csr_matrix,
) -> np.ndarray:
    """
    Given a user content profile vector v_u and the game feature matrix F,
    compute CBF scores for ALL games:

        CBF(u, i) = v_u · f_i  (cosine similarity)

    Assumes:
        - full_matrix_norm: CSR (n_games, d), rows L2-normalised.
        - user_vec: np.ndarray (d,), L2-normalised.

    Returns:
        scores: np.ndarray of shape (n_games,), where scores[i] = CBF(u, i).
    """
    if user_vec.ndim != 1:
        user_vec = user_vec.ravel()

    scores = full_matrix_norm.dot(user_vec)  # (n, d) ⋅ (d,) → (n,)
    return np.asarray(scores).ravel()
