import os
import argparse
from dotenv import load_dotenv

import numpy as np

from CBF.model import load_catalogue_and_features
from CBF.user import (
    fetch_owned_games,
    map_owned_to_indices,
    build_user_content_profile,
    recommend_cbf_user_plus_anchors_mmr,
)
from CBF.catalogue_update import ensure_user_games_in_catalogue_and_refresh


# ======================================================
# CONFIG

TOP_N = 20
MIN_PLAYTIME = 60          # Minimum minutes for an owned game to count strongly
CANDIDATE_POOL_SIZE = 500  # How many top-CBF games to consider before MMR
BETA_ANCHOR_BLEND = 0.3    # Weight for anchor_soft vs global CBF
LAMBDA_MMR = 0.7           # Relevance vs diversity in MMR


# ======================================================
# HYBRID HOOKS (CBF + CF) – NOT USED YET, FOR FUTURE CF

def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min–max normalisation to [0, 1] for a 1D score vector.

    If scores are constant, returns a zero vector to avoid NaNs.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores

    s_min = scores.min()
    s_max = scores.max()

    if s_max <= s_min:
        # All scores identical (or degenerate) → return zeros
        return np.zeros_like(scores)

    return (scores - s_min) / (s_max - s_min)


def combine_cbf_cf(
    cbf_scores: np.ndarray,
    cf_scores: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Combine CBF and CF scores into a single hybrid score vector.

    Hybrid(u, i) = alpha * CF(u, i) + (1 - alpha) * CBF(u, i)

    For now:
        - If cf_scores is None, this function just returns cbf_scores.
        - When CF is implemented, pass a same-length cf_scores array.
    """
    cbf_scores = np.asarray(cbf_scores, dtype=float)

    if cf_scores is None:
        # CF not available yet → pure CBF
        return cbf_scores

    cf_scores = np.asarray(cf_scores, dtype=float)
    if cf_scores.shape != cbf_scores.shape:
        raise ValueError(
            f"Shape mismatch: cbf_scores {cbf_scores.shape}, cf_scores {cf_scores.shape}"
        )

    cbf_norm = normalise_scores(cbf_scores)
    cf_norm = normalise_scores(cf_scores)

    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))  # clamp to [0, 1]

    hybrid = alpha * cf_norm + (1.0 - alpha) * cbf_norm
    return hybrid


# ======================================================
# MAIN

def main(steamid64: str):
    load_dotenv()
    api_key = os.getenv("STEAM_API_KEY")
    if not api_key:
        raise RuntimeError("STEAM_API_KEY not found in .env")

    steamid64 = steamid64.strip()
    if not steamid64:
        raise RuntimeError("SteamID64 is required")

    # --------------------------------------------------
    # 1) Load catalogue + sparse, L2-normalised feature matrix
    df, full_matrix_norm = load_catalogue_and_features()

    # --------------------------------------------------
    # 2) Fetch owned games from Steam
    owned_df = fetch_owned_games(steamid64, api_key)
    if owned_df.empty:
        print("\nNo visible games for this user – cannot build a content profile.")
        return

    # --------------------------------------------------
    # 3) Ensure all owned games exist in the catalogue; if not, extend catalogue
    #    and rebuild the feature matrix BEFORE building the user profile.
    print("\nChecking for owned games missing from the catalogue…")
    df, full_matrix_norm = ensure_user_games_in_catalogue_and_refresh(
        owned_df=owned_df,
        catalogue_df=df,
    )

    # After this point, df and full_matrix_norm are guaranteed to include
    # ALL games from owned_df (if metadata could be fetched).

    # --------------------------------------------------
    # 4) Map owned games to indices in the (potentially updated) catalogue
    owned_mapped = map_owned_to_indices(owned_df, df)
    if owned_mapped.empty:
        print("\nNone of the user's games exist in the catalogue even after update.")
        return

    # --------------------------------------------------
    # 5) Build a single user content profile vector
    print("\nBuilding user content profile (single vector in TF–IDF space)…")
    user_vec = build_user_content_profile(
        owned_mapped=owned_mapped,
        full_matrix_norm=full_matrix_norm,
        min_playtime=MIN_PLAYTIME,
    )

    if user_vec is None:
        print(
            "\nCould not build a content-based profile (cold-start or degenerate case). "
            "In a future hybrid system, you would fall back to CF/popularity here."
        )
        return

    # --------------------------------------------------
    # 6) Generate CBF recommendations using:
    #    - global user vector
    #    - anchor-based soft similarity
    #    - MMR diversity re-ranking
    print(
        "\nScoring games via global user vector + anchor soft scores + MMR "
        "(CBF-only for now, hybrid-ready)…"
    )

    recs = recommend_cbf_user_plus_anchors_mmr(
        catalogue_df=df,
        full_matrix_norm=full_matrix_norm,
        owned_mapped=owned_mapped,
        user_vec=user_vec,
        top_n=TOP_N,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        min_playtime=MIN_PLAYTIME,
        beta=BETA_ANCHOR_BLEND,
        lambda_mmr=LAMBDA_MMR,
    )

    if recs.empty:
        print("\nNo recommendations generated.")
        return

    # --------------------------------------------------
    # 7) (Optional, future) integrate CF here
    # For now, treat `cbf_anchor_combined` as the CBF signal.
    # Once CF is available, you can:
    #
    #   cf_scores_for_recs = ...
    #   recs['hybrid_score'] = combine_cbf_cf(
    #       cbf_scores=recs['cbf_anchor_combined'].to_numpy(),
    #       cf_scores=cf_scores_for_recs,
    #       alpha=0.5,
    #   )
    #
    # For now, hybrid_score = cbf_anchor_combined.
    recs = recs.copy()
    if "cbf_anchor_combined" in recs.columns:
        recs["hybrid_score"] = recs["cbf_anchor_combined"]
    else:
        # Fallback: just use pure CBF if for some reason the column is missing
        recs["hybrid_score"] = recs.get("cbf", 0.0)

    # --------------------------------------------------
    # 8) Display results
    print("\nTop Recommendations (currently CBF-only; hybrid + auto-catalogue-ready):")
    for _, row in recs.iterrows():
        cbf_val = row.get("cbf", np.nan)
        hybrid_val = row.get("hybrid_score", np.nan)
        try:
            cbf_str = f"{cbf_val:.4f}" if np.isfinite(cbf_val) else "nan"
        except Exception:
            cbf_str = "nan"
        try:
            hybrid_str = f"{hybrid_val:.4f}" if np.isfinite(hybrid_val) else "nan"
        except Exception:
            hybrid_str = "nan"

        print(
            f"- {row['title']} "
            f"(appid={row['appid']}, "
            f"cbf={cbf_str}, "
            f"hybrid={hybrid_str})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ludex CBF recommender (TF-IDF + anchors + MMR)."
    )
    parser.add_argument(
        "steamid64",
        help="SteamID64 of the user to recommend games for.",
    )
    args = parser.parse_args()

    main(args.steamid64)
