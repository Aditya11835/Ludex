import os
import argparse
from dotenv import load_dotenv

import numpy as np

from CBF.CBF_recommend import generate_cbf_recommendations


# ======================================================
# CONFIG (kept in main because they may affect hybrid)

TOP_N = 20
MIN_PLAYTIME = 60          # Minimum minutes for an owned game to count strongly
CANDIDATE_POOL_SIZE = 1000  # How many top-CBF games to consider before MMR
BETA_ANCHOR_BLEND = 0.3    # Weight for anchor_soft vs global CBF
LAMBDA_MMR = 0.7           # Relevance vs diversity in MMR


# ======================================================
# HYBRID HOOKS (CBF + CF) – CF not plugged in yet

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
    # 1) Run the CBF-only pipeline to get recommendations
    recs = generate_cbf_recommendations(
        steamid64=steamid64,
        api_key=api_key,
        top_n=TOP_N,
        min_playtime=MIN_PLAYTIME,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        beta_anchor_blend=BETA_ANCHOR_BLEND,
        lambda_mmr=LAMBDA_MMR,
    )

    if recs.empty:
        print("\nNo content-based recommendations generated for this user.")
        # In a future hybrid system, you can fall back to CF/popularity here.
        return

    # --------------------------------------------------
    # 2) (Optional, future) integrate CF here.
    #
    # For now, treat `cbf_anchor_combined` as the CBF signal.
    # Once CF is available, you can do something like:
    #
    #   cf_scores_for_recs = ...
    #   recs['hybrid_score'] = combine_cbf_cf(
    #       cbf_scores=recs['cbf_anchor_combined'].to_numpy(),
    #       cf_scores=cf_scores_for_recs,
    #       alpha=0.5,
    #   )
    #
    recs = recs.copy()
    if "cbf_anchor_combined" in recs.columns:
        recs["hybrid_score"] = recs["cbf_anchor_combined"]
    else:
        # Fallback: just use pure CBF if for some reason the column is missing
        recs["hybrid_score"] = recs.get("cbf", 0.0)

    # --------------------------------------------------
    # 3) Display results
    print("\nTop Recommendations (currently CBF-only; hybrid-ready):")
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
        description="Ludex CBF recommender (TF-IDF + anchors + MMR, hybrid-ready)."
    )
    parser.add_argument(
        "steamid64",
        help="SteamID64 of the user to recommend games for.",
    )
    args = parser.parse_args()

    main(args.steamid64)
