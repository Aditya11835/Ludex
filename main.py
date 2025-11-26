import os
import argparse
from dotenv import load_dotenv
import numpy as np

from CBF.CBF_recommend import generate_cbf_recommendations
from CF.CF_recommend import get_cf_scores_for_appids


# ======================================================
# CONFIG

TOP_N = 20
MIN_PLAYTIME = 60            # Minimum minutes for an owned game to count strongly
CANDIDATE_POOL_SIZE = 1000   # How many top-CBF games to consider before MMR
BETA_ANCHOR_BLEND = 0.3      # Weight for anchor_soft vs global CBF
LAMBDA_MMR = 0.7             # Relevance vs diversity in MMR
ALPHA_HYBRID = 0.5           # CF vs CBF weight in final hybrid score


# ======================================================
# HYBRID HOOKS (CBF + CF)

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

    If cf_scores is None, this function just returns cbf_scores.
    """
    cbf_scores = np.asarray(cbf_scores, dtype=float)

    if cf_scores is None:
        # CF not available → pure CBF
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
        # Future: fall back directly to pure CF or popularity here.
        return

    recs = recs.copy()

    # Base CBF signal
    if "cbf_anchor_combined" in recs.columns:
        cbf_base = recs["cbf_anchor_combined"].to_numpy(dtype=float)
    else:
        cbf_base = recs.get("cbf", 0.0).to_numpy(dtype=float)

    cf_scores = None  # default → pure CBF if anything fails

    # --------------------------------------------------
    # 2) CF Integration (modular)
    #    Compute CF scores for the same appids present in `recs`
    appids = recs["appid"].astype(int).to_numpy()

    try:
        cf_scores = get_cf_scores_for_appids(
            steamid64=steamid64,
            appids=appids,
            enrich_interactions=True,
            force_retrain=False,
        )
        if cf_scores is None:
            print("\n[HYBRID] CF unavailable for this user/candidate set; using pure CBF.")
    except Exception as e:
        print(f"\n[HYBRID] Warning: CF integration failed ({e!r}); falling back to CBF-only.")
        cf_scores = None

    # --------------------------------------------------
    # 3) Final hybrid score

    hybrid_scores = combine_cbf_cf(
        cbf_scores=cbf_base,
        cf_scores=cf_scores,
        alpha=ALPHA_HYBRID,
    )

    recs["cf_score"] = cf_scores if cf_scores is not None else np.zeros_like(cbf_base)
    recs["hybrid_score"] = hybrid_scores

    # --------------------------------------------------
    # 4) Display results

    print("\nTop Recommendations (Hybrid CBF + CF when available):")
    for _, row in recs.iterrows():
        cbf_val = row.get("cbf", np.nan)
        hybrid_val = row.get("hybrid_score", np.nan)
        cf_val = row.get("cf_score", np.nan)

        try:
            cbf_str = f"{float(cbf_val):.4f}" if np.isfinite(cbf_val) else "nan"
        except Exception:
            cbf_str = "nan"

        try:
            cf_str = f"{float(cf_val):.4f}" if np.isfinite(cf_val) else "nan"
        except Exception:
            cf_str = "nan"

        try:
            hybrid_str = f"{float(hybrid_val):.4f}" if np.isfinite(hybrid_val) else "nan"
        except Exception:
            hybrid_str = "nan"

        print(
            f"- {row['title']} "
            f"(appid={row['appid']}, "
            f"cbf={cbf_str}, "
            f"cf={cf_str}, "
            f"hybrid={hybrid_str})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ludex hybrid recommender (TF-IDF + anchors + MMR + CF ALS)."
    )
    parser.add_argument(
        "steamid64",
        help="SteamID64 of the user to recommend games for.",
    )
    args = parser.parse_args()

    main(args.steamid64)
