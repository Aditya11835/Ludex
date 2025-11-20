import os
from dotenv import load_dotenv

import numpy as np

from CBF.model import load_catalogue_and_features
from CBF.user import (
    fetch_owned_games,
    map_owned_to_indices,
    build_user_content_profile,
    score_games_cbf,
)
from CBF.catalogue_update import ensure_user_games_in_catalogue_and_refresh


# ======================================================
# CONFIG
# ======================================================

TOP_N = 15
MIN_PLAYTIME = 60  # Minimum minutes for an owned game to count strongly


# ======================================================
# HYBRID HOOKS (CBF + CF)
# ======================================================

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
# ======================================================

def main():
    load_dotenv()
    api_key = os.getenv("STEAM_API_KEY")
    if not api_key:
        raise RuntimeError("STEAM_API_KEY not found in .env")

    # Ask for SteamID64
    steamid64 = input("Enter your SteamID64: ").strip()
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
        # You can tune these if needed:
        # min_games_for_strict=5,
        # fallback_top_k=5,
        # max_strict_anchors=10,
    )

    if user_vec is None:
        print(
            "\nCould not build a content-based profile (cold-start or degenerate case). "
            "In a future hybrid system, you would fall back to CF/popularity here."
        )
        return

    # --------------------------------------------------
    # 6) Compute global CBF scores for ALL games
    print("Scoring all games via cosine similarity to the user profile (CBF)…")
    cbf_scores = score_games_cbf(user_vec, full_matrix_norm)  # shape (n_games,)
    cbf_vector = cbf_scores.copy()  # kept explicitly for CF + hybrid use

    # OPTIONAL: save to disk for offline evaluation / later steps
    # np.save("data/processed/last_user_cbf_scores.npy", cbf_vector)

    # --------------------------------------------------
    # 7) (Placeholder) Load / compute CF scores for this user
    # In the future, plug in your CF model here.
    cf_scores = None  # For now, pure CBF

    # --------------------------------------------------
    # 8) Combine CBF + CF into hybrid scores (for now: just CBF)
    hybrid_scores = combine_cbf_cf(cbf_vector, cf_scores, alpha=0.5)

    # Attach scores to catalogue
    rec_df = df.copy()
    rec_df["cbf_score"] = cbf_vector
    rec_df["hybrid_score"] = hybrid_scores

    # --------------------------------------------------
    # 9) Build recommendation list (mask owned, sort by hybrid score)
    owned_appids = set(owned_mapped["appid"])
    candidates = rec_df[~rec_df["appid"].isin(owned_appids)].copy()

    candidates = candidates.sort_values("hybrid_score", ascending=False)
    recs = candidates.head(TOP_N)

    # --------------------------------------------------
    # 10) Display results
    if recs.empty:
        print("\nNo recommendations generated.")
        return

    print("\nTop Recommendations (currently CBF-only; hybrid + auto-catalogue-ready):")
    for _, row in recs.iterrows():
        print(
            f"- {row['title']} "
            f"(appid={row['appid']}, "
            f"cbf={row['cbf_score']:.4f}, "
            f"hybrid={row['hybrid_score']:.4f})"
        )


if __name__ == "__main__":
    main()
