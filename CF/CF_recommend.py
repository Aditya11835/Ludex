"""
CF_recommend
------------

Collaborative-filtering (ALS) utilities for a single Steam user.

This module encapsulates the CF-only orchestration that used to live in
the standalone `recommend_for_user.py`, mirroring how
`CBF.CBF_recommend` encapsulates the CBF pipeline.

Public API
----------
    from CF.CF_recommend import get_cf_scores_for_appids

    cf_scores = get_cf_scores_for_appids(
        steamid64=...,
        appids=[570, 440, 730, ...],
    )

Returns
-------
    cf_scores : numpy.ndarray | None
        A 1D vector of CF scores aligned with the given `appids`.
        If CF cannot be used (missing interactions/model/user/overlap),
        returns None. When it returns an array, positions where CF has
        no factor for a given appid are filled with 0.0.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np

from CF.cf_model import load_cf_model, INTERACTIONS_CSV
from CF.interactions_update import ensure_users_in_data_and_retrain


def get_cf_scores_for_appids(
    steamid64: str,
    appids: Sequence[int],
    *,
    enrich_interactions: bool = True,
    force_retrain: bool = False,
) -> Optional[np.ndarray]:
    """
    Compute CF scores for a given user and a list/array of candidate appids.

    This is intentionally "low-level": it does NOT perform any ranking
    or sampling; it just returns a score for each appid, suitable for
    hybridisation with CBF scores.

    Steps
    -----
      1) Check that the interactions CSV exists (otherwise CF is disabled).
      2) Optionally ensure this user is present in the interactions data,
         retraining the ALS model if needed.
      3) Load the ALS model + user/item id mappings.
      4) Map SteamID64 -> user index in the ALS model.
      5) Map each candidate appid to an ALS item index (if present).
      6) For all candidates present in the model, compute:

            score(u, i) = <user_factors[u], item_factors[i]>

         and return a dense score vector aligned with `appids`.

    Parameters
    ----------
    steamid64 : str
        SteamID64 of the user.
    appids : Sequence[int]
        Candidate appids to score (e.g. from CBF pipeline).
    enrich_interactions : bool, default True
        If True, calls `ensure_users_in_data_and_retrain([steamid64])`
        before loading the CF model, so new users can be added.
    force_retrain : bool, default False
        Passed through to `load_cf_model(force_retrain=...)`. Usually
        False; set True only if you intentionally want to retrain ALS.

    Returns
    -------
    numpy.ndarray or None
        1D float64 array of length len(appids), where each entry is the
        raw CF score for that appid (0.0 if not present in the model).
        Returns None if CF cannot be used at all (e.g. interactions CSV
        missing, user not in model after enrichment, or no overlap).
    """
    steamid64 = str(steamid64).strip()
    if not steamid64:
        raise ValueError("steamid64 must be a non-empty string")

    # Normalise appids input
    appids_arr = np.asarray(list(appids), dtype=int)
    if appids_arr.size == 0:
        # Trivial case: nothing to score.
        return np.zeros(0, dtype=float)

    # --------------------------------------------------
    # 1) Interactions CSV presence
    if not INTERACTIONS_CSV.exists():
        # CF pipeline not ready yet → signal to caller to fall back.
        return None

    # --------------------------------------------------
    # 2) Ensure user exists in interactions (optional)
    try:
        if enrich_interactions:
            ensure_users_in_data_and_retrain([steamid64])
    except Exception as e:
        # Any issue in enrichment → treat CF as unavailable.
        print(f"[CF] Warning: ensure_users_in_data_and_retrain failed: {e!r}")
        return None

    # --------------------------------------------------
    # 3) Load CF model + ID mappings
    try:
        model, user_ids, item_ids = load_cf_model(force_retrain=force_retrain)
    except Exception as e:
        print(f"[CF] Warning: load_cf_model failed: {e!r}")
        return None

    # Normalise IDs
    user_ids = [str(sid) for sid in user_ids]
    item_ids = list(item_ids)

    # --------------------------------------------------
    # 4) Map SteamID64 -> user_idx
    if steamid64 not in user_ids:
        print(f"[CF] User {steamid64} is not in ALS model; CF disabled for this user.")
        return None

    user_idx = user_ids.index(steamid64)

    # --------------------------------------------------
    # 5) Build appid -> ALS item index lookup
    appid_to_item_idx = {int(a): i for i, a in enumerate(item_ids)}

    cf_scores = np.zeros(appids_arr.shape[0], dtype=float)

    positions: list[int] = []
    cf_item_indices: list[int] = []

    for pos, appid in enumerate(appids_arr):
        idx = appid_to_item_idx.get(int(appid))
        if idx is not None:
            positions.append(pos)
            cf_item_indices.append(idx)

    if not cf_item_indices:
        # None of the candidates exist in CF catalogue
        print("[CF] No overlap between candidates and CF item catalogue.")
        return None

    # --------------------------------------------------
    # 6) Compute dot-product scores for overlapping items
    cf_item_indices_arr = np.asarray(cf_item_indices, dtype=int)

    try:
        # user_factors: (n_users, k), item_factors: (n_items, k)
        user_vec_cf = model.user_factors[user_idx]                # (k,)
        item_factors_subset = model.item_factors[cf_item_indices_arr]  # (m, k)

        scores_subset = item_factors_subset @ user_vec_cf         # (m,)
    except Exception as e:
        print(f"[CF] Warning: failed to compute CF scores: {e!r}")
        return None

    for pos, s in zip(positions, scores_subset):
        cf_scores[pos] = float(s)

    return cf_scores
