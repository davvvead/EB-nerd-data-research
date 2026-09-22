"""
Discovery Headroom optimizer and common support index.
Optimizes counterfactual slates by swapping at most 2 items with candidates from
observable supply satisfying common support and relevance floor tau,
while retaining at least 97% of observed slate relevance.
Strictly gated by validation relevance-model qualification.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.config import (
    HEADROOM_MAX_SWAPS,
    HEADROOM_PRIMARY_RELEVANCE_TOLERANCE,
    HEADROOM_SENSITIVITY_TOLERANCES,
    COMMON_SUPPORT_N_PRIMARY,
    COMMON_SUPPORT_N_SENSITIVITY,
)


class CommonSupportIndex:
    """
    Tracks article in-view frequency across distinct users within 30-minute time buckets.
    Primary support threshold: >= 3 distinct users.
    Sensitivity threshold: >= 5 distinct users.
    """

    def __init__(self, impressions_df: pd.DataFrame):
        """
        Build common support index from impressions.
        Requires '_time_bucket', 'user_id', and '_slate' (list of article IDs).
        """
        # Explode slates to (time_bucket, user_id, article_id)
        records = []
        for bucket, uid, slate in zip(
            impressions_df["_time_bucket"],
            impressions_df["user_id"],
            impressions_df["_slate"],
        ):
            for aid in slate:
                records.append((bucket, uid, aid))

        if not records:
            self.bucket_article_user_counts: Dict[Any, Dict[int, int]] = {}
            return

        df_exp = pd.DataFrame(records, columns=["bucket", "user_id", "article_id"])
        # Distinct users per (bucket, article_id)
        counts = df_exp.groupby(["bucket", "article_id"])["user_id"].nunique().reset_index()
        
        # Build nested dictionary: bucket -> article_id -> distinct_user_count
        self.bucket_article_user_counts = {}
        for b, aid, cnt in zip(counts["bucket"], counts["article_id"], counts["user_id"]):
            if b not in self.bucket_article_user_counts:
                self.bucket_article_user_counts[b] = {}
            self.bucket_article_user_counts[b][aid] = int(cnt)

    def get_supported_articles(
        self,
        time_bucket: Any,
        min_distinct_users: int = COMMON_SUPPORT_N_PRIMARY,
    ) -> Set[int]:
        """Return set of article IDs with >= min_distinct_users in given time bucket."""
        b_dict = self.bucket_article_user_counts.get(time_bucket, {})
        return {aid for aid, cnt in b_dict.items() if cnt >= min_distinct_users}


@dataclass(frozen=True)
class HeadroomResult:
    """Outcome of Headroom optimization for a single impression."""
    observed_discovery: float
    observed_relevance: float
    optimized_discovery: float
    optimized_relevance: float
    discovery_gain: float
    relevance_retention_ratio: float
    n_swaps: int
    swapped_out_ids: List[int]
    swapped_in_ids: List[int]
    is_qualified: bool
    candidate_pool_size: int


def optimize_headroom_single_impression(
    observed_slate: List[int],
    article_novelty_map: Dict[int, float],
    article_relevance_map: Dict[int, float],
    candidate_ids: List[int],
    tau: float,
    qualification_status: str,
    relevance_tolerance: float = HEADROOM_PRIMARY_RELEVANCE_TOLERANCE,
    max_swaps: int = HEADROOM_MAX_SWAPS,
) -> HeadroomResult:
    """
    Perform exact constrained Headroom optimization for a single impression.
    
    Candidate pool must already satisfy:
      - 24h observable supply
      - common support threshold (>= 3 distinct users)
      - predicted relevance >= tau
      - candidate not in observed slate
      
    Constraints:
      - At most max_swaps (default 2)
      - Alternative mean relevance >= (1.0 - relevance_tolerance) * observed mean relevance
      
    Objective:
      - Maximize alternative slate discovery - observed slate discovery
    """
    if qualification_status != "QUALIFIED":
        raise PermissionError(
            f"Headroom optimizer is strictly gated by relevance model qualification. "
            f"Current status: '{qualification_status}' (must be 'QUALIFIED')."
        )

    k = len(observed_slate)
    if k == 0:
        return HeadroomResult(
            observed_discovery=0.0,
            observed_relevance=0.0,
            optimized_discovery=0.0,
            optimized_relevance=0.0,
            discovery_gain=0.0,
            relevance_retention_ratio=1.0,
            n_swaps=0,
            swapped_out_ids=[],
            swapped_in_ids=[],
            is_qualified=True,
            candidate_pool_size=0,
        )

    obs_d = np.array([
        article_novelty_map.get(aid, 0.0) if article_relevance_map.get(aid, 0.0) >= tau else 0.0
        for aid in observed_slate
    ], dtype=np.float64)
    obs_r = np.array([article_relevance_map.get(aid, 0.0) for aid in observed_slate], dtype=np.float64)

    obs_discovery = float(np.mean(obs_d))
    obs_relevance = float(np.mean(obs_r))
    min_required_total_r = (1.0 - relevance_tolerance) * np.sum(obs_r)
    min_delta_r = min_required_total_r - np.sum(obs_r)

    # Filter candidates strictly to those with r >= tau and not in slate
    obs_set = set(observed_slate)
    valid_cands = [
        cid for cid in candidate_ids
        if cid not in obs_set and article_relevance_map.get(cid, 0.0) >= tau
    ]

    if not valid_cands:
        return HeadroomResult(
            observed_discovery=obs_discovery,
            observed_relevance=obs_relevance,
            optimized_discovery=obs_discovery,
            optimized_relevance=obs_relevance,
            discovery_gain=0.0,
            relevance_retention_ratio=1.0,
            n_swaps=0,
            swapped_out_ids=[],
            swapped_in_ids=[],
            is_qualified=True,
            candidate_pool_size=0,
        )

    cand_d = np.array([article_novelty_map.get(cid, 0.0) for cid in valid_cands], dtype=np.float64)
    cand_r = np.array([article_relevance_map.get(cid, 0.0) for cid in valid_cands], dtype=np.float64)

    # If candidate pool is large, keep the most promising candidates
    # (top candidates by novelty and top by relevance to ensure feasible Pareto solutions)
    if len(valid_cands) > 40:
        top_nov_idx = np.argsort(-cand_d)[:30]
        top_rel_idx = np.argsort(-cand_r)[:15]
        selected_cand_idx = np.unique(np.concatenate([top_nov_idx, top_rel_idx]))
        valid_cands = [valid_cands[i] for i in selected_cand_idx]
        cand_d = cand_d[selected_cand_idx]
        cand_r = cand_r[selected_cand_idx]

    m = len(valid_cands)

    best_gain = 0.0
    best_n_swaps = 0
    best_out: List[int] = []
    best_in: List[int] = []
    best_alt_r = obs_relevance

    # 1-SWAP SEARCH
    # gain = (cand_d[j] - obs_d[i]) / k
    # delta_r = cand_r[j] - obs_r[i] >= min_delta_r
    gain_mat_1 = (cand_d[:, None] - obs_d[None, :]) / k
    delta_r_mat_1 = cand_r[:, None] - obs_r[None, :]
    valid_1 = delta_r_mat_1 >= min_delta_r

    if valid_1.any():
        masked_gain_1 = np.where(valid_1, gain_mat_1, -np.inf)
        max_1_idx = np.unravel_index(np.argmax(masked_gain_1), masked_gain_1.shape)
        if masked_gain_1[max_1_idx] > best_gain:
            best_gain = float(masked_gain_1[max_1_idx])
            best_n_swaps = 1
            best_in = [valid_cands[max_1_idx[0]]]
            best_out = [observed_slate[max_1_idx[1]]]
            best_alt_r = obs_relevance + float(delta_r_mat_1[max_1_idx]) / k

    # 2-SWAP SEARCH (if max_swaps >= 2 and at least 2 candidates and 2 slate items)
    if max_swaps >= 2 and m >= 2 and k >= 2:
        # Generate all 2-combinations of observed slate items
        slate_comb_i, slate_comb_j = np.triu_indices(k, k=1)
        obs_pair_d = obs_d[slate_comb_i] + obs_d[slate_comb_j]
        obs_pair_r = obs_r[slate_comb_i] + obs_r[slate_comb_j]
        n_obs_pairs = len(slate_comb_i)

        # Generate all 2-combinations of candidate items
        cand_comb_i, cand_comb_j = np.triu_indices(m, k=1)
        cand_pair_d = cand_d[cand_comb_i] + cand_d[cand_comb_j]
        cand_pair_r = cand_r[cand_comb_i] + cand_r[cand_comb_j]
        n_cand_pairs = len(cand_comb_i)

        gain_mat_2 = (cand_pair_d[:, None] - obs_pair_d[None, :]) / k
        delta_r_mat_2 = cand_pair_r[:, None] - obs_pair_r[None, :]
        valid_2 = delta_r_mat_2 >= min_delta_r

        if valid_2.any():
            masked_gain_2 = np.where(valid_2, gain_mat_2, -np.inf)
            max_2_idx = np.unravel_index(np.argmax(masked_gain_2), masked_gain_2.shape)
            if masked_gain_2[max_2_idx] > best_gain:
                best_gain = float(masked_gain_2[max_2_idx])
                best_n_swaps = 2
                c_idx1, c_idx2 = cand_comb_i[max_2_idx[0]], cand_comb_j[max_2_idx[0]]
                s_idx1, s_idx2 = slate_comb_i[max_2_idx[1]], slate_comb_j[max_2_idx[1]]
                best_in = [valid_cands[c_idx1], valid_cands[c_idx2]]
                best_out = [observed_slate[s_idx1], observed_slate[s_idx2]]
                best_alt_r = obs_relevance + float(delta_r_mat_2[max_2_idx]) / k

    opt_discovery = obs_discovery + best_gain
    retention_ratio = float(best_alt_r / obs_relevance) if obs_relevance > 0 else 1.0

    return HeadroomResult(
        observed_discovery=obs_discovery,
        observed_relevance=obs_relevance,
        optimized_discovery=opt_discovery,
        optimized_relevance=best_alt_r,
        discovery_gain=best_gain,
        relevance_retention_ratio=retention_ratio,
        n_swaps=best_n_swaps,
        swapped_out_ids=best_out,
        swapped_in_ids=best_in,
        is_qualified=True,
        candidate_pool_size=m,
    )
