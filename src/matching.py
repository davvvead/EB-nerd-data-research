"""
Deterministic 1:1 nearest-neighbor matching without replacement
comparing Low-breadth vs High-breadth impressions.
Follows exact frozen matching rules and tie-breaking specifications.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import MATCH_TIME_BUCKET_MINUTES, SLATE_SIZE_CALIPER


@dataclass(frozen=True)
class MatchingResults:
    """Summary and pair table of deterministic matched comparison."""
    pairs_df: pd.DataFrame
    eligible_low_count: int
    eligible_high_count: int
    matched_pairs_count: int
    low_match_rate: float
    high_match_rate: float
    unmatched_low_count: int
    unmatched_high_count: int


def perform_deterministic_matching(
    df: pd.DataFrame,
    breadth_col: str = "breadth_cohort",
    caliper: int = SLATE_SIZE_CALIPER,
) -> MatchingResults:
    """
    Perform 1:1 nearest-neighbor matching without replacement between Low and High breadth impressions.
    
    Exact matching strata:
      (time_bucket_30m, device_type, is_subscriber, session_stage, session_length_bin, history_length_bin)
      
    Caliper:
      abs(slate_size_low - slate_size_high) <= 1
      
    Execution order within each stratum:
      Low impressions processed in ascending order of:
        1. impression_time
        2. impression_id
        
      High candidate selected by:
        1. smallest absolute slate-size difference
        2. smallest absolute impression-time difference
        3. smallest impression_id
    """
    # Handle history_length_bin vs history_bin alias
    if "history_length_bin" not in df.columns and "history_bin" in df.columns:
        df = df.assign(history_length_bin=df["history_bin"])

    required_cols = {
        "impression_id", "impression_time", "_time_ns", "_slate_size",
        "device_type", "is_subscriber", "_session_stage", "_session_len_bin",
        "history_length_bin", "_time_bucket", breadth_col,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Matching input DataFrame missing required columns: {missing}")

    cohort_series = df[breadth_col].astype(str).str.lower()
    low_df = df[cohort_series == "low"].copy()
    high_df = df[cohort_series == "high"].copy()

    n_low = len(low_df)
    n_high = len(high_df)

    if n_low == 0 or n_high == 0:
        empty_pairs = pd.DataFrame()
        return MatchingResults(
            pairs_df=empty_pairs,
            eligible_low_count=n_low,
            eligible_high_count=n_high,
            matched_pairs_count=0,
            low_match_rate=0.0,
            high_match_rate=0.0,
            unmatched_low_count=n_low,
            unmatched_high_count=n_high,
        )

    stratum_cols = [
        "_time_bucket", "device_type", "is_subscriber",
        "_session_stage", "_session_len_bin", "history_length_bin",
    ]

    matched_pairs: List[Dict[str, Any]] = []
    
    # Group by exact strata
    low_grouped = low_df.groupby(stratum_cols, sort=False)
    high_grouped = high_df.groupby(stratum_cols, sort=False)
    high_strata_keys = set(high_grouped.groups.keys())

    for stratum_key, low_group in low_grouped:
        if stratum_key not in high_strata_keys:
            continue

        high_group = high_grouped.get_group(stratum_key)
        
        # Sort Low impressions deterministically
        sorted_low = low_group.sort_values(
            by=["impression_time", "impression_id"],
            ascending=[True, True]
        )

        # Active pool of High candidates in this stratum
        # Columns: impression_id, _slate_size, _time_ns, user_id, etc.
        high_pool = high_group.copy()

        for _, low_row in sorted_low.iterrows():
            if high_pool.empty:
                break

            low_slate = low_row["_slate_size"]
            low_time_ns = low_row["_time_ns"]
            low_id = low_row["impression_id"]

            # Apply caliper filter
            slate_diffs = np.abs(high_pool["_slate_size"].to_numpy() - low_slate)
            caliper_mask = slate_diffs <= caliper
            if not caliper_mask.any():
                continue

            candidates = high_pool[caliper_mask].copy()
            c_slate_diff = np.abs(candidates["_slate_size"].to_numpy() - low_slate)
            c_time_diff = np.abs(candidates["_time_ns"].to_numpy() - low_time_ns)
            c_ids = candidates["impression_id"].to_numpy()

            # Rank by (slate_diff, time_diff, impression_id)
            # Use stable lexicographic sort using lexsort: keys from last to first
            sort_order = np.lexsort((c_ids, c_time_diff, c_slate_diff))
            best_idx = sort_order[0]
            best_match = candidates.iloc[best_idx]
            best_high_id = best_match["impression_id"]

            matched_pairs.append({
                "low_impression_id": low_id,
                "high_impression_id": best_high_id,
                "low_user_id": low_row["user_id"],
                "high_user_id": best_match["user_id"],
                "low_slate_size": low_slate,
                "high_slate_size": best_match["_slate_size"],
                "slate_size_diff": int(best_match["_slate_size"] - low_slate),
                "low_time": low_row["impression_time"],
                "high_time": best_match["impression_time"],
                "time_diff_seconds": abs(low_time_ns - best_match["_time_ns"]) / 1e9,
                "time_bucket": stratum_key[0],
                "device_type": stratum_key[1],
                "is_subscriber": stratum_key[2],
                "session_stage": stratum_key[3],
                "session_len_bin": stratum_key[4],
                "history_length_bin": stratum_key[5],
            })

            # Remove matched High impression from candidate pool without replacement
            high_pool = high_pool[high_pool["impression_id"] != best_high_id]

    pairs_df = pd.DataFrame(matched_pairs)
    n_pairs = len(pairs_df)
    low_rate = float(n_pairs / n_low) if n_low > 0 else 0.0
    high_rate = float(n_pairs / n_high) if n_high > 0 else 0.0

    return MatchingResults(
        pairs_df=pairs_df,
        eligible_low_count=n_low,
        eligible_high_count=n_high,
        matched_pairs_count=n_pairs,
        low_match_rate=low_rate,
        high_match_rate=high_rate,
        unmatched_low_count=n_low - n_pairs,
        unmatched_high_count=n_high - n_pairs,
    )
