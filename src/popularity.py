"""
Time-safe click popularity index and recency decay weighting.
Strictly ensures zero future leakage and enforces the frozen protocol formula.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    POPULARITY_WINDOW_HOURS,
    RECENCY_HALF_LIFE_HOURS,
    ZERO_CLICK_PSEUDOCOUNT,
)
from src.data import epoch_ns, norm_list


class ClickPopularityIndex:
    """
    Time-safe click popularity index backed by sorted nanosecond timestamp arrays.
    Guarantees lookups only count clicks strictly prior to impression time: t_click < t_imp.
    """
    def __init__(self, click_arrays: Dict[Any, np.ndarray]):
        self.clicks = click_arrays  # article_id -> sorted np.ndarray of int64 ns timestamps

    @classmethod
    def from_behaviors(cls, behaviors_df: pd.DataFrame) -> ClickPopularityIndex:
        """Construct the index from all clicks in a behaviors DataFrame."""
        times = epoch_ns(behaviors_df["impression_time"])
        clicked_col = behaviors_df["article_ids_clicked"].map(norm_list).to_list()
        
        # Flatten clicks
        records = []
        for t_ns, c_list in zip(times, clicked_col):
            if not np.isnan(t_ns) and c_list:
                for aid in c_list:
                    records.append((aid, t_ns))
                    
        if not records:
            return cls({})

        # Group and sort by article
        records.sort(key=lambda x: (x[0], x[1]))
        grouped: Dict[Any, List[int]] = {}
        for aid, t_ns in records:
            if aid not in grouped:
                grouped[aid] = []
            grouped[aid].append(t_ns)
            
        click_arrays = {
            aid: np.array(ts, dtype=np.int64)
            for aid, ts in grouped.items()
        }
        return cls(click_arrays)

    def get_prior_clicks(
        self,
        article_id: Any,
        imp_time_ns: int,
        window_hours: float = POPULARITY_WINDOW_HOURS,
    ) -> int:
        """Count clicks for article_id in [imp_time_ns - window, imp_time_ns)."""
        arr = self.clicks.get(article_id)
        if arr is None or len(arr) == 0:
            return 0
        
        delta_ns = int(window_hours * 3600 * 1e9)
        # Strict inequality: t_click < imp_time_ns is enforced by side='left' at imp_time_ns
        right = int(np.searchsorted(arr, imp_time_ns, side="left"))
        left = int(np.searchsorted(arr, imp_time_ns - delta_ns, side="left"))
        return max(0, right - left)

    def get_prior_clicks_batch(
        self,
        article_ids: Union[List[Any], np.ndarray],
        imp_time_ns: int,
        window_hours: float = POPULARITY_WINDOW_HOURS,
    ) -> np.ndarray:
        """Vectorized click lookups for a collection of article IDs at a single timestamp."""
        out = np.zeros(len(article_ids), dtype=np.int32)
        delta_ns = int(window_hours * 3600 * 1e9)
        t_start = imp_time_ns - delta_ns

        for idx, aid in enumerate(article_ids):
            arr = self.clicks.get(aid)
            if arr is not None and len(arr) > 0:
                right = np.searchsorted(arr, imp_time_ns, side="left")
                left = np.searchsorted(arr, t_start, side="left")
                out[idx] = max(0, right - left)
        return out

    def compute_null_weights(
        self,
        article_ids: Union[List[Any], np.ndarray],
        published_ns: Union[List[int], np.ndarray],
        imp_time_ns: int,
        window_hours: float = POPULARITY_WINDOW_HOURS,
        half_life_hours: float = RECENCY_HALF_LIFE_HOURS,
    ) -> np.ndarray:
        """
        Compute primary null weights using frozen formula:
        (1 + clicks_in_prior_24h) * 2^(-article_age_hours / 12)
        Zero-click articles receive pseudocount 1 before recency decay.
        """
        clicks = self.get_prior_clicks_batch(article_ids, imp_time_ns, window_hours=window_hours)
        pub_arr = np.asarray(published_ns, dtype=np.int64)
        
        # Age in hours (must be >= 0)
        age_hours = np.maximum(0.0, (imp_time_ns - pub_arr) / (3600.0 * 1e9))
        
        # Weight formula
        weights = (1.0 + clicks.astype(np.float64)) * (2.0 ** (-age_hours / half_life_hours))
        return weights

    def save(self, path: Path | str) -> None:
        """Save click arrays to NPZ file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store as string keys
        str_dict = {str(k): v for k, v in self.clicks.items()}
        np.savez_compressed(path, **str_dict)

    @classmethod
    def load(cls, path: Path | str) -> ClickPopularityIndex:
        """Load click arrays from NPZ file."""
        path = Path(path)
        with np.load(path) as data:
            # Detect whether keys should be converted to int
            click_arrays = {}
            for k in data.files:
                try:
                    aid = int(k)
                except ValueError:
                    aid = k
                click_arrays[aid] = data[k]
        return cls(click_arrays)
