"""
Observable supply indexing and counterfactual analysis eligibility checking.
Observable supply is defined as articles published within a retrospective window:
    t_imp - window <= t_pub < t_imp
Primary window is 24h, with 12h and 48h robustness checks.
Uses exact nanosecond timestamps.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    PRIMARY_SUPPLY_WINDOW_HOURS,
    SUPPLY_SENSITIVITIES_HOURS,
    SUPPLY_RATIO_REQUIREMENT,
    ELIGIBLE_SLATE_SIZE_MIN,
    ELIGIBLE_SLATE_SIZE_MAX,
    MINIMUM_HISTORY_LENGTH,
)
from src.data import epoch_ns

HOURS_TO_NS = 3_600_000_000_000


class ObservableSupplyIndex:
    """
    Fast, binary-searchable index for retrieving observable supply candidates
    satisfying retrospective publication windows strictly before impression time.
    """

    def __init__(self, articles_df: pd.DataFrame):
        """
        Build supply index from articles metadata table.
        Table must contain 'article_id' and 'published_time'.
        """
        df = articles_df.dropna(subset=["published_time"]).copy()
        if "_published_ns" not in df.columns:
            df["_published_ns"] = epoch_ns(df["published_time"])
        
        # Sort strictly by published nanoseconds
        df = df.sort_values("_published_ns").reset_index(drop=True)
        self.article_ids: np.ndarray = df["article_id"].to_numpy(dtype=np.int64)
        self.published_ns: np.ndarray = df["_published_ns"].to_numpy(dtype=np.int64)
        self.n_articles: int = len(self.article_ids)
        
        # Mapping for fast ID to index lookup
        self.id_to_pub_ns: Dict[int, int] = dict(zip(self.article_ids, self.published_ns))

    def get_supply_ids(
        self,
        impression_time_ns: int,
        window_hours: int = PRIMARY_SUPPLY_WINDOW_HOURS,
    ) -> np.ndarray:
        """
        Retrieve array of article IDs published in [t_imp - window, t_imp).
        Strict predicate: t_imp - window <= t_pub < t_imp.
        """
        window_ns = int(window_hours * HOURS_TO_NS)
        start_ns = impression_time_ns - window_ns
        end_ns = impression_time_ns  # strictly less than end_ns via side='left'

        left_idx = int(np.searchsorted(self.published_ns, start_ns, side="left"))
        right_idx = int(np.searchsorted(self.published_ns, end_ns, side="left"))

        if left_idx >= right_idx:
            return np.empty(0, dtype=np.int64)
        return self.article_ids[left_idx:right_idx]

    def get_supply_count(
        self,
        impression_time_ns: int,
        window_hours: int = PRIMARY_SUPPLY_WINDOW_HOURS,
    ) -> int:
        """Count observable supply articles in [t_imp - window, t_imp)."""
        window_ns = int(window_hours * HOURS_TO_NS)
        start_ns = impression_time_ns - window_ns
        end_ns = impression_time_ns
        left_idx = int(np.searchsorted(self.published_ns, start_ns, side="left"))
        right_idx = int(np.searchsorted(self.published_ns, end_ns, side="left"))
        return max(0, right_idx - left_idx)

    def check_counterfactual_eligibility(
        self,
        slate_size: int,
        supply_count: int,
        history_length: int,
        ratio_req: float = SUPPLY_RATIO_REQUIREMENT,
        min_slate: int = ELIGIBLE_SLATE_SIZE_MIN,
        max_slate: int = ELIGIBLE_SLATE_SIZE_MAX,
        min_history: int = MINIMUM_HISTORY_LENGTH,
    ) -> bool:
        """
        Check if an impression satisfies counterfactual analysis eligibility:
        1. Observable supply >= 3 * slate_size
        2. 5 <= slate_size <= 26
        3. history_length >= 15
        """
        if slate_size < min_slate or slate_size > max_slate:
            return False
        if history_length < min_history:
            return False
        if supply_count < int(np.ceil(ratio_req * slate_size)):
            return False
        return True
