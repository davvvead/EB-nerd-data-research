"""
Pre-period user history profiles, entropy calculation, and breadth cohort assignment.
Derived strictly from 21-day pre-period history with zero behavior-period leakage.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.config import (
    MINIMUM_HISTORY_LENGTH,
    BREADTH_TERTILE_LOW_TO_MED,
    BREADTH_TERTILE_MED_TO_HIGH,
    HISTORY_LENGTH_BINS,
)
from src.data import norm_list


def entropy_from_counts(values: List[Any]) -> float:
    """Calculate Shannon entropy from a collection of categorical observations."""
    s = pd.Series(values).dropna()
    if s.empty:
        return 0.0
    counts = s.value_counts().to_numpy(dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def assign_breadth_cohort(entropy: float) -> str:
    """Assign low, medium, or high breadth cohort using frozen train cutpoints."""
    if np.isnan(entropy):
        return "unknown"
    if entropy <= BREADTH_TERTILE_LOW_TO_MED:
        return "low"
    elif entropy <= BREADTH_TERTILE_MED_TO_HIGH:
        return "medium"
    else:
        return "high"


def assign_history_bin(length: int) -> str:
    """Assign history length bin from frozen quartile boundaries."""
    for idx, (low, high) in enumerate(HISTORY_LENGTH_BINS):
        if low <= length <= high:
            if np.isinf(high):
                return f"{low}+"
            return f"{low}-{high}"
    return "unknown"


class UserProfile:
    """Holds pre-period history attributes for a single user."""
    __slots__ = (
        "user_id", "history_length", "unique_articles", "history_article_ids",
        "category_counts", "category_proportions", "category_entropy",
        "breadth_cohort", "history_bin", "topic_counts"
    )

    def __init__(
        self,
        user_id: Any,
        history_article_ids: List[Any],
        category_counts: Dict[int, int],
        category_entropy: float,
        breadth_cohort: str,
        history_bin: str,
        topic_counts: Dict[Any, int],
    ):
        self.user_id = user_id
        self.history_length = len(history_article_ids)
        self.unique_articles = len(set(history_article_ids))
        self.history_article_ids = history_article_ids
        self.category_counts = category_counts
        total_cats = sum(category_counts.values())
        self.category_proportions = (
            {c: cnt / total_cats for c, cnt in category_counts.items()}
            if total_cats > 0 else {}
        )
        self.category_entropy = category_entropy
        self.breadth_cohort = breadth_cohort
        self.history_bin = history_bin
        self.topic_counts = topic_counts

    def get_category_affinity(self, candidate_category: int) -> float:
        """Return proportion of user history articles in candidate category."""
        return self.category_proportions.get(candidate_category, 0.0)

    def get_topic_affinity(self, candidate_topics: List[Any]) -> float:
        """
        Calculate topic affinity frozen rule:
        Mean over topic t in candidate_topics of:
        (count of user's historical clicked articles containing t) / (total historical articles)
        If candidate has no topics, return 0.0.
        """
        if not candidate_topics or self.history_length == 0:
            return 0.0
        
        matches = [self.topic_counts.get(t, 0) / self.history_length for t in candidate_topics]
        return float(np.mean(matches))


def build_user_profiles(
    history_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    min_history_len: int = MINIMUM_HISTORY_LENGTH,
) -> Dict[Any, UserProfile]:
    """
    Build user history profiles strictly from pre-period history table and articles metadata.
    Filters users to history_length >= min_history_len.
    """
    # Create article lookup maps
    art_meta = articles_df.drop_duplicates("article_id").set_index("article_id")
    art_cat = art_meta["category"].to_dict()
    art_topics = {
        aid: set(norm_list(top))
        for aid, top in art_meta["topics"].items()
    }

    profiles = {}
    for _, row in history_df.iterrows():
        uid = row["user_id"]
        h_ids = norm_list(row["article_id_fixed"])
        h_len = len(h_ids)
        
        if h_len < min_history_len:
            continue
            
        cats = [art_cat[i] for i in h_ids if i in art_cat]
        cat_counts = pd.Series(cats).value_counts().to_dict()
        entropy = entropy_from_counts(cats)
        cohort = assign_breadth_cohort(entropy)
        h_bin = assign_history_bin(h_len)
        
        # Topic counts: number of historical articles containing each topic
        # For each article in history, count each unique topic once per article
        topic_counts: Dict[Any, int] = {}
        for i in h_ids:
            if i in art_topics:
                for t in art_topics[i]:
                    topic_counts[t] = topic_counts.get(t, 0) + 1

        profiles[uid] = UserProfile(
            user_id=uid,
            history_article_ids=h_ids,
            category_counts=cat_counts,
            category_entropy=entropy,
            breadth_cohort=cohort,
            history_bin=h_bin,
            topic_counts=topic_counts,
        )

    return profiles


def export_profiles_to_dataframe(profiles: Dict[Any, UserProfile]) -> pd.DataFrame:
    """Export user profile summaries to a pandas DataFrame."""
    rows = []
    for p in profiles.values():
        rows.append({
            "user_id": p.user_id,
            "history_length": p.history_length,
            "unique_articles": p.unique_articles,
            "category_entropy": p.category_entropy,
            "breadth_cohort": p.breadth_cohort,
            "history_bin": p.history_bin,
        })
    return pd.DataFrame(rows)
