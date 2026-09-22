"""
Validation-side analysis pipeline infrastructure and data processing utilities.
Enforces:
  1. Validation user profiling exclusively from validation/history.parquet using frozen train cutpoints.
  2. Multi-k semantic novelty caching for unique (user_id, article_id) pairs.
  3. Strict time-safe popularity across train and validation boundaries.
  4. Frozen relevance model scoring without retraining.
  5. Validation model qualification evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    ROOT_DIR,
    SEED,
    PRIMARY_HISTORY_K,
    SENSITIVITY_HISTORY_K,
    MINIMUM_HISTORY_LENGTH,
    BREADTH_TERTILE_LOW_TO_MED,
    BREADTH_TERTILE_MED_TO_HIGH,
    HISTORY_LENGTH_BINS,
    MIN_QUALIFICATION_AUC,
    MIN_QUALIFICATION_NDCG_LIFT,
    CACHE_TRAIN_DIR,
    CACHE_VAL_DIR,
    VALIDATION_HISTORY_PATH,
    TRAIN_HISTORY_PATH,
)
from src.data import norm_list, epoch_ns, load_history, load_articles
from src.embeddings import EmbeddingStore
from src.popularity import ClickPopularityIndex
from src.profiles import UserProfile, build_user_profiles
from src.relevance import RelevancePipeline, compute_mean_ndcg

VALIDATION_BEHAVIOR_START_TIME = pd.Timestamp("2023-05-25T07:00:00Z")


def validate_history_source(
    history_path: Path | str,
    behavior_start: pd.Timestamp = VALIDATION_BEHAVIOR_START_TIME,
) -> Dict[str, Any]:
    """
    Fail-fast runtime assertion verifying:
    1. The validation profile source path resolves strictly to ebnerd_small/validation/history.parquet.
    2. Its latest history timestamp strictly precedes held-out validation behavior start.
    3. Its history span is approximately 21 days (between 20.5 and 21.5 days).
    Raises ValueError, AssertionError, or FileNotFoundError if invalid.
    """
    p = Path(history_path).resolve()
    expected = Path(VALIDATION_HISTORY_PATH).resolve()
    train_path = Path(TRAIN_HISTORY_PATH).resolve()

    if p == train_path:
        raise ValueError(
            f"REJECTED: History path '{p}' is TRAIN history. "
            f"Validation user profiles must be derived strictly from '{expected}'."
        )

    if p != expected:
        raise ValueError(
            f"Invalid validation history source path: '{p}'. "
            f"Expected exact path: '{expected}'."
        )

    if not p.exists():
        raise FileNotFoundError(f"Validation history file not found: {p}")

    df = pd.read_parquet(p)
    if "impression_time_fixed" not in df.columns:
        raise KeyError(f"History table missing 'impression_time_fixed': {p}")

    # Collect timestamp values from nested lists
    times = []
    for row in df["impression_time_fixed"]:
        if row is not None and len(row) > 0:
            times.extend(row)

    if not times:
        raise ValueError(f"No timestamps found in history table: {p}")

    times_s = pd.to_datetime(times, utc=True)
    min_t = times_s.min()
    max_t = times_s.max()

    # Assertion 1: Latest timestamp strictly precedes validation behavior start
    if max_t >= behavior_start:
        raise AssertionError(
            f"Validation history latest timestamp ({max_t}) must strictly precede "
            f"validation behavior start ({behavior_start})."
        )

    # Assertion 2: Latest timestamp is adjacent to validation behavior start (gap <= 24h)
    gap_hours = (behavior_start - max_t).total_seconds() / 3600.0
    if gap_hours > 24.0:
        raise AssertionError(
            f"Validation history latest timestamp ({max_t}) is {gap_hours:.1f}h before "
            f"behavior start ({behavior_start}), indicating an obsolete or non-adjacent history file."
        )

    # Assertion 3: History span is approximately 21 days
    span_days = (max_t - min_t).total_seconds() / (24 * 3600.0)
    if not (20.5 <= span_days <= 21.5):
        raise AssertionError(
            f"Validation history span ({span_days:.2f} days) is not approximately 21 days."
        )

    return {
        "path": str(p),
        "min_time": str(min_t),
        "max_time": str(max_t),
        "span_days": float(span_days),
        "gap_seconds": float((behavior_start - max_t).total_seconds()),
        "distinct_users": int(df["user_id"].nunique()),
        "row_count": len(df),
        "verified": True,
    }


def build_validation_user_profiles(
    history_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    min_history_len: int = MINIMUM_HISTORY_LENGTH,
) -> Dict[Any, UserProfile]:
    """
    Construct validation user profiles exclusively from validation history.
    Applies FROZEN train-derived entropy tertile cutpoints and history length bins.
    Does NOT use validation behavior clicks.
    """
    return build_user_profiles(history_df, articles_df, min_history_len=min_history_len)


def load_validation_user_profiles(
    history_path: Path | str = VALIDATION_HISTORY_PATH,
    articles_df: Optional[pd.DataFrame] = None,
    min_history_len: int = MINIMUM_HISTORY_LENGTH,
) -> Dict[Any, UserProfile]:
    """
    Load validation user profiles with fail-fast source path verification.
    """
    validate_history_source(history_path)
    hist_df = load_history(history_path)
    if articles_df is None:
        articles_df = load_articles()
    return build_validation_user_profiles(hist_df, articles_df, min_history_len=min_history_len)


def build_unified_time_safe_popularity_index(
    train_behaviors_df: pd.DataFrame,
    validation_behaviors_df: pd.DataFrame,
) -> ClickPopularityIndex:
    """
    Build unified click popularity index containing train + validation clicks.
    The underlying query method ClickPopularityIndex.get_prior_clicks() enforces
    the strict retrospective temporal predicate:
        t_imp - 24h <= t_click < t_imp
    ensuring that same-time and future clicks are NEVER included in popularity calculations.
    """
    combined_behaviors = pd.concat([train_behaviors_df, validation_behaviors_df], ignore_index=True)
    return ClickPopularityIndex.from_behaviors(combined_behaviors)


def precompute_novelty_cache(
    user_ids_and_candidates: Dict[int, Set[int]],
    profiles: Dict[int, UserProfile],
    embedding_store: EmbeddingStore,
    k_values: Tuple[int, ...] = (3, 5, 10),
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute user_history x candidate_article semantic novelty once across all needed pairs.
    Returns DataFrame with columns: user_id, article_id, novelty_k3, novelty_k5, novelty_k10.
    Optionally saves to cache/validation/user_article_novelty.parquet.
    """
    records = []
    for uid, cand_set in user_ids_and_candidates.items():
        if uid not in profiles:
            continue
        u_prof = profiles[uid]
        c_list = list(cand_set)
        if not c_list:
            continue

        nov_multi = embedding_store.compute_novelty_multi_k(
            history_ids=u_prof.history_article_ids,
            candidate_ids=c_list,
            k_values=k_values,
        )

        for idx, aid in enumerate(c_list):
            rec = {
                "user_id": uid,
                "article_id": aid,
            }
            for k in k_values:
                rec[f"novelty_k{k}"] = float(nov_multi[k][idx])
            records.append(rec)

    df_cache = pd.DataFrame(records)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_cache.to_parquet(output_path, index=False)

    return df_cache


def score_candidate_articles_batch(
    user_id: int,
    candidate_ids: List[int],
    impression_time_ns: int,
    impression_time_dt: pd.Timestamp,
    device_type: Any,
    is_subscriber: bool,
    profile: UserProfile,
    articles_df: pd.DataFrame,
    popularity_index: ClickPopularityIndex,
    novelty_k5_map: Dict[Tuple[int, int], float],
    relevance_pipeline: RelevancePipeline,
) -> np.ndarray:
    """
    Generate predicted relevance r_hat(u, i) for a list of candidate articles
    under a specific impression context using the frozen relevance pipeline.
    """
    if not candidate_ids:
        return np.empty(0, dtype=np.float64)

    art_meta = articles_df.drop_duplicates("article_id").set_index("article_id")
    art_pub_ns = art_meta["_published_ns"].to_dict()
    art_cat = art_meta["category"].to_dict()
    art_prem = art_meta["premium"].to_dict()
    art_topics = {
        aid: list(set(norm_list(top)))
        for aid, top in art_meta["topics"].items()
    }

    prior_clicks = popularity_index.get_prior_clicks_batch(candidate_ids, impression_time_ns)

    hour = impression_time_dt.hour
    tod_sin = float(np.sin(2 * np.pi * hour / 24.0))
    tod_cos = float(np.cos(2 * np.pi * hour / 24.0))
    sub_int = 1 if is_subscriber else 0

    h_len = max(1, profile.history_length)
    rows = []

    for aid in candidate_ids:
        if aid not in art_pub_ns:
            # Fallback row with default values
            rows.append({
                "sim_to_history": 0.0,
                "category_affinity": 0.0,
                "topic_affinity": 0.0,
                "log_age_hours": 0.0,
                "log_clicks_24h": 0.0,
                "is_premium": 0,
                "is_subscriber": sub_int,
                "premium_x_subscriber": 0,
                "time_of_day_sin": tod_sin,
                "time_of_day_cos": tod_cos,
                "device_type": str(device_type),
            })
            continue

        nov_5 = novelty_k5_map.get((user_id, aid), 0.5)
        sim = 1.0 - nov_5

        cat = art_cat.get(aid, -1)
        cat_aff = float(profile.category_counts.get(cat, 0) / h_len)

        tops = art_topics.get(aid, [])
        top_aff = float(np.mean([profile.topic_counts.get(t, 0) for t in tops]) / h_len) if tops else 0.0

        pub_ns = art_pub_ns[aid]
        age_hours = max(0.0, (impression_time_ns - pub_ns) / 3.6e12)
        log_age = float(np.log1p(age_hours))

        clicks = prior_clicks.get(aid, 0)
        log_clicks = float(np.log1p(clicks))

        prem = 1 if art_prem.get(aid, False) else 0

        rows.append({
            "sim_to_history": sim,
            "category_affinity": cat_aff,
            "topic_affinity": top_aff,
            "log_age_hours": log_age,
            "log_clicks_24h": log_clicks,
            "is_premium": prem,
            "is_subscriber": sub_int,
            "premium_x_subscriber": prem * sub_int,
            "time_of_day_sin": tod_sin,
            "time_of_day_cos": tod_cos,
            "device_type": str(device_type),
        })

    feat_df = pd.DataFrame(rows)
    return relevance_pipeline.predict_proba(feat_df)


@dataclass(frozen=True)
class QualificationResult:
    """Outcome of validation relevance model qualification check."""
    auc: float
    ndcg: float
    baseline_ndcg: float
    ndcg_lift: float
    is_qualified: bool
    status_label: str  # "QUALIFIED" or "NOT_QUALIFIED"
    total_examples: int
    positive_count: int


def evaluate_relevance_qualification(
    impression_ids: np.ndarray,
    y_true: np.ndarray,
    y_score_model: np.ndarray,
    y_score_baseline: np.ndarray,
    min_auc: float = MIN_QUALIFICATION_AUC,
    min_lift: float = MIN_QUALIFICATION_NDCG_LIFT,
) -> QualificationResult:
    """
    Evaluate validation relevance model against frozen qualification rule:
      AUC >= 0.60 AND NDCG >= baseline_NDCG + 0.01
    """
    auc = float(roc_auc_score(y_true, y_score_model))
    ndcg = float(compute_mean_ndcg(impression_ids, y_true, y_score_model))
    base_ndcg = float(compute_mean_ndcg(impression_ids, y_true, y_score_baseline))
    lift = ndcg - base_ndcg

    qualified = bool(auc >= min_auc and lift >= min_lift)
    status_label = "QUALIFIED" if qualified else "NOT_QUALIFIED"

    return QualificationResult(
        auc=auc,
        ndcg=ndcg,
        baseline_ndcg=base_ndcg,
        ndcg_lift=lift,
        is_qualified=qualified,
        status_label=status_label,
        total_examples=len(y_true),
        positive_count=int(y_true.sum()),
    )
