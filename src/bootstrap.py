"""
User-clustered bootstrap and standardized effect size estimation.
Aggregates impression-level metrics to user-level first, then resamples
users with replacement across 2,000 replicates.
Calculates 95% percentile confidence intervals and between-user SD standardization.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    PRACTICAL_EFFECT_THRESHOLD_SD,
    SEED,
)


@dataclass(frozen=True)
class BootstrapEstimate:
    """Summary of a user-clustered bootstrap estimation."""
    metric_name: str
    n_users: int
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    between_user_sd: float
    standardized_effect: float
    practical_significance: bool  # |std_effect| >= 0.10 SD
    replicate_mean: float


def user_clustered_bootstrap(
    df: pd.DataFrame,
    metric_col: str,
    user_col: str = "user_id",
    n_replicates: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    practical_threshold_sd: float = PRACTICAL_EFFECT_THRESHOLD_SD,
    seed: int = SEED,
) -> BootstrapEstimate:
    """
    Perform user-clustered bootstrap for a single metric.
    
    1. Aggregates metric by user first (mean across impressions).
    2. Resamples users with replacement n_replicates times.
    3. Computes 95% percentile CI and between-user SD standardization.
    """
    # Drop rows where metric is NaN
    valid_df = df.dropna(subset=[metric_col]).copy()
    if valid_df.empty:
        return BootstrapEstimate(
            metric_name=metric_col,
            n_users=0,
            point_estimate=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            std_error=float("nan"),
            between_user_sd=float("nan"),
            standardized_effect=float("nan"),
            practical_significance=False,
            replicate_mean=float("nan"),
        )

    # Step 1: User-level aggregation
    user_means = valid_df.groupby(user_col)[metric_col].mean().to_numpy(dtype=np.float64)
    n_users = len(user_means)
    
    point_estimate = float(np.mean(user_means))
    between_user_sd = float(np.std(user_means, ddof=1)) if n_users > 1 else 0.0

    if n_users == 1 or between_user_sd == 0.0:
        return BootstrapEstimate(
            metric_name=metric_col,
            n_users=n_users,
            point_estimate=point_estimate,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            std_error=0.0,
            between_user_sd=between_user_sd,
            standardized_effect=0.0,
            practical_significance=False,
            replicate_mean=point_estimate,
        )

    # Step 2: Vectorized bootstrap resampling
    rng = np.random.default_rng(seed)
    # Sample matrix of indices: (n_replicates, n_users)
    boot_indices = rng.integers(0, n_users, size=(n_replicates, n_users))
    boot_means = np.mean(user_means[boot_indices], axis=1)

    # Step 3: Percentile confidence interval
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    std_error = float(np.std(boot_means, ddof=1))
    replicate_mean = float(np.mean(boot_means))

    # Step 4: Standardized effect size
    std_effect = float(point_estimate / between_user_sd) if between_user_sd > 0 else 0.0
    practical = bool(abs(std_effect) >= practical_threshold_sd)

    return BootstrapEstimate(
        metric_name=metric_col,
        n_users=n_users,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        between_user_sd=between_user_sd,
        standardized_effect=std_effect,
        practical_significance=practical,
        replicate_mean=replicate_mean,
    )


def bootstrap_matched_pairs(
    pairs_df: pd.DataFrame,
    low_metric_col: str,
    high_metric_col: str,
    cluster_col: str = "low_user_id",
    n_replicates: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    practical_threshold_sd: float = PRACTICAL_EFFECT_THRESHOLD_SD,
    seed: int = SEED,
) -> BootstrapEstimate:
    """
    Perform cluster-robust bootstrap for matched pair differences:
      diff = high_metric - low_metric
    Clustered on user level (default low_user_id).
    """
    df = pairs_df.copy()
    df["_pair_diff"] = df[high_metric_col] - df[low_metric_col]
    return user_clustered_bootstrap(
        df=df,
        metric_col="_pair_diff",
        user_col=cluster_col,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        practical_threshold_sd=practical_threshold_sd,
        seed=seed,
    )
