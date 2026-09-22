"""
Null slate generation and evaluation.
Generates 200 popularity+recency weighted same-size draws and 200 uniform random
same-size draws without replacement from observable supply.
Computes and caches null expectations once per impression.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import (
    SEED,
    NULL_DRAWS_PER_IMPRESSION,
    RECENCY_HALF_LIFE_HOURS,
    ZERO_CLICK_PSEUDOCOUNT,
)


def compute_item_sampling_weights(
    prior_clicks: np.ndarray,
    age_hours: np.ndarray,
    half_life_hours: float = RECENCY_HALF_LIFE_HOURS,
    pseudocount: float = ZERO_CLICK_PSEUDOCOUNT,
) -> np.ndarray:
    """
    Compute popularity + recency sampling weights:
    w_i = (pseudocount + clicks_24h) * 2 ** (-age_hours / half_life_hours)
    """
    clicks = np.asarray(prior_clicks, dtype=np.float64)
    ages = np.clip(np.asarray(age_hours, dtype=np.float64), 0.0, None)
    decay = 2.0 ** (-ages / half_life_hours)
    weights = (pseudocount + clicks) * decay
    return np.maximum(weights, 1e-12)


def draw_weighted_null_slates(
    candidate_indices: np.ndarray,
    weights: np.ndarray,
    slate_size: int,
    n_draws: int = NULL_DRAWS_PER_IMPRESSION,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw n_draws of size slate_size without replacement from candidate_indices,
    weighted proportionally to weights.
    Uses Gumbel-max trick / exponential keys (Efraimidis & Spirakis 2006).
    Returns array of shape (n_draws, slate_size).
    """
    m = len(candidate_indices)
    if m < slate_size:
        raise ValueError(f"Candidate pool size ({m}) < slate size ({slate_size})")
    
    if rng is None:
        rng = np.random.default_rng(SEED)

    w = np.asarray(weights, dtype=np.float64)
    log_w = np.log(np.maximum(w, 1e-12))

    # Standard Gumbel variates: G = -log(-log(U)) where U ~ Uniform(0, 1)
    # Using exponential variates: E ~ Exp(1) => G = -log(E)
    # Key = log(w) + G = log(w) - log(E)
    exp_variates = rng.exponential(scale=1.0, size=(n_draws, m))
    scores = log_w - np.log(np.maximum(exp_variates, 1e-12))

    # Top K indices per draw
    top_k = np.argpartition(-scores, slate_size - 1, axis=1)[:, :slate_size]
    # Sort descending within each draw for neatness
    row_indices = np.arange(n_draws)[:, None]
    sorted_order = np.argsort(-scores[row_indices, top_k], axis=1)
    chosen_candidate_slots = top_k[row_indices, sorted_order]

    return candidate_indices[chosen_candidate_slots]


def draw_uniform_null_slates(
    candidate_indices: np.ndarray,
    slate_size: int,
    n_draws: int = NULL_DRAWS_PER_IMPRESSION,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw n_draws of size slate_size without replacement uniformly at random.
    Returns array of shape (n_draws, slate_size).
    """
    m = len(candidate_indices)
    if m < slate_size:
        raise ValueError(f"Candidate pool size ({m}) < slate size ({slate_size})")

    if rng is None:
        rng = np.random.default_rng(SEED)

    scores = rng.random(size=(n_draws, m))
    top_k = np.argpartition(-scores, slate_size - 1, axis=1)[:, :slate_size]
    return candidate_indices[top_k]


def compute_null_expectations(
    candidate_discovery: np.ndarray,
    candidate_relevance: np.ndarray,
    candidate_novelty: np.ndarray,
    weights: np.ndarray,
    slate_size: int,
    tau: float,
    n_draws: int = NULL_DRAWS_PER_IMPRESSION,
    rng_seed: int = SEED,
) -> Dict[str, float]:
    """
    Vectorized computation of expected discovery and relevance coverage
    over 200 popularity-weighted and 200 uniform-random null slates.
    
    candidate_discovery: d(u, i) for each supply candidate
    candidate_relevance: r_hat(u, i) for each supply candidate
    candidate_novelty: novelty_k5(u, i) for each supply candidate
    """
    m = len(candidate_discovery)
    if m < slate_size:
        raise ValueError(f"Candidate pool size {m} < slate size {slate_size}")

    cand_indices = np.arange(m, dtype=np.int64)
    rng_pop = np.random.default_rng(rng_seed)
    rng_rand = np.random.default_rng(rng_seed + 1000003)

    # 1. Popularity-weighted draws
    pop_draws = draw_weighted_null_slates(
        cand_indices, weights, slate_size, n_draws=n_draws, rng=rng_pop
    )
    
    # 2. Uniform random draws
    rand_draws = draw_uniform_null_slates(
        cand_indices, slate_size, n_draws=n_draws, rng=rng_rand
    )

    # Pre-extract indicators
    above_floor = (candidate_relevance >= tau).astype(np.float64)

    def _eval_draws(draws: np.ndarray) -> Tuple[float, float, float]:
        # draws shape: (n_draws, slate_size)
        draw_disc = candidate_discovery[draws]  # shape (n_draws, slate_size)
        draw_cov = above_floor[draws]          # shape (n_draws, slate_size)
        draw_nov = candidate_novelty[draws]    # shape (n_draws, slate_size)

        # Discovery per draw: mean item discovery over slate
        mean_disc_per_draw = draw_disc.mean(axis=1)
        # Relevance coverage per draw: share of slate above floor
        mean_cov_per_draw = draw_cov.mean(axis=1)

        # Conditional novelty per draw: mean novelty among above-floor items
        sum_cov_per_draw = draw_cov.sum(axis=1)
        sum_nov_per_draw = (draw_nov * draw_cov).sum(axis=1)
        valid_mask = sum_cov_per_draw > 0
        
        if valid_mask.any():
            cond_nov_per_draw = sum_nov_per_draw[valid_mask] / sum_cov_per_draw[valid_mask]
            mean_cond_nov = float(np.mean(cond_nov_per_draw))
        else:
            mean_cond_nov = float("nan")

        return float(np.mean(mean_disc_per_draw)), float(np.mean(mean_cov_per_draw)), mean_cond_nov

    pop_disc, pop_cov, pop_cond_nov = _eval_draws(pop_draws)
    rand_disc, rand_cov, rand_cond_nov = _eval_draws(rand_draws)

    return {
        "null_discovery_pop": pop_disc,
        "null_relevance_cov_pop": pop_cov,
        "null_cond_novelty_pop": pop_cond_nov,
        "null_discovery_rand": rand_disc,
        "null_relevance_cov_rand": rand_cov,
        "null_cond_novelty_rand": rand_cond_nov,
    }
