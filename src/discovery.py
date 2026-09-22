"""
Discovery metrics, relevance floor gating, and stage decomposition.
Implements:
  d(u, i) = novelty_k5(u, i) if r_hat(u, i) >= tau else 0.0
  D(u, S) = mean item discovery over S
  Relevance coverage and conditional novelty.
  Available -> Exposed comparison (observed vs null).
  Exposed -> Consumed comparison with subscriber/non-subscriber premium choice rule.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def compute_item_discovery(
    novelty: Union[float, np.ndarray],
    predicted_relevance: Union[float, np.ndarray],
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Item-level discovery definition:
    d(u, i) = novelty(u, i) if predicted_relevance(u, i) >= tau else 0.0
    """
    nov = np.asarray(novelty, dtype=np.float64)
    rel = np.asarray(predicted_relevance, dtype=np.float64)
    disc = np.where(rel >= tau, nov, 0.0)
    if disc.ndim == 0:
        return float(disc)
    return disc


def compute_slate_discovery_metrics(
    novelties: np.ndarray,
    predicted_relevances: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    """
    Compute discovery, relevance coverage, and conditional novelty for a slate:
    - discovery: mean item discovery over slate
    - relevance_coverage: fraction of slate with r_hat >= tau
    - conditional_novelty: mean novelty among items with r_hat >= tau (NaN if none)
    - mean_relevance: mean predicted relevance over slate
    """
    if len(novelties) == 0:
        return {
            "discovery": float("nan"),
            "relevance_coverage": float("nan"),
            "conditional_novelty": float("nan"),
            "mean_relevance": float("nan"),
            "n_items": 0,
        }

    nov = np.asarray(novelties, dtype=np.float64)
    rel = np.asarray(predicted_relevances, dtype=np.float64)
    above_floor = rel >= tau
    
    disc = np.where(above_floor, nov, 0.0)
    discovery = float(np.mean(disc))
    coverage = float(np.mean(above_floor))
    mean_rel = float(np.mean(rel))

    if above_floor.any():
        cond_nov = float(np.mean(nov[above_floor]))
    else:
        cond_nov = float("nan")

    return {
        "discovery": discovery,
        "relevance_coverage": coverage,
        "conditional_novelty": cond_nov,
        "mean_relevance": mean_rel,
        "n_items": int(len(nov)),
    }


def compute_choice_stage_metrics(
    slate_article_ids: List[int],
    clicked_article_ids: List[int],
    article_novelty_map: Dict[int, float],
    article_relevance_map: Dict[int, float],
    article_premium_map: Dict[int, bool],
    is_subscriber: bool,
    tau: float,
) -> Dict[str, Union[float, int, bool]]:
    """
    Compute Exposed -> Consumed choice-stage metrics following the frozen protocol:
    For subscribers:
      choosable exposed = all in-view items
      consumed = clicked in-view items
    For non-subscribers:
      exclude premium items from BOTH choosable exposed and consumed clicked set.
    If no eligible consumed item remains:
      returns NaN for choice-stage metrics, with has_eligible_consumed=False.
    """
    clicked_in_slate = [aid for aid in clicked_article_ids if aid in slate_article_ids]

    if is_subscriber:
        choosable_ids = list(slate_article_ids)
        consumed_ids = clicked_in_slate
    else:
        choosable_ids = [aid for aid in slate_article_ids if not article_premium_map.get(aid, False)]
        consumed_ids = [aid for aid in clicked_in_slate if not article_premium_map.get(aid, False)]

    # Compute choosable exposed metrics
    choosable_nov = np.array([article_novelty_map.get(aid, 0.0) for aid in choosable_ids], dtype=np.float64)
    choosable_rel = np.array([article_relevance_map.get(aid, 0.0) for aid in choosable_ids], dtype=np.float64)
    exposed_metrics = compute_slate_discovery_metrics(choosable_nov, choosable_rel, tau)

    if len(consumed_ids) == 0:
        return {
            "has_eligible_consumed": False,
            "n_choosable": len(choosable_ids),
            "n_consumed": 0,
            "exposed_discovery": exposed_metrics["discovery"],
            "exposed_coverage": exposed_metrics["relevance_coverage"],
            "exposed_cond_novelty": exposed_metrics["conditional_novelty"],
            "consumed_discovery": float("nan"),
            "consumed_coverage": float("nan"),
            "consumed_cond_novelty": float("nan"),
            "choice_discovery_gap": float("nan"),
            "choice_coverage_gap": float("nan"),
            "choice_cond_novelty_gap": float("nan"),
        }

    consumed_nov = np.array([article_novelty_map.get(aid, 0.0) for aid in consumed_ids], dtype=np.float64)
    consumed_rel = np.array([article_relevance_map.get(aid, 0.0) for aid in consumed_ids], dtype=np.float64)
    consumed_metrics = compute_slate_discovery_metrics(consumed_nov, consumed_rel, tau)

    disc_gap = consumed_metrics["discovery"] - exposed_metrics["discovery"]
    cov_gap = consumed_metrics["relevance_coverage"] - exposed_metrics["relevance_coverage"]
    
    if np.isfinite(consumed_metrics["conditional_novelty"]) and np.isfinite(exposed_metrics["conditional_novelty"]):
        cond_nov_gap = consumed_metrics["conditional_novelty"] - exposed_metrics["conditional_novelty"]
    else:
        cond_nov_gap = float("nan")

    return {
        "has_eligible_consumed": True,
        "n_choosable": len(choosable_ids),
        "n_consumed": len(consumed_ids),
        "exposed_discovery": exposed_metrics["discovery"],
        "exposed_coverage": exposed_metrics["relevance_coverage"],
        "exposed_cond_novelty": exposed_metrics["conditional_novelty"],
        "consumed_discovery": consumed_metrics["discovery"],
        "consumed_coverage": consumed_metrics["relevance_coverage"],
        "consumed_cond_novelty": consumed_metrics["conditional_novelty"],
        "choice_discovery_gap": disc_gap,
        "choice_coverage_gap": cov_gap,
        "choice_cond_novelty_gap": cond_nov_gap,
    }
