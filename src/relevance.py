"""
Relevance model dataset construction, forward-chained out-of-fold training,
relevance floor (tau) derivation, and qualification evaluation.
Uses fold-local preprocessing and Logistic Regression strictly matching frozen specifications.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    ROOT_DIR,
    SEED,
    PRIMARY_HISTORY_K,
    MIN_QUALIFICATION_AUC,
    MIN_QUALIFICATION_NDCG_LIFT,
    CACHE_TRAIN_DIR,
)
from src.data import norm_list
from src.embeddings import EmbeddingStore
from src.profiles import UserProfile
from src.popularity import ClickPopularityIndex


NUMERIC_FEATURES = [
    "sim_to_history",
    "category_affinity",
    "topic_affinity",
    "log_age_hours",
    "log_clicks_24h",
    "is_premium",
    "is_subscriber",
    "premium_x_subscriber",
    "time_of_day_sin",
    "time_of_day_cos",
]
CATEGORICAL_FEATURES = ["device_type"]


def compute_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate nDCG for a single slate.
    If no positives exist in y_true, return NaN.
    """
    if y_true.sum() == 0:
        return np.nan
    
    order = np.argsort(-y_score)
    y_ranked = y_true[order]
    
    ranks = np.arange(1, len(y_ranked) + 1)
    discounts = 1.0 / np.log2(ranks + 1.0)
    dcg = (y_ranked * discounts).sum()
    
    # Ideal DCG
    ideal_y = np.sort(y_true)[::-1]
    idcg = (ideal_y * discounts).sum()
    
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def compute_mean_ndcg(
    impression_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Calculate mean nDCG across impressions having at least one click."""
    df = pd.DataFrame({
        "imp_id": impression_ids,
        "y": y_true,
        "score": y_score,
    })
    
    # Filter to impressions with at least one click
    clicked_imps = df.groupby("imp_id")["y"].sum()
    valid_imps = clicked_imps[clicked_imps > 0].index
    if len(valid_imps) == 0:
        return 0.0
        
    df_valid = df[df["imp_id"].isin(valid_imps)]
    ndcgs = []
    for _, g in df_valid.groupby("imp_id", sort=False):
        ndcg = compute_ndcg_at_k(g["y"].to_numpy(), g["score"].to_numpy())
        if not np.isnan(ndcg):
            ndcgs.append(ndcg)
            
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def extract_relevance_examples(
    frontpage_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    profiles: Dict[Any, UserProfile],
    popularity_index: ClickPopularityIndex,
    embedding_store: EmbeddingStore,
    k_sim: int = PRIMARY_HISTORY_K,
) -> pd.DataFrame:
    """
    Construct the tabular relevance training/evaluation dataset from front-page impressions.
    Optimized by caching semantic similarity on unique (user_id, article_id) pairs.
    """
    art_meta = articles_df.drop_duplicates("article_id").set_index("article_id")
    art_pub_ns = art_meta["_published_ns"].to_dict()
    art_cat = art_meta["category"].to_dict()
    art_prem = art_meta["premium"].to_dict()
    art_topics = {
        aid: list(set(norm_list(top)))
        for aid, top in art_meta["topics"].items()
    }

    # Filter impressions where user has profile
    valid_mask = frontpage_df["user_id"].isin(profiles)
    fp = frontpage_df[valid_mask].copy()

    # Determine unique (user_id, article_id) pairs needed for similarity
    unique_pairs: Dict[Any, Set[Any]] = {}
    for uid, slate in zip(fp["user_id"], fp["_slate"]):
        if uid not in unique_pairs:
            unique_pairs[uid] = set()
        unique_pairs[uid].update(slate)

    # Precompute semantic similarities per unique user
    user_art_sim: Dict[Tuple[Any, Any], float] = {}
    for uid, c_set in unique_pairs.items():
        u_prof = profiles[uid]
        c_list = list(c_set)
        nov_dict = embedding_store.compute_novelty_multi_k(
            history_ids=u_prof.history_article_ids,
            candidate_ids=c_list,
            k_values=(k_sim,),
        )
        # Novelty = 1.0 - sim => sim = 1.0 - novelty
        sims = 1.0 - nov_dict[k_sim]
        for aid, sim_val in zip(c_list, sims):
            user_art_sim[(uid, aid)] = float(sim_val) if np.isfinite(sim_val) else 0.0

    # Build row records using fast column zips
    rows = []
    imp_ids = fp["impression_id"].to_list()
    uids = fp["user_id"].to_list()
    times_ns = fp["_time_ns"].to_numpy()
    imp_times = fp["impression_time"].to_list()
    devs = fp["device_type"].to_list()
    subs = fp["is_subscriber"].astype(int).to_list()
    slates = fp["_slate"].to_list()
    clicked_lists = [norm_list(c) for c in fp["article_ids_clicked"]]

    for i in range(len(fp)):
        imp_id = imp_ids[i]
        uid = uids[i]
        t_ns = times_ns[i]
        imp_time = imp_times[i]
        dev = devs[i]
        is_sub = subs[i]
        slate = slates[i]
        clicked_set = set(slate).intersection(clicked_lists[i])
        u_prof = profiles[uid]

        hour = imp_time.hour
        tod_sin = float(np.sin(2 * np.pi * hour / 24.0))
        tod_cos = float(np.cos(2 * np.pi * hour / 24.0))

        # Prior clicks for all items in slate
        prior_clicks = popularity_index.get_prior_clicks_batch(slate, t_ns)

        for idx, aid in enumerate(slate):
            if aid not in art_pub_ns:
                continue

            y = 1 if aid in clicked_set else 0
            sim = user_art_sim.get((uid, aid), 0.0)
            cat = art_cat.get(aid, -1)
            cat_aff = u_prof.get_category_affinity(cat)
            top_aff = u_prof.get_topic_affinity(art_topics.get(aid, []))

            pub_ns = art_pub_ns[aid]
            age_h = max(0.0, (t_ns - pub_ns) / (3600.0 * 1e9))
            log_age = float(np.log1p(age_h))
            log_clicks = float(np.log1p(prior_clicks[idx]))

            prem = int(art_prem.get(aid, False))
            prem_sub = prem * is_sub

            # Null popularity weight for baseline nDCG calculation
            pop_rec_weight = (1.0 + prior_clicks[idx]) * (2.0 ** (-age_h / 12.0))

            rows.append({
                "impression_id": imp_id,
                "user_id": uid,
                "article_id": aid,
                "impression_time": imp_time,
                "_time_ns": t_ns,
                "y": y,
                "sim_to_history": sim,
                "category_affinity": cat_aff,
                "topic_affinity": top_aff,
                "log_age_hours": log_age,
                "log_clicks_24h": log_clicks,
                "is_premium": prem,
                "is_subscriber": is_sub,
                "premium_x_subscriber": prem_sub,
                "device_type": str(dev),
                "time_of_day_sin": tod_sin,
                "time_of_day_cos": tod_cos,
                "pop_rec_weight": pop_rec_weight,
            })

    return pd.DataFrame(rows)


def make_forward_chain_blocks(df: pd.DataFrame, n_blocks: int = 5) -> List[pd.DataFrame]:
    """
    Split impressions into n_blocks forward-chaining blocks.
    Strict rule: Forward-chain block boundaries must never split impressions
    sharing the same impression_time. Tied timestamps are placed in the later block.
    Guarantees max(training impression_time) < min(scored impression_time).
    """
    # Unique impressions sorted by time
    imp_meta = df[["impression_id", "_time_ns"]].drop_duplicates("impression_id").sort_values("_time_ns")
    n_total = len(imp_meta)
    target_size = n_total // n_blocks

    unique_times = imp_meta["_time_ns"].to_numpy()
    split_indices = []
    
    # Find boundary indices avoiding timestamp ties
    for b in range(1, n_blocks):
        nominal_idx = b * target_size
        nominal_time = unique_times[nominal_idx]
        # Find where this timestamp ends (all rows with this timestamp go to current/later block)
        # To ensure max(earlier) < min(later), boundary must be at the first instance of nominal_time
        boundary_idx = int(np.searchsorted(unique_times, nominal_time, side="left"))
        # If boundary_idx is 0 or matches previous, advance to right side
        if boundary_idx in split_indices or boundary_idx == 0:
            boundary_idx = int(np.searchsorted(unique_times, nominal_time, side="right"))
        split_indices.append(boundary_idx)

    # Segment unique impression IDs into blocks
    split_indices = [0] + split_indices + [n_total]
    blocks = []
    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        block_imp_ids = set(imp_meta.iloc[start:end]["impression_id"])
        block_df = df[df["impression_id"].isin(block_imp_ids)].copy()
        blocks.append(block_df)

    # Strict temporal boundary assertion
    for i in range(len(blocks) - 1):
        max_prev = blocks[i]["_time_ns"].max()
        min_next = blocks[i + 1]["_time_ns"].min()
        if not (max_prev < min_next):
            raise AssertionError(
                f"Temporal tie-break violated between block {i+1} (max {max_prev}) and block {i+2} (min {min_next})"
            )

    return blocks


class RelevancePipeline:
    """Encapsulates fold-local preprocessing and Logistic Regression model."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            random_state=SEED,
        )
        self.fitted_ = False

    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        num_vals = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        cat_vals = df[CATEGORICAL_FEATURES].astype(str)

        if fit:
            num_scaled = self.scaler.fit_transform(num_vals)
            cat_encoded = self.encoder.fit_transform(cat_vals)
        else:
            num_scaled = self.scaler.transform(num_vals)
            cat_encoded = self.encoder.transform(cat_vals)

        return np.hstack([num_scaled, cat_encoded])

    def fit(self, df: pd.DataFrame) -> RelevancePipeline:
        X = self._prepare_X(df, fit=True)
        y = df["y"].to_numpy(dtype=np.int32)
        self.model.fit(X, y)
        self.fitted_ = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Pipeline must be fitted before predict_proba")
        X = self._prepare_X(df, fit=False)
        # Return probability of positive class (click)
        return self.model.predict_proba(X)[:, 1]


def run_forward_chained_oof(blocks: List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
    """
    Run forward-chained out-of-fold relevance training on blocks B1..B5:
    B1 -> predict B2
    B1+B2 -> predict B3
    B1+B2+B3 -> predict B4
    B1+B2+B3+B4 -> predict B5
    Calculates OOF AUC, nDCG, Brier score, and freezes tau on B2..B5.
    """
    oof_dfs = []
    
    for fold in range(1, len(blocks)):
        train_blocks = blocks[:fold]
        train_df = pd.concat(train_blocks, ignore_index=True)
        eval_df = blocks[fold].copy()

        # Fit fold-local preprocessing + model strictly on training blocks
        pipe = RelevancePipeline()
        pipe.fit(train_df)

        # Predict on eval block
        scores = pipe.predict_proba(eval_df)
        eval_df["oof_score"] = scores
        eval_df["oof_fold"] = fold + 1
        oof_dfs.append(eval_df)

    pooled_oof = pd.concat(oof_dfs, ignore_index=True)

    # Diagnostic metrics on B2..B5
    y_true = pooled_oof["y"].to_numpy()
    y_score = pooled_oof["oof_score"].to_numpy()
    oof_auc = float(roc_auc_score(y_true, y_score))
    oof_brier = float(brier_score_loss(y_true, y_score))
    oof_ndcg = compute_mean_ndcg(pooled_oof["impression_id"].to_numpy(), y_true, y_score)

    # Baseline nDCG using popularity+recency weight
    baseline_ndcg = compute_mean_ndcg(
        pooled_oof["impression_id"].to_numpy(),
        y_true,
        pooled_oof["pop_rec_weight"].to_numpy(),
    )

    # Freeze tau: 20th percentile of OOF predicted relevance among clicked items (y=1)
    clicked_scores = pooled_oof.loc[pooled_oof["y"] == 1, "oof_score"].to_numpy()
    if len(clicked_scores) == 0:
        raise RuntimeError("No clicked items found in out-of-fold training set to compute tau")
    tau = float(np.percentile(clicked_scores, 20.0))

    diagnostics = {
        "oof_auc": oof_auc,
        "oof_ndcg": oof_ndcg,
        "oof_brier": oof_brier,
        "baseline_ndcg": baseline_ndcg,
        "ndcg_lift_over_baseline": oof_ndcg - baseline_ndcg,
        "tau_20th_percentile": tau,
        "total_oof_examples": len(pooled_oof),
        "clicked_oof_examples": int(len(clicked_scores)),
        "unclicked_oof_examples": int(len(y_true) - len(clicked_scores)),
        "clicked_score_distribution": {
            "min": float(np.min(clicked_scores)),
            "p20_tau": tau,
            "median": float(np.median(clicked_scores)),
            "mean": float(np.mean(clicked_scores)),
            "max": float(np.max(clicked_scores)),
        },
    }

    return pooled_oof, diagnostics, tau
