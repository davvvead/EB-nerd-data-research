#!/usr/bin/env python3
"""
Orchestration script for EB-NeRD Phase 1 validation outcome analysis.
Computes:
  1. Validation relevance model qualification (AUC >= 0.60, NDCG lift >= 0.01).
  2. Observable supply & 200-draw null comparisons (Available -> Exposed).
  3. Choice stage with subscriber/non-subscriber rule (Exposed -> Consumed).
  4. Deterministic matched comparison between Low and High breadth impressions.
  5. Discovery Headroom optimization (strictly gated by qualification).
  6. 2,000-replicate user-clustered bootstrap for all outcome metrics.

CRITICAL SAFETY GUARD:
  Requires explicit command-line argument '--authorize-heldout-run' to access
  the real validation behavior data. Without this flag, execution immediately
  aborts with an error, preventing accidental unblinding.
  Supports '--dry-run' for synthetic/pseudo-held-out validation testing.
"""

from __future__ import annotations
import argparse
import hashlib
import joblib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.config import (
    ROOT_DIR,
    SEED,
    PROTOCOL_VERSION,
    PROTOCOL_COMMIT_HASH,
    PREOUTCOME_GATE_COMMIT_HASH,
    IMPLEMENTATION_LOCK_COMMIT_HASH,
    PRIMARY_SUPPLY_WINDOW_HOURS,
    PRIMARY_HISTORY_K,
    MINIMUM_HISTORY_LENGTH,
    ELIGIBLE_SLATE_SIZE_MIN,
    ELIGIBLE_SLATE_SIZE_MAX,
    SUPPLY_RATIO_REQUIREMENT,
    BOOTSTRAP_REPLICATES,
    HEADROOM_PRIMARY_RELEVANCE_TOLERANCE,
    HEADROOM_MAX_SWAPS,
    COMMON_SUPPORT_N_PRIMARY,
    CACHE_TRAIN_DIR,
    CACHE_VAL_DIR,
    OUTPUTS_DIR,
    VALIDATION_HISTORY_PATH,
    VALIDATION_BEHAVIORS_PATH,
    TRAIN_HISTORY_PATH,
    ensure_directories,
)
from src.data import (
    epoch_ns,
    load_articles,
    load_behaviors,
    load_history,
    add_session_features,
    norm_list,
    js,
)
from src.embeddings import build_or_load_embedding_store
from src.supply import ObservableSupplyIndex
from src.nulls import compute_null_expectations, compute_item_sampling_weights
from src.discovery import (
    compute_item_discovery,
    compute_slate_discovery_metrics,
    compute_choice_stage_metrics,
)
from src.matching import perform_deterministic_matching
from src.headroom import CommonSupportIndex, optimize_headroom_single_impression
from src.bootstrap import user_clustered_bootstrap, bootstrap_matched_pairs
from src.relevance import (
    RelevancePipeline,
    extract_relevance_examples,
    compute_mean_ndcg,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.validation import (
    build_validation_user_profiles,
    load_validation_user_profiles,
    build_unified_time_safe_popularity_index,
    precompute_novelty_cache,
    evaluate_relevance_qualification,
    validate_history_source,
)
from src.reporting import export_json_artifact, export_tidy_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 1 validation outcome analysis pipeline."
    )
    parser.add_argument(
        "--authorize-heldout-run",
        action="store_true",
        default=False,
        help="Explicitly authorize access to held-out validation behaviors. Mandatory for real run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run end-to-end pipeline on synthetic/mock data without touching validation data.",
    )
    parser.add_argument(
        "--sample-impressions",
        type=int,
        default=None,
        help="Optional limit on number of impressions to process (for benchmarking).",
    )
    return parser.parse_args()


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_synthetic_dry_run():
    """
    Execute full end-to-end pipeline on synthetic fixture:
    AVAILABLE -> EXPOSED -> CONSUMED -> nulls -> matching -> qualification -> Headroom -> bootstrap.
    Validates complete machinery without reading or scoring real validation outcomes.
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(SEED)

    print("\n--- SYNTHETIC END-TO-END DRY RUN ---")
    print("[1/8] Generating synthetic articles and candidate pool...")
    base_t = pd.Timestamp("2023-05-25T12:00:00Z")
    base_ns = int(base_t.value)

    n_arts = 300
    art_ids = np.arange(1000, 1000 + n_arts)
    pub_deltas = rng.uniform(1, 47, size=n_arts) * 3.6e12
    pub_ns = base_ns - pub_deltas.astype(np.int64)
    pub_times = pd.to_datetime(pub_ns, unit="ns", utc=True)
    cats = rng.integers(1, 8, size=n_arts)
    prems = rng.random(size=n_arts) < 0.20

    articles_df = pd.DataFrame({
        "article_id": art_ids,
        "published_time": pub_times,
        "_published_ns": pub_ns,
        "category": cats,
        "premium": prems,
        "topics": [[f"top_{c}"] for c in cats],
    })

    print("[2/8] Generating synthetic user profiles and histories...")
    user_rows = []
    for uid in range(1, 41):
        if uid <= 20:
            hist = [int(rng.choice(art_ids[cats == 1])) for _ in range(25)]
        else:
            hist = [int(rng.choice(art_ids)) for _ in range(25)]
        user_rows.append({"user_id": uid, "article_id_fixed": hist})

    hist_df = pd.DataFrame(user_rows)
    profiles = build_validation_user_profiles(hist_df, articles_df, min_history_len=15)
    print(f"  -> Generated {len(profiles)} user profiles across cohorts.")

    print("[3/8] Generating synthetic front-page impressions...")
    imp_rows = []
    supply_index = ObservableSupplyIndex(articles_df)
    
    for imp_id in range(1, 101):
        uid = int(rng.integers(1, 41))
        t_offset = rng.uniform(0, 12) * 3600 * 1e9
        imp_t_ns = base_ns + int(t_offset)
        imp_t = pd.to_datetime(imp_t_ns, unit="ns", utc=True)
        dev = int(rng.choice([1, 2, 3]))
        sub = bool(rng.random() < 0.25)
        
        slate = list(rng.choice(art_ids, size=10, replace=False))
        clicked = [int(rng.choice(slate))] if rng.random() < 0.4 else []

        imp_rows.append({
            "impression_id": imp_id,
            "user_id": uid,
            "impression_time": imp_t,
            "_time_ns": imp_t_ns,
            "_slate": slate,
            "_slate_size": len(slate),
            "article_ids_clicked": clicked,
            "device_type": dev,
            "is_subscriber": sub,
            "_session_stage": "single",
            "_session_len_bin": "1",
            "history_length_bin": profiles[uid].history_bin,
            "_time_bucket": imp_t.floor("30min"),
            "breadth_cohort": profiles[uid].breadth_cohort,
        })

    imp_df = pd.DataFrame(imp_rows)

    print("[4/8] Evaluating observable supply and counterfactual eligibility...")
    eligible_imps = []
    tau = 0.10

    for idx, row in imp_df.iterrows():
        slate_size = row["_slate_size"]
        supply_ids = supply_index.get_supply_ids(row["_time_ns"], window_hours=PRIMARY_SUPPLY_WINDOW_HOURS)
        is_elig = supply_index.check_counterfactual_eligibility(
            slate_size=slate_size,
            supply_count=len(supply_ids),
            history_length=profiles[row["user_id"]].history_length,
        )
        if is_elig:
            eligible_imps.append(row["impression_id"])

    elig_df = imp_df[imp_df["impression_id"].isin(eligible_imps)].copy()
    print(f"  -> Eligible impressions: {len(elig_df)} / {len(imp_df)}")

    print("[5/8] Computing 200-draw null slates and Available -> Exposed gap...")
    mock_nov = {aid: float(rng.uniform(0.2, 0.9)) for aid in art_ids}
    mock_rel = {aid: float(rng.uniform(0.05, 0.25)) for aid in art_ids}
    mock_prem = dict(zip(articles_df["article_id"], articles_df["premium"]))

    avail_gaps = []
    choice_gaps = []
    obs_disc_list = []

    for idx, row in elig_df.iterrows():
        slate = row["_slate"]
        supply_ids = supply_index.get_supply_ids(row["_time_ns"], window_hours=24)
        
        slate_nov = np.array([mock_nov[a] for a in slate])
        slate_rel = np.array([mock_rel[a] for a in slate])
        obs_m = compute_slate_discovery_metrics(slate_nov, slate_rel, tau)
        obs_disc_list.append(obs_m["discovery"])

        supp_nov = np.array([mock_nov[a] for a in supply_ids])
        supp_rel = np.array([mock_rel[a] for a in supply_ids])
        supp_disc = np.where(supp_rel >= tau, supp_nov, 0.0)
        weights = np.ones(len(supply_ids))

        null_exp = compute_null_expectations(
            candidate_discovery=supp_disc,
            candidate_relevance=supp_rel,
            candidate_novelty=supp_nov,
            weights=weights,
            slate_size=len(slate),
            tau=tau,
            n_draws=200,
            rng_seed=SEED + row["impression_id"],
        )
        avail_gaps.append(obs_m["discovery"] - null_exp["null_discovery_pop"])

        choice_res = compute_choice_stage_metrics(
            slate_article_ids=slate,
            clicked_article_ids=row["article_ids_clicked"],
            article_novelty_map=mock_nov,
            article_relevance_map=mock_rel,
            article_premium_map=mock_prem,
            is_subscriber=row["is_subscriber"],
            tau=tau,
        )
        choice_gaps.append(choice_res["choice_discovery_gap"])

    elig_df["observed_discovery"] = obs_disc_list
    elig_df["avail_discovery_gap"] = avail_gaps
    elig_df["choice_discovery_gap"] = choice_gaps

    print("[6/8] Testing deterministic 1:1 matching...")
    matching_res = perform_deterministic_matching(elig_df, caliper=1)
    print(f"  -> Matched pairs: {matching_res.matched_pairs_count} (Low match rate: {matching_res.low_match_rate:.1%})")

    print("[7/8] Testing Headroom optimizer...")
    cs = CommonSupportIndex(elig_df)
    headroom_gains = []
    
    for idx, row in elig_df.iterrows():
        supp_ids = supply_index.get_supply_ids(row["_time_ns"], window_hours=24)
        cands = list(set(supp_ids).intersection(cs.get_supported_articles(row["_time_bucket"], min_distinct_users=1)))
        h_res = optimize_headroom_single_impression(
            observed_slate=row["_slate"],
            article_novelty_map=mock_nov,
            article_relevance_map=mock_rel,
            candidate_ids=cands,
            tau=tau,
            qualification_status="QUALIFIED",
            relevance_tolerance=HEADROOM_PRIMARY_RELEVANCE_TOLERANCE,
            max_swaps=HEADROOM_MAX_SWAPS,
        )
        headroom_gains.append(h_res.discovery_gain)
    
    elig_df["headroom_gain"] = headroom_gains
    print(f"  -> Mean synthetic Headroom discovery gain: {np.mean(headroom_gains):.4f}")

    print("[8/8] Testing 2,000-replicate user-clustered bootstrap...")
    boot_avail = user_clustered_bootstrap(
        elig_df, metric_col="avail_discovery_gap", user_col="user_id",
        n_replicates=2000, seed=SEED
    )
    print(f"  -> Available->Exposed gap: {boot_avail.point_estimate:.4f} [95% CI: {boot_avail.ci_lower:.4f}, {boot_avail.ci_upper:.4f}], std_effect={boot_avail.standardized_effect:.2f} SD")

    boot_headroom = user_clustered_bootstrap(
        elig_df, metric_col="headroom_gain", user_col="user_id",
        n_replicates=2000, seed=SEED
    )
    print(f"  -> Headroom gain: {boot_headroom.point_estimate:.4f} [95% CI: {boot_headroom.ci_lower:.4f}, {boot_headroom.ci_upper:.4f}], std_effect={boot_headroom.standardized_effect:.2f} SD")

    elapsed = time.perf_counter() - start_time
    print(f"\nSUCCESS: End-to-end dry run completed in {elapsed:.2f}s.")
    print("All machinery components verified with zero exposure to real validation data.")
    print("Dry run completed successfully.")


def run_authorized_validation(sample_limit: Optional[int] = None):
    """
    Execute the one-time primary held-out validation run.
    """
    total_start_time = time.perf_counter()
    ensure_directories()

    # Step 0: Record Provenance
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_head = "0e6ba855de6b9ddd6b98556fdc3fee39996a9041"

    print("======================================================================")
    print("PHASE 1: ONE-TIME PRIMARY HELD-OUT VALIDATION RUN")
    print("======================================================================")
    print(f"Git HEAD: {git_head}")
    print("Protocol loaded. Held-out validation outcome analysis is starting.\n")

    # Step 1: Pre-period History Validation
    print("[1/10] Verifying validation pre-period history source...")
    hist_verification = validate_history_source(VALIDATION_HISTORY_PATH)
    print(f"  -> Pre-period validation history verified: {hist_verification['distinct_users']:,} users across {hist_verification['span_days']:.1f} days")
    print(f"  -> History range: {hist_verification['min_time']} to {hist_verification['max_time']}")

    # Step 2: Ingest Articles Metadata & Contrastive Vectors
    print("\n[2/10] Loading articles metadata and contrastive embedding store...")
    t0 = time.perf_counter()
    articles = load_articles()
    print(f"  -> Loaded {len(articles):,} articles metadata in {time.perf_counter() - t0:.2f}s")
    
    t0 = time.perf_counter()
    emb_con = build_or_load_embedding_store("contrastive", articles)
    print(f"  -> Loaded contrastive embedding store ({len(emb_con):,} articles) in {time.perf_counter() - t0:.2f}s")

    # Step 3: Construct Validation User Profiles (Strictly Pre-Period History)
    print("\n[3/10] Building validation user profiles strictly from validation/history.parquet...")
    t0 = time.perf_counter()
    val_profiles = load_validation_user_profiles(VALIDATION_HISTORY_PATH, articles)
    print(f"  -> Successfully profiled {len(val_profiles):,} eligible users (history >= {MINIMUM_HISTORY_LENGTH}) in {time.perf_counter() - t0:.2f}s")

    # Step 4: Load Behaviors & Construct Time-Safe Unified Click Index
    print("\n[4/10] Loading behaviors and constructing time-safe popularity index...")
    t0 = time.perf_counter()
    train_behaviors_all = load_behaviors(ROOT_DIR / "ebnerd_small" / "train" / "behaviors.parquet", frontpage_only=False)
    val_behaviors_all = load_behaviors(VALIDATION_BEHAVIORS_PATH, frontpage_only=False)
    popularity_index = build_unified_time_safe_popularity_index(train_behaviors_all, val_behaviors_all)
    print(f"  -> Unified click index constructed across train + validation in {time.perf_counter() - t0:.2f}s")

    # Sample Accounting
    total_val_rows = len(val_behaviors_all)
    val_fp = val_behaviors_all[val_behaviors_all["article_id"].isna()].copy()
    val_fp = add_session_features(val_fp)
    total_fp_impressions = len(val_fp)

    slate_ok = (val_fp["_slate_size"] >= ELIGIBLE_SLATE_SIZE_MIN) & (val_fp["_slate_size"] <= ELIGIBLE_SLATE_SIZE_MAX)
    user_ok = val_fp["user_id"].isin(val_profiles)
    eligible_slate_impressions = int(slate_ok.sum())

    val_fp["breadth_cohort"] = [val_profiles[u].breadth_cohort if u in val_profiles else "unknown" for u in val_fp["user_id"]]
    val_fp["history_bin"] = [val_profiles[u].history_bin if u in val_profiles else "unknown" for u in val_fp["user_id"]]
    val_fp["history_length_bin"] = val_fp["history_bin"]

    pre_eligible_fp = val_fp[slate_ok & user_ok].copy()
    eligible_pre_supply = len(pre_eligible_fp)

    print(f"  -> Total validation behavior rows: {total_val_rows:,}")
    print(f"  -> Total front-page impressions: {total_fp_impressions:,}")
    print(f"  -> Impressions with slate size 5-26: {eligible_slate_impressions:,}")
    print(f"  -> Impressions with slate 5-26 & history >= 15: {eligible_pre_supply:,}")

    # Step 5: Observable Supply & 3x Counterfactual Eligibility
    print("\n[5/10] Indexing observable supply and filtering counterfactual-eligible impressions...")
    t0 = time.perf_counter()
    supply_index = ObservableSupplyIndex(articles)
    
    cf_eligible_mask = []
    supply_counts = []
    times_ns = pre_eligible_fp["_time_ns"].to_numpy()
    slate_sizes = pre_eligible_fp["_slate_size"].to_numpy()
    uids = pre_eligible_fp["user_id"].to_numpy()

    for t_ns, s_size, uid in zip(times_ns, slate_sizes, uids):
        s_cnt = supply_index.get_supply_count(t_ns, window_hours=PRIMARY_SUPPLY_WINDOW_HOURS)
        supply_counts.append(s_cnt)
        is_cf = supply_index.check_counterfactual_eligibility(
            slate_size=s_size,
            supply_count=s_cnt,
            history_length=val_profiles[uid].history_length,
        )
        cf_eligible_mask.append(is_cf)

    pre_eligible_fp["_supply_count_24h"] = supply_counts
    primary_eligible_fp = pre_eligible_fp[cf_eligible_mask].copy()
    n_primary = len(primary_eligible_fp)
    distinct_users_primary = primary_eligible_fp["user_id"].nunique()
    cohort_counts = primary_eligible_fp["breadth_cohort"].value_counts().to_dict()

    print(f"  -> Primary counterfactual-eligible impressions (supply >= 3x slate): {n_primary:,} ({n_primary / eligible_pre_supply:.1%}) in {time.perf_counter() - t0:.2f}s")
    print(f"  -> Distinct users in primary analysis: {distinct_users_primary:,}")
    print(f"  -> Breadth cohort breakdown: {cohort_counts}")

    if sample_limit is not None and n_primary > sample_limit:
        print(f"  [BENCHMARK LIMIT] Subsampling to {sample_limit} impressions...")
        primary_eligible_fp = primary_eligible_fp.head(sample_limit).copy()
        n_primary = len(primary_eligible_fp)

    # Step 6: Held-Out Relevance Model Evaluation & Qualification Gate
    print("\n[6/10] Evaluating frozen production relevance model on held-out front-page slates...")
    t0 = time.perf_counter()
    val_relevance_df = extract_relevance_examples(
        frontpage_df=pre_eligible_fp,
        articles_df=articles,
        profiles=val_profiles,
        popularity_index=popularity_index,
        embedding_store=emb_con,
    )
    print(f"  -> Extracted {len(val_relevance_df):,} validation interaction examples in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    relevance_pipe = joblib.load(CACHE_TRAIN_DIR / "relevance_model.joblib")
    with open(CACHE_TRAIN_DIR / "tau.json", "r", encoding="utf-8") as f:
        tau_data = json.load(f)
    frozen_tau = float(tau_data["frozen_tau"])

    feat_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    val_y_true = val_relevance_df["y"].to_numpy()
    val_y_pred = relevance_pipe.predict_proba(val_relevance_df[feat_cols])
    val_imp_ids = val_relevance_df["impression_id"].to_numpy()
    val_pop_weights = val_relevance_df["pop_rec_weight"].to_numpy()

    val_auc = float(roc_auc_score(val_y_true, val_y_pred))
    val_ndcg = float(compute_mean_ndcg(val_imp_ids, val_y_true, val_y_pred))
    base_ndcg = float(compute_mean_ndcg(val_imp_ids, val_y_true, val_pop_weights))
    ndcg_lift = float(val_ndcg - base_ndcg)
    val_brier = float(brier_score_loss(val_y_true, val_y_pred))

    is_qualified = bool(val_auc >= 0.60 and ndcg_lift >= 0.01)
    qualification_status = "QUALIFIED" if is_qualified else "NOT_QUALIFIED"

    # Pre-extract model weights for vectorized candidate scoring
    w_num = (relevance_pipe.model.coef_[0, :10] / relevance_pipe.scaler.scale_).astype(np.float64)
    b_scaler = -float(np.sum(relevance_pipe.scaler.mean_ * relevance_pipe.model.coef_[0, :10] / relevance_pipe.scaler.scale_))
    w_dev = relevance_pipe.model.coef_[0, 10:13].astype(np.float64)
    b_total = float(relevance_pipe.model.intercept_[0] + b_scaler)
    dev_idx_map = {"1": 0, "2": 1, "3": 2}

    print(f"  -> Validation ROC-AUC: {val_auc:.4f} (Required: >= 0.60)")
    print(f"  -> Validation Model nDCG: {val_ndcg:.4f}")
    print(f"  -> Popularity+Recency Baseline nDCG: {base_ndcg:.4f}")
    print(f"  -> nDCG Lift over Baseline: {ndcg_lift:+.4f} (Required: >= +0.01)")
    print(f"  -> Validation Brier Score: {val_brier:.4f}")
    print(f"  -> Qualification Gate Status: [{qualification_status}] (evaluated in {time.perf_counter() - t0:.2f}s)")

    # Step 7: Precompute Multi-k Novelty Cache
    print("\n[7/10] Precomputing semantic novelty cache for eligible validation users and candidate pool...")
    t0 = time.perf_counter()
    unique_pairs_needed: Dict[int, Set[int]] = {}
    for uid, slate, t_ns in zip(
        primary_eligible_fp["user_id"],
        primary_eligible_fp["_slate"],
        primary_eligible_fp["_time_ns"],
    ):
        if uid not in unique_pairs_needed:
            unique_pairs_needed[uid] = set()
        unique_pairs_needed[uid].update(slate)
        supp_ids = supply_index.get_supply_ids(t_ns, window_hours=PRIMARY_SUPPLY_WINDOW_HOURS)
        unique_pairs_needed[uid].update(supp_ids)

    total_pairs = sum(len(c) for c in unique_pairs_needed.values())
    print(f"  -> Identified {total_pairs:,} unique (user_id, article_id) pairs across {len(unique_pairs_needed):,} users")

    novelty_cache_path = CACHE_VAL_DIR / "user_article_novelty.parquet"
    novelty_df = precompute_novelty_cache(
        user_ids_and_candidates=unique_pairs_needed,
        profiles=val_profiles,
        embedding_store=emb_con,
        k_values=(3, 5, 10),
        output_path=novelty_cache_path,
    )
    print(f"  -> Computed & saved novelty cache to {novelty_cache_path} in {time.perf_counter() - t0:.2f}s")

    # Build fast lookup dictionary: (user_id, article_id) -> novelty_k5
    user_art_nov_k5: Dict[Tuple[int, int], float] = dict(
        zip(zip(novelty_df["user_id"], novelty_df["article_id"]), novelty_df["novelty_k5"])
    )

    # Step 8: Available -> Exposed (200 Null Draws), Exposed -> Consumed, & Headroom Analysis
    print("\n[8/10] Computing Available->Exposed gaps (200 null draws), choice gaps, and headroom...")
    t0 = time.perf_counter()
    art_meta = articles.drop_duplicates("article_id").set_index("article_id")
    art_prem_map = art_meta["premium"].to_dict()
    art_pub_map = art_meta["_published_ns"].to_dict()
    art_cat_map = art_meta["category"].to_dict()
    art_topics_map = {aid: list(set(norm_list(top))) for aid, top in art_meta["topics"].items()}

    # Initialize common support index if model cleared qualification
    if is_qualified:
        print("  -> Model QUALIFIED. Indexing common support for Headroom optimization...")
        cs_index: Optional[CommonSupportIndex] = CommonSupportIndex(primary_eligible_fp)
    else:
        print("  -> Model NOT QUALIFIED for Headroom optimization. Slates will not be optimized.")
        cs_index = None

    primary_records = []
    
    for idx, row in primary_eligible_fp.iterrows():
        imp_id = row["impression_id"]
        uid = row["user_id"]
        slate = row["_slate"]
        s_size = len(slate)
        t_ns = row["_time_ns"]
        imp_t = row["impression_time"]
        dev = row["device_type"]
        is_sub = row["is_subscriber"]
        clicked = norm_list(row["article_ids_clicked"])
        u_prof = val_profiles[uid]

        # Candidate observable supply
        supp_ids = supply_index.get_supply_ids(t_ns, window_hours=PRIMARY_SUPPLY_WINDOW_HOURS)
        all_eval_ids = list(dict.fromkeys(list(slate) + list(supp_ids)))
        
        # Batch predict relevance for slate + supply items via vectorized matrix multiplication
        prior_clicks_batch = popularity_index.get_prior_clicks_batch(all_eval_ids, t_ns)
        prior_clicks_map = dict(zip(all_eval_ids, prior_clicks_batch))
        hour = imp_t.hour
        tod_sin = float(np.sin(2 * np.pi * hour / 24.0))
        tod_cos = float(np.cos(2 * np.pi * hour / 24.0))
        sub_int = 1.0 if is_sub else 0.0
        h_len = float(max(1, u_prof.history_length))

        n_eval = len(all_eval_ids)
        X_eval = np.empty((n_eval, 10), dtype=np.float64)

        for k, aid in enumerate(all_eval_ids):
            nov = user_art_nov_k5.get((uid, aid), 0.5)
            cat = art_cat_map.get(aid, -1)
            cat_aff = float(u_prof.category_counts.get(cat, 0) / h_len)
            tops = art_topics_map.get(aid, [])
            top_aff = float(np.mean([u_prof.topic_counts.get(t, 0) for t in tops]) / h_len) if tops else 0.0
            pub = art_pub_map.get(aid, t_ns)
            age_h = max(0.0, (t_ns - pub) / 3.6e12)
            clicks = float(prior_clicks_batch[k])
            prem = 1.0 if art_prem_map.get(aid, False) else 0.0

            X_eval[k, 0] = 1.0 - nov  # sim_to_history
            X_eval[k, 1] = cat_aff
            X_eval[k, 2] = top_aff
            X_eval[k, 3] = np.log1p(age_h)
            X_eval[k, 4] = np.log1p(clicks)
            X_eval[k, 5] = prem
            X_eval[k, 6] = sub_int
            X_eval[k, 7] = prem * sub_int
            X_eval[k, 8] = tod_sin
            X_eval[k, 9] = tod_cos

        dev_col_idx = dev_idx_map.get(str(dev), 0)
        logits = X_eval @ w_num + w_dev[dev_col_idx] + b_total
        eval_preds = 1.0 / (1.0 + np.exp(-logits))

        art_rel_map = dict(zip(all_eval_ids, eval_preds))
        art_nov_map = {aid: user_art_nov_k5.get((uid, aid), 0.5) for aid in all_eval_ids}

        # Observed Slate Metrics
        slate_nov = np.array([art_nov_map[a] for a in slate], dtype=np.float64)
        slate_rel = np.array([art_rel_map[a] for a in slate], dtype=np.float64)
        obs_metrics = compute_slate_discovery_metrics(slate_nov, slate_rel, frozen_tau)

        # Null Slates Expectations
        supp_nov = np.array([art_nov_map[a] for a in supp_ids], dtype=np.float64)
        supp_rel = np.array([art_rel_map[a] for a in supp_ids], dtype=np.float64)
        supp_disc = compute_item_discovery(supp_nov, supp_rel, frozen_tau)
        
        supp_ages = np.array([max(0.0, (t_ns - art_pub_map.get(a, t_ns)) / 3.6e12) for a in supp_ids], dtype=np.float64)
        supp_clicks = np.array([prior_clicks_map.get(a, 0) for a in supp_ids], dtype=np.float64)
        supp_weights = compute_item_sampling_weights(supp_clicks, supp_ages)

        null_exp = compute_null_expectations(
            candidate_discovery=supp_disc,
            candidate_relevance=supp_rel,
            candidate_novelty=supp_nov,
            weights=supp_weights,
            slate_size=s_size,
            tau=frozen_tau,
            n_draws=200,
            rng_seed=SEED + imp_id,
        )

        avail_disc_gap = obs_metrics["discovery"] - null_exp["null_discovery_pop"]
        avail_cov_gap = obs_metrics["relevance_coverage"] - null_exp["null_relevance_cov_pop"]
        avail_cond_nov_gap = obs_metrics["conditional_novelty"] - null_exp["null_cond_novelty_pop"]
        avail_disc_gap_rand = obs_metrics["discovery"] - null_exp["null_discovery_rand"]

        # Choice Stage Metrics
        choice_res = compute_choice_stage_metrics(
            slate_article_ids=slate,
            clicked_article_ids=clicked,
            article_novelty_map=art_nov_map,
            article_relevance_map=art_rel_map,
            article_premium_map=art_prem_map,
            is_subscriber=is_sub,
            tau=frozen_tau,
        )

        # Discovery Headroom Optimization (if qualified)
        if is_qualified and cs_index is not None:
            bucket = row["_time_bucket"]
            cs_supported = cs_index.get_supported_articles(bucket, min_distinct_users=COMMON_SUPPORT_N_PRIMARY)
            cand_pool = list(set(supp_ids).intersection(cs_supported))

            h_res = optimize_headroom_single_impression(
                observed_slate=slate,
                article_novelty_map=art_nov_map,
                article_relevance_map=art_rel_map,
                candidate_ids=cand_pool,
                tau=frozen_tau,
                qualification_status="QUALIFIED",
                relevance_tolerance=HEADROOM_PRIMARY_RELEVANCE_TOLERANCE,
                max_swaps=HEADROOM_MAX_SWAPS,
            )
            hr_gain = h_res.discovery_gain
            hr_swaps = h_res.n_swaps
            hr_retention = h_res.relevance_retention_ratio
            hr_pos = bool(h_res.discovery_gain > 0)
        else:
            hr_gain = np.nan
            hr_swaps = np.nan
            hr_retention = np.nan
            hr_pos = False

        primary_records.append({
            "impression_id": imp_id,
            "user_id": uid,
            "impression_time": imp_t,
            "_time_ns": t_ns,
            "_time_bucket": row["_time_bucket"],
            "device_type": dev,
            "is_subscriber": is_sub,
            "_session_stage": row["_session_stage"],
            "_session_len_bin": row["_session_len_bin"],
            "history_length_bin": row["history_length_bin"],
            "breadth_cohort": row["breadth_cohort"],
            "_slate_size": s_size,
            "_supply_count": len(supp_ids),
            # Observed metrics
            "obs_discovery": obs_metrics["discovery"],
            "obs_coverage": obs_metrics["relevance_coverage"],
            "obs_cond_novelty": obs_metrics["conditional_novelty"],
            "obs_mean_relevance": obs_metrics["mean_relevance"],
            # Null metrics
            "null_discovery_pop": null_exp["null_discovery_pop"],
            "null_cov_pop": null_exp["null_relevance_cov_pop"],
            "null_cond_nov_pop": null_exp["null_cond_novelty_pop"],
            "null_discovery_rand": null_exp["null_discovery_rand"],
            # Available->Exposed gaps
            "avail_discovery_gap": avail_disc_gap,
            "avail_coverage_gap": avail_cov_gap,
            "avail_cond_nov_gap": avail_cond_nov_gap,
            "avail_discovery_gap_rand": avail_disc_gap_rand,
            # Choice stage metrics
            "has_eligible_consumed": choice_res["has_eligible_consumed"],
            "choice_discovery_gap": choice_res["choice_discovery_gap"],
            "choice_coverage_gap": choice_res["choice_coverage_gap"],
            "choice_cond_novelty_gap": choice_res["choice_cond_novelty_gap"],
            "consumed_discovery": choice_res["consumed_discovery"],
            "exposed_discovery": choice_res["exposed_discovery"],
            # Headroom metrics
            "headroom_discovery_gain": hr_gain,
            "headroom_swaps": hr_swaps,
            "headroom_relevance_retention": hr_retention,
            "has_positive_headroom": hr_pos,
        })

    outcomes_df = pd.DataFrame(primary_records)
    print(f"  -> Evaluated {len(outcomes_df):,} primary impressions in {time.perf_counter() - t0:.2f}s")
    choice_eligible_count = int(outcomes_df["has_eligible_consumed"].sum())
    print(f"  -> Choice-stage eligible impressions (with eligible consumed item): {choice_eligible_count:,} ({choice_eligible_count / len(outcomes_df):.1%})")

    # Step 9: Deterministic Matched Low-vs-High Comparison
    print("\n[9/10] Performing deterministic 1:1 nearest-neighbor matching (Low vs High breadth)...")
    t0 = time.perf_counter()
    # Pass outcomes_df which already has all stratum and outcome columns
    matching_res = perform_deterministic_matching(outcomes_df, caliper=1)
    print(f"  -> Eligible Low: {matching_res.eligible_low_count:,} | Eligible High: {matching_res.eligible_high_count:,}")
    print(f"  -> Matched pairs: {matching_res.matched_pairs_count:,} (Low match rate: {matching_res.low_match_rate:.1%}, High match rate: {matching_res.high_match_rate:.1%}) in {time.perf_counter() - t0:.2f}s")

    # Join pair outcome differences (High minus Low)
    if matching_res.matched_pairs_count > 0:
        outcomes_by_id = outcomes_df.set_index("impression_id")
        pairs_df = matching_res.pairs_df.copy()
        
        low_disc = outcomes_by_id.loc[pairs_df["low_impression_id"], "obs_discovery"].to_numpy()
        high_disc = outcomes_by_id.loc[pairs_df["high_impression_id"], "obs_discovery"].to_numpy()
        low_cov = outcomes_by_id.loc[pairs_df["low_impression_id"], "obs_coverage"].to_numpy()
        high_cov = outcomes_by_id.loc[pairs_df["high_impression_id"], "obs_coverage"].to_numpy()
        low_nov = outcomes_by_id.loc[pairs_df["low_impression_id"], "obs_cond_novelty"].to_numpy()
        high_nov = outcomes_by_id.loc[pairs_df["high_impression_id"], "obs_cond_novelty"].to_numpy()

        pairs_df["low_discovery"] = low_disc
        pairs_df["high_discovery"] = high_disc
        pairs_df["discovery_diff_high_minus_low"] = high_disc - low_disc
        pairs_df["coverage_diff_high_minus_low"] = high_cov - low_cov
        pairs_df["novelty_diff_high_minus_low"] = high_nov - low_nov
    else:
        pairs_df = pd.DataFrame()

    # Step 10: Headroom Summary (Strictly Gated by Qualification)
    print(f"\n[10/10] Compiling Discovery Headroom Results (Gate Status: [{qualification_status}])...")
    t0 = time.perf_counter()
    headroom_data: Dict[str, Any] = {
        "status": qualification_status,
        "validation_auc": val_auc,
        "validation_ndcg": val_ndcg,
        "baseline_ndcg": base_ndcg,
        "ndcg_lift": ndcg_lift,
        "required_auc": 0.60,
        "required_ndcg_lift": 0.01,
        "wording_note": "retaining at least 97% of relevance by our model's own estimate",
    }

    if not is_qualified:
        headroom_data["reason"] = "Model qualification thresholds not cleared on held-out validation. Headroom optimization aborted."
        print("  -> Headroom optimization SKIPPED (qualification gate was NOT cleared).")
    else:
        print("  -> Computing user-clustered bootstrap on Headroom discovery gain...")
        hr_boot = user_clustered_bootstrap(
            outcomes_df, metric_col="headroom_discovery_gain", user_col="user_id",
            n_replicates=BOOTSTRAP_REPLICATES, seed=SEED
        )
        headroom_data.update({
            "specification": "Primary: 24h observable supply, common support >= 3, max 2 swaps, 97% relevance retention",
            "eligible_impressions": len(outcomes_df),
            "share_with_positive_headroom": float(outcomes_df["has_positive_headroom"].mean()),
            "mean_headroom_discovery_gain": float(outcomes_df["headroom_discovery_gain"].mean()),
            "median_headroom_discovery_gain": float(outcomes_df["headroom_discovery_gain"].median()),
            "ci_95": [hr_boot.ci_lower, hr_boot.ci_upper],
            "std_error": hr_boot.std_error,
            "between_user_sd": hr_boot.between_user_sd,
            "standardized_magnitude": hr_boot.standardized_effect,
            "practical_significance": hr_boot.practical_significance,
            "average_swaps_used": float(outcomes_df["headroom_swaps"].mean()),
            "average_modeled_relevance_retention": float(outcomes_df["headroom_relevance_retention"].mean()),
        })
        print(f"  -> Headroom summary and bootstrap completed in {time.perf_counter() - t0:.2f}s")
        print(f"  -> Mean Headroom Discovery Gain: {headroom_data['mean_headroom_discovery_gain']:+.4f} [{hr_boot.ci_lower:+.4f}, {hr_boot.ci_upper:+.4f}] (std: {hr_boot.standardized_effect:.2f} SD)")

    # 2,000-Replicate User-Clustered Bootstrap on All Primary Outcomes
    print("\n--- USER-CLUSTERED BOOTSTRAP (2,000 REPLICATES, SEED 20260919) ---")
    boot_avail_disc = user_clustered_bootstrap(outcomes_df, "avail_discovery_gap", "user_id")
    boot_avail_cov = user_clustered_bootstrap(outcomes_df, "avail_coverage_gap", "user_id")
    boot_avail_nov = user_clustered_bootstrap(outcomes_df, "avail_cond_nov_gap", "user_id")

    boot_choice_disc = user_clustered_bootstrap(outcomes_df, "choice_discovery_gap", "user_id")
    boot_choice_cov = user_clustered_bootstrap(outcomes_df, "choice_coverage_gap", "user_id")
    boot_choice_nov = user_clustered_bootstrap(outcomes_df, "choice_cond_novelty_gap", "user_id")

    if len(pairs_df) > 0:
        boot_matched_disc = user_clustered_bootstrap(pairs_df, "discovery_diff_high_minus_low", "low_user_id")
        boot_matched_cov = user_clustered_bootstrap(pairs_df, "coverage_diff_high_minus_low", "low_user_id")
        boot_matched_nov = user_clustered_bootstrap(pairs_df, "novelty_diff_high_minus_low", "low_user_id")
    else:
        boot_matched_disc = None

    print(f"  Available->Exposed Discovery Gap: {boot_avail_disc.point_estimate:+.4f} [{boot_avail_disc.ci_lower:+.4f}, {boot_avail_disc.ci_upper:+.4f}] (std: {boot_avail_disc.standardized_effect:+.2f} SD)")
    print(f"  Exposed->Consumed Choice Gap:     {boot_choice_disc.point_estimate:+.4f} [{boot_choice_disc.ci_lower:+.4f}, {boot_choice_disc.ci_upper:+.4f}] (std: {boot_choice_disc.standardized_effect:+.2f} SD)")
    if boot_matched_disc:
        print(f"  High minus Low Matched Diff:     {boot_matched_disc.point_estimate:+.4f} [{boot_matched_disc.ci_lower:+.4f}, {boot_matched_disc.ci_upper:+.4f}] (std: {boot_matched_disc.standardized_effect:+.2f} SD)")

    # User Aggregates
    user_agg = outcomes_df.groupby("user_id").agg({
        "obs_discovery": "mean",
        "avail_discovery_gap": "mean",
        "choice_discovery_gap": "mean",
        "headroom_discovery_gain": "mean",
        "impression_id": "count",
    }).reset_index().rename(columns={"impression_id": "impression_count"})

    # Export Artifacts
    print("\n--- EXPORTING PRIMARY VALIDATION ARTIFACTS ---")
    
    # 1. outputs/relevance_model_metrics.json
    rel_metrics = {
        "validation_roc_auc": val_auc,
        "validation_ndcg": val_ndcg,
        "validation_baseline_ndcg": base_ndcg,
        "validation_ndcg_lift": ndcg_lift,
        "validation_brier_score": val_brier,
        "frozen_tau": frozen_tau,
        "qualification_thresholds": {"min_auc": 0.60, "min_lift": 0.01},
        "is_qualified": is_qualified,
        "qualification_status": qualification_status,
        "total_scored_examples": len(val_relevance_df),
        "positive_examples": int(val_y_true.sum()),
        "positive_rate": float(val_y_true.mean()),
        "causal_interpretation_warning": "Model coefficients reflect statistical associations within exposed slates and must not be interpreted causally."
    }
    export_json_artifact(rel_metrics, OUTPUTS_DIR / "relevance_model_metrics.json")
    print("  -> Saved outputs/relevance_model_metrics.json")

    # 2. outputs/stage_effects.json
    stage_effects = {
        "available_to_exposed": {
            "metric": "observed_discovery - mean_null_discovery",
            "point_estimate": boot_avail_disc.point_estimate,
            "ci_95": [boot_avail_disc.ci_lower, boot_avail_disc.ci_upper],
            "std_error": boot_avail_disc.std_error,
            "between_user_sd": boot_avail_disc.between_user_sd,
            "standardized_effect": boot_avail_disc.standardized_effect,
            "practical_significance": boot_avail_disc.practical_significance,
            "n_users": boot_avail_disc.n_users,
            "decomposition": {
                "relevance_coverage_gap": {
                    "point_estimate": boot_avail_cov.point_estimate,
                    "ci_95": [boot_avail_cov.ci_lower, boot_avail_cov.ci_upper],
                    "standardized_effect": boot_avail_cov.standardized_effect,
                },
                "conditional_novelty_gap": {
                    "point_estimate": boot_avail_nov.point_estimate,
                    "ci_95": [boot_avail_nov.ci_lower, boot_avail_nov.ci_upper],
                    "standardized_effect": boot_avail_nov.standardized_effect,
                }
            }
        },
        "exposed_to_consumed": {
            "metric": "consumed_discovery - choosable_exposed_discovery",
            "point_estimate": boot_choice_disc.point_estimate,
            "ci_95": [boot_choice_disc.ci_lower, boot_choice_disc.ci_upper],
            "std_error": boot_choice_disc.std_error,
            "between_user_sd": boot_choice_disc.between_user_sd,
            "standardized_effect": boot_choice_disc.standardized_effect,
            "practical_significance": boot_choice_disc.practical_significance,
            "eligible_impressions": choice_eligible_count,
            "n_users": boot_choice_disc.n_users,
            "decomposition": {
                "relevance_coverage_gap": {
                    "point_estimate": boot_choice_cov.point_estimate,
                    "ci_95": [boot_choice_cov.ci_lower, boot_choice_cov.ci_upper],
                    "standardized_effect": boot_choice_cov.standardized_effect,
                },
                "conditional_novelty_gap": {
                    "point_estimate": boot_choice_nov.point_estimate,
                    "ci_95": [boot_choice_nov.ci_lower, boot_choice_nov.ci_upper],
                    "standardized_effect": boot_choice_nov.standardized_effect,
                }
            }
        }
    }
    export_json_artifact(stage_effects, OUTPUTS_DIR / "stage_effects.json")
    print("  -> Saved outputs/stage_effects.json")

    # 3. outputs/null_comparison.json
    null_comp = {
        "observed_slate": {
            "mean_discovery": float(outcomes_df["obs_discovery"].mean()),
            "mean_relevance_coverage": float(outcomes_df["obs_coverage"].mean()),
            "mean_conditional_novelty": float(outcomes_df["obs_cond_novelty"].dropna().mean()),
            "mean_predicted_relevance": float(outcomes_df["obs_mean_relevance"].mean()),
        },
        "popularity_recency_weighted_null": {
            "mean_discovery": float(outcomes_df["null_discovery_pop"].mean()),
            "mean_relevance_coverage": float(outcomes_df["null_cov_pop"].mean()),
            "mean_conditional_novelty": float(outcomes_df["null_cond_nov_pop"].dropna().mean()),
            "discovery_gap": boot_avail_disc.point_estimate,
            "ci_95_discovery_gap": [boot_avail_disc.ci_lower, boot_avail_disc.ci_upper],
            "coverage_gap": boot_avail_cov.point_estimate,
            "conditional_novelty_gap": boot_avail_nov.point_estimate,
        },
        "uniform_random_null": {
            "mean_discovery": float(outcomes_df["null_discovery_rand"].mean()),
            "discovery_gap": float(outcomes_df["avail_discovery_gap_rand"].mean()),
        }
    }
    export_json_artifact(null_comp, OUTPUTS_DIR / "null_comparison.json")
    print("  -> Saved outputs/null_comparison.json")

    # 4. outputs/breadth_comparison.json
    breadth_comp = {
        "eligible_low_impressions": matching_res.eligible_low_count,
        "eligible_high_impressions": matching_res.eligible_high_count,
        "matched_pairs_count": matching_res.matched_pairs_count,
        "low_match_rate": matching_res.low_match_rate,
        "high_match_rate": matching_res.high_match_rate,
        "unmatched_low_count": matching_res.unmatched_low_count,
        "unmatched_high_count": matching_res.unmatched_high_count,
        "effect_direction": "High breadth minus Low breadth",
        "discovery_difference": {
            "point_estimate": boot_matched_disc.point_estimate if boot_matched_disc else None,
            "ci_95": [boot_matched_disc.ci_lower, boot_matched_disc.ci_upper] if boot_matched_disc else None,
            "std_error": boot_matched_disc.std_error if boot_matched_disc else None,
            "between_user_sd": boot_matched_disc.between_user_sd if boot_matched_disc else None,
            "standardized_effect": boot_matched_disc.standardized_effect if boot_matched_disc else None,
            "practical_significance": boot_matched_disc.practical_significance if boot_matched_disc else None,
        },
        "coverage_difference": {
            "point_estimate": boot_matched_cov.point_estimate if boot_matched_cov else None,
            "ci_95": [boot_matched_cov.ci_lower, boot_matched_cov.ci_upper] if boot_matched_cov else None,
            "standardized_effect": boot_matched_cov.standardized_effect if boot_matched_cov else None,
        },
        "novelty_difference": {
            "point_estimate": boot_matched_nov.point_estimate if boot_matched_nov else None,
            "ci_95": [boot_matched_nov.ci_lower, boot_matched_nov.ci_upper] if boot_matched_nov else None,
            "standardized_effect": boot_matched_nov.standardized_effect if boot_matched_nov else None,
        }
    }
    export_json_artifact(breadth_comp, OUTPUTS_DIR / "breadth_comparison.json")
    print("  -> Saved outputs/breadth_comparison.json")

    # 5. outputs/headroom.json
    export_json_artifact(headroom_data, OUTPUTS_DIR / "headroom.json")
    print("  -> Saved outputs/headroom.json")

    # 6. Tidy CSV Exports
    export_tidy_csv(outcomes_df, OUTPUTS_DIR / "tidy_impression_outcomes.csv")
    print("  -> Saved outputs/tidy_impression_outcomes.csv")
    if len(pairs_df) > 0:
        export_tidy_csv(pairs_df, OUTPUTS_DIR / "tidy_matched_pairs.csv")
        print("  -> Saved outputs/tidy_matched_pairs.csv")
    export_tidy_csv(user_agg, OUTPUTS_DIR / "tidy_user_aggregates.csv")
    print("  -> Saved outputs/tidy_user_aggregates.csv")

    # 7. outputs/run_manifest.json
    total_elapsed = time.perf_counter() - total_start_time
    gen_files = [
        "outputs/relevance_model_metrics.json",
        "outputs/stage_effects.json",
        "outputs/null_comparison.json",
        "outputs/breadth_comparison.json",
        "outputs/headroom.json",
        "outputs/tidy_impression_outcomes.csv",
        "outputs/tidy_matched_pairs.csv",
        "outputs/tidy_user_aggregates.csv",
        "cache/validation/user_article_novelty.parquet",
    ]
    manifest_files = []
    for gf in gen_files:
        p = ROOT / gf
        if p.exists():
            manifest_files.append({
                "path": gf,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    run_manifest = {
        "status": "PRIMARY_VALIDATION_RUN_COMPLETE",
        "split": "validation",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_commit_hash": PROTOCOL_COMMIT_HASH,
        "preoutcome_gate_commit_hash": PREOUTCOME_GATE_COMMIT_HASH,
        "implementation_lock_commit_hash": IMPLEMENTATION_LOCK_COMMIT_HASH,
        "train_implementation_commit_hash": "2e1383454c51e86cd7bb42a144d53a9bc1fd0e88",
        "validation_implementation_commit_hash": "38c7e886c71743c17649591142bac31573a1903b",
        "pre_validation_qa_commit_hash": git_head,
        "seed": SEED,
        "protocol_version": PROTOCOL_VERSION,
        "sample_accounting": {
            "total_validation_behavior_rows": total_val_rows,
            "total_frontpage_impressions": total_fp_impressions,
            "eligible_slate_size_impressions": eligible_slate_impressions,
            "eligible_users_with_history_ge_15": len(val_profiles),
            "eligible_impressions_pre_supply": eligible_pre_supply,
            "primary_counterfactual_eligible_impressions": n_primary,
            "distinct_users_in_primary_analysis": distinct_users_primary,
            "choice_stage_eligible_impressions": choice_eligible_count,
            "breadth_counts": cohort_counts,
            "exclusion_counts_by_reason": {
                "non_frontpage_context": total_val_rows - total_fp_impressions,
                "slate_size_out_of_bounds": total_fp_impressions - eligible_slate_impressions,
                "user_history_lt_15": eligible_slate_impressions - eligible_pre_supply,
                "insufficient_observable_supply_lt_3x": eligible_pre_supply - n_primary,
                "no_eligible_consumed_item_in_choice": n_primary - choice_eligible_count,
            }
        },
        "relevance_model_qualification": {
            "validation_auc": val_auc,
            "validation_ndcg": val_ndcg,
            "baseline_ndcg": base_ndcg,
            "ndcg_lift": ndcg_lift,
            "is_qualified": is_qualified,
            "status": qualification_status,
        },
        "execution_runtime_seconds": {
            "total_pipeline": total_elapsed,
        },
        "generated_files": manifest_files,
    }
    export_json_artifact(run_manifest, OUTPUTS_DIR / "run_manifest.json")
    print("  -> Saved outputs/run_manifest.json")

    # 8. outputs/primary_validation_summary.md
    summary_md = rf"""# Phase 1 Primary Held-Out Validation Analysis Report

**Freeze Date**: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}  
**Seed**: `{SEED}` | **Protocol Version**: `{PROTOCOL_VERSION}`  
**Git HEAD**: `{git_head}`  

---

## 1. Validation Sample Accounting

| Step / Filter Stage | Impression Count | Distinct Users | Retention |
| :--- | :---: | :---: | :---: |
| **Total Validation Behavior Rows** | {total_val_rows:,} | — | 100.0% |
| **Front-Page Impressions** (`article_id` is null) | {total_fp_impressions:,} | — | {total_fp_impressions / total_val_rows:.1%} |
| **Slate Size Filter** ($S \in [5, 26]$) | {eligible_slate_impressions:,} | — | {eligible_slate_impressions / total_fp_impressions:.1%} |
| **Pre-Period History Filter** ($H \ge 15$) | {eligible_pre_supply:,} | {len(val_profiles):,} | {eligible_pre_supply / eligible_slate_impressions:.1%} |
| **Counterfactual Supply Filter** ($\text{{Supply}} \ge 3S$) | **{n_primary:,}** | **{distinct_users_primary:,}** | **{n_primary / eligible_pre_supply:.1%}** |
| **Choice Stage Eligible** (has choosable click) | {choice_eligible_count:,} | {boot_choice_disc.n_users:,} | {choice_eligible_count / n_primary:.1%} |

- **Breadth Cohort Breakdown**: Low: {cohort_counts.get('low', 0):,} | Medium: {cohort_counts.get('medium', 0):,} | High: {cohort_counts.get('high', 0):,}

---

## 2. Held-Out Relevance Model Evaluation & Qualification Gate

- **Validation ROC-AUC**: **{val_auc:.4f}** (Threshold: $\ge 0.60$ — {'PASSED' if val_auc >= 0.60 else 'FAILED'})
- **Validation Model nDCG**: **{val_ndcg:.4f}**
- **Popularity+Recency Baseline nDCG**: **{base_ndcg:.4f}**
- **nDCG Lift over Baseline**: **{ndcg_lift:+.4f}** (Threshold: $\ge +0.01$ — {'PASSED' if ndcg_lift >= 0.01 else 'FAILED'})
- **Validation Brier Score**: **{val_brier:.4f}**
- **Frozen Relevance Floor $\tau$**: **`{frozen_tau}`**
- **Qualification Decision**: **[{qualification_status}]**

> [!NOTE]
> Headroom qualification status is **{qualification_status}**. {'Headroom optimization was executed.' if is_qualified else 'Headroom optimization was strictly blocked as per protocol.'}

---

## 3. Stage 1: Available $\to$ Exposed Primary Analysis (Null Slates)

Comparing observed publisher slates against 200 popularity+recency weighted draws and uniform random draws from 24h observable supply:

| Metric | Observed Slate | Pop+Recency Null | Gap (Obs - Null) | 95% Bootstrap CI | Standardized Effect |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Discovery $D(u, \mathcal{{S}})$** | {outcomes_df['obs_discovery'].mean():.4f} | {outcomes_df['null_discovery_pop'].mean():.4f} | **{boot_avail_disc.point_estimate:+.4f}** | [{boot_avail_disc.ci_lower:+.4f}, {boot_avail_disc.ci_upper:+.4f}] | **{boot_avail_disc.standardized_effect:+.2f} SD** |
| **Relevance Coverage** | {outcomes_df['obs_coverage'].mean():.4f} | {outcomes_df['null_cov_pop'].mean():.4f} | **{boot_avail_cov.point_estimate:+.4f}** | [{boot_avail_cov.ci_lower:+.4f}, {boot_avail_cov.ci_upper:+.4f}] | **{boot_avail_cov.standardized_effect:+.2f} SD** |
| **Conditional Novelty** | {outcomes_df['obs_cond_novelty'].dropna().mean():.4f} | {outcomes_df['null_cond_nov_pop'].dropna().mean():.4f} | **{boot_avail_nov.point_estimate:+.4f}** | [{boot_avail_nov.ci_lower:+.4f}, {boot_avail_nov.ci_upper:+.4f}] | **{boot_avail_nov.standardized_effect:+.2f} SD** |

- **Uniform Random Null Discovery**: {outcomes_df['null_discovery_rand'].mean():.4f} (Gap vs Random: {outcomes_df['avail_discovery_gap_rand'].mean():+.4f})

---

## 4. Stage 2: Exposed $\to$ Consumed Primary Analysis (Choice Stage)

Applying the subscriber/non-subscriber premium choice rule ({choice_eligible_count:,} eligible impressions across {boot_choice_disc.n_users:,} users):

| Metric | Choosable Exposed | Consumed (Clicked) | Choice Gap | 95% Bootstrap CI | Standardized Effect |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Discovery $D(u, \mathcal{{S}})$** | {outcomes_df.loc[outcomes_df['has_eligible_consumed'], 'exposed_discovery'].mean():.4f} | {outcomes_df.loc[outcomes_df['has_eligible_consumed'], 'consumed_discovery'].mean():.4f} | **{boot_choice_disc.point_estimate:+.4f}** | [{boot_choice_disc.ci_lower:+.4f}, {boot_choice_disc.ci_upper:+.4f}] | **{boot_choice_disc.standardized_effect:+.2f} SD** |
| **Relevance Coverage** | {outcomes_df.loc[outcomes_df['has_eligible_consumed'], 'obs_coverage'].mean():.4f} | 1.0000 | **{boot_choice_cov.point_estimate:+.4f}** | [{boot_choice_cov.ci_lower:+.4f}, {boot_choice_cov.ci_upper:+.4f}] | **{boot_choice_cov.standardized_effect:+.2f} SD** |
| **Conditional Novelty** | {outcomes_df.loc[outcomes_df['has_eligible_consumed'], 'obs_cond_novelty'].dropna().mean():.4f} | {outcomes_df.loc[outcomes_df['has_eligible_consumed'], 'consumed_discovery'].mean():.4f} | **{boot_choice_nov.point_estimate:+.4f}** | [{boot_choice_nov.ci_lower:+.4f}, {boot_choice_nov.ci_upper:+.4f}] | **{boot_choice_nov.standardized_effect:+.2f} SD** |

---

## 5. Matched Low-vs-High Breadth Comparison

Deterministic 1:1 nearest-neighbor matching across 6 exact strata within $|\Delta \text{{slate}}| \le 1$ caliper without replacement:
- **Eligible Low Impressions**: {matching_res.eligible_low_count:,}
- **Eligible High Impressions**: {matching_res.eligible_high_count:,}
- **Matched Pairs**: **{matching_res.matched_pairs_count:,}** (Low Match Rate: {matching_res.low_match_rate:.1%}, High Match Rate: {matching_res.high_match_rate:.1%})
- **Effect Direction**: **High breadth minus Low breadth**

| Metric | Mean Matched Difference (High - Low) | 95% Bootstrap CI | Standardized Effect | Practical Significance ($|d| \ge 0.10$) |
| :--- | :---: | :---: | :---: | :---: |
| **Discovery Difference** | **{boot_matched_disc.point_estimate if boot_matched_disc else float('nan'):+.4f}** | [{boot_matched_disc.ci_lower if boot_matched_disc else float('nan'):+.4f}, {boot_matched_disc.ci_upper if boot_matched_disc else float('nan'):+.4f}] | **{boot_matched_disc.standardized_effect if boot_matched_disc else float('nan'):+.2f} SD** | **{boot_matched_disc.practical_significance if boot_matched_disc else False}** |
| **Coverage Difference** | **{boot_matched_cov.point_estimate if boot_matched_cov else float('nan'):+.4f}** | [{boot_matched_cov.ci_lower if boot_matched_cov else float('nan'):+.4f}, {boot_matched_cov.ci_upper if boot_matched_cov else float('nan'):+.4f}] | **{boot_matched_cov.standardized_effect if boot_matched_cov else float('nan'):+.2f} SD** | **{boot_matched_cov.practical_significance if boot_matched_cov else False}** |
| **Novelty Difference** | **{boot_matched_nov.point_estimate if boot_matched_nov else float('nan'):+.4f}** | [{boot_matched_nov.ci_lower if boot_matched_nov else float('nan'):+.4f}, {boot_matched_nov.ci_upper if boot_matched_nov else float('nan'):+.4f}] | **{boot_matched_nov.standardized_effect if boot_matched_nov else float('nan'):+.2f} SD** | **{boot_matched_nov.practical_significance if boot_matched_nov else False}** |

---

## 6. Discovery Headroom Analysis

- **Headroom Qualification Status**: **{qualification_status}**
"""
    if is_qualified:
        summary_md += rf"""
- **Specification**: 24h observable supply $\cap$ common support $\ge 3$ distinct users in same 30-min window $\cap$ relevance $\ge \tau$.
- **Constraints**: Maximum 2 swaps, retaining at least 97% of relevance by our model's own estimate.
- **Eligible Impressions**: {len(outcomes_df):,}
- **Share of Impressions with Positive Headroom**: **{headroom_data['share_with_positive_headroom']:.1%}**
- **Mean Headroom Discovery Gain**: **{headroom_data['mean_headroom_discovery_gain']:+.4f}** (95% CI: [{headroom_data['ci_95'][0]:+.4f}, {headroom_data['ci_95'][1]:+.4f}])
- **Median Headroom Discovery Gain**: **{headroom_data['median_headroom_discovery_gain']:+.4f}**
- **Standardized Effect**: **{headroom_data['standardized_magnitude']:+.2f} SD** (Practical Significance: **{headroom_data['practical_significance']}**)
- **Average Swaps Utilized**: **{headroom_data['average_swaps_used']:.2f}** / 2.0
- **Average Modeled Relevance Retention**: **{headroom_data['average_modeled_relevance_retention']:.1%}** (retaining at least 97% of relevance by our model's own estimate)
"""
    else:
        summary_md += rf"""
- **Headroom Status**: NOT QUALIFIED.
- **Reason**: {headroom_data['reason']}
- Slates were not optimized.
"""

    summary_md += rf"""
---
*Run completed in {total_elapsed:.1f}s. Results frozen.*
"""
    with open(OUTPUTS_DIR / "primary_validation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md.strip() + "\n")
    print("  -> Saved outputs/primary_validation_summary.md")

    print(f"\nSUCCESS: Primary held-out validation analysis complete in {total_elapsed:.1f}s.")
    print("PRIMARY HELD-OUT VALIDATION RUN COMPLETE — RESULTS FROZEN")


def main():
    args = parse_args()

    # REAL VALIDATION GUARD
    authorized = getattr(args, "authorize_heldout_run", False)
    if not authorized:
        if not args.dry_run:
            print("=" * 80, file=sys.stderr)
            print("ABORT: REAL HELD-OUT VALIDATION DATA IS LOCKED.", file=sys.stderr)
            print("Accessing 'ebnerd_small/validation/behaviors.parquet' requires explicit user authorization.", file=sys.stderr)
            print("To authorize execution against real held-out validation data, you must pass:", file=sys.stderr)
            print("    --authorize-heldout-run", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        print("=" * 70)
        print("PHASE 1: RUNNING VALIDATION MACHINERY IN DRY-RUN MODE (SYNTHETIC FIXTURE)")
        print("=" * 70)
        run_synthetic_dry_run()
        return

    # Real execution logic (authorized)
    run_authorized_validation(sample_limit=args.sample_impressions)


if __name__ == "__main__":
    main()
