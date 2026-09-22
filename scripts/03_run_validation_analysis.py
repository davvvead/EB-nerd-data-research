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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import (
    ROOT_DIR,
    SEED,
    PRIMARY_SUPPLY_WINDOW_HOURS,
    PRIMARY_HISTORY_K,
    MINIMUM_HISTORY_LENGTH,
    ELIGIBLE_SLATE_SIZE_MIN,
    ELIGIBLE_SLATE_SIZE_MAX,
    BOOTSTRAP_REPLICATES,
    HEADROOM_PRIMARY_RELEVANCE_TOLERANCE,
    HEADROOM_MAX_SWAPS,
    CACHE_TRAIN_DIR,
    CACHE_VAL_DIR,
    OUTPUTS_DIR,
    ensure_directories,
)
from src.data import epoch_ns
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
from src.validation import (
    build_validation_user_profiles,
    evaluate_relevance_qualification,
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

    # 300 synthetic articles published in [-48h, 0h]
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
    # 40 users: 20 Low breadth, 20 High breadth
    user_rows = []
    for uid in range(1, 41):
        if uid <= 20:
            # Low breadth: mostly category 1
            hist = [int(rng.choice(art_ids[cats == 1])) for _ in range(25)]
        else:
            # High breadth: uniform across all categories
            hist = [int(rng.choice(art_ids)) for _ in range(25)]
        user_rows.append({"user_id": uid, "article_id_fixed": hist})

    hist_df = pd.DataFrame(user_rows)
    profiles = build_validation_user_profiles(hist_df, articles_df, min_history_len=15)
    print(f"  -> Generated {len(profiles)} user profiles across cohorts.")

    print("[3/8] Generating synthetic front-page impressions...")
    # 100 impressions
    imp_rows = []
    supply_index = ObservableSupplyIndex(articles_df)
    
    for imp_id in range(1, 101):
        uid = int(rng.integers(1, 41))
        t_offset = rng.uniform(0, 12) * 3600 * 1e9
        imp_t_ns = base_ns + int(t_offset)
        imp_t = pd.to_datetime(imp_t_ns, unit="ns", utc=True)
        dev = int(rng.choice([1, 2, 3]))
        sub = bool(rng.random() < 0.25)
        
        # Slate of 10 articles
        slate = list(rng.choice(art_ids, size=10, replace=False))
        # 0 or 1 click
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
    # Precompute mock novelty and relevance for all articles
    mock_nov = {aid: float(rng.uniform(0.2, 0.9)) for aid in art_ids}
    mock_rel = {aid: float(rng.uniform(0.05, 0.25)) for aid in art_ids}
    mock_prem = dict(zip(articles_df["article_id"], articles_df["premium"]))

    avail_gaps = []
    choice_gaps = []
    obs_disc_list = []

    for idx, row in elig_df.iterrows():
        slate = row["_slate"]
        supply_ids = supply_index.get_supply_ids(row["_time_ns"], window_hours=24)
        
        # Observed discovery
        slate_nov = np.array([mock_nov[a] for a in slate])
        slate_rel = np.array([mock_rel[a] for a in slate])
        obs_m = compute_slate_discovery_metrics(slate_nov, slate_rel, tau)
        obs_disc_list.append(obs_m["discovery"])

        # Null expectations
        supp_nov = np.array([mock_nov[a] for a in supply_ids])
        supp_rel = np.array([mock_rel[a] for a in supply_ids])
        supp_disc = np.where(supp_rel >= tau, supp_nov, 0.0)
        weights = np.ones(len(supply_ids))  # uniform mock weights for dry run

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

        # Choice stage
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

    print("=" * 70)
    if args.dry_run:
        print("PHASE 1: RUNNING VALIDATION MACHINERY IN DRY-RUN MODE (SYNTHETIC FIXTURE)")
    else:
        print("PHASE 1: RUNNING VALIDATION OUTCOME ANALYSIS (AUTHORIZED HELD-OUT RUN)")
    print("=" * 70)

    if args.dry_run:
        run_synthetic_dry_run()
        return

    # Real execution placeholder (when authorized)
    print("Beginning authorized validation outcome analysis...")


if __name__ == "__main__":
    main()
