#!/usr/bin/env python3
"""
Fit relevance model using forward-chaining and fold-local preprocessing.
Derives and freezes relevance threshold tau from out-of-fold train predictions only.
Trains final production relevance pipeline on full eligible train data.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CACHE_TRAIN_DIR,
    OUTPUTS_DIR,
    ensure_directories,
)
from src.relevance import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    make_forward_chain_blocks,
    run_forward_chained_oof,
    RelevancePipeline,
)


def main():
    start_time = time.perf_counter()
    print("=" * 70)
    print("PHASE 2: FITTING RELEVANCE MODEL & DERIVING FROZEN TAU (TRAIN ONLY)")
    print("=" * 70)

    ensure_directories()
    examples_path = CACHE_TRAIN_DIR / "train_relevance_examples.parquet"
    if not examples_path.exists():
        raise FileNotFoundError(
            f"Training examples not found at {examples_path}. Run scripts/01_build_train_artifacts.py first."
        )

    print("\n[1/5] Loading train relevance examples...")
    df = pd.read_parquet(examples_path)
    print(f"  -> Loaded {len(df):,} interaction rows across {df['impression_id'].nunique():,} impressions")
    print(f"  -> Total clicks: {df['y'].sum():,} ({df['y'].mean():.2%})")

    # 2. Segment into 5 forward-chaining blocks with strict timestamp tie-break
    print("\n[2/5] Partitioning impressions into 5 forward-chaining blocks...")
    blocks = make_forward_chain_blocks(df, n_blocks=5)
    for idx, b in enumerate(blocks, 1):
        t_min = b["impression_time"].min()
        t_max = b["impression_time"].max()
        print(f"  -> Block B{idx}: {len(b):,} interactions ({b['impression_id'].nunique():,} imps) | {t_min} to {t_max}")

    # 3. Run forward-chained out-of-fold training
    print("\n[3/5] Running forward-chained out-of-fold training (fold-local preprocessing)...")
    print("      Block sequence:")
    print("        B1 (warm-up) -> Predict B2")
    print("        B1+B2        -> Predict B3")
    print("        B1+B2+B3     -> Predict B4")
    print("        B1+B2+B3+B4  -> Predict B5")
    
    t0 = time.perf_counter()
    pooled_oof, diagnostics, tau = run_forward_chained_oof(blocks)
    oof_time = time.perf_counter() - t0
    print(f"  -> OOF training and evaluation completed in {oof_time:.1f}s")
    
    # Save OOF predictions
    oof_path = CACHE_TRAIN_DIR / "oof_predictions.parquet"
    pooled_oof[["impression_id", "user_id", "article_id", "y", "oof_score", "oof_fold"]].to_parquet(oof_path, index=False)
    print(f"  -> Saved OOF predictions to {oof_path}")

    # 4. Save tau and diagnostics
    print("\n[4/5] Freezing relevance floor tau and train diagnostics...")
    tau_info = {
        "frozen_tau": tau,
        "derivation_rule": "20th percentile of out-of-fold predicted relevance among clicked items in train blocks B2..B5",
        "oof_auc": diagnostics["oof_auc"],
        "oof_ndcg": diagnostics["oof_ndcg"],
        "oof_brier": diagnostics["oof_brier"],
        "baseline_ndcg": diagnostics["baseline_ndcg"],
        "ndcg_lift_over_baseline": diagnostics["ndcg_lift_over_baseline"],
        "sample_counts": {
            "total_oof_examples": diagnostics["total_oof_examples"],
            "clicked_oof_examples": diagnostics["clicked_oof_examples"],
            "unclicked_oof_examples": diagnostics["unclicked_oof_examples"],
        },
        "clicked_score_distribution": diagnostics["clicked_score_distribution"],
    }
    
    tau_path = CACHE_TRAIN_DIR / "tau.json"
    with open(tau_path, "w", encoding="utf-8") as f:
        json.dump(tau_info, f, indent=2)
    print(f"  -> Saved frozen tau to {tau_path}")

    # Also save to outputs directory for report generation
    metrics_path = OUTPUTS_DIR / "relevance_model_train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(tau_info, f, indent=2)
    print(f"  -> Saved train model metrics to {metrics_path}")

    # 5. Fit production relevance pipeline on all eligible train data
    print("\n[5/5] Fitting final production relevance pipeline on all eligible train examples (B1..B5)...")
    t0 = time.perf_counter()
    prod_pipe = RelevancePipeline()
    prod_pipe.fit(df)
    prod_path = CACHE_TRAIN_DIR / "relevance_model.joblib"
    joblib.dump(prod_pipe, prod_path)
    print(f"  -> Fitted and saved production pipeline to {prod_path} (took {time.perf_counter() - t0:.2f}s)")

    # Feature importances / coefficients summary
    feature_names = list(NUMERIC_FEATURES) + list(prod_pipe.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    coefs = prod_pipe.model.coef_[0]
    intercept = float(prod_pipe.model.intercept_[0])
    
    coef_summary = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    print("\nProduction Model Feature Coefficients:")
    print(f"  Intercept: {intercept:.4f}")
    for name, c in coef_summary:
        print(f"  {name:<30}: {c:+.4f}")

    print("\n" + "=" * 70)
    print("TRAIN RELEVANCE MODEL SUMMARY")
    print("=" * 70)
    print(f"OOF AUC                     : {diagnostics['oof_auc']:.4f}")
    print(f"OOF nDCG                    : {diagnostics['oof_ndcg']:.4f}")
    print(f"Baseline nDCG (Pop+Rec)     : {diagnostics['baseline_ndcg']:.4f}")
    print(f"nDCG Lift over Baseline     : {diagnostics['ndcg_lift_over_baseline']:+.4f}")
    print(f"OOF Brier Score             : {diagnostics['oof_brier']:.4f}")
    print(f"FROZEN TAU (20th percentile): {tau:.6f}")
    print(f"Clicked Score Range         : [{diagnostics['clicked_score_distribution']['min']:.4f}, {diagnostics['clicked_score_distribution']['max']:.4f}]")
    print(f"Clicked Score Median        : {diagnostics['clicked_score_distribution']['median']:.4f}")
    print(f"Total Runtime               : {time.perf_counter() - start_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
