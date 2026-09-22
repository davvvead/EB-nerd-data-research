#!/usr/bin/env python3
"""
Build and cache all train-side foundational artifacts:
1. User profiles and breadth cohort assignments.
2. Time-safe click popularity index.
3. Pre-normalized embedding stores.
4. Relevance training examples table.
Strictly isolated to train split with zero validation leakage.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import (
    ROOT_DIR,
    SEED,
    MINIMUM_HISTORY_LENGTH,
    ELIGIBLE_SLATE_SIZE_MIN,
    ELIGIBLE_SLATE_SIZE_MAX,
    CACHE_TRAIN_DIR,
    ensure_directories,
)
from src.data import (
    load_articles,
    load_behaviors,
    load_history,
    timestamp_selftest,
)
from src.embeddings import build_or_load_embedding_store
from src.profiles import build_user_profiles, export_profiles_to_dataframe
from src.popularity import ClickPopularityIndex
from src.relevance import extract_relevance_examples


def main():
    start_time = time.perf_counter()
    print("=" * 70)
    print("PHASE 1: BUILDING TRAIN ARTIFACTS (STRICT TRAIN ISOLATION)")
    print("=" * 70)

    ensure_directories()
    
    # 1. Timestamp resolution self-test
    print("\n[1/6] Running timestamp nanosecond resolution self-test...")
    selftest = timestamp_selftest()
    print(f"  -> Timestamp self-test PASSED (observed 24h delta: {selftest['observed_24h_ns']} ns)")

    # 2. Ingest articles metadata
    print("\n[2/6] Loading articles metadata...")
    articles = load_articles()
    print(f"  -> Loaded {len(articles):,} articles. Published time span: {articles['published_time'].min()} to {articles['published_time'].max()}")

    # 3. Build & cache pre-normalized embeddings
    print("\n[3/6] Building pre-normalized embedding stores...")
    t0 = time.perf_counter()
    emb_con = build_or_load_embedding_store("contrastive", articles)
    print(f"  -> Contrastive vectors: {len(emb_con):,} articles indexed (took {time.perf_counter() - t0:.2f}s)")
    t0 = time.perf_counter()
    emb_bert = build_or_load_embedding_store("bert", articles)
    print(f"  -> Multilingual BERT vectors: {len(emb_bert):,} articles indexed (took {time.perf_counter() - t0:.2f}s)")

    # 4. Ingest train history & build user profiles
    print("\n[4/6] Building train user profiles from 21-day pre-period history...")
    t0 = time.perf_counter()
    train_history = load_history(ROOT_DIR / "ebnerd_small" / "train" / "history.parquet")
    profiles = build_user_profiles(train_history, articles, min_history_len=MINIMUM_HISTORY_LENGTH)
    print(f"  -> Evaluated {len(train_history):,} total train users; retained {len(profiles):,} eligible users (hist_len >= {MINIMUM_HISTORY_LENGTH})")
    
    profiles_df = export_profiles_to_dataframe(profiles)
    profiles_df.to_parquet(CACHE_TRAIN_DIR / "user_profiles.parquet", index=False)
    print(f"  -> Saved user profiles to {CACHE_TRAIN_DIR / 'user_profiles.parquet'} (took {time.perf_counter() - t0:.2f}s)")
    cohort_counts = profiles_df["breadth_cohort"].value_counts().to_dict()
    print(f"  -> Breadth cohort breakdown: {cohort_counts}")

    # 5. Build time-safe click popularity index
    print("\n[5/6] Constructing time-safe click popularity index from train behaviors...")
    t0 = time.perf_counter()
    train_behaviors_all = load_behaviors(ROOT_DIR / "ebnerd_small" / "train" / "behaviors.parquet", frontpage_only=False)
    popularity_index = ClickPopularityIndex.from_behaviors(train_behaviors_all)
    popularity_index.save(CACHE_TRAIN_DIR / "click_index.npz")
    print(f"  -> Indexed clicks across {len(popularity_index.clicks):,} distinct articles")
    print(f"  -> Saved popularity index to {CACHE_TRAIN_DIR / 'click_index.npz'} (took {time.perf_counter() - t0:.2f}s)")

    # 6. Extract relevance training examples
    print("\n[6/6] Extracting relevance training examples from front-page impressions...")
    t0 = time.perf_counter()
    train_fp = load_behaviors(ROOT_DIR / "ebnerd_small" / "train" / "behaviors.parquet", frontpage_only=True)
    
    # Filter eligible front-page impressions
    eligible_mask = (
        (train_fp["_slate_size"] >= ELIGIBLE_SLATE_SIZE_MIN) &
        (train_fp["_slate_size"] <= ELIGIBLE_SLATE_SIZE_MAX) &
        (train_fp["user_id"].isin(profiles))
    )
    eligible_fp = train_fp[eligible_mask].copy()
    print(f"  -> Eligible front-page impressions: {len(eligible_fp):,} / {len(train_fp):,} ({len(eligible_fp)/len(train_fp):.1%})")
    
    relevance_df = extract_relevance_examples(
        frontpage_df=eligible_fp,
        articles_df=articles,
        profiles=profiles,
        popularity_index=popularity_index,
        embedding_store=emb_con,
    )
    relevance_path = CACHE_TRAIN_DIR / "train_relevance_examples.parquet"
    relevance_df.to_parquet(relevance_path, index=False)
    print(f"  -> Generated {len(relevance_df):,} item-level interaction rows")
    print(f"  -> Positive clicks: {relevance_df['y'].sum():,} ({relevance_df['y'].mean():.2%})")
    print(f"  -> Saved relevance dataset to {relevance_path} (took {time.perf_counter() - t0:.2f}s)")

    total_time = time.perf_counter() - start_time
    print(f"\nSUCCESS: Train-side artifacts successfully generated in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
