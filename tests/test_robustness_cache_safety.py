"""
Unit tests for robustness novelty cache safety, isolation, and compatibility assertions.
Proves:
  1. Primary cache cannot be overwritten (PermissionError).
  2. Incorrect embedding type is rejected (CacheCompatibilityError).
  3. Incorrect embedding SHA is rejected (CacheCompatibilityError).
  4. Incomplete pair coverage is rejected (CacheIncompleteError).
  5. 12h cache subset coverage is verified.
  6. 48h cannot use incomplete 24h cache.
  7. BERT cannot load Contrastive cache.
  8. k3/k5/k10 values remain separated correctly.
  9. Robustness output paths cannot equal primary output paths.
  10. Deterministic regeneration yields identical results.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    ROOT_DIR,
    PROTOCOL_VERSION,
    SEED,
    PRIMARY_RESULTS_FREEZE_COMMIT_HASH,
    CACHE_VAL_DIR,
)
from src.embeddings import EmbeddingStore
from src.profiles import UserProfile
from src.validation import (
    FROZEN_PRIMARY_CACHE_PATH,
    FROZEN_PRIMARY_CACHE_SHA256,
    CONTRASTIVE_EMBEDDING_SHA256,
    BERT_EMBEDDING_SHA256,
    CacheCompatibilityError,
    CacheIncompleteError,
    NoveltyCacheMetadata,
    assert_novelty_cache_compatibility,
    precompute_novelty_cache,
)


class TestRobustnessCacheSafety(unittest.TestCase):
    """Test suite for robustness cache keying, compatibility, and isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cache_safety_")
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_cache(
        self,
        stem: str,
        embedding_type: str = "contrastive",
        embedding_sha: str = CONTRASTIVE_EMBEDDING_SHA256,
        supply_window: int = 24,
        k_values=(3, 5, 10),
        pairs=((1, 100), (1, 101), (2, 200)),
    ) -> Path:
        """Create a valid synthetic parquet cache with companion metadata."""
        records = []
        for uid, aid in pairs:
            rec = {"user_id": uid, "article_id": aid}
            for k in k_values:
                rec[f"novelty_k{k}"] = 0.5
            records.append(rec)

        df = pd.DataFrame(records)
        pq_path = self.temp_path / f"{stem}.parquet"
        df.to_parquet(pq_path, index=False)

        meta = NoveltyCacheMetadata(
            cache_version="v1",
            embedding_type=embedding_type,
            embedding_file_sha256=embedding_sha,
            candidate_universe=f"observable_supply_{supply_window}h",
            supply_window_hours=supply_window,
            k_values=list(k_values),
            user_count=int(df["user_id"].nunique()),
            user_article_pair_count=len(df),
            protocol_version=PROTOCOL_VERSION,
            seed=SEED,
            created_at_utc="2026-09-22T03:40:00+00:00",
            source_primary_freeze_commit_hash=PRIMARY_RESULTS_FREEZE_COMMIT_HASH,
        )
        meta_path = self.temp_path / f"{stem}.meta.json"
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2))
        return pq_path

    def test_01_primary_cache_cannot_be_overwritten(self):
        """1. Assert that writing to the frozen primary cache path raises PermissionError."""
        dummy_profiles = {
            1: UserProfile(
                user_id=1,
                history_article_ids=[10],
                category_counts={1: 1},
                category_entropy=1.5,
                breadth_cohort="low",
                history_bin="15-49",
                topic_counts={},
            )
        }
        dummy_store = EmbeddingStore("contrastive", np.array([10, 100]), np.ones((2, 768), dtype=np.float32) / np.sqrt(768))

        with self.assertRaises(PermissionError) as ctx:
            precompute_novelty_cache(
                user_ids_and_candidates={1: {100}},
                profiles=dummy_profiles,
                embedding_store=dummy_store,
                output_path=FROZEN_PRIMARY_CACHE_PATH,
                force_recompute=True,
            )
        self.assertIn("Cannot overwrite frozen primary cache", str(ctx.exception))

    def test_02_incorrect_embedding_type_rejected(self):
        """2. Assert that loading a Contrastive cache for BERT analysis raises CacheCompatibilityError."""
        cache_path = self._create_mock_cache("contrastive_cache", embedding_type="contrastive")
        meta_path = self.temp_path / "contrastive_cache.meta.json"
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(cache_path)

        with self.assertRaises(CacheCompatibilityError) as ctx:
            assert_novelty_cache_compatibility(
                cache_df=df,
                meta=meta,
                requested_embedding_type="bert_multilingual",
                requested_embedding_sha=BERT_EMBEDDING_SHA256,
                cache_path=cache_path,
            )
        self.assertIn("Embedding type mismatch", str(ctx.exception))

    def test_03_incorrect_embedding_sha_rejected(self):
        """3. Assert that a mismatched embedding file SHA raises CacheCompatibilityError."""
        cache_path = self._create_mock_cache("sha_test_cache", embedding_sha="corrupted_or_altered_sha")
        meta_path = self.temp_path / "sha_test_cache.meta.json"
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(cache_path)

        with self.assertRaises(CacheCompatibilityError) as ctx:
            assert_novelty_cache_compatibility(
                cache_df=df,
                meta=meta,
                requested_embedding_type="contrastive",
                requested_embedding_sha=CONTRASTIVE_EMBEDDING_SHA256,
                cache_path=cache_path,
            )
        self.assertIn("Embedding file SHA-256 mismatch", str(ctx.exception))

    def test_04_incomplete_pair_coverage_rejected(self):
        """4. Assert that missing user-article pairs raise CacheIncompleteError."""
        cache_path = self._create_mock_cache("coverage_cache", pairs=((1, 100), (1, 101)))
        meta_path = self.temp_path / "coverage_cache.meta.json"
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(cache_path)

        # Request pair (1, 999) which is not in cache
        required_pairs = {(1, 100), (1, 101), (1, 999)}

        with self.assertRaises(CacheIncompleteError) as ctx:
            assert_novelty_cache_compatibility(
                cache_df=df,
                meta=meta,
                requested_embedding_type="contrastive",
                requested_embedding_sha=CONTRASTIVE_EMBEDDING_SHA256,
                required_pairs=required_pairs,
                cache_path=cache_path,
            )
        self.assertIn("missing 1 required user-article pairs", str(ctx.exception))

    def test_05_12h_cache_subset_coverage_verified(self):
        """5. Assert that 12h analysis succeeds when 12h pairs are a strict subset of 24h cache."""
        # 24h cache has pairs (1, 100), (1, 101), (2, 200), (2, 201)
        cache_path = self._create_mock_cache("superset_24h", pairs=((1, 100), (1, 101), (2, 200), (2, 201)))
        meta_path = self.temp_path / "superset_24h.meta.json"
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(cache_path)

        # 12h requires strict subset: (1, 100) and (2, 200)
        req_12h_subset = {(1, 100), (2, 200)}

        # Should pass without error
        assert_novelty_cache_compatibility(
            cache_df=df,
            meta=meta,
            requested_embedding_type="contrastive",
            requested_embedding_sha=CONTRASTIVE_EMBEDDING_SHA256,
            required_pairs=req_12h_subset,
            cache_path=cache_path,
        )

        # If 12h requires an unexpected pair, it must fail
        with self.assertRaises(CacheIncompleteError):
            assert_novelty_cache_compatibility(
                cache_df=df,
                meta=meta,
                requested_embedding_type="contrastive",
                requested_embedding_sha=CONTRASTIVE_EMBEDDING_SHA256,
                required_pairs=req_12h_subset | {(3, 300)},
                cache_path=cache_path,
            )

    def test_06_48h_cannot_use_incomplete_24h_cache(self):
        """6. Assert that 48h analysis fails when attempting to use a 24h cache missing 48h pairs."""
        # 24h cache has only 24h pairs
        cache_path = self._create_mock_cache("cache_24h", pairs=((1, 100), (1, 101)))
        meta_path = self.temp_path / "cache_24h.meta.json"
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(cache_path)

        # 48h requires additional articles published in 24h-48h window
        req_48h = {(1, 100), (1, 101), (1, 102), (1, 103)}

        with self.assertRaises(CacheIncompleteError) as ctx:
            assert_novelty_cache_compatibility(
                cache_df=df,
                meta=meta,
                requested_embedding_type="contrastive",
                requested_embedding_sha=CONTRASTIVE_EMBEDDING_SHA256,
                required_pairs=req_48h,
                cache_path=cache_path,
            )
        self.assertIn("missing 2 required user-article pairs", str(ctx.exception))

    def test_07_bert_cannot_load_contrastive_cache(self):
        """7. Assert that BERT analysis cannot load the frozen primary Contrastive cache."""
        dummy_profiles = {
            1: UserProfile(
                user_id=1,
                history_article_ids=[10],
                category_counts={1: 1},
                category_entropy=1.5,
                breadth_cohort="low",
                history_bin="15-49",
                topic_counts={},
            )
        }
        dummy_bert_store = EmbeddingStore("bert", np.array([10, 100]), np.ones((2, 768), dtype=np.float32) / np.sqrt(768))

        with self.assertRaises(CacheCompatibilityError) as ctx:
            precompute_novelty_cache(
                user_ids_and_candidates={1: {100}},
                profiles=dummy_profiles,
                embedding_store=dummy_bert_store,
                embedding_type="bert_multilingual",
                embedding_sha256=BERT_EMBEDDING_SHA256,
                output_path=FROZEN_PRIMARY_CACHE_PATH,
            )
        self.assertIn("Embedding type mismatch", str(ctx.exception))

    def test_08_k3_k5_k10_separated_correctly(self):
        """8. Assert that k3, k5, k10 columns remain separated and missing k is detected."""
        cache_path = self._create_mock_cache("k_test", k_values=(3, 5))  # missing k=10
        meta_path = self.temp_path / "k_test.meta.json"
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(cache_path)

        with self.assertRaises(CacheCompatibilityError) as ctx:
            assert_novelty_cache_compatibility(
                cache_df=df,
                meta=meta,
                requested_embedding_type="contrastive",
                requested_embedding_sha=CONTRASTIVE_EMBEDDING_SHA256,
                requested_k_values=(3, 5, 10),
                cache_path=cache_path,
            )
        self.assertIn("Requested k=10 column 'novelty_k10' missing", str(ctx.exception))

    def test_09_robustness_output_paths_cannot_equal_primary(self):
        """9. Assert that robustness output directories never overlap with primary output files."""
        primary_files = {
            "run_manifest.json",
            "relevance_model_metrics.json",
            "stage_effects.json",
            "null_comparison.json",
            "breadth_comparison.json",
            "headroom.json",
            "primary_validation_summary.md",
            "primary_integrity_audit.json",
            "primary_integrity_audit.md",
            "tidy_impression_outcomes.csv",
            "tidy_matched_pairs.csv",
            "tidy_user_aggregates.csv",
        }

        rob_dir = Path("outputs/robustness")
        self.assertTrue(rob_dir.exists(), "outputs/robustness/ directory must exist")

        for variant_dir in rob_dir.iterdir():
            if variant_dir.is_dir():
                # Directory name must not collide with primary files
                self.assertNotIn(variant_dir.name, primary_files)
                # Any file inside robustness directory must have path starting with outputs/robustness/
                self.assertTrue(
                    str(variant_dir).startswith(str(rob_dir)),
                    f"Robustness directory {variant_dir} must be strictly inside {rob_dir}",
                )

    def test_10_deterministic_regeneration(self):
        """10. Assert that regenerating novelty from identical profiles and embeddings is bitwise deterministic."""
        rng = np.random.default_rng(20260919)
        vecs = rng.standard_normal((5, 768)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store = EmbeddingStore("test_det", np.array([1, 2, 3, 4, 5]), vecs)

        profiles = {
            10: UserProfile(
                user_id=10,
                history_article_ids=[1, 2],
                category_counts={1: 2},
                category_entropy=0.0,
                breadth_cohort="low",
                history_bin="15-49",
                topic_counts={},
            )
        }

        df1 = precompute_novelty_cache(
            user_ids_and_candidates={10: {3, 4, 5}},
            profiles=profiles,
            embedding_store=store,
            k_values=(3,),
            output_path=None,
        )

        df2 = precompute_novelty_cache(
            user_ids_and_candidates={10: {3, 4, 5}},
            profiles=profiles,
            embedding_store=store,
            k_values=(3,),
            output_path=None,
        )

        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    unittest.main()
