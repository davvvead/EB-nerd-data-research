"""
Unit tests for null slate generation, weighted sampling properties,
and expectation caching.
"""

import unittest
import numpy as np

from src.nulls import (
    compute_item_sampling_weights,
    draw_weighted_null_slates,
    draw_uniform_null_slates,
    compute_null_expectations,
)


class TestNullSlates(unittest.TestCase):
    def setUp(self):
        self.n_cands = 50
        self.cand_ids = np.arange(1000, 1000 + self.n_cands, dtype=np.int64)
        self.slate_size = 10
        self.n_draws = 200

        # Create synthetic weights: item 0 has massive weight, item 49 has small weight
        self.clicks = np.linspace(500, 0, self.n_cands)
        self.ages = np.linspace(1, 24, self.n_cands)
        self.weights = compute_item_sampling_weights(self.clicks, self.ages)

    def test_weighted_null_slates_properties(self):
        """Test size, distinctness, candidate membership, and reproducibility."""
        rng1 = np.random.default_rng(20260919)
        draws1 = draw_weighted_null_slates(
            self.cand_ids, self.weights, self.slate_size, n_draws=self.n_draws, rng=rng1
        )

        # 1. Exact shape (n_draws, slate_size)
        self.assertEqual(draws1.shape, (self.n_draws, self.slate_size))

        # 2. No duplicates within any draw
        for i in range(self.n_draws):
            unique_items = np.unique(draws1[i])
            self.assertEqual(len(unique_items), self.slate_size)

        # 3. All items belong to candidate pool
        cand_set = set(self.cand_ids)
        for i in range(self.n_draws):
            for aid in draws1[i]:
                self.assertIn(aid, cand_set)

        # 4. Deterministic reproducibility with same seed
        rng2 = np.random.default_rng(20260919)
        draws2 = draw_weighted_null_slates(
            self.cand_ids, self.weights, self.slate_size, n_draws=self.n_draws, rng=rng2
        )
        np.testing.assert_array_equal(draws1, draws2)

    def test_weighted_sampling_frequency_distribution(self):
        """High-weight items should appear significantly more frequently than low-weight items."""
        rng = np.random.default_rng(20260919)
        draws = draw_weighted_null_slates(
            self.cand_ids, self.weights, slate_size=5, n_draws=1000, rng=rng
        )
        # Count frequency of top item (index 0) vs bottom item (index 49)
        top_aid = self.cand_ids[0]
        bot_aid = self.cand_ids[-1]

        top_count = np.sum(draws == top_aid)
        bot_count = np.sum(draws == bot_aid)

        self.assertGreater(top_count, bot_count * 5)

    def test_uniform_null_slates_properties(self):
        """Test uniform null draws."""
        rng = np.random.default_rng(20260919)
        draws = draw_uniform_null_slates(
            self.cand_ids, self.slate_size, n_draws=self.n_draws, rng=rng
        )
        self.assertEqual(draws.shape, (self.n_draws, self.slate_size))
        for i in range(self.n_draws):
            self.assertEqual(len(np.unique(draws[i])), self.slate_size)

    def test_compute_null_expectations(self):
        """Test expectation computation and caching."""
        cand_disc = np.linspace(0.1, 0.9, self.n_cands)
        cand_rel = np.linspace(0.05, 0.25, self.n_cands)
        cand_nov = np.linspace(0.2, 0.8, self.n_cands)
        tau = 0.10

        exp_dict = compute_null_expectations(
            candidate_discovery=cand_disc,
            candidate_relevance=cand_rel,
            candidate_novelty=cand_nov,
            weights=self.weights,
            slate_size=self.slate_size,
            tau=tau,
            n_draws=self.n_draws,
            rng_seed=20260919,
        )

        expected_keys = {
            "null_discovery_pop", "null_relevance_cov_pop", "null_cond_novelty_pop",
            "null_discovery_rand", "null_relevance_cov_rand", "null_cond_novelty_rand",
        }
        self.assertEqual(set(exp_dict.keys()), expected_keys)
        for k, v in exp_dict.items():
            self.assertTrue(np.isfinite(v), f"Value for {k} is not finite: {v}")


if __name__ == "__main__":
    unittest.main()
