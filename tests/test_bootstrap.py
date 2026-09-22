"""
Unit tests for user-clustered bootstrap estimation and standardized effect sizes.
"""

import unittest
import numpy as np
import pandas as pd

from src.bootstrap import (
    user_clustered_bootstrap,
    bootstrap_matched_pairs,
    BootstrapEstimate,
)


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        # 100 users, each having 1 to 5 impressions
        rng = np.random.default_rng(20260919)
        rows = []
        # User true mean effects drawn from N(0.05, 0.02^2)
        user_effects = rng.normal(loc=0.05, scale=0.02, size=100)
        for uid in range(100):
            n_imp = rng.integers(1, 6)
            for _ in range(n_imp):
                # Observation = user effect + noise
                val = user_effects[uid] + rng.normal(0, 0.005)
                rows.append({"user_id": uid, "metric": val})

        self.df = pd.DataFrame(rows)
        self.user_effects = user_effects

    def test_user_clustered_aggregation_and_reproducibility(self):
        """Bootstrap must aggregate to user level and be deterministically reproducible."""
        est1 = user_clustered_bootstrap(
            self.df, metric_col="metric", user_col="user_id",
            n_replicates=500, seed=20260919
        )
        est2 = user_clustered_bootstrap(
            self.df, metric_col="metric", user_col="user_id",
            n_replicates=500, seed=20260919
        )

        # 1. Number of clusters equals number of distinct users
        self.assertEqual(est1.n_users, 100)

        # 2. Point estimate equals mean of user means
        user_means = self.df.groupby("user_id")["metric"].mean()
        self.assertAlmostEqual(est1.point_estimate, float(user_means.mean()), places=6)

        # 3. Deterministic reproducibility
        self.assertEqual(est1.ci_lower, est2.ci_lower)
        self.assertEqual(est1.ci_upper, est2.ci_upper)
        self.assertEqual(est1.std_error, est2.std_error)

        # 4. CI bounds bracket point estimate
        self.assertLessEqual(est1.ci_lower, est1.point_estimate)
        self.assertGreaterEqual(est1.ci_upper, est1.point_estimate)

    def test_standardized_effect_computation(self):
        """Check standardized effect computation (point_estimate / between_user_sd)."""
        est = user_clustered_bootstrap(
            self.df, metric_col="metric", user_col="user_id",
            n_replicates=100, seed=20260919
        )
        expected_std_effect = est.point_estimate / est.between_user_sd
        self.assertAlmostEqual(est.standardized_effect, expected_std_effect, places=6)

    def test_matched_pairs_bootstrap(self):
        """Test cluster-robust bootstrap on matched pairs differences."""
        pairs = pd.DataFrame({
            "low_user_id": [1, 1, 2, 3, 4],
            "low_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            "high_val": [0.15, 0.25, 0.32, 0.45, 0.55],
        })
        est = bootstrap_matched_pairs(
            pairs, low_metric_col="low_val", high_metric_col="high_val",
            cluster_col="low_user_id", n_replicates=200, seed=20260919
        )
        self.assertEqual(est.n_users, 4)
        self.assertGreater(est.point_estimate, 0.0)


if __name__ == "__main__":
    unittest.main()
