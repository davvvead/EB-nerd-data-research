"""
Unit tests for time-safe click popularity index and recency decay.
Verifies zero future leakage, window boundaries, and zero-click pseudocount.
"""

import unittest
import numpy as np
import pandas as pd

from src.popularity import ClickPopularityIndex


class TestPopularity(unittest.TestCase):
    def setUp(self):
        # Create synthetic behavior data with known click timestamps
        # Article 101 clicked at t=10h, t=15h, t=20h, t=25h (where t=0 is base)
        # Article 102 clicked at t=5h, t=26h
        # Article 103 has 0 clicks
        base = pd.Timestamp("2023-05-18T00:00:00Z")
        self.base_ns = base.value
        
        self.beh_df = pd.DataFrame([
            {"impression_time": base + pd.Timedelta(hours=5), "article_ids_clicked": [102]},
            {"impression_time": base + pd.Timedelta(hours=10), "article_ids_clicked": [101]},
            {"impression_time": base + pd.Timedelta(hours=15), "article_ids_clicked": [101]},
            {"impression_time": base + pd.Timedelta(hours=20), "article_ids_clicked": [101]},
            {"impression_time": base + pd.Timedelta(hours=25), "article_ids_clicked": [101]},
            {"impression_time": base + pd.Timedelta(hours=26), "article_ids_clicked": [102]},
        ])
        self.index = ClickPopularityIndex.from_behaviors(self.beh_df)

    def test_strictly_prior_clicks(self):
        # Target impression at t=20h exactly
        # Clicks for 101 are at 10h, 15h, 20h, 25h
        # Strictly prior clicks in prior 24h must include 10h and 15h, but NOT 20h (simultaneous) or 25h (future)
        target_t = self.base_ns + int(pd.Timedelta(hours=20).value)
        prior_clicks = self.index.get_prior_clicks(101, target_t, window_hours=24)
        self.assertEqual(prior_clicks, 2, "Clicks at or after impression time must be excluded")

    def test_window_boundary(self):
        # Target impression at t=30h with 24h window (lookback to t=6h)
        # For 101: clicks at 10h, 15h, 20h, 25h all fall within [6h, 30h)
        target_t = self.base_ns + int(pd.Timedelta(hours=30).value)
        prior_clicks = self.index.get_prior_clicks(101, target_t, window_hours=24)
        self.assertEqual(prior_clicks, 4)

        # For 102: click at 5h is outside [6h, 30h), click at 26h is inside
        prior_clicks_102 = self.index.get_prior_clicks(102, target_t, window_hours=24)
        self.assertEqual(prior_clicks_102, 1)

    def test_zero_click_pseudocount(self):
        # Article 103 has 0 clicks
        target_t = self.base_ns + int(pd.Timedelta(hours=24).value)
        pub_t = self.base_ns + int(pd.Timedelta(hours=12).value)  # Age = 12h
        
        # Age = 12h, clicks = 0 -> weight = (1 + 0) * 2^(-12/12) = 0.5
        weights = self.index.compute_null_weights([103], [pub_t], target_t, window_hours=24)
        self.assertAlmostEqual(weights[0], 0.5, places=5)

    def test_formula_with_clicks(self):
        # Target at t=26h. Lookback 24h is [2h, 26h).
        # Article 101 has clicks at 10, 15, 20, 25 (4 clicks).
        # Published at t=2h -> Age = 24h.
        # Weight = (1 + 4) * 2^(-24/12) = 5 * 0.25 = 1.25
        target_t = self.base_ns + int(pd.Timedelta(hours=26).value)
        pub_t = self.base_ns + int(pd.Timedelta(hours=2).value)
        weights = self.index.compute_null_weights([101], [pub_t], target_t, window_hours=24)
        self.assertAlmostEqual(weights[0], 1.25, places=5)


if __name__ == "__main__":
    unittest.main()
