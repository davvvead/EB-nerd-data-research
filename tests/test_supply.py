"""
Unit tests for observable supply indexing, retrospective boundary enforcement,
and counterfactual analysis eligibility.
"""

import unittest
import numpy as np
import pandas as pd

from src.supply import ObservableSupplyIndex, HOURS_TO_NS
from src.data import epoch_ns


class TestObservableSupply(unittest.TestCase):
    def setUp(self):
        # Base timestamp: 2023-05-20 12:00:00 UTC
        self.base_time = pd.Timestamp("2023-05-20T12:00:00Z")
        self.base_ns = int(self.base_time.value)

        # Construct synthetic articles spanning boundaries
        # W_24h = 24 * 3600 * 1e9 ns
        w24 = int(24 * HOURS_TO_NS)
        
        # 101: exactly at t_imp - 24h (SHOULD BE INCLUDED)
        # 102: inside window (t_imp - 12h) (SHOULD BE INCLUDED)
        # 103: 1 ns before t_imp (t_imp - 1ns) (SHOULD BE INCLUDED)
        # 104: exactly at t_imp (t_imp) (MUST BE EXCLUDED)
        # 105: 1 ns after t_imp (t_imp + 1ns) (MUST BE EXCLUDED)
        # 106: 1 ns before t_imp - 24h (t_imp - 24h - 1ns) (MUST BE EXCLUDED)
        # 107: inside 48h window but outside 24h window (t_imp - 30h)
        articles_data = [
            {"article_id": 101, "published_time": pd.to_datetime(self.base_ns - w24, unit="ns", utc=True)},
            {"article_id": 102, "published_time": pd.to_datetime(self.base_ns - w24 // 2, unit="ns", utc=True)},
            {"article_id": 103, "published_time": pd.to_datetime(self.base_ns - 1, unit="ns", utc=True)},
            {"article_id": 104, "published_time": pd.to_datetime(self.base_ns, unit="ns", utc=True)},
            {"article_id": 105, "published_time": pd.to_datetime(self.base_ns + 1, unit="ns", utc=True)},
            {"article_id": 106, "published_time": pd.to_datetime(self.base_ns - w24 - 1, unit="ns", utc=True)},
            {"article_id": 107, "published_time": pd.to_datetime(self.base_ns - int(30 * HOURS_TO_NS), unit="ns", utc=True)},
        ]
        self.articles_df = pd.DataFrame(articles_data)
        self.index = ObservableSupplyIndex(self.articles_df)

    def test_strict_temporal_boundary(self):
        """Test strict predicate: t_imp - 24h <= t_pub < t_imp."""
        supply_ids = set(self.index.get_supply_ids(self.base_ns, window_hours=24))
        
        # Must include 101 (exact start), 102 (middle), 103 (1 ns before end)
        self.assertIn(101, supply_ids)
        self.assertIn(102, supply_ids)
        self.assertIn(103, supply_ids)

        # Must exclude 104 (published exactly at impression time)
        self.assertNotIn(104, supply_ids)
        # Must exclude 105 (future article)
        self.assertNotIn(105, supply_ids)
        # Must exclude 106 (published 1 ns before start of window)
        self.assertNotIn(106, supply_ids)
        # Must exclude 107 (published 30h ago)
        self.assertNotIn(107, supply_ids)

    def test_sensitivity_windows(self):
        """Test 12h and 48h robustness windows."""
        # 12h window: only 102 and 103
        supply_12h = set(self.index.get_supply_ids(self.base_ns, window_hours=12))
        self.assertEqual(supply_12h, {102, 103})

        # 48h window: includes 107 as well as 101, 102, 103
        supply_48h = set(self.index.get_supply_ids(self.base_ns, window_hours=48))
        self.assertIn(107, supply_48h)
        self.assertIn(101, supply_48h)
        self.assertNotIn(104, supply_48h)  # Still strictly before t_imp

    def test_counterfactual_eligibility(self):
        """Test counterfactual analysis eligibility conditions."""
        # Baseline valid: slate_size=10, supply_count=30 (3x), history=15 => True
        self.assertTrue(self.index.check_counterfactual_eligibility(
            slate_size=10, supply_count=30, history_length=15
        ))

        # Ineligible if supply < 3x slate
        self.assertFalse(self.index.check_counterfactual_eligibility(
            slate_size=10, supply_count=29, history_length=15
        ))

        # Ineligible if slate < 5
        self.assertFalse(self.index.check_counterfactual_eligibility(
            slate_size=4, supply_count=50, history_length=15
        ))

        # Ineligible if slate > 26
        self.assertFalse(self.index.check_counterfactual_eligibility(
            slate_size=27, supply_count=100, history_length=15
        ))

        # Ineligible if history < 15
        self.assertFalse(self.index.check_counterfactual_eligibility(
            slate_size=10, supply_count=50, history_length=14
        ))


if __name__ == "__main__":
    unittest.main()
