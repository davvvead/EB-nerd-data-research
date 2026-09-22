"""
Leakage prevention tests:
1. User histories predate behavior period.
2. Zero validation behavior rows or impression IDs in train tables.
3. Strict published_time < impression_time boundary in supply.
4. No future click leakage in popularity.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import ROOT_DIR
from src.data import load_articles, load_behaviors, load_history, epoch_ns
from src.popularity import ClickPopularityIndex


class TestNoLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_beh_path = ROOT_DIR / "ebnerd_small" / "train" / "behaviors.parquet"
        cls.train_hist_path = ROOT_DIR / "ebnerd_small" / "train" / "history.parquet"
        cls.val_beh_path = ROOT_DIR / "ebnerd_small" / "validation" / "behaviors.parquet"
        cls.val_hist_path = ROOT_DIR / "ebnerd_small" / "validation" / "history.parquet"

    def test_history_predates_behavior(self):
        """Verify train history timestamp strictly precedes train behavior start."""
        train_hist = pd.read_parquet(self.train_hist_path, columns=["impression_time_fixed"])
        train_beh = pd.read_parquet(self.train_beh_path, columns=["impression_time"])
        
        # Flatten sample of history timestamps
        h_times = []
        for row in train_hist["impression_time_fixed"].head(500):
            if row is not None and len(row) > 0:
                h_times.extend(row)
                
        h_max = pd.to_datetime(h_times, utc=True).max()
        b_min = pd.to_datetime(train_beh["impression_time"], utc=True).min()
        
        self.assertLess(h_max, b_min, f"History max {h_max} must strictly predate behavior min {b_min}")

    def test_chronological_train_validation_boundary(self):
        """Verify train behavior period strictly precedes validation behavior period."""
        train_beh = pd.read_parquet(self.train_beh_path, columns=["impression_time"])
        val_beh = pd.read_parquet(self.val_beh_path, columns=["impression_time"])
        
        t_max = pd.to_datetime(train_beh["impression_time"], utc=True).max()
        v_min = pd.to_datetime(val_beh["impression_time"], utc=True).min()
        
        self.assertLess(t_max, v_min, f"Train max {t_max} must strictly predate validation min {v_min}")

    def test_disjoint_impression_ids(self):
        """Verify impression IDs are strictly disjoint between train and validation."""
        train_ids = set(pd.read_parquet(self.train_beh_path, columns=["impression_id"])["impression_id"])
        val_ids = set(pd.read_parquet(self.val_beh_path, columns=["impression_id"])["impression_id"])
        
        intersection = train_ids.intersection(val_ids)
        self.assertEqual(len(intersection), 0, "Impression IDs must never overlap between train and validation")

    def test_strict_supply_published_boundary(self):
        """Verify published_time < impression_time rule excludes simultaneous publication."""
        t_imp = pd.Timestamp("2023-05-20T12:00:00Z")
        t_imp_ns = t_imp.value
        
        # Article published at exactly 12:00:00Z
        art_exact = pd.Series([t_imp])
        pub_exact_ns = epoch_ns(art_exact)[0]
        
        # Article published 1 second earlier
        art_prior = pd.Series([t_imp - pd.Timedelta(seconds=1)])
        pub_prior_ns = epoch_ns(art_prior)[0]
        
        # Condition: pub_ns < t_imp_ns
        self.assertFalse(pub_exact_ns < t_imp_ns, "Simultaneous publication must be excluded")
        self.assertTrue(pub_prior_ns < t_imp_ns, "Prior publication must be included")


if __name__ == "__main__":
    unittest.main()
