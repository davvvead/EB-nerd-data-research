"""
Unit tests for validation pipeline components:
- Frozen train-derived user profiling without leakage.
- Time-safe popularity across train/validation boundary.
- Model qualification evaluation.
- Authorization guard on 03_run_validation_analysis.py.
"""

import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    ROOT_DIR,
    BREADTH_TERTILE_LOW_TO_MED,
    BREADTH_TERTILE_MED_TO_HIGH,
    VALIDATION_HISTORY_PATH,
    TRAIN_HISTORY_PATH,
)
from src.data import epoch_ns
from src.popularity import ClickPopularityIndex
from src.validation import (
    build_validation_user_profiles,
    evaluate_relevance_qualification,
    validate_history_source,
    load_validation_user_profiles,
)


class TestValidationPipeline(unittest.TestCase):
    def test_validation_profile_source_guard(self):
        """The validation profile loader must strictly enforce validation history and reject train history."""
        # 1. Real validation history source must pass validation checks
        val_info = validate_history_source(VALIDATION_HISTORY_PATH)
        self.assertTrue(val_info["verified"])
        self.assertEqual(val_info["distinct_users"], 15342)
        self.assertAlmostEqual(val_info["span_days"], 21.0, delta=0.1)
        self.assertIn("2023-05-04", val_info["min_time"])
        self.assertIn("2023-05-25", val_info["max_time"])

        # 2. Substituting TRAIN history must fail with ValueError
        with self.assertRaises(ValueError) as ctx:
            validate_history_source(TRAIN_HISTORY_PATH)
        self.assertIn("TRAIN history", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            load_validation_user_profiles(history_path=TRAIN_HISTORY_PATH)
        self.assertIn("TRAIN history", str(ctx.exception))

        # 3. Arbitrary path must fail with ValueError
        with self.assertRaises(ValueError):
            validate_history_source("some/other/path/history.parquet")
    def test_validation_profiling_uses_frozen_train_cutoffs(self):
        """Validation profiles must use frozen train tertile cutoffs, not recalculate."""
        # 3 synthetic users with histories giving specific entropies
        articles_df = pd.DataFrame([
            {"article_id": 1, "category": 1, "topics": ["news"]},
            {"article_id": 2, "category": 2, "topics": ["sport"]},
            {"article_id": 3, "category": 3, "topics": ["finance"]},
        ])
        
        # User 1: 20 clicks all category 1 -> entropy = 0.0 <= LOW_TO_MED -> "Low"
        # User 2: 20 clicks spread evenly across categories 1, 2, 3 -> entropy = ln(3) ~ 1.0986
        # Wait, ln(3) ~ 1.0986 is still <= 1.4638!
        # To get entropy > 1.5905, we need e.g. 6 categories: ln(6) ~ 1.7917 > 1.5905!
        multi_cat_articles = pd.DataFrame([
            {"article_id": i, "category": i, "topics": [f"top_{i}"]}
            for i in range(1, 10)
        ])
        
        history_df = pd.DataFrame([
            # Low: all category 1
            {"user_id": 1, "article_id_fixed": [1] * 20},
            # High: 3 clicks each across 7 categories -> entropy = ln(7) ~ 1.9459 > 1.5905
            {"user_id": 2, "article_id_fixed": [1, 2, 3, 4, 5, 6, 7] * 3},
        ])

        profiles = build_validation_user_profiles(history_df, multi_cat_articles, min_history_len=15)

        self.assertEqual(profiles[1].breadth_cohort.lower(), "low")
        self.assertEqual(profiles[2].breadth_cohort.lower(), "high")

    def test_time_safe_popularity_boundary_train_to_validation(self):
        """Popularity index must correctly query across boundary with strict t_click < t."""
        t_split = pd.Timestamp("2023-05-25T07:00:00Z")
        t_split_ns = int(t_split.value)

        # Train clicks (before split)
        train_bh = pd.DataFrame([
            {
                "impression_time": t_split - pd.Timedelta(hours=2),
                "_time_ns": t_split_ns - int(2 * 3.6e12),
                "article_ids_clicked": [101],
            }
        ])
        # Validation clicks (at/after split)
        val_bh = pd.DataFrame([
            {
                "impression_time": t_split + pd.Timedelta(hours=1),
                "_time_ns": t_split_ns + int(1 * 3.6e12),
                "article_ids_clicked": [101],
            },
            {
                "impression_time": t_split + pd.Timedelta(hours=3),
                "_time_ns": t_split_ns + int(3 * 3.6e12),
                "article_ids_clicked": [101],
            }
        ])

        comb_bh = pd.concat([train_bh, val_bh], ignore_index=True)
        pop_idx = ClickPopularityIndex.from_behaviors(comb_bh)

        # Query 1: Exactly at validation click time (t_split + 1h)
        # MUST EXCLUDE the click at t_split + 1h (same-time click) and t_split + 3h (future)
        # MUST INCLUDE the train click at t_split - 2h (which is 3h prior <= 24h)
        t_q1 = t_split_ns + int(1 * 3.6e12)
        cnt1 = pop_idx.get_prior_clicks(101, t_q1, window_hours=24)
        self.assertEqual(cnt1, 1)

        # Query 2: At t_split + 2h
        # Must include train click (t_split - 2h) and first val click (t_split + 1h)
        t_q2 = t_split_ns + int(2 * 3.6e12)
        cnt2 = pop_idx.get_prior_clicks(101, t_q2, window_hours=24)
        self.assertEqual(cnt2, 2)

        # Query 3: At t_split + 23h
        # Train click was at t_split - 2h => age = 25h > 24h window => train click dropped!
        t_q3 = t_split_ns + int(23 * 3.6e12)
        cnt3 = pop_idx.get_prior_clicks(101, t_q3, window_hours=24)
        # Only the two validation clicks (at +1h and +3h) remain
        self.assertEqual(cnt3, 2)

    def test_relevance_qualification_logic(self):
        """Test pass/fail evaluation against AUC >= 0.60 and lift >= 0.01."""
        imp_ids = np.array([1, 1, 2, 2, 3, 3])
        y_true = np.array([1, 0, 1, 0, 0, 1])

        # Model 1: High AUC and lift
        score_good = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.7])
        score_base = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        res_pass = evaluate_relevance_qualification(imp_ids, y_true, score_good, score_base)
        self.assertTrue(res_pass.is_qualified)
        self.assertEqual(res_pass.status_label, "QUALIFIED")

        # Model 2: Sub-floor AUC
        score_poor = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1])
        res_fail = evaluate_relevance_qualification(imp_ids, y_true, score_poor, score_base)
        self.assertFalse(res_fail.is_qualified)
        self.assertEqual(res_fail.status_label, "NOT_QUALIFIED")

    def test_validation_authorization_guard(self):
        """Script 03 must abort without --authorize-heldout-run."""
        script_path = ROOT_DIR / "scripts" / "03_run_validation_analysis.py"
        python_bin = sys.executable

        # 1. Run without authorization -> Must fail with exit code 1
        res_abort = subprocess.run(
            [python_bin, str(script_path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_abort.returncode, 1)
        self.assertIn("ABORT: REAL HELD-OUT VALIDATION DATA IS LOCKED", res_abort.stderr)

        # 2. Run with --dry-run -> Must succeed with exit code 0
        res_dry = subprocess.run(
            [python_bin, str(script_path), "--dry-run"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_dry.returncode, 0)
        self.assertIn("Dry run completed successfully", res_dry.stdout)


if __name__ == "__main__":
    unittest.main()
