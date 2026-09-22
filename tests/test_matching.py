"""
Unit tests for deterministic 1:1 nearest-neighbor matching algorithm.
"""

import unittest
import pandas as pd
import numpy as np

from src.matching import perform_deterministic_matching
from src.data import epoch_ns


class TestMatching(unittest.TestCase):
    def setUp(self):
        # Construct synthetic dataframe with 2 strata
        # Stratum 1: bucket 1, mobile (1), subscriber=False, single, len 1, hist_bin "15-49"
        base_t = pd.Timestamp("2023-05-20T10:00:00Z")
        
        rows = [
            # Low impressions (2 impressions)
            {
                "impression_id": 10,
                "impression_time": base_t + pd.Timedelta(minutes=5),
                "_time_ns": epoch_ns(pd.Series([base_t + pd.Timedelta(minutes=5)]))[0],
                "_slate_size": 10,
                "device_type": 1,
                "is_subscriber": False,
                "_session_stage": "single",
                "_session_len_bin": "1",
                "history_length_bin": "15-49",
                "_time_bucket": base_t,
                "breadth_cohort": "Low",
                "user_id": 1,
            },
            {
                "impression_id": 20,
                "impression_time": base_t + pd.Timedelta(minutes=10),
                "_time_ns": epoch_ns(pd.Series([base_t + pd.Timedelta(minutes=10)]))[0],
                "_slate_size": 12,
                "device_type": 1,
                "is_subscriber": False,
                "_session_stage": "single",
                "_session_len_bin": "1",
                "history_length_bin": "15-49",
                "_time_bucket": base_t,
                "breadth_cohort": "Low",
                "user_id": 2,
            },
            # High candidates (3 candidates)
            # Candidate 101: slate 10 (exact match for imp 10, slate_diff=0), time min 6
            # Candidate 102: slate 11 (slate_diff=1 for imp 10, slate_diff=1 for imp 20)
            # Candidate 103: slate 15 (slate_diff=3, outside caliper for imp 20, caliper <= 1)
            {
                "impression_id": 101,
                "impression_time": base_t + pd.Timedelta(minutes=6),
                "_time_ns": epoch_ns(pd.Series([base_t + pd.Timedelta(minutes=6)]))[0],
                "_slate_size": 10,
                "device_type": 1,
                "is_subscriber": False,
                "_session_stage": "single",
                "_session_len_bin": "1",
                "history_length_bin": "15-49",
                "_time_bucket": base_t,
                "breadth_cohort": "High",
                "user_id": 101,
            },
            {
                "impression_id": 102,
                "impression_time": base_t + pd.Timedelta(minutes=9),
                "_time_ns": epoch_ns(pd.Series([base_t + pd.Timedelta(minutes=9)]))[0],
                "_slate_size": 11,
                "device_type": 1,
                "is_subscriber": False,
                "_session_stage": "single",
                "_session_len_bin": "1",
                "history_length_bin": "15-49",
                "_time_bucket": base_t,
                "breadth_cohort": "High",
                "user_id": 102,
            },
            {
                "impression_id": 103,
                "impression_time": base_t + pd.Timedelta(minutes=8),
                "_time_ns": epoch_ns(pd.Series([base_t + pd.Timedelta(minutes=8)]))[0],
                "_slate_size": 15,
                "device_type": 1,
                "is_subscriber": False,
                "_session_stage": "single",
                "_session_len_bin": "1",
                "history_length_bin": "15-49",
                "_time_bucket": base_t,
                "breadth_cohort": "High",
                "user_id": 103,
            },
        ]
        self.df = pd.DataFrame(rows)

    def test_deterministic_matching_execution(self):
        res = perform_deterministic_matching(self.df, caliper=1)

        self.assertEqual(res.eligible_low_count, 2)
        self.assertEqual(res.eligible_high_count, 3)
        self.assertEqual(res.matched_pairs_count, 2)
        self.assertAlmostEqual(res.low_match_rate, 1.0)
        self.assertAlmostEqual(res.high_match_rate, 2/3)

        pairs = res.pairs_df.set_index("low_impression_id")
        
        # Low 10 (earlier time) matched first:
        # Candidates within caliper: 101 (slate 10, diff 0), 102 (slate 11, diff 1)
        # 101 has smallest slate diff (0), so 10 must match 101!
        self.assertEqual(pairs.loc[10, "high_impression_id"], 101)

        # Low 20 matched second:
        # Candidate 101 is already matched without replacement.
        # Candidate 102 has slate 11 (diff 1 <= caliper).
        # Candidate 103 has slate 15 (diff 3 > caliper, filtered out).
        # So Low 20 must match 102!
        self.assertEqual(pairs.loc[20, "high_impression_id"], 102)


if __name__ == "__main__":
    unittest.main()
