"""
Unit tests for Discovery Headroom optimizer, common support index, and qualification gating.
"""

import unittest
import pandas as pd
import numpy as np

from src.headroom import (
    CommonSupportIndex,
    optimize_headroom_single_impression,
    HeadroomResult,
)


class TestHeadroom(unittest.TestCase):
    def test_qualification_gating(self):
        """Headroom optimization must be blocked if qualification_status != 'QUALIFIED'."""
        with self.assertRaises(PermissionError):
            optimize_headroom_single_impression(
                observed_slate=[1, 2],
                article_novelty_map={1: 0.5, 2: 0.5},
                article_relevance_map={1: 0.15, 2: 0.15},
                candidate_ids=[3],
                tau=0.10,
                qualification_status="NOT_QUALIFIED",
            )

    def test_common_support_index(self):
        """Articles must be seen by >= 3 distinct users in the 30-min window."""
        bucket = pd.Timestamp("2023-05-20T10:00:00Z")
        df = pd.DataFrame([
            {"_time_bucket": bucket, "user_id": 1, "_slate": [101, 102]},
            {"_time_bucket": bucket, "user_id": 2, "_slate": [101, 103]},
            {"_time_bucket": bucket, "user_id": 3, "_slate": [101, 104]},
            {"_time_bucket": bucket, "user_id": 4, "_slate": [102, 105]},
        ])
        cs = CommonSupportIndex(df)

        # 101 seen by users 1, 2, 3 (3 users >= 3)
        # 102 seen by users 1, 4 (2 users < 3)
        supported_3 = cs.get_supported_articles(bucket, min_distinct_users=3)
        self.assertEqual(supported_3, {101})

        # At threshold 2, both 101 and 102 are supported
        supported_2 = cs.get_supported_articles(bucket, min_distinct_users=2)
        self.assertEqual(supported_2, {101, 102})

    def test_headroom_optimizer_obvious_swap(self):
        """
        Controlled fixture:
        Observed slate of 4 items:
          Item 1: novelty=0.1, rel=0.20 (disc=0.1)
          Item 2: novelty=0.8, rel=0.20 (disc=0.8)
          Item 3: novelty=0.7, rel=0.20 (disc=0.7)
          Item 4: novelty=0.6, rel=0.20 (disc=0.6)
          Observed discovery = (0.1 + 0.8 + 0.7 + 0.6) / 4 = 0.55
          Observed relevance = 0.20

        Candidate pool:
          Candidate 101: novelty=0.9, rel=0.20 (r >= tau=0.10)
          Candidate 102: novelty=0.2, rel=0.20
          
        Swapping Item 1 out for Candidate 101:
          New discovery = (0.9 + 0.8 + 0.7 + 0.6) / 4 = 0.75
          Gain = 0.75 - 0.55 = 0.20
          Relevance retention = 100% >= 97%
        """
        tau = 0.10
        obs_slate = [1, 2, 3, 4]
        nov_map = {1: 0.1, 2: 0.8, 3: 0.7, 4: 0.6, 101: 0.9, 102: 0.2}
        rel_map = {1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20, 101: 0.20, 102: 0.20}

        res = optimize_headroom_single_impression(
            observed_slate=obs_slate,
            article_novelty_map=nov_map,
            article_relevance_map=rel_map,
            candidate_ids=[101, 102],
            tau=tau,
            qualification_status="QUALIFIED",
            relevance_tolerance=0.03,
            max_swaps=2,
        )

        self.assertEqual(res.n_swaps, 1)
        self.assertEqual(res.swapped_out_ids, [1])
        self.assertEqual(res.swapped_in_ids, [101])
        self.assertAlmostEqual(res.observed_discovery, 0.55)
        self.assertAlmostEqual(res.optimized_discovery, 0.75)
        self.assertAlmostEqual(res.discovery_gain, 0.20)
        self.assertAlmostEqual(res.relevance_retention_ratio, 1.0)

    def test_headroom_relevance_constraint_violation_prevention(self):
        """Candidate has high novelty but would violate the 97% relevance floor."""
        tau = 0.10
        # Slate has high relevance: 0.20 each, total=0.80. Min required total = 0.97 * 0.80 = 0.776
        # Max allowed relevance drop = 0.024
        obs_slate = [1, 2, 3, 4]
        # Candidate has novelty=0.99, but relevance=0.10 (drop = 0.10 > 0.024)
        nov_map = {1: 0.1, 2: 0.5, 3: 0.5, 4: 0.5, 999: 0.99}
        rel_map = {1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20, 999: 0.10}

        res = optimize_headroom_single_impression(
            observed_slate=obs_slate,
            article_novelty_map=nov_map,
            article_relevance_map=rel_map,
            candidate_ids=[999],
            tau=tau,
            qualification_status="QUALIFIED",
            relevance_tolerance=0.03,
            max_swaps=2,
        )
        # Cannot swap because 0.10 - 0.20 = -0.10 drop exceeds tolerance
        self.assertEqual(res.n_swaps, 0)
        self.assertAlmostEqual(res.discovery_gain, 0.0)


if __name__ == "__main__":
    unittest.main()
