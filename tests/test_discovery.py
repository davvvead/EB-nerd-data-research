"""
Unit tests for discovery metrics, relevance floor gating, and subscriber choice rules.
"""

import unittest
import numpy as np

from src.discovery import (
    compute_item_discovery,
    compute_slate_discovery_metrics,
    compute_choice_stage_metrics,
)


class TestDiscoveryMetrics(unittest.TestCase):
    def test_item_discovery_formula(self):
        tau = 0.10
        # Item 1: rel >= tau => disc = novelty
        d1 = compute_item_discovery(novelty=0.85, predicted_relevance=0.15, tau=tau)
        self.assertAlmostEqual(d1, 0.85)

        # Item 2: rel < tau => disc = 0.0
        d2 = compute_item_discovery(novelty=0.85, predicted_relevance=0.08, tau=tau)
        self.assertAlmostEqual(d2, 0.0)

        # Array version
        novs = np.array([0.5, 0.8, 0.2])
        rels = np.array([0.12, 0.05, 0.20])
        disc_arr = compute_item_discovery(novs, rels, tau=tau)
        np.testing.assert_allclose(disc_arr, [0.5, 0.0, 0.2])

    def test_slate_discovery_decomposition(self):
        tau = 0.10
        novs = np.array([0.6, 0.8, 0.4, 0.2])
        rels = np.array([0.15, 0.05, 0.20, 0.08])
        # Items 0 & 2 are above floor (disc = 0.6, 0.4; mean = 0.5 among above floor)
        # Items 1 & 3 are below floor (disc = 0.0, 0.0)
        # Total slate discovery = (0.6 + 0.0 + 0.4 + 0.0) / 4 = 0.25
        # Relevance coverage = 2 / 4 = 0.50
        # Conditional novelty = (0.6 + 0.4) / 2 = 0.50
        metrics = compute_slate_discovery_metrics(novs, rels, tau=tau)
        self.assertAlmostEqual(metrics["discovery"], 0.25)
        self.assertAlmostEqual(metrics["relevance_coverage"], 0.50)
        self.assertAlmostEqual(metrics["conditional_novelty"], 0.50)

    def test_choice_stage_subscriber_vs_nonsubscriber(self):
        tau = 0.10
        # Slate: items 1 (standard), 2 (premium), 3 (standard)
        slate = [1, 2, 3]
        clicked = [2]  # Clicked the premium item!
        
        nov_map = {1: 0.5, 2: 0.8, 3: 0.3}
        rel_map = {1: 0.15, 2: 0.20, 3: 0.05}
        prem_map = {1: False, 2: True, 3: False}

        # Case A: User is subscriber
        sub_res = compute_choice_stage_metrics(
            slate_article_ids=slate,
            clicked_article_ids=clicked,
            article_novelty_map=nov_map,
            article_relevance_map=rel_map,
            article_premium_map=prem_map,
            is_subscriber=True,
            tau=tau,
        )
        self.assertTrue(sub_res["has_eligible_consumed"])
        self.assertEqual(sub_res["n_choosable"], 3)
        self.assertEqual(sub_res["n_consumed"], 1)
        self.assertAlmostEqual(sub_res["consumed_discovery"], 0.8)

        # Case B: User is non-subscriber
        # Premium item 2 is excluded from both choosable and consumed!
        # Choosable: [1, 3]. Consumed: empty!
        nonsub_res = compute_choice_stage_metrics(
            slate_article_ids=slate,
            clicked_article_ids=clicked,
            article_novelty_map=nov_map,
            article_relevance_map=rel_map,
            article_premium_map=prem_map,
            is_subscriber=False,
            tau=tau,
        )
        self.assertFalse(nonsub_res["has_eligible_consumed"])
        self.assertEqual(nonsub_res["n_choosable"], 2)
        self.assertEqual(nonsub_res["n_consumed"], 0)
        self.assertTrue(np.isnan(nonsub_res["choice_discovery_gap"]))


if __name__ == "__main__":
    unittest.main()
