"""
Unit tests for forward-chained relevance model, fold-local preprocessing,
timestamp tie-breaking, and tau calculation.
"""

import unittest
import numpy as np
import pandas as pd

from src.relevance import (
    make_forward_chain_blocks,
    run_forward_chained_oof,
    RelevancePipeline,
    compute_ndcg_at_k,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


class TestRelevance(unittest.TestCase):
    def setUp(self):
        # Create synthetic tabular dataset across 10 distinct timestamps
        # 100 impressions total, 10 items per impression -> 1000 rows
        np.random.seed(42)
        base = pd.Timestamp("2023-05-18T00:00:00Z")
        
        rows = []
        for imp_idx in range(100):
            # Timestamps: 10 impressions per hour
            hour = imp_idx // 10
            t = base + pd.Timedelta(hours=hour)
            t_ns = t.value
            
            # Slate of 10 items
            for item_idx in range(10):
                aid = 1000 + item_idx
                y = 1 if item_idx == 0 else 0  # 1 click per impression
                
                rows.append({
                    "impression_id": imp_idx,
                    "user_id": imp_idx % 20,
                    "article_id": aid,
                    "impression_time": t,
                    "_time_ns": t_ns,
                    "y": y,
                    "sim_to_history": float(np.random.uniform(0.1, 0.9)),
                    "category_affinity": float(np.random.uniform(0.0, 1.0)),
                    "topic_affinity": float(np.random.uniform(0.0, 1.0)),
                    "log_age_hours": float(np.random.uniform(0.5, 3.0)),
                    "log_clicks_24h": float(np.random.uniform(0.0, 5.0)),
                    "is_premium": int(item_idx % 3 == 0),
                    "is_subscriber": int((imp_idx % 20) % 2 == 0),
                    "premium_x_subscriber": int(item_idx % 3 == 0 and (imp_idx % 20) % 2 == 0),
                    "device_type": str(imp_idx % 3),
                    "time_of_day_sin": float(np.sin(2 * np.pi * hour / 24.0)),
                    "time_of_day_cos": float(np.cos(2 * np.pi * hour / 24.0)),
                    "pop_rec_weight": float(np.random.uniform(0.1, 5.0)),
                })
        self.df = pd.DataFrame(rows)

    def test_make_forward_chain_blocks_strict_time(self):
        blocks = make_forward_chain_blocks(self.df, n_blocks=5)
        self.assertEqual(len(blocks), 5)
        
        # Verify strict inequality between adjacent blocks
        for i in range(len(blocks) - 1):
            max_prev = blocks[i]["_time_ns"].max()
            min_next = blocks[i + 1]["_time_ns"].min()
            self.assertLess(
                max_prev, min_next,
                f"Block {i} max time must be strictly less than block {i+1} min time"
            )

    def test_run_forward_chained_oof(self):
        blocks = make_forward_chain_blocks(self.df, n_blocks=5)
        pooled_oof, diagnostics, tau = run_forward_chained_oof(blocks)
        
        # B1 was warm-up, so pooled_oof contains blocks 2, 3, 4, 5
        self.assertIn("oof_score", pooled_oof.columns)
        self.assertEqual(set(pooled_oof["oof_fold"].unique()), {2, 3, 4, 5})
        
        # Verify tau is exactly 20th percentile of clicked items
        clicked_oof = pooled_oof.loc[pooled_oof["y"] == 1, "oof_score"].to_numpy()
        expected_tau = float(np.percentile(clicked_oof, 20.0))
        self.assertAlmostEqual(tau, expected_tau, places=6)
        
        # Diagnostics sanity
        self.assertGreaterEqual(diagnostics["oof_auc"], 0.0)
        self.assertLessEqual(diagnostics["oof_auc"], 1.0)
        self.assertGreaterEqual(diagnostics["oof_ndcg"], 0.0)
        self.assertLessEqual(diagnostics["oof_ndcg"], 1.0)

    def test_ndcg_calculation(self):
        y_true = np.array([1, 0, 0, 0])
        # Perfect ranking
        score_perfect = np.array([0.9, 0.5, 0.3, 0.1])
        ndcg_perfect = compute_ndcg_at_k(y_true, score_perfect)
        self.assertAlmostEqual(ndcg_perfect, 1.0, places=5)
        
        # Inverted ranking
        score_worst = np.array([0.1, 0.3, 0.5, 0.9])
        ndcg_worst = compute_ndcg_at_k(y_true, score_worst)
        self.assertLess(ndcg_worst, 1.0)
        self.assertGreater(ndcg_worst, 0.0)


if __name__ == "__main__":
    unittest.main()
