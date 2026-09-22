"""
Unit test verifying reproducible results under fixed random seed 20260919.
"""

import unittest
import numpy as np
import pandas as pd

from src.config import SEED
from src.relevance import make_forward_chain_blocks, run_forward_chained_oof


class TestReproducibility(unittest.TestCase):
    def test_deterministic_forward_chaining(self):
        np.random.seed(SEED)
        base = pd.Timestamp("2023-05-18T00:00:00Z")
        
        rows = []
        for imp_idx in range(60):
            hour = imp_idx // 10
            t = base + pd.Timedelta(hours=hour)
            t_ns = t.value
            for item_idx in range(5):
                aid = 1000 + item_idx
                y = 1 if item_idx == 0 else 0
                rows.append({
                    "impression_id": imp_idx,
                    "user_id": imp_idx % 10,
                    "article_id": aid,
                    "impression_time": t,
                    "_time_ns": t_ns,
                    "y": y,
                    "sim_to_history": float(np.random.uniform(0.1, 0.9)),
                    "category_affinity": float(np.random.uniform(0.0, 1.0)),
                    "topic_affinity": float(np.random.uniform(0.0, 1.0)),
                    "log_age_hours": float(np.random.uniform(0.5, 3.0)),
                    "log_clicks_24h": float(np.random.uniform(0.0, 5.0)),
                    "is_premium": int(item_idx % 2 == 0),
                    "is_subscriber": int((imp_idx % 10) % 2 == 0),
                    "premium_x_subscriber": int(item_idx % 2 == 0 and (imp_idx % 10) % 2 == 0),
                    "device_type": str(imp_idx % 2),
                    "time_of_day_sin": float(np.sin(2 * np.pi * hour / 24.0)),
                    "time_of_day_cos": float(np.cos(2 * np.pi * hour / 24.0)),
                    "pop_rec_weight": float(np.random.uniform(0.1, 5.0)),
                })
        df = pd.DataFrame(rows)

        # Run 1
        blocks1 = make_forward_chain_blocks(df, n_blocks=5)
        oof1, diag1, tau1 = run_forward_chained_oof(blocks1)

        # Run 2
        blocks2 = make_forward_chain_blocks(df, n_blocks=5)
        oof2, diag2, tau2 = run_forward_chained_oof(blocks2)

        # Check identical results
        self.assertEqual(tau1, tau2)
        self.assertEqual(diag1["oof_auc"], diag2["oof_auc"])
        self.assertEqual(diag1["oof_ndcg"], diag2["oof_ndcg"])
        np.testing.assert_allclose(oof1["oof_score"].to_numpy(), oof2["oof_score"].to_numpy())


if __name__ == "__main__":
    unittest.main()
