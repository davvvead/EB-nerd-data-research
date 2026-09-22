"""
Unit tests for embedding store, vector normalization, alignment, and top-k novelty.
"""

import unittest
import numpy as np

from src.embeddings import EmbeddingStore


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        # Create small 4-dimensional synthetic embeddings for 6 articles
        # A1: [1, 0, 0, 0]
        # A2: [0, 1, 0, 0]
        # A3: [0.7071, 0.7071, 0, 0]
        # A4: [0, 0, 1, 0]
        # A5: [0, 0, 0, 1]
        # A6: [0.5, 0.5, 0.5, 0.5]
        self.ids = np.array([1, 2, 3, 4, 5, 6])
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ], dtype=np.float32)
        self.store = EmbeddingStore("test", self.ids, vectors)

    def test_normalization_and_lookup(self):
        v1 = self.store.get_vector(1)
        self.assertIsNotNone(v1)
        self.assertAlmostEqual(float(np.linalg.norm(v1)), 1.0, places=5)
        self.assertIsNone(self.store.get_vector(999))

    def test_novelty_top_k(self):
        # User history: A1 and A2 (m=2)
        # Candidate: A1. Cosine sim with A1 is 1.0, with A2 is 0.0.
        # For k=1: top-1 sim is 1.0 -> novelty = 0.0
        # For k=2: top-2 mean sim is (1.0 + 0.0)/2 = 0.5 -> novelty = 0.5
        history = [1, 2]
        cands = [1, 4]  # A4 has dot product 0 with both A1 and A2
        
        res = self.store.compute_novelty_multi_k(history, cands, k_values=(1, 2))
        
        # A1 under k=1
        self.assertAlmostEqual(float(res[1][0]), 0.0, places=5)
        # A1 under k=2
        self.assertAlmostEqual(float(res[2][0]), 0.5, places=5)
        # A4 under k=1 and k=2: sim is 0 -> novelty = 1.0
        self.assertAlmostEqual(float(res[1][1]), 1.0, places=5)
        self.assertAlmostEqual(float(res[2][1]), 1.0, places=5)

    def test_k_greater_than_history(self):
        # When k > history length m, mean should average all m historical items
        history = [1, 2]
        cands = [3]  # A3 has sim 1/sqrt(2) ~ 0.7071 with both A1 and A2
        res = self.store.compute_novelty_multi_k(history, cands, k_values=(5,))
        expected_sim = 1.0 / np.sqrt(2)
        expected_nov = 1.0 - expected_sim
        self.assertAlmostEqual(float(res[5][0]), expected_nov, places=4)


if __name__ == "__main__":
    unittest.main()
