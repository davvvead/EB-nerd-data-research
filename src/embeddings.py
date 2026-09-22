"""
Embedding ingestion, pre-normalization, and fast top-k novelty computation.
Supports contrastive (primary) and multilingual BERT (robustness) vectors.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ROOT_DIR,
    CACHE_EMB_DIR,
    PRIMARY_SEMANTIC_REPRESENTATION,
    ROBUSTNESS_SEMANTIC_REPRESENTATION,
    EMBEDDING_DIMENSION,
    PRIMARY_HISTORY_K,
    SENSITIVITY_HISTORY_K,
    ensure_directories,
)


def detect_article_id(df: pd.DataFrame) -> str:
    """Identify the article ID column."""
    for c in ["article_id", "article_ids", "id"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not identify article ID column among {list(df.columns)}")


def detect_vectors(df: pd.DataFrame, id_col: str) -> Tuple[np.ndarray, str]:
    """Detect and extract embedding vectors as float32 matrix."""
    other = [c for c in df.columns if c != id_col]
    
    # List or array column
    for c in other:
        vals = df[c].dropna().head(5).tolist()
        if vals and any(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
            X = np.vstack([np.asarray(v, dtype=np.float32) for v in df[c]])
            return X, c
            
    # Object column with arrays
    for c in other:
        if df[c].dtype == object:
            sample = df[c].dropna().head(3)
            Xs = [np.asarray(v, dtype=np.float32) for v in sample]
            if Xs and all(x.ndim == 1 and len(x) > 1 for x in Xs):
                X = np.vstack([np.asarray(v, dtype=np.float32) for v in df[c]])
                return X, c
                
    # Numeric dimension columns
    num = [c for c in other if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) >= 10:
        X = df[num].to_numpy(dtype=np.float32)
        return X, f"numeric_{len(num)}_cols"
        
    raise ValueError(f"Could not detect embedding vectors in columns {list(df.columns)}")


class EmbeddingStore:
    """
    Stores pre-normalized article embeddings with fast indexed lookups.
    """
    def __init__(self, name: str, ids: np.ndarray, vectors: np.ndarray):
        self.name = name
        self.ids = np.asarray(ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.id_to_idx = {aid: idx for idx, aid in enumerate(self.ids)}
        
        # Verify normalization
        norms = np.linalg.norm(self.vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Vectors in {name} are not properly normalized")

    def __len__(self) -> int:
        return len(self.ids)

    def contains(self, article_id: Any) -> bool:
        return article_id in self.id_to_idx

    def get_vector(self, article_id: Any) -> Optional[np.ndarray]:
        idx = self.id_to_idx.get(article_id)
        if idx is not None:
            return self.vectors[idx]
        return None

    def get_vectors_for_ids(self, article_ids: List[Any]) -> Tuple[np.ndarray, List[Any]]:
        """Retrieve sub-matrix of vectors for given article IDs, ignoring missing."""
        valid_indices = []
        valid_ids = []
        for aid in article_ids:
            idx = self.id_to_idx.get(aid)
            if idx is not None:
                valid_indices.append(idx)
                valid_ids.append(aid)
        if not valid_indices:
            return np.empty((0, self.vectors.shape[1]), dtype=np.float32), []
        return self.vectors[valid_indices], valid_ids

    def compute_novelty_multi_k(
        self,
        history_ids: List[Any],
        candidate_ids: List[Any],
        k_values: Tuple[int, ...] = (3, 5, 10),
    ) -> Dict[int, np.ndarray]:
        """
        Compute novelty = 1.0 - mean(top-k cosine sim) for multiple k values simultaneously.
        history_ids: list of user's historical clicked article IDs.
        candidate_ids: list of candidate article IDs.
        Returns dict mapping k -> np.ndarray of novelty values of length len(candidate_ids).
        """
        V_H, _ = self.get_vectors_for_ids(history_ids)
        V_C, valid_cand_ids = self.get_vectors_for_ids(candidate_ids)
        
        n_cands = len(candidate_ids)
        out = {k: np.full(n_cands, np.nan, dtype=np.float32) for k in k_values}
        
        if len(V_H) == 0 or len(V_C) == 0:
            return out

        # Map candidate_ids back to output array positions
        cand_pos_map = {aid: idx for idx, aid in enumerate(candidate_ids)}
        valid_positions = np.array([cand_pos_map[aid] for aid in valid_cand_ids], dtype=int)

        # Cosine similarity matrix: shape (n_hist, n_valid_cands)
        # Both V_H and V_C are pre-normalized, so dot product is exact cosine similarity.
        S = np.matmul(V_H, V_C.T)  # (m, L)
        m = S.shape[0]

        for k in k_values:
            if m <= k:
                mean_sim = S.mean(axis=0)
            else:
                # Fast top-k selection along axis 0
                top_k = np.partition(S, -k, axis=0)[-k:, :]
                mean_sim = top_k.mean(axis=0)
            
            novelty = 1.0 - mean_sim
            out[k][valid_positions] = novelty

        return out


def build_or_load_embedding_store(
    name: str = "contrastive",
    articles_df: Optional[pd.DataFrame] = None,
    force_rebuild: bool = False,
) -> EmbeddingStore:
    """Load or generate pre-normalized embedding store."""
    ensure_directories()
    norm_path = CACHE_EMB_DIR / f"{name}_norm.npy"
    ids_path = CACHE_EMB_DIR / f"{name}_ids.npy"
    
    if not force_rebuild and norm_path.exists() and ids_path.exists():
        ids = np.load(ids_path, allow_pickle=True)
        vectors = np.load(norm_path)
        return EmbeddingStore(name, ids, vectors)

    # Determine file
    if name == "contrastive":
        file_name = PRIMARY_SEMANTIC_REPRESENTATION
    elif name == "bert":
        file_name = ROBUSTNESS_SEMANTIC_REPRESENTATION
    else:
        raise ValueError(f"Unknown embedding representation: {name}")

    path = ROOT_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    df = pd.read_parquet(path)
    aid_col = detect_article_id(df)
    X, col_name = detect_vectors(df, aid_col)
    ids = df[aid_col].to_numpy()

    # Filter finite vectors
    finite = np.isfinite(X).all(axis=1)
    ids, X = ids[finite], X[finite]

    # Filter to articles in articles table if provided
    if articles_df is not None:
        valid_set = set(articles_df["article_id"].dropna().unique())
        keep = np.array([i in valid_set for i in ids])
        ids, X = ids[keep], X[keep]

    # Pre-normalize vectors to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    X_norm = (X / norms).astype(np.float32)

    # Save cache
    np.save(norm_path, X_norm)
    np.save(ids_path, ids)

    return EmbeddingStore(name, ids, X_norm)
