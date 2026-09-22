"""
Data loading, schema validation, and feature preprocessing utilities.
Ensures nanosecond timestamp resolution and zero leakage.
"""

from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ROOT_DIR,
    ELIGIBLE_SLATE_SIZE_MIN,
    ELIGIBLE_SLATE_SIZE_MAX,
    MATCH_TIME_BUCKET_MINUTES,
)


def norm_list(v: Any) -> List[Any]:
    """Normalize array-like or serialized representations into a flat Python list."""
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    if isinstance(v, (list, tuple, np.ndarray, pd.Series, set)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            q = ast.literal_eval(s)
            if isinstance(q, (list, tuple, np.ndarray, set)):
                return list(q)
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return [v]


def epoch_ns(series: pd.Series) -> np.ndarray:
    """
    Return UTC epoch nanoseconds robustly even if source dtype is datetime64[us, UTC].
    CRITICAL: Forces datetime resolution to nanoseconds using as_unit('ns').
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return pd.DatetimeIndex(s).as_unit("ns").asi8


def timestamp_selftest() -> Dict[str, Any]:
    """Verify nanosecond timestamp arithmetic with a 24h delta check."""
    test = pd.Series(pd.array(
        ["2023-05-18T07:00:00Z", "2023-05-19T07:00:00Z"],
        dtype="datetime64[us, UTC]"
    ))
    vals = epoch_ns(test)
    observed = int(vals[1] - vals[0])
    expected = int(pd.Timedelta("24h").value)
    if observed != expected:
        raise RuntimeError(
            f"Timestamp self-test failed: observed {observed}, expected {expected}"
        )
    return {
        "input_dtype": str(test.dtype),
        "observed_24h_ns": observed,
        "expected_24h_ns": expected,
        "passed": True,
    }


def add_session_features(fp: pd.DataFrame) -> pd.DataFrame:
    """
    Add exact session position, session length, session stage, session length bins,
    and 30-minute time buckets to front-page behaviors.
    """
    df = fp.sort_values(["session_id", "impression_time", "impression_id"]).copy()
    g = df.groupby("session_id", sort=False)
    df["_session_pos"] = g.cumcount() + 1
    df["_session_len"] = g["impression_id"].transform("size")
    
    # Session stage: early (0-1/3], middle (1/3-2/3], late (2/3-1], single (len==1)
    frac = df["_session_pos"] / df["_session_len"].clip(lower=1)
    df["_session_stage"] = pd.cut(
        frac, [0.0, 1/3, 2/3, 1.0000001],
        labels=["early", "middle", "late"], include_lowest=True
    ).astype(object)
    df.loc[df["_session_len"] == 1, "_session_stage"] = "single"

    # Session length bins: 1, 2, 3-5, 6+
    def lb(n: int) -> str:
        if n <= 1: return "1"
        if n == 2: return "2"
        if n <= 5: return "3-5"
        return "6+"
    df["_session_len_bin"] = df["_session_len"].map(lb)
    
    # Slate parsing and 30-minute bucket
    df["_slate"] = df["article_ids_inview"].map(norm_list)
    df["_slate_size"] = df["_slate"].map(len)
    df["_time_bucket"] = df["impression_time"].dt.floor(f"{MATCH_TIME_BUCKET_MINUTES}min")
    df["_time_ns"] = epoch_ns(df["impression_time"])
    
    return df


def load_articles(path: Path | str = None) -> pd.DataFrame:
    """Load and validate the articles table."""
    if path is None:
        path = ROOT_DIR / "ebnerd_small" / "articles.parquet"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Articles file not found: {path}")
    
    df = pd.read_parquet(path)
    required = {"article_id", "published_time", "premium", "category", "topics"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Articles table missing required columns: {missing}")
    
    df["published_time"] = pd.to_datetime(df["published_time"], utc=True, errors="coerce")
    df["_published_ns"] = epoch_ns(df["published_time"])
    df["premium"] = df["premium"].fillna(False).astype(bool)
    df["category"] = df["category"].fillna(-1).astype(int)
    
    return df


def load_behaviors(path: Path | str, frontpage_only: bool = True) -> pd.DataFrame:
    """Load and validate behaviors table, filtering to front-page if requested."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Behaviors file not found: {path}")
    
    df = pd.read_parquet(path)
    required = {
        "impression_id", "article_id", "impression_time", "device_type",
        "article_ids_inview", "article_ids_clicked", "user_id",
        "is_subscriber", "session_id"
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Behaviors table missing required columns: {missing}")
    
    df["impression_time"] = pd.to_datetime(df["impression_time"], utc=True, errors="coerce")
    df["is_subscriber"] = df["is_subscriber"].fillna(False).astype(bool)
    
    if frontpage_only:
        df = df[df["article_id"].isna()].copy()
        df = add_session_features(df)
    
    return df


def load_history(path: Path | str) -> pd.DataFrame:
    """Load and validate user history table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    
    df = pd.read_parquet(path)
    required = {"user_id", "article_id_fixed"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"History table missing required columns: {missing}")
    
    df["article_id_fixed"] = df["article_id_fixed"].map(norm_list)
    df["history_length"] = df["article_id_fixed"].map(len)
    
    return df


def js(v: Any) -> Any:
    """Recursively convert NumPy and Pandas types to JSON-serializable types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [js(x) for x in v]
    return v


def qsummary(x: Any) -> Dict[str, Any]:
    """Calculate standard quantile summary."""
    s = pd.Series(x, dtype=float).dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0])
    return {
        "n": int(len(s)),
        "min": float(q.loc[0]),
        "p01": float(q.loc[.01]),
        "p05": float(q.loc[.05]),
        "p10": float(q.loc[.10]),
        "p25": float(q.loc[.25]),
        "median": float(q.loc[.50]),
        "p75": float(q.loc[.75]),
        "p90": float(q.loc[.90]),
        "p95": float(q.loc[.95]),
        "p99": float(q.loc[.99]),
        "max": float(q.loc[1.0]),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }
