#!/usr/bin/env python3
"""
EB-NeRD Gate v3 — fixes timestamp-resolution handling in supply windows.

This version is intentionally limited to the pre-outcome gate. It does NOT
compute Phase 1 outcome metrics or Discovery Headroom.

Key fix
-------
Pandas may preserve EB-NeRD timestamps as datetime64[us, UTC]. `DatetimeIndex.asi8`
then returns MICROSECONDS, not nanoseconds. v2 incorrectly assumed `asi8` was always
nanoseconds, so the 12/24/48h windows were ~1000x too wide.

v3 explicitly converts timestamps to nanosecond resolution with `.as_unit("ns")`
before integer arithmetic and includes a runtime self-test so this cannot silently
regress.
"""

from __future__ import annotations
import argparse, ast, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

WINDOWS = (12, 24, 48)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviors", required=True)
    p.add_argument("--articles", required=True)
    p.add_argument("--outdir", default="real_gate_output_v3")
    return p.parse_args()

def read_table(path):
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file: {p}")

def norm_list(v):
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
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # CRITICAL: preserve semantic time while forcing integer unit to nanoseconds.
    return pd.DatetimeIndex(s).as_unit("ns").asi8

def timestamp_selftest():
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

def qsummary(x):
    s = pd.Series(x, dtype=float).dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0, .05, .25, .5, .75, .95, 1])
    return {
        "n": int(len(s)),
        "min": float(q.loc[0]),
        "p05": float(q.loc[.05]),
        "p25": float(q.loc[.25]),
        "median": float(q.loc[.5]),
        "p75": float(q.loc[.75]),
        "p95": float(q.loc[.95]),
        "max": float(q.loc[1]),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }

def to_jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    return v

def main():
    a = parse_args()
    t0 = time.perf_counter()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selftest = timestamp_selftest()

    beh = read_table(a.behaviors)
    art = read_table(a.articles)

    required_b = {"article_id", "impression_time", "article_ids_inview"}
    required_a = {"article_id", "published_time", "premium"}
    missing_b = required_b - set(beh.columns)
    missing_a = required_a - set(art.columns)
    if missing_b or missing_a:
        raise KeyError(f"Missing behavior columns={missing_b}; article columns={missing_a}")

    beh["impression_time"] = pd.to_datetime(beh["impression_time"], utc=True, errors="coerce")
    art["published_time"] = pd.to_datetime(art["published_time"], utc=True, errors="coerce")

    fp = beh.loc[beh["article_id"].isna()].copy()
    fp["_slate_size"] = fp["article_ids_inview"].map(lambda x: len(norm_list(x)))

    imp_ns = epoch_ns(fp["impression_time"])
    pub = art[["article_id", "published_time", "premium"]].dropna(subset=["published_time"]).sort_values("published_time").copy()
    pub_ns = epoch_ns(pub["published_time"])

    premium_arr = pub["premium"].fillna(False).astype(bool).to_numpy()
    premium_prefix = np.concatenate([[0], np.cumsum(premium_arr.astype(int))])

    windows = {}
    for hours in WINDOWS:
        delta_ns = int(pd.Timedelta(hours=hours).value)
        left = np.searchsorted(pub_ns, imp_ns - delta_ns, side="left")
        right = np.searchsorted(pub_ns, imp_ns, side="left")
        counts = right - left
        premium_counts = premium_prefix[right] - premium_prefix[left]
        nonpremium_counts = counts - premium_counts
        slate = fp["_slate_size"].to_numpy()
        ratio = counts / np.where(slate == 0, np.nan, slate)

        windows[f"{hours}h"] = {
            "supply_count": qsummary(counts),
            "supply_to_slate_ratio": qsummary(ratio),
            "share_impressions_supply_ge_3x_slate": float(np.mean(counts >= 3 * slate)),
            "share_impressions_nonempty_supply": float(np.mean(counts > 0)),
            "premium_supply_count": qsummary(premium_counts),
            "nonpremium_supply_count": qsummary(nonpremium_counts),
        }

    report = {
        "script": "gate_supply_v3.py",
        "purpose": "Corrected pre-outcome observable-supply feasibility check",
        "timestamp_selftest": selftest,
        "inputs": {
            "behaviors": a.behaviors,
            "articles": a.articles,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "frontpage_rows": int(len(fp)),
        "slate_size": qsummary(fp["_slate_size"]),
        "supply_windows": windows,
        "precommitment_boundary": {
            "computes_outcome_metrics": False,
            "computes_discovery_headroom": False,
            "computes_final_cohort_disparity": False,
        },
        "runtime_seconds": time.perf_counter() - t0,
        "status": "completed",
    }

    out = outdir / "gate_supply_v3.json"
    out.write_text(json.dumps(to_jsonable(report), indent=2))
    print(json.dumps({
        "timestamp_selftest": report["timestamp_selftest"],
        "frontpage_rows": report["frontpage_rows"],
        "12h_median": report["supply_windows"]["12h"]["supply_count"]["median"],
        "24h_median": report["supply_windows"]["24h"]["supply_count"]["median"],
        "48h_median": report["supply_windows"]["48h"]["supply_count"]["median"],
        "24h_supply_to_slate_median": report["supply_windows"]["24h"]["supply_to_slate_ratio"]["median"],
    }, indent=2))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
