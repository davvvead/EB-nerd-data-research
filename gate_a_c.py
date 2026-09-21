#!/usr/bin/env python3
"""
EB-NeRD feasibility gates A-C.

Purpose
-------
Run only pre-outcome feasibility diagnostics:
A. Load/schema/join and observable supply-window checks.
B. Matched-context front-page slate overlap diagnostics.
C. Embedding agreement against fixed-seed random baselines.

This script intentionally DOES NOT compute:
- stage effect outcomes
- cohort disparities
- Discovery Headroom
- final hypothesis tests

That separation helps preserve the precommitment boundary.

Example
-------
python gate_a_c.py \
  --behaviors data/behaviors.parquet \
  --articles data/articles.parquet \
  --embeddings data/contrastive_vector.parquet \
  --outdir gate_output
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

DEFAULT_SEED = 20260919
SUPPLY_WINDOWS_H = (12, 24, 48)

ALIASES = {
    "user_id": ["user_id"],
    "impression_id": ["impression_id"],
    "impression_time": ["impression_time", "timestamp"],
    "article_id_context": ["article_id"],
    "session_id": ["session_id"],
    "device_type": ["device_type", "device"],
    "scroll_percentage": ["scroll_percentage"],
    "article_ids_inview": ["article_ids_inview", "inview_articles"],
    "article_ids_clicked": ["article_ids_clicked", "clicked_articles"],
    "is_subscriber": ["is_subscriber", "subscriber"],
    "article_id": ["article_id"],
    "published_time": ["published_time", "publication_time"],
    "premium": ["premium", "is_premium"],
    "category": ["category", "category_str"],
    "topics": ["topics", "topic"],
}

REQUIRED_BEHAVIOR_KEYS = [
    "user_id",
    "impression_id",
    "impression_time",
    "article_id_context",
    "session_id",
    "device_type",
    "scroll_percentage",
    "article_ids_inview",
]

REQUIRED_ARTICLE_KEYS = [
    "article_id",
    "published_time",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--behaviors", required=True, help="Behaviors parquet/csv file")
    p.add_argument("--articles", required=True, help="Articles parquet/csv file")
    p.add_argument("--embeddings", default=None, help="Optional article embedding parquet/csv")
    p.add_argument("--outdir", default="gate_output")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--time-bucket-min", type=int, default=30)
    p.add_argument("--scroll-bin-width", type=int, default=20)
    p.add_argument("--max-overlap-pairs", type=int, default=50000)
    p.add_argument("--embedding-sample", type=int, default=2000)
    p.add_argument("--embedding-bootstrap", type=int, default=500)
    p.add_argument("--frontpage-mode", choices=["article_id_null"], default="article_id_null")
    return p.parse_args()


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def resolve_col(df: pd.DataFrame, key: str, required: bool = True) -> Optional[str]:
    for c in ALIASES.get(key, [key]):
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required semantic column '{key}'. Tried {ALIASES.get(key, [key])}")
    return None


def normalize_list(v: Any) -> List[Any]:
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
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, np.ndarray, set)):
                return list(parsed)
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return [v]


def jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def qsummary(s: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    qs = s.quantile([0, .05, .25, .5, .75, .95, 1.0])
    return {
        "n": int(len(s)),
        "min": float(qs.loc[0]),
        "p05": float(qs.loc[0.05]),
        "p25": float(qs.loc[0.25]),
        "median": float(qs.loc[0.5]),
        "p75": float(qs.loc[0.75]),
        "p95": float(qs.loc[0.95]),
        "max": float(qs.loc[1.0]),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }


def ci_mean(x: np.ndarray, rng: np.random.Generator, n_boot: int = 500) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = rng.choice(x, size=len(x), replace=True).mean()
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(vals, .025)),
        "ci_high": float(np.quantile(vals, .975)),
    }


def schema_and_join_audit(beh: pd.DataFrame, art: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    bcols = {k: resolve_col(beh, k, required=(k in REQUIRED_BEHAVIOR_KEYS)) for k in ALIASES}
    acols = {k: resolve_col(art, k, required=(k in REQUIRED_ARTICLE_KEYS)) for k in ALIASES}

    imp_time = bcols["impression_time"]
    pub_time = acols["published_time"]
    beh[imp_time] = pd.to_datetime(beh[imp_time], errors="coerce", utc=True)
    art[pub_time] = pd.to_datetime(art[pub_time], errors="coerce", utc=True)

    aid = acols["article_id"]
    known_articles = set(art[aid].dropna().tolist())
    inview_col = bcols["article_ids_inview"]
    all_inview = []
    for x in beh[inview_col]:
        all_inview.extend(normalize_list(x))
    all_inview_nonnull = [x for x in all_inview if pd.notna(x)]
    missing = [x for x in all_inview_nonnull if x not in known_articles]

    audit = {
        "behavior_rows": len(beh),
        "article_rows": len(art),
        "behavior_columns": list(beh.columns),
        "article_columns": list(art.columns),
        "behavior_null_rate": {c: float(beh[c].isna().mean()) for c in beh.columns},
        "article_null_rate": {c: float(art[c].isna().mean()) for c in art.columns},
        "impression_date_min": beh[imp_time].min(),
        "impression_date_max": beh[imp_time].max(),
        "article_publish_min": art[pub_time].min(),
        "article_publish_max": art[pub_time].max(),
        "inview_article_references": len(all_inview_nonnull),
        "inview_missing_from_articles": len(missing),
        "inview_join_coverage": (
            1.0 - len(missing) / len(all_inview_nonnull) if all_inview_nonnull else None
        ),
    }
    return bcols, acols, audit


def frontpage_filter(beh: pd.DataFrame, bcols: Dict[str, str]) -> pd.DataFrame:
    ctx = bcols["article_id_context"]
    out = beh.loc[beh[ctx].isna()].copy()
    out["_slate"] = out[bcols["article_ids_inview"]].map(normalize_list)
    out["_slate_size"] = out["_slate"].map(len)
    return out


def add_session_features(fp: pd.DataFrame, bcols: Dict[str, str]) -> pd.DataFrame:
    fp = fp.sort_values([bcols["session_id"], bcols["impression_time"], bcols["impression_id"]]).copy()
    g = fp.groupby(bcols["session_id"], sort=False)
    fp["_session_pos"] = g.cumcount() + 1
    fp["_session_len"] = g[bcols["impression_id"]].transform("size")
    frac = fp["_session_pos"] / fp["_session_len"].clip(lower=1)
    fp["_session_stage"] = pd.cut(
        frac,
        bins=[0, 1/3, 2/3, 1.0000001],
        labels=["early", "middle", "late"],
        include_lowest=True,
    ).astype("object")
    fp.loc[fp["_session_len"] == 1, "_session_stage"] = "single"

    def len_bin(x: int) -> str:
        if x <= 1:
            return "1"
        if x == 2:
            return "2"
        if x <= 5:
            return "3-5"
        return "6+"
    fp["_session_len_bin"] = fp["_session_len"].map(len_bin)
    return fp


def supply_window_diagnostics(
    fp: pd.DataFrame,
    art: pd.DataFrame,
    bcols: Dict[str, str],
    acols: Dict[str, str],
) -> Dict[str, Any]:
    imp_times = fp[bcols["impression_time"]].astype("int64").to_numpy()
    pub = art[[acols["published_time"], acols["article_id"]]].dropna(subset=[acols["published_time"]]).copy()
    pub = pub.sort_values(acols["published_time"])
    pub_ns = pub[acols["published_time"]].astype("int64").to_numpy()

    premium_col = acols.get("premium")
    if premium_col:
        premium_series = art.set_index(acols["article_id"])[premium_col]
        pub["_premium"] = pub[acols["article_id"]].map(premium_series).fillna(False).astype(bool)
        prem_arr = pub["_premium"].to_numpy()
        prem_prefix = np.concatenate([[0], np.cumsum(prem_arr.astype(int))])
    else:
        prem_prefix = None

    out = {}
    for h in SUPPLY_WINDOWS_H:
        delta_ns = int(pd.Timedelta(hours=h).value)
        left = np.searchsorted(pub_ns, imp_times - delta_ns, side="left")
        right = np.searchsorted(pub_ns, imp_times, side="left")
        counts = right - left
        ratio = counts / fp["_slate_size"].replace(0, np.nan).to_numpy()
        record = {
            "supply_count": qsummary(pd.Series(counts)),
            "supply_to_slate_ratio": qsummary(pd.Series(ratio)),
            "share_impressions_supply_ge_3x_slate": float(
                np.nanmean(counts >= 3 * fp["_slate_size"].to_numpy())
            ),
            "share_impressions_nonempty_supply": float(np.mean(counts > 0)),
        }
        if prem_prefix is not None:
            prem_counts = prem_prefix[right] - prem_prefix[left]
            record["premium_supply_count"] = qsummary(pd.Series(prem_counts))
            record["nonpremium_supply_count"] = qsummary(pd.Series(counts - prem_counts))
        out[f"{h}h"] = record
    return out


def make_context_strata(
    fp: pd.DataFrame,
    bcols: Dict[str, str],
    time_bucket_min: int,
    scroll_bin_width: int,
) -> pd.DataFrame:
    out = fp.copy()
    out["_time_bucket"] = out[bcols["impression_time"]].dt.floor(f"{time_bucket_min}min")
    scroll = pd.to_numeric(out[bcols["scroll_percentage"]], errors="coerce")
    out["_scroll_bin"] = (np.floor(scroll / scroll_bin_width) * scroll_bin_width).astype("Int64")
    return out


def jaccard(a: Sequence[Any], b: Sequence[Any]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def matched_overlap_diagnostics(
    fp: pd.DataFrame,
    bcols: Dict[str, str],
    seed: int,
    time_bucket_min: int,
    scroll_bin_width: int,
    max_pairs: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = make_context_strata(fp, bcols, time_bucket_min, scroll_bin_width)
    device = bcols["device_type"]

    complete = x.dropna(subset=["_scroll_bin", device, "_session_stage", "_session_len_bin"]).copy()
    keys = ["_time_bucket", device, "_scroll_bin", "_session_stage", "_session_len_bin"]

    eligible_groups = []
    for key, g in complete.groupby(keys, dropna=False, observed=True):
        if len(g) >= 2:
            eligible_groups.append((key, g.index.to_numpy()))

    pairs = []
    if eligible_groups:
        order = rng.permutation(len(eligible_groups))
        for gi in order:
            _, idx = eligible_groups[gi]
            # Randomly pair records inside a stratum without exhaustive O(n^2) expansion.
            perm = rng.permutation(idx)
            for a, b in zip(perm[::2], perm[1::2]):
                pairs.append((a, b))
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

    scores = [jaccard(x.at[a, "_slate"], x.at[b, "_slate"]) for a, b in pairs]
    return {
        "match_definition": {
            "same_time_bucket_minutes": time_bucket_min,
            "same_device_type": True,
            "scroll_bin_width_percentage_points": scroll_bin_width,
            "same_session_stage": True,
            "same_session_length_bin": True,
            "session_length_bins": ["1", "2", "3-5", "6+"],
        },
        "frontpage_rows": len(x),
        "rows_with_complete_match_context": len(complete),
        "complete_context_share": float(len(complete) / len(x)) if len(x) else None,
        "eligible_strata": len(eligible_groups),
        "sampled_pairs": len(scores),
        "jaccard": qsummary(pd.Series(scores, dtype=float)),
    }


def extract_embedding_matrix(emb: pd.DataFrame) -> Tuple[str, np.ndarray, np.ndarray]:
    aid_col = resolve_col(emb, "article_id", required=True)
    candidate_cols = [c for c in emb.columns if c != aid_col]

    # Prefer an object/list vector column. CSV smoke tests may store lists as strings.
    for c in candidate_cols:
        sample = emb[c].dropna().head(5)
        looks_like_vector = False
        if len(sample):
            for v in sample:
                if isinstance(v, (list, tuple, np.ndarray)):
                    looks_like_vector = True
                    break
                if isinstance(v, str):
                    try:
                        parsed = ast.literal_eval(v)
                        if isinstance(parsed, (list, tuple, np.ndarray)):
                            looks_like_vector = True
                            break
                    except Exception:
                        pass
        if looks_like_vector:
            rows = []
            for v in emb[c]:
                if isinstance(v, str):
                    v = ast.literal_eval(v)
                rows.append(np.asarray(v, dtype=float))
            vecs = np.vstack(rows)
            return aid_col, emb[aid_col].to_numpy(), vecs

    # Otherwise treat numeric columns as vector dimensions.
    num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(emb[c])]
    if not num_cols:
        raise ValueError("Could not detect embedding vector column(s).")
    return aid_col, emb[aid_col].to_numpy(), emb[num_cols].to_numpy(dtype=float)


def topic_set(v: Any) -> set:
    return set(str(x) for x in normalize_list(v))


def embedding_agreement_test(
    emb: pd.DataFrame,
    art: pd.DataFrame,
    acols: Dict[str, str],
    seed: int,
    sample_n: int,
    n_boot: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    aid_col, ids, X = extract_embedding_matrix(emb)

    finite = np.isfinite(X).all(axis=1)
    ids, X = ids[finite], X[finite]

    art_meta_cols = [acols["article_id"]]
    category_col = acols.get("category")
    topics_col = acols.get("topics")
    if category_col:
        art_meta_cols.append(category_col)
    if topics_col:
        art_meta_cols.append(topics_col)

    meta = art[art_meta_cols].drop_duplicates(acols["article_id"]).set_index(acols["article_id"])
    keep = np.array([i in meta.index for i in ids])
    ids, X = ids[keep], X[keep]

    coverage = len(ids) / max(1, art[acols["article_id"]].nunique())
    if len(ids) < 3:
        return {
            "status": "insufficient_embedding_coverage",
            "embedding_article_count": len(ids),
            "article_id_coverage": coverage,
        }

    if len(ids) > sample_n:
        take = rng.choice(len(ids), size=sample_n, replace=False)
        ids_s, X_s = ids[take], X[take]
    else:
        ids_s, X_s = ids, X

    # Normalize and find nearest non-self neighbor.
    norms = np.linalg.norm(X_s, axis=1, keepdims=True)
    Xn = X_s / np.clip(norms, 1e-12, None)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xn)
    _, neigh = nn.kneighbors(Xn)
    nn_idx = neigh[:, 1]

    rand_idx = rng.integers(0, len(ids_s), size=len(ids_s))
    self_mask = rand_idx == np.arange(len(ids_s))
    rand_idx[self_mask] = (rand_idx[self_mask] + 1) % len(ids_s)

    result = {
        "embedding_article_count": len(ids),
        "article_id_coverage": float(coverage),
        "tested_articles": len(ids_s),
    }

    if category_col:
        cats = np.array([meta.at[i, category_col] if i in meta.index else None for i in ids_s], dtype=object)
        valid = pd.notna(cats) & pd.notna(cats[nn_idx]) & pd.notna(cats[rand_idx])
        if valid.sum():
            nn_same = (cats == cats[nn_idx]).astype(float)[valid]
            rand_same = (cats == cats[rand_idx]).astype(float)[valid]
            result["category_same_rate_nearest"] = ci_mean(nn_same, rng, n_boot)
            result["category_same_rate_random"] = ci_mean(rand_same, rng, n_boot)
            result["category_agreement_lift"] = float(nn_same.mean() - rand_same.mean())

    if topics_col:
        tsets = [topic_set(meta.at[i, topics_col]) if i in meta.index else set() for i in ids_s]
        nn_j = []
        rd_j = []
        for idx in range(len(ids_s)):
            a = tsets[idx]
            b = tsets[nn_idx[idx]]
            c = tsets[rand_idx[idx]]
            if a:
                nn_j.append(len(a & b) / len(a | b) if (a | b) else 0.0)
                rd_j.append(len(a & c) / len(a | c) if (a | c) else 0.0)
        if nn_j:
            nn_j = np.asarray(nn_j)
            rd_j = np.asarray(rd_j)
            result["topic_jaccard_nearest"] = ci_mean(nn_j, rng, n_boot)
            result["topic_jaccard_random"] = ci_mean(rd_j, rng, n_boot)
            result["topic_agreement_lift"] = float(nn_j.mean() - rd_j.mean())

    return result


def artifact_handoff(
    beh: pd.DataFrame,
    art: pd.DataFrame,
    emb: Optional[pd.DataFrame],
    bcols: Dict[str, str],
    acols: Dict[str, str],
) -> Dict[str, Any]:
    imp_time = bcols["impression_time"]
    pub_time = acols["published_time"]
    result = {
        "behavior_impression_range": [beh[imp_time].min(), beh[imp_time].max()],
        "article_publish_range": [art[pub_time].min(), art[pub_time].max()],
        "embedding_file_loaded": emb is not None,
    }
    if emb is not None:
        try:
            e_id = resolve_col(emb, "article_id", required=True)
            emb_ids = set(emb[e_id].dropna())
            analyzed_ids = set()
            for v in beh[bcols["article_ids_inview"]]:
                analyzed_ids.update(normalize_list(v))
            result["embedding_id_coverage_of_inview_articles"] = (
                len(analyzed_ids & emb_ids) / len(analyzed_ids) if analyzed_ids else None
            )
        except Exception as e:
            result["embedding_coverage_error"] = repr(e)
    return result


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "script": "gate_a_c.py",
        "purpose": "Pre-outcome EB-NeRD feasibility gates A-C",
        "seed": args.seed,
        "inputs": {
            "behaviors": args.behaviors,
            "articles": args.articles,
            "embeddings": args.embeddings,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "precommitment_boundary": {
            "computes_outcome_metrics": False,
            "computes_discovery_headroom": False,
            "computes_final_cohort_disparity": False,
        },
    }

    try:
        t0 = time.perf_counter()
        beh = read_table(args.behaviors)
        art = read_table(args.articles)
        emb = read_table(args.embeddings) if args.embeddings else None
        report["runtime_load_seconds"] = time.perf_counter() - t0

        bcols, acols, schema = schema_and_join_audit(beh, art)
        report["A_schema_join"] = schema

        fp = frontpage_filter(beh, bcols)
        fp = add_session_features(fp, bcols)
        report["A_frontpage"] = {
            "rows": len(fp),
            "share_of_behavior_rows": float(len(fp) / len(beh)) if len(beh) else None,
            "slate_size": qsummary(fp["_slate_size"]),
            "device_distribution": fp[bcols["device_type"]].astype(str).value_counts(dropna=False).to_dict(),
            "scroll_percentage": qsummary(pd.to_numeric(fp[bcols["scroll_percentage"]], errors="coerce")),
            "session_length": qsummary(fp["_session_len"]),
        }

        report["A_supply_windows"] = supply_window_diagnostics(fp, art, bcols, acols)

        report["B_matched_context_overlap"] = matched_overlap_diagnostics(
            fp=fp,
            bcols=bcols,
            seed=args.seed,
            time_bucket_min=args.time_bucket_min,
            scroll_bin_width=args.scroll_bin_width,
            max_pairs=args.max_overlap_pairs,
        )

        if emb is not None:
            report["C_embedding_agreement"] = embedding_agreement_test(
                emb=emb,
                art=art,
                acols=acols,
                seed=args.seed,
                sample_n=args.embedding_sample,
                n_boot=args.embedding_bootstrap,
            )
        else:
            report["C_embedding_agreement"] = {"status": "not_run_no_embedding_file"}

        report["D_handoff_manifest"] = artifact_handoff(beh, art, emb, bcols, acols)
        report["runtime_total_seconds"] = time.perf_counter() - start
        report["status"] = "completed"

    except Exception as e:
        report["status"] = "failed"
        report["error"] = repr(e)
        report["runtime_total_seconds"] = time.perf_counter() - start
        with open(outdir / "gate_report.json", "w") as f:
            json.dump(jsonable(report), f, indent=2)
        raise

    with open(outdir / "gate_report.json", "w") as f:
        json.dump(jsonable(report), f, indent=2)

    # Small human-readable summary. It intentionally reports diagnostics,
    # not "pass/fail" decisions that depend on the parameter-freeze commit.
    summary = {
        "frontpage_rows": report["A_frontpage"]["rows"],
        "median_slate_size": report["A_frontpage"]["slate_size"].get("median"),
        "24h_supply_median": report["A_supply_windows"]["24h"]["supply_count"].get("median"),
        "24h_supply_to_slate_median": report["A_supply_windows"]["24h"]["supply_to_slate_ratio"].get("median"),
        "matched_pair_count": report["B_matched_context_overlap"]["sampled_pairs"],
        "matched_jaccard_median": report["B_matched_context_overlap"]["jaccard"].get("median"),
        "embedding_status": report["C_embedding_agreement"].get("status", "ran"),
        "runtime_total_seconds": report["runtime_total_seconds"],
    }
    with open(outdir / "gate_summary.json", "w") as f:
        json.dump(jsonable(summary), f, indent=2)

    print(json.dumps(jsonable(summary), indent=2))
    print(f"\nWrote {outdir / 'gate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
