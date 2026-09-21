#!/usr/bin/env python3
"""
Final EB-NeRD pre-outcome gate.

This script does NOT compute Phase 1 findings.
It only:
1. compares two semantic embedding artifacts;
2. audits user-history distributions;
3. audits front-page slate/matching support;
4. reports common-support diagnostics;
5. produces the information needed for the parameter-freeze commit.

No validation outcome metrics or Discovery Headroom are computed.
"""

from __future__ import annotations
import argparse, ast, json, math, platform, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SEED = 20260919

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--articles", required=True)
    p.add_argument("--behaviors", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--bert", required=True)
    p.add_argument("--contrastive", required=True)
    p.add_argument("--outdir", default="final_preoutcome_gate")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--embedding-sample", type=int, default=5000)
    p.add_argument("--time-bucket-min", type=int, default=30)
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
            x = ast.literal_eval(s)
            if isinstance(x, (list, tuple, np.ndarray, set)):
                return list(x)
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return [v]

def js(v):
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

def qsummary(x):
    s = pd.Series(x, dtype=float).dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1])
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
        "max": float(q.loc[1]),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }

def detect_article_id(df):
    for c in ["article_id", "article_ids", "id"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not identify article ID column. Columns: {list(df.columns)}")

def detect_vectors(df, id_col):
    other = [c for c in df.columns if c != id_col]

    # Common case: one list/array vector column.
    for c in other:
        vals = df[c].dropna().head(5).tolist()
        if vals and any(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
            try:
                X = np.vstack([np.asarray(v, dtype=np.float32) for v in df[c]])
                return X, {"mode": "single_vector_column", "column": c, "dimension": int(X.shape[1])}
            except Exception:
                pass

    # Array-like values can sometimes come back as Python objects.
    for c in other:
        if df[c].dtype == object:
            sample = df[c].dropna().head(3)
            try:
                Xs = [np.asarray(v, dtype=np.float32) for v in sample]
                if Xs and all(x.ndim == 1 and len(x) > 1 for x in Xs):
                    X = np.vstack([np.asarray(v, dtype=np.float32) for v in df[c]])
                    return X, {"mode": "object_vector_column", "column": c, "dimension": int(X.shape[1])}
            except Exception:
                pass

    # Otherwise assume numeric dimensions.
    num = [c for c in other if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) >= 2:
        X = df[num].to_numpy(dtype=np.float32)
        return X, {"mode": "numeric_dimension_columns", "columns": num[:10], "n_columns": len(num), "dimension": len(num)}

    raise ValueError(f"Could not detect embedding vector in columns {list(df.columns)}")

def topic_set(v):
    return set(str(x) for x in norm_list(v))

def embedding_audit(name, emb, articles, rng, sample_n):
    aid = detect_article_id(emb)
    X, schema = detect_vectors(emb, aid)
    ids = emb[aid].to_numpy()

    finite = np.isfinite(X).all(axis=1)
    ids, X = ids[finite], X[finite]

    art = articles.drop_duplicates("article_id").set_index("article_id")
    covered = np.array([i in art.index for i in ids])
    ids, X = ids[covered], X[covered]

    article_coverage = len(set(ids)) / articles["article_id"].nunique()

    if len(ids) > sample_n:
        take = rng.choice(len(ids), size=sample_n, replace=False)
        ids_s, X_s = ids[take], X[take]
    else:
        ids_s, X_s = ids, X

    norms = np.linalg.norm(X_s, axis=1)
    valid_norm = norms > 1e-12
    ids_s, X_s, norms = ids_s[valid_norm], X_s[valid_norm], norms[valid_norm]
    Xn = X_s / norms[:, None]

    nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1).fit(Xn)
    dist, idx = nn.kneighbors(Xn)
    near = idx[:, 1]

    rand = rng.integers(0, len(ids_s), len(ids_s))
    same = rand == np.arange(len(ids_s))
    rand[same] = (rand[same] + 1) % len(ids_s)

    cats = np.array([art.at[i, "category"] for i in ids_s], dtype=object)
    cat_nn = (cats == cats[near]).astype(float)
    cat_rand = (cats == cats[rand]).astype(float)

    topics = [topic_set(art.at[i, "topics"]) for i in ids_s]
    topic_nn, topic_rand = [], []
    for j, a in enumerate(topics):
        if not a:
            continue
        b = topics[near[j]]
        c = topics[rand[j]]
        topic_nn.append(len(a & b) / len(a | b) if (a | b) else 0.0)
        topic_rand.append(len(a & c) / len(a | c) if (a | c) else 0.0)

    topic_nn = np.asarray(topic_nn, float)
    topic_rand = np.asarray(topic_rand, float)

    return {
        "name": name,
        "rows": len(emb),
        "usable_vectors": len(ids),
        "article_id_coverage": float(article_coverage),
        "vector_schema": schema,
        "vector_norm": qsummary(np.linalg.norm(X, axis=1)),
        "nearest_neighbor_cosine_similarity": qsummary(1.0 - dist[:, 1]),
        "category_same_rate_nearest": float(cat_nn.mean()),
        "category_same_rate_random": float(cat_rand.mean()),
        "category_agreement_lift": float(cat_nn.mean() - cat_rand.mean()),
        "topic_jaccard_nearest": float(topic_nn.mean()) if len(topic_nn) else None,
        "topic_jaccard_random": float(topic_rand.mean()) if len(topic_rand) else None,
        "topic_agreement_lift": float(topic_nn.mean() - topic_rand.mean()) if len(topic_nn) else None,
        "tested_articles": int(len(ids_s)),
    }

def entropy_from_counts(values):
    counts = pd.Series(values).value_counts().to_numpy(dtype=float)
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())

def history_audit(history, articles):
    art_cat = articles.set_index("article_id")["category"].to_dict()
    rows = []

    for _, r in history.iterrows():
        ids = norm_list(r["article_id_fixed"])
        cats = [art_cat.get(i) for i in ids if i in art_cat]
        cats = [c for c in cats if pd.notna(c)]
        rows.append({
            "user_id": r["user_id"],
            "history_length": len(ids),
            "unique_articles": len(set(ids)),
            "unique_categories": len(set(cats)),
            "category_entropy": entropy_from_counts(cats),
        })

    h = pd.DataFrame(rows)
    return {
        "users": len(h),
        "history_length": qsummary(h["history_length"]),
        "unique_articles": qsummary(h["unique_articles"]),
        "unique_categories": qsummary(h["unique_categories"]),
        "category_entropy": qsummary(h["category_entropy"]),
        "shares_at_history_thresholds": {
            str(n): float((h["history_length"] >= n).mean())
            for n in [3, 5, 10, 15, 20, 30, 50, 100]
        },
        "raw_user_diagnostics": h,
    }

def add_session_features(fp):
    x = fp.sort_values(["session_id", "impression_time", "impression_id"]).copy()
    g = x.groupby("session_id", sort=False)
    x["_session_pos"] = g.cumcount() + 1
    x["_session_len"] = g["impression_id"].transform("size")
    frac = x["_session_pos"] / x["_session_len"].clip(lower=1)
    x["_session_stage"] = pd.cut(
        frac, [0, 1/3, 2/3, 1.0000001],
        labels=["early", "middle", "late"], include_lowest=True
    ).astype(object)
    x.loc[x["_session_len"] == 1, "_session_stage"] = "single"

    def lb(n):
        if n <= 1: return "1"
        if n == 2: return "2"
        if n <= 5: return "3-5"
        return "6+"
    x["_session_len_bin"] = x["_session_len"].map(lb)
    x["_slate_size"] = x["article_ids_inview"].map(lambda z: len(norm_list(z)))
    x["_time_bucket"] = x["impression_time"].dt.floor("30min")
    return x

def exposure_support_audit(beh):
    fp = beh.loc[beh["article_id"].isna()].copy()
    fp["impression_time"] = pd.to_datetime(fp["impression_time"], utc=True)
    fp = add_session_features(fp)

    exploded = fp[
        ["user_id", "_time_bucket", "device_type", "_session_stage",
         "_session_len_bin", "_slate_size", "article_ids_inview"]
    ].explode("article_ids_inview")
    exploded = exploded.rename(columns={"article_ids_inview": "candidate_article_id"})

    # Support within broad temporal bucket first; exact full-context support is also reported.
    temporal_support = (
        exploded.groupby(["_time_bucket", "candidate_article_id"])["user_id"]
        .nunique()
        .rename("support")
        .reset_index()
    )

    contextual_support = (
        exploded.groupby([
            "_time_bucket", "device_type", "_session_stage",
            "_session_len_bin", "_slate_size", "candidate_article_id"
        ])["user_id"]
        .nunique()
        .rename("support")
        .reset_index()
    )

    def threshold_shares(s):
        return {str(n): float((s >= n).mean()) for n in [1, 2, 3, 5, 10, 20, 50]}

    # Matching stratum sizes, before article identity.
    strata = (
        fp.groupby(["_time_bucket", "device_type", "_session_stage",
                    "_session_len_bin", "_slate_size"])
        .size()
        .rename("n")
    )

    return {
        "frontpage_rows": len(fp),
        "slate_size": qsummary(fp["_slate_size"]),
        "matching_stratum_size": qsummary(strata),
        "share_impressions_in_strata_with_at_least": {
            str(n): float(
                (
                    fp.set_index(["_time_bucket", "device_type", "_session_stage",
                                  "_session_len_bin", "_slate_size"])
                      .index.map(strata).to_numpy() >= n
                ).mean()
            )
            for n in [2, 3, 5, 10, 20]
        },
        "temporal_article_support_30m": {
            "distribution": qsummary(temporal_support["support"]),
            "threshold_shares": threshold_shares(temporal_support["support"]),
        },
        "full_context_article_support_30m": {
            "distribution": qsummary(contextual_support["support"]),
            "threshold_shares": threshold_shares(contextual_support["support"]),
        },
    }

def main():
    a = parse_args()
    start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    articles = read_table(a.articles)
    behaviors = read_table(a.behaviors)
    history = read_table(a.history)
    bert = read_table(a.bert)
    contrastive = read_table(a.contrastive)

    # Required structural checks
    for col in ["article_id", "category", "topics", "published_time"]:
        if col not in articles.columns:
            raise KeyError(f"articles missing {col}")
    for col in ["user_id", "impression_id", "article_id", "impression_time",
                "device_type", "session_id", "article_ids_inview"]:
        if col not in behaviors.columns:
            raise KeyError(f"behaviors missing {col}")
    for col in ["user_id", "article_id_fixed"]:
        if col not in history.columns:
            raise KeyError(f"history missing {col}")

    behaviors["impression_time"] = pd.to_datetime(behaviors["impression_time"], utc=True)

    emb_bert = embedding_audit(
        "bert_base_multilingual_cased", bert, articles, rng, a.embedding_sample
    )
    emb_con = embedding_audit(
        "contrastive_vector", contrastive, articles, rng, a.embedding_sample
    )

    hist = history_audit(history, articles)
    user_diag = hist.pop("raw_user_diagnostics")
    user_diag.to_csv(outdir / "history_user_diagnostics.csv", index=False)

    support = exposure_support_audit(behaviors)

    report = {
        "script": "final_preoutcome_gate.py",
        "purpose": "Embedding comparison + parameter-freeze diagnostics only",
        "seed": a.seed,
        "precommitment_boundary": {
            "computes_validation_outcomes": False,
            "computes_stage_effects": False,
            "computes_discovery_headroom": False,
            "selects_winning_hypothesis": False,
        },
        "inputs": {
            "articles": a.articles,
            "behaviors": a.behaviors,
            "history": a.history,
            "bert": a.bert,
            "contrastive": a.contrastive,
        },
        "embedding_audits": {
            "bert": emb_bert,
            "contrastive": emb_con,
        },
        "history_audit": hist,
        "support_and_matching_audit": support,
        "already_locked": {
            "dataset": "EB-NeRD Small",
            "surface": "front page only",
            "primary_supply_window_hours": 24,
            "supply_sensitivity_hours": [12, 48],
            "counterfactual_supply_requirement": "observable supply >= 3x actual slate size",
            "primary_null_family": "pre-impression click popularity + recency",
            "reference_null": "same-size uniform random draw",
            "outcome_period": "validation",
            "training_period": "train",
            "uncertainty": "user-clustered bootstrap",
            "headline_exposure_wording": "publisher-side exposure",
            "max_headroom_swaps": 2,
            "primary_headroom_relevance_tolerance": 0.03,
        },
        "parameters_still_to_freeze_after_this_report": [
            "primary embedding representation",
            "minimum history length",
            "history-neighbor k",
            "eligible slate-size range",
            "common-support threshold n",
            "exact matching/caliper rules",
            "relevance-floor tau rule",
            "popularity+recency weighting/sampling rule",
            "minimum relevance-model quality threshold",
            "effect-size standardization population",
            "bootstrap replicate count",
        ],
        "runtime_seconds": time.perf_counter() - start,
        "status": "completed",
    }

    (outdir / "final_preoutcome_report.json").write_text(
        json.dumps(js(report), indent=2)
    )

    print(json.dumps({
        "bert": {
            "coverage": emb_bert["article_id_coverage"],
            "category_lift": emb_bert["category_agreement_lift"],
            "topic_lift": emb_bert["topic_agreement_lift"],
        },
        "contrastive": {
            "coverage": emb_con["article_id_coverage"],
            "category_lift": emb_con["category_agreement_lift"],
            "topic_lift": emb_con["topic_agreement_lift"],
        },
        "history_length": hist["history_length"],
        "matching_stratum_size": support["matching_stratum_size"],
        "temporal_support": support["temporal_article_support_30m"],
    }, indent=2))
    print(f"\nWrote {outdir / 'final_preoutcome_report.json'}")
    print(f"Wrote {outdir / 'history_user_diagnostics.csv'}")

if __name__ == "__main__":
    main()
