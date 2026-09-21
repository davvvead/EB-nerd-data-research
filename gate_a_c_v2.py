#!/usr/bin/env python3
"""
EB-NeRD feasibility gates A-C, v2.

Changes from v1
---------------
1) Fixes observable-supply timestamp arithmetic by forcing all epoch values
   to nanoseconds before subtracting Timedelta nanoseconds.
2) Removes front-page scroll_percentage from matched-context overlap because
   the real Small bundle contains scroll values for only a tiny fraction of
   front-page rows. Matching now uses:
      - same 30-minute time bucket
      - same device type
      - same session stage
      - same session-length bin
      - exact same exposed slate size
3) Keeps scroll completeness as a reported diagnostic.
4) Still computes no outcome metrics, cohort disparities, or headroom.

Example
-------
python gate_a_c_v2.py \
  --behaviors ebnerd_small/train/behaviors.parquet \
  --articles ebnerd_small/articles.parquet \
  --outdir real_gate_output_v2
"""

from __future__ import annotations
import argparse, ast, json, math, platform, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

REQ_B = ["user_id","impression_id","impression_time","article_id_context",
         "session_id","device_type","article_ids_inview"]
REQ_A = ["article_id","published_time"]

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--behaviors", required=True)
    p.add_argument("--articles", required=True)
    p.add_argument("--embeddings", default=None)
    p.add_argument("--outdir", default="gate_output_v2")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--time-bucket-min", type=int, default=30)
    p.add_argument("--max-overlap-pairs", type=int, default=50000)
    p.add_argument("--embedding-sample", type=int, default=2000)
    p.add_argument("--embedding-bootstrap", type=int, default=500)
    return p.parse_args()

def read_table(path):
    p=Path(path)
    if p.suffix.lower() in {".parquet",".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv",".gz"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file: {p}")

def resolve(df,key,required=True):
    for c in ALIASES.get(key,[key]):
        if c in df.columns: return c
    if required: raise KeyError(f"Missing {key}; tried {ALIASES.get(key,[key])}")
    return None

def norm_list(v):
    if v is None: return []
    if isinstance(v,float) and np.isnan(v): return []
    if isinstance(v,(list,tuple,np.ndarray,pd.Series,set)): return list(v)
    if isinstance(v,str):
        s=v.strip()
        if not s: return []
        try:
            q=ast.literal_eval(s)
            if isinstance(q,(list,tuple,np.ndarray,set)): return list(q)
        except Exception: pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return [v]

def js(v):
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return None if np.isnan(v) else float(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    if isinstance(v,pd.Timestamp): return v.isoformat()
    if isinstance(v,Path): return str(v)
    if isinstance(v,dict): return {str(k):js(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [js(x) for x in v]
    return v

def qsum(s):
    s=pd.to_numeric(s,errors="coerce").dropna()
    if s.empty: return {"n":0}
    q=s.quantile([0,.05,.25,.5,.75,.95,1])
    return {"n":int(len(s)),"min":float(q.loc[0]),"p05":float(q.loc[.05]),
            "p25":float(q.loc[.25]),"median":float(q.loc[.5]),
            "p75":float(q.loc[.75]),"p95":float(q.loc[.95]),"max":float(q.loc[1]),
            "mean":float(s.mean()),"sd":float(s.std(ddof=1)) if len(s)>1 else 0.0}

def epoch_ns(series: pd.Series) -> np.ndarray:
    """Return UTC epoch nanoseconds robustly even if source dtype is timestamp[us]."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # DatetimeIndex.asi8 is nanoseconds from Unix epoch.
    return pd.DatetimeIndex(s).asi8

def ci_mean(x,rng,n_boot=500):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return {"mean":None,"ci_low":None,"ci_high":None}
    vals=np.empty(n_boot)
    for b in range(n_boot):
        vals[b]=rng.choice(x,size=len(x),replace=True).mean()
    return {"mean":float(x.mean()),"ci_low":float(np.quantile(vals,.025)),
            "ci_high":float(np.quantile(vals,.975))}

def schema_audit(beh,art):
    bc={k:resolve(beh,k,k in REQ_B) for k in ALIASES}
    ac={k:resolve(art,k,k in REQ_A) for k in ALIASES}
    beh[bc["impression_time"]]=pd.to_datetime(beh[bc["impression_time"]],errors="coerce",utc=True)
    art[ac["published_time"]]=pd.to_datetime(art[ac["published_time"]],errors="coerce",utc=True)

    known=set(art[ac["article_id"]].dropna())
    refs=[]
    for x in beh[bc["article_ids_inview"]]: refs.extend(norm_list(x))
    missing=[x for x in refs if pd.notna(x) and x not in known]
    out={
      "behavior_rows":len(beh),"article_rows":len(art),
      "behavior_columns":list(beh.columns),"article_columns":list(art.columns),
      "behavior_null_rate":{c:float(beh[c].isna().mean()) for c in beh.columns},
      "article_null_rate":{c:float(art[c].isna().mean()) for c in art.columns},
      "impression_date_min":beh[bc["impression_time"]].min(),
      "impression_date_max":beh[bc["impression_time"]].max(),
      "article_publish_min":art[ac["published_time"]].min(),
      "article_publish_max":art[ac["published_time"]].max(),
      "inview_article_references":len(refs),
      "inview_missing_from_articles":len(missing),
      "inview_join_coverage":1-len(missing)/len(refs) if refs else None
    }
    return bc,ac,out

def frontpage(beh,bc):
    x=beh.loc[beh[bc["article_id_context"]].isna()].copy()
    x["_slate"]=x[bc["article_ids_inview"]].map(norm_list)
    x["_slate_size"]=x["_slate"].map(len)
    return x

def session_features(fp,bc):
    x=fp.sort_values([bc["session_id"],bc["impression_time"],bc["impression_id"]]).copy()
    g=x.groupby(bc["session_id"],sort=False)
    x["_session_pos"]=g.cumcount()+1
    x["_session_len"]=g[bc["impression_id"]].transform("size")
    frac=x["_session_pos"]/x["_session_len"].clip(lower=1)
    x["_session_stage"]=pd.cut(frac,[0,1/3,2/3,1.0000001],
        labels=["early","middle","late"],include_lowest=True).astype(object)
    x.loc[x["_session_len"]==1,"_session_stage"]="single"
    def lb(n):
        if n<=1:return "1"
        if n==2:return "2"
        if n<=5:return "3-5"
        return "6+"
    x["_session_len_bin"]=x["_session_len"].map(lb)
    return x

def supply(fp,art,bc,ac):
    imp=epoch_ns(fp[bc["impression_time"]])
    pub=art[[ac["published_time"],ac["article_id"]]].dropna(subset=[ac["published_time"]]).sort_values(ac["published_time"]).copy()
    pub_ns=epoch_ns(pub[ac["published_time"]])

    prem_col=ac.get("premium")
    prem_prefix=None
    if prem_col:
        m=art.set_index(ac["article_id"])[prem_col]
        arr=pub[ac["article_id"]].map(m).fillna(False).astype(bool).to_numpy()
        prem_prefix=np.concatenate([[0],np.cumsum(arr.astype(int))])

    out={}
    for h in SUPPLY_WINDOWS_H:
        delta=int(pd.Timedelta(hours=h).value)  # ns
        L=np.searchsorted(pub_ns,imp-delta,side="left")
        R=np.searchsorted(pub_ns,imp,side="left")
        cnt=R-L
        slate=fp["_slate_size"].to_numpy()
        ratio=cnt/np.where(slate==0,np.nan,slate)
        rec={
          "supply_count":qsum(pd.Series(cnt)),
          "supply_to_slate_ratio":qsum(pd.Series(ratio)),
          "share_impressions_supply_ge_3x_slate":float(np.mean(cnt>=3*slate)),
          "share_impressions_nonempty_supply":float(np.mean(cnt>0))
        }
        if prem_prefix is not None:
            pc=prem_prefix[R]-prem_prefix[L]
            rec["premium_supply_count"]=qsum(pd.Series(pc))
            rec["nonpremium_supply_count"]=qsum(pd.Series(cnt-pc))
        out[f"{h}h"]=rec
    return out

def jaccard(a,b):
    A,B=set(a),set(b)
    if not A and not B:return 1.0
    if not A or not B:return 0.0
    return len(A&B)/len(A|B)

def overlap(fp,bc,seed,bucket,max_pairs):
    rng=np.random.default_rng(seed)
    x=fp.copy()
    x["_time_bucket"]=x[bc["impression_time"]].dt.floor(f"{bucket}min")
    device=bc["device_type"]
    # v2: exact slate size replaces scroll-depth matching because front-page
    # scroll_percentage is nearly entirely missing in the real Small bundle.
    keys=["_time_bucket",device,"_session_stage","_session_len_bin","_slate_size"]
    complete=x.dropna(subset=[device,"_session_stage","_session_len_bin","_slate_size"])
    groups=[]
    for key,g in complete.groupby(keys,dropna=False,observed=True):
        if len(g)>=2: groups.append((key,g.index.to_numpy()))
    pairs=[]
    if groups:
        for gi in rng.permutation(len(groups)):
            idx=groups[gi][1]
            perm=rng.permutation(idx)
            for a,b in zip(perm[::2],perm[1::2]):
                pairs.append((a,b))
                if len(pairs)>=max_pairs: break
            if len(pairs)>=max_pairs: break
    vals=[jaccard(x.at[a,"_slate"],x.at[b,"_slate"]) for a,b in pairs]
    return {
      "match_definition":{
        "same_time_bucket_minutes":bucket,
        "same_device_type":True,
        "same_session_stage":True,
        "same_session_length_bin":True,
        "same_exposed_slate_size":True,
        "scroll_percentage_used":False,
        "reason_scroll_not_used":"Front-page scroll_percentage was nearly entirely missing in the real Small train bundle."
      },
      "frontpage_rows":len(x),
      "rows_with_complete_match_context":len(complete),
      "complete_context_share":float(len(complete)/len(x)) if len(x) else None,
      "eligible_strata":len(groups),"sampled_pairs":len(vals),
      "jaccard":qsum(pd.Series(vals,dtype=float))
    }

def extract_embeddings(df):
    aid=resolve(df,"article_id",True)
    others=[c for c in df.columns if c!=aid]
    for c in others:
        sm=df[c].dropna().head(5)
        if len(sm) and any(isinstance(v,(list,tuple,np.ndarray)) for v in sm):
            return aid,df[aid].to_numpy(),np.vstack([np.asarray(v,float) for v in df[c]])
    nums=[c for c in others if pd.api.types.is_numeric_dtype(df[c])]
    if not nums: raise ValueError("No embedding vector detected")
    return aid,df[aid].to_numpy(),df[nums].to_numpy(float)

def topics(v): return set(str(x) for x in norm_list(v))

def embedding_test(emb,art,ac,seed,sample_n,n_boot):
    rng=np.random.default_rng(seed)
    aid,ids,X=extract_embeddings(emb)
    finite=np.isfinite(X).all(axis=1); ids,X=ids[finite],X[finite]
    cols=[ac["article_id"]]
    cat=ac.get("category"); top=ac.get("topics")
    if cat: cols.append(cat)
    if top: cols.append(top)
    meta=art[cols].drop_duplicates(ac["article_id"]).set_index(ac["article_id"])
    keep=np.array([i in meta.index for i in ids]); ids,X=ids[keep],X[keep]
    cov=len(ids)/max(1,art[ac["article_id"]].nunique())
    if len(ids)<3:return {"status":"insufficient_embedding_coverage","embedding_article_count":len(ids),"article_id_coverage":cov}
    if len(ids)>sample_n:
        take=rng.choice(len(ids),size=sample_n,replace=False); ids,X=ids[take],X[take]
    Xn=X/np.clip(np.linalg.norm(X,axis=1,keepdims=True),1e-12,None)
    nn=NearestNeighbors(n_neighbors=2,metric="cosine").fit(Xn)
    _,ind=nn.kneighbors(Xn); nidx=ind[:,1]
    ridx=rng.integers(0,len(ids),size=len(ids)); m=ridx==np.arange(len(ids)); ridx[m]=(ridx[m]+1)%len(ids)
    out={"embedding_article_count":len(ids),"article_id_coverage":float(cov),"tested_articles":len(ids)}
    if cat:
        cats=np.array([meta.at[i,cat] for i in ids],dtype=object)
        v=pd.notna(cats)&pd.notna(cats[nidx])&pd.notna(cats[ridx])
        if v.sum():
            a=(cats==cats[nidx]).astype(float)[v]; b=(cats==cats[ridx]).astype(float)[v]
            out["category_same_rate_nearest"]=ci_mean(a,rng,n_boot)
            out["category_same_rate_random"]=ci_mean(b,rng,n_boot)
            out["category_agreement_lift"]=float(a.mean()-b.mean())
    if top:
        ts=[topics(meta.at[i,top]) for i in ids]; a=[]; b=[]
        for i in range(len(ids)):
            if not ts[i]: continue
            A=ts[i]; B=ts[nidx[i]]; C=ts[ridx[i]]
            a.append(len(A&B)/len(A|B) if A|B else 0)
            b.append(len(A&C)/len(A|C) if A|C else 0)
        if a:
            a=np.asarray(a); b=np.asarray(b)
            out["topic_jaccard_nearest"]=ci_mean(a,rng,n_boot)
            out["topic_jaccard_random"]=ci_mean(b,rng,n_boot)
            out["topic_agreement_lift"]=float(a.mean()-b.mean())
    return out

def main():
    a=args(); t0=time.perf_counter(); random.seed(a.seed); np.random.seed(a.seed)
    outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    report={
      "script":"gate_a_c_v2.py","seed":a.seed,
      "fixes":["timestamp_unit_bug_fixed","frontpage_scroll_removed_from_matching","exact_slate_size_added_to_matching"],
      "precommitment_boundary":{"computes_outcome_metrics":False,"computes_discovery_headroom":False,"computes_final_cohort_disparity":False},
      "inputs":{"behaviors":a.behaviors,"articles":a.articles,"embeddings":a.embeddings},
      "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__}
    }
    try:
        beh=read_table(a.behaviors); art=read_table(a.articles); emb=read_table(a.embeddings) if a.embeddings else None
        bc,ac,sa=schema_audit(beh,art); report["A_schema_join"]=sa
        fp=session_features(frontpage(beh,bc),bc)
        scroll_col=bc.get("scroll_percentage")
        scroll_complete=float(fp[scroll_col].notna().mean()) if scroll_col else None
        report["A_frontpage"]={
          "rows":len(fp),"share_of_behavior_rows":float(len(fp)/len(beh)) if len(beh) else None,
          "slate_size":qsum(fp["_slate_size"]),
          "device_distribution":fp[bc["device_type"]].astype(str).value_counts(dropna=False).to_dict(),
          "frontpage_scroll_complete_share":scroll_complete,
          "scroll_percentage":qsum(pd.to_numeric(fp[scroll_col],errors="coerce")) if scroll_col else {"n":0},
          "session_length":qsum(fp["_session_len"])
        }
        report["A_supply_windows"]=supply(fp,art,bc,ac)
        report["B_matched_context_overlap"]=overlap(fp,bc,a.seed,a.time_bucket_min,a.max_overlap_pairs)
        report["C_embedding_agreement"]=embedding_test(emb,art,ac,a.seed,a.embedding_sample,a.embedding_bootstrap) if emb is not None else {"status":"not_run_no_embedding_file"}
        report["D_handoff_manifest"]={
          "behavior_impression_range":[beh[bc["impression_time"]].min(),beh[bc["impression_time"]].max()],
          "article_publish_range":[art[ac["published_time"]].min(),art[ac["published_time"]].max()],
          "embedding_file_loaded":emb is not None
        }
        report["status"]="completed"
    except Exception as e:
        report["status"]="failed"; report["error"]=repr(e)
        raise
    finally:
        report["runtime_total_seconds"]=time.perf_counter()-t0
        with open(outdir/"gate_report_v2.json","w") as f: json.dump(js(report),f,indent=2)
    print(json.dumps(js({
      "frontpage_rows":report.get("A_frontpage",{}).get("rows"),
      "frontpage_scroll_complete_share":report.get("A_frontpage",{}).get("frontpage_scroll_complete_share"),
      "24h_supply_median":report.get("A_supply_windows",{}).get("24h",{}).get("supply_count",{}).get("median"),
      "24h_supply_to_slate_median":report.get("A_supply_windows",{}).get("24h",{}).get("supply_to_slate_ratio",{}).get("median"),
      "matched_pairs":report.get("B_matched_context_overlap",{}).get("sampled_pairs"),
      "matched_jaccard_median":report.get("B_matched_context_overlap",{}).get("jaccard",{}).get("median"),
      "embedding_status":report.get("C_embedding_agreement",{}).get("status","ran")
    }),indent=2))
    print(f"Wrote {outdir/'gate_report_v2.json'}")

if __name__=="__main__":
    main()
