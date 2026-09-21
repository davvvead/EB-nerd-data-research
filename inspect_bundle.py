#!/usr/bin/env python3
"""
Inspect an EB-NeRD bundle before the parameter-freeze commit.

Reports:
- file manifest + sizes
- train/validation behavior date ranges
- train/validation history schema and actual min/max history timestamps
- observed history span relative to each behavior split
- candidate embedding/artifact files found under the supplied root

No outcome metrics are computed.
"""
from pathlib import Path
import argparse, json, sys, platform, time
import pandas as pd
import numpy as np

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--out",default="bundle_report.json")
    return p.parse_args()

def js(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return None if np.isnan(x) else float(x)
    if isinstance(x,pd.Timestamp): return x.isoformat()
    if isinstance(x,Path): return str(x)
    if isinstance(x,dict): return {str(k):js(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [js(v) for v in x]
    return x

def flatten_times(series):
    vals=[]
    for v in series:
        if v is None: continue
        try:
            for t in v:
                if pd.notna(t): vals.append(t)
        except TypeError:
            if pd.notna(v): vals.append(v)
    if not vals: return None,None,0
    s=pd.to_datetime(pd.Series(vals),errors="coerce",utc=True).dropna()
    if s.empty: return None,None,0
    return s.min(),s.max(),len(s)

def inspect_table(path):
    df=pd.read_parquet(path) if path.suffix.lower() in {".parquet",".pq"} else pd.read_csv(path)
    rec={"rows":len(df),"columns":list(df.columns)}
    for c in ["impression_time","published_time"]:
        if c in df.columns:
            s=pd.to_datetime(df[c],errors="coerce",utc=True)
            rec[f"{c}_min"]=s.min(); rec[f"{c}_max"]=s.max()
    if "impression_time_fixed" in df.columns:
        mn,mx,n=flatten_times(df["impression_time_fixed"])
        rec["history_timestamp_min"]=mn; rec["history_timestamp_max"]=mx; rec["history_timestamp_values"]=n
    return rec

def main():
    a=parse(); root=Path(a.root); t0=time.perf_counter()
    files=[p for p in root.rglob("*") if p.is_file()]
    manifest=[]
    for p in files:
        manifest.append({"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"mb":round(p.stat().st_size/1024/1024,3)})
    report={
      "root":str(root.resolve()),
      "environment":{"python":sys.version,"platform":platform.platform(),"pandas":pd.__version__},
      "total_files":len(files),
      "total_mb":round(sum(p.stat().st_size for p in files)/1024/1024,3),
      "manifest":manifest,
      "candidate_artifact_files":[str(p.relative_to(root)) for p in files if any(k in p.name.lower() for k in ["embed","vector","artifact","bert","roberta","xlm"])],
      "splits":{}
    }
    for split in ["train","validation"]:
        sdir=root/split
        rec={}
        for name in ["behaviors.parquet","history.parquet"]:
            p=sdir/name
            if p.exists():
                rec[name]=inspect_table(p)
            else:
                rec[name]={"missing":True}
        b=rec.get("behaviors.parquet",{})
        h=rec.get("history.parquet",{})
        if b.get("impression_time_min") and h.get("history_timestamp_min"):
            start=pd.Timestamp(b["impression_time_min"])
            rec["history_days_before_behavior_start"]=(start-pd.Timestamp(h["history_timestamp_min"])).total_seconds()/86400
            rec["history_gap_to_behavior_start_hours"]=(start-pd.Timestamp(h["history_timestamp_max"])).total_seconds()/3600
        report["splits"][split]=rec

    report["runtime_seconds"]=time.perf_counter()-t0
    Path(a.out).write_text(json.dumps(js(report),indent=2))
    print(json.dumps(js({
      "root":report["root"],"total_mb":report["total_mb"],
      "candidate_artifact_files":report["candidate_artifact_files"],
      "train":report["splits"].get("train"),
      "validation":report["splits"].get("validation")
    }),indent=2))
    print(f"Wrote {a.out}")

if __name__=="__main__":
    main()
