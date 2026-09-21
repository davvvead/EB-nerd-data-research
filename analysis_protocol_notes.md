# Phase 1 Analysis Protocol Note

This note documents the locked Phase 1 analysis configuration prior to computing
held-out validation outcomes.

## Locked train-derived decisions

- Primary semantic representation: contrastive vectors.
- Robustness semantic representation: multilingual BERT vectors.
- Embedding coverage: both representations map to 100% of article IDs in the analysis table.
- Minimum user history length: 15 interactions.
- Retained share at minimum history threshold: 0.8978405864095622.
- Primary breadth cohorts: category-entropy tertiles from eligible training users.
- Breadth thresholds:
  - Low: entropy <= 1.463812
  - Medium: 1.463812 < entropy <= 1.590515
  - High: entropy > 1.590515
- History-length matching bins: 15-49, 50-111, 112-244, 245+.
- Eligible observed slate-size range: 5-26 (training 5th-95th percentile).
- Primary observable supply window: 24 hours.
- Supply sensitivity windows: 12 and 48 hours.
- Common-support threshold (primary): n >= 3 distinct users in the same 30-minute bucket.
- Common-support sensitivity threshold: n >= 5.
- Primary null construction: popularity plus recency, same-size sampling without replacement.
- Primary null weight: (1 + prior_24h_clicks) * 2^(-article_age_hours / 12).
- Relevance floor: 20th percentile of out-of-fold predicted relevance among clicked training items.
- Headroom reporting condition: relevance model must reach AUC >= 0.60 and exceed
  popularity+recency baseline held-out nDCG by at least 0.01.
- Uncertainty estimation: 2000 user-clustered bootstrap replicates.

## Change-control rule

Commit and push the parameter file and this note before running the held-out
Phase 1 outcome analysis.

Any subsequent protocol change must be labeled as one of:

1. Bug fix
2. Pre-specified sensitivity analysis
3. Explicit protocol deviation

No post-outcome parameter tuning should be performed without explicit change documentation.
