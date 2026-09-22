# Phase 1 Implementation Clarifications and Technical Lock

**Status**: FROZEN BEFORE VALIDATION OUTCOME ANALYSIS  
**Date**: 2026-09-21  
**Protocol Commit**: `cb14d50736d1b8c8202b1aa67158e15c3eabb804`  
**Base Commit**: `db8d1003358f50d51e66672ca3a6a8e8f02d1c0a`  

This document records the exact operational specifications and pre-implementation technical decisions resolved prior to computing or opening any held-out validation outcomes. These decisions operationalize `analysis_protocol_v1.yaml` and `analysis_protocol_notes.md` without altering the research question, design, or parameter boundaries.

---

### 1. Forward-Chained Out-of-Fold Relevance Estimation
- Train front-page impressions are divided into 5 contiguous chronological blocks $B_1, \dots, B_5$.
- **Boundary Tie-Breaking Rule**: Forward-chain block boundaries must never split impressions sharing the same `impression_time`. Blocks are approximately equal-sized by impression count, but when a boundary falls inside a timestamp tie, all impressions with that timestamp must be assigned to the later block. For every scored block:
  $$\max(\text{training impression\_time}) < \min(\text{scored impression\_time})$$
  must hold strictly.
- Model fitting strictly uses forward-chaining to prevent future observations from informing past predictions:
  - Block $B_1$: Warm-up fold.
  - Train on $B_1 \to$ Predict $B_2$.
  - Train on $B_1 \cup B_2 \to$ Predict $B_3$.
  - Train on $B_1 \cup B_2 \cup B_3 \to$ Predict $B_4$.
  - Train on $B_1 \cup B_2 \cup B_3 \cup B_4 \to$ Predict $B_5$.
- Out-of-fold diagnostic metrics (AUC, nDCG, Brier score) and the relevance floor $\tau$ are computed strictly on pooled out-of-fold predictions from blocks $B_2$ through $B_5$.
- $\tau$ is frozen as the 20th percentile of $\hat{p}_{\text{OOF}}$ among clicked ($y=1$) items in $B_2$ through $B_5$.
- The production relevance model is subsequently fit on all eligible train examples ($B_1 \dots B_5$).

### 2. Cross-Period User Identity Boundary
- Users may legitimately appear in both train and validation splits. User IDs are not required to be disjoint.
- Leakage boundaries strictly prohibit:
  - Validation behavior records or labels in model fitting.
  - Validation impression IDs in training tables.
  - Target-period validation behavior outcomes in user-history profile construction (history profiles use pre-period 21-day history only).
  - Validation outcomes in calculating $\tau$ or breadth tertiles.

### 3. Strict Observable-Supply Temporal Boundary
- For an impression at $t_{\text{imp}}$, observable supply requires:
  $$t_{\text{imp}} - \text{window} \le t_{\text{pub}} < t_{\text{imp}}$$
- Articles published at exactly $t_{\text{imp}}$ or later are strictly excluded ($t_{\text{pub}} < t_{\text{imp}}$).
- Applied identically to 12h, 24h, and 48h windows.

### 4. Unique User–Article Semantic Novelty Caching
- To avoid redundant $768$-dimensional embedding dot products across repeating impressions:
  - For each eligible user, determine the union of candidate articles appearing in their observed slates and observable supply.
  - Compute the user-history $\times$ unique-candidate similarity matrix once.
  - Derive and cache novelty for $k=5$ (primary) as well as $k=3$ and $k=10$ (robustness).
  - Cache table: `cache/validation/user_article_novelty.parquet` with columns:
    `user_id`, `article_id`, `novelty_k3`, `novelty_k5`, `novelty_k10`.

### 5. Deterministic 1:1 Matched Comparison Algorithm
- Low-vs-high breadth comparison uses greedy 1:1 nearest-neighbour matching without replacement.
- Exact matching dimensions:
  1. `_time_bucket`: 30-minute interval (`dt.floor("30min")`)
  2. `device_type`: exact raw categorical value
  3. `is_subscriber`: exact
  4. `_session_stage`: exact (`single`, `early`, `middle`, `late`)
  5. `_session_len_bin`: exact (`1`, `2`, `3-5`, `6+`)
  6. `_history_len_bin`: exact train quartile bin
- Within exact stratum, caliper on observed slate size:
  $$|\text{slate\_size}_{\text{low}} - \text{slate\_size}_{\text{high}}| \le 1$$
- **Matching Execution Order**:
  Within each exact matching stratum, Low-breadth impressions are processed in ascending order of:
  1. `impression_time`
  2. `impression_id`
  For each Low impression, select the available High-breadth impression using:
  1. Smallest absolute slate-size difference.
  2. Smallest absolute impression-time difference.
  3. Smallest `impression_id`.
  Once matched, the High impression is removed from the candidate pool. This ordering is deterministic and frozen before validation analysis.
- Each impression participates in at most one matched pair.
- Medium-breadth users remain purely descriptive/secondary.

### 6. Raw Device Type Encoding
- Raw values of `device_type` from the source data are treated directly as categorical levels without speculative hardware mappings (e.g. desktop vs mobile).

### 7. Topic Affinity Metric Definition
- For candidate article $i$ with topic set $T_i$:
  $$\text{TopicAffinity}(u, i) = \frac{1}{|T_i|} \sum_{t \in T_i} \frac{\text{count of clicked historical articles of } u \text{ containing topic } t}{\text{total clicked historical articles of } u}$$
- If $|T_i| = 0$, affinity is set to $0.0$.
- Derived exclusively from the user's pre-period history.

### 8. Fold-Local Preprocessing and Relevance Model Pipeline
- **Fold-Local Preprocessing**: For every forward-chained OOF fit, the entire scikit-learn preprocessing pipeline must be fitted only on that fold's historical training blocks:
  - $B_1$ preprocessing + model $\to B_2$
  - $B_1 \cup B_2$ preprocessing + model $\to B_3$
  - $B_1 \cup B_2 \cup B_3$ preprocessing + model $\to B_4$
  - $B_1 \cup B_2 \cup B_3 \cup B_4$ preprocessing + model $\to B_5$
  No scaler, encoder, imputer, or other fitted preprocessing object may be fit globally before OOF scoring.
- Pipeline Components:
  - `StandardScaler` for continuous numerical features (fitted fold-locally during OOF; fitted on $B_1 \dots B_5$ for production).
  - `OneHotEncoder(handle_unknown='ignore')` for `device_type` (fitted fold-locally during OOF; fitted on $B_1 \dots B_5$ for production).
  - `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight=None)`.
- Hyperparameters, feature representations, and scaling are frozen without post-hoc tuning.
- Model coefficients and feature names are saved for auditability.

### 9. Time-Safe Popularity & Boundary Handling
- Popularity signal: $w(i, t) = (1 + \text{clicks\_in\_prior\_24h}(i, t)) \cdot 2^{-\text{age\_hours}/12}$.
- Temporal condition: Only clicks with $t_{\text{click}} < t_{\text{imp}}$ contribute.
- For validation impressions early in the validation period, lookbacks spanning into the late train period are permitted as pure historical lookbacks.
- Unit tests specifically verify the train $\to$ validation lookback boundary.

### 10. Non-Subscriber Premium Choice Rule
- For non-subscribers (`is_subscriber == False`):
  - Premium articles are excluded from the choosable exposed slate $S_{\text{choosable}}$.
  - Premium clicked articles are excluded from the consumed set.
  - If no eligible non-premium consumed article remains for an impression, that impression does not contribute to the primary choice-stage gap metric.
  - Premium exposures are retained for descriptive reporting.

### 11. Rigorous Null Slate Sampling & Performance
- The 200 null draws per eligible impression required by the protocol are executed using chunked, vectorized sampling without replacement (Efraimidis-Spirakis key sorting).
- Pre-computation of per-impression metrics and per-user aggregation ensures reproducible 2,000-replicate user-clustered bootstrapping without re-running sampling loops.
- Benchmarks are measured directly on the environment rather than using speculative timing estimates.

### 12. Verification & Auditability
- The parameter files `analysis_protocol_v1.yaml` and `analysis_protocol_notes.md` remain unmodified.
- These decisions serve as the technical specification for all subsequent pipeline modules.
