# Modular Implementation Plan: EB-NeRD Phase 1 Analysis Pipeline

**Status**: FROZEN BEFORE VALIDATION OUTCOME ANALYSIS  
**Protocol Commit**: `cb14d50736d1b8c8202b1aa67158e15c3eabb804`  
**Base Commit**: `db8d1003358f50d51e66672ca3a6a8e8f02d1c0a`  
**Clarifications File**: `IMPLEMENTATION_DECISIONS.md`  

This plan defines the modular implementation of the Phase 1 analysis pipeline strictly following `analysis_protocol_v1.yaml` and incorporating all pre-implementation technical corrections frozen in `IMPLEMENTATION_DECISIONS.md`.

---

## 1. Train-Side Pipeline

### Objective & Boundary
Construct all user representations, time-safe click histories, exposed-item relevance examples, forward-chained out-of-fold relevance predictions, and the frozen relevance threshold $\tau$ strictly from the `train` split. Zero validation records, impression IDs, or behavior labels enter this stage.

### Modular Components
- `src/data.py`: Loads `ebnerd_small/articles.parquet`, `ebnerd_small/train/behaviors.parquet`, and `ebnerd_small/train/history.parquet`. Enforces schema integrity, extracts front-page impressions (`article_id.isna()`), and parses list columns with `norm_list()`.
- `src/embeddings.py`: Ingests `contrastive_vector.parquet` (primary) and `bert_base_multilingual_cased.parquet` (robustness). Pre-normalizes all 768-d vectors ($\|v\|_2 = 1.0$) into contiguous C-ordered float32 matrices mapped by `article_id`.
- `src/profiles.py`: Analyzes each user's 21-day clicked history (`train/history.parquet`):
  - Filters users by `history_length >= 15`.
  - Aggregates category distribution and calculates Shannon entropy: $H(u) = -\sum p_c \ln(p_c)$.
  - Assigns users to breadth cohorts using the frozen cutpoints:
    - **Low breadth**: $H(u) \le 1.4638116962471364$
    - **Medium breadth**: $1.4638116962471364 < H(u) \le 1.5905145971043473$
    - **High breadth**: $H(u) > 1.5905145971043473$
  - Assigns users to history length quartiles: $[15, 49]$, $[50, 111]$, $[112, 244]$, $[245, \infty)$.
  - Stores unique clicked topic sets and historical embedding matrices.
- `src/popularity.py`: Ingests all train clicks with nanosecond UTC timestamps (`epoch_ns`). Builds a sorted index of click timestamps per article.
- `src/relevance.py`:
  - Extracts exposed interaction pairs $(u, i)$ from train front-page slates ($5 \le \text{slate\_size} \le 26$, user history $\ge 15$).
  - Labels: $y=1$ if $i \in \text{article\_ids\_clicked}$, $y=0$ if $i \in \text{article\_ids\_inview} \setminus \text{article\_ids\_clicked}$.
  - Computes features strictly within train.
  - Partitions train impressions chronologically into 5 contiguous chronological blocks $B_1 \dots B_5$.
  - **Forward-Chain Timestamp Boundary Rule**:
    Forward-chain block boundaries must never split impressions sharing the same `impression_time`. Blocks should be approximately equal-sized by impression count, but when a boundary falls inside a timestamp tie, all impressions with that timestamp must be assigned to the later block. For every scored block:
    $$\max(\text{training impression\_time}) < \min(\text{scored impression\_time})$$
    must hold strictly.
  - Uses **forward-chaining**:
    - Block $B_1$: Warm-up fold.
    - Train on $B_1 \to$ Predict $B_2$.
    - Train on $B_1 \cup B_2 \to$ Predict $B_3$.
    - Train on $B_1 \cup B_2 \cup B_3 \to$ Predict $B_4$.
    - Train on $B_1 \cup B_2 \cup B_3 \cup B_4 \to$ Predict $B_5$.
  - **Fold-Local Preprocessing**: For every forward-chained OOF fit, the entire scikit-learn preprocessing pipeline (StandardScaler, OneHotEncoder) must be fitted strictly on that fold's historical training blocks:
    - $B_1$ preprocessing + model $\to B_2$
    - $B_1 \cup B_2$ preprocessing + model $\to B_3$
    - $B_1 \cup B_2 \cup B_3$ preprocessing + model $\to B_4$
    - $B_1 \cup B_2 \cup B_3 \cup B_4$ preprocessing + model $\to B_5$
    No scaler, encoder, imputer, or other fitted preprocessing object may be fit globally before OOF scoring.
  - Calculates out-of-fold diagnostic metrics (OOF AUC, OOF nDCG, Brier score) strictly on pooled predictions from $B_2 \dots B_5$.
  - Freezes $\tau$: Computes the exact 20th percentile of $\hat{p}_{\text{OOF}}(u, i)$ among positive examples ($y=1$) across blocks $B_2 \dots B_5$.
  - Fits the final production preprocessing pipeline and relevance model on all eligible train examples ($B_1 \dots B_5$).
- **Caching**: Saves `cache/train/user_profiles.parquet`, `cache/train/click_index.npz`, `cache/train/oof_predictions.parquet`, `cache/train/relevance_model.joblib`, and `cache/train/tau.json`.

---

## 2. Relevance Model Pipeline

### Formulation & Labels
- **Instance**: Candidate article $i$ appearing in front-page slate $S$ for user $u$ at impression time $t$.
- **Positive ($y=1$)**: Item in `article_ids_clicked`.
- **Negative ($y=0$)**: Item in `article_ids_inview` and not in `article_ids_clicked`.

### Feature Families (Strictly Frozen)
1. **Semantic Similarity to User History**: Top-5 mean cosine similarity between candidate article embedding and the user's historical clicked article embeddings:
   $$\text{sim}(u, i) = \frac{1}{5} \sum_{h \in \text{top-5}(H_u)} \langle v_i, v_h \rangle$$
2. **Publisher Category Affinity**: Proportion of articles in $H_u$ belonging to the candidate article's publisher category.
3. **Topic Affinity**: For candidate article $i$ with topic set $T_i$, topic affinity is:
   $$\text{TopicAffinity}(u, i) = \frac{1}{|T_i|} \sum_{t \in T_i} \frac{\text{count of clicked historical articles of } u \text{ containing topic } t}{\text{total clicked historical articles of } u}$$
   If $|T_i| = 0$, affinity is $0.0$.
4. **Article Freshness / Age**: $\log(1 + \text{age\_hours})$, where $\text{age\_hours} = (t_{\text{imp}} - t_{\text{pub}}) / 3600\,\text{s}$.
5. **Pre-Impression Click Popularity**: $\log(1 + \text{clicks\_in\_prior\_24h})$.
6. **Premium $\times$ Subscriber Interaction**: `is_premium` (0/1), `is_subscriber` (0/1), and their product `is_premium * is_subscriber` (0/1).
7. **Device Type**: Raw categorical levels of `device_type` one-hot encoded directly from train (`handle_unknown='ignore'`).
8. **Time-of-Day**: Cyclical representations $\sin(2\pi \cdot \text{hour} / 24)$ and $\cos(2\pi \cdot \text{hour} / 24)$.

### Frozen Model Architecture
- Continuous numerical features standardized using fold-local `StandardScaler` during OOF, and full-train `StandardScaler` for production.
- Raw categorical `device_type` encoded with fold-local `OneHotEncoder(handle_unknown='ignore')` during OOF, and full-train `OneHotEncoder` for production.
- Classifier:
  $$\text{LogisticRegression}(\text{penalty}="l2", C=1.0, \text{solver}="lbfgs", \text{max\_iter}=1000, \text{class\_weight}=\text{None})$$
- All hyperparameters, feature representations, and scalers are frozen without validation tuning. Coefficients and feature names are recorded for auditability.

### Metrics & Qualification
- **OOF AUC**: Global ROC-AUC computed across out-of-fold interactions in $B_2 \dots B_5$.
- **OOF nDCG**: Evaluated per impression with $\ge 1$ click ($\text{DCG} / \text{IDCG}$), reported as mean across impressions.
- **Calibration**: Brier score $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$.
- **Validation Qualification Gate**:
  $$\text{val\_AUC} \ge 0.60 \quad \text{AND} \quad \text{val\_nDCG} \ge \text{popularity+recency baseline nDCG} + 0.01$$
  Baseline ranks items in the slate strictly by $w_i = (1 + \text{clicks\_24h}[i]) \cdot 2^{-\text{age}[i]/12}$. If condition fails, Headroom is stamped `NOT QUALIFIED`.

---

## 3. Time-Safe Popularity

### Weight Formula
$$w(i, t) = (1 + \text{clicks\_in\_prior\_24h}(i, t)) \cdot 2^{-\frac{\text{age\_hours}(i, t)}{12}}$$

### Integrity Rules
- Only clicks with $t - 24\,\text{h} \le t_{\text{click}} < t$ contribute (strictly prior to impression time).
- Forward-looking aggregate fields (`total_pageviews`, `total_inviews`, `total_read_time`) are excluded.
- Zero-click articles receive pseudocount 1: $(1 + 0) \cdot 2^{-\text{age}/12} = 2^{-\text{age}/12} > 0$.
- Sorted integer nanosecond timestamp index per article with `np.searchsorted` allows efficient lookup.
- Early validation impressions naturally query late train clicks across the temporal boundary.

---

## 4. Validation-Side Pipeline (Design Only, Unopened)

### A. AVAILABLE (Observable Supply)
- Surface: Front-page impressions (`article_id.isna()`).
- Strict supply temporal boundary:
  $$t_{\text{imp}} - \text{window} \le t_{\text{pub}} < t_{\text{imp}}$$
  Articles published at exactly $t_{\text{imp}}$ or later are strictly excluded.
- Eligibility: $\text{supply\_count} \ge 3 \times |S|$, $5 \le |S| \le 26$, and $\text{user\_history\_len} \ge 15$.
- Sensitivities: 12h and 48h windows.

### B. EXPOSED (Observed Exposure)
- Extracted as `article_ids_inview`. Treated strictly as an unordered set of size $K$. No positional ranking claims.

### C. CONSUMED (Observed Choice)
- Articles in `article_ids_clicked` that belong to the exposed slate.
- **Non-Subscriber Premium Choice Rule**: For non-subscribers (`is_subscriber == False`), premium articles (`premium == True`) are excluded from both the choosable exposed set $S_{\text{choosable}}$ and the clicked consumed set. If no eligible non-premium consumed article remains, the impression does not contribute to the primary choice-stage gap. Premium exposures remain for descriptive reporting.

### D. Semantic Novelty Caching
- For each eligible user, determine the union of candidate articles appearing in their observed slates and observable supply.
- Compute the user-history $\times$ candidate similarity matrix once per unique $(u, i)$ pair.
- Cache $k=5$ (primary), $k=3$, and $k=10$ (robustness) in `cache/validation/user_article_novelty.parquet` with columns:
  `user_id`, `article_id`, `novelty_k3`, `novelty_k5`, `novelty_k10`.

### E. NULL SLATES (Chunked Memory-Safe 200 Draws)
- Primary null: 200 draws of size $K$ without replacement using weights $w(i, t)$ via vectorized Efraimidis-Spirakis sampling ($Key_i = \ln(U_i) / w_i$).
- Reference null: 200 uniform random draws of size $K$ without replacement.
- Evaluated in memory-safe chunks. Baseline expectations cached once per impression.

### F. Discovery Metric & Decomposition
- $d(u, i) = \text{novelty}(u, i)$ if $\hat{r}(u, i) \ge \tau$ else $0$. Slate: $D(u, S) = \frac{1}{|S|} \sum_{i \in S} d(u, i)$.
- Decomposition into relevance coverage and conditional novelty.

### G. Stage Effects
- Available $\to$ Exposed: $\Delta_{\text{exposure}} = D(u, S_{\text{exposed}}) - E[D(u, S_{\text{primary\_null}})]$.
- Exposed $\to$ Consumed: $\Delta_{\text{choice}} = D(u, S_{\text{consumed}}) - D(u, S_{\text{choosable}})$.

### H. Matched Breadth Analysis (Deterministic 1:1 Matching)
- Cohorts: Low vs High breadth using frozen train entropy cutpoints. Medium is descriptive only.
- Exact matching dimensions:
  1. `_time_bucket`: 30-minute interval (`dt.floor("30min")`)
  2. `device_type`: exact raw categorical value
  3. `is_subscriber`: exact
  4. `_session_stage`: exact (`single`, `early`, `middle`, `late`)
  5. `_session_len_bin`: exact (`1`, `2`, `3-5`, `6+`)
  6. `_history_len_bin`: exact train quartile bin
- Within exact stratum, slate size caliper: $|\text{slate\_size}_{\text{low}} - \text{slate\_size}_{\text{high}}| \le 1$.
- **Matching Execution Order**:
  Within each exact matching stratum, Low-breadth impressions are processed in ascending order of:
  1. `impression_time`
  2. `impression_id`
  For each Low impression, select the available High-breadth impression using:
  1. Smallest absolute slate-size difference.
  2. Smallest absolute impression-time difference.
  3. Smallest `impression_id`.
  Once matched, the High impression is removed from the candidate pool. This ordering is deterministic and frozen before validation analysis.
- Greedy matching without replacement; each impression in at most one pair.
- Reports: eligible counts, matched pairs, match rates, unmatched counts.

### I. Discovery Headroom (Conditional)
- Executed only if validation qualification gate passes.
- Substitution pool: $24\text{h observable supply} \cap \text{common support } (\ge 3 \text{ distinct users in 30m bucket}) \cap \{c \mid \hat{r}(u, c) \ge \tau\} \setminus S$.
- At most 2 swaps under constraint: alternative slate mean predicted relevance $\ge 97\%$ of observed slate mean predicted relevance. Sensitivities: 1% and 5%. Computed once per impression.

### J. Uncertainty (User-Clustered Bootstrap)
- Aggregate all impression-level metrics to the user level first.
- Resample users with replacement across 2,000 bootstrap replicates with fixed seed `20260919`.
- Compute 95% percentile confidence intervals: $[\text{quantile}(0.025), \text{quantile}(0.975)]$.
- Compute standardized effect sizes using metric-specific between-user SDs relative to $0.10\,\text{SD}$ practical threshold.

---

## 5. Leakage Prevention Guards

1. **Cross-Period User Boundary**: Users may appear in both train and validation splits. Leakage prevention strictly isolates validation behavioral impressions, clicks, labels, and target-period behaviors from model fitting and user-history profiling.
2. **Chronological Boundary Isolation**: Assert $\max(t_{\text{train\_imp}}) < \min(t_{\text{val\_imp}})$. Assert zero validation `impression_id` or validation behavior rows appear in training tables.
3. **Forward-Chain Timestamp Boundary Assertion**: Assert $\max(\text{training impression\_time}) < \min(\text{scored impression\_time})$ for every fold in forward-chained OOF modeling.
4. **Fold-Local Preprocessing Isolation**: Assert no scaler, encoder, or preprocessor fit on subsequent folds or global data is used during OOF prediction.
5. **Time-Safe Popularity Guard**: Assert $\max(t_{\text{counted\_clicks}}) < t_{\text{imp}}$ for every impression. Forbidden columns (`total_pageviews`, etc.) are never loaded.
6. **Supply Time Window Guard**: Assert all supply items satisfy $t_{\text{imp}} - \text{window} \le t_{\text{pub}} < t_{\text{imp}}$. Assert $t_{\text{pub}} \ge t_{\text{imp}}$ yields 0 articles.
7. **Parameter Immutability Guard**: Assert $\tau$, breadth tertile cutpoints, and history length bins are read strictly from train artifacts and the frozen protocol YAML.

---

## 6. Test Suite Plan

All tests located in `tests/` and run via `unittest`:
- `tests/test_supply.py`: Verify `epoch_ns()`, strict $t_{\text{pub}} < t_{\text{imp}}$ boundary, 12h/24h/48h windows, and $3\times$ slate-size filter.
- `tests/test_no_leakage.py`: Assert zero validation behavior rows/labels in train, test future simulated clicks are ignored, verify post-impression publications excluded.
- `tests/test_popularity.py`: Unit-test popularity formula, pseudocount, and train $\to$ validation lookback transition.
- `tests/test_relevance.py`: Verify forward-chaining logic ($B_1 \to B_2$, etc.), tie-break boundary assignment, fold-local preprocessing isolation, verify $\tau$ equals 20th percentile of clicked items in $B_2 \dots B_5$, verify AUC/nDCG calculations.
- `tests/test_null_slates.py`: Verify drawn slates have exact size $K$, verify zero duplicate items (without replacement), verify fixed seed reproducibility.
- `tests/test_matching.py`: Test deterministic 1:1 matching, tie-breaking order (Low sorted by time then id, High chosen by slate-size diff then time diff then id), caliper $\le 1$, and match rate diagnostics.
- `tests/test_headroom.py`: Verify swaps $\le 2$, individual item $\tau$ clearance, common support $n \ge 3$, and $97\%$ relevance retention.
- `tests/test_bootstrap.py`: Verify bootstrap resamples user entities with replacement, verify CI coverage logic.
- `tests/test_reproducibility.py`: End-to-end determinism check with seed `20260919`.

---

## 7. Output Contract

### Machine-Readable Outputs (`outputs/`)
- `run_manifest.json`: Commit hash, seed, protocol version, environment metadata, execution timestamps.
- `relevance_model_metrics.json`: Forward-chained train OOF metrics ($B_2 \dots B_5$), $\tau$, validation AUC/nDCG, baseline nDCG, Headroom qualification status.
- `stage_effects.json`: Available $\to$ Exposed and Exposed $\to$ Consumed effect sizes, 95% CIs, standardized effect sizes.
- `null_comparison.json`: Observed discovery, primary null expectation, random null expectation, decomposition.
- `breadth_comparison.json`: Low vs High breadth discovery distributions, match rates, matched differences, 95% CIs.
- `headroom.json`: Headroom effect size and sensitivities (if qualified) or formal disqualification audit.
- `robustness.json`: Direction, magnitude, CI, and primary conclusion stability across the pre-specified robustness battery.
- `outcome_interpretation.json`: Formal evaluation against the 7 pre-specified hypotheses.
- Tidy CSVs for all outputs (`stage_effects.csv`, `null_comparison.csv`, `breadth_comparison.csv`, `headroom.csv`, `robustness.csv`).

---

## 8. Execution Order & Checkpoints

```
[Phase 0: Precommitment Check]
  │  Verify Git commit cb14d50 predates run.
  │  Print "Protocol loaded. Validation outcome analysis has not yet started."
  ▼
[Phase 1: Train Artifact Construction] (scripts/01_build_train_artifacts.py)
  │  Build train user profiles, click index, and feature tables.
  ▼
[Phase 2: Relevance Model Fitting] (scripts/02_fit_relevance_model.py)
  │  Partition train impressions into B1..B5 with strict timestamp tie-break rule.
  │  Run forward-chaining with fold-local preprocessing (B1->B2, B1+B2->B3, etc.).
  │  Calculate OOF AUC, nDCG, Brier score on B2-B5.
  │  Freeze tau at 20th percentile of clicked train items in B2-B5.
  │  Fit final production relevance model and preprocessing on all eligible train data.
  ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ CHECKPOINT 1: PRE-OUTCOME INTEGRITY GATE (STOP & VERIFY)                      ║
║  - Run all unit tests (tests/test_*.py) on synthetic & train data.            ║
║  - Confirm 100% test pass.                                                    ║
║  - Verify zero validation outcomes have been opened or computed.              ║
║  - Print confirmation: "Train artifacts frozen. Ready for validation."        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
  │
  ▼
[Phase 3: Validation Model Qualification]
  │  Score validation impressions; compute held-out AUC and nDCG vs baseline.
  │  Evaluate Headroom qualification rule (AUC >= 0.60, nDCG >= baseline + 0.01).
  │  Record Headroom status: QUALIFIED or NOT QUALIFIED.
  ▼
[Phase 4: Validation Phase 1 Analysis] (scripts/03_run_validation_analysis.py)
  │  Filter eligible front-page validation slates (5-26, supply >= 3x, hist >= 15).
  │  Cache user-article novelty pairs (k=5, 3, 10).
  │  Generate chunked 200 null slates and random reference slates.
  │  Compute Available -> Exposed and Exposed -> Consumed stage effects.
  │  Run deterministic 1:1 matching for low-vs-high breadth comparison.
  │  Compute Headroom if and only if QUALIFIED.
  │  Aggregate to user level; run 2,000 user-clustered bootstrap replicates.
  ▼
[Phase 5: Robustness Battery] (scripts/04_run_robustness.py)
  │  Execute pre-specified robustness checks:
  │  12h/48h supply, multilingual BERT, k=3/10, n=5, 1%/5% headroom tolerances.
  ▼
[Phase 6: Final Asset Export] (scripts/05_export_phase1_assets.py)
  │  Export outputs/*.json and tidy CSVs.
  │  Generate findings summary and methodology documentation.
```
