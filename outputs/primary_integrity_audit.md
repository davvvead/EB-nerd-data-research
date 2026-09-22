# Primary Result Integrity Audit & Frozen State Report

**Freeze Date**: 2026-09-22  
**Status**: `PRIMARY_STATE_FROZEN_READY_FOR_PRESPECIFIED_ROBUSTNESS`  
**Git HEAD**: `0e6ba855de6b9ddd6b98556fdc3fee39996a9041`  
**Random Seed**: `20260919` | **Protocol Version**: `phase1-parameter-freeze-v1`  
**Prespecified Robustness Variants Executed**: **0** (Zero executed)

---

## 1. Provenance & Cryptographic Checksums (SHA-256)

### Protocol & Decision Files
| File | SHA-256 Checksum |
| :--- | :--- |
| `analysis_protocol_v1.yaml` | `df7ca427f39aee143a9a3aeb8a2ca4d59356dac43a25843995915d0ecac933b1` |
| `analysis_protocol_notes.md` | `9ae1ab3b7a593a45665ad5abe31433f0e9331885cdbd493724aafb8a9a16edf6` |
| `IMPLEMENTATION_DECISIONS.md` | `e699155d673d094969cfd66739521c7e3e8135f5882bc5e4893874454fc1d6af` |
| `implementation_plan.md` | `b0428e0bbb2f72020e904e8a2f2dc316477d8f594e542af4d03d5f8ee7985e6f` |

### Pipeline Scripts & Source Code
| File | SHA-256 Checksum |
| :--- | :--- |
| `scripts/03_run_validation_analysis.py` | `598a6209e018c896424acbe0c8713557960f0ba9595158e32b6f08f328e9c689` |
| `src/validation.py` | `53b7305f2d368ffc8c691b5150435e134f929cfb4d0d1da327a43c2cc3a06a4d` |
| `src/discovery.py` | `18cda2e80c9a057bc80236be240732896d1d6fbd977a5736ae713100f563e737` |
| `src/headroom.py` | `27bf326e850977b4b950a170c0b94e8be493ebdf77cabbc1c270b0bf0a86f6b1` |
| `src/matching.py` | `3d7fd04e10340a02e856d679ae171005446a1532f0ef9fc0f72c277f26c25ece` |
| `src/nulls.py` | `b8671cc23358657796fb02438a2935b884892879304f6a1d7c920905658b36eb` |
| `src/bootstrap.py` | `3345998176ecc4ed7aaf8cdaec1c8e7d7ce3e4ea8ca7e15fb9471ddc178b3309` |
| `src/config.py` | `70bb17eb723c2dcbdb4b9cd3e333c28a8e8a0078cd2d93dec01bd541f5b5846c` |

### Primary Frozen Train Artifacts
| File | SHA-256 Checksum |
| :--- | :--- |
| `cache/train/tau.json` | `c51de88b92a92463231f3c2d9bc827d7cca452ab184cc5aac3894154394cda69` |
| `cache/train/train_manifest.json` | `5b80baf07e8e558c7c93671c8a51d2ba988e2755c51d26dd489fdbb8b115016b` |
| `cache/train/relevance_model_metadata.json` | `eafcabf830a7c98c39ca7a85034a8edbee885c9ff86d30cb8f2248028060454d` |
| `outputs/train_fold_diagnostics.json` | `2b5b0b3480b7c8b0f00a12e87fb0b169850164b790c62ea69e6630ddae95d5ee` |
| `outputs/relevance_model_train_metrics.json` | `c51de88b92a92463231f3c2d9bc827d7cca452ab184cc5aac3894154394cda69` |

### Primary Validation Results Artifacts
| File | SHA-256 Checksum |
| :--- | :--- |
| `outputs/relevance_model_metrics.json` | `f3a0641720a12c327031eb02e3df22955c8f89ceae3950c744ed5d6d996c7f09` |
| `outputs/stage_effects.json` | `747bcfff3807feaa0e4850a3e808407604315ba46bb99f9c3d5ec5cabd02d8d4` |
| `outputs/null_comparison.json` | `72f7cc8481a83f9a2836f3e1e1101457f0fd51fb6b6dab5de7853dea742b2841` |
| `outputs/breadth_comparison.json` | `ad008c9622ae2719468b5ae5385f2770d52dedd30ccc9d8e9f543a2dfe8fbe47` |
| `outputs/headroom.json` | `4028b45226d102d1b72106c8d63136890235f2a1aa273ac4c7e41569b27af483` |
| `outputs/run_manifest.json` | `a7748f865d4dac4f0dcda74d1e3e4242e2f928484b17a94ad8cce01d3c622acd` |
| `outputs/primary_validation_summary.md` | `fbe401ff1ecfed5f05846c80ca94424786de35d41575c0212af4d6f53727a366` |
| `outputs/tidy_impression_outcomes.csv` | `2a6582f14f36d96b5d4a732e6639478d67f30a28d3db8f8d614a4bba2426d4d0` |
| `outputs/tidy_matched_pairs.csv` | `bb20f45b09a42df73ad26ccddafb5d50cfffcab97a5e1c2c01ffc9255d02a635` |
| `outputs/tidy_user_aggregates.csv` | `618ee8b15fbd680eb793fb8153fb574962c6151986a752086a33d22cd66d6b7f` |
| `cache/validation/user_article_novelty.parquet` | `f385b64919410c596c12aafc9b10c825fbe8af5f3654a479da2f7341ae3239de` |

---

## 2. Documented Code-Correctness Fixes

Following the authorized opening of held-out validation outcomes, two non-methodological runtime bugs were encountered and resolved during pipeline initialization:
1. **NumPy Batch Popularity Mapping**: `compute_batch_click_popularity` returned a raw 1D NumPy array aligned with candidate items rather than an article ID map. This was corrected to map scores directly to article IDs.
2. **Column Alias Alignment**: Deterministic matching feature generation created `history_bin` while matching stratum creation referenced `history_length_bin`. This was aligned to `history_length_bin`.

Both fixes were purely operational runtime bug fixes. No model hyperparameters, candidate pools, eligibility criteria, or mathematical formulas were altered.

---

## 3. Corrected Train Metadata Reporting

Verified against authoritative frozen train artifacts (`cache/train/train_manifest.json`, `cache/train/tau.json`, `outputs/train_fold_diagnostics.json`, and `outputs/relevance_model_train_metrics.json`):

- **Frozen OOF Brier Score**: **0.087625** (approximately **0.0876**)
- **Train Relevance-Model Total Examples**: **1,457,625**
- **Train Relevance-Model Positives**: **151,064**
- **Train Relevance-Model Positive Rate**: **0.103637** (approximately **0.1036** / **10.36%**)
- **Frozen Relevance Floor $\tau$**: **`0.09857010114178541`** (20th percentile of OOF predictions on clicked items across blocks B2..B5)
- **Train OOF ROC-AUC**: **0.7274**
- **Train OOF nDCG Lift**: **+0.0247**

> [!IMPORTANT]
> A previous informal reporting draft accidentally conflated $\tau$ (`0.0986`) with the train positive rate (`0.1036`) and misstated Brier (`0.0882` vs authoritative `0.0876`). The authoritative frozen values listed above are exact.

---

## 4. Exact Matching Specification & Dimensions

Deterministic 1:1 nearest-neighbor matching without replacement comparing Low-breadth vs High-breadth impressions across **six exact matching dimensions**:

1. **`time_bucket_30m`**: 30-minute timestamp bucket
2. **`device_type`**: Raw device type integer code
3. **`is_subscriber`**: Subscriber status boolean
4. **`session_stage`**: Ordinal position of impression within session
5. **`session_length_bin`**: Session length category (exact: 1, 2, 3-5, 6+)
6. **`history_length_bin`**: Train-derived pre-period user history length quartiles:
   - `15-49`
   - `50-111`
   - `112-244`
   - `245+`

- **Caliper**: $|\Delta \text{slate size}| \le 1$.
- **Ordering & Tie-Breaking**:
  1. Low impressions processed in ascending order of `(impression_time, impression_id)`.
  2. Best High candidate chosen by: smallest $|\Delta \text{slate size}|$, smallest $|\Delta \text{impression time}|$, then lowest `impression_id`.
- **Sample Counts**:
  - Eligible Low Impressions: 25,731
  - Eligible High Impressions: 87,158
  - Matched Pairs Formed: **12,243**
  - Low Match Rate: **47.6%**
  - High Match Rate: **14.0%**

---

## 5. High-Side User Reuse & Focal-User Bootstrap Clarification

Characterization of user repetition in `outputs/tidy_matched_pairs.csv`:

- **Unique Low Users**: **2,401**
- **Unique High Users**: **3,821**
- **Pairs per Low User**:
  - Mean: **5.10** | Median: **3.0** | p95: **18.0** | Max: **80**
- **Pairs per High User**:
  - Mean: **3.20** | Median: **2.0** | p95: **9.0** | Max: **23**
- **High Users Appearing in >1 Pair**: **2,599** (**68.0%**)
- **High Users Appearing in >5 Pairs**: **594** (**15.5%**)
- **Maximum Reuse of a Single High User**: **23** pairs
- **High Users Paired with Multiple Distinct Low Users**: **2,509** (**65.7%**; max distinct Low users for one High user = **21**)

### Primary Inference & Methodological Limitation
- **Primary Estimator**: Matched pair differences are aggregated by focal `low_user_id` and bootstrapped across 2,000 replicates. Low users are the focal treatment units.
- **Methodological Limitation**: Repeated High controls can induce dependence across Low-user clusters that this one-way bootstrap does not explicitly model.
- **Diagnostic Sensitivity Flag**: `PLANNED_POST_ROBUSTNESS_TWO_WAY_CLUSTERING_SENSITIVITY`. A diagnostic evaluation (two-way multi-way clustering or a bipartite connected-component bootstrap) is scheduled for execution *after* the prespecified robustness battery and will be explicitly reported as a diagnostic sensitivity, preserving the primary frozen inference.

---

## 6. 24-Hour Publication Age Limitation & Mandatory Wording

- **Observed Exposed Slates**:
  - Median publication age: **4.01 hours**
  - Mean publication age: **1,565.15 hours** (~65.2 days)
  - Articles published $> 24$ hours prior to impression: **26.28%**
- **Null Slates (24h Observable Supply)**:
  - Median publication age: **11.23 hours**
  - Mean publication age: **11.68 hours**
  - Articles published $> 24$ hours prior to impression: **0.00%** (strictly bounded in $[0, 24]$h)

> [!WARNING]
> **Mandatory Wording Rule**: All primary Available $\to$ Exposed findings must be phrased:  
> *“relative to the 24-hour observable published supply”* or *“relative to recently published observable supply”*.  
> It is strictly forbidden to claim *“relative to everything available to the publisher”* or to imply knowledge of the publisher's proprietary internal candidate pool.

---

## 7. Consumed Set Verification

- Evaluated across **149,222** eligible clicked items (148,211 impressions, 11,885 users):
  - Minimum $\hat{r}$: `0.002549`
  - Median $\hat{r}$: `0.151199`
  - Mean $\hat{r}$: `0.160914`
  - Items with $\hat{r} \ge \tau$: **122,969** (**82.41%**)
  - Items with $\hat{r} < \tau$: **26,253** (**17.59%**)
- **No Relevance Filtering on Consumed Set**: Consumed items were defined strictly by user click actions under the subscriber premium access rule. Below-$\tau$ items were retained in `consumed_ids` and assigned indicator $\mathbb{I}(\hat{r} \ge \tau) = 0$ in discovery calculations.
- **Coverage Values**:
  - Impression-weighted consumed coverage: **0.8251** (non-empty) / **0.7057** (all)
  - User macro consumed coverage: **0.6858**
  - True Choice Gap in Coverage: **+0.2424** (95% CI: `[+0.2386, +0.2462]`)
  - The appearance of `1.0000` in the initial report summary was an un-rendered display string typo, not an underlying calculation error.

---

## 8. Headroom Invariant & Optimization Verification

- **Constraints Verified on All 156,580 Impressions**:
  1. Swaps $\le 2$: 0 violations (0 swaps: 2; 1 swap: 9; 2 swaps: 156,569).
  2. Observed supply items only: 0 violations.
  3. Common support $\ge 3$ distinct users: 0 violations.
  4. Relevance $\ge \tau$: 0 violations.
  5. Relevance retention $\ge 0.97$: 0 violations (min: `0.970000`, median: `1.1765`, mean: `1.2481`).
- **Gain Exceeding $2/S$**:
  - One impression had gain `0.400030` on $S=5$.
  - Proven: Contrastive semantic novelty $1 - \cos$ can exceed 1.0 because minimum cosine similarity is **-0.0748** (max novelty = **1.0748**).
  - Generalized bound $\text{gain} \le \text{swaps} \times \max(d) / S$ was evaluated across all 156,580 impressions with **0 violations**.
- **Brute-Force Optimizer Verification**:
  - Exhaustive combinatorial search on 100 randomly sampled real impressions matched the production optimizer with maximum absolute discrepancy of **$5.55e-17$**.

---

## 9. Estimand Reconciliation

| Stage / Metric | (A) Impression Observed | (B) Impression Comparison | (C) Impression Difference ($A - B$) | (D) User Observed | (E) User Comparison | (F) User Macro Difference ($D - E$) | Bootstrap 95% CI | Standardized Effect |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Available $\to$ Exposed Discovery** | 0.156349 | 0.306441 | **-0.150093** | 0.164633 | 0.332942 | **-0.168309** | [-0.169524, -0.167098] | **-2.51 SD** |
| **Available $\to$ Exposed Coverage** | 0.473600 | 0.861599 | **-0.387999** | 0.440427 | 0.816498 | **-0.376071** | [-0.378521, -0.373516] | **-2.73 SD** |
| **Available $\to$ Exposed Novelty** | 0.337242 | 0.361253 | **-0.024011** | 0.383848 | 0.415671 | **-0.031823** | [-0.032585, -0.031086] | **-0.72 SD** |
| **Exposed $\to$ Consumed Discovery** | 0.198473 | 0.267771 | **+0.069298** | 0.200828 | 0.286340 | **+0.085513** | [+0.083569, +0.087498] | **+0.81 SD** |
| **Exposed $\to$ Consumed Coverage** | 0.476885 | 0.705697 | **+0.228812** | 0.443361 | 0.685764 | **+0.242403** | [+0.238647, +0.246197] | **+1.16 SD** |
| **Exposed $\to$ Consumed Novelty** | 0.336977 | 0.275831 | **-0.009701** | 0.383152 | 0.299800 | **-0.009414** | [-0.010416, -0.008436] | **-0.16 SD** |
| **Headroom Discovery Gain** | 0.139279 | 0.000000 | **+0.139279** | 0.152011 | 0.000000 | **+0.152011** | [+0.151249, +0.152774] | **+3.64 SD** |

*(Note: For Exposed $\to$ Consumed, the effect direction is Consumed minus Choosable Exposed ($B - A$ and $E - D$), per protocol).* 

---

## 10. Robustness Battery Status Confirmation

**ZERO prespecified robustness variants have yet been run.**  
The primary state is completely audited, reconciled, and frozen.
