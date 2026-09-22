# Pre-Run QA & Final Integrity Checkpoint

**Date**: 2026-09-21  
**Readiness Status**: **READY FOR ONE-TIME HELD-OUT VALIDATION RUN**  
**Holdout Status**: **Held-out validation outcomes remain unopened.**

---

## 1. Provenance Commit Hashes & Protocol Integrity

| Level | Commit Hash / SHA-256 | Description |
| :--- | :--- | :--- |
| **Protocol Commit** | `cb14d50736d1b8c8202b1aa67158e15c3eabb804` | Parameter freeze baseline |
| **Pre-Outcome Gate Commit** | `db8d1003358f50d51e66672ca3a6a8e8f02d1c0a` | Readiness audit gate |
| **Implementation Lock Commit** | `083e93104483658b6418900770de7d357665c8e8` | Frozen plan & decisions |
| **Train Implementation Commit** | `2e1383454c51e86cd7bb42a144d53a9bc1fd0e88` | Train pipeline & foundation |
| **Train Provenance Commit** | `d757597d48f415ca3f49d98c4501259b9e335135` | Train manifest provenance record |
| **Validation Implementation Commit** | `38c7e886c71743c17649591142bac31573a1903b` | Validation machinery (HEAD) |
| **analysis_protocol_v1.yaml SHA-256** | `df7ca427f39aee143a9a3aeb8a2ca4d59356dac43a25843995915d0ecac933b1` | Verified immutable |
| **analysis_protocol_notes.md SHA-256** | `9ae1ab3b7a593a45665ad5abe31433f0e9331885cdbd493724aafb8a9a16edf6` | Verified immutable |
| **IMPLEMENTATION_DECISIONS.md SHA-256**| `e699155d673d094969cfd66739521c7e3e8135f5882bc5e4893874454fc1d6af` | Verified immutable |
| **implementation_plan.md SHA-256** | `b0428e0bbb2f72020e904e8a2f2dc316477d8f594e542af4d03d5f8ee7985e6f` | Verified immutable |

---

## 2. Frozen Parameter Integrity

All experimental parameters remain locked to their frozen values:
- **Seed**: `20260919`
- **Relevance Floor $\tau$**: `0.09857010114178541` (derived strictly from train OOF clicks in $B_2 \dots B_5$)
- **Observable Supply**: Primary retrospective window = 24h ($t_{\text{imp}} - 24\text{h} \le t_{\text{pub}} < t_{\text{imp}}$); sensitivities = 12h, 48h.
- **Counterfactual Supply Eligibility**: Observable supply $\ge 3\times$ slate size; slate size $5 \le |\mathcal{S}| \le 26$; history length $\ge 15$.
- **User Breadth Shannon Entropy Cutoffs**: Low $\le 1.4638116962471364$; Medium $1.4638116962471364 < H(u) \le 1.5905145971043473$; High $> 1.5905145971043473$.
- **User History Length Bins**: $[15, 49]$, $[50, 111]$, $[112, 244]$, $[245, \infty)$.
- **Semantic Novelty**: Primary history $k=5$; robustness $k \in \{3, 10\}$.
- **Common Support**: $\ge 3$ distinct users per 30-minute bucket (sensitivity $\ge 5$).
- **Null Slates**: 200 popularity+recency weighted draws and 200 uniform random draws per impression.
- **Discovery Headroom**: Max 2 swaps; primary relevance retention $\ge 97\%$ (sensitivities 99%, 95%). Gated by model qualification.
- **Bootstrap**: User-clustered; 2,000 replicates; percentile 95% CIs; between-user SD standardization.

---

## 3. History File Boundaries & Profile Source Verification

Raw verification of history parquet files directly:
- **Train History (`ebnerd_small/train/history.parquet`)**:
  - Row count: 15,143 users | Total clicks: 2,426,247
  - Span: `2023-04-27 07:00:00+00:00` $\to$ `2023-05-18 06:59:59+00:00` (21.0 days)
  - Precedes train behaviors start (`2023-05-18 07:00:01+00:00`) by 2 seconds.
- **Validation History (`ebnerd_small/validation/history.parquet`)**:
  - Row count: 15,342 users | Total clicks: 2,204,173
  - Span: `2023-05-04 07:00:00+00:00` $\to$ `2023-05-25 06:59:59+00:00` (21.0 days)
  - Precedes validation behaviors start (`2023-05-25 07:00:02+00:00`) by 3 seconds.

### Source Code Guard
- In [src/validation.py](file:///Users/davvead/Downloads/ebnerd_gate_package/src/validation.py), `validate_history_source()` strictly verifies:
  1. Source path resolves to `ebnerd_small/validation/history.parquet` (explicitly rejects `train/history.parquet`).
  2. Latest timestamp strictly precedes `2023-05-25 07:00:00+00:00`.
  3. Span is approximately 21 days ($20.5 \le \text{span} \le 21.5$ days).
- [scripts/03_run_validation_analysis.py](file:///Users/davvead/Downloads/ebnerd_gate_package/scripts/03_run_validation_analysis.py) calls `validate_history_source(VALIDATION_HISTORY_PATH)` as step `[0/8]` prior to loading behaviors.
- Unit test `test_validation_profile_source_guard` in `tests/test_validation_pipeline.py` passes and confirms that substituting train history raises `ValueError`.

---

## 4. Held-Out Boundary & Authorization Guard Status

- **Guard Enforcement**: `scripts/03_run_validation_analysis.py` contains an active security gate. Invoking without `--authorize-heldout-run` immediately aborts with exit code 1.
- **Data Boundary Distinction**:
  - `ebnerd_small/validation/history.parquet`: Allowed 21-day pre-period history (`2023-05-04` to `2023-05-25`) used strictly to build baseline user interest profiles using frozen train cutoffs.
  - `ebnerd_small/validation/behaviors.parquet`: Held-out target-period behavior outcomes. Has NOT been opened, read, or scored.
- **Outcome Artifact Scan**:
  - `cache/validation/`: Clean/empty directory (no outcome caches exist).
  - `outputs/`: No validation outcome JSONs exist (`run_manifest.json`, `stage_effects.json`, `null_comparison.json`, `breadth_comparison.json`, `headroom.json` are absent).

---

## 5. Production Model Immutability & Leakage Prevention

- **Frozen Pipeline**: Model weights (`cache/train/relevance_model.joblib`) and relevance floor $\tau$ are loaded read-only without refitting or rescaling.
- **Categorical Safety**: `OneHotEncoder(handle_unknown="ignore")` silently handles any unseen device categories as all-zeros.
- **Time-Safe Popularity**: For any impression at time $t$, clicks must satisfy $t - 24\text{h} \le t_{\text{click}} < t$. Same-time clicks ($t_{\text{click}} = t$) and future clicks ($t_{\text{click}} > t$) are strictly excluded.
- **Publication Boundary**: Articles published at or after impression time ($t_{\text{pub}} \ge t$) are strictly excluded from observable supply.

---

## 6. Null Sampler Implementation

- **Algorithm**: Vectorized Gumbel-top-$k$ exponential key sampling (the continuous formulation of Efraimidis & Spirakis 2006 for weighted sampling without replacement).
- **Sampling Mechanism**:
  $$w_i = (1 + \text{clicks\_24h}(i, t)) \cdot 2^{-\text{age\_hours}(i, t)/12.0}$$
  Draw independent standard exponential variates $E_i \sim \text{Exp}(1.0)$. Keys are formed as:
  $$s_i = \log(\max(w_i, 10^{-12})) - \log(\max(E_i, 10^{-12}))$$
  Top $K$ candidates are extracted without replacement via `np.argpartition(-scores, K - 1)`.
- **RNG Seeding**: Deterministic per-impression seed `SEED + impression_id`.

---

## 7. Verification Status

- **Unit Test Suite**: **38 / 38 unit tests passing** across 12 test modules (runtime: 2.85s).
- **Synthetic End-to-End Dry Run**: Fully exercised (AVAILABLE $\to$ EXPOSED $\to$ CONSUMED $\to$ nulls $\to$ matching $\to$ qualification $\to$ Headroom $\to$ bootstrap) in 0.18s with zero exposure to real validation behavior data.
- **Source Working Tree**: Clean, verified, and ready for authorized execution.

---

**Held-out validation outcomes remain unopened.**
