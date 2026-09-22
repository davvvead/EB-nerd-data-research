# Phase 1 Primary Held-Out Validation Analysis Report

**Freeze Date**: 2026-09-22  
**Seed**: `20260919` | **Protocol Version**: `phase1-parameter-freeze-v1`  
**Git HEAD**: `0e6ba855de6b9ddd6b98556fdc3fee39996a9041`  

---

## 1. Validation Sample Accounting

| Step / Filter Stage | Impression Count | Distinct Users | Retention |
| :--- | :---: | :---: | :---: |
| **Total Validation Behavior Rows** | 244,647 | — | 100.0% |
| **Front-Page Impressions** (`article_id` is null) | 173,345 | — | 70.9% |
| **Slate Size Filter** ($S \in [5, 26]$) | 161,750 | — | 93.3% |
| **Pre-Period History Filter** ($H \ge 15$) | 156,580 | 13,564 | 96.8% |
| **Counterfactual Supply Filter** ($\text{Supply} \ge 3S$) | **156,580** | **11,967** | **100.0%** |
| **Choice Stage Eligible** (has choosable click) | 148,211 | 11,885 | 94.7% |

- **Breadth Cohort Breakdown**: Low: 25,731 | Medium: 43,691 | High: 87,158

---

## 2. Held-Out Relevance Model Evaluation & Qualification Gate

- **Validation ROC-AUC**: **0.7317** (Threshold: $\ge 0.60$ — PASSED)
- **Validation Model nDCG**: **0.6526**
- **Popularity+Recency Baseline nDCG**: **0.6352**
- **nDCG Lift over Baseline**: **+0.0175** (Threshold: $\ge +0.01$ — PASSED)
- **Validation Brier Score**: **0.0852**
- **Frozen Relevance Floor $\tau$**: **`0.09857010114178541`**
- **Qualification Decision**: **[QUALIFIED]**

> [!NOTE]
> Headroom qualification status is **QUALIFIED**. Headroom optimization was executed.

---

## 3. Stage 1: Available $\to$ Exposed Primary Analysis (Null Slates)

Comparing observed publisher slates against 200 popularity+recency weighted draws and uniform random draws relative to the 24-hour observable published supply:

### Primary Inferential Estimand: User-Level Macro Means (Bootstrap Unit: User)

| Metric | User Observed Mean (D) | User Null Mean (E) | User Mean Gap (F = D - E) | 95% Bootstrap CI | Standardized Effect |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Discovery $D(u, \mathcal{S})$** | 0.1646 | 0.3329 | **-0.1683** | [-0.1695, -0.1671] | **-2.51 SD** |
| **Relevance Coverage** | 0.4404 | 0.8165 | **-0.3761** | [-0.3785, -0.3735] | **-2.73 SD** |
| **Conditional Novelty** | 0.3838 | 0.4157 | **-0.0318** | [-0.0326, -0.0311] | **-0.72 SD** |

### Descriptive Impression-Weighted Means ($N = 156,580$)

| Metric | Impression Observed (A) | Impression Null (B) | Impression Mean Diff (C = A - B) |
| :--- | :---: | :---: | :---: |
| **Discovery $D(u, \mathcal{S})$** | 0.1563 | 0.3064 | **-0.1501** |
| **Relevance Coverage** | 0.4736 | 0.8616 | **-0.3880** |
| **Conditional Novelty** | 0.3372 | 0.3613 | **-0.0240** |

- **Uniform Random Null Discovery**: User Macro: 0.2520 (Gap: -0.0874) | Impression-Weighted: 0.2296 (Gap: -0.0732)

---

## 4. Stage 2: Exposed $\to$ Consumed Primary Analysis (Choice Stage)

Applying the subscriber/non-subscriber premium choice rule (148,211 eligible impressions across 11,885 users):

### Primary Inferential Estimand: User-Level Macro Means (Bootstrap Unit: User)

| Metric | User Choosable Exposed (D) | User Consumed (E) | User Choice Gap (F = E - D) | 95% Bootstrap CI | Standardized Effect |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Discovery $D(u, \mathcal{S})$** | 0.2008 | 0.2863 | **+0.0855** | [+0.0836, +0.0875] | **+0.81 SD** |
| **Relevance Coverage** | 0.4434 | 0.6858 | **+0.2424** | [+0.2386, +0.2462] | **+1.16 SD** |
| **Conditional Novelty** | 0.3832 | 0.2998 | **-0.0094** | [-0.0104, -0.0084] | **-0.16 SD** |

### Descriptive Impression-Weighted Means ($N = 148,211$)

| Metric | Choosable Exposed (A) | Consumed (B) | Impression Mean Diff (C = B - A) |
| :--- | :---: | :---: | :---: |
| **Discovery $D(u, \mathcal{S})$** | 0.1985 | 0.2678 | **+0.0693** |
| **Relevance Coverage** | 0.4769 | 0.7057 | **+0.2288** |
| **Conditional Novelty** | 0.3370 | 0.2758 | **-0.0097** |

---

## 5. Matched Low-vs-High Breadth Comparison

Deterministic 1:1 nearest-neighbor matching without replacement across 6 exact matching dimensions (30m time bucket, raw device type, subscriber status, session stage, session length bin, train-derived history length quartiles) within $|\Delta \text{slate}| \le 1$ caliper:
- **Eligible Low Impressions**: 25,731
- **Eligible High Impressions**: 87,158
- **Matched Pairs**: **12,243** (Low Match Rate: 47.6%, High Match Rate: 14.0%)
- **Effect Direction**: **High breadth minus Low breadth**

| Metric | Mean Matched Difference (High - Low) | 95% Bootstrap CI | Standardized Effect | Practical Significance ($|d| \ge 0.10$) |
| :--- | :---: | :---: | :---: | :---: |
| **Discovery Difference** | **+0.0068** | [+0.0031, +0.0107] | **+0.07 SD** | **False** |
| **Coverage Difference** | **+0.0071** | [-0.0010, +0.0158] | **+0.03 SD** | **False** |
| **Novelty Difference** | **+0.0084** | [+0.0048, +0.0124] | **+0.09 SD** | **False** |

---

## 6. Discovery Headroom Analysis

- **Headroom Qualification Status**: **QUALIFIED**

- **Specification**: 24h observable supply $\cap$ common support $\ge 3$ distinct users in same 30-min window $\cap$ relevance $\ge \tau$.
- **Constraints**: Maximum 2 swaps, retaining at least 97% of relevance by our model's own estimate.
- **Eligible Impressions**: 156,580
- **Share of Impressions with Positive Headroom**: **100.0%**
- **Mean Headroom Discovery Gain**: **+0.1393** (95% CI: [+0.1512, +0.1528])
- **Median Headroom Discovery Gain**: **+0.1330**
- **Standardized Effect**: **+3.64 SD** (Practical Significance: **True**)
- **Average Swaps Utilized**: **2.00** / 2.0
- **Average Modeled Relevance Retention**: **124.8%** (retaining at least 97% of relevance by our model's own estimate)

---
*Run completed in 462.8s. Results frozen.*
