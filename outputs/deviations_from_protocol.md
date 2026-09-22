# Protocol Deviations and Technical Adjustments Log

**Status**: TRAIN-SIDE FOUNDATION AUDIT  
**Date**: 2026-09-21  
**Protocol Commit**: `cb14d50736d1b8c8202b1aa67158e15c3eabb804`  
**Implementation Lock Commit**: `083e93104483658b6418900770de7d357665c8e8`  

---

## 1. Scientific Protocol Deviations

**No protocol deviations occurred during train-side implementation.**

All substantive specifications defined in `analysis_protocol_v1.yaml` and `analysis_protocol_notes.md` were followed verbatim:
- Minimum user history length = 15.
- Eligible front-page slate size = 5 to 26.
- Relevance-model training eligibility: front-page impression (`article_id` is null), observed slate size 5 to 26 inclusive, user history length $\ge 15$. (The observable supply ratio requirement $\ge 3\times$ slate size is strictly a counterfactual-analysis eligibility condition and was NOT applied to relevance training).
- Category breadth Shannon entropy tertile cutpoints from eligible train users:
  - Low breadth: $\le 1.4638116962471364$
  - Medium breadth: $1.4638116962471364 < H(u) \le 1.5905145971043473$
  - High breadth: $> 1.5905145971043473$
- User history length quartile bins: $[15, 49]$, $[50, 111]$, $[112, 244]$, $[245, \infty)$.
- Primary semantic novelty: $1 - \text{mean}(\text{top-5 cosine similarities})$.
- Time-safe popularity formula: $(1 + \text{clicks\_24h}) \cdot 2^{-\text{age\_hours}/12}$ with zero-click pseudocount 1.
- No future aggregate fields (`total_pageviews`, `total_inviews`, `total_read_time`).
- Relevance floor $\tau$: 20th percentile of out-of-fold predicted relevance among clicked training examples.
- Seed: 20260919.

---

## 2. Code-Correctness Fixes and Operational Clarifications

The following technical implementation adjustments were resolved to ensure exact and watertight implementation of the frozen specification:

1. **Forward-Chaining Timestamp-Tie Handling**:
   - *Issue*: Contiguous chronological segmentation into 5 blocks could potentially divide identical timestamps across block boundaries, violating strict temporal precedence.
   - *Fix*: Implemented boundary tie-breaking where all impressions sharing an identical timestamp at a boundary are assigned to the later block. For all scored blocks $B_2 \dots B_5$, $\max(\text{training } t) < \min(\text{scored } t)$ holds strictly.
   - *Classification*: Code-correctness fix.

2. **Fold-Local Preprocessing Isolation**:
   - *Issue*: Fitting `StandardScaler` or `OneHotEncoder` globally before forward-chaining would leak future feature distributions into earlier evaluation blocks.
   - *Fix*: Fitted preprocessing pipelines fold-locally strictly inside each expanding historical training window ($B_1 \to B_2$, $B_1 \cup B_2 \to B_3$, etc.). The production preprocessor was subsequently fitted on all eligible train examples.
   - *Classification*: Code-correctness fix.

3. **Deterministic 1:1 Matched Comparison Ordering**:
   - *Issue*: Greedy matching within strata could depend on arbitrary DataFrame row ordering.
   - *Fix*: Frozen deterministic processing order: Low-breadth impressions sorted by `(impression_time, impression_id)`, matched to High-breadth candidates by `(abs(slate_size_diff), abs(time_diff), impression_id)`.
   - *Classification*: Pre-implementation operational freeze (recorded in `IMPLEMENTATION_DECISIONS.md`).

4. **Array Topic Parsing Handling**:
   - *Issue*: `pd.notna()` evaluation on NumPy array topic fields raised an ambiguous truth value exception during data loading.
   - *Fix*: Replaced conditional check with robust `norm_list()` handling that converts arrays and serialized lists safely.
   - *Classification*: Bug fix.
