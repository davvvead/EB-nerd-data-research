# Locked Analysis Contract

Status: methodology locked before outcome analysis.

This file separates decisions that are fixed now from numeric parameters that may only be fixed after the feasibility gates and before any outcome metric is computed.

## Research question

When potentially relevant information is observably available, do comparable users receive comparable opportunities to discover it, and at what stage is that opportunity lost?

Conceptual pipeline:

AVAILABLE -> EXPOSED -> CONSUMED

The analysis does not claim to observe the publisher's internal candidate generator, ranking position, or whether a front-page slate was editorial, personalized, or hybrid.

## Locked scope

- Primary dataset: EB-NeRD Small, subject to Gate D confirmation of downloaded files and artifact coverage.
- Primary surface: front-page impressions only.
- Exposure: membership in `article_ids_inview`.
- No slot-level claims because in-view lists are shuffled.
- Observable supply: articles published before an impression and inside a freshness window.
- Primary freshness window: 24 hours after the parameter-freeze commit confirms that this window is non-degenerate.
- Robustness windows: 12 and 48 hours.
- Primary null: same-size popularity-plus-recency slates.
- Random same-size slates: reference baseline only.
- User-clustered uncertainty.
- Paywall-aware consumption analysis stratified by subscription status.
- `total_inviews`, `total_pageviews`, and other forward-looking aggregate article fields are excluded from time-valid popularity construction.

## 1. Held-out relevance scoring

All relevance scores used in outcome metrics and Discovery Headroom must be out-of-sample.

Primary rule:

1. Fit the relevance model on earlier impressions.
2. Tune or calibrate only on a subsequent validation period.
3. Compute exposure outcomes and Discovery Headroom only on later held-out impressions.
4. No analyzed impression may contribute its own label to model fitting or calibration.

If Gate D shows that a single chronological split would leave insufficient data, use temporal cross-fitting. Each impression must still be scored by a model that was not trained on that impression. The exact date cutoffs are frozen in the parameter commit before outcome metrics are run.

Limitation to report: the relevance model is trained on exposed items. Even with common-support restrictions, recency and historical popularity may absorb unobserved publisher prominence because true position is unavailable.

## 2. Popularity-plus-recency null

Popularity must be computed only from behavior occurring before the target impression.

Popularity signal:

- Primary source: prior clicks.
- Optional robustness signal: prior read-time totals computed only from past events.
- Never use in-view counts as the primary popularity signal because those already contain publisher exposure decisions.

The exact combination of popularity and recency, whether weighted sampling or top-k selection, the lookback length, smoothing, and treatment of zero-count new articles are frozen in the parameter commit.

Brand-new zero-count articles must retain nonzero eligibility through the recency component.

Each counterfactual slate has the same size as the observed slate.

## 3. One discovery definition

For user u and article i:

    d(u,i) = novelty(u,i), if predicted_relevance(u,i) >= tau_u
             0, otherwise

For slate S:

    D(u,S) = mean_i_in_S d(u,i)

Novelty is defined from pre-period user history, not target-period exposure.

Primary user-history similarity uses nearest-history semantic similarity rather than only a centroid. The neighbor count is frozen before outcome analysis.

### Discovery decomposition

A low D can arise for two different reasons. Every main exposure result must therefore be decomposed into:

1. Relevance coverage: share of slate items with predicted relevance >= tau_u.
2. Conditional novelty: mean novelty among items above tau_u.

This prevents "narrow familiar exposure" from being confused with "poorly matched exposure."

## 4. Discovery Headroom

Headroom asks whether one or two substitutions could increase discovery while retaining relevance by the model's own estimate.

Constraints:

- Same slate size.
- Maximum swaps: one or two, with the primary value frozen in the parameter commit.
- Every substituted item must satisfy the per-item relevance floor tau_u.
- Primary substitution pool: common support only.
- Broader observable-supply substitution pool: upper-bound sensitivity analysis.
- Primary aggregate relevance tolerance: 3%.
- Robustness tolerances: 1% and 5%.

Locked wording:

"The alternative retained X% of relevance by our model's own estimate."

Never claim equivalent preservation of user value.

## 5. Common support

The primary headroom pool contains supply items actually observed in-view for at least n comparable users in the same defined context window.

The exact n and context definition are frozen after gate outputs and before outcome analysis.

Both common-support headroom and broad-pool headroom are reported if headroom is used in the final story.

## 6. User comparisons

Primary comparison avoids unsupervised clustering.

Primary user grouping:

- pre-period interest breadth or entropy tertiles
- optionally dominant publisher category as a descriptive secondary grouping

Users are matched or weighted on pre-specified context variables, including at minimum:

- history length
- history breadth
- prior activity
- scroll depth
- session stage or length
- device
- subscription status
- time context
- observed slate size

Exact bins and calipers are frozen in the parameter commit.

Clustering is secondary only. If used, its algorithm, features, scaling, random seed, and k must be frozen before outcome metrics are computed.

## 7. Choice-stage analysis and paywalls

For non-subscribers, premium items are excluded from the freely choosable set used to interpret exposed-to-consumed transitions.

Subscribers and non-subscribers are analyzed separately for the choice stage.

Read-time and scroll depth among clicked items are only a sanity check. Conditioning on a click creates selection, so these outcomes are not evidence that users generally value novel content.

Allowed wording:

"Among articles that were clicked, novel-but-model-relevant items did or did not show a substantial downstream engagement penalty."

## 8. Primary stage results

The main output is a stage-by-stage effect-size profile, not a forced single category.

Possible interpretations include:

- supply-constrained environment
- exposure-stage narrowing
- popularity/recency-explained narrowing
- choice-stage narrowing
- multi-stage narrowing
- no material effect
- mixed or inconclusive

Popularity-plus-recency is the primary null. Random is a reference.

An observed slate below random but approximately matching popularity-plus-recency means the narrowing is explainable by popularity and recency under this design.

## 9. Practical effect thresholds

The project uses delta = 0.10 standardized SD as the working practical-equivalence threshold, but each outcome family gets its own scale.

The SD for each family is computed from between-user variation in the matched primary analysis sample after aggregating impression-level measures to the user level.

Exposure gaps, headroom, and engagement sanity checks do not share one common SD.

The exact aggregation and SD formulas are frozen in the parameter commit.

## 10. Bootstrap and compute

- Generate fixed-seed null slates once.
- Compute per-impression stage metrics once.
- Run the constrained headroom search once for eligible impressions.
- Aggregate to users.
- Bootstrap users, not impressions or optimizer runs.
- Use user-clustered 95% bootstrap intervals.
- Record seeds, sample IDs, environment versions, and runtime.

## 11. Two-commit precommitment process

### Commit 1: Gates and parameter freeze

After Gates A-C and Gate D artifact inspection, freeze all remaining numeric values in `precommit_parameters.yaml`.

This commit must occur before any final outcome metric is computed.

Push it to a remote repository so the timestamp is external.

### Commit 2: Outcomes

Only after Commit 1 may the pipeline compute:

- stage effect sizes
- matched user comparisons
- Discovery Headroom
- outcome confidence intervals

Methodological changes after Commit 1 must be documented as deviations, not silently edited.

## 12. Parameters that must be frozen in Commit 1

- chronological training, validation, and held-out analysis date ranges
- tau_u rule
- common-support n
- matching bins and calipers
- eligible slate-size range or fixed-k rule
- k in history-neighbor similarity
- minimum history length
- user sample size
- sample seed
- popularity lookback
- popularity-plus-recency formula
- zero-count article rule
- number of baseline draws per impression
- primary maximum swap count
- SD and aggregation definitions for each delta family
- cluster method and k only if secondary clustering is retained

## 13. Timebox and fallback

Target schedule:

- Sept 23: feasibility gates complete
- Sept 26: parameter-freeze commit pushed remotely
- Oct 10: primary results complete
- Oct 10-16: visual design and writing
- Oct 17-18: buffer, QA, and submission

Phase 1 fallback if Discovery Headroom is late, weak, unstable, or unsupported:

1. stage decomposition
2. popularity-plus-recency and random null comparisons
3. matched-user exposure comparison or shared-window result
4. discovery decomposition into relevance coverage and conditional novelty
5. methods, limitations, and CRTC connection

Headroom is a stretch result, not a dependency for a complete Phase 1 submission.

One team member must own visual communication and PDF assembly separately from the analysis implementation.
