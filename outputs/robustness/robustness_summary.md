# EB-NeRD Phase 1 Prespecified Robustness Battery Audit Report

**Status**: `COMPLETE — RESULTS FROZEN`  
**Protocol Version**: `phase1-parameter-freeze-v1`  
**Primary Freeze Commit**: `598b3553346b9be36195b074530827448fdc8e4b`  
**Timestamp UTC**: `2026-09-22T05:10:55.303013+00:00`  

---
## 1. Available -> Exposed Discovery Comparison

| Specification | Eligible Imp / Users | User-Macro Gap [95% CI] | Std Effect | Relevance Cov Gap | Cond Novelty Gap | Qualitative Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary 24h Contrastive k5** | 156,580 / 11,967 | -0.1683 [-0.1695, -0.1671] | -2.51 SD | -0.3761 | -0.0318 | Baseline |
| **Supply 12H** | 142,368 / 11,820 | -0.1720 [-0.1733, -0.1707] | -2.48 SD | -0.3866 | -0.0319 | SAMPLE-RESTRICTED; SAME DIRECTION (-0.1720, -2.48 SD) |
| **Supply 48H** | 156,580 / 11,967 | -0.1585 [-0.1597, -0.1573] | -2.42 SD | -0.3542 | -0.0309 | SAME SAMPLE; SAME DIRECTION; MODESTLY ATTENUATED |
| **Bert** | 156,580 / 11,967 | -0.0081 [-0.0082, -0.0079] | -1.08 SD | -0.3761 | -0.0002 | SAME DIRECTION, MATERIALLY ATTENUATED UNDER ALTERNATIVE SEMANTIC SCALE |
| **Novelty K3** | 156,580 / 11,967 | -0.1529 [-0.1540, -0.1518] | -2.51 SD | -0.3761 | -0.0313 | same direction and similar magnitude |
| **Novelty K10** | 156,580 / 11,967 | -0.1927 [-0.1941, -0.1914] | -2.51 SD | -0.3761 | -0.0301 | same direction and similar magnitude |

*(Note: For BERT, raw novelty magnitudes are not directly comparable across embedding representations because representation geometry changes the novelty scale; standardized effect remains -1.08 SD. For Supply 48h, sample is identical to Primary: 156,580 impressions / 11,967 users; 24h gap = -0.1683, 48h gap = -0.1585, absolute change = +0.0098, ~5.8% magnitude reduction; 26.28% of exposed items were published >24h prior. Supply 12h is sample-restricted to 142,368 impressions / 11,820 users.)*

---
## 2. Exposed -> Consumed Choice Stage Comparison

| Specification | Discovery Gap [95% CI] | Std Effect | Cond Novelty Gap | Qualitative Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Primary 24h Contrastive k5** | +0.0855 [+0.0836, +0.0875] | +0.81 SD | -0.0094 | Baseline |
| **Bert** | +0.0049 [+0.0045, +0.0053] | +0.22 SD | -0.0001 | SAME DIRECTION, MATERIALLY ATTENUATED |
| **Novelty K3** | +0.0766 [+0.0748, +0.0784] | +0.78 SD | -0.0085 | same direction and similar magnitude |
| **Novelty K10** | +0.1018 [+0.0996, +0.1040] | +0.85 SD | -0.0100 | same direction and similar magnitude |

*(Note: Across all specifications, the Exposed $\to$ Consumed discovery gap is positive, driven primarily by higher relevance coverage among chosen items, while the conditional novelty gap is slightly negative: primary -0.0094, BERT -0.0001, k=3 -0.0085, k=10 -0.0100. Thus, discovery increases at the choice stage because consumed items have much higher relevance, not because users actively prefer higher novelty conditional on exposure.)*

---
## 3. Matched Breadth Discovery Difference (High minus Low)

| Specification | High minus Low Diff [95% CI] | Std Effect | Qualitative Classification |
| :--- | :--- | :--- | :--- |
| **Primary 24h Contrastive k5** | +0.0068 [+0.0031, +0.0107] | +0.07 SD | Baseline |
| **Bert** | +0.0006 [+0.0002, +0.0010] | +0.06 SD | same direction, similar standardized magnitude (+0.06 SD vs +0.07 SD, practically small) |
| **Novelty K3** | +0.0058 [+0.0025, +0.0093] | +0.07 SD | same direction and similar magnitude |
| **Novelty K10** | +0.0077 [+0.0035, +0.0122] | +0.07 SD | same direction and similar magnitude |

---
## 4. Discovery Headroom Optimization

| Specification | Eligible N | Pos HR % | User-Macro Mean Gain [95% CI] | Median Gain | Std Effect | Mean Swaps | Mean Ret % | Qualitative Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary 24h Contrastive k5 (CS>=3, ret>=97%)** | 156,580 | 100.0% | +0.1520 [+0.1512, +0.1528] | +0.1330 | 3.64 SD | 2.00 | 124.8% | Baseline |
| **Supply 12H** | 142,368 | 100.0% | +0.1531 [+0.1523, +0.1538] | +0.1335 | 3.80 SD | 2.00 | 126.6% | same direction and similar magnitude |
| **Supply 48H** | 156,580 | 100.0% | +0.1531 [+0.1523, +0.1539] | +0.1344 | 3.66 SD | 2.00 | 124.6% | same direction and similar magnitude |
| **Bert** | 156,580 | 100.0% | +0.0261 [+0.0257, +0.0265] | +0.0151 | 1.16 SD | 2.00 | 133.2% | SAME DIRECTION, MATERIALLY ATTENUATED |
| **Novelty K3** | 156,580 | 100.0% | +0.1457 [+0.1450, +0.1465] | +0.1274 | 3.62 SD | 2.00 | 124.9% | same direction and similar magnitude |
| **Novelty K10** | 156,580 | 100.0% | +0.1628 [+0.1620, +0.1637] | +0.1430 | 3.67 SD | 2.00 | 124.8% | same direction and similar magnitude |
| **Common Support 5** | 156,580 | 100.0% | +0.1504 [+0.1497, +0.1512] | +0.1314 | 3.63 SD | 2.00 | 124.9% | same direction and similar magnitude |
| **Headroom Retention 99** | 156,580 | 100.0% | +0.1519 [+0.1511, +0.1527] | +0.1328 | 3.64 SD | 2.00 | 124.9% | essentially unchanged under relevance-retention sensitivity |
| **Headroom Retention 95** | 156,580 | 100.0% | +0.1521 [+0.1514, +0.1529] | +0.1332 | 3.65 SD | 2.00 | 124.7% | essentially unchanged under relevance-retention sensitivity |

*(Note: The primary inferential estimand is the user-macro mean (+0.1520 [95% CI: +0.1512, +0.1528], +3.64 SD). The descriptive impression-weighted mean is +0.1393. All variants are compared using user-macro means matching their displayed user-clustered 95% CIs. Headroom retention sensitivities at 99% (+0.1519) and 95% (+0.1521) are essentially unchanged compared to 97% baseline (+0.1520).)*

---
## 5. Robustness Interpretation & Answers to Key Evaluation Questions

1. **Does the Available $\to$ Exposed finding remain negative at 48h?**
   **YES**. The Available $\to$ Exposed discovery gap remains negative and statistically significant at -0.1585 [-0.1597, -0.1573] (-2.42 SD).

2. **How much does its magnitude change from 24h to 48h?**
   The 24h gap is -0.1683 and the 48h gap is -0.1585, representing an absolute change of +0.0098 and an absolute-magnitude reduction of approximately 5.8%. Both analyses evaluate the exact same sample of 156,580 impressions and 11,967 users. 26.28% of observed exposed items were published older than 24h prior to impression time, confirming that admitting older supply modestly attenuates but does not eliminate the gatekeeper exposure novelty deficit.

3. **Does the finding survive the BERT novelty representation?**
   **YES**. The Available $\to$ Exposed finding survives the multilingual BERT novelty representation with discovery gap -0.0081 [-0.0082, -0.0079] and standardized effect -1.08 SD. Raw novelty magnitudes are not directly comparable across embedding representations because representation geometry changes the novelty scale; under standardized metrics, the exposure deficit remains statistically significant and substantial (-1.08 SD). Furthermore, Exposed $\to$ Consumed discovery gap (+0.0049, +0.22 SD; driven by higher relevance coverage with conditional novelty slightly lower at -0.0001), Matched Breadth (+0.0006, +0.06 SD), and Headroom (+0.0261, +1.16 SD; 100.0% positive) all replicate their primary qualitative directions under BERT.

4. **Is it stable for k=3 and k=10?**
   **YES**. The finding is exceptionally stable across history depth: k=3 gap is -0.1529 [-0.1540, -0.1518] (-2.51 SD) and k=10 gap is -0.1927 [-0.1941, -0.1914] (-2.51 SD), displaying completely invariant standardized effect sizes (-2.51 SD).

5. **Does Headroom remain positive with common support $\ge$ 5?**
   **YES**. Discovery Headroom remains positive under stricter common support ($n \ge 5$ distinct users) with user-macro mean gain +0.1504 [+0.1497, +0.1512] (median +0.1314, 3.63 SD, positive headroom share: 100.0%).

6. **Does Headroom remain positive while retaining $\ge$ 99% modeled relevance?**
   **YES**. Discovery Headroom remains strictly positive even under the tightened 99% relevance retention constraint with user-macro mean gain +0.1519 [+0.1511, +0.1527] (median +0.1328, 3.64 SD, positive headroom share: 100.0%). Results are essentially unchanged under relevance-retention sensitivity across the 95% to 99% range (+0.1521 to +0.1519 vs +0.1520 primary user-macro baseline).

7. **Does ANY prespecified variant reverse any major primary conclusion?**
   **NO**. Zero prespecified variants reverse the sign or qualitative conclusion of any major primary finding. All Available $\to$ Exposed discovery gaps remain negative, Exposed $\to$ Consumed discovery gaps remain positive (driven primarily by higher relevance coverage, while conditional novelty is slightly lower among consumed items), Matched Breadth differences remain positive, and Headroom gains remain positive across all 8 variants.

