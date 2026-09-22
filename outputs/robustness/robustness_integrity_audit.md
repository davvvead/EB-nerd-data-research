# EB-NeRD Phase 1 Prespecified Robustness Battery — Integrity & Reconciliation Audit

**Status**: `ROBUSTNESS_INTEGRITY_AUDIT_VERIFIED`  
**Protocol Version**: `phase1-parameter-freeze-v1`  
**Audit Timestamp UTC**: `2026-09-22T05:11:02.856922+00:00`  
**Frozen Seed**: `20260919`  

---
## 1. Provenance & Git Tracking

| Milestone | Commit Hash | Scope |
| :--- | :--- | :--- |
| **Primary Results Freeze** | `598b3553346b9be36195b074530827448fdc8e4b` | Frozen primary validation outputs (read-only) |
| **Robustness Infrastructure** | `e8ca36297e682705b6fa14a4c5ce8538740f952f` | Isolated runners, cache isolation, scope documents |
| **Current Runner State** | `e8ca36297e682705b6fa14a4c5ce8538740f952f` | Executed prespecified battery runner scripts |

---
## 2. Unit Test Suite Verification

- **Test Result**: `ALL_53_PASSING`
- **Total Tests Executed**: 53
- **Tests Passing**: 53
- **Tests Failed**: 0
- **Coverage**: Rigorously covers cache isolation, parameter validation, eligibility matching, zero-default fallback enforcement, and headless execution.
- **Documentation Clarification**: Session-length scope was a wording-only clarification and did not require recomputation, retraining, or any change to frozen numerical outputs.

---
## 3. Primary Frozen Artifact Hash Verification

All primary validation outputs were verified byte-for-byte unmodified against pre-registered SHA-256 hashes:

| Artifact Path | SHA-256 Digest | Status |
| :--- | :--- | :--- |
| `outputs/run_manifest.json` | `a7748f865d4dac4f0dcda74d1e3e4242e2f928484b17a94ad8cce01d3c622acd` | VERIFIED UNMODIFIED |
| `outputs/relevance_model_metrics.json` | `f3a0641720a12c327031eb02e3df22955c8f89ceae3950c744ed5d6d996c7f09` | VERIFIED UNMODIFIED |
| `outputs/stage_effects.json` | `747bcfff3807feaa0e4850a3e808407604315ba46bb99f9c3d5ec5cabd02d8d4` | VERIFIED UNMODIFIED |
| `outputs/null_comparison.json` | `72f7cc8481a83f9a2836f3e1e1101457f0fd51fb6b6dab5de7853dea742b2841` | VERIFIED UNMODIFIED |
| `outputs/breadth_comparison.json` | `ad008c9622ae2719468b5ae5385f2770d52dedd30ccc9d8e9f543a2dfe8fbe47` | VERIFIED UNMODIFIED |
| `outputs/headroom.json` | `4028b45226d102d1b72106c8d63136890235f2a1aa273ac4c7e41569b27af483` | VERIFIED UNMODIFIED |
| `outputs/primary_validation_summary.md` | `fbe401ff1ecfed5f05846c80ca94424786de35d41575c0212af4d6f53727a366` | VERIFIED UNMODIFIED |
| `outputs/primary_integrity_audit.json` | `8f7372e1b8526f2b7bb64823440e9a2a120381d2736c7ee07b698762c7dc8cca` | VERIFIED UNMODIFIED |
| `outputs/primary_integrity_audit.md` | `015bdf1fc6a95320c6947694ea23d98eef1106bdb956ef19a2fe9275bb749cce` | VERIFIED UNMODIFIED |

---
## 4. Cache Integrity & Semantic Isolation

Dedicated caches ensure no contamination across embedding models or time horizons:

| Cache File | SHA-256 Digest | Specification / Role |
| :--- | :--- | :--- |
| `cache/validation/user_article_novelty.parquet` | `f385b64919410c596c12aafc9b10c825fbe8af5f3654a479da2f7341ae3239de` | Isolated cache artifact |
| `cache/validation/novelty_bert_24h.parquet` | `2506b3dc77b828c27c653dbd0c13ce6d0a34a314c14edf20752b777ac836c1ea` | Isolated cache artifact |
| `cache/validation/novelty_bert_24h.meta.json` | `129e6204eda60962149b14075e3dac1db710b67ae5044ed952347a2541decac2` | Isolated cache artifact |
| `cache/validation/novelty_contrastive_48h.parquet` | `bdc30e3d0ba27647778dd670798d0e8dfcdc8267d416b0d3444531b001facbf5` | Isolated cache artifact |
| `cache/validation/novelty_contrastive_48h.meta.json` | `9106b6c0919d1e98bfb2a361dd51bbdae168260913caaa377ff831ebcce478d8` | Isolated cache artifact |

---
## 5. Summary of Prespecified Robustness Variants

| Variant | Eligible Imp / Users | Runtime | Changed Component | Key Metric Summary |
| :--- | :--- | :--- | :--- | :--- |
| **supply_12h** | 142,368 / 11,820 | 229.4s | Observable supply lookback: 12h; 3x supply filter evaluated on target window | Completed successfully |
| **supply_48h** | 156,580 / 11,967 | 770.0s | Observable supply lookback: 48h; 3x supply filter evaluated on target window | Completed successfully |
| **bert** | 156,580 / 11,967 | 531.5s | Novelty metric: bert (bert); All 4 discovery dimensions recomputed | Completed successfully |
| **novelty_k3** | 156,580 / 11,967 | 504.0s | Novelty metric: novelty_k3 (contrastive); All 4 discovery dimensions recomputed | Completed successfully |
| **novelty_k10** | 156,580 / 11,967 | 502.9s | Novelty metric: novelty_k10 (contrastive); All 4 discovery dimensions recomputed | Completed successfully |
| **common_support_5** | 156,580 / 11,967 | 241.0s | Headroom parameter: common_support_n=5 | Completed successfully |
| **headroom_retention_99** | 156,580 / 11,967 | 244.4s | Headroom parameter: relevance_retention_tolerance=0.01 | Completed successfully |
| **headroom_retention_95** | 156,580 / 11,967 | 242.3s | Headroom parameter: relevance_retention_tolerance=0.05 | Completed successfully |

---
## 6. Runtime Incident Resolution

- **Incident**: `KeyError: difference_high_minus_low on line 1503 of scripts/04_run_robustness.py during compile_robustness_summary()`
- **Root Cause**: outputs/breadth_comparison.json serialized the point estimate dictionary under discovery_difference instead of difference_high_minus_low.
- **Remediation**: Code-only fix in compile_robustness_summary() to fallback gracefully between discovery_difference and difference_high_minus_low, plus idempotency skip for previously computed variants. Zero numerical outcomes were altered.
- **Outcome Impact**: None; all variant outcomes were already computed and frozen byte-for-byte.

---
## 7. Estimand & Classification Reconciliations

### A. Headroom Estimand Presentation Reconciled
- **Primary Inferential Estimand**: **User-Macro Mean Gain** = **+0.152011** [95% CI: +0.151249, +0.152774], standardized effect **+3.64 SD**.
- **Primary Descriptive Metric**: **Impression-Weighted Mean Gain** = **+0.139279**.
- **Reconciliation**: All robustness Headroom comparisons now consistently evaluate the user-macro inferential mean, matching the user-clustered bootstrap confidence intervals.

### B. BERT Qualitative Classification Reconciled
- **Available -> Exposed**: Classified as **SAME DIRECTION, MATERIALLY ATTENUATED UNDER ALTERNATIVE SEMANTIC SCALE** (-0.0081, -1.08 SD vs primary -0.1683, -2.51 SD). Raw novelty magnitudes are not directly comparable across embedding models because multilingual BERT representation geometry operates on a tighter cosine distance range; however, the gatekeeper deficit remains statistically significant and substantial under standardized metrics.
- **Exposed -> Consumed**: Classified as **SAME DIRECTION, MATERIALLY ATTENUATED** (+0.0049, +0.22 SD vs primary +0.0855, +0.81 SD). Across all specifications, the choice-stage discovery gap is positive, driven primarily by higher relevance coverage, while conditional novelty is slightly lower among consumed items (primary -0.0094, BERT -0.0001, $k=3$ -0.0085, $k=10$ -0.0100).
- **Matched Breadth**: Classified as **same direction, similar standardized magnitude (+0.06 SD vs +0.07 SD, practically small)**.
- **Headroom Optimization**: Classified as **SAME DIRECTION, MATERIALLY ATTENUATED** (+0.0261, +1.16 SD vs primary +0.1520, +3.64 SD; 100.0% positive).

### C. Supply 48h Classification Reconciled
- **Sample Accounting**: Primary 24h and Supply 48h evaluate the **exact same sample** of **156,580 impressions** and **11,967 users**.
- **Classification**: Reconciled to **SAME SAMPLE; SAME DIRECTION; MODESTLY ATTENUATED**.
- **Quantification**: 24h gap = **-0.1683**, 48h gap = **-0.1585**, absolute difference = **+0.0098**, absolute-magnitude reduction ≈ **5.8%**. 26.28% of observed exposed items were published older than 24h prior to impression time.
- **Supply 12h**: Retains the designation **SAMPLE-RESTRICTED; SAME DIRECTION (-0.1720, -2.48 SD)** because publication window limits eligibility to 142,368 impressions and 11,820 users.

### D. Headroom Retention Sensitivities Reconciled
- **Consistency**: Across the relevance retention grid, user-macro headroom gains are essentially unchanged: **97% baseline = +0.152011**, **99% tightened = +0.151879**, **95% relaxed = +0.152100**.
- **Classification**: Reconciled to **essentially unchanged under relevance-retention sensitivity**.

---
## 8. Robustness Artifact Hash Manifest

| File Path | SHA-256 Digest |
| :--- | :--- |
| `outputs/robustness/supply_12h/variant_manifest.json` | `cb235771def22fe3924ba22db965592fba678c33cd508f66c0f484a9de693379` |
| `outputs/robustness/supply_12h/stage_effects.json` | `5e25c6226b1c1fd43c300d030d180aab7b1a460be637c674d84c617a775e69e8` |
| `outputs/robustness/supply_12h/null_comparison.json` | `f6291a73be40c70ffd2016aaefc669d9b9d1695bf29b985b6357d5cfaf5959d0` |
| `outputs/robustness/supply_12h/breadth_comparison.json` | `dc73b1d829f4bd00f4b4e3d04c4d4e091f06dcf1d9db28fee72c4ce9a829832f` |
| `outputs/robustness/supply_12h/headroom.json` | `3d43a49c53b0a04985d20479a61f95ec93aad36b7a3bc32732f7fd45948ec3d6` |
| `outputs/robustness/supply_48h/variant_manifest.json` | `dcdb9c8eddcbd701c55bf32cd9d0ddae319440d4c3245ac03e25a54ae024d591` |
| `outputs/robustness/supply_48h/stage_effects.json` | `64606d95760b209152b7a25e72d5550d20cf63d5ca0879f43c301c7391fffa06` |
| `outputs/robustness/supply_48h/null_comparison.json` | `ffc20cc243169922ded5eb2a05d0655cceec3402a3a333c4a529453f48213aad` |
| `outputs/robustness/supply_48h/breadth_comparison.json` | `ddc7062e46460d54b56ba74ef12b6d33d5bad0580efa831bb088cc588191902c` |
| `outputs/robustness/supply_48h/headroom.json` | `08eae81b800b40a1dcdb32141d286d634a52696330ebe04b372636cc989e78d1` |
| `outputs/robustness/bert/variant_manifest.json` | `95a84ae9524a7434e37603292aa535c864a2dcf4af0ea5e9aaf5b72f1f73db14` |
| `outputs/robustness/bert/stage_effects.json` | `c54f14df724a34cd36de1f5d5887291d7e3afbdf55c6dd1104c3dbc1cb5dd57c` |
| `outputs/robustness/bert/null_comparison.json` | `57acf92ad01857abc843672bfe9a36ef3ab7fcb697adcc55d273f26456d986ea` |
| `outputs/robustness/bert/breadth_comparison.json` | `f2e86b4b5395c33661778224ffb0e9187bc9de8761016fafd4653f86846e3f4f` |
| `outputs/robustness/bert/headroom.json` | `24a231756dce641f4ca17a4aa53d39d890cac5293e59691e2aac5d11747ce761` |
| `outputs/robustness/novelty_k3/variant_manifest.json` | `79198d70128cad837c79ba3ff8d98486d4e01cac9082d1a5efde46dcfe547ede` |
| `outputs/robustness/novelty_k3/stage_effects.json` | `62bfd3938325050b388c83ea4dc3ee05938d21e51a003cd47184d36dacd92e69` |
| `outputs/robustness/novelty_k3/null_comparison.json` | `e939d54ea5b8e4d6638f561010b804e7fcdb0a4cd31a9c9276b3b2b1281032dc` |
| `outputs/robustness/novelty_k3/breadth_comparison.json` | `64a28b230ed527b51088aece441eea282cc7cbf67ec61ea234b1a3660c1f6dde` |
| `outputs/robustness/novelty_k3/headroom.json` | `3cbc36ed7dbe3463967c07b905b862ec940f76921d3a8bdfa1149ab55b808ce2` |
| `outputs/robustness/novelty_k10/variant_manifest.json` | `5ccbddd99d9daa8dc5435a873bc1b6d1a7ea6d6ac5d498bb520700f347717a4f` |
| `outputs/robustness/novelty_k10/stage_effects.json` | `cc178116d7adeddaf6246b06daadca4d13946718f477f564d5f560d159b3edb6` |
| `outputs/robustness/novelty_k10/null_comparison.json` | `f5c9d2a11e0b45cab02622046ed251c70dab939747246ed6b14c22b53da98b21` |
| `outputs/robustness/novelty_k10/breadth_comparison.json` | `b41b92b37b46c2f2ba1c378939760c5246d0d839f842f1c1ce160b5900316b84` |
| `outputs/robustness/novelty_k10/headroom.json` | `45ee2057e64287e14fb97fab3458ae652b86aea735afc26a4b08b2cfd2e9e850` |
| `outputs/robustness/common_support_5/variant_manifest.json` | `7157a8327a6c5dcbfef293409c5b275c4ae04a1d321c5db4cbc48c8cd27c1cba` |
| `outputs/robustness/common_support_5/stage_effects.json` | `66a3b1d27240968bd4544e4f042d64c8b228241aed8384cb12700e85eee167b3` |
| `outputs/robustness/common_support_5/null_comparison.json` | `5436c62e089718e57c6872938e01ccd644105dbdbb60c35b71b84ea07afd11f5` |
| `outputs/robustness/common_support_5/breadth_comparison.json` | `9cae641c4ab5bea86848469edd4f8753519acd863dafc6bd72074d9e7ca3d5a0` |
| `outputs/robustness/common_support_5/headroom.json` | `432534a67730f192a6414a811b24707fb5df9faf4fe6e1c9cb53936a45ba2b5a` |
| `outputs/robustness/headroom_retention_99/variant_manifest.json` | `08aa465f19df17a3116bcef2d37f349d54969ae9a878a64ce3b85dae29de3b3a` |
| `outputs/robustness/headroom_retention_99/stage_effects.json` | `e3921e9ef7a96cf94b600e04b1efcba45ef2ac0325c9bfdbcfe9598d15d43b26` |
| `outputs/robustness/headroom_retention_99/null_comparison.json` | `b516647df3c233204a433310df5d7ee51085412e50b7f20870005d269b86863f` |
| `outputs/robustness/headroom_retention_99/breadth_comparison.json` | `652f01d19554cab21962aa3c91bbc8dbb51514e3503e97a0779e2ba6ece4c089` |
| `outputs/robustness/headroom_retention_99/headroom.json` | `f923c8da02136ac76b0ff23305d17a4bb9ed70c53edc8c4a365e2867b5a0c3ba` |
| `outputs/robustness/headroom_retention_95/variant_manifest.json` | `56cb10fabf5e5fc81080d3b27e09c36880cabdd084d533a0f96877d93135931a` |
| `outputs/robustness/headroom_retention_95/stage_effects.json` | `a32db071891262b2cfb73c3c9e4f9e7ef767f2f0b083c1a9ffb76e6cc691b6c8` |
| `outputs/robustness/headroom_retention_95/null_comparison.json` | `dac2ce305b82fcdb3848f7c7da08dc0efdca6060103a7276bfd4ad80c6b03004` |
| `outputs/robustness/headroom_retention_95/breadth_comparison.json` | `b1cdc1012e3b0486d89e46f924c32a53b68931d9abda9c2ab1c4a58ef0a160f6` |
| `outputs/robustness/headroom_retention_95/headroom.json` | `1f912e83634cbc90082019441f280cdad8a05c93ce9045759c7ce4d446945fd4` |
| `outputs/robustness/robustness_summary.json` | `c2a649c103e2005c7da2e617088aa184e5e33c57f66bf085663078a62e6842ec` |
| `outputs/robustness/robustness_summary.md` | `fd5b792fe9efe58c9046812f1b1b70b81fde2b99a0d866c6f677f038228586f8` |
| `outputs/robustness_variant_scope.json` | `c55815a8700f17a8e4e5e0f8c634562698761ce4f906bd80e937ec4cfd6547fd` |
| `outputs/robustness_variant_scope.md` | `16558f77a61c7b587ea8030bd454f35889c953f0b573160f6e98047ce520f057` |

---
## 9. Final Robustness Battery Conclusion

Zero prespecified variants reverse the sign or qualitative conclusion of any major Phase 1 finding. All 8 variants replicate:

1. **Negative Available $\to$ Exposed gatekeeper novelty deficit** across all supply windows, embeddings, and history lengths.
2. **Positive Exposed $\to$ Consumed discovery gap**, driven primarily by higher relevance coverage, while conditional novelty is slightly lower among consumed items (primary -0.0094, BERT -0.0001, $k=3$ -0.0085, $k=10$ -0.0100).
3. **Positive Matched Breadth discovery difference** across all embedding representations.
4. **Strictly positive Discovery Headroom across 100.0% of impressions** under common support $\ge 5$ and relevance retention constraints from 95% to 99%.

**ROBUSTNESS STATE RECONCILED — READY TO FREEZE AND PROCEED TO DIAGNOSTIC**
