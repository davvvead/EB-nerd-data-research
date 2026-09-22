"""
Generate outputs/robustness_variant_scope.json and outputs/robustness_variant_scope.md.
Defines the exact operational boundaries, changed components, frozen components,
and ambiguity analyses across all 8 prespecified robustness variants.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

PRIMARY_FREEZE_COMMIT = "598b3553346b9be36195b074530827448fdc8e4b"

variants_scope = {
    "supply_12h": {
        "variant_id": "supply_12h",
        "description": "Retrospective observable supply window reduced from 24h to 12h",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: line 11 (supply_sensitivity_hours: [12, 48])",
            "analysis_protocol_notes.md: line 21 (Supply sensitivity windows: 12 and 48 hours)"
        ],
        "changed_components": [
            "Observable supply window definition: [t_imp - 12h, t_imp)",
            "Available -> Exposed candidate pool for 200 null slates drawn from 12h supply",
            "Counterfactual supply eligibility condition: observable_supply_count >= 3 * observed_slate_size evaluated on 12h supply",
            "Headroom substitution candidate pool: 12h observable supply intersected with common support n >= 3"
        ],
        "frozen_components": [
            "Trained relevance model (frozen weights from cache/train/relevance_model.joblib)",
            "Relevance floor tau (0.09857010114178541 from cache/train/tau.json)",
            "Semantic representation: Contrastive vectors (contrastive_vector.parquet)",
            "Semantic novelty definition: k=5",
            "Observed exposed slates and choice-stage consumption outcomes",
            "Deterministic 1:1 breadth matching (matches on context covariates)",
            "Common support threshold (n >= 3 distinct users)",
            "Headroom relevance retention constraint (>= 97%)"
        ],
        "novelty_cache_policy": "Subset verification: May safely reuse cache/validation/user_article_novelty.parquet (24h Contrastive) ONLY after asserting that 100% of required 12h user-article pairs are present in the 24h cache.",
        "isolated_output_directory": "outputs/robustness/supply_12h/",
        "ambiguity_analysis": "Unambiguous. Protocol explicitly defines 12h as a supply sensitivity window."
    },
    "supply_48h": {
        "variant_id": "supply_48h",
        "description": "Retrospective observable supply window expanded from 24h to 48h",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: line 12 (supply_sensitivity_hours: [12, 48])",
            "analysis_protocol_notes.md: line 21 (Supply sensitivity windows: 12 and 48 hours)"
        ],
        "changed_components": [
            "Observable supply window definition: [t_imp - 48h, t_imp)",
            "Available -> Exposed candidate pool for 200 null slates drawn from 48h supply",
            "Counterfactual supply eligibility condition: observable_supply_count >= 3 * observed_slate_size evaluated on 48h supply",
            "Headroom substitution candidate pool: 48h observable supply intersected with common support n >= 3"
        ],
        "frozen_components": [
            "Trained relevance model (frozen weights from cache/train/relevance_model.joblib)",
            "Relevance floor tau (0.09857010114178541 from cache/train/tau.json)",
            "Semantic representation: Contrastive vectors (contrastive_vector.parquet)",
            "Semantic novelty definition: k=5",
            "Observed exposed slates and choice-stage consumption outcomes",
            "Deterministic 1:1 breadth matching",
            "Common support threshold (n >= 3 distinct users)",
            "Headroom relevance retention constraint (>= 97%)"
        ],
        "novelty_cache_policy": "Expansion required: The 24h primary cache is missing 2,408,253 pairs (28.96% missing; 54.42% of impressions affected). A dedicated cache cache/validation/novelty_contrastive_48h.parquet must be computed and stored before running outcomes.",
        "isolated_output_directory": "outputs/robustness/supply_48h/",
        "ambiguity_analysis": "Unambiguous. Protocol explicitly defines 48h as a supply sensitivity window."
    },
    "bert": {
        "variant_id": "bert",
        "description": "Semantic novelty evaluated using Multilingual BERT embeddings instead of Contrastive embeddings",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: lines 18-20 (semantic_representation: primary: contrastive_vector.parquet, robustness: bert_base_multilingual_cased.parquet)",
            "analysis_protocol_notes.md: lines 8-9 (Robustness semantic representation: multilingual BERT vectors)"
        ],
        "changed_components": [
            "Semantic embedding store: bert_base_multilingual_cased.parquet (768-dimensional float32 unit-normalized vectors)",
            "Candidate novelty computation: d(u, i) = 1 - mean(top-5 BERT cosine similarities between candidate article and user history)",
            "Evaluated discovery metric: D(u, S) across Available -> Exposed, Exposed -> Consumed, Matched Breadth, and Headroom"
        ],
        "frozen_components": [
            "Observable supply window (24 hours)",
            "Candidate pools and null slate sampling weights",
            "Observed slates and choice consumption sets",
            "Deterministic 1:1 breadth matching pairs",
            "Relevance model coefficients and frozen tau (see Ambiguity Analysis below)"
        ],
        "novelty_cache_policy": "Strict isolation: Dedicated cache cache/validation/novelty_bert_24h.parquet with companion metadata asserting embedding_type == 'bert'. Must NEVER load or share cache/validation/user_article_novelty.parquet.",
        "isolated_output_directory": "outputs/robustness/bert/",
        "ambiguity_analysis": "SCOPE BOUNDARY TO CONFIRM: In analysis_protocol_v1.yaml, bert_base_multilingual_cased.parquet is listed under 'semantic_representation: robustness:', which defines 'novelty_definition:'. Under 'relevance_model: primary_features:', the model feature is explicitly named 'contrastive semantic similarity to user history'. The prespecified primary scope is therefore that BERT varies the novelty metric d(u, i) while keeping the frozen train relevance model (tau = 0.098570) fixed. Retraining the relevance model with BERT would require refitting on train, deriving tau_bert, and re-verifying qualification gates. We document the primary scope as varying the novelty measurement while keeping the trained relevance model fixed, and flag full model retraining as a separate multi-stage model variant."
    },
    "novelty_k3": {
        "variant_id": "novelty_k3",
        "description": "Semantic novelty history sensitivity: k=3 instead of primary k=5",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: lines 24-27 (history_similarity_k: 5, history_similarity_k_sensitivity: [3, 10])",
            "IMPLEMENTATION_DECISIONS.md: Item 4 (Derive and cache novelty for k=5 as well as k=3 and k=10)"
        ],
        "changed_components": [
            "Novelty calculation: d(u, i) = 1 - mean(top-3 cosine similarities)",
            "Discovery metrics across all stages computed using novelty_k3"
        ],
        "frozen_components": [
            "Observable supply window (24 hours)",
            "Semantic representation: Contrastive vectors",
            "Trained relevance model and frozen tau (relevance model features remain unchanged)",
            "Observed slates, choice consumption sets, and matched pairs",
            "Headroom constraints and common support"
        ],
        "novelty_cache_policy": "Reuses novelty_k3 column from cache/validation/user_article_novelty.parquet (which already pre-computed k=3, 5, 10 for the 24h universe).",
        "isolated_output_directory": "outputs/robustness/novelty_k3/",
        "ambiguity_analysis": "Unambiguous. Protocol and implementation decisions explicitly pre-computed k=3, 5, 10 in the primary novelty cache."
    },
    "novelty_k10": {
        "variant_id": "novelty_k10",
        "description": "Semantic novelty history sensitivity: k=10 instead of primary k=5",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: lines 24-27 (history_similarity_k: 5, history_similarity_k_sensitivity: [3, 10])",
            "IMPLEMENTATION_DECISIONS.md: Item 4 (Derive and cache novelty for k=5 as well as k=3 and k=10)"
        ],
        "changed_components": [
            "Novelty calculation: d(u, i) = 1 - mean(top-10 cosine similarities)",
            "Discovery metrics across all stages computed using novelty_k10"
        ],
        "frozen_components": [
            "Observable supply window (24 hours)",
            "Semantic representation: Contrastive vectors",
            "Trained relevance model and frozen tau",
            "Observed slates, choice consumption sets, and matched pairs",
            "Headroom constraints and common support"
        ],
        "novelty_cache_policy": "Reuses novelty_k10 column from cache/validation/user_article_novelty.parquet.",
        "isolated_output_directory": "outputs/robustness/novelty_k10/",
        "ambiguity_analysis": "Unambiguous. Protocol and implementation decisions explicitly pre-computed k=3, 5, 10 in the primary novelty cache."
    },
    "common_support_5": {
        "variant_id": "common_support_5",
        "description": "Common-support sensitivity threshold for Headroom: candidate article must appear in-view for >= 5 distinct users in 30-min window (vs >= 3)",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: lines 56-61 (common_support: primary_n_distinct_users: 3, sensitivity_n: 5)",
            "analysis_protocol_notes.md: lines 22-23 (Common-support sensitivity threshold: n >= 5)"
        ],
        "changed_components": [
            "Headroom candidate substitution pool: filtered to articles with >= 5 distinct user in-views in concurrent 30-minute window"
        ],
        "frozen_components": [
            "Stage 1 Available -> Exposed analysis and null draws",
            "Stage 2 Exposed -> Consumed choice analysis",
            "Matched Low-vs-High breadth comparison",
            "Relevance model, tau, and semantic novelty (k=5 contrastive)",
            "Observable supply window (24 hours)",
            "Headroom swap budget (max 2 swaps) and relevance retention constraint (>= 97%)"
        ],
        "novelty_cache_policy": "Reuses cache/validation/user_article_novelty.parquet (the common support filter strictly narrows the candidate pool).",
        "isolated_output_directory": "outputs/robustness/common_support_5/",
        "ambiguity_analysis": "Unambiguous. Protocol explicitly defines sensitivity_n: 5 under common_support."
    },
    "headroom_retention_99": {
        "variant_id": "headroom_retention_99",
        "description": "Headroom modeled relevance retention constraint tightened from 97% to 99% (tolerance 0.01)",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: lines 109-112 (primary_relevance_tolerance: 0.03, sensitivity_relevance_tolerances: [0.01, 0.05])",
            "LOCKED_ANALYSIS_SPEC.md: line 99 (Robustness tolerances: 1% and 5%)"
        ],
        "changed_components": [
            "Headroom optimizer retention constraint: sum(r_hat) >= 0.99 * sum(r_hat_observed)"
        ],
        "frozen_components": [
            "Stage 1, Stage 2, and Matched breadth comparisons",
            "Relevance model and tau floor",
            "Semantic novelty (k=5 contrastive)",
            "Observable supply window (24 hours)",
            "Common support threshold (n >= 3)",
            "Maximum swap budget (2 swaps)"
        ],
        "novelty_cache_policy": "Reuses cache/validation/user_article_novelty.parquet.",
        "isolated_output_directory": "outputs/robustness/headroom_retention_99/",
        "ambiguity_analysis": "Unambiguous. Protocol explicitly lists 0.01 under sensitivity_relevance_tolerances."
    },
    "headroom_retention_95": {
        "variant_id": "headroom_retention_95",
        "description": "Headroom modeled relevance retention constraint relaxed from 97% to 95% (tolerance 0.05)",
        "authoritative_sources": [
            "analysis_protocol_v1.yaml: lines 109-112 (primary_relevance_tolerance: 0.03, sensitivity_relevance_tolerances: [0.01, 0.05])",
            "LOCKED_ANALYSIS_SPEC.md: line 99 (Robustness tolerances: 1% and 5%)"
        ],
        "changed_components": [
            "Headroom optimizer retention constraint: sum(r_hat) >= 0.95 * sum(r_hat_observed)"
        ],
        "frozen_components": [
            "Stage 1, Stage 2, and Matched breadth comparisons",
            "Relevance model and tau floor",
            "Semantic novelty (k=5 contrastive)",
            "Observable supply window (24 hours)",
            "Common support threshold (n >= 3)",
            "Maximum swap budget (2 swaps)"
        ],
        "novelty_cache_policy": "Reuses cache/validation/user_article_novelty.parquet.",
        "isolated_output_directory": "outputs/robustness/headroom_retention_95/",
        "ambiguity_analysis": "Unambiguous. Protocol explicitly lists 0.05 under sensitivity_relevance_tolerances."
    }
}

out_json = {
    "status": "ROBUSTNESS_SCOPE_AUDIT_COMPLETE",
    "protocol_version": "phase1-parameter-freeze-v1",
    "primary_results_freeze_commit_hash": PRIMARY_FREEZE_COMMIT,
    "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "total_prespecified_variants": len(variants_scope),
    "variants": variants_scope
}

Path("outputs/robustness_variant_scope.json").write_text(json.dumps(out_json, indent=2))
print("Saved outputs/robustness_variant_scope.json")

# Generate Markdown report
md_lines = [
    "# Phase 1 Prespecified Robustness Battery Scope Audit",
    "",
    f"**Freeze Date**: 2026-09-22  ",
    f"**Status**: `ROBUSTNESS_SCOPE_AUDIT_COMPLETE`  ",
    f"**Protocol Version**: `phase1-parameter-freeze-v1`  ",
    f"**Primary Freeze Commit**: `{PRIMARY_FREEZE_COMMIT}`  ",
    "**Outcomes Computed**: **0** (Infrastructure and scope audit only; outcomes remain strictly uncomputed)  ",
    "",
    "---",
    "",
    "## 1. Executive Summary & Variant Matrix",
    "",
    "The locked Phase 1 protocol (`analysis_protocol_v1.yaml`) specifies exactly **eight prespecified robustness variants**. Every variant perturbs a single well-defined parameter while strictly preserving all other methodological and model components.",
    "",
    "| Variant ID | Perturbation | Affected Component | Frozen Components | Novelty Cache Identity | Isolated Output Directory |",
    "| :--- | :--- | :--- | :--- | :--- | :--- |",
    "| `supply_12h` | Supply Lookback = 12h | Stage 1 Nulls, 3x filter, Headroom Pool | Relevance Model, tau, Novelty k=5, Choice, Matching | Contrastive 24h (subset verified) | `outputs/robustness/supply_12h/` |",
    "| `supply_48h` | Supply Lookback = 48h | Stage 1 Nulls, 3x filter, Headroom Pool | Relevance Model, tau, Novelty k=5, Choice, Matching | Contrastive 48h (expanded cache) | `outputs/robustness/supply_48h/` |",
    "| `bert` | Representation = BERT | Novelty metric d(u,i), Discovery D(u,S) | Supply 24h, Relevance Model, tau, Choice, Matching | Multilingual BERT 24h (isolated) | `outputs/robustness/bert/` |",
    "| `novelty_k3` | History k = 3 | Novelty metric d(u,i), Discovery D(u,S) | Supply 24h, Relevance Model, tau, Choice, Matching | Contrastive 24h (k3 column) | `outputs/robustness/novelty_k3/` |",
    "| `novelty_k10` | History k = 10 | Novelty metric d(u,i), Discovery D(u,S) | Supply 24h, Relevance Model, tau, Choice, Matching | Contrastive 24h (k10 column) | `outputs/robustness/novelty_k10/` |",
    "| `common_support_5` | Common Support n >= 5 | Headroom substitution candidate pool | Stage 1, Stage 2, Matching, Relevance Model, tau | Contrastive 24h | `outputs/robustness/common_support_5/` |",
    "| `headroom_retention_99` | Relevance Retention >= 99% | Headroom optimization constraint | Stage 1, Stage 2, Matching, Relevance Model, tau | Contrastive 24h | `outputs/robustness/headroom_retention_99/` |",
    "| `headroom_retention_95` | Relevance Retention >= 95% | Headroom optimization constraint | Stage 1, Stage 2, Matching, Relevance Model, tau | Contrastive 24h | `outputs/robustness/headroom_retention_95/` |",
    "",
    "---",
    "",
    "## 2. Detailed Variant Scope Specifications",
    ""
]

for vid, vdata in variants_scope.items():
    md_lines.extend([
        f"### {vid.upper()}: {vdata['description']}",
        f"- **Authoritative Protocol Citations**:",
    ])
    for src in vdata["authoritative_sources"]:
        md_lines.append(f"  - `{src}`")
    md_lines.extend([
        f"- **Changed Components**:",
    ])
    for chg in vdata["changed_components"]:
        md_lines.append(f"  - {chg}")
    md_lines.extend([
        f"- **Frozen Components**:",
    ])
    for frz in vdata["frozen_components"]:
        md_lines.append(f"  - {frz}")
    md_lines.extend([
        f"- **Novelty Cache Policy**: {vdata['novelty_cache_policy']}",
        f"- **Output Directory**: `{vdata['isolated_output_directory']}`",
        f"- **Ambiguity & Boundary Analysis**: {vdata['ambiguity_analysis']}",
        "",
        "---",
        ""
    ])

md_lines.extend([
    "## 3. Ambiguity Resolution & Methodological Boundary Lock",
    "",
    "### Multilingual BERT Scope",
    "- **Question**: Does the Multilingual BERT sensitivity change *only* the semantic novelty metric $d(u, i) = 1 - \\text{mean\\_top\\_k\\_cos}$, or does it also retrain the relevance model with BERT semantic features?",
    "- **Finding from Frozen Protocol**:",
    "  - In `analysis_protocol_v1.yaml`, `bert_base_multilingual_cased.parquet` is specified strictly under `semantic_representation: robustness:`, which governs `novelty_definition: 1 - mean(top-k cosine similarities...)`.",
    "  - In contrast, under `relevance_model: primary_features:`, the semantic feature is explicitly locked as `contrastive semantic similarity to user history`.",
    "  - `IMPLEMENTATION_DECISIONS.md` Item 8 locks the production relevance model fitted on all eligible train examples, whose coefficients and metadata are saved in `cache/train/relevance_model.joblib` and `cache/train/tau.json`.",
    "  - Retraining the relevance model with BERT features would constitute a compound specification change (new model fit, new out-of-fold scoring, new $\\tau_{\\text{BERT}}$, new qualification evaluation).",
    "  - **Resolved Prespecified Scope**: The prespecified BERT robustness variant tests the sensitivity of the novelty metric $d(u, i)$ and resulting discovery outcomes to the semantic representation, while keeping the frozen train relevance model and relevance floor $\\tau = 0.098570$ fixed. A compound retraining model is classified as a non-primary exploratory model variant, preserving pre-registration clarity.",
    "",
    "### Semantic History $k \\in \\{3, 10\\}$ Scope",
    "- **Question**: Does varying $k$ alter the relevance model features or only the discovery novelty metric?",
    "- **Finding from Frozen Protocol**:",
    "  - In `analysis_protocol_v1.yaml` lines 24-27, $k=3$ and $k=10$ are sensitivities of `history_similarity_k: 5` under `semantic_representation:`.",
    "  - In `IMPLEMENTATION_DECISIONS.md` Item 4, $k=3, 5, 10$ were explicitly pre-computed and stored in the novelty cache.",
    "  - **Resolved Scope**: $k=3$ and $k=10$ vary only the discovery novelty metric $d(u, i)$, while the relevance model and $\\tau$ remain strictly frozen.",
    "",
    "---",
    "",
    "## 4. Confirmation of Robustness Run Status",
    "",
    "**ZERO robustness outcomes have been computed.**  ",
    "All output directories contain only unrun initial manifests. All primary frozen artifacts remain untouched.",
    ""
])

Path("outputs/robustness_variant_scope.md").write_text("\n".join(md_lines))
print("Saved outputs/robustness_variant_scope.md")
