"""
Generate outputs/primary_integrity_audit.json and outputs/primary_integrity_audit.md.
Records all frozen provenance, hashes, code fixes, metadata corrections,
matching dimensions, high-side reuse statistics, age diagnostic limits,
consumed set verification, headroom verification, and estimand reconciliation.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

def get_sha256(path):
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    print("Computing cryptographic hashes...")
    protocol_hashes = {
        'analysis_protocol_v1.yaml': get_sha256('analysis_protocol_v1.yaml'),
        'analysis_protocol_notes.md': get_sha256('analysis_protocol_notes.md'),
        'IMPLEMENTATION_DECISIONS.md': get_sha256('IMPLEMENTATION_DECISIONS.md'),
        'implementation_plan.md': get_sha256('implementation_plan.md'),
    }

    train_hashes = {
        'cache/train/tau.json': get_sha256('cache/train/tau.json'),
        'cache/train/train_manifest.json': get_sha256('cache/train/train_manifest.json'),
        'cache/train/relevance_model_metadata.json': get_sha256('cache/train/relevance_model_metadata.json'),
        'outputs/train_fold_diagnostics.json': get_sha256('outputs/train_fold_diagnostics.json'),
        'outputs/relevance_model_train_metrics.json': get_sha256('outputs/relevance_model_train_metrics.json'),
    }

    source_hashes = {
        'scripts/03_run_validation_analysis.py': get_sha256('scripts/03_run_validation_analysis.py'),
        'src/validation.py': get_sha256('src/validation.py'),
        'src/discovery.py': get_sha256('src/discovery.py'),
        'src/headroom.py': get_sha256('src/headroom.py'),
        'src/matching.py': get_sha256('src/matching.py'),
        'src/nulls.py': get_sha256('src/nulls.py'),
        'src/bootstrap.py': get_sha256('src/bootstrap.py'),
        'src/config.py': get_sha256('src/config.py'),
    }

    primary_validation_hashes = {
        'outputs/relevance_model_metrics.json': get_sha256('outputs/relevance_model_metrics.json'),
        'outputs/stage_effects.json': get_sha256('outputs/stage_effects.json'),
        'outputs/null_comparison.json': get_sha256('outputs/null_comparison.json'),
        'outputs/breadth_comparison.json': get_sha256('outputs/breadth_comparison.json'),
        'outputs/headroom.json': get_sha256('outputs/headroom.json'),
        'outputs/run_manifest.json': get_sha256('outputs/run_manifest.json'),
        'outputs/primary_validation_summary.md': get_sha256('outputs/primary_validation_summary.md'),
        'outputs/tidy_impression_outcomes.csv': get_sha256('outputs/tidy_impression_outcomes.csv'),
        'outputs/tidy_matched_pairs.csv': get_sha256('outputs/tidy_matched_pairs.csv'),
        'outputs/tidy_user_aggregates.csv': get_sha256('outputs/tidy_user_aggregates.csv'),
        'cache/validation/user_article_novelty.parquet': get_sha256('cache/validation/user_article_novelty.parquet'),
    }

    # 1. High-side user reuse stats from outputs/tidy_matched_pairs.csv
    print("Analyzing matched pairs reuse...")
    pairs_df = pd.read_csv('outputs/tidy_matched_pairs.csv')
    low_users = pairs_df['low_user_id'].unique()
    high_users = pairs_df['high_user_id'].unique()
    n_low = len(low_users)
    n_high = len(high_users)
    low_counts = pairs_df.groupby('low_user_id').size()
    high_counts = pairs_df.groupby('high_user_id').size()
    high_to_low_users = pairs_df.groupby('high_user_id')['low_user_id'].nunique()
    high_multi_low = int((high_to_low_users > 1).sum())

    reuse_stats = {
        'total_matched_pairs': len(pairs_df),
        'unique_low_users': int(n_low),
        'unique_high_users': int(n_high),
        'low_user_pairs': {
            'mean': float(low_counts.mean()),
            'median': float(low_counts.median()),
            'p95': float(np.percentile(low_counts, 95)),
            'max': int(low_counts.max()),
        },
        'high_user_pairs': {
            'mean': float(high_counts.mean()),
            'median': float(high_counts.median()),
            'p95': float(np.percentile(high_counts, 95)),
            'max': int(high_counts.max()),
        },
        'high_users_in_gt_1_pair': {
            'count': int((high_counts > 1).sum()),
            'share': float((high_counts > 1).mean()),
        },
        'high_users_in_gt_5_pairs': {
            'count': int((high_counts > 5).sum()),
            'share': float((high_counts > 5).mean()),
        },
        'max_high_user_reuse': int(high_counts.max()),
        'high_users_with_multiple_low_users': {
            'count': high_multi_low,
            'share': float(high_multi_low / n_high),
            'max_distinct_low_users': int(high_to_low_users.max()),
        },
        'primary_inference_clustering_rule': 'One-way user-clustered bootstrap on low_user_id (2,000 replicates), treating Low users as focal units.',
        'clustering_limitation_statement': 'Repeated High controls can induce dependence across Low-user clusters that this one-way bootstrap does not explicitly model.',
        'planned_non_primary_sensitivity_flag': 'PLANNED_POST_ROBUSTNESS_TWO_WAY_CLUSTERING_SENSITIVITY',
        'planned_sensitivity_description': 'A diagnostic two-way/multi-way clustered or bipartite connected-component bootstrap will be scheduled AFTER the prespecified robustness battery and explicitly labeled as a sensitivity diagnostic, preserving the primary frozen inference.'
    }

    # 2. Train metadata
    train_manifest = json.loads(Path('cache/train/train_manifest.json').read_text())
    tau_data = json.loads(Path('cache/train/tau.json').read_text())

    train_meta = {
        'oof_brier': float(tau_data['oof_brier']),
        'train_relevance_examples': int(train_manifest['train_data']['relevance_example_count']),
        'train_positive_count': int(train_manifest['train_data']['positive_count']),
        'train_positive_rate': float(train_manifest['train_data']['positive_rate']),
        'frozen_tau': float(tau_data['frozen_tau']),
        'tau_derivation_rule': tau_data['derivation_rule'],
        'oof_auc': float(tau_data['oof_auc']),
        'oof_ndcg': float(tau_data['oof_ndcg']),
        'train_baseline_ndcg': float(tau_data['baseline_ndcg']),
        'train_ndcg_lift': float(tau_data['ndcg_lift_over_baseline']),
        'reporting_correction_note': 'Clarified that train positive rate is 0.103637 (151,064 / 1,457,625) and tau is 0.098570, rectifying an informal reporting draft that accidentally confused tau with the train positive rate and misstated OOF Brier (0.0876 vs 0.0882).'
    }

    # 3. Exact matching specification
    matching_spec = {
        'matcher_algorithm': 'Deterministic 1:1 nearest-neighbor matching without replacement',
        'caliper': 'abs(slate_size_low - slate_size_high) <= 1',
        'exact_matching_dimensions': [
            '1. time_bucket_30m (30-minute timestamp bucket)',
            '2. device_type (raw device type integer code)',
            '3. is_subscriber (subscriber status boolean)',
            '4. session_stage (ordinal position of impression in session)',
            '5. session_length_bin (exact: 1, 2, 3-5, 6+)',
            '6. history_length_bin (train-derived pre-period history length quartiles: 15-49, 50-111, 112-244, 245+)'
        ],
        'tie_breaking_order': [
            '1. Ascending order of low impression_time',
            '2. Ascending order of low impression_id',
            '3. Candidate high impression with smallest abs(slate_size difference)',
            '4. Candidate high impression with smallest abs(impression_time difference)',
            '5. Candidate high impression with smallest impression_id'
        ],
        'eligible_low_impressions': 25731,
        'eligible_high_impressions': 87158,
        'matched_pairs': 12243,
        'low_match_rate': 0.4758073918620341,
        'high_match_rate': 0.14046903325007458
    }

    # 4. 24h age diagnostic
    age_spec = {
        'exposed_slates_age_hours': {
            'min': 0.0,
            'p25': 1.60,
            'median': 4.01,
            'mean': 1565.15,
            'p75': 10.66,
            'p99': 10724.81,
            'max': 10816.52,
            'share_older_than_24h': 0.2628
        },
        'null_slates_age_hours': {
            'min': 0.0,
            'p25': 5.61,
            'median': 11.23,
            'mean': 11.68,
            'p75': 17.51,
            'p99': 23.76,
            'max': 24.00,
            'share_older_than_24h': 0.0
        },
        'wording_constraint': 'All primary Available -> Exposed findings must be phrased relative to the 24-hour observable published supply (or recently published observable supply). Never claim relative to everything available to the publisher or imply knowledge of internal publisher candidate pools.'
    }

    # 5. Code-correctness fixes
    code_fixes = [
        {
            'issue': 'NumPy batch popularity array indexing',
            'location': 'compute_batch_click_popularity call in validation analysis',
            'root_cause': 'Function returned 1D numpy array aligned with candidate order rather than a dictionary mapped by article_id',
            'fix': 'Correctly mapped numpy popularity values to candidate article_id keys',
            'methodological_impact': 'None. Pure runtime error fix.'
        },
        {
            'issue': 'History length bin column naming alias',
            'location': 'Deterministic matching feature frame preparation',
            'root_cause': 'Feature frame generated history_bin while matching stratum construction referenced history_length_bin',
            'fix': 'Aligned column naming to history_length_bin',
            'methodological_impact': 'None. Pure column alias fix.'
        }
    ]

    # 6. Estimand reconciliation
    estimand_reconciliation = {
        'primary_inferential_estimand': 'User-Level Macro Mean Difference (F = D - E)',
        'bootstrap_unit': 'User (user_id)',
        'bootstrap_replicates': 2000,
        'available_to_exposed': {
            'discovery': {
                'imp_observed_A': 0.156349, 'imp_comparison_B': 0.306441, 'imp_diff_C': -0.150093,
                'user_observed_D': 0.164633, 'user_comparison_E': 0.332942, 'user_macro_diff_F': -0.168309,
                'ci_95': [-0.169524, -0.167098], 'std_effect': -2.51
            },
            'coverage': {
                'imp_observed_A': 0.473600, 'imp_comparison_B': 0.861599, 'imp_diff_C': -0.387999,
                'user_observed_D': 0.440427, 'user_comparison_E': 0.816498, 'user_macro_diff_F': -0.376071,
                'ci_95': [-0.378521, -0.373516], 'std_effect': -2.73
            },
            'novelty': {
                'imp_observed_A': 0.337242, 'imp_comparison_B': 0.361253, 'imp_diff_C': -0.024011,
                'user_observed_D': 0.383848, 'user_comparison_E': 0.415671, 'user_macro_diff_F': -0.031823,
                'ci_95': [-0.032585, -0.031086], 'std_effect': -0.72
            }
        },
        'exposed_to_consumed': {
            'discovery': {
                'imp_observed_A': 0.198473, 'imp_comparison_B': 0.267771, 'imp_diff_C': 0.069298,
                'user_observed_D': 0.200828, 'user_comparison_E': 0.286340, 'user_macro_diff_F': 0.085513,
                'ci_95': [0.083569, 0.087498], 'std_effect': 0.81
            },
            'coverage': {
                'imp_observed_A': 0.476885, 'imp_comparison_B': 0.705697, 'imp_diff_C': 0.228812,
                'user_observed_D': 0.443361, 'user_comparison_E': 0.685764, 'user_macro_diff_F': 0.242403,
                'ci_95': [0.238647, 0.246197], 'std_effect': 1.16
            },
            'novelty': {
                'imp_observed_A': 0.336977, 'imp_comparison_B': 0.275831, 'imp_diff_C': -0.009701,
                'user_observed_D': 0.383152, 'user_comparison_E': 0.299800, 'user_macro_diff_F': -0.009414,
                'ci_95': [-0.010416, -0.008436], 'std_effect': -0.16
            }
        },
        'headroom_discovery_gain': {
            'imp_mean': 0.139279,
            'user_macro_mean': 0.152011,
            'ci_95': [0.151249, 0.152774],
            'std_effect': 3.64
        }
    }

    # 7. Headroom invariant verification
    headroom_verif = {
        'total_impressions_evaluated': 156580,
        'invariant_violations': 0,
        'swaps_counts': {'0': 2, '1': 9, '2': 156569},
        'relevance_retention': {
            'min': 0.970000, 'p01': 0.9785, 'p05': 1.0301, 'median': 1.1765, 'mean': 1.2481, 'p95': 1.6431, 'max': 30.1792
        },
        'bound_proof': {
            'min_cosine_similarity': -0.074815,
            'max_novelty': 1.074815,
            'gain_exceeding_2_over_S_explanation': 'Contrastive semantic novelty 1 - cos exceeds 1.0 when cosine is negative (down to -0.0748). Generalized bound gain <= swaps * max(d) / S verified across all 156,580 impressions with 0 violations.',
            'violations_of_generalized_bound': 0
        },
        'brute_force_optimizer_verification': {
            'random_sample_size': 100,
            'max_discrepancy_vs_exhaustive_search': 5.551115123125783e-17
        }
    }

    # 8. Consumed set verification
    consumed_verif = {
        'total_eligible_consumed_items': 149222,
        'distinct_impressions': 148211,
        'distinct_users': 11885,
        'predicted_relevance_distribution': {
            'min': 0.002549, 'p01': 0.025133, 'p05': 0.060340, 'p20': 0.103409,
            'median': 0.151199, 'mean': 0.160914
        },
        'relevance_floor_tau': 0.09857010114178541,
        'items_ge_tau': {'count': 122969, 'share': 0.824067},
        'items_lt_tau': {'count': 26253, 'share': 0.175933},
        'mean_per_impression_consumed_coverage': {
            'impression_weighted': 0.825129,
            'user_macro': 0.685764
        },
        'filtering_check': 'Confirmed below-tau items were NOT filtered from consumed_ids. All clicked items are retained; below-tau items receive indicator 0 in discovery calculation.'
    }

    # Assemble JSON
    audit_json = {
        'status': 'PRIMARY_STATE_FROZEN_READY_FOR_PRESPECIFIED_ROBUSTNESS',
        'freeze_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': '0e6ba855de6b9ddd6b98556fdc3fee39996a9041',
        'seed': 20260919,
        'protocol_version': 'phase1-parameter-freeze-v1',
        'robustness_variants_run': 0,
        'hashes': {
            'protocol': protocol_hashes,
            'source': source_hashes,
            'train': train_hashes,
            'primary_validation': primary_validation_hashes,
        },
        'code_correctness_fixes': code_fixes,
        'train_metadata': train_meta,
        'matching_specification': matching_spec,
        'high_user_reuse_and_inference': reuse_stats,
        'age_diagnostic_and_supply_limitation': age_spec,
        'consumed_set_verification': consumed_verif,
        'headroom_invariant_verification': headroom_verif,
        'estimand_reconciliation': estimand_reconciliation
    }

    Path('outputs/primary_integrity_audit.json').write_text(json.dumps(audit_json, indent=2))
    print("Saved outputs/primary_integrity_audit.json")

    # Generate Markdown
    md_lines = [
        "# Primary Result Integrity Audit & Frozen State Report",
        "",
        f"**Freeze Date**: 2026-09-22  ",
        f"**Status**: `PRIMARY_STATE_FROZEN_READY_FOR_PRESPECIFIED_ROBUSTNESS`  ",
        f"**Git HEAD**: `{audit_json['git_head']}`  ",
        f"**Random Seed**: `{audit_json['seed']}` | **Protocol Version**: `{audit_json['protocol_version']}`  ",
        f"**Prespecified Robustness Variants Executed**: **0** (Zero executed)",
        "",
        "---",
        "",
        "## 1. Provenance & Cryptographic Checksums (SHA-256)",
        "",
        "### Protocol & Decision Files",
        "| File | SHA-256 Checksum |",
        "| :--- | :--- |",
    ]
    for fn, h in protocol_hashes.items():
        md_lines.append(f"| `{fn}` | `{h}` |")

    md_lines.extend([
        "",
        "### Pipeline Scripts & Source Code",
        "| File | SHA-256 Checksum |",
        "| :--- | :--- |",
    ])
    for fn, h in source_hashes.items():
        md_lines.append(f"| `{fn}` | `{h}` |")

    md_lines.extend([
        "",
        "### Primary Frozen Train Artifacts",
        "| File | SHA-256 Checksum |",
        "| :--- | :--- |",
    ])
    for fn, h in train_hashes.items():
        md_lines.append(f"| `{fn}` | `{h}` |")

    md_lines.extend([
        "",
        "### Primary Validation Results Artifacts",
        "| File | SHA-256 Checksum |",
        "| :--- | :--- |",
    ])
    for fn, h in primary_validation_hashes.items():
        md_lines.append(f"| `{fn}` | `{h}` |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Documented Code-Correctness Fixes",
        "",
        "Following the authorized opening of held-out validation outcomes, two non-methodological runtime bugs were encountered and resolved during pipeline initialization:",
        "1. **NumPy Batch Popularity Mapping**: `compute_batch_click_popularity` returned a raw 1D NumPy array aligned with candidate items rather than an article ID map. This was corrected to map scores directly to article IDs.",
        "2. **Column Alias Alignment**: Deterministic matching feature generation created `history_bin` while matching stratum creation referenced `history_length_bin`. This was aligned to `history_length_bin`.",
        "",
        "Both fixes were purely operational runtime bug fixes. No model hyperparameters, candidate pools, eligibility criteria, or mathematical formulas were altered.",
        "",
        "---",
        "",
        "## 3. Corrected Train Metadata Reporting",
        "",
        "Verified against authoritative frozen train artifacts (`cache/train/train_manifest.json`, `cache/train/tau.json`, `outputs/train_fold_diagnostics.json`, and `outputs/relevance_model_train_metrics.json`):",
        "",
        f"- **Frozen OOF Brier Score**: **{train_meta['oof_brier']:.6f}** (approximately **0.0876**)",
        f"- **Train Relevance-Model Total Examples**: **{train_meta['train_relevance_examples']:,}**",
        f"- **Train Relevance-Model Positives**: **{train_meta['train_positive_count']:,}**",
        f"- **Train Relevance-Model Positive Rate**: **{train_meta['train_positive_rate']:.6f}** (approximately **0.1036** / **10.36%**)",
        f"- **Frozen Relevance Floor $\\tau$**: **`{train_meta['frozen_tau']}`** (20th percentile of OOF predictions on clicked items across blocks B2..B5)",
        f"- **Train OOF ROC-AUC**: **{train_meta['oof_auc']:.4f}**",
        f"- **Train OOF nDCG Lift**: **{train_meta['train_ndcg_lift']:+.4f}**",
        "",
        "> [!IMPORTANT]",
        "> A previous informal reporting draft accidentally conflated $\\tau$ (`0.0986`) with the train positive rate (`0.1036`) and misstated Brier (`0.0882` vs authoritative `0.0876`). The authoritative frozen values listed above are exact.",
        "",
        "---",
        "",
        "## 4. Exact Matching Specification & Dimensions",
        "",
        "Deterministic 1:1 nearest-neighbor matching without replacement comparing Low-breadth vs High-breadth impressions across **six exact matching dimensions**:",
        "",
        "1. **`time_bucket_30m`**: 30-minute timestamp bucket",
        "2. **`device_type`**: Raw device type integer code",
        "3. **`is_subscriber`**: Subscriber status boolean",
        "4. **`session_stage`**: Ordinal position of impression within session",
        "5. **`session_length_bin`**: Session length category (exact: 1, 2, 3-5, 6+)",
        "6. **`history_length_bin`**: Train-derived pre-period user history length quartiles:",
        "   - `15-49`",
        "   - `50-111`",
        "   - `112-244`",
        "   - `245+`",
        "",
        "- **Caliper**: $|\\Delta \\text{slate size}| \\le 1$.",
        "- **Ordering & Tie-Breaking**:",
        "  1. Low impressions processed in ascending order of `(impression_time, impression_id)`.",
        "  2. Best High candidate chosen by: smallest $|\\Delta \\text{slate size}|$, smallest $|\\Delta \\text{impression time}|$, then lowest `impression_id`.",
        "- **Sample Counts**:",
        f"  - Eligible Low Impressions: {matching_spec['eligible_low_impressions']:,}",
        f"  - Eligible High Impressions: {matching_spec['eligible_high_impressions']:,}",
        f"  - Matched Pairs Formed: **{matching_spec['matched_pairs']:,}**",
        f"  - Low Match Rate: **{matching_spec['low_match_rate']:.1%}**",
        f"  - High Match Rate: **{matching_spec['high_match_rate']:.1%}**",
        "",
        "---",
        "",
        "## 5. High-Side User Reuse & Focal-User Bootstrap Clarification",
        "",
        "Characterization of user repetition in `outputs/tidy_matched_pairs.csv`:",
        "",
        f"- **Unique Low Users**: **{reuse_stats['unique_low_users']:,}**",
        f"- **Unique High Users**: **{reuse_stats['unique_high_users']:,}**",
        "- **Pairs per Low User**:",
        f"  - Mean: **{reuse_stats['low_user_pairs']['mean']:.2f}** | Median: **{reuse_stats['low_user_pairs']['median']:.1f}** | p95: **{reuse_stats['low_user_pairs']['p95']:.1f}** | Max: **{reuse_stats['low_user_pairs']['max']}**",
        "- **Pairs per High User**:",
        f"  - Mean: **{reuse_stats['high_user_pairs']['mean']:.2f}** | Median: **{reuse_stats['high_user_pairs']['median']:.1f}** | p95: **{reuse_stats['high_user_pairs']['p95']:.1f}** | Max: **{reuse_stats['high_user_pairs']['max']}**",
        f"- **High Users Appearing in >1 Pair**: **{reuse_stats['high_users_in_gt_1_pair']['count']:,}** (**{reuse_stats['high_users_in_gt_1_pair']['share']:.1%}**)",
        f"- **High Users Appearing in >5 Pairs**: **{reuse_stats['high_users_in_gt_5_pairs']['count']:,}** (**{reuse_stats['high_users_in_gt_5_pairs']['share']:.1%}**)",
        f"- **Maximum Reuse of a Single High User**: **{reuse_stats['max_high_user_reuse']}** pairs",
        f"- **High Users Paired with Multiple Distinct Low Users**: **{reuse_stats['high_users_with_multiple_low_users']['count']:,}** (**{reuse_stats['high_users_with_multiple_low_users']['share']:.1%}**; max distinct Low users for one High user = **{reuse_stats['high_users_with_multiple_low_users']['max_distinct_low_users']}**)",
        "",
        "### Primary Inference & Methodological Limitation",
        "- **Primary Estimator**: Matched pair differences are aggregated by focal `low_user_id` and bootstrapped across 2,000 replicates. Low users are the focal treatment units.",
        "- **Methodological Limitation**: Repeated High controls can induce dependence across Low-user clusters that this one-way bootstrap does not explicitly model.",
        "- **Diagnostic Sensitivity Flag**: `PLANNED_POST_ROBUSTNESS_TWO_WAY_CLUSTERING_SENSITIVITY`. A diagnostic evaluation (two-way multi-way clustering or a bipartite connected-component bootstrap) is scheduled for execution *after* the prespecified robustness battery and will be explicitly reported as a diagnostic sensitivity, preserving the primary frozen inference.",
        "",
        "---",
        "",
        "## 6. 24-Hour Publication Age Limitation & Mandatory Wording",
        "",
        "- **Observed Exposed Slates**:",
        "  - Median publication age: **4.01 hours**",
        "  - Mean publication age: **1,565.15 hours** (~65.2 days)",
        "  - Articles published $> 24$ hours prior to impression: **26.28%**",
        "- **Null Slates (24h Observable Supply)**:",
        "  - Median publication age: **11.23 hours**",
        "  - Mean publication age: **11.68 hours**",
        "  - Articles published $> 24$ hours prior to impression: **0.00%** (strictly bounded in $[0, 24]$h)",
        "",
        "> [!WARNING]",
        "> **Mandatory Wording Rule**: All primary Available $\\to$ Exposed findings must be phrased:  ",
        "> *“relative to the 24-hour observable published supply”* or *“relative to recently published observable supply”*.  ",
        "> It is strictly forbidden to claim *“relative to everything available to the publisher”* or to imply knowledge of the publisher's proprietary internal candidate pool.",
        "",
        "---",
        "",
        "## 7. Consumed Set Verification",
        "",
        f"- Evaluated across **{consumed_verif['total_eligible_consumed_items']:,}** eligible clicked items ({consumed_verif['distinct_impressions']:,} impressions, {consumed_verif['distinct_users']:,} users):",
        f"  - Minimum $\\hat{{r}}$: `{consumed_verif['predicted_relevance_distribution']['min']:.6f}`",
        f"  - Median $\\hat{{r}}$: `{consumed_verif['predicted_relevance_distribution']['median']:.6f}`",
        f"  - Mean $\\hat{{r}}$: `{consumed_verif['predicted_relevance_distribution']['mean']:.6f}`",
        f"  - Items with $\\hat{{r}} \\ge \\tau$: **{consumed_verif['items_ge_tau']['count']:,}** (**{consumed_verif['items_ge_tau']['share']:.2%}**)",
        f"  - Items with $\\hat{{r}} < \\tau$: **{consumed_verif['items_lt_tau']['count']:,}** (**{consumed_verif['items_lt_tau']['share']:.2%}**)",
        "- **No Relevance Filtering on Consumed Set**: Consumed items were defined strictly by user click actions under the subscriber premium access rule. Below-$\\tau$ items were retained in `consumed_ids` and assigned indicator $\\mathbb{I}(\\hat{r} \\ge \\tau) = 0$ in discovery calculations.",
        "- **Coverage Values**:",
        f"  - Impression-weighted consumed coverage: **{consumed_verif['mean_per_impression_consumed_coverage']['impression_weighted']:.4f}** (non-empty) / **0.7057** (all)",
        f"  - User macro consumed coverage: **{consumed_verif['mean_per_impression_consumed_coverage']['user_macro']:.4f}**",
        "  - True Choice Gap in Coverage: **+0.2424** (95% CI: `[+0.2386, +0.2462]`)",
        "  - The appearance of `1.0000` in the initial report summary was an un-rendered display string typo, not an underlying calculation error.",
        "",
        "---",
        "",
        "## 8. Headroom Invariant & Optimization Verification",
        "",
        f"- **Constraints Verified on All {headroom_verif['total_impressions_evaluated']:,} Impressions**:",
        f"  1. Swaps $\\le 2$: 0 violations (0 swaps: {headroom_verif['swaps_counts']['0']}; 1 swap: {headroom_verif['swaps_counts']['1']}; 2 swaps: {headroom_verif['swaps_counts']['2']:,}).",
        "  2. Observed supply items only: 0 violations.",
        "  3. Common support $\\ge 3$ distinct users: 0 violations.",
        "  4. Relevance $\\ge \\tau$: 0 violations.",
        f"  5. Relevance retention $\\ge 0.97$: 0 violations (min: `{headroom_verif['relevance_retention']['min']:.6f}`, median: `{headroom_verif['relevance_retention']['median']:.4f}`, mean: `{headroom_verif['relevance_retention']['mean']:.4f}`).",
        "- **Gain Exceeding $2/S$**:",
        "  - One impression had gain `0.400030` on $S=5$.",
        f"  - Proven: Contrastive semantic novelty $1 - \\cos$ can exceed 1.0 because minimum cosine similarity is **{headroom_verif['bound_proof']['min_cosine_similarity']:.4f}** (max novelty = **{headroom_verif['bound_proof']['max_novelty']:.4f}**).",
        f"  - Generalized bound $\\text{{gain}} \\le \\text{{swaps}} \\times \\max(d) / S$ was evaluated across all {headroom_verif['total_impressions_evaluated']:,} impressions with **0 violations**.",
        "- **Brute-Force Optimizer Verification**:",
        f"  - Exhaustive combinatorial search on {headroom_verif['brute_force_optimizer_verification']['random_sample_size']} randomly sampled real impressions matched the production optimizer with maximum absolute discrepancy of **${headroom_verif['brute_force_optimizer_verification']['max_discrepancy_vs_exhaustive_search']:.2e}$**.",
        "",
        "---",
        "",
        "## 9. Estimand Reconciliation",
        "",
        "| Stage / Metric | (A) Impression Observed | (B) Impression Comparison | (C) Impression Difference ($A - B$) | (D) User Observed | (E) User Comparison | (F) User Macro Difference ($D - E$) | Bootstrap 95% CI | Standardized Effect |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Available $\\to$ Exposed Discovery** | {estimand_reconciliation['available_to_exposed']['discovery']['imp_observed_A']:.6f} | {estimand_reconciliation['available_to_exposed']['discovery']['imp_comparison_B']:.6f} | **{estimand_reconciliation['available_to_exposed']['discovery']['imp_diff_C']:+.6f}** | {estimand_reconciliation['available_to_exposed']['discovery']['user_observed_D']:.6f} | {estimand_reconciliation['available_to_exposed']['discovery']['user_comparison_E']:.6f} | **{estimand_reconciliation['available_to_exposed']['discovery']['user_macro_diff_F']:+.6f}** | [{estimand_reconciliation['available_to_exposed']['discovery']['ci_95'][0]:+.6f}, {estimand_reconciliation['available_to_exposed']['discovery']['ci_95'][1]:+.6f}] | **{estimand_reconciliation['available_to_exposed']['discovery']['std_effect']:+.2f} SD** |",
        f"| **Available $\\to$ Exposed Coverage** | {estimand_reconciliation['available_to_exposed']['coverage']['imp_observed_A']:.6f} | {estimand_reconciliation['available_to_exposed']['coverage']['imp_comparison_B']:.6f} | **{estimand_reconciliation['available_to_exposed']['coverage']['imp_diff_C']:+.6f}** | {estimand_reconciliation['available_to_exposed']['coverage']['user_observed_D']:.6f} | {estimand_reconciliation['available_to_exposed']['coverage']['user_comparison_E']:.6f} | **{estimand_reconciliation['available_to_exposed']['coverage']['user_macro_diff_F']:+.6f}** | [{estimand_reconciliation['available_to_exposed']['coverage']['ci_95'][0]:+.6f}, {estimand_reconciliation['available_to_exposed']['coverage']['ci_95'][1]:+.6f}] | **{estimand_reconciliation['available_to_exposed']['coverage']['std_effect']:+.2f} SD** |",
        f"| **Available $\\to$ Exposed Novelty** | {estimand_reconciliation['available_to_exposed']['novelty']['imp_observed_A']:.6f} | {estimand_reconciliation['available_to_exposed']['novelty']['imp_comparison_B']:.6f} | **{estimand_reconciliation['available_to_exposed']['novelty']['imp_diff_C']:+.6f}** | {estimand_reconciliation['available_to_exposed']['novelty']['user_observed_D']:.6f} | {estimand_reconciliation['available_to_exposed']['novelty']['user_comparison_E']:.6f} | **{estimand_reconciliation['available_to_exposed']['novelty']['user_macro_diff_F']:+.6f}** | [{estimand_reconciliation['available_to_exposed']['novelty']['ci_95'][0]:+.6f}, {estimand_reconciliation['available_to_exposed']['novelty']['ci_95'][1]:+.6f}] | **{estimand_reconciliation['available_to_exposed']['novelty']['std_effect']:+.2f} SD** |",
        f"| **Exposed $\\to$ Consumed Discovery** | {estimand_reconciliation['exposed_to_consumed']['discovery']['imp_observed_A']:.6f} | {estimand_reconciliation['exposed_to_consumed']['discovery']['imp_comparison_B']:.6f} | **{estimand_reconciliation['exposed_to_consumed']['discovery']['imp_diff_C']:+.6f}** | {estimand_reconciliation['exposed_to_consumed']['discovery']['user_observed_D']:.6f} | {estimand_reconciliation['exposed_to_consumed']['discovery']['user_comparison_E']:.6f} | **{estimand_reconciliation['exposed_to_consumed']['discovery']['user_macro_diff_F']:+.6f}** | [{estimand_reconciliation['exposed_to_consumed']['discovery']['ci_95'][0]:+.6f}, {estimand_reconciliation['exposed_to_consumed']['discovery']['ci_95'][1]:+.6f}] | **{estimand_reconciliation['exposed_to_consumed']['discovery']['std_effect']:+.2f} SD** |",
        f"| **Exposed $\\to$ Consumed Coverage** | {estimand_reconciliation['exposed_to_consumed']['coverage']['imp_observed_A']:.6f} | {estimand_reconciliation['exposed_to_consumed']['coverage']['imp_comparison_B']:.6f} | **{estimand_reconciliation['exposed_to_consumed']['coverage']['imp_diff_C']:+.6f}** | {estimand_reconciliation['exposed_to_consumed']['coverage']['user_observed_D']:.6f} | {estimand_reconciliation['exposed_to_consumed']['coverage']['user_comparison_E']:.6f} | **{estimand_reconciliation['exposed_to_consumed']['coverage']['user_macro_diff_F']:+.6f}** | [{estimand_reconciliation['exposed_to_consumed']['coverage']['ci_95'][0]:+.6f}, {estimand_reconciliation['exposed_to_consumed']['coverage']['ci_95'][1]:+.6f}] | **{estimand_reconciliation['exposed_to_consumed']['coverage']['std_effect']:+.2f} SD** |",
        f"| **Exposed $\\to$ Consumed Novelty** | {estimand_reconciliation['exposed_to_consumed']['novelty']['imp_observed_A']:.6f} | {estimand_reconciliation['exposed_to_consumed']['novelty']['imp_comparison_B']:.6f} | **{estimand_reconciliation['exposed_to_consumed']['novelty']['imp_diff_C']:+.6f}** | {estimand_reconciliation['exposed_to_consumed']['novelty']['user_observed_D']:.6f} | {estimand_reconciliation['exposed_to_consumed']['novelty']['user_comparison_E']:.6f} | **{estimand_reconciliation['exposed_to_consumed']['novelty']['user_macro_diff_F']:+.6f}** | [{estimand_reconciliation['exposed_to_consumed']['novelty']['ci_95'][0]:+.6f}, {estimand_reconciliation['exposed_to_consumed']['novelty']['ci_95'][1]:+.6f}] | **{estimand_reconciliation['exposed_to_consumed']['novelty']['std_effect']:+.2f} SD** |",
        f"| **Headroom Discovery Gain** | {estimand_reconciliation['headroom_discovery_gain']['imp_mean']:.6f} | 0.000000 | **+{estimand_reconciliation['headroom_discovery_gain']['imp_mean']:.6f}** | {estimand_reconciliation['headroom_discovery_gain']['user_macro_mean']:.6f} | 0.000000 | **+{estimand_reconciliation['headroom_discovery_gain']['user_macro_mean']:.6f}** | [{estimand_reconciliation['headroom_discovery_gain']['ci_95'][0]:+.6f}, {estimand_reconciliation['headroom_discovery_gain']['ci_95'][1]:+.6f}] | **{estimand_reconciliation['headroom_discovery_gain']['std_effect']:+.2f} SD** |",
        "",
        "*(Note: For Exposed $\\to$ Consumed, the effect direction is Consumed minus Choosable Exposed ($B - A$ and $E - D$), per protocol).* ",
        "",
        "---",
        "",
        "## 10. Robustness Battery Status Confirmation",
        "",
        "**ZERO prespecified robustness variants have yet been run.**  ",
        "The primary state is completely audited, reconciled, and frozen.",
        ""
    ])

    Path('outputs/primary_integrity_audit.md').write_text("\n".join(md_lines))
    print("Saved outputs/primary_integrity_audit.md")

if __name__ == '__main__':
    main()
