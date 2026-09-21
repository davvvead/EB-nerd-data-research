# Final EB-NeRD Pre-Outcome Gate

Run this **before** any Phase 1 outcome analysis.

It compares the two downloaded embedding artifacts and reports the diagnostics
needed to freeze the final open parameters.

## Command

From the folder shown in your screenshot:

```bash
python final_preoutcome_gate.py \
  --articles ebnerd_small/articles.parquet \
  --behaviors ebnerd_small/train/behaviors.parquet \
  --history ebnerd_small/train/history.parquet \
  --bert bert_base_multilingual_cased.parquet \
  --contrastive contrastive_vector.parquet \
  --outdir final_preoutcome_gate_output
```

## Outputs

Upload these back to ChatGPT:

```text
final_preoutcome_gate_output/final_preoutcome_report.json
final_preoutcome_gate_output/history_user_diagnostics.csv
```

## Important

This script does not touch validation behaviors and does not compute any
competition outcome. After reviewing its report, we can freeze the remaining
parameters and then run the actual held-out Phase 1 analysis.
