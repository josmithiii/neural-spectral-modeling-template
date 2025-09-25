# Neural Spectral Modeling Template (NSMT) - Experiments Overview

This document summarizes the results of all experiments in the NSMT experiments directory `configs/experiment/`.

### Template Experiments (Essential Set)

| Experiment Name | Loss Type | Aggregate Metric | log10_decay_time | wah_position | Batch Size | Num Epochs | Runtime | Parameters |
|-----------------|-----------|------------------|------------------|--------------|------------|------------|---------|------------|
| example | JND-weighted | 0.3846 | 0.3269 | 0.4423 | 64 | 10 | 0m3.515s | 1.1 M |
| trivial_micro_small | JND-weighted | 0.0769 | 0.0962 | 0.0577 | 32 | 5 | 0m2.928s | 10.6 K |
| trivial_micro_small_regression | MSE/MAE | 6.5828 | 6.2463 | 6.9194 | 32 | 5 | 0m2.627s | 9.6 K |
| trivial_tiny_small | JND-weighted | 0.2692 | 0.2500 | 0.2885 | 32 | 8 | 0m3.300s | 39.9 K |
| trivial_vit_micro_small | JND-weighted | 0.0962 | 0.1538 | 0.0385 | 32 | 8 | 0m3.389s | 23.0 K |
| wah_cnn_tiny | JND-weighted | 0.9451 | 0.8993 | 0.9908 | 64 | 200 | 8m36.196s | 39.9 K |
| wah_cnn_tiny_regression | MSE/MAE | 0.0175 | 0.0261 | 0.0090 | 128 | 200 | 4m37.751s | 1.1 M |

### Current Experiments 9/24/25

```
  example.yaml
  trivial_micro_small_regression.yaml
  trivial_micro_small.yaml
  trivial_tiny_small.yaml
  trivial_vit_micro_small.yaml
  wah_cnn_tiny_ordinal.yaml
  wah_cnn_tiny_quantized.yaml
  wah_cnn_tiny_regression.yaml
  wah_cnn_tiny_soft_target.yaml
  wah_cnn_tiny_weighted.yaml
  wah_cnn_tiny.yaml
```
---

## Loss Calculation Notes

- Aggregate Metric is the mean of the reported test metrics for `log10_decay_time` and `wah_position` (falling back to `test/loss` when a head metric is missing).
- Values in the `log10_decay_time` and `wah_position` columns are the exact test metrics logged for each head (accuracy for classification heads, MAE for regression heads) rounded to four decimals.
- Batch Size and Num Epochs are parsed from the Hydra data/trainer configuration lines; `Num Epochs` reflects the configured `max_epochs`.
- Runtime uses the shell `real` timer when available (falling back to log timestamp deltas); Parameters are taken from the Lightning model summary output.

## Updating Notes

The table was created by
[../scripts/extract_logs.py](../scripts/extract_logs.py)
after running
[../scripts/run_all_experiments.sh](../scripts/run_all_experiments.sh)
(Run it with the `--csv [outpath]` option to write out a spreadsheet version.)
