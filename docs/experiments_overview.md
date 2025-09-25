# Neural Spectral Modeling Template (NSMT) - Experiments Overview

This document summarizes the results of all experiments in the NSMT experiments directory `configs/experiment/`.

### All Template Experiments Run 9/25/25 1:20 PM After Commit e451bc2

| Experiment Name                | Loss Type    | Aggregate Metric | log10_decay_time | wah_position   | Batch Size | Num Epochs | Runtime    | Parameters |
|--------------------------------|--------------|------------------|------------------|----------------|------------|------------|------------|------------|
| example                        | JND-weighted | 0.3846           | 0.3269           | 0.4423         | 64         | 10         | 0m3.664s   | 1.1 M      |
| trivial_micro_small            | JND-weighted | 0.0769           | 0.0962           | 0.0577         | 32         | 5          | 0m3.365s   | 10.6 K     |
| trivial_micro_small_regression | MSE/MAE      | 6.5828           | 6.2463           | 6.9194         | 32         | 5          | 0m2.590s   | 9.6 K      |
| trivial_tiny_small             | JND-weighted | 0.2692           | 0.2500           | 0.2885         | 32         | 8          | 0m3.114s   | 39.9 K     |
| trivial_vit_micro_small        | JND-weighted | 0.0962           | 0.1538           | 0.0385         | 32         | 8          | 0m3.458s   | 23.0 K     |
| wah_cnn_tiny                   | JND-weighted | 0.9451           | 0.8993           | 0.9908         | 64         | 200        | 8m47.907s  | 39.9 K     |
| wah_cnn_tiny_ordinal           | JND-weighted | 0.9281           | 0.8657           | 0.9905         | 64         | 200        | 8m38.339s  | 39.9 K     |
| wah_cnn_tiny_quantized         | JND-weighted | 0.9281           | 0.8657           | 0.9905         | 64         | 200        | 8m56.311s  | 39.9 K     |
| wah_cnn_tiny_regression        | MSE/MAE      | 0.0176           | 0.0262           | 0.0090         | 128        | 200        | 4m6.557s   | 1.1 M      |
| wah_cnn_tiny_soft_target       | JND-weighted | 0.9281           | 0.8657           | 0.9905         | 64         | 200        | 8m59.354s  | 39.9 K     |
| wah_cnn_tiny_weighted          | JND-weighted | 0.9281           | 0.8657           | 0.9905         | 64         | 200        | 8m52.446s  | 39.9 K     |

CSV file written: experiment_results.csv

## Loss Calculation Notes

- Aggregate Metric is the mean of the reported test metrics for `log10_decay_time` and `wah_position` (falling back to `test/loss` when a head metric is missing).
- Per-head columns report the exact metric logged (accuracy for classification heads, MAE for regression heads); values are rounded to 4 decimals.
- Batch Size and Num Epochs are parsed from the Hydra data/trainer configuration lines; `Num Epochs` reflects the configured `max_epochs`.
- Runtime uses the shell `real` timer when present (falls back to log timestamps otherwise); Parameters come from the Lightning model summary output.

## Updating Notes

The table was created by
[../scripts/extract_logs.py](../scripts/extract_logs.py)
after running
[../scripts/run_all_experiments.sh](../scripts/run_all_experiments.sh)
(Run it with the `--csv [outpath]` option to write out a spreadsheet version.)
