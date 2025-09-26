# Neural Spectral Modeling Template (NSMT) - Experiments Overview

This document summarizes the results of all experiments in the NSMT experiments directory `configs/experiment/`.

## Updating Notes

The table was created by
[../scripts/extract_logs.py](../scripts/extract_logs.py)
after running
[../scripts/run_all_experiments.sh](../scripts/run_all_experiments.sh)
(Run it with the `--csv [outpath]` option to write out a spreadsheet version.)

### All Template Experiments Run 9/25/25 2:30 AM After Commit 923b231

| Experiment Name                     | Loss Type             | Aggregate Metric | log10_decay_time | wah_position   | Batch Size | Num Epochs | Runtime      | Parameters |
|-------------------------------------|----------------------|------------------|------------------|----------------|------------|------------|--------------|------------|
| example                             | cross_entropy        | 0.3846↑          | 0.3269↑          | 0.4423↑        | 64         | 10         | 0m3.664s     | 1.1 M      |
| trivial_micro_small                 | cross_entropy        | 0.0769↑          | 0.0962↑          | 0.0577↑        | 32         | 5          | 0m3.365s     | 10.6 K     |
| trivial_micro_small_regression      | normalized_regression| 6.5828↓          | 6.2463↓          | 6.9194↓        | 32         | 5          | 0m2.590s     | 9.6 K      |
| trivial_tiny_small                  | cross_entropy        | 0.2692↑          | 0.2500↑          | 0.2885↑        | 32         | 8          | 0m3.114s     | 39.9 K     |
| trivial_vit_micro_small             | cross_entropy        | 0.0962↑          | 0.1538↑          | 0.0385↑        | 32         | 8          | 0m3.458s     | 23.0 K     |
| wah_cnn_medium                      | unknown              | 0.9483↑          | 0.9081↑          | 0.9884↑        | 64         | 201        | 11m46s       | 1.1 M      |
| wah_cnn_medium_auxiliary            | unknown              | 0.9094↑          | 0.8255↑          | 0.9933↑        | 64         | 201        | 13m50.537s   | 1.1 M      |
| wah_cnn_medium_auxiliary_regression | normalized_regression| 0.0183↓          | 0.0277↓          | 0.0088↓        | 128        | 201        | 8m42.861s    | 1.1 M      |
| wah_cnn_medium_regression           | unknown              | 0.0182↓          | 0.0265↓          | 0.0099↓        | 128        | 198        | 10m18.015s   | 1.1 M      |
| wah_cnn_tiny                        | cross_entropy        | 0.9451↑          | 0.8993↑          | 0.9908↑        | 64         | 201        | 10m49.095s   | 39.9 K     |
| wah_cnn_tiny_auxiliary              | cross_entropy        | 0.9423↑          | 0.8935↑          | 0.9912↑        | 64         | 201        | 13m59.264s   | 42.0 K     |
| wah_cnn_tiny_auxiliary_regression   | normalized_regression| 0.0237↓          | 0.0308↓          | 0.0167↓        | 128        | 115        | 4m26.397s    | 39.0 K     |
| wah_cnn_tiny_ordinal                | ordinal_regression   | 0.0008↑          | 0.0003↑          | 0.0012↑        | 64         | 197        | 18m1.756s    | 39.9 K     |
| wah_cnn_tiny_quantized              | quantized_regression | 7.2608↓          | 6.3026↓          | 8.2190↓        | 64         | 115        | 2m49.203s    | 37.9 K     |
| wah_cnn_tiny_regression             | normalized_regression| 0.0191↓          | 0.0268↓          | 0.0115↓        | 128        | 201        | 7m31.150s    | 37.9 K     |
| wah_cnn_tiny_soft_target            | soft_target          | 0.8721↑          | 0.8007↑          | 0.9435↑        | 64         | 187        | 137m8.290s   | 39.9 K     |
| wah_cnn_tiny_weighted               | weighted_cross_entropy| 0.9364↑          | 0.8901↑          | 0.9826↑        | 64         | 201        | 10m4.668s    | 39.9 K     |
| wah_vit_medium                      | unknown              | 0.9353↑          | 0.8962↑          | 0.9744↑        | 32         | 154        | 42m30.171s   | 6.5 M      |
| wah_vit_medium_regression           | unknown              | 0.0179↓          | 0.0239↓          | 0.0119↓        | 32         | 163        | 39m10.322s   | 6.5 M      |
| wah_vit_tiny                        | unknown              | 0.8433↑          | 0.8523↑          | 0.8343↑        | 64         | 134        | 8m1.620s     | 116 K      |
| wah_vit_tiny_regression             | unknown              | 0.0259↓          | 0.0320↓          | 0.0198↓        | 64         | 120        | 5m43.606s    | 114 K      |

Notes:
- Loss Type shows the configured loss function from model config (e.g., cross_entropy, normalized_regression, ordinal).
- Classification models (cross_entropy, ordinal) use JND-weighted accuracy metrics; regression models use MSE/MAE loss functions.
- Arrows indicate optimization direction: ↑ for higher-is-better (accuracies), ↓ for lower-is-better (losses/errors).
- Aggregate Metric is the mean of the available per-head test metrics for log10_decay_time and wah_position (falls back to test/loss when heads are missing).
- Values marked with * indicate fallback to test/loss due to missing head metrics.
- Per-head columns report the exact metric logged (accuracy for classification heads, MAE for regression heads); values are rounded to 4 decimals.
- Batch Size is parsed from the Hydra data configuration line.
- Num Epochs shows actual epochs completed when available (from training completion log), otherwise falls back to configured max_epochs.
- Runtime uses the shell `real` timer when present (falls back to log timestamps otherwise); Parameters come from the Lightning model summary output.

CSV file written: experiment_results.csv

### All Template Experiments Run 9/25/25 ~3 PM After Commit 4d3ac44

| Experiment Name                | Loss Type             | Aggregate Metric | log10_decay_time | wah_position   | Batch Size | Num Epochs | Runtime      | Parameters |
|--------------------------------|-----------------------|------------------|------------------|----------------|------------|------------|--------------|------------|
| example                        | cross_entropy         | 0.3846↑          | 0.3269↑          | 0.4423↑        | 64         | 10         | 0m3.664s     | 1.1 M      |
| trivial_micro_small            | cross_entropy         | 0.0769↑          | 0.0962↑          | 0.0577↑        | 32         | 5          | 0m3.365s     | 10.6 K     |
| trivial_micro_small_regression | normalized_regression | 6.5828↓          | 6.2463↓          | 6.9194↓        | 32         | 5          | 0m2.590s     | 9.6 K      |
| trivial_tiny_small             | cross_entropy         | 0.2692↑          | 0.2500↑          | 0.2885↑        | 32         | 8          | 0m3.114s     | 39.9 K     |
| trivial_vit_micro_small        | cross_entropy         | 0.0962↑          | 0.1538↑          | 0.0385↑        | 32         | 8          | 0m3.458s     | 23.0 K     |
| wah_cnn_medium                 | unknown               | 0.9483↑          | 0.9081↑          | 0.9884↑        | 64         | 201        | 11m46s       | 1.1 M      |
| wah_cnn_medium_regression      | unknown               | 0.0182↓          | 0.0265↓          | 0.0099↓        | 128        | 198        | 10m18.015s   | 1.1 M      |
| wah_cnn_tiny                   | cross_entropy         | 0.9451↑          | 0.8993↑          | 0.9908↑        | 64         | 201        | 10m49.095s   | 39.9 K     |
| wah_cnn_tiny_ordinal           | ordinal_regression    | 0.0008↑          | 0.0003↑          | 0.0012↑        | 64         | 197        | 18m1.756s    | 39.9 K     |
| wah_cnn_tiny_quantized         | quantized_regression  | 7.2608↓          | 6.3026↓          | 8.2190↓        | 64         | 115        | 2m49.203s    | 37.9 K     |
| wah_cnn_tiny_regression        | normalized_regression | 0.0191↓          | 0.0268↓          | 0.0115↓        | 128        | 201        | 7m31.150s    | 37.9 K     |
| wah_cnn_tiny_soft_target       | soft_target           | 0.8721↑          | 0.8007↑          | 0.9435↑        | 64         | 187        | 137m8.290s   | 39.9 K     |
| wah_cnn_tiny_weighted          | weighted_cross_entropy| 0.9364↑          | 0.8901↑          | 0.9826↑        | 64         | 201        | 10m4.668s    | 39.9 K     |
| wah_vit_medium                 | unknown               | 0.9353↑          | 0.8962↑          | 0.9744↑        | 32         | 154        | 42m30.171s   | 6.5 M      |
| wah_vit_medium_regression      | unknown               | 0.0179↓          | 0.0239↓          | 0.0119↓        | 32         | 163        | 39m10.322s   | 6.5 M      |
| wah_vit_tiny                   | unknown               | 0.8433↑          | 0.8523↑          | 0.8343↑        | 64         | 134        | 8m1.620s     | 116 K      |
| wah_vit_tiny_regression        | unknown               | 0.0259↓          | 0.0320↓          | 0.0198↓        | 64         | 120        | 5m43.606s    | 114 K      |

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

