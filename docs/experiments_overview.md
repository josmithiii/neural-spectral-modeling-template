# Neural Spectral Modeling Template (NSMT) - Experiments Overview

This document summarizes the results of all experiments in the NSMT experiments directory `configs/experiment/`.

## Experiment Summary Table

| Experiment Name | Loss Type | Aggregate Metric | log10_decay_time | wah_position | Batch Size | Num Epochs | Runtime | Parameters |
|-----------------|-----------|------------------|------------------|----------------|------------|------------|---------|------------|
| auxiliary_cnn_16kdss | JND-weighted | 0.5481 | 0.3846 | 0.7115 | 128 | 30 | 0m5.843s | 1.1 M |
| auxiliary_cnn_16kdss.yaml | JND-weighted | 0.3942 | 0.3846 | 0.4038 | 128 | 30 | 0m11.208s | 1.1 M |
| auxiliary_cnn_256dss | JND-weighted | 0.6346 | 0.5192 | 0.7500 | 32 | 50 | 0m10.514s | 1.1 M |
| auxiliary_cnn_256dss.yaml | JND-weighted | 0.4904 | 0.4615 | 0.5192 | 32 | 50 | 0m16.181s | 1.1 M |
| ewtr | MSE/MAE | 0.0212 | 0.0309 | 0.0115 | 128 | 200 | 4m38.106s | 1.1 M |
| example | JND-weighted | 0.4519 | 0.2885 | 0.6154 | 64 | 10 | 0m4.085s | 1.1 M |
| example.yaml | JND-weighted | 0.3846 | 0.3269 | 0.4423 | 64 | 10 | 0m7.035s | 1.1 M |
| trivial_medium_large.yaml | JND-weighted | 0.8486 | 0.7553 | 0.9420 | 128 | 5 | 0m15.032s | 1.1 M |
| trivial_micro_large.yaml | JND-weighted | 0.7258 | 0.5481 | 0.9036 | 64 | 10 | 0m34.155s | 10.6 K |
| trivial_micro_large_regression.yaml | MSE/MAE | 7.2604 | 6.3027 | 8.2181 | 64 | 10 | 0m24.206s | 9.6 K |
| trivial_micro_small.yaml | JND-weighted | 0.0769 | 0.0962 | 0.0577 | 32 | 5 | 0m6.730s | 10.6 K |
| trivial_micro_small_regression.yaml | MSE/MAE | 6.5828 | 6.2463 | 6.9194 | 32 | 5 | 0m7.101s | 9.6 K |
| trivial_tiny_large.yaml | JND-weighted | 0.8261 | 0.7641 | 0.8880 | 64 | 15 | 0m46.633s | 39.9 K |
| trivial_tiny_small.yaml | JND-weighted | 0.2692 | 0.2500 | 0.2885 | 32 | 8 | 0m6.914s | 39.9 K |
| trivial_vit_micro_large.yaml | JND-weighted | 0.7943 | 0.7968 | 0.7919 | 64 | 15 | 1m12.185s | 23.0 K |
| trivial_vit_micro_small.yaml | JND-weighted | 0.0962 | 0.1538 | 0.0385 | 32 | 8 | 0m7.469s | 23.0 K |
| trivial_vit_micro_small_regression.yaml | MSE/MAE | 6.6071 | 6.2703 | 6.9438 | 32 | 8 | 0m9.208s | 22.0 K |
| trivial_vit_tiny_large.yaml | JND-weighted | 0.7710 | 0.7458 | 0.7962 | 64 | 20 | 1m55.341s | 116 K |
| trivial_vit_tiny_small.yaml | JND-weighted | 0.0962 | 0.1154 | 0.0769 | 32 | 10 | 0m8.338s | 116 K |
| wah_cnn.yaml | JND-weighted | 0.9843 | 0.9750 | 0.9936 | 64 | 500 | 227m26.072s | 1.1 M |
| wah_cnn_large.yaml | JND-weighted | 0.9843 | 0.9750 | 0.9936 | 64 | 500 | 23m39.558s | 1.1 M |
| wah_cnn_large_aux.yaml | JND-weighted | 0.9843 | 0.9750 | 0.9936 | 64 | 500 | 33m51.714s | 1.1 M |
| wah_cnn_tiny.yaml | JND-weighted | 0.9451 | 0.8993 | 0.9908 | 64 | 200 | 58m25.354s | 39.9 K |
| wah_cnn_tiny_aux.yaml | JND-weighted | 0.9451 | 0.8993 | 0.9908 | 64 | 200 | 17m58.590s | 39.9 K |
| wah_cnn_tiny_regression.yaml | MSE/MAE | 0.0175 | 0.0261 | 0.0090 | 128 | 200 | 6m17.804s | 1.1 M |

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
