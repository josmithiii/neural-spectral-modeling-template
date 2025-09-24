# Wah Experiments Reference

The wah pedal experiments are the first polished examples in NSMT. Use this guide to reproduce the results, interpret metrics, and troubleshoot issues.

## Prerequisites

- Dataset: `make gdl` (16K saw + wah, decay + pedal-angle variation). Use `make gds` for fast smoke tests.
- Hardware: Apple Silicon with MPS acceleration or a recent CUDA GPU. CPU works but takes much longer.
- Environment: virtualenv created via `sh setup.sh`; activate before running commands.

## Training Commands

```bash
# Classification heads (distance-aware ordinal losses)
make ewt

# Regression heads (NormalizedRegressionLoss per parameter)
make ewtr
```

Both targets regenerate the large dataset if missing, then run the corresponding Hydra experiment (`wah_cnn_tiny.yaml` or `wah_cnn_tiny_regression.yaml`). Logs and checkpoints appear under `logs/train/runs/<timestamp>/`.

## Expected Results (from [../experiments_overview.md](../experiments_overview.md))

| Experiment Name | Loss Type | Aggregate Metric | log10_decay_time | wah_position | Batch Size | Num Epochs | Runtime | Parameters |
|-----------------|-----------|------------------|------------------|----------------|------------|------------|---------|------------|
| wah_cnn_tiny.yaml | JND-weighted | 0.9451 | 0.8993 | 0.9908 | 64 | 200 | 58m25.354s | 39.9 K |
| wah_cnn_tiny_regression.yaml | MSE/MAE | 0.0175 | 0.0261 | 0.0090 | 128 | 200 | 6m17.804s | 1.1 M |

Use `tensorboard --logdir logs/train/runs/` or `make tensorboard` to confirm the learning curves plateau smoothly.

## Evaluation

After training, run:

```bash
make evwt    # Classification evaluation
make evwtr   # Regression evaluation
make ae      # Audio reconstruction comparison (optional but recommended)
```

Evaluation prints per-head metrics and writes audio comparisons to `audio_eval_results/`.

## Troubleshooting

- **Accuracy < 0.8 / MAE > 0.07**: Verify dataset metadata with `make vdl` and confirm the run picked up the large dataset. Small sets or stale checkpoints can degrade results.
- **Loss spikes mid-training**: Check for GPU/Metal warnings. If reproducible, reduce `trainer.max_epochs` to 60 and rerun.
- **Evaluation fails to load checkpoint**: Ensure `logs/train/runs/.../checkpoints/` contains `.ckpt` files; pass `CKPT=` to the make target if multiple runs exist.
- **Audio artifacts in evaluation**: Confirm STFT parameters match the dataset by inspecting `vimh_dataset_info.json` and `configs/audio_eval.yaml`.
z
