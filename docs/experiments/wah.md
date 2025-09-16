# Wah Experiments Reference

The wah pedal experiments are the most polished workflows in NSMT. Use this guide to reproduce the published results, interpret metrics, and troubleshoot issues.

## Prerequisites

- Dataset: `make gdl` (16K saw + wah, decay + pedal-angle variation). Use `make gds` for fast smoke tests.
- Hardware: Apple Silicon with MPS acceleration or a recent CUDA GPU. CPU works but increases training time dramatically.
- Environment: virtualenv created via `sh setup.sh`; activate before running commands.

## Training Commands

```bash
# Classification heads (distance-aware ordinal losses)
make ewt

# Regression heads (NormalizedRegressionLoss per parameter)
make ewtr
```

Both targets regenerate the large dataset if missing, then run the corresponding Hydra experiment (`wah_cnn_tiny.yaml` or `wah_cnn_tiny_regression.yaml`). Logs and checkpoints appear under `logs/train/runs/<timestamp>/`.

## Expected Results

| Metric (validation)                    | Classification (`make ewt`) | Regression (`make ewtr`) |
| -------------------------------------- | --------------------------- | ------------------------- |
| Per-head accuracy                      | ≥ 0.85                      | —                         |
| Per-head mean absolute error (scaled)  | —                           | ≤ 0.05                    |
| Overall loss                           | 0.35–0.45                   | 0.03–0.05                |
| Training duration (M2 Max, MPS)        | ~10 minutes                 | ~5 minutes                |

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

## Sharing Results

Include in pull requests:

- Hydra command used (`python src/train.py experiment=wah_cnn_tiny trainer=mps`, etc.).
- Snapshot of validation metrics (`test/log10_decay_time_acc`, `test/wah_position_mae`).
- Link to the run directory inside `logs/train/runs/`.
- Optional audio examples from `audio_eval_results/`.

This guidance targets experienced audio developers who need confidence that the baseline wah modeling tasks are performing as expected before branching into new architectures or datasets.

### Sample Log Snippet

```text
Epoch 68 | test/log10_decay_time_acc=0.91 | test/wah_position_acc=0.88
Epoch 68 | val/loss=0.38 | val/avg_acc=0.90
Epoch 50 | test/log10_decay_time_mae=0.037 | test/wah_position_mae=0.041
```
