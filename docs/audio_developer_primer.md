# Audio Developer Primer

This primer bridges the gap for audio software engineers who are new to modern ML workflows. It outlines the minimal signal-processing and machine-learning concepts used throughout the NSMT repository.

## Spectrogram Basics

- **Representation**: VIMH datasets store magnitude-style spectrograms. Height encodes frequency bins (typically 32–64), width encodes time frames, and additional channels hold alternate views (e.g., instantaneous frequency, modulation spectra).
- **Generation**: Datasets derive from STFT or mel front ends described in `configs/stft/` and `configs/mel/`. Inspect `vimh_dataset_info.json` for FFT size, hop length, and sampling rate to ensure your synthesis pipeline matches.
- **Quantization**: Parameter values are quantized to 8-bit codes that approximate perceptual just-noticeable differences. Maintain meaningful `min`, `max`, and `step` values when adding new parameters so loss functions remain well scaled.

## Hydra and Experiment Composition

- **Entry Points**: `src/train.py` and `src/eval.py` accept Hydra overrides (e.g., `python src/train.py model=cnn_64k data=vimh trainer=mps`).
- **Experiments**: YAML files in `configs/experiment/` bundle data, model, trainer, and callback settings for reproducible runs (`wah_cnn_tiny.yaml`, `wah_cnn_tiny_regression.yaml`).
- **Overrides**: Command-line overrides let you adjust epochs, learning rates, or datasets without touching files (`trainer.max_epochs=50`, `data.data_dir=...`).

## Model Outputs

- **Classification/Ordinal Heads**: For discrete quantized parameters, models use per-parameter classifiers or ordinal losses. Predictions report accuracy and perceptually scaled regression errors.
- **Regression Heads**: Direct regression is supported via `NormalizedRegressionLoss`, which interprets each head as a normalized scalar between 0 and 1.
- **Auto-Configuration**: `_configure_vimh_model_config` reads dataset metadata before instantiation to wire heads, loss weights, and parameter ranges—no manual head definitions required.

## Evaluation Workflow

- **Metric Reporting**: Per-head metrics (accuracy or MAE) log under `logs/train/runs/<timestamp>/`. Use TensorBoard or CLI summaries to monitor progress.
- **Audio Reconstruction**: `make ae` resynthesizes audio using ground-truth and predicted parameters. Listening critically helps catch failure modes that metrics miss.
- **Parameter Diagnostics**: `make vpr` surfaces distributions and uniformity checks to validate dataset generation; skewed ranges often signal config mistakes.

## Next Steps

- Start with `make ewt` and `make ewtr` to confirm your environment matches the reference results in [experiments/wah.md](experiments/wah.md).
- When extending datasets, copy an existing synth config under `configs/synth/` and update the audio + parameter metadata fields.
- For architecture experiments, begin with `model=cnn_tiny` or `model=vit_micro`, then iterate via Hydra overrides.

Keep this primer handy as you explore new signal chains or integrate NSMT components into larger audio applications.
