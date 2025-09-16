# Audio Reconstruction Evaluation

Use the audio reconstruction evaluator to hear how well a trained model predicts synthesis parameters. It re-synthesizes audio from predicted parameters, compares against ground truth, and exports plots plus WAV files.

## Quick Start

```bash
python src/audio_reconstruction_eval.py
```

This command:
- Locates the latest checkpoint under `logs/train/`.
- Launches an interactive widget (in Jupyter/GUI environments) for browsing samples.
- Displays waveforms, spectrograms, parameter traces, and summary metrics.
- Saves audio comparisons to `audio_eval_results/` by default.

## Alternative Usage

```bash
python src/audio_reconstruction_eval.py interactive=false             # Batch mode
python src/audio_reconstruction_eval.py ckpt_path=path/to/epoch.ckpt   # Explicit checkpoint
python src/audio_reconstruction_eval.py data=vimh_32x32x1_8000Hz       # Custom dataset config
python src/audio_reconstruction_eval.py save_audio=true output_dir=out # Custom result dir
```

All options mirror `configs/audio_eval.yaml` and can be overridden on the CLI.

## What to Inspect

For each sample the evaluator:
1. Runs inference on the spectrogram input.
2. Reconstructs audio using true and predicted parameters.
3. Calculates metrics (per-parameter error, waveform SNR, correlation, RMSE).
4. Renders plots showing parameter trajectories and spectral differences.
5. Writes WAV files so you can perform listening tests.

Key indicators:
- **Classification runs (`make ewt`)**: look for per-head accuracy ≥ 0.85.
- **Regression runs (`make ewtr`)**: mean absolute error below 0.05 on normalized scales is a strong baseline.
- **Audio comparisons**: audible differences should be subtle; large spectral discrepancies usually indicate data/model mismatch.

## Troubleshooting

- No checkpoint found → supply `ckpt_path=` manually or confirm training produced `logs/train/runs/...`.
- Widget fails to open → use `interactive=false` to run in CLI-only environments.
- Audio sounds aliased → ensure dataset sample rate in metadata matches your playback and generation settings.

See [experiments/wah.md](experiments/wah.md) for reference metrics and log excerpts from healthy runs.
