# Project Features

## Overview

This repository is a practical template for neural spectral modeling built on Lightning + Hydra. It centers on the VIMH dataset format and provides ready‑to‑run experiments, models, losses, and tools for audio‑parameter prediction from images.

## Key Capabilities

- VIMH dataset support
  - Variable image size and channels with embedded metadata
  - Multihead labels for 1–255 quantized continuous parameters
  - Auto‑configuration of heads and ranges from `vimh_dataset_info.json`

- Distance‑aware loss functions (for quantized targets)
  - OrdinalRegressionLoss: continuous, distance‑aware predictions
  - QuantizedRegressionLoss: direct regression on quantized values
  - WeightedCrossEntropyLoss: classification with distance weighting

- Ready model configs
  - CNN families: `cnn_micro`, `cnn_tiny`, `cnn_64k` (+ auxiliary/regression/ordinal variants)
  - ViT families: `vit_micro`, `vit_tiny`
  - Auto‑head wiring via `VIMHLitModule` with dataset introspection

- Dataset generation utilities
  - `generate_vimh.py` for small/large synthetic sets (Saw + Moog VCF)
  - Make targets for small (`sds`) and large (`sdl`) datasets and Moog variants

- Evaluation and analysis
  - Audio reconstruction evaluation (`src/audio_reconstruction_eval.py`)
  - Model diagram generation (`viz/` make targets)
  - TensorBoard logging via `make tensorboard`

- Developer experience
  - Hydra config packs under `configs/`
  - Pytest test suite with fast and slow markers
  - Pre‑commit formatting and linting via `make format`

## Helpful Make Targets

- Datasets: `sds`, `sdl`, `sdmb`, `sdme`, `sdmr`, `sdma`
- Display datasets: `dds`, `ddl`, `ddr`
- Train: `tr` (defaults), `trq` (quick), `trs` (small), `trl` (large), `ex` (example)
- Experiments: `emb`, `eme`, `emr`, `emvit*`
- Diagrams: `td`, `tds`, `tdsa`, `tdv`
- Eval audio: `ae`
- Utilities: `lc`, `tensorboard`, `format`, `test`, `test-all`

## Common Patterns

```bash
# Quick sanity check (1 epoch)
python src/train.py trainer.max_epochs=1

# Example experiment (uses VIMH + CNN 64k by default)
python src/train.py experiment=example

# Choose model/data/trainer explicitly
python src/train.py model=cnn_64k data=vimh trainer=mps

# Switch loss style (ordinal or regression variants)
python src/train.py model=cnn_64k_ordinal
python src/train.py model=cnn_64k_regression
```

See also:

- [vimh.md](vimh.md) for dataset details
- [vimh_loss_functions.md](vimh_loss_functions.md) for loss design
- [architectures.md](architectures.md) for available model configs
