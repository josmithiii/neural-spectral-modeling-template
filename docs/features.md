# Project Features

## Overview

NSMT is a Lightning + Hydra template optimized for VIMH spectrogram datasets. It ships ready-to-run experiments, auto-configuring models, and evaluation tools for predicting audio synthesis parameters.

## Key Capabilities

- **VIMH dataset workflow**
  - Variable image dimensions with metadata-driven auto-configuration
  - Multihead labels spanning classification, ordinal, and regression outputs
  - Tooling for generation (`make gd*`), visualization (`make dd*`), and inspection (`make vd*`, `make vp*`)

- **Distance-aware loss stack**
  - Ordinal regression, quantized regression, and weighted cross-entropy variants
  - Automatic parameter range wiring from `vimh_dataset_info.json`

- **Model zoo**
  - CNN variants: `cnn_micro`, `cnn_tiny`, `cnn_64k` (+ auxiliary/regression/ordinal options)
  - ViT variants: `vit_micro`, `vit_tiny`
  - Auto head configuration through `VIMHLitModule`

- **Evaluation and analysis**
  - Audio reconstruction evaluator (`make ae`)
  - Architecture diagrams (`make td*`)
  - Parameter distribution reporting (chi-square heuristic via `make vp*`)

- **Developer experience**
  - Hydra config packs under `configs/`
  - Pytest suite with fast/slow markers (`make test`, `make test-all`)
  - Pre-commit formatting + lint (`make format`)

## Helpful Make Targets

- Datasets: `gds`, `gdl`, `gdmb`, `gdme`, `gdmr`, `gdas`
- Display datasets: `dds`, `ddl`, `ddr`
- Train: `tr`, `trq`, `trs`, `trl`, `ex`
- Wah reference experiments: `ewt`, `ewtr`, `evwt`, `evwtr`
- Diagrams: `td`, `tds`, `tdsa`, `tdsc`
- Utilities: `lc`, `tensorboard`, `format`, `test`, `test-all`, `verify-docs`, `ae`

## Common Patterns

```bash
# Quick sanity check (1 epoch)
python src/train.py trainer.max_epochs=1

# Preferred wah experiment (classification)
python src/train.py experiment=wah_cnn_tiny trainer=mps

# Switch to regression mode
python src/train.py experiment=wah_cnn_tiny_regression trainer=mps

# Override optimizer details
python src/train.py model.optimizer.lr=0.0005 trainer.gradient_clip_val=0.5
```

See also:

- [vimh.md](vimh.md) for dataset details
- [vimh_loss_functions.md](vimh_loss_functions.md) for loss design
- [architectures.md](architectures.md) for available model configs
- [experiments/wah.md](experiments/wah.md) for expected metrics and troubleshooting
