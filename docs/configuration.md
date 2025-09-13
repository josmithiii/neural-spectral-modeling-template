# Configuration System and Best Practices

## Overview

This project uses Hydra to compose configs for data, models, training, callbacks, and experiments. Below are the key patterns tailored to the VIMH workflow in this repo.

## Configuration Layout

```
configs/
├── train.yaml                 # Main training configuration
├── eval.yaml                  # Evaluation (metrics/checkpointing)
├── audio_eval.yaml            # Audio reconstruction eval config
├── data/                      # Data modules (e.g., vimh.yaml)
├── model/                     # Model configs (cnn_*, vit_*)
├── trainer/                   # Lightning trainer configs
├── callbacks/                 # Checkpointing, early stopping, progress bars
├── logger/                    # Logging backends
├── experiment/                # Complete, reproducible experiment bundles
├── mel/ | stft/ | synth/      # Signal processing & synthesis parameters
├── hydra/                     # Hydra-specific settings
└── local/                     # User-specific config (optional, gitignored)
```

### Composition via defaults

`configs/train.yaml` composes the run:

```yaml
# configs/train.yaml (excerpt)
defaults:
  - _self_
  - data: vimh
  - model: cnn_64k
  - callbacks: default
  - logger: tensorboard
  - trainer: default
  - paths: default
  - extras: default
  - hydra: default
  - experiment: null
  - hparams_search: null
  - optional local: default
  - debug: null
```

Override any of these at the CLI, e.g. `python src/train.py model=vit_tiny trainer=mps`.

## VIMH‑Aware Models

Model configs target `src.models.vimh_lit_module.VIMHLitModule`. Heads and loss parameters are auto‑configured from the dataset’s `vimh_dataset_info.json` at runtime.

Example (cnn_64k):

```yaml
_target_: src.models.vimh_lit_module.VIMHLitModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
  weight_decay: 0.0001

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 10

loss_weights: {}

net:
  _target_: src.models.components.simple_cnn.SimpleCNN
  input_channels: 1
  conv1_channels: 64
  conv2_channels: 128
  fc_hidden: 512
  heads_config: { synth_param1: 10 }  # placeholder; replaced by auto-config
  dropout: 0.5
  input_size: 32

compile: false

auto_configure_from_dataset: true
```

### Loss selection

- Ordinal: `model=cnn_64k_ordinal`
- Regression: `model=cnn_64k_regression`
- Auxiliary inputs: `model=cnn_64k_auxiliary`

See [vimh_loss_functions.md](vimh_loss_functions.md) for details.

## Preflight Label Validation

- Purpose: catch degenerate targets before training (e.g., uniform labels)
- Config (in `configs/train.yaml`):
  - `preflight.enabled` (default: true)
  - `preflight.label_diversity_batches` (default: 3)
- Overrides:
  - `python src/train.py preflight.enabled=false`
  - `python src/train.py preflight.label_diversity_batches=5`

## Experiments

Experiment configs fix data, model, trainer, and callbacks for reproducibility.

Example (`configs/experiment/example.yaml`):

```yaml
# @package _global_
defaults:
  - override /data: vimh
  - override /model: cnn_64k
  - override /callbacks: default
  - override /trainer: default

tags: ["vimh", "cnn"]
seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 0.002

data:
  batch_size: 64
```

Run with `python src/train.py experiment=example`.

## Command‑Line Overrides

```bash
# Hardware
python src/train.py trainer=mps           # Mac MPS

# Time and batch size
python src/train.py trainer.max_epochs=5 data.batch_size=32

# Switch architecture
python src/train.py model=vit_tiny | model=cnn_tiny

# Adjust optimizer
python src/train.py model.optimizer.lr=0.0005
```

## Debugging & Introspection

```bash
# Print resolved config
python src/train.py --cfg job

# Print specific section
python src/train.py --cfg job --package model

# Dry-run validation
python src/train.py --cfg job trainer.max_epochs=0
```

## Tips

- Prefer experiment configs for reproducibility and sharing
- Use tags to group related runs (see `configs/train.yaml` and logger configs)
- Keep dataset and model in sync: VIMH metadata drives model heads

See also:

- [architectures.md](architectures.md)
- [multihead_data_architecture.md](multihead_data_architecture.md)
