# Neural Network Architectures

## Overview

This repository ships practical CNN and ViT configurations for VIMH‑style multihead regression/classification. Architectures are selected via Hydra, e.g. `python src/train.py model=cnn_64k`.

## Supported Model Configs

| Family | Configs                                             | Notes                                   |
| ------ | ---------------------------------------------------- | --------------------------------------- |
| CNN    | `cnn_micro`, `cnn_tiny`, `cnn_64k`                  | Size variants; multihead by default      |
| CNN    | `cnn_64k_ordinal`, `cnn_64k_regression`, `cnn_64k_auxiliary` | Loss/feature variants |
| ViT    | `vit_micro`, `vit_tiny`                             | Lightweight Vision Transformers          |

Experimental components (ConvNeXtV2, EfficientNet) exist under `src/models/components/` but are not wired with default configs here.

## CNN (SimpleCNN)

- Input: typically 32×32×C spectrogram images (C=1 by default)
- Blocks: 2 conv stages → pooling → MLP head(s)
- Multihead: one head per parameter (auto‑configured from VIMH metadata)
- Auxiliary features: optional scalar inputs for fusion (`cnn_64k_auxiliary`)
- Param count: depends on heads and image size (see diagrams in `viz/`)

Usage examples:

```bash
python src/train.py model=cnn_micro           # tiny baseline
python src/train.py model=cnn_64k             # standard CNN
python src/train.py model=cnn_64k_ordinal     # distance‑aware loss
python src/train.py model=cnn_64k_regression  # direct regression
```

## Vision Transformer (ViT)

- Patch embedding over 2D inputs; lightweight configs for small images
- Heads: mapped from VIMH metadata via `VIMHLitModule`

Usage examples:

```bash
python src/train.py model=vit_micro
python src/train.py model=vit_tiny
```

## Adding a New Architecture

1. Implement a component under `src/models/components/`.
2. Create a model config in `configs/model/` pointing to your component and to `VIMHLitModule`.
3. Ensure `heads_config` is present as placeholder if you rely on auto‑configuration.

Minimal template:

```yaml
_target_: src.models.vimh_lit_module.VIMHLitModule
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 10
net:
  _target_: src.models.components.my_network.MyNetwork
  # your params here
auto_configure_from_dataset: true
```

## Tips

- Use `viz/simple_model_diagram.py` or `viz/enhanced_model_diagrams.py` to inspect shapes and parameter counts.
- For Mac, prefer `trainer=mps` and set `data.num_workers=0` for stability.
