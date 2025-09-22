# Multihead Data Architecture

This repository uses a layered data stack so spectrogram datasets, metadata, and model wiring stay in sync. The focus is audio-oriented VIMH data, but the abstractions also cover historic CIFAR-style loaders.

## Layers at a Glance

```
┌─────────────────────────────────────────────┐
│ Application: VIMH workflow                  │
│ ├─ vimh_datamodule.py                       │
│ └─ vimh_dataset.py                          │
├─────────────────────────────────────────────┤
│ Foundation: multihead_dataset_base.py       │
│ Helpers: generic_multihead_dataset.py,      │
│          multihead_dataset.py (legacy)      │
└─────────────────────────────────────────────┘
```

### `vimh_dataset.py`

- Parses self-describing VIMH binaries or pickle caches.
- Returns tensors shaped `(channels, height, width)` and a dict of parameter values.
- Pulls parameter ranges, names, and audio settings from `vimh_dataset_info.json`.
- Supports Lightning transforms via `flexible_transforms.py`.

### `vimh_datamodule.py`

- Wraps the dataset with train/val/test splits, batch sizes, and num_workers settings.
- Exposes metadata helpers so `_configure_vimh_model_config` can adjust the model before instantiation.
- Handles `auto_configure_from_dataset` flags and optional preflight checks for label diversity.

### `multihead_dataset_base.py`

- Shared machinery for parsing the binary layout, validating metadata, and constructing heads.
- Used both by VIMH and historical multihead datasets.
- Provides convenience logic for checking class counts, dequantization, and shape inference.

### `generic_multihead_dataset.py`

- Retained for loading archived multihead sets (e.g., CIFAR-100-MH). The auto-detection routines still work but are not part of the main workflow.
- Useful if you ingest multihead data produced outside the NSMT generators.

### `multihead_dataset.py`

- Legacy strategy engine that synthesizes auxiliary labels for MNIST/CIFAR. Helpful for regression testing and demonstrations, but not used in the wah experiments.

## Why the Separation?

- **Metadata-first**: Models rely on dataset metadata for head counts and ranges; keeping parsing logic centralized avoids accidental drift.
- **Extensibility**: New audio datasets can inherit from `multihead_dataset_base.py` or reuse VIMH readers without duplicating validation logic.
- **Backward compatibility**: Older synthetic label strategies remain available for tests while the documentation focuses on the spectrogram workflow.

## Common Entry Points

- `src/data/vimh_datamodule.VIMHDataModule`: Lightning DataModule used by all experiments.
- `src/utils/vimh_utils.py`: Metadata utility helpers used in training auto-configuration and tests.
- `src/train._configure_vimh_model_config`: Pre-instantiation hook that enriches Hydra configs with heads, loss weights, and regression parameters.

Understanding these layers makes it easier to add new spectrogram generators, integrate external datasets, or debug head misconfiguration warnings.
