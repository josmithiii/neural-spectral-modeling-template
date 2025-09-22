# Quick Reference

Power-user cheat sheet for day-to-day work on NSMT.

## Make Targets (Essential)

```bash
# SETUP
make h            # Show all targets
make lc           # List configs grouped by Hydra package

# QUICK TESTS
make trq          # 1-epoch sanity train (default experiment wiring)

# TRAINING
make tr           # Train with defaults (see configs/train.yaml)
make trs          # Train on small VIMH dataset (auto-generates if missing)
make trl          # Train on large VIMH dataset (auto-generates if missing)

# EXPERIMENTS
make ex           # Baseline CNN experiment
make ewt          # Wah CNN tiny classification (production-ready)
make ewtr         # Wah CNN tiny regression (production-ready)

# DATASETS (synthesize & view)
make gds          # Small wah dataset (wraps gdws target)
make gdl          # Large wah dataset (wraps gdwl target)
make ddr|dds|ddl  # Display most-recent/small/large dataset samples
make vdr|vds|vdl  # Print dataset metadata
make vpr|vps|vpl  # Parameter distribution analysis

# UTILITIES
make test         # Fast pytest suite (excludes slow)
make test-all     # Full pytest suite
make format       # Run formatting+lint stack
make verify-docs  # Check Markdown links/headings/deprecated commands
make tensorboard  # Launch TensorBoard on localhost:6006
make td|tds|tdsa  # Generate architecture diagrams
make ae           # Audio reconstruction evaluation
```

## Architectures

| Name        | Type        | Params | Best For            | Config        |
| ----------- | ----------- | ------ | ------------------- | ------------- |
| CNN Micro   | CNN         | ~8K    | Quick prototyping   | `cnn_micro`   |
| CNN Tiny    | CNN         | ~8–64K | Compact baselines   | `cnn_tiny`    |
| CNN Medium  | CNN         | ~64K*  | Standard spectral   | `cnn_medium`  |
| ViT Micro   | Transformer | small  | Global correlations | `vit_micro`   |
| ViT Tiny    | Transformer | small  | Global correlations | `vit_tiny`    |

(*Parameter counts vary with multihead configuration.)

## Datasets

| Dataset Target | Size        | Description                            | Use Case          |
| -------------- | ----------- | -------------------------------------- | ----------------- |
| `make gds`     | 256 samples | Saw + wah, decay + pedal angle varied  | Smoke testing     |
| `make gdl`     | 16K samples | Saw + wah, decay + pedal angle varied  | Wah training      |

## Common Commands

```bash
# Basic training with MPS (Mac)
python src/train.py trainer=mps data.num_workers=0

# Train with a preconfigured experiment
python src/train.py experiment=wah_cnn_tiny

# Override parameters inline
python src/train.py trainer.max_epochs=20 data.batch_size=32

# Evaluate a specific checkpoint
python src/train.py test=true train=false ckpt_path=logs/train/runs/.../checkpoints/epoch_050.ckpt

# Generate VIMH dataset directly
python generate_vimh.py --config-name=synth/generate_saw_wah

# Audio evaluation (non-interactive batch)
python src/audio_reconstruction_eval.py interactive=false num_samples=10
```

## File Structure

```
├── src/
│   ├── train.py                     # Training entry point
│   ├── eval.py                      # Evaluation entry point
│   ├── audio_reconstruction_eval.py # Audio reconstruction eval tool
│   ├── models/                      # Lightning modules + components
│   ├── data/                        # VIMH data modules and loaders
│   └── utils/                       # Shared utilities
├── configs/                         # Hydra configs (data/model/trainer/...)
├── tests/                           # Pytest suite
├── generate_vimh.py                # Dataset generation CLI
└── data/                            # Generated datasets (gitignored)
```

## Expert Tips

- Prefer `trainer=mps` on Apple Silicon; set `data.num_workers=0` to avoid hangs.
- Run `make format` before commits to align with pre-commit hooks.
- Use `make vpr` to confirm parameter coverage after generating datasets.
- TensorBoard logs land in `logs/train/runs/`.
- `make h` surfaces new targets—check it after updates.
