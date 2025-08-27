# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an extended Lightning-Hydra-Template for deep learning projects using PyTorch Lightning and Hydra configuration management. The template provides a clean, organized structure for rapid ML experimentation with minimal boilerplate.  In this template project we've extended the original template as described in ./docs/extensions.md

## Core Technologies

- **PyTorch Lightning**: High-level PyTorch wrapper for organizing model training
- **Hydra**: Configuration management framework for complex applications
- **Python 3.8+**: Main programming language
- **pytest**: Testing framework

## Common Commands

### Training
```bash
# Basic training
python src/train.py

# Train with specific trainer (CPU/GPU/MPS)
python src/train.py trainer=cpu
python src/train.py trainer=gpu
python src/train.py trainer=mps

# Train with experiment config
python src/train.py experiment=example

# Makefile shortcuts
make train          # or make t - basic training
make trainmps       # or make tmps - train with MPS on Mac
make help           # generate the list of make targets
```

### Testing
```bash
# Run tests (excluding slow tests)
pytest -k "not slow"
make test

# Run all tests
pytest
make test-all

# Run specific test file
pytest tests/test_train.py
```

### Code Quality
```bash
# Run pre-commit hooks (formatting, linting)
pre-commit run -a
make format
```

### Evaluation
```bash
# Evaluate checkpoint
python src/eval.py ckpt_path="/path/to/checkpoint.ckpt"
```

### Environment Management
```bash
source .venv/bin/activate.csh

# Alternative activation shortcuts (see Makefile)
make activate    # Shows alias setup for 'a' command
make deactivate  # Shows alias setup for 'd' command
```

### Visualization and Analysis
```bash
# Generate model architecture diagrams
make td          # Text + graphical diagrams
make tda         # All model architectures
make tds         # Simple text-only diagrams  
make tdss        # Sample architectures comparison

# Compare architectures systematically (3 epochs each)
make ca          # Compare medium-sized architectures
```

### Extended Training Commands
```bash
# Quick training (1 epoch, limited batches)
make tq          # SimpleDenseNet quick
make tqc         # CNN quick
make tqv         # ViT quick
make tqcn        # ConvNeXt quick
make tqa         # All architectures quick

# Specific architecture training
make trc         # Train CNN (SimpleCNN)
make trvs        # Train small ViT (~38K params)
make trcns       # Train ConvNeXt small (~68K params)
```

### VIMH (Variable Image MultiHead) Training
```bash
# VIMH dataset experiments
make evimh       # VIMH CNN 16K dataset samples
make evimho      # VIMH ordinal regression
make evimhr      # VIMH pure regression heads

# Direct VIMH training
python src/train.py experiment=vimh_cnn_16kdss
python examples/vimh_training.py --demo --save-plots
```

### Environment Management
```bash
source .venv/bin/activate.csh
```

## Troubleshooting Notes
- When you see "No module named 'rootutils'", it means we need to say `source .venv/bin/activate`
- Use MPS trainer for Mac: `python src/train.py trainer=mps` (nearly always used by user)
- For VIMH training, set `num_workers: 0` in data config as MPS doesn't support multiprocessing

## Architecture Overview

### Extended Template Features
This is an **extended** Lightning-Hydra-Template with major enhancements:
- **Multiple architectures**: SimpleDenseNet, SimpleCNN, ConvNeXt-V2, ViT, EfficientNet
- **VIMH (Variable Image MultiHead)**: Advanced multihead dataset format with auto-configuration
- **Configurable losses**: Hydra-managed loss functions, no hardcoding
- **50+ make targets**: Convenient shortcuts with abbreviations
- **Backward compatibility**: All original template functionality preserved

### Configuration System (Hydra)
- **Main configs**: `configs/train.yaml` and `configs/eval.yaml` define default training/evaluation settings
- **Modular configs**: Organized by component type in `configs/` subdirectories:
  - `data/`: Data module configurations (MNIST, VIMH)
  - `model/`: Model configurations (multiple architectures with parameter variants)
  - `trainer/`: Lightning trainer configurations (CPU, GPU, MPS, DDP)
  - `callbacks/`: Training callbacks
  - `logger/`: Logging configurations
  - `experiment/`: Complete experiment configurations (50+ experiments)
- **Config composition**: Uses Hydra's `defaults` list to compose configurations
- **Override system**: Parameters can be overridden via command line (e.g., `python src/train.py trainer.max_epochs=20`)
- **Auto-configuration**: VIMH models auto-configure from dataset metadata

### Code Structure
- **`src/train.py`**: Main training entry point using Hydra configuration
- **`src/eval.py`**: Evaluation entry point for trained models
- **`src/models/`**: Lightning modules (model implementations)
- **`src/data/`**: Lightning data modules (data loading/preprocessing)
- **`src/utils/`**: Utility functions for logging, instantiation, etc.
- **Dynamic instantiation**: Uses `hydra.utils.instantiate()` to create objects from config `_target_` paths

### Key Patterns
- **Lightning modules**: Follow standard PyTorch Lightning patterns with `training_step`, `validation_step`, `test_step`, `configure_optimizers`
- **Data modules**: Implement `prepare_data`, `setup`, and dataloader methods
- **Metric tracking**: Uses torchmetrics for proper metric calculation across devices
- **Hyperparameter logging**: All init parameters automatically saved via `self.save_hyperparameters()`

### Project Structure
```
├── configs/              # Hydra configuration files
├── src/
│   ├── train.py         # Main training script
│   ├── eval.py          # Evaluation script
│   ├── models/          # Lightning modules
│   ├── data/            # Data modules
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── logs/                # Training logs and checkpoints
└── data/                # Dataset storage
```

### Dependencies and Tools
- **Core ML**: `torch>=2.0.0`, `lightning>=2.0.0`, `torchmetrics>=0.11.4`, `torchvision>=0.15.0`
- **Config**: `hydra-core==1.3.2`, `hydra-colorlog==1.2.0`, `hydra-optuna-sweeper==1.2.0`
- **Utilities**: `rootutils` (project root setup), `rich` (terminal formatting)
- **Development**: `pre-commit`, `pytest`
- **Visualization**: `torchview` (model visualization), `torchviz` (computational graph)
- **Optional Loggers**: `wandb`, `neptune-client`, `mlflow`, `comet-ml`, `aim>=3.16.2`

## Development Guidelines

- **Config-driven development**: Add new models/data by creating config files, not code changes
- **Modular design**: Keep components (models, data, callbacks) independent and configurable
- **Lightning conventions**: Follow PyTorch Lightning style guide and method ordering
- **Type hints**: Use Python type hints throughout the codebase
- **Metric naming**: Use `/` in metric names for logger organization (e.g., `train/loss`, `val/acc`)

## User-Specific Notes

- User prefers `uv` for Python environment management
- User works with signal processing and sound synthesis but is new to ML implementation details
- User values fast iteration and minimal boilerplate for research
- Makefile provides convenient shortcuts for common tasks with meaningful abbreviations
- MPS (Metal Performance Shaders) support for Mac training available and nearly always used by user

## Important Development Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User
