# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an extended Lightning-Hydra-Template for deep learning projects using PyTorch Lightning and Hydra configuration management. The template provides a clean, organized structure for rapid ML experimentation with minimal boilerplate.  In this template project we've extended the original template as described in ./README-CONFIG.md

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
```

## Troubleshooting Notes
- When you see "No module named 'rootutils'", it means we need to say `source .venv/bin/activate`

## Architecture Overview

### Configuration System (Hydra)
- **Main configs**: `configs/train.yaml` and `configs/eval.yaml` define default training/evaluation settings
- **Modular configs**: Organized by component type in `configs/` subdirectories:
  - `data/`: Data module configurations
  - `model/`: Model configurations  
  - `trainer/`: Lightning trainer configurations
  - `callbacks/`: Training callbacks
  - `logger/`: Logging configurations
  - `experiment/`: Complete experiment configurations
- **Config composition**: Uses Hydra's `defaults` list to compose configurations
- **Override system**: Parameters can be overridden via command line (e.g., `python src/train.py trainer.max_epochs=20`)

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
- **Core ML**: `torch`, `lightning`, `torchmetrics`, `torchvision`
- **Config**: `hydra-core`, `hydra-colorlog`, `hydra-optuna-sweeper`
- **Utilities**: `rootutils` (project root setup), `rich` (terminal formatting)
- **Development**: `pre-commit`, `pytest`, `black`, `isort`, `flake8`

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
- Makefile provides convenient shortcuts for common tasks
- MPS (Metal Performance Shaders) support for Mac training available and nearly always used by user
