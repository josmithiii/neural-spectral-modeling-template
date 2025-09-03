# Configuration System and Best Practices

## Overview

The Lightning-Hydra-Template-Extended uses Hydra for configuration management, enabling flexible, reproducible, and modular experiment setup. This guide covers configuration patterns, best practices, and the experiment system.

## 🎛️ Configuration Architecture

### Hydra Configuration Hierarchy
```
configs/
├── train.yaml                 # Main training configuration
├── eval.yaml                  # Main evaluation configuration
├── data/                      # Data module configurations
├── model/                     # Model configurations
├── trainer/                   # Lightning trainer configurations
├── callbacks/                 # Training callbacks
├── logger/                    # Logging configurations
├── experiment/                # Complete experiment configurations
├── hydra/                     # Hydra-specific settings
└── local/                     # User-specific configurations (not versioned)
```

### Configuration Composition
Hydra composes configurations through the `defaults` list:

```yaml
# configs/train.yaml
defaults:
  - _self_
  - data: mnist.yaml
  - model: mnist.yaml
  - callbacks: default.yaml
  - logger: null
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml
  - experiment: null
  - hparams_search: null
  - optional local: default.yaml
  - debug: null
```

## 🔧 Enhanced Configuration Features

### 1. Configurable Loss Functions

**Before** (hardcoded):
```python
# In model code
self.criterion = torch.nn.CrossEntropyLoss()
```

**Now** (configurable):
```yaml
# configs/model/*.yaml
criterion:
  _target_: torch.nn.CrossEntropyLoss

# Or with parameters
criterion:
  _target_: torch.nn.CrossEntropyLoss
  weight: [1.0, 2.0, 1.5]
  label_smoothing: 0.1
```

### 2. Architecture-Specific Patterns

**Model Configuration Template**:
```yaml
# configs/model/{architecture}.yaml
_target_: src.models.mnist_module.MNISTLitModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
  weight_decay: 0.0

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 10

criterion:
  _target_: torch.nn.CrossEntropyLoss

net:
  _target_: src.models.components.{network}.{Class}
  # Architecture-specific parameters here

compile: false
```

### 3. Multihead Configuration

**Multihead Model Pattern**:
```yaml
# configs/model/mnist_multihead_cnn_422k.yaml
_target_: src.models.mnist_module.MNISTLitModule

# Multiple criteria for different tasks
criteria:
  digit:
    _target_: torch.nn.CrossEntropyLoss
  thickness:
    _target_: torch.nn.CrossEntropyLoss
  smoothness:
    _target_: torch.nn.CrossEntropyLoss

# Task weighting
loss_weights:
  digit: 1.0
  thickness: 0.5
  smoothness: 0.5

net:
  _target_: src.models.components.simple_cnn.SimpleCNN
  heads_config:
    digit: 10
    thickness: 5
    smoothness: 3
```

## ✅ Preflight Label Validation

- Purpose: catches degenerate targets before training (e.g., all zeros from mis-decoding).
- Config (in `configs/train.yaml`):
  - `preflight.enabled`: `true` by default.
  - `preflight.label_diversity_batches`: number of train batches to sample (default `3`).
- Behavior: logs unique-label previews per head across sampled batches and fails fast if any head has ≤1 unique label.
- Override examples:
  - `python src/train.py preflight.enabled=false`
  - `python src/train.py preflight.label_diversity_batches=5`
- Note: the dataloader also checks per-batch label diversity; it asserts during training and only warns during validation/test.


## 🧪 Experiment Configuration System

### What are Experiment Configs?

Experiment configs provide complete, reproducible specifications for research:

- **Complete setup**: All parameters in one place
- **Reproducibility**: Fixed seeds and hyperparameters
- **Version control**: Lock in successful configurations
- **One-command execution**: Complex setups with simple commands

### When to Use Each Approach

| Use Case | Command-Line Overrides | Experiment Configs |
|----------|----------------------|-------------------|
| **Quick exploration** | ✅ `python src/train.py model=mnist_cnn` | ❌ Too rigid |
| **Parameter testing** | ✅ `python src/train.py model.optimizer.lr=0.01` | ❌ Overkill |
| **Reproducible research** | ❌ Hard to reproduce exactly | ✅ Fixed configuration |
| **Paper results** | ❌ Parameters can vary | ✅ Version controlled |
| **Baseline comparisons** | ❌ Inconsistent setup | ✅ Standardized |

### Experiment Configuration Structure

```yaml
# configs/experiment/example.yaml
# @package _global_

defaults:
  - override /data: mnist
  - override /model: mnist
  - override /callbacks: default
  - override /trainer: default

# Fixed for reproducibility
seed: 12345
tags: ["mnist", "simple_dense_net", "baseline"]

# Specific hyperparameters
trainer:
  max_epochs: 10
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 0.002
  net:
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

data:
  batch_size: 64

logger:
  wandb:
    tags: ${tags}
    group: "mnist_baseline"
```

### Creating Experiment Configs

1. **Start with defaults**: Begin with working base configurations
2. **Override specifically**: Only change what's necessary
3. **Fix the seed**: Ensure reproducibility
4. **Tag appropriately**: Enable easy filtering and comparison
5. **Document purpose**: Add comments explaining the experiment goal

## 🎯 Configuration Best Practices

### 1. Systematic Hyperparameter Organization

**Learning Rate Experiments**:
```bash
# Test different learning rates
python src/train.py model.optimizer.lr=0.01 tags="[lr_study,0.01]"
python src/train.py model.optimizer.lr=0.001 tags="[lr_study,0.001]"
python src/train.py model.optimizer.lr=0.0001 tags="[lr_study,0.0001]"
```

**Architecture Comparison**:
```bash
# Same hyperparameters, different architectures
python src/train.py experiment=baseline_config model=mnist_cnn tags="[arch_study,cnn]"
python src/train.py experiment=baseline_config model=mnist_convnext_68k tags="[arch_study,convnext]"
python src/train.py experiment=baseline_config model=mnist_vit_210k tags="[arch_study,vit]"
```

### 2. Hierarchical Configuration Strategy

**Base Configuration**:
```yaml
# configs/experiment/_base_mnist.yaml
defaults:
  - override /data: mnist
  - override /trainer: default

seed: 12345
trainer:
  max_epochs: 20
  gradient_clip_val: 0.5
data:
  batch_size: 64
```

**Specific Experiments**:
```yaml
# configs/experiment/mnist_cnn_study.yaml
defaults:
  - _base_mnist
  - override /model: mnist_cnn_421k

tags: ["mnist", "cnn", "study"]
model:
  optimizer:
    lr: 0.002
```

### 3. Environment-Specific Configuration

**Local Configuration** (not version controlled):
```yaml
# configs/local/default.yaml
# @package _global_

# Machine-specific paths
data_dir: /path/to/your/data
log_dir: /path/to/your/logs

# Hardware optimization
data:
  num_workers: 8
  pin_memory: true

trainer:
  accelerator: mps  # or gpu, cpu
```

## 🔄 Dynamic Configuration

### Command-Line Overrides

**Basic Overrides**:
```bash
# Simple parameter changes
python src/train.py trainer.max_epochs=20
python src/train.py model.optimizer.lr=0.01
python src/train.py data.batch_size=128

# Multiple overrides
python src/train.py trainer.max_epochs=20 model.optimizer.lr=0.01 data.batch_size=128
```

**Adding New Parameters**:
```bash
# Add new parameters with +
python src/train.py +model.new_param="value"
python src/train.py +trainer.new_flag=true
```

**Removing Parameters**:
```bash
# Remove parameters with ~
python src/train.py ~model.scheduler
python src/train.py ~callbacks
```

### Variable Interpolation

**Referencing Other Config Values**:
```yaml
# configs/experiment/example.yaml
batch_size: 64
tags: ["experiment", "batch_${data.batch_size}"]

data:
  batch_size: ${batch_size}

logger:
  wandb:
    tags: ${tags}
    name: "experiment_batch_${batch_size}"
```

**Environment Variables**:
```yaml
# Reference environment variables
data_dir: ${oc.env:DATA_DIR,/default/path}
wandb_project: ${oc.env:WANDB_PROJECT,"default_project"}
```

## 🏷️ Tagging Strategy

### Hierarchical Tagging
```yaml
tags: ["dataset", "architecture", "experiment_type", "version"]

# Examples:
tags: ["mnist", "cnn", "baseline", "v1"]
tags: ["cifar10", "convnext", "benchmark", "v2"]
tags: ["mnist", "multihead", "research", "v1"]
```

### Filtering and Analysis
```bash
# Filter by tags in logs/experiments
ls logs/train/runs/ | grep "mnist.*cnn"
ls logs/train/runs/ | grep "baseline.*v1"
```

## 🎨 Advanced Configuration Patterns

### 1. Conditional Configuration
```yaml
# configs/model/adaptive_model.yaml
defaults:
  - mnist_cnn_421k

# Override based on data
net:
  input_channels: ???  # Will be set by data module

# Conditional parameters
optimizer:
  lr: ${oc.select:net.input_channels,0.001}  # Different LR for different inputs
```

### 2. Configuration Validation
```yaml
# configs/experiment/validated_experiment.yaml
# @package _global_

defaults:
  - override /data: mnist
  - override /model: mnist_cnn_421k

# Validation constraints
_target_: ???  # Must be specified
trainer:
  max_epochs: ???  # Must be specified

# With defaults
model:
  optimizer:
    lr: ${oc.decode:'${oc.env:LEARNING_RATE,0.001}'}
```

### 3. Sweeps and Hyperparameter Search
```yaml
# configs/hparams_search/mnist_optuna.yaml
# @package _global_

defaults:
  - override /hydra/sweeper: optuna

optimized_metric: "val/acc_best"

hydra:
  sweeper:
    direction: maximize
    n_trials: 20
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      data.batch_size: choice(32, 64, 128, 256)
      model.net.lin1_size: choice(64, 128, 256)
```

**Usage**:
```bash
python src/train.py -m hparams_search=mnist_optuna experiment=example
```

## 🔍 Configuration Debugging

### 1. Print Resolved Configuration
```bash
# See final configuration
python src/train.py --cfg job

# Print specific sections
python src/train.py --cfg job --package model
python src/train.py --cfg job --package data
```

### 2. Validate Configuration
```bash
# Check for errors without running
python src/train.py --cfg job trainer.max_epochs=0
```

### 3. Configuration Diff
```bash
# Compare configurations
python src/train.py --cfg job > config1.yaml
python src/train.py model=mnist_cnn --cfg job > config2.yaml
diff config1.yaml config2.yaml
```

## 📊 Configuration Management Workflow

### 1. Development Phase
```bash
# Quick iteration with overrides
python src/train.py model=mnist_cnn trainer.max_epochs=5
python src/train.py model=mnist_cnn model.optimizer.lr=0.01 trainer.max_epochs=5
```

### 2. Experimentation Phase
```bash
# Create experiment configs for promising combinations
# Save as configs/experiment/my_experiment.yaml
python src/train.py experiment=my_experiment
```

### 3. Production Phase
```bash
# Use stable experiment configs
python src/train.py experiment=production_baseline
python src/train.py experiment=production_advanced
```

## 🔗 Integration Points

The configuration system integrates with:
- **Lightning modules**: Automatic parameter injection
- **Data modules**: Dynamic configuration based on data properties
- **Callbacks**: Configurable training behavior
- **Loggers**: Automatic hyperparameter logging
- **Testing**: Reproducible test configurations

For specific architecture configurations, see [README-ARCHITECTURES.md](README-ARCHITECTURES.md).
For multihead configuration patterns, see [README-MULTIHEAD.md](README-MULTIHEAD.md).
