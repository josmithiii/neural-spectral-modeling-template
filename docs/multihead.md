# Multihead Classification System

## Overview

The multihead classification system enables a single neural network to predict multiple related tasks simultaneously. This approach leverages shared feature learning while maintaining separate prediction heads for each task, providing efficiency and regularization benefits.

## 🎯 Concept

### What is Multihead Classification?

Multihead classification allows one model to solve multiple related prediction tasks:
- **Shared backbone**: Common feature extraction layers
- **Multiple heads**: Task-specific prediction layers
- **Joint training**: Simultaneous optimization across all tasks
- **Efficiency**: One forward pass, multiple predictions

### Benefits

1. **Shared Learning**: Features learned for one task benefit others
2. **Regularization**: Multi-task objective prevents overfitting
3. **Efficiency**: Single model vs. multiple separate models
4. **Research Value**: Enables multi-task learning experiments

## 🧠 MNIST Multihead Implementation

### Task Definition

The MNIST multihead system predicts three related tasks from handwritten digits:

| Task | Classes | Description | Examples |
|------|---------|-------------|----------|
| **Digit** | 10 | Primary digit classification | 0, 1, 2, ..., 9 |
| **Thickness** | 5 | Stroke thickness estimation | very thin → very thick |
| **Smoothness** | 3 | Character smoothness | angular → medium → smooth |

### Synthetic Label Generation

Since MNIST only provides digit labels, the system generates thickness and smoothness labels using intelligent heuristics:

#### Thickness Mapping
```python
# Based on digit complexity and stroke patterns
thickness_map = {
    0: 2,  # Medium - curved but simple
    1: 1,  # Thin - minimal strokes
    2: 3,  # Thick - complex curves
    3: 3,  # Thick - multiple curves
    4: 2,  # Medium - angular but simple
    5: 3,  # Thick - complex shape
    6: 2,  # Medium - curved
    7: 1,  # Thin - simple diagonal
    8: 4,  # Very thick - most complex
    9: 3   # Thick - curved and complex
}
```

#### Smoothness Mapping
```python
# Based on geometric properties
smoothness_map = {
    0: 2,  # Smooth - circular
    1: 0,  # Angular - straight lines
    2: 1,  # Medium - mixed curves/angles
    3: 1,  # Medium - some curves
    4: 0,  # Angular - sharp angles
    5: 1,  # Medium - mixed features
    6: 2,  # Smooth - curved
    7: 0,  # Angular - diagonal line
    8: 2,  # Smooth - circular shapes
    9: 2   # Smooth - curved
}
```

### Architecture

The multihead CNN extends the standard SimpleCNN:

**Shared Backbone**:
- Conv2d(1→32, 3×3) + BatchNorm + ReLU + MaxPool
- Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool
- AdaptiveAvgPool2d(7×7)
- Linear(3136→128) + ReLU + Dropout(0.25)

**Multiple Heads**:
- **Digit head**: Linear(128→10) for digit classification
- **Thickness head**: Linear(128→5) for thickness estimation
- **Smoothness head**: Linear(128→3) for smoothness assessment

## ⚙️ Configuration

### Model Configuration
```yaml
# configs/model/mnist_mh_cnn_422k.yaml
_target_: src.models.mnist_module.MNISTLitModule

# Separate criteria for each task
criteria:
  digit:
    _target_: torch.nn.CrossEntropyLoss
  thickness:
    _target_: torch.nn.CrossEntropyLoss
  smoothness:
    _target_: torch.nn.CrossEntropyLoss

# Task weighting for loss combination
loss_weights:
  digit: 1.0        # Primary task - full weight
  thickness: 0.5    # Secondary task - reduced weight
  smoothness: 0.5   # Secondary task - reduced weight

# Network with multihead configuration
net:
  _target_: src.models.components.simple_cnn.SimpleCNN
  input_channels: 1
  conv1_channels: 32
  conv2_channels: 64
  fc_hidden: 128
  heads_config:
    digit: 10       # 10 digit classes
    thickness: 5    # 5 thickness levels
    smoothness: 3   # 3 smoothness levels
  dropout: 0.25
```

### Data Configuration
```yaml
# configs/data/multihead_mnist.yaml
_target_: src.data.mnist_datamodule.MNISTDataModule
data_dir: ${paths.data_dir}
batch_size: 128
train_val_test_split: [55_000, 5_000, 10_000]
num_workers: 2
pin_memory: False
persistent_workers: False

# Enable multihead label generation
multihead: true
```

### Complete Experiment
```yaml
# configs/experiment/multihead_cnn_mnist.yaml
defaults:
  - override /data: multihead_mnist
  - override /model: mnist_mh_cnn_422k
  - override /trainer: default

seed: 12345
tags: ["mnist", "multihead", "cnn"]

trainer:
  min_epochs: 1
  max_epochs: 10
  accelerator: auto
  devices: 1

model:
  optimizer:
    lr: 0.002

data:
  batch_size: 64

# Override callback monitoring metrics for multihead setup
callbacks:
  model_checkpoint:
    monitor: "val/acc_best"
    mode: "max"
  early_stopping:
    monitor: "val/acc_best"
    mode: "max"

# Override the optimized metric for this multihead model
optimized_metric: "val/digit_acc"
```

## 🚀 Usage

### Basic Training
```bash
# Run multihead experiment
python src/train.py experiment=multihead_cnn_mnist
make emhcm

# Quick test
python src/train.py experiment=multihead_cnn_mnist +trainer.fast_dev_run=true
```

### Custom Loss Weighting
```bash
# Emphasize digit task
python src/train.py experiment=multihead_cnn_mnist \
  model.loss_weights.digit=2.0 \
  model.loss_weights.thickness=0.5 \
  model.loss_weights.smoothness=0.5

# Equal weighting
python src/train.py experiment=multihead_cnn_mnist \
  model.loss_weights.digit=1.0 \
  model.loss_weights.thickness=1.0 \
  model.loss_weights.smoothness=1.0

# Focus on secondary tasks
python src/train.py experiment=multihead_cnn_mnist \
  model.loss_weights.digit=0.5 \
  model.loss_weights.thickness=1.0 \
  model.loss_weights.smoothness=1.0
```

### CIFAR-10 Multihead
```bash
# Multihead CNN on CIFAR-10
python src/train.py experiment=multihead_cnn_cifar10
make emhcc10
```

## 📊 Metrics and Logging

### Logged Metrics

**Single-head models**:
- `train/acc`, `val/acc`, `test/acc`
- `train/loss`, `val/loss`, `test/loss`

**Multihead models**:
- `train/digit_acc`, `val/digit_acc`, `test/digit_acc`
- `train/thickness_acc`, `val/thickness_acc`, `test/thickness_acc`
- `train/smoothness_acc`, `val/smoothness_acc`, `test/smoothness_acc`
- `train/loss`, `val/loss`, `test/loss` (combined)

### Performance Interpretation

**Expected Performance (10 epochs)**:
- **Digit accuracy**: 95-99% (primary task, well-defined)
- **Thickness accuracy**: 60-80% (synthetic labels, harder task)
- **Smoothness accuracy**: 70-85% (fewer classes, easier than thickness)

**Quick Test Performance (1 epoch)**:
- **Digit accuracy**: ~7.8%
- **Thickness accuracy**: ~39%
- **Smoothness accuracy**: ~52%

*Note: Quick test results are from minimal training and not representative of full performance*

## 🔬 Research Applications

### Multi-task Learning Studies
```bash
# Compare single-task vs multi-task learning
python src/train.py model=mnist_cnn trainer.max_epochs=20 tags="[single_task,digit]"
python src/train.py experiment=multihead_cnn_mnist trainer.max_epochs=20 tags="[multi_task,all]"
```

### Loss Weighting Experiments
```bash
# Study impact of loss weighting
for weight in 0.1 0.5 1.0 2.0; do
  python src/train.py experiment=multihead_cnn_mnist \
    model.loss_weights.thickness=$weight \
    tags="[weight_study,thickness_${weight}]"
done
```

### Architecture Comparison
```bash
# Compare architectures on multihead task
python src/train.py experiment=multihead_cnn_mnist tags="[arch_study,cnn]"
# Note: Other architectures need multihead implementation
```

## 🛠️ Implementation Details

### Dataset Wrapper
The `MultiheadDataset` class wraps the original dataset to add synthetic labels:

```python
# src/data/multihead_dataset.py
class MultiheadDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, idx):
        image, digit_label = self.base_dataset[idx]

        # Generate synthetic labels
        thickness_label = self.thickness_map[digit_label.item()]
        smoothness_label = self.smoothness_map[digit_label.item()]

        labels = {
            'digit': digit_label,
            'thickness': torch.tensor(thickness_label),
            'smoothness': torch.tensor(smoothness_label)
        }

        return image, labels
```

### Loss Computation
```python
# In MNISTLitModule for multihead models
def _calculate_loss(self, outputs, targets):
    total_loss = 0
    losses = {}

    for head_name, output in outputs.items():
        if head_name in self.criteria:
            loss = self.criteria[head_name](output, targets[head_name])
            weight = self.loss_weights.get(head_name, 1.0)
            losses[f"{head_name}_loss"] = loss
            total_loss += weight * loss

    losses["total_loss"] = total_loss
    return total_loss, losses
```

## 🔍 Advanced Usage

### Custom Synthetic Labels
You can modify the label generation logic for research:

```python
# Custom thickness mapping focusing on even/odd
thickness_map = {
    0: 0, 2: 0, 4: 0, 6: 0, 8: 0,  # Even digits - thin
    1: 4, 3: 4, 5: 4, 7: 4, 9: 4   # Odd digits - thick
}
```

### Adding New Tasks
To add a fourth task (e.g., "orientation"):

1. **Update heads_config**:
```yaml
heads_config:
  digit: 10
  thickness: 5
  smoothness: 3
  orientation: 4  # New task
```

2. **Add label mapping**:
```python
orientation_map = {0: 0, 1: 1, 2: 2, ...}  # Your logic here
```

3. **Update loss configuration**:
```yaml
criteria:
  orientation:
    _target_: torch.nn.CrossEntropyLoss
loss_weights:
  orientation: 0.5
```

## Learning Rate

The multihead experiment `make exp-multihead-cnn-mnist` has a higher
learning rate (0.002) than the 0.001 value in the single-head
experiment `make exp-cnn-mnist`.  This was tried because it's
performing multi-task learning with three simultaneous classification
tasks:
  - digit: 10 classes (standard MNIST)
  - thickness: 5 classes
  - smoothness: 3 classes

Higher learning rates are common in multi-task setups because:
  1. Gradient competition between multiple loss functions can reduce effective update magnitudes
  2. Shared feature learning requires more aggressive updates to find representations useful for all tasks
  3. Loss weighting (digit=1.0, thickness/smoothness=0.5) affects the overall gradient scale

The single-task CNN only learns digit classification, so it can use a more conservative learning rate.


## 🎯 Best Practices

### Loss Weighting Strategy
1. **Start equal**: Begin with equal weights for all tasks
2. **Primary task emphasis**: Give higher weight to main task (digit: 1.0, others: 0.5)
3. **Experimental tuning**: Use grid search for optimal weighting

### Training Tips
1. **Monitor all metrics**: Watch individual task performance
2. **Early stopping**: Use combined loss or primary task for stopping
3. **Learning rate**: May need lower LR to stabilize multi-task training (more epochs)

### Evaluation
1. **Task-specific metrics**: Evaluate each task independently
2. **Combined performance**: Consider weighted average of accuracies
3. **Ablation studies**: Compare with single-task baselines

## 🔧 Generic MultiheadLitModule

This template now includes a generic `MultiheadLitModule` that can work with any multihead dataset format:

### Usage with Auto-Configuration

```python
from src.models.multihead_module import MultiheadLitModule

# Auto-configure from dataset
model = MultiheadLitModule(
    net=your_network,
    optimizer=torch.optim.Adam,
    scheduler=torch.optim.lr_scheduler.StepLR,
    auto_configure_from_dataset=True  # Key feature
)

# Model automatically configures heads and criteria from dataset metadata
```

### Manual Configuration

```python
# Manual configuration
model = MultiheadLitModule(
    net=your_network,
    optimizer=torch.optim.Adam,
    scheduler=torch.optim.lr_scheduler.StepLR,
    criteria={
        'param_0': torch.nn.CrossEntropyLoss(),
        'param_1': torch.nn.CrossEntropyLoss(),
    },
    loss_weights={'param_0': 1.0, 'param_1': 0.5},
    auto_configure_from_dataset=False
)
```

### Key Features

- **Auto-configuration**: Automatically sets up heads and criteria from dataset
- **Metadata-based configuration**: Uses dataset metadata ranges when available
- **Loss weighting**: Configurable weights for different tasks
- **Backward compatibility**: Works with single-head models
- **Metrics tracking**: Separate accuracy metrics for each head
- **Lightning integration**: Full compatibility with Lightning ecosystem
- **Format validation**: Comprehensive validation with auto-detection

### Example Configuration

```yaml
# configs/model/generic_multihead.yaml
_target_: src.models.multihead_module.MultiheadLitModule

# Auto-configure from dataset
auto_configure_from_dataset: true

# Optional loss weighting
loss_weights:
  param_0: 1.0
  param_1: 0.8

# Network (heads_config will be auto-filled)
net:
  _target_: src.models.components.simple_cnn.SimpleCNN
  input_channels: 3
  input_size: 32
```

## 🔗 Integration

The multihead system integrates with:
- **Standard architectures**: Works with CNN, can be extended to others
- **CIFAR datasets**: Available for CIFAR-10 experiments
- **CIFAR-100-MH format**: New binary format for real multihead labels
- **Generic datasets**: Auto-detection for various multihead formats
- **Lightning logging**: All metrics automatically tracked
- **Hydra configuration**: Fully configurable through YAML
- **Auto-configuration**: Dynamic setup from dataset metadata
- **Format validation**: Comprehensive validation with error handling

For CIFAR-100-MH format details, see [docs/cifar100mh.md](cifar100mh.md).
For architecture details, see [README-ARCHITECTURES.md](README-ARCHITECTURES.md).
For configuration patterns, see [README-CONFIGURATION.md](README-CONFIGURATION.md).
