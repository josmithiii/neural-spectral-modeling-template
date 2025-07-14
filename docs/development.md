# Development Guide

## Overview

This guide covers the development philosophy, file organization, extension patterns, and integration approach used in the Lightning-Hydra-Template-Extended project.

## 🎯 Development Philosophy

### Non-Destructive Extensions
The project follows a **non-destructive extension** approach:

- **Add, don't modify**: Create new files instead of changing existing ones
- **Preserve compatibility**: Original functionality remains unchanged
- **Easy rollback**: Extensions can be removed without affecting the base template
- **Zero risk**: Existing workflows continue to work
- **Incremental adoption**: Users can adopt new features gradually

### Configuration-Driven Development
- **No code changes for common experiments**: Use Hydra configuration instead
- **Version-controlled configurations**: All experiments are reproducible
- **Consistent patterns**: Follow established configuration conventions
- **Modular design**: Components are independent and configurable

### Research-Focused Design
- **Rapid iteration**: Minimize boilerplate for quick experimentation
- **Systematic comparison**: Enable fair architecture and hyperparameter comparison
- **Reproducible results**: Fixed seeds and deterministic configurations
- **Literature compatibility**: Competitive baselines and standard benchmarks

## 📁 File Organization

### Extension File Structure
```
├── configs/
│   ├── model/
│   │   ├── mnist_cnn_*.yaml          # CNN variants
│   │   ├── mnist_convnext_*.yaml     # ConvNeXt variants
│   │   ├── mnist_vit_*.yaml          # Vision Transformer variants
│   │   ├── mnist_efficientnet_*.yaml # EfficientNet variants
│   │   ├── mnist_multihead_*.yaml    # Multihead variants
│   │   ├── cifar10_*.yaml            # CIFAR-10 optimized models
│   │   └── cifar100_*.yaml           # CIFAR-100 optimized models
│   ├── data/
│   │   ├── mnist_mh.yaml             # MNIST multihead data configuration
│   │   ├── mnist_vit_995.yaml        # MNIST ViT-specific data loading for SOTA exp
│   │   ├── cifar10.yaml              # CIFAR-10 data loading
│   │   ├── cifar100.yaml             # CIFAR-100 fine-grained
│   │   └── cifar100_coarse.yaml      # CIFAR-100 coarse-grained
│   └── experiment/
│       ├── multihead_*.yaml          # Multihead experiments
│       ├── vit_*.yaml                # ViT experiments
│       ├── convnext_*.yaml           # ConvNeXt experiments
│       ├── cifar10_benchmark_*.yaml  # CIFAR-10 benchmarks
│       └── cifar100_*.yaml           # CIFAR-100 benchmarks
├── src/
│   ├── models/components/
│   │   ├── simple_cnn.py             # CNN with multihead support
│   │   ├── convnext_v2.py            # ConvNeXt-V2 implementation
│   │   ├── vision_transformer.py     # Vision Transformer implementation
│   │   └── simple_efficientnet.py    # EfficientNet implementation
│   └── data/
│       ├── multihead_dataset.py      # Multihead label generation
│       ├── mnist_vit_995_datamodule.py # ViT-specific data loading
│       ├── cifar10_datamodule.py     # CIFAR-10 data module
│       └── cifar100_datamodule.py    # CIFAR-100 data module
├── tests/
│   ├── test_multihead.py             # Multihead testing
│   ├── test_cifar10_datamodule.py    # CIFAR-10 tests
│   └── test_cifar100_datamodule.py   # CIFAR-100 tests
├── benchmarks/
│   └── scripts/
│       └── benchmark_cifar.py        # Automated benchmark suite
├── Makefile                          # Enhanced make targets
└── README-*.md                       # Modular documentation
```

### Naming Conventions

**Configuration Files**:
- `{dataset}_{architecture}_{size}.yaml` - Model configs
- `{feature}_{dataset}.yaml` - Data configs
- `{dataset}_{purpose}_{architecture}.yaml` - Experiment configs

**Source Files**:
- `{architecture}.py` - Architecture implementations
- `{feature}_dataset.py` - Dataset extensions
- `{dataset}_datamodule.py` - Data modules
- `test_{feature}.py` - Test files

**Make Targets**:
- `{action}-{architecture}-{size}` - Full names
- `{a}{ac}{s}` - Abbreviations (action-architecture-size)

## 🔧 Extension Patterns

### 1. Adding New Architectures

**Step 1: Implement Architecture**
```python
# src/models/components/my_architecture.py
import torch
from torch import nn

class MyArchitecture(nn.Module):
    def __init__(self, input_size: int = 784, output_size: int = 10, **kwargs):
        super().__init__()
        # Implementation here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass implementation
        return output
```

**Step 2: Create Configuration**
```yaml
# configs/model/mnist_myarch_SIZE.yaml
_target_: src.models.mnist_module.MNISTLitModule

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

criterion:
  _target_: torch.nn.CrossEntropyLoss

net:
  _target_: src.models.components.my_architecture.MyArchitecture
  input_size: 784
  output_size: 10
  # Architecture-specific parameters

compile: false
```

**Step 3: Add Tests**
```python
# tests/test_my_architecture.py
import torch
from src.models.components.my_architecture import MyArchitecture

def test_my_architecture():
    model = MyArchitecture(input_size=784, output_size=10)
    x = torch.randn(2, 784)
    output = model(x)
    assert output.shape == (2, 10)
```

**Step 4: Add Make Targets**
```makefile
# In Makefile
train-myarch:
	python src/train.py model=mnist_myarch_SIZE

train-quick-myarch:
	python src/train.py model=mnist_myarch_SIZE trainer.max_epochs=1 trainer.limit_train_batches=10

# Abbreviations
trma: train-myarch
tqma: train-quick-myarch
```

### 2. Adding New Datasets

**Step 1: Create Data Module**
```python
# src/data/mydataset_datamodule.py
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch

class MyDatasetDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data/", batch_size: int = 64, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Download data if needed
        pass

    def setup(self, stage: str = None):
        # Create datasets
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
```

**Step 2: Create Data Configuration**
```yaml
# configs/data/mydataset.yaml
_target_: src.data.mydataset_datamodule.MyDatasetDataModule
data_dir: ${paths.data_dir}
batch_size: 64
num_workers: 0
pin_memory: False
```

**Step 3: Create Model Variants**
```yaml
# configs/model/mydataset_cnn_SIZE.yaml
# Based on existing CNN config with dataset-specific adaptations
net:
  _target_: src.models.components.simple_cnn.SimpleCNN
  input_channels: 3  # RGB instead of grayscale
  # Other adaptations
```

### 3. Adding Experiment Configurations

**Systematic Experiment Design**:
```yaml
# configs/experiment/mydataset_benchmark_cnn.yaml
# @package _global_

defaults:
  - override /data: mydataset
  - override /model: mydataset_cnn_SIZE
  - override /trainer: default
  - override /callbacks: default

seed: 12345
tags: ["mydataset", "cnn", "benchmark"]

trainer:
  max_epochs: 50

model:
  optimizer:
    lr: 0.001

data:
  batch_size: 128
```

## 🧪 Testing Strategy

### Test Categories

**1. Unit Tests**
- Individual component functionality
- Architecture forward passes
- Data loading correctness

**2. Integration Tests**
- End-to-end training loops
- Configuration instantiation
- Multihead functionality

**3. Smoke Tests**
- Quick validation runs
- Architecture compatibility
- Benchmark functionality

### Test Implementation

**Architecture Tests**:
```python
def test_architecture_shapes():
    """Test that architecture produces correct output shapes."""
    model = MyArchitecture(input_size=784, output_size=10)
    x = torch.randn(2, 784)
    output = model(x)
    assert output.shape == (2, 10)

def test_architecture_gradients():
    """Test that gradients flow correctly."""
    model = MyArchitecture(input_size=784, output_size=10)
    x = torch.randn(2, 784)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
```

**Configuration Tests**:
```python
def test_config_instantiation():
    """Test that configurations can be instantiated."""
    from hydra import compose, initialize

    with initialize(config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=my_model"])
        model = hydra.utils.instantiate(cfg.model)
        assert model is not None
```

## 🔄 Integration with Original Template

### Preserved Functionality
All original template features remain fully functional:

- **Original models**: SimpleDenseNet works unchanged
- **Original data**: MNIST data loading preserved
- **Original configs**: All existing configurations work
- **Original workflows**: Training, evaluation, testing unchanged
- **Original documentation**: Base README.md preserved

### Enhanced Functionality
Extensions enhance without breaking:

- **Enhanced models**: New architectures added alongside originals
- **Enhanced data**: New datasets complement MNIST
- **Enhanced configs**: New configurations follow same patterns
- **Enhanced workflows**: New make targets supplement originals
- **Enhanced documentation**: Modular docs complement original

### Upgrade Path
Users can adopt extensions incrementally:

1. **Start familiar**: Use original SimpleDenseNet on MNIST
2. **Try new architecture**: Switch to CNN with `model=mnist_cnn`
3. **Explore datasets**: Try CIFAR with `experiment=cifar10_benchmark_cnn`
4. **Advanced features**: Experiment with multihead classification
5. **Full adoption**: Use comprehensive benchmark suites

## 🎨 Code Style and Conventions

### Python Code Style
- **Type hints**: Use throughout for better IDE support
- **Docstrings**: Document classes and complex methods
- **Imports**: Follow PEP 8 import ordering
- **Naming**: Use descriptive names, avoid abbreviations in code

### Configuration Style
- **Consistent structure**: Follow established config patterns
- **Clear naming**: Use descriptive configuration names
- **Parameter grouping**: Organize related parameters together
- **Documentation**: Add comments for non-obvious choices

### Documentation Style
- **Modular organization**: Each README focuses on specific topic
- **Clear examples**: Provide concrete usage examples
- **Cross-references**: Link related sections across files
- **Update synchronization**: Keep docs in sync with code changes

## 🚀 Performance Considerations

### Architecture Optimization
- **Parameter efficiency**: Provide multiple size variants
- **Memory optimization**: Use efficient operations where possible
- **Hardware adaptation**: Support MPS, GPU, and CPU execution
- **Batch size adaptation**: Provide guidance for different hardware

### Configuration Optimization
- **Lazy loading**: Use Hydra's lazy instantiation
- **Parameter validation**: Catch configuration errors early
- **Default values**: Provide sensible defaults for all parameters
- **Hardware detection**: Automatically use best available accelerator

## 🔗 Extension Guidelines

### For Contributors

**Before Adding Features**:
1. Check if it fits the non-destructive philosophy
2. Ensure backward compatibility is maintained
3. Follow established naming conventions
4. Add appropriate tests and documentation

**For New Architectures**:
1. Implement with configurable parameters
2. Provide multiple size variants
3. Add comprehensive tests
4. Document expected performance

**For New Datasets**:
1. Follow PyTorch Lightning DataModule patterns
2. Provide data augmentation options
3. Include proper test/validation splits
4. Document dataset characteristics

### For Users

**Customization Strategy**:
1. Start with existing configurations
2. Use command-line overrides for experimentation
3. Create experiment configs for reproducible research
4. Contribute successful patterns back to the project

**Best Practices**:
1. Use descriptive tags for experiment tracking
2. Fix seeds for reproducible results
3. Document significant configuration changes
4. Share successful experiment configurations

This development approach ensures that the Lightning-Hydra-Template-Extended remains maintainable, extensible, and user-friendly while providing powerful capabilities for deep learning research.
