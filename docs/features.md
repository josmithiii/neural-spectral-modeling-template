# Lightning-Hydra-Template-Extended Features

## Overview

The [Lightning-Hydra-Template-Extended](https://github.com/josmithiii/lightning-hydra-template-extended.git) project extends the original [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) with powerful new capabilities for deep learning research while maintaining full backward compatibility.

## 🎯 Key Features

### 1. CIFAR Benchmark Suite
Comprehensive benchmarking capabilities for computer vision research:
- **CIFAR-10**: 10 classes, 32×32 RGB images
- **CIFAR-100**: 100 fine-grained classes + 20 coarse superclasses
- **Multiple architectures**: CNN, ConvNeXt, ViT, EfficientNet
- **Expected performance**: 85-95% (CIFAR-10), 55-75% (CIFAR-100)

### 2. Configurable Loss Functions
Loss functions are now configurable through Hydra:
```yaml
# Before: hardcoded in code
self.criterion = torch.nn.CrossEntropyLoss()

# Now: configurable in YAML
criterion:
  _target_: torch.nn.CrossEntropyLoss
```

### 3. Multiple Architecture Support
Easy switching between neural network architectures:

| Architecture | Parameters | Description |
|-------------|------------|-------------|
| **SimpleDenseNet** | 8K-68K | Original fully-connected network |
| **SimpleCNN** | 8K-3.3M | Convolutional neural network |
| **ConvNeXt-V2** | 18K-725K | Modern CNN with Global Response Normalization |
| **Vision Transformer** | 38K-821K | Transformer on image patches |
| **EfficientNet** | 22K-7M | Highly efficient CNN architecture |

### 4. Multihead Classification
Single models can predict multiple related tasks simultaneously:
- **Primary task**: Digit classification (0-9)
- **Secondary tasks**: Thickness estimation, smoothness assessment
- **Benefits**: Shared learning, regularization, efficiency

### 5. Experiment Configuration System
Reproducible research through complete experiment specifications:
- Fixed seeds for reproducibility
- Version-controlled hyperparameters
- Single-command execution
- Standardized baselines

### 6. Enhanced Make Targets
Convenient shortcuts for common tasks:
- **Training**: `make train`, `make trc` (CNN), `make trcn` (ConvNeXt)
- **Quick tests**: `make tq`, `make tqc`, `make tqcn`
- **Benchmarks**: `make cb10c` (CIFAR-10), `make cbs` (full suite)


## 📊 Expected Performance

### MNIST (Quick Tests - 1 epoch)
| Architecture | Parameters | Accuracy | Speed |
|-------------|------------|----------|-------|
| SimpleDenseNet | 68K | ~56.6% | Fast ⚡ |
| SimpleCNN | 421K | ~74.8% | Medium 🚀 |
| ConvNeXt-V2 | 73K | ~68.3% | Medium 🚀 |

### CIFAR (Full Training)
| Dataset | Architecture | Expected Accuracy |
|---------|-------------|------------------|
| CIFAR-10 | SimpleCNN | 85-92% |
| CIFAR-10 | ConvNeXt | 90-95% |
| CIFAR-100 | SimpleCNN | 55-70% |
| CIFAR-100 | ConvNeXt | 70-80% |

## 🔗 Integration

All original Lightning-Hydra template features remain fully functional:
- Original make targets work unchanged
- Hydra configuration system enhanced, not replaced
- Lightning module structure preserved
- Testing framework compatible
- Logging and callbacks unchanged

## 📚 Documentation

For detailed information, see:
- **[architectures.md](architectures.md)** - Architecture details and comparisons
- **[benchmarks.md](benchmarks.md)** - CIFAR benchmark system
- **[multihead.md](multihead.md)** - Multihead classification
- **[makefile.md](makefile.md)** - Complete make targets reference
- **[configuration.md](configuration.md)** - Configuration patterns
- **[development.md](development.md)** - Development and extension guide

## 🛠️ Common Usage Patterns

### Architecture Exploration
```bash
# Compare architectures with same hyperparameters
python src/train.py trainer.max_epochs=10                    # SimpleDenseNet
python src/train.py model=mnist_cnn trainer.max_epochs=10    # SimpleCNN
python src/train.py model=mnist_convnext_68k trainer.max_epochs=10  # ConvNeXt
```

### Custom Configuration
```bash
# Custom loss function
python src/train.py model.criterion._target_=torch.nn.NLLLoss

# Custom architecture parameters
python src/train.py model=mnist_cnn model.net.conv1_channels=64 model.net.dropout=0.5

# Hardware selection
python src/train.py trainer.accelerator=mps    # Mac Metal Performance Shaders
python src/train.py trainer.accelerator=gpu    # CUDA GPU
python src/train.py trainer.accelerator=cpu    # CPU fallback
```

This extension provides a comprehensive platform for deep learning research with modern architectures, systematic benchmarking, and reproducible experiments. 🚀
