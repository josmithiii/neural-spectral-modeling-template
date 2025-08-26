# Neural Network Architectures

## Overview

The Lightning-Hydra-Template-Extended supports multiple neural network architectures optimized for different use cases. All architectures are easily switchable through Hydra configuration.

## 🏗️ Architecture Comparison

| Architecture | Parameters | Type | Best For | Config Files |
|-------------|------------|------|----------|-------------|
| **SimpleDenseNet** | 8K-68K | Fully-connected | Quick prototyping | `mnist_sdn_*.yaml` |
| **SimpleMLP** | 8K-68K | Fully-connected (no BN) | Batch size = 1 | `simple_mlp_*.yaml` |
| **SimpleCNN** | 8K-3.3M | Convolutional | Image classification | `mnist_cnn_*.yaml` |
| **ConvNeXt-V2** | 18K-725K | Modern CNN | Efficient performance | `mnist_convnext_*.yaml` |
| **Vision Transformer** | 38K-821K | Transformer | Large-scale datasets | `mnist_vit_*.yaml` |
| **EfficientNet** | 22K-7M | Efficient CNN | Mobile/edge deployment | `mnist_efficientnet_*.yaml` |

## 📐 Architecture Details

### SimpleDenseNet (Original)
**Type**: Fully-connected neural network
**Best for**: Quick prototyping and baseline comparisons

**Architecture**:
- **Input**: Flattened 28×28 images (784 features)
- **Hidden layers**: [64, 128, 64] neurons
- **Normalization**: BatchNorm after each layer
- **Activation**: ReLU
- **Output**: 10 classes (MNIST)

**Characteristics**:
- **Parameters**: 68,048
- **Speed**: Fast training and inference ⚡
- **Memory**: Low requirements
- **Accuracy**: Good baseline performance

**Usage**:
```bash
# Default architecture
python src/train.py
make train

# Different sizes
python src/train.py model=mnist_sdn_8k     # 8K parameters
python src/train.py model=mnist_sdn_68k    # 68K parameters
```

### SimpleMLP
**Type**: Fully-connected neural network without BatchNorm
**Best for**: Batch size = 1 scenarios, inference-only applications

**Architecture**:
- **Input**: Flattened 28×28 images (784 features)
- **Hidden layers**: Configurable (default: [256, 128])
- **Normalization**: None (crucial for batch_size=1)
- **Activation**: ReLU
- **Dropout**: Configurable (default: 0.1)
- **Output**: Configurable classes

**Key Difference from SimpleDenseNet**:
SimpleMLP omits BatchNorm layers, making it suitable for scenarios where batch_size=1 is required (e.g., real-time inference, audio synthesis applications).

**Characteristics**:
- **Parameters**: Configurable (8K-68K typical range)
- **Speed**: Fast training and inference ⚡
- **Memory**: Low requirements
- **Batch size**: Works with any batch size including 1

**Usage**:
```bash
# Train SimpleMLP
python src/train.py model=simple_mlp

# With specific configuration
python src/train.py model=simple_mlp_256_128  # Hidden layers [256, 128]
python src/train.py trainer.datamodule.batch_size=1  # Batch size 1
```

### SimpleCNN
**Type**: Convolutional neural network
**Best for**: Image classification with spatial structure preservation

**Architecture**:
- **Input**: Raw 28×28 images (preserves spatial structure)
- **Conv Layer 1**: 1→32 channels, 3×3 kernel + BatchNorm + ReLU + MaxPool
- **Conv Layer 2**: 32→64 channels, 3×3 kernel + BatchNorm + ReLU + MaxPool
- **Global Pooling**: AdaptiveAvgPool2d(7×7)
- **Classifier**: Linear(3136→128) + ReLU + Dropout(0.25) + Linear(128→10)

**Characteristics**:
- **Parameters**: 421,482 (single-head), 422,330 (multihead)
- **Speed**: Medium training speed 🚀
- **Memory**: Moderate requirements
- **Accuracy**: Higher than SimpleDenseNet

**Multihead Support**:
The SimpleCNN architecture supports multihead classification for multi-task learning:
- **Primary head**: Digit classification (10 classes)
- **Secondary heads**: Thickness (5 classes), Smoothness (3 classes)

**Auxiliary Features Support**:
SimpleCNN can incorporate auxiliary scalar features alongside image data:
- **Hybrid input**: Image tensor + auxiliary scalar vector
- **Feature fusion**: Auxiliary features processed through separate MLP, concatenated with CNN features
- **Use case**: Incorporating measured parameters, metadata, or other scalar inputs

**Usage**:
```bash
# Single-head CNN
python src/train.py model=mnist_cnn_421k
make trc

# Multihead CNN
python src/train.py experiment=multihead_cnn_mnist
make emhcm

# Different sizes
python src/train.py model=mnist_cnn_8k     # 8K parameters
python src/train.py model=mnist_cnn_421k   # 421K parameters

# With auxiliary features (requires compatible dataset)
python src/train.py model.net.auxiliary_input_size=5  # 5 auxiliary features
```

### ConvNeXt-V2
**Type**: Modern convolutional neural network with Global Response Normalization
**Best for**: Efficient performance with modern architectural improvements

**Architecture**:
- **Stem**: Adaptive conv layer (2×2 stride 2 for MNIST, 4×4 stride 4 for ImageNet)
- **4 Stages**: Progressive downsampling with residual blocks
- **ConvNeXt Block**:
  - 7×7 depthwise convolution
  - LayerNorm normalization
  - 4× MLP expansion
  - **Global Response Normalization (GRN)** - key innovation
  - 1×1 pointwise convolution
- **Classifier**: Global average pooling + linear projection

**Key Innovation - Global Response Normalization**:
GRN improves training stability and feature learning by normalizing responses across spatial and channel dimensions.

**Characteristics**:
- **Parameters**: 18K (tiny), 73K (small), 288K (base), 725K (large)
- **Speed**: Similar to CNN, faster than ViT for smaller models 🚀
- **Memory**: Efficient memory usage
- **Accuracy**: Superior to standard CNNs

**Usage**:
```bash
# Different sizes
python src/train.py model=mnist_convnext_18k    # Tiny
python src/train.py model=mnist_convnext_68k    # Small
python src/train.py model=mnist_convnext_288k   # Base
python src/train.py model=mnist_convnext_725k   # Large

# Make shortcuts
make trcns   # Small (~73K)
make trcnm   # Medium (~288K)
make trcnl   # Large (~725K)
```

### Vision Transformer (ViT)
**Type**: Transformer architecture applied to images via patch embeddings
**Best for**: Large-scale datasets and attention-based learning

**Architecture**:
- **Patch Embedding**: 28×28 images → 7×7 patches → 16 patches → embedded vectors
- **Positional Encoding**: Learnable position embeddings for each patch
- **Transformer Blocks**:
  - Multi-head self-attention
  - MLP with GELU activation
  - Residual connections
  - LayerNorm (not BatchNorm)
- **Classification Head**: Global average pooling + linear projection

**Key Benefits**:
- **Attention mechanism**: Captures long-range dependencies
- **Parallelizable**: Highly efficient training
- **Scalable**: Performance improves with data and model size
- **SOTA potential**: State-of-the-art results on large datasets

**Characteristics**:
- **Parameters**: 38K (tiny), 210K (small), 821K (base)
- **Speed**: Slower than CNN for small models, competitive at scale
- **Memory**: Higher memory requirements
- **Accuracy**: Excellent on larger datasets

**SOTA Configuration**:
The `vit_mnist_995.yaml` experiment achieves state-of-the-art MNIST performance:
- 210K parameters
- 200 epochs training
- Custom normalization and augmentation
- Specialized data loading pipeline

**Usage**:
```bash
# Different sizes
python src/train.py model=mnist_vit_38k     # Tiny
python src/train.py model=mnist_vit_210k    # Small
python src/train.py model=mnist_vit_821k    # Base

# SOTA experiment
python src/train.py experiment=vit_mnist_995
make evit

# Quick test
make tqv
```

### EfficientNet
**Type**: Highly efficient CNN architecture with compound scaling
**Best for**: Mobile deployment and resource-constrained environments

**Architecture**:
- **Inverted Residual Blocks**: Mobile-optimized building blocks
- **Squeeze-and-Excitation**: Channel attention mechanism
- **Compound Scaling**: Balanced scaling of depth, width, and resolution
- **Swish Activation**: Learnable activation function

**Characteristics**:
- **Parameters**: 22K (tiny), 210K (small), 7M (large)
- **Speed**: Optimized for inference efficiency
- **Memory**: Very efficient memory usage
- **Accuracy**: Excellent accuracy-to-parameter ratio

**Usage**:
```bash
# Different sizes
python src/train.py model=mnist_efficientnet_22k    # Tiny
python src/train.py model=mnist_efficientnet_210k   # Small
python src/train.py model=mnist_efficientnet_7m     # Large
```

## 🎯 CIFAR-Optimized Architectures

All architectures have CIFAR-10 and CIFAR-100 variants optimized for 32×32 RGB images:

### CIFAR Configuration Files
```
configs/model/
├── cifar10_cnn_64k.yaml           # CNN for CIFAR-10 (3.3M params)
├── cifar10_convnext_210k.yaml     # ConvNeXt for CIFAR-10 (288K params)
├── cifar10_vit_210k.yaml          # ViT for CIFAR-10 (210K params)
├── cifar10_efficientnet_210k.yaml # EfficientNet for CIFAR-10 (210K params)
├── cifar100_cnn_64k.yaml          # CNN for CIFAR-100 (3.3M params)
├── cifar100_convnext_210k.yaml    # ConvNeXt for CIFAR-100 (290K params)
├── cifar100_vit_210k.yaml         # ViT for CIFAR-100 (210K params)
└── cifar100_efficientnet_210k.yaml # EfficientNet for CIFAR-100 (210K params)
```

### CIFAR Usage
```bash
# CIFAR-10 architectures
python src/train.py model=cifar10_cnn_64k
python src/train.py model=cifar10_convnext_210k
python src/train.py model=cifar10_vit_210k

# CIFAR-100 architectures
python src/train.py model=cifar100_cnn_64k
python src/train.py model=cifar100_convnext_210k
python src/train.py model=cifar100_vit_210k
```

## 🔧 Adding New Architectures

### Step 1: Create Architecture Component
```python
# src/models/components/my_network.py
import torch
from torch import nn

class MyNetwork(nn.Module):
    def __init__(self, input_size: int = 784, output_size: int = 10, hidden_size: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.view(x.size(0), -1))
```

### Step 2: Create Configuration
```yaml
# configs/model/my_model.yaml
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
  _target_: src.models.components.my_network.MyNetwork
  input_size: 784
  output_size: 10
  hidden_size: 128

compile: false
```

### Step 3: Use New Architecture
```bash
python src/train.py model=my_model
```

## 🎯 Architecture Selection Guide

### For Quick Prototyping
- **SimpleDenseNet**: Fast iteration, simple debugging
- **SimpleCNN (small)**: Basic spatial understanding

### For Research
- **ConvNeXt-V2**: Modern CNN with excellent efficiency
- **Vision Transformer**: Attention-based learning, scalable
- **SimpleCNN (multihead)**: Multi-task learning experiments

### For Production
- **EfficientNet**: Mobile-optimized, efficient inference
- **ConvNeXt-V2 (small)**: Good balance of accuracy and speed

### For Benchmarking
- **All architectures**: Systematic comparison across datasets
- **CIFAR variants**: Standardized computer vision evaluation

## 📊 Performance Characteristics

### Training Speed (relative)
1. **SimpleDenseNet**: Fastest ⚡⚡⚡
2. **EfficientNet**: Fast ⚡⚡
3. **ConvNeXt-V2**: Medium ⚡
4. **SimpleCNN**: Medium ⚡
5. **Vision Transformer**: Slowest (small models)

### Memory Usage (relative)
1. **SimpleDenseNet**: Lowest 📱
2. **EfficientNet**: Low 📱
3. **ConvNeXt-V2**: Medium 💻
4. **SimpleCNN**: Medium 💻
5. **Vision Transformer**: Highest 🖥️

### Accuracy Potential
1. **Vision Transformer**: Highest (with scale) 🏆
2. **ConvNeXt-V2**: High 🥈
3. **EfficientNet**: High 🥈
4. **SimpleCNN**: Medium 🥉
5. **SimpleDenseNet**: Baseline ⭐

For detailed benchmarking results, see [README-BENCHMARKS.md](README-BENCHMARKS.md).
