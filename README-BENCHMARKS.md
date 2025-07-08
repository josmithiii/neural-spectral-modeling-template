# CIFAR Benchmark Suite

## Overview

The CIFAR Benchmark Suite provides comprehensive evaluation capabilities for computer vision models using the standard CIFAR-10 and CIFAR-100 datasets. This system enables systematic performance comparison across multiple architectures with literature-competitive baselines.

## 🎯 Datasets

### CIFAR-10
- **Classes**: 10 categories
- **Images**: 32×32 RGB
- **Categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Expected accuracy**: 85-95% (literature competitive)

### CIFAR-100
- **Classes**: 100 fine-grained categories
- **Images**: 32×32 RGB
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Expected accuracy**: 55-75% (challenging dataset)

### CIFAR-100 Coarse
- **Classes**: 20 coarse superclasses
- **Images**: 32×32 RGB
- **Grouping**: 5 fine classes per coarse class
- **Expected accuracy**: 75-85% (easier task)

## 🏗️ Supported Architectures

| Architecture | CIFAR-10 Params | CIFAR-100 Params | Expected Performance |
|-------------|-----------------|------------------|---------------------|
| **SimpleCNN** | 3.3M | 3.3M | CIFAR-10: 85-92%, CIFAR-100: 55-70% |
| **ConvNeXt** | 288K | 290K | CIFAR-10: 90-95%, CIFAR-100: 70-80% |
| **Vision Transformer** | 210K | 210K | CIFAR-10: 88-93%, CIFAR-100: 65-75% |
| **EfficientNet** | 210K | 210K | CIFAR-10: 89-94%, CIFAR-100: 67-77% |

## 🚀 Quick Start

### Quick Validation Tests (5 epochs)
```bash
# Individual quick tests
make cbq10c      # CIFAR-10 CNN
make cbq10cn     # CIFAR-10 ConvNeXt
make cbq100c     # CIFAR-100 CNN

# All quick tests
make cbqa        # All CIFAR quick validations
```

### Full Benchmarks
```bash
# CIFAR-10 benchmarks
make cb10c       # CNN on CIFAR-10
make cb10cn      # ConvNeXt on CIFAR-10
make cb10v       # Vision Transformer on CIFAR-10
make cb10e       # EfficientNet on CIFAR-10

# CIFAR-100 benchmarks
make cb100c      # CNN on CIFAR-100
make cb100cn     # ConvNeXt on CIFAR-100
make cb100cc     # CNN on CIFAR-100 coarse

# Complete benchmark suites
make cbs10       # All CIFAR-10 architectures
make cbs100      # All CIFAR-100 architectures
make cbs         # Automated benchmark suite
make cbsa        # Complete CIFAR benchmark suite
```

## 📊 Benchmark Results

### Quick Validation (5 epochs)
Results from rapid validation runs to verify functionality:

| Architecture | Dataset | Accuracy | Training Time | Status |
|-------------|---------|----------|---------------|--------|
| SimpleCNN | CIFAR-10 | ~45% | ~1.5 min | ✅ Verified |
| ConvNeXt | CIFAR-10 | ~42% | ~2 min | ✅ Verified |
| SimpleCNN | CIFAR-100 | ~15% | ~1.5 min | ✅ Verified |

*Note: 5-epoch results are for validation only; full training required for competitive performance*

### Expected Full Training Performance

Based on literature baselines and architecture capabilities:

#### CIFAR-10 (50 epochs)
| Architecture | Expected Accuracy | Literature Baseline | Training Time |
|-------------|------------------|-------------------|---------------|
| SimpleCNN | 85-92% | Competitive | ~15 min |
| ConvNeXt | 90-95% | State-of-the-art | ~18 min |
| Vision Transformer | 88-93% | Modern | ~25 min |
| EfficientNet | 89-94% | Efficient | ~20 min |

#### CIFAR-100 (75 epochs)
| Architecture | Expected Accuracy | Literature Baseline | Training Time |
|-------------|------------------|-------------------|---------------|
| SimpleCNN | 55-70% | Challenging | ~22 min |
| ConvNeXt | 70-80% | Advanced | ~28 min |
| Vision Transformer | 65-75% | Modern | ~35 min |
| EfficientNet | 67-77% | Efficient | ~30 min |

#### CIFAR-100 Coarse (50 epochs)
| Architecture | Expected Accuracy | Literature Baseline | Training Time |
|-------------|------------------|-------------------|---------------|
| SimpleCNN | 75-85% | Easier task | ~18 min |
| ConvNeXt | 85-90% | Advanced | ~22 min |

## 🎛️ Configuration Files

### Model Configurations
```
configs/model/
├── cifar10_cnn_64k.yaml           # SimpleCNN for CIFAR-10
├── cifar10_convnext_210k.yaml     # ConvNeXt for CIFAR-10
├── cifar10_vit_210k.yaml          # Vision Transformer for CIFAR-10
├── cifar10_efficientnet_210k.yaml # EfficientNet for CIFAR-10
├── cifar100_cnn_64k.yaml          # SimpleCNN for CIFAR-100
├── cifar100_convnext_210k.yaml    # ConvNeXt for CIFAR-100
├── cifar100_vit_210k.yaml         # Vision Transformer for CIFAR-100
├── cifar100_efficientnet_210k.yaml # EfficientNet for CIFAR-100
└── cifar100_coarse_cnn_64k.yaml   # SimpleCNN for CIFAR-100 coarse
```

### Data Configurations
```
configs/data/
├── cifar10.yaml                    # CIFAR-10 data loading
├── cifar100.yaml                   # CIFAR-100 fine-grained
└── cifar100_coarse.yaml            # CIFAR-100 coarse-grained
```

### Experiment Configurations
```
configs/experiment/
├── cifar10_benchmark_cnn.yaml     # CIFAR-10 CNN benchmark
├── cifar10_benchmark_convnext.yaml # CIFAR-10 ConvNeXt benchmark
├── cifar10_benchmark_vit.yaml     # CIFAR-10 ViT benchmark
├── cifar10_benchmark_efficientnet.yaml # CIFAR-10 EfficientNet benchmark
├── cifar100_benchmark_cnn.yaml    # CIFAR-100 CNN benchmark
├── cifar100_benchmark_convnext.yaml # CIFAR-100 ConvNeXt benchmark
├── cifar100_cnn.yaml              # CIFAR-100 standard experiment
└── cifar100_coarse_cnn.yaml       # CIFAR-100 coarse benchmark
```

## 🔄 Automated Benchmark Suite

### Using the Benchmark Script
```bash
# Direct script execution
python benchmarks/scripts/benchmark_cifar.py

# Through make targets
make cbs        # Automated benchmark suite
make cbsa       # Complete CIFAR benchmark suite
```

### Benchmark Suite Features
- **Systematic execution**: Runs all architecture/dataset combinations
- **Progress tracking**: Real-time status updates
- **Result aggregation**: Automatic result collection
- **Error handling**: Graceful failure recovery
- **Report generation**: Comprehensive performance summary

## 📈 Performance Analysis

### Accuracy vs Parameters
Understanding the accuracy-parameter trade-offs:

| Architecture | CIFAR-10 Params | CIFAR-10 Acc | Efficiency |
|-------------|-----------------|---------------|------------|
| ConvNeXt | 288K | 90-95% | ⭐⭐⭐⭐⭐ |
| ViT | 210K | 88-93% | ⭐⭐⭐⭐ |
| EfficientNet | 210K | 89-94% | ⭐⭐⭐⭐⭐ |
| SimpleCNN | 3.3M | 85-92% | ⭐⭐⭐ |

### Training Speed vs Accuracy
```
Fast Training:     SimpleCNN > EfficientNet > ConvNeXt > ViT
High Accuracy:     ConvNeXt > EfficientNet > ViT > SimpleCNN
Best Efficiency:   ConvNeXt > EfficientNet > ViT > SimpleCNN
```

## 🛠️ Custom Benchmarks

### Running Individual Experiments
```bash
# Custom hyperparameters
python src/train.py experiment=cifar10_benchmark_cnn trainer.max_epochs=100 model.optimizer.lr=0.01

# Architecture comparison with same settings
python src/train.py experiment=cifar10_benchmark_cnn trainer.max_epochs=25 tags="[cifar10,cnn,comparison]"
python src/train.py experiment=cifar10_benchmark_convnext trainer.max_epochs=25 tags="[cifar10,convnext,comparison]"

# Hardware optimization
python src/train.py experiment=cifar10_benchmark_cnn trainer.accelerator=mps data.num_workers=8
```

### Comparing CIFAR-100 Modes
```bash
# Fine-grained classification (100 classes)
python src/train.py experiment=cifar100_cnn trainer.max_epochs=75

# Coarse-grained classification (20 classes)
python src/train.py experiment=cifar100_coarse_cnn trainer.max_epochs=50
```

## 📋 Benchmark Checklist

### Before Running Benchmarks
- [ ] Environment activated (`source .venv/bin/activate`)
- [ ] GPU/MPS available (check with `make tmps` or `make tg`)
- [ ] Sufficient disk space for datasets and logs
- [ ] Time allocated (full benchmarks take hours)

### Quick Validation
- [ ] Run `make cbq10c` to verify CIFAR-10 CNN works
- [ ] Run `make cbq10cn` to verify CIFAR-10 ConvNeXt works
- [ ] Check logs in `logs/train/runs/` for results

### Full Benchmark Suite
- [ ] Run `make cbs10` for complete CIFAR-10 evaluation
- [ ] Run `make cbs100` for complete CIFAR-100 evaluation
- [ ] Use `make cbs` for automated systematic comparison

## 📊 Results Interpretation

### Accuracy Thresholds
- **CIFAR-10**: >90% is excellent, >85% is good, >80% is acceptable
- **CIFAR-100**: >70% is excellent, >60% is good, >50% is acceptable
- **CIFAR-100 Coarse**: >85% is excellent, >80% is good, >75% is acceptable

### Training Progress Indicators
- **Loss convergence**: Should decrease steadily
- **Validation accuracy**: Should increase then plateau
- **Overfitting signs**: Validation accuracy drops while training continues to improve

### Performance Comparison
Use tags for systematic comparison:
```bash
# Tag all runs for easy comparison
python src/train.py experiment=cifar10_benchmark_cnn tags="[benchmark,cifar10,cnn,v1]"
python src/train.py experiment=cifar10_benchmark_convnext tags="[benchmark,cifar10,convnext,v1]"
```

## 🔗 Integration

The benchmark suite integrates seamlessly with:
- **Lightning logging**: All metrics automatically tracked
- **Hydra configuration**: Reproducible experiment setup
- **Make targets**: Convenient command shortcuts
- **Testing framework**: Automated validation

For more details on architectures, see [README-ARCHITECTURES.md](README-ARCHITECTURES.md).
For make target reference, see [README-MAKEFILE.md](README-MAKEFILE.md).
