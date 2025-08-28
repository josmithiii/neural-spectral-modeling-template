# LHTE Tutorial Example Sequence: From Setup to Advanced Experiments

This tutorial example sequence provides hands-on learning using the
Lightning Hydra Template Extended (LHTE). Follow these experiments in
order to explore the capabilities of modern deep learning
architectures on various datasets.

## Prerequisites

```bash
# Ensure you're in the project root and environment is activated
source .venv/bin/activate.csh  # or your preferred activation method
```

---

## 🚀 Phase 1: Quick Start and Environment Validation

### Example 1: Basic Setup Verification

**Objective**: Verify your environment is working correctly.

```bash
# Run the basic test suite
make t

# Clean any previous outputs
make c

# Test the simplest architecture quickly
make tq
```

**Expected `make tq` Results**:
- Tests pass without errors
- SimpleDenseNet trains for 1 epoch (~30 seconds)
- Final accuracy around 56-60% on MNIST

> **Make Note:** All make targets have a short and a long form, and we're using the short forms here.
> For example, the above examples could be typed as `make test`, `make clean`, and `make train-quick`.
> For a complete list of make targets (auto-generated), say `make h` (or `make help`).
> For a hopefully up-to-date more organized list (thanks Claude), see [makefile.md](./makefile.md) .

**What was Illustrated**: Environment setup, basic PyTorch Lightning workflow

---

### Example 2: Architecture Quick Tour

**Objective**: Compare all architectures with minimal time investment.

```bash
# Test all architectures with 1 epoch each (~3-4 minutes total)
make tqa
```

**Expected Results**:
- SimpleDenseNet: ~56% accuracy
- SimpleCNN: ~74% accuracy
- ViT: ~65% accuracy
- ConvNeXt: ~68% accuracy

**What was Illustrated**: Relative architecture performance, parameter efficiency

---

### Example 3: Architecture Deep Dive Comparison

**Objective**: Fair comparison with equal training time.

```bash
# Compare architectures with 3 epochs each (~10 minutes total)
make ca
```

**Expected Results**:
- Better convergence patterns visible
- CNN shows consistent improvement
- ConvNeXt demonstrates efficiency
- Check `logs/` directory for detailed results

**What was Illustrated**: Training dynamics, convergence patterns, logging system

---

## 🎯 Phase 2: MNIST Mastery

### Example 4: MNIST CNN Deep Dive

**Objective**: Achieve high performance on MNIST with CNN.

```bash
# Single-head CNN experiment (~5 minutes)
make ecm
```

**Expected Results**: ~99.1% test accuracy
**What was Illustrated**: Full training pipeline, checkpoint saving, metric tracking

---

### Example 5: State-of-the-Art MNIST with ViT

**Objective**: Achieve 99.5% accuracy on MNIST using Vision Transformers.

```bash
# SOTA ViT experiment (~15 minutes)
make ev995
```

**Expected Results**: 99.5% validation accuracy
**What was Illustrated**: ViT scaling, attention mechanisms, hyperparameter optimization

---

### Example 6: Multi-Head Learning on MNIST

**Objective**: Learn multiple tasks simultaneously.

```bash
# Multi-head CNN: digit classification + thickness + smoothness
make emhcm
```

**Expected Results**:
- Digit classification: ~99.1%
- Thickness classification: ~99.2%
- Smoothness classification: ~99.2%

**What was Illustrated**: Multi-task learning, shared representations, multiple loss functions

---

## 🖼️ Phase 3: CIFAR Computer Vision Benchmarks

### Example 7: CIFAR-10 Quick Validation

**Objective**: Rapid validation of CIFAR-10 capabilities.

```bash
# Quick CIFAR-10 tests across architectures (~15 minutes)
make cbqa
```

**Expected Results**:
- 5-epoch results for sanity checking
- CNN: ~45%, ConvNeXt: ~42%
- Validates data loading and basic training

**What was Illustrated**: CIFAR-10 complexity vs MNIST, architecture scaling

---

### Example 8: CIFAR-10 CNN Benchmark

**Objective**: Achieve literature-competitive results on CIFAR-10.

```bash
# Full CIFAR-10 CNN benchmark (~15 minutes)
make cb10c
```

**Expected Results**: 85-92% test accuracy
**What was Illustrated**: Data augmentation, regularization, longer training schedules

---

### Example 9: CIFAR-10 ConvNeXt Excellence

**Objective**: State-of-the-art CIFAR-10 performance.

```bash
# CIFAR-10 ConvNeXt benchmark (~18 minutes)
make cb10cn
```

**Expected Results**: 90-95% test accuracy
**What was Illustrated**: Modern CNN architectures, Global Response Normalization, efficiency

---

### Example 10: CIFAR-10 Architecture Comparison

**Objective**: Systematic comparison across all architectures.

```bash
# Run all CIFAR-10 benchmarks
make cbs10
```

**Expected Results**:
- CNN: 85-92%
- ConvNeXt: 90-95% (best)
- ViT: 88-93%
- EfficientNet: 89-94%

**What was Illustrated**: Architecture strengths, parameter vs accuracy tradeoffs

---

## 🎓 Phase 4: Advanced CIFAR-100 Challenges

### Example 11: CIFAR-100 Fine-Grained Classification

**Objective**: Tackle the challenging 100-class problem.

```bash
# CIFAR-100 CNN baseline
make cb100c
```

**Expected Results**: 55-70% test accuracy (much harder than CIFAR-10!)
**What was Illustrated**: Fine-grained classification challenges, class imbalance

---

### Example 12: CIFAR-100 ConvNeXt Mastery

**Objective**: Best performance on CIFAR-100.

```bash
# CIFAR-100 ConvNeXt benchmark
make cb100cn
```

**Expected Results**: 70-80% test accuracy
**What was Illustrated**: Architecture advantages on complex datasets

---

### Example 13: CIFAR-100 Coarse-to-Fine Hierarchy

**Objective**: Leverage hierarchical structure.

```bash
# CIFAR-100 coarse classification (20 superclasses)
make cb100cc
```

**Expected Results**: 75-85% accuracy on coarse classes
**What was Illustrated**: Hierarchical classification, label structure exploitation

---

## 🎵 Phase 5: Audio and VIMH Advanced Topics

### Example 14: VIMH Audio Synthesis Parameter Prediction

**Objective**: Predict synthesis parameters from audio spectrograms.

```bash
# VIMH CNN with 16K samples
make evimh
```

**Expected Results**: Regression performance on note number and velocity
**What was Illustrated**: Regression heads, continuous parameter prediction, audio ML

---

### Example 15: VIMH Ordinal Regression

**Objective**: Use distance-aware loss for ordered parameters.

```bash
# VIMH ordinal regression experiment
make evimho
```

**Expected Results**: Better performance on ordered parameters
**What was Illustrated**: Ordinal regression, distance-aware losses, parameter relationships

---

### Example 16: VIMH Pure Regression

**Objective**: Direct continuous parameter prediction.

```bash
# VIMH pure regression experiment
make evimhr
```

**Expected Results**: Smooth parameter predictions
**What was Illustrated**: Regression vs classification, continuous output spaces

---

## 📊 Phase 6: Comprehensive Analysis

### Example 17: Model Architecture Visualization

**Objective**: Understand model structures visually.

```bash
# Generate architecture diagrams
make td

# Generate all architecture diagrams
make tda

# Generate simple text diagrams for comparison
make tdss
```

**What was Illustrated**: Model complexity visualization, parameter distribution, computational graphs

---

### Example 18: Systematic Benchmarking

**Objective**: Complete performance evaluation.

```bash
# Run complete benchmark suite (takes ~2-3 hours)
make cbsa
```

**Expected Results**: Comprehensive performance matrix across all architectures and datasets
**What was Illustrated**: Systematic evaluation, performance patterns, architecture selection

---

### Example 19: Multi-Head CIFAR-10 Exploration

**Objective**: Advanced multi-task learning.

```bash
# Multi-head CNN on CIFAR-10
make emhcc10
```

**Expected Results**: Simultaneous object classification + auxiliary tasks
**What was Illustrated**: Multi-task computer vision, auxiliary task design

---

## 🔬 Phase 7: Research and Development

### Example 20: Custom Configuration Experiments

**Objective**: Learn to create custom experiments.

```bash
# Look at experiment configurations
ls configs/experiment/

# Run specific experiment configurations
python src/train.py experiment=cifar10_benchmark_vit

# Override specific parameters
python src/train.py experiment=cifar10_benchmark_cnn trainer.max_epochs=100 model.lr=0.001
```

**What was Illustrated**: Hydra configuration system, hyperparameter overrides, experiment design

---

### Example 21: Performance Optimization

**Objective**: Optimize training for your hardware.

```bash
# Use MPS on Mac (nearly always used by user per CLAUDE.md)
python src/train.py trainer=mps

# Experiment with batch sizes
python src/train.py data.batch_size=128 trainer=mps

# Try different optimizers
python src/train.py model.optimizer.lr=0.01 model.optimizer.weight_decay=1e-4
```

**What was Illustrated**: Hardware acceleration, batch size effects, optimizer tuning

---

### Example 22: Advanced VIMH Dataset Creation

**Objective**: Create your own VIMH datasets.

```bash
# Look at existing VIMH data
ls data/

# Examine VIMH structure
python examples/vimh_training.py --demo --save-plots
```

**What was Illustrated**: Dataset format design, multi-parameter learning, self-describing data

---

## 📚 Appendix: Troubleshooting Common Issues

### Environment Issues
```bash
# If you see "No module named 'rootutils'"
source .venv/bin/activate.csh

# Clean slate
make c && make cl
```

### Training Issues
```bash
# For MPS/Mac users (most common setup)
python src/train.py trainer=mps data.num_workers=0

# Memory issues - reduce batch size
python src/train.py data.batch_size=32
```

### Debugging
```bash
# Run tests to verify everything works
make t

# Quick sanity check
make tq

# Check logs
ls logs/train/runs/
```

---

## 🎯 Examples Summary

**Beginner (Examples 1-6)**: Environment, basic architectures, MNIST mastery
**Intermediate (Examples 7-13)**: CIFAR benchmarks, architecture comparison
**Advanced (Examples 14-19)**: VIMH audio ML, systematic benchmarking
**Expert (Examples 20-22)**: Custom experiments, optimization, dataset creation

---

## 📖 Additional Resources

- **Architecture Details**: See `docs/architectures.md`
- **Benchmark Analysis**: See `docs/benchmarks.md`
- **VIMH Format**: See `docs/vimh.md`
- **Make Targets**: See `docs/makefile.md`
- **Configuration**: See `docs/configuration.md`
