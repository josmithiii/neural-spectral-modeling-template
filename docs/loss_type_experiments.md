# Loss Type Experiments

This document explains the different loss function experiments available in the wah_cnn_tiny series.

## Overview

The NSMT template includes 6 wah_cnn_tiny experiments, each demonstrating a different loss function approach for quantized continuous parameters:

| Experiment | Loss Type | Use Case | Key Benefit |
|------------|-----------|----------|-------------|
| `wah_cnn_tiny` | CrossEntropyLoss | Classification baseline | Standard approach, discrete predictions |
| `wah_cnn_tiny_regression` | NormalizedRegressionLoss | Pure regression | Continuous outputs in [0,1] space |
| `wah_cnn_tiny_ordinal` | OrdinalRegressionLoss | **Recommended** | Distance-aware, perceptual units |
| `wah_cnn_tiny_quantized` | QuantizedRegressionLoss | Lightweight regression | Simpler than ordinal, still distance-aware |
| `wah_cnn_tiny_weighted` | WeightedCrossEntropyLoss | Enhanced classification | Keeps discrete predictions + distance penalty |
| `wah_cnn_tiny_soft_target` | SoftTargetLoss | Smooth classification | KL divergence with soft target distributions |

## Usage

```bash
# Run individual experiments
make ewt         # CrossEntropyLoss (standard classification)
make ewtr        # NormalizedRegressionLoss (pure regression)
python src/train.py experiment=wah_cnn_tiny_ordinal    # OrdinalRegressionLoss (recommended)
python src/train.py experiment=wah_cnn_tiny_quantized  # QuantizedRegressionLoss
python src/train.py experiment=wah_cnn_tiny_weighted   # WeightedCrossEntropyLoss
python src/train.py experiment=wah_cnn_tiny_soft_target # SoftTargetLoss

# Compare all loss types (will run for ~30 minutes total on MPS)
for exp in wah_cnn_tiny wah_cnn_tiny_regression wah_cnn_tiny_ordinal wah_cnn_tiny_quantized wah_cnn_tiny_weighted wah_cnn_tiny_soft_target; do
  echo "Running $exp..."
  python src/train.py experiment=$exp trainer=mps
done
```

## Loss Function Details

### 1. CrossEntropyLoss (Baseline)
- **Experiment**: `wah_cnn_tiny`
- **Output**: Discrete classes (argmax)
- **Metrics**: Accuracy
- **Use when**: Need baseline comparison or discrete outputs

### 2. NormalizedRegressionLoss (Pure Regression)
- **Experiment**: `wah_cnn_tiny_regression`
- **Output**: Continuous values in [0,1]
- **Metrics**: MAE in perceptual units
- **Use when**: Want true regression with sigmoid outputs

### 3. OrdinalRegressionLoss (Recommended)
- **Experiment**: `wah_cnn_tiny_ordinal`
- **Output**: Continuous (weighted average of class probabilities)
- **Metrics**: Accuracy + loss in perceptual units
- **Use when**: Want distance-aware classification with continuous predictions
- **Benefits**: Best of both worlds - classification structure + regression continuity

### 4. QuantizedRegressionLoss (Lightweight)
- **Experiment**: `wah_cnn_tiny_quantized`
- **Output**: Continuous (clamped to [0, num_classes-1])
- **Metrics**: Accuracy + loss in perceptual units
- **Use when**: Want simpler distance-aware regression without softmax overhead

### 5. WeightedCrossEntropyLoss (Enhanced Classification)
- **Experiment**: `wah_cnn_tiny_weighted`
- **Output**: Discrete classes (argmax)
- **Metrics**: Accuracy with distance penalties
- **Use when**: Must keep discrete outputs but want distance awareness

### 6. SoftTargetLoss (Smooth Classification)
- **Experiment**: `wah_cnn_tiny_soft_target`
- **Output**: Discrete classes (argmax)
- **Metrics**: Accuracy with smoother training
- **Use when**: Want to reduce quantization artifacts in classification

## Configuration Examples

Each loss type is configured through explicit `criteria:` sections in model configs:

```yaml
# OrdinalRegressionLoss (configs/model/cnn_tiny_ordinal.yaml)
criteria:
  log10_decay_time:
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256
    param_range: 1.0  # Auto-configured from dataset metadata
    regression_loss: l1
    alpha: 0.1

# SoftTargetLoss (configs/model/cnn_tiny_soft_target.yaml)
criteria:
  wah_position:
    _target_: src.models.soft_target_loss.SoftTargetLoss
    num_classes: 256
    mode: triangular
    width: 2
```

## Expected Performance

All experiments use the same CNN architecture (~40K parameters) and dataset (16K samples), so differences reflect loss function effectiveness:

- **CrossEntropyLoss**: ~94% accuracy, no distance awareness
- **NormalizedRegressionLoss**: ~0.017 MAE, continuous outputs
- **OrdinalRegressionLoss**: ~94% accuracy + continuous predictions + distance awareness
- **QuantizedRegressionLoss**: Similar to ordinal but with lower computational overhead
- **WeightedCrossEntropyLoss**: ~94% accuracy with reduced large-error penalties
- **SoftTargetLoss**: ~94% accuracy with smoother probability distributions

## When to Use Each Loss Type

### Choose OrdinalRegressionLoss when:
- Working with quantized continuous parameters (most common case)
- Want both classification accuracy metrics and continuous predictions
- Need loss values interpretable as parameter errors
- This is the **recommended default** for VIMH datasets

### Choose NormalizedRegressionLoss when:
- Want pure regression outputs in normalized [0,1] space
- Working with parameters that have very different ranges
- Need sigmoid-activated outputs for downstream processing

### Choose CrossEntropyLoss when:
- Need baseline comparison
- Working with truly discrete categories (not quantized continuous)
- Downstream code expects discrete class predictions

### Choose QuantizedRegressionLoss when:
- Want distance awareness with minimal computational overhead
- Working with scalar outputs rather than classification heads
- Need simpler alternative to ordinal regression

### Choose WeightedCrossEntropyLoss when:
- Must maintain discrete outputs for compatibility
- Want some distance awareness without changing prediction format
- Working with existing classification-based evaluation code

### Choose SoftTargetLoss when:
- Want to reduce quantization artifacts in classification
- Training with noisy or uncertain labels
- Need smoother probability distributions during training

## Implementation Notes

- All loss functions auto-configure parameter ranges from dataset metadata
- Loss values in ordinal/quantized regression are in perceptual units (directly interpretable)
- Regression losses work with both classification and regression head architectures
- All experiments use identical training hyperparameters for fair comparison

For detailed loss function mathematics and configuration options, see [docs/vimh_loss_functions.md](vimh_loss_functions.md).