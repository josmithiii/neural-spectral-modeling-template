# VIMH Loss Functions for Quantized Continuous Parameters

## Problem Statement

The original VIMH implementation used `CrossEntropyLoss` for quantized continuous parameters (0-255), treating them as independent classes. This is suboptimal because:

1. **No distance awareness**: Predicting 101 vs 200 when the target is 100 gets equal penalty
2. **Discrete predictions**: Uses argmax, losing continuity information
3. **Poor generalization**: Model doesn't learn the ordinal relationship between values

## Solution: Distance-Aware Loss Functions

### 1. OrdinalRegressionLoss (Recommended)

**Best for**: VIMH datasets with quantized continuous parameters

```python
from src.models.losses import OrdinalRegressionLoss

criterion = OrdinalRegressionLoss(
    num_classes=256,
    param_range=2.0,  # actual parameter range (max - min) in perceptual units
    regression_loss='l1',  # or 'l2', 'huber'
    alpha=0.1,  # classification regularization weight
)
```

**How it works**:
- Converts logits to probabilities using softmax
- Computes continuous prediction as weighted average: `pred = Σ(prob_i × class_center_i)`
- Maps quantized distance to actual parameter space (perceptual units)
- Applies regression loss (L1/L2/Huber) in perceptual space
- Returns loss in perceptual units (actual parameter range units)
- Optionally adds classification term for training stability

**Benefits**:
- Distance-aware: Closer predictions get lower penalties
- Continuous predictions: Output is continuous, not discrete
- Perceptual units: Loss values directly interpretable as parameter error
- Stable training: Classification term helps with convergence
- Flexible: Supports different regression losses and num_classes
- Auto-configured: Parameter ranges loaded automatically from dataset metadata

### 2. QuantizedRegressionLoss

**Best for**: Direct regression on quantized values

```python
from src.models.losses import QuantizedRegressionLoss

criterion = QuantizedRegressionLoss(
    num_classes=256,
    param_range=2.0,  # actual parameter range (max - min) in perceptual units
    loss_type='l1'  # or 'l2', 'huber'
)
```

**How it works**:
- Treats model output as single continuous value
- Applies regression loss directly to predictions
- Clamps predictions to valid range [0, num_classes-1]

**Benefits**:
- Simple and direct
- Pure regression approach
- Lower computational overhead

### 3. WeightedCrossEntropyLoss

**Best for**: Keeping classification framework with distance awareness

```python
from src.models.losses import WeightedCrossEntropyLoss

criterion = WeightedCrossEntropyLoss(
    num_classes=256,
    distance_power=2.0  # Higher = more penalty for distant errors
)
```

**How it works**:
- Weights classification errors by distance from target
- Distant predictions get exponentially higher penalties
- Maintains discrete predictions (argmax)

**Benefits**:
- Minimal changes to existing code
- Still uses classification metrics
- Distance-aware penalties

## Configuration Examples

### Original CrossEntropyLoss (Current)
```yaml
# configs/model/cnn_64k.yaml
criteria:
  note_number:
    _target_: torch.nn.CrossEntropyLoss
  note_velocity:
    _target_: torch.nn.CrossEntropyLoss
```

### New OrdinalRegressionLoss (Recommended)
```yaml
# configs/model/cnn_64k_ordinal.yaml
criteria:
  note_number:
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256
    param_range: 1.0  # Placeholder - auto-updated from dataset metadata
    regression_loss: l1
    alpha: 0.1
  note_velocity:
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256
    param_range: 1.0  # Placeholder - auto-updated from dataset metadata
    regression_loss: l1
    alpha: 0.1
```

## Usage

### Training with Ordinal Regression Loss

```bash
# Use the new distance-aware loss
python src/train.py experiment=cnn_16kdss_ordinal

# Or use the make target
make evimho
```

### Comparison with Original Loss

```bash
# Original classification loss
python src/train.py experiment=cnn_16kdss

# New distance-aware loss
python src/train.py experiment=cnn_16kdss_ordinal
```

## Implementation Details

### Model Changes

The `MultiheadLitModule` now supports both classification and regression predictions:

```python
def _compute_predictions(self, logits: torch.Tensor, criterion, head_name: str) -> torch.Tensor:
    if self._is_regression_loss(criterion):
        if isinstance(criterion, OrdinalRegressionLoss):
            # Weighted average of probabilities
            probs = F.softmax(logits, dim=1)
            class_centers = torch.arange(criterion.num_classes, device=logits.device)
            preds = torch.sum(probs * class_centers.unsqueeze(0), dim=1)
        else:
            # Direct regression output
            preds = logits.squeeze(-1)
    else:
        # Classification: argmax
        preds = torch.argmax(logits, dim=1)
    return preds
```

### Backward Compatibility

- Existing configurations continue to work unchanged
- Classification metrics (accuracy) still computed correctly
- Model architecture remains the same

## Performance Comparison

Based on test results with the 16K resonarium dataset:

| Loss Function | Test Accuracy | Predictions | Distance Awareness | Loss Units |
|---------------|---------------|-------------|-------------------|------------|
| CrossEntropyLoss | ~0.5% | Discrete (argmax) | ❌ No | Arbitrary |
| OrdinalRegressionLoss | ~0.5% | Continuous | ✅ Yes | Perceptual |

**Note**: Both show similar accuracy because the task is genuinely challenging. The key difference is that ordinal regression:
- Penalizes distant errors more than close ones
- Produces continuous predictions that better represent the underlying parameters
- Returns loss values in perceptual units (directly interpretable as parameter error)
- Should lead to better generalization with more training

## Loss Function Comparison Example

```python
# Target: 100, Predictions: [101, 105, 200]
# Distances: [1, 5, 100] quantization steps
# Parameter range: 2.0 units (e.g., 50-52 Hz)

# CrossEntropyLoss: All wrong answers penalized equally
# Loss: [6.84, 4.51, 6.23] - no correlation with distance

# OrdinalRegressionLoss: Distant errors penalized more (in perceptual units)
# Quantization step: 2.0/255 = 0.00784 units per step
# Loss: [0.00784, 0.0392, 0.784] - directly interpretable as parameter error
```

## Benefits of Perceptual Units

### Why Use Perceptual Units?

1. **Direct Interpretability**: Loss values directly represent parameter error (e.g., loss=0.05 means 0.05 units off)
2. **Consistent Learning Rates**: Same learning rate works across all parameters regardless of their ranges
3. **Meaningful Comparisons**: Loss values are comparable across different parameter types
4. **Physical Intuition**: Loss values correspond to actual parameter deviations
5. **Auto-Configuration**: Parameter ranges automatically loaded from dataset metadata

### Example: Mixed Parameter Ranges

```yaml
# Parameters with different ranges - all return loss in perceptual units
criteria:
  frequency:  # Range: 440-880 Hz (440 Hz range)
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256
    param_range: 440.0  # Auto-updated from dataset metadata
    regression_loss: l1
  amplitude:  # Range: 0.0-1.0 (1.0 range)
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256
    param_range: 1.0   # Auto-updated from dataset metadata
    regression_loss: l1
```

### Automatic Parameter Range Detection

The system automatically loads parameter ranges from dataset metadata:
- **VIMH datasets**: Ranges loaded from `vimh_dataset_info.json`
- **Auto-update**: Loss functions updated with actual parameter ranges during training
- **Fallback**: Uses placeholder value if metadata unavailable

## Best Practices

1. **Use OrdinalRegressionLoss** for VIMH datasets with quantized continuous parameters
2. **Let parameter ranges auto-configure** from dataset metadata
3. **Start with L1 regression loss** (robust to outliers)
4. **Set alpha=0.1** for classification regularization
5. **Monitor both accuracy and loss** to understand model behavior
6. **Compare with CrossEntropyLoss** to validate improvements
7. **Interpret loss values as parameter deviations** in perceptual units

## Future Enhancements

- **Custom metrics**: Add distance-based accuracy metrics
- **Adaptive weighting**: Learn optimal alpha during training
- **Multi-scale loss**: Combine losses at different scales
- **Curriculum learning**: Start with classification, gradually shift to regression

This distance-aware loss function framework provides a much more appropriate approach for training on quantized continuous parameters in VIMH datasets.
