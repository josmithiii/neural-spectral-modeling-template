# VIMH Loss Functions

## Problem Statement

The original VIMH implementation used `CrossEntropyLoss` for quantized continuous
parameters (0-255), treating them as independent classes. This is suboptimal because:

1. **No distance awareness**: Predicting 101 vs 200 when the target is 100 receives equal penalty
2. **Discrete predictions**: Argmax collapses the ordinal structure into hard classes
3. **Poor generalization**: The model fails to learn that neighbouring values are more
   similar than distant ones

## VIMH Dataset Step Parameter Foundation

The VIMH dataset format stores parameter metadata with each parameter having:
- `min`: minimum value in perceptual units (e.g., 50.0 Hz)
- `max`: maximum value in perceptual units (e.g., 52.0 Hz)
- `step`: quantization step size in perceptual units (e.g., 0.00784 Hz)

This `step` parameter is fundamental to the loss function design:

**Quantization Structure**: `num_classes = 1 + round((max - min) / step)`
- Ensures the discrete class space properly represents the continuous parameter space
- Each class boundary corresponds to one perceptual step size

**Loss Scaling**: `quantization_step = param_range / (num_classes - 1)`
- Converts discrete class distances to perceptual units for interpretable loss values
- Enables direct comparison of loss magnitudes across parameters with different ranges

**JND-Based Weighting**: `loss_weight = param_range / step`
- Parameters with finer quantization (smaller steps) get higher loss weights
- Balances learning across parameters based on their perceptual resolution

## Supported Loss Functions

`src/train.py` relies on `src.models.losses.create_loss_function` to instantiate criteria
from Hydra configs. The factory understands standard PyTorch losses, the custom losses in
`src/models/losses.py`, and `SoftTargetLoss` in `src/models/soft_target_loss.py`.

- **CrossEntropyLoss** (`torch.nn.CrossEntropyLoss`): legacy classification heads;
  argmax predictions; no distance signal.  Used by default (no `output_mode`).
- **OrdinalRegressionLoss** (`src.models.losses.OrdinalRegressionLoss`): default for
  quantized continuous parameters; distance-aware and expressed in perceptual units.
- **QuantizedRegressionLoss** (`src.models.losses.QuantizedRegressionLoss`): lightweight
  regression over quantized bins; retains step awareness without softmax.
- **NormalizedRegressionLoss** (`src.models.losses.NormalizedRegressionLoss`): used when
  `output_mode=regression`; operates in `[0,1]` space and denormalizes to perceptual units.
- **WeightedCrossEntropyLoss** (`src.models.losses.WeightedCrossEntropyLoss`): keeps
  classification heads but adds a power-law distance penalty.
- **MultiScaleSpectralLoss** (`src.models.losses.MultiScaleSpectralLoss`): multi-resolution STFT distance for waveform or spectrogram comparisons.
- **SoftTargetLoss** (`src.models.soft_target_loss.SoftTargetLoss`): builds soft ordinal targets and optimizes KL divergence for smoother classification.

The sections below detail behaviour, configuration, and integration for each custom loss.

## 1. OrdinalRegressionLoss (Recommended)

**Best for**: VIMH datasets with quantized continuous parameters.

```python
from src.models.losses import OrdinalRegressionLoss

criterion = OrdinalRegressionLoss(
    num_classes=256,
    param_range=2.0,  # actual parameter range (max - min) in perceptual units
    regression_loss="l1",  # or "l2", "huber"
    alpha=0.1,  # classification regularization weight
)
```

**How it works**:

- Converts logits to probabilities with softmax
- Computes continuous prediction as weighted average: `pred = Σ(prob_i × class_center_i)`
- Calculates distance in quantization steps between prediction and target
- Converts to perceptual units: `loss = distance_steps * quantization_step`
  - Where `quantization_step = param_range / (num_classes - 1)`
  - This ensures loss values are in the same units as the original parameter
- Applies regression loss (L1/L2/Huber) in perceptual space
- Optionally adds cross-entropy regularization (`alpha > 0`)

**Benefits**:

- Distance-aware: closer predictions receive lower penalties
- Continuous predictions: outputs remain continuous rather than argmax bins
- Perceptual units: loss values equal parameter error in the target units
- Stable training: the optional classification term improves convergence
- Auto-configuration: `VIMHLitModule` injects parameter ranges from dataset metadata when
  `auto_configure_from_dataset=True`

## 2. QuantizedRegressionLoss

**Best for**: Lightweight regression on quantized targets when logits already encode a
single scalar.

```python
from src.models.losses import QuantizedRegressionLoss

criterion = QuantizedRegressionLoss(
    num_classes=256,
    param_range=2.0,
    loss_type="l1",  # or "l2", "huber"
)
```

**How it works**:

- Treats the model output as a single continuous value in range `[0, num_classes-1]`
- Clamps predictions to valid range
- Calculates distance in quantization steps between prediction and target
- Converts to perceptual units: `loss = distance_steps * quantization_step`
  - Uses the same `quantization_step = param_range / (num_classes - 1)` formula
- Applies regression loss (L1/L2/Huber) directly in perceptual space

**Benefits**:

- Simple drop-in replacement for scalar heads
- Maintains distance awareness without softmax
- Lower computational overhead than ordinal regression
- Auto-configuration updates `param_range` using the same metadata pipeline as
  OrdinalRegressionLoss

## 3. NormalizedRegressionLoss (Also Recommended)

**Best for**: Pure regression mode (`model.output_mode=regression`) where heads emit
sigmoid outputs in `[0, 1]`.

```python
from src.models.losses import NormalizedRegressionLoss

criterion = NormalizedRegressionLoss(
    param_range=(50.0, 52.0),  # (min, max) in perceptual units
    loss_type="mse",  # or "l1", "huber"
    return_perceptual_units=True,
)
```

**How it works**:

- Clamps predictions to `[0, 1]` and normalizes targets using `(min, max)` bounds
- Applies regression loss in normalized space
- Optionally scales the loss back into perceptual units (`return_perceptual_units=True`)

**Benefits**:

- Matches the regression heads created when `output_mode=regression`
- Auto-configured by `src/train.py` using dataset metadata (`cfg.model.criteria` receives
  (min, max) tuples)
- Supports consistent loss magnitudes across parameters with different ranges

## 4. WeightedCrossEntropyLoss

**Best for**: Maintaining the classification pipeline while adding ordinal penalties.

```python
from src.models.losses import WeightedCrossEntropyLoss

criterion = WeightedCrossEntropyLoss(
    num_classes=256,
    distance_power=2.0,  # higher power = stronger punishment for distant errors
    base_weight=1.0,
)
```

**How it works**:

- Computes standard cross entropy per sample
- Adds a power-law distance penalty: `|i - target| ** distance_power`
- Keeps predictions discrete (argmax)

**Benefits**:

- Minimal changes to legacy classification code
- Explicitly discourages large ordinal errors
- Plays nicely with accuracy metrics already in place

## 5. SoftTargetLoss

**Best for**: Classification heads that benefit from smoothed target distributions.

```python
from src.models.soft_target_loss import SoftTargetLoss

criterion = SoftTargetLoss(
    num_classes=256,
    mode="triangular",  # or "gaussian", "log-triangular"
    width=2,
    sigma=2.5,  # used for gaussian mode
)
```

**How it works**:

- Builds a soft target distribution around each label (triangular, logarithmic, or
  gaussian support)
- Uses KL divergence between model probabilities and the soft targets

**Benefits**:

- Reduces quantization artefacts when neighbouring bins should share probability mass
- Compatible with the factory by setting `_target_: src.models.soft_target_loss.SoftTargetLoss`
- Can be combined with accuracy metrics because predictions remain discrete

## 6. MultiScaleSpectralLoss

**Best for**: Audio or spectral outputs where multi-resolution STFT comparisons are meaningful.

```python
from src.models.losses import MultiScaleSpectralLoss

criterion = MultiScaleSpectralLoss(
    max_n_fft=2048,
    num_scales=6,
    hop_lengths=None,  # defaults to n_fft // 4 per scale
    p=1.0,
)
```

**How it works**:

- Builds several `MagnitudeSTFT` operators from `max_n_fft` down to smaller windows
- Computes an L1 or L2 distance at each scale and averages across scales

**Benefits**:

- Captures both fine and coarse spectral structure
- Proven implementation adapted from the PNP codebase for spectral analysis tasks
- Drop-in option for experiments that reconstruct audio directly

## Configuration Examples

### Original CrossEntropyLoss (legacy)

```yaml
# configs/model/cnn_medium.yaml
criteria:
  note_number:
    _target_: torch.nn.CrossEntropyLoss
  note_velocity:
    _target_: torch.nn.CrossEntropyLoss
```

### OrdinalRegressionLoss with metadata auto-update

```yaml
# configs/model/cnn_medium_ordinal.yaml
criteria:
  note_number:
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256
    param_range: 1.0  # placeholder; replaced at runtime from VIMH metadata
    regression_loss: l1
    alpha: 0.1
  note_velocity:
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256
    param_range: 1.0
    regression_loss: l1
    alpha: 0.1
```

### NormalizedRegressionLoss heads (regression mode)

```yaml
# configs/model/cnn_medium_regression.yaml
output_mode: regression
criteria:
  log10_decay_time:
    _target_: src.models.losses.NormalizedRegressionLoss
    param_range: [0.0, 1.0]  # (min, max); overwritten by metadata during train
    loss_type: mse
  wah_position:
    _target_: src.models.losses.NormalizedRegressionLoss
    param_range: [0.0, 1.0]
    loss_type: mse
```

### SoftTargetLoss for smoothed classification

```yaml
criteria:
  wah_position:
    _target_: src.models.soft_target_loss.SoftTargetLoss
    num_classes: 256
    mode: gaussian
    width: 4
    sigma: 3.0
```

## Simple Usage (Recommended)

The easiest way to use different loss functions is with the new `loss_type` parameter:

```bash
# OrdinalRegressionLoss (recommended for VIMH)
python src/train.py experiment=wah_cnn_tiny_ordinal trainer=mps

# All loss types available:
python src/train.py experiment=wah_cnn_tiny trainer=mps                    # cross_entropy
python src/train.py experiment=wah_cnn_tiny_regression trainer=mps         # normalized_regression
python src/train.py experiment=wah_cnn_tiny_ordinal trainer=mps            # ordinal_regression
python src/train.py experiment=wah_cnn_tiny_quantized trainer=mps          # quantized_regression
python src/train.py experiment=wah_cnn_tiny_weighted trainer=mps           # weighted_cross_entropy
python src/train.py experiment=wah_cnn_tiny_soft_target trainer=mps        # soft_target
```

## Advanced Usage (Explicit Configuration)

For custom configurations, you can still specify explicit `criteria:`:

```bash
# Enable ordinal losses on the wah tiny experiment
python src/train.py experiment=wah_cnn_tiny model=cnn_medium_ordinal trainer=mps

# Multi-scale spectral loss example
python src/train.py experiment=wah_cnn_tiny model=cnn_medium_ordinal trainer=mps \
  model.criteria.waveform._target_=src.models.losses.MultiScaleSpectralLoss
```

## Implementation Details

### Step Parameter Integration

The VIMH dataset's `step` parameter drives the entire loss function architecture:

1. **Dataset Loading**: `idx = int(round((actual_value - min) / step))` converts perceptual values to class indices
2. **Class Count Calculation**: `num_classes = 1 + round((max - min) / step)` ensures proper quantization coverage
3. **Loss Scaling**: `quantization_step = param_range / (num_classes - 1)` converts class distances to perceptual units
4. **JND Weighting**: `loss_weight = param_range / step` balances multi-head learning based on perceptual resolution

### Runtime Configuration

- `VIMHLitModule` updates loss instances with dataset metadata during `setup()`.
  `param_range` is refreshed for ordinal or quantized losses, and `(min, max)` bounds feed
  normalized regression.
- The `step` parameter from dataset metadata automatically determines `num_classes` and `quantization_step`
- `_compute_predictions` returns continuous outputs for regression-aware losses and argmax
  for classification, keeping metrics consistent.
- `create_loss_function` raises a clear `ValueError` if an unknown `_target_` appears, so
  typos surface early.

## Performance Comparison (Ordinal vs Cross-Entropy)

Based on validation results with the 16K resonarium dataset:

| Loss Function | Test Accuracy | Predictions | Distance Awareness | Loss Units |
| ------------- | ------------- | ----------- | ------------------ | ---------- |
| CrossEntropyLoss | ~0.5% | Discrete (argmax) | ❌ No | Arbitrary |
| OrdinalRegressionLoss | ~0.5% | Continuous | ✅ Yes | Perceptual |

Both achieve similar accuracy because the task is challenging, but ordinal regression:

- Penalizes distant errors more than close ones
- Produces continuous predictions that better match the underlying parameters
- Returns loss values directly interpretable as parameter error
- Generally improves generalization as training length and data scale increase

## Loss Function Comparison Example

```python
# Example parameter: frequency range 50.0-52.0 Hz, step=0.00784 Hz
# This gives: num_classes = 1 + round((52.0 - 50.0) / 0.00784) = 256
# And: quantization_step = 2.0 / (256 - 1) = 0.00784 Hz per class

# Target: class 100 (50.784 Hz), Predictions: classes [101, 105, 200]
# Distances: [1, 5, 100] quantization steps
# Perceptual distances: [0.00784, 0.0392, 0.784] Hz

# CrossEntropyLoss: all wrong answers penalized equally
# Loss: [6.84, 4.51, 6.23] - arbitrary units, no correlation with perceptual distance

# OrdinalRegressionLoss: distant errors penalized proportionally (perceptual units)
# Loss: [0.00784, 0.0392, 0.784] Hz - directly interpretable as frequency error
# A loss of 0.0392 means the prediction is ~0.04 Hz off target

# This demonstrates how the step parameter enables perceptually meaningful loss values
```

## Benefits of Step-Based Perceptual Units

1. **Direct interpretability**: Loss values equal parameter deviations (e.g., loss=0.05 ->
   0.05 units off in the original parameter space)
2. **Consistent learning rates**: One learning rate works across parameters with different ranges
   because losses are normalized to their perceptual scales
3. **Meaningful comparisons**: Loss values are comparable across heterogeneous heads
   since they're all in their respective perceptual units
4. **Physical intuition**: Easy to relate metrics to domain knowledge (e.g., "0.1 Hz error")
5. **Automatic configuration**: Parameter bounds and steps are pulled from dataset metadata,
   ensuring the quantization structure matches the data generation process
6. **JND-aware weighting**: Parameters with finer perceptual resolution (smaller steps)
   automatically receive higher loss weights, balancing multi-head learning

### Example: Mixed Parameter Ranges with Step-Based Weighting

```yaml
# Parameters with different ranges - all return loss in perceptual units
# Step parameter determines both quantization and loss weighting automatically

# Dataset metadata example:
# parameter_mappings:
#   frequency: {min: 440.0, max: 880.0, step: 1.72}    # 256 classes, coarse steps
#   amplitude: {min: 0.0, max: 1.0, step: 0.00392}     # 256 classes, fine steps

criteria:
  frequency:  # Range: 440 Hz, step: 1.72 Hz -> weight = 440/1.72 = 256
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256  # = 1 + round(440/1.72)
    param_range: 440.0  # auto-configured from metadata
    regression_loss: l1
  amplitude:  # Range: 1.0, step: 0.00392 -> weight = 1.0/0.00392 = 255
    _target_: src.models.losses.OrdinalRegressionLoss
    num_classes: 256  # = 1 + round(1.0/0.00392)
    param_range: 1.0  # auto-configured from metadata
    regression_loss: l1

# Final loss weights (normalized): {frequency: 1.0, amplitude: 0.996}
# Both parameters get nearly equal weighting despite vastly different ranges
# because they have similar perceptual resolution (256 steps each)
```

## Best Practices

- Prefer `OrdinalRegressionLoss` for quantized perceptual parameters unless you need pure
  regression.
- Allow metadata auto-configuration to set real parameter ranges (leave placeholders in configs).
- Start with `regression_loss="l1"` (ordinal) or `loss_type="mse"` (normalized
  regression) for robustness.
- Keep `alpha=0.1` for ordinal loss regularization unless tuning shows otherwise.
- Monitor both accuracy and loss: accuracy tracks order, loss reflects perceptual error.
- Compare against `CrossEntropyLoss` baselines when introducing new loss settings.

## Future Enhancements

- Custom distance-based metrics for evaluation
- Adaptive weighting between regression and classification terms
- Hybrid heads that mix ordinal and normalized regression losses
- Curriculum schedules that transition from soft targets to ordinal regression

This distance-aware loss framework provides a more appropriate approach for training on
quantized continuous parameters in VIMH datasets while still supporting regression and
spectral tasks when needed.
