# AVIX Cherry-Pick Plan

## Overview
This document outlines the general-purpose features from the AVIX project that should be cherry-picked back into the Lightning-Hydra-Template-Extended project. The AVIX project (Audio-Visual Index eXtraction) started as a fork of this template and added many enhancements for audio synthesis parameter prediction. We want to bring back the general-purpose improvements while excluding audio-specific features.

## Cherry-Pick Categories

### 1. ✅ **Architecture Enhancements**

#### 1.1 SimpleMLP Architecture
- **File**: `src/models/components/simple_mlp.py` (new file from AVIX)
- **Purpose**: MLP without BatchNorm for batch_size=1 scenarios
- **Benefits**: Provides flexibility when batch normalization isn't appropriate

#### 1.2 Hybrid CNN with Auxiliary Features
- **Files**: `src/models/components/simple_cnn.py` (modifications)
- **Changes**:
  - Added `auxiliary_input_size` and `auxiliary_hidden_size` parameters
  - Support for combining CNN features with scalar auxiliary inputs
  - Feature fusion before classification heads
- **General Purpose**: Useful for any multimodal learning task combining images with metadata

#### 1.3 Vision Transformer Implementation Choice
- **File**: `src/models/components/vision_transformer.py`
- **Feature**: `use_torch_layers` parameter to switch between educational and production implementations
- **Benefits**: Flexibility between learning/debugging and production performance

### 2. ✅ **Training and Optimization Improvements**

#### 2.1 Gradient Statistics Callback
- **Files**:
  - `src/utils/gradient_stats_callback.py` (new)
  - `configs/callbacks/gradient_stats.yaml` (new)
- **Purpose**: Track gradient statistics during training for optimization analysis
- **Features**:
  - Signal-to-noise ratio tracking
  - Per-layer gradient norms
  - Stability metrics across epochs
- **General Purpose**: Valuable debugging tool for any deep learning project

#### 2.2 Enhanced Progress Bar
- **Files**:
  - `src/utils/custom_progress_bar.py` (new)
  - `configs/callbacks/grouped_progress_bar.yaml` (new)
  - `configs/callbacks/grouped_progress_bar_with_gradients.yaml` (new)
- **Purpose**: Better progress visualization with grouped metrics
- **Benefits**: Cleaner training output, especially for multihead models

#### 2.3 Default TensorBoard Logging
- **File**: `configs/train.yaml`
- **Change**: Enable TensorBoard by default instead of null logger
- **Benefits**: Immediate gradient tracking and visualization support

### 3. ✅ **Data Handling Enhancements**

#### 3.1 Soft Target Support for VIMH
- **Files**: `src/data/vimh_dataset.py`, `src/data/vimh_datamodule.py`
- **Feature**: `target_width` parameter for Gaussian soft targets
- **Benefits**:
  - Reduces quantization boundary artifacts
  - Better generalization for discretized continuous values
  - Backward compatible (target_width=0.0 for hard targets)

#### 3.2 Flexible Transforms
- **File**: `src/data/flexible_transforms.py` (new)
- **Purpose**: Dynamic transform adjustment for different image sizes
- **General Purpose**: Useful for any variable-dimension dataset

#### 3.3 Dataset Wrapper Utilities
- **File**: `src/data/dataset_wrapper.py` (new)
- **Purpose**: Generic dataset wrapping utilities
- **Benefits**: Easier dataset manipulation and preprocessing

### 4. ✅ **Configuration System Improvements**

#### 4.1 DataLoader Warning Suppression
- **File**: `configs/extras/default.yaml`
- **Feature**: `ignore_dataloader_warnings` flag
- **Purpose**: Suppress num_workers warnings (especially for MPS)

#### 4.2 Enhanced Trainer Defaults
- **File**: `configs/trainer/default.yaml`
- **Addition**: `log_every_n_steps: 10` for detailed metric tracking

#### 4.3 Progress Bar Configuration Fix
- **File**: `configs/callbacks/rich_progress_bar.yaml`
- **Change**: Switch from RichProgressBar to standard ProgressBar
- **Reason**: Fixes Rich console compatibility issues

### 5. ✅ **Development Workflow Improvements**

#### 5.1 Additional Make Targets
- **File**: `Makefile`
- **New targets to cherry-pick**:
  - `clean-data`: Clean synthesized datasets
  - `format`: Run pre-commit formatting
  - Architecture comparison targets
  - Tensorboard launch helpers

#### 5.2 CLAUDE.md Updates
- **File**: `CLAUDE.md`
- **Additions**:
  - Cleaning and maintenance commands
  - `make help` documentation note
  - Updated architecture descriptions (SimpleDenseNet as MLP clarification)

### 6. ❌ **Audio-Specific Features to EXCLUDE**

These should NOT be cherry-picked as they are domain-specific:

- AVIX module and datamodule (`avix_module.py`, `avix_datamodule.py`, `avix_dataset.py`)
- JND (Just-Noticeable Difference) loss functions and accuracy metrics
- Direct Gradient loss for audio synthesis parameters
- Audio-specific feature extractors (`spectrogram_features.py`, `auxiliary_features.py` for audio)
- SimpleSawSynth and other audio synthesizer components
- All AVIX-specific experiment configs (`configs/experiment/avix_*.yaml`)
- AVIX-specific model configs (`configs/model/avix_*.yaml`)
- Audio-specific data configs (`configs/data/avix_*.yaml`)

## Implementation Approach

Since we're working with diffs rather than commits, the implementation approach will be:

1. **Manual File Copy**: For new files, copy them directly from AVIX
2. **Selective Patching**: For modified files, manually apply relevant changes
3. **Testing**: Run existing tests to ensure backward compatibility
4. **Documentation**: Update docs to reflect new capabilities

## Priority Order

1. **High Priority** (Core improvements):
   - Gradient statistics callback
   - Soft target support
   - Hybrid CNN auxiliary features
   - SimpleMLP architecture

2. **Medium Priority** (Quality of life):
   - Progress bar improvements
   - Configuration fixes
   - Make target additions
   - Default TensorBoard logging

3. **Low Priority** (Nice to have):
   - Dataset wrapper utilities
   - Flexible transforms
   - Documentation updates

## Testing Checklist

After cherry-picking, verify:
- [ ] All existing tests pass
- [ ] MNIST training still works
- [ ] CIFAR benchmarks run correctly
- [ ] Multihead training functions properly
- [ ] VIMH datasets load correctly
- [ ] New gradient statistics callback works
- [ ] Soft targets work with backward compatibility
- [ ] Hybrid CNN with auxiliary features trains

## Notes

- The AVIX project has good separation between audio-specific and general-purpose code
- Most improvements are additive and maintain backward compatibility
- The gradient tracking features are particularly valuable for any deep learning research
- The soft target support is general-purpose despite being motivated by audio quantization
