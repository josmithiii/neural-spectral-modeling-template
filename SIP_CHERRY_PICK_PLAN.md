# SIP Cherry-Pick Plan

## Overview

This document identifies general-purpose image classification features from the `synthmatch-image-proc` (SIP) project that should be merged back into this upstream template. The SIP project started as a fork of this template and has developed several improvements that could benefit general ML practitioners.

## ✅ Completed Cherry-Pick Features

### Core Infrastructure ✅
- **✅ MultiScaleSpectralLoss**: PNP-proven multi-resolution spectral analysis loss function
- **✅ DistanceLoss**: Base class for distance-based losses with configurable norms
- **✅ create_loss_function**: Factory pattern for config-driven loss instantiation
- **✅ SoftTargetLoss**: General-purpose soft target loss for ordinal classification
- **✅ Enhanced loss testing**: Comprehensive test suite for loss functions (`test_loss_factory.py`)
- **✅ single_step_inference.py**: Utility for quick model inference testing
- **✅ Enhanced Makefile targets**: Added `dc/dclean` for data cleaning utilities
- **✅ Metadata improvements**: Removed deprecated 'scale' field from dataset metadata

### Perfect Synergy with Existing AVIX Features ✅
The SIP cherry-pick perfectly complements the existing AVIX enhancements:
- **AVIX provided**: SimpleMLP, gradient tracking, custom progress bars, dataset wrappers, auxiliary CNN features
- **SIP added**: Advanced loss functions, loss factory, inference utilities, general-purpose soft targets
- **Result**: Complete advanced ML research template with both training infrastructure and sophisticated loss functions

## Features to Cherry-Pick

### High Priority - Core Infrastructure

#### 1. **Enhanced Loss Functions (`src/models/losses.py`)**
- **MultiScaleSpectralLoss**: PNP-proven multi-resolution spectral analysis
  - Useful for any frequency-domain data (spectrograms, time-frequency representations)
  - Research-backed implementation from published ICASSP/TASLP papers
  - Superior gradient properties compared to basic MSE/L1
- **DistanceLoss**: Base class for distance-based losses with configurable norms
- **AuralossWrapper**: Generic wrapper for external loss libraries with spectrogram support
- **create_loss_function**: Factory pattern for config-driven loss instantiation

#### 2. **Testing Infrastructure**
- **test_multihead_loss_configs.py**: Framework for testing multiple loss configurations
- **test_vimh_format.py**: General VIMH dataset format validation (useful for multihead datasets)

#### 3. **Development Tools**
- **single_step_inference.py**: Utility for quick model inference testing
- **Enhanced Makefile targets**:
  - `dc/dclean`: Data cleaning utilities
  - Additional training shortcuts and abbreviations

### Medium Priority - Convenience Features

#### 4. **Data Handling Improvements**
- **custom_cifar100_datamodule.py**: Enhanced CIFAR-100 with custom preprocessing
- **Metadata improvements**: Better parameter range handling in dataset metadata
- **Enhanced VIMH utilities**: Improved VIMH dataset processing (keep general parts only)

#### 5. **Configuration Enhancements**
- **soft_target_loss.py**: Soft target loss implementations for improved training
- **Enhanced experiment configs**: Better experiment organization patterns
- **Improved defaults**: Better default configurations for common use cases

### Lower Priority - Optional Improvements

#### 6. **Documentation Enhancements**
- **Configuration documentation**: Improved docs on loss configuration patterns
- **Best practices**: Guidelines for multi-head and multi-loss training

### Features to EXCLUDE (Audio-Specific)

The following SIP features are **audio-specific** and should NOT be cherry-picked:

- All auraloss dependencies and integrations
- Audio synthesis utilities (`src/synth_simple.py`, `src/utils/synth_utils.py`)
- Audio-specific data modules (resonarium, STK datasets)
- Audio-specific model configurations (synthesis, auraloss-based)
- Audio-specific experiments and training targets
- Audio generation and display utilities (`generate_vimh.py`, `display_vimh.py`)
- Audio-specific documentation (`docs/auraloss_usage.md`)

## Implementation Strategy

### Phase 1: Core Loss Functions
1. Cherry-pick `MultiScaleSpectralLoss` and `DistanceLoss` from `src/models/losses.py`
2. Add `create_loss_function` factory pattern (without auraloss dependencies)
3. Update relevant tests

### Phase 2: Infrastructure
1. Add general-purpose testing patterns from `test_multihead_loss_configs.py`
2. Integrate `single_step_inference.py` utility
3. Update Makefile with general-purpose convenience targets

### Phase 3: Data and Configuration
1. Cherry-pick `custom_cifar100_datamodule.py` improvements
2. Add metadata range handling improvements
3. Update configuration documentation

### Phase 4: Optional Enhancements
1. Add `soft_target_loss.py` if needed
2. Improve default configurations based on SIP learnings
3. Update development workflow documentation

## Dependencies to Consider

### Safe to Add
- No new dependencies needed for core loss functions
- Standard PyTorch/Lightning dependencies sufficient

### Avoid Adding
- `auraloss` - audio-specific
- `torchaudio` - audio-specific
- Any audio synthesis libraries

## Benefits

1. **Better loss functions**: Research-proven spectral losses for frequency-domain data
2. **Improved testing**: Better patterns for multi-loss configuration testing
3. **Enhanced workflow**: More convenient development utilities
4. **Better defaults**: Improved configurations based on real-world usage
5. **Future-ready**: Patterns that support more complex training scenarios

## Notes

- The SIP project has proven these enhancements in real research use
- Focus on general-purpose computer vision and ML infrastructure
- Maintain backward compatibility with existing template functionality
- Keep the template clean and focused on image classification tasks
