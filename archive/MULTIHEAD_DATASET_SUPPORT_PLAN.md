# Multihead Dataset Support Plan: VIMH Format

## Overview

This plan outlines the implementation of support for a new generalized
multihead dataset format called VIMH based on the previous
CIFAR-100-MH support. The format extends traditional single-label
datasets to support multiple classification heads with arbitrary
metadata.

## Current Status Analysis

### Existing Multihead Implementation
- **Current approach**: Synthetic label generation from single labels (e.g., MNIST digit → thickness + smoothness)
- **Limitation**: Only supports synthetic auxiliary tasks, not true multihead datasets
- **Scope**: Limited to predefined transformations

### Proposed Format Innovation
- **Novel approach**: Binary format with explicit metadata for multiple real labels
- **Advantage**: Supports arbitrary numbers of heads with real (not synthetic) labels
- **Generalization**: Includes image dimensions and channel information

## Dataset Format Specification

### Core Format Structure
```
[metadata_bytes][image_data][label_data]
```

### Metadata Specification (Generalized)
```
[height] [width] [channels] [N] [param1_id] [param1_val] [param2_id] [param2_val] ... [paramN_id] [paramN_val]
```

Where:
- `height`, `width`, `channels`: Image dimensions (1 byte each, 0-255)
- `N`: Number of classification heads (1 byte, 0-255)
- `param_id`: Parameter/head identifier (1 byte, 0-255)
- `param_val`: Parameter/head value (1 byte, 0-255)

### VIMH Example
```
[32] [32] [3] [2] [note_number_id] [note_number_val] [note_velocity_id] [note_velocity_val]
```

### Image Data
- Raw pixel data following metadata
- Size: `height × width × channels` bytes
- Format: Sequential pixel values (0-255)

## Implementation Plan

### Phase 1: Core Dataset Infrastructure

#### 1.1 Create Base Multihead Dataset Class
**File**: `src/data/multihead_dataset_base.py`

```python
class MultiheadDatasetBase(Dataset):
    """Base class for multihead datasets with arbitrary format support."""

    def __init__(self, data_path: str, metadata_format: Dict[str, Any])
    def parse_metadata(self, metadata_bytes: bytes) -> Dict[str, int]
    def parse_image_data(self, image_bytes: bytes, dims: Tuple[int, int, int]) -> torch.Tensor
    def parse_label_data(self, label_bytes: bytes, num_heads: int) -> Dict[str, int]
    def validate_format(self) -> bool
```

#### 1.2 Create VIMH Dataset Implementation
**File**: `src/data/vimh_dataset.py`

```python
class VIMHDataset(MultiheadDatasetBase):
    """VIMH format dataset implementation."""

    def __init__(self, data_path: str, train: bool = True)
    def _load_metadata_config(self) -> Dict[str, Any]
    def _parse_sample(self, sample_bytes: bytes) -> Tuple[torch.Tensor, Dict[str, int]]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]
    def __len__(self) -> int
```

#### 1.3 Create Generic Multihead Dataset Loader
**File**: `src/data/generic_multihead_dataset.py`

```python
class GenericMultiheadDataset(MultiheadDatasetBase):
    """Generic multihead dataset for arbitrary formats."""

    def __init__(self, data_path: str, format_config: Dict[str, Any])
    def _auto_detect_format(self) -> Dict[str, Any]
    def _validate_config(self, config: Dict[str, Any]) -> bool
```

### Phase 2: Data Module Integration

#### 2.1 Create VIMH Data Module
**File**: `src/data/vimh_datamodule.py`

```python
class VIMHDataModule(LightningDataModule):
    """Lightning data module for VIMH datasets."""

    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 0)
    def prepare_data(self) -> None
    def setup(self, stage: Optional[str] = None) -> None
    def train_dataloader(self) -> DataLoader
    def val_dataloader(self) -> DataLoader
    def test_dataloader(self) -> DataLoader
    def _multihead_collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

#### 2.2 Update Existing Data Module Factory
**File**: `src/data/__init__.py`
- Add imports for new multihead datasets
- Register VIMH data module

### Phase 3: Model Configuration Support

#### 3.1 Create Model Configurations
**Files**:
- `configs/data/vimh.yaml`
- `configs/model/vimh_cnn_*.yaml`
- `configs/model/vimh_convnext_*.yaml`
- `configs/model/vimh_efficientnet_*.yaml`
- `configs/model/vimh_vit_*.yaml`

#### 3.2 Create Experiment Configurations
**Files**:
- `configs/experiment/vimh_cnn.yaml`
- `configs/experiment/vimh_convnext.yaml`
- `configs/experiment/vimh_efficientnet.yaml`
- `configs/experiment/vimh_vit.yaml`

### Phase 4: Lightning Module Enhancement

#### 4.1 Update MNISTLitModule for Generic Multihead
**File**: `src/models/mnist_module.py`
- Rename to `multihead_module.py` or create new `generic_multihead_module.py`
- Remove MNIST-specific assumptions
- Support arbitrary head configurations from dataset metadata

#### 4.2 Dynamic Head Configuration
```python
class GenericMultiheadModule(LightningModule):
    """Generic Lightning module for multihead classification."""

    def __init__(self, net: torch.nn.Module, heads_config: Dict[str, int], ...)
    def _auto_configure_from_dataset(self, dataset: MultiheadDatasetBase) -> Dict[str, int]
    def _setup_metrics(self) -> None
    def _setup_criteria(self) -> None
```

### Phase 5: Testing and Validation

#### 5.1 Unit Tests
**File**: `tests/test_multihead_datasets.py`

```python
def test_vimh_dataset_loading()
def test_generic_multihead_format_detection()
def test_metadata_parsing()
def test_image_reconstruction()
def test_label_extraction()
def test_dataloader_integration()
```

#### 5.2 Integration Tests
**File**: `tests/test_multihead_training.py`

```python
def test_vimh_training_pipeline()
def test_model_head_configuration()
def test_loss_computation()
def test_metric_tracking()
```

#### 5.3 Format Validation Tests
**File**: `tests/test_format_validation.py`

```python
def test_format_autodetection()
def test_corrupted_data_handling()
def test_metadata_validation()
def test_dimension_consistency()
```

### Phase 6: Documentation and Examples

#### 6.1 Create Format Documentation
**File**: `docs/multihead_dataset_format.md`
- Detailed format specification
- Binary layout examples
- Metadata schema documentation
- Extension guidelines

#### 6.2 Create Usage Examples
**File**: `examples/vimh_training.py`
- Complete training pipeline example
- Data loading demonstration
- Model configuration examples

#### 6.3 Update Main Documentation
**Files**:
- `README.md`: Add multihead dataset support section
- `docs/extensions.md`: Document new multihead capabilities
- `docs/multihead.md`: Expand with new format information

## Advanced Features (Future Phases)

### Phase 7: Format Extensions

#### 7.1 Variable-Length Metadata Support
- Support for datasets with varying metadata lengths
- Dynamic parsing based on header information

#### 7.2 Compression Support
- Optional compression for image data
- Metadata-driven decompression

#### 7.3 Multi-Resolution Support
- Support for datasets with varying image dimensions
- Automatic resizing and padding

### Phase 8: Tools and Utilities

#### 8.1 Dataset Conversion Tools
**File**: `tools/convert_to_multihead.py`
- Convert existing datasets to multihead format
- Support for CIFAR-10/100, MNIST, custom datasets

#### 8.2 Format Validation Tools
**File**: `tools/validate_multihead_dataset.py`
- Comprehensive dataset validation
- Format compliance checking
- Statistics generation

#### 8.3 Dataset Creation Tools
**File**: `tools/create_multihead_dataset.py`
- Create new multihead datasets from images and labels
- Batch processing capabilities
- Metadata generation

## Implementation Priority

### High Priority (Phase 1-3)
1. Core dataset classes and VIMH implementation
2. Data module integration
3. Basic model configurations

### Medium Priority (Phase 4-5)
1. Lightning module generalization
2. Comprehensive testing
3. Format validation

### Low Priority (Phase 6-8)
1. Documentation and examples
2. Advanced features
3. Utility tools

## Technical Considerations

### Performance Optimizations
- Memory-mapped file access for large datasets
- Lazy loading of image data
- Efficient batch collation
- Caching for frequently accessed metadata

### Error Handling
- Robust parsing with detailed error messages
- Graceful degradation for corrupted data
- Format version compatibility checking

### Backwards Compatibility
- Maintain existing synthetic multihead support
- Provide migration path for existing configs
- Clear separation between synthetic and real multihead modes

## Existing Format Comparison

### Traditional Formats
- **CIFAR-10/100**: Single label per image, fixed dimensions
- **ImageNet**: Single label, variable dimensions, external metadata
- **COCO**: Multiple objects, JSON annotations, complex metadata

### Proposed VIMH Advantages
- **Embedded metadata**: Self-describing format
- **Multiple real labels**: Not synthetic/derived
- **Compact binary**: Efficient storage and loading
- **Arbitrary dimensions**: Flexible image sizes
- **Extensible**: Easy to add new label types

### Novel Aspects
This format appears to be novel in combining:
1. Embedded image dimensions in binary format
2. Multiple real (not synthetic) labels
3. Self-describing metadata structure
4. Compact binary representation
5. Arbitrary number of classification heads

No existing standard format provides this exact combination of features, making this a valuable contribution to multihead/multi-label dataset formats.

## Success Criteria

### Functional Requirements
- [ ] Successfully load VIMH datasets
- [ ] Support arbitrary numbers of heads (1-255)
- [ ] Handle variable image dimensions
- [ ] Maintain backward compatibility
- [ ] Achieve equivalent performance to existing single-head datasets

### Performance Requirements
- [ ] Dataset loading time comparable to existing formats
- [ ] Memory usage within 2x of single-head equivalent
- [ ] Training speed within 10% of single-head baseline
- [ ] Support datasets up to 1M samples

### Quality Requirements
- [ ] 100% test coverage for core dataset classes
- [ ] Comprehensive documentation
- [ ] Validation tools for format compliance
- [ ] Clear error messages for malformed data

This plan provides a comprehensive roadmap for implementing support
for the novel VIMH multihead dataset format while maintaining
flexibility for future extensions and ensuring robust, well-tested
implementation.
