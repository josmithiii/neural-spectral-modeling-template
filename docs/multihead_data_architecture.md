# 🧠 Multihead Data Architecture: Theory of Operation

This document explains the **theory of operation** for the multihead dataset architecture, including why we have three different dataset components and when to use each one.

## 🏗️ Architecture Overview

The multihead dataset system uses a **layered architecture** with three complementary components:

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  multihead_dataset.py          │  generic_multihead_dataset.py │
│  (Synthetic Labels)            │  (Real Binary Labels)         │
│  - Strategy Pattern            │  - Auto-detection             │
│  - MNIST/CIFAR wrapper         │  - Format validation           │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    FOUNDATION LAYER                         │
│              multihead_dataset_base.py                      │
│              (Abstract Base Class)                          │
│              - Common data loading                          │
│              - Binary format parsing                        │
│              - Label structure validation                   │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Component Analysis

### 1. `multihead_dataset_base.py` - The Foundation 🏗️

**File**: `src/data/multihead_dataset_base.py`

**Purpose**: Abstract base class that handles the **common mechanics** of multihead dataset loading.

#### Key Responsibilities

- **Binary format parsing**: Handles the core label format `[N, param_id, param_val, ...]`
- **Image reconstruction**: Converts flat arrays to PyTorch tensors
- **Format validation**: Ensures data consistency and structure
- **Heads configuration**: Automatically calculates number of classes per head
- **File I/O**: Handles both single files and directory structures

#### Core Methods

```python
def _parse_label_metadata(self, label_data) -> Tuple[Dict[str, int], Dict[str, int]]
def _reconstruct_image(self, image_data, metadata) -> torch.Tensor
def _validate_format(self) -> bool
def get_heads_config(self) -> Dict[str, int]
```

#### Why It Exists

- **DRY Principle**: Avoid code duplication across different dataset types
- **Consistency**: Ensures all multihead datasets behave the same way
- **Extensibility**: Easy to add new dataset formats by inheriting from this base
- **Validation**: Centralized format validation and error handling

#### Design Patterns

- **Abstract Base Class**: Defines interface and common functionality
- **Template Method**: Subclasses implement `_get_sample_metadata()`

---

### 2. `generic_multihead_dataset.py` - The Smart Adapter 🧠

**File**: `src/data/generic_multihead_dataset.py`

**Purpose**: Handles **real multihead datasets** with embedded binary labels and auto-detection capabilities.

#### Key Responsibilities

- **Format auto-detection**: Automatically detects CIFAR-100-MH vs generic formats
- **Metadata completion**: Fills in missing format configuration fields
- **Image dimension inference**: Supports different image sizes (784, 3072 pixels)
- **Transform support**: Applies torchvision transforms
- **Factory pattern**: Provides `MultiheadDatasetFactory` for easy instantiation

#### Core Features

```python
# Auto-detection from file content
def _auto_detect_format(self, data_path: str) -> Dict[str, Any]
def _detect_from_content(self, file_path: Path) -> Dict[str, Any]

# Format configurations
def _get_cifar100mh_format(self) -> Dict[str, Any]
def _get_default_format(self) -> Dict[str, Any]

# Factory for easy instantiation
class MultiheadDatasetFactory:
    @staticmethod
    def create_dataset(data_path: str, dataset_type: str = 'auto')
```

#### Auto-Detection Logic

1. **Metadata Files**: Looks for `cifar100mh_dataset_info.json`, `dataset_info.json`
2. **File Structure**: Detects `train_batch`, `test_batch` patterns
3. **Content Analysis**: Analyzes image dimensions and label structure
4. **Format Inference**: Creates appropriate configuration automatically

#### Why It Exists

- **Real-world datasets**: For datasets with actual multihead labels (not synthetic)
- **Flexibility**: Can handle various binary formats without manual configuration
- **Research tool**: Supports unknown/custom dataset formats with auto-detection
- **User-friendly**: Minimal configuration required

#### Use Cases

- CIFAR-100-MH format datasets
- Custom binary multihead datasets
- Research datasets with embedded metadata
- Unknown format datasets requiring auto-detection

---

### 3. `multihead_dataset.py` - The Strategy Engine 🎯

**File**: `src/data/multihead_dataset.py`

**Purpose**: Converts **single-label datasets** to multihead using intelligent **synthetic label generation**.

#### Key Responsibilities

- **Strategy pattern**: Different label generation strategies per dataset type
- **Synthetic label generation**: Creates meaningful auxiliary tasks
- **Backward compatibility**: Maintains existing MNIST/CIFAR functionality
- **Domain knowledge**: Embeds expert knowledge about dataset characteristics

#### Strategy Implementations

```python
class MNISTStrategy(MultiheadLabelStrategy):
    """Generate thickness and smoothness labels for MNIST digits"""

class CIFAR10Strategy(MultiheadLabelStrategy):
    """Generate domain, mobility, and size labels for CIFAR-10"""

class CIFAR100Strategy(MultiheadLabelStrategy):
    """Generate coarse, domain, and complexity labels for CIFAR-100"""
```

#### Label Generation Examples

**MNIST Strategy**:

- **Primary**: Original digit (0-9)
- **Thickness**: Thin vs thick digits based on pixel density
- **Smoothness**: Smooth vs rough based on edge complexity

**CIFAR-10 Strategy**:

- **Primary**: Original class (airplane, car, etc.)
- **Domain**: Natural vs artificial objects
- **Mobility**: Mobile vs stationary objects
- **Size**: Large vs small typical object size

**CIFAR-100 Strategy**:

- **Primary**: Fine labels (specific objects)
- **Coarse**: Coarse labels (object categories)
- **Domain**: Indoor vs outdoor vs abstract
- **Complexity**: Simple vs complex shapes

#### Why It Exists

- **Educational purposes**: Demonstrates multihead learning on standard datasets
- **Rapid prototyping**: Quickly experiment with multihead architectures
- **Benchmark creation**: Creates consistent multihead benchmarks from single-label data
- **Research facilitation**: Easy access to multihead versions of standard datasets

#### Use Cases

- Teaching multihead classification concepts
- Prototyping new multihead architectures
- Creating benchmark datasets for evaluation
- Research requiring auxiliary task learning

## 🔄 When to Use Each Component

### Decision Tree

```
Do you have a dataset with real multihead labels?
├── YES → Use generic_multihead_dataset.py
│         ├── CIFAR-100-MH format? → Auto-detected ✅
│         ├── Custom format? → Auto-detected ✅
│         └── Unknown format? → Auto-detected ✅
│
└── NO → Use multihead_dataset.py
          ├── MNIST? → MNISTStrategy (digit/thickness/smoothness)
          ├── CIFAR-10? → CIFAR10Strategy (class/domain/mobility/size)
          └── CIFAR-100? → CIFAR100Strategy (fine/coarse/domain/complexity)
```

### Specific Usage Guidelines

#### Use `multihead_dataset_base.py` when:

- ❌ **Never directly** - it's an abstract base class
- ✅ **Inheriting** to create new multihead dataset types
- ✅ **Understanding** the core multihead data format

#### Use `generic_multihead_dataset.py` when:

- ✅ **Loading real multihead datasets** (CIFAR-100-MH format)
- ✅ **Unknown dataset formats** that need auto-detection
- ✅ **Custom binary datasets** with embedded metadata
- ✅ **Research with varied formats** - let auto-detection handle it

#### Use `multihead_dataset.py` when:

- ✅ **Learning/teaching multihead concepts** with familiar datasets
- ✅ **Prototyping multihead architectures** quickly
- ✅ **Synthetic label experiments** on standard datasets
- ✅ **Benchmark creation** with consistent auxiliary tasks

## 🎯 Practical Examples

### Example 1: Research with Real Multihead Data

```python
from src.data.generic_multihead_dataset import GenericMultiheadDataset

# Let auto-detection handle the format
dataset = GenericMultiheadDataset(
    data_path="path/to/cifar100mh/",
    auto_detect=True
)

# Auto-detects format, configures heads, validates data
print(f"Heads config: {dataset.get_heads_config()}")
# Output: {'param_0': 10, 'param_1': 20}
```

### Example 2: Teaching Multihead Concepts

```python
from src.data.multihead_dataset import MultiheadDataset
import torchvision.datasets as datasets

# Create synthetic multihead labels from MNIST
mnist_base = datasets.MNIST(root="data", train=True, download=True)
multihead_mnist = MultiheadDataset(mnist_base, dataset_type='mnist')

# Get a sample with multiple labels
image, labels = multihead_mnist[0]
print(f"Labels: {labels}")
# Output: {'digit': 7, 'thickness': 1, 'smoothness': 0}
```

### Example 3: Custom Dataset Development

```python
from src.data.multihead_dataset_base import MultiheadDatasetBase

class MyCustomDataset(MultiheadDatasetBase):
    """Custom multihead dataset with domain-specific logic"""

    def _get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Extract custom metadata for each sample"""
        return {
            'custom_field': 'value',
            'sample_index': idx,
            'processing_timestamp': time.time()
        }

    def custom_analysis(self):
        """Domain-specific analysis methods"""
        return self.get_dataset_statistics()

# Usage
custom_dataset = MyCustomDataset(
    data_path="path/to/custom/data",
    metadata_format=custom_format_config
)
```

## 🔬 Design Patterns and Principles

### Design Patterns Used

1. **Abstract Base Class**: `MultiheadDatasetBase` provides common interface
2. **Strategy Pattern**: `MultiheadLabelStrategy` allows different label generation
3. **Factory Pattern**: `MultiheadDatasetFactory` simplifies dataset creation
4. **Template Method**: Base class defines algorithm, subclasses implement details

### SOLID Principles

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Easy to extend with new strategies or formats
- **Liskov Substitution**: All implementations can be used interchangeably
- **Interface Segregation**: Clean, focused interfaces
- **Dependency Inversion**: Depend on abstractions, not concrete classes

### Benefits of This Architecture

1. **Separation of Concerns**: Each component handles a specific aspect
2. **Code Reuse**: Common functionality in base class
3. **Extensibility**: Easy to add new dataset types or strategies
4. **Maintainability**: Clear structure and responsibilities
5. **Testability**: Each component can be tested independently
6. **Flexibility**: Supports both synthetic and real multihead datasets

## 🚀 Integration with Lightning Framework

The architecture integrates seamlessly with PyTorch Lightning:

```python
from src.data.multihead_datamodule import CIFAR100MHDataModule

# Lightning DataModule automatically uses appropriate dataset type
dm = CIFAR100MHDataModule(
    data_dir="path/to/cifar100mh",
    batch_size=32,
    num_workers=4
)

# Auto-configures from dataset metadata
dm.setup()
heads_config = dm.heads_config  # Automatically detected
```

## 📚 Related Documentation

- [Multihead Classification System](multihead.md) - MNIST multihead implementation
- [CIFAR-100-MH Format](cifar100mh.md) - Binary format specification
- [Configuration Guide](README-CONFIGURATION.md) - Hydra configuration patterns
- [Architecture Overview](README-ARCHITECTURES.md) - Model architectures

## 🔧 Future Extensions

The architecture is designed for easy extension:

1. **New Dataset Formats**: Inherit from `MultiheadDatasetBase`
2. **New Label Strategies**: Implement `MultiheadLabelStrategy`
3. **Advanced Auto-detection**: Enhance format detection logic
4. **Custom Transforms**: Add dataset-specific preprocessing
5. **Streaming Support**: Add support for large-scale datasets

This modular design ensures the system can grow and adapt to new research needs while maintaining backward compatibility and clean code organization.
