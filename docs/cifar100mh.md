# CIFAR-100-MH Dataset Support

## Overview

The CIFAR-100-MH (Multi-Head) dataset format extends the standard CIFAR binary format to support multiple classification heads with embedded metadata. This format enables multihead learning with real (non-synthetic) labels from multiple diverse datasets, providing a powerful tool for multi-task learning research.

## 🎯 Format Specification

### Binary Format Structure

CIFAR-100-MH uses a self-describing binary format with embedded metadata:

```
Label Structure: [N] [param1_id] [param1_val] [param2_id] [param2_val] ...
```

Where:
- **N**: Number of classification heads (1-255)
- **param_id**: Parameter identifier (0-255)
- **param_val**: Parameter value (0-255)

### Metadata File

Each dataset includes a JSON metadata file (`cifar100mh_dataset_info.json`):

```json
{
  "format": "CIFAR-100-MH",
  "version": "1.0",
  "dataset_name": "example_dataset",
  "n_samples": 60000,
  "train_samples": 50000,
  "test_samples": 10000,
  "image_size": "32x32x3",
  "parameter_names": ["note_number", "note_velocity"],
  "parameter_mappings": {
    "note_number": {
      "min": 0,
      "max": 127,
      "description": "MIDI note number",
      "scale": "linear"
    },
    "note_velocity": {
      "min": 0,
      "max": 127,
      "description": "MIDI note velocity",
      "scale": "linear"
    }
  }
}
```

## 🏗️ Implementation Architecture

### Core Components

1. **`CIFAR100MHDataset`**: PyTorch dataset class for CIFAR-100-MH format
2. **`CIFAR100MHDataModule`**: Lightning data module with custom collate function
3. **`MultiheadLitModule`**: Generic Lightning module for multihead training
4. **`MultiheadDatasetBase`**: Base class for multihead dataset implementations

### Key Features

- **Dynamic head configuration**: Automatically configures model heads from dataset metadata
- **Custom collate function**: Handles batching of multihead labels
- **Auto-configuration**: Models automatically adapt to dataset structure
- **Metadata validation**: Ensures format compliance and data integrity
- **Transform support**: Full compatibility with torchvision transforms

## 📊 Dataset Classes

### CIFAR100MHDataset

```python
from src.data.cifar100mh_dataset import CIFAR100MHDataset

# Load training data
train_dataset = CIFAR100MHDataset(
    root="path/to/cifar100mh/data",
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)

# Get dataset information
print(f"Dataset size: {len(train_dataset)}")
print(f"Image shape: {train_dataset.get_image_shape()}")
print(f"Heads config: {train_dataset.get_heads_config()}")

# Access sample
image, labels = train_dataset[0]
print(f"Image shape: {image.shape}")  # torch.Size([3, 32, 32])
print(f"Labels: {labels}")  # {'param_0': tensor(42), 'param_1': tensor(108)}
```

### CIFAR100MHDataModule

```python
from src.data.cifar100mh_datamodule import CIFAR100MHDataModule

# Create data module
dm = CIFAR100MHDataModule(
    data_dir="path/to/cifar100mh/data",
    batch_size=64,
    num_workers=4,
    pin_memory=True
)

# Setup data module
dm.setup()

# Access dataloaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()

# Get dataset information
info = dm.get_dataset_info()
print(f"Heads config: {info['heads_config']}")
print(f"Image shape: {info['image_shape']}")
print(f"Train samples: {info['num_train_samples']}")
```

## ⚙️ Configuration

### Data Configuration

```yaml
# configs/data/cifar100mh.yaml
_target_: src.data.cifar100mh_datamodule.CIFAR100MHDataModule

data_dir: ${paths.data_dir}/cifar100mh
batch_size: 64
num_workers: 4
pin_memory: True
persistent_workers: True

# Optional custom transforms
train_transform:
  _target_: torchvision.transforms.Compose
  transforms:
    - _target_: torchvision.transforms.ToTensor
    - _target_: torchvision.transforms.Normalize
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2023, 0.1994, 0.2010]
    - _target_: torchvision.transforms.RandomHorizontalFlip
      p: 0.5
    - _target_: torchvision.transforms.RandomCrop
      size: 32
      padding: 4
```

### Model Configuration

```yaml
# configs/model/cifar100mh_cnn_64k.yaml
_target_: src.models.multihead_module.MultiheadLitModule

# Auto-configure from dataset
auto_configure_from_dataset: true

# Optional explicit loss weighting
loss_weights:
  note_number: 1.0
  note_velocity: 0.8

# Network configuration
net:
  _target_: src.models.components.simple_cnn.SimpleCNN
  input_channels: 3
  conv1_channels: 32
  conv2_channels: 64
  fc_hidden: 128
  input_size: 32
  # heads_config auto-filled from dataset

# Optimizer and scheduler
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 1e-4

scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 10
  gamma: 0.1
```

### Complete Experiment

```yaml
# configs/experiment/cifar100mh_cnn.yaml
defaults:
  - override /data: cifar100mh
  - override /model: cifar100mh_cnn_64k
  - override /trainer: default

seed: 12345
tags: ["cifar100mh", "multihead", "cnn"]

trainer:
  max_epochs: 50
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 0.001

data:
  batch_size: 64
```

## 🚀 Usage Examples

### Basic Training

```bash
# Train with CNN architecture
python src/train.py experiment=cifar100mh_cnn

# Train with ConvNeXt
python src/train.py experiment=cifar100mh_convnext

# Train with EfficientNet
python src/train.py experiment=cifar100mh_efficientnet

# Train with Vision Transformer
python src/train.py experiment=cifar100mh_vit
```

### Quick Testing

```bash
# Fast development run
python src/train.py experiment=cifar100mh_cnn trainer.fast_dev_run=true

# Single epoch test
python src/train.py experiment=cifar100mh_cnn trainer.max_epochs=1
```

### Custom Loss Weighting

```bash
# Emphasize first parameter
python src/train.py experiment=cifar100mh_cnn \
  model.loss_weights.note_number=2.0 \
  model.loss_weights.note_velocity=0.5

# Equal weighting
python src/train.py experiment=cifar100mh_cnn \
  model.loss_weights.note_number=1.0 \
  model.loss_weights.note_velocity=1.0
```

## 📈 Metrics and Logging

### Logged Metrics

**Training Metrics**:
- `train/loss` - Combined weighted loss
- `train/param_0_acc` - Accuracy for first classification head
- `train/param_1_acc` - Accuracy for second classification head

**Validation Metrics**:
- `val/loss` - Combined validation loss
- `val/param_0_acc` - Validation accuracy for first head
- `val/param_1_acc` - Validation accuracy for second head
- `val/acc_best` - Best validation accuracy (primary task)

**Test Metrics**:
- `test/loss` - Combined test loss
- `test/param_0_acc` - Test accuracy for first head
- `test/param_1_acc` - Test accuracy for second head

### Performance Monitoring

```python
# Access training metrics
import wandb

# Log custom metrics
wandb.log({
    "custom/combined_accuracy": (acc_param_0 + acc_param_1) / 2,
    "custom/accuracy_variance": abs(acc_param_0 - acc_param_1),
    "custom/loss_ratio": loss_param_0 / loss_param_1
})
```

## 🔬 Advanced Features

### Auto-Configuration

The system automatically configures model heads from dataset metadata:

```python
# Model automatically adapts to dataset structure
model = MultiheadLitModule(
    net=SimpleCNN(input_channels=3, input_size=32),
    optimizer=torch.optim.Adam,
    scheduler=torch.optim.lr_scheduler.StepLR,
    auto_configure_from_dataset=True  # Key feature
)

# After setup, model.net.heads_config is automatically populated
# model.criteria is automatically configured for each head
```

### Custom Collate Function

CIFAR100MHDataModule includes a custom collate function for multihead labels:

```python
def _multihead_collate_fn(self, batch):
    """Custom collate function for multihead labels."""
    images = []
    labels_dict = {}

    for image, labels in batch:
        images.append(image)
        if not labels_dict:
            for head_name in labels.keys():
                labels_dict[head_name] = []
        for head_name, label_value in labels.items():
            labels_dict[head_name].append(label_value)

    batched_images = torch.stack(images)
    batched_labels = {
        head_name: torch.tensor(label_list, dtype=torch.long)
        for head_name, label_list in labels_dict.items()
    }

    return batched_images, batched_labels
```

### Dataset Information API

```python
# Get comprehensive dataset information
dataset = CIFAR100MHDataset(data_dir, train=True)

# Basic info
print(f"Dataset length: {len(dataset)}")
print(f"Image shape: {dataset.get_image_shape()}")
print(f"Heads config: {dataset.get_heads_config()}")

# Parameter information
param_info = dataset.get_parameter_info('note_number')
print(f"Parameter description: {param_info['description']}")
print(f"Parameter range: {param_info['min']}-{param_info['max']}")

# Dataset statistics
stats = dataset.get_dataset_statistics()
print(f"Class distribution: {stats['class_distribution']}")
print(f"Parameter statistics: {stats['parameter_statistics']}")
```

## 🛠️ Implementation Details

### Dataset Loading Pipeline

1. **Metadata Loading**: Parse JSON metadata file
2. **Data File Detection**: Locate train/test batch files
3. **Format Validation**: Verify data format compliance
4. **Label Parsing**: Extract multihead labels from binary format
5. **Image Reconstruction**: Convert flat arrays to tensors
6. **Transform Application**: Apply preprocessing transforms

### Error Handling

The implementation includes comprehensive error handling:

```python
try:
    dataset = CIFAR100MHDataset(data_dir, train=True)
except FileNotFoundError:
    print("Dataset files not found. Check data_dir path.")
except ValueError as e:
    print(f"Format validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Memory Optimization

- **Lazy loading**: Images loaded only when accessed
- **Efficient batching**: Custom collate function optimizes memory usage
- **Transform caching**: Preprocessing results can be cached
- **Persistent workers**: Reduce worker startup overhead

## 🧪 Testing and Validation

### Format Validation

```python
# Test dataset format compliance
pytest tests/test_multihead_datasets.py::TestCIFAR100MHDataset -v

# Test format validation
pytest tests/test_format_validation.py -v

# Test training integration
pytest tests/test_multihead_training.py -v
```

### Performance Testing

```bash
# Benchmark data loading
python -m pytest tests/test_multihead_datasets.py::TestIntegration::test_batch_loading_compatibility -v

# Test memory usage
python -c "
import torch
from src.data.cifar100mh_datamodule import CIFAR100MHDataModule
dm = CIFAR100MHDataModule('data/cifar100mh', batch_size=64)
dm.setup()
loader = dm.train_dataloader()
batch = next(iter(loader))
print(f'Batch memory: {batch[0].element_size() * batch[0].nelement() / 1024**2:.2f} MB')
"
```

## 🔍 Troubleshooting

### Common Issues

1. **File Not Found**: Ensure data directory contains required files
   - `train_batch` (pickle file)
   - `test_batch` (pickle file)
   - `cifar100mh_dataset_info.json` (metadata)

2. **Format Errors**: Check label format compliance
   - Labels must follow `[N, param_id, param_val, ...]` structure
   - All samples must have consistent label structure

3. **Memory Issues**: Optimize batch size and worker count
   - Reduce `batch_size` for memory constraints
   - Adjust `num_workers` based on system resources

4. **Transform Errors**: Ensure transforms match data format
   - Use `ToTensor()` for numpy arrays
   - Skip `ToTensor()` if data is already tensors

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create dataset with debug info
dataset = CIFAR100MHDataset(data_dir, train=True)
print(f"Dataset debug info: {dataset.get_dataset_statistics()}")
```

## 🔗 Integration with Existing Code

### Backward Compatibility

The CIFAR-100-MH implementation maintains compatibility with existing Lightning patterns:

```python
# Works with existing Lightning trainer
trainer = Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, datamodule=dm)

# Compatible with existing callbacks
from lightning.pytorch.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(monitor='val/loss')
trainer = Trainer(callbacks=[checkpoint_callback])
```

### Extension Points

Extend the system for custom datasets:

```python
# Custom dataset inheriting from MultiheadDatasetBase
class CustomMultiheadDataset(MultiheadDatasetBase):
    def _get_sample_metadata(self, idx):
        # Custom metadata extraction
        return {'custom_field': value}

    def _parse_labels(self, raw_labels):
        # Custom label parsing logic
        return parsed_labels
```

## 📚 Related Documentation

- [Multihead Classification System](multihead.md) - MNIST multihead implementation
- [Dataset Configuration](../configs/data/) - Configuration examples
- [Model Architecture](../src/models/components/) - Network implementations
- [Training Scripts](../src/train.py) - Training entry points

## 🎯 Best Practices

### Dataset Preparation
1. **Validate format**: Use format validation tests before training
2. **Check metadata**: Ensure parameter mappings are correct
3. **Verify splits**: Confirm train/test split sizes match expectations

### Training Configuration
1. **Start simple**: Begin with CNN architecture for baseline
2. **Monitor all heads**: Track performance of each classification head
3. **Adjust weighting**: Experiment with loss weights for optimal performance

### Performance Optimization
1. **Use persistent workers**: Set `persistent_workers=True` for faster loading
2. **Optimize batch size**: Balance memory usage and training speed
3. **Pin memory**: Enable `pin_memory=True` for GPU training

This comprehensive documentation provides everything needed to work with CIFAR-100-MH datasets in the Lightning-Hydra template framework.
