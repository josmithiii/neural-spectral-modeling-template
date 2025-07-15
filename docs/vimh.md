# VIMH Dataset Format

**Variable Image MultiHead (VIMH)** is a generalized dataset format designed for training multihead neural networks on images with multiple varying parameters per sample. Originally inspired by CIFAR-100, VIMH extends beyond fixed dimensions to support any image size with a self-describing variable-length label format.

## Key Features

- **Variable Image Dimensions**: Supports any image size (height×width×channels)
- **Self-Describing**: Each sample includes metadata (height, width, channels)
- **MultiHead Ready**: Encodes 0-255 varying parameters per sample
- **Efficient**: 8-bit quantization with ~100 perceptual resolution steps
- **Format Flexible**: Works with RGB (32x32x3), grayscale (28x28x1), and custom sizes

## Quick Start

```bash
# Generate VIMH dataset from STK synthesizer
make vimh-stk
# or: python data2vimh.py --input-dataset stk --output-dir data-vimh/

# Generate VIMH dataset from Resonarium synthesizer
make vimh-res
# or: python data2vimh.py --input-dataset resonarium --output-dir data-vimh/

# Display VIMH dataset with interactive viewer
make display-vimh
# or: python display_vimh.py data-vimh/vimh-32x32x3_*_resonarium_2p
```

## Output Formats

- **Binary**: Self-describing binary format for efficient loading
- **Pickle**: Python-friendly format with `vimh_labels` field
- **Both**: Default - creates both formats

## Format Specification

### Binary Layout Per Sample
```
Metadata: 6 bytes (height, width, channels) - three 16-bit fields
Label Data: 1 + 2N bytes
  - Byte 0: N (number of varying parameters)
  - Bytes 1,2: param1_id, param1_val
  - Bytes 3,4: param2_id, param2_val
  - ...
  - Bytes 2N-1,2N: paramN_id, paramN_val
Image Data: height*width*channels bytes
```

### Example: 2-Parameter Dataset (32x32x3 like CIFAR-100)
```
Metadata: [32, 32, 3] (height=32, width=32, channels=3)
Labels: [2, 0, 191, 1, 127] (N=2 parameters)
Image: 3072 bytes (32*32*3 RGB spectrogram)
Total: 3 + 5 + 3072 = 3080 bytes per sample
```

If note_number=51.5 (range 50-52) and note_velocity=81.0 (range 80-82):
```
Complete sample: [32, 32, 3, 2, 0, 191, 1, 127, <3072 image bytes>]
  height=32, width=32, channels=3
  N=2 parameters
  param_0 (note_number): 191 → 51.5
  param_1 (note_velocity): 127 → 81.0
```

### Example: VIMH Metadata for MNIST Images (28x28x1)
```
Metadata: [28, 28, 1] (height=28, width=28, channels=1)
Labels: [1, 0, 128] (N=1 parameter)
Image: 784 bytes (28*28*1 grayscale)
Total: 3 + 3 + 784 = 790 bytes per sample
```

## Dataset Structure

Each VIMH dataset includes binary or pickle format, or both:

```
data-vimh/vimh-32x32x3_8000Hz_1p0s_256dss_resonarium_2p/
├── train          # Binary training data
├── test           # Binary test data
├── train_batch    # Pickle training data
├── test_batch     # Pickle test data
└── vimh_dataset_info.json  # Dataset information
```

### Dataset Info JSON
```json
{
  "format": "VIMH",
  "version": "1.0",
  "height": 32,
  "width": 32,
  "channels": 3,
  "varying_parameters": 2,
  "parameter_names": ["note_number", "note_velocity"],
  "label_encoding": {
    "format": "[height] [width] [channels] [N] [param1_id] [param1_val] [param2_id] [param2_val] ...",
    "metadata_bytes": 3,
    "N_range": [0, 255],
    "param_id_range": [0, 255],
    "param_val_range": [0, 255]
  },
  "parameter_mappings": { /* full parameter info */ }
}
```

## Benefits

- **Fully generalized**: Height, width, and channels metadata allows any image size
- **Self-describing**: N tells you how many parameters each sample has
- **Scalable**: Supports 0-255 varying parameters per synthesizer
- **Efficient**: 8-bit quantization provides ~100 perceptual resolution steps
- **Multihead CNN ready**: Enables training CNNs with multiple output heads
- **Format flexible**: Supports spectrograms (32x32x3), MNIST (28x28x1), and custom sizes

## Use Cases

### Audio Synthesis
- **Spectrograms**: 32x32x3 mel spectrograms from synthesizers
- **Parameters**: Varying synthesis parameters (frequency, amplitude, filters)
- **Training**: Multihead CNNs to predict synthesis parameters from audio

### Computer Vision
- **MNIST Extension**: 28x28x1 images with varying parameters (digit style, thickness)
- **Custom Vision**: Any image size with associated continuous parameters
- **Multi-task Learning**: Train models with multiple regression outputs

### Scientific Data
- **Simulations**: Images from physics/chemistry simulations with varying conditions
- **Measurements**: Experimental data with multiple measured parameters
- **Analysis**: Train models to predict experimental conditions from images

## Compatibility

### Parameter Types
- **Varying parameters**: Different min/max values, encoded in labels
- **Fixed parameters**: min=max, not encoded in labels (stored in metadata)

### Limitations
- Maximum 255 varying parameters per sample
- Parameter values quantized to 0-255 range
- Image dimensions limited to 255x255 (due to single byte metadata)
- Maximum 255 channels per image

## Makefile Targets

| Target | Description |
|--------|-------------|
| `vimh-stk` | Convert STK dataset to VIMH |
| `vimh-res` | Convert Resonarium dataset to VIMH |
| `vimh-all` | Convert all datasets to VIMH |
| `display-vimh` | Display VIMH dataset viewer |

## Technical Implementation

### Label Encoding
Each sample's parameters are encoded as a sequence of (parameter_id, parameter_value) pairs:
- **parameter_id**: Index into the varying parameters list (0-255)
- **parameter_value**: Quantized parameter value (0-255)

### Parameter Quantization
```python
# Normalize to [0,1], then quantize to [0,255]
normalized_value = (actual_value - param_min) / (param_max - param_min)
quantized_value = int(normalized_value * 255)
```

### Parameter Dequantization
```python
# Dequantize from [0,255] back to actual parameter range
normalized_value = quantized_value / 255.0
actual_value = param_min + normalized_value * (param_max - param_min)
```

## Version History

- **v2.0**: Added height, width, channels metadata for full generalization
- **v1.0**: Initial implementation based on CIFAR-100 structure

## Related Formats

VIMH was originally inspired by CIFAR-100 but extends it significantly:
- **CIFAR-100**: Fixed 32x32x3 images, 2 labels (coarse/fine)
- **VIMH**: Variable image dimensions, 0-255 parameters with metadata

The format maintains some CIFAR-100 compatibility for 32x32x3 images while enabling much broader applications through its generalized design.
