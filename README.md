# Neural Spectral Modeling Template (NSMT)

### Status: _**Alpha**_

## Overview

The [**Neural Spectral Modeling Template
(NSMT)**](https://github.com/josmithiii/neural-spectral-modeling-template.git)
is a fork of the [Lightning-Hydra-Template-Extended
(LHTE)](https://github.com/josmithiii/lightning-hydra-template-extended.git)
for neural image processing and classification, which in turn is an
extension of the [Lightning-Hydra-Template
(LHT)](https://github.com/ashleve/lightning-hydra-template) for
projects using PyTorch Lightning and Hydra in their machine-learning
workflows. The LHT includes a single image-classification example
using a Multi-Layer Perceptron (MLP) on MNIST (hand-written digits).
The LHTE adds various neural architectures such as Convolutional
Neural Networks (CNN) and Vision Transformer (ViT).

The NSTM treats all images as _spectral representations_ for audio.
This means that the image _height_ and _width_ are typically
interpreted as _frequency_ and _time_, respectively. The _channel
dimension_ (typically color channels in images), is generally used for
_alternate spectral representations_ such as

- Alternate time-frequency resolutions (spectrograms with different settings),
- Instantaneous Frequency (time-derivative of the spectral phase)
- Group Delay (frequency-derivative of the spectral phase),
- Modulation Spectra (spectrogram of the spectrogram modulus),

and so on.

In principle, we don't _need_ alternative input representations, or
even spectrograms, because a large neural network can learn to compute
them as needed.  However, such "end-to-end" approaches require much
more computation and training data, as examples here will show.

Thus, the purpose of the NSMT spectra-based approach is to facilitate
_small, accurate, and fast_ neural networks for audio processing and
classification. We accept the inductive priors of human hearing, and
in return we get to use more efficient neural architectures such as
the CNNs with conditioning inputs. The examples will illustrate the
benefits of this choice for selected audio tasks.

The NSMT project uses only the Variable Image Multi-Head (VIMH)
dataset format from the LHTE. (The LHT supports only MNIST datasets, and
the [LHTE](https://github.com/josmithiii/lightning-hydra-template-extended.git)
adds CIFAR and VIMH support to that.) The CIFAR and MNIST support are
dropped here because VIMH can support those image resolutions,
channel depths, and metadata, as special cases.

### 📚 Docs

- Overview: [docs/index.md](docs/index.md)
- Features: [docs/features.md](docs/features.md)
- Quick Reference: [docs/quickref.md](docs/quickref.md)
- Tutorial: [docs/tutorial_sequence.md](docs/tutorial_sequence.md)
- VIMH Format: [docs/vimh.md](docs/vimh.md)
- VIMH Losses: [docs/vimh_loss_functions.md](docs/vimh_loss_functions.md)
- Architectures: [docs/architectures.md](docs/architectures.md)
- Configuration: [docs/configuration.md](docs/configuration.md)
- Audio Evaluation: [docs/audio_eval.md](docs/audio_eval.md)

### 🚀 Quick Start

```bash
# Set up the environment (uv)
sh setup.sh

# Look over all make targets available
make h

# ===== DATASET GENERATION =====

# Generate small synthetic dataset (256 samples) - now uses wah pedal
make sds    # or: python generate_vimh.py --config-name=synth/generate_saw_wah

# Generate large synthetic dataset (16k samples)
make sdl    # or: python generate_vimh.py --config-name=synth/generate_saw_wah dataset.size=16384

# Generate all small Wah and Moog VCF datasets (basic, envelope, resonance)
make sda

# Generate small synthetic dataset (512 samples) using sawtooth into wah pedal
make sdw

# Generate small synthetic dataset (512 samples) using sawtooth into wah pedal controlled by ADSR envelope
make sdwe

# ===== DATASET DISPLAY =====

# Display most recently created dataset
make ddr    # or: python display_vimh.py

# Display specific datasets
make dds    # small dataset
make ddl    # large dataset

# Print dataset metadata
make vdr    # or: python vimhd.py  # prints latest dataset metadata
make vds    # or: python vimhd.py path/to/small-example-dataset
make vdl    # or: python vimhd.py path/to/larger-example-dataset
            # or: python vimhd.py path/to/any-dataset

# Analyze parameter distributions (NEW!)
make vpr    # or: python vimhd.py -p  # analyze latest dataset parameters
make vps    # or: python vimhd.py -p path/to/small-example-dataset
make vpl    # or: python vimhd.py -p path/to/larger-example-dataset
            # Shows histograms, statistics, and uniformity tests

# ===== TRAINING EXPERIMENTS =====

# Quick example experiment (small and quick for testing)
make ex     # CNN on default dataset

# Trivial dataset experiments (small models for testing)
make etms   # Micro CNN (~2K params) on small dataset, ordinal classification output
make etmsr  # Micro CNN (~2K params) on small dataset, regression output (1 float/head)
make etts   # Tiny CNN (~8K params) on small dataset
make etml   # Micro CNN on large dataset
make ettl   # Tiny CNN on large dataset
make etall  # Run all trivial experiments

# ViT experiments on trivial datasets (quick tests)
make evitms # Micro ViT (~8K params) on small dataset
make evitts # Tiny ViT (~25K params) on small dataset
make evitall # Run all ViT trivial experiments

# Wah Pedal experiments
make ew     # CNN training on dataset sdw (sawtooth + wah + decay envelope)
make ewe    # CNN training on dataset sdwe (sdw + ADSR wah control)

# Moog VCF experiments using CNN architecture
make emb    # Basic Moog VCF (4 params)
make eme    # Moog envelope sweep (10 params), ordinal classification output
make emer   # Moog envelope sweep (10 params), regression output
make emr    # High-resonance Moog (8 params)
make emall  # Generate datasets + train all Moog CNNs

# Moog VCF experiments using ViT architecture
make emvitb # ViT on basic Moog VCF
make emvite # ViT on Moog envelope
make emvitr # ViT on high-resonance Moog
make emvitgta # Generate + train all Moog ViTs

# ===== DIRECT TRAINING =====

# Train with default config
make tr     # or: python src/train.py

# Quick sanity check (1 epoch)
make trq    # or: python src/train.py trainer.max_epochs=1

# Train on specific datasets
make trs    # small dataset
make trl    # large dataset

# ===== TESTING & VISUALIZATION =====

# Run tests
make t      # fast tests only
make ta     # all tests

# Generate model diagrams
make td     # enhanced diagrams for all architectures
make tds    # simple text-only diagrams
make tdl    # list available model configs

# ===== UTILITIES =====

# Audio evaluation of latest checkpoint
make ae

# Clean up
make c      # clean all autogenerated files
make dc     # clean data files only
make cl     # clean logs only

# Code formatting
make f      # run pre-commit hooks

# TensorBoard
make tb     # launch on port 6006

# List configurations
make lc     # list all model, data, experiment configs

```

### 📊 Dataset Format

VIMH datasets use a structured format with:

- **Images**: Variable dimensions (e.g., 32x32x3, 28x28x1)
- **Labels**: `[N] [param1_id] [param1_val] [param2_id] [param2_val] ...`
- **Metadata**: JSON file with parameter mappings and dataset info
- **Validation**: Cross-validation across directory name, JSON, and binary sources

### 🔧 Configuration

```yaml
# configs/data/vimh_256dss.yaml
_target_: src.data.vimh_datamodule.VIMHDataModule
data_dir: data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p  # Now using wah pedal
batch_size: 128
num_workers: 4

# Dataset generation with configurable naming
# configs/synth/generate_saw_wah.yaml
dataset:
  name: "saw_wah"  # Custom dataset name for output directory
  size: 256
synthesizer:
  filter_type: "wah"  # "moog" or "wah" supported
  parameters:
    log10_filter_cutoff_hz: # Variable wah pedal frequency
      min_value: 3.0
      max_value: 3.602

# Model auto-configures from dataset
# configs/experiment/cnn.yaml
defaults:
  - override /data: vimh
  - override /model: cnn_64k
```

### 📈 Performance

- **Loading Optimization**: 10x faster initialization with efficient dimension detection
- **Memory Efficiency**: Optimized transform adjustment for different image sizes
- **Training Speed**: Comparable to single-head models with minimal overhead
- **Scalability**: Supports datasets up to 1M+ samples

### 🛠️ Use Cases

- **Audio Effects Modeling**: Predict Moog VCF and wah pedal parameters from spectrograms
- **Audio Synthesis**: Image-to-audio parameter mapping with configurable filter types
- **Computer Vision**: Multi-target regression tasks with custom dataset naming
- **Scientific Computing**: Parameter prediction from visual data
- **Research**: Multihead neural network architectures with spectral audio processing
