# Neural Spectral Modeling Template (NSMT)

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
interpreted as _frequency_ and _time_, respectively.  The _channel
dimension_ (typically color channels in images), is generally used for
_alternate spectral representations_ such as

* Alternate time-frequency resolutions (spectrograms with different settings),
* Instantaneous Frequency (time-derivative of the spectral phase)
* Group Delay (frequency-derivative of the spectral phase), 
* Modulation Spectra (spectrogram of the spectrogram modules), 

and so on.

Some might note that we don't _need_ alternative input
representations, or even spectrograms, because a large neural network
can learn to compute them when needed.  In principle, the input to the
network can be an undifferentiated _bit stream_ containing the audio.
However, such "end-to-end" approaches require much more computation
and training data, as the examples here will show.

Thus, the purpose of the NSMT spectra-based approach is to facilitate
_small, accurate, and fast_ neural networks for audio processing and
classification.  We accept the inductive priors of human hearing, and
in return we get to use more efficient neural architectures such as
the CNNs with conditioning inputs.  The examples will illustrate the
benefits of this choice for selected audio tasks.

The NSMT project uses only the Variable Image Multi-Head (VIMH)
dataset format from the LHTE.  (The LHT supports MNIST datasets, and
the
[LHTE](https://github.com/josmithiii/lightning-hydra-template-extended.git).
adds CIFAR and VIMH support to that.)  The CIFAR and MNIST support can
be dropped here because VIMH can support those image resolutions,
channel depths, and metadata, as special cases.

For documentation, see [docs/index.md](docs/index.md).

### 🚀 Quick Start

```bash
# Set up the environment (uv)
sh setup.sh

# Look over all make targets available (same as `make help`)
make h

# Generate a default small VIMH dataset for some quick tests (same as `make sds`)
# (writes ./data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p)
python generate_vimh.py --config-name=synth/generate_simple_saw

# Generate a default large VIMH dataset for some quick tests (same as `make sdl`)
# (writes ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p)
python generate_vimh.py --config-name=synth/generate_simple_saw


# Display the most recently created dataset (default) (`make ddr`)
python display_vimh.py

# Display the small example VIMH dataset (256 samples) (`make dds`)
python display_vimh.py data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p

# Display the larger example VIMH dataset (16k samples) (`make ddl`)
python display_vimh.py data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p


# Full training with experiment config
python src/train.py experiment=cnn_16kdss

# Look over all configuration overrides available (see especially "experiment: ...")
python src/train.py --help

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
data_dir: data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p
batch_size: 128
num_workers: 4

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

- **Audio Synthesis**: Image-to-audio parameter mapping
- **Computer Vision**: Multi-target regression tasks
- **Scientific Computing**: Parameter prediction from visual data
- **Research**: Multihead neural network architectures
