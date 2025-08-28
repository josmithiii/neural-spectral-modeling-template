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

### 🚀 Quick Start

```bash
# Set up the environment (uv)
sh setup.sh

```bash
# Generate a default VIMH dataset for some quick tests
python generate_vimh.py

```bash
# Train with VIMH dataset
python src/train.py experiment=vimh_cnn

# Run complete training example
python examples/vimh_example.py

# Full training with experiment config
python src/train.py experiment=vimh_cnn_16kdss
```

### 📊 Dataset Format

VIMH datasets use a structured format with:
- **Images**: Variable dimensions (e.g., 32x32x3, 28x28x1)
- **Labels**: `[N] [param1_id] [param1_val] [param2_id] [param2_val] ...`
- **Metadata**: JSON file with parameter mappings and dataset info
- **Validation**: Cross-validation across directory name, JSON, and binary sources

### 🔧 Configuration

```yaml
# configs/data/vimh.yaml
_target_: src.data.vimh_datamodule.VIMHDataModule
data_dir: data/vimh-32x32_8000Hz_1p0s_256dss_resonarium_2p
batch_size: 128
num_workers: 4

# Model auto-configures from dataset
# configs/experiment/vimh_cnn.yaml
defaults:
  - override /data: vimh
  - override /model: vimh_cnn_64k
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

For detailed documentation, see [docs/vimh.md](docs/vimh.md),
[docs/configuration.md](docs/configuration.md), and
[docs/multihead.md](docs/multihead.md).
