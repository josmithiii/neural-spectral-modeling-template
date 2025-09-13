# Neural Spectral Modeling Template Documentation

Welcome. This project focuses on VIMH-based neural spectral modeling: training models to predict audio synthesis parameters from spectrogram-like images. The docs below reflect the current repo (configs, make targets, and code).

## Documentation Map

### Getting Started

- [features.md](features.md) — Key capabilities and what’s included
- [quickref.md](quickref.md) — One‑page cheat sheet of common tasks
- [tutorial_sequence.md](tutorial_sequence.md) — Step‑by‑step walkthrough using make targets

### Technical Details

- [vimh.md](vimh.md) — VIMH dataset format and metadata
- [vimh_loss_functions.md](vimh_loss_functions.md) — Distance‑aware losses for quantized parameters
- [architectures.md](architectures.md) — Supported model configs and options

### Usage & Configuration

- [configuration.md](configuration.md) — Hydra config layout and best practices
- [audio_eval.md](audio_eval.md) — Audio reconstruction evaluation workflow
- Project README: [../README.md](../README.md)

## Quick Links

- Train quickly: `make trq` or `python src/train.py trainer.max_epochs=1`
- Run an example experiment: `python src/train.py experiment=example`
- Generate a small dataset: `make sds` then visualize with `make dds`
- Launch TensorBoard: `make tensorboard` then open http://localhost:6006

## External References

- Lightning: https://lightning.ai/docs/pytorch/stable
- Hydra: https://hydra.cc

Note: Older references to CIFAR/MNIST and pages like benchmarks.md or makefile.md have been removed or refocused for this repository.
