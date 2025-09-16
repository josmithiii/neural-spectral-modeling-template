# Neural Spectral Modeling Template Documentation

Welcome to the VIMH-first documentation set. NSMT focuses on predicting audio synthesis parameters from spectrogram-like inputs using PyTorch Lightning and Hydra. Every page here tracks the current codebase, make targets, and configs.

## Documentation Map

### Getting Started

- [features.md](features.md) — Capabilities and what ships in the template
- [quickref.md](quickref.md) — One-page command cheat sheet
- [tutorial_sequence.md](tutorial_sequence.md) — End-to-end walkthrough with make targets
- [audio_developer_primer.md](audio_developer_primer.md) — Signal-processing + ML refresher for audio developers

### Technical Details

- [vimh.md](vimh.md) — Dataset format, metadata, and tooling
- [vimh_loss_functions.md](vimh_loss_functions.md) — Distance-aware losses for quantized parameters
- [architectures.md](architectures.md) — Supported model configs and options
- [multihead_data_architecture.md](multihead_data_architecture.md) — How VIMH data flows through the loaders

### Usage & Configuration

- [configuration.md](configuration.md) — Hydra layout and override patterns
- [audio_eval.md](audio_eval.md) — Reconstruction evaluation workflow
- Wah experiments: [experiments/wah.md](experiments/wah.md)
- [command_reference.md](command_reference.md) — Snapshot of available make targets
- Project README: [../README.md](../README.md)

## Quick Links

- Generate the default small dataset: `make gds` (wraps `make gdws`), then inspect with `make dds`
- Spin up the large wah dataset: `make gdl`, preview with `make ddl`
- Train quickly for smoke testing: `make trq`
- Run the supported wah experiments: `make ewt` (classification) or `make ewtr` (regression)
- Launch TensorBoard: `make tensorboard` and open http://localhost:6006

## External References

- Lightning: https://lightning.ai/docs/pytorch/stable
- Hydra: https://hydra.cc

Legacy CIFAR/MNIST guidance has been retired; all docs now reflect the audio-centric NSMT workflow.
