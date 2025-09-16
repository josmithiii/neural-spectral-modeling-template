# Final Polish Plan

## Documentation Refresh (High Priority)
- Align make target references in `docs/index.md`, `docs/quickref.md`, `docs/features.md`, `docs/tutorial_sequence.md`, and `docs/vimh.md` with the current `Makefile` (`gds`, `gdws`, `t`, `ta`, etc.).
- Replace legacy MNIST/CIFAR context across `docs/multihead_data_architecture.md`, `docs/vimh.md`, and `src/models/components/simple_cnn.py` docstrings with audio-focused explanations that match the VIMH workflow.
- Merge or retire redundant overview pages (`docs/features.md` vs README) and ensure tone/style is consistent (limit emojis, adopt the professional voice used in `README.md`).
- Expand README quick-start follow-ups with pointers to `make ewt`, `make ewtr`, and expected metrics so newcomers can confirm their setup beyond the smoke tests.
- Create an “Audio Developer Primer” doc that covers the minimal signal-processing and ML background needed to use VIMH datasets, linking to dataset generation, loss choices, and evaluation tooling.

## Experiment & Evaluation Guidance
- Document the maturity of the wah pedal experiments (ewt/ewtr) with short “what good looks like” sections and sample log excerpts under `docs/audio_eval.md` or a new `docs/experiments/wah.md`.
- Add a troubleshooting matrix for dataset generation and training (common failure modes, required disk space, Mac MPS tweaks) to keep Quick Start runs frictionless.
- Record canonical Hydra overrides for power users (e.g., regression vs ordinal, auxiliary heads) in `docs/configuration.md` with tested examples.

## Codebase Cleanup
- Delete `src/models/components/simple_cnn.py.orig` (obsolete copy) and audit git history for similar artifacts.
- Update `src/models/components/simple_cnn.py` docstrings/examples to use VIMH audio terminology; remove the MNIST-focused `__main__` block or replace it with a VIMH smoke test.
- Narrow broad `try/except` blocks in `_configure_vimh_model_config` (src/train.py) so metadata errors fail fast instead of silently falling back to empty heads.
- Provide automated sanity tests for regression-mode auto-configuration (add pytest case ensuring loss weights and criteria are populated from a fixture metadata file).

## Developer Experience
- Generate a consolidated command reference (`make h` snapshot) and surface it in docs/ (perhaps `docs/command_reference.md`) so users can diff updates as the Makefile evolves.
- Add CI targets (or local `make verify-docs`) that lint Markdown/links to catch stale references and enforce consistent heading style.
- Consider publishing example outputs (plots/audio snippets) under `viz/` or `audio_eval_results/README.md` so contributors know the intended presentation quality.
