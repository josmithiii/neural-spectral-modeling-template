Release Readiness Summary and Known Issues - 9/07/25 - codex

Scope of validation
- Verified imports across key modules: `src.train`, `src.eval`, `src.utils.architecture_utils`, `src.data.vimh_datamodule`, `src.models.vimh_lit_module`, core model components.
- Ran targeted unit tests with plugin autoload disabled to avoid environment-specific plugin issues:
  - VIMH dataset unit tests (subset) pass.
  - VIMH datamodule setup/dataloaders (subset) pass.
  - Vision Transformer forward pass parametrized tests pass.
  - Config composition instantiation passes.
- Adjusted preflight label-diversity check to avoid masking datamodule setup errors and to fail clearly when data is missing.

Changes made
- src/train.py: Fix preflight behavior
  - Do not swallow exceptions from `datamodule.setup('fit')`.
  - If `train_dataloader()` is not available, skip preflight gracefully instead of raising indirect `TypeError`s inside `torch.utils.data`.
  - Rationale: When dataset is missing or setup legitimately fails, the training entry should surface a clear, actionable error message. This also prevents confusing errors during quick smoke tests.

What passed locally (subset)
- tests/test_vimh_datasets.py::TestVIMHDataset::test_init_train_dataset ✔
- tests/test_vimh_datasets.py::TestVIMHDataModule::test_setup_datamodule ✔
- tests/test_vision_transformer.py::test_vit_forward_pass (all parametrizations) ✔
- tests/test_configs.py::test_train_config ✔

Important known issues and follow-ups

FIXED: 1) Default training requires a local VIMH dataset
   - Symptom: Running training (including `fast_dev_run`) fails with `FileNotFoundError` if the default VIMH dataset directory doesn’t exist.
   - Details: `configs/data/vimh.yaml` points to `${paths.root_dir}/data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p`. If this path is missing, `VIMHDataModule.prepare_data()` raises a `FileNotFoundError` with instructions.
   - Recommendation:
     - Generate a small sample dataset before training: `make sds` (or `python generate_vimh.py`).
     - Or override the path for experiments: `python src/train.py data.data_dir=<your_dataset_dir>`.
     - Optionally provide a “debug/dummy” datamodule or experiment config that creates an in-memory dataset for quick smoke tests.

2) Preflight label-diversity check behavior
   - Status: Fixed to avoid swallowing setup errors. Now fails clearly when data is missing, or skips if loaders are unavailable.
   - Recommendation: Keep enabled for real training; disable with `preflight.enabled=false` when running on bespoke datasets or during bring-up if necessary.

3) Pytest in constrained environments
   - Symptom: In some sandboxes, `pytest` may hang or crash due to third-party plugin auto-discovery.
   - Workaround: Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to ensure fast, deterministic unit test execution.
     Example: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -k "not slow"`.

FIXED: 4) Quick train targets depend on data
   - `make trq` (and most experiment targets) require a dataset on disk. Ensure you run a synth step first (`make sds`) or point `data.data_dir` to an existing VIMH dataset.

5) GPU-specific tests and slow markers
   - Some tests are marked `@pytest.mark.slow` or gated by `RunIf(min_gpus=1)`. These are expected to be skipped or omitted from the fast path.

Release checklist suggestions
- [ ] Ensure README/quickstart prominently instructs users to synthesize or provide a dataset before training.
- [ ] Optionally add a minimal “debug” experiment that uses a tiny in-memory or auto-generated dataset for instant smoke tests.
- [x] Preflight check surfaces clear errors and avoids noisy TypeErrors.
- [ ] Consider adding a Makefile target that chains dataset synthesis + quick train (e.g., `make sds trq`).

Notes
- No changes were made to tests, configs, or default datamodule selection to preserve current behavior and expectations. The preflight fix improves error clarity without altering training semantics.
