# Repository Guidelines

## Project Structure & Module Organization
Source code lives under `src/`; training entry point is `src/train.py`, evaluation is `src/eval.py`, data pipelines are in `src/data/`, models in `src/models/`, and common helpers in `src/utils/`. Hydra configs sit in `configs/` with subfolders for `data/`, `model/`, `trainer/`, and `experiment/`. Tests reside in `tests/` (pytest), while datasets go in `data/`, logs in `logs/`, figures in `viz/`, and docs in `docs/`.

## Build, Test, and Development Commands
Run `make tq` for a quick smoke train (1 epoch on VIMH). Launch full experiments via `python src/train.py experiment=<name>`; e.g., `experiment=example`. Switch configs with overrides such as `python src/train.py model=cnn_medium data=vimh trainer=mps`. Execute fast tests with `make test` or `pytest -k "not slow"`; run the entire suite with `make test-all`. Format and lint with `make format`. Start TensorBoard using `make tensorboard` and open `http://localhost:6006`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents and keep line length under 99 (Black default). Public APIs should declare type hints. Use snake_case for modules and functions, CamelCase for classes, and UPPER_CASE for constants. Config names stay lower_snake (e.g., `cnn_medium`). Formatting and linting flow through Black, isort (black profile), docformatter, flake8, and bandit; run them together with `make format` before pushing.

## Testing Guidelines
Tests use pytest with markers; long scenarios should be annotated `@pytest.mark.slow`. Locate tests under `tests/test_*.py` and name functions `test_*`. Validate changes locally with `pytest -q`; add `pytest` (full run) when altering training logic or configs. Keep assertions meaningful and avoid network-bound or heavyweight data downloads inside unit tests.

## Commit & Pull Request Guidelines
Stage only files touched for the current change set. Write commit subjects in the imperative mood (optional scope like `train:`). Pull requests should outline the change, Hydra command(s) executed, and sample logs from `logs/train/runs/`, plus any linked issues. Ensure tests pass, `make format` is clean, and configs are reproducible via explicit overrides.

## Security & Configuration Tips
Avoid remote checkpoints; rely on local paths only. Prefer Hydra overrides (`trainer.max_epochs=3`) rather than editing YAML defaults. Environment and path helpers come from `rootutils`, which sets `PROJECT_ROOT` for stable file references.
