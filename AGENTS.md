# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` (training in `src/train.py`, evaluation in `src/eval.py`, data modules under `src/data/`, models under `src/models/`, utilities under `src/utils/`).
- Configs: `configs/` (Hydra YAML for `data/`, `model/`, `trainer/`, `experiment/`).
- Tests: `tests/` (pytest suite with fast and `slow` markers).
- Assets & outputs: datasets under `data/`, logs under `logs/`, figures under `viz/`, docs in docs/*.

## Build, Test, and Development Commands
- Quick train smoke test: `make tq` (1 epoch on VIMH with small batches).
- Run experiments: `python src/train.py experiment=<name>` (e.g., `experiment=example`).
- Select configs: `python src/train.py model=cnn_64k data=vimh trainer=mps`.
- Tests (fast): `make test` or `pytest -k "not slow"`.
- Tests (all): `make test-all`.
- Format & lint: `make format` (pre-commit: black, isort, flake8, bandit, etc.).
- TensorBoard: `make tensorboard` then open `http://localhost:6006`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indents, type hints for public APIs.
- Formatting: black (line length 99), isort (black profile), docformatter; run via `make format`.
- Linting: flake8 (with selected ignores) and bandit; keep warnings low.
- Config names: lower_snake (e.g., `cnn_64k`, `vimh_16kdss`).
- Modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_CASE`.

## Testing Guidelines
- Framework: pytest with markers and parametrization.
- Location: `tests/test_*.py`; name tests `test_*` and mark long ones `@pytest.mark.slow`.
- Run locally before PR: `pytest -q` (fast path), then full `pytest` if changing training behavior).
- Coverage: keep meaningful assertions; avoid network and large downloads in unit tests.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject; optional scope (e.g., `train: fix audio eval`).
- PRs: include description of changes, Hydra command(s) used, sample logs (`logs/train/runs/`), and linked issues.
- Checklist: tests pass, `make format` clean, configs reproducible (pin overrides in the PR body).

## Security & Configuration Tips
- Checkpoints: loading remote URLs is blocked; use local files only.
- Config: prefer Hydra overrides (e.g., `trainer.max_epochs=3`) instead of editing YAML.
- Env: see `.env.example`, `requirements.txt`, and `environment.yaml`; `rootutils` sets `PROJECT_ROOT` for stable paths.
