# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Training, evaluation, data modules, and models (e.g., `src/train.py`, `src/data/`, `src/models/`).
- `configs/`: Hydra configs for `data/`, `model/`, `trainer/`, and `experiment/` presets.
- `tests/`: Pytest suite (`test_*.py`) with fast and `slow` markers.
- `data/`, `logs/`, `outputs/`, `diagrams/`: Generated artifacts; keep out of commits.
- `viz/`, `examples/`, `docs/`: Diagrams, usage samples, and documentation.

## Build, Test, and Development Commands
- Environment: `sh setup.sh` to create the virtual env; list tasks with `make h`.
- Datasets: `make sds` (small), `make sdl` (large), `make sdma` (Moog set); view with `make ddr`.
- Train: `make tr` (defaults in `configs/train.yaml`), quick check `make trq`, or override: `python src/train.py experiment=trivial_micro_small`.
- Experiments: `make ex` (example), `make etms`/`etts` (trivial CNN), `make emb`/`eme`/`emr` (Moog CNN). List configs: `make lc`.
- Tests: `make t` (fast), `make ta` (all). Format/lint: `make f`. TensorBoard: `make tb`.

## Coding Style & Naming Conventions
- Python 3.8+; 4‑space indent; limit lines to 99 (Black).
- Naming: `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE` constants; tests `test_*.py`.
- Tools (pre-commit): Black, isort, flake8, docformatter, bandit, mdformat, codespell, nbqa. Run `make f` before commits.
- Configs: prefer Hydra overrides over code edits; add reusable YAML under `configs/{data,model,experiment}`.

## Testing Guidelines
- Framework: Pytest with `slow` marker; fast path excludes slow by default.
- Run: `pytest -k "not slow"` or target a file: `pytest tests/test_vimh_datasets.py -q`.
- Add unit tests for new modules and regression tests for fixes; keep data-light and mark heavy/long tests as `@pytest.mark.slow`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subjects; optional scope (e.g., `models:`, `data:`, `main:`). One logical change per commit.
- PRs: clear description, linked issues, reproduction steps, key command(s) used (e.g., Hydra overrides), and results (logs, metrics, or diagrams). Ensure `make f` and `make t` pass; update docs/configs when behavior changes.

## Security & Configuration Tips
- Do not commit secrets; use `.env.example` as a template. Store datasets in `data/` and logs in `logs/`. Use bandit via pre-commit and prefer config-driven changes over hard-coded paths.
