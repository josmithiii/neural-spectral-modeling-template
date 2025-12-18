SHELL := /bin/bash

# ENVIRONMENT CHECKING

check-env: ## Check if virtual environment is activated
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "ERROR: Virtual environment is not activated!"; \
		echo ""; \
		echo "Please activate the environment first:"; \
		echo "  source .venv/bin/activate    # for bash/zsh"; \
		echo "  source .venv/bin/activate.csh # for csh/tcsh"; \
		echo ""; \
		echo "Or run: bash setup.bash to create the environment first"; \
		exit 1; \
	fi
	@echo "Virtual environment is active: $$VIRTUAL_ENV"

h help:  ## Show help
	@grep -E '^[.a-zA-Z0-9_ -]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | less -R

# GENERATE DATASET MAKE TARGETS "gd"

gds generate-dataset-small: check-env ## Synthesize a small example VIMH dataset with SawSynth + Wah (256 samples)
	(make gdws)          # Default is currently the Saw + Wah dataset below

gdl generate-dataset-large: check-env ## Synthesize a larger example VIMH dataset with SawSynth + Wah (16k samples)
	(make gdwl)          # Default is currently the Saw + Wah dataset below

gdws generate-dataset-wah-small: check-env ## Synthesize VIMH dataset with Saw + Wah - decay-time and pedal-angle varied (256 samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_saw_wah; \
	fi
	ls ./data/

gdwl generate-dataset-wah-large: check-env ## Synthesize VIMH dataset with Saw + Wah - decay-time and pedal-angle varied (16k samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_saw_wah dataset.size=16384; \
	fi
	ls ./data/

gdwe generate-dataset-wah-envelope: check-env ## Synthesize VIMH dataset with Saw + Wah ADSR envelope settings varied (512 samples)
	@if [ -f ./data/vimh-32x64x1_8000Hz_2p0s_512dss_wah_envelope_9p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x64x1_8000Hz_2p0s_512dss_wah_envelope_9p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_wah_envelope; \
	fi
	ls ./data/

gdwdel generate-dataset-wah-delay: check-env ## Synthesize VIMH dataset with Saw + Wah + onset delay (3 params: decay, wah, onset_delay_ms) - 16k samples
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_del_3p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_del_3p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_saw_wah_del dataset.size=16384; \
	fi
	ls ./data/

gdas generate-dataset-all-small: gdws gdwe ## Generate all small wah datasets

# DUMP VIMH DATASET METADATA "vd"

vdr vd vimh-dump-recent: check-env ## Dump the metadata of the most recent VIMH dataset in ./data/
	python vimhd.py

vds vimh-dump-small: check-env ## Dump the metadata of the small example SawSynth dataset (256 samples)
	python vimhd.py ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

vdl vimh-dump-large: check-env ## Dump the metadata of the larger example VIMH dataset with SawSynth (16k samples)
	python vimhd.py ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p

# ANALYZE VIMH PARAMETER DISTRIBUTIONS "vp"

vpr vp vimh-params-recent: check-env ## Analyze parameter distributions in the most recent VIMH dataset
	python vimhd.py -p

vps vimh-params-small: check-env ## Analyze parameter distributions in the small example SawSynth dataset (256 samples)
	python vimhd.py -p ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

vpl vimh-params-large: check-env ## Analyze parameter distributions in the larger example VIMH dataset (16k samples)
	python vimhd.py -p ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p

# DISPLAY VIMH DATASETS "dd"

ddr display-dataset-recent: check-env ## Display the most recently created dataset (default)
	python display_vimh.py

dds display-dataset-small: check-env generate-dataset-small ## Display the small example VIMH dataset (256 samples)
	python display_vimh.py ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

ddl display-dataset-large: check-env gdl ## Display the larger example VIMH dataset (16k samples)
	python display_vimh.py ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p

# EXPERIMENTS "e" - Complete Configuration Examples

ex exp-example: check-env generate-dataset-small ## Train CNN on default dataset
	time python src/train.py experiment=example  # ./configs/experiment/example.yaml

ea exp-all: check-env ## Run ALL experiments, capturing outputs in experiment_logs/
	time bash scripts/run_all_experiments.sh --force && python ./scripts/extract_logs.py --csv > experiments_overview.md

en exp-new: check-env ## Run all new experiments not having a log yet, capturing their outputs in experiment_logs/
	time bash scripts/run_all_experiments.sh --jobs 1 && python ./scripts/extract_logs.py --csv > experiments_overview.md

exp-clean: ## Clean all experiment logs in ./experiment_logs/
	/bin/rm -rf ./experiment_logs/*-log.txt

xl extract-logs: check-env
	python ./scripts/extract_logs.py

xlu extract-logs-update: check-env
	python ./scripts/extract_logs.py --csv > experiments_overview.md

# TRIVIAL DATASET EXPERIMENTS "et" - Small models for testing on trivial synthetic data

etms exp-trivial-micro-small: check-env gds ## Micro CNN (~10K params) on small dataset (256 samples) - ordinal classification loss
	time python src/train.py experiment=trivial_micro_small

etmsr exp-trivial-micro-small-regression: check-env gds ## Micro CNN (~10K params) on small dataset (256 samples) - regression loss
	time python src/train.py experiment=trivial_micro_small_regression

etts exp-trivial-tiny-small: check-env gds ## Tiny CNN (~40K params) on small dataset (256 samples)
	time python src/train.py experiment=trivial_tiny_small

# TRIVIAL DATASET ViT EXPERIMENTS "evit" - Small ViT models for testing on trivial synthetic data

evitms exp-trivial-vit-micro-small: check-env gds ## Micro ViT (~23K params) on small dataset (256 samples) - ordinal classification loss
	time python src/train.py experiment=trivial_vit_micro_small

ewt exp-wah-tiny: check-env gdwl ## Train "tiny" (~40K) CNN on dataset gdwl (large sawtooth + wah + decay envelope) [~9 min to train on Mac MPS]
	time python src/train.py experiment=wah_cnn_tiny

ewtq exp-wah-tiny-quick: check-env gdwl ## Quick version ewt (exp-wah-tiny) to produce a checkpoint fast for testing (1 epoch, small dataset)
	time python src/train.py experiment=wah_cnn_tiny trainer.max_epochs=1 data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

ewtr exp-wah-tiny-regression: check-env gdwl ## Train "tiny" (~40K) CNN-regression on dataset gdwl (large sawtooth + wah + decay envelope)
	time python src/train.py experiment=wah_cnn_tiny_regression

# Quick variant of ewtr: 1 epoch on the small dataset to produce a checkpoint fast
ewtrq exp-wah-tiny-regression-quick: check-env gdws ## Quick tiny regression run (1 epoch, small dataset) to produce a checkpoint fast
	python src/train.py experiment=wah_cnn_tiny_regression trainer.max_epochs=1 data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

ewta exp-wah-tiny-auxiliary: check-env gdwl ## Train "tiny" (~40K) CNN with auxiliary input (decay_time) on dataset gdwl - classification
	time python src/train.py experiment=wah_cnn_tiny_auxiliary

ewtar exp-wah-tiny-auxiliary-regression: check-env gdwl ## Train "tiny" (~40K) CNN with auxiliary input (decay_time) on dataset gdwl - regression
	time python src/train.py experiment=wah_cnn_tiny_auxiliary_regression

ewm exp-wah-medium: check-env gdwl ## Train "medium" (~1.1M) CNN on dataset gdwl (large sawtooth + wah + decay envelope)
	time python src/train.py experiment=wah_cnn_medium

ewmr exp-wah-medium-regression: check-env gdwl ## Train "medium" (1.1M) CNN-regression on dataset gdwl (large sawtooth + wah + decay envelope) [~4.6 min to train on Mac MPS]
	time python src/train.py experiment=wah_cnn_medium_regression

ewma exp-wah-medium-auxiliary: check-env gdwl ## Train "medium" (~1.1M) CNN with auxiliary input (decay_time) on dataset gdwl - classification
	time python src/train.py experiment=wah_cnn_medium_auxiliary

ewmar exp-wah-medium-auxiliary-regression: check-env gdwl ## Train "medium" (~1.1M) CNN with auxiliary input (decay_time) on dataset gdwl - regression
	time python src/train.py experiment=wah_cnn_medium_auxiliary_regression

ewvt exp-wah-vit-tiny: check-env gdwl ## Train "tiny" (~25K) ViT on dataset gdwl (large sawtooth + wah + decay envelope)
	time python src/train.py experiment=wah_vit_tiny

ewvtr exp-wah-vit-tiny-regression: check-env gdwl ## Train "tiny" (~25K) ViT-regression on dataset gdwl (large sawtooth + wah + decay envelope)
	time python src/train.py experiment=wah_vit_tiny_regression

ewvm exp-wah-vit-medium: check-env gdwl ## Train "medium" (~1M+) ViT on dataset gdwl (large sawtooth + wah + decay envelope)
	time python src/train.py experiment=wah_vit_medium

ewvmr exp-wah-vit-medium-regression: check-env gdwl ## Train "medium" (~1M+) ViT-regression on dataset gdwl (large sawtooth + wah + decay envelope)
	time python src/train.py experiment=wah_vit_medium_regression

# SAW+WAH+DELAY (WDEL) VIMH EXPERIMENTS
# 3 params: log10_decay_time, wah_position, onset_delay_ms

ewdel exp-wah-delay-cnn-medium: check-env gdwdel ## Train VIMH medium CNN on saw+wah+delay (3 params, 16k samples)
	time python src/train.py experiment=wah_del_cnn_medium

# EVALS "ev" - Evaluate saved checkpoints

evwt eval-wah-tiny: check-env ## Evaluate latest wah_cnn_tiny best checkpoint (set CKPT=path/to.ckpt to override)
	@set -e; \
	if [ -z "$(CKPT)" ]; then \
		echo "[eval] Locating latest wah_cnn_tiny (classification) run..."; \
		RUN_DIR=$$(for d in $$(ls -td logs/train/runs/* 2>/dev/null); do \
			if [ -f $$d/tags.log ] \
			   && grep -q "wah" $$d/tags.log \
			   && grep -q "cnn_tiny" $$d/tags.log \
			   && ! grep -q "regression" $$d/tags.log; then \
				echo $$d; break; \
			fi; \
		done); \
		if [ -z "$$RUN_DIR" ]; then echo "No matching run found in logs/train/runs"; exit 1; fi; \
		echo "[eval] Using run: $$RUN_DIR"; \
		CKPT_PATH=$$(ls -t $$RUN_DIR/checkpoints/epoch_*.ckpt 2>/dev/null | head -1); \
		if [ -z "$$CKPT_PATH" ]; then CKPT_PATH=$$RUN_DIR/checkpoints/last.ckpt; fi; \
		echo "[eval] Using checkpoint: $$CKPT_PATH"; \
		time python src/train.py experiment=wah_cnn_tiny train=false test=true ckpt_path=$$CKPT_PATH; \
	else \
		echo "[eval] Using checkpoint: $(CKPT)"; \
		time python src/train.py experiment=wah_cnn_tiny train=false test=true ckpt_path=$(CKPT); \
	fi

evwtr eval-wah-tiny-regression: check-env ## Evaluate latest wah_cnn_tiny_regression best checkpoint (set CKPT=path/to.ckpt to override)
	@set -e; \
	if [ -z "$(CKPT)" ]; then \
		echo "[eval] Locating latest wah_cnn_tiny_regression run..."; \
		RUN_DIR=$$(for d in $$(ls -td logs/train/runs/* 2>/dev/null); do \
			if [ -f $$d/tags.log ] \
			   && grep -q "wah" $$d/tags.log \
			   && grep -q "cnn_tiny" $$d/tags.log \
			   && grep -q "regression" $$d/tags.log; then \
				echo $$d; break; \
			fi; \
		done); \
		if [ -z "$$RUN_DIR" ]; then echo "No matching regression run found in logs/train/runs"; exit 1; fi; \
		echo "[eval] Using run: $$RUN_DIR"; \
		CKPT_PATH=$$(ls -t $$RUN_DIR/checkpoints/epoch_*.ckpt 2>/dev/null | head -1); \
		if [ -z "$$CKPT_PATH" ]; then CKPT_PATH=$$RUN_DIR/checkpoints/last.ckpt; fi; \
		echo "[eval] Using checkpoint: $$CKPT_PATH"; \
		time python src/train.py experiment=wah_cnn_tiny_regression train=false test=true ckpt_path=$$CKPT_PATH; \
	else \
		echo "[eval] Using checkpoint: $(CKPT)"; \
		time python src/train.py experiment=wah_cnn_tiny_regression train=false test=true ckpt_path=$(CKPT); \
	fi

# AUDIO EVAL
ae ae_latest audio-eval-latest: check-env ## Display eval of latest best model checkpoint using default dataset using src/audio_reconstruction_eval.py
	python src/audio_reconstruction_eval.py

# AUDIO-EVAL helpers: select specific runs/checkpoints quickly
# - Use `make ae_reg` to open latest run tagged with "regression"
# - Use `make ae_prev` to open the second most recent run (any experiment)
# - Use `make ae_filter FILTER=<grep>` to open the latest run whose tags.log contains FILTER
#   Examples:
#     make aef FILTER=wah_cnn_tiny
#     make aef FILTER=regression
# - You can still override with CKPT=path/to/checkpoint.ckpt

aer ae_reg audio-eval-regression: check-env ## Audio-eval latest run with "regression" tag (auto-picks best/last ckpt)
	@set -e; \
	if [ -z "$(CKPT)" ]; then \
		echo "[ae] Locating latest run tagged 'regression'..."; \
		RUN_DIR=$$(for d in $$(ls -td logs/train/runs/* 2>/dev/null); do \
			if [ -f $$d/tags.log ] && grep -qi "regression" $$d/tags.log; then echo $$d; break; fi; \
		done); \
		if [ -z "$$RUN_DIR" ]; then echo "No matching regression run found in logs/train/runs"; exit 1; fi; \
		echo "[ae] Using run: $$RUN_DIR"; \
		CKPT_PATH=$$(ls -t $$RUN_DIR/checkpoints/epoch_*.ckpt 2>/dev/null | head -1); \
		if [ -z "$$CKPT_PATH" ]; then CKPT_PATH=$$RUN_DIR/checkpoints/last.ckpt; fi; \
		echo "[ae] Using checkpoint: $$CKPT_PATH"; \
		python src/audio_reconstruction_eval.py ckpt_path=$$CKPT_PATH; \
	else \
		echo "[ae] Using checkpoint: $(CKPT)"; \
		python src/audio_reconstruction_eval.py ckpt_path=$(CKPT); \
	fi

aep ae_prev audio-eval-previous: check-env ## Audio-eval the second most recent run (any experiment)
	@set -e; \
	if [ -z "$(CKPT)" ]; then \
		echo "[ae] Locating previous run (2nd newest)..."; \
		RUN_DIR=$$(ls -td logs/train/runs/* 2>/dev/null | sed -n '2p'); \
		if [ -z "$$RUN_DIR" ]; then echo "No previous run found in logs/train/runs"; exit 1; fi; \
		echo "[ae] Using run: $$RUN_DIR"; \
		CKPT_PATH=$$(ls -t $$RUN_DIR/checkpoints/epoch_*.ckpt 2>/dev/null | head -1); \
		if [ -z "$$CKPT_PATH" ]; then CKPT_PATH=$$RUN_DIR/checkpoints/last.ckpt; fi; \
		echo "[ae] Using checkpoint: $$CKPT_PATH"; \
		python src/audio_reconstruction_eval.py ckpt_path=$$CKPT_PATH; \
	else \
		echo "[ae] Using checkpoint: $(CKPT)"; \
		python src/audio_reconstruction_eval.py ckpt_path=$(CKPT); \
	fi

aef ae_filter audio-eval-filter: check-env ## Audio-eval latest run whose tags.log contains FILTER=... (optional EXCLUDE=...)
	@set -e; \
	if [ -z "$(CKPT)" ]; then \
		if [ -z "$(FILTER)" ]; then echo "Provide FILTER=... or CKPT=..."; exit 2; fi; \
		echo "[ae] Locating latest run with filter '$(FILTER)'$(if $(EXCLUDE), and excluding '$(EXCLUDE)',)..."; \
		RUN_DIR=$$(for d in $$(ls -td logs/train/runs/* 2>/dev/null); do \
			if [ -f $$d/tags.log ] && grep -qi "$(FILTER)" $$d/tags.log $(if $(EXCLUDE),&& ! grep -qi "$(EXCLUDE)" $$d/tags.log,); then echo $$d; break; fi; \
		done); \
		if [ -z "$$RUN_DIR" ]; then echo "No run matched FILTER='$(FILTER)'"; exit 1; fi; \
		echo "[ae] Using run: $$RUN_DIR"; \
		CKPT_PATH=$$(ls -t $$RUN_DIR/checkpoints/epoch_*.ckpt 2>/dev/null | head -1); \
		if [ -z "$$CKPT_PATH" ]; then CKPT_PATH=$$RUN_DIR/checkpoints/last.ckpt; fi; \
		echo "[ae] Using checkpoint: $$CKPT_PATH"; \
		python src/audio_reconstruction_eval.py ckpt_path=$$CKPT_PATH; \
	else \
		echo "[ae] Using checkpoint: $(CKPT)"; \
		python src/audio_reconstruction_eval.py ckpt_path=$(CKPT); \
	fi

ae_cls audio-eval-classification: check-env ## Audio-eval latest classification run (exclude 'regression')
	@set -e; \
	if [ -z "$(CKPT)" ]; then \
		echo "[ae] Locating latest classification run (excluding 'regression')..."; \
		RUN_DIR=$$(for d in $$(ls -td logs/train/runs/* 2>/dev/null); do \
			if [ -f $$d/tags.log ] && ! grep -qi "regression" $$d/tags.log; then echo $$d; break; fi; \
		done); \
		if [ -z "$$RUN_DIR" ]; then echo "No classification run found in logs/train/runs"; exit 1; fi; \
		echo "[ae] Using run: $$RUN_DIR"; \
		CKPT_PATH=$$(ls -t $$RUN_DIR/checkpoints/epoch_*.ckpt 2>/dev/null | head -1); \
		if [ -z "$$CKPT_PATH" ]; then CKPT_PATH=$$RUN_DIR/checkpoints/last.ckpt; fi; \
		echo "[ae] Using checkpoint: $$CKPT_PATH"; \
		python src/audio_reconstruction_eval.py ckpt_path=$$CKPT_PATH; \
	else \
		echo "[ae] Using checkpoint: $(CKPT)"; \
		python src/audio_reconstruction_eval.py ckpt_path=$(CKPT); \
	fi

# CLEANING MAKE TARGETS

c clean: ## Clean all generated files except logs and datasets
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage
	rm -rf ./diagrams/
	rm -rf ./outputs/
	rm -rf ./audio_eval_results/
	(cd ./docs/presentation && make clean)

cd clean-data: ## Clean data files
	rm -rf data/*

cl clean-logs: ## Clean run logs
	rm -rf logs/**

dc distclean: clean clean-data clean-logs ## Clean back to original distribution (except for hidden files)

# TESTING TARGETS "t", "ta"

t0 tests0: check-env etms sep etmsr sep evitms ## Tests of basic CNN and ViT archs on classes and regression

t test: check-env ## Run fast pytest tests
	pytest -k "not slow"

ta test-all: check-env ## Run all pytest tests
	pytest

# TEST DIAGRAM TARGETS "td*"

td tda text-diagram-all: check-env ## Generate enhanced diagrams for all architectures (text + graphical)
	python viz/enhanced_model_diagrams.py

tdl text-diagram-list: check-env ## List available model configs for diagrams
	python viz/enhanced_model_diagrams.py --list-configs

tds text-diagram-simple: check-env ## Generate simple text-only diagrams (default cnn_medium)
	python viz/simple_model_diagram.py

tdsc text-diagram-simple-config: check-env ## Generate simple diagram for specific config (usage: make tdsc CONFIG=cnn_medium)
	python viz/simple_model_diagram.py --config $(CONFIG)

tdsl text-diagram-simple-list: check-env ## List available configs for simple diagrams
	python viz/simple_model_diagram.py --list-configs

tdsa text-diagram-simple-all: check-env ## Generate simple diagrams for all architectures
	python viz/simple_model_diagram.py --config cnn_medium
	python viz/simple_model_diagram.py --config cnn_medium_ordinal
	python viz/simple_model_diagram.py --config cnn_medium_regression
	python viz/simple_model_diagram.py --config cnn_medium_auxiliary
	python viz/simple_model_diagram.py --config cnn_micro
	python viz/simple_model_diagram.py --config cnn_tiny
	python viz/simple_model_diagram.py --config vit_micro
	python viz/simple_model_diagram.py --config vit_tiny

tdv text-diagram-vgg: check-env ## Generate VGG-style architecture diagrams (EPS + PNG)
	python viz/vgg_style_diagrams.py

# RAW TRAINING TARGETS "tr" (no "experiment" - use hydra overrides to set desired config - experiments recommended instead)

tr train: check-env generate-dataset-small ## Train default model on default dataset (`make tr`) - defaults defined in ./configs/train.yaml
	time python src/train.py

trq train-quick: check-env gds ## Train super quickly the default model and dataset (quick sanity test to see if things are working)
	python src/train.py trainer.max_epochs=1

trs train-vimh-small: check-env gds ## Train the small example VIMH dataset using the default model (CNN medium)
	time python src/train.py data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

trl train-vimh-large: check-env gdl ## Train the large example VIMH dataset using the default model (CNN medium)
	time python src/train.py data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p

# UTILITY TARGETS

f format: ## Run pre-commit hooks
	pre-commit run -a

verify-docs: check-env ## Run documentation checks (links, headings, deprecated commands)
	python scripts/verify_docs.py

fp format-preview: ## Preview docformatter actions
	docformatter -c -d -r --black --wrap-summaries=99 --wrap-descriptions=99 --style=sphinx src tests

fdc format-docstrings-check: ## Run docformatter pre-commit hook (manual stage)
	pre-commit run docformatter --hook-stage manual -a

fm format-markdown: ## Run Prettier on Markdown/YAML only
	pre-commit run prettier -a

fn flake8-now: ## Run flake8 lint manually on src/tests and key scripts
	pre-commit run flake8 --hook-stage manual -a

fc format-configs: ## Prettier-format only YAML in configs/
	@FILES=$(shell git ls-files 'configs/**/*.yaml' 'configs/*.yaml'); \
	if [ -n "$$FILES" ]; then \
		pre-commit run prettier --files $$FILES; \
	else \
		echo "No config YAML files found"; \
	fi

sy sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main

tb tensorboard: ## Launch TensorBoard on port 6006
	@lsof -i :6006 >/dev/null 2>&1 && echo "TensorBoard already running on port 6006" || \
		(echo "Starting TensorBoard on port 6006..." && tensorboard --logdir logs/train/runs/ --reload_interval 1 --port 6006 &)
	@echo "Open http://localhost:6006/"

tbr tensorboard-regression: ## Launch TensorBoard on port 6008
	@lsof -i :6006 >/dev/null 2>&1 && echo "TensorBoard already running on port 6008" || \
		(echo "Starting TensorBoard for Regression on port 6008..." && tensorboard --logdir logs/train/runs/ --tag=val/loss --tag=val/mae_best --reload_interval 1 --port 6008 &)
	@echo "Open http://localhost:6006/"

a activate: ## Activate the uv environment
	@echo "Add to ~/.tcshrc: alias a 'echo \"source .venv/bin/activate.csh\" && source .venv/bin/activate.csh'"
	@echo "Then just type: a"

d deactivate: ## Deactivate the uv environment
	@echo "Add to ~/.tcshrc: alias d 'echo deactivate && deactivate'"
	@echo "Then just type: d"

lc list-configs: ## List available model configurations
	@echo "Available model configs:"
	@find configs/model -name "*.yaml" | sed 's|configs/model/||' | sed 's|\.yaml||' | sort
	@echo "\nAvailable data configs:"
	@find configs/data -name "*.yaml" | sed 's|configs/data/||' | sed 's|\.yaml||' | sort
	@echo "\nAvailable experiment configs:"
	@find configs/experiment -name "*.yaml" | sed 's|configs/experiment/||' | sed 's|\.yaml||' | sort

sep:
	@printf '%*s\n' 180 '' | tr ' ' '+'
