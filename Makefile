h help:  ## Show help
	@grep -E '^[.a-zA-Z0-9_ -]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | less -R

# GENERATE DATASET MAKE TARGETS "gd"

gds generate-dataset-small: ## Synthesize a small example VIMH dataset with SawSynth + Wah (256 samples)
	(make gdws)          # Default is currently the Saw + Wah dataset below

gdl generate-dataset-large: ## Synthesize a larger example VIMH dataset with SawSynth + Wah (16k samples)
	(make gdwl)          # Default is currently the Saw + Wah dataset below

gdws generate-dataset-wah-small: ## Synthesize VIMH dataset with Saw + Wah - decay-time and pedal-angle varied (256 samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_saw_wah; \
	fi
	ls ./data/

gdwl generate-dataset-wah-large: ## Synthesize VIMH dataset with Saw + Wah - decay-time and pedal-angle varied (16k samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_saw_wah dataset.size=16384; \
	fi
	ls ./data/

gdwe generate-dataset-wah-envelope: ## Synthesize VIMH dataset with Saw + Wah ADSR envelope settings varied (512 samples)
	@if [ -f ./data/vimh-32x64x1_8000Hz_2p0s_512dss_wah_envelope_9p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x64x1_8000Hz_2p0s_512dss_wah_envelope_9p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_wah_envelope; \
	fi
	ls ./data/

gdmb generate-dataset-moog-basic: ## Synthesize VIMH dataset with basic Saw + Moog VCF (256 samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_4p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_4p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_moog_basic; \
	fi
	ls ./data/

gdme generate-dataset-moog-envelope: ## Synthesize VIMH dataset with Saw + Moog envelope sweeps (512 samples)
	@if [ -f ./data/vimh-32x64x1_8000Hz_2p0s_512dss_saw_wah_10p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x64x1_8000Hz_2p0s_512dss_saw_wah_10p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_moog_envelope; \
	fi
	ls ./data/

gdmr generate-dataset-moog-resonance: ## Synthesize VIMH dataset with Saw + high-resonance Moog exploration (384 samples)
	@if [ -f ./data/vimh-48x48x1_8000Hz_1p5s_384dss_saw_wah_8p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-48x48x1_8000Hz_1p5s_384dss_saw_wah_8p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_moog_resonance; \
	fi
	ls ./data/

gdas generate-dataset-all-small: gdws gdwe gdmb gdme gdmr ## Generate all small datasets

# DUMP VIMH DATASET METADATA "vd"

vdr vd vimh-dump-recent: ## Dump the metadata of the most recent VIMH dataset in ./data/
	python vimhd.py

vds vimh-dump-small: ## Dump the metadata of the small example SawSynth dataset (256 samples)
	python vimhd.py ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

vdl vimh-dump-large: ## Dump the metadata of the larger example VIMH dataset with SawSynth (16k samples)
	python vimhd.py ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p

# ANALYZE VIMH PARAMETER DISTRIBUTIONS "vp"

vpr vp vimh-params-recent: ## Analyze parameter distributions in the most recent VIMH dataset
	python vimhd.py -p

vps vimh-params-small: ## Analyze parameter distributions in the small example SawSynth dataset (256 samples)
	python vimhd.py -p ./data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

vpl vimh-params-large: ## Analyze parameter distributions in the larger example VIMH dataset (16k samples)
	python vimhd.py -p ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p

# DISPLAY VIMH DATASETS "dd"

ddr display-dataset-recent: ## Display the most recently created dataset (default)
	python display_vimh.py

dds display-dataset-small: generate-dataset-small ## Display the small example VIMH dataset (256 samples)
	python display_vimh.py

ddl display-dataset-large: gdl ## Display the larger example VIMH dataset (16k samples)
	python display_vimh.py

# EXPERIMENTS "e" - Complete Configuration Examples

ex exp-example: generate-dataset-small ## Train CNN on default dataset
	time python src/train.py experiment=example  # ./configs/experiment/example.yaml

# TRIVIAL DATASET EXPERIMENTS "et" - Small models for testing on trivial synthetic data

etms exp-trivial-micro-small: gds ## Micro CNN (~2K params) on small dataset (256 samples) - ordinal classification loss
	time python src/train.py experiment=trivial_micro_small

etmsr exp-trivial-micro-small-regression: gds ## Micro CNN (~2K params) on small dataset (256 samples) - regression loss
	time python src/train.py experiment=trivial_micro_small_regression

etmsrdt exp-trivial-micro-small-regression-decay-time: gds ## Micro CNN (~2K params) on small dataset (256 samples) - regression loss on log10_decay_time only
	time python src/train.py experiment=trivial_micro_small_regression callbacks.model_checkpoint.monitor="val/log10_decay_time_mae" callbacks.early_stopping.monitor="val/log10_decay_time_mae" optimized_metric="val/log10_decay_time_mae"

etts exp-trivial-tiny-small: gds ## Tiny CNN (~8K params) on small dataset (256 samples)
	time python src/train.py experiment=trivial_tiny_small

etml exp-trivial-micro-large: gdl ## Micro CNN (~2K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_micro_large

etmlr exp-trivial-micro-large-regression: gdl ## Micro CNN (~2K params) on large dataset (16K samples) - regression loss
	time python src/train.py experiment=trivial_micro_large_regression

etmlrdt exp-trivial-micro-large-regression-decay-time: gdl ## Micro CNN (~2K params) on large dataset (16K samples) - regression loss on log10_decay_time only
	time python src/train.py experiment=trivial_micro_large_regression callbacks.model_checkpoint.monitor="val/log10_decay_time_mae" callbacks.early_stopping.monitor="val/log10_decay_time_mae" optimized_metric="val/log10_decay_time_mae"

etmldt exp-trivial-micro-large-decay-time: gdl ## Micro CNN (~2K params) on large dataset (16K samples) - ordinal loss on log10_decay_time only
	time python src/train.py experiment=trivial_micro_large callbacks.model_checkpoint.monitor="val/log10_decay_time_acc" callbacks.early_stopping.monitor="val/log10_decay_time_acc" optimized_metric="val/log10_decay_time_acc"

ettl exp-trivial-tiny-large: gdl ## Tiny CNN (~8K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_tiny_large

etmedl exp-trivial-medium-large: gdl ## "Medium" CNN (actually 1.4M params) on large dataset - for comparison
	time python src/train.py experiment=trivial_medium_large

etall: ex etms etts etml ettl et64l ## Run all trivial dataset experiments: ex etms etts etml ettl et64l

# TRIVIAL DATASET ViT EXPERIMENTS "evit" - Small ViT models for testing on trivial synthetic data

evitms exp-trivial-vit-micro-small: gds ## Micro ViT (~8K params) on small dataset (256 samples)
	time python src/train.py experiment=trivial_vit_micro_small

evitmsr exp-trivial-vit-micro-small-regression: gds ## Micro ViT (~8K params) on small dataset (256 samples) - regression variant (placeholder)
	time python src/train.py experiment=trivial_vit_micro_small_regression

evitts exp-trivial-vit-tiny-small: gds ## Tiny ViT (~25K params) on small dataset (256 samples)
	time python src/train.py experiment=trivial_vit_tiny_small

evitml exp-trivial-vit-micro-large: gdl ## Micro ViT (~8K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_vit_micro_large

evittl exp-trivial-vit-tiny-large: gdl ## Tiny ViT (~25K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_vit_tiny_large

evitall: evitms evitts evitml evittl ## Run all ViT trivial dataset experiments

# MOOG VCF DATASET EXPERIMENTS "em" - CNN training on Moog filter datasets

emb exp-moog-basic: gdmb ## Train CNN on basic Moog VCF dataset (4 params)
	time python src/train.py experiment=moog_cnn_basic

eme exp-moog-envelope: gdme ## Train CNN on Moog envelope sweep dataset (10 params)
	time python src/train.py experiment=moog_cnn_envelope

emer exp-moog-envelope-regression: gdme ## Train CNN on Moog envelope sweep dataset (10 params) using regression loss
	time python src/train.py experiment=moog_cnn_envelope_regression

emr exp-moog-resonance: gdmr ## Train CNN on high-resonance Moog dataset (8 params)
	time python src/train.py experiment=moog_cnn_resonance

ew exp-wah: gdws ## Train CNN on dataset gdws (small sawtooth + wah + decay envelope)
	time python src/train.py experiment=wah_cnn

ewl exp-wah-large: gdwl ## Train "large" (~1.4M) CNN on dataset gdwl (large sawtooth + wah + decay envelope) [~14 min to train on Mac MPS]
	time python src/train.py experiment=wah_cnn_large

ewla exp-wah-large-aux: gdwl ## Train "large" (~1.4M) Hybrid CNN-MLP on dataset gdwl (large sawtooth + wah + decay envelope) extracting decay as aux feature
	time python src/train.py experiment=wah_cnn_large_aux

ewt exp-wah-tiny: gdwl ## Train "tiny" (~41K) CNN on dataset gdwl (large sawtooth + wah + decay envelope) [~10 min to train on Mac MPS]
	time python src/train.py experiment=wah_cnn_tiny

ewtq exp-wah-tiny-quick: gdwl ## Quick version ewt (exp-wah-tiny) to produce a checkpoint fast for testing (1 epoch, small dataset)
	time python src/train.py experiment=wah_cnn_tiny trainer.max_epochs=1 data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

ewtr exp-wah-tiny-regression: gdwl ## Train "tiny" (1.1M) CNN-pure-regression on dataset gdwl (large sawtooth + wah + decay envelope) [~4.2 min to train on Mac MPS]
	time python src/train.py experiment=wah_cnn_tiny_regression

# Quick variant of ewtr: 1 epoch on the small dataset to produce a checkpoint fast
ewtrq exp-wah-tiny-regression-quick: gdws ## Quick regression run (1 epoch, small dataset) to produce a checkpoint fast
	python src/train.py experiment=wah_cnn_tiny_regression trainer.max_epochs=1 data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

ewta exp-wah-tiny-aux: gdwl ## Train "tiny" (~43K) Hybrib CNN-MLP on dataset gdwl (large sawtooth + wah + decay envelope) extracting decay as aux feature
	time python src/train.py experiment=wah_cnn_tiny_aux  # Result: hurts slightly - bug?  It helped in the regression case

ewe exp-wah-envelope: gdwe ## CNN training on dataset gdwe (gdw + ADSR wah control)
	time python src/train.py experiment=wah_cnn_envelope

emall: emall-gen emb eme emr ## Generate datasets and train CNNs on all Moog VCF experiments

emall-gen: gdmb gdme gdmr ## Generate all Moog datasets before training

emall-train: emb eme emr ## Run all Moog dataset training experiments

# MOOG VCF ViT EXPERIMENTS "emvit" - ViT training on Moog filter datasets (experimental)

emvitb exp-moog-vit-basic: gdmb ## Train ViT on basic Moog VCF dataset (4 params) - square 32x32
	time python src/train.py experiment=moog_vit_basic

emvite exp-moog-vit-envelope: gdme ## Train ViT on Moog envelope sweep dataset (10 params) - rectangular 32x64
	time python src/train.py experiment=moog_vit_envelope

emvitr exp-moog-vit-resonance: gdmr ## Train ViT on high-resonance Moog dataset (8 params) - square 48x48
	time python src/train.py experiment=moog_vit_resonance

emvit emvit-train-all: emvitb emvite emvitr ## Run all Moog ViT training experiments

emvitgta emvit-gen-train-all: emall-gen emvit ## Generate datasets and train ViTs on all Moog VCF experiments

# EVALS "ev" - Evaluate saved checkpoints

evwt eval-wah-tiny: ## Evaluate latest wah_cnn_tiny best checkpoint (set CKPT=path/to.ckpt to override)
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

evwtr eval-wah-tiny-regression: ## Evaluate latest wah_cnn_tiny_regression best checkpoint (set CKPT=path/to.ckpt to override)
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
ae ae_latest audio-eval-latest: ## Display eval of latest best model checkpoint using default dataset using src/audio_reconstruction_eval.py
	python src/audio_reconstruction_eval.py

# AUDIO-EVAL helpers: select specific runs/checkpoints quickly
# - Use `make ae_reg` to open latest run tagged with "regression"
# - Use `make ae_prev` to open the second most recent run (any experiment)
# - Use `make ae_filter FILTER=<grep>` to open the latest run whose tags.log contains FILTER
#   Examples:
#     make aef FILTER=wah_cnn_tiny
#     make aef FILTER=regression
# - You can still override with CKPT=path/to/checkpoint.ckpt

aer ae_reg audio-eval-regression: ## Audio-eval latest run with "regression" tag (auto-picks best/last ckpt)
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

aep ae_prev audio-eval-previous: ## Audio-eval the second most recent run (any experiment)
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

aef ae_filter audio-eval-filter: ## Audio-eval latest run whose tags.log contains FILTER=... (optional EXCLUDE=...)
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

ae_cls audio-eval-classification: ## Audio-eval latest classification run (exclude 'regression')
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

cd clean-data: ## Clean data files
	rm -rf data/*

cl clean-logs: ## Clean run logs
	rm -rf logs/**

dc distclean: clean clean-data clean-logs ## Clean back to original distribution (except for hidden files)

# TESTING TARGETS "t", "ta"

t0 tests0: etms sep etmsr sep evitms sep evitmsr ## Tests of basic CNN and ViT archs on classes and regression

t test: ## Run fast pytest tests
	pytest -k "not slow"

ta test-all: ## Run all pytest tests
	pytest

# TEST DIAGRAM TARGETS "td*"

td tda test-diagram-all: ## Generate enhanced diagrams for all architectures (text + graphical)
	python viz/enhanced_model_diagrams.py

tdl test-diagram-list: ## List available model configs for diagrams
	python viz/enhanced_model_diagrams.py --list-configs

tds test-diagram-simple: ## Generate simple text-only diagrams (default cnn_medium)
	python viz/simple_model_diagram.py

tdsc test-diagram-simple-config: ## Generate simple diagram for specific config (usage: make tdsc CONFIG=cnn_medium)
	python viz/simple_model_diagram.py --config $(CONFIG)

tdsl test-diagram-simple-list: ## List available configs for simple diagrams
	python viz/simple_model_diagram.py --list-configs

tdsa test-diagram-simple-all: ## Generate simple diagrams for all architectures
	python viz/simple_model_diagram.py --config cnn_medium
	python viz/simple_model_diagram.py --config cnn_medium_ordinal
	python viz/simple_model_diagram.py --config cnn_medium_regression
	python viz/simple_model_diagram.py --config cnn_medium_auxiliary
	python viz/simple_model_diagram.py --config cnn_micro
	python viz/simple_model_diagram.py --config cnn_tiny
	python viz/simple_model_diagram.py --config vit_micro
	python viz/simple_model_diagram.py --config vit_tiny

tdv test-diagram-vgg: ## Generate VGG-style architecture diagrams (EPS + PNG)
	python viz/vgg_style_diagrams.py

# RAW TRAINING TARGETS "tr" (no "experiment" - use hydra overrides to set desired config - experiments recommended instead)

tr train: generate-dataset-small ## Train default model on default dataset (`make tr`) - defaults defined in ./configs/train.yaml
	time python src/train.py

trq train-quick: gds ## Train super quickly the default model and dataset (quick sanity test to see if things are working)
	python src/train.py trainer.max_epochs=1

trs train-vimh-small: gds ## Train the small example VIMH dataset using the default model (CNN medium)
	time python src/train.py data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p

trl train-vimh-large: gdl ## Train the large example VIMH dataset using the default model (CNN medium)
	time python src/train.py data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p

# UTILITY TARGETS

f format: ## Run pre-commit hooks
	pre-commit run -a

verify-docs: ## Run documentation checks (links, headings, deprecated commands)
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
