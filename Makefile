h help:  ## Show help
	@grep -E '^[.a-zA-Z0-9_ -]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | less -R

# SYNTHETIC DATASET MAKE TARGETS "sd"

sds synth-dataset-small: ## Synthesize a small example VIMH dataset with SawSynth (256 samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_simple_saw; \
	fi
	ls ./data/

sdl synth-dataset-large: ## Synthesize a larger example VIMH dataset with SawSynth (16k samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p"; \
	else \
		time python generate_vimh.py --config-name=synth/generate_simple_saw dataset.size=16384; \
	fi
	ls ./data/

sdmb synth-dataset-moog-basic: ## Synthesize VIMH dataset with basic Saw + Moog VCF (256 samples)
	@if [ -f ./data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_4p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_4p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_moog_basic; \
	fi
	ls ./data/

sdme synth-dataset-moog-envelope: ## Synthesize VIMH dataset with Saw + Moog envelope sweeps (512 samples)
	@if [ -f ./data/vimh-32x64x1_8000Hz_2p0s_512dss_simple_10p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-32x64x1_8000Hz_2p0s_512dss_simple_10p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_moog_envelope; \
	fi
	ls ./data/

sdmr synth-dataset-moog-resonance: ## Synthesize VIMH dataset with Saw + high-resonance Moog exploration (384 samples)
	@if [ -f ./data/vimh-48x48x1_8000Hz_1p5s_384dss_simple_8p/vimh_dataset_info.json ]; then \
		echo "Dataset already exists: data/vimh-48x48x1_8000Hz_1p5s_384dss_simple_8p"; \
	else \
		python generate_vimh.py --config-name=synth/generate_moog_resonance; \
	fi
	ls ./data/

sdma synth-dataset-moog-all: sdmb sdme sdmr ## Generate all Moog VCF datasets: basic, envelope, resonance

# DUMP VIMH DATASET METADATA "vd"

vdr vd vimh-dump-recent: ## Dump the metadata of the most recent VIMH dataset in ./data/
	python vimhd.py

vds vimh-dump-small: ## Dump the metadata of the small example SawSynth dataset (256 samples)
	python vimhd.py ./data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p

vdl vimh-dump-large: ## Dump the metadata of the larger example VIMH dataset with SawSynth (16k samples)
	python vimhd.py ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p

# ANALYZE VIMH PARAMETER DISTRIBUTIONS "vp"

vpr vp vimh-params-recent: ## Analyze parameter distributions in the most recent VIMH dataset
	python vimhd.py -p

vps vimh-params-small: ## Analyze parameter distributions in the small example SawSynth dataset (256 samples)
	python vimhd.py -p ./data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p

vpl vimh-params-large: ## Analyze parameter distributions in the larger example VIMH dataset (16k samples)
	python vimhd.py -p ./data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p

# DISPLAY VIMH DATASETS "dd"

ddr display-dataset-recent: ## Display the most recently created dataset (default)
	python display_vimh.py

dds display-dataset-small: synth-dataset-small ## Display the small example VIMH dataset (256 samples)
	python display_vimh.py

ddl display-dataset-large: sdl ## Display the larger example VIMH dataset (16k samples)
	python display_vimh.py

# EXPERIMENTS "e" - Complete Configuration Examples

ex exp-example: synth-dataset-small ## Train CNN on default dataset
	time python src/train.py experiment=example  # ./configs/experiment/example.yaml

# TRIVIAL DATASET EXPERIMENTS "et" - Small models for testing on trivial synthetic data

etms exp-trivial-micro-small: sds ## Micro CNN (~2K params) on small dataset (256 samples) - ordinal classification loss
	time python src/train.py experiment=trivial_micro_small

etmsr exp-trivial-micro-small-regression: sds ## Micro CNN (~2K params) on small dataset (256 samples) - regression loss
	time python src/train.py experiment=trivial_micro_small_regression

etmsrdt exp-trivial-micro-small-regression-decay-time: sds ## Micro CNN (~2K params) on small dataset (256 samples) - regression loss on log10_decay_time only
	time python src/train.py experiment=trivial_micro_small_regression callbacks.model_checkpoint.monitor="val/log10_decay_time_mae" callbacks.early_stopping.monitor="val/log10_decay_time_mae" optimized_metric="val/log10_decay_time_mae"

etts exp-trivial-tiny-small: sds ## Tiny CNN (~8K params) on small dataset (256 samples)
	time python src/train.py experiment=trivial_tiny_small

etml exp-trivial-micro-large: sdl ## Micro CNN (~2K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_micro_large

etmlr exp-trivial-micro-large-regression: sdl ## Micro CNN (~2K params) on large dataset (16K samples) - regression loss
	time python src/train.py experiment=trivial_micro_large_regression

etmlrdt exp-trivial-micro-large-regression-decay-time: sdl ## Micro CNN (~2K params) on large dataset (16K samples) - regression loss on log10_decay_time only
	time python src/train.py experiment=trivial_micro_large_regression callbacks.model_checkpoint.monitor="val/log10_decay_time_mae" callbacks.early_stopping.monitor="val/log10_decay_time_mae" optimized_metric="val/log10_decay_time_mae"

etmldt exp-trivial-micro-large-decay-time: sdl ## Micro CNN (~2K params) on large dataset (16K samples) - ordinal loss on log10_decay_time only
	time python src/train.py experiment=trivial_micro_large callbacks.model_checkpoint.monitor="val/log10_decay_time_acc" callbacks.early_stopping.monitor="val/log10_decay_time_acc" optimized_metric="val/log10_decay_time_acc"

ettl exp-trivial-tiny-large: sdl ## Tiny CNN (~8K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_tiny_large

et64l exp-trivial-64k-large: sdl ## "64K" CNN (actually 1.4M params) on large dataset - for comparison
	time python src/train.py experiment=trivial_64k_large

etall: ex etms etts etml ettl et64l ## Run all trivial dataset experiments: ex etms etts etml ettl et64l

# TRIVIAL DATASET ViT EXPERIMENTS "evit" - Small ViT models for testing on trivial synthetic data

evitms exp-trivial-vit-micro-small: sds ## Micro ViT (~8K params) on small dataset (256 samples)
	time python src/train.py experiment=trivial_vit_micro_small

evitmsr exp-trivial-vit-micro-small-regression: sds ## Micro ViT (~8K params) on small dataset (256 samples) - regression variant (placeholder)
	time python src/train.py experiment=trivial_vit_micro_small_regression

evitts exp-trivial-vit-tiny-small: sds ## Tiny ViT (~25K params) on small dataset (256 samples)
	time python src/train.py experiment=trivial_vit_tiny_small

evitml exp-trivial-vit-micro-large: sdl ## Micro ViT (~8K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_vit_micro_large

evittl exp-trivial-vit-tiny-large: sdl ## Tiny ViT (~25K params) on large dataset (16K samples)
	time python src/train.py experiment=trivial_vit_tiny_large

evitall: evitms evitts evitml evittl ## Run all ViT trivial dataset experiments

# MOOG VCF DATASET EXPERIMENTS "em" - CNN training on Moog filter datasets

emb exp-moog-basic: sdmb ## Train CNN on basic Moog VCF dataset (4 params)
	time python src/train.py experiment=moog_cnn_basic

eme exp-moog-envelope: sdme ## Train CNN on Moog envelope sweep dataset (10 params)
	time python src/train.py experiment=moog_cnn_envelope

emer exp-moog-envelope-regression: sdme ## Train CNN on Moog envelope sweep dataset (10 params) using regression loss
	time python src/train.py experiment=moog_cnn_envelope_regression

emr exp-moog-resonance: sdmr ## Train CNN on high-resonance Moog dataset (8 params)
	time python src/train.py experiment=moog_cnn_resonance

emall: emall-gen emb eme emr ## Generate datasets and train CNNs on all Moog VCF experiments

emall-gen: sdmb sdme sdmr ## Generate all Moog datasets before training

emall-train: emb eme emr ## Run all Moog dataset training experiments

# MOOG VCF ViT EXPERIMENTS "emvit" - ViT training on Moog filter datasets (experimental)

emvitb exp-moog-vit-basic: sdmb ## Train ViT on basic Moog VCF dataset (4 params) - square 32x32
	time python src/train.py experiment=moog_vit_basic

emvite exp-moog-vit-envelope: sdme ## Train ViT on Moog envelope sweep dataset (10 params) - rectangular 32x64
	time python src/train.py experiment=moog_vit_envelope

emvitr exp-moog-vit-resonance: sdmr ## Train ViT on high-resonance Moog dataset (8 params) - square 48x48
	time python src/train.py experiment=moog_vit_resonance

emvit emvit-train-all: emvitb emvite emvitr ## Run all Moog ViT training experiments

emvitgta emvit-gen-train-all: emall-gen emvit ## Generate datasets and train ViTs on all Moog VCF experiments

# AUDIO EVAL
ae audio-eval: ## Eval latest best model checkpoint using default dataset using src/audio_reconstruction_eval.py
	python src/audio_reconstruction_eval.py

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

cl clean-logs: ## Clean logs
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

tds test-diagram-simple: ## Generate simple text-only diagrams (default cnn_64k)
	python viz/simple_model_diagram.py

tdsc test-diagram-simple-config: ## Generate simple diagram for specific config (usage: make tdsc CONFIG=cnn_64k)
	python viz/simple_model_diagram.py --config $(CONFIG)

tdsl test-diagram-simple-list: ## List available configs for simple diagrams
	python viz/simple_model_diagram.py --list-configs

tdsa test-diagram-simple-all: ## Generate simple diagrams for all architectures
	python viz/simple_model_diagram.py --config cnn_64k
	python viz/simple_model_diagram.py --config cnn_64k_ordinal
	python viz/simple_model_diagram.py --config cnn_64k_regression
	python viz/simple_model_diagram.py --config cnn_64k_auxiliary
	python viz/simple_model_diagram.py --config cnn_micro
	python viz/simple_model_diagram.py --config cnn_tiny
	python viz/simple_model_diagram.py --config vit_micro
	python viz/simple_model_diagram.py --config vit_tiny

tdv test-diagram-vgg: ## Generate VGG-style architecture diagrams (EPS + PNG)
	python viz/vgg_style_diagrams.py

# RAW TRAINING TARGETS "tr" (no "experiment" - use hydra overrides to set desired config - experiments recommended instead)

tr train: synth-dataset-small ## Train default model on default dataset (`make tr`) - defaults defined in ./configs/train.yaml
	time python src/train.py

trq train-quick: sds ## Train super quickly the default model and dataset (quick sanity test to see if things are working)
	python src/train.py trainer.max_epochs=1

trs train-vimh-small: sds ## Train the small example VIMH dataset using the default model (CNN 64k)
	time python src/train.py data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p

trl train-vimh-large: sdl ## Train the large example VIMH dataset using the default model (CNN 64k)
	time python src/train.py data.data_dir=data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p

# UTILITY TARGETS

f format: ## Run pre-commit hooks
	pre-commit run -a

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
