# Command Reference

Snapshot of `make h` output (auto-generated).

| Target(s) | Description |
| --------- | ----------- |
| `h help` | Show help |
| `@grep -E '^[.a-zA-Z0-9_ -]+` | .*$$' $(MAKEFILE_LIST) \| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' \| less -R |
| `gds generate-dataset-small` | Synthesize a small example VIMH dataset with SawSynth + Wah (256 samples) |
| `gdl generate-dataset-large` | Synthesize a larger example VIMH dataset with SawSynth + Wah (16k samples) |
| `gdws generate-dataset-wah-small` | Synthesize VIMH dataset with Saw + Wah - decay-time and pedal-angle varied (256 samples) |
| `gdwl generate-dataset-wah-large` | Synthesize VIMH dataset with Saw + Wah - decay-time and pedal-angle varied (16k samples) |
| `gdwe generate-dataset-wah-envelope` | Synthesize VIMH dataset with Saw + Wah ADSR envelope settings varied (512 samples) |
| `gdas generate-dataset-all-small` | Generate all small wah-focused datasets |
| `vdr vd vimh-dump-recent` | Dump the metadata of the most recent VIMH dataset in ./data/ |
| `vds vimh-dump-small` | Dump the metadata of the small example SawSynth dataset (256 samples) |
| `vdl vimh-dump-large` | Dump the metadata of the larger example VIMH dataset with SawSynth (16k samples) |
| `vpr vp vimh-params-recent` | Analyze parameter distributions in the most recent VIMH dataset |
| `vps vimh-params-small` | Analyze parameter distributions in the small example SawSynth dataset (256 samples) |
| `vpl vimh-params-large` | Analyze parameter distributions in the larger example VIMH dataset (16k samples) |
| `ddr display-dataset-recent` | Display the most recently created dataset (default) |
| `dds display-dataset-small` | Display the small example VIMH dataset (256 samples) |
| `ddl display-dataset-large` | Display the larger example VIMH dataset (16k samples) |
| `ex exp-example` | Train CNN on default dataset |
| `etms exp-trivial-micro-small` | Micro CNN (~10K params) on small dataset (256 samples) - ordinal classification loss |
| `etmsr exp-trivial-micro-small-regression` | Micro CNN (~10K params) on small dataset (256 samples) - regression loss |
| `etts exp-trivial-tiny-small` | Tiny CNN (~40K params) on small dataset (256 samples) |
| `evitms exp-trivial-vit-micro-small` | Micro ViT (~23K params) on small dataset (256 samples) |
| `ewt exp-wah-tiny` | Train "tiny" (~40K) CNN on dataset gdwl (large sawtooth + wah + decay envelope) [~9 min to train on Mac MPS] |
| `ewtq exp-wah-tiny-quick` | Quick version ewt (exp-wah-tiny) to produce a checkpoint fast for testing (1 epoch, small dataset) |
| `ewtr exp-wah-tiny-regression` | Train "tiny" (1.1M) CNN-pure-regression on dataset gdwl (large sawtooth + wah + decay envelope) [~4.6 min to train on Mac MPS] |
| `ewtrq exp-wah-tiny-regression-quick` | Quick regression run (1 epoch, small dataset) to produce a checkpoint fast |
| `evwt eval-wah-tiny` | Evaluate latest wah_cnn_tiny best checkpoint (set CKPT=path/to.ckpt to override) |
| `evwtr eval-wah-tiny-regression` | Evaluate latest wah_cnn_tiny_regression best checkpoint (set CKPT=path/to.ckpt to override) |
| `ae ae_latest audio-eval-latest` | Display eval of latest best model checkpoint using default dataset using src/audio_reconstruction_eval.py |
| `aer ae_reg audio-eval-regression` | Audio-eval latest run with "regression" tag (auto-picks best/last ckpt) |
| `aep ae_prev audio-eval-previous` | Audio-eval the second most recent run (any experiment) |
| `aef ae_filter audio-eval-filter` | Audio-eval latest run whose tags.log contains FILTER=... (optional EXCLUDE=...) |
| `ae_cls audio-eval-classification` | Audio-eval latest classification run (exclude 'regression') |
| `c clean` | Clean all generated files except logs and datasets |
| `cd clean-data` | Clean data files |
| `cl clean-logs` | Clean run logs |
| `dc distclean` | Clean back to original distribution (except for hidden files) |
| `t0 tests0` | Tests of basic CNN and ViT archs on classes and regression |
| `t test` | Run fast pytest tests |
| `ta test-all` | Run all pytest tests |
| `td tda text-diagram-all` | Generate enhanced diagrams for all architectures (text + graphical) |
| `tdl text-diagram-list` | List available model configs for diagrams |
| `tds text-diagram-simple` | Generate simple text-only diagrams (default cnn_medium) |
| `tdsc text-diagram-simple-config` | Generate simple diagram for specific config (usage: make tdsc CONFIG=cnn_medium) |
| `tdsl text-diagram-simple-list` | List available configs for simple diagrams |
| `tdsa text-diagram-simple-all` | Generate simple diagrams for all architectures |
| `tdv text-diagram-vgg` | Generate VGG-style architecture diagrams (EPS + PNG) |
| `tr train` | Train default model on default dataset (`make tr`) - defaults defined in ./configs/train.yaml |
| `trq train-quick` | Train super quickly the default model and dataset (quick sanity test to see if things are working) |
| `trs train-vimh-small` | Train the small example VIMH dataset using the default model (CNN Medium) |
| `trl train-vimh-large` | Train the large example VIMH dataset using the default model (CNN Medium) |
| `f format` | Run pre-commit hooks |
| `fp format-preview` | Preview docformatter actions |
| `fdc format-docstrings-check` | Run docformatter pre-commit hook (manual stage) |
| `fm format-markdown` | Run Prettier on Markdown/YAML only |
| `fn flake8-now` | Run flake8 lint manually on src/tests and key scripts |
| `fc format-configs` | Prettier-format only YAML in configs/ |
| `sy sync` | Merge changes from main branch to your current branch |
| `tb tensorboard` | Launch TensorBoard on port 6006 |
| `a activate` | Activate the uv environment |
| `d deactivate` | Deactivate the uv environment |
| `lc list-configs` | List available model configurations |
