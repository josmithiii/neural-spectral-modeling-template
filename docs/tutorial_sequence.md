# Tutorial Sequence: From Setup to Wah Experiments

Follow this sequence to go from a clean checkout to the supported wah-pedal experiments. Commands assume the project root.

## Phase 0: Environment

```bash
sh setup.sh                 # Create uv-based virtualenv and install deps
source .venv/bin/activate    # Or activate.csh for csh/tcsh users
```

## Phase 1: Verify the Stack

```bash
make test         # Fast pytest subset
make format       # Optional: confirm formatting hooks succeed
make trq          # One-epoch smoke train (writes logs/train/runs/...)
```

You should see a Lightning summary and a checkpoint created inside `logs/train/runs/`.

## Phase 2: Build and Inspect Data

```bash
make gds          # Generate 256-sample wah dataset (wraps gdws)
make dds          # Visualize the recent dataset
make vds          # Inspect metadata and parameter ranges
```

For larger runs:

```bash
make gdl          # 16K-sample wah dataset (wraps gdwl)
make ddl          # Visualize large dataset samples
make vpl          # Verify parameter distributions (chi-square summary)
```

## Phase 3: Train Models

```bash
make tr           # Default training config (cnn_64k on VIMH defaults)
make trs          # Train on the small dataset
make trl          # Train on the large dataset
python src/train.py experiment=wah_cnn_tiny trainer=mps
```

Tips:
- Append `trainer=mps data.num_workers=0` on Apple Silicon.
- Override epochs inline: `trainer.max_epochs=25`.
- Switch models via `model=cnn_tiny` or `model=vit_tiny`.

## Phase 4: Wah Reference Experiments

```bash
make ewt          # Wah CNN tiny (classification heads)
make ewtr         # Wah CNN tiny regression
make evwt         # Evaluate latest classification checkpoint
make evwtr        # Evaluate latest regression checkpoint
```

Good runs produce per-head accuracies above 0.85 (classification) or MAE below 0.05 (regression) on the validation set. Sample logs live in `logs/train/runs/<timestamp>/`.

## Phase 5: Audio Reconstruction Evaluation

```bash
make ae                                # Interactive playback + plots
python src/audio_reconstruction_eval.py interactive=false save_audio=true
```

Inspect the generated WAV files and plots under `audio_eval_results/`.

## Phase 6: Moog Experiments (Optional)

```bash
make gdmb && make emb      # Basic Moog CNN (4 params)
make gdme && make eme      # Moog envelope CNN (10 params)
make gdmr && make emr      # High-resonance Moog CNN
```

## Phase 7: Visualize Architectures

```bash
make tds          # Simple diagrams (text)
make td           # Enhanced diagrams (graphviz if installed)
make tdsa         # Simple diagrams for all configs
```

## Troubleshooting Checklist

- Dataset generation fails → ensure `data/` writable and disk space > 2 GB.
- MPS runtime errors → add `data.num_workers=0` or fall back to CPU.
- Training diverges → confirm dataset metadata matches model heads via `make vdr`.
- Audio eval finds no checkpoint → pass `ckpt_path=` explicitly to the script.

Refer to [audio_developer_primer.md](audio_developer_primer.md) for more background on the signal-processing steps used throughout this workflow.
