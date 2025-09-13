# Tutorial Sequence: From Setup to VIMH Experiments

This walkthrough focuses on the VIMH workflow used in this repository: generating datasets, training CNN/ViT models, and evaluating with audio reconstruction.

## Prerequisites

```bash
# In project root, activate your environment
source .venv/bin/activate.csh  # or your shell’s activate
```

---

## Phase 1: Validate Setup

### Example 1 — Sanity check

```bash
make test         # Fast pytest path
make clean        # Optional: clear caches
make trq          # 1-epoch quick train to verify stack
```

What you get: end‑to‑end run, logs in `logs/train/runs/`, and a checkpoint.

---

## Phase 2: Create and Inspect VIMH Data

### Example 2 — Generate a small dataset (256 samples)

```bash
make sds          # Synth small SawSynth VIMH dataset
make dds          # Visualize recent dataset
```

### Example 3 — Generate a larger dataset (16k)

```bash
make sdl
make ddl
```

---

## Phase 3: Train on VIMH

### Example 4 — Default training (uses configs/train.yaml defaults)

```bash
make tr           # or: python src/train.py
```

### Example 5 — Train on specific datasets

```bash
make trs          # Small dataset (creates if missing)
make trl          # Large dataset (creates if missing)
```

### Example 6 — Run the example experiment

```bash
python src/train.py experiment=example
```

Tips:

- Mac: add `trainer=mps` for Metal acceleration
- Adjust epochs: `trainer.max_epochs=10`
- Switch models: `model=cnn_64k` | `model=cnn_64k_ordinal` | `model=vit_tiny`

---

## Phase 4: Audio Evaluation

### Example 7 — Hear reconstructions

```bash
make ae                                # Auto-discovers latest checkpoint
python src/audio_reconstruction_eval.py # Equivalent; see configs/audio_eval.yaml
```

Use `interactive=true` for a widget, and `save_audio=true` to export WAVs.

---

## Phase 5: Moog VCF Experiments

### Example 8 — Generate datasets and train CNNs

```bash
make sdmb && make emb    # Basic (4 params)
make sdme && make eme    # Envelope (10 params)
make sdmr && make emr    # Resonance (8 params)
```

### Example 9 — Train ViTs on Moog datasets (experimental)

```bash
make emvitb | make emvite | make emvitr
```

---

## Phase 6: Visualize Architectures

### Example 10 — Generate diagrams

```bash
make tds          # Simple text diagrams (default config)
make td           # Enhanced diagrams for all architectures
make tdsa         # Simple diagrams for all configs
```

---

## Appendix: Troubleshooting

### Environment Issues

```bash
# If you see "No module named 'rootutils'"
source .venv/bin/activate.csh

# Clean slate
make clean && make clean-logs
```

### Training Issues

```bash
# For MPS/Mac users
python src/train.py trainer=mps data.num_workers=0

# Memory issues — reduce batch size
python src/train.py data.batch_size=32
```

### Debugging

```bash
# Run tests to verify everything works
make test

# Quick sanity check
make trq

# Check logs
ls logs/train/runs/
```

---

## Additional Resources

- Architecture details: `docs/architectures.md`
- VIMH format: `docs/vimh.md`
- Losses for VIMH: `docs/vimh_loss_functions.md`
- Configuration patterns: `docs/configuration.md`
