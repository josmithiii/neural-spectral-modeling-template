# VIMH Dataset Format

Variable Image MultiHead (VIMH) is the dataset format used throughout NSMT. It packages spectrogram-like tensors and quantized synthesis parameters in a self-describing binary layout so models can auto-configure heads, ranges, and loss types.

## Audio-Centric Design

- **Variable resolution**: Height = frequency bins, width = time frames, channels = alternate spectral representations, successive time frames, adjacent spectra, layer neural capacity, etc.
- **Quantized parameters**: Each sample stores up to 255 varying synthesis parameters (0–255 codes) with metadata describing min/max/step for dequantization.
- **Self-contained metadata**: `vimh_dataset_info.json` records dimensions, audio settings, parameter names, and perceptual scaling choices.

## Binary Layout per Sample

```
[height:uint16][width:uint16][channels:uint16]
[N:uint8][param_0_id:uint8][param_0_val:uint8] ... [param_N-1_id][param_N-1_val]
[pixel_data:uint8^(height*width*channels)]
```

Parameter IDs map into `parameter_mappings` inside the JSON file, which also provides the value range and quantization step. De/quantization follows:

```python
normalized = (actual - param_min) / (param_max - param_min)
quantized = int(round(normalized * 255))
actual = param_min + (quantized / 255.0) * (param_max - param_min)
```

## Dataset Structure

```
data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p/
├── train/                     # Binary training data
├── test/                      # Binary test data
├── train_batch                # Optional pickle cache
├── test_batch                 # Optional pickle cache
└── vimh_dataset_info.json     # Metadata
```

`vimh_dataset_info.json` fields you should pay attention to:

- `height`, `width`, `channels`: Input tensor shape (C,H,W after torch conversion).
- `parameter_names`: Ordered list matching the varying heads.
- `parameter_mappings`: Dict describing `min`, `max`, `step`, optional `num_classes`, and textual notes.
- `audio_settings`: STFT/mel configuration used during synthesis (present for generated datasets).

## Tooling

- Generate datasets: `make gds` (256-sample wah) and `make gdl` (16K-sample wah).
- Inspect metadata: `make vdr` (latest), `make vds` (small wah), `make vdl` (large wah).
- Analyze parameter coverage: `make vpr|vps|vpl` – emits stats, ranges, and chi-square uniformity heuristics.
- Visualize samples: `make ddr|dds|ddl` – renders spectrogram grids for quick sanity checks.

`vimhd.py` powers the `vd*` and `vp*` targets. Run it directly to point at custom paths:

```bash
python vimhd.py data/vimh-32x32x1_8000Hz_1p0s_256dss_saw_wah_2p
python vimhd.py -p data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_2p
```

## Auto-Configuration Pipeline

1. **DataModule** (`src/data/vimh_datamodule.py`) reads metadata to size transforms and loaders.
2. **`_configure_vimh_model_config`** (in `src/train.py`) injects head definitions, regression ranges, and JND-based loss weights before instantiating the model.
3. **`VIMHLitModule`** builds the Lightning module with the supplied criteria.

As long as `vimh_dataset_info.json` is present, new datasets will automatically map to the correct head counts and ranges without editing YAML.

## Tips

- Keep dataset directories under `data/` to avoid path overrides.
- When crafting new generators, populate `parameter_mappings`—the auto-config relies on it.
- Use the parameter distribution report after generation to catch skewed sampling before training.
- For regression workflows, prefer quantization steps that reflect perceptual just noticeable differences (JNDs); loss weighting normalizes by the number of JND steps.

For background on how these datasets feed the loaders, see [multihead_data_architecture.md](multihead_data_architecture.md).
