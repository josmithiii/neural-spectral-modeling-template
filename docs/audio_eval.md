# 🎵 Audio Reconstruction Evaluation - Quick Start

## Simplest Usage (Recommended)

After training your model, just run:

```bash
python src/audio_reconstruction_eval.py
```

This will:
- ✅ **Auto-discover** your latest checkpoint from `logs/train/`
- ✅ **Launch interactive mode** for browsing test samples
- ✅ **Show side-by-side** true vs predicted audio comparisons
- ✅ **Enable audio playback** (in Jupyter/interactive environments)

## What You'll See

The interactive widget provides:
- **Slider** to browse through test samples
- **Plots** showing waveforms, spectrograms, and parameters
- **Play buttons** to hear true vs predicted audio
- **Metrics** displaying prediction quality

## Alternative Usage

### Batch Mode (Non-Interactive)
```bash
python src/audio_reconstruction_eval.py interactive=false
```

### Specific Checkpoint
```bash
python src/audio_reconstruction_eval.py ckpt_path=path/to/your/checkpoint.ckpt
```

### Custom Dataset
```bash
python src/audio_reconstruction_eval.py data=vimh_256dss
```

### Save Audio Files
```bash
python src/audio_reconstruction_eval.py save_audio=true output_dir=my_results
```

## Configuration Options

All options can be modified in `configs/audio_eval.yaml` or passed as command line arguments:

- **`ckpt_path`**: Path to checkpoint (auto-discovers if null)
- **`interactive`**: Launch GUI widget (default: true)
- **`num_samples`**: Number of samples to evaluate (default: 5)
- **`save_audio`**: Export audio files (default: true)
- **`output_dir`**: Where to save results (default: "audio_eval_results")

## What Gets Evaluated

For each test sample, the system:

1. **Runs inference** on the spectrogram to predict synthesis parameters
2. **Synthesizes audio** using both true and predicted parameters
3. **Computes metrics** (RMSE, SNR, correlation) between audio signals
4. **Visualizes** waveforms, spectrograms, and parameter comparisons
5. **Exports audio** files for manual listening tests

## Key Metrics to Watch

- **Parameter Errors**: How accurately does your model predict synthesis parameters?
- **SNR (dB)**: Signal-to-noise ratio of reconstructed audio (higher = better)
- **Correlation**: Waveform similarity between true and predicted (closer to 1.0 = better)
- **RMSE**: Root mean square error between audio signals (lower = better)

## Tips for Analysis

🎯 **Good Results**: 
- Parameter relative errors < 10%
- SNR > 20 dB
- Correlation > 0.8
- Audio sounds perceptually similar

🔧 **If Results Are Poor**:
- Check if model needs more training
- Verify synthesis parameters match training data
- Ensure STFT/mel settings are consistent
- Try different model architectures

## Examples

See `examples/audio_reconstruction_example.py` for detailed usage examples and analysis workflows.

---

**This evaluation system lets you directly hear how well your neural spectral models are learning to map spectrograms back to synthesis parameters!**
