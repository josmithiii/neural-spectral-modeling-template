# Transcription Pipeline: Orientation Guide

This document explains the full processing chain triggered by `make all` in the `examples/external_project/` directory. This pipeline is designed to test the **NSMT (Neural Spectral Modeling Template)** model's ability to refine note onsets and predict synthesis parameters.

---

## 1. Data Generation (`make gen`)
**Script:** `generate_test_audio.py`

This step creates the "Ground Truth" audio file (`test_100.wav`) used for evaluation.
- **Synthesis**: It pulls 100 random samples from the VIMH dataset and synthesizes them using the project's DSP synthesizers (e.g., Sawtooth + Wah + Delay).
- **Concatentation**: The 1.0s samples are concatenated into one long audio file.
- **Truth Metadata**: It creates `test_100_truth.txt` and `test_100_truth.mid`.
- **The Delay Parameter**: Crucially, each note is synthesized with a random `onset_delay_ms` (0–200ms). The "True" onset in the audio is the start of the 1.0s window PLUS this delay.

---

## 2. Parameter Prediction (`make pred-multi`)
**Script:** `predict_params.py`

This is the core "AI" step. It attempts to transcribe the audio without knowing the truth.

### Step A: Coarse Detection (The DSP Layer)
The script uses standard energy-based detection (`RMS envelope derivative`) to find the **approximate** start of every note. This is usually accurate to within ~20ms, but often "late" on soft attacks.

### Step B: The 100Hz "Model Domain" (Pitch-Aware Resampling)
The neural network was trained on audio fixed at exactly **100 Hz**. However, real music (like your test file) might be at **440 Hz**.
- **The Problem**: If we resample a 1.0s chunk of 440Hz audio to 100Hz, it becomes **4.4 seconds long**. The model gets confused by this "temporal smear."
- **The Fix**: We use **Pitch-Aware Segmentation**. We calculate the ratio (`440 / 100 = 4.4`) and extract only **0.22s** of original audio. When resampled, this becomes exactly **1.0s of audio @ 100Hz**, matching the model's training space perfectly.

### Step C: Fine Refinement (The Pre-roll Strategy)
To give the AI the best chance of success, we use a **100ms Pre-roll**.
1. We take our "Coarse" detection.
2. We "look back" 100ms and start the model's window there.
3. In the model's eyes, the note *should* start at exactly **t=100ms**.
4. The AI predicts an `onset_delay_ms`.
   - If it predicts **100.0**, it agrees with the coarse detector.
   - If it predicts **105.0**, it says the detector was 5ms early.
   - If it predicts **95.0**, it says the detector was 5ms late.

### Step D: Temporal Scaling
Any delay refined by the AI must be scaled back. A 10ms refinement in the "slow" 100Hz domain is only a ~2.2ms refinement in the "fast" 440Hz domain. The script handles this math automatically using the `resample_factor`.

---

## 3. Visual Comparison (`make compare`)
**Script:** `compare_midi.py`

This launches a Matplotlib GUI to visually inspect the results.

### The Legend:
- **Green Dotted Line (Truth)**: The absolute truth from the synthesis metadata. This is the "Target" we want to hit.
- **Orange Dashed Line (MIDI)**: The ground-truth MIDI file. (Used to compare against other systems like Basic Pitch).
- **Blue Solid Line (NSMT)**: The final output of the AI pipeline.

### The Research Goal:
Currently, the Blue lines are often **100ms early**. This is because the model is predicting `0.0` for the delay (it's "blind" to the feature). Your task is to improve the model or the spectrogram representation so that the AI starts predicting values near `100.0`, causing the **Blue** lines to "snap" onto the **Green** truth lines.

