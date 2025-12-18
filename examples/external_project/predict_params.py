#!/usr/bin/env python3
"""
Example: Predict synthesis parameters from a .wav file using a trained NSMT model.

This demonstrates using nsm-synth-match as an external package.

Usage:
    python predict_params.py input.wav [checkpoint.ckpt]
    python predict_params.py --multi input.wav [checkpoint.ckpt]
    python predict_params.py --multi --midi-out output.mid input.wav [checkpoint.ckpt]

If no checkpoint is specified, uses the reference saw+wah+delay model.
The --multi flag enables multi-note processing with onset detection.
The --midi-out flag specifies a MIDI output file for the detected notes.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np

# Add project root to path
import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@dataclass
class DetectedNote:
    """Represents a detected note with all its properties."""
    onset_time: float  # Time in seconds
    detected_f0: float  # Detected fundamental frequency in Hz
    midi_note: int  # MIDI note number (0-127)
    velocity: int  # MIDI velocity (0-127)
    duration: float  # Note duration in seconds
    params: Dict[str, float]  # Predicted synthesis parameters

# Default checkpoint name (will be resolved to latest timestamped version)
DEFAULT_CKPT = "wah_del_cnn_medium.ckpt"


def resolve_checkpoint(name: str) -> Optional[str]:
    """Resolve a checkpoint name to its full path.

    Searches checkpoints/reference for timestamped versions like:
    2025-01-01_12-00-00_wah_del_cnn_medium.ckpt

    Returns the latest matching checkpoint path, or None if not found.
    """
    # If it's already an absolute path that exists, use it
    if Path(name).is_absolute() and Path(name).exists():
        return name

    # If it exists relative to cwd, use it
    if Path(name).exists():
        return str(Path(name).resolve())

    # Search in checkpoints/reference for timestamped versions
    ref_dir = PROJECT_ROOT / "checkpoints/reference"
    if ref_dir.exists():
        # Match files ending with the provided name (handles timestamp prefix)
        matches = sorted(ref_dir.glob(f"*{name}"))
        if matches:
            return str(matches[-1])  # Latest by sort order

    return None


def load_wav(wav_path: str, target_sr: int = 8000) -> np.ndarray:
    """Load a wav file and resample to target sample rate."""
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("Please install soundfile: pip install soundfile")

    audio, sr = sf.read(wav_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import resampy
            audio = resampy.resample(audio, sr, target_sr)
        except ImportError:
            print(f"Warning: Sample rate is {sr}Hz, expected {target_sr}Hz.")
            print("Install resampy for automatic resampling: pip install resampy")

    return audio.astype(np.float32)


def load_model_and_config(ckpt_path: str) -> tuple:
    """Load checkpoint and reconstruct model with proper configuration.

    Returns the raw network (not LightningModule) for inference simplicity.
    """
    import json
    from src.models.components.simple_cnn import SimpleCNN

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", {})

    # Load dataset info from data_dir (contains spectrogram config)
    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    data_dir = datamodule_hparams.get("data_dir", "")
    dataset_info = {}
    if data_dir:
        info_path = Path(data_dir) / "vimh_dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                dataset_info = json.load(f)
            print(f"  Loaded dataset info from: {info_path}")

    # Merge dataset_info into hparams for spectrogram processing
    hparams["spectrogram_config"] = dataset_info.get("spectrogram_config", {})
    hparams["mel_config"] = dataset_info.get("mel_config", {})
    hparams["image_shape"] = [
        dataset_info.get("channels", 1),
        dataset_info.get("height", 32),
        dataset_info.get("width", 64),
    ]
    hparams["heads"] = []
    for name in dataset_info.get("parameter_names", []):
        mapping = dataset_info.get("parameter_mappings", {}).get(name, {})
        hparams["heads"].append({
            "name": name,
            "min": mapping.get("min", 0.0),
            "max": mapping.get("max", 1.0),
        })

    # Extract heads config from checkpoint weights
    heads_config = []
    head_keys = sorted(set(
        k.split(".")[2] for k in state_dict.keys()
        if k.startswith("net.heads.") and ".0.weight" in k
    ))

    for head_name in head_keys:
        weight_key = f"net.heads.{head_name}.0.weight"
        if weight_key in state_dict:
            num_classes = state_dict[weight_key].shape[0]
            # Get param bounds from hparams if available
            heads_from_hparams = hparams.get("heads", [])
            head_info = next((h for h in heads_from_hparams if h.get("name") == head_name), {})
            heads_config.append({
                "name": head_name,
                "num_classes": num_classes,
                "min": head_info.get("min", 0.0),
                "max": head_info.get("max", 1.0),
            })

    # Get image shape from hparams
    image_shape = hparams.get("image_shape", [1, 32, 64])
    in_channels = image_shape[0] if len(image_shape) == 3 else 1

    # Infer CNN architecture from weights
    conv1_weight = state_dict.get("net.conv_layers.0.weight")
    conv2_weight = state_dict.get("net.conv_layers.4.weight")
    fc_weight = state_dict.get("net.shared_features.1.weight")

    conv1_out = conv1_weight.shape[0] if conv1_weight is not None else 64
    conv2_out = conv2_weight.shape[0] if conv2_weight is not None else 128
    fc_hidden = fc_weight.shape[0] if fc_weight is not None else 512

    # Infer input_size from FC layer input dimension
    # FC input = conv2_channels * adaptive_pool_h * adaptive_pool_w
    if fc_weight is not None:
        fc_input_size = fc_weight.shape[1]
        # Assuming adaptive_pool = (4, 4), solve for input_size
        # fc_input_size = conv2_out * 4 * 4 = conv2_out * 16
        adaptive_pool_area = fc_input_size // conv2_out
        # For (4,4) pool, input_size should be 32 (32//4=8, then adaptive to 4x4)
        if adaptive_pool_area == 16:  # 4x4 pool
            input_size = 32
        elif adaptive_pool_area == 49:  # 7x7 pool
            input_size = 28
        else:
            input_size = 32  # Default
    else:
        input_size = 32

    # Build heads dict for CNN
    heads_dict = {h["name"]: h["num_classes"] for h in heads_config}

    # Determine output mode from checkpoint
    output_mode = hparams.get("output_mode", "regression")

    # Get parameter names for regression mode
    parameter_names = [h["name"] for h in heads_config]

    # Create network (just the CNN, not the full LightningModule)
    net = SimpleCNN(
        input_channels=in_channels,
        conv1_channels=conv1_out,
        conv2_channels=conv2_out,
        fc_hidden=fc_hidden,
        heads_config=heads_dict,
        dropout=0.5,
        input_size=input_size,
        output_mode=output_mode,
        parameter_names=parameter_names,
    )

    # Extract just the network weights (remove "net." prefix from state_dict keys)
    net_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            net_state_dict[k[4:]] = v  # Remove "net." prefix

    # Load weights
    net.load_state_dict(net_state_dict, strict=False)
    net.eval()

    return net, hparams, heads_config, dataset_info


def audio_to_spectrogram(audio: np.ndarray, hparams: Dict[str, Any], dataset_info: Dict[str, Any]) -> torch.Tensor:
    """Convert audio to spectrogram tensor suitable for model input."""
    from src.utils.synth_utils import SpectrogramProcessor

    # Get spectrogram config from hparams
    spec_config = hparams.get("spectrogram_config", hparams.get("spectrogram", {}))
    mel_config = hparams.get("mel_config", {})
    image_shape = hparams.get("image_shape", [1, 32, 64])

    # Extract dimensions
    height = image_shape[1] if len(image_shape) >= 2 else 32
    width = image_shape[2] if len(image_shape) >= 3 else 64
    sample_rate = spec_config.get("sample_rate", 8000)

    # Build stft_config dict
    stft_config = {
        "type": spec_config.get("type", "mel"),
        "n_fft": spec_config.get("n_fft", 512),
        "n_window": spec_config.get("n_window", 128),
        "hop_length": spec_config.get("hop_length", 128),
        "window_type": spec_config.get("window_type", "hann"),
    }

    # Build mel_config dict
    if not mel_config:
        mel_config = {
            "n_mels": height,
            "f_min": spec_config.get("f_min", 20.0),
            "f_max": spec_config.get("f_max", sample_rate / 2),
        }

    processor = SpectrogramProcessor(
        sample_rate=sample_rate,
        height=height,
        width=width,
        stft_config=stft_config,
        mel_config=mel_config,
    )

    # Build params dict to pass to audio_to_spectrogram
    # This prevents F0 estimation when note_number or omega is fixed/known
    # During training, the true params were passed, so we must match that behavior
    spec_params = {}

    # Check for fixed parameters that indicate F0
    fixed_params = dataset_info.get("fixed_parameters", {})
    if "note_number" in fixed_params:
        spec_params["note_number"] = fixed_params["note_number"].get("value", 69.0)
    elif "omega" in fixed_params:
        spec_params["omega"] = fixed_params["omega"].get("value", 2.5)
    else:
        # If omega is a trained parameter, we need to pass a dummy value
        # to prevent F0 estimation (training used true omega values)
        trained_params = dataset_info.get("parameter_names", [])
        if "omega" in trained_params:
            # Pass dummy omega to use fixed STFT params (matches training)
            spec_params["omega"] = 3.0  # ~1000 Hz, dummy value
        elif "note_number" not in trained_params:
            # Default: pass note_number to prevent F0 estimation
            spec_params["note_number"] = 69.0

    spec_np, _, _ = processor.audio_to_spectrogram(spec_params, audio)

    # Convert to float tensor and normalize to [0, 1]
    spec_tensor = torch.from_numpy(spec_np.astype(np.float32)) / 255.0

    # Add batch dimension: [1, C, H, W]
    if spec_tensor.dim() == 2:
        spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)
    elif spec_tensor.dim() == 3:
        spec_tensor = spec_tensor.unsqueeze(0)

    return spec_tensor


def detect_onsets(audio: np.ndarray, sample_rate: int,
                  hop_length: int = 512,
                  threshold: float = 0.1) -> np.ndarray:
    """
    Detect note onsets using energy-based onset detection.

    Uses a simple but effective approach:
    1. Compute short-time energy (RMS envelope)
    2. Compute spectral flux (rate of spectral change)
    3. Find peaks in the combined onset function

    Returns array of onset times in seconds.
    """
    # Compute RMS envelope
    frame_length = hop_length * 2
    num_frames = (len(audio) - frame_length) // hop_length + 1

    if num_frames < 2:
        return np.array([0.0])  # Just return start if audio too short

    rms = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)

    # Compute onset strength as positive derivative of log energy
    log_rms = np.log(rms + 1e-10)
    onset_strength = np.diff(log_rms)
    onset_strength = np.maximum(onset_strength, 0)  # Only positive changes

    # Normalize
    if onset_strength.max() > 0:
        onset_strength = onset_strength / onset_strength.max()

    # Find peaks above threshold
    onset_frames = []

    # Always include first frame if there's energy
    if rms[0] > threshold * rms.max():
        onset_frames.append(0)

    # Find local maxima in onset strength
    min_frames_between = int(0.1 * sample_rate / hop_length)  # 100ms minimum

    for i in range(1, len(onset_strength) - 1):
        if onset_strength[i] > threshold:
            # Check if local maximum
            if onset_strength[i] >= onset_strength[i-1] and onset_strength[i] >= onset_strength[i+1]:
                # Check minimum distance from last onset
                if not onset_frames or (i - onset_frames[-1]) >= min_frames_between:
                    onset_frames.append(i + 1)  # +1 because diff shifts by 1

    # Convert frames to times
    onset_times = np.array(onset_frames) * hop_length / sample_rate

    return onset_times


def extract_note_segments(audio: np.ndarray, onset_times: np.ndarray,
                          sample_rate: int, segment_duration: float = 1.0) -> List[Tuple[float, np.ndarray]]:
    """
    Extract audio segments starting at each onset time.

    Returns list of (onset_time, audio_segment) tuples.
    """
    segment_samples = int(segment_duration * sample_rate)
    segments = []

    for onset_time in onset_times:
        start_sample = int(onset_time * sample_rate)
        end_sample = start_sample + segment_samples

        # Pad with zeros if segment extends beyond audio
        if end_sample <= len(audio):
            segment = audio[start_sample:end_sample]
        else:
            segment = np.zeros(segment_samples, dtype=audio.dtype)
            available = len(audio) - start_sample
            if available > 0:
                segment[:available] = audio[start_sample:]

        segments.append((onset_time, segment))

    return segments


def estimate_f0(audio: np.ndarray, sample_rate: int,
                f_min: float = 50.0, f_max: float = 500.0) -> float:
    """
    Estimate fundamental frequency using autocorrelation.

    Returns estimated F0 in Hz, or 0.0 if no clear pitch detected.
    """
    # Use first 100ms of audio for pitch estimation (after brief onset)
    onset_skip = int(0.01 * sample_rate)  # Skip first 10ms (attack transient)
    analysis_len = int(0.1 * sample_rate)  # Analyze 100ms

    if len(audio) < onset_skip + analysis_len:
        analysis_len = len(audio) - onset_skip
    if analysis_len < int(0.02 * sample_rate):
        return 0.0  # Not enough audio

    segment = audio[onset_skip:onset_skip + analysis_len]

    # Apply window to reduce edge effects
    window = np.hanning(len(segment))
    segment = segment * window

    # Compute autocorrelation via FFT (faster than direct computation)
    n = len(segment)
    fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft = np.fft.rfft(segment, fft_size)
    autocorr = np.fft.irfft(fft * np.conj(fft))[:n]

    # Normalize
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Find lag range corresponding to f_min and f_max
    lag_min = int(sample_rate / f_max)
    lag_max = int(sample_rate / f_min)
    lag_max = min(lag_max, n - 1)

    if lag_min >= lag_max:
        return 0.0

    # Find the highest peak in the valid lag range
    search_region = autocorr[lag_min:lag_max + 1]

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(search_region) - 1):
        if search_region[i] > search_region[i-1] and search_region[i] > search_region[i+1]:
            if search_region[i] > 0.2:  # Minimum correlation threshold
                peaks.append((search_region[i], i + lag_min))

    if not peaks:
        return 0.0

    # Take the highest peak
    best_corr, best_lag = max(peaks, key=lambda x: x[0])

    # Parabolic interpolation for sub-sample accuracy
    if best_lag > 0 and best_lag < n - 1:
        y0 = autocorr[best_lag - 1]
        y1 = autocorr[best_lag]
        y2 = autocorr[best_lag + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-10:
            delta = 0.5 * (y0 - y2) / denom
            best_lag = best_lag + delta

    f0 = sample_rate / best_lag
    return f0


def resample_to_target_f0(audio: np.ndarray, detected_f0: float,
                          target_f0: float, sample_rate: int) -> np.ndarray:
    """
    Resample audio so that detected_f0 becomes target_f0.

    This shifts the pitch by resampling, which also changes duration.
    The output is truncated/padded to match original length.
    """
    if detected_f0 <= 0 or abs(detected_f0 - target_f0) < 1.0:
        return audio  # No resampling needed

    # Resampling factor: if detected is 200Hz and target is 100Hz,
    # we need to slow down by 2x (resample_factor = 2.0)
    resample_factor = detected_f0 / target_f0

    try:
        import resampy
        # Resample to stretch/compress
        new_sr = int(sample_rate / resample_factor)
        if new_sr < 1000 or new_sr > 96000:
            return audio  # Unreasonable resampling, skip

        resampled = resampy.resample(audio, sample_rate, new_sr)
        # Resample back to original sample rate
        resampled = resampy.resample(resampled, new_sr, sample_rate)

        # Match original length
        original_len = len(audio)
        if len(resampled) >= original_len:
            return resampled[:original_len]
        else:
            # Pad with zeros
            result = np.zeros(original_len, dtype=audio.dtype)
            result[:len(resampled)] = resampled
            return result

    except ImportError:
        print("Warning: resampy not installed, skipping pitch normalization")
        return audio


def f0_to_midi_note(f0: float) -> int:
    """Convert frequency in Hz to MIDI note number.

    MIDI note 69 = A4 = 440 Hz
    Formula: note = 69 + 12 * log2(f0 / 440)
    """
    if f0 <= 0:
        return 0
    midi_note = 69 + 12 * np.log2(f0 / 440.0)
    return int(round(midi_note))


def estimate_note_velocity(audio: np.ndarray, sample_rate: int) -> int:
    """Estimate MIDI velocity from audio RMS level.

    Returns velocity in range 1-127.
    """
    # Use first 50ms after onset
    analysis_samples = int(0.05 * sample_rate)
    if len(audio) < analysis_samples:
        analysis_samples = len(audio)

    rms = np.sqrt(np.mean(audio[:analysis_samples] ** 2) + 1e-10)

    # Map RMS to velocity (assuming normalized audio in [-1, 1])
    # Typical RMS for loud notes ~0.3, quiet notes ~0.03
    # log scale: -30 dB to 0 dB -> velocity 1-127
    db = 20 * np.log10(rms + 1e-10)
    db = np.clip(db, -40, 0)  # Limit range
    velocity = int(1 + (db + 40) / 40 * 126)  # Map to 1-127
    return np.clip(velocity, 1, 127)


def write_midi_file(notes: List[DetectedNote], output_path: str, tempo: int = 120) -> None:
    """Write detected notes to a MIDI file.

    Args:
        notes: List of DetectedNote objects
        output_path: Path to output MIDI file
        tempo: Tempo in BPM (default 120)
    """
    try:
        import mido
    except ImportError:
        sys.exit("Please install mido: pip install mido")

    # Create MIDI file with one track
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo (microseconds per beat)
    tempo_us = int(60_000_000 / tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))

    # Track name
    track.append(mido.MetaMessage('track_name', name='NSMT Detected Notes', time=0))

    # Convert seconds to MIDI ticks
    # Default: 480 ticks per beat
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = ticks_per_beat * tempo / 60

    # Sort notes by onset time
    sorted_notes = sorted(notes, key=lambda n: n.onset_time)

    # Build list of MIDI events (time, is_note_on, note, velocity)
    events = []
    for note in sorted_notes:
        if note.midi_note <= 0 or note.midi_note > 127:
            continue  # Skip invalid notes

        on_time = note.onset_time
        off_time = note.onset_time + note.duration

        events.append((on_time, True, note.midi_note, note.velocity))
        events.append((off_time, False, note.midi_note, 0))

    # Sort by time, with note-offs before note-ons at same time
    events.sort(key=lambda e: (e[0], e[1]))

    # Convert to delta times and add to track
    current_time = 0
    for event_time, is_note_on, note_num, velocity in events:
        delta_seconds = event_time - current_time
        delta_ticks = int(delta_seconds * ticks_per_second)

        if is_note_on:
            track.append(mido.Message('note_on', note=note_num, velocity=velocity, time=delta_ticks))
        else:
            track.append(mido.Message('note_off', note=note_num, velocity=0, time=delta_ticks))

        current_time = event_time

    # End of track
    track.append(mido.MetaMessage('end_of_track', time=0))

    # Save
    mid.save(output_path)
    print(f"\nWrote MIDI file: {output_path}")
    print(f"  {len(sorted_notes)} notes, tempo={tempo} BPM")


def predict_single(model, spectrogram: torch.Tensor, heads_config: List[Dict]) -> Dict[str, float]:
    """Run inference on a single spectrogram and return parameter dict."""
    with torch.no_grad():
        outputs = model(spectrogram)

    params = {}
    for head in heads_config:
        name = head["name"]
        param_min = head["min"]
        param_max = head["max"]
        num_classes = head["num_classes"]

        if isinstance(outputs, dict):
            pred = outputs.get(name)
        else:
            pred = outputs

        if pred is None:
            continue

        if num_classes == 1:
            # Regression output
            pred_normalized = pred.item()
            pred_value = param_min + pred_normalized * (param_max - param_min)
        else:
            # Classification output
            pred_idx = pred.argmax(dim=-1).item()
            pred_value = param_min + (pred_idx / (num_classes - 1)) * (param_max - param_min)

        params[name] = pred_value

    return params


def process_multi_note(wav_path: str, ckpt_path: str, target_f0: float = 100.0,
                       midi_out: Optional[str] = None) -> List[DetectedNote]:
    """Process a multi-note audio file and print parameters for each note.

    Each note segment is pitch-shifted to target_f0 before inference,
    since the model was trained with F0 fixed at 100 Hz.

    Returns list of DetectedNote objects. If midi_out is specified,
    also writes a MIDI file.
    """
    print(f"Loading model from: {ckpt_path}")
    model, hparams, heads_config, dataset_info = load_model_and_config(ckpt_path)

    # Get sample rate and segment duration from config
    spec_config = hparams.get("spectrogram_config", hparams.get("spectrogram", {}))
    sample_rate = spec_config.get("sample_rate", 8000)

    # Segment duration should match training data (default 1 second)
    segment_duration = dataset_info.get("duration", 1.0)

    # Load audio
    print(f"Loading audio: {wav_path}")
    audio = load_wav(wav_path, target_sr=sample_rate)
    total_duration = len(audio) / sample_rate
    print(f"  Duration: {total_duration:.2f}s, {len(audio)} samples")
    print(f"  Target F0: {target_f0:.1f} Hz (model training pitch)")

    # Detect onsets
    print("\nDetecting note onsets...")
    onset_times = detect_onsets(audio, sample_rate)
    print(f"  Found {len(onset_times)} onsets")

    # Extract segments
    segments = extract_note_segments(audio, onset_times, sample_rate, segment_duration)

    # Process each segment
    print(f"\nProcessing {len(segments)} notes with pitch normalization...")
    print("=" * 90)

    # Print header
    param_names = [h["name"] for h in heads_config]
    header = f"{'Note':>4} {'Time':>8} {'F0':>7} {'MIDI':>5} {'Vel':>4}"
    for name in param_names:
        header += f" {name:>16}"
    print(header)
    print("-" * 90)

    detected_notes: List[DetectedNote] = []

    for i, (onset_time, segment) in enumerate(segments):
        # Estimate pitch of this segment
        detected_f0 = estimate_f0(segment, sample_rate)

        # Resample to normalize pitch to target F0
        if detected_f0 > 0:
            normalized_segment = resample_to_target_f0(segment, detected_f0, target_f0, sample_rate)
        else:
            normalized_segment = segment
            detected_f0 = 0.0  # Mark as unknown

        # Convert to spectrogram
        spectrogram = audio_to_spectrogram(normalized_segment, hparams, dataset_info)

        # Predict parameters
        params = predict_single(model, spectrogram, heads_config)

        # Convert F0 to MIDI note
        midi_note = f0_to_midi_note(detected_f0) if detected_f0 > 0 else 0

        # Estimate velocity from original (non-normalized) segment
        velocity = estimate_note_velocity(segment, sample_rate)

        # Estimate duration (time until next note or default 0.5s)
        if i < len(segments) - 1:
            next_onset = segments[i + 1][0]
            duration = min(next_onset - onset_time, 2.0)  # Cap at 2 seconds
        else:
            duration = 0.5  # Default for last note

        # Adjust onset time if onset_delay_ms is predicted
        adjusted_onset = onset_time
        if "onset_delay_ms" in params:
            adjusted_onset = onset_time + params["onset_delay_ms"] / 1000.0

        # Create DetectedNote
        note = DetectedNote(
            onset_time=adjusted_onset,
            detected_f0=detected_f0,
            midi_note=midi_note,
            velocity=velocity,
            duration=duration,
            params=params,
        )
        detected_notes.append(note)

        # Print row
        f0_str = f"{detected_f0:>6.1f}" if detected_f0 > 0 else "   N/A"
        midi_str = f"{midi_note:>4}" if midi_note > 0 else "   -"
        row = f"{i+1:>4} {onset_time:>7.3f}s {f0_str} {midi_str} {velocity:>4}"
        for name in param_names:
            value = params.get(name, 0.0)
            row += f" {value:>16.4f}"
        print(row)

    print("-" * 90)
    print(f"\nProcessed {len(segments)} notes from {total_duration:.2f}s audio")

    # Write MIDI if requested
    if midi_out:
        write_midi_file(detected_notes, midi_out)

    return detected_notes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    multi_mode = False
    midi_out = None

    # Parse flags
    while args and args[0].startswith("--"):
        if args[0] == "--multi":
            multi_mode = True
            args = args[1:]
        elif args[0] == "--midi-out":
            if len(args) < 2:
                sys.exit("--midi-out requires a filename argument")
            midi_out = args[1]
            args = args[2:]
        else:
            sys.exit(f"Unknown option: {args[0]}")

    if not args:
        print(__doc__)
        sys.exit(1)

    wav_path = args[0]
    ckpt_name = args[1] if len(args) > 1 else DEFAULT_CKPT

    # Resolve checkpoint (handles timestamped versions in checkpoints/reference)
    ckpt_path = resolve_checkpoint(ckpt_name)
    if not ckpt_path:
        sys.exit(f"Checkpoint not found: {ckpt_name}\nPlease specify a valid checkpoint path.")

    # Handle multi-note mode
    if multi_mode:
        process_multi_note(wav_path, ckpt_path, midi_out=midi_out)
        return

    # Single-note mode doesn't support MIDI output
    if midi_out:
        print("Warning: --midi-out only works with --multi mode, ignoring")

    print(f"Loading model from: {ckpt_path}")

    # Load model and config
    model, hparams, heads_config, dataset_info = load_model_and_config(ckpt_path)

    # Get sample rate
    spec_config = hparams.get("spectrogram_config", hparams.get("spectrogram", {}))
    sample_rate = spec_config.get("sample_rate", 8000)

    # Load and process audio
    print(f"Loading audio: {wav_path}")
    audio = load_wav(wav_path, target_sr=sample_rate)
    print(f"  Duration: {len(audio)/sample_rate:.2f}s, {len(audio)} samples")

    # Convert to spectrogram
    spectrogram = audio_to_spectrogram(audio, hparams, dataset_info)
    print(f"  Spectrogram shape: {spectrogram.shape}")

    # Run inference
    print("\nPredicting parameters...")
    with torch.no_grad():
        outputs = model(spectrogram)

    # Extract and display predictions
    print("\nPredicted parameters:")
    print("-" * 40)

    for head in heads_config:
        name = head["name"]
        param_min = head["min"]
        param_max = head["max"]
        num_classes = head["num_classes"]

        # Get prediction for this head
        if isinstance(outputs, dict):
            pred = outputs.get(name)
        else:
            pred = outputs

        if pred is None:
            print(f"  {name}: (not found in output)")
            continue

        # Convert to parameter value
        if num_classes == 1:
            # Regression output - denormalize from [0,1] to [min,max]
            pred_normalized = pred.item()
            pred_value = param_min + pred_normalized * (param_max - param_min)
        else:
            # Classification output - get argmax and scale
            pred_idx = pred.argmax(dim=-1).item()
            pred_value = param_min + (pred_idx / (num_classes - 1)) * (param_max - param_min)

        print(f"  {name}: {pred_value:.4f}")

    print("-" * 40)
    print("\nDone!")


if __name__ == "__main__":
    main()
