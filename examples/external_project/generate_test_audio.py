#!/usr/bin/env python3
"""
Generate test audio from a VIMH dataset by synthesizing from ground truth parameters.

This utility creates a concatenated audio file from the test set, useful for:
1. Testing predict_params.py onset detection
2. Comparing NSMT vs Basic Pitch detections with compare_midi.py
3. Validating synthesis parameter round-trips

If no checkpoint is specified, uses the reference saw+wah+delay model.

Usage:
    # Default: uses reference checkpoint, generates from all test samples
    python generate_test_audio.py

    # Generate 100 samples to a specific output file
    python generate_test_audio.py -n 100 -o test_100.wav

    # Use a specific checkpoint
    python generate_test_audio.py path/to/checkpoint.ckpt

    # From dataset directory directly
    python generate_test_audio.py --data-dir data/vimh-32x32x1_8000Hz_1p0s_16384dss_saw_wah_del_3p

Workflow example:
    # 1. Generate test audio with ground truth
    python generate_test_audio.py -n 100 -o test_100.wav

    # 2. Run NSMT prediction on it
    python predict_params.py --multi --midi-out test_100_nsmt.mid test_100.wav

    # 3. Compare NSMT vs ground truth
    python compare_midi.py test_100.wav test_100_nsmt.mid test_100_truth.mid

Outputs:
    - test_sequence.wav: Concatenated audio from test set samples
    - test_sequence_truth.txt: Ground truth onset times and parameters
    - test_sequence_truth.mid: MIDI file with ground truth onsets (if pitch info available)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.vimh_dataset import VIMHDataset
from src.utils.synth_utils import SimpleSawSynth

# Default checkpoint (absolute path from project root) - same as predict_params.py
DEFAULT_CKPT = str(PROJECT_ROOT / "checkpoints/reference/2025-12-18_03-41-00.ckpt")

@dataclass
class SynthesizedNote:
    """A synthesized note with its ground truth parameters."""
    onset_time: float       # Onset time in the concatenated audio (seconds)
    duration: float         # Note duration (seconds)
    params: Dict[str, float]  # All synthesis parameters
    midi_note: Optional[int] = None  # MIDI note if pitch info available
    velocity: int = 64      # Default MIDI velocity


def load_dataset_from_checkpoint(ckpt_path: str) -> Tuple[VIMHDataset, Dict[str, Any]]:
    """Load test dataset from checkpoint metadata.

    Returns:
        Tuple of (test_dataset, dataset_info)
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Get data directory from checkpoint
    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    data_dir = datamodule_hparams.get("data_dir")

    if not data_dir:
        print("Error: Checkpoint missing data_dir in datamodule_hyper_parameters")
        sys.exit(1)

    if not Path(data_dir).exists():
        print(f"Error: Dataset directory not found: {data_dir}")
        sys.exit(1)

    return load_dataset_from_dir(data_dir)


def load_dataset_from_dir(data_dir: str) -> Tuple[VIMHDataset, Dict[str, Any]]:
    """Load test dataset from directory.

    Returns:
        Tuple of (test_dataset, dataset_info)
    """
    data_path = Path(data_dir)

    # Load dataset info
    info_path = data_path / "vimh_dataset_info.json"
    if not info_path.exists():
        print(f"Error: Dataset info not found: {info_path}")
        sys.exit(1)

    with open(info_path) as f:
        dataset_info = json.load(f)

    # Create test dataset
    test_dataset = VIMHDataset(data_path=str(data_path), train=False)

    return test_dataset, dataset_info


def get_synth_for_dataset(dataset_info: Dict[str, Any], sample_rate: int) -> Any:
    """Create appropriate synthesizer based on dataset info."""
    synth_type = dataset_info.get("synth_type", "saw")

    if synth_type in ("percussion", "perc"):
        return PercussionSynth(sample_rate=sample_rate)
    else:
        return SimpleSawSynth(sample_rate=sample_rate)


def extract_params_from_sample(
    dataset: VIMHDataset,
    idx: int,
    param_names: List[str]
) -> Dict[str, float]:
    """Extract true parameter values from a dataset sample."""
    metadata = dataset._get_sample_metadata(idx)

    params = {}
    for param_name in param_names:
        info_key = f"{param_name}_info"
        if info_key in metadata:
            params[param_name] = metadata[info_key]["actual_value"]
        else:
            # Try to get from labels directly (for backwards compatibility)
            labels = metadata.get("labels", {})
            if param_name in labels:
                # Get mapping to denormalize
                mappings = dataset.metadata_format.get("parameter_mappings", {})
                if param_name in mappings:
                    mapping = mappings[param_name]
                    norm_val = float(labels[param_name])
                    if norm_val > 1.0:  # Quantized 0-255
                        norm_val = norm_val / 255.0
                    params[param_name] = mapping["min"] + norm_val * (mapping["max"] - mapping["min"])
                else:
                    params[param_name] = float(labels[param_name])

    return params


def estimate_midi_note(params: Dict[str, float]) -> Optional[int]:
    """Estimate MIDI note from parameters if pitch info available."""
    # Check for note_number parameter (percussion synth)
    if "note_number" in params:
        return int(round(params["note_number"]))

    # Check for base_freq or similar
    if "base_freq" in params:
        freq = params["base_freq"]
        # Convert Hz to MIDI: MIDI = 69 + 12 * log2(freq/440)
        if freq > 0:
            midi = 69 + 12 * np.log2(freq / 440.0)
            return int(round(midi))

    return None


def estimate_velocity(params: Dict[str, float]) -> int:
    """Estimate MIDI velocity from parameters."""
    # Check for note_velocity parameter
    if "note_velocity" in params:
        return int(round(params["note_velocity"]))

    # Could also use amplitude or other params
    return 64  # Default


def write_midi_file(notes: List[SynthesizedNote], output_path: str, tempo_bpm: float = 120.0):
    """Write ground truth MIDI file."""
    try:
        import mido
    except ImportError:
        print("Warning: mido not installed, skipping MIDI output")
        return

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    ticks_per_beat = mid.ticks_per_beat

    # Sort notes by onset time
    sorted_notes = sorted(notes, key=lambda n: n.onset_time)

    # Convert to MIDI events
    current_tick = 0
    for note in sorted_notes:
        if note.midi_note is None:
            continue

        # Clamp MIDI note to valid range
        midi_note = max(0, min(127, note.midi_note))
        velocity = max(1, min(127, note.velocity))

        # Convert time to ticks
        note_tick = int(note.onset_time * tempo_bpm / 60.0 * ticks_per_beat)
        delta = max(0, note_tick - current_tick)

        # Note on
        track.append(mido.Message('note_on', note=midi_note, velocity=velocity, time=delta))

        # Note off (after duration)
        duration_ticks = int(note.duration * tempo_bpm / 60.0 * ticks_per_beat)
        track.append(mido.Message('note_off', note=midi_note, velocity=0, time=duration_ticks))

        current_tick = note_tick + duration_ticks

    mid.save(output_path)
    print(f"Wrote MIDI: {output_path}")


def generate_test_audio(
    ckpt_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    output_path: str = "test_sequence.wav",
    use_train: bool = False,
) -> Tuple[np.ndarray, List[SynthesizedNote]]:
    """Generate concatenated test audio from dataset.

    Args:
        ckpt_path: Path to checkpoint (uses its training dataset)
        data_dir: Direct path to dataset directory
        num_samples: Number of samples to use (None = all)
        output_path: Output WAV file path
        use_train: Use training set instead of test set

    Returns:
        Tuple of (audio_array, list_of_notes)
    """
    # Load dataset
    if ckpt_path:
        print(f"Loading dataset from checkpoint: {ckpt_path}")
        dataset, dataset_info = load_dataset_from_checkpoint(ckpt_path)
    elif data_dir:
        print(f"Loading dataset from directory: {data_dir}")
        dataset, dataset_info = load_dataset_from_dir(data_dir)
    else:
        print("Error: Must specify either checkpoint or data directory")
        sys.exit(1)

    # Get config
    sample_rate = dataset_info.get("sample_rate", 8000)
    duration = dataset_info.get("duration", 1.0)
    param_names = dataset_info.get("parameter_names", [])
    synth_type = dataset_info.get("synth_type", "saw")

    print(f"\nDataset info:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Synth type: {synth_type}")
    print(f"  Parameters: {param_names}")
    print(f"  Test samples: {len(dataset)}")

    # Create synthesizer
    synth = get_synth_for_dataset(dataset_info, sample_rate)

    # Determine number of samples
    total_samples = len(dataset)
    if num_samples is None:
        num_samples = total_samples
    num_samples = min(num_samples, total_samples)

    print(f"\nGenerating audio from {num_samples} samples...")

    # Generate audio for each sample
    audio_segments = []
    notes = []
    current_time = 0.0

    for i in range(num_samples):
        # Extract parameters
        params = extract_params_from_sample(dataset, i, param_names)
        params["duration"] = duration

        # Synthesize audio
        try:
            audio = synth.generate_audio(params)
        except Exception as e:
            print(f"Warning: Failed to synthesize sample {i}: {e}")
            continue

        # Record note info
        note = SynthesizedNote(
            onset_time=current_time,
            duration=duration,
            params=params.copy(),
            midi_note=estimate_midi_note(params),
            velocity=estimate_velocity(params),
        )
        notes.append(note)

        # Append audio
        audio_segments.append(audio)
        current_time += len(audio) / sample_rate

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    # Concatenate audio
    full_audio = np.concatenate(audio_segments)
    total_duration = len(full_audio) / sample_rate

    print(f"\nTotal audio duration: {total_duration:.2f}s ({len(notes)} notes)")

    # Write audio file
    try:
        import soundfile as sf
        sf.write(output_path, full_audio, sample_rate)
        print(f"Wrote audio: {output_path}")
    except ImportError:
        print("Error: soundfile not installed")
        sys.exit(1)

    # Write ground truth text file
    truth_path = Path(output_path).with_suffix(".txt").with_stem(
        Path(output_path).stem + "_truth"
    )
    with open(truth_path, "w") as f:
        f.write(f"# Ground truth for {output_path}\n")
        f.write(f"# Sample rate: {sample_rate} Hz\n")
        f.write(f"# Note duration: {duration} s\n")
        f.write(f"# Synth type: {synth_type}\n")
        f.write(f"# Parameters: {', '.join(param_names)}\n")
        f.write(f"# Total notes: {len(notes)}\n")
        f.write("#\n")
        f.write("# onset_time(s)")
        if notes and notes[0].midi_note is not None:
            f.write("\tmidi_note\tvelocity")
        for pname in param_names:
            f.write(f"\t{pname}")
        f.write("\n")

        for note in notes:
            f.write(f"{note.onset_time:.4f}")
            if note.midi_note is not None:
                f.write(f"\t{note.midi_note}\t{note.velocity}")
            for pname in param_names:
                f.write(f"\t{note.params.get(pname, 0.0):.6f}")
            f.write("\n")

    print(f"Wrote ground truth: {truth_path}")

    # Write MIDI file if we have pitch info
    has_pitch = any(n.midi_note is not None for n in notes)
    if has_pitch:
        midi_path = Path(output_path).with_suffix(".mid").with_stem(
            Path(output_path).stem + "_truth"
        )
        write_midi_file(notes, str(midi_path))

    return full_audio, notes


def main():
    epilog = """
Examples:
  python generate_test_audio.py                    # Use default checkpoint, all test samples
  python generate_test_audio.py -n 100             # Generate 100 samples
  python generate_test_audio.py -n 100 -o test.wav # Specify output file
  python generate_test_audio.py path/to/model.ckpt # Use specific checkpoint

Workflow:
  1. python generate_test_audio.py -n 100 -o test_100.wav
  2. python predict_params.py --multi --midi-out test_100_nsmt.mid test_100.wav
  3. python compare_midi.py test_100.wav test_100_nsmt.mid test_100_truth.mid

If no checkpoint is specified, uses the reference saw+wah+delay model.
"""
    parser = argparse.ArgumentParser(
        description="Generate test audio from VIMH dataset ground truth",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Checkpoint path or dataset directory (default: reference model)"
    )
    parser.add_argument(
        "--data-dir",
        help="Dataset directory (alternative to checkpoint)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of samples to use (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        default="test_sequence.wav",
        help="Output WAV file path (default: test_sequence.wav)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Use training set instead of test set"
    )

    args = parser.parse_args()

    # Determine source
    ckpt_path = None
    data_dir = args.data_dir

    if args.source:
        source_path = Path(args.source)

        if source_path.exists():
            if source_path.is_dir():
                data_dir = str(source_path)
            elif source_path.suffix == ".ckpt":
                ckpt_path = str(source_path)
            else:
                print(f"Error: Unknown source type: {args.source}")
                sys.exit(1)
        elif source_path.suffix == ".ckpt":
            # Try to resolve from reference checkpoints (handle timestamped names)
            ref_dir = PROJECT_ROOT / "checkpoints/reference"
            if ref_dir.exists():
                # Look for files ending with the provided name (handling the timestamp prefix)
                # e.g., "experiment.ckpt" matches "2025-01-01_12-00-00_experiment.ckpt"
                matches = sorted(ref_dir.glob(f"*{source_path.name}"))
                if matches:
                    ckpt_path = str(matches[-1])
                    print(f"Resolved '{args.source}' to latest reference: {ckpt_path}")
                else:
                    print(f"Error: Checkpoint not found locally or in references: {args.source}")
                    sys.exit(1)
            else:
                print(f"Error: Checkpoint not found: {args.source}")
                sys.exit(1)
        else:
            print(f"Error: Source not found: {args.source}")
            sys.exit(1)

    if not ckpt_path and not data_dir:
        # Use default checkpoint (same as predict_params.py)
        default_ckpt = Path(DEFAULT_CKPT)
        if default_ckpt.exists():
            ckpt_path = str(default_ckpt)
            print(f"Using default checkpoint: {ckpt_path}")
        else:
            print(f"Error: Default checkpoint not found: {DEFAULT_CKPT}")
            print("Specify a checkpoint or --data-dir explicitly.")
            sys.exit(1)

    generate_test_audio(
        ckpt_path=ckpt_path,
        data_dir=data_dir,
        num_samples=args.num_samples,
        output_path=args.output,
        use_train=args.train,
    )


if __name__ == "__main__":
    main()
