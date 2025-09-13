#!/usr/bin/env python3
"""
VIMH Dataset Metadata Display Utility

Prints metadata for VIMH (Variable Image MultiHead) datasets.
Defaults to the most recently created dataset in ./data/vimh-*

Usage:
    python vimhd.py [dataset_path]

Examples:
    python vimhd.py                                              # Latest dataset
    python vimhd.py data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_3p  # Specific dataset
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def find_latest_dataset() -> Path:
    """Find the most recently created VIMH dataset in ./data/"""
    data_dir = Path("./data")
    if not data_dir.exists():
        raise FileNotFoundError("./data directory not found")

    vimh_datasets = list(data_dir.glob("vimh-*"))
    if not vimh_datasets:
        raise FileNotFoundError("No VIMH datasets found in ./data/")

    # Sort by modification time, newest first
    latest_dataset = max(vimh_datasets, key=lambda p: p.stat().st_mtime)
    return latest_dataset


def load_dataset_metadata(dataset_path: Path) -> Dict[str, Any]:
    """Load VIMH dataset metadata from vimh_dataset_info.json"""
    metadata_file = dataset_path / "vimh_dataset_info.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file) as f:
        return json.load(f)


def format_parameter_info(
    param_name: str, param_data: Dict[str, Any], is_varying: bool = True
) -> str:
    """Format parameter information for display"""
    if is_varying:
        min_val = param_data.get("min", "N/A")
        max_val = param_data.get("max", "N/A")
        step = param_data.get("step", "N/A")
        num_classes = param_data.get("num_classes", "N/A")
        desc = param_data.get("description", "No description")

        if min_val == max_val:
            range_str = f"{min_val} (fixed)"
        else:
            range_str = f"{min_val} to {max_val} (step: {step})"

        return f"    {param_name}: {range_str}, classes: {num_classes}\n      {desc}"
    else:
        value = param_data.get("value", "N/A")
        desc = param_data.get("description", "No description")
        return f"    {param_name}: {value} (fixed)\n      {desc}"


def print_dataset_metadata(metadata: Dict[str, Any], dataset_path: Path) -> None:
    """Print formatted dataset metadata"""

    print(f"VIMH Dataset: {dataset_path.name}")
    print("=" * 60)

    # Basic info
    print(f"Format: {metadata.get('format', 'N/A')} v{metadata.get('version', 'N/A')}")
    print(f"Output format: {metadata.get('output_format', 'N/A')}")
    print()

    # Image dimensions
    h, w, c = (
        metadata.get("height", "N/A"),
        metadata.get("width", "N/A"),
        metadata.get("channels", "N/A"),
    )
    print(f"Image dimensions: {h}×{w}×{c}")

    if "channel_labels" in metadata:
        labels = ", ".join(metadata["channel_labels"])
        print(f"Channel labels: {labels}")
    print()

    # Sample counts
    train_samples = metadata.get("train_samples", "N/A")
    test_samples = metadata.get("test_samples", "N/A")
    total_samples = metadata.get("total_samples", "N/A")
    print(f"Samples: {train_samples} train, {test_samples} test, {total_samples} total")
    print()

    # Audio/synthesis parameters
    if "sample_rate" in metadata:
        print(f"Audio: {metadata['sample_rate']} Hz, {metadata.get('duration', 'N/A')}s duration")
        print(f"Synth type: {metadata.get('synth_type', 'N/A')}")
        print()

    # Parameters
    varying_params = metadata.get("varying_parameters", 0)
    param_names = metadata.get("parameter_names", [])
    print(f"Varying parameters: {varying_params}")
    if param_names:
        print(f"Parameter names: {', '.join(param_names)}")
    print()

    # Parameter mappings
    if "parameter_mappings" in metadata:
        print("Parameter Details:")
        param_mappings = metadata["parameter_mappings"]

        # Separate varying and fixed parameters
        varying_param_names = set(param_names)

        if varying_param_names:
            print("  Varying:")
            for param_name in param_names:
                if param_name in param_mappings:
                    print(format_parameter_info(param_name, param_mappings[param_name], True))

        fixed_params = metadata.get("fixed_parameters", {})
        if fixed_params:
            print("  Fixed:")
            for param_name, param_data in fixed_params.items():
                print(format_parameter_info(param_name, param_data, False))
        print()

    # Spectrogram configuration
    if "spectrogram_config" in metadata:
        spec_config = metadata["spectrogram_config"]
        print("Spectrogram Configuration:")
        print(f"  Type: {spec_config.get('type', 'N/A')}")
        print(f"  Method: {spec_config.get('method', 'N/A')}")
        print(f"  FFT size: {spec_config.get('n_fft', 'N/A')}")
        print(
            f"  Window: {spec_config.get('window_type', 'N/A')} ({spec_config.get('n_window', 'N/A')} samples)"
        )
        print(f"  Hop length: {spec_config.get('hop_length', 'N/A')}")
        print(f"  Frequency bins: {spec_config.get('n_bins', 'N/A')}")
        if "bins_per_harmonic" in spec_config:
            print(f"  Bins per harmonic: {spec_config['bins_per_harmonic']}")
        print()

    # Mel configuration
    if "mel_config" in metadata:
        mel_config = metadata["mel_config"]
        print("Mel Configuration:")
        print(f"  Freq min: {mel_config.get('freq_min', 'N/A')} Hz")
        print(f"  Freq max ratio: {mel_config.get('freq_max_ratio', 'N/A')}")
        print()

    # Pre-emphasis
    if "pre_emphasis_coefficient" in metadata:
        coeff = metadata["pre_emphasis_coefficient"]
        print(f"Pre-emphasis coefficient: {coeff}")
        print()

    # File sizes
    train_file = dataset_path / "train"
    test_file = dataset_path / "test"

    file_sizes = []
    if train_file.exists():
        size_mb = train_file.stat().st_size / (1024 * 1024)
        file_sizes.append(f"train: {size_mb:.1f} MB")
    if test_file.exists():
        size_mb = test_file.stat().st_size / (1024 * 1024)
        file_sizes.append(f"test: {size_mb:.1f} MB")

    if file_sizes:
        print(f"File sizes: {', '.join(file_sizes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Display VIMH dataset metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help="Path to VIMH dataset directory (defaults to latest in ./data/vimh-*)",
    )

    args = parser.parse_args()

    try:
        if args.dataset:
            dataset_path = Path(args.dataset)
        else:
            dataset_path = find_latest_dataset()

        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}", file=sys.stderr)
            sys.exit(1)

        if not dataset_path.is_dir():
            print(f"Error: Path is not a directory: {dataset_path}", file=sys.stderr)
            sys.exit(1)

        metadata = load_dataset_metadata(dataset_path)
        print_dataset_metadata(metadata, dataset_path)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in metadata file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
