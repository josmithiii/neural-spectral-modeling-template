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
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


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


def extract_parameter_values(
    binary_file: Path, param_names: List[str], param_mappings: Dict[str, Any]
) -> Tuple[List[List[float]], List[str]]:
    """Extract parameter values from VIMH binary file.

    Returns:
        param_values: List of parameter value lists (one per sample)
        denormalized_param_names: List of parameter names with denormalized ranges
    """
    if not binary_file.exists():
        raise FileNotFoundError(f"Binary file not found: {binary_file}")

    QUANTIZATION_LEVELS = 255
    param_values = []

    with open(binary_file, "rb") as f:
        while True:
            # Read metadata: height, width, channels (6 bytes)
            metadata_bytes = f.read(6)
            if len(metadata_bytes) < 6:
                break  # End of file

            height, width, channels = struct.unpack("<HHH", metadata_bytes)

            # Read scale factors (8 bytes)
            scale_bytes = f.read(8)
            if len(scale_bytes) < 8:
                break
            spec_min, spec_max = struct.unpack("<ff", scale_bytes)

            # Read number of parameters (1 byte)
            num_params_bytes = f.read(1)
            if len(num_params_bytes) < 1:
                break
            num_params = struct.unpack("B", num_params_bytes)[0]

            # Read parameter pairs (2 bytes each: param_id, quantized_value)
            sample_params = []
            for _ in range(num_params):
                param_pair_bytes = f.read(2)
                if len(param_pair_bytes) < 2:
                    break
                param_id, quantized_value = struct.unpack("BB", param_pair_bytes)

                # Denormalize from quantized value back to original range
                if param_id < len(param_names):
                    param_name = param_names[param_id]
                    if param_name in param_mappings:
                        param_info = param_mappings[param_name]
                        min_val = param_info.get("min", 0.0)
                        max_val = param_info.get("max", 1.0)

                        # Convert quantized [0, QUANTIZATION_LEVELS] back to normalized [0, 1]
                        normalized = quantized_value / QUANTIZATION_LEVELS
                        # Denormalize to original parameter range
                        denormalized = min_val + normalized * (max_val - min_val)
                        sample_params.append(denormalized)
                    else:
                        sample_params.append(quantized_value / QUANTIZATION_LEVELS)
                else:
                    sample_params.append(quantized_value / QUANTIZATION_LEVELS)

            param_values.append(sample_params)

            # Skip image data
            image_size = height * width * channels
            f.seek(image_size, 1)  # Skip forward by image_size bytes

    return param_values, param_names


def analyze_parameter_distributions(
    param_values: List[List[float]], param_names: List[str], param_mappings: Dict[str, Any]
) -> None:
    """Analyze and print parameter distribution statistics."""
    if not param_values or not param_names:
        print("No parameter values to analyze")
        return

    print("Parameter Distribution Analysis:")
    print("=" * 50)

    # Convert to numpy array for easier analysis
    param_array = np.array(param_values)

    for i, param_name in enumerate(param_names):
        if i >= param_array.shape[1]:
            continue

        values = param_array[:, i]
        param_info = param_mappings.get(param_name, {})
        min_expected = param_info.get("min", 0.0)
        max_expected = param_info.get("max", 1.0)

        print(f"\n{param_name}:")
        print(f"  Expected range: [{min_expected:.3f}, {max_expected:.3f}]")
        print(f"  Actual range:   [{values.min():.3f}, {values.max():.3f}]")
        print(f"  Mean: {values.mean():.3f}")
        print(f"  Std:  {values.std():.3f}")
        print(f"  Median: {np.median(values):.3f}")

        # Check for uniform distribution by looking at histogram
        hist, bin_edges = np.histogram(values, bins=10)
        expected_per_bin = len(values) / 10

        # Chi-square-like test for uniformity (simplified)
        chi_stat = np.sum((hist - expected_per_bin) ** 2 / expected_per_bin)
        print(f"  Uniformity test (chi-square like): {chi_stat:.2f}")
        print(f"    (Lower values indicate more uniform distribution)")

        # Show histogram bins
        print("  Histogram (10 bins):")
        for j, (count, left_edge, right_edge) in enumerate(
            zip(hist, bin_edges[:-1], bin_edges[1:])
        ):
            bar_width = int(count * 40 / max(hist))  # Scale to 40 chars max
            bar = "█" * bar_width
            print(f"    [{left_edge:.3f}-{right_edge:.3f}]: {count:3d} {bar}")


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

    parser.add_argument(
        "-p",
        "--parameters",
        action="store_true",
        help="Analyze parameter value distributions from binary files",
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

        # Analyze parameter distributions if requested
        if args.parameters:
            print("\n" + "=" * 60)
            param_names = metadata.get("parameter_names", [])
            param_mappings = metadata.get("parameter_mappings", {})

            if not param_names:
                print("No varying parameters found in dataset")
            else:
                # Analyze both train and test files
                for split in ["train", "test"]:
                    binary_file = dataset_path / split
                    if binary_file.exists():
                        print(f"\n{split.upper()} SET:")
                        try:
                            param_values, _ = extract_parameter_values(
                                binary_file, param_names, param_mappings
                            )
                            analyze_parameter_distributions(
                                param_values, param_names, param_mappings
                            )
                        except Exception as e:
                            print(f"Error analyzing {split} parameters: {e}")
                    else:
                        print(f"\n{split.upper()} SET: file not found")

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
