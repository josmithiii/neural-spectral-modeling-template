#!/usr/bin/env python3
"""
Clean SimpleSawSynth dataset generation for VIMH format.
This is a rewrite for the current template-based project structure.
"""
import argparse
import json
import logging
import math
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# Import shared synthesis utilities
from src.utils.synth_utils import (
    DEFAULT_DURATION,
    DEFAULT_LOG10_DECAY_TIME,
    DEFAULT_NOTE_NUMBER,
    DEFAULT_NOTE_VELOCITY,
    MIN_SPECTROGRAM_VALUE,
    QUANTIZATION_LEVELS,
    SimpleSawSynth,
    SpectrogramProcessor,
    pre_emphasis_filter,
    prepare_channels,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRAIN_TEST_SPLIT = 0.8


def get_default_channel_labels(channels: int) -> List[str]:
    """Get default channel labels based on channel count."""
    if channels == 1:
        return ["Gray"]
    elif channels == 3:
        return ["R", "G", "B"]
    elif channels == 4:
        return ["R", "G", "B", "A"]
    else:
        return [f"Ch{i}" for i in range(channels)]


def extract_temporal_envelope(spectrogram: np.ndarray, kernel_size: int = 5, eps: float = 1e-10) -> np.ndarray:
    """
    Extract temporal envelope (energy evolution over time).

    Args:
        spectrogram: Input spectrogram [H, W, C] or [H, W]
        kernel_size: Size of smoothing kernel
        eps: Small value for numerical stability

    Returns:
        Temporal envelope broadcasted to full spectrogram size
    """
    # Ensure we have 3D input
    if spectrogram.ndim == 2:
        spectrogram = np.expand_dims(spectrogram, axis=2)  # [H, W, 1]

    H, W, C = spectrogram.shape

    # Convert to torch for processing
    spec_tensor = torch.from_numpy(spectrogram).float()
    spec_tensor = spec_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    # Compute energy per time frame (sum over frequency)
    temporal_energy = spec_tensor.sum(dim=2, keepdim=True)  # [1, C, 1, W]

    # Smooth the envelope to remove noise
    if kernel_size >= 3 and W >= kernel_size:
        # Apply smoothing to each channel separately
        smoothed_channels = []
        for c in range(C):
            channel_energy = temporal_energy[:, c:c+1, :, :]  # [1, 1, 1, W]
            kernel = torch.ones(1, 1, 1, kernel_size, device=spec_tensor.device) / kernel_size
            smoothed = F.conv2d(channel_energy, kernel, padding=(0, kernel_size//2))
            smoothed_channels.append(smoothed)
        temporal_energy = torch.cat(smoothed_channels, dim=1)

    # Broadcast back to full spectrogram size
    temporal_envelope = temporal_energy.expand(-1, -1, H, -1)  # [1, C, H, W]

    # Convert back to numpy and transpose to [H, W, C]
    result = temporal_envelope.squeeze(0).permute(1, 2, 0).numpy()

    return result


def extract_spectral_envelope(spectrogram: np.ndarray, kernel_size: int = 7, eps: float = 1e-10) -> np.ndarray:
    """
    Extract spectral envelope (energy distribution over frequency).

    Args:
        spectrogram: Input spectrogram [H, W, C] or [H, W]
        kernel_size: Size of smoothing kernel
        eps: Small value for numerical stability

    Returns:
        Spectral envelope broadcasted to full spectrogram size
    """
    # Ensure we have 3D input
    if spectrogram.ndim == 2:
        spectrogram = np.expand_dims(spectrogram, axis=2)  # [H, W, 1]

    H, W, C = spectrogram.shape

    # Convert to torch for processing
    spec_tensor = torch.from_numpy(spectrogram).float()
    spec_tensor = spec_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    # Compute average spectrum (average over time)
    spectral_profile = spec_tensor.mean(dim=3, keepdim=True)  # [1, C, H, 1]

    # Smooth the spectral profile
    if kernel_size >= 3 and H >= kernel_size:
        # Apply smoothing to each channel separately
        smoothed_channels = []
        for c in range(C):
            channel_profile = spectral_profile[:, c:c+1, :, :]  # [1, 1, H, 1]
            kernel = torch.ones(1, 1, kernel_size, 1, device=spec_tensor.device) / kernel_size
            smoothed = F.conv2d(channel_profile, kernel, padding=(kernel_size//2, 0))
            smoothed_channels.append(smoothed)
        spectral_profile = torch.cat(smoothed_channels, dim=1)

    # Broadcast back to full spectrogram size
    spectral_envelope = spectral_profile.expand(-1, -1, -1, W)  # [1, C, H, W]

    # Convert back to numpy and transpose to [H, W, C]
    result = spectral_envelope.squeeze(0).permute(1, 2, 0).numpy()

    return result


def add_envelope_channels(
    spectrogram: np.ndarray,
    add_temporal_envelope: bool,
    add_spectral_envelope: bool,
    normalize: bool,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Add envelope channels to spectrogram and optionally apply normalization.

    Args:
        spectrogram: Input spectrogram [H, W] or [H, W, C]
        add_temporal_envelope: Whether to add temporal envelope channels
        add_spectral_envelope: Whether to add spectral envelope channels
        normalize: Whether to divide original channels by envelopes
        eps: Small value for numerical stability

    Returns:
        Enhanced spectrogram with envelope channels
        - For single-channel input: [normalized_channel, temporal_env, spectral_env]
        - For multi-channel input: [normalized_channels..., temporal_envs..., spectral_envs...]
    """
    if not (add_temporal_envelope or add_spectral_envelope or normalize):
        return spectrogram

    # Ensure 3D input
    if spectrogram.ndim == 2:
        spectrogram = np.expand_dims(spectrogram, axis=2)

    H, W, C = spectrogram.shape

    # For envelope generation from single-channel input, we want single envelopes, not per-channel
    if C == 1 and (add_temporal_envelope or add_spectral_envelope):
        # Single-channel case: generate one envelope per type from the single channel
        channels_to_add = []

        # Extract envelopes from the single channel
        temporal_envelope = None
        spectral_envelope = None

        if add_temporal_envelope or normalize:
            temporal_envelope = extract_temporal_envelope(spectrogram)
            # For single channel, take only the first envelope channel
            temporal_envelope = temporal_envelope[:, :, :1]

        if add_spectral_envelope or normalize:
            spectral_envelope = extract_spectral_envelope(spectrogram)
            # For single channel, take only the first envelope channel
            spectral_envelope = spectral_envelope[:, :, :1]

        # Apply normalization if requested
        normalized_spec = spectrogram.copy()
        if normalize:
            if temporal_envelope is not None:
                normalized_spec = normalized_spec / (temporal_envelope + eps)
            if spectral_envelope is not None:
                normalized_spec = normalized_spec / (spectral_envelope + eps)

        # Build final channel list
        all_channels = [normalized_spec]
        if add_temporal_envelope:
            all_channels.append(temporal_envelope)
        if add_spectral_envelope:
            all_channels.append(spectral_envelope)

        result = np.concatenate(all_channels, axis=2)
        return result

    else:
        # Multi-channel case: original behavior
        channels_to_add = []

        # Extract envelopes if needed
        temporal_envelope = None
        spectral_envelope = None

        if add_temporal_envelope or normalize:
            temporal_envelope = extract_temporal_envelope(spectrogram)

        if add_spectral_envelope or normalize:
            spectral_envelope = extract_spectral_envelope(spectrogram)

        # Apply normalization if requested
        normalized_spec = spectrogram.copy()
        if normalize:
            if temporal_envelope is not None:
                normalized_spec = normalized_spec / (temporal_envelope + eps)
            if spectral_envelope is not None:
                normalized_spec = normalized_spec / (spectral_envelope + eps)

        # Add envelope channels
        if add_temporal_envelope:
            channels_to_add.append(temporal_envelope)
        if add_spectral_envelope:
            channels_to_add.append(spectral_envelope)

        # Concatenate all channels
        if channels_to_add:
            all_channels = [normalized_spec] + channels_to_add
            result = np.concatenate(all_channels, axis=2)
        else:
            result = normalized_spec

        return result


def calculate_total_channels(
    base_channels: int,
    add_temporal_envelope: bool,
    add_spectral_envelope: bool
) -> int:
    """Calculate total number of channels after adding envelopes."""
    if add_temporal_envelope or add_spectral_envelope:
        # When using envelopes, we start with 1 base channel and add envelope channels
        total = 1  # Base normalized channel
        if add_temporal_envelope:
            total += 1  # One temporal envelope
        if add_spectral_envelope:
            total += 1  # One spectral envelope
        return total
    else:
        # Standard case without envelopes
        return base_channels


def generate_channel_labels(
    base_labels: List[str],
    add_temporal_envelope: bool,
    add_spectral_envelope: bool,
    normalize: bool = False
) -> List[str]:
    """Generate channel labels including envelope channels."""
    if add_temporal_envelope or add_spectral_envelope:
        # When using envelopes, we use only the first base label as the source
        labels = []
        base_label = base_labels[0]  # Use first label (usually "Gray")

        # Base channel (always normalized when using envelopes)
        if normalize:
            labels.append(f"{base_label}_normalized")
        else:
            labels.append(base_label)

        # Add envelope labels
        if add_temporal_envelope:
            labels.append(f"{base_label}_temporal_envelope")
        if add_spectral_envelope:
            labels.append(f"{base_label}_spectral_envelope")

        return labels
    else:
        # Standard case without envelopes
        return base_labels


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration parameters."""
    if cfg.dataset.size <= 0:
        raise ValueError(f"Dataset size must be positive, got {cfg.dataset.size}")
    if cfg.dataset.sampling_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {cfg.dataset.sampling_rate}")
    if cfg.dataset.duration <= 0:
        raise ValueError(f"Duration must be positive, got {cfg.dataset.duration}")
    if cfg.generate.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {cfg.generate.batch_size}")
    if cfg.generate.height <= 0 or cfg.generate.width <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got {cfg.generate.height}x{cfg.generate.width}"
        )
    if cfg.generate.channels not in [1, 3]:
        raise ValueError(f"Channels must be 1 or 3, got {cfg.generate.channels}")
    if cfg.stft.type not in ["mel", "stft"]:
        raise ValueError(f"STFT type must be 'mel' or 'stft', got {cfg.stft.type}")


class ParameterGenerator:
    """Handles parameter generation and validation."""

    def __init__(self, params_config: Dict[str, Dict[str, Any]]):
        self.params_config = params_config
        self.varying_params = []

        for param_name, param_info in params_config.items():
            if param_info["min_value"] != param_info["max_value"]:
                self.varying_params.append(param_name)

    def generate_random_parameters(self, duration: float) -> Tuple[Dict[str, float], List[float]]:
        """Generate random parameters and normalized label vector."""
        params = {"duration": duration}
        label_vector = []

        for param_name, param_info in self.params_config.items():
            min_val = param_info["min_value"]
            max_val = param_info["max_value"]
            step_val = param_info.get("step", None)  # Optional step parameter

            if min_val == max_val:
                value = min_val
            else:
                value = np.random.uniform(min_val, max_val)

            params[param_name] = value

            # Only add varying parameters to labels
            if min_val != max_val:
                normalized = (value - min_val) / (max_val - min_val)
                label_vector.append(normalized)

        return params, label_vector


def generate_sample_batch(
    synth: SimpleSawSynth,
    param_generator: ParameterGenerator,
    spectrogram_processor: SpectrogramProcessor,
    batch_size: int,
    duration: float,
    channels: int,
    pre_emphasis_coeff: float = 0.0,
    add_temporal_envelope: bool = False,
    add_spectral_envelope: bool = False,
    normalize: bool = False,
    eps: float = 1e-10,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[float, float]]]:
    """Generate a batch of spectrograms with scale factors.

    Returns:
        spectrograms: List of normalized spectrograms
        labels: List of parameter label vectors
        scale_factors: List of (spec_min, spec_max) tuples for each spectrogram
    """
    spectrograms = []
    labels = []
    scale_factors = []

    for _ in range(batch_size):
        try:
            # Generate parameters
            params, label_vector = param_generator.generate_random_parameters(duration)

            # Generate audio
            audio = synth.generate_audio(params)

            # Apply pre-emphasis filter if configured
            if pre_emphasis_coeff > 0:
                audio = pre_emphasis_filter(audio, pre_emphasis_coeff)

            # Process spectrogram and get scale factors
            spectrogram, spec_min, spec_max = spectrogram_processor.audio_to_spectrogram(
                params, audio
            )

            # Handle channels - if using envelopes, skip prepare_channels and let add_envelope_channels handle it
            if add_temporal_envelope or add_spectral_envelope:
                # Use original single-channel spectrogram for envelope generation
                final_spectrogram = add_envelope_channels(
                    spectrogram,  # Pass original 2D spectrogram
                    add_temporal_envelope,
                    add_spectral_envelope,
                    normalize,
                    eps
                )

                # Convert back to uint8 after floating-point envelope operations
                spec_min = final_spectrogram.min()
                spec_max = final_spectrogram.max()
                if spec_max > spec_min:
                    final_spectrogram = ((final_spectrogram - spec_min) / (spec_max - spec_min) * 255.0).astype(np.uint8)
                else:
                    final_spectrogram = np.full_like(final_spectrogram, 128, dtype=np.uint8)
            else:
                # Standard multi-channel replication
                final_spectrogram = prepare_channels(spectrogram, channels)
                final_spectrogram = add_envelope_channels(
                    final_spectrogram,
                    add_temporal_envelope,
                    add_spectral_envelope,
                    normalize,
                    eps
                )

            spectrograms.append(final_spectrogram)
            labels.append(np.array(label_vector))
            scale_factors.append((spec_min, spec_max))

        except Exception as e:
            logger.error(f"Error generating sample: {e}")
            raise

    return spectrograms, labels, scale_factors


def save_vimh_dataset(
    all_spectrograms: List[np.ndarray],
    all_labels: List[np.ndarray],
    all_scale_factors: List[Tuple[float, float]],
    param_names: List[str],
    params_config: Dict[str, Dict[str, Any]],
    dataset_name: str,
    output_dir: str,
    dataset_size: int,
    sample_rate: int,
    duration: float,
    height: int,
    width: int,
    channels: int,
    stft_config: Dict[str, Any],
    mel_config: Dict[str, Any],
    pre_emphasis_coeff: float,
    synth_type: str = "simple",
    pickle_format: bool = False,
    channel_labels: Optional[List[str]] = None,
) -> None:
    """Save dataset in VIMH format."""
    logger.info("🎯 Saving dataset in VIMH format")

    # Set up channel labels (use defaults if not provided)
    if channel_labels is None:
        channel_labels = get_default_channel_labels(channels)
    elif len(channel_labels) != channels:
        raise ValueError(f"Channel labels length {len(channel_labels)} doesn't match channels {channels}")

    # Use configured dimensions
    logger.info(f"VIMH dimensions: {height}x{width}x{channels}")
    logger.info(f"VIMH channel labels: {channel_labels}")
    logger.info(f"VIMH parameters: {len(param_names)} varying parameters")
    logger.info(f"Parameter order: {param_names}")

    # Create VIMH binary data
    vimh_data = []

    for i, (spectrogram, label_vector, scale_factors) in enumerate(
        zip(all_spectrograms, all_labels, all_scale_factors)
    ):
        # Metadata: height, width, channels (6 bytes, 2 bytes each) + scale factors (8 bytes, 4 bytes each)
        metadata = struct.pack("<HHH", height, width, channels)
        spec_min, spec_max = scale_factors
        scale_data = struct.pack("<ff", spec_min, spec_max)  # Two float32 values

        # Label data: N + parameter pairs
        num_params = len(param_names)
        label_data = struct.pack("B", num_params)

        # Add each parameter (id, value) pair
        for param_idx, param_value in enumerate(label_vector):
            # Quantize normalized value [0,1] to [0,QUANTIZATION_LEVELS]
            quantized_value = int(param_value * QUANTIZATION_LEVELS)
            quantized_value = max(0, min(QUANTIZATION_LEVELS, quantized_value))
            label_data += struct.pack("BB", param_idx, quantized_value)

        # Image data: flatten spectrogram in planar format
        if channels == 1:
            if len(spectrogram.shape) == 3:
                spectrogram = spectrogram.squeeze()
            image_data = spectrogram.flatten().tobytes()
        else:
            if len(spectrogram.shape) == 2:
                spectrogram = np.expand_dims(spectrogram, axis=2)
            # Convert to planar format: transpose (H,W,C) -> (C,H,W) before flattening
            spectrogram = spectrogram.transpose(2, 0, 1)
            image_data = spectrogram.flatten().tobytes()

        # Combine all data for this sample (now includes scale factors)
        sample_data = metadata + scale_data + label_data + image_data
        vimh_data.append(sample_data)

    # Split into train/test
    split_idx = int(TRAIN_TEST_SPLIT * len(vimh_data))
    train_data = vimh_data[:split_idx]
    test_data = vimh_data[split_idx:]

    # Save binary files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train")
    test_path = os.path.join(output_dir, "test")

    with open(train_path, "wb") as f:
        for sample in train_data:
            f.write(sample)

    with open(test_path, "wb") as f:
        for sample in test_data:
            f.write(sample)

    logger.info(f"Saved {len(train_data)} training samples to {train_path}")
    logger.info(f"Saved {len(test_data)} test samples to {test_path}")

    # Save pickle format only if requested
    if pickle_format:
        import pickle

        train_pickle_path = os.path.join(output_dir, "train_batch")
        test_pickle_path = os.path.join(output_dir, "test_batch")

        # Convert to pickle format with optimized memory usage
        def create_pickle_data(data_list: List[bytes]) -> Dict[str, Any]:
            n_samples = len(data_list)

            # Calculate actual image size from first sample
            if n_samples > 0:
                first_sample = data_list[0]
                sample_bytes = first_sample[
                    14:
                ]  # Skip metadata (6 bytes) + scale factors (8 bytes)
                num_params = sample_bytes[0]
                image_data = sample_bytes[1 + 2 * num_params :]
                actual_image_size = len(image_data)
            else:
                actual_image_size = height * width * channels

            # Pre-allocate arrays for better memory efficiency
            data_array = np.empty((n_samples, actual_image_size), dtype=np.uint8)
            vimh_labels = []
            scale_factors_array = np.empty((n_samples, 2), dtype=np.float32)  # spec_min, spec_max

            for i, sample_data in enumerate(data_list):
                try:
                    # Parse the sample data to extract image, labels, and scale factors
                    # Extract metadata and scale factors
                    metadata = sample_data[:6]
                    scale_bytes = sample_data[6:14]  # 8 bytes for two float32 values
                    spec_min, spec_max = struct.unpack("<ff", scale_bytes)
                    scale_factors_array[i] = [spec_min, spec_max]

                    # Skip metadata and scale factors (first 14 bytes)
                    sample_bytes = sample_data[14:]

                    # Parse labels
                    num_params = sample_bytes[0]
                    label_bytes = sample_bytes[1 : 1 + 2 * num_params]

                    # Convert to VIMH label format
                    labels = [num_params]
                    for j in range(0, len(label_bytes), 2):
                        param_id = label_bytes[j]
                        param_val = label_bytes[j + 1]
                        labels.extend([param_id, param_val])

                    vimh_labels.append(labels)

                    # Extract image data directly into pre-allocated array
                    image_data = sample_bytes[1 + 2 * num_params :]
                    data_array[i] = np.frombuffer(image_data, dtype=np.uint8)

                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    raise

            return {
                "data": data_array,
                "vimh_labels": vimh_labels,
                "scale_factors": scale_factors_array,
                "height": height,
                "width": width,
                "channels": channels,
                "image_size": actual_image_size,
            }

        train_pickle_data = create_pickle_data(train_data)
        test_pickle_data = create_pickle_data(test_data)

        with open(train_pickle_path, "wb") as f:
            pickle.dump(train_pickle_data, f)

        with open(test_pickle_path, "wb") as f:
            pickle.dump(test_pickle_data, f)

        logger.info(
            f"Saved {len(train_data)} training samples to {train_pickle_path} (pickle format)"
        )
        logger.info(f"Saved {len(test_data)} test samples to {test_pickle_path} (pickle format)")

    # Create parameter mappings for display compatibility
    parameter_mappings = {}
    fixed_parameters = {}
    for param_name, param_info in params_config.items():
        parameter_mappings[param_name] = {
            "min": param_info["min_value"],
            "step": param_info.get("step", (param_info["max_value"] - param_info["min_value"]) / 16),
            "max": param_info["max_value"],
            "description": f"simple parameter: {param_name}",
        }
        # Track fixed parameters (min == max)
        if param_info["min_value"] == param_info["max_value"]:
            fixed_parameters[param_name] = {
                "value": param_info["min_value"],
                "description": param_info.get("description", f"Fixed {param_name}"),
            }

    # Create VIMH dataset info
    vimh_info = {
        "format": "VIMH",
        "version": "2.1",
        "dataset_name": dataset_name,
        "output_format": "both" if pickle_format else "binary",
        "height": height,
        "width": width,
        "channels": channels,
        "channel_labels": channel_labels,
        "varying_parameters": len(param_names),
        "parameter_names": param_names,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "total_samples": dataset_size,
        "n_samples": dataset_size,
        "sample_rate": sample_rate,
        "duration": duration,
        "synth_type": synth_type,
        "image_size": f"{height}x{width}x{channels}",
        "parameter_mappings": parameter_mappings,
        "fixed_parameters": fixed_parameters,
        "pre_emphasis_coefficient": pre_emphasis_coeff,
        "spectrogram_config": {
            "sample_rate": sample_rate,
            "type": stft_config["type"],
            "n_fft": stft_config.get("n_fft", 512),
            "n_window": stft_config.get("n_window", stft_config.get("n_fft", 512)),
            "hop_length": stft_config.get("hop_length", 64),
            "window_type": stft_config.get("window_type", "rectangular"),
            "bins_per_harmonic": stft_config.get("bins_per_harmonic", 1.0),
            "n_bins": height,
            "method": "efficient_leaf",
        },
        "mel_config": {
            "freq_min": mel_config.get("freq_min", 40.0),
            "freq_max_ratio": mel_config.get("freq_max_ratio", 0.9),
        },
        "label_encoding": {
            "format": "[height] [width] [channels] [spec_min] [spec_max] [N] [param1_id] [param1_val] [param2_id] [param2_val] ...",
            "metadata_bytes": 14,  # 6 bytes for dimensions + 8 bytes for scale factors
            "scale_factors": {
                "spec_min": "float32 - minimum dB value before normalization",
                "spec_max": "float32 - maximum dB value before normalization",
            },
            "N_range": [0, QUANTIZATION_LEVELS],
            "param_id_range": [0, QUANTIZATION_LEVELS],
            "param_val_range": [0, QUANTIZATION_LEVELS],
        },
    }

    with open(os.path.join(output_dir, "vimh_dataset_info.json"), "w") as f:
        json.dump(vimh_info, f, indent=4)

    logger.info("✅ VIMH dataset generation completed successfully")
    logger.info(f"Dataset with {dataset_size} samples saved to {output_dir}")
    logger.info(f"Format: VIMH {height}x{width}x{channels} with {len(param_names)} parameters")


def show_help():
    """Show custom help message for generate_vimh.py"""
    help_text = """
Generate VIMH dataset for audio spectrogram classification

Usage:
  python generate_vimh.py [OPTIONS] [HYDRA_OVERRIDES...]

Options:
  -p, --pickle         Generate pickle format files in addition to binary format
                        (default: binary only)
  -s, --shuffle        Shuffle generated samples to randomize order (adds _shuffled suffix)
  --temporal-envelope  Add temporal envelope channels (energy evolution over time)
  --spectral-envelope  Add spectral envelope channels (energy distribution over frequency)
  --normalize          Apply spectral normalization (divide by envelopes, requires envelope flags)
  -h, --help           Show this help message and exit
  --hydra-help         Show Hydra-specific help

Examples:
  python generate_vimh.py                    # Generate binary format only (default)
  python generate_vimh.py --pickle           # Generate both binary and pickle formats
  python generate_vimh.py --shuffle          # Generate with shuffled sample order
  python generate_vimh.py -p -s              # Generate VIMH dataset with pickle format and shuffling
  python generate_vimh.py --temporal-envelope --spectral-envelope  # Add envelope channels
  python generate_vimh.py --temporal-envelope --spectral-envelope --normalize  # Full normalization
  python generate_vimh.py --config-name=foo  # Use different Hydra config
  python generate_vimh.py dataset.size=1000  # Override dataset size

Notes:
  - Binary format is ~50% smaller and faster to generate
  - Pickle format is needed for compatibility with legacy VIMHDataset code - FIXME: Add dataset binary support to all
  - All other options are configured via Hydra config files in configs/
  - Use --hydra-help to see all available configuration options
    """.strip()
    print(help_text)


@hydra.main(version_base="1.3", config_path="configs", config_name="generate_simple")
def main(cfg: DictConfig) -> None:
    """Main function for generating SimpleSawSynth dataset in VIMH format."""
    # Parse command line arguments for --pickle flag from original argv
    original_argv = getattr(sys, "_original_argv", sys.argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pickle", "-p", action="store_true", help="Write out dataset in pickle format")
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle generated samples to randomize order")
    parser.add_argument("--temporal-envelope", action="store_true", help="Add temporal envelope channels (energy evolution over time)")
    parser.add_argument("--spectral-envelope", action="store_true", help="Add spectral envelope channels (energy distribution over frequency)")
    parser.add_argument("--normalize", action="store_true", help="Apply spectral normalization (divide by envelopes, requires envelope flags)")
    args, unknown = parser.parse_known_args(original_argv)

    logger.info("🚀 Starting SimpleSawSynth dataset generation")
    if args.pickle:
        logger.info("📦 Pickle format enabled")
    else:
        logger.info("📦 Binary format only")

    if args.shuffle:
        logger.info("🔀 Shuffle mode enabled - samples will be randomized")

    if args.temporal_envelope:
        logger.info("📈 Temporal envelope extraction enabled")

    if args.spectral_envelope:
        logger.info("📊 Spectral envelope extraction enabled")

    if args.normalize:
        logger.info("🎯 Spectral normalization enabled")
        if not (args.temporal_envelope or args.spectral_envelope):
            logger.error("❌ Normalization requires at least one envelope type (--temporal-envelope or --spectral-envelope)")
            sys.exit(1)

    try:
        # Validate configuration
        validate_config(cfg)

        # Extract configuration
        dataset_size = cfg.dataset.size
        sample_rate = cfg.dataset.sampling_rate
        duration = cfg.dataset.duration
        batch_size = cfg.generate.batch_size
        pre_emphasis_coeff = cfg.generate.pre_emphasis_coefficient

        # Extract output dimensions
        height = cfg.generate.height
        width = cfg.generate.width
        base_channels = cfg.generate.channels

        # Calculate total channels including envelopes
        total_channels = calculate_total_channels(
            base_channels, args.temporal_envelope, args.spectral_envelope
        )

        # Extract STFT configuration
        stft_config = cfg.stft
        mel_config = cfg.mel

        # Create parameter generator
        param_generator = ParameterGenerator(cfg.synthesizer.parameters)
        varying_params = param_generator.varying_params
        vimh_param_generator = None  # Not used in random mode

        logger.info("📊 Dataset configuration:")
        logger.info(f"  • Samples: {dataset_size}")
        logger.info(f"  • Sample rate: {sample_rate} Hz")
        logger.info(f"  • Duration: {duration}s")
        logger.info(f"  • Batch size: {batch_size}")
        logger.info(f"  • Varying parameters: {len(varying_params)}")
        logger.info(f"  • Parameter order: {varying_params}")

        # Create output directory with expected format: vimh-32x32x3_8000Hz_1p0s_16384dss_simple_2p
        # or vimh-vimh-... for VIMH mode, with _shuffled suffix if shuffled
        duration_str = f"{duration}s".replace(".", "p")
        prefix = "vimh"
        shuffle_suffix = "_shuffled" if args.shuffle else ""

        # Add envelope suffix to dataset name
        envelope_suffix = ""
        if args.temporal_envelope and args.spectral_envelope:
            if args.normalize:
                envelope_suffix = "_normalized"
            else:
                envelope_suffix = "_envelopes"
        elif args.temporal_envelope:
            envelope_suffix = "_temporal"
        elif args.spectral_envelope:
            envelope_suffix = "_spectral"

        dataset_name = f"{prefix}-{height}x{width}x{total_channels}_{sample_rate}Hz_{duration_str}_{dataset_size}dss_simple_{len(varying_params)}p{envelope_suffix}{shuffle_suffix}"
        output_dir = os.path.join(cfg.generate.output_dir, dataset_name)

        # Create synthesizer
        synth = SimpleSawSynth(sample_rate=sample_rate)

        # Create spectrogram processor
        spectrogram_processor = SpectrogramProcessor(
            sample_rate=sample_rate,
            height=height,
            width=width,
            stft_config=stft_config,
            mel_config=mel_config,
        )

        # Log spectrogram type and configuration
        logger.info(
            f"Using EfficientLeaf {stft_config['type']} spectrogram with {height} frequency bins"
        )
        logger.info(
            f"STFT config: n_fft={stft_config.get('n_fft', 512)}, hop_length={stft_config.get('hop_length', 64)}"
        )

        # Generate dataset
        num_batches = (dataset_size + batch_size - 1) // batch_size
        all_spectrograms = []
        all_labels = []
        all_scale_factors = []

        logger.info(f"Generating {num_batches} batches...")

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, dataset_size - batch_idx * batch_size)

            # Random parameter generation
            spectrograms, labels, scale_factors = generate_sample_batch(
                synth,
                param_generator,
                spectrogram_processor,
                current_batch_size,
                duration,
                base_channels,
                pre_emphasis_coeff,
                args.temporal_envelope,
                args.spectral_envelope,
                args.normalize,
            )

            all_spectrograms.extend(spectrograms)
            all_labels.extend(labels)
            all_scale_factors.extend(scale_factors)

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Completed batch {batch_idx + 1}/{num_batches}")

        # Shuffle samples if requested
        if args.shuffle:
            logger.info("🔀 Shuffling generated samples...")
            # Create shuffle indices
            indices = list(range(len(all_spectrograms)))
            np.random.shuffle(indices)

            # Shuffle all arrays in sync
            all_spectrograms = [all_spectrograms[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]
            all_scale_factors = [all_scale_factors[i] for i in indices]

            logger.info(f"Shuffled {len(indices)} samples")

        # Pre-emphasis is now applied during batch generation if configured
        if pre_emphasis_coeff > 0:
            logger.info(
                f"Pre-emphasis filter applied during generation (coefficient: {pre_emphasis_coeff})"
            )

        # Save in VIMH format
        params_config = param_generator.params_config

        # Generate channel labels including envelopes
        base_labels = get_default_channel_labels(base_channels)
        final_channel_labels = generate_channel_labels(
            base_labels,
            args.temporal_envelope,
            args.spectral_envelope,
            args.normalize
        )

        save_vimh_dataset(
            all_spectrograms,
            all_labels,
            all_scale_factors,
            varying_params,
            params_config,
            dataset_name,
            output_dir,
            dataset_size,
            sample_rate,
            duration,
            height,
            width,
            total_channels,
            stft_config,
            mel_config,
            pre_emphasis_coeff,
            synth_type="simple",
            pickle_format=args.pickle,
            channel_labels=final_channel_labels,
        )

        logger.info("✅ Dataset generation completed successfully")
        logger.info(f"DATASET_PATH: {output_dir}")

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    # Check for help flags before Hydra processes them
    if "-h" in sys.argv or "--help" in sys.argv:
        show_help()
        sys.exit(0)

    # Store original argv and filter out custom flags so Hydra doesn't see them
    sys._original_argv = sys.argv[:]
    custom_flags = [
        "--pickle", "-p", "--shuffle", "-s",
        "--temporal-envelope", "--spectral-envelope", "--normalize"
    ]
    filtered_argv = []
    for arg in sys.argv:
        if arg not in custom_flags:
            filtered_argv.append(arg)
    sys.argv = filtered_argv

    try:
        main()
    finally:
        # Restore original argv
        sys.argv = sys._original_argv
