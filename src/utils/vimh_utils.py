"""Utility functions for VIMH dataset metadata handling."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_vimh_metadata(data_dir: str) -> Dict:
    """Load VIMH dataset metadata from JSON file.

    :param data_dir: Path to dataset directory
    :return: Dictionary containing dataset metadata
    :raises FileNotFoundError: If metadata file doesn't exist
    :raises json.JSONDecodeError: If metadata file is malformed
    """
    metadata_file = Path(data_dir) / 'vimh_dataset_info.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"VIMH metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata


def get_parameter_names_from_metadata(data_dir: str) -> List[str]:
    """Get parameter names from VIMH dataset metadata.

    :param data_dir: Path to dataset directory
    :return: List of parameter names
    """
    metadata = load_vimh_metadata(data_dir)
    return metadata.get('parameter_names', [])


def get_image_dimensions_from_metadata(data_dir: str) -> Tuple[int, int, int]:
    """Get image dimensions from VIMH dataset metadata.

    :param data_dir: Path to dataset directory
    :return: Tuple of (height, width, channels)
    """
    metadata = load_vimh_metadata(data_dir)
    height = metadata.get('height', 32)
    width = metadata.get('width', 32)
    channels = metadata.get('channels', 3)
    return height, width, channels


def get_parameter_ranges_from_metadata(data_dir: str) -> Dict[str, Tuple[float, float]]:
    """Get parameter ranges from VIMH dataset metadata.

    :param data_dir: Path to dataset directory
    :return: Dictionary mapping parameter names to (min, max) tuples
    """
    metadata = load_vimh_metadata(data_dir)
    parameter_ranges = {}

    if 'parameter_names' in metadata and 'parameter_mappings' in metadata:
        param_names = metadata['parameter_names']
        param_mappings = metadata['parameter_mappings']

        for param_name in param_names:
            if param_name in param_mappings:
                mapping = param_mappings[param_name]
                parameter_ranges[param_name] = (mapping['min'], mapping['max'])

    return parameter_ranges


def get_heads_config_from_metadata(data_dir: str) -> Dict[str, int]:
    """Get heads configuration from VIMH dataset metadata.

    :param data_dir: Path to dataset directory
    :return: Dictionary mapping head names to number of classes
    """
    metadata = load_vimh_metadata(data_dir)
    heads_config = {}

    if 'parameter_names' in metadata:
        param_names = metadata['parameter_names']

        if 'parameter_mappings' not in metadata:
            raise ValueError("VIMH metadata missing 'parameter_mappings'")

        param_mappings = metadata['parameter_mappings']
        for param_name in param_names:
            if param_name not in param_mappings:
                raise ValueError(f"Parameter '{param_name}' not found in parameter_mappings")
            info = param_mappings[param_name]
            if not all(k in info for k in ('min', 'max', 'step')):
                raise ValueError(f"Parameter '{param_name}' missing min/max/step in metadata")
            step = float(info['step'])
            if step <= 0:
                raise ValueError(f"Parameter '{param_name}' has non-positive step: {step}")
            num = (float(info['max']) - float(info['min'])) / step
            steps = int(round(num))
            if abs(num - steps) > 1e-6:
                raise ValueError(f"Parameter '{param_name}' has non-integer steps: (max-min)/step={num}")
            heads_config[param_name] = steps + 1

    return heads_config
