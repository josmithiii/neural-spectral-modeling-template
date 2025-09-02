import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from .multihead_dataset_base import MultiheadDatasetBase
from ..utils.auxiliary_features import extract_auxiliary_features, compute_temporal_envelope_from_spectrogram


class VIMHDataset(MultiheadDatasetBase):
    """Variable Image MultiHead (VIMH) format dataset implementation.

    This dataset loader handles the VIMH format with embedded metadata
    for multiple classification heads. The format includes image dimensions,
    number of heads, and parameter mappings in the label data.

    Format specification:
    - Label format: [height] [width] [channels] [N] [param1_id] [param1_val] ... [paramN_id] [paramN_val]
    - Image data: Flattened pixel values in CHW format
    - Metadata: JSON file with parameter mappings and dataset information
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        target_width: float = 0.0,
        auxiliary_features: Optional[list] = None
    ):
        """Initialize VIMH dataset.

        :param data_path: Path to dataset directory or file
        :param train: Whether to load training or test data
        :param transform: Optional transform to apply to images
        :param target_transform: Optional transform to apply to labels
        :param target_width: Standard deviation for soft targets (0.0 = hard targets)
        :param auxiliary_features: List of auxiliary feature types to extract (e.g., ["decay_time"])
        """
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.target_width = target_width
        self.auxiliary_features = auxiliary_features or []

        # Determine the correct file path
        data_path = Path(data_path)
        if data_path.is_dir():
            # Load from directory structure
            self.data_dir = data_path
            self.metadata_file = self.data_dir / 'vimh_dataset_info.json'
            
            # Try both pickle and binary formats (pickle format first for backward compatibility)
            candidate_files = [
                self.data_dir / ('train_batch' if train else 'test_batch'),  # pickle format
                self.data_dir / ('train' if train else 'test')                # binary format
            ]
            
            self.batch_file = None
            for candidate in candidate_files:
                if candidate.exists():
                    self.batch_file = candidate
                    break
            
            if self.batch_file is None:
                raise FileNotFoundError(
                    f"No {'training' if train else 'test'} data found in {self.data_dir}. "
                    f"Looked for: {[f.name for f in candidate_files]}"
                )
        else:
            # Single file specified
            self.batch_file = data_path
            self.data_dir = data_path.parent
            self.metadata_file = self.data_dir / 'vimh_dataset_info.json'

        # Load metadata configuration
        metadata_format = self._load_metadata_config()

        # Initialize base class
        super().__init__(str(self.batch_file), metadata_format)

        # Override heads_config with metadata parameter mappings if available
        self._calculate_heads_config_from_metadata()

        # Validate dataset integrity
        self._validate_dataset()

    def _load_metadata_config(self) -> Dict[str, Any]:
        """Load dataset metadata configuration from JSON file.

        :return: Metadata configuration dictionary
        """
        if not self.metadata_file.exists():
            # Provide default configuration
            return {
                'format': 'VIMH',
                'version': '1.0',
                'parameter_names': ['param_0', 'param_1'],
                'label_encoding': {
                    'format': '[height] [width] [channels] [N] [param1_id] [param1_val] ...',
                    'metadata_bytes': 6,
                    'N_range': [0, 255],
                    'param_id_range': [0, 255],
                    'param_val_range': [0, 255]
                }
            }

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Failed to load metadata from {self.metadata_file}: {e}")

    def _calculate_heads_config_from_metadata(self) -> None:
        """Calculate heads configuration from metadata parameter mappings if available."""
        if 'parameter_mappings' in self.metadata_format:
            param_mappings = self.metadata_format['parameter_mappings']

            # Only include varying parameters in heads_config
            varying_params = self.metadata_format.get('parameter_names', [])

            # Update heads_config with metadata ranges for varying parameters only
            for param_name in varying_params:
                if param_name in param_mappings:
                    param_info = param_mappings[param_name]
                    if 'min' in param_info and 'max' in param_info:
                        # For continuous parameters, use 256 classes (0-255 quantization)
                        self.heads_config[param_name] = 256

    def _validate_dataset(self) -> None:
        """Validate dataset integrity and format compliance."""
        # Check file existence
        if not self.batch_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.batch_file}")

        # Validate format version if specified
        if 'format' in self.metadata_format:
            expected_format = 'VIMH'
            actual_format = self.metadata_format.get('format')
            if actual_format != expected_format:
                print(f"Warning: Expected format '{expected_format}', got '{actual_format}'")

        # Validate sample count matches metadata
        if 'train_samples' in self.metadata_format and self.train:
            expected_samples = self.metadata_format['train_samples']
            actual_samples = len(self.samples)
            if actual_samples != expected_samples:
                print(f"Warning: Expected {expected_samples} training samples, got {actual_samples}")
        elif 'test_samples' in self.metadata_format and not self.train:
            expected_samples = self.metadata_format['test_samples']
            actual_samples = len(self.samples)
            if actual_samples != expected_samples:
                print(f"Warning: Expected {expected_samples} test samples, got {actual_samples}")

    def _create_soft_targets(self, class_index: int, num_classes: int, target_width: float) -> torch.Tensor:
        """Create soft targets as Gaussian distribution around true class.

        :param class_index: True class index
        :param num_classes: Total number of classes
        :param target_width: Standard deviation for soft targets
        :return: Soft target distribution or hard target index
        """
        if target_width == 0.0:
            return class_index  # Hard targets (backward compatible)

        # Soft targets - Gaussian distribution
        class_indices = torch.arange(num_classes, dtype=torch.float32)
        distances = (class_indices - class_index) ** 2
        weights = torch.exp(-distances / (2 * target_width ** 2))
        return weights / weights.sum()  # Normalize to probability distribution

    def _get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample.

        :param idx: Sample index
        :return: Metadata dictionary
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")

        # Parse metadata from the sample's label structure
        # Handle both 2-tuple (pickle format) and 3-tuple (binary format) cases
        sample = self.samples[idx]
        if len(sample) == 2:
            _, labels = sample
        else:
            _, labels, _ = sample

        # Extract parameter information
        metadata = {
            'sample_index': idx,
            'labels': labels.copy(),
            'image_shape': self.image_shape,
            'dataset_type': 'train' if self.train else 'test'
        }

        # Add parameter descriptions if available
        if 'parameter_mappings' in self.metadata_format:
            param_mappings = self.metadata_format['parameter_mappings']
            for param_name, param_value in labels.items():
                if param_name in param_mappings:
                    mapping_info = param_mappings[param_name]

                    # Handle both sources consistently:
                    # - Pickle path stores 0–255 quantized integers
                    # - Binary path stores 0–1 normalized floats
                    # Convert to a normalized value in [0,1] and also report a quantized value.
                    if isinstance(param_value, (int, np.integer)) and 0 <= int(param_value) <= 255:
                        quantized_value = int(param_value)
                        normalized_value = quantized_value / 255.0
                    else:
                        # Treat as normalized already (float tensor/np scalar). Clamp for safety.
                        normalized_value = float(np.clip(param_value, 0.0, 1.0))
                        quantized_value = int(round(normalized_value * 255))

                    # Dequantize parameter value back to actual range
                    param_min = mapping_info.get('min', 0)
                    param_max = mapping_info.get('max', 255)
                    actual_value = param_min + normalized_value * (param_max - param_min)

                    metadata[f'{param_name}_info'] = {
                        'quantized_value': quantized_value,
                        'normalized_value': normalized_value,
                        'actual_value': actual_value,
                        'description': mapping_info.get('description', ''),
                        'range': [param_min, param_max],
                        'scale': mapping_info.get('scale', 'linear')
                    }

        return metadata

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int], Optional[torch.Tensor]]:
        """Get a sample from the dataset.

        :param idx: Sample index
        :return: Tuple of (image_tensor, labels_dict, auxiliary_features)
        """
        image, labels = super().__getitem__(idx)
        
        # Store the original image for auxiliary feature extraction (before transforms)
        original_image = image.clone()

        # Apply soft targets if enabled
        if self.target_width > 0.0:
            soft_labels = {}
            for param_name, quantized_value in labels.items():
                num_classes = self.heads_config.get(param_name, 256)  # Default to 256 classes
                soft_labels[param_name] = self._create_soft_targets(quantized_value, num_classes, self.target_width)
            labels = soft_labels

        # Apply transforms if specified
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        # Extract auxiliary features if requested
        auxiliary_features = None
        if self.auxiliary_features:
            # Create data dict for auxiliary feature extraction
            data_dict = {'image': original_image.unsqueeze(0)}  # Add batch dimension for processing
            
            # Extract auxiliary features and squeeze batch dimension since we're processing single samples
            batch_features = extract_auxiliary_features(data_dict, self.auxiliary_features)
            auxiliary_features = batch_features.squeeze(0)  # [num_features] instead of [1, num_features]

        return image, labels, auxiliary_features

    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific parameter.

        :param param_name: Name of the parameter
        :return: Parameter information dictionary
        """
        if 'parameter_mappings' not in self.metadata_format:
            return {'name': param_name, 'description': 'No metadata available'}

        param_mappings = self.metadata_format['parameter_mappings']
        if param_name not in param_mappings:
            return {'name': param_name, 'description': 'Parameter not found in metadata'}

        return param_mappings[param_name]

    def get_class_distribution(self) -> Dict[str, Dict[int, int]]:
        """Get class distribution for each classification head.

        :return: Dictionary mapping head names to class counts
        """
        distribution = {}

        # Initialize counters for each head
        for head_name in self.heads_config.keys():
            distribution[head_name] = {}

        # Count occurrences
        for _, labels in self.samples:
            for head_name, label_value in labels.items():
                if label_value not in distribution[head_name]:
                    distribution[head_name][label_value] = 0
                distribution[head_name][label_value] += 1

        return distribution

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics.

        :return: Dictionary with dataset statistics
        """
        stats = self.get_dataset_info()
        stats.update({
            'train_mode': self.train,
            'class_distribution': self.get_class_distribution(),
            'transforms': {
                'image_transform': self.transform is not None,
                'target_transform': self.target_transform is not None
            }
        })

        # Add parameter-specific statistics
        if 'parameter_mappings' in self.metadata_format:
            param_stats = {}
            for param_name in self.heads_config.keys():
                param_info = self.get_parameter_info(param_name)
                param_values = [labels[param_name] for _, labels in self.samples if param_name in labels]

                param_stats[param_name] = {
                    'description': param_info.get('description', ''),
                    'min_quantized': min(param_values) if param_values else None,
                    'max_quantized': max(param_values) if param_values else None,
                    'min_actual': param_info.get('min', 0),
                    'max_actual': param_info.get('max', 255),
                    'unique_values': len(set(param_values)) if param_values else 0,
                    'total_samples': len(param_values)
                }

            stats['parameter_statistics'] = param_stats

        return stats


def create_vimh_datasets(
    data_dir: str,
    transform: Optional[callable] = None,
    target_transform: Optional[callable] = None,
    target_width: float = 0.0
) -> Tuple[VIMHDataset, VIMHDataset]:
    """Create train and test VIMH datasets.

    :param data_dir: Directory containing the dataset files
    :param transform: Optional transform to apply to images
    :param target_transform: Optional transform to apply to labels
    :param target_width: Standard deviation for soft targets (0.0 = hard targets)
    :return: Tuple of (train_dataset, test_dataset)
    """
    train_dataset = VIMHDataset(
        data_dir,
        train=True,
        transform=transform,
        target_transform=target_transform,
        target_width=target_width
    )

    test_dataset = VIMHDataset(
        data_dir,
        train=False,
        transform=transform,
        target_transform=target_transform,
        target_width=target_width
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test with the example dataset
    import sys
    from pathlib import Path

    # Add src to path for imports
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Test dataset loading
    example_data_dir = "data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p"

    try:
        print("Testing VIMH dataset loading...")

        # Load training dataset
        train_dataset = VIMHDataset(example_data_dir, train=True)
        print(f"✓ Loaded training dataset: {len(train_dataset)} samples")

        # Load test dataset
        test_dataset = VIMHDataset(example_data_dir, train=False)
        print(f"✓ Loaded test dataset: {len(test_dataset)} samples")

        # Test sample access
        sample_image, sample_labels = train_dataset[0]
        print(f"✓ Sample image shape: {sample_image.shape}")
        print(f"✓ Sample labels: {sample_labels}")

        # Print dataset info
        print(f"✓ Heads configuration: {train_dataset.get_heads_config()}")
        print(f"✓ Image shape: {train_dataset.get_image_shape()}")

        # Print statistics
        stats = train_dataset.get_dataset_statistics()
        print(f"✓ Dataset statistics available with {len(stats)} fields")

        print("\nVIMH dataset implementation successful!")

    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
