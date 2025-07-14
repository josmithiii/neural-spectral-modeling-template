import json
import struct
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
import numpy as np
from .multihead_dataset_base import MultiheadDatasetBase


class GenericMultiheadDataset(MultiheadDatasetBase):
    """Generic multihead dataset for arbitrary formats.

    This class provides a flexible implementation for loading multihead datasets
    with various binary formats. It supports auto-detection of format specifications
    and can handle custom metadata structures.
    """

    def __init__(
        self,
        data_path: str,
        format_config: Optional[Dict[str, Any]] = None,
        auto_detect: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        """Initialize generic multihead dataset.

        :param data_path: Path to dataset file or directory
        :param format_config: Optional format configuration dictionary
        :param auto_detect: Whether to auto-detect format if config not provided
        :param transform: Optional transform to apply to images
        :param target_transform: Optional transform to apply to labels
        """
        self.transform = transform
        self.target_transform = target_transform
        self.auto_detect = auto_detect

        # Load or detect format configuration
        if format_config is None and auto_detect:
            format_config = self._auto_detect_format(data_path)
        elif format_config is None:
            # Use default generic format
            format_config = self._get_default_format()

        # Validate configuration
        self._validate_config(format_config)

        # Initialize base class
        super().__init__(data_path, format_config)

    def _auto_detect_format(self, data_path: str) -> Dict[str, Any]:
        """Auto-detect dataset format from files and structure.

        :param data_path: Path to dataset
        :return: Detected format configuration
        """
        data_path = Path(data_path)

        # Look for metadata files
        metadata_candidates = [
            'cifar100mh_dataset_info.json',
            'dataset_info.json',
            'metadata.json',
            'format.json'
        ]

        if data_path.is_dir():
            # Check for metadata files in directory
            for candidate in metadata_candidates:
                metadata_file = data_path / candidate
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            config = json.load(f)
                        print(f"Auto-detected format from {metadata_file}")

                        # Ensure required fields are present
                        if 'label_encoding' not in config:
                            config['label_encoding'] = {
                                'format': '[N] [param_id] [param_val] ...',
                                'N_range': [0, 255],
                                'param_id_range': [0, 255],
                                'param_val_range': [0, 255]
                            }

                        return config
                    except (json.JSONDecodeError, IOError):
                        continue

            # Check for standard dataset files
            if (data_path / 'train_batch').exists() or (data_path / 'test_batch').exists():
                # Assume CIFAR-100-MH format
                return self._get_cifar100mh_format()

        else:
            # Single file - try to detect format from content
            return self._detect_from_content(data_path)

        # Default fallback
        print("Warning: Could not auto-detect format, using default")
        return self._get_default_format()

    def _detect_from_content(self, file_path: Path) -> Dict[str, Any]:
        """Detect format from file content analysis.

        :param file_path: Path to dataset file
        :return: Detected format configuration
        """
        try:
            # Try pickle format first
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'data' in data and 'labels' in data:
                # Analyze image and label data
                images = data['data']
                labels = data['labels']

                if not images or not labels:
                    return self._get_default_format()

                # Get image dimensions from actual data
                first_image = images[0]
                image_size = len(first_image)

                # Try to infer dimensions from common image sizes
                if image_size == 784:  # 28x28x1 (MNIST-like)
                    height, width, channels = 28, 28, 1
                elif image_size == 3072:  # 32x32x3 (CIFAR-like)
                    height, width, channels = 32, 32, 3
                else:
                    # Default assumption for generic format
                    height, width, channels = 32, 32, 3

                # Analyze label structure
                first_label = labels[0] if labels else []

                # Check if it's the standard multihead format [N, param_id, param_val, ...]
                if len(first_label) >= 3:
                    num_heads = first_label[0]
                    expected_length = 1 + (num_heads * 2)

                    if len(first_label) >= expected_length:
                        return {
                            'format': 'auto-detected',
                            'label_encoding': {
                                'format': '[N] [param_id] [param_val] ...',
                                'detected_dimensions': [height, width, channels],
                                'detected_heads': num_heads
                            },
                            'parameter_names': [f'param_{i}' for i in range(num_heads)],
                            'image_size': f'{height}x{width}x{channels}'
                        }

        except Exception:
            pass

        # Binary format detection could be added here
        return self._get_default_format()

    def _get_default_format(self) -> Dict[str, Any]:
        """Get default format configuration.

        :return: Default format configuration
        """
        return {
            'format': 'generic-multihead',
            'version': '1.0',
            'label_encoding': {
                'format': '[height] [width] [channels] [N] [param1_id] [param1_val] ...',
                'N_range': [0, 255],
                'param_id_range': [0, 255],
                'param_val_range': [0, 255]
            },
            'parameter_names': ['param_0', 'param_1'],
            'default_image_size': '32x32x3'
        }

    def _get_cifar100mh_format(self) -> Dict[str, Any]:
        """Get CIFAR-100-MH format configuration.

        :return: CIFAR-100-MH format configuration
        """
        return {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'label_encoding': {
                'format': '[height] [width] [channels] [N] [param1_id] [param1_val] ...',
                'N_range': [0, 255],
                'param_id_range': [0, 255],
                'param_val_range': [0, 255]
            },
            'parameter_names': ['param_0', 'param_1'],
            'image_size': '32x32x3'
        }

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate format configuration.

        :param config: Configuration dictionary
        :return: True if valid
        :raises: ValueError if invalid
        """
        required_fields = ['format', 'label_encoding']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in format config: {field}")

        # Validate label encoding format
        label_encoding = config['label_encoding']
        if 'format' not in label_encoding:
            raise ValueError("Missing 'format' in label_encoding")

        return True

    def _get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample.

        :param idx: Sample index
        :return: Metadata dictionary
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")

        _, labels = self.samples[idx]

        metadata = {
            'sample_index': idx,
            'labels': labels.copy(),
            'image_shape': self.image_shape,
            'format': self.metadata_format.get('format', 'unknown')
        }

        return metadata

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Get a sample from the dataset.

        :param idx: Sample index
        :return: Tuple of (image_tensor, labels_dict)
        """
        image, labels = super().__getitem__(idx)

        # Apply transforms if specified
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return image, labels

    def add_custom_parameter_mapping(self, param_id: int, param_name: str, description: str = "") -> None:
        """Add custom parameter mapping.

        :param param_id: Parameter ID
        :param param_name: Parameter name
        :param description: Parameter description
        """
        if 'parameter_mappings' not in self.metadata_format:
            self.metadata_format['parameter_mappings'] = {}

        self.metadata_format['parameter_mappings'][param_name] = {
            'id': param_id,
            'description': description
        }

        # Update parameter names list
        if 'parameter_names' not in self.metadata_format:
            self.metadata_format['parameter_names'] = []

        while len(self.metadata_format['parameter_names']) <= param_id:
            self.metadata_format['parameter_names'].append(f'param_{len(self.metadata_format["parameter_names"])}')

        self.metadata_format['parameter_names'][param_id] = param_name

    def save_format_config(self, output_path: str) -> None:
        """Save current format configuration to file.

        :param output_path: Path to save configuration
        """
        config_to_save = self.metadata_format.copy()
        config_to_save['dataset_info'] = self.get_dataset_info()

        with open(output_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)

    def get_format_summary(self) -> Dict[str, Any]:
        """Get summary of the detected/configured format.

        :return: Format summary dictionary
        """
        return {
            'format_name': self.metadata_format.get('format', 'unknown'),
            'version': self.metadata_format.get('version', 'unknown'),
            'label_encoding': self.metadata_format.get('label_encoding', {}),
            'parameter_names': self.metadata_format.get('parameter_names', []),
            'auto_detected': self.auto_detect,
            'num_samples': len(self.samples),
            'image_shape': self.image_shape,
            'heads_config': self.heads_config
        }


class MultiheadDatasetFactory:
    """Factory class for creating multihead datasets."""

    @staticmethod
    def create_dataset(
        data_path: str,
        dataset_type: str = 'auto',
        **kwargs
    ) -> MultiheadDatasetBase:
        """Create a multihead dataset of the specified type.

        :param data_path: Path to dataset
        :param dataset_type: Type of dataset ('auto', 'cifar100mh', 'generic')
        :param kwargs: Additional arguments for dataset constructor
        :return: Dataset instance
        """
        if dataset_type == 'auto':
            # Auto-detect dataset type
            data_path_obj = Path(data_path)

            # Check for CIFAR-100-MH indicators
            if data_path_obj.is_dir():
                metadata_file = data_path_obj / 'cifar100mh_dataset_info.json'
                if metadata_file.exists():
                    dataset_type = 'cifar100mh'
                else:
                    dataset_type = 'generic'
            else:
                dataset_type = 'generic'

        if dataset_type == 'cifar100mh':
            from .cifar100mh_dataset import CIFAR100MHDataset
            return CIFAR100MHDataset(data_path, **kwargs)
        elif dataset_type == 'generic':
            return GenericMultiheadDataset(data_path, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def list_supported_formats() -> List[str]:
        """List supported dataset formats.

        :return: List of supported format names
        """
        return ['cifar100mh', 'generic']

    @staticmethod
    def get_format_info(format_name: str) -> Dict[str, Any]:
        """Get information about a specific format.

        :param format_name: Name of the format
        :return: Format information dictionary
        """
        formats = {
            'cifar100mh': {
                'name': 'CIFAR-100-MH',
                'description': 'CIFAR-100 Multihead format with embedded metadata',
                'label_structure': '[height] [width] [channels] [N] [param_id] [param_val] ...',
                'file_format': 'pickle',
                'metadata_file': 'cifar100mh_dataset_info.json'
            },
            'generic': {
                'name': 'Generic Multihead',
                'description': 'Generic multihead format with auto-detection',
                'label_structure': 'Auto-detected or configurable',
                'file_format': 'pickle or binary',
                'metadata_file': 'auto-detected or configurable'
            }
        }

        return formats.get(format_name, {'error': f'Unknown format: {format_name}'})


if __name__ == "__main__":
    # Test the generic dataset and factory
    print("Testing GenericMultiheadDataset and factory...")

    # Test factory
    factory = MultiheadDatasetFactory()
    print(f"Supported formats: {factory.list_supported_formats()}")

    # Test format info
    for format_name in factory.list_supported_formats():
        info = factory.get_format_info(format_name)
        print(f"Format '{format_name}': {info['description']}")

    print("Generic multihead dataset implementation complete!")
