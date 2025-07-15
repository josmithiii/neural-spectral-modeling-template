import struct
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
from torch.utils.data import Dataset
import numpy as np


class MultiheadDatasetBase(Dataset, ABC):
    """Base class for multihead datasets with arbitrary format support.

    This class provides the foundation for loading datasets with multiple classification
    heads and embedded metadata. It supports various binary formats with self-describing
    metadata structures.
    """

    def __init__(self, data_path: str, metadata_format: Optional[Dict[str, Any]] = None):
        """Initialize the multihead dataset.

        :param data_path: Path to the dataset file or directory
        :param metadata_format: Optional format specification for parsing
        """
        self.data_path = Path(data_path)
        self.metadata_format = metadata_format or {}
        self.samples = []
        self.heads_config = {}
        self.image_shape = None

        # Load and validate the dataset
        self._load_dataset()
        self._validate_format()

    def _load_dataset(self) -> None:
        """Load the dataset from the specified path."""
        if self.data_path.is_file():
            self._load_from_file()
        elif self.data_path.is_dir():
            self._load_from_directory()
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

    def _load_from_file(self) -> None:
        """Load dataset from a single file."""
        if self.data_path.suffix == '.pkl' or 'batch' in self.data_path.name:
            # Handle pickle format (like CIFAR batches)
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            self._parse_pickle_data(data)
        else:
            # Handle binary format
            with open(self.data_path, 'rb') as f:
                self._parse_binary_data(f.read())

    def _load_from_directory(self) -> None:
        """Load dataset from a directory with multiple files."""
        # Look for standard files
        train_file = self.data_path / 'train_batch'
        test_file = self.data_path / 'test_batch'
        metadata_file = self.data_path / 'cifar100mh_dataset_info.json'

        # Load metadata if available
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                self.metadata_format = json.load(f)

        # Load training data
        if train_file.exists():
            with open(train_file, 'rb') as f:
                data = pickle.load(f)
            self._parse_pickle_data(data)

    def _parse_pickle_data(self, data: Dict[str, Any]) -> None:
        """Parse data from pickle format.

        :param data: Dictionary containing 'data' and label keys
        """
        if 'data' not in data:
            raise ValueError("Pickle data must contain 'data' key")

        # Look for label keys in common formats
        label_key = None
        for possible_key in ['labels', 'vimh_labels', 'cifar100mh_labels', 'multihead_labels']:
            if possible_key in data:
                label_key = possible_key
                break

        if label_key is None:
            raise ValueError("Pickle data must contain labels key ('labels', 'vimh_labels', 'cifar100mh_labels', or 'multihead_labels')")

        images = data['data']  # Shape: (n_samples, height*width*channels)
        labels = data[label_key]  # Shape: (n_samples, label_bytes)

        # For VIMH format, extract image dimensions from pickle data
        if self.metadata_format.get('format') == 'VIMH':
            if 'height' in data and 'width' in data and 'channels' in data:
                self.metadata_format['height'] = data['height']
                self.metadata_format['width'] = data['width']
                self.metadata_format['channels'] = data['channels']

        # Parse each sample
        for i in range(len(images)):
            image_data = images[i]
            label_data = labels[i]

            # Parse metadata and labels from label_data
            metadata, label_dict = self._parse_label_metadata(label_data)

            # Reconstruct image tensor
            image_tensor = self._reconstruct_image(image_data, metadata)

            self.samples.append((image_tensor, label_dict))

        # Set image shape and heads config from first sample
        if self.samples:
            self.image_shape = self.samples[0][0].shape

            # Calculate heads config by finding max value for each head across all samples
            self.heads_config = {}
            if self.samples[0][1]:  # If we have labels
                # Collect all parameter names that appear in any sample
                all_param_names = set()
                for _, labels in self.samples:
                    all_param_names.update(labels.keys())

                # Now calculate heads config for all parameters
                for param_name in all_param_names:
                    # Find max value for this parameter across all samples that have it
                    values = [sample[1][param_name] for sample in self.samples if param_name in sample[1]]
                    if values:
                        max_val = max(values)
                        self.heads_config[param_name] = max_val + 1  # +1 for number of classes

    def _parse_label_metadata(self, label_data: Union[List[int], np.ndarray]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Parse metadata and labels from label data bytes.

        Format depends on the dataset format:
        - VIMH: [height] [width] [channels] [N] [param1_id] [param1_val] [param2_id] [param2_val] ...
        - CIFAR-100-MH: [N] [param1_id] [param1_val] [param2_id] [param2_val] ...

        :param label_data: Array/list of label bytes
        :return: Tuple of (metadata_dict, labels_dict)
        """
        if isinstance(label_data, np.ndarray):
            label_data = label_data.tolist()

        if len(label_data) < 1:
            raise ValueError(f"Label data too short: {len(label_data)}, expected at least 1 byte")

        # Check if this is VIMH format
        is_vimh = self.metadata_format.get('format') == 'VIMH'

        if is_vimh:
            # VIMH format: [N] [param1_id] [param1_val] [param2_id] [param2_val] ...
            # Image dimensions are in the pickle data metadata, not in the label data
            num_heads = label_data[0]
            labels_start = 1

            # Get dimensions from metadata - they should be already set from pickle data
            height = self.metadata_format.get('height', 32)
            width = self.metadata_format.get('width', 32)
            channels = self.metadata_format.get('channels', 3)
        else:
            # CIFAR-100-MH format: [N] [param1_id] [param1_val] ...
            num_heads = label_data[0]
            labels_start = 1

            # Get image dimensions from metadata if available, otherwise infer from data shape
            if 'image_size' in self.metadata_format:
                size_str = self.metadata_format['image_size']
                if 'x' in size_str:
                    parts = size_str.split('x')
                    height, width, channels = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    height, width, channels = 32, 32, 3  # Default
            else:
                height, width, channels = 32, 32, 3  # Default for CIFAR format

        metadata = {
            'height': height,
            'width': width,
            'channels': channels,
            'num_heads': num_heads
        }

        # Parse parameter labels
        expected_length = labels_start + (num_heads * 2)  # labels_start + N pairs of (id, val)
        if len(label_data) < expected_length:
            raise ValueError(f"Label data length {len(label_data)} insufficient for {num_heads} heads, expected {expected_length}")

        labels_dict = {}
        for i in range(num_heads):
            param_id = label_data[labels_start + i * 2]
            param_val = label_data[labels_start + i * 2 + 1]

            # Use parameter names from metadata if available
            param_name = self._get_parameter_name(param_id)
            labels_dict[param_name] = param_val

        return metadata, labels_dict

    def _get_parameter_name(self, param_id: int) -> str:
        """Get parameter name from ID using metadata format.

        :param param_id: Parameter ID
        :return: Parameter name
        """
        if 'parameter_names' in self.metadata_format and param_id < len(self.metadata_format['parameter_names']):
            return self.metadata_format['parameter_names'][param_id]
        else:
            return f'param_{param_id}'

    def _reconstruct_image(self, image_data: Union[List[int], np.ndarray], metadata: Dict[str, int]) -> torch.Tensor:
        """Reconstruct image tensor from flattened data.

        :param image_data: Flattened image data
        :param metadata: Metadata containing image dimensions
        :return: Image tensor of shape (channels, height, width)
        """
        if isinstance(image_data, list):
            image_data = np.array(image_data)

        height = metadata['height']
        width = metadata['width']
        channels = metadata['channels']

        expected_size = height * width * channels
        actual_size = image_data.size

        if actual_size != expected_size:
            raise ValueError(f"Sample has inconsistent image shape: expected {height}x{width}x{channels} ({expected_size} pixels), got {actual_size} pixels")

        # Reshape to (height, width, channels) then convert to (channels, height, width)
        image = image_data.reshape(height, width, channels)
        image = np.transpose(image, (2, 0, 1))  # CHW format

        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(image).float() / 255.0

        return image_tensor

    def _parse_binary_data(self, data: bytes) -> None:
        """Parse dataset from raw binary data.

        :param data: Raw binary data
        """
        # This is a placeholder for direct binary format parsing
        # Implementation depends on specific binary format
        raise NotImplementedError("Direct binary parsing not yet implemented")

    def _validate_format(self) -> bool:
        """Validate the loaded dataset format.

        :return: True if format is valid
        :raises: ValueError if format is invalid
        """
        if not self.samples:
            raise ValueError("No samples loaded")

        # Check that all samples have consistent structure
        first_sample = self.samples[0]
        first_image_shape = first_sample[0].shape
        first_num_heads = len(first_sample[1])

        for i, (image, labels) in enumerate(self.samples):
            if image.shape != first_image_shape:
                raise ValueError(f"Sample {i} has inconsistent image shape: {image.shape} vs {first_image_shape}")

            # Check that number of heads is consistent
            if len(labels) != first_num_heads:
                raise ValueError(f"Sample {i} has inconsistent label keys: {len(labels)} heads vs {first_num_heads} heads")

        return True

    def get_heads_config(self) -> Dict[str, int]:
        """Get the configuration of classification heads.

        :return: Dictionary mapping head names to number of classes
        """
        return self.heads_config.copy()

    def get_image_shape(self) -> Tuple[int, int, int]:
        """Get the shape of images in the dataset.

        :return: Tuple of (channels, height, width)
        """
        return self.image_shape

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information.

        :return: Dictionary with dataset metadata
        """
        return {
            'num_samples': len(self.samples),
            'image_shape': self.image_shape,
            'heads_config': self.heads_config,
            'metadata_format': self.metadata_format,
            'data_path': str(self.data_path)
        }

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Get a sample from the dataset.

        :param idx: Sample index
        :return: Tuple of (image_tensor, labels_dict)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        return self.samples[idx]

    @abstractmethod
    def _get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample (abstract method).

        :param idx: Sample index
        :return: Metadata dictionary
        """
        pass


if __name__ == "__main__":
    # Basic testing - this will be expanded in unit tests
    print("MultiheadDatasetBase class created successfully")
    print("This is an abstract base class - use CIFAR100MHDataset for concrete implementation")
