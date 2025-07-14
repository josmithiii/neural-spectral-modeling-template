import pytest
import torch
import numpy as np
import pickle
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from src.data.multihead_dataset_base import MultiheadDatasetBase
from src.data.cifar100mh_dataset import CIFAR100MHDataset, create_cifar100mh_datasets
from src.data.generic_multihead_dataset import GenericMultiheadDataset, MultiheadDatasetFactory


class MockMultiheadDataset(MultiheadDatasetBase):
    """Mock implementation for testing the base class."""

    def _get_sample_metadata(self, idx):
        return {'mock_metadata': True, 'index': idx}


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_cifar100mh_data():
    """Create mock CIFAR-100-MH data for testing."""
    # Create image data (32x32x3 = 3072 pixels)
    images = []
    labels = []

    for i in range(10):  # 10 samples
        # Random image data
        image_data = np.random.randint(0, 256, size=3072, dtype=np.uint8).tolist()
        images.append(image_data)

        # Label format: [N, param1_id, param1_val, param2_id, param2_val]
        label_data = [2, 0, i % 10, 1, (i * 2) % 20]
        labels.append(label_data)

    return {
        'data': images,
        'labels': labels
    }


@pytest.fixture
def mock_metadata():
    """Create mock metadata configuration."""
    return {
        'format': 'CIFAR-100-MH',
        'version': '1.0',
        'dataset_name': 'test_dataset',
        'n_samples': 10,
        'train_samples': 8,
        'test_samples': 2,
        'image_size': '32x32x3',
        'channels': 3,
        'varying_parameters': 2,
        'parameter_names': ['note_number', 'note_velocity'],
        'label_encoding': {
            'format': '[height] [width] [channels] [N] [param1_id] [param1_val] [param2_id] [param2_val]',
            'N_range': [0, 255],
            'param_id_range': [0, 255],
            'param_val_range': [0, 255]
        },
        'parameter_mappings': {
            'note_number': {
                'min': 0,
                'max': 10,
                'scale': 'linear',
                'description': 'Test note number parameter'
            },
            'note_velocity': {
                'min': 0,
                'max': 20,
                'scale': 'linear',
                'description': 'Test note velocity parameter'
            }
        }
    }


def create_test_dataset_files(temp_dir: Path, mock_data: dict, mock_metadata: dict):
    """Create test dataset files in temporary directory."""
    # Create data files
    train_file = temp_dir / 'train_batch'
    test_file = temp_dir / 'test_batch'
    metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

    # Split data into train/test using correct label key
    train_data = {
        'data': mock_data['data'][:8],
        'cifar100mh_labels': mock_data['labels'][:8]
    }
    test_data = {
        'data': mock_data['data'][8:],
        'cifar100mh_labels': mock_data['labels'][8:]
    }

    # Save pickle files
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(mock_metadata, f, indent=2)

    return train_file, test_file, metadata_file


class TestMultiheadDatasetBase:
    """Test cases for MultiheadDatasetBase class."""

    def test_init_with_nonexistent_path(self):
        """Test initialization with non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            MockMultiheadDataset("/nonexistent/path")

    def test_parse_label_metadata_valid(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test parsing of valid label metadata."""
        train_file, _, _ = create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        # Create dataset and test metadata parsing
        dataset = MockMultiheadDataset(str(train_file), mock_metadata)

        # Test metadata parsing directly
        label_data = [2, 0, 5, 1, 10]
        metadata, labels = dataset._parse_label_metadata(label_data)

        assert metadata['height'] == 32  # From metadata
        assert metadata['width'] == 32   # From metadata
        assert metadata['channels'] == 3 # From metadata
        assert metadata['num_heads'] == 2
        assert labels['note_number'] == 5
        assert labels['note_velocity'] == 10

    def test_parse_label_metadata_insufficient_data(self, mock_metadata):
        """Test parsing with insufficient label data."""
        dataset = MockMultiheadDataset.__new__(MockMultiheadDataset)
        dataset.metadata_format = mock_metadata

        # Too short label data
        with pytest.raises(ValueError, match="Label data too short"):
            dataset._parse_label_metadata([])

        # Insufficient data for number of heads
        with pytest.raises(ValueError, match="insufficient for .* heads"):
            dataset._parse_label_metadata([2, 0])  # Claims 2 heads but only 1 param

    def test_reconstruct_image(self, mock_metadata):
        """Test image reconstruction from flattened data."""
        dataset = MockMultiheadDataset.__new__(MockMultiheadDataset)
        dataset.metadata_format = mock_metadata

        # Create test image data (2x2x3 = 12 pixels)
        image_data = list(range(12))
        metadata = {'height': 2, 'width': 2, 'channels': 3}

        image_tensor = dataset._reconstruct_image(image_data, metadata)

        assert image_tensor.shape == (3, 2, 2)  # CHW format
        assert image_tensor.dtype == torch.float32
        assert torch.all(image_tensor >= 0) and torch.all(image_tensor <= 1)  # Normalized

    def test_validate_format_inconsistent_shapes(self, temp_dir, mock_metadata):
        """Test format validation with inconsistent image shapes."""
        # Create inconsistent data
        images = [
            np.random.randint(0, 256, size=3072).tolist(),  # 32x32x3
            np.random.randint(0, 256, size=784).tolist(),   # 28x28x1 (different!)
        ]
        labels = [
            [1, 0, 5],
            [1, 0, 3],
        ]

        mock_data = {'data': images, 'labels': labels}
        train_file = temp_dir / 'train_batch'

        with open(train_file, 'wb') as f:
            pickle.dump(mock_data, f)

        with pytest.raises(ValueError, match="inconsistent image shape"):
            MockMultiheadDataset(str(train_file), mock_metadata)


class TestCIFAR100MHDataset:
    """Test cases for CIFAR100MHDataset class."""

    def test_init_train_dataset(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test initialization of training dataset."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = CIFAR100MHDataset(str(temp_dir), train=True)

        assert len(dataset) == 8  # Training samples
        assert dataset.train is True
        assert dataset.get_image_shape() == (3, 32, 32)

        heads_config = dataset.get_heads_config()
        assert 'note_number' in heads_config
        assert 'note_velocity' in heads_config

    def test_init_test_dataset(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test initialization of test dataset."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = CIFAR100MHDataset(str(temp_dir), train=False)

        assert len(dataset) == 2  # Test samples
        assert dataset.train is False

    def test_getitem(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test getting individual samples."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = CIFAR100MHDataset(str(temp_dir), train=True)

        image, labels = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert isinstance(labels, dict)
        assert 'note_number' in labels
        assert 'note_velocity' in labels

    def test_getitem_with_transforms(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test getting samples with transforms applied."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        # Simple transform that doubles the image values
        image_transform = lambda x: x * 2
        target_transform = lambda x: {k: v * 2 for k, v in x.items()}

        dataset = CIFAR100MHDataset(
            str(temp_dir),
            train=True,
            transform=image_transform,
            target_transform=target_transform
        )

        image, labels = dataset[0]

        # Check that transforms were applied
        assert torch.max(image) > 1.0  # Should be > 1 due to doubling
        assert all(v % 2 == 0 for v in labels.values())  # All labels should be even

    def test_get_parameter_info(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test getting parameter information."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = CIFAR100MHDataset(str(temp_dir), train=True)

        param_info = dataset.get_parameter_info('note_number')
        assert 'description' in param_info
        assert param_info['description'] == 'Test note number parameter'

        # Test non-existent parameter
        unknown_info = dataset.get_parameter_info('unknown_param')
        assert 'Parameter not found' in unknown_info['description']

    def test_get_class_distribution(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test getting class distribution."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = CIFAR100MHDataset(str(temp_dir), train=True)

        distribution = dataset.get_class_distribution()

        assert 'note_number' in distribution
        assert 'note_velocity' in distribution
        assert isinstance(distribution['note_number'], dict)
        assert len(distribution['note_number']) > 0  # Should have some class counts

    def test_get_dataset_statistics(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test getting comprehensive dataset statistics."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = CIFAR100MHDataset(str(temp_dir), train=True)

        stats = dataset.get_dataset_statistics()

        assert 'num_samples' in stats
        assert 'image_shape' in stats
        assert 'heads_config' in stats
        assert 'train_mode' in stats
        assert 'class_distribution' in stats
        assert 'parameter_statistics' in stats

        # Check parameter statistics
        param_stats = stats['parameter_statistics']
        assert 'note_number' in param_stats
        assert 'min_value' in param_stats['note_number']
        assert 'max_value' in param_stats['note_number']

    def test_missing_metadata_file(self, temp_dir, mock_cifar100mh_data):
        """Test handling of missing metadata file."""
        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(mock_cifar100mh_data, f)

        # Should work with default metadata
        dataset = CIFAR100MHDataset(str(temp_dir), train=True)
        assert len(dataset) > 0

    def test_create_cifar100mh_datasets(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test factory function for creating train/test datasets."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        train_dataset, test_dataset = create_cifar100mh_datasets(str(temp_dir))

        assert isinstance(train_dataset, CIFAR100MHDataset)
        assert isinstance(test_dataset, CIFAR100MHDataset)
        assert train_dataset.train is True
        assert test_dataset.train is False
        assert len(train_dataset) == 8
        assert len(test_dataset) == 2


class TestGenericMultiheadDataset:
    """Test cases for GenericMultiheadDataset class."""

    def test_init_with_auto_detect(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test initialization with auto-detection."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = GenericMultiheadDataset(str(temp_dir), auto_detect=True)

        assert len(dataset) > 0
        assert dataset.metadata_format['format'] == 'CIFAR-100-MH'

    def test_init_with_custom_config(self, temp_dir, mock_cifar100mh_data):
        """Test initialization with custom configuration."""
        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(mock_cifar100mh_data, f)

        custom_config = {
            'format': 'custom-format',
            'label_encoding': {
                'format': '[height] [width] [channels] [N] [param_id] [param_val] ...'
            },
            'parameter_names': ['custom_param1', 'custom_param2']
        }

        dataset = GenericMultiheadDataset(str(train_file), format_config=custom_config)

        assert dataset.metadata_format['format'] == 'custom-format'
        assert len(dataset) > 0

    def test_auto_detect_format_from_content(self, temp_dir, mock_cifar100mh_data):
        """Test auto-detection from file content."""
        train_file = temp_dir / 'data.pkl'
        with open(train_file, 'wb') as f:
            pickle.dump(mock_cifar100mh_data, f)

        dataset = GenericMultiheadDataset(str(train_file), auto_detect=True)

        assert len(dataset) > 0
        detected_format = dataset.metadata_format
        assert 'label_encoding' in detected_format

    def test_add_custom_parameter_mapping(self, temp_dir, mock_cifar100mh_data):
        """Test adding custom parameter mappings."""
        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(mock_cifar100mh_data, f)

        dataset = GenericMultiheadDataset(str(train_file))

        dataset.add_custom_parameter_mapping(0, 'custom_param', 'Custom parameter description')

        assert 'parameter_mappings' in dataset.metadata_format
        assert 'custom_param' in dataset.metadata_format['parameter_mappings']
        assert dataset.metadata_format['parameter_names'][0] == 'custom_param'

    def test_save_format_config(self, temp_dir, mock_cifar100mh_data):
        """Test saving format configuration to file."""
        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(mock_cifar100mh_data, f)

        dataset = GenericMultiheadDataset(str(train_file))

        config_file = temp_dir / 'saved_config.json'
        dataset.save_format_config(str(config_file))

        assert config_file.exists()

        # Load and verify saved config
        with open(config_file, 'r') as f:
            saved_config = json.load(f)

        assert 'format' in saved_config
        assert 'dataset_info' in saved_config

    def test_get_format_summary(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test getting format summary."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = GenericMultiheadDataset(str(temp_dir))

        summary = dataset.get_format_summary()

        assert 'format_name' in summary
        assert 'version' in summary
        assert 'label_encoding' in summary
        assert 'parameter_names' in summary
        assert 'num_samples' in summary
        assert 'image_shape' in summary
        assert 'heads_config' in summary


class TestMultiheadDatasetFactory:
    """Test cases for MultiheadDatasetFactory class."""

    def test_create_dataset_auto_cifar100mh(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test auto-creation of CIFAR-100-MH dataset."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        factory = MultiheadDatasetFactory()
        dataset = factory.create_dataset(str(temp_dir), dataset_type='auto')

        assert isinstance(dataset, CIFAR100MHDataset)
        assert len(dataset) > 0

    def test_create_dataset_explicit_type(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test explicit dataset type creation."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        factory = MultiheadDatasetFactory()

        # Test CIFAR-100-MH
        cifar_dataset = factory.create_dataset(str(temp_dir), dataset_type='cifar100mh')
        assert isinstance(cifar_dataset, CIFAR100MHDataset)

        # Test generic
        generic_dataset = factory.create_dataset(str(temp_dir), dataset_type='generic')
        assert isinstance(generic_dataset, GenericMultiheadDataset)

    def test_create_dataset_invalid_type(self, temp_dir):
        """Test creation with invalid dataset type."""
        factory = MultiheadDatasetFactory()

        with pytest.raises(ValueError, match="Unknown dataset type"):
            factory.create_dataset(str(temp_dir), dataset_type='invalid_type')

    def test_list_supported_formats(self):
        """Test listing supported formats."""
        factory = MultiheadDatasetFactory()
        formats = factory.list_supported_formats()

        assert isinstance(formats, list)
        assert 'cifar100mh' in formats
        assert 'generic' in formats

    def test_get_format_info(self):
        """Test getting format information."""
        factory = MultiheadDatasetFactory()

        # Test valid format
        cifar_info = factory.get_format_info('cifar100mh')
        assert 'name' in cifar_info
        assert 'description' in cifar_info
        assert cifar_info['name'] == 'CIFAR-100-MH'

        # Test invalid format
        invalid_info = factory.get_format_info('invalid')
        assert 'error' in invalid_info


class TestIntegration:
    """Integration tests for multihead dataset functionality."""

    def test_end_to_end_dataset_loading(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test complete end-to-end dataset loading workflow."""
        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        # Test factory auto-detection
        factory = MultiheadDatasetFactory()
        dataset = factory.create_dataset(str(temp_dir))

        # Test dataset functionality
        assert len(dataset) > 0

        # Test sample access
        image, labels = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(labels, dict)

        # Test metadata access
        heads_config = dataset.get_heads_config()
        assert len(heads_config) > 0

        # Test iteration
        sample_count = 0
        for sample_image, sample_labels in dataset:
            assert isinstance(sample_image, torch.Tensor)
            assert isinstance(sample_labels, dict)
            sample_count += 1
            if sample_count >= 3:  # Test first few samples
                break

        assert sample_count >= 3

    def test_batch_loading_compatibility(self, temp_dir, mock_cifar100mh_data, mock_metadata):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        create_test_dataset_files(temp_dir, mock_cifar100mh_data, mock_metadata)

        dataset = CIFAR100MHDataset(str(temp_dir), train=True)

        # Test DataLoader creation
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Test batch loading
        batch_images, batch_labels = next(iter(dataloader))

        assert isinstance(batch_images, torch.Tensor)
        assert batch_images.shape[0] == 2  # Batch size
        assert isinstance(batch_labels, dict)

        # Check that each label type has correct batch dimension
        for label_name, label_tensor in batch_labels.items():
            assert isinstance(label_tensor, torch.Tensor)
            assert label_tensor.shape[0] == 2  # Batch size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
