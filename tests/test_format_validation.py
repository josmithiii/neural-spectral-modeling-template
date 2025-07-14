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
from src.data.cifar100mh_dataset import CIFAR100MHDataset
from src.data.generic_multihead_dataset import GenericMultiheadDataset


class MockFormatValidator(MultiheadDatasetBase):
    """Mock implementation for testing format validation."""

    def _get_sample_metadata(self, idx):
        return {'mock_metadata': True, 'index': idx}


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestFormatAutodetection:
    """Test format auto-detection functionality."""

    def test_detect_cifar100mh_format(self, temp_dir):
        """Test auto-detection of CIFAR-100-MH format."""
        # Create CIFAR-100-MH format data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(5)]
        labels = [[2, 0, i % 10, 1, (i * 2) % 20] for i in range(5)]

        data = {'data': images, 'cifar100mh_labels': labels}

        # Create metadata file
        metadata = {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'parameter_names': ['param_0', 'param_1']
        }

        # Save files
        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Test auto-detection
        dataset = GenericMultiheadDataset(str(temp_dir), auto_detect=True)

        assert dataset.metadata_format['format'] == 'CIFAR-100-MH'
        assert 'param_0' in dataset.get_heads_config()
        assert 'param_1' in dataset.get_heads_config()

    def test_detect_generic_format(self, temp_dir):
        """Test auto-detection of generic format."""
        # Create generic format data
        images = [np.random.randint(0, 256, size=784).tolist() for _ in range(5)]
        labels = [[1, 0, i % 5] for i in range(5)]  # Different format

        data = {'data': images, 'labels': labels}

        # Save file
        data_file = temp_dir / 'data.pkl'
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        # Test auto-detection
        dataset = GenericMultiheadDataset(str(data_file), auto_detect=True)

        assert len(dataset) == 5
        assert dataset.metadata_format is not None

    def test_detect_invalid_format(self, temp_dir):
        """Test detection of invalid format."""
        # Create invalid data
        invalid_data = {'invalid_key': [1, 2, 3]}

        # Save file
        data_file = temp_dir / 'invalid.pkl'
        with open(data_file, 'wb') as f:
            pickle.dump(invalid_data, f)

        # Test that it raises appropriate error
        with pytest.raises(ValueError, match="data.*must contain.*data.*key"):
            GenericMultiheadDataset(str(data_file), auto_detect=True)


class TestDataValidation:
    """Test data validation functionality."""

    def test_validate_consistent_image_dimensions(self, temp_dir):
        """Test validation of consistent image dimensions."""
        # Create data with consistent dimensions
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should not raise error
        dataset = MockFormatValidator(str(train_file))
        assert len(dataset) == 3

    def test_validate_inconsistent_image_dimensions(self, temp_dir):
        """Test validation catches inconsistent image dimensions."""
        # Create data with inconsistent dimensions
        images = [
            np.random.randint(0, 256, size=3072).tolist(),  # 32x32x3
            np.random.randint(0, 256, size=784).tolist(),   # 28x28x1
            np.random.randint(0, 256, size=3072).tolist(),  # 32x32x3
        ]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should raise error
        with pytest.raises(ValueError, match="inconsistent image shape"):
            MockFormatValidator(str(train_file))

    def test_validate_consistent_label_structure(self, temp_dir):
        """Test validation of consistent label structure."""
        # Create data with consistent labels
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]  # All same structure

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should not raise error
        dataset = MockFormatValidator(str(train_file))
        assert len(dataset) == 3

    def test_validate_inconsistent_label_structure(self, temp_dir):
        """Test validation catches inconsistent label structure."""
        # Create data with inconsistent labels
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [
            [2, 0, 1, 1, 2],     # 2 heads
            [1, 0, 1],           # 1 head - inconsistent!
            [2, 0, 1, 1, 2],     # 2 heads
        ]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should raise error
        with pytest.raises(ValueError, match="inconsistent label keys"):
            MockFormatValidator(str(train_file))

    def test_validate_label_data_length(self, temp_dir):
        """Test validation of label data length."""
        # Create data with insufficient label data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [
            [2, 0, 1],           # Claims 2 heads but only has 1 param
            [2, 0, 1, 1],        # Claims 2 heads but incomplete
            [2, 0, 1, 1, 2],     # Complete
        ]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should raise error
        with pytest.raises(ValueError, match="insufficient for.*heads"):
            MockFormatValidator(str(train_file))

    def test_validate_empty_dataset(self, temp_dir):
        """Test validation of empty dataset."""
        # Create empty data
        data = {'data': [], 'cifar100mh_labels': []}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should raise error
        with pytest.raises(ValueError, match="No samples loaded"):
            MockFormatValidator(str(train_file))


class TestMetadataValidation:
    """Test metadata validation functionality."""

    def test_validate_metadata_format_version(self, temp_dir):
        """Test validation of metadata format version."""
        # Create data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        # Create metadata with wrong format
        metadata = {
            'format': 'WRONG-FORMAT',
            'version': '1.0',
            'parameter_names': ['param_0', 'param_1']
        }

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should show warning but not fail
        dataset = CIFAR100MHDataset(str(temp_dir), train=True)
        assert len(dataset) == 3

    def test_validate_metadata_sample_count(self, temp_dir):
        """Test validation of metadata sample count."""
        # Create data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        # Create metadata with wrong sample count
        metadata = {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'train_samples': 5,  # Wrong count
            'parameter_names': ['param_0', 'param_1']
        }

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should show warning but not fail
        dataset = CIFAR100MHDataset(str(temp_dir), train=True)
        assert len(dataset) == 3

    def test_validate_metadata_parameter_names(self, temp_dir):
        """Test validation of parameter names in metadata."""
        # Create data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        # Create metadata with parameter names
        metadata = {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'parameter_names': ['note_number', 'note_velocity']
        }

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should use parameter names from metadata
        dataset = CIFAR100MHDataset(str(temp_dir), train=True)
        heads_config = dataset.get_heads_config()

        assert 'note_number' in heads_config
        assert 'note_velocity' in heads_config

    def test_validate_corrupted_metadata(self, temp_dir):
        """Test handling of corrupted metadata file."""
        # Create data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Create corrupted metadata
        with open(metadata_file, 'w') as f:
            f.write('invalid json content {')

        # Should raise error
        with pytest.raises(ValueError, match="Failed to load metadata"):
            CIFAR100MHDataset(str(temp_dir), train=True)


class TestDimensionValidation:
    """Test dimension validation functionality."""

    def test_validate_image_dimension_consistency(self, temp_dir):
        """Test validation of image dimension consistency."""
        # Create data with explicit dimensions
        height, width, channels = 32, 32, 3
        image_size = height * width * channels

        images = [np.random.randint(0, 256, size=image_size).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        metadata = {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'image_size': f'{height}x{width}x{channels}',
            'parameter_names': ['param_0', 'param_1']
        }

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should work correctly
        dataset = CIFAR100MHDataset(str(temp_dir), train=True)
        assert dataset.get_image_shape() == (channels, height, width)

    def test_validate_wrong_image_dimensions(self, temp_dir):
        """Test validation catches wrong image dimensions."""
        # Create data with wrong size for declared dimensions
        height, width, channels = 32, 32, 3
        wrong_size = 28 * 28 * 1  # Wrong size

        images = [np.random.randint(0, 256, size=wrong_size).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        metadata = {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'image_size': f'{height}x{width}x{channels}',
            'parameter_names': ['param_0', 'param_1']
        }

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should raise error
        with pytest.raises(ValueError, match="inconsistent image shape"):
            CIFAR100MHDataset(str(temp_dir), train=True)

    def test_validate_variable_image_dimensions(self, temp_dir):
        """Test validation of variable image dimensions."""
        # Create data with different but valid dimensions
        images = [
            np.random.randint(0, 256, size=3072).tolist(),  # 32x32x3
            np.random.randint(0, 256, size=3072).tolist(),  # 32x32x3
            np.random.randint(0, 256, size=3072).tolist(),  # 32x32x3
        ]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should work for consistent dimensions
        dataset = MockFormatValidator(str(train_file))
        assert len(dataset) == 3


class TestParameterValidation:
    """Test parameter validation functionality."""

    def test_validate_parameter_ranges(self, temp_dir):
        """Test validation of parameter ranges."""
        # Create data with parameters in range
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]  # Values in range

        data = {'data': images, 'cifar100mh_labels': labels}

        metadata = {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'parameter_names': ['param_0', 'param_1'],
            'parameter_mappings': {
                'param_0': {'min': 0, 'max': 10, 'description': 'Test param 0'},
                'param_1': {'min': 0, 'max': 10, 'description': 'Test param 1'}
            }
        }

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should work correctly
        dataset = CIFAR100MHDataset(str(temp_dir), train=True)
        assert len(dataset) == 3

    def test_validate_parameter_id_consistency(self, temp_dir):
        """Test validation of parameter ID consistency."""
        # Create data with consistent parameter IDs
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [
            [2, 0, 1, 1, 2],     # param_id 0 and 1
            [2, 0, 2, 1, 3],     # param_id 0 and 1
            [2, 0, 3, 1, 4],     # param_id 0 and 1
        ]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should work correctly
        dataset = MockFormatValidator(str(train_file))
        assert len(dataset) == 3

    def test_validate_unknown_parameter_ids(self, temp_dir):
        """Test handling of unknown parameter IDs."""
        # Create data with unknown parameter IDs
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [
            [2, 0, 1, 1, 2],     # param_id 0 and 1
            [2, 99, 2, 1, 3],    # param_id 99 (unknown) and 1
            [2, 0, 3, 1, 4],     # param_id 0 and 1
        ]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should handle gracefully (create generic param names)
        dataset = MockFormatValidator(str(train_file))
        assert len(dataset) == 3


class TestCompressionValidation:
    """Test validation of compressed format support."""

    def test_validate_uncompressed_format(self, temp_dir):
        """Test validation of standard uncompressed format."""
        # Create standard uncompressed data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(3)]
        labels = [[2, 0, i, 1, i*2] for i in range(3)]

        data = {'data': images, 'cifar100mh_labels': labels}

        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        # Should work correctly
        dataset = MockFormatValidator(str(train_file))
        assert len(dataset) == 3

    def test_validate_format_compliance(self, temp_dir):
        """Test overall format compliance validation."""
        # Create fully compliant data
        images = [np.random.randint(0, 256, size=3072).tolist() for _ in range(5)]
        labels = [[2, 0, i % 10, 1, (i * 2) % 20] for i in range(5)]

        data = {'data': images, 'cifar100mh_labels': labels}

        metadata = {
            'format': 'CIFAR-100-MH',
            'version': '1.0',
            'dataset_name': 'test_dataset',
            'n_samples': 5,
            'train_samples': 5,
            'image_size': '32x32x3',
            'parameter_names': ['param_0', 'param_1'],
            'parameter_mappings': {
                'param_0': {'min': 0, 'max': 9, 'description': 'Test param 0'},
                'param_1': {'min': 0, 'max': 19, 'description': 'Test param 1'}
            }
        }

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'cifar100mh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(data, f)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should work perfectly
        dataset = CIFAR100MHDataset(str(temp_dir), train=True)
        assert len(dataset) == 5
        assert dataset.get_image_shape() == (3, 32, 32)

        heads_config = dataset.get_heads_config()
        assert 'param_0' in heads_config
        assert 'param_1' in heads_config
        assert heads_config['param_0'] == 10  # 0-9 + 1
        assert heads_config['param_1'] == 20  # 0-19 + 1

        # Test getting parameter info
        param_info = dataset.get_parameter_info('param_0')
        assert 'description' in param_info
        assert param_info['description'] == 'Test param 0'

        # Test statistics
        stats = dataset.get_dataset_statistics()
        assert 'num_samples' in stats
        assert 'parameter_statistics' in stats
        assert stats['num_samples'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
