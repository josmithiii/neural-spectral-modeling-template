import pytest
import torch
import numpy as np
import pickle
import json
import tempfile
import shutil
from pathlib import Path

from src.data.vimh_dataset import VIMHDataset, create_vimh_datasets
from src.data.vimh_datamodule import VIMHDataModule


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vimh_data():
    """Create mock VIMH data for testing."""
    # Create image data (32x32x3 = 3072 pixels)
    images = []
    labels = []

    for i in range(10):  # 10 samples
        # Random image data
        image_data = np.random.randint(0, 256, size=3072, dtype=np.uint8).tolist()
        images.append(image_data)

        # VIMH label format: [N] [param1_id] [param1_val] [param2_id] [param2_val]
        label_data = [2, 0, i % 256, 1, (i * 2) % 256]
        labels.append(label_data)

    return {
        'data': images,
        'vimh_labels': labels,
        'height': 32,
        'width': 32,
        'channels': 3,
        'image_size': 3072
    }


@pytest.fixture
def mock_vimh_metadata():
    """Create mock VIMH metadata configuration."""
    return {
        'format': 'VIMH',
        'version': '3.0',
        'dataset_name': 'test_vimh_dataset',
        'n_samples': 10,
        'train_samples': 8,
        'test_samples': 2,
        'image_size': '32x32x3',
        'height': 32,
        'width': 32,
        'channels': 3,
        'varying_parameters': 2,
        'parameter_names': ['note_number', 'note_velocity'],
        'label_encoding': {
            'format': '[height] [width] [channels] [N] [param1_id] [param1_val] [param2_id] [param2_val] ...',
            'metadata_bytes': 6,
            'N_range': [0, 255],
            'param_id_range': [0, 255],
            'param_val_range': [0, 255]
        },
        'parameter_mappings': {
            'note_number': {
                'min': 50.0,
                'max': 52.0,
                'scale': 'linear',
                'description': 'Test note number parameter'
            },
            'note_velocity': {
                'min': 80.0,
                'max': 82.0,
                'scale': 'linear',
                'description': 'Test note velocity parameter'
            }
        }
    }


def create_test_vimh_files(temp_dir: Path, mock_data: dict, mock_metadata: dict):
    """Create test VIMH dataset files in temporary directory."""
    # Create data files
    train_file = temp_dir / 'train_batch'
    test_file = temp_dir / 'test_batch'
    metadata_file = temp_dir / 'vimh_dataset_info.json'

    # Split data into train/test
    train_data = {
        'data': mock_data['data'][:8],
        'vimh_labels': mock_data['vimh_labels'][:8],
        'height': mock_data['height'],
        'width': mock_data['width'],
        'channels': mock_data['channels']
    }
    test_data = {
        'data': mock_data['data'][8:],
        'vimh_labels': mock_data['vimh_labels'][8:],
        'height': mock_data['height'],
        'width': mock_data['width'],
        'channels': mock_data['channels']
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


class TestVIMHDataset:
    """Test cases for VIMHDataset class."""

    def test_init_train_dataset(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test initialization of training dataset."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=True)

        assert len(dataset) == 8  # Training samples
        assert dataset.train is True
        assert dataset.get_image_shape() == (3, 32, 32)

        heads_config = dataset.get_heads_config()
        assert 'note_number' in heads_config
        assert 'note_velocity' in heads_config
        assert heads_config['note_number'] == 256  # 8-bit quantization
        assert heads_config['note_velocity'] == 256

    def test_init_test_dataset(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test initialization of test dataset."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=False)

        assert len(dataset) == 2  # Test samples
        assert dataset.train is False

    def test_getitem(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test getting individual samples."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=True)

        image, labels = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert isinstance(labels, dict)
        assert 'note_number' in labels
        assert 'note_velocity' in labels

    def test_getitem_with_transforms(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test getting samples with transforms applied."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        # Simple transform that doubles the image values
        image_transform = lambda x: x * 2
        target_transform = lambda x: {k: v * 2 for k, v in x.items()}

        dataset = VIMHDataset(
            str(temp_dir),
            train=True,
            transform=image_transform,
            target_transform=target_transform
        )

        image, labels = dataset[0]

        # Check that transforms were applied
        assert torch.max(image) > 1.0  # Should be > 1 due to doubling
        assert all(v % 2 == 0 for v in labels.values())  # All labels should be even

    def test_get_parameter_info(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test getting parameter information."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=True)

        param_info = dataset.get_parameter_info('note_number')
        assert 'description' in param_info
        assert param_info['description'] == 'Test note number parameter'
        assert param_info['min'] == 50.0
        assert param_info['max'] == 52.0

        # Test non-existent parameter
        unknown_info = dataset.get_parameter_info('unknown_param')
        assert 'Parameter not found' in unknown_info['description']

    def test_get_sample_metadata(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test getting sample metadata with parameter dequantization."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=True)

        metadata = dataset._get_sample_metadata(0)

        assert 'sample_index' in metadata
        assert 'labels' in metadata
        assert 'image_shape' in metadata
        assert 'note_number_info' in metadata
        assert 'note_velocity_info' in metadata

        # Check dequantization
        note_info = metadata['note_number_info']
        assert 'quantized_value' in note_info
        assert 'actual_value' in note_info
        assert note_info['actual_value'] >= 50.0
        assert note_info['actual_value'] <= 52.0

    def test_get_class_distribution(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test getting class distribution."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=True)

        distribution = dataset.get_class_distribution()

        assert 'note_number' in distribution
        assert 'note_velocity' in distribution
        assert isinstance(distribution['note_number'], dict)
        assert len(distribution['note_number']) > 0  # Should have some class counts

    def test_get_dataset_statistics(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test getting comprehensive dataset statistics."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=True)

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
        assert 'min_quantized' in param_stats['note_number']
        assert 'max_quantized' in param_stats['note_number']
        assert 'min_actual' in param_stats['note_number']
        assert 'max_actual' in param_stats['note_number']

    def test_missing_metadata_file(self, temp_dir, mock_vimh_data):
        """Test handling of missing metadata file."""
        train_file = temp_dir / 'train_batch'
        with open(train_file, 'wb') as f:
            pickle.dump(mock_vimh_data, f)

        # Should work with default metadata
        dataset = VIMHDataset(str(temp_dir), train=True)
        assert len(dataset) > 0

    def test_create_vimh_datasets(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test factory function for creating train/test datasets."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        train_dataset, test_dataset = create_vimh_datasets(str(temp_dir))

        assert isinstance(train_dataset, VIMHDataset)
        assert isinstance(test_dataset, VIMHDataset)
        assert train_dataset.train is True
        assert test_dataset.train is False
        assert len(train_dataset) == 8
        assert len(test_dataset) == 2

    def test_variable_image_sizes(self, temp_dir, mock_vimh_metadata):
        """Test support for variable image sizes."""
        # Create data with different image size (28x28x1 like MNIST)
        images = []
        labels = []

        for i in range(5):
            # MNIST-like image data (28x28x1 = 784 pixels)
            image_data = np.random.randint(0, 256, size=784, dtype=np.uint8).tolist()
            images.append(image_data)

            # VIMH label format for 28x28x1
            label_data = [1, 0, i % 256]  # 1 parameter
            labels.append(label_data)

        mock_data = {
            'data': images,
            'vimh_labels': labels,
            'height': 28,
            'width': 28,
            'channels': 1,
            'image_size': 784
        }

        # Update metadata for 28x28x1 images
        mock_metadata = mock_vimh_metadata.copy()
        mock_metadata.update({
            'height': 28,
            'width': 28,
            'channels': 1,
            'varying_parameters': 1,
            'parameter_names': ['digit_class']
        })

        train_file = temp_dir / 'train_batch'
        metadata_file = temp_dir / 'vimh_dataset_info.json'

        with open(train_file, 'wb') as f:
            pickle.dump(mock_data, f)

        with open(metadata_file, 'w') as f:
            json.dump(mock_metadata, f, indent=2)

        dataset = VIMHDataset(str(temp_dir), train=True)

        assert dataset.get_image_shape() == (1, 28, 28)
        assert len(dataset) == 5

        # Test sample
        image, labels = dataset[0]
        assert image.shape == (1, 28, 28)
        assert 'digit_class' in labels


class TestVIMHDataModule:
    """Test cases for VIMHDataModule class."""

    def test_init_datamodule(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test initialization of VIMH data module."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        assert dm.hparams.data_dir == str(temp_dir)
        assert dm.hparams.batch_size == 4
        assert dm.hparams.num_workers == 0

    def test_setup_datamodule(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test setup of VIMH data module."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        dm.setup()

        assert dm.data_train is not None
        assert dm.data_val is not None
        assert dm.data_test is not None
        assert len(dm.heads_config) == 2
        assert dm.image_shape == (3, 32, 32)

    def test_dataloaders(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test creation of data loaders."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Test batch loading
        batch_images, batch_labels = next(iter(train_loader))
        assert isinstance(batch_images, torch.Tensor)
        assert isinstance(batch_labels, dict)
        assert 'note_number' in batch_labels
        assert 'note_velocity' in batch_labels

    def test_multihead_collate_fn(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test multihead collate function."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=2,
            num_workers=0
        )

        dm.setup()

        # Create mock batch data
        batch = [
            (torch.randn(3, 32, 32), {'note_number': 10, 'note_velocity': 20}),
            (torch.randn(3, 32, 32), {'note_number': 15, 'note_velocity': 25})
        ]

        batched_images, batched_labels = dm._multihead_collate_fn(batch)

        assert batched_images.shape == (2, 3, 32, 32)
        assert batched_labels['note_number'].shape == (2,)
        assert batched_labels['note_velocity'].shape == (2,)
        assert batched_labels['note_number'].tolist() == [10, 15]
        assert batched_labels['note_velocity'].tolist() == [20, 25]

    def test_get_dataset_info(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test getting dataset information."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        dm.setup()

        info = dm.get_dataset_info()

        assert 'heads_config' in info
        assert 'image_shape' in info
        assert 'num_train_samples' in info
        assert 'num_val_samples' in info
        assert 'num_test_samples' in info
        assert info['num_train_samples'] == 8
        assert info['num_test_samples'] == 2

    def test_num_classes_property(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test num_classes property."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        dm.setup()

        num_classes = dm.num_classes
        assert isinstance(num_classes, dict)
        assert 'note_number' in num_classes
        assert 'note_velocity' in num_classes
        assert num_classes['note_number'] == 256
        assert num_classes['note_velocity'] == 256

    def test_adjust_transforms_for_image_size(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test transform adjustment based on image size."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        # Test 32x32 image adjustment
        dm._adjust_transforms_for_image_size(32, 32)
        assert dm.train_transform is not None
        assert dm.val_transform is not None

        # Test 28x28 image adjustment
        dm._adjust_transforms_for_image_size(28, 28)
        assert dm.train_transform is not None
        assert dm.val_transform is not None

    def test_efficient_dimension_detection(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test efficient dimension detection methods."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        # Test JSON metadata loading
        json_dims = dm._load_image_dims_from_json(str(temp_dir))
        assert json_dims == (32, 32, 3)

        # Test binary metadata validation
        binary_dims = dm._validate_binary_metadata(str(temp_dir))
        assert binary_dims == (32, 32, 3)

        # Test unified dimension detection
        detected_dims = dm._detect_and_validate_image_dimensions(str(temp_dir))
        assert detected_dims == (32, 32, 3)

    def test_dimension_validation_consistency(self, temp_dir):
        """Test dimension validation with consistent sources."""
        import pickle
        import json

        # Create consistent data with vimh- prefix in directory name
        vimh_dir = temp_dir / "vimh-28x28x1_test_dataset"
        vimh_dir.mkdir()

        # Create consistent metadata
        metadata = {
            'format': 'VIMH',
            'height': 28,
            'width': 28,
            'channels': 1,
            'parameter_names': ['digit'],
            'parameter_mappings': {'digit': {'min': 0, 'max': 9}}
        }

        # Create consistent pickle data
        mock_data = {
            'data': [list(range(784))],  # 28*28*1 = 784
            'vimh_labels': [[1, 0, 128]],
            'height': 28,
            'width': 28,
            'channels': 1
        }

        # Save files
        with open(vimh_dir / 'vimh_dataset_info.json', 'w') as f:
            json.dump(metadata, f)
        with open(vimh_dir / 'train_batch', 'wb') as f:
            pickle.dump(mock_data, f)

        dm = VIMHDataModule(
            data_dir=str(vimh_dir),
            batch_size=4,
            num_workers=0
        )

        # Test directory name parsing
        dir_dims = dm._parse_image_dims_from_path(str(vimh_dir))
        assert dir_dims == (28, 28, 1)

        # Test full validation - should succeed with all sources agreeing
        detected_dims = dm._detect_and_validate_image_dimensions(str(vimh_dir))
        assert detected_dims == (28, 28, 1)

    def test_dimension_validation_mismatch(self, temp_dir):
        """Test dimension validation with inconsistent sources."""
        import pickle
        import json

        # Create inconsistent data
        vimh_dir = temp_dir / "vimh-32x32x3_test_dataset"  # Directory says 32x32x3
        vimh_dir.mkdir()

        # But JSON says 28x28x1
        metadata = {
            'format': 'VIMH',
            'height': 28,
            'width': 28,
            'channels': 1,
            'parameter_names': ['digit'],
            'parameter_mappings': {'digit': {'min': 0, 'max': 9}}
        }

        # And pickle data says 28x28x1 too
        mock_data = {
            'data': [list(range(784))],  # 28*28*1 = 784
            'vimh_labels': [[1, 0, 128]],
            'height': 28,
            'width': 28,
            'channels': 1
        }

        # Save files
        with open(vimh_dir / 'vimh_dataset_info.json', 'w') as f:
            json.dump(metadata, f)
        with open(vimh_dir / 'train_batch', 'wb') as f:
            pickle.dump(mock_data, f)

        dm = VIMHDataModule(
            data_dir=str(vimh_dir),
            batch_size=4,
            num_workers=0
        )

        # Test should raise ValueError due to dimension mismatch
        with pytest.raises(ValueError, match="Dimension mismatch"):
            dm._detect_and_validate_image_dimensions(str(vimh_dir))

    def test_efficient_heads_config_loading(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test efficient heads configuration loading from JSON."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )

        # Test efficient heads config loading
        heads_config = dm._load_dataset_metadata(str(temp_dir))
        assert isinstance(heads_config, dict)
        assert 'note_number' in heads_config
        assert 'note_velocity' in heads_config
        assert heads_config['note_number'] == 256
        assert heads_config['note_velocity'] == 256


class TestVIMHIntegration:
    """Integration tests for VIMH dataset functionality."""

    def test_end_to_end_vimh_workflow(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test complete end-to-end VIMH workflow."""
        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        # Test dataset creation
        dataset = VIMHDataset(str(temp_dir), train=True)
        assert len(dataset) > 0

        # Test data module creation
        dm = VIMHDataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0
        )
        dm.setup()

        # Test dataloader creation
        train_loader = dm.train_dataloader()
        batch_images, batch_labels = next(iter(train_loader))

        assert isinstance(batch_images, torch.Tensor)
        assert isinstance(batch_labels, dict)
        assert batch_images.shape[0] <= 4  # Batch size

        # Test parameter info retrieval
        param_info = dataset.get_parameter_info('note_number')
        assert 'min' in param_info
        assert 'max' in param_info

    def test_compatibility_with_pytorch_dataloader(self, temp_dir, mock_vimh_data, mock_vimh_metadata):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        create_test_vimh_files(temp_dir, mock_vimh_data, mock_vimh_metadata)

        dataset = VIMHDataset(str(temp_dir), train=True)

        # Create custom collate function for testing
        def custom_collate(batch):
            images = []
            labels_dict = {}

            for image, labels in batch:
                images.append(image)
                if not labels_dict:
                    for head_name in labels.keys():
                        labels_dict[head_name] = []
                for head_name, label_value in labels.items():
                    labels_dict[head_name].append(label_value)

            batched_images = torch.stack(images)
            batched_labels = {}
            for head_name, label_list in labels_dict.items():
                batched_labels[head_name] = torch.tensor(label_list, dtype=torch.long)

            return batched_images, batched_labels

        # Test DataLoader creation
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=custom_collate)

        # Test batch loading
        batch_images, batch_labels = next(iter(dataloader))

        assert isinstance(batch_images, torch.Tensor)
        assert batch_images.shape[0] == 3  # Batch size
        assert isinstance(batch_labels, dict)

        # Check that each label type has correct batch dimension
        for label_name, label_tensor in batch_labels.items():
            assert isinstance(label_tensor, torch.Tensor)
            assert label_tensor.shape[0] == 3  # Batch size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
