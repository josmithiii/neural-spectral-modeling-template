from pathlib import Path

import pytest
import torch

from src.data.cifar10_datamodule import CIFAR10DataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_cifar10_datamodule(batch_size: int) -> None:
    """Tests `CIFAR10DataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = CIFAR10DataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "cifar-10-batches-py").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 60_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert x.shape == (batch_size, 3, 32, 32)  # CIFAR-10 is 3x32x32


def test_cifar10_datamodule_num_classes() -> None:
    """Test that CIFAR-10 datamodule returns correct number of classes."""
    dm = CIFAR10DataModule()
    assert dm.num_classes == 10


def test_cifar10_datamodule_class_names() -> None:
    """Test that CIFAR-10 datamodule returns correct class names."""
    dm = CIFAR10DataModule()
    expected_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    assert dm.class_names == expected_classes


def test_cifar10_datamodule_transforms() -> None:
    """Test that CIFAR-10 datamodule has correct transforms."""
    dm = CIFAR10DataModule()

    # Check that transforms are defined
    assert dm.transforms is not None
    assert dm.train_transforms is not None

    # Check that train transforms include augmentation
    transform_names = [type(t).__name__ for t in dm.train_transforms.transforms]
    assert 'RandomCrop' in transform_names
    assert 'RandomHorizontalFlip' in transform_names
    assert 'ToTensor' in transform_names
    assert 'Normalize' in transform_names
