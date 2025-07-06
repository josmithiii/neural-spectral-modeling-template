from pathlib import Path

import pytest
import torch

from src.data.cifar100_datamodule import CIFAR100DataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_cifar100_datamodule(batch_size: int) -> None:
    """Tests `CIFAR100DataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = CIFAR100DataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "cifar-100-python").exists()

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
    assert x.shape == (batch_size, 3, 32, 32)  # CIFAR-100 is 3x32x32


def test_cifar100_datamodule_fine_labels() -> None:
    """Test that CIFAR-100 datamodule works with fine labels (100 classes)."""
    dm = CIFAR100DataModule(use_coarse_labels=False)
    assert dm.num_classes == 100
    assert len(dm.class_names) == 100
    assert len(dm.fine_class_names) == 100


def test_cifar100_datamodule_coarse_labels() -> None:
    """Test that CIFAR-100 datamodule works with coarse labels (20 classes)."""
    dm = CIFAR100DataModule(use_coarse_labels=True)
    assert dm.num_classes == 20
    assert len(dm.class_names) == 20
    assert len(dm.coarse_class_names) == 20


def test_cifar100_class_names() -> None:
    """Test that CIFAR-100 datamodule returns correct class names."""
    dm = CIFAR100DataModule()

    # Test fine class names
    fine_classes = dm.fine_class_names
    assert 'apple' in fine_classes
    assert 'aquarium_fish' in fine_classes
    assert 'wolf' in fine_classes
    assert 'worm' in fine_classes
    assert len(fine_classes) == 100

    # Test coarse class names
    coarse_classes = dm.coarse_class_names
    assert 'aquatic_mammals' in coarse_classes
    assert 'vehicles_1' in coarse_classes
    assert 'vehicles_2' in coarse_classes
    assert len(coarse_classes) == 20


def test_cifar100_datamodule_transforms() -> None:
    """Test that CIFAR-100 datamodule has correct transforms."""
    dm = CIFAR100DataModule()

    # Check that transforms are defined
    assert dm.transforms is not None
    assert dm.train_transforms is not None

    # Check that train transforms include augmentation (including extra rotation for CIFAR-100)
    transform_names = [type(t).__name__ for t in dm.train_transforms.transforms]
    assert 'RandomCrop' in transform_names
    assert 'RandomHorizontalFlip' in transform_names
    assert 'RandomRotation' in transform_names  # CIFAR-100 specific enhancement
    assert 'ToTensor' in transform_names
    assert 'Normalize' in transform_names


@pytest.mark.parametrize("use_coarse", [True, False])
def test_cifar100_label_types(use_coarse: bool) -> None:
    """Test CIFAR-100 with both fine and coarse label types.

    :param use_coarse: Whether to test coarse labels (20 classes) or fine labels (100 classes).
    """
    dm = CIFAR100DataModule(use_coarse_labels=use_coarse, batch_size=32)
    dm.prepare_data()
    dm.setup()

    expected_classes = 20 if use_coarse else 100
    assert dm.num_classes == expected_classes

    # Test that labels are in the correct range
    batch = next(iter(dm.train_dataloader()))
    _, y = batch
    assert y.min() >= 0
    assert y.max() < expected_classes


def test_cifar100_enhanced_augmentation() -> None:
    """Test that CIFAR-100 has enhanced augmentation compared to CIFAR-10."""
    dm = CIFAR100DataModule()

    # CIFAR-100 should have RandomRotation which CIFAR-10 doesn't have
    transform_names = [type(t).__name__ for t in dm.train_transforms.transforms]
    assert 'RandomRotation' in transform_names, "CIFAR-100 should have RandomRotation for enhanced augmentation"
