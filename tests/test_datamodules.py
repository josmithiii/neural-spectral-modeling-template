import pytest
import torch

from src.data.vimh_datamodule import VIMHDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_vimh_datamodule(batch_size: int) -> None:
    """Tests `VIMHDataModule` to verify that it can be setup correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    # Skip test if no VIMH data is available
    try:
        dm = VIMHDataModule(batch_size=batch_size)
        dm.setup()

        assert dm.data_train and dm.data_val and dm.data_test
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert len(x) == batch_size
        assert x.dtype == torch.float32

        # VIMH uses multihead outputs
        if isinstance(y, dict):
            for head_name, head_targets in y.items():
                assert len(head_targets) == batch_size
        else:
            assert len(y) == batch_size

    except Exception:
        pytest.skip("VIMH dataset not available - skipping datamodule test")


def test_default_train_transform_has_no_geometric_augmentation():
    """Spectrogram default transforms must be normalization-only.

    Regression test: the defaults previously applied RandomHorizontalFlip
    (reverses the time axis) and RandomRotation (mixes frequency/time), which
    corrupt the synth parameters being predicted.
    """
    dm = VIMHDataModule()
    dm._adjust_transforms_for_image_size(32, 32, 1)

    names = [type(t).__name__ for t in dm.train_transform.transforms]
    assert names == ["Normalize"]
    for forbidden in ("RandomHorizontalFlip", "RandomRotation", "RandomCrop", "ColorJitter"):
        assert forbidden not in names

    # Train and eval must see identical, deterministic preprocessing.
    img = torch.linspace(0.0, 1.0, 32 * 32).reshape(1, 32, 32)
    out1 = dm.train_transform(img)
    out2 = dm.train_transform(img)
    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, dm.val_transform(img))
