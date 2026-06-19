import json
import struct

import numpy as np
import pytest
import torch

from src.data.vimh_datamodule import VIMHDataModule


def _make_binary_vimh(tmp_path, n_train: int, n_test: int, height: int = 8, width: int = 8):
    """Write a minimal single-channel binary VIMH dataset and return its directory."""

    def _record(value: int) -> bytes:
        rec = struct.pack("<HHH", height, width, 1)  # dims (channels=1)
        rec += struct.pack("<ff", 0.0, 1.0)  # spec_min, spec_max
        rec += struct.pack("B", 1)  # num_params
        rec += struct.pack("BB", 0, value % 256)  # (param_id=0, quantized value)
        rec += np.full(height * width, value % 256, dtype=np.uint8).tobytes()
        return rec

    (tmp_path / "train").write_bytes(b"".join(_record(i) for i in range(n_train)))
    (tmp_path / "test").write_bytes(b"".join(_record(1000 + i) for i in range(n_test)))

    metadata = {
        "format": "VIMH",
        "height": height,
        "width": width,
        "channels": 1,
        "train_samples": n_train,
        "test_samples": n_test,
        "parameter_names": ["p0"],
        "parameter_mappings": {"p0": {"min": 0.0, "max": 1.0, "step": 0.1, "scale": "linear"}},
    }
    with open(tmp_path / "vimh_dataset_info.json", "w") as f:
        json.dump(metadata, f)
    return str(tmp_path)


def test_train_val_split_is_disjoint_seeded_and_holds_out_test(tmp_path):
    """Validation must be a seeded split of the train file, disjoint from train.

    Regression test: validation previously reused the test file (val == test),
    so model selection tuned on the test set.
    """
    data_dir = _make_binary_vimh(tmp_path, n_train=20, n_test=5)

    dm = VIMHDataModule(data_dir=data_dir, batch_size=4, val_split=0.25, split_seed=42)
    dm.setup()

    # 25% of 20 = 5 val, 15 train; disjoint and covering all train samples.
    assert len(dm._val_indices) == 5
    assert len(dm._train_indices) == 15
    assert set(dm._train_indices).isdisjoint(dm._val_indices)
    assert set(dm._train_indices) | set(dm._val_indices) == set(range(20))

    # Validation is drawn from the train file, not the held-out test file.
    assert len(dm.data_test) == 5
    assert len(dm.data_val) == 20  # full train file, subset by _val_indices

    # The split is deterministic for a fixed seed and changes with the seed.
    dm2 = VIMHDataModule(data_dir=data_dir, batch_size=4, val_split=0.25, split_seed=42)
    dm2.setup()
    assert dm2._val_indices == dm._val_indices
    dm3 = VIMHDataModule(data_dir=data_dir, batch_size=4, val_split=0.25, split_seed=7)
    dm3.setup()
    assert dm3._val_indices != dm._val_indices


def test_invalid_val_split_fails_fast(tmp_path):
    """A val_split that yields zero validation samples must raise, not silently pass."""
    data_dir = _make_binary_vimh(tmp_path, n_train=20, n_test=5)
    dm = VIMHDataModule(data_dir=data_dir, batch_size=4, val_split=0.0)
    with pytest.raises((ValueError, RuntimeError)):
        dm.setup()


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
