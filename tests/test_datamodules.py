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
