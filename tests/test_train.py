import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import _link_best_checkpoint, _select_test_checkpoint, train
from tests.helpers.run_if import RunIf


def _fake_trainer(best_model_path: str, fast_dev_run: bool = False, has_ckpt_cb: bool = True):
    from types import SimpleNamespace

    cb = SimpleNamespace(best_model_path=best_model_path, monitor="val/acc_best", best_model_score=0.5)
    return SimpleNamespace(
        checkpoint_callback=cb if has_ckpt_cb else None, fast_dev_run=fast_dev_run, global_rank=0
    )


def test_select_test_checkpoint() -> None:
    """Test the best checkpoint of this run, also when resuming from ckpt_path."""
    from omegaconf import OmegaConf

    resumed = OmegaConf.create({"train": True, "ckpt_path": "resume.ckpt"})
    assert _select_test_checkpoint(resumed, _fake_trainer("best.ckpt")) == "best.ckpt"
    trained = OmegaConf.create({"train": True})
    assert _select_test_checkpoint(trained, _fake_trainer("", has_ckpt_cb=False)) is None
    assert _select_test_checkpoint(trained, _fake_trainer("", fast_dev_run=True)) is None
    with pytest.raises(RuntimeError, match="no best checkpoint"):
        _select_test_checkpoint(trained, _fake_trainer(""))
    with pytest.raises(ValueError, match="requires ckpt_path"):
        _select_test_checkpoint(OmegaConf.create({"train": False}), _fake_trainer(""))


def test_link_best_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "epoch_003.ckpt").write_bytes(b"best")
    _link_best_checkpoint(_fake_trainer(str(tmp_path / "epoch_003.ckpt")))
    link = tmp_path / "best.ckpt"
    assert link.is_symlink() and os.readlink(link) == "epoch_003.ckpt"

    # Never replace a real file
    link.unlink()
    link.write_bytes(b"real checkpoint")
    with pytest.raises(RuntimeError, match="Refusing"):
        _link_best_checkpoint(_fake_trainer(str(tmp_path / "epoch_003.ckpt")))
    assert link.read_bytes() == b"real checkpoint"


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
@pytest.mark.skip(
    reason="DDP test fails with PyTorch 2.6 checkpoint loading in multiprocessing - known issue"
)
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        # Unseeded, the accuracy comparison below was flaky (~1 in 3 runs failed)
        cfg_train.seed = 12345
        # Configure checkpoint callback to save every epoch for testing
        cfg_train.callbacks.model_checkpoint.every_n_epochs = 1
        cfg_train.callbacks.model_checkpoint.save_top_k = -1  # Save all checkpoints

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    # Use parameter-specific accuracy metrics for multihead predictions
    assert (
        metric_dict_1["train/log10_decay_time_acc"] <= metric_dict_2["train/log10_decay_time_acc"]
    )
    assert metric_dict_1["val/log10_decay_time_acc"] <= metric_dict_2["val/log10_decay_time_acc"]
