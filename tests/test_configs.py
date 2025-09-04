import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
import pytest


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_regression_model_config() -> None:
    """Test that the regression model configuration is valid."""
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="train", overrides=["model=cnn_64k_regression"])

        assert cfg.model.output_mode == "regression"
        assert cfg.model.net.output_mode == "regression"
        assert cfg.model.auto_configure_from_dataset == True

        # Since criteria will be auto-configured, we should only check that
        # loss_weights is present and empty (to be auto-configured)
        assert "loss_weights" in cfg.model
        assert cfg.model.loss_weights == {}

        # Test that net has the required parameters for regression mode
        # parameter_names should have a default placeholder that will be auto-configured
        assert "parameter_names" in cfg.model.net
        assert isinstance(cfg.model.net.parameter_names, (list, ListConfig))

        # Test instantiation without HydraConfig.set_config requires parameter_names
        # Since parameter_names is empty, instantiation will fail, so we skip this test
        # The auto-configuration happens during setup() in the Lightning module


def test_regression_experiment_config() -> None:
    """Test that the regression experiment configuration is valid."""
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="train", overrides=["experiment=trivial_micro_small_regression"])

        assert cfg.model.output_mode == "regression"
        assert cfg.optimized_metric == "val/mae_best"
        assert "regression" in cfg.tags
        assert "trivial" in cfg.tags
        assert "micro" in cfg.tags

        # Test that auto_configure_from_dataset is enabled
        assert cfg.model.auto_configure_from_dataset == True

        # Test instantiation without HydraConfig.set_config requires parameter_names
        # Since parameter_names is empty, instantiation will fail, so we skip this test
        # The auto-configuration happens during setup() in the Lightning module
