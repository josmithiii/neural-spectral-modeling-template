import json
from pathlib import Path
from typing import Any, Dict

import pytest
from omegaconf import OmegaConf

from src.models.vimh_lit_module import VIMHLitModule, output_mode_for_loss_type
from src.train import configure_vimh_run_config


@pytest.fixture()
def sample_vimh_metadata(tmp_path: Path) -> Path:
    metadata = {
        "height": 32,
        "width": 32,
        "channels": 1,
        "parameter_names": ["log10_decay_time", "wah_position"],
        "parameter_mappings": {
            "log10_decay_time": {
                "min": -2.0,
                "max": 0.3,
                "step": 0.1,
            },
            "wah_position": {
                "min": 0.0,
                "max": 1.0,
                "step": 0.02,
            },
        },
    }
    metadata_path = tmp_path / "vimh_dataset_info.json"
    metadata_path.write_text(json.dumps(metadata))
    return tmp_path


def _cfg(data_dir: Path, model: Dict[str, Any], data: Dict[str, Any] = None):
    return OmegaConf.create(
        {
            "data": {
                "_target_": "src.data.vimh_datamodule.VIMHDataModule",
                "data_dir": str(data_dir),
                **(data or {}),
            },
            "model": {"auto_configure_from_dataset": True, "net": {}, **model},
        }
    )


def test_output_mode_comes_from_loss_type():
    assert output_mode_for_loss_type("normalized_regression") == "regression"
    for loss_type in ("cross_entropy", "ordinal_regression", "weighted_cross_entropy", "soft_target"):
        assert output_mode_for_loss_type(loss_type) == "classification"
    with pytest.raises(ValueError, match="Unknown loss_type"):
        output_mode_for_loss_type("quantized_regression")


def test_regression_auto_configuration_populates_losses_and_label_mode(sample_vimh_metadata: Path):
    cfg = _cfg(sample_vimh_metadata, {"loss_type": "normalized_regression"})

    configure_vimh_run_config(cfg)

    assert cfg.data.label_mode == "regression"
    assert cfg.model.net.parameter_names == ["log10_decay_time", "wah_position"]
    assert cfg.model.net.output_mode == "regression"
    assert cfg.model.net.heads_config is None

    criteria = OmegaConf.to_container(cfg.model.criteria, resolve=True)
    assert set(criteria.keys()) == {"log10_decay_time", "wah_position"}
    assert criteria["log10_decay_time"]["_target_"] == "src.models.losses.NormalizedRegressionLoss"
    assert tuple(criteria["log10_decay_time"]["param_range"]) == (-2.0, 0.3)
    # Loss weights are left to VIMHLitModule (JND-based when empty)
    assert "loss_weights" not in cfg.model


def test_classification_sets_class_index_labels_and_heads(sample_vimh_metadata: Path):
    cfg = _cfg(sample_vimh_metadata, {"loss_type": "cross_entropy"})
    configure_vimh_run_config(cfg)
    assert cfg.data.label_mode == "classification"
    assert dict(cfg.model.net.heads_config) == {"log10_decay_time": 24, "wah_position": 51}


def test_conflicting_label_mode_raises(sample_vimh_metadata: Path):
    """A regression loss with class-index labels used to train on garbage targets (make etmsr)."""
    cfg = _cfg(
        sample_vimh_metadata,
        {"loss_type": "normalized_regression"},
        {"label_mode": "classification"},
    )
    with pytest.raises(ValueError, match="conflicts"):
        configure_vimh_run_config(cfg)


def test_auxiliary_features_are_inputs_not_heads(sample_vimh_metadata: Path):
    """The auxiliary input must be wired even when the net config lacks the key."""
    cfg = _cfg(
        sample_vimh_metadata,
        {"loss_type": "cross_entropy"},
        {"auxiliary_features": ["log10_decay_time"]},
    )
    configure_vimh_run_config(cfg)
    assert dict(cfg.model.net.heads_config) == {"wah_position": 51}
    assert cfg.model.net.auxiliary_input_size == 1


def test_unknown_auxiliary_feature_raises(sample_vimh_metadata: Path):
    cfg = _cfg(
        sample_vimh_metadata, {"loss_type": "cross_entropy"}, {"auxiliary_features": ["decay_time"]}
    )
    with pytest.raises(ValueError, match="not dataset parameters"):
        configure_vimh_run_config(cfg)


def test_regression_criteria_for_non_head_raises(sample_vimh_metadata: Path):
    cfg = _cfg(
        sample_vimh_metadata,
        {"loss_type": "normalized_regression", "criteria": {"log10_decay_time": {"loss_type": "l1"}}},
        {"auxiliary_features": ["log10_decay_time"]},
    )
    with pytest.raises(ValueError, match="not prediction heads"):
        configure_vimh_run_config(cfg)


def test_jnd_loss_weights_normalized_to_finest_head(sample_vimh_metadata: Path):
    metadata = json.loads((sample_vimh_metadata / "vimh_dataset_info.json").read_text())
    weights = VIMHLitModule._compute_jnd_weights(metadata, ["log10_decay_time", "wah_position"])
    # wah_position spans the most JND steps (50) -> 1.0; log10_decay_time spans 23
    assert weights["wah_position"] == pytest.approx(1.0)
    assert weights["log10_decay_time"] == pytest.approx(23 / 50)
