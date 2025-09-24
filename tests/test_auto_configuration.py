import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.vimh_configurator import configure_vimh_model


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


def test_regression_auto_configuration_populates_losses_and_weights(sample_vimh_metadata: Path):
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "src.data.vimh_datamodule.VIMHDataModule", "data_dir": str(sample_vimh_metadata)},
            "model": {
                "output_mode": "regression",
                "auto_configure_from_dataset": True,
                "net": {},
            },
        }
    )

    configure_vimh_model(cfg)

    assert cfg.model.net.parameter_names == ["log10_decay_time", "wah_position"]
    assert cfg.model.net.output_mode == "regression"
    assert cfg.model.net.heads_config is None

    criteria = OmegaConf.to_container(cfg.model.criteria, resolve=True)
    assert set(criteria.keys()) == {"log10_decay_time", "wah_position"}
    assert criteria["log10_decay_time"]["_target_"] == "src.models.losses.NormalizedRegressionLoss"
    assert tuple(criteria["log10_decay_time"]["param_range"]) == (-2.0, 0.3)

    weights = dict(cfg.model.loss_weights)
    assert set(weights.keys()) == {"log10_decay_time", "wah_position"}
    # wah_position should normalize to 1.0 because it has the highest JND count
    assert pytest.approx(weights["wah_position"]) == 1.0
    assert 0.0 < weights["log10_decay_time"] < 1.0
