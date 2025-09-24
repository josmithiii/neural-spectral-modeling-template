"""Helpers for configuring VIMH models from dataset metadata."""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils import RankedLogger
from src.utils.vimh_utils import (
    get_heads_config_from_metadata,
    get_parameter_names_from_metadata,
    get_parameter_ranges_from_metadata,
    load_vimh_metadata,
)

log = RankedLogger(__name__, rank_zero_only=True)


def configure_vimh_model(cfg: DictConfig) -> None:
    """Mutate the model section of the config using VIMH dataset metadata.

    This mirrors the logic previously embedded in ``train.py`` so the same auto-configuration can be
    reused for evaluation. The function is a no-op when the configured dataset is not VIMH or the
    model disables ``auto_configure_from_dataset``.
    """
    if not getattr(cfg, "data", None) or not getattr(cfg.data, "_target_", None):
        return
    if "vimh" not in cfg.data._target_.lower():
        return
    if not getattr(cfg, "model", None) or not hasattr(cfg.model, "auto_configure_from_dataset"):
        return
    if not cfg.model.auto_configure_from_dataset:
        return

    try:
        parameter_names = get_parameter_names_from_metadata(cfg.data.data_dir)
    except Exception as exc:  # pragma: no cover - surfaced by caller
        log.warning(f"Unable to load VIMH metadata; skipping auto-configuration: {exc}")
        return

    if not parameter_names:
        log.debug("No VIMH parameter names discovered; skipping auto-configuration.")
        return

    log.info(f"Configuring model with parameter names from dataset: {parameter_names}")
    output_mode = getattr(cfg.model, "output_mode", None)

    if output_mode == "regression":
        with open_dict(cfg.model):
            cfg.model.net.parameter_names = parameter_names
            cfg.model.net.output_mode = "regression"
            cfg.model.net.heads_config = None
    else:
        heads_config = get_heads_config_from_metadata(cfg.data.data_dir)
        with open_dict(cfg.model):
            cfg.model.net.heads_config = heads_config

    if output_mode == "regression":
        param_ranges = get_parameter_ranges_from_metadata(cfg.data.data_dir)
        base_criteria_cfg: Dict[str, Any] = {}
        for param_name in parameter_names:
            if param_name not in param_ranges:
                raise KeyError(f"Missing parameter range for '{param_name}' in metadata")
            param_range = param_ranges[param_name]
            base_criteria_cfg[param_name] = {
                "_target_": "src.models.losses.NormalizedRegressionLoss",
                "param_range": tuple(param_range),
            }

        merged_criteria_cfg: Dict[str, Any] = {}
        user_criteria = getattr(cfg.model, "criteria", None)
        for head, base_cfg in base_criteria_cfg.items():
            merged = dict(base_cfg)
            if user_criteria and head in user_criteria:
                for key, value in user_criteria[head].items():
                    if key not in ("_target_", "param_range"):
                        merged[key] = value
            merged_criteria_cfg[head] = merged

        with open_dict(cfg.model):
            cfg.model.criteria = OmegaConf.create(merged_criteria_cfg)

        log.info(
            f"Auto-configured regression loss functions for: {list(merged_criteria_cfg.keys())}"
        )

    configure_loss_weights = not getattr(cfg.model, "loss_weights", None)
    if configure_loss_weights:
        metadata = load_vimh_metadata(cfg.data.data_dir)
        param_mappings = metadata.get("parameter_mappings", {})
        loss_weights = {}
        for param_name in parameter_names:
            if param_name not in param_mappings:
                raise KeyError(f"Parameter '{param_name}' missing from parameter_mappings")
            mapping = param_mappings[param_name]
            step = float(mapping["step"])
            if step <= 0:
                raise ValueError(f"Parameter '{param_name}' has non-positive step: {step}")
            param_range = float(mapping["max"]) - float(mapping["min"])
            loss_weights[param_name] = float(param_range / step)

        if loss_weights:
            max_weight = max(loss_weights.values()) or 1.0
            loss_weights = {name: weight / max_weight for name, weight in loss_weights.items()}

        with open_dict(cfg.model):
            cfg.model.loss_weights = loss_weights
        log.info(f"Auto-configured JND-based loss_weights (normalized): {cfg.model.loss_weights}")

