# Disable PyTorch 2.6 weights_only restriction for trusted LOCAL checkpoints
import os.path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

# Register custom resolver to flatten path separators in experiment names
# Enables experiment=wah/del/cnn/medium -> run dir ..._wah_del_cnn_medium
OmegaConf.register_new_resolver("flatten", lambda x: x.replace("/", "_") if x else x)

_original_torch_load = torch.load


def _patched_torch_load(
    f, map_location=None, pickle_module=None, weights_only=None, mmap=None, **kwargs
):
    # Only allow loading from local files, not URLs
    if isinstance(f, str):
        if f.startswith(("http://", "https://", "ftp://", "ftps://")):
            raise ValueError(f"Remote checkpoint loading not allowed for security: {f}")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Checkpoint file not found: {f}")
    # Force weights_only=False for trusted local research checkpoints
    return _original_torch_load(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=False,
        mmap=mmap,
        **kwargs,
    )


torch.load = _patched_torch_load

# Also patch Lightning's internal checkpoint loading
try:
    from lightning.fabric.utilities import cloud_io

    cloud_io._load = _patched_torch_load
except ImportError:
    pass

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.vimh_utils import load_vimh_metadata
from src.utils.architecture_utils import (
    ArchitectureMetadataExtractor,
    create_spectrogram_transforms,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _configure_vimh_model_config(cfg: DictConfig) -> None:
    """Configure model config for VIMH datasets before instantiation."""
    try:
        from src.utils.vimh_utils import (
            get_heads_config_from_metadata,
            get_parameter_names_from_metadata,
            get_parameter_ranges_from_metadata,
            load_vimh_metadata,
        )
    except ImportError as exc:
        log.warning(f"VIMH utilities unavailable; skipping auto-configuration: {exc}")
        return

    if not getattr(cfg, "data", None) or not getattr(cfg.data, "data_dir", None):
        return
    if not getattr(cfg, "model", None) or not hasattr(cfg.model, "net"):
        return

    parameter_names = get_parameter_names_from_metadata(cfg.data.data_dir)
    if not parameter_names:
        log.debug("No VIMH parameter names discovered; skipping auto-configuration.")
        return

    # Remove auxiliary features from parameter_names (they are inputs, not predictions)
    auxiliary_features = getattr(cfg.data, "auxiliary_features", None) or []
    if auxiliary_features:
        original_params = parameter_names.copy()
        parameter_names = [p for p in parameter_names if p not in auxiliary_features]
        removed = [p for p in original_params if p not in parameter_names]
        if removed:
            log.info(
                f"Removed auxiliary features from prediction targets: {removed}"
            )
            log.info(
                f"Auxiliary features (measured inputs): {auxiliary_features}"
            )

    if not parameter_names:
        log.warning("No parameters left to predict after removing auxiliary features!")
        return

    log.info(f"Configuring model to predict parameters: {parameter_names}")
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

        # Determine loss type from config (default to normalized_regression)
        loss_type = getattr(cfg.model, "loss_type", "normalized_regression")

        # Map loss_type to loss class
        if loss_type == "normalized_regression":
            loss_class = "src.models.losses.NormalizedRegressionLoss"
            log.info("Using NormalizedRegressionLoss (parameter MSE)")
        else:
            # Fall back to normalized regression for other loss types
            loss_class = "src.models.losses.NormalizedRegressionLoss"
            log.info(f"Using NormalizedRegressionLoss for loss_type={loss_type}")

        base_criteria_cfg: Dict[str, Any] = {}

        # Per-parameter losses (NormalizedRegressionLoss, etc.)
        for param_name in parameter_names:
            if param_name not in param_ranges:
                raise KeyError(f"Missing parameter range for '{param_name}' in metadata")
            param_range = param_ranges[param_name]
            base_criteria_cfg[param_name] = {
                "_target_": loss_class,
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
        # Per-parameter JND-based weights
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
        log.info(f"Auto-configured loss_weights: {cfg.model.loss_weights}")

    # Configure auxiliary_input_size if auxiliary features are present
    if auxiliary_features:
        # Check if model.net config exists and update auxiliary_input_size
        if hasattr(cfg.model, "net") and "auxiliary_input_size" in cfg.model.net:
            with open_dict(cfg.model.net):
                cfg.model.net.auxiliary_input_size = len(auxiliary_features)
            log.info(
                f"Configured model auxiliary_input_size={len(auxiliary_features)} for features: {auxiliary_features}"
            )


def _setup_datamodule_with_transforms(cfg: DictConfig) -> LightningDataModule:
    """Setup datamodule with appropriate transforms for the dataset type."""
    datamodule_kwargs = {}

    if "vimh" in cfg.data._target_.lower():
        # Use shared spectrogram transforms utility
        transforms_kwargs, log_message = create_spectrogram_transforms()
        datamodule_kwargs.update(transforms_kwargs)
        log.info(log_message)

    return hydra.utils.instantiate(cfg.data, **datamodule_kwargs)


def _preflight_check_label_diversity(
    datamodule: LightningDataModule, max_batches: int = 3
) -> None:
    """Validate that training labels vary across a few batches per head.

    Raises a ValueError if any head shows a single unique class across the sampled batches.
    """
    try:
        # Ensure setup ran so loaders are available; if setup fails, surface the error
        try:
            datamodule.setup("fit")
        except Exception as e:
            # If setup fails (e.g., dataset missing), don't proceed with preflight
            raise

        # Skip this check for regression label mode where labels are continuous
        try:
            if hasattr(datamodule, "hparams"):
                label_mode = str(
                    getattr(datamodule.hparams, "label_mode", "classification")
                ).lower()
                if label_mode == "regression":
                    log.info("Preflight skipped: regression label mode (continuous targets)")
                    return
        except Exception:
            pass

        loader = datamodule.train_dataloader()
        if loader is None:
            # No loader available yet; skip preflight quietly
            log.info("Preflight skipped: train_dataloader not available")
            return
        it = iter(loader)
        uniques: Dict[str, set] = {}
        sampled = 0
        while sampled < max_batches:
            try:
                batch = next(it)
            except StopIteration:
                break
            sampled += 1
            images, labels = batch[0], batch[1]
            for head, tens in labels.items():
                if head not in uniques:
                    uniques[head] = set()
                try:
                    if tens.ndim == 1 and tens.dtype in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                    ):
                        uniques[head].update(tens.tolist())
                except Exception:
                    # Non-scalar labels or different dtype – skip diversity check for this head
                    pass

        # Log a brief summary of unique labels observed per head
        for head in sorted(uniques.keys()):
            vals = sorted(list(uniques[head]))
            preview = ", ".join(map(str, vals[:10])) + (" …" if len(vals) > 10 else "")
            log.info(
                f"Preflight head '{head}': {len(vals)} unique label(s) across {sampled} batch(es): [{preview}]"
            )

        problems = [h for h, s in uniques.items() if len(s) <= 1]
        if problems:
            details = ", ".join(f"{h}: {sorted(list(uniques[h]))}" for h in problems)
            raise ValueError(
                f"Label preflight failed: non-diverse targets for heads [{', '.join(problems)}]. "
                f"Observed unique labels across {sampled} batch(es): {details}. "
                f"This often indicates label decoding issues."
            )
    except Exception:
        # Re-raise to be handled by the caller with existing error handling/logging
        raise


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")

    # Setup datamodule with appropriate transforms
    datamodule: LightningDataModule = _setup_datamodule_with_transforms(cfg)

    # For VIMH datasets, configure model config BEFORE instantiation (cleaner approach from /l/av)
    if (
        "vimh" in cfg.data._target_.lower()
        and hasattr(cfg.model, "auto_configure_from_dataset")
        and cfg.model.auto_configure_from_dataset
    ):
        _configure_vimh_model_config(cfg)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"MODEL PARAMS: {total_params:,} total, {trainable_params:,} trainable")

    # Log important model configuration details
    if hasattr(model, "output_mode"):
        log.info(f"Model output mode: {model.output_mode}")
    if hasattr(model, "criteria") and model.criteria:
        criteria_info = {
            name: type(criterion).__name__ for name, criterion in model.criteria.items()
        }
        log.info(f"Model loss functions: {criteria_info}")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Extract and store architecture metadata for checkpoint reconstruction
    if hasattr(model, "net"):
        metadata_extractor = ArchitectureMetadataExtractor()
        metadata_extractor.extract_and_store_metadata(model, datamodule)

    # Store experiment name in model hparams for later reference
    # This enables the evaluator to display the experiment configuration
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        experiment_name = hydra_cfg.runtime.choices.get("experiment", None)
        if experiment_name:
            model.hparams["experiment_name"] = experiment_name
            log.info(f"Stored experiment name in checkpoint: {experiment_name}")
    except Exception:
        pass  # Not critical if experiment name can't be stored

    if cfg.get("train"):
        # Preflight: ensure label diversity across a few batches before fitting
        enabled = True
        batches = 3
        try:
            if hasattr(cfg, "preflight"):
                enabled = getattr(cfg.preflight, "enabled", True)
                batches = getattr(cfg.preflight, "label_diversity_batches", 3)
        except Exception:
            pass

        if enabled:
            try:
                _preflight_check_label_diversity(datamodule, max_batches=int(batches))
                log.info("Label preflight passed (diverse targets across heads)")
            except Exception as e:
                log.error(f"Label preflight failed: {e}")
                raise
        else:
            log.info("Preflight checks disabled via config")

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.get("ckpt_path") or trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Print the key configs being used
    log.info("=" * 60)
    # Extract config names from hydra context
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        model_config = hydra_cfg.runtime.choices.get("model", "unknown")
        data_config = hydra_cfg.runtime.choices.get("data", "unknown")
        trainer_config = hydra_cfg.runtime.choices.get("trainer", "unknown")
        experiment_config = hydra_cfg.runtime.choices.get("experiment", None)
    except:
        # Fallback if hydra context not available
        model_config = "unknown"
        data_config = "unknown"
        trainer_config = "unknown"
        experiment_config = None

    log.info(f"MODEL CONFIG:     {model_config} ({cfg.model._target_})")
    data_dir = getattr(cfg.data, "data_dir", "unknown")
    # Show relative path if it's under project root
    import os

    if data_dir != "unknown" and os.path.isabs(data_dir):
        try:
            data_dir = os.path.relpath(data_dir)
        except:
            pass  # Keep original if relpath fails
    batch_size = getattr(cfg.data, "batch_size", "unknown")

    # Try to read synth_type from dataset metadata for display
    synth_type_display = "unknown"
    if data_dir != "unknown" and os.path.exists(data_dir):
        metadata_path = os.path.join(data_dir, "vimh_dataset_info.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    synth_type_display = metadata.get("synth_type", "unknown")
            except:
                pass

    log.info(f"DATA CONFIG:      {data_config} (data_dir={data_dir}, batch_size={batch_size}, synth_type={synth_type_display})")
    max_epochs = getattr(cfg.trainer, "max_epochs", "unknown")
    log.info(
        f"TRAINER CONFIG:   {trainer_config} ({cfg.trainer._target_}, max_epochs={max_epochs})"
    )
    if experiment_config:
        log.info(f"EXPERIMENT:       {experiment_config}")
    else:
        log.info(f"EXPERIMENT:       none")
    log.info(f"TAGS:             {cfg.get('tags', 'none')}")
    if cfg.get("seed"):
        log.info(f"SEED:             {cfg.seed}")
    log.info("=" * 60)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # (skip when running test-only mode since val metrics won't exist)
    if cfg.get("train", True):
        metric_value = get_metric_value(
            metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
        )
    else:
        metric_value = None

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()