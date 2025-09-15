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
from src.utils.architecture_utils import (
    ArchitectureMetadataExtractor,
    create_spectrogram_transforms,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _configure_vimh_model_config(cfg: DictConfig) -> None:
    """Configure model config for VIMH datasets before instantiation.

    This is a cleaner approach adopted from /l/av that modifies the config
    before model instantiation rather than rebuilding the model afterward.
    """
    try:
        from src.utils.vimh_utils import (
            get_heads_config_from_metadata,
            get_parameter_names_from_metadata,
            load_vimh_metadata,
        )

        parameter_names = get_parameter_names_from_metadata(cfg.data.data_dir)
        if parameter_names and hasattr(cfg.model, "net"):
            log.info(f"Configuring model with parameter names from dataset: {parameter_names}")

            # Configure model based on output mode
            if hasattr(cfg.model, "output_mode") and cfg.model.output_mode == "regression":
                # For regression mode, set parameter_names in the network config
                # Use open_dict to allow adding dynamic keys under structured configs
                with open_dict(cfg.model):
                    cfg.model.net.parameter_names = parameter_names
                    # Ensure output_mode is set at network level too
                    cfg.model.net.output_mode = "regression"
            else:
                # For classification/ordinal mode, use heads_config
                heads_config = get_heads_config_from_metadata(cfg.data.data_dir)
                with open_dict(cfg.model):
                    cfg.model.net.heads_config = heads_config

            # Auto-configure regression loss functions for each parameter
            if hasattr(cfg.model, "output_mode") and cfg.model.output_mode == "regression":
                try:
                    from src.utils.vimh_utils import get_parameter_ranges_from_metadata

                    param_ranges = get_parameter_ranges_from_metadata(cfg.data.data_dir)
                    # Build a Hydra-instantiable criteria dict instead of inserting objects
                    base_criteria_cfg: Dict[str, Any] = {}
                    for param_name in parameter_names:
                        param_range = param_ranges.get(param_name, (0.0, 1.0))
                        base_criteria_cfg[param_name] = {
                            "_target_": "src.models.losses.NormalizedRegressionLoss",
                            "param_range": (
                                tuple(param_range)
                                if isinstance(param_range, (list, tuple))
                                else (0.0, float(param_range))
                            ),
                        }

                    # If user provided extra kwargs for criteria (e.g., loss_type), merge them per head
                    merged_criteria_cfg: Dict[str, Any] = {}
                    user_criteria = getattr(cfg.model, "criteria", None)
                    for head, base_cfg in base_criteria_cfg.items():
                        merged = dict(base_cfg)
                        try:
                            if user_criteria and head in user_criteria:
                                for k, v in user_criteria[head].items():
                                    if k not in ("_target_", "param_range"):
                                        merged[k] = v
                        except Exception:
                            # If user_criteria isn't a mapping, ignore gracefully
                            pass
                        merged_criteria_cfg[head] = merged

                    with open_dict(cfg.model):
                        cfg.model.criteria = OmegaConf.create(merged_criteria_cfg)

                    log.info(
                        f"Auto-configured regression loss functions for: {list(merged_criteria_cfg.keys())}"
                    )
                except Exception as e:
                    log.warning(f"Failed to configure regression losses: {e}")

            # Auto-configure loss_weights using JND-based approach from /l/av
            if (
                not hasattr(cfg.model, "loss_weights")
                or not cfg.model.loss_weights
                or len(cfg.model.loss_weights) == 0
            ):
                try:
                    metadata = load_vimh_metadata(cfg.data.data_dir)
                    param_mappings = metadata.get("parameter_mappings", {})
                    loss_weights = {}

                    for param_name in parameter_names:
                        if param_name in param_mappings:
                            mapping = param_mappings[param_name]
                            param_range = mapping["max"] - mapping["min"]
                            step = mapping.get("step", 1.0)
                            # Weight proportional to number of JND steps
                            jnd_steps = param_range / step
                            loss_weights[param_name] = float(jnd_steps)
                        else:
                            loss_weights[param_name] = 1.0

                    # Normalize weights so maximum weight is 1.0
                    if loss_weights:
                        max_weight = max(loss_weights.values())
                        loss_weights = {
                            name: weight / max_weight for name, weight in loss_weights.items()
                        }

                    with open_dict(cfg.model):
                        cfg.model.loss_weights = loss_weights
                    log.info(
                        f"Auto-configured JND-based loss_weights (normalized): {cfg.model.loss_weights}"
                    )
                except Exception as e:
                    # Fallback to equal weights
                    with open_dict(cfg.model):
                        cfg.model.loss_weights = {name: 1.0 for name in parameter_names}
                    log.warning(f"Failed to calculate JND-based weights, using equal weights: {e}")
                    log.info(f"Auto-configured loss_weights: {cfg.model.loss_weights}")
    except Exception as e:
        log.warning(f"Failed to auto-configure model from dataset metadata: {e}")


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
    except:
        # Fallback if hydra context not available
        model_config = "unknown"
        data_config = "unknown"
        trainer_config = "unknown"

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
    log.info(f"DATA CONFIG:      {data_config} (data_dir={data_dir}, batch_size={batch_size})")
    max_epochs = getattr(cfg.trainer, "max_epochs", "unknown")
    log.info(
        f"TRAINER CONFIG:   {trainer_config} ({cfg.trainer._target_}, max_epochs={max_epochs})"
    )
    if cfg.get("experiment"):
        log.info(f"EXPERIMENT:       {cfg.experiment}")
    else:
        log.info(f"EXPERIMENT:       none")
    log.info(f"TAGS:             {cfg.get('tags', 'none')}")
    if cfg.get("seed"):
        log.info(f"SEED:             {cfg.seed}")
    log.info("=" * 60)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
