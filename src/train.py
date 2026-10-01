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
from src.utils.vimh_utils import load_vimh_metadata
from src.utils.architecture_utils import ArchitectureMetadataExtractor

log = RankedLogger(__name__, rank_zero_only=True)


def configure_vimh_run_config(cfg: DictConfig) -> None:
    """Pre-configure the model and data configs from VIMH dataset metadata.

    Must run before the datamodule and model are instantiated (train.py and eval.py
    both call it), so every entry point sees identical wiring:

    - Output mode comes from ``model.loss_type`` alone (``output_mode_for_loss_type``);
      ``data.label_mode`` is set to match (regression -> physical-unit float targets,
      classification -> class indices). A conflicting explicit ``data.label_mode`` raises.
    - Network heads = dataset parameters minus ``data.auxiliary_features`` (which are
      measured inputs, not predictions).
    - Regression: one ``NormalizedRegressionLoss`` per head with the dataset bounds,
      merged with any per-head user keys in ``model.criteria`` (e.g. ``loss_type: l1``).
    - ``model.net.auxiliary_input_size`` = number of auxiliary features.
    - Network input geometry from the dataset: ``image_size`` = [height, width] (ViT),
      ``n_channels`` / ``input_channels`` = channels, when the net config has those keys.

    Loss weights are left to ``VIMHLitModule`` (JND-based when ``model.loss_weights`` is empty).
    """
    from src.models.vimh_lit_module import output_mode_for_loss_type
    from src.utils.vimh_utils import (
        get_heads_config_from_metadata,
        get_image_dimensions_from_metadata,
        get_parameter_names_from_metadata,
        get_parameter_ranges_from_metadata,
    )

    data_dir = cfg.data.data_dir
    # "cross_entropy" is VIMHLitModule's default loss_type
    output_mode = output_mode_for_loss_type(cfg.model.get("loss_type", "cross_entropy"))

    label_mode = "regression" if output_mode == "regression" else "classification"
    explicit_label_mode = cfg.data.get("label_mode")
    if explicit_label_mode is not None and explicit_label_mode != label_mode:
        raise ValueError(
            f"data.label_mode={explicit_label_mode} conflicts with model.loss_type="
            f"{cfg.model.loss_type} ({output_mode}); remove data.label_mode or fix loss_type"
        )
    with open_dict(cfg.data):
        cfg.data.label_mode = label_mode

    auxiliary_features = list(cfg.data.get("auxiliary_features") or [])
    all_parameter_names = get_parameter_names_from_metadata(data_dir)
    unknown_aux = [a for a in auxiliary_features if a not in all_parameter_names]
    if unknown_aux:
        raise ValueError(
            f"Auxiliary features {unknown_aux} are not dataset parameters {all_parameter_names}"
        )
    parameter_names = [p for p in all_parameter_names if p not in auxiliary_features]
    if not parameter_names:
        raise ValueError(
            f"No parameters left to predict: dataset parameters {all_parameter_names}, "
            f"auxiliary features {auxiliary_features}"
        )
    if auxiliary_features:
        log.info(f"Auxiliary features (measured inputs, not predicted): {auxiliary_features}")
    log.info(f"Configuring model to predict parameters: {parameter_names} ({output_mode})")

    all_heads = get_heads_config_from_metadata(data_dir)
    heads_config = {name: all_heads[name] for name in parameter_names}
    with open_dict(cfg.model):
        if output_mode == "regression":
            cfg.model.net.parameter_names = parameter_names
            cfg.model.net.output_mode = "regression"
            cfg.model.net.heads_config = None
        else:
            cfg.model.net.heads_config = heads_config
            if "output_mode" in cfg.model.net:
                cfg.model.net.output_mode = "classification"
        # Always set (0 when unused) so a network that lacks auxiliary support fails at
        # instantiation instead of silently ignoring the auxiliary input.
        if auxiliary_features or "auxiliary_input_size" in cfg.model.net:
            cfg.model.net.auxiliary_input_size = len(auxiliary_features)
        # Input geometry from the dataset (spectrograms are often non-square, e.g. 32x64)
        height, width, channels = get_image_dimensions_from_metadata(data_dir)
        if "image_size" in cfg.model.net:  # VisionTransformer
            cfg.model.net.image_size = [height, width]
        for key in ("n_channels", "input_channels"):
            if key in cfg.model.net:
                cfg.model.net[key] = channels

    if output_mode == "regression":
        param_bounds = get_parameter_ranges_from_metadata(data_dir)
        user_criteria = cfg.model.get("criteria") or {}
        unknown_heads = [h for h in user_criteria if h not in parameter_names]
        if unknown_heads:
            raise ValueError(
                f"model.criteria has entries {unknown_heads} that are not prediction heads "
                f"{parameter_names}"
            )
        criteria_cfg: Dict[str, Any] = {}
        for head in parameter_names:
            if head not in param_bounds:
                raise KeyError(f"Missing parameter range for '{head}' in metadata")
            merged: Dict[str, Any] = {
                "_target_": "src.models.losses.NormalizedRegressionLoss",
                "param_range": tuple(param_bounds[head]),
            }
            for key, value in (user_criteria.get(head) or {}).items():
                if key in ("_target_", "param_range"):
                    raise ValueError(f"model.criteria.{head}.{key} is set from dataset metadata")
                merged[key] = value
            criteria_cfg[head] = merged
        with open_dict(cfg.model):
            cfg.model.criteria = OmegaConf.create(criteria_cfg)
        log.info(f"Auto-configured regression loss functions for: {list(criteria_cfg)}")


def _preflight_check_label_diversity(
    datamodule: LightningDataModule, max_batches: int = 3
) -> None:
    """Validate that training labels vary across a few batches per head.

    Raises a ValueError if any head shows a single unique class across the sampled batches.
    """
    datamodule.setup("fit")

    # Skip this check for regression label mode where labels are continuous
    if datamodule.hparams.label_mode == "regression":
        log.info("Preflight skipped: regression label mode (continuous targets)")
        return

    it = iter(datamodule.train_dataloader())
    uniques: Dict[str, set] = {}
    sampled = 0
    while sampled < max_batches:
        try:
            batch = next(it)
        except StopIteration:
            break
        sampled += 1
        labels = batch[1]
        for head, tens in labels.items():
            # Hard class-index targets only (soft targets are 2-D float distributions)
            if tens.ndim == 1 and not torch.is_floating_point(tens):
                uniques.setdefault(head, set()).update(tens.tolist())

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


def _link_best_checkpoint(trainer: Trainer) -> None:
    """Point ``<checkpoint dir>/best.ckpt`` at the best checkpoint of this run.

    ``ls -t epoch_*.ckpt`` finds the newest top-k checkpoint, not the best one, so
    Makefile targets and the audio evaluator use this symlink instead.
    """
    ckpt_cb = trainer.checkpoint_callback
    if ckpt_cb is None or not ckpt_cb.best_model_path or trainer.global_rank != 0:
        return
    best = os.path.abspath(ckpt_cb.best_model_path)
    link = os.path.join(os.path.dirname(best), "best.ckpt")
    if os.path.abspath(link) == best:
        return  # ModelCheckpoint(filename="best") already wrote it there
    if os.path.lexists(link):
        if not os.path.islink(link):
            raise RuntimeError(
                f"Refusing to replace real file {link} with a best-checkpoint symlink; "
                f"use a per-run checkpoint dirpath and a filename other than 'best'"
            )
        os.remove(link)
    os.symlink(os.path.basename(best), link)
    log.info(f"Best checkpoint ({ckpt_cb.monitor}={ckpt_cb.best_model_score}): {best} -> {link}")


def _select_test_checkpoint(cfg: DictConfig, trainer: Trainer) -> Optional[str]:
    """Choose the checkpoint to test.

    - After training: the best checkpoint of this run (even when resuming from
      ``ckpt_path``, which is the pre-resume state). With checkpointing disabled or
      ``fast_dev_run``, the final in-memory weights (returns None).
    - Test-only (``train=false``): ``ckpt_path`` is required.
    """
    if not cfg.get("train"):
        if not cfg.get("ckpt_path"):
            raise ValueError("test=true with train=false requires ckpt_path=<checkpoint>")
        return cfg.ckpt_path
    ckpt_cb = trainer.checkpoint_callback
    if ckpt_cb is None or trainer.fast_dev_run:
        log.warning("*** No checkpointing in this run; testing the final in-memory weights")
        return None
    if not ckpt_cb.best_model_path:
        raise RuntimeError(
            f"Training finished but no best checkpoint was saved (monitor={ckpt_cb.monitor})"
        )
    return ckpt_cb.best_model_path


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
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    # Configure model + data configs from the dataset BEFORE instantiating either
    configure_vimh_run_config(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

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

    # Store the experiment name and the fully configured (dataset-wired) model and data
    # configs in the checkpoint hparams, so a checkpoint can be rebuilt exactly without
    # the run's .hydra directory (see audio_reconstruction_eval.py).
    from hydra.core.hydra_config import HydraConfig

    if HydraConfig.initialized():
        experiment_name = HydraConfig.get().runtime.choices.get("experiment", None)
        if experiment_name:
            model.hparams["experiment_name"] = experiment_name
            log.info(f"Stored experiment name in checkpoint: {experiment_name}")
    model.hparams["run_config"] = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
    }

    if cfg.get("train"):
        # Preflight: ensure label diversity across a few batches before fitting
        preflight = cfg.get("preflight") or {}
        if preflight.get("enabled", True):
            _preflight_check_label_diversity(
                datamodule, max_batches=int(preflight.get("label_diversity_batches", 3))
            )
            log.info("Label preflight passed (diverse targets across heads)")
        else:
            log.info("Preflight checks disabled via config")

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        _link_best_checkpoint(trainer)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = _select_test_checkpoint(cfg, trainer)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Tested checkpoint: {ckpt_path or 'final in-memory weights'}")

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
    from hydra.core.hydra_config import HydraConfig

    choices = HydraConfig.get().runtime.choices
    model_config = choices.get("model", "unknown")
    data_config = choices.get("data", "unknown")
    trainer_config = choices.get("trainer", "unknown")
    experiment_config = choices.get("experiment", None)

    log.info(f"MODEL CONFIG:     {model_config} ({cfg.model._target_})")
    data_dir = os.path.relpath(cfg.data.data_dir)
    batch_size = getattr(cfg.data, "batch_size", "unknown")
    synth_type_display = load_vimh_metadata(cfg.data.data_dir).get("synth_type", "unknown")

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
    if cfg.get("seed") is not None:
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