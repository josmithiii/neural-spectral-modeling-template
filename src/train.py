from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Disable PyTorch 2.6 weights_only restriction for trusted LOCAL checkpoints
import os.path
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, mmap=None, **kwargs):
    # Only allow loading from local files, not URLs
    if isinstance(f, str):
        if f.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
            raise ValueError(f"Remote checkpoint loading not allowed for security: {f}")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Checkpoint file not found: {f}")
    # Force weights_only=False for trusted local research checkpoints
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, mmap=mmap, **kwargs)
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

log = RankedLogger(__name__, rank_zero_only=True)


def _preflight_check_label_diversity(datamodule: LightningDataModule, max_batches: int = 3) -> None:
    """Validate that training labels vary across a few batches per head.

    Raises a ValueError if any head shows a single unique class across the sampled batches.
    """
    try:
        # Ensure setup ran so loaders are available
        try:
            datamodule.setup("fit")
        except Exception:
            # If the Trainer will call setup later, it's still OK — we just need loaders now
            pass

        loader = datamodule.train_dataloader()
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
                    if tens.ndim == 1 and tens.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                        uniques[head].update(tens.tolist())
                except Exception:
                    # Non-scalar labels or different dtype – skip diversity check for this head
                    pass

        # Log a brief summary of unique labels observed per head
        for head in sorted(uniques.keys()):
            vals = sorted(list(uniques[head]))
            preview = ", ".join(map(str, vals[:10])) + (" …" if len(vals) > 10 else "")
            log.info(f"Preflight head '{head}': {len(vals)} unique label(s) across {sampled} batch(es): [{preview}]")

        problems = [h for h, s in uniques.items() if len(s) <= 1]
        if problems:
            details = ", ".join(f"{h}: {sorted(list(uniques[h]))}" for h in problems)
            raise ValueError(
                f"Label preflight failed: non-diverse targets for heads [{', '.join(problems)}]. "
                f"Observed unique labels across {sampled} batch(es): {details}. "
                f"This often indicates label decoding issues."
            )
    except Exception as e:
        # Re-raise with clearer context
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
    
    # For VIMH spectrogram datasets, use appropriate transforms instead of image-oriented ones
    datamodule_kwargs = {}
    if 'vimh' in cfg.data._target_.lower():
        from torchvision.transforms import transforms
        
        # Create spectrogram-appropriate transforms
        # No color jittering, rotation, or other image augmentations that don't make sense for spectrograms
        spectrogram_train_transform = transforms.Compose([
            # For now, just use identity transform - we can add spectrogram-specific augmentations later
            # No normalization either to preserve the original spectrogram values for audio reconstruction
        ])
        
        spectrogram_val_test_transform = transforms.Compose([
            # Identity transform for validation and test - no modifications
        ])
        
        datamodule_kwargs.update({
            'train_transform': spectrogram_train_transform,
            'val_transform': spectrogram_val_test_transform,
            'test_transform': spectrogram_val_test_transform
        })
        
        log.info("Using spectrogram-appropriate transforms (no image augmentations)")
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, **datamodule_kwargs)

    # For VIMH datasets, configure model with parameter names from metadata
    if 'vimh' in cfg.data._target_.lower() and hasattr(cfg.model, 'auto_configure_from_dataset') and cfg.model.auto_configure_from_dataset:
        try:
            from src.utils.vimh_utils import get_parameter_names_from_metadata, get_heads_config_from_metadata, get_image_dimensions_from_metadata
            
            # Auto-configure input channels from dataset metadata
            if hasattr(cfg.model, 'net') and hasattr(cfg.model.net, 'input_channels'):
                height, width, channels = get_image_dimensions_from_metadata(cfg.data.data_dir)
                if cfg.model.net.input_channels != channels:
                    log.info(f"Auto-configuring network input channels: {cfg.model.net.input_channels} -> {channels}")
                    cfg.model.net.input_channels = channels
            
            parameter_names = get_parameter_names_from_metadata(cfg.data.data_dir)
            if parameter_names and hasattr(cfg.model, 'net'):
                log.info(f"Configuring model with parameter names from dataset: {parameter_names}")

                # Configure model based on output mode
                if hasattr(cfg.model, 'output_mode') and cfg.model.output_mode == 'regression':
                    # For regression mode, use parameter_names
                    cfg.model.net.parameter_names = parameter_names
                else:
                    # For classification/ordinal mode, use heads_config
                    heads_config = get_heads_config_from_metadata(cfg.data.data_dir)
                    cfg.model.net.heads_config = heads_config

                # Auto-configure loss_weights (equal weight for all parameters)
                if not hasattr(cfg.model, 'loss_weights') or not cfg.model.loss_weights or len(cfg.model.loss_weights) == 0:
                    cfg.model.loss_weights = {name: 1.0 for name in parameter_names}
                    log.info(f"Auto-configured loss_weights: {cfg.model.loss_weights}")

            # Auto-configure auxiliary input size based on dataset auxiliary features
            if (hasattr(cfg.data, 'auxiliary_features') and cfg.data.auxiliary_features and 
                hasattr(cfg.model, 'net') and hasattr(cfg.model.net, 'auxiliary_input_size')):
                
                # Calculate auxiliary input size based on feature types
                auxiliary_input_size = 0
                for feature_type in cfg.data.auxiliary_features:
                    if feature_type == "decay_time":
                        auxiliary_input_size += 1  # decay_time extracts 1 feature
                    # Add other feature types as needed in the future
                
                if auxiliary_input_size > 0:
                    old_size = cfg.model.net.get('auxiliary_input_size', 0)
                    cfg.model.net.auxiliary_input_size = auxiliary_input_size
                    log.info(f"Auto-configured auxiliary_input_size: {old_size} -> {auxiliary_input_size} (based on {cfg.data.auxiliary_features})")
        except Exception as e:
            log.warning(f"Failed to auto-configure model from dataset metadata: {e}")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Log important model configuration details
    if hasattr(model, 'output_mode'):
        log.info(f"Model output mode: {model.output_mode}")
    if hasattr(model, 'criteria') and model.criteria:
        criteria_info = {name: type(criterion).__name__ for name, criterion in model.criteria.items()}
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

    if cfg.get("train"):
        # Preflight: ensure label diversity across a few batches before fitting
        try:
            _preflight_check_label_diversity(datamodule, max_batches=3)
            log.info("Label preflight passed (diverse targets across heads)")
        except Exception as e:
            log.error(f"Label preflight failed: {e}")
            raise

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
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
    log.info("="*60)
    # Extract config names from hydra context
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        model_config = hydra_cfg.runtime.choices.get('model', 'unknown')
        data_config = hydra_cfg.runtime.choices.get('data', 'unknown') 
        trainer_config = hydra_cfg.runtime.choices.get('trainer', 'unknown')
    except:
        # Fallback if hydra context not available
        model_config = 'unknown'
        data_config = 'unknown'
        trainer_config = 'unknown'
    
    log.info(f"MODEL CONFIG:     {model_config} ({cfg.model._target_})")
    data_dir = getattr(cfg.data, 'data_dir', 'unknown')
    # Show relative path if it's under project root
    import os
    if data_dir != 'unknown' and os.path.isabs(data_dir):
        try:
            data_dir = os.path.relpath(data_dir)
        except:
            pass  # Keep original if relpath fails
    log.info(f"DATA CONFIG:      {data_config} (data_dir={data_dir})")
    log.info(f"TRAINER CONFIG:   {trainer_config} ({cfg.trainer._target_})")
    if cfg.get("experiment"):
        log.info(f"EXPERIMENT:       {cfg.experiment}")
    else:
        log.info(f"EXPERIMENT:       none")
    log.info(f"TAGS:             {cfg.get('tags', 'none')}")
    if cfg.get("seed"):
        log.info(f"SEED:             {cfg.seed}")
    log.info("="*60)

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
