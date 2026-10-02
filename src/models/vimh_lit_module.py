from typing import Any, Dict, Optional, Tuple

import inspect
import math
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from ..data.multihead_dataset_base import MultiheadDatasetBase
from ..utils.pylogger import RankedLogger
from .losses import (
    NormalizedRegressionLoss,
    OrdinalRegressionLoss,
    WeightedCrossEntropyLoss,
)
from .soft_target_loss import SoftTargetLoss
from .jnd_accuracy import JNDToleranceAccuracy, ExactAccuracy

# All supported loss types. ``loss_type`` is the single source of truth for the
# output mode: regression loss types get one sigmoid output per head and
# continuous (physical-unit) targets; all others get class logits and class-index targets.
LOSS_TYPES = (
    "cross_entropy",
    "ordinal_regression",
    "weighted_cross_entropy",
    "soft_target",
    "normalized_regression",
)
REGRESSION_LOSS_TYPES = ("normalized_regression",)

log = RankedLogger(__name__, rank_zero_only=True)


def output_mode_for_loss_type(loss_type: str) -> str:
    """Return ``"regression"`` or ``"classification"`` for a loss type.

    :raises ValueError: if ``loss_type`` is not one of ``LOSS_TYPES``.
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Must be one of: {', '.join(LOSS_TYPES)}")
    return "regression" if loss_type in REGRESSION_LOSS_TYPES else "classification"


class VIMHLitModule(LightningModule):
    """Lightning module for VIMH (Variable Image MultiHead) datasets.

    This module supports:
    - Multiple classification heads with different numbers of classes
    - Dynamic head configuration from dataset metadata
    - Configurable loss functions and weights per head
    - Proper metrics tracking for each head
    - Backward compatibility with single-head models

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    # Class-level configuration
    overlays = False  # Set to False for separate TensorBoard cards - True for overlays (not working yet)
    train_prefix = "" if overlays else "train/"
    val_prefix = "" if overlays else "val/"
    test_prefix = "" if overlays else "test/"

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Optional[torch.nn.Module] = None,
        criteria: Optional[Dict[str, torch.nn.Module]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        compile: bool = False,
        auto_configure_from_dataset: bool = True,
        loss_type: str = "cross_entropy",
    ) -> None:
        """Initialize a `VIMHLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param criterion: The loss function to use for training (backward compatibility).
        :param criteria: Dict of loss functions for multihead training.
        :param loss_weights: Optional weights for combining losses from different heads.
            When empty and auto-configuring, JND-based weights are computed from the
            dataset metadata (see ``_compute_jnd_weights``). When given, the keys must
            match the configured heads exactly.
        :param compile: Whether to compile the model.
        :param auto_configure_from_dataset: Whether to auto-configure heads from dataset.
        :param loss_type: Type of loss function, one of ``LOSS_TYPES``. It alone determines
            the output mode (see :func:`output_mode_for_loss_type`).
        """
        super().__init__()

        # Store for later use in setup
        self.auto_configure_from_dataset = auto_configure_from_dataset
        self._initial_criteria = criteria
        self._initial_criterion = criterion
        self._initial_loss_weights = loss_weights
        # Set once heads/criteria/weights have been configured from a dataset, so
        # later setup() calls (validate/test) never rebuild (re-randomize) trained heads.
        self._dataset_configured = False
        # Per-head (min, max) physical parameter bounds and dataset class counts,
        # set by auto-configuration.
        self.param_bounds: Dict[str, Tuple[float, float]] = {}
        self.heads_config: Dict[str, int] = {}

        self.loss_type = loss_type
        self.output_mode = output_mode_for_loss_type(loss_type)

        if criteria is None and criterion is not None:
            # Single loss for a single-head network
            net_heads = list(getattr(net, "heads_config", None) or {})
            if len(net_heads) > 1:
                raise ValueError(f"'criterion' given for a network with heads {net_heads}; use 'criteria'")
            criteria = {net_heads[0] if net_heads else "head_0": criterion}
        elif criteria is None:
            # Will be configured later in setup() if auto_configure_from_dataset is True
            if not auto_configure_from_dataset:
                raise ValueError(
                    "Must provide either 'criterion' or 'criteria' or set auto_configure_from_dataset=True"
                )
            criteria = {}

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["net", "criterion", "criteria", "scheduler"]
        )

        self.net = net
        # Handle DictConfig objects in criteria (from Hydra configuration)
        self.criteria = self._process_criteria(criteria)
        # Equal weights until dataset configuration (which uses loss_weights if given, else JND weights)
        self.loss_weights = dict(loss_weights) if loss_weights else {name: 1.0 for name in self.criteria}
        self.is_multihead = len(self.criteria) > 1

        # Store scheduler separately since it's not in hparams
        self.scheduler = scheduler

        # Will be initialized in setup()
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.val_acc_best = None

    def _create_loss_function(self, loss_type: str, num_classes: int = 256, param_range: float = 1.0, regression_loss_type: str = "mse") -> torch.nn.Module:
        """Create a loss function based on loss_type string.

        :param loss_type: One of ``LOSS_TYPES``.
        :param num_classes: Number of classes for classification-based losses
        :param param_range: Parameter range for regression-based losses
        :param regression_loss_type: Loss type for normalized regression ('mse', 'l1', 'huber')
        :return: Configured loss function
        """
        output_mode_for_loss_type(loss_type)  # validates loss_type
        if loss_type == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif loss_type == "ordinal_regression":
            return OrdinalRegressionLoss(
                num_classes=num_classes,
                param_range=param_range,
                regression_loss="l1",
                alpha=0.1
            )
        elif loss_type == "weighted_cross_entropy":
            return WeightedCrossEntropyLoss(
                num_classes=num_classes,
                distance_power=2.0,
                base_weight=1.0
            )
        elif loss_type == "soft_target":
            return SoftTargetLoss(
                num_classes=num_classes,
                mode="triangular",
                width=2
            )
        else:  # normalized_regression
            return NormalizedRegressionLoss(
                param_range=(0.0, 1.0),  # Real bounds applied by _update_criteria_with_parameter_ranges
                loss_type=regression_loss_type,
                return_perceptual_units=True
            )

    def _process_criteria(self, criteria: Dict[str, any]) -> Dict[str, torch.nn.Module]:
        """Process criteria dict, handling DictConfig objects from Hydra.

        :param criteria: Dict that may contain DictConfig objects with loss_type fields
        :return: Dict of instantiated loss functions
        """
        if not criteria:
            return {}

        from omegaconf import DictConfig
        processed_criteria = {}

        for head_name, criterion in criteria.items():
            if isinstance(criterion, DictConfig):
                # Extract loss_type from DictConfig
                if 'loss_type' in criterion:
                    loss_type_config = criterion['loss_type']

                    # Map specific regression loss types to normalized_regression
                    if loss_type_config in ['l1', 'mse', 'huber']:
                        loss_type = 'normalized_regression'
                        regression_loss_type = loss_type_config
                    else:
                        loss_type = loss_type_config
                        regression_loss_type = 'mse'  # default

                    # Create the loss function with default parameters
                    # Will be updated with proper ranges in auto-configuration
                    processed_criteria[head_name] = self._create_loss_function(
                        loss_type=loss_type,
                        num_classes=256,
                        param_range=1.0,
                        regression_loss_type=regression_loss_type
                    )
                else:
                    raise ValueError(
                        f"Criterion config for head '{head_name}' has no 'loss_type': {dict(criterion)}"
                    )
            else:
                # Already a proper loss function
                processed_criteria[head_name] = criterion

        return processed_criteria

    def _setup_metrics(self) -> None:
        """Setup metrics based on current network configuration."""
        head_configs = self._heads_config()

        # Metrics for each head
        self.train_metrics = torch.nn.ModuleDict()
        self.val_metrics = torch.nn.ModuleDict()
        self.test_metrics = torch.nn.ModuleDict()

        for head_name, num_classes in head_configs.items():
            if self.output_mode == "regression":
                # For regression, use MAE as the primary metric
                from torchmetrics.regression import MeanAbsoluteError

                self.train_metrics[f"{head_name}_mae"] = MeanAbsoluteError()
                self.val_metrics[f"{head_name}_mae"] = MeanAbsoluteError()
                self.test_metrics[f"{head_name}_mae"] = MeanAbsoluteError()

                # JND tolerance accuracies (val/test only; continuous predictions are
                # converted to class-index units)
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    self.val_metrics[metric_name] = JNDToleranceAccuracy(
                        tolerance_jnds=tolerance, num_classes=num_classes
                    )
                    self.test_metrics[metric_name] = JNDToleranceAccuracy(
                        tolerance_jnds=tolerance, num_classes=num_classes
                    )
            else:
                # For classification, use exact accuracy
                self.train_metrics[f"{head_name}_acc"] = ExactAccuracy()
                self.val_metrics[f"{head_name}_acc"] = ExactAccuracy()
                self.test_metrics[f"{head_name}_acc"] = ExactAccuracy()

                # JND tolerance accuracies (val/test only)
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    self.val_metrics[metric_name] = JNDToleranceAccuracy(
                        tolerance_jnds=tolerance, num_classes=num_classes
                    )
                    self.test_metrics[metric_name] = JNDToleranceAccuracy(
                        tolerance_jnds=tolerance, num_classes=num_classes
                    )

        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def _heads_config(self) -> Dict[str, int]:
        """Return head name -> number of classes (dataset quantization levels).

        After dataset configuration this is the dataset's class count for every head,
        also in regression mode (where the network itself has one output per head).
        """
        if self.heads_config:
            return self.heads_config
        return self._net_heads_config()

    def _net_heads_config(self) -> Dict[str, int]:
        """Return the wrapped network's heads config, failing loudly if it has none."""
        heads_config = getattr(self.net, "heads_config", None)
        if not heads_config:
            raise RuntimeError(
                f"Network {type(self.net).__name__} has no heads_config; configure heads "
                f"(auto_configure_from_dataset=true or an explicit heads_config) before use."
            )
        return heads_config

    def _setup_criteria(self) -> None:
        """Validate that criteria and loss weights cover exactly the network's heads."""
        heads = set(self._heads_config().keys())
        if set(self.criteria.keys()) != heads:
            raise RuntimeError(
                f"Loss criteria heads {sorted(self.criteria.keys())} do not match "
                f"network heads {sorted(heads)}"
            )
        if set(self.loss_weights.keys()) != heads:
            raise RuntimeError(
                f"loss_weights heads {sorted(self.loss_weights.keys())} do not match "
                f"network heads {sorted(heads)}"
            )
        self.is_multihead = len(self.criteria) > 1

    def _auto_configure_from_dataset(self, dataset: MultiheadDatasetBase) -> None:
        """Configure heads, criteria, loss weights and parameter bounds from a dataset.

        Runs the full configuration once. Later calls (Lightning calls ``setup`` again
        for validate/test) only verify that the dataset still matches, so trained
        heads are never rebuilt. The network is rebuilt only if its pre-configured
        heads differ from the dataset's (e.g. config placeholders); ``train.py``
        normally pre-configures them so no rebuild happens at all.

        Auxiliary features (measured inputs, see ``VIMHDataset.auxiliary_features``)
        are excluded from the prediction heads.

        :param dataset: The dataset to configure from
        :raises ValueError: on any mismatch between the model config and the dataset.
        """
        auxiliary = set(getattr(dataset, "auxiliary_features", None) or [])
        heads_config = {
            name: n for name, n in dataset.get_heads_config().items() if name not in auxiliary
        }
        if not heads_config:
            raise ValueError(
                f"No prediction heads left: dataset heads {list(dataset.get_heads_config())}, "
                f"auxiliary features {sorted(auxiliary)}"
            )

        if self._dataset_configured:
            if self.heads_config != heads_config:
                raise ValueError(
                    f"Dataset heads {heads_config} differ from the heads this model was "
                    f"configured with {self.heads_config}"
                )
            return

        image_shape = dataset.get_image_shape()
        if image_shape is not None:
            net_channels = self._infer_input_channels()
            if net_channels != image_shape[0]:
                raise ValueError(
                    f"Network {type(self.net).__name__} expects {net_channels} input channel(s) "
                    f"but the dataset images have shape {tuple(image_shape)} (C, H, W); fix the "
                    f"model config's input_channels/n_channels."
                )

        self.param_bounds = self._param_bounds_from_metadata(dataset, heads_config)
        self.heads_config = heads_config
        self._configure_net_heads(heads_config)
        self._configure_criteria(heads_config)
        self._configure_loss_weights(dataset, heads_config)
        self.is_multihead = len(self.criteria) > 1
        self._dataset_configured = True

    @staticmethod
    def _param_bounds_from_metadata(
        dataset: MultiheadDatasetBase, heads: Dict[str, int]
    ) -> Dict[str, Tuple[float, float]]:
        """Return (min, max) physical bounds per head from the dataset metadata."""
        mappings = (dataset.metadata_format or {}).get("parameter_mappings")
        if not mappings:
            raise ValueError("Dataset metadata has no 'parameter_mappings'")
        bounds = {}
        for name in heads:
            if name not in mappings or "min" not in mappings[name] or "max" not in mappings[name]:
                raise KeyError(f"Dataset metadata has no min/max bounds for head '{name}'")
            bounds[name] = (float(mappings[name]["min"]), float(mappings[name]["max"]))
        return bounds

    def _heads_match(self, current: Dict[str, int], desired: Dict[str, int]) -> bool:
        """Whether network heads ``current`` already implement ``desired``."""
        if self.output_mode == "regression":
            return list(current) == list(desired)  # one output per head; class counts unused
        return current == desired

    def _configure_net_heads(self, heads_config: Dict[str, int]) -> None:
        """Make the network's heads match ``heads_config`` (rebuilding only if they differ)."""
        current = getattr(self.net, "heads_config", None) or {}
        if (
            self._heads_match(current, heads_config)
            and getattr(self.net, "output_mode", self.output_mode) == self.output_mode
        ):
            return
        if not callable(getattr(self.net, "_build_heads", None)):
            raise ValueError(
                f"Network {type(self.net).__name__} heads {current} do not match dataset heads "
                f"{heads_config} and it has no _build_heads() to reconfigure them."
            )
        log.info(f"Configuring network heads from dataset: {heads_config} ({self.output_mode})")
        if hasattr(self.net, "output_mode"):
            self.net.output_mode = self.output_mode
        if hasattr(self.net, "parameter_names"):
            self.net.parameter_names = list(heads_config.keys())
        self.net._build_heads(heads_config)

    def _configure_criteria(self, heads_config: Dict[str, int]) -> None:
        """Use the user's criteria (validated) or create one per head from ``loss_type``."""
        if self._initial_criteria:
            if set(self.criteria) != set(heads_config):
                raise ValueError(
                    f"Configured criteria heads {sorted(self.criteria)} do not match dataset "
                    f"heads {sorted(heads_config)}"
                )
        else:
            self.criteria = {
                name: self._create_loss_function(
                    loss_type=self.loss_type,
                    num_classes=num_classes,
                    param_range=self.param_bounds[name][1] - self.param_bounds[name][0],
                )
                for name, num_classes in heads_config.items()
            }
        self._update_criteria_with_parameter_ranges()

    def _configure_loss_weights(
        self, dataset: MultiheadDatasetBase, heads_config: Dict[str, int]
    ) -> None:
        """Use the user's loss weights (validated) or compute JND-based ones."""
        if self._initial_loss_weights:
            weights = dict(self._initial_loss_weights)
            if set(weights) != set(heads_config):
                raise ValueError(
                    f"loss_weights heads {sorted(weights)} do not match dataset heads "
                    f"{sorted(heads_config)}"
                )
        else:
            weights = self._compute_jnd_weights(dataset.metadata_format, list(heads_config))
            log.info(f"Auto-configured JND-based loss weights: {weights}")
        self.loss_weights = weights

    def _update_criteria_with_parameter_ranges(self) -> None:
        """Apply the per-head physical parameter bounds to range-aware criteria."""
        for head_name, criterion in self.criteria.items():
            if isinstance(criterion, (OrdinalRegressionLoss, NormalizedRegressionLoss)):
                if head_name not in self.param_bounds:
                    raise KeyError(f"No parameter bounds for head '{head_name}'")
                pmin, pmax = self.param_bounds[head_name]
            if isinstance(criterion, OrdinalRegressionLoss):
                criterion.param_range = pmax - pmin
                criterion.quantization_step = criterion.param_range / (criterion.num_classes - 1)
            elif isinstance(criterion, NormalizedRegressionLoss):
                criterion.param_min, criterion.param_max = pmin, pmax
                criterion.param_range = pmax - pmin

    def _bounds_for_head(self, head_name: str) -> Tuple[float, float]:
        """Physical (min, max) bounds of a head; fails loudly if not configured."""
        if head_name not in self.param_bounds:
            raise RuntimeError(
                f"No parameter bounds for head '{head_name}'; regression needs the model to be "
                f"configured from a dataset (auto_configure_from_dataset) or param_bounds set."
            )
        return self.param_bounds[head_name]

    def _compute_predictions(
        self, logits: torch.Tensor, criterion, head_name: str
    ) -> torch.Tensor:
        """Compute predictions: physical units (regression), expected class index
        (ordinal regression) or argmax class index (other classification losses)."""
        if self.output_mode == "regression":
            # Regression heads output sigmoid-activated [0,1] values; map to physical units
            param_min, param_max = self._bounds_for_head(head_name)
            return param_min + logits.squeeze(-1) * (param_max - param_min)
        if isinstance(criterion, OrdinalRegressionLoss):
            # Weighted average of class probabilities
            probs = F.softmax(logits, dim=1)
            class_centers = torch.arange(
                criterion.num_classes, device=logits.device, dtype=torch.float32
            )
            return torch.sum(probs * class_centers.unsqueeze(0), dim=1)
        return torch.argmax(logits, dim=1)

    def _to_jnd_index_space(self, values: torch.Tensor, head_name: str) -> torch.Tensor:
        """Map regression values from physical parameter units to class-index units.

        JND tolerance is measured in quantization steps (one JND == one step), so
        :class:`JNDToleranceAccuracy` must receive class indices, not physical
        parameter values. Uses the same bounds as ``_compute_predictions`` and the
        dataset's per-head class count.

        :param values: Tensor of regression values in physical parameter units.
        :param head_name: Name of the prediction head.
        :return: Values mapped to class-index units.
        """
        param_min, param_max = self._bounds_for_head(head_name)
        num_classes = self._heads_config()[head_name]
        if num_classes < 2:
            raise RuntimeError(f"Head '{head_name}' has {num_classes} class(es); need >= 2")
        step = (param_max - param_min) / (num_classes - 1)
        return (values - param_min) / step

    def forward(self, x: torch.Tensor, auxiliary: Optional[torch.Tensor] = None):
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :param auxiliary: Optional auxiliary tensor.
        :return: A tensor of logits (single head) or dict of logits (multihead).
        """
        if "auxiliary" in inspect.signature(self.net.forward).parameters:
            return self.net(x, auxiliary)
        if auxiliary is not None:
            raise ValueError(
                f"Batch has auxiliary features but {type(self.net).__name__}.forward() takes no "
                f"auxiliary input; use a network with auxiliary support (e.g. SimpleCNN)"
            )
        return self.net(x)

    def to_torchscript(self, file_path=None, method='script', example_inputs=None):
        """Override to prevent TorchScript conversion issues with multihead dict outputs."""
        raise NotImplementedError(
            "TorchScript conversion not supported for multihead models with dict outputs. "
            "Use torch.onnx.export() instead if needed."
        )

    def on_fit_start(self) -> None:
        """Lightning hook that is called when fitting begins (before training)."""
        # Disable graph logging for VIMH models
        # TensorBoard's torch.jit.trace doesn't support dict outputs used by VIMH internally
        if hasattr(self, "logger") and self.logger is not None:
            num_heads = len(self.criteria) if self.criteria else 0

            for logger in (self.logger if isinstance(self.logger, list) else [self.logger]):
                if getattr(logger, "_log_graph", False):  # TensorBoardLogger only
                    logger._log_graph = False
                    print(
                        f"ℹ️  Disabled graph logging for VIMH model (uses dict outputs internally, {num_heads} head(s))"
                    )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        if self.val_loss is not None:
            self.val_loss.reset()
        if self.val_metrics is not None:
            for metric in self.val_metrics.values():
                metric.reset()
        if self.val_acc_best is not None:
            self.val_acc_best.reset()

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A dict of predictions per head.
            - A dict of target labels per head.
        """
        if len(batch) == 3:
            x, y, auxiliary = batch
        else:
            x, y = batch
            auxiliary = None

        logits = self.forward(x, auxiliary)

        head_names = list(self.criteria.keys())
        if not isinstance(logits, dict) or not isinstance(y, dict):
            if len(head_names) != 1:
                raise RuntimeError(
                    f"Network outputs and targets must be dicts keyed by head for "
                    f"{len(head_names)} heads {head_names}"
                )
            if not isinstance(logits, dict):
                logits = {head_names[0]: logits}
            if not isinstance(y, dict):
                y = {head_names[0]: y}
        missing = [h for h in head_names if h not in logits or h not in y]
        if missing:
            raise RuntimeError(
                f"Heads {missing} missing from network outputs {list(logits)} "
                f"or targets {list(y)}"
            )
        for h in head_names:
            self._check_target_dtype(h, y[h])

        losses = {h: self.criteria[h](logits[h], y[h]) for h in head_names}
        total_loss = sum(self.loss_weights[h] * losses[h] for h in head_names)
        preds = {
            h: self._compute_predictions(logits[h], self.criteria[h], h) for h in head_names
        }
        return total_loss, preds, {h: y[h] for h in head_names}

    def _check_target_dtype(self, head_name: str, target: torch.Tensor) -> None:
        """Fail loudly when targets are in the wrong label space for the output mode.

        Regression needs physical-unit float targets (``data.label_mode=regression``);
        classification needs integer class indices (or 2-D soft-target distributions).
        """
        is_float = torch.is_floating_point(target)
        if self.output_mode == "regression" and not is_float:
            raise TypeError(
                f"Head '{head_name}': regression needs float physical-unit targets but got "
                f"{target.dtype} class indices; set data.label_mode=regression"
            )
        if self.output_mode == "classification" and is_float and target.dim() == 1:
            raise TypeError(
                f"Head '{head_name}': classification needs integer class-index targets but got "
                f"{target.dtype}; set data.label_mode=classification"
            )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Ensure metrics are initialized
        if self.train_loss is None:
            self._setup_metrics()

        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics (JND tolerance accuracies are tracked for val/test only)
        self.train_loss(loss)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.train_metrics:
                    self.train_metrics[f"{head_name}_mae"](
                        preds_dict[head_name], targets_dict[head_name]
                    )
            else:
                if f"{head_name}_acc" in self.train_metrics:
                    self.train_metrics[f"{head_name}_acc"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

        # Log metrics
        self.log(self.train_prefix + "loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.train_metrics:
                    # Always include parameter name for consistent TensorBoard grouping
                    metric_name = self.train_prefix + f"{head_name}_mae"
                    self.log(
                        metric_name,
                        self.train_metrics[f"{head_name}_mae"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                if f"{head_name}_acc" in self.train_metrics:
                    # Always include parameter name for consistent TensorBoard grouping
                    metric_name = self.train_prefix + f"{head_name}_acc"
                    self.log(
                        metric_name,
                        self.train_metrics[f"{head_name}_acc"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # Ensure metrics are initialized
        if self.val_loss is None:
            self._setup_metrics()

        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.val_metrics:
                    self.val_metrics[f"{head_name}_mae"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

                # Update JND tolerance accuracies for regression. JND tolerance is
                # measured in quantization steps, so convert physical-unit preds and
                # targets to class-index space first.
                jnd_preds = self._to_jnd_index_space(preds_dict[head_name], head_name)
                jnd_targets = self._to_jnd_index_space(targets_dict[head_name], head_name)
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name in self.val_metrics:
                        self.val_metrics[metric_name](jnd_preds, jnd_targets)
            else:
                if f"{head_name}_acc" in self.val_metrics:
                    self.val_metrics[f"{head_name}_acc"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

                # Update JND tolerance accuracies for classification
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name in self.val_metrics:
                        self.val_metrics[metric_name](
                            preds_dict[head_name], targets_dict[head_name]
                        )

        # Log metrics
        self.log(self.val_prefix + "loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.val_metrics:
                    # Always include parameter name for consistent TensorBoard grouping
                    metric_name = self.val_prefix + f"{head_name}_mae"
                    self.log(
                        metric_name,
                        self.val_metrics[f"{head_name}_mae"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                if f"{head_name}_acc" in self.val_metrics:
                    # Always include parameter name for consistent TensorBoard grouping
                    metric_name = self.val_prefix + f"{head_name}_acc"
                    self.log(
                        metric_name,
                        self.val_metrics[f"{head_name}_acc"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            for tolerance in [1, 3, 5]:
                metric_name_base = f"{head_name}_acc_jnd{tolerance}"
                self.log(
                    self.val_prefix + metric_name_base,
                    self.val_metrics[metric_name_base],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends.

        Tracks the best epoch over ALL heads (checkpointing/early stopping monitor these):

        - classification: ``val/acc_best`` = best mean per-head exact accuracy
        - regression: ``val/mae_best`` = best mean per-head MAE normalized by that
          head's parameter range (dimensionless, so heads in different units count equally)
        """
        heads = list(self.criteria.keys())
        if self.output_mode == "regression":
            nmae = torch.stack(
                [
                    self.val_metrics[f"{h}_mae"].compute()
                    / (self._bounds_for_head(h)[1] - self._bounds_for_head(h)[0])
                    for h in heads
                ]
            ).mean()
            # MaxMetric tracks the best (lowest) normalized MAE via its negation
            self.val_acc_best(-nmae)
            self.log(self.val_prefix + "mae_best", -self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        else:
            acc = torch.stack([self.val_metrics[f"{h}_acc"].compute() for h in heads]).mean()
            self.val_acc_best(acc)
            self.log(self.val_prefix + "acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # Ensure metrics are initialized
        if self.test_loss is None:
            self._setup_metrics()

        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.test_loss(loss)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.test_metrics:
                    self.test_metrics[f"{head_name}_mae"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

                # Update JND tolerance accuracies for regression. JND tolerance is
                # measured in quantization steps, so convert physical-unit preds and
                # targets to class-index space first.
                jnd_preds = self._to_jnd_index_space(preds_dict[head_name], head_name)
                jnd_targets = self._to_jnd_index_space(targets_dict[head_name], head_name)
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name in self.test_metrics:
                        self.test_metrics[metric_name](jnd_preds, jnd_targets)
            else:
                if f"{head_name}_acc" in self.test_metrics:
                    self.test_metrics[f"{head_name}_acc"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

                # Update JND tolerance accuracies for classification
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name in self.test_metrics:
                        self.test_metrics[metric_name](
                            preds_dict[head_name], targets_dict[head_name]
                        )

        # Log metrics
        self.log(self.test_prefix + "loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.test_metrics:
                    # Always include parameter name for consistent TensorBoard grouping
                    metric_name = self.test_prefix + f"{head_name}_mae"
                    self.log(
                        metric_name,
                        self.test_metrics[f"{head_name}_mae"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

                # Log JND tolerance accuracies for regression
                for tolerance in [1, 3, 5]:
                    metric_name_base = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name_base in self.test_metrics:
                        # Always include parameter name for consistent TensorBoard grouping
                        metric_name = self.test_prefix + metric_name_base
                        self.log(
                            metric_name,
                            self.test_metrics[metric_name_base],
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,  # Don't clutter progress bar, but do log
                        )
            else:
                if f"{head_name}_acc" in self.test_metrics:
                    # Always include parameter name for consistent TensorBoard grouping
                    metric_name = self.test_prefix + f"{head_name}_acc"
                    self.log(
                        metric_name,
                        self.test_metrics[f"{head_name}_acc"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

                # Log JND tolerance accuracies for classification
                for tolerance in [1, 3, 5]:
                    metric_name_base = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name_base in self.test_metrics:
                        # Always include parameter name for consistent TensorBoard grouping
                        metric_name = self.test_prefix + metric_name_base
                        self.log(
                            metric_name,
                            self.test_metrics[metric_name_base],
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,  # Don't clutter progress bar, but do log
                        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.auto_configure_from_dataset:
            datamodule = getattr(self.trainer, "datamodule", None)
            dataset = getattr(datamodule, "data_train", None)
            if not isinstance(dataset, MultiheadDatasetBase):
                raise RuntimeError(
                    f"auto_configure_from_dataset=True needs trainer.datamodule.data_train to be "
                    f"a MultiheadDatasetBase after datamodule.setup(); got {type(dataset).__name__}"
                )
            self._auto_configure_from_dataset(dataset)

        # Validate criteria and set up metrics
        self._setup_criteria()
        self._setup_metrics()

        # Set example input for TensorBoard graph logging - infer from network
        if not hasattr(self, "example_input_array") or self.example_input_array is None:
            # Only for single-head models (TensorBoard's JIT tracer can't handle dict outputs)
            # without auxiliary input (the summary forward pass passes the image only)
            if not self.is_multihead and not getattr(self.net, "auxiliary_input_size", 0):
                example_input_shape = self._infer_example_input_shape()
                self.example_input_array = torch.randn(*example_input_shape)

        # Compile model if requested
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _infer_input_channels(self) -> int:
        """Infer the number of input channels from the network configuration."""
        # Check if network has explicit input_channels attribute
        if hasattr(self.net, "input_channels"):
            return self.net.input_channels

        # Check for n_channels attribute (VisionTransformer)
        if hasattr(self.net, "n_channels"):
            return self.net.n_channels

        # Try to infer from first convolutional layer
        if hasattr(self.net, "conv_layers") and hasattr(self.net.conv_layers, "0"):
            first_layer = self.net.conv_layers[0]
            if hasattr(first_layer, "in_channels"):
                return first_layer.in_channels

        # Check embedding layer for ViT-style networks
        if hasattr(self.net, "embedding"):
            if hasattr(self.net.embedding, "conv") and hasattr(self.net.embedding.conv, "in_channels"):
                return self.net.embedding.conv.in_channels
            # Also check conv1 for backward compatibility
            if hasattr(self.net.embedding, "conv1") and hasattr(self.net.embedding.conv1, "in_channels"):
                return self.net.embedding.conv1.in_channels

        # Default fallback
        return 1

    def _infer_example_input_shape(self) -> Tuple[int, int, int, int]:
        """Infer a reasonable example input shape for the wrapped network."""
        channels = self._infer_input_channels()
        height, width = self._infer_spatial_dims()
        return (1, channels, height, width)

    def _infer_spatial_dims(self) -> Tuple[int, int]:
        """Infer input spatial dimensions, falling back to 32x32 when unknown."""
        candidates = [
            getattr(self.net, attr, None)
            for attr in (
                "input_shape",
                "input_resolution",
                "input_size",
                "image_size",
                "img_size",
            )
        ]

        # Some modules keep spatial info on a nested embedding module
        embedding = getattr(self.net, "embedding", None)
        if embedding is not None:
            candidates.extend(
                getattr(embedding, attr, None)
                for attr in ("input_shape", "input_resolution", "image_size", "img_size")
            )

        for candidate in candidates:
            dims = self._normalize_to_hw(candidate)
            if dims is not None:
                return dims

        return (32, 32)

    @staticmethod
    def _compute_jnd_weights(metadata_format: dict, heads: list) -> Dict[str, float]:
        """Compute JND-based loss weights from dataset metadata.

        Each head's loss is weighted by the number of JND (Just Noticeable Difference)
        steps its parameter spans, (max - min) / step, normalized so the finest-resolved
        head has weight 1.0.

        :param metadata_format: Dataset metadata containing parameter mappings
        :param heads: Head (parameter) names to weight
        :return: Dictionary of head names to JND-based weights
        """
        param_mappings = (metadata_format or {}).get("parameter_mappings")
        if not param_mappings:
            raise ValueError("Dataset metadata has no 'parameter_mappings' for JND loss weights")
        steps = {}
        for name in heads:
            info = param_mappings.get(name)
            if info is None or "step" not in info:
                raise KeyError(f"Parameter '{name}' has no 'step' in metadata parameter_mappings")
            step = float(info["step"])
            if step <= 0:
                raise ValueError(f"Parameter '{name}' has non-positive step {step}")
            steps[name] = (float(info["max"]) - float(info["min"])) / step
        max_steps = max(steps.values())
        return {name: n / max_steps for name, n in steps.items()}

    @staticmethod
    def _normalize_to_hw(value: Optional[Any]) -> Optional[Tuple[int, int]]:
        """Convert assorted metadata formats into an (height, width) tuple."""
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            if len(value) == 3:
                # Assume (channels, height, width)
                return int(value[1]), int(value[2])
            if len(value) == 2:
                return int(value[0]), int(value[1])
            if len(value) == 1:
                val = int(value[0])
                return (val, val)

        if isinstance(value, int):
            if value <= 0:
                return None
            root = int(round(math.sqrt(value)))
            if root * root == value:
                return (root, root)
            return (value, value)

        if isinstance(value, torch.Size):
            return VIMHLitModule._normalize_to_hw(tuple(value))

        return None

    def get_heads_config(self) -> Dict[str, int]:
        """Get the current heads configuration.

        :return: Dictionary mapping head names to number of classes
        """
        if hasattr(self.net, "heads_config"):
            return self.net.heads_config
        return {}

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset configuration.

        :return: Dictionary with dataset information
        """
        info = {
            "heads_config": self.get_heads_config(),
            "is_multihead": self.is_multihead,
            "auto_configure_from_dataset": self.auto_configure_from_dataset,
            "criteria_keys": list(self.criteria.keys()) if self.criteria else [],
            "loss_weights": self.loss_weights,
        }
        return info


if __name__ == "__main__":
    # Basic test
    print("VIMHLitModule class created successfully")
    print("Use this module for VIMH (Variable Image MultiHead) datasets")
