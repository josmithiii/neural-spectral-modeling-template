from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from ..data.multihead_dataset_base import MultiheadDatasetBase
from .losses import (
    NormalizedRegressionLoss,
    OrdinalRegressionLoss,
    QuantizedRegressionLoss,
    WeightedCrossEntropyLoss,
)
from .soft_target_loss import SoftTargetLoss
from .jnd_accuracy import JNDToleranceAccuracy, ExactAccuracy


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
        output_mode: Optional[str] = None,  # Deprecated, use loss_type instead
    ) -> None:
        """Initialize a `VIMHLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param criterion: The loss function to use for training (backward compatibility).
        :param criteria: Dict of loss functions for multihead training.
        :param loss_weights: Optional weights for combining losses from different heads.
        :param compile: Whether to compile the model.
        :param auto_configure_from_dataset: Whether to auto-configure heads from dataset.
        :param loss_type: Type of loss function - "cross_entropy", "ordinal_regression", "quantized_regression", "weighted_cross_entropy", "soft_target", "normalized_regression".
        :param output_mode: Deprecated, use loss_type instead.
        """
        super().__init__()

        # Store for later use in setup
        self.auto_configure_from_dataset = auto_configure_from_dataset
        self._initial_criteria = criteria
        self._initial_criterion = criterion
        self._initial_loss_weights = loss_weights

        # Handle backward compatibility for output_mode
        if output_mode is not None:
            if output_mode == "regression":
                loss_type = "normalized_regression"
            elif output_mode == "classification":
                loss_type = "cross_entropy"

        self.loss_type = loss_type
        # Keep output_mode for backward compatibility in other parts of code
        self.output_mode = "regression" if loss_type in ["normalized_regression", "quantized_regression"] else "classification"

        # Backward compatibility handling
        if criteria is None and criterion is not None:
            criteria = {"head_0": criterion}
        elif criteria is None:
            # Will be configured later in setup() if auto_configure_from_dataset is True
            if not auto_configure_from_dataset:
                raise ValueError(
                    "Must provide either 'criterion' or 'criteria' or set auto_configure_from_dataset=True"
                )
            criteria = {}

        # If auto_configure_from_dataset is True but we have a network with heads_config,
        # initialize placeholder criteria (will be replaced in setup with correct loss type)
        if auto_configure_from_dataset and not criteria and hasattr(net, "heads_config"):
            criteria = {
                head_name: torch.nn.CrossEntropyLoss() for head_name in net.heads_config.keys()
            }

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["net", "criterion", "criteria", "scheduler"]
        )

        self.net = net
        # Handle DictConfig objects in criteria (from Hydra configuration)
        self.criteria = self._process_criteria(criteria)
        self.loss_weights = loss_weights or {name: 1.0 for name in self.criteria.keys()}
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

        :param loss_type: One of "cross_entropy", "ordinal_regression", "quantized_regression",
                         "weighted_cross_entropy", "soft_target", "normalized_regression"
        :param num_classes: Number of classes for classification-based losses
        :param param_range: Parameter range for regression-based losses
        :param regression_loss_type: Loss type for normalized regression ('mse', 'l1', 'huber')
        :return: Configured loss function
        """
        if loss_type == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif loss_type == "ordinal_regression":
            return OrdinalRegressionLoss(
                num_classes=num_classes,
                param_range=param_range,
                regression_loss="l1",
                alpha=0.1
            )
        elif loss_type == "quantized_regression":
            return QuantizedRegressionLoss(
                num_classes=num_classes,
                param_range=param_range,
                loss_type="l1"
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
        elif loss_type == "normalized_regression":
            return NormalizedRegressionLoss(
                param_range=(0.0, 1.0),  # Will be updated by auto-configuration
                loss_type=regression_loss_type,
                return_perceptual_units=True
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of: "
                           "cross_entropy, ordinal_regression, quantized_regression, "
                           "weighted_cross_entropy, soft_target, normalized_regression")

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
                    # Fallback to default normalized regression for DictConfig without loss_type
                    processed_criteria[head_name] = self._create_loss_function(
                        loss_type='normalized_regression',
                        num_classes=256,
                        param_range=1.0
                    )
            else:
                # Already a proper loss function
                processed_criteria[head_name] = criterion

        return processed_criteria

    def _setup_metrics(self) -> None:
        """Setup metrics based on current network configuration."""
        # Get head configurations
        if hasattr(self.net, "heads_config"):
            head_configs = self.net.heads_config
        else:
            # Fallback for backward compatibility
            head_configs = {"head_0": 10}

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

                # Add JND tolerance accuracies for regression (converting continuous predictions to class indices)
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    self.train_metrics[metric_name] = JNDToleranceAccuracy(
                        tolerance_jnds=tolerance, num_classes=num_classes
                    )
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

                # Add JND tolerance accuracies for classification
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    self.train_metrics[metric_name] = JNDToleranceAccuracy(
                        tolerance_jnds=tolerance, num_classes=num_classes
                    )
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

    def _setup_criteria(self) -> None:
        """Setup loss criteria based on current network configuration."""
        if hasattr(self.net, "heads_config"):
            head_configs = self.net.heads_config
        else:
            # Fallback for backward compatibility
            head_configs = {"head_0": 10}

        # Initialize criteria if not already set
        if not self.criteria:
            self.criteria = {}
            for head_name in head_configs.keys():
                if self.output_mode == "regression":
                    self.criteria[head_name] = NormalizedRegressionLoss()
                else:
                    self.criteria[head_name] = torch.nn.CrossEntropyLoss()

        # Initialize loss weights if not already set
        if not self.loss_weights:
            self.loss_weights = {name: 1.0 for name in self.criteria.keys()}

        # Update multihead flag
        self.is_multihead = len(self.criteria) > 1

    def _auto_configure_from_dataset(self, dataset: MultiheadDatasetBase) -> None:
        """Auto-configure heads and input channels from dataset metadata.

        :param dataset: The dataset to configure from
        """
        heads_config = dataset.get_heads_config()

        # Auto-configure input channels from dataset image shape
        if hasattr(self.net, "input_channels") and hasattr(dataset, "image_shape"):
            dataset_channels = dataset.image_shape[0] if dataset.image_shape else 3
            if self.net.input_channels != dataset_channels:
                print(
                    f"Auto-configuring network input channels: {self.net.input_channels} -> {dataset_channels}"
                )
                self.net.input_channels = dataset_channels

                # If the network has already been built (has conv layers), we need to rebuild the first layer
                if hasattr(self.net, "conv_layers") and hasattr(self.net.conv_layers, "0"):
                    import torch.nn as nn

                    first_conv = self.net.conv_layers[0]
                    if hasattr(first_conv, "in_channels"):
                        # Rebuild the first convolutional layer with correct input channels
                        new_first_conv = nn.Conv2d(
                            in_channels=dataset_channels,
                            out_channels=first_conv.out_channels,
                            kernel_size=first_conv.kernel_size,
                            stride=first_conv.stride,
                            padding=first_conv.padding,
                            bias=first_conv.bias is not None,
                        )
                        self.net.conv_layers[0] = new_first_conv

        # Update network heads configuration
        if hasattr(self.net, "heads_config"):
            # Check if parameter_names was pre-configured (e.g., filtered for auxiliary features)
            if hasattr(self.net, "parameter_names") and self.net.parameter_names:
                # Filter heads_config to only include pre-configured parameters
                filtered_heads = {
                    name: heads_config[name]
                    for name in self.net.parameter_names
                    if name in heads_config
                }
                if filtered_heads:
                    print(
                        f"Using pre-configured parameter subset: {list(filtered_heads.keys())} "
                        f"(filtered from dataset: {list(heads_config.keys())})"
                    )
                    heads_config = filtered_heads

            self.net.heads_config = heads_config

            # Update parameter_names if the network has this attribute (needed for regression mode)
            if hasattr(self.net, "parameter_names"):
                self.net.parameter_names = list(heads_config.keys())

            # If the network has a _build_heads method, use it to rebuild heads
            # Call for networks that need dynamic head rebuilding
            if hasattr(self.net, "_build_heads") and callable(getattr(self.net, "_build_heads")):
                network_name = type(self.net).__name__
                if network_name in ["VisionTransformer", "SimpleCNN"]:
                    # Update the network's output_mode to match the LitModule's mode
                    if hasattr(self.net, "output_mode"):
                        self.net.output_mode = self.output_mode
                    self.net._build_heads(heads_config)
                    # Update the network's multihead flag after rebuilding
                    if hasattr(self.net, "is_multihead"):
                        self.net.is_multihead = len(heads_config) > 1
        else:
            # If network doesn't have heads_config, log a warning
            print(
                f"Warning: Network {type(self.net).__name__} doesn't have heads_config attribute"
            )

        # Auto-configure JND-based loss weights if available
        # Only apply JND weights if user hasn't explicitly set weights (all weights == 1.0)
        user_set_weights = not all(w == 1.0 for w in self.loss_weights.values())

        if hasattr(dataset, 'metadata_format') and dataset.metadata_format:
            jnd_weights = self._compute_jnd_weights(dataset.metadata_format)
            if jnd_weights and not user_set_weights:
                print(f"Auto-configuring JND-based loss weights: {jnd_weights}")
                self.loss_weights.update(jnd_weights)
            elif jnd_weights and user_set_weights:
                print(f"Skipping JND weight auto-configuration (user provided explicit weights: {self.loss_weights})")

        # Update criteria if using auto-configuration
        if self.auto_configure_from_dataset:
            # Check if we have hardcoded placeholder heads that need replacement
            placeholder_heads = ["digit", "synth_param1"]
            has_hardcoded_placeholder = False
            for placeholder in placeholder_heads:
                if (
                    self.criteria
                    and placeholder in self.criteria
                    and placeholder not in heads_config
                ):
                    has_hardcoded_placeholder = True
                    break

            # Check if criteria were auto-generated (user didn't provide initial criteria)
            criteria_were_auto_generated = self._initial_criteria is None

            if self.criteria and not has_hardcoded_placeholder and not criteria_were_auto_generated:
                # If criteria were pre-configured (and not placeholders), preserve them and update with parameter ranges
                self._update_criteria_with_parameter_ranges(dataset)
            else:
                # No pre-configured criteria or has placeholders - replace with dataset heads using loss_type
                self.criteria = {}
                for head_name, num_classes in heads_config.items():
                    # Get parameter range for this head
                    param_range = self._get_param_range_for_head(dataset, head_name)
                    # Create loss function based on loss_type
                    self.criteria[head_name] = self._create_loss_function(
                        loss_type=self.loss_type,
                        num_classes=num_classes,
                        param_range=param_range
                    )
                # NormalizedRegressionLoss is created with placeholder bounds (0, 1)
                # by _create_loss_function (param_range is ignored there because it
                # needs (min, max) bounds, not a range size). Apply the real per-head
                # bounds from the dataset so perceptual-unit scaling is correct.
                self._update_criteria_with_parameter_ranges(dataset)

            # Update loss weights - reset them if we replaced criteria due to hardcoded placeholder
            if not self.loss_weights or has_hardcoded_placeholder:
                self.loss_weights = {name: 1.0 for name in self.criteria.keys()}

        # Update multihead flag
        self.is_multihead = len(self.criteria) > 1

    def _update_criteria_with_parameter_ranges(self, dataset: MultiheadDatasetBase) -> None:
        """Update existing criteria with parameter ranges from dataset metadata.

        :param dataset: The dataset containing parameter range information
        """
        # Get parameter ranges and bounds from datamodule (for VIMH datasets)
        param_ranges = {}
        param_bounds = {}
        try:
            if hasattr(self.trainer, "datamodule"):
                if hasattr(self.trainer.datamodule, "param_ranges"):
                    param_ranges = self.trainer.datamodule.param_ranges
                if hasattr(self.trainer.datamodule, "param_bounds"):
                    param_bounds = self.trainer.datamodule.param_bounds
        except RuntimeError:
            # No trainer attached, try dataset directly
            pass

        if not param_ranges and hasattr(dataset, "param_ranges"):
            param_ranges = dataset.param_ranges
            if hasattr(dataset, "param_bounds"):
                param_bounds = dataset.param_bounds

        # Update criteria that need parameter ranges or bounds
        for head_name, criterion in self.criteria.items():
            if isinstance(criterion, (OrdinalRegressionLoss, QuantizedRegressionLoss)):
                if head_name in param_ranges:
                    param_range = param_ranges[head_name]
                    # Update the criterion with the actual parameter range
                    criterion.param_range = param_range
                    criterion.quantization_step = param_range / (criterion.num_classes - 1)
                    print(f"Updated {head_name} loss with parameter range: {param_range}")
                else:
                    print(
                        f"Warning: No parameter range found for {head_name}, using default: {criterion.param_range}"
                    )
            elif isinstance(criterion, NormalizedRegressionLoss):
                if head_name in param_bounds:
                    param_bound = param_bounds[head_name]
                    # Update the criterion with the actual parameter bounds
                    criterion.param_min, criterion.param_max = param_bound
                    criterion.param_range = criterion.param_max - criterion.param_min
                    print(
                        f"Updated {head_name} regression loss with parameter bounds: {param_bound}"
                    )
                else:
                    print(
                        f"Warning: No parameter bounds found for {head_name}, using default: ({criterion.param_min}, {criterion.param_max})"
                    )

    def _get_param_range_for_head(
        self, dataset: MultiheadDatasetBase, head_name: str
    ) -> float:
        """Get parameter range for a specific head from dataset metadata.

        :param dataset: The dataset containing parameter range information
        :param head_name: Name of the parameter head
        :return: Parameter range size (max - min) as float
        """
        # Try to get parameter ranges from datamodule first
        param_ranges = {}
        try:
            if hasattr(self.trainer, "datamodule"):
                if hasattr(self.trainer.datamodule, "param_ranges"):
                    param_ranges = self.trainer.datamodule.param_ranges
        except RuntimeError:
            # No trainer attached, try dataset directly
            pass

        # If not found in datamodule, try dataset directly
        if not param_ranges and hasattr(dataset, "param_ranges"):
            param_ranges = dataset.param_ranges

        # Look for the specific parameter range
        if head_name in param_ranges:
            param_range = param_ranges[head_name]
            if isinstance(param_range, (tuple, list)) and len(param_range) == 2:
                # Return range size (max - min)
                return float(param_range[1] - param_range[0])
            elif isinstance(param_range, (int, float)):
                # Already a range size
                return float(param_range)
            else:
                raise ValueError(
                    f"Parameter range for '{head_name}' must be a number or (min, max) tuple; received {param_range}."
                )

        # Try to get from dataset metadata if it's a VIMH dataset
        if hasattr(dataset, "_metadata") and dataset._metadata:
            param_mappings = dataset._metadata.get("parameter_mappings", {})
            if head_name in param_mappings:
                mapping = param_mappings[head_name]
                if "min" in mapping and "max" in mapping:
                    # Return range size (max - min)
                    return float(mapping["max"] - mapping["min"])
                raise KeyError(
                    f"Parameter '{head_name}' metadata missing 'min' or 'max' bounds."
                )

        # Default fallback
        print(f"Warning: No parameter range information available for head '{head_name}', using default: 1.0")
        return 1.0

    def _is_regression_loss(self, criterion) -> bool:
        """Check if a loss function is regression-based."""
        regression_losses = (
            OrdinalRegressionLoss,
            QuantizedRegressionLoss,
            NormalizedRegressionLoss,
        )
        return isinstance(criterion, regression_losses)

    def _compute_predictions(
        self, logits: torch.Tensor, criterion, head_name: str
    ) -> torch.Tensor:
        """Compute predictions based on loss function type."""
        if self.output_mode == "regression":
            # For pure regression mode, logits are already sigmoid-activated [0,1] values
            # Convert to parameter space using datamodule bounds
            normalized_preds = logits.squeeze(-1)

            # Get parameter bounds for denormalization
            param_bounds = None
            try:
                if hasattr(self.trainer, "datamodule") and hasattr(
                    self.trainer.datamodule, "param_bounds"
                ):
                    param_bounds = self.trainer.datamodule.param_bounds
            except RuntimeError:
                # No trainer attached, use fallback
                param_bounds = None

            if not param_bounds or head_name not in param_bounds:
                # Fall back to normalized predictions when bounds are unavailable
                preds = normalized_preds
            else:
                param_min, param_max = param_bounds[head_name]
                preds = param_min + normalized_preds * (param_max - param_min)
        elif self._is_regression_loss(criterion):
            if isinstance(criterion, OrdinalRegressionLoss):
                # For ordinal regression, use weighted average of class probabilities
                probs = F.softmax(logits, dim=1)
                class_centers = torch.arange(
                    criterion.num_classes, device=logits.device, dtype=torch.float32
                )
                preds = torch.sum(probs * class_centers.unsqueeze(0), dim=1)
            elif isinstance(criterion, QuantizedRegressionLoss):
                # For quantized regression, output should be single continuous value
                preds = logits.squeeze(-1) if logits.dim() > 1 else logits
                preds = torch.clamp(preds, 0, criterion.num_classes - 1)
            else:
                # For standard regression losses, assume single output
                preds = logits.squeeze(-1) if logits.dim() > 1 else logits
        else:
            # Classification: use argmax
            preds = torch.argmax(logits, dim=1)

        return preds

    def _to_jnd_index_space(self, values: torch.Tensor, head_name: str) -> torch.Tensor:
        """Map regression values from physical parameter units to class-index units.

        JND tolerance is measured in quantization steps (one JND == one step), so
        :class:`JNDToleranceAccuracy` must receive class indices, not physical
        parameter values. In regression mode both predictions (see
        ``_compute_predictions``) and targets (datamodule target transform) are in
        physical units, which made ``*_acc_jnd*`` meaningless (e.g. a head spanning
        [0, 0.9] collapsed to indices 0/1). Convert with the same datamodule bounds
        and per-head class count used to denormalize predictions.

        :param values: Tensor of regression values in physical parameter units.
        :param head_name: Name of the prediction head.
        :return: Values mapped to class-index units (unchanged if bounds unavailable,
            in which case predictions are already in normalized [0, 1] space).
        """
        param_bounds = None
        try:
            if hasattr(self.trainer, "datamodule") and hasattr(
                self.trainer.datamodule, "param_bounds"
            ):
                param_bounds = self.trainer.datamodule.param_bounds
        except RuntimeError:
            param_bounds = None

        heads_config = getattr(self.net, "heads_config", {})
        if not param_bounds or head_name not in param_bounds or head_name not in heads_config:
            return values

        param_min, param_max = param_bounds[head_name]
        num_classes = heads_config[head_name]
        step = (param_max - param_min) / max(1, (num_classes - 1))
        if step == 0:
            return values
        return (values - param_min) / step

    def forward(self, x: torch.Tensor, auxiliary: Optional[torch.Tensor] = None):
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :param auxiliary: Optional auxiliary tensor.
        :return: A tensor of logits (single head) or dict of logits (multihead).
        """
        if hasattr(self.net, "forward") and "auxiliary" in self.net.forward.__code__.co_varnames:
            return self.net(x, auxiliary)
        else:
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
                if hasattr(logger, "log_graph") and logger._log_graph:
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
            # Backward compatibility
            x, y = batch
            auxiliary = None

        logits = self.forward(x, auxiliary)

        if self.is_multihead:
            # Multihead case: y is dict, logits should be dict
            if isinstance(logits, dict):
                # True multihead: logits is dict
                losses = {}
                for head_name in self.criteria.keys():
                    if head_name in y and head_name in logits:
                        losses[head_name] = self.criteria[head_name](
                            logits[head_name], y[head_name]
                        )
                    else:
                        print(f"Warning: Head {head_name} not found in targets or logits")
            else:
                # Regression mode with single tensor output: split logits by parameter
                # logits shape: [batch_size, num_parameters]
                losses = {}
                logits_dict = {}
                param_names = list(self.criteria.keys())

                for i, head_name in enumerate(param_names):
                    if i < logits.shape[1] and head_name in y:
                        logits_dict[head_name] = logits[:, i : i + 1]  # Keep shape [batch_size, 1]
                        losses[head_name] = self.criteria[head_name](
                            logits_dict[head_name].squeeze(-1), y[head_name].float()
                        )
                    else:
                        print(f"Warning: Head {head_name} not found in targets or logits")

                # Update logits to dict format for consistency
                logits = logits_dict

            total_loss = sum(self.loss_weights[name] * loss for name, loss in losses.items())

            preds = {
                head_name: self._compute_predictions(
                    logits_head, self.criteria[head_name], head_name
                )
                for head_name, logits_head in logits.items()
            }

            return total_loss, preds, y
        else:
            # Single head case: y is tensor, logits is tensor (backward compatibility)
            head_name = next(iter(self.criteria.keys()))
            if isinstance(y, dict):
                # Extract single head from dict
                y_single = y[head_name]
            else:
                y_single = y

            if isinstance(logits, dict):
                # Extract single head from dict
                logits_single = logits[head_name]
            else:
                logits_single = logits

            loss = self.criteria[head_name](logits_single, y_single)
            preds = self._compute_predictions(logits_single, self.criteria[head_name], head_name)

            return loss, {head_name: preds}, {head_name: y_single}

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

        # Update metrics
        self.train_loss(loss)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.train_metrics:
                    self.train_metrics[f"{head_name}_mae"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

                # Update JND tolerance accuracies for regression. JND tolerance is
                # measured in quantization steps, so convert physical-unit preds and
                # targets to class-index space first.
                jnd_preds = self._to_jnd_index_space(preds_dict[head_name], head_name)
                jnd_targets = self._to_jnd_index_space(targets_dict[head_name], head_name)
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name in self.train_metrics:
                        self.train_metrics[metric_name](jnd_preds, jnd_targets)
            else:
                if f"{head_name}_acc" in self.train_metrics:
                    self.train_metrics[f"{head_name}_acc"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

                # Update JND tolerance accuracies for classification
                for tolerance in [1, 3, 5]:
                    metric_name = f"{head_name}_acc_jnd{tolerance}"
                    if metric_name in self.train_metrics:
                        self.train_metrics[metric_name](
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

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        if self.output_mode == "regression":
            # For regression mode, track best MAE (lower is better) of primary task
            primary_head = next(iter(self.criteria.keys()))
            if f"{primary_head}_mae" in self.val_metrics:
                mae = self.val_metrics[f"{primary_head}_mae"].compute()
                # Note: for MAE, we want to track the minimum (best), so we negate it
                self.val_acc_best(
                    -mae
                )  # Store negative MAE so MaxMetric tracks the best (lowest) MAE
            self.log(self.val_prefix + "mae_best", -self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        else:
            # For classification mode, track best accuracy
            if self.is_multihead:
                # For multihead, track best accuracy of primary task (first head)
                primary_head = next(iter(self.criteria.keys()))
                if f"{primary_head}_acc" in self.val_metrics:
                    acc = self.val_metrics[f"{primary_head}_acc"].compute()
                    self.val_acc_best(acc)
            else:
                # For single head, track the only accuracy
                head_name = next(iter(self.criteria.keys()))
                if f"{head_name}_acc" in self.val_metrics:
                    acc = self.val_metrics[f"{head_name}_acc"].compute()
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
        # Auto-configure from dataset if enabled
        if self.auto_configure_from_dataset and hasattr(self.trainer, "datamodule"):
            if (
                hasattr(self.trainer.datamodule, "data_train")
                and self.trainer.datamodule.data_train is not None
            ):
                dataset = self.trainer.datamodule.data_train
                if isinstance(dataset, MultiheadDatasetBase):
                    self._auto_configure_from_dataset(dataset)

                    # Set example_input_array from dataset for TensorBoard computational graph
                    # Note: Disabled for multihead models due to dict output incompatibility with JIT tracer
                    if (
                        not hasattr(self, "example_input_array")
                        or self.example_input_array is None
                    ):
                        if len(dataset.get_heads_config()) == 1:
                            # Only enable for single-head models to avoid JIT tracer dict output issues
                            try:
                                sample = dataset[0]
                                if isinstance(sample, (tuple, list)) and len(sample) > 0:
                                    # Use actual sample from dataset with batch dimension
                                    self.example_input_array = sample[0].unsqueeze(0)
                            except Exception:
                                # Fallback based on dataset metadata if available
                                if hasattr(dataset, "image_shape"):
                                    self.example_input_array = torch.randn(1, *dataset.image_shape)
                                else:
                                    self.example_input_array = torch.randn(1, 1, 32, 32)

        # Setup criteria and metrics
        self._setup_criteria()
        self._setup_metrics()

        # Set example input for TensorBoard graph logging - infer from network
        if not hasattr(self, "example_input_array") or self.example_input_array is None:
            # Only set for single-head models to avoid TensorBoard JIT tracer issues with dict outputs
            if not self.is_multihead:
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

    def _compute_jnd_weights(self, metadata_format: dict) -> dict:
        """Compute JND-based loss weights from dataset metadata.

        Weight each head's loss by the number of JNDs (Just Noticeable Differences).
        For classification: JNDs = 1 + (max-min)/step, where step=1 for discrete classes
        For regression: JNDs = 1 + (max-min)/step, using provided step size

        :param metadata_format: Dataset metadata containing parameter mappings
        :return: Dictionary of head names to JND-based weights
        """
        jnd_weights = {}

        if 'parameter_mappings' not in metadata_format:
            return jnd_weights

        param_mappings = metadata_format['parameter_mappings']

        for param_name, param_info in param_mappings.items():
            param_min = param_info.get('min', 0)
            param_max = param_info.get('max', 1)
            param_step = param_info.get('step', None)

            # For parameters with step=0, they're constant (no variation)
            if param_step is None or param_step <= 0:
                jnd_weights[param_name] = 1.0
            else:
                # Compute number of JNDs: 1 + (max-min)/step
                num_jnds = 1 + (param_max - param_min) / param_step
                jnd_weights[param_name] = float(num_jnds)

        return jnd_weights

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
