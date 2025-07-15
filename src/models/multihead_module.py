from typing import Any, Dict, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from ..data.multihead_dataset_base import MultiheadDatasetBase
from .losses import OrdinalRegressionLoss, QuantizedRegressionLoss, WeightedCrossEntropyLoss


class MultiheadLitModule(LightningModule):
    """Generic Lightning module for multihead classification tasks.

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
    ) -> None:
        """Initialize a `MultiheadLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param criterion: The loss function to use for training (backward compatibility).
        :param criteria: Dict of loss functions for multihead training.
        :param loss_weights: Optional weights for combining losses from different heads.
        :param compile: Whether to compile the model.
        :param auto_configure_from_dataset: Whether to auto-configure heads from dataset.
        """
        super().__init__()

        # Store for later use in setup
        self.auto_configure_from_dataset = auto_configure_from_dataset
        self._initial_criteria = criteria
        self._initial_criterion = criterion
        self._initial_loss_weights = loss_weights

        # Backward compatibility handling
        if criteria is None and criterion is not None:
            criteria = {'head_0': criterion}
        elif criteria is None:
            # Will be configured later in setup() if auto_configure_from_dataset is True
            if not auto_configure_from_dataset:
                raise ValueError("Must provide either 'criterion' or 'criteria' or set auto_configure_from_dataset=True")
            criteria = {}

        # If auto_configure_from_dataset is True but we have a network with heads_config,
        # initialize criteria based on network heads
        if auto_configure_from_dataset and not criteria and hasattr(net, 'heads_config'):
            criteria = {head_name: torch.nn.CrossEntropyLoss() for head_name in net.heads_config.keys()}

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion", "criteria", "scheduler"])

        self.net = net
        self.criteria = criteria
        self.loss_weights = loss_weights or {name: 1.0 for name in criteria.keys()}
        self.is_multihead = len(criteria) > 1

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

    def _setup_metrics(self) -> None:
        """Setup metrics based on current network configuration."""
        # Get head configurations
        if hasattr(self.net, 'heads_config'):
            head_configs = self.net.heads_config
        else:
            # Fallback for backward compatibility
            head_configs = {'head_0': 10}

        # Metrics for each head
        self.train_metrics = torch.nn.ModuleDict()
        self.val_metrics = torch.nn.ModuleDict()
        self.test_metrics = torch.nn.ModuleDict()

        for head_name, num_classes in head_configs.items():
            self.train_metrics[f"{head_name}_acc"] = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_metrics[f"{head_name}_acc"] = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_metrics[f"{head_name}_acc"] = Accuracy(task="multiclass", num_classes=num_classes)

        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def _setup_criteria(self) -> None:
        """Setup loss criteria based on current network configuration."""
        if hasattr(self.net, 'heads_config'):
            head_configs = self.net.heads_config
        else:
            # Fallback for backward compatibility
            head_configs = {'head_0': 10}

        # Initialize criteria if not already set
        if not self.criteria:
            self.criteria = {}
            for head_name in head_configs.keys():
                self.criteria[head_name] = torch.nn.CrossEntropyLoss()

        # Initialize loss weights if not already set
        if not self.loss_weights:
            self.loss_weights = {name: 1.0 for name in self.criteria.keys()}

        # Update multihead flag
        self.is_multihead = len(self.criteria) > 1

    def _auto_configure_from_dataset(self, dataset: MultiheadDatasetBase) -> None:
        """Auto-configure heads from dataset metadata.

        :param dataset: The dataset to configure from
        """
        heads_config = dataset.get_heads_config()

        # Update network heads configuration
        if hasattr(self.net, 'heads_config'):
            self.net.heads_config = heads_config
        else:
            # If network doesn't have heads_config, log a warning
            print(f"Warning: Network {type(self.net).__name__} doesn't have heads_config attribute")

        # Update criteria if using auto-configuration
        if self.auto_configure_from_dataset:
            # If criteria were pre-configured, preserve them and update with parameter ranges
            if self.criteria:
                self._update_criteria_with_parameter_ranges(dataset)
            else:
                # No pre-configured criteria, use default CrossEntropyLoss
                self.criteria = {}
                for head_name in heads_config.keys():
                    self.criteria[head_name] = torch.nn.CrossEntropyLoss()

            # Update loss weights if not already set
            if not self.loss_weights:
                self.loss_weights = {name: 1.0 for name in self.criteria.keys()}

        # Update multihead flag
        self.is_multihead = len(self.criteria) > 1

    def _update_criteria_with_parameter_ranges(self, dataset: MultiheadDatasetBase) -> None:
        """Update existing criteria with parameter ranges from dataset metadata.

        :param dataset: The dataset containing parameter range information
        """
        # Get parameter ranges from datamodule (for VIMH datasets)
        param_ranges = {}
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'param_ranges'):
            param_ranges = self.trainer.datamodule.param_ranges
        elif hasattr(dataset, 'param_ranges'):
            param_ranges = dataset.param_ranges

        # Update criteria that need parameter ranges
        for head_name, criterion in self.criteria.items():
            if isinstance(criterion, (OrdinalRegressionLoss, QuantizedRegressionLoss)):
                if head_name in param_ranges:
                    param_range = param_ranges[head_name]
                    # Update the criterion with the actual parameter range
                    criterion.param_range = param_range
                    criterion.quantization_step = param_range / (criterion.num_classes - 1)
                    print(f"Updated {head_name} loss with parameter range: {param_range}")
                else:
                    print(f"Warning: No parameter range found for {head_name}, using default: {criterion.param_range}")

    def _is_regression_loss(self, criterion) -> bool:
        """Check if a loss function is regression-based."""
        regression_losses = (
            OrdinalRegressionLoss,
            QuantizedRegressionLoss,
            torch.nn.MSELoss,
            torch.nn.L1Loss,
            torch.nn.SmoothL1Loss,
            torch.nn.HuberLoss
        )
        return isinstance(criterion, regression_losses)

    def _compute_predictions(self, logits: torch.Tensor, criterion, head_name: str) -> torch.Tensor:
        """Compute predictions based on loss function type."""
        if self._is_regression_loss(criterion):
            if isinstance(criterion, OrdinalRegressionLoss):
                # For ordinal regression, use weighted average of class probabilities
                probs = F.softmax(logits, dim=1)
                class_centers = torch.arange(criterion.num_classes, device=logits.device, dtype=torch.float32)
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

    def forward(self, x: torch.Tensor):
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits (single head) or dict of logits (multihead).
        """
        return self.net(x)

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
        x, y = batch
        logits = self.forward(x)

        if self.is_multihead:
            # Multihead case: y is dict, logits is dict
            losses = {}
            for head_name in self.criteria.keys():
                if head_name in y and head_name in logits:
                    losses[head_name] = self.criteria[head_name](logits[head_name], y[head_name])
                else:
                    print(f"Warning: Head {head_name} not found in targets or logits")

            total_loss = sum(self.loss_weights[name] * loss for name, loss in losses.items())

            preds = {
                head_name: self._compute_predictions(logits_head, self.criteria[head_name], head_name)
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
            if f"{head_name}_acc" in self.train_metrics:
                self.train_metrics[f"{head_name}_acc"](preds_dict[head_name], targets_dict[head_name])

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if f"{head_name}_acc" in self.train_metrics:
                metric_name = f"train/{head_name}_acc" if self.is_multihead else "train/acc"
                self.log(metric_name, self.train_metrics[f"{head_name}_acc"],
                        on_step=False, on_epoch=True, prog_bar=True)

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
            if f"{head_name}_acc" in self.val_metrics:
                self.val_metrics[f"{head_name}_acc"](preds_dict[head_name], targets_dict[head_name])

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if f"{head_name}_acc" in self.val_metrics:
                metric_name = f"val/{head_name}_acc" if self.is_multihead else "val/acc"
                self.log(metric_name, self.val_metrics[f"{head_name}_acc"],
                        on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
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

        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

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
            if f"{head_name}_acc" in self.test_metrics:
                self.test_metrics[f"{head_name}_acc"](preds_dict[head_name], targets_dict[head_name])

        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if f"{head_name}_acc" in self.test_metrics:
                metric_name = f"test/{head_name}_acc" if self.is_multihead else "test/acc"
                self.log(metric_name, self.test_metrics[f"{head_name}_acc"],
                        on_step=False, on_epoch=True, prog_bar=True)

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
        if self.auto_configure_from_dataset and hasattr(self.trainer, 'datamodule'):
            if hasattr(self.trainer.datamodule, 'data_train') and self.trainer.datamodule.data_train is not None:
                dataset = self.trainer.datamodule.data_train
                if isinstance(dataset, MultiheadDatasetBase):
                    self._auto_configure_from_dataset(dataset)

        # Setup criteria and metrics
        self._setup_criteria()
        self._setup_metrics()

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

    def get_heads_config(self) -> Dict[str, int]:
        """Get the current heads configuration.

        :return: Dictionary mapping head names to number of classes
        """
        if hasattr(self.net, 'heads_config'):
            return self.net.heads_config
        return {}

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset configuration.

        :return: Dictionary with dataset information
        """
        info = {
            'heads_config': self.get_heads_config(),
            'is_multihead': self.is_multihead,
            'auto_configure_from_dataset': self.auto_configure_from_dataset,
            'criteria_keys': list(self.criteria.keys()) if self.criteria else [],
            'loss_weights': self.loss_weights,
        }
        return info


if __name__ == "__main__":
    # Basic test
    print("MultiheadLitModule class created successfully")
    print("Use this module for generic multihead classification tasks")
