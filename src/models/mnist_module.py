from typing import Any, Dict, Tuple, Optional, Union

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class MNISTLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

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
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param criterion: The loss function to use for training (backward compatibility).
        :param criteria: Dict of loss functions for multihead training.
        :param loss_weights: Optional weights for combining losses from different heads.
        :param compile: Whether to compile the model.
        """
        super().__init__()

        # Backward compatibility handling
        if criteria is None and criterion is not None:
            criteria = {'digit': criterion}
        elif criteria is None:
            raise ValueError("Must provide either 'criterion' or 'criteria'")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion", "criteria"])

        self.net = net
        self.criteria = criteria
        self.loss_weights = loss_weights or {name: 1.0 for name in criteria.keys()}
        self.is_multihead = len(criteria) > 1

        # Dynamic metric creation based on network heads config
        if hasattr(net, 'heads_config'):
            head_configs = net.heads_config
        else:
            # Fallback for backward compatibility
            head_configs = {'digit': 10}

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
        self.val_loss.reset()
        for metric in self.val_metrics.values():
            metric.reset()
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
                losses[head_name] = self.criteria[head_name](logits[head_name], y[head_name])

            total_loss = sum(self.loss_weights[name] * loss for name, loss in losses.items())

            preds = {
                head_name: torch.argmax(logits_head, dim=1)
                for head_name, logits_head in logits.items()
            }

            return total_loss, preds, y
        else:
            # Single head case: y is tensor, logits is tensor (backward compatibility)
            head_name = next(iter(self.criteria.keys()))
            loss = self.criteria[head_name](logits, y)
            preds = torch.argmax(logits, dim=1)

            return loss, {head_name: preds}, {head_name: y}

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)
        for head_name in preds_dict.keys():
            self.train_metrics[f"{head_name}_acc"](preds_dict[head_name], targets_dict[head_name])

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
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
        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        for head_name in preds_dict.keys():
            self.val_metrics[f"{head_name}_acc"](preds_dict[head_name], targets_dict[head_name])

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            metric_name = f"val/{head_name}_acc" if self.is_multihead else "val/acc"
            self.log(metric_name, self.val_metrics[f"{head_name}_acc"],
                    on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        if self.is_multihead:
            # For multihead, track best accuracy of primary task (first head)
            primary_head = next(iter(self.criteria.keys()))
            acc = self.val_metrics[f"{primary_head}_acc"].compute()
        else:
            # For single head, track the only accuracy
            head_name = next(iter(self.criteria.keys()))
            acc = self.val_metrics[f"{head_name}_acc"].compute()

        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.test_loss(loss)
        for head_name in preds_dict.keys():
            self.test_metrics[f"{head_name}_acc"](preds_dict[head_name], targets_dict[head_name])

        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
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
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None, None)
