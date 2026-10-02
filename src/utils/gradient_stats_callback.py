"""Callback for tracking gradient statistics across batches and epochs."""

from typing import Dict, List

import lightning as L
import numpy as np
import torch


class GradientStatsCallback(L.Callback):
    """Callback to log detailed gradient statistics during training.

    Tracks the distribution of per-parameter-tensor gradient norms to help analyze
    training dynamics and inform batch size / optimization choices.

    Cost: one device-to-host copy of the per-tensor norms per step (needed for the
    epoch statistics); the full gradient histogram is gathered only on logging steps.
    """

    def __init__(
        self,
        log_every_n_steps: int = 10,
        log_histogram: bool = True,
        track_layer_gradients: bool = True,
    ):
        """Initialize gradient stats callback.

        Args:
            log_every_n_steps: How often to log gradient stats
            log_histogram: Whether to log gradient histograms to tensorboard
            track_layer_gradients: Whether to track per-layer gradient stats
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_histogram = log_histogram
        self.track_layer_gradients = track_layer_gradients

        # Track gradient stats across steps
        self.gradient_norms: List[float] = []
        self.step_count = 0

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log gradient statistics before optimizer step."""
        self.step_count += 1

        named_grads = [(n, p.grad) for n, p in pl_module.named_parameters() if p.grad is not None]
        if not named_grads:
            return

        # Per-tensor norms computed on the device, copied to the host once
        grad_norms = torch.stack([g.detach().norm() for _, g in named_grads]).cpu().numpy()

        grad_norm_mean = float(np.mean(grad_norms))
        self.gradient_norms.append(grad_norm_mean)  # for epoch-level statistics

        if self.step_count % self.log_every_n_steps != 0:
            return

        grad_norm_std = float(np.std(grad_norms))
        pl_module.log("grad_stats/norm_mean", grad_norm_mean, on_step=True, on_epoch=False)
        pl_module.log("grad_stats/norm_std", grad_norm_std, on_step=True, on_epoch=False)
        pl_module.log("grad_stats/norm_max", float(np.max(grad_norms)), on_step=True, on_epoch=False)
        pl_module.log("grad_stats/norm_min", float(np.min(grad_norms)), on_step=True, on_epoch=False)
        pl_module.log("grad_stats/variance", float(np.var(grad_norms)), on_step=True, on_epoch=False)
        # Spread of norms across parameter tensors (not a gradient signal-to-noise
        # ratio, which would need per-sample gradients)
        if grad_norm_std > 0:
            pl_module.log(
                "grad_stats/norm_mean_over_std",
                grad_norm_mean / grad_norm_std,
                on_step=True,
                on_epoch=False,
            )

        if self.track_layer_gradients:
            layer_stats: Dict[str, List[float]] = {}
            for (name, _), norm in zip(named_grads, grad_norms):
                layer_stats.setdefault(name.split(".")[0], []).append(float(norm))
            for layer_name, layer_grads in layer_stats.items():
                pl_module.log(
                    f"grad_stats/layer_{layer_name}_mean",
                    float(np.mean(layer_grads)),
                    on_step=True,
                    on_epoch=False,
                )

        # Gradient histogram (TensorBoardLogger only), gathered only on logging steps
        experiment = getattr(pl_module.logger, "experiment", None)
        if self.log_histogram and hasattr(experiment, "add_histogram"):
            all_grads = torch.cat([g.detach().flatten() for _, g in named_grads]).cpu()
            experiment.add_histogram("gradients/all_params", all_grads, global_step=trainer.global_step)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log epoch-level gradient statistics."""
        if not self.gradient_norms:
            return

        # Epoch-level gradient statistics
        epoch_grad_mean = np.mean(self.gradient_norms)
        epoch_grad_std = np.std(self.gradient_norms)
        epoch_grad_stability = epoch_grad_std / epoch_grad_mean if epoch_grad_mean > 0 else 0

        pl_module.log("grad_stats/epoch_mean", epoch_grad_mean, on_step=False, on_epoch=True)
        pl_module.log("grad_stats/epoch_std", epoch_grad_std, on_step=False, on_epoch=True)
        pl_module.log(
            "grad_stats/epoch_stability", epoch_grad_stability, on_step=False, on_epoch=True
        )

        # Reset for next epoch
        self.gradient_norms = []
