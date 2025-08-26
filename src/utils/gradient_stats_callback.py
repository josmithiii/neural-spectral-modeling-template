"""Callback for tracking gradient statistics across batches and epochs."""

import lightning as L
import torch
import numpy as np
from typing import Any, Dict, Optional


class GradientStatsCallback(L.Callback):
    """Callback to log detailed gradient statistics during training.

    Tracks gradient norms, variance, and other statistics to help analyze
    training dynamics and inform batch size / optimization choices.
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
        self.gradient_norms = []
        self.step_count = 0

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log gradient statistics before optimizer step."""
        self.step_count += 1

        # Collect gradient norms for all parameters
        grad_norms = []
        grad_values = []
        layer_stats = {}

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                if self.track_layer_gradients:
                    # Track per-layer statistics
                    layer_name = name.split('.')[0]  # Get top-level module name
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = []
                    layer_stats[layer_name].append(grad_norm)

                # Collect gradient values for histogram
                if self.log_histogram:
                    grad_values.extend(param.grad.flatten().detach().cpu().numpy())

        if not grad_norms:
            return

        # Calculate overall gradient statistics
        grad_norm_mean = np.mean(grad_norms)
        grad_norm_std = np.std(grad_norms)
        grad_norm_max = np.max(grad_norms)
        grad_norm_min = np.min(grad_norms)

        # Store for epoch-level statistics
        self.gradient_norms.append(grad_norm_mean)

        # Log every N steps
        if self.step_count % self.log_every_n_steps == 0:
            # Overall gradient stats
            pl_module.log("grad_stats/norm_mean", grad_norm_mean, on_step=True, on_epoch=False)
            pl_module.log("grad_stats/norm_std", grad_norm_std, on_step=True, on_epoch=False)
            pl_module.log("grad_stats/norm_max", grad_norm_max, on_step=True, on_epoch=False)
            pl_module.log("grad_stats/norm_min", grad_norm_min, on_step=True, on_epoch=False)

            # Signal-to-noise ratio (higher is better)
            if grad_norm_std > 0:
                snr = grad_norm_mean / grad_norm_std
                pl_module.log("grad_stats/signal_noise_ratio", snr, on_step=True, on_epoch=False)

            # Gradient variance (lower usually better for convergence)
            grad_variance = np.var(grad_norms)
            pl_module.log("grad_stats/variance", grad_variance, on_step=True, on_epoch=False)

            # Per-layer gradient stats
            if self.track_layer_gradients:
                for layer_name, layer_grads in layer_stats.items():
                    layer_mean = np.mean(layer_grads)
                    pl_module.log(f"grad_stats/layer_{layer_name}_mean", layer_mean, on_step=True, on_epoch=False)

            # Log gradient histogram to tensorboard
            if self.log_histogram and hasattr(pl_module.logger, 'experiment'):
                try:
                    # For TensorBoardLogger
                    if hasattr(pl_module.logger.experiment, 'add_histogram'):
                        pl_module.logger.experiment.add_histogram(
                            "gradients/all_params",
                            np.array(grad_values),
                            global_step=trainer.global_step
                        )
                except Exception:
                    # Silently continue if histogram logging fails
                    pass

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
        pl_module.log("grad_stats/epoch_stability", epoch_grad_stability, on_step=False, on_epoch=True)

        # Reset for next epoch
        self.gradient_norms = []
