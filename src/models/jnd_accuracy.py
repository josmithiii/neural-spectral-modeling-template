"""JND-based accuracy metrics for synthesizer parameter evaluation."""

import torch
import torch.nn as nn
from torchmetrics import Metric
from typing import Optional


class JNDToleranceAccuracy(Metric):
    """
    Just Noticeable Difference (JND) tolerance-based accuracy metric.

    Uses triangular tolerance: accuracy = 1.0 at exact match, linearly decreasing
    to 0.0 at ±tolerance_jnds away. Designed for synthesizer parameters where
    each quantization step represents one JND.

    Args:
        tolerance_jnds: Number of JNDs (quantization steps) for full tolerance window
                       Total triangle width = 2 * tolerance_jnds + 1
                       Default 5 gives 11-step triangle (±5 JNDs)
        num_classes: Number of parameter quantization levels (for clamping)
    """

    def __init__(
        self,
        tolerance_jnds: int = 5,
        num_classes: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tolerance_jnds = tolerance_jnds
        self.num_classes = num_classes

        # State variables
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update accuracy state with new predictions and targets.

        Args:
            preds: Either raw logits [batch_size, num_classes] or continuous predictions [batch_size]
            target: Target indices [batch_size]
        """
        # Handle both logits and continuous predictions
        if preds.dim() > 1 and preds.shape[1] > 1:
            # Get predicted class indices from logits
            pred_indices = torch.argmax(preds, dim=1)
        else:
            # Already continuous predictions - round to nearest class index
            pred_indices = torch.round(preds.squeeze(-1) if preds.dim() > 1 else preds)

        # Calculate absolute distance in quantization steps (JNDs)
        distances = torch.abs(pred_indices - target).float()

        # Apply triangular tolerance function
        # accuracy = max(0, 1 - distance / tolerance_jnds)
        tolerance_scores = torch.clamp(
            1.0 - distances / self.tolerance_jnds,
            min=0.0,
            max=1.0
        )

        # Update state
        self.correct += tolerance_scores.sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the final JND tolerance accuracy."""
        return self.correct / self.total


class ExactAccuracy(Metric):
    """
    Standard exact-match accuracy for comparison.
    Wrapper around torchmetrics accuracy for consistency.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update with exact match accuracy."""
        # Handle both logits and continuous predictions
        if preds.dim() > 1 and preds.shape[1] > 1:
            # Get predicted class indices from logits
            pred_indices = torch.argmax(preds, dim=1)
        else:
            # Already continuous predictions - round to nearest class index
            pred_indices = torch.round(preds.squeeze(-1) if preds.dim() > 1 else preds)

        self.correct += (pred_indices == target).sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute exact accuracy."""
        return self.correct.float() / self.total


class MultiToleranceAccuracy(nn.Module):
    """
    Multiple tolerance levels for comprehensive evaluation.
    Reports exact, ±1, ±2, and ±5 JND accuracy.
    """

    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()
        self.exact_acc = ExactAccuracy()
        self.jnd_1_acc = JNDToleranceAccuracy(tolerance_jnds=1, num_classes=num_classes)
        self.jnd_2_acc = JNDToleranceAccuracy(tolerance_jnds=2, num_classes=num_classes)
        self.jnd_5_acc = JNDToleranceAccuracy(tolerance_jnds=5, num_classes=num_classes)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update all tolerance metrics."""
        self.exact_acc.update(preds, target)
        self.jnd_1_acc.update(preds, target)
        self.jnd_2_acc.update(preds, target)
        self.jnd_5_acc.update(preds, target)

    def compute(self) -> dict:
        """Compute all tolerance accuracies."""
        return {
            "exact": self.exact_acc.compute(),
            "jnd_1": self.jnd_1_acc.compute(),
            "jnd_2": self.jnd_2_acc.compute(),
            "jnd_5": self.jnd_5_acc.compute()
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.exact_acc.reset()
        self.jnd_1_acc.reset()
        self.jnd_2_acc.reset()
        self.jnd_5_acc.reset()