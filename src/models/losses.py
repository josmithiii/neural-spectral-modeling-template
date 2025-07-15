"""Custom loss functions for multihead models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for quantized continuous parameters in perceptual units.

    This loss function treats the problem as regression with discrete ordinal values,
    where the numerical distance between predicted and actual values matters.

    For VIMH datasets, the labels are quantized values (0-255) representing
    continuous parameters. This loss function:
    1. Converts class logits to continuous predictions using weighted averaging
    2. Maps quantized distance to actual parameter space (perceptual units)
    3. Applies regression loss (L1, L2, or Huber) in perceptual space
    4. Optionally adds a classification term for regularization
    5. Returns loss in perceptual units (no normalization)

    Args:
        num_classes: Number of ordinal classes (e.g., 256 for VIMH)
        param_range: Actual parameter range (max - min) in perceptual units
        regression_loss: Type of regression loss ('l1', 'l2', 'huber')
        alpha: Weight for classification term (0.0 = pure regression)
        huber_delta: Delta parameter for Huber loss
        normalize_loss: DEPRECATED - loss is now in perceptual units
    """

    def __init__(
        self,
        num_classes: int,
        param_range: float,
        regression_loss: str = 'l1',
        alpha: float = 0.1,
        huber_delta: float = 1.0,
        normalize_loss: bool = False  # Deprecated, kept for compatibility
    ):
        super().__init__()
        self.num_classes = num_classes
        self.param_range = param_range
        self.regression_loss = regression_loss
        self.alpha = alpha
        self.huber_delta = huber_delta

        # Calculate perceptual step size
        self.quantization_step = param_range / (num_classes - 1)

        # Create ordinal class centers (0, 1, 2, ..., num_classes-1)
        self.register_buffer('class_centers', torch.arange(num_classes, dtype=torch.float32))

        # Classification loss for regularization
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ordinal regression loss.

        Args:
            logits: Raw logits from model [batch_size, num_classes]
            targets: Target class indices [batch_size]

        Returns:
            Loss value in perceptual units
        """
        batch_size = logits.size(0)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Convert to continuous predictions using weighted averaging
        # pred_continuous = sum(prob_i * class_center_i)
        class_centers = self.class_centers.to(logits.device)
        pred_continuous = torch.sum(probs * class_centers.unsqueeze(0), dim=1)

        # Convert target indices to continuous values
        target_continuous = targets.float()

        # Calculate distance in quantization steps
        if self.regression_loss == 'l1':
            distance_steps = F.l1_loss(pred_continuous, target_continuous)
        elif self.regression_loss == 'l2':
            distance_steps = F.mse_loss(pred_continuous, target_continuous)
        elif self.regression_loss == 'huber':
            distance_steps = F.huber_loss(pred_continuous, target_continuous, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown regression loss: {self.regression_loss}")

        # Convert distance to perceptual units
        perceptual_distance = distance_steps * self.quantization_step

        # Add classification term for regularization (helps with training stability)
        if self.alpha > 0:
            class_loss = self.cross_entropy(logits, targets)
            total_loss = perceptual_distance + self.alpha * class_loss
        else:
            total_loss = perceptual_distance

        return total_loss


class QuantizedRegressionLoss(nn.Module):
    """
    Simplified quantized regression loss for continuous parameters in perceptual units.

    This loss function directly applies regression loss to the output logits,
    treating the model output as a continuous prediction in the range [0, num_classes-1].

    Args:
        num_classes: Number of quantized levels (e.g., 256 for VIMH)
        param_range: Actual parameter range (max - min) in perceptual units
        loss_type: Type of regression loss ('l1', 'l2', 'huber')
        huber_delta: Delta parameter for Huber loss
        normalize_loss: DEPRECATED - loss is now in perceptual units
    """

    def __init__(
        self,
        num_classes: int,
        param_range: float,
        loss_type: str = 'l1',
        huber_delta: float = 1.0,
        normalize_loss: bool = False  # Deprecated, kept for compatibility
    ):
        super().__init__()
        self.num_classes = num_classes
        self.param_range = param_range
        self.loss_type = loss_type
        self.huber_delta = huber_delta

        # Calculate perceptual step size
        self.quantization_step = param_range / (num_classes - 1)

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of quantized regression loss.

        Args:
            output: Model output (single continuous value per sample) [batch_size, 1]
            targets: Target class indices [batch_size]

        Returns:
            Loss value in perceptual units
        """
        # Convert target indices to continuous values
        target_continuous = targets.float()

        # Ensure output is squeezed to match target shape
        if output.dim() > 1:
            output = output.squeeze(-1)

        # Clamp predictions to valid range
        output = torch.clamp(output, 0, self.num_classes - 1)

        # Calculate distance in quantization steps
        if self.loss_type == 'l1':
            distance_steps = F.l1_loss(output, target_continuous)
        elif self.loss_type == 'l2':
            distance_steps = F.mse_loss(output, target_continuous)
        elif self.loss_type == 'huber':
            distance_steps = F.huber_loss(output, target_continuous, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Convert distance to perceptual units
        perceptual_distance = distance_steps * self.quantization_step

        return perceptual_distance


class NormalizedRegressionLoss(nn.Module):
    """
    Regression loss for normalized [0,1] parameter values.

    This loss function is designed for pure regression heads that output sigmoid-activated
    values in the [0,1] range. It normalizes the targets to [0,1] space, computes the
    regression loss, and optionally scales the result back to parameter space for
    interpretability.

    Args:
        param_range: Tuple of (min, max) values for the parameter in its original space
        loss_type: Type of regression loss ('mse', 'l1', 'huber')
        huber_delta: Delta parameter for Huber loss (in normalized space)
        return_perceptual_units: Whether to scale loss back to parameter space
    """

    def __init__(
        self,
        param_range: Tuple[float, float],
        loss_type: str = "mse",
        huber_delta: float = 0.1,
        return_perceptual_units: bool = True,
    ):
        super().__init__()
        self.param_min, self.param_max = param_range
        self.param_range = self.param_max - self.param_min
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.return_perceptual_units = return_perceptual_units

        # Ensure valid parameter range
        if self.param_range <= 0:
            raise ValueError(f"Parameter range must be positive, got {self.param_range}")

    def forward(self, normalized_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute regression loss between normalized predictions and targets.

        Args:
            normalized_pred: Sigmoid-activated predictions in [0,1] range [batch_size, 1]
            target: Target values in original parameter space [batch_size]

        Returns:
            Loss value (in parameter space if return_perceptual_units=True)
        """
        # Ensure predictions are in [0,1] range (should be from sigmoid)
        normalized_pred = torch.clamp(normalized_pred.squeeze(-1), 0.0, 1.0)

        # Normalize targets to [0,1] range
        normalized_target = (target - self.param_min) / self.param_range
        normalized_target = torch.clamp(normalized_target, 0.0, 1.0)

        # Compute loss in normalized space
        if self.loss_type == "mse":
            loss = F.mse_loss(normalized_pred, normalized_target)
        elif self.loss_type == "l1":
            loss = F.l1_loss(normalized_pred, normalized_target)
        elif self.loss_type == "huber":
            loss = F.huber_loss(normalized_pred, normalized_target, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Convert back to parameter space for interpretability
        if self.return_perceptual_units:
            return loss * self.param_range
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted CrossEntropyLoss that penalizes distant predictions more than close ones.

    This keeps the classification framework but adds distance-based penalties.
    The weight for each class is based on its distance from the true class.

    Args:
        num_classes: Number of classes
        distance_power: Power to raise distances to (higher = more penalty for distant errors)
        base_weight: Base weight for correct class
    """

    def __init__(
        self,
        num_classes: int,
        distance_power: float = 2.0,
        base_weight: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.distance_power = distance_power
        self.base_weight = base_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of weighted cross entropy loss.

        Args:
            logits: Raw logits from model [batch_size, num_classes]
            targets: Target class indices [batch_size]

        Returns:
            Weighted cross entropy loss
        """
        batch_size = logits.size(0)

        # Create distance matrix: distance[i,j] = |i - j|
        class_indices = torch.arange(self.num_classes, device=logits.device)
        target_expanded = targets.unsqueeze(1)  # [batch_size, 1]
        distances = torch.abs(class_indices.unsqueeze(0) - target_expanded)  # [batch_size, num_classes]

        # Convert distances to weights: closer predictions get lower weights (less penalty)
        weights = self.base_weight + distances.float() ** self.distance_power

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Calculate weighted negative log likelihood
        # We want to minimize: sum(weight_i * prob_i * log(prob_i))
        # But we need to be careful about the target class
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Calculate loss as weighted sum of probabilities for incorrect classes
        # plus standard cross entropy for correct class
        cross_entropy_loss = F.cross_entropy(logits, targets, reduction='none')

        # Add distance-based penalty for incorrect predictions
        incorrect_mask = (class_indices.unsqueeze(0) != target_expanded).float()
        distance_penalty = torch.sum(weights * incorrect_mask * probs, dim=1)

        total_loss = cross_entropy_loss + distance_penalty

        return total_loss.mean()
