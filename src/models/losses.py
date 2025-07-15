"""Custom loss functions for multihead models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for quantized continuous parameters.

    This loss function treats the problem as regression with discrete ordinal values,
    where the numerical distance between predicted and actual values matters.

    For VIMH datasets, the labels are quantized values (0-255) representing
    continuous parameters. This loss function:
    1. Converts class logits to continuous predictions using weighted averaging
    2. Applies regression loss (L1, L2, or Huber) on the continuous space
    3. Optionally adds a classification term for regularization
    4. Normalizes loss to [0,1] by dividing by maximum possible distance

    Args:
        num_classes: Number of ordinal classes (e.g., 256 for VIMH)
        regression_loss: Type of regression loss ('l1', 'l2', 'huber')
        alpha: Weight for classification term (0.0 = pure regression)
        huber_delta: Delta parameter for Huber loss
        normalize_loss: Whether to normalize loss to [0,1] range (default: True)
    """

    def __init__(
        self,
        num_classes: int,
        regression_loss: str = 'l1',
        alpha: float = 0.1,
        huber_delta: float = 1.0,
        normalize_loss: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.regression_loss = regression_loss
        self.alpha = alpha
        self.huber_delta = huber_delta
        self.normalize_loss = normalize_loss

        # Create ordinal class centers (0, 1, 2, ..., num_classes-1)
        self.register_buffer('class_centers', torch.arange(num_classes, dtype=torch.float32))

        # Maximum possible distance for normalization
        self.max_distance = num_classes - 1

        # Classification loss for regularization
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ordinal regression loss.

        Args:
            logits: Raw logits from model [batch_size, num_classes]
            targets: Target class indices [batch_size]

        Returns:
            Combined loss value
        """
        batch_size = logits.size(0)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Convert to continuous predictions using weighted averaging
        # pred_continuous = sum(prob_i * class_center_i)
        pred_continuous = torch.sum(probs * self.class_centers.unsqueeze(0), dim=1)

        # Convert target indices to continuous values
        target_continuous = targets.float()

        # Calculate regression loss
        if self.regression_loss == 'l1':
            reg_loss = F.l1_loss(pred_continuous, target_continuous)
        elif self.regression_loss == 'l2':
            reg_loss = F.mse_loss(pred_continuous, target_continuous)
        elif self.regression_loss == 'huber':
            reg_loss = F.huber_loss(pred_continuous, target_continuous, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown regression loss: {self.regression_loss}")

        # Normalize regression loss to [0,1] range
        if self.normalize_loss:
            reg_loss = reg_loss / self.max_distance

        # Add classification term for regularization (helps with training stability)
        if self.alpha > 0:
            class_loss = self.cross_entropy(logits, targets)
            total_loss = reg_loss + self.alpha * class_loss
        else:
            total_loss = reg_loss

        return total_loss


class QuantizedRegressionLoss(nn.Module):
    """
    Simplified quantized regression loss for continuous parameters.

    This loss function directly applies regression loss to the output logits,
    treating the model output as a continuous prediction in the range [0, num_classes-1].

    Args:
        num_classes: Number of quantized levels (e.g., 256 for VIMH)
        loss_type: Type of regression loss ('l1', 'l2', 'huber')
        huber_delta: Delta parameter for Huber loss
        normalize_loss: Whether to normalize loss to [0,1] range (default: True)
    """

    def __init__(
        self,
        num_classes: int,
        loss_type: str = 'l1',
        huber_delta: float = 1.0,
        normalize_loss: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.normalize_loss = normalize_loss

        # Maximum possible distance for normalization
        self.max_distance = num_classes - 1

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of quantized regression loss.

        Args:
            output: Model output (single continuous value per sample) [batch_size, 1]
            targets: Target class indices [batch_size]

        Returns:
            Regression loss value
        """
        # Convert target indices to continuous values
        target_continuous = targets.float()

        # Ensure output is squeezed to match target shape
        if output.dim() > 1:
            output = output.squeeze(-1)

        # Clamp predictions to valid range
        output = torch.clamp(output, 0, self.num_classes - 1)

        # Calculate regression loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(output, target_continuous)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(output, target_continuous)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(output, target_continuous, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Normalize loss to [0,1] range
        if self.normalize_loss:
            loss = loss / self.max_distance

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
