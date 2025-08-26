"""Custom loss functions for multihead models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List


# ==============================================================================
# PNP-Adapted Loss Functions
# Adapted from /l/pnp/src/pnp_synth/neural/loss.py for proven multi-scale spectral analysis
# ==============================================================================

class DistanceLoss(nn.Module):
    """Base class for distance-based losses, adapted from PNP codebase."""
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p
        self.ops = []  # Will be populated by subclasses

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance between tensors."""
        if self.p == 1.0:
            return torch.abs(x - y).mean()
        elif self.p == 2.0:
            return torch.norm(x - y, p=self.p)
        else:
            return torch.norm(x - y, p=self.p)

    def forward(self, x: torch.Tensor, y: torch.Tensor, transform_y: bool = True) -> torch.Tensor:
        """Forward pass computing multi-scale distance."""
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for op in self.ops:
            loss += self.dist(op(x), op(y) if transform_y else y)
        loss /= len(self.ops) if self.ops else 1
        return loss


class MagnitudeSTFT(nn.Module):
    """STFT magnitude computation module, adapted from PNP."""
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude."""
        # Ensure we have proper window device placement
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)

        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        ).abs()


class MultiScaleSpectralLoss(DistanceLoss):
    """
    Multi-resolution STFT loss from PNP codebase, adapted for general-purpose spectral analysis.

    This proven implementation provides superior spectral analysis compared to basic MSE/L1
    by using multiple STFT resolutions simultaneously. It has been validated in published
    research (ICASSP, TASLP papers) and provides robust multi-scale spectral comparison.

    Useful for any time-series or frequency-domain data where multi-scale analysis is beneficial.

    Args:
        max_n_fft: Maximum STFT window size (power of 2)
        num_scales: Number of different STFT scales to use
        hop_lengths: Optional list of hop lengths (auto-computed if None)
        mag_w: Weight for magnitude loss (currently unused but kept for compatibility)
        logmag_w: Weight for log-magnitude loss (currently unused but kept for compatibility)
        p: Norm to use for distance computation (1.0 for L1, 2.0 for L2)
    """

    def __init__(
        self,
        max_n_fft: int = 2048,
        num_scales: int = 6,
        hop_lengths: Optional[List[int]] = None,
        mag_w: float = 1.0,
        logmag_w: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        # Ensure we can create all scales
        assert max_n_fft // (2 ** (num_scales - 1)) > 1, \
            f"max_n_fft={max_n_fft} too small for num_scales={num_scales}"

        # Create STFT window sizes at multiple scales
        self.max_n_fft = max_n_fft
        self.n_ffts = [max_n_fft // (2**i) for i in range(num_scales)]
        self.hop_lengths = (
            [n // 4 for n in self.n_ffts] if hop_lengths is None else hop_lengths
        )

        # Weights for different components (kept for future extensibility)
        self.mag_w = mag_w
        self.logmag_w = logmag_w

        self.create_ops()

    def create_ops(self):
        """Create STFT operators for each scale."""
        self.ops = [
            MagnitudeSTFT(n_fft, self.hop_lengths[i])
            for i, n_fft in enumerate(self.n_ffts)
        ]


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


def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.

    Args:
        loss_config: Configuration dictionary with '_target_' key and parameters

    Returns:
        Configured loss function
    """
    if '_target_' not in loss_config:
        raise ValueError("Loss configuration must contain '_target_' key")

    target = loss_config['_target_']
    params = {k: v for k, v in loss_config.items() if k != '_target_'}

    # Handle standard PyTorch losses
    if target == 'torch.nn.CrossEntropyLoss':
        return nn.CrossEntropyLoss(**params)
    elif target == 'torch.nn.MSELoss':
        return nn.MSELoss(**params)
    elif target == 'torch.nn.L1Loss':
        return nn.L1Loss(**params)
    elif target == 'torch.nn.HuberLoss':
        return nn.HuberLoss(**params)

    # Handle custom losses
    elif target == 'src.models.losses.OrdinalRegressionLoss':
        return OrdinalRegressionLoss(**params)
    elif target == 'src.models.losses.QuantizedRegressionLoss':
        return QuantizedRegressionLoss(**params)
    elif target == 'src.models.losses.WeightedCrossEntropyLoss':
        return WeightedCrossEntropyLoss(**params)
    elif target == 'src.models.losses.NormalizedRegressionLoss':
        return NormalizedRegressionLoss(**params)
    elif target == 'src.models.losses.MultiScaleSpectralLoss':
        return MultiScaleSpectralLoss(**params)
    elif target == 'src.models.soft_target_loss.SoftTargetLoss':
        from src.models.soft_target_loss import SoftTargetLoss
        return SoftTargetLoss(**params)

    else:
        raise ValueError(f"Unknown loss function target: {target}")
