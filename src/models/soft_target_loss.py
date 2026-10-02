# From 4o conversation https://chatgpt.com/c/687b5125-0568-800f-affc-0ae9e6b69c9a

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetLoss(nn.Module):
    def __init__(self, num_classes: int, mode: str = "triangular", width: int = 1, sigma: float = 1.0):
        """
        Soft target loss function for classification tasks with ordinal structure.

        Provides smooth probability distributions around target classes to reduce
        quantization artifacts and improve generalization for discretized continuous values.

        Args:
            num_classes: total number of discrete bins.
            mode: 'triangular', 'gaussian', or 'log-triangular'.
            width: for triangular and log-triangular: maximum bin offset.
            sigma: for gaussian: standard deviation in bin indices (support spans all bins).
        """
        super().__init__()
        if mode not in {"triangular", "gaussian", "log-triangular"}:
            raise ValueError(f"Unknown SoftTargetLoss mode '{mode}'")
        self.num_classes = num_classes
        self.mode = mode
        self.width = width
        self.sigma = sigma

    def soft_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Soft target distributions [B, C] for integer bin indices [B] (rows sum to 1)."""
        bins = torch.arange(self.num_classes, device=targets.device)
        dist = (bins.unsqueeze(0) - targets.unsqueeze(1)).abs().float()  # [B, C]
        if self.mode == "triangular":
            weights = (self.width + 1 - dist).clamp(min=0.0)
        elif self.mode == "log-triangular":
            weights = torch.exp(-dist) * (dist <= self.width)
        else:  # gaussian
            weights = torch.exp(-0.5 * (dist / self.sigma) ** 2)
        return weights / weights.sum(dim=1, keepdim=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: tensor of shape [B, C], raw unnormalized scores
            targets: tensor of shape [B], int bin indices
        Returns:
            loss: KL divergence between softmax(logits) and soft target distributions
        """
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Logit dimension {logits.shape[1]} must match num_classes {self.num_classes}"
            )
        log_probs = F.log_softmax(logits, dim=1)
        return F.kl_div(log_probs, self.soft_targets(targets), reduction="batchmean")


# Example Usage:

# loss_fn = SoftTargetLoss(num_classes=101, mode='gaussian', sigma=10.0)  # JND bins over 0–100

# logits = model(inputs)         # [B, 101]
# targets = torch.randint(0, 101, (B,))  # true class indices
# loss = loss_fn(logits, targets)
