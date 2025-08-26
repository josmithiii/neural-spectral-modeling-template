# From 4o conversation https://chatgpt.com/c/687b5125-0568-800f-affc-0ae9e6b69c9a

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftTargetLoss(nn.Module):
    def __init__(self, num_classes, mode='triangular', width=1, sigma=1.0):
        """
        Soft target loss function for classification tasks with ordinal structure.

        Provides smooth probability distributions around target classes to reduce
        quantization artifacts and improve generalization for discretized continuous values.

        Args:
            num_classes: total number of discrete bins.
            mode: 'triangular', 'gaussian', or 'log-triangular'.
            width: for triangular and log-triangular: maximum bin offset.
            sigma: for gaussian: standard deviation in bin indices.
        """
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.width = width
        self.sigma = sigma
        assert mode in {'triangular', 'gaussian', 'log-triangular'}

    def forward(self, logits, targets):
        """
        Args:
            logits: tensor of shape [B, C], raw unnormalized scores
            targets: tensor of shape [B], int bin indices
        Returns:
            loss: KL divergence between softmax(logits) and soft target distributions
        """
        B, C = logits.shape
        assert C == self.num_classes, "Logit dimension must match num_classes"

        # Compute soft targets
        soft_targets = torch.zeros((B, C), device=logits.device)
        for i in range(B):
            c = targets[i].item()
            support = range(max(0, c - self.width), min(C, c + self.width + 1))
            if self.mode == 'triangular':
                for j in support:
                    soft_targets[i, j] = self.width + 1 - abs(j - c)
            elif self.mode == 'log-triangular':
                for j in support:
                    soft_targets[i, j] = math.exp(-(abs(j - c)))
            elif self.mode == 'gaussian':
                for j in support:
                    soft_targets[i, j] = math.exp(-0.5 * ((j - c) / self.sigma) ** 2)

        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)  # normalize

        log_probs = F.log_softmax(logits, dim=1)
        return F.kl_div(log_probs, soft_targets, reduction='batchmean')

# Example Usage:

# loss_fn = SoftTargetLoss(num_classes=101, mode='gaussian', sigma=10.0)  # JND bins over 0–100

# logits = model(inputs)         # [B, 101]
# targets = torch.randint(0, 101, (B,))  # true class indices
# loss = loss_fn(logits, targets)
