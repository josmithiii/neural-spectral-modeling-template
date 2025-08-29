import torch
from torch import nn
from typing import Dict, Optional


class SimpleMLP(nn.Module):
    """A simple fully-connected neural net with Dropout and without BatchNorm (for batch_size=1), otherwise same as SimpleDenseNet."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: Optional[int] = None,
        heads_config: Optional[Dict[str, int]] = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a `SimpleMLP` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer (backward compatibility).
        :param heads_config: Dict mapping head names to number of classes for multihead.
        :param dropout: Dropout probability.
        """
        super().__init__()

        # Backward compatibility: convert old single-head config to multihead
        if heads_config is None:
            if output_size is not None:
                heads_config = {'digit': output_size}
            else:
                heads_config = {'digit': 10}  # Default MNIST

        self.heads_config = heads_config
        self.is_multihead = len(heads_config) > 1

        # Shared feature extraction layers (no BatchNorm for batch_size=1 compatibility)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lin2_size, lin3_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification heads
        self.heads = nn.ModuleDict()
        for head_name, num_classes in heads_config.items():
            self.heads[head_name] = nn.Linear(lin3_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions (single head) or dict of tensors (multihead).
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        # Extract features
        features = self.feature_extractor(x)

        # Multihead case: return dict of predictions
        if self.is_multihead:
            return {head_name: head(features) for head_name, head in self.heads.items()}

        # Single head case: return tensor directly (backward compatibility)
        return self.heads['digit'](features)


if __name__ == "__main__":
    _ = SimpleMLP()
