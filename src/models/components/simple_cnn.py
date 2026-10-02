from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn


def _pool_bins(n: int, target: int = 4) -> int:
    """Smallest divisor of ``n`` that is >= ``target`` (``n`` itself if ``n`` < ``target``).

    Equal-size bins are required by AdaptiveAvgPool2d on MPS; never pooling below
    ``target`` bins keeps time/frequency resolution (a prime n gives n bins, not 1).
    """
    if n < 1:
        raise ValueError("SimpleCNN needs inputs of at least 4x4 (two 2x max-pools)")
    return next(d for d in range(min(target, n), n + 1) if n % d == 0)


class SimpleCNN(nn.Module):
    """Lightweight convolutional network for VIMH spectrogram inputs."""

    def __init__(
        self,
        input_channels: int = 1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc_hidden: int = 128,
        output_size: Optional[int] = None,
        heads_config: Optional[Dict[str, int]] = None,
        dropout: float = 0.25,
        input_size: Union[int, Sequence[int]] = 28,
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        auxiliary_input_size: int = 0,
        auxiliary_hidden_size: int = 32,
    ) -> None:
        """Initialize a SimpleCNN module.

        :param input_channels: Number of input channels (spectrogram representations, default 1).
        :param conv1_channels: Number of output channels for first conv layer.
        :param conv2_channels: Number of output channels for second conv layer.
        :param fc_hidden: Number of hidden units in fully connected layer.
        :param output_size: Number of output classes of a single placeholder head
            (used only when ``heads_config`` is None).
        :param heads_config: Dict mapping head names to number of classes for multihead.
        :param dropout: Dropout probability.
        :param input_size: Input spectrogram size, int (square) or [height, width]; train.py sets it from the dataset.
        :param output_mode: Output mode - "classification" or "regression".
        :param parameter_names: List of parameter names for regression mode.
        :param parameter_ranges: Dict mapping parameter names to (min, max) ranges.
        :param auxiliary_input_size: Size of auxiliary scalar input vector (0 = no aux input).
        :param auxiliary_hidden_size: Hidden size for auxiliary input processing.
        """
        super().__init__()

        # Store output mode and parameter information
        self.input_channels = input_channels
        self.output_mode = output_mode
        self.parameter_names = parameter_names or []
        self.parameter_ranges = parameter_ranges or {}
        self.auxiliary_input_size = auxiliary_input_size
        self.auxiliary_hidden_size = auxiliary_hidden_size
        self.fc_hidden = fc_hidden
        self.input_size = input_size
        if isinstance(input_size, int):
            height, width = input_size, input_size
        else:
            height, width = (int(v) for v in input_size)
        self.input_resolution = (height, width)

        if output_mode == "regression":
            # One output per parameter; empty until auto-configured from the dataset
            heads_config = {name: 1 for name in self.parameter_names}
        elif heads_config is None:
            # Placeholder head, replaced by auto-configuration from the dataset
            heads_config = {"digit": output_size if output_size is not None else 10}

        # After two stride-2 MaxPool2d layers the feature map is (H//4, W//4). MPS only
        # supports AdaptiveAvgPool2d when each output size divides its input size, so
        # pick per-axis bin counts that do (e.g. 32x32 -> 4x4, 28x28 -> 7x7, 32x100 -> 4x5).
        self.adaptive_pool_size = (_pool_bins(height // 4), _pool_bins(width // 4))

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second conv block
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Adaptive pooling with MPS-safe size
            nn.AdaptiveAvgPool2d(self.adaptive_pool_size),
        )

        # Calculate linear layer input size based on adaptive pool size
        linear_input_size = (
            conv2_channels * self.adaptive_pool_size[0] * self.adaptive_pool_size[1]
        )

        self.shared_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Auxiliary input processing (if enabled)
        if auxiliary_input_size > 0:
            self.auxiliary_net = nn.Sequential(
                nn.Linear(auxiliary_input_size, auxiliary_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout / 2),  # Less dropout for auxiliary features
                nn.Linear(auxiliary_hidden_size, auxiliary_hidden_size),
                nn.ReLU(),
            )
        else:
            self.auxiliary_net = None

        self._build_heads(heads_config)

    def forward(self, x: torch.Tensor, auxiliary: Optional[torch.Tensor] = None):
        """Perform a single forward pass through the network.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :param auxiliary: Auxiliary input tensor of shape (batch_size, auxiliary_input_size);
            required if and only if the network was built with ``auxiliary_input_size > 0``.
        :return: A tensor of logits (single head) or dict of logits (multihead).
        """
        if not self.heads:
            raise RuntimeError("SimpleCNN has no heads; configure heads_config/parameter_names")
        if (self.auxiliary_net is None) != (auxiliary is None):
            raise ValueError(
                f"SimpleCNN built with auxiliary_input_size={self.auxiliary_input_size} but "
                f"called with auxiliary={'None' if auxiliary is None else tuple(auxiliary.shape)}"
            )

        x = self.conv_layers(x)
        features = self.shared_features(x)
        if self.auxiliary_net is not None:
            features = torch.cat([features, self.auxiliary_net(auxiliary)], dim=1)

        if self.is_multihead:
            return {head_name: head(features) for head_name, head in self.heads.items()}
        return next(iter(self.heads.values()))(features)

    def _build_heads(self, heads_config: Dict[str, int]) -> None:
        """(Re)build the output heads (classification or regression per ``output_mode``)."""
        combined_feature_size = self.fc_hidden + (
            self.auxiliary_hidden_size if self.auxiliary_input_size > 0 else 0
        )
        if self.output_mode == "regression":
            # Sigmoid outputs in [0, 1], denormalized to parameter units by the LitModule
            self.heads = nn.ModuleDict(
                {
                    head_name: nn.Sequential(nn.Linear(combined_feature_size, 1), nn.Sigmoid())
                    for head_name in heads_config.keys()
                }
            )
        else:
            self.heads = nn.ModuleDict(
                {
                    head_name: nn.Linear(combined_feature_size, num_classes)
                    for head_name, num_classes in heads_config.items()
                }
            )
        self.heads_config = heads_config
        self.is_multihead = len(heads_config) > 1


if __name__ == "__main__":
    # Quick smoke tests using VIMH-style spectrogram tensors
    batch = torch.randn(2, 1, 32, 32)  # Batch of 32x32 spectrograms

    model_multi = SimpleCNN(
        input_channels=1,
        heads_config={"log10_decay_time": 96, "wah_position": 96},
        input_size=32,
    )
    outputs = model_multi(batch)
    print(f"Input shape: {batch.shape}")
    for head, tensor in outputs.items():
        print(f"{head}: {tensor.shape}")
