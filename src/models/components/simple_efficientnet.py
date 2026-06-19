import math
from typing import Dict, Optional

import torch
from torch import nn


class SwishActivation(nn.Module):
    """Swish activation function used in EfficientNet."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class DropPath(nn.Module):
    """Stochastic depth: drop the residual branch per sample during training.

    Unlike ``nn.Dropout2d`` (which zeros whole channels of a feature map), this
    drops the entire branch for a subset of samples, which is the regularization
    EfficientNet's MBConv stochastic-depth schedule is meant to apply.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block used in EfficientNet."""

    def __init__(self, in_channels: int, reduction_ratio: int = 4) -> None:
        """Initialize SE block.

        :param in_channels: Number of input channels.
        :param reduction_ratio: Reduction ratio for SE block.
        """
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            SwishActivation(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block used in EfficientNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
    ) -> None:
        """Initialize MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for depthwise conv.
        :param stride: Stride for depthwise conv.
        :param expand_ratio: Expansion ratio for inverted bottleneck.
        :param se_ratio: Squeeze-excitation ratio.
        :param drop_rate: Dropout rate for stochastic depth.
        """
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate

        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                SwishActivation(),
            )
        else:
            self.expand = nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            SwishActivation(),
        )

        # Squeeze-and-Excitation
        if se_ratio > 0:
            self.se = SqueezeExcitation(expanded_channels, int(1 / se_ratio))
        else:
            self.se = nn.Identity()

        # Point-wise convolution
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Stochastic depth (applied to the residual branch only, see forward).
        self.drop_path = DropPath(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_residual:
            # Drop the whole residual branch per sample (no-op when drop_rate == 0
            # or in eval), then add the identity shortcut.
            out = self.drop_path(out) + x

        return out


class SimpleEfficientNet(nn.Module):
    """A simplified EfficientNet-B0 for MNIST classification."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: Optional[int] = None,
        heads_config: Optional[Dict[str, int]] = None,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.2,
    ) -> None:
        """Initialize SimpleEfficientNet.

        :param input_channels: Number of input channels (1 for grayscale MNIST).
        :param num_classes: Number of output classes (backward compatibility).
        :param heads_config: Dict mapping head names to number of classes for multihead.
        :param width_mult: Width multiplier for scaling channels.
        :param depth_mult: Depth multiplier for scaling layers.
        :param dropout_rate: Dropout rate for classifier.
        :param drop_path_rate: Maximum stochastic-depth (DropPath) rate; applied as a
            schedule increasing linearly from 0 across all MBConv blocks.
        """
        super().__init__()

        # Backward compatibility: convert old single-head config to multihead
        if heads_config is None:
            if num_classes is not None:
                heads_config = {"digit": num_classes}
            else:
                heads_config = {"digit": 10}  # Default MNIST

        self.heads_config = heads_config
        self.is_multihead = len(heads_config) > 1

        # Stem convolution
        stem_channels = self._scale_width(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            SwishActivation(),
        )

        # MBConv blocks configuration for simplified B0
        # Format: (expand_ratio, channels, num_blocks, stride, kernel_size)
        block_configs = [
            (1, 16, 1, 1, 3),  # Stage 1
            (6, 24, 2, 2, 3),  # Stage 2
            (6, 40, 2, 2, 5),  # Stage 3
            (6, 80, 3, 2, 3),  # Stage 4
            (6, 112, 3, 1, 5),  # Stage 5
            (6, 192, 4, 2, 5),  # Stage 6
            (6, 320, 1, 1, 3),  # Stage 7
        ]

        # Build MBConv blocks. Scale per-stage depths first so a single global
        # stochastic-depth schedule can increase linearly from 0 to drop_path_rate
        # across *all* blocks (the previous schedule reset per stage and divided by
        # the stage count, so the first block of every stage had rate 0).
        self.blocks = nn.ModuleList()
        in_channels = stem_channels

        scaled_configs = [
            (expand_ratio, channels, self._scale_depth(num_blocks, depth_mult), stride, kernel_size)
            for expand_ratio, channels, num_blocks, stride, kernel_size in block_configs
        ]
        total_blocks = sum(num_blocks for _, _, num_blocks, _, _ in scaled_configs)
        dp_rates = (
            [r.item() for r in torch.linspace(0.0, drop_path_rate, total_blocks)]
            if total_blocks > 0
            else []
        )

        block_idx = 0
        for expand_ratio, channels, num_blocks, stride, kernel_size in scaled_configs:
            out_channels = self._scale_width(channels, width_mult)

            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25,
                        drop_rate=dp_rates[block_idx],
                    )
                )
                in_channels = out_channels
                block_idx += 1

        # Feature extraction head
        head_channels = self._scale_width(1280, width_mult)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            SwishActivation(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
        )

        # Classification heads
        if self.is_multihead:
            self.heads = nn.ModuleDict(
                {
                    head_name: nn.Linear(head_channels, num_classes)
                    for head_name, num_classes in heads_config.items()
                }
            )
        else:
            # Single head (backward compatibility)
            head_name, num_classes = next(iter(heads_config.items()))
            self.classifier = nn.Linear(head_channels, num_classes)

    def _scale_width(self, channels: int, width_mult: float) -> int:
        """Scale width (channels) by multiplier."""
        return int(math.ceil(channels * width_mult / 8) * 8)

    def _scale_depth(self, num_blocks: int, depth_mult: float) -> int:
        """Scale depth (number of blocks) by multiplier."""
        return int(math.ceil(num_blocks * depth_mult))

    def forward(self, x: torch.Tensor):
        """Perform a single forward pass through the network.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: A tensor of logits (single head) or dict of logits (multihead).
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        features = self.feature_extractor(x)

        if self.is_multihead:
            return {head_name: head(features) for head_name, head in self.heads.items()}
        else:
            # Single head output (backward compatibility)
            return self.classifier(features)


if __name__ == "__main__":
    # Test backward compatibility (single head)
    print("Testing single head mode (backward compatibility):")
    model_single = SimpleEfficientNet()
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    output_single = model_single(x)
    print(f"Input shape: {x.shape}")
    print(f"Single head output shape: {output_single.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model_single.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test multihead mode
    print("\nTesting multihead mode:")
    model_multi = SimpleEfficientNet(heads_config={"digit": 10, "thickness": 5, "smoothness": 3})
    output_multi = model_multi(x)
    print(f"Multihead output type: {type(output_multi)}")
    for head_name, logits in output_multi.items():
        print(f"  {head_name}: {logits.shape}")
