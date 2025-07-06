import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        kernel_size (int): Kernel size for depthwise convolution. Default: 7
    """

    def __init__(self, dim: int, drop_path: float = 0.0, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2

    Args:
        input_size (int): Input image size. Default: 28 (for MNIST)
        in_chans (int): Number of input image channels. Default: 1 (for MNIST)
        output_size (int): Number of classes for classification head. Default: 10
        depths (tuple(int)): Number of blocks at each stage. Default: [2, 2, 6, 2]
        dims (tuple(int)): Feature dimension at each stage. Default: [48, 96, 192, 384]
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.0
        kernel_size (int): Kernel size for depthwise convolutions. Default: 7
    """

    def __init__(
        self,
        input_size: int = 28,
        in_chans: int = 1,
        output_size: int = 10,
        depths: tuple = (2, 2, 6, 2),
        dims: tuple = (48, 96, 192, 384),
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_chans = in_chans
        self.output_size = output_size
        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()

        # Stem layer - adapted for different input sizes
        if input_size == 28:  # MNIST
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        elif input_size == 32:  # CIFAR-10/100 - use smaller stride for better feature preservation
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        else:  # ImageNet and other larger inputs
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)

        # Intermediate downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size=kernel_size) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], output_size)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Factory functions for different sizes optimized for MNIST
def convnext_v2_mnist_tiny(input_size: int = 28, in_chans: int = 1, output_size: int = 10, **kwargs):
    """ConvNeXt-V2 Tiny for MNIST (~8K parameters)"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(1, 1, 1, 1),
        dims=(4, 8, 16, 32),
        drop_path_rate=0.0,
        **kwargs
    )


def convnext_v2_mnist_small(input_size: int = 28, in_chans: int = 1, output_size: int = 10, **kwargs):
    """ConvNeXt-V2 Small for MNIST (~68K parameters)"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(1, 1, 2, 1),
        dims=(8, 16, 32, 64),
        drop_path_rate=0.0,
        **kwargs
    )


def convnext_v2_mnist_base(input_size: int = 28, in_chans: int = 1, output_size: int = 10, **kwargs):
    """ConvNeXt-V2 Base for MNIST (~210K parameters)"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 4, 2),
        dims=(12, 24, 48, 96),
        drop_path_rate=0.1,
        **kwargs
    )


def convnext_v2_mnist_large(input_size: int = 28, in_chans: int = 1, output_size: int = 10, **kwargs):
    """ConvNeXt-V2 Large for MNIST (~821K parameters)"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(3, 3, 6, 3),
        dims=(16, 32, 64, 128),
        drop_path_rate=0.2,
        **kwargs
    )


def convnext_v2_official_tiny_benchmark(input_size: int = 28, in_chans: int = 1, output_size: int = 10, **kwargs):
    """ConvNeXt-V2 Official Tiny Benchmark - exact match to Facebook's canonical config"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 4, 2),
        dims=(12, 24, 48, 96),
        drop_path_rate=0.2,  # Official Tiny drop_path: 0.2
        head_init_scale=0.001,  # Official head_init_scale
        **kwargs
    )


# Standard ConvNeXt-V2 variants for ImageNet (keeping original sizes)
def convnext_v2_atto(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Atto"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        **kwargs
    )


def convnext_v2_femto(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Femto"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        **kwargs
    )


def convnext_v2_pico(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Pico"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 256, 512),
        **kwargs
    )


def convnext_v2_nano(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Nano"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 8, 2),
        dims=(80, 160, 320, 640),
        **kwargs
    )


def convnext_v2_tiny(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Tiny"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        **kwargs
    )


def convnext_v2_base(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Base"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        **kwargs
    )


def convnext_v2_large(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Large"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        **kwargs
    )


def convnext_v2_huge(input_size: int = 224, in_chans: int = 3, output_size: int = 1000, **kwargs):
    """ConvNeXt-V2 Huge"""
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(3, 3, 27, 3),
        dims=(352, 704, 1408, 2816),
        **kwargs
    )


# CIFAR-10 optimized variants with smaller kernels and reduced stride
def convnext_v2_cifar10_64k(input_size: int = 32, in_chans: int = 3, output_size: int = 10, **kwargs):
    """ConvNeXt-V2 optimized for CIFAR-10 (~64K parameters)

    Key optimizations for 32x32 images:
    - 2x2 stride-2 stem (vs 4x4 stride-4)
    - 3x3 depthwise convolutions (vs 7x7)
    - Reduced weight decay and drop path
    - Smaller channel dimensions
    """
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 4, 2),
        dims=(8, 16, 32, 64),
        drop_path_rate=0.05,  # Reduced from 0.1
        kernel_size=3,  # Smaller kernel for 32x32 images
        **kwargs
    )


def convnext_v2_cifar10_128k(input_size: int = 32, in_chans: int = 3, output_size: int = 10, **kwargs):
    """ConvNeXt-V2 optimized for CIFAR-10 (~128K parameters)

    Key optimizations for 32x32 images:
    - 2x2 stride-2 stem (vs 4x4 stride-4)
    - 3x3 depthwise convolutions (vs 7x7)
    - Reduced weight decay and drop path
    - Medium channel dimensions
    """
    return ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 6, 2),
        dims=(12, 24, 48, 96),
        drop_path_rate=0.05,  # Reduced from 0.1
        kernel_size=3,  # Smaller kernel for 32x32 images
        **kwargs
    )
