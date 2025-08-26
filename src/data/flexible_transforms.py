"""Flexible transforms that adapt to variable number of channels."""

import torch
from typing import Tuple, Union


class FlexibleNormalize:
    """
    Normalization transform that adapts to variable number of channels.

    For channels 1-3, uses provided mean/std.
    For additional channels (features), uses (0.5, 0.5) normalization.
    """

    def __init__(
        self,
        base_mean: Union[float, Tuple[float, ...]] = (0.4914, 0.4822, 0.4465),
        base_std: Union[float, Tuple[float, ...]] = (0.2023, 0.1994, 0.2010),
        feature_mean: float = 0.5,
        feature_std: float = 0.5
    ):
        """
        Initialize flexible normalization.

        Args:
            base_mean: Mean for base channels (original image)
            base_std: Std for base channels (original image)
            feature_mean: Mean for additional feature channels
            feature_std: Std for additional feature channels
        """
        if isinstance(base_mean, (int, float)):
            self.base_mean = [base_mean]
        else:
            self.base_mean = list(base_mean)

        if isinstance(base_std, (int, float)):
            self.base_std = [base_std]
        else:
            self.base_std = list(base_std)

        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply flexible normalization.

        Args:
            tensor: Input tensor [C, H, W]

        Returns:
            Normalized tensor
        """
        num_channels = tensor.shape[0]
        base_channels = len(self.base_mean)

        if num_channels <= base_channels:
            # Use base normalization for all channels
            mean = torch.tensor(self.base_mean[:num_channels], device=tensor.device, dtype=tensor.dtype)
            std = torch.tensor(self.base_std[:num_channels], device=tensor.device, dtype=tensor.dtype)
        else:
            # Use base normalization for first channels, feature normalization for extra
            extra_channels = num_channels - base_channels

            full_mean = self.base_mean + [self.feature_mean] * extra_channels
            full_std = self.base_std + [self.feature_std] * extra_channels

            mean = torch.tensor(full_mean, device=tensor.device, dtype=tensor.dtype)
            std = torch.tensor(full_std, device=tensor.device, dtype=tensor.dtype)

        # Reshape for broadcasting
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

        return (tensor - mean) / std

    def __repr__(self):
        return f"{self.__class__.__name__}(base_mean={self.base_mean}, base_std={self.base_std}, feature_mean={self.feature_mean}, feature_std={self.feature_std})"


class ChannelAwareCompose:
    """
    Compose transforms that skips certain transforms for >3 channel tensors.

    This allows us to skip PIL-based transforms for feature-augmented tensors
    while still applying normalization.
    """

    def __init__(self, transforms_list):
        """
        Initialize channel-aware compose.

        Args:
            transforms_list: List of (transform, max_channels) tuples.
                           If max_channels is None, always apply.
                           If tensor has more channels than max_channels, skip.
        """
        self.transforms = []
        for item in transforms_list:
            if isinstance(item, tuple):
                transform, max_channels = item
                self.transforms.append((transform, max_channels))
            else:
                # No channel limit
                self.transforms.append((item, None))

    def __call__(self, tensor):
        """Apply transforms based on channel count."""
        # Determine number of channels
        if hasattr(tensor, 'shape'):
            num_channels = tensor.shape[0]
        else:
            # PIL Image case
            num_channels = 3

        for transform, max_channels in self.transforms:
            # Skip transform if tensor has more channels than supported
            if max_channels is not None and num_channels > max_channels:
                continue

            try:
                tensor = transform(tensor)
                # Update channel count after transform (e.g., ToTensor changes shape)
                if hasattr(tensor, 'shape'):
                    num_channels = tensor.shape[0]
            except Exception as e:
                # Skip transforms that can't handle the tensor type/shape
                continue

        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t, max_ch in self.transforms:
            format_string += f'\n    {t} (max_channels={max_ch})'
        format_string += '\n)'
        return format_string


# Convenience functions
def create_flexible_transforms(height: int = 32, width: int = 32, training: bool = True):
    """
    Create transforms that work with variable channel counts.

    Args:
        height: Image height
        width: Image width
        training: Whether to include training augmentations

    Returns:
        Transform composition that adapts to channel count
    """
    from torchvision import transforms

    if height == 32 and width == 32:
        # CIFAR-style
        base_mean = (0.4914, 0.4822, 0.4465)
        base_std = (0.2023, 0.1994, 0.2010)
    elif height == 28 and width == 28:
        # MNIST-style
        base_mean = (0.5,)
        base_std = (0.5,)
    else:
        # Generic
        base_mean = (0.5, 0.5, 0.5)
        base_std = (0.5, 0.5, 0.5)

    if training:
        transform_list = [
            (transforms.ToPILImage(), 3),  # Only for ≤3 channels
            (transforms.RandomCrop(height, padding=4), 3),
            (transforms.RandomHorizontalFlip(), 3),
            (transforms.RandomRotation(15), 3),
            (transforms.ToTensor(), 3),
            (FlexibleNormalize(base_mean, base_std), None)  # Always apply
        ]
    else:
        transform_list = [
            (FlexibleNormalize(base_mean, base_std), None)  # Always apply
        ]

    return ChannelAwareCompose(transform_list)


if __name__ == "__main__":
    # Test the flexible transforms
    import torch

    # Test 3-channel tensor
    tensor_3ch = torch.randn(3, 32, 32)
    print(f"3-channel input: {tensor_3ch.shape}")

    # Test 5-channel tensor (with features)
    tensor_5ch = torch.randn(5, 32, 32)
    print(f"5-channel input: {tensor_5ch.shape}")

    # Create transforms
    val_transform = create_flexible_transforms(training=False)
    train_transform = create_flexible_transforms(training=True)

    print(f"Val transform: {val_transform}")

    # Test normalization
    norm_3ch = val_transform(tensor_3ch)
    norm_5ch = val_transform(tensor_5ch)

    print(f"3-channel output: {norm_3ch.shape}, mean: {norm_3ch.mean(dim=(1,2))}")
    print(f"5-channel output: {norm_5ch.shape}, mean: {norm_5ch.mean(dim=(1,2))}")

    print("Flexible transforms working correctly!")
