import torch
from torch import nn
from typing import Dict, Optional


class SimpleCNN(nn.Module):
    """A simple convolutional neural network for MNIST classification."""

    def __init__(
        self,
        input_channels: int = 1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc_hidden: int = 128,
        output_size: Optional[int] = None,
        heads_config: Optional[Dict[str, int]] = None,
        dropout: float = 0.25,
    ) -> None:
        """Initialize a SimpleCNN module.

        :param input_channels: Number of input channels (1 for grayscale MNIST).
        :param conv1_channels: Number of output channels for first conv layer.
        :param conv2_channels: Number of output channels for second conv layer.
        :param fc_hidden: Number of hidden units in fully connected layer.
        :param output_size: Number of output classes (backward compatibility).
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
            
            # Adaptive pooling to handle any input size
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        self.shared_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_channels * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Multiple heads or single head for backward compatibility
        if self.is_multihead:
            self.heads = nn.ModuleDict({
                head_name: nn.Linear(fc_hidden, num_classes)
                for head_name, num_classes in heads_config.items()
            })
        else:
            # Single head (backward compatibility)
            head_name, num_classes = next(iter(heads_config.items()))
            self.classifier = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor):
        """Perform a single forward pass through the network.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: A tensor of logits (single head) or dict of logits (multihead).
        """
        x = self.conv_layers(x)
        shared_features = self.shared_features(x)
        
        if self.is_multihead:
            return {
                head_name: head(shared_features)
                for head_name, head in self.heads.items()
            }
        else:
            # Single head output (backward compatibility)
            return self.classifier(shared_features)


if __name__ == "__main__":
    # Test backward compatibility (single head)
    print("Testing single head mode (backward compatibility):")
    model_single = SimpleCNN()
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    output_single = model_single(x)
    print(f"Input shape: {x.shape}")
    print(f"Single head output shape: {output_single.shape}")
    
    # Test multihead mode
    print("\nTesting multihead mode:")
    model_multi = SimpleCNN(heads_config={'digit': 10, 'thickness': 5, 'smoothness': 3})
    output_multi = model_multi(x)
    print(f"Multihead output type: {type(output_multi)}")
    for head_name, logits in output_multi.items():
        print(f"  {head_name}: {logits.shape}") 