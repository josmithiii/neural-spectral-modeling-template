import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A simple convolutional neural network for MNIST classification."""

    def __init__(
        self,
        input_channels: int = 1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc_hidden: int = 128,
        output_size: int = 10,
        dropout: float = 0.25,
    ) -> None:
        """Initialize a SimpleCNN module.

        :param input_channels: Number of input channels (1 for grayscale MNIST).
        :param conv1_channels: Number of output channels for first conv layer.
        :param conv2_channels: Number of output channels for second conv layer.
        :param fc_hidden: Number of hidden units in fully connected layer.
        :param output_size: Number of output classes.
        :param dropout: Dropout probability.
        """
        super().__init__()

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
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_channels * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: A tensor of logits.
        """
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test the model
    model = SimpleCNN()
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 