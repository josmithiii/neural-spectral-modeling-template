#!/usr/bin/env python3
"""Simple model diagram generation using PyTorch's torchviz."""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.components.simple_cnn import SimpleCNN

def create_model_summary(model, input_shape=(1, 1, 28, 28)):
    """Create a text summary of the model architecture."""
    print("=" * 80)
    print(f"Model Architecture Summary")
    print("=" * 80)

    # Print model structure
    print("\nModel Structure:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter Count:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass and show shapes
    print(f"\nForward Pass Shape Analysis:")
    print(f"Input shape: {input_shape}")

    with torch.no_grad():
        x = torch.randn(*input_shape)

        # Track intermediate shapes
        print(f"Input: {x.shape}")

        # Conv layers
        conv_out = model.conv_layers(x)
        print(f"After conv layers: {conv_out.shape}")

        # Shared features
        shared_out = model.shared_features(conv_out)
        print(f"After shared features: {shared_out.shape}")

        # Final output
        final_out = model(x)
        if isinstance(final_out, dict):
            print(f"Final output (multihead):")
            for head_name, logits in final_out.items():
                print(f"  {head_name}: {logits.shape}")
        else:
            print(f"Final output: {final_out.shape}")

def create_ascii_diagram(model):
    """Create an ASCII diagram of the model architecture."""
    print("\n" + "="*80)
    print("ASCII Architecture Diagram")
    print("="*80)

    print("""
    Input (1x28x28)
           │
           ▼
    ┌─────────────────┐
    │   Conv2d(1→3)   │  3x3 kernel, padding=1
    │   BatchNorm2d   │
    │   ReLU          │
    │   MaxPool2d     │  2x2, stride=2
    └─────────────────┘
           │ (3x14x14)
           ▼
    ┌─────────────────┐
    │   Conv2d(3→6)   │  3x3 kernel, padding=1
    │   BatchNorm2d   │
    │   ReLU          │
    │   MaxPool2d     │  2x2, stride=2
    └─────────────────┘
           │ (6x7x7)
           ▼
    ┌─────────────────┐
    │ AdaptiveAvgPool │  → (6x7x7)
    │    Flatten      │  → (294,)
    │  Linear(294→25) │
    │      ReLU       │
    │   Dropout(0.25) │
    └─────────────────┘
           │ (25,)
           ▼
    ┌─────────────────┐
    │  Linear(25→10)  │  Final classifier
    └─────────────────┘
           │
           ▼
        Output (10,)
    """)

def main():
    # Create model with configuration from mnist_cnn_8k.yaml
    model = SimpleCNN(
        input_channels=1,
        conv1_channels=3,
        conv2_channels=6,
        fc_hidden=25,
        output_size=10,
        dropout=0.25
    )

    # Generate summary and diagram
    create_model_summary(model)
    create_ascii_diagram(model)

    print("\n" + "="*80)
    print("Model diagram generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
