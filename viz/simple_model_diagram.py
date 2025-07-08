#!/usr/bin/env python3
"""Simple model diagram generation using PyTorch's torchviz."""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import argparse

# Set up project root and imports
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.simple_cnn import SimpleCNN

def create_model_summary(model, input_shape=(1, 1, 28, 28), model_name="Model"):
    """Create a text summary of the model architecture."""
    print("=" * 80)
    print(f"{model_name} Architecture Summary")
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
        if hasattr(model, 'conv_layers'):
            conv_out = model.conv_layers(x)
            print(f"After conv layers: {conv_out.shape}")

        # Shared features
        if hasattr(model, 'shared_features'):
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

def create_ascii_diagram():
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

def generate_from_config(config_name: str):
    """Generate diagrams from a Hydra config file."""
    print(f"\nGenerating diagrams from config: {config_name}")

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Get absolute path to configs directory
    config_dir = str(Path(__file__).parent.parent / "configs")

    try:
        # Initialize Hydra with the configs directory
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load the specific model config
            cfg = compose(config_name="train.yaml", overrides=[f"model={config_name}"])

            # Instantiate the model
            model = hydra.utils.instantiate(cfg.model)

            # Determine input shape based on config
            input_shape = (1, 1, 28, 28)  # Default for MNIST
            if 'cifar' in config_name.lower():
                input_shape = (1, 3, 32, 32)  # CIFAR input shape

            # Generate summary and diagram
            create_model_summary(model, input_shape=input_shape, model_name=f"Model: {config_name}")
            create_ascii_diagram()

    except Exception as e:
        print(f"Error loading config {config_name}: {e}")
        print("Falling back to hardcoded SimpleCNN...")

        # Fallback to hardcoded model
        model = SimpleCNN(
            input_channels=1,
            conv1_channels=3,
            conv2_channels=6,
            fc_hidden=25,
            output_size=10,
            dropout=0.25
        )

        create_model_summary(model, model_name=f"SimpleCNN (fallback for {config_name})")
        create_ascii_diagram()

def main():
    parser = argparse.ArgumentParser(description="Generate model architecture diagrams")
    parser.add_argument("--config", "-c", default="mnist_cnn_8k",
                       help="Model config name (default: mnist_cnn_8k)")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available model configs")

    args = parser.parse_args()

    if args.list_configs:
        config_path = Path("configs/model")
        if config_path.exists():
            print("Available model configs:")
            for config_file in config_path.glob("*.yaml"):
                print(f"  {config_file.stem}")
        else:
            print("No configs/model directory found")
        return

    generate_from_config(args.config)

    print("\n" + "="*80)
    print("Model diagram generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
