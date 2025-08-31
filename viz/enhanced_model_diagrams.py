#!/usr/bin/env python3
"""Enhanced model diagram generation with both text and graphical output."""

import torch
import torch.nn as nn
from torchviz import make_dot
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

def create_text_summary(model, input_shape=(1, 1, 28, 28), model_name="Model"):
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

def create_graphical_diagram(model, input_shape=(1, 1, 28, 28), model_name="model", output_dir="diagrams"):
    """Create a graphical diagram using torchviz."""
    print(f"\nGenerating graphical diagram for {model_name}...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create sample input
    x = torch.randn(*input_shape, requires_grad=True)

    # Forward pass
    y = model(x)

    # Handle multihead output
    if isinstance(y, dict):
        # For multihead, visualize the first head
        first_head = next(iter(y.values()))
        dot = make_dot(first_head, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        dot.graph_attr['label'] = f'{model_name} (Multihead - First Head)'
    else:
        dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        dot.graph_attr['label'] = f'{model_name} Architecture'

    # Customize appearance
    dot.graph_attr['rankdir'] = 'TB'  # Top to bottom
    dot.graph_attr['size'] = '12,16'
    dot.graph_attr['dpi'] = '300'

    # Save the diagram
    filename = f"{model_name.lower().replace(' ', '_')}_graph"
    dot.render(output_path / filename, format='png', cleanup=True)
    dot.render(output_path / filename, format='pdf', cleanup=True)

    print(f"Saved graphical diagrams:")
    print(f"  PNG: {output_path / filename}.png")
    print(f"  PDF: {output_path / filename}.pdf")

    return dot

def create_ascii_diagram_cnn():
    """Create an ASCII diagram for CNN architecture."""
    print("\n" + "="*80)
    print("CNN Architecture Flow Diagram")
    print("="*80)

    print("""
    Input (1×28×28) MNIST Image
           │
           ▼
    ┌─────────────────────────────┐
    │     CONV BLOCK 1            │
    │  Conv2d(1→3, 3×3, pad=1)    │  ← 3×3 convolution
    │  BatchNorm2d(3)             │  ← normalize activations
    │  ReLU()                     │  ← activation function
    │  MaxPool2d(2×2, stride=2)   │  ← downsample by 2
    └─────────────────────────────┘
           │ (3×14×14)
           ▼
    ┌─────────────────────────────┐
    │     CONV BLOCK 2            │
    │  Conv2d(3→6, 3×3, pad=1)    │  ← double channels
    │  BatchNorm2d(6)             │  ← normalize activations
    │  ReLU()                     │  ← activation function
    │  MaxPool2d(2×2, stride=2)   │  ← downsample by 2
    └─────────────────────────────┘
           │ (6×7×7)
           ▼
    ┌─────────────────────────────┐
    │   FEATURE EXTRACTION        │
    │  AdaptiveAvgPool2d(7×7)     │  ← ensure 7×7 output
    │  Flatten()                  │  ← reshape to 1D
    └─────────────────────────────┘
           │ (294,) = 6×7×7
           ▼
    ┌─────────────────────────────┐
    │    CLASSIFIER HEAD          │
    │  Linear(294→25)             │  ← hidden layer
    │  ReLU()                     │  ← activation
    │  Dropout(0.25)              │  ← regularization
    │  Linear(25→10)              │  ← output layer
    └─────────────────────────────┘
           │
           ▼
        Output (10,) Class Logits
    """)

def generate_from_config(config_name: str, output_dir: str = "diagrams"):
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

            # Generate both text and graphical diagrams
            create_text_summary(model, model_name=f"Model: {config_name}")
            create_ascii_diagram_cnn()
            create_graphical_diagram(model, model_name=config_name, output_dir=output_dir)

    except Exception as e:
        print(f"Error loading config {config_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate model architecture diagrams")
    parser.add_argument("--config", "-c", default=None,
                       help="Model config name (default: mnist_cnn_8k)")
    parser.add_argument("--output", "-o", default="diagrams",
                       help="Output directory for diagrams (default: diagrams)")
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

    generate_from_config(args.config, args.output)

    print(f"\n{'='*80}")
    print("Model diagram generation complete!")
    print(f"Check the '{args.output}' directory for graphical outputs")
    print("="*80)

if __name__ == "__main__":
    main()
