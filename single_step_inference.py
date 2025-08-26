#!/usr/bin/env python3
"""
Single-step inference script for loading trained models and running forward passes.
"""

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import rootutils

# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.multihead_module import MultiheadLitModule
from src.models.mnist_module import MNISTLitModule
from src.data.cifar10_datamodule import CIFAR10DataModule
from src.data.mnist_datamodule import MNISTDataModule


def load_model_from_checkpoint(checkpoint_path: str, model_class=None):
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        model_class: Optional model class (auto-detected if None)

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from: {checkpoint_path}")

    # Try to auto-detect model type from checkpoint
    if model_class is None:
        # You can add logic here to detect model type from checkpoint metadata
        # For now, assume MultiheadLitModule as default
        model_class = MultiheadLitModule

    # Load model from checkpoint
    model = model_class.load_from_checkpoint(checkpoint_path)
    model.eval()  # Set to evaluation mode

    return model


def single_forward_pass(model, input_tensor: torch.Tensor):
    """Perform a single forward pass through the model.

    Args:
        model: Loaded model
        input_tensor: Input tensor (single sample or batch)

    Returns:
        Model predictions
    """
    with torch.no_grad():  # Disable gradient computation for inference
        output = model.forward(input_tensor)

    return output


def process_predictions(output, model):
    """Process raw model output into interpretable predictions.

    Args:
        output: Raw model output (logits or dict of logits)
        model: The model (for accessing prediction methods)

    Returns:
        Processed predictions
    """
    if isinstance(output, dict):
        # Multihead model - process each head
        predictions = {}
        for head_name, logits in output.items():
            if hasattr(model, '_compute_predictions'):
                # Use model's prediction computation if available
                head_criterion = model.criteria.get(head_name)
                if head_criterion:
                    predictions[head_name] = model._compute_predictions(
                        logits, head_criterion, head_name
                    )
                else:
                    # Fallback to argmax for classification
                    predictions[head_name] = torch.argmax(logits, dim=1)
            else:
                predictions[head_name] = torch.argmax(logits, dim=1)
        return predictions
    else:
        # Single head model
        return torch.argmax(output, dim=1)


def demo_mnist_inference():
    """Demo inference on MNIST-like data."""
    print("\n" + "="*50)
    print("MNIST Inference Demo")
    print("="*50)

    # Create dummy MNIST-like input
    batch_size = 3
    input_tensor = torch.randn(batch_size, 1, 28, 28)

    # For demo, create a simple model instead of loading checkpoint
    # In practice, you'd use: model = load_model_from_checkpoint("path/to/checkpoint.ckpt")
    from src.models.components.simple_cnn import SimpleCNN
    from src.models.mnist_module import MNISTLitModule

    # Create model (normally you'd load from checkpoint)
    net = SimpleCNN(heads_config={'digit': 10})
    model = MNISTLitModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.StepLR,
        criteria={'digit': torch.nn.CrossEntropyLoss()}
    )
    model.eval()

    print(f"Input shape: {input_tensor.shape}")

    # Forward pass
    output = single_forward_pass(model, input_tensor)
    predictions = process_predictions(output, model)

    print(f"Raw output type: {type(output)}")
    if isinstance(output, dict):
        for head_name, logits in output.items():
            print(f"  {head_name} logits shape: {logits.shape}")
            print(f"  {head_name} predictions: {predictions[head_name]}")
    else:
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {predictions}")


def demo_single_sample_inference():
    """Demo inference on a single sample."""
    print("\n" + "="*50)
    print("Single Sample Inference Demo")
    print("="*50)

    # Single sample (add batch dimension)
    single_sample = torch.randn(1, 1, 28, 28)  # Batch size = 1

    from src.models.components.simple_cnn import SimpleCNN
    from src.models.multihead_module import MultiheadLitModule

    # Create multihead model
    net = SimpleCNN(heads_config={'digit': 10, 'thickness': 5})
    model = MultiheadLitModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.StepLR,
        criteria={
            'digit': torch.nn.CrossEntropyLoss(),
            'thickness': torch.nn.CrossEntropyLoss()
        },
        loss_weights={'digit': 1.0, 'thickness': 1.0}
    )
    model.eval()

    print(f"Single sample shape: {single_sample.shape}")

    # Forward pass
    output = single_forward_pass(model, single_sample)
    predictions = process_predictions(output, model)

    print("Multihead predictions:")
    for head_name, pred in predictions.items():
        print(f"  {head_name}: {pred.item()}")  # .item() for single values


def inference_from_checkpoint_example():
    """Example of loading from an actual checkpoint."""
    print("\n" + "="*50)
    print("Checkpoint Loading Example")
    print("="*50)

    # This would be used with an actual checkpoint file
    checkpoint_path = "logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/epoch_XXX.ckpt"

    print("To use with a real checkpoint:")
    print(f"1. Train a model: python src/train.py")
    print(f"2. Find checkpoint in: logs/train/runs/*/checkpoints/")
    print(f"3. Load with: model = load_model_from_checkpoint('{checkpoint_path}')")
    print(f"4. Run inference: output = single_forward_pass(model, your_input)")

    # Example usage code:
    example_code = '''
# Load trained model
model = load_model_from_checkpoint("path/to/your/checkpoint.ckpt")

# Prepare your input (single sample or batch)
input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 example

# Run inference
output = single_forward_pass(model, input_tensor)
predictions = process_predictions(output, model)

print(f"Predictions: {predictions}")
'''
    print("\nExample code:")
    print(example_code)


if __name__ == "__main__":
    print("Single-Step Inference Examples")
    print("This script demonstrates how to use your forward methods for inference.")

    # Run demos
    demo_mnist_inference()
    demo_single_sample_inference()
    inference_from_checkpoint_example()

    print("\n" + "="*50)
    print("✅ Inference demos completed!")
    print("Use the load_model_from_checkpoint() and single_forward_pass() functions")
    print("with your actual trained checkpoints.")
