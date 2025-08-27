#!/usr/bin/env python3
"""Create a compact MNIST samples figure for slides."""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up project root
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def create_mnist_slide_samples(output_dir="viz/diagrams"):
    """Create compact MNIST sample images optimized for slides."""
    print("Generating compact MNIST samples for slides...")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # Select one sample per digit
    samples = []
    labels = []
    targets_found = set()

    for img, label in dataset:
        if label not in targets_found:
            samples.append(img.squeeze().numpy())
            labels.append(label)
            targets_found.add(label)
            if len(samples) >= 10:
                break

    # Sort by label for consistent display
    sorted_indices = np.argsort(labels)
    samples = [samples[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Create horizontal figure
    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    fig.suptitle('MNIST Samples (28×28 Grayscale Digits)', fontsize=12, fontweight='bold')

    for i, (ax, img, label) in enumerate(zip(axes, samples, labels)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{label}', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / "mnist_horizontal.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"MNIST horizontal samples saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    output_dir = "viz/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    create_mnist_slide_samples(output_dir)
    print("MNIST slide samples generated successfully!")
