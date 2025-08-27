#!/usr/bin/env python3
"""Generate sample images from datasets for presentation slides."""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up project root
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def create_mnist_samples(output_dir="viz/diagrams", num_samples=8):
    """Create MNIST sample images for presentation."""
    print("Generating MNIST sample images...")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # Select diverse samples (one per digit)
    samples = []
    labels = []
    targets_found = set()

    for img, label in dataset:
        if label not in targets_found:
            samples.append(img.squeeze().numpy())
            labels.append(label)
            targets_found.add(label)
            if len(samples) >= num_samples:
                break

    # Sort by label for consistent display
    sorted_indices = np.argsort(labels)
    samples = [samples[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('MNIST Dataset Samples (Handwritten Digits)', fontsize=14, fontweight='bold')

    for i, (ax, img, label) in enumerate(zip(axes.flat, samples, labels)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Digit {label}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / "mnist_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"MNIST samples saved to {output_path}")
    plt.close()

def create_cifar10_samples(output_dir="viz/diagrams", num_samples=10):
    """Create CIFAR-10 sample images for presentation."""
    print("Generating CIFAR-10 sample images...")

    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Select one sample per class
    samples = []
    labels = []
    targets_found = set()

    for img, label in dataset:
        if label not in targets_found:
            # Convert tensor to numpy and transpose for matplotlib
            img_np = img.permute(1, 2, 0).numpy()
            samples.append(img_np)
            labels.append(label)
            targets_found.add(label)
            if len(samples) >= num_samples:
                break

    # Sort by label
    sorted_indices = np.argsort(labels)
    samples = [samples[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('CIFAR-10 Dataset Samples (32×32 RGB Objects)', fontsize=14, fontweight='bold')

    for i, (ax, img, label) in enumerate(zip(axes.flat, samples, labels)):
        ax.imshow(img)
        ax.set_title(f'{classes[label]}', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / "cifar10_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"CIFAR-10 samples saved to {output_path}")
    plt.close()

def create_cifar100_samples(output_dir="viz/diagrams", num_samples=10):
    """Create CIFAR-100 sample images showing fine-grained classification."""
    print("Generating CIFAR-100 sample images...")

    # Load CIFAR-100 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )

    # Get some diverse samples
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    samples = []
    labels = []

    for idx in indices:
        img, label = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        samples.append(img_np)
        labels.append(label)

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('CIFAR-100 Dataset Samples (100 Fine-Grained Classes)', fontsize=14, fontweight='bold')

    for i, (ax, img, label) in enumerate(zip(axes.flat, samples, labels)):
        ax.imshow(img)
        ax.set_title(f'Class {label}', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / "cifar100_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"CIFAR-100 samples saved to {output_path}")
    plt.close()

def create_architecture_diagrams(output_dir="viz/diagrams"):
    """Create simple architecture diagrams."""
    print("Creating architecture diagrams...")

    # CNN Architecture
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define architecture components
    components = [
        ("Input\n28×28×1", 0, 6, 'lightblue'),
        ("Conv1\n3×3, 32→64", 2, 6, 'lightgreen'),
        ("MaxPool\n2×2", 4, 6, 'lightcoral'),
        ("Conv2\n3×3, 64→128", 6, 6, 'lightgreen'),
        ("MaxPool\n2×2", 8, 6, 'lightcoral'),
        ("AdaptivePool\n7×7", 10, 6, 'lightyellow'),
        ("Flatten\n→ 6272", 12, 6, 'lightgray'),
        ("FC\n→ 128", 14, 6, 'lightpink'),
        ("Output\n→ 10", 16, 6, 'orange')
    ]

    # Draw components
    for name, x, y, color in components:
        rect = plt.Rectangle((x, y-0.5), 1.5, 1, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+0.75, y, name, ha='center', va='center', fontsize=10, fontweight='bold')

        # Add arrows between components (except last)
        if x < 16:
            ax.arrow(x+1.5, y, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    ax.set_xlim(-0.5, 18)
    ax.set_ylim(4.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SimpleCNN Architecture Flow', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = Path(output_dir) / "cnn_architecture.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"CNN architecture diagram saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    output_dir = "viz/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    create_mnist_samples(output_dir)
    create_cifar10_samples(output_dir)
    create_cifar100_samples(output_dir)
    create_architecture_diagrams(output_dir)

    print("\nAll diagrams and samples generated successfully!")
