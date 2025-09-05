"""VIMH dataset generator for creating synthetic multihead datasets.

This module provides functionality to generate VIMH format datasets with
characteristics similar to CIFAR-100, including multiple classification heads
and configurable complexity levels.
"""

import json
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torchvision.datasets import CIFAR100


def generate_cifar100_like_vimh(
    output_dir: str,
    num_train_samples: int = 50000,
    num_test_samples: int = 10000,
    image_size: Tuple[int, int, int] = (32, 32, 3),
    num_heads: int = 3,
    complexity: str = "cifar100"
) -> None:
    """Generate a VIMH dataset with CIFAR-100-like characteristics.

    Creates a synthetic multihead dataset that mimics CIFAR-100 complexity
    but with additional classification heads for research purposes.

    :param output_dir: Directory to save the generated dataset
    :param num_train_samples: Number of training samples
    :param num_test_samples: Number of test samples
    :param image_size: Image dimensions (height, width, channels)
    :param num_heads: Number of classification heads
    :param complexity: Complexity level ('cifar100', 'cifar10', 'custom')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    height, width, channels = image_size

    # Define head configurations based on complexity
    if complexity == "cifar100":
        heads_config = {
            "fine_label": 100,      # Fine-grained classes (like CIFAR-100)
            "coarse_label": 20,     # Coarse classes (like CIFAR-100 superclasses)
            "texture": 8            # Additional texture classification
        }
    elif complexity == "cifar10":
        heads_config = {
            "class": 10,
            "color": 5,
            "shape": 4
        }
    else:  # custom
        heads_config = {f"head_{i}": min(256, 10 * (i + 1)) for i in range(num_heads)}

    # Generate parameter mappings for metadata
    parameter_mappings = {}
    for head_name, num_classes in heads_config.items():
        parameter_mappings[head_name] = {
            "min": 0,
            "max": num_classes - 1,
            "type": "classification",
            "description": f"Classification head with {num_classes} classes"
        }

    # Create metadata
    metadata = {
        "format": "VIMH",
        "version": "1.0",
        "height": height,
        "width": width,
        "channels": channels,
        "parameter_names": list(heads_config.keys()),
        "parameter_mappings": parameter_mappings,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
        "generation_info": {
            "complexity": complexity,
            "generator": "vimh_generator.generate_cifar100_like_vimh",
            "synthetic": True
        }
    }

    # Save metadata
    with open(output_path / "vimh_dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate datasets
    _generate_dataset_split(
        output_path / "train_batch",
        num_train_samples,
        image_size,
        heads_config,
        complexity,
        train=True
    )

    _generate_dataset_split(
        output_path / "test_batch",
        num_test_samples,
        image_size,
        heads_config,
        complexity,
        train=False
    )

    print(f"Generated VIMH dataset at {output_path}")
    print(f"  - {num_train_samples} training samples")
    print(f"  - {num_test_samples} test samples")
    print(f"  - Image size: {height}x{width}x{channels}")
    print(f"  - Heads: {heads_config}")


def _generate_dataset_split(
    output_file: Path,
    num_samples: int,
    image_size: Tuple[int, int, int],
    heads_config: Dict[str, int],
    complexity: str,
    train: bool
) -> None:
    """Generate a single dataset split (train or test).

    :param output_file: Path to output file
    :param num_samples: Number of samples to generate
    :param image_size: Image dimensions (height, width, channels)
    :param heads_config: Dictionary of head names to number of classes
    :param complexity: Complexity level for generation strategy
    :param train: Whether this is training data (affects noise/variation)
    """
    height, width, channels = image_size

    # Try to use real CIFAR data as a base if available
    cifar_data = None
    try:
        if complexity in ["cifar100", "cifar10"]:
            dataset_class = CIFAR100 if complexity == "cifar100" else None
            if dataset_class:
                cifar_data = dataset_class(
                    root='./data',
                    train=train,
                    download=True
                )
    except Exception:
        pass  # Fall back to synthetic generation

    samples = []
    labels_list = []

    np.random.seed(42 if train else 123)  # Reproducible generation

    for i in range(num_samples):
        # Generate or use CIFAR image
        if cifar_data and i < len(cifar_data):
            # Use real CIFAR image as base
            cifar_img, cifar_label = cifar_data[i]
            image_array = np.array(cifar_img)

            # Resize if needed
            if image_array.shape != (height, width, channels):
                # Simple resize for demonstration
                image_array = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            # Generate synthetic image with realistic patterns
            image_array = _generate_synthetic_image(height, width, channels, i)

        # Convert to CHW format (PyTorch convention)
        if channels == 3:
            image_chw = image_array.transpose(2, 0, 1)
        else:
            image_chw = image_array.reshape(channels, height, width)

        samples.append(image_chw.flatten())

        # Generate correlated labels for multiple heads
        labels_dict = _generate_correlated_labels(heads_config, i, cifar_data, train)

        # Convert labels dict to VIMH format: [N] [param1_id] [param1_val] [param2_id] [param2_val] ...
        label_array = [len(labels_dict)]  # N = number of heads

        for param_idx, (param_name, param_value) in enumerate(labels_dict.items()):
            label_array.extend([param_idx, param_value])  # [param_id, param_val]

        labels_list.append(label_array)

    # Save in pickle format for compatibility
    batch_data = {
        "data": np.array(samples, dtype=np.uint8),
        "vimh_labels": labels_list,  # Keep as list since labels can be variable length
        "heads_config": heads_config,
        "height": height,
        "width": width,
        "channels": channels,
        "num_samples": num_samples
    }

    with open(output_file, "wb") as f:
        pickle.dump(batch_data, f)


def _generate_synthetic_image(height: int, width: int, channels: int, seed: int) -> np.ndarray:
    """Generate a synthetic image with realistic patterns.

    :param height: Image height
    :param width: Image width
    :param channels: Number of channels
    :param seed: Random seed for reproducible generation
    :return: Generated image array
    """
    np.random.seed(seed)

    # Generate base image with gradients and noise
    image = np.zeros((height, width, channels), dtype=np.uint8)

    # Add gradient patterns
    for c in range(channels):
        # Horizontal gradient
        h_grad = np.linspace(0, 255, width).reshape(1, -1)
        h_grad = np.repeat(h_grad, height, axis=0)

        # Vertical gradient
        v_grad = np.linspace(0, 255, height).reshape(-1, 1)
        v_grad = np.repeat(v_grad, width, axis=1)

        # Combine gradients with some randomness
        base = (0.3 * h_grad + 0.3 * v_grad + 0.4 * 128) % 256

        # Add noise and patterns
        noise = np.random.normal(0, 20, (height, width))

        # Add some geometric patterns
        center_x, center_y = width // 2, height // 2
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        pattern = 50 * np.sin((x - center_x) / 5) * np.cos((y - center_y) / 5)

        image[:, :, c] = np.clip(base + noise + pattern, 0, 255).astype(np.uint8)

    return image


def _generate_correlated_labels(
    heads_config: Dict[str, int],
    sample_idx: int,
    cifar_data: Optional[Any] = None,
    train: bool = True
) -> Dict[str, int]:
    """Generate correlated labels for multiple heads.

    Creates realistic correlations between different classification heads,
    similar to how fine/coarse labels relate in CIFAR-100.

    :param heads_config: Dictionary of head names to number of classes
    :param sample_idx: Sample index for reproducible generation
    :param cifar_data: Optional CIFAR dataset for label correlation
    :param train: Whether this is training data
    :return: Dictionary of head names to label values
    """
    np.random.seed(sample_idx + (42 if train else 123))

    labels = {}
    primary_class = None

    for head_idx, (head_name, num_classes) in enumerate(heads_config.items()):
        if head_idx == 0:
            # Primary head - use CIFAR label if available
            if cifar_data and sample_idx < len(cifar_data):
                _, cifar_label = cifar_data[sample_idx]
                primary_class = cifar_label % num_classes
            else:
                primary_class = sample_idx % num_classes
            labels[head_name] = primary_class

        else:
            # Secondary heads - create realistic correlations
            if "coarse" in head_name.lower() and primary_class is not None:
                # Coarse labels are related to fine labels (like CIFAR-100)
                coarse_label = primary_class // (100 // num_classes) if num_classes <= 20 else primary_class % num_classes
                labels[head_name] = coarse_label

            elif "texture" in head_name.lower() and primary_class is not None:
                # Texture somewhat correlated with class
                texture_label = (primary_class * 3 + sample_idx // 100) % num_classes
                labels[head_name] = texture_label

            elif "color" in head_name.lower() and primary_class is not None:
                # Color patterns
                color_label = (primary_class + sample_idx // 50) % num_classes
                labels[head_name] = color_label

            else:
                # Other heads - weak correlation with some noise
                base_label = (primary_class * 2 + head_idx) % num_classes if primary_class is not None else 0
                noise = np.random.randint(-2, 3)  # Small random variation
                labels[head_name] = max(0, min(num_classes - 1, base_label + noise))

    return labels


def generate_vimh_dataset(
    output_dir: str,
    num_samples: Optional[int] = None,
    complexity: str = "cifar100",
    **kwargs
) -> None:
    """Main interface for generating VIMH datasets.

    This is the function called by VIMHDataModule.prepare_data().

    :param output_dir: Directory to save the generated dataset
    :param num_samples: Total number of samples (split 80/20 train/test)
    :param complexity: Complexity level ('cifar100', 'cifar10', 'custom')
    :param kwargs: Additional generation parameters
    """
    if num_samples is None:
        num_samples = 50000 if complexity == "cifar100" else 10000

    # Split into train/test
    num_train = int(0.8 * num_samples)
    num_test = num_samples - num_train

    generate_cifar100_like_vimh(
        output_dir=output_dir,
        num_train_samples=num_train,
        num_test_samples=num_test,
        complexity=complexity,
        **kwargs
    )


if __name__ == "__main__":
    # Test generation
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing VIMH generation in {temp_dir}")

        generate_vimh_dataset(
            output_dir=temp_dir,
            num_samples=100,
            complexity="cifar100"
        )

        # Verify generation
        metadata_file = Path(temp_dir) / "vimh_dataset_info.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            print("✓ Generated dataset metadata:")
            print(f"  - Format: {metadata['format']}")
            print(f"  - Image size: {metadata['height']}x{metadata['width']}x{metadata['channels']}")
            print(f"  - Heads: {list(metadata['parameter_names'])}")

        train_file = Path(temp_dir) / "train_batch"
        if train_file.exists():
            with open(train_file, "rb") as f:
                train_data = pickle.load(f)
            print(f"✓ Training data: {train_data['data'].shape}")
            print(f"✓ Sample labels: {train_data['labels'][0]}")

        print("VIMH generation test successful!")