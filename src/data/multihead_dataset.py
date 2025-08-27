import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class MultiheadLabelStrategy(ABC):
    """Abstract base class for multihead label generation strategies."""

    @abstractmethod
    def get_multihead_labels(self, label: int) -> Dict[str, int]:
        """Convert single label to multihead labels dict."""
        pass


class MNISTStrategy(MultiheadLabelStrategy):
    """Strategy for MNIST digit classification with auxiliary tasks."""

    def get_multihead_labels(self, digit_label: int) -> Dict[str, int]:
        return {
            'digit': digit_label,                           # Original digit (0-9)
            'thickness': self._get_thickness(digit_label),  # Digit-dependent thickness
            'smoothness': self._get_smoothness(digit_label) # Digit-dependent smoothness
        }

    def _get_thickness(self, digit: int) -> int:
        """Map digit to thickness class based on stroke complexity.

        Odd digits tend to be thicker (more complex strokes)
        Even digits tend to be thinner (simpler strokes)

        Returns:
            int: thickness class (0-4) where 0=very thin, 4=very thick
        """
        if digit % 2 == 1:  # Odd: 1,3,5,7,9
            return min(4, (digit // 2) + 2)  # Maps to 2,3,4,4,4
        else:  # Even: 0,2,4,6,8
            return digit // 2  # Maps to 0,1,2,3,4

    def _get_smoothness(self, digit: int) -> int:
        """Map digit to smoothness class based on stroke curvature.

        Angular digits: 1,4,7 → very angular
        Curved digits: 0,3,6,8,9 → very smooth
        Mixed digits: 2,5 → medium smoothness

        Returns:
            int: smoothness class (0-2) where 0=angular, 1=medium, 2=smooth
        """
        if digit in [1, 4, 7]:
            return 0  # Angular
        elif digit in [0, 3, 6, 8, 9]:
            return 2  # Smooth
        else:  # 2, 5
            return 1  # Medium


class MultiheadDataset(Dataset):
    """Generic wrapper that converts single labels to multihead labels using strategies."""

    def __init__(self, base_dataset, dataset_type: str):
        """Initialize multihead dataset with specified strategy.

        Args:
            base_dataset: The underlying dataset to wrap
            dataset_type: Type of dataset ('mnist')
        """
        self.base_dataset = base_dataset
        self.dataset_type = dataset_type

        # Create strategy based on dataset type
        if dataset_type == 'mnist':
            self.strategy = MNISTStrategy()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        multihead_labels = self.strategy.get_multihead_labels(label)
        return image, multihead_labels


# Backward compatibility aliases
class MultiheadMNISTDataset(MultiheadDataset):
    """Backward compatibility wrapper for MNIST multihead dataset."""

    def __init__(self, base_dataset):
        super().__init__(base_dataset, 'mnist')


if __name__ == "__main__":
    # Test all strategies
    print("Testing MNIST multihead mappings:")
    print("Digit | Thickness | Smoothness")
    print("------|-----------|----------")

    mnist_strategy = MNISTStrategy()
    for digit in range(10):
        labels = mnist_strategy.get_multihead_labels(digit)
        print(f"  {digit}   |     {labels['thickness']}     |     {labels['smoothness']}")

