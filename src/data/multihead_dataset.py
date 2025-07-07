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


class CIFAR10Strategy(MultiheadLabelStrategy):
    """Strategy for CIFAR-10 classification with semantic auxiliary tasks."""

    def __init__(self):
        # CIFAR-10 class names for reference
        self.cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

    def get_multihead_labels(self, class_label: int) -> Dict[str, int]:
        return {
            'class': class_label,                                # Original class (0-9)
            'domain': self._get_domain_cifar10(class_label),     # Living vs non-living
            'mobility': self._get_mobility_cifar10(class_label), # Mobile vs stationary
            'size': self._get_size_cifar10(class_label)          # Small vs large objects
        }

    def _get_domain_cifar10(self, class_id: int) -> int:
        """Map CIFAR-10 class to domain (living vs non-living).

        Returns:
            int: 0=non-living, 1=living
        """
        living_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
        return 1 if class_id in living_classes else 0

    def _get_mobility_cifar10(self, class_id: int) -> int:
        """Map CIFAR-10 class to mobility (mobile vs stationary).

        Returns:
            int: 0=stationary, 1=mobile, 2=highly_mobile
        """
        highly_mobile = [0, 1, 2]  # airplane, automobile, bird
        mobile = [3, 4, 5, 7, 8]   # cat, deer, dog, horse, ship
        # stationary = [6, 9]       # frog, truck

        if class_id in highly_mobile:
            return 2
        elif class_id in mobile:
            return 1
        else:
            return 0

    def _get_size_cifar10(self, class_id: int) -> int:
        """Map CIFAR-10 class to typical size category.

        Returns:
            int: 0=small, 1=medium, 2=large
        """
        small = [2, 3, 6]      # bird, cat, frog
        large = [0, 1, 4, 7, 8, 9]  # airplane, automobile, deer, horse, ship, truck
        # medium = [5]           # dog

        if class_id in small:
            return 0
        elif class_id == 5:    # dog
            return 1
        else:
            return 2


class CIFAR100Strategy(MultiheadLabelStrategy):
    """Strategy for CIFAR-100 classification with hierarchical auxiliary tasks."""

    def __init__(self):
        # CIFAR-100 coarse categories (20 classes)
        self.cifar100_coarse = [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_vegetables',
            'household_electrical', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor', 'large_natural_outdoor', 'large_omnivores_herbivores',
            'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
            'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]

    def get_multihead_labels(self, fine_label: int) -> Dict[str, int]:
        return {
            'fine_class': fine_label,                              # Original fine class (0-99)
            'coarse_class': self._get_coarse_cifar100(fine_label), # Coarse category (0-19)
            'domain': self._get_domain_cifar100(fine_label),       # Living vs non-living
            'complexity': self._get_complexity_cifar100(fine_label) # Visual complexity
        }

    def _get_coarse_cifar100(self, fine_class: int) -> int:
        """Map CIFAR-100 fine class to coarse class.

        This uses the standard CIFAR-100 coarse/fine mapping.
        """
        # CIFAR-100 coarse mapping (5 fine classes per coarse class)
        return fine_class // 5

    def _get_domain_cifar100(self, fine_class: int) -> int:
        """Map CIFAR-100 fine class to domain (living vs non-living).

        Returns:
            int: 0=non-living, 1=living
        """
        coarse_class = self._get_coarse_cifar100(fine_class)
        living_coarse = [1, 7, 8, 11, 12, 13, 14, 15, 16]  # fish, insects, carnivores, etc.
        return 1 if coarse_class in living_coarse else 0

    def _get_complexity_cifar100(self, fine_class: int) -> int:
        """Map CIFAR-100 fine class to visual complexity.

        Returns:
            int: 0=simple, 1=medium, 2=complex
        """
        coarse_class = self._get_coarse_cifar100(fine_class)

        # Simple: geometric objects, vehicles
        simple_coarse = [3, 18, 19]  # food_containers, vehicles_1, vehicles_2

        # Complex: natural scenes, living things
        complex_coarse = [1, 2, 4, 7, 8, 10, 11, 13, 14, 15, 16, 17]

        if coarse_class in simple_coarse:
            return 0
        elif coarse_class in complex_coarse:
            return 2
        else:
            return 1


class MultiheadDataset(Dataset):
    """Generic wrapper that converts single labels to multihead labels using strategies."""

    def __init__(self, base_dataset, dataset_type: str):
        """Initialize multihead dataset with specified strategy.

        Args:
            base_dataset: The underlying dataset to wrap
            dataset_type: Type of dataset ('mnist', 'cifar10', 'cifar100')
        """
        self.base_dataset = base_dataset
        self.dataset_type = dataset_type

        # Create strategy based on dataset type
        if dataset_type == 'mnist':
            self.strategy = MNISTStrategy()
        elif dataset_type == 'cifar10':
            self.strategy = CIFAR10Strategy()
        elif dataset_type == 'cifar100':
            self.strategy = CIFAR100Strategy()
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


class MultiheadCIFARDataset(MultiheadDataset):
    """Backward compatibility wrapper for CIFAR multihead dataset."""

    def __init__(self, base_dataset, dataset_type='cifar10'):
        super().__init__(base_dataset, dataset_type)


if __name__ == "__main__":
    # Test all strategies
    print("Testing MNIST multihead mappings:")
    print("Digit | Thickness | Smoothness")
    print("------|-----------|----------")

    mnist_strategy = MNISTStrategy()
    for digit in range(10):
        labels = mnist_strategy.get_multihead_labels(digit)
        print(f"  {digit}   |     {labels['thickness']}     |     {labels['smoothness']}")

    print("\nTesting CIFAR-10 multihead mappings:")
    print("Class | Name       | Domain | Mobility | Size")
    print("------|------------|--------|----------|-----")

    cifar10_strategy = CIFAR10Strategy()
    for i, class_name in enumerate(cifar10_strategy.cifar10_classes):
        labels = cifar10_strategy.get_multihead_labels(i)
        print(f"  {i}   | {class_name:<10} |   {labels['domain']}    |    {labels['mobility']}     |  {labels['size']}")

    print("\nTesting CIFAR-100 coarse mapping (first 20 fine classes):")
    print("Fine | Coarse | Domain | Complexity")
    print("-----|--------|--------|----------")

    cifar100_strategy = CIFAR100Strategy()
    for fine_class in range(20):
        labels = cifar100_strategy.get_multihead_labels(fine_class)
        print(f" {fine_class:2d}  |   {labels['coarse_class']:2d}   |   {labels['domain']}    |     {labels['complexity']}")
