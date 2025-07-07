import torch
from torch.utils.data import Dataset
from typing import Dict, Any


class MultiheadCIFARDataset(Dataset):
    """Wrapper that converts CIFAR single labels to multihead labels."""

    def __init__(self, base_dataset, dataset_type='cifar10'):
        self.base_dataset = base_dataset
        self.dataset_type = dataset_type

        # CIFAR-10 class names
        self.cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # CIFAR-100 coarse categories (20 classes)
        self.cifar100_coarse = [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_vegetables',
            'household_electrical', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor', 'large_natural_outdoor', 'large_omnivores_herbivores',
            'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
            'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        if self.dataset_type == 'cifar10':
            multihead_labels = {
                'class': label,                                    # Original class (0-9)
                'domain': self._get_domain_cifar10(label),        # Living vs non-living
                'mobility': self._get_mobility_cifar10(label),    # Mobile vs stationary
                'size': self._get_size_cifar10(label)             # Small vs large objects
            }
        else:  # cifar100
            multihead_labels = {
                'fine_class': label,                              # Original fine class (0-99)
                'coarse_class': self._get_coarse_cifar100(label), # Coarse category (0-19)
                'domain': self._get_domain_cifar100(label),       # Living vs non-living
                'complexity': self._get_complexity_cifar100(label) # Visual complexity
            }

        return image, multihead_labels

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


if __name__ == "__main__":
    # Test the mappings for CIFAR-10
    print("Testing CIFAR-10 multihead mappings:")
    print("Class | Name       | Domain | Mobility | Size")
    print("------|------------|--------|----------|-----")

    dataset = MultiheadCIFARDataset(None, 'cifar10')
    for i, class_name in enumerate(dataset.cifar10_classes):
        domain = dataset._get_domain_cifar10(i)
        mobility = dataset._get_mobility_cifar10(i)
        size = dataset._get_size_cifar10(i)
        print(f"  {i}   | {class_name:<10} |   {domain}    |    {mobility}     |  {size}")

    print("\nTesting CIFAR-100 coarse mapping (first 20 fine classes):")
    print("Fine | Coarse | Domain | Complexity")
    print("-----|--------|--------|----------")

    for fine_class in range(20):
        coarse = dataset._get_coarse_cifar100(fine_class)
        domain = dataset._get_domain_cifar100(fine_class)
        complexity = dataset._get_complexity_cifar100(fine_class)
        print(f" {fine_class:2d}  |   {coarse:2d}   |   {domain}    |     {complexity}")
