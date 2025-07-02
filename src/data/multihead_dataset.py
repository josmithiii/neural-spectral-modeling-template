import torch
from torch.utils.data import Dataset
from typing import Dict, Any


class MultiheadMNISTDataset(Dataset):
    """Wrapper that converts MNIST single labels to multihead labels."""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, digit_label = self.base_dataset[idx]
        
        # Create multiple labels from the digit
        multihead_labels = {
            'digit': digit_label,                           # Original digit (0-9)
            'thickness': self._get_thickness(digit_label),  # Digit-dependent thickness
            'smoothness': self._get_smoothness(digit_label) # Digit-dependent smoothness
        }
        
        return image, multihead_labels
    
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


if __name__ == "__main__":
    # Test the mappings
    print("Testing digit-to-multihead mappings:")
    print("Digit | Thickness | Smoothness")
    print("------|-----------|----------")
    
    for digit in range(10):
        dataset = MultiheadMNISTDataset(None)  # We only need the methods
        thickness = dataset._get_thickness(digit)
        smoothness = dataset._get_smoothness(digit)
        print(f"  {digit}   |     {thickness}     |     {smoothness}")