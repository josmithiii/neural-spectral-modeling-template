"""Dataset wrapper utilities for applying transforms to split datasets."""

from typing import Optional, Callable, Any, Tuple
from torch.utils.data import Dataset


class TransformWrapper(Dataset):
    """Wrapper to apply transforms to subset datasets from random_split."""

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        """Initialize the transform wrapper.

        :param dataset: The underlying dataset (potentially a Subset from random_split)
        :param transform: Transform to apply to the data
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get item from dataset and apply transform if specified."""
        image, target = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def get_heads_config(self):
        """Forward heads_config from the underlying dataset."""
        # Navigate through Subset and other wrapper layers to get to the original dataset
        dataset = self.dataset
        traversal_path = [type(dataset).__name__]

        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
            traversal_path.append(type(dataset).__name__)

        # Handle ConcatDataset specially - it has 'datasets' (plural) attribute
        if hasattr(dataset, 'datasets') and hasattr(dataset.datasets, '__iter__'):
            # For ConcatDataset, try to get heads_config from the first dataset that has it
            for i, sub_dataset in enumerate(dataset.datasets):
                if hasattr(sub_dataset, 'get_heads_config') and callable(getattr(sub_dataset, 'get_heads_config')):
                    try:
                        result = sub_dataset.get_heads_config()
                        if result:  # Only return if not empty
                            return result
                    except Exception as e:
                        continue
                elif hasattr(sub_dataset, 'heads_config') and sub_dataset.heads_config:
                    result = sub_dataset.heads_config
                    return result

            return {}

        # First try calling the method (for VIMH and other multihead datasets)
        if hasattr(dataset, 'get_heads_config'):
            method = getattr(dataset, 'get_heads_config')
            if callable(method):
                try:
                    result = dataset.get_heads_config()
                    return result
                except Exception as e:
                    import traceback
                    traceback.print_exc()

        # Fallback to attribute access (for backwards compatibility)
        if hasattr(dataset, 'heads_config'):
            result = dataset.heads_config
            return result

        return {}

    def __getattr__(self, name: str):
        """Forward attribute access to the underlying dataset."""
        # Special handling for parameter_mappings - traverse through ConcatDataset if needed
        if name == 'parameter_mappings':
            return self._get_parameter_mappings()

        # First try the wrapped dataset
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)

        # If it's a Subset, try the original dataset
        if hasattr(self.dataset, 'dataset'):
            dataset = self.dataset.dataset
            if hasattr(dataset, name):
                return getattr(dataset, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _get_parameter_mappings(self):
        """Get parameter_mappings by traversing the dataset chain."""
        # Navigate through Subset and other wrapper layers
        dataset = self.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        # Handle ConcatDataset specially
        if hasattr(dataset, 'datasets') and hasattr(dataset.datasets, '__iter__'):
            # For ConcatDataset, try to get parameter_mappings from the first dataset that has it
            for sub_dataset in dataset.datasets:
                if hasattr(sub_dataset, 'parameter_mappings'):
                    param_mappings = getattr(sub_dataset, 'parameter_mappings')
                    if param_mappings is not None:
                        return param_mappings
            return None

        # Single dataset case
        if hasattr(dataset, 'parameter_mappings'):
            return getattr(dataset, 'parameter_mappings')

        return None
