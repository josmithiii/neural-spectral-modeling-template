from typing import Any, Dict, Optional, Tuple, List
import torch
import json
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .vimh_dataset import VIMHDataset, create_vimh_datasets


class VIMHDataModule(LightningDataModule):
    """`LightningDataModule` for the VIMH (Variable Image MultiHead) dataset.

    The VIMH dataset is a generalized format for multihead neural networks that supports:
    - Variable image dimensions (height×width×channels)
    - Self-describing variable-length label format
    - 0-255 varying parameters per sample
    - Efficient 8-bit quantization

    Format specification:
    - Binary format with embedded metadata: [height] [width] [channels] [N] [param1_id] [param1_val] ...
    - Multiple real classification heads
    - Self-describing metadata structure

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data-vimh/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        train_transform: Optional[transforms.Compose] = None,
        val_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize a `VIMHDataModule`.

        :param data_dir: The data directory containing VIMH files. Defaults to `"data-vimh/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param persistent_workers: Whether to use persistent workers. Defaults to `True`.
        :param train_transform: Optional transforms for training data.
        :param val_transform: Optional transforms for validation data.
        :param test_transform: Optional transforms for test data.
        """
        super().__init__()

        # persistent_workers requires num_workers > 0
        self.persistent_workers = persistent_workers and num_workers > 0

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Default transforms for variable-size images
        # These will be adjusted based on the actual image dimensions
        # Note: Data is already converted to tensor by the dataset
        self.default_transforms = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Generic normalization
        ])

        # Enhanced training transforms with data augmentation
        # Will be adjusted based on actual image size
        self.default_train_transforms = transforms.Compose([
            transforms.ToPILImage(),  # Convert tensor back to PIL for augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Set transforms (use provided or defaults)
        self.train_transform = train_transform or self.default_train_transforms
        self.val_transform = val_transform or self.default_transforms
        self.test_transform = test_transform or self.default_transforms

        # Track if we're using defaults (for later adjustment)
        self.using_default_train_transform = train_transform is None
        self.using_default_val_transform = val_transform is None
        self.using_default_test_transform = test_transform is None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        # Will be set after loading the dataset
        self.heads_config: Dict[str, int] = {}
        self.parameter_ranges: Dict[str, float] = {}  # Parameter ranges for perceptual loss
        self.image_shape: Tuple[int, int, int] = (3, 32, 32)  # Default, will be updated

    def _adjust_transforms_for_image_size(self, height: int, width: int) -> None:
        """Adjust transforms based on actual image dimensions."""
        # Update the default train transforms with proper padding/cropping
        if height == 32 and width == 32:
            # Use CIFAR-style transforms for 32x32 images
            self.default_train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.default_transforms = transforms.Compose([
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif height == 28 and width == 28:
            # Use MNIST-style transforms for 28x28 images
            self.default_train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            self.default_transforms = transforms.Compose([
                transforms.Normalize((0.5,), (0.5,))
            ])

        # Update transforms if they were using defaults
        if self.using_default_train_transform:
            self.train_transform = self.default_train_transforms
        if self.using_default_val_transform:
            self.val_transform = self.default_transforms
        if self.using_default_test_transform:
            self.test_transform = self.default_transforms

    def _parse_image_dims_from_path(self, data_dir: str) -> Optional[Tuple[int, int, int]]:
        """Extract image dimensions from dataset directory name.

        Looks for pattern 'vimh-<height>x<width>x<channels>' at start of directory name.

        :param data_dir: Path to dataset directory
        :return: (height, width, channels) tuple or None if pattern not found
        """
        try:
            dir_name = Path(data_dir).name
            if dir_name.startswith('vimh-'):
                # Extract "32x32x3" from "vimh-32x32x3_8000Hz_1p0s_256dss_resonarium_2p"
                parts = dir_name.split('_')[0]  # "vimh-32x32x3"
                dims = parts.split('-')[1]      # "32x32x3"
                h, w, c = map(int, dims.split('x'))
                return h, w, c
        except (IndexError, ValueError):
            pass
        return None

    def _load_image_dims_from_json(self, data_dir: str) -> Optional[Tuple[int, int, int]]:
        """Load image dimensions from dataset metadata JSON.

        :param data_dir: Path to dataset directory
        :return: (height, width, channels) tuple or None if file not found or invalid
        """
        try:
            metadata_file = Path(data_dir) / 'vimh_dataset_info.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    h = metadata['height']
                    w = metadata['width']
                    c = metadata['channels']
                    return h, w, c
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
        return None

    def _validate_binary_metadata(self, data_dir: str) -> Optional[Tuple[int, int, int]]:
        """Validate dimensions by sampling binary data metadata.

        Checks the first 6 bytes of a few samples to ensure consistency.

        :param data_dir: Path to dataset directory
        :return: (height, width, channels) tuple or None if validation fails
        """
        try:
            import pickle
            import struct

            # Check pickle format first (easier to parse)
            train_file = Path(data_dir) / 'train_batch'
            if train_file.exists():
                with open(train_file, 'rb') as f:
                    data = pickle.load(f)
                    if 'height' in data and 'width' in data and 'channels' in data:
                        return data['height'], data['width'], data['channels']

            # Check binary format if pickle not available
            binary_file = Path(data_dir) / 'train'
            if binary_file.exists():
                with open(binary_file, 'rb') as f:
                    # Read first sample metadata (first 6 bytes)
                    metadata_bytes = f.read(6)
                    if len(metadata_bytes) == 6:
                        # Unpack as 3 uint16 values: height, width, channels
                        h, w, c = struct.unpack('<HHH', metadata_bytes)
                        return h, w, c

        except (FileNotFoundError, pickle.UnpicklingError, struct.error):
            pass
        return None

    def _fallback_dimension_detection(self, data_dir: str) -> Tuple[int, int, int]:
        """Fallback method: load temporary dataset for dimension detection.

        :param data_dir: Path to dataset directory
        :return: (height, width, channels) tuple
        """
        temp_dataset = VIMHDataset(data_dir, train=True)
        c, h, w = temp_dataset.get_image_shape()  # PyTorch format (C, H, W)
        return h, w, c

    def _detect_and_validate_image_dimensions(self, data_dir: str) -> Tuple[int, int, int]:
        """Detect image dimensions with cross-validation across all sources.

        Validates consistency between directory name, JSON metadata, and binary data.

        :param data_dir: Path to dataset directory
        :return: (height, width, channels) tuple
        :raises ValueError: If dimensions are inconsistent across sources
        """
        # Method 1: Parse from directory name (if pattern exists)
        dir_dims = self._parse_image_dims_from_path(data_dir)

        # Method 2: Read from JSON metadata
        json_dims = self._load_image_dims_from_json(data_dir)

        # Method 3: Validate against binary data (sample a few records)
        binary_dims = self._validate_binary_metadata(data_dir)

        # Cross-validation logic
        available_sources = []
        if dir_dims:
            available_sources.append(f"directory={dir_dims}")
        if json_dims:
            available_sources.append(f"json={json_dims}")
        if binary_dims:
            available_sources.append(f"binary={binary_dims}")

        # Check for consistency among all available sources
        all_dims = [dim for dim in [dir_dims, json_dims, binary_dims] if dim is not None]

        if len(all_dims) == 0:
            # No metadata found, use fallback
            return self._fallback_dimension_detection(data_dir)

        if len(set(all_dims)) == 1:
            # All sources agree (or only one source available)
            return all_dims[0]
        else:
            # Inconsistent dimensions across sources
            sources_info = ", ".join(available_sources)
            raise ValueError(
                f"Dimension mismatch in dataset '{data_dir}': {sources_info}. "
                f"All metadata sources must agree on image dimensions."
            )

    def _load_dataset_metadata(self, data_dir: str) -> Dict[str, int]:
        """Load dataset heads configuration efficiently.

        :param data_dir: Path to dataset directory
        :return: Dictionary mapping head names to number of classes
        """
        try:
            # Try to load from JSON first (fastest)
            metadata_file = Path(data_dir) / 'vimh_dataset_info.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Calculate heads config from parameter mappings
                heads_config = {}
                if 'parameter_names' in metadata and 'parameter_mappings' in metadata:
                    param_names = metadata['parameter_names']
                    param_mappings = metadata['parameter_mappings']

                    for param_name in param_names:
                        if param_name in param_mappings:
                            # For continuous parameters, use 256 classes (0-255 quantization)
                            heads_config[param_name] = 256

                return heads_config

        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

        # Fallback: load temporary dataset
        temp_dataset = VIMHDataset(data_dir, train=True)
        return temp_dataset.get_heads_config()

    def _load_parameter_ranges(self, data_dir: str) -> Dict[str, float]:
        """Load parameter ranges for perceptual loss calculation.

        :param data_dir: Path to dataset directory
        :return: Dictionary mapping parameter names to their ranges (max - min)
        """
        parameter_ranges = {}

        try:
            # Try to load from JSON metadata
            metadata_file = Path(data_dir) / 'vimh_dataset_info.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                if 'parameter_names' in metadata and 'parameter_mappings' in metadata:
                    param_names = metadata['parameter_names']
                    param_mappings = metadata['parameter_mappings']

                    for param_name in param_names:
                        if param_name in param_mappings:
                            param_info = param_mappings[param_name]
                            param_range = param_info['max'] - param_info['min']
                            parameter_ranges[param_name] = param_range

        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

        return parameter_ranges

    def _multihead_collate_fn(self, batch: List[Tuple[torch.Tensor, Dict[str, int]]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Custom collate function for multihead labels.

        Converts a batch of (image, labels_dict) pairs into batched tensors.

        :param batch: List of (image_tensor, labels_dict) tuples
        :return: Tuple of (batched_images, batched_labels_dict)
        """
        images = []
        labels_dict = {}

        # Separate images and labels
        for image, labels in batch:
            images.append(image)

            # Initialize label tensors on first iteration
            if not labels_dict:
                for head_name in labels.keys():
                    labels_dict[head_name] = []

            # Collect labels for each head
            for head_name, label_value in labels.items():
                labels_dict[head_name].append(label_value)

        # Stack images
        batched_images = torch.stack(images)

        # Convert label lists to tensors
        batched_labels = {}
        for head_name, label_list in labels_dict.items():
            batched_labels[head_name] = torch.tensor(label_list, dtype=torch.long)

        return batched_images, batched_labels

    @property
    def num_classes(self) -> Dict[str, int]:
        """Get the number of classes for each head.

        :return: Dictionary mapping head names to number of classes.
        """
        return self.heads_config.copy()

    @property
    def param_ranges(self) -> Dict[str, float]:
        """Get the parameter ranges for perceptual loss calculation.

        :return: Dictionary mapping parameter names to their ranges (max - min).
        """
        return self.parameter_ranges.copy()

    @property
    def class_names(self) -> Dict[str, List[str]]:
        """Get class names for each head (if available from metadata).

        :return: Dictionary mapping head names to class name lists.
        """
        # This would need to be populated from dataset metadata
        # For now, return generic names
        class_names_dict = {}
        for head_name, num_classes in self.heads_config.items():
            class_names_dict[head_name] = [f"{head_name}_class_{i}" for i in range(num_classes)]
        return class_names_dict

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # For VIMH, we assume the data is already prepared/downloaded
        # This method can be extended to handle automatic dataset download/preparation
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            try:
                # Efficiently detect image dimensions with cross-validation
                height, width, channels = self._detect_and_validate_image_dimensions(self.hparams.data_dir)
                self.image_shape = (channels, height, width)  # PyTorch format (C, H, W)

                # Load heads configuration efficiently
                self.heads_config = self._load_dataset_metadata(self.hparams.data_dir)

                # Load parameter ranges for perceptual loss
                self.parameter_ranges = self._load_parameter_ranges(self.hparams.data_dir)

                # Adjust transforms based on detected image dimensions
                self._adjust_transforms_for_image_size(height, width)

                # Now load datasets with correctly adjusted transforms
                self.data_train = VIMHDataset(
                    self.hparams.data_dir,
                    train=True,
                    transform=self.train_transform
                )

                self.data_test = VIMHDataset(
                    self.hparams.data_dir,
                    train=False,
                    transform=self.test_transform
                )

                # For validation, we'll use the test dataset with val transforms
                # In a real scenario, you might want to split the training data
                self.data_val = VIMHDataset(
                    self.hparams.data_dir,
                    train=False,
                    transform=self.val_transform
                )

            except Exception as e:
                raise RuntimeError(f"Failed to load VIMH dataset from {self.hparams.data_dir}: {e}")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            collate_fn=self._multihead_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            collate_fn=self._multihead_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            collate_fn=self._multihead_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {
            'heads_config': self.heads_config,
            'image_shape': self.image_shape,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        self.heads_config = state_dict.get('heads_config', {})
        self.image_shape = state_dict.get('image_shape', (3, 32, 32))

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded dataset.

        :return: Dictionary with dataset information.
        """
        if self.data_train is None:
            return {'error': 'Dataset not loaded yet. Call setup() first.'}

        return {
            'heads_config': self.heads_config,
            'image_shape': self.image_shape,
            'num_train_samples': len(self.data_train) if self.data_train else 0,
            'num_val_samples': len(self.data_val) if self.data_val else 0,
            'num_test_samples': len(self.data_test) if self.data_test else 0,
            'batch_size': self.hparams.batch_size,
            'num_workers': self.hparams.num_workers,
            'data_dir': self.hparams.data_dir,
        }


if __name__ == "__main__":
    # Test the data module
    import sys
    from pathlib import Path

    # Add src to path for imports
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        print("Testing VIMH data module...")

        # Initialize data module
        dm = VIMHDataModule(
            data_dir="data-vimh/vimh-32x32_8000Hz_1p0s_256dss_resonarium_2p",
            batch_size=4,
            num_workers=0
        )

        # Setup the data module
        dm.setup()

        print(f"✓ Data module setup successful")
        print(f"✓ Dataset info: {dm.get_dataset_info()}")

        # Test data loaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        print(f"✓ Created dataloaders - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

        # Test a batch
        batch_images, batch_labels = next(iter(train_loader))
        print(f"✓ Batch images shape: {batch_images.shape}")
        print(f"✓ Batch labels: {[f'{k}: {v.shape}' for k, v in batch_labels.items()]}")

        print("\nVIMH data module implementation successful!")

    except Exception as e:
        print(f"✗ Error testing data module: {e}")
        import traceback
        traceback.print_exc()
