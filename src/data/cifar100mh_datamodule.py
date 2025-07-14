from typing import Any, Dict, Optional, Tuple, List
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .cifar100mh_dataset import CIFAR100MHDataset, create_cifar100mh_datasets


class CIFAR100MHDataModule(LightningDataModule):
    """`LightningDataModule` for the CIFAR-100-MH multihead dataset.

    The CIFAR-100-MH dataset extends the CIFAR format to support multiple classification
    heads with embedded metadata. Each sample contains an image and multiple real labels
    (not synthetic) for different classification tasks.

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
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        train_transform: Optional[transforms.Compose] = None,
        val_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize a `CIFAR100MHDataModule`.

        :param data_dir: The data directory containing CIFAR-100-MH files. Defaults to `"data/"`.
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

        # Default transforms for CIFAR-32x32 images
        # Use the same normalization as CIFAR datasets
        self.default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Enhanced training transforms with data augmentation
        self.default_train_transforms = transforms.Compose([
            transforms.ToPILImage(),  # Convert tensor back to PIL for augmentation
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Set transforms (use provided or defaults)
        self.train_transform = train_transform or self.default_train_transforms
        self.val_transform = val_transform or self.default_transforms
        self.test_transform = test_transform or self.default_transforms

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        # Will be set after loading the dataset
        self.heads_config: Dict[str, int] = {}
        self.image_shape: Tuple[int, int, int] = (3, 32, 32)  # Default

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
        # For CIFAR-100-MH, we assume the data is already prepared/downloaded
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
                # Load training and test datasets
                self.data_train = CIFAR100MHDataset(
                    self.hparams.data_dir,
                    train=True,
                    transform=self.train_transform
                )

                self.data_test = CIFAR100MHDataset(
                    self.hparams.data_dir,
                    train=False,
                    transform=self.test_transform
                )

                # For validation, we'll use the test dataset with val transforms
                # In a real scenario, you might want to split the training data
                self.data_val = CIFAR100MHDataset(
                    self.hparams.data_dir,
                    train=False,
                    transform=self.val_transform
                )

                # Extract dataset configuration
                self.heads_config = self.data_train.get_heads_config()
                self.image_shape = self.data_train.get_image_shape()

            except Exception as e:
                raise RuntimeError(f"Failed to load CIFAR-100-MH dataset from {self.hparams.data_dir}: {e}")

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
        print("Testing CIFAR-100-MH data module...")

        # Initialize data module
        dm = CIFAR100MHDataModule(
            data_dir="data/cifar-100-mh-example",
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

        print("\nCIFAR-100-MH data module implementation successful!")

    except Exception as e:
        print(f"✗ Error testing data module: {e}")
        import traceback
        traceback.print_exc()
