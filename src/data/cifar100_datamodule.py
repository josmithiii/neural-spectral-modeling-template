from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms


class CIFAR100WithCoarseLabels(CIFAR100):
    """CIFAR100 dataset wrapper that returns coarse labels instead of fine labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        # Convert fine label to coarse label using CIFAR-100 mapping
        coarse_target = self._get_coarse_label(target)
        return img, coarse_target

    def _get_coarse_label(self, fine_label: int) -> int:
        """Convert fine label (0-99) to coarse label (0-19) using CIFAR-100 groupings."""
        # CIFAR-100 fine-to-coarse mapping (each group of 5 fine labels maps to 1 coarse label)
        return fine_label // 5


class CIFAR100DataModule(LightningDataModule):
    """`LightningDataModule` for the CIFAR-100 dataset.

    The CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes containing 600 images
    each. There are 500 training images and 100 testing images per class. The 100 classes in the
    CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to
    which it belongs) and a "coarse" label (the superclass to which it belongs).

    Here are the 20 superclasses in CIFAR-100:
    - aquatic mammals: beaver, dolphin, otter, seal, whale
    - fish: aquarium fish, flatfish, ray, shark, trout
    - flowers: orchids, poppies, roses, sunflowers, tulips
    - food containers: bottles, bowls, cans, cups, plates
    - fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
    - household electrical devices: clock, computer keyboard, lamp, telephone, television
    - household furniture: bed, chair, couch, table, wardrobe
    - insects: bee, beetle, butterfly, caterpillar, cockroach
    - large carnivores: bear, leopard, lion, tiger, wolf
    - large man-made outdoor things: bridge, castle, house, road, skyscraper
    - large natural outdoor scenes: cloud, forest, mountain, plain, sea
    - large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
    - medium-sized mammals: fox, porcupine, possum, raccoon, skunk
    - non-insect invertebrates: crab, lobster, snail, spider, worm
    - people: baby, boy, girl, man, woman
    - reptiles: crocodile, dinosaur, lizard, snake, turtle
    - small mammals: hamster, mouse, rabbit, shrew, squirrel
    - trees: maple, oak, palm, pine, willow
    - vehicles 1: bicycle, bus, motorcycle, pickup truck, train
    - vehicles 2: lawn-mower, rocket, streetcar, tank, tractor

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
        train_val_test_split: Tuple[int, int, int] = (45_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        use_coarse_labels: bool = False,
    ) -> None:
        """Initialize a `CIFAR100DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(45_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param persistent_workers: Whether to use persistent workers. Defaults to `False`.
        :param use_coarse_labels: Whether to use coarse labels (20 classes) instead of fine labels (100 classes). Defaults to `False`.
        """
        super().__init__()

        # persistent_workers requires num_workers > 0
        self.persistent_workers = persistent_workers and num_workers > 0

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # CIFAR-100 uses the same normalization as CIFAR-10
        # These are the standard normalization values for CIFAR datasets
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )

        # Enhanced data augmentation for CIFAR-100 (more challenging dataset)
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),  # Additional augmentation for CIFAR-100
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of CIFAR-100 classes (100 for fine labels, 20 for coarse labels).
        """
        return 20 if self.hparams.use_coarse_labels else 100

    @property
    def fine_class_names(self) -> list:
        """Get the fine-grained class names (100 classes).

        :return: List of CIFAR-100 fine class names.
        """
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

    @property
    def coarse_class_names(self) -> list:
        """Get the coarse-grained class names (20 superclasses).

        :return: List of CIFAR-100 coarse class names.
        """
        return [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
            'medium-sized_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
            'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]

    @property
    def class_names(self) -> list:
        """Get the class names based on the label type being used.

        :return: List of class names (coarse or fine).
        """
        return self.coarse_class_names if self.hparams.use_coarse_labels else self.fine_class_names

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        CIFAR100(self.hparams.data_dir, train=True, download=True)
        CIFAR100(self.hparams.data_dir, train=False, download=True)

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

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Choose dataset class based on label type
            dataset_class = CIFAR100WithCoarseLabels if self.hparams.use_coarse_labels else CIFAR100

            # Use augmented transforms for training data
            trainset = dataset_class(
                self.hparams.data_dir,
                train=True,
                transform=self.train_transforms
            )
            testset = dataset_class(
                self.hparams.data_dir,
                train=False,
                transform=self.transforms
            )

            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

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
        )

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
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = CIFAR100DataModule()
