from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN, FashionMNIST

DATA_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
    "fmnist": {
        "mean": (0.2860, 0.2860, 0.2860),
        "std": (0.3530, 0.3530, 0.3530),
    },
}


class ContrastiveDataset(Dataset):
    """Wraps a standard vision dataset to return two augmentations per sample.

    Given an underlying dataset returning `(image, label)`, this wrapper
    applies the same random augmentation pipeline twice to yield a pair
    of correlated views (`x1`, `x2`).  The label and the original index
    are returned for convenience, although FedX only uses the views.
    """

    def __init__(self, dataset: Dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2, label, index


def _get_base_transform(dataset: str, train: bool) -> transforms.Compose:
    """Return a torchvision transform for the given dataset.

    For training datasets a random crop with padding and random horizontal
    flip are applied.  For Fashion‑MNIST, images are converted to three
    channels so that a single backbone can be used for all datasets.
    """
    stats = DATA_STATS[dataset]
    mean, std = stats["mean"], stats["std"]

    if dataset in {"cifar10", "svhn"}:
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    elif dataset == "fmnist":
        if train:
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return transform


def get_dataset(dataset: str, root: str, train: bool, download: bool = True) -> Dataset:
    """Load a raw torchvision dataset without contrastive augmentation.

    Args:
        dataset: One of ``'cifar10'``, ``'svhn'`` or ``'fmnist'``.
        root: Directory in which to download or look for the data.
        train: Whether to return the training split (or validation/test split).
        download: Whether to automatically download the dataset if missing.

    Returns:
        A :class:`torch.utils.data.Dataset` instance.
    """
    dataset_lower = dataset.lower()
    if dataset_lower == "cifar10":
        return CIFAR10(root=root, train=train, transform=None, download=download)
    elif dataset_lower == "svhn":
        split = "train" if train else "test"
        return SVHN(root=root, split=split, transform=None, download=download)
    elif dataset_lower == "fmnist":
        return FashionMNIST(root=root, train=train, transform=None, download=download)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def get_data_loaders(
    dataset: str,
    data_dir: str,
    n_clients: int,
    batch_size: int,
    beta: float,
    seed: int,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[List[DataLoader], DataLoader, List[int]]:
    """Prepare federated data loaders.

    This helper loads the full training and test datasets, partitions the
    training indices among clients using a Dirichlet distribution and
    returns a list of per‑client :class:`DataLoader` objects along with
    the global test loader.  Each training loader yields tuples
    ``(x1, x2, label, index)``.  The returned list ``samples_per_client``
    contains the number of training samples assigned to each client.

    Args:
        dataset: Dataset name (``cifar10``, ``svhn`` or ``fmnist``).
        data_dir: Directory to cache/download the datasets.
        n_clients: Number of federated clients.
        batch_size: Mini‑batch size for training.
        beta: Dirichlet distribution parameter controlling non‑IIDness.
        seed: Random seed for deterministic partitioning.
        num_workers: Number of worker processes per data loader.
        download: Whether to download the dataset if absent.

    Returns:
        (client_loaders, test_loader, samples_per_client)
    """
    full_train = get_dataset(dataset, data_dir, train=True, download=download)
    full_test = get_dataset(dataset, data_dir, train=False, download=download)

    if dataset.lower() == "svhn":
        labels = np.array(full_train.labels)
    else:
        labels = np.array(full_train.targets)

    from .partition import dirichlet_partition

    client_indices = dirichlet_partition(labels, n_clients, beta, seed)

    transform_train = _get_base_transform(dataset.lower(), train=True)
    client_loaders: List[DataLoader] = []
    samples_per_client: List[int] = []
    for idxs in client_indices:
        subset = Subset(full_train, idxs)
        cont_dataset = ContrastiveDataset(subset, transform_train)
        loader = DataLoader(
            cont_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        client_loaders.append(loader)
        samples_per_client.append(len(subset))

    transform_test = _get_base_transform(dataset.lower(), train=False)
    class _TestWrapper(Dataset):
        def __init__(self, base_dataset: Dataset, transform: transforms.Compose):
            self.dataset = base_dataset
            self.transform = transform

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx: int):
            img, label = self.dataset[idx]
            x = self.transform(img)
            return x, label

    test_dataset = _TestWrapper(full_test, transform_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return client_loaders, test_loader, samples_per_client
