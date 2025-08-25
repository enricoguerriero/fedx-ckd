from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.data.loaders import get_dataset
import src.data.loaders as data_loaders
from src.models.resnet_cifar import resnet18
from src.fedx.utils import set_seed, get_device, init_wandb, compute_accuracy, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralised baseline training")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "svhn", "fmnist"], help="Dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--wandb_project", type=str, default=None, help="Optional W&B project name")
    parser.add_argument("--outdir", type=str, default="runs", help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    return parser.parse_args()


class SimpleDataset(Dataset):
    """Wrap a base dataset applying a transform to return tensors."""

    def __init__(self, base_dataset: Dataset, transform: callable):
        self.dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return self.transform(img), label


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    data_dir = os.path.join(os.getcwd(), "data")
    train_base = get_dataset(args.dataset, data_dir, train=True, download=True)
    test_base = get_dataset(args.dataset, data_dir, train=False, download=True)
    transform_train = data_loaders._get_base_transform(args.dataset, train=True)
    transform_test = data_loaders._get_base_transform(args.dataset, train=False)
    train_dataset = SimpleDataset(train_base, transform_train)
    test_dataset = SimpleDataset(test_base, transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    run_name = f"{args.dataset}_central_s{args.seed}"
    wandb_run = init_wandb(args.wandb_project, run_name, vars(args))

    os.makedirs(args.outdir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        test_acc = compute_accuracy(model, test_loader, device)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc}, step=epoch)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, test_acc={test_acc:.2f}%")
    checkpoint_path = os.path.join(args.outdir, f"{args.dataset}_central_s{args.seed}.pt")
    save_checkpoint(model, checkpoint_path)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
