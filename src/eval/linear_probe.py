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
from src.models.projector import FedXModel
from src.fedx.utils import set_seed, get_device, init_wandb, compute_accuracy


class FeatureDataset(Dataset):
    """Wrap a base dataset applying a transform and returning tensors.

    Unlike the contrastive dataset used during FL, this dataset returns
    a single transformed image and its label.
    """

    def __init__(self, base_dataset: Dataset, transform: callable):
        self.dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        x = self.transform(img)
        return x, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe for FedX representations")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "svhn", "fmnist"], help="Dataset used during training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved global model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the linear classifier")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for probing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device for computation")
    parser.add_argument("--wandb_project", type=str, default=None, help="Optional W&B project name")
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    data_dir = os.path.join(os.getcwd(), "data")
    train_base = get_dataset(args.dataset, data_dir, train=True, download=True)
    test_base = get_dataset(args.dataset, data_dir, train=False, download=True)
    transform = data_loaders._get_base_transform(args.dataset, train=False)
    train_dataset = FeatureDataset(train_base, transform)
    test_dataset = FeatureDataset(test_base, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    backbone = resnet18(num_classes=10)
    fedx_model = FedXModel(backbone, projection_dim=256)
    state = torch.load(args.checkpoint, map_location=device)
    fedx_model.load_state_dict(state)
    fedx_model.to(device)
    for param in fedx_model.parameters():
        param.requires_grad = False
    fedx_model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        feat, _, _ = fedx_model(dummy)
        feature_dim = feat.shape[1]

    classifier = nn.Linear(feature_dim, 10).to(device)
    optimiser = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    class ProbeModel(nn.Module):
        def __init__(self, enc: FedXModel, clf: nn.Linear):
            super().__init__()
            self.enc = enc
            self.clf = clf

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats, _, _ = self.enc(x)
            return self.clf(feats)

    probe_model = ProbeModel(fedx_model, classifier)

    run_name = f"{args.dataset}_probe_s{args.seed}"
    wandb_run = init_wandb(args.wandb_project, run_name, vars(args))
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.no_grad():
                feats, _, _ = fedx_model(inputs)
            logits = classifier(feats)
            loss = criterion(logits, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        test_acc = compute_accuracy(probe_model, test_loader, device)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc}, step=epoch)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, test_acc={test_acc:.2f}%")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
