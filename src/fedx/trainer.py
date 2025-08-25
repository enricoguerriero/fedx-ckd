from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from src.data.loaders import get_data_loaders
from src.models.resnet_cifar import resnet18
from src.models.projector import FedXModel
from .client import Client
from .server import Server
from .utils import (
    set_seed,
    get_device,
    init_wandb,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FedX with cross‑knowledge distillation")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "svhn", "fmnist"], help="Dataset to use")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of federated clients")
    parser.add_argument("--epochs_per_round", type=int, default=1, help="Local epochs per communication round")
    parser.add_argument("--n_rounds", type=int, default=10, help="Number of federated communication rounds")
    parser.add_argument("--beta", type=float, default=0.5, help="Dirichlet concentration parameter for non‑IID split")
    parser.add_argument("--batch_size", type=int, default=128, help="Local batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", help="Device to train on (auto/cuda/cpu)")
    parser.add_argument("--wandb_project", type=str, default=None, help="Optional W&B project name for logging")
    parser.add_argument("--outdir", type=str, default="runs", help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for NT‑Xent loss")
    parser.add_argument("--t_teacher", type=float, default=0.1, help="Temperature t for teacher in JS loss")
    parser.add_argument("--t_student", type=float, default=0.1, help="Temperature t2 for student in JS loss")
    parser.add_argument("--projection_dim", type=int, default=256, help="Output dimensionality of projection head")
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    data_dir = os.path.join(os.getcwd(), "data")
    client_loaders, test_loader, samples_per_client = get_data_loaders(
        dataset=args.dataset,
        data_dir=data_dir,
        n_clients=args.n_clients,
        batch_size=args.batch_size,
        beta=args.beta,
        seed=args.seed,
        download=True,
    )

    backbone = resnet18(num_classes=10)
    global_model = FedXModel(backbone, projection_dim=args.projection_dim)

    clients = [Client(i, loader, device) for i, loader in enumerate(client_loaders)]
    server = Server(global_model, clients, samples_per_client, device)

    run_name = f"{args.dataset}_fedx_s{args.seed}"
    wandb_run = init_wandb(args.wandb_project, run_name, vars(args))

    os.makedirs(args.outdir, exist_ok=True)

    for round_idx in range(1, args.n_rounds + 1):
        mean_loss = server.train_round(
            epochs_per_round=args.epochs_per_round,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            t_teacher=args.t_teacher,
            t_student=args.t_student,
        )
        if wandb_run is not None:
            wandb_run.log({"round": round_idx, "train_loss": mean_loss}, step=round_idx)
        print(f"Round {round_idx}/{args.n_rounds}: mean training loss = {mean_loss:.4f}")
        checkpoint_path = os.path.join(args.outdir, f"global_round_{round_idx:03d}.pt")
        save_checkpoint(server.global_model, checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
