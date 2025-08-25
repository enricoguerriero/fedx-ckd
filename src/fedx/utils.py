from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed random number generators for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Parse a device specification string into a :class:`torch.device`.

    Args:
        device_str: One of ``'auto'``, ``'cuda'`` / ``'cuda:0'`` / ``'cpu'``.

    Returns:
        The corresponding torch device.
    """
    device_str = device_str.lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Persist a model's state_dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, map_location: Optional[str] = None) -> None:
    """Load a model's state_dict from disk in place."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)


@torch.no_grad()
def compute_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Compute Top‑1 classification accuracy over a data loader.

    The given model must produce class logits when called.  For linear
    probing and centralized baseline training this helper is used to
    evaluate test performance.
    """
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


def init_wandb(project: Optional[str], run_name: str, config: dict) -> Optional[object]:
    """Initialise a Weights & Biases run if a project is specified.

    The returned object is a W&B run or ``None`` if W&B is unavailable or
    no project name was provided.  To avoid network calls when
    executed offline, the run is initialised in offline mode.
    """
    if not project:
        return None
    try:
        import wandb

        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            mode="offline",
        )
        return run
    except Exception:
        return None
