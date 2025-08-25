from __future__ import annotations

from typing import List, Tuple, Dict

import copy
import torch
from torch import nn

from .client import Client


class Server:
    """Central server for federated cross‑knowledge distillation."""

    def __init__(
        self,
        global_model: nn.Module,
        clients: List[Client],
        samples_per_client: List[int],
        device: torch.device,
    ) -> None:
        self.global_model = global_model.to(device)
        self.clients = clients
        self.samples_per_client = samples_per_client
        self.device = device

    def aggregate(self, client_updates: List[Tuple[dict, int]]) -> None:
        """Aggregate client models via weighted averaging.

        Args:
            client_updates: A list of tuples ``(state_dict, n_samples)``
                returned by each client after local training.
        """
        total_samples = float(sum(n for _, n in client_updates))
        new_state: Dict[str, torch.Tensor] = {}
        for key in self.global_model.state_dict().keys():
            new_state[key] = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)
        for state_dict, n_samples in client_updates:
            weight = n_samples / total_samples
            for key in new_state.keys():
                new_state[key] += state_dict[key].to(self.device) * weight
        self.global_model.load_state_dict(new_state)

    def train_round(
        self,
        epochs_per_round: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        temperature: float,
        t_teacher: float,
        t_student: float,
    ) -> float:
        """Perform a single federated round.

        Each client receives the current global model, trains locally and
        returns its updated state dict.  The server then aggregates
        these models via weighted averaging.  The mean local loss across
        clients is returned for logging.

        Args:
            epochs_per_round: Number of local epochs for each client.
            lr: Learning rate for client updates.
            momentum: Momentum term for client updates.
            weight_decay: Weight decay for client updates.
            temperature: Temperature for contrastive loss.
            t_teacher: Temperature ``t`` for the teacher in JS loss.
            t_student: Temperature ``t2`` for the student in JS loss.

        Returns:
            The average training loss across all clients.
        """
        client_updates: List[Tuple[dict, int]] = []
        losses: List[float] = []
        for client in self.clients:
            state, n_samples, mean_loss = client.update(
                self.global_model,
                epochs=epochs_per_round,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                temperature=temperature,
                t_teacher=t_teacher,
                t_student=t_student,
            )
            client_updates.append((state, n_samples))
            losses.append(mean_loss)
        self.aggregate(client_updates)
        return float(sum(losses) / max(len(losses), 1))
