from __future__ import annotations

import copy
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .loss import nt_xent, js_loss


class Client:
    """Represents a single federated client.

    Args:
        client_id: Integer identifier.
        train_loader: DataLoader providing tuples ``(x1, x2, label, index)``.
        device: ``torch.device`` on which to perform computations.
    """

    def __init__(self, client_id: int, train_loader: DataLoader, device: torch.device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device

    def update(
        self,
        global_model: nn.Module,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        temperature: float,
        t_teacher: float,
        t_student: float,
    ) -> Tuple[dict, int, float]:
        """Train the local model using cross‑knowledge distillation.

        A fresh copy of the global model is created and optimised on the
        client's local data.  After training, the updated state dict and
        number of local samples are returned along with the mean loss.

        Args:
            global_model: The current global model acting as the teacher.
            epochs: Number of local epochs to train.
            lr: Learning rate for SGD.
            momentum: Momentum for SGD.
            weight_decay: Weight decay (L2 penalty).
            temperature: Temperature for the contrastive loss.
            t_teacher: Temperature ``t`` in the JS loss for the teacher.
            t_student: Temperature ``t2`` in the JS loss for the student.

        Returns:
            A tuple ``(state_dict, n_samples, mean_loss)``.
        """
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        global_model = global_model.to(self.device)
        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        random_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        random_iter = iter(random_loader)

        local_model.train()
        global_model.eval()
        losses: list[float] = []
        for epoch in range(epochs):
            for (x1, x2, _label, _idx) in self.train_loader:
                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                try:
                    random_x, _, _, _ = next(random_iter)
                except StopIteration:
                    random_iter = iter(random_loader)
                    random_x, _, _, _ = next(random_iter)
                random_x = random_x.to(self.device, non_blocking=True)

                all_x = torch.cat([x1, x2, random_x], dim=0)
                _feats_s, proj_s, pred_s = local_model(all_x)
                with torch.no_grad():
                    _feats_t, proj_t, pred_t = global_model(all_x)

                b = x1.size(0)
                proj_s_orig, proj_s_pos, proj_s_rand = proj_s.split([b, b, random_x.size(0)], dim=0)
                pred_s_orig, pred_s_pos, pred_s_rand = pred_s.split([b, b, random_x.size(0)], dim=0)
                proj_t_orig, proj_t_pos, proj_t_rand = proj_t.split([b, b, random_x.size(0)], dim=0)
                pred_t_orig, pred_t_pos, pred_t_rand = pred_t.split([b, b, random_x.size(0)], dim=0)

                loss_nt_local = nt_xent(proj_s_orig, proj_s_pos, temperature)
                loss_nt_global = nt_xent(pred_s_orig, proj_t_pos, temperature)
                loss_nt = loss_nt_local + loss_nt_global

                loss_js_local = js_loss(proj_s_orig, proj_s_pos, proj_s_rand, t_student, t_student)
                loss_js_global = js_loss(pred_s_orig, pred_s_pos, proj_t_rand, t_teacher, t_teacher)
                loss_js = loss_js_local + loss_js_global

                loss = loss_nt + loss_js
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        mean_loss = float(sum(losses) / max(len(losses), 1))
        return local_model.state_dict(), len(self.train_loader.dataset), mean_loss
