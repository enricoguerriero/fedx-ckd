from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(x1: torch.Tensor, x2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute the normalised temperature‑scaled cross entropy (NT‑Xent) loss.

    Args:
        x1: Tensor of shape ``(B, D)`` containing projected embeddings from
            view 1.
        x2: Tensor of shape ``(B, D)`` containing projected embeddings from
            view 2.
        temperature: Temperature parameter controlling the sharpness of the
            similarity distribution.

    Returns:
        A scalar tensor representing the NT‑Xent loss over the batch.
    """
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def js_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    xa: torch.Tensor,
    t: float = 0.1,
    t2: float = 0.01,
) -> torch.Tensor:
    """Compute the Jensen–Shannon loss between pairs of embeddings.

    For two embedding sets ``x1`` and ``x2`` and a set of anchor features
    ``xa``, the JS loss computes the distance between their similarity
    distributions relative to the anchors.  A lower temperature ``t2``
    defines the target distribution and a higher temperature ``t`` defines
    the input distribution【140825399353834†L24-L33】.

    Args:
        x1: Tensor of shape ``(B, D)`` representing predicted embeddings.
        x2: Tensor of shape ``(B, D)`` representing a second set of
            predicted embeddings.
        xa: Tensor of shape ``(B, D)`` containing anchor embeddings.
        t: Temperature used for the input distributions (teacher/student).
        t2: Temperature used for the target distribution.

    Returns:
        A scalar tensor with the JS divergence between the similarity
        distributions of ``x1`` and ``x2``.
    """
    pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
    pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
    target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2.0
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0
