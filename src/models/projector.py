from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """Two‑layer projection MLP used in FedX.

    Args:
        input_dim: Dimensionality of the backbone feature vector.
        projection_dim: Dimensionality of the output embedding.
    """

    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    """Two‑layer prediction MLP used in FedX.

    Args:
        projection_dim: Dimensionality of the input and output embedding.
        hidden_dim: Dimensionality of the hidden layer.  By default it
            equals ``projection_dim`` as in BYOL.
    """

    def __init__(self, projection_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = projection_dim
        self.net = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FedXModel(nn.Module):
    """FedX model combining a backbone, projection head and prediction head.

    The backbone produces high‑dimensional features (512 for ResNet‑18),
    which are then passed through the projection head and prediction head
    respectively.  During contrastive training, the projection output
    participates in the NT‑Xent loss and the prediction output is used
    for relational JS loss.

    Args:
        backbone: A feature extractor that implements ``forward_features``.
        projection_dim: Output dimensionality of the projection head.
        hidden_dim: Hidden dimensionality of the prediction head.  If
            ``None`` the prediction head uses ``projection_dim``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 256,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat = self.backbone.forward_features(dummy)
            feature_dim = feat.shape[1]
        self.projection = ProjectionHead(feature_dim, projection_dim)
        self.prediction = PredictionHead(projection_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return backbone features, projection and prediction.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            A tuple ``(features, projected, predicted)``.
        """
        feats = self.backbone.forward_features(x)
        proj = self.projection(feats)
        pred = self.prediction(proj)
        return feats, proj, pred
