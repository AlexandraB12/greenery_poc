# reco_engine/model_utils.py
from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from typing import Tuple, Optional


class CNNScore(nn.Module):
    """
    Même architecture que ton backend_score_api/model.py
    - backbone resnet18 pré-entraîné
    - embedding 512D
    - fc 512 -> 1 (score charme)
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        emb = self.backbone(x)          # (B, 512)
        score = self.fc(emb)            # (B, 1)
        return score.squeeze(1)         # (B,)

    @torch.no_grad()
    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x)          # (B, 512)
        score = self.fc(emb).squeeze(1) # (B,)
        return score, emb


def load_torch_model(model_path: str, device: str = "cpu") -> CNNScore:
    model = CNNScore()
    state = torch.load(model_path, map_location="cpu")

    # .pth est un OrderedDict => direct
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_score_and_embedding(model: CNNScore, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retourne:
      - score charme (B,)
      - embedding (B,512)
    """
    return model.forward_with_embedding(x)
