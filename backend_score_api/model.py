# Chargement CNN / définition du modèle

import torch
from torch import nn
from torchvision import models

class CNNScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        emb = self.backbone(x)
        score = self.fc(emb)
        return score.squeeze(1)

def load_model(path):
    model = CNNScore()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
