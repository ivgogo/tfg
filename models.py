# ================ Required imports & libraries ================

# Torch
from torch.nn import functional as F
import torch.nn as nn
import torch

# Pytorch Image Models library
import timm

# Basic libraries
import pandas as pd
import numpy as np

# ================ Models & functions ================

# Custom implementation of EfficientNet architecture
class SimpleEffnet(nn.Module):
    def __init__(self, number_classes=1, n_metadata_features=0, pretrained=True):
        super(SimpleEffnet, self).__init__()

        self.number_classes = number_classes
        self.n_meta_features = n_metadata_features
        self.pretrained = pretrained
        self.use_meta = n_metadata_features > 0

        self.model = timm.create_model(model_name="efficientnet_b3", pretrained=pretrained, num_classes=number_classes)

        # Number of features last layer
        in_ch = self.model.get_classifier().in_features
        
        if self.use_meta:
            self.meta_layers = nn.Sequential(
                nn.Linear(n_metadata_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            in_ch += n_metadata_features
        
        self.last_layer = nn.Linear(in_ch, number_classes)
        self.model.classifier = nn.Identity()

    def forward(self, x, metadata=None):
        x = self.model(x)
        if self.use_meta:
            x = torch.cat((x, metadata), dim=1)  # Combina las características extraídas de la imagen con los metadatos
        x = self.last_layer(x)
        return x

# Focal Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss