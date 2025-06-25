
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, Subset
from torchvision import transforms
import torchvision.models as models
import requests
import onnxruntime as ort
import json
import io
import sys
import base64
from typing import Tuple
import pickle
from tqdm import tqdm
import time
import kornia.augmentation as K


# Assume the model to be similiar to ResNet18
class SmallResNet18(nn.Module):
    def __init__(self, output_dim=1024, dropout_p=0.3, pretrained=False):

        super().__init__()

        # Load a pretrained version for improved performance
        self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Changed to work on smaller image sizes as well, might reduce dimesions too quicly otherwise
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        # self.backbone.maxpool = nn.Identity()

        # Take everything except last layer
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add own head 
        self.head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        self.out = nn.Linear(512, output_dim)
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = x + self.head(x)  
        return self.out(x)

