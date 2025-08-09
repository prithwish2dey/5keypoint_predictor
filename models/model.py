import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configurations and hyperparameters
IMG_SIZE = 224                 # Input image size for ViT
NUM_KEYPOINTS = 5             # Number of facial keypoints to predict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTKeypointModel(nn.Module):
    """
    Vision Transformer backbone for keypoint regression.
    Outputs 2*NUM_KEYPOINTS values (x,y for each keypoint) normalized between 0 and 1.
    """
    def __init__(self, num_keypoints=NUM_KEYPOINTS, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, 2 * num_keypoints)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.sigmoid(x)  # Normalize outputs between 0 and 1
        return x
