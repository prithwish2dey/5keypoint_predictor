import os
import zipfile
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.model import ViTKeypointModel

# ==== Flask setup ====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==== Global variables ====
DEVICE = torch.device("cpu")  # Render free tier runs CPU only
model = None  # Lazy-load
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_model():
    """Load model from zipped file if not already loaded."""
    global model
    if model is None:
        zip_path = "models/vit_keypoint_model_new.zip"
        pth_path = "models/vit_keypoint_model_new.pth"

        # Unzip the model file if it doesn't already exist
        if os.path.exists(zip_path) and not os.path.exists(pth_path):
            with zipfile.ZipFile(zip_path, 'r') as_
