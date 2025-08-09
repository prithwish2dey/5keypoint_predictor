# import os
# import torch
# import cv2
# import numpy as np
# from flask import Flask, render_template, request, send_from_directory
# from PIL import Image
# from torchvision import transforms
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # ==== Import your ViT model ====
# from models.model import ViTKeypointModel  # Replace with actual model file

# # ==== Flask setup ====
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # ==== Load model ====
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ViTKeypointModel(num_keypoints=5, pretrained=False)
# model.load_state_dict(torch.load("models/vit_keypoint_model_new.pth", map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # ==== Image transform ====
# transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
# ])

# def predict_keypoints(img_pil):
#     img_np = np.array(img_pil.convert("RGB"))
#     orig_h, orig_w = img_np.shape[:2]

#     augmented = transform(image=img_np)
#     img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         preds = model(img_tensor).squeeze(0).cpu().numpy()
#     keypoints = preds.reshape(-1, 2)

#     keypoints[:, 0] *= 224
#     keypoints[:, 1] *= 224

#     scale_x = orig_w / 224
#     scale_y = orig_h / 224
#     keypoints[:, 0] *= scale_x
#     keypoints[:, 1] *= scale_y

#     img_out = img_np.copy()
#     for (x, y) in keypoints:
#         cv2.circle(img_out, (int(x), int(y)), 4, (255, 0, 0), -1)

#     return Image.fromarray(img_out)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "image" not in request.files:
#             return render_template("index.html", error="No file uploaded")

#         file = request.files["image"]
#         if file.filename == "":
#             return render_template("index.html", error="No file selected")

#         if file:
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(filepath)

#             img_pil = Image.open(filepath)
#             output_img = predict_keypoints(img_pil)

#             output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_" + file.filename)
#             output_img.save(output_path)

#             return render_template("index.html",
#                                    input_image=file.filename,
#                                    output_image="output_" + file.filename)
#     return render_template("index.html")

# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == "__main__":
#     app.run(debug=True)










import os
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.model import ViTKeypointModel

# ==== Flask setup ====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==== Global variables ====
DEVICE = torch.device("cpu")  # Force CPU on Render free tier
model = None  # Will lazy-load
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_model():
    """Load the model only when needed."""
    global model
    if model is None:
        model = ViTKeypointModel(num_keypoints=5, pretrained=False)
        state = torch.load("models/vit_keypoint_model_new.pth", map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()

def predict_keypoints(img_pil):
    load_model()  # Ensure model is loaded
    img_np = np.array(img_pil.convert("RGB"))
    orig_h, orig_w = img_np.shape[:2]

    augmented = transform(image=img_np)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(img_tensor).squeeze(0).cpu().numpy()
    keypoints = preds.reshape(-1, 2)

    keypoints[:, 0] *= 224
    keypoints[:, 1] *= 224

    scale_x = orig_w / 224
    scale_y = orig_h / 224
    keypoints[:, 0] *= scale_x
    keypoints[:, 1] *= scale_y

    img_out = img_np.copy()
    for (x, y) in keypoints:
        cv2.circle(img_out, (int(x), int(y)), 4, (255, 0, 0), -1)

    return Image.fromarray(img_out)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="No file selected")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img_pil = Image.open(filepath)
        output_img = predict_keypoints(img_pil)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_" + file.filename)
        output_img.save(output_path)

        return render_template("index.html",
                               input_image=file.filename,
                               output_image="output_" + file.filename)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a port
    app.run(host="0.0.0.0", port=port)