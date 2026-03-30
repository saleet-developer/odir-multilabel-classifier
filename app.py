import gradio as gr
import torch
import numpy as np
import cv2
from src.model import DRMultiLabelModel
from src.preprocessing import preprocess_retinal
from src.utils import generate_gradcam

def is_valid_fundus(img_rgb):
    """
    Heuristic to check if an image is a retinal fundus photo.
    Returns True if likely a fundus, False otherwise.
    """
    # 1. Check image dimensions (at least 100x100)
    h, w = img_rgb.shape[:2]
    if h < 100 or w < 100:
        return False

    # 2. Foreground ratio (non‑black pixels)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    fg_ratio = np.sum(mask > 0) / (h * w)
    if fg_ratio < 0.4 or fg_ratio > 0.9:
        return False

    # 3. Circularity of the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0 or area == 0:
        return False
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity < 0.6:
        return False

    # 4. Colour distribution – red channel should dominate
    r_mean = np.mean(img_rgb[:, :, 0])
    g_mean = np.mean(img_rgb[:, :, 1])
    b_mean = np.mean(img_rgb[:, :, 2])
    # In fundus, red is typically > green and > blue; also green is often > blue
    if not (r_mean > g_mean and r_mean > b_mean):
        return False

    return True

# Load model and thresholds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DRMultiLabelModel().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

thresholds = np.load('thresholds.npy')
target_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

# Grad‑CAM target layer (adjust if needed)
target_layer = model.backbone.stages[-1]   # for ConvNeXt

def predict(image):
    try:
        # ----- Validate image -----
        if not is_valid_fundus(image):
            raise ValueError("❌ The uploaded image does not appear to be a retinal fundus photo. Please upload a valid fundus image (circular, reddish, with visible optic disc/vessels).")

        # ----- Preprocess and infer -----
        img_tensor, _ = preprocess_retinal(image)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # Apply thresholds
        preds = (probs > thresholds).astype(int)
        results = {target_names[i]: f"{probs[i]:.3f}" for i in range(len(target_names))}

        # Generate heatmap for top predicted class (if any)
        heatmap = None
        if preds.any():
            top_idx = np.argmax(probs)
            heatmap = generate_gradcam(model, img_tensor, target_layer, top_idx)

        return results, heatmap

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        blank_heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
        return {"Error": error_msg}, blank_heatmap

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Fundus Image"),
    outputs=[
        gr.Label(num_top_classes=8, label="Disease Probabilities"),
        gr.Image(type="numpy", label="Grad‑CAM Heatmap")
    ],
    title="Multi‑Disease Retinal Image Classifier",
    description="Detects up to 8 eye diseases from a fundus photograph.\n\n**Diseases:** Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other.\n\n*This tool is for research purposes only and not a certified medical device.*",
    examples=[["examples/example1.jpg"], ["examples/example2.jpg"]]   # optional
)

iface.launch(share=True)