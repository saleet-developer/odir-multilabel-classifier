import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, img_tensor, target_layer, class_idx):
    """
    Generate Grad‑CAM heatmap for a specific class.
    """
    # Denormalize for overlay
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_vis = img_tensor[0].cpu().permute(1,2,0).numpy()
    img_vis = img_vis * std + mean
    img_vis = np.clip(img_vis, 0, 1)

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_vis, grayscale_cam, use_rgb=True)
    return visualization