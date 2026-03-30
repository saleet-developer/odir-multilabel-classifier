import cv2
import numpy as np
import torch

def preprocess_retinal(img_rgb, target_size=512):
    """
    Preprocess a retinal fundus image: auto-crop, letterbox, CLAHE, Ben Graham, resize, normalize.
    Returns:
        tensor: normalized image tensor (1, 3, target_size, target_size)
        resized: the image after letterbox+resize (for visualisation)
    Raises:
        ValueError if the input image is invalid.
    """
    if img_rgb is None:
        raise ValueError("No image provided.")
    if not isinstance(img_rgb, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(img_rgb)}")
    if img_rgb.size == 0:
        raise ValueError("Image is empty.")
    if len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Image shape must be (H, W, 3), got {img_rgb.shape}")

    # Auto-crop (remove black borders)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        cropped = img_rgb[ymin:ymax+1, xmin:xmax+1]
    else:
        cropped = img_rgb

    # Letterbox to square
    h, w = cropped.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    # Resize
    resized = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # CLAHE
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Ben Graham's colour constancy
    blurred = cv2.GaussianBlur(enhanced, (0,0), 10)
    final = cv2.addWeighted(enhanced, 4, blurred, -4, 128)

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    final = final.astype(np.float32) / 255.0
    final = (final - mean) / std

    # Convert to tensor
    tensor = torch.from_numpy(final).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, resized