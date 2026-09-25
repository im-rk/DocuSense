"""Page preprocessing for Model 1. Shared by dataset builders (training) and pipeline (serving)."""
import numpy as np
from PIL import Image

IMG_SIZE = 256


def letterbox_gray(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Any PIL image -> uint8 (size,size) grayscale: scaled to fit (aspect kept), centred on white."""
    im = img.convert("L")
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.asarray(im)
    return canvas


def to_tensor_input(gray: np.ndarray, mean: float, std: float):
    """uint8 (H,W) -> float32 (1,1,H,W) normalised with dataset statistics (NOT ImageNet)."""
    x = (gray.astype(np.float32) / 255.0 - mean) / std
    return x[None, None]
