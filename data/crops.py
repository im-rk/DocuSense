"""Word-crop extraction for the 'look' branch. Shared by dataset builders and pipeline (serving)."""
import numpy as np
from PIL import Image, ImageOps

CROP_H, CROP_W = 32, 96


def make_crop(gray: np.ndarray, box, out_h: int = CROP_H, out_w: int = CROP_W) -> np.ndarray:
    """gray: uint8 (H,W) page. box: x0,y0,x1,y1 pixels. Returns uint8 (out_h,out_w): the word, contrast-stretched,
    aspect-preserved, left-aligned, padded with its own background level."""
    H, W = gray.shape
    x0 = int(max(0, np.floor(box[0]))); y0 = int(max(0, np.floor(box[1])))
    x1 = int(min(W, np.ceil(box[2])));  y1 = int(min(H, np.ceil(box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.full((out_h, out_w), 255, dtype=np.uint8)
    im = ImageOps.autocontrast(Image.fromarray(gray[y0:y1, x0:x1]))
    s = min(out_h / im.height, out_w / im.width)
    nw, nh = max(1, round(im.width * s)), max(1, round(im.height * s))
    arr = np.asarray(im.resize((nw, nh), Image.BILINEAR))
    canvas = np.full((out_h, out_w), int(np.median(arr)), dtype=np.uint8)
    top = (out_h - nh) // 2
    canvas[top:top + nh, :nw] = arr
    return canvas


def make_crops(gray: np.ndarray, boxes) -> np.ndarray:
    """-> uint8 (N, out_h, out_w)."""
    if len(boxes) == 0:
        return np.zeros((0, CROP_H, CROP_W), dtype=np.uint8)
    return np.stack([make_crop(gray, b) for b in boxes])
