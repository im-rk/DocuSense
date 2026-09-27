"""Load an uploaded file as PIL page images."""
import os

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ALLOWED_EXT = IMAGE_EXT | {".pdf"}


def load_pages(path: str, dpi: int = 200, max_pages: int = 1, poppler_path: str = "") -> list:
    """-> list of RGB PIL images (first `max_pages` pages). Honours EXIF rotation for phone photos."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pdf2image import convert_from_path
        kw = {"poppler_path": poppler_path} if poppler_path else {}
        pages = convert_from_path(path, dpi=dpi, first_page=1, last_page=max_pages, **kw)
        return [p.convert("RGB") for p in pages]
    if ext not in IMAGE_EXT:
        raise ValueError(f"Unsupported file type: {ext}")
    im = Image.open(path)
    im.seek(0)
    return [ImageOps.exif_transpose(im).convert("RGB")]


def to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))
