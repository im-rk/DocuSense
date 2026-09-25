"""Model 1 inference (CPU)."""
import numpy as np
import torch
from PIL import Image

from models.classifier.architecture import build_classifier
from models.classifier.preprocess import IMG_SIZE, letterbox_gray, to_tensor_input


def load_classifier(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device)
    m = build_classifier(ck["config"]["arch"], ck["config"]["n_cls"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m, {"classes": ck["classes"], "mean": ck["mean"], "std": ck["std"], "device": device,
               "metrics": ck.get("metrics", {})}


@torch.no_grad()
def classify(model, meta: dict, img) -> tuple:
    """img: PIL image or uint8 grayscale array -> (class_name, confidence, {class: prob})."""
    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    x = torch.from_numpy(to_tensor_input(letterbox_gray(pil, IMG_SIZE), meta["mean"], meta["std"])).to(meta["device"])
    p = torch.softmax(model(x), 1)[0].cpu().numpy()
    i = int(p.argmax())
    return meta["classes"][i], float(p[i]), {c: float(v) for c, v in zip(meta["classes"], p)}
