"""Model 2 inference (CPU): OCR words -> labelled words."""
import numpy as np
import torch

from data.crops import make_crops
from data.reading_order import order_words
from models.tagger.architecture import build_tagger
from models.tagger.features import MAX_WORDS, crops_to_float, encode_chars, pos_features
from models.tagger.labels import LABELS


def load_tagger(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device)
    model = build_tagger(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    vocab = {ch: i + 2 for i, ch in enumerate(ck["chars"])}
    return model, vocab, {"metrics": ck.get("metrics", {}), "device": device}


def build_inputs(ordered: list, gray: np.ndarray, vocab: dict):
    """Ordered words (with is_first/is_last) + page -> the same tensors the dataset yields (parity with training)."""
    H, W = gray.shape
    boxes = np.array([w["box"] for w in ordered], dtype=np.float32)
    return (torch.from_numpy(encode_chars([w["text"] for w in ordered], vocab)),
            torch.from_numpy(crops_to_float(make_crops(gray, boxes))),
            torch.from_numpy(pos_features(boxes, W, H, [w["is_first"] for w in ordered], [w["is_last"] for w in ordered])))


@torch.no_grad()
def tag_words(model, vocab: dict, gray: np.ndarray, words: list, device: str = "cpu") -> list:
    """words: OCR output [{text, box}] (any order). Returns words in reading order with
    line, is_first, is_last, label (int), label_name, prob added."""
    ordered = order_words([w for w in words if str(w.get("text", "")).strip()])
    out = []
    for s in range(0, len(ordered), MAX_WORDS):                     # very long pages are tagged in chunks
        chunk = ordered[s: s + MAX_WORDS]
        chars, crops, pos = (t[None].to(device) for t in build_inputs(chunk, gray, vocab))
        pad = torch.zeros(1, len(chunk), dtype=torch.bool, device=device)
        prob = model(chars, crops, pos, pad).softmax(-1)[0].cpu().numpy()
        for w, p in zip(chunk, prob):
            k = int(p.argmax())
            out.append(dict(w, label=k, label_name=LABELS[k], prob=float(p[k])))
    return out
