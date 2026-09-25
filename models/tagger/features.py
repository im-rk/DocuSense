"""Feature construction for Model 2. Used by BOTH training (dataset.py) and serving (predict.py) so the
tensors are identical - that is the train/serve parity rule."""
import numpy as np

from models.tagger.vocab import PAD, UNK

MAX_CHARS = 24
MAX_WORDS = 768
CROP_H, CROP_W = 32, 96
POS_DIM = 10


def pos_features(boxes: np.ndarray, W: float, H: float, is_first, is_last) -> np.ndarray:
    """boxes (N,4) pixel x0,y0,x1,y1 -> (N,10): x0,y0,x1,y1,w,h,xc,yc (0-1 of page), is_first, is_last."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    x0, y0, x1, y1 = b[:, 0] / W, b[:, 1] / H, b[:, 2] / W, b[:, 3] / H
    f = np.stack([x0, y0, x1, y1, x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2,
                  np.asarray(is_first, dtype=np.float32), np.asarray(is_last, dtype=np.float32)], axis=1)
    f[:, :8] = np.clip(f[:, :8], 0.0, 1.0)
    return f.astype(np.float32)


def encode_chars(texts, vocab: dict, max_len: int = MAX_CHARS) -> np.ndarray:
    """list of strings -> (N, max_len) int64 character ids."""
    out = np.full((len(texts), max_len), PAD, dtype=np.int64)
    for i, t in enumerate(texts):
        for j, ch in enumerate(t[:max_len]):
            out[i, j] = vocab.get(ch, UNK)
    return out


def crops_to_float(crops: np.ndarray) -> np.ndarray:
    """uint8 (N,32,96) -> float32 (N,1,32,96); ink is high, background ~0."""
    return (1.0 - crops.astype(np.float32) / 255.0)[:, None, :, :]
