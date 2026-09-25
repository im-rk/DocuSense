"""Model 1 dataset: reads the cached 256x256 uint8 pages built by data/build_rvl_cache.py."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def augment_img(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scan/photo-style degradations on a uint8 (H,W) page. Applied to ALL classes so the model cannot use
    'looks like a fax scan vs a photo' as a shortcut."""
    import cv2
    h, w = img.shape
    ang, sc = rng.uniform(-3, 3), rng.uniform(0.93, 1.07)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-0.04, 0.04, 2) * (w, h)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.4:                                           # low-resolution scan
        f = rng.uniform(0.5, 0.9)
        img = cv2.resize(cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h))
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    x = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    if rng.random() < 0.4:
        x += rng.normal(0, rng.uniform(2, 10), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                                           # JPEG artefacts
        ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        if ok:
            x = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return x


class ClsDataset(Dataset):
    def __init__(self, data_dir: str, split: str, mean: float, std: float, augment: bool = False,
                 max_items: int = None):
        self.x = np.load(os.path.join(data_dir, f"cls_{split}_x.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(data_dir, f"cls_{split}_y.npy"))
        self.n = len(self.y) if not max_items else min(max_items, len(self.y))
        self.mean, self.std, self.augment, self._rng = mean, std, augment, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = np.array(self.x[i])
        if self.augment:
            if self._rng is None:                                    # per-worker RNG (fork-safe)
                self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
            img = augment_img(img, self._rng)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x)[None], int(self.y[i])
