"""
DocuSense — Extractor Model V3 Training Script (Kaggle)
=========================================================
Triple-Branch Architecture (trained 100% from scratch):

  Branch 1 — Visual  : CNN on 128×64 word image crop   → 256-dim
  Branch 2 — Position: MLP on 4 normalized coordinates  → 64-dim
  Branch 3 — Text    : Char-CNN on word ASCII characters → 64-dim
  ─────────────────────────────────────────────────────────────────
  Fusion: 256+64+64 = 384-dim → MLP → 7 semantic classes

Why V2 plateaued at 0.50 Macro F1 and why V3 will be better:
  V2 was BLIND — it couldn't read "Total:" vs "Invoice:" because
  a 64×64 crop of any word looks like the same blurry pixel blob.
  V3 can READ — the Char-CNN learns that words starting with
  "Total", "INV-", "Date" belong to specific classes.

Datasets:
  Primary      : CORD-v2 (naver-clova-ix/cord-v2) — receipts
  Supplementary: FUNSD   (nielsr/funsd)            — scanned forms
  Combined     : ~41,000 word-region training samples

Target: Macro F1 ≥ 0.80 on 6 active classes
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

print("--- DocuSense Extractor V3 (Triple-Branch: Visual + Position + Text) ---")

# =========================================================
# 1. Constants
# =========================================================
NUM_CLASSES = 7
CLASS_NAMES = [
    "VENDOR_NAME",       # 0
    "INVOICE_NUMBER",    # 1
    "DATE",              # 2
    "LINE_ITEM_DESC",    # 3
    "LINE_ITEM_AMOUNT",  # 4
    "TOTAL_AMOUNT",      # 5
    "OTHER",             # 6
]
ALL_LABELS = list(range(NUM_CLASSES))

# Character encoding constants
CHAR_VOCAB_SIZE = 96    # 0=padding, 1–95=printable ASCII (32–126)
MAX_WORD_LEN    = 32    # truncate/pad every word to this length

def encode_word(text: str) -> list:
    """
    Encode a word string as a list of character indices.
    Maps printable ASCII (32–126) → indices 1–95.
    Unknown chars → 1 (space). Pads to MAX_WORD_LEN with 0.
    """
    indices = []
    for ch in text[:MAX_WORD_LEN]:
        code = ord(ch)
        indices.append((code - 31) if 32 <= code <= 126 else 1)
    # Zero-pad to fixed length
    indices += [0] * (MAX_WORD_LEN - len(indices))
    return indices

# ── CORD-v2 category → our 7 classes ─────────────────────
# Keys verified with explore_cord_categories() on the real HuggingFace dataset.
CORD_LABEL_MAP = {
    "store_name_value":      0,   # VENDOR_NAME
    "store_name_key":        0,
    "invoice_number":        1,   # INVOICE_NUMBER (rare in CORD receipts)
    "store_info_date":       2,   # DATE
    "payment_info_date":     2,
    "menu.nm":               3,   # LINE_ITEM_DESC
    "menu.sub.nm":           3,
    "menu.unitprice":        3,
    "menu.sub.unitprice":    3,
    "menu.etc":              3,
    "menu.price":            4,   # LINE_ITEM_AMOUNT
    "menu.sub.price":        4,
    "menu.cnt":              4,
    "menu.sub.cnt":          4,
    "menu.num":              4,
    "menu.discountprice":    4,
    "menu.itemsubtotal":     4,
    "total.total_price":     5,   # TOTAL_AMOUNT
    "total.cashprice":       5,
    "total.changeprice":     5,
    "total.menutype_cnt":    5,
    "total.menuqty_cnt":     5,
    "total.creditcardprice": 5,
    "total.emoneyprice":     5,
    "total.total_etc":       5,
    "sub_total.subtotal_price": 5,
    "sub_total.tax_price":      5,
    "sub_total.discount_price": 5,
    "sub_total.service_price":  5,
    "sub_total.othersvc_price": 5,
    "sub_total.etc":         6,   # OTHER (misc subtotal notes)
    "void_menu.nm":          6,
    "void_menu.price":       6,
    "menu.vatyn":            6,
}

# ── FUNSD IOB tag → our 7 classes ────────────────────────
FUNSD_LABEL_MAP = {
    0: 6,   # O          → OTHER
    1: 0,   # B-HEADER   → VENDOR_NAME
    2: 0,   # I-HEADER   → VENDOR_NAME
    3: 2,   # B-QUESTION → DATE / field label
    4: 2,   # I-QUESTION → DATE / field label
    5: 5,   # B-ANSWER   → TOTAL_AMOUNT / value
    6: 5,   # I-ANSWER   → TOTAL_AMOUNT / value
}

# ── Image transforms ──────────────────────────────────────
def build_transform(split: str):
    """128×64 landscape crop — better aspect ratio for word images."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# =========================================================
# 2. Datasets
# =========================================================
class CORDDataset(Dataset):
    """
    CORD-v2 from HuggingFace.
    - category is at the LINE level (not word level)
    - quad is a 4-corner polygon: use min/max of all corners for bbox
    - text field is stored and passed to Char-CNN branch
    """
    def __init__(self, split="train"):
        from datasets import load_dataset
        hf_split = {"val": "validation"}.get(split, split)
        print(f"Loading CORD-v2 [{hf_split}]...")
        raw = load_dataset("naver-clova-ix/cord-v2", split=hf_split)

        self.samples   = []
        self.transform = build_transform(split)

        for doc in raw:
            image = doc["image"].convert("RGB")
            W, H  = image.size

            try:
                gt = json.loads(doc["ground_truth"])
            except Exception:
                continue

            for line in gt.get("valid_line", []):
                category  = line.get("category", "")
                our_label = CORD_LABEL_MAP.get(category, 6)

                for word_info in line.get("words", []):
                    text = word_info.get("text", "").strip()
                    quad = word_info.get("quad", {})

                    if not text or not quad:
                        continue

                    # Derive bounding box from all 4 polygon corners
                    try:
                        xs = [int(quad[f"x{i}"]) for i in range(1, 5)]
                        ys = [int(quad[f"y{i}"]) for i in range(1, 5)]
                    except (KeyError, TypeError):
                        continue

                    x1 = max(0, min(xs)); y1 = max(0, min(ys))
                    x2 = min(W, max(xs)); y2 = min(H, max(ys))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    self.samples.append({
                        "image":        image,
                        "crop_box":     (x1, y1, x2, y2),
                        "pos_features": [(x1+x2)/2/W, (y1+y2)/2/H,
                                         (x2-x1)/W,   (y2-y1)/H],
                        "text":         text,
                        "label":        our_label,
                    })

        print(f"  CORD-v2 [{hf_split}]: {len(self.samples):,} word regions")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        crop = s["image"].crop(s["crop_box"])
        return {
            "visual":     self.transform(crop),
            "positional": torch.tensor(s["pos_features"], dtype=torch.float32),
            "text":       torch.tensor(encode_word(s["text"]), dtype=torch.long),
            "label":      torch.tensor(s["label"], dtype=torch.long),
        }


class FUNSDDataset(Dataset):
    """
    FUNSD from HuggingFace.
    - bboxes normalized to 1000×1000 grid
    - words list contains the actual text for the Char-CNN branch
    """
    def __init__(self, split="train"):
        from datasets import load_dataset
        print(f"Loading FUNSD [{split}]...")
        raw = load_dataset("nielsr/funsd", split=split)

        self.samples   = []
        self.transform = build_transform(split)

        for doc in raw:
            image = doc["image"].convert("RGB")
            W, H  = image.size

            for word, bbox, ner_tag in zip(doc["words"],
                                           doc["bboxes"],
                                           doc["ner_tags"]):
                x1_n, y1_n, x2_n, y2_n = bbox

                px1 = max(0, int(x1_n * W / 1000))
                py1 = max(0, int(y1_n * H / 1000))
                px2 = min(W, int(x2_n * W / 1000))
                py2 = min(H, int(y2_n * H / 1000))

                if px2 <= px1 or py2 <= py1:
                    continue

                self.samples.append({
                    "image":        image,
                    "crop_box":     (px1, py1, px2, py2),
                    "pos_features": [(x1_n+x2_n)/2/1000, (y1_n+y2_n)/2/1000,
                                     (x2_n-x1_n)/1000,   (y2_n-y1_n)/1000],
                    "text":         word,
                    "label":        FUNSD_LABEL_MAP.get(ner_tag, 6),
                })

        print(f"  FUNSD [{split}]: {len(self.samples):,} word regions")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        crop = s["image"].crop(s["crop_box"])
        return {
            "visual":     self.transform(crop),
            "positional": torch.tensor(s["pos_features"], dtype=torch.float32),
            "text":       torch.tensor(encode_word(s["text"]), dtype=torch.long),
            "label":      torch.tensor(s["label"], dtype=torch.long),
        }


# =========================================================
# 3. Model Architecture
# =========================================================
class CharCNN(nn.Module):
    """
    Character-level CNN — trained completely from scratch.
    Reads the actual word text and learns patterns like:
      "Total*"  → TOTAL_AMOUNT
      "INV-*"   → INVOICE_NUMBER
      "Date*"   → DATE
      "$*" / "*.00" → AMOUNT
    
    Architecture:
      Embedding(96, 32) → Parallel Conv1D(k=2,3,5) → MaxPool → Linear → 64-dim
    """
    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, embed_dim=32, out_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Parallel convolutions to capture different n-gram lengths
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)

        # 64 + 64 + 64 = 192 → out_dim
        self.fc   = nn.Linear(192, out_dim)
        self.relu = nn.ReLU(inplace=True)

        # He initialisation for linear layers
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """x: (batch, MAX_WORD_LEN) int indices"""
        e = self.embedding(x)           # (B, L, 32)
        e = e.transpose(1, 2)           # (B, 32, L)  — Conv1d expects (B,C,L)

        # Global max-pool over sequence dimension
        c2 = self.relu(self.conv2(e)).max(dim=2)[0]  # (B, 64)
        c3 = self.relu(self.conv3(e)).max(dim=2)[0]  # (B, 64)
        c5 = self.relu(self.conv5(e)).max(dim=2)[0]  # (B, 64)

        cat = torch.cat([c2, c3, c5], dim=1)          # (B, 192)
        return self.relu(self.fc(cat))                 # (B, out_dim)


class WordRegionExtractorV3(nn.Module):
    """
    Triple-Branch Word Region Classifier.

    Branch 1 — Visual  : 4-layer CNN on 128×64 crop  → 256-dim
    Branch 2 — Position: MLP(4 → 64)                 →  64-dim
    Branch 3 — Text    : CharCNN on ASCII chars       →  64-dim
    Fusion             : Linear(384 → 128 → classes)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # ── Branch 1: Visual CNN → 256-dim ───────────────
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),   # → (B, 256)
        )

        # ── Branch 2: Position MLP → 64-dim ──────────────
        self.position = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # ── Branch 3: Text Char-CNN → 64-dim ─────────────
        self.text_branch = CharCNN(out_dim=64)

        # ── Fusion: 256 + 64 + 64 = 384 → classes ────────
        self.head = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, image, position, text_indices):
        v = self.visual(image)               # (B, 256)
        p = self.position(position)          # (B, 64)
        t = self.text_branch(text_indices)   # (B, 64)
        return self.head(torch.cat([v, p, t], dim=1))


# =========================================================
# 4. Loss — Weighted CrossEntropy with manual TOTAL_AMOUNT boost
# =========================================================
# Focal Loss was removed because it suppresses loss for high-confidence
# easy examples. TOTAL_AMOUNT has many easy examples (numbers at bottom
# of receipts), so Focal Loss cuts their gradient by ~96%, stopping the
# model from learning the hard cases. Result was Recall=0.13.
# Plain weighted CrossEntropy with an explicit TOTAL_AMOUNT boost is
# simpler, more stable, and fixes the recall problem.


# =========================================================
# 5. Helpers
# =========================================================
def compute_class_weights(dataset, num_classes):
    """Inverse-frequency weights. Zero-count classes get weight=0."""
    counts = torch.zeros(num_classes)
    for ds in (dataset.datasets if hasattr(dataset, "datasets") else [dataset]):
        for s in ds.samples:
            counts[s["label"]] += 1

    print(f"  Label counts : {[int(c) for c in counts.tolist()]}")
    mask    = counts > 0
    weights = torch.zeros(num_classes)
    weights[mask] = 1.0 / counts[mask]
    total = weights.sum()
    if total > 0:
        weights = weights / total * mask.sum().float()

    print(f"  Raw weights  : {[f'{w:.3f}' for w in weights.tolist()]}")
    return weights


def evaluate(model, loader, device):
    """Macro F1 + full classification report over all NUM_CLASSES."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            vis    = batch["visual"].to(device)
            pos    = batch["positional"].to(device)
            text   = batch["text"].to(device)
            labels = batch["label"].to(device)
            preds  = model(vis, pos, text).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds,
                        labels=ALL_LABELS, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        labels=ALL_LABELS, target_names=CLASS_NAMES, zero_division=0,
    )
    return macro_f1, report


# =========================================================
# 6. Training Loop
# =========================================================
def train():
    SAVE_PATH  = "/kaggle/working/extractor_v3.pth"
    EPOCHS     = 60
    BATCH_SIZE = 64
    LR         = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    # ── Datasets ──────────────────────────────────────────
    cord_train  = CORDDataset(split="train")
    cord_val    = CORDDataset(split="validation")
    funsd_train = FUNSDDataset(split="train")
    funsd_val   = FUNSDDataset(split="test")

    train_ds = ConcatDataset([cord_train, funsd_train])
    val_ds   = ConcatDataset([cord_val,   funsd_val])

    print(f"\nCombined train : {len(train_ds):,} samples")
    print(f"Combined val   : {len(val_ds):,} samples\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────
    model = WordRegionExtractorV3(num_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # ── Class weights ─────────────────────────────────────
    print("Computing class weights...")
    weights = compute_class_weights(train_ds, NUM_CLASSES).to(device)

    # TOTAL_AMOUNT (class 5) has the most samples → gets lowest auto-weight.
    # This causes Recall=0.13 because the model barely tries to predict it.
    # Manually boost it 2.5× so the model pays more attention.
    if weights[5] > 0:
        weights[5] = weights[5] * 2.5
        # Re-normalize so weights sum correctly
        nz = (weights > 0).sum().float()
        weights = weights / weights.sum() * nz
    print(f"  TOTAL_AMOUNT boosted. Final: {[f'{w:.3f}' for w in weights.tolist()]}\n")

    # Plain weighted CrossEntropy — simple and stable
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  patience=6, factor=0.5)

    history = {"train_loss": [], "val_f1": []}
    best_f1 = 0.0

    # ── Epoch loop ────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            vis    = batch["visual"].to(device)
            pos    = batch["positional"].to(device)
            text   = batch["text"].to(device)
            labels = batch["label"].to(device)

            logits = model(vis, pos, text)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total

        # Evaluate
        raw_model      = model.module if isinstance(model, nn.DataParallel) else model
        val_f1, report = evaluate(raw_model, val_loader, device)

        scheduler.step(val_f1)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Macro F1: {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            state_dict = (model.module.state_dict()
                          if isinstance(model, nn.DataParallel)
                          else model.state_dict())
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":               val_f1,
                "arch":                 "WordRegionExtractorV3",
            }, SAVE_PATH)
            print(f"  -> New best! Macro F1={val_f1:.3f} saved to {SAVE_PATH}")
            print(report)

    # ── Training curves ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"]); ax1.set_title("Training Loss");  ax1.set_xlabel("Epoch")
    ax2.plot(history["val_f1"]);     ax2.set_title("Val Macro F1");   ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("/kaggle/working/extractor_v3_curves.png")
    print(f"\nBest Val Macro F1 : {best_f1:.3f}")
    print("Curves saved      : /kaggle/working/extractor_v3_curves.png")


if __name__ == "__main__":
    train()
