"""
DocuSense — Extractor Model V2 Training Script (Kaggle)
==========================================================
PRD-compliant version:
  - Dataset     : CORD-v2 (primary) + FUNSD (supplementary) — combined
  - Architecture: Dual-branch CNN (visual 256-dim + position 64-dim)
  - Epochs      : 50
  - LR Scheduler: ReduceLROnPlateau (PRD spec)
  - Metric      : Macro F1 (not just accuracy)
  - Classes     : 7
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, classification_report

print("--- DocuSense Extractor V2 (CORD-v2 + FUNSD) ---")

# =========================================================
# 0. DIAGNOSTIC — Run this first to see real CORD categories
# =========================================================
def explore_cord_categories():
    """
    Prints every unique category string that exists in CORD-v2 train split.
    Run this ONCE to verify our CORD_LABEL_MAP keys are correct.
    After seeing the output, we can fix any mismatched category names.
    """
    from datasets import load_dataset
    import collections

    print("\n[EXPLORE] Loading CORD-v2 train to find all category names...")
    raw = load_dataset("naver-clova-ix/cord-v2", split="train")

    category_counts = collections.Counter()

    for doc in raw:
        try:
            gt = json.loads(doc["ground_truth"])
        except Exception:
            continue
        for line in gt.get("valid_line", []):
            for word_info in line.get("words", []):
                # category is at the LINE level, not the word level
                cat = line.get("category", "__missing__")
                category_counts[cat] += 1

    print(f"\n[EXPLORE] Found {len(category_counts)} unique CORD categories:\n")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        mapped = CORD_LABEL_MAP.get(cat, "→ OTHER (unmapped)")
        print(f"  {count:6d}x  {cat:40s}  {mapped}")
    print()


# =========================================================
# 1. Constants
# =========================================================
NUM_CLASSES = 7
CLASS_NAMES = [
    "VENDOR_NAME",      # 0
    "INVOICE_NUMBER",   # 1
    "DATE",             # 2
    "LINE_ITEM_DESC",   # 3
    "LINE_ITEM_AMOUNT", # 4
    "TOTAL_AMOUNT",     # 5
    "OTHER",            # 6
]
ALL_LABELS = list(range(NUM_CLASSES))

# ── CORD-v2 category → our 7 classes ─────────────────────
# Category names verified by explore_cord_categories() on the real dataset.
CORD_LABEL_MAP = {
    # VENDOR_NAME (0)
    "store_name_value": 0,
    "store_name_key":   0,

    # INVOICE_NUMBER (1) — not present in CORD receipts, kept for completeness
    "invoice_number":   1,

    # DATE (2)
    "store_info_date":   2,
    "payment_info_date": 2,

    # LINE_ITEM_DESC (3)
    "menu.nm":           3,
    "menu.sub.nm":       3,   # Real name is menu.sub.nm not menu.sub_nm
    "menu.unitprice":    3,
    "menu.sub.unitprice":3,
    "menu.etc":          3,

    # LINE_ITEM_AMOUNT (4)
    "menu.price":        4,
    "menu.sub.price":    4,   # Real name is menu.sub.price not menu.sub_price
    "menu.cnt":          4,
    "menu.sub.cnt":      4,   # Real name is menu.sub.cnt
    "menu.num":          4,   # Quantity field
    "menu.discountprice":4,
    "menu.itemsubtotal": 4,

    # TOTAL_AMOUNT (5)
    "total.total_price":         5,
    "total.cashprice":           5,
    "total.changeprice":         5,
    "total.menutype_cnt":        5,
    "total.menuqty_cnt":         5,   # Real name differs from menutype_cnt
    "total.creditcardprice":     5,
    "total.emoneyprice":         5,
    "total.total_etc":           5,
    "sub_total.subtotal_price":  5,   # Real name uses underscore: sub_total
    "sub_total.tax_price":       5,
    "sub_total.discount_price":  5,
    "sub_total.service_price":   5,
    "sub_total.othersvc_price":  5,
    "sub_total.etc":             6,   # misc subtotal notes → OTHER

    # OTHER (6) — everything else falls through via .get(category, 6)
    "void_menu.nm":     6,
    "void_menu.price":  6,
    "menu.vatyn":       6,
}

# ── FUNSD IOB tag → our 7 classes ────────────────────────
FUNSD_LABEL_MAP = {
    0: 6,  # O           → OTHER
    1: 0,  # B-HEADER    → VENDOR_NAME
    2: 0,  # I-HEADER    → VENDOR_NAME
    3: 2,  # B-QUESTION  → DATE (field labels)
    4: 2,  # I-QUESTION  → DATE (field labels)
    5: 5,  # B-ANSWER    → TOTAL_AMOUNT (values)
    6: 5,  # I-ANSWER    → TOTAL_AMOUNT (values)
}

# ── Shared image transform ────────────────────────────────
def build_transform(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# =========================================================
# 2. CORD-v2 Dataset
# =========================================================
class CORDDataset(Dataset):
    """
    CORD-v2 from HuggingFace.
    
    Key fix: CORD's 'quad' is a 4-corner polygon:
      x1,y1 = top-left   x2,y2 = top-right
      x3,y3 = bottom-right  x4,y4 = bottom-left
    We derive the bounding box using min/max of all 4 corners.
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
                # category is at the LINE level — all words in a line share it
                category = line.get("category", "")
                our_label = CORD_LABEL_MAP.get(category, 6)

                for word_info in line.get("words", []):
                    text = word_info.get("text", "").strip()
                    quad = word_info.get("quad", {})

                    if not text or not quad:
                        continue

                    # Derive tight bounding box from all 4 polygon corners
                    try:
                        xs = [int(quad[f"x{i}"]) for i in range(1, 5)]
                        ys = [int(quad[f"y{i}"]) for i in range(1, 5)]
                    except (KeyError, TypeError):
                        continue

                    x1, y1 = max(0, min(xs)), max(0, min(ys))
                    x2, y2 = min(W, max(xs)), min(H, max(ys))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    self.samples.append({
                        "image":        image,
                        "crop_box":     (x1, y1, x2, y2),
                        "pos_features": [
                            (x1 + x2) / 2 / W,
                            (y1 + y2) / 2 / H,
                            (x2 - x1) / W,
                            (y2 - y1) / H,
                        ],
                        "label": our_label,
                    })

        print(f"  CORD-v2 [{hf_split}]: {len(self.samples):,} word regions")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        crop  = s["image"].crop(s["crop_box"])
        return {
            "visual":     self.transform(crop),
            "positional": torch.tensor(s["pos_features"], dtype=torch.float32),
            "label":      torch.tensor(s["label"],         dtype=torch.long),
        }


# =========================================================
# 3. FUNSD Dataset
# =========================================================
class FUNSDDataset(Dataset):
    """
    FUNSD from HuggingFace.
    Bboxes are normalised to a 1000×1000 grid.
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

            for bbox, ner_tag in zip(doc["bboxes"], doc["ner_tags"]):
                x1_n, y1_n, x2_n, y2_n = bbox  # normalised to 1000

                px1 = max(0, int(x1_n * W / 1000))
                py1 = max(0, int(y1_n * H / 1000))
                px2 = min(W, int(x2_n * W / 1000))
                py2 = min(H, int(y2_n * H / 1000))

                if px2 <= px1 or py2 <= py1:
                    continue

                self.samples.append({
                    "image":        image,
                    "crop_box":     (px1, py1, px2, py2),
                    "pos_features": [
                        (x1_n + x2_n) / 2 / 1000,
                        (y1_n + y2_n) / 2 / 1000,
                        (x2_n - x1_n) / 1000,
                        (y2_n - y1_n) / 1000,
                    ],
                    "label": FUNSD_LABEL_MAP.get(ner_tag, 6),
                })

        print(f"  FUNSD [{split}]: {len(self.samples):,} word regions")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        crop = s["image"].crop(s["crop_box"])
        return {
            "visual":     self.transform(crop),
            "positional": torch.tensor(s["pos_features"], dtype=torch.float32),
            "label":      torch.tensor(s["label"],         dtype=torch.long),
        }


# =========================================================
# 4. Model Architecture  (PRD spec)
# =========================================================
class WordRegionExtractor(nn.Module):
    """
    Dual-branch CNN:
      Visual branch   : 4-layer CNN → 256-dim
      Position branch : MLP(4→64)   → 64-dim
      Fusion          : 320-dim → num_classes
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Visual branch → 256-dim
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
            nn.Flatten(),       # → (B, 256)
        )

        # Position branch → 64-dim
        self.position = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # Fusion: 256 + 64 = 320 → num_classes
        self.head = nn.Sequential(
            nn.Linear(320, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
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

    def forward(self, image, position):
        v = self.visual(image)          # (B, 256)
        p = self.position(position)     # (B, 64)
        return self.head(torch.cat([v, p], dim=1))


# =========================================================
# 5. Helpers
# =========================================================
def compute_class_weights(dataset, num_classes):
    """Inverse-frequency weights. Zero-count classes get weight=0."""
    counts = torch.zeros(num_classes)
    for ds in (dataset.datasets if hasattr(dataset, "datasets") else [dataset]):
        for s in ds.samples:
            counts[s["label"]] += 1

    print(f"  Label counts : {counts.tolist()}")
    mask    = counts > 0
    weights = torch.zeros(num_classes)
    weights[mask] = 1.0 / counts[mask]
    total = weights.sum()
    if total > 0:
        weights = weights / total * mask.sum().float()

    print(f"  Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    return weights


def evaluate(model, loader, device):
    """
    Return (macro_f1, report_string).
    Always evaluates all NUM_CLASSES labels even if some are absent
    in predictions — avoids sklearn ValueError.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            vis    = batch["visual"].to(device)
            pos    = batch["positional"].to(device)
            labels = batch["label"].to(device)
            preds  = model(vis, pos).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Pass labels= to force evaluation of all 7 classes
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
    SAVE_PATH  = "/kaggle/working/extractor_v2.pth"
    EPOCHS     = 80    # was 50 — loss was still decreasing at epoch 50
    BATCH_SIZE = 64
    LR         = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    # ── Datasets ──────────────────────────────────────────
    cord_train = CORDDataset(split="train")
    cord_val   = CORDDataset(split="validation")

    funsd_train = FUNSDDataset(split="train")
    funsd_val   = FUNSDDataset(split="test")    # FUNSD has no 'validation'

    train_ds = ConcatDataset([cord_train, funsd_train])
    val_ds   = ConcatDataset([cord_val,   funsd_val])

    print(f"\nCombined train : {len(train_ds):,} samples")
    print(f"Combined val   : {len(val_ds):,} samples\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────
    model = WordRegionExtractor(num_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Class weights ─────────────────────────────────────
    print(f"Computing class weights from training set...")
    weights = compute_class_weights(train_ds, NUM_CLASSES).to(device)

    # TOTAL_AMOUNT (class 5) has the most samples so gets the lowest weight.
    # But its recall was only 0.39, meaning the model misses 61% of totals.
    # We manually boost its weight to force the model to pay more attention.
    if weights[5] > 0:
        weights[5] = weights[5] * 2.0
        total = weights.sum()
        weights = weights / total * (weights > 0).sum().float()  # re-normalize
    print(f"  Boosted TOTAL_AMOUNT weight. Final weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    print()

    # ── Optimiser & Scheduler (PRD spec) ──────────────────
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    history  = {"train_loss": [], "val_f1": []}
    best_f1  = 0.0

    # ── Epoch loop ────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            vis    = batch["visual"].to(device)
            pos    = batch["positional"].to(device)
            labels = batch["label"].to(device)

            logits = model(vis, pos)
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

        # Validate — use unwrapped model for evaluate()
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
            best_f1   = val_f1
            state_dict = (model.module.state_dict()
                          if isinstance(model, nn.DataParallel)
                          else model.state_dict())
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":               val_f1,
            }, SAVE_PATH)
            print(f"  -> New best! Macro F1={val_f1:.3f} saved to {SAVE_PATH}")
            print(report)

    # ── Training curves ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"]); ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(history["val_f1"]);     ax2.set_title("Val Macro F1");  ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("/kaggle/working/extractor_v2_curves.png")
    print(f"\nBest Val Macro F1 : {best_f1:.3f}")
    print("Curves saved      : /kaggle/working/extractor_v2_curves.png")


if __name__ == "__main__":
    # explore_cord_categories()  # Only needed once to verify category names
    train()
