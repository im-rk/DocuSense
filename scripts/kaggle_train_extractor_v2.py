"""
DocuSense — Extractor Model V2 Training Script (Kaggle)
==========================================================
PRD-compliant version:
  - Dataset : CORD-v2 (primary) + FUNSD (supplementary) — combined
  - Architecture : Dual-branch CNN (visual 256-dim + position 64-dim)
  - Epochs  : 50
  - LR Scheduler: ReduceLROnPlateau (PRD spec)
  - Metric  : Macro F1 (not just accuracy)
  - Classes : 7
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

# ── CORD-v2 category → our class ──────────────────────────
CORD_LABEL_MAP = {
    # VENDOR_NAME (0)
    "store_name_value": 0,
    "store_name_key":   0,

    # INVOICE_NUMBER (1)
    "invoice_number": 1,

    # DATE (2)
    "store_info_date":   2,
    "payment_info_date": 2,

    # LINE_ITEM_DESC (3)
    "menu.nm":        3,
    "menu.sub_nm":    3,
    "menu.unitprice": 3,

    # LINE_ITEM_AMOUNT (4)
    "menu.price":         4,
    "menu.sub_price":     4,
    "menu.cnt":           4,
    "menu.discountprice": 4,
    "menu.itemsubtotal":  4,

    # TOTAL_AMOUNT (5)
    "total.total_price":       5,
    "total.cashprice":         5,
    "total.changeprice":       5,
    "total.menutype_cnt":      5,
    "subtotal.subtotal_price": 5,
    "subtotal.tax_price":      5,
    "subtotal.discount_price": 5,
    "subtotal.service_price":  5,

    # OTHER (6) — catch-all; unlisted keys also → 6
}

# ── FUNSD IOB tag → our class ─────────────────────────────
FUNSD_LABEL_MAP = {
    0: 6,   # O          → OTHER
    1: 0,   # B-HEADER   → VENDOR_NAME
    2: 0,   # I-HEADER   → VENDOR_NAME
    3: 2,   # B-QUESTION → DATE / field label (best proxy)
    4: 2,   # I-QUESTION → DATE / field label
    5: 5,   # B-ANSWER   → TOTAL_AMOUNT / value
    6: 5,   # I-ANSWER   → TOTAL_AMOUNT / value
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
    else:
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
    Loads CORD-v2 from HuggingFace.
    Each word in every receipt becomes one training sample.
    Ground truth is a nested JSON string — we parse it manually.
    """
    def __init__(self, split="train"):
        from datasets import load_dataset

        # CORD-v2 splits: train / validation / test
        print(f"Loading CORD-v2 [{split}]...")
        raw = load_dataset("naver-clova-ix/cord-v2", split=split)

        self.samples   = []
        self.transform = build_transform(split)

        for doc in raw:
            image  = doc["image"].convert("RGB")
            W, H   = image.size

            # ground_truth is a raw JSON string — must parse it
            try:
                gt = json.loads(doc["ground_truth"])
            except Exception:
                continue

            valid_lines = gt.get("valid_line", [])

            for line in valid_lines:
                for word_info in line.get("words", []):
                    text     = word_info.get("text", "").strip()
                    category = word_info.get("category", "")
                    quad     = word_info.get("quad", {})

                    if not text or not quad:
                        continue

                    # quad keys: x1 y1 x2 y2  (pixel coords)
                    x1 = int(quad.get("x1", 0))
                    y1 = int(quad.get("y1", 0))
                    x2 = int(quad.get("x2", 0))
                    y2 = int(quad.get("y2", 0))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    our_label = CORD_LABEL_MAP.get(category, 6)  # unknown → OTHER

                    self.samples.append({
                        "image":        image,
                        "crop_box":     (x1, y1, x2, y2),
                        "pos_features": [
                            ((x1 + x2) / 2) / W,   # x_center normalised
                            ((y1 + y2) / 2) / H,   # y_center normalised
                            (x2 - x1)        / W,   # width normalised
                            (y2 - y1)        / H,   # height normalised
                        ],
                        "label": our_label,
                    })

        print(f"  CORD-v2 [{split}]: {len(self.samples):,} word regions")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        crop  = s["image"].crop(s["crop_box"])
        image = self.transform(crop)
        pos   = torch.tensor(s["pos_features"], dtype=torch.float32)
        label = torch.tensor(s["label"],         dtype=torch.long)
        return {"visual": image, "positional": pos, "label": label}


# =========================================================
# 3. FUNSD Dataset
# =========================================================
class FUNSDDataset(Dataset):
    """
    Loads FUNSD from HuggingFace.
    FUNSD bboxes are normalised to a 1000×1000 grid.
    """
    def __init__(self, split="train"):
        from datasets import load_dataset

        print(f"Loading FUNSD [{split}]...")
        raw = load_dataset("nielsr/funsd", split=split)

        self.samples   = []
        self.transform = build_transform(split)

        for doc in raw:
            image  = doc["image"].convert("RGB")
            W, H   = image.size

            for bbox, ner_tag in zip(doc["bboxes"], doc["ner_tags"]):
                x1_n, y1_n, x2_n, y2_n = bbox   # normalised to 1000

                # convert to pixel coords
                px1 = max(0,  int(x1_n * W / 1000))
                py1 = max(0,  int(y1_n * H / 1000))
                px2 = min(W,  int(x2_n * W / 1000))
                py2 = min(H,  int(y2_n * H / 1000))

                if px2 <= px1 or py2 <= py1:
                    continue

                our_label = FUNSD_LABEL_MAP.get(ner_tag, 6)

                self.samples.append({
                    "image":        image,
                    "crop_box":     (px1, py1, px2, py2),
                    "pos_features": [
                        ((x1_n + x2_n) / 2) / 1000.0,
                        ((y1_n + y2_n) / 2) / 1000.0,
                        (x2_n - x1_n)        / 1000.0,
                        (y2_n - y1_n)        / 1000.0,
                    ],
                    "label": our_label,
                })

        print(f"  FUNSD [{split}]: {len(self.samples):,} word regions")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        crop  = s["image"].crop(s["crop_box"])
        image = self.transform(crop)
        pos   = torch.tensor(s["pos_features"], dtype=torch.float32)
        label = torch.tensor(s["label"],         dtype=torch.long)
        return {"visual": image, "positional": pos, "label": label}


# =========================================================
# 4. Model Architecture  (PRD spec: 256-dim visual + 64-dim position)
# =========================================================
class WordRegionExtractor(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # ── Visual branch: 4-layer CNN → 256-dim ──────────
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   # → (B, 256, 1, 1)
            nn.Flatten(),                   # → (B, 256)
        )

        # ── Position branch: MLP → 64-dim ─────────────────
        self.position = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # ── Fusion head: 256 + 64 = 320 → num_classes ─────
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image, position):
        v = self.visual(image)        # (B, 256)
        p = self.position(position)   # (B, 64)
        return self.head(torch.cat([v, p], dim=1))  # (B, num_classes)


# =========================================================
# 5. Helpers
# =========================================================
def compute_class_weights(dataset, num_classes):
    """Count labels across a ConcatDataset and return inverse-freq weights.
    Classes with zero samples get weight=0 so they don't distort training.
    """
    counts = torch.zeros(num_classes)
    for ds in (dataset.datasets if hasattr(dataset, "datasets") else [dataset]):
        for s in ds.samples:
            counts[s["label"]] += 1
    print(f"  Label counts: {counts.tolist()}")

    # Only compute inverse-freq for classes that actually exist in the data
    weights = torch.zeros(num_classes)
    non_zero_mask = counts > 0
    weights[non_zero_mask] = 1.0 / counts[non_zero_mask]

    # Normalize only over non-zero classes
    total = weights.sum()
    if total > 0:
        weights = weights / total * non_zero_mask.sum().float()

    return weights


def evaluate(model, loader, device):
    """Return macro F1 + full classification report."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            vis    = batch["visual"].to(device)
            pos    = batch["positional"].to(device)
            labels = batch["label"].to(device)
            preds  = model(vis, pos).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, zero_division=0
    )
    return macro_f1, report


# =========================================================
# 6. Training Loop
# =========================================================
def train():
    SAVE_PATH  = "/kaggle/working/extractor_v2.pth"
    EPOCHS     = 50
    BATCH_SIZE = 64
    LR         = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Build datasets ────────────────────────────────────
    cord_train = CORDDataset(split="train")
    cord_val   = CORDDataset(split="validation")

    funsd_train = FUNSDDataset(split="train")
    funsd_val   = FUNSDDataset(split="test")    # FUNSD has no 'validation' split

    # Combine: CORD-v2 (primary) + FUNSD (supplementary)
    train_ds = ConcatDataset([cord_train, funsd_train])
    val_ds   = ConcatDataset([cord_val,   funsd_val])

    print(f"\nCombined train: {len(train_ds):,} samples")
    print(f"Combined val  : {len(val_ds):,} samples\n")

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
    print("Computing class weights from training set...")
    weights = compute_class_weights(train_ds, NUM_CLASSES).to(device)
    print(f"  Weights: {[f'{w:.3f}' for w in weights.tolist()]}\n")

    # ── Optimiser & scheduler (PRD spec) ──────────────────
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  patience=5, factor=0.5)

    history = {"train_loss": [], "val_f1": []}
    best_f1 = 0.0

    # ── Epoch loop ────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

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

        # Validate with Macro F1
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        val_f1, report = evaluate(raw_model, val_loader, device)

        # Step scheduler on val F1 (higher = better)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Macro F1: {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            state_dict = (model.module.state_dict()
                          if isinstance(model, nn.DataParallel)
                          else model.state_dict())
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":               val_f1,
            }, SAVE_PATH)
            print(f"  -> New best! Macro F1={val_f1:.3f} — saved to {SAVE_PATH}")
            print(report)

    # ── Training curves ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"]); ax1.set_title("Training Loss");  ax1.set_xlabel("Epoch")
    ax2.plot(history["val_f1"]);     ax2.set_title("Val Macro F1");   ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("/kaggle/working/extractor_v2_curves.png")
    print(f"\nBest Val Macro F1: {best_f1:.3f}")
    print("Saved curves to /kaggle/working/extractor_v2_curves.png")


if __name__ == "__main__":
    train()
