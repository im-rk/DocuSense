import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from PIL import Image

print("--- Kaggle Extractor (FUNSD) Training Script ---")

# ==========================================
# 1. Dataset (FUNSD Word Regions)
# ==========================================
# FUNSD Labels: 0 (O), 1 (B-HEADER), 2 (I-HEADER), 3 (B-QUESTION), 4 (I-QUESTION), 5 (B-ANSWER), 6 (I-ANSWER)
NUM_CLASSES = 7

class FUNSDWordDataset(Dataset):
    def __init__(self, split="train"):
        from datasets import load_dataset
        
        print(f"Downloading FUNSD {split} dataset...")
        raw = load_dataset("nielsr/funsd", split=split)
        
        self.samples = []
        
        # We will extract each individual word as a separate training sample
        # along with its relative bounding box coordinates for the positional branch.
        for doc in raw:
            image = doc['image'].convert('RGB')
            width, height = image.size
            
            for word, bbox, label in zip(doc['words'], doc['bboxes'], doc['ner_tags']):
                # FUNSD bboxes are [x1, y1, x2, y2] normalized to 1000x1000
                x1, y1, x2, y2 = bbox
                
                # Add 10% padding to give the CNN spatial context around the word
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                # Convert back to actual image pixel coordinates with padding
                px1 = max(0, int(x1 * width / 1000) - pad_x)
                py1 = max(0, int(y1 * height / 1000) - pad_y)
                px2 = min(width, int(x2 * width / 1000) + pad_x)
                py2 = min(height, int(y2 * height / 1000) + pad_y)
                
                # Prevent invalid crops
                if px2 <= px1 or py2 <= py1:
                    continue
                    
                # We store the image object and coordinates so we can crop lazily in __getitem__
                # to save RAM.
                self.samples.append({
                    "image": image,
                    "crop_box": (px1, py1, px2, py2),
                    # Normalized positional features: [x_center, y_center, width, height]
                    "pos_features": [
                        ((x1 + x2) / 2) / 1000.0,
                        ((y1 + y2) / 2) / 1000.0,
                        (x2 - x1) / 1000.0,
                        (y2 - y1) / 1000.0
                    ],
                    "label": label
                })

        self.transform = transforms.Compose([
            transforms.Resize((64, 256)), # Standard size for word crops
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Extracted {len(self.samples)} word regions from {split} documents.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Lazy cropping
        crop = sample["image"].crop(sample["crop_box"])
        vis_feat = self.transform(crop)
        
        pos_feat = torch.tensor(sample["pos_features"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        
        return {"visual": vis_feat, "positional": pos_feat, "label": label}


# ==========================================
# 2. Dual-Branch Architecture
# ==========================================
class WordRegionExtractor(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Branch 1: Visual Features (CNN) - Upgraded Capacity
        self.visual_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((2, 8)) # Output: 256 * 2 * 8 = 4096 dims
        )
        
        # Branch 2: Positional Features (MLP) - Upgraded Capacity
        self.positional_branch = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        
        # Fusion Classifier
        # 4096 (visual) + 128 (positional) = 4224
        self.classifier = nn.Sequential(
            nn.Linear(4224, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, vis, pos):
        v_feat = self.visual_branch(vis)
        v_feat = torch.flatten(v_feat, 1) # [batch_size, 4096]
        
        p_feat = self.positional_branch(pos) # [batch_size, 128]
        
        # Concatenate features
        fused = torch.cat((v_feat, p_feat), dim=1) # [batch_size, 4224]
        
        return self.classifier(fused)


# ==========================================
# 3. Training Loop
# ==========================================
def train():
    save_path = "/kaggle/working/extractor_v1.pth"
    epochs = 20
    batch_size = 128
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize Datasets
    train_ds = FUNSDWordDataset(split="train")
    
    # FUNSD doesn't have a specific 'validation' split in HF, only 'train' and 'test'
    val_ds = FUNSDWordDataset(split="test")
            
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = WordRegionExtractor(num_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Calculate class weights for imbalance
    # FUNSD is heavily imbalanced (lots of 'O' tags, fewer 'HEADER' tags)
    print("Calculating class weights...")
    class_counts = torch.zeros(NUM_CLASSES)
    for sample in train_ds.samples:
        class_counts[sample["label"]] += 1
        
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-5)
    weights = weights / weights.sum() * NUM_CLASSES # Normalize
    weights = weights.to(device)
    print(f"Class weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_acc": []}
    best_val_acc = 0.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            vis = batch['visual'].to(device)
            pos = batch['positional'].to(device)
            labels = batch['label'].to(device)

            logits = model(vis, pos)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                vis = batch['visual'].to(device)
                pos = batch['positional'].to(device)
                labels = batch['label'].to(device)
                logits = model(vis, pos)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Extract underlying state dict if using DataParallel
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  -> New best! Saved to {save_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"]); ax1.set_title("Training Loss")
    ax2.plot(history["val_acc"]);    ax2.set_title("Validation Accuracy")
    plt.savefig("/kaggle/working/extractor_training_curves.png")
    print("Saved curves to /kaggle/working/extractor_training_curves.png")

if __name__ == "__main__":
    train()
