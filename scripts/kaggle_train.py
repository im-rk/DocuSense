import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from PIL import Image

print("--- Kaggle RVL-CDIP Training Script ---")

# ==========================================
# 1. Dataset
# ==========================================
RVL_CLASS_MAP = {
    0: 0,   # letter -> LETTER
    1: 1,   # form -> FORM
    2: 2,   # invoice -> INVOICE
    9: 3,   # advertisement -> ADVERTISEMENT
    15: 4,  # scientific -> SCIENTIFIC
}

class RVLCDIPDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        from datasets import load_dataset
        
        print("Loading Parquet dataset (cached on disk)...")
        raw = load_dataset("chainyo/rvl-cdip", split=split)
        
        print(f"Filtering {split} dataset for our 5 classes...")
        # Get all labels quickly without loading images into RAM
        all_labels = raw['label']
        valid_indices = [i for i, lbl in enumerate(all_labels) if lbl in RVL_CLASS_MAP]
        
        if max_samples:
            valid_indices = valid_indices[:max_samples]
            
        # Select creates a fast, memory-mapped view taking 0 RAM
        self.raw = raw.select(valid_indices)

        self.transform = self._build_transform(split)
        print(f"Loaded {len(self.raw)} {split} images.")

    def _build_transform(self, split):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        sample = self.raw[idx]
        image = sample["image"].convert('RGB')
        label = RVL_CLASS_MAP[sample['label']]
        return {"image": self.transform(image), "label": torch.tensor(label)}


# ==========================================
# 2. Architecture
# ==========================================
class DocumentClassifierCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==========================================
# 3. Training Loop
# ==========================================
def train():
    save_path = "/kaggle/working/classifier_v1.pth"
    epochs = 30
    batch_size = 32
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize Datasets
    train_ds = RVLCDIPDataset(split="train")
    
    try:
        val_ds = RVLCDIPDataset(split="val")
    except:
        try:
            val_ds = RVLCDIPDataset(split="validation")
        except:
            val_ds = RVLCDIPDataset(split="test")
            
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = DocumentClassifierCNN(num_classes=5).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_acc": []}
    best_val_acc = 0.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(images)
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
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(images)
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
    plt.savefig("/kaggle/working/classifier_training_curves.png")
    print("Saved curves to /kaggle/working/classifier_training_curves.png")

if __name__ == "__main__":
    train()
