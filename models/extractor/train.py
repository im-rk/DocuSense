import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

def train(save_path="models/checkpoints/extractor_v1.pth", epochs=50, batch_size=64, use_class_weights=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    from models.extractor.dataset import ExtractorDataset
    train_ds = ExtractorDataset(split="train")
    val_ds   = ExtractorDataset(split="validation")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    from models.extractor.architecture import WordRegionExtractor
    model = WordRegionExtractor(num_classes=7).to(device)
    
    if use_class_weights:
        class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 1.0, 1.5, 0.1]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1 = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            positions = batch['position'].to(device)
            labels = batch['label'].to(device)

            logits = model(images, positions)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                positions = batch['position'].to(device)
                labels = batch['label'].to(device)
                logits = model(images, positions)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(f1)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Macro F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': f1,
            }, save_path)
            print(f"  -> New best! Saved to {save_path}")

    return best_f1

if __name__ == "__main__":
    train()
