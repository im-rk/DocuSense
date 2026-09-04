import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

def train(save_path="models/checkpoints/classifier_v1.pth", epochs=30, batch_size=32):
    # Kaggle output routing
    if os.path.exists("/kaggle/working"):
        save_path = "/kaggle/working/classifier_v1.pth"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Data
    from models.classifier.dataset import RVLCDIPDataset
    train_ds = RVLCDIPDataset(split="train")
    
    # Ensure validation split name matches what vaclavpechtor dataset uses
    try:
        val_ds = RVLCDIPDataset(split="validation")
    except ValueError:
        val_ds = RVLCDIPDataset(split="test")
        
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2)

    # Model
    from models.classifier.architecture import DocumentClassifierCNN
    model = DocumentClassifierCNN(num_classes=5).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history = {"train_loss": [], "val_acc": []}
    best_val_acc = 0.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        # --- Train ---
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

        # --- Validate ---
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

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  -> New best! Saved to {save_path}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"]); ax1.set_title("Training Loss")
    ax2.plot(history["val_acc"]);    ax2.set_title("Validation Accuracy")
    plt.savefig("classifier_training_curves.png")
    print("Saved: classifier_training_curves.png")
    return best_val_acc

if __name__ == "__main__":
    train()
