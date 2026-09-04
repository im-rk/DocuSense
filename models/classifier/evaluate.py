import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate(model_path="models/checkpoints/classifier_v1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from models.classifier.architecture import DocumentClassifierCNN
    model = DocumentClassifierCNN(num_classes=5)
    
    if not os.path.exists(model_path):
        print(f"Checkpoint not found at {model_path}. Train the model first.")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    from models.classifier.dataset import RVLCDIPDataset, RVL_CLASS_MAP
    test_ds = RVLCDIPDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    class_names = ["LETTER", "FORM", "INVOICE", "ADVERTISEMENT", "SCIENTIFIC"]
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Document Classifier')
    plt.savefig('classifier_confusion_matrix.png')
    print("Saved confusion matrix to classifier_confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
