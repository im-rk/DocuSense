import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

RVL_CLASS_NAMES = ["LETTER", "FORM", "INVOICE", "ADVERTISEMENT", "SCIENTIFIC"]

def classify_document(model, image: Image.Image, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(img_t)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        
    return RVL_CLASS_NAMES[pred.item()], conf.item()
