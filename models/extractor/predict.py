import torch
import torchvision.transforms as transforms
from PIL import Image

def extract_words(model, image: Image.Image, words_with_boxes: list, device="cpu"):
    """
    model: WordRegionExtractor
    image: Full document image
    words_with_boxes: [{'text': 'Total', 'x1': 100, 'y1': 200, 'x2': 150, 'y2': 220, 
                        'x1_norm': 0.1, ...}, ...]
    Returns: list of dicts with added 'label' and 'conf'
    """
    if not words_with_boxes:
        return []

    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Process all crops in a batch
    batch_images = []
    batch_positions = []
    
    W, H = image.size
    
    for word in words_with_boxes:
        box = (word['x1'], word['y1'], word['x2'], word['y2'])
        # Handle invalid boxes
        if box[2] <= box[0] or box[3] <= box[1]:
            # fallback to 1x1 crop
            crop = image.crop((0, 0, 1, 1)).convert('RGB')
        else:
            crop = image.crop(box).convert('RGB')
            
        img_t = transform(crop)
        batch_images.append(img_t)
        
        pos_t = torch.tensor([word.get('x1_norm', word['x1']/W),
                              word.get('y1_norm', word['y1']/H),
                              word.get('x2_norm', word['x2']/W),
                              word.get('y2_norm', word['y2']/H)], dtype=torch.float32)
        batch_positions.append(pos_t)
        
    img_batch = torch.stack(batch_images).to(device)
    pos_batch = torch.stack(batch_positions).to(device)
    
    CLASS_NAMES = ["VENDOR_NAME", "INVOICE_NUMBER", "DATE", "AMOUNT", "LINE_ITEM_DESC", "LINE_ITEM_AMOUNT", "OTHER"]
    
    with torch.no_grad():
        logits = model(img_batch, pos_batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confs, preds = torch.max(probs, 1)
        
    for i, word in enumerate(words_with_boxes):
        word['label'] = CLASS_NAMES[preds[i].item()]
        word['conf'] = confs[i].item()
        
    return words_with_boxes
