import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

# We use 5 of RVL-CDIP's 16 classes
RVL_CLASS_MAP = {
    0: 0,   # letter -> LETTER
    1: 1,   # form -> FORM
    2: 2,   # invoice -> INVOICE
    9: 3,   # advertisement -> ADVERTISEMENT
    15: 4,  # scientific -> SCIENTIFIC
}

class RVLCDIPDataset(Dataset):
    def __init__(self, split="train", kaggle_dir="/kaggle/input/rvl-cdip", max_samples=None):
        self.samples = []
        
        # 1. Check if running in Kaggle with the dataset mounted
        if os.path.exists(kaggle_dir):
            print(f"Loading from Kaggle mount: {kaggle_dir}")
            label_file = os.path.join(kaggle_dir, "labels", f"{split}.txt")
            image_dir = os.path.join(kaggle_dir, "images")
            
            with open(label_file, "r") as f:
                for line in f:
                    img_path, label_idx = line.strip().split()
                    label_idx = int(label_idx)
                    
                    if label_idx in RVL_CLASS_MAP:
                        self.samples.append({
                            "image_path": os.path.join(image_dir, img_path),
                            "label": RVL_CLASS_MAP[label_idx]
                        })
        # 2. Fall back to HuggingFace if not in Kaggle
        else:
            print("Kaggle mount not found, falling back to HuggingFace...")
            raw = load_dataset("chainyo/rvl-cdip", split=split)
            for s in raw:
                if s['label'] in RVL_CLASS_MAP:
                    self.samples.append({
                        "image": s['image'],
                        "label": RVL_CLASS_MAP[s['label']]
                    })
                    
        if max_samples:
            self.samples = self.samples[:max_samples]

        self.transform = self._build_transform(split)

    def _build_transform(self, split):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Handle Kaggle local path vs HuggingFace PIL Image
        if "image_path" in sample:
            image = Image.open(sample["image_path"]).convert('RGB')
        else:
            image = sample['image'].convert('RGB')
            
        label = sample['label']
        return {"image": self.transform(image), "label": torch.tensor(label)}
