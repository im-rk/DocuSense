from datasets import load_dataset
import matplotlib.pyplot as plt
import os

def download_and_explore():
    os.makedirs('scripts', exist_ok=True)
    
    print("Downloading RVL-CDIP (classifier dataset)...")
    rvl = load_dataset("abarsky/rvl_cdip_small", split="train")  # start with a small subset
    print(f"RVL-CDIP: {len(rvl)} samples, classes: {rvl.features['label'].names}")
    
    print("Downloading CORD-v2 (extractor dataset)...")
    cord = load_dataset("naver-clova-ix/cord-v2", trust_remote_code=True)
    print(f"CORD train: {len(cord['train'])}, test: {len(cord['test'])}")
    
    print("Downloading FUNSD (form dataset)...")
    funsd = load_dataset("nielsr/funsd", trust_remote_code=True)
    print(f"FUNSD train: {len(funsd['train'])}, test: {len(funsd['test'])}")
    
    # Visualise a sample from each
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rvl[0]['image'])
    axes[0].set_title(f"RVL-CDIP: {rvl.features['label'].names[rvl[0]['label']]}")
    axes[1].imshow(cord['train'][0]['image'])
    axes[1].set_title("CORD-v2 receipt")
    axes[2].imshow(funsd['train'][0]['image'])
    axes[2].set_title("FUNSD form")
    plt.savefig("data_samples.png")
    print("Saved data_samples.png — open it to see what the data looks like")

if __name__ == "__main__":
    download_and_explore()
