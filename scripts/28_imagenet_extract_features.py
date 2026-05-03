# scripts/35_imagenet1000_extract_features.py
# 提取 ImageNet-mini 全部 1000 类的 CLIP 特征

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

def extract_imagenet1000_features(image_dir, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    all_labels = []
    class_names = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(class_names)} classes")
    
    total_images = 0
    for class_idx, class_name in enumerate(tqdm(class_names, desc="Processing classes")):
        class_path = image_dir / class_name
        image_files = list(class_path.glob("*.JPEG")) + list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_name}")
            continue
        
        for img_file in image_files:
            try:
                image = Image.open(img_file).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.get_image_features(pixel_values=inputs['pixel_values'])
                    image_features = outputs.pooler_output
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_features.append(image_features.cpu().numpy())
                all_labels.append(class_idx)
                total_images += 1
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    print(f"\nTotal images processed: {total_images}")
    
    if len(all_features) == 0:
        print("Error: No features extracted!")
        return
    
    # Save
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    
    np.save(output_dir / "imagenet1000_features.npy", features)
    np.save(output_dir / "imagenet1000_labels.npy", labels)
    
    print(f"Saved {len(features)} features to {output_dir}")
    print(f"Feature shape: {features.shape}")

if __name__ == '__main__':
    image_dir = "/home/cc/hpdic/imagenet-mini/train"
    output_dir = "/home/cc/hpdic/EGA/embeddings"
    
    extract_imagenet1000_features(image_dir, output_dir)