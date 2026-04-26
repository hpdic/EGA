import os
import torch
import clip
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm

def main():
    # 1. Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load frozen ViT model
    print("Loading CLIP ViT model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 3. Prepare CIFAR10 dataset
    print("Preparing dataset...")
    os.makedirs("../data", exist_ok=True)
    dataset = CIFAR10(root="../data", download=True, train=False)

    all_features = []
    all_labels = []

    # 4. Extract features without gradient calculation
    print("Extracting features...")
    with torch.no_grad():
        for img, label in tqdm(dataset, desc="Processing Images"):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    # 5. Save features to disk
    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels)

    os.makedirs(os.path.expanduser("~/hpdic/EGA/embeddings"), exist_ok=True)
    features_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_vit_b32_features.npy")
    labels_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_vit_b32_labels.npy")   

    np.save(features_path, features_array)
    np.save(labels_path, labels_array)

    print("\nExtraction Complete!")
    print(f"Features shape: {features_array.shape}")
    print(f"Saved to: {features_path}")

if __name__ == "__main__":
    main()