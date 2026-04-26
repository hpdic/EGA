# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import clip
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm

def main():
    # 1. Define all absolute paths first
    base_dir = os.path.expanduser("~/hpdic/EGA")
    data_dir = os.path.join(base_dir, "data")
    embed_dir = os.path.join(base_dir, "embeddings")
    
    features_path = os.path.join(embed_dir, "cifar10_vit_b32_features.npy")
    labels_path = os.path.join(embed_dir, "cifar10_vit_b32_labels.npy")

    # 2. Check if output files already exist to skip everything
    if os.path.exists(features_path) and os.path.exists(labels_path):
        print(f"Features already exist at {features_path}. Skipping extraction.")
        return

    # 3. Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 4. Load frozen ViT model
    print("Loading CLIP ViT model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 5. Prepare CIFAR10 dataset
    # The download=True argument in torchvision is idempotent 
    # it only downloads if files are missing or corrupted.
    print("Preparing dataset...")
    os.makedirs(data_dir, exist_ok=True)
    dataset = CIFAR10(root=data_dir, download=True, train=False)

    all_features = []
    all_labels = []

    # 6. Extract features without gradient calculation
    print("Extracting features...")
    with torch.no_grad():
        for img, label in tqdm(dataset, desc="Processing Images"):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    # 7. Save features to disk
    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels)

    os.makedirs(embed_dir, exist_ok=True)
    np.save(features_path, features_array)
    np.save(labels_path, labels_array)

    print("\nExtraction Complete!")
    print(f"Features shape: {features_array.shape}")
    print(f"Saved to: {features_path}")

if __name__ == "__main__":
    main()