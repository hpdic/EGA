# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026

import os
import torch
import clip
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm

def extract_features(model_type, checkpoint_path=None):
    base_dir = os.path.expanduser("~/hpdic/EGA")
    data_dir = os.path.join(base_dir, "data")
    embed_dir = os.path.join(base_dir, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)

    features_path = os.path.join(embed_dir, f"cifar10_{model_type}_features.npy")
    labels_path = os.path.join(embed_dir, f"cifar10_{model_type}_labels.npy")

    if os.path.exists(features_path):
        print(f"Features for {model_type} already exist. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.float()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading {model_type} weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif checkpoint_path:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    model.eval()
    dataset = CIFAR10(root=data_dir, download=True, train=False)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for img, label in tqdm(dataset, desc=f"Extracting CIFAR10 {model_type}"):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    np.save(features_path, np.concatenate(all_features, axis=0))
    np.save(labels_path, np.array(all_labels))
    print(f"Saved: {features_path}")

def main():
    base_dir = os.path.expanduser("~/hpdic/EGA")
    
    # 所有的模型权重路径
    ega_ckpt = os.path.join(base_dir, "checkpoints/ega_vit_b32.pth")
    icon_ckpt = os.path.join(base_dir, "checkpoints/icon_vit_b32.pth")
    srl_ckpt = os.path.join(base_dir, "checkpoints/srl_vit_b32.pth")

    # 1. 提取原始 CLIP 特征
    extract_features("original")
    # 2. 提取 EGA 特征
    extract_features("ega", ega_ckpt)
    # 3. 提取 ICon 特征
    extract_features("icon", icon_ckpt)
    # 4. 提取 SRL 特征
    extract_features("srl", srl_ckpt)

if __name__ == "__main__":
    main()