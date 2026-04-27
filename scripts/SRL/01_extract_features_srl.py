# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import clip
from torchvision.datasets import CIFAR100
import numpy as np
from tqdm import tqdm

def main():
    base_dir = os.path.expanduser("~/hpdic/EGA")
    data_dir = os.path.join(base_dir, "data")
    embed_dir = os.path.join(base_dir, "embeddings")
    
    # 这里的路径改为 SRL 专属
    features_path = os.path.join(embed_dir, "cifar100_srl_features.npy")
    labels_path = os.path.join(embed_dir, "cifar100_srl_labels.npy")

    # 指向咱们刚刚跑完的 SRL 权重文件
    srl_checkpoint_path = os.path.join(base_dir, "checkpoints", "srl_vit_b32.pth")

    if os.path.exists(features_path):
        print(f"SRL features already exist. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 强制转为 float32 保证推理精度
    model.float()
    
    if os.path.exists(srl_checkpoint_path):
        print(f"Loading SRL weights from {srl_checkpoint_path}")
        model.load_state_dict(torch.load(srl_checkpoint_path, map_location=device))
    else:
        print(f"Warning: SRL checkpoint not found at {srl_checkpoint_path}.")
        return

    model.eval()

    os.makedirs(data_dir, exist_ok=True)
    dataset = CIFAR100(root=data_dir, download=True, train=False)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for img, label in tqdm(dataset, desc="Processing CIFAR100 with SRL"):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels)

    os.makedirs(embed_dir, exist_ok=True)
    np.save(features_path, features_array)
    np.save(labels_path, labels_array)
    print(f"Saved to: {features_path}")

if __name__ == "__main__":
    main()