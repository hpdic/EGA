# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import clip
from torchvision import datasets
import numpy as np
from tqdm import tqdm

def extract_and_save(dataset_name, dataset, model, preprocess, device, embed_dir):
    features_path = os.path.join(embed_dir, f'{dataset_name}_vit_b32_features.npy')
    labels_path = os.path.join(embed_dir, f'{dataset_name}_vit_b32_labels.npy')

    if os.path.exists(features_path):
        print(f'[{dataset_name}] Features already exist. Skipping.')
        return

    all_features = []
    all_labels = []

    with torch.no_grad():
        for img, label in tqdm(dataset, desc=f'Processing {dataset_name}'):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels)

    np.save(features_path, features_array)
    np.save(labels_path, labels_array)
    print(f'[{dataset_name}] Saved to: {features_path}')

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    data_dir = os.path.join(base_dir, 'data')
    embed_dir = os.path.join(base_dir, 'embeddings')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    # 1. 提取源域数据：Flowers102 (用于训练 ICon/SRL 和 EGA)
    print("\n--- Starting Source Domain: Flowers102 ---")
    flowers_dataset = datasets.Flowers102(root=data_dir, split='train', download=True)
    extract_and_save('flowers102_train', flowers_dataset, model, preprocess, device, embed_dir)

    # 2. 提取目标域数据：FGVC Aircraft (用于跨域测试)
    print("\n--- Starting Target Domain: FGVCAircraft ---")
    aircraft_dataset = datasets.FGVCAircraft(root=data_dir, split='test', download=True)
    extract_and_save('aircraft_test', aircraft_dataset, model, preprocess, device, embed_dir)

    print("\nAll extractions completed successfully!")

if __name__ == '__main__':
    main()