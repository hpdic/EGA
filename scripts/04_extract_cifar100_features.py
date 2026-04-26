# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026
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
    base_dir = os.path.expanduser('~/hpdic/EGA')
    data_dir = os.path.join(base_dir, 'data')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    features_path = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    labels_path = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')

    if os.path.exists(features_path):
        print(f'CIFAR-100 features already exist. Skipping.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    os.makedirs(data_dir, exist_ok=True)
    dataset = CIFAR100(root=data_dir, download=True, train=False)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for img, label in tqdm(dataset, desc='Processing CIFAR-100'):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels)

    os.makedirs(embed_dir, exist_ok=True)
    np.save(features_path, features_array)
    np.save(labels_path, labels_array)
    print(f'Saved to: {features_path}')

if __name__ == '__main__':
    main()