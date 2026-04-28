# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
import os
import torch
import clip
from torchvision import datasets
import numpy as np
from tqdm import tqdm
from torch.utils.data import ConcatDataset

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    os.makedirs(embed_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    
    # 合并 Flowers102 的所有数据集分支，总计 8189 张图
    data_dir = os.path.join(base_dir, 'data')
    ds_train = datasets.Flowers102(root=data_dir, split='train', download=True)
    ds_val = datasets.Flowers102(root=data_dir, split='val', download=True)
    ds_test = datasets.Flowers102(root=data_dir, split='test', download=True)
    full_dataset = ConcatDataset([ds_train, ds_val, ds_test])

    all_features = []
    all_labels = []

    with torch.no_grad():
        for img, label in tqdm(full_dataset, desc='Extracting All Flowers'):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    np.save(os.path.join(embed_dir, 'flowers_all_features.npy'), np.concatenate(all_features, axis=0))
    np.save(os.path.join(embed_dir, 'flowers_all_labels.npy'), np.array(all_labels))
    print('Flowers All features saved.')

if __name__ == '__main__':
    main()