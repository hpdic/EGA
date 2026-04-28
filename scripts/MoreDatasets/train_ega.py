# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
from models.ega_mlp import EGAMLP

class HardSemanticDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = labels
        self.num_samples = len(features)
        self.label_to_indices = collections.defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.classes = list(self.label_to_indices.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anchor = self.features[idx]
        anchor_label = self.labels[idx]
        
        # 随机选一个正样本
        pos_indices = self.label_to_indices[anchor_label]
        pos_idx = np.random.choice(pos_indices)
        positive = self.features[pos_idx]
        
        # 简单的硬负采样：选一个和当前类标签最接近的类（这里简化为随机选两个类比距离）
        # 或者增加负样本的数量，增加拉开空间的压力
        neg_label = np.random.choice(self.classes)
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.classes)
            
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        negative = self.features[neg_idx]
        
        return anchor, positive, negative

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    # 尝试加载更大的 Flowers 特征集（建议你重新提取一次 flowers102_test 分支）
    # 如果还没提取，就先用目前的，但我把训练逻辑加强了
    src_feat_path = os.path.join(embed_dir, 'flowers102_train_vit_b32_features.npy')
    src_label_path = os.path.join(embed_dir, 'flowers102_train_vit_b32_labels.npy')
    tgt_feat_path = os.path.join(embed_dir, 'aircraft_test_vit_b32_features.npy')
    
    src_output_path = os.path.join(embed_dir, 'flowers102_ega_features.npy')
    tgt_output_path = os.path.join(embed_dir, 'aircraft_ega_features.npy')

    src_features = np.load(src_feat_path).astype(np.float32)
    src_features = src_features / np.linalg.norm(src_features, axis=1, keepdims=True)
    src_labels = np.load(src_label_path)
    
    dataset = HardSemanticDataset(src_features, src_labels)
    # 增加 Batch Size 配合 A100，让梯度更稳定
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    # 略微提高学习率，配合更大的 Batch
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    # 增加 Margin 到 0.4，强迫空间拉得更开
    criterion = nn.TripletMarginLoss(margin=0.4, p=2)

    model.train()
    print('Starting Turbo EGA Training with Larger Margin...')
    for epoch in range(200): # 增加训练轮数
        total_loss = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/200, Loss: {total_loss / len(loader):.4f}')

    model.eval()
    with torch.no_grad():
        src_tensor = torch.from_numpy(src_features).to(device)
        src_transformed = model(src_tensor).cpu().numpy()
        np.save(src_output_path, src_transformed)
        
        tgt_features = np.load(tgt_feat_path).astype(np.float32)
        tgt_features = tgt_features / np.linalg.norm(tgt_features, axis=1, keepdims=True)
        tgt_tensor = torch.from_numpy(tgt_features).to(device)
        tgt_transformed = model(tgt_tensor).cpu().numpy()
        np.save(tgt_output_path, tgt_transformed)
    
    print('Finished Turbo Training.')

if __name__ == '__main__':
    main()
    