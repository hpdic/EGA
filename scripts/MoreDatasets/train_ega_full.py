# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
from models.ega_mlp import EGAMLP

class RobustDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = labels
        self.label_to_indices = collections.defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.classes = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        anchor = self.features[idx]
        label = self.labels[idx]
        pos_idx = np.random.choice(self.label_to_indices[label])
        neg_label = np.random.choice([c for c in self.classes if c != label])
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        return anchor, self.features[pos_idx], self.features[neg_idx]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    # 加载 8189 个样本的全量花卉特征
    features = np.load(os.path.join(embed_dir, 'flowers_all_features.npy')).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(os.path.join(embed_dir, 'flowers_all_labels.npy'))
    
    loader = DataLoader(RobustDataset(features, labels), batch_size=1024, shuffle=True)
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    
    # 提高学习率到 1e-4，增加 Margin 到 0.5 增加拉力
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)

    model.train()
    print('Training EGA on 8k Flowers samples...')
    for epoch in range(150):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            
    # 保存并应用到飞机数据集
    model.eval()
    with torch.no_grad():
        tgt_feat = np.load(os.path.join(embed_dir, 'aircraft_test_vit_b32_features.npy')).astype(np.float32)
        tgt_feat /= np.linalg.norm(tgt_feat, axis=1, keepdims=True)
        transformed = model(torch.from_numpy(tgt_feat).to(device)).cpu().numpy()
        np.save(os.path.join(embed_dir, 'aircraft_ega_features.npy'), transformed)
    print('Final Transfer Done.')

if __name__ == '__main__':
    main()