# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 28, 2026

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
import faiss

# 1. 模型定义
class EGAMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        nn.init.zeros_(self.adapter[3].weight)
        nn.init.zeros_(self.adapter[3].bias)

    def forward(self, x):
        out = x + self.adapter(x)
        out = F.normalize(out, p=2, dim=1)
        return out

# 2. 数据集逻辑
class SensitivityDataset(Dataset):
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
        anchor_label = self.labels[idx]
        pos_idx = np.random.choice(self.label_to_indices[anchor_label])
        neg_label = np.random.choice(self.classes)
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.classes)
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        return anchor, self.features[pos_idx], self.features[neg_idx]

# 3. 核心训练与评估函数
def train_and_eval(margin, hidden_dim, train_feats, train_labels, test_feats, test_labels, device):
    model = EGAMLP(input_dim=512, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    loader = DataLoader(SensitivityDataset(train_feats, train_labels), batch_size=1024, shuffle=True)
    
    # 缩短训练轮数以加快灵敏度测试速度
    for epoch in range(80):
        model.train()
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_feats).to(device)
        transformed = model(test_tensor).cpu().numpy()
        
    # 评估 Recall@1 (nprobe=1)
    # 我们将测试集分为 base (75%) 和 query (25%)
    split_idx = int(len(transformed) * 0.75)
    base = transformed[:split_idx]
    query = transformed[split_idx:]
    labels_base = test_labels[:split_idx]
    labels_query = test_labels[split_idx:]
    
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, I = index.search(query, 1)
    recall = np.sum(labels_base[I.flatten()] == labels_query) / len(query)
    return recall

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    feat_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_features.npy')
    label_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_labels.npy')

    features = np.load(feat_path).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)
    
    # 使用 80/20 类切分保证严谨性
    unique_classes = np.unique(labels)
    np.random.seed(42)
    np.random.shuffle(unique_classes)
    train_classes = unique_classes[:80]
    test_classes = unique_classes[80:]
    
    train_feats = features[np.isin(labels, train_classes)]
    train_labels = labels[np.isin(labels, train_classes)]
    test_feats = features[np.isin(labels, test_classes)]
    test_labels = labels[np.isin(labels, test_classes)]

    # 参数列表
    margin_list = [0.1, 0.2, 0.3, 0.5]
    hidden_list = [512, 1024, 2048, 4096]
    
    results_margin = {}
    results_hidden = {}

    print('Starting Sensitivity Analysis: Triplet Margin')
    for m in margin_list:
        rec = train_and_eval(m, 2048, train_feats, train_labels, test_feats, test_labels, device)
        results_margin[m] = rec
        print(f'Margin: {m} | Recall: {rec:.4f}')

    print('\nStarting Sensitivity Analysis: Hidden Dimension')
    for h in hidden_list:
        rec = train_and_eval(0.2, h, train_feats, train_labels, test_feats, test_labels, device)
        results_hidden[h] = rec
        print(f'Hidden Dim: {h} | Recall: {rec:.4f}')

    # 打印最终报表
    print('\n' + '='*30)
    print('Final Sensitivity Results')
    print('='*30)
    print('Margin sensitivity (Hidden=2048):')
    for m, r in results_margin.items():
        print(f'  m={m:<5}: {r:.4f}')
    
    print('\nDimension sensitivity (Margin=0.2):')
    for h, r in results_hidden.items():
        print(f'  d={h:<5}: {r:.4f}')

if __name__ == '__main__':
    main()


# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/11_sensitivity.py 
# Starting Sensitivity Analysis: Triplet Margin
# Margin: 0.1 | Recall: 0.8420
# Margin: 0.2 | Recall: 0.8180
# Margin: 0.3 | Recall: 0.8160
# Margin: 0.5 | Recall: 0.7400

# Starting Sensitivity Analysis: Hidden Dimension
# Hidden Dim: 512 | Recall: 0.8300
# Hidden Dim: 1024 | Recall: 0.8340
# Hidden Dim: 2048 | Recall: 0.8200
# Hidden Dim: 4096 | Recall: 0.8240

# ==============================
# Final Sensitivity Results
# ==============================
# Margin sensitivity (Hidden=2048):
#   m=0.1  : 0.8420
#   m=0.2  : 0.8180
#   m=0.3  : 0.8160
#   m=0.5  : 0.7400

# Dimension sensitivity (Margin=0.2):
#   d=512  : 0.8300
#   d=1024 : 0.8340
#   d=2048 : 0.8200
#   d=4096 : 0.8240
# (venv) cc@uc-a100:~/hpdic/EGA$ 

