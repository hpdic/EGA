# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
import faiss

# --- 1. 定义实验变体模型 ---

class AblationMLP(nn.Module):
    def __init__(self, variant='full', input_dim=512, hidden_dim=2048):
        super().__init__()
        self.variant = variant
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 零初始化测试：只有非 'no_zero' 变体才执行零初始化
        if variant != 'no_zero':
            nn.init.zeros_(self.adapter[3].weight)
            nn.init.zeros_(self.adapter[3].bias)

    def forward(self, x):
        # 残差连接测试
        if self.variant == 'no_res':
            out = self.adapter(x)
        else:
            out = x + self.adapter(x)
        
        # L2 归一化测试
        if self.variant != 'no_norm':
            out = F.normalize(out, p=2, dim=1)
        return out

# --- 2. 核心训练与评估逻辑 ---

class SemanticDataset(Dataset):
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

def evaluate(model, features, labels, k=1):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        feat_tensor = torch.from_numpy(features).to(device)
        transformed = model(feat_tensor).cpu().numpy()
    
    # 模拟 02_train_ega 中的切分进行评估
    base = transformed[:8000]
    query = transformed[8000:]
    dim = base.shape[1]
    
    index = faiss.IndexFlatL2(dim)
    index.add(base)
    _, I = index.search(query, k)
    
    # 简单的 Recall@1 逻辑
    # 这里直接用索引匹配，因为 IndexFlatL2 在 120 epoch 后 Recall 通常极高
    # 我们主要看它对流形质量的改观
    return I  # 实际运行中你可以加入更复杂的召回率计算

def run_experiment(variant, train_feats, train_labels, all_feats, all_labels, device):
    model = AblationMLP(variant=variant).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    loader = DataLoader(SemanticDataset(train_feats, train_labels), batch_size=1024, shuffle=True)
    
    for epoch in range(120):
        model.train()
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        all_tensor = torch.from_numpy(all_feats).to(device)
        transformed = model(all_tensor).cpu().numpy()
        
    # 修正这里的作用域：使用传入的 all_labels
    base, query = transformed[:8000], transformed[8000:]
    labels_base, labels_query = all_labels[:8000], all_labels[8000:]
    
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, I = index.search(query, 1)
    
    correct = np.sum(labels_base[I.flatten()] == labels_query)
    return correct / len(query)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    feat_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_features.npy')
    label_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_labels.npy')

    features = np.load(feat_path).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)
    
    train_feats = features[:8000]
    train_labels = labels[:8000]

    variants = ['full', 'no_res', 'no_zero', 'no_norm']
    results = {}

    print("Starting Ablation Study on CIFAR-100 Features...")
    for v in variants:
        acc = run_experiment(v, train_feats, train_labels, features, labels, device)
        results[v] = acc
        print(f"Variant: {v:<10} | Test Accuracy: {acc:.4f}")

    print("\nFinal Results Table:")
    print("-" * 35)
    for v, acc in results.items():
        print(f"{v:<15} | {acc:.4f}")

if __name__ == '__main__':
    main()


# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/10_ablation.py 
# Starting Ablation Study on CIFAR-100 Features...
# Variant: full       | Test Accuracy: 0.7015
# Variant: no_res     | Test Accuracy: 0.0065
# Variant: no_zero    | Test Accuracy: 0.6840
# Variant: no_norm    | Test Accuracy: 0.6590

# Final Results Table:
# -----------------------------------
# full            | 0.7015
# no_res          | 0.0065
# no_zero         | 0.6840
# no_norm         | 0.6590
# (venv) cc@uc-a100:~/hpdic/EGA$ 
