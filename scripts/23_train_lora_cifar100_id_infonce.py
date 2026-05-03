# scripts/22_train_lora_cifar100_id_infonce.py
# CIFAR-100 ID 测试专用脚本（LoRA + InfoNCE + capacity-matched）
# 专门用于防御 reviewer 对 LoRA 对照不公平的批评

import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import faiss
import argparse


class LoRAAdapter(nn.Module):
    def __init__(self, dim=512, rank=128, alpha=16):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T
        out = x + self.scale * delta
        return F.normalize(out, p=2, dim=1)


class ClassAwareDataset(Dataset):
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
        a = self.features[idx]
        a_label = self.labels[idx]
        p_idx = np.random.choice(self.label_to_indices[a_label])
        n_label = np.random.choice(self.classes)
        while n_label == a_label:
            n_label = np.random.choice(self.classes)
        n_idx = np.random.choice(self.label_to_indices[n_label])
        return a, self.features[p_idx], self.features[n_idx]


class StandardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SupConInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.t
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        eye = torch.eye(features.shape[0], device=features.device)
        mask = mask - eye
        exp_sim = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        pos_count = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()


def train_lora_infonce(train_feats, train_labels, device, dim, rank=128, epochs=120):
    print(f'  Training LoRA + InfoNCE (rank={rank}) ...')
    loader = DataLoader(StandardDataset(train_feats, train_labels),
                        batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    
    model = LoRAAdapter(dim=dim, rank=rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SupConInfoNCE().to(device)

    model.train()
    for epoch in range(epochs):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'    epoch {epoch+1}/{epochs}')
    return model


def eval_lp_ar(features, labels, name):
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    split = int(len(features) * 0.75)
    base = features[:split]
    base_labels = labels[:split]
    query = features[split:]
    query_labels = labels[split:]
    dim = base.shape[1]

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = 1
    _, ret = ivf.search(query, 1)

    correct = 0
    for i in range(len(query_labels)):
        if base_labels[ret[i, 0]] == query_labels[i]:
            correct += 1
    lp = correct / len(query_labels)

    print(f'  {name:25} → LP@1 = {lp:.4f}  (nprobe=1)')
    return lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    features = np.load(os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')).astype(np.float32)
    labels = np.load(os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy'))

    split = int(len(features) * 0.75)
    train_feats = features[:split]
    train_labels = labels[:split]
    test_feats = features[split:]
    test_labels = labels[split:]

    print(f"CIFAR-100 ID Test (75/25 split)")
    print(f"  Train samples: {len(train_labels)} | Test samples: {len(test_labels)}")
    print(f"  LoRA rank = {args.rank}\n")

    dim = features.shape[1]

    # LoRA + InfoNCE (最重要配置)
    model = train_lora_infonce(train_feats, train_labels, device, dim, rank=args.rank)
    model.eval()
    with torch.no_grad():
        feats = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    eval_lp_ar(feats, test_labels, f"LoRA+InfoNCE (r={args.rank})")

    print("\nDone.")


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/23_train_lora_cifar100_id_infonce.py --rank 64
# CIFAR-100 ID Test (75/25 split)
#   Train samples: 7500 | Test samples: 2500
#   LoRA rank = 64

#   Training LoRA + InfoNCE (rank=64) ...
#     epoch 30/120
#     epoch 60/120
#     epoch 90/120
#     epoch 120/120
# WARNING clustering 1875 points to 100 centroids: please provide at least 3900 training points
#   LoRA+InfoNCE (r=64)       → LP@1 = 0.6256  (nprobe=1)

# Done.
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/23_train_lora_cifar100_id_inornce.py --rank 128
# python: can't open file '/home/cc/hpdic/EGA/scripts/23_train_lora_cifar100_id_inornce.py': [Errno 2] No such file or directory
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/23_train_lora_cifar100_id_infonce.py --rank 128
# CIFAR-100 ID Test (75/25 split)
#   Train samples: 7500 | Test samples: 2500
#   LoRA rank = 128

#   Training LoRA + InfoNCE (rank=128) ...
#     epoch 30/120
#     epoch 60/120
#     epoch 90/120
#     epoch 120/120
# WARNING clustering 1875 points to 100 centroids: please provide at least 3900 training points
#   LoRA+InfoNCE (r=128)      → LP@1 = 0.6176  (nprobe=1)

# Done.
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 

