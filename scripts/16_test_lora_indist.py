# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# In-distribution test of LoRA on CIFAR-100.
# This is the go/no-go experiment: if LoRA matches or beats EGA on
# CIFAR-100 in-distribution, the paper's "EGA is the only ID+OOD-friendly
# adapter" positioning collapses.
#
# Usage:
#   python scripts/test_lora_indist.py

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


# ─────────────────────────────────────────────
# LoRA adapter (same as scripts/train_eval_lora.py)
# ─────────────────────────────────────────────

class LoRAAdapter(nn.Module):
    def __init__(self, dim=512, rank=8, alpha=16):
        super().__init__()
        self.rank   = rank
        self.scale  = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T
        out   = x + self.scale * delta
        return F.normalize(out, p=2, dim=1)


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class ClassAwareDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels   = labels
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
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────

class SupConInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim      = torch.matmul(features, features.T) / self.t
        labels   = labels.contiguous().view(-1, 1)
        mask     = torch.eq(labels, labels.T).float().to(features.device)
        eye  = torch.eye(features.shape[0], device=features.device)
        mask = mask - eye
        exp_sim = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        pos_count = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()


# ─────────────────────────────────────────────
# Train routines
# ─────────────────────────────────────────────

def train_lora_triplet(features, labels, device, dim, epochs=150):
    loader    = DataLoader(ClassAwareDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = LoRAAdapter(dim=dim, rank=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    model.train()
    for epoch in range(epochs):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'    epoch {epoch+1}/{epochs}')
    return model


def train_lora_infonce(features, labels, device, dim, epochs=150):
    loader    = DataLoader(StandardDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = LoRAAdapter(dim=dim, rank=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SupConInfoNCE().to(device)

    model.train()
    for epoch in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'    epoch {epoch+1}/{epochs}')
    return model


# ─────────────────────────────────────────────
# Eval (in-distribution: train AND eval on CIFAR-100, 75/25 split)
# ─────────────────────────────────────────────

def calculate_label_precision(retrieved, base_labels, query_labels, k):
    count = 0
    for i in range(len(query_labels)):
        neighbor_labels = base_labels[retrieved[i, :k]]
        count += np.sum(neighbor_labels == query_labels[i])
    return count / (len(query_labels) * k)


def calculate_anns_recall(retrieved, gt, k):
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)


def eval_indist(features, labels, label_text, k_list=(1, 3, 5, 10),
                nprobe_list=(1, 5, 10), seed=42):
    """75/25 split for in-distribution retrieval eval."""
    print(f'\nEvaluating: {label_text}')
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    np.random.seed(seed)
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels   = labels[idx]

    split        = int(len(features) * 0.75)
    base         = features[:split]
    base_labels  = labels[:split]
    query        = features[split:]
    query_labels = labels[split:]
    dim          = base.shape[1]

    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    nlist = 10
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f'  K={k}')
        for np_ in nprobe_list:
            ivf.nprobe = np_
            _, ret = ivf.search(query, k)
            lp = calculate_label_precision(ret, base_labels, query_labels, k)
            ar = calculate_anns_recall(ret, gt, k)
            print(f'    nprobe={np_:>2}: LP={lp:.4f}  AR={ar:.4f}')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    base_dir  = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    feat_path  = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    label_path = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels   = np.load(label_path)

    print(f'\nCIFAR-100 in-distribution test')
    print(f'Total samples: {len(labels)}, classes: {len(np.unique(labels))}')

    dim = features.shape[1]

    # ── Baseline: raw CLIP ──────────────────────────────────────────
    eval_indist(features, labels, 'Raw CLIP (frozen baseline)')

    # ── LoRA + Triplet ──────────────────────────────────────────────
    print('\n[1/2] Training LoRA + Triplet on full CIFAR-100 ...')
    np.random.seed(42); torch.manual_seed(42)
    model_a = train_lora_triplet(features, labels, device, dim)
    model_a.eval()
    with torch.no_grad():
        out_a = model_a(torch.from_numpy(features).float().to(device)).cpu().numpy()
    eval_indist(out_a, labels, 'LoRA + Triplet (in-distribution)')

    # ── LoRA + InfoNCE ──────────────────────────────────────────────
    print('\n[2/2] Training LoRA + InfoNCE on full CIFAR-100 ...')
    np.random.seed(42); torch.manual_seed(42)
    model_b = train_lora_infonce(features, labels, device, dim)
    model_b.eval()
    with torch.no_grad():
        out_b = model_b(torch.from_numpy(features).float().to(device)).cpu().numpy()
    eval_indist(out_b, labels, 'LoRA + InfoNCE (in-distribution)')

    print('\n')
    print('=' * 60)
    print('  Reference: EGA on CIFAR-100 reaches LP@1, nprobe=1 ~ 0.70')
    print('  GO if LoRA both LP@1, nprobe=1 < ~0.65')
    print('  NO-GO if LoRA matches or beats EGA')
    print('=' * 60)


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/16_test_lora_indist.py 
# Device: cuda

# CIFAR-100 in-distribution test
# Total samples: 10000, classes: 100

# Evaluating: Raw CLIP (frozen baseline)
#   K=1
#     nprobe= 1: LP=0.5488  AR=0.8052
#     nprobe= 5: LP=0.5728  AR=0.9948
#     nprobe=10: LP=0.5728  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.5071  AR=0.7840
#     nprobe= 5: LP=0.5348  AR=0.9935
#     nprobe=10: LP=0.5344  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.4797  AR=0.7744
#     nprobe= 5: LP=0.5121  AR=0.9933
#     nprobe=10: LP=0.5118  AR=0.9999
#   K=10
#     nprobe= 1: LP=0.4353  AR=0.7604
#     nprobe= 5: LP=0.4712  AR=0.9913
#     nprobe=10: LP=0.4718  AR=1.0000

# [1/2] Training LoRA + Triplet on full CIFAR-100 ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150

# Evaluating: LoRA + Triplet (in-distribution)
#   K=1
#     nprobe= 1: LP=0.6120  AR=0.9084
#     nprobe= 5: LP=0.6232  AR=0.9996
#     nprobe=10: LP=0.6232  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.5824  AR=0.9064
#     nprobe= 5: LP=0.5943  AR=0.9996
#     nprobe=10: LP=0.5943  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.5627  AR=0.9033
#     nprobe= 5: LP=0.5762  AR=0.9997
#     nprobe=10: LP=0.5762  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.5342  AR=0.8979
#     nprobe= 5: LP=0.5504  AR=0.9994
#     nprobe=10: LP=0.5504  AR=1.0000

# [2/2] Training LoRA + InfoNCE on full CIFAR-100 ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150

# Evaluating: LoRA + InfoNCE (in-distribution)
#   K=1
#     nprobe= 1: LP=0.6684  AR=0.8792
#     nprobe= 5: LP=0.6716  AR=0.9992
#     nprobe=10: LP=0.6712  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.6427  AR=0.8756
#     nprobe= 5: LP=0.6543  AR=0.9993
#     nprobe=10: LP=0.6545  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.6251  AR=0.8718
#     nprobe= 5: LP=0.6381  AR=0.9991
#     nprobe=10: LP=0.6382  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.5970  AR=0.8650
#     nprobe= 5: LP=0.6149  AR=0.9988
#     nprobe=10: LP=0.6152  AR=1.0000


# ============================================================
#   Reference: EGA on CIFAR-100 reaches LP@1, nprobe=1 ~ 0.70
#   GO if LoRA both LP@1, nprobe=1 < ~0.65
#   NO-GO if LoRA matches or beats EGA
# ============================================================
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 