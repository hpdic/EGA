# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# In-distribution test of ICon and SRL on CIFAR-100.
# Completes the in-distribution comparison matrix:
#   {CLIP, EGA, ICon, SRL, LoRA+Triplet, LoRA+InfoNCE} × {LP, AR}
#
# Usage:
#   python scripts/test_icon_srl_indist.py

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

from models.ega_mlp import EGAMLP


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class StandardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ─────────────────────────────────────────────
# Losses (same as Aircraft / Food-101 scripts)
# ─────────────────────────────────────────────

class IConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features  = F.normalize(features, p=2, dim=1)
        sim       = torch.matmul(features, features.T) / self.temperature
        labels    = labels.contiguous().view(-1, 1)
        mask      = torch.eq(labels, labels.T).float().to(features.device)
        log_probs = F.log_softmax(sim, dim=1)
        target    = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return F.kl_div(log_probs, target, reduction='batchmean')


class SRLLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_gen=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_gen  = lambda_gen

    def forward(self, features, labels):
        features   = F.normalize(features, p=2, dim=1)
        bs         = features.shape[0]
        logits     = torch.matmul(features, features.T) / self.temperature
        mask       = torch.eq(labels.view(-1, 1),
                              labels.view(1, -1)).float().to(features.device)
        exp_logits = torch.exp(logits) * (1 - torch.eye(bs).to(features.device))
        uniformity = torch.log(exp_logits.sum(dim=1)).mean()
        pos_logits = (exp_logits * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        homog      = -torch.log(pos_logits + 1e-8).mean()
        return homog + self.lambda_gen * uniformity


# ─────────────────────────────────────────────
# Train routine
# ─────────────────────────────────────────────

def train_contrastive(features, labels, loss_fn, device, dim, name, epochs=150):
    loader    = DataLoader(StandardDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'    {name} epoch {epoch+1}/{epochs}')
    return model


# ─────────────────────────────────────────────
# Eval (75/25 split, in-distribution)
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

    # ── ICon ──────────────────────────────────────────────────
    print('\n[1/2] Training ICon on full CIFAR-100 ...')
    np.random.seed(42); torch.manual_seed(42)
    model_icon = train_contrastive(features, labels,
                                   IConLoss().to(device), device, dim, 'ICon')
    model_icon.eval()
    with torch.no_grad():
        out_icon = model_icon(torch.from_numpy(features).float().to(device)).cpu().numpy()
    eval_indist(out_icon, labels, 'ICon (in-distribution)')

    # ── SRL ──────────────────────────────────────────────────
    print('\n[2/2] Training SRL on full CIFAR-100 ...')
    np.random.seed(42); torch.manual_seed(42)
    model_srl = train_contrastive(features, labels,
                                  SRLLoss().to(device), device, dim, 'SRL')
    model_srl.eval()
    with torch.no_grad():
        out_srl = model_srl(torch.from_numpy(features).float().to(device)).cpu().numpy()
    eval_indist(out_srl, labels, 'SRL (in-distribution)')

    print('\n')
    print('=' * 60)
    print('  Reference (CIFAR-100, K=1, nprobe=1):')
    print('    CLIP          : LP=0.549')
    print('    EGA           : LP~0.70 (paper claim)')
    print('    LoRA+Triplet  : LP=0.612')
    print('    LoRA+InfoNCE  : LP=0.668')
    print('    ICon          : LP=??? (this run)')
    print('    SRL           : LP=??? (this run)')
    print('=' * 60)


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/17_test_icon_srl_indist.py 
# Device: cuda

# CIFAR-100 in-distribution test
# Total samples: 10000, classes: 100

# [1/2] Training ICon on full CIFAR-100 ...
#     ICon epoch 30/150
#     ICon epoch 60/150
#     ICon epoch 90/150
#     ICon epoch 120/150
#     ICon epoch 150/150

# Evaluating: ICon (in-distribution)
#   K=1
#     nprobe= 1: LP=0.9988  AR=0.9916
#     nprobe= 5: LP=0.9992  AR=0.9996
#     nprobe=10: LP=0.9992  AR=0.9996
#   K=3
#     nprobe= 1: LP=0.9987  AR=0.9907
#     nprobe= 5: LP=0.9995  AR=1.0000
#     nprobe=10: LP=0.9995  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.9986  AR=0.9901
#     nprobe= 5: LP=0.9994  AR=1.0000
#     nprobe=10: LP=0.9994  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.9981  AR=0.9887
#     nprobe= 5: LP=0.9994  AR=0.9999
#     nprobe=10: LP=0.9994  AR=0.9999

# [2/2] Training SRL on full CIFAR-100 ...
#     SRL epoch 30/150
#     SRL epoch 60/150
#     SRL epoch 90/150
#     SRL epoch 120/150
#     SRL epoch 150/150

# Evaluating: SRL (in-distribution)
#   K=1
#     nprobe= 1: LP=0.9916  AR=0.9908
#     nprobe= 5: LP=0.9928  AR=1.0000
#     nprobe=10: LP=0.9928  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.9905  AR=0.9895
#     nprobe= 5: LP=0.9928  AR=1.0000
#     nprobe=10: LP=0.9928  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.9894  AR=0.9891
#     nprobe= 5: LP=0.9926  AR=1.0000
#     nprobe=10: LP=0.9926  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.9888  AR=0.9888
#     nprobe= 5: LP=0.9928  AR=1.0000
#     nprobe=10: LP=0.9928  AR=1.0000


# ============================================================
#   Reference (CIFAR-100, K=1, nprobe=1):
#     CLIP          : LP=0.549
#     EGA           : LP~0.70 (paper claim)
#     LoRA+Triplet  : LP=0.612
#     LoRA+InfoNCE  : LP=0.668
#     ICon          : LP=??? (this run)
#     SRL           : LP=??? (this run)
# ============================================================
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 