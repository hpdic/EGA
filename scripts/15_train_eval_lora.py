# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# LoRA baseline for retrieval adapter on top of frozen CLIP features.
#
# Trains TWO LoRA configurations on each OOD dataset (Aircraft, Food-101, CIFAR-10):
#   Config A: LoRA + Triplet  (same loss as EGA, different architecture)
#   Config B: LoRA + InfoNCE  (standard LoRA practice, global contrastive)
#
# Saves transformed features and reports LP@1 / AR@1.
#
# Usage:
#   python scripts/train_eval_lora.py
#
# Requires: torch, faiss-cpu, numpy. No external LoRA library — implemented inline.

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
# LoRA adapter
# ─────────────────────────────────────────────

class LoRAAdapter(nn.Module):
    """
    Pure LoRA adapter: y = x + (B @ A)(x) / r * alpha,
    where A: (r, d), B: (d, r). A is initialized Kaiming-normal, B is zero.
    Acts as identity at initialization (residual + zero-init), like EGA's
    high-level design pattern but without the gating/MLP layers.
    """
    def __init__(self, dim=512, rank=8, alpha=16):
        super().__init__()
        self.rank   = rank
        self.scale  = alpha / rank
        # Kaiming-normal init on A; zero init on B → BA = 0 at start
        self.A = nn.Parameter(torch.empty(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

    def forward(self, x):
        # x: (batch, dim)
        delta = (x @ self.A.T) @ self.B.T          # (batch, dim)
        out   = x + self.scale * delta
        return F.normalize(out, p=2, dim=1)        # keep on hypersphere


# ─────────────────────────────────────────────
# Reproducibility & class split
# ─────────────────────────────────────────────

def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_class(features, labels, train_ratio=0.8):
    unique_classes = np.unique(labels)
    np.random.seed(42)
    np.random.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes  = unique_classes[num_train:]
    train_mask = np.isin(labels, train_classes)
    test_mask  = np.isin(labels, test_classes)
    return (features[train_mask], labels[train_mask]), \
           (features[test_mask],  labels[test_mask])


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class ClassAwareDataset(Dataset):
    """For triplet loss: returns (anchor, positive, negative)."""
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
    """For supervised contrastive: returns (feature, label)."""
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
    """
    Standard supervised contrastive (InfoNCE-style) loss for LoRA + global config.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim      = torch.matmul(features, features.T) / self.t
        labels   = labels.contiguous().view(-1, 1)
        mask     = torch.eq(labels, labels.T).float().to(features.device)
        # Remove self-similarity from positives
        eye  = torch.eye(features.shape[0], device=features.device)
        mask = mask - eye

        exp_sim = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        # Mean log-prob over positives
        pos_count = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()


# ─────────────────────────────────────────────
# Training routines
# ─────────────────────────────────────────────

def train_lora_triplet(train_feats, train_labels, device, dim,
                       epochs=150, batch_size=256, margin=0.2, rank=8):
    """Config A: LoRA + Triplet loss."""
    loader    = DataLoader(ClassAwareDataset(train_feats, train_labels),
                           batch_size=batch_size, shuffle=True)
    model     = LoRAAdapter(dim=dim, rank=rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

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


def train_lora_infonce(train_feats, train_labels, device, dim,
                       epochs=150, batch_size=256, rank=8):
    """Config B: LoRA + supervised InfoNCE loss."""
    loader    = DataLoader(StandardDataset(train_feats, train_labels),
                           batch_size=batch_size, shuffle=True)
    model     = LoRAAdapter(dim=dim, rank=rank).to(device)
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


# ─────────────────────────────────────────────
# Evaluation (matches existing scripts)
# ─────────────────────────────────────────────

def calculate_anns_recall(retrieved, gt, k):
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)


def calculate_label_precision(retrieved, base_labels, query_labels, k):
    count = 0
    for i in range(len(query_labels)):
        neighbor_labels = base_labels[retrieved[i, :k]]
        count += np.sum(neighbor_labels == query_labels[i])
    return count / (len(query_labels) * k)


def eval_method(features, labels, dataset_name, method_name,
                k=1, nlist=10, nprobe=1, seed=42):
    """75/25 split, returns (LP@k, AR@k) at given nprobe."""
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
    _, gt = exact.search(query, k)

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = nprobe
    _, ret = ivf.search(query, k)

    lp = calculate_label_precision(ret, base_labels, query_labels, k)
    ar = calculate_anns_recall(ret, gt, k)
    return lp, ar


# ─────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────

def run_for_dataset(name, feat_path, label_path, device,
                    needs_class_split=True, epochs=150):
    print(f'\n{"=" * 60}')
    print(f'  Dataset: {name}')
    print(f'{"=" * 60}')

    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        print(f'  Files missing for {name}; skipping.')
        return None

    features = np.load(feat_path).astype(np.float32)
    labels   = np.load(label_path)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    if needs_class_split:
        (train_feats, train_labels), (test_feats, test_labels) = \
            split_by_class(features, labels)
        print(f'  Train classes: {len(np.unique(train_labels))}, '
              f'samples: {len(train_labels)}')
        print(f'  Unseen classes: {len(np.unique(test_labels))}, '
              f'samples: {len(test_labels)}')
    else:
        # CIFAR-10: trained on CIFAR-100, eval on full CIFAR-10
        # → no in-dataset split for training, but we still split test for retrieval
        train_feats, train_labels = None, None
        test_feats,  test_labels  = features, labels
        print(f'  CIFAR-10: using all {len(labels)} samples as test')

    dim = features.shape[1]

    # CIFAR-10 special case: use CIFAR-100 features for training
    if not needs_class_split:
        c100_feat  = os.path.expanduser('~/hpdic/EGA/embeddings/cifar100_vit_b32_features.npy')
        c100_label = os.path.expanduser('~/hpdic/EGA/embeddings/cifar100_vit_b32_labels.npy')
        train_feats  = np.load(c100_feat).astype(np.float32)
        train_feats  = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
        train_labels = np.load(c100_label)
        print(f'  Training on CIFAR-100: {len(train_labels)} samples, '
              f'{len(np.unique(train_labels))} classes')

    results = {}

    # ── Config A: LoRA + Triplet ──────────────────────────────────────
    print(f'\n  [A] Training LoRA + Triplet ...')
    set_global_seed(42)
    model_a = train_lora_triplet(train_feats, train_labels, device, dim, epochs=epochs)
    model_a.eval()
    with torch.no_grad():
        feats_a = model_a(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_a, ar_a = eval_method(feats_a, test_labels, name, 'LoRA+Triplet')
    print(f'    LP@1 = {lp_a:.4f}  AR@1 = {ar_a:.4f}')
    results['LoRA+Triplet'] = (lp_a, ar_a)

    # ── Config B: LoRA + InfoNCE ──────────────────────────────────────
    print(f'\n  [B] Training LoRA + InfoNCE ...')
    set_global_seed(42)
    model_b = train_lora_infonce(train_feats, train_labels, device, dim, epochs=epochs)
    model_b.eval()
    with torch.no_grad():
        feats_b = model_b(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_b, ar_b = eval_method(feats_b, test_labels, name, 'LoRA+InfoNCE')
    print(f'    LP@1 = {lp_b:.4f}  AR@1 = {ar_b:.4f}')
    results['LoRA+InfoNCE'] = (lp_b, ar_b)

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    set_global_seed(42)
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    base_dir  = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    all_results = {}

    # ── FGVC-Aircraft (80 seen / 20 unseen split) ────────────────────
    res = run_for_dataset(
        name='FGVC-Aircraft',
        feat_path=os.path.join(embed_dir, 'aircraft_test_vit_b32_features.npy'),
        label_path=os.path.join(embed_dir, 'aircraft_test_vit_b32_labels.npy'),
        device=device,
        needs_class_split=True,
    )
    if res: all_results['Aircraft'] = res

    # ── Food-101 (80 seen / 21 unseen split) ─────────────────────────
    res = run_for_dataset(
        name='Food-101',
        feat_path=os.path.join(embed_dir, 'food101_features.npy'),
        label_path=os.path.join(embed_dir, 'food101_labels.npy'),
        device=device,
        needs_class_split=True,
    )
    if res: all_results['Food-101'] = res

    # ── CIFAR-10 (cross-dataset OOD: train on CIFAR-100, eval on CIFAR-10) ─
    res = run_for_dataset(
        name='CIFAR-10',
        feat_path=os.path.join(embed_dir, 'cifar10_vit_b32_features.npy'),
        label_path=os.path.join(embed_dir, 'cifar10_vit_b32_labels.npy'),
        device=device,
        needs_class_split=False,
    )
    if res: all_results['CIFAR-10'] = res

    # ── Final summary ────────────────────────────────────────────────
    print('\n')
    print('=' * 60)
    print('  Final Summary  (K=1, nprobe=1)')
    print('=' * 60)
    print(f'  {"Dataset":<14} | {"Config":<14} | {"LP@1":>7} | {"AR@1":>7}')
    print(f'  {"-"*14}-+-{"-"*14}-+-{"-"*7}-+-{"-"*7}')
    for ds, methods in all_results.items():
        for method, (lp, ar) in methods.items():
            print(f'  {ds:<14} | {method:<14} | {lp:>7.4f} | {ar:>7.4f}')

if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/15_train_eval_lora.py 
# Device: cuda

# ============================================================
#   Dataset: FGVC-Aircraft
# ============================================================
#   Train classes: 80, samples: 2664
#   Unseen classes: 20, samples: 669

#   [A] Training LoRA + Triplet ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
#     LP@1 = 0.5774  AR@1 = 0.8810

#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
#     LP@1 = 0.5595  AR@1 = 0.9107

# ============================================================
#   Dataset: Food-101
# ============================================================
#   Train classes: 80, samples: 20000
#   Unseen classes: 21, samples: 5250

#   [A] Training LoRA + Triplet ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
#     LP@1 = 0.8332  AR@1 = 0.8987

#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
#     LP@1 = 0.8781  AR@1 = 0.9018

# ============================================================
#   Dataset: CIFAR-10
# ============================================================
#   CIFAR-10: using all 10000 samples as test
#   Training on CIFAR-100: 10000 samples, 100 classes

#   [A] Training LoRA + Triplet ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
#     LP@1 = 0.8696  AR@1 = 0.8664

#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
#     LP@1 = 0.8796  AR@1 = 0.8564


# ============================================================
#   Final Summary  (K=1, nprobe=1)
# ============================================================
#   Dataset        | Config         |    LP@1 |    AR@1
#   ---------------+----------------+---------+--------
#   Aircraft       | LoRA+Triplet   |  0.5774 |  0.8810
#   Aircraft       | LoRA+InfoNCE   |  0.5595 |  0.9107
#   Food-101       | LoRA+Triplet   |  0.8332 |  0.8987
#   Food-101       | LoRA+InfoNCE   |  0.8781 |  0.9018
#   CIFAR-10       | LoRA+Triplet   |  0.8696 |  0.8664
#   CIFAR-10       | LoRA+InfoNCE   |  0.8796 |  0.8564
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 
