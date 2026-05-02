# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# 3-seed evaluation of CIFAR-10 OOD retrieval (trained on CIFAR-100)
# across all 6 methods to produce mean +- std error bars.
#
# Methods:
#   - Raw CLIP (no training; just eval-side seed varies for index sampling)
#   - EGA
#   - ICon
#   - SRL
#   - LoRA + Triplet
#   - LoRA + InfoNCE
#
# Each trainable method is trained 3 times with different seeds (42, 123, 456),
# evaluated on CIFAR-10 at K=1, nprobe=1.
#
# Usage:
#   python scripts/19_cifar10_3seed.py

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
# LoRA adapter
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

class IConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        features  = F.normalize(features, p=2, dim=1)
        sim       = torch.matmul(features, features.T) / self.t
        labels    = labels.contiguous().view(-1, 1)
        mask      = torch.eq(labels, labels.T).float().to(features.device)
        log_probs = F.log_softmax(sim, dim=1)
        target    = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return F.kl_div(log_probs, target, reduction='batchmean')


class SRLLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_gen=0.5):
        super().__init__()
        self.t   = temperature
        self.lam = lambda_gen

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        bs       = features.shape[0]
        logits   = torch.matmul(features, features.T) / self.t
        mask     = torch.eq(labels.view(-1, 1),
                            labels.view(1, -1)).float().to(features.device)
        exp_logits = torch.exp(logits) * (1 - torch.eye(bs).to(features.device))
        unif     = torch.log(exp_logits.sum(dim=1)).mean()
        pos      = (exp_logits * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        homog    = -torch.log(pos + 1e-8).mean()
        return homog + self.lam * unif


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
# Training routines
# ─────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ega(features, labels, device, dim, seed, epochs=150):
    set_seed(seed)
    loader    = DataLoader(ClassAwareDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    model.train()
    for _ in range(epochs):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def train_icon_or_srl(features, labels, loss_fn, device, dim, seed, epochs=150):
    set_seed(seed)
    loader    = DataLoader(StandardDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for _ in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def train_lora_triplet(features, labels, device, dim, seed, epochs=150):
    set_seed(seed)
    loader    = DataLoader(ClassAwareDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = LoRAAdapter(dim=dim, rank=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    model.train()
    for _ in range(epochs):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def train_lora_infonce(features, labels, device, dim, seed, epochs=150):
    set_seed(seed)
    loader    = DataLoader(StandardDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = LoRAAdapter(dim=dim, rank=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SupConInfoNCE().to(device)
    model.train()
    for _ in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


# ─────────────────────────────────────────────
# Eval (CIFAR-10 OOD: train on CIFAR-100, eval on CIFAR-10)
# ─────────────────────────────────────────────

def eval_lp_ar(features, labels, k=1, nprobe=1, nlist=10, eval_seed=42):
    """75/25 split eval; returns (LP@k, AR@k)."""
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    np.random.seed(eval_seed)
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

    lp_count = 0
    for i in range(len(query_labels)):
        lp_count += np.sum(base_labels[ret[i, :k]] == query_labels[i])
    lp = lp_count / (len(query_labels) * k)

    ar_count = 0
    for i in range(len(gt)):
        ar_count += len(np.intersect1d(ret[i, :k], gt[i, :k]))
    ar = ar_count / (len(gt) * k)

    return lp, ar


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    base_dir  = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    # Train on CIFAR-100, eval on CIFAR-10
    train_feats  = np.load(os.path.join(embed_dir,
                       'cifar100_vit_b32_features.npy')).astype(np.float32)
    train_feats  = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
    train_labels = np.load(os.path.join(embed_dir,
                       'cifar100_vit_b32_labels.npy'))

    eval_feats   = np.load(os.path.join(embed_dir,
                       'cifar10_vit_b32_features.npy')).astype(np.float32)
    eval_feats   = eval_feats / np.linalg.norm(eval_feats, axis=1, keepdims=True)
    eval_labels  = np.load(os.path.join(embed_dir,
                       'cifar10_vit_b32_labels.npy'))

    print(f'Train (CIFAR-100): {len(train_labels)} samples, '
          f'{len(np.unique(train_labels))} classes')
    print(f'Eval  (CIFAR-10) : {len(eval_labels)} samples, '
          f'{len(np.unique(eval_labels))} classes')

    dim   = train_feats.shape[1]
    seeds = [42, 123, 456]

    # Storage: results[method] = {'lp': [seed1, seed2, seed3], 'ar': [...]}
    results = collections.defaultdict(lambda: {'lp': [], 'ar': []})

    # ── Raw CLIP (no training; same numbers across seeds) ────────────
    print('\n[1/6] Raw CLIP (deterministic, single eval) ...')
    lp, ar = eval_lp_ar(eval_feats, eval_labels)
    for s in seeds:
        results['CLIP']['lp'].append(lp)
        results['CLIP']['ar'].append(ar)
    print(f'    LP={lp:.4f}  AR={ar:.4f} (replicated for 3 seeds)')

    # ── EGA × 3 seeds ────────────────────────────────────────────────
    for i, s in enumerate(seeds):
        print(f'\n[2/6] EGA seed={s} ({i+1}/3) ...')
        model = train_ega(train_feats, train_labels, device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(eval_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, eval_labels)
        results['EGA']['lp'].append(lp)
        results['EGA']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # ── ICon × 3 seeds ───────────────────────────────────────────────
    for i, s in enumerate(seeds):
        print(f'\n[3/6] ICon seed={s} ({i+1}/3) ...')
        model = train_icon_or_srl(train_feats, train_labels,
                                  IConLoss().to(device), device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(eval_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, eval_labels)
        results['ICon']['lp'].append(lp)
        results['ICon']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # ── SRL × 3 seeds ────────────────────────────────────────────────
    for i, s in enumerate(seeds):
        print(f'\n[4/6] SRL seed={s} ({i+1}/3) ...')
        model = train_icon_or_srl(train_feats, train_labels,
                                  SRLLoss().to(device), device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(eval_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, eval_labels)
        results['SRL']['lp'].append(lp)
        results['SRL']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # ── LoRA + Triplet × 3 seeds ─────────────────────────────────────
    for i, s in enumerate(seeds):
        print(f'\n[5/6] LoRA+Triplet seed={s} ({i+1}/3) ...')
        model = train_lora_triplet(train_feats, train_labels, device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(eval_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, eval_labels)
        results['LoRA+Triplet']['lp'].append(lp)
        results['LoRA+Triplet']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # ── LoRA + InfoNCE × 3 seeds ─────────────────────────────────────
    for i, s in enumerate(seeds):
        print(f'\n[6/6] LoRA+InfoNCE seed={s} ({i+1}/3) ...')
        model = train_lora_infonce(train_feats, train_labels, device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(eval_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, eval_labels)
        results['LoRA+InfoNCE']['lp'].append(lp)
        results['LoRA+InfoNCE']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # ── Final summary ────────────────────────────────────────────────
    print('\n')
    print('=' * 70)
    print('  CIFAR-10 OOD Final Summary  (K=1, nprobe=1, mean ± std over 3 seeds)')
    print('=' * 70)
    print(f'  {"Method":<14} | {"LP@1 (mean ± std)":>22} | {"AR@1 (mean ± std)":>22}')
    print(f'  {"-"*14}-+-{"-"*22}-+-{"-"*22}')

    method_order = ['CLIP', 'ICon', 'SRL', 'LoRA+InfoNCE', 'LoRA+Triplet', 'EGA']
    for m in method_order:
        lps = np.array(results[m]['lp'])
        ars = np.array(results[m]['ar'])
        lp_str = f'{lps.mean():.4f} ± {lps.std():.4f}'
        ar_str = f'{ars.mean():.4f} ± {ars.std():.4f}'
        print(f'  {m:<14} | {lp_str:>22} | {ar_str:>22}')

    # Save raw numbers for paper
    save_path = os.path.join(base_dir, 'cifar10_3seed_results.txt')
    with open(save_path, 'w') as f:
        f.write('CIFAR-10 OOD 3-seed results (seeds: 42, 123, 456)\n')
        f.write('=' * 70 + '\n')
        for m in method_order:
            lps = results[m]['lp']
            ars = results[m]['ar']
            f.write(f'{m}: LP={lps}  AR={ars}\n')
            f.write(f'  mean LP = {np.mean(lps):.4f} ± {np.std(lps):.4f}\n')
            f.write(f'  mean AR = {np.mean(ars):.4f} ± {np.std(ars):.4f}\n\n')
    print(f'\nDetailed results saved to: {save_path}')


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/19_cifar10_3seed.py 
# Device: cuda
# Train (CIFAR-100): 10000 samples, 100 classes
# Eval  (CIFAR-10) : 10000 samples, 10 classes

# [1/6] Raw CLIP (deterministic, single eval) ...
#     LP=0.8804  AR=0.8632 (replicated for 3 seeds)

# [2/6] EGA seed=42 (1/3) ...
#     LP=0.8124  AR=0.8384

# [2/6] EGA seed=123 (2/3) ...
#     LP=0.8076  AR=0.8264

# [2/6] EGA seed=456 (3/3) ...
#     LP=0.8088  AR=0.8336

# [3/6] ICon seed=42 (1/3) ...
#     LP=0.5720  AR=0.8004

# [3/6] ICon seed=123 (2/3) ...
#     LP=0.5592  AR=0.7928

# [3/6] ICon seed=456 (3/3) ...
#     LP=0.5488  AR=0.7968

# [4/6] SRL seed=42 (1/3) ...
#     LP=0.5476  AR=0.8172

# [4/6] SRL seed=123 (2/3) ...
#     LP=0.5332  AR=0.8200

# [4/6] SRL seed=456 (3/3) ...
#     LP=0.5284  AR=0.8192

# [5/6] LoRA+Triplet seed=42 (1/3) ...
#     LP=0.8696  AR=0.8664

# [5/6] LoRA+Triplet seed=123 (2/3) ...
#     LP=0.8728  AR=0.8448

# [5/6] LoRA+Triplet seed=456 (3/3) ...
#     LP=0.8828  AR=0.8708

# [6/6] LoRA+InfoNCE seed=42 (1/3) ...
#     LP=0.8796  AR=0.8564

# [6/6] LoRA+InfoNCE seed=123 (2/3) ...
#     LP=0.8812  AR=0.8664

# [6/6] LoRA+InfoNCE seed=456 (3/3) ...
#     LP=0.8936  AR=0.8852


# ======================================================================
#   CIFAR-10 OOD Final Summary  (K=1, nprobe=1, mean ± std over 3 seeds)
# ======================================================================
#   Method         |      LP@1 (mean ± std) |      AR@1 (mean ± std)
#   ---------------+------------------------+-----------------------
#   CLIP           |        0.8804 ± 0.0000 |        0.8632 ± 0.0000
#   ICon           |        0.5600 ± 0.0095 |        0.7967 ± 0.0031
#   SRL            |        0.5364 ± 0.0082 |        0.8188 ± 0.0012
#   LoRA+InfoNCE   |        0.8848 ± 0.0063 |        0.8693 ± 0.0119
#   LoRA+Triplet   |        0.8751 ± 0.0056 |        0.8607 ± 0.0114
#   EGA            |        0.8096 ± 0.0020 |        0.8328 ± 0.0049

# Detailed results saved to: /home/cc/hpdic/EGA/cifar10_3seed_results.txt
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 