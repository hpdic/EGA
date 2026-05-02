# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# 3-seed evaluation of Aircraft and Food-101 OOD retrieval across all 6 methods
# to produce mean +- std error bars for Table 2.
#
# Methods:
#   - Raw CLIP (no training)
#   - EGA
#   - ICon
#   - SRL
#   - LoRA + Triplet
#   - LoRA + InfoNCE
#
# Each trainable method is trained 3 times with different seeds (42, 123, 456),
# evaluated on the corresponding OOD dataset at K=1, nprobe=1.
#
# Usage:
#   python scripts/20_aircraft_food_3seed.py

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
# Reproducibility & class split
# ─────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_class(features, labels, train_ratio=0.8, split_seed=42):
    """Split into seen/unseen classes. split_seed kept fixed across model seeds
    so the same class partition is used in all 3 runs (only model init varies)."""
    unique_classes = np.unique(labels)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes  = unique_classes[num_train:]
    train_mask = np.isin(labels, train_classes)
    test_mask  = np.isin(labels, test_classes)
    return (features[train_mask], labels[train_mask]), \
           (features[test_mask],  labels[test_mask])


# ─────────────────────────────────────────────
# Training routines
# ─────────────────────────────────────────────

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
# Eval (75/25 split on unseen classes)
# ─────────────────────────────────────────────

def eval_lp_ar(features, labels, k=1, nprobe=1, nlist=10, eval_seed=42):
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
# Per-dataset 3-seed runner
# ─────────────────────────────────────────────

def run_dataset(name, feat_path, label_path, device, seeds):
    """Train all 6 methods × 3 seeds on this dataset (class-split OOD setting)."""
    print(f'\n{"=" * 70}')
    print(f'  Dataset: {name}')
    print(f'{"=" * 70}')

    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        print(f'  Files missing for {name}; skipping.')
        return None

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels   = np.load(label_path)

    (train_feats, train_labels), (test_feats, test_labels) = \
        split_by_class(features, labels)
    print(f'  Train classes: {len(np.unique(train_labels))}, '
          f'samples: {len(train_labels)}')
    print(f'  Unseen classes: {len(np.unique(test_labels))}, '
          f'samples: {len(test_labels)}')

    dim = features.shape[1]
    results = collections.defaultdict(lambda: {'lp': [], 'ar': []})

    # Raw CLIP (deterministic)
    print('\n  [1/6] Raw CLIP (deterministic)')
    lp, ar = eval_lp_ar(test_feats, test_labels)
    for _ in seeds:
        results['CLIP']['lp'].append(lp)
        results['CLIP']['ar'].append(ar)
    print(f'    LP={lp:.4f}  AR={ar:.4f} (replicated for 3 seeds)')

    # EGA
    for i, s in enumerate(seeds):
        print(f'\n  [2/6] EGA seed={s} ({i+1}/3)')
        model = train_ega(train_feats, train_labels, device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, test_labels)
        results['EGA']['lp'].append(lp); results['EGA']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # ICon
    for i, s in enumerate(seeds):
        print(f'\n  [3/6] ICon seed={s} ({i+1}/3)')
        model = train_icon_or_srl(train_feats, train_labels,
                                  IConLoss().to(device), device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, test_labels)
        results['ICon']['lp'].append(lp); results['ICon']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # SRL
    for i, s in enumerate(seeds):
        print(f'\n  [4/6] SRL seed={s} ({i+1}/3)')
        model = train_icon_or_srl(train_feats, train_labels,
                                  SRLLoss().to(device), device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, test_labels)
        results['SRL']['lp'].append(lp); results['SRL']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # LoRA + Triplet
    for i, s in enumerate(seeds):
        print(f'\n  [5/6] LoRA+Triplet seed={s} ({i+1}/3)')
        model = train_lora_triplet(train_feats, train_labels, device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, test_labels)
        results['LoRA+Triplet']['lp'].append(lp); results['LoRA+Triplet']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    # LoRA + InfoNCE
    for i, s in enumerate(seeds):
        print(f'\n  [6/6] LoRA+InfoNCE seed={s} ({i+1}/3)')
        model = train_lora_infonce(train_feats, train_labels, device, dim, s)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, test_labels)
        results['LoRA+InfoNCE']['lp'].append(lp); results['LoRA+InfoNCE']['ar'].append(ar)
        print(f'    LP={lp:.4f}  AR={ar:.4f}')

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    base_dir  = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    seeds         = [42, 123, 456]
    method_order  = ['CLIP', 'ICon', 'SRL', 'LoRA+InfoNCE', 'LoRA+Triplet', 'EGA']
    all_results   = {}

    # Aircraft
    res = run_dataset(
        name='FGVC-Aircraft',
        feat_path=os.path.join(embed_dir, 'aircraft_test_vit_b32_features.npy'),
        label_path=os.path.join(embed_dir, 'aircraft_test_vit_b32_labels.npy'),
        device=device, seeds=seeds)
    if res: all_results['Aircraft'] = res

    # Food-101
    res = run_dataset(
        name='Food-101',
        feat_path=os.path.join(embed_dir, 'food101_features.npy'),
        label_path=os.path.join(embed_dir, 'food101_labels.npy'),
        device=device, seeds=seeds)
    if res: all_results['Food-101'] = res

    # ── Final summary ────────────────────────────────────────────────
    print('\n')
    print('=' * 80)
    print('  Aircraft & Food-101 Final Summary  (K=1, nprobe=1, mean ± std over 3 seeds)')
    print('=' * 80)
    for ds, results in all_results.items():
        print(f'\n  {ds}')
        print(f'  {"Method":<14} | {"LP@1 (mean ± std)":>22} | {"AR@1 (mean ± std)":>22}')
        print(f'  {"-"*14}-+-{"-"*22}-+-{"-"*22}')
        for m in method_order:
            lps = np.array(results[m]['lp'])
            ars = np.array(results[m]['ar'])
            lp_str = f'{lps.mean():.4f} ± {lps.std():.4f}'
            ar_str = f'{ars.mean():.4f} ± {ars.std():.4f}'
            print(f'  {m:<14} | {lp_str:>22} | {ar_str:>22}')

    # Save raw numbers
    save_path = os.path.join(base_dir, 'aircraft_food_3seed_results.txt')
    with open(save_path, 'w') as f:
        f.write('Aircraft & Food-101 OOD 3-seed results (seeds: 42, 123, 456)\n')
        f.write('=' * 80 + '\n')
        for ds, results in all_results.items():
            f.write(f'\n=== {ds} ===\n')
            for m in method_order:
                lps = results[m]['lp']
                ars = results[m]['ar']
                f.write(f'{m}: LP={lps}  AR={ars}\n')
                f.write(f'  mean LP = {np.mean(lps):.4f} ± {np.std(lps):.4f}\n')
                f.write(f'  mean AR = {np.mean(ars):.4f} ± {np.std(ars):.4f}\n')
    print(f'\nDetailed results saved to: {save_path}')


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/20_aircraft_food_3seed.py 
# Device: cuda

# ======================================================================
#   Dataset: FGVC-Aircraft
# ======================================================================
#   Train classes: 80, samples: 2664
#   Unseen classes: 20, samples: 669

#   [1/6] Raw CLIP (deterministic)
#     LP=0.5119  AR=0.7738 (replicated for 3 seeds)

#   [2/6] EGA seed=42 (1/3)
#     LP=0.6190  AR=0.8869

#   [2/6] EGA seed=123 (2/3)
#     LP=0.6310  AR=0.9048

#   [2/6] EGA seed=456 (3/3)
#     LP=0.5833  AR=0.9226

#   [3/6] ICon seed=42 (1/3)
#     LP=0.4226  AR=0.8393

#   [3/6] ICon seed=123 (2/3)
#     LP=0.5000  AR=0.8452

#   [3/6] ICon seed=456 (3/3)
#     LP=0.4881  AR=0.8750

#   [4/6] SRL seed=42 (1/3)
#     LP=0.4107  AR=0.8690

#   [4/6] SRL seed=123 (2/3)
#     LP=0.3333  AR=0.8571

#   [4/6] SRL seed=456 (3/3)
#     LP=0.3810  AR=0.9107

#   [5/6] LoRA+Triplet seed=42 (1/3)
#     LP=0.5774  AR=0.8810

#   [5/6] LoRA+Triplet seed=123 (2/3)
#     LP=0.5536  AR=0.8988

#   [5/6] LoRA+Triplet seed=456 (3/3)
#     LP=0.5774  AR=0.8988

#   [6/6] LoRA+InfoNCE seed=42 (1/3)
#     LP=0.5595  AR=0.9107

#   [6/6] LoRA+InfoNCE seed=123 (2/3)
#     LP=0.5119  AR=0.8750

#   [6/6] LoRA+InfoNCE seed=456 (3/3)
#     LP=0.5417  AR=0.8869

# ======================================================================
#   Dataset: Food-101
# ======================================================================
#   Train classes: 80, samples: 20000
#   Unseen classes: 21, samples: 5250

#   [1/6] Raw CLIP (deterministic)
#     LP=0.8812  AR=0.8423 (replicated for 3 seeds)

#   [2/6] EGA seed=42 (1/3)
#     LP=0.7921  AR=0.8736

#   [2/6] EGA seed=123 (2/3)
#     LP=0.7928  AR=0.8941

#   [2/6] EGA seed=456 (3/3)
#     LP=0.7890  AR=0.8705

#   [3/6] ICon seed=42 (1/3)
#     LP=0.5522  AR=0.8225

#   [3/6] ICon seed=123 (2/3)
#     LP=0.5567  AR=0.8195

#   [3/6] ICon seed=456 (3/3)
#     LP=0.5781  AR=0.8218

#   [4/6] SRL seed=42 (1/3)
#     LP=0.5613  AR=0.8088

#   [4/6] SRL seed=123 (2/3)
#     LP=0.5567  AR=0.8195

#   [4/6] SRL seed=456 (3/3)
#     LP=0.5377  AR=0.8446

#   [5/6] LoRA+Triplet seed=42 (1/3)
#     LP=0.8332  AR=0.8987

#   [5/6] LoRA+Triplet seed=123 (2/3)
#     LP=0.8416  AR=0.8987

#   [5/6] LoRA+Triplet seed=456 (3/3)
#     LP=0.8248  AR=0.8819

#   [6/6] LoRA+InfoNCE seed=42 (1/3)
#     LP=0.8781  AR=0.9018

#   [6/6] LoRA+InfoNCE seed=123 (2/3)
#     LP=0.8858  AR=0.9056

#   [6/6] LoRA+InfoNCE seed=456 (3/3)
#     LP=0.8858  AR=0.9094


# ================================================================================
#   Aircraft & Food-101 Final Summary  (K=1, nprobe=1, mean ± std over 3 seeds)
# ================================================================================

#   Aircraft
#   Method         |      LP@1 (mean ± std) |      AR@1 (mean ± std)
#   ---------------+------------------------+-----------------------
#   CLIP           |        0.5119 ± 0.0000 |        0.7738 ± 0.0000
#   ICon           |        0.4702 ± 0.0340 |        0.8532 ± 0.0156
#   SRL            |        0.3750 ± 0.0319 |        0.8790 ± 0.0230
#   LoRA+InfoNCE   |        0.5377 ± 0.0196 |        0.8909 ± 0.0148
#   LoRA+Triplet   |        0.5694 ± 0.0112 |        0.8929 ± 0.0084
#   EGA            |        0.6111 ± 0.0202 |        0.9048 ± 0.0146

#   Food-101
#   Method         |      LP@1 (mean ± std) |      AR@1 (mean ± std)
#   ---------------+------------------------+-----------------------
#   CLIP           |        0.8812 ± 0.0000 |        0.8423 ± 0.0000
#   ICon           |        0.5623 ± 0.0113 |        0.8213 ± 0.0013
#   SRL            |        0.5519 ± 0.0102 |        0.8243 ± 0.0150
#   LoRA+InfoNCE   |        0.8832 ± 0.0036 |        0.9056 ± 0.0031
#   LoRA+Triplet   |        0.8332 ± 0.0068 |        0.8931 ± 0.0079
#   EGA            |        0.7913 ± 0.0016 |        0.8794 ± 0.0105

# Detailed results saved to: /home/cc/hpdic/EGA/aircraft_food_3seed_results.txt
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 