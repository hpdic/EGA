# RS4z emergency rebuttal: fill in missing EGA margin points (0.5x / 2.0x default=0.2)
# on CIFAR-10 and Food-101, following the exact protocol of the existing rebuttal scripts:
#   - Food-101: within-dataset 80/20 seen/unseen class split (scripts/57_margin_geometry_sweep.py)
#   - CIFAR-10: cross-dataset OOD (train on CIFAR-100, eval on all of CIFAR-10) (scripts/19_cifar10_3seed.py)
# Usage:
#   python scripts/58_rs4z_missing_margins.py --dataset food101 --margins 0.1 0.4 --seeds 42
#   python scripts/58_rs4z_missing_margins.py --dataset cifar10 --margins 0.1 0.4 --seeds 42

import os
import time
import json
import argparse
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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_by_class(features, labels, train_ratio=0.8, split_seed=42):
    unique_classes = np.unique(labels)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes = unique_classes[num_train:]
    train_mask = np.isin(labels, train_classes)
    test_mask = np.isin(labels, test_classes)
    return (features[train_mask], labels[train_mask]), (features[test_mask], labels[test_mask])

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

def eval_lp_ar(features, labels, k=1, nprobe=1, nlist=10, eval_seed=42):
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    np.random.seed(eval_seed)
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels = labels[idx]

    split = int(len(features) * 0.75)
    base = features[:split]
    base_labels = labels[:split]
    query = features[split:]
    query_labels = labels[split:]
    dim = base.shape[1]

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

def train_ega(train_feats, train_labels, margin, seed, device, dim, epochs=150, batch_size=256):
    set_seed(seed)
    loader = DataLoader(ClassAwareDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    model.train()
    t0 = time.time()
    for _ in range(epochs):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        scheduler.step()
    wall_time = time.time() - t0
    return model, wall_time

def run_food101(margin, seed, device, base_dir, ckpt_dir):
    embed_dir = os.path.join(base_dir, 'embeddings')
    features = np.load(os.path.join(embed_dir, 'food101_features.npy')).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(os.path.join(embed_dir, 'food101_labels.npy'))
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels, train_ratio=0.8, split_seed=42)
    dim = train_feats.shape[1]
    model, wall_time = train_ega(train_feats, train_labels, margin, seed, device, dim)
    model.eval()
    with torch.no_grad():
        ood_out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp, ar = eval_lp_ar(ood_out, test_labels, k=1, nprobe=1, nlist=10, eval_seed=42)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ega_food101_m{margin}_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)
    return {'dataset': 'Food-101', 'margin': margin, 'seed': seed, 'lp1': float(lp), 'ar1': float(ar),
            'wall_time': float(wall_time), 'ckpt_path': ckpt_path}

def run_aircraft(margin, seed, device, base_dir, ckpt_dir):
    embed_dir = os.path.join(base_dir, 'embeddings')
    features = np.load(os.path.join(embed_dir, 'aircraft_test_vit_b32_features.npy')).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(os.path.join(embed_dir, 'aircraft_test_vit_b32_labels.npy'))
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels, train_ratio=0.8, split_seed=42)
    dim = train_feats.shape[1]
    model, wall_time = train_ega(train_feats, train_labels, margin, seed, device, dim)
    model.eval()
    with torch.no_grad():
        ood_out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp, ar = eval_lp_ar(ood_out, test_labels, k=1, nprobe=1, nlist=10, eval_seed=42)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ega_aircraft_m{margin}_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)
    return {'dataset': 'Aircraft', 'margin': margin, 'seed': seed, 'lp1': float(lp), 'ar1': float(ar),
            'wall_time': float(wall_time), 'ckpt_path': ckpt_path}

def run_cifar10(margin, seed, device, base_dir, ckpt_dir):
    embed_dir = os.path.join(base_dir, 'embeddings')
    train_feats = np.load(os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')).astype(np.float32)
    train_feats = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
    train_labels = np.load(os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy'))
    eval_feats = np.load(os.path.join(embed_dir, 'cifar10_vit_b32_features.npy')).astype(np.float32)
    eval_feats = eval_feats / np.linalg.norm(eval_feats, axis=1, keepdims=True)
    eval_labels = np.load(os.path.join(embed_dir, 'cifar10_vit_b32_labels.npy'))
    dim = train_feats.shape[1]
    model, wall_time = train_ega(train_feats, train_labels, margin, seed, device, dim)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(eval_feats).float().to(device)).cpu().numpy()
    lp, ar = eval_lp_ar(out, eval_labels, k=1, nprobe=1, nlist=10, eval_seed=42)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ega_cifar10_m{margin}_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)
    return {'dataset': 'CIFAR-10', 'margin': margin, 'seed': seed, 'lp1': float(lp), 'ar1': float(ar),
            'wall_time': float(wall_time), 'ckpt_path': ckpt_path}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['food101', 'cifar10', 'aircraft'])
    ap.add_argument('--margins', nargs='+', type=float, required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    ckpt_dir = os.path.join(base_dir, f'models/rs4z_missing_margins')

    results = []
    for m in args.margins:
        for s in args.seeds:
            print(f'[{args.dataset}] margin={m} seed={s} ...')
            if args.dataset == 'food101':
                res = run_food101(m, s, device, base_dir, ckpt_dir)
            elif args.dataset == 'aircraft':
                res = run_aircraft(m, s, device, base_dir, ckpt_dir)
            else:
                res = run_cifar10(m, s, device, base_dir, ckpt_dir)
            print(f"  --> LP@1={res['lp1']:.4f} AR@1={res['ar1']:.4f} time={res['wall_time']:.1f}s")
            results.append(res)

    out_path = os.path.join(base_dir, f'artifacts/rs4z_fast/missing_margins_{args.dataset}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # append to existing if present
    existing = []
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
    existing.extend(results)
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'Saved {len(results)} new results to {out_path}')

if __name__ == '__main__':
    main()
