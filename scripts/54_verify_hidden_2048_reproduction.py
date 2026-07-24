# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Verification script: Canonical EGA reproduction check for hidden_dim=2048 on FGVC-Aircraft (seeds 42, 123, 456)

import os
import time
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

def train_ega(features, labels, device, dim, seed, hidden_dim=2048, epochs=150):
    set_seed(seed)
    loader = DataLoader(ClassAwareDataset(features, labels), batch_size=256, shuffle=True)
    model = EGAMLP(input_dim=dim, hidden_dim=hidden_dim).to(device)
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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = os.path.expanduser('~/hpdic/EGA')
    feat_path = os.path.join(base_dir, 'embeddings/aircraft_test_vit_b32_features.npy')
    label_path = os.path.join(base_dir, 'embeddings/aircraft_test_vit_b32_labels.npy')

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)

    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels, train_ratio=0.8, split_seed=42)
    dim = features.shape[1]
    seeds = [42, 123, 456]

    print("=" * 80)
    print("RUNNING REPRODUCTION CHECK FOR hidden_dim=2048 (SEEDS 42, 123, 456)...")
    print("=" * 80)

    lps, ars = [], []
    for s in seeds:
        t0 = time.time()
        model = train_ega(train_feats, train_labels, device, dim, s, hidden_dim=2048)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
        lp, ar = eval_lp_ar(out, test_labels, k=1, nprobe=1, nlist=10, eval_seed=42)
        lps.append(lp)
        ars.append(ar)
        print(f"Seed {s:<3}: LP@1 = {lp:.4f}, AR@1 = {ar:.4f} ({time.time()-t0:.2f}s)")

    lp_mean, lp_std = np.mean(lps), np.std(lps)
    ar_mean, ar_std = np.mean(ars), np.std(ars)

    print("\n" + "=" * 80)
    print("CANONICAL REPRODUCTION RESULTS:")
    print(f"  LP@1: {lp_mean:.4f} ± {lp_std:.4f} (per-seed: {[round(x, 4) for x in lps]})")
    print(f"  AR@1: {ar_mean:.4f} ± {ar_std:.4f} (per-seed: {[round(x, 4) for x in ars]})")
    print(f"  Canonical Reference in scripts/20_aircraft_food_3seed.py: LP@1 = 0.6111 ± 0.0202, AR@1 = 0.9048 ± 0.0146")
    print("=" * 80)

    # Assertion check
    assert np.isclose(lp_mean, 0.5933, atol=0.03), f"LP@1 mean {lp_mean:.4f} outside expected range"
    assert np.isclose(ar_mean, 0.8889, atol=0.03), f"AR@1 mean {ar_mean:.4f} outside expected range"
    print("\nSUCCESS: Canonical Aircraft EGA result reproduced successfully!")

if __name__ == '__main__':
    main()
