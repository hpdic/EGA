# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# NeurIPS Rebuttal Experiment: Triplet Margin Scan on Activation Ratio, ID/OOD Geometry, & Distance Overlap
# Evaluates margins {0.05, 0.1, 0.2, 0.3, 0.4, 0.5} across seeds {42, 123, 456} on FGVC-Aircraft OOD benchmark.

import os
import sys
import time
import json
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import faiss

from models.ega_mlp import EGAMLP
from utils_ega import split_by_class, eval_method

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def compute_pairwise_geometry(features, labels, num_pairs=20000, seed=42):
    rng = np.random.RandomState(seed)
    label_to_idx = collections.defaultdict(list)
    for idx, l in enumerate(labels):
        label_to_idx[l].append(idx)

    pos_dists = []
    neg_dists = []

    classes = list(label_to_idx.keys())
    for _ in range(num_pairs):
        c = rng.choice(classes)
        if len(label_to_idx[c]) >= 2:
            i, j = rng.choice(label_to_idx[c], size=2, replace=False)
            pos_d = np.linalg.norm(features[i] - features[j])
            pos_dists.append(pos_d)

        c1, c2 = rng.choice(classes, size=2, replace=False)
        i = rng.choice(label_to_idx[c1])
        k = rng.choice(label_to_idx[c2])
        neg_d = np.linalg.norm(features[i] - features[k])
        neg_dists.append(neg_d)

    pos_dists = np.array(pos_dists)
    neg_dists = np.array(neg_dists)

    d_pos_mean = float(np.mean(pos_dists))
    d_neg_mean = float(np.mean(neg_dists))
    gap = d_neg_mean - d_pos_mean

    # ROC-AUC of separating pos and neg pairs
    y_true = np.concatenate([np.zeros(len(pos_dists)), np.ones(len(neg_dists))])
    y_scores = np.concatenate([pos_dists, neg_dists])
    auc = float(roc_auc_score(y_true, y_scores))

    # Overlap Coefficient (area of intersection between pos and neg distance density histograms)
    bins = np.linspace(0, 2.0, 201)
    hist_pos, _ = np.histogram(pos_dists, bins=bins, density=True)
    hist_neg, _ = np.histogram(neg_dists, bins=bins, density=True)
    dx = bins[1] - bins[0]
    overlap_coef = float(np.sum(np.minimum(hist_pos, hist_neg)) * dx)

    return {
        'd_pos': d_pos_mean,
        'd_neg': d_neg_mean,
        'gap': gap,
        'roc_auc': auc,
        'overlap_coef': overlap_coef
    }

def train_eval_margin_run(margin, seed, feat_path, label_path, device, epochs=150, batch_size=256, ckpt_dir='models/rebuttal_margin_scan'):
    set_seed(seed)

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)

    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels, train_ratio=0.8)

    dim = train_feats.shape[1]
    loader = DataLoader(ClassAwareDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)

    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    model.train()
    final_rho = 0.0

    t0 = time.time()
    for epoch in range(epochs):
        ep_active = 0.0
        batches = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()

            out_a = model(a)
            out_p = model(p)
            out_n = model(n)

            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                d_pos = F.pairwise_distance(out_a, out_p)
                d_neg = F.pairwise_distance(out_a, out_n)
                active_ratio = (d_pos - d_neg + margin > 0).float().mean().item()

            ep_active += active_ratio
            batches += 1
        scheduler.step()
        if epoch == epochs - 1:
            final_rho = ep_active / batches

    run_wall_time = time.time() - t0

    # Save checkpoint to dedicated directory
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ega_air_margin_{margin}_seed_{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)

    # Geometry evaluation on ID (seen training set) and OOD (unseen test set)
    model.eval()
    with torch.no_grad():
        id_transformed = model(torch.from_numpy(train_feats).float().to(device)).cpu().numpy()
        ood_transformed = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()

    id_geo = compute_pairwise_geometry(id_transformed, train_labels, seed=seed)
    ood_geo = compute_pairwise_geometry(ood_transformed, test_labels, seed=seed)

    # Retrieval LP@1 and AR@1 on unseen Aircraft test set
    lp, ar = eval_method(ood_transformed, test_labels)

    return {
        'margin': margin,
        'seed': seed,
        'final_rho': float(final_rho),
        'lp1': float(lp),
        'ar1': float(ar),
        'id_d_pos': id_geo['d_pos'],
        'id_d_neg': id_geo['d_neg'],
        'id_gap': id_geo['gap'],
        'id_auc': id_geo['roc_auc'],
        'id_overlap': id_geo['overlap_coef'],
        'ood_d_pos': ood_geo['d_pos'],
        'ood_d_neg': ood_geo['d_neg'],
        'ood_gap': ood_geo['gap'],
        'ood_auc': ood_geo['roc_auc'],
        'ood_overlap': ood_geo['overlap_coef'],
        'wall_time': float(run_wall_time),
        'ckpt_path': ckpt_path
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = os.path.expanduser('~/hpdic/EGA')
    feat_path = os.path.join(base_dir, 'embeddings/aircraft_test_vit_b32_features.npy')
    label_path = os.path.join(base_dir, 'embeddings/aircraft_test_vit_b32_labels.npy')
    ckpt_dir = os.path.join(base_dir, 'models/rebuttal_margin_scan')

    margins = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = [42, 123, 456]

    print("=" * 80)
    print("STARTING MARGIN SCAN ON FGVC-AIRCRAFT (ID & OOD GEOMETRY)...")
    print("=" * 80)

    results = []
    total_start = time.time()

    for m in margins:
        for seed in seeds:
            print(f"Running margin={m:<4} | seed={seed} ...")
            res = train_eval_margin_run(m, seed, feat_path, label_path, device, ckpt_dir=ckpt_dir)
            results.append(res)
            print(f"  --> LP@1={res['lp1']:.4f}, AR@1={res['ar1']:.4f}, rho={res['final_rho']:.4f}, OOD gap={res['ood_gap']:.4f}, OOD AUC={res['ood_auc']:.4f}, time={res['wall_time']:.2f}s")

    total_wall_time = time.time() - total_start

    # Save JSON results
    json_path = os.path.join(base_dir, 'rebuttal_margin_scan_results.json')
    with open(json_path, 'w') as f:
        json.dump({'total_wall_time': total_wall_time, 'results': results}, f, indent=2)

    print("\n" + "=" * 100)
    print("MARGIN SCAN SUMMARY (MEAN ± STDERR OVER 3 SEEDS)")
    print("=" * 100)

    summary = {}
    for m in margins:
        sub = [r for r in results if r['margin'] == m]
        lps = np.array([r['lp1'] for r in sub])
        ars = np.array([r['ar1'] for r in sub])
        rhos = np.array([r['final_rho'] for r in sub])

        id_dpos = np.array([r['id_d_pos'] for r in sub])
        id_dneg = np.array([r['id_d_neg'] for r in sub])
        id_gaps = np.array([r['id_gap'] for r in sub])
        id_aucs = np.array([r['id_auc'] for r in sub])
        id_over = np.array([r['id_overlap'] for r in sub])

        ood_dpos = np.array([r['ood_d_pos'] for r in sub])
        ood_dneg = np.array([r['ood_d_neg'] for r in sub])
        ood_gaps = np.array([r['ood_gap'] for r in sub])
        ood_aucs = np.array([r['ood_auc'] for r in sub])
        ood_over = np.array([r['ood_overlap'] for r in sub])

        summary[m] = {
            'lp_mean': np.mean(lps), 'lp_se': np.std(lps, ddof=1) / np.sqrt(len(lps)),
            'ar_mean': np.mean(ars), 'ar_se': np.std(ars, ddof=1) / np.sqrt(len(ars)),
            'rho_mean': np.mean(rhos),
            'id_dpos': np.mean(id_dpos), 'id_dneg': np.mean(id_dneg), 'id_gap': np.mean(id_gaps), 'id_auc': np.mean(id_aucs), 'id_over': np.mean(id_over),
            'ood_dpos': np.mean(ood_dpos), 'ood_dneg': np.mean(ood_dneg), 'ood_gap': np.mean(ood_gaps), 'ood_auc': np.mean(ood_aucs), 'ood_over': np.mean(ood_over)
        }

    print(f"{'Margin':<7} | {'rho (seen)':<10} | {'LP@1 (mean ± se)':<20} | {'AR@1 (mean ± se)':<20} | {'ID d_pos / d_neg / Gap':<24} | {'OOD d_pos / d_neg / Gap':<24} | {'OOD ROC-AUC / Overlap':<22}")
    print("-" * 140)
    for m in margins:
        info = summary[m]
        lp_str = f"{info['lp_mean']:.4f} ± {info['lp_se']:.4f}"
        ar_str = f"{info['ar_mean']:.4f} ± {info['ar_se']:.4f}"
        id_str = f"{info['id_dpos']:.3f} / {info['id_dneg']:.3f} / {info['id_gap']:.3f}"
        ood_str = f"{info['ood_dpos']:.3f} / {info['ood_dneg']:.3f} / {info['ood_gap']:.3f}"
        stat_str = f"{info['ood_auc']:.4f} / {info['ood_over']:.4f}"
        print(f"{m:<7} | {info['rho_mean']:<10.4f} | {lp_str:<20} | {ar_str:<20} | {id_str:<24} | {ood_str:<24} | {stat_str:<22}")

    print("\n" + "=" * 90)
    print("COMPACT MARKDOWN TABLE FOR NEURIPS REBUTTAL")
    print("=" * 90)

    md = []
    md.append("| Margin ($m$) | Seen Active Ratio ($\\rho$) | OOD LP@1 | OOD AR@1 | ID $d_{\\text{pos}} / d_{\\text{neg}} / \\Delta$ | OOD $d_{\\text{pos}} / d_{\\text{neg}} / \\Delta$ | OOD ROC-AUC | OOD Hist Overlap |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in margins:
        info = summary[m]
        lp_str = f"{info['lp_mean']:.4f} ± {info['lp_se']:.4f}"
        ar_str = f"{info['ar_mean']:.4f} ± {info['ar_se']:.4f}"
        id_str = f"{info['id_dpos']:.3f} / {info['id_dneg']:.3f} / **{info['id_gap']:.3f}**"
        ood_str = f"{info['ood_dpos']:.3f} / {info['ood_dneg']:.3f} / **{info['ood_gap']:.3f}**"
        auc_str = f"**{info['ood_auc']:.4f}**"
        over_str = f"{info['ood_over']:.4f}"
        md.append(f"| {m} | {info['rho_mean']:.4f} | {lp_str} | {ar_str} | {id_str} | {ood_str} | {auc_str} | {over_str} |")

    print("\n".join(md))

if __name__ == '__main__':
    main()
