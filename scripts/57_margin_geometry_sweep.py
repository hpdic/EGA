# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Reviewer 1UVB Rebuttal Experiment: Triplet Margin Sweep on Gradient Activity, ID/OOD Geometry, & Retrieval
# Evaluates margins m in {0.05, 0.1, 0.2, 0.3, 0.4, 0.5} across seeds {42, 123, 456} on FGVC-Aircraft.

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

def generate_pair_indices(labels, num_pairs=20000, sampling_seed=42):
    rng = np.random.RandomState(sampling_seed)
    label_to_idx = collections.defaultdict(list)
    for idx, l in enumerate(labels):
        label_to_idx[l].append(idx)

    classes = list(label_to_idx.keys())
    pos_pairs = []
    neg_pairs = []

    for _ in range(num_pairs):
        # Pos pair
        c = rng.choice(classes)
        if len(label_to_idx[c]) >= 2:
            i, j = rng.choice(label_to_idx[c], size=2, replace=False)
            pos_pairs.append((i, j))
        # Neg pair
        c1, c2 = rng.choice(classes, size=2, replace=False)
        i = rng.choice(label_to_idx[c1])
        k = rng.choice(label_to_idx[c2])
        neg_pairs.append((i, k))

    return np.array(pos_pairs), np.array(neg_pairs)

def compute_geometry_from_pairs(features, pos_pairs, neg_pairs):
    # Compute L2 distances for fixed positive and negative pairs
    pos_dists = np.linalg.norm(features[pos_pairs[:, 0]] - features[pos_pairs[:, 1]], axis=1)
    neg_dists = np.linalg.norm(features[neg_pairs[:, 0]] - features[neg_pairs[:, 1]], axis=1)

    d_pos_mean = float(np.mean(pos_dists))
    d_neg_mean = float(np.mean(neg_dists))
    gap = d_neg_mean - d_pos_mean

    # ROC-AUC of distinguishing neg pairs (label 1, larger distance) from pos pairs (label 0, smaller distance)
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

def train_eval_margin_run(margin, seed, feat_path, label_path, device, epochs=150, batch_size=256, ckpt_dir='models/rebuttal_margin_geometry'):
    set_seed(seed)

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)

    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels, train_ratio=0.8, split_seed=42)

    dim = train_feats.shape[1]
    loader = DataLoader(ClassAwareDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)

    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    model.train()
    epoch_losses = []
    epoch_rhos = []

    t0 = time.time()
    for epoch in range(epochs):
        ep_loss = 0.0
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

            ep_loss += loss.item()
            ep_active += active_ratio
            batches += 1
        scheduler.step()

        epoch_losses.append(ep_loss / batches)
        epoch_rhos.append(ep_active / batches)

    run_wall_time = time.time() - t0

    # Save checkpoint to dedicated folder
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ega_air_m{margin}_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)

    # Epoch specific rhos (1-indexed: epoch 10, 50, 100, 150)
    rho_ep10 = float(epoch_rhos[9])
    rho_ep50 = float(epoch_rhos[49])
    rho_ep100 = float(epoch_rhos[99])
    final_rho = float(epoch_rhos[149])
    final_loss = float(epoch_losses[149])

    # Evaluate geometry on ID (seen training classes) and OOD (unseen test classes)
    model.eval()
    with torch.no_grad():
        id_transformed = model(torch.from_numpy(train_feats).float().to(device)).cpu().numpy()
        ood_transformed = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()

    # Pre-generated fixed pairs per seed for ID and OOD
    id_pos_pairs, id_neg_pairs = generate_pair_indices(train_labels, num_pairs=20000, sampling_seed=seed)
    ood_pos_pairs, ood_neg_pairs = generate_pair_indices(test_labels, num_pairs=20000, sampling_seed=seed)

    id_geo = compute_geometry_from_pairs(id_transformed, id_pos_pairs, id_neg_pairs)
    ood_geo = compute_geometry_from_pairs(ood_transformed, ood_pos_pairs, ood_neg_pairs)

    # Retrieval LP@1 and AR@1 using canonical nlist=10, nprobe=1 setup
    lp, ar = eval_lp_ar(ood_transformed, test_labels, k=1, nprobe=1, nlist=10, eval_seed=42)

    return {
        'margin': margin,
        'seed': seed,
        'final_rho': final_rho,
        'rho_ep10': rho_ep10,
        'rho_ep50': rho_ep50,
        'rho_ep100': rho_ep100,
        'final_loss': final_loss,
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
    ckpt_dir = os.path.join(base_dir, 'models/rebuttal_margin_geometry')

    margins = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = [42, 123, 456]

    print("=" * 95)
    print("STARTING REVIEWER 1UVB MARGIN SWEEP EXPERIMENT ON FGVC-AIRCRAFT...")
    print("  Evaluating margins m in {0.05, 0.1, 0.2, 0.3, 0.4, 0.5} across seeds {42, 123, 456}")
    print("=" * 95)

    results = []
    total_start = time.time()

    for m in margins:
        for seed in seeds:
            print(f"Running margin={m:<4} | seed={seed:<3} ...")
            res = train_eval_margin_run(m, seed, feat_path, label_path, device, ckpt_dir=ckpt_dir)
            results.append(res)
            print(f"  --> LP@1={res['lp1']:.4f}, AR@1={res['ar1']:.4f}, rho={res['final_rho']:.4f}, OOD gap={res['ood_gap']:.4f}, OOD AUC={res['ood_auc']:.4f}, time={res['wall_time']:.2f}s")

    total_wall_time = time.time() - total_start

    # Save JSON results
    json_path = os.path.join(base_dir, 'rebuttal_margin_geometry_results.json')
    with open(json_path, 'w') as f:
        json.dump({'total_wall_time': total_wall_time, 'results': results}, f, indent=2)

    # Compute Summary
    summary = {}
    for m in margins:
        sub = [r for r in results if r['margin'] == m]
        rhos = np.array([r['final_rho'] for r in sub])
        lps = np.array([r['lp1'] for r in sub])
        ars = np.array([r['ar1'] for r in sub])

        id_dpos = np.array([r['id_d_pos'] for r in sub])
        id_dneg = np.array([r['id_d_neg'] for r in sub])
        id_gaps = np.array([r['id_gap'] for r in sub])
        id_aucs = np.array([r['id_auc'] for r in sub])

        ood_dpos = np.array([r['ood_d_pos'] for r in sub])
        ood_dneg = np.array([r['ood_d_neg'] for r in sub])
        ood_gaps = np.array([r['ood_gap'] for r in sub])
        ood_aucs = np.array([r['ood_auc'] for r in sub])

        summary[m] = {
            'rho_mean': np.mean(rhos), 'rho_se': np.std(rhos, ddof=1) / np.sqrt(len(rhos)),
            'id_dpos_mean': np.mean(id_dpos), 'id_dpos_se': np.std(id_dpos, ddof=1) / np.sqrt(len(id_dpos)),
            'id_dneg_mean': np.mean(id_dneg), 'id_dneg_se': np.std(id_dneg, ddof=1) / np.sqrt(len(id_dneg)),
            'id_gap_mean': np.mean(id_gaps), 'id_gap_se': np.std(id_gaps, ddof=1) / np.sqrt(len(id_gaps)),
            'id_auc_mean': np.mean(id_aucs), 'id_auc_se': np.std(id_aucs, ddof=1) / np.sqrt(len(id_aucs)),
            'ood_dpos_mean': np.mean(ood_dpos), 'ood_dpos_se': np.std(ood_dpos, ddof=1) / np.sqrt(len(ood_dpos)),
            'ood_dneg_mean': np.mean(ood_dneg), 'ood_dneg_se': np.std(ood_dneg, ddof=1) / np.sqrt(len(ood_dneg)),
            'ood_gap_mean': np.mean(ood_gaps), 'ood_gap_se': np.std(ood_gaps, ddof=1) / np.sqrt(len(ood_gaps)),
            'ood_auc_mean': np.mean(ood_aucs), 'ood_auc_se': np.std(ood_aucs, ddof=1) / np.sqrt(len(ood_aucs)),
            'lp_mean': np.mean(lps), 'lp_se': np.std(lps, ddof=1) / np.sqrt(len(lps)),
            'ar_mean': np.mean(ars), 'ar_se': np.std(ars, ddof=1) / np.sqrt(len(ars))
        }

    print("\n" + "=" * 120)
    print("TABLE A: MARGIN SWEEP SUMMARY (MEAN ± STDERR OVER 3 SEEDS)")
    print("=" * 120)

    md_a = []
    md_a.append("| Margin | Active Ratio | ID $d_{\\text{pos}}$ | ID $d_{\\text{neg}}$ | ID Gap | ID AUC | OOD $d_{\\text{pos}}$ | OOD $d_{\\text{neg}}$ | OOD Gap | OOD AUC | OOD LP@1 | OOD AR@1 |")
    md_a.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in margins:
        s = summary[m]
        rho_s = f"{s['rho_mean']:.4f} ± {s['rho_se']:.4f}"
        id_dp_s = f"{s['id_dpos_mean']:.4f} ± {s['id_dpos_se']:.4f}"
        id_dn_s = f"{s['id_dneg_mean']:.4f} ± {s['id_dneg_se']:.4f}"
        id_g_s = f"{s['id_gap_mean']:.4f} ± {s['id_gap_se']:.4f}"
        id_auc_s = f"{s['id_auc_mean']:.4f} ± {s['id_auc_se']:.4f}"
        ood_dp_s = f"{s['ood_dpos_mean']:.4f} ± {s['ood_dpos_se']:.4f}"
        ood_dn_s = f"{s['ood_dneg_mean']:.4f} ± {s['ood_dneg_se']:.4f}"
        ood_g_s = f"{s['ood_gap_mean']:.4f} ± {s['ood_gap_se']:.4f}"
        ood_auc_s = f"{s['ood_auc_mean']:.4f} ± {s['ood_auc_se']:.4f}"
        lp_s = f"{s['lp_mean']:.4f} ± {s['lp_se']:.4f}"
        ar_s = f"{s['ar_mean']:.4f} ± {s['ar_se']:.4f}"

        row = f"| {m} | {rho_s} | {id_dp_s} | {id_dn_s} | {id_g_s} | {id_auc_s} | {ood_dp_s} | {ood_dn_s} | {ood_g_s} | {ood_auc_s} | {lp_s} | {ar_s} |"
        md_a.append(row)

    print("\n".join(md_a))

    print("\n" + "=" * 120)
    print("TABLE B: PER-SEED RAW RESULTS FOR ALL 18 RUNS")
    print("=" * 120)

    md_b = []
    md_b.append("| Margin | Seed | Final $\\rho$ | $\\rho_{\\text{ep10}}$ | $\\rho_{\\text{ep50}}$ | $\\rho_{\\text{ep100}}$ | Final Loss | ID $d_{\\text{pos}}$ | ID $d_{\\text{neg}}$ | ID Gap | ID AUC | OOD $d_{\\text{pos}}$ | OOD $d_{\\text{neg}}$ | OOD Gap | OOD AUC | OOD Overlap | OOD LP@1 | OOD AR@1 |")
    md_b.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        row = f"| {r['margin']} | {r['seed']} | {r['final_rho']:.4f} | {r['rho_ep10']:.4f} | {r['rho_ep50']:.4f} | {r['rho_ep100']:.4f} | {r['final_loss']:.4f} | {r['id_d_pos']:.4f} | {r['id_d_neg']:.4f} | {r['id_gap']:.4f} | {r['id_auc']:.4f} | {r['ood_d_pos']:.4f} | {r['ood_d_neg']:.4f} | {r['ood_gap']:.4f} | {r['ood_auc']:.4f} | {r['ood_overlap']:.4f} | {r['lp1']:.4f} | {r['ar1']:.4f} |"
        md_b.append(row)

    print("\n".join(md_b))

if __name__ == '__main__':
    main()
