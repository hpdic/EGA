# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Corrected NeurIPS Rebuttal Experiment: Hidden Dimension Ablation Sweep on FGVC-Aircraft (OOD)
# Evaluates EGAMLP with hidden_dim in {256, 512, 1024, 2048} across seeds {42, 123, 456} using canonical protocol (nlist=10, nprobe=1).

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_convergence_epoch(epoch_losses, threshold_ratio=0.05):
    final_loss = epoch_losses[-1]
    threshold = final_loss * (1.0 + threshold_ratio)
    conv_epoch = len(epoch_losses)
    for epoch_idx in range(len(epoch_losses)):
        if all(l <= threshold for l in epoch_losses[epoch_idx:]):
            conv_epoch = epoch_idx + 1
            break
    return conv_epoch

def train_eval_ega_hdim(hidden_dim, seed, feat_path, label_path, device, epochs=150, batch_size=256, margin=0.2, ckpt_dir='models/rebuttal_hidden_dim_corrected'):
    set_seed(seed)

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)

    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels, train_ratio=0.8, split_seed=42)
    dim = train_feats.shape[1]
    loader = DataLoader(ClassAwareDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)

    model = EGAMLP(input_dim=dim, hidden_dim=hidden_dim).to(device)
    num_params = count_parameters(model)

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

    conv_epoch = find_convergence_epoch(epoch_losses, threshold_ratio=0.05)
    final_rho = epoch_rhos[-1]

    # Save checkpoint to dedicated folder
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ega_h{hidden_dim}_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)

    # Evaluate LP@1 and AR@1 using canonical nlist=10, nprobe=1 setup
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_feats).float().to(device)
        transformed = model(test_tensor).cpu().numpy()

    lp, ar = eval_lp_ar(transformed, test_labels, k=1, nprobe=1, nlist=10, eval_seed=42)

    return {
        'hidden_dim': hidden_dim,
        'seed': seed,
        'num_params': num_params,
        'lp1': float(lp),
        'ar1': float(ar),
        'final_rho': float(final_rho),
        'conv_epoch': conv_epoch,
        'wall_time': float(run_wall_time),
        'ckpt_path': ckpt_path,
        'epoch_losses': epoch_losses,
        'epoch_rhos': epoch_rhos
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = os.path.expanduser('~/hpdic/EGA')
    feat_path = os.path.join(base_dir, 'embeddings/aircraft_test_vit_b32_features.npy')
    label_path = os.path.join(base_dir, 'embeddings/aircraft_test_vit_b32_labels.npy')
    ckpt_dir = os.path.join(base_dir, 'models/rebuttal_hidden_dim_corrected')

    hidden_dims = [256, 512, 1024, 2048]
    seeds = [42, 123, 456]

    print("=" * 90)
    print("STARTING CORRECTED HIDDEN DIMENSION ABLATION SWEEP ON FGVC-AIRCRAFT (OOD)...")
    print("  Using canonical nlist=10, nprobe=1 FAISS evaluation protocol")
    print("=" * 90)

    results = []
    total_start = time.time()

    for hdim in hidden_dims:
        for seed in seeds:
            print(f"Running hidden_dim={hdim:<4} | seed={seed:<3} ...")
            res = train_eval_ega_hdim(hdim, seed, feat_path, label_path, device, ckpt_dir=ckpt_dir)
            results.append(res)
            print(f"  --> LP@1={res['lp1']:.4f}, AR@1={res['ar1']:.4f}, rho={res['final_rho']:.4f}, conv_epoch={res['conv_epoch']}, time={res['wall_time']:.2f}s")

    total_wall_time = time.time() - total_start

    # Save JSON results
    json_path = os.path.join(base_dir, 'rebuttal_hidden_dim_corrected_results.json')
    with open(json_path, 'w') as f:
        json.dump({'total_wall_time': total_wall_time, 'results': results}, f, indent=2)

    print("\n" + "=" * 105)
    print("CORRECTED HIDDEN DIMENSION ABLATION SUMMARY (MEAN ± STDERR OVER 3 SEEDS)")
    print("=" * 105)

    summary = {}
    for hdim in hidden_dims:
        sub = [r for r in results if r['hidden_dim'] == hdim]
        lps = np.array([r['lp1'] for r in sub])
        ars = np.array([r['ar1'] for r in sub])
        rhos = np.array([r['final_rho'] for r in sub])
        convs = np.array([r['conv_epoch'] for r in sub])
        times = np.array([r['wall_time'] for r in sub])
        num_params = sub[0]['num_params']

        lp_mean, lp_se = np.mean(lps), np.std(lps, ddof=1) / np.sqrt(len(lps))
        ar_mean, ar_se = np.mean(ars), np.std(ars, ddof=1) / np.sqrt(len(ars))
        rho_mean = np.mean(rhos)
        conv_mean = np.mean(convs)
        time_mean = np.mean(times)

        summary[hdim] = {
            'params': num_params,
            'lp_mean': lp_mean, 'lp_se': lp_se,
            'ar_mean': ar_mean, 'ar_se': ar_se,
            'rho_mean': rho_mean,
            'conv_mean': conv_mean,
            'time_mean': time_mean
        }

    print(f"{'Hidden Dim':<10} | {'Params':<10} | {'LP@1 (mean ± stderr)':<24} | {'AR@1 (mean ± stderr)':<24} | {'Final rho':<10} | {'Conv Epoch':<11} | {'Wall Time':<10}")
    print("-" * 115)
    for hdim in hidden_dims:
        info = summary[hdim]
        lp_str = f"{info['lp_mean']:.4f} ± {info['lp_se']:.4f}"
        ar_str = f"{info['ar_mean']:.4f} ± {info['ar_se']:.4f}"
        param_str = f"{info['params']:,}"
        print(f"{hdim:<10} | {param_str:<10} | {lp_str:<24} | {ar_str:<24} | {info['rho_mean']:.4f}     | {info['conv_mean']:<11.1f} | {info['time_mean']:.2f}s")

    print("\n" + "=" * 90)
    print("COMPACT MARKDOWN TABLE FOR NEURIPS REBUTTAL")
    print("=" * 90)

    md = []
    md.append("| Hidden Dim | Trainable Params | LP@1 (mean ± stderr) | AR@1 (mean ± stderr) | Final Active Triplet Ratio ($\\rho$) | Conv. Epoch (within 5%) | Wall Time (s) |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for hdim in hidden_dims:
        info = summary[hdim]
        p_str = f"{info['params']/1e6:.2f}M"
        lp_str = f"{info['lp_mean']:.4f} ± {info['lp_se']:.4f}"
        ar_str = f"{info['ar_mean']:.4f} ± {info['ar_se']:.4f}"
        rho_str = f"{info['rho_mean']:.4f}"
        conv_str = f"{info['conv_mean']:.1f}"
        t_str = f"{info['time_mean']:.2f}s"
        md.append(f"| {hdim} | {p_str} | {lp_str} | {ar_str} | {rho_str} | {conv_str} | {t_str} |")

    print("\n".join(md))

if __name__ == '__main__':
    main()
