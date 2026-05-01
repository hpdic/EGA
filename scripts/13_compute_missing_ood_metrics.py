# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# Compute missing metrics for the OOD summary table:
#   - CIFAR-10  : Label Precision @ K=1, nprobe=1   for CLIP/ICon/SRL/EGA
#   - Food-101  : ANNS Recall    @ K=1, nprobe=1    for CLIP/ICon/SRL/EGA
# All under K=1, nprobe=1 settings.

import os
import numpy as np
import faiss


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_label_precision(retrieved, base_labels, query_labels, k=1):
    """
    Fraction of retrieved top-k neighbors that share the same class as the query.
    """
    count = 0
    for i in range(len(query_labels)):
        neighbor_labels = base_labels[retrieved[i, :k]]
        count += np.sum(neighbor_labels == query_labels[i])
    return count / (len(query_labels) * k)


def compute_anns_recall(retrieved, gt, k=1):
    """
    Fraction of ground-truth top-k neighbors recovered by the approximate index.
    """
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)


# ─────────────────────────────────────────────
# Run a 75/25 split eval on a given feature/label pair
# ─────────────────────────────────────────────

def run_eval(features, labels, dataset_name, method_name,
             nlist=10, nprobe=1, k=1, seed=42):
    """
    Same 75/25 split protocol as your existing eval scripts.
    Reports (Label Precision @ k, ANNS Recall @ k).
    """
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    np.random.seed(seed)
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels   = labels[idx]

    split_idx    = int(len(features) * 0.75)
    base         = features[:split_idx]
    base_labels  = labels[:split_idx]
    query        = features[split_idx:]
    query_labels = labels[split_idx:]

    dim = base.shape[1]

    # ground truth: brute-force exact top-k (cosine via L2 on normalized)
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, k)

    # IVF approximate index
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = nprobe
    _, ret = ivf.search(query, k)

    lp = compute_label_precision(ret, base_labels, query_labels, k)
    ar = compute_anns_recall(ret, gt, k)
    return lp, ar


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    base_dir  = os.path.expanduser('~/hpdic/EGA/embeddings')

    # ========================================================
    # CIFAR-10 — need Label Precision (CLIP/ICon/SRL/EGA)
    # ========================================================
    print('=' * 60)
    print('  CIFAR-10  (K=1, nprobe=1)')
    print('=' * 60)

    cifar10_labels = np.load(os.path.join(base_dir, 'cifar10_vit_b32_labels.npy'))

    cifar10_methods = [
        ('CLIP', 'cifar10_vit_b32_features.npy'),
        ('ICon', 'cifar10_icon_features.npy'),
        ('SRL',  'cifar10_srl_features.npy'),
        ('EGA',  'cifar10_ega_features.npy'),
    ]

    print(f'  {"Method":<8}| {"Label Precision":>16} | {"ANNS Recall":>12}')
    print(f'  {"-"*8}+-{"-"*16}-+-{"-"*12}')
    for name, fname in cifar10_methods:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f'  {name:<8}|   FILE NOT FOUND ({fname})')
            continue
        feats = np.load(fpath)
        if len(feats) != len(cifar10_labels):
            print(f'  {name:<8}|   LENGTH MISMATCH ({len(feats)} vs {len(cifar10_labels)})')
            continue
        lp, ar = run_eval(feats, cifar10_labels, 'CIFAR-10', name)
        print(f'  {name:<8}| {lp:>16.4f} | {ar:>12.4f}')

    # ========================================================
    # Food-101 — need ANNS Recall (CLIP/ICon/SRL/EGA)
    # ========================================================
    print()
    print('=' * 60)
    print('  Food-101  (K=1, nprobe=1)')
    print('=' * 60)

    food_labels_path = os.path.join(base_dir, 'food101_labels.npy')
    if not os.path.exists(food_labels_path):
        print(f'  Labels file not found: {food_labels_path}')
        print('  Skipping Food-101.')
        return

    food_labels = np.load(food_labels_path)

    # Adjust filenames if yours differ
    food_methods = [
        ('CLIP', 'food101_features.npy'),
        ('ICon', 'food101_icon_features.npy'),
        ('SRL',  'food101_srl_features.npy'),
        ('EGA',  'food101_ega_features.npy'),
    ]

    print(f'  {"Method":<8}| {"Label Precision":>16} | {"ANNS Recall":>12}')
    print(f'  {"-"*8}+-{"-"*16}-+-{"-"*12}')
    for name, fname in food_methods:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f'  {name:<8}|   FILE NOT FOUND ({fname})')
            continue
        feats = np.load(fpath)
        if len(feats) != len(food_labels):
            print(f'  {name:<8}|   LENGTH MISMATCH ({len(feats)} vs {len(food_labels)})')
            continue
        lp, ar = run_eval(feats, food_labels, 'Food-101', name)
        print(f'  {name:<8}| {lp:>16.4f} | {ar:>12.4f}')


if __name__ == '__main__':
    main()