# scripts/27_linear_probing_ood_3seed.py
# Zero-shot retrieval OOD (CIFAR-100 as base, test set as query)
# 3 seeds for mean ± std

import os
import numpy as np
import faiss

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

def run_one_seed(cifar100_feat_path, cifar100_label_path, test_feat_path, test_label_path, seed=42):
    np.random.seed(seed)

    # Load data
    base_feats = np.load(cifar100_feat_path).astype(np.float32)
    base_labels = np.load(cifar100_label_path)
    query_feats = np.load(test_feat_path).astype(np.float32)
    query_labels = np.load(test_label_path)

    # Normalize
    base_feats = base_feats / np.linalg.norm(base_feats, axis=1, keepdims=True)
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)

    dim = base_feats.shape[1]

    # Build index on CIFAR-100 (base)
    exact = faiss.IndexFlatL2(dim)
    exact.add(base_feats)
    _, gt = exact.search(query_feats, 1)

    nlist = 100
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base_feats)
    ivf.add(base_feats)
    ivf.nprobe = 1
    _, ret = ivf.search(query_feats, 1)

    lp = calculate_label_precision(ret, base_labels, query_labels, 1)
    ar = calculate_anns_recall(ret, gt, 1)
    return lp, ar

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    cifar100_feat = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    cifar100_label = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')

    datasets = [
        ("Aircraft", "aircraft_test_vit_b32_features.npy", "aircraft_test_vit_b32_labels.npy"),
        ("Food-101", "food101_features.npy", "food101_labels.npy"),
        ("CIFAR-10", "cifar10_vit_b32_features.npy", "cifar10_vit_b32_labels.npy"),
    ]

    seeds = [42, 123, 456]
    print(f'\n=== Zero-shot Retrieval OOD (3 seeds) ===\n')

    for name, test_feat, test_label in datasets:
        print(f'=== {name} ===')
        lp_list, ar_list = [], []
        for seed in seeds:
            lp, ar = run_one_seed(
                cifar100_feat,
                cifar100_label,
                os.path.join(embed_dir, test_feat),
                os.path.join(embed_dir, test_label),
                seed=seed
            )
            lp_list.append(lp)
            ar_list.append(ar)
            print(f'  Seed {seed}: LP@1 = {lp:.4f}, AR@1 = {ar:.4f}')

        print(f'  Mean ± std: LP@1 = {np.mean(lp_list):.4f} ± {np.std(lp_list):.4f}, '
              f'AR@1 = {np.mean(ar_list):.4f} ± {np.std(ar_list):.4f}\n')

if __name__ == '__main__':
    main()