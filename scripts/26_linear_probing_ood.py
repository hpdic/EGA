# scripts/27_linear_probing_ood_3seed.py
# Linear Probing OOD (3 seeds, mean ± std)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 用 CIFAR-100 作为 train set
    train_feats = np.load(cifar100_feat_path).astype(np.float32)
    train_labels = np.load(cifar100_label_path)
    test_feats = np.load(test_feat_path).astype(np.float32)
    test_labels = np.load(test_label_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_feats = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
    test_feats = test_feats / np.linalg.norm(test_feats, axis=1, keepdims=True)

    train_X = torch.from_numpy(train_feats).float().to(device)
    train_y = torch.from_numpy(train_labels).long().to(device)
    test_X = torch.from_numpy(test_feats).float().to(device)

    num_classes = len(np.unique(train_labels))
    model = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_out = model(train_X).cpu().numpy()
        test_out = model(test_X).cpu().numpy()

    dim = train_out.shape[1]
    exact = faiss.IndexFlatL2(dim)
    exact.add(train_out)
    _, gt = exact.search(test_out, 1)

    nlist = 50
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(train_out)
    ivf.add(train_out)
    ivf.nprobe = 1
    _, ret = ivf.search(test_out, 1)

    lp = calculate_label_precision(ret, train_labels, test_labels, 1)
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
    print(f'\n=== Linear Probing OOD (3 seeds) ===\n')

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


# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/26_linear_probing_ood.py 

# === Linear Probing OOD (3 seeds) ===

# === Aircraft ===
#   Seed 42: LP@1 = 0.0150, AR@1 = 0.4527
#   Seed 123: LP@1 = 0.0081, AR@1 = 0.4629
#   Seed 456: LP@1 = 0.0168, AR@1 = 0.4047
#   Mean ± std: LP@1 = 0.0133 ± 0.0038, AR@1 = 0.4401 ± 0.0254

# === Food-101 ===
#   Seed 42: LP@1 = 0.0056, AR@1 = 0.4846
#   Seed 123: LP@1 = 0.0075, AR@1 = 0.5260
#   Seed 456: LP@1 = 0.0101, AR@1 = 0.5379
#   Mean ± std: LP@1 = 0.0077 ± 0.0019, AR@1 = 0.5162 ± 0.0228

# === CIFAR-10 ===
#   Seed 42: LP@1 = 0.0060, AR@1 = 0.6475
#   Seed 123: LP@1 = 0.0062, AR@1 = 0.6327
#   Seed 456: LP@1 = 0.0072, AR@1 = 0.6559
#   Mean ± std: LP@1 = 0.0065 ± 0.0005, AR@1 = 0.6454 ± 0.0096

# (venv) cc@uc-a100:~/hpdic/EGA$ 

