# scripts/26_linear_probing.py
# Linear Probing baseline on CIFAR-100 (75/25 split + 3 seeds)
# Reports mean ± std for Table 1 comparison

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

def run_linear_probing(features, labels, seed=42, epochs=50):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 75/25 split
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels = labels[idx]
    split = int(len(features) * 0.75)
    train_feats, train_labels = features[:split], labels[:split]
    test_feats, test_labels = features[split:], labels[split:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Normalize
    train_feats = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
    test_feats = test_feats / np.linalg.norm(test_feats, axis=1, keepdims=True)

    # Convert to tensor
    train_X = torch.from_numpy(train_feats).float().to(device)
    train_y = torch.from_numpy(train_labels).long().to(device)
    test_X = torch.from_numpy(test_feats).float().to(device)
    test_y = torch.from_numpy(test_labels).long().to(device)

    # Linear classifier
    num_classes = len(np.unique(labels))
    model = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'    Linear Probing epoch {epoch+1}/{epochs} (seed={seed})')

    # Eval
    model.eval()
    with torch.no_grad():
        train_out = model(train_X).cpu().numpy()
        test_out = model(test_X).cpu().numpy()

    # Build index on TRAIN set
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

    feat_path = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    label_path = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')

    features = np.load(feat_path).astype(np.float32)
    labels = np.load(label_path)

    print(f'\n=== Linear Probing on CIFAR-100 (75/25 split, 3 seeds) ===\n')

    seeds = [42, 123, 456]
    lp_list, ar_list = [], []

    for seed in seeds:
        print(f'=== Seed {seed} ===')
        lp, ar = run_linear_probing(features, labels, seed=seed)
        lp_list.append(lp)
        ar_list.append(ar)
        print(f'  LP@1 = {lp:.4f}, AR@1 = {ar:.4f}\n')

    print('=' * 70)
    print('Linear Probing (mean ± std over 3 seeds)')
    print(f'LP@1 = {np.mean(lp_list):.4f} ± {np.std(lp_list):.4f}')
    print(f'AR@1 = {np.mean(ar_list):.4f} ± {np.std(ar_list):.4f}')
    print('=' * 70)

if __name__ == '__main__':
    main()


# (venv) cc@uc-a100:~/hpdic/EGA$ 
# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/25_linear_probing_id.py 

# === Linear Probing on CIFAR-100 (75/25 split, 3 seeds) ===

# === Seed 42 ===
#     Linear Probing epoch 10/50 (seed=42)
#     Linear Probing epoch 20/50 (seed=42)
#     Linear Probing epoch 30/50 (seed=42)
#     Linear Probing epoch 40/50 (seed=42)
#     Linear Probing epoch 50/50 (seed=42)
#   LP@1 = 0.5092, AR@1 = 0.7500

# === Seed 123 ===
#     Linear Probing epoch 10/50 (seed=123)
#     Linear Probing epoch 20/50 (seed=123)
#     Linear Probing epoch 30/50 (seed=123)
#     Linear Probing epoch 40/50 (seed=123)
#     Linear Probing epoch 50/50 (seed=123)
#   LP@1 = 0.5104, AR@1 = 0.7560

# === Seed 456 ===
#     Linear Probing epoch 10/50 (seed=456)
#     Linear Probing epoch 20/50 (seed=456)
#     Linear Probing epoch 30/50 (seed=456)
#     Linear Probing epoch 40/50 (seed=456)
#     Linear Probing epoch 50/50 (seed=456)
#   LP@1 = 0.5140, AR@1 = 0.7540

# ======================================================================
# Linear Probing (mean ± std over 3 seeds)
# LP@1 = 0.5112 ± 0.0020
# AR@1 = 0.7533 ± 0.0025
# ======================================================================
# (venv) cc@uc-a100:~/hpdic/EGA$ 