# scripts/24_id_3seed.py
# In-distribution performance on CIFAR-100 with 75/25 split + 3 seeds
# Reports mean ± std for Table 1 (K=1, nprobe=1)

import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import faiss

from models.ega_mlp import EGAMLP


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


def train_ega(features, labels, device, dim, epochs=150, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loader = DataLoader(ClassAwareDataset(features, labels),
                        batch_size=256, shuffle=True)
    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    model.train()
    for epoch in range(epochs):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'    EGA epoch {epoch+1}/{epochs} (seed={seed})')
    return model


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


def run_one_seed(features, labels, seed=42):
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
    dim = features.shape[1]

    # Train EGA on train set
    model = train_ega(train_feats, train_labels, device, dim, epochs=150, seed=seed)
    model.eval()
    with torch.no_grad():
        train_out = model(torch.from_numpy(train_feats).float().to(device)).cpu().numpy()
        test_out = model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()

    # Normalize
    train_out = train_out / np.linalg.norm(train_out, axis=1, keepdims=True)
    test_out = test_out / np.linalg.norm(test_out, axis=1, keepdims=True)

    # Build index on TRAIN set（正确！）
    dim = train_out.shape[1]
    exact = faiss.IndexFlatL2(dim)
    exact.add(train_out)
    _, gt = exact.search(test_out, 1)

    nlist = 50  # 降低 nlist，避免警告
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(train_out)
    ivf.add(train_out)
    ivf.nprobe = 1
    _, ret = ivf.search(test_out, 1)

    lp = calculate_label_precision(ret, train_labels, test_labels, 1)
    ar = calculate_anns_recall(ret, gt, 1)
    return lp, ar


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    feat_path = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    label_path = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)

    print(f'\nCIFAR-100 in-distribution test (75/25 split, 3 seeds)')
    print(f'Total samples: {len(labels)}, classes: {len(np.unique(labels))}')

    seeds = [42, 123, 456]
    lp_list, ar_list = [], []

    for seed in seeds:
        print(f'\n=== Seed {seed} ===')
        lp, ar = run_one_seed(features, labels, seed=seed)
        lp_list.append(lp)
        ar_list.append(ar)
        print(f'  LP@1 = {lp:.4f}, AR@1 = {ar:.4f}')

    print('\n' + '=' * 60)
    print('  Table 1 (CIFAR-100 ID, K=1, nprobe=1, mean ± std over 3 seeds)')
    print(f'  EGA: LP@1 = {np.mean(lp_list):.4f} ± {np.std(lp_list):.4f}')
    print(f'       AR@1 = {np.mean(ar_list):.4f} ± {np.std(ar_list):.4f}')
    print('=' * 60)


# if __name__ == '__main__':
#     main()

# (venv) cc@uc-a100:~/hpdic/EGA$ 
# (venv) cc@uc-a100:~/hpdic/EGA$ 
# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/24_id_3seed.py 
# Device: cuda

# CIFAR-100 in-distribution test (75/25 split, 3 seeds)
# Total samples: 10000, classes: 100

# === Seed 42 ===
#     EGA epoch 30/150 (seed=42)
#     EGA epoch 60/150 (seed=42)
#     EGA epoch 90/150 (seed=42)
#     EGA epoch 120/150 (seed=42)
#     EGA epoch 150/150 (seed=42)
#   LP@1 = 0.6700, AR@1 = 0.8812

# === Seed 123 ===
#     EGA epoch 30/150 (seed=123)
#     EGA epoch 60/150 (seed=123)
#     EGA epoch 90/150 (seed=123)
#     EGA epoch 120/150 (seed=123)
#     EGA epoch 150/150 (seed=123)
#   LP@1 = 0.6856, AR@1 = 0.8648

# === Seed 456 ===
#     EGA epoch 30/150 (seed=456)
#     EGA epoch 60/150 (seed=456)
#     EGA epoch 90/150 (seed=456)
#     EGA epoch 120/150 (seed=456)
#     EGA epoch 150/150 (seed=456)
#   LP@1 = 0.6872, AR@1 = 0.8728

# ============================================================
#   Table 1 (CIFAR-100 ID, K=1, nprobe=1, mean ± std over 3 seeds)
#   EGA: LP@1 = 0.6809 ± 0.0078
#        AR@1 = 0.8729 ± 0.0067
# ============================================================
# (venv) cc@uc-a100:~/hpdic/EGA$ 