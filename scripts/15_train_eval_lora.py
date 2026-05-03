# scripts/15_train_eval_lora.py
# Capacity-matched LoRA OOD evaluation (r=64 / r=128)

import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import faiss
import argparse


class LoRAAdapter(nn.Module):
    def __init__(self, dim=512, rank=128, alpha=16):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T
        out = x + self.scale * delta
        return F.normalize(out, p=2, dim=1)


def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_class(features, labels, train_ratio=0.8):
    unique_classes = np.unique(labels)
    np.random.seed(42)
    np.random.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes = unique_classes[num_train:]
    train_mask = np.isin(labels, train_classes)
    test_mask = np.isin(labels, test_classes)
    return (features[train_mask], labels[train_mask]), \
           (features[test_mask], labels[test_mask])


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


class StandardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SupConInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.t
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        eye = torch.eye(features.shape[0], device=features.device)
        mask = mask - eye
        exp_sim = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        pos_count = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()


def train_lora_triplet(train_feats, train_labels, device, dim, epochs=150,
                       batch_size=256, margin=0.2, rank=128):
    loader = DataLoader(ClassAwareDataset(train_feats, train_labels),
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = LoRAAdapter(dim=dim, rank=rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

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
            print(f'    epoch {epoch+1}/{epochs}  (rank={rank})')
    return model


def train_lora_infonce(train_feats, train_labels, device, dim, epochs=150,
                       batch_size=256, rank=128):
    loader = DataLoader(StandardDataset(train_feats, train_labels),
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = LoRAAdapter(dim=dim, rank=rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SupConInfoNCE().to(device)

    model.train()
    for epoch in range(epochs):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'    epoch {epoch+1}/{epochs}  (rank={rank})')
    return model


def calculate_anns_recall(retrieved, gt, k):
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)


def calculate_label_precision(retrieved, base_labels, query_labels, k):
    count = 0
    for i in range(len(query_labels)):
        neighbor_labels = base_labels[retrieved[i, :k]]
        count += np.sum(neighbor_labels == query_labels[i])
    return count / (len(query_labels) * k)


def eval_method(features, labels, k=1, nlist=100, nprobe=1, seed=42):
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    np.random.seed(seed)
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

    lp = calculate_label_precision(ret, base_labels, query_labels, k)
    ar = calculate_anns_recall(ret, gt, k)
    return lp, ar


def run_for_dataset(name, feat_path, label_path, device, rank, epochs=150):
    print(f'\n{"=" * 70}')
    print(f'  Dataset: {name}  (LoRA rank = {rank})')
    print(f'{"=" * 70}')

    features = np.load(feat_path).astype(np.float32)
    labels = np.load(label_path)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    if "cifar10" in feat_path.lower():
        c100_feat = os.path.expanduser('~/hpdic/EGA/embeddings/cifar100_vit_b32_features.npy')
        c100_label = os.path.expanduser('~/hpdic/EGA/embeddings/cifar100_vit_b32_labels.npy')
        train_feats = np.load(c100_feat).astype(np.float32)
        train_feats = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
        train_labels = np.load(c100_label)
        test_feats, test_labels = features, labels
    else:
        (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)

    dim = features.shape[1]

    results = {}

    # LoRA + Triplet
    print('  [A] Training LoRA + Triplet ...')
    model_a = train_lora_triplet(train_feats, train_labels, device, dim, epochs=epochs, rank=rank)
    model_a.eval()
    with torch.no_grad():
        feats_a = model_a(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_a, ar_a = eval_method(feats_a, test_labels)
    print(f'    LP@1 = {lp_a:.4f}  AR@1 = {ar_a:.4f}')
    results['LoRA+Triplet'] = (lp_a, ar_a)

    # LoRA + InfoNCE
    print('  [B] Training LoRA + InfoNCE ...')
    model_b = train_lora_infonce(train_feats, train_labels, device, dim, epochs=epochs, rank=rank)
    model_b.eval()
    with torch.no_grad():
        feats_b = model_b(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_b, ar_b = eval_method(feats_b, test_labels)
    print(f'    LP@1 = {lp_b:.4f}  AR@1 = {ar_b:.4f}')
    results['LoRA+InfoNCE'] = (lp_b, ar_b)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128, help="LoRA rank (64 or 128 for capacity-matched)")
    args = parser.parse_args()

    set_global_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device} | LoRA rank: {args.rank}\n')

    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    datasets = [
        ("FGVC-Aircraft", "aircraft_test_vit_b32_features.npy", "aircraft_test_vit_b32_labels.npy"),
        ("Food-101",      "food101_features.npy",               "food101_labels.npy"),
        ("CIFAR-10",      "cifar10_vit_b32_features.npy",      "cifar10_vit_b32_labels.npy"),
    ]

    for name, feat_file, label_file in datasets:
        feat_path = os.path.join(embed_dir, feat_file)
        label_path = os.path.join(embed_dir, label_file)
        run_for_dataset(name, feat_path, label_path, device, rank=args.rank)

if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ cd ~/hpdic/EGA

# # 跑 r=64 的 OOD
# python scripts/15_train_eval_lora.py --rank 64

# # 跑 r=128 的 OOD
# python scripts/15_train_eval_lora.py --rank 128
# Device: cuda | LoRA rank: 64


# ======================================================================
#   Dataset: FGVC-Aircraft  (LoRA rank = 64)
# ======================================================================
#   [A] Training LoRA + Triplet ...
#     epoch 30/150  (rank=64)
#     epoch 60/150  (rank=64)
#     epoch 90/150  (rank=64)
#     epoch 120/150  (rank=64)
#     epoch 150/150  (rank=64)
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#     LP@1 = 0.5833  AR@1 = 0.6667
#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150  (rank=64)
#     epoch 60/150  (rank=64)
#     epoch 90/150  (rank=64)
#     epoch 120/150  (rank=64)
#     epoch 150/150  (rank=64)
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#     LP@1 = 0.5417  AR@1 = 0.6190

# ======================================================================
#   Dataset: Food-101  (LoRA rank = 64)
# ======================================================================
#   [A] Training LoRA + Triplet ...
#     epoch 30/150  (rank=64)
#     epoch 60/150  (rank=64)
#     epoch 90/150  (rank=64)
#     epoch 120/150  (rank=64)
#     epoch 150/150  (rank=64)
#     LP@1 = 0.8599  AR@1 = 0.6756
#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150  (rank=64)
#     epoch 60/150  (rank=64)
#     epoch 90/150  (rank=64)
#     epoch 120/150  (rank=64)
#     epoch 150/150  (rank=64)
#     LP@1 = 0.8667  AR@1 = 0.6504

# ======================================================================
#   Dataset: CIFAR-10  (LoRA rank = 64)
# ======================================================================
#   [A] Training LoRA + Triplet ...
#     epoch 30/150  (rank=64)
#     epoch 60/150  (rank=64)
#     epoch 90/150  (rank=64)
#     epoch 120/150  (rank=64)
#     epoch 150/150  (rank=64)
#     LP@1 = 0.8768  AR@1 = 0.6216
#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150  (rank=64)
#     epoch 60/150  (rank=64)
#     epoch 90/150  (rank=64)
#     epoch 120/150  (rank=64)
#     epoch 150/150  (rank=64)
#     LP@1 = 0.8788  AR@1 = 0.6008
# Device: cuda | LoRA rank: 128


# ======================================================================
#   Dataset: FGVC-Aircraft  (LoRA rank = 128)
# ======================================================================
#   [A] Training LoRA + Triplet ...
#     epoch 30/150  (rank=128)
#     epoch 60/150  (rank=128)
#     epoch 90/150  (rank=128)
#     epoch 120/150  (rank=128)
#     epoch 150/150  (rank=128)
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#     LP@1 = 0.5417  AR@1 = 0.6726
#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150  (rank=128)
#     epoch 60/150  (rank=128)
#     epoch 90/150  (rank=128)
#     epoch 120/150  (rank=128)
#     epoch 150/150  (rank=128)
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#     LP@1 = 0.5298  AR@1 = 0.6131

# ======================================================================
#   Dataset: Food-101  (LoRA rank = 128)
# ======================================================================
#   [A] Training LoRA + Triplet ...
#     epoch 30/150  (rank=128)
#     epoch 60/150  (rank=128)
#     epoch 90/150  (rank=128)
#     epoch 120/150  (rank=128)
#     epoch 150/150  (rank=128)
#     LP@1 = 0.8568  AR@1 = 0.6740
#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150  (rank=128)
#     epoch 60/150  (rank=128)
#     epoch 90/150  (rank=128)
#     epoch 120/150  (rank=128)
#     epoch 150/150  (rank=128)
#     LP@1 = 0.8599  AR@1 = 0.6702

# ======================================================================
#   Dataset: CIFAR-10  (LoRA rank = 128)
# ======================================================================
#   [A] Training LoRA + Triplet ...
#     epoch 30/150  (rank=128)
#     epoch 60/150  (rank=128)
#     epoch 90/150  (rank=128)
#     epoch 120/150  (rank=128)
#     epoch 150/150  (rank=128)
#     LP@1 = 0.8756  AR@1 = 0.6096
#   [B] Training LoRA + InfoNCE ...
#     epoch 30/150  (rank=128)
#     epoch 60/150  (rank=128)
#     epoch 90/150  (rank=128)
#     epoch 120/150  (rank=128)
#     epoch 150/150  (rank=128)
#     LP@1 = 0.8728  AR@1 = 0.5940
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 