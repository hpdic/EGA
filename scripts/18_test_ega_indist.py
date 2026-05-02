# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# Verify EGA's in-distribution performance on CIFAR-100 with the exact same
# protocol as test_lora_indist.py and test_icon_srl_indist.py:
#   - same 75/25 split, same seed=42
#   - same nlist=10, same nprobe values
#   - report LP@k and AR@k for k in {1,3,5,10}
#
# Usage:
#   python scripts/test_ega_indist.py

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

from models.ega_mlp import EGAMLP


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class ClassAwareDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels   = labels
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


# ─────────────────────────────────────────────
# Train EGA (same hyperparams as Aircraft / Food-101 scripts)
# ─────────────────────────────────────────────

def train_ega(features, labels, device, dim, epochs=150):
    loader    = DataLoader(ClassAwareDataset(features, labels),
                           batch_size=256, shuffle=True)
    model     = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
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
            print(f'    EGA epoch {epoch+1}/{epochs}')
    return model


# ─────────────────────────────────────────────
# Eval (75/25 split, in-distribution, same as others)
# ─────────────────────────────────────────────

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


def eval_indist(features, labels, label_text, k_list=(1, 3, 5, 10),
                nprobe_list=(1, 5, 10), seed=42):
    print(f'\nEvaluating: {label_text}')
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    np.random.seed(seed)
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels   = labels[idx]

    split        = int(len(features) * 0.75)
    base         = features[:split]
    base_labels  = labels[:split]
    query        = features[split:]
    query_labels = labels[split:]
    dim          = base.shape[1]

    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    nlist = 10
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f'  K={k}')
        for np_ in nprobe_list:
            ivf.nprobe = np_
            _, ret = ivf.search(query, k)
            lp = calculate_label_precision(ret, base_labels, query_labels, k)
            ar = calculate_anns_recall(ret, gt, k)
            print(f'    nprobe={np_:>2}: LP={lp:.4f}  AR={ar:.4f}')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    base_dir  = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    feat_path  = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    label_path = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels   = np.load(label_path)

    print(f'\nCIFAR-100 in-distribution test (EGA verification)')
    print(f'Total samples: {len(labels)}, classes: {len(np.unique(labels))}')

    dim = features.shape[1]

    # ── Train EGA on full CIFAR-100 ─────────────────────────────────
    print('\nTraining EGA on full CIFAR-100 ...')
    np.random.seed(42); torch.manual_seed(42)
    model = train_ega(features, labels, device, dim)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(features).float().to(device)).cpu().numpy()

    # Save for reuse if needed
    np.save(os.path.join(embed_dir, 'cifar100_ega_features.npy'), out)
    print(f'\nSaved transformed features to embeddings/cifar100_ega_features.npy')

    eval_indist(out, labels, 'EGA (in-distribution, this run)')

    print('\n')
    print('=' * 60)
    print('  Compare against other methods on CIFAR-100 (K=1, nprobe=1):')
    print('    CLIP          : LP=0.549,  AR=0.805')
    print('    EGA (this run): LP=????,   AR=????')
    print('    LoRA+Triplet  : LP=0.612,  AR=0.908')
    print('    LoRA+InfoNCE  : LP=0.668,  AR=0.879')
    print('    SRL           : LP=0.992,  AR=0.991')
    print('    ICon          : LP=0.999,  AR=0.992')
    print('=' * 60)


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/18_test_ega_indist.py 
# Device: cuda

# CIFAR-100 in-distribution test (EGA verification)
# Total samples: 10000, classes: 100

# Training EGA on full CIFAR-100 ...
#     EGA epoch 30/150
#     EGA epoch 60/150
#     EGA epoch 90/150
#     EGA epoch 120/150
#     EGA epoch 150/150

# Saved transformed features to embeddings/cifar100_ega_features.npy

# Evaluating: EGA (in-distribution, this run)
#   K=1
#     nprobe= 1: LP=0.8508  AR=0.9328
#     nprobe= 5: LP=0.8524  AR=0.9992
#     nprobe=10: LP=0.8524  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.8377  AR=0.9285
#     nprobe= 5: LP=0.8445  AR=0.9996
#     nprobe=10: LP=0.8445  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.8314  AR=0.9263
#     nprobe= 5: LP=0.8394  AR=0.9996
#     nprobe=10: LP=0.8394  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.8180  AR=0.9198
#     nprobe= 5: LP=0.8302  AR=0.9995
#     nprobe=10: LP=0.8302  AR=1.0000


# ============================================================
#   Compare against other methods on CIFAR-100 (K=1, nprobe=1):
#     CLIP          : LP=0.549,  AR=0.805
#     EGA (this run): LP=????,   AR=????
#     LoRA+Triplet  : LP=0.612,  AR=0.908
#     LoRA+InfoNCE  : LP=0.668,  AR=0.879
#     SRL           : LP=0.992,  AR=0.991
#     ICon          : LP=0.999,  AR=0.992
# ============================================================
# (venv) (base) cc@uc-a100:~/hpdic/EGA$     