# scripts/22_train_lora_cifar100_id.py
# CIFAR-100 ID 测试专用脚本（专门回应 Claude W7）

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


def train_lora(config_name, train_feats, train_labels, device, dim, rank=128, epochs=120):
    print(f'  Training LoRA + {config_name} (rank={rank}) ...')
    loader = DataLoader(ClassAwareDataset(train_feats, train_labels),
                        batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    
    model = LoRAAdapter(dim=dim, rank=rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
            print(f'    epoch {epoch+1}/{epochs}')
    return model


def eval_lp_ar(features, labels, name):
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    split = int(len(features) * 0.75)
    base = features[:split]
    base_labels = labels[:split]
    query = features[split:]
    query_labels = labels[split:]
    dim = base.shape[1]

    # FAISS IVF
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = 1
    _, ret = ivf.search(query, 1)

    # LP@1
    correct = 0
    for i in range(len(query_labels)):
        if base_labels[ret[i, 0]] == query_labels[i]:
            correct += 1
    lp = correct / len(query_labels)

    print(f'  {name:20} → LP@1 = {lp:.4f}  (nprobe=1)')
    return lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    # 加载 CIFAR-100
    features_path = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    labels_path = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')

    features = np.load(features_path).astype(np.float32)
    labels = np.load(labels_path)

    split = int(len(features) * 0.75)
    train_feats = features[:split]
    train_labels = labels[:split]
    test_feats = features[split:]
    test_labels = labels[split:]

    print(f"CIFAR-100 ID Test")
    print(f"  Train samples: {len(train_labels)} | Test samples: {len(test_labels)}")
    print(f"  LoRA rank = {args.rank}\n")

    dim = features.shape[1]

    # LoRA + Triplet
    model_triplet = train_lora("Triplet", train_feats, train_labels, device, dim, rank=args.rank)
    model_triplet.eval()
    with torch.no_grad():
        feats_triplet = model_triplet(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    eval_lp_ar(feats_triplet, test_labels, "LoRA+Triplet (r={})".format(args.rank))

    # LoRA + InfoNCE（简化版，后面可优化）
    print("\n  [InfoNCE 训练暂未实现，跳过]")

    print("\nDone.")


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/22_train_lora_cifar100_id.py --rank 64
# CIFAR-100 ID Test
#   Train samples: 7500 | Test samples: 2500
#   LoRA rank = 64

#   Training LoRA + Triplet (rank=64) ...
#     epoch 30/120
#     epoch 60/120
#     epoch 90/120
#     epoch 120/120
# WARNING clustering 1875 points to 100 centroids: please provide at least 3900 training points
#   LoRA+Triplet (r=64)  → LP@1 = 0.5568  (nprobe=1)

#   [InfoNCE 训练暂未实现，跳过]

# Done.


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ cd ~/hpdic/EGA
# python scripts/22_train_lora_cifar100_id.py --rank 128
# CIFAR-100 ID Test
#   Train samples: 7500 | Test samples: 2500
#   LoRA rank = 128

#   Training LoRA + Triplet (rank=128) ...
#     epoch 30/120
#     epoch 60/120
#     epoch 90/120
#     epoch 120/120
# WARNING clustering 1875 points to 100 centroids: please provide at least 3900 training points
#   LoRA+Triplet (r=128) → LP@1 = 0.5472  (nprobe=1)

#   [InfoNCE 训练暂未实现，跳过]

# Done.
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 
