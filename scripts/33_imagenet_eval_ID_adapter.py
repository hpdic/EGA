# scripts/51_eval_id_imagenet1000.py
# In-Distribution evaluation on ImageNet-1000 (same as CIFAR-100 ID table)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
import faiss
from models.ega_mlp import EGAMLP


# ─────────────────────────────────────────────
# LoRA Adapter (same as before)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────

class IConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        log_probs = F.log_softmax(sim, dim=1)
        target = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return F.kl_div(log_probs, target, reduction='batchmean')


class SRLLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_gen=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_gen = lambda_gen

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        bs = features.shape[0]
        logits = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(features.device)
        exp_logits = torch.exp(logits) * (1 - torch.eye(bs).to(features.device))
        uniformity = torch.log(exp_logits.sum(dim=1)).mean()
        pos_logits = (exp_logits * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        homog = -torch.log(pos_logits + 1e-8).mean()
        return homog + self.lambda_gen * uniformity


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


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class TripletDataset(Dataset):
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


# ─────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────

def train_icon(features, labels, device, dim, epochs=100, batch_size=512):
    loader = DataLoader(StandardDataset(features, labels), batch_size=batch_size, shuffle=True)
    model = EGAMLP(input_dim=dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = IConLoss().to(device)
    model.train()
    for epoch in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f'    ICon epoch {epoch+1}/{epochs}')
    return model


def train_srl(features, labels, device, dim, epochs=100, batch_size=512):
    loader = DataLoader(StandardDataset(features, labels), batch_size=batch_size, shuffle=True)
    model = EGAMLP(input_dim=dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SRLLoss().to(device)
    model.train()
    for epoch in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f'    SRL epoch {epoch+1}/{epochs}')
    return model


def train_lora_triplet(features, labels, device, dim, epochs=150, batch_size=512, rank=128):
    loader = DataLoader(TripletDataset(features, labels), batch_size=batch_size, shuffle=True)
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
            print(f'    LoRA+Triplet epoch {epoch+1}/{epochs}')
    return model


def train_lora_infonce(features, labels, device, dim, epochs=150, batch_size=512, rank=128):
    loader = DataLoader(StandardDataset(features, labels), batch_size=batch_size, shuffle=True)
    model = LoRAAdapter(dim=dim, rank=rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SupConInfoNCE().to(device)
    model.train()
    for epoch in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'    LoRA+InfoNCE epoch {epoch+1}/{epochs}')
    return model


# ─────────────────────────────────────────────
# ID Evaluation (75/25 split, same classes)
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


def run_id_evaluation(features, labels, label_text, k_list=[1, 3, 5, 10]):
    print(f'\n=== {label_text} (In-Distribution) ===')
    
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    # 75/25 split (same classes)
    split_idx = int(len(features) * 0.75)
    base = features[:split_idx]
    base_labels = labels[:split_idx]
    query = features[split_idx:]
    query_labels = labels[split_idx:]
    
    dim = base.shape[1]
    
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    nlist = 10
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f'\nK={k}')
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            recall = calculate_anns_recall(ret, gt, k)
            precision = calculate_label_precision(ret, base_labels, query_labels, k)
            print(f'  nprobe {nprobe}: LP = {precision:.4f} | Recall = {recall:.4f}')


# ─────────────────────────────────────────────
# Main (ID Evaluation)
# ─────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')
    
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    # Load all ImageNet-1000 features
    features = np.load(os.path.join(embed_dir, 'imagenet1000_features.npy')).astype(np.float32)
    labels = np.load(os.path.join(embed_dir, 'imagenet1000_labels.npy'))
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    print(f'Total samples: {len(features)}, Total classes: {len(np.unique(labels))}')
    print('This is IN-DISTRIBUTION evaluation (75/25 split on all 1000 classes)\n')
    
    dim = features.shape[1]
    
    # Train all methods on ALL data (ID setting)
    print('Training all methods on ImageNet-1000 (ID setting)...\n')
    
    # 1. ICon
    print('=' * 70)
    print('Training ICon...')
    icon_model = train_icon(features, labels, device, dim, epochs=100)
    icon_model.eval()
    with torch.no_grad():
        icon_out = icon_model(torch.from_numpy(features).float().to(device)).cpu().numpy()
    run_id_evaluation(icon_out, labels, 'ICon')
    
    # 2. SRL
    print('=' * 70)
    print('Training SRL...')
    srl_model = train_srl(features, labels, device, dim, epochs=100)
    srl_model.eval()
    with torch.no_grad():
        srl_out = srl_model(torch.from_numpy(features).float().to(device)).cpu().numpy()
    run_id_evaluation(srl_out, labels, 'SRL')
    
    # 3. LoRA+Triplet
    print('=' * 70)
    print('Training LoRA+Triplet (r=128)...')
    lora_triplet = train_lora_triplet(features, labels, device, dim, epochs=150, rank=128)
    lora_triplet.eval()
    with torch.no_grad():
        lora_triplet_out = lora_triplet(torch.from_numpy(features).float().to(device)).cpu().numpy()
    run_id_evaluation(lora_triplet_out, labels, 'LoRA+Triplet r=128')
    
    # 4. LoRA+InfoNCE
    print('=' * 70)
    print('Training LoRA+InfoNCE (r=128)...')
    lora_infonce = train_lora_infonce(features, labels, device, dim, epochs=150, rank=128)
    lora_infonce.eval()
    with torch.no_grad():
        lora_infonce_out = lora_infonce(torch.from_numpy(features).float().to(device)).cpu().numpy()
    run_id_evaluation(lora_infonce_out, labels, 'LoRA+InfoNCE r=128')
    
    print('\n' + '=' * 70)
    print('ID Evaluation completed!')
    print('=' * 70)


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/33_imagenet_eval_ID.py 
# Device: cuda

# Total samples: 34745, Total classes: 1000
# This is IN-DISTRIBUTION evaluation (75/25 split on all 1000 classes)

# Training all methods on ImageNet-1000 (ID setting)...

# ======================================================================
# Training ICon...
#     ICon epoch 20/100
#     ICon epoch 40/100
#     ICon epoch 60/100
#     ICon epoch 80/100
#     ICon epoch 100/100

# === ICon (In-Distribution) ===

# K=1
#   nprobe 1: LP = 0.8301 | Recall = 0.8920
#   nprobe 5: LP = 0.8421 | Recall = 0.9994
#   nprobe 10: LP = 0.8421 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.8091 | Recall = 0.8843
#   nprobe 5: LP = 0.8278 | Recall = 0.9994
#   nprobe 10: LP = 0.8279 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.7938 | Recall = 0.8798
#   nprobe 5: LP = 0.8174 | Recall = 0.9993
#   nprobe 10: LP = 0.8174 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.7542 | Recall = 0.8721
#   nprobe 5: LP = 0.7931 | Recall = 0.9992
#   nprobe 10: LP = 0.7933 | Recall = 1.0000
# ======================================================================
# Training SRL...
#     SRL epoch 20/100
#     SRL epoch 40/100
#     SRL epoch 60/100
#     SRL epoch 80/100
#     SRL epoch 100/100

# === SRL (In-Distribution) ===

# K=1
#   nprobe 1: LP = 0.7975 | Recall = 0.7876
#   nprobe 5: LP = 0.8272 | Recall = 0.9960
#   nprobe 10: LP = 0.8273 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.7662 | Recall = 0.7748
#   nprobe 5: LP = 0.8129 | Recall = 0.9944
#   nprobe 10: LP = 0.8133 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.7412 | Recall = 0.7691
#   nprobe 5: LP = 0.7993 | Recall = 0.9938
#   nprobe 10: LP = 0.7997 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.6883 | Recall = 0.7561
#   nprobe 5: LP = 0.7719 | Recall = 0.9921
#   nprobe 10: LP = 0.7728 | Recall = 1.0000
# ======================================================================
# Training LoRA+Triplet (r=128)...
#     LoRA+Triplet epoch 30/150
#     LoRA+Triplet epoch 60/150
#     LoRA+Triplet epoch 90/150
#     LoRA+Triplet epoch 120/150
#     LoRA+Triplet epoch 150/150

# === LoRA+Triplet r=128 (In-Distribution) ===

# K=1
#   nprobe 1: LP = 0.4979 | Recall = 0.8946
#   nprobe 5: LP = 0.5153 | Recall = 0.9994
#   nprobe 10: LP = 0.5155 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.4607 | Recall = 0.8838
#   nprobe 5: LP = 0.4805 | Recall = 0.9994
#   nprobe 10: LP = 0.4806 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.4318 | Recall = 0.8773
#   nprobe 5: LP = 0.4525 | Recall = 0.9994
#   nprobe 10: LP = 0.4526 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.3789 | Recall = 0.8698
#   nprobe 5: LP = 0.4055 | Recall = 0.9991
#   nprobe 10: LP = 0.4057 | Recall = 1.0000
# ======================================================================
# Training LoRA+InfoNCE (r=128)...
#     LoRA+InfoNCE epoch 30/150
#     LoRA+InfoNCE epoch 60/150
#     LoRA+InfoNCE epoch 90/150
#     LoRA+InfoNCE epoch 120/150
#     LoRA+InfoNCE epoch 150/150

# === LoRA+InfoNCE r=128 (In-Distribution) ===

# K=1
#   nprobe 1: LP = 0.5705 | Recall = 0.8749
#   nprobe 5: LP = 0.5896 | Recall = 0.9993
#   nprobe 10: LP = 0.5896 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.5361 | Recall = 0.8660
#   nprobe 5: LP = 0.5583 | Recall = 0.9990
#   nprobe 10: LP = 0.5584 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.5076 | Recall = 0.8620
#   nprobe 5: LP = 0.5352 | Recall = 0.9988
#   nprobe 10: LP = 0.5354 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.4573 | Recall = 0.8528
#   nprobe 5: LP = 0.4904 | Recall = 0.9983
#   nprobe 10: LP = 0.4906 | Recall = 1.0000

# ======================================================================
# ID Evaluation completed!
# ======================================================================
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 