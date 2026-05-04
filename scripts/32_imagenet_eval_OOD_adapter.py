# scripts/50_train_eval_baselines_imagenet1000.py
# Train and evaluate ICon, SRL, LoRA+Triplet, LoRA+InfoNCE on ImageNet-1000 unseen classes

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
from sklearn.model_selection import train_test_split
from models.ega_mlp import EGAMLP


# ─────────────────────────────────────────────
# LoRA Adapter
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
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        log_probs = F.log_softmax(sim_matrix, dim=1)
        target_probs = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return F.kl_div(log_probs, target_probs, reduction='batchmean')


class SRLLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_gen=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_gen = lambda_gen

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        logits = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(features.device)
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(features.device))
        uniformity_loss = torch.log(exp_logits.sum(dim=1)).mean()
        pos_logits = (exp_logits * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        homogeneity_loss = -torch.log(pos_logits + 1e-8).mean()
        return homogeneity_loss + self.lambda_gen * uniformity_loss


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

def train_icon(train_feats, train_labels, device, dim, epochs=100, batch_size=512):
    loader = DataLoader(StandardDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    model = EGAMLP(input_dim=dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = IConLoss().to(device)

    model.train()
    for epoch in range(epochs):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f'    ICon epoch {epoch+1}/{epochs}')
    return model


def train_srl(train_feats, train_labels, device, dim, epochs=100, batch_size=512):
    loader = DataLoader(StandardDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    model = EGAMLP(input_dim=dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SRLLoss().to(device)

    model.train()
    for epoch in range(epochs):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f'    SRL epoch {epoch+1}/{epochs}')
    return model


def train_lora_triplet(train_feats, train_labels, device, dim, epochs=150, batch_size=512, rank=128):
    loader = DataLoader(TripletDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
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


def train_lora_infonce(train_feats, train_labels, device, dim, epochs=150, batch_size=512, rank=128):
    loader = DataLoader(StandardDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
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
            print(f'    LoRA+InfoNCE epoch {epoch+1}/{epochs}')
    return model


# ─────────────────────────────────────────────
# Evaluation
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


def run_evaluation(features, labels, label_text, k_list=[1, 3, 5, 10]):
    print(f'\n=== {label_text} ===')
    
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
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
# Main
# ─────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')
    
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    # Load ImageNet-1000 features
    features = np.load(os.path.join(embed_dir, 'imagenet1000_features.npy')).astype(np.float32)
    labels = np.load(os.path.join(embed_dir, 'imagenet1000_labels.npy'))
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    print(f'Total samples: {len(features)}, Total classes: {len(np.unique(labels))}')
    
    # Split: 80% classes for training, 20% classes for testing (unseen)
    _, test_idx = train_test_split(
        np.arange(len(features)), 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    test_features = features[test_idx]
    test_labels = labels[test_idx]
    
    print(f'Test samples (unseen classes): {len(test_features)}')
    print(f'Test classes: {len(np.unique(test_labels))}\n')
    
    dim = features.shape[1]
    
    # Use all data for training (or you can use train split)
    train_features = features
    train_labels = labels
    
    results = {}
    
    # 1. ICon
    print('=' * 70)
    print('Training ICon...')
    icon_model = train_icon(train_features, train_labels, device, dim, epochs=100)
    icon_model.eval()
    with torch.no_grad():
        icon_out = icon_model(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    run_evaluation(icon_out, test_labels, 'ICon (Unseen Classes)')
    
    # 2. SRL
    print('=' * 70)
    print('Training SRL...')
    srl_model = train_srl(train_features, train_labels, device, dim, epochs=100)
    srl_model.eval()
    with torch.no_grad():
        srl_out = srl_model(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    run_evaluation(srl_out, test_labels, 'SRL (Unseen Classes)')
    
    # 3. LoRA+Triplet (rank=128)
    print('=' * 70)
    print('Training LoRA+Triplet (rank=128)...')
    lora_triplet = train_lora_triplet(train_features, train_labels, device, dim, epochs=150, rank=128)
    lora_triplet.eval()
    with torch.no_grad():
        lora_triplet_out = lora_triplet(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    run_evaluation(lora_triplet_out, test_labels, 'LoRA+Triplet r=128 (Unseen Classes)')
    
    # 4. LoRA+InfoNCE (rank=128)
    print('=' * 70)
    print('Training LoRA+InfoNCE (rank=128)...')
    lora_infonce = train_lora_infonce(train_features, train_labels, device, dim, epochs=150, rank=128)
    lora_infonce.eval()
    with torch.no_grad():
        lora_infonce_out = lora_infonce(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    run_evaluation(lora_infonce_out, test_labels, 'LoRA+InfoNCE r=128 (Unseen Classes)')
    
    print('\n' + '=' * 70)
    print('All baselines completed!')
    print('=' * 70)


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/32_imagenet_eval_adapters.py 
# Device: cuda

# Total samples: 34745, Total classes: 1000
# Test samples (unseen classes): 6949
# Test classes: 1000

# ======================================================================
# Training ICon...
#     ICon epoch 20/100
#     ICon epoch 40/100
#     ICon epoch 60/100
#     ICon epoch 80/100
#     ICon epoch 100/100

# === ICon (Unseen Classes) ===

# K=1
#   nprobe 1: LP = 0.7480 | Recall = 0.8435
#   nprobe 5: LP = 0.7952 | Recall = 1.0000
#   nprobe 10: LP = 0.7952 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.6555 | Recall = 0.8262
#   nprobe 5: LP = 0.7338 | Recall = 0.9990
#   nprobe 10: LP = 0.7334 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.5540 | Recall = 0.8113
#   nprobe 5: LP = 0.6384 | Recall = 0.9982
#   nprobe 10: LP = 0.6388 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.3593 | Recall = 0.7848
#   nprobe 5: LP = 0.4308 | Recall = 0.9963
#   nprobe 10: LP = 0.4310 | Recall = 1.0000
# ======================================================================
# Training SRL...
#     SRL epoch 20/100
#     SRL epoch 40/100
#     SRL epoch 60/100
#     SRL epoch 80/100
#     SRL epoch 100/100

# === SRL (Unseen Classes) ===

# K=1
#   nprobe 1: LP = 0.6899 | Recall = 0.7296
#   nprobe 5: LP = 0.7860 | Recall = 0.9839
#   nprobe 10: LP = 0.7883 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.5652 | Recall = 0.6993
#   nprobe 5: LP = 0.6993 | Recall = 0.9772
#   nprobe 10: LP = 0.7081 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.4699 | Recall = 0.6808
#   nprobe 5: LP = 0.6075 | Recall = 0.9709
#   nprobe 10: LP = 0.6175 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.3051 | Recall = 0.6340
#   nprobe 5: LP = 0.4089 | Recall = 0.9520
#   nprobe 10: LP = 0.4174 | Recall = 1.0000
# ======================================================================
# Training LoRA+Triplet (rank=128)...
#     LoRA+Triplet epoch 30/150
#     LoRA+Triplet epoch 60/150
#     LoRA+Triplet epoch 90/150
#     LoRA+Triplet epoch 120/150
#     LoRA+Triplet epoch 150/150

# === LoRA+Triplet r=128 (Unseen Classes) ===

# K=1
#   nprobe 1: LP = 0.4028 | Recall = 0.8723
#   nprobe 5: LP = 0.4229 | Recall = 1.0000
#   nprobe 10: LP = 0.4229 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.3234 | Recall = 0.8677
#   nprobe 5: LP = 0.3481 | Recall = 0.9992
#   nprobe 10: LP = 0.3485 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.2654 | Recall = 0.8634
#   nprobe 5: LP = 0.2886 | Recall = 0.9985
#   nprobe 10: LP = 0.2891 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.1872 | Recall = 0.8471
#   nprobe 5: LP = 0.2084 | Recall = 0.9982
#   nprobe 10: LP = 0.2089 | Recall = 1.0000
# ======================================================================
# Training LoRA+InfoNCE (rank=128)...
#     LoRA+InfoNCE epoch 30/150
#     LoRA+InfoNCE epoch 60/150
#     LoRA+InfoNCE epoch 90/150
#     LoRA+InfoNCE epoch 120/150
#     LoRA+InfoNCE epoch 150/150

# === LoRA+InfoNCE r=128 (Unseen Classes) ===

# K=1
#   nprobe 1: LP = 0.4620 | Recall = 0.8360
#   nprobe 5: LP = 0.4983 | Recall = 0.9994
#   nprobe 10: LP = 0.4983 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.3834 | Recall = 0.8236
#   nprobe 5: LP = 0.4296 | Recall = 0.9981
#   nprobe 10: LP = 0.4294 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.3182 | Recall = 0.8152
#   nprobe 5: LP = 0.3616 | Recall = 0.9974
#   nprobe 10: LP = 0.3619 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.2201 | Recall = 0.7999
#   nprobe 5: LP = 0.2554 | Recall = 0.9964
#   nprobe 10: LP = 0.2558 | Recall = 1.0000

# ======================================================================
# All baselines completed!
# ======================================================================
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 