# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>

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

def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_by_class(features, labels, train_ratio=0.8):
    unique_classes = np.unique(labels)
    np.random.seed(42)
    np.random.shuffle(unique_classes)
    
    num_train_classes = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train_classes]
    test_classes = unique_classes[num_train_classes:]
    
    train_mask = np.isin(labels, train_classes)
    test_mask = np.isin(labels, test_classes)
    
    return (features[train_mask], labels[train_mask]), (features[test_mask], labels[test_mask])

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
        anchor = self.features[idx]
        anchor_label = self.labels[idx]
        
        pos_idx = np.random.choice(self.label_to_indices[anchor_label])
        
        neg_label = np.random.choice(self.classes)
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.classes)
            
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        
        return anchor, self.features[pos_idx], self.features[neg_idx]

class StandardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def run_evaluation(features, labels, label_text, k_list):
    print(f'\nEvaluating: {label_text}')
    
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
        print(f'\nResults for K={k}')
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            recall = calculate_anns_recall(ret, gt, k)
            precision = calculate_label_precision(ret, base_labels, query_labels, k)
            print(f'nprobe {nprobe}: Label Precision = {precision:.4f} | ANNS Recall = {recall:.4f}')

def train_ega(train_feats, train_labels, device):
    loader = DataLoader(ClassAwareDataset(train_feats, train_labels), batch_size=256, shuffle=True)
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=150)
    margin = 0.2
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    model.train()
    for epoch in range(150):
        total_active = 0.0
        total_batches = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            out_a = model(a)
            out_p = model(p)
            out_n = model(n)
            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                d_pos = F.pairwise_distance(out_a, out_p)
                d_neg = F.pairwise_distance(out_a, out_n)
                active_ratio = (d_pos - d_neg + margin > 0).float().mean()
                total_active += active_ratio.item()
                total_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/150 | Active triplet ratio: {total_active/total_batches:.3f}")

    return model

def train_contrastive_baseline(train_feats, train_labels, loss_fn, device):
    loader = DataLoader(StandardDataset(train_feats, train_labels), batch_size=256, shuffle=True)
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=150)

    model.train()
    for epoch in range(150):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            out_feats = model(feats)
            loss = loss_fn(out_feats, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model

def main():
    set_global_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    feat_path = os.path.join(embed_dir, 'aircraft_test_vit_b32_features.npy')
    label_path = os.path.join(embed_dir, 'aircraft_test_vit_b32_labels.npy')
    
    features = np.load(feat_path).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)
    
    print('Splitting dataset by class (80% Train, 20% Test)...')
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)
    
    print('1/3 Training Fair EGA...')
    ega_model = train_ega(train_feats, train_labels, device)
    
    print('2/3 Training Fair ICon (MLP Adapter Only)...')
    icon_model = train_contrastive_baseline(train_feats, train_labels, IConLoss().to(device), device)
    
    print('3/3 Training Fair SRL (MLP Adapter Only)...')
    srl_model = train_contrastive_baseline(train_feats, train_labels, SRLLoss().to(device), device)
    
    ega_model.eval()
    icon_model.eval()
    srl_model.eval()
    
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_feats).to(device)
        ega_out = ega_model(test_tensor).cpu().numpy()
        icon_out = icon_model(test_tensor).cpu().numpy()
        srl_out = srl_model(test_tensor).cpu().numpy()
        
    k_list = [1, 3, 5, 10]
    run_evaluation(test_feats, test_labels, 'Original CLIP', k_list)
    run_evaluation(ega_out, test_labels, 'EGA (Fair Adapter)', k_list)
    run_evaluation(icon_out, test_labels, 'ICon (Fair Adapter)', k_list)
    run_evaluation(srl_out, test_labels, 'SRL (Fair Adapter)', k_list)
    
    print('\nExecution completed.')

if __name__ == '__main__':
    main()

# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/Aircraft/train_eval_aircraft.py 
# Splitting dataset by class (80% Train, 20% Test)...
# 1/3 Training Fair EGA...
# Epoch 10/150 | Active triplet ratio: 0.174
# Epoch 20/150 | Active triplet ratio: 0.126
# Epoch 30/150 | Active triplet ratio: 0.116
# Epoch 40/150 | Active triplet ratio: 0.099
# Epoch 50/150 | Active triplet ratio: 0.078
# Epoch 60/150 | Active triplet ratio: 0.070
# Epoch 70/150 | Active triplet ratio: 0.073
# Epoch 80/150 | Active triplet ratio: 0.062
# Epoch 90/150 | Active triplet ratio: 0.052
# Epoch 100/150 | Active triplet ratio: 0.047
# Epoch 110/150 | Active triplet ratio: 0.051
# Epoch 120/150 | Active triplet ratio: 0.046
# Epoch 130/150 | Active triplet ratio: 0.041
# Epoch 140/150 | Active triplet ratio: 0.044
# Epoch 150/150 | Active triplet ratio: 0.035
# 2/3 Training Fair ICon (MLP Adapter Only)...
# 3/3 Training Fair SRL (MLP Adapter Only)...

# Evaluating: Original CLIP

# Results for K=1
# nprobe 1: Label Precision = 0.5119 | ANNS Recall = 0.7738
# nprobe 5: Label Precision = 0.5238 | ANNS Recall = 0.9940
# nprobe 10: Label Precision = 0.5238 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.4782 | ANNS Recall = 0.7857
# nprobe 5: Label Precision = 0.5079 | ANNS Recall = 0.9980
# nprobe 10: Label Precision = 0.5079 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.4536 | ANNS Recall = 0.7786
# nprobe 5: Label Precision = 0.4690 | ANNS Recall = 0.9976
# nprobe 10: Label Precision = 0.4679 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.4095 | ANNS Recall = 0.7685
# nprobe 5: Label Precision = 0.4185 | ANNS Recall = 0.9976
# nprobe 10: Label Precision = 0.4173 | ANNS Recall = 1.0000

# Evaluating: EGA (Fair Adapter)

# Results for K=1
# nprobe 1: Label Precision = 0.5476 | ANNS Recall = 0.8929
# nprobe 5: Label Precision = 0.5536 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5536 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.5337 | ANNS Recall = 0.8929
# nprobe 5: Label Precision = 0.5536 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5536 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.5060 | ANNS Recall = 0.8869
# nprobe 5: Label Precision = 0.5238 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5238 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.4720 | ANNS Recall = 0.8792
# nprobe 5: Label Precision = 0.4905 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.4905 | ANNS Recall = 1.0000

# Evaluating: ICon (Fair Adapter)

# Results for K=1
# nprobe 1: Label Precision = 0.4226 | ANNS Recall = 0.8750
# nprobe 5: Label Precision = 0.4226 | ANNS Recall = 0.9940
# nprobe 10: Label Precision = 0.4226 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.4048 | ANNS Recall = 0.8452
# nprobe 5: Label Precision = 0.4127 | ANNS Recall = 0.9960
# nprobe 10: Label Precision = 0.4127 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.3964 | ANNS Recall = 0.8321
# nprobe 5: Label Precision = 0.4095 | ANNS Recall = 0.9940
# nprobe 10: Label Precision = 0.4131 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.3887 | ANNS Recall = 0.8018
# nprobe 5: Label Precision = 0.3970 | ANNS Recall = 0.9929
# nprobe 10: Label Precision = 0.3994 | ANNS Recall = 1.0000

# Evaluating: SRL (Fair Adapter)

# Results for K=1
# nprobe 1: Label Precision = 0.4405 | ANNS Recall = 0.8631
# nprobe 5: Label Precision = 0.4940 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.4940 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.4167 | ANNS Recall = 0.8135
# nprobe 5: Label Precision = 0.4345 | ANNS Recall = 0.9980
# nprobe 10: Label Precision = 0.4325 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.3976 | ANNS Recall = 0.8095
# nprobe 5: Label Precision = 0.4060 | ANNS Recall = 0.9976
# nprobe 10: Label Precision = 0.4071 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.3696 | ANNS Recall = 0.7857
# nprobe 5: Label Precision = 0.3875 | ANNS Recall = 0.9958
# nprobe 10: Label Precision = 0.3887 | ANNS Recall = 1.0000

# Execution completed.
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 