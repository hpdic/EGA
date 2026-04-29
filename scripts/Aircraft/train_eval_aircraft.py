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
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    model.train()
    for epoch in range(150):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        scheduler.step()
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

# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/Aircraft/train_eval_aircraft_split.py 
# Splitting dataset by class (80% Train, 20% Test)...
# Training EGA with Cosine Annealing on 80 known aircraft classes...

# Evaluating: Original CLIP on 20 Unseen Classes

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

# Evaluating: EGA on 20 Unseen Classes

# Results for K=1
# nprobe 1: Label Precision = 0.6071 | ANNS Recall = 0.8988
# nprobe 5: Label Precision = 0.6071 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.6071 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.5675 | ANNS Recall = 0.9028
# nprobe 5: Label Precision = 0.5615 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5615 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.5321 | ANNS Recall = 0.8976
# nprobe 5: Label Precision = 0.5381 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5381 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.5030 | ANNS Recall = 0.8786
# nprobe 5: Label Precision = 0.5054 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5054 | ANNS Recall = 1.0000

# Evaluating: ICon on 20 Unseen Classes

# Results for K=1
# nprobe 1: Label Precision = 0.6190 | ANNS Recall = 0.9345
# nprobe 5: Label Precision = 0.6071 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.6071 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.6012 | ANNS Recall = 0.9286
# nprobe 5: Label Precision = 0.6151 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.6151 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.5631 | ANNS Recall = 0.9179
# nprobe 5: Label Precision = 0.5821 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5821 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.5173 | ANNS Recall = 0.9131
# nprobe 5: Label Precision = 0.5369 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5369 | ANNS Recall = 1.0000

# Evaluating: SRL on 20 Unseen Classes

# Results for K=1
# nprobe 1: Label Precision = 0.5774 | ANNS Recall = 0.9583
# nprobe 5: Label Precision = 0.5774 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5774 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.5496 | ANNS Recall = 0.9365
# nprobe 5: Label Precision = 0.5556 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5556 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.5405 | ANNS Recall = 0.9321
# nprobe 5: Label Precision = 0.5417 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5417 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.4863 | ANNS Recall = 0.9060
# nprobe 5: Label Precision = 0.4976 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.4976 | ANNS Recall = 1.0000

# Execution completed.
# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/Aircraft/train_extract_baselines_aircraft.py 
# Splitting dataset by class (80% Train, 20% Test)...
# 1/3 Training Fair EGA...
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
# nprobe 1: Label Precision = 0.6071 | ANNS Recall = 0.8988
# nprobe 5: Label Precision = 0.6071 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.6071 | ANNS Recall = 1.0000

# Results for K=3
# nprobe 1: Label Precision = 0.5675 | ANNS Recall = 0.9028
# nprobe 5: Label Precision = 0.5615 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5615 | ANNS Recall = 1.0000

# Results for K=5
# nprobe 1: Label Precision = 0.5321 | ANNS Recall = 0.8976
# nprobe 5: Label Precision = 0.5381 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5381 | ANNS Recall = 1.0000

# Results for K=10
# nprobe 1: Label Precision = 0.5030 | ANNS Recall = 0.8786
# nprobe 5: Label Precision = 0.5054 | ANNS Recall = 1.0000
# nprobe 10: Label Precision = 0.5054 | ANNS Recall = 1.0000

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
# (venv) cc@uc-a100:~/hpdic/EGA$     