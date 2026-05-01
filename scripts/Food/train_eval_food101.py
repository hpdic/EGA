# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# Train EGA / ICon / SRL adapters on Food-101 (80 seen / 21 unseen split)
# Uses the real EGAMLP from models/ega_mlp.py (not a simplified version).
# Skips CLIP feature extraction if features/labels already exist.

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

# IMPORTANT: use the real EGAMLP, not a local definition
from models.ega_mlp import EGAMLP


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Class split: 80% seen / 20% unseen
# ─────────────────────────────────────────────

def split_by_class(features, labels, train_ratio=0.8):
    unique_classes = np.unique(labels)
    np.random.seed(42)
    np.random.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes  = unique_classes[num_train:]
    train_mask = np.isin(labels, train_classes)
    test_mask  = np.isin(labels, test_classes)
    return (features[train_mask], labels[train_mask]), \
           (features[test_mask],  labels[test_mask])


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
        anchor       = self.features[idx]
        anchor_label = self.labels[idx]
        pos_idx      = np.random.choice(self.label_to_indices[anchor_label])
        neg_label    = np.random.choice(self.classes)
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.classes)
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        return anchor, self.features[pos_idx], self.features[neg_idx]


class StandardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ─────────────────────────────────────────────
# Loss functions (same as Aircraft script)
# ─────────────────────────────────────────────

class IConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features  = F.normalize(features, p=2, dim=1)
        sim       = torch.matmul(features, features.T) / self.temperature
        labels    = labels.contiguous().view(-1, 1)
        mask      = torch.eq(labels, labels.T).float().to(features.device)
        log_probs = F.log_softmax(sim, dim=1)
        target    = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return F.kl_div(log_probs, target, reduction='batchmean')


class SRLLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_gen=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_gen  = lambda_gen

    def forward(self, features, labels):
        features   = F.normalize(features, p=2, dim=1)
        bs         = features.shape[0]
        logits     = torch.matmul(features, features.T) / self.temperature
        mask       = torch.eq(labels.view(-1, 1),
                              labels.view(1, -1)).float().to(features.device)
        exp_logits = torch.exp(logits) * (1 - torch.eye(bs).to(features.device))
        uniformity = torch.log(exp_logits.sum(dim=1)).mean()
        pos_logits = (exp_logits * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        homog      = -torch.log(pos_logits + 1e-8).mean()
        return homog + self.lambda_gen * uniformity


# ─────────────────────────────────────────────
# Training routines
# ─────────────────────────────────────────────

def train_ega(train_feats, train_labels, device, epochs=150):
    loader    = DataLoader(ClassAwareDataset(train_feats, train_labels),
                           batch_size=256, shuffle=True)
    model     = EGAMLP(input_dim=train_feats.shape[1], hidden_dim=2048).to(device)
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
            print(f'  EGA epoch {epoch+1}/{epochs}')
    return model


def train_contrastive(train_feats, train_labels, loss_fn, device, name, epochs=150):
    loader    = DataLoader(StandardDataset(train_feats, train_labels),
                           batch_size=256, shuffle=True)
    model     = EGAMLP(input_dim=train_feats.shape[1], hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(feats), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f'  {name} epoch {epoch+1}/{epochs}')
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

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


def run_evaluation(features, labels, label_text, k_list=(1, 3, 5, 10),
                   nprobe_list=(1, 5, 10), seed=42):
    print(f'\nEvaluating: {label_text}')
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
            ar = calculate_anns_recall(ret, gt, k)
            lp = calculate_label_precision(ret, base_labels, query_labels, k)
            print(f'    nprobe={np_:>2}: LP={lp:.4f}  AR={ar:.4f}')


# ─────────────────────────────────────────────
# CLIP feature extraction (only if not cached)
# ─────────────────────────────────────────────

def extract_food_features(device, base_dir, embed_dir):
    feat_path  = os.path.join(embed_dir, 'food101_features.npy')
    label_path = os.path.join(embed_dir, 'food101_labels.npy')

    if os.path.exists(feat_path) and os.path.exists(label_path):
        print(f'Loading cached Food-101 features from {feat_path}')
        return np.load(feat_path), np.load(label_path)

    # Lazy imports — only needed if we have to extract from scratch
    import clip
    from torchvision import datasets
    from tqdm import tqdm

    print('Downloading Food-101 and extracting CLIP ViT-B/32 features ...')
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    data_dir = os.path.join(base_dir, 'data')
    dataset  = datasets.Food101(root=data_dir, download=True, split='test')

    all_feats, all_labels = [], []
    with torch.no_grad():
        for img, label in tqdm(dataset):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_tensor)
            all_feats.append(feat.cpu().numpy())
            all_labels.append(label)

    features = np.concatenate(all_feats, axis=0)
    labels   = np.array(all_labels)

    np.save(feat_path,  features)
    np.save(label_path, labels)
    print(f'Saved features to {feat_path} and labels to {label_path}')
    return features, labels


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    set_global_seed(42)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir  = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    os.makedirs(embed_dir, exist_ok=True)

    # ── Load or extract CLIP features ────────────────────────────────
    features, labels = extract_food_features(device, base_dir, embed_dir)
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    print(f'\nTotal samples : {len(labels)}')
    print(f'Total classes : {len(np.unique(labels))}')

    # ── Class split ──────────────────────────────────────────────────
    print('\nSplitting by class (80% train / 20% unseen test) ...')
    (train_feats, train_labels), (test_feats, test_labels) = \
        split_by_class(features, labels)
    print(f'  Train classes : {len(np.unique(train_labels))}, '
          f'Train samples : {len(train_labels)}')
    print(f'  Unseen classes: {len(np.unique(test_labels))}, '
          f'Unseen samples: {len(test_labels)}')

    # ── Train all three adapters ─────────────────────────────────────
    print('\n[1/3] Training EGA ...')
    ega_model = train_ega(train_feats, train_labels, device)

    print('\n[2/3] Training ICon ...')
    icon_model = train_contrastive(train_feats, train_labels,
                                   IConLoss().to(device), device, 'ICon')

    print('\n[3/3] Training SRL ...')
    srl_model = train_contrastive(train_feats, train_labels,
                                  SRLLoss().to(device),  device, 'SRL')

    # ── Transform unseen-class test set ──────────────────────────────
    ega_model.eval(); icon_model.eval(); srl_model.eval()
    with torch.no_grad():
        test_t   = torch.from_numpy(test_feats).to(device)
        ega_out  = ega_model(test_t).cpu().numpy()
        icon_out = icon_model(test_t).cpu().numpy()
        srl_out  = srl_model(test_t).cpu().numpy()

    # ── Save transformed features ────────────────────────────────────
    np.save(os.path.join(embed_dir, 'food101_ega_features.npy'),  ega_out)
    np.save(os.path.join(embed_dir, 'food101_icon_features.npy'), icon_out)
    np.save(os.path.join(embed_dir, 'food101_srl_features.npy'),  srl_out)
    np.save(os.path.join(embed_dir, 'food101_unseen_labels.npy'), test_labels)
    print('\nSaved adapter outputs and unseen-class labels to embeddings/')

    # ── Evaluate all four ────────────────────────────────────────────
    run_evaluation(test_feats, test_labels, 'Original CLIP (Food-101 unseen)')
    run_evaluation(ega_out,    test_labels, 'EGA (Food-101 unseen)')
    run_evaluation(icon_out,   test_labels, 'ICon (Food-101 unseen)')
    run_evaluation(srl_out,    test_labels, 'SRL (Food-101 unseen)')

    print('\nDone.')


if __name__ == '__main__':
    main()