# imagenet_eval_unified.py
# Usage:
#   python imagenet_eval_unified.py --mode ood --seed 42
#   python imagenet_eval_unified.py --mode ood --seed 123
#   python imagenet_eval_unified.py --mode ood --seed 456

import argparse
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
# LoRA Adapter
# ─────────────────────────────────────────────

class LoRAAdapter(nn.Module):
    def __init__(self, dim=512, rank=128, alpha=16):
        super().__init__()
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
        self.t = temperature
    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.t
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        log_probs = F.log_softmax(sim, dim=1)
        target = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return F.kl_div(log_probs, target, reduction='batchmean')


class SRLLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_gen=0.5):
        super().__init__()
        self.t = temperature
        self.lambda_gen = lambda_gen
    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        bs = features.shape[0]
        logits = torch.matmul(features, features.T) / self.t
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(features.device)
        eye = torch.eye(bs).to(features.device)
        exp_logits = torch.exp(logits) * (1 - eye)
        uniformity = torch.log(exp_logits.sum(dim=1) + 1e-8).mean()
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
        pos_indices = self.label_to_indices[a_label]
        p_idx = np.random.choice(pos_indices)
        while p_idx == idx and len(pos_indices) > 1:
            p_idx = np.random.choice(pos_indices)
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
# Training
# ─────────────────────────────────────────────

def train_loss_based(features, labels, device, dim, loss_fn, model_class, lr, epochs, batch_size=512, name=''):
    loader = DataLoader(StandardDataset(features, labels), batch_size=batch_size, shuffle=True)
    model = model_class().to(device) if callable(model_class) else model_class.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for epoch in range(epochs):
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(feats), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f'    {name} epoch {epoch+1}/{epochs}')
    return model


def train_triplet(features, labels, device, dim, model_class, lr, epochs, batch_size=512, name=''):
    loader = DataLoader(TripletDataset(features, labels), batch_size=batch_size, shuffle=True)
    model = model_class().to(device) if callable(model_class) else model_class.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        if (epoch + 1) % 20 == 0:
            print(f'    {name} epoch {epoch+1}/{epochs}')
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(features, labels, name, eval_seed, k_list=(1, 3, 5, 10)):
    print(f'\n=== {name} ===')
    np.random.seed(eval_seed)
    perm = np.random.permutation(len(features))
    features = features[perm]
    labels = labels[perm]
    split = int(len(features) * 0.75)
    base, base_labels = features[:split], labels[:split]
    query, query_labels = features[split:], labels[split:]
    dim = base.shape[1]
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))
    nlist = 10
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)
    for k in k_list:
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            recall = np.mean([len(np.intersect1d(ret[i, :k], gt[i, :k])) for i in range(len(gt))]) / k
            lp_count = sum(np.sum(base_labels[ret[i, :k]] == query_labels[i]) for i in range(len(query_labels)))
            lp = lp_count / (len(query_labels) * k)
            if k == 1 and nprobe == 1:
                print(f'  K={k}, nprobe={nprobe}: LP={lp:.4f}, Recall={recall:.4f}  ← headline')
            else:
                print(f'  K={k}, nprobe={nprobe}: LP={lp:.4f}, Recall={recall:.4f}')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['id', 'ood'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed controlling: (1) class split (OOD mode), '
                             '(2) sample split (ID mode), (3) torch random init, '
                             '(4) numpy RNG for triplet sampling, (5) eval permutation.')
    args = parser.parse_args()

    # CRITICAL: seed everything for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Mode: {args.mode.upper()} | Seed: {args.seed} | Device: {device} | Epochs: {args.epochs}')

    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')

    features = np.load(os.path.join(embed_dir, 'imagenet1000_features.npy')).astype(np.float32)
    labels = np.load(os.path.join(embed_dir, 'imagenet1000_labels.npy'))
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    print(f'Loaded: {len(features)} samples, {len(np.unique(labels))} classes')

    # ═══════════════════════════════════════════════════════════
    # Split logic by mode (uses --seed)
    # ═══════════════════════════════════════════════════════════
    if args.mode == 'id':
        np.random.seed(args.seed)
        perm = np.random.permutation(len(features))
        split_idx = int(0.8 * len(features))
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        train_classes = set(np.unique(train_labels))
        test_classes = set(np.unique(test_labels))
        assert len(train_classes & test_classes) > 0, "ID setup broken: no overlap"
        print(f'ID setup: {len(train_features)} train samples ({len(train_classes)} classes), '
              f'{len(test_features)} test samples ({len(test_classes)} classes)')
        print(f'  Class overlap: {len(train_classes & test_classes)}')

    elif args.mode == 'ood':
        all_classes = np.unique(labels)
        np.random.seed(args.seed)
        np.random.shuffle(all_classes)
        n_train_classes = int(0.8 * len(all_classes))
        seen_classes = set(all_classes[:n_train_classes])
        unseen_classes = set(all_classes[n_train_classes:])
        train_mask = np.isin(labels, list(seen_classes))
        test_mask = np.isin(labels, list(unseen_classes))
        train_features = features[train_mask]
        train_labels = labels[train_mask]
        test_features = features[test_mask]
        test_labels = labels[test_mask]
        train_classes_actual = set(np.unique(train_labels))
        test_classes_actual = set(np.unique(test_labels))
        overlap = train_classes_actual & test_classes_actual
        assert len(overlap) == 0, f"OOD setup BROKEN: class overlap = {overlap}"
        print(f'OOD setup: {len(train_features)} train samples ({len(train_classes_actual)} classes), '
              f'{len(test_features)} test samples ({len(test_classes_actual)} classes)')
        print(f'  Class overlap: 0 (verified disjoint ✓)')

    # Restore seed AFTER split, so training stochasticity also uses --seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dim = features.shape[1]
    EPOCHS = args.epochs

    # ═══════════════════════════════════════════════════════════
    # Train & evaluate
    # ═══════════════════════════════════════════════════════════

    print('\n' + '=' * 70)
    print('CLIP (frozen baseline)')
    evaluate(test_features, test_labels, 'CLIP (frozen)', eval_seed=args.seed)

    print('\n' + '=' * 70)
    print('Training ICon...')
    icon_model = train_loss_based(
        train_features, train_labels, device, dim,
        loss_fn=IConLoss().to(device),
        model_class=lambda: EGAMLP(input_dim=dim),
        lr=1e-4, epochs=EPOCHS, name='ICon'
    )
    icon_model.eval()
    with torch.no_grad():
        out = icon_model(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    evaluate(out, test_labels, 'ICon', eval_seed=args.seed)

    print('\n' + '=' * 70)
    print('Training SRL...')
    srl_model = train_loss_based(
        train_features, train_labels, device, dim,
        loss_fn=SRLLoss().to(device),
        model_class=lambda: EGAMLP(input_dim=dim),
        lr=1e-4, epochs=EPOCHS, name='SRL'
    )
    srl_model.eval()
    with torch.no_grad():
        out = srl_model(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    evaluate(out, test_labels, 'SRL', eval_seed=args.seed)

    print('\n' + '=' * 70)
    print('Training LoRA+InfoNCE (r=128)...')
    lora_info = train_loss_based(
        train_features, train_labels, device, dim,
        loss_fn=SupConInfoNCE().to(device),
        model_class=lambda: LoRAAdapter(dim=dim, rank=128),
        lr=1e-3, epochs=EPOCHS, name='LoRA+InfoNCE'
    )
    lora_info.eval()
    with torch.no_grad():
        out = lora_info(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    evaluate(out, test_labels, 'LoRA+InfoNCE r=128', eval_seed=args.seed)

    print('\n' + '=' * 70)
    print('Training LoRA+Triplet (r=128)...')
    lora_trip = train_triplet(
        train_features, train_labels, device, dim,
        model_class=lambda: LoRAAdapter(dim=dim, rank=128),
        lr=1e-3, epochs=EPOCHS, name='LoRA+Triplet'
    )
    lora_trip.eval()
    with torch.no_grad():
        out = lora_trip(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    evaluate(out, test_labels, 'LoRA+Triplet r=128', eval_seed=args.seed)

    print('\n' + '=' * 70)
    print('Training EGA...')
    ega_model = train_triplet(
        train_features, train_labels, device, dim,
        model_class=lambda: EGAMLP(input_dim=dim),
        lr=1e-4, epochs=EPOCHS, name='EGA'
    )
    ega_model.eval()
    with torch.no_grad():
        out = ega_model(torch.from_numpy(test_features).float().to(device)).cpu().numpy()
    evaluate(out, test_labels, 'EGA', eval_seed=args.seed)

    save_path = os.path.join(base_dir, f'models/ega_imagenet1000_{args.mode}_{EPOCHS}ep_seed{args.seed}.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ega_model.state_dict(), save_path)
    print(f'\nEGA model saved: {save_path}')

    print('\n' + '=' * 70)
    print(f'{args.mode.upper()} mode, seed={args.seed} completed.')
    print('=' * 70)


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/35_imagenet_all.py --mode ood --seed 42
# Mode: OOD | Seed: 42 | Device: cuda | Epochs: 100
# Loaded: 34745 samples, 1000 classes
# OOD setup: 27831 train samples (800 classes), 6914 test samples (200 classes)
#   Class overlap: 0 (verified disjoint ✓)

# ======================================================================
# CLIP (frozen baseline)

# === CLIP (frozen) ===
#   K=1, nprobe=1: LP=0.6836, Recall=0.8288  ← headline
#   K=1, nprobe=5: LP=0.7166, Recall=0.9971
#   K=1, nprobe=10: LP=0.7154, Recall=1.0000
#   K=3, nprobe=1: LP=0.6418, Recall=0.8053
#   K=3, nprobe=5: LP=0.6755, Recall=0.9958
#   K=3, nprobe=10: LP=0.6751, Recall=1.0000
#   K=5, nprobe=1: LP=0.6050, Recall=0.7932
#   K=5, nprobe=5: LP=0.6496, Recall=0.9954
#   K=5, nprobe=10: LP=0.6493, Recall=1.0000
#   K=10, nprobe=1: LP=0.5267, Recall=0.7730
#   K=10, nprobe=5: LP=0.5839, Recall=0.9936
#   K=10, nprobe=10: LP=0.5842, Recall=1.0000

# ======================================================================
# Training ICon...
#     ICon epoch 20/100
#     ICon epoch 40/100
#     ICon epoch 60/100
#     ICon epoch 80/100
#     ICon epoch 100/100

# === ICon ===
#   K=1, nprobe=1: LP=0.6125, Recall=0.8305  ← headline
#   K=1, nprobe=5: LP=0.6368, Recall=0.9948
#   K=1, nprobe=10: LP=0.6362, Recall=1.0000
#   K=3, nprobe=1: LP=0.5724, Recall=0.8194
#   K=3, nprobe=5: LP=0.6002, Recall=0.9958
#   K=3, nprobe=10: LP=0.6005, Recall=1.0000
#   K=5, nprobe=1: LP=0.5403, Recall=0.8096
#   K=5, nprobe=5: LP=0.5776, Recall=0.9950
#   K=5, nprobe=10: LP=0.5781, Recall=1.0000
#   K=10, nprobe=1: LP=0.4881, Recall=0.7977
#   K=10, nprobe=5: LP=0.5371, Recall=0.9943
#   K=10, nprobe=10: LP=0.5382, Recall=1.0000

# ======================================================================
# Training SRL...
#     SRL epoch 20/100
#     SRL epoch 40/100
#     SRL epoch 60/100
#     SRL epoch 80/100
#     SRL epoch 100/100

# === SRL ===
#   K=1, nprobe=1: LP=0.6848, Recall=0.7577  ← headline
#   K=1, nprobe=5: LP=0.7154, Recall=0.9780
#   K=1, nprobe=10: LP=0.7172, Recall=1.0000
#   K=3, nprobe=1: LP=0.6347, Recall=0.7465
#   K=3, nprobe=5: LP=0.6854, Recall=0.9749
#   K=3, nprobe=10: LP=0.6873, Recall=1.0000
#   K=5, nprobe=1: LP=0.5927, Recall=0.7377
#   K=5, nprobe=5: LP=0.6586, Recall=0.9733
#   K=5, nprobe=10: LP=0.6612, Recall=1.0000
#   K=10, nprobe=1: LP=0.5236, Recall=0.7202
#   K=10, nprobe=5: LP=0.6112, Recall=0.9690
#   K=10, nprobe=10: LP=0.6146, Recall=1.0000

# ======================================================================
# Training LoRA+InfoNCE (r=128)...
#     LoRA+InfoNCE epoch 20/100
#     LoRA+InfoNCE epoch 40/100
#     LoRA+InfoNCE epoch 60/100
#     LoRA+InfoNCE epoch 80/100
#     LoRA+InfoNCE epoch 100/100

# === LoRA+InfoNCE r=128 ===
#   K=1, nprobe=1: LP=0.7548, Recall=0.8762  ← headline
#   K=1, nprobe=5: LP=0.7756, Recall=0.9994
#   K=1, nprobe=10: LP=0.7762, Recall=1.0000
#   K=3, nprobe=1: LP=0.7220, Recall=0.8699
#   K=3, nprobe=5: LP=0.7501, Recall=0.9988
#   K=3, nprobe=10: LP=0.7500, Recall=1.0000
#   K=5, nprobe=1: LP=0.6945, Recall=0.8612
#   K=5, nprobe=5: LP=0.7284, Recall=0.9980
#   K=5, nprobe=10: LP=0.7289, Recall=1.0000
#   K=10, nprobe=1: LP=0.6424, Recall=0.8511
#   K=10, nprobe=5: LP=0.6853, Recall=0.9978
#   K=10, nprobe=10: LP=0.6859, Recall=1.0000

# ======================================================================
# Training LoRA+Triplet (r=128)...
#     LoRA+Triplet epoch 20/100
#     LoRA+Triplet epoch 40/100
#     LoRA+Triplet epoch 60/100
#     LoRA+Triplet epoch 80/100
#     LoRA+Triplet epoch 100/100

# === LoRA+Triplet r=128 ===
#   K=1, nprobe=1: LP=0.7212, Recall=0.8930  ← headline
#   K=1, nprobe=5: LP=0.7415, Recall=0.9988
#   K=1, nprobe=10: LP=0.7415, Recall=1.0000
#   K=3, nprobe=1: LP=0.6831, Recall=0.8805
#   K=3, nprobe=5: LP=0.7087, Recall=0.9994
#   K=3, nprobe=10: LP=0.7087, Recall=1.0000
#   K=5, nprobe=1: LP=0.6524, Recall=0.8740
#   K=5, nprobe=5: LP=0.6809, Recall=0.9985
#   K=5, nprobe=10: LP=0.6813, Recall=1.0000
#   K=10, nprobe=1: LP=0.5858, Recall=0.8661
#   K=10, nprobe=5: LP=0.6225, Recall=0.9987
#   K=10, nprobe=10: LP=0.6230, Recall=1.0000

# ======================================================================
# Training EGA...
#     EGA epoch 20/100
#     EGA epoch 40/100
#     EGA epoch 60/100
#     EGA epoch 80/100
#     EGA epoch 100/100

# === EGA ===
#   K=1, nprobe=1: LP=0.7189, Recall=0.8606  ← headline
#   K=1, nprobe=5: LP=0.7420, Recall=0.9971
#   K=1, nprobe=10: LP=0.7415, Recall=1.0000
#   K=3, nprobe=1: LP=0.6713, Recall=0.8610
#   K=3, nprobe=5: LP=0.7058, Recall=0.9973
#   K=3, nprobe=10: LP=0.7060, Recall=1.0000
#   K=5, nprobe=1: LP=0.6405, Recall=0.8558
#   K=5, nprobe=5: LP=0.6777, Recall=0.9971
#   K=5, nprobe=10: LP=0.6777, Recall=1.0000
#   K=10, nprobe=1: LP=0.5868, Recall=0.8456
#   K=10, nprobe=5: LP=0.6341, Recall=0.9965
#   K=10, nprobe=10: LP=0.6344, Recall=1.0000

# EGA model saved: /home/cc/hpdic/EGA/models/ega_imagenet1000_ood_100ep_seed42.pth

# ======================================================================
# OOD mode, seed=42 completed.
# ======================================================================
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 

# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/35_imagenet_all.py --mode ood --seed 123
# python scripts/35_imagenet_all.py --mode ood --seed 456
# Mode: OOD | Seed: 123 | Device: cuda | Epochs: 100
# Loaded: 34745 samples, 1000 classes
# OOD setup: 27757 train samples (800 classes), 6988 test samples (200 classes)
#   Class overlap: 0 (verified disjoint ✓)

# ======================================================================
# CLIP (frozen baseline)

# === CLIP (frozen) ===
#   K=1, nprobe=1: LP=0.6800, Recall=0.8512  ← headline
#   K=1, nprobe=5: LP=0.7041, Recall=0.9914
#   K=1, nprobe=10: LP=0.7035, Recall=1.0000
#   K=3, nprobe=1: LP=0.6276, Recall=0.8384
#   K=3, nprobe=5: LP=0.6621, Recall=0.9929
#   K=3, nprobe=10: LP=0.6632, Recall=1.0000
#   K=5, nprobe=1: LP=0.5878, Recall=0.8331
#   K=5, nprobe=5: LP=0.6277, Recall=0.9926
#   K=5, nprobe=10: LP=0.6293, Recall=1.0000
#   K=10, nprobe=1: LP=0.5169, Recall=0.8164
#   K=10, nprobe=5: LP=0.5649, Recall=0.9914
#   K=10, nprobe=10: LP=0.5667, Recall=1.0000

# ======================================================================
# Training ICon...
#     ICon epoch 20/100
#     ICon epoch 40/100
#     ICon epoch 60/100
#     ICon epoch 80/100
#     ICon epoch 100/100

# === ICon ===
#   K=1, nprobe=1: LP=0.5924, Recall=0.8409  ← headline
#   K=1, nprobe=5: LP=0.6222, Recall=0.9948
#   K=1, nprobe=10: LP=0.6234, Recall=1.0000
#   K=3, nprobe=1: LP=0.5564, Recall=0.8386
#   K=3, nprobe=5: LP=0.5869, Recall=0.9960
#   K=3, nprobe=10: LP=0.5890, Recall=1.0000
#   K=5, nprobe=1: LP=0.5327, Recall=0.8330
#   K=5, nprobe=5: LP=0.5674, Recall=0.9945
#   K=5, nprobe=10: LP=0.5685, Recall=1.0000
#   K=10, nprobe=1: LP=0.4864, Recall=0.8230
#   K=10, nprobe=5: LP=0.5232, Recall=0.9934
#   K=10, nprobe=10: LP=0.5244, Recall=1.0000

# ======================================================================
# Training SRL...
#     SRL epoch 20/100
#     SRL epoch 40/100
#     SRL epoch 60/100
#     SRL epoch 80/100
#     SRL epoch 100/100

# === SRL ===
#   K=1, nprobe=1: LP=0.6709, Recall=0.7859  ← headline
#   K=1, nprobe=5: LP=0.6961, Recall=0.9863
#   K=1, nprobe=10: LP=0.6978, Recall=1.0000
#   K=3, nprobe=1: LP=0.6224, Recall=0.7659
#   K=3, nprobe=5: LP=0.6703, Recall=0.9781
#   K=3, nprobe=10: LP=0.6728, Recall=1.0000
#   K=5, nprobe=1: LP=0.5906, Recall=0.7630
#   K=5, nprobe=5: LP=0.6501, Recall=0.9778
#   K=5, nprobe=10: LP=0.6506, Recall=1.0000
#   K=10, nprobe=1: LP=0.5255, Recall=0.7515
#   K=10, nprobe=5: LP=0.5996, Recall=0.9758
#   K=10, nprobe=10: LP=0.6026, Recall=1.0000

# ======================================================================
# Training LoRA+InfoNCE (r=128)...
#     LoRA+InfoNCE epoch 20/100
#     LoRA+InfoNCE epoch 40/100
#     LoRA+InfoNCE epoch 60/100
#     LoRA+InfoNCE epoch 80/100
#     LoRA+InfoNCE epoch 100/100

# === LoRA+InfoNCE r=128 ===
#   K=1, nprobe=1: LP=0.7504, Recall=0.8832  ← headline
#   K=1, nprobe=5: LP=0.7705, Recall=0.9994
#   K=1, nprobe=10: LP=0.7705, Recall=1.0000
#   K=3, nprobe=1: LP=0.7090, Recall=0.8777
#   K=3, nprobe=5: LP=0.7350, Recall=0.9989
#   K=3, nprobe=10: LP=0.7350, Recall=1.0000
#   K=5, nprobe=1: LP=0.6813, Recall=0.8728
#   K=5, nprobe=5: LP=0.7148, Recall=0.9978
#   K=5, nprobe=10: LP=0.7148, Recall=1.0000
#   K=10, nprobe=1: LP=0.6243, Recall=0.8667
#   K=10, nprobe=5: LP=0.6674, Recall=0.9969
#   K=10, nprobe=10: LP=0.6677, Recall=1.0000

# ======================================================================
# Training LoRA+Triplet (r=128)...
#     LoRA+Triplet epoch 20/100
#     LoRA+Triplet epoch 40/100
#     LoRA+Triplet epoch 60/100
#     LoRA+Triplet epoch 80/100
#     LoRA+Triplet epoch 100/100

# === LoRA+Triplet r=128 ===
#   K=1, nprobe=1: LP=0.7069, Recall=0.8764  ← headline
#   K=1, nprobe=5: LP=0.7258, Recall=1.0000
#   K=1, nprobe=10: LP=0.7258, Recall=1.0000
#   K=3, nprobe=1: LP=0.6629, Recall=0.8643
#   K=3, nprobe=5: LP=0.6903, Recall=0.9989
#   K=3, nprobe=10: LP=0.6907, Recall=1.0000
#   K=5, nprobe=1: LP=0.6276, Recall=0.8604
#   K=5, nprobe=5: LP=0.6618, Recall=0.9983
#   K=5, nprobe=10: LP=0.6618, Recall=1.0000
#   K=10, nprobe=1: LP=0.5700, Recall=0.8503
#   K=10, nprobe=5: LP=0.6137, Recall=0.9977
#   K=10, nprobe=10: LP=0.6143, Recall=1.0000

# ======================================================================
# Training EGA...
#     EGA epoch 20/100
#     EGA epoch 40/100
#     EGA epoch 60/100
#     EGA epoch 80/100
#     EGA epoch 100/100

# === EGA ===
#   K=1, nprobe=1: LP=0.7035, Recall=0.8586  ← headline
#   K=1, nprobe=5: LP=0.7344, Recall=0.9966
#   K=1, nprobe=10: LP=0.7344, Recall=1.0000
#   K=3, nprobe=1: LP=0.6651, Recall=0.8556
#   K=3, nprobe=5: LP=0.6970, Recall=0.9971
#   K=3, nprobe=10: LP=0.6970, Recall=1.0000
#   K=5, nprobe=1: LP=0.6316, Recall=0.8501
#   K=5, nprobe=5: LP=0.6664, Recall=0.9970
#   K=5, nprobe=10: LP=0.6671, Recall=1.0000
#   K=10, nprobe=1: LP=0.5726, Recall=0.8450
#   K=10, nprobe=5: LP=0.6179, Recall=0.9968
#   K=10, nprobe=10: LP=0.6183, Recall=1.0000

# EGA model saved: /home/cc/hpdic/EGA/models/ega_imagenet1000_ood_100ep_seed123.pth

# ======================================================================
# OOD mode, seed=123 completed.
# ======================================================================
# Mode: OOD | Seed: 456 | Device: cuda | Epochs: 100
# Loaded: 34745 samples, 1000 classes
# OOD setup: 27715 train samples (800 classes), 7030 test samples (200 classes)
#   Class overlap: 0 (verified disjoint ✓)

# ======================================================================
# CLIP (frozen baseline)

# === CLIP (frozen) ===
#   K=1, nprobe=1: LP=0.6615, Recall=0.8367  ← headline
#   K=1, nprobe=5: LP=0.6923, Recall=0.9989
#   K=1, nprobe=10: LP=0.6934, Recall=1.0000
#   K=3, nprobe=1: LP=0.6250, Recall=0.8172
#   K=3, nprobe=5: LP=0.6625, Recall=0.9979
#   K=3, nprobe=10: LP=0.6631, Recall=1.0000
#   K=5, nprobe=1: LP=0.5865, Recall=0.8089
#   K=5, nprobe=5: LP=0.6354, Recall=0.9969
#   K=5, nprobe=10: LP=0.6359, Recall=1.0000
#   K=10, nprobe=1: LP=0.5135, Recall=0.7941
#   K=10, nprobe=5: LP=0.5767, Recall=0.9953
#   K=10, nprobe=10: LP=0.5774, Recall=1.0000

# ======================================================================
# Training ICon...
#     ICon epoch 20/100
#     ICon epoch 40/100
#     ICon epoch 60/100
#     ICon epoch 80/100
#     ICon epoch 100/100

# === ICon ===
#   K=1, nprobe=1: LP=0.5586, Recall=0.8305  ← headline
#   K=1, nprobe=5: LP=0.5825, Recall=0.9960
#   K=1, nprobe=10: LP=0.5830, Recall=1.0000
#   K=3, nprobe=1: LP=0.5284, Recall=0.8284
#   K=3, nprobe=5: LP=0.5565, Recall=0.9964
#   K=3, nprobe=10: LP=0.5571, Recall=1.0000
#   K=5, nprobe=1: LP=0.5078, Recall=0.8187
#   K=5, nprobe=5: LP=0.5398, Recall=0.9958
#   K=5, nprobe=10: LP=0.5406, Recall=1.0000
#   K=10, nprobe=1: LP=0.4633, Recall=0.8040
#   K=10, nprobe=5: LP=0.5007, Recall=0.9949
#   K=10, nprobe=10: LP=0.5012, Recall=1.0000

# ======================================================================
# Training SRL...
#     SRL epoch 20/100
#     SRL epoch 40/100
#     SRL epoch 60/100
#     SRL epoch 80/100
#     SRL epoch 100/100

# === SRL ===
#   K=1, nprobe=1: LP=0.6428, Recall=0.7838  ← headline
#   K=1, nprobe=5: LP=0.6769, Recall=0.9846
#   K=1, nprobe=10: LP=0.6809, Recall=1.0000
#   K=3, nprobe=1: LP=0.6052, Recall=0.7783
#   K=3, nprobe=5: LP=0.6483, Recall=0.9860
#   K=3, nprobe=10: LP=0.6496, Recall=1.0000
#   K=5, nprobe=1: LP=0.5784, Recall=0.7697
#   K=5, nprobe=5: LP=0.6328, Recall=0.9843
#   K=5, nprobe=10: LP=0.6333, Recall=1.0000
#   K=10, nprobe=1: LP=0.5188, Recall=0.7559
#   K=10, nprobe=5: LP=0.5884, Recall=0.9796
#   K=10, nprobe=10: LP=0.5904, Recall=1.0000

# ======================================================================
# Training LoRA+InfoNCE (r=128)...
#     LoRA+InfoNCE epoch 20/100
#     LoRA+InfoNCE epoch 40/100
#     LoRA+InfoNCE epoch 60/100
#     LoRA+InfoNCE epoch 80/100
#     LoRA+InfoNCE epoch 100/100

# === LoRA+InfoNCE r=128 ===
#   K=1, nprobe=1: LP=0.7349, Recall=0.8714  ← headline
#   K=1, nprobe=5: LP=0.7554, Recall=0.9994
#   K=1, nprobe=10: LP=0.7554, Recall=1.0000
#   K=3, nprobe=1: LP=0.6951, Recall=0.8646
#   K=3, nprobe=5: LP=0.7266, Recall=0.9979
#   K=3, nprobe=10: LP=0.7266, Recall=1.0000
#   K=5, nprobe=1: LP=0.6712, Recall=0.8636
#   K=5, nprobe=5: LP=0.7059, Recall=0.9981
#   K=5, nprobe=10: LP=0.7060, Recall=1.0000
#   K=10, nprobe=1: LP=0.6160, Recall=0.8546
#   K=10, nprobe=5: LP=0.6614, Recall=0.9975
#   K=10, nprobe=10: LP=0.6619, Recall=1.0000

# ======================================================================
# Training LoRA+Triplet (r=128)...
#     LoRA+Triplet epoch 20/100
#     LoRA+Triplet epoch 40/100
#     LoRA+Triplet epoch 60/100
#     LoRA+Triplet epoch 80/100
#     LoRA+Triplet epoch 100/100

# === LoRA+Triplet r=128 ===
#   K=1, nprobe=1: LP=0.6832, Recall=0.8811  ← headline
#   K=1, nprobe=5: LP=0.7048, Recall=0.9994
#   K=1, nprobe=10: LP=0.7048, Recall=1.0000
#   K=3, nprobe=1: LP=0.6492, Recall=0.8754
#   K=3, nprobe=5: LP=0.6750, Recall=0.9992
#   K=3, nprobe=10: LP=0.6752, Recall=1.0000
#   K=5, nprobe=1: LP=0.6220, Recall=0.8714
#   K=5, nprobe=5: LP=0.6529, Recall=0.9993
#   K=5, nprobe=10: LP=0.6532, Recall=1.0000
#   K=10, nprobe=1: LP=0.5666, Recall=0.8616
#   K=10, nprobe=5: LP=0.6044, Recall=0.9986
#   K=10, nprobe=10: LP=0.6048, Recall=1.0000

# ======================================================================
# Training EGA...
#     EGA epoch 20/100
#     EGA epoch 40/100
#     EGA epoch 60/100
#     EGA epoch 80/100
#     EGA epoch 100/100

# === EGA ===
#   K=1, nprobe=1: LP=0.6820, Recall=0.8908  ← headline
#   K=1, nprobe=5: LP=0.7042, Recall=0.9966
#   K=1, nprobe=10: LP=0.7042, Recall=1.0000
#   K=3, nprobe=1: LP=0.6466, Recall=0.8788
#   K=3, nprobe=5: LP=0.6741, Recall=0.9977
#   K=3, nprobe=10: LP=0.6746, Recall=1.0000
#   K=5, nprobe=1: LP=0.6209, Recall=0.8706
#   K=5, nprobe=5: LP=0.6519, Recall=0.9973
#   K=5, nprobe=10: LP=0.6523, Recall=1.0000
#   K=10, nprobe=1: LP=0.5706, Recall=0.8621
#   K=10, nprobe=5: LP=0.6133, Recall=0.9966
#   K=10, nprobe=10: LP=0.6139, Recall=1.0000

# EGA model saved: /home/cc/hpdic/EGA/models/ega_imagenet1000_ood_100ep_seed456.pth

# ======================================================================
# OOD mode, seed=456 completed.
# ======================================================================
# (venv) (base) cc@uc-a100:~/hpdic/EGA$     