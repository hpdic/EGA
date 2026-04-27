# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Baseline comparison: Raw-L2 vs Raw-IP vs EGA-L2
# Usage: python eval_baselines.py --features /path/to/features.npy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import faiss
from tqdm import tqdm
import argparse

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class FeatureGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class EGABlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        self.gate = FeatureGating(dim)

    def forward(self, x):
        res = self.conv_branch(x)
        res = self.gate(res)
        return x + res


class EGAMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            EGABlock(input_dim, hidden_dim) for _ in range(num_blocks)
        ])
        self.refiner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.refiner(out)
        return F.normalize(out, p=2, dim=1)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TripletDataset(Dataset):
    def __init__(self, features, neighbors):
        self.features = torch.from_numpy(features).float()
        self.neighbors = neighbors
        self.n = len(features)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        anchor   = self.features[idx]
        pos_idx  = np.random.choice(self.neighbors[idx])
        positive = self.features[pos_idx]
        neg_idx  = np.random.randint(0, self.n)
        while neg_idx in self.neighbors[idx] or neg_idx == idx:
            neg_idx = np.random.randint(0, self.n)
        negative = self.features[neg_idx]
        return anchor, positive, negative


# ─────────────────────────────────────────────
# Ground truth: brute-force cosine (IP on normalized)
# ─────────────────────────────────────────────

def build_ground_truth(features, k=10):
    """
    Returns ground_truth[i] = indices of true top-k neighbors of i,
    excluding self. Uses brute-force inner product (= cosine on normalized).
    Shape: (N, k)
    """
    dim = features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(features)
    _, I = index.search(features, k + 1)  # +1 to exclude self
    return I[:, 1:]  # drop self → shape (N, k)


# ─────────────────────────────────────────────
# Recall@K  (self-match safe)
# ─────────────────────────────────────────────

def compute_recall(index, queries, ground_truth_topk, nprobe, recall_k):
    """
    Recall@recall_k: fraction of queries where at least one true
    neighbor appears in the top-recall_k retrieved results.

    Bug fix vs v1: retrieves recall_k+1 results then drops self (index i),
    so self-match never inflates recall.
    """
    index.nprobe = nprobe
    _, I = index.search(queries, recall_k + 1)   # +1 to absorb self-match

    true_neighbors = ground_truth_topk[:, :recall_k]  # (N, recall_k)

    hits = 0
    for i in range(len(queries)):
        retrieved = [int(x) for x in I[i] if int(x) != i][:recall_k]
        if any(int(gt) in retrieved for gt in true_neighbors[i]):
            hits += 1
    return hits / len(queries)


# ─────────────────────────────────────────────
# Build IVF index
# ─────────────────────────────────────────────

def build_ivf(features, nlist, metric):
    dim = features.shape[1]
    if metric == "l2":
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)
    index.train(features)
    index.add(features)
    return index


# ─────────────────────────────────────────────
# Train EGA
# ─────────────────────────────────────────────

def train_ega(train_features, device, epochs=120, batch_size=1024,
              k_neighbors=30, margin=1.2):
    print("\n[EGA] Building kNN graph for triplet mining ...")
    dim = train_features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(train_features)
    _, indices = index.search(train_features, k_neighbors + 1)
    neighbors = indices[:, 1:]

    dataset = TripletDataset(train_features, neighbors)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=True)

    model     = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    print(f"[EGA] Training for {epochs} epochs ...")
    model.train()
    for epoch in tqdm(range(epochs), desc="EGA training"):
        epoch_loss = 0.0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            tqdm.write(f"  Epoch {epoch+1:3d}/{epochs}  "
                       f"loss={epoch_loss/len(loader):.4f}")
    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   required=True,
                        help="Path to CLIP .npy features, shape (N, 512)")
    parser.add_argument("--nlist",      type=int, default=100)
    parser.add_argument("--epochs",     type=int, default=120)
    parser.add_argument("--train_n",    type=int, default=8000)
    parser.add_argument("--save_ega",   default=None,
                        help="Path to save/load EGA features .npy")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, load EGA features from --save_ega")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load & normalize ──────────────────────────────────────────────
    print(f"\nLoading features from {args.features} ...")
    features = np.load(args.features).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    N, dim   = features.shape
    print(f"  Shape: {features.shape}")

    # ── Ground truth ──────────────────────────────────────────────────
    MAX_K = 10
    print(f"\nBuilding ground truth (brute-force cosine/IP, top-{MAX_K}) ...")
    ground_truth = build_ground_truth(features, k=MAX_K)
    print(f"  ground_truth shape : {ground_truth.shape}")
    print(f"  Sample — query 0, true top-5 neighbors: {ground_truth[0, :5]}")

    # ── Train or load EGA ─────────────────────────────────────────────
    import os
    if args.skip_train and args.save_ega and os.path.exists(args.save_ega):
        print(f"\n[EGA] Loading pre-computed features from {args.save_ega}")
        ega_features = np.load(args.save_ega).astype(np.float32)
    else:
        train_features = features[:args.train_n]
        model = train_ega(train_features, device, epochs=args.epochs)
        model.eval()
        with torch.no_grad():
            ega_features = model(
                torch.from_numpy(features).to(device)
            ).cpu().numpy()
        if args.save_ega:
            np.save(args.save_ega, ega_features)
            print(f"EGA features saved to {args.save_ega}")

    # ── Build IVF indexes ─────────────────────────────────────────────
    nlist = min(args.nlist, N // 10)
    print(f"\nBuilding IVF indexes (nlist={nlist}) ...")
    ivf_l2  = build_ivf(features,     nlist, "l2")
    ivf_ip  = build_ivf(features,     nlist, "ip")
    ivf_ega = build_ivf(ega_features, nlist, "l2")

    # ── Full evaluation table ─────────────────────────────────────────
    recall_ks   = [1, 3, 5, 10]
    nprobe_list = [1, 5, 10]

    for k in recall_ks:
        print(f"\n{'='*65}")
        print(f"  Recall@{k}  (ground truth: brute-force cosine)")
        print(f"{'='*65}")
        print(f"  {'nprobe':>8} | {'Raw-L2':>10} | {'Raw-IP':>10} | {'EGA-L2':>10}")
        print(f"  {'-'*9}+{'-'*12}+{'-'*12}+{'-'*12}")

        for nprobe in nprobe_list:
            r_l2  = compute_recall(ivf_l2,  features,     ground_truth, nprobe, k)
            r_ip  = compute_recall(ivf_ip,  features,     ground_truth, nprobe, k)
            r_ega = compute_recall(ivf_ega, ega_features, ground_truth, nprobe, k)
            print(f"  {nprobe:>8} | {r_l2:>10.4f} | {r_ip:>10.4f} | {r_ega:>10.4f}")

    # ── Key diagnostic ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  [Diagnostic] Recall@1")
    r_l2_1  = compute_recall(ivf_l2,  features,     ground_truth, 1, 1)
    r_ip_1  = compute_recall(ivf_ip,  features,     ground_truth, 1, 1)
    r_ega_1 = compute_recall(ivf_ega, ega_features, ground_truth, 1, 1)
    r_l2_5  = compute_recall(ivf_l2,  features,     ground_truth, 5, 1)

    print(f"  Raw-L2  nprobe=1 : {r_l2_1:.4f}")
    print(f"  Raw-IP  nprobe=1 : {r_ip_1:.4f}")
    print(f"  EGA-L2  nprobe=1 : {r_ega_1:.4f}  (vs Raw-IP: {r_ega_1 - r_ip_1:+.4f})")
    print(f"  Raw-L2  nprobe=5 : {r_l2_5:.4f}  "
          f"(EGA@1 {'≥' if r_ega_1 >= r_l2_5 else '<'} this)")

    print()
    if r_ega_1 > r_ip_1 + 0.02:
        print("  >> EGA brings real gains BEYOND metric choice. Worth investigating.")
    elif abs(r_ega_1 - r_ip_1) <= 0.02:
        print("  >> EGA ≈ Raw-IP. Original gain was metric artifact, not geometry fix.")
        print("     Reframe: EGA learns to approximate cosine retrieval in L2 space.")
    else:
        print("  >> Raw-IP > EGA. Original paper gains were metric artifact.")


if __name__ == "__main__":
    main()
