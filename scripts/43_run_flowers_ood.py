#!/usr/bin/env python3
"""
Oxford Flowers-102 OOD evaluation: EGA vs LoRA+Triplet vs Frozen CLIP.
Usage:
  python scripts/run_flowers_ood.py
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.ega_mlp import EGAMLP
from utils_ega import TripletDataset, split_by_class, eval_method
import torchvision
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

# ────────── LoRA Adapter ──────────
class LoRAAdapter(nn.Module):
    def __init__(self, dim=512, rank=128, alpha=16):
        super().__init__()
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T
        return F.normalize(x + self.scale * delta, p=2, dim=1)

# ────────── Training helpers ──────────
def train_lora_triplet(train_feats, train_labels, device, dim, epochs=150, rank=128, margin=0.2):
    loader = DataLoader(TripletDataset(train_feats, train_labels),
                        batch_size=256, shuffle=True, num_workers=4)
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
    return model

def train_ega_triplet(train_feats, train_labels, device, dim, epochs=150, margin=0.2):
    loader = DataLoader(TripletDataset(train_feats, train_labels),
                        batch_size=256, shuffle=True, num_workers=4)
    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
    return model

# ────────── Feature extraction ──────────
def extract_flowers_features(device, output_dir="embeddings"):
    """Download Flowers-102 via torchvision and extract CLIP features."""
    # torchvision Flowers102 download
    train_set = torchvision.datasets.Flowers102(
        root="./data", split="train", download=True,
        transform=transforms.Compose([transforms.Resize((224, 224))])
    )
    test_set = torchvision.datasets.Flowers102(
        root="./data", split="test", download=True,
        transform=transforms.Compose([transforms.Resize((224, 224))])
    )
    # val set is also available, but we'll combine train+val for training (or just use train)
    val_set = torchvision.datasets.Flowers102(
        root="./data", split="val", download=True,
        transform=transforms.Compose([transforms.Resize((224, 224))])
    )
    
    # Combine train+val as our training pool (if needed), but for OOD we need class split
    # We'll combine all splits into one big feature set, then do 80/20 class split
    all_images = []
    all_labels = []
    for dataset, split_name in [(train_set, "train"), (val_set, "val"), (test_set, "test")]:
        for img, label in tqdm(dataset, desc=f"Extracting {split_name}"):
            all_images.append(img)
            all_labels.append(label)
    
    # Extract CLIP features
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    features_list = []
    for img in tqdm(all_images, desc="CLIP encoding"):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            feat = outputs.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        features_list.append(feat.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0).astype(np.float32)
    labels = np.array(all_labels)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "flowers_features.npy"), features)
    np.save(os.path.join(output_dir, "flowers_labels.npy"), labels)
    print(f"Saved {len(features)} features, {len(np.unique(labels))} classes")
    return features, labels

# ────────── Main ──────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    
    # Extract or load features
    feat_path = "embeddings/flowers_features.npy"
    label_path = "embeddings/flowers_labels.npy"
    if not os.path.exists(feat_path):
        print("Extracting features...")
        features, labels = extract_flowers_features(device)
    else:
        features = np.load(feat_path)
        labels = np.load(label_path)
    
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # OOD class split (80/20)
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)
    dim = features.shape[1]
    print(f"Train: {len(train_feats)} samples, Test: {len(test_feats)} samples (unseen classes)")
    
    # Frozen CLIP baseline
    lp_frozen, ar_frozen = eval_method(test_feats, test_labels)
    print(f"Frozen CLIP: LP@1={lp_frozen:.4f}, AR@1={ar_frozen:.4f}")
    
    # LoRA+Triplet r=128
    print("Training LoRA+Triplet (r=128)...")
    lora_model = train_lora_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    lora_model.eval()
    with torch.no_grad():
        lora_feats = lora_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_lora, ar_lora = eval_method(lora_feats, test_labels)
    print(f"LoRA+Triplet: LP@1={lp_lora:.4f}, AR@1={ar_lora:.4f}")
    
    # EGA
    print("Training EGA...")
    ega_model = train_ega_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    ega_model.eval()
    with torch.no_grad():
        ega_feats = ega_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_ega, ar_ega = eval_method(ega_feats, test_labels)
    print(f"EGA: LP@1={lp_ega:.4f}, AR@1={ar_ega:.4f}")
    
    print("\n=== Oxford Flowers-102 OOD Results ===")
    print(f"{'Method':20s} {'LP@1':>8s} {'AR@1':>8s}")
    print(f"{'Frozen CLIP':20s} {lp_frozen:8.4f} {ar_frozen:8.4f}")
    print(f"{'LoRA+Triplet':20s} {lp_lora:8.4f} {ar_lora:8.4f}")
    print(f"{'EGA':20s} {lp_ega:8.4f} {ar_ega:8.4f}")

if __name__ == "__main__":
    main()