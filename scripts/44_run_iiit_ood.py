#!/usr/bin/env python3
"""
Oxford-IIIT Pet OOD evaluation (评估协议对齐主实验 Table 2):
  - 动态 75/25 切分索引库/查询集 (避免除零)
  - nlist=10 (与论文主表一致)
  - nprobe=1
  - 保持原有训练流程不变
Usage:
  python scripts/44_run_iiit_ood.py
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.ega_mlp import EGAMLP
from utils_ega import TripletDataset, split_by_class
import torchvision
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import faiss

# ── LoRA Adapter ──
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

# ── Training ──
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

# ── Feature extraction from Oxford-IIIT Pet ──
def extract_pet_features(device, output_dir="embeddings"):
    trainval_set = torchvision.datasets.OxfordIIITPet(
        root="./data", split="trainval", download=True,
        transform=transforms.Compose([transforms.Resize((224, 224))])
    )
    test_set = torchvision.datasets.OxfordIIITPet(
        root="./data", split="test", download=True,
        transform=transforms.Compose([transforms.Resize((224, 224))])
    )

    all_images, all_labels = [], []
    for dataset, split_name in [(trainval_set, "trainval"), (test_set, "test")]:
        for img, label in tqdm(dataset, desc=f"Loading {split_name}"):
            all_images.append(img)
            all_labels.append(label)

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
    np.save(os.path.join(output_dir, "pet_features.npy"), features)
    np.save(os.path.join(output_dir, "pet_labels.npy"), labels)
    print(f"Saved {len(features)} features, {len(np.unique(labels))} classes")
    return features, labels

# ── 评估 (动态 75/25 切分，nlist=10) ──
def evaluate(features, labels, k=1, nlist=10, nprobe=1):
    """75/25 动态切分，与你主实验 Table 2 对齐"""
    np.random.seed(42)
    perm = np.random.permutation(len(features))
    features, labels = features[perm], labels[perm]
    
    split = int(len(features) * 0.75)
    base, base_labels = features[:split], labels[:split]
    query, query_labels = features[split:], labels[split:]
    
    dim = base.shape[1]
    
    # Ground truth (exact)
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, k)
    
    # IVF index
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = nprobe
    _, ret = ivf.search(query, k)
    
    # 计算指标
    recall = np.mean([len(np.intersect1d(ret[i, :k], gt[i, :k])) for i in range(len(gt))]) / k
    lp = sum(np.sum(base_labels[ret[i, :k]] == query_labels[i]) for i in range(len(query_labels)))
    lp /= (len(query_labels) * k)
    
    return lp, recall

# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    feat_path = "embeddings/pet_features.npy"
    label_path = "embeddings/pet_labels.npy"
    if not os.path.exists(feat_path):
        print("Extracting features (this will download Oxford-IIIT Pet)...")
        features, labels = extract_pet_features(device)
    else:
        features = np.load(feat_path)
        labels = np.load(label_path)

    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)
    dim = features.shape[1]
    print(f"Total classes: {len(np.unique(labels))}")
    print(f"Train: {len(train_feats)} samples ({len(np.unique(train_labels))} classes)")
    print(f"Test: {len(test_feats)} samples ({len(np.unique(test_labels))} classes) [unseen]")

    # Frozen CLIP baseline
    lp, ar = evaluate(test_feats, test_labels)
    print(f"Frozen CLIP: LP@1={lp:.4f}, AR@1={ar:.4f}")

    # LoRA+Triplet r=128
    print("Training LoRA+Triplet (r=128)...")
    lora_model = train_lora_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    lora_model.eval()
    with torch.no_grad():
        lora_feats = lora_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_l, ar_l = evaluate(lora_feats, test_labels)
    print(f"LoRA+Triplet: LP@1={lp_l:.4f}, AR@1={ar_l:.4f}")

    # EGA
    print("Training EGA...")
    ega_model = train_ega_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    ega_model.eval()
    with torch.no_grad():
        ega_feats = ega_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_e, ar_e = evaluate(ega_feats, test_labels)
    print(f"EGA: LP@1={lp_e:.4f}, AR@1={ar_e:.4f}")

    print("\n=== Oxford-IIIT Pet OOD Results ===")
    print(f"{'Method':20s} {'LP@1':>8s} {'AR@1':>8s}")
    print(f"{'Frozen CLIP':20s} {lp:8.4f} {ar:8.4f}")
    print(f"{'LoRA+Triplet':20s} {lp_l:8.4f} {ar_l:8.4f}")
    print(f"{'EGA':20s} {lp_e:8.4f} {ar_e:8.4f}")

if __name__ == "__main__":
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/44_run_iiit_ood.py 
# Total classes: 37
# Train: 5771 samples (29 classes)
# Test: 1578 samples (8 classes) [unseen]
# Frozen CLIP: LP@1=0.9595, AR@1=0.8785
# Training LoRA+Triplet (r=128)...
# LoRA+Triplet: LP@1=0.9595, AR@1=0.9342
# Training EGA...
# EGA: LP@1=0.9595, AR@1=0.9544

# === Oxford-IIIT Pet OOD Results ===
# Method                   LP@1     AR@1
# Frozen CLIP            0.9595   0.8785
# LoRA+Triplet           0.9595   0.9342
# EGA                    0.9595   0.9544
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 

