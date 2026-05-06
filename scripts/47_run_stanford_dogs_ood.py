#!/usr/bin/env python3
"""
Stanford Dogs OOD evaluation: EGA vs LoRA+Triplet (r=128).
评估协议对齐主实验 (nlist=10, nprobe=1, 75/25 动态切分)。

cd ~/hpdic/EGA/data

# 下载（约 800MB，速度较快）
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

# 解压
tar -xf images.tar

# 解压后会生成 Images/ 目录，里面有 120 个子文件夹
# 每个子文件夹就是一种狗，图片已经按类别分好
# 
#       
Usage:
  python scripts/run_stanford_dogs_ood.py
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.ega_mlp import EGAMLP
from utils_ega import TripletDataset, split_by_class
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
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

def extract_dogs_features(device, output_dir="embeddings"):
    """从本地 Images/ 目录加载 Stanford Dogs 并提取 CLIP 特征（逐张处理）"""
    images_dir = os.path.expanduser("~/hpdic/EGA/data/Images")
    if not os.path.exists(images_dir):
        print("Stanford Dogs images not found!")
        print("Download: wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar")
        print("Extract:  tar -xf images.tar -C ~/hpdic/EGA/data/")
        return None, None

    transform = transforms.Compose([transforms.Resize((224, 224))])
    dataset = ImageFolder(root=images_dir, transform=transform)
    print(f"Loaded {len(dataset)} images, {len(dataset.classes)} classes")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    all_features = []
    all_labels_list = []
    for img, label in tqdm(dataset, desc="CLIP encoding"):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            feat = outputs.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_features.append(feat.cpu().numpy())
        all_labels_list.append(label)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels_arr = np.array(all_labels_list)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "dogs_features.npy"), features)
    np.save(os.path.join(output_dir, "dogs_labels.npy"), labels_arr)
    print(f"Saved {len(features)} features, {len(np.unique(labels_arr))} classes")
    return features, labels_arr

# ── 评估 ──
def evaluate(features, labels, k=1, nlist=10, nprobe=1):
    np.random.seed(42)
    perm = np.random.permutation(len(features))
    features, labels = features[perm], labels[perm]
    split = int(len(features) * 0.75)
    base, base_labels = features[:split], labels[:split]
    query, query_labels = features[split:], labels[split:]
    dim = base.shape[1]
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, k)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = nprobe
    _, ret = ivf.search(query, k)
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

    feat_path = "embeddings/dogs_features.npy"
    label_path = "embeddings/dogs_labels.npy"
    if not os.path.exists(feat_path):
        print("Extracting features...")
        features, labels = extract_dogs_features(device)
        if features is None:
            return
    else:
        features = np.load(feat_path)
        labels = np.load(label_path)

    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)
    dim = features.shape[1]
    print(f"Total classes: {len(np.unique(labels))}")
    print(f"Train: {len(train_feats)} samples ({len(np.unique(train_labels))} classes)")
    print(f"Test: {len(test_feats)} samples ({len(np.unique(test_labels))} classes) [unseen]")

    lp, ar = evaluate(test_feats, test_labels)
    print(f"Frozen CLIP: LP@1={lp:.4f}, AR@1={ar:.4f}")

    print("Training LoRA+Triplet (r=128)...")
    lora_model = train_lora_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    lora_model.eval()
    with torch.no_grad():
        lora_feats = lora_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_l, ar_l = evaluate(lora_feats, test_labels)
    print(f"LoRA+Triplet: LP@1={lp_l:.4f}, AR@1={ar_l:.4f}")

    print("Training EGA...")
    ega_model = train_ega_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    ega_model.eval()
    with torch.no_grad():
        ega_feats = ega_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_e, ar_e = evaluate(ega_feats, test_labels)
    print(f"EGA: LP@1={lp_e:.4f}, AR@1={ar_e:.4f}")

    print("\n=== Stanford Dogs OOD Results ===")
    print(f"{'Method':20s} {'LP@1':>8s} {'AR@1':>8s}")
    print(f"{'Frozen CLIP':20s} {lp:8.4f} {ar:8.4f}")
    print(f"{'LoRA+Triplet':20s} {lp_l:8.4f} {ar_l:8.4f}")
    print(f"{'EGA':20s} {lp_e:8.4f} {ar_e:8.4f}")

if __name__ == "__main__":
    main()