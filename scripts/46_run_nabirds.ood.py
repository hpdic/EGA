#!/usr/bin/env python3
"""
NABirds OOD evaluation: EGA vs LoRA+Triplet (r=128).
评估协议对齐主实验 (nlist=10, nprobe=1, 75/25 动态切分)。
第一次运行时，脚本会自动下载并解压 NABirds 数据集。
Usage:
  python scripts/run_nabirds_ood.py
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.ega_mlp import EGAMLP
from utils_ega import TripletDataset, split_by_class
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import tarfile, requests
from PIL import Image
import faiss
import sys

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

# ────────── Training ──────────
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

# ── 自动下载并解压 NABirds ──
def download_nabirds(data_dir="./data"):
    """下载并解压 NABirds，返回图像目录和标注文件路径"""
    nabirds_dir = os.path.join(data_dir, "nabirds")
    images_dir = os.path.join(nabirds_dir, "images")
    # NABirds 标注文件在解压后的根目录
    # 检查是否已存在
    if os.path.exists(images_dir):
        # 查找标注文件
        for f in os.listdir(nabirds_dir):
            if f.endswith('.txt') and 'image_class_labels' in f:
                labels_file = os.path.join(nabirds_dir, f)
                break
        else:
            labels_file = None
        
        if labels_file and os.path.exists(labels_file):
            print(f"✅ NABirds already exists at {nabirds_dir}")
            return images_dir, labels_file

    # 尝试使用 fgvcdata 库（如果已安装）
    try:
        import fgvcdata
        dataset = fgvcdata.datasets.NABirds(root=data_dir, download=True)
        print(f"✅ NABirds downloaded via fgvcdata at {nabirds_dir}")
        # 从 fgvcdata 中获取图像路径和标签
        # 这里需要根据实际情况调整
        return None, None  # 后续手动处理
    except ImportError:
        print("⚙️  fgvcdata not installed. Attempting manual download...")
    
    # 手动下载
    url = "https://dl.allaboutbirds.org/nabirds"
    # NABirds 通常需要分别下载 images 和 annotations
    # 这里提供一个简化的下载逻辑
    print("📥 Attempting to download NABirds...")
    print("This dataset requires manual download from https://dl.allaboutbirds.org/nabirds")
    print("Please download the images and annotations tar files and place them in ./data/nabirds/")
    print("Then extract them and ensure the following structure:")
    print("  ./data/nabirds/images/")
    print("  ./data/nabirds/image_class_labels.txt")
    print("  (or similar annotation files)")
    sys.exit(1)  # 暂时中止，等待手动下载

# ── 从本地文件提取特征 ──
def extract_nabirds_features(device, output_dir="embeddings"):
    # 首先尝试使用 fgvcdata 加载
    try:
        import fgvcdata
        from fgvcdata.datasets import NABirds
        
        # 尝试加载数据集
        dataset = NABirds(root="./data", download=True, transform=transforms.Compose([transforms.Resize((224, 224))]))
        
        # 从数据集中提取所有图像和标签
        all_features, all_labels_list = [], []
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
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
        
    except ImportError:
        print("fgvcdata not available. Installing it may help: pip install fgvcdata")
        print("Alternatively, manually download NABirds and place images in ./data/nabirds/images/")
        print("and annotations in ./data/nabirds/")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "nabirds_features.npy"), features)
    np.save(os.path.join(output_dir, "nabirds_labels.npy"), labels_arr)
    print(f"Saved {len(features)} features, {len(np.unique(labels_arr))} classes")
    return features, labels_arr

# ── 评估 (75/25 动态切分，nlist=10，与主实验对齐) ──
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

    feat_path = "embeddings/nabirds_features.npy"
    label_path = "embeddings/nabirds_labels.npy"
    if not os.path.exists(feat_path):
        print("Extracting features (this will download NABirds)...")
        features, labels = extract_nabirds_features(device)
    else:
        features = np.load(feat_path)
        labels = np.load(label_path)

    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)
    dim = features.shape[1]
    print(f"Total classes: {len(np.unique(labels))}")
    print(f"Train: {len(train_feats)} samples ({len(np.unique(train_labels))} classes)")
    print(f"Test: {len(test_feats)} samples ({len(np.unique(test_labels))} classes) [unseen]")

    # Frozen CLIP
    lp, ar = evaluate(test_feats, test_labels)
    print(f"Frozen CLIP: LP@1={lp:.4f}, AR@1={ar:.4f}")

    # LoRA+Triplet
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

    print("\n=== NABirds OOD Results ===")
    print(f"{'Method':20s} {'LP@1':>8s} {'AR@1':>8s}")
    print(f"{'Frozen CLIP':20s} {lp:8.4f} {ar:8.4f}")
    print(f"{'LoRA+Triplet':20s} {lp_l:8.4f} {ar_l:8.4f}")
    print(f"{'EGA':20s} {lp_e:8.4f} {ar_e:8.4f}")

if __name__ == "__main__":
    main()