#!/usr/bin/env python3
"""
CUB-200-2011 OOD evaluation: EGA vs LoRA+Triplet (r=128).
评估协议对齐主实验 (nlist=10, nprobe=1, 75/25 动态切分)。
第一次运行时，脚本会自动下载并解压 CUB-200-2011 数据集。
Usage:
  python scripts/45_run_cub_ood.py
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

# ── 自动下载并解压 CUB-200-2011 ──
def download_cub(data_dir="./data"):
    """下载并解压 CUB-200-2011，返回解压后的根目录"""
    cub_dir = os.path.join(data_dir, "CUB_200_2011")
    images_dir = os.path.join(cub_dir, "images")
    labels_file = os.path.join(cub_dir, "image_class_labels.txt")
    images_txt = os.path.join(cub_dir, "images.txt")
    
    if os.path.exists(images_dir) and os.path.exists(labels_file) and os.path.exists(images_txt):
        print(f"✅ CUB-200-2011 already exists at {cub_dir}")
        return cub_dir

    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tgz_path = os.path.join(data_dir, "CUB_200_2011.tgz")
    
    print(f"📥 Downloading CUB-200-2011 (1.2 GB)...")
    response = requests.get(url, stream=True)
    with open(tgz_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
            f.write(chunk)
    
    print(f"📦 Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(data_dir)
    
    os.remove(tgz_path)
    print(f"✅ CUB-200-2011 ready at {cub_dir}")
    return cub_dir

# ── 从本地文件提取特征 ──
def extract_cub_features(device, output_dir="embeddings"):
    cub_dir = download_cub()
    images_dir = os.path.join(cub_dir, "images")
    images_txt = os.path.join(cub_dir, "images.txt")
    labels_txt = os.path.join(cub_dir, "image_class_labels.txt")
    
    # 读取 images.txt 建立 image_id -> file_path 映射
    id_to_path = {}
    with open(images_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id = int(parts[0])
                path = parts[1]  # 例如 "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
                id_to_path[img_id] = path
    
    # 读取 image_class_labels.txt 建立 image_id -> class_id 映射 (0-index)
    id_to_label = {}
    with open(labels_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id = int(parts[0])
                label = int(parts[1]) - 1  # 转为0索引
                id_to_label[img_id] = label
    
    # 加载 CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224))])
    
    all_features, all_labels_list = [], []
    for img_id, rel_path in tqdm(id_to_path.items(), desc="CLIP encoding"):
        if img_id not in id_to_label:
            continue
        full_img_path = os.path.join(images_dir, rel_path)
        if not os.path.exists(full_img_path):
            continue
        img = Image.open(full_img_path).convert("RGB")
        img = transform(img)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            feat = outputs.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_features.append(feat.cpu().numpy())
        all_labels_list.append(id_to_label[img_id])
    
    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels_arr = np.array(all_labels_list)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "cub_features.npy"), features)
    np.save(os.path.join(output_dir, "cub_labels.npy"), labels_arr)
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

    feat_path = "embeddings/cub_features.npy"
    label_path = "embeddings/cub_labels.npy"
    if not os.path.exists(feat_path):
        print("Extracting features (this will download CUB-200-2011)...")
        features, labels = extract_cub_features(device)
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

    print("\n=== CUB-200-2011 OOD Results ===")
    print(f"{'Method':20s} {'LP@1':>8s} {'AR@1':>8s}")
    print(f"{'Frozen CLIP':20s} {lp:8.4f} {ar:8.4f}")
    print(f"{'LoRA+Triplet':20s} {lp_l:8.4f} {ar_l:8.4f}")
    print(f"{'EGA':20s} {lp_e:8.4f} {ar_e:8.4f}")

if __name__ == "__main__":
    main()