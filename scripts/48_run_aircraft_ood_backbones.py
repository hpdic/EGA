#!/usr/bin/env python3
"""
Aircraft OOD on DINOv2-large and SigLIP backbones.
Only EGA vs LoRA+Triplet (r=128). 评估协议对齐主实验。
Usage:
  python scripts/run_aircraft_ood_backbones.py --backbone dinov2
  python scripts/run_aircraft_ood_backbones.py --backbone siglip
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from models.ega_mlp import EGAMLP
from utils_ega import TripletDataset, split_by_class
import torchvision.transforms as transforms
from PIL import Image
import faiss

# ── backbone 配置 ──
BACKBONE_CONFIG = {
    "dinov2": {
        "model_name": "facebook/dinov2-large",
        "dim": 1024,
        "feature_save": "aircraft_dinov2_features.npy",
        "label_save": "aircraft_dinov2_labels.npy",
    },
    "siglip": {
        "model_name": "google/siglip-large-patch16-384",
        "dim": 1024,
        "feature_save": "aircraft_siglip_features.npy",
        "label_save": "aircraft_siglip_labels.npy",
    },
}

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

def extract_features(backbone_key, cfg, device, output_dir="embeddings"):
    """用 torchvision 加载 Aircraft 数据并提取特征"""
    from transformers import AutoImageProcessor, AutoModel
    import torchvision.datasets as datasets

    # 根目录设为包含 fgvc-aircraft-2013b 的上一级目录
    data_root = os.path.expanduser("~/hpdic/EGA/data")
    
    print(f"Loading backbone: {cfg['model_name']}...")
    processor = AutoImageProcessor.from_pretrained(cfg["model_name"])
    model = AutoModel.from_pretrained(cfg["model_name"]).to(device)
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224))])

    # 直接使用 torchvision 加载 train 和 test 集合（不需要下载，因为已存在）
    train_set = datasets.FGVCAircraft(root=data_root, split="train", annotation_level="variant", download=False, transform=transform)
    test_set = datasets.FGVCAircraft(root=data_root, split="test", annotation_level="variant", download=False, transform=transform)

    all_images, all_labels = [], []
    for ds in [train_set, test_set]:
        for img, label in ds:
            all_images.append(img)
            all_labels.append(label)
    
    print(f"Found {len(all_images)} images")
    
    all_features = []
    for img in tqdm(all_images, desc=f"Extracting {backbone_key}"):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            if "dinov2" in backbone_key:
                outputs = model(**inputs)
                feat = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:  # siglip
                outputs = model.get_image_features(**inputs)
                feat = outputs
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_features.append(feat.cpu().numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels_arr = np.array(all_labels)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, cfg["feature_save"]), features)
    np.save(os.path.join(output_dir, cfg["label_save"]), labels_arr)
    print(f"Saved {len(features)} features, dim={features.shape[1]}")
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
    parser.add_argument("--backbone", type=str, required=True, choices=["dinov2", "siglip"])
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    
    cfg = BACKBONE_CONFIG[args.backbone]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    
    # 提取或加载特征
    feat_path = os.path.join("embeddings", cfg["feature_save"])
    label_path = os.path.join("embeddings", cfg["label_save"])
    if not os.path.exists(feat_path):
        print(f"Extracting features for {args.backbone}...")
        features, labels = extract_features(args.backbone, cfg, device)
        if features is None:
            return
    else:
        features = np.load(feat_path)
        labels = np.load(label_path)
    
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    dim = cfg["dim"]
    print(f"Feature dim: {dim}")
    
    # OOD split
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)
    print(f"Train: {len(train_feats)} samples, Test: {len(test_feats)} samples [unseen]")
    
    # Frozen baseline
    lp, ar = evaluate(test_feats, test_labels)
    print(f"Frozen {args.backbone}: LP@1={lp:.4f}, AR@1={ar:.4f}")
    
    # LoRA+Triplet
    print(f"Training LoRA+Triplet on {args.backbone}...")
    lora_model = train_lora_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    lora_model.eval()
    with torch.no_grad():
        lora_feats = lora_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_l, ar_l = evaluate(lora_feats, test_labels)
    print(f"LoRA+Triplet: LP@1={lp_l:.4f}, AR@1={ar_l:.4f}")
    
    # EGA
    print(f"Training EGA on {args.backbone}...")
    ega_model = train_ega_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    ega_model.eval()
    with torch.no_grad():
        ega_feats = ega_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_e, ar_e = evaluate(ega_feats, test_labels)
    print(f"EGA: LP@1={lp_e:.4f}, AR@1={ar_e:.4f}")
    
    print(f"\n=== Aircraft OOD on {args.backbone} ===")
    print(f"{'Method':20s} {'LP@1':>8s} {'AR@1':>8s}")
    print(f"{'Frozen ' + args.backbone:20s} {lp:8.4f} {ar:8.4f}")
    print(f"{'LoRA+Triplet':20s} {lp_l:8.4f} {ar_l:8.4f}")
    print(f"{'EGA':20s} {lp_e:8.4f} {ar_e:8.4f}")

if __name__ == "__main__":
    main()