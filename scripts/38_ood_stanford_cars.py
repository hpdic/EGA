# scripts/run_stanford_cars_ood.py
"""
Stanford Cars OOD evaluation: EGA vs LoRA+Triplet (r=128).
Usage:
  python scripts/run_stanford_cars_ood.py --cars_dir /path/to/stanford_cars
"""
import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from utils_ega import TripletDataset, split_by_class, eval_method
from models.ega_mlp import EGAMLP
from scripts.15_train_eval_lora import LoRAAdapter, train_lora_triplet  # 复用已有实现
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

def extract_features(cars_dir, output_dir, device):
    """Extract CLIP ViT-B/32 features for Stanford Cars dataset."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    # Stanford Cars 官方结构: cars_train/ 和 cars_test/，每个子目录下是类别文件夹
    features_list, labels_list = [], []
    class_names = sorted(os.listdir(os.path.join(cars_dir, "cars_train")))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for split in ["cars_train", "cars_test"]:
        split_path = os.path.join(cars_dir, split)
        for class_name in tqdm(class_names, desc=f"Extracting {split}"):
            class_dir = os.path.join(split_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                try:
                    img = Image.open(os.path.join(class_dir, img_file)).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        feat = model.get_image_features(**inputs)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                    features_list.append(feat.cpu().numpy())
                    labels_list.append(class_to_idx[class_name])
                except Exception:
                    pass
    
    features = np.concatenate(features_list, axis=0).astype(np.float32)
    labels = np.array(labels_list)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "cars_features.npy"), features)
    np.save(os.path.join(output_dir, "cars_labels.npy"), labels)
    print(f"Saved {len(features)} features, {len(np.unique(labels))} classes")
    return features, labels

def train_ega_triplet(train_feats, train_labels, device, dim, epochs=150, margin=0.2):
    """Train EGA with triplet loss (reuses EGAMLP)."""
    loader = DataLoader(TripletDataset(train_feats, train_labels),
                        batch_size=256, shuffle=True, num_workers=4)
    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cars_dir", type=str, required=True, help="Path to Stanford Cars root (contains cars_train/ and cars_test/)")
    parser.add_argument("--output_dir", type=str, default="embeddings")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    
    # Extract features if not already present
    feat_path = os.path.join(args.output_dir, "cars_features.npy")
    label_path = os.path.join(args.output_dir, "cars_labels.npy")
    if not os.path.exists(feat_path):
        extract_features(args.cars_dir, args.output_dir, device)
    
    features = np.load(feat_path)
    labels = np.load(label_path)
    # Normalize
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # OOD split: 80% classes for training, 20% for test
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)
    dim = features.shape[1]
    
    results = {}
    # 1. LoRA+Triplet r=128
    print("Training LoRA+Triplet (r=128)...")
    lora_model = train_lora_triplet(train_feats, train_labels, device, dim,
                                    epochs=args.epochs, rank=128)
    lora_model.eval()
    with torch.no_grad():
        lora_feats = lora_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_lora, ar_lora = eval_method(lora_feats, test_labels)
    results["LoRA+Triplet"] = (lp_lora, ar_lora)
    print(f"LoRA+Triplet: LP@1={lp_lora:.4f}, AR@1={ar_lora:.4f}")
    
    # 2. EGA
    print("Training EGA...")
    ega_model = train_ega_triplet(train_feats, train_labels, device, dim, epochs=args.epochs)
    ega_model.eval()
    with torch.no_grad():
        ega_feats = ega_model(torch.from_numpy(test_feats).float().to(device)).cpu().numpy()
    lp_ega, ar_ega = eval_method(ega_feats, test_labels)
    results["EGA"] = (lp_ega, ar_ega)
    print(f"EGA: LP@1={lp_ega:.4f}, AR@1={ar_ega:.4f}")
    
    # Frozen CLIP baseline
    lp_frozen, ar_frozen = eval_method(test_feats, test_labels)
    results["Frozen CLIP"] = (lp_frozen, ar_frozen)
    print(f"Frozen CLIP: LP@1={lp_frozen:.4f}, AR@1={ar_frozen:.4f}")
    
    print("\nSummary:")
    for name, (lp, ar) in results.items():
        print(f"{name:20s}: LP@1={lp:.4f}, AR@1={ar:.4f}")

if __name__ == "__main__":
    main()