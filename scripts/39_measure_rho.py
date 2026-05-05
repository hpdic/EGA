# scripts/measure_rho.py
"""
Measure active triplet ratio on a given dataset using a trained EGA model.
Usage:
  python scripts/measure_rho.py --dataset cifar100
  python scripts/measure_rho.py --dataset food101
"""
import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from models.ega_mlp import EGAMLP
from utils_ega import TripletDataset

def compute_rho(model, features, labels, device, margin=0.2, num_samples=50000, batch_size=256):
    """Compute active triplet ratio rho = fraction of triplets where d(a,p)-d(a,n)+m > 0."""
    model.eval()
    dataset = TripletDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    active_count = 0
    total_count = 0
    
    with torch.no_grad():
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            a_emb = model(a)
            p_emb = model(p)
            n_emb = model(n)
            d_ap = torch.norm(a_emb - p_emb, p=2, dim=1)
            d_an = torch.norm(a_emb - n_emb, p=2, dim=1)
            active = (d_ap - d_an + margin) > 0
            active_count += active.sum().item()
            total_count += len(a)
            if total_count >= num_samples:
                break
    
    rho = active_count / total_count if total_count > 0 else 0.0
    return rho

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar100", "food101", "aircraft", "imagenet"])
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained EGA model (.pth)")
    parser.add_argument("--feat_path", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=50000)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default paths (adjust to your setup)
    base_dir = os.path.expanduser("~/hpdic/EGA")
    embed_dir = os.path.join(base_dir, "embeddings")
    model_dir = os.path.join(base_dir, "models")
    
    config = {
        "cifar100": {
            "feat": os.path.join(embed_dir, "cifar100_vit_b32_features.npy"),
            "label": os.path.join(embed_dir, "cifar100_vit_b32_labels.npy"),
            "model": os.path.join(model_dir, "ega_cifar100.pth")
        },
        "food101": {
            "feat": os.path.join(embed_dir, "food101_features.npy"),
            "label": os.path.join(embed_dir, "food101_labels.npy"),
            "model": os.path.join(model_dir, "ega_food101.pth")
        },
        "aircraft": {
            "feat": os.path.join(embed_dir, "aircraft_test_vit_b32_features.npy"),
            "label": os.path.join(embed_dir, "aircraft_test_vit_b32_labels.npy"),
            "model": os.path.join(model_dir, "ega_aircraft.pth")
        },
        "imagenet": {
            "feat": os.path.join(embed_dir, "imagenet1000_features.npy"),
            "label": os.path.join(embed_dir, "imagenet1000_labels.npy"),
            "model": os.path.join(model_dir, "ega_imagenet1000_150epoch.pth")
        }
    }
    
    cfg = config[args.dataset]
    feat_path = args.feat_path or cfg["feat"]
    label_path = args.label_path or cfg["label"]
    model_path = args.model_path or cfg["model"]
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Specify --model_path or ensure the default path exists.")
        return
    
    features = np.load(feat_path).astype(np.float32)
    labels = np.load(label_path)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Load model
    model = EGAMLP(input_dim=features.shape[1], hidden_dim=2048).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    rho = compute_rho(model, features, labels, device, num_samples=args.num_samples)
    print(f"Dataset: {args.dataset}")
    print(f"Active triplet ratio rho = {rho:.4f} ({rho*100:.2f}%)")
    print(f"Inactive (zero-gradient) fraction = {1-rho:.4f} ({(1-rho)*100:.2f}%)")

if __name__ == "__main__":
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/39_measure_rho.py --dataset cifar100
# Dataset: cifar100
# Active triplet ratio rho = 0.0205 (2.05%)
# Inactive (zero-gradient) fraction = 0.9795 (97.95%)
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 

# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/39_measure_rho.py --dataset food101
# Dataset: food101
# Active triplet ratio rho = 0.0084 (0.84%)
# Inactive (zero-gradient) fraction = 0.9916 (99.16%)
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 