#!/usr/bin/env python3
"""
Margin OOD scan on Aircraft.
For each margin m in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], train an EGA adapter on CIFAR-100
(if not already trained) and evaluate LP@1 and AR@1 on the unseen classes of Aircraft.
Usage: python scripts/run_margin_ood_scan.py
"""

import os, argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils_ega import TripletDataset, split_by_class, eval_method
from models.ega_mlp import EGAMLP

# Fixed random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MARGINS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
EPOCHS = 150
BATCH_SIZE = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
BASE_DIR = os.path.expanduser('~/hpdic/EGA')
EMBED_DIR = os.path.join(BASE_DIR, 'embeddings')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

CIFAR_FEAT_PATH = os.path.join(EMBED_DIR, 'cifar100_vit_b32_features.npy')
CIFAR_LABEL_PATH = os.path.join(EMBED_DIR, 'cifar100_vit_b32_labels.npy')
AIRCRAFT_FEAT_PATH = os.path.join(EMBED_DIR, 'aircraft_test_vit_b32_features.npy')
AIRCRAFT_LABEL_PATH = os.path.join(EMBED_DIR, 'aircraft_test_vit_b32_labels.npy')

def train_ega_on_cifar(margin, model_path):
    """Train an EGA adapter on CIFAR-100 with a given margin and save it."""
    print(f"Training EGA on CIFAR-100 with margin={margin} ...")
    features = np.load(CIFAR_FEAT_PATH).astype(np.float32)
    labels = np.load(CIFAR_LABEL_PATH)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    dataset = TripletDataset(features, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    
    model.train()
    for epoch in range(EPOCHS):
        for a, p, n in loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return model

def get_ega_model(margin):
    """Load or train an EGA model for a given margin."""
    model_path = os.path.join(MODEL_DIR, f"ega_cifar100_m{margin}.pth")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = EGAMLP(input_dim=512, hidden_dim=2048).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    else:
        model = train_ega_on_cifar(margin, model_path)
        model.eval()
        return model

def main():
    # Load Aircraft features and split into seen/unseen classes
    print("Loading Aircraft features...")
    air_features = np.load(AIRCRAFT_FEAT_PATH).astype(np.float32)
    air_labels = np.load(AIRCRAFT_LABEL_PATH)
    air_features = air_features / np.linalg.norm(air_features, axis=1, keepdims=True)
    
    # Split by class (80% seen, 20% unseen) – we only evaluate on the unseen split
    (train_feats, train_labels), (test_feats, test_labels) = split_by_class(air_features, air_labels)
    print(f"Aircraft unseen test samples: {len(test_feats)}")
    
    # Evaluate frozen CLIP baseline
    print("Evaluating Frozen CLIP on Aircraft unseen classes...")
    lp_frozen, ar_frozen = eval_method(test_feats, test_labels)
    print(f"Frozen CLIP: LP@1={lp_frozen:.4f}, AR@1={ar_frozen:.4f}\n")
    
    # Evaluate EGA with different margins
    print(f"{'Margin':<10} {'LP@1':<10} {'AR@1':<10}")
    print("-" * 30)
    results = {}
    for m in MARGINS:
        model = get_ega_model(m)
        with torch.no_grad():
            feats = model(torch.from_numpy(test_feats).float().to(DEVICE)).cpu().numpy()
        lp, ar = eval_method(feats, test_labels)
        results[m] = (lp, ar)
        print(f"{m:<10} {lp:<10.4f} {ar:<10.4f}")
    
    # Optionally, save results to a file
    with open(os.path.join(MODEL_DIR, "margin_ood_results.txt"), "w") as f:
        f.write(f"Frozen CLIP: LP@1={lp_frozen:.4f}, AR@1={ar_frozen:.4f}\n")
        f.write(f"{'Margin':<10} {'LP@1':<10} {'AR@1':<10}\n")
        for m in MARGINS:
            lp, ar = results[m]
            f.write(f"{m:<10} {lp:<10.4f} {ar:<10.4f}\n")

if __name__ == '__main__':
    main()