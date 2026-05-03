# scripts/moreBackbones/02_train_ega_backbone.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
import argparse
from models.ega_mlp import EGAMLP   # 使用你原来的 EGAMLP

class SemanticCIFAR100Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = labels
        self.num_samples = len(features)
        
        self.label_to_indices = collections.defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
            
        self.classes = list(self.label_to_indices.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anchor = self.features[idx]
        anchor_label = self.labels[idx]
        
        # positive
        pos_indices = self.label_to_indices[anchor_label]
        pos_idx = np.random.choice(pos_indices)
        while pos_idx == idx and len(pos_indices) > 1:
            pos_idx = np.random.choice(pos_indices)
        positive = self.features[pos_idx]
        
        # negative
        neg_label = np.random.choice(self.classes)
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.classes)
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        negative = self.features[neg_idx]
        
        return anchor, positive, negative

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2-large",
                        choices=["clip", "dinov2-base", "dinov2-large", "siglip"])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser("~/hpdic/EGA")
    
    suffix = args.backbone.replace("-", "_")
    features_path = os.path.join(base_dir, f"embeddings/cifar100_{suffix}_features.npy")
    labels_path = os.path.join(base_dir, "embeddings/cifar100_labels.npy")
    
    model_save_path = os.path.join(base_dir, f"models/ega_{suffix}.pth")
    output_npy_path = os.path.join(base_dir, f"embeddings/cifar100_{suffix}_ega_features.npy")

    # 加载特征
    features = np.load(features_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)  # L2 normalize
    labels = np.load(labels_path)
    
    train_features = features[:8000]
    train_labels = labels[:8000]
    
    dataset = SemanticCIFAR100Dataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    
    # 根据 backbone 设置 input_dim
    input_dim_dict = {
        "clip": 512,
        "dinov2-base": 768,
        "dinov2-large": 1024,
        "siglip": 1152
    }
    input_dim = input_dim_dict[args.backbone]
    
    model = EGAMLP(input_dim=input_dim, hidden_dim=2048).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    model.train()
    print(f"🚀 Starting EGA Training on {args.backbone.upper()} (input_dim={input_dim})...")
    
    for epoch in range(120):
        total_loss = 0.0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(a), model(p), model(n))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/120, Loss: {total_loss / len(loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    
    # 对全部特征做变换并保存
    model.eval()
    all_feats = torch.from_numpy(features).to(device)
    with torch.no_grad():
        transformed = model(all_feats).cpu().numpy()
    
    np.save(output_npy_path, transformed)
    print(f"✅ Training finished! EGA features saved to: {output_npy_path}")

if __name__ == "__main__":
    main()