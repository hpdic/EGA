import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.ega_mlp import EGAMLP
from utils_ega import TripletDataset  # 假设 utils_ega.py 已包含 TripletDataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    feat_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_features.npy')
    label_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_labels.npy')
    model_save_path = os.path.join(base_dir, 'models/ega_cifar100.pth')

    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)

    dataset = TripletDataset(features, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=150)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    model.train()
    print("Training EGA on CIFAR-100 (150 epochs)...")
    for epoch in range(150):
        total_loss = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}/150, Loss: {total_loss / len(loader):.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()