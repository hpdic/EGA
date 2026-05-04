# scripts/48_train_ega_150epoch_imagenet1000.py
# 150 epoch 版本（最终版）

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
from models.ega_mlp import EGAMLP
from torch.optim.lr_scheduler import CosineAnnealingLR

class TripletDataset(Dataset):
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
        
        pos_indices = self.label_to_indices[anchor_label]
        pos_idx = np.random.choice(pos_indices)
        while pos_idx == idx and len(pos_indices) > 1:
            pos_idx = np.random.choice(pos_indices)
        positive = self.features[pos_idx]
        
        neg_label = np.random.choice(self.classes)
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.classes)
            
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        negative = self.features[neg_idx]
        
        return anchor, positive, negative

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    
    features_path = os.path.join(base_dir, 'embeddings/imagenet1000_features.npy')
    labels_path = os.path.join(base_dir, 'embeddings/imagenet1000_labels.npy')
    model_save_path = os.path.join(base_dir, 'models/ega_imagenet1000_150epoch.pth')
    output_npy_path = os.path.join(base_dir, 'embeddings/imagenet1000_ega_150epoch.npy')

    features = np.load(features_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(labels_path)
    
    print(f"Total samples: {len(features)}, Total classes: {len(np.unique(labels))}")
    
    train_features = features
    train_labels = labels
    
    dataset = TripletDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=150)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    model.train()
    print('Starting EGA Training on ImageNet-1000 (150 epochs)...')
    for epoch in range(100):
        total_loss = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/150, Loss: {total_loss / len(loader):.4f}')

    torch.save(model.state_dict(), model_save_path)
    print(f'\nModel saved to {model_save_path}')
    
    # 转换所有特征
    model.eval()
    all_feats = torch.from_numpy(features).to(device)
    with torch.no_grad():
        transformed = model(all_feats).cpu().numpy()
    np.save(output_npy_path, transformed)
    print(f'Transformed features saved to {output_npy_path}')

if __name__ == '__main__':
    main()

