# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import collections
from models.ega_mlp import EGAMLP

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
    
    features_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_features.npy')
    labels_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_labels.npy')
    model_save_path = os.path.join(base_dir, 'models/ega_bridge_cifar100.pth')
    output_npy_path = os.path.join(base_dir, 'embeddings/cifar100_ega_features.npy')

    features = np.load(features_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(labels_path)
    
    train_features = features[:8000]
    train_labels = labels[:8000]
    
    dataset = SemanticCIFAR100Dataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    model.train()
    print('Starting Final EGA Training with Calibrated Hypersphere Manifold...')
    for epoch in range(120):
        total_loss = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(a), model(p), model(n))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/120, Loss: {total_loss / len(loader):.4f}')

    torch.save(model.state_dict(), model_save_path)
    model.eval()
    all_feats = torch.from_numpy(features).to(device)
    with torch.no_grad():
        transformed = model(all_feats).cpu().numpy()
    np.save(output_npy_path, transformed)
    print(f'Finished training. Saved to {output_npy_path}')

if __name__ == '__main__':
    main()

# (venv) cc@uc-a100:~/hpdic/EGA$ python ./scripts/02_train_ega_cifar100.py 
# Starting Final EGA Training with Calibrated Hypersphere Manifold...
# Epoch 10/120, Loss: 0.0286
# Epoch 20/120, Loss: 0.0222
# Epoch 30/120, Loss: 0.0181
# Epoch 40/120, Loss: 0.0161
# Epoch 50/120, Loss: 0.0153
# Epoch 60/120, Loss: 0.0127
# Epoch 70/120, Loss: 0.0121
# Epoch 80/120, Loss: 0.0116
# Epoch 90/120, Loss: 0.0094
# Epoch 100/120, Loss: 0.0086
# Epoch 110/120, Loss: 0.0074
# Epoch 120/120, Loss: 0.0079
# Finished training. Saved to /home/cc/hpdic/EGA/embeddings/cifar100_ega_features.npy    