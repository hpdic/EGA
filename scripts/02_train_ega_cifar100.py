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
import faiss
from tqdm import tqdm
from models.ega_mlp import EGAMLP

class CIFAR100Dataset(Dataset):
    def __init__(self, features, neighbors):
        self.features = torch.from_numpy(features).float()
        self.neighbors = neighbors
        self.num_samples = len(features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anchor = self.features[idx]
        pos_idx = np.random.choice(self.neighbors[idx])
        positive = self.features[pos_idx]
        neg_idx = np.random.randint(0, self.num_samples)
        while neg_idx in self.neighbors[idx] or neg_idx == idx:
            neg_idx = np.random.randint(0, self.num_samples)
        negative = self.features[neg_idx]
        return anchor, positive, negative

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    
    features_path = os.path.join(base_dir, 'embeddings/cifar100_vit_b32_features.npy')
    model_save_path = os.path.join(base_dir, 'models/ega_bridge_cifar100.pth')
    output_npy_path = os.path.join(base_dir, 'embeddings/cifar100_ega_features.npy')

    features = np.load(features_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    train_features = features[:8000]
    
    index = faiss.IndexFlatL2(512)
    index.add(train_features)
    _, indices = index.search(train_features, 31)
    neighbors = indices[:, 1:]
    
    dataset = CIFAR100Dataset(train_features, neighbors)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.TripletMarginLoss(margin=1.2, p=2)

    model.train()
    for epoch in range(120):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_save_path)
    model.eval()
    all_feats = torch.from_numpy(features).to(device)
    with torch.no_grad():
        transformed = model(all_feats).cpu().numpy()
    np.save(output_npy_path, transformed)
    print(f'Finished training. Saved to {output_npy_path}')

if __name__ == '__main__':
    main()