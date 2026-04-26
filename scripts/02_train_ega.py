import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import faiss
from tqdm import tqdm
from models.ega_mlp import EGAMLP

class AdvancedTripletDataset(Dataset):
    def __init__(self, features, neighbors):
        self.features = torch.from_numpy(features).float()
        self.neighbors = neighbors
        self.num_samples = len(features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anchor = self.features[idx]
        # Pick a positive sample from the pre-calculated neighbors
        pos_idx = np.random.choice(self.neighbors[idx])
        positive = self.features[pos_idx]
        
        # Harder negative mining: pick a random point that is definitely not a neighbor
        neg_idx = np.random.randint(0, self.num_samples)
        while neg_idx in self.neighbors[idx] or neg_idx == idx:
            neg_idx = np.random.randint(0, self.num_samples)
        negative = self.features[neg_idx]
        
        return anchor, positive, negative

def get_original_neighbors(features, k=20):
    print("Pre-calculating manifold neighbors...")
    dim = features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(features)
    _, indices = index.search(features, k + 1)
    return indices[:, 1:]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_vit_b32_features.npy")
    output_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_ega_features.npy")

    features = np.load(features_path).astype(np.float32)
    # Ensure input features are normalized before training
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    train_features = features[:8000]
    
    neighbors = get_original_neighbors(train_features, k=30)
    dataset = AdvancedTripletDataset(train_features, neighbors)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = EGAMLP(input_dim=512, hidden_dim=2048).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    # Increased margin to force clearer geometric separation
    criterion = nn.TripletMarginLoss(margin=1.2, p=2)

    print("Training enhanced EGA model...")
    model.train()
    for epoch in range(150):
        epoch_loss = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            
            a_out, p_out, n_out = model(a), model(p), model(n)
            loss = criterion(a_out, p_out, n_out)
            
            # Semantic reconstruction loss to prevent drifting too far
            recon_loss = torch.mean((a_out - a)**2)
            
            total_loss = loss + 0.05 * recon_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/150], Loss: {epoch_loss/len(loader):.6f}")

    model.eval()
    all_features_tensor = torch.from_numpy(features).float().to(device)
    with torch.no_grad():
        transformed = model(all_features_tensor).cpu().numpy()
    
    np.save(output_path, transformed)
    print(f"Enhanced EGA Transformation complete. Saved to {output_path}")

if __name__ == "__main__":
    main()

#
# Example Output:
#

# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/02_train_ega.py
# Pre-calculating manifold neighbors...
# Training enhanced EGA model...
# Epoch [25/150], Loss: 0.217645
# Epoch [50/150], Loss: 0.205627
# Epoch [75/150], Loss: 0.200030
# Epoch [100/150], Loss: 0.192480
# Epoch [125/150], Loss: 0.190569
# Epoch [150/150], Loss: 0.192629
# Enhanced EGA Transformation complete. Saved to /home/cc/hpdic/EGA/embeddings/cifar10_ega_features.npy    