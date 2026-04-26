import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import faiss
from tqdm import tqdm

# Import the model we defined earlier
from models.ega_mlp import EGAMLP

class NeighborDataset(Dataset):
    def __init__(self, features, neighbors):
        self.features = torch.from_numpy(features).float()
        self.neighbors = neighbors

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return the anchor point and a randomly sampled original neighbor
        anchor = self.features[idx]
        neighbor_idx = np.random.choice(self.neighbors[idx])
        positive = self.features[neighbor_idx]
        return anchor, positive

def get_original_neighbors(features, k=10):
    print("Pre-calculating original manifold neighbors...")
    dim = features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(features)
    # We find k+1 neighbors because the closest one is always the point itself
    _, indices = index.search(features, k + 1)
    return indices[:, 1:]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define absolute paths
    features_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_vit_b32_features.npy")
    model_save_path = os.path.expanduser("~/hpdic/EGA/models/ega_bridge.pth")
    output_features_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_ega_features.npy")

    # 1. Load original features
    features = np.load(features_path).astype(np.float32)
    train_features = features[:8000] # Use the same split as the eval script
    
    # 2. Find neighbors in the twisted manifold
    neighbors = get_original_neighbors(train_features, k=20)
    
    # 3. Setup Dataset and Model
    dataset = NeighborDataset(train_features, neighbors)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = EGAMLP(input_dim=512, hidden_dim=1024).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 4. Training Loop
    print("Training EGA to flatten the manifold...")
    model.train()
    for epoch in range(50):
        epoch_loss = 0
        for anchors, positives in loader:
            anchors, positives = anchors.to(device), positives.to(device)
            
            optimizer.zero_grad()
            
            # Map both points through the EGA bridge
            anchor_out = model(anchors)
            positive_out = model(positives)
            
            # Loss: Neighbors in the manifold should be close in Euclidean space
            loss = criterion(anchor_out, positive_out)
            
            # Regularization: Keep the output close to the original to preserve semantics
            reg_loss = criterion(anchor_out, anchors)
            
            total_loss = loss + 0.1 * reg_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {epoch_loss/len(loader):.6f}")

    # 5. Transform all features and save
    print("Transforming all features with the trained EGA bridge...")
    model.eval()
    all_features_tensor = torch.from_numpy(features).float().to(device)
    with torch.no_grad():
        transformed_features = model(all_features_tensor).cpu().numpy()
    
    np.save(output_features_path, transformed_features)
    torch.save(model.state_dict(), model_save_path)
    print(f"Success: Transformed features saved to {output_features_path}")

if __name__ == "__main__":
    main()