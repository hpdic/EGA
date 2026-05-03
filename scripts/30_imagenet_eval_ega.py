# scripts/31_imagenet_train_eval_ega.py
# 训练 + 评估 EGA（ImageNet-100，80/20 划分）

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import faiss

from models.ega_mlp import EGAMLP

def train_and_eval_ega(features_path, labels_path, output_dir, epochs=100, batch_size=128, lr=1e-3, test_size=0.2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load features and labels
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Total samples: {len(features)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Split train/test (80/20)
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), 
        test_size=test_size, 
        random_state=42, 
        stratify=labels
    )
    
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    test_features = features[test_idx]
    test_labels = labels[test_idx]
    
    print(f"Train samples: {len(train_features)}")
    print(f"Test samples: {len(test_features)}")
    
    # Convert to tensors
    train_features = torch.FloatTensor(train_features)
    train_labels = torch.LongTensor(train_labels)
    test_features = torch.FloatTensor(test_features)
    test_labels = torch.LongTensor(test_labels)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = features.shape[1]  # 512
    model = EGAMLP(input_dim=input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Accuracy={accuracy:.2f}%")
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "ega_imagenet100.pth")
    print(f"\nModel saved to {output_dir / 'ega_imagenet100.pth'}")
    
    # Evaluation on test set
    model.eval()
    all_ega_features = []
    all_test_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            all_ega_features.append(outputs.cpu().numpy())
            all_test_labels.append(batch_labels.numpy())
    
    ega_features = np.concatenate(all_ega_features, axis=0)
    test_labels = np.concatenate(all_test_labels, axis=0)
    
    print(f"\nTest EGA features shape: {ega_features.shape}")
    
    # Build FAISS index on test set
    dim = ega_features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(ega_features)
    
    # Search
    k = 100
    D, I = index.search(ega_features, k)
    
    # Compute metrics
    lp_at_1 = 0
    ar_at_1 = 0
    
    for i in range(len(test_labels)):
        if test_labels[I[i, 0]] == test_labels[i]:
            lp_at_1 += 1
        if I[i, 0] == i:
            ar_at_1 += 1
    
    lp_at_1 = lp_at_1 / len(test_labels)
    ar_at_1 = ar_at_1 / len(test_labels)
    
    print(f"\n=== ImageNet-100 EGA Evaluation (Test Set) ===")
    print(f"Label Precision@1: {lp_at_1:.4f}")
    print(f"ANNS Recall@1: {ar_at_1:.4f}")
    
    # Save results
    with open(output_dir / "imagenet100_ega_results.txt", "w") as f:
        f.write(f"Label Precision@1: {lp_at_1:.4f}\n")
        f.write(f"ANNS Recall@1: {ar_at_1:.4f}\n")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    features_path = "/home/cc/hpdic/EGA/embeddings/imagenet100_features.npy"
    labels_path = "/home/cc/hpdic/EGA/embeddings/imagenet100_labels.npy"
    output_dir = "/home/cc/hpdic/EGA/models"
    
    train_and_eval_ega(features_path, labels_path, output_dir, epochs=100, batch_size=128, lr=1e-3, test_size=0.2)