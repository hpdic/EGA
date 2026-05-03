# scripts/29_imagenet_train_ega.py
# 训练 EGA 模型（ImageNet-100）

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

# 导入 EGA 模型
from models.ega_mlp import EGAMLP


def train_ega(features_path, labels_path, output_dir, epochs=100, batch_size=128, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load features and labels
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Convert to tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    # Create dataset and dataloader
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
        
        for batch_features, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "ega_imagenet100.pth")
    print(f"\nModel saved to {output_dir / 'ega_imagenet100.pth'}")

if __name__ == '__main__':
    features_path = "/home/cc/hpdic/EGA/embeddings/imagenet100_features.npy"
    labels_path = "/home/cc/hpdic/EGA/embeddings/imagenet100_labels.npy"
    output_dir = "/home/cc/hpdic/EGA/models"
    
    train_ega(features_path, labels_path, output_dir, epochs=100, batch_size=128, lr=1e-3)