# scripts/40_imagenet1000_eval_ega.py
# 评估 EGA（ImageNet-1000，测试集）

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import faiss

from models.ega_mlp import EGAMLP

def eval_ega_imagenet1000(features_path, labels_path, model_path, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load features and labels
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Convert to tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    # Create dataloader
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    # Load model
    input_dim = features.shape[1]  # 512
    checkpoint = torch.load(model_path, map_location=device)
    
    ega_model = EGAMLP(input_dim=input_dim).to(device)
    ega_model.load_state_dict(checkpoint['ega_model'])
    ega_model.eval()
    
    classifier = torch.nn.Linear(512, 1000).to(device)
    classifier.load_state_dict(checkpoint['classifier'])
    classifier.eval()
    
    # Extract features
    all_ega_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            ega_features = ega_model(batch_features)
            outputs = classifier(ega_features)
            all_ega_features.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.numpy())
    
    ega_features = np.concatenate(all_ega_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"EGA features shape: {ega_features.shape}")
    
    # Build FAISS index
    dim = ega_features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(ega_features)
    
    # Search
    k = 100
    D, I = index.search(ega_features, k)
    
    # Compute metrics
    lp_at_1 = 0
    ar_at_1 = 0
    
    for i in range(len(labels)):
        if labels[I[i, 0]] == labels[i]:
            lp_at_1 += 1
        if I[i, 0] == i:
            ar_at_1 += 1
    
    lp_at_1 = lp_at_1 / len(labels)
    ar_at_1 = ar_at_1 / len(labels)
    
    print(f"\n=== ImageNet-1000 EGA Evaluation ===")
    print(f"Label Precision@1: {lp_at_1:.4f}")
    print(f"ANNS Recall@1: {ar_at_1:.4f}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "imagenet1000_ega_results.txt", "w") as f:
        f.write(f"Label Precision@1: {lp_at_1:.4f}\n")
        f.write(f"ANNS Recall@1: {ar_at_1:.4f}\n")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    features_path = "/home/cc/hpdic/EGA/embeddings/imagenet1000_features.npy"
    labels_path = "/home/cc/hpdic/EGA/embeddings/imagenet1000_labels.npy"
    model_path = "/home/cc/hpdic/EGA/models/ega_imagenet1000.pth"
    output_dir = "/home/cc/hpdic/EGA/embeddings"
    
    eval_ega_imagenet1000(features_path, labels_path, model_path, output_dir)