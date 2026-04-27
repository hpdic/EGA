# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from torchvision.datasets import CIFAR100
from tqdm import tqdm

class SRLLoss(nn.Module):
    """
    Simplified implementation of SRL (CVPR 2025) geometric constraints.
    Focuses on Uniformity (Wrapping) and Smoothness (Homogeneity).
    """
    def __init__(self, temperature=0.1, lambda_gen=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_gen = lambda_gen

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        # 1. Similarity Matrix
        logits = torch.matmul(features, features.T) / self.temperature
        
        # 2. Mask for positive pairs
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(features.device)
        
        # 3. Geometric Uniformity (Wrapping Logic)
        # Penalize over-clustering in tiny regions to force features to wrap around the sphere
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(features.device))
        uniformity_loss = torch.log(exp_logits.sum(dim=1)).mean()
        
        # 4. Local Homogeneity (Alignment Logic)
        pos_logits = (exp_logits * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        homogeneity_loss = -torch.log(pos_logits + 1e-8).mean()
        
        return homogeneity_loss + self.lambda_gen * uniformity_loss

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    data_dir = os.path.join(base_dir, 'data')
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(ckpt_dir, 'srl_vit_b32.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.float()
    model.train()
    
    # Freeze Text Encoder
    for param in model.transformer.parameters():
        param.requires_grad = False
        
    train_loader = DataLoader(
        CIFAR100(root=data_dir, download=True, train=True, transform=preprocess),
        batch_size=256, shuffle=True, num_workers=8, drop_last=True
    )
    
    optimizer = torch.optim.AdamW(model.visual.parameters(), lr=1e-5, weight_decay=0.01)
    srl_criterion = SRLLoss().to(device)
    
    print('Starting SRL training on CIFAR100...')
    for epoch in range(5):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/5')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model.encode_image(imgs).float()
            loss = srl_criterion(features, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    torch.save(model.state_dict(), checkpoint_path)
    print(f'SRL Weights saved to: {checkpoint_path}')

if __name__ == '__main__':
    main()