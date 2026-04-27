# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from torchvision.datasets import CIFAR100
from tqdm import tqdm

class IConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        log_probs = F.log_softmax(sim_matrix, dim=1)
        
        target_probs = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
        return loss

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    data_dir = os.path.join(base_dir, 'data')
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(ckpt_dir, 'icon_vit_b32.pth')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model, preprocess = clip.load('ViT-B/32', device=device)
    
    model.float()
    
    model.train()
    for param in model.transformer.parameters():
        param.requires_grad = False
        
    train_dataset = CIFAR100(root=data_dir, download=True, train=True, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, drop_last=True)
    
    optimizer = torch.optim.AdamW(model.visual.parameters(), lr=1e-5, weight_decay=0.01)
    icon_criterion = IConLoss(temperature=0.07).to(device)
    
    epochs = 5 
    
    print(f'Starting ICon fine tuning on CIFAR100 for {epochs} epochs...')
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            features = model.encode_image(imgs).float()
            
            loss = icon_criterion(features, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
        
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Training complete. Weights saved to: {checkpoint_path}')

if __name__ == '__main__':
    main()

# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts$ python ICon/00_train_icon.py 
# Using device: cuda
# /home/cc/hpdic/EGA/venv/lib/python3.13/site-packages/torchvision/datasets/cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
#   entry = pickle.load(f, encoding="latin1")
# Starting ICon fine tuning on CIFAR100 for 5 epochs...
# Epoch 1/5: 100%|███████████████████████████████| 195/195 [01:41<00:00,  1.92it/s, loss=0.9244]
# Epoch 1 completed. Average Loss: 1.0309
# Epoch 2/5: 100%|███████████████████████████████| 195/195 [01:43<00:00,  1.89it/s, loss=0.6014]
# Epoch 2 completed. Average Loss: 0.5904
# Epoch 3/5: 100%|███████████████████████████████| 195/195 [01:42<00:00,  1.90it/s, loss=0.4289]
# Epoch 3 completed. Average Loss: 0.4071
# Epoch 4/5: 100%|███████████████████████████████| 195/195 [01:43<00:00,  1.89it/s, loss=0.3557]
# Epoch 4 completed. Average Loss: 0.3013
# Epoch 5/5: 100%|███████████████████████████████| 195/195 [01:42<00:00,  1.90it/s, loss=0.2280]
# Epoch 5 completed. Average Loss: 0.2083
# Training complete. Weights saved to: /home/cc/hpdic/EGA/checkpoints/icon_vit_b32.pth
# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts$ 
