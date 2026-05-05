# scripts/lora_rank_scaling.py
# LoRA rank 扫描：从 64 到 512，看在 CIFAR-100 ID 和 Aircraft OOD 上的表现

import os, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from utils_ega import split_by_class, eval_method, TripletDataset

class LoRAAdapter(nn.Module):
    def __init__(self, dim=512, rank=128, alpha=16):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T
        out = x + self.scale * delta
        return F.normalize(out, p=2, dim=1)

def train_lora_triplet(train_feats, train_labels, device, dim, epochs=150,
                       batch_size=256, margin=0.2, rank=128):
    loader = DataLoader(TripletDataset(train_feats, train_labels),
                        batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)
    model = LoRAAdapter(dim=dim, rank=rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    model.train()
    for epoch in range(epochs):
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                loss = criterion(model(a), model(p), model(n))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        # if (epoch + 1) % 50 == 0:
        #     print(f'      epoch {epoch+1}/{epochs}')
    return model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f'GPU: {torch.cuda.get_device_name(0)}\n')

    base_dir = os.path.expanduser('~/hpdic/EGA/embeddings')

    # ---- CIFAR-100 ID ----
    c100_feat = np.load(os.path.join(base_dir, 'cifar100_vit_b32_features.npy')).astype(np.float32)
    c100_feat = c100_feat / np.linalg.norm(c100_feat, axis=1, keepdims=True)
    c100_labels = np.load(os.path.join(base_dir, 'cifar100_vit_b32_labels.npy'))

    # ---- Aircraft OOD ----
    air_feat = np.load(os.path.join(base_dir, 'aircraft_test_vit_b32_features.npy')).astype(np.float32)
    air_feat = air_feat / np.linalg.norm(air_feat, axis=1, keepdims=True)
    air_labels = np.load(os.path.join(base_dir, 'aircraft_test_vit_b32_labels.npy'))
    (air_train, air_train_lab), (air_test, air_test_lab) = split_by_class(air_feat, air_labels)

    dim = air_feat.shape[1]

    print(f'{"Rank":<8} {"CIFAR-100 LP@1":<18} {"Aircraft LP@1":<18}')
    print('-' * 44)

    for rank in [64, 128, 256, 512]:
        print(f'{rank:<8}', end='', flush=True)

        # OOD: Aircraft
        model = train_lora_triplet(air_train, air_train_lab, device, dim,
                                    epochs=150, rank=rank)
        model.eval()
        with torch.no_grad():
            feats_ood = model(torch.from_numpy(air_test).float().to(device)).cpu().numpy()
        ood_lp, _ = eval_method(feats_ood, air_test_lab)

        # ID: CIFAR-100
        id_model = train_lora_triplet(c100_feat, c100_labels, device, dim,
                                       epochs=150, rank=rank)
        id_model.eval()
        with torch.no_grad():
            id_feats = id_model(torch.from_numpy(c100_feat).float().to(device)).cpu().numpy()
        id_lp, _ = eval_method(id_feats, c100_labels)

        print(f'{id_lp:<18.4f} {ood_lp:<18.4f}')

    # EGA 参考值
    print('-' * 44)
    print(f'EGA      0.7050            0.6110')


if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/37_lora_rank_scaling.py 
# GPU: NVIDIA A100 80GB PCIe

# Rank     CIFAR-100 LP@1     Aircraft LP@1     
# --------------------------------------------
# 64            epoch 50/150
#       epoch 100/150
#       epoch 150/150
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#       epoch 50/150
#       epoch 100/150
#       epoch 150/150
# 0.6456             0.5655            
# 128           epoch 50/150
#       epoch 100/150
#       epoch 150/150
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#       epoch 50/150
#       epoch 100/150
#       epoch 150/150
# 0.6460             0.5595            
# 256           epoch 50/150
#       epoch 100/150
#       epoch 150/150
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#       epoch 50/150
#       epoch 100/150
#       epoch 150/150
# 0.6472             0.5655            
# 512           epoch 50/150
#       epoch 100/150
#       epoch 150/150
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
#       epoch 50/150
#       epoch 100/150
#       epoch 150/150
# 0.6480             0.5298            
# --------------------------------------------
# EGA      0.7050            0.6110
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 