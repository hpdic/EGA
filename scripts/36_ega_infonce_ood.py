# scripts/ega_infonce_ood.py
import os, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from models.ega_mlp import EGAMLP
from utils_ega import SupConInfoNCE, split_by_class, eval_method

class FlatDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = torch.from_numpy(feats).float()
        self.labels = labels
    def __len__(self): return len(self.feats)
    def __getitem__(self, idx): return self.feats[idx], self.labels[idx]

def train_ega_infonce(train_feats, train_labels, device, dim, epochs=150):
    loader = DataLoader(
        FlatDataset(train_feats, train_labels),
        batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True         # pin_memory 加速 CPU→GPU 传输
    )

    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SupConInfoNCE().to(device)
    scaler = GradScaler()                       # A100 混合精度

    model.train()
    for epoch in range(epochs):
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():                    # 自动混合精度
                loss = criterion(model(feats), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if (epoch + 1) % 30 == 0:
            print(f'    epoch {epoch+1}/{epochs}')

    return model

def run_ood(name, feat_path, label_path, device):
    features = np.load(feat_path).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.load(label_path)

    if "cifar10" in feat_path.lower():
        c100_feat = np.load(os.path.expanduser(
            '~/hpdic/EGA/embeddings/cifar100_vit_b32_features.npy'))
        c100_feat = c100_feat / np.linalg.norm(c100_feat, axis=1, keepdims=True)
        c100_labels = np.load(os.path.expanduser(
            '~/hpdic/EGA/embeddings/cifar100_vit_b32_labels.npy'))
        train_feats, train_labels = c100_feat, c100_labels
        test_feats, test_labels = features, labels
    else:
        (train_feats, train_labels), (test_feats, test_labels) = split_by_class(features, labels)

    dim = features.shape[1]
    model = train_ega_infonce(train_feats, train_labels, device, dim)
    model.eval()
    with torch.no_grad(), autocast():           # 推理也用混合精度
        transformed = model(
            torch.from_numpy(test_feats).float().to(device)
        ).cpu().numpy()
    lp, ar = eval_method(transformed, test_labels)
    print(f'{name}  EGA+InfoNCE  LP@1={lp:.4f}  AR@1={ar:.4f}')
    return lp, ar

if __name__ == '__main__':
    # A100 优化设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True   # 自动找最优卷积算法
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    base_dir = os.path.expanduser('~/hpdic/EGA/embeddings')
    datasets = [
        ("FGVC-Aircraft", "aircraft_test_vit_b32_features.npy", "aircraft_test_vit_b32_labels.npy"),
        ("Food-101",      "food101_features.npy",               "food101_labels.npy"),
        ("CIFAR-10",      "cifar10_vit_b32_features.npy",      "cifar10_vit_b32_labels.npy"),
    ]
    for name, f, l in datasets:
        run_ood(name, os.path.join(base_dir, f), os.path.join(base_dir, l), device)


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/36_ega_infonce_ood.py 
# Using GPU: NVIDIA A100 80GB PCIe
# Memory: 85.1 GB
# /home/cc/hpdic/EGA/scripts/36_ega_infonce_ood.py:27: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
#   scaler = GradScaler()                       # A100 混合精度
# /home/cc/hpdic/EGA/scripts/36_ega_infonce_ood.py:37: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with autocast():                    # 自动混合精度
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
# /home/cc/hpdic/EGA/scripts/36_ega_infonce_ood.py:69: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with torch.no_grad(), autocast():           # 推理也用混合精度
# WARNING clustering 501 points to 100 centroids: please provide at least 3900 training points
# FGVC-Aircraft  EGA+InfoNCE  LP@1=0.4583  AR@1=0.5893
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
# Food-101  EGA+InfoNCE  LP@1=0.6672  AR@1=0.6580
#     epoch 30/150
#     epoch 60/150
#     epoch 90/150
#     epoch 120/150
#     epoch 150/150
# CIFAR-10  EGA+InfoNCE  LP@1=0.7100  AR@1=0.5128
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 