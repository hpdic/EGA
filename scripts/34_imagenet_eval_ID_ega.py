# scripts/52_eval_clip_ega_id.py
# 只评估 CLIP 和 EGA 的 In-Distribution 性能

import torch
import numpy as np
import os
import faiss
from models.ega_mlp import EGAMLP

def calculate_label_precision(retrieved, base_labels, query_labels, k):
    count = 0
    for i in range(len(query_labels)):
        neighbor_labels = base_labels[retrieved[i, :k]]
        count += np.sum(neighbor_labels == query_labels[i])
    return count / (len(query_labels) * k)

def calculate_anns_recall(retrieved, gt, k):
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)

def run_id_evaluation(features, labels, label_text, k_list=[1, 3, 5, 10]):
    print(f'\n=== {label_text} (In-Distribution) ===')
    
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    # 75/25 split
    split_idx = int(len(features) * 0.75)
    base = features[:split_idx]
    base_labels = labels[:split_idx]
    query = features[split_idx:]
    query_labels = labels[split_idx:]
    
    dim = base.shape[1]
    
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    nlist = 10
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f'\nK={k}')
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            recall = calculate_anns_recall(ret, gt, k)
            precision = calculate_label_precision(ret, base_labels, query_labels, k)
            print(f'  nprobe {nprobe}: LP = {precision:.4f} | Recall = {recall:.4f}')

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    # Load all features
    features = np.load(os.path.join(embed_dir, 'imagenet1000_features.npy')).astype(np.float32)
    labels = np.load(os.path.join(embed_dir, 'imagenet1000_labels.npy'))
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    print(f'Total samples: {len(features)}, Total classes: {len(np.unique(labels))}')
    print('In-Distribution evaluation (75/25 split on all 1000 classes)\n')
    
    # 1. CLIP (frozen) - just evaluate raw features
    run_id_evaluation(features, labels, 'CLIP (frozen)')
    
    # 2. EGA - load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ega_model = EGAMLP(input_dim=512).to(device)
    ega_model.load_state_dict(torch.load(os.path.join(base_dir, 'models/ega_imagenet1000_150epoch.pth'), map_location=device))
    ega_model.eval()
    
    with torch.no_grad():
        ega_out = ega_model(torch.from_numpy(features).float().to(device)).cpu().numpy()
    
    run_id_evaluation(ega_out, labels, 'EGA 150 Epoch')

if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/34_imagenet_eval_ID_ega.py 
# Total samples: 34745, Total classes: 1000
# In-Distribution evaluation (75/25 split on all 1000 classes)


# === CLIP (frozen) (In-Distribution) ===

# K=1
#   nprobe 1: LP = 0.4616 | Recall = 0.8454
#   nprobe 5: LP = 0.4818 | Recall = 0.9963
#   nprobe 10: LP = 0.4826 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.4092 | Recall = 0.8350
#   nprobe 5: LP = 0.4365 | Recall = 0.9954
#   nprobe 10: LP = 0.4370 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.3745 | Recall = 0.8290
#   nprobe 5: LP = 0.4045 | Recall = 0.9946
#   nprobe 10: LP = 0.4053 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.3188 | Recall = 0.8190
#   nprobe 5: LP = 0.3511 | Recall = 0.9933
#   nprobe 10: LP = 0.3522 | Recall = 1.0000

# === EGA 150 Epoch (In-Distribution) ===

# K=1
#   nprobe 1: LP = 0.6075 | Recall = 0.8695
#   nprobe 5: LP = 0.6215 | Recall = 0.9985
#   nprobe 10: LP = 0.6217 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.5690 | Recall = 0.8612
#   nprobe 5: LP = 0.5943 | Recall = 0.9983
#   nprobe 10: LP = 0.5946 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.5422 | Recall = 0.8549
#   nprobe 5: LP = 0.5729 | Recall = 0.9980
#   nprobe 10: LP = 0.5732 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.4888 | Recall = 0.8454
#   nprobe 5: LP = 0.5306 | Recall = 0.9975
#   nprobe 10: LP = 0.5312 | Recall = 1.0000
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 