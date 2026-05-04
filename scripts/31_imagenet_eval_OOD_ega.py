# scripts/49_eval_150epoch_ega.py
# 评估 150 epoch 版本

import torch
import numpy as np
import os
import faiss
from sklearn.model_selection import train_test_split
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

def run_evaluation(features, labels, label_text, k_list=[1, 3, 5, 10]):
    print(f'\n=== {label_text} ===')
    
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
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
    
    features = np.load(os.path.join(base_dir, 'embeddings/imagenet1000_features.npy'))
    labels = np.load(os.path.join(base_dir, 'embeddings/imagenet1000_labels.npy'))
    
    _, test_idx = train_test_split(
        np.arange(len(features)), 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    test_features = features[test_idx]
    test_labels = labels[test_idx]
    
    print(f'Test samples (unseen classes): {len(test_features)}')
    print(f'Test classes: {len(np.unique(test_labels))}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ega_model = EGAMLP(input_dim=512).to(device)
    ega_model.load_state_dict(torch.load(os.path.join(base_dir, 'models/ega_imagenet1000_150epoch.pth'), map_location=device))
    ega_model.eval()
    
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_features).to(device)
        ega_out = ega_model(test_tensor).cpu().numpy()
    
    k_list = [1, 3, 5, 10]
    
    run_evaluation(test_features, test_labels, 'Original CLIP (Unseen Classes)', k_list)
    run_evaluation(ega_out, test_labels, 'EGA 150 Epoch (Unseen Classes)', k_list)

if __name__ == '__main__':
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/31_imagenet_eval_clip.py 
# Test samples (unseen classes): 6949
# Test classes: 1000

# === Original CLIP (Unseen Classes) ===

# K=1
#   nprobe 1: LP = 0.3320 | Recall = 0.8441
#   nprobe 5: LP = 0.3654 | Recall = 0.9983
#   nprobe 10: LP = 0.3654 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.2585 | Recall = 0.8324
#   nprobe 5: LP = 0.2890 | Recall = 0.9969
#   nprobe 10: LP = 0.2894 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.2072 | Recall = 0.8224
#   nprobe 5: LP = 0.2362 | Recall = 0.9963
#   nprobe 10: LP = 0.2366 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.1400 | Recall = 0.7998
#   nprobe 5: LP = 0.1620 | Recall = 0.9938
#   nprobe 10: LP = 0.1625 | Recall = 1.0000

# === EGA 150 Epoch (Unseen Classes) ===

# K=1
#   nprobe 1: LP = 0.5276 | Recall = 0.8740
#   nprobe 5: LP = 0.5587 | Recall = 0.9977
#   nprobe 10: LP = 0.5598 | Recall = 1.0000

# K=3
#   nprobe 1: LP = 0.4277 | Recall = 0.8556
#   nprobe 5: LP = 0.4687 | Recall = 0.9973
#   nprobe 10: LP = 0.4703 | Recall = 1.0000

# K=5
#   nprobe 1: LP = 0.3536 | Recall = 0.8466
#   nprobe 5: LP = 0.3991 | Recall = 0.9963
#   nprobe 10: LP = 0.3999 | Recall = 1.0000

# K=10
#   nprobe 1: LP = 0.2502 | Recall = 0.8317
#   nprobe 5: LP = 0.2880 | Recall = 0.9948
#   nprobe 10: LP = 0.2892 | Recall = 1.0000
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 