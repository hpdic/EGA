# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss
import os

def calculate_recall(retrieved, gt, k):
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)

def run_evaluation(features_path, label_text, k_list):
    print(f'\nEvaluating: {label_text}')
    if not os.path.exists(features_path):
        print(f'File not found: {features_path}')
        return
        
    features = np.load(features_path).astype(np.float32)
    base = features[:8000]
    query = features[8000:]
    dim = base.shape[1]
    
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    nlist = 1000
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f'\nRecall@{k}:')
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            print(f'nprobe {nprobe}: {calculate_recall(ret, gt, k):.4f}')

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA/embeddings')
    orig = os.path.join(base_dir, 'cifar100_vit_b32_features.npy')
    ega = os.path.join(base_dir, 'cifar100_ega_features.npy')
    k_list = [1, 3, 5, 10]
    
    run_evaluation(orig, 'CIFAR-100 Original', k_list)
    run_evaluation(ega, 'CIFAR-100 EGA', k_list)

if __name__ == '__main__':
    main()

#
# Example Output:
#

# (venv) cc@uc-a100:~/hpdic/EGA$ python ./scripts/05_eval_generalization.py 

# Evaluating: CIFAR-100 Original
# WARNING clustering 8000 points to 1000 centroids: please provide at least 39000 training points

# Recall@1:
# nprobe 1: 0.5130
# nprobe 5: 0.8455
# nprobe 10: 0.9265

# Recall@3:
# nprobe 1: 0.4432
# nprobe 5: 0.8133
# nprobe 10: 0.9088

# Recall@5:
# nprobe 1: 0.4046
# nprobe 5: 0.7825
# nprobe 10: 0.8886

# Recall@10:
# nprobe 1: 0.3399
# nprobe 5: 0.7319
# nprobe 10: 0.8557

# Evaluating: CIFAR-100 EGA
# WARNING clustering 8000 points to 1000 centroids: please provide at least 39000 training points

# Recall@1:
# nprobe 1: 0.6865
# nprobe 5: 0.9865
# nprobe 10: 0.9995

# Recall@3:
# nprobe 1: 0.6218
# nprobe 5: 0.9788
# nprobe 10: 0.9977

# Recall@5:
# nprobe 1: 0.5728
# nprobe 5: 0.9710
# nprobe 10: 0.9963

# Recall@10:
# nprobe 1: 0.4780
# nprobe 5: 0.9484
# nprobe 10: 0.9930
# (venv) cc@uc-a100:~/hpdic/EGA$ 