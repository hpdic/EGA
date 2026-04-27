# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026
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
    print(f"\nEvaluating: {label_text}")
    if not os.path.exists(features_path):
        print(f"File not found: {features_path}")
        return
        
    features = np.load(features_path).astype(np.float32)
    base = features[:8000]
    query = features[8000:]
    dim = base.shape[1]
    
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    nlist = 100
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f"\nRecall@{k}:")
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            print(f"nprobe {nprobe}: {calculate_recall(ret, gt, k):.4f}")

def main():
    base_dir = os.path.expanduser("~/hpdic/EGA/embeddings")
    orig = os.path.join(base_dir, "cifar100_vit_b32_features.npy")
    ega = os.path.join(base_dir, "cifar100_ega_features.npy")
    icon = os.path.join(base_dir, "cifar100_icon_features.npy")
    srl = os.path.join(base_dir, "cifar100_srl_features.npy")
    k_list = [1, 3, 5, 10]
    
    run_evaluation(orig, "CIFAR-100 Original CLIP", k_list)
    run_evaluation(ega, "CIFAR-100 EGA", k_list)
    run_evaluation(icon, "CIFAR-100 ICon", k_list)
    run_evaluation(srl, "CIFAR-100 SRL", k_list)

if __name__ == "__main__":
    main()

# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ python 04_eval_anns_recall.py 

# Evaluating: CIFAR-100 Original CLIP

# Recall@1:
# nprobe 1: 0.6665
# nprobe 5: 0.9370
# nprobe 10: 0.9800

# Recall@3:
# nprobe 1: 0.6363
# nprobe 5: 0.9292
# nprobe 10: 0.9765

# Recall@5:
# nprobe 1: 0.6198
# nprobe 5: 0.9203
# nprobe 10: 0.9736

# Recall@10:
# nprobe 1: 0.5930
# nprobe 5: 0.9060
# nprobe 10: 0.9684

# Evaluating: CIFAR-100 EGA

# Recall@1:
# nprobe 1: 0.7985
# nprobe 5: 0.9835
# nprobe 10: 0.9990

# Recall@3:
# nprobe 1: 0.7823
# nprobe 5: 0.9798
# nprobe 10: 0.9967

# Recall@5:
# nprobe 1: 0.7767
# nprobe 5: 0.9777
# nprobe 10: 0.9953

# Recall@10:
# nprobe 1: 0.7600
# nprobe 5: 0.9733
# nprobe 10: 0.9946

# Evaluating: CIFAR-100 ICon

# Recall@1:
# nprobe 1: 0.9615
# nprobe 5: 0.9975
# nprobe 10: 0.9995

# Recall@3:
# nprobe 1: 0.9572
# nprobe 5: 0.9985
# nprobe 10: 0.9997

# Recall@5:
# nprobe 1: 0.9540
# nprobe 5: 0.9982
# nprobe 10: 0.9996

# Recall@10:
# nprobe 1: 0.9507
# nprobe 5: 0.9983
# nprobe 10: 0.9998

# Evaluating: CIFAR-100 SRL

# Recall@1:
# nprobe 1: 0.9570
# nprobe 5: 0.9985
# nprobe 10: 0.9995

# Recall@3:
# nprobe 1: 0.9545
# nprobe 5: 0.9988
# nprobe 10: 0.9997

# Recall@5:
# nprobe 1: 0.9533
# nprobe 5: 0.9985
# nprobe 10: 0.9996

# Recall@10:
# nprobe 1: 0.9485
# nprobe 5: 0.9987
# nprobe 10: 0.9996
# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ 