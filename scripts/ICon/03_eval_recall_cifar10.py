# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026

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
    print(f"\nEvaluating Generalization: {label_text}")
    if not os.path.exists(features_path):
        print(f"File not found: {features_path}")
        return
        
    features = np.load(features_path).astype(np.float32)
    # CIFAR10 测试集一共 10000 张，咱们依然用 8000/2000 分割
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
    k_list = [1, 3, 5, 10]
    
    run_evaluation(os.path.join(base_dir, "cifar10_original_features.npy"), "CIFAR10 Original CLIP", k_list)
    run_evaluation(os.path.join(base_dir, "cifar10_ega_features.npy"), "CIFAR10 EGA", k_list)
    run_evaluation(os.path.join(base_dir, "cifar10_icon_features.npy"), "CIFAR10 ICon", k_list)

if __name__ == "__main__":
    main()

# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ 
# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ 
# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ 
# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ python ICon/03_eval_recall_cifar10.py 

# Evaluating Generalization: CIFAR10 Original CLIP

# Recall@1:
# nprobe 1: 0.6170
# nprobe 5: 0.9555
# nprobe 10: 0.9925

# Recall@3:
# nprobe 1: 0.5887
# nprobe 5: 0.9442
# nprobe 10: 0.9873

# Recall@5:
# nprobe 1: 0.5707
# nprobe 5: 0.9399
# nprobe 10: 0.9857

# Recall@10:
# nprobe 1: 0.5363
# nprobe 5: 0.9291
# nprobe 10: 0.9819

# Evaluating Generalization: CIFAR10 EGA

# Recall@1:
# nprobe 1: 0.8580
# nprobe 5: 0.9985
# nprobe 10: 0.9995

# Recall@3:
# nprobe 1: 0.8418
# nprobe 5: 0.9978
# nprobe 10: 0.9997

# Recall@5:
# nprobe 1: 0.8332
# nprobe 5: 0.9986
# nprobe 10: 0.9997

# Recall@10:
# nprobe 1: 0.8034
# nprobe 5: 0.9981
# nprobe 10: 0.9997

# Evaluating Generalization: CIFAR10 ICon

# Recall@1:
# nprobe 1: 0.7875
# nprobe 5: 0.9910
# nprobe 10: 0.9990

# Recall@3:
# nprobe 1: 0.7665
# nprobe 5: 0.9862
# nprobe 10: 0.9973

# Recall@5:
# nprobe 1: 0.7543
# nprobe 5: 0.9839
# nprobe 10: 0.9976

# Recall@10:
# nprobe 1: 0.7349
# nprobe 5: 0.9809
# nprobe 10: 0.9966
# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ 