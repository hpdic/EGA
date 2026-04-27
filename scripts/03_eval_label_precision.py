# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss
import os

def calculate_label_precision(retrieved_indices, query_labels, base_labels, k):
    correct_matches = 0
    total_queries = len(query_labels)
    
    for i in range(total_queries):
        q_label = query_labels[i]
        # Map the retrieved indices back to their physical world labels
        ret_labels = base_labels[retrieved_indices[i, :k]]
        
        # Count how many retrieved items share the exact same class as the query
        correct_matches += np.sum(ret_labels == q_label)
        
    return correct_matches / (total_queries * k)

def run_label_evaluation(features_path, labels_path, label_text, k_list, split_idx):
    print(f"\nEvaluating Semantic Precision: {label_text}")
    if not os.path.exists(features_path):
        print(f"Feature file not found: {features_path}")
        return
    if not os.path.exists(labels_path):
        print(f"Label file not found: {labels_path}")
        return
        
    # Load both features and physical labels
    features = np.load(features_path).astype(np.float32)
    labels = np.load(labels_path)
    
    # Strictly align the splits for both arrays
    base_features = features[:split_idx]
    query_features = features[split_idx:]
    base_labels = labels[:split_idx]
    query_labels = labels[split_idx:]
    
    dim = base_features.shape[1]
    nlist = 100
    
    # Initialize the standard FAISS index
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base_features)
    ivf.add(base_features)

    for k in k_list:
        print(f"\nLabel Precision@{k}:")
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            # search returns distances and indices, we only need indices
            _, ret_indices = ivf.search(query_features, k)
            
            # The ultimate test: evaluate against physical labels
            precision = calculate_label_precision(ret_indices, query_labels, base_labels, k)
            print(f"nprobe {nprobe}: {precision:.4f}")

def main():
    base_dir = os.path.expanduser("~/hpdic/EGA/embeddings")
    
    orig_features = os.path.join(base_dir, "cifar100_vit_b32_features.npy")
    ega_features = os.path.join(base_dir, "cifar100_ega_features.npy")
    icon_features = os.path.join(base_dir, "cifar100_icon_features.npy")
    srl_features = os.path.join(base_dir, "cifar100_srl_features.npy")

    # Assuming your feature extraction script saved the labels here
    labels_file = os.path.join(base_dir, "cifar100_vit_b32_labels.npy") 
    
    k_list = [1, 3, 5, 10]
    split_idx = 8000
    
    run_label_evaluation(orig_features, labels_file, "CIFAR100 Original CLIP", k_list, split_idx)
    run_label_evaluation(ega_features, labels_file, "CIFAR100 EGA", k_list, split_idx)
    run_label_evaluation(icon_features, labels_file, "CIFAR100 ICon", k_list, split_idx)
    run_label_evaluation(srl_features, labels_file, "CIFAR100 SRL", k_list, split_idx)

if __name__ == "__main__":
    main()

# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ python 03_eval_label_precision.py 

# Evaluating Semantic Precision: CIFAR100 Original CLIP

# Label Precision@1:
# nprobe 1: 0.5560
# nprobe 5: 0.5900
# nprobe 10: 0.5940

# Label Precision@3:
# nprobe 1: 0.5177
# nprobe 5: 0.5485
# nprobe 10: 0.5532

# Label Precision@5:
# nprobe 1: 0.4984
# nprobe 5: 0.5245
# nprobe 10: 0.5292

# Label Precision@10:
# nprobe 1: 0.4643
# nprobe 5: 0.4853
# nprobe 10: 0.4896

# Evaluating Semantic Precision: CIFAR100 EGA

# Label Precision@1:
# nprobe 1: 0.6955
# nprobe 5: 0.7025
# nprobe 10: 0.7045

# Label Precision@3:
# nprobe 1: 0.6835
# nprobe 5: 0.6868
# nprobe 10: 0.6855

# Label Precision@5:
# nprobe 1: 0.6734
# nprobe 5: 0.6784
# nprobe 10: 0.6784

# Label Precision@10:
# nprobe 1: 0.6587
# nprobe 5: 0.6545
# nprobe 10: 0.6526

# Evaluating Semantic Precision: CIFAR100 ICon

# Label Precision@1:
# nprobe 1: 0.8430
# nprobe 5: 0.8405
# nprobe 10: 0.8400

# Label Precision@3:
# nprobe 1: 0.8425
# nprobe 5: 0.8402
# nprobe 10: 0.8402

# Label Precision@5:
# nprobe 1: 0.8398
# nprobe 5: 0.8399
# nprobe 10: 0.8396

# Label Precision@10:
# nprobe 1: 0.8384
# nprobe 5: 0.8394
# nprobe 10: 0.8395

# Evaluating Semantic Precision: CIFAR100 SRL

# Label Precision@1:
# nprobe 1: 0.8070
# nprobe 5: 0.8065
# nprobe 10: 0.8060

# Label Precision@3:
# nprobe 1: 0.8048
# nprobe 5: 0.8067
# nprobe 10: 0.8067

# Label Precision@5:
# nprobe 1: 0.7982
# nprobe 5: 0.8027
# nprobe 10: 0.8025

# Label Precision@10:
# nprobe 1: 0.7918
# nprobe 5: 0.7980
# nprobe 10: 0.7980
# (venv) cc@uc-a100:~/hpdic/EGA/scripts$ 