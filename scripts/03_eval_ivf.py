# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss
import time
import os

def calculate_recall(retrieved_indices, ground_truth_indices, k):
    count = 0
    num_queries = len(ground_truth_indices)
    for i in range(num_queries):
        # Comparison between top-k retrieved and top-k ground truth
        intersect = np.intersect1d(retrieved_indices[i, :k], ground_truth_indices[i, :k])
        count += len(intersect)
    return count / (num_queries * k)

def run_evaluation(features_path, label_text, k_list):
    print(f"\n--- Evaluating: {label_text} ---")
    if not os.path.exists(features_path):
        print(f"File not found: {features_path}")
        return
        
    features = np.load(features_path).astype(np.float32)
    
    # Split: 8000 for database, 2000 for query
    base_features = features[:8000]
    query_features = features[8000:]
    dim = base_features.shape[1]
    
    # Generate ground truth for the maximum K in the list
    max_k = max(k_list)
    exact_index = faiss.IndexFlatL2(dim)
    exact_index.add(base_features)
    _, ground_truth_indices = exact_index.search(query_features, max_k)

    # IVF setup with 1000 centroids
    nlist = 1000
    quantizer = faiss.IndexFlatL2(dim)
    ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    ivf_index.train(base_features)
    ivf_index.add(base_features)

    # Header for the result table
    header = f"{'K':<5} | {'nprobe':<8} | {'Recall':<10}"
    print(header)
    print("." * len(header))

    for k in k_list:
        for nprobe in [1, 5, 10]:
            ivf_index.nprobe = nprobe
            _, ivf_indices = ivf_index.search(query_features, k)
            recall = calculate_recall(ivf_indices, ground_truth_indices, k)
            print(f"{k:<5} | {nprobe:<8} | {recall:<10.4f}")

def main():
    # Absolute paths to embedding files
    base_path = os.path.expanduser("~/hpdic/EGA/embeddings")
    orig_path = os.path.join(base_path, "cifar10_vit_b32_features.npy")
    ega_path = os.path.join(base_path, "cifar10_ega_features.npy")
    
    # Target K values
    k_values = [1, 3, 5, 10]
    
    run_evaluation(orig_path, "Original CLIP Features", k_values)
    run_evaluation(ega_path, "EGA Corrected Features", k_values)

if __name__ == "__main__":
    main()

#
# Example Output:
#

# (venv) cc@uc-a100:~/hpdic/EGA$ 
# (venv) cc@uc-a100:~/hpdic/EGA$ python scripts/03_eval_ivf.py 

# --- Evaluating: Original CLIP Features ---
# WARNING clustering 8000 points to 1000 centroids: please provide at least 39000 training points
# K     | nprobe   | Recall    
# .............................
# 1     | 1        | 0.4310    
# 1     | 5        | 0.8210    
# 1     | 10       | 0.9280    
# 3     | 1        | 0.3650    
# 3     | 5        | 0.7757    
# 3     | 10       | 0.9028    
# 5     | 1        | 0.3312    
# 5     | 5        | 0.7463    
# 5     | 10       | 0.8867    
# 10    | 1        | 0.2806    
# 10    | 5        | 0.6920    
# 10    | 10       | 0.8482    

# --- Evaluating: EGA Corrected Features ---
# WARNING clustering 8000 points to 1000 centroids: please provide at least 39000 training points
# K     | nprobe   | Recall    
# .............................
# 1     | 1        | 0.6845    
# 1     | 5        | 0.9870    
# 1     | 10       | 0.9995    
# 3     | 1        | 0.6120    
# 3     | 5        | 0.9788    
# 3     | 10       | 0.9975    
# 5     | 1        | 0.5606    
# 5     | 5        | 0.9696    
# 5     | 10       | 0.9970    
# 10    | 1        | 0.4652    
# 10    | 5        | 0.9471    
# 10    | 10       | 0.9952    
# (venv) cc@uc-a100:~/hpdic/EGA$ 