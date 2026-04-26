import numpy as np
import faiss
import time
import os

def calculate_recall(retrieved_indices, ground_truth_indices, k=5):
    """
    Standard ANN Recall: How many of the true L2 nearest neighbors 
    did the approximate search (IVF) actually find?
    """
    count = 0
    num_queries = len(ground_truth_indices)
    for i in range(num_queries):
        # Check intersection between IVF results and Exact L2 results
        intersect = np.intersect1d(retrieved_indices[i], ground_truth_indices[i])
        count += len(intersect)
    # Average recall across all queries
    return count / (num_queries * k)

def main():
    print("Loading extracted ViT features...")
    features_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_vit_b32_features.npy")
    labels_path = os.path.expanduser("~/hpdic/EGA/embeddings/cifar10_vit_b32_labels.npy")
    
    features = np.load(features_path)
    # Ensure float32 for Faiss
    features = features.astype(np.float32)

    # Split into database (8000) and queries (2000)
    base_features = features[:8000]
    query_features = features[8000:]
    
    dim = base_features.shape[1]
    k_neighbors = 5

    print("\n=== Stage 1: Exact L2 Search (Generating Ground Truth) ===")
    exact_index = faiss.IndexFlatL2(dim)
    exact_index.add(base_features)
    
    t0 = time.time()
    # These indices are the "True Neighbors" we want to find
    _, ground_truth_indices = exact_index.search(query_features, k_neighbors)
    exact_time = time.time() - t0
    print(f"Ground Truth generated in {exact_time:.4f} seconds")

    print("\n=== Stage 2: IVF Search (Testing Geometric Navigability) ===")
    # nlist=1000 means roughly 8 points per cell. 
    # This creates very fine Voronoi boundaries to stress-test the manifold.
    nlist = 1000  
    quantizer = faiss.IndexFlatL2(dim)
    ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    
    print(f"Training IVF index with nlist={nlist}...")
    ivf_index.train(base_features)
    ivf_index.add(base_features)

    # Testing Recall@5 with different nprobe settings
    print(f"{'nprobe':<10} | {'Recall@5':<12} | {'Search Time':<12}")
    print("-" * 40)
    
    for nprobe in [1, 5, 10, 50]:
        ivf_index.nprobe = nprobe
        t0 = time.time()
        _, ivf_indices = ivf_index.search(query_features, k_neighbors)
        ivf_time = time.time() - t0
        
        # Call the new recall function using ground_truth_indices
        recall = calculate_recall(ivf_indices, ground_truth_indices, k=k_neighbors)
        print(f"{nprobe:<10} | {recall:<12.4f} | {ivf_time:<12.4f}s")

if __name__ == "__main__":
    main()

#
# Example output:
#
# (venv) cc@uc-a100:~/hpdic/EGA$ python ./scripts/03_eval_ivf.py 
# Loading extracted ViT features...

# === Stage 1: Exact L2 Search (Generating Ground Truth) ===
# Ground Truth generated in 0.2611 seconds

# === Stage 2: IVF Search (Testing Geometric Navigability) ===
# Training IVF index with nlist=1000...
# WARNING clustering 8000 points to 1000 centroids: please provide at least 39000 training points
# nprobe     | Recall@5     | Search Time 
# ----------------------------------------
# 1          | 0.3331       | 0.0180      s
# 5          | 0.7519       | 0.0175      s
# 10         | 0.8863       | 0.0192      s
# 50         | 0.9950       | 0.0184      s
# (venv) cc@uc-a100:~/hpdic/EGA$ 
