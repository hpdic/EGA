# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026

import numpy as np
import faiss
import os

def calculate_anns_recall(retrieved, gt, k):
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)

def calculate_label_precision(retrieved, base_labels, query_labels, k):
    count = 0
    for i in range(len(query_labels)):
        neighbor_labels = base_labels[retrieved[i, :k]]
        count += np.sum(neighbor_labels == query_labels[i])
    return count / (len(query_labels) * k)

def run_evaluation(features_path, labels_path, label_text, k_list):
    print(f'\nEvaluating: {label_text}')
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f'File missing: {features_path} or {labels_path}')
        return
        
    features = np.load(features_path).astype(np.float32)
    labels = np.load(labels_path)
    
    # 随机打乱以确保 base 和 query 集合包含相同的类别
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    # 动态切分：取前 2500 个作为索引库，剩下的作为查询集
    split_idx = 2500
    base = features[:split_idx]
    base_labels = labels[:split_idx]
    query = features[split_idx:]
    query_labels = labels[split_idx:]
    
    dim = base.shape[1]
    
    # 建立精确索引
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    # 建立近似索引
    nlist = 50
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f'\n--- Results for K={k} ---')
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            
            recall = calculate_anns_recall(ret, gt, k)
            precision = calculate_label_precision(ret, base_labels, query_labels, k)
            
            print(f'nprobe {nprobe}: Label Precision = {precision:.4f} | ANNS Recall = {recall:.4f}')

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA/embeddings')
    
    orig_feat = os.path.join(base_dir, 'aircraft_test_vit_b32_features.npy')
    ega_feat = os.path.join(base_dir, 'aircraft_ega_features.npy')
    labels = os.path.join(base_dir, 'aircraft_test_vit_b32_labels.npy')
    
    k_list = [1, 3, 5, 10]
    
    run_evaluation(orig_feat, labels, 'Aircraft Original CLIP (Baseline)', k_list)
    run_evaluation(ega_feat, labels, 'Aircraft EGA (Transfer from Flowers)', k_list)

if __name__ == '__main__':
    main()