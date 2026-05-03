# scripts/moreBackbones/04_eval_anns_recall_backbone.py
import numpy as np
import faiss
import os
import argparse

def calculate_recall(retrieved, gt, k):
    count = 0
    for i in range(len(gt)):
        intersect = np.intersect1d(retrieved[i, :k], gt[i, :k])
        count += len(intersect)
    return count / (len(gt) * k)

def run_evaluation(features_path, label_text, k_list, split_idx=8000):
    print(f"\n=== ANNS Recall: {label_text} ===")
    if not os.path.exists(features_path):
        print(f"❌ File not found: {features_path}")
        return

    features = np.load(features_path).astype(np.float32)
    base = features[:split_idx]
    query = features[split_idx:]
    dim = base.shape[1]

    # Ground truth
    exact = faiss.IndexFlatL2(dim)
    exact.add(base)
    _, gt = exact.search(query, max(k_list))

    # IVF index
    nlist = 100
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    for k in k_list:
        print(f"\nRecall@{k}:")
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret = ivf.search(query, k)
            recall = calculate_recall(ret, gt, k)
            print(f"  nprobe={nprobe}: {recall:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2-large",
                        choices=["clip", "dinov2-base", "dinov2-large", "siglip"])
    args = parser.parse_args()

    base_dir = os.path.expanduser("~/hpdic/EGA")
    embed_dir = os.path.join(base_dir, "embeddings")

    suffix = args.backbone.replace("-", "_")
    raw_path = os.path.join(embed_dir, f"cifar100_{suffix}_features.npy")
    ega_path = os.path.join(embed_dir, f"cifar100_{suffix}_ega_features.npy")

    k_list = [1, 3, 5, 10]

    run_evaluation(raw_path, f"{args.backbone.upper()} (Raw)", k_list)
    run_evaluation(ega_path, f"{args.backbone.upper()} + EGA", k_list)

if __name__ == "__main__":
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts/moreBackbones$ 
# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts/moreBackbones$ 
# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts/moreBackbones$ python 04_eval_anns_recall_backbone.py --backbone dinov2-large

# === ANNS Recall: DINOV2-LARGE (Raw) ===

# Recall@1:
#   nprobe=1: 0.8785
#   nprobe=5: 0.9800
#   nprobe=10: 0.9930

# Recall@3:
#   nprobe=1: 0.8637
#   nprobe=5: 0.9715
#   nprobe=10: 0.9882

# Recall@5:
#   nprobe=1: 0.8556
#   nprobe=5: 0.9680
#   nprobe=10: 0.9849

# Recall@10:
#   nprobe=1: 0.8391
#   nprobe=5: 0.9612
#   nprobe=10: 0.9805

# === ANNS Recall: DINOV2-LARGE + EGA ===

# Recall@1:
#   nprobe=1: 0.9310
#   nprobe=5: 0.9985
#   nprobe=10: 1.0000

# Recall@3:
#   nprobe=1: 0.9308
#   nprobe=5: 0.9963
#   nprobe=10: 0.9993

# Recall@5:
#   nprobe=1: 0.9287
#   nprobe=5: 0.9960
#   nprobe=10: 0.9993

# Recall@10:
#   nprobe=1: 0.9262
#   nprobe=5: 0.9957
#   nprobe=10: 0.9991
# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts/moreBackbones$ 


# python 04_eval_anns_recall_backbone.py --backbone siglip

# === ANNS Recall: SIGLIP (Raw) ===

# Recall@1:
#   nprobe=1: 0.8015
#   nprobe=5: 0.9765
#   nprobe=10: 0.9945

# Recall@3:
#   nprobe=1: 0.7917
#   nprobe=5: 0.9712
#   nprobe=10: 0.9918

# Recall@5:
#   nprobe=1: 0.7776
#   nprobe=5: 0.9689
#   nprobe=10: 0.9915

# Recall@10:
#   nprobe=1: 0.7530
#   nprobe=5: 0.9621
#   nprobe=10: 0.9896

# === ANNS Recall: SIGLIP + EGA ===

# Recall@1:
#   nprobe=1: 0.9155
#   nprobe=5: 0.9975
#   nprobe=10: 0.9995

# Recall@3:
#   nprobe=1: 0.9202
#   nprobe=5: 0.9962
#   nprobe=10: 0.9992

# Recall@5:
#   nprobe=1: 0.9175
#   nprobe=5: 0.9954
#   nprobe=10: 0.9988

# Recall@10:
#   nprobe=1: 0.9106
#   nprobe=5: 0.9950
#   nprobe=10: 0.9989
# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts/moreBackbones$ 