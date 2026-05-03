# scripts/moreBackbones/03_eval_label_precision_backbone.py
import numpy as np
import faiss
import os
import argparse

def calculate_label_precision(retrieved_indices, query_labels, base_labels, k):
    correct_matches = 0
    total_queries = len(query_labels)
    for i in range(total_queries):
        q_label = query_labels[i]
        ret_labels = base_labels[retrieved_indices[i, :k]]
        correct_matches += np.sum(ret_labels == q_label)
    return correct_matches / (total_queries * k)

def run_evaluation(features_path, labels_path, label_text, k_list, split_idx=8000):
    print(f"\n=== Semantic Precision: {label_text} ===")
    if not os.path.exists(features_path):
        print(f"❌ Feature file not found: {features_path}")
        return

    features = np.load(features_path).astype(np.float32)
    labels = np.load(labels_path)

    base_features = features[:split_idx]
    query_features = features[split_idx:]
    base_labels = labels[:split_idx]
    query_labels = labels[split_idx:]

    dim = base_features.shape[1]
    nlist = 100
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base_features)
    ivf.add(base_features)

    for k in k_list:
        print(f"\nLabel Precision@{k}:")
        for nprobe in [1, 5, 10]:
            ivf.nprobe = nprobe
            _, ret_indices = ivf.search(query_features, k)
            precision = calculate_label_precision(ret_indices, query_labels, base_labels, k)
            print(f"  nprobe={nprobe}: {precision:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2-large",
                        choices=["clip", "dinov2-base", "dinov2-large", "siglip"])
    args = parser.parse_args()

    base_dir = os.path.expanduser("~/hpdic/EGA")
    embed_dir = os.path.join(base_dir, "embeddings")
    labels_path = os.path.join(embed_dir, "cifar100_labels.npy")

    suffix = args.backbone.replace("-", "_")
    raw_path = os.path.join(embed_dir, f"cifar100_{suffix}_features.npy")
    ega_path = os.path.join(embed_dir, f"cifar100_{suffix}_ega_features.npy")

    k_list = [1, 3, 5, 10]

    run_evaluation(raw_path, labels_path, f"{args.backbone.upper()} (Raw)", k_list)
    run_evaluation(ega_path, labels_path, f"{args.backbone.upper()} + EGA", k_list)

if __name__ == "__main__":
    main()


# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts/moreBackbones$ python 03_eval_label_precision_backbone.py --backbone dinov2-large

# === Semantic Precision: DINOV2-LARGE (Raw) ===

# Label Precision@1:
#   nprobe=1: 0.8530
#   nprobe=5: 0.8715
#   nprobe=10: 0.8740

# Label Precision@3:
#   nprobe=1: 0.8370
#   nprobe=5: 0.8503
#   nprobe=10: 0.8530

# Label Precision@5:
#   nprobe=1: 0.8310
#   nprobe=5: 0.8411
#   nprobe=10: 0.8434

# Label Precision@10:
#   nprobe=1: 0.8160
#   nprobe=5: 0.8208
#   nprobe=10: 0.8230

# === Semantic Precision: DINOV2-LARGE + EGA ===

# Label Precision@1:
#   nprobe=1: 0.9005
#   nprobe=5: 0.9040
#   nprobe=10: 0.9050

# Label Precision@3:
#   nprobe=1: 0.8963
#   nprobe=5: 0.8997
#   nprobe=10: 0.9002

# Label Precision@5:
#   nprobe=1: 0.8918
#   nprobe=5: 0.8945
#   nprobe=10: 0.8948

# Label Precision@10:
#   nprobe=1: 0.8883
#   nprobe=5: 0.8870
#   nprobe=10: 0.8868
# (venv) (base) cc@uc-a100:~/hpdic/EGA/scripts/moreBackbones$     


# python 03_eval_label_precision_backbone.py --backbone siglip

# === Semantic Precision: SIGLIP (Raw) ===

# Label Precision@1:
#   nprobe=1: 0.7740
#   nprobe=5: 0.7955
#   nprobe=10: 0.7970

# Label Precision@3:
#   nprobe=1: 0.7540
#   nprobe=5: 0.7760
#   nprobe=10: 0.7762

# Label Precision@5:
#   nprobe=1: 0.7435
#   nprobe=5: 0.7661
#   nprobe=10: 0.7652

# Label Precision@10:
#   nprobe=1: 0.7177
#   nprobe=5: 0.7383
#   nprobe=10: 0.7377

# === Semantic Precision: SIGLIP + EGA ===

# Label Precision@1:
#   nprobe=1: 0.8595
#   nprobe=5: 0.8640
#   nprobe=10: 0.8635

# Label Precision@3:
#   nprobe=1: 0.8590
#   nprobe=5: 0.8577
#   nprobe=10: 0.8568

# Label Precision@5:
#   nprobe=1: 0.8562
#   nprobe=5: 0.8547
#   nprobe=10: 0.8542

# Label Precision@10:
#   nprobe=1: 0.8537
#   nprobe=5: 0.8498
#   nprobe=10: 0.8494