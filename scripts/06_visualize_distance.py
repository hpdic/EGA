import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def main():
    base_dir = os.path.expanduser("~/hpdic/EGA")
    embed_dir = os.path.join(base_dir, "embeddings")
    orig_path = os.path.join(embed_dir, "cifar100_vit_b32_features.npy")
    ega_path = os.path.join(embed_dir, "cifar100_ega_features.npy")

    fig_save_dir = os.path.join(base_dir, "paper/fig")
    os.makedirs(fig_save_dir, exist_ok=True)
    save_path = os.path.join(fig_save_dir, "distance_distribution_grid.pdf")

    if not os.path.exists(ega_path):
        print("Error: EGA features not found.")
        return

    orig_features = np.load(orig_path).astype(np.float32)
    ega_features = np.load(ega_path).astype(np.float32)

    base_orig = orig_features[:8000]
    query_orig = orig_features[8000:8500] 
    
    base_ega = ega_features[:8000]
    query_ega = ega_features[8000:8500]

    print("Calculating ground truth neighbors...")
    nn = NearestNeighbors(n_neighbors=8, metric="l2")
    nn.fit(base_orig)
    _, neighbor_indices = nn.kneighbors(query_orig)

    np.random.seed(42)
    random_indices = np.random.randint(0, 8000, size=(500, 50))

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14
    })
    
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    k_values = [1, 2, 4, 8]

    for col_idx, k in enumerate(k_values):
        print(f"Processing K={k}...")
        orig_pos, orig_neg = [], []
        ega_pos, ega_neg = [], []

        for i in range(500):
            q_orig, q_ega = query_orig[i], query_ega[i]
            pos_idx = neighbor_indices[i, :k]
            neg_idx = random_indices[i, :k * 5]

            orig_pos.extend(np.linalg.norm(base_orig[pos_idx] - q_orig, axis=1))
            orig_neg.extend(np.linalg.norm(base_orig[neg_idx] - q_orig, axis=1))

            ega_pos.extend(np.linalg.norm(base_ega[pos_idx] - q_ega, axis=1))
            ega_neg.extend(np.linalg.norm(base_ega[neg_idx] - q_ega, axis=1))

        ax = axes[0, col_idx]
        ax.hist(orig_pos, bins=30, alpha=0.7, density=True, color="royalblue", label=f"Top {k} Neighbors")
        ax.hist(orig_neg, bins=40, alpha=0.5, density=True, color="lightgray", label="Background")
        ax.set_title(f"CLIP Latent Manifold (K={k})", pad=15)
        ax.set_xlim([0, 12])
        ax.legend(loc="upper right")
        if col_idx == 0:
            ax.set_ylabel("Density", fontweight="bold")

        ax = axes[1, col_idx]
        ax.hist(ega_pos, bins=30, alpha=0.9, density=True, color="gold", edgecolor="black", linewidth=0.5, label=f"Top {k} Neighbors")
        ax.hist(ega_neg, bins=40, alpha=0.5, density=True, color="lightgray", label="Background")
        ax.set_title(f"EGA Flattened Space (K={k})", pad=15)
        ax.set_xlim([0, 2.0])
        ax.set_xlabel("L2 Distance", fontweight="bold")
        ax.legend(loc="upper right")
        if col_idx == 0:
            ax.set_ylabel("Density", fontweight="bold")

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Paper ready grid plot saved to: {save_path}")

if __name__ == "__main__":
    main()