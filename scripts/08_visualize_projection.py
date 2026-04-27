# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026

import os
import numpy as np
import matplotlib.pyplot as plt
import umap
import warnings

warnings.filterwarnings('ignore')

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    
    orig_path = os.path.join(embed_dir, 'cifar100_vit_b32_features.npy')
    ega_path = os.path.join(embed_dir, 'cifar100_ega_features.npy')
    labels_path = os.path.join(embed_dir, 'cifar100_vit_b32_labels.npy')
    
    save_path = os.path.join(base_dir, 'paper/fig/manifold_projection.pdf')

    orig_features = np.load(orig_path).astype(np.float32)
    ega_features = np.load(ega_path).astype(np.float32)
    labels = np.load(labels_path)
    
    orig_features /= np.linalg.norm(orig_features, axis=1, keepdims=True)
    ega_features /= np.linalg.norm(ega_features, axis=1, keepdims=True)

    train_size = 8000
    base_orig = orig_features[:train_size]
    base_ega = ega_features[:train_size]
    subset_labels = labels[:train_size]

    # Use a fixed seed and print the selected classes as proof
    random_seed = 666
    np.random.seed(random_seed)
    all_unique_classes = np.unique(subset_labels)
    target_classes = np.random.choice(all_unique_classes, 10, replace=False)
    
    print(f'Randomly selected classes using seed {random_seed}:')
    print(target_classes)

    samples_per_class = 20
    class_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
    ]

    print('Projecting manifolds...')
    reducer_orig = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
    proj_orig = reducer_orig.fit_transform(base_orig)

    reducer_ega = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
    proj_ega = reducer_ega.fit_transform(base_ega)

    plt.rcParams.update({'font.size': 22, 'font.family': 'serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    def render_plot(ax, proj, title, title_color):
        ax.scatter(proj[:, 0], proj[:, 1], c='lightgray', s=5, alpha=0.08, rasterized=True)
        
        for i, cls_id in enumerate(target_classes):
            idx = np.where(subset_labels == cls_id)[0]
            if len(idx) > 0:
                sel = np.random.choice(idx, min(len(idx), samples_per_class), replace=False)
                ax.scatter(proj[sel, 0], proj[sel, 1], 
                           c=class_colors[i], s=120, edgecolors='black', 
                           linewidth=0.8, label=f'Class {cls_id}', zorder=10)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color('black')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=32, color=title_color, pad=30)

    render_plot(ax1, proj_orig, 'Latent Manifold (Original CLIP)', '#FDB927')
    render_plot(ax2, proj_ega, 'EGA-flattened (Ours)', '#552583')

    handles, lgd_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, lgd_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, fontsize=18)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f'Visualization completed. Seed used: {random_seed}')

if __name__ == '__main__':
    main()