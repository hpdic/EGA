# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    fig_save_dir = os.path.join(base_dir, 'paper/fig')
    os.makedirs(fig_save_dir, exist_ok=True)
    # Keeping the original filename as requested
    save_path = os.path.join(fig_save_dir, 'recall_tradeoff_grid.pdf')

    k_values = [1, 3, 5, 10]
    nprobes = [1, 5, 10]

    # CIFAR100 Original data (Latent Manifold)
    cifar100_orig = {
        1: [0.6665, 0.9370, 0.9800],
        3: [0.6363, 0.9292, 0.9765],
        5: [0.6198, 0.9203, 0.9736],
        10: [0.5930, 0.9060, 0.9684]
    }
    
    # CIFAR100 EGA data (EGA-flattened)
    cifar100_ega = {
        1: [0.8245, 0.9900, 0.9975],
        3: [0.8140, 0.9842, 0.9962],
        5: [0.8052, 0.9827, 0.9964],
        10: [0.7870, 0.9771, 0.9948]
    }

    # Style: Purple and Gold
    color_latent = '#FDB927' # Gold
    color_ega = '#552583'    # Purple

    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 28,
        'legend.fontsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.family': 'serif'
    })

    # Clean 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)

    for col_idx, k in enumerate(k_values):
        ax = axes[col_idx]
        
        # Plot Baseline
        ax.plot(nprobes, cifar100_orig[k], marker='o', linestyle='--', 
                linewidth=4, markersize=12, color=color_latent, label='Latent Manifold')
        
        # Plot EGA-flattened
        ax.plot(nprobes, cifar100_ega[k], marker='s', linestyle='-', 
                linewidth=4, markersize=12, color=color_ega, label='EGA-flattened')
        
        # Simplified title focusing only on the K parameter
        ax.set_title(f'$K={k}$', pad=15)
        ax.set_ylim([0.4, 1.05])
        ax.set_xticks(nprobes)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if col_idx == 0:
            ax.set_ylabel('Recall')
        
        ax.set_xlabel('nprobe')
        
        # Move legend to bottom-right as requested
        ax.legend(loc='lower right', frameon=False)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'Recall grid plot updated (clean titles, bottom-right legend): {save_path}')

if __name__ == '__main__':
    main()