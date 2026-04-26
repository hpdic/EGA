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
    save_path = os.path.join(fig_save_dir, 'recall_tradeoff_grid.pdf')

    k_values = [1, 3, 5, 10]
    nprobes = [1, 5, 10]

    cifar10_orig = {
        1: [0.4310, 0.8210, 0.9280],
        3: [0.3650, 0.7757, 0.9028],
        5: [0.3312, 0.7463, 0.8867],
        10: [0.2806, 0.6920, 0.8482]
    }
    cifar10_ega = {
        1: [0.6845, 0.9870, 0.9995],
        3: [0.6120, 0.9788, 0.9975],
        5: [0.5606, 0.9696, 0.9970],
        10: [0.4652, 0.9471, 0.9952]
    }
    
    cifar100_orig = {
        1: [0.5130, 0.8455, 0.9265],
        3: [0.4432, 0.8133, 0.9088],
        5: [0.4046, 0.7825, 0.8886],
        10: [0.3399, 0.7319, 0.8557]
    }
    cifar100_ega = {
        1: [0.6865, 0.9865, 0.9995],
        3: [0.6218, 0.9788, 0.9977],
        5: [0.5728, 0.9710, 0.9963],
        10: [0.4780, 0.9484, 0.9930]
    }

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    for col_idx, k in enumerate(k_values):
        ax1 = axes[0, col_idx]
        ax1.plot(nprobes, cifar10_orig[k], marker='o', linestyle='dotted', linewidth=3, markersize=8, color='royalblue', label='Original')
        ax1.plot(nprobes, cifar10_ega[k], marker='s', linestyle='solid', linewidth=3, markersize=8, color='darkorange', label='EGA Space')
        ax1.set_title(f'CIFAR10 (K={k})', pad=15)
        ax1.set_ylim([0.2, 1.05])
        ax1.set_xticks(nprobes)
        ax1.grid(True, linestyle='dotted', alpha=0.6)
        if col_idx == 0:
            ax1.set_ylabel('Recall')
        if col_idx == 3:
            ax1.legend(loc='lower right')

        ax2 = axes[1, col_idx]
        ax2.plot(nprobes, cifar100_orig[k], marker='o', linestyle='dotted', linewidth=3, markersize=8, color='royalblue', label='Original')
        ax2.plot(nprobes, cifar100_ega[k], marker='s', linestyle='solid', linewidth=3, markersize=8, color='darkorange', label='EGA Space')
        ax2.set_title(f'CIFAR100 (K={k})', pad=15)
        ax2.set_ylim([0.2, 1.05])
        ax2.set_xticks(nprobes)
        ax2.grid(True, linestyle='dotted', alpha=0.6)
        ax2.set_xlabel('nprobe')
        if col_idx == 0:
            ax2.set_ylabel('Recall')
        if col_idx == 3:
            ax2.legend(loc='lower right')

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'Recall tradeoff grid plot saved to: {save_path}')

if __name__ == '__main__':
    main()