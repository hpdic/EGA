# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 28, 2026

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    fig_save_dir = os.path.join(base_dir, 'paper/fig')
    os.makedirs(fig_save_dir, exist_ok=True)
    
    save_path = os.path.join(fig_save_dir, 'food_recall.pdf')

    k_values = [1, 3, 5, 10]
    nprobes = [1, 5, 10]
    x_indexes = np.arange(len(nprobes))
    bar_width = 0.2

    # Latest Data: ANNS Recall strictly on 21 unseen Food-101 classes (Fair Adapter)
    orig_rec = {
        1: [0.8423, 0.9977, 1.0000],
        3: [0.8370, 0.9977, 1.0000],
        5: [0.8264, 0.9974, 1.0000],
        10: [0.8212, 0.9968, 1.0000]
    }
    
    icon_rec = {
        1: [0.8378, 0.9970, 1.0000],
        3: [0.8228, 0.9964, 0.9997],
        5: [0.8126, 0.9968, 1.0000],
        10: [0.7997, 0.9959, 1.0000]
    }
    
    srl_rec = {
        1: [0.8401, 0.9985, 1.0000],
        3: [0.8264, 0.9977, 1.0000],
        5: [0.8155, 0.9970, 1.0000],
        10: [0.8068, 0.9959, 1.0000]
    }
    
    ega_rec = {
        1: [0.8972, 0.9962, 1.0000],
        3: [0.8903, 0.9970, 1.0000],
        5: [0.8844, 0.9960, 1.0000],
        10: [0.8808, 0.9968, 1.0000]
    }

    # Color Palette: Baseline (Gray), Baselines (Blue/Green), Proposed (Purple)
    color_orig = '#808080'
    color_icon = '#4B8BBE'
    color_srl = '#3CB371'
    color_ega = '#552583'

    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 28,
        'legend.fontsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.family': 'serif',
        'hatch.linewidth': 2.0
    })

    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)

    for col_idx, k in enumerate(k_values):
        ax = axes[col_idx]
        
        ax.bar(x_indexes - 1.5 * bar_width, orig_rec[k], width=bar_width, 
               facecolor='none', edgecolor=color_orig, hatch='///', 
               linewidth=2.5, label='Latent Manifold')
               
        ax.bar(x_indexes - 0.5 * bar_width, icon_rec[k], width=bar_width, 
               facecolor='none', edgecolor=color_icon, hatch='\\\\\\', 
               linewidth=2.5, label='ICon Adapter')
        
        ax.bar(x_indexes + 0.5 * bar_width, srl_rec[k], width=bar_width, 
               facecolor='none', edgecolor=color_srl, hatch='xxx', 
               linewidth=2.5, label='SRL Adapter')
               
        ax.bar(x_indexes + 1.5 * bar_width, ega_rec[k], width=bar_width, 
               facecolor='none', edgecolor=color_ega, hatch='...', 
               linewidth=2.5, label='EGA Adapter (Ours)')
        
        ax.set_title(f'$K={k}$', pad=15)
        
        ax.set_ylim([0.75, 1.02])
        ax.set_xticks(x_indexes)
        ax.set_xticklabels(nprobes)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        if col_idx == 0:
            ax.set_ylabel('ANNS Recall')
        
        ax.set_xlabel('nprobe')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'Food-101 Recall plot saved: {save_path}')

if __name__ == '__main__':
    main()
    