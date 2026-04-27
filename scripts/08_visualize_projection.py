# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026

import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Global plotting configurations for publication quality
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def plot_recall_final():
    # Path settings
    base_dir = os.path.expanduser('~/hpdic/EGA')
    save_dir = os.path.join(base_dir, 'paper/fig')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cifar10_generalization_recall.pdf')

    # Data definition
    k_labels = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10']
    nprobes = [1, 5, 10]
    
    # Results data for each model [n=1, n=5, n=10]
    results = {
        'Original CLIP': [
            [0.6170, 0.9555, 0.9925], [0.5887, 0.9442, 0.9873], 
            [0.5707, 0.9399, 0.9857], [0.5363, 0.9291, 0.9819]
        ],
        'ICon': [
            [0.7875, 0.9910, 0.9990], [0.7665, 0.9862, 0.9973], 
            [0.7543, 0.9839, 0.9976], [0.7349, 0.9809, 0.9966]
        ],
        'SRL': [
            [0.8305, 0.9925, 0.9990], [0.7983, 0.9893, 0.9988], 
            [0.7902, 0.9873, 0.9981], [0.7719, 0.9844, 0.9975]
        ],
        'EGA': [
            [0.8580, 0.9985, 0.9995], [0.8418, 0.9978, 0.9997], 
            [0.8332, 0.9986, 0.9997], [0.8034, 0.9981, 0.9997]
        ]
    }

    # Style definitions: Purple for EGA, Gold for SRL
    colors = {'Original CLIP': '#A0A0A0', 'ICon': '#333333', 'SRL': '#FDB927', 'EGA': '#552583'}
    markers = {'Original CLIP': 'o', 'ICon': '^', 'SRL': 'D', 'EGA': 's'}

    # Using wider aspect ratio (24x6) as requested
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))

    lines = []
    labels = []

    for i, k_val in enumerate(k_labels):
        ax = axes[i]
        
        # Plot model lines
        for model in ['Original CLIP', 'ICon', 'SRL', 'EGA']:
            ln, = ax.plot(nprobes, [results[model][i][j] for j in range(3)], 
                          color=colors[model], marker=markers[model], 
                          linewidth=2.5, markersize=10, alpha=0.9)
            if i == 0:
                lines.append(ln)
                labels.append(model)

        # Style each subplot: removing italics from titles
        ax.set_title(k_val, fontsize=22, pad=15)
        ax.set_xlabel('nprobe', fontsize=18)
        ax.set_xticks(nprobes)
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Inset Zoom focusing on n=10 area
        # Positioned to avoid overlapping with data points as much as possible
        ax_ins = inset_axes(ax, width='35%', height='30%', loc='lower right', 
                            bbox_to_anchor=(-0.05, 0.15, 1, 1), bbox_transform=ax.transAxes)
        
        for model in ['Original CLIP', 'ICon', 'SRL', 'EGA']:
            ax_ins.plot(nprobes, [results[model][i][j] for j in range(3)], 
                        color=colors[model], marker=markers[model], 
                        linewidth=1.8, markersize=6)
        
        # Set zoom limits for high-precision differences at n=10
        ax_ins.set_xlim(9.6, 10.4)
        ax_ins.set_ylim(0.994, 1.0005)
        ax_ins.set_xticks([10])
        ax_ins.tick_params(labelsize=10)
        
        # Connect main plot to zoom area
        mark_inset(ax, ax_ins, loc1=2, loc2=4, fc='none', ec='0.6', linestyle='--', alpha=0.5)

        if i == 0:
            ax.set_ylabel('Recall Score', fontsize=20)

    # Place legend centered at the top
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
               ncol=4, fontsize=18, frameon=False)

    plt.tight_layout()
    # Save the professional PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'Final production PDF saved to: {save_path}')

if __name__ == '__main__':
    plot_recall_final()