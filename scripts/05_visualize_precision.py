import os
import matplotlib.pyplot as plt
import numpy as np

# EGA Project: High-Impact Visualization Script
# Purpose: Compare Label Precision using Hollow + Texture contrast
# Color Scheme: Purple (#552583) and Gold (#FDB927)

k_values = [1, 3, 5, 10]
nprobes = [1, 5, 10]

latent_manifold = {
    1: [0.5560, 0.5900, 0.5940],
    3: [0.5177, 0.5485, 0.5532],
    5: [0.4984, 0.5245, 0.5292],
    10: [0.4643, 0.4853, 0.4896]
}

ega_flattened = {
    1: [0.7050, 0.7080, 0.7065],
    3: [0.6855, 0.6827, 0.6817],
    5: [0.6804, 0.6766, 0.6757],
    10: [0.6676, 0.6600, 0.6583]
}

color_ega = '#552583' 
color_latent = '#FDB927'

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 28,
    'legend.fontsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'font.family': 'serif',
    'mathtext.fontset': 'stix'
})

fig, axes = plt.subplots(1, 4, figsize=(32, 8), sharey=True)

x = np.arange(len(nprobes))
width = 0.35

for i, k in enumerate(k_values):
    ax = axes[i]
    
    # Plot baseline: Hollow (white face) with gold edges and diagonal hatching
    ax.bar(x - width/2, latent_manifold[k], width, label='Latent Manifold', 
           facecolor='white', edgecolor=color_latent, linewidth=3, hatch='//')
    
    # Plot proposed: Hollow (white face) with purple edges and cross hatching
    ax.bar(x + width/2, ega_flattened[k], width, label='EGA-flattened (Ours)', 
           facecolor='white', edgecolor=color_ega, linewidth=3, hatch='xx')

    ax.set_title(f'$K={k}$')
    ax.set_xlabel('$nprobe$')
    if i == 0:
        ax.set_ylabel('Label Precision')

    ax.set_xticks(x)
    ax.set_xticklabels(nprobes)
    ax.grid(True, axis='y', linestyle='solid', alpha=0.2)
    ax.set_ylim(0.2, 0.8)

# Add global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=2, frameon=False, prop={'size': 32})

plt.tight_layout()

# Save as high-resolution PDF
plt.savefig(os.path.expanduser('~/hpdic/EGA/paper/fig/label_precision.pdf'), dpi=300, bbox_inches='tight')