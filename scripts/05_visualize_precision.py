import os

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# EGA Project: High-Impact Visualization Script
# Purpose: Compare Label Precision between Latent Manifold and EGA-flattened
# Color Scheme: Purple (#552583) and Gold (#FDB927)
# --------------------------------------------------------------------------

# Define evaluation metrics: K-neighbors and nprobe (search depth)
k_values = [1, 3, 5, 10]
nprobes = [1, 5, 10]

# Experimental data for baseline (Original CLIP space)
# Represented as 'Latent Manifold' to emphasize the geometric distortion
latent_manifold = {
    1: [0.5560, 0.5900, 0.5940],
    3: [0.5177, 0.5485, 0.5532],
    5: [0.4984, 0.5245, 0.5292],
    10: [0.4643, 0.4853, 0.4896]
}

# Experimental data for proposed method (EGA-flattened)
# Demonstrates superior semantic condensation and indexing friendliness
ega_flattened = {
    1: [0.7050, 0.7080, 0.7065],
    3: [0.6855, 0.6827, 0.6817],
    5: [0.6804, 0.6766, 0.6757],
    10: [0.6676, 0.6600, 0.6583]
}

# Professional Purple-Gold Color Palette
color_ega = '#552583' # Classic Deep Purple
color_latent = '#FDB927'    # Academic Gold Accent

# Configure global RC parameters for large-scale font rendering in papers
# Optimized for CVPR/ICCV/NeurIPS double-column LaTeX templates
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

# Initialize 1x4 subplot figure with shared Y-axis for direct comparison
fig, axes = plt.subplots(1, 4, figsize=(32, 8), sharey=True)

for i, k in enumerate(k_values):
    ax = axes[i]
    
    # Plot baseline: Latent Manifold (Dashed line for original state)
    ax.plot(nprobes, latent_manifold[k], marker='o', linestyle='--', linewidth=4,
            markersize=14, label='Latent Manifold', color=color_latent, alpha=0.8)
    
    # Plot proposed: EGA-flattened (Solid line for optimized state)
    ax.plot(nprobes, ega_flattened[k], marker='D', linestyle='-', linewidth=4,
            markersize=14, label='EGA-flattened (Ours)', color=color_ega)

    # Labeling subplots using LaTeX for mathematical symbols
    ax.set_title(f'$K={k}$')
    ax.set_xlabel('$nprobe$')
    if i == 0:
        ax.set_ylabel('Label Precision')

    # Configure grid and axis visibility for better readability
    ax.set_xticks(nprobes)
    ax.grid(True, linestyle='-', alpha=0.2)
    ax.set_ylim(0.2, 0.8)

# Add a prominent global legend at the top center
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=2, frameon=False, prop={'size': 32, 'weight': 'bold'})

plt.tight_layout()

# Save as high-resolution PDF/PNG for direct inclusion in submission
plt.savefig(os.path.expanduser('~/hpdic/EGA/paper/fig/label_precision.pdf'), dpi=300, bbox_inches='tight')