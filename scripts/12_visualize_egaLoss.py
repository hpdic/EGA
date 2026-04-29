# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    fig_save_dir = os.path.join(base_dir, 'paper/fig')
    os.makedirs(fig_save_dir, exist_ok=True)
    save_path = os.path.join(fig_save_dir, 'mechanism_sparse_gradient.pdf')

    # ── Active triplet ratio data (from training log) ──────────────────
    epochs      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    active_ratio = [0.174, 0.126, 0.116, 0.099, 0.078, 0.070, 0.073,
                    0.062, 0.052, 0.047, 0.051, 0.046, 0.041, 0.044, 0.035]

    # ── Label Precision @ K=1, nprobe=1 on unseen FGVC Aircraft classes ─
    methods         = ['CLIP\nBaseline', 'EGA\n(Ours)', 'ICon\nAdapter', 'SRL\nAdapter']
    label_precision = [0.5119, 0.5476, 0.4226, 0.4405]

    # ── Colors (same palette as existing figures) ──────────────────────
    color_ega      = '#552583'   # Purple  — EGA
    color_baseline = '#FDB927'   # Gold    — CLIP baseline
    color_icon     = '#AAAAAA'   # Gray    — ICon
    color_srl      = '#CCCCCC'   # Light gray — SRL
    bar_colors     = [color_baseline, color_ega, color_icon, color_srl]

    # ── Global style ───────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.family': 'serif',
    })

    fig = plt.figure(figsize=(20, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.35)

    # ── Left panel: Active triplet ratio decay ─────────────────────────
    ax_left = fig.add_subplot(gs[0])

    ax_left.plot(epochs, [r * 100 for r in active_ratio],
                 marker='o', linestyle='-', linewidth=3, markersize=9,
                 color=color_ega, label='Active triplet ratio')

    # Annotate start and end
    ax_left.annotate(f'{active_ratio[0]*100:.1f}%',
                     xy=(epochs[0], active_ratio[0]*100),
                     xytext=(epochs[0] + 8, active_ratio[0]*100 + 0.8),
                     fontsize=16, color=color_ega)
    ax_left.annotate(f'{active_ratio[-1]*100:.1f}%',
                     xy=(epochs[-1], active_ratio[-1]*100),
                     xytext=(epochs[-1] - 28, active_ratio[-1]*100 + 0.8),
                     fontsize=16, color=color_ega)

    # Horizontal reference line at final value
    ax_left.axhline(y=active_ratio[-1]*100, linestyle=':', linewidth=2,
                    color=color_ega, alpha=0.5)

    ax_left.set_xlabel('Training Epoch')
    ax_left.set_ylabel('Active Triplet Ratio (%)')
    ax_left.set_title('Gradient Sparsity During Training')
    ax_left.set_xlim([0, 160])
    ax_left.set_ylim([0, 22])
    ax_left.set_xticks(epochs[::2])   # every other epoch label
    ax_left.grid(True, linestyle=':', alpha=0.6)

    # Shaded region annotation
    ax_left.fill_between(epochs, [r * 100 for r in active_ratio], 0,
                         alpha=0.08, color=color_ega)
    ax_left.text(80, 12,
                 '96.5% of triplets\nproduce zero gradient\nat convergence',
                 fontsize=15, color=color_ega, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=color_ega, alpha=0.7))

    # ── Right panel: Label Precision bar chart ─────────────────────────
    ax_right = fig.add_subplot(gs[1])

    x      = np.arange(len(methods))
    bars   = ax_right.bar(x, label_precision, width=0.55,
                          color=bar_colors, edgecolor='white', linewidth=1.5)

    # Dashed baseline reference line (CLIP)
    ax_right.axhline(y=label_precision[0], linestyle='--', linewidth=2.5,
                     color=color_baseline, alpha=0.8, label='CLIP Baseline')

    # Value labels on bars
    for bar, val in zip(bars, label_precision):
        ax_right.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.005,
                      f'{val:.3f}',
                      ha='center', va='bottom', fontsize=15,
                      fontweight='bold' if val == max(label_precision) else 'normal')

    # Highlight collapse region
    ax_right.axhspan(0, label_precision[0], alpha=0.04,
                     color='red', label='Below CLIP baseline')

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(methods, fontsize=17)
    ax_right.set_ylabel('Label Precision @ K=1, nprobe=1')
    ax_right.set_title('OOD Retrieval on Unseen Classes\n(FGVC Aircraft)')
    ax_right.set_ylim([0.30, 0.65])
    ax_right.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax_right.legend(loc='upper right', frameon=False, fontsize=15)

    # ── Save ───────────────────────────────────────────────────────────
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'Saved: {save_path}')

if __name__ == '__main__':
    main()