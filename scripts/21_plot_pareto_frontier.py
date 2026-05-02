# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# Generate Pareto frontier figure for the paper.
# Three panels: (ID LP@1) vs (OOD LP@1) for each of the 3 OOD datasets.
# Each panel shows all 6 methods as scatter points; Pareto frontier is drawn
# as a line connecting the non-dominated points; EGA highlighted.
#
# Usage:
#   python scripts/21_plot_pareto.py
#
# Output: paper/fig/pareto_frontier.pdf

import os
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Real data from paper tables
# ─────────────────────────────────────────────

# Table 1: CIFAR-100 in-distribution LP@1
ID_LP = {
    'CLIP':         0.549,
    'ICon':         0.999,
    'SRL':          0.992,
    'LoRA+InfoNCE': 0.668,
    'LoRA+Triplet': 0.612,
    'EGA':          0.705,
}

# Table 2: OOD LP@1 (mean over 3 seeds)
OOD_LP = {
    'CIFAR-10': {
        'CLIP':         0.880,
        'ICon':         0.560,
        'SRL':          0.536,
        'LoRA+InfoNCE': 0.885,
        'LoRA+Triplet': 0.875,
        'EGA':          0.810,
    },
    'Aircraft': {
        'CLIP':         0.512,
        'ICon':         0.470,
        'SRL':          0.375,
        'LoRA+InfoNCE': 0.538,
        'LoRA+Triplet': 0.569,
        'EGA':          0.611,
    },
    'Food-101': {
        'CLIP':         0.881,
        'ICon':         0.562,
        'SRL':          0.552,
        'LoRA+InfoNCE': 0.883,
        'LoRA+Triplet': 0.833,
        'EGA':          0.791,
    },
}

# Visual style
METHOD_STYLE = {
    'CLIP':         {'color': '#888888', 'marker': 's', 'size': 110},
    'ICon':         {'color': '#d62728', 'marker': '^', 'size': 110},
    'SRL':          {'color': '#ff7f0e', 'marker': 'v', 'size': 110},
    'LoRA+InfoNCE': {'color': '#2ca02c', 'marker': 'D', 'size': 100},
    'LoRA+Triplet': {'color': '#17becf', 'marker': 'P', 'size': 110},
    'EGA':          {'color': '#1f77b4', 'marker': '*', 'size': 360},
}

METHOD_ORDER = ['CLIP', 'ICon', 'SRL', 'LoRA+InfoNCE', 'LoRA+Triplet', 'EGA']


def pareto_front(points):
    """Given list of (x, y, name), return non-dominated subset, sorted by x."""
    front = []
    for i, (x_i, y_i, n_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j, _) in enumerate(points):
            if i == j:
                continue
            if x_j >= x_i and y_j >= y_i and (x_j > x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            front.append((x_i, y_i, n_i))
    front.sort(key=lambda t: t[0])
    return front


def plot_panel(ax, dataset_name):
    points = []
    for m in METHOD_ORDER:
        x = ID_LP[m]
        y = OOD_LP[dataset_name][m]
        points.append((x, y, m))

    front = pareto_front(points)
    if len(front) >= 2:
        xs = [p[0] for p in front]
        ys = [p[1] for p in front]
        ax.plot(xs, ys, '--', color='#999999', linewidth=1.4,
                zorder=1, label='Pareto frontier')

    for x, y, m in points:
        style = METHOD_STYLE[m]
        edge = 'black' if m == 'EGA' else 'none'
        lw   = 1.2 if m == 'EGA' else 0
        ax.scatter(x, y,
                   c=style['color'], marker=style['marker'],
                   s=style['size'], edgecolors=edge, linewidths=lw,
                   zorder=3, label=m)
        offset_x, offset_y = 0.012, 0.012
        if m == 'CLIP':
            offset_x, offset_y = -0.012, -0.025
            ax.text(x + offset_x, y + offset_y, m,
                    fontsize=9, ha='right', va='top', zorder=4)
        elif m == 'EGA':
            ax.text(x + offset_x, y + offset_y, m,
                    fontsize=10, fontweight='bold', ha='left', va='bottom',
                    color=style['color'], zorder=4)
        else:
            ax.text(x + offset_x, y + offset_y, m,
                    fontsize=9, ha='left', va='bottom', zorder=4)

    ax.set_xlabel('ID LP@1 (CIFAR-100)', fontsize=11)
    ax.set_ylabel(f'OOD LP@1 ({dataset_name})', fontsize=11)
    ax.set_title(dataset_name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(0.48, 1.06)
    ys_all = [OOD_LP[dataset_name][m] for m in METHOD_ORDER]
    ymin = min(ys_all) - 0.05
    ymax = max(ys_all) + 0.06
    ax.set_ylim(ymin, ymax)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, dataset in zip(axes, ['CIFAR-10', 'Aircraft', 'Food-101']):
        plot_panel(ax, dataset)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    unique = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h, l)); seen.add(l)
    fig.legend([u[0] for u in unique], [u[1] for u in unique],
               loc='lower center', ncol=len(unique),
               bbox_to_anchor=(0.5, -0.02),
               fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = os.path.expanduser('~/hpdic/EGA/paper/fig')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pareto_frontier.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()