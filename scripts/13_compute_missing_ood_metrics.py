# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
#
# Plot OOD Label Precision@1 across three benchmarks and three nprobe values.
# Auto-fills missing CIFAR-10 numbers by re-running eval on cached features.

import os
import numpy as np
import faiss
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Eval helpers (same protocol as your training script)
# ─────────────────────────────────────────────

def calculate_label_precision(retrieved, base_labels, query_labels, k=1):
    count = 0
    for i in range(len(query_labels)):
        neighbor_labels = base_labels[retrieved[i, :k]]
        count += np.sum(neighbor_labels == query_labels[i])
    return count / (len(query_labels) * k)


def eval_lp_across_nprobe(features, labels, nprobe_list=(1, 5, 10),
                          k=1, nlist=10, seed=42):
    """75/25 split, returns LP@k for each nprobe."""
    features = features.astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    np.random.seed(seed)
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels   = labels[idx]

    split        = int(len(features) * 0.75)
    base         = features[:split]
    base_labels  = labels[:split]
    query        = features[split:]
    query_labels = labels[split:]
    dim          = base.shape[1]

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    ivf.train(base)
    ivf.add(base)

    out = {}
    for np_ in nprobe_list:
        ivf.nprobe = np_
        _, ret = ivf.search(query, k)
        out[np_] = calculate_label_precision(ret, base_labels, query_labels, k)
    return out


# ─────────────────────────────────────────────
# Load CIFAR-10 features and compute LP across nprobe
# ─────────────────────────────────────────────

def get_cifar10_lp(embed_dir):
    print('Computing CIFAR-10 LP@1 across nprobe ...')
    labels = np.load(os.path.join(embed_dir, 'cifar10_vit_b32_labels.npy'))

    methods = {
        'CLIP': 'cifar10_vit_b32_features.npy',
        'EGA':  'cifar10_ega_features.npy',
        'ICon': 'cifar10_icon_features.npy',
        'SRL':  'cifar10_srl_features.npy',
    }

    out = {}
    for name, fname in methods.items():
        feats = np.load(os.path.join(embed_dir, fname))
        out[name] = eval_lp_across_nprobe(feats, labels)
        print(f'  {name:5s} : '
              f'nprobe=1: {out[name][1]:.4f}  '
              f'nprobe=5: {out[name][5]:.4f}  '
              f'nprobe=10: {out[name][10]:.4f}')
    return out


# ─────────────────────────────────────────────
# Hard-coded numbers from your training-time eval logs
# ─────────────────────────────────────────────

AIRCRAFT_LP = {
    'CLIP': {1: 0.5119, 5: 0.5238, 10: 0.5238},
    'EGA':  {1: 0.5476, 5: 0.5536, 10: 0.5536},
    'ICon': {1: 0.4226, 5: 0.4226, 10: 0.4226},
    'SRL':  {1: 0.4405, 5: 0.4940, 10: 0.4940},
}

FOOD_LP = {
    'CLIP': {1: 0.8812, 5: 0.8888, 10: 0.8880},
    'EGA':  {1: 0.7982, 5: 0.8165, 10: 0.8165},
    'ICon': {1: 0.5567, 5: 0.5720, 10: 0.5712},
    'SRL':  {1: 0.5202, 5: 0.5255, 10: 0.5255},
}


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

def plot_panel(ax, data, title, baseline_method='CLIP', nprobes=(1, 5, 10)):
    """Plot one panel (one dataset). Lines: CLIP/ICon/SRL/EGA across nprobe."""
    styles = {
        'CLIP': dict(color='#FDB927', marker='o', linestyle='--',
                     linewidth=2.8, markersize=10, label='CLIP (frozen)'),
        'ICon': dict(color='#888888', marker='^', linestyle='-',
                     linewidth=2.8, markersize=10, label='ICon'),
        'SRL':  dict(color='#444444', marker='v', linestyle='-',
                     linewidth=2.8, markersize=10, label='SRL'),
        'EGA':  dict(color='#552583', marker='s', linestyle='-',
                     linewidth=3.0, markersize=11, label='EGA (Ours)'),
    }

    for method in ['CLIP', 'ICon', 'SRL', 'EGA']:
        ys = [data[method][np_] for np_ in nprobes]
        ax.plot(list(nprobes), ys, **styles[method])

    # Shaded "below baseline" region
    clip_baseline = data[baseline_method][1]
    ax.axhspan(0, clip_baseline, color='red', alpha=0.04)

    ax.set_title(title, fontsize=22, pad=10)
    ax.set_xlabel('nprobe', fontsize=20)
    ax.set_xticks(list(nprobes))
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, linestyle=':', alpha=0.5)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    base_dir  = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    fig_dir   = os.path.join(base_dir, 'paper/fig')
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, 'ood_label_precision.pdf')

    # ── Compute CIFAR-10 LP (we don't have it cached) ────────────────
    cifar10_lp = get_cifar10_lp(embed_dir)

    # ── Plot 1×3 ─────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.0), sharey=False)

    plot_panel(axes[0], cifar10_lp,    'CIFAR-10')
    plot_panel(axes[1], AIRCRAFT_LP,   'FGVC-Aircraft')
    plot_panel(axes[2], FOOD_LP,       'Food-101')

    axes[0].set_ylabel('Label Precision@1', fontsize=20)

    # Per-panel y-range (datasets have very different scales)
    axes[0].set_ylim([0.75, 0.92])
    axes[1].set_ylim([0.40, 0.58])
    axes[2].set_ylim([0.45, 0.92])

    # One legend across the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=4, fontsize=18, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'\nSaved: {save_path}')


if __name__ == '__main__':
    main()

# (venv) (base) cc@uc-a100:~/hpdic/EGA$ python scripts/Food/train_eval_food101.py 
# Loading cached Food-101 features from /home/cc/hpdic/EGA/embeddings/food101_features.npy

# Total samples : 25250
# Total classes : 101

# Splitting by class (80% train / 20% unseen test) ...
#   Train classes : 80, Train samples : 20000
#   Unseen classes: 21, Unseen samples: 5250

# [1/3] Training EGA ...
#   EGA epoch 30/150
#   EGA epoch 60/150
#   EGA epoch 90/150
#   EGA epoch 120/150
#   EGA epoch 150/150

# [2/3] Training ICon ...
#   ICon epoch 30/150
#   ICon epoch 60/150
#   ICon epoch 90/150
#   ICon epoch 120/150
#   ICon epoch 150/150

# [3/3] Training SRL ...
#   SRL epoch 30/150
#   SRL epoch 60/150
#   SRL epoch 90/150
#   SRL epoch 120/150
#   SRL epoch 150/150

# Saved adapter outputs and unseen-class labels to embeddings/

# Evaluating: Original CLIP (Food-101 unseen)
#   K=1
#     nprobe= 1: LP=0.8812  AR=0.8423
#     nprobe= 5: LP=0.8888  AR=0.9977
#     nprobe=10: LP=0.8880  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.8705  AR=0.8370
#     nprobe= 5: LP=0.8789  AR=0.9977
#     nprobe=10: LP=0.8789  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.8583  AR=0.8264
#     nprobe= 5: LP=0.8673  AR=0.9974
#     nprobe=10: LP=0.8673  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.8351  AR=0.8212
#     nprobe= 5: LP=0.8518  AR=0.9968
#     nprobe=10: LP=0.8519  AR=1.0000

# Evaluating: EGA (Food-101 unseen)
#   K=1
#     nprobe= 1: LP=0.7982  AR=0.8858
#     nprobe= 5: LP=0.8165  AR=0.9970
#     nprobe=10: LP=0.8165  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.7746  AR=0.8771
#     nprobe= 5: LP=0.7865  AR=0.9959
#     nprobe=10: LP=0.7857  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.7613  AR=0.8725
#     nprobe= 5: LP=0.7736  AR=0.9966
#     nprobe=10: LP=0.7736  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.7437  AR=0.8653
#     nprobe= 5: LP=0.7575  AR=0.9966
#     nprobe=10: LP=0.7575  AR=1.0000

# Evaluating: ICon (Food-101 unseen)
#   K=1
#     nprobe= 1: LP=0.5567  AR=0.8111
#     nprobe= 5: LP=0.5720  AR=0.9962
#     nprobe=10: LP=0.5712  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.5395  AR=0.7987
#     nprobe= 5: LP=0.5570  AR=0.9954
#     nprobe=10: LP=0.5560  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.5216  AR=0.7893
#     nprobe= 5: LP=0.5354  AR=0.9950
#     nprobe=10: LP=0.5363  AR=0.9998
#   K=10
#     nprobe= 1: LP=0.5050  AR=0.7740
#     nprobe= 5: LP=0.5208  AR=0.9931
#     nprobe=10: LP=0.5211  AR=1.0000

# Evaluating: SRL (Food-101 unseen)
#   K=1
#     nprobe= 1: LP=0.5202  AR=0.8271
#     nprobe= 5: LP=0.5255  AR=0.9962
#     nprobe=10: LP=0.5255  AR=1.0000
#   K=3
#     nprobe= 1: LP=0.5161  AR=0.8071
#     nprobe= 5: LP=0.5179  AR=0.9959
#     nprobe=10: LP=0.5182  AR=1.0000
#   K=5
#     nprobe= 1: LP=0.5046  AR=0.7998
#     nprobe= 5: LP=0.5115  AR=0.9944
#     nprobe=10: LP=0.5110  AR=1.0000
#   K=10
#     nprobe= 1: LP=0.4940  AR=0.7846
#     nprobe= 5: LP=0.5045  AR=0.9926
#     nprobe=10: LP=0.5035  AR=1.0000

# Done.
# (venv) (base) cc@uc-a100:~/hpdic/EGA$ 