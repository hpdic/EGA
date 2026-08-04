# RS4z emergency rebuttal, Experiment 2: small target-validation selection simulation.
# Candidate library (per scope decision): Frozen + EGA at margins {0.1, 0.2(default), 0.4}.
# InfoNCE / InfoNCE+preserve excluded: no live checkpoints exist in this environment and the
# plan forbids training new models "for this experiment" (see checks.md).
#
# Protocol (fixed BEFORE reading any test label):
#   1. For a given dataset, take the canonical unseen-class OOD pool (same pool used for the
#      dataset's headline LP@1/AR@1 number in Experiment 1).
#   2. Split it once via the canonical 75/25 gallery/query split (eval_seed=42, same as
#      utils_ega.eval_method / scripts/58's eval_lp_ar) -> FIXED gallery, FIXED query pool.
#      This split is identical across all candidates (it depends only on the label array,
#      which is shared; only the embedded feature values differ per candidate).
#   3. For each budget N in {5,10,20} and each of 10 independent stratified draws:
#        - sample up to N examples per class from the QUERY POOL as the "validation subset"
#          (labels of this small subset are the only labels ever used for selection);
#        - the remaining query-pool examples (never touched for selection) form that draw's
#          held-out TEST query set.
#        - For each candidate: compute validation LP@1 (query=validation subset, gallery=fixed
#          gallery) and test LP@1 (query=remaining test set, gallery=fixed gallery).
#        - Select argmax validation LP@1; regret = oracle_test_LP1 - selected_test_LP1.
#
# Usage: python scripts/59_rs4z_selection_sim.py

import os
import json
import collections
import numpy as np
import torch
import faiss

from models.ega_mlp import EGAMLP

def split_by_class(features, labels, train_ratio=0.8, split_seed=42):
    unique_classes = np.unique(labels)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes = unique_classes[num_train:]
    train_mask = np.isin(labels, train_classes)
    test_mask = np.isin(labels, test_classes)
    return (features[train_mask], labels[train_mask]), (features[test_mask], labels[test_mask])

def canonical_gallery_query_split(labels, eval_seed=42, gallery_ratio=0.75):
    np.random.seed(eval_seed)
    idx = np.random.permutation(len(labels))
    split = int(len(labels) * gallery_ratio)
    return idx[:split], idx[split:]   # gallery_idx, query_idx

def lp_at_1(gallery_feats, gallery_labels, query_feats, query_labels, nlist=10, nprobe=1):
    if len(query_labels) == 0:
        return None
    gallery_feats = gallery_feats.astype(np.float32)
    query_feats = query_feats.astype(np.float32)
    dim = gallery_feats.shape[1]
    nlist_eff = max(1, min(nlist, len(gallery_feats) // 4 if len(gallery_feats) >= 4 else 1))
    quantizer = faiss.IndexFlatL2(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist_eff, faiss.METRIC_L2)
    ivf.train(gallery_feats)
    ivf.add(gallery_feats)
    ivf.nprobe = nprobe
    _, ret = ivf.search(query_feats, 1)
    correct = np.sum(gallery_labels[ret[:, 0]] == query_labels)
    return correct / len(query_labels)

def stratified_draw(labels_in_pool, n_per_class, rng):
    """labels_in_pool: labels of the query pool (local indices 0..len-1). Returns (val_local_idx, test_local_idx)."""
    by_class = collections.defaultdict(list)
    for i, l in enumerate(labels_in_pool):
        by_class[l].append(i)
    val_idx = []
    for c, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        take = min(n_per_class, max(0, len(idxs) - 1))  # leave >=1 for test if possible
        val_idx.extend(idxs[:take].tolist())
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(set(range(len(labels_in_pool))) - set(val_idx.tolist())))
    return val_idx, test_idx

def load_ega(ckpt_path, dim, device):
    model = EGAMLP(input_dim=dim, hidden_dim=2048).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def get_candidate_embeddings(dataset_key, unseen_feats_norm, device, base_dir):
    """Returns dict: candidate_name -> transformed (unseen_feats.shape) numpy array."""
    dim = unseen_feats_norm.shape[1]
    ckpt_dir = os.path.join(base_dir, 'models', 'rs4z_missing_margins')
    ckpts = {
        'EGA 0.5x (m=0.1)': os.path.join(ckpt_dir, f'ega_{dataset_key}_m0.1_seed42.pth'),
        'EGA 1.0x (m=0.2)': os.path.join(ckpt_dir, f'ega_{dataset_key}_m0.2_seed42.pth'),
        'EGA 2.0x (m=0.4)': os.path.join(ckpt_dir, f'ega_{dataset_key}_m0.4_seed42.pth'),
    }
    out = {'Frozen': unseen_feats_norm}
    for name, path in ckpts.items():
        if not os.path.exists(path):
            print(f'  [WARN] missing checkpoint {path}, skipping candidate {name}')
            continue
        model = load_ega(path, dim, device)
        with torch.no_grad():
            t = model(torch.from_numpy(unseen_feats_norm).float().to(device)).cpu().numpy()
        out[name] = t
    return out

def load_dataset_pool(dataset, base_dir):
    embed_dir = os.path.join(base_dir, 'embeddings')
    if dataset in ('Aircraft', 'Food-101'):
        fname = 'aircraft_test_vit_b32' if dataset == 'Aircraft' else 'food101'
        features = np.load(os.path.join(embed_dir, f'{fname}_features.npy')).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        labels = np.load(os.path.join(embed_dir, f'{fname}_labels.npy'))
        _, (unseen_feats, unseen_labels) = split_by_class(features, labels, train_ratio=0.8, split_seed=42)
    else:  # CIFAR-10: entire eval set is the OOD/unseen pool (trained on CIFAR-100)
        unseen_feats = np.load(os.path.join(embed_dir, 'cifar10_vit_b32_features.npy')).astype(np.float32)
        unseen_feats = unseen_feats / np.linalg.norm(unseen_feats, axis=1, keepdims=True)
        unseen_labels = np.load(os.path.join(embed_dir, 'cifar10_vit_b32_labels.npy'))
    return unseen_feats, unseen_labels

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.expanduser('~/hpdic/EGA')
    dataset_keys = {'CIFAR-10': 'cifar10', 'Aircraft': 'aircraft', 'Food-101': 'food101'}
    budgets = [5, 10, 20]
    n_draws = 10

    all_rows = []  # per (dataset, budget) aggregate rows
    all_selection_events = []  # raw per-draw events for raw_results.csv

    for dataset, key in dataset_keys.items():
        print(f'=== {dataset} ===')
        unseen_feats, unseen_labels = load_dataset_pool(dataset, base_dir)
        gallery_idx, query_idx = canonical_gallery_query_split(unseen_labels, eval_seed=42)
        query_labels_full = unseen_labels[query_idx]

        candidates = get_candidate_embeddings(key, unseen_feats, device, base_dir)
        print(f'  candidates: {list(candidates.keys())}')

        # Precompute per-candidate gallery/query feature views
        cand_gallery = {name: feats[gallery_idx] for name, feats in candidates.items()}
        cand_query = {name: feats[query_idx] for name, feats in candidates.items()}
        gallery_labels = unseen_labels[gallery_idx]

        # Oracle test LP@1 per candidate (using the FULL query pool as test set, no validation carve-out)
        oracle_full = {}
        for name in candidates:
            oracle_full[name] = lp_at_1(cand_gallery[name], gallery_labels, cand_query[name], query_labels_full)
        print(f'  oracle full-query LP@1 per candidate: {oracle_full}')

        for budget in budgets:
            regrets = []
            within1 = []
            selected_counts = collections.Counter()
            for draw in range(n_draws):
                rng = np.random.RandomState(10_000 * budget + draw)
                val_local, test_local = stratified_draw(query_labels_full, budget, rng)
                if len(val_local) == 0 or len(test_local) == 0:
                    continue

                val_scores = {}
                test_scores = {}
                for name in candidates:
                    qf = cand_query[name]
                    ql = query_labels_full
                    val_scores[name] = lp_at_1(cand_gallery[name], gallery_labels, qf[val_local], ql[val_local])
                    test_scores[name] = lp_at_1(cand_gallery[name], gallery_labels, qf[test_local], ql[test_local])

                selected = max(val_scores, key=lambda n: val_scores[n])
                oracle_name = max(test_scores, key=lambda n: test_scores[n])
                oracle_test = test_scores[oracle_name]
                selected_test = test_scores[selected]
                regret = oracle_test - selected_test
                regrets.append(regret)
                within1.append(1 if regret <= 0.01 else 0)
                selected_counts[selected] += 1

                all_selection_events.append({
                    'experiment': 'selection', 'dataset': dataset, 'budget_per_class': budget,
                    'draw': draw, 'selected': selected, 'oracle': oracle_name,
                    'selected_test_lp1': selected_test, 'oracle_test_lp1': oracle_test, 'regret': regret,
                })

            if not regrets:
                continue
            regrets = np.array(regrets)
            frozen_rate = selected_counts.get('Frozen', 0) / len(regrets)
            ega_rate = sum(v for k, v in selected_counts.items() if k.startswith('EGA')) / len(regrets)
            row = {
                'dataset': dataset, 'budget': budget,
                'mean_regret': float(np.mean(regrets)), 'median_regret': float(np.median(regrets)),
                'within1_frac': float(np.mean(within1)),
                'frozen_rate': frozen_rate, 'ega_rate': ega_rate,
                'n_draws_used': len(regrets),
                'selected_counts': dict(selected_counts),
            }
            all_rows.append(row)
            print(f'  budget={budget}: mean_regret={row["mean_regret"]:.4f} within1={row["within1_frac"]:.2f} '
                  f'frozen_rate={frozen_rate:.2f} ega_rate={ega_rate:.2f} counts={dict(selected_counts)}')

    out_dir = os.path.join(base_dir, 'artifacts', 'rs4z_fast')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'selection_raw.json'), 'w') as f:
        json.dump({'aggregate': all_rows, 'events': all_selection_events}, f, indent=2)
    print('Saved selection_raw.json')

if __name__ == '__main__':
    main()
