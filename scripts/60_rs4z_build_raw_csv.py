import json, csv, os

base = os.path.expanduser('~/hpdic/EGA')
out_dir = os.path.join(base, 'artifacts', 'rs4z_fast')
rows = []

def add(experiment, dataset, method, hyper, seed, split, lp1, ar1, source_file, command):
    rows.append(dict(experiment=experiment, dataset=dataset, method=method, hyperparameter=hyper,
                      seed=seed, split=split, lp1=lp1, ar1=ar1, source_file=source_file, command=command))

# ---- Paper table (tab:ood_summary), camera-ready, mean over 3 seeds ----
paper_rows = [
    ('CIFAR-10', 'Frozen', 0.880, 0.863),
    ('CIFAR-10', 'ICon', 0.560, 0.797),
    ('CIFAR-10', 'SRL', 0.536, 0.819),
    ('CIFAR-10', 'InfoNCE(LoRA)', 0.885, 0.869),
    ('CIFAR-10', 'LoRA+Triplet', 0.875, 0.861),
    ('CIFAR-10', 'EGA_default', 0.810, 0.833),
    ('Aircraft', 'Frozen', 0.512, 0.773),
    ('Aircraft', 'ICon', 0.470, 0.853),
    ('Aircraft', 'SRL', 0.375, 0.879),
    ('Aircraft', 'InfoNCE(LoRA)', 0.538, 0.891),
    ('Aircraft', 'LoRA+Triplet', 0.569, 0.893),
    ('Aircraft', 'EGA_default', 0.611, 0.905),
    ('Food-101', 'Frozen', 0.881, 0.842),
    ('Food-101', 'ICon', 0.562, 0.821),
    ('Food-101', 'SRL', 0.552, 0.824),
    ('Food-101', 'InfoNCE(LoRA)', 0.883, 0.906),
    ('Food-101', 'LoRA+Triplet', 0.833, 0.893),
    ('Food-101', 'EGA_default', 0.791, 0.879),
    ('ImageNet-1K', 'Frozen', 0.684, 0.829),
    ('ImageNet-1K', 'ICon', 0.620, 0.849),
    ('ImageNet-1K', 'SRL', 0.665, 0.747),
    ('ImageNet-1K', 'InfoNCE(LoRA)', 0.753, 0.878),
    ('ImageNet-1K', 'LoRA+Triplet', 0.714, 0.883),
    ('ImageNet-1K', 'EGA_default', 0.711, 0.868),
]
for ds, method, lp1, ar1 in paper_rows:
    hyper = '0.2' if method == 'EGA_default' else ''
    add('exp1_or_exp3_baseline', ds, method, hyper, 'mean_of_42_123_456', 'ood_unseen',
        lp1, ar1, 'paper/EGA.tex#tab:ood_summary', 'camera-ready paper table (no command; pre-existing)')

# ---- InfoNCE+preserve (rebuttal_preserve_reg_results.json) ----
preserve = json.load(open(os.path.join(base, 'rebuttal_preserve_reg_results.json')))['results']
for r in preserve:
    add('exp1', r['dataset'], 'InfoNCE+preserve', r['lambda_reg'], r['seed'], 'ood_unseen',
        r['lp1'], r['ar1'], 'rebuttal_preserve_reg_results.json',
        'scripts/50_infonce_preserve_reg_ood.py (pre-existing rebuttal run)')

# ---- Aircraft margin_geometry sweep (rebuttal_margin_geometry_results.json) ----
mg = json.load(open(os.path.join(base, 'rebuttal_margin_geometry_results.json')))['results']
for r in mg:
    add('exp1_reference', 'Aircraft', 'EGA', r['margin'], r['seed'], 'ood_unseen',
        r['lp1'], r['ar1'], 'rebuttal_margin_geometry_results.json',
        'scripts/57_margin_geometry_sweep.py (pre-existing rebuttal run, referenced not primary)')

# ---- New emergency margin runs tonight ----
for ds_key, ds_name in [('cifar10', 'CIFAR-10'), ('food101', 'Food-101'), ('aircraft', 'Aircraft')]:
    fpath = os.path.join(out_dir, f'missing_margins_{ds_key}.json')
    if not os.path.exists(fpath):
        continue
    data = json.load(open(fpath))
    for r in data:
        cmd = f"python scripts/58_rs4z_missing_margins.py --dataset {ds_key} --margins {r['margin']} --seeds {r['seed']}"
        add('exp1', ds_name, 'EGA', r['margin'], r['seed'], 'ood_unseen',
            round(r['lp1'], 4), round(r['ar1'], 4), f'artifacts/rs4z_fast/missing_margins_{ds_key}.json', cmd)

# ---- Selection simulation (Experiment 2) ----
sel = json.load(open(os.path.join(out_dir, 'selection_raw.json')))
for e in sel['events']:
    add('exp2', e['dataset'], f"selected={e['selected']}", e['budget_per_class'], e['draw'], 'validation_then_test',
        round(e['selected_test_lp1'], 4), '', 'artifacts/rs4z_fast/selection_raw.json',
        'python scripts/59_rs4z_selection_sim.py')

# ---- Regime (Experiment 3) — derived from paper table, computed inline in regime_results.md ----
add('exp3', 'multiple', 'derived-from-paper-table', '', '', '',
    '', '', 'paper/EGA.tex#tab:ood_summary', 'no new command; arithmetic on paper table, see regime_results.md')

out_path = os.path.join(out_dir, 'raw_results.csv')
with open(out_path, 'w', newline='') as f:
    fieldnames = ['experiment', 'dataset', 'method', 'hyperparameter', 'seed', 'split', 'lp1', 'ar1', 'source_file', 'command']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'Wrote {len(rows)} rows to {out_path}')
