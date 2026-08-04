# Experiment 1 — Minimal EGA Tradeoff Table

Default triplet margin = 0.2 (`nn.TripletMarginLoss(margin=0.2, p=2)`, `scripts/02_train_ega.py`).
Margin triplet: 0.5×=0.1, 1.0×=0.2 (default), 2.0×=0.4.

**Sources** (see `raw_results.csv` for exact command/seed/file mapping):
- Frozen, InfoNCE (=LoRA+InfoNCE), EGA-default: `paper/EGA.tex` Table `tab:ood_summary`
  (camera-ready, mean±std over 3 seeds). A fresh single-seed re-run of EGA-default tonight
  (same pipeline as the new margin points) reproduces these within ~0.6-0.8pp on all 3
  datasets (0.8164/0.6190/0.7852 vs paper's 0.810/0.611/0.791) — a useful consistency check,
  not used as the reported number.
- InfoNCE+preserve: `rebuttal_preserve_reg_results.json`, best λ (highest LP@1, consistently
  λ=10.0 across all 3 datasets), mean±SE over 3 seeds.
- EGA 0.5×/2.0×: new emergency runs tonight (`scripts/58_rs4z_missing_margins.py`), same
  training/eval code for all 3 datasets. CIFAR-10 and Food-101 have 3 seeds (42,123,456);
  Aircraft has **1 seed only** (marked below) — flagged as a real limitation, not padded.

## LP@1 table

| Method | Margin / λ | CIFAR-10 LP@1 | Aircraft LP@1 | Food-101 LP@1 | Mean LP@1 | Worst LP@1 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | — | 0.880 | 0.512 | 0.881 | 0.758 | 0.512 |
| InfoNCE (LoRA) | — | 0.885 ± .006 | 0.538 ± .020 | 0.883 ± .004 | 0.769 | 0.538 |
| InfoNCE+preserve | λ=10.0 | 0.8865 ± .0013 | 0.5774 ± .0137 | 0.8776 ± .0028 | 0.7805 | 0.5774 |
| EGA | 0.5× (m=0.1) | 0.8455 ± .0047 | 0.6071 (single seed) | 0.8243 ± .0035 | 0.7590 | 0.6071 |
| EGA | 1.0× (m=0.2, default) | 0.810 ± .002 | 0.611 ± .020 | 0.791 ± .002 | 0.737 | 0.611 |
| EGA | 2.0× (m=0.4) | 0.7336 (single seed) | 0.5774 (single seed) | 0.7174 (single seed) | 0.676 | 0.5774 |

## AR@1 table

| Method | Margin / λ | CIFAR-10 AR@1 | Aircraft AR@1 | Food-101 AR@1 | Mean AR@1 |
|---|---:|---:|---:|---:|---:|
| Frozen | — | 0.863 | 0.773 | 0.842 | 0.826 |
| InfoNCE (LoRA) | — | 0.869 ± .012 | 0.891 ± .015 | 0.906 ± .003 | 0.889 |
| InfoNCE+preserve | λ=10.0 | 0.5924 ± .0046 | 0.5615 ± .0052 | 0.6514 ± .0086 | 0.602 |
| EGA | 0.5× (m=0.1) | 0.8556 ± .0050 | 0.8869 (single seed) | 0.8883 ± .0028 | 0.877 |
| EGA | 1.0× (m=0.2, default) | 0.833 ± .005 | 0.905 ± .015 | 0.879 ± .011 | 0.872 |
| EGA | 2.0× (m=0.4) | 0.8324 (single seed) | 0.8810 (single seed) | 0.8743 (single seed) | 0.863 |

## Required calculations

Selected non-default point: **EGA 0.5× (m=0.1)** — the only alternative margin that improves mean
LP@1 over default (0.759 vs 0.737) while costing only 0.4pp of worst-case LP@1 (0.6071 vs 0.611,
on Aircraft). EGA 2.0× is dominated (worse than both 0.5× and default on every dataset) and is not
a useful operating point.

```
average gap recovered = (balanced_EGA_mean - default_EGA_mean) / (best_non_EGA_mean - default_EGA_mean)
best_non_EGA_mean = max(Frozen .758, InfoNCE .769, InfoNCE+preserve .7805) = .7805 (InfoNCE+preserve)
= (0.7590 - 0.7373) / (0.7805 - 0.7373) = 0.0217 / 0.0432 = 0.50  (50%)

worst-case benefit retained = (balanced_EGA_worst - frozen_worst) / (default_EGA_worst - frozen_worst)
= (0.6071 - 0.512) / (0.611 - 0.512) = 0.0951 / 0.099 = 0.96  (96%)
```

Both denominators are positive; ratios are used as-is.

## Interpretation

EGA 0.5× does **not** strictly dominate the default point (it is 0.4pp worse on Aircraft), so we
do not claim a full Pareto frontier — only a genuine, non-dominated controllable tradeoff:

> The margin sweep exposes a controllable tradeoff. The alternative EGA operating point (0.5×
> default margin) improves mean LP@1 from 0.737 to 0.759 — recovering 50% of the mean-LP@1 gap to
> the strongest non-EGA baseline — while retaining 96% of the default point's worst-case gain over
> the frozen encoder. The originally reported configuration is a conservative point on a tunable
> curve, not the only behavior EGA can realize.
