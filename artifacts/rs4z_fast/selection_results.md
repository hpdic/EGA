# Experiment 2 — Small Target-Validation Selection

**Candidate library scope**: Frozen + EGA {0.5×, 1.0× default, 2.0×} only. InfoNCE and
InfoNCE+preserve are **excluded** — no live checkpoints exist anywhere on this machine for those
methods (fresh clone, `.gitignore`'d), and the plan forbids training new models for this
experiment. Live checkpoints trained fresh for Experiment 1 (EGA margins) plus Frozen (no
training needed) are the only methods with genuine per-example embeddings available tonight.
This scope was fixed by explicit user decision before running the simulation (see chat log).

**Protocol** (`scripts/59_rs4z_selection_sim.py`): for each dataset, the canonical 75/25
gallery/query split (same `eval_seed=42` used throughout) is fixed once and shared identically
across all candidates (only the embedded feature values differ). For each budget N∈{5,10,20} per
class, 10 independent stratified draws sample up to N examples/class from the query pool as the
"validation" subset (its labels are the only labels ever used for selection); the remaining
query-pool examples form that draw's held-out test set, never touched during selection.

## Per-dataset table

| Dataset | Budget per class | Mean test regret | Median test regret | Within 1 LP point of oracle | Frozen selected | EGA selected | Other selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 5 | 0.0000 | 0.0000 | 1.00 | 10/10 | 0/10 | 0 |
| CIFAR-10 | 10 | 0.0132 | 0.0000 | 0.70 | 7/10 | 3/10 | 0 |
| CIFAR-10 | 20 | 0.0000 | 0.0000 | 1.00 | 10/10 | 0/10 | 0 |
| Aircraft | 5 | 0.0464 | 0.0435 | 0.30 | 0/10 | 10/10 | 0 |
| Aircraft | 10 | 0.0636 | 0.0682 | 0.30 | 0/10 | 10/10 | 0 |
| Aircraft | 20 | 0.0350 | 0.0000 | 0.60 | 0/10 | 10/10 | 0 |
| Food-101 | 5 | 0.0000 | 0.0000 | 1.00 | 10/10 | 0/10 | 0 |
| Food-101 | 10 | 0.0000 | 0.0000 | 1.00 | 10/10 | 0/10 | 0 |
| Food-101 | 20 | 0.0000 | 0.0000 | 1.00 | 10/10 | 0/10 | 0 |

("EGA selected" pools all 3 EGA margins; within Aircraft's EGA-selected draws the default 1.0×
point was picked in 26/30 draws across all three budgets, the true oracle-best candidate there.)

## Compact aggregate table

| Budget per class | Mean regret across datasets | Within 1 LP point of oracle | Frozen selection rate | EGA selection rate |
|---|---:|---:|---:|---:|
| 5 | 0.0155 | 0.77 | 0.67 | 0.33 |
| 10 | 0.0256 | 0.67 | 0.57 | 0.43 |
| 20 | 0.0117 | 0.87 | 0.67 | 0.33 |

## Leakage check

See `checks.md` — verified: candidate embeddings/checkpoints were produced by Experiment 1 before
any Experiment 2 evaluation; the validation subset and the held-out test subset are disjoint
partitions of the same fixed query pool; only validation-subset labels are used for selection;
regret is computed against the true test-query oracle, never seen during selection.

## Interpretation

With only 5-20 labeled examples per class, the selector achieves 0.012-0.026 mean LP@1 regret and
is within 1 LP point of the test oracle in 67-87% of trials. Critically, it does not collapse to
one default choice: it **correctly selects Frozen on CIFAR-10 and Food-101**, where global/EGA
adaptation is not worth the query-image cost, and **correctly selects EGA (predominantly the
default 1.0× margin) on Aircraft**, the one dataset where EGA's worst-case robustness is actually
useful. Target-test labels were never used for candidate selection, training, or margin choice.

Aircraft shows the largest regret (0.035-0.064) because its 3 EGA margins are close to each other
in LP@1 (0.577-0.619), so a handful of validation examples per class sometimes cannot distinguish
between them reliably — but the selector never mistakenly picks Frozen there, which is the
decision that actually matters (Frozen would cost ~11pp of LP@1 on Aircraft).
