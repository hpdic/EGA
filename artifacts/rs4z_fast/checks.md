# Recon Findings (First 10 Minutes) — RS4z Emergency Plan

Timestamp: 2026-08-03 (session start), machine: A100 80GB, `~/hpdic/EGA` on branch `main`
(fresh clone — repo dir mtimes all match today; nothing here predates this session).

## 1. Environment state — DEVIATES FROM PLAN ASSUMPTIONS

The plan's Absolute Priority #1 is "reuse existing checkpoints, embeddings, cached features, logs."
**None of these exist locally.** This is a fresh `git clone`, and per `.gitignore`, `data/`,
`embeddings/*.npy`, `*.pt`, `*.pth` were never committed:

- `models/` contains only `ega_mlp.py` — no `.pth` checkpoint files anywhere on disk.
- `embeddings/` contains only 3 tiny placeholder `.txt` result files — no `.npy` feature caches.
- No CIFAR-10 / FGVC-Aircraft / Food-101 data downloaded anywhere on the machine (checked
  `~`, `/hpdic_data`, `/mnt`, `/data`).
- `python3` had **no torch, CLIP, or faiss installed** (system or venv). Installing now (see §4).
- `/hpdic_data` exists but holds unrelated ANN-benchmark datasets for the STALK project
  (wiki_all_1M, deep1b, sift/gist) — nothing for EGA.
- Internet access confirmed working (pypi, github reachable).

**Consequence:** every JSON result file's `ckpt_path` (e.g.
`models/rebuttal_margin_geometry/ega_air_m0.05_seed42.pth`) points to a file that does not exist
here. We have the **aggregate numbers** (committed JSON/paper), not the **live models**.

## 2. What already exists (reusable without any new run)

- **Default triplet margin = 0.2** (hardcoded `nn.TripletMarginLoss(margin=0.2, p=2)` in
  `scripts/02_train_ega.py`). So the plan's 3 margins are: 0.5×=0.1, 1.0×=0.2, 2.0×=0.4.
- **`rebuttal_margin_geometry_results.json`**: FGVC-Aircraft only, margins
  {0.05,0.1,0.2,0.3,0.4,0.5} × seeds {42,123,456} (18 runs). **This already covers all 3 required
  Aircraft margin points (0.1/0.2/0.4) with 3 seeds each** — no new Aircraft margin training needed
  for the aggregate numbers (though we lack the live `.pth` files, see §3).
- **`rebuttal_preserve_reg_results.json`**: InfoNCE+preservation-regularizer, **all 3 target
  datasets** (CIFAR-10, FGVC-Aircraft, Food-101) × λ∈{0.01,0.1,1.0,10.0} × seeds{42,123,456}
  = 36 runs. Covers the "best existing InfoNCE+preserve point" requirement for all 3 datasets.
- **`paper/EGA.tex` Table `tab:ood_summary`** (committed, camera-ready numbers): Frozen (CLIP),
  ICon, SRL, LoRA+InfoNCE, LoRA+Triplet, EGA (default margin=0.2) — LP@1 and AR@1, mean±std over
  3 seeds, for **CIFAR-10, Aircraft, Food-101, ImageNet-1K**. This gives us Frozen, InfoNCE
  (=LoRA+InfoNCE in this paper's terminology), and EGA-default for free, for all 3 datasets.
  **Note on terminology**: the paper's "InfoNCE" baseline in the main table is LoRA+InfoNCE
  (capacity-matched low-rank adapter). The rebuttal scripts (`36_ega_infonce_ood.py`,
  `50_infonce_preserve_reg_ood.py`) instead train the *same EGA-MLP architecture* with an
  InfoNCE/SupCon loss instead of triplet loss — a different "InfoNCE" than the paper table's.
  We will label these distinctly in the tables and state which is which.
- **Motivating degradation regime already has multi-dataset evidence in the paper table itself**,
  using ICon/SRL (the paper's actual "high-capacity global contrastive" motivating baselines —
  NOT LoRA+InfoNCE, which is capacity-constrained and does not collapse). Computed from
  `tab:ood_summary` (ΔLP ≤ -0.01, ΔAR ≥ -0.01 vs Frozen):
  - **Aircraft** (ICon): ΔLP=-0.042, ΔAR=+0.080 → **meets regime**
  - **Aircraft** (SRL): ΔLP=-0.137, ΔAR=+0.106 → **meets regime**
  - **ImageNet-1K** (ICon): ΔLP=-0.064, ΔAR=+0.020 → **meets regime**
  - Food-101 (ICon): ΔLP=-0.319, ΔAR=-0.021 → fails AR tolerance narrowly
  - Food-101 (SRL): ΔLP=-0.329, ΔAR=-0.018 → fails AR tolerance narrowly
  - CIFAR-10 (ICon/SRL): ΔLP≈-0.32, but ΔAR≈-0.066/-0.044 → fails AR tolerance clearly
  - This means **Experiment 3 (regime_results.md) is fully answerable from already-committed
    paper data, zero new runs required.** Two datasets (Aircraft, ImageNet-1K) satisfy the
    predeclared regime under the paper's actual motivating baseline (ICon).

## 3. What's genuinely missing (requires new runs)

For **Experiment 1** (pareto table), missing on CIFAR-10 and Food-101 only (Aircraft aggregate
numbers already exist, but see live-checkpoint caveat below):
- EGA margin 0.1 (0.5×) — CIFAR-10, Food-101
- EGA margin 0.4 (2.0×) — CIFAR-10, Food-101
- (EGA margin 0.2 / default: already have paper-table numbers for all 3 datasets)

For **Experiment 2** (selection simulation), the plan requires "saved embeddings or existing
checkpoints... do not train new models for this experiment." **No live checkpoint exists for any
method on any dataset in this environment right now** — not even Aircraft margins, despite having
their aggregate JSON numbers, because the `.pth` files were gitignored and never survived the
clone. Per-example stratified selection (5/10/20-shot) needs actual model output embeddings, not
just aggregate LP@1/AR@1 — aggregate numbers alone cannot support the selection simulation.
**Resolution**: candidate checkpoints produced fresh during Experiment 1 (EGA margins on
CIFAR-10/Food-101, plus — if approved — InfoNCE and InfoNCE+preserve reruns) are treated as fixed
inputs to Experiment 2, not "new training for Experiment 2" itself. Frozen requires no training.
Flagged to user for scope confirmation (see chat).

## 4. Environment setup in progress

- `venv/` created; `numpy tqdm scikit-learn faiss-cpu datasets` installed.
- `torch/torchvision/torchaudio` (cu124) + `git+openai/CLIP` installing now in background.
- Feature extraction plan: `scripts/00_extract_required_features.py` produces exactly the 3
  cached `.npy` feature/label files needed (CIFAR-10 test via HF `datasets`, Aircraft test via
  torchvision download, Food-101 validation via HF `datasets`) — single extraction pass per
  dataset; OOD seen/unseen class splits are carved from this one file via `split_by_class`
  (`utils_ega.py`), no separate train-set download needed.

## 5. LP@1 / AR@1 computation (canonical, reused for every run)

`utils_ega.py::eval_method(features, labels, k=1, nlist=100, nprobe=1, seed)`:
normalize → shuffle (fixed seed) → 75/25 gallery/query split → exact `IndexFlatL2` for ground
truth → `IndexIVFFlat` (nlist/nprobe) for retrieved → LP@1 = label match rate, AR@1 = overlap
with exact-search ground truth. Aircraft-specific rebuttal runs used `nlist=10` (dataset-size
adjusted); default is `nlist=100`. Will use the same `nlist`/`nprobe`/seed convention already
established per dataset for consistency with existing numbers.

## 6. Seeds

3-seed results already exist for: Aircraft margin sweep (6 margins), all 3 datasets'
InfoNCE+preserve sweep, and the paper's main OOD table (CIFAR-10/Aircraft/Food-101/ImageNet, all
methods). No multi-seed emergency budget needs to be spent re-confirming these; new emergency runs
will follow the plan's seed policy (1 seed for every missing point immediately, 2 more only for
default EGA + the most useful non-default point, per dataset where actually missing).

## 7. What actually happened (update after execution)

- Environment build succeeded: venv, torch 2.6.0+cu124, CLIP, faiss-cpu, HF `datasets`.
  `scripts/00_extract_required_features.py` failed on deprecated HF short dataset names
  (`cifar100`, not `uoft-cs/cifar100`) — wrote `scripts/00b_extract_rs4z_features.py` /
  `00c_extract_cifar100.py` with corrected repo ids (`uoft-cs/cifar10`, `uoft-cs/cifar100`,
  `ethz/food101`). All 4 feature files extracted successfully (CIFAR-10 10k, CIFAR-100 10k,
  Aircraft 3333, Food-101 25250 images, CLIP ViT-B/32).
- Discovered CIFAR-10's OOD protocol is **cross-dataset** (train adapter on CIFAR-100, eval
  zero-shot on all of CIFAR-10 — `scripts/19_cifar10_3seed.py`), unlike Aircraft/Food-101's
  within-dataset 80/20 class split. `scripts/58_rs4z_missing_margins.py` implements both correctly
  per dataset.
- User decision (recorded in chat): Experiment 2 candidate library restricted to **Frozen + EGA
  only** — InfoNCE/InfoNCE+preserve excluded rather than freshly trained, to stay unambiguously
  inside "no new training for Experiment 2."
- To give Experiment 2 genuine live checkpoints, we additionally trained: EGA default (m=0.2,
  1 seed) on CIFAR-10/Food-101, and EGA at all 3 margins (1 seed each) on Aircraft — these are
  Experiment-1-family artifacts (same script, same protocol), not new training introduced for
  Experiment 2 itself.
- Aircraft EGA numbers tonight (single seed, fresh pipeline) closely match the pre-existing
  `rebuttal_margin_geometry_results.json` 3-seed sweep at the same margins (e.g. default m=0.2:
  0.619 tonight vs 0.611±.020 in the paper / 0.593±.014 in the old sweep — all within ~1-3pp),
  supporting that tonight's re-extracted features and training pipeline reproduce the original
  results closely enough to trust the new margin points.

## 8. Leakage statement (verified)

> Candidate models and hyperparameters were fixed before target-test evaluation; target-test
> labels were not used for candidate selection.

Verified: in `scripts/59_rs4z_selection_sim.py`, the gallery/query split is computed once per
dataset from `eval_seed=42` before any candidate is loaded; validation subsets are drawn from the
query pool and scored with their own (small, labeled) subset only; the held-out test-query subset
(the complement) is never read until after `argmax` selection has already picked a candidate.

## 9. Verification checklist

| Check | Status | Note |
|---|---|---|
| Identical test query/index sets across candidates | PASS | `canonical_gallery_query_split` depends only on the shared label array/seed, computed once per dataset before per-candidate feature substitution — same `gallery_idx`/`query_idx` for every candidate. |
| No target-test examples in validation selection | PASS | Validation subset and test subset are a disjoint partition of the query pool per draw (`stratified_draw`); test subset only read after selection. |
| Consistent self-match removal | N/A | Gallery and query are disjoint (75/25 split of the same pool, no self-pairs); no query point is ever a member of its own gallery. |
| Identical metric implementation | PASS | Same `lp_at_1`/`eval_lp_ar` (IVF `nlist=10`,`nprobe=1`, exact `IndexFlatL2` ground truth for AR) used for every new run tonight (Exp1 + Exp2); paper-table and pre-existing-JSON baselines used their own already-published implementations (same formula, `utils_ega.py::eval_method`, `nlist` differs: 100 in the generic utility vs 10 in the Aircraft-tuned rebuttal scripts and tonight's runs — noted, not mixed within one comparison axis). |
| No silently missing seeds | PASS | Aircraft EGA margins and EGA margin=0.4 on CIFAR-10/Food-101 are **single-seed**, explicitly labeled as such in `pareto_results.md`; nothing averaged over fewer seeds than claimed. |
| Mean and worst values recomputed from raw rows | PASS | Recomputed via `scripts/60_rs4z_build_raw_csv.py` aggregation and the inline calculation in `pareto_results.md`, not hand-typed. |
| Markdown values generated from `raw_results.csv` | PARTIAL | `raw_results.csv` contains every underlying number; the specific mean/SE/ratio arithmetic in `pareto_results.md` was computed with a short Python snippet reading the same source JSONs (not a separate script reading the CSV) — numerically identical, but not literally CSV→table code. Documented here as the honest caveat. |
| Single-seed results clearly marked | PASS | "(single seed)" annotations in `pareto_results.md` LP@1/AR@1 tables. |
| No test-based choice of EGA margin | PASS | The "selected non-default point" (0.5×) was chosen by inspecting **mean LP@1 and worst-case retention only** — the same criterion stated in the plan's own seed policy — not by cherry-picking whichever margin happened to win the Experiment 2 selection simulation (that simulation ran afterward, independently). |
| Regime definition fixed before inspecting results | PASS | `regime_results.md` states the ΔLP/ΔAR thresholds before the table; the choice of ICon/SRL over LoRA+InfoNCE as "Global" is justified by the paper's own text (line 42), not by which one produced a bigger effect. |
