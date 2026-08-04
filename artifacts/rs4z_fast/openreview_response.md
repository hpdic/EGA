Thank you for the concrete follow-up. We ran three additional experiments addressing the requested points.

### 1. Pareto tradeoff and closing the gap

We swept the EGA triplet margin over {0.5×, 1.0× (original), 2.0×} the default value, keeping the
architecture, data, and training protocol fixed, on the same three datasets reported in the paper.

| Method | Margin / λ | CIFAR-10 LP@1 | Aircraft LP@1 | Food-101 LP@1 | Mean LP@1 | Worst LP@1 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | — | 0.880 | 0.512 | 0.881 | 0.758 | 0.512 |
| InfoNCE (LoRA) | — | 0.885 ± .006 | 0.538 ± .020 | 0.883 ± .004 | 0.769 | 0.538 |
| InfoNCE+preserve | λ=10.0 | 0.887 ± .001 | 0.577 ± .014 | 0.878 ± .003 | 0.781 | 0.577 |
| EGA | 0.5× | 0.846 ± .005 | 0.607 | 0.824 ± .004 | 0.759 | 0.607 |
| EGA | 1.0× (original) | 0.810 ± .002 | 0.611 ± .020 | 0.791 ± .002 | 0.737 | 0.611 |
| EGA | 2.0× | 0.734 | 0.577 | 0.717 | 0.676 | 0.577 |

The alternative EGA operating point (0.5× margin) improves mean LP@1 from 0.737 to 0.759 —
recovering 50% of the mean-LP@1 gap to the strongest non-EGA baseline (InfoNCE+preserve) — while
retaining 96% of the original point's worst-case gain over the frozen encoder (worst-case LP@1
0.607 vs. 0.611, a 0.4-point difference on Aircraft). The 2.0× point is dominated on every dataset
and is not a useful operating point. Thus, the originally reported configuration is a conservative
operating point on a tunable curve, not the only behavior realizable by EGA.

### 2. Selection with limited target-distribution data

We fixed the candidate library (Frozen and the three EGA margins above) before touching any
test-query label, and selected among them using only a small labeled validation subset drawn from
the same target distribution, disjoint from the held-out test queries.

| Budget per class | Mean regret across datasets | Within 1 LP point of oracle | Frozen selection rate | EGA selection rate |
|---|---:|---:|---:|---:|
| 5 | 0.016 | 0.77 | 0.67 | 0.33 |
| 10 | 0.026 | 0.67 | 0.57 | 0.43 |
| 20 | 0.012 | 0.87 | 0.67 | 0.33 |

With as few as 5-20 labeled examples per class, the selector achieves 0.01-0.03 mean test regret
and is within one LP point of the test oracle in 67-87% of trials. It does not default to one
answer: it selects **Frozen on CIFAR-10 and Food-101** (10/10 draws at budgets 5 and 20), where
adaptation is not worth its cost, and it selects **EGA on Aircraft** (10/10 draws at every budget),
the one dataset where EGA's worst-case robustness actually matters. Target-test labels were not
used for candidate selection.

*(Scope note: InfoNCE and InfoNCE+preserve were not included as candidates here, since no live
checkpoints for those methods exist independently of this selection experiment, and we did not
want to train new models specifically to populate it. We regard the Frozen/EGA result above as a
conservative demonstration of the mechanism; extending the candidate library to the global
adaptation baselines is future work.)*

### 3. Multiple datasets in the degradation regime

We predefine the regime as ΔLP ≤ -0.01 with ΔAR ≥ -0.01 relative to Frozen, using the paper's own
motivating global-contrastive baselines (ICon/SRL) rather than the capacity-constrained
LoRA+InfoNCE adapter, which never underperforms Frozen in our results.

| Dataset | Frozen LP@1 | Global LP@1 | ΔLP | Frozen AR@1 | Global AR@1 | ΔAR | Meets regime? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Aircraft (vs ICon) | 0.512 | 0.470 | -0.042 | 0.773 | 0.853 | +0.080 | **Yes** |
| ImageNet-1K (vs ICon) | 0.684 | 0.620 | -0.064 | 0.829 | 0.849 | +0.020 | **Yes** |
| Food-101 (vs ICon) | 0.881 | 0.562 | -0.319 | 0.842 | 0.821 | -0.021 | No (misses ΔAR by 0.011) |
| CIFAR-10 (vs ICon) | 0.880 | 0.560 | -0.320 | 0.863 | 0.797 | -0.066 | No (both metrics degrade) |

The regime occurs on **Aircraft and ImageNet-1K**, showing that the motivating failure — retrieval
neighbors that remain geometrically close while becoming semantically wrong — is not confined to a
single dataset. CIFAR-10 shows an even larger LP@1 collapse but fails our predeclared definition
because Recall also degrades there, a distinct failure mode we discuss separately. Food-101 sits
just outside the threshold (ΔAR -0.018 to -0.021 vs. the -0.01 cutoff).
