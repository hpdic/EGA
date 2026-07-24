# NeurIPS Rebuttal Analysis: Supervised Contrastive + Pretrained Embedding Preservation Baseline

## Executive Summary
This document archives the empirical evaluation of the baseline requested by the NeurIPS reviewer for the **Euclidean Geodesic Alignment (EGA)** project:
A global supervised contrastive objective ($\text{InfoNCE}$) combined with a regularizer that preserves the frozen pretrained embedding:

$$\mathcal{L} = \mathcal{L}_{\text{InfoNCE}}(\text{out}, \text{labels}) + \lambda_{\text{reg}} \cdot \frac{1}{B} \sum_{i=1}^B \|\text{out}_i - \text{feats}_i\|_2^2$$

where both $\text{feats}$ and $\text{out} = \text{EGAMLP}(\text{feats})$ remain $L_2$-normalized unit vectors.

---

## Experimental Setup
* **Repository**: EGA
* **Git Commit**: `0442350caeceba3184c96f52a032f283b957ff7d`
* **Hardware**: NVIDIA A100 80GB PCIe GPU, 256 CPU Threads
* **Environment**: PyTorch 2.1.2+cu121, CUDA 12.6, FAISS-CPU 1.14.3, OpenAI CLIP (ViT-B/32)
* **Sweep Grid**:
  * $\lambda_{\text{reg}} \in \{0.01, 0.1, 1.0, 10.0\}$
  * Random Seeds $\in \{42, 123, 456\}$
  * Datasets: `CIFAR-10`, `FGVC-Aircraft`, `Food-101`
* **Evaluation Protocol**:
  * Protocol: $K=1$, $n_{\text{probe}}=1$, $N_{\text{list}}=100$, 75/25 gallery/query split on unseen test classes
  * Metrics: Label Precision at $K=1$ ($\text{LP}@1$), ANNS Recall at $K=1$ ($\text{AR}@1$), and Mean Embedding Displacement $\|\text{out} - \text{feats}\|_2$

---

## Verification Sanity Checks
Prior to the full sweep, verification runs on `FGVC-Aircraft` confirmed:
1. **$\lambda=0$ Reproduction**: $\text{LP}@1 = 0.5238, \text{AR}@1 = 0.5774$, reproducing baseline EGA+InfoNCE behavior within expected seed variation.
2. **`loss_preserve` Stability**: Finite and non-zero during training ($\text{Epoch 1} = 1.4206, \text{Epoch 150} = 1.6169$).
3. **Displacement Sensitivity**: Increasing $\lambda_{\text{reg}}$ monotonically reduces mean $\|out - feats\|_2$:
   $$\lambda_{\text{reg}} \in \{0.0, 0.01, 0.1, 1.0, 10.0\} \implies \|\text{out} - \text{feats}\|_2 \in \{1.2594, 1.1254, 0.8467, 0.5962, 0.2746\}$$
4. **Checkpoint Isolation**: Checkpoints saved strictly to `models/rebuttal_preserve_reg`, leaving existing checkpoints untouched.

---

## Statistical Summary (Mean ± Standard Error over 3 Seeds)

| $\lambda_{\text{reg}}$ | Dataset | LP@1 ($\text{Mean} \pm \text{SE}$) | AR@1 ($\text{Mean} \pm \text{SE}$) | Mean Displacement $\|\text{out} - \text{feats}\|_2$ | Worst-Case LP@1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0.01** | CIFAR-10 | $0.7051 \pm 0.0078$ | $0.5101 \pm 0.0141$ | 1.0992 | **0.5060** |
| **0.01** | FGVC-Aircraft | $0.5060 \pm 0.0268$ | $0.5734 \pm 0.0286$ | 1.1243 | |
| **0.01** | Food-101 | $0.6923 \pm 0.0075$ | $0.6296 \pm 0.0113$ | 0.9726 | |
| **0.1** | CIFAR-10 | $0.7477 \pm 0.0044$ | $0.5207 \pm 0.0095$ | 0.8190 | **0.5377** |
| **0.1** | FGVC-Aircraft | $0.5377 \pm 0.0189$ | $0.5357 \pm 0.0273$ | 0.8455 | |
| **0.1** | Food-101 | $0.7350 \pm 0.0052$ | $0.6405 \pm 0.0063$ | 0.7159 | |
| **1.0** | CIFAR-10 | $0.8036 \pm 0.0019$ | $0.5497 \pm 0.0059$ | 0.5747 | **0.5099** |
| **1.0** | FGVC-Aircraft | $0.5099 \pm 0.0201$ | $0.5198 \pm 0.0501$ | 0.5969 | |
| **1.0** | Food-101 | $0.8198 \pm 0.0007$ | $0.6611 \pm 0.0039$ | 0.4699 | |
| **10.0** | CIFAR-10 | $0.8865 \pm 0.0013$ | $0.5924 \pm 0.0046$ | 0.2400 | **0.5774** |
| **10.0** | FGVC-Aircraft | $0.5774 \pm 0.0137$ | $0.5615 \pm 0.0052$ | 0.2762 | |
| **10.0** | Food-101 | $0.8776 \pm 0.0028$ | $0.6514 \pm 0.0086$ | 0.2070 | |

---

## NeurIPS Rebuttal Markdown Table

```markdown
| $\lambda_{\text{reg}}$ | CIFAR-10 LP@1 | CIFAR-10 AR@1 | Aircraft LP@1 | Aircraft AR@1 | Food-101 LP@1 | Food-101 AR@1 | Worst LP@1 | Mean Displ. (CIFAR/Air/Food) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | 0.7051 ± 0.0078 | 0.5101 ± 0.0141 | 0.5060 ± 0.0268 | 0.5734 ± 0.0286 | 0.6923 ± 0.0075 | 0.6296 ± 0.0113 | **0.5060** | 1.099 / 1.124 / 0.973 |
| 0.1  | 0.7477 ± 0.0044 | 0.5207 ± 0.0095 | 0.5377 ± 0.0189 | 0.5357 ± 0.0273 | 0.7350 ± 0.0052 | 0.6405 ± 0.0063 | **0.5377** | 0.819 / 0.845 / 0.716 |
| 1.0  | 0.8036 ± 0.0019 | 0.5497 ± 0.0059 | 0.5099 ± 0.0201 | 0.5198 ± 0.0501 | 0.8198 ± 0.0007 | 0.6611 ± 0.0039 | **0.5099** | 0.575 / 0.597 / 0.470 |
| 10.0 | 0.8865 ± 0.0013 | 0.5924 ± 0.0046 | 0.5774 ± 0.0137 | 0.5615 ± 0.0052 | 0.8776 ± 0.0028 | 0.6514 ± 0.0086 | **0.5774** | 0.240 / 0.276 / 0.207 |
```
