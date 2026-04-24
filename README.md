# Distributionally Aligned Contrastive Preference Learning

In this code-base we extend [*Contrastive Preference Learning: Learning From Human Feedback without RL*](https://arxiv.org/abs/2310.13639) (Hejna et al., 2023) with two new distributionally aligned preference learning objectives based on optimal transport.

The original CPL paper and codebase can be found at [jhejna/cpl](https://github.com/jhejna/cpl). This repository is based on a frozen version of [research-lightning](https://github.com/jhejna/research-lightning).

---

## Overview

Standard CPL trains a policy by contrasting preferred and rejected segments **within each preference pair**. This repository introduces two new algorithms that instead apply **1-D optimal transport (OT) matching** across the batch before computing the contrastive loss, producing distributional alignment between preferred and rejected behaviors.

### Algorithms

**CPL (baseline)**
The original contrastive preference learning algorithm. Trains a policy π_θ by minimizing a softmax cross-entropy loss over preference pairs. Supports both direct pair comparison (`mode: comparison`) and rank-based matching (`mode: rank`).

**CPL with uAOT** (`CPL_uAOT`, `research/algs/cpl_uaot.py`)
Unpaired Alignment via Optimal Transport. At each training step:
1. Computes segment scores for all preferred and rejected segments in the batch under π_θ.
2. Sorts preferred scores and rejected scores **independently** from lowest to highest.
3. Matches the i-th ranked preferred score with the i-th ranked rejected score (1-D OT via the northwest corner method).
4. Computes the CPL loss over these OT-matched pairs.

This allows cross-pair comparisons: a highly-ranked preferred segment is compared against a highly-ranked rejected one, providing a stronger distributional alignment signal than within-pair comparison alone. No reference policy is required.

**CPL with pAOT** (`CPL_pAOT`, `research/algs/cpl_paot.py`)
Paired Alignment via Optimal Transport. Operates on per-pair **margins** (preferred score minus rejected score) rather than raw scores, and uses a frozen reference policy π_ref as an anchor:
1. Trains π_ref via behavior cloning for `ref_bc_steps` steps. π_θ is untouched during this phase.
2. Freezes π_ref. Optionally warms up π_θ via BC for `theta_bc_steps` steps (default 0, keeping π_θ fresh).
3. For each preference pair, computes the margin under π_θ and under π_ref.
4. Sorts both sets of margins independently and OT-matches them by rank.
5. Trains π_θ to produce margins that exceed the reference margins after OT re-matching.

The reference policy logic is isolated in `research/algs/reference_policy.py` and can be replaced with an alternative (e.g. BCO) without modifying the main training file.

### Theoretical Basis

All three algorithms share the same two assumptions from the original CPL paper:
- **Regret-based Boltzmann rationality**: annotators prefer segments with higher cumulative discounted optimal advantage.
- **MaxEnt optimality**: the optimal policy maximizes an entropy-regularized return, inducing the bijection A*(s,a) ≡ α log π*(a|s).

The uAOT and pAOT objectives are derived in `literature/aot_cpl.tex`.

---

## Repository Structure

```
research/
  algs/
    cpl.py              # baseline CPL
    cpl_kl.py           # CPL with KL regularization from reference policy
    cpl_uaot.py         # CPL with Unpaired AOT (this work)
    cpl_paot.py         # CPL with Paired AOT (this work)
    reference_policy.py # isolated reference policy utilities for pAOT
    bc.py               # behavior cloning baseline
    piql.py             # preference-based IQL
    sac.py              # soft actor-critic
  datasets/
    feedback_buffer.py  # preference dataset loader (comparison and rank modes)
  networks/             # MLP and DrQv2 policy/encoder architectures
  utils/
    trainer.py          # main training loop
    config.py           # YAML config loading and class resolution

configs/
  mw_state_dense/       # state-based MetaWorld, dense reward
  mw_state_sparse/      # state-based MetaWorld, sparse reward
  mw_image_dense/       # image-based MetaWorld, dense reward
  mw_image_sparse/      # image-based MetaWorld, sparse reward
  # each directory contains: cpl.yaml, cpl_uaot.yaml, cpl_paot.yaml,
  #                          cpl_bc.yaml, cpl_kl.yaml, piql.yaml, sft.yaml

literature/
  aot_cpl.tex           # mathematical formalization of uAOT and pAOT objectives
  cpl.pdf               # original CPL paper
  aot.pdf               # reference material on alignment via optimal transport

scripts/
  train.py              # training entry point
  evaluate.py           # evaluation entry point
```

---

## Installation

### Standard (local)

1. Clone the repository: `git clone https://github.com/mduboef/aot-cpl`
2. Create the conda environment: `conda env create -f environment_<cpu or gpu>.yaml`
3. Install the research package: `pip install -e .`
4. Download the MetaWorld preference datasets from the [original CPL repository](https://github.com/jhejna/cpl) and extract into `datasets/` at the repository root.

### Google Colab (recommended for GPU access)

The notebook `AOT_CPL.ipynb` contains a fully self-contained setup sequence for Colab. It handles MuJoCo binary installation, dependency pinning (numpy<2, Cython 0.29.36), and dataset extraction from Google Drive. Run all setup cells top to bottom before the training cells.

The dataset zip should be placed at `/content/drive/MyDrive/aotCPL/mw.zip` on your Drive, containing `mw/pref/` at the root of the zip.

---

## Training

```bash
python scripts/train.py --config path/to/config.yaml --path path/to/output/dir
```

For example, to train CPL_uAOT on the state-dense MetaWorld drawer-open task:

```bash
python scripts/train.py --config configs/mw_state_dense/cpl_uaot.yaml --path runs/uaot_dense
```

Results (checkpoints, `log.csv`, TensorBoard events) are saved to the output directory.

---

## Key Hyperparameters

| Parameter | Algorithms | Description |
|-----------|-----------|-------------|
| `alpha` | all | Temperature scaling log-probs into advantage scores. Most important hyperparameter. |
| `bc_steps` | CPL, CPL_uAOT | BC warmup steps before contrastive phase begins. |
| `ref_bc_steps` | CPL_pAOT | Steps to BC-train π_ref. π_θ is untouched during this phase. |
| `theta_bc_steps` | CPL_pAOT | Steps to BC-warm π_θ after π_ref is frozen. Default 0 (fresh π_θ). |
| `bc_coeff` | all | Weight of BC regularization term during contrastive phase. |
| `contrastive_bias` | CPL only | Asymmetry parameter in the baseline CPL loss. |

### Data mode note
CPL_uAOT and CPL_pAOT require `mode: comparison` in the dataset config (each batch contains paired segments with preference labels). Baseline CPL configs use `mode: rank` by default but also support `mode: comparison`.

---
