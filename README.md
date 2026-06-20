# Anchor-Attention Encoding for Fully Inductive Temporal Knowledge Graph Reasoning

Code for the KARMA family of inductive temporal KGR models.

**Paper:**
> Bhullar, A. & Kobti, Z. (2026). *Anchor-Attention Encoding for Fully Inductive Temporal Knowledge Graph Reasoning*. ASONAM 2026 (under review).



## Models
| Model | Key Idea | Best U-MRR |
|---|---|---|
| KARMA-base | Anchor-attention inductive encoder | 0.111 (ICEWS18) |
| KARMA-T | Transductive-inductive hybrid gate | **0.784** (WIKI-clean) |
| KARMA-A | Adaptive cluster anchors | **0.120** (ICEWS18) |
| KARMA-C | Contrastive anchor pretraining | 0.577 (WIKI-clean) |
| KARMA-F | Future-aware encoding | 0.772 (WIKI-clean) |
| KARMA-R | Relational path anchors | 0.769 (WIKI-clean) |
| KARMA-RL | Soft temporal rule anchors | 0.488 (WIKI-clean) |
| KARMA+ | Full stack combination | 0.034 (ICEWS14) |

## Novel Metrics

- **U-MRR**: MRR on unseen entities only
- **TGG**: Transductive-to-Generative Gap (seen MRR minus U-MRR)

## Installation

```bash
pip install torch geoopt numpy scipy
```

## Usage

```bash
cd models/KARMA-T
python train.py --dataset WIKI-clean --gpu 0 --epochs 500 --patience 20

cd models/KARMA-A
python train.py --dataset ICEWS18-renet --gpu 0 --epochs 500 --patience 20
```

## Results 

### Standard Benchmark MRR

| Dataset | Model | Seen MRR | Seen H@1 | Seen H@10 | Unseen MRR | Unseen H@1 | Unseen H@10 | TGG↓ |
|---------|-------|----------|----------|-----------|------------|------------|-------------|------|
| **Future Prediction (t > T_train)** | | | | | | | | |
| WIKI-clean | TTransE† | 0.584 | — | — | 0.001 | 0.000 | 0.000 | 0.583 |
| | KARMA-base | 0.632 | 0.613 | 0.661 | 0.018 | 0.003 | 0.042 | 0.615 |
| | KARMA-C | 0.624 | 0.576 | 0.686 | 0.577 | 0.482 | 0.723 | 0.060 |
| | KARMA-RL | 0.617 | 0.562 | 0.687 | 0.488 | 0.401 | 0.632 | 0.150 |
| | KARMA-R | 0.663 | 0.628 | 0.708 | 0.769 | 0.671 | 0.912 | -0.131 |
| | KARMA-F | 0.636 | 0.613 | 0.670 | 0.772 | 0.674 | 0.901 | -0.136 |
| | **KARMA-T** | **0.637** | **0.611** | **0.672** | **0.784** | **0.681** | **0.923** | **-0.148** |
| ICEWS18 | TTransE† | 0.084 | — | 0.220 | 0.000 | 0.000 | 0.000 | 0.084 |
| | RE-GCN† | 0.373 | — | 0.584 | 0.000 | 0.000 | 0.000 | 0.373 |
| | KARMA-base | 0.180 | 0.094 | 0.358 | 0.111 | 0.077 | 0.172 | 0.068 |
| | KARMA-T | 0.212 | 0.116 | 0.408 | 0.094 | 0.054 | 0.154 | 0.118 |
| | KARMA-F | 0.219 | 0.122 | 0.413 | 0.101 | 0.070 | 0.159 | 0.118 |
| | **KARMA-A** | **0.209** | **0.111** | **0.405** | **0.120** | **0.093** | **0.176** | **0.089** |
| YAGO15k | KARMA-base | 0.617 | 0.553 | 0.729 | 0.027 | 0.000 | 0.052 | 0.590 |
| | KARMA-RL | 0.614 | 0.556 | 0.716 | 0.085 | 0.061 | 0.128 | 0.556 |
| | **KARMA-T** | **0.639** | **0.576** | **0.726** | **0.127** | **0.093** | **0.181** | **0.512** |
| ICEWS14 | RE-GCN† | 0.458 | — | 0.632 | 0.000 | 0.000 | 0.000 | 0.458 |
| | KARMA-base | 0.107 | 0.047 | 0.223 | 0.002 | 0.000 | 0.000 | 0.105 |
| | KARMA-T | 0.183 | 0.094 | 0.360 | 0.028 | 0.006 | 0.059 | 0.155 |
| | **KARMA-RL** | **0.174** | **0.086** | **0.360** | **0.034** | **0.012** | **0.074** | **0.143** |
| **Missing Fact Completion (t ≤ T_train)** | | | | | | | | |
| WIKIDATA12k | TNTComplEx† | 0.357 | — | 0.490 | 0.000 | 0.000 | 0.000 | 0.357 |
| | KARMA-base | 0.223 | 0.143 | 0.385 | 0.004 | 0.000 | 0.000 | 0.219 |
| | **KARMA-C** | **0.252** | **0.148** | **0.499** | 0.000 | 0.000 | 0.000 | — |
| YAGO11k | TNTComplEx† | 0.341 | — | 0.490 | 0.000 | 0.000 | 0.000 | 0.341 |
| | KARMA-base | 0.100 | 0.047 | 0.195 | 0.003 | 0.000 | 0.000 | 0.097 |
| | **KARMA-C** | **0.183** | **0.110** | **0.346** | 0.000 | 0.000 | 0.000 | — |
| **Best KARMA-X** | | | | | **0.784** | **0.681** | **0.923** | **-0.148** |
| Best transductive baseline | | | | | 0.001 | 0.000 | 0.000 | — |
| **Improvement** | | | | | **784×** | — | — | — |

† Transductive: Unseen-MRR ≡ 0 by definition. TGG = Seen MRR − Unseen MRR; negative = unseen predicted more accurately than seen.

### Inductive Evaluation (U-MRR)

| Dataset | Model | Seen MRR | Seen H@1 | Seen H@10 | Unseen MRR | Unseen H@1 | Unseen H@10 | TGG↓ |
|---------|-------|----------|----------|-----------|------------|------------|-------------|------|
| **Future Prediction (t > T_train)** | | | | | | | | |
| WIKI-clean | TTransE† | 0.584 | — | — | 0.001 | 0.000 | 0.000 | 0.583 |
| | KARMA-base | 0.632 | 0.613 | 0.661 | 0.018 | 0.003 | 0.042 | 0.615 |
| | KARMA-C | 0.624 | 0.576 | 0.686 | 0.577 | 0.482 | 0.723 | 0.060 |
| | KARMA-RL | 0.617 | 0.562 | 0.687 | 0.488 | 0.401 | 0.632 | 0.150 |
| | KARMA-R | 0.663 | 0.628 | 0.708 | 0.769 | 0.671 | 0.912 | -0.131 |
| | KARMA-F | 0.636 | 0.613 | 0.670 | 0.772 | 0.674 | 0.901 | -0.136 |
| | **KARMA-T** | **0.637** | **0.611** | **0.672** | **0.784** | **0.681** | **0.923** | **-0.148** |
| ICEWS18 | TTransE† | 0.084 | — | 0.220 | 0.000 | 0.000 | 0.000 | 0.084 |
| | RE-GCN† | 0.373 | — | 0.584 | 0.000 | 0.000 | 0.000 | 0.373 |
| | KARMA-base | 0.180 | 0.094 | 0.358 | 0.111 | 0.077 | 0.172 | 0.068 |
| | KARMA-T | 0.212 | 0.116 | 0.408 | 0.094 | 0.054 | 0.154 | 0.118 |
| | KARMA-F | 0.219 | 0.122 | 0.413 | 0.101 | 0.070 | 0.159 | 0.118 |
| | **KARMA-A** | **0.209** | **0.111** | **0.405** | **0.120** | **0.093** | **0.176** | **0.089** |
| YAGO15k | KARMA-base | 0.617 | 0.553 | 0.729 | 0.027 | 0.000 | 0.052 | 0.590 |
| | KARMA-RL | 0.614 | 0.556 | 0.716 | 0.085 | 0.061 | 0.128 | 0.556 |
| | **KARMA-T** | **0.639** | **0.576** | **0.726** | **0.127** | **0.093** | **0.181** | **0.512** |
| ICEWS14 | RE-GCN† | 0.458 | — | 0.632 | 0.000 | 0.000 | 0.000 | 0.458 |
| | KARMA-base | 0.107 | 0.047 | 0.223 | 0.002 | 0.000 | 0.000 | 0.105 |
| | KARMA-T | 0.183 | 0.094 | 0.360 | 0.028 | 0.006 | 0.059 | 0.155 |
| | **KARMA-RL** | **0.174** | **0.086** | **0.360** | **0.034** | **0.012** | **0.074** | **0.143** |
| **Missing Fact Completion (t ≤ T_train)** | | | | | | | | |
| WIKIDATA12k | TNTComplEx† | 0.357 | — | 0.490 | 0.000 | 0.000 | 0.000 | 0.357 |
| | KARMA-base | 0.223 | 0.143 | 0.385 | 0.004 | 0.000 | 0.000 | 0.219 |
| | **KARMA-C** | **0.252** | **0.148** | **0.499** | 0.000 | 0.000 | 0.000 | — |
| YAGO11k | TNTComplEx† | 0.341 | — | 0.490 | 0.000 | 0.000 | 0.000 | 0.341 |
| | KARMA-base | 0.100 | 0.047 | 0.195 | 0.003 | 0.000 | 0.000 | 0.097 |
| | **KARMA-C** | **0.183** | **0.110** | **0.346** | 0.000 | 0.000 | 0.000 | — |
| **Best KARMA-X** | | | | | **0.784** | **0.681** | **0.923** | **-0.148** |
| Best transductive baseline | | | | | 0.001 | 0.000 | 0.000 | — |
| **Improvement** | | | | | **784×** | — | — | — |

† Transductive: Unseen-MRR ≡ 0 by definition. TGG = Seen MRR − Unseen MRR; negative = unseen predicted more accurately than seen.

### New Inductive Benchmark Results

| Dataset | Best Model | MRR | Unseen MRR | Unseen H@10 |
|---------|------------|-----|------------|-------------|
| **SciInductTKG** (flat/chain, 4 relations) | | | | |
| Emerge | KARMA-T | 0.453 | **0.740** | 0.746 |
| Rare | KARMA-A | 0.025 | 0.006 | 0.017 |
| Cold | KARMA-A | 0.093 | 0.094 | 0.117 |
| **PolitInductTKG** (heterogeneous, 260 relations) | | | | |
| Emerge | KARMA-A | 0.135 | **0.411** | 0.523 |
| Cold | KARMA-T | 0.135 | 0.185 | 0.342 |
| **WikiInductTKG** (sparse, 3 relations — open challenge) | | | | |
| Emerge | KARMA-C | 0.449 | 0.000 | 0.000 |
| Cold | KARMA-base | 0.110 | 0.009 | 0.000 |

### Model Selection Guide

| Graph Property | Best Variant | Reason |
|----------------|-------------|--------|
| Rel/Ent < 3, rich history | KARMA-T | Hybrid gate maximizes transductive strength |
| Rel/Ent > 5, heterogeneous | KARMA-A | Clusters capture entity-type patterns |
| Future prediction, periodic events | KARMA-F | Predicts relation periodicity |
| Missing facts, sparse graph | KARMA-C | Contrastive smoothness regularizes embeddings |
| Rule-amenable, political | KARMA-RL | Rules generalize to unseen via patterns |
| Unknown structure | KARMA-T | Best average performance |

## Citation

```bibtex
@inproceedings{bhullar2026karmax,
  author    = {Bhullar, Amangel and Kobti, Ziad},
  title     = {Anchor-Attention Encoding for Fully Inductive
               Temporal Knowledge Graph Reasoning},
  booktitle = {Proceedings of ASONAM 2026},
  year      = {2026},
  note      = {Under review}
}
```

## Contact

Amangel Bhullar — bhull113@uwindsor.ca
Supervisor: Dr. Ziad Kobti — kobti@uwindsor.ca
School of Computer Science, University of Windsor
