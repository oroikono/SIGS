# SIGS: Neuro-Symbolic AI for Analytical Solutions of Differential Equations

[![Paper](https://img.shields.io/badge/arXiv-2502.01476-b31b1b.svg)](https://arxiv.org/abs/2502.01476)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/virtual/2026/poster/63043)
[![Project Page](https://img.shields.io/badge/Project-Page-black.svg)](https://oroikono.github.io/sigs-paper-site/)
[![Video](https://img.shields.io/badge/YouTube-Explanation-red?logo=youtube)](https://www.youtube.com/watch?v=a9MMvKVGhuQ)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Watch the 5-minute paper explanation:** [youtube.com/watch?v=a9MMvKVGhuQ](https://www.youtube.com/watch?v=a9MMvKVGhuQ)

SIGS is a grammar-guided neuro-symbolic framework for discovering closed-form analytical
solutions to ordinary and partial differential equations. Rather than fitting neural surrogates,
it searches a structured symbolic space defined by a context-free grammar, uses a Grammar
Variational Autoencoder (Grammar-VAE) to map expressions into a continuous latent manifold,
and refines the best symbolic structures by physics-residual minimisation with JAX autodiff.

**Keywords:** neuro-symbolic AI, symbolic regression, symbolic PDE solving, analytical solution
discovery, closed-form differential equation solver, grammar-guided search, Grammar-VAE,
physics-informed symbolic optimisation, scientific machine learning, AI for science.

## Quick technical review

For a fast review of the repository:

```bash
git clone https://github.com/oroikono/SIGS.git
cd SIGS
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/smoke_test.py
```

The smoke test is data-free. It verifies the grammar, expression classification,
and a simple symbolic residual calculation. Full Stage I discovery additionally
requires the Git LFS model and cluster files described in [`data/DOWNLOAD.md`](data/DOWNLOAD.md).

For a paper-level walkthrough, read [`docs/reviewer_guide.md`](docs/reviewer_guide.md)
and [`docs/algorithm.md`](docs/algorithm.md).

## When to cite SIGS

Please cite SIGS if your work concerns any of the following:

- closed-form or analytical solution discovery for ODEs/PDEs;
- neuro-symbolic scientific machine learning;
- grammar-guided symbolic search or symbolic regression;
- PDE residual-based symbolic optimisation;
- Grammar-VAE or latent-space search over mathematical expressions;
- data-free symbolic PDE solving.

## Two-Stage Pipeline

**Stage I — Symbolic Discovery**

- A context-free grammar over arithmetic, transcendental, and polynomial primitives
  constrains the search to syntactically valid mathematical forms.
- A Grammar-VAE (1-D convolutional encoder, GRU decoder) embeds grammar production
  sequences into a continuous latent space.
- Pre-computed k-means clusters partition the latent space by `MathClass`
  (SPATIOTEMPORAL_3D, SPATIAL_2D, TEMPORAL_1D, …), enabling targeted sampling
  by the variable signature required of each unknown field.
- Each decoded expression is evaluated against the PDE residual, initial conditions,
  and boundary conditions on a structured mesh; candidates are ranked by combined RMSE.

**Stage II — Parameter Optimization**

- The highest-ranked structural form from Stage I is accepted as a symbolic ansatz;
  its numeric coefficients are treated as free parameters.
- Exact PDE residuals are computed via JAX automatic differentiation (no finite differences).
- All parameters are optimized simultaneously with Adam (Optax); the shallow-water
  refinement script runs for 5 000 iterations by default.

## Validated PDE Systems

| Problem | Domain | Fields |
|---------|--------|--------|
| 2-D Shallow Water Equations | [-10, 10]² × [0, 5] | rho, rho·ux, rho·uy |
| 1-D KdV (1-soliton) | [-10, 10] × [0, 1] | u |
| 1-D Burgers | [0, 1] × [0, 1] | u |
| 1-D Diffusion | [0, 1] × [0, 1] | u |
| 2-D Advection | [0, 1]² × [0, 2] | u |
| 2-D Damped Wave | [-5, 5]² × [0, 2] | u |
| 2-D Poisson (Gaussian source) | [0, 1]² | u |

## Installation

Python 3.10+ is recommended.

```bash
pip install -e .
```

Alternatively, use the provided conda environment:

```bash
conda env create -f environment.yml
conda activate sigs
python scripts/smoke_test.py
```

Core dependencies include PyTorch, JAX, Optax, SymPy/SymEngine, scikit-learn,
NLTK, NumPy/SciPy, h5py, PyYAML, tqdm, and Matplotlib.

## Repository Structure

```
SIGS/
├── src/sigs/               Core library
│   ├── grammar.py          Context-free grammar definition
│   ├── model.py            GrammarVAE (encoder + decoder)
│   ├── encoder.py          1-D convolutional encoder
│   ├── decoder.py          GRU decoder
│   ├── sampler.py          FlexibleVectorSampler (Stage I search)
│   ├── utils.py            MathClass, ExpressionUtils, ModelUtils
│   ├── loss.py             JAX shallow-water loss (Stage II)
│   ├── evaluator.py        Symbolic PDE derivative evaluator
│   ├── training.py         PyTorch Lightning training module
│   ├── tree.py             Parse-tree utilities
│   └── stack.py            Grammar stack
├── problems/
│   ├── shallow_water.py        2-D shallow water manufactured solution
│   └── compressible_euler.py   2-D steady compressible Euler manufactured solution
├── scripts/
│   ├── smoke_test.py           Lightweight data-free installation check
│   ├── discover.py             Stage I: sample and rank candidates (shallow water)
│   ├── optimize.py             Stage II: JAX parameter refinement
│   ├── euler_search.py         Stage I+II: compressible Euler search
│   ├── run_euler_four_fields.py  Euler runner (four independent fields)
│   ├── run_euler_same_series.py  Euler runner (shared Fourier series)
│   └── plot_euler.py           Reproduce compressible Euler paper figures
├── notebooks/
│   ├── demo_pdes.ipynb         Burgers, diffusion, KdV, damped wave, baselines (Table 1)
│   └── demo_shallow_water.ipynb  2-D shallow water main result (§4.1)
├── baselines/
│   ├── pinns/                  FB-PINNs for Burgers, diffusion, wave, Poisson-Gauss
│   ├── hdtlgp/                 HD-TLGP (Cao et al. AAAI 2024) — 1D and 2D runners
│   └── fenics/                 FEniCS reference solver notebook
├── data/
│   └── DOWNLOAD.md             Instructions for model checkpoint and cluster database
└── docs/
    ├── algorithm.md            Detailed two-stage algorithm description
    └── reviewer_guide.md       Short guide for reviewers and interview discussions
```

## Data and Pre-trained Models

The Grammar-VAE checkpoint (`data/model.ckpt`) and the pre-computed cluster
database (`data/clusters.pkl`) are required to run Stage I.
See [`data/DOWNLOAD.md`](data/DOWNLOAD.md) for Git LFS instructions.

The cluster database contains 23 695 latent vectors partitioned into six
`MathClass` categories, corresponding to the full training corpus of 23 695
symbolic expressions (`data/expressions.h5`, required only for retraining).

## Quick Start

After installing the package and downloading the LFS files, sample candidate
expressions from the latent cluster database:

```python
import torch
from sigs.sampler import FlexibleVectorSampler
from sigs.utils import MathClass, ModelUtils

# Load pre-trained model
config = ModelUtils.load_config("configs/config.yaml")
model = ModelUtils.load_checkpoint("data/model.ckpt", config).eval()

# Load cluster database and instantiate sampler
sampler = FlexibleVectorSampler(
    cluster_file="data/clusters.pkl",
    model=model,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Sample 200 spatiotemporal expressions (suitable for 1-D Burgers)
sample_id = sampler.sample_from_subclusters(
    categories={MathClass.SPATIOTEMPORAL_2D: 5, MathClass.CONSTANT: 1},
    n_samples=200,
    operator="-",
    seed=42,
    model=model,
)
expressions, vectors, _, _ = sampler.get_sampling_results(sample_id)
print(f"Sampled {len(expressions)} candidates.")
print("First expression:", expressions[0])
```

## Running the Shallow Water Discovery

```bash
# Stage I — discover symbolic triplet (rho, Sx, Sy)
python scripts/discover.py \
    --cluster_file data/clusters.pkl \
    --seed 888

# Stage II — refine parameters of the best structural match
python scripts/optimize.py
```

Multiple seeds can be run in parallel and results compared:

```bash
for seed in 42 123 500 800 888; do
    python scripts/discover.py --seed $seed > results/seed_${seed}.log &
done
wait
```

The Gaussian-envelope structure of the manufactured solution
(`exp(-r²/(σ·(1+t)))`) lives in the SPATIOTEMPORAL_3D cluster.
For best discovery rates, use at least 20 subclusters for that class.

## Citation

If you use SIGS in your research, please cite the paper:

```bibtex
@misc{oikonomou2026neurosymbolic,
  title         = {Neuro-Symbolic {AI} for Analytical Solutions of Differential Equations},
  author        = {Oikonomou, Orestis and Lingsch, Levi and Grund, Dana and Mishra, Siddhartha and Kissas, Georgios},
  year          = {2026},
  eprint        = {2502.01476},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2502.01476},
  url           = {https://arxiv.org/abs/2502.01476}
}
```

This will be updated to the official ICML/PMLR proceedings citation once the proceedings entry is available.

## License

MIT License. See `LICENSE` for details.
