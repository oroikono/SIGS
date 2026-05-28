# SIGS: Neuro-Symbolic AI for Analytical Solutions of PDEs

[![Paper](https://img.shields.io/badge/arXiv-2502.01476-b31b1b.svg)](https://arxiv.org/abs/2502.01476)
[![Video](https://img.shields.io/badge/YouTube-Explanation-red?logo=youtube)](https://www.youtube.com/watch?v=a9MMvKVGhuQ)

> **Watch the 5-minute paper explanation:** [youtube.com/watch?v=a9MMvKVGhuQ](https://www.youtube.com/watch?v=a9MMvKVGhuQ)

SIGS is a framework for discovering closed-form analytical solutions to systems
of partial differential equations. Rather than fitting neural surrogates, it
searches a structured symbolic space defined by a context-free grammar, using a
Grammar Variational Autoencoder (Grammar-VAE) to map that space into a continuous
latent manifold and k-means clustering to partition it into semantically coherent
expression families. The best structural match is then refined to numerical precision
by gradient-based parameter optimization via JAX autodiff.

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
- All parameters are optimized simultaneously with Adam (Optax) for up to 5 000 iterations.

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

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- JAX / Jaxlib >= 0.4.20
- Optax >= 0.1.7
- symengine >= 0.11.0
- sympy >= 1.12
- scikit-learn >= 1.2
- nltk >= 3.8
- numpy, scipy, h5py, pyyaml, tqdm, matplotlib

```bash
pip install -r requirements.txt
```

Or use the provided conda environment:

```bash
conda env create -f environment.yml
conda activate sigs
```

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
    └── algorithm.md            Detailed two-stage algorithm description
```

## Data and Pre-trained Models

The Grammar-VAE checkpoint (`data/model.ckpt`) and the pre-computed cluster
database (`data/clusters.pkl`) are required to run Stage I.
See `data/DOWNLOAD.md` for download instructions.

The cluster database contains ~23 000 latent vectors partitioned into six
`MathClass` categories. The Grammar-VAE was trained on a dataset of ~50 000
symbolic expressions generated from the CFG grammar (`data/expressions.h5`,
required only for retraining).

## Quick Start

```python
import sys
sys.path.insert(0, "src")

import torch
from sigs.sampler import FlexibleVectorSampler
from sigs.utils import MathClass, ModelUtils

# Load pre-trained model
config = ModelUtils.load_config("configs/config.yaml")
model  = ModelUtils.load_checkpoint("data/model.ckpt", config).eval()

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

If you use SIGS in your research, please cite:

```bibtex
@article{oikonomou2025sigs,
  title   = {Neuro-Symbolic {AI} for Analytical Solutions of Differential Equations},
  author  = {Oikonomou, Orestis and Lingsch, Levi and Grund, Dana and
             Mishra, Siddhartha and Kissas, Georgios},
  journal = {arXiv preprint arXiv:2502.01476},
  year    = {2025},
  url     = {https://arxiv.org/abs/2502.01476}
}
```

## License

MIT License. See `LICENSE` for details.
