# Data and Model Files

Two large files are required to run Stage I discovery. They are tracked via
Git LFS and will be downloaded automatically when you clone the repository
with LFS support:

```bash
git lfs install
git clone https://github.com/oroikono/SIGS
```

If Git LFS is not available, download the files manually and place them in
this `data/` directory:

| File | Size | Description |
|------|------|-------------|
| `data/clusters.pkl` | ~4.4 MB | Pre-computed latent cluster database (23 000 vectors across 6 MathClass categories) |
| `data/model.ckpt` | ~50 MB | Pre-trained Grammar-VAE weights (epoch 107, val ELBO = 27.65) |

The training dataset used to generate the cluster database is stored in
`data/expressions.h5` (~368 MB) and is required only if you want to
retrain the Grammar-VAE from scratch.

## Configs

The model configuration is in `configs/config.yaml`.
Update `LODE_CONFIG`, `LODE_CKPT` environment variables to override paths:

```bash
export LODE_CKPT=/path/to/your/checkpoint.ckpt
export LODE_CONFIG=/path/to/config.yaml
python notebooks/demo_shallow_water.ipynb
```
