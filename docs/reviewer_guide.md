# Reviewer guide

For a fast technical review, use this order:

1. Run `python scripts/smoke_test.py` and `pytest -q`.
2. Read [`algorithm.md`](algorithm.md) for the two-stage discovery procedure.
3. Inspect `src/sigs/model.py`, `grammar.py`, and `sampler.py` for the core model
   and constrained decoding path.
4. Inspect `src/sigs/huggingface.py` for the boundary between the native research
   model and standard Transformers serialization/training interfaces.
5. Read [`reproducibility.md`](reproducibility.md) before attempting paper-scale
   experiments that require the LFS artifacts.

The public API intentionally separates:

- the symbolic grammar and native PyTorch Grammar-VAE;
- optional Lightning training;
- optional Stage I clustering/search;
- optional JAX/Optax coefficient refinement;
- optional Hugging Face Transformers, Datasets, Accelerate, and Hub
  interoperability.

This keeps a basic installation reviewable while preserving the original
paper-scale workflow behind explicit dependency extras.
