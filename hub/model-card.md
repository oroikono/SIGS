---
library_name: sigs
license: mit
datasets:
  - oroikono/sigs-symbolic-pde-corpus
arxiv: 2502.01476
tags:
  - transformers
  - pytorch
  - scientific-machine-learning
  - neuro-symbolic-ai
  - symbolic-regression
  - pde
---

# SIGS ICML 2026 Grammar-VAE

This repository contains the grammar variational autoencoder used by SIGS,
**Neuro-Symbolic AI for Analytical Solutions of Differential Equations**
(Oikonomou et al., ICML 2026).

## Model description

SIGS embeds syntactically valid mathematical expressions into a continuous
latent space and decodes latent vectors under a context-free grammar. The wider
SIGS pipeline searches that space for closed-form candidates and then refines
their constants against differential-equation residuals.

This is a grammar model, not a natural-language model. Its inputs are one-hot
grammar-production sequences with shape `[batch, productions, sequence_length]`.

## Loading

Install SIGS with its Hugging Face dependencies, then load the checkpoint:

```python
from sigs.huggingface import SIGSModel

model = SIGSModel.from_pretrained("oroikono/sigs-icml-2026")
```

The associated production-sequence corpus can be represented with Hugging Face
Datasets, and `examples/train_with_accelerate.py` provides a distributed training
entry point.

## Intended use

- Research on grammar-guided symbolic generation.
- Reproducing the symbolic proposal component of SIGS.
- Closed-form differential-equation solution discovery with the wider SIGS
  search and residual-refinement pipeline.

## Limitations

- Generated candidates are restricted to the published grammar and maximum
  production-sequence length.
- The model alone does not certify that an expression solves a differential
  equation; residual, initial-condition, and boundary-condition checks remain
  necessary.
- Performance outside the PDE families and expression corpus reported in the
  paper has not been established.

## Citation

```bibtex
@misc{oikonomou2026neurosymbolic,
  title={Neuro-Symbolic AI for Analytical Solutions of Differential Equations},
  author={Oikonomou, Orestis and Lingsch, Levi and Grund, Dana and Mishra, Siddhartha and Kissas, Georgios},
  year={2026},
  eprint={2502.01476},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
