---
license: mit
tags:
  - symbolic-regression
  - scientific-machine-learning
  - grammar
  - pde
pretty_name: SIGS Grammar Production Corpus
---

# SIGS Grammar Production Corpus

This dataset contains the one-hot grammar-production sequences used to train the
Grammar-VAE in **SIGS: Neuro-Symbolic AI for Analytical Solutions of
Differential Equations** (Oikonomou et al., ICML 2026).

## Structure

Each row contains:

- `inputs`: a float32 tensor shaped `[grammar productions, sequence length]`;
- `labels`: the corresponding integer production indices shaped
  `[sequence length]`.

The deterministic default split uses seed 42 with 70% training, 20% validation,
and 10% test examples.

## Intended use

- Training or evaluating grammar-guided symbolic generative models.
- Reproducing the SIGS Grammar-VAE representation-learning stage.
- Studying structured representations for symbolic differential equations.

## Limitations

The corpus reflects the published SIGS grammar and expression-generation
procedure. It should not be treated as representative of all valid mathematical
expressions or differential equations. Models trained on it inherit its grammar,
maximum-length, and sampling biases.

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
