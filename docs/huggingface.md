# Hugging Face integration

SIGS exposes tested integrations for Transformers, Datasets, Accelerate, and the
Hub without changing the semantics of the native grammar variational
autoencoder. Inputs remain one-hot grammar production sequences shaped
`[batch, productions, sequence_length]`; SIGS is not presented as a text
language model.

## Install

```bash
pip install -e ".[huggingface]"
```

Install `.[train,huggingface]` when converting the original Lightning training
environment or retraining the model.

## Native configuration conversion

```python
import yaml

from sigs.huggingface import SIGSConfig

with open("configs/config.yaml") as handle:
    native_config = yaml.safe_load(handle)

config = SIGSConfig.from_native_config(native_config)
config.save_pretrained("exported/sigs-grammar-vae")
```

`SIGSConfig.to_native_config()` performs the inverse conversion for the model
architecture and inference-time sampling fields.

## Convert the paper checkpoint

```bash
python examples/export_to_huggingface.py \
  --checkpoint data/model.ckpt \
  --config configs/config.yaml \
  --output-dir exported/sigs-grammar-vae
```

The command strips the Lightning `model.` prefix, validates the native state
dictionary strictly, and writes `config.json` plus safe model weights. It does
not upload anything unless `--repo-id` is supplied.

## Load and run

```python
import torch

from sigs.huggingface import SIGSModel

model = SIGSModel.from_pretrained("oroikono/sigs-icml-2026")
inputs = torch.zeros(1, model.config.vocab_size, model.config.sequence_length)
outputs = model(inputs)

print(outputs.logits.shape)
print(outputs.latent_mean.shape)
print(outputs.latent_logvar.shape)
```

Supplying integer `labels` with shape `[batch, sequence_length]` also returns an
ELBO-style scalar loss. This makes the wrapper usable with a custom collator and
the Transformers `Trainer` while keeping the paper's native trainer available.

## AutoModel in an installed SIGS environment

SIGS is an external architecture, so register it once in the Python process:

```python
from transformers import AutoModel
from sigs.huggingface import register_with_transformers

register_with_transformers()
model = AutoModel.from_pretrained("oroikono/sigs-icml-2026")
```

Direct `SIGSModel.from_pretrained(...)` does not require registration. A future
upstream Transformers integration would remove this local-registration step, but
that should only be proposed after the public checkpoint and API are stable.

## Datasets

Convert the original HDF5 corpus into deterministic Arrow-backed train,
validation, and test splits:

```python
from sigs.hf_data import load_hdf5_dataset

dataset = load_hdf5_dataset(
    "data/expressions.h5",
    validation_split=0.2,
    test_split=0.1,
    seed=42,
)
print(dataset)
```

Each row contains float32 `inputs` shaped `[productions, sequence_length]` and
integer `labels` shaped `[sequence_length]`. Conversion uses
`Dataset.from_generator`, so examples are written incrementally to Arrow rather
than first loading the complete HDF5 tensor into memory.

To save locally or publish a Dataset Viewer-compatible repository:

```bash
python examples/export_dataset_to_huggingface.py \
  --data data/expressions.h5 \
  --output-dir exported/sigs-corpus \
  --repo-id oroikono/sigs-symbolic-pde-corpus
```

## Accelerate

`examples/train_with_accelerate.py` is a compact, standard distributed training
loop over the Transformers model and Datasets corpus:

```bash
accelerate config
accelerate launch examples/train_with_accelerate.py \
  --data data/expressions.h5 \
  --config configs/config.yaml \
  --output-dir exported/sigs-grammar-vae \
  --mixed-precision bf16
```

The example uses `Accelerator.prepare`, gradient accumulation, distributed metric
gathering, and process-safe `save_pretrained`. It is intentionally an
interoperability example; the paper's Lightning trainer remains the source of
truth for its KL schedule and geometric regularizers.

## Publish to the Hub

Authenticate with `hf auth login`, review the generated model card, and run:

```bash
python examples/export_to_huggingface.py \
  --checkpoint data/model.ckpt \
  --config configs/config.yaml \
  --output-dir exported/sigs-grammar-vae \
  --repo-id oroikono/sigs-icml-2026
```

The repository should contain the generated weights/config and a completed model
card based on [`hub/model-card.md`](../hub/model-card.md). Include the ICML paper,
training corpus, intended scientific use, limitations, evaluation results,
license, and exact SIGS package version.
