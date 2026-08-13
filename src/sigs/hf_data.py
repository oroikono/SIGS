"""Hugging Face Datasets integration for SIGS production sequences."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import h5py
import numpy as np
from datasets import Array2D, Dataset, DatasetDict, Features, Value
from datasets import Sequence as HFSequence


def _read_shape(path: str | Path) -> tuple[int, int, int]:
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError("HDF5 file must contain a 'data' dataset")
        shape = handle["data"].shape

    if len(shape) == 4 and shape[1] == 1:
        count, _, sequence_length, vocab_size = shape
    elif len(shape) == 3:
        count, sequence_length, vocab_size = shape
    else:
        raise ValueError(
            "data must have shape [examples, 1, length, vocabulary] or "
            f"[examples, length, vocabulary], got {shape}"
        )
    return int(count), int(sequence_length), int(vocab_size)


def _generate_examples(path: str, indices: Sequence[int]) -> Iterator[dict]:
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        for index in indices:
            production_sequence = np.asarray(data[index], dtype=np.float32)
            if production_sequence.ndim == 3:
                production_sequence = production_sequence.squeeze(0)
            labels = production_sequence.argmax(axis=-1).astype(np.int64)
            yield {
                "inputs": production_sequence.T,
                "labels": labels,
            }


def load_hdf5_dataset(
    path: str | Path,
    *,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> DatasetDict:
    """Load the SIGS HDF5 corpus as deterministic Arrow-backed splits.

    Each example contains ``inputs`` shaped ``[vocabulary, sequence_length]``
    and integer ``labels`` shaped ``[sequence_length]``. Data is generated into
    Arrow incrementally, so the full HDF5 tensor is never loaded into RAM.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")
    if validation_split < 0 or test_split < 0:
        raise ValueError("validation_split and test_split must be non-negative")
    if validation_split + test_split >= 1:
        raise ValueError("validation_split + test_split must be less than 1")

    count, sequence_length, vocab_size = _read_shape(path)
    shuffled = np.random.default_rng(seed).permutation(count).tolist()
    test_count = int(test_split * count)
    validation_count = int(validation_split * count)
    split_indices = {
        "train": shuffled[test_count + validation_count :],
        "validation": shuffled[test_count : test_count + validation_count],
        "test": shuffled[:test_count],
    }
    features = Features(
        {
            "inputs": Array2D(
                shape=(vocab_size, sequence_length),
                dtype="float32",
            ),
            "labels": HFSequence(Value("int64"), length=sequence_length),
        }
    )

    return DatasetDict(
        {
            split: Dataset.from_generator(
                _generate_examples,
                features=features,
                gen_kwargs={"path": str(path), "indices": indices},
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
            for split, indices in split_indices.items()
        }
    )
