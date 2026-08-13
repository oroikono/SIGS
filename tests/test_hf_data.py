"""Tests for the Hugging Face Datasets bridge."""

from __future__ import annotations

import h5py
import numpy as np
import torch

from sigs.hf_data import load_hdf5_dataset


def test_load_hdf5_dataset_builds_deterministic_typed_splits(tmp_path):
    path = tmp_path / "expressions.h5"
    labels = np.arange(10 * 4, dtype=np.int64).reshape(10, 4) % 3
    data = np.eye(3, dtype=np.float32)[labels][:, None, :, :]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)

    cache = tmp_path / "cache"
    first = load_hdf5_dataset(
        path,
        validation_split=0.2,
        test_split=0.1,
        seed=7,
        cache_dir=cache,
    )
    second = load_hdf5_dataset(
        path,
        validation_split=0.2,
        test_split=0.1,
        seed=7,
        cache_dir=cache,
    )

    assert {name: len(split) for name, split in first.items()} == {
        "train": 7,
        "validation": 2,
        "test": 1,
    }
    assert first["train"][0] == second["train"][0]
    example = first["train"][0]
    assert np.asarray(example["inputs"]).shape == (3, 4)
    assert np.asarray(example["labels"]).shape == (4,)
    assert first["train"].features["inputs"].dtype == "float32"
    torch_example = first["train"].with_format("torch")[0]
    assert torch_example["inputs"].dtype == torch.float32
    assert torch_example["labels"].dtype == torch.int64


def test_load_hdf5_dataset_rejects_invalid_splits(tmp_path):
    path = tmp_path / "expressions.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((2, 1, 4, 3), dtype=np.float32))

    try:
        load_hdf5_dataset(
            path,
            validation_split=0.7,
            test_split=0.3,
            cache_dir=tmp_path / "cache",
        )
    except ValueError as error:
        assert "less than 1" in str(error)
    else:
        raise AssertionError("invalid split fractions should fail")
