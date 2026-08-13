"""Shared lightweight fixtures for SIGS tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def tiny_native_config() -> dict:
    return {
        "model": {
            "encoder": {
                "hidden_size": 8,
                "conv_sizes": [4],
                "kernel_sizes": [2],
                "use_batch_norm": False,
                "dropout": 0.0,
            },
            "decoder": {
                "hidden_size": 8,
                "num_layers": 1,
                "rnn_type": "gru",
                "dropout": 0.0,
                "mode": "positional",
            },
            "shared": {"z_dim": 3, "max_length": 8, "output_size": 7},
        },
        "training": {"tightness": 1.0, "decoder_samples": 2},
    }
