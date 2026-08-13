"""Unit tests for native model semantics."""

from __future__ import annotations

import pytest
import torch

from sigs.decoder import Decoder
from sigs.model import GrammarVAE


def test_forward_has_stable_shapes_and_decoder_sample_count(tiny_native_config):
    model = GrammarVAE(tiny_native_config)
    inputs = torch.randn(2, 7, 8)

    logits, mean, logvar = model(inputs)

    assert logits.shape == (2, 2, 8, 7)
    assert mean.shape == logvar.shape == (2, 3)


def test_sampling_with_local_generator_does_not_reset_global_rng(tiny_native_config):
    model = GrammarVAE(tiny_native_config)
    mean = torch.zeros(2, 3)
    logvar = torch.zeros_like(mean)

    torch.manual_seed(11)
    expected_next = torch.rand(1)
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(99)
    model.sample(mean, logvar, generator=generator)
    actual_next = torch.rand(1)

    assert torch.equal(actual_next, expected_next)


def test_decoder_accepts_one_or_many_samples(tiny_native_config):
    decoder_config = tiny_native_config["model"]["decoder"]
    shared = tiny_native_config["model"]["shared"]
    decoder = Decoder(
        latent_rep_size=shared["z_dim"],
        hidden_size=decoder_config["hidden_size"],
        output_size=shared["output_size"],
        max_length=shared["max_length"],
        num_layers=decoder_config["num_layers"],
    )

    assert decoder(torch.randn(2, 3)).shape == (2, 1, 8, 7)
    assert decoder(torch.randn(2, 4, 3)).shape == (2, 4, 8, 7)


def test_unimplemented_decoder_mode_fails_at_construction():
    with pytest.raises(ValueError, match="positional"):
        Decoder(3, 8, 7, 8, mode="autoregressive")
