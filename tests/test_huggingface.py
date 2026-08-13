"""Contract tests for the optional Transformers adapter."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModel

from sigs.huggingface import (
    SIGSConfig,
    SIGSModel,
    register_with_transformers,
)
from sigs.model import GrammarVAE


def test_config_round_trips_native_shape(tiny_native_config):
    config = SIGSConfig.from_native_config(tiny_native_config)

    assert config.to_native_config() == tiny_native_config


def test_transformers_forward_returns_typed_output_and_loss(tiny_native_config):
    config = SIGSConfig.from_native_config(tiny_native_config)
    model = SIGSModel(config)
    inputs = torch.randn(2, 7, 8)
    labels = torch.randint(0, 7, (2, 8))

    output = model(inputs, labels=labels)

    assert output.loss.ndim == 0
    assert output.logits.shape == (2, 2, 8, 7)
    assert output.latent_mean.shape == output.latent_logvar.shape == (2, 3)


def test_save_pretrained_round_trip_and_auto_model(tmp_path, tiny_native_config):
    config = SIGSConfig.from_native_config(tiny_native_config)
    model = SIGSModel(config)
    model.save_pretrained(tmp_path)

    restored = SIGSModel.from_pretrained(tmp_path)
    assert restored.config.to_native_config() == tiny_native_config
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])

    register_with_transformers()
    automatic = AutoModel.from_pretrained(tmp_path)
    assert isinstance(automatic, SIGSModel)


def test_wrap_native_model_preserves_parameters(tiny_native_config):
    native = GrammarVAE(tiny_native_config)
    config = SIGSConfig.from_native_config(tiny_native_config)

    wrapped = SIGSModel.from_native_model(native, config)

    for key, value in native.state_dict().items():
        assert torch.equal(value, wrapped.grammar_vae.state_dict()[key])


def test_convert_lightning_checkpoint(tmp_path, tiny_native_config):
    native = GrammarVAE(tiny_native_config)
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in native.state_dict().items()}},
        checkpoint,
    )
    config = SIGSConfig.from_native_config(tiny_native_config)

    converted = SIGSModel.from_lightning_checkpoint(checkpoint, config)

    for key, value in native.state_dict().items():
        assert torch.equal(value, converted.grammar_vae.state_dict()[key])


def test_convert_lightning_checkpoint_rejects_git_lfs_pointer(tmp_path, tiny_native_config):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:0123456789abcdef\nsize 73537427\n"
    )
    config = SIGSConfig.from_native_config(tiny_native_config)

    with pytest.raises(ValueError, match="Git LFS pointer"):
        SIGSModel.from_lightning_checkpoint(checkpoint, config)
