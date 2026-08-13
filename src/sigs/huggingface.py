"""Optional Hugging Face Transformers integration for SIGS.

Install it with ``pip install 'sigs[huggingface]'``.  The native model remains
the source of truth; this module provides standard configuration, typed outputs,
``save_pretrained``/``from_pretrained``, and Hub interoperability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from sigs.model import GrammarVAE


class SIGSConfig(PretrainedConfig):
    """Serializable architecture and inference configuration for SIGS."""

    model_type = "sigs"

    def __init__(
        self,
        *,
        vocab_size: int = 53,
        sequence_length: int = 72,
        latent_size: int = 32,
        encoder_hidden_size: int = 256,
        encoder_conv_sizes: tuple[int, ...] = (64, 128, 256),
        encoder_kernel_sizes: tuple[int, ...] = (2, 3, 4),
        encoder_layer_norm: bool = False,
        encoder_dropout: float = 0.0,
        decoder_hidden_size: int = 512,
        decoder_num_layers: int = 1,
        decoder_rnn_type: str = "gru",
        decoder_dropout: float = 0.0,
        decoder_mode: str = "positional",
        decoder_samples: int = 1,
        tightness: float = 1.0,
        kl_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(encoder_conv_sizes) != len(encoder_kernel_sizes):
            raise ValueError("encoder_conv_sizes and encoder_kernel_sizes must have equal length")
        if vocab_size < 1 or sequence_length < 1 or latent_size < 1:
            raise ValueError("vocab_size, sequence_length, and latent_size must be positive")

        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.latent_size = int(latent_size)
        self.encoder_hidden_size = int(encoder_hidden_size)
        self.encoder_conv_sizes = list(encoder_conv_sizes)
        self.encoder_kernel_sizes = list(encoder_kernel_sizes)
        self.encoder_layer_norm = bool(encoder_layer_norm)
        self.encoder_dropout = float(encoder_dropout)
        self.decoder_hidden_size = int(decoder_hidden_size)
        self.decoder_num_layers = int(decoder_num_layers)
        self.decoder_rnn_type = decoder_rnn_type
        self.decoder_dropout = float(decoder_dropout)
        self.decoder_mode = decoder_mode
        self.decoder_samples = int(decoder_samples)
        self.tightness = float(tightness)
        self.kl_weight = float(kl_weight)
        self.architectures = ["SIGSModel"]

    @classmethod
    def from_native_config(cls, config: dict, **kwargs) -> "SIGSConfig":
        """Build a Transformers config from the original nested SIGS config."""

        encoder = config["model"]["encoder"]
        decoder = config["model"]["decoder"]
        shared = config["model"]["shared"]
        training = config["training"]
        return cls(
            vocab_size=shared["output_size"],
            sequence_length=shared["max_length"],
            latent_size=shared["z_dim"],
            encoder_hidden_size=encoder["hidden_size"],
            encoder_conv_sizes=tuple(encoder["conv_sizes"]),
            encoder_kernel_sizes=tuple(encoder["kernel_sizes"]),
            encoder_layer_norm=encoder["use_batch_norm"],
            encoder_dropout=encoder.get("dropout", 0.0),
            decoder_hidden_size=decoder["hidden_size"],
            decoder_num_layers=decoder["num_layers"],
            decoder_rnn_type=decoder["rnn_type"],
            decoder_dropout=decoder.get("dropout", 0.0),
            decoder_mode=decoder.get("mode", "positional"),
            decoder_samples=training.get("decoder_samples", 1),
            tightness=training.get("tightness", 1.0),
            **kwargs,
        )

    def to_native_config(self) -> dict:
        """Return the nested dictionary consumed by :class:`GrammarVAE`."""

        return {
            "model": {
                "encoder": {
                    "hidden_size": self.encoder_hidden_size,
                    "conv_sizes": list(self.encoder_conv_sizes),
                    "kernel_sizes": list(self.encoder_kernel_sizes),
                    "use_batch_norm": self.encoder_layer_norm,
                    "dropout": self.encoder_dropout,
                },
                "decoder": {
                    "hidden_size": self.decoder_hidden_size,
                    "num_layers": self.decoder_num_layers,
                    "rnn_type": self.decoder_rnn_type,
                    "dropout": self.decoder_dropout,
                    "mode": self.decoder_mode,
                },
                "shared": {
                    "z_dim": self.latent_size,
                    "max_length": self.sequence_length,
                    "output_size": self.vocab_size,
                },
            },
            "training": {
                "tightness": self.tightness,
                "decoder_samples": self.decoder_samples,
            },
        }


@dataclass
class SIGSOutput(ModelOutput):
    """Typed output returned by :class:`SIGSModel`."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    latent_mean: Optional[torch.FloatTensor] = None
    latent_logvar: Optional[torch.FloatTensor] = None


class SIGSModel(PreTrainedModel):
    """Transformers-compatible wrapper around the native SIGS Grammar-VAE."""

    config_class = SIGSConfig
    base_model_prefix = "grammar_vae"
    main_input_name = "inputs"

    def __init__(self, config: SIGSConfig):
        super().__init__(config)
        self.grammar_vae = GrammarVAE(config.to_native_config())
        # Establish Transformers' weight-tying/loading bookkeeping. Our
        # ``_init_weights`` hook is intentionally a no-op, so this preserves the
        # native GrammarVAE initialization.
        self.post_init()

    def _init_weights(self, module) -> None:
        # Native PyTorch modules initialize themselves before PreTrainedModel sees
        # them. Keeping that initialization preserves parity with GrammarVAE.
        del module

    def forward(
        self,
        inputs: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        *,
        num_samples: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ) -> SIGSOutput | tuple:
        """Run the grammar VAE and optionally compute its ELBO-style loss."""

        logits, latent_mean, latent_logvar = self.grammar_vae(inputs, num_samples=num_samples)
        loss = None
        if labels is not None:
            if labels.shape != (inputs.shape[0], self.config.sequence_length):
                raise ValueError(
                    f"labels must have shape [batch, sequence_length], got {tuple(labels.shape)}"
                )
            reconstruction = (
                F.cross_entropy(
                    logits.mean(dim=1).reshape(-1, self.config.vocab_size),
                    labels.reshape(-1),
                )
                * self.config.sequence_length
            )
            loss = reconstruction + self.config.kl_weight * GrammarVAE.kl_divergence(
                latent_mean, latent_logvar
            )

        if return_dict is None:
            return_dict = self.config.return_dict
        if not return_dict:
            values = (logits, latent_mean, latent_logvar)
            return ((loss,) + values) if loss is not None else values
        return SIGSOutput(
            loss=loss,
            logits=logits,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
        )

    @classmethod
    def from_native_model(
        cls,
        native_model: GrammarVAE,
        config: SIGSConfig,
    ) -> "SIGSModel":
        """Wrap an in-memory native model without changing its parameters."""

        model = cls(config)
        model.grammar_vae.load_state_dict(native_model.state_dict(), strict=True)
        return model

    @classmethod
    def from_lightning_checkpoint(
        cls,
        checkpoint: str | Path,
        config: SIGSConfig,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> "SIGSModel":
        """Convert the paper's Lightning checkpoint to a Transformers model."""

        checkpoint = Path(checkpoint)
        with checkpoint.open("rb") as handle:
            header = handle.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{checkpoint} is a Git LFS pointer, not the checkpoint data; "
                "install Git LFS and run 'git lfs pull --include=data/model.ckpt'"
            )

        try:
            payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility
            payload = torch.load(checkpoint, map_location=map_location)
        state_dict = payload.get("state_dict", payload)
        native_state = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        if not native_state:
            raise ValueError("checkpoint does not contain Lightning model.* parameters")

        model = cls(config)
        model.grammar_vae.load_state_dict(native_state, strict=strict)
        return model


def register_with_transformers() -> None:
    """Register SIGS with ``AutoConfig`` and ``AutoModel`` in this process."""

    try:
        AutoConfig.register(SIGSConfig.model_type, SIGSConfig, exist_ok=True)
    except TypeError:  # older Transformers releases
        AutoConfig.register(SIGSConfig.model_type, SIGSConfig)
    AutoModel.register(SIGSConfig, SIGSModel, exist_ok=True)
