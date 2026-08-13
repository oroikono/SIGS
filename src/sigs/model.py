"""Native PyTorch implementation of the SIGS grammar variational autoencoder."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from nltk import Nonterminal
from torch import nn

from sigs.decoder import Decoder
from sigs.encoder import Encoder
from sigs.grammar import GCFG, S, get_mask
from sigs.stack import Stack


class GrammarVAE(nn.Module):
    """Grammar variational autoencoder used by SIGS.

    The module follows normal PyTorch device semantics: callers move the model
    and tensors with :meth:`torch.nn.Module.to`. ``forward`` never guesses a
    device or silently copies its inputs.

    Parameters
    ----------
    config:
        Native SIGS configuration dictionary. The model architecture is read
        from ``config["model"]`` and sampling defaults from ``config["training"]``.
    """

    def __init__(self, config: dict):
        super().__init__()
        enc_config = config["model"]["encoder"]
        dec_config = config["model"]["decoder"]
        shared_config = config["model"]["shared"]

        self.encoder = Encoder(
            input_dim=shared_config["output_size"],
            hidden_dim=enc_config["hidden_size"],
            z_dim=shared_config["z_dim"],
            max_length=shared_config["max_length"],
            conv_sizes=enc_config["conv_sizes"],
            kernel_sizes=enc_config["kernel_sizes"],
            use_batch_norm=enc_config["use_batch_norm"],
            dropout=enc_config.get("dropout", 0.0),
        )
        self.decoder = Decoder(
            latent_rep_size=shared_config["z_dim"],
            hidden_size=dec_config["hidden_size"],
            output_size=shared_config["output_size"],
            max_length=shared_config["max_length"],
            rnn_type=dec_config["rnn_type"],
            num_layers=dec_config["num_layers"],
            dropout=dec_config.get("dropout", 0.0),
            mode=dec_config.get("mode", "positional"),
        )

        self.max_length = int(shared_config["max_length"])
        self.tightness = float(config["training"].get("tightness", 1.0))
        self.decoder_samples = int(config["training"].get("decoder_samples", 1))
        if self.decoder_samples < 1:
            raise ValueError("training.decoder_samples must be at least 1")

    @property
    def device(self) -> torch.device:
        """Return the device holding the model parameters."""

        return next(self.parameters()).device

    def sample(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        num_samples: Optional[int] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample latent vectors with the reparameterization trick.

        Returns a tensor shaped ``[batch, samples, latent_dim]``.
        """

        if mu.shape != logvar.shape or mu.ndim != 2:
            raise ValueError("mu and logvar must have identical [batch, latent_dim] shapes")
        count = self.decoder_samples if num_samples is None else int(num_samples)
        if count < 1:
            raise ValueError("num_samples must be at least 1")

        batch_size, latent_dim = mu.shape
        mean = mu.unsqueeze(1)
        std = (0.5 * logvar).exp().unsqueeze(1)
        eps = torch.randn(
            batch_size,
            count,
            latent_dim,
            device=mu.device,
            dtype=mu.dtype,
            generator=generator,
        )
        return mean + eps * std * self.tightness

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Return the batch-mean KL divergence from a unit Gaussian."""

        if mu.shape != logvar.shape:
            raise ValueError("mu and logvar must have identical shapes")
        per_dimension = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
        return per_dimension.mean(dim=1).mean()

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode production sequences and decode sampled latent vectors.

        ``targets`` remains accepted for backward compatibility with the original
        research scripts; loss computation is intentionally owned by the trainer.
        """

        del targets
        if x.device != self.device:
            raise RuntimeError(
                f"input is on {x.device}, but SIGS is on {self.device}; "
                "move them to the same device before calling the model"
            )
        mu, logvar = self.encoder(x)
        z = self.sample(mu, logvar, num_samples, generator=generator)
        return self.decoder(z), mu, logvar

    def generate(
        self,
        z: torch.Tensor,
        *,
        sample: bool = False,
        seed: Optional[int] = None,
    ) -> List:
        """Decode one latent vector into a grammar-production sequence.

        ``seed`` controls a local generator and does not alter the global RNG.
        """

        if z.ndim == 1:
            z = z.unsqueeze(0).unsqueeze(0)
        elif z.ndim == 2:
            z = z.unsqueeze(1)
        if z.ndim != 3 or z.shape[:2] != (1, 1):
            raise ValueError("generate expects exactly one latent vector")
        if z.device != self.device:
            raise RuntimeError(f"latent vector is on {z.device}, but SIGS is on {self.device}")

        logits = self.decoder(z)[0, 0]
        stack = Stack(grammar=GCFG, start_symbol=S)
        rules = []
        generator = None
        if seed is not None:
            generator = torch.Generator(device=z.device).manual_seed(seed)

        for step in range(self.max_length):
            if not stack.nonempty:
                break
            symbol = stack.pop()
            mask = get_mask(symbol, stack.grammar, as_variable=True).to(
                device=z.device, dtype=torch.bool
            )
            if not torch.any(mask):
                raise RuntimeError(f"grammar has no admissible production for {symbol}")
            masked_logits = logits[step].masked_fill(~mask, float("-inf"))
            if sample:
                probabilities = torch.softmax(masked_logits, dim=-1)
                index = torch.multinomial(probabilities, 1, generator=generator).item()
            else:
                index = int(masked_logits.argmax())

            rule = stack.grammar.productions()[index]
            rules.append(rule)
            for child in reversed(rule.rhs()):
                if isinstance(child, Nonterminal):
                    stack.push(child)

        if stack.nonempty:
            raise RuntimeError(
                f"generation reached max_length={self.max_length} before completing the tree"
            )
        return rules

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode production sequences into Gaussian parameters."""

        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into production logits."""

        return self.decoder(z)
