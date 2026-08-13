"""Decoder used by the SIGS grammar variational autoencoder."""

from __future__ import annotations

import torch
from torch import nn


class Decoder(nn.Module):
    """Decode latent vectors into logits over grammar productions."""

    def __init__(
        self,
        latent_rep_size: int,
        hidden_size: int,
        output_size: int,
        max_length: int,
        rnn_type: str = "gru",
        num_layers: int = 3,
        dropout: float = 0.0,
        mode: str = "positional",
    ):
        super().__init__()
        if mode != "positional":
            raise ValueError("only the checkpoint-compatible 'positional' decoder is implemented")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.mode = mode
        self.max_length = int(max_length)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.linear_in = nn.Linear(latent_rep_size, hidden_size, bias=False)
        self.elu = nn.ELU()

        recurrent_dropout = float(dropout) if num_layers > 1 else 0.0
        rnn_kind = rnn_type.lower()
        if rnn_kind == "gru":
            rnn_class = nn.GRU
        elif rnn_kind == "rnn":
            rnn_class = nn.RNN
        else:
            raise ValueError(f"unsupported rnn_type={rnn_type!r}; use 'gru' or 'rnn'")
        self.rnn = rnn_class(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.time_distributed_dense = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``[batch, samples, length, productions]``."""

        if z.ndim == 2:
            z = z.unsqueeze(1)
        if z.ndim != 3:
            raise ValueError(
                "z must have shape [batch, latent_dim] or [batch, samples, latent_dim]"
            )

        batch_size, num_samples, latent_dim = z.shape
        z_flat = z.reshape(batch_size * num_samples, latent_dim)
        hidden = self.elu(self.linear_in(z_flat))
        hidden = hidden.unsqueeze(1).expand(-1, self.max_length, -1)
        hidden, _ = self.rnn(hidden)
        logits = self.time_distributed_dense(self.elu(hidden))
        return logits.reshape(batch_size, num_samples, self.max_length, self.output_size)
