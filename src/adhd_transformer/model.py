"""Transformer encoder for AAL functional-connectivity matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture settings for the connectivity transformer."""

    n_rois: int = 116
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    classifier_hidden: int = 32
    dropout: float = 0.1

    def validate(self) -> None:
        if self.n_rois < 2:
            raise ValueError("n_rois must be at least 2")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic sinusoidal encoding for the ordered ROI tokens."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.encoding[:, : tokens.size(1)]


class FunctionalConnectivityTransformer(nn.Module):
    """Binary classifier operating on square ROI connectivity matrices.

    Input shape: ``(batch, n_rois, n_rois)``. Each matrix row is treated as
    one ROI token whose features are its connectivity to every other ROI.
    The returned tensor contains unnormalized ADHD logits.
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        self.config.validate()

        self.input_projection = nn.Linear(self.config.n_rois, self.config.d_model)
        self.position = SinusoidalPositionalEncoding(
            self.config.d_model, self.config.n_rois
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
            norm=nn.LayerNorm(self.config.d_model),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.classifier_hidden),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden, 1),
        )

    def forward(self, connectivity: torch.Tensor) -> torch.Tensor:
        expected = (self.config.n_rois, self.config.n_rois)
        if connectivity.ndim != 3 or tuple(connectivity.shape[-2:]) != expected:
            raise ValueError(
                f"Expected input shape (batch, {expected[0]}, {expected[1]}), "
                f"received {tuple(connectivity.shape)}"
            )

        tokens = self.input_projection(connectivity.float())
        tokens = self.position(tokens)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)
