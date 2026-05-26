from __future__ import annotations

import torch
from torch import nn


class _TemporalAttentionPool(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class PSRCNNLSTMRegressor(nn.Module):
    """Compact PSR-CNN-LSTM regressor.

    Input shape: [batch, reconstructed_time, embedding_dim].
    """

    def __init__(
        self,
        embedding_dim: int,
        conv_channels: int = 16,
        lstm_hidden: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNBiLSTMAttentionRegressor(nn.Module):
    """Lightweight CNN-BiLSTM-Attention regressor over causal flow windows."""

    def __init__(
        self,
        input_dim: int = 1,
        conv_channels: int = 16,
        lstm_hidden: int = 24,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        repr_dim = 2 * lstm_hidden
        self.pool = _TemporalAttentionPool(repr_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(repr_dim, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.bilstm(z)
        pooled = self.pool(out)
        return self.head(pooled).squeeze(-1)


class _CausalConv1d(nn.Module):
    """1D convolution with explicit left padding to preserve causality."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = int(dilation) * (int(kernel_size) - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)


class BiLSTMCTPALiteRegressor(nn.Module):
    """BiLSTM with causal temporal pattern attention.

    This compact comparator preserves the two-branch idea of the recent
    BO-BiLSTM-CTPA passenger-flow architecture: a recurrent branch models the
    local sequence, while dilated causal convolutions form temporal patterns that
    are attention-pooled before late fusion.
    """

    def __init__(
        self,
        input_dim: int = 1,
        conv_channels: int = 24,
        lstm_hidden: int = 24,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        recurrent_dim = 2 * lstm_hidden
        self.recurrent_pool = _TemporalAttentionPool(recurrent_dim)

        self.pattern = nn.Sequential(
            _CausalConv1d(input_dim, conv_channels, kernel_size=3, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(conv_channels, conv_channels, kernel_size=3, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(conv_channels, conv_channels, kernel_size=3, dilation=4),
            nn.GELU(),
        )
        self.pattern_pool = _TemporalAttentionPool(conv_channels)

        fusion_dim = recurrent_dim + conv_channels
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.bilstm(x)
        recurrent_repr = self.recurrent_pool(recurrent)

        pattern = self.pattern(x.transpose(1, 2)).transpose(1, 2)
        pattern_repr = self.pattern_pool(pattern)

        fused = torch.cat([recurrent_repr, pattern_repr], dim=-1)
        return self.head(fused).squeeze(-1)


class _LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MSDInformerBiLSTMLiteRegressor(nn.Module):
    """Multi-scale decomposition + attention/BiLSTM fusion comparator.

    The attention branch is a compact Informer-style approximation, while the
    recurrent branch follows the BiLSTM part of the cited decomposition-based rail
    passenger-flow family.
    """

    def __init__(
        self,
        input_dim: int = 3,
        max_len: int = 24,
        d_model: int = 32,
        nhead: int = 4,
        ff_dim: int = 64,
        lstm_hidden: int = 24,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = _LearnedPositionalEncoding(max_len=max_len, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attn_pool = _TemporalAttentionPool(d_model)

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_pool = _TemporalAttentionPool(2 * lstm_hidden)

        fusion_dim = d_model + 2 * lstm_hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.proj(x)
        attn = self.pos(attn)
        attn = self.attn_encoder(attn)
        attn_repr = self.attn_pool(attn)

        recurrent, _ = self.bilstm(x)
        recurrent_repr = self.lstm_pool(recurrent)

        fused = torch.cat([attn_repr, recurrent_repr], dim=-1)
        return self.head(fused).squeeze(-1)
