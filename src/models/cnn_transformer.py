"""
MultiScale CNN Transformer for NFL Player Trajectory Prediction

Three parallel dilated-convolution streams capture temporal patterns at
different scales (dilation 1, 2, 3). Their outputs are concatenated, fused
through a linear projection, and then processed by a lightweight 2-layer
Spatial-Temporal Transformer with multi-query attention pooling and a
ResidualMLP prediction head.

Public LB: 0.548 (20-fold CV, horizontal flip augmentation).

Architecture (from actual Kaggle submission 'st-multiscale-cnn-w10-20fold'):
    MultiScale Conv1D (d=1,2,3) -> Fusion -> 2x TransformerEncoderLayer
    -> Multi-query attention pooling -> ResidualMLP -> cumsum trajectory

Author: Glenn Dalbey
"""

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """Residual MLP block used as the prediction head.

    Two-layer feedforward network with a skip connection from input to
    the hidden dimension, followed by a linear output projection.
    """

    def __init__(self, d_in, d_hidden, d_out, dropout=0.2):
        """
        Args:
            d_in: Input dimension.
            d_hidden: Hidden layer dimension.
            d_out: Output dimension.
            dropout: Dropout probability applied after each activation.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.proj = nn.Linear(d_in, d_hidden) if d_in != d_hidden else nn.Identity()
        self.out = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x: Tensor of shape (batch, d_in).

        Returns:
            Tensor of shape (batch, d_out).
        """
        h = self.drop(self.act(self.fc1(x)))
        h = self.drop(self.act(self.fc2(h)) + self.proj(x))
        return self.out(h)


class MultiScaleCNNTransformer(nn.Module):
    """Multi-scale CNN with a lightweight Transformer for trajectory prediction.

    Three parallel 1-D convolution branches with increasing dilation rates
    extract temporal features at multiple time scales.  The branch outputs are
    concatenated and linearly fused, then fed into a 2-layer Transformer
    encoder.  A multi-query attention pooling layer compresses the sequence
    into a fixed-size representation decoded by a ResidualMLP into cumulative
    (dx, dy) displacements.
    """

    def __init__(
        self,
        input_dim,
        horizon=94,
        hidden_dim=128,
        n_layers=2,
        n_heads=8,
        n_querys=2,
        mlp_hidden_dim=256,
        window_size=10,
        kernel_size=3,
        dropout=0.1,
    ):
        """
        Args:
            input_dim: Number of input features per frame.
            horizon: Maximum number of future frames to predict.
            hidden_dim: Hidden dimension used by convolutions and transformer.
            n_layers: Number of TransformerEncoderLayer blocks (default 2).
            n_heads: Number of attention heads.
            n_querys: Number of learnable pooling queries.
            mlp_hidden_dim: Hidden dimension inside the ResidualMLP head.
            window_size: Expected number of input frames.
            kernel_size: Convolution kernel size for all branches (default 3).
            dropout: Dropout rate.
        """
        super().__init__()

        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.n_querys = n_querys

        # Three parallel dilated convolution branches
        # Conv dimensions: (batch, channels, time) so we permute input
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=2 * (kernel_size // 2), dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=3 * (kernel_size // 2), dilation=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion: project concatenated branches back to hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned positional embedding (added after CNN fusion)
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim))
        self.embed_dropout = nn.Dropout(dropout)

        # Lightweight Transformer encoder (2 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Multi-query attention pooling
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, n_querys, hidden_dim))

        # Prediction head
        self.head = ResidualMLP(
            d_in=n_querys * hidden_dim,
            d_hidden=mlp_hidden_dim,
            d_out=horizon * 2,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, input_dim).

        Returns:
            Cumulative displacement tensor of shape (batch, horizon, 2)
            where the last dimension is (dx, dy).
        """
        B, T, _ = x.shape

        # Permute to (batch, features, time) for Conv1d
        x_conv = x.permute(0, 2, 1)

        # Multi-scale convolution branches
        c1 = self.conv1(x_conv)  # (B, hidden_dim, T')
        c2 = self.conv2(x_conv)
        c3 = self.conv3(x_conv)

        # Truncate / pad to match T (dilated convs may shift length)
        c1 = c1[:, :, :T]
        c2 = c2[:, :, :T]
        c3 = c3[:, :, :T]

        # Concatenate along feature dimension and permute back
        cat = torch.cat([c1, c2, c3], dim=1)  # (B, hidden_dim*3, T)
        cat = cat.permute(0, 2, 1)  # (B, T, hidden_dim*3)

        # Fuse to hidden_dim
        h = self.fusion(cat)  # (B, T, hidden_dim)

        # Add positional embeddings
        h = h + self.pos_embed[:, :T, :]
        h = self.embed_dropout(h)

        # Transformer encoding
        h = self.transformer_encoder(h)

        # Attention pooling
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.flatten(start_dim=1)  # (B, n_querys * hidden_dim)

        # Predict and accumulate
        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)
        out = torch.cumsum(out, dim=1)

        return out


def create_multiscale_cnn(
    input_dim,
    horizon=94,
    hidden_dim=128,
    n_layers=2,
    n_heads=8,
    n_querys=2,
    mlp_hidden_dim=256,
    window_size=10,
    kernel_size=3,
    dropout=0.1,
    device=None,
):
    """Factory function to create a MultiScaleCNNTransformer instance.

    Args:
        input_dim: Number of input features per frame.
        horizon: Maximum future frames to predict (default 94).
        hidden_dim: Hidden dimension (default 128).
        n_layers: Number of transformer layers (default 2).
        n_heads: Number of attention heads (default 8).
        n_querys: Number of pooling queries (default 2).
        mlp_hidden_dim: MLP head hidden dimension (default 256).
        window_size: Number of input frames (default 10).
        kernel_size: Convolution kernel size (default 3).
        dropout: Dropout rate (default 0.1).
        device: Target device. If None, stays on CPU.

    Returns:
        A MultiScaleCNNTransformer model instance, optionally moved to ``device``.
    """
    model = MultiScaleCNNTransformer(
        input_dim=input_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_querys=n_querys,
        mlp_hidden_dim=mlp_hidden_dim,
        window_size=window_size,
        kernel_size=kernel_size,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model
