"""
ST Transformer - Spatial-Temporal Transformer for NFL Player Trajectory Prediction

6-Layer Transformer Encoder with learned positional embeddings, multi-query
attention pooling, and a residual MLP prediction head. Predicts cumulative
(dx, dy) displacements for each future frame.

Best single model: 0.547 Public LB (20-fold CV, horizontal flip augmentation).

Architecture (from actual Kaggle submission '6layer-seed700-flip-only'):
    Input projection -> Positional embedding -> 6x TransformerEncoderLayer
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


class STTransformer(nn.Module):
    """Spatial-Temporal Transformer for trajectory prediction.

    Projects per-frame features into a hidden space, adds learned positional
    embeddings, encodes with a stack of TransformerEncoderLayers (pre-norm),
    pools via multi-query cross-attention, and decodes through a ResidualMLP.
    The output is a cumulative-sum trajectory of (dx, dy) per future frame.
    """

    def __init__(
        self,
        input_dim,
        horizon=94,
        hidden_dim=128,
        n_layers=6,
        n_heads=8,
        n_querys=2,
        mlp_hidden_dim=256,
        window_size=10,
        dropout=0.1,
    ):
        """
        Args:
            input_dim: Number of input features per frame.
            horizon: Maximum number of future frames to predict.
            hidden_dim: Transformer hidden / embedding dimension.
            n_layers: Number of TransformerEncoderLayer blocks.
            n_heads: Number of attention heads.
            n_querys: Number of learnable pooling queries.
            mlp_hidden_dim: Hidden dimension inside the ResidualMLP head.
            window_size: Expected number of input frames (for positional embedding).
            dropout: Dropout rate used throughout the model.
        """
        super().__init__()

        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_querys = n_querys

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learned positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim))
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder (pre-norm)
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

        # Project input features to hidden dimension
        x = self.input_proj(x)

        # Add positional embeddings (truncated if T < window_size)
        x = x + self.pos_embed[:, :T, :]
        x = self.embed_dropout(x)

        # Encode with transformer
        h = self.transformer_encoder(x)

        # Attention pooling with learnable queries
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.flatten(start_dim=1)  # (B, n_querys * hidden_dim)

        # Predict per-step displacements and accumulate
        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)
        out = torch.cumsum(out, dim=1)

        return out


def create_st_transformer(
    input_dim,
    horizon=94,
    hidden_dim=128,
    n_layers=6,
    n_heads=8,
    n_querys=2,
    mlp_hidden_dim=256,
    window_size=10,
    dropout=0.1,
    device=None,
):
    """Factory function to create an STTransformer instance.

    Args:
        input_dim: Number of input features per frame.
        horizon: Maximum future frames to predict (default 94).
        hidden_dim: Transformer hidden dimension (default 128).
        n_layers: Number of transformer layers (default 6).
        n_heads: Number of attention heads (default 8).
        n_querys: Number of pooling queries (default 2).
        mlp_hidden_dim: MLP head hidden dimension (default 256).
        window_size: Number of input frames (default 10).
        dropout: Dropout rate (default 0.1).
        device: Target device (e.g. 'cuda' or 'cpu'). If None, stays on CPU.

    Returns:
        An STTransformer model instance, optionally moved to ``device``.
    """
    model = STTransformer(
        input_dim=input_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_querys=n_querys,
        mlp_hidden_dim=mlp_hidden_dim,
        window_size=window_size,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model
