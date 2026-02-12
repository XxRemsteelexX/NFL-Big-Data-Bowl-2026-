"""
GRU Model (JointSeqModel) for NFL Player Trajectory Prediction

A GRU-based sequence model with multi-query attention pooling and a
residual MLP head. Supports optional bidirectional encoding and an input
skip connection. Predicts cumulative (dx, dy) displacements.

Two variants are present in the competition code:

1. Simple GRU (from nfl_gnn.py) -- single-query attention, simple MLP head,
   hidden_dim=128. Used with 167 geometric features including route
   clustering, GNN neighbor embeddings, and geometric endpoint prediction.

2. Enhanced GRU / SeqModel (from nfl_gru.py) -- bidirectional GRU, multi-query
   attention pooling, ResidualMLP head, input residual projection.
   hidden_dim=128 (or 64 for production ensemble).

The ``JointSeqModel`` class below unifies both variants through constructor
arguments. The default configuration matches the production ensemble model
(GRU Seed 27, 0.557 Public LB, 20-fold CV).

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


class JointSeqModel(nn.Module):
    """GRU sequence model with attention pooling for trajectory prediction.

    Encodes a window of per-frame features with a multi-layer GRU, pools the
    hidden states via learnable-query cross-attention, and decodes with a
    ResidualMLP into cumulative (dx, dy) displacements.

    The model supports both unidirectional and bidirectional GRU encoding. When
    bidirectional mode is enabled, the GRU output dimension doubles and an
    optional input-to-hidden residual projection is added.
    """

    def __init__(
        self,
        input_dim,
        horizon=94,
        hidden_dim=64,
        num_layers=2,
        n_heads=4,
        n_querys=2,
        bidirectional=False,
        use_residual=True,
        dropout=0.1,
    ):
        """
        Args:
            input_dim: Number of input features per frame.
            horizon: Maximum number of future frames to predict.
            hidden_dim: GRU hidden dimension (output is 2x if bidirectional).
            num_layers: Number of stacked GRU layers.
            n_heads: Number of attention heads for pooling.
            n_querys: Number of learnable pooling queries.
            bidirectional: Whether to use a bidirectional GRU.
            use_residual: Whether to add an input residual to GRU output.
            dropout: Dropout rate.
        """
        super().__init__()

        self.horizon = horizon
        self.bidirectional = bidirectional
        self.use_residual = use_residual

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Effective hidden size after GRU
        h_out = hidden_dim * (2 if bidirectional else 1)

        # Optional input residual projection
        self.in_proj = nn.Linear(input_dim, h_out) if use_residual else None

        # Multi-query attention pooling
        self.pool_ln = nn.LayerNorm(h_out)
        self.pool_attn = nn.MultiheadAttention(
            h_out,
            num_heads=n_heads,
            batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, n_querys, h_out))

        # Prediction head
        self.head = ResidualMLP(
            d_in=h_out * n_querys,
            d_hidden=hidden_dim,
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
        h, _ = self.gru(x)  # (B, T, h_out)

        # Add input residual if enabled
        if self.use_residual and self.in_proj is not None:
            h = h + self.in_proj(x)

        B = h.size(0)

        # Multi-query attention pooling
        q = self.pool_query.expand(B, -1, -1)  # (B, n_querys, h_out)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.reshape(B, -1)  # (B, n_querys * h_out)

        # Predict per-step displacements and accumulate
        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)
        out = torch.cumsum(out, dim=1)

        return out


def create_gru_model(
    input_dim,
    horizon=94,
    hidden_dim=64,
    num_layers=2,
    n_heads=4,
    n_querys=2,
    bidirectional=False,
    use_residual=True,
    dropout=0.1,
    device=None,
):
    """Factory function to create a JointSeqModel (GRU) instance.

    Args:
        input_dim: Number of input features per frame.
        horizon: Maximum future frames to predict (default 94).
        hidden_dim: GRU hidden dimension (default 64).
        num_layers: Number of GRU layers (default 2).
        n_heads: Number of attention heads (default 4).
        n_querys: Number of pooling queries (default 2).
        bidirectional: Whether to use bidirectional GRU (default False).
        use_residual: Whether to add input residual (default True).
        dropout: Dropout rate (default 0.1).
        device: Target device. If None, stays on CPU.

    Returns:
        A JointSeqModel instance, optionally moved to ``device``.
    """
    model = JointSeqModel(
        input_dim=input_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_heads=n_heads,
        n_querys=n_querys,
        bidirectional=bidirectional,
        use_residual=use_residual,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model
