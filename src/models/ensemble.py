"""
Ensemble Predictor for NFL Player Trajectory Prediction

Combines predictions from multiple model architectures (ST Transformer,
MultiScale CNN, GRU, Position-Specific ST) using inverse-score weighted
averaging. Supports optional test-time augmentation (horizontal flip).

Best ensemble: 0.541 Public LB (4-model weighted average with TTA).

Ensemble weights (from actual submission 'ensemble_4model_SIMPLE.ipynb'):
    ST Transformer (6L): 0.2517
    Multiscale CNN:      0.2517
    GRU (Seed 27):       0.2476
    Position-Specific:   0.2490

Author: Glenn Dalbey
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


# Default ensemble weights from the best submission
DEFAULT_WEIGHTS = {
    "st_transformer": 0.2517,
    "multiscale_cnn": 0.2517,
    "gru": 0.2476,
    "position_st": 0.2490,
}

# Field dimensions (yards)
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3


def calculate_weights(lb_scores: Dict[str, float]) -> Dict[str, float]:
    """Calculate ensemble weights from public leaderboard scores.

    Uses inverse-score weighting: lower LB score (better) receives a higher
    weight. Weights are normalized to sum to 1.

    Args:
        lb_scores: Mapping of model name to its public LB score.

    Returns:
        Mapping of model name to its normalized weight.
    """
    inv_scores = {k: 1.0 / v for k, v in lb_scores.items()}
    total = sum(inv_scores.values())
    return {k: v / total for k, v in inv_scores.items()}


class EnsemblePredictor:
    """Weighted ensemble of multiple trajectory prediction models.

    Loads fold-level models and scalers for each component architecture,
    then produces predictions via weighted averaging across both folds and
    model types. Optionally applies horizontal-flip test-time augmentation.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        field_length: float = FIELD_LENGTH,
        field_width: float = FIELD_WIDTH,
    ):
        """
        Args:
            weights: Per-model weights (keys are model names). Defaults to
                the competition ensemble weights.
            device: Torch device string ('cuda' or 'cpu').
            field_length: Field length in yards for clipping (default 120.0).
            field_width: Field width in yards for clipping (default 53.3).
        """
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self.device = torch.device(device)
        self.field_length = field_length
        self.field_width = field_width

        # Storage: model_name -> list of (model, scaler) tuples across folds
        self.models: Dict[str, List[torch.nn.Module]] = {}
        self.scalers: Dict[str, List] = {}

        self.loaded = False

    def register_model(
        self,
        name: str,
        models: List[torch.nn.Module],
        scalers: Optional[List] = None,
    ):
        """Register a set of fold models under the given name.

        Args:
            name: Model identifier (must match a key in ``self.weights``).
            models: List of trained PyTorch model instances (one per fold).
            scalers: Optional list of sklearn-compatible scalers (one per fold).
        """
        self.models[name] = models
        self.scalers[name] = scalers if scalers is not None else [None] * len(models)
        self.loaded = bool(self.models)

    def _predict_single_model(
        self,
        model: torch.nn.Module,
        scaler,
        sequences: List[np.ndarray],
    ) -> np.ndarray:
        """Run inference for one model on a batch of sequences.

        Args:
            model: A trained PyTorch model.
            scaler: An sklearn-compatible scaler (or None).
            sequences: List of (window_size, n_features) arrays.

        Returns:
            Predictions array of shape (N, horizon, 2).
        """
        if scaler is not None:
            X = [scaler.transform(s) for s in sequences]
        else:
            X = sequences

        X_tensor = torch.tensor(
            np.stack(X).astype(np.float32)
        ).to(self.device)

        model.eval()
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()

        return preds

    def _horizontal_flip_sequence(
        self,
        seq: np.ndarray,
        y_idx: int = 1,
    ) -> np.ndarray:
        """Apply horizontal flip to a single sequence for TTA.

        Mirrors the y-coordinate across the field midline.

        Args:
            seq: Array of shape (window_size, n_features).
            y_idx: Column index of the y-coordinate.

        Returns:
            A flipped copy of the sequence.
        """
        flipped = seq.copy()
        flipped[:, y_idx] = self.field_width - flipped[:, y_idx]
        return flipped

    def predict(
        self,
        sequences: List[np.ndarray],
        use_tta: bool = True,
        y_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ensemble prediction over all registered models.

        For each registered model group, averages predictions across folds,
        optionally applies horizontal-flip TTA, and then combines model
        groups via the configured weights.

        Args:
            sequences: List of (window_size, n_features) input arrays.
            use_tta: Whether to apply horizontal-flip TTA (default True).
            y_idx: Column index of the y-coordinate in sequences (for TTA).

        Returns:
            Tuple of (ens_dx, ens_dy) arrays, each of shape (N, horizon).
        """
        all_dx: List[np.ndarray] = []
        all_dy: List[np.ndarray] = []
        all_weights: List[float] = []

        for model_name, fold_models in self.models.items():
            if not fold_models:
                continue

            fold_scalers = self.scalers.get(model_name, [None] * len(fold_models))
            weight = self.weights.get(model_name, 1.0 / max(len(self.models), 1))

            fold_preds = []
            for model, scaler in zip(fold_models, fold_scalers):
                preds = self._predict_single_model(model, scaler, sequences)

                if use_tta:
                    flipped_seqs = [
                        self._horizontal_flip_sequence(s, y_idx=y_idx)
                        for s in sequences
                    ]
                    preds_flip = self._predict_single_model(
                        model, scaler, flipped_seqs
                    )
                    # Average original and flipped (negate dy of flipped)
                    preds[:, :, 0] = (preds[:, :, 0] + preds_flip[:, :, 0]) / 2.0
                    preds[:, :, 1] = (preds[:, :, 1] - preds_flip[:, :, 1]) / 2.0

                fold_preds.append(preds)

            # Average across folds for this model type
            model_preds = np.mean(fold_preds, axis=0)
            all_dx.append(model_preds[:, :, 0] * weight)
            all_dy.append(model_preds[:, :, 1] * weight)
            all_weights.append(weight)

        # Normalise in case weights don't sum to 1
        total_weight = sum(all_weights) if all_weights else 1.0
        ens_dx = sum(all_dx) / total_weight
        ens_dy = sum(all_dy) / total_weight

        return ens_dx, ens_dy

    def predict_clipped(
        self,
        sequences: List[np.ndarray],
        last_x: np.ndarray,
        last_y: np.ndarray,
        use_tta: bool = True,
        y_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ensemble prediction and return absolute positions clipped to field.

        Convenience wrapper that adds last-known positions to the predicted
        displacements and clips to field boundaries.

        Args:
            sequences: List of (window_size, n_features) input arrays.
            last_x: Array of shape (N,) with last known x positions.
            last_y: Array of shape (N,) with last known y positions.
            use_tta: Whether to apply horizontal-flip TTA.
            y_idx: Column index of y-coordinate for TTA.

        Returns:
            Tuple of (abs_x, abs_y) arrays, each of shape (N, horizon),
            clipped to [0, field_length] and [0, field_width] respectively.
        """
        ens_dx, ens_dy = self.predict(sequences, use_tta=use_tta, y_idx=y_idx)

        abs_x = last_x[:, None] + ens_dx
        abs_y = last_y[:, None] + ens_dy

        abs_x = np.clip(abs_x, 0.0, self.field_length)
        abs_y = np.clip(abs_y, 0.0, self.field_width)

        return abs_x, abs_y


def create_ensemble(
    weights: Optional[Dict[str, float]] = None,
    device: str = "cuda",
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
) -> EnsemblePredictor:
    """Factory function to create an EnsemblePredictor.

    Args:
        weights: Per-model weights. Defaults to the competition ensemble
            weights (inverse LB score weighting).
        device: Torch device string ('cuda' or 'cpu').
        field_length: Field length in yards for clipping (default 120.0).
        field_width: Field width in yards for clipping (default 53.3).

    Returns:
        An EnsemblePredictor instance ready for model registration via
        ``register_model()``.
    """
    return EnsemblePredictor(
        weights=weights,
        device=device,
        field_length=field_length,
        field_width=field_width,
    )
