"""
NFL Big Data Bowl 2026 - Model Definitions

Contains all model architectures used in the competition:

    - STTransformer: 6-Layer Spatial-Temporal Transformer (0.547 LB)
    - MultiScaleCNNTransformer: Multi-scale CNN + 2-Layer Transformer (0.548 LB)
    - JointSeqModel: GRU with attention pooling (0.557 LB)
    - EnsemblePredictor: Weighted multi-model ensemble (0.541 LB)

Author: Glenn Dalbey
"""

from .st_transformer import STTransformer, create_st_transformer
from .cnn_transformer import MultiScaleCNNTransformer, create_multiscale_cnn
from .gru import JointSeqModel, create_gru_model
from .ensemble import EnsemblePredictor, create_ensemble

__all__ = [
    "STTransformer",
    "create_st_transformer",
    "MultiScaleCNNTransformer",
    "create_multiscale_cnn",
    "JointSeqModel",
    "create_gru_model",
    "EnsemblePredictor",
    "create_ensemble",
]
