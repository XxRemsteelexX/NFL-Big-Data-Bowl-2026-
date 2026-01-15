"""
NFL Big Data Bowl 2026 - Player Trajectory Prediction

A comprehensive deep learning package for predicting NFL player movements.

Main modules:
    - models: ST Transformer, GRU, CNN, Ensemble
    - data: Preprocessing, augmentation, datasets
    - training: Training utilities, losses, metrics
    - inference: Prediction and TTA
    - utils: Helpers and visualization

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main components for easy access
from .config import Config, STTransformerConfig, GRUConfig, MultiScaleCNNConfig

# Models
from .models.st_transformer import STTransformer, create_st_transformer
from .models.gru import JointSeqModel, create_gru_model
from .models.cnn_transformer import MultiScaleCNNTransformer, create_multiscale_cnn
from .models.ensemble import EnsemblePredictor, create_ensemble

# Data processing
from .data.preprocessing import (
    preprocess_pipeline,
    add_basic_features,
    add_ball_relative_features,
    add_temporal_features,
    add_geometric_features,
    prepare_sequences
)
from .data.augmentation import (
    horizontal_flip_dataframe,
    unflip_predictions,
    speed_perturbation,
    time_warp,
    augment_training_data,
    apply_tta
)

__all__ = [
    # Config
    'Config',
    'STTransformerConfig',
    'GRUConfig',
    'MultiScaleCNNConfig',

    # Models
    'STTransformer',
    'create_st_transformer',
    'JointSeqModel',
    'create_gru_model',
    'MultiScaleCNNTransformer',
    'create_multiscale_cnn',
    'EnsemblePredictor',
    'create_ensemble',

    # Data preprocessing
    'preprocess_pipeline',
    'add_basic_features',
    'add_ball_relative_features',
    'add_temporal_features',
    'add_geometric_features',
    'prepare_sequences',

    # Augmentation
    'horizontal_flip_dataframe',
    'unflip_predictions',
    'speed_perturbation',
    'time_warp',
    'augment_training_data',
    'apply_tta',
]
