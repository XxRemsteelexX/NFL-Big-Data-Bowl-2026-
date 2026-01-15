"""
Data Processing Module

Preprocessing, feature engineering, and augmentation for NFL tracking data.
"""

from .preprocessing import (
    preprocess_pipeline,
    add_basic_features,
    add_ball_relative_features,
    add_temporal_features,
    add_geometric_features,
    prepare_sequences,
    height_to_feet,
    get_velocity,
    compute_geometric_endpoint,
)

from .augmentation import (
    horizontal_flip_dataframe,
    unflip_predictions,
    speed_perturbation,
    time_warp,
    augment_training_data,
    apply_tta,
)

__all__ = [
    # Preprocessing
    'preprocess_pipeline',
    'add_basic_features',
    'add_ball_relative_features',
    'add_temporal_features',
    'add_geometric_features',
    'prepare_sequences',
    'height_to_feet',
    'get_velocity',
    'compute_geometric_endpoint',

    # Augmentation
    'horizontal_flip_dataframe',
    'unflip_predictions',
    'speed_perturbation',
    'time_warp',
    'augment_training_data',
    'apply_tta',
]
