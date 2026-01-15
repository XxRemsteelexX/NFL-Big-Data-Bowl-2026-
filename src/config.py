"""
Configuration Module

Central configuration for all models and training.

Author: NFL Big Data Bowl 2026
"""

from pathlib import Path
import torch


class Config:
    """
    Default configuration for NFL Big Data Bowl 2026 models.

    All models use these default settings unless overridden.
    """

    # ========== Data Settings ==========
    DATA_DIR = Path("/mnt/raid0/Kaggle Big Data Bowl/data/raw")
    WINDOW_SIZE = 10  # frames of history
    MAX_FUTURE_HORIZON = 94  # max future frames to predict

    # Field dimensions
    FIELD_X_MAX = 120.0  # yards
    FIELD_Y_MAX = 53.3  # yards

    # ========== Model Settings ==========
    # ST Transformer
    HIDDEN_DIM = 128
    N_LAYERS = 6  # transformer layers
    N_HEADS = 8  # attention heads
    N_QUERYS = 2  # pooling queries
    MLP_HIDDEN_DIM = 256
    N_RES_BLOCKS = 2

    # GRU
    GRU_HIDDEN_DIM = 64
    GRU_NUM_LAYERS = 2

    # CNN
    CNN_N_LAYERS = 2  # lighter transformer for CNN model

    # ========== Training Settings ==========
    N_FOLDS = 20  # cross-validation folds (use 5 for quick demo)
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    PATIENCE = 15  # early stopping patience
    GRAD_CLIP = 1.0  # gradient clipping

    # Optimizer
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = "AdamW"

    # Scheduler
    SCHEDULER = "ReduceLROnPlateau"  # or "CosineAnnealing"
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    # Loss
    LOSS_DELTA = 0.5  # Huber delta
    LOSS_TIME_DECAY = 0.03  # temporal decay weight
    LOSS_SMOOTHNESS = 0.01  # smoothness regularization (optional)

    # ========== Augmentation Settings ==========
    HORIZONTAL_FLIP = True
    SPEED_PERTURBATION = False  # typically only for some models
    SPEED_NOISE_STD = 0.1
    TIME_WARP = False  # not used in best models
    TIME_WARP_SIGMA = 0.2

    # Test-Time Augmentation
    USE_TTA = True
    TTA_AUGMENTATIONS = ['horizontal_flip']

    # ========== Ensemble Settings ==========
    ENSEMBLE_WEIGHTS = {
        'st_transformer': 0.2517,
        'multiscale_cnn': 0.2517,
        'position_st': 0.2490,
        'gru': 0.2476
    }

    # ========== Output Settings ==========
    OUTPUT_DIR = Path("./models")
    SAVE_DIR = OUTPUT_DIR  # alias

    # Logging
    LOG_INTERVAL = 10  # log every N epochs
    SAVE_BEST_ONLY = True

    # ========== Device Settings ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== Feature Groups ==========
    # Feature groups for modular feature engineering
    FEATURE_GROUPS = [
        "basic",  # x, y, s, a, dir, o
        "velocity",  # velocity_x, velocity_y
        "acceleration",  # acceleration_x, acceleration_y
        "ball_relative",  # distance_to_ball, angle_to_ball
        "player_attributes",  # height, weight, BMI
        "roles",  # is_receiver, is_passer, etc.
        "temporal",  # lag features, rolling stats
        "geometric",  # geometric endpoint features
    ]

    # ========== Paths ==========
    @classmethod
    def set_data_dir(cls, path):
        """Update data directory."""
        cls.DATA_DIR = Path(path)

    @classmethod
    def set_output_dir(cls, path):
        """Update output directory."""
        cls.OUTPUT_DIR = Path(path)
        cls.SAVE_DIR = cls.OUTPUT_DIR

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print(f"\nData:")
        print(f"  DATA_DIR: {cls.DATA_DIR}")
        print(f"  WINDOW_SIZE: {cls.WINDOW_SIZE}")
        print(f"  MAX_FUTURE_HORIZON: {cls.MAX_FUTURE_HORIZON}")

        print(f"\nModel:")
        print(f"  HIDDEN_DIM: {cls.HIDDEN_DIM}")
        print(f"  N_LAYERS: {cls.N_LAYERS}")
        print(f"  N_HEADS: {cls.N_HEADS}")

        print(f"\nTraining:")
        print(f"  N_FOLDS: {cls.N_FOLDS}")
        print(f"  BATCH_SIZE: {cls.BATCH_SIZE}")
        print(f"  LEARNING_RATE: {cls.LEARNING_RATE}")
        print(f"  EPOCHS: {cls.EPOCHS}")
        print(f"  PATIENCE: {cls.PATIENCE}")

        print(f"\nAugmentation:")
        print(f"  HORIZONTAL_FLIP: {cls.HORIZONTAL_FLIP}")
        print(f"  USE_TTA: {cls.USE_TTA}")

        print(f"\nDevice:")
        print(f"  DEVICE: {cls.DEVICE}")
        print("=" * 80)


# Model-specific configs

class STTransformerConfig(Config):
    """Configuration for ST Transformer (6-Layer)."""
    N_LAYERS = 6
    HIDDEN_DIM = 128
    BATCH_SIZE = 256
    WINDOW_SIZE = 10


class MultiScaleCNNConfig(Config):
    """Configuration for Multiscale CNN."""
    N_LAYERS = 2  # lighter transformer
    HIDDEN_DIM = 128
    BATCH_SIZE = 256
    WINDOW_SIZE = 10


class GRUConfig(Config):
    """Configuration for GRU model."""
    GRU_HIDDEN_DIM = 64
    GRU_NUM_LAYERS = 2
    WINDOW_SIZE = 9  # GRU uses W9
    BATCH_SIZE = 128


class GeometricConfig(Config):
    """Configuration for Geometric model."""
    HIDDEN_DIM = 64
    BATCH_SIZE = 96
    LEARNING_RATE = 3e-4
    WINDOW_SIZE = 9


if __name__ == "__main__":
    # Test config
    print("Testing Config...")

    Config.print_config()

    print("\n Config module working!")
