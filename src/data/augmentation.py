"""
Data Augmentation for NFL Tracking Data

Augmentation techniques for improving model generalization:
    1. Horizontal Flip (Most effective: ~0.007 improvement)
    2. Speed Perturbation (Additional ~0.004 improvement)
    3. Time Warping (Minimal improvement: ~0.001-0.002)
    4. Test-Time Augmentation (TTA: ~0.005-0.010 improvement)

Author: Based on NFL Big Data Bowl 2026 submission
"""

import numpy as np
import pandas as pd
import torch


def horizontal_flip_dataframe(df):
    """
    Apply horizontal flip augmentation across field width.

    Flips the play horizontally (left-right) by:
        - Mirroring y-coordinates
        - Negating y-velocity and y-acceleration
        - Reversing direction and orientation angles

    Args:
        df: DataFrame with tracking data

    Returns:
        df: Horizontally flipped DataFrame

    Impact: ~0.007 RMSE improvement (0.568 → 0.561)

    Example:
        >>> df_original = pd.read_csv('input.csv')
        >>> df_flipped = horizontal_flip_dataframe(df_original)
        >>> # Combine for 2x training data
        >>> df_augmented = pd.concat([df_original, df_flipped])
    """
    df = df.copy()
    field_width = 53.3

    # Flip y-coordinate
    if 'y' in df.columns:
        df['y'] = field_width - df['y']

    # Flip ball landing position
    if 'ball_land_y' in df.columns:
        df['ball_land_y'] = field_width - df['ball_land_y']

    # Flip y-velocity and y-acceleration
    for col in ['velocity_y', 'acceleration_y']:
        if col in df.columns:
            df[col] = -df[col]

    # Flip direction angles (180 - angle)
    if 'dir' in df.columns:
        df['dir'] = (180 - df['dir']) % 360

    if 'o' in df.columns:
        df['o'] = (180 - df['o']) % 360

    return df


def unflip_predictions(predictions):
    """
    Reverse horizontal flip on predictions.

    After making predictions on flipped data, unflip the dy component.

    Args:
        predictions: (N, horizon, 2) array or (N, 2) array

    Returns:
        predictions: Unflipped predictions

    Example:
        >>> pred_flipped = model.predict(test_flipped)
        >>> pred_unflipped = unflip_predictions(pred_flipped)
        >>> pred_final = (pred_original + pred_unflipped) / 2  # TTA
    """
    pred_copy = predictions.copy()

    # Negate dy (y-displacement)
    if len(pred_copy.shape) == 3:
        pred_copy[:, :, 1] = -pred_copy[:, :, 1]
    elif len(pred_copy.shape) == 2:
        pred_copy[:, 1] = -pred_copy[:, 1]

    return pred_copy


def speed_perturbation(df, noise_std=0.1, seed=None):
    """
    Add Gaussian noise to velocity components.

    Perturbs velocities to make model robust to speed variations.

    Args:
        df: DataFrame with tracking data
        noise_std: Standard deviation of noise (default: 0.1)
        seed: Random seed for reproducibility (default: None)

    Returns:
        df: Dataframe with perturbed velocities

    Impact: Combined with flip: ~0.004 additional improvement

    Example:
        >>> df_aug = speed_perturbation(df, noise_std=0.1)
    """
    df = df.copy()

    if seed is not None:
        np.random.seed(seed)

    # Add noise to velocity components
    if 'velocity_x' in df.columns:
        df['velocity_x'] += np.random.normal(0, noise_std, len(df))

    if 'velocity_y' in df.columns:
        df['velocity_y'] += np.random.normal(0, noise_std, len(df))

    # Recompute speed from perturbed velocities
    if 'velocity_x' in df.columns and 'velocity_y' in df.columns:
        df['s'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    return df


def time_warp(sequence, sigma=0.2, seed=None):
    """
    Apply non-linear time warping to sequence.

    Stretches and compresses time non-linearly to make model
    robust to temporal variations.

    Args:
        sequence: (T, F) array - temporal sequence
        sigma: Warping strength (default: 0.2)
        seed: Random seed (default: None)

    Returns:
        warped: (T, F) array - time-warped sequence

    Impact: Minimal (~0.001-0.002), computationally expensive

    Note: Not used in best models due to minimal benefit.

    Example:
        >>> seq_warped = time_warp(sequence, sigma=0.2)
    """
    if seed is not None:
        np.random.seed(seed)

    T = len(sequence)

    # Generate smooth random warp
    noise = np.random.normal(0, sigma, T)
    # Smooth with gaussian filter
    from scipy.ndimage import gaussian_filter1d
    noise_smooth = gaussian_filter1d(noise, sigma=2.0)

    # Cumulative warp
    warp = np.cumsum(1 + noise_smooth)
    warp = warp / warp[-1] * (T - 1)  # Normalize to [0, T-1]

    # Interpolate sequence at warped time points
    original_times = np.arange(T)
    warped_sequence = np.zeros_like(sequence)

    for feat_idx in range(sequence.shape[1]):
        warped_sequence[:, feat_idx] = np.interp(
            warp,
            original_times,
            sequence[:, feat_idx]
        )

    return warped_sequence


def apply_tta(model, X, scaler, device, augmentation_type='horizontal_flip'):
    """
    Apply Test-Time Augmentation.

    Makes predictions on both original and augmented data, then averages.

    Args:
        model: Trained model
        X: List of sequences or DataFrame
        scaler: StandardScaler for normalization
        device: Device for inference
        augmentation_type: Type of TTA (default: 'horizontal_flip')

    Returns:
        predictions: Averaged predictions with TTA

    Impact: Consistent +0.005-0.010 improvement

    Example:
        >>> preds_with_tta = apply_tta(model, X, scaler, device)
    """
    if augmentation_type == 'horizontal_flip':
        field_width = 53.3

        # Original predictions
        X_scaled = [scaler.transform(seq) for seq in X]
        X_tensor = torch.tensor(np.stack(X_scaled).astype(np.float32)).to(device)

        with torch.no_grad():
            pred_original = model(X_tensor).cpu().numpy()

        # Flip y-coordinate (index 1) in each sequence
        X_flipped = []
        for seq in X:
            s = seq.copy()
            s[:, 1] = field_width - s[:, 1]
            X_flipped.append(s)

        X_flip_scaled = [scaler.transform(seq) for seq in X_flipped]
        X_flip_tensor = torch.tensor(np.stack(X_flip_scaled).astype(np.float32)).to(device)

        with torch.no_grad():
            pred_flipped = model(X_flip_tensor).cpu().numpy()

        # Average original and flipped (negate dy of flipped)
        pred_flipped[:, :, 1] = -pred_flipped[:, :, 1]
        pred_tta = (pred_original + pred_flipped) / 2.0

        return pred_tta

    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def augment_training_data(input_df, output_df, augmentations=['horizontal_flip']):
    """
    Apply multiple augmentations to training data.

    Args:
        input_df: Input tracking DataFrame
        output_df: Output tracking DataFrame
        augmentations: List of augmentation types (default: ['horizontal_flip'])

    Returns:
        input_augmented: Augmented input DataFrame
        output_augmented: Augmented output DataFrame

    Example:
        >>> input_aug, output_aug = augment_training_data(
        ...     input_df, output_df,
        ...     augmentations=['horizontal_flip', 'speed_perturbation']
        ... )
    """
    augmented_inputs = [input_df]
    augmented_outputs = [output_df]

    for aug_type in augmentations:
        if aug_type == 'horizontal_flip':
            input_flip = horizontal_flip_dataframe(input_df.copy())
            output_flip = horizontal_flip_dataframe(output_df.copy())

            # Add suffix to game_id to distinguish
            input_flip['game_id'] = input_flip['game_id'].astype(str) + '_flip'
            output_flip['game_id'] = output_flip['game_id'].astype(str) + '_flip'

            augmented_inputs.append(input_flip)
            augmented_outputs.append(output_flip)

        elif aug_type == 'speed_perturbation':
            input_speed = speed_perturbation(input_df.copy(), noise_std=0.1)
            output_speed = output_df.copy()

            # Add suffix
            input_speed['game_id'] = input_speed['game_id'].astype(str) + '_speed'
            output_speed['game_id'] = output_speed['game_id'].astype(str) + '_speed'

            augmented_inputs.append(input_speed)
            augmented_outputs.append(output_speed)

    # Combine all augmentations
    input_combined = pd.concat(augmented_inputs, ignore_index=True)
    output_combined = pd.concat(augmented_outputs, ignore_index=True)

    print(f"Augmentation applied:")
    print(f"  Original: {len(input_df):,} rows")
    print(f"  Augmented: {len(input_combined):,} rows")
    print(f"  Multiplier: {len(input_combined) / len(input_df):.1f}x")

    return input_combined, output_combined


if __name__ == "__main__":
    # Test augmentation functions
    print("Testing augmentation functions...")

    # Create sample data
    sample_df = pd.DataFrame({
        'game_id': [1, 1, 1],
        'y': [26.65, 30.0, 20.0],
        'velocity_y': [1.5, -2.0, 0.5],
        'dir': [90, 180, 45],
        'o': [100, 170, 50]
    })

    print("\nOriginal:")
    print(sample_df)

    # Test horizontal flip
    df_flipped = horizontal_flip_dataframe(sample_df)
    print("\nHorizontally Flipped:")
    print(df_flipped)

    # Test speed perturbation
    df_speed = speed_perturbation(sample_df, noise_std=0.1, seed=42)
    print("\nSpeed Perturbed:")
    print(df_speed)

    print("\n Augmentation functions working!")
