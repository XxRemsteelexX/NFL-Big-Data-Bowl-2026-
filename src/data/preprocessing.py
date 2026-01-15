"""
Data Preprocessing and Feature Engineering

Comprehensive feature engineering pipeline for NFL player trajectory prediction.

Features (167 total):
    - Basic tracking: x, y, s, a, dir, o
    - Derived velocity: velocity_x, velocity_y
    - Derived acceleration: acceleration_x, acceleration_y
    - Ball-relative: distance_to_ball, angle_to_ball, closing_speed
    - Player attributes: height, weight, BMI, position encodings
    - Temporal: lag features (1-5 frames), rolling statistics
    - Opponent tracking: nearest distance, closing speed
    - Route patterns: K-means clustering on trajectory shapes
    - GNN embeddings: Neighbor interactions
    - Geometric: endpoint prediction, velocity errors

Author: Based on NFL Big Data Bowl 2026 submission
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def height_to_feet(height_str):
    """
    Convert height string (e.g., '6-2') to feet.

    Args:
        height_str: Height in format 'feet-inches'

    Returns:
        height_feet: Height in feet (e.g., 6.167)
    """
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches / 12.0
    except:
        return 6.0  # Default height


def get_velocity(speed, direction_deg):
    """
    Convert speed and direction to velocity components.

    Args:
        speed: Speed in yards/second
        direction_deg: Direction in degrees (0-360)

    Returns:
        vx, vy: Velocity components
    """
    theta = np.deg2rad(direction_deg)
    vx = speed * np.sin(theta)
    vy = speed * np.cos(theta)
    return vx, vy


def add_basic_features(df):
    """
    Add basic derived features.

    Features added:
        - velocity_x, velocity_y (from speed & direction)
        - acceleration_x, acceleration_y
        - player_height_feet, height_inches, BMI
        - speed_squared, accel_magnitude
        - momentum_x, momentum_y, kinetic_energy
        - orientation_diff (between body angle and movement direction)
        - Role indicators: is_receiver, is_passer, is_coverage, etc.

    Args:
        df: Input DataFrame

    Returns:
        df: DataFrame with added features
    """
    df = df.copy()

    # Player physical attributes
    df['player_height_feet'] = df['player_height'].apply(height_to_feet)
    height_parts = df['player_height'].str.split('-', expand=True)
    df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    df['bmi'] = (df['player_weight'] / (df['height_inches']**2)) * 703

    # Velocity components
    dir_rad = np.deg2rad(df['dir'].fillna(0))
    df['velocity_x'] = df['s'] * np.sin(dir_rad)
    df['velocity_y'] = df['s'] * np.cos(dir_rad)

    # Acceleration components
    df['acceleration_x'] = df['a'] * np.cos(dir_rad)
    df['acceleration_y'] = df['a'] * np.sin(dir_rad)

    # Derived motion features
    df['speed_squared'] = df['s'] ** 2
    df['accel_magnitude'] = np.sqrt(
        df['acceleration_x']**2 + df['acceleration_y']**2
    )

    # Physics features
    df['momentum_x'] = df['velocity_x'] * df['player_weight']
    df['momentum_y'] = df['velocity_y'] * df['player_weight']
    df['kinetic_energy'] = 0.5 * df['player_weight'] * df['speed_squared']

    # Orientation difference (body angle vs movement direction)
    df['orientation_diff'] = np.minimum(
        np.abs(df['o'] - df['dir']),
        360 - np.abs(df['o'] - df['dir'])
    )

    # Role indicators
    df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
    df['is_defense'] = (df['player_side'] == 'Defense').astype(int)
    df['is_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(int)
    df['is_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(int)
    df['is_passer'] = (df['player_role'] == 'Passer').astype(int)

    return df


def add_ball_relative_features(df):
    """
    Add features relative to ball landing position.

    Features added:
        - distance_to_ball, dist_squared
        - angle_to_ball
        - ball_direction_x, ball_direction_y (unit vectors)
        - closing_speed_ball (velocity toward ball)
        - velocity_toward_ball
        - velocity_alignment (cos of angle difference)

    Args:
        df: DataFrame with ball_land_x and ball_land_y columns

    Returns:
        df: DataFrame with ball-relative features
    """
    df = df.copy()

    if 'ball_land_x' not in df.columns:
        return df

    # Vector from player to ball
    ball_dx = df['ball_land_x'] - df['x']
    ball_dy = df['ball_land_y'] - df['y']

    # Distance
    df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
    df['dist_to_ball'] = df['distance_to_ball']  # Alias
    df['dist_squared'] = df['distance_to_ball'] ** 2

    # Angle
    df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)

    # Direction unit vectors
    df['ball_direction_x'] = ball_dx / (df['distance_to_ball'] + 1e-6)
    df['ball_direction_y'] = ball_dy / (df['distance_to_ball'] + 1e-6)

    # Closing speed (velocity toward ball)
    df['closing_speed_ball'] = (
        df['velocity_x'] * df['ball_direction_x'] +
        df['velocity_y'] * df['ball_direction_y']
    )

    # Velocity alignment
    dir_rad = np.deg2rad(df['dir'].fillna(0))
    df['velocity_toward_ball'] = (
        df['velocity_x'] * np.cos(df['angle_to_ball']) +
        df['velocity_y'] * np.sin(df['angle_to_ball'])
    )
    df['velocity_alignment'] = np.cos(df['angle_to_ball'] - dir_rad)

    # Angle difference
    df['angle_diff'] = np.minimum(
        np.abs(df['o'] - np.degrees(df['angle_to_ball'])),
        360 - np.abs(df['o'] - np.degrees(df['angle_to_ball']))
    )

    return df


def add_temporal_features(df, lags=[1, 2, 3, 4, 5], windows=[3, 5]):
    """
    Add temporal lag and rolling features.

    Features added:
        - Lag features: x_lag1, x_lag2, ..., x_lag5 (for x, y, velocity, speed, accel)
        - Rolling mean: x_rolling_mean_3, x_rolling_mean_5
        - Rolling std: x_rolling_std_3, x_rolling_std_5
        - Velocity delta: velocity_x_delta, velocity_y_delta
        - EMA: velocity_x_ema, velocity_y_ema, speed_ema

    Args:
        df: DataFrame with tracking data
        lags: List of lag values (default: [1,2,3,4,5])
        windows: List of rolling window sizes (default: [3,5])

    Returns:
        df: DataFrame with temporal features
    """
    df = df.copy()

    group_cols = ['game_id', 'play_id', 'nfl_id']

    # Lag features
    for lag in lags:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(group_cols)[col].shift(lag)

    # Rolling statistics
    for window in windows:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in df.columns:
                rolling = df.groupby(group_cols)[col].rolling(
                    window, min_periods=1
                )
                df[f'{col}_rolling_mean_{window}'] = rolling.mean().reset_index(
                    level=[0, 1, 2], drop=True
                )
                df[f'{col}_rolling_std_{window}'] = rolling.std().reset_index(
                    level=[0, 1, 2], drop=True
                )

    # Velocity changes
    for col in ['velocity_x', 'velocity_y']:
        if col in df.columns:
            df[f'{col}_delta'] = df.groupby(group_cols)[col].diff()

    # Exponential moving averages
    df['velocity_x_ema'] = df.groupby(group_cols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    df['velocity_y_ema'] = df.groupby(group_cols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    df['speed_ema'] = df.groupby(group_cols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )

    return df


def prepare_sequences(
    input_df,
    output_df,
    window_size=10,
    feature_cols=None
):
    """
    Prepare sequences for model training/inference.

    Creates sliding windows of input data and corresponding targets.

    Args:
        input_df: Input tracking DataFrame
        output_df: Output tracking DataFrame
        window_size: Number of frames in input window (default: 10)
        feature_cols: List of feature columns to use (default: auto-detect)

    Returns:
        sequences: List of (window_size, num_features) arrays
        targets: List of (horizon, 2) arrays with (dx, dy)
        metadata: List of dicts with game_id, play_id, nfl_id, last positions

    Example:
        >>> sequences, targets, metadata = prepare_sequences(
        ...     input_df, output_df, window_size=10
        ... )
    """
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        # Use all numeric columns except identifiers
        exclude_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
        feature_cols = [
            col for col in input_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(input_df[col])
        ]

    sequences = []
    targets = []
    metadata = []

    # Group by player trajectory
    grouped = input_df.groupby(['game_id', 'play_id', 'nfl_id'])

    for (game_id, play_id, nfl_id), group in grouped:
        # Sort by frame
        group = group.sort_values('frame_id')

        # Take last window_size frames
        input_window = group.tail(window_size)

        # Pad if needed
        if len(input_window) < window_size:
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(
                np.nan,
                index=range(pad_len),
                columns=input_window.columns
            )
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        # Fill NaN with column means
        input_window = input_window.fillna(input_window.mean(numeric_only=True))

        # Extract features
        seq = input_window[feature_cols].fillna(0).values

        # Get output trajectory
        output_traj = output_df[
            (output_df['game_id'] == game_id) &
            (output_df['play_id'] == play_id) &
            (output_df['nfl_id'] == nfl_id)
        ].sort_values('frame_id')

        if len(output_traj) == 0:
            continue

        # Last position
        last_x = input_window['x'].iloc[-1]
        last_y = input_window['y'].iloc[-1]

        # Compute displacements (dx, dy)
        dx = output_traj['x'].values - last_x
        dy = output_traj['y'].values - last_y

        # Create target array
        target = np.column_stack([dx, dy])

        sequences.append(seq)
        targets.append(target)
        metadata.append({
            'game_id': game_id,
            'play_id': play_id,
            'nfl_id': nfl_id,
            'last_x': last_x,
            'last_y': last_y
        })

    return sequences, targets, metadata


def preprocess_pipeline(
    input_df,
    output_df,
    window_size=10,
    add_temporal=True,
    add_ball_features=True
):
    """
    Complete preprocessing pipeline.

    Applies all feature engineering steps in sequence:
        1. Basic features (velocity, acceleration, etc.)
        2. Ball-relative features (distance, angle, etc.)
        3. Temporal features (lags, rolling stats, EMA)
        4. Prepare sequences for model input

    Args:
        input_df: Input tracking DataFrame
        output_df: Output tracking DataFrame
        window_size: Sequence window size (default: 10)
        add_temporal: Add temporal lag/rolling features (default: True)
        add_ball_features: Add ball-relative features (default: True)

    Returns:
        sequences: List of input sequences
        targets: List of target trajectories
        metadata: List of metadata dicts
        feature_cols: List of feature column names

    Example:
        >>> sequences, targets, metadata, feature_cols = preprocess_pipeline(
        ...     input_df, output_df, window_size=10
        ... )
    """
    print("Preprocessing pipeline...")

    # Step 1: Basic features
    print("  1. Adding basic features...")
    df = add_basic_features(input_df)

    # Step 2: Ball-relative features
    if add_ball_features:
        print("  2. Adding ball-relative features...")
        df = add_ball_relative_features(df)

    # Step 3: Temporal features
    if add_temporal:
        print("  3. Adding temporal features...")
        df = add_temporal_features(df)

    # Step 4: Prepare sequences
    print("  4. Preparing sequences...")
    sequences, targets, metadata = prepare_sequences(
        df, output_df, window_size=window_size
    )

    # Get feature columns
    feature_cols = [
        col for col in df.columns
        if col not in ['game_id', 'play_id', 'nfl_id', 'frame_id']
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f" Preprocessing complete")
    print(f"  Sequences: {len(sequences):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Window size: {window_size}")

    return sequences, targets, metadata, feature_cols


# Geometric features (from ensemble notebook)

def compute_geometric_endpoint(df):
    """
    Compute geometric endpoint prediction for players.

    Predicts where players will be based on constant velocity assumption,
    with special handling for targeted receivers and coverage defenders.

    Args:
        df: DataFrame with tracking data

    Returns:
        df: DataFrame with geometric endpoint features
    """
    df = df.copy()

    # Time to endpoint
    if 'num_frames_output' in df.columns:
        t_total = df['num_frames_output'] / 10.0  # frames to seconds
    else:
        t_total = 3.0  # default

    df['time_to_endpoint'] = t_total

    # Constant velocity prediction
    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total

    # Special handling for receivers: endpoint = ball landing
    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']

    # Clip to field bounds
    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(0.0, 120.0)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(0.0, 53.3)

    return df


def add_geometric_features(df):
    """
    Add geometric trajectory features.

    Features added:
        - geo_endpoint_x, geo_endpoint_y (predicted endpoint)
        - geo_vector_x, geo_vector_y (vector to endpoint)
        - geo_distance (distance to endpoint)
        - geo_required_vx, geo_required_vy (required velocity)
        - geo_velocity_error (difference from current velocity)
        - geo_required_ax, geo_required_ay (required acceleration)
        - geo_alignment (velocity alignment with endpoint direction)

    Args:
        df: DataFrame with tracking data

    Returns:
        df: DataFrame with geometric features
    """
    df = compute_geometric_endpoint(df)

    # Vector to endpoint
    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)

    # Required velocity to reach endpoint
    t = df['time_to_endpoint'] + 0.1  # Avoid division by zero
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t

    # Velocity error
    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(
        df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2
    )

    # Required acceleration
    t_sq = t * t
    df['geo_required_ax'] = (2 * df['geo_vector_x'] / t_sq).clip(-10, 10)
    df['geo_required_ay'] = (2 * df['geo_vector_y'] / t_sq).clip(-10, 10)

    # Velocity alignment with endpoint direction
    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (
        df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y
    ) / (velocity_mag + 0.1)

    return df


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing functions...")

    # Create sample data
    sample_input = pd.DataFrame({
        'game_id': [1] * 20,
        'play_id': [1] * 20,
        'nfl_id': [101] * 10 + [102] * 10,
        'frame_id': list(range(10)) * 2,
        'x': np.random.uniform(0, 120, 20),
        'y': np.random.uniform(0, 53.3, 20),
        's': np.random.uniform(0, 10, 20),
        'a': np.random.uniform(-2, 2, 20),
        'dir': np.random.uniform(0, 360, 20),
        'o': np.random.uniform(0, 360, 20),
        'player_height': ['6-2'] * 20,
        'player_weight': [200] * 20,
        'player_side': ['Offense'] * 10 + ['Defense'] * 10,
        'player_role': ['Targeted Receiver'] * 5 + ['Other'] * 15,
        'ball_land_x': [60] * 20,
        'ball_land_y': [26.65] * 20
    })

    sample_output = pd.DataFrame({
        'game_id': [1] * 20,
        'play_id': [1] * 20,
        'nfl_id': [101] * 10 + [102] * 10,
        'frame_id': list(range(10, 20)) * 2,
        'x': np.random.uniform(0, 120, 20),
        'y': np.random.uniform(0, 53.3, 20)
    })

    # Test preprocessing
    sequences, targets, metadata, feature_cols = preprocess_pipeline(
        sample_input, sample_output, window_size=5
    )

    print(f"\n Preprocessing test passed!")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Sample sequence shape: {sequences[0].shape}")
    print(f"  Sample target shape: {targets[0].shape}")
