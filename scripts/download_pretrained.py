#!/usr/bin/env python3
"""
Download Pretrained Models from Kaggle

Downloads pretrained model weights from Kaggle datasets.

Usage:
    python scripts/download_pretrained.py --all
    python scripts/download_pretrained.py --models 6layer_st gru_seed27
    python scripts/download_pretrained.py --list

Kaggle Datasets (update these with your actual dataset names):
    - 6layer-seed700-flip-only (ST Transformer 6L)
    - st-multiscale-cnn-w10-20fold (Multiscale CNN)
    - gru-w9-seed27-20fold (GRU)
    - nfl-bdb-2026-position-st-combined (Position-ST)

Author: NFL Big Data Bowl 2026
"""

import argparse
import subprocess
from pathlib import Path
import json


# Kaggle dataset mappings
# TODO: Update with your actual Kaggle username and dataset names
KAGGLE_DATASETS = {
    '6layer_st': {
        'dataset': 'gdalbey/6layer-seed700-flip-only',
        'description': '6-Layer ST Transformer (0.547 LB)',
        'files': ['*.pt', '*.pkl', 'cv_results.json', '*.py']
    },
    'multiscale_cnn': {
        'dataset': 'gdalbey/st-multiscale-cnn-w10-20fold',
        'description': 'Multiscale CNN + Transformer (0.548 LB)',
        'files': ['*.pt', '*.pkl', 'cv_results.json']
    },
    'gru_seed27': {
        'dataset': 'gdalbey/gru-w9-seed27-20fold',
        'description': 'GRU with Geometric Features (0.557 LB)',
        'files': ['*.pt', '*.pkl', 'route_*.pkl', 'cv_results.json']
    },
    'position_st': {
        'dataset': 'gdalbey/nfl-bdb-2026-position-st-combined',
        'description': 'Position-Specific ST Models (0.553 LB)',
        'files': ['**/model.pt', '**/scaler.pkl', '**/*.json', '*.py']
    },
    'geometric': {
        'dataset': 'gdalbey/geo-w9-h64-b96-lr3e4',
        'description': 'Geometric Attention Network (0.559 LB)',
        'files': ['*.pt', '*.pkl', 'cv_results.json']
    },
}


def list_datasets():
    """List available datasets."""
    print("Available Pretrained Models:")
    print("=" * 80)
    for name, info in KAGGLE_DATASETS.items():
        print(f"\n{name}:")
        print(f"  Dataset: {info['dataset']}")
        print(f"  Description: {info['description']}")
    print("\n" + "=" * 80)


def download_kaggle_dataset(dataset_name, output_dir):
    """
    Download a Kaggle dataset using kaggle CLI.

    Args:
        dataset_name: Kaggle dataset name (username/dataset-name)
        output_dir: Output directory

    Returns:
        success: True if successful
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {dataset_name}")
    print(f"  → {output_dir}")

    try:
        # Use kaggle CLI
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', dataset_name,
            '-p', str(output_dir),
            '--unzip'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print(f"  Downloaded successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] {e}")
        print(f"  stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  [ERROR] kaggle CLI not found")
        print("  Please install: pip install kaggle")
        print("  And configure: kaggle config")
        return False


def download_model(model_name, output_dir='pretrained'):
    """
    Download a specific model.

    Args:
        model_name: Model name (e.g., '6layer_st')
        output_dir: Base output directory

    Returns:
        success: True if successful
    """
    if model_name not in KAGGLE_DATASETS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available: {list(KAGGLE_DATASETS.keys())}")
        return False

    info = KAGGLE_DATASETS[model_name]
    model_output_dir = Path(output_dir) / model_name

    success = download_kaggle_dataset(info['dataset'], model_output_dir)

    if success:
        # Verify files exist
        print(f"  Verifying files...")
        pt_files = list(model_output_dir.glob('**/*.pt'))
        pkl_files = list(model_output_dir.glob('**/*.pkl'))

        print(f"    Found {len(pt_files)} .pt files")
        print(f"    Found {len(pkl_files)} .pkl files")

        if len(pt_files) == 0:
            print(f"    WARNING: No .pt files found")

    return success


def download_all(output_dir='pretrained'):
    """
    Download all pretrained models.

    Args:
        output_dir: Base output directory
    """
    print("=" * 80)
    print("DOWNLOADING ALL PRETRAINED MODELS")
    print("=" * 80)

    success_count = 0
    total = len(KAGGLE_DATASETS)

    for model_name in KAGGLE_DATASETS.keys():
        if download_model(model_name, output_dir):
            success_count += 1

    print("\n" + "=" * 80)
    print(f"DOWNLOAD COMPLETE: {success_count}/{total} successful")
    print("=" * 80)

    if success_count < total:
        print("\nWARNING: Some downloads failed. Check:")
        print("  1. Kaggle API is installed: pip install kaggle")
        print("  2. Kaggle credentials configured: ~/.kaggle/kaggle.json")
        print("  3. Dataset names are correct (update KAGGLE_DATASETS)")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Download pretrained models from Kaggle'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to download (e.g., 6layer_st gru_seed27)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pretrained',
        help='Output directory (default: pretrained)'
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.all:
        download_all(args.output)
    elif args.models:
        for model_name in args.models:
            download_model(model_name, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
