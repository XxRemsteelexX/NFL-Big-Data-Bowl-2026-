#!/usr/bin/env python3
"""
Generate README figures for NFL Big Data Bowl 2026.

Creates publication-quality charts from competition results data.

Usage:
    python scripts/generate_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


# Output directory
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Color palette
COLORS = {
    'ensemble': '#1a73e8',
    'transformer': '#ea4335',
    'cnn': '#fbbc04',
    'rnn': '#34a853',
    'geometric': '#9334e6',
    'accent': '#ff6d01',
    'grid': '#e0e0e0',
    'text': '#202124',
    'bg': '#ffffff',
}


def setup_style():
    """Apply consistent plot style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.facecolor': COLORS['bg'],
        'figure.facecolor': COLORS['bg'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': COLORS['grid'],
    })


def generate_model_performance():
    """Horizontal bar chart of all model Public LB scores."""
    models = [
        '3-Model Ensemble',
        '4-Model Ensemble',
        '6L ST Transformer',
        'Multiscale CNN',
        'Position-Specific ST',
        'GRU (Seed 27)',
        'Geometric Network',
    ]
    scores = [0.540, 0.541, 0.547, 0.548, 0.553, 0.557, 0.559]
    colors = [
        COLORS['ensemble'],
        COLORS['ensemble'],
        COLORS['transformer'],
        COLORS['cnn'],
        COLORS['transformer'],
        COLORS['rnn'],
        COLORS['geometric'],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor='white', linewidth=0.5)

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() - 0.001, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', ha='right', fontweight='bold',
                color='white', fontsize=11)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Public Leaderboard Score (lower is better)', fontsize=12)
    ax.set_title('Model Performance Comparison', pad=15)
    ax.set_xlim(0.535, 0.565)
    ax.axvline(x=0.540, color=COLORS['ensemble'], linestyle='--', alpha=0.4, linewidth=1)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['ensemble'], label='Ensemble'),
        mpatches.Patch(color=COLORS['transformer'], label='Transformer'),
        mpatches.Patch(color=COLORS['cnn'], label='CNN + Transformer'),
        mpatches.Patch(color=COLORS['rnn'], label='RNN (GRU)'),
        mpatches.Patch(color=COLORS['geometric'], label='Geometric'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = FIGURES_DIR / 'model_performance.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def generate_augmentation_impact():
    """Bar chart showing progressive improvement from augmentations."""
    augmentations = ['None\n(Baseline)', 'Flip\nOnly', 'Flip +\nSpeed', 'Flip + Speed\n+ TTA']
    scores = [0.568, 0.561, 0.557, 0.552]
    deltas = [0, -0.007, -0.011, -0.016]

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(augmentations))
    bar_colors = [COLORS['grid'], COLORS['cnn'], COLORS['accent'], COLORS['transformer']]
    bars = ax.bar(x_pos, scores, color=bar_colors, width=0.6, edgecolor='white', linewidth=0.5)

    # Add score labels on bars
    for bar, score, delta in zip(bars, scores, deltas):
        label = f'{score:.3f}'
        if delta < 0:
            label += f'\n({delta:+.3f})'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=COLORS['text'])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(augmentations, fontsize=10)
    ax.set_ylabel('Public LB Score (lower is better)', fontsize=12)
    ax.set_title('Impact of Data Augmentation on Model Performance', pad=15)
    ax.set_ylim(0.545, 0.575)

    # Arrow showing total improvement
    ax.annotate('', xy=(3, 0.553), xytext=(0, 0.569),
                arrowprops=dict(arrowstyle='->', color=COLORS['ensemble'],
                                lw=2, connectionstyle='arc3,rad=-0.2'))
    ax.text(1.5, 0.571, 'Total: -0.016 improvement',
            ha='center', fontsize=10, color=COLORS['ensemble'], fontweight='bold')

    plt.tight_layout()
    path = FIGURES_DIR / 'augmentation_impact.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def generate_architecture_depth():
    """Line chart of transformer depth vs score."""
    layers = [2, 4, 6, 8]
    scores = [0.594, 0.552, 0.547, 0.549]
    times = [6, 14, 16, 24]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Score line
    line1 = ax1.plot(layers, scores, 'o-', color=COLORS['transformer'],
                     linewidth=2.5, markersize=10, label='Public LB Score', zorder=5)

    # Highlight optimal point
    ax1.plot(6, 0.547, 'o', color=COLORS['ensemble'], markersize=16, zorder=6, alpha=0.3)
    ax1.annotate('Optimal (6L)', xy=(6, 0.547), xytext=(6.5, 0.555),
                 fontsize=10, fontweight='bold', color=COLORS['ensemble'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['ensemble'], lw=1.5))

    ax1.set_xlabel('Number of Transformer Layers', fontsize=12)
    ax1.set_ylabel('Public LB Score (lower is better)', fontsize=12,
                   color=COLORS['transformer'])
    ax1.set_xticks(layers)
    ax1.set_xticklabels(['2L', '4L', '6L', '8L'], fontsize=11)
    ax1.tick_params(axis='y', labelcolor=COLORS['transformer'])

    # Training time on secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.bar(layers, times, width=0.4, alpha=0.2, color=COLORS['rnn'],
                    label='Training Time', zorder=1)
    ax2.set_ylabel('Training Time (hours, 20-fold)', fontsize=12, color=COLORS['rnn'])
    ax2.tick_params(axis='y', labelcolor=COLORS['rnn'])
    ax2.set_ylim(0, 35)

    ax1.set_title('Transformer Depth vs Performance', pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + [mpatches.Patch(color=COLORS['rnn'], alpha=0.2)],
               labels1 + ['Training Time (hours)'],
               loc='upper right', fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / 'architecture_depth.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def generate_ensemble_weights():
    """Donut chart of 3-model ensemble weights."""
    labels = ['ST Transformer\n(0.547 LB)', 'Multiscale CNN\n(0.548 LB)', 'GRU Seed 27\n(0.557 LB)']
    sizes = [33.3, 33.3, 33.4]
    colors = [COLORS['transformer'], COLORS['cnn'], COLORS['rnn']]

    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        textprops=dict(fontsize=11),
    )

    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    # Center text
    ax.text(0, 0, '3-Model\nEnsemble\n0.540 LB', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['text'])

    ax.set_title('Competition Ensemble Composition', pad=20, fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = FIGURES_DIR / 'ensemble_weights.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def main():
    """Generate all figures."""
    setup_style()

    print('Generating README figures...')
    generate_model_performance()
    generate_augmentation_impact()
    generate_architecture_depth()
    generate_ensemble_weights()
    print(f'\nDone. All figures saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
