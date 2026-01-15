"""
NFL Big Data Bowl 2026 - Player Trajectory Prediction
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nfl-bdb-2026",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning Models for NFL Player Trajectory Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nfl-big-data-bowl-2026",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nfl-big-data-bowl-2026/issues",
        "Source": "https://github.com/yourusername/nfl-big-data-bowl-2026",
        "Kaggle Competition": "https://www.kaggle.com/competitions/nfl-big-data-bowl-2026",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "polars>=0.18.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "pyyaml>=6.0",
        "gdown>=4.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "ipywidgets>=7.6.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "plotly>=5.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nfl-train=scripts.train.train_st_transformer:main",
            "nfl-predict=scripts.inference.predict:main",
            "nfl-ensemble=scripts.inference.predict_ensemble:main",
            "nfl-download=scripts.download_pretrained:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "nfl",
        "big-data-bowl",
        "deep-learning",
        "trajectory-prediction",
        "pytorch",
        "transformer",
        "time-series",
        "sports-analytics",
    ],
)
