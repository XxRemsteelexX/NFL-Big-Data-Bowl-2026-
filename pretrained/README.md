# Pretrained Model Weights

This directory contains pretrained weights for all models.

**Note**: Due to file size limitations, weights are not stored in Git. Download them from Kaggle datasets.

---

##  Download Instructions

### Option 1: Using Download Script (Recommended)

```bash
# From repository root
python scripts/download_pretrained.py --all

# Or download specific models
python scripts/download_pretrained.py --models 6layer_st gru_seed27
```

### Option 2: Manual Download from Kaggle

#### Prerequisites:
1. Install Kaggle API: `pip install kaggle`
2. Configure credentials: `~/.kaggle/kaggle.json`
3. See: https://github.com/Kaggle/kaggle-api

#### Download Commands:

```bash
# ST Transformer (6-Layer)
kaggle datasets download -d gdalbey/6layer-seed700-flip-only -p pretrained/6layer_st_20fold --unzip

# Multiscale CNN
kaggle datasets download -d gdalbey/st-multiscale-cnn-w10-20fold -p pretrained/multiscale_cnn_20fold --unzip

# GRU (Seed 27)
kaggle datasets download -d gdalbey/gru-w9-seed27-20fold -p pretrained/gru_seed27_20fold --unzip

# Position-Specific ST
kaggle datasets download -d gdalbey/nfl-bdb-2026-position-st-combined -p pretrained/position_st_combined --unzip

# Geometric Network
kaggle datasets download -d gdalbey/geo-w9-h64-b96-lr3e4 -p pretrained/geometric_w9_5fold --unzip
```

**TODO**: Replace `gdalbey` with your actual Kaggle username.

---

##  Expected Structure After Download

```
pretrained/
├── README.md (this file)
│
├── 6layer_st_20fold/
│   ├── model_fold1.pt ... model_fold20.pt
│   ├── scaler_fold1.pkl ... scaler_fold20.pkl
│   ├── cv_results.json
│   └── *.py (model code from submission)
│
├── multiscale_cnn_20fold/
│   ├── model_fold1.pt ... model_fold20.pt
│   ├── scaler_fold*.pkl
│   └── cv_results.json
│
├── gru_seed27_20fold/
│   ├── model_fold_0.pt ... model_fold_19.pt
│   ├── scaler_fold_0.pkl ... scaler_fold_19.pkl
│   ├── route_kmeans.pkl
│   ├── route_scaler.pkl
│   └── cv_results.json
│
├── position_st_combined/
│   ├── wr/
│   │   └── fold1/ ... fold5/
│   │       ├── model.pt
│   │       ├── scaler.pkl
│   │       └── features.json
│   ├── te/ (same structure)
│   ├── ball_carriers/ (same structure)
│   └── src/*.py
│
└── geometric_w9_5fold/
    ├── model_fold1.pt ... model_fold5.pt
    ├── scaler_fold1.pkl ... scaler_fold5.pkl
    ├── route_kmeans.pkl
    ├── route_scaler.pkl
    └── cv_results.json
```

---

##  Model Details & File Sizes

| Model | Folds | Size (approx) | CV Score | Public LB |
|-------|-------|---------------|----------|-----------|
| 6-Layer ST | 20 | ~300 MB | 0.0750 | 0.547 |
| Multiscale CNN | 20 | ~150 MB | ~0.0751 | 0.548 |
| GRU Seed 27 | 20 | ~50 MB | 0.0798 | 0.557 |
| Position ST | 3×5 | ~200 MB | ~0.075 | 0.553 |
| Geometric | 5 | ~100 MB | 0.0828 | 0.559 |
| **Total** | - | **~800 MB - 1 GB** | - | **0.541 (ensemble)** |

---

##  Verifying Downloads

After downloading, verify files:

```bash
# Check ST Transformer
ls -lh pretrained/6layer_st_20fold/*.pt | wc -l
# Should show: 20

# Check GRU
ls -lh pretrained/gru_seed27_20fold/model_fold_*.pt | wc -l
# Should show: 20

# Check scalers
ls -lh pretrained/*/scaler*.pkl | wc -l
# Should show: 40-60 files (depending on models)
```

---

##  Using Pretrained Models

### Quick Start:

```python
from pathlib import Path
from src.models import create_st_transformer
import torch
import joblib

# Path to pretrained
model_dir = Path('pretrained/6layer_st_20fold')

# Load fold 1
model = create_st_transformer(input_dim=167)
model.load_state_dict(torch.load(model_dir / 'model_fold1.pt'))
scaler = joblib.load(model_dir / 'scaler_fold1.pkl')

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(x_scaled)

print(" Pretrained model loaded and working!")
```

### Load All Folds for Ensemble:

```python
models = []
scalers = []

for fold in range(1, 21):  # 20 folds
    # Load model
    model = create_st_transformer(input_dim=167)
    model.load_state_dict(
        torch.load(model_dir / f'model_fold{fold}.pt')
    )
    model.eval()
    models.append(model)

    # Load scaler
    scaler = joblib.load(model_dir / f'scaler_fold{fold}.pkl')
    scalers.append(scaler)

print(f" Loaded {len(models)} folds for ensemble")
```

See `notebooks/03_inference_ensemble_guide.ipynb` for complete examples.

---

##  Kaggle Dataset Links

Update these with your actual Kaggle datasets:

- **ST Transformer (6L)**: https://www.kaggle.com/datasets/gdalbey/6layer-seed700-flip-only
- **Multiscale CNN**: https://www.kaggle.com/datasets/gdalbey/st-multiscale-cnn-w10-20fold
- **GRU Seed 27**: https://www.kaggle.com/datasets/gdalbey/gru-w9-seed27-20fold
- **Position ST**: https://www.kaggle.com/datasets/gdalbey/nfl-bdb-2026-position-st-combined
- **Geometric**: https://www.kaggle.com/datasets/gdalbey/geo-w9-h64-b96-lr3e4

---

##  Troubleshooting

### Problem: "kaggle: command not found"
```bash
pip install kaggle
```

### Problem: "Unauthorized"
1. Get API token from https://www.kaggle.com/settings
2. Place in `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Problem: "Dataset not found"
- Verify dataset names in `scripts/download_pretrained.py`
- Make sure datasets are public on Kaggle
- Check your Kaggle username

### Problem: "Out of disk space"
- Models total ~1 GB
- Ensure you have at least 2 GB free

---

##  Questions?

See main README.md or open an issue on GitHub.

**Note**: These are the actual models that achieved 0.541 Public LB on Kaggle's NFL Big Data Bowl 2026!

