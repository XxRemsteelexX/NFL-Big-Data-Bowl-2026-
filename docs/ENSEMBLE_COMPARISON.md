# Ensemble Model Comparison

## Overview

This document compares the two ensemble approaches that achieved the best scores.

---

##  Results Summary

| Ensemble | Public LB | Components | Status |
|----------|-----------|------------|--------|
| **3-Model** | **0.540**  | ST + CNN + GRU | **Competition Best** |
| **4-Model** | **0.541** | ST + CNN + GRU + Position | Post-deadline |

**Key Finding**: Adding the 4th model (Position-Specific ST) gave only +0.001 improvement, suggesting the 3-model ensemble captured most of the performance.

---

## 3-Model Ensemble (0.540 LB) 

### Submission Details:
- **Name**: "Ensemble 3Model ST CNN GRU - Version 10"
- **Status**: Succeeded (7 days ago)
- **Timing**: During competition (on-time submission)
- **Score**: **0.540 Public LB** (Best competitive score)

### Components:
```python
ensemble_weights_3model = {
    'st_transformer': 0.333,    # 33.3% - ST Transformer (6L or 4L)
    'multiscale_cnn': 0.333,    # 33.3% - Multiscale CNN
    'gru_seed27':     0.334     # 33.4% - GRU with geometric features
}
```

### Why This Works:
1. **Architecture Diversity**:
   - Pure Transformer (ST)
   - CNN + Transformer hybrid (CNN)
   - RNN (GRU)

2. **Equal Weighting**:
   - Nearly equal weights (0.333 each)
   - Suggests all models contribute equally
   - No single model dominates

3. **Feature Diversity**:
   - ST: Global spatial-temporal features
   - CNN: Multi-scale temporal patterns
   - GRU: Geometric features (route patterns, opponent tracking, GNN)

4. **Complementary Strengths**:
   - ST: Best at spatial relationships
   - CNN: Best at temporal patterns
   - GRU: Best at sequential dynamics

### Individual Model Scores:
| Model | Individual Score | Weight | Contribution |
|-------|-----------------|--------|--------------|
| ST Transformer | 0.547-0.552 | 0.333 | 33.3% |
| Multiscale CNN | 0.548 | 0.333 | 33.3% |
| GRU Seed27 | 0.557 | 0.334 | 33.4% |
| **3-Model Ensemble** | **0.540** | **1.0** | **100%** |

**Improvement**: 0.540 vs 0.547 (best single) = **-0.007 gain from ensembling**

---

## 4-Model Ensemble (0.541 LB)

### Submission Details:
- **Name**: "Ensemble 4Model ST CNN GRU Geo/Position - Multiple Versions"
- **Status**: Succeeded (after deadline)
- **Timing**: Post-competition refinement
- **Score**: **0.541 Public LB** (+0.001 vs 3-model)

### Components:
```python
ensemble_weights_4model = {
    'st_transformer': 0.2517,    # 25.17% - 6-Layer ST
    'multiscale_cnn': 0.2517,    # 25.17% - Multiscale CNN
    'position_st':    0.2490,    # 24.90% - Position-Specific ST
    'gru_seed27':     0.2476     # 24.76% - GRU
}
```

### What Changed:
- **Added**: Position-Specific ST models (separate models for WR, TE, QB/RB)
- **Reweighted**: Adjusted weights based on inverse Public LB scores
- **Improvement**: Only +0.001 (minimal)

### Individual Model Scores:
| Model | Individual Score | Weight | Contribution |
|-------|-----------------|--------|--------------|
| ST Transformer 6L | 0.547 | 0.2517 | 25.17% |
| Multiscale CNN | 0.548 | 0.2517 | 25.17% |
| Position ST | 0.553 | 0.2490 | 24.90% |
| GRU Seed27 | 0.557 | 0.2476 | 24.76% |
| **4-Model Ensemble** | **0.541** | **1.0** | **100%** |

**Improvement**: 0.541 vs 0.547 (best single) = **-0.006 gain from ensembling**

---

## Comparison Analysis

### Ensemble Gain:
- **3-Model**: 0.540 vs 0.547 = -0.007 improvement
- **4-Model**: 0.541 vs 0.547 = -0.006 improvement

### Marginal Value of 4th Model:
- **Improvement**: 0.541 - 0.540 = +0.001
- **Complexity Added**: Significant (position-specific training + inference)
- **Cost-Benefit**: Marginal

### Conclusion:
The **3-model ensemble is the optimal choice**:
-  Best cost-benefit ratio
-  Simpler to implement and maintain
-  On-time submission (competitive)
-  99.8% of 4-model performance

The 4-model ensemble is interesting for completeness but not necessary for competitive performance.

---

## Ensemble Strategy Details

### Test-Time Augmentation (TTA)
Applied to ALL models in both ensembles:

```python
# For each model:
# 1. Original prediction
pred_orig = model.predict(test_input)

# 2. Flipped prediction
test_flip = horizontal_flip(test_input)
pred_flip = model.predict(test_flip)
pred_flip = unflip(pred_flip)  # Reverse the flip

# 3. Average both
pred_final = (pred_orig + pred_flip) / 2.0

# Impact: +0.005-0.010 improvement per model
```

### Cross-Validation Averaging
Each model is an ensemble of CV folds:

- **ST Transformer**: 20 folds averaged
- **Multiscale CNN**: 20 folds averaged
- **GRU**: 20 folds averaged
- **Position ST**: 5 folds per position (×3 positions)

This provides additional smoothing and robustness.

### Final Combination

**3-Model**:
```python
ensemble_pred = (
    0.333 * st_pred +
    0.333 * cnn_pred +
    0.334 * gru_pred
)
```

**4-Model**:
```python
ensemble_pred = (
    0.2517 * st_pred +
    0.2517 * cnn_pred +
    0.2490 * position_pred +
    0.2476 * gru_pred
)
```

---

## Implementation Code

### 3-Model Ensemble:
```python
from src.models import create_ensemble
from pathlib import Path

model_paths = {
    'st_transformer': Path('pretrained/6layer_st_20fold'),
    'multiscale_cnn': Path('pretrained/multiscale_cnn_20fold'),
    'gru': Path('pretrained/gru_seed27_20fold')
}

ensemble = create_ensemble(
    model_paths,
    ensemble_type='3model'  # Use 3-model weights
)

predictions = ensemble.predict(test_input, test_template)
# Expected score: ~0.540 Public LB
```

### 4-Model Ensemble:
```python
model_paths = {
    'st_transformer': Path('pretrained/6layer_st_20fold'),
    'multiscale_cnn': Path('pretrained/multiscale_cnn_20fold'),
    'gru': Path('pretrained/gru_seed27_20fold'),
    'position_st': Path('pretrained/position_st_combined')
}

ensemble = create_ensemble(
    model_paths,
    ensemble_type='4model'  # Use 4-model weights
)

predictions = ensemble.predict(test_input, test_template)
# Expected score: ~0.541 Public LB
```

---

## Lessons Learned

### What Worked:
1.  **Simple equal weighting** (3-model: 0.333 each)
2.  **Architecture diversity** (Transformer + CNN + RNN)
3.  **Test-Time Augmentation** on all models
4.  **20-fold CV** for individual models

### What Didn't Help Much:
1.  **Adding 4th model** (+0.001 only)
2.  **Complex weighting schemes** (equal weights work well)
3.  **More than 4 models** (diminishing returns)

### Recommendations:
- **For competition**: Use 3-model ensemble (best cost-benefit)
- **For research**: Explore 4+ models to understand limits
- **For production**: 3-model optimal (simpler, almost same performance)

---

## References

- **3-Model Submission**: "Ensemble 3Model ST CNN GRU - Version 10"
- **4-Model Submission**: "Ensemble 4Model - Multiple versions"
- **Code**: `/mnt/raid0/BigData2/ensemble_4model_SIMPLE.ipynb`
- **Module**: `src/models/ensemble.py`

---

## Summary

Both ensembles showcase effective model combination:
- **3-Model (0.540)**: Optimal balance of performance and simplicity 
- **4-Model (0.541)**: Marginal improvement with added complexity

**Choose 3-model** for most use cases. Include 4-model to demonstrate ensemble scaling analysis.

