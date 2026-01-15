# 7 Models to Showcase - NFL Big Data Bowl 2026

## Model Selection Summary

These 7 models represent the best and most diverse architectures from 771 completed training runs.

---

## 1. ST Transformer (4-Layer)  FLAGSHIP

**Public LB Score**: 0.552
**CV Score**: 0.0756 (estimated)
**Architecture**: Spatial-Temporal Transformer with 4 encoder layers

### Model Details:
- **Path**: `/mnt/raid0/BigData2/models/st_transformer_w10_b512_lr5e4_l4_20fold_seed42_FLIP`
- **Config**:
  - Window: 10 frames
  - Hidden dim: 128
  - Layers: 4
  - Batch: 512
  - Learning rate: 5e-4
  - Augmentation: Horizontal flip
- **Training**: 20-fold CV, Seed 42
- **Why showcase**: Best balance of performance and training speed, production-ready

### Pretrained Weights:
```
st_transformer_w10_b512_lr5e4_l4_20fold_seed42_FLIP/
├── model_fold_*.pt  (20 folds)
├── scaler_fold_*.pkl  (20 folds)
└── cv_results.json
```

---

## 2. 6-Layer ST Transformer  BEST SINGLE MODEL

**Public LB Score**: 0.547
**CV Score**: 0.0750
**Architecture**: Deep Spatial-Temporal Transformer with frozen fine-tuning

### Model Details:
- **Path**: `/mnt/raid0/BigData2/models/6LAYER_FINETUNE_FROZEN_20fold`
- **Config**:
  - Window: 10 frames
  - Hidden dim: 128
  - Layers: 6
  - Batch: 256
  - Learning rate: 1e-4 (fine-tuning)
  - Pretrained from: 6LAYER_w10_b256_lr1e3_20fold_seed42_FLIP_ONLY
  - **Frozen**: input_proj, pos_embed, transformer_encoder
  - **Trainable**: pool_ln, pool_attn, pool_query, head
- **Training**: 20-fold CV, Seed 42, Frozen encoder fine-tuning
- **Why showcase**: Best single model, demonstrates transfer learning

### Pretrained Weights:
```
6LAYER_FINETUNE_FROZEN_20fold/
├── model_fold_*.pt  (20 folds)
├── scaler_fold_*.pkl  (20 folds)
└── cv_results.json
```

---

## 3. GRU (Seed 27)  RNN BASELINE

**Public LB Score**: 0.557
**CV Score**: 0.0798
**Architecture**: Bidirectional GRU with attention pooling

### Model Details:
- **Path**: `/mnt/raid0/BigData2/models/gru_w9_h64_flip_speed_seed27_20fold`
- **Config**:
  - Window: 9 frames
  - Hidden dim: 64
  - Layers: 2
  - GRU: Bidirectional
  - Augmentation: Horizontal flip + Speed perturbation
- **Training**: 20-fold CV, Seed 27
- **Why showcase**: Simple RNN baseline, fast training, good performance

### Pretrained Weights:
```
gru_w9_h64_flip_speed_seed27_20fold/
├── model_fold_*.pt  (20 folds)
├── scaler_fold_*.pkl  (20 folds)
└── cv_results.json
```

---

## 4. Geometric Attention Network  NOVEL ARCHITECTURE

**Public LB Score**: 0.559
**CV Score**: 0.0828
**Architecture**: Geometric-aware transformer with spatial distance modulation

### Model Details:
- **Path**: `/mnt/raid0/BigData2/models/geo_w9_h64_b96_lr3e4`
- **Config**:
  - Window: 9 frames
  - Hidden dim: 64
  - Batch: 96
  - Learning rate: 3e-4
  - Special features: Geometric relationships, Voronoi cells, relative distances
- **Training**: 5-fold CV (need to check if 20-fold exists)
- **Why showcase**: Unique architecture, novel geometric features

### Pretrained Weights:
```
geo_w9_h64_b96_lr3e4/
├── model_fold*.pt  (5 folds)
├── scaler_fold*.pkl  (5 folds)
└── cv_results.json
```

**Note**: Check for 20-fold version: `geo_w9_h64_b96_augmented_lr3e4_20fold`

---

## 5. Multiscale CNN (4-Layer)  CNN APPROACH

**Public LB Score**: 0.548
**CV Score**: 0.0794
**Architecture**: Multi-scale convolutional network with parallel temporal receptive fields

### Model Details:
- **Path**: `/mnt/raid0/BigData2/models/4L_CNN_Transformer_NO_BAD_PLAY_20fold_FLIP_SPEED`
- **Config**:
  - Window: 9 frames (likely)
  - Hidden dim: 128
  - Layers: 4 convolutional layers
  - Parallel streams: 3x3, 5x5, 7x7 kernels
  - Augmentation: Horizontal flip + Speed perturbation
  - Data: Filtered "bad plays"
- **Training**: 20-fold CV
- **Why showcase**: Different paradigm (CNN vs Transformer), competitive performance

### Pretrained Weights:
```
4L_CNN_Transformer_NO_BAD_PLAY_20fold_FLIP_SPEED/
├── model_fold_*.pt  (20 folds)
├── scaler_fold_*.pkl  (20 folds)
└── cv_results.json
```

---

## 6. Perceiver Co4  ADVANCED ARCHITECTURE

**Public LB Score**: 0.564
**CV Score**: ~0.078 (estimated)
**Architecture**: Perceiver with latent bottleneck and cross-attention

### Model Details:
- **Path**: Need to identify best Co4 model
- **Candidates**:
  - `co4_optimized_20fold_FLIP_SPEED` (if exists)
  - `co4_w9_b64_flip_only` (submitted version)
  - `co4_w8_b64_flip_only`
- **Config**:
  - Window: 8-9 frames
  - Latent dim: 64
  - Num latents: Variable
  - Cross-attention blocks: 4
  - Augmentation: Horizontal flip (or flip+speed)
- **Training**: 20-fold CV preferred
- **Why showcase**: Novel architecture, different approach, good for advanced users

### Pretrained Weights:
```
TBD - need to identify best Co4 model with weights
```

---

## 7. Ensemble (4-Model)  BEST OVERALL

**Public LB Score**: 0.540-0.541
**CV Score**: N/A (ensemble of CV models)
**Architecture**: Weighted ensemble of ST + CNN + GRU + Geometric

### Ensemble Components:
1. ST Transformer (4L or 6L)
2. Multiscale CNN (4L)
3. GRU (Seed 27)
4. Geometric Network

### Ensemble Strategy:
```python
# Weighted average based on CV scores
weights = {
    'st_transformer': 0.30,
    'cnn': 0.25,
    'gru': 0.25,
    'geometric': 0.20
}

# Prediction
ensemble_pred = sum(w * model.predict(X) for model, w in zip(models, weights))
```

### Why showcase:
- Best overall performance
- Demonstrates ensembling technique
- Easy to create from individual models

### Implementation:
- **Code**: Create ensemble wrapper in `src/models/ensemble.py`
- **Weights**: Uses individual model weights from above
- **Notebook**: Show ensemble creation and inference

---

## Pretrained Weights Summary

### Total Size Estimate:
- ST Transformer (4L): ~200MB (20 folds)
- 6-Layer ST: ~300MB (20 folds)
- GRU: ~50MB (20 folds)
- Geometric: ~100MB (5 folds) or ~400MB (20 folds)
- CNN: ~150MB (20 folds)
- Perceiver Co4: ~200MB (estimated)
- **Total**: ~1-1.5 GB

### Hosting Strategy:
1. **Google Drive** (initial): Free, easy sharing
2. **Hugging Face Hub** (long-term): Professional, versioned
3. **Kaggle Datasets**: Accessible to competition community

### Download Script:
Create `scripts/download_pretrained.py` to:
1. Download from Google Drive/Hugging Face
2. Verify checksums
3. Extract to `pretrained/` directory

---

## Models to MENTION (Not full implementation)

These will be briefly described in README and docs but not fully implemented:

1. **Dual-Attention GRU** (0.568)
   - Enhanced GRU with spatial + temporal attention
   - Marginal improvement over standard GRU

2. **CNN-GRU Hybrid** (0.569)
   - Combined CNN feature extraction with GRU
   - Interesting but not better than individual models

3. **Position-Specific Models** (0.553)
   - Separate models for different positions (QB, WR, etc.)
   - Didn't beat general models

4. **LTC Networks** (0.588-0.608)
   - Liquid Time-Constant networks (continuous-time RNNs)
   - Novel but underperformed

5. **BiGRU** (0.583)
   - Bidirectional GRU
   - Surprisingly worse than uni-directional

6. **LightGBM** (7.724)
   - Gradient boosting failed completely
   - Shows deep learning is necessary

---

## Action Items

### Immediate:
- [ ] Verify all 7 model paths exist
- [ ] Check for 20-fold versions where only 5-fold found
- [ ] Identify best Perceiver Co4 model
- [ ] Document exact configurations
- [ ] Create model card for each

### For Upload:
- [ ] Select best fold from each model for quick demo
- [ ] Create single-fold versions for faster download
- [ ] Create "lite" package with 1-2 folds per model
- [ ] Create "full" package with all 20 folds

### For Code:
- [ ] Extract model definitions to src/models/
- [ ] Create inference wrappers
- [ ] Create training scripts
- [ ] Create config files for each

---

## Verification Checklist

For each model, verify:
- [ ] CV results exist and match reported scores
- [ ] Model weights (.pt files) exist for all folds
- [ ] Scaler files (.pkl) exist for all folds
- [ ] Can load and run inference successfully
- [ ] Understand exact architecture and hyperparameters
- [ ] Have training script that reproduces the model

