# BigData2 Lessons & Synthetic Data Notes

Thorough recap of what worked and what did not in the `/mnt/raid0/BigData2` experiments, plus clear guidance on synthetic datasets.

## Sources reviewed
- /mnt/raid0/BigData2/FINAL_EXPERIMENT_RESULTS.md — 125 geometric runs and hyperparameter sweeps
- /mnt/raid0/BigData2/LB_RESULTS_ANALYSIS.md — CV→LB gaps, ensemble failures, TTA issues
- /mnt/raid0/BigData2/SYNTHETIC_RESULTS_ANALYSIS.md — synthetic data impact and failure modes

## Headline metrics
- Best single real-data LB (baseline GRU seed27): 0.557
- Best synthetic-heavy LB: 0.559 (worse than baseline despite better CV)
- Ensemble/TTA misfire: 3.674 LB when noise + speed scaling were applied pre-scaler
- Best CV from sweeps: 0.0827 (window 11, batch 256, lr 3e-4, h192) — did not translate to LB gains when over-engineered

## What worked (carried over into this repo)
- Feature pipeline stability: The 167-feature stack (basic kinematics, ball-relative, temporal, geometric) stayed robust across windows and models.
- Conservative hyperparameters: Batch size 256 with windows 9–13 and learning rates 2e-4–3e-4 consistently topped the CV leaderboards.
- Simple augmentations: Horizontal flip (+speed jitter when needed) was the safest gain; this is what we ship in src/data/augmentation.py.
- Model depth sweet spot: Medium-depth architectures (6-layer ST Transformer, 2-layer GRU) generalized better than deeper variants or heavy stacking.

## What did not work (and why)
- TTA gone wrong (LB_RESULTS_ANALYSIS.md): Noise + speed scaling applied before the scaler, destroying feature distributions and cratering LB (0.589 → 3.674). Only use the lightweight flip TTA already provided.
- Large, similar ensembles: A 12-model multi-seed ensemble matched the single-seed LB (0.589). Diversity was too low to help; more models did not add value.
- Coverage feature variants: Coverage-heavy feature sets consistently underperformed the 167-feature baseline; they amplified distribution mismatch.
- Synthetic-heavy training: CV ticked up, but LB stayed flat or regressed due to distribution mismatch (details below).

## Synthetic dataset takeaways
- Performance (SYNTHETIC_RESULTS_ANALYSIS.md):
  - 500-play and 5,240-play synthetic sets improved CV by ~3%, yet LB stayed at 0.584 or slipped to 0.559 versus the real-data baseline (0.557 LB).
  - Distribution mismatch was the root cause: synthetic routes were too clean and predictable, so models overfit to patterns absent in real test data.
  - Coverage features + synthetic data made things worse.
- Competition rule: Synthetic datasets are not competition-legal. Keep them for offline ablations or research only; do not train or submit competition-facing models with them.
- Where they live: Examples in /mnt/raid0/BigData2/ — synthetic_w9s42_v3/, synthetic_v2/, synthetic_season_*, synthetic_fixed/, with generation scripts generate_synthetic_*.py.
- Safe usage pattern: If you experiment offline, mix a small slice of synthetic with real data, monitor LB gaps closely, and never publish or submit models trained on synthetic-only inputs.

## Actionable guidance for this repo
- Use the walkthrough notebook notebooks/04_feature_engineering.ipynb to inspect engineered columns, window sizes, and sequence shapes before training.
- Stick to the 167-feature baseline with horizontal flip (and optional speed jitter) unless new evidence shows gains.
- Prefer real-data training for any releasable model; keep synthetic data strictly for internal analysis or stress tests.
- Avoid heavy TTA and large homogeneous ensembles; the simplest flip TTA and small, diverse ensembles performed most reliably.
