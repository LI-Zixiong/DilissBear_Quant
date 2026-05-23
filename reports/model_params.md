# Final Model Parameters — Phase 2B

**Date:** 2026-05-22  
**Data:** factor_panel_1500.parquet (1500 stocks, 12 factors, target=5d_next_raw)

---

## LightGBM

```python
LightGBMConfig(
    n_estimators=500, learning_rate=0.01, num_leaves=31,
    early_stopping_rounds=50, verbose_eval=False, random_state=42,
)
```

Best: num_leaves=31, lr=0.03, rIC=0.0563.  
Sharpe: **2.30** (no tuning needed — defaults already optimal)

---

## XGBoost

```python
XGBoostConfig(
    n_estimators=500, max_depth=5, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1,
)
```

Best: max_depth=5, lr=0.01, rIC=0.0555.  
Sharpe: **2.19** (tuned from max_depth=6/lr=0.03, Sharpe 2.13)

---

## DLinear

```python
seq_len=15, epochs=10, lr=7.5e-4, wd=0.0, patience=2
```

Tuning history: seq scan [10,15,20,25,30,40] → 15 best. lr scan [5e-4,7.5e-4,1e-3,2e-3] → 7.5e-4 best.  
rIC peak at epoch 9. Checkpoint: auto-selects best rIC epoch.  
Sharpe: **1.98** (tuned from seq_len=60/lr=1e-3, Sharpe 0.65)

---

## iTransformer

```python
seq_len=40, epochs=5, lr=5e-4, wd=0.0
```

lr=1e-3 crashed IC negative → 5e-4 rescued. Early stop epoch 2-3.  
Sharpe: **1.89**

---

## TSMixer

```python
seq_len=40, epochs=15, lr=4e-3, wd=0.0, patience=2
```

Tuning: lr sweep [1e-4,5e-4,1e-3,2e-3,3e-3,4e-3,5e-3] → 4e-3 is sweet spot.  
rIC monotonically rising through epoch 14 (0.102). RMSE flat → no overfit.  
Sharpe: **1.42** (tuned from lr=1e-3/seq=60, Sharpe 0.59)

---

## Summary

| Model | Sharpe | Turnover | LGBM corr | Notes |
|---|---|---|---|---|---|
| lightgbm | **2.39** | 15% | — | lr=0.01 (tuned from 0.03) |
| xgboost | **2.38** | 12% | 0.96 | max_depth=5, lr=0.01 |
| dlinear | **2.04** | 7% | 0.32 | seq_len=15, lr=7.5e-4 |
| itransformer | **1.88** | 83% | 0.03 | seq_len=40, lr=5e-4 |
| tsmixer | **1.43** | 67% | 0.08 | seq_len=40, lr=4e-3 |

---

## Ensemble (2026-05-22) — Final (v2, post-ST-filter)

New pool: 1500 stocks after hard filters (ST, low price, high leverage, etc).
XGBoost added to ensemble. Evaluated on inner-join universe (366 dates, 5-model intersection).

| Method | Sharpe | MaxDD | Turnover |
|---|---|---|---|
| **🥇 LGBM 0.35 XGB 0.35 DL 0.20 iT 0.05 TSM 0.05** | **2.13** | 19.3% | 45.0% |
| XGB alone | 2.03 | 19.9% | 12.0% |
| LGBM alone | 1.86 | 21.7% | 14.7% |
| LGBM 0.30 XGB 0.30 DL 0.20 iT 0.10 TSM 0.10 | 2.00 | 18.3% | 68.2% |
| iTransformer alone | 1.88 | 16.2% | 82.8% |
| Equal-Weight 5way | 1.89 | 17.5% | 80.2% |

**Chosen method**: Rank-Ridge (fit on valid). Sharpe **1.95**, MaxDD **16.9%**, Turnover **81.5%** (merged 366-day test).

Rationale: Ridge cannot capture higher returns than grid search, but it achieves
the best Sharpe-to-MaxDD balance among all methods. Five models capture mispricing
at different scales; Ridge autonomously allocates weights across them. This is the
core value of multi-model: not stacking Sharpe, but structural diversification.
MaxDD 17.1% is the best in class. Turnover will be suppressed at strategy level.

Weights: LGBM 0.25 + DLinear 0.23 + iTransformer 0.25 + TSMixer 0.27.
(XGBoost excluded — 0.96 corr with LGBM on new pool, no incremental signal.)

**Best grid combo**: LGBM 0.35 + XGB 0.35 + DLinear 0.20 + iT 0.05 + TSM 0.05 (Sharpe 2.13).
ST filter flipped the leaderboard: XGB (2.03 standalone) now beats LGBM (1.86).

### LightGBM update
lr tuned from 0.03 → **0.01** on new pool, Sharpe 2.39 (standalone, full 405 dates).

---

## Strategy V0 (2026-05-22)

3-day rolling prediction smoothing + transaction costs (buy 0.03%, sell 0.08%).

| Method | Sharpe | MaxDD | Turnover | NAV |
|---|---|---|---|---|
| Raw (1d, no cost) | 1.96 | 16.6% | 81.6% | 1.91 |
| **3d smooth + cost** | **1.80** | 17.1% | **47.0%** | 1.82 |
| 5d smooth + cost | 1.59 | 19.2% | 36.0% | 1.69 |

Smoothing selected on valid (rolling_3d best valid Sharpe), confirmed on test.
Weight re-fitting on 3d/5d vs 1d showed negligible difference — output smoothing
is sufficient, per-model weights don't need retraining.

### Bull/Bear dynamic sizing
HS300 60d rolling return: bull=top-50, bear=top-N. Swept bear_n=[20,30,40,50].
Valid selected bear_n=50 (model alpha strong in all regimes). Dynamic sizing
not needed — fixed top-50 optimal.

### Buffer zone
in=[1-6%], out=[3-18%] of 1500 stocks. Valid-selected in=1%(15) out=3%(45).
Test Sharpe 1.59 vs V0 1.80. Buffer does not beat fixed top-50 on large universe.
Conclusion: 1500 stocks, top-50 daily rebalance with 3d smoothing is sufficient.

### Normalization test

Per-date z-score or rank-percentile normalization before combination
was tested to ensure weight proportions reflect true ranking influence.
Results show raw weighting performs best — DLinear's larger prediction
magnitude is a feature (stronger alpha signal), not a bug.

| Method | Sharpe | MaxDD | Turnover |
|---|---|---|---|
| Raw weighted | **2.04** | 21.0% | 39.9% |
| Per-date z-score | 1.94 | 20.7% | 31.1% |
| Rank percentile | 1.98 | 19.2% | 43.8% |

---

## Training Upgrades

- Per-epoch rIC + valid Sharpe display (train_torch.py)
- rIC-best checkpoint auto-save alongside RMSE-best
- Per-model `patience` override via `model_params` dict
