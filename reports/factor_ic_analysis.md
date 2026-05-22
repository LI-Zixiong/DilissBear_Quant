# Factor IC Analysis — Final (2026-05-22, v2 post-ST-filter)

## Data
- Universe: inner-join intersection (530,384 rows, 366 dates)
- Pool: 1500 stocks after ST/low-price/high-leverage filter
- Ensemble: Rank-Ridge (fit on valid)
- Evaluation: daily cross-sectional Rank IC vs 1d forward return

## Per-Factor Rank IC

| Factor | mean rIC | ICIR | pos% | vs Ensemble |
|---|---|---|---|---|
| STREV | +0.070 | **+0.44** | 68.0% | +0.08 |
| GROWTH | +0.024 | +0.21 | 56.6% | -0.09 |
| VALUE | +0.033 | +0.17 | 53.6% | +0.18 |
| EARNYLD | +0.021 | +0.12 | 54.9% | -0.05 |
| BETA | +0.008 | +0.03 | 51.1% | +0.01 |
| LTREV | +0.002 | +0.02 | 50.3% | +0.14 |
| LEVERAGE | -0.003 | -0.05 | 50.3% | -0.02 |
| MOMENTUM | -0.012 | -0.07 | 47.0% | -0.17 |
| SIZE | -0.032 | -0.19 | 40.7% | -0.31 |
| SIZENL | -0.032 | -0.19 | 40.7% | -0.31 |
| RESVOL | -0.036 | -0.17 | 42.6% | -0.13 |
| LIQUIDITY | -0.045 | -0.20 | 41.5% | -0.17 |

## Ensemble Performance

| Metric | Value |
|---|---|
| rIC mean | 0.045 |
| ICIR | **0.517** |
| pos% | **69.7%** |
| Sharpe | 1.95 |
| MaxDD | 16.9% |

## Key Findings (v2)

1. **STREV still dominates** — ICIR 0.44, unchanged post-ST-filter.

2. **SIZE anti-exposure halved** — corr dropped from -0.54 to -0.31. ST filter removed micro-cap junk that the model was exploiting. The remaining pool is healthier.

3. **Holdings diversified** — April 2026 top picks spread across 10+ stocks (max 8/21 days), industries: chemicals, equipment, IT, textiles. No single stock dominates.

4. **Rank-Ridge controls drawdown** — MaxDD 17.1% vs grid combos at 19-21%. Selected for strategy V0 despite lower Sharpe.
