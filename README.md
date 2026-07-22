# JulongQuant · 巨龙量化

A-share quantitative stock-selection research framework. Pipeline: factor data → 3-model training → EW rank 5d ensemble → account-level real backtest.

> **V3 — 7/2026.** Three-model ensemble (LightGBM + DLinear + GatedDW-TCN), 5d causal smoothing, close-to-close execution. Real Sharpe 1.63, FF5 alpha 64.8%/yr.

## Results (2024-04 ~ 2026-07, corrected engine)

| | Sharpe | NAV | MaxDD | Turnover | Notes |
|---|---|---|---|---|---|
| **EW rank 5d (real)** | **1.63** | **2.43** | 16.4% | 10.7% | lots, limits, costs |
| EW rank 5d (paper) | 1.76 | 3.05 | 18.4% | 15.1% | close-to-close, continuous |
| Rank-Ridge 5d (paper) | 1.76 | 3.05 | 18.3% | 15.4% | RidgeCV weights |
| Bayes LLR w=63 (real) | 1.50 | 2.05 | 15.1% | 13.8% | industry-conditional LLR |
| ZZ500 | — | 1.62 | — | — | benchmark |
| ZZ1000 | — | 1.60 | — | — | benchmark |

**FF5 Alpha**: 64.8%/yr (t=3.62, Newey-West HAC). R² < 0.10 — pure stock-selection alpha, no factor loading.

## Stack

Python 3.11, PyTorch, LightGBM. 6 models in zoo (LGBM, XGBoost, DLinear, iTransformer, PatchTST, GatedDW-TCN). Production ensemble uses 3 models (LGBM, DLinear, GatedDW-TCN). Account-based real backtest with integer lots, limit filters, and smart-hold.

## Quick Start

```bash
source .venv/Scripts/activate

# Full evaluation (EW + Rank-Ridge + Bayes)
python -m scripts.evaluation.run_full_eval --mode llr --run-kind full

# Real backtest (Bayes + EW 5d comparison)
python -m scripts.evaluation.run_full_eval --mode llr --run-kind real

# FF5 attribution
python temp/build_ff5_factors.py
python temp/ff5_attribution.py
```

## Structure

```
src/
  backtest/     engine, portfolio, metrics, real_backtest, bayes_blender, ensemble
  models/       6 models (stable)
  experiment/   config, data, returns, runner, evaluation
  pipeline/     base_panel, factor_panel
  utils/        seed, logger, stats (Newey-West)
scripts/
  evaluation/   run_full_eval, real_backtest
  dataset/      build pipelines, daily_update
  experiment/   run_experiment, tune_experiment
dataset/
  input/        CSMAR, Tushare, indices (gitignored)
  processed/    unified_daily_panel, factor_panel, ff5_factors (gitignored)
reports/
  strategy_v1/  evidence/, diagnostic reports
```

## License

Non-commercial research and educational use only. Not financial advice. Backtested performance does not guarantee future results. See [LICENSE](./LICENSE).
