# JulongQuant · 巨龙量化

A-share quantitative stock-selection research framework. Pipeline: factor data → model training → Bayes blending → account-level real backtest.

> **V1** — 6/2026. Open-to-open execution, smart-hold, FF5 attribution. Sharpe 1.42 real, alpha 13.3%/yr after factor controls.

## Results (real backtest, 2024-03 ~ 2026-06)

| | Sharpe | NAV | MaxDD | Notes |
|---|---|---|---|---|
| **Strategy (real)** | **1.42** | **2.03** | 14.8% | open-to-open, smart-hold, budget lots |
| Strategy (paper) | 2.00 | 3.26 | 15.0% | close-to-close, continuous weights |
| ZZ500 | — | 1.46 | — | benchmark |
| ZZ1000 | — | 1.45 | — | benchmark |

**FF5 Alpha**: 13.3%/yr (t=2.69, Newey-West). Selection residual Sharpe **1.91**, market-neutral (corr -0.13).  
See `reports/strategy_v1/V1_REPORT.md` for full attribution.

## Stack

Python 3.11, PyTorch, LightGBM, XGBoost. 6 models (LGBM, XGBoost, DLinear, iTransformer, PatchTST, GatedDW-TCN). Bayes V2 rolling-window ensemble. Account-based real backtest with integer lots, limit filters, and smart-hold.

## Quick Start

```bash
source .venv/Scripts/activate

# Run real backtest (budget mode)
python -m scripts.evaluation.real_backtest --position-sizing budget

# Full evaluation sweep
python -m scripts.evaluation.run_full_eval --mode full

# FF5 attribution
python temp/build_ff5_factors.py
python temp/ff5_attribution.py
```

## Structure

```
src/
  backtest/     real_backtest (account engine), bayes_blender, ensemble, portfolio, metrics
  models/       6 models (stable)
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
  strategy_v1/  V1_REPORT.md, evidence/
```

## License

Non-commercial research and educational use only. Not financial advice. Backtested performance does not guarantee future results. See [LICENSE](./LICENSE).
