# JulongQuant · 巨龙量化

A-share quantitative stock-selection research framework. Pipeline: factor data → 3-model training → EW rank 10d ensemble → account-level real backtest → live daily trading.

> **V3.1 — 7/2026.** 10d causal smoothing, close-to-close execution, fixed buffer + regime defense, budget position sizing (98% deployment). Official ZZ1500 benchmark (ZZ500+ZZ1000 6:5 market-cap weighted).

## Results (2022-01 ~ 2026-07, corrected engine)

| Strategy | Sharpe | NAV | MaxDD | Turnover | Notes |
|---|---:|---:|---:|---:|---|
| **s1 (buf50 baseline)** | **1.16** | **3.90** | 42.1% | 8.8% | fixed buffer, no defense |
| **s2 (buf120 + Bear20%)** | **1.36** | **4.34** | 20.7% | 8.6% | buffer + regime defense |
| ZZ1500 (official) | 0.26 | 1.16 | 40.4% | — | ZZ500+ZZ1000 6:5 cap-weighted |

## Live Trading

```bash
source .venv/Scripts/activate

# Daily pipeline
python -m scripts.dataset.live_pipeline --project-root D:\JulongQuant

# Strategy evaluation (test period only)
python -m scripts.evaluation.run_full_eval --mode llr --run-kind real

# Strategy grid search
python -m scripts.evaluation.run_strategy_grid --smooth 10

# Buffer diagnostic
python -m scripts.evaluation.buffer_diagnostic

# Index data update
python -m scripts.dataset.daily_update --update-indices
```

## Strategy Parameters (s2)

| Parameter | Value | Notes |
|---|---|---|
| Smooth window | 10d | EW rank smoothing |
| Position sizing | Budget | Bin-weighted equal-notional |
| Cash ratio | 98% | High deployment |
| Buffer | Fixed-120 | Exit threshold keeps top-120 held stocks |
| Bear defense | Score < -0.35 + falling + not bull | Reduces to 20% cash |
| Recovery | 3-day linear | 20% → 46% → 72% → 98% |

## Stack

Python 3.11, PyTorch, LightGBM. 6 models in zoo (LGBM, XGBoost, DLinear, iTransformer, PatchTST, GatedDW-TCN). Production ensemble uses 3 models (LGBM, DLinear, GatedDW-TCN). Account-based real backtest with integer lots, limit filters, T+1, min ¥5 commission, stamp tax.

## Structure

```
src/
  backtest/     engine, portfolio, metrics, real_backtest, live_account, ensemble
  models/       6 models (stable)
  experiment/   config, data, returns, runner
  pipeline/     live_store, tushare_client, factor_panel, live_factor_*
  predict/      live_predictor, live_predictor_incremental
scripts/
  evaluation/   run_full_eval, run_strategy_grid, live_portfolio, buffer_diagnostic
  dataset/      live_pipeline, daily_update, live_postprocess
website/        SPA dashboard (index.html, convert_data.py)
dataset/
  input/        CSMAR, Tushare, indices (gitignored)
  processed/    unified_daily_panel, factor_panel (gitignored)
  cache/        live store, grid cache (gitignored)
reports/
  strategy_v1/  evidence/, diagnostic reports, NAV curves
pics/           Generated comparison plots (gitignored)
```

## License

Non-commercial research and educational use only. Not financial advice. Backtested performance does not guarantee future results. See [LICENSE](./LICENSE).
