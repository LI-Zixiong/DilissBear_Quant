# JulongQuant · 巨龙量化

Quantitative stock-selection research framework. Pipeline: raw factor data → preprocessing → sliding-window dataset → model training → prediction → Bayes blending → backtest. PyTorch + LightGBM + XGBoost. Python 3.11. Non-commercial research license.

## V2 Production (2026-06-09)

**Bayes V2 Auto-Blender** — fully data-driven ensemble. Hierarchy weights and LLR clip recalibrated automatically from each 63-day rolling window. No hardcoded model-specific parameters.

| Strategy | Sharpe | NAV | MaxDD | Turnover |
|---|---|---|---|---|
| **Bayes V2** | **1.985** | **3.163** | **16.5%** | **25.9%** |
| gated_dwtcn | 1.895 | 2.776 | 15.8% | 15.7% |
| EW rank 3d | 1.894 | 3.105 | 19.0% | 35.8% |
| Rank-Ridge 3d | 1.847 | 3.038 | 19.2% | 36.7% |

**Config**: smooth=3d, auto_clip p85 (≈1.31), auto_weights from top-bin positive count-weighted LLR. Rolling window=63d. Universe=ZZ500+ZZ1000 (1500 stocks). Split=0.6/0.2/0.2. Transaction costs included (buy 0.03%, sell 0.08%).

## Model Zoo

| Model | Type | Factors | Test Sharpe | Key traits |
|---|---|---|---|---|
| LGBM | tabular | 43 (no limit-up/down) | 1.83 | leaf-wise, feat_frac=0.8 |
| XGBoost | tabular | 55 | 1.77 | colsample=0.12 forces tree diversity |
| DLinear | torch | 30 (slow-varying) | 1.69 | seq=20, linear decomposition |
| GatedDW-TCN | torch | 46 (interaction) | 1.90 | 3,047 params, rank-8 gate, dil=(1,2,4) |

## Project Structure

```
JulongQuant/
├── src/
│   ├── backtest/
│   │   ├── bayes_blender.py        V2 auto-calibrating Bayes ensemble
│   │   ├── ensemble_methods.py     single/EW/dual/rank-ridge fusion
│   │   ├── ensemble_utils.py       smoothing, ranking, backtest helpers
│   │   ├── engine.py               backtest engine with transaction costs
│   │   ├── portfolio.py            portfolio construction (top_n)
│   │   └── metrics.py              Sharpe, MaxDD, IC, turnover
│   ├── models/                     6 models (stable — see CLAUDE.md)
│   ├── train/                      train_tabular + train_torch
│   ├── predict/                    generate_predictions
│   ├── data/                       loader, preprocess, dataset_builder (stable)
│   ├── experiment/                 config, data, split, returns, model_factory, runner, report, tuning
│   └── pipeline/                   base_panel, factor_panel
├── scripts/
│   ├── evaluation/
│   │   └── run_full_eval.py        unified evaluation (single → EW → dual → Ridge → Bayes V2)
│   ├── experiment/
│   │   ├── run_experiment.py       end-to-end experiment
│   │   └── tune_experiment.py      grid tuning
│   └── dataset/                    build_base_panel, calc_factor_panel, select_universe, daily_update
├── temp/                           diagnostic + scan scripts (not tracked)
├── dataset/
│   ├── input/                      CSMAR, Tushare, Aindustry.xlsx
│   ├── processed/                  unified_daily_panel, factor_panel_1500_54_ind
│   └── output/                     experiment outputs
├── mds/                            FACTOR_LIBRARY.md
├── reports/                        generated reports + evidence CSVs
├── tests/                          25 tests
├── CLAUDE.md                       project rules + stable file list
├── requirements.txt
└── LICENSE
```

## Quick Reference

```bash
source .venv/Scripts/activate

# Run full evaluation (all strategies + Bayes V2)
python -m scripts.evaluation.run_full_eval

# Bayes V2 only
python -m scripts.evaluation.run_full_eval --mode bayes

# Run experiment pipeline
python -m scripts.experiment.run_experiment

# Run tests
python -m pytest tests/ -v
```

## License

Non-commercial research and educational use only. Commercial use prohibited without prior written permission. Not financial advice. Backtested performance does not guarantee future results. Use at your own risk. See [LICENSE](./LICENSE).
