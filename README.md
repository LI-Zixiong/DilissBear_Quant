## Project Structure

```
JulongQuant/
│
├── dataset/              # Data storage
│   ├── input/            #   CSMAR raw data, Tushare daily panel, Aindustry.xlsx
│   ├── processed/        #   unified_daily_panel, financial_quarterly_panel, factor_panel_54
│   └── output/           #   Experiment outputs, backtest results, tuning runs
│
├── mds/                  # Documentation
│   └── FACTOR_LIBRARY.md #   54-factor library with formulas and ICIR rankings
│
├── reports/              # Generated experiment reports + factor audit
│
├── scripts/              # Runnable entry points
│   ├── dataset/          #   Data pipeline
│   │   ├── build_base_panel.py       Tushare + CSMAR → unified daily panel
│   │   ├── calc_factor_panel.py       Full 54-factor computation
│   │   ├── add_factors.py            Column extension (new factors only)
│   │   ├── select_universe.py        1500-stock stratified selection
│   │   └── daily_update.py           Daily incremental update
│   ├── experiment/       #   Experiments
│   │   ├── run_experiment.py         Single experiment entry point
│   │   └── tune_experiment.py        Grid tuning entry point
│   ├── evaluation/       #   Evaluation
│   │   ├── ensemble.py              Rank-Ridge + grid ensemble
│   │   └── check_experiment.py      Output validation
│   ├── factor/           #   Factor analysis
│   │   └── audit_factors.py         ICIR ranking + correlation + auto-selection
│   └── strategy/         #   Strategy research
│       └── strategy_v0.py           Prediction smoothing + backtest
│
├── src/                  # Source code
│   ├── data/             #   Data pipeline (stable — do not modify)
│   │   ├── loader.py             Parquet data loader
│   │   ├── preprocess.py         Factor preprocessing
│   │   └── dataset_builder.py    Sliding window dataset
│   ├── models/           #   Model implementations (6 models, stable)
│   ├── train/            #   Training logic
│   ├── predict/          #   Prediction
│   ├── backtest/         #   Backtesting engine
│   ├── utils/            #   Utilities
│   ├── experiment/       #   Experiment orchestration
│   │   ├── config.py             ExperimentConfig dataclass
│   │   ├── data.py               Data loading + preprocessing
│   │   ├── split.py              Chronological date split
│   │   ├── returns.py            Return frame construction + alignment
│   │   ├── model_factory.py      Model construction + param resolution
│   │   ├── runner.py             End-to-end experiment orchestrator
│   │   ├── evaluation.py         IC/top-bottom-spread/backtest per split
│   │   ├── report.py             Markdown report generation
│   │   └── tuning.py             Grid search runner
│   └── pipeline/         #   Data pipeline (reusable)
│       ├── base_panel.py         Base panel build + incremental + checks
│       └── factor_panel.py       54-factor compute + incremental + column extension
│
├── tests/                # Unit tests (25 tests)
│
├── .gitignore
├── CLAUDE.md
├── STAGE_0.md
├── requirements.txt
├── LICENSE
└── README.md
```

## License and Disclaimer

This project is released for non-commercial research and educational use only.

Commercial use is prohibited without prior written permission from the authors.

This project is not financial advice, investment advice, trading advice, or a recommendation to buy, sell, hold, or trade any financial instrument. Backtested or simulated performance does not guarantee future results. Use at your own risk.

See [LICENSE](./LICENSE) for details.
