# FE800-Market-Regime-Detection
Financial Engineering (FE800) project on market regime detection and regime-aware portfolio allocation. Implements and compares HMM, Jump Models, and Deep Learning approaches for dynamic asset allocation.

```
FE800-Market-Regime-Detection
│
├── data/                          # Raw and processed datasets
│   │
│   ├── processed/                    # processed datasets
│
├── notebooks/                     # Research notebooks and experiments
│
├── src/                           # Main source code
│   │
│   ├── models/                    # Regime detection models
│   │   ├── hmm_model.py           | Members: Niti / Darshan  | Reference: HMM vs HSMM (2024)
│   │   ├── semi_markov_model.py   | Members: Nabil / Willie  | Reference: HMM vs HSMM (2024)
│   │   ├── jump_model.py          | Members: Ojaus           | Reference: Regularized Jump Models (2024)
│   │   ├── jump_models/           | Local package — Continuous Jump Model source
│   │   └── ml_regime_model.py     | Members: Mina            | Reference: DeePM (2026) & CJM (2024)
│   │
│   ├── data/                      # Data loading and preprocessing
│   │   └── data_loader.py
│   │
│   ├── features/                  # Feature engineering
│   │   └── feature_engineering.py
│   │
│   ├── portfolio/                 # Mean-variance portfolio optimizer (CVXPY)
│   │   └── portfolio_allocator.py
│   │
│   └── backtest/                  # Walk-forward backtesting engine
│       └── backtest_engine.py
│
├── config/                        # Model and risk profile configuration
│   └── config.py
│
├── dashboard/                     # Shiny for Python interactive dashboard
│   └── app.py                     
│
├── results/                       # Output results
│   ├── figures/
│   └── tables/
│
├── docs/                # Final report and presentations
│
├── cache/                         # Cached model runs
├── exports/                       # PDF exports
│
├── requirements.txt               # Python dependencies
└── README.md

```


## Installation

```bash
git clone https://github.com/mina-fahim/FE800-Market-Regime-Detection.git
cd FE800-Market-Regime-Detection
pip install -r requirements.txt
pip install -e src/models/jump_models/
```

---

## Run the Dashboard

```bash
shiny run dashboard/app.py
```




