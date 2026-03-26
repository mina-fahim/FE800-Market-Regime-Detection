# FE800-Market-Regime-Detection
Financial Engineering (FE800) project on market regime detection and regime-aware portfolio allocation. Implements and compares HMM, Jump Models, and Deep Learning approaches for dynamic asset allocation.

```
FE800-Market-Regime-Detection
│
├── data/                # Raw and processed datasets
│
├── notebooks/           # Research notebooks and experiments
│
├── src/                 # Main source code
│   │
│   ├── models/          # Regime detection models
│   │   ├── hmm_model.py
│   │   ├── semi_markov_model.py
│   │   ├── jump_model.py
│   │   ├── rhsm_model.py
│   │   └── ml_regime_model.py | Reference: DeePM (2026) + CJM (Continuous Jump Model, 2024) 
│   │
│   ├── data/            # Data loading and preprocessing
│   │   └── data_loader.py
│   │
│   ├── features/        # Feature engineering
│   │   └── feature_engineering.py
│   │
│   ├── portfolio/       # Portfolio allocation logic
│   │   └── portfolio_allocator.py
│   │
│   └── backtest/        # Backtesting engine
│       └── backtest_engine.py
│
├── results/             # Output results
│   ├── figures/
│   └── tables/
│
├── dashboard/           # Visualization dashboard
│
├── docs/                # Documentation
│
├── requirements.txt     # Python dependencies
└── README.md
```
