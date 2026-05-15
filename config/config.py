# =============================================================
# config/config.py — All constants, paths, and parameters
# Change values here only — everything else reads from here
# =============================================================

# -------------------------------------------------------------
# DATA PATHS
# -------------------------------------------------------------
FEATURE_PATH    = "data/processed/features_model_ready_v1.csv"
ETF_PATH        = "data/etf_prices.csv"
BOND_PROXY_PATH = "data/bond_return_proxies.csv"
DATE_COL        = "Date"
TRAIN_END       = "2018-12-31"

# -------------------------------------------------------------
# ASSET UNIVERSE
# -------------------------------------------------------------
EQUITY_ASSETS = [
    "State Street SPDR S&P 500 ETF Trust",
    "iShares S&P 500 Value ETF",
    "iShares S&P 500 Growth ETF",
    "iShares Russell 2000 ETF",
]
BOND_ASSETS = [
    "iShares 1-3 Year Treasury Bond ETF",
    "iShares 7-10 Year Treasury Bond ETF",
    "iShares 20+ Year Treasury Bond ETF",
    "iShares iBoxx USD Investment Grade Corporate Bond ETF",
    "iShares iBoxx USD High Yield Corporate Bond ETF",
]
ALT_ASSETS = ["SPDR Gold Shares"]
ALL_ASSETS = EQUITY_ASSETS + BOND_ASSETS + ALT_ASSETS

TICKER_MAP = {
    "State Street SPDR S&P 500 ETF Trust"                   : "SPY",
    "iShares S&P 500 Value ETF"                             : "IVE",
    "iShares S&P 500 Growth ETF"                            : "IVW",
    "iShares Russell 2000 ETF"                              : "IWM",
    "iShares 1-3 Year Treasury Bond ETF"                    : "SHY",
    "iShares 7-10 Year Treasury Bond ETF"                   : "IEF",
    "iShares 20+ Year Treasury Bond ETF"                    : "TLT",
    "iShares iBoxx USD Investment Grade Corporate Bond ETF" : "LQD",
    "iShares iBoxx USD High Yield Corporate Bond ETF"       : "HYG",
    "SPDR Gold Shares"                                      : "GLD",
}
EQUITY_T = ["SPY", "IVE", "IVW", "IWM"]
BOND_T   = ["SHY", "IEF", "TLT", "LQD", "HYG"]
ALT_T    = ["GLD"]
ALL_T    = EQUITY_T + BOND_T + ALT_T

# -------------------------------------------------------------
# BENCHMARK
# -------------------------------------------------------------
BENCHMARK_WEIGHTS = {"equity": 0.55, "bonds": 0.40, "alt": 0.05}

# -------------------------------------------------------------
# RISK PROFILES
# -------------------------------------------------------------
RISK_PROFILES = {
    "Conservative": {
        "gamma_Bull": 2.0, "gamma_Transition": 3.0, "gamma_Bear": 6.0,
        "equity_min_Bull": 0.40, "equity_max_Bull": 0.60,
        "equity_min_Trans": 0.30, "equity_max_Trans": 0.50,
        "bond_min_Bear": 0.40,
    },
    "Moderate": {
        "gamma_Bull": 1.0, "gamma_Transition": 2.0, "gamma_Bear": 4.0,
        "equity_min_Bull": 0.60, "equity_max_Bull": 0.80,
        "equity_min_Trans": 0.45, "equity_max_Trans": 0.65,
        "bond_min_Bear": 0.25,
    },
    "Moderate-Aggressive": {
        "gamma_Bull": 0.9, "gamma_Transition": 1.7, "gamma_Bear": 3.0,
        "equity_min_Bull": 0.65, "equity_max_Bull": 0.85,
        "equity_min_Trans": 0.50, "equity_max_Trans": 0.70,
        "bond_min_Bear": 0.15,
    },
    "Aggressive": {
        "gamma_Bull": 0.8, "gamma_Transition": 1.5, "gamma_Bear": 2.5,
        "equity_min_Bull": 0.70, "equity_max_Bull": 0.90,
        "equity_min_Trans": 0.55, "equity_max_Trans": 0.75,
        "bond_min_Bear": 0.10,
    },
}
# -------------------------------------------------------------
# PCA / GMM  (exploratory)
# -------------------------------------------------------------
N_PCA        = 14
PCA_SEED     = 42
N_REGIMES    = 5
GMM_COV_TYPE = "diag"
GMM_REG      = 1e-2
GMM_N_INIT   = 100
GMM_SEED     = 42
REGIME_MERGE_MAP = {
    0: "Downside Risk",
    1: "Downside Risk",
    2: "Normal/Growth",
    3: "Late Cycle",
    4: "Normal/Growth",
}

# -------------------------------------------------------------
# ML MODEL PARAMS  — CORE ENGINE UNCHANGED
# -------------------------------------------------------------
REFIT_MONTHS    = 12
FIRST_TRAIN_END = TRAIN_END

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1,
)

VIX_BULL_MAX      = 18.0
VIX_BEAR_MIN      = 25.0
DD_BEAR_THRESH    = -0.125
SMOOTH_WINDOW     = 5
YIELD_RISE_THRESH = 0.001
REGIME_ORDER      = ["Bull", "Transition", "Bear"]

RULE_DROP_COLS = [
    "Volatility_Risk_Sentiment_features__VIX",
    "equity_market_dynamics_features__SPX_drawdown",
] + [f"Volatility_Risk_Sentiment_features__VIX__rmean_{w}d"       for w in [5,10,21]] \
  + [f"Volatility_Risk_Sentiment_features__VIX__rstd_{w}d"        for w in [5,10,21]] \
  + [f"equity_market_dynamics_features__SPX_drawdown__rmean_{w}d" for w in [5,10,21]] \
  + [f"equity_market_dynamics_features__SPX_drawdown__rstd_{w}d"  for w in [5,10,21]]

CONFIRM_DAYS_ENTER = 7
CONFIRM_DAYS_EXIT  = 1
SWITCH_THRESHOLD   = 0.75
MIN_PROB_GAP       = 0.15
MIN_HOLD_DAYS      = {"Bull": 21, "Transition": 15, "Bear": 10}

# -------------------------------------------------------------
# HMM PARAMS
# -------------------------------------------------------------
HMM_N_COMPONENTS = 3
HMM_N_ITER       = 200
HMM_COV_TYPE     = "full"
HMM_TOL          = 1e-4

# -------------------------------------------------------------
# HSMM PARAMS
# -------------------------------------------------------------
HSMM_N_COMPONENTS = 3
HSMM_N_ITER       = 100
HSMM_MAX_DURATION = 60

# -------------------------------------------------------------
# JUMP MODEL PARAMS
# -------------------------------------------------------------
JUMP_N_REGIMES  = 3
JUMP_LOOKBACK   = 252

# -------------------------------------------------------------
# RHSM PARAMS
# -------------------------------------------------------------
RHSM_N_COMPONENTS = 3
RHSM_LAMBDA       = 0.1
RHSM_N_ITER       = 100

# -------------------------------------------------------------
# PORTFOLIO OPTIMIZER DEFAULTS
# -------------------------------------------------------------
GAMMA            = 3.0
MAX_WEIGHT       = 0.40
MAX_ACTIVE_TILT  = 0.50
GROUP_TILT_LIMIT = 0.50
L2_REG           = 0.00005
ANN_FACTOR       = 252

GAMMA_BY_REGIME = {"Bull": 1.0, "Transition": 2.0, "Bear": 4.0}

REGIME_SOLVE_PARAMS = {
    "Bull": dict(
        max_active_tilt=0.35, group_tilt_limit=0.40,
        equity_max=0.80, equity_min=0.60,
        bond_min=0.10,   bond_max=0.40,
        alt_min=None,    alt_max=0.20,
    ),
    "Transition": dict(
        max_active_tilt=0.25, group_tilt_limit=0.30,
        equity_max=0.65, equity_min=0.45,
        bond_min=0.20,   bond_max=0.45,
        alt_min=0.05,    alt_max=0.25,
    ),
    "Bear": dict(
        max_active_tilt=0.20, group_tilt_limit=0.25,
        equity_max=None, equity_min=None,
        bond_min=None,   bond_max=None,
        alt_min=None,    alt_max=0.20,
    ),
}

# -------------------------------------------------------------
# BACKTEST
# -------------------------------------------------------------
TC_BPS = 5

# -------------------------------------------------------------
# CACHE
# -------------------------------------------------------------
CACHE_DIR        = "cache"
CACHE_INDEX_FILE = "cache/cache_index.json"
