# =============================================================
# src/data/data_loader.py — Load features, returns, split
# Shared across all models
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))

import numpy as np
import pandas as pd
from config import (
    FEATURE_PATH, ETF_PATH, BOND_PROXY_PATH, DATE_COL,
    TRAIN_END, ALL_ASSETS, TICKER_MAP,
    EQUITY_T, BOND_T, ALT_T, ALL_T, BENCHMARK_WEIGHTS,
)


def _parse_dates(df, col=DATE_COL):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col]).sort_values(col).set_index(col)


def _pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().iloc[1:]


def load_features(path: str = FEATURE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _parse_dates(df)
    print(f"[features]  shape={df.shape}  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_asset_returns(
    etf_path:  str  = ETF_PATH,
    bond_path: str  = BOND_PROXY_PATH,
    assets:    list = ALL_ASSETS,
) -> pd.DataFrame:
    etf_raw = pd.read_csv(etf_path, index_col=0).iloc[1:]
    etf_raw.index = pd.to_datetime(etf_raw.index, errors="coerce")
    etf_raw = etf_raw[~etf_raw.index.isna()].sort_index()
    etf_raw = etf_raw.apply(pd.to_numeric, errors="coerce")
    etf_returns = _pct_returns(etf_raw)

    bond_raw = pd.read_csv(bond_path, index_col=0, parse_dates=True)
    bond_raw = bond_raw.sort_index().apply(pd.to_numeric, errors="coerce")

    merged    = etf_returns.join(bond_raw, how="outer")
    available = [a for a in assets if a in merged.columns]
    missing   = [a for a in assets if a not in merged.columns]
    if missing:
        print(f"[returns]  WARNING — not found: {missing}")

    returns = merged[available].dropna(how="all")
    print(f"[returns]   shape={returns.shape}  {returns.index[0].date()} → {returns.index[-1].date()}")
    return returns


def make_train_test_split(
    features:  pd.DataFrame,
    returns:   pd.DataFrame,
    train_end: str = TRAIN_END,
) -> dict:
    cutoff  = pd.Timestamp(train_end)
    X_train = features[features.index <= cutoff]
    X_test  = features[features.index >  cutoff]
    r_train = returns[returns.index   <= cutoff]
    r_test  = returns[returns.index   >  cutoff]

    common_train = X_train.index.intersection(r_train.index)
    common_test  = X_test.index.intersection(r_test.index)
    X_train = X_train.loc[common_train];  r_train = r_train.loc[common_train]
    X_test  = X_test.loc[common_test];    r_test  = r_test.loc[common_test]

    assert X_train.index.max() <= cutoff,                          "Train leaks!"
    assert X_test.index.min()  >  cutoff,                          "Test starts before cutoff!"
    assert len(X_train.index.intersection(X_test.index)) == 0,     "Overlap!"

    print(f"TRAIN: {X_train.index[0].date()} → {X_train.index[-1].date()} ({len(X_train):,}d)")
    print(f"TEST : {X_test.index[0].date()}  → {X_test.index[-1].date()}  ({len(X_test):,}d)")
    return dict(X_train=X_train, X_test=X_test,
                r_train=r_train, r_test=r_test, cutoff=cutoff)


def make_benchmark_series(r_train: pd.DataFrame) -> pd.Series:
    r_t   = r_train.rename(columns=TICKER_MAP)
    n_eq  = len(EQUITY_T); n_bd = len(BOND_T); n_alt = len(ALT_T)
    bm_w  = (  [BENCHMARK_WEIGHTS["equity"] / n_eq]  * n_eq
             + [BENCHMARK_WEIGHTS["bonds"]  / n_bd]  * n_bd
             + [BENCHMARK_WEIGHTS["alt"]    / n_alt] * n_alt)
    bm    = pd.Series(bm_w, index=ALL_T).reindex(r_t.columns).fillna(0)
    bm   /= bm.sum()
    return bm


def load_all(train_end: str = TRAIN_END) -> dict:
    """Convenience: load everything and return one dict."""
    features = load_features()
    returns  = load_asset_returns()
    split    = make_train_test_split(features, returns, train_end)
    bm       = make_benchmark_series(split["r_train"])
    X_all    = pd.concat([split["X_train"], split["X_test"]]).sort_index()
    r_all    = pd.concat([split["r_train"], split["r_test"]]).sort_index()
    return {**split, "X_all": X_all, "r_all": r_all,
            "benchmark_series": bm, "features": features, "returns": returns}
