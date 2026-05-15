# =============================================================
# src/portfolio/portfolio_allocator.py
# Shared MV optimizer — used by ALL models
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))

import numpy as np
import pandas as pd
import cvxpy as cp
from config import (
    EQUITY_T, BOND_T, ALT_T, TICKER_MAP,
    GAMMA, MAX_WEIGHT, MAX_ACTIVE_TILT, GROUP_TILT_LIMIT, L2_REG,
    GAMMA_BY_REGIME, REGIME_SOLVE_PARAMS, YIELD_RISE_THRESH,
    RISK_PROFILES, REGIME_ORDER, FIRST_TRAIN_END, ANN_FACTOR,
)


def solve_mv_portfolio(
    mu, cov, benchmark_weights,
    equity_assets, bond_assets, alt_assets,
    gamma=GAMMA, max_weight=MAX_WEIGHT,
    max_active_tilt=MAX_ACTIVE_TILT,
    group_tilt_limit=GROUP_TILT_LIMIT,
    l2_reg=L2_REG,
    equity_max=None, equity_min=None,
    bond_min=None,   bond_max=None,
    alt_min=None,    alt_max=None,
    asset_caps=None,
) -> pd.Series:
    """Core MV optimizer — maximize μᵀw − γ·wᵀΣw − λ‖w‖²"""
    assets = list(mu.index)
    n      = len(assets)
    wb     = benchmark_weights.reindex(assets).fillna(0).values
    mu_vec = mu.reindex(assets).fillna(0).values
    sigma  = cov.reindex(index=assets, columns=assets).fillna(0).values
    w      = cp.Variable(n)

    eq_idx  = [i for i,a in enumerate(assets) if a in equity_assets]
    bd_idx  = [i for i,a in enumerate(assets) if a in bond_assets]
    alt_idx = [i for i,a in enumerate(assets) if a in alt_assets]

    obj  = cp.Maximize(mu_vec@w - gamma*cp.quad_form(w,sigma) - l2_reg*cp.sum_squares(w))
    cons = [cp.sum(w)==1, w>=0, w<=max_weight,
            w-wb<=max_active_tilt, wb-w<=max_active_tilt]
    for idx in [eq_idx, bd_idx, alt_idx]:
        if idx:
            cons += [cp.sum(w[idx]) - sum(wb[i] for i in idx) <=  group_tilt_limit,
                     sum(wb[i] for i in idx) - cp.sum(w[idx]) <=  group_tilt_limit]
    if equity_max is not None and eq_idx:  cons.append(cp.sum(w[eq_idx]) <= equity_max)
    if equity_min is not None and eq_idx:  cons.append(cp.sum(w[eq_idx]) >= equity_min)
    if bond_min   is not None and bd_idx:  cons.append(cp.sum(w[bd_idx]) >= bond_min)
    if bond_max   is not None and bd_idx:  cons.append(cp.sum(w[bd_idx]) <= bond_max)
    if alt_min    is not None and alt_idx: cons.append(cp.sum(w[alt_idx]) >= alt_min)
    if alt_max    is not None and alt_idx: cons.append(cp.sum(w[alt_idx]) <= alt_max)
    if asset_caps:
        for ticker, cap in asset_caps.items():
            if ticker in assets:
                cons.append(w[assets.index(ticker)] <= cap)

    cp.Problem(obj, cons).solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        return pd.Series(wb, index=assets)
    sol = np.clip(w.value, 0, None)
    return pd.Series(sol/sol.sum(), index=assets)


def build_regime_portfolios(
    r_train_t:            pd.DataFrame,
    tr_rule_al:           pd.Series,
    benchmark_series:     pd.Series,
    yield_trend_at_refit: float,
    risk_profile:         str = "Moderate",
    custom_gamma:         dict = None,
) -> pd.DataFrame:
    """Build Bull/Transition/Bear portfolios for one fold."""
    rp             = RISK_PROFILES[risk_profile]
    gamma_by_reg   = custom_gamma or {
        "Bull":       rp["gamma_Bull"],
        "Transition": rp["gamma_Transition"],
        "Bear":       rp["gamma_Bear"],
    }
    rising_yields  = yield_trend_at_refit > YIELD_RISE_THRESH
    regime_ports   = {}

    solve_params = {
        "Bull": dict(
            max_active_tilt=0.35, group_tilt_limit=0.40,
            equity_max=rp["equity_max_Bull"],
            equity_min=rp["equity_min_Bull"],
            bond_min=0.10, bond_max=0.40,
            alt_min=None, alt_max=0.20,
        ),
        "Transition": dict(
            max_active_tilt=0.25, group_tilt_limit=0.30,
            equity_max=rp["equity_max_Trans"],
            equity_min=rp["equity_min_Trans"],
            bond_min=0.20, bond_max=0.45,
            alt_min=0.05, alt_max=0.25,
        ),
        "Bear": dict(
            max_active_tilt=0.20, group_tilt_limit=0.25,
            equity_max=None, equity_min=None,
            bond_min=rp.get("bond_min_Bear", None),
            bond_max=None, alt_min=None, alt_max=0.20,
        ),
    }

    for regime in REGIME_ORDER:
        idx   = tr_rule_al[tr_rule_al == regime].index
        r_sub = r_train_t.loc[r_train_t.index.intersection(idx)]
        if len(r_sub) < 20:
            r_sub = r_train_t
        if regime == "Bear":
            mu  = 0.6*r_sub.mean() + 0.4*r_train_t.mean()
            cov = 0.6*r_sub.cov()  + 0.4*r_train_t.cov()
        else:
            mu, cov = r_sub.mean(), r_sub.cov()

        p          = solve_params[regime].copy()
        asset_caps = None
        if regime == "Bear" and rising_yields:
            asset_caps    = {"TLT": 0.05, "IEF": 0.05}
            p["bond_min"] = 0.35

        regime_ports[regime] = solve_mv_portfolio(
            mu=mu, cov=cov,
            benchmark_weights=benchmark_series,
            equity_assets=EQUITY_T, bond_assets=BOND_T, alt_assets=ALT_T,
            gamma=gamma_by_reg[regime],
            max_weight=MAX_WEIGHT, l2_reg=L2_REG,
            asset_caps=asset_caps, **p,
        )
    return pd.DataFrame(regime_ports).T


def build_mv_benchmark(
    r_all: pd.DataFrame,
    benchmark_series: pd.Series,
    first_train_end: str = FIRST_TRAIN_END,
    refit_months: int = 12,
) -> pd.DataFrame:
    """No-regime MV baseline — walk-forward expanding window."""
    first_train_end = pd.Timestamp(first_train_end)
    all_dates       = r_all.index
    mv_weights_ts   = {}
    t = first_train_end
    while t < all_dates[-1]:
        predict_end = min(t + pd.DateOffset(months=refit_months), all_dates[-1])
        r_tr = r_all[r_all.index <= t].rename(columns=TICKER_MAP)
        r_tr = r_tr.reindex(columns=benchmark_series.index).dropna(how="all")
        if len(r_tr) > 100:
            w = solve_mv_portfolio(
                mu=r_tr.mean(), cov=r_tr.cov(),
                benchmark_weights=benchmark_series,
                equity_assets=EQUITY_T, bond_assets=BOND_T, alt_assets=ALT_T,
                gamma=2.0, max_weight=MAX_WEIGHT,
                max_active_tilt=0.25, group_tilt_limit=0.30, l2_reg=L2_REG,
            )
            for dt in all_dates[(all_dates > t) & (all_dates <= predict_end)]:
                mv_weights_ts[dt] = w
        t = predict_end
    mv_w_df = pd.DataFrame(mv_weights_ts).T
    print(f"[MV benchmark]  {len(mv_w_df)} days computed")
    return mv_w_df
