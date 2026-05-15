# =============================================================
# src/models/ml_regime_model.py
# ML Regime Model — GMM + XGBoost Walk-Forward
# Author: Mina Fahim
# Reference: DeePM (2026), CJM (2024)
# CORE ENGINE UNCHANGED
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

from config import (
    N_PCA, N_REGIMES, GMM_N_INIT, GMM_SEED, GMM_COV_TYPE, GMM_REG,
    REFIT_MONTHS, FIRST_TRAIN_END, TICKER_MAP,
    REGIME_ORDER, RULE_DROP_COLS,
    VIX_BULL_MAX, VIX_BEAR_MIN, DD_BEAR_THRESH, SMOOTH_WINDOW,
    YIELD_RISE_THRESH, MAX_WEIGHT, L2_REG, EQUITY_T,
    CONFIRM_DAYS_ENTER, CONFIRM_DAYS_EXIT,
    SWITCH_THRESHOLD, MIN_PROB_GAP, MIN_HOLD_DAYS,
    XGB_PARAMS,
)
from features.feature_engineering import add_regime_discriminators, add_lag_features
from portfolio.portfolio_allocator import build_regime_portfolios

# Fixed label encoder
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.fit(REGIME_ORDER)


# =============================================================
# RULE LABELS  — circularity fix
# =============================================================
def build_rule_labels(X_df: pd.DataFrame) -> pd.Series:
    vix   = X_df["Volatility_Risk_Sentiment_features__VIX"]
    dd    = X_df["equity_market_dynamics_features__SPX_drawdown"]
    vix_s = vix.rolling(SMOOTH_WINDOW, min_periods=1).mean()
    dd_s  = dd.rolling(SMOOTH_WINDOW,  min_periods=1).mean()
    def classify(row):
        if row["dd"] < DD_BEAR_THRESH or row["vix"] >= VIX_BEAR_MIN: return "Bear"
        elif row["vix"] >= VIX_BULL_MAX:                               return "Transition"
        return "Bull"
    return pd.DataFrame({"vix":vix_s,"dd":dd_s}).apply(classify,axis=1)


# =============================================================
# ASYMMETRIC REGIME SIGNAL
# =============================================================
def build_asymmetric_regime_signal(
    regime_probs_df,
    confirm_enter=CONFIRM_DAYS_ENTER,
    confirm_exit=CONFIRM_DAYS_EXIT,
    switch_threshold=SWITCH_THRESHOLD,
    min_prob_gap=MIN_PROB_GAP,
) -> pd.Series:
    col_to_regime = {f"prob_{r}": r for r in REGIME_ORDER}
    raw_dominant  = regime_probs_df.idxmax(axis=1).map(col_to_regime)
    max_prob      = regime_probs_df.max(axis=1)
    confirmed     = pd.Series(index=regime_probs_df.index, dtype=object)
    current       = None

    for i in range(len(raw_dominant)):
        new = raw_dominant.iloc[i]; p = max_prob.iloc[i]
        window_size = confirm_exit if (current in ["Bear","Transition"] and new=="Bull") else confirm_enter
        if i < window_size-1:
            confirmed.iloc[i] = current if current else new; continue
        window = raw_dominant.iloc[i-window_size+1:i+1]
        if current is not None and new != current:
            gap_ok = (regime_probs_df[f"prob_{new}"].iloc[i] -
                      regime_probs_df[f"prob_{current}"].iloc[i]) >= min_prob_gap
        else:
            gap_ok = True
        if window.nunique()==1 and p>=switch_threshold and gap_ok:
            confirmed.iloc[i] = new; current = new
        else:
            confirmed.iloc[i] = current if current else new

    confirmed = confirmed.ffill()
    fv = confirmed.first_valid_index()
    if fv: confirmed.iloc[:confirmed.index.get_loc(fv)] = confirmed[fv]
    print("Confirmed regime counts (asymmetric + gap filter):")
    print(confirmed.value_counts().to_string())
    return confirmed


# =============================================================
# SINGLE FOLD FITTER — CORE UNCHANGED
# =============================================================
def fit_and_predict_fold(
    X_all, r_all, benchmark_series,
    train_end, predict_end,
    risk_profile="Moderate", custom_gamma=None,
    progress_callback=None,
):
    X_tr = X_all[X_all.index <= train_end]
    r_tr = r_all[r_all.index <= train_end]
    if len(X_tr) < 500:
        print(f"  ⚠️  {train_end.date()} — train too small"); return None

    if progress_callback: progress_callback("discriminators")
    X_tr_aug  = add_regime_discriminators(X_tr)
    X_pr_full = add_regime_discriminators(X_all[X_all.index <= predict_end])
    X_pr_aug  = X_pr_full[(X_pr_full.index > train_end) & (X_pr_full.index <= predict_end)]
    if len(X_pr_aug) == 0: return None

    if progress_callback: progress_callback("pca")
    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr_aug)
    X_pr_s    = scaler.transform(X_pr_aug)
    pca       = PCA(n_components=N_PCA, random_state=GMM_SEED)
    X_tr_pca  = pca.fit_transform(X_tr_s)
    pca.transform(X_pr_s)

    if progress_callback: progress_callback("gmm")
    gmm = GaussianMixture(n_components=N_REGIMES, covariance_type=GMM_COV_TYPE,
                          reg_covar=GMM_REG, n_init=GMM_N_INIT, random_state=GMM_SEED)
    gmm.fit(X_tr_pca)

    if progress_callback: progress_callback("labels")
    rule_shifted = build_rule_labels(X_pr_full).shift(-1)
    tr_rule      = rule_shifted.reindex(X_tr_aug.index).dropna()
    if len(tr_rule) == 0: return None

    if progress_callback: progress_callback("lag_features")
    combined_lag = add_lag_features(pd.concat([X_tr_aug, X_pr_aug]))
    Xf_tr = combined_lag.loc[X_tr_aug.index].dropna()
    Xf_pr = combined_lag.loc[X_pr_aug.index].dropna()
    drop_p = [c for c in RULE_DROP_COLS if c in Xf_tr.columns]
    Xf_tr  = Xf_tr.drop(columns=drop_p)
    Xf_pr  = Xf_pr.drop(columns=drop_p)
    common_tr = Xf_tr.index.intersection(tr_rule.index)
    if len(common_tr) == 0: return None
    Xf_tr = Xf_tr.loc[common_tr]
    y_str = tr_rule.loc[common_tr].values
    y_enc = LABEL_ENCODER.transform(y_str)

    if progress_callback: progress_callback("xgboost")
    classes, counts = np.unique(y_enc, return_counts=True)
    cw = {int(c): len(y_enc)/(len(classes)*cnt) for c,cnt in zip(classes,counts)}
    sw = np.array([cw[c] for c in y_enc])
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(Xf_tr, y_enc, sample_weight=sw)
    if len(Xf_pr) == 0: return None
    prob_arr = xgb.predict_proba(Xf_pr)
    prob_df  = pd.DataFrame(prob_arr, index=Xf_pr.index,
                             columns=[f"prob_{c}" for c in LABEL_ENCODER.classes_])

    if progress_callback: progress_callback("portfolio")
    yield_trend_at_refit = X_tr_aug["_yield_trend_63d"].iloc[-1]
    r_tr_t    = r_tr.rename(columns=TICKER_MAP).reindex(
                    columns=benchmark_series.index).dropna(how="all")
    tr_rule_al= tr_rule.reindex(Xf_tr.index)
    regime_ports_df = build_regime_portfolios(
        r_tr_t, tr_rule_al, benchmark_series,
        yield_trend_at_refit, risk_profile, custom_gamma,
    )

    bear_tlt = regime_ports_df.loc["Bear","TLT"] if "TLT" in regime_ports_df.columns else 0
    bear_ief = regime_ports_df.loc["Bear","IEF"] if "IEF" in regime_ports_df.columns else 0
    bear_shy = regime_ports_df.loc["Bear","SHY"] if "SHY" in regime_ports_df.columns else 0
    print(f"  [{train_end.date()} → {predict_end.date()}]  "
          f"train={len(X_tr_aug)}d  predict={len(prob_df)}d  "
          f"YldTrend={yield_trend_at_refit:.3f}  "
          f"Bear:TLT={bear_tlt:.0%} IEF={bear_ief:.0%} SHY={bear_shy:.0%}  "
          f"Bull:Eq={regime_ports_df.loc['Bull',EQUITY_T].sum():.0%}")
    return {"probs": prob_df, "portfolios": regime_ports_df}


# =============================================================
# WALK-FORWARD LOOP
# =============================================================
def run_walk_forward(
    X_all, r_all, benchmark_series,
    risk_profile="Moderate", custom_gamma=None,
    first_train_end=FIRST_TRAIN_END, refit_months=REFIT_MONTHS,
    progress_callback=None,
):
    """
    Main entry point — called by dashboard and run_all.py.
    progress_callback(fold_num, total_folds, step_name) → update UI
    """
    first_train_end = pd.Timestamp(first_train_end)
    all_dates       = X_all.index
    predict_periods = []
    t = first_train_end
    while t < all_dates[-1]:
        predict_end = min(t + pd.DateOffset(months=refit_months), all_dates[-1])
        predict_periods.append((t, predict_end))
        t = predict_end

    total_folds = len(predict_periods)
    print(f"\nML Walk-Forward: {total_folds} folds, refit every {refit_months}m\n")

    all_probs: list = []; fold_portfolios: dict = {}

    for fold_num, (train_end, pred_end) in enumerate(predict_periods, 1):
        print(f"\n--- Fold {fold_num}/{total_folds} ---")

        def _cb(step):
            if progress_callback:
                progress_callback(fold_num, total_folds, step)

        result = fit_and_predict_fold(
            X_all, r_all, benchmark_series, train_end, pred_end,
            risk_profile, custom_gamma, _cb,
        )
        if result is None: continue
        all_probs.append(result["probs"])
        for dt in result["probs"].index:
            fold_portfolios[dt] = result["portfolios"]

    if not all_probs:
        raise ValueError("All folds returned None — check data paths.")

    regime_probs_wf = pd.concat(all_probs).sort_index()
    regime_probs_wf = regime_probs_wf[~regime_probs_wf.index.duplicated(keep="last")]
    print(f"\n✅  ML Walk-Forward complete  "
          f"{regime_probs_wf.index[0].date()} → {regime_probs_wf.index[-1].date()}  "
          f"({len(regime_probs_wf)} days)")
    return regime_probs_wf, fold_portfolios
