# =============================================================
# src/models/hmm_model.py
# Hidden Markov Model Regime Detector
# Authors: Niti Bhavesh Shah / Darshan Nanjegowda
# Reference: HMM vs HSMM (2024)
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import (
    N_PCA, PCA_SEED, HMM_N_COMPONENTS, HMM_N_ITER, HMM_COV_TYPE, HMM_TOL,
    REFIT_MONTHS, FIRST_TRAIN_END, TICKER_MAP, REGIME_ORDER,
)
from features.feature_engineering import add_regime_discriminators
from portfolio.portfolio_allocator import build_regime_portfolios

# Regime mapping: HMM states → Bull/Transition/Bear
# Updated after characterize_hmm_regimes() is called
HMM_REGIME_MAP = {0: "Bear", 1: "Transition", 2: "Bull"}


def _check_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM
    except ImportError:
        raise ImportError("HMM model requires hmmlearn: pip install hmmlearn")


# =============================================================
# CHARACTERIZE HMM REGIMES  (call after fit to update map)
# =============================================================
def characterize_hmm_regimes(model, feature_names=None) -> dict:
    """
    Print HMM regime means and suggest Bull/Trans/Bear mapping.
    Call this after fitting on training data to validate the map.
    """
    means = model.means_
    n     = model.n_components

    # Use first component (usually most variance-bearing PC) to rank
    ranking = np.argsort(means[:, 0])  # ascending = Bear first if PC1 ~ risk factor

    suggested = {}
    labels = ["Bear", "Transition", "Bull"] if n == 3 else [f"regime_{i}" for i in range(n)]
    for rank, state in enumerate(ranking):
        suggested[state] = labels[min(rank, len(labels)-1)]
        print(f"  HMM State {state}: mean_PC1={means[state,0]:.3f}  → {suggested[state]}")

    print("\nTransition Matrix:")
    tm = pd.DataFrame(model.transmat_.round(3),
                      index=[f"S{i}" for i in range(n)],
                      columns=[f"S{i}" for i in range(n)])
    print(tm.to_string())
    return suggested


# =============================================================
# PREDICT PROBABILITIES + MAP TO REGIME NAMES
# =============================================================
def predict_regime_probs(model, X_pca, index, regime_map=HMM_REGIME_MAP) -> pd.DataFrame:
    """
    Run HMM posterior decoding on X_pca.
    Returns DataFrame with columns prob_Bull, prob_Transition, prob_Bear.
    """
    posteriors = model.predict_proba(X_pca)   # (T, n_components)

    # Aggregate posteriors by regime name
    prob_dict = {r: np.zeros(len(X_pca)) for r in REGIME_ORDER}
    for state, regime in regime_map.items():
        if state < posteriors.shape[1]:
            prob_dict[regime] += posteriors[:, state]

    df = pd.DataFrame({f"prob_{r}": prob_dict[r] for r in REGIME_ORDER}, index=index)
    # Normalize rows to sum to 1
    row_sums = df.sum(axis=1)
    df = df.div(row_sums, axis=0).fillna(1/len(REGIME_ORDER))
    return df


# =============================================================
# CONFIRMED SIGNAL  (simple argmax with min-hold)
# =============================================================
def build_hmm_signal(prob_df, min_hold_days=None) -> pd.Series:
    if min_hold_days is None:
        min_hold_days = {"Bull": 21, "Transition": 15, "Bear": 10}
    col_map    = {f"prob_{r}": r for r in REGIME_ORDER}
    dominant   = prob_df.idxmax(axis=1).map(col_map)
    confirmed  = pd.Series(index=prob_df.index, dtype=object)
    current    = None; days_held = 0

    for i, (dt, new) in enumerate(dominant.items()):
        hold = min_hold_days.get(current, 15) if current else 0
        if current is None or (new != current and days_held >= hold):
            current = new; days_held = 0
        confirmed.iloc[i] = current
        days_held += 1

    confirmed = confirmed.ffill()
    print("HMM confirmed regime counts:")
    print(confirmed.value_counts().to_string())
    return confirmed


# =============================================================
# SINGLE FOLD
# =============================================================
def fit_and_predict_fold(
    X_all, r_all, benchmark_series,
    train_end, predict_end,
    risk_profile="Moderate", custom_gamma=None,
    regime_map=None, progress_callback=None,
):
    GaussianHMM = _check_hmmlearn()
    if regime_map is None: regime_map = HMM_REGIME_MAP

    X_tr = X_all[X_all.index <= train_end]
    r_tr = r_all[r_all.index <= train_end]
    if len(X_tr) < 500: return None

    if progress_callback: progress_callback("features")
    X_tr_aug  = add_regime_discriminators(X_tr)
    X_pr_full = add_regime_discriminators(X_all[X_all.index <= predict_end])
    X_pr_aug  = X_pr_full[(X_pr_full.index > train_end) & (X_pr_full.index <= predict_end)]
    if len(X_pr_aug) == 0: return None

    if progress_callback: progress_callback("pca")
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr_aug)
    X_pr_s   = scaler.transform(X_pr_aug)
    pca      = PCA(n_components=N_PCA, random_state=PCA_SEED)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_pr_pca = pca.transform(X_pr_s)

    if progress_callback: progress_callback("hmm_fit")
    model = GaussianHMM(
        n_components=HMM_N_COMPONENTS,
        covariance_type=HMM_COV_TYPE,
        n_iter=HMM_N_ITER,
        tol=HMM_TOL,
        random_state=PCA_SEED,
        verbose=False,
    )
    model.fit(X_tr_pca)
    if progress_callback: progress_callback("hmm_predict")
    prob_df = predict_regime_probs(model, X_pr_pca, X_pr_aug.index, regime_map)

    if progress_callback: progress_callback("portfolio")
    # Use HMM Viterbi labels to build rule-equivalent labels for portfolio
    viterbi_tr  = model.predict(X_tr_pca)
    mapped_tr   = pd.Series([regime_map.get(s,"Bull") for s in viterbi_tr],
                             index=X_tr_aug.index)
    yield_trend = X_tr_aug["_yield_trend_63d"].iloc[-1]
    r_tr_t      = r_tr.rename(columns=TICKER_MAP).reindex(
                      columns=benchmark_series.index).dropna(how="all")

    regime_ports_df = build_regime_portfolios(
        r_tr_t, mapped_tr, benchmark_series,
        yield_trend, risk_profile, custom_gamma,
    )
    print(f"  [HMM {train_end.date()} → {predict_end.date()}]  "
          f"converged={model.monitor_.converged}  "
          f"train={len(X_tr_aug)}d  predict={len(prob_df)}d")
    return {"probs": prob_df, "portfolios": regime_ports_df}


# =============================================================
# WALK-FORWARD LOOP
# =============================================================
def run_walk_forward(
    X_all, r_all, benchmark_series,
    risk_profile="Moderate", custom_gamma=None,
    first_train_end=FIRST_TRAIN_END, refit_months=REFIT_MONTHS,
    regime_map=None, progress_callback=None,
):
    first_train_end = pd.Timestamp(first_train_end)
    all_dates       = X_all.index
    predict_periods = []
    t = first_train_end
    while t < all_dates[-1]:
        predict_end = min(t + pd.DateOffset(months=refit_months), all_dates[-1])
        predict_periods.append((t, predict_end)); t = predict_end

    total_folds = len(predict_periods)
    print(f"\nHMM Walk-Forward: {total_folds} folds, refit every {refit_months}m\n")
    all_probs = []; fold_portfolios = {}

    for fold_num, (train_end, pred_end) in enumerate(predict_periods, 1):
        print(f"\n--- HMM Fold {fold_num}/{total_folds} ---")
        def _cb(step):
            if progress_callback: progress_callback(fold_num, total_folds, step)
        result = fit_and_predict_fold(
            X_all, r_all, benchmark_series, train_end, pred_end,
            risk_profile, custom_gamma, regime_map, _cb,
        )
        if result is None: continue
        all_probs.append(result["probs"])
        for dt in result["probs"].index:
            fold_portfolios[dt] = result["portfolios"]

    if not all_probs: raise ValueError("HMM: all folds returned None")
    regime_probs_wf = pd.concat(all_probs).sort_index()
    regime_probs_wf = regime_probs_wf[~regime_probs_wf.index.duplicated(keep="last")]
    print(f"\n✅  HMM Walk-Forward complete  ({len(regime_probs_wf)} days)")
    return regime_probs_wf, fold_portfolios
