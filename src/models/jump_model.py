# =============================================================
# src/models/jump_model.py
# Continuous Jump Model (CJM) — Walk-Forward Regime Detection
# Author: Ojaus Mane
# Engine: jumpmodels library (Aydinhan, Kolm, Mulvey, Shu 2024)
#
# Install engine first:
#   pip install -e <path/to/jump-models/>
#
# This wrapper keeps identical inputs/outputs to all other
# models in this project so the dashboard calls it the same way.
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    REFIT_MONTHS, FIRST_TRAIN_END, TICKER_MAP, REGIME_ORDER,
)
from features.feature_engineering import add_regime_discriminators
from portfolio.portfolio_allocator import build_regime_portfolios

# ── Import colleague's engine ─────────────────────────────────

from src.models.jump_models.jumpmodels.jump import JumpModel as _CJM
from src.models.jump_models.jumpmodels.preprocess import DataClipperStd as _Clipper
CJM_AVAILABLE = True


# ── Hyperparameters (from colleague's sweep results) ──────────
# jump_penalty=250 best default from hyperparameter sweep
# cont=True        → Continuous Jump Model (CJM), not discrete JM
# mode_loss=True   → mode loss penalty for TPM correspondence
# clip_mul=3.0     → winsorize features at ±3 std (colleague's preprocessing)
CJM_N_COMPONENTS = 3        # Bull, Transition, Bear
CJM_JUMP_PENALTY = 250.0
CJM_GRID_SIZE    = 0.05
CJM_MODE_LOSS    = True
CJM_N_INIT       = 3        # reduced for walk-forward speed
CJM_MAX_ITER     = 500
CJM_CLIP_MUL     = 3.0      # winsorization multiplier
CJM_SORT_BY      = "cumret" # sort states: state 0 = highest cumret = Bull

# SPX column used for state sorting (same as colleague's DEFAULT_RET_COL)
SPX_COL = "equity_market_dynamics_features__SPX_return"


# =============================================================
# PREPROCESSING — matches colleague's exact pipeline
# =============================================================

def _get_spx_ret(X: pd.DataFrame) -> np.ndarray:
    """Extract SPX return series for regime sorting."""
    if SPX_COL in X.columns:
        return X[SPX_COL].fillna(0).to_numpy()
    for col in X.columns:
        if "spx" in col.lower():
            return X[col].fillna(0).to_numpy()
    return np.zeros(len(X))


# =============================================================
# REGIME MAPPING
# State 0 = highest cumret = Bull (sorted by cumret in .fit())
# =============================================================

def _map_proba_to_regime_df(proba_arr: np.ndarray, index) -> pd.DataFrame:
    """
    Map CJM integer-state probability array → prob_Bull/Transition/Bear.
    States are sorted by cumret so state 0 = Bull, 1 = Transition, 2 = Bear.
    """
    n_states = proba_arr.shape[1]
    regime_names = ["Bull", "Transition", "Bear"][:n_states]

    df = pd.DataFrame(index=index)
    for i, regime in enumerate(regime_names):
        df[f"prob_{regime}"] = proba_arr[:, i]

    # Ensure all three columns exist
    for r in REGIME_ORDER:
        if f"prob_{r}" not in df.columns:
            df[f"prob_{r}"] = 0.0

    # Normalise rows
    row_sums = df.sum(axis=1).replace(0, 1)
    return df.div(row_sums, axis=0)


# =============================================================
# SINGLE FOLD
# =============================================================

def fit_and_predict_fold(
    X_all, r_all, benchmark_series,
    train_end, predict_end,
    risk_profile="Moderate", custom_gamma=None,
    progress_callback=None,
):
    if not CJM_AVAILABLE:
        raise RuntimeError(
            "jumpmodels not installed.\n"
            "Run: pip install -e <path/to/jump-models/>"
        )

    X_tr = X_all[X_all.index <= train_end]
    r_tr = r_all[r_all.index <= train_end]
    if len(X_tr) < 500:
        return None

    # ── Feature engineering ───────────────────────────────────
    if progress_callback: progress_callback("features")
    X_tr_aug  = add_regime_discriminators(X_tr)
    X_pr_full = add_regime_discriminators(X_all[X_all.index <= predict_end])
    X_pr_aug  = X_pr_full[
        (X_pr_full.index > train_end) & (X_pr_full.index <= predict_end)
    ]
    if len(X_pr_aug) == 0:
        return None

    # ── Preprocessing — colleague's exact pipeline ────────────
    if progress_callback: progress_callback("jump_fit")

    # Step 1: DataClipperStd — fit on train only (library class)
    clipper   = _Clipper(mul=CJM_CLIP_MUL)
    X_tr_clip = clipper.fit_transform(X_tr_aug.to_numpy())
    X_pr_clip = clipper.transform(X_pr_aug.to_numpy())

    # Step 2: StandardScaler — fit on train only
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_clip)   # numpy (n_train, n_feat)
    X_pr_sc  = scaler.transform(X_pr_clip)        # numpy (n_test,  n_feat)

    # SPX returns for state sorting
    spx_train = _get_spx_ret(X_tr_aug)

    # ── Fit CJM ───────────────────────────────────────────────
    model = _CJM(
        n_components = CJM_N_COMPONENTS,
        jump_penalty = CJM_JUMP_PENALTY,
        cont         = True,
        grid_size    = CJM_GRID_SIZE,
        mode_loss    = CJM_MODE_LOSS,
        random_state = 42,
        max_iter     = CJM_MAX_ITER,
        n_init       = CJM_N_INIT,
        verbose      = 0,
    )
    model.fit(X_tr_sc, ret_ser=spx_train, sort_by=CJM_SORT_BY)

    # ── Online predict — pass test array only (matches colleague)
    # colleague: cjm.predict_proba_online(X_test_arr)
    if progress_callback: progress_callback("jump_predict")
    proba_online = model.predict_proba_online(X_pr_sc)  # (n_test, n_states)
    prob_df      = _map_proba_to_regime_df(proba_online, X_pr_aug.index)

    # ── Portfolio construction ─────────────────────────────────
    if progress_callback: progress_callback("portfolio")

    # In-sample labels for MV portfolio (already sorted: 0=Bull)
    in_sample_labels = model.labels_  # shape (n_train,)
    regime_names     = ["Bull", "Transition", "Bear"][:CJM_N_COMPONENTS]
    mapped_tr = pd.Series(
        [regime_names[min(int(s), len(regime_names)-1)]
         for s in in_sample_labels],
        index=X_tr_aug.index,
    )

    yield_trend = X_tr_aug["_yield_trend_63d"].iloc[-1]
    r_tr_t = r_tr.rename(columns=TICKER_MAP).reindex(
        columns=benchmark_series.index
    ).dropna(how="all")

    regime_ports_df = build_regime_portfolios(
        r_tr_t, mapped_tr, benchmark_series,
        yield_trend, risk_profile, custom_gamma,
    )

    sw = int((prob_df.idxmax(axis=1) != prob_df.idxmax(axis=1).shift()).sum())
    print(f"  [CJM {train_end.date()} -> {predict_end.date()}]  "
          f"train={len(X_tr_sc)}d  predict={len(prob_df)}d  "
          f"jp={CJM_JUMP_PENALTY}  switches~{sw}")

    return {"probs": prob_df, "portfolios": regime_ports_df}


# =============================================================
# WALK-FORWARD LOOP
# Identical signature to all other models — called the same way
# =============================================================

def run_walk_forward(
    X_all, r_all, benchmark_series,
    risk_profile="Moderate", custom_gamma=None,
    first_train_end=FIRST_TRAIN_END, refit_months=REFIT_MONTHS,
    progress_callback=None,
):
    first_train_end = pd.Timestamp(first_train_end)
    all_dates       = X_all.index
    predict_periods = []
    t = first_train_end
    while t < all_dates[-1]:
        predict_end = min(t + pd.DateOffset(months=refit_months), all_dates[-1])
        predict_periods.append((t, predict_end))
        t = predict_end

    total_folds = len(predict_periods)
    print(f"\nJump Walk-Forward: {total_folds} folds\n")
    all_probs = []; fold_portfolios = {}

    for fold_num, (train_end, pred_end) in enumerate(predict_periods, 1):
        print(f"\n--- Jump Fold {fold_num}/{total_folds} ---")

        def _cb(step):
            if progress_callback:
                progress_callback(fold_num, total_folds, step)

        result = fit_and_predict_fold(
            X_all, r_all, benchmark_series, train_end, pred_end,
            risk_profile, custom_gamma, _cb,
        )
        if result is None:
            continue

        all_probs.append(result["probs"])
        for dt in result["probs"].index:
            fold_portfolios[dt] = result["portfolios"]

    if not all_probs:
        raise ValueError("Jump: all folds returned None")

    regime_probs_wf = pd.concat(all_probs).sort_index()
    regime_probs_wf = regime_probs_wf[~regime_probs_wf.index.duplicated(keep="last")]
    print(f"\n✅  Jump Walk-Forward complete  ({len(regime_probs_wf)} days)")
    return regime_probs_wf, fold_portfolios
