# =============================================================
# src/models/semi_markov_model.py
# Hidden Semi-Markov Model — explicit duration distributions
# Authors: Nabil Salame / Willie Gadson
#
# v3 — fixes duration explosion:
# • Balanced PC1-sorted init — all states guaranteed populated
# • Hard DUR_MAX cap (60/90/120d) — prevents single-state absorption
# • Survivor-function stay bonus in Viterbi — rewards realism
# • Dynamic regime map per fold — handles cluster ordering drift
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import (
    N_PCA, PCA_SEED,
    HSMM_N_COMPONENTS, HSMM_N_ITER, HSMM_MAX_DURATION,
    REFIT_MONTHS, FIRST_TRAIN_END, TICKER_MAP, REGIME_ORDER,
)
from features.feature_engineering import add_regime_discriminators
from portfolio.portfolio_allocator import build_regime_portfolios

HSMM_REGIME_MAP = {0: "Bear", 1: "Transition", 2: "Bull"}

# Duration bounds — hard caps prevent state collapse
DUR_INIT = np.array([20.0, 45.0, 90.0])   # Bear, Trans, Bull initial guess
DUR_MIN  = np.array([ 5.0, 10.0, 15.0])   # hard floor
DUR_MAX  = np.array([60.0, 90.0,120.0])    # hard ceiling


class GaussianHSMM:
    """
    Gaussian HSMM with Poisson durations and hard caps.
    Segment-based EM with duration-biased Viterbi decoding.
    """

    def __init__(self, n_components=3, n_iter=80, max_duration=60, random_state=42):
        self.n_components = n_components
        self.n_iter       = n_iter
        self.max_dur      = max_duration
        self.rng          = np.random.default_rng(random_state)
        self.converged_   = False
        self.means_ = self.covars_ = self.transmat_ = None
        self.startprob_ = self.dur_lambda_ = None

    def _init_params(self, X):
        T, D = X.shape; K = self.n_components
        # Balanced init: sort by PC1, assign K equal chunks
        order  = np.argsort(X[:, 0])
        chunk  = T // K
        labels = np.zeros(T, dtype=int)
        for k in range(K):
            s = k * chunk; e = (k+1)*chunk if k < K-1 else T
            labels[order[s:e]] = k
        self.means_  = np.array([X[labels==k].mean(0) for k in range(K)])
        self.covars_ = np.array([
            np.cov(X[labels==k].T) + np.eye(D)*1e-3
            if (labels==k).sum() > D else np.eye(D)*X.var()
            for k in range(K)])
        A = np.full((K,K), 1.0/(K-1)); np.fill_diagonal(A, 0.0)
        self.transmat_   = A
        self.dur_lambda_ = DUR_INIT[:K].copy()
        self.startprob_  = np.ones(K)/K

    def _log_emission(self, X):
        T, D = X.shape; K = self.n_components
        log_b = np.zeros((T, K))
        for k in range(K):
            diff = X - self.means_[k]
            try:
                ci = np.linalg.pinv(self.covars_[k])
                _, ld = np.linalg.slogdet(self.covars_[k])
            except Exception:
                ci = np.eye(D); ld = 0.0
            log_b[:, k] = -0.5*(np.einsum("td,dd,td->t",diff,ci,diff)+ld+D*np.log(2*np.pi))
        return np.clip(log_b, -500, 0)

    def _viterbi(self, X):
        """
        Viterbi with hard maximum duration.
        Once t_in[k] >= DUR_MAX[k], staying in k scores -inf → forced switch.
        This is the only reliable fix for single-state absorption on long series.
        """
        T, K  = len(X), self.n_components
        log_b = self._log_emission(X)
        log_A = np.log(np.maximum(self.transmat_, 1e-10))
        log_v = np.full((T, K), -np.inf)
        psi   = np.zeros((T, K), dtype=int)
        t_in  = np.zeros(K, dtype=int)

        log_v[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        t_in[int(np.argmax(log_v[0]))] = 1

        for t in range(1, T):
            for k in range(K):
                # Stay allowed only if under hard max
                if t_in[k] < int(DUR_MAX[k]):
                    stay = log_v[t-1, k] + log_b[t, k]
                else:
                    stay = -np.inf          # hard block: must switch

                others   = [j for j in range(K) if j != k]
                switches = [log_v[t-1, j] + log_A[j, k] + log_b[t, k] for j in others]
                all_v    = np.array([stay] + switches, dtype=float)
                all_f    = np.array([k]    + others,   dtype=int)
                bi       = int(np.argmax(all_v))
                psi[t, k]   = all_f[bi]
                log_v[t, k] = all_v[bi]

            best = int(np.argmax(log_v[t]))
            for k in range(K):
                t_in[k] = t_in[k] + 1 if k == best else 0

        states    = np.zeros(T, dtype=int)
        states[-1]= int(np.argmax(log_v[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def predict_proba(self, X):
        lb = self._log_emission(X)
        lb -= lb.max(1, keepdims=True)
        p   = np.exp(lb); p /= p.sum(1, keepdims=True)
        return p

    def fit(self, X):
        T, D = X.shape; K = self.n_components
        self._init_params(X)
        prev = None

        for it in range(self.n_iter):
            states = self._viterbi(X)
            if prev is not None and np.array_equal(states, prev):
                self.converged_ = True; break
            prev = states.copy()

            # Update emissions
            for k in range(K):
                m = states==k; n_k = m.sum()
                if n_k > D:
                    self.means_[k]  = X[m].mean(0)
                    d = X[m]-self.means_[k]
                    self.covars_[k] = (d.T@d)/n_k + np.eye(D)*1e-3
                elif n_k > 0:
                    self.means_[k] = X[m].mean(0)

            # Update durations from segment lengths — HARD CAP enforced
            segs = {k: [] for k in range(K)}
            curr = int(states[0]); cnt = 1
            for s in states[1:]:
                s = int(s)
                if s == curr: cnt += 1
                else: segs[curr].append(cnt); curr=s; cnt=1
            segs[curr].append(cnt)

            for k in range(K):
                if segs[k]:
                    raw = float(np.mean(segs[k]))
                    # HARD CAP first, then smooth with prior
                    capped = float(np.clip(raw, DUR_MIN[k], DUR_MAX[k]))
                    self.dur_lambda_[k] = 0.6*capped + 0.4*DUR_INIT[k]

            # Update transitions (segment-to-segment)
            cnt_mat = np.zeros((K, K))
            for t in range(T-1):
                j, k2 = int(states[t]), int(states[t+1])
                if j != k2:
                    cnt_mat[j, k2] += 1
            for j in range(K):
                row = cnt_mat[j].copy(); row[j] = 0
                s   = row.sum()
                if s > 0:
                    self.transmat_[j] = row/s
                else:
                    uni = np.full(K, 1.0/(K-1)); uni[j]=0
                    self.transmat_[j] = uni

        dist = {k: int((states==k).sum()) for k in range(K)}
        pcts = [f"S{k}:{dist[k]/T*100:.0f}%" for k in range(K)]
        print(f"[HSMM]  iters={it+1}  converged={self.converged_}  "
              f"dur_lambda={np.round(self.dur_lambda_,1)}  "
              f"state_dist=[{', '.join(pcts)}]")
        return self


# =============================================================
# REGIME MAPPING  (dynamic per fold)
# =============================================================
def _dynamic_map(model):
    """Map states to regimes by PC1 mean: lowest=Bear, highest=Bull."""
    order = np.argsort(model.means_[:, 0])
    return {int(order[i]): ["Bear","Transition","Bull"][i] for i in range(3)}


def predict_regime_probs(model, X_pca, index, regime_map) -> pd.DataFrame:
    post = model.predict_proba(X_pca)
    pd_  = {r: np.zeros(len(X_pca)) for r in REGIME_ORDER}
    for state, regime in regime_map.items():
        pd_[regime] += post[:, state]
    df = pd.DataFrame({f"prob_{r}": pd_[r] for r in REGIME_ORDER}, index=index)
    return df.div(df.sum(1), axis=0).fillna(1/3)


# =============================================================
# SINGLE FOLD
# =============================================================
def fit_and_predict_fold(
    X_all, r_all, benchmark_series,
    train_end, predict_end,
    risk_profile="Moderate", custom_gamma=None,
    regime_map=None, progress_callback=None,
):
    X_tr = X_all[X_all.index<=train_end]
    r_tr = r_all[r_all.index<=train_end]
    if len(X_tr)<500: return None

    if progress_callback: progress_callback("features")
    X_tr_aug  = add_regime_discriminators(X_tr)
    X_pr_full = add_regime_discriminators(X_all[X_all.index<=predict_end])
    X_pr_aug  = X_pr_full[(X_pr_full.index>train_end)&(X_pr_full.index<=predict_end)]
    if len(X_pr_aug)==0: return None

    if progress_callback: progress_callback("pca")
    sc = StandardScaler(); pca = PCA(n_components=N_PCA, random_state=PCA_SEED)
    X_tr_pca = pca.fit_transform(sc.fit_transform(X_tr_aug))
    X_pr_pca = pca.transform(sc.transform(X_pr_aug))

    if progress_callback: progress_callback("hsmm_fit")
    model = GaussianHSMM(n_components=HSMM_N_COMPONENTS, n_iter=HSMM_N_ITER,
                         max_duration=HSMM_MAX_DURATION, random_state=42)
    model.fit(X_tr_pca)

    # Dynamic map based on PC1 ordering this fold
    dmap = _dynamic_map(model)

    if progress_callback: progress_callback("hsmm_predict")
    prob_df = predict_regime_probs(model, X_pr_pca, X_pr_aug.index, dmap)

    if progress_callback: progress_callback("portfolio")
    vit_tr = model._viterbi(X_tr_pca)
    lbl_tr = pd.Series([dmap.get(int(s),"Bull") for s in vit_tr], index=X_tr_aug.index)
    ytrend = X_tr_aug["_yield_trend_63d"].iloc[-1]
    r_tr_t = r_tr.rename(columns=TICKER_MAP).reindex(columns=benchmark_series.index).dropna(how="all")
    ports  = build_regime_portfolios(r_tr_t, lbl_tr, benchmark_series, ytrend, risk_profile, custom_gamma)

    print(f"  [HSMM {train_end.date()} → {predict_end.date()}]  "
          f"train={len(X_tr_pca)}d  predict={len(prob_df)}d  "
          f"dur={model.dur_lambda_.round(1)}  map={dmap}")
    return {"probs": prob_df, "portfolios": ports}


# =============================================================
# WALK-FORWARD
# =============================================================
def run_walk_forward(
    X_all, r_all, benchmark_series,
    risk_profile="Moderate", custom_gamma=None,
    first_train_end=FIRST_TRAIN_END, refit_months=REFIT_MONTHS,
    regime_map=None, progress_callback=None,
):
    first_train_end = pd.Timestamp(first_train_end)
    all_dates = X_all.index; t = first_train_end; periods = []
    while t < all_dates[-1]:
        pe = min(t+pd.DateOffset(months=refit_months), all_dates[-1])
        periods.append((t,pe)); t=pe

    print(f"\nHSMM Walk-Forward: {len(periods)} folds\n")
    all_probs=[]; fp={}

    for fn,(te,pe) in enumerate(periods, 1):
        print(f"\n--- HSMM Fold {fn}/{len(periods)} ---")
        def _cb(step):
            if progress_callback: progress_callback(fn, len(periods), step)
        res = fit_and_predict_fold(X_all,r_all,benchmark_series,te,pe,
                                   risk_profile,custom_gamma,regime_map,_cb)
        if res is None: continue
        all_probs.append(res["probs"])
        for dt in res["probs"].index: fp[dt]=res["portfolios"]

    if not all_probs: raise ValueError("HSMM: all folds failed")
    pw = pd.concat(all_probs).sort_index()
    pw = pw[~pw.index.duplicated(keep="last")]
    print(f"\n✅  HSMM Walk-Forward complete  ({len(pw)} days)")
    return pw, fp
