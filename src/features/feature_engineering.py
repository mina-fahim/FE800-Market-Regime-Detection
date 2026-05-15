# =============================================================
# src/features/feature_engineering.py
# Shared across all models
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from config import N_PCA, PCA_SEED, N_REGIMES, GMM_COV_TYPE, GMM_REG, GMM_N_INIT, GMM_SEED

MOMENTUM_COLS = [
    "equity_market_dynamics_features__SPX_return",
    "equity_market_dynamics_features__SPX_volatility_20d",
    "Volatility_Risk_Sentiment_features__VIX",
    "Volatility_Risk_Sentiment_features__VIXChange",
    "credit_features__HY_IG_Spread",
    "credit_features__HY_Change",
    "Interest_Rate_Environment_features__Slope_10_2",
    "CrossAsset_RiskSignals_features__Equity_bond_corr",
    "Safe_Haven_Features__GLD_SPX_Ratio_Return",
    "_yield_trend_63d",
    "_vix_yield_interaction",
]


def add_regime_discriminators(df: pd.DataFrame) -> pd.DataFrame:
    out  = df.copy()
    gt10 = df["Interest_Rate_Environment_features__GT10 Govt"].shift(1)
    out["_yield_trend_63d"]       = gt10 - gt10.shift(63)
    spx  = df["equity_market_dynamics_features__SPX_return"].shift(1)
    out["_spx_cum_21d"]           = (1+spx).rolling(21).apply(lambda x: x.prod(), raw=True) - 1
    vix  = df["Volatility_Risk_Sentiment_features__VIX"].shift(1)
    out["_vix_yield_interaction"] = vix * out["_yield_trend_63d"]
    return out.dropna()


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MOMENTUM_COLS:
        if col not in df.columns:
            continue
        for w in [5, 10, 21]:
            rolled = df[col].shift(1).rolling(w)
            out[f"{col}__rmean_{w}d"] = rolled.mean()
            out[f"{col}__rstd_{w}d"]  = rolled.std()
    return out


def fit_pca_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, n_pca: int = N_PCA):
    scaler      = StandardScaler()
    X_tr_s      = scaler.fit_transform(X_train)
    X_te_s      = scaler.transform(X_test)
    pca         = PCA(n_components=n_pca, random_state=PCA_SEED)
    X_tr_pca    = pca.fit_transform(X_tr_s)
    X_te_pca    = pca.transform(X_te_s)
    cum_var     = pca.explained_variance_ratio_.cumsum()
    print(f"[PCA]  {n_pca} components → {cum_var[-1]*100:.1f}% variance")
    return X_tr_pca, X_te_pca, scaler, pca


def transform_new(X: pd.DataFrame, scaler, pca) -> np.ndarray:
    return pca.transform(scaler.transform(X))


# --- GMM exploratory (full data, not walk-forward) ---
def run_gmm_analysis(X_train_pca, n_regimes=N_REGIMES):
    gmm = GaussianMixture(n_components=n_regimes, covariance_type=GMM_COV_TYPE,
                          reg_covar=GMM_REG, n_init=GMM_N_INIT, random_state=GMM_SEED)
    gmm.fit(X_train_pca)
    labels = gmm.predict(X_train_pca)
    probs  = gmm.predict_proba(X_train_pca)
    return gmm, labels, probs


def aic_bic_search(X_pca, k_range=range(2, 11), n_init=20):
    results = []
    for k in k_range:
        g = GaussianMixture(n_components=k, covariance_type=GMM_COV_TYPE,
                            reg_covar=GMM_REG, n_init=n_init,
                            random_state=GMM_SEED).fit(X_pca)
        results.append({"k": k, "BIC": g.bic(X_pca), "AIC": g.aic(X_pca)})
    return pd.DataFrame(results)


def silhouette_analysis(X_pca, k_range=(3, 4, 5)):
    results = {}
    for k in k_range:
        g = GaussianMixture(n_components=k, covariance_type=GMM_COV_TYPE,
                            reg_covar=GMM_REG, n_init=GMM_N_INIT,
                            random_state=GMM_SEED).fit(X_pca)
        labels = g.predict(X_pca)
        score  = silhouette_score(X_pca, labels, sample_size=3000, random_state=42)
        results[k] = round(score, 4)
        print(f"  k={k}  Silhouette={score:.4f}")
    return results


def compute_transition_matrix(labels: np.ndarray, n: int) -> pd.DataFrame:
    mat = np.zeros((n, n))
    for i in range(len(labels) - 1):
        mat[labels[i], labels[i+1]] += 1
    mat = mat / mat.sum(axis=1, keepdims=True)
    return pd.DataFrame(mat, index=range(n), columns=range(n)).round(3)
