
# =============================================================

from shiny import App, ui, render, reactive
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import threading
import queue
import os
import sys
import json
import pickle
import hashlib
import time
import itertools
from datetime import datetime


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.join(BASE_DIR, "config"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import (
    N_PCA, N_REGIMES, REGIME_MERGE_MAP,
    GMM_COV_TYPE, GMM_REG, GMM_N_INIT, GMM_SEED
)

from features.feature_engineering import (
    fit_pca_pipeline,
    run_gmm_analysis,
    aic_bic_search,
    silhouette_analysis,
    compute_transition_matrix
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

body { background:#0A1628 !important; color:#E8EDF5 !important; font-family:'DM Sans',sans-serif !important; }
.navbar { background:#06101F !important; border-bottom:2px solid #C9A84C !important; }
.navbar-brand { color:#C9A84C !important; font-weight:700 !important; font-size:17px !important; letter-spacing:0.05em !important; }
.navbar .nav-link { color:#7A8BA0 !important; font-size:13px !important; padding:14px 16px !important; }
.navbar .nav-link:hover, .navbar .nav-link.active { color:#C9A84C !important; border-bottom:2px solid #C9A84C !important; }
.card { background:#0F1E35 !important; border:1px solid #1E3050 !important; border-radius:10px !important; margin-bottom:16px !important; }
.card-header { background:#0D1E36 !important; border-bottom:1px solid #1E3050 !important; color:#C9A84C !important; font-size:12px !important; font-weight:600 !important; letter-spacing:0.08em !important; text-transform:uppercase !important; padding:12px 16px !important; }
.form-control, .form-select { background:#071120 !important; border:1px solid #1E3050 !important; color:#E8EDF5 !important; border-radius:6px !important; font-size:13px !important; }
.form-control:focus, .form-select:focus { border-color:#C9A84C !important; box-shadow:0 0 0 2px rgba(201,168,76,0.2) !important; }
label, .form-label, .control-label { color:#A8C4E0 !important; font-size:12px !important; font-weight:500 !important; margin-bottom:4px !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#06101F; }
::-webkit-scrollbar-thumb { background:#1E3050; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#C9A84C; }
.nav-tabs { border-bottom:1px solid #1E3050 !important; }
.nav-tabs .nav-link { color:#7A8BA0 !important; border:none !important; font-size:13px !important; }
.nav-tabs .nav-link.active { color:#C9A84C !important; border-bottom:2px solid #C9A84C !important; background:none !important; }
thead th, .dataframe th, table th { background:#0D1E36 !important; color:#C9A84C !important; font-size:11px !important; text-transform:uppercase !important; letter-spacing:0.08em !important; padding:10px 12px !important; border-bottom:1px solid #1E3050 !important; }
tbody td, .dataframe td, table td { background:#0F1E35 !important; color:#E8EDF5 !important; font-size:13px !important; border-bottom:1px solid #12243A !important; padding:8px 12px !important; }
tbody tr:hover td { background:#12243A !important; }
input[type=checkbox] { accent-color:#C9A84C !important; }

/* ── New tab shared styles ──────────────────────────── */
.section-lbl{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#7A8BA0;margin:0 0 10px;}
.btn-sm-gold{background:transparent!important;border:1px solid #C9A84C!important;color:#C9A84C!important;font-size:11px!important;font-weight:600!important;padding:4px 12px!important;border-radius:6px!important;cursor:pointer!important;}
.btn-sm-red{background:transparent!important;border:1px solid #E74C3C!important;color:#E74C3C!important;font-size:11px!important;font-weight:600!important;padding:4px 12px!important;border-radius:6px!important;cursor:pointer!important;}
/* Compare */
.cmp-card{background:#0A1628;border:1px solid #1E3050;border-radius:8px;padding:12px 14px;margin-bottom:8px;display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;}
.cmp-model{font-size:13px;font-weight:600;color:#E8EDF5;}
.cmp-meta{font-size:11px;color:#7A8BA0;margin-top:2px;}
/* Data analysis */
.da-controls{background:#0D1E36;border:1px solid #1E3050;border-radius:10px;padding:16px 20px;margin-bottom:16px;display:flex;gap:20px;align-items:flex-end;flex-wrap:wrap;}
.da-run-btn{background:linear-gradient(135deg,#1A6B3A,#27AE60)!important;border:none!important;color:white!important;font-weight:600!important;font-size:13px!important;padding:8px 24px!important;border-radius:8px!important;cursor:pointer!important;}


/* ── Run Model tab ───────────────────────────────────── */
.run-section-label {
    font-size:10px; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:#7A8BA0; margin:16px 0 8px;
}
.btn-run-model {
    background:linear-gradient(135deg,#8C1515,#B01E1E) !important;
    border:none !important; color:white !important; font-weight:700 !important;
    font-size:16px !important; padding:14px !important; border-radius:10px !important;
    width:100% !important; letter-spacing:0.05em !important; cursor:pointer !important;
    margin-top:12px !important;
    box-shadow:0 4px 20px rgba(140,21,21,0.4) !important;
}
.btn-run-model:hover { opacity:0.9 !important; transform:translateY(-1px) !important; }
.btn-run-model:disabled { opacity:0.5 !important; cursor:not-allowed !important; }
.status-pill {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:12px; font-weight:600; letter-spacing:0.05em;
}
.pill-ready   { background:rgba(30,48,80,0.8);   color:#7A8BA0; border:1px solid #1E3050; }
.pill-running { background:rgba(140,21,21,0.2);  color:#FF8080; border:1px solid #8C1515; }
.pill-done    { background:rgba(46,204,113,0.12);color:#2ECC71; border:1px solid rgba(46,204,113,0.3); }
.pill-cache   { background:rgba(201,168,76,0.12);color:#C9A84C; border:1px solid rgba(201,168,76,0.3); }
.pill-error   { background:rgba(231,76,60,0.12); color:#E74C3C; border:1px solid rgba(231,76,60,0.3); }
.log-box {
    background:#06101F; border:1px solid #1E3050; border-radius:8px;
    padding:10px 14px; max-height:180px; overflow-y:auto; margin-top:10px;
    font-family:'DM Mono',monospace; font-size:11px; line-height:1.8;
}
.log-line       { color:#7A8BA0; }
.log-line.ok    { color:#2ECC71; }
.log-line.err   { color:#E74C3C; }
.log-line.fold  { color:#C9A84C; }
.error-box {
    background:rgba(231,76,60,0.08); border:1px solid rgba(231,76,60,0.3);
    border-radius:10px; padding:20px 24px; margin-bottom:16px;
}
.error-title { font-size:16px; font-weight:700; color:#E74C3C; margin-bottom:8px; }
.error-file  {
    background:#06101F; border:1px solid #1E3050; border-radius:6px;
    padding:8px 14px; font-family:'DM Mono',monospace; font-size:12px;
    color:#C9A84C; margin:6px 0;
}
.cache-row {
    display:grid; grid-template-columns:1fr auto; gap:10px;
    align-items:end; margin-bottom:10px;
}
.btn-load-cache {
    background:transparent !important; border:1px solid #C9A84C !important;
    color:#C9A84C !important; font-size:12px !important; font-weight:600 !important;
    padding:7px 16px !important; border-radius:6px !important; cursor:pointer !important;
    white-space:nowrap !important;
}
.btn-load-cache:hover { background:rgba(201,168,76,0.1) !important; }


/* ── Welcome ─────────────────────────────────────────── */
.welcome-card { max-width:520px; margin:60px auto; background:#0F1E35; border:1px solid #1E3050; border-radius:16px; padding:48px 44px; text-align:center; }
.welcome-logo { font-size:48px; margin-bottom:16px; }
.welcome-title { font-size:26px; font-weight:700; color:#C9A84C; margin-bottom:8px; }
.welcome-sub { font-size:14px; color:#7A8BA0; line-height:1.6; margin-bottom:32px; }
.welcome-input-wrap { text-align:left; margin-bottom:18px; }
.welcome-input-wrap label { font-size:13px !important; color:#A8C4E0 !important; margin-bottom:6px !important; display:block !important; }
.btn-start { background:linear-gradient(135deg,#8C1515,#B01E1E) !important; border:none !important; color:white !important; font-weight:600 !important; font-size:15px !important; padding:12px 36px !important; border-radius:8px !important; width:100% !important; letter-spacing:0.04em !important; cursor:pointer !important; margin-top:8px !important; }

/* ── Survey ──────────────────────────────────────────── */
.survey-question { background:#0F1E35; border:1px solid #1E3050; border-radius:10px; padding:20px 24px; margin-bottom:14px; }
.survey-question .q-number { font-size:11px; font-weight:700; color:#C9A84C; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:6px; }
.survey-question .q-text { font-size:16px; font-weight:500; color:#E8EDF5; margin-bottom:16px; line-height:1.5; }
.survey-question .shiny-input-radiogroup .radio label { display:block !important; background:#0A1628 !important; border:1px solid #1E3050 !important; border-radius:8px !important; padding:11px 16px !important; margin-bottom:7px !important; cursor:pointer !important; font-size:14px !important; color:#A8C4E0 !important; transition:all 0.15s !important; font-weight:400 !important;width:100% !important; }
.survey-question .shiny-input-radiogroup .radio label:hover { border-color:#C9A84C !important; background:#0D1829 !important; color:#E8EDF5 !important; }
.survey-question .shiny-input-radiogroup .radio input[type="radio"] { accent-color:#C9A84C !important; margin-right:10px !important; }
.survey-question .shiny-input-radiogroup > label { display:none !important; }
.progress-wrap { margin-bottom:24px; }
.progress-label { display:flex; justify-content:space-between; font-size:12px; color:#7A8BA0; margin-bottom:6px; }
.progress-track { background:#0A1628; border-radius:4px; height:8px; border:1px solid #1E3050; overflow:hidden; }
.progress-fill { height:100%; border-radius:4px; transition:width 0.4s ease; }
.profile-card { border-radius:12px; padding:24px; border:2px solid; margin-bottom:14px; }
.profile-name { font-size:24px; font-weight:700; margin-bottom:8px; }
.profile-desc { font-size:13px; line-height:1.6; opacity:0.85; margin-bottom:14px; }
.gamma-row { display:flex; gap:10px; margin-bottom:10px; }
.gamma-box { flex:1; background:#0A1628; border:1px solid #1E3050; border-radius:8px; padding:12px; text-align:center; }
.gamma-box .g-label { font-size:10px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#7A8BA0; margin-bottom:5px; }
.gamma-box .g-value { font-size:22px; font-weight:700; }
.btn-apply { background:linear-gradient(135deg,#1A6B3A,#27AE60) !important; border:none !important; color:white !important; font-weight:600 !important; font-size:15px !important; padding:12px 28px !important; border-radius:8px !important; width:100% !important; cursor:pointer !important; margin-top:8px !important; }
.applied-banner { background:rgba(46,204,113,0.1); border:1px solid rgba(46,204,113,0.3); border-radius:8px; padding:12px; color:#2ECC71; font-size:13px; text-align:center; margin-top:8px; }
.age-cap-note { background:rgba(243,156,18,0.1); border:1px solid rgba(243,156,18,0.3); border-radius:6px; padding:8px 12px; color:#F39C12; font-size:12px; margin-bottom:10px; }

/* ── KPI Cards ───────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.kpi-card {
    background: #0F1E35;
    border: 1px solid #1E3050;
    border-radius: 10px;
    padding: 16px 14px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #C9A84C);
}
.kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7A8BA0;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.kpi-sub {
    font-size: 11px;
    color: #7A8BA0;
    margin-top: 4px;
}

/* ── Sidebar ─────────────────────────────────────────── */
.sidebar {
    background: #0F1E35 !important;
    border-right: 1px solid #1E3050 !important;
    padding: 16px !important;
}
.sidebar-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7A8BA0;
    margin: 16px 0 8px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #1E3050;
    margin: 16px 0;
}

/* ── Demo badge ──────────────────────────────────────── */
.demo-badge {
    display: inline-block;
    background: rgba(201,168,76,0.12);
    border: 1px solid rgba(201,168,76,0.3);
    color: #C9A84C;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 8px;
    vertical-align: middle;
}


"""

# ══════════════════════════════════════════════════════════════
# SCORING ENGINE  (unchanged from Step 3)
# ══════════════════════════════════════════════════════════════

def age_profile_cap(age):
    if age < 51:   return None
    if age < 61:   return "Moderate"
    if age < 71:   return "Moderate"
    return "Conservative"

def apply_age_cap(profile, cap):
    order = ["Conservative","Moderate","Aggressive"]
    if cap is None: return profile
    if profile not in order: profile = "Moderate"
    if cap not in order: cap = "Moderate"
    return order[min(order.index(profile), order.index(cap))]

def score_survey(q1,q2,q3,q4,q5,q6,q7,q8,age=40):
    base   = {1:6.0,2:4.0,3:2.5,4:1.5}.get(q4,4.0)
    adj    = ({1:+1.5,2:+0.5,3:0.0,4:-0.5}.get(q2,0)
            + {1:+0.8,2:0.0,3:-0.5}.get(q5,0)
            + {1:+0.8,2:+0.3,3:0.0,4:-0.3}.get(q6,0)
            + (0.3 if 35<=age<=50 else 0))
    floor  = {1:5.0,2:3.5,3:2.0,4:1.0}.get(q1,3.0)
    ceil_  = {1:8.0,2:5.5,3:3.5,4:2.0}.get(q3,5.0)
    bg     = max(floor, min(ceil_, base+adj))
    rel    = ({1:0.0,2:0.5,3:1.0,4:1.0}.get(q7,0.5)
            + {1:0.0,2:0.5,3:1.0,4:1.0}.get(q8,0.5)) / 2
    bg     = round(rel*bg + (1-rel)*3.5, 2)
    raw    = ("Conservative" if bg>=5 else "Moderate" if bg>=3.5
              else "Aggressive")
    cap    = age_profile_cap(age)
    prof   = apply_age_cap(raw, cap)
    bm     = {"Conservative":{"eq":.30,"bond":.60,"alt":.10},
              "Moderate":{"eq":.55,"bond":.40,"alt":.05},
              "Aggressive":{"eq":.85,"bond":.10,"alt":.05}}[prof]
    eq_b   = {"Conservative":{"bull_max":.60,"bull_min":.40,"bear_min":.20},
              "Moderate":{"bull_max":.75,"bull_min":.55,"bear_min":.15},
              "Aggressive":{"bull_max":.90,"bull_min":.70,"bear_min":.05}}[prof]
    return {"profile":prof,"raw_profile":raw,"age_capped":(prof!=raw),"cap_label":cap,
            "gamma_bull":round(bg*.30,2),"gamma_trans":round(bg*.55,2),"gamma_bear":bg,
            "benchmark":bm,"eq_bounds":eq_b}

def profile_meta(profile):
    return {"Conservative":{"color":"#4FC3F7","bg":"rgba(79,195,247,0.08)","border":"#4FC3F7","icon":"🛡️",
              "desc":"Your priority is protecting what you have. You prefer steady, predictable returns.",
              "bm_label":"30% Equity / 60% Bonds / 10% Gold"},
            "Moderate":{"color":"#C9A84C","bg":"rgba(201,168,76,0.08)","border":"#C9A84C","icon":"⚖️",
              "desc":"You want a balance between growth and protection. You can handle some market swings.",
              "bm_label":"55% Equity / 40% Bonds / 5% Gold"},
            "Aggressive":{"color":"#2ECC71","bg":"rgba(46,204,113,0.08)","border":"#2ECC71","icon":"🚀",
              "desc":"You prioritize maximum long-term growth and are comfortable with significant volatility.",
              "bm_label":"85% Equity / 10% Bonds / 5% Gold"}}.get(profile, 
            {"color":"#C9A84C","bg":"rgba(201,168,76,0.08)","border":"#C9A84C","icon":"⚖️",
              "desc":"You want a balance between growth and protection.",
              "bm_label":"55% Equity / 40% Bonds / 5% Gold"})

# ══════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# ══════════════════════════════════════════════════════════════

def gen_demo(init=100_000, profile="Moderate"):
    """
    Generate realistic synthetic backtest data.
    Returns three DataFrames: walk-forward, benchmark, MV no-regime.
    Each has columns: ret, cum, regime, rebalanced.
    """
    ANN   = 252
    dates = pd.bdate_range("2019-01-02", "2025-12-31")
    n     = len(dates)
    rng   = np.random.default_rng(42)

    # Profile-specific return parameters
    params = {
        "Conservative":        {"mu":0.082, "sig":0.088},
        "Moderate":            {"mu":0.133, "sig":0.124},
        "Aggressive":          {"mu":0.182, "sig":0.172},
    }.get(profile, {"mu":0.133,"sig":0.124})

    mu_d  = params["mu"] / ANN
    sig_d = params["sig"] / np.sqrt(ANN)

    # Regime sequence — realistic pattern
    regime_seq = np.where(
        (dates.year == 2020) & (dates.month.isin([2,3,4])), "Bear",
        np.where(
            (dates.year == 2022) & (dates.month >= 1), "Bear",
            np.where(dates.year.isin([2020,2021]) & (dates.month > 4), "Transition",
                     "Bull")
        )
    )

    # Rebalance events (regime switches)
    rebal = pd.Series(False, index=dates)
    switch_dates = [
        "2020-02-20","2020-03-23","2020-06-08",
        "2022-01-03","2022-06-16","2022-10-12",
        "2023-01-02","2024-01-02","2025-01-02",
    ]
    for sd in switch_dates:
        try:
            closest = dates[dates >= pd.Timestamp(sd)][0]
            rebal[closest] = True
        except IndexError:
            pass

    def make_bt(mu, sig, label="wf"):
        r  = rng.normal(mu, sig, n)
        to = np.where(rebal, rng.uniform(0.5, 1.5, n), 0.0)
        bt = pd.DataFrame({
            "ret":        r,
            "regime":     regime_seq,
            "rebalanced": rebal.values,
            "turnover":   to,
        }, index=dates)
        bt["cum"] = (1 + bt["ret"]).cumprod()
        return bt

    bt_wf  = make_bt(mu_d,         sig_d)
    bt_bm  = make_bt(0.084/ANN,    0.124/np.sqrt(ANN))
    bt_mv  = make_bt(0.109/ANN,    0.112/np.sqrt(ANN))
    return bt_wf, bt_bm, bt_mv


# ══════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════

# Dark theme dict applied to every chart
DARK = dict(
    plot_bgcolor  = "#0A1628",
    paper_bgcolor = "#0F1E35",
    font          = dict(family="DM Sans, sans-serif", color="#E8EDF5", size=12),
    legend        = dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#E8EDF5", size=11)),
    margin        = dict(l=50, r=20, t=30, b=50),
    xaxis         = dict(gridcolor="#1A2E48", zeroline=False,
                         tickfont=dict(color="#7A8BA0")),
    yaxis         = dict(gridcolor="#1A2E48", zeroline=False,
                         tickfont=dict(color="#7A8BA0")),
)

def apply_dark(fig, height=460):
    """Apply dark theme to any Plotly figure."""
    fig.update_layout(height=height, **DARK)
    fig.update_xaxes(gridcolor="#1A2E48", zeroline=False, linecolor="#1A2E48")
    fig.update_yaxes(gridcolor="#1A2E48", zeroline=False, linecolor="#1A2E48")
    return fig

def to_html(fig):
    """Convert Plotly figure to HTML string for render.ui."""
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))


# ══════════════════════════════════════════════════════════════
# MODEL CONFIG  (no rHSM)
# ══════════════════════════════════════════════════════════════

MODELS = {
    "ML (GMM + XGBoost)": "ML",
    "HMM":                "HMM",
    "Semi-Markov (HSMM)": "HSMM",
    "CJM":     "Jump",
}

# Expected data files — shown in error message if missing
REQUIRED_FILES = {
    "Feature matrix": "data/processed/features_model_ready_v1.csv",
    "ETF prices":     "data/etf_prices.csv",
    "Bond proxies":   "data/bond_return_proxies.csv",
}

# ══════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_INDEX = os.path.join(CACHE_DIR, "cache_index.json")
os.makedirs(CACHE_DIR, exist_ok=True)


def _make_key(model_key, profile, tc, refit, thresh):
    """Unique cache key from the parameters that affect results."""
    raw    = f"{model_key}_{profile}_{tc}_{refit}_{thresh}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{model_key}_{profile}_{digest}"


def _load_index():
    try:
        if os.path.exists(CACHE_INDEX):
            with open(CACHE_INDEX) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_index(idx):
    try:
        with open(CACHE_INDEX, "w") as f:
            json.dump(idx, f, indent=2, default=str)
    except Exception:
        pass


def cache_exists(key):
    return os.path.exists(os.path.join(CACHE_DIR, f"{key}.pkl"))


def load_cache(key):
    with open(os.path.join(CACHE_DIR, f"{key}.pkl"), "rb") as f:
        return pickle.load(f)

def hex_to_rgba(hex_color, alpha=0.08):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
    
def save_cache(key, result, meta):
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(result, f)
    idx = _load_index()
    idx = [e for e in idx if e.get("key") != key]
    idx.append({
        "key":      key,
        "model":    meta.get("model", ""),
        "profile":  meta.get("profile", ""),
        "sharpe":   meta.get("sharpe", 0),
        "ann_ret":  meta.get("ann_ret", ""),
        "switches": meta.get("switches", 0),
        "runtime":  meta.get("runtime", 0),
        "date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "size_mb":  round(os.path.getsize(path) / 1e6, 1),
    })
    _save_index(idx)


def cache_choices():
    """Return dict of {key: display_label} for dropdown."""
    idx = sorted(_load_index(),
                 key=lambda x: str(x.get("date", "")), reverse=True)
    out = {}
    for e in idx[:20]:
        k = e.get("key", "")
        if k:
            label = (f"{e.get('model','')} | {e.get('profile','')} | "
                     f"Sharpe {e.get('sharpe',0):.3f} | {str(e.get('date',''))[-8:]}")
            out[k] = label
    return out


def check_data_files():
    """
    Check which required data files exist.
    Returns (all_ok: bool, missing: list of (label, path))
    """
    base    = os.path.join(os.path.dirname(__file__), "..")
    missing = []
    for label, rel_path in REQUIRED_FILES.items():
        full = os.path.join(base, rel_path)
        if not os.path.exists(full):
            missing.append((label, rel_path))
    return (len(missing) == 0), missing


# ══════════════════════════════════════════════════════════════
# SURVEY QUESTION HELPER  (unchanged)
# ══════════════════════════════════════════════════════════════

def survey_q(number, total, qid, question_text, choices):
    full_choices = {"": "— Select your answer —", **choices}
    return ui.div(
        {"class": "survey-question"},
        ui.div(f"Question {number} of {total}", {"class": "q-number"}),
        ui.div(question_text, {"class": "q-text"}),
        ui.input_radio_buttons(id=qid, label=None,
                               choices=full_choices, selected=""),
    )


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

app_ui = ui.page_navbar(

    ui.head_content(ui.tags.style(CSS)),

    # ══════════════════════════════════════════════════════════
    # TAB 1 — RISK PROFILE
    # ══════════════════════════════════════════════════════════
    ui.nav_panel("🎯 Risk Profile",
        ui.output_ui("survey_page"),
    ),

    # ══════════════════════════════════════════════════════════
    # TAB 2 — OVERVIEW
    # ══════════════════════════════════════════════════════════
    ui.nav_panel("📊 Overview",

        # KPI row — full width above everything
        ui.output_ui("kpi_cards"),

        # Sidebar + chart
        # LESSON: layout_sidebar splits page into a narrow sidebar
        # and a wide main area. sidebar() defines the left panel.
        ui.layout_sidebar(
            ui.sidebar(

                ui.div("Display Options", {"class": "sidebar-label",
                                           "style": "margin-top:0;"}),

                # Date range filter
                # LESSON: input_date_range returns a tuple (start, end)
                # Read in server as: input.date_range()[0] and [1]
                ui.input_date_range(
                    "date_range", "Date Range",
                    start="2019-01-02",
                    end="2025-12-31",
                    min="2019-01-02",
                    max="2025-12-31",
                ),

                ui.tags.hr({"class": "sidebar-divider"}),
                ui.div("Show Strategies", {"class": "sidebar-label"}),

                # Checkboxes — which lines to show on chart
                # LESSON: input_checkbox_group returns a list of
                # selected values. If nothing selected → empty list.
                ui.output_ui("show_lines_ui"),
                

                ui.tags.hr({"class": "sidebar-divider"}),
                ui.div("Chart Overlays", {"class": "sidebar-label"}),

                ui.input_checkbox("show_regimes", "Regime shading",  value=True),
                ui.input_checkbox("show_switches","Switch lines",     value=True),

                width=230,
                style="background:#0F1E35;border-right:1px solid #1E3050;",
            ),

            # Main chart area
            ui.output_ui("overview_chart"),
        ),
    ),

    # ══════════════════════════════════════════════════════════
    # TAB 3 — RUN MODEL
    # ══════════════════════════════════════════════════════════
    ui.nav_panel("▶ Run Model",
        ui.layout_columns(

            # ── LEFT: settings ────────────────────────────────
            ui.card(
                ui.card_header("Run Settings"),

                ui.output_ui("client_run_summary"),

                ui.tags.div("Model", {"class": "run-section-label"}),
                ui.input_select(
                    "run_model", label=None,
                    choices=list(MODELS.keys()),
                    selected="ML (GMM + XGBoost)",
                ),

                ui.tags.div("Advanced Settings", {"class": "run-section-label"}),
                ui.layout_columns(
                    ui.input_numeric("run_tc",     "TC (bps)",         value=5,    min=0,   max=50),
                    ui.input_numeric("run_refit",  "Refit (months)",   value=12,   min=1,   max=36),
                    ui.input_numeric("run_thresh", "Switch Threshold", value=0.75, min=0.5, max=1.0, step=0.01),
                    col_widths=[4, 4, 4],
                ),

                ui.tags.button(
                    "▶  Run Model",
                    id="run_btn",
                    class_="btn btn-run-model action-button",
                ),
            ),

            # ── RIGHT: status + progress ──────────────────────
            ui.div(
                ui.card(
                    ui.card_header("Status"),
                    ui.output_ui("run_status"),
                    ui.output_ui("run_progress"),
                ),
                ui.card(
                    ui.card_header("Previous Runs — Load from Cache"),
                    ui.output_ui("cache_panel"),
                ),
            ),

            col_widths=[5, 7],
        ),
    ),

    # ══════════════════════════════════════════════════════════
    # ── TAB 5: ANNUAL ANALYTICS ───────────────────────────────
    ui.nav_panel("📅 Annual",
        ui.layout_columns(
            ui.card(ui.card_header("Annual Return (%)"),    ui.output_ui("ann_ret_chart")),
            ui.card(ui.card_header("Annual Sharpe"),        ui.output_ui("ann_sharpe_chart")),
            col_widths=[6,6],
        ),
        ui.layout_columns(
            ui.card(ui.card_header("Annual Sortino"),       ui.output_ui("ann_sortino_chart")),
            ui.card(ui.card_header("Annual Max Drawdown"),  ui.output_ui("ann_dd_chart")),
            col_widths=[6,6],
        ),
        ui.card(ui.card_header("Monthly Return Heatmap — Walk-Forward"),
                ui.output_ui("heatmap_chart")),
        ui.card(ui.card_header("Monthly Active Return — vs Benchmark"),
        ui.output_ui("heatmap_active_chart")),
    ),

    # ── TAB 6: REGIME ANALYSIS ────────────────────────────────
    ui.nav_panel("🎯 Regimes",
        ui.layout_columns(
            ui.card(ui.card_header("Regime Attribution by Year"), ui.output_ui("regime_bar_chart")),
            ui.card(ui.card_header("Drawdown Curves"),            ui.output_ui("drawdown_chart")),
            col_widths=[6,6],
        ),
        ui.card(ui.card_header("Regime Alpha Table"), ui.output_data_frame("regime_alpha_tbl")),
    ),

    # ── TAB 7: RISK METRICS ───────────────────────────────────
    ui.nav_panel("📐 Risk",
        ui.card(ui.card_header("Full Risk Dashboard — Walk-Forward vs Benchmark"),
                ui.output_data_frame("risk_tbl")),
        ui.layout_columns(
            ui.card(ui.card_header("VaR & CVaR (95%)"),    ui.output_ui("var_chart")),
            ui.card(ui.card_header("Return Distribution"), ui.output_ui("dist_chart")),
            col_widths=[6,6],
        ),
    ),

    # ── TAB 8: TURNOVER ───────────────────────────────────────
    ui.nav_panel("🔄 Turnover",
        ui.layout_columns(
            ui.card(ui.card_header("Annual Switches & Turnover"), ui.output_ui("turnover_bar")),
            ui.card(ui.card_header("Cumulative TC"),              ui.output_ui("cum_tc_chart")),
            col_widths=[6,6],
        ),
        ui.card(ui.card_header("Turnover Detail"), ui.output_data_frame("turnover_tbl")),
    ),

    # ── TAB 9: PORTFOLIO WEIGHTS ──────────────────────────────
    ui.nav_panel("📦 Weights",
    
        ui.card(
            ui.card_header("Current Allocation Recommendation"),
            ui.output_ui("allocation_recommendation"),
        ),
    
        ui.card(
            ui.card_header("Current Allocation — Model vs Benchmark"),
            ui.output_data_frame("current_allocation_tbl"),
        ),
    
        ui.card(ui.card_header("Portfolio Weights Over Time"),  ui.output_ui("weights_chart")),
        ui.card(ui.card_header("Average Weight by Regime"),    ui.output_ui("weights_regime_chart")),
    ),
    # ── TAB 10: MODEL COMPARISON ──────────────────────────────
    ui.nav_panel("⚖️ Compare",
        ui.layout_columns(
            ui.card(
                ui.card_header("Select Cached Runs"),
                ui.output_ui("compare_selector"),
            ),
            ui.card(
                ui.card_header("Cumulative Wealth Comparison"),
                ui.output_ui("compare_chart"),
            ),
            col_widths=[4,8],
        ),
        ui.card(ui.card_header("Performance Summary"), ui.output_data_frame("compare_tbl")),
        ui.card(ui.card_header("Regime Performance by Model"), ui.output_data_frame("compare_regime_tbl")),
    ),

    # ── TAB 11: CACHE MANAGEMENT ──────────────────────────────
    ui.nav_panel("💾 Cache",
        ui.layout_columns(
            ui.card(
                ui.card_header("Delete Cache Entry"),
                ui.output_ui("delete_cache_ui"),
            ),
            ui.card(
                ui.card_header("All Cached Runs"),
                ui.output_data_frame("cache_index_tbl"),
            ),
            col_widths=[4,8],
        ),
    ),

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 12 — UI
    # ══════════════════════════════════════════════════════════════════════════════
    
    # ── TAB 13: EXPORT ────────────────────────────────────────
    ui.nav_panel("📄 Export",
        ui.div(
            {"style": "max-width:700px;margin:40px auto;"},
            ui.card(
                ui.card_header("RAAM Client Report"),
                ui.p(
                    "Generate a personalized PDF report including your risk profile, "
                    "the model used, backtest performance summary, and current allocation recommendation.",
                    style="font-size:13px;color:#7A8BA0;margin-bottom:20px;line-height:1.6;"
                ),
                ui.tags.button(
                    "⬇  Generate My RAAM Report",
                    id="pdf_a_btn",
                    class_="da-run-btn action-button",
                    style="width:100%;margin-top:4px;",
                ),
                ui.output_ui("pdf_a_status"),
                ui.output_ui("pdf_a_download"),
            ),
        ),
    ),

    # ── TAB 13: CONFIG ────────────────────────────────────────
    # TAB 4 — CONFIG
    # ══════════════════════════════════════════════════════════
    ui.nav_panel("⚙️ Config",
        ui.layout_columns(
            ui.card(
                ui.card_header("Applied Risk Profile"),
                ui.output_ui("config_profile_summary"),
            ),
            ui.card(
                ui.card_header("Override (Advisor Use)"),
                ui.input_numeric("ov_gamma_bull",  "γ Bull",             value=1.5, min=0.1, max=10, step=0.1),
                ui.input_numeric("ov_gamma_trans", "γ Transition",       value=3.0, min=0.1, max=10, step=0.1),
                ui.input_numeric("ov_gamma_bear",  "γ Bear",             value=4.0, min=0.1, max=10, step=0.1),
                ui.tags.hr(style="border-color:#1E3050;"),
                ui.input_numeric("ov_eq_pct",   "Benchmark Equity %",   value=55, min=0, max=100),
                ui.input_numeric("ov_bond_pct", "Benchmark Bond %",     value=40, min=0, max=100),
                ui.input_numeric("ov_alt_pct",  "Benchmark Alt/Gold %", value=5,  min=0, max=100),
                ui.tags.small("Pre-filled from survey. Change only if needed.",
                              style="color:#7A8BA0;font-size:11px;"),
            ),
            col_widths=[5, 7],
        ),
    ),

    title=ui.tags.div(
        ui.tags.div("RAAM", style="font-size:20px;font-weight:800;letter-spacing:0.1em;"),
        ui.tags.div("Regime-Aware Asset Management", style="font-size:12px;opacity:0.75;"),
        style="color:#C9A84C;line-height:1.05;"
    ),
    bg="#06101F",
    inverse=True,
)


# ══════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════

def server(input, output, session):

    # ── State ─────────────────────────────────────────────────
    survey_started  = reactive.Value(False)
    applied_profile = reactive.Value(None)
    client_name     = reactive.Value("")
    client_age      = reactive.Value(40)
    client_init     = reactive.Value(100_000)   # NEW: initial investment

    # ══════════════════════════════════════════════════════════
    # SURVEY PAGE  (welcome → questions)
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def survey_page():
        return _welcome_screen() if not survey_started.get() else _survey_screen()

    def _welcome_screen():
        return ui.div(
            {"style": "padding:20px;"},
            ui.HTML(
                "<div class='welcome-card'>"
                "<div class='welcome-logo'>⬡</div>"
                "<div class='welcome-title'>Welcome to RAAM </div>"
                "<div class='welcome-sub'>Before we build your personalized portfolio strategy, "
                "we need to understand your goals and comfort with risk.<br><br>"
                "This takes about <b style='color:#C9A84C;'>2 minutes</b>.</div>"
                "</div>"
            ),
            ui.div(
                {"style": "max-width:520px;margin:-20px auto 0;padding:0 44px 48px;"},

                # Name
                ui.div({"class": "welcome-input-wrap"},
                    ui.tags.label("Your first name"),
                    ui.input_text("welcome_name", label=None,
                                  placeholder="e.g. John", value=""),
                ),
                ui.div({"class": "welcome-input-wrap"},
                    ui.tags.label("Your Last name"),
                    ui.input_text("welcome_Lastname", label=None,
                                  placeholder="e.g. John", value=""),
                ),

                # Age
                ui.div({"class": "welcome-input-wrap"},
                    ui.tags.label("Your age"),
                    ui.input_numeric("welcome_age", label=None,
                                     value='', min=18, max=100),
                ),

                # Initial investment — NEW
                ui.div({"class": "welcome-input-wrap"},
                    ui.tags.label("Initial investment amount ($)"),
                    ui.input_numeric("welcome_init", label=None,
                                     value='', min=1_000,
                                     step=10_000),
                    ui.tags.p(
                        "This is used to calculate your projected final portfolio value.",
                        style="font-size:11px;color:#7A8BA0;margin-top:4px;"
                    ),
                ),

                ui.tags.button(
                    "Start My Risk Assessment  →",
                    id="start_btn",
                    class_="btn btn-start action-button",
                ),
            ),
        )

    def _survey_screen():
        name = client_name.get()
        hi   = f"Hi {name}," if name else ""
        return ui.div(
            {"style": "padding:36px 24px;max-width:1180px;margin:0 auto;"},
            ui.output_ui("survey_progress"),
            ui.layout_columns(
                ui.div(
                    survey_q(1,8,"q1", f"{hi} When do you plan to start using this money?",
                        {"1":"Within 3 years — I may need it soon","2":"In 3 to 7 years",
                         "3":"In 7 to 15 years","4":"15 years or more — long-term wealth"}),
                    survey_q(2,8,"q2","Your portfolio drops 25% in 3 months. What do you do?",
                        {"1":"Sell everything — I can't watch this","2":"Sell some to reduce exposure",
                         "3":"Hold and wait — it will recover","4":"Buy more — buying opportunity"}),
                    survey_q(3,8,"q3","In a really bad year, what's the most you could lose and stay invested?",
                        {"1":"Nothing — I need my capital protected","2":"Up to 10%",
                         "3":"Up to 25% — uncomfortable but I'd hold","4":"More than 25% — long-term mindset"}),
                    survey_q(4,8,"q4","Which investment feels right for you? (All over 5 years)",
                        {"1":"3% per year guaranteed","2":"Likely 7%, could lose 5% in a bad year",
                         "3":"Likely 12%, could lose 15% in a bad year","4":"Likely 20%, could lose 30% in a bad year"}),
                    survey_q(5,8,"q5","Which would upset you more?",
                        {"1":"Losing $10K is far worse than missing a $10K gain",
                         "2":"Both feel about the same",
                         "3":"Missing a big gain bothers me more than a loss"}),
                    survey_q(6,8,"q6","How stable is your income outside this investment?",
                        {"1":"Very unstable — this may be all I have","2":"Somewhat stable",
                         "3":"Stable — steady job or income","4":"Very stable — multiple sources"}),
                    survey_q(7,8,"q7","How familiar are you with investing?",
                        {"1":"Beginner","2":"Some experience",
                         "3":"Experienced — been through cycles","4":"Very experienced"}),
                    survey_q(8,8,"q8","How confident are you in the answers you just gave?",
                        {"1":"Not sure","2":"Somewhat confident",
                         "3":"Confident","4":"Very confident"}),
                ),
                ui.div(
                    ui.output_ui("profile_result"),
                    style="position:sticky;top:90px;"
                ),
                col_widths=[7, 5],
            ),
        )

    # ── Start button ──────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.start_btn)
    def _start():
        name = (input.welcome_name() or "").strip()
        age  = input.welcome_age() or 40
        init = input.welcome_init() or 100_000
        if age < 18 or age > 100: age = 40
        if init < 1_000: init = 100_000
        client_name.set(name)
        client_age.set(int(age))
        client_init.set(int(init))
        survey_started.set(True)

    # ── Survey score ──────────────────────────────────────────
    @reactive.calc
    def survey_score():
        if not survey_started.get():
            return {"answered": 0, "result": None}
        ans = {"q1":input.q1(),"q2":input.q2(),"q3":input.q3(),"q4":input.q4(),
               "q5":input.q5(),"q6":input.q6(),"q7":input.q7(),"q8":input.q8()}
        answered = sum(1 for v in ans.values() if v not in (None,""))
        if answered < 8:
            return {"answered": answered, "result": None}
        ia  = {k:int(v) for k,v in ans.items()}
        res = score_survey(ia["q1"],ia["q2"],ia["q3"],ia["q4"],
                           ia["q5"],ia["q6"],ia["q7"],ia["q8"],
                           age=client_age.get())
        res["answered"] = 8
        return {"answered": 8, "result": res}

    # ── Progress bar ──────────────────────────────────────────
    @output
    @render.ui
    def survey_progress():
        if not survey_started.get(): return ui.div()
        sc   = survey_score(); n = sc["answered"]
        pct  = int(n/8*100)
        col  = "#2ECC71" if n==8 else "#C9A84C"
        name = client_name.get()
        lbl  = f"Hi {name} — " if name else ""
        return ui.HTML(
            f"<div class='progress-wrap'>"
            f"<div class='progress-label'><span>{lbl}Survey Progress</span>"
            f"<span style='color:{col};font-weight:600;'>{n} of 8 answered</span></div>"
            f"<div class='progress-track'>"
            f"<div class='progress-fill' style='width:{pct}%;"
            f"background:linear-gradient(90deg,#8C1515,{col});'></div></div></div>"
        )

    # ── Profile result panel ──────────────────────────────────
    @output
    @render.ui
    def profile_result():
        if not survey_started.get(): return ui.div()
        sc   = survey_score(); n = sc["answered"]
        name = client_name.get(); age = client_age.get()
        hi   = name if name else "you"

        if n == 0:
            return ui.HTML(
                "<div style='background:#0F1E35;border:1px solid #1E3050;border-radius:10px;"
                "padding:32px;text-align:center;margin-top:20px;'>"
                "<div style='font-size:36px;margin-bottom:12px;'>🎯</div>"
                "<div style='font-size:15px;font-weight:600;color:#E8EDF5;margin-bottom:8px;'>"
                "Your Risk Profile</div>"
                "<div style='font-size:13px;color:#7A8BA0;line-height:1.6;'>"
                "Answer the questions to see your personalized profile.</div></div>"
            )
        if n < 8:
            left = 8-n
            return ui.HTML(
                "<div style='background:#0F1E35;border:1px solid #1E3050;border-radius:10px;"
                "padding:32px;text-align:center;margin-top:20px;'>"
                f"<div style='font-size:24px;letter-spacing:4px;color:#C9A84C;margin-bottom:12px;'>"
                f"{'●'*n}<span style='color:#1E3050;'>{'○'*(8-n)}</span></div>"
                f"<div style='font-size:14px;color:#7A8BA0;'>"
                f"{left} more question{'s' if left>1 else ''} to go…</div></div>"
            )

        res  = sc["result"]; meta = profile_meta(res["profile"])
        c    = meta["color"]; bg = meta["bg"]

        cap_note = (
            f"<div class='age-cap-note'>⚠️  Based on your age ({age}), your profile was "
            f"adjusted from <b>{res['raw_profile']}</b> to <b>{res['profile']}</b>.</div>"
        ) if res["age_capped"] else ""

        bm = res["benchmark"]
        bm_html = (
            "<div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:14px;'>"
            f"<span style='background:rgba(79,195,247,0.12);color:#4FC3F7;padding:5px 12px;border-radius:16px;font-size:12px;font-weight:600;'>Equity {int(bm['eq']*100)}%</span>"
            f"<span style='background:rgba(201,168,76,0.12);color:#C9A84C;padding:5px 12px;border-radius:16px;font-size:12px;font-weight:600;'>Bonds {int(bm['bond']*100)}%</span>"
            f"<span style='background:rgba(46,204,113,0.12);color:#2ECC71;padding:5px 12px;border-radius:16px;font-size:12px;font-weight:600;'>Gold {int(bm['alt']*100)}%</span>"
            "</div>"
        )
        gamma_html = (
            "<div class='gamma-row'>"
            f"<div class='gamma-box'><div class='g-label'>Bull γ</div><div class='g-value' style='color:#2ECC71;'>{res['gamma_bull']}</div></div>"
            f"<div class='gamma-box'><div class='g-label'>Trans γ</div><div class='g-value' style='color:#F39C12;'>{res['gamma_trans']}</div></div>"
            f"<div class='gamma-box'><div class='g-label'>Bear γ</div><div class='g-value' style='color:#E74C3C;'>{res['gamma_bear']}</div></div>"
            "</div>"
        )

        applied     = applied_profile.get()
        applied_html= (
            "<div class='applied-banner'>✓  Profile applied — Config tab updated</div>"
        ) if (applied and applied["profile"]==res["profile"]) else ""

        return ui.div(
            ui.HTML(cap_note),
            ui.HTML(
                f"<div class='profile-card' style='background:{bg};border-color:{c};'>"
                f"<div style='font-size:32px;margin-bottom:8px;'>{meta['icon']}</div>"
                f"<div class='profile-name' style='color:{c};'>{hi.title()}, you are<br>{res['profile']}</div>"
                f"<div class='profile-desc' style='color:#A8C4E0;'>{meta['desc']}</div>"
                f"<div style='font-size:11px;color:#7A8BA0;margin-bottom:14px;'>Benchmark: {meta['bm_label']}</div>"
                f"{bm_html}</div>"
            ),
            ui.div(
                ui.tags.p("Risk Parameters",
                    style="font-size:11px;font-weight:700;color:#7A8BA0;"
                          "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;"),
                ui.HTML(gamma_html),
                style="background:#0F1E35;border:1px solid #1E3050;border-radius:10px;padding:14px;margin-bottom:12px;",
            ),
            ui.tags.button("✓  Apply This Profile to Model",
                id="apply_btn", class_="btn btn-apply action-button"),
            ui.HTML(applied_html),
        )

    # ── Apply button ──────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.apply_btn)
    def _apply():
        sc = survey_score()
        if sc["result"] is None: return
        res = sc["result"]
        applied_profile.set(res)
        ui.update_numeric("ov_gamma_bull",  value=res["gamma_bull"])
        ui.update_numeric("ov_gamma_trans", value=res["gamma_trans"])
        ui.update_numeric("ov_gamma_bear",  value=res["gamma_bear"])
        ui.update_numeric("ov_eq_pct",   value=int(res["benchmark"]["eq"]   * 100))
        ui.update_numeric("ov_bond_pct", value=int(res["benchmark"]["bond"] * 100))
        ui.update_numeric("ov_alt_pct",  value=int(res["benchmark"]["alt"]  * 100))

    # ══════════════════════════════════════════════════════════
    # DEMO DATA  — reactive so it updates when profile changes
    # ══════════════════════════════════════════════════════════

    @reactive.calc
    def demo_data():
        """
        Returns (bt_wf, bt_bm, bt_mv, init) — synthetic demo.
        Used as fallback when no real model has been run.
        """
        ap      = applied_profile.get()
        profile = ap["profile"] if ap else "Moderate"
        init    = client_init.get()
        bt_wf, bt_bm, bt_mv = gen_demo(init=init, profile=profile)
        try:
            s = pd.Timestamp(str(input.date_range()[0]))
            e = pd.Timestamp(str(input.date_range()[1]))
        except Exception:
            s = pd.Timestamp("2019-01-02")
            e = pd.Timestamp("2025-12-31")
        def trim(bt):
            sub = bt[(bt.index >= s) & (bt.index <= e)].copy()
            if len(sub) > 0:
                sub["cum"] = (1 + sub["ret"]).cumprod()
            return sub
        return trim(bt_wf), trim(bt_bm), trim(bt_mv), init

    @reactive.calc
    def current_results():
        """
        Master data source for all chart tabs.
        Returns (bt_wf, bt_bm, bt_mv, init, is_real).
        - If a real model has been run → use real results
        - Otherwise → fall back to demo data
        Date filter is applied to both paths.
        """
        rr   = run_results.get()
        init = client_init.get()

        # Apply date range filter
        try:
            s = pd.Timestamp(str(input.date_range()[0]))
            e = pd.Timestamp(str(input.date_range()[1]))
        except Exception:
            s = pd.Timestamp("2019-01-02")
            e = pd.Timestamp("2025-12-31")

        def trim(bt):
            sub = bt[(bt.index >= s) & (bt.index <= e)].copy()
            if len(sub) > 0:
                sub["cum"] = (1 + sub["ret"]).cumprod()
            return sub

        if rr is not None:
            # Real model results
            used_init = rr.get("init", init)
            return (trim(rr["bt_wf"]), trim(rr["bt_bm"]),
                    trim(rr["bt_mv"]), used_init, True)

        # Demo fallback
        bt_wf, bt_bm, bt_mv, init = demo_data()
        return bt_wf, bt_bm, bt_mv, init, False

    # ══════════════════════════════════════════════════════════
    # KPI CARDS
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def show_lines_ui():
        rr = run_results.get()
    
        if rr is not None:
            model = rr.get("model", "Model")
            wf_label = f"Walk-Forward ({model})"
        else:
            wf_label = "Walk-Forward (Demo)"
    
        return ui.input_checkbox_group(
            "show_lines", label=None,
            choices={
                "wf": wf_label,
                "bm": "Benchmark",
                "mv": "MV No-Regime",
            },
            selected=["wf", "bm", "mv"],
        )
    @output
    @render.ui
    def kpi_cards():
        bt_wf, bt_bm, _, init, is_real = current_results()
        if len(bt_wf) == 0:
            return ui.div()

        ANN  = 252
        r    = bt_wf["ret"]
        cum  = bt_wf["cum"]

        # Metrics
        final_val = init * cum.iloc[-1]
        ar        = ((1+r).prod()**(ANN/len(r)) - 1) * 100
        vol       = r.std() * np.sqrt(ANN)
        sh        = round((r.mean()*ANN) / (vol) if vol > 0 else 0, 2)
        mdd       = ((cum - cum.cummax()) / cum.cummax()).min() * 100
        sw        = int(bt_wf["rebalanced"].sum())

        # Profile name for display
        ap      = applied_profile.get()
        profile = ap["profile"] if ap else "Moderate"
        name    = client_name.get()

        def kpi(label, value, color, sub=""):
            sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
            return (
                f"<div class='kpi-card' style='--accent:{color};'>"
                f"<div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value' style='color:{color};'>{value}</div>"
                f"{sub_html}</div>"
            )

        name_str  = f"{name} · " if name else ""
        if is_real:
            rr       = run_results.get()
            model_lbl= rr.get("model","Model") if rr else "Model"
            badge    = (f"<span style='background:rgba(46,204,113,0.12);"
                        f"border:1px solid rgba(46,204,113,0.3);color:#2ECC71;"
                        f"font-size:10px;font-weight:700;letter-spacing:0.08em;"
                        f"text-transform:uppercase;padding:3px 10px;border-radius:20px;"
                        f"margin-left:6px;'>LIVE — {model_lbl}</span>")
        else:
            badge    = "<span class='demo-badge'>DEMO DATA</span>"

        demo_note = (f"<div style='font-size:11px;color:#7A8BA0;margin-bottom:12px;'>"
                     f"{name_str}{profile} Profile {badge}</div>")

        return ui.HTML(
            demo_note +
            "<div class='kpi-grid'>"
            + kpi("Portfolio Value",  f"${final_val:,.0f}", "#C9A84C", f"from ${init:,.0f}")
            + kpi("Ann. Return",      f"{ar:.1f}%",          "#2ECC71")
            + kpi("Sharpe Ratio",     f"{sh}",               "#4FC3F7")
            + kpi("Max Drawdown",     f"{mdd:.1f}%",         "#E74C3C")
            + kpi("Switches",         f"{sw}",               "#9B59B6", "regime changes")
            + "</div>"
        )

    # ══════════════════════════════════════════════════════════
    # OVERVIEW CHART
    # LESSON: make_subplots creates a figure with multiple panels.
    # row_heights controls how tall each panel is (fractions).
    # shared_xaxes=True links the x-axis zoom across panels.
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def overview_chart():
        # Force reactivity — read run_results and date_range to ensure chart rerenders
        _ = run_results.get()
        _ = input.date_range()
        _ = input.show_lines()
        _ = input.show_regimes()
        _ = input.show_switches()
        
        bt_wf, bt_bm, bt_mv, init, is_real = current_results()
        if len(bt_wf) == 0:
            return ui.HTML("<div style='color:#7A8BA0;padding:40px;'>No data.</div>")

        show    = list(input.show_lines())
        do_reg  = input.show_regimes()
        do_sw   = input.show_switches()

        # ── Two-panel chart: wealth (top) + regime bar (bottom) ─
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.80, 0.20],
            vertical_spacing=0.03,
        )

        # Line colors and styles
        rr   = run_results.get()

        lines = {
            "wf": (bt_wf, f"{rr.get('model','Model') if rr else 'Model'} Walk-Forward", "#C9A84C", "solid", 2.2),
            "bm": (bt_bm, "Benchmark",     "#4FC3F7", "dash",  1.8),
            "mv": (bt_mv, "MV No-Regime",  "#1ABC9C", "dot",   1.8),
        }

        for key, (bt, name, color, dash, lw) in lines.items():
            if key in show:
                fig.add_trace(
                    go.Scatter(
                        x    = bt.index,
                        y    = bt["cum"] * init,
                        name = name,
                        line = dict(color=color, width=lw, dash=dash),
                        hovertemplate = "%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
                    ),
                    row=1, col=1,
                )

        # Regime shading — colored bands behind the chart
        if do_reg and "wf" in show:
            reg_colors = {"Bull":"#2ECC71", "Transition":"#F39C12", "Bear":"#E74C3C"}
            y_hi = bt_wf["cum"].max() * init * 1.03
            y_lo = bt_wf["cum"].min() * init * 0.97
            for regime, color in reg_colors.items():
                mask = bt_wf["regime"] == regime
            
                # find continuous segments
                for k, g in itertools.groupby(range(len(mask)), key=lambda i: mask.iloc[i]):
                    if not k:
                        continue
            
                    idx = list(g)
                    x_segment = bt_wf.index[idx]
            
                    fig.add_vrect(
                        x0=x_segment[0],
                        x1=x_segment[-1],
                        fillcolor=color,
                        opacity=0.15,
                        line_width=0,
                        layer="below"
                    )

        # Switch lines — vertical grey lines at rebalance dates
        if do_sw and "wf" in show:
            for dt in bt_wf[bt_wf["rebalanced"]].index:
                fig.add_vline(
                    x           = dt,
                    line_width  = 0.6,
                    line_color  = "#3A5070",
                    opacity     = 0.7,
                    row         = "all",
                )

        # Regime timeline bar (bottom panel)
        # LESSON: go.Bar with barmode="overlay" stacks colored segments
        reg_cols = {"Bull":"#2ECC71","Transition":"#F39C12","Bear":"#E74C3C"}
        for regime, color in reg_cols.items():
            mask = bt_wf["regime"] == regime
            if mask.any():
                fig.add_trace(
                    go.Bar(
                        x          = bt_wf.index[mask],
                        y          = np.ones(mask.sum()),
                        name       = regime,
                        marker_color = color,
                        opacity    = 0.85,
                        showlegend = False,
                        hoverinfo  = "skip",
                    ),
                    row=2, col=1,
                )

        # Apply dark theme + axis labels
        apply_dark(fig, height=580)
        fig.update_layout(
            barmode = "overlay",
            legend  = dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#E8EDF5", size=11)),
            yaxis   = dict(title="Portfolio Value ($)", tickformat="$,.0f",
                           gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0")),
            yaxis2  = dict(showticklabels=False, showgrid=False),
            xaxis2  = dict(title="Date", tickfont=dict(color="#7A8BA0")),
        )

        return to_html(fig)

    # ── Config summary ────────────────────────────────────────
    @output
    @render.ui
    def config_profile_summary():
        applied = applied_profile.get()
        name    = client_name.get()
        if applied is None:
            return ui.HTML(
                "<div style='color:#7A8BA0;padding:20px;text-align:center;font-size:13px;'>"
                "Complete the Risk Profile survey first,<br>"
                "then click <b style='color:#C9A84C;'>Apply This Profile to Model</b>."
                "</div>"
            )
        meta = profile_meta(applied["profile"])
        c    = meta["color"]
        bm   = applied["benchmark"]
        who  = f"{name}'s Profile" if name else "Applied Profile"
        return ui.HTML(
            f"<div style='padding:8px;'>"
            f"<div style='font-size:20px;font-weight:700;color:{c};margin-bottom:4px;'>"
            f"{meta['icon']}  {who} — {applied['profile']}</div>"
            f"<div style='font-size:12px;color:#7A8BA0;margin-bottom:14px;'>"
            f"Benchmark: {int(bm['eq']*100)}% Eq / {int(bm['bond']*100)}% Bond / {int(bm['alt']*100)}% Alt</div>"
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;'>"
            f"<div style='background:#0A1628;border:1px solid #1E3050;border-radius:6px;padding:10px;text-align:center;'>"
            f"<div style='font-size:10px;color:#7A8BA0;'>Bull γ</div>"
            f"<div style='font-size:18px;font-weight:700;color:#2ECC71;'>{applied['gamma_bull']}</div></div>"
            f"<div style='background:#0A1628;border:1px solid #1E3050;border-radius:6px;padding:10px;text-align:center;'>"
            f"<div style='font-size:10px;color:#7A8BA0;'>Trans γ</div>"
            f"<div style='font-size:18px;font-weight:700;color:#F39C12;'>{applied['gamma_trans']}</div></div>"
            f"<div style='background:#0A1628;border:1px solid #1E3050;border-radius:6px;padding:10px;text-align:center;'>"
            f"<div style='font-size:10px;color:#7A8BA0;'>Bear γ</div>"
            f"<div style='font-size:18px;font-weight:700;color:#E74C3C;'>{applied['gamma_bear']}</div></div>"
            f"</div></div>"
        )


    # ══════════════════════════════════════════════════════════
    # RUN MODEL SERVER LOGIC
    # ══════════════════════════════════════════════════════════

    # State for run tab
    is_running   = reactive.Value(False)
    run_status_v = reactive.Value("ready")   # ready/running/done/cache/error
    run_error_v  = reactive.Value("")
    run_results  = reactive.Value(None)      # stores bt_wf, bt_bm, bt_mv

    # LESSON: queue.Queue is thread-safe.
    # The worker thread puts progress messages in pq.
    # The server polls pq every 0.5s and appends to progress_log.
    pq           = queue.Queue()
    progress_log = reactive.Value([])

    # ── Client summary on run tab ─────────────────────────────
    @output
    @render.ui
    def client_run_summary():
        name    = client_name.get()
        init    = client_init.get()
        ap      = applied_profile.get()
        profile = ap["profile"] if ap else "Not set yet"
        color   = profile_meta(ap["profile"])["color"] if ap else "#7A8BA0"
        html_status = '<span style=\"color:#2ECC71;\">✓ Survey applied</span>' if ap else '<span style=\"color:#E74C3C;\">⚠ Complete survey first</span>'
        return ui.HTML(
            f"<div style='background:#0A1628;border:1px solid #1E3050;border-radius:8px;"
            f"padding:12px 16px;margin-bottom:16px;'>"
            f"<div style='font-size:13px;color:#E8EDF5;font-weight:600;margin-bottom:6px;'>"
            f"{name if name else 'Client'}</div>"
            f"<div style='display:flex;gap:16px;font-size:12px;color:#7A8BA0;'>"
            f"<span>💰 ${init:,.0f}</span>"
            f"<span style='color:{color};'>🎯 {profile}</span>"
            f"{html_status}"
            f"</div></div>"
        )

    # ── Run button ────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        # Don't allow double-run
        if is_running.get():
            return

        # Check data files first — Option A: clear error
        files_ok, missing = check_data_files()
        if not files_ok:
            run_status_v.set("error")
            msg = "MISSING_FILES:" + "|".join(f"{l}::{p}" for l,p in missing)
            run_error_v.set(msg)
            return

        # Get config snapshot
        model_key = MODELS.get(input.run_model(), "ML")
        ap        = applied_profile.get()
        profile   = ap["profile"] if ap else "Moderate"
        gamma     = ap if ap else {"gamma_bull":1.5,"gamma_trans":3.0,"gamma_bear":4.0}
        tc        = input.run_tc()
        refit     = input.run_refit()
        thresh    = input.run_thresh()
        init      = client_init.get()

        # Check cache first
        key = _make_key(model_key, profile, tc, refit, thresh)
        if cache_exists(key):
            try:
                data = load_cache(key)
                run_results.set(data)
                run_status_v.set("cache")
                progress_log.set([f"⚡  Cache hit — loaded {key}"])
                return
            except Exception as e:
                run_error_v.set(str(e))

        # Clear previous log and start
        is_running.set(True)
        run_status_v.set("running")
        run_error_v.set("")
        progress_log.set(["Starting model…"])

        def _progress(fold, total, step):
            pq.put(f"fold_{fold}:{total}:{step}")

        def _worker():
            t0 = time.time()
            try:
                # Add project to path
                project = os.path.join(os.path.dirname(__file__), "..")
                for p in [project, os.path.join(project,"src"),
                          os.path.join(project,"config")]:
                    if p not in sys.path:
                        sys.path.insert(0, p)

                import importlib
                from data.data_loader import load_all
                from backtest.backtest_engine import (
                    backtest_wf, backtest_static, backtest_mv, trim_to_test,
                    sharpe, ann_return, max_drawdown,
                )
                from portfolio.portfolio_allocator import build_mv_benchmark
                from config import TICKER_MAP, FIRST_TRAIN_END

                pq.put("status:Loading data…")
                data = load_all()
                
                if ap:
                    eq_pct   = ap["benchmark"]["eq"]
                    bond_pct = ap["benchmark"]["bond"]
                    alt_pct  = ap["benchmark"]["alt"]
                else:
                    eq_pct, bond_pct, alt_pct = 0.55, 0.40, 0.05
                
                bm = pd.Series({
                    "SPY": eq_pct * 0.45,
                    "IVE": eq_pct * 0.20,
                    "IVW": eq_pct * 0.20,
                    "IWM": eq_pct * 0.15,
                
                    "SHY": bond_pct * 0.15,
                    "IEF": bond_pct * 0.20,
                    "TLT": bond_pct * 0.25,
                    "LQD": bond_pct * 0.25,
                    "HYG": bond_pct * 0.15,
                
                    "GLD": alt_pct,
                })
                

                # Import the right model
                mod_map = {
                    "ML":   "models.ml_regime_model",
                    "HMM":  "models.hmm_model",
                    "HSMM": "models.semi_markov_model",
                    "Jump": "models.jump_model",
                }
                pq.put(f"status:Running {model_key} walk-forward…")
                mod = importlib.import_module(mod_map[model_key])

                custom_gamma = {
                    "Bull":       gamma.get("gamma_bull", 1.5),
                    "Transition": gamma.get("gamma_trans", 3.0),
                    "Bear":       gamma.get("gamma_bear",  4.0),
                }

                regime_probs, fold_ports = mod.run_walk_forward(
                    data["X_all"], data["r_all"], bm,
                    risk_profile  = profile,
                    custom_gamma  = custom_gamma,
                    refit_months  = refit,
                    progress_callback = _progress,
                )

                # Build confirmed signal
                if hasattr(mod, "build_asymmetric_regime_signal"):
                    confirmed = mod.build_asymmetric_regime_signal(
                        regime_probs,
                        switch_threshold = thresh,
                    )
                else:
                    col_map   = {"prob_Bull":"Bull","prob_Transition":"Transition","prob_Bear":"Bear"}
                    confirmed = regime_probs.idxmax(axis=1).map(col_map)

                pq.put("status:Running backtest…")
                r_all_t = data["r_all"].rename(columns=TICKER_MAP).reindex(columns=bm.index)
                bt_wf, wt = backtest_wf(r_all_t, confirmed, fold_ports, tc_bps=tc)
                mv_w      = build_mv_benchmark(data["r_all"], bm, FIRST_TRAIN_END, refit)
                bt_bm     = backtest_static(r_all_t, bm)
                bt_mv     = backtest_mv(r_all_t, mv_w, tc_bps=tc)

                result = {
                    "bt_wf":  trim_to_test(bt_wf),
                    "bt_bm":  trim_to_test(bt_bm),
                    "bt_mv":  trim_to_test(bt_mv),
                    "benchmark_weights": bm,
                    "weights": wt,
                    "model":   model_key,
                    "profile": profile,
                    "init":    init,
                }

                # Compute summary stats for cache index
                r  = result["bt_wf"]["ret"]
                sh = round(float(sharpe(r)), 3)
                ar = f"{ann_return(r)*100:.1f}%"
                sw = int(result["bt_wf"]["rebalanced"].sum())

                save_cache(key, result, {
                    "model":   model_key,
                    "profile": profile,
                    "sharpe":  sh,
                    "ann_ret": ar,
                    "switches":sw,
                    "runtime": round(time.time()-t0, 1),
                })

                pq.put(("RESULT", result))
                pq.put(f"status:Done — Sharpe {sh}  Ann. Return {ar}  Switches {sw}")
                pq.put("DONE:ok")

            except Exception as e:
                import traceback
                pq.put(f"status:Error — {str(e)}")
                pq.put(f"ERROR:{str(e)}")
                pq.put("DONE:err")

        threading.Thread(target=_worker, daemon=True).start()

    # ── Poll queue every 0.5s ─────────────────────────────────
    # LESSON: reactive.invalidate_later(n) makes this effect
    # re-run every n seconds automatically. This is how we
    # get real-time progress updates without websocket magic.
    @reactive.effect
    def _poll():
        reactive.invalidate_later(0.5)
        log = progress_log.get().copy()
        changed = False

        while not pq.empty():
            msg = pq.get_nowait()
            changed = True
            
            if isinstance(msg, tuple) and msg[0] == "RESULT":
                run_results.set(msg[1])
                continue

            if msg.startswith("fold_"):
                # fold_1:7:xgboost
                parts  = msg[5:].split(":")
                fold   = parts[0]; total = parts[1]; step = parts[2]
                log.append(f"Fold {fold}/{total} — {step}")

            elif msg.startswith("status:"):
                log.append(msg[7:])

            elif msg.startswith("ERROR:"):
                run_error_v.set(msg[6:])
                run_status_v.set("error")

            elif msg == "DONE:ok":
                is_running.set(False)
                run_status_v.set("done")

            elif msg == "DONE:err":
                is_running.set(False)

        if changed:
            progress_log.set(log[-20:])   # keep last 20 lines

    # ── Status pill ───────────────────────────────────────────
    @output
    @render.ui
    def run_status():
        status = run_status_v.get()
        err    = run_error_v.get()

        # ── Missing files error — detailed explanation ─────────
        if status == "error" and err.startswith("MISSING_FILES:"):
            pairs   = err[14:].split("|")
            missing = [p.split("::") for p in pairs if "::" in p]
            file_rows = "".join([
                f"<div class='error-file'>❌  {label}<br>"
                f"<span style='color:#7A8BA0;font-size:11px;'>Expected at: {path}</span></div>"
                for label, path in missing
            ])
            return ui.HTML(
                "<div class='error-box'>"
                "<div class='error-title'>⚠️  Data Files Not Found</div>"
                "<div style='font-size:13px;color:#A8C4E0;margin-bottom:12px;'>"
                "The following data files are required to run the model. "
                "Please place them in the correct location relative to the "
                "<code style='color:#C9A84C;'>regime_allocation/</code> folder:</div>"
                + file_rows +
                "<div style='font-size:12px;color:#7A8BA0;margin-top:14px;'>"
                "Once files are in place, click Run again.</div>"
                "</div>"
            )

        # ── Other error ────────────────────────────────────────
        if status == "error":
            return ui.HTML(
                f"<span class='status-pill pill-error'>❌  Error</span>"
                f"<div style='font-size:12px;color:#E74C3C;margin-top:8px;'>{err[:200]}</div>"
            )

        pill = {
            "ready":   ("<span class='status-pill pill-ready'>Ready</span>"
                        "<div style='font-size:12px;color:#7A8BA0;margin-top:8px;'>"
                        "Configure settings above and click Run.</div>"),
            "running": "<span class='status-pill pill-running'>⟳  Running…</span>",
            "done":    "<span class='status-pill pill-done'>✓  Complete</span>",
            "cache":   "<span class='status-pill pill-cache'>⚡  Loaded from Cache</span>",
        }.get(status, "")

        return ui.HTML(pill)

    # ── Progress log ──────────────────────────────────────────
    @output
    @render.ui
    def run_progress():
        log  = progress_log.get()
        stat = run_status_v.get()

        if not log and stat == "ready":
            return ui.div()

        # Progress bar
        n_folds = 7
        done    = sum(1 for l in log if l.startswith("Fold") and "/" in l)
        pct     = 100 if stat == "done" else min(int(done/n_folds*90), 90)
        col     = "#2ECC71" if stat == "done" else "#C9A84C"

        bar_html = (
            f"<div style='background:#06101F;border-radius:4px;height:8px;"
            f"border:1px solid #1E3050;overflow:hidden;margin-bottom:10px;'>"
            f"<div style='height:100%;width:{pct}%;border-radius:4px;"
            f"background:linear-gradient(90deg,#8C1515,{col});"
            f"transition:width 0.4s ease;'></div></div>"
        )

        # Log lines
        def line_class(l):
            if "✅" in l or "Done" in l or "Complete" in l: return "ok"
            if "Error" in l or "❌" in l: return "err"
            if "Fold" in l: return "fold"
            return ""

        log_html = "".join([
            f"<div class='log-line {line_class(l)}'>{l}</div>"
            for l in log
        ])

        return ui.HTML(
            bar_html +
            f"<div class='log-box'>{log_html}</div>"
        )

    # ── Cache panel ───────────────────────────────────────────
    @output
    @render.ui
    def cache_panel():
        choices = cache_choices()

        if not choices:
            return ui.HTML(
                "<div style='color:#7A8BA0;font-size:13px;padding:8px;'>"
                "No cached runs yet. Run a model to save results.</div>"
            )

        options_html = "".join([
            f"<option value='{k}'>{v}</option>"
            for k, v in choices.items()
        ])

        # Build cache table
        idx = _load_index()
        rows = "".join([
            f"<tr>"
            f"<td>{e.get('model','')}</td>"
            f"<td>{e.get('profile','')}</td>"
            f"<td style='color:#2ECC71;'>{e.get('sharpe',0):.3f}</td>"
            f"<td>{e.get('ann_ret','')}</td>"
            f"<td>{e.get('switches',0)}</td>"
            f"<td>{str(e.get('date',''))[-8:]}</td>"
            f"</tr>"
            for e in sorted(idx, key=lambda x: str(x.get('date','')), reverse=True)[:8]
        ])

        table_html = (
            "<table style='width:100%;border-collapse:collapse;margin-top:12px;font-size:12px;'>"
            "<thead><tr>"
            "<th>Model</th><th>Profile</th><th>Sharpe</th>"
            "<th>Ann Ret</th><th>Sw</th><th>Time</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
        ) if rows else ""

        return ui.div(
            ui.div(
                {"class": "cache-row"},
                ui.input_select("cache_key", label=None,
                                choices={"": "— select a run —", **choices}),
                ui.tags.button("Load",
                               id="load_cache_btn",
                               class_="btn btn-load-cache action-button"),
            ),
            ui.HTML(table_html),
        )

    # ── Load from cache button ────────────────────────────────
    @reactive.effect
    @reactive.event(input.load_cache_btn)
    def _load_cache():
        key = input.cache_key()
        if not key: # or not cache_exists(key):
            return
        try:
            data = load_cache(key)
            run_results.set(data)
            run_status_v.set("cache")
            progress_log.set([f"⚡  Loaded from cache: {key}"])
        except Exception as e:
            run_error_v.set(str(e))
            run_status_v.set("error")


    # ══════════════════════════════════════════════════════════
    # SHARED METRIC HELPER
    # ══════════════════════════════════════════════════════════

    def _met(r, cum, bm_r=None, ANN=252):
        """Compute 20 metrics from a return series + cum wealth."""
        if len(r) < 5: return {}
        vol  = r.std()*np.sqrt(ANN)
        ar   = (1+r).prod()**(ANN/len(r))-1
        sh   = (r.mean()*ANN)/vol if vol>0 else 0
        dw   = r[r<0]; dv=np.sqrt((dw**2).mean())*np.sqrt(ANN) if len(dw)>0 else np.nan
        so   = (r.mean()*ANN)/dv if (dv and dv>0) else 0
        mdd  = ((cum-cum.cummax())/cum.cummax()).min()
        cal  = ar/abs(mdd) if mdd!=0 else 0
        v95  = float(np.percentile(r,5)); cv95=float(r[r<=v95].mean()) if (r<=v95).any() else v95
        v99  = float(np.percentile(r,1)); cv99=float(r[r<=v99].mean()) if (r<=v99).any() else v99
        try:
            from scipy.stats import skew as _sk, kurtosis as _ku
            sk=float(_sk(r.dropna())); ku=float(_ku(r.dropna()))
        except: sk=ku=0.0
        gains=r[r>0].sum(); losses=abs(r[r<0].sum())
        omega=gains/losses if losses>0 else np.nan
        pain=abs(((cum-cum.cummax())/cum.cummax()).mean())
        wr=((1+r).resample("ME").prod()-1>0).mean()*100
        out=dict(ar=ar,vol=vol,sh=sh,so=so,cal=cal,mdd=mdd,
                 v95=v95,cv95=cv95,v99=v99,cv99=cv99,
                 sk=sk,ku=ku,omega=omega,pain=pain,wr=wr)
        if bm_r is not None:
            bm2=bm_r.reindex(r.index).fillna(0)
            cov=np.cov(r.fillna(0).values,bm2.values)
            beta=cov[0,1]/cov[1,1] if cov[1,1]>0 else 1.0
            alpha=(r.mean()-beta*bm2.mean())*ANN
            act=r-bm2; te=act.std()*np.sqrt(ANN)
            ir=(act.mean()*ANN)/te if te>0 else 0
            out.update(beta=beta,alpha=alpha,te=te,ir=ir)
        return out

    # ── Annual bar helper ─────────────────────────────────────
    def _ann_bar(metric, scale=1, fmt="{:.2f}"):
        bt_wf,bt_bm,bt_mv,init,_ = current_results()
        if len(bt_wf)==0:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No data.</div>")
        ANN=252; years=sorted(bt_wf.index.year.unique())
        x=np.arange(len(years)); w=0.26
        fig=go.Figure()
        for i,(nm,bt,col) in enumerate([
            ("Benchmark",bt_bm,"#4FC3F7"),
            ("MV",bt_mv,"#1ABC9C"),
            ("Walk-Fwd",bt_wf,"#C9A84C"),
        ]):
            vals=[]
            for y in years:
                sub=bt[bt.index.year==y]; r=sub["ret"]; cum=(1+r).cumprod()
                if len(sub)<5: vals.append(0); continue
                if metric=="ret":   vals.append(((1+r).prod()-1)*scale)
                elif metric=="sh":
                    v=r.std()*np.sqrt(ANN)
                    vals.append((r.mean()*ANN/v if v>0 else 0)*scale)
                elif metric=="so":
                    dw=r[r<0]; dv=np.sqrt((dw**2).mean())*np.sqrt(ANN) if len(dw)>0 else np.nan
                    vals.append((r.mean()*ANN/dv if (dv and dv>0) else 0)*scale)
                elif metric=="mdd":
                    vals.append(((cum-cum.cummax())/cum.cummax()).min()*scale)
            fig.add_trace(go.Bar(
                x=list(x+i*w), y=vals, name=nm,
                marker_color=col, width=w*0.9,
                text=[fmt.format(v) for v in vals],
                textposition="outside", textfont=dict(size=9,color="#E8EDF5"),
            ))
        fig.update_layout(
            barmode="group", height=320,
            plot_bgcolor="#0A1628", paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"), margin=dict(l=40,r=20,t=30,b=40),
            xaxis=dict(tickvals=list(x+w),ticktext=[str(y) for y in years],
                       gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)),
        )
        fig.add_hline(y=0,line_color="#3A5070",line_width=1)
        return to_html(fig)

    # ══════════════════════════════════════════════════════════
    # TAB 5 — ANNUAL
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def ann_ret_chart():     return _ann_bar("ret",100,"{:.1f}%")

    @output
    @render.ui
    def ann_sharpe_chart():  return _ann_bar("sh",1,"{:.2f}")

    @output
    @render.ui
    def ann_sortino_chart(): return _ann_bar("so",1,"{:.2f}")

    @output
    @render.ui
    def ann_dd_chart():      return _ann_bar("mdd",100,"{:.1f}%")

    @output
    @render.ui
    def heatmap_chart():
        import calendar
        bt_wf,_,_,_,_ = current_results()
        if len(bt_wf)==0:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No data.</div>")
        r_m  = (1+bt_wf["ret"]).resample("ME").prod()-1
        piv  = r_m.groupby([r_m.index.year,r_m.index.month]).first().unstack()*100
        piv.columns=[calendar.month_abbr[m] for m in piv.columns]
        vals=piv.values; vmax=max(abs(np.nanmin(vals)),abs(np.nanmax(vals)),5)
        text=[[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in vals]
        fig=go.Figure(go.Heatmap(
            z=vals, x=list(piv.columns), y=[str(y) for y in piv.index],
            text=text, texttemplate="%{text}",
            colorscale=[[0,"#E74C3C"],[0.5,"#152742"],[1,"#2ECC71"]],
            zmid=0,zmin=-vmax,zmax=vmax,
             colorbar=dict(
                tickfont=dict(color="#7A8BA0"),
                title=dict(
                    text="%",
                    font=dict(color="#7A8BA0")
                )
            ),
            hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
        ))
        fig.update_layout(height=300,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=60,t=20,b=40),
            xaxis=dict(tickfont=dict(color="#7A8BA0")),
            yaxis=dict(tickfont=dict(color="#7A8BA0")))
        return to_html(fig)
    @output
    @render.ui
    def heatmap_active_chart():
        import calendar
        bt_wf, bt_bm, _, _, _ = current_results()
    
        if len(bt_wf) == 0:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No data.</div>")
    
        # Monthly returns
        r_wf = (1+bt_wf["ret"]).resample("ME").prod() - 1
        r_bm = (1+bt_bm["ret"]).resample("ME").prod() - 1
    
        # Active return
        r_m = (r_wf - r_bm)
    
        piv = r_m.groupby([r_m.index.year, r_m.index.month]).first().unstack() * 100
        piv.columns = [calendar.month_abbr[m] for m in piv.columns]
    
        vals = piv.values
        vmax = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 5)
    
        text = [[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in vals]
    
        fig = go.Figure(go.Heatmap(
            z=vals,
            x=list(piv.columns),
            y=[str(y) for y in piv.index],
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0, "#C0392B"],
                [0.5, "#0F1E35"],
                [1, "#27AE60"]
            ],
            zmid=0,
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(
                tickfont=dict(color="#7A8BA0"),
                title=dict(text="%", font=dict(color="#7A8BA0"))
            ),
            hovertemplate="<b>%{y} %{x}</b><br>Active: %{z:.2f}%<extra></extra>",
        ))
    
        fig.update_layout(
            height=300,
            plot_bgcolor="#0A1628",
            paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),
            margin=dict(l=40, r=60, t=20, b=40),
            xaxis=dict(tickfont=dict(color="#7A8BA0")),
            yaxis=dict(tickfont=dict(color="#7A8BA0"))
        )
    
        return to_html(fig)

    # ══════════════════════════════════════════════════════════
    # TAB 6 — REGIME ANALYSIS
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def regime_bar_chart():
        bt_wf,_,_,_,_ = current_results()
        if len(bt_wf)==0 or "regime" not in bt_wf.columns:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No data.</div>")
        rows=[]
        for yr in sorted(bt_wf.index.year.unique()):
            sub=bt_wf[bt_wf.index.year==yr]; c=sub["regime"].value_counts()
            for reg in ["Bull","Transition","Bear"]:
                rows.append({"Year":str(yr),"Regime":reg,
                             "Pct":round(c.get(reg,0)/len(sub)*100,1)})
        df=pd.DataFrame(rows)
        fig=go.Figure()
        for reg,col in [("Bull","#2ECC71"),("Transition","#F39C12"),("Bear","#E74C3C")]:
            sub=df[df["Regime"]==reg]
            fig.add_trace(go.Bar(x=sub["Year"],y=sub["Pct"],
                                 name=reg,marker_color=col,opacity=0.85))
        fig.update_layout(barmode="stack",height=320,
            plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=20,b=40),
            xaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),title="% of Year"),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)))
        return to_html(fig)

    @output
    @render.ui
    def drawdown_chart():
        bt_wf,bt_bm,bt_mv,_,_ = current_results()
        if len(bt_wf)==0:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No data.</div>")
        fig=go.Figure()
        for nm,bt,col,dash in [("Benchmark",bt_bm,"#4FC3F7","dash"),
                                 ("MV",bt_mv,"#1ABC9C","dot"),
                                 ("Walk-Fwd",bt_wf,"#C9A84C","solid")]:
            dd=(bt["cum"]-bt["cum"].cummax())/bt["cum"].cummax()*100
            fig.add_trace(go.Scatter(x=bt.index,y=dd,name=nm,
                fill="tozeroy",fillcolor=hex_to_rgba(col, 0.08),
                line=dict(color=col,width=1.8,dash=dash),
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra>"+nm+"</extra>"))
        fig.add_hline(y=0,line_color="#3A5070",line_width=1)
        fig.update_layout(height=320,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=20,b=40),
            xaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),title="Drawdown (%)"),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)))
        return to_html(fig)

    @output
    @render.data_frame
    def regime_alpha_tbl():
        bt_wf,bt_bm,_,_,_ = current_results(); ANN=252
        if len(bt_wf)==0 or "regime" not in bt_wf.columns:
            return render.DataGrid(pd.DataFrame({"Info":["Run a model first."]}))
        bm_r=bt_bm["ret"].reindex(bt_wf.index).fillna(0)
        rows=[]
        for reg in ["Bull","Transition","Bear"]:
            mask=bt_wf["regime"]==reg
            if mask.sum()<5: continue
            wfr=bt_wf.loc[mask,"ret"]; bmr=bm_r[mask]
            cum=(1+wfr).cumprod();bm_cum=(1+bmr).cumprod(); n=ANN/len(wfr)
            wfa=(1+wfr).prod()**n-1; bma=(1+bmr).prod()**n-1
            vol=wfr.std()*np.sqrt(ANN)
            rows.append({"Regime":reg,"Days":len(wfr),
                "% Time":f"{len(wfr)/len(bt_wf)*100:.1f}%",
                "Model Ret":f"{wfa*100:.2f}%","Bmark Ret":f"{bma*100:.2f}%",
                "Alpha":f"{(wfa-bma)*100:+.2f}%",
                "Sharpe":f"{wfr.mean()*ANN/vol:.3f}" if vol>0 else "—",
                "BM Sharpe":f"{bmr.mean()*ANN/(bmr.std()*np.sqrt(ANN)):.3f}",
                "Max DD":f"{((cum-cum.cummax())/cum.cummax()).min()*100:.1f}%",
                "BM Max DD":f"{((bm_cum-bm_cum.cummax())/bm_cum.cummax()).min()*100:.1f}%"})
        return render.DataGrid(pd.DataFrame(rows),height="200px")

    # ══════════════════════════════════════════════════════════
    # TAB 7 — RISK METRICS
    # ══════════════════════════════════════════════════════════

    @output
    @render.data_frame
    def risk_tbl():
        bt_wf,bt_bm,_,init,_ = current_results()
        if len(bt_wf)==0:
            return render.DataGrid(pd.DataFrame({"Info":["Run a model first."]}))
        bm_r=bt_bm["ret"].reindex(bt_wf.index).fillna(0)
        rows=[]
        for nm,bt in [("Benchmark",bt_bm),("Walk-Forward",bt_wf)]:
            r=bt["ret"]; cum=bt["cum"]
            m=_met(r,cum,bm_r if nm=="Walk-Forward" else None)
            sw=int(bt.get("rebalanced",pd.Series(dtype=bool)).sum())
            rows.append({
                "Strategy":nm,
                "Ann Return":    f"{m.get('ar',0)*100:.2f}%",
                "Ann Vol":       f"{m.get('vol',0)*100:.2f}%",
                "Sharpe":        f"{m.get('sh',0):.3f}",
                "Sortino":       f"{m.get('so',0):.3f}",
                "Calmar":        f"{m.get('cal',0):.3f}",
                "Max DD":        f"{m.get('mdd',0)*100:.1f}%",
                "VaR 95%":       f"{m.get('v95',0)*100:.3f}%",
                "CVaR 95%":      f"{m.get('cv95',0)*100:.3f}%",
                "VaR 99%":       f"{m.get('v99',0)*100:.3f}%",
                "CVaR 99%":      f"{m.get('cv99',0)*100:.3f}%",
                "Skewness":      f"{m.get('sk',0):.3f}",
                "Ex Kurtosis":   f"{m.get('ku',0):.3f}",
                "Omega":         f"{m.get('omega',0):.3f}" if not np.isnan(m.get('omega',np.nan)) else "—",
                "Pain Index":    f"{m.get('pain',0)*100:.3f}%",
                "Win Rate":      f"{m.get('wr',0):.1f}%",
                "Beta":          f"{m.get('beta',1):.3f}" if nm=="Walk-Forward" else "—",
                "Alpha":         f"{m.get('alpha',0)*100:.2f}%" if nm=="Walk-Forward" else "—",
                "Track Error":   f"{m.get('te',0)*100:.2f}%" if nm=="Walk-Forward" else "—",
                "Info Ratio":    f"{m.get('ir',0):.3f}" if nm=="Walk-Forward" else "—",
                "Switches":      str(sw),
                "Final Value":   f"${init*cum.iloc[-1]:,.0f}",
            })
        df=pd.DataFrame(rows).T.reset_index()
        df.columns=["Metric","Benchmark","Walk-Forward"]
        return render.DataGrid(df,height="520px")

    @output
    @render.ui
    def var_chart():
        bt_wf,bt_bm,_,_,_ = current_results()
        if len(bt_wf)==0:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No data.</div>")
        labels=["Benchmark","Walk-Forward"]
        v95=[float(np.percentile(bt_bm["ret"],5))*100,
             float(np.percentile(bt_wf["ret"],5))*100]
        cv95=[float(bt_bm["ret"][bt_bm["ret"]<=bt_bm["ret"].quantile(.05)].mean())*100,
              float(bt_wf["ret"][bt_wf["ret"]<=bt_wf["ret"].quantile(.05)].mean())*100]
        x=[0,1]
        fig=go.Figure()
        for offset,name,vals,col in [(0,"VaR 95%",v95,"#F39C12"),
                                      (0.38,"CVaR 95%",cv95,"#E74C3C")]:
            fig.add_trace(go.Bar(
                x=[v+offset for v in x],y=vals,name=name,
                marker_color=col,width=0.35,
                text=[f"{v:.3f}%" for v in vals],
                textposition="outside",textfont=dict(color="#E8EDF5",size=10)))
        fig.update_layout(barmode="group",height=320,
            plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=20,b=40),
            xaxis=dict(tickvals=x,ticktext=labels,
                       gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),
                       title="Daily Loss (%)"),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)))
        return to_html(fig)

    @output
    @render.ui
    def dist_chart():
        bt_wf,bt_bm,_,_,_ = current_results()
        if len(bt_wf)==0:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No data.</div>")
        fig=go.Figure()
        for nm,bt,col in [("Benchmark",bt_bm,"#4FC3F7"),("Walk-Fwd",bt_wf,"#C9A84C")]:
            fig.add_trace(go.Histogram(
                x=bt["ret"]*100,name=nm,opacity=0.65,nbinsx=80,
                marker_color=col,
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra>"+nm+"</extra>"))
        fig.update_layout(barmode="overlay",height=320,
            plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=20,b=40),
            xaxis=dict(title="Daily Return (%)",gridcolor="#1A2E48",
                       tickfont=dict(color="#7A8BA0")),
            yaxis=dict(title="Count",gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)))
        return to_html(fig)

    # ══════════════════════════════════════════════════════════
    # TAB 8 — TURNOVER
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def turnover_bar():
        bt_wf,_,_,_,_ = current_results()
        if len(bt_wf)==0 or "turnover" not in bt_wf.columns:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Turnover data available after running the real model.</div>")
        ato=bt_wf.groupby(bt_wf.index.year)["turnover"].sum()
        asw=bt_wf.groupby(bt_wf.index.year)["rebalanced"].sum().astype(int)
        x=list(range(len(ato)))
        fig=go.Figure(go.Bar(
            x=x,y=list(ato.values),marker_color="#C9A84C",opacity=0.85,
            text=[f"{int(sw)}sw" for sw in asw.values],
            textposition="outside",textfont=dict(color="#E8EDF5",size=10)))
        fig.update_layout(height=320,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=20,b=40),
            xaxis=dict(tickvals=x,ticktext=[str(y) for y in ato.index],
                       gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),
                       title="Annual Turnover"))
        return to_html(fig)

    @output
    @render.ui
    def cum_tc_chart():
        bt_wf,_,_,_,_ = current_results()
        if len(bt_wf)==0 or "turnover" not in bt_wf.columns:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Turnover data available after running the real model.</div>")
        TC_BPS=5
        cum_to=bt_wf["turnover"].cumsum()
        total_tc=cum_to.iloc[-1]*TC_BPS
        fig=go.Figure(go.Scatter(
            x=bt_wf.index,y=cum_to,fill="tozeroy",
            line=dict(color="#C9A84C",width=2),
            fillcolor="rgba(201,168,76,0.12)",
            hovertemplate="%{x|%Y-%m-%d}<br>Cum TO: %{y:.2f}<extra></extra>"))
        fig.add_annotation(
            x=bt_wf.index[int(len(bt_wf)*0.55)],y=cum_to.max()*0.65,
            text=f"Total TC ≈ {total_tc:.0f} bps ({total_tc/100:.2f}%)",
            showarrow=False,bgcolor="#0F1E35",bordercolor="#C9A84C",
            font=dict(color="#C9A84C",size=12))
        fig.update_layout(height=320,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=20,b=40),
            xaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),
                       title="Cumulative Turnover"))
        return to_html(fig)

    @output
    @render.data_frame
    def turnover_tbl():
        bt_wf,_,_,_,_ = current_results()
        if len(bt_wf)==0 or "turnover" not in bt_wf.columns:
            return render.DataGrid(pd.DataFrame({"Info":["Available after running real model."]}))
        TC_BPS=5
        ato=bt_wf.groupby(bt_wf.index.year)["turnover"].sum()
        asw=bt_wf.groupby(bt_wf.index.year)["rebalanced"].sum().astype(int)
        avg=bt_wf.groupby(bt_wf.index.year).apply(
            lambda g: g.loc[g["rebalanced"],"turnover"].mean()
            if g["rebalanced"].any() else 0)
        df=pd.DataFrame({"Year":ato.index,"Switches":asw.values,
            "Turnover":ato.round(3).values,"Avg/Switch":avg.round(3).values,
            "TC (bps)":(ato*TC_BPS).round(1).values})
        total=pd.DataFrame([{"Year":"TOTAL","Switches":int(asw.sum()),
            "Turnover":round(ato.sum(),3),
            "Avg/Switch":round(bt_wf.loc[bt_wf["rebalanced"],"turnover"].mean(),3)
                         if bt_wf["rebalanced"].any() else 0,
            "TC (bps)":round(ato.sum()*TC_BPS,1)}])
        return render.DataGrid(pd.concat([df,total],ignore_index=True),height="300px")

    # ══════════════════════════════════════════════════════════
    # TAB 9 — PORTFOLIO WEIGHTS
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def allocation_recommendation():
        rr = run_results.get()
    
        if rr is None or rr.get("weights") is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Available after running a real model.</div>")
    
        wt = rr["weights"]
        bt_wf = rr["bt_wf"]
    
        if not isinstance(wt, pd.DataFrame) or wt.empty:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No weight data.</div>")
    
        asof = wt.index[-1]
        latest_regime = bt_wf.loc[bt_wf.index <= asof, "regime"].iloc[-1] if "regime" in bt_wf.columns else "Unknown"
    
        if latest_regime == "Bull":
            stance = "Risk-on allocation"
            reason = "The model is in a favorable regime, so the recommendation can lean toward growth assets."
            color = "#2ECC71"
        elif latest_regime == "Transition":
            stance = "Balanced allocation"
            reason = "The model is detecting a mixed regime, so the recommendation stays closer to benchmark."
            color = "#F39C12"
        elif latest_regime == "Bear":
            stance = "Defensive allocation"
            reason = "The model is in a risk-off regime, so the recommendation should reduce equity-sensitive exposure."
            color = "#E74C3C"
        else:
            stance = "Benchmark-aware allocation"
            reason = "No clear regime label is available, so review the model weights relative to benchmark."
            color = "#7A8BA0"
    
        return ui.HTML(
            f"""
            <div style='padding:18px 22px;'>
                <div style='font-size:11px;color:#7A8BA0;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;'>
                    Recommendation as of {asof.strftime('%Y-%m-%d')}
                </div>
                <div style='font-size:24px;font-weight:700;color:{color};margin-bottom:6px;'>
                    {stance}
                </div>
                <div style='font-size:14px;color:#E8EDF5;margin-bottom:8px;'>
                    Current regime: <b>{latest_regime}</b>
                </div>
                <div style='font-size:13px;color:#A8C4E0;line-height:1.6;'>
                    {reason}
                </div>
            </div>
            """
        )
    @output
    @render.data_frame
    def current_allocation_tbl():
        rr = run_results.get()
    
        if rr is None or rr.get("weights") is None:
            return render.DataGrid(
                pd.DataFrame({"Message": ["Available after running a real model."]}),
                height="120px"
            )
    
        wt = rr["weights"]
    
        if not isinstance(wt, pd.DataFrame) or wt.empty:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No weight data."]}),
                height="120px"
            )
    
        model_w = wt.iloc[-1].copy()
    
        bm_ser = rr.get("benchmark_weights")
        bm_w = bm_ser.copy()
    
        df = pd.DataFrame({
            "ETF": model_w.index.astype(str),
            "Model Weight": (model_w.values * 100).round(2),
            "Benchmark Weight": (bm_w.values * 100).round(2),
            "Active Weight": ((model_w.values - bm_w.values) * 100).round(2),
        })
    
        df["Model Weight"] = df["Model Weight"].map(lambda x: f"{x:.2f}%")
        df["Benchmark Weight"] = df["Benchmark Weight"].map(lambda x: f"{x:.2f}%")
        df["Active Weight"] = df["Active Weight"].map(lambda x: f"{x:+.2f}%")
    
        return render.DataGrid(df, height="320px")
    @output
    @render.ui
    def weights_chart():
        rr=run_results.get()
        if rr is None or rr.get("weights") is None:
            return ui.HTML(
                "<div style='background:#0F1E35;border:1px solid #1E3050;border-radius:10px;"
                "padding:32px;text-align:center;color:#7A8BA0;font-size:13px;'>"
                "Portfolio weights are available after running the real model.<br>"
                "<span style='font-size:11px;margin-top:8px;display:block;'>"
                "Go to ▶ Run Model and run a model first.</span></div>")
        wt=rr["weights"]
        if not isinstance(wt,pd.DataFrame) or wt.empty:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No weight data.</div>")
        bt_wf=rr["bt_wf"]; common=wt.index.intersection(bt_wf.index)
        wt=wt.loc[common]
        pal=["#4FC3F7","#C9A84C","#2ECC71","#9B59B6","#E74C3C",
             "#1ABC9C","#F39C12","#3498DB","#E67E22","#95A5A6"]
        fig=go.Figure()
        for i,col in enumerate(wt.columns):
            fig.add_trace(go.Scatter(
                x=wt.index,y=wt[col],name=str(col),
                stackgroup="one",fill="tonexty",
                line=dict(width=0.5),marker_color=pal[i%len(pal)],
                hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>"))
        fig.update_layout(height=420,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=50,r=20,t=20,b=40),
            xaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),
                       title="Weight",tickformat=".0%"),
            legend=dict(orientation="h",y=1.06,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)))
        return to_html(fig)

    @output
    @render.ui
    def weights_regime_chart():
        rr=run_results.get()
        if rr is None or rr.get("weights") is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;text-align:center;'>Available after running a model.</div>")
        wt=rr["weights"]; bt_wf=rr["bt_wf"]
        if not isinstance(wt,pd.DataFrame) or wt.empty:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No weight data.</div>")
        common=wt.index.intersection(bt_wf.index)
        wt=wt.loc[common]; bt=bt_wf.loc[common]
        rows=[{"Regime":reg,**wt[bt["regime"]==reg].mean().round(3).to_dict()}
              for reg in ["Bull","Transition","Bear"] if (bt["regime"]==reg).sum()>0]
        if not rows:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>No regime data.</div>")
        df=pd.DataFrame(rows).set_index("Regime")
        pal=["#4FC3F7","#C9A84C","#2ECC71","#9B59B6","#E74C3C",
             "#1ABC9C","#F39C12","#3498DB","#E67E22","#95A5A6"]
        fig=go.Figure([go.Bar(x=list(df.index),y=list(df[c]),
            name=str(c),marker_color=pal[i%len(pal)])
            for i,c in enumerate(df.columns)])
        fig.update_layout(barmode="stack",height=320,
            plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=50,r=20,t=20,b=40),
            xaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),
                       title="Avg Weight",tickformat=".0%"),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)))
        return to_html(fig)

    # ══════════════════════════════════════════════════════════
    # TAB 10 — MODEL COMPARISON
    # ══════════════════════════════════════════════════════════

    compare_keys = reactive.Value([])

    @output
    @render.ui
    def compare_selector():
        idx = _load_index()
        if not idx:
            return ui.HTML("<div style='color:#7A8BA0;font-size:13px;padding:8px;'>"
                           "No cached runs yet. Run models first.</div>")
        # Build choices dict for checkbox group — key: display label
        choices = {}
        for e in sorted(idx, key=lambda x: str(x.get("date","")), reverse=True)[:15]:
            k = e.get("key","")
            if not k: continue
            label = (f"{e.get('model','')} | {e.get('profile','')} | "
                     f"Sharpe {e.get('sharpe',0):.3f} | {str(e.get('date',''))[-8:]}")
            choices[k] = label

        if not choices:
            return ui.HTML("<div style='color:#7A8BA0;padding:8px;'>No valid runs.</div>")

        return ui.div(
            # LESSON: input_checkbox_group inside render.ui works perfectly.
            # The id must be unique and stable — "compare_check" is fixed.
            # selected= syncs with compare_keys reactive value.
            ui.input_checkbox_group(
                "compare_check",
                label=None,
                choices=choices,
                selected=compare_keys.get(),
            ),
            ui.tags.button("Clear All", id="compare_clear_btn",
                class_="btn-sm-red action-button",
                style="margin-top:8px;width:100%;"),
        )

    # Sync compare_keys whenever checkbox group changes
    @reactive.effect
    def _sync_compare():
        try:
            vals = list(input.compare_check() or [])
            if vals != compare_keys.get():
                compare_keys.set(vals)
        except Exception:
            pass

    @reactive.effect
    @reactive.event(input.compare_clear_btn)
    def _clear_compare():
        compare_keys.set([])
        ui.update_checkbox_group("compare_check", selected=[])

    @output
    @render.ui
    def compare_chart():
        keys=compare_keys.get()
        if not keys:
            return ui.HTML(
                "<div style='background:#0F1E35;border:1px solid #1E3050;border-radius:10px;"
                "padding:32px;text-align:center;color:#7A8BA0;font-size:13px;'>"
                "Select runs from the left panel to compare.</div>")
        init=client_init.get()
        pal=["#C9A84C","#4FC3F7","#2ECC71","#9B59B6","#E74C3C","#F39C12","#1ABC9C"]
        fig=go.Figure(); bm_added=False
        for i,k in enumerate(keys):
            if not cache_exists(k): continue
            try:
                data=load_cache(k)
                e=next((x for x in _load_index() if x.get("key")==k),{})
                label=f"{e.get('model',k)} — {e.get('profile','')}"
                bt=data.get("bt_wf",data.get("bt_wf_test"))
                if bt is None or len(bt)==0: continue
                if "cum" not in bt.columns:
                    bt["cum"]=(1+bt["ret"].fillna(0)).cumprod()
                fig.add_trace(go.Scatter(
                    x=bt.index,y=bt["cum"]*init,name=label,
                    line=dict(color=pal[i%len(pal)],width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>"+label+"</extra>"))
                if not bm_added:
                    bt_bm=data.get("bt_bm",data.get("bt_bmark_test"))
                    if bt_bm is not None and len(bt_bm)>0:
                        if "cum" not in bt_bm.columns:
                            bt_bm["cum"]=(1+bt_bm["ret"].fillna(0)).cumprod()
                        fig.add_trace(go.Scatter(
                            x=bt_bm.index,y=bt_bm["cum"]*init,name="Benchmark",
                            line=dict(color="#7A8BA0",width=1.5,dash="dash"),
                            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Benchmark</extra>"))
                        bm_added=True
            except: continue
        fig.update_layout(height=400,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=50,r=20,t=20,b=40),
            xaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0")),
            yaxis=dict(gridcolor="#1A2E48",tickfont=dict(color="#7A8BA0"),
                       title="Portfolio Value ($)",tickformat="$,.0f"),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=11)))
        return to_html(fig)

    @output
    @render.data_frame
    def compare_tbl():
        keys=compare_keys.get()
        if not keys:
            return render.DataGrid(pd.DataFrame({"Info":["Select runs to compare."]}))
        init=client_init.get(); rows=[]
        for k in keys:
            if not cache_exists(k): continue
            try:
                data=load_cache(k)
                e=next((x for x in _load_index() if x.get("key")==k),{})
                bt=data.get("bt_wf",data.get("bt_wf_test"))
                bt_bm=data.get("bt_bm",data.get("bt_bmark_test"))
                if bt is None or len(bt)==0: continue
                if "cum" not in bt.columns:
                    bt["cum"]=(1+bt["ret"].fillna(0)).cumprod()
                r=bt["ret"]; cum=bt["cum"]
                bm_r=bt_bm["ret"].reindex(r.index).fillna(0) if bt_bm is not None else None
                m=_met(r,cum,bm_r)
                sw=int(bt.get("rebalanced",pd.Series(dtype=bool)).sum())
                rows.append({
                    "Model":e.get("model",""),"Profile":e.get("profile",""),
                    "Final $":f"${init*cum.iloc[-1]:,.0f}",
                    "Ann Return":f"{m.get('ar',0)*100:.1f}%",
                    "Sharpe":f"{m.get('sh',0):.3f}",
                    "Sortino":f"{m.get('so',0):.3f}",
                    "Max DD":f"{m.get('mdd',0)*100:.1f}%",
                    "CVaR 95%":f"{m.get('cv95',0)*100:.3f}%",
                    "Alpha":f"{m.get('alpha',0)*100:.2f}%" if m.get('alpha') else "—",
                    "Info Ratio":f"{m.get('ir',0):.3f}" if m.get('ir') else "—",
                    "Switches":str(sw),
                    "Date":str(e.get("date",""))[-10:],
                })
            except: continue
        return render.DataGrid(pd.DataFrame(rows),height="260px")

    @output
    @render.data_frame
    def compare_regime_tbl():
        """Per-regime breakdown for each selected model."""
        keys = compare_keys.get()
        if not keys:
            return render.DataGrid(pd.DataFrame({"Info":["Select runs to compare."]}))
        ANN = 252; rows = []
        for k in keys:
            if not cache_exists(k): continue
            try:
                data = load_cache(k)
                e    = next((x for x in _load_index() if x.get("key")==k), {})
                bt   = data.get("bt_wf", data.get("bt_wf_test"))
                bt_bm= data.get("bt_bm", data.get("bt_bmark_test"))
                if bt is None or len(bt)==0 or "regime" not in bt.columns: continue
                if "cum" not in bt.columns:
                    bt["cum"] = (1+bt["ret"].fillna(0)).cumprod()
                bm_r = bt_bm["ret"].reindex(bt.index).fillna(0) if bt_bm is not None else pd.Series(0,index=bt.index)
                model_lbl = f"{e.get('model','')} | {e.get('profile','')}"
                for reg in ["Bull","Transition","Bear"]:
                    mask = bt["regime"] == reg
                    if mask.sum() < 5: continue
                    r   = bt.loc[mask,"ret"]
                    bm  = bm_r[mask]
                    cum = (1+r).cumprod()
                    n   = ANN/len(r)
                    ar  = (1+r).prod()**n - 1
                    bma = (1+bm).prod()**n - 1
                    vol = r.std()*np.sqrt(ANN)
                    sh  = r.mean()*ANN/vol if vol>0 else 0
                    mdd = ((cum-cum.cummax())/cum.cummax()).min()
                    rows.append({
                        "Model":   model_lbl,
                        "Regime":  reg,
                        "Days":    int(mask.sum()),
                        "% Time":  f"{mask.mean()*100:.1f}%",
                        "Ann Ret": f"{ar*100:.2f}%",
                        "BM Ret":  f"{bma*100:.2f}%",
                        "Alpha":   f"{(ar-bma)*100:+.2f}%",
                        "Sharpe":  f"{sh:.3f}",
                        "Max DD":  f"{mdd*100:.1f}%",
                    })
            except: continue
        if not rows:
            return render.DataGrid(pd.DataFrame({"Info":["No regime data in selected runs."]}))
        return render.DataGrid(pd.DataFrame(rows), height="340px")



    @output
    @render.ui
    def delete_cache_ui():
        ch=cache_choices()
        if not ch:
            return ui.HTML("<div style='color:#7A8BA0;font-size:13px;padding:8px;'>"
                           "No cached runs yet.</div>")
        return ui.div(
            ui.input_select("del_cache_key","Select run to delete",
                            choices={"":"— select —",**ch}),
            ui.output_ui("delete_confirm_ui"),
            ui.tags.button("🗑  Delete Selected Run",
                id="delete_cache_btn",
                class_="btn-sm-red action-button",
                style="margin-top:10px;width:100%;padding:8px;"),
            ui.output_ui("delete_status_ui"),
        )

    delete_status = reactive.Value("")

    @output
    @render.ui
    def delete_confirm_ui():
        k=input.del_cache_key() if hasattr(input,"del_cache_key") else ""
        if not k: return ui.div()
        e=next((x for x in _load_index() if x.get("key")==k),{})
        if not e: return ui.div()
        return ui.HTML(
            f"<div style='background:rgba(231,76,60,0.08);border:1px solid rgba(231,76,60,0.3);"
            f"border-radius:6px;padding:10px 12px;margin-top:10px;font-size:12px;'>"
            f"<b style='color:#E74C3C;'>Will delete:</b><br>"
            f"<span style='color:#A8C4E0;'>{e.get('model','')} — {e.get('profile','')} — "
            f"Sharpe {e.get('sharpe',0):.3f} — {str(e.get('date',''))[-10:]}</span>"
            f"</div>"
        )

    @output
    @render.ui
    def delete_status_ui():
        s=delete_status.get()
        if not s: return ui.div()
        col="#2ECC71" if "Deleted" in s else "#E74C3C"
        return ui.HTML(
            f"<div style='margin-top:8px;font-size:12px;color:{col};'>{s}</div>")

    @reactive.effect
    @reactive.event(input.delete_cache_btn)
    def _delete_cache():
        k=input.del_cache_key()
        if not k:
            delete_status.set("⚠ Please select a run to delete.")
            return
        try:
            path=os.path.join(CACHE_DIR,f"{k}.pkl")
            if os.path.exists(path): os.remove(path)
            idx=_load_index()
            idx=[e for e in idx if e.get("key")!=k]
            _save_index(idx)
            delete_status.set(f"✓  Deleted {k}")
        except Exception as ex:
            delete_status.set(f"❌  Error: {ex}")

    @output
    @render.data_frame
    def cache_index_tbl():
        idx=_load_index()
        if not idx:
            return render.DataGrid(pd.DataFrame({"Status":["No cached runs yet."]}))
        df=pd.DataFrame(idx)
        keep=[c for c in ["model","profile","sharpe","ann_ret","switches",
                           "runtime","date","size_mb","key"] if c in df.columns]
        out=df[keep].rename(columns={"model":"Model","profile":"Profile",
            "sharpe":"Sharpe","ann_ret":"Ann Ret","switches":"Sw",
            "runtime":"Runtime(s)","date":"Date","size_mb":"MB","key":"Key"})
        if "Date" in out.columns:
            out=out.sort_values("Date",ascending=False)
        return render.DataGrid(out,height="400px",filters=True)


    # ══════════════════════════════════════════════════════════
    # TAB 12 — DATA ANALYSIS
    # ══════════════════════════════════════════════════════════

    da_results = reactive.Value(None)
    da_running = reactive.Value(False)
    da_status  = reactive.Value("Ready — configure settings above and click Run.")
    da_pq      = queue.Queue()


    @output
    @render.ui
    def da_status_ui():
        s       = da_status.get()
        running = da_running.get()
        col     = "#C9A84C" if running else ("#2ECC71" if "✓" in s else "#7A8BA0")
        return ui.HTML(f"<div class='da-status' style='color:{col};'>{s}</div>")


    @reactive.effect
    @reactive.event(input.da_run_btn)
    def _da_run():
        if da_running.get():
            return
        da_running.set(True)
        da_status.set("Loading features…")
        da_results.set(None)

        try:
            s = pd.Timestamp(str(input.da_dates()[0]))
            e = pd.Timestamp(str(input.da_dates()[1]))
        except Exception:
            s = pd.Timestamp("2001-01-01")
            e = pd.Timestamp("2025-12-31")

        try:
            split_date = pd.Timestamp(str(input.da_split_date()))
        except Exception:
            split_date = pd.Timestamp("2014-01-01")

        if not (s < split_date < e):
            da_status.set("❌  Split date must be strictly between start and end dates.")
            da_running.set(False)
            return

        k     = int(input.da_k()   or 5)
        n_pca = int(input.da_pca() or 14)

        def _worker():
            try:
                project = os.path.join(os.path.dirname(__file__), "..")
                for p in [project,
                          os.path.join(project, "src"),
                          os.path.join(project, "config"),
                          os.path.join(project, "src", "features")]:
                    if p not in sys.path:
                        sys.path.insert(0, p)

                from config import (
                    N_PCA, N_REGIMES, GMM_COV_TYPE, GMM_REG, GMM_N_INIT, GMM_SEED,
                    PCA_SEED, REGIME_MERGE_MAP,
                    VIX_BULL_MAX, VIX_BEAR_MIN, DD_BEAR_THRESH, SMOOTH_WINDOW,
                )
                from data.data_loader import load_features
                from features.feature_engineering import compute_transition_matrix

                from sklearn.preprocessing import StandardScaler, MinMaxScaler
                from sklearn.decomposition import PCA
                from sklearn.mixture import GaussianMixture
                from sklearn.metrics import (
                    silhouette_score, classification_report,
                    confusion_matrix, accuracy_score,
                )

                # ── 1. Load & filter ──────────────────────────────────────────
                da_pq.put("Loading feature data…")
                X_all = load_features()
                X_sub = X_all[(X_all.index >= s) & (X_all.index <= e)].dropna()
                if len(X_sub) < 100:
                    da_pq.put("ERROR:Not enough data in selected range.")
                    return

                # ── 2. Standardise ────────────────────────────────────────────
                da_pq.put(f"Standardising features on {len(X_sub)} days…")
                scaler   = StandardScaler()
                X_scaled = scaler.fit_transform(X_sub)

                pca_full = PCA(random_state=PCA_SEED)
                pca_full.fit(X_scaled)
                ev_full = pca_full.explained_variance_ratio_
                cum_ev  = np.cumsum(ev_full)

                # ── 3. PCA ────────────────────────────────────────────────────
                da_pq.put(f"Running PCA ({n_pca} components)…")
                actual_n_pca = min(n_pca, X_scaled.shape[1])
                pca          = PCA(n_components=actual_n_pca, random_state=PCA_SEED)
                X_pca        = pca.fit_transform(X_scaled)
                ev           = pca.explained_variance_ratio_

                # ── 4. k-selection ────────────────────────────────────────────
                da_pq.put("Evaluating k = 3, 4, 5 (Silhouette / AIC / BIC)…")
                k_vals                             = [3, 4, 5]
                sil_scores, aic_scores, bic_scores = [], [], []
                for ki in k_vals:
                    gm = GaussianMixture(
                        n_components=ki, covariance_type=GMM_COV_TYPE,
                        reg_covar=GMM_REG, n_init=10, random_state=GMM_SEED)
                    gm.fit(X_pca)
                    lbl_ki = gm.predict(X_pca)
                    sil_scores.append(float(silhouette_score(
                        X_pca, lbl_ki,
                        sample_size=min(3000, len(X_pca)), random_state=42)))
                    aic_scores.append(float(gm.aic(X_pca)))
                    bic_scores.append(float(gm.bic(X_pca)))

                # ── 5. Final GMM ──────────────────────────────────────────────
                da_pq.put(f"Fitting GMM (k={k}, cov={GMM_COV_TYPE}, reg={GMM_REG})…")
                gmm = GaussianMixture(
                    n_components=k, covariance_type=GMM_COV_TYPE,
                    reg_covar=GMM_REG, n_init=GMM_N_INIT, random_state=GMM_SEED)
                gmm.fit(X_pca)
                labels = gmm.predict(X_pca)
                probs  = gmm.predict_proba(X_pca)

                transition_matrix = compute_transition_matrix(labels, k)

                # ── 6. XGBoost on raw X features ─────────────────────────────
                da_pq.put(f"Training XGBoost (split {split_date.date()})…")
                try:
                    import xgboost as xgb
                    from sklearn.utils.class_weight import compute_class_weight

                    y_series   = pd.Series(labels, index=X_sub.index, name="regime")
                    common_idx = X_sub.index.intersection(y_series.index)
                    X_aligned  = X_sub.loc[common_idx]
                    y_aligned  = y_series.loc[common_idx]

                    X_train = X_aligned[X_aligned.index < split_date]
                    X_test  = X_aligned[X_aligned.index >= split_date]
                    y_train = y_aligned[y_aligned.index < split_date]
                    y_test  = y_aligned[y_aligned.index >= split_date]

                    if len(X_train) < 50 or len(X_test) < 20:
                        raise ValueError(
                            f"Split {split_date.date()} leaves too few rows "
                            f"(train={len(X_train)}, test={len(X_test)}).")

                    classes           = np.unique(y_train)
                    class_weights     = compute_class_weight(
                        class_weight="balanced", classes=classes, y=y_train)
                    class_weight_dict = dict(zip(classes, class_weights))
                    sample_weights    = y_train.map(class_weight_dict)

                    xgb_model = xgb.XGBClassifier(
                        objective="multi:softprob",
                        num_class=k,
                        n_estimators=2000,
                        max_depth=4,
                        learning_rate=0.05,
                        min_child_weight=5,
                        gamma=0.2,
                        subsample=0.7,
                        colsample_bytree=0.7,
                        reg_alpha=0.5,
                        reg_lambda=2.0,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=42,
                    )
                    xgb_model.fit(X_train, y_train,
                                  sample_weight=sample_weights, verbose=False)

                    xgb_preds = xgb_model.predict(X_test).astype(int)
                    xgb_index = X_test.index
                    xgb_ok    = True

                    acc_5 = accuracy_score(y_test, xgb_preds)
                    class_report_5 = pd.DataFrame(
                        classification_report(y_test, xgb_preds,
                                              output_dict=True, zero_division=0)
                    ).T.round(3)
                    cm_5 = confusion_matrix(y_test, xgb_preds)

                    map_dict     = {0: 0, 1: 0, 3: 1, 2: 2, 4: 2}
                    target_names = ["Downside Risk", "Late Cycle", "Normal/Growth"]

                    y_test_merged = np.array([map_dict[v] for v in y_test])
                    y_pred_merged = np.array([map_dict[v] for v in xgb_preds])

                    acc_3 = accuracy_score(y_test_merged, y_pred_merged)
                    class_report_3 = pd.DataFrame(
                        classification_report(y_test_merged, y_pred_merged,
                                              target_names=target_names,
                                              output_dict=True, zero_division=0)
                    ).T.round(3)
                    labels_3_order = target_names
                    cm_3           = confusion_matrix(y_test_merged, y_pred_merged)

                    model_gains = pd.DataFrame({
                        "Metric":         ["Accuracy", "Signal Quality", "Cost Efficiency",
                                           "Train days", "Test days", "Split date"],
                        "5-Class GMM":    [f"{acc_5:.1%}", "Fine-grained", "Higher Turnover",
                                           str(len(X_train)), str(len(X_test)),
                                           str(split_date.date())],
                        "Merged 3-Class": [f"{acc_3:.1%}", "Actionable", "Lower Turnover",
                                           str(len(X_train)), str(len(X_test)),
                                           str(split_date.date())],
                        "Change":         [f"{(acc_3-acc_5):+.1%}", "Improved Reliability",
                                           "Lower Tx Costs", "—", "—", "—"],
                    })

                except Exception as ex_xgb:
                    # ── clean fallback — NO dangling variables ────────────────
                    test_idx       = X_sub.index[X_sub.index >= split_date]
                    xgb_preds      = pd.Series(labels, index=X_sub.index).reindex(test_idx).values
                    xgb_index      = test_idx
                    xgb_ok         = False
                    acc_5          = float("nan")
                    acc_3          = float("nan")
                    class_report_5 = pd.DataFrame({"Info": [f"XGBoost skipped: {ex_xgb}"]})
                    class_report_3 = pd.DataFrame({"Info": [f"XGBoost skipped: {ex_xgb}"]})
                    cm_5           = np.zeros((k, k))
                    cm_3           = np.zeros((3, 3))
                    labels_3_order = ["Downside Risk", "Late Cycle", "Normal/Growth"]
                    model_gains    = pd.DataFrame({"Info": [f"XGBoost skipped: {ex_xgb}"]})
                    da_pq.put(f"XGBoost skipped: {ex_xgb}")

                # ── 7. Cluster characterisation ───────────────────────────────
                da_pq.put("Computing cluster characterisation…")

                # A. Rule-based regime name per day (build_rule_labels from config thresholds)
                vix_col = "Volatility_Risk_Sentiment_features__VIX"
                dd_col  = "equity_market_dynamics_features__SPX_drawdown"

                if vix_col in X_sub.columns and dd_col in X_sub.columns:
                    vix_s = X_sub[vix_col].rolling(SMOOTH_WINDOW, min_periods=1).mean()
                    dd_s  = X_sub[dd_col].rolling(SMOOTH_WINDOW,  min_periods=1).mean()

                    def _classify(row):
                        if row["dd"] < DD_BEAR_THRESH or row["vix"] >= VIX_BEAR_MIN:
                            return "Bear"
                        elif row["vix"] >= VIX_BULL_MAX:
                            return "Transition"
                        return "Bull"

                    rule_labels = (
                        pd.DataFrame({"vix": vix_s, "dd": dd_s})
                        .apply(_classify, axis=1)
                    )
                else:
                    rule_labels = pd.Series("Unknown", index=X_sub.index)

                # B. Dominant rule name per GMM cluster
                cluster_rule_name = {}
                for c in range(k):
                    mask = labels == c
                    if mask.sum() > 0:
                        dominant = rule_labels.loc[X_sub.index[mask]].value_counts().idxmax()
                    else:
                        dominant = "Unknown"
                    cluster_rule_name[c] = dominant

                # C. Regime profile — exact notebook: .groupby(regime).mean().T.round(4)
                REGIME_SOURCE_COLS = [
                    "equity_market_dynamics_features__SPX_return",
                    "Volatility_Risk_Sentiment_features__VIX",
                    "credit_features__HY_IG_Spread",
                    "Macro_Inflation_Environment_features__CPI_change_3m_lag1",
                    "Interest_Rate_Environment_features__Slope_10_2",
                    "CrossAsset_RiskSignals_features__Equity_bond_corr",
                ]
                available_src = [c for c in REGIME_SOURCE_COLS if c in X_sub.columns]

                regime_profile_T = (
                    X_sub[available_src]
                    .assign(regime=labels)
                    .groupby("regime")
                    .mean()
                    .T
                    .round(4)
                )
                regime_profile_T.index.name   = "Feature"
                regime_profile_T.columns.name = "Regime"

                SHORT_NAME_MAP = {
                    "equity_market_dynamics_features__SPX_return":              "SPX Return",
                    "Volatility_Risk_Sentiment_features__VIX":                  "VIX",
                    "credit_features__HY_IG_Spread":                            "HY/IG Spread",
                    "Macro_Inflation_Environment_features__CPI_change_3m_lag1": "CPI Chg 3m",
                    "Interest_Rate_Environment_features__Slope_10_2":           "Slope 10-2",
                    "CrossAsset_RiskSignals_features__Equity_bond_corr":        "EQ-Bond Corr",
                }
                regime_profile_T.index = [
                    SHORT_NAME_MAP.get(i, i) for i in regime_profile_T.index
                ]

                regime_cols = sorted(regime_profile_T.columns)

                rule_row = pd.DataFrame(
                    [[cluster_rule_name.get(c, "—") for c in regime_cols]],
                    index=["Rule Regime"], columns=regime_cols)

                map_row = pd.DataFrame(
                    [[REGIME_MERGE_MAP.get(c, "—") for c in regime_cols]],
                    index=["Mapped Regime"], columns=regime_cols)

                days_row = pd.DataFrame(
                    [[int((labels == c).sum()) for c in regime_cols]],
                    index=["Days"], columns=regime_cols)

                pct_row = pd.DataFrame(
                    [[f"{(labels == c).mean()*100:.1f}%" for c in regime_cols]],
                    index=["% Time"], columns=regime_cols)

                avg_prob_row = pd.DataFrame(
                    [[f"{probs[labels == c, c].mean():.3f}" for c in regime_cols]],
                    index=["Avg Prob"], columns=regime_cols)

                regime_profile_display = pd.concat([
                    rule_row,
                    map_row,
                    days_row,
                    pct_row,
                    avg_prob_row,
                    regime_profile_T,
                ]).reset_index()

                regime_profile_display.columns = (
                    ["Feature"] + [f"Regime {c}" for c in regime_cols]
                )

                # D. Supporting arrays
                feat_names  = list(X_sub.columns)
                X_sub_arr   = X_sub.values
                clust_means = np.array([X_sub_arr[labels == c].mean(0) for c in range(k)])
                mn          = MinMaxScaler((-1, 1))
                clust_norm  = mn.fit_transform(clust_means.T).T
                top_feats   = X_sub.std().nlargest(20).index.tolist()
                corr        = X_sub[top_feats].corr().values
                corr_labels = top_feats

                # ── 8. Pack results ───────────────────────────────────────────
                result_dict = {
                    # PCA
                    "ev":             ev,
                    "ev_full":        ev_full,
                    "cum_ev":         cum_ev,
                    "actual_n_pca":   actual_n_pca,
                    "X_pca":          X_pca,
                    # GMM
                    "labels":         labels,
                    "probs":          probs,
                    "index":          X_sub.index,
                    "k":              k,
                    "n_pca":          n_pca,
                    "date_range":     f"{s.date()} to {e.date()}",
                    # k-selection
                    "k_vals":         k_vals,
                    "sil_scores":     sil_scores,
                    "aic_scores":     aic_scores,
                    "bic_scores":     bic_scores,
                    # transition
                    "transition_matrix": transition_matrix,
                    # XGBoost
                    "xgb_preds":      xgb_preds,
                    "xgb_index":      xgb_index,
                    "xgb_ok":         xgb_ok,
                    "split_date":     split_date,
                    # metrics
                    "class_report_5": class_report_5,
                    "class_report_3": class_report_3,
                    "cm_5":           cm_5,
                    "cm_3":           cm_3,
                    "labels_3_order": labels_3_order,
                    "acc_5":          acc_5,
                    "acc_3":          acc_3,
                    "model_gains":    model_gains,
                    # cluster characterisation
                    "regime_profile_display": regime_profile_display,
                    "cluster_rule_name":      cluster_rule_name,
                    "clust_norm":             clust_norm,
                    "feat_names":             feat_names,
                    "corr":                   corr,
                    "corr_labels":            corr_labels,
                }

                da_pq.put(("RESULT", result_dict))
                da_pq.put(
                    f"DONE:✓  GMM complete — k={k}, {len(X_sub)} days, "
                    f"PCA explains {ev.sum()*100:.1f}%  |  split: {split_date.date()}"
                )

            except Exception as ex:
                da_pq.put(f"ERROR:{ex}")
                da_pq.put("DONE:Error — check that data files are available.")

        threading.Thread(target=_worker, daemon=True).start()


    @reactive.effect
    def _da_poll():
        if da_running.get() or not da_pq.empty():
            reactive.invalidate_later(0.5)
        while not da_pq.empty():
            msg = da_pq.get_nowait()
            if isinstance(msg, tuple) and msg[0] == "RESULT":
                da_results.set(msg[1]); continue
            if msg.startswith("ERROR:"):
                da_status.set(f"❌  {msg[6:]}"); da_running.set(False)
            elif msg.startswith("DONE:"):
                da_status.set(msg[5:]); da_running.set(False)
            else:
                da_status.set(f"⟳  {msg}")


    # ══════════════════════════════════════════════════════════
    # RENDER FUNCTIONS
    # ══════════════════════════════════════════════════════════

    @output
    @render.ui
    def da_feature_matrix():
        groups = {
            "Equity Market":          ["SPX return","20d volatility","drawdown","momentum","MA signal"],
            "Volatility & Sentiment": ["VIX level","VIX change","VIX 3m–spot term structure"],
            "Credit":                 ["HY spread","IG spread","HY/IG changes","HY–IG differential"],
            "Rates & Curve":          ["GT2","GT10","GT30","10–2 slope","30–10 slope","yield changes"],
            "Macro & Inflation":      ["CPI change","CPI YoY","inflation regime","ISM","UMich","CB confidence"],
            "Global Risk":            ["DXY level","DXY return","DXY momentum","DXY volatility","DXY–SPX corr"],
            "Safe Haven":             ["Gold return","GLD/SPX ratio","ratio return","gold momentum"],
            "Monetary":               ["M2 growth 1m","M2 growth 3m","M2 YoY lag","NBER recession lag"],
            "Augmented Features":     ["_yield_trend_63d","_spx_cum_21d","_vix_yield_interaction"],
        }
        cards = ""
        for name, items in groups.items():
            rows   = "".join(f"<li style='margin-bottom:4px;color:#A8C4E0;font-size:13px;'>{x}</li>" for x in items)
            cards += (f"<div style='background:#0F1E35;border:1px solid #1E3050;"
                      f"border-radius:10px;padding:16px 18px;'>"
                      f"<div style='font-size:13px;font-weight:700;color:#C9A84C;"
                      f"margin-bottom:8px;'>{name}</div>"
                      f"<ul style='padding-left:18px;margin:0;'>{rows}</ul></div>")
        return ui.HTML(
            f"<div style='padding:10px 0 4px;'>"
            f"<div style='font-size:20px;font-weight:700;color:#E8EDF5;margin-bottom:6px;'>"
            f"Data & Feature Matrix</div>"
            f"<div style='font-size:13px;color:#7A8BA0;margin-bottom:16px;'>"
            f"49 daily macro and market signals, lagged at least one day before use.</div>"
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);"
            f"gap:14px;margin-bottom:16px;'>{cards}</div></div>")


    @output
    @render.ui
    def da_pca_chart():
        res = da_results.get()
        if res is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Run GMM analysis first.</div>")
        ev  = res["ev"]
        cum = np.cumsum(ev)
        n   = len(ev)
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Explained Variance per Component","Cumulative Variance"])
        fig.add_trace(go.Bar(x=list(range(1,n+1)), y=list(ev*100),
            marker_color="#C9A84C", opacity=0.85,
            hovertemplate="PC%{x}: %{y:.2f}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1,n+1)), y=list(cum*100),
            line=dict(color="#2ECC71", width=2),
            hovertemplate="PC%{x}: cum %{y:.1f}%<extra></extra>"), row=1, col=2)
        fig.add_hline(y=80, line_dash="dash", line_color="#E74C3C",
                      annotation_text="80%", row=1, col=2)
        fig.add_vline(x=res["actual_n_pca"], line_dash="dash",
                      line_color="#C9A84C", row=1, col=1)
        fig.update_layout(height=380, plot_bgcolor="#0A1628", paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"), showlegend=False,
            margin=dict(l=40,r=20,t=40,b=40))
        fig.update_xaxes(gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0"))
        fig.update_yaxes(gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0"))
        return to_html(fig)


    @output
    @render.ui
    def da_ksel_chart():
        res = da_results.get()
        if res is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Run analysis first.</div>")
        k_vals = res["k_vals"]; x = [str(ki) for ki in k_vals]
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=3,
            subplot_titles=["Silhouette Score","AIC — Lower is Better","BIC — Lower is Better"])
        sil = res["sil_scores"]; best_sil = int(np.argmax(sil))
        fig.add_trace(go.Bar(x=x, y=sil,
            marker_color=["#2ECC71" if i==best_sil else "#C9A84C" for i in range(len(k_vals))],
            text=[f"{v:.4f}" for v in sil], textposition="outside",
            textfont=dict(color="#E8EDF5",size=10)), row=1, col=1)
        aic = res["aic_scores"]; best_aic = int(np.argmin(aic))
        fig.add_trace(go.Bar(x=x, y=aic,
            marker_color=["#2ECC71" if i==best_aic else "#4FC3F7" for i in range(len(k_vals))],
            text=[f"{v:.0f}" for v in aic], textposition="outside",
            textfont=dict(color="#E8EDF5",size=10)), row=1, col=2)
        bic = res["bic_scores"]; best_bic = int(np.argmin(bic))
        fig.add_trace(go.Bar(x=x, y=bic,
            marker_color=["#2ECC71" if i==best_bic else "#9B59B6" for i in range(len(k_vals))],
            text=[f"{v:.0f}" for v in bic], textposition="outside",
            textfont=dict(color="#E8EDF5",size=10)), row=1, col=3)
        fig.update_layout(height=370, plot_bgcolor="#0A1628", paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"), showlegend=False, margin=dict(l=40,r=20,t=50,b=40),
            title=dict(text="GMM Model Selection — k=5 preferred for clearer risk signaling.",
                       font=dict(color="#7A8BA0",size=11)))
        fig.update_xaxes(gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0"), title_text="k")
        fig.update_yaxes(gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0"))
        return to_html(fig)


    @output
    @render.data_frame
    def da_silhouette_tbl():
        res = da_results.get()
        if res is None:
            return render.DataGrid(pd.DataFrame({"Info": ["Run analysis first."]}))
        df = pd.DataFrame({"k": res["k_vals"],
                            "Average Silhouette": [round(x,4) for x in res["sil_scores"]]})
        df["Interpretation"] = np.where(df["k"]==5,
            "Most useful for risk signaling","Less distinct / more overlap")
        return render.DataGrid(df, height="180px")


    @output
    @render.ui
    def da_gmm_chart():
        res = da_results.get()
        if res is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Run GMM analysis first.</div>")
        idx   = pd.to_datetime(res["index"])
        probs = res["probs"]
        k     = res["k"]
        pal   = ["#C0392B","#E67E22","#2ECC71","#4FC3F7","#9B59B6"]
        fig   = go.Figure()
        for c in range(k):
            fig.add_trace(go.Scatter(
                x=idx, y=probs[:,c], mode="lines", stackgroup="one",
                name=f"regime_prob_{c}",
                line=dict(color=pal[c%len(pal)], width=0.5),
                hovertemplate=f"regime_prob_{c}: %{{y:.2f}}<extra></extra>"))
        sd = res.get("split_date")
        if sd is not None:
            sd = pd.to_datetime(sd)
            fig.add_shape(type="line", x0=sd, x1=sd, y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(color="#FFFFFF", dash="dash", width=1))
            fig.add_annotation(x=sd, y=1.03, xref="x", yref="paper",
                               text=f"Split {sd.date()}", showarrow=False,
                               font=dict(color="#FFFFFF", size=10))
        fig.update_layout(
            height=460, plot_bgcolor="#0A1628", paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"), margin=dict(l=40,r=20,t=40,b=40),
            title=dict(
                text=f"Stacked GMM Regime Probabilities — k={k} | {res['date_range']}",
                font=dict(color="#C9A84C",size=13)),
            yaxis=dict(title="Probability", range=[0,1],
                       gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0")),
            xaxis=dict(gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0")),
            legend=dict(orientation="h", y=-0.01, bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=10)))
        return to_html(fig)


    @output
    @render.data_frame
    def da_stats_tbl():
        res = da_results.get()
        if res is None:
            return render.DataGrid(pd.DataFrame({"Info": ["Run analysis first."]}))
        df = res.get("regime_profile_display")
        if df is None:
            return render.DataGrid(pd.DataFrame({"Info": ["No profile data."]}))
        return render.DataGrid(df, height="420px", summary=False)


    @output
    @render.ui
    def da_transition_chart():
        res = da_results.get()
        if res is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Run analysis first.</div>")
        tm  = res["transition_matrix"]
        fig = go.Figure(go.Heatmap(
            z=tm.values,
            x=[f"C{i}" for i in tm.columns],
            y=[f"C{i}" for i in tm.index],
            text=[[f"{v:.2f}" for v in row] for row in tm.values],
            texttemplate="%{text}",
            colorscale=[[0,"#152742"],[1,"#2ECC71"]],
            colorbar=dict(tickfont=dict(color="#7A8BA0"),
                          title=dict(text="Prob.",font=dict(color="#7A8BA0"))),
            hovertemplate="From %{y} → %{x}<br>Prob: %{z:.3f}<extra></extra>"))
        fig.update_layout(height=420, plot_bgcolor="#0A1628", paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"), margin=dict(l=60,r=60,t=50,b=50),
            title=dict(text="Regime Persistence and Transition Dynamics",
                       font=dict(color="#C9A84C",size=13)),
            xaxis=dict(title="Next Regime",    tickfont=dict(color="#7A8BA0")),
            yaxis=dict(title="Current Regime", tickfont=dict(color="#7A8BA0")))
        return to_html(fig)


    @output
    @render.ui
    def da_xgb_chart():
        res = da_results.get()
        if res is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Run analysis first.</div>")
        if not res.get("xgb_ok", False):
            return ui.HTML("<div style='color:#F39C12;padding:20px;'>XGBoost prediction not available.</div>")
        idx     = res["index"]
        gmm_lbl = pd.Series(res["labels"], index=idx)
        xgb_lbl = pd.Series(res["xgb_preds"], index=res["xgb_index"])
        common  = xgb_lbl.index
        gmm_aln = gmm_lbl.reindex(common)
        agreement = gmm_aln.values == xgb_lbl.values
        agree_pct = agreement.mean() * 100
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.4,0.4,0.2], vertical_spacing=0.04,
            subplot_titles=["GMM Cluster (test period)","XGBoost Prediction","Agreement"])
        pal = ["#C0392B","#E67E22","#2ECC71","#4FC3F7","#9B59B6"]
        k   = res["k"]
        for c in range(k):
            mask = gmm_aln == c
            if mask.any():
                fig.add_trace(go.Scatter(x=common[mask], y=np.full(mask.sum(),c),
                    mode="markers",
                    marker=dict(color=pal[c%len(pal)],size=2,symbol="square"),
                    name=f"GMM C{c}", showlegend=True), row=1, col=1)
        for c in range(k):
            mask = xgb_lbl == c
            if mask.any():
                fig.add_trace(go.Scatter(x=common[mask], y=np.full(mask.sum(),c),
                    mode="markers",
                    marker=dict(color=pal[c%len(pal)],size=2,symbol="square"),
                    showlegend=False), row=2, col=1)
        fig.add_trace(go.Bar(x=common, y=agreement.astype(int),
            marker_color=["#2ECC71" if a else "#E74C3C" for a in agreement],
            showlegend=False), row=3, col=1)
        fig.update_layout(height=540, plot_bgcolor="#0A1628", paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"), margin=dict(l=50,r=20,t=60,b=40),
            title=dict(
                text=(f"GMM vs XGBoost 5-Class — OOS Accuracy: {agree_pct:.1f}%  |  "
                      f"Test: {res['split_date'].date()} → {res['index'][-1].date()}"),
                font=dict(color="#C9A84C",size=13)),
            legend=dict(orientation="h",y=1.06,bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8EDF5",size=10)))
        fig.update_xaxes(gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0"))
        fig.update_yaxes(gridcolor="#1A2E48", tickfont=dict(color="#7A8BA0"))
        return to_html(fig)


    @output
    @render.data_frame
    def da_class_report():
        res = da_results.get()
        if res is None:
            return render.DataGrid(pd.DataFrame({"Info": ["Run analysis first."]}))
        return render.DataGrid(res["class_report_5"], height="340px")


    @output
    @render.ui
    def da_merged_confusion():
        res = da_results.get()
        if res is None:
            return ui.HTML("<div style='color:#7A8BA0;padding:20px;'>Run analysis first.</div>")
        cm   = res["cm_3"]
        labs = res["labels_3_order"]
        fig  = go.Figure(go.Heatmap(
            z=cm, x=labs, y=labs,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            colorscale=[[0,"#152742"],[1,"#2ECC71"]],
            colorbar=dict(tickfont=dict(color="#7A8BA0"),
                          title=dict(text="Count",font=dict(color="#7A8BA0"))),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"))
        acc_val = res["acc_3"]
        acc_str = f"{acc_val:.1%}" if acc_val == acc_val else "N/A"
        fig.update_layout(height=420, plot_bgcolor="#0A1628", paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"), margin=dict(l=130,r=60,t=50,b=110),
            title=dict(
                text=f"Merged 3-Class Confusion Matrix — Accuracy: {acc_str}",
                font=dict(color="#C9A84C",size=13)),
            xaxis=dict(title="Predicted", tickfont=dict(color="#7A8BA0")),
            yaxis=dict(title="Actual",    tickfont=dict(color="#7A8BA0")))
        return to_html(fig)


    @output
    @render.data_frame
    def da_merged_report():
        res = da_results.get()
        if res is None:
            return render.DataGrid(pd.DataFrame({"Info": ["Run analysis first."]}))
        return render.DataGrid(res["class_report_3"], height="300px")


    @output
    @render.data_frame
    def da_model_gains():
        res = da_results.get()
        if res is None:
            return render.DataGrid(pd.DataFrame({"Info": ["Run analysis first."]}))
        return render.DataGrid(res["model_gains"], height="280px")

    # ══════════════════════════════════════════════════════════
    # TAB 13 — EXPORT (PDF A: Client Report, PDF B: Dashboard)
    # ══════════════════════════════════════════════════════════

    pdf_a_path  = reactive.Value(None)
    pdf_b_path  = reactive.Value(None)
    pdf_a_msg   = reactive.Value("")
    pdf_b_msg   = reactive.Value("")
    EXPORT_DIR  = os.path.join(os.path.dirname(__file__), "..", "exports")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    @output
    @render.ui
    def pdf_a_status():
        msg = pdf_a_msg.get()
        if not msg: return ui.div()
        col = "#2ECC71" if "✓" in msg else "#E74C3C" if "❌" in msg else "#C9A84C"
        return ui.HTML(f"<div style='margin-top:8px;font-size:12px;color:{col};'>{msg}</div>")

    @output
    @render.ui
    def pdf_a_download():
        p = pdf_a_path.get()
        if not p or not os.path.exists(p): return ui.div()
        fname = os.path.basename(p)
        return ui.HTML(
            f"<div style='margin-top:12px;background:#0A1628;border:1px solid #2ECC71;"
            f"border-radius:8px;padding:12px;text-align:center;'>"
            f"<div style='font-size:12px;color:#2ECC71;margin-bottom:8px;'>✓ Report ready</div>"
            f"<div style='font-size:11px;color:#7A8BA0;'>{fname}</div>"
            f"<div style='font-size:11px;color:#7A8BA0;margin-top:4px;'>"
            f"File saved to: exports/{fname}</div></div>"
        )

    @output
    @render.ui
    def pdf_b_status():
        msg = pdf_b_msg.get()
        if not msg: return ui.div()
        col = "#2ECC71" if "✓" in msg else "#E74C3C" if "❌" in msg else "#C9A84C"
        return ui.HTML(f"<div style='margin-top:8px;font-size:12px;color:{col};'>{msg}</div>")

    @output
    @render.ui
    def pdf_b_download():
        p = pdf_b_path.get()
        if not p or not os.path.exists(p): return ui.div()
        fname = os.path.basename(p)
        return ui.HTML(
            f"<div style='margin-top:12px;background:#0A1628;border:1px solid #2ECC71;"
            f"border-radius:8px;padding:12px;text-align:center;'>"
            f"<div style='font-size:12px;color:#2ECC71;margin-bottom:8px;'>✓ Export ready</div>"
            f"<div style='font-size:11px;color:#7A8BA0;'>{fname}</div>"
            f"<div style='font-size:11px;color:#7A8BA0;margin-top:4px;'>"
            f"File saved to: exports/{fname}</div></div>"
        )

    # ── Option A: Professional Client Report ──────────────────
    @reactive.effect
    @reactive.event(input.pdf_a_btn)
    def _gen_pdf_a():
        pdf_a_msg.set("⟳  Generating your RAAM report...")
        pdf_a_path.set(None)
    
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer,
                Table, TableStyle, HRFlowable, PageBreak
            )
            from reportlab.lib.enums import TA_CENTER
    
            # ── Data ─────────────────────────────────────────
            name = client_name.get() or "Valued Client"
    
            ap = applied_profile.get()
            profile = ap["profile"] if ap else "Moderate"
    
            init = float(client_init.get() or 100000)
    
            date_str = datetime.now().strftime("%B %d, %Y")
    
            rr = run_results.get() or {}
            model_lbl = rr.get("model", "—")
    
            fname = f"RAAM_Report_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            fpath = os.path.join(EXPORT_DIR, fname)
    
            # ── Colors: brighter text + darker tables ─────────
            GOLD  = colors.HexColor("#C9A84C")
            LIGHT = colors.HexColor("#FFFFFF")
            MUTED = colors.HexColor("#475569")
            SLATE = colors.HexColor("#0F1E35")
            DARK  = colors.HexColor("#0D1E36")
            LINE  = colors.HexColor("#1E3050")
    
            # ── Document ──────────────────────────────────────
            doc = SimpleDocTemplate(
                fpath,
                pagesize=letter,
                leftMargin=1 * inch,
                rightMargin=1 * inch,
                topMargin=1 * inch,
                bottomMargin=1 * inch,
            )
    
            def S(name_s, **kw):
                return ParagraphStyle(name_s, **kw)
                
            def tbl_style(header_color=None, row_colors=None):
                header_color = header_color or colors.HexColor("#142847")
                rc = row_colors or [
                    colors.HexColor("#F4F6FA"),
                    colors.HexColor("#E8EDF5"),
                ]
            
                return TableStyle([
                    # Header
                    ("BACKGROUND",     (0, 0), (-1, 0), header_color),
                    ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",       (0, 0), (-1, 0), 10),
            
                    # Body
                    ("TEXTCOLOR",      (0, 1), (-1, -1), colors.HexColor("#0D1E36")),
                    ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE",       (0, 1), (-1, -1), 9.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), rc),
            
                    # Borders and spacing
                    ("GRID",           (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING",     (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
                    ("LEFTPADDING",    (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING",   (0, 0), (-1, -1), 10),
            
                    # Alignment
                    ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
                    ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
                ])
    
            story = []
    
            # ════════════════════════════════════════════════
            # PAGE 1 — Cover / Thank You
            # ════════════════════════════════════════════════
    
            story.append(Spacer(1, 1.5 * inch))
    
            story.append(Paragraph(
                "RAAM",
                S(
                    "brand",
                    fontSize=36,
                    textColor=GOLD,
                    alignment=TA_CENTER,
                    fontName="Helvetica-Bold",
                    spaceAfter=25,
                ),
            ))

            story.append(HRFlowable(
                width="50%",
                color=GOLD,
                thickness=1.5,
                hAlign="CENTER",
            ))
    
            story.append(Paragraph(
                "Regime-Aware Asset Management",
                S(
                    "sub",
                    fontSize=13,
                    textColor=MUTED,
                    alignment=TA_CENTER,
                    spaceAfter=40,
                ),
            ))
    

    
            story.append(Spacer(1, 0.5 * inch))
    
            story.append(Paragraph(
                f"Thank you for using the RAAM Platform, {name}.",
                S(
                    "ty",
                    fontSize=15,
                    textColor=DARK,
                    alignment=TA_CENTER,
                    fontName="Helvetica-Bold",
                    leading=22,
                    spaceAfter=20,
                ),
            ))
    
            story.append(Paragraph(
                "Your personalized portfolio report has been generated based on your risk profile "
                "and the regime-aware model you selected. This report covers your risk profile summary, "
                "backtest performance overview, and current allocation recommendation.",
                S(
                    "body",
                    fontSize=11,
                    textColor=MUTED,
                    alignment=TA_CENTER,
                    leading=18,
                    spaceAfter=30,
                ),
            ))
    
            info_data = [
                ["Report Date",       date_str],
                ["Client Name",       name],
                ["Risk Profile",      profile],
                ["Model Selected",    model_lbl],
                ["Initial Investment", f"${init:,.0f}"],
            ]
            
            info_tbl = Table(
                info_data,
                colWidths=[2.0 * inch, 3.8 * inch]
            )
            
            info_tbl.setStyle(TableStyle([
            
                # Alternating soft background
                ("ROWBACKGROUNDS",
                    (0, 0), (-1, -1),
                    [
                        colors.HexColor("#F8FAFC"),
                        colors.HexColor("#EEF2F7")
                    ]
                ),
            
                # Left column labels
                ("TEXTCOLOR",
                    (0, 0), (0, -1),
                    colors.HexColor("#475569")
                ),
            
                ("FONTNAME",
                    (0, 0), (0, -1),
                    "Helvetica-Bold"
                ),
            
                ("FONTSIZE",
                    (0, 0), (0, -1),
                    9.5
                ),
            
                # Right column values
                ("TEXTCOLOR",
                    (1, 0), (1, -1),
                    colors.HexColor("#0F172A")
                ),
            
                ("FONTNAME",
                    (1, 0), (1, -1),
                    "Helvetica-Bold"
                ),
            
                ("FONTSIZE",
                    (1, 0), (1, -1),
                    10.5
                ),
            
                # Elegant borders
                ("LINEBELOW",
                    (0, 0), (-1, -2),
                    0.35,
                    colors.HexColor("#CBD5E1")
                ),
            
                ("BOX",
                    (0, 0), (-1, -1),
                    0.75,
                    colors.HexColor("#CBD5E1")
                ),
            
                # Padding
                ("TOPPADDING",
                    (0, 0), (-1, -1),
                    10
                ),
            
                ("BOTTOMPADDING",
                    (0, 0), (-1, -1),
                    10
                ),
            
                ("LEFTPADDING",
                    (0, 0), (-1, -1),
                    16
                ),
            
                ("RIGHTPADDING",
                    (0, 0), (-1, -1),
                    16
                ),
            
                ("VALIGN",
                    (0, 0), (-1, -1),
                    "MIDDLE"
                ),
            
                ("ALIGN",
                    (0, 0), (-1, -1),
                    "LEFT"
                ),
            ]))
    
            story.append(info_tbl)
            story.append(PageBreak())
    
            # ════════════════════════════════════════════════
            # PAGE 2 — Risk Profile
            # ════════════════════════════════════════════════
            
            story.append(Paragraph(
                "1.  Your Risk Profile",
                S("h1", fontSize=16, textColor=GOLD, fontName="Helvetica-Bold", spaceAfter=6),
            ))
            
            story.append(HRFlowable(width="100%", color=LINE, thickness=1))
            story.append(Spacer(1, 0.2 * inch))
            
            if not ap:
                ap = {
                    "profile": profile,
                    "gamma_bull": 1.05,
                    "gamma_trans": 1.93,
                    "gamma_bear": 3.50,
                    "bond_min_Bear": 0.25,
                    "benchmark": {"eq": 0.55, "bond": 0.40, "alt": 0.05},
                }
            
            profile = ap.get("profile", profile)
            bm = ap["benchmark"]
            
            desc_map = {
                "Conservative": (
                    "Your priority is protecting capital with steady, predictable returns. "
                    "The model uses higher risk aversion parameters to reduce equity exposure "
                    "and increase bond allocation, especially during bear regimes."
                ),
                "Moderate": (
                    "You seek a balance between growth and protection with manageable volatility. "
                    "The model balances equity and bond exposure dynamically across regimes."
                ),
                "Aggressive": (
                    "You prioritize maximum long-term growth and accept significant short-term volatility. "
                    "The model maintains higher equity exposure across all regimes."
                ),
            }
            
            story.append(Paragraph(
                profile,
                S("prof", fontSize=18, textColor=GOLD, fontName="Helvetica-Bold", spaceAfter=8),
            ))
            
            story.append(Paragraph(
                desc_map.get(profile, "Balanced approach."),
                S("pdesc", fontSize=11, textColor=colors.HexColor("#0F172A"), leading=17, spaceAfter=20),
            ))
            
            # ── Table 1: Risk Aversion Parameters ─────────────────────
            story.append(Paragraph(
                "Risk Aversion Parameters",
                S("sh_risk", fontSize=11, textColor=colors.HexColor("#475569"),
                  fontName="Helvetica-Bold", spaceAfter=8),
            ))
            
            risk_data = [
                ["Regime", "Bull", "Transition", "Bear"],
                [
                    "Risk Aversion (γ)",
                    str(ap["gamma_bull"]),
                    str(ap["gamma_trans"]),
                    str(ap["gamma_bear"]),
                ],
            ]
            
            risk_tbl = Table(
                risk_data,
                colWidths=[2.0 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch],
            )
            
            risk_tbl.setStyle(tbl_style())
            story.append(risk_tbl)
            
            story.append(Spacer(1, 0.25 * inch))
            
            # ── Table 2: Static Benchmark Allocation ───────────────────
            story.append(Paragraph(
                "Static Benchmark Allocation",
                S("sh_bm", fontSize=11, textColor=colors.HexColor("#475569"),
                  fontName="Helvetica-Bold", spaceAfter=8),
            ))
            
            benchmark_data = [
                ["Asset Class", "Benchmark Weight"],
                ["Equity", f"{int(bm['eq'] * 100)}%"],
                ["Bonds", f"{int(bm['bond'] * 100)}%"],
                ["Gold", f"{int(bm['alt'] * 100)}%"],
            ]
            
            benchmark_tbl = Table(
                benchmark_data,
                colWidths=[3.0 * inch, 2.0 * inch],
            )
            
            benchmark_tbl.setStyle(tbl_style())
            story.append(benchmark_tbl)
            
            story.append(PageBreak())
                
            # ════════════════════════════════════════════════
            # PAGE 3 — Backtest Performance
            # ════════════════════════════════════════════════
    
            story.append(Paragraph(
                "2.  Backtest Performance Overview",
                S(
                    "h2",
                    fontSize=16,
                    textColor=GOLD,
                    fontName="Helvetica-Bold",
                    spaceAfter=6,
                ),
            ))
    
            story.append(HRFlowable(width="100%", color=LINE, thickness=1))
            story.append(Spacer(1, 0.1 * inch))
    
            story.append(Paragraph(
                f"Model: {model_lbl}    Period: 2019 - 2025    Method: Walk-Forward Validation",
                S(
                    "src",
                    fontSize=10,
                    textColor=MUTED,
                    spaceAfter=14,
                ),
            ))
    
            bt_wf, bt_bm, _, _, is_real = current_results()
            ANN = 252
    
            if len(bt_wf) > 5:
                r = bt_wf["ret"]
                cum = bt_wf["cum"]
                bm_r = bt_bm["ret"].reindex(r.index).fillna(0)
    
                m = _met(r, cum, bm_r)
                bm_m = _met(bm_r, bt_bm["cum"])
    
                sw = int(bt_wf.get("rebalanced", pd.Series(dtype=bool)).sum())
    
                story.append(Paragraph(
                    "Key Performance Metrics",
                    S(
                        "sh2",
                        fontSize=11,
                        textColor=MUTED,
                        fontName="Helvetica-Bold",
                        spaceAfter=8,
                    ),
                ))
    
                kpi_data = [
                    ["Metric", "RAAM Model", "Benchmark"],
                    ["Final Value", f"${init * cum.iloc[-1]:,.0f}", f"${init * bt_bm['cum'].iloc[-1]:,.0f}"],
                    ["Ann. Return", f"{m.get('ar', 0) * 100:.2f}%", f"{bm_m.get('ar', 0) * 100:.2f}%"],
                    ["Sharpe Ratio", f"{m.get('sh', 0):.3f}", f"{bm_m.get('sh', 0):.3f}"],
                    ["Sortino Ratio", f"{m.get('so', 0):.3f}", f"{bm_m.get('so', 0):.3f}"],
                    ["Calmar Ratio", f"{m.get('cal', 0):.3f}", f"{bm_m.get('cal', 0):.3f}"],
                    ["Max Drawdown", f"{m.get('mdd', 0) * 100:.1f}%", f"{bm_m.get('mdd', 0) * 100:.1f}%"],
                    ["Sharpe Alpha", f"{m.get('sh', 0) - bm_m.get('sh', 0):.3f}", "0.000"],
                    ["Regime Switches", str(sw), "0"],
                ]
    
                kt = Table(kpi_data, colWidths=[2.5 * inch, 2 * inch, 2 * inch])
                kt.setStyle(tbl_style())
                story.append(kt)
                story.append(Spacer(1, 0.25 * inch))
    
                if "regime" in bt_wf.columns:
                    story.append(Paragraph(
                        "Regime Alpha Attribution",
                        S(
                            "sh3",
                            fontSize=11,
                            textColor=MUTED,
                            fontName="Helvetica-Bold",
                            spaceAfter=8,
                        ),
                    ))
    
                    reg_data = [["Regime", "Days", "% Time", "Model Ret", "BM Ret", "Alpha", "Sharpe"]]
    
                    for reg in ["Bull", "Transition", "Bear"]:
                        mask = bt_wf["regime"] == reg
    
                        if mask.sum() < 5:
                            continue
    
                        wfr = bt_wf.loc[mask, "ret"]
                        bmr2 = bm_r[mask]
    
                        n2 = ANN / len(wfr)
                        wfa = (1 + wfr).prod() ** n2 - 1
                        bma2 = (1 + bmr2).prod() ** n2 - 1
    
                        vol2 = wfr.std() * np.sqrt(ANN)
                        sh2 = wfr.mean() * ANN / vol2 if vol2 > 0 else 0
    
                        reg_data.append([
                            reg,
                            str(int(mask.sum())),
                            f"{mask.mean() * 100:.1f}%",
                            f"{wfa * 100:.2f}%",
                            f"{bma2 * 100:.2f}%",
                            f"{(wfa - bma2) * 100:+.2f}%",
                            f"{sh2:.3f}",
                        ])
    
                    rt = Table(
                        reg_data,
                        colWidths=[
                            1.1 * inch,
                            0.7 * inch,
                            0.8 * inch,
                            1.1 * inch,
                            1.0 * inch,
                            0.9 * inch,
                            0.8 * inch,
                        ],
                    )
                    rt.setStyle(tbl_style())
                    story.append(rt)
    
            story.append(PageBreak())
    
            # ════════════════════════════════════════════════
            # PAGE 4 — Recommended Allocation
            # ════════════════════════════════════════════════
    
            story.append(Paragraph(
                "3.  Recommended Allocation",
                S(
                    "h4",
                    fontSize=16,
                    textColor=GOLD,
                    fontName="Helvetica-Bold",
                    spaceAfter=6,
                ),
            ))
    
            story.append(HRFlowable(width="100%", color=LINE, thickness=1))
            story.append(Spacer(1, 0.2 * inch))
    
            story.append(Paragraph(
                "Based on the latest regime detected by the model, "
                "the following portfolio allocation is recommended:",
                S(
                    "body2",
                    fontSize=11,
                    textColor=MUTED,
                    leading=17,
                    spaceAfter=16,
                ),
            ))
    
            weights_df = rr.get("weights")
    
            if weights_df is not None and len(weights_df) > 0:
                latest_w = weights_df.iloc[-1].to_dict()
    
                latest_regime = (
                    bt_wf["regime"].iloc[-1]
                    if "regime" in bt_wf.columns
                    else "Unknown"
                )
    
                story.append(Paragraph(
                    f"Current Detected Regime:  {latest_regime}",
                    S(
                        "reg",
                        fontSize=13,
                        textColor=GOLD,
                        fontName="Helvetica-Bold",
                        spaceAfter=12,
                    ),
                ))
    
                bm_eq = ap["benchmark"]["eq"] if ap else 0.55
                bm_bond = ap["benchmark"]["bond"] if ap else 0.40
                bm_alt = ap["benchmark"]["alt"] if ap else 0.05
    
                bm = rr.get("benchmark_weights")
    
                wt_data = [["ETF", "RAAM Weight", "Benchmark", "Active Tilt"]]
    
                for etf, w in sorted(latest_w.items(), key=lambda x: -abs(x[1])):
                    bw = bm.get(etf, 0.0)
                    tilt = w - bw
    
                    wt_data.append([
                        etf,
                        f"{w * 100:.1f}%",
                        f"{bw * 100:.1f}%",
                        f"{tilt * 100:+.1f}%",
                    ])
    
                wt = Table(
                    wt_data,
                    colWidths=[
                        1.5 * inch,
                        1.5 * inch,
                        1.5 * inch,
                        1.5 * inch,
                    ],
                )
                wt.setStyle(tbl_style())
                story.append(wt)
    
            else:
                story.append(Paragraph(
                    "No model results found. Please run the model from the Run Model tab first, "
                    "then generate the report.",
                    S(
                        "na",
                        fontSize=11,
                        textColor=LIGHT,
                        leading=17,
                    ),
                ))
    
            # ── Footer ────────────────────────────────────────
            story.append(Spacer(1, 0.5 * inch))
            story.append(HRFlowable(width="100%", color=LINE, thickness=1))
            story.append(Spacer(1, 0.1 * inch))
    
            story.append(Paragraph(
                f"RAAM Platform  |  Stevens Institute of Technology — FE800  |  {date_str}  |  "
                f"For informational and educational purposes only. Not financial advice.",
                S(
                    "footer",
                    fontSize=8,
                    textColor=MUTED,
                    alignment=TA_CENTER,
                ),
            ))
    
            doc.build(story)
    
            pdf_a_path.set(fpath)
            pdf_a_msg.set(f"✓  Report generated: {fname}")
    
        except ImportError:
            pdf_a_msg.set("❌  reportlab not installed. Run: pip install reportlab")
    
        except Exception as ex:
            pdf_a_msg.set(f"❌  Error: {ex}")

    # ── Figure builders (kept for internal use) ──────────────
    def _build_overview_fig(bt_wf, bt_bm, bt_mv, init):
        fig = go.Figure()
        for nm,bt,col,dash in [("Walk-Fwd",bt_wf,"#C9A84C","solid"),
                                 ("Benchmark",bt_bm,"#4FC3F7","dash"),
                                 ("MV",bt_mv,"#1ABC9C","dot")]:
            if len(bt)>0:
                fig.add_trace(go.Scatter(x=bt.index,y=bt["cum"]*init,
                    name=nm,line=dict(color=col,width=2,dash=dash)))
        fig.update_layout(height=500,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=50,r=20,t=30,b=50),
            yaxis=dict(tickformat="$,.0f",title="Portfolio Value ($)",gridcolor="#1A2E48"),
            xaxis=dict(gridcolor="#1A2E48"),
            legend=dict(orientation="h",y=1.06,bgcolor="rgba(0,0,0,0)",font=dict(color="#E8EDF5")))
        return fig

    def _build_annual_fig(bt_wf, bt_bm, bt_mv, init):
        ANN=252; years=sorted(bt_wf.index.year.unique()); x=np.arange(len(years)); w=0.26
        fig=go.Figure()
        for i,(nm,bt,col) in enumerate([("BM",bt_bm,"#4FC3F7"),("MV",bt_mv,"#1ABC9C"),("WF",bt_wf,"#C9A84C")]):
            vals=[]
            for y in years:
                sub=bt[bt.index.year==y]; r=sub["ret"]
                vals.append(((1+r).prod()-1)*100 if len(sub)>5 else 0)
            fig.add_trace(go.Bar(x=list(x+i*w),y=vals,name=nm,marker_color=col,width=w*0.9))
        fig.update_layout(barmode="group",height=500,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=30,b=50),
            xaxis=dict(tickvals=list(x+w),ticktext=[str(y) for y in years],gridcolor="#1A2E48"),
            yaxis=dict(gridcolor="#1A2E48",title="Return (%)"),
            legend=dict(orientation="h",y=1.06,bgcolor="rgba(0,0,0,0)",font=dict(color="#E8EDF5")))
        return fig

    def _build_regime_fig(bt_wf, bt_bm, bt_mv, init):
        fig=go.Figure()
        if "regime" in bt_wf.columns:
            for reg,col in [("Bull","#2ECC71"),("Transition","#F39C12"),("Bear","#E74C3C")]:
                sub=bt_wf[bt_wf["regime"]==reg]
                dd=(sub["cum"]-sub["cum"].cummax())/sub["cum"].cummax()*100
                fig.add_trace(go.Scatter(x=sub.index,y=sub["cum"]*init,name=reg,
                    line=dict(color=col,width=2)))
        fig.update_layout(height=500,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=50,r=20,t=30,b=50),
            yaxis=dict(tickformat="$,.0f",gridcolor="#1A2E48"),xaxis=dict(gridcolor="#1A2E48"),
            legend=dict(orientation="h",y=1.06,bgcolor="rgba(0,0,0,0)",font=dict(color="#E8EDF5")))
        return fig

    def _build_risk_fig(bt_wf, bt_bm, bt_mv, init):
        r=bt_wf["ret"]; cum=bt_wf["cum"]
        bm_r=bt_bm["ret"].reindex(r.index).fillna(0)
        m=_met(r,cum,bm_r)
        metrics=["Sharpe","Sortino","Calmar","Alpha"]
        vals=[m.get("sh",0),m.get("so",0),m.get("cal",0),m.get("alpha",0)*100]
        cols=["#2ECC71" if v>0 else "#E74C3C" for v in vals]
        fig=go.Figure(go.Bar(x=metrics,y=vals,marker_color=cols,
            text=[f"{v:.3f}" for v in vals],textposition="outside",
            textfont=dict(color="#E8EDF5")))
        fig.update_layout(height=500,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=30,b=50),
            xaxis=dict(gridcolor="#1A2E48"),yaxis=dict(gridcolor="#1A2E48"))
        return fig

    def _build_turnover_fig(bt_wf, bt_bm, bt_mv, init):
        if "turnover" not in bt_wf.columns:
            return go.Figure()
        ato=bt_wf.groupby(bt_wf.index.year)["turnover"].sum()
        x=list(range(len(ato)))
        fig=go.Figure(go.Bar(x=x,y=list(ato.values),marker_color="#C9A84C"))
        fig.update_layout(height=500,plot_bgcolor="#0A1628",paper_bgcolor="#0F1E35",
            font=dict(color="#E8EDF5"),margin=dict(l=40,r=20,t=30,b=50),
            xaxis=dict(tickvals=x,ticktext=[str(y) for y in ato.index],gridcolor="#1A2E48"),
            yaxis=dict(gridcolor="#1A2E48",title="Annual Turnover"))
        return fig


app = App(app_ui, server)