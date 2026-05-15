#!/usr/bin/env python
# =============================================================
# run_all.py — Run all models and compare
#
# Usage:
#   python run_all.py                        # all models
#   python run_all.py --models ML HMM        # specific models
#   python run_all.py --no-plots             # skip charts
#   python run_all.py --risk Aggressive      # risk profile
# =============================================================

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "config"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

from data.data_loader import load_all
from backtest.backtest_engine import (
    backtest_wf, backtest_static, backtest_mv, trim_to_test,
    print_all_results, plot_model_comparison, full_summary,
)
from portfolio.portfolio_allocator import build_mv_benchmark
from config import TICKER_MAP, FIRST_TRAIN_END, REFIT_MONTHS, TC_BPS

ALL_MODELS = ["ML", "HMM", "HSMM", "Jump", "RHSM"]
MODEL_LABELS = {
    "ML":   "ML (GMM+XGBoost)",
    "HMM":  "HMM",
    "HSMM": "Semi-Markov (HSMM)",
    "Jump": "Jump Diffusion",
    "RHSM": "Reg. HSM (rHSM)",
}


def load_model(name):
    if name == "ML":
        from models import ml_regime_model as m
        return m
    elif name == "HMM":
        from models import hmm_model as m
        return m
    elif name == "HSMM":
        from models import semi_markov_model as m
        return m
    elif name == "Jump":
        from models import jump_model as m
        return m
    elif name == "RHSM":
        from models import rhsm_model as m
        return m
    raise ValueError(f"Unknown model: {name}")


def run_model(name, data, risk_profile, show_progress=True):
    mod     = load_model(name)
    X_all   = data["X_all"]
    r_all   = data["r_all"]
    bm      = data["benchmark_series"]

    def progress(fold, total, step):
        if show_progress:
            print(f"  [{name}] Fold {fold}/{total} — {step}")

    t0 = time.time()
    try:
        regime_probs, fold_ports = mod.run_walk_forward(
            X_all, r_all, bm,
            risk_profile=risk_profile,
            progress_callback=progress,
        )
    except Exception as e:
        print(f"❌  {name} failed: {e}")
        return None

    # Build confirmed signal
    if hasattr(mod, "build_asymmetric_regime_signal"):
        confirmed = mod.build_asymmetric_regime_signal(regime_probs)
    elif hasattr(mod, "build_hmm_signal"):
        confirmed = mod.build_hmm_signal(regime_probs)
    else:
        # Default: argmax
        from config import REGIME_ORDER
        col_map   = {f"prob_{r}": r for r in REGIME_ORDER}
        confirmed = regime_probs.idxmax(axis=1).map(col_map)

    r_all_t = r_all.rename(columns=TICKER_MAP).reindex(columns=bm.index)
    bt_wf, wt_wf = backtest_wf(r_all_t, confirmed, fold_ports)
    bt_wf_test   = trim_to_test(bt_wf)
    runtime      = (time.time()-t0)/60
    print(f"✅  {name} complete  ({runtime:.1f} min)")
    return bt_wf_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--risk",    default="Moderate",
                        choices=["Conservative","Moderate","Aggressive"])
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--initial", type=float, default=100_000)
    args = parser.parse_args()

    print("Loading data...")
    data    = load_all()
    bm      = data["benchmark_series"]
    r_all_t = data["r_all"].rename(columns=TICKER_MAP).reindex(columns=bm.index)

    # Shared baselines
    bt_bmark_all = backtest_static(r_all_t, bm)
    mv_weights   = build_mv_benchmark(data["r_all"], bm, FIRST_TRAIN_END, REFIT_MONTHS)
    bt_mv_all    = backtest_mv(r_all_t, mv_weights)
    bt_bmark     = trim_to_test(bt_bmark_all)
    bt_mv        = trim_to_test(bt_mv_all)

    # Run all selected models
    results = {}
    for name in args.models:
        print(f"\n{'='*60}")
        print(f"Running {MODEL_LABELS[name]}  (risk={args.risk})")
        print("="*60)
        bt = run_model(name, data, args.risk)
        if bt is not None:
            results[MODEL_LABELS[name]] = bt

    # Print comparison table
    if results:
        print(f"\n{'='*65}")
        print("MODEL COMPARISON SUMMARY")
        print("="*65)
        rows = [full_summary(bt_bmark, "Benchmark",    args.initial),
                full_summary(bt_mv,    "MV No-Regime", args.initial)]
        for label, bt in results.items():
            rows.append(full_summary(bt, label, args.initial))
        import pandas as pd
        print(pd.DataFrame(rows).T.to_string())

        # Plot comparison
        if not args.no_plots:
            all_bts = {"Benchmark": bt_bmark, "MV No-Regime": bt_mv, **results}
            plot_model_comparison(all_bts, args.initial)


if __name__ == "__main__":
    main()
