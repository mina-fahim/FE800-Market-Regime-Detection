# =============================================================
# src/backtest/backtest_engine.py
# Shared backtest engine, metrics, and plots — ALL models use this
# =============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../config"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import calendar
import warnings
warnings.filterwarnings("ignore")
from config import TC_BPS, MIN_HOLD_DAYS, ANN_FACTOR, FIRST_TRAIN_END, REGIME_ORDER

COLORS = {
    "Benchmark":  "steelblue",
    "MV":         "green",
    "Walk-Fwd":   "darkorange",
    "Bull":       "#2ecc71",
    "Transition": "#f39c12",
    "Bear":       "#e74c3c",
}

# =============================================================
# BACKTEST ENGINES
# =============================================================

def trim_to_test(bt, first_train_end=FIRST_TRAIN_END):
    bt = bt[bt.index >= pd.Timestamp(first_train_end) + pd.offsets.BDay(1)].copy()
    bt["cum"] = (1 + bt["ret"]).cumprod()
    return bt


def backtest_wf(
    returns_df, confirmed_regime, fold_portfolios,
    min_hold_days=MIN_HOLD_DAYS, tc_bps=TC_BPS,
):
    common     = returns_df.index.intersection(confirmed_regime.index)
    returns_df = returns_df.loc[common]
    confirmed  = confirmed_regime.loc[common].ffill().dropna()
    returns_df = returns_df.loc[confirmed.index]
    assets     = list(returns_df.columns)
    current_w  = pd.Series(0.0, index=assets)
    current_reg= None; days_held = 0
    rets, turnover, rebal_flag, reg_log, wts = [], [], [], [], []

    for dt in returns_df.index:
        r          = returns_df.loc[dt].fillna(0)
        new_regime = confirmed.loc[dt]
        ports_df   = fold_portfolios.get(dt, None)
        hold_thresh= min_hold_days.get(current_reg, 21) if current_reg else 21
        should_sw  = (current_reg is None) or (new_regime != current_reg and days_held >= hold_thresh)

        if should_sw and ports_df is not None:
            target_w = ports_df.loc[new_regime].reindex(assets).fillna(0.0)
            if target_w.sum() > 0: target_w /= target_w.sum()
            to = np.abs(current_w - target_w).sum()
            current_w = target_w.copy(); current_reg = new_regime; days_held = 0
        else:
            to = 0.0; days_held += 1

        cost = (tc_bps/10000) * to
        rets.append(float((current_w * r).sum()) - cost)
        turnover.append(to); rebal_flag.append(should_sw and ports_df is not None)
        reg_log.append(current_reg); wts.append(current_w.copy())
        post  = current_w * (1+r); denom = post.sum()
        current_w = post/denom if denom > 0 else current_w.copy()

    bt = pd.DataFrame({"ret":rets,"turnover":turnover,"rebalanced":rebal_flag,"regime":reg_log},
                      index=returns_df.index)
    bt["cum"] = (1+bt["ret"]).cumprod()
    wt_df     = pd.DataFrame(wts, index=returns_df.index)
    print(f"Switches={int(bt['rebalanced'].sum())}  Turnover={bt['turnover'].sum():.2f}")
    return bt, wt_df


def backtest_static(returns_df, weights, rebalance_months=6, tc_bps=TC_BPS):
    assets    = list(returns_df.columns)
    target_w  = weights.reindex(assets).fillna(0); target_w /= target_w.sum()
    months    = pd.Series(returns_df.index).dt.to_period("M")
    rebal     = pd.Series(months != months.shift(1), index=returns_df.index)
    current_w = pd.Series(0.0, index=assets)
    rets, turnover = [], []
    for dt in returns_df.index:
        r = returns_df.loc[dt].fillna(0)
        if current_w.sum() == 0 or rebal.loc[dt]:
            to = np.abs(current_w - target_w).sum(); current_w = target_w.copy()
        else: to = 0.0
        rets.append(float((current_w*r).sum()) - (tc_bps/10000)*to); turnover.append(to)
        post = current_w*(1+r); denom = post.sum()
        current_w = post/denom if denom > 0 else target_w.copy()
    out = pd.DataFrame({"ret":rets,"turnover":turnover}, index=returns_df.index)
    out["cum"] = (1+out["ret"]).cumprod()
    return out


def backtest_mv(returns_df, mv_weights_df, tc_bps=TC_BPS):
    common     = returns_df.index.intersection(mv_weights_df.index)
    returns_df = returns_df.loc[common]; weights_df = mv_weights_df.loc[common]
    assets     = list(returns_df.columns)
    current_w  = pd.Series(0.0, index=assets); last_w = None
    rets, turnover = [], []
    for dt in returns_df.index:
        r        = returns_df.loc[dt].fillna(0)
        target_w = weights_df.loc[dt].reindex(assets).fillna(0); target_w /= target_w.sum()
        if last_w is None or not target_w.equals(last_w):
            to = np.abs(current_w - target_w).sum(); current_w = target_w.copy(); last_w = target_w.copy()
        else: to = 0.0
        rets.append(float((current_w*r).sum()) - (tc_bps/10000)*to); turnover.append(to)
        post = current_w*(1+r); denom = post.sum()
        current_w = post/denom if denom > 0 else target_w.copy()
    out = pd.DataFrame({"ret":rets,"turnover":turnover}, index=returns_df.index)
    out["cum"] = (1+out["ret"]).cumprod()
    return out


# =============================================================
# METRICS
# =============================================================

def sharpe(r, ann=ANN_FACTOR):
    return (r.mean()*ann)/(r.std()*np.sqrt(ann)) if r.std()>0 else np.nan

def sortino(r, ann=ANN_FACTOR):
    down = r[r<0]
    dv   = np.sqrt((down**2).mean())*np.sqrt(ann) if len(down)>0 else np.nan
    return (r.mean()*ann)/dv if dv and dv>0 else np.nan

def calmar(r, cum, ann=ANN_FACTOR):
    ar  = (1+r).prod()**(ann/len(r))-1
    mdd = abs(max_drawdown(cum))
    return ar/mdd if mdd>0 else np.nan

def max_drawdown(cum):
    return ((cum-cum.cummax())/cum.cummax()).min()

def win_rate_monthly(r):
    m = (1+r).resample("ME").prod()-1
    return round((m>0).mean()*100, 1)

def ann_return(r, ann=ANN_FACTOR):
    return (1+r).prod()**(ann/len(r))-1

def full_summary(bt, name, initial_investment=100_000):
    r   = bt["ret"]; cum = bt["cum"]
    ar  = ann_return(r)
    return pd.Series({
        "Initial ($)":      f"${initial_investment:,.0f}",
        "Final ($)":        f"${initial_investment * cum.iloc[-1]:,.0f}",
        "Total Return":     f"{(cum.iloc[-1]-1)*100:.1f}%",
        "Ann. Return":      f"{ar*100:.2f}%",
        "Ann. Vol":         f"{r.std()*np.sqrt(ANN_FACTOR)*100:.2f}%",
        "Sharpe":           round(sharpe(r), 3),
        "Sortino":          round(sortino(r), 3),
        "Calmar":           round(calmar(r,cum), 3),
        "Max Drawdown":     f"{max_drawdown(cum)*100:.1f}%",
        "Win Rate (Mo)":    f"{win_rate_monthly(r)}%",
        "Switches":         int(bt.get("rebalanced", pd.Series()).sum()),
    }, name=name)

def annual_metrics(bt, ann=ANN_FACTOR):
    out = {}
    for y in sorted(bt.index.year.unique()):
        sub = bt[bt.index.year==y]
        if len(sub)<5: continue
        r = sub["ret"]; cum = (1+r).cumprod()
        out[y] = {"Ret": round((1+r).prod()-1,3),
                  "Sharpe": round(sharpe(r,ann),2),
                  "Sortino": round(sortino(r,ann),2),
                  "MaxDD": round(max_drawdown(cum),3),
                  "Win%": win_rate_monthly(r)}
    return pd.DataFrame(out).T

def regime_alpha_table(bt_wf, bt_bmark, ann=ANN_FACTOR):
    bmark_r = bt_bmark["ret"].reindex(bt_wf.index)
    rows = []
    for regime in REGIME_ORDER:
        mask = bt_wf["regime"]==regime
        wf_r = bt_wf.loc[mask,"ret"]; bm_r = bmark_r[mask]
        if len(wf_r)<5: continue
        cum   = (1+wf_r).cumprod()
        n_ann = ann/len(wf_r)
        wf_ann= (1+wf_r).prod()**n_ann - 1
        bm_ann= (1+bm_r).prod()**n_ann - 1
        rows.append(pd.Series({
            "Days":         len(wf_r),
            "% Time":       f"{len(wf_r)/len(bt_wf)*100:.1f}%",
            "Regime Ret":   f"{wf_ann*100:.2f}%",
            "Bmark Ret":    f"{bm_ann*100:.2f}%",
            "Alpha":        f"{(wf_ann-bm_ann)*100:+.2f}%",
            "Regime Sharpe":round(sharpe(wf_r,ann),3),
            "Bmark Sharpe": round(sharpe(bm_r,ann),3),
            "Δ Sharpe":     round(sharpe(wf_r,ann)-sharpe(bm_r,ann),3),
            "Sortino":      round(sortino(wf_r,ann),3),
            "MaxDD":        f"{max_drawdown(cum)*100:.1f}%",
        }, name=regime))
    return pd.DataFrame(rows)

def turnover_table(bt_wf, tc_bps=TC_BPS):
    rows = []
    for y in sorted(bt_wf.index.year.unique()):
        sub  = bt_wf[bt_wf.index.year==y]
        n_sw = int(sub["rebalanced"].sum())
        to   = round(sub["turnover"].sum(),3)
        avg  = round(sub.loc[sub["rebalanced"],"turnover"].mean(),3) if n_sw>0 else 0
        rows.append(pd.Series({"Switches":n_sw,"Total Turnover":to,
                               "Avg/Switch":avg,"TC (bps)":round(to*tc_bps,1)}, name=y))
    df = pd.DataFrame(rows)
    df.loc["TOTAL"] = df.sum(); df.loc["TOTAL","Avg/Switch"] = round(
        bt_wf.loc[bt_wf["rebalanced"],"turnover"].mean(),3)
    return df

def drawdown_table(bt, min_dd=-0.05):
    cum = bt["cum"]; dd = (cum-cum.cummax())/cum.cummax()
    rows = []; in_dd=False; peak_dt=trough_dt=trough_v=None
    for dt, val in dd.items():
        if not in_dd and val<0:
            in_dd=True; idx=cum.index.get_loc(dt)
            peak_dt=cum.index[idx-1] if idx>0 else dt
            trough_dt=dt; trough_v=val
        elif in_dd:
            if val<trough_v: trough_dt=dt; trough_v=val
            if val>=0:
                if trough_v<=min_dd:
                    rows.append({"Peak":peak_dt.date(),"Trough":trough_dt.date(),
                                 "Recovery":dt.date(),"Depth":f"{trough_v*100:.1f}%",
                                 "DD Days":(trough_dt-peak_dt).days,
                                 "Rec Days":(dt-trough_dt).days})
                in_dd=False
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def print_all_results(bt_wf, bt_bmark, bt_mv, test_start, initial_investment=100_000):
    print(f"\n{'='*65}\nFULL PERFORMANCE — TEST FROM {test_start.date()}\n{'='*65}")
    summary = pd.DataFrame([
        full_summary(bt_bmark, "Benchmark", initial_investment),
        full_summary(bt_mv,    "MV NoRegime", initial_investment),
        full_summary(bt_wf,    "Walk-Forward", initial_investment),
    ]).T
    print(summary.to_string())
    print(f"\n{'='*65}\nANNUAL COMPARISON\n{'='*65}")
    ann = pd.concat([annual_metrics(bt_bmark),annual_metrics(bt_mv),annual_metrics(bt_wf)],
                    axis=1, keys=["Benchmark","MV","WalkFwd"])
    print(ann.to_string())
    print(f"\n{'='*65}\nREGIME ALPHA TABLE\n{'='*65}")
    print(regime_alpha_table(bt_wf, bt_bmark).T.to_string())
    print(f"\n{'='*65}\nTURNOVER\n{'='*65}")
    print(turnover_table(bt_wf).to_string())


# =============================================================
# PLOTS
# =============================================================

def plot_cumulative(bt_wf, bt_bmark, bt_mv, test_start,
                    switch_threshold=0.75, min_prob_gap=0.15,
                    yield_rise_thresh=0.001, min_hold_bear=10,
                    initial_investment=100_000):
    fig, axes = plt.subplots(3,1,figsize=(14,12),sharex=True,
                              gridspec_kw={"height_ratios":[3,1,1]})
    ax1,ax2,ax3 = axes
    scale = initial_investment

    ax1.plot(bt_bmark.index, bt_bmark["cum"]*scale, label="Benchmark 55/40/5",
             lw=2, color=COLORS["Benchmark"])
    ax1.plot(bt_mv.index,    bt_mv["cum"]*scale,    label="MV No-Regime",
             lw=2, color=COLORS["MV"], linestyle="--")
    ax1.plot(bt_wf.index,    bt_wf["cum"]*scale,    label="Walk-Forward Regime",
             lw=2, color=COLORS["Walk-Fwd"])

    for regime, color in [("Bear","#e74c3c"),("Bull","#2ecc71"),("Transition","#f39c12")]:
        mask = bt_wf["regime"]==regime
        ax1.fill_between(bt_wf.index,
                         bt_wf["cum"].min()*scale*0.97,
                         bt_wf["cum"].max()*scale*1.02,
                         where=mask, alpha=0.08, color=color)
    for dt in bt_wf[bt_wf["rebalanced"]].index:
        ax1.axvline(dt, color="gray", lw=0.5, alpha=0.4)

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax1.set_title(f"Portfolio Value — TEST from {test_start.date()}  "
                  f"(Initial ${initial_investment:,.0f}  thresh={switch_threshold}  "
                  f"gap={min_prob_gap}  Bear_hold={min_hold_bear}d)")
    ax1.set_ylabel("Portfolio Value ($)"); ax1.legend(loc="upper left"); ax1.grid(True,alpha=0.3)

    def rs(bt,w=63):
        r = bt["ret"]
        return (r.rolling(w).mean()*ANN_FACTOR)/(r.rolling(w).std()*np.sqrt(ANN_FACTOR))
    ax2.plot(bt_bmark.index,rs(bt_bmark),color=COLORS["Benchmark"],lw=1,label="Benchmark")
    ax2.plot(bt_mv.index,   rs(bt_mv),   color=COLORS["MV"],lw=1,linestyle="--",label="MV")
    ax2.plot(bt_wf.index,   rs(bt_wf),   color=COLORS["Walk-Fwd"],lw=1,label="WF Regime")
    ax2.axhline(0,color="black",lw=0.8,linestyle="--")
    ax2.set_ylabel("Rolling Sharpe (63d)"); ax2.legend(fontsize=8,loc="upper left")
    ax2.grid(True,alpha=0.3)

    for regime,color in [("Bull","#2ecc71"),("Transition","#f39c12"),("Bear","#e74c3c")]:
        mask = bt_wf["regime"]==regime
        ax3.fill_between(bt_wf.index,0,1,where=mask,alpha=0.75,color=color,label=regime)
    ax3.set_yticks([]); ax3.set_ylabel("Active Regime")
    ax3.legend(loc="upper left",fontsize=8); ax3.set_xlabel("Date"); ax3.grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_drawdown(bt_wf, bt_bmark, bt_mv):
    fig, ax = plt.subplots(figsize=(14,4))
    for bt, lbl, color, ls in [(bt_bmark,"Benchmark",COLORS["Benchmark"],"-"),
                                (bt_mv,"MV No-Regime",COLORS["MV"],"--"),
                                (bt_wf,"Walk-Forward",COLORS["Walk-Fwd"],"-")]:
        dd = (bt["cum"]-bt["cum"].cummax())/bt["cum"].cummax()*100
        ax.plot(bt.index, dd, label=lbl, lw=1.5, color=color, linestyle=ls)
        ax.fill_between(bt.index, dd, 0, alpha=0.07, color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Drawdown (%)"); ax.set_title("Drawdown Comparison")
    ax.legend(loc="lower left"); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_rolling_return(bt_wf, bt_bmark, bt_mv):
    fig, axes = plt.subplots(2,1,figsize=(14,9),sharex=True,
                              gridspec_kw={"height_ratios":[2,1]})
    ax1, ax2 = axes
    for bt, lbl, color, ls in [(bt_bmark,"Benchmark",COLORS["Benchmark"],"-"),
                                (bt_mv,"MV No-Regime",COLORS["MV"],"--"),
                                (bt_wf,"Walk-Forward",COLORS["Walk-Fwd"],"-")]:
        r12 = (1+bt["ret"]).rolling(252).apply(lambda x: x.prod()-1,raw=True)*100
        ax1.plot(bt.index, r12, label=lbl, lw=1.5, color=color, linestyle=ls)
    ax1.axhline(0,color="black",lw=0.8,linestyle="--")
    ax1.set_ylabel("Rolling 12m Return (%)"); ax1.legend(loc="upper left"); ax1.grid(True,alpha=0.3)

    for regime,color in [("Bull","#2ecc71"),("Transition","#f39c12"),("Bear","#e74c3c")]:
        mask = bt_wf["regime"]==regime
        ax2.fill_between(bt_wf.index,0,1,where=mask,alpha=0.75,color=color,label=regime)
    ax2.set_yticks([]); ax2.set_ylabel("Regime")
    ax2.legend(loc="upper left",fontsize=8); ax2.set_xlabel("Date"); ax2.grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_monthly_heatmap(bt_wf):
    r_m  = (1+bt_wf["ret"]).resample("ME").prod()-1
    piv  = r_m.groupby([r_m.index.year,r_m.index.month]).first().unstack()*100
    piv.columns = [calendar.month_abbr[m] for m in piv.columns]
    vmax = max(abs(piv.values[~np.isnan(piv.values)]).max(), 5)
    fig, ax = plt.subplots(figsize=(14,3.5))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=10)
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index,   fontsize=10)
    ax.set_title("Walk-Forward Monthly Returns (%)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=8,
                        color="black" if abs(v)<8 else "white")
    plt.tight_layout(); plt.show()


def plot_turnover(bt_wf, tc_bps=TC_BPS):
    # Daily
    fig, ax = plt.subplots(figsize=(14,3.5))
    ax.bar(bt_wf.index, bt_wf["turnover"], color="steelblue", alpha=0.6, width=1)
    for dt in bt_wf[bt_wf["rebalanced"]].index:
        ax.axvline(dt, color="darkorange", lw=1.5, alpha=0.85)
    ax.set_ylabel("Daily Turnover"); ax.set_title("Daily Turnover  (orange = switch events)")
    ax.grid(True,alpha=0.3); plt.tight_layout(); plt.show()

    # Cumulative
    fig, ax = plt.subplots(figsize=(14,3))
    cum_to = bt_wf["turnover"].cumsum()
    ax.plot(bt_wf.index, cum_to, color="steelblue", lw=2)
    ax.fill_between(bt_wf.index, 0, cum_to, alpha=0.15, color="steelblue")
    total_tc = cum_to.iloc[-1]*tc_bps
    ax.text(bt_wf.index[int(len(bt_wf)*0.55)], cum_to.max()*0.65,
            f"Total TC: {total_tc:.0f} bps  ({total_tc/100:.2f}%)",
            fontsize=11, color="steelblue",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="steelblue",alpha=0.8))
    ax.set_ylabel("Cumulative Turnover"); ax.set_title("Cumulative Turnover")
    ax.grid(True,alpha=0.3); plt.tight_layout(); plt.show()

    # Annual bar
    ato = bt_wf.groupby(bt_wf.index.year)["turnover"].sum()
    asw = bt_wf.groupby(bt_wf.index.year)["rebalanced"].sum()
    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(range(len(ato)), ato.values, color="steelblue", alpha=0.75, width=0.6)
    for bar, sw in zip(bars, asw.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{int(sw)} sw", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(ato))); ax.set_xticklabels([str(y) for y in ato.index])
    ax.set_ylabel("Annual Turnover"); ax.set_xlabel("Year")
    ax.set_title("Annual Turnover with Switch Count")
    ax.grid(True,alpha=0.3,axis="y"); plt.tight_layout(); plt.show()


def plot_annual_metrics(bt_wf, bt_bmark, bt_mv):
    bts   = [("Benchmark",bt_bmark,COLORS["Benchmark"]),
             ("MV",bt_mv,COLORS["MV"]),
             ("Walk-Fwd",bt_wf,COLORS["Walk-Fwd"])]
    years = sorted(bt_wf.index.year.unique())
    metrics= [("Ret","Annual Return (%)","Return",True),
              ("Sharpe","Annual Sharpe","Sharpe",False),
              ("Sortino","Annual Sortino","Sortino",False),
              ("MaxDD","Annual MaxDD (%)","MaxDD",True),
              ("Win%","Annual Win Rate (%)","Win%",False)]
    fig, axes = plt.subplots(2,3,figsize=(18,10))
    fig.suptitle("Annual Performance Metrics", fontsize=14, fontweight="bold")
    for ax,(col,title,_,is_pct) in zip(axes.flatten(),metrics):
        scale=100 if is_pct else 1; w=0.25; x=np.arange(len(years))
        for i,(lbl,bt,color) in enumerate(bts):
            am  = annual_metrics(bt)
            vals= [am.loc[y,col]*scale if y in am.index else 0 for y in years]
            bars= ax.bar(x+i*w, vals, width=w, label=lbl, color=color, alpha=0.85)
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x+w); ax.set_xticklabels([str(y) for y in years], fontsize=9)
        ax.set_title(title,fontsize=10); ax.axhline(0,color="black",lw=0.8)
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3,axis="y")
    axes.flatten()[-1].set_visible(False)
    plt.tight_layout(); plt.show()


def plot_model_comparison(results: dict, initial_investment=100_000):
    """
    results = {"Model Name": bt_wf_test, ...}
    Plots all models on the same cumulative wealth chart.
    """
    colors = ["darkorange","steelblue","green","purple","red","brown"]
    fig, axes = plt.subplots(2,1,figsize=(14,10),sharex=False,
                              gridspec_kw={"height_ratios":[3,1]})
    ax1, ax2 = axes
    for i,(name,bt) in enumerate(results.items()):
        ax1.plot(bt.index, bt["cum"]*initial_investment,
                 label=name, lw=2, color=colors[i%len(colors)])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax1.set_ylabel("Portfolio Value ($)"); ax1.set_title("Model Comparison — All Strategies")
    ax1.legend(loc="upper left"); ax1.grid(True,alpha=0.3)

    # Sharpe comparison bar
    sharpes     = {name: round(sharpe(bt["ret"]),3) for name,bt in results.items()}
    bar_names   = list(sharpes.keys())
    bar_vals    = list(sharpes.values())
    bar_x       = list(range(len(bar_names)))
    ax2.bar(bar_x, bar_vals,
            color=[colors[i%len(colors)] for i in range(len(bar_names))], alpha=0.8)
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(bar_names, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Sharpe Ratio"); ax2.set_title("Sharpe Ratio Comparison")
    ax2.axhline(0,color="black",lw=0.8); ax2.grid(True,alpha=0.3,axis="y")
    for i,v in enumerate(bar_vals):
        ax2.text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout(); plt.show()
