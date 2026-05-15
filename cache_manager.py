# =============================================================
# cache_manager.py — Cache management for dashboard
# Saves and loads backtest results with unique keys
# =============================================================

import os, json, pickle, hashlib, time
from datetime import datetime
import pandas as pd

# Always point cache to the project-level /cache folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CACHE_DIR = os.path.join(BASE_DIR, "cache")
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "cache_index.json")



def _make_cache_key(model, risk_profile, start_date, end_date,
                    tc_bps, switch_threshold, refit_months,
                    eq_pct, bond_pct, alt_pct) -> str:
    """Create unique cache key from all params that affect results."""
    raw = (f"{model}_{risk_profile}_{start_date}_{end_date}_"
           f"{tc_bps}bps_{switch_threshold}thresh_{refit_months}m_"
           f"eq{eq_pct}_bd{bond_pct}_alt{alt_pct}")
    # Shorten with hash for safety
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    safe = raw.replace(" ","").replace("/","").replace("\\","")[:60]
    return f"{safe}_{h}"


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(CACHE_INDEX_FILE):
        with open(CACHE_INDEX_FILE, "w") as f:
            json.dump([], f)


def load_cache_index() -> list:
    _ensure_cache_dir()
    try:
        with open(CACHE_INDEX_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_cache_index(index: list):
    with open(CACHE_INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2, default=str)


def cache_exists(key: str) -> bool:
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    return os.path.exists(path)


def load_cache(key: str) -> dict:
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cache(key: str, data: dict, metadata: dict):
    _ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)

    # Update index
    index = load_cache_index()
    # Remove old entry if exists
    index = [e for e in index if e.get("key") != key]
    index.append({
        "key":          key,
        "model":        metadata.get("model",""),
        "risk_profile": metadata.get("risk_profile",""),
        "start_date":   str(metadata.get("start_date","")),
        "end_date":     str(metadata.get("end_date","")),
        "tc_bps":       metadata.get("tc_bps",5),
        "switch_threshold": metadata.get("switch_threshold",0.75),
        "refit_months": metadata.get("refit_months",12),
        "eq_pct":       metadata.get("eq_pct",55),
        "bond_pct":     metadata.get("bond_pct",40),
        "alt_pct":      metadata.get("alt_pct",5),
        "sharpe":       round(metadata.get("sharpe",0),3),
        "ann_return":   metadata.get("ann_return",""),
        "max_dd":       metadata.get("max_dd",""),
        "switches":     metadata.get("switches",0),
        "runtime_min":  round(metadata.get("runtime_min",0),1),
        "run_date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "file_size_mb": round(os.path.getsize(path)/1e6, 1),
    })
    _save_cache_index(index)
    print(f"[Cache]  Saved → {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def delete_cache(key: str):
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(path):
        os.remove(path)
    index = load_cache_index()
    index = [e for e in index if e.get("key") != key]
    _save_cache_index(index)
    print(f"[Cache]  Deleted → {key}")


def get_or_run(
    model_name, risk_profile, start_date, end_date,
    tc_bps, switch_threshold, refit_months,
    eq_pct, bond_pct, alt_pct,
    run_fn,               # callable() → returns result dict
    progress_callback=None,
) -> tuple[dict, bool]:
    """
    Returns (result_dict, from_cache).
    If cache hit: load instantly.
    If miss: run model, save to cache, return result.
    """
    key = _make_cache_key(model_name, risk_profile, start_date, end_date,
                          tc_bps, switch_threshold, refit_months,
                          eq_pct, bond_pct, alt_pct)

    if cache_exists(key):
        print(f"[Cache]  HIT → loading {key}")
        return load_cache(key), True

    print(f"[Cache]  MISS → running {model_name}")
    t0 = time.time()
    result = run_fn()
    runtime = (time.time() - t0) / 60

    # Compute summary stats for index
    bt_wf = result.get("bt_wf_test")
    sharpe_val = 0.0; ann_ret = ""; max_dd = ""; switches = 0
    if bt_wf is not None:
        from src.backtest.backtest_engine import sharpe, ann_return, max_drawdown
        r = bt_wf["ret"]
        sharpe_val = sharpe(r)
        ann_ret    = f"{ann_return(r)*100:.1f}%"
        max_dd     = f"{max_drawdown(bt_wf['cum'])*100:.1f}%"
        switches   = int(bt_wf.get("rebalanced", pd.Series()).sum())

    metadata = dict(
        model=model_name, risk_profile=risk_profile,
        start_date=start_date, end_date=end_date,
        tc_bps=tc_bps, switch_threshold=switch_threshold,
        refit_months=refit_months,
        eq_pct=eq_pct, bond_pct=bond_pct, alt_pct=alt_pct,
        sharpe=sharpe_val, ann_return=ann_ret,
        max_dd=max_dd, switches=switches, runtime_min=runtime,
    )
    save_cache(key, result, metadata)
    return result, False
