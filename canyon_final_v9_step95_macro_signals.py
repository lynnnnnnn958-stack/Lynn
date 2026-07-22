#!/usr/bin/env python3
"""
Canyon v9 — Step 95: Macro Signals Layer
=========================================
Fetches macro market data and computes a set of regime indicators that
characterise the current macro environment: yield curve shape, VIX term
structure, credit spread direction, USD trend, gold vs equities signal,
and a composite macro score.

Data sources
------------
All data is pulled from yfinance (free, no API key needed):
  ^VIX    — CBOE VIX spot
  ^VIX3M  — 3-month VIX (fallback: VIXY ETF)
  ^TNX    — 10Y Treasury yield (%)
  ^IRX    — 13-week T-bill yield (%)
  LQD     — IG corporate bond ETF
  HYG     — HY corporate bond ETF
  UUP     — USD bullish ETF (DXY proxy)
  SPY     — S&P 500 ETF
  TLT     — 20Y Treasury ETF
  GLD     — Gold ETF

Outputs
-------
  macro_signals.json    — latest snapshot (overwritten each run)
  macro_signals.csv     — historical log (one row appended per run)

Usage
-----
  python3 canyon_final_v9_step95_macro_signals.py
  python3 canyon_final_v9_step95_macro_signals.py --dry-run
  python3 canyon_final_v9_step95_macro_signals.py --history 504
"""

import argparse
import json
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

OUT_JSON = ROOT / "macro_signals.json"
OUT_CSV  = ROOT / "macro_signals.csv"

# ── Ticker universe ──────────────────────────────────────────────────────────

MACRO_TICKERS = {
    "vix":   "^VIX",
    "vix3m": "^VIX3M",
    "tny10": "^TNX",
    "tny3m": "^IRX",
    "lqd":   "LQD",
    "hyg":   "HYG",
    "uup":   "UUP",
    "spy":   "SPY",
    "tlt":   "TLT",
    "gld":   "GLD",
    "iwm":  "IWM",
    "eem":  "EEM",
    "qqq":  "QQQ",
    "tip":  "TIP",
    "rsp":  "RSP",
}

VIX3M_FALLBACK = "VIXY"   # ETF fallback when ^VIX3M is unavailable

LOOKBACK_DAYS  = 252       # calendar days to download
RETURN_WINDOW  = 20       # trading days for 20-day return
MAX_RETRIES    = 2
RETRY_SLEEP    = 5        # seconds between retries


# ─────────────────────────────────────────────────────────────
# 1.  DATA FETCH
# ─────────────────────────────────────────────────────────────

def _download_single(yf_symbol: str, period: str = f"{LOOKBACK_DAYS}d") -> pd.Series:
    """
    Download Close prices for one symbol.  Returns a pd.Series indexed by
    Date, or an empty Series on failure.  Retries up to MAX_RETRIES times.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                yf_symbol,
                period=period,
                auto_adjust=True,
                progress=False,
            )
            if raw is None or raw.empty:
                raise ValueError("empty result")
            # Flatten MultiIndex if present (yfinance batch download quirk)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].squeeze()
            close.index = pd.to_datetime(close.index)
            close.index.name = "Date"
            close = close.dropna()
            if close.empty:
                raise ValueError("all-NaN close")
            return close
        except Exception as exc:
            if attempt < MAX_RETRIES:
                print(f"    [retry {attempt}/{MAX_RETRIES}] {yf_symbol}: {exc}")
                time.sleep(RETRY_SLEEP)
            else:
                print(f"    [WARN] {yf_symbol} unavailable after {MAX_RETRIES} tries: {exc}")
                return pd.Series(dtype=float, name=yf_symbol)


def fetch_all_data() -> dict[str, pd.Series]:
    """
    Fetch 60 days of Close data for every ticker in MACRO_TICKERS.
    Returns a dict keyed by the logical name (e.g. 'vix', 'spy', …).
    Uses VIXY as fallback for vix3m if ^VIX3M fails or returns no data.
    """
    print("  Fetching macro data from yfinance …")
    data: dict[str, pd.Series] = {}

    for name, symbol in MACRO_TICKERS.items():
        print(f"    {name:8s}  {symbol}")
        s = _download_single(symbol)

        # VIX3M fallback
        if name == "vix3m" and s.empty:
            print(f"    [WARN] ^VIX3M failed — trying VIXY fallback")
            s = _download_single(VIX3M_FALLBACK)
            if not s.empty:
                print("    [INFO] Using VIXY as vix3m proxy")

        data[name] = s

    return data


# ─────────────────────────────────────────────────────────────
# 2.  20-DAY RETURN HELPERS
# ─────────────────────────────────────────────────────────────

def last_value(series: pd.Series) -> float | None:
    """Return the most recent non-NaN scalar, or None if series is empty."""
    if series is None or series.empty:
        return None
    val = series.dropna().iloc[-1]
    return float(val)


def return_20d(series: pd.Series) -> float | None:
    """
    Compute the 20-trading-day return from the Close series.
    Returns None when fewer than 21 observations are available.
    """
    clean = series.dropna()
    if len(clean) < RETURN_WINDOW + 1:
        return None
    end   = float(clean.iloc[-1])
    start = float(clean.iloc[-(RETURN_WINDOW + 1)])
    if start == 0:
        return None
    return (end - start) / start


# ─────────────────────────────────────────────────────────────
# 3.  INDICATOR FUNCTIONS
# ─────────────────────────────────────────────────────────────

def yield_curve(tny10_last: float, tny3m_last: float) -> tuple[float, str]:
    """
    Yield curve shape: spread = 10Y minus 3M (in percentage points).
    > 1.0  → STEEP
    0 – 1  → FLAT
    < 0    → INVERTED
    """
    spread = tny10_last - tny3m_last
    if spread > 1.0:
        label = "STEEP"
    elif spread >= 0.0:
        label = "FLAT"
    else:
        label = "INVERTED"
    return round(spread, 4), label


def vix_term_structure(vix_spot: float, vix3m: float) -> tuple[float, str]:
    """
    VIX term-structure ratio: vix3m / vix_spot.
    > 1.10  → CONTANGO  (calm, normal backwardation in futures)
    0.90 – 1.10 → FLAT
    < 0.90  → BACKWARDATION (fear, short-term vol bid up)
    """
    if vix_spot == 0:
        return 1.0, "FLAT"
    ratio = vix3m / vix_spot
    if ratio > 1.10:
        label = "CONTANGO"
    elif ratio >= 0.90:
        label = "FLAT"
    else:
        label = "BACKWARDATION"
    return round(ratio, 4), label


def credit_spread_signal(lqd_20d_ret: float, hyg_20d_ret: float) -> tuple[float, str]:
    """
    Credit spread direction: HYG minus LQD 20-day return.
    HY outperforming IG → spreads compressing → risk-on.
    > +0.5%  → RISK_ON
    -0.5% – +0.5% → NEUTRAL
    < -0.5%  → RISK_OFF
    """
    spread_ret = hyg_20d_ret - lqd_20d_ret
    if spread_ret > 0.005:
        label = "RISK_ON"
    elif spread_ret >= -0.005:
        label = "NEUTRAL"
    else:
        label = "RISK_OFF"
    return round(spread_ret, 6), label


def dxy_trend(uup_20d_ret: float) -> tuple[float, str]:
    """
    USD trend based on UUP 20-day return.
    > +1%   → STRONG_USD
    -1% – +1% → NEUTRAL
    < -1%   → WEAK_USD
    """
    if uup_20d_ret > 0.01:
        label = "STRONG_USD"
    elif uup_20d_ret >= -0.01:
        label = "NEUTRAL"
    else:
        label = "WEAK_USD"
    return round(uup_20d_ret, 6), label


def gold_signal(gld_20d_ret: float, spy_20d_ret: float) -> tuple[float, str]:
    """
    Gold vs SPY 20-day return comparison.
    Gold outperforming SPY by > 2%  → RISK_OFF
    Gold lagging SPY by > 2%        → RISK_ON
    Otherwise                       → NEUTRAL
    """
    diff = gld_20d_ret - spy_20d_ret
    if diff > 0.02:
        label = "RISK_OFF"
    elif diff < -0.02:
        label = "RISK_ON"
    else:
        label = "NEUTRAL"
    return round(diff, 6), label


def sma_cross_signal(spy_series: pd.Series) -> tuple[float, str]:
    clean = spy_series.dropna()
    if len(clean) < 200:
        return 1.0, "INSUFFICIENT_DATA"
    sma50  = float(clean.rolling(50).mean().iloc[-1])
    sma200 = float(clean.rolling(200).mean().iloc[-1])
    if sma200 == 0:
        return 1.0, "NEUTRAL"
    ratio = sma50 / sma200
    if ratio > 1.01:
        label = "GOLDEN"
    elif ratio < 0.99:
        label = "DEATH"
    else:
        label = "NEUTRAL"
    return round(ratio, 4), label


def vix_regime_signal(vix_series: pd.Series) -> tuple[float, str]:
    clean = vix_series.dropna()
    if len(clean) < 30:
        return 1.0, "NORMAL_VOL"
    vix_now = float(clean.iloc[-1])
    window  = min(90, len(clean))
    vix_avg = float(clean.rolling(window).mean().iloc[-1])
    if vix_avg == 0:
        return 1.0, "NORMAL_VOL"
    ratio = vix_now / vix_avg
    if ratio > 1.3:
        label = "ELEVATED_VOL"
    elif ratio < 0.7:
        label = "SUPPRESSED_VOL"
    else:
        label = "NORMAL_VOL"
    return round(ratio, 4), label


def small_large_signal(iwm_20d: float, spy_20d: float) -> tuple[float, str]:
    diff = iwm_20d - spy_20d
    if diff > 0.02:
        label = "SMALL_LEADS"
    elif diff < -0.02:
        label = "LARGE_LEADS"
    else:
        label = "NEUTRAL"
    return round(diff, 6), label


def em_dm_signal(eem_20d: float, spy_20d: float) -> tuple[float, str]:
    diff = eem_20d - spy_20d
    if diff > 0.02:
        label = "EM_RISK_ON"
    elif diff < -0.02:
        label = "EM_RISK_OFF"
    else:
        label = "NEUTRAL"
    return round(diff, 6), label


def growth_value_signal(qqq_20d: float, spy_20d: float) -> tuple[float, str]:
    diff = qqq_20d - spy_20d
    if diff > 0.02:
        label = "GROWTH_LEADS"
    elif diff < -0.02:
        label = "VALUE_LEADS"
    else:
        label = "NEUTRAL"
    return round(diff, 6), label


def breadth_signal(rsp_20d: float, spy_20d: float) -> tuple[float, str]:
    diff = rsp_20d - spy_20d
    if diff > 0.01:
        label = "BROAD_RALLY"
    elif diff < -0.01:
        label = "NARROW_RALLY"
    else:
        label = "NEUTRAL"
    return round(diff, 6), label


def hmm_regime_2state(macro_csv_path: "Path") -> tuple[str, float]:
    """
    L2 institutional: 2-state HMM regime from macro signal history.
    States: RISK_ON / RISK_OFF.
    Uses macro_score history as the observation sequence.
    Returns (regime_label, transition_probability_to_opposite_state).

    Method:
      1. Try hmmlearn GaussianHMM (2 states, 50 iter)
      2. Fallback: Gaussian mixture threshold on macro_score rolling z-score
    """
    if not macro_csv_path.exists():
        return "NEUTRAL", 0.5

    try:
        hist = pd.read_csv(macro_csv_path, parse_dates=["date"])
        if "macro_score" not in hist.columns or len(hist) < 20:
            return "NEUTRAL", 0.5

        hist = hist.sort_values("date").dropna(subset=["macro_score"])
        scores = hist["macro_score"].values.reshape(-1, 1).astype(float)

        try:
            from hmmlearn import hmm as _hmm
            model = _hmm.GaussianHMM(n_components=2, covariance_type="full",
                                      n_iter=50, random_state=42)
            model.fit(scores)
            states = model.predict(scores)
            # State with higher mean = RISK_ON
            mean0 = float(scores[states == 0].mean()) if (states == 0).any() else 50.0
            mean1 = float(scores[states == 1].mean()) if (states == 1).any() else 50.0
            risk_on_state = 0 if mean0 > mean1 else 1
            current_state = int(states[-1])
            is_risk_on = (current_state == risk_on_state)
            # Transition probability from current state to opposite
            trans_prob = float(model.transmat_[current_state, 1 - current_state])
            return ("RISK_ON" if is_risk_on else "RISK_OFF"), round(trans_prob, 4)

        except ImportError:
            # Fallback: rolling z-score threshold on macro_score
            mu  = float(np.mean(scores))
            sig = float(np.std(scores)) + 1e-6
            z   = (scores[-1, 0] - mu) / sig
            # Rolling 10-day mean trend (regime persistence)
            if len(scores) >= 10:
                trend = float(scores[-1, 0]) - float(scores[-10, 0])
            else:
                trend = 0.0
            current_score = float(scores[-1, 0])
            # Transition probability estimate: distance from threshold
            dist_from_threshold = abs(current_score - mu) / sig
            trans_prob = float(np.exp(-dist_from_threshold * 0.7))   # higher distance = less likely to flip
            trans_prob = max(0.02, min(0.48, trans_prob))
            return ("RISK_ON" if current_score > mu else "RISK_OFF"), round(trans_prob, 4)

    except Exception:
        return "NEUTRAL", 0.5


def macro_composite_score(
    yield_curve_val: float,
    vts_ratio: float,
    credit_spread_val: float,
    dxy_ret: float,
    vix_level: float,
    sma_ratio: float = 1.0,
    vix_regime_ratio: float = 1.0,
    small_large_diff: float = 0.0,
    em_diff: float = 0.0,
    breadth_diff: float = 0.0,
) -> tuple[float, str]:
    """Composite macro score 0-100. Higher = more risk-on. Now incorporates 10 signals."""
    score = 0.0

    if yield_curve_val > 1.0:
        score += 12
    elif yield_curve_val >= 0.0:
        score += 8

    if vts_ratio > 1.05:
        score += 16
    elif vts_ratio >= 0.95:
        score += 10
    else:
        score += 3

    if credit_spread_val > 0.0:
        score += 16
    elif credit_spread_val > -0.005:
        score += 8
    else:
        score += 3

    if vix_level < 15:
        score += 20
    elif vix_level < 20:
        score += 16
    elif vix_level < 25:
        score += 10
    elif vix_level < 35:
        score += 4

    if dxy_ret < -0.01:
        score += 8
    elif dxy_ret <= 0.01:
        score += 6
    else:
        score += 3

    if sma_ratio > 1.01:
        score += 6
    elif sma_ratio >= 0.99:
        score += 3

    if vix_regime_ratio < 0.7:
        score += 8
    elif vix_regime_ratio <= 1.3:
        score += 4

    if small_large_diff > 0.02:
        score += 8
    elif small_large_diff >= -0.02:
        score += 4

    if breadth_diff > 0.01:
        score += 6
    elif breadth_diff >= -0.01:
        score += 3

    if em_diff > 0.02:
        score += 4
    elif em_diff >= -0.02:
        score += 2

    score = min(score, 100.0)

    if score > 65:
        signal = "RISK_ON"
    elif score >= 40:
        signal = "NEUTRAL"
    else:
        signal = "RISK_OFF"

    return round(score, 1), signal


# ─────────────────────────────────────────────────────────────
# 4.  MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def run_macro_signals(history_days: int = 252, dry_run: bool = False) -> dict:
    """
    Orchestrate the full macro signal computation pipeline.
    Returns the snapshot dict (also written to JSON/CSV unless dry_run).
    """
    print("\n" + "=" * 62)
    print("  Canyon v9 — Step 95  Macro Signals Layer")
    print("=" * 62)

    # ── 1. Fetch ────────────────────────────────────────────────
    data = fetch_all_data()

    # ── 2. 20-day returns for price instruments ─────────────────
    ret: dict[str, float | None] = {}
    for key in ("lqd", "hyg", "uup", "spy", "gld", "tlt", "iwm", "eem", "qqq", "tip", "rsp"):
        ret[key] = return_20d(data.get(key, pd.Series(dtype=float)))

    # ── 3. Yield levels ─────────────────────────────────────────
    yield_10y = last_value(data["tny10"])
    yield_3m  = last_value(data["tny3m"])

    # TNX and IRX are quoted in percentage points — no scaling needed.
    # yfinance sometimes returns IRX already in %; treat as-is.

    # ── 4. VIX levels ───────────────────────────────────────────
    vix_spot  = last_value(data["vix"])
    vix3m_val = last_value(data["vix3m"])

    today_str = date.today().isoformat()

    # ── 5. Guard: warn if any critical value is missing ─────────
    def _safe(val: float | None, fallback: float, label: str) -> float:
        if val is None:
            print(f"    [WARN] {label} unavailable — using fallback {fallback}")
            return fallback
        return val

    vix_spot  = _safe(vix_spot,  20.0, "VIX spot")
    vix3m_val = _safe(vix3m_val, vix_spot, "VIX3M")
    yield_10y = _safe(yield_10y, 4.0,  "10Y yield")
    yield_3m  = _safe(yield_3m,  5.0,  "3M yield")

    for key, fallback in [("lqd", 0.0), ("hyg", 0.0), ("uup", 0.0),
                          ("spy", 0.0), ("gld", 0.0), ("tlt", 0.0)]:
        if ret[key] is None:
            print(f"    [WARN] {key.upper()} 20-day return unavailable — using 0.0")
            ret[key] = 0.0

    # ── 6. Compute indicators ───────────────────────────────────
    yc_val,  yc_sig  = yield_curve(yield_10y, yield_3m)
    vts_val, vts_sig = vix_term_structure(vix_spot, vix3m_val)
    cs_val,  cs_sig  = credit_spread_signal(ret["lqd"], ret["hyg"])
    dxy_val, dxy_sig = dxy_trend(ret["uup"])
    gd_diff, gd_sig  = gold_signal(ret["gld"], ret["spy"])

    # New cross-asset indicators
    sma_ratio,     sma_sig     = sma_cross_signal(data.get("spy", pd.Series(dtype=float)))
    vix_reg_ratio, vix_reg_sig = vix_regime_signal(data.get("vix", pd.Series(dtype=float)))
    sl_diff,       sl_sig      = small_large_signal(ret.get("iwm") or 0.0, ret.get("spy") or 0.0)
    em_d,          em_sig_val  = em_dm_signal(ret.get("eem") or 0.0, ret.get("spy") or 0.0)
    gv_d,          gv_sig      = growth_value_signal(ret.get("qqq") or 0.0, ret.get("spy") or 0.0)
    br_d,          br_sig      = breadth_signal(ret.get("rsp") or 0.0, ret.get("spy") or 0.0)

    # ── 7. Composite score ──────────────────────────────────────
    macro_score, macro_sig = macro_composite_score(
        yc_val, vts_val, cs_val, ret.get("uup") or 0.0, vix_spot,
        sma_ratio=sma_ratio,
        vix_regime_ratio=vix_reg_ratio,
        small_large_diff=sl_diff,
        em_diff=em_d,
        breadth_diff=br_d,
    )

    # ── 7b. L2 institutional: HMM 2-state regime ────────────────
    hmm_regime, hmm_trans_prob = hmm_regime_2state(OUT_CSV)

    # ── 8. Build snapshot ───────────────────────────────────────
    snapshot: dict = {
        "date":              today_str,
        "vix":               round(vix_spot, 2),
        "vix3m":             round(vix3m_val, 2),
        "vix_term_structure": vts_val,
        "vts_signal":        vts_sig,
        "yield_10y":         round(yield_10y, 4),
        "yield_3m":          round(yield_3m, 4),
        "yield_curve":       yc_val,
        "yield_curve_signal": yc_sig,
        "credit_spread_ret": cs_val,
        "credit_signal":     cs_sig,
        "dxy_20d_ret":       dxy_val,
        "dxy_signal":        dxy_sig,
        "gold_vs_spy_ret":   gd_diff,
        "gold_signal":       gd_sig,
        "macro_score":       macro_score,
        "macro_signal":      macro_sig,
        "spy_20d_ret":       round(ret.get("spy") or 0.0, 6),
        "tlt_20d_ret":       round(ret.get("tlt") or 0.0, 6),
        "sma_cross_ratio":   sma_ratio,
        "sma_cross_signal":  sma_sig,
        "vix_regime_ratio":  vix_reg_ratio,
        "vix_regime_signal": vix_reg_sig,
        "small_large_diff":  sl_diff,
        "small_large_signal": sl_sig,
        "em_dm_diff":        em_d,
        "em_dm_signal":      em_sig_val,
        "growth_value_diff": gv_d,
        "growth_value_signal": gv_sig,
        "breadth_diff":      br_d,
        "breadth_signal":    br_sig,
        "iwm_20d_ret":       round(ret.get("iwm") or 0.0, 6),
        "qqq_20d_ret":       round(ret.get("qqq") or 0.0, 6),
        "eem_20d_ret":       round(ret.get("eem") or 0.0, 6),
        "tip_20d_ret":       round(ret.get("tip") or 0.0, 6),
        "rsp_20d_ret":       round(ret.get("rsp") or 0.0, 6),
        # L2 institutional: HMM regime
        "hmm_regime":            hmm_regime,
        "hmm_transition_prob":   hmm_trans_prob,
    }

    # ── 9. Print summary table ──────────────────────────────────
    _print_summary(snapshot)

    if dry_run:
        print("\n  [DRY-RUN] No files written.")
        return snapshot

    # ── 10. Write JSON (latest snapshot) ────────────────────────
    OUT_JSON.write_text(json.dumps(snapshot, indent=2))
    print(f"\n  Wrote: {OUT_JSON}")

    # ── 11. Append to CSV (history) ─────────────────────────────
    _append_csv(snapshot, history_days)
    print(f"  Wrote: {OUT_CSV}  (history_days={history_days})")

    print("\n  Done.\n")
    return snapshot


# ─────────────────────────────────────────────────────────────
# 5.  CSV HELPERS
# ─────────────────────────────────────────────────────────────

def _append_csv(snapshot: dict, history_days: int) -> None:
    """Append snapshot to CSV; prune rows older than history_days."""
    row_df = pd.DataFrame([snapshot])
    row_df["date"] = pd.to_datetime(row_df["date"])

    if OUT_CSV.exists():
        try:
            hist = pd.read_csv(OUT_CSV, parse_dates=["date"])
        except Exception:
            hist = pd.DataFrame()
    else:
        hist = pd.DataFrame()

    # Drop today's row if already present (idempotent runs)
    if not hist.empty and "date" in hist.columns:
        today_ts = pd.Timestamp(date.today())
        hist = hist[hist["date"].dt.normalize() != today_ts]

    combined = pd.concat([hist, row_df], ignore_index=True)

    # Prune to history_days
    if "date" in combined.columns:
        combined = combined.sort_values("date")
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=history_days)
        combined = combined[combined["date"] >= cutoff]

    combined.to_csv(OUT_CSV, index=False)


# ─────────────────────────────────────────────────────────────
# 6.  PRINT SUMMARY
# ─────────────────────────────────────────────────────────────

_SIGNAL_WIDTH = 14

def _fmt_signal(label: str) -> str:
    """Right-pad signal label to fixed width."""
    return label.ljust(_SIGNAL_WIDTH)


def _print_summary(s: dict) -> None:
    """Print a clean formatted table of macro indicators."""
    print()
    print("  ┌" + "─" * 58 + "┐")
    print("  │  Canyon v9 — Macro Signals Summary" + " " * 22 + "│")
    print("  │  Date: " + s["date"] + " " * 48 + "│")
    print("  ├" + "─" * 58 + "┤")
    print(f"  │  {'Indicator':<28}  {'Value':>8}  {'Signal':<14}│")
    print("  ├" + "─" * 58 + "┤")

    rows = [
        ("VIX Spot",            f"{s['vix']:>8.2f}",   ""),
        ("VIX 3M",              f"{s['vix3m']:>8.2f}",  ""),
        ("VIX Term Structure",  f"{s['vix_term_structure']:>8.3f}", s["vts_signal"]),
        ("10Y Yield (%)",       f"{s['yield_10y']:>8.3f}", ""),
        ("3M Yield (%)",        f"{s['yield_3m']:>8.3f}",  ""),
        ("Yield Curve (10Y-3M)",f"{s['yield_curve']:>8.3f}", s["yield_curve_signal"]),
        ("Credit Spread Ret",   f"{s['credit_spread_ret']*100:>7.3f}%", s["credit_signal"]),
        ("DXY 20d Return",      f"{s['dxy_20d_ret']*100:>7.3f}%", s["dxy_signal"]),
        ("Gold vs SPY 20d",     f"{s['gold_vs_spy_ret']*100:>7.3f}%", s["gold_signal"]),
        ("SPY 20d Return",      f"{s['spy_20d_ret']*100:>7.3f}%", ""),
        ("TLT 20d Return",      f"{s['tlt_20d_ret']*100:>7.3f}%", ""),
        ("IWM 20d Return",    f"{s.get('iwm_20d_ret', 0)*100:>7.3f}%", s.get("small_large_signal", "")),
        ("QQQ 20d Return",    f"{s.get('qqq_20d_ret', 0)*100:>7.3f}%", s.get("growth_value_signal", "")),
        ("EEM 20d Return",    f"{s.get('eem_20d_ret', 0)*100:>7.3f}%", s.get("em_dm_signal", "")),
        ("RSP vs SPY Breadth",f"{s.get('breadth_diff', 0)*100:>7.3f}%", s.get("breadth_signal", "")),
        ("SPY SMA 50/200",    f"{s.get('sma_cross_ratio', 1.0):>8.3f}", s.get("sma_cross_signal", "")),
        ("VIX vs 90d Avg",    f"{s.get('vix_regime_ratio', 1.0):>8.3f}", s.get("vix_regime_signal", "")),
    ]
    for name, val, sig in rows:
        print(f"  │  {name:<28}  {val:>8}  {_fmt_signal(sig)}│")

    print("  ├" + "─" * 58 + "┤")
    score_bar = _score_bar(s["macro_score"])
    print(f"  │  {'Macro Score':<28}  {s['macro_score']:>7.1f}   {_fmt_signal(s['macro_signal'])}│")
    print(f"  │  {score_bar:<56}  │")
    print("  └" + "─" * 58 + "┘")


def _score_bar(score: float, width: int = 40) -> str:
    """ASCII progress bar for composite score (0-100)."""
    filled = int(round(score / 100 * width))
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.0f}/100"


# ─────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canyon v9 Step 95 — Macro Signals Layer"
    )
    p.add_argument(
        "--history",
        type=int,
        default=252,
        metavar="N",
        help="Number of calendar days to retain in macro_signals.csv (default: 252)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals but do not write any output files",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_macro_signals(history_days=args.history, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
