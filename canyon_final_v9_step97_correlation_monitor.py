#!/usr/bin/env python3
"""
Canyon v9 — Step 97: Multi-Strategy Correlation Monitor
========================================================
Tracks the rolling statistical relationship between Canyon v9 paper returns
and major benchmarks / the v11 QQQ-Hunter strategy.

Why this matters:
  If Canyon v9 and QQQ-Hunter become highly correlated (ρ > 0.70), the two
  strategies no longer diversify each other — combined portfolio risk rises
  without a proportional return benefit. Similarly, beta creep toward SPY/QQQ
  signals the portfolio is drifting from its alpha-seeking objective toward
  passive market exposure.

Metrics computed per rolling window (21d / 63d):
  - Pearson correlation vs SPY, QQQ, v11
  - Rolling beta vs SPY (systematic risk exposure)
  - Rolling beta vs QQQ (tech/momentum tilt)
  - Tracking error vs SPY (annualized)
  - Combined portfolio Sharpe (v9 50% + v11 50%)

Outputs:
  correlation_monitor.csv    — one row per date, rolling metrics
  correlation_report.md      — summary with current readings and alerts

Inputs:
  paper_trading_log.csv      — Canyon v9 daily pnl_today column
  paper_trading_history.csv  — v11 strategy signals (used to proxy v11 returns)
  sp500_prices.parquet / sp500_price_cache.csv  — SPY, QQQ prices

Usage:
  python3 canyon_final_v9_step97_correlation_monitor.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
OUT_CSV     = ROOT / "correlation_monitor.csv"
OUT_REPORT  = ROOT / "correlation_report.md"

# Alert thresholds
CORR_WARN   = 0.65   # correlation above this → warn
CORR_ALERT  = 0.80   # correlation above this → alert
BETA_WARN   = 0.80   # beta above this → portfolio behaving like index
WINDOWS     = [21, 63]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_v9_returns() -> pd.Series:
    """Load Canyon v9 daily paper returns from paper_trading_log.csv."""
    p = ROOT / "paper_trading_log.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    if "date" not in df.columns or "pnl_today" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    s = pd.to_numeric(df["pnl_today"], errors="coerce").dropna()
    return s.rename("v9")


def load_benchmark_returns(tickers: list[str], lookback: int = 252) -> pd.DataFrame:
    """Load benchmark prices and compute daily returns (DuckDB → Parquet → CSV)."""
    # Try data layer first
    try:
        from canyon_data_layer import returns as _dl_returns
        rets = _dl_returns(tickers, lookback=lookback + 5)
        if not rets.empty:
            wanted = [t for t in tickers if t in rets.columns]
            if wanted:
                return rets[wanted]
    except Exception:
        pass

    # Fallback: read price CSV directly
    for fname in ("sp500_price_cache.csv", "backtest_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                df = df.sort_index().tail(lookback + 5)
                wanted = [t for t in tickers if t in df.columns]
                if wanted:
                    rets = df[wanted].pct_change(fill_method=None).dropna(how="all")
                    return rets.tail(lookback)
            except Exception:
                pass
    return pd.DataFrame()


def load_v11_returns() -> pd.Series:
    """
    Proxy v11 QQQ-Hunter returns from paper_trading_history.csv.
    We use the tqqq_weight column as a rough proxy: higher TQQQ weight →
    more market beta on up days. Since we don't have actual v11 P&L, we
    load from v11_backtest_monthly.csv if available.
    """
    # Try monthly backtest (proxy)
    p_monthly = ROOT / "v11_backtest_monthly.csv"
    if p_monthly.exists():
        try:
            df = pd.read_csv(p_monthly)
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            ret_col  = next((c for c in df.columns
                             if any(k in c.lower() for k in ("return","ret","pnl","gain"))), None)
            if date_col and ret_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                s = pd.to_numeric(df[ret_col], errors="coerce").dropna()
                return s.rename("v11")
        except Exception:
            pass

    # Fallback: paper_trading_history.csv tqqq_weight as activity proxy
    p = ROOT / "paper_trading_history.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            if date_col and "tqqq_weight" in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                df["tqqq_weight"] = pd.to_numeric(df["tqqq_weight"], errors="coerce")
                # tqqq_weight change as proxy for v11 positioning change
                return df["tqqq_weight"].diff().dropna().rename("v11_proxy")
        except Exception:
            pass
    return pd.Series(dtype=float)


# ── Correlation computation ───────────────────────────────────────────────────

def rolling_stats(
    s1: pd.Series,
    s2: pd.Series,
    window: int,
) -> pd.DataFrame:
    """
    Compute rolling correlation and beta between two return series.
    Beta = Cov(s1, s2) / Var(s2)
    """
    aligned = pd.concat([s1, s2], axis=1).dropna()
    if len(aligned) < window:
        return pd.DataFrame()

    n1, n2 = aligned.columns[0], aligned.columns[1]

    corr = aligned[n1].rolling(window).corr(aligned[n2])
    # Rolling beta via covariance / variance
    cov_  = aligned[n1].rolling(window).cov(aligned[n2])
    var_  = aligned[n2].rolling(window).var()
    beta  = cov_ / (var_ + 1e-12)

    df = pd.DataFrame({
        f"corr_{window}d":  corr.round(4),
        f"beta_{window}d":  beta.round(4),
    }, index=aligned.index)
    return df.dropna()


def tracking_error(
    strategy: pd.Series,
    benchmark: pd.Series,
    window: int = 63,
    ann_factor: int = 252,
) -> pd.Series:
    """Annualized rolling tracking error (std of active returns)."""
    active = strategy - benchmark
    aligned = pd.concat([active], axis=1).dropna()
    te = aligned.iloc[:, 0].rolling(window).std() * np.sqrt(ann_factor)
    return te.round(6)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    print("  [Corr] Loading Canyon v9 paper returns …")
    v9 = load_v9_returns()
    if v9.empty:
        print("  [Corr] No v9 returns found — paper_trading_log.csv missing or no pnl_today")
        return pd.DataFrame()
    print(f"  [Corr] v9: {len(v9)} days  ({v9.index[0].date()} → {v9.index[-1].date()})")

    print("  [Corr] Loading benchmark prices (SPY, QQQ) …")
    bench_rets = load_benchmark_returns(["SPY", "QQQ"], lookback=max(WINDOWS) * 3)
    spy = bench_rets["SPY"] if "SPY" in bench_rets.columns else pd.Series(dtype=float)
    qqq = bench_rets["QQQ"] if "QQQ" in bench_rets.columns else pd.Series(dtype=float)

    print("  [Corr] Loading v11 proxy returns …")
    v11 = load_v11_returns()

    # Build master aligned frame
    frames = {"v9": v9}
    if not spy.empty: frames["SPY"] = spy
    if not qqq.empty: frames["QQQ"] = qqq
    if not v11.empty: frames["v11"] = v11

    master = pd.concat(frames.values(), axis=1, keys=frames.keys()).dropna(how="all")
    master.index = pd.to_datetime(master.index, errors="coerce")
    master = master[master.index.notna()].sort_index()

    if len(master) < min(WINDOWS):
        print(f"  [Corr] Only {len(master)} aligned days — need {min(WINDOWS)} minimum")
        return pd.DataFrame()

    # Compute rolling stats for each window × each benchmark
    result_parts = [master[["v9"]].rename(columns={"v9": "v9_return"})]

    for benchmark in ("SPY", "QQQ", "v11"):
        if benchmark not in master.columns:
            continue
        for w in WINDOWS:
            part = rolling_stats(
                master["v9"].rename("v9"),
                master[benchmark].rename(benchmark),
                window=w,
            )
            if not part.empty:
                result_parts.append(
                    part.rename(columns={
                        f"corr_{w}d": f"corr_vs_{benchmark}_{w}d",
                        f"beta_{w}d":  f"beta_vs_{benchmark}_{w}d",
                    })
                )

    # Tracking error vs SPY
    if not spy.empty:
        v9_aligned = master["v9"].reindex(spy.index).dropna()
        spy_aligned = spy.reindex(v9_aligned.index).dropna()
        common = v9_aligned.index.intersection(spy_aligned.index)
        if len(common) >= 21:
            te = tracking_error(v9_aligned[common], spy_aligned[common], window=21)
            result_parts.append(te.rename("tracking_error_vs_spy_21d").to_frame())

    # Combined v9+v11 Sharpe (equal weight, 63-day rolling)
    if "v11" in master.columns:
        combo = 0.5 * master["v9"] + 0.5 * master["v11"]
        ann   = combo.rolling(63).mean() * 252
        vol   = combo.rolling(63).std() * np.sqrt(252)
        sharpe = (ann / (vol + 1e-9)).round(4)
        result_parts.append(sharpe.rename("combined_sharpe_63d").to_frame())

    final = pd.concat(result_parts, axis=1).sort_index()
    final.index.name = "date"
    final.to_csv(OUT_CSV)
    print(f"  [Corr] Saved {len(final)} rows → {OUT_CSV.name}")
    return final


def write_report(df: pd.DataFrame) -> None:
    if df.empty:
        return

    latest = df.dropna(how="all").iloc[-1]
    date_s = str(df.index[-1])[:10]

    def _fmt(val, pct: bool = False) -> str:
        if pd.isna(val):
            return "—"
        return f"{val:.1%}" if pct else f"{val:.3f}"

    # Current readings
    c21_spy = latest.get("corr_vs_SPY_21d", np.nan)
    c21_qqq = latest.get("corr_vs_QQQ_21d", np.nan)
    c63_spy = latest.get("corr_vs_SPY_63d", np.nan)
    b21_spy = latest.get("beta_vs_SPY_21d", np.nan)
    b21_qqq = latest.get("beta_vs_QQQ_21d", np.nan)
    te_spy  = latest.get("tracking_error_vs_spy_21d", np.nan)
    combo_s = latest.get("combined_sharpe_63d", np.nan)

    # Alerts
    alerts: list[str] = []
    if not np.isnan(c21_spy) and c21_spy > CORR_ALERT:
        alerts.append(f"ALERT: 21d correlation vs SPY = {c21_spy:.2f} — portfolio behaving like index")
    elif not np.isnan(c21_spy) and c21_spy > CORR_WARN:
        alerts.append(f"WARN:  21d correlation vs SPY = {c21_spy:.2f} — diversification eroding")
    if not np.isnan(b21_spy) and b21_spy > BETA_WARN:
        alerts.append(f"WARN:  21d beta vs SPY = {b21_spy:.2f} — high systematic exposure")
    if not np.isnan(c21_qqq) and c21_qqq > CORR_ALERT:
        alerts.append(f"ALERT: 21d correlation vs QQQ = {c21_qqq:.2f} — tech/momentum crowding")

    alert_block = "\n".join(f"  - {a}" for a in alerts) if alerts else "  - No alerts — diversification within normal range"

    report = f"""# Correlation Monitor — {date_s}

## Current Readings

| Metric | 21-day | 63-day |
|--------|:------:|:------:|
| Correlation vs SPY | {_fmt(c21_spy)} | {_fmt(c63_spy)} |
| Correlation vs QQQ | {_fmt(c21_qqq)} | — |
| Beta vs SPY | {_fmt(b21_spy)} | — |
| Beta vs QQQ | {_fmt(b21_qqq)} | — |
| Tracking Error vs SPY (ann.) | {_fmt(te_spy, pct=True)} | — |
| Combined v9+v11 Sharpe | — | {_fmt(combo_s)} |

## Alerts

{alert_block}

## Thresholds

| Level | Correlation | Beta |
|-------|:-----------:|:----:|
| Watch | > {CORR_WARN:.0%} | > {BETA_WARN:.0%} |
| Alert | > {CORR_ALERT:.0%} | — |

---
*Signal: Canyon v9 paper returns vs SPY/QQQ benchmarks.
Higher correlation = less unique alpha, more market beta exposure.*
"""
    OUT_REPORT.write_text(report)
    print(f"  [Corr] Report saved → {OUT_REPORT.name}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"Canyon v9 — Correlation Monitor  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60 + "\n")

    df = run()
    if not df.empty:
        write_report(df)
        latest = df.dropna(how="all").iloc[-1]
        print("\nLatest readings:")
        for col in df.columns:
            v = latest.get(col, np.nan)
            if not np.isnan(v):
                print(f"  {col}: {v:.4f}")
    else:
        print("  Not enough data yet — run daily pipeline first")

    print("\n" + "=" * 60)
    print("Step 97 Complete")
    print("=" * 60)
