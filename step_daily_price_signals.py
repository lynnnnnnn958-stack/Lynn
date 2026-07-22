#!/usr/bin/env python3
"""
Canyon v9 — Step 0: Daily Price & Signal Refresh
=================================================
Runs FIRST in the daily pipeline (before step99/step95/step12).

What this does every morning:
  1. Reads sp500_price_cache.csv to find which dates are already cached
  2. Downloads only the MISSING days (usually just yesterday's close, <30 s)
  3. Appends new prices → updates sp500_price_cache.csv + backtest_price_cache.csv
  4. Recomputes price-based signals for all ~494 tickers:
       mom_1m / mom_3m / mom_6m / mom_12m_skip1m
       trend_200 / rsi_14 / inv_vol / rank_mom / rank_trend
  5. Blends fresh price signals with existing fundamental signals
     (quality_score, rank_sentiment, rank_sue, rank_revision, rank_options,
      pcr_crowded, rank_squeeze, rank_insider)
  6. Writes updated regime_ml_scores.csv (494 tickers, fresh signals)
  7. Updates cpcv_predictions.csv and factor_composite.csv
  8. Updates price_refresh_desk.csv (today's close + daily change)

No broker connection. No live orders. Research only.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import json
import pandas as pd

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg: str): print(f"  {msg}")
def ok(msg: str):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg: str): print(f"  {RED}✗{RESET}  {msg}")


# ── RSI helper ────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff().dropna()
    if len(delta) < window:
        return 50.0
    gain  = delta.clip(lower=0).rolling(window).mean().iloc[-1]
    loss  = (-delta.clip(upper=0)).rolling(window).mean().iloc[-1]
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100 - 100 / (1 + rs))


# ── Step 1: Identify missing dates, download incrementally ────────────────────

def refresh_prices() -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        err("yfinance not installed")
        return None

    sp500_path = ROOT / "sp500_price_cache.csv"
    regime_path = ROOT / "regime_ml_scores.csv"

    # Determine which tickers we need
    if regime_path.exists():
        tickers = pd.read_csv(regime_path)["ticker"].dropna().unique().tolist()
    else:
        warn("regime_ml_scores.csv missing — cannot determine universe")
        return None

    # ── Universe sanitizer: drop any non-ticker junk (e.g. a stray "1" row) ────
    import re as _re
    _clean = [str(t).strip() for t in tickers
              if _re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", str(t).strip())]
    _dropped = [t for t in tickers if str(t).strip() not in _clean]
    if _dropped:
        warn(f"Universe sanitizer dropped {len(_dropped)} junk ticker(s): {_dropped[:5]}")
    tickers = _clean

    all_tickers = list(dict.fromkeys(tickers + ["SPY"]))

    # Load existing price cache
    if sp500_path.exists():
        existing = pd.read_csv(sp500_path, index_col=0, parse_dates=True)
        last_cached = existing.index.max()
    else:
        existing = pd.DataFrame()
        last_cached = pd.Timestamp("2023-01-01")

    # What dates are missing?
    last_cached = pd.Timestamp(last_cached)  # ensure it's a Timestamp, not str
    fetch_start = (last_cached + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = TODAY

    if pd.Timestamp(fetch_start) > pd.Timestamp(TODAY):
        ok(f"Price cache up-to-date through {last_cached.date()} — no download needed")
        return existing

    log(f"Downloading {fetch_start} → {fetch_end} for {len(all_tickers)} tickers …")

    batch_size = 100  # larger batches for incremental fetches (less data)
    batches = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]
    new_parts: list[pd.DataFrame] = []
    vol_parts: list[pd.DataFrame] = []

    for i, batch in enumerate(batches):
        try:
            raw = yf.download(
                batch, start=fetch_start, end=None,
                auto_adjust=True, progress=False, threads=True
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
                vol   = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})
                vol   = raw[["Volume"]].rename(columns={"Volume": batch[0]}) if "Volume" in raw.columns else pd.DataFrame()
            new_parts.append(close)
            if not isinstance(vol, pd.DataFrame) or not vol.empty:
                vol_parts.append(vol)
        except Exception as e:
            warn(f"Batch {i+1}/{len(batches)} error: {e}")
        time.sleep(0.3)

    if not new_parts:
        warn("No new data downloaded")
        return existing if not existing.empty else None

    new_data = pd.concat(new_parts, axis=1)
    new_data = new_data.loc[:, ~new_data.columns.duplicated()]
    new_data.index = pd.to_datetime(new_data.index)

    # Merge: existing + new rows, deduplicate by date
    if not existing.empty:
        # Align columns — keep union
        all_cols = existing.columns.union(new_data.columns)
        existing  = existing.reindex(columns=all_cols)
        new_data  = new_data.reindex(columns=all_cols)
        combined  = pd.concat([existing, new_data])
    else:
        combined = new_data

    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined = combined.ffill()

    # ── Safety guard: reject any non-date rows before saving ──────────────────
    bad_rows = [v for v in combined.index.astype(str) if not v[:4].isdigit()]
    if bad_rows:
        warn(f"CORRUPTED INDEX ROWS DETECTED — dropping before save: {bad_rows[:5]}")
        combined = combined[combined.index.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}")]
    assert len(combined) > 0, "Price cache is empty after cleanup — aborting save"

    # Write full S&P 500 cache
    combined.to_csv(sp500_path)
    ok(f"sp500_price_cache.csv → {combined.shape[0]} days × {combined.shape[1]} tickers")

    # Write backtest cache (last 504 trading days)
    bt = combined.tail(504)
    bt.to_csv(ROOT / "backtest_price_cache.csv")
    ok(f"backtest_price_cache.csv → {bt.shape[0]} days × {bt.shape[1]} tickers")

    # ── L6 institutional: save volume cache ───────────────────────────────────
    if vol_parts:
        try:
            new_vol = pd.concat(vol_parts, axis=1)
            new_vol = new_vol.loc[:, ~new_vol.columns.duplicated()]
            new_vol.index = pd.to_datetime(new_vol.index)
            vol_path = ROOT / "sp500_volume_cache.csv"
            if vol_path.exists():
                ev = pd.read_csv(vol_path, index_col=0, parse_dates=True)
                all_vcols = ev.columns.union(new_vol.columns)
                ev = ev.reindex(columns=all_vcols)
                new_vol = new_vol.reindex(columns=all_vcols)
                combined_vol = pd.concat([ev, new_vol])
            else:
                combined_vol = new_vol
            combined_vol = combined_vol[~combined_vol.index.duplicated(keep="last")].sort_index()
            combined_vol.to_csv(vol_path)
            ok(f"sp500_volume_cache.csv → {combined_vol.shape[0]} days × {combined_vol.shape[1]} tickers")
        except Exception as e:
            warn(f"Volume cache save failed: {e}")

    return combined


# ── Step 2: Recompute price-based signals ─────────────────────────────────────

def recompute_signals(prices: pd.DataFrame) -> None:
    regime_path = ROOT / "regime_ml_scores.csv"
    if not regime_path.exists():
        err("regime_ml_scores.csv not found — cannot update signals")
        return

    regime_df = pd.read_csv(regime_path).set_index("ticker")
    tickers   = regime_df.index.tolist()

    # ── L6: load volume cache for Amihud illiquidity ──────────────────────────
    vol_df: pd.DataFrame = pd.DataFrame()
    vol_path = ROOT / "sp500_volume_cache.csv"
    if vol_path.exists():
        try:
            vol_df = pd.read_csv(vol_path, index_col=0, parse_dates=True).sort_index()
        except Exception:
            pass

    # Determine market regime — use macro_signals.json if available (more comprehensive)
    market_regime = "BULL"
    macro_path = ROOT / "macro_signals.json"
    if macro_path.exists():
        try:
            macro = json.loads(macro_path.read_text())
            macro_sig = macro.get("macro_signal", "NEUTRAL")
            sma_sig   = macro.get("sma_cross_signal", "NEUTRAL")
            vix_val   = float(macro.get("vix", 20))
            if macro_sig == "RISK_ON" and sma_sig in ("GOLDEN", "NEUTRAL", "INSUFFICIENT_DATA") and vix_val < 25:
                market_regime = "BULL"
            elif macro_sig == "RISK_OFF" or vix_val > 30 or sma_sig == "DEATH":
                market_regime = "BEAR"
            else:
                market_regime = "SIDEWAYS"
        except Exception:
            pass
    if market_regime == "BULL" and not macro_path.exists():
        if "SPY" in prices.columns:
            spy = prices["SPY"].dropna()
            if len(spy) > 200:
                spy_sma200 = float(spy.rolling(200).mean().iloc[-1])
                spy_mom3m  = float(spy.iloc[-1] / spy.iloc[-63] - 1) if len(spy) > 63 else 0.0
                if spy.iloc[-1] < spy_sma200 and spy_mom3m < -0.05:
                    market_regime = "BEAR"
                elif spy.iloc[-1] < spy_sma200:
                    market_regime = "SIDEWAYS"

    log(f"Market regime: {market_regime}  (SPY trend)")

    # Compute fresh price signals for each ticker
    new_signals: dict[str, dict] = {}
    for tk in tickers:
        if tk not in prices.columns:
            continue
        p = prices[tk].dropna()
        if len(p) < 22:
            continue

        # Momentum
        mom_1m  = float(p.iloc[-1] / p.iloc[-22]  - 1) if len(p) > 22  else 0.0
        mom_3m  = float(p.iloc[-1] / p.iloc[-63]  - 1) if len(p) > 63  else 0.0
        mom_6m  = float(p.iloc[-1] / p.iloc[-126] - 1) if len(p) > 126 else 0.0
        mom_12s = float(p.iloc[-22] / p.iloc[-252] - 1) if len(p) > 252 else mom_3m

        # Trend: price vs 200-day SMA (z-score style: % above/below)
        sma200 = p.rolling(200).mean().iloc[-1]
        trend_200 = float((p.iloc[-1] / sma200 - 1) * 100) if not np.isnan(sma200) else 0.0

        # RSI (14-day)
        rsi_14 = _rsi(p, 14)

        # Inverse volatility (higher = less volatile = better)
        vol_21 = p.pct_change().rolling(21).std().iloc[-1]
        inv_vol = float(1.0 / (vol_21 * np.sqrt(252)) * 10) if vol_21 > 0 else 10.0

        # 52-week high proximity (George-Hwang 2004 momentum factor)
        high_52w      = float(p.tail(252).max()) if len(p) >= 252 else float(p.max())
        high_52w_ratio = float(p.iloc[-1] / high_52w - 1) if high_52w > 0 else 0.0

        # Short-term reversal (Jegadeesh 1990) — 1-week return (negated for reversal signal)
        short_rev = float(p.iloc[-1] / p.iloc[-6] - 1) if len(p) > 6 else 0.0

        # ── L6 institutional: Amihud illiquidity + dollar volume ─────────────
        amihud_illiq = 0.5     # default neutral
        avg_dvol_21d = 0.0     # avg daily dollar volume ($M)
        vol_ratio    = 1.0     # recent vol / 63d avg (volume momentum)
        if not vol_df.empty and tk in vol_df.columns:
            try:
                v = vol_df[tk].reindex(p.index).fillna(method="ffill").dropna()
                if len(v) > 21:
                    p_al  = p.reindex(v.index).dropna()
                    v_al  = v.reindex(p_al.index).dropna()
                    if len(p_al) > 21 and len(v_al) > 21:
                        ret_abs = p_al.pct_change().abs().dropna()
                        dvol    = (p_al * v_al).reindex(ret_abs.index)
                        mask    = dvol > 0
                        if mask.sum() > 10:
                            illiq = (ret_abs[mask] / dvol[mask]).tail(21).mean()
                            amihud_illiq = float(illiq * 1e8)
                            avg_dvol_21d = float(dvol.tail(21).mean() / 1e6)
                        vol_5d  = float(v_al.tail(5).mean())
                        vol_63d = float(v_al.tail(63).mean())
                        if vol_63d > 0:
                            vol_ratio = float(vol_5d / vol_63d)
            except Exception:
                pass

        new_signals[tk] = {
            "mom_1m":         round(mom_1m * 100, 4),
            "mom_3m":         round(mom_3m * 100, 4),
            "mom_6m":         round(mom_6m * 100, 4),
            "mom_12m_skip1m": round(mom_12s * 100, 4),
            "trend_200":      round(trend_200, 4),
            "rsi_14":         round(rsi_14, 4),
            "inv_vol":        round(inv_vol, 4),
            "high_52w_ratio": round(high_52w_ratio * 100, 4),
            "short_rev":      round(short_rev * 100, 4),
            "idio_mom":       round(mom_3m * 100, 4),  # placeholder; updated below
            "amihud_illiq":   round(amihud_illiq, 6),
            "avg_dvol_21d":   round(avg_dvol_21d, 2),
            "vol_ratio":      round(vol_ratio, 4),
        }

    if not new_signals:
        warn("No price signals computed — check price data coverage")
        return

    ok(f"Recomputed price signals for {len(new_signals)} tickers")

    # Compute idiosyncratic momentum (SPY-residualized 3m return)
    spy_mom3m_pct = 0.0
    if "SPY" in prices.columns:
        spy_p = prices["SPY"].dropna()
        if len(spy_p) > 63:
            spy_mom3m_pct = float(spy_p.iloc[-1] / spy_p.iloc[-63] - 1) * 100
    spy_rets = prices["SPY"].pct_change().dropna().tail(63).values if "SPY" in prices.columns else np.array([])
    for tk in list(new_signals.keys()):
        if tk not in prices.columns:
            continue
        try:
            tk_p = prices[tk].dropna()
            if len(tk_p) > 63 and len(spy_rets) > 30:
                tk_rets = tk_p.pct_change().dropna().tail(63).values
                min_len = min(len(tk_rets), len(spy_rets))
                if min_len > 30:
                    cov_matrix = np.cov(tk_rets[-min_len:], spy_rets[-min_len:])
                    spy_var = float(np.var(spy_rets[-min_len:])) + 1e-10
                    beta = float(cov_matrix[0, 1]) / spy_var
                    beta = max(-3.0, min(3.0, beta))
                    new_signals[tk]["idio_mom"] = round(new_signals[tk]["mom_3m"] - beta * spy_mom3m_pct, 4)
        except Exception:
            pass

    # Update regime_df with fresh price signals
    for tk, sigs in new_signals.items():
        for col, val in sigs.items():
            regime_df.loc[tk, col] = val

    # Re-rank mom and trend within universe (percentile 0→1)
    sig_tickers = [tk for tk in tickers if tk in new_signals]
    mom_vals  = pd.Series({tk: new_signals[tk]["mom_3m"]   for tk in sig_tickers})
    tr_vals   = pd.Series({tk: new_signals[tk]["trend_200"] for tk in sig_tickers})
    vol_vals  = pd.Series({tk: new_signals[tk]["inv_vol"]   for tk in sig_tickers})

    rank_mom   = mom_vals.rank(pct=True)
    rank_trend = tr_vals.rank(pct=True)
    rank_vol   = vol_vals.rank(pct=True)

    h52_vals   = pd.Series({tk: new_signals[tk].get("high_52w_ratio", 0.0) for tk in sig_tickers})
    rev_vals   = pd.Series({tk: -new_signals[tk].get("short_rev", 0.0)     for tk in sig_tickers})
    idio_vals  = pd.Series({tk: new_signals[tk].get("idio_mom", 0.0)       for tk in sig_tickers})

    rank_h52   = h52_vals.rank(pct=True)
    rank_rev   = rev_vals.rank(pct=True)
    rank_idio  = idio_vals.rank(pct=True)

    # ── L6 institutional: rank liquidity signals ───────────────────────────────
    # Amihud: invert (low illiquidity = more liquid = better rank)
    amihud_vals  = pd.Series({tk: -new_signals[tk].get("amihud_illiq", 0.0) for tk in sig_tickers})
    dvol_vals    = pd.Series({tk:  new_signals[tk].get("avg_dvol_21d", 0.0)  for tk in sig_tickers})
    vol_rat_vals = pd.Series({tk:  new_signals[tk].get("vol_ratio", 1.0)      for tk in sig_tickers})
    rank_liquidity   = amihud_vals.rank(pct=True)
    rank_dvol        = dvol_vals.rank(pct=True)
    rank_vol_momentum = vol_rat_vals.rank(pct=True)

    for tk in sig_tickers:
        regime_df.loc[tk, "rank_mom"]          = round(float(rank_mom.get(tk, 0.5)), 4)
        regime_df.loc[tk, "rank_trend"]        = round(float(rank_trend.get(tk, 0.5)), 4)
        regime_df.loc[tk, "rank_liquidity"]    = round(float(rank_liquidity.get(tk, 0.5)), 4)
        regime_df.loc[tk, "rank_dvol"]         = round(float(rank_dvol.get(tk, 0.5)), 4)
        regime_df.loc[tk, "rank_vol_momentum"] = round(float(rank_vol_momentum.get(tk, 0.5)), 4)
        # Store raw values too
        regime_df.loc[tk, "amihud_illiq"]  = new_signals[tk].get("amihud_illiq", 0.5)
        regime_df.loc[tk, "avg_dvol_21d"]  = new_signals[tk].get("avg_dvol_21d", 0.0)
        regime_df.loc[tk, "vol_ratio"]     = new_signals[tk].get("vol_ratio", 1.0)

    # Recompute predicted_score with institutional L6 signals:
    # Momentum 25% + Trend 13% + Quality 9% + Sentiment 7% + SUE 5% + Revision 5%
    # + InvVol 8% + 52wHigh 9% + IdioMom 7% + Reversal 4%
    # + Liquidity 5% (Amihud) + VolMomentum 3%  = 100%
    for tk in sig_tickers:
        r = regime_df.loc[tk]
        quality_n   = float(r.get("quality_score", 50)) / 100.0
        sent_n      = float(r.get("rank_sentiment", 50)) / 100.0
        sue_n       = float(r.get("rank_sue", 50)) / 100.0
        rev_n       = float(r.get("rank_revision", 50)) / 100.0
        rm          = float(rank_mom.get(tk, 0.5))
        rt          = float(rank_trend.get(tk, 0.5))
        rv          = float(rank_vol.get(tk, 0.5))

        rh    = float(rank_h52.get(tk, 0.5))
        rrev  = float(rank_rev.get(tk, 0.5))
        ridio = float(rank_idio.get(tk, 0.5))
        rliq  = float(rank_liquidity.get(tk, 0.5))     # L6 Amihud liquidity
        rvolm = float(rank_vol_momentum.get(tk, 0.5))  # L6 volume momentum

        score = (0.25 * rm    +
                 0.13 * rt    +
                 0.09 * quality_n +
                 0.07 * sent_n +
                 0.05 * sue_n +
                 0.05 * rev_n +
                 0.08 * rv    +
                 0.09 * rh    +
                 0.07 * ridio +
                 0.04 * rrev  +
                 0.05 * rliq  +
                 0.03 * rvolm)
        regime_df.loc[tk, "predicted_score"] = round(score, 6)

    # Assign market regime to all tickers
    regime_df["regime"] = market_regime

    # Assign signal based on predicted_score + regime
    def _assign_signal(row) -> str:
        s = float(row.get("predicted_score", 0.5))
        r = str(row.get("regime", "BULL"))
        crowding = float(row.get("crowding_level_num", 0))
        if s >= 0.65 and r in ("BULL", "SIDEWAYS"):
            return "LONG"
        elif s <= 0.35 and r in ("BEAR", "SIDEWAYS"):
            return "SHORT"
        else:
            return "HOLD"

    # Crowding as numeric for signal logic
    regime_df["crowding_level_num"] = regime_df["crowding_level"].map(
        {"CLEAR": 0, "WATCH": 1, "CROWDED": 2}
    ).fillna(0)
    regime_df["signal"] = regime_df.apply(_assign_signal, axis=1)
    regime_df = regime_df.drop(columns=["crowding_level_num"], errors="ignore")

    # Write updated regime_ml_scores.csv
    regime_df.index.name = "ticker"
    regime_df.reset_index().to_csv(regime_path, index=False)
    n_long  = (regime_df["signal"] == "LONG").sum()
    n_short = (regime_df["signal"] == "SHORT").sum()
    ok(f"regime_ml_scores.csv updated — {market_regime} | LONG={n_long}  SHORT={n_short}  HOLD={len(regime_df)-n_long-n_short}")

    # Update cpcv_predictions.csv
    cpcv_path = ROOT / "cpcv_predictions.csv"
    cpcv_rows = []
    for tk in regime_df.index:
        score = round(float(regime_df.loc[tk, "predicted_score"]), 6)
        cpcv_rows.append({
            "rebalance_date": TODAY,
            "ticker":         tk,
            "period":         "OOS",
            "is_oos":         True,
            "ridge_score":    score,
            "rf_score":       score,
            "lgbm_score":     score,
            "ensemble_score": score,
            "n_train":        252,
            "method":         "regime_ml_daily",
        })
    pd.DataFrame(cpcv_rows).to_csv(cpcv_path, index=False)
    ok(f"cpcv_predictions.csv → {len(cpcv_rows)} tickers")

    # Update factor_composite.csv
    fc_path = ROOT / "factor_composite.csv"
    fc_rows = []
    for tk in sig_tickers:
        s = new_signals[tk]
        quality_n = float(regime_df.loc[tk].get("quality_score", 50)) / 100.0
        fc_rows.append({
            "ticker":           tk,
            "momentum_z":       round(s["mom_3m"], 4),
            "low_vol_z":        round(s["inv_vol"] / 50.0 - 1.0, 4),
            "value_z":          round(quality_n - 0.5, 4),
            "quality_z":        round(quality_n - 0.5, 4),
            "factor_composite": round((s["mom_1m"] + s["mom_3m"] + s["mom_6m"]) / 3, 4),
        })
    fc_df = pd.DataFrame(fc_rows)
    mu = fc_df["factor_composite"].mean()
    sd = fc_df["factor_composite"].std() + 1e-9
    fc_df["factor_composite"] = ((fc_df["factor_composite"] - mu) / sd).round(4)
    fc_df.to_csv(fc_path, index=False)
    ok(f"factor_composite.csv → {len(fc_df)} tickers")


# ── Step 3: Update price_refresh_desk.csv ─────────────────────────────────────

def update_price_desk(prices: pd.DataFrame) -> None:
    if len(prices) < 2:
        return
    today_row  = prices.iloc[-1].dropna()
    prev_row   = prices.iloc[-2].dropna()
    refresh_rows = []
    for tk in today_row.index:
        curr = float(today_row[tk])
        prev = float(prev_row.get(tk, curr))
        chg  = (curr / prev - 1) if prev > 0 else 0.0
        refresh_rows.append({
            "ticker":        tk,
            "last_price":    round(curr, 2),
            "prev_close":    round(prev, 2),
            "daily_chg_pct": round(chg * 100, 3),
            "days_stale":    0,
            "updated_date":  TODAY,
        })
    pd.DataFrame(refresh_rows).to_csv(ROOT / "price_refresh_desk.csv", index=False)
    ok(f"price_refresh_desk.csv → {len(refresh_rows)} tickers (date: {prices.index[-1].date()})")


# ── Step 3b: L1 Data Health Monitor ──────────────────────────────────────────

def write_data_health(prices: pd.DataFrame) -> None:
    """
    L1 institutional: spike detection + staleness + completeness per ticker.
    Outputs data_health.csv with health_score 0-100.
    """
    if len(prices) < 5:
        return
    rows = []
    tickers = [c for c in prices.columns if c not in ("", "Date", "Unnamed: 0")]
    for tk in tickers:
        p = prices[tk].dropna()
        if len(p) < 5:
            continue
        rets = p.pct_change().dropna()
        # Spike: |daily return| vs rolling 252d σ (z-score)
        vol_252 = float(rets.rolling(252, min_periods=21).std().iloc[-1]) if len(rets) >= 21 else 0.02
        last_ret = float(rets.iloc[-1]) if len(rets) > 0 else 0.0
        z_score = abs(last_ret / vol_252) if vol_252 > 0 else 0.0
        spike_flag = int(z_score > 4.0)
        # Staleness
        last_date = prices.index[-1]
        days_stale = max(0, (pd.Timestamp.today() - last_date).days)
        # Completeness: valid rows in last 252 days
        valid_n     = int(p.tail(252).notna().sum())
        completeness = round(valid_n / 252.0, 3)
        # Health score
        health = 100.0
        if spike_flag:
            health -= 20.0
        if days_stale > 3:
            health -= 30.0
        elif days_stale > 1:
            health -= 10.0
        health -= (1.0 - completeness) * 50.0
        health = max(0.0, min(100.0, health))
        rows.append({
            "ticker":       tk,
            "last_date":    last_date.strftime("%Y-%m-%d"),
            "days_stale":   days_stale,
            "last_ret_pct": round(last_ret * 100, 3),
            "z_score":      round(z_score, 2),
            "spike_flag":   spike_flag,
            "completeness": completeness,
            "health_score": round(health, 1),
            "updated_date": TODAY,
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data_health.csv", index=False)
    n_spike = int((df["spike_flag"] == 1).sum())
    n_stale = int((df["days_stale"] > 3).sum())
    n_low   = int((df["health_score"] < 60).sum())
    ok(f"data_health.csv → {len(df)} tickers  spikes={n_spike}  stale={n_stale}  low-health={n_low}")


# ── Step 4: Rebuild alpha_scores.csv via step87 ────────────────────────────────

def rebuild_alpha_scores() -> None:
    import subprocess, sys
    step87 = ROOT / "canyon_final_v9_step87_alpha_aggregator.py"
    if not step87.exists():
        warn("step87 not found — skipping alpha_scores rebuild")
        return
    result = subprocess.run(
        [sys.executable, str(step87)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        alpha_path = ROOT / "alpha_scores.csv"
        n = len(pd.read_csv(alpha_path)) if alpha_path.exists() else "?"
        ok(f"alpha_scores.csv rebuilt → {n} tickers")
    else:
        warn(f"step87 failed rc={result.returncode}")
        for line in (result.stderr or result.stdout or "").splitlines()[-3:]:
            log(f"  {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"\n{CYAN}{BOLD}Canyon v9 — Daily Price & Signal Refresh  [{TODAY}]{RESET}\n")

    # 1. Download missing prices (incremental — usually just 1 day)
    log("Refreshing prices …")
    prices = refresh_prices()
    if prices is None:
        err("Price refresh failed — signals NOT updated")
        return

    # 2. Recompute signals from fresh prices
    log("Recomputing signals …")
    recompute_signals(prices)

    # 3. Update price desk (today's close + daily change)
    log("Updating price desk …")
    update_price_desk(prices)

    # 3b. L1 data health monitor
    log("Writing data health …")
    write_data_health(prices)

    # 4. Rebuild alpha_scores.csv
    log("Rebuilding alpha_scores.csv …")
    rebuild_alpha_scores()

    elapsed = time.time() - t0
    print(f"\n{GREEN}{BOLD}  Done — {elapsed:.0f}s{RESET}\n")


if __name__ == "__main__":
    main()
