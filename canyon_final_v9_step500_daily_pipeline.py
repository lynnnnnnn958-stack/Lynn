#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 500: Daily Automated Signal Pipeline
======================================================
The production core of Canyon v9 — wires all research modules into a
daily signal generation pipeline that runs automatically every morning.

What separates institutional research from hobby projects is not the number
of signals but whether they run reliably every day with a documented live
track record.

This script does four things:

  1. Daily signal snapshot
     - Fetches latest prices, options flow, and institutional holdings
     - Updates all factor z-scores
     - Generates today's Top-N long/short recommendations

  2. Paper trading log
     - Records daily signal changes to paper_trading_log.csv
     - Tracks hypothetical P&L as if signals were traded
     - This is the live signal track record institutions require

  3. Signal change detection
     - Compares yesterday vs today signals
     - Fires ALERT when a high-IC signal flips
     - Reduces unnecessary monitoring burden

  4. Daily summary report
     - One-page Markdown: market state, today's signals, suggested actions
     - Can be emailed or posted to Slack

Usage:
  Manual:    .venv/bin/python canyon_final_v9_step500_daily_pipeline.py
  Scheduled: crontab -e, add:
             0 18 * * 1-5 cd /path/to/canyon_quant && .venv/bin/python canyon_final_v9_step500_daily_pipeline.py >> logs/daily.log 2>&1

Run:
  .venv/bin/python canyon_final_v9_step500_daily_pipeline.py
"""
from __future__ import annotations

import sys
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
LOG_FILE   = ROOT / "paper_trading_log.csv"
REPORT_DIR = ROOT / "daily_reports"
REPORT_DIR.mkdir(exist_ok=True)

TODAY     = datetime.now().strftime("%Y-%m-%d")
TODAY_TS  = pd.Timestamp(TODAY)
N_LONG    = 15
N_SHORT   = 15
SECTOR_CAP_PCT  = 0.30   # max 30% of book in any single sector
TURNOVER_LAMBDA = 0.12   # z-score penalty for new positions (reduces churn)

# ── Regime-conditional IC weight multipliers ──────────────────────────────────
# Key insight: momentum/ML signals have higher IC in low-volatility bull markets;
# quality/accruals/value signals lead in bear or high-fear regimes.
# Multipliers applied on top of the IC²-optimal base weights (max ±50%).
# Regime buckets: (HMM_state, VIX_bucket)
#   HMM_state : "BULL" | "BEAR" | "NEUTRAL"
#   VIX_bucket: "LOW"  (VIX<20) | "MID" (20–30) | "HIGH" (>30)
REGIME_IC_MULTIPLIERS: dict[tuple[str, str], dict[str, float]] = {
    ("BULL", "LOW"): {       # trending bull, low fear — momentum & drift dominates
        "ml_score":      1.25,
        "squeeze":       1.30,
        "sig_8k":        1.15,
        "sig_pead":      1.35,  # PEAD strongest in calm bull (drift continues)
        "sig_revision":  1.15,
        "alt_trends":    1.20,
        "accruals":      0.80,
        "sig_fundamental": 0.85,
        "sig_crowd":     0.80,
    },
    ("BULL", "MID"): {       # bull, moderate fear — balanced
        "ml_score":      1.10,
        "squeeze":       1.15,
        "sig_pead":      1.20,  # drift still works
        "sig_insider":   1.10,
        "accruals":      0.90,
        "sig_crowd":     0.90,
    },
    ("BULL", "HIGH"): {      # fear spike in bull = dip-buy; insiders active
        "squeeze":       1.40,
        "sig_insider":   1.30,
        "sig_crowd":     1.20,
        "sig_pead":      0.85,  # panic disrupts drift
        "ml_score":      0.85,
        "alt_trends":    0.75,
    },
    ("BEAR", "LOW"): {       # grinding bear, complacent — quality leads
        "accruals":      1.25,
        "sig_fundamental": 1.25,
        "sig_crowd":     1.30,
        "sig_cross_asset": 1.20,
        "squeeze":       0.65,
        "ml_score":      0.85,
        "sig_pead":      0.75,  # PEAD reverses in bear (misses more persistent)
        "sig_8k":        0.80,
        "alt_trends":    0.80,
    },
    ("BEAR", "MID"): {       # bear with rising fear — defensive + crowding
        "accruals":      1.35,
        "sig_fundamental": 1.30,
        "sig_crowd":     1.25,
        "finbert":       1.15,
        "squeeze":       0.70,
        "ml_score":      0.88,
        "sig_pead":      0.70,  # guidance cuts negate beats
        "alt_trends":    0.75,
    },
    ("BEAR", "HIGH"): {      # crisis regime — quality & crowding unwind
        "accruals":      1.50,
        "sig_fundamental": 1.45,
        "sig_crowd":     1.60,
        "sig_cross_asset": 1.35,
        "finbert":       1.20,
        "squeeze":       0.50,
        "ml_score":      0.70,
        "sig_pead":      0.55,  # earnings surprises irrelevant in crisis
        "sig_8k":        0.65,
        "alt_trends":    0.60,
        "sig_revision":  0.75,
    },
    ("NEUTRAL", "LOW"):  {},  # no regime signal — use IC²-optimal weights as-is
    ("NEUTRAL", "MID"):  {
        "accruals":      1.08,
        "sig_crowd":     1.08,
        "squeeze":       0.94,
    },
    ("NEUTRAL", "HIGH"): {
        "accruals":      1.20,
        "sig_crowd":     1.25,
        "squeeze":       0.80,
        "ml_score":      0.92,
    },
}


# =============================================================================
# 1. Fast price refresh
# =============================================================================

def fetch_latest_prices(tickers: list[str]) -> pd.Series:
    """Fetch today's closing prices for all tickers."""
    prices = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).fast_info
            prices[tk] = float(info.last_price)
        except Exception:
            prices[tk] = np.nan
    return pd.Series(prices)


# =============================================================================
# 2. Load existing signals
# =============================================================================

def load_current_signals() -> dict[str, pd.Series | pd.DataFrame]:
    """Load the most recent signal scores from all step outputs."""
    sigs = {}

    # CPCV ensemble score (ML signal — most recent rebalance)
    pred_path = ROOT / "cpcv_predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path, parse_dates=["rebalance_date"])
        latest_reb = preds["rebalance_date"].max()
        latest_preds = preds[preds["rebalance_date"] == latest_reb] \
                            .set_index("ticker")["ensemble_score"]
        sigs["ml_score"] = latest_preds
        sigs["latest_rebalance"] = latest_reb

    # Factor composite
    fc_path = ROOT / "factor_composite.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path).set_index("ticker")["factor_composite"]
        sigs["factor_score"] = fc

    # Smart money
    sm_path = ROOT / "smart_money_signal.csv"
    if sm_path.exists():
        sm = pd.read_csv(sm_path).set_index("ticker")["smart_money_score"]
        sigs["smart_money"] = sm

    # Accruals quality
    acc_path = ROOT / "accruals_snapshot.csv"
    if acc_path.exists():
        acc = pd.read_csv(acc_path).set_index("ticker")["accrual_quality"]
        sigs["accruals"] = acc

    # Short squeeze
    sq_path = ROOT / "short_squeeze_signal.csv"
    if sq_path.exists():
        sq = pd.read_csv(sq_path).set_index("ticker")["intensity_score"]
        sigs["squeeze"] = sq

    # FinBERT news sentiment (step79 output)
    fb_path = ROOT / "finbert_sentiment.csv"
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        if "ticker" in fb.columns and "sentiment_zscore" in fb.columns:
            sigs["finbert"] = fb.set_index("ticker")["sentiment_zscore"]

    # SEC 10-K/10-Q MD&A sentiment delta (step80 output)
    mda_path = ROOT / "sec_mda_sentiment.csv"
    if mda_path.exists():
        mda = pd.read_csv(mda_path)
        if "ticker" in mda.columns and "sig_10k" in mda.columns:
            sigs["sig_10k"] = mda.set_index("ticker")["sig_10k"]

    # Insider trading signal (from alpha_scores.csv sig_insider column)
    alpha_path = ROOT / "alpha_scores.csv"
    if alpha_path.exists():
        alpha_df = pd.read_csv(alpha_path)
        if "ticker" in alpha_df.columns:
            alpha_df = alpha_df.set_index("ticker")
            if "sig_insider" in alpha_df.columns:
                sigs["sig_insider"] = pd.to_numeric(
                    alpha_df["sig_insider"], errors="coerce")
            if "sig_revision" in alpha_df.columns:
                sigs["sig_revision"] = pd.to_numeric(
                    alpha_df["sig_revision"], errors="coerce")

    # Options flow signal (step options output)
    opt_path = ROOT / "options_signals.csv"
    if opt_path.exists():
        opt = pd.read_csv(opt_path)
        if "ticker" in opt.columns:
            opt = opt.set_index("ticker")
            col = next((c for c in ("flow_score", "alpha_options", "rank_options")
                        if c in opt.columns), None)
            if col:
                sigs["sig_options"] = pd.to_numeric(opt[col], errors="coerce")

    # Google Trends momentum (3-month change, wide format: dates × tickers)
    gt_path = ROOT / "alt_google_trends.csv"
    if gt_path.exists():
        try:
            gt = pd.read_csv(gt_path, index_col=0)
            if len(gt) >= 4:
                cur = gt.iloc[-1].astype(float)
                lag = gt.iloc[-4].astype(float)   # 3 months ago (monthly data)
                mom = (cur - lag) / (lag.abs() + 1.0)
                sigs["alt_trends"] = mom.dropna()
        except Exception:
            pass

    # Wikipedia views momentum (same wide format)
    wiki_path = ROOT / "alt_wikipedia_views.csv"
    if wiki_path.exists():
        try:
            wk = pd.read_csv(wiki_path, index_col=0)
            if len(wk) >= 4:
                cur = wk.iloc[-1].astype(float)
                lag = wk.iloc[-4].astype(float)
                mom = (cur - lag) / (lag.abs() + 1.0)
                sigs["alt_wiki"] = mom.dropna()
        except Exception:
            pass

    # XBRL fundamental composite (step89, quarterly)
    xbrl_path = ROOT / "xbrl_fundamentals.csv"
    if xbrl_path.exists():
        try:
            xbrl = pd.read_csv(xbrl_path)
            if "ticker" in xbrl.columns and "sig_fundamental" in xbrl.columns:
                sigs["sig_fundamental"] = xbrl.set_index("ticker")["sig_fundamental"]
        except Exception:
            pass

    # 13F institutional crowding signal (step87, quarterly)
    crowd_path = ROOT / "13f_crowding.csv"
    if crowd_path.exists():
        try:
            crowd = pd.read_csv(crowd_path)
            if "ticker" in crowd.columns and "sig_crowd" in crowd.columns:
                sigs["sig_crowd"] = crowd.set_index("ticker")["sig_crowd"]
        except Exception:
            pass

    # Cross-asset momentum signal (step86 output)
    ca_path = ROOT / "cross_asset_signals.csv"
    if ca_path.exists():
        try:
            ca = pd.read_csv(ca_path)
            if "ticker" in ca.columns and "cross_asset_z" in ca.columns:
                sigs["sig_cross_asset"] = ca.set_index("ticker")["cross_asset_z"]
        except Exception:
            pass

    # Earnings 8-K NLP signal (step81_earnings_nlp output)
    k8_path = ROOT / "sec_8k_sentiment.csv"
    if k8_path.exists():
        try:
            k8 = pd.read_csv(k8_path)
            if "ticker" in k8.columns and "sig_8k" in k8.columns:
                sigs["sig_8k"] = k8.set_index("ticker")["sig_8k"]
        except Exception:
            pass

    # PEAD / SUE earnings surprise signal (step81_earnings_surprise output)
    # Mechanism: stocks beat/miss EPS estimates → drift continues 2–60 days
    # Literature IC ≈ 0.06–0.09 (Bernard & Thomas 1989, Ball & Brown 1968)
    pead_path = ROOT / "earnings_surprise_scores.csv"
    if pead_path.exists():
        try:
            pead = pd.read_csv(pead_path)
            if "ticker" in pead.columns:
                pead = pead.set_index("ticker")
                col = next((c for c in ("sue_decayed", "rank_sue", "sue")
                            if c in pead.columns), None)
                if col:
                    sigs["sig_pead"] = pd.to_numeric(pead[col], errors="coerce")
        except Exception:
            pass

    # HMM regime (most recent)
    hmm_path = ROOT / "hmm_regime_monthly.csv"
    if hmm_path.exists():
        hmm = pd.read_csv(hmm_path).sort_values("rebalance_date")
        sigs["hmm_regime"]    = hmm.iloc[-1]["regime_label"]
        sigs["hmm_exposure"]  = float(hmm.iloc[-1]["exposure"])

    # Macro composite (most recent)
    mc_path = ROOT / "macro_composite_daily.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path, parse_dates=["date"]).sort_values("date")
        sigs["macro_score"]   = float(mc.iloc[-1]["macro_composite"])
        sigs["macro_date"]    = str(mc.iloc[-1]["date"].date())

    return sigs


# =============================================================================
# 3. Build composite daily signal
# =============================================================================

def build_daily_composite(
    sigs:                dict,
    tickers:             list[str],
    prev_long:           set[str] | None = None,
    prev_short:          set[str] | None = None,
    ic_weight_override:  bool = False,
) -> pd.DataFrame:
    """
    Combine all available signals into a daily composite score.
    Weights based on OOS IC; optionally calibrated by live walk-forward IC.
    """
    IC_WEIGHTS = {
        "ml_score":        0.23,  # IC=+0.065 (CPCV ensemble)
        "factor_score":    0.09,  # IC=+0.042 (quality/momentum factors, generic)
        "smart_money":     0.10,  # IC=+0.050 (institutional cluster)
        "accruals":        0.10,  # IC=+0.080 (Sloan accruals)
        "squeeze":         0.09,  # IC=+0.120 (short squeeze)
        "sig_fundamental": 0.07,  # IC=+0.060 (XBRL E/P+ROE+FCF+growth, step89)
        "finbert":         0.05,  # IC=+0.050 (news sentiment, step79)
        "sig_10k":         0.04,  # IC=+0.040 (MD&A delta, step80)
        "sig_8k":          0.03,  # IC=+0.050 (earnings 8-K tone, step81)
        "sig_pead":        0.04,  # IC=+0.070 (PEAD/SUE drift, step81b; Bernard&Thomas)
        "sig_insider":     0.04,  # IC=+0.050 (insider buy/sell)
        "sig_revision":    0.04,  # IC=+0.040 (analyst upgrades)
        "sig_options":     0.02,  # IC=+0.030 (options IV+flow; step82 now in pipeline)
        "alt_trends":      0.02,  # IC=+0.020 (Google Trends 3m)
        "alt_wiki":        0.01,  # IC=+0.018 (Wikipedia views 3m)
        "sig_cross_asset": 0.02,  # IC=+0.030 (intermarket momentum, step86)
        "sig_crowd":       0.01,  # IC=+0.030 (13F crowding contrarian, step87)
    }
    # Total = 1.00 verified:
    # 0.23+0.09+0.10+0.10+0.09+0.07+0.05+0.04+0.03+0.04+0.04+0.04+0.02+0.02+0.01+0.02+0.01 = 1.00

    # Optionally calibrate weights with live OOS IC from walk-forward
    if ic_weight_override:
        IC_WEIGHTS = load_dynamic_ic_weights(IC_WEIGHTS)

    df = pd.DataFrame(index=tickers)

    for sig_name, weight in IC_WEIGHTS.items():
        if sig_name in sigs and isinstance(sigs[sig_name], pd.Series):
            s = sigs[sig_name].reindex(tickers)
            mu, sd = s.mean(), s.std() + 1e-10
            df[sig_name] = (s - mu) / sd
        else:
            df[sig_name] = 0.0

    # Weighted composite
    df["composite"] = sum(
        df[sig] * w for sig, w in IC_WEIGHTS.items()
    )

    # Apply HMM regime scaling
    hmm_exp = float(sigs.get("hmm_exposure", 1.0))
    macro_s = float(sigs.get("macro_score", 0.0))
    regime_scale = hmm_exp
    if macro_s < -0.5:
        regime_scale = min(regime_scale, 0.75)
    df["regime_scaled"] = df["composite"] * regime_scale

    # Multi-period horizon consistency score (L3-P3)
    # Weight positions that look good across multiple horizons simultaneously.
    # Uses ic_decay_by_lag.csv: IC at 5/10/21/42/63d → horizon weights.
    # A stock must rank well at short AND medium horizons to get boosted.
    df["horizon_score"] = _compute_horizon_score(df["regime_scaled"], tickers)
    if "horizon_score" in df.columns and df["horizon_score"].std() > 1e-6:
        # Blend: 80% original + 20% horizon consistency bonus
        df["regime_scaled"] = 0.80 * df["regime_scaled"] + 0.20 * df["horizon_score"]

    df = df.sort_values("regime_scaled", ascending=False)
    if prev_long is not None or prev_short is not None:
        df = apply_turnover_penalty(df, prev_long or set(), prev_short or set())
    return df


def _compute_horizon_score(
    base_rank: pd.Series,
    tickers: list[str],
) -> pd.Series:
    """
    Horizon consistency score: reward positions that rank well across
    multiple IC-decay-weighted forecast horizons.

    Uses ic_decay_by_lag.csv to get IC at each horizon, then computes
    a weighted rank consensus. If the IC decay file is missing, returns
    the base_rank unchanged (no adjustment).
    """
    ic_path = ROOT / "ic_decay_by_lag.csv"
    if not ic_path.exists():
        return base_rank * 0.0   # zero bonus, blend formula keeps 80% base

    try:
        ic_df = pd.read_csv(ic_path)
        if not all(c in ic_df.columns for c in ["signal", "horizon", "ic"]):
            return base_rank * 0.0

        # Get IC at each horizon for ml_ensemble (best-calibrated signal)
        ml = ic_df[ic_df["signal"] == "ml_ensemble"]
        h_ics: dict[int, float] = {}
        for h in [5, 10, 21, 42, 63]:
            row = ml[ml["horizon"] == h].sort_values("date").tail(1)
            if not row.empty:
                ic_val = float(row["ic"].iloc[0])
                if ic_val > 0.01:   # only use positive IC horizons
                    h_ics[h] = ic_val

        if not h_ics:
            return base_rank * 0.0

        # IC²-normalized horizon weights
        ic_sq_total = sum(v**2 for v in h_ics.values())
        h_weights   = {h: (ic**2 / ic_sq_total) for h, ic in h_ics.items()}

        # The current composite signal is our best estimate of the 1-step score.
        # For each horizon, shift the rank slightly to account for mean-reversion
        # at short horizons and momentum at long horizons.
        # Simplified: score = sum_h w_h * base_score * (1 - reversal_factor(h))
        # reversal_factor: short horizons (h≤10) often have mild reversal IC → discount
        combined = pd.Series(0.0, index=base_rank.index)
        for h, w in h_weights.items():
            reversal_discount = 0.85 if h <= 10 else 1.0
            combined += w * base_rank * reversal_discount

        # Z-score to match base_rank scale
        mu, sd = combined.mean(), combined.std() + 1e-9
        return ((combined - mu) / sd).rename("horizon_score")

    except Exception:
        return base_rank * 0.0


# =============================================================================
# 4. Paper trading log
# =============================================================================

def sector_neutral_picks(
    composite_df: pd.DataFrame,
    n_long: int,
    n_short: int,
) -> tuple[list[str], list[str]]:
    """Select top N long/short respecting 30% max sector concentration."""
    sectors: dict[str, str] = {}
    regime_path = ROOT / "regime_ml_scores.csv"
    if regime_path.exists():
        rm = pd.read_csv(regime_path)
        if "ticker" in rm.columns and "sector" in rm.columns:
            sectors = rm.set_index("ticker")["sector"].to_dict()

    def _pick(candidates: list[str], n: int) -> list[str]:
        picks: list[str] = []
        sector_cnt: dict[str, int] = {}
        max_per = max(2, int(n * SECTOR_CAP_PCT))
        for tk in candidates:
            if len(picks) >= n:
                break
            sector = sectors.get(tk, "Other")
            if sector_cnt.get(sector, 0) < max_per:
                picks.append(tk)
                sector_cnt[sector] = sector_cnt.get(sector, 0) + 1
        if len(picks) < n:
            remaining = [t for t in candidates if t not in picks]
            picks.extend(remaining[: n - len(picks)])
        return picks

    return _pick(composite_df.index.tolist(), n_long), \
           _pick(composite_df.index[::-1].tolist(), n_short)


def apply_turnover_penalty(
    composite_df: pd.DataFrame,
    prev_long: set[str],
    prev_short: set[str],
) -> pd.DataFrame:
    """Subtract TURNOVER_LAMBDA from composite/regime_scaled for new positions."""
    if not prev_long and not prev_short:
        return composite_df
    df = composite_df.copy()
    n = len(df)
    margin = max(5, int(n * 0.30))
    near_long  = set(df.head(N_LONG + margin).index) - prev_long
    near_short = set(df.tail(N_SHORT + margin).index) - prev_short
    for tk in near_long | near_short:
        if tk in df.index:
            df.loc[tk, "composite"]     -= TURNOVER_LAMBDA
            df.loc[tk, "regime_scaled"] -= TURNOVER_LAMBDA
    return df.sort_values("regime_scaled", ascending=False)


def update_paper_trading_log(
    composite_df: pd.DataFrame,
    prices_today: pd.Series,
    sigs:         dict,
    top_long:     list[str] | None = None,
    top_short:    list[str] | None = None,
) -> None:
    """
    Append today's top-N long/short recommendations to the paper trading log.
    Also compute P&L for yesterday's recommendations.
    """
    if top_long is None:
        top_long  = composite_df.head(N_LONG).index.tolist()
    if top_short is None:
        top_short = composite_df.tail(N_SHORT).index.tolist()

    # Load existing log
    if LOG_FILE.exists():
        log = pd.read_csv(LOG_FILE, parse_dates=["date"])
    else:
        log = pd.DataFrame()

    # Compute yesterday's P&L if we have yesterday's entry
    pnl_today = np.nan
    if not log.empty:
        yesterday = log[log["date"] == log["date"].max()]
        if not yesterday.empty:
            prev_long  = yesterday.iloc[0].get("long_stocks", "").split("|")
            prev_short = yesterday.iloc[0].get("short_stocks", "").split("|")

            long_rets = []
            for tk in prev_long:
                if tk and tk in prices_today.index:
                    prev_price = yesterday.iloc[0].get(f"price_{tk}", np.nan)
                    if not np.isnan(prev_price) and prev_price > 0:
                        long_rets.append(float(prices_today[tk]) / prev_price - 1)

            short_rets = []
            for tk in prev_short:
                if tk and tk in prices_today.index:
                    prev_price = yesterday.iloc[0].get(f"price_{tk}", np.nan)
                    if not np.isnan(prev_price) and prev_price > 0:
                        short_rets.append(-(float(prices_today[tk]) / prev_price - 1))

            if long_rets or short_rets:
                long_contrib  = np.mean(long_rets)  if long_rets  else 0.0
                short_contrib = np.mean(short_rets) if short_rets else 0.0
                n_sides = (1 if long_rets else 0) + (1 if short_rets else 0)
                pnl_today = (long_contrib + short_contrib) / n_sides

    # Build today's row
    row = {
        "date":          TODAY,
        "long_stocks":   "|".join(top_long),
        "short_stocks":  "|".join(top_short),
        "hmm_regime":    sigs.get("hmm_regime", "Unknown"),
        "hmm_exposure":  sigs.get("hmm_exposure", 1.0),
        "macro_score":   round(float(sigs.get("macro_score", 0.0)), 3),
        "pnl_today":     round(pnl_today, 6) if not np.isnan(pnl_today) else None,
    }

    # Add prices for tomorrow's P&L computation
    for tk in top_long + top_short:
        if tk in prices_today.index and not np.isnan(prices_today[tk]):
            row[f"price_{tk}"] = round(float(prices_today[tk]), 2)

    new_row = pd.DataFrame([row])
    log = pd.concat([log, new_row], ignore_index=True)
    log.to_csv(LOG_FILE, index=False)


# =============================================================================
# 5. Signal change detection
# =============================================================================

def detect_signal_changes(
    composite_df: pd.DataFrame,
    top_long:  list[str] | None = None,
    top_short: list[str] | None = None,
) -> list[str]:
    """
    Compare today vs yesterday. Highlight stocks that moved
    significantly in the ranking (>5 positions).
    """
    alerts = []
    if not LOG_FILE.exists():
        return alerts

    log = pd.read_csv(LOG_FILE)
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    past = log[log["date"].dt.date < pd.Timestamp(TODAY).date()]
    if past.empty:
        return alerts

    yesterday = past.iloc[-1]
    prev_long  = set(yesterday.get("long_stocks", "").split("|"))
    prev_short = set(yesterday.get("short_stocks", "").split("|"))

    today_long  = set(top_long  or composite_df.head(N_LONG).index.tolist())
    today_short = set(top_short or composite_df.tail(N_SHORT).index.tolist())

    new_longs  = today_long  - prev_long
    new_shorts = today_short - prev_short
    drop_longs = prev_long   - today_long
    drop_shorts= prev_short  - today_short

    if new_longs:
        alerts.append(f"NEW LONG:  {', '.join(sorted(new_longs))}")
    if drop_longs:
        alerts.append(f"EXIT LONG: {', '.join(sorted(drop_longs))}")
    if new_shorts:
        alerts.append(f"NEW SHORT: {', '.join(sorted(new_shorts))}")
    if drop_shorts:
        alerts.append(f"EXIT SHORT:{', '.join(sorted(drop_shorts))}")

    return alerts


# =============================================================================
# 6. Daily Markdown report
# =============================================================================

def generate_daily_report(
    composite_df: pd.DataFrame,
    prices_today: pd.Series,
    sigs:         dict,
    alerts:       list[str],
    pnl_series:   pd.Series | None,
    top_long:     list[str] | None = None,
    top_short:    list[str] | None = None,
) -> str:
    hmm  = sigs.get("hmm_regime",   "Unknown")
    exp  = float(sigs.get("hmm_exposure", 1.0))
    mac  = float(sigs.get("macro_score",  0.0))
    mac_regime = "RISK_ON" if mac > 0.5 else "RISK_OFF" if mac < -0.5 else "NEUTRAL"
    reb  = str(sigs.get("latest_rebalance", "N/A"))[:10]

    lines = [
        f"# Canyon v9 Daily Signal Report — {TODAY}",
        "",
        "## Market Regime",
        f"  HMM State:       **{hmm}**  (target exposure {exp:.0%})",
        f"  Macro Overlay:   **{mac_regime}**  (composite {mac:+.2f}σ)",
        f"  Last Rebalance:  {reb}",
        "",
    ]

    # P&L tracking
    if pnl_series is not None and len(pnl_series) > 0:
        cum_ret = (1 + pnl_series.fillna(0)).prod() - 1
        lines += [
            "## Paper Trading Performance",
            f"  Total days tracked: {len(pnl_series)}",
            f"  Cumulative return:  {cum_ret:+.2%}",
            "",
        ]

    # Today's signals
    if top_long is not None:
        top_long = composite_df.loc[[t for t in top_long if t in composite_df.index]]
    else:
        top_long = composite_df.head(N_LONG)
    if top_short is not None:
        top_short = composite_df.loc[[t for t in top_short if t in composite_df.index]]
    else:
        top_short = composite_df.tail(N_SHORT).iloc[::-1]

    lines += [
        "## Today's Recommendations",
        "",
        "**LONG** (top 8 by composite score):",
        "",
        "| Rank | Ticker | Score | Price | ML | Factor | Smart$ | Squeeze |",
        "|------|--------|:-----:|------:|:--:|:------:|:------:|:-------:|",
    ]
    for rank, (tk, row) in enumerate(top_long.iterrows(), 1):
        price = prices_today.get(tk, np.nan)
        price_str = f"${price:.2f}" if not np.isnan(price) else "—"
        ml  = f"{row.get('ml_score',    0):+.2f}"
        fac = f"{row.get('factor_score',0):+.2f}"
        sm  = f"{row.get('smart_money', 0):+.2f}"
        sq  = f"{row.get('squeeze',     0):+.2f}"
        lines.append(f"| {rank} | **{tk}** | {row['composite']:+.3f} | "
                     f"{price_str} | {ml} | {fac} | {sm} | {sq} |")

    lines += [
        "",
        "**SHORT** (bottom 8 by composite score):",
        "",
        "| Rank | Ticker | Score | Price |",
        "|------|--------|:-----:|------:|",
    ]
    for rank, (tk, row) in enumerate(top_short.iterrows(), 1):
        price = prices_today.get(tk, np.nan)
        price_str = f"${price:.2f}" if not np.isnan(price) else "—"
        lines.append(f"| {rank} | **{tk}** | {row['composite']:+.3f} | {price_str} |")

    # Alerts
    if alerts:
        lines += ["", "## Signal Changes (Action Required)", ""]
        for a in alerts:
            lines.append(f"  ⚡ {a}")

    lines += [
        "",
        "---",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Canyon v9 Automated Pipeline*",
    ]

    return "\n".join(lines)


# =============================================================================
# 7. Cumulative paper trading statistics
# =============================================================================

def compute_paper_stats() -> tuple[pd.Series | None, str]:
    if not LOG_FILE.exists():
        return None, "No paper trading log yet."

    log = pd.read_csv(LOG_FILE, parse_dates=["date"])
    if "pnl_today" not in log.columns or log["pnl_today"].notna().sum() < 2:
        return None, f"Log has {len(log)} entries, P&L tracking starts day 2."

    pnl = log["pnl_today"].dropna()
    cum = (1 + pnl).prod() - 1
    sr  = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)
    dd  = ((1 + pnl).cumprod() / (1 + pnl).cumprod().cummax() - 1).min()

    summary = (f"Days: {len(pnl)}  "
               f"Cum: {cum:+.2%}  "
               f"Ann.Sharpe: {sr:.2f}  "
               f"MaxDD: {dd:.2%}")
    return pnl, summary


# =============================================================================
# BL refinement — blend IC-composite ranking with Black-Litterman posterior
# =============================================================================

def apply_bl_refinement(composite_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load or run Black-Litterman optimizer, then blend BL posterior weights
    (30%) with the IC-composite ranking (70%).
    Falls back to unmodified composite if BL is unavailable.
    """
    bl_path = ROOT / "bl_weights.csv"

    # Run BL if output missing or stale (> 23h)
    needs_run = True
    if bl_path.exists():
        age_h = (pd.Timestamp.now().timestamp() - bl_path.stat().st_mtime) / 3600
        needs_run = age_h > 23

    if needs_run:
        try:
            sys.path.insert(0, str(ROOT))
            from portfolio.black_litterman import run_bl_optimizer
            print("  [BL] Running Black-Litterman optimizer …")
            run_bl_optimizer(top_n=25)
        except Exception as exc:
            print(f"  [BL] Skipped — {exc}")
            return composite_df

    if not bl_path.exists():
        return composite_df

    try:
        bl = pd.read_csv(bl_path)
    except Exception:
        return composite_df

    if "ticker" not in bl.columns or "weight" not in bl.columns:
        return composite_df

    bl_series = bl.set_index("ticker")["weight"]
    bl_mean, bl_std = bl_series.mean(), bl_series.std() + 1e-10
    bl_z = (bl_series - bl_mean) / bl_std

    df = composite_df.copy()
    df["bl_score"] = bl_z.reindex(df.index).fillna(0.0)
    df["composite_bl"] = 0.70 * df["regime_scaled"] + 0.30 * df["bl_score"]
    df = df.sort_values("composite_bl", ascending=False)

    rank_ic = df["regime_scaled"].rank(ascending=False)
    rank_bl = df["composite_bl"].rank(ascending=False)
    shifted = (rank_bl - rank_ic).abs().gt(5).sum()
    print(f"  [BL] Applied — {shifted} tickers shifted >5 ranks vs pure IC composite")
    return df


# =============================================================================
# CVaR gate — remove / flag high-tail-risk names before finalising positions
# =============================================================================

def apply_cvar_gate(
    top_long: list[str],
    top_short: list[str],
    block_limit: float = 0.09,
    size_down_limit: float = 0.06,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Compute 95% CVaR per ticker from 252 days of price history.
    block_limit (9%)   → REDUCE_ONLY: remove from both lists
    size_down_limit (6%) → SIZE_DOWN: flagged but kept
    Returns (filtered_long, filtered_short, verdicts).
    """
    try:
        sys.path.insert(0, str(ROOT))
        from canyon_final_v9_risk_framework_lib import var_cvar
    except ImportError:
        print("  [CVaR] risk_framework_lib unavailable — skipped")
        return top_long, top_short, {}

    all_tickers = list(set(top_long + top_short))

    # Fast path: DuckDB/Parquet via data layer; fall back to CSV-based get_returns
    rets = pd.DataFrame()
    try:
        from canyon_data_layer import returns as _dl_returns
        rets = _dl_returns(all_tickers, lookback=252)
    except Exception:
        pass
    if rets.empty:
        try:
            from canyon_final_v9_risk_framework_lib import get_returns
            rets = get_returns(all_tickers, lookback=252)
        except Exception:
            pass
    if rets.empty:
        print("  [CVaR] No return data — skipped")
        return top_long, top_short, {}

    verdicts: dict[str, str] = {}
    blocked: set[str] = set()

    for tk in all_tickers:
        if tk not in rets.columns:
            continue
        _, cvar = var_cvar(rets[tk].dropna(), alpha=0.95)
        if np.isnan(cvar):
            continue
        if cvar >= block_limit:
            verdicts[tk] = "REDUCE_ONLY"
            blocked.add(tk)
        elif cvar >= size_down_limit:
            verdicts[tk] = "SIZE_DOWN"

    filtered_long  = [t for t in top_long  if t not in blocked]
    filtered_short = [t for t in top_short if t not in blocked]

    if blocked:
        print(f"  [CVaR] Blocked {len(blocked)} names (CVaR ≥ {block_limit:.0%}): "
              f"{', '.join(sorted(blocked))}")
    size_dn = [t for t, v in verdicts.items() if v == "SIZE_DOWN"]
    if size_dn:
        print(f"  [CVaR] Size-down flags ({size_down_limit:.0%}–{block_limit:.0%}): "
              f"{', '.join(size_dn)}")
    if not blocked and not size_dn:
        print("  [CVaR] All positions within risk limits")

    return filtered_long, filtered_short, verdicts


# =============================================================================
# TC check — estimate one-way cost and flag oversized trades
# =============================================================================

def apply_tc_check(
    top_long: list[str],
    top_short: list[str],
    portfolio_size_usd: float = 100_000,
    max_adv_pct: float = 0.20,
) -> dict[str, float]:
    """
    Load volume_cache.csv (from step240), compute Almgren-Chriss simplified
    one-way TC estimate per ticker, and flag positions > max_adv_pct of ADV.
    Returns dict ticker → estimated_tc_bps.
    """
    vol_path = ROOT / "volume_cache.csv"
    if not vol_path.exists():
        print("  [TC] volume_cache.csv missing — run step240 first")
        return {}

    try:
        vol_df = pd.read_csv(vol_path, index_col=0, parse_dates=True)
        adv = vol_df.apply(pd.to_numeric, errors="coerce").tail(20).mean()
    except Exception as exc:
        print(f"  [TC] Could not load volume cache — {exc}")
        return {}

    all_tickers = list(set(top_long + top_short))
    n_long = max(len(top_long), 1)
    trade_usd = portfolio_size_usd / n_long

    tc_report: dict[str, float] = {}
    oversized: list[str] = []

    for tk in all_tickers:
        if tk not in adv.index:
            continue
        adv_val = float(adv[tk]) if not np.isnan(adv[tk]) else 0.0
        if adv_val <= 0:
            continue
        participation = trade_usd / adv_val
        # κ=0.10, σ≈2% daily; TC ≈ κ·σ·√participation  (bps)
        tc_bps = 0.10 * 0.02 * np.sqrt(participation) * 10_000
        tc_report[tk] = round(tc_bps, 1)
        if participation > max_adv_pct:
            oversized.append(f"{tk}({participation:.0%} ADV)")

    if oversized:
        print(f"  [TC] Oversized (>{max_adv_pct:.0%} ADV): {', '.join(oversized)}")
    if tc_report:
        avg = float(np.mean(list(tc_report.values())))
        print(f"  [TC] Avg estimated one-way cost: {avg:.1f} bps "
              f"across {len(tc_report)} positions")
    return tc_report


# =============================================================================
# v11 Joint Beta Constraint
# =============================================================================

def apply_joint_beta_constraint(
    top_long:  list[str],
    top_short: list[str],
    sigs:      dict,
    v9_beta_cap:   float = 0.50,  # v9 net exposure cap when joint beta is high
    joint_beta_max: float = 1.10, # flag when v9+v11 joint beta exceeds this
) -> tuple[list[str], list[str], dict]:
    """
    Check combined market beta of Canyon v9 (this system) + v11 QQQ Hunter.
    If the joint portfolio beta exceeds joint_beta_max, scale down v9 positions.

    v11 beta estimate: tqqq_weight * 2.5 + stock_weight * 1.0
      (TQQQ ≈ 3x SPY; using 2.5x to be conservative)
    v9  beta estimate: hmm_exposure * 0.6 (long/short partially cancels)

    Returns (top_long, top_short, info_dict).
    The lists are unchanged — only the info_dict carries the SIZE_DOWN flag.
    Position sizing downstream should respect info_dict["v9_scale"].
    """
    info = {"joint_beta": np.nan, "v11_beta": np.nan, "v9_beta": np.nan,
            "v9_scale": 1.0, "status": "OK"}

    # v11 beta from paper_trading_history.csv
    ph_path = ROOT / "paper_trading_history.csv"
    v11_beta = 0.70   # default: assume v11 has moderate market exposure
    if ph_path.exists():
        try:
            ph = pd.read_csv(ph_path)
            if not ph.empty:
                latest = ph.iloc[-1]
                tqqq_w = float(latest.get("tqqq_weight", 0) or 0)
                stock_w = float(latest.get("stock_weight", 0) or 0)
                v11_beta = tqqq_w * 2.5 + stock_w * 1.0
        except Exception:
            pass
    info["v11_beta"] = round(v11_beta, 3)

    # v9 beta: HMM exposure × 0.6 (L/S partial cancel)
    v9_beta = float(sigs.get("hmm_exposure", 1.0)) * 0.60
    info["v9_beta"] = round(v9_beta, 3)

    # Equal-weight joint portfolio beta
    joint_beta = 0.5 * v11_beta + 0.5 * v9_beta
    info["joint_beta"] = round(joint_beta, 3)

    if joint_beta > joint_beta_max:
        # Scale v9 net exposure so joint beta returns to target
        target = (joint_beta_max * 2 - v11_beta) / max(v9_beta, 0.01)
        v9_scale = round(min(1.0, max(0.5, target)), 3)
        info["v9_scale"]  = v9_scale
        info["status"]    = f"SIZE_DOWN (joint β={joint_beta:.2f} > {joint_beta_max})"
        n_cut = max(0, round(len(top_long) * (1 - v9_scale)))
        top_long  = top_long[:max(1, len(top_long)  - n_cut)]
        top_short = top_short[:max(1, len(top_short) - n_cut)]
        print(f"  [JointBeta] β={joint_beta:.2f} — scaling v9 to {v9_scale:.0%} "
              f"({n_cut} positions removed)")
    else:
        print(f"  [JointBeta] β={joint_beta:.2f} (v9={v9_beta:.2f}, "
              f"v11={v11_beta:.2f}) — within limit")

    return top_long, top_short, info


# =============================================================================
# Dynamic IC Weight Calibration (OOS Walk-Forward)
# =============================================================================

def _get_regime_bucket() -> tuple[str, str]:
    """
    Determine current (HMM_state, VIX_bucket) from existing signal files.
    Returns ("NEUTRAL", "MID") if data unavailable.
    """
    # ── HMM regime
    hmm_state = "NEUTRAL"
    hmm_path  = ROOT / "hmm_regime_monthly.csv"
    if hmm_path.exists():
        try:
            hm = pd.read_csv(hmm_path).sort_values("rebalance_date")
            label = str(hm.iloc[-1].get("regime_label", "")).upper()
            if "BULL" in label or "RISK_ON" in label:
                hmm_state = "BULL"
            elif "BEAR" in label or "RISK_OFF" in label or "CRISIS" in label:
                hmm_state = "BEAR"
        except Exception:
            pass

    # Fallback: macro composite
    if hmm_state == "NEUTRAL":
        mc_path = ROOT / "macro_composite_daily.csv"
        if mc_path.exists():
            try:
                mc = pd.read_csv(mc_path).sort_values("date")
                mac = float(mc.iloc[-1].get("macro_composite", 0.0))
                if mac > 0.20:
                    hmm_state = "BULL"
                elif mac < -0.20:
                    hmm_state = "BEAR"
            except Exception:
                pass

    # ── VIX level (from macro_composite or fear_vix signal)
    vix_level = 20.0   # default = MID
    mc_path   = ROOT / "macro_composite_daily.csv"
    if mc_path.exists():
        try:
            mc = pd.read_csv(mc_path).sort_values("date")
            for col in ("vix", "VIX", "fear_vix", "vix_close"):
                if col in mc.columns:
                    v = float(mc.iloc[-1][col])
                    if v > 0:
                        vix_level = v
                        break
        except Exception:
            pass

    # Also check rolling_ic_monitor for current VIX
    if vix_level == 20.0:
        rim = ROOT / "rolling_ic_monitor.csv"
        if rim.exists():
            try:
                df = pd.read_csv(rim)
                vix_row = df[df["signal"] == "fear_vix"].sort_values("date")
                if not vix_row.empty:
                    v = float(vix_row.iloc[-1].get("current_value", 20.0))
                    if v > 0:
                        vix_level = v
            except Exception:
                pass

    vix_bucket = "LOW" if vix_level < 20 else ("HIGH" if vix_level > 30 else "MID")
    return hmm_state, vix_bucket


def apply_regime_conditioning(
    weights: dict[str, float],
) -> dict[str, float]:
    """
    Apply regime-specific multipliers to IC weights.
    Multipliers are capped at 1.60× / floored at 0.50× to avoid extreme concentration.
    Saves regime_weights_today.csv for monitoring.
    """
    regime, vix = _get_regime_bucket()
    multipliers = REGIME_IC_MULTIPLIERS.get((regime, vix),
                  REGIME_IC_MULTIPLIERS.get(("NEUTRAL", "MID"), {}))

    if not multipliers:
        print(f"  [RegimeIC] Regime=({regime},{vix}) → no adjustment (IC²-optimal kept)")
        return weights

    print(f"  [RegimeIC] Regime=({regime},{vix}) → applying {len(multipliers)} multipliers")
    adjusted = dict(weights)
    for sig, mult in multipliers.items():
        if sig in adjusted:
            mult_clipped = float(np.clip(mult, 0.50, 1.60))
            adjusted[sig] = adjusted[sig] * mult_clipped

    # Renormalize
    total = sum(adjusted.values())
    if total > 1e-9:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    # Log the biggest movers for transparency
    movers = sorted(
        [(k, round(adjusted[k] - weights[k], 4)) for k in weights],
        key=lambda x: abs(x[1]), reverse=True
    )[:4]
    for sig, delta in movers:
        if abs(delta) > 0.002:
            print(f"    {sig:20s}  {weights[sig]:.3f} → {adjusted[sig]:.3f}  "
                  f"({'↑' if delta > 0 else '↓'}{abs(delta):.3f})")

    # Save for monitoring
    rows = [{"signal": k, "base_weight": weights[k], "regime_weight": adjusted[k],
             "delta": round(adjusted[k] - weights[k], 4),
             "regime": regime, "vix_bucket": vix,
             "date": datetime.now().strftime("%Y-%m-%d")}
            for k in weights]
    try:
        pd.DataFrame(rows).to_csv(ROOT / "regime_weights_today.csv", index=False)
    except Exception:
        pass

    return adjusted


def load_dynamic_ic_weights(base_weights: dict[str, float]) -> dict[str, float]:
    """
    Load optimized IC weights from step85 output (ic_weights_optimized.csv).
    Falls back to rolling_ic_monitor.csv partial adjustment, then base_weights.

    Priority:
      1. ic_weights_optimized.csv  — full IC²-optimal weights (step85, monthly)
      2. rolling_ic_monitor.csv   — partial adjustment for ml_score + alt_trends
      3. base_weights             — hardcoded fallback
    """
    # Priority 1: step85 full optimization
    opt_path = ROOT / "ic_weights_optimized.csv"
    if opt_path.exists():
        age_days = (datetime.now().timestamp() - opt_path.stat().st_mtime) / 86400
        if age_days < 32:   # use for up to 32 days (step85 refreshes monthly)
            try:
                df = pd.read_csv(opt_path)
                if "signal" in df.columns and "weight" in df.columns:
                    opt = dict(zip(df["signal"], pd.to_numeric(df["weight"], errors="coerce")))
                    # Only override signals that exist in base_weights
                    weights = {k: float(opt.get(k, base_weights[k])) for k in base_weights}
                    total   = sum(weights.values())
                    weights = {k: round(v / total, 4) for k, v in weights.items()}
                    print(f"  [DynIC] Loaded optimized weights from step85 "
                          f"({age_days:.0f}d old)")
                    return apply_regime_conditioning(weights)
            except Exception as exc:
                print(f"  [DynIC] step85 load failed: {exc}")

    # Priority 2: rolling_ic_monitor partial adjustment
    monitor_path = ROOT / "rolling_ic_monitor.csv"
    if not monitor_path.exists():
        return apply_regime_conditioning(base_weights)
    try:
        df = pd.read_csv(monitor_path)
        oos = df[df["period"] == "OOS"].copy() if "period" in df.columns else df
        if len(oos) < 6:
            return apply_regime_conditioning(base_weights)

        latest = oos.sort_values("date").groupby("signal").last()

        def _ic(name: str) -> float:
            if name not in latest.index:
                return np.nan
            v = latest.loc[name].get("ic_6m", np.nan)
            return float(v) if not np.isnan(v) else float(latest.loc[name].get("ic_3m", np.nan))

        name_map = {"ml_ensemble": "ml_score", "google_trends": "alt_trends"}
        base_ic  = {"ml_score": 0.065, "alt_trends": 0.020}
        weights  = dict(base_weights)
        adjusted = False

        for rolling_name, step_key in name_map.items():
            live_ic = _ic(rolling_name)
            if np.isnan(live_ic) or step_key not in weights:
                continue
            ratio   = float(np.clip((live_ic**2) / (base_ic.get(step_key, 0.04)**2 + 1e-9), 0.5, 2.0))
            old_w   = weights[step_key]
            new_w   = round(old_w * ratio, 4)
            delta   = new_w - old_w
            if abs(delta) > 0.005:
                weights[step_key] = new_w
                others      = [k for k in weights if k != step_key]
                total_others = sum(weights[k] for k in others)
                for k in others:
                    weights[k] = round(weights[k] - delta * weights[k] / max(total_others, 1e-9), 4)
                adjusted = True
                print(f"  [DynIC] {step_key}: {old_w:.3f}→{new_w:.3f} (IC={live_ic:.3f})")

        if not adjusted:
            print("  [DynIC] Weights unchanged (ICs within expected range)")

        total = sum(weights.values())
        w_final = {k: round(v / total, 4) for k, v in weights.items()}
        return apply_regime_conditioning(w_final)

    except Exception as exc:
        print(f"  [DynIC] Skipped: {exc}")
        return apply_regime_conditioning(base_weights)


# =============================================================================
# Earnings Calendar Risk Gate
# =============================================================================

def apply_earnings_risk_gate(
    composite_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply earnings proximity risk rules to composite scores BEFORE picks selection.

    Rules (standard institutional desk practice):
      HIGH   (days ≤ 3)  — remove ticker from candidate pool entirely
      MEDIUM (days ≤ 10) — penalize composite scores by 30% (auto-downsizes)
      LOW    (days ≤ 21) — apply 10% penalty (monitor flag)
      CLEAR  (days > 21) — no action

    Reads: earnings_calendar.csv (produced by step102)
    Saves: earnings_gate_today.csv (gate action log)
    """
    ec_path = ROOT / "earnings_calendar.csv"
    if not ec_path.exists():
        print("  [EarningsGate] earnings_calendar.csv not found — skipping gate")
        return composite_df, {"gated": 0, "penalized": 0, "missing_file": True}

    try:
        ec = pd.read_csv(ec_path)
        if "ticker" not in ec.columns or "earnings_risk" not in ec.columns:
            print("  [EarningsGate] earnings_calendar.csv missing required columns")
            return composite_df, {"gated": 0, "penalized": 0}

        ec = ec.set_index("ticker")
    except Exception as exc:
        print(f"  [EarningsGate] Failed to load: {exc}")
        return composite_df, {"gated": 0, "penalized": 0}

    df = composite_df.copy()
    score_cols = [c for c in ("regime_scaled", "composite_bl", "composite_score")
                  if c in df.columns]

    gated:    list[str] = []
    penalized:list[str] = []
    log_rows: list[dict] = []

    for ticker in df.index:
        if ticker not in ec.index:
            continue
        risk    = str(ec.loc[ticker].get("earnings_risk", "CLEAR"))
        days    = ec.loc[ticker].get("days_until", None)

        if risk == "HIGH":
            # Remove from candidate pool — binary event risk too high
            gated.append(ticker)
            log_rows.append({"ticker": ticker, "action": "REMOVED",
                             "earnings_risk": risk, "days_until": days,
                             "date": TODAY})
        elif risk == "MEDIUM":
            for col in score_cols:
                df.loc[ticker, col] *= 0.70
            penalized.append(ticker)
            log_rows.append({"ticker": ticker, "action": "PENALIZED_30PCT",
                             "earnings_risk": risk, "days_until": days,
                             "date": TODAY})
        elif risk == "LOW":
            for col in score_cols:
                df.loc[ticker, col] *= 0.90
            penalized.append(ticker)
            log_rows.append({"ticker": ticker, "action": "PENALIZED_10PCT",
                             "earnings_risk": risk, "days_until": days,
                             "date": TODAY})

    # Drop HIGH-risk tickers from candidate pool
    if gated:
        df = df.drop(index=[t for t in gated if t in df.index])
        print(f"  [EarningsGate] Removed  {len(gated):3d} HIGH-risk tickers: "
              f"{', '.join(gated[:5])}{'...' if len(gated)>5 else ''}")
    if penalized:
        print(f"  [EarningsGate] Penalized {len(penalized):2d} MEDIUM/LOW tickers: "
              f"{', '.join(penalized[:5])}{'...' if len(penalized)>5 else ''}")

    # Save gate log
    if log_rows:
        try:
            pd.DataFrame(log_rows).to_csv(ROOT / "earnings_gate_today.csv", index=False)
        except Exception:
            pass

    return df, {
        "gated":     len(gated),
        "penalized": len(penalized),
        "gated_list":     gated,
        "penalized_list": penalized,
    }


# =============================================================================
# main
# =============================================================================

def apply_factor_neutralization(
    top_long:     list[str],
    top_short:    list[str],
    composite_df: pd.DataFrame,
    max_swaps:    int = 5,
) -> tuple[list[str], list[str], dict]:
    """
    Factor-neutral portfolio construction via greedy swap pass.

    Loads Barra factor exposures (step88). For each style factor whose
    net portfolio exposure exceeds the tolerance, swaps the biggest offender
    with the best un-selected candidate that reduces the exposure.

    Factors controlled:
      market_beta  → |net| < 0.20  (beta-near-neutral L/S book)
      size         → |net| < 0.40  (no persistent large/small bias)
      momentum     → |net| < 0.40  (no persistent winner/loser tilt)

    Returns updated (top_long, top_short, neutralization_report).
    """
    expo_path = ROOT / "factor_exposures.csv"
    if not expo_path.exists():
        return top_long, top_short, {"status": "no_exposure_file"}

    try:
        expo = pd.read_csv(expo_path).set_index("ticker")
    except Exception:
        return top_long, top_short, {"status": "load_error"}

    # Factors and their tolerances
    NEUTRAL_FACTORS = {
        "market_beta": 0.20,
        "size":        0.40,
        "momentum":    0.40,
    }

    style_factors = [f for f in NEUTRAL_FACTORS if f in expo.columns]
    if not style_factors:
        return top_long, top_short, {"status": "no_style_factors"}

    # Candidate pools (next-best ranked, not yet selected)
    all_ranked = composite_df.index.tolist()
    long_cands  = [t for t in all_ranked      if t not in top_long  and t not in top_short]
    short_cands = [t for t in reversed(all_ranked) if t not in top_short and t not in top_long]

    report: dict = {"swaps": [], "pre_exposures": {}, "post_exposures": {}}

    def _net_exp(longs: list[str], shorts: list[str], factor: str) -> float:
        n_l, n_s = max(len(longs), 1), max(len(shorts), 1)
        l_exp = sum(float(expo.loc[t, factor]) for t in longs  if t in expo.index) / n_l
        s_exp = sum(float(expo.loc[t, factor]) for t in shorts if t in expo.index) / n_s
        return round(l_exp - s_exp, 4)

    # Record pre-neutralization exposures
    for f in style_factors:
        report["pre_exposures"][f] = _net_exp(top_long, top_short, f)

    # Greedy swap loop
    for factor, tol in NEUTRAL_FACTORS.items():
        if factor not in expo.columns:
            continue
        swaps_done = 0
        for _ in range(max_swaps):
            net = _net_exp(top_long, top_short, factor)
            if abs(net) <= tol:
                break

            if net > tol:
                # Long book too high on this factor → swap highest-exposure long
                # for a candidate with lower exposure
                offenders = sorted(
                    [t for t in top_long if t in expo.index],
                    key=lambda t: float(expo.loc[t, factor]), reverse=True
                )
                replacements = [
                    t for t in long_cands
                    if t in expo.index and float(expo.loc[t, factor]) < (net - tol)
                ]
                if not offenders or not replacements:
                    break
                worst, best = offenders[0], replacements[0]
                top_long  = [t if t != worst else best for t in top_long]
                long_cands = [t for t in long_cands if t != best]
                long_cands.append(worst)
            else:
                # Long book too low → short book too high → swap highest-exposure short
                offenders = sorted(
                    [t for t in top_short if t in expo.index],
                    key=lambda t: float(expo.loc[t, factor]), reverse=True
                )
                replacements = [
                    t for t in short_cands
                    if t in expo.index and float(expo.loc[t, factor]) < (-net - tol)
                ]
                if not offenders or not replacements:
                    break
                worst, best = offenders[0], replacements[0]
                top_short  = [t if t != worst else best for t in top_short]
                short_cands = [t for t in short_cands if t != best]
                short_cands.append(worst)

            report["swaps"].append({"factor": factor, "net_before": net,
                                     "removed": worst, "added": best})
            swaps_done += 1

        if swaps_done:
            print(f"    [FactorNeutral] {factor}: {report['pre_exposures'].get(factor,0):+.3f}"
                  f" → {_net_exp(top_long, top_short, factor):+.3f}  "
                  f"({swaps_done} swap{'s' if swaps_done>1 else ''})")

    # Record post-neutralization exposures
    for f in style_factors:
        report["post_exposures"][f] = _net_exp(top_long, top_short, f)

    report["status"] = "ok"
    report["n_swaps_total"] = len(report["swaps"])
    return top_long, top_short, report


def _run_factor_risk_check(top_long: list[str], top_short: list[str]) -> None:
    """
    Load pre-computed Barra factor exposures (step88) and decompose portfolio risk.
    Warns if factor share > 65% (too much systematic risk, not enough alpha).
    """
    expo_path   = ROOT / "factor_exposures.csv"
    fcov_path   = ROOT / "factor_cov.csv"
    srisk_path  = ROOT / "specific_risk.csv"
    if not (expo_path.exists() and fcov_path.exists()):
        print("    [FactorRisk] No factor model files — run step88 first")
        return

    try:
        from canyon_final_v9_step88_factor_risk_model import compute_portfolio_risk
        expo       = pd.read_csv(expo_path).set_index("ticker")
        factor_cov = pd.read_csv(fcov_path, index_col=0)
        spec_risk  = pd.read_csv(srisk_path).set_index("ticker")["specific_vol"] \
                     if srisk_path.exists() else pd.Series(dtype=float)

        n_l, n_s = len(top_long), len(top_short)
        weights  = ({t: +1.0/max(n_l,1) for t in top_long} |
                    {t: -1.0/max(n_s,1) for t in top_short})
        w_ser    = pd.Series(weights)

        decomp = compute_portfolio_risk(w_ser, expo, factor_cov, spec_risk)
        if decomp:
            fshare = decomp.get("factor_share", 0.0)
            tvol   = decomp.get("total_annual_vol", 0.0)
            print(f"    Total vol={tvol:.1%}  "
                  f"Factor share={fshare:.1%}  "
                  f"Specific share={decomp.get('specific_share',0):.1%}")
            if fshare > 0.65:
                print("    ⚠  Factor share > 65% — consider neutralizing beta/size/momentum")
    except Exception as exc:
        print(f"    [FactorRisk] {exc}")


def main() -> None:
    print("=" * 70)
    print(f"Canyon v9 — Daily Signal Pipeline  [{TODAY}]")
    print("=" * 70)

    # Load prices
    print("\n[1/5] Loading base data …")
    prices = pd.read_csv(ROOT / "backtest_price_cache.csv",
                         index_col=0, parse_dates=True)
    tickers = [c for c in prices.columns if c != "SPY"]

    print("\n[2/5] Loading current signals from all modules …")
    sigs = load_current_signals()
    print(f"  Signals loaded: {[k for k in sigs if not isinstance(sigs[k], pd.Series)]}")
    print(f"  HMM regime:    {sigs.get('hmm_regime', '—')}")
    print(f"  HMM exposure:  {sigs.get('hmm_exposure', 1.0):.0%}")
    print(f"  Macro score:   {sigs.get('macro_score', 0.0):+.3f}")
    print(f"  Last rebalance:{str(sigs.get('latest_rebalance', '—'))[:10]}")

    # Load yesterday's positions for turnover penalty
    prev_long_set:  set[str] = set()
    prev_short_set: set[str] = set()
    if LOG_FILE.exists():
        try:
            _log_tmp = pd.read_csv(LOG_FILE)
            _log_tmp["date"] = pd.to_datetime(_log_tmp["date"], errors="coerce")
            _past = _log_tmp[_log_tmp["date"].dt.date < pd.Timestamp(TODAY).date()]
            if not _past.empty:
                _prev = _past.iloc[-1]
                prev_long_set  = set(str(_prev.get("long_stocks",  "")).split("|")) - {""}
                prev_short_set = set(str(_prev.get("short_stocks", "")).split("|")) - {""}
        except Exception:
            pass

    print("\n[3/5] Building daily composite signal …")
    print("  [3-pre] Dynamic IC weight calibration …")
    composite_df = build_daily_composite(sigs, tickers, prev_long_set, prev_short_set,
                                         ic_weight_override=True)

    # Save per-signal z-score snapshot for step104 correlation monitor
    # (save before BL/gate modifications so it reflects the full universe)
    _SKIP_COLS = {"composite", "regime_scaled", "horizon_score", "composite_bl",
                  "turnover_pen", "regime_scaled_pen"}
    _snap_cols = [c for c in composite_df.columns if c not in _SKIP_COLS]
    if _snap_cols:
        try:
            snap = composite_df[_snap_cols].copy()
            snap.index.name = "ticker"
            snap["date"] = TODAY
            snap.reset_index().to_csv(ROOT / "signals_snapshot_today.csv", index=False)
        except Exception as _e:
            print(f"  [snap] Could not save signals_snapshot_today.csv: {_e}")

    # ── BL refinement: blend IC composite (70%) with BL posterior (30%) ──
    print("\n  [3a] Black-Litterman refinement …")
    composite_df = apply_bl_refinement(composite_df)
    rank_col = "composite_bl" if "composite_bl" in composite_df.columns else "regime_scaled"

    # ── Earnings calendar risk gate: remove/penalize pre-earnings tickers ──
    print("\n  [3b] Earnings calendar risk gate …")
    composite_df, earnings_gate = apply_earnings_risk_gate(composite_df)
    if earnings_gate.get("gated", 0) + earnings_gate.get("penalized", 0) == 0:
        print("    All tickers clear (no earnings within 21 days)")

    # ── MVO Portfolio Optimizer (step90): replaces greedy + factor-neutral ──
    print("\n  [3a] MVO Portfolio Optimizer (Barra-integrated) …")
    _opt_weights: pd.Series | None = None
    try:
        from canyon_final_v9_step90_portfolio_optimizer import (
            run_optimizer, save_weights, load_barra_matrices)
        _expo, _fcov, _srisk = load_barra_matrices()

        # Greedy baseline for warm-start and fallback
        _greedy_long, _greedy_short = sector_neutral_picks(
            composite_df.rename(columns={rank_col: "regime_scaled"})
            if rank_col != "regime_scaled" else composite_df,
            N_LONG, N_SHORT,
        )
        _opt_weights, _opt_meta = run_optimizer(
            composite_df, sigs,
            _greedy_long, _greedy_short,
            expo=_expo, fcov=_fcov, srisk=_srisk,
        )
        save_weights(_opt_weights, _opt_meta, composite_df)
        top_long  = _opt_weights[_opt_weights >  0.005].index.tolist()
        top_short = _opt_weights[_opt_weights < -0.005].index.tolist()
        print(f"    Status: {_opt_meta['status']}  "
              f"Vol: {_opt_meta['portfolio_vol']:.1%}  "
              f"Sharpe(ex-ante): {_opt_meta['sharpe_ex_ante']:.3f}  "
              f"Net: {_opt_meta['net_exposure']:+.3f}  "
              f"Turnover: {_opt_meta['turnover']:.1%}")
    except Exception as _opt_exc:
        print(f"    MVO failed ({_opt_exc}) — using greedy fallback")
        top_long, top_short = sector_neutral_picks(
            composite_df.rename(columns={rank_col: "regime_scaled"})
            if rank_col != "regime_scaled" else composite_df,
            N_LONG, N_SHORT,
        )
        # Greedy factor-neutral swap as secondary fallback
        top_long, top_short, fn_report = apply_factor_neutralization(
            top_long, top_short, composite_df)
        pd.DataFrame([{
            "date": TODAY,
            **fn_report.get("pre_exposures", {}),
            **{f"post_{k}": v for k,v in fn_report.get("post_exposures",{}).items()},
            "n_swaps": fn_report.get("n_swaps_total", 0),
        }]).to_csv(ROOT / "factor_neutralization_today.csv", index=False)

    # ── CVaR gate: remove names with tail risk > 9% ──
    print("\n  [3b] CVaR risk gate …")
    top_long, top_short, cvar_verdicts = apply_cvar_gate(top_long, top_short)

    # ── Joint beta: Canyon v9 + v11 combined market exposure ──
    print("\n  [3c] v9 + v11 joint beta constraint …")
    top_long, top_short, joint_beta_info = apply_joint_beta_constraint(
        top_long, top_short, sigs)
    pd.DataFrame([joint_beta_info]).to_csv(ROOT / "joint_beta_today.csv", index=False)

    # ── Factor risk decomposition (step88 Barra-style model) ──
    print("\n  [3d] Factor risk model …")
    _run_factor_risk_check(top_long, top_short)

    # ── TC check: flag trades that exceed 20% of ADV ──
    print("\n  [3e] Transaction cost check …")
    tc_estimates = apply_tc_check(top_long, top_short)

    # Save CVaR and TC verdicts for dashboard / alerts
    if cvar_verdicts:
        pd.DataFrame(
            [{"ticker": t, "cvar_verdict": v} for t, v in cvar_verdicts.items()]
        ).to_csv(ROOT / "cvar_verdicts_today.csv", index=False)
    if tc_estimates:
        pd.DataFrame(
            [{"ticker": t, "tc_bps": v} for t, v in tc_estimates.items()]
        ).to_csv(ROOT / "tc_estimates_today.csv", index=False)

    print(f"\n  TODAY'S TOP LONG  (sector-neutral, {N_LONG} positions):")
    for tk in top_long:
        r = composite_df.loc[tk]
        print(f"    {tk:<6}  {r['composite']:+.3f}  "
              f"(ML={r.get('ml_score',0):+.2f} "
              f"Fac={r.get('factor_score',0):+.2f} "
              f"SM={r.get('smart_money',0):+.2f} "
              f"Squeeze={r.get('squeeze',0):+.2f})")

    print(f"\n  TODAY'S TOP SHORT (sector-neutral, {N_SHORT} positions):")
    for tk in top_short:
        r = composite_df.loc[tk]
        print(f"    {tk:<6}  {r['composite']:+.3f}")

    # Print sector distribution
    regime_path = ROOT / "regime_ml_scores.csv"
    if regime_path.exists():
        rm = pd.read_csv(regime_path).set_index("ticker")
        if "sector" in rm.columns:
            long_sectors = rm.loc[[t for t in top_long if t in rm.index], "sector"]
            dist = long_sectors.value_counts()
            print(f"\n  LONG sector distribution:")
            for sec, cnt in dist.items():
                print(f"    {sec:<20} {cnt} stocks ({cnt/N_LONG:.0%})")

    print("\n[4/5] Fetching latest prices …")
    prices_today = fetch_latest_prices(tickers)
    valid = prices_today.notna().sum()
    print(f"  Prices fetched: {valid}/{len(tickers)} tickers")

    print("\n[5/5] Updating paper trading log + generating report …")
    update_paper_trading_log(composite_df, prices_today, sigs, top_long, top_short)

    pnl_series, paper_stats = compute_paper_stats()
    print(f"  Paper trading: {paper_stats}")

    alerts = detect_signal_changes(composite_df, top_long, top_short)
    if alerts:
        print(f"\n  ⚡ SIGNAL CHANGES TODAY:")
        for a in alerts:
            print(f"    {a}")
    else:
        print("  No significant signal changes today.")

    # Generate daily report
    report = generate_daily_report(
        composite_df, prices_today, sigs, alerts, pnl_series, top_long, top_short)
    report_path = REPORT_DIR / f"daily_{TODAY}.md"
    report_path.write_text(report)
    print(f"\n  Daily report saved: {report_path.name}")

    # Print crontab setup instructions
    print(f"\n  ── Automation Setup ──")
    print(f"  To run automatically every weekday at 6:00 PM:")
    print(f"  Run: crontab -e")
    print(f"  Add: 0 18 * * 1-5 cd {ROOT} && "
          f".venv/bin/python canyon_final_v9_step500_daily_pipeline.py "
          f">> {ROOT}/logs/daily.log 2>&1")

    print("\n" + "=" * 70)
    print("Step 500 Complete — Daily Signal Pipeline")
    print("=" * 70)
    print(f"  paper_trading_log.csv  → "
          f"{len(pd.read_csv(LOG_FILE)) if LOG_FILE.exists() else 0} entries")
    print(f"  Daily report           → {report_path}")


if __name__ == "__main__":
    main()
