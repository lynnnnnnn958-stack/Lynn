#!/usr/bin/env python3
"""
canyon_final_v9_step87_alpha_aggregator.py
==========================================
THE MISSING HUB of Canyon v9.

Reads every signal CSV produced by steps 66-85, normalises each
cross-sectionally, combines into one alpha score (0-100) per ticker,
converts to annualised expected-return inputs for the optimizer, and
writes today's final BUY list.

Outputs:
  alpha_scores.csv     ticker × alpha_score + signal breakdown + mu_override
  daily_picks.csv      top-N ranked buy list with suggested weights

Universe: all tickers present in regime_ml_scores.csv (~495 S&P 500 names).
Tickers missing from a particular signal receive a neutral score (50).
This ensures every stock is scored even when some signals are unavailable.

Usage:
  python canyon_final_v9_step87_alpha_aggregator.py
  python canyon_final_v9_step87_alpha_aggregator.py --top 30
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RF_ANNUAL   = 0.053     # risk-free rate
TOP_N       = 15        # default number of picks to output
ALPHA_SCALE = 0.25      # alpha range: score=100 → RF + 25%, score=0 → RF − 25%
POSITION_CAP = 0.15     # max weight per ticker in daily_picks

# Signal weights — must sum to 1.0
# Each entry: (csv_file, score_column, neutral_value, weight)
# higher score = more bullish for all signals
SIGNAL_CONFIG: list[tuple[str, str, float, float]] = [
    # name                      file                             col              neutral  weight
    ("regime_ml",   "regime_ml_scores.csv",           "predicted_score",   0.5,    0.28),
    ("quality",     "fundamental_quality_rank.csv",   "quality_score",     50.0,   0.18),
    ("revision",    "earnings_revision_scores.csv",   "revision_score",    50.0,   0.14),
    ("surprise",    "earnings_surprise_scores.csv",   "rank_sue",          50.0,   0.10),
    ("sentiment",   "finbert_sentiment.csv",          "rank_sentiment",    50.0,   0.09),
    ("squeeze",     "short_interest_scores.csv",      "rank_squeeze",      50.0,   0.08),
    ("insider",     "insider_signal_scores.csv",      "rank_insider",      50.0,   0.07),
    ("options",     "options_signals.csv",            "rank_options",      50.0,   0.06),
    # ML ensemble covers 38 core tickers but has the highest measured IC (0.44)
    # Tickers not covered receive neutral fill (50) — no penalty for being outside coverage
    ("ml_ensemble", "ml_signal_scores.csv",           "ensemble_score",    0.5,    0.10),
    # Institutional momentum: 4-component composite (CS 12-1m + 52wk-high + vol-scaled + residual)
    # Step 111 output. Half-life 20d (rebalance monthly). Regime-dampened internally in step111.
    # Note: momentum features are ALSO inputs to regime_ml/ml_ensemble (step77/66).
    # This standalone signal lets the aggregator explicitly weight momentum vs other factors.
    ("momentum",    "momentum_scores.csv",            "momentum_score",    50.0,   0.10),
    # Institutional L4: Sloan (1996) accrual anomaly — low accruals outperform (IC ~0.04-0.06)
    # Half-life 90d: quarterly earnings data, same window as quality signal.
    ("accruals",    "accrual_scores.csv",              "accrual_score",     50.0,   0.06),
    # Institutional L4: Piotroski (2000) F-score — 9-factor fundamental quality (IC ~0.03-0.05)
    # Half-life 90d: based on annual/quarterly financials.
    ("piotroski",   "piotroski_scores.csv",            "piotroski_score",   50.0,   0.05),
    # Institutional L5: Insider cluster score — multiple insiders buying = strong signal
    # Half-life 30d: insider activity windows, same as existing insider signal.
    ("ins_cluster", "insider_cluster_scores.csv",      "cluster_score_rank",50.0,   0.05),
]

# Signal half-lives — how long each signal remains predictive (trading days).
# W18: Empirical half-lives loaded from signal_halflife.csv when available;
# hardcoded defaults used as fallback.
#
# Used to compute exponential time-decay: decay = exp(-ln(2) * age_days / half_life).
# If decay < 0.20 (> ~2.3x half-life old), signal weight is capped at 20% of original.
_HALFLIFE_DEFAULTS: dict[str, float] = {
    "options":     2.0,    # 1-3 day validity; yesterday = nearly full weight
    "sentiment":   5.0,    # NLP sentiment: 3-7 day validity
    "squeeze":     7.0,    # short interest data updated weekly
    "ml_ensemble": 7.0,    # ML model: retrained weekly
    "regime_ml":   7.0,    # regime ML: retrained weekly
    "surprise":    10.0,   # earnings surprise: post-earnings drift 5-15 days
    "revision":    15.0,   # analyst revision: 2-4 week lag to consensus update
    "insider":     30.0,   # insider: SEC Form 4 lag + trading window ~30d
    "quality":     90.0,   # fundamentals: quarterly earnings validity
    "momentum":    20.0,
    "accruals":    90.0,
    "piotroski":   90.0,
    "ins_cluster": 30.0,
}


def _load_empirical_halflives() -> dict[str, float]:
    """
    W18: Load empirically measured half-lives from signal_halflife.csv.
    Falls back to hardcoded defaults for signals not in the CSV.
    """
    halflife_path = Path(__file__).parent / "signal_halflife.csv"
    empirical: dict[str, float] = {}
    if halflife_path.exists():
        try:
            import pandas as _pd_hl
            hl_df = _pd_hl.read_csv(halflife_path)
            if "signal" in hl_df.columns and "halflife_days" in hl_df.columns:
                for _, row in hl_df.iterrows():
                    sig_name = str(row["signal"]).lower()
                    hl_val   = float(row["halflife_days"])
                    if not np.isnan(hl_val) and hl_val > 0:
                        empirical[sig_name] = hl_val
                if empirical:
                    print(f"  [step87] W18: Loaded {len(empirical)} empirical half-lives from signal_halflife.csv")
        except Exception:
            pass  # fall through to defaults

    # Merge: empirical overrides defaults where available
    merged = dict(_HALFLIFE_DEFAULTS)
    for sig, hl in empirical.items():
        # Map price signal names to step87 signal names
        name_map = {
            "mom_1m": "momentum", "mom_3m": "momentum", "mom_12m": "momentum",
            "vol_21d": "squeeze",  # short squeeze uses vol signals
            "residual_mom": "momentum",
        }
        step87_name = name_map.get(sig, sig)
        if step87_name in merged:
            merged[step87_name] = hl
    return merged


SIGNAL_HALF_LIVES: dict[str, float] = _load_empirical_halflives()

# ─────────────────────────────────────────────────────────────────────────────
# Regime-conditional signal weights
# ─────────────────────────────────────────────────────────────────────────────

# Each market regime emphasises different signals.
# Keys match the "name" field in SIGNAL_CONFIG.
# Weights are normalised at runtime; they need NOT sum to 1.0.
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL": {
        # In bull + low-VIX, price signals are crowded and composite IC is near zero.
        # Options flow (UOA, net call premium) is the leading differentiator —
        # smart money positioning shows up in options 2-5 days before price moves.
        # Momentum is a strong positive factor in trending bull markets (J-T effect).
        # Trimmed regime_ml (0.24→0.20) and quality (0.15→0.13) to make room for momentum.
        "regime_ml":   0.20,   # trimmed: ML already uses momentum features internally
        "quality":     0.13,   # trimmed: quality premium is weak in full bull
        "revision":    0.14,
        "surprise":    0.10,
        "sentiment":   0.07,
        "squeeze":     0.08,
        "insider":     0.07,
        "options":     0.13,   # flow leads price in low-fear bull
        "ml_ensemble": 0.10,
        "momentum":    0.10,   # strong in bull: 12-1m winners keep winning
        "accruals":    0.04,   # weak in bull (quality premia suppressed)
        "piotroski":   0.03,
        "ins_cluster": 0.04,
    },
    "LATE_BULL": {
        # Late-cycle: quality premium rises, sentiment fades.
        # Options flow flags smart-money exits/hedges before reversal.
        # Momentum REDUCED (0.08 vs 0.14 in BULL): late-cycle momentum starts to
        # fade and can reverse violently at cycle peak. Step111 dampens at ×0.85.
        # Trimmed regime_ml (0.22→0.18) and quality (0.22→0.20) to make room.
        "regime_ml":   0.18,   # trimmed
        "quality":     0.20,   # still elevated but trimmed
        "revision":    0.14,
        "surprise":    0.09,
        "sentiment":   0.05,
        "squeeze":     0.05,
        "insider":     0.10,
        "options":     0.13,   # smart money hedges visible in flow
        "ml_ensemble": 0.10,
        "momentum":    0.06,   # reduced: late-cycle momentum unreliable; crash risk
        "accruals":    0.04,
        "piotroski":   0.03,
        "ins_cluster": 0.05,   # insider cluster buying near cycle peaks is informative
    },
    "BEAR": {
        # Bear markets: quality premium is highest.
        # Positive revisions/surprises are rare but powerful.
        # Contrarian: ignore sentiment (usually wrong at bottoms).
        # Insider buying near lows is the most reliable signal.
        # Momentum set to MINIMAL (0.04): bear→bull transitions produce the most
        # violent momentum crashes (prior-year losers surge first and fastest).
        # Step111 also dampens internally at ×0.45 in BEAR regime.
        # Trimmed regime_ml (0.15→0.11) and quality (0.28→0.26) to make room.
        "regime_ml":   0.11,   # trimmed
        "quality":     0.22,   # still dominant; quality premium largest in bear
        "revision":    0.16,
        "surprise":    0.10,
        "sentiment":   0.04,
        "squeeze":     0.05,
        "insider":     0.10,
        "options":     0.08,
        "ml_ensemble": 0.10,
        "momentum":    0.04,   # minimal: momentum crashes hardest in bear recoveries
        "accruals":    0.08,   # high weight in bear: accrual anomaly largest in downturns
        "piotroski":   0.07,   # F-score most predictive in bear (sorting quality stocks)
        "ins_cluster": 0.05,   # insider buying at bear lows is a strong signal
    },
    "SIDEWAYS": {
        # Range-bound market: no single signal dominates.
        # Quality and regime ML still lead.
        # Momentum moderate (0.10): cross-sectional momentum still works in flat
        # markets (top decile still beats bottom decile) but with reduced magnitude.
        # Trimmed regime_ml (0.20→0.16) and quality (0.20→0.18) to make room.
        "regime_ml":   0.16,   # trimmed
        "quality":     0.16,   # trimmed
        "revision":    0.13,
        "surprise":    0.09,
        "sentiment":   0.07,
        "squeeze":     0.07,
        "insider":     0.07,
        "options":     0.07,
        "ml_ensemble": 0.10,
        "momentum":    0.08,   # moderate: J-T momentum works in range-bound too
        "accruals":    0.06,
        "piotroski":   0.05,
        "ins_cluster": 0.05,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-sector regime: sector ETF → SMA-based BULL / BEAR / SIDEWAYS
# ─────────────────────────────────────────────────────────────────────────────

# GICS sector name → representative sector ETF
SECTOR_ETF_MAP: dict[str, str] = {
    "Information Technology":  "XLK",
    "Technology":              "XLK",
    "Semiconductors":          "XLK",
    "Energy":                  "XLE",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Industrials":             "XLI",
    "Consumer Staples":        "XLP",
    "Consumer Discretionary":  "XLY",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
    "Communication Services":  "XLC",
    "Communication":           "XLC",
}


def _get_sector_regimes() -> dict[str, str]:
    """
    Compute per-sector regime from sector ETF prices.
    Returns {sector_name: regime} where regime ∈ {BULL, BEAR, SIDEWAYS}.

    Algorithm:
      20d SMA > 50d SMA  AND  price > 50d SMA  →  BULL
      20d SMA < 50d SMA  AND  price < 50d SMA  →  BEAR
      otherwise                                →  SIDEWAYS
    """
    results: dict[str, str] = {}
    for price_path in (ROOT / "backtest_price_cache.csv",
                       ROOT / "sp500_price_cache.csv"):
        if not price_path.exists():
            continue
        try:
            prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
            prices = prices.sort_index()
            seen: set[str] = set()
            for sector, etf in SECTOR_ETF_MAP.items():
                if sector in seen or etf not in prices.columns:
                    continue
                s = prices[etf].dropna()
                if len(s) < 55:
                    continue
                price_now = float(s.iloc[-1])
                sma20     = float(s.iloc[-20:].mean())
                sma50     = float(s.iloc[-50:].mean())
                if sma20 > sma50 and price_now > sma50:
                    results[sector] = "BULL"
                elif sma20 < sma50 and price_now < sma50:
                    results[sector] = "BEAR"
                else:
                    results[sector] = "SIDEWAYS"
                seen.add(sector)
            if results:
                break
        except Exception:
            continue
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Directional options signal builder
# ─────────────────────────────────────────────────────────────────────────────

def _load_options_directional(
    universe_set: set[str],
    tickers: list[str],
) -> pd.Series:
    """
    Build a directionally-correct options alpha signal (0-100, 100 = most bullish flow).

    THE CORE FIX — rank_options treats ALL unusual options activity as bullish.
    This is fundamentally wrong:
      pcr_vol=10 + uoa_bear_flag=1  →  rank_options=100  (was: "STRONG BUY")
      pcr_vol=0.3 + net_call_premium>0 →  rank_options=100  (correct: BUY signal)

    This function separates bull-flow and bear-flow:

    Bull composite  (high score = call-side money flow):
      + net_call_premium rank   (positive $ into calls)
      + (100 - pcr_vol rank)    (low put/call ratio = call dominance)
      + uoa_flag=1 AND NOT uoa_bear_flag  (unusual CALL activity)
      + negative iv_skew        (calls more expensive than puts)

    Bear composite  (high score = put-side money flow):
      + pcr_vol rank             (high put/call ratio)
      + uoa_bear_flag            (unusual PUT activity flagged)
      + negative net_call_prem   (net $ into puts)
      + positive iv_skew         (puts more expensive = fear premium)

    Final = rank(bull_norm - 0.5 * bear_norm) → 0-100
    The 0.5 factor on bear slightly softens the penalty: don't punish a stock
    just because it has active two-sided options (straddle positioning).
    """
    path = ROOT / "options_signals.csv"
    if not path.exists():
        return pd.Series(50.0, index=tickers)

    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            return pd.Series(50.0, index=tickers)

        # Keep most recent date snapshot
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            dc    = date_cols[0]
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            df    = df[df[dc] == df[dc].max()]

        df = df.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")

        if len(df) < 5:
            # Too sparse — fall back to existing rank_options
            return _load_signal("options_signals.csv", "rank_options", 50.0, universe_set)

        bull = pd.Series(0.0, index=df.index)
        bear = pd.Series(0.0, index=df.index)

        if "net_call_premium" in df.columns:
            ncp   = df["net_call_premium"].fillna(0)
            bull += ncp.rank(pct=True) * 40.0       # positive $-flow → bullish
            bear += (-ncp).rank(pct=True) * 40.0    # negative $-flow → bearish

        if "pcr_vol" in df.columns:
            pcr   = df["pcr_vol"].clip(0, 10).fillna(1)
            bull += (1 - pcr.rank(pct=True)) * 30.0   # low PCR = calls > puts = bullish
            bear += pcr.rank(pct=True) * 30.0           # high PCR = puts heavy = bearish

        if "uoa_bear_flag" in df.columns and "uoa_flag" in df.columns:
            bull_uoa = ((df["uoa_flag"].fillna(0) == 1) &
                        (df["uoa_bear_flag"].fillna(0) == 0)).astype(float)
            bear_uoa = df["uoa_bear_flag"].fillna(0).astype(float)
            bull    += bull_uoa * 20.0
            bear    += bear_uoa * 25.0    # put flag weighted slightly higher (conviction)
        elif "uoa_flag" in df.columns:
            # Can't separate direction — split credit
            uoa   = df["uoa_flag"].fillna(0).astype(float)
            bull += uoa * 10.0

        if "iv_skew" in df.columns:
            skew  = df["iv_skew"].fillna(0)
            bull += (-skew).rank(pct=True) * 10.0   # negative skew = call demand
            bear += skew.rank(pct=True) * 10.0        # positive skew = put fear

        # Normalise each composite to 0-100 range
        bull_max = float(bull.max())
        bear_max = float(bear.max())
        bull_n   = (bull / bull_max * 100) if bull_max > 0 else pd.Series(50.0, index=df.index)
        bear_n   = (bear / bear_max * 100) if bear_max > 0 else pd.Series(0.0,  index=df.index)

        # Net directional: bull dominates when call flow is strong; bear penalises
        net    = bull_n - 0.5 * bear_n      # approximate range −50 to +100
        result = net.rank(pct=True) * 100   # cross-sectional percentile → 0-100

        result   = result.reindex(tickers).fillna(50.0)
        n_bull_f = int((result > 60).sum())
        n_bear_f = int((result < 40).sum())
        print(f"  [opts-dir] directional: {n_bull_f} bull-flow  {n_bear_f} bear-flow  "
              f"(rank_options ← directional net-flow composite)")
        return result

    except Exception as e:
        print(f"  [opts-dir] error ({e}) — falling back to rank_options")
        return _load_signal("options_signals.csv", "rank_options", 50.0, universe_set)


def _get_current_regime() -> str:
    """
    Read the current market regime from persisted files.

    Priority:
      1. regime_current.json  (written by Step 76, most recent)
      2. Last row of regime_history.csv  (Step 76 historical log)
      3. Default → "BULL"
    """
    import json as _json

    # 1. regime_current.json
    json_path = ROOT / "regime_current.json"
    if json_path.exists():
        try:
            data = _json.loads(json_path.read_text())
            r = str(data.get("regime", "")).upper()
            if r in REGIME_WEIGHTS:
                return r
        except Exception:
            pass

    # 2. regime_history.csv — last row
    hist_path = ROOT / "regime_history.csv"
    if hist_path.exists():
        try:
            df = pd.read_csv(hist_path)
            if "regime" in df.columns and not df.empty:
                r = str(df["regime"].iloc[-1]).upper()
                if r in REGIME_WEIGHTS:
                    return r
        except Exception:
            pass

    return "BULL"   # conservative default

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cross_rank(series: pd.Series) -> pd.Series:
    """Normalise to [0, 100] using cross-sectional percentile rank."""
    ranked = series.rank(pct=True, na_option="keep") * 100.0
    return ranked


def _load_signal(
    filename: str,
    score_col: str,
    neutral: float,
    universe: set[str],
) -> pd.Series:
    """
    Load one signal CSV, return a Series indexed by ticker (0-100 normalised).
    Missing tickers get the neutral value (before normalisation).
    """
    path = ROOT / filename
    if not path.exists():
        print(f"  [SKIP] {filename} not found")
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns or score_col not in df.columns:
            print(f"  [SKIP] {filename} missing required columns (ticker, {score_col})")
            return pd.Series(dtype=float)

        # If there is a date column, keep only the most recent snapshot
        date_cols = [c for c in df.columns if "date" in c.lower() or "rebalance" in c.lower()]
        if date_cols:
            dc = date_cols[0]
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            latest = df[dc].max()
            df = df[df[dc] == latest]

        # Deduplicate by ticker (keep last)
        df = df.drop_duplicates(subset=["ticker"], keep="last")
        df = df.set_index("ticker")[score_col].astype(float)

        # Cross-sectional rank on the available tickers (→ 0-100 percentile)
        df = _cross_rank(df)

        # Tickers not in this signal get exactly 50 (neutral / median rank)
        df = df.reindex(list(universe)).fillna(50.0)
        return df

    except Exception as e:
        print(f"  [ERROR] loading {filename}: {e}")
        return pd.Series(dtype=float)


def _load_universe() -> tuple[list[str], pd.DataFrame]:
    """
    Returns (ticker_list, regime_df).
    Universe = all tickers in regime_ml_scores.csv (≈495 S&P 500 names).
    Falls back to sp500_price_cache.csv column names if regime file missing.
    """
    regime_path = ROOT / "regime_ml_scores.csv"
    if regime_path.exists():
        try:
            df = pd.read_csv(regime_path)
            if "ticker" in df.columns:
                df = df.drop_duplicates(subset=["ticker"], keep="last")
                tickers = df["ticker"].dropna().tolist()
                print(f"  [universe] {len(tickers)} tickers from regime_ml_scores.csv")
                return tickers, df
        except Exception as e:
            print(f"  [WARN] regime_ml_scores.csv error: {e}")

    # Fallback: sp500 price cache column names
    price_cache = ROOT / "sp500_price_cache.csv"
    if price_cache.exists():
        try:
            cols = pd.read_csv(price_cache, nrows=0).columns.tolist()
            tickers = [c for c in cols if c not in ("", "Date", "Unnamed: 0")]
            print(f"  [universe] {len(tickers)} tickers from sp500_price_cache.csv (fallback)")
            return tickers, pd.DataFrame()
        except Exception:
            pass

    raise RuntimeError("Cannot load universe: neither regime_ml_scores.csv nor "
                       "sp500_price_cache.csv found.")


# ─────────────────────────────────────────────────────────────────────────────
# Score trend helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score_trend(tickers: list[str]) -> pd.Series:
    """
    Return a Series of trend arrows indexed by ticker, derived from
    alpha_score_history.csv (appended each run by write_outputs).

    Labels:
      ▲▲  Rising strongly  (Δscore > +10 over window)
      ▲   Rising           (Δscore +3 to +10)
      →   Stable           (Δscore within ±3)
      ▼   Falling          (Δscore −3 to −10)
      ▼▼  Falling strongly (Δscore < −10)
      ★   New entrant      (not yet in history)
    """
    hist_path = ROOT / "alpha_score_history.csv"
    default   = pd.Series("★", index=tickers)

    if not hist_path.exists():
        return default

    try:
        hist = pd.read_csv(hist_path)
        if not {"date", "ticker", "alpha_score"}.issubset(hist.columns):
            return default

        dates = sorted(hist["date"].unique())
        if len(dates) < 2:
            return default

        # Use last 5 trading days
        recent = dates[-5:]
        hist   = hist[hist["date"].isin(recent)]
        pivot  = hist.pivot_table(index="date", columns="ticker",
                                  values="alpha_score", aggfunc="last")

        trend: dict[str, str] = {}
        for t in tickers:
            if t not in pivot.columns:
                trend[t] = "★"
                continue
            col = pivot[t].dropna()
            if len(col) < 2:
                trend[t] = "★"
                continue
            delta = float(col.iloc[-1]) - float(col.iloc[0])
            if   delta >  10:  trend[t] = "▲▲"
            elif delta >   3:  trend[t] = "▲"
            elif delta < -10:  trend[t] = "▼▼"
            elif delta <  -3:  trend[t] = "▼"
            else:              trend[t] = "→"

        return pd.Series(trend).reindex(tickers).fillna("★")

    except Exception as e:
        print(f"  [WARN] score trend: {e}")
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_alpha(top_n: int = TOP_N) -> tuple:
    """
    Main entry point.
    Returns (alpha_df, picks_df, reduce_df, current_regime, eff_weights).
    """
    print("\n" + "=" * 60)
    print("Canyon v9  Step 87 — Alpha Aggregator")
    print("=" * 60)

    # ── 1. Universe ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading universe …")
    tickers, regime_df = _load_universe()
    universe_set = set(tickers)

    # ── 1.5. Regime detection → conditional signal weights ───────────────────
    current_regime = _get_current_regime()
    regime_w       = REGIME_WEIGHTS.get(current_regime, REGIME_WEIGHTS["BULL"])
    # Build per-signal effective weights (regime overrides SIGNAL_CONFIG base)
    eff_weights    = {name: regime_w.get(name, base_w)
                      for name, _, _, _, base_w in SIGNAL_CONFIG}
    top3_w = sorted(regime_w.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  [regime] {current_regime} detected  →  top-3 signals: "
          + ", ".join(f"{k}={v:.0%}" for k, v in top3_w))

    # ── 1.6. IC-calibrated weight overlay (Step 94, optional) ───────────────
    # If signal_weights.json exists and is < 7 days old, blend IC multipliers
    # on top of the regime weights.  This means signal weights are now doubly
    # adaptive: regime-conditional AND empirically IC-weighted.
    _sw_path = ROOT / "signal_weights.json"
    if _sw_path.exists():
        try:
            import json as _json
            import time as _time
            _age_days = (_time.time() - _sw_path.stat().st_mtime) / 86400
            if _age_days <= 7:
                _sw = _json.loads(_sw_path.read_text())
                _mults = _sw.get("ic_multipliers", {})
                if _mults:
                    # Apply multiplier — no renormalise here; single norm at combination
                    for name in eff_weights:
                        eff_weights[name] *= _mults.get(name, 1.0)
                    _boosted = sorted(
                        [(k, _mults.get(k, 1.0)) for k in eff_weights],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                    print(f"  [ic-cal] signal_weights.json applied  "
                          f"(age={_age_days:.1f}d)  "
                          f"top-boosted: "
                          + ", ".join(f"{k}×{v:.2f}" for k, v in _boosted))
                else:
                    print("  [ic-cal] signal_weights.json found but no ic_multipliers — skipped")
            else:
                print(f"  [ic-cal] signal_weights.json stale ({_age_days:.1f}d) — skipped")
        except Exception as _e:
            print(f"  [ic-cal] could not read signal_weights.json: {_e}")

    # ── 1.7. VIX regime — dynamic weight tilt ────────────────────────────────
    # High VIX (fear): shift toward Quality + Revision (defensive fundamentals).
    # Low VIX (calm):  shift toward Squeeze + Options (vol-selling, momentum plays).
    # Source priority: macro_signals.json → regime_current.json → no adjustment.
    _vix_level: float | None = None
    for _vix_src in [ROOT / "macro_signals.json", ROOT / "regime_current.json"]:
        if _vix_src.exists():
            try:
                import json as _json
                _vd = _json.loads(_vix_src.read_text())
                _v = _vd.get("vix") or _vd.get("vix_spot")
                if _v is not None:
                    _vix_level = float(_v)
                    break
            except Exception:
                pass

    if _vix_level is not None:
        if _vix_level > 30:
            # Fear regime: quality & revision are most predictive.
            # Momentum crushed: VIX>30 is often the start of a crash; step111 also
            # applies internal dampening (×0.60 for VIX>25, ×0.45 BEAR regime).
            _vix_mults = {"quality": 1.40, "revision": 1.25, "surprise": 1.10,
                          "sentiment": 0.65, "squeeze": 0.70, "options": 0.75,
                          "momentum": 0.40}  # ← halved again on top of step111 dampening
            _vix_label = f"FEAR (VIX={_vix_level:.1f})"
        elif _vix_level < 18:
            # Calm/complacent regime (VIX < 18): options flow and squeeze are most
            # reliable — smart money positions aggressively when fear is absent.
            # Momentum gets a boost: low-volatility bull markets are where J-T effect
            # is strongest (AQR data shows momentum IC highest when VIX < 15-18).
            _vix_mults = {"squeeze": 1.35, "options": 1.50, "regime_ml": 1.10,
                          "sentiment": 1.10, "quality": 0.85,
                          "momentum": 1.30}  # ← boosted: J-T effect strongest in calm markets
            _vix_label = f"CALM (VIX={_vix_level:.1f})"
        else:
            _vix_mults = {}
            _vix_label = f"NORMAL (VIX={_vix_level:.1f})"

        if _vix_mults:
            for name in eff_weights:
                eff_weights[name] *= _vix_mults.get(name, 1.0)
            # no renormalise here — single norm at combination step
            _vix_top = sorted(_vix_mults.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [vix-tilt] {_vix_label}  boosted: "
                  + ", ".join(f"{k}×{v:.2f}" for k, v in _vix_top))
        else:
            print(f"  [vix-tilt] {_vix_label} — no weight adjustment")
    else:
        print("  [vix-tilt] VIX not available — skipped")

    # ── 1.8. Macro stress overlay ────────────────────────────────────────────
    # Detects regime-level macro stress from VIX term structure, credit spreads,
    # and yield curve.  In STRESSED mode, the options signal is recomputed in
    # DEFENSIVE mode — stocks where smart money is buying PUTS are penalized.
    #
    # Why this matters:
    #   In clear macro events (tariffs, Fed shock, credit event), the bear-options
    #   flow shows up BEFORE price moves.  rank_options treats any high-activity
    #   options as bullish — wrong when pcr_vol=10 and uoa_bear_flag=1.
    #
    # Stress score (0 = calm, 3+ = elevated, 6+ = stressed):
    #   VIX contango ratio > 1.15  : +1  (forward fear building)
    #   VIX contango ratio > 1.30  : +1  (strong forward fear)
    #   Yield curve inverted < 0   : +2
    #   Credit signal WIDENING     : +2
    #   macro_score < 40           : +2
    #   macro_score < 25           : +2  (additive with above)
    _macro_stress_score  = 0
    _macro_stress_label  = "NORMAL"
    _options_bear_signal: dict[str, float] = {}   # ticker → 0-100 (100=most bearish flow)

    try:
        import json as _json
        _ms_path = ROOT / "macro_signals.json"
        if _ms_path.exists():
            _ms = _json.loads(_ms_path.read_text())

            _vts   = float(_ms.get("vix_term_structure", 1.0))
            _yc    = float(_ms.get("yield_curve", 1.0))
            _credit= str(_ms.get("credit_signal", "NEUTRAL")).upper()
            _mscore= float(_ms.get("macro_score", 50))

            if _vts > 1.15:
                _macro_stress_score += 1
            if _vts > 1.30:
                _macro_stress_score += 1
            if _yc < 0:
                _macro_stress_score += 2
            if "WIDEN" in _credit or _credit in ("STRESS", "WIDE"):
                _macro_stress_score += 2
            if _mscore < 40:
                _macro_stress_score += 2
            if _mscore < 25:
                _macro_stress_score += 2

            if _macro_stress_score >= 6:
                _macro_stress_label = "STRESSED"
            elif _macro_stress_score >= 3:
                _macro_stress_label = "ELEVATED"
            else:
                _macro_stress_label = "NORMAL"

            print(f"  [macro] stress={_macro_stress_score}  ({_macro_stress_label})  "
                  f"VTS={_vts:.3f}  YC={_yc:.3f}  credit={_credit}  macro_score={_mscore:.0f}")

            # In ELEVATED or STRESSED: build directional bear-options signal
            # Stocks with heavy put buying should score LOW (avoid them)
            if _macro_stress_score >= 3:
                _opts_path = ROOT / "options_signals.csv"
                if _opts_path.exists():
                    try:
                        _odf = pd.read_csv(_opts_path)
                        if "ticker" in _odf.columns:
                            # Bear flow score: normalised composite of put-side signals
                            # pcr_vol   : put/call ratio (high = heavy put buying)
                            # uoa_bear_flag : unusual PUT activity detected
                            # net_call_premium < 0 : net $ flow into puts
                            _odf["_pcr_norm"] = _odf["pcr_vol"].clip(0, 5) / 5.0 \
                                if "pcr_vol" in _odf.columns else 0.0
                            _odf["_bear_flag"] = _odf["uoa_bear_flag"].fillna(0) \
                                if "uoa_bear_flag" in _odf.columns else 0.0
                            _odf["_net_put"]   = (_odf["net_call_premium"] < 0).astype(float) \
                                if "net_call_premium" in _odf.columns else 0.0
                            _odf["_bear_flow"] = (
                                0.50 * _odf["_pcr_norm"] +
                                0.30 * _odf["_bear_flag"] +
                                0.20 * _odf["_net_put"]
                            ) * 100.0   # 0-100, higher = more bearish options flow

                            # Invert to alpha: high bear flow → LOW alpha
                            _odf["_bear_options_alpha"] = 100.0 - _odf["_bear_flow"]
                            _options_bear_signal = dict(zip(
                                _odf["ticker"].astype(str),
                                _odf["_bear_options_alpha"].fillna(50.0)
                            ))
                            _n_bear = int((_odf["_bear_flag"] == 1).sum())
                            print(f"  [macro] DEFENSIVE options signal built  "
                                  f"({_n_bear} tickers with active PUT flow flagged as AVOID)")
                    except Exception as _oe:
                        print(f"  [macro] options bear signal error: {_oe}")

            # Weight adjustment in ELEVATED/STRESSED
            if _macro_stress_score >= 3:
                # Reduce momentum-following; raise options & quality (defensive)
                _stress_mult = 1.0 + 0.15 * min(_macro_stress_score - 2, 4) / 4.0
                eff_weights["options"]   = eff_weights.get("options", 0.06) * _stress_mult
                eff_weights["quality"]   = eff_weights.get("quality", 0.15) * (1.0 + 0.10)
                eff_weights["sentiment"] = eff_weights.get("sentiment", 0.07) * 0.60
                print(f"  [macro] {_macro_stress_label}: options×{_stress_mult:.2f}  "
                      f"quality×1.10  sentiment×0.60")
    except Exception as _me:
        print(f"  [macro] stress overlay skipped: {_me}")

    # ── 1.9. Signal time-decay multipliers ────────────────────────────────────
    # Different signals have very different validity windows:
    #   options flow from 3 days ago ≈ worthless (market has already moved)
    #   fundamental quality from last quarter ≈ fully valid
    #
    # Apply exponential decay: decay = exp(-ln(2) * age_days / half_life)
    # If decay < 0.20 (> 2.3× half-life old), clamp to 0.10 to prevent the
    # signal from disappearing entirely (floor ensures the CSV is still loaded).
    #
    # age_days = time since signal CSV was last modified (proxy for data vintage)
    import math   as _math
    import time   as _time2
    _decay_log: list[str] = []

    for _dname, _dfile, _dcol, _dneutral, _dbase in SIGNAL_CONFIG:
        if _dname not in eff_weights:
            continue
        _half  = SIGNAL_HALF_LIVES.get(_dname, 30.0)
        _dpath = ROOT / _dfile
        if _dpath.exists():
            _age   = (_time2.time() - _dpath.stat().st_mtime) / 86400
            _decay = _math.exp(-_math.log(2) * _age / _half)
            _decay = max(_decay, 0.10)    # floor: never below 10%
            if _decay < 0.99:             # only report if actually decayed
                eff_weights[_dname] *= _decay
                if _decay < 0.50:
                    _decay_log.append(f"{_dname}×{_decay:.2f}(age={_age:.1f}d)")

    if _decay_log:
        print(f"  [decay] stale signals (decay<0.50): {', '.join(_decay_log)}")
    else:
        print("  [decay] all signal files fresh — no weight reduction applied")

    # ── 1.10. Per-sector regime detection ────────────────────────────────────
    # Market regime is NOT uniform. XLK can be BULL while XLE is BEAR on the
    # same day.  Detect per-sector regime from sector ETF 20d/50d SMA crossover.
    # This feeds section 3.8 where per-ticker alpha is adjusted ±3pts based on
    # whether the ticker's sector ETF is in a bull or bear trend.
    _sector_regimes: dict[str, str] = _get_sector_regimes()
    if _sector_regimes:
        _bull_secs = [s for s, r in _sector_regimes.items() if r == "BULL"]
        _bear_secs = [s for s, r in _sector_regimes.items() if r == "BEAR"]
        print(f"  [sector-regime] {len(_sector_regimes)} sectors detected  →  "
              f"BULL: {len(_bull_secs)}  BEAR: {len(_bear_secs)}  "
              f"SIDEWAYS: {len(_sector_regimes)-len(_bull_secs)-len(_bear_secs)}")
        if _bear_secs:
            print(f"    Bear sectors: {', '.join(_bear_secs[:6])}")
    else:
        print("  [sector-regime] ETF data unavailable — skipped")

    # ── 2. Load individual signals ────────────────────────────────────────────
    # FIX 1 — neutral-fill bias:
    #   • Entire signal missing → drop signal, redistribute its weight (signal skipped)
    #   • Ticker missing within a signal → fill with that signal's cross-sectional
    #     TRIMMED MEAN (not 50).  50 is wrong when a signal's average is e.g. 62.
    # FIX 2 — double normalization:
    #   All regime / IC / VIX multipliers have already been applied above.  We do
    #   ONE normalization here, not one per overlay pass.
    print("\n[2/5] Loading signals …")
    signal_series: dict[str, pd.Series] = {}
    signal_weights_used: dict[str, float] = {}

    for name, filename, col, neutral, base_w in SIGNAL_CONFIG:
        weight = eff_weights.get(name, base_w)
        if weight == 0:
            continue

        # ── Options signal: ALWAYS directionally-corrected ──────────────────
        # BUG FIXED: rank_options treated ALL unusual options activity as bullish.
        # High pcr_vol=10 + uoa_bear_flag=1 → rank_options=100 → "STRONG BUY".
        # This is backwards: heavy put buying is a BEARISH signal.
        #
        # Now: two modes based on macro stress level:
        #   MACRO STRESSED (score≥3): use defensive bear-flow alpha from section 1.8
        #     (stocks with heavy PUT buying score LOW → avoid)
        #   NORMAL: use directional net-flow composite from _load_options_directional()
        #     (properly separates bull call-flow vs bear put-flow)
        # In BOTH cases, rank_options is no longer used raw.
        if name == "options":
            if _options_bear_signal:
                # MACRO STRESSED: defensive bear-flow alpha (built in section 1.8)
                s_full = pd.Series(_options_bear_signal).reindex(tickers).fillna(50.0)
                print(f"        options         weight={weight:.0%}  "
                      f"[MACRO DEFENSIVE — inverted bear-flow  stress={_macro_stress_score}]")
            else:
                # NORMAL: directional bull/bear flow composite (always-on fix)
                s_full = _load_options_directional(universe_set, tickers)
                print(f"        options         weight={weight:.0%}  [DIRECTIONAL net-flow]")
            signal_series[name]       = s_full
            signal_weights_used[name] = weight
            continue

        s = _load_signal(filename, col, neutral, universe_set)
        if s.empty:
            # FIX 1a: skip entire missing signal — don't inject 50 bias
            print(f"        {name:15s}  weight={weight:.0%}  → MISSING (signal dropped, weight redistributed)")
            continue
        # FIX 1b: fill per-ticker gaps with trimmed-mean of the signal,
        # not with 50.  Trimmed mean (5th–95th pct) is robust to outliers.
        s_full = s.reindex(tickers)
        n_missing = s_full.isna().sum()
        if n_missing > 0:
            sig_vals = s_full.dropna().values
            if len(sig_vals) >= 10:
                lo, hi = np.percentile(sig_vals, [5, 95])
                trimmed_mean = float(sig_vals[(sig_vals >= lo) & (sig_vals <= hi)].mean())
            else:
                trimmed_mean = float(sig_vals.mean()) if len(sig_vals) > 0 else 50.0
            s_full = s_full.fillna(trimmed_mean)
            if n_missing <= 10:
                print(f"        {name:15s}  weight={weight:.0%}  tickers={len(sig_vals)}  "
                      f"filled {n_missing} gaps → mean={trimmed_mean:.1f}")
            else:
                print(f"        {name:15s}  weight={weight:.0%}  tickers={len(sig_vals)}  "
                      f"filled {n_missing} gaps → trimmed_mean={trimmed_mean:.1f}")
        else:
            print(f"        {name:15s}  weight={weight:.0%}  tickers={s_full.notna().sum()}")
        signal_series[name]       = s_full
        signal_weights_used[name] = weight

    # ── 3. Weighted combination → raw composite ───────────────────────────────
    # FIX 2: single normalization pass — weights already reflect all overlays
    total_weight = sum(signal_weights_used.values()) or 1.0
    print(f"\n[3/5] Combining {len(signal_series)} signals "
          f"[regime={current_regime}, total_weight={total_weight*100:.0f}→normalised 100%] …")
    combo = pd.Series(0.0, index=tickers)
    for name, s in signal_series.items():
        effective_w = signal_weights_used[name] / total_weight
        combo += effective_w * s

    # Keep the raw weighted composite — do NOT re-rank.
    # Re-ranking compresses spread so rank-1=100 and rank-2=99.8 (indistinguishable).
    # The raw combo is already on [0, 100] since each input signal is 0-100 percentile.
    alpha_score = combo.round(2)

    # ── 3.5. Seasonal sector bias (Step 100 output) ───────────────────────────
    # Adds a tilt (±5 pts max) toward sectors historically strong this month.
    # Raised from ±3 to ±5 so it is meaningful relative to the 80-pt live spread.
    # Source: current_month_sector_bias.json written by step100_sector_calendar.py.
    _seasonal_path = ROOT / "current_month_sector_bias.json"
    _sector_bonus = pd.Series(0.0, index=tickers)
    if _seasonal_path.exists():
        try:
            import json as _json
            _sb = _json.loads(_seasonal_path.read_text())
            _month_name = _sb.get("month_name", "")
            _all_sectors: dict = _sb.get("all_sectors", {})  # {sector: score 0-100}

            if _all_sectors:
                # Load sector map to know each ticker's sector
                _sm_p = ROOT / "sector_map.csv"
                if _sm_p.exists():
                    _sm_df = pd.read_csv(_sm_p)
                    if {"ticker", "sector"}.issubset(_sm_df.columns):
                        _t2s = _sm_df.set_index("ticker")["sector"].to_dict()
                        for t in tickers:
                            sec = _t2s.get(t, "")
                            if sec and sec in _all_sectors:
                                # seasonal_score 0-100; 50=neutral
                                # Bonus: (score-50)/50 * 5 → range [-5, +5] pts
                                _sector_bonus[t] = (_all_sectors[sec] - 50.0) / 50.0 * 5.0

                        _n_boosted = (_sector_bonus > 0.5).sum()
                        _n_reduced = (_sector_bonus < -0.5).sum()
                        print(f"  [seasonal] {_month_name} bias applied  "
                              f"+bonus: {_n_boosted} tickers  "
                              f"-reduced: {_n_reduced} tickers  (±5pt range)")
        except Exception as _se:
            print(f"  [seasonal] sector bias skipped: {_se}")

    alpha_score = (alpha_score + _sector_bonus).clip(0, 100).round(2)

    # ── 3.6. Macro event catalyst overlay (Step 101) ──────────────────────────
    # Apply per-ticker macro catalyst scores from the event engine.
    # catalyst_score 0-100: >50 = tailwind (+pts), <50 = headwind (-pts).
    # Max adjustment: ±8 pts to preserve alpha spread.
    # Source: macro_catalyst_scores.csv written by step101_macro_event_engine.py
    _catalyst_path = ROOT / "macro_catalyst_scores.csv"
    _catalyst_bonus = pd.Series(0.0, index=tickers)
    if _catalyst_path.exists():
        try:
            import json as _json
            import time as _time
            _cat_age_days = (_time.time() - _catalyst_path.stat().st_mtime) / 86400
            if _cat_age_days <= 1:   # only use if fresh (run today)
                _cat_df = pd.read_csv(_catalyst_path)
                if {"ticker", "catalyst_score"}.issubset(_cat_df.columns):
                    _cat_map = _cat_df.set_index("ticker")["catalyst_score"].to_dict()
                    _n_bull = 0
                    _n_bear = 0
                    for t in tickers:
                        cs = _cat_map.get(t)
                        if cs is not None:
                            # (catalyst_score - 50) / 50 * 8 → range [-8, +8] pts
                            bonus = (float(cs) - 50.0) / 50.0 * 8.0
                            _catalyst_bonus[t] = bonus
                            if bonus > 0.5:
                                _n_bull += 1
                            elif bonus < -0.5:
                                _n_bear += 1
                    print(f"  [catalyst] macro events applied  "
                          f"tailwind: {_n_bull}  headwind: {_n_bear}  "
                          f"(file age: {_cat_age_days*24:.1f}h)")
            else:
                print(f"  [catalyst] macro_catalyst_scores.csv stale "
                      f"({_cat_age_days:.1f}d) — run step101 to refresh")
        except Exception as _ce:
            print(f"  [catalyst] overlay skipped: {_ce}")
    else:
        print("  [catalyst] macro_catalyst_scores.csv not found — "
              "run canyon_final_v9_step101_macro_event_engine.py to enable")

    alpha_score = (alpha_score + _catalyst_bonus).clip(0, 100).round(2)

    # ── 3.8. Per-sector regime alpha adjustment ───────────────────────────────
    # Tickers in a BULL sector (ETF 20d > 50d SMA) receive +3pts.
    # Tickers in a BEAR sector receive −3pts.
    # This encodes the insight that individual stocks ride or fight their sector
    # trend regardless of the overall market regime:
    #   NVDA in BULL tech → add 3pts even if overall regime is SIDEWAYS
    #   XOM  in BEAR energy → subtract 3pts even if oil reports good earnings
    #
    # Requires: sector_map.csv + ETF price data (computed in section 1.10).
    _sector_regime_adj = pd.Series(0.0, index=tickers)
    if _sector_regimes:
        _sm_adj_p = ROOT / "sector_map.csv"
        if _sm_adj_p.exists():
            try:
                _sm_adj   = pd.read_csv(_sm_adj_p)
                if {"ticker", "sector"}.issubset(_sm_adj.columns):
                    _t2sec    = _sm_adj.set_index("ticker")["sector"].to_dict()
                    _n_bull_a = 0
                    _n_bear_a = 0
                    for t in tickers:
                        _sr = _sector_regimes.get(_t2sec.get(t, ""), "SIDEWAYS")
                        if _sr == "BULL":
                            _sector_regime_adj[t] = +3.0
                            _n_bull_a += 1
                        elif _sr == "BEAR":
                            _sector_regime_adj[t] = -3.0
                            _n_bear_a += 1
                    print(f"  [sector-adj] +3pt bull-sector: {_n_bull_a}  "
                          f"-3pt bear-sector: {_n_bear_a}  (±3pt range)")
            except Exception as _srae:
                print(f"  [sector-adj] skipped: {_srae}")
    else:
        print("  [sector-adj] no sector-regime data — skipped")

    alpha_score = (alpha_score + _sector_regime_adj).clip(0, 100).round(2)

    # ── 3.9. Earnings event risk overlay ─────────────────────────────────────
    # Tickers with earnings reports ≤3 trading days away get alpha cut 50%.
    # Rationale: options IV spikes, stock moves are binary and unpredictable,
    # and a stale alpha signal gives no edge in the final 72h before earnings.
    # Uses earnings_calendar.csv produced by step102_earnings_calendar.py.
    #
    # HIGH  risk (days_until ≤ 3) → alpha × 0.50   (halve the signal)
    # MEDIUM risk (days_until ≤ 7) → alpha × 0.75  (moderate caution)
    #
    # The score is anchored to neutral (50) not zeroed, so we never force a
    # buy to become a hard avoid purely due to earnings timing.
    _earn_path = ROOT / "earnings_calendar.csv"
    _earn_adj  = pd.Series(1.0, index=tickers)   # multiplier (default = no change)
    _n_high_earn = 0
    _n_med_earn  = 0
    if _earn_path.exists():
        try:
            import time as _time
            _ec_age = (_time.time() - _earn_path.stat().st_mtime) / 86400
            if _ec_age <= 7:   # only use if refreshed within a week
                _ec  = pd.read_csv(_earn_path)
                if {"ticker", "days_until"}.issubset(_ec.columns):
                    _ec["days_until"] = pd.to_numeric(_ec["days_until"], errors="coerce")
                    _ec_map = _ec.dropna(subset=["days_until"]).set_index("ticker")["days_until"]
                    for t in tickers:
                        d = _ec_map.get(t)
                        if d is None:
                            continue
                        if d <= 3:
                            # Blend 50% toward neutral (50): new = 50 + (score-50)*0.50
                            _earn_adj[t] = 0.50
                            _n_high_earn += 1
                        elif d <= 7:
                            # Blend 25% toward neutral: new = 50 + (score-50)*0.75
                            _earn_adj[t] = 0.75
                            _n_med_earn += 1
                    if _n_high_earn or _n_med_earn:
                        print(f"  [earnings] pre-earnings overlay: "
                              f"HIGH(≤3d)={_n_high_earn}  MEDIUM(≤7d)={_n_med_earn}")
                    else:
                        print(f"  [earnings] no imminent earnings detected (age={_ec_age:.1f}d)")
                else:
                    print("  [earnings] earnings_calendar.csv missing expected columns")
            else:
                print(f"  [earnings] earnings_calendar.csv stale ({_ec_age:.1f}d) — skipped")
        except Exception as _ee:
            print(f"  [earnings] overlay skipped: {_ee}")
    else:
        print("  [earnings] earnings_calendar.csv not found — "
              "run step102_earnings_calendar.py to enable")

    # Apply earnings multiplier: blend alpha toward 50 (neutral)
    # Formula: adj_score = 50 + (alpha_score - 50) * multiplier
    # multiplier=1.0 → unchanged; 0.50 → half-strength; 0.0 → neutral=50
    if (_n_high_earn + _n_med_earn) > 0:
        _earn_mult = _earn_adj.reindex(tickers).fillna(1.0)
        alpha_score = (50.0 + (alpha_score - 50.0) * _earn_mult).clip(0, 100).round(2)

    # ── 4. Build alpha_scores DataFrame ──────────────────────────────────────
    print("\n[4/5] Building output tables …")
    alpha_df = pd.DataFrame({"ticker": tickers, "alpha_score": alpha_score.values})
    alpha_df["alpha_rank"] = (
        alpha_df["alpha_score"].rank(ascending=False, method="min").astype(int)
    )

    # Per-signal scores for transparency
    for name in signal_series:
        alpha_df[f"sig_{name}"] = signal_series[name].values

    # Alpha → annualised expected return (feeds Step 63 optimizer)
    # score=50 → RF; score=100 → RF + ALPHA_SCALE; score=0 → RF − ALPHA_SCALE
    alpha_df["mu_override"] = (
        RF_ANNUAL + (alpha_df["alpha_score"] - 50.0) / 50.0 * ALPHA_SCALE
    ).round(4)

    # Attach regime metadata from regime_ml_scores.csv
    if not regime_df.empty:
        meta_cols = [c for c in ["ticker", "regime", "signal", "sector", "crowding_level"]
                     if c in regime_df.columns]
        meta = regime_df[meta_cols].drop_duplicates("ticker").set_index("ticker")
        alpha_df = alpha_df.set_index("ticker").join(meta, how="left").reset_index()
        alpha_df.rename(columns={"index": "ticker"}, inplace=True)

    # ── Sector: sector_map.csv is the authoritative source (canonical GICS) ──
    # regime_ml_scores uses abbreviations ("Tech", "Healthcare", "Consumer Disc",
    # "Semiconductors") that are NOT canonical GICS sector names. sector_map.csv
    # (built by Step 88) normalises all 495 tickers to the 11 official GICS names.
    # We always prefer sector_map.csv over regime_ml sector column.
    sector_map_path = ROOT / "sector_map.csv"
    if sector_map_path.exists():
        try:
            sm = pd.read_csv(sector_map_path)
            if {"ticker", "sector"}.issubset(sm.columns):
                sm_dict = sm.set_index("ticker")["sector"]
                # Override: map ALL tickers from sector_map.csv (canonical names).
                # Fall back to existing regime_ml sector ONLY for tickers absent
                # from sector_map.csv (shouldn't happen after full step88 run).
                existing_sector = alpha_df.get("sector", pd.Series(dtype=object))
                alpha_df["sector"] = (
                    alpha_df["ticker"].map(sm_dict)
                    .fillna(existing_sector)
                )
                n_known = (~alpha_df["sector"].isin(["Other", "", None]) &
                           alpha_df["sector"].notna()).sum()
                print(f"  [sector_map] {n_known}/{len(alpha_df)} tickers have canonical sector")
        except Exception as e:
            print(f"  [WARN] sector_map.csv: {e}")

    alpha_df = alpha_df.sort_values("alpha_rank").reset_index(drop=True)

    # ── Assign concentrated LONG/SHORT signals based on rank ─────────────────
    # Top 15 by alpha score = LONG candidates; bottom 15 = SHORT candidates
    n_universe = len(alpha_df)
    alpha_df["signal"] = "NEUTRAL"
    alpha_df.loc[alpha_df["alpha_rank"] <= top_n, "signal"] = "LONG"
    alpha_df.loc[alpha_df["alpha_rank"] > n_universe - top_n, "signal"] = "SHORT"
    n_long_assigned  = (alpha_df["signal"] == "LONG").sum()
    n_short_assigned = (alpha_df["signal"] == "SHORT").sum()
    print(f"  [signal] LONG={n_long_assigned}  SHORT={n_short_assigned}  NEUTRAL={n_universe - n_long_assigned - n_short_assigned}")

    # ── 5. Daily picks (top N) ────────────────────────────────────────────────
    eligible = alpha_df[alpha_df["signal"] != "SHORT"].copy()
    picks = eligible.head(top_n).copy()

    # ── Exposure override (Step 110 circuit breaker) ─────────────────────────
    # Step 110 writes exposure_override.json when portfolio drawdown exceeds
    # the soft (10%) or hard (15%) threshold.  We honour it here by scaling
    # the effective POSITION_CAP so that all positions are proportionally
    # reduced rather than simply dropping them.
    #   exposure_multiplier = 1.0  →  normal (no override)
    #   exposure_multiplier = 0.5  →  defensive (10-15% drawdown)
    #   exposure_multiplier = 0.2  →  emergency (>15% drawdown)
    _override_path = ROOT / "exposure_override.json"
    _exposure_mult = 1.0
    _effective_pos_cap = POSITION_CAP
    if _override_path.exists():
        try:
            import json as _json
            _ov = _json.loads(_override_path.read_text())
            _ov_mult = float(_ov.get("exposure_multiplier", 1.0))
            _ov_date = str(_ov.get("date", ""))
            _ov_level = str(_ov.get("circuit_level", "NONE"))
            # Only honour if the override was written today (stale override
            # from a week-old run should not throttle normal trading)
            _today_str = datetime.now().strftime("%Y-%m-%d")
            _ov_age_days = 0
            if _ov_date:
                try:
                    from datetime import datetime as _dt2
                    _ov_dt = _dt2.strptime(_ov_date, "%Y-%m-%d")
                    _ov_age_days = (datetime.now() - _ov_dt).days
                except Exception:
                    pass
            if _ov_age_days <= 1 and _ov_mult < 1.0:
                _exposure_mult     = _ov_mult
                _effective_pos_cap = POSITION_CAP * _ov_mult
                print(f"  [circuit] exposure_override.json: "
                      f"mult={_ov_mult:.2f}  level={_ov_level}  "
                      f"dd={_ov.get('drawdown_pct', '?')}%  "
                      f"effective_cap={_effective_pos_cap:.3f}")
            elif _ov_age_days > 1:
                print(f"  [circuit] exposure_override.json stale ({_ov_age_days}d) — skipped")
            else:
                print(f"  [circuit] No circuit breaker active  [mult=1.0]")
        except Exception as _ove:
            print(f"  [circuit] exposure_override.json read error: {_ove}")
    else:
        print(f"  [circuit] exposure_override.json not found — run step110 first")

    # Rank-proportional weights with position cap
    scores = picks["alpha_score"].values
    raw_w  = scores / scores.sum()
    raw_w  = np.clip(raw_w, 0, _effective_pos_cap)
    raw_w /= raw_w.sum()
    picks["weight_pct"] = (raw_w * 100).round(2)

    # Action labels
    picks["action"] = picks["alpha_score"].apply(
        lambda s: "STRONG BUY" if s >= 80 else ("BUY" if s >= 60 else "WATCH")
    )

    # Top driving signal
    sig_cols = [c for c in picks.columns if c.startswith("sig_")]
    if sig_cols:
        picks["top_signal"] = picks[sig_cols].idxmax(axis=1).str.replace("sig_", "")

    # ── Delta: compare to yesterday's picks ──────────────────────────────────
    prev_path = ROOT / "daily_picks_prev.csv"
    if prev_path.exists():
        try:
            prev = pd.read_csv(prev_path)[["ticker", "alpha_rank"]].rename(
                columns={"alpha_rank": "prev_rank"}
            )
            picks = picks.merge(prev, on="ticker", how="left")
            picks["rank_change"] = (
                picks["prev_rank"] - picks["alpha_rank"]
            ).fillna(0).astype(int)
            picks["status"] = picks.apply(
                lambda r: "🆕 NEW"   if pd.isna(r.get("prev_rank")) or r["prev_rank"] > top_n
                     else ("📈 UP"   if r["rank_change"] > 3
                     else ("📉 DOWN" if r["rank_change"] < -3
                     else "➡ STEADY")),
                axis=1,
            )
        except Exception:
            pass

    # ── Score trend (from alpha_score_history.csv — built up over daily runs) ─
    trend_series = _compute_score_trend(tickers)
    picks["score_trend"] = picks["ticker"].map(trend_series).fillna("★")

    # Build final column list safely
    keep = ["ticker", "action", "weight_pct", "alpha_score", "alpha_rank"]
    for optional in ["score_trend", "rank_change", "status",
                     "top_signal", "regime", "sector"] + sig_cols:
        if optional in picks.columns:
            keep.append(optional)
    picks = picks[keep].reset_index(drop=True)

    # ── Reduce / avoid: bottom-quartile stocks ────────────────────────────────
    reduce_df = (
        eligible[eligible["alpha_score"] < 30]
        .sort_values("alpha_score")
        .head(20)[
            ["ticker", "alpha_score", "alpha_rank"]
            + (["sector"] if "sector" in eligible.columns else [])
            + (["signal"] if "signal" in eligible.columns else [])
        ]
        .reset_index(drop=True)
    )
    reduce_df["action"] = "REDUCE / AVOID"

    return alpha_df, picks, reduce_df, current_regime, eff_weights


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    alpha_df: pd.DataFrame,
    picks: pd.DataFrame,
    reduce_df: pd.DataFrame,
    current_regime: str = "BULL",
    eff_weights: dict | None = None,
) -> None:
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Rotate: save current picks as "yesterday" before overwriting
    picks_path = ROOT / "daily_picks.csv"
    prev_path  = ROOT / "daily_picks_prev.csv"
    if picks_path.exists():
        import shutil
        shutil.copy(picks_path, prev_path)

    # alpha_scores.csv — full universe
    alpha_path = ROOT / "alpha_scores.csv"
    alpha_df.to_csv(alpha_path, index=False)
    print(f"  [written] {alpha_path}  ({len(alpha_df)} rows)")

    # ── Append to alpha_score_history.csv (powers _compute_score_trend) ──────
    hist_path = ROOT / "alpha_score_history.csv"
    snapshot  = alpha_df[["ticker", "alpha_score"]].copy()
    snapshot.insert(0, "date", today_str)
    if hist_path.exists():
        try:
            existing  = pd.read_csv(hist_path)
            existing  = existing[existing["date"] != today_str]   # deduplicate today
            hist_full = pd.concat([existing, snapshot], ignore_index=True)
        except Exception:
            hist_full = snapshot
    else:
        hist_full = snapshot
    # Trim to last 30 trading days to control file size
    if "date" in hist_full.columns:
        keep_dates = sorted(hist_full["date"].unique())[-30:]
        hist_full  = hist_full[hist_full["date"].isin(keep_dates)]
    hist_full.to_csv(hist_path, index=False)
    n_days = hist_full["date"].nunique() if not hist_full.empty else 0
    print(f"  [written] {hist_path}  ({n_days} days accumulated)")

    # daily_picks.csv — top-N buy list
    picks.to_csv(picks_path, index=False)
    print(f"  [written] {picks_path}  ({len(picks)} rows)")

    # daily_shorts.csv — bottom-N short list
    short_cols = ["ticker", "alpha_score", "alpha_rank"] + \
                 (["sector"] if "sector" in alpha_df.columns else []) + \
                 (["signal"] if "signal" in alpha_df.columns else [])
    short_cols = [c for c in short_cols if c in alpha_df.columns]
    shorts_df = (alpha_df[alpha_df["signal"] == "SHORT"]
                 .sort_values("alpha_rank", ascending=False)
                 .head(TOP_N)[short_cols]
                 .reset_index(drop=True))
    shorts_path = ROOT / "daily_shorts.csv"
    shorts_df.to_csv(shorts_path, index=False)
    print(f"  [written] {shorts_path}  ({len(shorts_df)} rows)")

    # daily_reduce.csv — reduce / avoid list
    reduce_path = ROOT / "daily_reduce.csv"
    reduce_df.to_csv(reduce_path, index=False)
    print(f"  [written] {reduce_path}  ({len(reduce_df)} rows)")

    # ── alpha_report.md — human-readable summary ──────────────────────────────
    ew        = eff_weights or {}
    total_w   = sum(ew.values()) or 1.0
    base_w_map = {name: w for name, _, _, _, w in SIGNAL_CONFIG}

    report_lines = [
        "# Canyon v9 — Alpha Aggregator Report (Step 87)",
        f"Generated: {ts}",
        f"**Regime: {current_regime}** (regime-conditional weights applied)",
        "",
        f"Universe: {len(alpha_df)} tickers | Picks: {len(picks)}",
        "",
        "## Signal Weights Used",
        f"*Regime: {current_regime} — weights differ from defaults in BEAR / LATE_BULL*",
        "",
        "| Signal | Regime Weight | Base Weight |",
        "|--------|---------------|-------------|",
    ]
    for name in ew:
        rw   = ew[name]
        bw   = base_w_map.get(name, 0)
        flag = " ★" if abs(rw - bw) > 0.02 else ""
        report_lines.append(f"| {name} | {rw/total_w:.0%} | {bw:.0%} |{flag}")

    report_lines += [
        "",
        "## Today's Top Picks",
        "",
        "| Rank | Ticker | Action | Weight | Score | Trend | Sector |",
        "|------|--------|--------|--------|-------|-------|--------|",
    ]
    for _, row in picks.iterrows():
        sector = str(row.get("sector", "—") or "—")
        trend  = str(row.get("score_trend", ""))
        report_lines.append(
            f"| {row['alpha_rank']} | {row['ticker']} | {row['action']} | "
            f"{row['weight_pct']:.1f}% | {row['alpha_score']:.1f} | {trend} | {sector} |"
        )

    report_lines += [
        "",
        "## Notes",
        "- alpha_score: cross-sectional percentile rank (0=worst, 100=best)",
        "- weight_pct: rank-proportional allocation with 15% position cap",
        "- mu_override: annualised expected return fed to Step 63 optimizer",
        "- score_trend: ▲▲/▲/→/▼/▼▼ from alpha_score_history.csv (last 5 days)",
        "- Regime weights adjust dynamically each run based on detected regime",
        "- Run Step 63 after Step 87 to get risk-model adjusted weights",
    ]

    rp = ROOT / "alpha_report.md"
    rp.write_text("\n".join(report_lines))
    print(f"  [written] {rp}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(picks: pd.DataFrame) -> None:
    print()
    print("─" * 72)
    print(f"{'CANYON v9 — TODAY\'S PICKS':^72}")
    print("─" * 72)
    print(f"{'#':<4} {'Ticker':<7} {'Action':<12} {'Wt%':>5} {'Score':>7}  "
          f"{'Trend':<6}  {'Sector'}")
    print("─" * 72)
    for _, row in picks.iterrows():
        sector = str(row.get("sector", ""))[:18] if "sector" in row.index else ""
        trend  = str(row.get("score_trend", "")) if "score_trend" in row.index else ""
        print(f"{row['alpha_rank']:<4} {row['ticker']:<7} {row['action']:<12} "
              f"{row['weight_pct']:>5.1f}  {row['alpha_score']:>6.1f}   "
              f"{trend:<6}  {sector}")
    print("─" * 72)
    total_w  = picks["weight_pct"].sum()
    n_strong = (picks["action"] == "STRONG BUY").sum()
    print(f"Total weight: {total_w:.1f}%   Strong buys: {n_strong}   "
          f"Buys: {(picks['action']=='BUY').sum()}")
    print("─" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Canyon v9 Step 87 — Alpha Aggregator")
    parser.add_argument("--top",     type=int, default=TOP_N,
                        help=f"Number of picks to output (default {TOP_N})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print but don't write files")
    args = parser.parse_args()

    alpha_df, picks, reduce_df, current_regime, eff_weights = aggregate_alpha(top_n=args.top)

    print_summary(picks)

    if not args.dry_run:
        print("\n[5/5] Writing outputs …")
        write_outputs(alpha_df, picks, reduce_df,
                      current_regime=current_regime, eff_weights=eff_weights)
        if not reduce_df.empty:
            print(f"\n  ⚠  Reduce/Avoid list: {len(reduce_df)} stocks with alpha < 30")
            print("     " + "  ".join(reduce_df["ticker"].head(8).tolist()) + " …")
    else:
        print("\n[DRY-RUN] files not written")

    print(f"\n✓ Alpha aggregation complete — {len(picks)} picks ready "
          f"[regime: {current_regime}]")
    print(f"  Next: run Step 63 (portfolio optimizer) to apply risk model\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
