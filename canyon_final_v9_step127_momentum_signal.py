#!/usr/bin/env python3
"""
Canyon v9 — Step 111: Institutional Momentum Signal
=====================================================
Four-component price momentum score for the full S&P 500 universe.
Outputs momentum_scores.csv — consumed by Step 87 alpha aggregator.

Wall Street momentum research built in:
────────────────────────────────────────────────────────────────────
1. Cross-Sectional Momentum  (Jegadeesh & Titman, 1993)
   ─ 12-month total return, skip most recent month to avoid the
     documented 1-month reversal (microstructure noise, bid-ask bounce).
   ─ "Buy winners, sell losers" — the original momentum factor.
   ─ Rebalanced monthly. IC historically 0.05–0.15.

2. 52-Week High Ratio  (George & Hwang, 2004)
   ─ price / rolling_252d_max  → [0, 1]
   ─ Stocks trading near their 52-week high are psychologically
     "anchored" — analysts and investors anchor targets to the old
     high and underreact to new information.  Stocks that break out
     above the 52-week high continue to outperform.
   ─ Orthogonal to traditional momentum; improves composite IC.

3. Volatility-Scaled Momentum  (Barroso & Santa-Clara, 2015 / AQR)
   ─ mom_12_1 / realized_vol_21d
   ─ Normalises momentum exposure for time-varying risk.  A 40%
     return in a low-vol stock carries far more information than a
     40% return in a high-vol stock.
   ─ Reduces "momentum crashes" in bear-market recoveries by 50%+.
   ─ Used by AQR Momentum Fund (AMOMX) as the core risk adjustment.

4. Residual / Idiosyncratic Momentum  (Blitz, Huij & Martens, 2011)
   ─ Daily returns regressed on SPY (12-month window, OLS beta).
   ─ Residual = stock-specific return (removes market-driven moves).
   ─ 12-1 month cumulative residual = "alpha momentum."
   ─ More persistent across market regimes than raw price momentum.
   ─ Less correlated with value, beta, and sector factors.

Composite score:
   momentum_score = 0.35 × rank(cs_mom)
                  + 0.30 × rank(high52_ratio)
                  + 0.25 × rank(vol_scaled_mom)
                  + 0.10 × rank(residual_mom)
   → ranked cross-sectionally → 0-100

Momentum Crash Protection:
   Momentum crashes are acute during sharp market recoveries (BEAR→BULL
   transition): the prior-period losers (which momentum would short) are
   the first to surge.  AQR and Two Sigma both use regime conditioning.

   This file writes a regime_dampened flag.  Step 87 reads
   regime_current.json and applies an additional multiplier on top.

Universe:
   All tickers in sp500_price_cache.csv (≈496 S&P 500 names).
   Tickers with < 126 days of price history receive NaN (dropped).

Inputs
------
  sp500_price_cache.csv   — daily OHLCV or close prices (date × ticker)
  regime_current.json     — current BULL/BEAR/SIDEWAYS/LATE_BULL regime
  macro_signals.json      — VIX level for crash-protection conditioning

Outputs
-------
  momentum_scores.csv     — ticker, momentum_score (0-100), sub-scores,
                            regime_dampened, momentum_rank

Usage
-----
  python3 canyon_final_v9_step111_momentum_signal.py
  python3 canyon_final_v9_step111_momentum_signal.py --top 200
  python3 canyon_final_v9_step111_momentum_signal.py --lookback 504
  python3 canyon_final_v9_step111_momentum_signal.py --no-regime-damp
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Paths ─────────────────────────────────────────────────────────────────────
PRICE_CSV    = ROOT / "sp500_price_cache.csv"
SPY_CSV      = ROOT / "spy_price_cache.csv"      # SPY benchmark for residual momentum
REGIME_JSON  = ROOT / "regime_current.json"
MACRO_JSON   = ROOT / "macro_signals.json"
OUT_CSV      = ROOT / "momentum_scores.csv"
OUT_MD       = ROOT / "momentum_signal_report.md"

SPY_CACHE_TTL_DAYS = 1   # Re-download SPY if cache > 1 day old

# ── Constants ─────────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 252       # 12 months of trading days
SKIP_DAYS       = 21        # skip-1-month reversal avoidance (Jegadeesh-Titman)
HIGH52_WINDOW   = 252       # 252 trading days = 52-week high window
VOL_WINDOW      = 21        # realized volatility window (days)
MIN_HISTORY     = 126       # minimum data points for a valid score (6 months)
ANN_FACTOR      = 252       # annualise realized vol

# Composite weights (must sum to 1.0)
W_CS_MOM      = 0.35   # Cross-sectional momentum (J-T)
W_HIGH52      = 0.30   # 52-week high ratio (G-H)
W_VOL_SCALED  = 0.25   # Volatility-scaled momentum (AQR)
W_RESIDUAL    = 0.10   # Residual / idiosyncratic momentum

# Regime-based crash protection multipliers (applied to final composite)
# Momentum crashes are most severe during bear-market recoveries.
REGIME_DAMP: dict[str, float] = {
    "BULL":      1.00,   # full momentum signal in bull market
    "LATE_BULL": 0.85,   # slight dampening — momentum starting to fade
    "SIDEWAYS":  0.75,   # moderate dampening — mean reversion more likely
    "BEAR":      0.45,   # heavy dampening — momentum crashes in bear recoveries
}

# VIX-based additional dampening (applied ON TOP of regime dampening)
# VIX > 35: additional 0.60× (momentum most dangerous in fear spikes)
# VIX > 25: additional 0.80×
VIX_DAMP_35  = 0.60
VIX_DAMP_25  = 0.80


# =============================================================================
# 1. Price loader
# =============================================================================

def load_prices() -> pd.DataFrame:
    """
    Load sp500_price_cache.csv.  Returns wide DataFrame (date × ticker).
    Sorts by date ascending.  Drops all-NaN columns.
    """
    if not PRICE_CSV.exists():
        raise FileNotFoundError(
            f"sp500_price_cache.csv not found at {PRICE_CSV}\n"
            "Run: python3 canyon_final_v9_step75_universe_expansion.py"
        )
    prices = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")
    return prices


def load_spy_prices(prices_index: pd.DatetimeIndex) -> pd.Series | None:
    """
    Load SPY closing prices for the residual momentum calculation.

    Strategy (in order):
      1. Check spy_price_cache.csv — use if < SPY_CACHE_TTL_DAYS old.
      2. Download from yfinance (1d interval, 2000-01-01 to today).
      3. Save to spy_price_cache.csv for future runs.

    Returns aligned pd.Series indexed to prices_index, or None on failure.
    """
    import time as _time

    # ── Try cache first ───────────────────────────────────────────────────────
    if SPY_CSV.exists():
        age_days = (_time.time() - SPY_CSV.stat().st_mtime) / 86400
        if age_days < SPY_CACHE_TTL_DAYS:
            try:
                spy_df = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
                spy_df = spy_df.sort_index()
                col = "Close" if "Close" in spy_df.columns else spy_df.columns[0]
                spy_s = spy_df[col].reindex(prices_index)
                n_valid = spy_s.notna().sum()
                if n_valid >= 200:
                    print(f"  SPY cache hit ({age_days:.1f}d old) — {n_valid} days aligned")
                    return spy_s
            except Exception as e:
                print(f"  SPY cache read failed: {e}")

    # ── Download from yfinance ────────────────────────────────────────────────
    print("  Downloading SPY from yfinance …")
    try:
        import yfinance as yf
        spy_raw = yf.download("SPY", start="2000-01-01", progress=False, auto_adjust=True)
        if spy_raw.empty:
            print("  SPY download returned empty — residual momentum unavailable")
            return None

        # Handle MultiIndex columns (yfinance ≥ 0.2)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)

        close_col = "Close" if "Close" in spy_raw.columns else spy_raw.columns[0]
        spy_close = spy_raw[[close_col]].rename(columns={close_col: "Close"})
        spy_close.index = pd.to_datetime(spy_close.index)
        spy_close = spy_close.sort_index()

        # Save cache
        spy_close.to_csv(SPY_CSV)
        print(f"  SPY downloaded: {len(spy_close)} rows → saved to spy_price_cache.csv")

        spy_s = spy_close["Close"].reindex(prices_index)
        n_valid = spy_s.notna().sum()
        print(f"  SPY aligned to prices_index: {n_valid} valid days")
        return spy_s if n_valid >= 200 else None

    except Exception as e:
        print(f"  SPY download failed: {e}")
        return None


# =============================================================================
# 2. Momentum sub-signals
# =============================================================================

def compute_cs_momentum(prices: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional Jegadeesh-Titman momentum.

    Return over [t-252, t-21] (12 months, skip 1 month).
    Uses simple total return (not log return) to match academic literature.

    Parameters
    ----------
    prices : wide DataFrame, last row = today

    Returns
    -------
    pd.Series  — raw 12-1 return per ticker (not yet ranked)
    """
    if len(prices) < LOOKBACK_DAYS + 5:
        return pd.Series(dtype=float)

    end_row   = prices.iloc[-SKIP_DAYS - 1]       # price as of 1 month ago
    start_row = prices.iloc[-(LOOKBACK_DAYS + 1)] # price as of 12 months ago

    # Total return = (P_{t-21} / P_{t-252}) - 1
    raw_mom = (end_row / start_row) - 1.0

    # Winsorise at 1st/99th percentile to remove data errors
    lo, hi = np.nanpercentile(raw_mom.dropna(), [1, 99])
    raw_mom = raw_mom.clip(lo, hi)
    return raw_mom


def compute_52w_high(prices: pd.DataFrame) -> pd.Series:
    """
    George & Hwang 52-week high proximity ratio.

    ratio = today's price / max(price over last 252 trading days)

    Stocks near their 52-week high (ratio → 1.0) are expected to
    outperform; stocks far below their 52-week high (ratio → 0.0)
    are expected to underperform.

    Returns
    -------
    pd.Series  — ratio ∈ (0, 1] per ticker (not yet ranked)
    """
    if len(prices) < HIGH52_WINDOW:
        return pd.Series(dtype=float)

    rolling_max = prices.tail(HIGH52_WINDOW).max()
    current_px  = prices.iloc[-1]

    ratio = current_px / rolling_max.replace(0, np.nan)
    ratio = ratio.clip(0.0, 1.0)      # by definition can't exceed 1.0
    return ratio


def compute_vol_scaled_momentum(
    prices: pd.DataFrame, cs_mom: pd.Series
) -> pd.Series:
    """
    AQR / Barroso-Santa-Clara volatility-scaled momentum.

    vol_scaled = mom_12_1 / annualised_realised_vol_21d

    Intuition: a 30% return on a 15%-vol stock is much more
    informative than a 30% return on a 60%-vol stock.  Dividing
    by vol gives the "Sharpe ratio of the past year."

    Realized vol = rolling 21-day std of daily log returns × sqrt(252)

    Returns
    -------
    pd.Series  — risk-adjusted momentum per ticker (not yet ranked)
    """
    if len(prices) < VOL_WINDOW + 5:
        return pd.Series(dtype=float)

    log_rets = np.log(prices / prices.shift(1))
    vol_21d  = log_rets.tail(VOL_WINDOW).std() * np.sqrt(ANN_FACTOR)

    # Avoid division by near-zero vol (illiquid tickers)
    vol_21d  = vol_21d.replace(0, np.nan)
    vol_21d  = vol_21d.clip(lower=0.05)   # floor at 5% annualised vol

    vol_scaled = cs_mom / vol_21d
    return vol_scaled


def compute_residual_momentum(prices: pd.DataFrame) -> pd.Series:
    """
    Idiosyncratic / residual momentum (Blitz, Huij & Martens, 2011).

    Algorithm:
      1. Get SPY log-returns over the full lookback window.
      2. For each ticker: OLS regression of daily log-returns on SPY
         (using the lookback window, skip most recent month).
      3. Cumulative sum of OLS residuals = firm-specific return
         (alpha momentum) over 12-1 months.
      4. Rank cross-sectionally.

    Captures momentum in the part of returns not explained by the
    broad market — more persistent, less correlated with value factor,
    and more robust across market regimes.

    Returns
    -------
    pd.Series  — cumulative OLS residual per ticker (not yet ranked)
    """
    if "SPY" not in prices.columns:
        return pd.Series(dtype=float)

    if len(prices) < LOOKBACK_DAYS + 5:
        return pd.Series(dtype=float)

    # Window: [t-252, t-21] (skip last month — same as JT momentum)
    window = prices.iloc[-(LOOKBACK_DAYS + 1): -SKIP_DAYS]
    if len(window) < 60:
        return pd.Series(dtype=float)

    log_rets = np.log(window / window.shift(1)).dropna()
    spy_rets = log_rets["SPY"].values.reshape(-1, 1)

    residual_sums: dict[str, float] = {}

    for ticker in prices.columns:
        if ticker == "SPY":
            continue
        stock_rets = log_rets.get(ticker)
        if stock_rets is None or stock_rets.isna().sum() > len(log_rets) * 0.20:
            continue

        y = stock_rets.values
        x = spy_rets[:, 0]
        # Align lengths (some tickers may have fewer rows)
        n = min(len(y), len(x))
        if n < 40:
            continue
        y, x = y[:n], x[:n]

        # Remove NaN pairs
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue

        slope, intercept, _, _, _ = spstats.linregress(x[mask], y[mask])
        residuals = y[mask] - (intercept + slope * x[mask])
        residual_sums[ticker] = float(residuals.sum())

    return pd.Series(residual_sums)


# =============================================================================
# 3. Regime / VIX conditioning
# =============================================================================

def load_regime_and_vix() -> tuple[str, float | None]:
    """
    Read current market regime and VIX level.

    Returns
    -------
    regime : "BULL" | "BEAR" | "SIDEWAYS" | "LATE_BULL"
    vix    : float | None
    """
    regime = "BULL"   # safe default
    vix    = None

    for src in (MACRO_JSON, REGIME_JSON):
        if src.exists():
            try:
                d = json.loads(src.read_text())
                if regime == "BULL":
                    r = str(d.get("regime", d.get("current_regime", "BULL"))).upper()
                    if r in REGIME_DAMP:
                        regime = r
                if vix is None:
                    v = d.get("vix") or d.get("vix_spot")
                    if v is not None:
                        vix = float(v)
            except Exception:
                pass

    return regime, vix


def apply_crash_protection(
    raw_score: pd.Series,
    regime: str,
    vix: float | None,
    enabled: bool = True,
) -> tuple[pd.Series, float, bool]:
    """
    Dampen momentum score toward neutral (50) in crash-prone regimes.

    Formula:
        dampened = 50 + (raw_score - 50) × total_multiplier

    total_multiplier = regime_mult × vix_mult

    Returns
    -------
    dampened_score : pd.Series (0-100)
    total_mult     : float
    was_dampened   : bool
    """
    if not enabled:
        return raw_score, 1.0, False

    regime_mult = REGIME_DAMP.get(regime, 1.0)

    vix_mult = 1.0
    if vix is not None:
        if vix > 35:
            vix_mult = VIX_DAMP_35
        elif vix > 25:
            vix_mult = VIX_DAMP_25

    total_mult  = regime_mult * vix_mult
    was_dampened = total_mult < 0.99

    if was_dampened:
        dampened = 50.0 + (raw_score - 50.0) * total_mult
    else:
        dampened = raw_score

    return dampened.clip(0, 100), total_mult, was_dampened


# =============================================================================
# 4. Cross-sectional ranker
# =============================================================================

def cs_rank(s: pd.Series, label: str = "") -> pd.Series:
    """
    Rank a raw signal cross-sectionally to [0, 100] percentile.
    Drops NaN before ranking; NaN tickers get NaN in output.
    """
    valid = s.dropna()
    if len(valid) < 5:
        if label:
            print(f"  [momentum] {label}: only {len(valid)} valid tickers — skipped")
        return pd.Series(dtype=float)
    ranked = valid.rank(pct=True) * 100.0
    return ranked


# =============================================================================
# 5. Main composite builder
# =============================================================================

def compute_momentum_scores(
    top_n: int = 0,
    lookback: int = LOOKBACK_DAYS,
    no_regime_damp: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline:
      1. Load prices
      2. Compute 4 momentum sub-signals
      3. Cross-sectional rank each → 0-100
      4. Weighted composite
      5. Apply regime/VIX crash protection
      6. Return scored DataFrame

    Parameters
    ----------
    top_n          : if > 0, return only top N tickers by score
    lookback       : number of trading days for 12-month window
    no_regime_damp : if True, skip crash protection damping
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*65}")
    print(f"Canyon v9 — Step 111: Momentum Signal  [{ts}]")
    print(f"{'='*65}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading prices …")
    prices = load_prices()
    tickers_all = list(prices.columns)
    print(f"  {len(tickers_all)} tickers × {len(prices)} days  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    # Filter to tickers with at least MIN_HISTORY of data
    valid_mask = prices.notna().sum() >= MIN_HISTORY
    prices_valid = prices.loc[:, valid_mask]
    print(f"  {valid_mask.sum()} tickers with ≥{MIN_HISTORY} days history")

    # ── Sub-signal 1: Cross-sectional momentum ───────────────────────────────
    print("\n[2/5] Computing momentum sub-signals …")

    cs_mom_raw = compute_cs_momentum(prices_valid)
    cs_mom_r   = cs_rank(cs_mom_raw, "cs_momentum")
    n_cs = len(cs_mom_r.dropna())
    cs_mean = cs_mom_raw.dropna().mean()
    cs_std  = cs_mom_raw.dropna().std()
    print(f"  cs_momentum (J-T 12-1m):   {n_cs} tickers  "
          f"mean={cs_mean*100:+.1f}%  std={cs_std*100:.1f}%")

    # ── Sub-signal 2: 52-week high ratio ─────────────────────────────────────
    high52_raw = compute_52w_high(prices_valid)
    high52_r   = cs_rank(high52_raw, "52w_high")
    n_52 = len(high52_r.dropna())
    h52_mean = high52_raw.dropna().mean()
    print(f"  52w_high_ratio (G-H):       {n_52} tickers  "
          f"mean={h52_mean:.3f}  (1.0=at all-time-high)")

    # ── Sub-signal 3: Volatility-scaled momentum ──────────────────────────────
    vol_mom_raw = compute_vol_scaled_momentum(prices_valid, cs_mom_raw)
    vol_mom_r   = cs_rank(vol_mom_raw, "vol_scaled")
    n_vs = len(vol_mom_r.dropna())
    vs_mean = vol_mom_raw.dropna().mean()
    print(f"  vol_scaled_mom (AQR):       {n_vs} tickers  "
          f"mean={vs_mean:+.3f}  (Sharpe of past year)")

    # ── Sub-signal 4: Residual / idiosyncratic momentum ───────────────────────
    # Inject SPY into prices_valid so compute_residual_momentum can regress on it.
    # We fetch SPY from a separate cache (spy_price_cache.csv) rather than the
    # S&P 500 constituent cache which intentionally excludes the index ETF.
    spy_series = load_spy_prices(prices_valid.index)
    if spy_series is not None and spy_series.notna().sum() >= 200:
        prices_for_residual = prices_valid.copy()
        prices_for_residual["SPY"] = spy_series
    else:
        prices_for_residual = prices_valid   # residual will be skipped inside

    residual_raw = compute_residual_momentum(prices_for_residual)
    residual_r   = cs_rank(residual_raw, "residual")
    n_res = len(residual_r.dropna())
    if n_res > 0:
        res_mean = residual_raw.dropna().mean()
        print(f"  residual_mom (Blitz):       {n_res} tickers  "
              f"mean_cumres={res_mean:+.4f}")
    else:
        print("  residual_mom: SPY unavailable — skipped (weight redistributed)")

    # ── Composite score ──────────────────────────────────────────────────────
    print("\n[3/5] Building composite momentum score …")

    # Align all sub-signal ranks on full ticker set
    base_index = cs_mom_r.index.union(high52_r.index).union(vol_mom_r.index)
    if len(residual_r) > 0:
        base_index = base_index.union(residual_r.index)

    cs_a   = cs_mom_r.reindex(base_index)
    h52_a  = high52_r.reindex(base_index)
    vs_a   = vol_mom_r.reindex(base_index)
    res_a  = residual_r.reindex(base_index) if len(residual_r) > 0 else pd.Series(np.nan, index=base_index)

    # Adaptive weighting: if residual_mom is missing, redistribute weight
    if n_res < 20:
        # Redistribute residual weight proportionally to other 3
        total_w = W_CS_MOM + W_HIGH52 + W_VOL_SCALED
        w_cs  = W_CS_MOM    / total_w
        w_h52 = W_HIGH52    / total_w
        w_vs  = W_VOL_SCALED / total_w
        w_res = 0.0
        print(f"  Residual momentum unavailable — weights redistributed: "
              f"cs={w_cs:.2f} h52={w_h52:.2f} vs={w_vs:.2f}")
    else:
        w_cs  = W_CS_MOM
        w_h52 = W_HIGH52
        w_vs  = W_VOL_SCALED
        w_res = W_RESIDUAL

    # Fill NaN with 50 (neutral) for combination
    cs_f   = cs_a.fillna(50.0)
    h52_f  = h52_a.fillna(50.0)
    vs_f   = vs_a.fillna(50.0)
    res_f  = res_a.fillna(50.0) if w_res > 0 else pd.Series(0.0, index=base_index)

    composite = (w_cs  * cs_f
               + w_h52 * h52_f
               + w_vs  * vs_f
               + w_res * res_f)
    composite = composite.clip(0, 100)

    # Re-rank composite cross-sectionally → final 0-100 score
    # (prevents weight artifacts from neutral fills from compressing spread)
    n_valid_comp = composite.notna().sum()
    composite_ranked = composite.rank(pct=True) * 100.0
    spread = composite_ranked.max() - composite_ranked.min()
    print(f"  Composite: {n_valid_comp} tickers  "
          f"spread={spread:.1f}pts  mean={composite_ranked.mean():.1f}")

    # ── Regime / VIX crash protection ────────────────────────────────────────
    print("\n[4/5] Applying momentum crash protection …")
    regime, vix = load_regime_and_vix()
    _vix_str = f"{vix:.1f}" if vix is not None else "N/A"
    print(f"  Regime: {regime}  VIX: {_vix_str}")

    dampened, total_mult, was_dampened = apply_crash_protection(
        composite_ranked, regime, vix, enabled=(not no_regime_damp)
    )

    regime_label = regime
    if was_dampened:
        regime_label += f"  [dampened ×{total_mult:.2f}]"
        print(f"  Crash protection ACTIVE: total_mult={total_mult:.2f}  "
              f"(regime×{REGIME_DAMP.get(regime,1):.2f} "
              f"VIX×{VIX_DAMP_25 if vix and 25<vix<=35 else (VIX_DAMP_35 if vix and vix>35 else 1.0):.2f})")
    else:
        print(f"  No crash protection needed  (mult={total_mult:.2f})")

    # ── Build output DataFrame ────────────────────────────────────────────────
    print("\n[5/5] Writing outputs …")
    out = pd.DataFrame({
        "ticker":          base_index,
        "momentum_score":  dampened.values.round(2),
        "sub_cs_mom":      cs_a.reindex(base_index).values.round(2),
        "sub_high52":      h52_a.reindex(base_index).values.round(2),
        "sub_vol_scaled":  vs_a.reindex(base_index).values.round(2),
        "sub_residual":    res_a.reindex(base_index).values.round(2),
        "cs_mom_raw_pct":  cs_mom_raw.reindex(base_index).values.round(4),
        "high52_ratio":    high52_raw.reindex(base_index).values.round(4),
        "regime_dampened": was_dampened,
        "regime":          regime,
        "vix":             vix if vix is not None else np.nan,
        "total_damp_mult": total_mult,
    })

    out["momentum_rank"] = out["momentum_score"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype("Int64")

    out = out.sort_values("momentum_score", ascending=False).reset_index(drop=True)

    # Top-N filter
    if top_n > 0:
        out = out.head(top_n)

    out.to_csv(OUT_CSV, index=False)
    print(f"  [written] {OUT_CSV}  ({len(out)} rows)")

    # ── Markdown report ───────────────────────────────────────────────────────
    _write_report(out, regime, vix, total_mult, was_dampened,
                  n_cs, n_52, n_vs, n_res)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    top10 = out.head(10)[["ticker", "momentum_score", "sub_cs_mom",
                           "sub_high52", "sub_vol_scaled", "sub_residual"]]
    print("  Top 10 momentum picks:")
    print(f"  {'Ticker':8s} {'Score':6s} {'CS_Mom':7s} {'52wH':6s} {'VolScl':7s} {'Resid':6s}")
    for _, row in top10.iterrows():
        def _f(v):
            return f"{v:.1f}" if pd.notna(v) else "  N/A"
        print(f"  {row['ticker']:8s} {row['momentum_score']:.1f}    "
              f"{_f(row['sub_cs_mom'])} {_f(row['sub_high52'])} "
              f"{_f(row['sub_vol_scaled'])} {_f(row['sub_residual'])}")
    print(f"  Regime: {regime_label}")
    print(f"  Output: {OUT_CSV.name}")
    print(f"{'─'*65}\n")

    return out


# =============================================================================
# 6. Report writer
# =============================================================================

def _write_report(
    df: pd.DataFrame,
    regime: str,
    vix: float | None,
    total_mult: float,
    was_dampened: bool,
    n_cs: int, n_52: int, n_vs: int, n_res: int,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    _vix_str_r = f"{vix:.1f}" if vix is not None else "N/A"

    lines = [
        "# Canyon v9 — Momentum Signal Report (Step 111)",
        f"Generated: {ts}",
        "",
        "## Signal Coverage",
        "",
        f"| Sub-Signal | Tickers | Method | Reference |",
        f"|-----------|---------|--------|-----------|",
        f"| Cross-sectional momentum | {n_cs} | 12-1 month return, cross-sectional rank | Jegadeesh & Titman (1993) |",
        f"| 52-week high ratio | {n_52} | price / 252d rolling max | George & Hwang (2004) |",
        f"| Vol-scaled momentum | {n_vs} | mom_12_1 / realised_vol_21d | Barroso & Santa-Clara (2015) / AQR |",
        f"| Residual momentum | {n_res} | cumulative OLS residual vs SPY | Blitz, Huij & Martens (2011) |",
        "",
        "## Composite Weights",
        "",
        f"| Sub-Signal | Weight |",
        f"|-----------|--------|",
        f"| Cross-sectional momentum | 35% |",
        f"| 52-week high ratio | 30% |",
        f"| Volatility-scaled momentum | 25% |",
        f"| Residual momentum | 10% |",
        "",
        "## Crash Protection",
        "",
        f"| Setting | Value |",
        f"|---------|-------|",
        f"| Regime | {regime} |",
        f"| VIX | {_vix_str_r} |",
        f"| Total Dampening Multiplier | {total_mult:.2f}× |",
        f"| Was Dampened | {'Yes — scores blended toward neutral 50' if was_dampened else 'No — full signal'} |",
        "",
    ]

    if was_dampened:
        lines += [
            "> ⚠️  **Momentum crash protection active.**  "
            f"Scores dampened toward neutral by ×{total_mult:.2f}.  ",
            "> In BEAR regimes and high-VIX environments, momentum historically crashes  ",
            "> as prior-year losers surge violently during recovery rallies.  ",
            "> AQR, Two Sigma, and D.E. Shaw all use regime-conditional momentum scaling.",
            "",
        ]

    lines += [
        "## Top 20 Momentum Picks",
        "",
        "| Rank | Ticker | Score | CS_Mom | 52wH | VolScl | Residual | 12m_Raw% |",
        "|------|--------|-------|--------|------|--------|----------|---------|",
    ]

    for _, row in df.head(20).iterrows():
        def _f(v, fmt=".1f"):
            return f"{v:{fmt}}" if pd.notna(v) else "N/A"
        lines.append(
            f"| {int(row.get('momentum_rank', 0))} "
            f"| {row['ticker']} "
            f"| {_f(row['momentum_score'])} "
            f"| {_f(row['sub_cs_mom'])} "
            f"| {_f(row['sub_high52'])} "
            f"| {_f(row['sub_vol_scaled'])} "
            f"| {_f(row['sub_residual'])} "
            f"| {_f(row.get('cs_mom_raw_pct', float('nan'))*100, '+.1f')}% |"
        )

    lines += [
        "",
        "## Bottom 10 (Momentum Avoidance List)",
        "",
        "| Rank | Ticker | Score | CS_Mom | 52wH | 12m_Raw% |",
        "|------|--------|-------|--------|------|---------|",
    ]
    for _, row in df.tail(10).sort_values("momentum_score").iterrows():
        def _f(v, fmt=".1f"):
            return f"{v:{fmt}}" if pd.notna(v) else "N/A"
        raw_pct = row.get("cs_mom_raw_pct", float("nan"))
        raw_str = f"{raw_pct*100:+.1f}%" if pd.notna(raw_pct) else "N/A"
        lines.append(
            f"| {int(row.get('momentum_rank', 0))} "
            f"| {row['ticker']} "
            f"| {_f(row['momentum_score'])} "
            f"| {_f(row['sub_cs_mom'])} "
            f"| {_f(row['sub_high52'])} "
            f"| {raw_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## How to Read This Signal",
        "",
        "- **Score 80–100**: Strong momentum — stock is outperforming peers on all four measures.",
        "  Best entries are stocks with high CS_Mom AND high 52wH (near their yearly peak).",
        "- **Score 50–80**: Moderate momentum — positive but not top-decile.",
        "- **Score 20–50**: Weak or mixed — some measures negative.",
        "- **Score 0–20**: Momentum reversal candidates — consider as short/avoid signals.",
        "",
        "### Why Skip the Most Recent Month?",
        "Short-term return reversal (1-month) is well-documented: stocks that rose last month",
        "tend to slightly reverse next month due to bid-ask bounce and microstructure noise.",
        "Skipping t-21 to t-0 avoids this contamination (Jegadeesh & Titman, 1993).",
        "",
        "### Why 52-Week High?",
        "Investors anchor mentally to historical highs. When a stock approaches its 52-week",
        "high, analysts and portfolio managers hesitate to raise targets or add exposure —",
        "but eventually capitulate when the high is breached, creating sustained outperformance.",
        "(George & Hwang, 2004 — one of the most-replicated findings in behavioral finance.)",
        "",
        "### Why Volatility-Scale?",
        "A 40% return on a 60%-vol biotech carries little signal (expected at 1σ).",
        "A 40% return on a 15%-vol large-cap is 2.7σ — that's real momentum.",
        "Dividing by realized vol gives each stock's 'momentum Sharpe ratio'.",
        "AQR's Momentum Fund uses this adjustment to halve momentum crash risk.",
        "",
        "### Crash Protection Logic",
        "Momentum crashes occur when the market transitions from BEAR to BULL:",
        "the stocks momentum would short (recent losers) surge first and hardest.",
        "In BEAR regime this signal is dampened ×0.45; VIX>35 adds ×0.60.",
        "This is consistent with AQR's published momentum crash defense (Momentum Momentum).",
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"  [written] {OUT_MD}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 111 — Institutional Momentum Signal"
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="Return only top N tickers (default: all)"
    )
    parser.add_argument(
        "--lookback", type=int, default=LOOKBACK_DAYS,
        help=f"Lookback window in trading days (default {LOOKBACK_DAYS} = 12 months)"
    )
    parser.add_argument(
        "--no-regime-damp", action="store_true",
        help="Disable momentum crash protection dampening"
    )
    args = parser.parse_args()

    compute_momentum_scores(
        top_n=args.top,
        lookback=args.lookback,
        no_regime_damp=args.no_regime_damp,
    )
