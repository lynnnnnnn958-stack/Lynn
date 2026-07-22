#!/usr/bin/env python3
"""
Canyon v9 — Step 86: Cross-Asset Momentum Signal
=================================================
Extracts equity-predictive signals from commodity, FX, and fixed income
momentum patterns. Cross-asset flows often lead equity sector rotation.

Signal design (all free data, no paid sources):
  Commodities:  GLD, SLV, USO, UNG, CORN, WEAT → sector mapping
  FX / Dollar:  UUP (USD Index ETF), FXE, FXY, FXF → risk-on/off
  Fixed income: TLT (long-duration), HYG (high-yield credit)
  Volatility:   VXX, SVXY → fear gauge
  Intermarket:  Copper (COPX) / Gold ratio → global growth
  Yield curve:  TLT vs SHY slope → recession indicator

Intermarket → equity mapping:
  Dollar DOWN  → EM outperform, commodity stocks boost
  Copper UP    → industrials / materials outperform
  HYG UP (credit spreads tightening) → risk-on → small caps
  TLT DOWN (yields rising) → financials outperform, utilities underperform
  Gold UP      → defensives outperform
  Oil UP       → energy sector boost

Output:
  cross_asset_signals.csv  — sector-level signals + per-ticker scores
  cross_asset_report.md    — daily intermarket summary

Freshness: regenerated daily (free data, fast to fetch)

Usage:
  python3 canyon_final_v9_step86_cross_asset.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
OUT_CSV    = ROOT / "cross_asset_signals.csv"
OUT_REPORT = ROOT / "cross_asset_report.md"

LOOKBACK_DAYS = 63   # 3-month momentum window
SIGNAL_DAYS   = 5    # 1-week momentum for confirmation
FRESHNESS_DAYS = 1   # regenerate daily

# ── Cross-asset universe ──────────────────────────────────────────────────────

CROSS_ASSET = {
    # Macro ETFs and their role in intermarket analysis
    "UUP":  ("fx",      "dollar_idx"),
    "FXE":  ("fx",      "euro"),
    "FXY":  ("fx",      "yen"),
    "GLD":  ("gold",    "gold"),
    "SLV":  ("silver",  "silver"),
    "USO":  ("oil",     "oil"),
    "COPX": ("copper",  "copper_miners"),
    "TLT":  ("rates",   "long_bond"),
    "SHY":  ("rates",   "short_bond"),
    "HYG":  ("credit",  "high_yield"),
    "LQD":  ("credit",  "inv_grade"),
    "VXX":  ("vol",     "vix_futures"),
    "XME":  ("metals",  "metals_mining"),
    "XLE":  ("energy",  "energy_sector"),
    "XLB":  ("material","materials_sector"),
}

# ── Sector impact mapping ─────────────────────────────────────────────────────
# (cross_asset_signal, direction) → (equity_sector_etf, impact_direction)
# direction: +1 = rising favors sector; -1 = falling favors sector
IMPACT_MAP = [
    ("dollar_idx",    -1,  "XLK",  +0.4),   # weak USD → tech/EM positive
    ("dollar_idx",    -1,  "XLB",  +0.6),   # weak USD → materials
    ("dollar_idx",    -1,  "XLI",  +0.4),   # weak USD → industrials
    ("dollar_idx",    +1,  "XLU",  +0.3),   # strong USD → defensives
    ("oil",           +1,  "XLE",  +0.8),   # oil up → energy
    ("oil",           +1,  "XLB",  +0.3),   # oil up → materials
    ("oil",           -1,  "XLY",  +0.3),   # oil down → consumer discretionary
    ("gold",          +1,  "XLB",  +0.4),   # gold up → materials
    ("gold",          +1,  "XLU",  +0.3),   # gold up → utilities (defensive)
    ("copper_miners", +1,  "XLI",  +0.5),   # copper up → industrials
    ("copper_miners", +1,  "XLB",  +0.5),   # copper up → materials
    ("long_bond",     -1,  "XLF",  +0.6),   # falling TLT (rising rates) → financials
    ("long_bond",     -1,  "XLU",  -0.5),   # falling TLT → utilities hurt
    ("long_bond",     +1,  "XLU",  +0.5),   # rising TLT (falling rates) → utilities
    ("long_bond",     +1,  "XLRE", +0.5),   # rising TLT → REITs
    ("high_yield",    +1,  "XLY",  +0.5),   # HYG up (credit ok) → risk-on
    ("high_yield",    +1,  "XLK",  +0.4),   # HYG up → tech risk-on
    ("high_yield",    -1,  "XLU",  +0.4),   # HYG down (stress) → defensives
    ("vix_futures",   +1,  "XLU",  +0.5),   # VIX up → defensives
    ("vix_futures",   +1,  "XLV",  +0.3),   # VIX up → healthcare (defensive)
    ("vix_futures",   -1,  "XLY",  +0.5),   # VIX down → consumer discretionary
    ("vix_futures",   -1,  "XLK",  +0.5),   # VIX down → tech (risk-on)
]

SECTOR_TICKERS = {
    "XLE": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY"],
    "XLF": ["JPM","BAC","WFC","GS","MS","BLK","AXP","C","BK","MMC"],
    "XLB": ["LIN","APD","SHW","ECL","NEM","FCX","NUE","CF","MOS"],
    "XLI": ["HON","UNP","CAT","GE","BA","DE","WM","ETN","CSX","NOC"],
    "XLU": ["NEE","DUK","SO","D","AEP","EXC","SRE","PEG","XEL","ED"],
    "XLK": ["AAPL","MSFT","NVDA","AVGO","AMD","ORCL","INTC","CSCO","AMAT","TXN"],
    "XLV": ["UNH","JNJ","LLY","ABT","MRK","TMO","AMGN","DHR","SYK","MDT"],
    "XLY": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","BKNG","LOW","CMG"],
    "XLRE":["AMT","PLD","CCI","EQIX","PSA","O","SPG","DLR","AVB","EQR"],
}


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], days: int = 130) -> pd.DataFrame:
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=f"{days}d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"].dropna(how="all", axis=1)
        return raw.dropna(how="all", axis=1)
    except Exception as exc:
        print(f"  [CrossAsset] yfinance error: {exc}")
        return pd.DataFrame()


# ── Signal computation ────────────────────────────────────────────────────────

def compute_momentum(prices: pd.DataFrame, window: int) -> pd.Series:
    if len(prices) < window + 1:
        return pd.Series(dtype=float)
    ret = prices.iloc[-1] / prices.iloc[-(window+1)] - 1
    mu, sd = ret.mean(), ret.std()
    if sd < 1e-9:
        return ret * 0.0
    return ((ret - mu) / sd).rename("z_score")


def compute_cross_asset_scores(prices: pd.DataFrame) -> dict[str, float]:
    """
    For each cross-asset driver, compute z-scored momentum signal.
    Returns dict of {role_name: z_score}.
    """
    scores = {}
    for ticker, (asset_class, role) in CROSS_ASSET.items():
        if ticker not in prices.columns:
            continue
        px = prices[ticker].dropna()
        if len(px) < LOOKBACK_DAYS + 1:
            continue
        ret_63 = float(px.iloc[-1] / px.iloc[-(LOOKBACK_DAYS+1)] - 1)
        ret_5  = float(px.iloc[-1] / px.iloc[-(SIGNAL_DAYS+1)] - 1) if len(px) > SIGNAL_DAYS else 0.0
        # Blend: 70% 3m momentum, 30% 1w confirmation
        blended = 0.70 * ret_63 + 0.30 * ret_5
        scores[role] = blended
    return scores


def compute_sector_signals(ca_scores: dict[str, float]) -> dict[str, float]:
    """
    Map cross-asset scores → equity sector scores using IMPACT_MAP.
    Returns {sector_etf: aggregate_signal}.
    """
    sector_raw: dict[str, float] = {}
    for role, direction, sector_etf, weight in IMPACT_MAP:
        if role not in ca_scores:
            continue
        ca_signal = ca_scores[role] * direction * weight
        sector_raw[sector_etf] = sector_raw.get(sector_etf, 0.0) + ca_signal

    return sector_raw


def expand_to_tickers(sector_scores: dict[str, float]) -> pd.Series:
    """
    Assign each stock the score of its primary sector.
    Stocks in sectors not in IMPACT_MAP get score 0.
    """
    rows = {}
    for sector_etf, score in sector_scores.items():
        for ticker in SECTOR_TICKERS.get(sector_etf, []):
            rows[ticker] = rows.get(ticker, 0.0) + score
    s = pd.Series(rows)
    mu, sd = s.mean(), s.std()
    return ((s - mu) / (sd + 1e-9)).rename("cross_asset_z")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    if OUT_CSV.exists():
        age = (datetime.now().timestamp() - OUT_CSV.stat().st_mtime) / 86400
        if age < FRESHNESS_DAYS:
            print(f"  [CrossAsset] Output {age:.1f}d old — skipping.")
            return pd.read_csv(OUT_CSV)

    print("  [CrossAsset] Fetching cross-asset prices …")
    tickers_to_fetch = list(CROSS_ASSET.keys())
    prices = fetch_prices(tickers_to_fetch, days=130)

    if prices.empty:
        print("  [CrossAsset] No data — skipping")
        return pd.DataFrame()

    print(f"  [CrossAsset] Got {len(prices.columns)} tickers, {len(prices)} rows")

    ca_scores      = compute_cross_asset_scores(prices)
    sector_scores  = compute_sector_signals(ca_scores)
    ticker_scores  = expand_to_tickers(sector_scores)

    # Build output DataFrame
    rows = []
    for ticker, z in ticker_scores.items():
        sector = next((s for s, tks in SECTOR_TICKERS.items() if ticker in tks), "")
        rows.append({
            "ticker":          ticker,
            "sector":          sector,
            "cross_asset_z":   round(z, 4),
            "sector_score":    round(sector_scores.get(sector, 0.0), 4),
            "updated_date":    datetime.now().strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows).sort_values("cross_asset_z", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"  [CrossAsset] Saved {len(df)} tickers → {OUT_CSV.name}")
    return df


def write_report(df: pd.DataFrame, ca_scores: dict[str, float],
                 sector_scores: dict[str, float]) -> None:
    if df.empty:
        return
    def _pct(v):
        return f"{v:+.3f}" if not pd.isna(v) else "—"

    ca_rows = "".join(f"| {k} | {v:+.4f} |\n" for k,v in sorted(
        ca_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    sec_rows = "".join(f"| **{k}** | {v:+.4f} |\n" for k,v in sorted(
        sector_scores.items(), key=lambda x: x[1], reverse=True))
    top_long  = df.head(5)[["ticker","cross_asset_z"]].to_string(index=False)
    top_short = df.tail(5)[["ticker","cross_asset_z"]].to_string(index=False)

    report = f"""# Cross-Asset Signal Report — {datetime.now():%Y-%m-%d}

## Intermarket Scores

| Driver | Signal (z) |
|--------|:----------:|
{ca_rows}
## Sector Impact (Composite)

| Sector ETF | Aggregate Score |
|------------|:--------------:|
{sec_rows}
## Top 5 Long (Cross-Asset Tailwind)

```
{top_long}
```

## Top 5 Short (Cross-Asset Headwind)

```
{top_short}
```

---
*Cross-asset signals are overlaid on the stock-specific composite.
Weight in IC_WEIGHTS: sig_cross_asset = 0.03.*
"""
    OUT_REPORT.write_text(report)
    print(f"  [CrossAsset] Report → {OUT_REPORT.name}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"Canyon v9 — Cross-Asset Momentum  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60 + "\n")

    prices        = fetch_prices(list(CROSS_ASSET.keys()), days=130)
    ca_scores     = compute_cross_asset_scores(prices)
    sector_scores = compute_sector_signals(ca_scores)
    ticker_scores = expand_to_tickers(sector_scores)

    rows = []
    for ticker, z in ticker_scores.items():
        sector = next((s for s, tks in SECTOR_TICKERS.items() if ticker in tks), "")
        rows.append({"ticker": ticker, "sector": sector,
                     "cross_asset_z": round(z,4),
                     "sector_score": round(sector_scores.get(sector,0.0),4),
                     "updated_date": datetime.now().strftime("%Y-%m-%d")})
    df = pd.DataFrame(rows).sort_values("cross_asset_z", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    write_report(df, ca_scores, sector_scores)

    print(f"\n[Top intermarket drivers]")
    for k, v in sorted(ca_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:6]:
        print(f"  {k:20s} {v:+.4f}")

    print("\n" + "=" * 60)
    print("Step 86 Complete")
    print("=" * 60)
