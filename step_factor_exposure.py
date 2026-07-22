#!/usr/bin/env python3
"""
Canyon — step_factor_exposure.py
==================================
Compute real-time factor exposures of the live Alpaca paper portfolio:
  • Sector concentration (% AUM by GICS sector)
  • Market-cap bucket distribution (Mega / Large / Mid / Small)
  • Beta-weighted net & gross exposure
  • Long vs short leg breakdown (for SHORT book)
  • Single-name concentration (top 5 positions)

Reads:
  alpaca_book_state.json    — current positions by book
  alpha_scores.csv          — sector, market_cap (if present)
  sp500_price_cache.csv     — used to compute rolling 1-year beta vs SPY

Outputs:
  factor_exposure.json      — structured exposure report
  factor_exposure.csv       — one row per ticker with book/sector/beta/cap

Runs after step_alpaca_execution.py (Step 384).
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg): print(f"  {msg}")
def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")

# Market-cap buckets (approximate, USD billions)
CAP_BUCKETS = [
    ("Mega",  300e9),
    ("Large",  10e9),
    ("Mid",    2e9),
    ("Small",  0),
]


# ── Load live positions ───────────────────────────────────────────────────────

def load_positions() -> list[dict]:
    """
    Returns list of {ticker, book, dollar_value, is_short}
    dollar_value is positive for longs, negative for shorts.
    """
    state_path = ROOT / "alpaca_book_state.json"
    if not state_path.exists():
        warn("alpaca_book_state.json not found")
        return []

    try:
        state = json.loads(state_path.read_text())
    except Exception as e:
        err(f"Could not read state: {e}")
        return []

    rows = []
    for book_name, data in state.items():
        positions = data.get("positions", {})
        long_tickers  = set(data.get("long_tickers",  []))
        short_tickers = set(data.get("short_tickers", []))

        for ticker, val in positions.items():
            is_short = (val < 0) or (ticker in short_tickers and ticker not in long_tickers)
            rows.append({
                "ticker":       ticker,
                "book":         book_name,
                "dollar_value": float(val),
                "is_short":     is_short,
            })
    return rows


# ── Load sector / market-cap metadata ────────────────────────────────────────

def load_ticker_meta() -> pd.DataFrame:
    """Returns DataFrame[ticker → sector, market_cap_approx]."""
    meta_cols = {"ticker", "sector", "market_cap", "market_cap_approx", "market_cap_usd"}
    for fname in ["alpha_scores.csv", "daily_picks.csv"]:
        p = ROOT / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            avail = [c for c in df.columns if c in meta_cols]
            if "ticker" in avail and len(avail) > 1:
                df = df[avail].drop_duplicates("ticker").set_index("ticker")
                # Normalise market-cap column name
                for alias in ("market_cap_approx", "market_cap_usd", "market_cap"):
                    if alias in df.columns:
                        df["market_cap"] = df[alias]
                        break
                ok(f"Loaded ticker metadata from {fname} ({len(df)} tickers)")
                return df
        except Exception:
            pass
    return pd.DataFrame()


# ── Compute rolling beta vs SPY ───────────────────────────────────────────────

def compute_betas(tickers: list[str], window: int = 252) -> dict[str, float]:
    """Compute 1-year rolling beta for each ticker vs SPY."""
    price_path = ROOT / "sp500_price_cache.csv"
    if not price_path.exists():
        return {}

    try:
        prices = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
        # Use SPY as market proxy — it may be in the cache or we compute a equal-weight proxy
        spy_path = ROOT / "spy_price_cache.csv"
        if spy_path.exists():
            spy = pd.read_csv(spy_path, index_col=0, parse_dates=True).sort_index().iloc[:, 0]
        else:
            # Fallback: equal-weighted average of all tickers
            spy = prices.pct_change().mean(axis=1).add(1).cumprod()
            spy = spy / spy.iloc[0] * 100

        ret_spy    = spy.pct_change().dropna()
        ret_prices = prices[tickers].pct_change().dropna() if tickers else pd.DataFrame()

        if ret_prices.empty or len(ret_prices) < 60:
            return {}

        recent_spy    = ret_spy.iloc[-window:]
        recent_prices = ret_prices.iloc[-window:]
        aligned       = recent_prices.reindex(recent_spy.index).dropna(how="all")

        betas = {}
        for tk in tickers:
            if tk not in aligned.columns:
                continue
            common = aligned[tk].dropna().index.intersection(recent_spy.index)
            if len(common) < 60:
                betas[tk] = 1.0
                continue
            x = recent_spy.loc[common].values
            y = aligned.loc[common, tk].values
            cov = np.cov(x, y, ddof=1)
            betas[tk] = float(cov[0, 1] / cov[0, 0]) if cov[0, 0] > 0 else 1.0
        return betas
    except Exception as e:
        warn(f"Beta computation failed: {e}")
        return {}


# ── Cap bucket ────────────────────────────────────────────────────────────────

def cap_bucket(mktcap_usd: float | None) -> str:
    if mktcap_usd is None or pd.isna(mktcap_usd):
        return "Unknown"
    for name, threshold in CAP_BUCKETS:
        if mktcap_usd >= threshold:
            return name
    return "Unknown"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}Canyon — Factor Exposure{RESET}  {TODAY}")

    positions = load_positions()
    if not positions:
        warn("No positions to analyse")
        return

    meta  = load_ticker_meta()
    pos_df = pd.DataFrame(positions)
    tickers = pos_df["ticker"].unique().tolist()

    log(f"Analysing {len(tickers)} positions across {pos_df['book'].nunique()} books …")

    # Merge metadata
    if not meta.empty:
        pos_df = pos_df.join(meta, on="ticker", how="left")
    if "sector" not in pos_df.columns:
        pos_df["sector"] = "Unknown"
    if "market_cap" not in pos_df.columns:
        pos_df["market_cap"] = np.nan

    # Compute betas
    log("Computing 1-year rolling betas …")
    betas = compute_betas(tickers)
    pos_df["beta"] = pos_df["ticker"].map(betas).fillna(1.0)
    pos_df["cap_bucket"] = pos_df["market_cap"].apply(cap_bucket)
    pos_df["beta_adj_exposure"] = pos_df["dollar_value"] * pos_df["beta"]

    total_long  = pos_df.loc[~pos_df["is_short"], "dollar_value"].sum()
    total_short = pos_df.loc[ pos_df["is_short"], "dollar_value"].sum()  # negative
    gross_exp   = pos_df["dollar_value"].abs().sum()
    net_exp     = total_long + total_short   # short is negative
    net_beta    = pos_df["beta_adj_exposure"].sum()

    ok(f"Long: ${total_long:,.0f}  Short: ${total_short:,.0f}  "
       f"Net: ${net_exp:,.0f}  Beta-adj net: ${net_beta:,.0f}")

    # ── Sector concentration ──────────────────────────────────────────────
    sector_aum = (
        pos_df.groupby("sector")["dollar_value"].sum()
            .sort_values(ascending=False)
    )
    sector_pct = (sector_aum / gross_exp * 100).round(1)

    print(f"\n  {'Sector':<30} {'AUM $':>12} {'%AUM':>8}")
    print(f"  {'─'*30} {'─'*12} {'─'*8}")
    for sec, pct in sector_pct.items():
        bar = "█" * int(abs(pct) / 3)
        sign = "+" if sector_aum[sec] >= 0 else "-"
        print(f"  {sec:<30} {sign}${abs(sector_aum[sec]):>10,.0f} {pct:>7.1f}% {bar}")

    # ── Cap bucket distribution ───────────────────────────────────────────
    cap_aum = (
        pos_df.groupby("cap_bucket")["dollar_value"].agg(lambda x: x.abs().sum())
            .reindex(["Mega", "Large", "Mid", "Small", "Unknown"], fill_value=0)
    )
    cap_pct = (cap_aum / gross_exp * 100).round(1)

    print(f"\n  {'Cap Bucket':<12} {'Gross AUM':>12} {'%':>8}")
    print(f"  {'─'*12} {'─'*12} {'─'*8}")
    for bucket, pct in cap_pct.items():
        if cap_aum[bucket] > 0:
            print(f"  {bucket:<12} ${cap_aum[bucket]:>10,.0f} {pct:>7.1f}%")

    # ── Top 5 single-name concentration ──────────────────────────────────
    top5 = pos_df.nlargest(5, pos_df["dollar_value"].abs().name if False else "dollar_value")
    top5_by_abs = pos_df.reindex(pos_df["dollar_value"].abs().sort_values(ascending=False).index).head(5)

    print(f"\n  Top 5 positions by gross exposure:")
    for _, r in top5_by_abs.iterrows():
        direction = "SHORT" if r["is_short"] else "LONG"
        print(f"  {r['ticker']:>6}  {direction}  ${abs(r['dollar_value']):>9,.0f}  "
              f"β={r['beta']:.2f}  {r.get('sector', '?')}")

    # ── Save outputs ──────────────────────────────────────────────────────
    exposure_report = {
        "as_of":           TODAY,
        "n_positions":     len(pos_df),
        "gross_exposure":  round(float(gross_exp), 2),
        "net_exposure":    round(float(net_exp), 2),
        "net_beta_adj":    round(float(net_beta), 2),
        "long_aum":        round(float(total_long), 2),
        "short_aum":       round(float(total_short), 2),
        "l_s_ratio":       round(float(total_long / abs(total_short)), 2) if total_short != 0 else None,
        "sector_pct":      sector_pct.to_dict(),
        "cap_bucket_pct":  cap_pct.to_dict(),
        "books":           {
            book: {
                "n_positions": int(grp.shape[0]),
                "net_aum":     round(float(grp["dollar_value"].sum()), 2),
                "gross_aum":   round(float(grp["dollar_value"].abs().sum()), 2),
                "avg_beta":    round(float(grp["beta"].mean()), 3),
            }
            for book, grp in pos_df.groupby("book")
        },
    }

    out_json = ROOT / "factor_exposure.json"
    with open(out_json, "w") as f:
        json.dump(exposure_report, f, indent=2, default=str)
    ok(f"factor_exposure.json saved")

    out_csv = ROOT / "factor_exposure.csv"
    pos_df.to_csv(out_csv, index=False)
    ok(f"factor_exposure.csv → {len(pos_df)} rows")

    print(f"\n{GREEN}✓ Factor exposure analysis complete{RESET}\n")


if __name__ == "__main__":
    main()
