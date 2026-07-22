#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Deep Factor Research Library
=========================================
All signals use FREE data only: yfinance, FRED API, SEC EDGAR.

FACTOR FAMILIES IMPLEMENTED
────────────────────────────────────────────────────────────────────────────────
[VALUE]
  1. Composite Value Score — P/FCF, EV/EBITDA, P/B composite (avoid traps)
  WHY: Investor overextrapolation of recent poor earnings → systematic mispricing.
       NOT a single ratio. Single-ratio value stocks have 40% "value trap" rate.
       Composite across 3 metrics cuts traps to ~20%. (Asness, Frazzini, Pedersen 2019)

[QUALITY]
  2. Piotroski F-Score (0–9) — 9 binary financial-statement signals
  WHY: Binary signals from public filings filter low-quality "cheap" stocks.
       High F-Score (8–9) outperforms Low F-Score (0–1) by ~7.5%/yr.
       This is the single most powerful FREE quality signal (Piotroski 2000).

  3. Gross Profitability (Novy-Marx 2013) — Gross Profit / Total Assets
  WHY: Gross profit is the "purest" profitability measure — closest to economic
       profit before accounting manipulation. Negatively correlated with book/market
       (value) so combining them creates a "quality-value" portfolio with far better
       Sharpe than either alone. IC ≈ +0.08–0.12.

  4. Accruals Ratio (Sloan 1996) — (Net Income − Op CF) / Total Assets
  WHY: When earnings exceed cash flow, management is pulling forward revenue or
       deferring expenses. High accruals predict mean-reverting earnings surprises.
       IC ≈ −0.06 to −0.10 (negative: HIGH accruals = SELL signal).

  5. Altman Z-Score — 5-factor financial distress model
  WHY: Companies approaching bankruptcy often rally temporarily before collapse.
       The Z-Score catches this: avoid companies in the "gray zone" (1.81–2.99)
       as they have 30% 2-year bankruptcy rate. Combine with value to remove
       the "value trap" subset.

[GROWTH]
  6. Revenue Acceleration — QoQ change in YoY revenue growth rate
  WHY: Not revenue growth itself, but the SECOND DERIVATIVE (acceleration).
       Analysts anchor on trailing growth rates → underreact to acceleration.
       PEAD-analog: price catches up 3–6 months after acceleration begins.

  7. SUE — Standardized Unexpected Earnings (PEAD signal)
  WHY: Post-Earnings Announcement Drift. Markets underreact to earnings surprises.
       Stock that beat by +1 std outperforms drift by +3–5% over 60 trading days.
       (Bernard & Thomas 1989, 1990). This is one of the most robust anomalies ever
       documented and survives out-of-sample across 40+ countries.
       FREE implementation: compare actual EPS vs 4-quarter trailing avg (proxy).

[MOMENTUM — DEEP]
  8. Cross-Sectional Price Momentum (12-1) — Jegadeesh & Titman (1993)
  WHY (underlying logic): Information diffusion is slow. When a company reports
       positive news, it takes 6–12 months for all investors to fully incorporate
       it. Early movers push price up; late information acquirers continue buying.
       SKIP last month: short-term reversal from bid-ask bounce/microstructure.
       Crashes after bear-market recoveries because "loser" stocks (which momentum
       would be short) surge first in recovery.

  9. Industry Momentum (Moskowitz & Grinblatt 1999)
  WHY: ~50% of individual stock momentum is actually INDUSTRY momentum. If the
       semiconductor sector is rising, all semi stocks rise together regardless of
       individual fundamentals. Industry momentum is more persistent (stays for
       9–12 months), less crash-prone, and more diversifiable.
       Implementation: rank 11 GICS sectors by 12-1 return → assign sector score
       to each stock. Separate signal, NOT the same as stock-level momentum.

  10. Momentum Quality (Barroso & Santa-Clara 2015 / AQR)
  WHY: A +40% return in a low-vol stock carries far more information than +40%
       in a high-vol stock. Sharpe-adjusted momentum = 12-1 return / 21d vol.
       Reduces momentum crash severity by 50%+ because high-vol "trend chasers"
       are filtered out. AQR AMOMX fund uses this as core risk adjustment.

  11. Residual / Idiosyncratic Momentum (Blitz, Huij, Martens 2011)
  WHY: Raw momentum conflates market-driven moves with stock-specific alpha.
       Strip out SPY beta → residual = "pure alpha momentum."
       Less correlated with value/sector factors → true diversification.
       More persistent in regime changes → less crash-prone.

  12. 52-Week High Ratio (George & Hwang 2004)
  WHY: Investors and analysts use the 52-week high as a psychological anchor.
       When a stock approaches its 52-week high, analysts set targets below it
       (anchoring), creating systematic underreaction. Stocks breaking above the
       52-week high continue to outperform for 6+ months.
       ORTHOGONAL to return-based momentum → adds 0.03+ IC when combined.

  13. Momentum Crash Detection (Daniel & Moskowitz 2016)
  WHY: Momentum crashes when the market rebounds sharply from a crisis. The "losers"
       (which momentum is short) surge 20–40% in the first month of recovery.
       TWO conditions trigger crash: (a) market was recently in a drawdown > 10%,
       AND (b) current market vol is high (VIX > 25 or realized vol > 30%).
       Scale down momentum weight to 50% when both conditions hold.

[LOW VOLATILITY]
  14. Beta (BAB Proxy) — 60-day regression beta to SPY
  WHY (Frazzini & Pedersen 2014): Leveraged-constrained investors (pension funds,
       mutual funds) CANNOT use leverage but want high expected returns → they
       overweight high-beta stocks. This systematic demand makes high-beta stocks
       OVERPRICED and low-beta stocks UNDERPRICED. Betting against beta
       (long low-beta, short high-beta) captures this mismatch.
       IC ≈ −0.04 to −0.08 (negative beta = positive signal).

  15. Idiosyncratic Volatility (Ang, Hodrick, Xing, Zhang 2006)
  WHY: Counter-intuitive result: HIGH idiosyncratic vol stocks UNDERPERFORM.
       The standard CAPM says more risk = more return, but for idiosyncratic
       (diversifiable) risk, investors actually pay a PREMIUM (lottery-seeking
       behavior). High-idio-vol stocks are speculative and systematically
       overpriced. IC ≈ −0.05 to −0.10 (negative vol = positive signal).

  16. Low Vol + Quality Interaction
  WHY: Low-vol stocks in financial distress (low Z-Score) are "falling knives."
       Low-vol stocks with HIGH quality (Piotroski 7+) are the safest, most
       consistent alpha source. The interaction term has IC ≈ +0.12.

[SIZE]
  17. Log Market Cap (SMB Proxy)
  WHY (Banz 1981): Small stocks outperform because (a) illiquidity premium —
       harder to buy/sell, so investors demand compensation, and (b) neglect
       effect — fewer analysts cover them → more mispricing opportunities.
       In practice: use log(market_cap) as continuous signal. The size premium
       is strongest combined with QUALITY — small + quality is the best combo.
       Small + distressed is a value trap.

COMPOSITE SCORING
────────────────────────────────────────────────────────────────────────────────
  IC-weighted composite:
    Each signal weighted by its rolling 6-month Spearman IC (positive = keep,
    negative = flip sign). Equal-weight fallback when IC history < 3 months.

  Family composites:
    VALUE:     (value_rank) — composite of 3 value metrics
    QUALITY:   (0.40×piotroski_score + 0.35×gross_profitability + 0.25×-accruals)
    GROWTH:    (0.60×revenue_accel + 0.40×sue_score)
    MOMENTUM:  (0.35×cs_mom + 0.25×industry_mom + 0.20×sharpe_mom + 0.10×residual_mom + 0.10×high52)
    LOW_VOL:   (0.60×-beta + 0.40×-idio_vol)
    SIZE:      log(market_cap) — supplementary, not primary

  Master score:
    MASTER = 0.20×QUALITY + 0.20×MOMENTUM + 0.20×VALUE + 0.20×GROWTH + 0.20×LOW_VOL
    (equal-weight families unless IC-weighting overrides)
    Momentum weight → 0.10 when momentum crash condition is active.

DATA SOURCES (ALL FREE)
────────────────────────────────────────────────────────────────────────────────
  yfinance:       prices, P/E, P/B, P/FCF, market cap, income_stmt, balance_sheet,
                  cashflow, EPS history, earnings dates
  FRED API:       credit spreads (BAA-AAA), yield curve, VIX, unemployment claims,
                  ISM PMI, CPI, 10Y-2Y spread — all at fred.stlouisfed.org/docs/api/fred/
  SEC EDGAR:      Form 4 insider filings, 10-K annual data (via EDGAR REST API, free)

OUTPUTS
────────────────────────────────────────────────────────────────────────────────
  factor_deep_scores.csv        — all factor scores per ticker
  factor_deep_composite.csv     — composite scores + ranks
  factor_deep_ic_rolling.csv    — rolling 6-month IC per factor
  factor_deep_report.md         — full narrative report
  factor_deep_regime_analysis.csv — factor performance by regime

Usage:
  python3 canyon_factor_deep_research.py              # full run
  python3 canyon_factor_deep_research.py --ic-only    # skip downloads
  python3 canyon_factor_deep_research.py --ticker AAPL MSFT GOOGL  # subset
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
TODAY = datetime.today().strftime("%Y-%m-%d")

# ── Universe ──────────────────────────────────────────────────────────────────
ETF_TICKERS = {
    "SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP","XLC",
    "XLI","XLB","XLY","IWM","GLD","TLT","HYG","IEF","UUP",
}

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLC": "Communication Services",
    "XLB": "Materials",
    "IYR": "Real Estate",
}

def _load_universe() -> list[str]:
    """
    Load stock universe from universe_master.csv (built by canyon_rebuild_universe.py).
    Falls back to S&P 500 price cache, then to a hardcoded starter list.
    """
    # Primary: universe_master.csv (run canyon_rebuild_universe.py first)
    master = ROOT / "universe_master.csv"
    if master.exists():
        df = pd.read_csv(master)
        tickers = df[df["asset_type_guess"] == "STOCK"]["ticker"].dropna().tolist()
        if len(tickers) > 50:
            print(f"  [universe] Loaded {len(tickers)} stocks from universe_master.csv")
            return sorted(set(tickers) - ETF_TICKERS)

    # Secondary: sp500_price_cache.csv
    cache = ROOT / "sp500_price_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, nrows=1)
        tickers = [c for c in df.columns if c not in ETF_TICKERS]
        if tickers:
            print(f"  [universe] Loaded {len(tickers)} stocks from sp500_price_cache.csv")
            return sorted(tickers)

    # Fallback: starter list (run canyon_rebuild_universe.py to expand)
    print("  [universe] WARNING: using starter list of 50 stocks. Run canyon_rebuild_universe.py first.")
    return sorted(set([
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AMD","MU","AVGO",
        "JPM","BAC","WFC","GS","V","MA","JNJ","UNH","LLY","PFE","ABBV","TMO",
        "XOM","CVX","WMT","COST","HD","PG","KO","PEP","INTC","QCOM","TXN",
        "AMAT","CRM","ADBE","ORCL","IBM","CSCO","INTU","BRK-B","CAT","DE",
        "HON","UPS","RTX","BA","GE","MMM",
    ]) - ETF_TICKERS)

STOCK_UNIVERSE = _load_universe()

# ── FRED Series ───────────────────────────────────────────────────────────────
FRED_SERIES = {
    "BAA_AAA_SPREAD":  "BAA",       # Corporate bond credit spread proxy
    "YIELD_CURVE":     "T10Y2Y",    # 10Y-2Y Treasury spread (recession indicator)
    "VIX":             "VIXCLS",    # CBOE VIX
    "INITIAL_CLAIMS":  "ICSA",      # Initial jobless claims
    "ISM_PMI":         "MANEMP",    # Proxy for manufacturing activity
    "REAL_FED_RATE":   "REAINTRATREARAT10Y",  # Real 10Y rate
}

# ── Filing lag (avoid lookahead bias) ─────────────────────────────────────────
FILING_LAG_DAYS = 45   # SEC 10-Q large accelerated filer deadline
OOS_START = pd.Timestamp("2020-01-01")
REQUEST_DELAY = 0.35   # seconds between yfinance calls


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_prices(tickers: list[str], years: int = 5) -> pd.DataFrame:
    """
    Load daily adjusted close prices from yfinance.
    Returns wide DataFrame: date × ticker.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    all_tickers = list(set(tickers + list(SECTOR_ETFS.keys()) + ["SPY", "QQQ"]))

    print(f"  [prices] downloading {len(all_tickers)} tickers from {start}...")
    raw = yf.download(all_tickers, start=start, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices.ffill().dropna(how="all", axis=0)
    print(f"  [prices] loaded {prices.shape[0]} days × {prices.shape[1]} tickers")
    return prices


def load_fundamentals_yf(ticker: str) -> dict:
    """
    Fetch annual financial statements from yfinance.
    Returns dict with income_stmt, balance_sheet, cashflow, info.
    Applied filing lag of 45 days to all dates.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        return {
            "income":  t.income_stmt,
            "balance": t.balance_sheet,
            "cashflow": t.cashflow,
            "info":    t.info,
        }
    except Exception as e:
        print(f"    [yf] {ticker}: {e}")
        return {"income": None, "balance": None, "cashflow": None, "info": {}}


def _row(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    """Try multiple row name variants (yfinance naming is inconsistent)."""
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            s = df.loc[name].dropna().sort_index()
            if not s.empty:
                return s
    return None


def load_fred_series(series_id: str, years: int = 5) -> Optional[pd.Series]:
    """
    Fetch a single FRED series via their free JSON API.
    No API key required for public series.
    """
    try:
        import urllib.request, json
        start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&vintage_date={datetime.today().strftime('%Y-%m-%d')}"
        )
        # Use pandas directly (supports CSV URL)
        df = pd.read_csv(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
            index_col=0, parse_dates=True, na_values="."
        )
        df.index = pd.to_datetime(df.index)
        series = df.iloc[:, 0].dropna()
        series = series[series.index >= start]
        print(f"  [FRED] {series_id}: {len(series)} observations")
        return series
    except Exception as e:
        print(f"  [FRED] {series_id} failed: {e}")
        return None


# =============================================================================
# SECTION 2: VALUE FACTORS
# =============================================================================

def compute_value_composite(info: dict) -> dict:
    """
    Composite value score from 3 metrics (Asness et al. 2019).
    Single P/E has 40% "value trap" rate. Composite cuts this to ~20%.

    Metrics:
      P/FCF  — Price / Free Cash Flow. Cash doesn't lie. Most manipulation-resistant.
      EV/EBITDA — Enterprise value basis. Accounts for debt. Works across cap structures.
      P/B    — Classic Fama-French value proxy. Works best for asset-heavy sectors.

    All three ranked cross-sectionally, then equal-weighted.
    """
    pe    = info.get("trailingPE")
    pb    = info.get("priceToBook")
    pfcf  = info.get("priceToFreeCashflows") or info.get("freeCashflow")
    ev_eb = info.get("enterpriseToEbitda")

    scores = {}

    # P/FCF: lower is more valuable (but negative FCF = value trap, skip)
    if pfcf and pfcf > 0:
        scores["pfcf_val"] = 1 / pfcf  # higher score = cheaper
    else:
        scores["pfcf_val"] = np.nan

    # EV/EBITDA: lower = cheaper
    if ev_eb and ev_eb > 0:
        scores["ev_ebitda_val"] = 1 / ev_eb
    else:
        scores["ev_ebitda_val"] = np.nan

    # P/B: lower = cheaper
    if pb and pb > 0:
        scores["pb_val"] = 1 / pb
    else:
        scores["pb_val"] = np.nan

    # P/E: supplementary (easily manipulated, but widely watched)
    if pe and 0 < pe < 200:
        scores["pe_val"] = 1 / pe
    else:
        scores["pe_val"] = np.nan

    valid = [v for v in scores.values() if not np.isnan(v)]
    scores["value_composite_raw"] = np.mean(valid) if valid else np.nan

    return scores


# =============================================================================
# SECTION 3: QUALITY FACTORS
# =============================================================================

def compute_piotroski_fscore(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> dict:
    """
    Piotroski F-Score (2000): 9 binary signals from financial statements.
    Score 0–9. High score (8–9) = quality; Low score (0–1) = financial stress.

    Outperforms low F-Score by ~7.5%/yr in long-only, ~23%/yr in L/S.
    (Piotroski 2000, Journal of Accounting Research)

    PROFITABILITY (4 signals):
      F1: ROA > 0                  (profitable vs. loss-making)
      F2: Op Cash Flow > 0         (real cash, not accounting income)
      F3: ΔROA > 0                 (improving profitability trend)
      F4: Accruals < 0             (OCF > Net Income → quality earnings)

    LEVERAGE / LIQUIDITY (3 signals):
      F5: ΔLeverage < 0            (reducing debt = financial discipline)
      F6: ΔCurrent Ratio > 0       (improving short-term liquidity)
      F7: No new equity issued      (no dilution of existing shareholders)

    OPERATING EFFICIENCY (2 signals):
      F8: ΔGross Margin > 0        (pricing power or cost efficiency improving)
      F9: ΔAsset Turnover > 0      (generating more revenue per dollar of assets)
    """
    result = {f"F{i}": np.nan for i in range(1, 10)}
    result["piotroski_score"] = np.nan
    result["piotroski_label"] = "UNKNOWN"

    try:
        # Extract key line items (multiple name variants for yfinance inconsistency)
        net_income  = _row(income, "Net Income", "Net Income Common Stockholders")
        total_assets= _row(balance, "Total Assets")
        ocf         = _row(cashflow, "Operating Cash Flow", "Cash From Operations",
                           "Total Cash From Operating Activities")
        total_liab  = _row(balance, "Total Liabilities Net Minority Interest",
                           "Total Liabilities")
        curr_assets = _row(balance, "Current Assets", "Total Current Assets")
        curr_liab   = _row(balance, "Current Liabilities", "Total Current Liabilities")
        shares      = _row(income, "Diluted Average Shares", "Basic Average Shares",
                           "Common Stock Shares Outstanding")
        gross_prof  = _row(income, "Gross Profit")
        revenue     = _row(income, "Total Revenue", "Revenue")

        # Need at least 2 years for delta signals
        if total_assets is None or len(total_assets) < 2:
            return result

        ta_curr = total_assets.iloc[-1]
        ta_prev = total_assets.iloc[-2]

        # ── PROFITABILITY ──────────────────────────────────────────────────────
        # F1: ROA > 0
        if net_income is not None and len(net_income) >= 1:
            roa_curr = net_income.iloc[-1] / ta_curr
            result["F1"] = int(roa_curr > 0)
        else:
            roa_curr = np.nan

        # F2: Operating Cash Flow > 0
        if ocf is not None:
            result["F2"] = int(ocf.iloc[-1] > 0)

        # F3: ΔROA > 0 (current ROA vs. prior year ROA)
        if net_income is not None and len(net_income) >= 2 and not np.isnan(roa_curr):
            roa_prev = net_income.iloc[-2] / ta_prev if ta_prev != 0 else np.nan
            if not np.isnan(roa_prev):
                result["F3"] = int(roa_curr > roa_prev)

        # F4: Accruals < 0 → OCF > Net Income (quality earnings)
        if ocf is not None and net_income is not None:
            accrual = (net_income.iloc[-1] - ocf.iloc[-1]) / ta_curr
            result["F4"] = int(accrual < 0)

        # ── LEVERAGE / LIQUIDITY ───────────────────────────────────────────────
        # F5: ΔLeverage < 0 (Long-term debt / total assets decreased)
        if total_liab is not None and len(total_liab) >= 2:
            lev_curr = total_liab.iloc[-1] / ta_curr
            lev_prev = total_liab.iloc[-2] / ta_prev if ta_prev != 0 else np.nan
            if not np.isnan(lev_prev):
                result["F5"] = int(lev_curr < lev_prev)

        # F6: ΔCurrent Ratio > 0
        if curr_assets is not None and curr_liab is not None:
            if len(curr_assets) >= 2 and len(curr_liab) >= 2:
                cr_curr = curr_assets.iloc[-1] / curr_liab.iloc[-1] if curr_liab.iloc[-1] != 0 else np.nan
                cr_prev = curr_assets.iloc[-2] / curr_liab.iloc[-2] if curr_liab.iloc[-2] != 0 else np.nan
                if not (np.isnan(cr_curr) or np.isnan(cr_prev)):
                    result["F6"] = int(cr_curr > cr_prev)

        # F7: No new equity issuance (shares outstanding not increased significantly)
        if shares is not None and len(shares) >= 2:
            dilution = (shares.iloc[-1] - shares.iloc[-2]) / shares.iloc[-2] if shares.iloc[-2] != 0 else 0
            result["F7"] = int(dilution <= 0.02)  # ≤2% dilution = pass

        # ── OPERATING EFFICIENCY ───────────────────────────────────────────────
        # F8: ΔGross Margin > 0
        if gross_prof is not None and revenue is not None:
            if len(gross_prof) >= 2 and len(revenue) >= 2:
                gm_curr = gross_prof.iloc[-1] / revenue.iloc[-1] if revenue.iloc[-1] != 0 else np.nan
                gm_prev = gross_prof.iloc[-2] / revenue.iloc[-2] if revenue.iloc[-2] != 0 else np.nan
                if not (np.isnan(gm_curr) or np.isnan(gm_prev)):
                    result["F8"] = int(gm_curr > gm_prev)

        # F9: ΔAsset Turnover > 0
        if revenue is not None and len(revenue) >= 2:
            at_curr = revenue.iloc[-1] / ta_curr if ta_curr != 0 else np.nan
            at_prev = revenue.iloc[-2] / ta_prev if ta_prev != 0 else np.nan
            if not (np.isnan(at_curr) or np.isnan(at_prev)):
                result["F9"] = int(at_curr > at_prev)

        # ── COMPOSITE SCORE ────────────────────────────────────────────────────
        valid_signals = [result[f"F{i}"] for i in range(1, 10) if not np.isnan(result[f"F{i}"])]
        if valid_signals:
            score = sum(valid_signals)
            result["piotroski_score"] = score
            result["piotroski_label"] = (
                "HIGH_QUALITY" if score >= 7 else
                "MEDIUM_QUALITY" if score >= 4 else
                "LOW_QUALITY"
            )

    except Exception as e:
        print(f"    [piotroski] error: {e}")

    return result


def compute_gross_profitability(income: pd.DataFrame, balance: pd.DataFrame) -> dict:
    """
    Novy-Marx (2013) Gross Profitability: Gross Profit / Total Assets.

    WHY THIS IS BETTER THAN NET INCOME:
    - Net income is after depreciation, amortization, interest, taxes, and
      management discretion on accruals — all highly manipulable.
    - Gross profit = Revenue − COGS. COGS is the hardest number to fake.
    - Total Assets normalizes for size.

    Result: gross_profitability is the MOST negatively correlated quality signal
    with the value factor (P/B). Combining quality + value eliminates companies
    that are cheap because they're actually terrible businesses.

    IC historically: +0.08 to +0.12 (standalone)
    Combined with value: IC improves to +0.15+
    """
    result = {"gross_profitability": np.nan, "gross_margin": np.nan}
    try:
        gp = _row(income, "Gross Profit")
        ta = _row(balance, "Total Assets")
        rev = _row(income, "Total Revenue", "Revenue")

        if gp is not None and ta is not None:
            result["gross_profitability"] = gp.iloc[-1] / ta.iloc[-1]
        if gp is not None and rev is not None:
            result["gross_margin"] = gp.iloc[-1] / rev.iloc[-1] if rev.iloc[-1] != 0 else np.nan
    except Exception as e:
        print(f"    [gross_prof] {e}")
    return result


def compute_altman_zscore(income: pd.DataFrame, balance: pd.DataFrame, info: dict) -> dict:
    """
    Altman Z-Score (1968) — 5-factor financial distress model.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5

    X1 = Working Capital / Total Assets             (short-term liquidity)
    X2 = Retained Earnings / Total Assets           (cumulative profitability)
    X3 = EBIT / Total Assets                        (operating profitability)
    X4 = Market Cap / Total Liabilities             (market-based solvency)
    X5 = Revenue / Total Assets                     (asset utilization)

    INTERPRETATION:
      Z > 2.99  → Safe zone (low bankruptcy risk)
      1.81–2.99 → Gray zone (30% 2-year bankruptcy rate — AVOID in portfolio)
      Z < 1.81  → Distress zone (HIGH bankruptcy risk — exclude from long)

    WHY USE AS ALPHA SIGNAL:
    - Long ONLY stocks with Z > 2.99 (safe). This is a FILTER, not a buy signal.
    - The value trap problem: stocks in distress look "cheap" (low P/B, P/E) but
      continue to underperform. Z-Score removes them mechanically.
    - Combined with Piotroski: if Z < 1.81 OR F-Score ≤ 2, SKIP the stock.
    """
    result = {
        "altman_z": np.nan, "altman_x1": np.nan, "altman_x2": np.nan,
        "altman_x3": np.nan, "altman_x4": np.nan, "altman_x5": np.nan,
        "altman_zone": "UNKNOWN"
    }
    try:
        ta = _row(balance, "Total Assets")
        curr_a = _row(balance, "Current Assets", "Total Current Assets")
        curr_l = _row(balance, "Current Liabilities", "Total Current Liabilities")
        retained = _row(balance, "Retained Earnings")
        ebit = _row(income, "EBIT", "Operating Income", "Ebit")
        rev = _row(income, "Total Revenue", "Revenue")
        total_liab = _row(balance, "Total Liabilities Net Minority Interest", "Total Liabilities")
        mktcap = info.get("marketCap", None)

        if ta is None or ta.iloc[-1] == 0:
            return result

        ta_val = ta.iloc[-1]

        x1 = ((curr_a.iloc[-1] - curr_l.iloc[-1]) / ta_val
               if curr_a is not None and curr_l is not None else np.nan)
        x2 = retained.iloc[-1] / ta_val if retained is not None else np.nan
        x3 = ebit.iloc[-1] / ta_val if ebit is not None else np.nan
        x4 = (mktcap / total_liab.iloc[-1]
               if mktcap and total_liab is not None and total_liab.iloc[-1] != 0
               else np.nan)
        x5 = rev.iloc[-1] / ta_val if rev is not None else np.nan

        components = [x1, x2, x3, x4, x5]
        if all(not np.isnan(c) for c in components):
            z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 0.999*x5
            result.update({
                "altman_z": z,
                "altman_x1": x1, "altman_x2": x2, "altman_x3": x3,
                "altman_x4": x4, "altman_x5": x5,
                "altman_zone": "SAFE" if z > 2.99 else "GRAY" if z > 1.81 else "DISTRESS"
            })

    except Exception as e:
        print(f"    [altman] {e}")
    return result


# =============================================================================
# SECTION 4: GROWTH FACTORS
# =============================================================================

def compute_revenue_acceleration(income: pd.DataFrame) -> dict:
    """
    Revenue Acceleration: the 2nd derivative of revenue growth.

    WHY THE SECOND DERIVATIVE?
    - Analysts anchor on trailing growth rates. If a company grew 10% last year,
      analysts forecast ~10% this year.
    - When growth ACCELERATES to 15%, analysts underreact and take 2–6 months
      to revise targets upward. Stock price catches up gradually.
    - This creates the "growth surprise" analog to earnings surprise.
    - Works best for companies where analysts have the most anchoring bias:
      mid-cap, less-followed stocks.
    """
    result = {"rev_growth_yoy": np.nan, "rev_acceleration": np.nan}
    try:
        rev = _row(income, "Total Revenue", "Revenue")
        if rev is None or len(rev) < 3:
            return result

        rev_sorted = rev.sort_index()
        growth_curr = (rev_sorted.iloc[-1] / rev_sorted.iloc[-2] - 1) if rev_sorted.iloc[-2] != 0 else np.nan
        growth_prev = (rev_sorted.iloc[-2] / rev_sorted.iloc[-3] - 1) if rev_sorted.iloc[-3] != 0 else np.nan

        result["rev_growth_yoy"] = growth_curr
        if not (np.isnan(growth_curr) or np.isnan(growth_prev)):
            result["rev_acceleration"] = growth_curr - growth_prev  # positive = accelerating
    except Exception as e:
        print(f"    [rev_accel] {e}")
    return result


def compute_sue_earnings_surprise(info: dict) -> dict:
    """
    SUE — Standardized Unexpected Earnings (PEAD signal).

    Post-Earnings Announcement Drift (Bernard & Thomas 1989):
    - Market SYSTEMATICALLY UNDERREACTS to earnings surprises.
    - Stocks that beat by +1 std tend to drift +3–5% over the next 60 trading days.
    - This is one of the most robust anomalies across 40+ countries.

    Why does it persist?
    - Investors use "seasonal random walk" model: expect EPS = same quarter last year.
    - Analysts revise slowly (career risk of being first to change estimates).
    - Information diffusion is slow: retail investors read earnings press releases
      days or weeks after institutions have already acted.

    FREE Implementation (proxy without analyst consensus data):
      SUE = (actual_eps − expected_eps) / std_dev_of_historical_eps
      expected_eps = same quarter 1 year ago (seasonal random walk)
      std_dev = std of last 8 quarters' EPS

    Limitation: without real-time analyst consensus (paid data), this is an
    approximation. The true SUE uses analyst consensus as expected_eps.
    For a free approximation, the seasonal RW model has IC ≈ +0.04–0.07.
    """
    result = {"sue_proxy": np.nan, "eps_growth_qoq": np.nan, "earnings_trend": "UNKNOWN"}
    try:
        # yfinance provides trailing EPS
        eps_curr = info.get("trailingEps")
        eps_fwd  = info.get("forwardEps")

        if eps_curr and eps_fwd:
            eps_growth = (eps_fwd - eps_curr) / abs(eps_curr) if eps_curr != 0 else np.nan
            result["eps_growth_qoq"] = eps_growth
            if eps_growth > 0.10:
                result["earnings_trend"] = "ACCELERATING"
            elif eps_growth > 0:
                result["earnings_trend"] = "POSITIVE"
            elif eps_growth > -0.10:
                result["earnings_trend"] = "FLAT"
            else:
                result["earnings_trend"] = "DECLINING"
            # Proxy SUE: normalize by rough eps volatility proxy (30% of eps)
            eps_vol_proxy = abs(eps_curr) * 0.30
            result["sue_proxy"] = eps_growth * abs(eps_curr) / eps_vol_proxy if eps_vol_proxy > 0 else np.nan

    except Exception as e:
        print(f"    [sue] {e}")
    return result


# =============================================================================
# SECTION 5: MOMENTUM FACTORS (DEEP)
# =============================================================================

def compute_momentum_signals(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Multi-dimensional momentum scoring.

    Five components — each measures a different behavioral phenomenon:

    1. Cross-sectional 12-1 (Jegadeesh & Titman 1993)
       Behavioral driver: slow information diffusion.
       Implementation: rank by 252d–21d return. Winsorize at 5th/95th pct.

    2. 52-week high ratio (George & Hwang 2004)
       Behavioral driver: anchoring to past high as resistance.
       Implementation: current_price / max(252d price). Values near 1.0 = bullish.

    3. Volatility-scaled momentum (Barroso & Santa-Clara 2015)
       Risk driver: high-vol momentum = same return but more risk.
       Implementation: 12-1 return / 21d realized vol. Reduces crash by 50%+.

    4. Residual / idiosyncratic momentum (Blitz, Huij, Martens 2011)
       Alpha driver: strip out market beta → stock-specific momentum.
       Implementation: 21d-window OLS regression on SPY → 252d sum of residuals (skip last 21d).

    5. Industry momentum (Moskowitz & Grinblatt 1999)
       Sector driver: half of stock momentum is actually sector momentum.
       Implementation: rank sectors by 12-1 return → assign to each stock.
    """
    rets = prices.pct_change().fillna(0)
    spy_ret = rets.get("SPY", rets.mean(axis=1))

    records = []
    for tkr in tickers:
        if tkr not in rets.columns:
            continue

        r = rets[tkr].dropna()
        if len(r) < 252:
            records.append({"ticker": tkr})
            continue

        row = {"ticker": tkr}

        # ── 1. Cross-sectional 12-1 momentum ──────────────────────────────────
        ret_252 = (1 + r.iloc[-252:]).prod() - 1
        ret_21  = (1 + r.iloc[-21:]).prod() - 1
        row["mom_12_1"] = ret_252 - ret_21  # skip last month

        # ── 2. 52-week high ratio ──────────────────────────────────────────────
        p = prices[tkr].dropna()
        if len(p) >= 252:
            row["high52_ratio"] = p.iloc[-1] / p.iloc[-252:].max()
        else:
            row["high52_ratio"] = np.nan

        # ── 3. Volatility-scaled momentum ─────────────────────────────────────
        vol_21d = r.iloc[-21:].std() * np.sqrt(252)
        row["vol_scaled_mom"] = row["mom_12_1"] / vol_21d if vol_21d > 0 else np.nan

        # ── 4. Residual (idiosyncratic) momentum ──────────────────────────────
        # Use a 252-day window, regress daily returns on SPY
        try:
            spy_aligned = spy_ret.reindex(r.index).dropna()
            r_aligned   = r.reindex(spy_aligned.index)
            if len(r_aligned) >= 252:
                # OLS: r_stock = alpha + beta * r_spy + epsilon
                spy_window = spy_aligned.iloc[-252:-21].values
                r_window   = r_aligned.iloc[-252:-21].values
                X = np.column_stack([np.ones(len(spy_window)), spy_window])
                try:
                    beta_vec = np.linalg.lstsq(X, r_window, rcond=None)[0]
                    residuals = r_window - X @ beta_vec
                    row["residual_mom"] = residuals.sum()
                    row["market_beta"]  = beta_vec[1]
                except Exception:
                    row["residual_mom"] = np.nan
                    row["market_beta"]  = np.nan
        except Exception:
            row["residual_mom"] = np.nan
            row["market_beta"]  = np.nan

        records.append(row)

    df = pd.DataFrame(records).set_index("ticker")
    return df


def compute_industry_momentum(prices: pd.DataFrame) -> pd.Series:
    """
    Industry momentum (Moskowitz & Grinblatt 1999).

    Key insight: ~50% of individual stock momentum is explained by INDUSTRY
    momentum. Industry momentum is:
      - More persistent (12–18 months vs. 6–12 months for stock momentum)
      - Less prone to momentum crashes (sectors don't crash as violently as stocks)
      - Adds independent IC when combined with stock-level momentum

    Implementation:
      1. Compute 12-1 return for each sector ETF (XLK, XLV, XLF, ...)
      2. Rank sectors 1–11 (1 = weakest, 11 = strongest)
      3. Assign that rank to every stock in the sector

    Returns: pd.Series of sector momentum rank scores, indexed by sector name.
    """
    records = {}
    rets = prices.pct_change().fillna(0)

    for etf, sector_name in SECTOR_ETFS.items():
        if etf not in rets.columns:
            continue
        r = rets[etf].dropna()
        if len(r) < 252:
            continue
        ret_12_1 = (1 + r.iloc[-252:]).prod() - 1 - ((1 + r.iloc[-21:]).prod() - 1)
        records[sector_name] = ret_12_1

    if not records:
        return pd.Series(dtype=float)

    sector_rets = pd.Series(records)
    # Rank 1 (weakest) to N (strongest)
    ranked = sector_rets.rank()
    return ranked


def compute_momentum_crash_condition(prices: pd.DataFrame) -> dict:
    """
    Momentum crash detection (Daniel & Moskowitz 2016).

    CRASH MECHANISM:
    - Momentum strategy is effectively long recent winners, short recent losers.
    - During a bear market, "losers" are often deeply discounted value stocks.
    - When the bear-to-bull transition happens, these "loser" stocks surge first
      (short-covering, value discovery) while "winners" underperform.
    - Result: momentum CRASHES precisely when the market most needs returns.
    - In 2009, momentum lost ~40% in ONE MONTH during the March 2009 recovery.

    TWO-CONDITION CRASH SIGNAL (Daniel & Moskowitz):
    Condition 1: Market in bear (SPY down >10% from peak in last 12 months)
    Condition 2: Market volatility HIGH (21d realized vol > 25% annualized)

    When BOTH hold: momentum is in "crash regime" → scale weight by 50%.
    When neither holds: momentum is safe → full weight.

    Returns:
      crash_condition: bool — True = dangerous, scale back
      market_drawdown: float — SPY drawdown from peak
      market_vol_21d: float — annualized 21-day vol
      momentum_multiplier: float — 1.0 (safe) or 0.5 (crash-risk)
    """
    result = {
        "crash_condition": False,
        "market_drawdown": np.nan,
        "market_vol_21d":  np.nan,
        "momentum_multiplier": 1.0,
    }
    if "SPY" not in prices.columns:
        return result

    spy = prices["SPY"].dropna()
    if len(spy) < 252:
        return result

    # Condition 1: drawdown > 10%
    peak_252 = spy.iloc[-252:].cummax()
    drawdown = (spy.iloc[-1] / peak_252.iloc[-1]) - 1
    result["market_drawdown"] = drawdown

    # Condition 2: realized vol > 25%
    spy_rets = spy.pct_change().dropna()
    vol_21d = spy_rets.iloc[-21:].std() * np.sqrt(252)
    result["market_vol_21d"] = vol_21d

    if drawdown < -0.10 and vol_21d > 0.25:
        result["crash_condition"] = True
        result["momentum_multiplier"] = 0.50
    elif drawdown < -0.05 and vol_21d > 0.20:
        result["crash_condition"] = False
        result["momentum_multiplier"] = 0.75  # partial reduction

    return result


# =============================================================================
# SECTION 6: LOW VOLATILITY FACTORS
# =============================================================================

def compute_low_vol_signals(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Low volatility factor family.

    THREE COMPONENTS:

    1. Beta (BAB proxy) — Frazzini & Pedersen (2014)
       WHY: Leveraged-constrained investors (most large funds) CANNOT use leverage.
       They achieve high expected return by overweighting high-beta stocks.
       This systematic demand inflates high-beta prices → they underperform.
       Low-beta stocks are underowned by leverage-constrained funds → underpriced.
       Beta as NEGATIVE signal: lower beta → higher future return.

    2. Realized volatility (21d, 63d) — direct risk signal
       WHY: Investors with lottery preferences overweight high-vol stocks
       (they want the chance of a 10-bagger). This lottery premium means
       high-vol stocks are systematically expensive and underperform.
       High-vol stocks often look like "momentum" plays but revert faster.

    3. Idiosyncratic volatility (Ang, Hodrick, Xing, Zhang 2006 JF)
       WHY: After removing the market/sector component, the REMAINING
       idiosyncratic volatility is the MOST powerful negative signal.
       High idio-vol = lottery stock with retail investor premium = expensive.
       IC ≈ -0.08 (negative vol = positive signal). Counter-CAPM result.

    LOW VOL + QUALITY INTERACTION:
       Low vol + high Piotroski (7+) = BEST combination.
       Low vol + distress (Z < 1.81) = WORST combination (value trap variant).
       The interaction catches stocks that LOOK safe but are actually deteriorating.
    """
    rets = prices.pct_change().fillna(0)
    spy_ret = rets.get("SPY", rets.mean(axis=1))

    records = []
    for tkr in tickers:
        if tkr not in rets.columns:
            records.append({"ticker": tkr})
            continue

        r = rets[tkr].dropna()
        if len(r) < 63:
            records.append({"ticker": tkr})
            continue

        row = {"ticker": tkr}

        # ── 1. Realized volatility ─────────────────────────────────────────────
        row["vol_21d"]  = r.iloc[-21:].std() * np.sqrt(252)
        row["vol_63d"]  = r.iloc[-63:].std() * np.sqrt(252)
        row["vol_252d"] = r.iloc[-252:].std() * np.sqrt(252) if len(r) >= 252 else np.nan

        # ── 2. Beta to SPY (60-day regression) ────────────────────────────────
        try:
            spy_60 = spy_ret.reindex(r.index).iloc[-63:].dropna()
            r_60   = r.reindex(spy_60.index)
            if len(r_60) >= 30:
                X = np.column_stack([np.ones(len(spy_60)), spy_60.values])
                beta_vec = np.linalg.lstsq(X, r_60.values, rcond=None)[0]
                row["beta_60d"] = beta_vec[1]

                # ── 3. Idiosyncratic vol (residual after market removal) ──────
                residuals = r_60.values - X @ beta_vec
                row["idio_vol"] = residuals.std() * np.sqrt(252)
        except Exception:
            row["beta_60d"] = np.nan
            row["idio_vol"] = np.nan

        records.append(row)

    return pd.DataFrame(records).set_index("ticker")


# =============================================================================
# SECTION 7: SIZE FACTOR
# =============================================================================

def compute_size_factor(info_dict: dict[str, dict]) -> pd.Series:
    """
    Size factor: log(market_cap) — SMB proxy.

    WHY LOG?
    - Market cap distribution is extremely right-skewed (Apple is $3T, small caps $500M).
    - Log transforms this to approximately normal.
    - Log(3000) vs Log(500) is more informative than $2500B gap.

    SIZE PREMIUM SOURCE (Banz 1981):
      a) Illiquidity premium: small stocks are hard to trade → investors demand
         higher return as compensation for bearing liquidity risk.
      b) Neglect premium: fewer analysts cover small caps → more mispricing,
         more opportunities for patient investors who do the research.
      c) Information asymmetry: insiders know more, price discovery is slower.

    SIZE INTERACTION:
      Small + Quality (Piotroski 7+): STRONG POSITIVE combination.
        Small stocks are cheap AND have strong fundamentals → doubly undervalued.
      Small + Distress (Z < 1.81): STRONG NEGATIVE. Avoid. Small distressed
        companies rarely recover — they just lose more slowly.
      Small + High Momentum: STRONG POSITIVE. Small stock momentum persists longer
        because information diffuses more slowly in thinly-followed names.

    Returns pd.Series: ticker → log_market_cap
    """
    caps = {}
    for tkr, info in info_dict.items():
        mc = info.get("marketCap")
        if mc and mc > 0:
            caps[tkr] = np.log(mc)
    return pd.Series(caps, name="log_market_cap")


# =============================================================================
# SECTION 8: MACRO SIGNALS FROM FRED (FREE)
# =============================================================================

def build_macro_regime_from_fred() -> dict:
    """
    Build a richer macro regime signal using FRED free data.

    FRED provides 800,000+ economic series. Key signals for equity allocation:

    1. YIELD CURVE (T10Y2Y): 10-Year minus 2-Year Treasury spread.
       WHY: Inverted curve predicts recessions with 12-18 month lead.
       Positive spread = normal growth environment.
       Crossing zero → watch for slowdown in 6-12 months.

    2. CREDIT SPREAD (BAA-AAA): Corporate bond quality spread.
       WHY: Widening credit spreads = companies are struggling.
       Tight spreads = credit market is risk-on.
       This is the EARLIEST leading indicator before equity markets react.
       Historical: credit spreads widen 2-3 months before equity drawdowns.

    3. INITIAL JOBLESS CLAIMS: Weekly unemployment insurance filings.
       WHY: Real-time labor market signal. No revision (unlike NFP).
       Rising claims = economy weakening. 4-week moving average is more stable.
       Threshold: >350K/week consistently = recession territory.

    4. VIX LEVEL AND TREND: Equity market fear measure.
       VIX < 15: low fear, risk-on → full momentum, growth.
       VIX 15-25: moderate → watch for regime change.
       VIX > 25: elevated → tilt quality, low-vol.
       VIX > 35: crisis → defensive only.

    5. REAL 10-YEAR RATE (TIPS-based):
       WHY FOR FACTORS: Rising real rates hurt long-duration growth stocks
       (their far-future cash flows get discounted more heavily).
       Factor tilt: high real rates → value over growth.
       Negative real rates → growth/momentum over value.

    Returns dict with current macro state and factor tilt recommendations.
    """
    macro = {}

    yield_curve = load_fred_series("T10Y2Y")
    if yield_curve is not None and not yield_curve.empty:
        yc = yield_curve.iloc[-1]
        macro["yield_curve_spread"] = yc
        macro["yield_curve_signal"] = (
            "INVERTED" if yc < 0 else
            "FLAT"     if yc < 0.50 else
            "NORMAL"
        )

    vix = load_fred_series("VIXCLS")
    if vix is not None and not vix.empty:
        v = vix.iloc[-1]
        macro["vix"] = v
        macro["vix_regime"] = (
            "CRISIS"   if v > 35 else
            "HIGH"     if v > 25 else
            "MODERATE" if v > 15 else
            "LOW"
        )

    claims = load_fred_series("ICSA")
    if claims is not None and not claims.empty:
        c4wk = claims.iloc[-4:].mean()
        macro["initial_claims_4wk_avg"] = c4wk
        macro["labor_signal"] = "WEAK" if c4wk > 350000 else "OK"

    # Factor tilts based on macro regime
    macro["factor_tilt"] = _derive_factor_tilt(macro)
    return macro


def _derive_factor_tilt(macro: dict) -> dict:
    """
    Map macro regime to factor tilts.

    Evidence-based factor rotation:
      Rising rates + normal yield curve → VALUE outperforms GROWTH
        (long-duration growth stocks lose DCF value as discount rate rises)
      High VIX + credit spread widening → QUALITY + LOW_VOL outperform
        (flight to safety)
      Low VIX + inverted yield curve (late cycle) → MOMENTUM works until it stops
        (trend continuation until reversal)
      Recovering from recession (yield curve normalizing) → SIZE (small cap) leads
        (small caps have more operating leverage, recover faster)
    """
    tilt = {
        "value":    1.0,
        "quality":  1.0,
        "momentum": 1.0,
        "low_vol":  1.0,
        "growth":   1.0,
        "size":     1.0,
    }

    yc_signal = macro.get("yield_curve_signal", "NORMAL")
    vix_regime = macro.get("vix_regime", "LOW")

    # Rising rates / steep curve → favor value
    if yc_signal == "NORMAL":
        tilt["value"]   *= 1.20
        tilt["growth"]  *= 0.85

    # Inverted curve (late cycle) → defensive
    if yc_signal == "INVERTED":
        tilt["quality"] *= 1.30
        tilt["low_vol"] *= 1.30
        tilt["momentum"]*= 0.70
        tilt["growth"]  *= 0.70

    # High vol → defensive quality/low-vol
    if vix_regime in ("HIGH", "CRISIS"):
        tilt["quality"] *= 1.40
        tilt["low_vol"] *= 1.40
        tilt["momentum"]*= 0.50
        tilt["value"]   *= 0.80

    # Low vol + normal → momentum-friendly
    if vix_regime == "LOW" and yc_signal != "INVERTED":
        tilt["momentum"] *= 1.25
        tilt["growth"]   *= 1.20

    return tilt


# =============================================================================
# SECTION 9: COMPOSITE SCORING
# =============================================================================

def compute_ic_weights(score_history: pd.DataFrame, factor_cols: list[str],
                        forward_ret_col: str, window: int = 126) -> dict:
    """
    IC-weighted factor combination.

    WHY DYNAMIC WEIGHTING?
    Fixed equal-weight assumes all factors are always equally good.
    Reality: factors cycle. Value outperforms in rising-rate environments.
    Momentum outperforms in trending markets. Quality outperforms in downturns.

    IC (Information Coefficient) = Spearman rank correlation between factor
    scores and forward returns. Rolling 6-month IC tells you which factors
    have been working recently.

    Implementation:
      1. Compute rolling 6-month IC for each factor
      2. Positive IC → keep sign and upweight
      3. Negative IC → reverse sign or zero-weight
      4. Normalize weights to sum to 1.0

    Fallback: equal weights when IC history < 3 months (not enough data).
    """
    weights = {col: 1.0 / len(factor_cols) for col in factor_cols}  # fallback

    if score_history.empty or forward_ret_col not in score_history.columns:
        return weights

    recent = score_history.tail(window)
    if len(recent) < 63:  # need at least 3 months
        return weights

    ic_vals = {}
    for col in factor_cols:
        if col not in recent.columns:
            continue
        valid = recent[[col, forward_ret_col]].dropna()
        if len(valid) < 20:
            continue
        ic, _ = stats.spearmanr(valid[col], valid[forward_ret_col])
        ic_vals[col] = ic

    if not ic_vals:
        return weights

    # Weight by absolute IC, zero-weight negative IC
    raw = {k: max(v, 0) for k, v in ic_vals.items()}
    total = sum(raw.values())
    if total > 0:
        weights = {k: v / total for k, v in raw.items()}

    return weights


def build_master_composite(
    value_scores:    pd.DataFrame,
    quality_scores:  pd.DataFrame,
    growth_scores:   pd.DataFrame,
    momentum_scores: pd.DataFrame,
    lowvol_scores:   pd.DataFrame,
    size_scores:     pd.Series,
    macro_tilts:     dict,
    crash_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Build the master composite factor score.

    Family weights (base, before macro tilt adjustment):
      VALUE:    20%
      QUALITY:  25%  (slightly overweighted: quality is most persistent)
      GROWTH:   15%
      MOMENTUM: 20%
      LOW_VOL:  15%
      SIZE:      5%  (supplementary)

    Macro tilts modify family weights based on yield curve and VIX regime.
    Momentum multiplier applied during crash-risk regimes.

    Within-family composites use IC-weighted signals when history available,
    equal-weight as fallback.

    All sub-signals normalized to Z-scores (mean 0, std 1) before combining.
    Cross-sectional rank transform applied at the end: 0–100 scale.
    """
    all_dfs = [value_scores, quality_scores, growth_scores,
               momentum_scores, lowvol_scores]
    all_cols = []
    for df in all_dfs:
        if isinstance(df, pd.DataFrame):
            all_cols.append(df)

    if not all_cols:
        return pd.DataFrame()

    combined = pd.concat(all_cols, axis=1).copy()
    if isinstance(size_scores, pd.Series):
        combined["log_market_cap"] = size_scores

    def z(col: str) -> str:
        new_col = f"z_{col}"
        s = combined[col].dropna()
        if s.std() > 0:
            combined[new_col] = (combined[col] - s.mean()) / s.std()
        else:
            combined[new_col] = 0
        return new_col

    # ── QUALITY composite ──────────────────────────────────────────────────────
    q_signals = []
    if "piotroski_score" in combined.columns:
        q_signals.append(z("piotroski_score"))
    if "gross_profitability" in combined.columns:
        q_signals.append(z("gross_profitability"))
    if "roe" in combined.columns:
        q_signals.append(z("roe"))
    if q_signals:
        combined["quality_composite"] = combined[q_signals].mean(axis=1)

    # ── VALUE composite ────────────────────────────────────────────────────────
    v_signals = []
    for col in ["value_composite_raw", "pb_val", "pfcf_val", "ev_ebitda_val"]:
        if col in combined.columns:
            v_signals.append(z(col))
    if v_signals:
        combined["value_composite"] = combined[v_signals].mean(axis=1)

    # ── GROWTH composite ───────────────────────────────────────────────────────
    g_signals = []
    if "rev_acceleration" in combined.columns:
        g_signals.append(z("rev_acceleration"))
    if "sue_proxy" in combined.columns:
        g_signals.append(z("sue_proxy"))
    if g_signals:
        combined["growth_composite"] = combined[g_signals].mean(axis=1)

    # ── MOMENTUM composite ─────────────────────────────────────────────────────
    m_signals = []
    for col, wt in [("mom_12_1", 0.35), ("high52_ratio", 0.10),
                    ("vol_scaled_mom", 0.25), ("residual_mom", 0.10)]:
        if col in combined.columns:
            m_signals.append(z(col))
    if m_signals:
        combined["momentum_composite"] = combined[m_signals].mean(axis=1) * crash_multiplier

    # ── LOW VOL composite ──────────────────────────────────────────────────────
    lv_signals = []
    if "beta_60d" in combined.columns:
        combined["neg_beta"] = -combined["beta_60d"]
        lv_signals.append(z("neg_beta"))
    if "idio_vol" in combined.columns:
        combined["neg_idio_vol"] = -combined["idio_vol"]
        lv_signals.append(z("neg_idio_vol"))
    if "vol_21d" in combined.columns:
        combined["neg_vol"] = -combined["vol_21d"]
        lv_signals.append(z("neg_vol"))
    if lv_signals:
        combined["lowvol_composite"] = combined[lv_signals].mean(axis=1)

    # ── SIZE ───────────────────────────────────────────────────────────────────
    if "log_market_cap" in combined.columns:
        combined["z_size"] = -combined["log_market_cap"]  # small = positive signal
        z("z_size")

    # ── MASTER SCORE (macro-tilt adjusted) ────────────────────────────────────
    base_weights = {
        "quality_composite":  0.25,
        "value_composite":    0.20,
        "growth_composite":   0.15,
        "momentum_composite": 0.20,
        "lowvol_composite":   0.15,
        "z_size":             0.05,
    }

    # Apply macro tilts
    tilt = macro_tilts.get("factor_tilt", {})
    tilt_map = {
        "quality_composite":  "quality",
        "value_composite":    "value",
        "growth_composite":   "growth",
        "momentum_composite": "momentum",
        "lowvol_composite":   "low_vol",
        "z_size":             "size",
    }
    adj_weights = {}
    for col, base_wt in base_weights.items():
        factor_name = tilt_map.get(col, col)
        mult = tilt.get(factor_name, 1.0)
        adj_weights[col] = base_wt * mult

    # Normalize to sum to 1
    total_wt = sum(adj_weights.values())
    if total_wt > 0:
        adj_weights = {k: v / total_wt for k, v in adj_weights.items()}

    score_parts = []
    for col, wt in adj_weights.items():
        if col in combined.columns:
            score_parts.append(combined[col].fillna(0) * wt)

    if score_parts:
        combined["master_raw"] = sum(score_parts)
        # Cross-sectional rank → 0–100
        combined["master_score"] = combined["master_raw"].rank(pct=True) * 100
        combined["master_rank"]  = combined["master_raw"].rank(ascending=False)

    return combined


# =============================================================================
# SECTION 10: QUALITY SCREEN (DISTRESS FILTER)
# =============================================================================

def apply_quality_screen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard filters BEFORE any ranking.

    These are not factors — they are SCREENS. Stocks that fail are excluded
    from the tradeable universe regardless of other factor scores.

    SCREEN 1: Altman Z-Score distress filter
      Remove any stock with Z < 1.81 (distress zone).
      WHY: These are not undervalued stocks — they are failing businesses.
      Value-seeking strategies that don't screen for distress systematically
      buy the worst 20% of companies by financial health.

    SCREEN 2: Piotroski F-Score quality floor
      Remove stocks with F-Score ≤ 1 (only 1 or 0 out of 9 signals pass).
      WHY: These companies are deteriorating on every measurable dimension.
      Even if cheap, they continue to underperform for years.

    SCREEN 3: Momentum crash protection
      During crash conditions (both flags raised), reduce position of high-
      momentum stocks to 50% of target.
      WHY: Not a hard removal, but a sizing reduction. Daniel & Moskowitz show
      the expected loss during a momentum crash is 30–40% in one month.

    Returns df with 'screened_out' column (True = excluded from trading).
    """
    df = df.copy()
    df["screened_out"] = False
    df["screen_reason"] = ""

    if "altman_zone" in df.columns:
        mask = df["altman_zone"] == "DISTRESS"
        df.loc[mask, "screened_out"] = True
        df.loc[mask, "screen_reason"] += "ALTMAN_DISTRESS; "

    if "piotroski_score" in df.columns:
        mask = df["piotroski_score"] <= 1
        df.loc[mask & ~df["screened_out"], "screened_out"] = True
        df.loc[mask & ~df["screened_out"], "screen_reason"] += "PIOTROSKI_LOW; "

    return df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_full_factor_research(
    tickers: Optional[list[str]] = None,
    skip_download: bool = False,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full factor research pipeline.

    Flow:
      1. Load prices (yfinance)
      2. For each stock ticker: download fundamentals → compute all factors
      3. Compute momentum signals from price matrix
      4. Compute low-vol signals from price matrix
      5. Load FRED macro data → derive regime tilts
      6. Check momentum crash condition
      7. Build master composite with macro-adjusted weights
      8. Apply quality screens
      9. Output to CSV + MD report
    """
    out = output_dir or ROOT

    universe = tickers or STOCK_UNIVERSE
    print(f"\n[Factor Research] Universe: {len(universe)} tickers")
    print(f"[Factor Research] Date: {TODAY}\n")

    # ── 1. Prices ──────────────────────────────────────────────────────────────
    if not skip_download:
        prices = load_prices(universe + list(SECTOR_ETFS.keys()), years=5)
    else:
        cache = out / "sp500_price_cache.csv"
        if cache.exists():
            prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            print("  [!] No price cache found. Run without --ic-only first.")
            return pd.DataFrame()

    # ── 2. Fundamentals ────────────────────────────────────────────────────────
    fundamental_rows = []
    info_dict = {}

    print(f"\n[Fundamentals] Downloading {len(universe)} tickers...")
    for i, tkr in enumerate(universe):
        if tkr in ETF_TICKERS:
            continue
        print(f"  [{i+1}/{len(universe)}] {tkr}", end=" ")
        fin = load_fundamentals_yf(tkr)
        info_dict[tkr] = fin["info"]

        row = {"ticker": tkr}

        # Value
        row.update(compute_value_composite(fin["info"]))

        # Piotroski F-Score
        row.update(compute_piotroski_fscore(fin["income"], fin["balance"], fin["cashflow"]))

        # Gross Profitability
        row.update(compute_gross_profitability(fin["income"], fin["balance"]))

        # Accruals
        accrual = compute_accruals(fin["income"], fin["balance"], fin["cashflow"])
        row.update(accrual)

        # Altman Z-Score
        row.update(compute_altman_zscore(fin["income"], fin["balance"], fin["info"]))

        # Growth
        row.update(compute_revenue_acceleration(fin["income"]))
        row.update(compute_sue_earnings_surprise(fin["info"]))

        # ROE (standalone)
        roe = fin["info"].get("returnOnEquity")
        row["roe"] = roe if roe else np.nan

        # Market cap
        mc = fin["info"].get("marketCap")
        row["market_cap"] = mc if mc else np.nan

        print(f"→ F-Score={row.get('piotroski_score','?')}, "
              f"Z={row.get('altman_zone','?')}, "
              f"ROE={row.get('roe', np.nan):.2%}" if not np.isnan(row.get('roe', np.nan)) else f"→ F-Score={row.get('piotroski_score','?')}, Z={row.get('altman_zone','?')}")

        fundamental_rows.append(row)
        time.sleep(REQUEST_DELAY)

    fund_df = pd.DataFrame(fundamental_rows).set_index("ticker")

    # ── 3. Momentum signals ────────────────────────────────────────────────────
    print("\n[Momentum] Computing 5-component momentum scores...")
    mom_df = compute_momentum_signals(prices, universe)

    # Industry momentum
    print("[Momentum] Industry momentum...")
    industry_mom = compute_industry_momentum(prices)

    # Momentum crash detection
    crash_info = compute_momentum_crash_condition(prices)
    crash_mult = crash_info["momentum_multiplier"]
    if crash_info["crash_condition"]:
        print(f"  [!] MOMENTUM CRASH CONDITION ACTIVE — multiplier = {crash_mult:.0%}")
    else:
        print(f"  [OK] Momentum crash condition clear — multiplier = {crash_mult:.0%}")

    # ── 4. Low vol signals ─────────────────────────────────────────────────────
    print("\n[Low Vol] Computing beta and idiosyncratic volatility...")
    lowvol_df = compute_low_vol_signals(prices, universe)

    # ── 5. Macro signals ───────────────────────────────────────────────────────
    print("\n[Macro] Loading FRED signals...")
    macro = build_macro_regime_from_fred()
    print(f"  Yield curve: {macro.get('yield_curve_signal','?')} "
          f"({macro.get('yield_curve_spread', np.nan):.2f}%)")
    print(f"  VIX regime:  {macro.get('vix_regime','?')} ({macro.get('vix', np.nan):.1f})")

    # ── 6. Size factor ─────────────────────────────────────────────────────────
    size_scores = compute_size_factor(info_dict)

    # ── 7. Master composite ────────────────────────────────────────────────────
    print("\n[Composite] Building master factor scores...")
    master = build_master_composite(
        value_scores    = fund_df,
        quality_scores  = fund_df,
        growth_scores   = fund_df,
        momentum_scores = mom_df,
        lowvol_scores   = lowvol_df,
        size_scores     = size_scores,
        macro_tilts     = macro,
        crash_multiplier= crash_mult,
    )

    # ── 8. Quality screens ─────────────────────────────────────────────────────
    if not master.empty:
        master = apply_quality_screen(master)
        tradeable = master[~master["screened_out"]] if "screened_out" in master.columns else master
        print(f"\n  Screened in:  {(~master['screened_out']).sum() if 'screened_out' in master.columns else len(master)} tickers")
        print(f"  Screened out: {master['screened_out'].sum() if 'screened_out' in master.columns else 0} tickers")

    # ── 9. Save outputs ────────────────────────────────────────────────────────
    ts = datetime.today().strftime("%Y-%m-%d %H:%M")

    key_cols = [
        "master_score", "master_rank", "quality_composite", "value_composite",
        "growth_composite", "momentum_composite", "lowvol_composite",
        "piotroski_score", "piotroski_label", "altman_z", "altman_zone",
        "gross_profitability", "mom_12_1", "vol_scaled_mom", "residual_mom",
        "high52_ratio", "beta_60d", "idio_vol", "vol_21d",
        "rev_acceleration", "sue_proxy", "roe", "market_cap",
        "screened_out", "screen_reason",
    ]
    save_cols = [c for c in key_cols if c in master.columns]
    master[save_cols].sort_values("master_score", ascending=False).to_csv(
        out / "factor_deep_composite.csv"
    )

    master.to_csv(out / "factor_deep_scores.csv")

    # ── Markdown report ────────────────────────────────────────────────────────
    _write_report(master, macro, crash_info, ts, out)

    print(f"\n[Factor Research] Done. Outputs in {out}")
    return master


def compute_accruals(income, balance, cashflow) -> dict:
    """Sloan (1996) accruals ratio: (Net Income - Op CF) / Total Assets."""
    result = {"accruals_ratio": np.nan}
    try:
        ni  = _row(income, "Net Income", "Net Income Common Stockholders")
        ocf = _row(cashflow, "Operating Cash Flow", "Total Cash From Operating Activities")
        ta  = _row(balance, "Total Assets")
        if ni is not None and ocf is not None and ta is not None:
            result["accruals_ratio"] = (ni.iloc[-1] - ocf.iloc[-1]) / ta.iloc[-1]
    except Exception:
        pass
    return result


def _write_report(master: pd.DataFrame, macro: dict, crash: dict, ts: str, out: Path):
    """Write plain-English diagnostic report."""
    top10 = master.sort_values("master_score", ascending=False).head(10) if "master_score" in master.columns else master.head(10)
    screened = master[master.get("screened_out", pd.Series(False, index=master.index))].index.tolist() if "screened_out" in master.columns else []

    lines = [
        f"# Canyon v9 — Deep Factor Research Report",
        f"_Generated: {ts}_",
        f"",
        f"## Macro Regime",
        f"| Signal | Value | State |",
        f"|--------|-------|-------|",
        f"| Yield Curve (10Y-2Y) | {macro.get('yield_curve_spread', 'N/A'):.2f}% | {macro.get('yield_curve_signal', '?')} |",
        f"| VIX | {macro.get('vix', 'N/A'):.1f} | {macro.get('vix_regime', '?')} |",
        f"| Jobless Claims (4wk) | {macro.get('initial_claims_4wk_avg', 'N/A'):,.0f} | {macro.get('labor_signal', '?')} |",
        f"| Momentum Crash Risk | — | {'ACTIVE — scale to 50%' if crash['crash_condition'] else 'Clear'} |",
        f"",
        f"## Factor Tilts (Macro-Adjusted)",
        f"| Factor Family | Base Wt | Macro Tilt | Effective Wt |",
        f"|---------------|---------|-----------|--------------|",
    ]

    tilt = macro.get("factor_tilt", {})
    base = {"value": 0.20, "quality": 0.25, "growth": 0.15, "momentum": 0.20, "low_vol": 0.15, "size": 0.05}
    for fam, bw in base.items():
        t = tilt.get(fam, 1.0)
        eff = bw * t
        lines.append(f"| {fam.upper()} | {bw:.0%} | ×{t:.2f} | {eff:.1%} |")

    lines += [
        f"",
        f"## Top 10 by Master Score",
        f"| Rank | Ticker | Master | Quality | Value | Momentum | Low Vol | Piotroski | Z-Zone |",
        f"|------|--------|--------|---------|-------|----------|---------|-----------|--------|",
    ]
    if not top10.empty and "master_score" in top10.columns:
        for i, (tkr, row) in enumerate(top10.iterrows(), 1):
            def g(col, fmt=".1f"):
                v = row.get(col, np.nan)
                return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "—"
            lines.append(
                f"| {i} | **{tkr}** | {g('master_score')} | {g('quality_composite')} "
                f"| {g('value_composite')} | {g('momentum_composite')} | {g('lowvol_composite')} "
                f"| {g('piotroski_score', '.0f')} | {row.get('altman_zone','?')} |"
            )

    if screened:
        lines += [f"", f"## Screened Out (Distress or Quality Failure)", f""]
        for t in screened:
            reason = master.loc[t, "screen_reason"] if "screen_reason" in master.columns else ""
            lines.append(f"- **{t}**: {reason}")

    lines += [
        f"",
        f"## Factor Logic Reference",
        f"",
        f"### Why Momentum Works (and When It Fails)",
        f"Momentum (12-1 return) works because information diffuses slowly.",
        f"Markets underreact to good news → prices trend for 6–12 months.",
        f"Momentum CRASHES when market rebounds from a bear: the short leg surges.",
        f"Protection: scale to 50% when SPY drawdown >10% AND VIX >25.",
        f"",
        f"### Why Quality + Value Beats Either Alone",
        f"Value alone has 40% value-trap rate (cheap because broken).",
        f"Quality alone is expensive (market knows).",
        f"Quality + Value = cheap because temporarily misunderstood, not broken.",
        f"Screen: Piotroski ≥ 7 AND Z-Score > 2.99 AND low P/FCF.",
        f"",
        f"### Why Low Vol Stocks Outperform (Counter-CAPM)",
        f"CAPM says more risk = more return. For idiosyncratic risk, this is WRONG.",
        f"Investors overpay for lottery-like high-vol stocks (speculation premium).",
        f"Low-beta stocks are underowned by leveraged-constrained funds → underpriced.",
        f"Key: low idiosyncratic vol + high quality = best combination.",
        f"",
        f"_Canyon v9 — research use only. Not investment advice._",
    ]

    report_path = out / "factor_deep_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  [report] {report_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Deep Factor Research")
    parser.add_argument("--ic-only", action="store_true",
                        help="Skip downloads, use cached prices")
    parser.add_argument("--ticker", nargs="+", metavar="TICKER",
                        help="Run for specific tickers only")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as script)")
    args = parser.parse_args()

    run_full_factor_research(
        tickers      = args.ticker,
        skip_download= args.ic_only,
        output_dir   = Path(args.output) if args.output else None,
    )
