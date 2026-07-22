#!/usr/bin/env python3
"""
Canyon v9 — Step 78: Deep Fundamental Analysis
===============================================
Pulls real quarterly financial statements from yfinance and computes
6 quality factors that are specifically predictive in BEAR markets
(when momentum features flip sign and quality/value take over).

Why these 6 factors?
--------------------
  FCF Yield        : positive FCF / market cap → "can the company fund itself?"
  Accruals Ratio   : (net income - op CF) / assets → low accruals = real earnings
  Gross Margin δ   : QoQ improvement → pricing power and cost discipline
  Debt/EBITDA      : leverage ratio → who survives a bear market?
  Revenue Growth   : TTM YoY → is the business actually growing?
  ROE              : net income / equity → capital efficiency

Output
------
  fundamental_deep_scores.csv   — one row per ticker, raw + composite score
  fundamental_deep_report.md    — narrative with top/bottom quintile tables
  fundamental_quality_rank.csv  — ranked by composite quality score

Usage
-----
  python3 canyon_final_v9_step78_deep_fundamentals.py
  python3 canyon_final_v9_step78_deep_fundamentals.py --ticker AAPL
  python3 canyon_final_v9_step78_deep_fundamentals.py --fast    # top 40 tickers only
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# Input: load universe from step75 cache, or step66 ml scores
TICKER_SOURCES = [
    ROOT / "sp500_tickers.json",
    ROOT / "ml_signal_scores.csv",
    ROOT / "universe_expanded_features.csv",
    ROOT / "master_10_layer_decision_matrix_v2.csv",
]

OUT_SCORES  = ROOT / "fundamental_deep_scores.csv"
OUT_RANKED  = ROOT / "fundamental_quality_rank.csv"
OUT_REPORT  = ROOT / "fundamental_deep_report.md"

# Thresholds for quality labels
FCF_YIELD_GOOD  =  0.03   # > 3% = attractive
ACCRUALS_GOOD   =  0.05   # < 5% = high quality (low accruals)
DEBT_EBITDA_OK  =  3.0    # < 3× = manageable
REV_GROWTH_GOOD =  0.05   # > 5% = growing

TOP_N_TICKERS = 60        # max tickers to process (rate-limit friendly)
SLEEP_SEC     = 0.25      # between yfinance calls


# ─────────────────────────────────────────────────────────────
# 1.  UNIVERSE
# ─────────────────────────────────────────────────────────────

def load_tickers(fast: bool = False) -> list[str]:
    import json, time

    # Try sp500_tickers.json first
    for src in TICKER_SOURCES:
        if not src.exists():
            continue
        try:
            if src.suffix == ".json":
                data = json.loads(src.read_text())
                tickers = data.get("tickers", [])
                if tickers:
                    print(f"  Universe: {len(tickers)} tickers from {src.name}")
                    break
            elif src.suffix == ".csv":
                df = pd.read_csv(src)
                # find ticker column
                col = next((c for c in ["ticker", "Ticker", "TICKER", "Symbol"] if c in df.columns), None)
                if col:
                    tickers = df[col].dropna().unique().tolist()
                    print(f"  Universe: {len(tickers)} tickers from {src.name}")
                    break
        except Exception:
            continue
    else:
        # hard fallback
        tickers = [
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","UNH","JPM",
            "V","XOM","LLY","JNJ","MA","AVGO","PG","HD","COST","MRK","CVX","ABBV",
            "NFLX","BAC","KO","PEP","ADBE","WMT","TMO","ORCL","CSCO","DIS","ACN",
            "MCD","ABT","DHR","INTC","TXN","QCOM","AMGN","GS","MS","CAT","BA","GE",
        ]
        print(f"  Universe: {len(tickers)} fallback tickers")

    # Limit for fast mode or rate limiting (overridden by caller via top_n)
    limit = 40 if fast else TOP_N_TICKERS
    tickers = tickers[:limit]
    print(f"  Processing: {len(tickers)} tickers")
    return tickers


def load_tickers_n(n: int, fast: bool = False) -> list:
    """Like load_tickers() but overrides the hard cap with n."""
    tickers = load_tickers(fast=fast)
    # load_tickers already capped at TOP_N_TICKERS; re-load raw list for larger n
    if n > TOP_N_TICKERS:
        import json
        sp500_cache = Path(__file__).parent / "sp500_tickers.json"
        raw_tickers = []
        if sp500_cache.exists():
            with open(sp500_cache) as f:
                data = json.load(f)
                raw_tickers = data["tickers"] if isinstance(data, dict) else data
        if not raw_tickers:
            raw_tickers = tickers  # fallback to whatever load_tickers got
        return raw_tickers[:n]
    return tickers[:n]


# ─────────────────────────────────────────────────────────────
# 2.  DATA FETCH PER TICKER
# ─────────────────────────────────────────────────────────────

def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _latest_ttm(df: pd.DataFrame, col: str) -> float:
    """Sum the last 4 quarters of a quarterly series (TTM)."""
    if df is None or df.empty or col not in df.index:
        return np.nan
    row = df.loc[col].dropna()
    if len(row) < 1:
        return np.nan
    return float(row.iloc[:4].sum())


def _latest_q(df: pd.DataFrame, col: str, q: int = 0) -> float:
    """Get quarter q (0=latest) of a quarterly series."""
    if df is None or df.empty or col not in df.index:
        return np.nan
    row = df.loc[col].dropna()
    if len(row) <= q:
        return np.nan
    return float(row.iloc[q])


def fetch_fundamentals(ticker: str) -> dict:
    """Pull quarterly financials, balance sheet, cash flow for one ticker."""
    result = {"ticker": ticker}

    try:
        tk = yf.Ticker(ticker)

        # Market cap (for FCF yield)
        mcap = np.nan
        try:
            mcap = _safe_float(tk.fast_info.market_cap)
        except Exception:
            pass
        if np.isnan(mcap):
            try:
                mcap = _safe_float(tk.info.get("marketCap", np.nan))
            except Exception:
                pass
        result["market_cap"] = mcap

        # Quarterly income statement
        try:
            inc = tk.quarterly_income_stmt
            if inc is None or inc.empty:
                inc = tk.quarterly_financials
        except Exception:
            inc = None

        # Annual income statement (for YoY revenue growth — quarterly only has 4 qtrs)
        try:
            inc_annual = tk.income_stmt
        except Exception:
            inc_annual = None

        # Quarterly cash flow
        try:
            cf = tk.quarterly_cashflow
        except Exception:
            cf = None

        # Quarterly balance sheet
        try:
            bs = tk.quarterly_balance_sheet
        except Exception:
            bs = None

        # ── Net Income TTM ──────────────────────────────────
        ni_ttm = np.nan
        for col in ["Net Income", "NetIncome", "Net Income From Continuing Operations"]:
            ni_ttm = _latest_ttm(inc, col)
            if not np.isnan(ni_ttm):
                break
        result["net_income_ttm"] = ni_ttm

        # ── Revenue TTM ─────────────────────────────────────
        rev_ttm = np.nan
        for col in ["Total Revenue", "Revenue", "Revenues"]:
            rev_ttm = _latest_ttm(inc, col)
            if not np.isnan(rev_ttm):
                break
        result["revenue_ttm"] = rev_ttm

        # Revenue prior year — use ANNUAL income stmt (most reliable for YoY)
        # annual stmt has 4 fiscal years; iloc[0]=latest year, iloc[1]=prior year
        rev_ttm_1y = np.nan
        for col in ["Total Revenue", "Revenue", "Revenues"]:
            if inc_annual is not None and not inc_annual.empty and col in inc_annual.index:
                row_data = inc_annual.loc[col].dropna()
                if len(row_data) >= 2:
                    rev_ttm_1y = float(row_data.iloc[1])  # prior fiscal year revenue
                    break
        result["revenue_ttm_1y"] = rev_ttm_1y

        # Also override rev_ttm from annual if quarterly was NaN (more complete)
        if np.isnan(result.get("revenue_ttm", np.nan)):
            for col in ["Total Revenue", "Revenue", "Revenues"]:
                if inc_annual is not None and not inc_annual.empty and col in inc_annual.index:
                    row_data = inc_annual.loc[col].dropna()
                    if len(row_data) >= 1:
                        result["revenue_ttm"] = float(row_data.iloc[0])
                        break

        # ── Gross Profit ────────────────────────────────────
        gp_q0 = np.nan
        rev_q0 = np.nan
        gp_q4 = np.nan
        rev_q4 = np.nan
        for col in ["Gross Profit", "GrossProfit"]:
            gp_q0 = _latest_q(inc, col, 0)
            gp_q4 = _latest_q(inc, col, 4)
            if not np.isnan(gp_q0):
                break
        for col in ["Total Revenue", "Revenue"]:
            rev_q0 = _latest_q(inc, col, 0)
            rev_q4 = _latest_q(inc, col, 4)
            if not np.isnan(rev_q0):
                break
        result["gross_margin_q0"] = gp_q0 / rev_q0 if (rev_q0 and rev_q0 != 0) else np.nan
        result["gross_margin_q4"] = gp_q4 / rev_q4 if (rev_q4 and rev_q4 != 0) else np.nan

        # ── Operating Cash Flow TTM ─────────────────────────
        ocf_ttm = np.nan
        for col in ["Operating Cash Flow", "Cash From Operations",
                    "Total Cash From Operating Activities"]:
            ocf_ttm = _latest_ttm(cf, col)
            if not np.isnan(ocf_ttm):
                break
        result["ocf_ttm"] = ocf_ttm

        # ── CapEx TTM ───────────────────────────────────────
        capex_ttm = np.nan
        for col in ["Capital Expenditure", "Capital Expenditures", "CapEx"]:
            capex_ttm = _latest_ttm(cf, col)
            if not np.isnan(capex_ttm):
                break
        # capex is usually negative in CF statement; take abs
        if not np.isnan(capex_ttm):
            capex_ttm = abs(capex_ttm)
        result["capex_ttm"] = capex_ttm

        # ── Total Assets ────────────────────────────────────
        assets = np.nan
        for col in ["Total Assets", "TotalAssets"]:
            assets = _latest_q(bs, col, 0)
            if not np.isnan(assets):
                break
        result["total_assets"] = assets

        # ── Total Debt ──────────────────────────────────────
        debt = np.nan
        for col in ["Total Debt", "Long Term Debt", "LongTermDebt",
                    "Total Long Term Debt"]:
            debt = _latest_q(bs, col, 0)
            if not np.isnan(debt):
                break
        result["total_debt"] = debt

        # ── Total Equity ────────────────────────────────────
        equity = np.nan
        for col in ["Stockholders Equity", "Total Stockholder Equity",
                    "Common Stock Equity", "Total Equity"]:
            equity = _latest_q(bs, col, 0)
            if not np.isnan(equity):
                break
        result["total_equity"] = equity

        # ── EBITDA (estimate) ───────────────────────────────
        ebitda_ttm = np.nan
        for col in ["EBITDA", "Ebitda"]:
            ebitda_ttm = _latest_ttm(inc, col)
            if not np.isnan(ebitda_ttm):
                break
        # fallback: op income + D&A
        if np.isnan(ebitda_ttm):
            oi = np.nan
            da = np.nan
            for col in ["Operating Income", "EBIT"]:
                oi = _latest_ttm(inc, col)
                if not np.isnan(oi):
                    break
            for col in ["Depreciation", "Depreciation And Amortization",
                        "Reconciled Depreciation"]:
                da = _latest_ttm(cf, col)
                if not np.isnan(da):
                    break
            if not np.isnan(oi) and not np.isnan(da):
                ebitda_ttm = oi + abs(da)
        result["ebitda_ttm"] = ebitda_ttm

    except Exception as e:
        result["_error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────
# 3.  FACTOR COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_factors(row: dict) -> dict:
    """
    Compute the 6 quality factors from raw financials.
    All factors are signed so that HIGHER = BETTER quality.
    """
    f = {"ticker": row["ticker"]}

    mcap   = row.get("market_cap",     np.nan)
    ni     = row.get("net_income_ttm", np.nan)
    ocf    = row.get("ocf_ttm",        np.nan)
    capex  = row.get("capex_ttm",      np.nan)
    assets = row.get("total_assets",   np.nan)
    debt   = row.get("total_debt",     np.nan)
    equity = row.get("total_equity",   np.nan)
    ebitda = row.get("ebitda_ttm",     np.nan)
    rev    = row.get("revenue_ttm",    np.nan)
    rev_1y = row.get("revenue_ttm_1y", np.nan)
    gm0    = row.get("gross_margin_q0",np.nan)
    gm4    = row.get("gross_margin_q4",np.nan)

    # 1. FCF Yield = (OCF - CapEx) / Market Cap
    #    Higher = better (positive cash generation relative to valuation)
    fcf = (ocf - capex) if (not np.isnan(ocf) and not np.isnan(capex)) else np.nan
    f["fcf"]       = fcf
    f["fcf_yield"] = fcf / mcap if (not np.isnan(fcf) and mcap and mcap > 0) else np.nan

    # 2. Accruals Ratio = (Net Income - OCF) / Total Assets
    #    LOWER = better (earnings backed by cash), so we negate for consistent direction
    if not np.isnan(ni) and not np.isnan(ocf) and not np.isnan(assets) and assets > 0:
        f["accruals_ratio"]    = (ni - ocf) / assets
        f["accruals_quality"]  = -f["accruals_ratio"]  # higher = better quality
    else:
        f["accruals_ratio"]   = np.nan
        f["accruals_quality"] = np.nan

    # 3. Gross Margin Δ (QoQ, 1-year lookback)
    #    Positive = expanding margins = pricing power
    if not np.isnan(gm0) and not np.isnan(gm4):
        f["gross_margin"]       = gm0
        f["gross_margin_delta"] = gm0 - gm4   # positive = improvement
    else:
        f["gross_margin"]       = gm0
        f["gross_margin_delta"] = np.nan

    # 4. Debt/EBITDA  (lower = safer)
    #    We store raw ratio and negate for scoring direction
    if not np.isnan(debt) and not np.isnan(ebitda) and ebitda > 0:
        f["debt_ebitda"]       = debt / ebitda
        f["debt_safety"]       = -min(f["debt_ebitda"], 20.0)  # cap at 20x, negate
    else:
        f["debt_ebitda"]  = np.nan
        f["debt_safety"]  = np.nan

    # 5. Revenue Growth (YoY TTM)
    if not np.isnan(rev) and not np.isnan(rev_1y) and rev_1y != 0:
        f["revenue_growth"] = (rev - rev_1y) / abs(rev_1y)
    else:
        f["revenue_growth"] = np.nan

    # 6. ROE = Net Income TTM / Total Equity
    if not np.isnan(ni) and not np.isnan(equity) and equity > 0:
        f["roe"] = ni / equity
    else:
        f["roe"] = np.nan

    return f


def rank_normalize(series: pd.Series) -> pd.Series:
    """Convert to 0-100 cross-sectional rank, handling NaN."""
    valid = series.dropna()
    if len(valid) < 2:
        return series * np.nan
    ranked = series.rank(pct=True) * 100
    return ranked


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight composite of 6 quality factors, all rank-normalized 0-100.
    Higher score = better quality company.
    """
    score_cols = []
    for col, weight in [
        ("fcf_yield",        1.5),   # most important: can the company fund itself?
        ("accruals_quality", 1.0),   # earnings quality
        ("gross_margin_delta", 0.5), # momentum in fundamentals
        ("debt_safety",      1.5),   # balance sheet safety (bear market survival)
        ("revenue_growth",   1.0),   # business growth
        ("roe",              0.5),   # capital efficiency
    ]:
        if col not in df.columns:
            continue
        rank_col = f"rank_{col}"
        df[rank_col] = rank_normalize(df[col])
        score_cols.append((rank_col, weight))

    if not score_cols:
        df["quality_score"] = np.nan
        return df

    # Weighted average ignoring NaN components (so partial data still scores)
    def weighted_mean_row(row):
        vals  = [(row[rc], w) for rc, w in score_cols if not np.isnan(row[rc])]
        if not vals:
            return np.nan
        num   = sum(v * w for v, w in vals)
        denom = sum(w for _, w in vals)
        return num / denom

    df["quality_score"] = df.apply(weighted_mean_row, axis=1)

    # Quality label
    def label(s):
        if np.isnan(s): return "N/A"
        if s >= 75: return "HIGH"
        if s >= 50: return "ABOVE_AVG"
        if s >= 25: return "BELOW_AVG"
        return "LOW"
    df["quality_label"] = df["quality_score"].apply(label)

    return df


# ─────────────────────────────────────────────────────────────
# 4.  REPORT
# ─────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n  = len(df)
    n_hi  = (df["quality_label"] == "HIGH").sum()
    n_low = (df["quality_label"] == "LOW").sum()

    lines = [
        "# Canyon v9 — Step 78: Deep Fundamental Analysis",
        f"Generated: {ts}  |  {n} tickers analyzed",
        "",
        "## Why Fundamentals Matter in Bear Markets",
        "In BULL regimes, momentum (price → price) explains most cross-sectional returns.",
        "In BEAR regimes, the companies that survive best share these traits:",
        "- Positive free cash flow (can fund operations without new debt)",
        "- Low accruals (earnings backed by real cash, not accounting choices)",
        "- Strong balance sheet (low Debt/EBITDA → can survive credit tightening)",
        "- Growing revenue (not just margin-cutting)",
        "",
        "## Factor Definitions",
        "| Factor | Formula | Higher = |",
        "|--------|---------|---------|",
        "| FCF Yield | (OCF - CapEx) / Market Cap | Better value, self-funded |",
        "| Accruals Quality | -(NI - OCF) / Assets | Real earnings (low accruals) |",
        "| Gross Margin Δ | Current GM% - Year-ago GM% | Pricing power improving |",
        "| Debt Safety | -min(Debt/EBITDA, 20) | Less leverage risk |",
        "| Revenue Growth | TTM YoY% | Actually growing |",
        "| ROE | NI / Equity | Efficient capital use |",
        "",
        f"## Quality Distribution  (n={n})",
        f"- HIGH (≥75):       {n_hi}  ({n_hi/n*100:.0f}%)" if n > 0 else "- No data",
        f"- ABOVE_AVG (50-75): {(df['quality_label']=='ABOVE_AVG').sum()}",
        f"- BELOW_AVG (25-50): {(df['quality_label']=='BELOW_AVG').sum()}",
        f"- LOW (<25):         {n_low}  ({n_low/n*100:.0f}%)" if n > 0 else "",
        "",
        "## Top 20 Quality Stocks (Bear-Market Shelters)",
        "| # | Ticker | Quality Score | FCF Yield | Debt/EBITDA | Rev Growth | ROE |",
        "|---|--------|--------------|-----------|-------------|-----------|-----|",
    ]

    top20 = df.sort_values("quality_score", ascending=False).head(20)
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        def fv(col, fmt=".2f"):
            v = row.get(col, np.nan)
            if isinstance(v, float) and np.isnan(v): return "—"
            try: return f"{v:{fmt}}"
            except: return str(v)
        lines.append(
            f"| {i} | {row['ticker']} | {fv('quality_score','.1f')} "
            f"| {fv('fcf_yield','.2%')} | {fv('debt_ebitda','.1f')}x "
            f"| {fv('revenue_growth','.1%')} | {fv('roe','.1%')} |"
        )

    lines += [
        "",
        "## Bottom 10 (Avoid in Bear Markets)",
        "| # | Ticker | Quality Score | FCF Yield | Debt/EBITDA |",
        "|---|--------|--------------|-----------|-------------|",
    ]
    bot10 = df.sort_values("quality_score").head(10)
    for i, (_, row) in enumerate(bot10.iterrows(), 1):
        def fv(col, fmt=".2f"):
            v = row.get(col, np.nan)
            if isinstance(v, float) and np.isnan(v): return "—"
            try: return f"{v:{fmt}}"
            except: return str(v)
        lines.append(
            f"| {i} | {row['ticker']} | {fv('quality_score','.1f')} "
            f"| {fv('fcf_yield','.2%')} | {fv('debt_ebitda','.1f')}x |"
        )

    lines += [
        "",
        "## How to Use with Step 77 (Regime ML)",
        "The `quality_score` column is added to the feature panel in BEAR/SIDEWAYS regimes.",
        "When the regime detector (Step 76) outputs BEAR, high quality_score tickers",
        "should receive higher weight — they are the companies most likely to outperform",
        "in a credit-tightening, risk-off environment.",
        "",
        "## Data Source",
        "All data from yfinance quarterly_income_stmt / quarterly_cashflow / quarterly_balance_sheet.",
        "TTM = trailing twelve months (last 4 quarters summed).",
        "Data may lag 1-3 months vs actual reporting dates.",
    ]

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────

def run(tickers: list[str]) -> pd.DataFrame:
    import time

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Canyon v9 — Step 78: Deep Fundamental Analysis      ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print(f"[1/4] Fetching financials for {len(tickers)} tickers …")
    raw_rows = []
    for i, t in enumerate(tickers, 1):
        print(f"  [{i:3d}/{len(tickers)}] {t:<8s}", end=" ", flush=True)
        try:
            row = fetch_fundamentals(t)
            err = row.get("_error", "")
            if err:
                print(f"WARN: {err[:60]}")
            else:
                print(f"MCap={row.get('market_cap', 0)/1e9:.0f}B  "
                      f"Rev={row.get('revenue_ttm', 0)/1e9:.0f}B")
        except Exception as e:
            row = {"ticker": t, "_error": str(e)}
            print(f"FAIL: {e}")
        raw_rows.append(row)
        time.sleep(SLEEP_SEC)

    print(f"\n[2/4] Computing factors …")
    factor_rows = [compute_factors(r) for r in raw_rows]
    df = pd.DataFrame(factor_rows)

    print(f"[3/4] Ranking and scoring …")
    df = compute_composite(df)

    # Sort by composite
    df = df.sort_values("quality_score", ascending=False).reset_index(drop=True)

    # Save
    print(f"[4/4] Saving outputs …")
    df.to_csv(OUT_SCORES, index=False)
    print(f"  fundamental_deep_scores.csv  ({len(df)} rows)")

    # Ranked summary
    rank_cols = ["ticker", "quality_score", "quality_label",
                 "fcf_yield", "accruals_ratio", "gross_margin", "gross_margin_delta",
                 "debt_ebitda", "revenue_growth", "roe"]
    rank_df = df[[c for c in rank_cols if c in df.columns]]
    rank_df.to_csv(OUT_RANKED, index=False)
    print(f"  fundamental_quality_rank.csv  ({len(rank_df)} rows)")

    write_report(df)

    # Print key results
    print("\n>>> TOP 10 QUALITY SCORES (bear-market shelters):")
    print(f"  {'Ticker':<8}  {'Score':>6}  {'FCF Yield':>10}  {'Debt/EBITDA':>12}  {'RevGrowth':>10}")
    print(f"  {'-'*56}")
    for _, row in df.head(10).iterrows():
        fcf_s   = f"{row.get('fcf_yield',np.nan):.1%}"  if not np.isnan(row.get('fcf_yield', np.nan)) else "N/A"
        de_s    = f"{row.get('debt_ebitda',np.nan):.1f}x" if not np.isnan(row.get('debt_ebitda', np.nan)) else "N/A"
        rg_s    = f"{row.get('revenue_growth',np.nan):.1%}" if not np.isnan(row.get('revenue_growth', np.nan)) else "N/A"
        sc_s    = f"{row.get('quality_score',np.nan):.1f}" if not np.isnan(row.get('quality_score', np.nan)) else "N/A"
        print(f"  {row['ticker']:<8}  {sc_s:>6}  {fcf_s:>10}  {de_s:>12}  {rg_s:>10}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Step 78: Deep Fundamentals")
    parser.add_argument("--ticker", type=str, help="Single ticker to analyze")
    parser.add_argument("--fast",   action="store_true", help="Top 40 tickers only")
    parser.add_argument("--top",    type=int, default=TOP_N_TICKERS, help="Max tickers")
    args = parser.parse_args()

    if args.ticker:
        print(f"\n[Single ticker: {args.ticker}]")
        raw = fetch_fundamentals(args.ticker)
        factors = compute_factors(raw)
        print("\nRaw fundamentals:")
        for k, v in raw.items():
            if k not in ("ticker", "_error"):
                print(f"  {k:<25} {v}")
        print("\nComputed factors:")
        for k, v in factors.items():
            if k != "ticker":
                print(f"  {k:<25} {v}")
    else:
        tickers = load_tickers_n(args.top, fast=args.fast)
        run(tickers)
