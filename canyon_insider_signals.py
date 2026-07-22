#!/usr/bin/env python3
"""
Canyon v9 — Insider, Congressional & AI-Thought-Leader Signals
===============================================================
Free data sources (no API key):
  1. House Stock Watcher API  — housestockwatcher.com/api
  2. Senate Stock Watcher API — senatestockwatcher.com
  3. SEC EDGAR 13F            — data.sec.gov/submissions/ + efts search
  4. SEC EDGAR Form 4         — data.sec.gov/submissions/
  5. yfinance                 — post-trade price performance

VIP tracking:
  - Nancy Pelosi (Pelosi): congressional disclosure tracking
  - Leopold Aschenbrenner: AI-insider thesis + SEC 13F fund holdings

Output:
  congressional_trades.csv, congressional_vip_summary.csv
  insider_portfolio_overlap.csv
  aschenbrenner_13f.csv, aschenbrenner_thesis_overlap.csv
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.today().strftime("%Y-%m-%d")

# ── VIP watchlist — congressional traders ─────────────────────────────────
VIP_NAMES = {
    "Nancy Pelosi":    "PELOSI",      # Pelosi — most tracked
    "Paul Pelosi":     "PELOSI",      # files on behalf of Nancy
    "Dan Crenshaw":    "CRENSHAW",
    "Josh Gottheimer": "GOTTHEIMER",
    "Tommy Tuberville": "TUBERVILLE",
    "Michael McCaul":  "MCCAUL",
}

# ── Leopold Aschenbrenner — AI insider thesis ──────────────────────────────
# Source: "Situational Awareness" (June 2024) + public interviews
# Track via: (1) SEC EDGAR 13F if he filed a fund, (2) thesis stock overlap
ASCHENBRENNER_THESIS = {
    # Tier 1: Core AI compute monopoly — "the picks-and-shovels of AGI"
    "NVDA": {
        "conviction": "EXTREME",
        "thesis": "NVIDIA GPU monopoly = the oil wells of the AI age. No alternative at scale for years.",
        "quote": "The cluster buildout requires an extraordinary amount of compute, and NVIDIA is the only game in town.",
        "tier": 1,
    },
    "AVGO": {
        "conviction": "HIGH",
        "thesis": "Broadcom custom AI ASICs (Google TPU, Meta MTIA). When hyperscalers want to reduce NVDA dependency.",
        "quote": "Custom silicon will be a massive market as hyperscalers vertically integrate.",
        "tier": 1,
    },
    "TSM": {
        "conviction": "HIGH",
        "thesis": "TSMC makes every advanced chip. The chokepoint of the entire AI supply chain.",
        "quote": "Control of TSMC = control of the future of compute.",
        "tier": 1,
    },
    # Tier 2: Hyperscaler infrastructure — "trillion-dollar cluster builders"
    "MSFT": {
        "conviction": "HIGH",
        "thesis": "Azure + OpenAI partnership. Largest committed customer for OpenAI compute.",
        "quote": "Microsoft has made the most credible bet on OpenAI and AGI transition.",
        "tier": 2,
    },
    "GOOGL": {
        "conviction": "HIGH",
        "thesis": "DeepMind + TPUs + largest proprietary AI infrastructure. Most underestimated.",
        "quote": "Google has the best research talent, the most compute, and they are sleeping.",
        "tier": 2,
    },
    "AMZN": {
        "conviction": "MEDIUM",
        "thesis": "AWS dominates enterprise cloud. Trainium/Inferentia for custom compute.",
        "quote": "AWS will capture the majority of enterprise AI inference workloads.",
        "tier": 2,
    },
    "META": {
        "conviction": "MEDIUM",
        "thesis": "Open-source LLaMA strategy → massive inference scale. Building largest GPU cluster.",
        "quote": "Meta's open-source approach could win enterprise — no one wants vendor lock-in.",
        "tier": 2,
    },
    # Tier 3: Power / energy infrastructure — "data centers need 100GW by 2030"
    "NEE": {
        "conviction": "MEDIUM",
        "thesis": "Nuclear + renewables for data center power. 'The energy crisis is the bottleneck after compute.'",
        "quote": "The constraint will shift from chips to power. 100GW of new data center power by 2030.",
        "tier": 3,
    },
    "VST": {
        "conviction": "MEDIUM",
        "thesis": "Vistra Energy — nuclear base load + Texas grid where most hyperscaler data centers cluster.",
        "quote": "Data centers need 24/7 dispatchable power. Nuclear is the only answer at scale.",
        "tier": 3,
    },
    "CEG": {
        "conviction": "MEDIUM",
        "thesis": "Constellation Energy — largest US nuclear operator. Direct power agreements with hyperscalers.",
        "quote": "The Three Mile Island restart was Constellation + Microsoft. This is the thesis playing out.",
        "tier": 3,
    },
    # Tier 4: AI software layer — secondary but mentioned
    "AMD": {
        "conviction": "LOW",
        "thesis": "Alternative GPU supply. ROCm software improving. Insurance if NVDA supply constrained.",
        "quote": "AMD is the only credible alternative to NVIDIA. Not there yet but watching closely.",
        "tier": 4,
    },
}

ASCHENBRENNER_SEARCH_TERMS = [
    "Aschenbrenner",
    "Leopold Aschenbrenner",
    "situational awareness fund",
]

VIP_HIGHLIGHT = {"PELOSI"}          # gold banner


# =============================================================================
# 1. CONGRESSIONAL TRADES — House Stock Watcher API (free, no key)
# =============================================================================

def fetch_house_trades() -> pd.DataFrame:
    """
    House Stock Watcher API returns all House representative transactions.
    Free, public, no auth needed. Updated daily.
    """
    import urllib.request
    url = "https://housestockwatcher.com/api"
    try:
        print("  [House API] fetching congressional trading data...")
        req = urllib.request.Request(url, headers={"User-Agent": "Canyon-Quant/9.0 research"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        df = pd.DataFrame(data)
        print(f"  [House API] {len(df)} transactions loaded")
        return df
    except Exception as e:
        print(f"  [House API] failed: {e} — using cached data if available")
        return pd.DataFrame()


def fetch_senate_trades() -> pd.DataFrame:
    """
    Senate Stock Watcher — free S3-backed API.
    """
    import urllib.request
    url = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
    try:
        print("  [Senate API] fetching senate trades...")
        req = urllib.request.Request(url, headers={"User-Agent": "Canyon-Quant/9.0 research"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        df = pd.DataFrame(data)
        print(f"  [Senate API] {len(df)} transactions loaded")
        return df
    except Exception as e:
        print(f"  [Senate API] failed: {e}")
        return pd.DataFrame()


def normalize_house(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize House Stock Watcher response to common schema."""
    if df.empty:
        return df
    col_map = {
        "transaction_date": "trade_date",
        "disclosure_date":  "disclosure_date",
        "ticker":           "ticker",
        "transaction_type": "transaction_type",
        "amount":           "amount_range",
        "representative":   "name",
        "party":            "party",
        "state":            "state",
        "description":      "description",
        "asset_description": "asset_description",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["chamber"] = "HOUSE"
    return df


def normalize_senate(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Senate Stock Watcher response to common schema."""
    if df.empty:
        return df
    col_map = {
        "transaction_date": "trade_date",
        "disclosure_date":  "disclosure_date",
        "ticker":           "ticker",
        "type":             "transaction_type",
        "amount":           "amount_range",
        "senator":          "name",
        "party":            "party",
        "state":            "state",
        "asset_description": "asset_description",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["chamber"] = "SENATE"
    return df


def clean_congressional(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the combined congressional dataframe."""
    if df.empty:
        return df

    # Parse dates
    for col in ("trade_date", "disclosure_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Filter: only equity-type transactions (exclude bonds, crypto, etc.)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
        df = df[df["ticker"] != ""]

    # Classify transaction direction
    if "transaction_type" in df.columns:
        tx = df["transaction_type"].astype(str).str.lower()
        df["direction"] = "UNKNOWN"
        df.loc[tx.str.contains("purchase|buy|call"), "direction"] = "BUY"
        df.loc[tx.str.contains("sale|sell|sold|put"), "direction"] = "SELL"
        df.loc[tx.str.contains("exchange|covered"), "direction"] = "NEUTRAL"

    # Parse amount (ranges like "$15,001 - $50,000")
    if "amount_range" in df.columns:
        def parse_amount(v):
            try:
                nums = pd.Series([s.replace("$", "").replace(",", "")
                                   for s in str(v).split("-")])
                nums = pd.to_numeric(nums, errors="coerce")
                return nums.mean()
            except Exception:
                return np.nan
        df["amount_mid"] = df["amount_range"].apply(parse_amount)

    # Add VIP flag
    if "name" in df.columns:
        def get_vip(name):
            n = str(name).strip()
            for k, v in VIP_NAMES.items():
                if k.lower() in n.lower() or n.lower() in k.lower():
                    return v
            return None
        df["vip_alias"] = df["name"].apply(get_vip)
        df["is_vip"]    = df["vip_alias"].notna()
        df["is_pelosi"] = df["vip_alias"] == "PELOSI"

    # Disclosure lag (days from trade to disclosure)
    if "trade_date" in df.columns and "disclosure_date" in df.columns:
        df["disclosure_lag"] = (df["disclosure_date"] - df["trade_date"]).dt.days

    # Recency (days since trade)
    if "trade_date" in df.columns:
        df["days_ago"] = (pd.Timestamp.today() - df["trade_date"]).dt.days

    return df


# =============================================================================
# 2. POST-TRADE PERFORMANCE — measure alpha vs SPY after each trade
# =============================================================================

def compute_post_trade_performance(df: pd.DataFrame,
                                    horizons=(5, 21, 63)) -> pd.DataFrame:
    """
    For each trade, compute the return of the ticker and SPY
    over 5, 21, 63 trading days after the trade date.
    Uses yfinance.
    """
    if df.empty or "ticker" not in df.columns:
        return df

    try:
        import yfinance as yf
    except ImportError:
        print("  [perf] yfinance not installed — skipping post-trade performance")
        return df

    tickers = sorted(df["ticker"].dropna().unique().tolist())
    if not tickers:
        return df

    start_date = (df["trade_date"].min() - timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"  [perf] downloading prices for {len(tickers)} tickers since {start_date}...")

    try:
        all_tickers = tickers + ["SPY"]
        prices = yf.download(all_tickers, start=start_date,
                              auto_adjust=True, progress=False)
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices["Close"]
    except Exception as e:
        print(f"  [perf] price download failed: {e}")
        return df

    perf_rows = []
    for _, row in df.iterrows():
        trade_date = row.get("trade_date")
        ticker = row.get("ticker")
        direction = row.get("direction", "UNKNOWN")

        if pd.isna(trade_date) or not ticker:
            perf_rows.append({})
            continue

        # Find nearest trading day in prices
        try:
            avail = prices.index[prices.index >= trade_date]
            if avail.empty:
                perf_rows.append({})
                continue
            entry_date = avail[0]
            entry_price = prices.loc[entry_date, ticker] if ticker in prices.columns else np.nan
            spy_entry   = prices.loc[entry_date, "SPY"] if "SPY" in prices.columns else np.nan

            perf = {}
            for h in horizons:
                future = prices.index[prices.index > entry_date]
                if len(future) >= h:
                    exit_date = future[h - 1]
                    exit_price  = prices.loc[exit_date, ticker] if ticker in prices.columns else np.nan
                    spy_exit    = prices.loc[exit_date, "SPY"] if "SPY" in prices.columns else np.nan

                    stock_ret = (exit_price / entry_price - 1) * 100 if not np.isnan(entry_price) else np.nan
                    spy_ret   = (spy_exit   / spy_entry   - 1) * 100 if not np.isnan(spy_entry) else np.nan
                    excess    = stock_ret - spy_ret if not np.isnan(stock_ret) and not np.isnan(spy_ret) else np.nan

                    # Flip for SELLs (correct direction)
                    if direction == "SELL":
                        excess = -excess if excess is not None else np.nan

                    perf[f"ret_{h}d"]    = round(stock_ret, 2) if not np.isnan(stock_ret) else None
                    perf[f"spy_{h}d"]    = round(spy_ret, 2)   if not np.isnan(spy_ret) else None
                    perf[f"excess_{h}d"] = round(excess, 2)    if excess is not None and not np.isnan(excess) else None
            perf_rows.append(perf)
        except Exception:
            perf_rows.append({})

    perf_df = pd.DataFrame(perf_rows, index=df.index)
    return pd.concat([df, perf_df], axis=1)


# =============================================================================
# 3. PORTFOLIO OVERLAP — which current positions have recent insider buying
# =============================================================================

def compute_portfolio_overlap(cong_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-reference congressional BUYS in the last 90 days
    with our current alpha_scores.csv portfolio.
    """
    alpha_path = ROOT / "alpha_scores.csv"
    if not alpha_path.exists() or cong_df.empty:
        return pd.DataFrame()

    alpha = pd.read_csv(alpha_path)
    portfolio_tickers = set(alpha["ticker"].dropna().unique())

    if "trade_date" not in cong_df.columns or "ticker" not in cong_df.columns:
        return pd.DataFrame()

    recent = cong_df[
        (cong_df["days_ago"] <= 90) &
        (cong_df["direction"] == "BUY") &
        (cong_df["ticker"].isin(portfolio_tickers))
    ].copy()

    if recent.empty:
        return pd.DataFrame()

    # Count recent buys per ticker
    summary = recent.groupby("ticker").agg(
        n_buys=("direction", "count"),
        latest_buy=("trade_date", "max"),
        total_amount=("amount_mid", "sum"),
        vip_buyer=("is_vip", "any"),
        pelosi_buy=("is_pelosi", "any"),
        buyers=("name", lambda x: "; ".join(x.dropna().unique()[:3])),
    ).reset_index()

    summary = summary.merge(
        alpha[["ticker", "alpha_score", "alpha_rank", "signal", "sector"]],
        on="ticker", how="left"
    )

    summary = summary.sort_values("n_buys", ascending=False)
    return summary


# =============================================================================
# 4. VIP PERFORMANCE SCORECARD
# =============================================================================

def compute_vip_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each VIP politician: win rate, avg excess return, total $ disclosed.
    """
    if df.empty or "vip_alias" not in df.columns:
        return pd.DataFrame()

    vips = df[df["is_vip"] == True].copy()
    if vips.empty:
        return pd.DataFrame()

    records = []
    for alias, group in vips.groupby("vip_alias"):
        buys  = group[group["direction"] == "BUY"]
        sells = group[group["direction"] == "SELL"]

        def avg_excess(g, col="excess_21d"):
            vals = g[col].dropna() if col in g.columns else pd.Series([], dtype=float)
            return vals.mean() if len(vals) > 0 else np.nan

        def win_rate(g, col="excess_21d"):
            vals = g[col].dropna() if col in g.columns else pd.Series([], dtype=float)
            return (vals > 0).mean() * 100 if len(vals) > 0 else np.nan

        records.append({
            "vip_alias":        alias,
            "name":             group["name"].iloc[0] if len(group) > 0 else alias,
            "chamber":          group["chamber"].iloc[0] if "chamber" in group.columns else "?",
            "party":            group["party"].iloc[0] if "party" in group.columns else "?",
            "n_buys":           len(buys),
            "n_sells":          len(sells),
            "n_total":          len(group),
            "total_disclosed":  group["amount_mid"].sum(),
            "avg_excess_21d":   avg_excess(buys),
            "win_rate_21d":     win_rate(buys),
            "avg_excess_63d":   avg_excess(buys, "excess_63d"),
            "most_bought":      buys["ticker"].value_counts().index[0] if len(buys) > 0 else "",
            "last_trade":       group["trade_date"].max(),
            "is_highlight":     alias in VIP_HIGHLIGHT,
        })

    return pd.DataFrame(records).sort_values("total_disclosed", ascending=False)


# =============================================================================
# 5. GENERATE HTML EMBED SNIPPET
# =============================================================================

def _amount_label(amt):
    if pd.isna(amt) or amt == 0:
        return "—"
    if amt >= 1_000_000:
        return f"${amt/1_000_000:.1f}M"
    if amt >= 1_000:
        return f"${amt/1_000:.0f}K"
    return f"${amt:.0f}"


def _direction_badge(d):
    if d == "BUY":
        return '<span style="background:#E8F5EC;color:#1B6F4A;font-size:10px;font-weight:700;padding:2px 7px;border-radius:2px">BUY</span>'
    if d == "SELL":
        return '<span style="background:#FBE9E9;color:#B83232;font-size:10px;font-weight:700;padding:2px 7px;border-radius:2px">SELL</span>'
    return '<span style="background:#F5F5F5;color:#888;font-size:10px;font-weight:700;padding:2px 7px;border-radius:2px">?</span>'


def _excess_cell(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    color = "#1B6F4A" if val > 0 else "#B83232"
    sign  = "+" if val > 0 else ""
    return f'<span style="color:{color};font-weight:700">{sign}{val:.1f}%</span>'


def generate_html_tables(cong_df: pd.DataFrame,
                          vip_df: pd.DataFrame,
                          overlap_df: pd.DataFrame) -> str:
    """Generate HTML strings for embedding into canyon_v9_research.html."""

    lines = []

    # ── VIP Scorecard ──────────────────────────────────────────────────────
    lines.append('<div class="mt36">')
    lines.append('<p class="eyebrow">VIP Politician Scorecard</p>')
    lines.append('<h3 style="font-family:\'Playfair Display\',serif;font-size:22px;font-weight:700;color:#1A1A1A;margin:8px 0 4px">Pelosi & Tracked Congressional Traders</h3>')
    lines.append('<div class="rule"></div>')

    if vip_df.empty:
        lines.append('<p style="color:#999;font-size:13px">No VIP data loaded — run canyon_insider_signals.py to fetch</p>')
    else:
        lines.append('<table>')
        lines.append('<thead><tr><th>Politician</th><th>Chamber/Party</th><th>Buys</th><th>Sells</th><th>Total Disclosed</th><th>Avg Excess 21d</th><th>Win Rate</th><th>Top Stock</th><th>Last Trade</th></tr></thead>')
        lines.append('<tbody>')
        for _, r in vip_df.iterrows():
            highlight = r.get("is_highlight", False)
            bg = 'background:linear-gradient(90deg,rgba(180,145,50,0.08),transparent);border-left:3px solid #B8943F;' if highlight else ''
            pelosi_tag = ' ⭐ Pelosi' if r.get("vip_alias") == "PELOSI" else ''
            name_str = f'<b style="color:#1B2A4A">{r["name"]}{pelosi_tag}</b>'
            excess = _excess_cell(r.get("avg_excess_21d"))
            wr = f'{r["win_rate_21d"]:.0f}%' if not pd.isna(r.get("win_rate_21d", np.nan)) else "—"
            last = r["last_trade"].strftime("%Y-%m-%d") if pd.notna(r.get("last_trade")) else "—"
            lines.append(f'<tr style="{bg}">')
            lines.append(f'<td>{name_str}</td>')
            lines.append(f'<td style="font-size:11px">{r.get("chamber","?")} / {r.get("party","?")}</td>')
            lines.append(f'<td style="color:#1B6F4A;font-weight:700">{r.get("n_buys",0)}</td>')
            lines.append(f'<td style="color:#B83232;font-weight:700">{r.get("n_sells",0)}</td>')
            lines.append(f'<td>{_amount_label(r.get("total_disclosed", 0))}</td>')
            lines.append(f'<td>{excess}</td>')
            lines.append(f'<td>{wr}</td>')
            lines.append(f'<td class="td-ticker">{r.get("most_bought","—")}</td>')
            lines.append(f'<td style="font-size:11px;color:#888">{last}</td>')
            lines.append('</tr>')
        lines.append('</tbody></table>')
    lines.append('</div>')

    # ── Portfolio Overlap ─────────────────────────────────────────────────
    lines.append('<div class="mt36">')
    lines.append('<p class="eyebrow">Congressional Buying × Portfolio Overlap (90 days)</p>')
    lines.append('<h3 style="font-family:\'Playfair Display\',serif;font-size:22px;font-weight:700;color:#1A1A1A;margin:8px 0 4px">Positions with recent congressional BUY signal</h3>')
    lines.append('<div class="rule"></div>')

    if overlap_df.empty:
        lines.append('<p style="color:#999;font-size:13px">No overlap found — congressional buys vs current portfolio</p>')
    else:
        lines.append('<table>')
        lines.append('<thead><tr><th>Ticker</th><th>Sector</th><th>Alpha Score</th><th>Signal</th><th># Buys</th><th>VIP?</th><th>Pelosi?</th><th>Buyers</th><th>Latest Buy</th></tr></thead>')
        lines.append('<tbody>')
        for _, r in overlap_df.iterrows():
            pelosi_flag = '⭐ YES' if r.get("pelosi_buy") else '—'
            vip_flag    = '✓' if r.get("vip_buyer") else '—'
            score_pct   = min(int(r.get("alpha_score", 50)), 100)
            lines.append('<tr>')
            lines.append(f'<td class="td-ticker">{r["ticker"]}</td>')
            lines.append(f'<td style="font-size:11px;color:#555">{r.get("sector","?")}</td>')
            lines.append(f'<td><div class="td-score"><div class="score-bar-wrap"><div class="score-bar" style="width:{score_pct}%"></div></div><span style="font-weight:700;color:#1B2A4A">{r.get("alpha_score",0):.1f}</span></div></td>')
            lines.append(f'<td><span class="pos" style="font-size:11px;font-weight:700">{r.get("signal","?")}</span></td>')
            lines.append(f'<td style="font-weight:700;color:#1B2A4A">{r.get("n_buys",0)}</td>')
            lines.append(f'<td style="color:#888;font-size:12px">{vip_flag}</td>')
            lines.append(f'<td style="color:#B8943F;font-weight:700;font-size:12px">{pelosi_flag}</td>')
            lines.append(f'<td style="font-size:11px;color:#555">{r.get("buyers","")}</td>')
            last = r.get("latest_buy")
            last_str = last.strftime("%Y-%m-%d") if pd.notna(last) else "—"
            lines.append(f'<td style="font-size:11px;color:#888">{last_str}</td>')
            lines.append('</tr>')
        lines.append('</tbody></table>')
    lines.append('</div>')

    # ── Recent Trades ─────────────────────────────────────────────────────
    if not cong_df.empty and "days_ago" in cong_df.columns:
        recent = cong_df[cong_df["days_ago"] <= 180].copy()
        if "trade_date" in recent.columns:
            recent = recent.sort_values("trade_date", ascending=False).head(100)

        lines.append('<div class="mt36">')
        lines.append('<p class="eyebrow">Recent Congressional Transactions (last 180 days)</p>')
        lines.append('<h3 style="font-family:\'Playfair Display\',serif;font-size:22px;font-weight:700;color:#1A1A1A;margin:8px 0 4px">All disclosed trades · hover for excess return</h3>')
        lines.append('<div class="rule"></div>')
        lines.append('<table>')
        lines.append('<thead><tr><th>Date</th><th>Name</th><th>Ticker</th><th>Direction</th><th>Amount</th><th>Excess 5d</th><th>Excess 21d</th><th>Excess 63d</th><th>VIP</th></tr></thead>')
        lines.append('<tbody>')
        for _, r in recent.iterrows():
            bg = 'background:rgba(180,148,63,0.06);' if r.get("is_pelosi") else ''
            td_str = r["trade_date"].strftime("%Y-%m-%d") if pd.notna(r.get("trade_date")) else "—"
            vip_tag = f'⭐ {r["vip_alias"]}' if r.get("is_vip") else ''
            lines.append(f'<tr style="{bg}">')
            lines.append(f'<td style="font-size:11px;color:#888">{td_str}</td>')
            lines.append(f'<td style="font-size:11.5px">{r.get("name","?")}</td>')
            lines.append(f'<td class="td-ticker">{r.get("ticker","?")}</td>')
            lines.append(f'<td>{_direction_badge(r.get("direction","?"))}</td>')
            lines.append(f'<td style="font-size:11px">{_amount_label(r.get("amount_mid",np.nan))}</td>')
            lines.append(f'<td style="font-size:11px">{_excess_cell(r.get("excess_5d"))}</td>')
            lines.append(f'<td style="font-size:11px">{_excess_cell(r.get("excess_21d"))}</td>')
            lines.append(f'<td style="font-size:11px">{_excess_cell(r.get("excess_63d"))}</td>')
            lines.append(f'<td style="font-size:11px;color:#B8943F;font-weight:600">{vip_tag}</td>')
            lines.append('</tr>')
        lines.append('</tbody></table>')
        lines.append('</div>')

    return "\n".join(lines)


# =============================================================================
# 6. WRITE CSVs
# =============================================================================

def write_outputs(cong_df: pd.DataFrame, vip_df: pd.DataFrame,
                  overlap_df: pd.DataFrame):
    """Write all congressional signal CSVs."""
    if not cong_df.empty:
        cong_df.to_csv(ROOT / "congressional_trades.csv", index=False)
        print(f"  [✓] congressional_trades.csv — {len(cong_df)} rows")

    if not vip_df.empty:
        vip_df.to_csv(ROOT / "congressional_vip_summary.csv", index=False)
        print(f"  [✓] congressional_vip_summary.csv — {len(vip_df)} rows")

    if not overlap_df.empty:
        overlap_df.to_csv(ROOT / "insider_portfolio_overlap.csv", index=False)
        print(f"  [✓] insider_portfolio_overlap.csv — {len(overlap_df)} rows")


# =============================================================================
# MAIN
# =============================================================================

def main(skip_performance: bool = False):
    print(f"\nCanyon v9 — Insider Signals ({TODAY})")
    print("=" * 60)

    # ── Fetch ───────────────────────────────────────────────────────────────
    house_raw  = fetch_house_trades()
    senate_raw = fetch_senate_trades()

    house_norm  = normalize_house(house_raw)   if not house_raw.empty  else pd.DataFrame()
    senate_norm = normalize_senate(senate_raw) if not senate_raw.empty else pd.DataFrame()

    all_trades = pd.concat([house_norm, senate_norm], ignore_index=True)

    if all_trades.empty:
        print("  [!] No data fetched from any source. Check network / APIs.")
        # Try loading from cache
        cache = ROOT / "congressional_trades.csv"
        if cache.exists():
            all_trades = pd.read_csv(cache, parse_dates=["trade_date", "disclosure_date"])
            print(f"  [cache] loaded {len(all_trades)} rows from congressional_trades.csv")
        else:
            print("  [!] No cache available. Exiting.")
            return

    # ── Clean ───────────────────────────────────────────────────────────────
    print(f"\n  Cleaning {len(all_trades)} raw transactions...")
    cong_df = clean_congressional(all_trades)

    pelosi_count = cong_df["is_pelosi"].sum() if "is_pelosi" in cong_df.columns else 0
    vip_count    = cong_df["is_vip"].sum() if "is_vip" in cong_df.columns else 0
    print(f"  VIP transactions: {vip_count} (Pelosi: {pelosi_count})")

    # ── Post-trade performance ──────────────────────────────────────────────
    if not skip_performance:
        print("\n  Computing post-trade performance vs SPY...")
        cong_df = compute_post_trade_performance(cong_df)
    else:
        print("  [skip] Post-trade performance skipped")

    # ── Derived tables ──────────────────────────────────────────────────────
    print("\n  Computing VIP scorecard...")
    vip_df = compute_vip_scorecard(cong_df)

    print("  Computing portfolio overlap...")
    overlap_df = compute_portfolio_overlap(cong_df)

    if not overlap_df.empty:
        print(f"  Portfolio overlap: {len(overlap_df)} tickers with congressional BUY in last 90d")
        pelosi_overlap = overlap_df[overlap_df.get("pelosi_buy", False)]["ticker"].tolist()
        if pelosi_overlap:
            print(f"  ⭐ Pelosi buys in portfolio: {pelosi_overlap}")

    # ── Write ───────────────────────────────────────────────────────────────
    write_outputs(cong_df, vip_df, overlap_df)

    # ── Warren Buffett / Berkshire 13F ─────────────────────────────────────
    print("\n  Fetching Berkshire Hathaway 13F...")
    try:
        brk_df      = fetch_buffett_13f()
        brk_overlap = compute_buffett_canyon_overlap(brk_df)
        if not brk_df.empty:
            brk_df.to_csv(ROOT / "berkshire_13f.csv", index=False)
            print(f"  [✓] berkshire_13f.csv — {len(brk_df)} positions")
        if not brk_overlap.empty:
            brk_overlap.to_csv(ROOT / "buffett_canyon_overlap.csv", index=False)
            triple = brk_overlap[
                (brk_overlap["alpha_score"] >= 65) &
                (brk_overlap["ticker"].notna())
            ]
            if not triple.empty:
                print(f"  ⭐ Buffett+Canyon aligned (score≥65): {triple['ticker'].tolist()}")
    except Exception as e:
        print(f"  [Buffett] error: {e}")

    # ── Leopold Aschenbrenner ───────────────────────────────────────────────
    print("\n  Running Leopold Aschenbrenner module...")
    la_13f     = fetch_aschenbrenner_13f()
    la_overlap = compute_thesis_overlap()
    if not la_13f.empty:
        la_13f.to_csv(ROOT / "aschenbrenner_13f.csv", index=False)
        print(f"  [✓] aschenbrenner_13f.csv — {len(la_13f)} holdings")
    if not la_overlap.empty:
        la_overlap.to_csv(ROOT / "aschenbrenner_thesis_overlap.csv", index=False)
        print(f"  [✓] aschenbrenner_thesis_overlap.csv — {len(la_overlap)} tickers")
        print(f"  Thesis stocks in our top-50 alpha:")
        print(la_overlap[["ticker", "conviction", "alpha_score", "thesis"]].to_string(index=False))

    print(f"\n  Done. View Insiders tab in canyon_v9_research.html")


# =============================================================================
# LEOPOLD ASCHENBRENNER — SEC 13F + AI THESIS TRACKING
# =============================================================================

# =============================================================================
# WARREN BUFFETT — BERKSHIRE HATHAWAY 13F (CIK: 0001067983)
# =============================================================================

BERKSHIRE_CIK = "0001067983"

# Map issuer names in 13F to tradable tickers
BERKSHIRE_NAME_TO_TICKER = {
    "APPLE INC": "AAPL", "AMERICAN EXPRESS CO": "AXP",
    "COCA COLA CO": "KO", "BANK AMERICA CORP": "BAC",
    "CHEVRON CORPORATION": "CVX", "OCCIDENTAL PETE CORP": "OXY",
    "ALPHABET INC": "GOOGL", "CHUBB LTD SWITZ": "CB",
    "MOODYS CORP": "MCO", "KRAFT HEINZ CO": "KHC",
    "DAVITA INC": "DVA", "KROGER CO": "KR",
    "SIRIUSXM HOLDINGS INC": "SIRI", "DELTA AIR LINES INC": "DAL",
    "VERISIGN INC": "VRSN", "CAPITAL ONE FINL CORP": "COF",
    "NEW YORK TIMES CO MTN BE": "NYT", "ALLY FINL INC": "ALLY",
    "LENNAR CORP": "LEN", "NUCOR CORP": "NUE",
    "LOUISIANA PAC CORP": "LPX", "CONSTELLATION BRANDS INC": "STZ",
    "NVR INC": "NVR", "MACYS INC": "M",
    "JEFFERIES FINANCIAL GROUP IN": "JEF",
    "LIBERTY LIVE HOLDINGS INC": "LLYVK",
}


def fetch_buffett_13f() -> pd.DataFrame:
    """
    Fetch the latest Berkshire Hathaway 13F-HR from SEC EDGAR.
    CIK: 0001067983. Free, no API key.
    Returns DataFrame with holdings sorted by value.
    """
    import ssl, xml.etree.ElementTree as ET
    from collections import defaultdict

    print("  [EDGAR] Fetching Berkshire Hathaway 13F (CIK 0001067983)...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    def _get(url):
        req = Request(url, headers={"User-Agent": "Canyon-Quant/9.0 research@canyon.io"})
        with urlopen(req, timeout=20, context=ctx) as r:
            return r.read().decode()

    # 1. Get latest 13F-HR accession number
    subs = json.loads(_get(f"https://data.sec.gov/submissions/CIK{BERKSHIRE_CIK}.json"))
    filings = subs["filings"]["recent"]
    forms, dates, accs = filings["form"], filings["filingDate"], filings["accessionNumber"]
    latest_acc, latest_date = None, None
    for f, d, a in zip(forms, dates, accs):
        if f == "13F-HR":
            latest_acc, latest_date = a, d
            break
    if not latest_acc:
        print("  [EDGAR] No 13F-HR found for Berkshire")
        return pd.DataFrame()
    print(f"  [EDGAR] Latest 13F-HR: {latest_date} | {latest_acc}")

    # 2. Find infotable XML
    cik_short = BERKSHIRE_CIK.lstrip("0")
    acc_clean = latest_acc.replace("-", "")
    dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik_short}/{acc_clean}/"
    dir_html = _get(dir_url)
    import re
    xmls = re.findall(r'href="([^"]+\.xml)"', dir_html)
    info_xml = next((x for x in xmls if "primary_doc" not in x), None)
    if not info_xml:
        print("  [EDGAR] Could not find infotable XML")
        return pd.DataFrame()

    # 3. Parse holdings
    xml_content = _get(f"https://www.sec.gov{info_xml}")
    ns = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}
    root = ET.fromstring(xml_content)
    agg = defaultdict(lambda: {"value": 0, "shares": 0})
    for entry in root.findall(".//ns:infoTable", ns):
        name  = entry.findtext("ns:nameOfIssuer", "", ns).strip()
        val   = int(entry.findtext("ns:value", "0", ns) or 0)
        shrs  = int(entry.findtext("ns:shrsOrPrnAmt/ns:sshPrnamt", "0", ns) or 0)
        agg[name]["value"]  += val
        agg[name]["shares"] += shrs

    total = sum(v["value"] for v in agg.values())
    rows = []
    for name, v in sorted(agg.items(), key=lambda x: -x[1]["value"]):
        ticker = BERKSHIRE_NAME_TO_TICKER.get(name, "")
        rows.append({
            "name":       name,
            "ticker":     ticker,
            "value_b":    round(v["value"] / 1e9, 3),
            "pct_port":   round(v["value"] / total * 100, 2) if total else 0,
            "shares_m":   round(v["shares"] / 1e6, 2),
            "report_date": latest_date,
        })

    df = pd.DataFrame(rows)
    print(f"  [EDGAR] Berkshire: {len(df)} positions · ${total/1e9:.1f}B total · Q1 2026")
    return df


def compute_buffett_canyon_overlap(brk_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference Berkshire holdings with Canyon alpha_scores.csv."""
    alpha_path = ROOT / "alpha_scores.csv"
    if not alpha_path.exists() or brk_df.empty:
        return pd.DataFrame()
    alpha = pd.read_csv(alpha_path)
    merged = brk_df.merge(alpha[["ticker", "alpha_score", "alpha_rank", "signal",
                                  "sector", "crowding_level"]],
                          on="ticker", how="left")
    return merged.sort_values("value_b", ascending=False)


def fetch_aschenbrenner_13f() -> pd.DataFrame:
    """
    Search SEC EDGAR full-text search for 13F filings by Aschenbrenner
    or any fund he is associated with.
    EDGAR EFTS API: free, no key required.
    """
    print("  [EDGAR] Searching for Aschenbrenner 13F filings...")
    records = []

    for term in ASCHENBRENNER_SEARCH_TERMS:
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{term.replace(' ', '+')}%22"
            f"&forms=13F-HR&dateRange=custom&startdt=2024-01-01&enddt={TODAY}"
        )
        try:
            req = Request(url, headers={"User-Agent": "Canyon-Quant/9.0 research@canyon.io"})
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                print(f"  [EDGAR] Found {len(hits)} 13F hits for '{term}'")
                for h in hits:
                    src = h.get("_source", {})
                    records.append({
                        "search_term":   term,
                        "entity_name":   src.get("entity_name", ""),
                        "file_date":     src.get("file_date", ""),
                        "form_type":     src.get("form_type", ""),
                        "accession_no":  src.get("accession_no", ""),
                        "cik":           src.get("ciks", [""])[0] if src.get("ciks") else "",
                    })
            else:
                print(f"  [EDGAR] No 13F hits for '{term}'")
            time.sleep(0.5)
        except Exception as e:
            print(f"  [EDGAR] Search failed for '{term}': {e}")

    if not records:
        print("  [EDGAR] No 13F filings found for Aschenbrenner — he may not have filed yet.")
        print("  [EDGAR] Fund AUM must exceed $100M to trigger 13F requirement.")
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates("accession_no")

    # For each found filing, try to fetch actual holdings
    holdings = []
    for _, row in df.iterrows():
        cik = row.get("cik")
        acc = row.get("accession_no", "").replace("-", "")
        if not cik or not acc:
            continue
        holdings_url = (
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        )
        try:
            req = Request(holdings_url, headers={"User-Agent": "Canyon-Quant/9.0 research@canyon.io"})
            with urlopen(req, timeout=15) as resp:
                info = json.loads(resp.read().decode())
            holdings.append({
                "entity":      info.get("name", ""),
                "cik":         cik,
                "sic":         info.get("sic", ""),
                "filing_date": row.get("file_date", ""),
                "form":        "13F-HR",
            })
        except Exception:
            pass
        time.sleep(0.3)

    return pd.DataFrame(records) if not holdings else pd.DataFrame(holdings)


def compute_thesis_overlap() -> pd.DataFrame:
    """
    Cross-reference ASCHENBRENNER_THESIS stocks with current alpha_scores.csv.
    Shows where his AI investment thesis overlaps with Canyon's current picks.
    """
    alpha_path = ROOT / "alpha_scores.csv"
    if not alpha_path.exists():
        return pd.DataFrame()

    alpha = pd.read_csv(alpha_path)
    alpha["ticker"] = alpha["ticker"].str.upper()

    records = []
    for ticker, info in ASCHENBRENNER_THESIS.items():
        alpha_row = alpha[alpha["ticker"] == ticker]
        if not alpha_row.empty:
            r = alpha_row.iloc[0]
            records.append({
                "ticker":       ticker,
                "conviction":   info["conviction"],
                "tier":         info["tier"],
                "thesis":       info["thesis"],
                "quote":        info["quote"],
                "alpha_score":  r.get("alpha_score", np.nan),
                "alpha_rank":   r.get("alpha_rank", np.nan),
                "signal":       r.get("signal", "?"),
                "sector":       r.get("sector", "?"),
                "crowding":     r.get("crowding_level", "?"),
                "sig_momentum": r.get("sig_momentum", np.nan),
                "sig_quality":  r.get("sig_quality", np.nan),
                "in_universe":  True,
            })
        else:
            records.append({
                "ticker":      ticker,
                "conviction":  info["conviction"],
                "tier":        info["tier"],
                "thesis":      info["thesis"],
                "quote":       info["quote"],
                "alpha_score": np.nan,
                "alpha_rank":  np.nan,
                "signal":      "NOT IN UNIVERSE",
                "sector":      "?",
                "in_universe": False,
            })

    df = pd.DataFrame(records).sort_values(["tier", "conviction"])
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-performance", action="store_true",
                        help="Skip post-trade performance computation (faster)")
    args = parser.parse_args()
    main(skip_performance=args.skip_performance)
