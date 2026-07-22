#!/usr/bin/env python3
"""
Canyon v9 — Institutional Layer Upgrades (L3–L10)
===================================================
Runs AFTER step0 (price + L1/L6 signals) and BEFORE step87 (alpha aggregator).

L3  Sector:   12-1 momentum, relative strength, cycle stage  → sector_momentum.csv
L4  Funds:    Operating accruals anomaly, Piotroski F-score   → accrual_scores.csv, piotroski_scores.csv
L5  Event:    Insider cluster detection, buy/sell intensity   → insider_cluster_scores.csv
L7  Options:  IV Rank (IVR), skew percentile                  → options_ivrank.csv
L8  Risk:     60d correlation matrix, CVaR, sized pos limits  → correlation_risk.csv, position_risk_limits.csv
L9  Exec:     ADV-based slippage + market impact model        → execution_cost_estimates.csv
L10 Learn:    Per-signal rolling IC, decay detection          → signal_ic_attribution.csv

No broker connection.  No live orders.  Research only.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def ok(msg: str):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str): print(f"  {YELLOW}⚠{RESET}  {msg}")
def log(msg: str):  print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def _load_prices(tail: int = 504) -> pd.DataFrame:
    for fname in ("sp500_price_cache.csv", "backtest_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
                return df.tail(tail)
            except Exception:
                continue
    return pd.DataFrame()


def _load_volume(tail: int = 504) -> pd.DataFrame:
    p = ROOT / "sp500_volume_cache.csv"
    if p.exists():
        try:
            return pd.read_csv(p, index_col=0, parse_dates=True).sort_index().tail(tail)
        except Exception:
            pass
    return pd.DataFrame()


def _cross_rank(series: pd.Series) -> pd.Series:
    """0-100 cross-sectional percentile rank."""
    return series.rank(pct=True, na_option="keep") * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# L3 — Sector Momentum & Cycle Stage
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_ETFS = {
    "Information Technology":  "XLK",
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
}

# Cycle stage rules based on 12-1 momentum and RS vs SPY percentile rank
# Sectors in top-quartile momentum AND positive RS → EXPANSION
# Top-half momentum + declining RS → PEAK
# Bottom-half momentum + negative RS → CONTRACTION
# Bottom-quartile momentum + improving RS → TROUGH
def _cycle_stage(mom_rank: float, rs_rank: float, rs_trend: float) -> str:
    if mom_rank >= 0.75 and rs_rank >= 0.5:
        return "EXPANSION"
    elif mom_rank >= 0.5 and rs_rank < 0.5 and rs_trend < 0:
        return "PEAK"
    elif mom_rank < 0.5 and rs_rank < 0.5:
        return "CONTRACTION"
    elif mom_rank < 0.25 and rs_trend > 0:
        return "TROUGH"
    else:
        return "NEUTRAL"


def upgrade_l3_sector_momentum(prices: pd.DataFrame) -> None:
    """
    L3: Compute 12-1 momentum, relative strength vs SPY, and cycle stage
    for all 11 GICS sector ETFs.  Writes sector_momentum.csv.
    """
    if prices.empty or "SPY" not in prices.columns:
        warn("L3: no price data — skipping sector momentum")
        return

    spy = prices["SPY"].dropna()
    rows = []

    for sector, etf in SECTOR_ETFS.items():
        if etf not in prices.columns:
            continue
        s = prices[etf].dropna()
        if len(s) < 63:
            continue

        # 12-1 month momentum (skip most recent month)
        mom_12_1 = float(s.iloc[-22] / s.iloc[-252] - 1) * 100 if len(s) >= 252 else float(s.iloc[-22] / s.iloc[0] - 1) * 100
        # 3m and 1m momentum
        mom_3m   = float(s.iloc[-1] / s.iloc[-63]  - 1) * 100 if len(s) >= 63  else 0.0
        mom_1m   = float(s.iloc[-1] / s.iloc[-22]  - 1) * 100 if len(s) >= 22  else 0.0
        # Relative strength vs SPY (20d rolling ratio, current vs 60d ago)
        spy_al = spy.reindex(s.index).dropna()
        s_al   = s.reindex(spy_al.index).dropna()
        rs_now = float((s_al / spy_al).iloc[-1])  if len(s_al) > 0 else 1.0
        rs_60d = float((s_al / spy_al).iloc[-60]) if len(s_al) > 60 else rs_now
        rs_trend = rs_now - rs_60d  # positive = sector gaining on SPY
        # Above/below 50d and 200d SMA
        sma_50  = float(s.rolling(50).mean().iloc[-1])  if len(s) >= 50  else float(s.mean())
        sma_200 = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else float(s.mean())
        above_50  = int(s.iloc[-1] > sma_50)
        above_200 = int(s.iloc[-1] > sma_200)

        rows.append({
            "sector":      sector,
            "etf":         etf,
            "mom_12_1":    round(mom_12_1, 3),
            "mom_3m":      round(mom_3m, 3),
            "mom_1m":      round(mom_1m, 3),
            "rs_vs_spy":   round(rs_now, 4),
            "rs_trend_60d":round(rs_trend * 100, 3),
            "above_50d":   above_50,
            "above_200d":  above_200,
            "cycle_stage": "",   # filled after ranking
            "updated_date": TODAY,
        })

    if not rows:
        warn("L3: no sector ETF data found in price cache")
        return

    df = pd.DataFrame(rows)

    # Cross-rank momentum and RS for cycle stage determination
    if len(df) >= 4:
        mom_ranks = df["mom_12_1"].rank(pct=True)
        rs_ranks  = df["rs_vs_spy"].rank(pct=True)
        df["mom_rank"] = mom_ranks.round(3)
        df["rs_rank"]  = rs_ranks.round(3)
        df["cycle_stage"] = [
            _cycle_stage(float(mom_ranks.iloc[i]), float(rs_ranks.iloc[i]),
                         float(df["rs_trend_60d"].iloc[i]))
            for i in range(len(df))
        ]
    else:
        df["mom_rank"] = 0.5
        df["rs_rank"]  = 0.5

    df.to_csv(ROOT / "sector_momentum.csv", index=False)
    top = df.nlargest(3, "mom_12_1")["sector"].tolist()
    bottom = df.nsmallest(3, "mom_12_1")["sector"].tolist()
    ok(f"sector_momentum.csv → {len(df)} sectors | Top: {top[:2]} | Bottom: {bottom[:2]}")


# ─────────────────────────────────────────────────────────────────────────────
# L4 — Fundamentals: Accrual Anomaly + Piotroski F-Score
# ─────────────────────────────────────────────────────────────────────────────

def _piotroski_from_cache(data: dict) -> int:
    """
    Piotroski (2000) F-score: 9 binary factors.
    Returns score 0-9 (higher = better quality).
    Uses whatever financial fields are available from fundamental_cache.json.
    """
    score = 0
    eps    = float(data.get("eps",    data.get("trailingEps", 0)) or 0)
    roa    = float(data.get("returnOnAssets", 0) or 0)
    cfo    = float(data.get("operatingCashflow", data.get("freeCashflow", 0)) or 0)
    debt   = float(data.get("totalDebt",  0) or 0)
    assets = float(data.get("totalAssets", 1) or 1)
    curr_r = float(data.get("currentRatio", 1) or 1)
    shares = float(data.get("sharesOutstanding", 0) or 0)
    margin = float(data.get("grossMargins",  data.get("profitMargins", 0)) or 0)
    at_turn = float(data.get("assetTurnover", data.get("revenueGrowth", 0)) or 0)

    # Profitability
    if roa > 0:         score += 1  # F1: positive ROA
    if cfo > 0:         score += 1  # F2: positive operating cash flow
    if roa > 0:         score += 1  # F3: increasing ROA (proxy: positive ROA)
    if cfo > assets * 0.01 and assets > 0:
        score += 1                  # F4: CFO / assets > 0 (accruals)
    # Leverage / Liquidity
    leverage = debt / assets if assets > 0 else 0
    if leverage < 0.5:  score += 1  # F5: low leverage
    if curr_r > 1.0:    score += 1  # F6: current ratio > 1
    # Operating efficiency
    if margin > 0.1:    score += 1  # F7: positive gross margin
    if at_turn > 0:     score += 1  # F8: positive asset turnover / revenue growth
    if eps > 0:         score += 1  # F9: positive EPS
    return score


def _try_edgar_pit_accruals() -> pd.DataFrame:
    """
    W12: Compute Sloan accruals from EDGAR PIT fundamentals.
    Returns DataFrame with [ticker, accrual_raw] or empty DF if data unavailable.
    PIT guarantee: only uses filings with know_date <= today.
    """
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return pd.DataFrame()
    try:
        import sys; sys.path.insert(0, str(ROOT))
        from data.edgar_pit import load_pit_fundamentals, compute_pit_accruals
        pit_df = load_pit_fundamentals(pit_path)
        if pit_df.empty:
            return pd.DataFrame()
        as_of = pd.Timestamp(TODAY)
        accruals = compute_pit_accruals(pit_df, as_of)
        if accruals.empty:
            return pd.DataFrame()
        df = accruals.reset_index()
        df.columns = ["ticker", "accrual_raw"]
        ok(f"L4 EDGAR PIT accruals: {len(df)} tickers (PIT: know_date ≤ {TODAY})")
        return df
    except Exception as e:
        warn(f"L4 EDGAR PIT accruals failed: {e}")
        return pd.DataFrame()


def _try_edgar_pit_piotroski() -> pd.DataFrame:
    """
    W12: Compute Piotroski F-score from EDGAR PIT fundamentals.
    Uses PIT data so no fundamental lookahead bias.
    Returns DataFrame with [ticker, piotroski_raw] or empty DF if unavailable.
    """
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return pd.DataFrame()
    try:
        import sys; sys.path.insert(0, str(ROOT))
        from data.edgar_pit import load_pit_fundamentals, get_pit_snapshot
        pit_df = load_pit_fundamentals(pit_path)
        if pit_df.empty:
            return pd.DataFrame()
        as_of = pd.Timestamp(TODAY)

        # Fetch each required metric
        ni     = get_pit_snapshot(pit_df, "net_income",    as_of)
        op_cf  = get_pit_snapshot(pit_df, "op_cf",         as_of)
        assets = get_pit_snapshot(pit_df, "total_assets",  as_of)
        liab   = get_pit_snapshot(pit_df, "total_liabilities", as_of)

        # Compute simplified 5-factor PIT Piotroski (from what EDGAR provides)
        tickers = ni.index.intersection(op_cf.index).intersection(assets.index)
        if tickers.empty:
            return pd.DataFrame()

        scores = {}
        for tkr in tickers:
            s = 0
            ni_v  = ni.get(tkr, 0) or 0
            cf_v  = op_cf.get(tkr, 0) or 0
            ast_v = assets.get(tkr, 1) or 1
            lib_v = liab.get(tkr, 0) if tkr in liab.index else 0

            roa = ni_v / ast_v
            if roa > 0:           s += 1   # F1: positive ROA
            if cf_v > 0:          s += 1   # F2: positive operating CF
            if cf_v > ni_v:       s += 1   # F4: CFO > NI (earnings quality)
            lev = lib_v / ast_v
            if lev < 0.5:         s += 1   # F5: low leverage
            if ni_v > 0:          s += 1   # F9: positive net income
            scores[tkr] = s

        df = pd.DataFrame(list(scores.items()), columns=["ticker", "piotroski_raw"])
        ok(f"L4 EDGAR PIT Piotroski: {len(df)} tickers (5-factor PIT version)")
        return df
    except Exception as e:
        warn(f"L4 EDGAR PIT Piotroski failed: {e}")
        return pd.DataFrame()


def upgrade_l4_fundamentals() -> None:
    """
    L4: Accrual anomaly + Piotroski F-score.
    Sources (in priority order):
      1. EDGAR PIT fundamentals (W12 upgrade — truly point-in-time)
      2. accruals_snapshot.csv (pre-existing, may have lookahead)
      3. fundamental_cache.json (pre-existing, non-PIT)
    Outputs:
      - accrual_scores.csv    (accrual_score 0-100: high = low accruals = better)
      - piotroski_scores.csv  (piotroski_score 0-9 → normalised 0-100)
    """
    # ── W12: Try EDGAR PIT accruals first ────────────────────────────────────
    pit_acc = _try_edgar_pit_accruals()
    if not pit_acc.empty:
        pit_acc["updated_date"] = TODAY
        pit_acc["accrual_score"] = _cross_rank(-pit_acc["accrual_raw"]).round(2)
        pit_acc.to_csv(ROOT / "accrual_scores.csv", index=False)
        ok(f"accrual_scores.csv → {len(pit_acc)} tickers (EDGAR PIT source)")
    else:
        # ── Fallback: accruals_snapshot.csv ──────────────────────────────────
        # ── Accrual anomaly (Sloan 1996) ─────────────────────────────────────
        pass  # fall through to original logic below

    # ── W12: Try EDGAR PIT Piotroski first ───────────────────────────────────
    pit_pio = _try_edgar_pit_piotroski()
    if not pit_pio.empty:
        pit_pio["updated_date"] = TODAY
        pit_pio["piotroski_score"] = _cross_rank(pit_pio["piotroski_raw"]).round(2)
        pit_pio.to_csv(ROOT / "piotroski_scores.csv", index=False)
        n_high = int((pit_pio["piotroski_raw"] >= 4).sum())  # ≥4 of 5 = high quality
        ok(f"piotroski_scores.csv → {len(pit_pio)} tickers (EDGAR PIT)  high-quality(F≥4)={n_high}")

    # ── Fallback: accruals_snapshot.csv (when EDGAR PIT unavailable) ─────────
    if pit_acc.empty:
        acc_path = ROOT / "accruals_snapshot.csv"
        acc_rows = []
        if acc_path.exists():
            try:
                acc_df = _read_csv(acc_path)
                for col in ("operating_accruals", "accrual_ratio", "accruals_to_assets"):
                    if col in acc_df.columns:
                        acc_df[col] = pd.to_numeric(acc_df[col], errors="coerce")
                        acc_clean = acc_df[["ticker", col]].dropna()
                        for _, row in acc_clean.iterrows():
                            acc_rows.append({
                                "ticker":      str(row["ticker"]).upper(),
                                "accrual_raw": float(row[col]),
                                "updated_date": TODAY,
                            })
                        break
            except Exception as e:
                warn(f"L4 accruals_snapshot.csv error: {e}")

        if acc_rows:
            acc_out = pd.DataFrame(acc_rows).drop_duplicates("ticker")
            acc_out["accrual_score"] = _cross_rank(-acc_out["accrual_raw"]).round(2)
            acc_out.to_csv(ROOT / "accrual_scores.csv", index=False)
            ok(f"accrual_scores.csv → {len(acc_out)} tickers (from accruals_snapshot.csv)")
        else:
            warn("L4: accruals_snapshot.csv not found or empty — accrual_scores.csv skipped")

    # ── Fallback: Piotroski from fundamental_cache.json (when EDGAR PIT unavailable)
    if pit_pio.empty:
        cache_path = ROOT / "fundamental_cache.json"
        if not cache_path.exists():
            warn("L4: fundamental_cache.json not found — piotroski_scores.csv skipped")
            return
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            warn(f"L4: fundamental_cache.json read error: {e}")
            return

        pio_rows = []
        for ticker, data in cache.items():
            if not isinstance(data, dict):
                continue
            f_score = _piotroski_from_cache(data)
            pio_rows.append({
                "ticker":        str(ticker).upper(),
                "piotroski_raw": f_score,
                "updated_date":  TODAY,
            })

        if pio_rows:
            pio_out = pd.DataFrame(pio_rows).drop_duplicates("ticker")
            pio_out["piotroski_score"] = _cross_rank(pio_out["piotroski_raw"]).round(2)
            pio_out.to_csv(ROOT / "piotroski_scores.csv", index=False)
            n_high = int((pio_out["piotroski_raw"] >= 7).sum())
            ok(f"piotroski_scores.csv → {len(pio_out)} tickers (fundamental_cache)  high-quality(F≥7)={n_high}")
        else:
            warn("L4: no Piotroski data computed — fundamental_cache.json may be empty")


# ─────────────────────────────────────────────────────────────────────────────
# L5 — Insider Cluster Detection
# ─────────────────────────────────────────────────────────────────────────────

def _try_form4_cluster(lookback_days: int = 90) -> pd.DataFrame:
    """
    W13: Compute insider cluster scores from EDGAR Form 4 data (PIT).
    Returns DataFrame with [ticker, buy_count, sell_count, cluster_flag,
    cluster_score, net_buy_ratio] or empty DF if Form 4 cache unavailable.
    """
    form4_path = ROOT / "edgar_form4_cache.csv"
    if not form4_path.exists():
        return pd.DataFrame()
    try:
        import sys; sys.path.insert(0, str(ROOT))
        from data.edgar_form4 import compute_insider_signal
        df = pd.read_csv(form4_path, parse_dates=["filed_date", "txn_date"])
        as_of = pd.Timestamp(TODAY)
        window_start = as_of - pd.Timedelta(days=lookback_days)

        # Only use PIT-valid transactions: filed_date <= today
        window = df[
            (df["filed_date"] >= window_start) &
            (df["filed_date"] <= as_of) &
            (df["txn_code"].isin(["P", "S"]))
        ].copy()

        if window.empty:
            return pd.DataFrame()

        # Compute cluster metrics per ticker
        rows = []
        for ticker, grp in window.groupby("ticker"):
            buys  = grp[grp["txn_code"] == "P"]
            sells = grp[grp["txn_code"] == "S"]
            buy_count  = len(buys)
            sell_count = len(sells)
            buy_value  = float((buys["shares"].fillna(0) * buys["price"].fillna(0)).sum())
            sell_value = float((sells["shares"].fillna(0) * sells["price"].fillna(0)).sum())
            c_suite_buy = int(buys["is_c_suite"].any() if len(buys) > 0 else 0)

            cluster_flag = int(buy_count >= 2 and sell_count == 0)
            total = buy_count + sell_count
            net_buy_ratio = buy_count / total if total > 0 else 0.5
            value_ratio   = buy_value / (sell_value + 1.0)

            # Cluster score: Lakonishok & Lee (2001) style
            cluster_score = (
                0.35 * cluster_flag +
                0.25 * min(net_buy_ratio, 1.0) +
                0.25 * c_suite_buy +                          # C-suite bonus
                0.15 * min(np.log1p(value_ratio) / 5.0, 1.0)
            ) * 100.0

            rows.append({
                "ticker":        ticker.upper(),
                "buy_count":     buy_count,
                "sell_count":    sell_count,
                "cluster_flag":  cluster_flag,
                "net_buy_ratio": round(net_buy_ratio, 3),
                "cluster_score": round(cluster_score, 2),
                "c_suite_buy":   c_suite_buy,
                "updated_date":  TODAY,
                "source":        "edgar_form4",  # tracks data source
            })

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        ok(f"L5 Form 4 cluster: {len(result)} tickers, {lookback_days}d lookback")
        return result

    except Exception as e:
        warn(f"L5 Form 4 cluster failed: {e}")
        return pd.DataFrame()


def upgrade_l5_insider_clusters() -> None:
    """
    L5: Institutional insider cluster detection.
    Clusters = multiple corporate insiders buying same ticker within 30 days.
    Sources (priority order):
      1. EDGAR Form 4 (W13 upgrade — real SEC filings, true PIT)
      2. insider_signal_scores.csv (existing pipeline, non-PIT)
    Output:  insider_cluster_scores.csv
    """
    # W13: Try Form 4 first
    form4_clusters = _try_form4_cluster(lookback_days=90)
    if not form4_clusters.empty:
        form4_clusters["cluster_score_rank"] = _cross_rank(form4_clusters["cluster_score"]).round(2)
        form4_clusters.to_csv(ROOT / "insider_cluster_scores.csv", index=False)
        n_cluster = int((form4_clusters["cluster_flag"] == 1).sum())
        ok(f"insider_cluster_scores.csv → {len(form4_clusters)} tickers (EDGAR Form 4)  clusters={n_cluster}")
        return  # Done — no need for fallback

    # Fallback: existing insider_signal_scores.csv
    ins_path = ROOT / "insider_signal_scores.csv"
    if not ins_path.exists():
        warn("L5: insider_signal_scores.csv not found — skipping")
        return

    try:
        ins = pd.read_csv(ins_path)
        if "ticker" not in ins.columns:
            warn("L5: insider_signal_scores.csv missing ticker column")
            return

        ins = ins.drop_duplicates("ticker").copy()
        rows = []

        for _, row in ins.iterrows():
            ticker = str(row["ticker"]).upper()

            # Count insider buyers in last 30 days
            buy_count  = int(float(row.get("insider_buy_count",  row.get("buy_count",   0)) or 0))
            sell_count = int(float(row.get("insider_sell_count", row.get("sell_count",  0)) or 0))
            buy_value  = float(row.get("insider_buy_value",  row.get("buy_value",  0)) or 0)
            sell_value = float(row.get("insider_sell_value", row.get("sell_value", 0)) or 0)
            rank_ins   = float(row.get("rank_insider", 50) or 50)

            # Cluster: 2+ buyers with no sellers = strongest signal (Lakonishok & Lee 2001)
            cluster_flag  = int(buy_count >= 2 and sell_count == 0)
            # Net buy ratio: buyers / (buyers + sellers), nan-safe
            total = buy_count + sell_count
            net_buy_ratio = buy_count / total if total > 0 else 0.5
            # Value intensity: $ bought vs $ sold
            value_ratio   = buy_value / (sell_value + 1.0)  # +1 avoids div/0

            # Cluster score: combines rank_insider + cluster flag + value intensity
            cluster_score = (
                0.40 * rank_ins / 100.0 +
                0.30 * cluster_flag +
                0.20 * min(net_buy_ratio, 1.0) +
                0.10 * min(np.log1p(value_ratio) / 5.0, 1.0)
            ) * 100.0

            rows.append({
                "ticker":          ticker,
                "buy_count":       buy_count,
                "sell_count":      sell_count,
                "cluster_flag":    cluster_flag,
                "net_buy_ratio":   round(net_buy_ratio, 3),
                "cluster_score":   round(cluster_score, 2),
                "updated_date":    TODAY,
            })

        out = pd.DataFrame(rows)
        # Cross-rank the cluster score
        out["cluster_score_rank"] = _cross_rank(out["cluster_score"]).round(2)
        out.to_csv(ROOT / "insider_cluster_scores.csv", index=False)
        n_cluster = int((out["cluster_flag"] == 1).sum())
        ok(f"insider_cluster_scores.csv → {len(out)} tickers  active clusters={n_cluster}")

    except Exception as e:
        warn(f"L5: insider cluster error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# L6 — 8-K Earnings Call FinBERT Sentiment (W22)
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_l6_earnings_sentiment() -> None:
    """
    L6 (W22): Fetch recent 8-K earnings releases, score with FinBERT, output z-scores.

    Sources (priority order):
      1. signals/earnings_call.py → SEC EDGAR 8-K + FinBERT (real-time)
      2. finbert_sentiment.csv (existing pipeline — no earnings-specific filter)

    Output: earnings_call_sentiment.csv  (ticker, filed_date, sentiment_score)
            Also updates finbert_sentiment.csv with new FinBERT scores when available.
    """
    import time as _t
    _cache = ROOT / "earnings_call_sentiment.csv"
    if _cache.exists() and (_t.time() - _cache.stat().st_mtime) < 23 * 3600:
        ok(f"L6: earnings_call_sentiment.csv is fresh (<23h) — skipping FinBERT fetch")
        return

    try:
        from signals.earnings_call import fetch_8k_sentiment, get_earnings_sentiment
        tickers = _get_universe_tickers()
        if not tickers:
            warn("L6: No ticker universe available — skipping")
            return

        sentiment_df = fetch_8k_sentiment(
            tickers,
            lookback_days=90,
            cache_path=ROOT / "earnings_call_sentiment.csv",
        )
        if sentiment_df.empty:
            warn("L6: No 8-K filings found (FinBERT may not be installed)")
            return

        # Also write a rank-normalised finbert_sentiment.csv for step87 consumption
        today = pd.Timestamp.today()
        scores = get_earnings_sentiment(sentiment_df, as_of=today, lookback_days=90)
        if not scores.empty:
            # Convert [-1, +1] z-scores → percentile 0-100 (compatible with step87's neutral=50)
            rank_sent = scores.rank(pct=True) * 100
            out = rank_sent.reset_index()
            out.columns = ["ticker", "rank_sentiment"]
            out["sentiment_zscore"] = scores.values
            out["updated_date"] = str(today.date())
            out.to_csv(ROOT / "finbert_sentiment.csv", index=False)
            ok(f"earnings_call_sentiment.csv + finbert_sentiment.csv → {len(scores)} tickers")

    except Exception as e:
        warn(f"L6: Earnings sentiment error: {e}")
        # Graceful: existing finbert_sentiment.csv stays unchanged


def _get_universe_tickers() -> list[str]:
    """Load ticker universe from price cache or S&P 500 constituents."""
    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            try:
                df = pd.read_csv(p, nrows=0)
                return [c for c in df.columns if c != "SPY"][:200]
            except Exception:
                pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# L7 — Options IV Rank (IVR) + Skew Percentile
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_l7_options_ivrank() -> None:
    """
    L7: IV Rank (IVR) = (current_IV - 52w_low_IV) / (52w_high_IV - 52w_low_IV)
    High IVR = options expensive → premium selling opportunity, or pre-event fear.
    Low IVR  = options cheap → pre-catalyst buy opportunity.

    Sources: vix_iv_rank_cache.json, options_signals.csv
    Output:  options_ivrank.csv
    """
    rows = []

    # ── From vix_iv_rank_cache.json (if exists) ────────────────────────────
    ivr_path = ROOT / "vix_iv_rank_cache.json"
    if ivr_path.exists():
        try:
            cache = json.loads(ivr_path.read_text(encoding="utf-8"))
            for ticker, data in cache.items():
                if not isinstance(data, dict):
                    continue
                iv_now  = float(data.get("iv_current", data.get("iv_30d",  0)) or 0)
                iv_high = float(data.get("iv_52w_high", iv_now * 1.5) or iv_now * 1.5)
                iv_low  = float(data.get("iv_52w_low",  iv_now * 0.5) or iv_now * 0.5)
                iv_range = iv_high - iv_low
                ivr     = float((iv_now - iv_low) / iv_range) if iv_range > 0 else 0.5
                skew    = float(data.get("skew", data.get("iv_skew", 0)) or 0)
                rows.append({
                    "ticker":         str(ticker).upper(),
                    "iv_current":     round(iv_now, 4),
                    "iv_52w_high":    round(iv_high, 4),
                    "iv_52w_low":     round(iv_low, 4),
                    "iv_rank":        round(min(max(ivr, 0.0), 1.0), 4),
                    "skew":           round(skew, 4),
                    "updated_date":   TODAY,
                })
        except Exception as e:
            warn(f"L7: vix_iv_rank_cache.json error: {e}")

    # ── Supplement from options_signals.csv ───────────────────────────────
    opts_path = ROOT / "options_signals.csv"
    if opts_path.exists():
        try:
            opts = pd.read_csv(opts_path)
            if "ticker" in opts.columns and "iv_30d" in opts.columns:
                existing_tickers = {r["ticker"] for r in rows}
                opts = opts.drop_duplicates("ticker")
                for _, row in opts.iterrows():
                    tk = str(row["ticker"]).upper()
                    if tk in existing_tickers:
                        continue
                    iv_now = float(row.get("iv_30d", row.get("iv_current", 0.3)) or 0.3)
                    # Estimate 52w range from available data (rough proxy)
                    iv_high = iv_now * 1.5
                    iv_low  = iv_now * 0.6
                    skew    = float(row.get("iv_skew", 0) or 0)
                    rows.append({
                        "ticker":       tk,
                        "iv_current":   round(iv_now, 4),
                        "iv_52w_high":  round(iv_high, 4),
                        "iv_52w_low":   round(iv_low, 4),
                        "iv_rank":      0.5,   # unknown without history
                        "skew":         round(skew, 4),
                        "updated_date": TODAY,
                    })
        except Exception as e:
            warn(f"L7: options_signals.csv error: {e}")

    if not rows:
        warn("L7: no IV data available — options_ivrank.csv skipped")
        return

    out = pd.DataFrame(rows).drop_duplicates("ticker")
    # IVR signal: high IVR (>0.80) + positive skew = bearish options environment
    # Low IVR (<0.20) = cheap options, good pre-catalyst entry
    out["ivr_signal"] = out["iv_rank"].apply(
        lambda x: "HIGH_IV"   if x > 0.80 else
                  ("LOW_IV"   if x < 0.20 else "NORMAL_IV")
    )
    # Score for alpha aggregator: low IV rank is more bullish (cheaper options)
    # Invert: score = 100 - iv_rank*100, then cross-rank
    out["ivr_score"] = _cross_rank(1.0 - out["iv_rank"]).round(2)
    out.to_csv(ROOT / "options_ivrank.csv", index=False)
    n_high = int((out["ivr_signal"] == "HIGH_IV").sum())
    n_low  = int((out["ivr_signal"] == "LOW_IV").sum())
    ok(f"options_ivrank.csv → {len(out)} tickers  HIGH_IV={n_high}  LOW_IV={n_low}")


# ─────────────────────────────────────────────────────────────────────────────
# L8 — Correlation Risk Matrix + CVaR + Position Limits
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_l8_correlation_risk(prices: pd.DataFrame) -> None:
    """
    L8: Institutional portfolio risk management.
    1. 60-day pairwise correlation matrix for current positions + watchlist
    2. Historical CVaR (5th percentile of 1-day portfolio return)
    3. Correlation-adjusted position limits (reduce limit when highly correlated)

    Outputs: correlation_risk.csv, position_risk_limits.csv
    """
    if prices.empty:
        warn("L8: no price data — skipping correlation risk")
        return

    # Determine relevant universe: paper ledger positions + alpha top-30
    tickers_to_analyze: list[str] = []

    ledger_path = ROOT / "paper_portfolio_ledger.csv"
    if ledger_path.exists():
        try:
            led = pd.read_csv(ledger_path)
            if "ticker" in led.columns:
                tickers_to_analyze += led["ticker"].dropna().str.upper().tolist()
        except Exception:
            pass

    alpha_path = ROOT / "alpha_scores.csv"
    if alpha_path.exists():
        try:
            alpha = pd.read_csv(alpha_path).nlargest(30, "alpha_score") if "alpha_score" in pd.read_csv(alpha_path).columns else pd.read_csv(alpha_path).head(30)
            if "ticker" in alpha.columns:
                tickers_to_analyze += alpha["ticker"].dropna().str.upper().tolist()
        except Exception:
            pass

    # Deduplicate, keep only those with price data
    seen = set()
    universe = []
    for t in tickers_to_analyze:
        if t not in seen and t in prices.columns:
            seen.add(t)
            universe.append(t)

    if len(universe) < 4:
        warn(f"L8: only {len(universe)} tickers with price data — skipping")
        return

    # 60-day returns matrix
    rets_60 = prices[universe].tail(63).pct_change().dropna()
    if len(rets_60) < 20:
        warn("L8: insufficient return history for correlation")
        return

    # ── Pairwise correlation matrix ───────────────────────────────────────────
    corr_matrix = rets_60.corr()

    # ── Per-ticker risk metrics ───────────────────────────────────────────────
    rows = []
    for tk in universe:
        if tk not in rets_60.columns:
            continue
        tk_rets = rets_60[tk].dropna()
        # Average pairwise correlation with all other tickers
        corr_row = corr_matrix[tk].drop(tk, errors="ignore").dropna()
        avg_corr = float(corr_row.mean()) if len(corr_row) > 0 else 0.0

        # CVaR: expected shortfall at 5% level (60-day lookback)
        if len(tk_rets) >= 10:
            var_5 = float(np.percentile(tk_rets, 5))
            cvar_5 = float(tk_rets[tk_rets <= var_5].mean()) if (tk_rets <= var_5).any() else var_5
        else:
            var_5 = -0.03
            cvar_5 = -0.05

        # Annualised vol (60d)
        vol_60d = float(tk_rets.std() * np.sqrt(252))

        # Correlation-adjusted position limit:
        # Base = 15%. High avg correlation (>0.70) compresses limit.
        # Limit_adj = base * (1 - max(0, avg_corr - 0.40) / 0.60)
        base_limit = 0.15
        corr_penalty = max(0.0, (avg_corr - 0.40) / 0.60)
        adj_limit = round(base_limit * (1.0 - corr_penalty * 0.50), 4)
        adj_limit = max(0.02, adj_limit)  # floor 2%

        rows.append({
            "ticker":          tk,
            "avg_corr_60d":    round(avg_corr, 4),
            "cvar_5pct":       round(cvar_5 * 100, 3),   # in %
            "var_5pct":        round(var_5  * 100, 3),
            "vol_60d_ann":     round(vol_60d * 100, 2),
            "adj_pos_limit":   round(adj_limit * 100, 2), # in %
            "corr_risk_flag":  int(avg_corr > 0.70),
            "updated_date":    TODAY,
        })

    if not rows:
        warn("L8: no rows computed for correlation risk")
        return

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "correlation_risk.csv", index=False)

    # Position limits summary (for execution gate reference)
    limits = out[["ticker", "adj_pos_limit", "corr_risk_flag", "cvar_5pct"]].copy()
    limits.to_csv(ROOT / "position_risk_limits.csv", index=False)

    n_flagged = int((out["corr_risk_flag"] == 1).sum())
    avg_limit = float(out["adj_pos_limit"].mean())
    ok(f"correlation_risk.csv → {len(out)} tickers  high-corr={n_flagged}  avg_limit={avg_limit:.1f}%")

    # ── Save correlation matrix for reference ─────────────────────────────────
    corr_path = ROOT / "correlation_matrix_60d.csv"
    corr_matrix.round(4).to_csv(corr_path)
    ok(f"correlation_matrix_60d.csv → {corr_matrix.shape[0]}×{corr_matrix.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# L9 — Execution Cost Model (ADV-based slippage)
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_l9_slippage_model(prices: pd.DataFrame, volume: pd.DataFrame) -> None:
    """
    L9: Almgren-Chriss (2005) slippage model.
    Slippage_bps ≈ k × sqrt(order_size_pct / ADV) × daily_vol × 10000
    where k ≈ 0.314 (empirical constant for US equities).

    Also estimates:
    - Market impact: MI_bps ≈ η × (order_size_pct)^0.6 × daily_vol × 10000
    - Effective spread proxy: 0.1 × daily_vol × 10000 (for liquid stocks)

    Output: execution_cost_estimates.csv
    """
    if prices.empty:
        warn("L9: no price data — skipping slippage model")
        return

    K_IMPACT = 0.314   # Almgren-Chriss permanent impact constant
    ETA_TEMP  = 0.142  # temporary impact constant
    DEFAULT_ORDER_PCT = 0.01   # assume 1% of ADV as typical order size

    rows = []
    tickers = [c for c in prices.columns if c not in ("", "Date", "Unnamed: 0")]

    for tk in tickers[:500]:  # limit to first 500 for speed
        p = prices[tk].dropna()
        if len(p) < 22:
            continue

        rets = p.pct_change().dropna()
        daily_vol = float(rets.tail(21).std())

        # ADV in shares from volume cache, or estimate from price volatility
        if not volume.empty and tk in volume.columns:
            v = volume[tk].dropna()
            adv_shares = float(v.tail(21).mean()) if len(v) >= 21 else 0.0
            adv_dollar = adv_shares * float(p.iloc[-1]) / 1e6  # $M
        else:
            # Fallback: estimate ADV from price level (rough institutional proxy)
            price_now = float(p.iloc[-1])
            adv_dollar = max(price_now * 100000 / 1e6, 0.1)  # assume 100k shares
            adv_shares = adv_dollar * 1e6 / max(price_now, 1.0)

        # Slippage (bps): permanent price impact
        if adv_shares > 0 and daily_vol > 0:
            order_frac = DEFAULT_ORDER_PCT
            perm_impact_bps = float(K_IMPACT * np.sqrt(order_frac) * daily_vol * 10000)
            temp_impact_bps = float(ETA_TEMP * (order_frac ** 0.6) * daily_vol * 10000)
            total_cost_bps  = round(perm_impact_bps + temp_impact_bps, 1)
        else:
            total_cost_bps  = 10.0   # default 10bps for illiquid

        # Spread proxy
        spread_bps = round(max(5.0, 0.1 * daily_vol * 10000), 1)

        # Liquidity tier: institutional classification
        if adv_dollar >= 100:
            liquidity_tier = "MEGA"       # >$100M ADV  (almost no impact)
        elif adv_dollar >= 20:
            liquidity_tier = "LARGE"      # $20-100M ADV
        elif adv_dollar >= 5:
            liquidity_tier = "MID"        # $5-20M ADV
        elif adv_dollar >= 1:
            liquidity_tier = "SMALL"      # $1-5M ADV (execute over 3-5 days)
        else:
            liquidity_tier = "MICRO"      # <$1M ADV  (avoid or spread over weeks)

        rows.append({
            "ticker":            tk,
            "adv_dollar_m":      round(adv_dollar, 2),
            "daily_vol_pct":     round(daily_vol * 100, 3),
            "slippage_bps":      total_cost_bps,
            "perm_impact_bps":   round(perm_impact_bps if adv_shares > 0 else 5.0, 1),
            "temp_impact_bps":   round(temp_impact_bps if adv_shares > 0 else 5.0, 1),
            "spread_bps":        spread_bps,
            "total_cost_bps":    total_cost_bps,
            "liquidity_tier":    liquidity_tier,
            "updated_date":      TODAY,
        })

    if not rows:
        warn("L9: no slippage rows computed")
        return

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "execution_cost_estimates.csv", index=False)
    micro_count = int((out["liquidity_tier"] == "MICRO").sum())
    avg_cost    = float(out["total_cost_bps"].mean())
    ok(f"execution_cost_estimates.csv → {len(out)} tickers  avg_cost={avg_cost:.1f}bps  micro={micro_count}")


# ─────────────────────────────────────────────────────────────────────────────
# L10 — Per-Signal IC Attribution + Decay Detection
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_l10_ic_attribution(prices: pd.DataFrame) -> None:
    """
    L10: Rolling information coefficient (IC) per signal.
    IC = Spearman rank correlation between signal_score and forward_return_20d.
    Uses alpha_score_history.csv (signal breakdown) + price cache (fwd returns).

    Output: signal_ic_attribution.csv
    """
    hist_path = ROOT / "alpha_score_history.csv"
    if not hist_path.exists():
        warn("L10: alpha_score_history.csv not found — skipping IC attribution")
        return

    if prices.empty:
        warn("L10: no price data for IC computation — skipping")
        return

    try:
        hist = pd.read_csv(hist_path)
        if "date" not in hist.columns or "ticker" not in hist.columns:
            warn("L10: alpha_score_history.csv missing required columns")
            return

        hist["date"] = pd.to_datetime(hist["date"])
        dates = sorted(hist["date"].unique())

        if len(dates) < 5:
            warn(f"L10: only {len(dates)} history dates — need ≥5 for IC")
            return

        # Find signal columns (sig_xxx pattern from step87)
        sig_cols = [c for c in hist.columns if c.startswith("sig_")]
        if not sig_cols:
            # Fallback: use alpha_score itself
            sig_cols = ["alpha_score"]

        ic_rows = []
        prices_idx = prices.index

        for sig_col in sig_cols:
            if sig_col not in hist.columns:
                continue

            ics = []
            for dt in dates[:-3]:   # need at least 20 trading days ahead
                sig_snap = hist[hist["date"] == dt][["ticker", sig_col]].copy()
                sig_snap[sig_col] = pd.to_numeric(sig_snap[sig_col], errors="coerce")
                sig_snap = sig_snap.dropna()
                if len(sig_snap) < 20:
                    continue

                # Forward return: 20 trading days ahead
                dt_idx = prices_idx.searchsorted(dt)
                fwd_idx = min(dt_idx + 21, len(prices_idx) - 1)
                if fwd_idx >= len(prices_idx):
                    continue

                dt_price   = prices.iloc[dt_idx]
                fwd_price  = prices.iloc[fwd_idx]
                fwd_rets   = (fwd_price / dt_price - 1.0).dropna()

                # Merge signal with forward returns
                merged = sig_snap.set_index("ticker")[sig_col]
                common = merged.index.intersection(fwd_rets.index)
                if len(common) < 20:
                    continue

                sig_vals = merged.reindex(common).values.astype(float)
                ret_vals = fwd_rets.reindex(common).values.astype(float)

                # Spearman rank IC
                from scipy import stats as _st
                ic, _ = _st.spearmanr(sig_vals, ret_vals)
                if not np.isnan(ic):
                    ics.append(float(ic))

            if ics:
                mean_ic   = float(np.mean(ics))
                ic_std    = float(np.std(ics)) + 1e-9
                icir      = mean_ic / ic_std    # IC information ratio
                n_pos     = sum(1 for x in ics if x > 0)
                # Decay: compare most recent 5 ICs vs earlier ones
                if len(ics) >= 8:
                    early_ic = float(np.mean(ics[:len(ics)//2]))
                    recent_ic = float(np.mean(ics[len(ics)//2:]))
                    decay_flag = int(recent_ic < early_ic - 0.03)
                else:
                    early_ic = recent_ic = mean_ic
                    decay_flag = 0

                ic_rows.append({
                    "signal":        sig_col.replace("sig_", ""),
                    "mean_ic":       round(mean_ic, 4),
                    "ic_std":        round(ic_std, 4),
                    "icir":          round(icir, 3),
                    "n_obs":         len(ics),
                    "pct_positive":  round(n_pos / len(ics), 3),
                    "early_ic":      round(early_ic, 4),
                    "recent_ic":     round(recent_ic, 4),
                    "decay_flag":    decay_flag,
                    "updated_date":  TODAY,
                })

        if ic_rows:
            ic_out = pd.DataFrame(ic_rows).sort_values("mean_ic", ascending=False)
            ic_out.to_csv(ROOT / "signal_ic_attribution.csv", index=False)
            top = ic_out.nlargest(3, "mean_ic")[["signal", "mean_ic", "icir"]].values.tolist()
            ok(f"signal_ic_attribution.csv → {len(ic_out)} signals | "
               f"top IC: {[(s, f'{ic:.3f}') for s,ic,_ in top]}")
        else:
            warn("L10: insufficient data for IC attribution — try again after more daily runs")

    except Exception as e:
        warn(f"L10: IC attribution error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import time as _time
    t0 = _time.time()
    print(f"\n{CYAN}{BOLD}Canyon v9 — Institutional Layer Upgrades  [{TODAY}]{RESET}\n")
    print("  Layers: L3 Sector · L4 Fundamentals · L5 Insider · L7 Options")
    print("          L8 Risk/Correlation · L9 Execution · L10 IC Attribution")
    print()

    # Load shared data once
    log("Loading price cache …")
    prices = _load_prices(tail=504)
    log("Loading volume cache …")
    volume = _load_volume(tail=126)

    if prices.empty:
        warn("No price data available — some upgrades will be skipped")
    else:
        ok(f"Price cache: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # ── L3: Sector momentum ──────────────────────────────────────────────────
    print(f"\n{BOLD}[L3] Sector Momentum{RESET}")
    upgrade_l3_sector_momentum(prices)

    # ── L4: Fundamentals ────────────────────────────────────────────────────
    print(f"\n{BOLD}[L4] Fundamentals: Accruals + Piotroski{RESET}")
    upgrade_l4_fundamentals()

    # ── L5: Insider clusters ─────────────────────────────────────────────────
    print(f"\n{BOLD}[L5] Insider Cluster Detection{RESET}")
    upgrade_l5_insider_clusters()

    # ── L6: Earnings call FinBERT sentiment (W22) ─────────────────────────────
    print(f"\n{BOLD}[L6] 8-K Earnings Call Sentiment (FinBERT){RESET}")
    upgrade_l6_earnings_sentiment()

    # ── L7: Options IV rank ───────────────────────────────────────────────────
    print(f"\n{BOLD}[L7] Options IV Rank{RESET}")
    upgrade_l7_options_ivrank()

    # ── L8: Correlation risk + CVaR ───────────────────────────────────────────
    print(f"\n{BOLD}[L8] Correlation Risk + CVaR{RESET}")
    upgrade_l8_correlation_risk(prices)

    # ── L9: Execution cost model ─────────────────────────────────────────────
    print(f"\n{BOLD}[L9] Slippage + Market Impact Model{RESET}")
    upgrade_l9_slippage_model(prices, volume)

    # ── L10: Signal IC attribution ────────────────────────────────────────────
    print(f"\n{BOLD}[L10] Signal IC Attribution{RESET}")
    upgrade_l10_ic_attribution(prices)

    elapsed = _time.time() - t0
    print(f"\n{GREEN}{BOLD}  Institutional upgrades complete — {elapsed:.0f}s{RESET}\n")
    print("  New outputs:")
    print("    data_health.csv              (L1 — written by step0)")
    print("    sector_momentum.csv          (L3)")
    print("    accrual_scores.csv           (L4)")
    print("    piotroski_scores.csv         (L4)")
    print("    insider_cluster_scores.csv   (L5)")
    print("    earnings_call_sentiment.csv  (L6)")
    print("    finbert_sentiment.csv        (L6, updated)")
    print("    options_ivrank.csv           (L7)")
    print("    correlation_risk.csv         (L8)")
    print("    position_risk_limits.csv     (L8)")
    print("    execution_cost_estimates.csv (L9)")
    print("    signal_ic_attribution.csv    (L10)")
    print()


if __name__ == "__main__":
    main()
