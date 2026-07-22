#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 9 — Portfolio Exposure Dashboard

Purpose:
Aggregates Step 7 execution_gate_review.csv / pre_trade_order_ticket.csv
Aggregates into a risk exposure panel, checking:
1. Tech/semiconductor concentration
2. ETF and single-stock overlap
3. Whether tactical book is too full
4. Whether any single position is oversized
5. Whether a real order draft actually exists

Principles:
- No market data download
- No fabricated market data
- Reads only locally generated review files, order drafts, and trade logs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


ROOT = Path.cwd()

REVIEW_FILE = ROOT / "execution_gate_review.csv"
ORDER_FILE = ROOT / "pre_trade_order_ticket.csv"
JOURNAL_FILE = ROOT / "canyon_trade_journal.csv"

OUT_MD = ROOT / "exposure_dashboard.md"
OUT_CSV = ROOT / "exposure_dashboard.csv"
OUT_WARNINGS = ROOT / "exposure_warnings.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ticker taxonomy
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY: Dict[str, Dict[str, str]] = {
    # broad ETFs
    "SPY":  {"asset_type": "ETF", "sector": "Broad Market", "theme": "US Beta", "risk_bucket": "BROAD_BETA"},
    "QQQ":  {"asset_type": "ETF", "sector": "Technology-heavy", "theme": "Mega-cap Growth", "risk_bucket": "TECH_GROWTH"},
    "DIA":  {"asset_type": "ETF", "sector": "Broad Market", "theme": "Dow", "risk_bucket": "BROAD_BETA"},
    "IWM":  {"asset_type": "ETF", "sector": "Small Caps", "theme": "Small-cap Beta", "risk_bucket": "SMALL_CAP"},
    "VTI":  {"asset_type": "ETF", "sector": "Broad Market", "theme": "US Total Market", "risk_bucket": "BROAD_BETA"},

    # sector ETFs
    "XLK":  {"asset_type": "ETF", "sector": "Technology", "theme": "Software/Hardware", "risk_bucket": "TECH_GROWTH"},
    "XLC":  {"asset_type": "ETF", "sector": "Communication Services", "theme": "Mega-cap Internet", "risk_bucket": "TECH_GROWTH"},
    "XLE":  {"asset_type": "ETF", "sector": "Energy", "theme": "Oil/Gas", "risk_bucket": "ENERGY"},
    "XLF":  {"asset_type": "ETF", "sector": "Financials", "theme": "Banks/Insurance", "risk_bucket": "FINANCIALS"},
    "XLV":  {"asset_type": "ETF", "sector": "Healthcare", "theme": "Healthcare Defensive", "risk_bucket": "HEALTHCARE"},
    "XLI":  {"asset_type": "ETF", "sector": "Industrials", "theme": "Industrials", "risk_bucket": "CYCLICALS"},
    "XLY":  {"asset_type": "ETF", "sector": "Consumer Discretionary", "theme": "Consumer Cyclical", "risk_bucket": "CYCLICALS"},
    "XLP":  {"asset_type": "ETF", "sector": "Consumer Staples", "theme": "Defensive Consumer", "risk_bucket": "DEFENSIVE"},
    "XLU":  {"asset_type": "ETF", "sector": "Utilities", "theme": "Defensive Yield", "risk_bucket": "DEFENSIVE"},
    "XLB":  {"asset_type": "ETF", "sector": "Materials", "theme": "Materials", "risk_bucket": "CYCLICALS"},
    "XLRE": {"asset_type": "ETF", "sector": "Real Estate", "theme": "Rates Sensitive", "risk_bucket": "RATES_SENSITIVE"},

    # semiconductor ETFs
    "SMH":  {"asset_type": "ETF", "sector": "Semiconductors", "theme": "AI/Semicap", "risk_bucket": "SEMICONDUCTOR"},
    "SOXX": {"asset_type": "ETF", "sector": "Semiconductors", "theme": "Semiconductor ETF", "risk_bucket": "SEMICONDUCTOR"},

    # macro hedges
    "GLD":  {"asset_type": "ETF", "sector": "Gold", "theme": "Gold Hedge", "risk_bucket": "HEDGE"},
    "IAU":  {"asset_type": "ETF", "sector": "Gold", "theme": "Gold Hedge", "risk_bucket": "HEDGE"},
    "TLT":  {"asset_type": "ETF", "sector": "Long Bonds", "theme": "Duration Hedge", "risk_bucket": "RATES_SENSITIVE"},
    "IEF":  {"asset_type": "ETF", "sector": "Treasury Bonds", "theme": "Intermediate Duration", "risk_bucket": "RATES_SENSITIVE"},
    "SHY":  {"asset_type": "ETF", "sector": "Treasury Bonds", "theme": "Short Duration", "risk_bucket": "CASH_LIKE"},
    "BIL":  {"asset_type": "ETF", "sector": "T-Bills", "theme": "Cash-like", "risk_bucket": "CASH_LIKE"},

    # mega-cap / tech / semis
    "AAPL": {"asset_type": "Stock", "sector": "Technology", "theme": "Mega-cap Tech", "risk_bucket": "TECH_GROWTH"},
    "MSFT": {"asset_type": "Stock", "sector": "Technology", "theme": "Mega-cap Tech", "risk_bucket": "TECH_GROWTH"},
    "NVDA": {"asset_type": "Stock", "sector": "Semiconductors", "theme": "AI/Semicap", "risk_bucket": "SEMICONDUCTOR"},
    "AMD":  {"asset_type": "Stock", "sector": "Semiconductors", "theme": "AI/Semicap", "risk_bucket": "SEMICONDUCTOR"},
    "MU":   {"asset_type": "Stock", "sector": "Semiconductors", "theme": "Memory/Semicap", "risk_bucket": "SEMICONDUCTOR"},
    "AVGO": {"asset_type": "Stock", "sector": "Semiconductors", "theme": "AI/Semicap", "risk_bucket": "SEMICONDUCTOR"},
    "TSM":  {"asset_type": "Stock", "sector": "Semiconductors", "theme": "Foundry/Semicap", "risk_bucket": "SEMICONDUCTOR"},
    "ASML": {"asset_type": "Stock", "sector": "Semicap Equipment", "theme": "Semicap Equipment", "risk_bucket": "SEMICONDUCTOR"},
    "GOOGL":{"asset_type": "Stock", "sector": "Communication Services", "theme": "Mega-cap Internet/AI", "risk_bucket": "TECH_GROWTH"},
    "GOOG": {"asset_type": "Stock", "sector": "Communication Services", "theme": "Mega-cap Internet/AI", "risk_bucket": "TECH_GROWTH"},
    "META": {"asset_type": "Stock", "sector": "Communication Services", "theme": "Mega-cap Internet/AI", "risk_bucket": "TECH_GROWTH"},
    "AMZN": {"asset_type": "Stock", "sector": "Consumer Discretionary", "theme": "Mega-cap Internet/Cloud", "risk_bucket": "TECH_GROWTH"},
    "TSLA": {"asset_type": "Stock", "sector": "Consumer Discretionary", "theme": "High Beta Growth", "risk_bucket": "HIGH_BETA"},
}


ETF_COMPONENT_OVERLAPS: Dict[str, List[str]] = {
    # approximate overlap map, used for risk warning only, not exact holdings
    "SMH":  ["NVDA", "AMD", "MU", "AVGO", "TSM", "ASML"],
    "SOXX": ["NVDA", "AMD", "MU", "AVGO", "TSM", "ASML"],
    "XLK":  ["AAPL", "MSFT", "NVDA", "AVGO", "AMD"],
    "QQQ":  ["AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA", "AVGO", "AMD"],
    "XLC":  ["GOOGL", "GOOG", "META"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Utility
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return pd.DataFrame()


def clean_weight(x) -> float:
    """Accept 0.025, '2.5%', '2.5', etc. Return decimal weight."""
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        v = float(s)
        # if someone wrote 2.5 instead of 0.025, treat it as percent
        if abs(v) > 1.0:
            return v / 100.0
        return v
    except Exception:
        return 0.0


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def taxonomy_for(ticker: str) -> Dict[str, str]:
    t = str(ticker).upper().strip()
    return TAXONOMY.get(t, {
        "asset_type": "Unknown",
        "sector": "Unknown",
        "theme": "Unclassified",
        "risk_bucket": "UNKNOWN",
    })


def normalize_review_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    ticker_col = find_col(df, ["ticker", "symbol"])
    sleeve_col = find_col(df, ["sleeve", "account"])
    planned_col = find_col(df, ["planned_weight", "planned", "weight"])
    approved_col = find_col(df, ["approved_weight", "approved_max", "approved"])
    decision_col = find_col(df, ["decision", "evidence_decision"])
    status_col = find_col(df, ["gate_status", "status"])
    intent_col = find_col(df, ["order_intent", "intent"])

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].astype(str).str.upper().str.strip() if ticker_col else ""
    out["sleeve"] = df[sleeve_col].astype(str).str.upper().str.strip() if sleeve_col else "UNKNOWN"
    out["planned_weight"] = df[planned_col].apply(clean_weight) if planned_col else 0.0
    out["approved_weight"] = df[approved_col].apply(clean_weight) if approved_col else out["planned_weight"]
    out["decision"] = df[decision_col].astype(str) if decision_col else "UNKNOWN"
    out["gate_status"] = df[status_col].astype(str) if status_col else "UNKNOWN"
    out["order_intent"] = df[intent_col].astype(str) if intent_col else "NONE"

    # choose the lower of planned/approved as effective review exposure
    out["effective_weight"] = out[["planned_weight", "approved_weight"]].min(axis=1)
    out.loc[out["effective_weight"] < 0, "effective_weight"] = 0.0

    # add taxonomy
    tax = out["ticker"].apply(taxonomy_for).apply(pd.Series)
    out = pd.concat([out, tax], axis=1)

    return out


def normalize_order_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    ticker_col = find_col(df, ["ticker", "symbol"])
    weight_col = find_col(df, ["approved_weight", "weight", "target_weight", "order_weight"])
    intent_col = find_col(df, ["order_intent", "intent"])
    status_col = find_col(df, ["status", "order_status"])

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].astype(str).str.upper().str.strip() if ticker_col else ""
    out["order_weight"] = df[weight_col].apply(clean_weight) if weight_col else 0.0
    out["order_intent"] = df[intent_col].astype(str) if intent_col else "UNKNOWN"
    out["order_status"] = df[status_col].astype(str) if status_col else "UNKNOWN"
    tax = out["ticker"].apply(taxonomy_for).apply(pd.Series)
    out = pd.concat([out, tax], axis=1)
    return out


def pct(x: float) -> str:
    return f"{x:.1%}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Exposure analysis
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_exposures(review: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if review.empty:
        return {
            "by_ticker": pd.DataFrame(),
            "by_sleeve": pd.DataFrame(),
            "by_bucket": pd.DataFrame(),
            "by_sector": pd.DataFrame(),
        }

    by_ticker = (
        review.groupby(["ticker", "asset_type", "sector", "theme", "risk_bucket"], dropna=False)
        ["effective_weight"].sum()
        .reset_index()
        .sort_values("effective_weight", ascending=False)
    )

    by_sleeve = (
        review.groupby("sleeve", dropna=False)["effective_weight"].sum()
        .reset_index()
        .sort_values("effective_weight", ascending=False)
    )

    by_bucket = (
        review.groupby("risk_bucket", dropna=False)["effective_weight"].sum()
        .reset_index()
        .sort_values("effective_weight", ascending=False)
    )

    by_sector = (
        review.groupby("sector", dropna=False)["effective_weight"].sum()
        .reset_index()
        .sort_values("effective_weight", ascending=False)
    )

    return {
        "by_ticker": by_ticker,
        "by_sleeve": by_sleeve,
        "by_bucket": by_bucket,
        "by_sector": by_sector,
    }


def build_warnings(review: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    warnings = []

    if review.empty:
        return pd.DataFrame([{
            "level": "ERROR",
            "issue": "No execution_gate_review.csv found or file is empty",
            "detail": "Run Step 7 first.",
            "action": "Do not trade until review file exists."
        }])

    total_eff = float(review["effective_weight"].sum())
    bucket_exp = review.groupby("risk_bucket")["effective_weight"].sum().to_dict()
    sleeve_exp = review.groupby("sleeve")["effective_weight"].sum().to_dict()
    ticker_exp = review.groupby("ticker")["effective_weight"].sum().to_dict()

    semi = bucket_exp.get("SEMICONDUCTOR", 0.0)
    tech = bucket_exp.get("TECH_GROWTH", 0.0)
    high_beta = bucket_exp.get("HIGH_BETA", 0.0)

    if semi > 0.30:
        warnings.append({
            "level": "HIGH",
            "issue": "Semiconductor concentration",
            "detail": f"SEMICONDUCTOR exposure is {pct(semi)}, above 30%.",
            "action": "Reduce duplicate SMH/SOXX/AMD/MU/NVDA exposure before any new tactical trade."
        })
    elif semi > 0.20:
        warnings.append({
            "level": "MEDIUM",
            "issue": "Semiconductor concentration",
            "detail": f"SEMICONDUCTOR exposure is {pct(semi)}.",
            "action": "Avoid adding more semis unless evidence is exceptional."
        })

    if tech + semi > 0.45:
        warnings.append({
            "level": "HIGH",
            "issue": "Tech-growth cluster too large",
            "detail": f"TECH_GROWTH + SEMICONDUCTOR exposure is {pct(tech + semi)}, above 45%.",
            "action": "Add diversification or reduce tactical tech positions."
        })
    elif tech + semi > 0.35:
        warnings.append({
            "level": "MEDIUM",
            "issue": "Tech-growth cluster elevated",
            "detail": f"TECH_GROWTH + SEMICONDUCTOR exposure is {pct(tech + semi)}.",
            "action": "Do not let one AI/tech theme control account-level risk."
        })

    tactical = sleeve_exp.get("TACTICAL", 0.0)
    if tactical > 0.25:
        warnings.append({
            "level": "MEDIUM",
            "issue": "Tactical sleeve near/above target",
            "detail": f"TACTICAL effective exposure is {pct(tactical)}.",
            "action": "Keep single-name tactical sizing small; do not convert all candidates into trades."
        })

    for tk, w in ticker_exp.items():
        if w > 0.10:
            warnings.append({
                "level": "HIGH",
                "issue": "Single ticker concentration",
                "detail": f"{tk} total effective exposure is {pct(w)}.",
                "action": "Review whether this is ETF+stock duplication or intended core exposure."
            })
        elif w > 0.06 and tk not in ["SPY", "QQQ", "GLD", "TLT", "SHY", "BIL"]:
            warnings.append({
                "level": "MEDIUM",
                "issue": "Single ticker concentration",
                "detail": f"{tk} exposure is {pct(w)}.",
                "action": "Require stronger thesis and stop plan before increasing."
            })

    # ETF and component overlap warning
    tickers = set(ticker_exp.keys())
    for etf, comps in ETF_COMPONENT_OVERLAPS.items():
        if etf in tickers:
            present = [c for c in comps if c in tickers]
            if present:
                detail_w = ticker_exp.get(etf, 0.0) + sum(ticker_exp.get(c, 0.0) for c in present)
                warnings.append({
                    "level": "MEDIUM",
                    "issue": "ETF + component overlap",
                    "detail": f"{etf} overlaps with {', '.join(present)}; combined approximate related exposure {pct(detail_w)}.",
                    "action": "Prefer either ETF or top component; avoid paying for the same risk twice."
                })

    # Order-ticket status
    if orders.empty:
        warnings.append({
            "level": "INFO",
            "issue": "No pre-trade order ticket",
            "detail": "No order drafts were generated.",
            "action": "This is normal if manual checks are not all YES or order_intent is not PAPER/LIVE."
        })
    else:
        live = orders["order_intent"].astype(str).str.upper().isin(["LIVE"]).sum()
        paper = orders["order_intent"].astype(str).str.upper().isin(["PAPER"]).sum()
        warnings.append({
            "level": "INFO",
            "issue": "Order drafts exist",
            "detail": f"{len(orders)} draft rows found: PAPER={paper}, LIVE={live}.",
            "action": "Verify LIMIT_ONLY and all manual checks before sending anything."
        })

    if not warnings:
        warnings.append({
            "level": "OK",
            "issue": "No major concentration warning",
            "detail": f"Total effective reviewed exposure is {pct(total_eff)}.",
            "action": "Still complete manual news/earnings/liquidity checks before any trade."
        })

    return pd.DataFrame(warnings)


def to_md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data._"
    d = df.copy().head(max_rows)
    for col in d.columns:
        if "weight" in col.lower() or col.lower() in ["effective_weight", "planned_weight", "approved_weight", "order_weight"]:
            d[col] = d[col].apply(lambda x: pct(float(x)) if pd.notna(x) else "")
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def build_report(review: pd.DataFrame, orders: pd.DataFrame, aggs: Dict[str, pd.DataFrame], warnings: pd.DataFrame) -> str:
    lines = []
    lines.append("# Canyon v9 Step 9 — Portfolio Exposure Dashboard")
    lines.append("")
    lines.append("## Data Integrity")
    lines.append("")
    lines.append("- Source files are local CSVs generated by previous Canyon steps.")
    lines.append("- This report does not download prices and does not create synthetic market data.")
    lines.append(f"- execution_gate_review.csv: {'FOUND' if REVIEW_FILE.exists() else 'MISSING'}")
    lines.append(f"- pre_trade_order_ticket.csv: {'FOUND' if ORDER_FILE.exists() else 'MISSING'}")
    lines.append(f"- canyon_trade_journal.csv: {'FOUND' if JOURNAL_FILE.exists() else 'MISSING'}")
    lines.append("")

    lines.append("## Exposure Summary")
    lines.append("")
    if review.empty:
        lines.append("_No review data found. Run Step 7 first._")
    else:
        total_eff = review["effective_weight"].sum()
        lines.append(f"- Total reviewed effective exposure: **{pct(float(total_eff))}**")
        lines.append(f"- Number of reviewed candidates: **{len(review)}**")
        lines.append("")

    lines.append("### By Risk Bucket")
    lines.append(to_md_table(aggs.get("by_bucket", pd.DataFrame())))
    lines.append("")

    lines.append("### By Sector")
    lines.append(to_md_table(aggs.get("by_sector", pd.DataFrame())))
    lines.append("")

    lines.append("### By Sleeve")
    lines.append(to_md_table(aggs.get("by_sleeve", pd.DataFrame())))
    lines.append("")

    lines.append("### By Ticker")
    lines.append(to_md_table(aggs.get("by_ticker", pd.DataFrame()), max_rows=30))
    lines.append("")

    lines.append("## Concentration / Overlap Warnings")
    lines.append(to_md_table(warnings, max_rows=50))
    lines.append("")

    lines.append("## Order Draft Status")
    lines.append("")
    if orders.empty:
        lines.append("- No order drafts found. This is expected unless manual checks are set to YES and order_intent is PAPER/LIVE.")
    else:
        lines.append(to_md_table(orders, max_rows=30))
    lines.append("")

    lines.append("## PM Rules")
    lines.append("")
    lines.append("- WATCHLIST is not an order.")
    lines.append("- EvidenceCard ALLOW means it can enter review, not that it should be bought.")
    lines.append("- REVIEW_OR_REDUCE should default to reduced size or skip.")
    lines.append("- If ETF and component stock overlap, prefer one expression of the same risk.")
    lines.append("- Do not let semiconductor/AI theme become the whole account.")
    lines.append("- Use LIMIT_ONLY by default; do not use market orders for momentum chase.")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 86)
    print("🏔 CANYON v9 Step 9")
    print("Portfolio Exposure Dashboard × Concentration Guard")
    print("=" * 86)

    review_raw = read_csv_if_exists(REVIEW_FILE)
    order_raw = read_csv_if_exists(ORDER_FILE)

    review = normalize_review_df(review_raw)
    orders = normalize_order_df(order_raw)

    aggs = aggregate_exposures(review)
    warnings = build_warnings(review, orders)

    # Save combined exposure table
    by_ticker = aggs.get("by_ticker", pd.DataFrame())
    if not by_ticker.empty:
        by_ticker.to_csv(OUT_CSV, index=False)
    else:
        pd.DataFrame().to_csv(OUT_CSV, index=False)

    warnings.to_csv(OUT_WARNINGS, index=False)

    report = build_report(review, orders, aggs, warnings)
    OUT_MD.write_text(report, encoding="utf-8")

    print("\nExposure summary:")
    if review.empty:
        print("  No execution_gate_review.csv data found.")
    else:
        print(f"  Reviewed candidates: {len(review)}")
        print(f"  Total effective exposure: {review['effective_weight'].sum():.1%}")

    print("\nTop risk buckets:")
    if not aggs["by_bucket"].empty:
        for _, row in aggs["by_bucket"].head(8).iterrows():
            print(f"  {row['risk_bucket']:<18} {row['effective_weight']:.1%}")

    print("\nWarnings:")
    for _, row in warnings.head(12).iterrows():
        print(f"  [{row['level']}] {row['issue']}: {row['detail']}")

    print("\nFiles generated:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_WARNINGS}")
    print("\nNext Step 10: build Scenario Stress Test + position sizing from the exposure dashboard.")


if __name__ == "__main__":
    main()
