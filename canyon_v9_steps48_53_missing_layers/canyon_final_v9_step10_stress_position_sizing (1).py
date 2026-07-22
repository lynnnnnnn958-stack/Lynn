#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 10 — Scenario Stress Test + Position Sizing

Purpose:
Based on local review tables from Step 7/9, run portfolio stress tests and position scaling recommendations.

It answers:
1. If semiconductors drop 5%, roughly how much does the portfolio lose?
2. If Nasdaq/tech pulls back, which sleeve carries the most risk?
3. Which candidates should be halved, skipped, or kept at REVIEW?
4. What should the final suggested_weight be?

Principles:
- No automatic order placement
- No market data downloads
- No fabricated prices
- Uses only local execution_gate_review.csv and rule-based scenario shocks
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


ROOT = Path.cwd()

REVIEW_FILE = ROOT / "execution_gate_review.csv"
OUT_MD = ROOT / "stress_position_sizing_report.md"
OUT_SCENARIO = ROOT / "scenario_stress_results.csv"
OUT_SIZING = ROOT / "position_sizing_recommendations.csv"


TAXONOMY: Dict[str, Dict[str, str]] = {
    "SPY":  {"asset_type": "ETF", "sector": "Broad Market", "theme": "US Beta", "risk_bucket": "BROAD_BETA"},
    "QQQ":  {"asset_type": "ETF", "sector": "Technology-heavy", "theme": "Mega-cap Growth", "risk_bucket": "TECH_GROWTH"},
    "DIA":  {"asset_type": "ETF", "sector": "Broad Market", "theme": "Dow", "risk_bucket": "BROAD_BETA"},
    "IWM":  {"asset_type": "ETF", "sector": "Small Caps", "theme": "Small-cap Beta", "risk_bucket": "SMALL_CAP"},

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

    "SMH":  {"asset_type": "ETF", "sector": "Semiconductors", "theme": "AI/Semicap", "risk_bucket": "SEMICONDUCTOR"},
    "SOXX": {"asset_type": "ETF", "sector": "Semiconductors", "theme": "Semiconductor ETF", "risk_bucket": "SEMICONDUCTOR"},

    "GLD":  {"asset_type": "ETF", "sector": "Gold", "theme": "Gold Hedge", "risk_bucket": "HEDGE"},
    "IAU":  {"asset_type": "ETF", "sector": "Gold", "theme": "Gold Hedge", "risk_bucket": "HEDGE"},
    "TLT":  {"asset_type": "ETF", "sector": "Long Bonds", "theme": "Duration Hedge", "risk_bucket": "RATES_SENSITIVE"},
    "IEF":  {"asset_type": "ETF", "sector": "Treasury Bonds", "theme": "Intermediate Duration", "risk_bucket": "RATES_SENSITIVE"},
    "SHY":  {"asset_type": "ETF", "sector": "Treasury Bonds", "theme": "Short Duration", "risk_bucket": "CASH_LIKE"},
    "BIL":  {"asset_type": "ETF", "sector": "T-Bills", "theme": "Cash-like", "risk_bucket": "CASH_LIKE"},

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
    "SMH":  ["NVDA", "AMD", "MU", "AVGO", "TSM", "ASML"],
    "SOXX": ["NVDA", "AMD", "MU", "AVGO", "TSM", "ASML"],
    "XLK":  ["AAPL", "MSFT", "NVDA", "AVGO", "AMD"],
    "QQQ":  ["AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA", "AVGO", "AMD"],
    "XLC":  ["GOOGL", "GOOG", "META"],
}


SCENARIOS: Dict[str, Dict[str, float]] = {
    "SOFT_PULLBACK": {
        "BROAD_BETA": -0.015, "TECH_GROWTH": -0.025, "SEMICONDUCTOR": -0.035,
        "ENERGY": -0.010, "FINANCIALS": -0.012, "HEALTHCARE": -0.006,
        "DEFENSIVE": -0.004, "HEDGE": 0.004, "RATES_SENSITIVE": 0.002,
        "CASH_LIKE": 0.000, "UNKNOWN": -0.010,
    },
    "SEMIS_MINUS_5": {
        "BROAD_BETA": -0.010, "TECH_GROWTH": -0.025, "SEMICONDUCTOR": -0.050,
        "ENERGY": 0.000, "FINANCIALS": -0.004, "HEALTHCARE": -0.003,
        "DEFENSIVE": -0.002, "HEDGE": 0.002, "RATES_SENSITIVE": 0.000,
        "CASH_LIKE": 0.000, "UNKNOWN": -0.010,
    },
    "NASDAQ_RISK_OFF": {
        "BROAD_BETA": -0.025, "TECH_GROWTH": -0.045, "SEMICONDUCTOR": -0.060,
        "ENERGY": -0.015, "FINANCIALS": -0.020, "HEALTHCARE": -0.010,
        "DEFENSIVE": -0.008, "HEDGE": 0.008, "RATES_SENSITIVE": 0.004,
        "CASH_LIKE": 0.000, "UNKNOWN": -0.020,
    },
    "RATES_UP": {
        "BROAD_BETA": -0.012, "TECH_GROWTH": -0.030, "SEMICONDUCTOR": -0.035,
        "ENERGY": 0.006, "FINANCIALS": 0.004, "HEALTHCARE": -0.006,
        "DEFENSIVE": -0.010, "HEDGE": -0.006, "RATES_SENSITIVE": -0.035,
        "CASH_LIKE": 0.000, "UNKNOWN": -0.010,
    },
    "ENERGY_REVERSAL": {
        "BROAD_BETA": -0.006, "TECH_GROWTH": 0.000, "SEMICONDUCTOR": -0.005,
        "ENERGY": -0.050, "FINANCIALS": -0.006, "HEALTHCARE": 0.000,
        "DEFENSIVE": 0.000, "HEDGE": 0.003, "RATES_SENSITIVE": 0.002,
        "CASH_LIKE": 0.000, "UNKNOWN": -0.006,
    },
    "BROAD_MARKET_MINUS_3": {
        "BROAD_BETA": -0.030, "TECH_GROWTH": -0.040, "SEMICONDUCTOR": -0.055,
        "ENERGY": -0.030, "FINANCIALS": -0.032, "HEALTHCARE": -0.018,
        "DEFENSIVE": -0.015, "HEDGE": 0.010, "RATES_SENSITIVE": 0.004,
        "CASH_LIKE": 0.000, "UNKNOWN": -0.030,
    },
}


def pct(x: float) -> str:
    return f"{x:.2%}"


def clean_weight(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        v = float(s)
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()


def normalize_review(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

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
    out["decision"] = df[decision_col].astype(str).str.upper() if decision_col else "UNKNOWN"
    out["gate_status"] = df[status_col].astype(str).str.upper() if status_col else "UNKNOWN"
    out["order_intent"] = df[intent_col].astype(str).str.upper() if intent_col else "NONE"
    out["effective_weight"] = out[["planned_weight", "approved_weight"]].min(axis=1).clip(lower=0)

    tax = out["ticker"].apply(taxonomy_for).apply(pd.Series)
    out = pd.concat([out, tax], axis=1)
    return out


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data._"
    d = df.copy().head(max_rows)
    for col in d.columns:
        if "weight" in col.lower() or "loss" in col.lower() or "pnl" in col.lower() or "shock" in col.lower() or "reduction" in col.lower():
            def fmt(v):
                try:
                    return pct(float(v))
                except Exception:
                    return str(v)
            d[col] = d[col].apply(fmt)
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def scenario_stress(review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if review.empty:
        return pd.DataFrame()

    for scenario_name, shocks in SCENARIOS.items():
        pnl_by_row = []
        for _, r in review.iterrows():
            bucket = str(r.get("risk_bucket", "UNKNOWN"))
            w = float(r.get("effective_weight", 0.0))
            shock = float(shocks.get(bucket, shocks.get("UNKNOWN", -0.01)))
            pnl_by_row.append(w * shock)

        total_pnl = float(np.sum(pnl_by_row))
        rows.append({
            "scenario": scenario_name,
            "estimated_pnl": total_pnl,
            "estimated_loss": min(total_pnl, 0.0),
            "breaches_1pct": total_pnl < -0.01,
            "breaches_2pct": total_pnl < -0.02,
            "breaches_5pct": total_pnl < -0.05,
        })

    return pd.DataFrame(rows).sort_values("estimated_pnl")


def scenario_by_sleeve(review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if review.empty:
        return pd.DataFrame()

    for scenario_name, shocks in SCENARIOS.items():
        temp = review.copy()
        temp["shock"] = temp["risk_bucket"].map(lambda b: shocks.get(b, shocks.get("UNKNOWN", -0.01)))
        temp["estimated_pnl"] = temp["effective_weight"] * temp["shock"]
        by = temp.groupby("sleeve")["estimated_pnl"].sum().reset_index()
        by["scenario"] = scenario_name
        rows.append(by)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[["scenario", "sleeve", "estimated_pnl"]]


def build_sizing(review: pd.DataFrame) -> pd.DataFrame:
    if review.empty:
        return pd.DataFrame()

    out = review.copy()

    bucket_exp = out.groupby("risk_bucket")["effective_weight"].sum().to_dict()
    tech_plus_semi = bucket_exp.get("TECH_GROWTH", 0.0) + bucket_exp.get("SEMICONDUCTOR", 0.0)
    semi_exp = bucket_exp.get("SEMICONDUCTOR", 0.0)

    tickers = set(out["ticker"].astype(str))
    overlap_tickers = set()
    for etf, comps in ETF_COMPONENT_OVERLAPS.items():
        if etf in tickers:
            present = [c for c in comps if c in tickers]
            if present:
                overlap_tickers.add(etf)
                overlap_tickers.update(present)

    suggested = []
    actions = []
    reasons = []

    for _, r in out.iterrows():
        ticker = str(r["ticker"])
        sleeve = str(r["sleeve"])
        decision = str(r["decision"]).upper()
        bucket = str(r["risk_bucket"])
        cur = float(r["effective_weight"])

        cap = cur
        action = "KEEP_REVIEW"
        reason = []

        if "REVIEW_OR_REDUCE" in decision:
            cap = min(cap, cur * 0.50, 0.0125)
            action = "REDUCE_OR_SKIP"
            reason.append("decision=REVIEW_OR_REDUCE")
        elif "REVIEW" in decision and "ALLOW" not in decision:
            cap = min(cap, 0.025)
            action = "REVIEW_ONLY"
            reason.append("decision=REVIEW")
        elif "ALLOW" in decision:
            if sleeve == "TACTICAL":
                cap = min(cap, 0.025)
                action = "ALLOW_AFTER_MANUAL_CHECKS"
                reason.append("TACTICAL cap 2.5%")
            else:
                cap = min(cap, 0.10)
                action = "ALLOW_AFTER_MANUAL_CHECKS"
                reason.append("non-tactical cap")

        if bucket == "SEMICONDUCTOR" and semi_exp > 0.20:
            cap *= 0.75
            reason.append(f"semi exposure elevated {semi_exp:.1%}")
            if action.startswith("ALLOW"):
                action = "ALLOW_REDUCED"

        if bucket in ["TECH_GROWTH", "SEMICONDUCTOR"] and tech_plus_semi > 0.35:
            if sleeve == "TACTICAL":
                cap *= 0.50
                reason.append(f"tech+semi cluster elevated {tech_plus_semi:.1%}; tactical reduced")
                action = "ALLOW_REDUCED" if "ALLOW" in action else action

        if ticker in overlap_tickers:
            cap *= 0.75
            reason.append("ETF/component overlap")
            if "ALLOW" in action:
                action = "ALLOW_REDUCED"

        if bucket == "UNKNOWN":
            cap = min(cap, 0.01)
            action = "REVIEW_ONLY"
            reason.append("unknown taxonomy")

        cap = max(0.0, round(float(cap), 4))

        if cap < 0.005 and cur > 0:
            action = "SKIP_UNLESS_EXCEPTIONAL"
            reason.append("suggested weight below 0.5%")

        suggested.append(cap)
        actions.append(action)
        reasons.append("; ".join(reason) if reason else "no additional sizing reduction")

    out["suggested_weight"] = suggested
    out["suggested_action"] = actions
    out["sizing_reason"] = reasons
    out["reduction_from_effective"] = out["effective_weight"] - out["suggested_weight"]

    cols = [
        "ticker", "sleeve", "decision", "risk_bucket", "sector",
        "planned_weight", "approved_weight", "effective_weight",
        "suggested_weight", "reduction_from_effective",
        "suggested_action", "sizing_reason"
    ]
    return out[cols].sort_values(["suggested_action", "suggested_weight"], ascending=[True, False])


def build_report(review: pd.DataFrame, stress: pd.DataFrame, sleeve_stress: pd.DataFrame, sizing: pd.DataFrame) -> str:
    lines = []
    lines.append("# Canyon v9 Step 10 — Scenario Stress Test + Position Sizing")
    lines.append("")
    lines.append("## Data Rules")
    lines.append("")
    lines.append("- This report uses local review files only.")
    lines.append("- It does not download prices.")
    lines.append("- Scenario shocks are rule-based stress assumptions, not forecasts.")
    lines.append("- Suggested weights are review recommendations, not orders.")
    lines.append("")

    if review.empty:
        lines.append("## Missing Data")
        lines.append("")
        lines.append("execution_gate_review.csv was not found or was empty. Run Step 7 first.")
        return "\n".join(lines)

    total = float(review["effective_weight"].sum())
    lines.append("## Current Reviewed Exposure")
    lines.append("")
    lines.append(f"- Total effective reviewed exposure: **{pct(total)}**")
    lines.append(f"- Candidate count: **{len(review)}**")
    lines.append("")

    lines.append("## Scenario Stress Results")
    lines.append("")
    lines.append(md_table(stress, max_rows=20))
    lines.append("")

    if not stress.empty:
        worst = stress.iloc[0]
        lines.append("### Worst Scenario")
        lines.append("")
        lines.append(f"- Worst scenario: **{worst['scenario']}**")
        lines.append(f"- Estimated portfolio impact: **{pct(float(worst['estimated_pnl']))}**")
        if float(worst["estimated_pnl"]) < -0.02:
            lines.append("- Interpretation: this candidate book is too sensitive for direct conversion into trades.")
        elif float(worst["estimated_pnl"]) < -0.01:
            lines.append("- Interpretation: manageable, but tactical sizing should stay reduced.")
        else:
            lines.append("- Interpretation: stress impact is relatively contained, subject to manual checks.")
        lines.append("")

    lines.append("## Scenario Impact by Sleeve")
    lines.append("")
    lines.append(md_table(sleeve_stress.sort_values(["scenario", "estimated_pnl"]), max_rows=60))
    lines.append("")

    lines.append("## Position Sizing Recommendations")
    lines.append("")
    lines.append(md_table(sizing, max_rows=60))
    lines.append("")

    lines.append("## Execution Discipline")
    lines.append("")
    lines.append("- Do not trade directly from this file.")
    lines.append("- Any TACTICAL name still requires news, earnings date, pre/after-market move, bid-ask, and liquidity checks.")
    lines.append("- REVIEW_OR_REDUCE defaults to half-size or skip.")
    lines.append("- If scenario loss exceeds 2%, reduce tactical and overlapping theme exposures first.")
    lines.append("- Prefer one expression of the same thesis: ETF or single stock, not both at full size.")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 10")
    print("Scenario Stress Test × Position Sizing")
    print("=" * 88)

    raw = read_csv(REVIEW_FILE)
    review = normalize_review(raw)

    if review.empty:
        print("No execution_gate_review.csv found. Run Step 7 first.")
        OUT_MD.write_text(build_report(review, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), encoding="utf-8")
        return

    stress = scenario_stress(review)
    sleeve_stress = scenario_by_sleeve(review)
    sizing = build_sizing(review)

    stress.to_csv(OUT_SCENARIO, index=False)
    sizing.to_csv(OUT_SIZING, index=False)

    report = build_report(review, stress, sleeve_stress, sizing)
    OUT_MD.write_text(report, encoding="utf-8")

    print("\nWorst scenarios:")
    for _, row in stress.head(6).iterrows():
        print(f"  {row['scenario']:<22} estimated_pnl={row['estimated_pnl']:+.2%}")

    print("\nTop sizing recommendations:")
    for _, row in sizing.head(12).iterrows():
        print(
            f"  {row['ticker']:<6} {row['sleeve']:<16} "
            f"eff={row['effective_weight']:.1%} -> suggested={row['suggested_weight']:.1%} "
            f"{row['suggested_action']}"
        )

    print("\nFiles generated:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_SCENARIO}")
    print(f"  {OUT_SIZING}")

    print("\nNext Step 11: create a lightweight dashboard so you can view PM report, exposures, stress tests, and journal in one page.")


if __name__ == "__main__":
    main()
