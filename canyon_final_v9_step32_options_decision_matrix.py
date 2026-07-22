#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 32 — Options Decision Matrix

Merges the Gamma table from Step 23/25 + the Kill Zone table from Step 24 + the Pre-trade table from Step 20
into a single “options decision matrix” that a trading desk can actually read.

It answers:
- Is this ticker a gamma squeeze watch or no setup?
- Is it likely to be pinned / IV crushed / theta bled?
- Current recommendation: skip options, paper stock only, wait for breakout confirmation, or continue paper review?
- Is LIVE allowed? Default NO.

No orders. No broker connection. No real buy/sell instructions.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()

GAMMA = ROOT / "gamma_squeeze_candidates.csv"
KILL = ROOT / "option_kill_zone_risk.csv"
PRE = ROOT / "pre_trade_checklist.csv"

OUT_CSV = ROOT / "options_decision_matrix.csv"
OUT_MD = ROOT / "options_decision_matrix.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(x, default=np.nan):
    try:
        s = str(x).replace("%", "").replace(",", "").strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def risk_from_pretrade(pre: pd.DataFrame, ticker: str) -> dict:
    if pre.empty or "ticker" not in pre.columns:
        return {
            "pretrade_status": "",
            "portfolio_risk_light": "",
            "paper_allowed": "",
            "live_allowed": "NO",
            "pretrade_reasons": "",
        }
    p = pre[pre["ticker"].astype(str).str.upper().str.strip() == ticker]
    if p.empty:
        return {
            "pretrade_status": "",
            "portfolio_risk_light": "",
            "paper_allowed": "",
            "live_allowed": "NO",
            "pretrade_reasons": "",
        }
    r = p.iloc[0]
    return {
        "pretrade_status": str(r.get("final_status", "")),
        "portfolio_risk_light": str(r.get("risk_light", "")),
        "paper_allowed": str(r.get("paper_allowed", "")),
        "live_allowed": str(r.get("live_allowed", "NO")) or "NO",
        "pretrade_reasons": str(r.get("reasons", "")),
    }


def label_level(label: str) -> str:
    s = str(label).upper()
    if "HIGH" in s:
        return "HIGH"
    if "MEDIUM" in s:
        return "MEDIUM"
    if "LOW" in s:
        return "LOW"
    return "NONE"


def decision_rule(row: dict) -> tuple[str, str, str]:
    gamma_level = label_level(row.get("gamma_squeeze_label", ""))
    kill_level = label_level(row.get("option_kill_zone_label", ""))
    risk = str(row.get("portfolio_risk_light", "")).upper()
    confidence = str(row.get("data_confidence", "")).upper()
    pre_status = str(row.get("pretrade_status", "")).upper()

    notes = []

    if "LOW_CONFIDENCE" in confidence:
        notes.append("Options data confidence is low; results should be treated as observation only.")

    if "BLOCKED" in pre_status:
        return "SKIP", "BLOCKED_BY_PRETRADE", "Blocked by pre-trade check."

    if "ALREADY_CLOSED" in pre_status:
        return "SKIP", "ALREADY_CLOSED", "Already paper closed; do not duplicate sample."

    if kill_level == "HIGH":
        notes.append("Kill Zone is high; short-dated options are prone to pin/IV crush/theta bleed.")
        if gamma_level in {"HIGH", "MEDIUM"}:
            return "WAIT", "WAIT_BREAKOUT_CONFIRMATION", "Gamma has momentum, but Kill Zone is high; wait for price to break the call wall with confirmed volume."
        return "SKIP_OPTIONS", "AVOID_SHORT_DATED_OPTIONS", "Not suitable for buying weekly/short-dated options; at most observe the underlying stock."

    if risk == "RED":
        notes.append("Portfolio risk light is RED; live trading not permitted.")
        if gamma_level in {"HIGH", "MEDIUM"} and kill_level in {"LOW", "NONE", "MEDIUM"}:
            return "PAPER_ONLY", "STOCK_OR_ETF_PAPER_ONLY", "Only very small paper review allowed; prefer stock/ETF, do not chase short-dated options."
        return "RESEARCH_ONLY", "NO_LIVE_NO_OPTIONS", "Risk light is RED and gamma edge is insufficient; research first, no trading."

    if gamma_level == "HIGH" and kill_level in {"LOW", "NONE"}:
        return "PAPER_REVIEW", "GAMMA_WATCH_PAPER_ONLY", "Can continue paper review, but still requires news, earnings, liquidity, and spread checks."

    if gamma_level == "MEDIUM" and kill_level in {"LOW", "NONE", "MEDIUM"}:
        return "WATCH", "WATCHLIST_WAIT_TRIGGER", "Add to watchlist; wait for call wall breakout or volume/price confirmation."

    if gamma_level in {"LOW", "NONE"} and kill_level in {"MEDIUM", "HIGH"}:
        return "SKIP_OPTIONS", "NO_GAMMA_PLUS_KILL_RISK", "No clear gamma edge and option decay risk is present."

    if gamma_level in {"LOW", "NONE"}:
        return "RESEARCH_ONLY", "NO_CLEAR_OPTIONS_EDGE", "No clear options edge; revert to stock/ETF research."

    return "WATCH", "MANUAL_REVIEW", "; ".join(notes) if notes else "Manual review required."


def build_matrix() -> pd.DataFrame:
    gamma = read_csv(GAMMA)
    kill = read_csv(KILL)
    pre = read_csv(PRE)

    tickers = set()
    if not gamma.empty and "ticker" in gamma.columns:
        tickers.update(gamma["ticker"].astype(str).str.upper().str.strip().tolist())
    if not kill.empty and "ticker" in kill.columns:
        tickers.update(kill["ticker"].astype(str).str.upper().str.strip().tolist())
    if not pre.empty and "ticker" in pre.columns:
        tickers.update(pre["ticker"].astype(str).str.upper().str.strip().tolist())

    tickers = sorted([t for t in tickers if t and t not in {"CASH", "TACTICAL_CASH"}])

    rows = []
    for t in tickers:
        g = gamma[gamma["ticker"].astype(str).str.upper().str.strip() == t].iloc[0].to_dict() if (not gamma.empty and "ticker" in gamma.columns and not gamma[gamma["ticker"].astype(str).str.upper().str.strip() == t].empty) else {}
        k = kill[kill["ticker"].astype(str).str.upper().str.strip() == t].iloc[0].to_dict() if (not kill.empty and "ticker" in kill.columns and not kill[kill["ticker"].astype(str).str.upper().str.strip() == t].empty) else {}
        p = risk_from_pretrade(pre, t)

        row = {
            "ticker": t,
            "spot": fnum(g.get("spot", k.get("spot", np.nan))),
            "data_confidence": g.get("data_confidence", ""),
            "gamma_squeeze_label": g.get("gamma_squeeze_label", ""),
            "gamma_squeeze_score": fnum(g.get("gamma_squeeze_score", np.nan)),
            "call_wall_strike": fnum(g.get("call_wall_strike", np.nan)),
            "call_wall_distance": fnum(g.get("call_wall_distance", np.nan)),
            "put_wall_strike": fnum(g.get("put_wall_strike", np.nan)),
            "put_wall_distance": fnum(g.get("put_wall_distance", np.nan)),
            "net_gex_1pct_proxy": fnum(g.get("net_gex_1pct_proxy", np.nan)),
            "near_call_volume_oi_ratio": fnum(g.get("near_call_volume_oi_ratio", np.nan)),
            "option_kill_zone_label": k.get("option_kill_zone_label", ""),
            "option_kill_zone_score": fnum(k.get("option_kill_zone_score", np.nan)),
            "max_pain_proxy": fnum(k.get("max_pain_proxy", np.nan)),
            "max_pain_distance": fnum(k.get("max_pain_distance", np.nan)),
            "near_expiry_oi_ratio": fnum(k.get("near_expiry_oi_ratio", np.nan)),
            "median_spread_pct": fnum(k.get("median_spread_pct", np.nan)),
            "pretrade_status": p["pretrade_status"],
            "portfolio_risk_light": p["portfolio_risk_light"],
            "paper_allowed": p["paper_allowed"],
            "live_allowed": p["live_allowed"],
        }

        decision, rule, explanation = decision_rule(row)
        row["final_options_decision"] = decision
        row["rule"] = rule
        row["explanation"] = explanation

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        order = {
            "PAPER_REVIEW": 0,
            "PAPER_ONLY": 1,
            "WATCH": 2,
            "WAIT": 3,
            "RESEARCH_ONLY": 4,
            "SKIP_OPTIONS": 5,
            "SKIP": 6,
        }
        out["_sort"] = out["final_options_decision"].map(order).fillna(9)
        out = out.sort_values(["_sort", "gamma_squeeze_score"], ascending=[True, False]).drop(columns=["_sort"])
    return out


def fmt_table(df: pd.DataFrame, max_rows=50) -> str:
    if df.empty:
        return "_No data._"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if any(k in c for k in ["distance", "ratio", "spread"]):
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        if "gex" in c:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def build_report(df: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 32 — Options Decision Matrix")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This combines Gamma Watch, Option Kill Zone, and Pre-trade Risk into one decision table.")
    md.append("")
    md.append("## Decision Summary")
    md.append("")
    if df.empty:
        md.append("_No rows._")
    else:
        s = df["final_options_decision"].value_counts().reset_index()
        s.columns = ["decision", "count"]
        md.append(s.to_markdown(index=False))
    md.append("")
    md.append("## Matrix")
    md.append("")
    cols = [
        "ticker", "spot", "final_options_decision", "rule",
        "gamma_squeeze_label", "gamma_squeeze_score",
        "option_kill_zone_label", "option_kill_zone_score",
        "portfolio_risk_light", "live_allowed", "explanation",
    ]
    md.append(fmt_table(df[[c for c in cols if c in df.columns]], max_rows=80))
    md.append("")
    md.append("## Full Detail")
    md.append("")
    md.append(fmt_table(df, max_rows=80))
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- This is not a buy/sell signal.")
    md.append("- LIVE is blocked by default.")
    md.append("- Gamma Watch + High Kill Zone means wait for breakout confirmation, not chase weekly OTM options.")
    md.append("- RED portfolio risk means paper only or research only.")
    md.append("- Low confidence data means treat the signal as a weak screen.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 32")
    print("Options Decision Matrix")
    print("=" * 88)

    df = build_matrix()
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_report(df), encoding="utf-8")

    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df[["ticker", "final_options_decision", "rule", "gamma_squeeze_label", "option_kill_zone_label", "portfolio_risk_light"]].to_string(index=False))

    print()
    print("Files generated:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")
    print()
    print("Next: open options_decision_matrix.md")


if __name__ == "__main__":
    main()
