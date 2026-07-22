#!/usr/bin/env python3
"""
Canyon v9 Step 123 - Institutional Portfolio Builder.

Research-only. No broker connection. No live orders.

This step upgrades portfolio construction from "picks plus filters" into a
constraint-aware target book:
  - sleeve budgets
  - single-name caps
  - sector caps
  - risk-gate caps
  - TCA/capacity limits
  - beta/factor review flags
  - turnover budget

The output is a research target, not an order ticket.

Outputs:
  institutional_target_weights.csv
  institutional_sleeve_allocations.csv
  portfolio_constraint_matrix.csv
  portfolio_turnover_budget.csv
  portfolio_construction_state.json
  institutional_portfolio_builder_report.md
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    load_current_book,
    normalize_weight,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_TARGET = ROOT / "institutional_target_weights.csv"
OUT_SLEEVE = ROOT / "institutional_sleeve_allocations.csv"
OUT_CONSTRAINT = ROOT / "portfolio_constraint_matrix.csv"
OUT_TURNOVER = ROOT / "portfolio_turnover_budget.csv"
OUT_STATE = ROOT / "portfolio_construction_state.json"
OUT_REPORT = ROOT / "institutional_portfolio_builder_report.md"

NORMAL_MAX_GROSS = 1.00
RISK_REVIEW_MAX_GROSS = 0.70
SIZE_DOWN_MAX_GROSS = 0.50
REDUCE_ONLY_MAX_GROSS = 0.25
SINGLE_NAME_MAX = 0.04
EVENT_NAME_MAX = 0.025
TECH_SECTOR_MAX = 0.28
DEFAULT_SECTOR_MAX = 0.25
TURNOVER_BUDGET = 0.35


def num(value: Any, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else default


def action_factor(action: str) -> float:
    text = str(action).upper()
    if text in {"BLOCK_NEW", "BLOCKED"}:
        return 0.0
    if text == "REDUCE_ONLY":
        return 0.25
    if text == "SIZE_DOWN":
        return 0.60
    if text in {"MISSING_DATA_REVIEW", "REVIEW"}:
        return 0.75
    return 1.0


def status_from_score(score: float) -> str:
    if score >= 80:
        return "CLEAR"
    if score >= 60:
        return "REVIEW"
    if score >= 40:
        return "SIZE_DOWN"
    return "BLOCK_NEW"


def classify_sleeve(row: pd.Series) -> str:
    ticker = clean_ticker(row.get("ticker"))
    top_signal = str(row.get("top_signal", "")).upper()
    action = str(row.get("final_risk_action", row.get("action", ""))).upper()
    sector = str(row.get("sector", ""))
    if action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        return "Risk Control"
    if ticker in {"SPY", "QQQ", "XLK", "XLF", "XLV", "XLE", "IYR", "TLT", "GLD"}:
        return "Core Hedge"
    if "SURPRISE" in top_signal or "EARN" in top_signal:
        return "Event"
    if num(row.get("alpha_score")) >= 80 and sector in {"Technology", "Health Care", "Consumer Discretionary", "Industrials"}:
        return "Core"
    return "Tactical"


def target_gross_from_master(master_action: str, master_mult: float) -> float:
    text = str(master_action).upper()
    if text in {"BLOCK_NEW", "BLOCKED", "REDUCE_ONLY"}:
        return min(REDUCE_ONLY_MAX_GROSS, master_mult)
    if text == "SIZE_DOWN":
        return min(SIZE_DOWN_MAX_GROSS, master_mult)
    if text in {"REVIEW", "MISSING_DATA_REVIEW"}:
        return min(RISK_REVIEW_MAX_GROSS, master_mult)
    return min(NORMAL_MAX_GROSS, master_mult if master_mult > 0 else NORMAL_MAX_GROSS)


def load_inputs() -> pd.DataFrame:
    book = load_current_book(prefer_filtered=True)
    if book.empty:
        return pd.DataFrame()
    book = book.copy()
    book["ticker"] = book["ticker"].apply(clean_ticker)
    book["weight"] = book["weight"].apply(normalize_weight)
    gate = read_csv_safe(ROOT / "final_risk_gate.csv")
    tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    event_gate = read_csv_safe(ROOT / "event_research_gate.csv")
    sector = read_csv_safe(ROOT / "sector_active_exposure.csv")

    if not gate.empty and "ticker" in gate.columns:
        keep = [c for c in [
            "ticker", "final_risk_action", "recommended_risk_weight",
            "recommended_risk_weight_pct", "max_allowed_weight", "reason_stack",
        ] if c in gate.columns]
        gate = gate[keep].copy()
        gate["ticker"] = gate["ticker"].apply(clean_ticker)
        book = book.merge(gate, on="ticker", how="left")

    if not tca.empty and "ticker" in tca.columns:
        keep = [c for c in [
            "ticker", "total_tca_cost_bps", "execution_status",
            "participation_rate_pct", "avg_20d_dollar_volume",
        ] if c in tca.columns]
        tca = tca[keep].copy()
        tca["ticker"] = tca["ticker"].apply(clean_ticker)
        book = book.merge(tca, on="ticker", how="left")

    if not event_gate.empty and "ticker" in event_gate.columns:
        keep = [c for c in ["ticker", "event_gate", "event_risk_score", "event_research_score"] if c in event_gate.columns]
        event_gate = event_gate[keep].copy()
        event_gate["ticker"] = event_gate["ticker"].apply(clean_ticker)
        book = book.merge(event_gate, on="ticker", how="left")

    if not sector.empty and {"sector", "cap_status", "sector_cap"}.issubset(sector.columns):
        sector_keep = sector[["sector", "cap_status", "sector_cap"]].copy()
        book = book.merge(sector_keep, on="sector", how="left")

    return book


def build_targets(book: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if book.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    master_state = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})
    master_action = str(master_state.get("master_risk_action", "REVIEW"))
    master_mult = float(master_state.get("master_exposure_multiplier", 0.70) or 0.70)
    target_gross = target_gross_from_master(master_action, master_mult)

    work = book.copy()
    work["sleeve"] = work.apply(classify_sleeve, axis=1)
    work["alpha_score"] = pd.to_numeric(work.get("alpha_score", 50), errors="coerce").fillna(50.0)
    work["current_weight"] = pd.to_numeric(work.get("weight", 0), errors="coerce").fillna(0.0)
    work["score_weight"] = np.maximum(work["alpha_score"], 1.0)

    sector_caps = {}
    for sector_name in work["sector"].fillna("Unknown").astype(str).unique():
        sector_caps[sector_name] = TECH_SECTOR_MAX if sector_name == "Technology" else DEFAULT_SECTOR_MAX

    rows = []
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        sleeve = str(row.get("sleeve"))
        current_weight = normalize_weight(row.get("current_weight"))
        risk_action = str(row.get("final_risk_action", row.get("action", "REVIEW"))).upper()
        event_action = str(row.get("event_gate", "REVIEW")).upper()
        execution_status = str(row.get("execution_status", "REVIEW")).upper()
        sector_status = str(row.get("cap_status", "CLEAR")).upper()

        hard_cap = SINGLE_NAME_MAX
        if sleeve == "Event":
            hard_cap = EVENT_NAME_MAX
        if risk_action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
            hard_cap = min(hard_cap, max(current_weight * 0.50, 0.0025))
        if event_action in {"SIZE_DOWN", "BLOCK_NEW"}:
            hard_cap = min(hard_cap, EVENT_NAME_MAX)
        risk_cap = num(row.get("recommended_risk_weight"))
        if not np.isfinite(risk_cap):
            risk_cap = num(row.get("recommended_risk_weight_pct")) / 100.0
        if np.isfinite(risk_cap) and risk_cap > 0:
            hard_cap = min(hard_cap, risk_cap)

        raw_score = float(row.get("score_weight"))
        raw_weight = raw_score / max(float(work["score_weight"].sum()), 1e-12) * target_gross
        factor = action_factor(risk_action) * action_factor(event_action) * action_factor(execution_status)
        if sector_status in {"SIZE_DOWN", "BLOCK_NEW"}:
            factor *= 0.80
        proposed = min(raw_weight * max(factor, 0.0), hard_cap)
        if risk_action in {"BLOCK_NEW", "BLOCKED"} or event_action in {"BLOCK_NEW", "BLOCKED"}:
            proposed = 0.0

        status = "CLEAR"
        reasons = []
        if risk_action != "CLEAR":
            reasons.append(f"risk gate {risk_action}")
        if event_action != "CLEAR":
            reasons.append(f"event gate {event_action}")
        if execution_status != "CLEAR":
            reasons.append(f"execution {execution_status}")
        if proposed < raw_weight * 0.75:
            status = "SIZE_DOWN"
        if proposed == 0.0:
            status = "BLOCK_NEW"
        if risk_action == "REDUCE_ONLY":
            status = "REDUCE_ONLY"
        if not reasons:
            reasons.append("within current constraints")

        rows.append({
            "ticker": ticker,
            "sector": row.get("sector", "Unknown"),
            "sleeve": sleeve,
            "alpha_score": row.get("alpha_score"),
            "current_weight_pct": current_weight * 100,
            "raw_score_weight_pct": raw_weight * 100,
            "hard_cap_pct": hard_cap * 100,
            "target_weight_pct": proposed * 100,
            "target_weight": proposed,
            "master_action": master_action,
            "final_risk_action": risk_action,
            "event_gate": event_action,
            "execution_status": execution_status,
            "target_status": status,
            "reason": "; ".join(reasons),
            "source_file": "daily_picks_filtered.csv / final_risk_gate.csv / event_research_gate.csv / institutional_tca_cost_estimates.csv",
        })

    target = pd.DataFrame(rows)
    if not target.empty:
        for sector_name, cap in sector_caps.items():
            mask = target["sector"].astype(str) == sector_name
            sector_sum = float(target.loc[mask, "target_weight"].sum())
            if sector_sum > cap and sector_sum > 0:
                scale = cap / sector_sum
                target.loc[mask, "target_weight"] *= scale
                target.loc[mask, "target_weight_pct"] = target.loc[mask, "target_weight"] * 100
                target.loc[mask, "target_status"] = target.loc[mask, "target_status"].where(
                    target.loc[mask, "target_status"].isin(["BLOCK_NEW", "REDUCE_ONLY"]),
                    "SIZE_DOWN",
                )
                target.loc[mask, "reason"] = target.loc[mask, "reason"] + f"; sector cap {sector_name}"

    target_sum = float(target["target_weight"].sum()) if not target.empty else 0.0
    cash_weight = max(0.0, 1.0 - target_sum)

    sleeve_targets = {
        "Core": 0.35,
        "Tactical": 0.10,
        "Event": 0.05,
        "Core Hedge": 0.10,
        "Risk Control": 0.00,
        "Cash": cash_weight,
    }
    sleeve_rows = []
    for sleeve, budget in sleeve_targets.items():
        actual = float(target.loc[target["sleeve"] == sleeve, "target_weight"].sum()) if sleeve != "Cash" and not target.empty else cash_weight
        status = "CLEAR"
        if sleeve != "Cash" and actual > budget * 1.25 and budget > 0:
            status = "REVIEW"
        if sleeve == "Cash" and actual < 0.20 and master_action == "SIZE_DOWN":
            status = "REVIEW"
        sleeve_rows.append({
            "sleeve": sleeve,
            "target_budget_pct": budget * 100,
            "actual_target_pct": actual * 100,
            "budget_gap_pct": (actual - budget) * 100,
            "status": status,
            "source_file": "institutional_target_weights.csv",
        })
    sleeve = pd.DataFrame(sleeve_rows)

    constraint_rows = []
    gross = target_sum
    max_single = float(target["target_weight"].max()) if not target.empty else 0.0
    tech_weight = float(target.loc[target["sector"].astype(str) == "Technology", "target_weight"].sum()) if not target.empty else 0.0
    turnover = float(np.abs(target["target_weight"] - target["current_weight_pct"] / 100.0).sum()) if not target.empty else 0.0
    beta = read_csv_safe(ROOT / "portfolio_beta_report.csv")
    spy_beta = num(beta.loc[beta["factor"].astype(str) == "SPY_beta", "portfolio_beta"].iloc[0]) if not beta.empty and "factor" in beta.columns and (beta["factor"].astype(str) == "SPY_beta").any() else np.nan
    checks = [
        ("Gross exposure", gross, target_gross, "CLEAR" if gross <= target_gross + 1e-9 else "SIZE_DOWN", "Target gross should respect master risk state."),
        ("Single-name max", max_single, SINGLE_NAME_MAX, "CLEAR" if max_single <= SINGLE_NAME_MAX + 1e-9 else "SIZE_DOWN", "No one ticker should dominate the research book."),
        ("Technology sector max", tech_weight, TECH_SECTOR_MAX, "CLEAR" if tech_weight <= TECH_SECTOR_MAX + 1e-9 else "SIZE_DOWN", "Avoid hidden mega-tech concentration."),
        ("Turnover budget", turnover, TURNOVER_BUDGET, "CLEAR" if turnover <= TURNOVER_BUDGET else "REVIEW", "Turnover should not eat the signal."),
        ("SPY beta review", spy_beta, 1.10, "CLEAR" if np.isfinite(spy_beta) and abs(spy_beta) <= 1.10 else "REVIEW", "Portfolio beta should be explicit and reviewed."),
        ("Cash reserve", cash_weight, 0.20 if master_action == "SIZE_DOWN" else 0.05, "CLEAR" if cash_weight >= (0.20 if master_action == "SIZE_DOWN" else 0.05) else "REVIEW", "Risk-down states should preserve cash budget."),
    ]
    for name, current, limit, status, note in checks:
        constraint_rows.append({
            "constraint": name,
            "current_value": current,
            "limit_value": limit,
            "status": status,
            "note": note,
            "source_file": "institutional_target_weights.csv / portfolio_beta_report.csv",
        })
    constraints = pd.DataFrame(constraint_rows)

    turnover_rows = []
    if not target.empty:
        for _, row in target.iterrows():
            current = float(row.get("current_weight_pct", 0.0)) / 100.0
            target_w = float(row.get("target_weight", 0.0))
            turnover_rows.append({
                "ticker": row.get("ticker"),
                "current_weight_pct": current * 100,
                "target_weight_pct": target_w * 100,
                "turnover_pct": abs(target_w - current) * 100,
                "direction": "UP" if target_w > current else ("DOWN" if target_w < current else "FLAT"),
                "source_file": "daily_picks_filtered.csv / institutional_target_weights.csv",
            })
    turnover_df = pd.DataFrame(turnover_rows)
    return target, sleeve, constraints, turnover_df


def write_outputs(target: pd.DataFrame, sleeve: pd.DataFrame, constraints: pd.DataFrame, turnover: pd.DataFrame) -> None:
    target.to_csv(OUT_TARGET, index=False)
    sleeve.to_csv(OUT_SLEEVE, index=False)
    constraints.to_csv(OUT_CONSTRAINT, index=False)
    turnover.to_csv(OUT_TURNOVER, index=False)
    status_scores = {"CLEAR": 85, "REVIEW": 65, "SIZE_DOWN": 45, "REDUCE_ONLY": 35, "BLOCK_NEW": 20}
    target_score = float(target["target_status"].astype(str).str.upper().map(status_scores).mean()) if not target.empty else 20.0
    constraint_score = float(constraints["status"].astype(str).str.upper().map(status_scores).mean()) if not constraints.empty else 20.0
    score = 0.60 * target_score + 0.40 * constraint_score
    state = {
        "date": today_str(),
        "portfolio_construction_score": round(score, 1),
        "overall_status": status_from_score(score),
        "target_gross_pct": round(float(target["target_weight"].sum()) * 100, 2) if not target.empty else 0.0,
        "cash_reserve_pct": round(max(0.0, 1.0 - float(target["target_weight"].sum())) * 100, 2) if not target.empty else 100.0,
        "constraint_flags": int((constraints["status"].astype(str).str.upper() != "CLEAR").sum()) if not constraints.empty else 0,
        "research_only": True,
        "no_broker_connection": True,
        "truth": "Research target weights only. No broker connection and no live order path.",
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Product Truth",
        "",
        state["truth"],
        "",
        f"- Portfolio construction score: {state['portfolio_construction_score']}",
        f"- Overall status: {state['overall_status']}",
        f"- Target gross: {state['target_gross_pct']}%",
        f"- Cash reserve: {state['cash_reserve_pct']}%",
        "",
        "## Constraint Matrix",
        "",
        df_to_markdown(constraints, max_rows=40),
        "",
        "## Target Weights",
        "",
        df_to_markdown(target, max_rows=80),
        "",
        "## Sleeve Allocations",
        "",
        df_to_markdown(sleeve, max_rows=20),
        "",
        "## Turnover Budget",
        "",
        df_to_markdown(turnover, max_rows=80),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 123 - Institutional Portfolio Builder", sections)


def main() -> None:
    book = load_inputs()
    target, sleeve, constraints, turnover = build_targets(book)
    write_outputs(target, sleeve, constraints, turnover)
    state = read_json_safe(OUT_STATE, {})
    print(f"[step123] wrote {OUT_TARGET.name}: {len(target)} target rows")
    print(f"[step123] score={state.get('portfolio_construction_score')} status={state.get('overall_status')} target_gross={state.get('target_gross_pct')}%")
    print(f"[step123] wrote {OUT_SLEEVE.name}, {OUT_CONSTRAINT.name}, {OUT_TURNOVER.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
