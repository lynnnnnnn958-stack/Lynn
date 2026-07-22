#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 44 — Ten-Layer Architecture Registry

This creates the formal 10-layer architecture map for Canyon.
It does not trade, does not fetch data, does not connect to broker.

Outputs:
- canyon_10_layer_architecture.md
- canyon_layer_registry.csv
- canyon_layer_status_audit.csv
- canyon_layer_build_plan.md
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path.cwd()

OUT_ARCH = ROOT / "canyon_10_layer_architecture.md"
OUT_REGISTRY = ROOT / "canyon_layer_registry.csv"
OUT_AUDIT = ROOT / "canyon_layer_status_audit.csv"
OUT_PLAN = ROOT / "canyon_layer_build_plan.md"


LAYERS = [
    {
        "layer_id": "L1",
        "layer_name": "Data & Universe Integrity",
        "purpose": "Make sure every signal is based on traceable, non-fabricated data.",
        "core_question": "Do we trust the ticker, price, timestamp, source, and coverage?",
        "expected_outputs": [
            "daily_pm_report.md",
            "market_data_snapshot.csv",
            "data_quality_report.md",
            "universe_master.csv",
        ],
        "current_known_outputs": [
            "daily_pm_report.md",
        ],
        "decision_use": "Blocks downstream decisions when source/timestamp/coverage is missing.",
        "user_vision_mapping": "Your system must not make up data; this is the evidence floor.",
        "next_build": "Build a data provenance table: ticker, source, timestamp, last price, stale flag, missing flag.",
    },
    {
        "layer_id": "L2",
        "layer_name": "Macro & Regime",
        "purpose": "Classify market regime: risk-on/risk-off, rates, liquidity, volatility, dollar, inflation.",
        "core_question": "Is the broad market regime supportive or hostile?",
        "expected_outputs": [
            "macro_regime_signals.csv",
            "macro_regime_report.md",
            "index_breadth_dashboard.csv",
            "volatility_regime.csv",
        ],
        "current_known_outputs": [
            "scenario_stress_results.csv",
        ],
        "decision_use": "Controls gross exposure, sector aggression, and whether shorts/hedges are needed.",
        "user_vision_mapping": "This supports sector rotation and avoids trading against the macro tape.",
        "next_build": "Add SPY/QQQ/IWM/TLT/UUP/VIX proxy regime and breadth signals.",
    },
    {
        "layer_id": "L3",
        "layer_name": "Sector & Theme Rotation",
        "purpose": "Rank sectors/themes by relative strength, trend persistence, and crowding.",
        "core_question": "Which sector/theme deserves attention now?",
        "expected_outputs": [
            "sector_rotation_scores.csv",
            "sector_rotation_report.md",
            "theme_heatmap.csv",
            "exposure_dashboard.csv",
        ],
        "current_known_outputs": [
            "exposure_dashboard.csv",
            "exposure_warnings.csv",
        ],
        "decision_use": "Selects sleeve candidates and prevents overconcentration in one theme.",
        "user_vision_mapping": "This is the rotating-with-market-sleeve you wanted.",
        "next_build": "Build sector ETF ranking: XLK/SMH/XLE/XLF/XLV/XLI/XLY/XLP/XLU/IYR.",
    },
    {
        "layer_id": "L4",
        "layer_name": "Fundamental, Quality & Valuation",
        "purpose": "Separate long-term hold quality from short-term hype.",
        "core_question": "Is this a business worth holding, or only a tactical trade?",
        "expected_outputs": [
            "fundamental_quality_valuation.csv",
            "fundamental_report.md",
            "long_term_hold_candidates.csv",
            "valuation_risk_flags.csv",
        ],
        "current_known_outputs": [],
        "decision_use": "Decides long-term sleeve eligibility and avoids turning trades into bag-holds.",
        "user_vision_mapping": "This supports your long-term hold / hedge / quality sleeve.",
        "next_build": "Add revenue growth, margin, FCF, debt, valuation percentile, earnings trend, moat notes.",
    },
    {
        "layer_id": "L5",
        "layer_name": "Event, News, SEC & Insider",
        "purpose": "Capture catalysts, regulatory events, Form 4 insider activity, earnings, and news risk.",
        "core_question": "What fresh event can change expectations?",
        "expected_outputs": [
            "sec_event_layer.csv",
            "evidence_cards.csv",
            "insider_form4_signals.csv",
            "news_event_risk.csv",
            "earnings_calendar_check.csv",
        ],
        "current_known_outputs": [
            "sec_event_layer.csv",
            "evidence_cards.csv",
        ],
        "decision_use": "Blocks trades before unchecked earnings/news; boosts trades with real catalysts.",
        "user_vision_mapping": "This is your macro top-down → insider buying → short squeeze/event filter.",
        "next_build": "Add explicit earnings-date check, Form 4 buy/sell classification, and event freshness scoring.",
    },
    {
        "layer_id": "L6",
        "layer_name": "Price, Technical & Microstructure",
        "purpose": "Detect short-term irrationality, breakouts, reversals, and failed moves.",
        "core_question": "Is price action confirming the thesis now?",
        "expected_outputs": [
            "technical_signal_matrix.csv",
            "tactical_candidates.csv",
            "breakout_reversal_watchlist.csv",
            "intraday_liquidity_proxy.csv",
        ],
        "current_known_outputs": [
            "position_sizing_recommendations.csv",
        ],
        "decision_use": "Times paper entries and stops false thesis-based trades.",
        "user_vision_mapping": "This supports short-term irrational market profit attempts.",
        "next_build": "Add multi-horizon trend, RSI/ATR, gap, volume spike, relative strength, failed breakout flags.",
    },
    {
        "layer_id": "L7",
        "layer_name": "Options, Dealer Gamma & Kill Zone",
        "purpose": "Measure option-chain pressure, gamma watch, call/put walls, and option buyer danger zones.",
        "core_question": "Is the options market creating squeeze pressure or killing option buyers?",
        "expected_outputs": [
            "options_chain_snapshot.csv",
            "gamma_squeeze_candidates.csv",
            "option_kill_zone_risk.csv",
            "options_decision_matrix.csv",
            "action_cards.csv",
        ],
        "current_known_outputs": [
            "options_chain_snapshot.csv",
            "gamma_squeeze_candidates.csv",
            "option_kill_zone_risk.csv",
            "options_decision_matrix.csv",
            "action_cards.csv",
        ],
        "decision_use": "Prevents chasing weekly OTM options; identifies wait/paper-only/skip states.",
        "user_vision_mapping": "This is the options/dealer hedging layer we just built.",
        "next_build": "Improve with true OI history, IV rank, term structure, and better dealer-positioning proxies.",
    },
    {
        "layer_id": "L8",
        "layer_name": "Portfolio Risk, Stress & Sizing",
        "purpose": "Control concentration, stress loss, factor exposure, and position size.",
        "core_question": "How much can we lose if the thesis is wrong or the regime shifts?",
        "expected_outputs": [
            "exposure_warnings.csv",
            "scenario_stress_results.csv",
            "position_sizing_recommendations.csv",
            "stress_position_sizing_report.md",
        ],
        "current_known_outputs": [
            "exposure_warnings.csv",
            "scenario_stress_results.csv",
            "position_sizing_recommendations.csv",
            "stress_position_sizing_report.md",
        ],
        "decision_use": "Caps suggested weight and blocks live action when risk is RED.",
        "user_vision_mapping": "This is the math-based risk and portfolio adjustment layer.",
        "next_build": "Add correlation matrix, marginal contribution to risk, drawdown budget, sleeve-level risk parity.",
    },
    {
        "layer_id": "L9",
        "layer_name": "Execution, Pre-trade, Paper Ledger & Runbook",
        "purpose": "Turn ideas into controlled paper actions only after checks.",
        "core_question": "What exactly are we allowed to do, and what is forbidden?",
        "expected_outputs": [
            "pre_trade_checklist.csv",
            "execution_gate_review.csv",
            "paper_portfolio_ledger.csv",
            "tonight_action_plan.md",
            "watch_triggers.csv",
            "action_cards.md",
        ],
        "current_known_outputs": [
            "pre_trade_checklist.csv",
            "execution_gate_review.csv",
            "paper_portfolio_ledger.csv",
            "tonight_action_plan.md",
            "watch_triggers.csv",
            "action_cards.md",
        ],
        "decision_use": "Creates allowed actions: WAIT, PAPER_ONLY, SKIP, RESEARCH_ONLY.",
        "user_vision_mapping": "This converts the model into a daily operating system without accidental trading.",
        "next_build": "Add paper-trade button helper, trigger logs, entry/exit rule templates, and audit notes.",
    },
    {
        "layer_id": "L10",
        "layer_name": "Learning, Attribution & Meta-Strategy",
        "purpose": "Learn from closed paper trades and decide what should be downweighted or reviewed.",
        "core_question": "Which sleeve/thesis/condition actually worked after closing trades?",
        "expected_outputs": [
            "learning_attribution_report.md",
            "learning_attribution_summary.csv",
            "learning_weight_suggestions.csv",
            "system_health_check.csv",
            "strategy_failure_flags.csv",
        ],
        "current_known_outputs": [
            "learning_attribution_report.md",
            "learning_attribution_summary.csv",
            "learning_weight_suggestions.csv",
            "system_health_check.csv",
        ],
        "decision_use": "Prevents overfitting and updates sleeve confidence only after enough samples.",
        "user_vision_mapping": "This is the machine-learning/self-adjusting part, but kept conservative.",
        "next_build": "Add sample-size gates, failure flags, regime-conditioned attribution, and no-trade learning.",
    },
]


def exists_any(files: list[str]) -> list[str]:
    return [f for f in files if (ROOT / f).exists()]


def missing(files: list[str]) -> list[str]:
    return [f for f in files if not (ROOT / f).exists()]


def maturity(found_count: int, expected_count: int) -> tuple[int, str]:
    if expected_count <= 0:
        return 0, "NO_SPEC"
    ratio = found_count / expected_count
    if ratio >= 0.90:
        return 5, "Operational"
    if ratio >= 0.65:
        return 4, "Usable"
    if ratio >= 0.40:
        return 3, "Partial"
    if ratio >= 0.15:
        return 2, "Skeleton"
    if found_count > 0:
        return 1, "Trace"
    return 0, "Missing"


def build_registry() -> pd.DataFrame:
    rows = []
    for layer in LAYERS:
        expected = layer["expected_outputs"]
        found = exists_any(expected)
        miss = missing(expected)
        score, status = maturity(len(found), len(expected))
        rows.append({
            "layer_id": layer["layer_id"],
            "layer_name": layer["layer_name"],
            "purpose": layer["purpose"],
            "core_question": layer["core_question"],
            "decision_use": layer["decision_use"],
            "user_vision_mapping": layer["user_vision_mapping"],
            "expected_outputs": ", ".join(expected),
            "found_outputs": ", ".join(found),
            "missing_outputs": ", ".join(miss),
            "maturity_score_0_5": score,
            "maturity_status": status,
            "next_build": layer["next_build"],
        })
    return pd.DataFrame(rows)


def build_architecture_md(df: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 — 10-Layer Investment System Architecture")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Core thesis")
    md.append("")
    md.append("The system is not an options screener. The options layer is only one layer. The full system should support:")
    md.append("")
    md.append("- short-term irrationality capture")
    md.append("- risk-controlled portfolio adjustment")
    md.append("- intraday/overnight tactical sleeve")
    md.append("- long-term hold / hedge sleeve")
    md.append("- sector rotation")
    md.append("- conservative self-learning after enough closed paper trades")
    md.append("")
    md.append("## Layer overview")
    md.append("")
    show = df[["layer_id", "layer_name", "maturity_score_0_5", "maturity_status", "core_question", "next_build"]].copy()
    md.append(show.to_markdown(index=False))
    md.append("")
    md.append("## Detailed layer design")
    md.append("")
    for _, r in df.iterrows():
        md.append(f"### {r['layer_id']} — {r['layer_name']}")
        md.append("")
        md.append(f"**Purpose:** {r['purpose']}")
        md.append("")
        md.append(f"**Core question:** {r['core_question']}")
        md.append("")
        md.append(f"**Decision use:** {r['decision_use']}")
        md.append("")
        md.append(f"**User vision mapping:** {r['user_vision_mapping']}")
        md.append("")
        md.append(f"**Current maturity:** {r['maturity_score_0_5']}/5 — {r['maturity_status']}")
        md.append("")
        md.append(f"**Found outputs:** {r['found_outputs'] or 'None'}")
        md.append("")
        md.append(f"**Missing outputs:** {r['missing_outputs'] or 'None'}")
        md.append("")
        md.append(f"**Next build:** {r['next_build']}")
        md.append("")
    md.append("## Immediate build order")
    md.append("")
    md.append("1. L1 Data & Universe Integrity — because no layer should trust stale or invented data.")
    md.append("2. L2 Macro & Regime — because sector rotation and risk need market context.")
    md.append("3. L3 Sector & Theme Rotation — because your system wants to follow market leadership.")
    md.append("4. L4 Fundamentals — because long-term hold decisions cannot rely on options/technical signals.")
    md.append("5. L5 Event/News/SEC/Insider — because catalysts drive short-term irrationality.")
    md.append("6. L6 Price/Technical — because timing still matters.")
    md.append("7. L7 Options/Gamma — already partly built; improve later.")
    md.append("8. L8 Risk/Sizing — already usable; improve with correlation and risk contribution.")
    md.append("9. L9 Execution/Paper — already usable; improve triggers and audit trail.")
    md.append("10. L10 Learning — already conservative; improve after more closed paper samples.")
    md.append("")
    return "\n".join(md)


def build_plan_md(df: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 — Layer Build Plan")
    md.append("")
    md.append("## What is wrong right now")
    md.append("")
    md.append("The options layer became detailed before the full 10-layer architecture was equally detailed. That is useful, but unbalanced.")
    md.append("")
    md.append("## Correct build strategy")
    md.append("")
    md.append("Do not add random indicators. Build one layer at a time, with clear input, output, gate, and decision impact.")
    md.append("")
    md.append("## Next 5 engineering steps")
    md.append("")
    md.append("1. Step 45: Master Decision Matrix — combine all layer states into one ticker-level table.")
    md.append("2. Step 46: 10-Layer Dashboard — show maturity, missing files, and decisions.")
    md.append("3. Step 47: Data Provenance Layer — source, timestamp, stale data, missing data flags.")
    md.append("4. Step 48: Macro/Regime Layer — SPY/QQQ/IWM/TLT/VIX/DXY proxy signals.")
    md.append("5. Step 49: Sector Rotation Layer — ETF leadership and theme rotation scores.")
    md.append("")
    md.append("## Layer maturity audit")
    md.append("")
    md.append(df[["layer_id", "layer_name", "maturity_score_0_5", "maturity_status", "missing_outputs", "next_build"]].to_markdown(index=False))
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 44")
    print("Ten-Layer Architecture Registry")
    print("=" * 88)

    df = build_registry()
    df.to_csv(OUT_REGISTRY, index=False)
    df.to_csv(OUT_AUDIT, index=False)

    OUT_ARCH.write_text(build_architecture_md(df), encoding="utf-8")
    OUT_PLAN.write_text(build_plan_md(df), encoding="utf-8")

    print(df[["layer_id", "layer_name", "maturity_score_0_5", "maturity_status"]].to_string(index=False))
    print()
    print("Files generated:")
    print(f"  {OUT_ARCH}")
    print(f"  {OUT_REGISTRY}")
    print(f"  {OUT_AUDIT}")
    print(f"  {OUT_PLAN}")
    print()
    print("Next: open canyon_10_layer_architecture.md")


if __name__ == "__main__":
    main()
