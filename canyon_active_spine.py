#!/usr/bin/env python3
"""
Canyon v9 Active Spine Audit.

This is not another trading engine. It is a product map that separates the
system into:
  - core spine: actually drives the research decision
  - support evidence: useful, but should stay behind expanders/source trails
  - calibration / QA: important, but not daily front-page material
  - legacy / compatibility: do not delete blindly, but do not treat as the system

Research-only. No broker connection. No live orders.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from canyon_final_v9_risk_framework_lib import ROOT, df_to_markdown, today_str, write_json, write_markdown_report
from canyon_final_v9_step70_daily_runner_all import ENGINES, WEEKLY_ENGINES


OUT_SPINE = ROOT / "canyon_active_spine.csv"
OUT_AUDIT = ROOT / "canyon_step_usefulness_audit.csv"
OUT_STATE = ROOT / "canyon_active_spine_state.json"
OUT_REPORT = ROOT / "canyon_active_spine_report.md"


ACTIVE_CORE = {
    61, 76, 95, 88, 142, 170, 143, 99, 129, 160, 171, 87, 127, 84,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 123, 157, 158, 125,
    173, 174, 178, 181, 182, 183, 184,
}

ACTIVE_SUPPORT = {
    63, 65, 66, 67, 68, 69, 691, 77, 79, 80, 81, 82, 83, 85, 89, 93, 94,
    96, 98, 100, 101, 102, 103, 104, 105, 107, 109, 110, 120, 124, 126,
    128, 130, 144, 145, 146, 147, 148, 149, 150, 151, 152, 167, 168, 169,
    172, 175, 176, 177, 179, 180,
}

CALIBRATION_QA = {
    62, 90, 91, 92, 108, 121, 122, 131, 132, 133, 134, 135, 136, 137,
    138, 139, 140, 141, 155, 156, 159, 161, 162, 163, 164, 165, 166,
}

LEGACY_COMPATIBILITY = {56}
WEEKLY_OR_SLOW = {75, 78}


WORKSTREAMS = [
    {
        "spine_order": 1,
        "workstream": "Data Truth",
        "core_steps": "61, 121, 159, 161, 163",
        "what_it_decides": "Can we trust the data, timestamp, and source trail?",
        "why_it_is_truly_useful": "Bad data can make every downstream score fake. This layer blocks false confidence.",
        "should_be_front_page": "Yes, as a compact health line only.",
        "depth_gap": "Needs real point-in-time vendor data before it becomes institution-grade.",
    },
    {
        "spine_order": 2,
        "workstream": "Market Regime",
        "core_steps": "76, 95, 101",
        "what_it_decides": "Is the broad market supportive, hostile, short-biased, or stressed?",
        "why_it_is_truly_useful": "It sets the environment before ticker stories. Risk and route should obey this.",
        "should_be_front_page": "Yes, one regime verdict and source trail.",
        "depth_gap": "Needs calibrated macro factor sensitivities and cycle-specific thresholds.",
    },
    {
        "spine_order": 3,
        "workstream": "Sector And Theme Cycle",
        "core_steps": "88, 142, 170, 143",
        "what_it_decides": "Which sector/subsector is early, mature, late-cycle, or lagging?",
        "why_it_is_truly_useful": "This is where your semiconductor-vs-software view belongs. It should not be a surface label.",
        "should_be_front_page": "Yes, but as sector thesis, not a raw table.",
        "depth_gap": "Needs deeper industry supply-chain, earnings revision, capex, and leadership handoff evidence.",
    },
    {
        "spine_order": 4,
        "workstream": "News To Causal Chain",
        "core_steps": "99, 129, 160, 171",
        "what_it_decides": "Which ticker is directly affected by news, and which peers/suppliers/customers get read-through?",
        "why_it_is_truly_useful": "This turns headlines into mapped targets and prevents random news chasing.",
        "should_be_front_page": "Yes, inside News Room and ticker memo.",
        "depth_gap": "Needs verified timestamps, source links, and more historical event samples.",
    },
    {
        "spine_order": 5,
        "workstream": "Signal Engine",
        "core_steps": "87, 127, 84, 94, 156, 162",
        "what_it_decides": "Does the signal have evidence, IC, decay, and failure analysis?",
        "why_it_is_truly_useful": "A score without validation is decoration. This decides whether signal weight deserves trust.",
        "should_be_front_page": "No. Show only the conclusion; keep details in validation views.",
        "depth_gap": "Needs more true out-of-sample signal history, not proxy-only backtests.",
    },
    {
        "spine_order": 6,
        "workstream": "Portfolio Construction",
        "core_steps": "123, 157, 63",
        "what_it_decides": "What portfolio would be allowed if signals and risk gates pass?",
        "why_it_is_truly_useful": "It converts ideas into constrained weights instead of equal-weight guesses.",
        "should_be_front_page": "Only as final allowed weight / no-new-exposure state.",
        "depth_gap": "Needs robust optimization, expected return uncertainty, turnover budget, and sleeve-level constraints.",
    },
    {
        "spine_order": 7,
        "workstream": "Institutional Risk",
        "core_steps": "111, 112, 113, 114, 115, 116, 117, 118, 131",
        "what_it_decides": "Can the portfolio survive VaR/CVaR, vol target, drawdown, correlation, liquidity, event, and tail risk?",
        "why_it_is_truly_useful": "This is one of the real high-value parts. It should override options and alpha.",
        "should_be_front_page": "Yes, as a red/green permission layer.",
        "depth_gap": "Needs threshold calibration from more true historical data and intraday risk.",
    },
    {
        "spine_order": 8,
        "workstream": "Execution And Cost",
        "core_steps": "125, 158",
        "what_it_decides": "Would spread, slippage, market impact, and participation rate destroy the edge?",
        "why_it_is_truly_useful": "A profitable signal can be unusable after costs. This is essential depth.",
        "should_be_front_page": "Only when it blocks a ticker.",
        "depth_gap": "Needs real bid/ask, volume curve, auction, and fill-quality assumptions.",
    },
    {
        "spine_order": 9,
        "workstream": "Options Sleeve",
        "core_steps": "82, 90, 128, 173, 174",
        "what_it_decides": "Whether options are allowed at all, and if so call/put/hedge/defined-risk only.",
        "why_it_is_truly_useful": "Useful only after risk, source, spread/TCA, IV/Greeks, gamma, and trigger gates clear.",
        "should_be_front_page": "No. Show only the permission answer in ticker memo.",
        "depth_gap": "Needs full Greeks book risk, IV surface, term structure, and dealer positioning history.",
    },
    {
        "spine_order": 10,
        "workstream": "Decision Desk",
        "core_steps": "119, 178, 181, 182, 183, 184",
        "what_it_decides": "What should be read first, what is blocked, what source proves it, and what to do next.",
        "why_it_is_truly_useful": "This is the product surface. It turns many engines into one usable workflow.",
        "should_be_front_page": "Yes. This should be the main UI spine.",
        "depth_gap": "Needs fewer raw tables and more concise narrative with source drilldowns.",
    },
    {
        "spine_order": 11,
        "workstream": "Backtest And Audit",
        "core_steps": "62, 122, 155, 164, 165, 166",
        "what_it_decides": "Is the backtest admissible, biased, stale, or only a prototype?",
        "why_it_is_truly_useful": "This prevents fake confidence. But it is a QA layer, not the daily dashboard.",
        "should_be_front_page": "No. Show only credibility status.",
        "depth_gap": "Needs point-in-time, survivorship-bias-free, walk-forward validation.",
    },
]


def tier_for_step(step_id: int) -> tuple[str, str, str]:
    if step_id in ACTIVE_CORE:
        return (
            "ACTIVE_CORE",
            "Keep. This can influence the current decision spine or the main dashboard answer.",
            "Show as a summarized conclusion, not as raw step clutter.",
        )
    if step_id in ACTIVE_SUPPORT:
        return (
            "ACTIVE_SUPPORT",
            "Keep as evidence. Useful when drilling into a ticker or source, but not a front-page step.",
            "Move behind expanders/source trails.",
        )
    if step_id in CALIBRATION_QA:
        return (
            "CALIBRATION_QA",
            "Keep for validation, calibration, or audit. Important for depth, not daily navigation.",
            "Run weekly or when testing logic; show only pass/fail summary.",
        )
    if step_id in LEGACY_COMPATIBILITY:
        return (
            "LEGACY_COMPATIBILITY",
            "Do not delete yet, but this should not define the product anymore.",
            "Hide from main UI and keep only for backward compatibility.",
        )
    if step_id in WEEKLY_OR_SLOW:
        return (
            "WEEKLY_OR_SLOW",
            "Useful but slow. It should not be part of every quick daily run.",
            "Run weekly/manual; show freshness status.",
        )
    return (
        "REVIEW_NEEDED",
        "Not classified yet. It may be useful, duplicate, or stale.",
        "Audit before surfacing.",
    )


def build_step_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_id, name, script, args in list(ENGINES) + list(WEEKLY_ENGINES):
        tier, verdict, ui_policy = tier_for_step(int(step_id))
        rows.append({
            "step_id": int(step_id),
            "engine_name": name,
            "script": script,
            "tier": tier,
            "usefulness_verdict": verdict,
            "ui_policy": ui_policy,
            "runner_args": " ".join(args),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).sort_values(["tier", "step_id"]).reset_index(drop=True)


def main() -> None:
    spine = pd.DataFrame(WORKSTREAMS)
    audit = build_step_audit()
    spine.to_csv(OUT_SPINE, index=False)
    audit.to_csv(OUT_AUDIT, index=False)

    tier_counts = audit["tier"].value_counts().to_dict()
    state = {
        "status": "ACTIVE_SPINE_AUDIT_COMPLETE",
        "date": today_str(),
        "runner_engine_count": int(len(audit)),
        "workstream_count": int(len(spine)),
        "active_core_steps": int(tier_counts.get("ACTIVE_CORE", 0)),
        "active_support_steps": int(tier_counts.get("ACTIVE_SUPPORT", 0)),
        "calibration_qa_steps": int(tier_counts.get("CALIBRATION_QA", 0)),
        "legacy_compatibility_steps": int(tier_counts.get("LEGACY_COMPATIBILITY", 0)),
        "weekly_or_slow_steps": int(tier_counts.get("WEEKLY_OR_SLOW", 0)),
        "product_decision": "Show workstreams, not step count. Core depth comes from source-backed gates, not more numbered files.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Product decision",
        state["product_decision"],
        "",
        "## What is truly useful",
        "The real system is the 11-workstream spine below. Individual step files are implementation details.",
        "",
        df_to_markdown(spine, max_rows=20),
        "",
        "## Step tier counts",
        df_to_markdown(pd.DataFrame([tier_counts]), max_rows=1),
        "",
        "## Full step usefulness audit",
        df_to_markdown(audit, max_rows=130),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Active Spine Audit", sections)
    print(
        f"Active spine complete: {len(spine)} workstreams, {len(audit)} engines, "
        f"{state['active_core_steps']} active core steps"
    )


if __name__ == "__main__":
    main()
