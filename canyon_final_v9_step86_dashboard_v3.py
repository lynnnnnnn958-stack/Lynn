#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 dashboard launcher.

The full Step 86 source was recovered from the last known-good Python 3.14
bytecode snapshot after a local file overwrite. This launcher loads that
snapshot, keeps every existing dashboard function available, and overrides only
the top-level navigation so Live / Paper and Risk / Portfolio are explicit.
"""

from __future__ import annotations

import marshal
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
CACHED_DASHBOARD = ROOT / "canyon_final_v9_step86_dashboard_v3.cached_20260603_1551.pyc"
BROWSER_DAILY_RUN_STATE = ROOT / "browser_daily_update_state.json"
BROWSER_DAILY_RUN_LOG = ROOT / "browser_daily_update.log"


def _load_cached_dashboard() -> dict:
    if not CACHED_DASHBOARD.exists():
        raise FileNotFoundError(
            f"Missing cached dashboard snapshot: {CACHED_DASHBOARD}"
        )

    raw = CACHED_DASHBOARD.read_bytes()
    code = marshal.loads(raw[16:])
    namespace = {
        "__name__": "canyon_step86_cached",
        "__file__": __file__,
        "__package__": None,
        "__builtins__": __builtins__,
    }
    exec(code, namespace)
    return namespace


_cached = _load_cached_dashboard()
for _name, _value in _cached.items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


_ORIGINAL_TAB_LIVE_PAPER_MONITOR = globals().get("tab_live_paper_monitor")
_ORIGINAL_TAB_SYSTEM_STATUS = globals().get("tab_system_status")
_ORIGINAL_TAB_PERFORMANCE = globals().get("tab_performance")
_ORIGINAL_TAB_NEWS_ROOM = globals().get("tab_news_room")
_ORIGINAL_SHOW_STATUS_TABLE = globals().get("_show_status_table")
_ORIGINAL_PLAIN_STATUS = globals().get("_plain_status")
_ORIGINAL_CLEAN_DISPLAY = globals().get("_clean_display")


def _run_with_markdown_replacements(fn, replacements: dict[str, str]):
    original_markdown = st.markdown

    def patched_markdown(body, *args, **kwargs):
        if isinstance(body, str):
            body = replacements.get(body, body)
        return original_markdown(body, *args, **kwargs)

    st.markdown = patched_markdown
    try:
        return fn()
    finally:
        st.markdown = original_markdown


def _run_with_plain_streamlit_text(fn):
    originals = {
        "caption": st.caption,
        "error": st.error,
        "info": st.info,
        "markdown": st.markdown,
        "success": st.success,
        "warning": st.warning,
        "write": st.write,
    }

    def clean_body(body):
        if isinstance(body, str):
            return _human_text(body, max_len=None)
        return body

    def clean_markup(body):
        if not isinstance(body, str):
            return body
        replacements = {
            "Risk: SIZE_DOWN": "Risk: Use smaller size",
            "Risk: REDUCE_ONLY": "Risk: Reduce only",
            "Event: CLEAR": "Event: Clear",
            "Backtest": "Old-data test",
            "backtest": "old-data test",
            "Causal": "Story link",
            "causal": "story link",
            "Correlation": "Moving together",
            "correlation": "moving together",
            "DATA_GAP": "Missing data",
            "Deep analysis for this section": "How to read this page",
            "Execution": "Trading cost",
            "execution": "trading cost",
            "Gate": "Check",
            "gate": "check",
            "Gross exposure": "Account size used",
            "gross exposure": "account size used",
            "IC": "model follow-up score",
            "Live IC": "Live follow-up check",
            "NAV": "Account value",
            "PM ": "Desk ",
            "Proof Sharpe": "Trust score",
            "proof-adjusted": "trust score",
            "Risk command center": "Simple safety answer",
            "Risk repair path": "What to fix first",
            "Route": "Choice",
            "route": "choice",
            "Sharpe": "Performance score",
            "Signal proof": "Model follow-up check",
            "Source health": "Data source status",
            "TCA": "Trading cost",
            "VaR": "Bad-day estimate",
            "CVaR": "Very bad-day estimate",
            "Vehicle": "Stock / option choice",
            "vehicle": "stock / option choice",
            "SIZE_DOWN": "Use smaller size",
            "REDUCE_ONLY": "Reduce only",
            "PUT_OR_HEDGE_RESEARCH_ONLY": "Put or hedge research only",
            "CALL_RESEARCH_ONLY": "Call research only",
            "RISK_REDUCTION_ONLY": "Risk reduction only",
            "MANUAL_SPREAD_LIQUIDITY_CHECK": "Manual spread and liquidity check",
            "NOT_IN_RISK_BOOK_REVIEW": "Not in risk book; review manually",
            "STOCK_OR_ETF_RESEARCH_ONLY": "Stock or ETF research only",
            "PEER_READ_THROUGH": "Related stock",
            "UNKNOWN_NEEDS_DATA": "Unknown; needs data",
            "NO_GO": "Not allowed",
            "read-through": "related-stock effect",
            "veto": "stop",
        }
        text = body
        for raw, friendly in replacements.items():
            text = text.replace(raw, friendly)
        return text

    st.caption = lambda body, *args, **kwargs: originals["caption"](clean_body(body), *args, **kwargs)
    st.error = lambda body, *args, **kwargs: originals["error"](clean_body(body), *args, **kwargs)
    st.info = lambda body, *args, **kwargs: originals["info"](clean_body(body), *args, **kwargs)
    st.markdown = lambda body, *args, **kwargs: originals["markdown"](clean_markup(body), *args, **kwargs)
    st.success = lambda body, *args, **kwargs: originals["success"](clean_body(body), *args, **kwargs)
    st.warning = lambda body, *args, **kwargs: originals["warning"](clean_body(body), *args, **kwargs)
    st.write = lambda body=None, *args, **kwargs: originals["write"](clean_body(body), *args, **kwargs)
    try:
        return fn()
    finally:
        for name, original in originals.items():
            setattr(st, name, original)


def _to_float(value, default=None):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _money(value) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    sign = "-" if num < 0 else ""
    return f"{sign}${abs(num):,.2f}"


def _pct_display(value) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    if abs(num) <= 1.5:
        num *= 100
    return f"{num:+.2f}%"


_FRIENDLY_PHRASES = {
    "ALREADY_CLOSED_DO_NOT_REPEAT": "Already closed; do not repeat",
    "APPROVE_TINY_PAPER_REVIEW": "Approve tiny paper review",
    "PM_EVIDENCE_ACCEPTANCE_GATE_ACTIVE": "Evidence review is active",
    "PM_EVIDENCE_REVIEW_TRIAGE_ACTIVE": "Evidence review is active",
    "PM_EVIDENCE_SOURCE_PROOF_DESK_ACTIVE": "Outside-source check is active",
    "PM_EVIDENCE_PROOF_ACCEPTANCE_BRIDGE_ACTIVE": "Evidence handoff is active",
    "NO_VERIFIED_SOURCE_PROOF_READY": "No verified source proof is ready yet",
    "AUDITABLE_LOCAL_EVENT_TIME": "Local event time is auditable",
    "CALL_RESEARCH_ONLY": "Call idea, research only",
    "CALL_REVIEW": "Call idea needs review",
    "CAUSAL_REVIEW_REQUIRED": "Story links need review",
    "CONTEXT_ONLY": "Context only",
    "CONTRADICTED_REVIEW_REQUIRED": "Contradicted; review manually",
    "DATA_GAP": "Missing data",
    "BENEFICIARY": "Direct possible winner",
    "PEER_READ_THROUGH": "Related stock",
    "UPSTREAM_BENEFICIARY": "Supplier or upstream winner",
    "DOWNSTREAM_BENEFICIARY": "Customer or downstream winner",
    "VULNERABLE_TARGET": "Possible loser",
    "VALIDATED_RESEARCH_LINK": "Research link is supported",
    "HYPOTHESIS_NEEDS_VALIDATION": "Idea still needs proof",
    "NO_RELIABILITY_DATA": "No reliability history yet",
    "WAIT_FOR_PRICE_CONFIRMATION": "Wait for price confirmation",
    "MIXED_UP": "Mixed but improving",
    "DOWNTREND_CONFIRMED": "Downtrend confirmed",
    "NO_MONITOR_DATA": "No live monitor data yet",
    "DOWNSIDE_WATCH_OR_HEDGE_RESEARCH": "Downside watch or hedge research",
    "EVENT_TIME_REVIEW_REQUIRED": "Event timing needs review",
    "FIX_DATA_FIRST": "Fix missing data first",
    "HAS_VALIDATED_EDGES": "Some links are validated",
    "IN_UNIVERSE": "In research universe",
    "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL": "Local audit only; not institutional proof",
    "LOW_LOOKAHEAD_RISK_LOCAL": "Low local look-ahead risk",
    "MODEL_FORWARD_PENDING": "Waiting for live forward proof",
    "MANUAL_SPREAD_LIQUIDITY_CHECK": "Manual spread and liquidity check",
    "NO_BROKER_CONNECTION": "No broker connection",
    "NO_DATA": "No data",
    "NO_DIRECTIONAL_TRADE": "No directional trade",
    "NO_GO": "Do not use options",
    "NO_LIVE_ORDERS": "No live orders",
    "NEEDS_HUMAN_DECISION": "Needs human decision",
    "NEEDS_OUTSIDE_CONFIRMATION": "Needs outside confirmation",
    "NEEDS_REVIEW": "Needs review",
    "NOT_IN_RISK_BOOK_REVIEW": "Not in risk book; review manually",
    "P1_REVIEW_CONTRADICTION": "Urgent review: price disagrees with story",
    "P2_VALIDATE": "Validate before trusting",
    "P3_CONTEXT": "Context only",
    "PAPER_ONLY": "Paper only",
    "PRICE_DISAGREES": "Price action disagrees",
    "PUT_RESEARCH_ONLY": "Put idea, research only",
    "PUT_OR_HEDGE_RESEARCH_ONLY": "Put or hedge idea, research only",
    "READTHROUGH_RESEARCH_BOARD_ACTIVE": "News map is active",
    "REDUCE_ONLY": "Reduce only",
    "RISK_REDUCTION_ONLY": "Risk reduction only",
    "RESEARCH_ONLY": "Research only",
    "RISK_BLOCKED": "Risk blocks action",
    "RISK_REDUCTION_FIRST": "Reduce risk first",
    "SEED_REVIEW_ONLY": "Risk seed needs review",
    "SIZE_DOWN": "Use smaller size",
    "STOCK_OR_ETF_RESEARCH_ONLY": "Stock or ETF research only",
    "NO_KELLY_UNTIL_LIVE_IC": "No formula sizing until the model proves it works live",
    "SOURCE_REACTION_CALIBRATION_ONLY_MODEL_FORWARD_PENDING": "Early evidence only; wait for live proof",
    "TINY_PAPER_ONLY": "Tiny paper only",
    "UNKNOWN_NEEDS_DATA": "Unknown; needs data",
    "WATCH_EVENT_PROOF_FIRST": "Verify event first",
    "WATCH_FOR_CONFIRMATION": "Watch, but prove it first",
    "WATCH_ONLY": "Watch only",
}


_FRIENDLY_KEYS = {
    "action": "Action",
    "blocker": "Blocker",
    "correlation": "Moving together",
    "execution": "Trading cost",
    "option": "Options",
    "option_no_go_checks": "Options block",
    "price": "Price",
    "risk": "Safety",
    "source": "Source",
    "spread": "Trading cost and volume",
    "volume": "Volume",
}


_SOURCE_LABELS = {
    "desk_monitor_ticker_state.csv": "Price and volume monitor",
    "desk_monitor_events.csv": "Live-style monitor events",
    "event_causal_validation_queue.csv": "News proof list",
    "event_readthrough_event_summary.csv": "News map",
    "event_readthrough_target_ranking.csv": "News target map",
    "execution_tca_ticker_cards.csv": "Trading-cost check",
    "macro_scenario_stress.csv": "Macro stress test",
    "options_tca_no_go_audit.csv": "Options cost check",
    "risk_desk_breach_table.csv": "Risk limit table",
    "sharpe4_manual_proof_review_gate.csv": "Manual evidence review",
    "single_name_risk_budget.csv": "Single-stock risk budget",
    "vol_target_state.json": "Movement target",
    "portfolio_var_cvar_summary.csv": "Portfolio loss estimate",
    "crisis_correlation_stress.csv": "Crisis correlation stress test",
    "earnings_gap_down_risk.csv": "Earnings jump-risk check",
    "momentum_scores.csv": "Momentum signal (Step 127)",
    "momentum_signal_report.md": "Momentum signal report",
    "spy_price_cache.csv": "SPY benchmark prices (for residual momentum)",
    "alpha_scores.csv": "Combined alpha scores (all signals aggregated)",
    "weekly_report.md": "Weekly performance report",
    "weekly_summary.json": "Weekly summary state",
}


def _friendly_source_label(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "Local system files"
    parts = [p.strip() for p in re.split(r"[;,|]", text) if p.strip()]
    labels = []
    for part in parts:
        fname = part.split("/")[-1].strip()
        labels.append(_SOURCE_LABELS.get(fname, fname.replace("_", " ").replace(".csv", "").replace(".json", "").title()))
    unique = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    return ", ".join(unique[:3]) + ("..." if len(unique) > 3 else "")


_FRIENDLY_COLUMNS = {
    "account_equity": "Account Equity",
    "acceptance_status": "Evidence Decision",
    "accepted_count": "Accepted",
    "accepted_field_count": "Accepted Fields",
    "accepted_fields": "Accepted Fields",
    "acceptable_proof": "What Counts As Proof",
    "action_hint": "Suggested Next Step",
    "allowed_next_state": "Allowed Next State",
    "approval_lane": "Approval Lane",
    "approval_score_0_100": "Approval Score",
    "approved_cap_pct": "Approved Cap",
    "avg_causal_confidence": "Story Link Trust",
    "base_cost_bps": "Normal Cost",
    "bid_ask_spread_bps": "Bid/Ask Spread",
    "best_event_score": "Best Event Score",
    "best_horizon": "Best Holding Time",
    "best_mean_ic": "Best Signal Skill",
    "biggest_gap": "Biggest Gap",
    "blocker_rows": "Blocker Rows",
    "blocker_type": "Blocker Type",
    "blocked_field_count": "Blocked Fields",
    "blocking_gates": "Blocking Checks",
    "bridge_decision": "Bridge Decision",
    "bridge_rows": "Bridge Rows",
    "bridge_state": "Bridge State",
    "calibration_action": "Calibration Action",
    "calibrated_reliability_status": "Reliability",
    "calibrated_research_action": "Research Action",
    "card": "Card",
    "can_use_for_sizing": "Can Size From This",
    "causal_chain_status": "Story Link Status",
    "causal_confidence_score": "Story Link Trust",
    "causal_confidence_0_100": "Story Link Trust",
    "causal_permission": "Can Use News For",
    "causal_thesis": "Why The Story Links",
    "chain_role": "Industry Link",
    "check": "Check",
    "confidence_0_100": "Confidence",
    "conflict_count": "Conflicts",
    "conflict_reason": "Conflict Reason",
    "copy_instruction": "Copy Instruction",
    "current_step204_decision": "Current Step204 Decision",
    "current_final_gate": "Current Final Check",
    "current_final_permission": "Current Final Permission",
    "current_first_blocker": "Current First Blocker",
    "current_weight_pct": "Current Weight",
    "decision_note": "Decision Note",
    "decision": "Decision",
    "decision_date": "Decision Date",
    "decision_id": "Decision ID",
    "days_stale_vs_today": "Days Stale",
    "done_when": "Done When",
    "draft_fill_count": "Draft Fill Count",
    "draft_filled_cells": "Draft Filled Cells",
    "draft_rows": "Draft Rows",
    "directional_route": "Choice",
    "entry_price": "Entry Price",
    "entry_price_date": "Entry Price Date",
    "execution_proof_open": "Trading-Cost Proof Open",
    "execution_permission": "Trading-Cost Permission",
    "execution_status": "Trading-Cost Status",
    "event_score": "Event Score",
    "event_size_policy": "Event Size Policy",
    "event_time_status": "Event Time Status",
    "eval_price": "Future Price",
    "eval_price_date": "Future Price Date",
    "existing_value": "Existing Value",
    "expected_fill_rate_pct": "Expected Fill Rate",
    "failure_mode": "Failure Pattern",
    "field_name": "Field",
    "field_group": "Evidence Type",
    "field_count": "Field Count",
    "field_review_count": "Evidence Fields",
    "final_permission": "Final Permission",
    "final_risk_action": "Risk Gate",
    "final_weight_pct": "Final Weight",
    "first_blocker": "First Blocker",
    "first_blocking_gate": "First Blocking Check",
    "first_missing_proof": "First Missing Proof",
    "gate": "Check",
    "gate_state": "Check State",
    "headline": "Headline",
    "hedge": "Hedge",
    "high_confidence_count": "High Confidence",
    "high_confidence_suggestions": "High Confidence Suggestions",
    "high_priority_count": "High Priority",
    "high_priority_evidence_fields": "High-Priority Fields",
    "high_priority_field_count": "High-Priority Fields",
    "high_priority_needs_proof_count": "High-Priority Missing Proof",
    "horizon_days": "Horizon Days",
    "forward_return_pct": "Forward Return",
    "gate_or_blocker": "Check Or Blocker",
    "industry_chain_read": "Industry Chain Link",
    "impact_score": "Impact Score",
    "liquidity_read": "Liquidity Read",
    "liquidity_snapshot_date": "Liquidity Snapshot Date",
    "missing_fields_plain": "Missing Fields",
    "latest_news_title": "Latest News",
    "latest_price": "Latest Price",
    "latest_price_date": "Latest Price Date",
    "latest_download_date": "Latest Download Date",
    "latest_download_close": "Latest Download Close",
    "link": "Source Link",
    "lookahead_risk": "Look-Ahead Risk",
    "low_confidence_count": "Low Confidence",
    "low_confidence_suggestions": "Low Confidence Suggestions",
    "human_note": "Human Note",
    "how_to_decide": "How To Decide",
    "ignored_count": "Ignored",
    "math_optimizer_wants_pct": "Optimizer Wants",
    "max_seed_cap_if_all_manual_gates_clear_pct": "Max Seed Cap If All Gates Clear",
    "max_tiny_paper_review_pct": "Max Tiny Paper Review",
    "max_paper_weight_pct": "Max Paper Weight",
    "medium_term": "Medium Term",
    "market_tone": "Tone",
    "market_snapshot_confidence": "Market Snapshot",
    "matching_step204_row_found": "Matching Step204 Row",
    "main_blocker": "Main Blocker",
    "medium_confidence_count": "Medium Confidence",
    "missing_proof": "Missing Proof",
    "missing_price_count": "Missing Prices",
    "missed_winner_count": "Missed Winner Count",
    "module": "Module",
    "monitor_status": "Monitor Status",
    "next_step": "Next Step",
    "news_and_execution": "News And Execution",
    "news_direction": "News Direction",
    "news_logic": "Why It Matters",
    "news_rank": "News Rank",
    "needs_external_confirmation_count": "Needs Outside Confirmation",
    "needs_proof_count": "Needs Proof",
    "non_option_gates_passed": "Non-Option Checks Passed",
    "no_price_count": "No Price",
    "news_proof_open": "News Proof Open",
    "option_permission": "Option Permission",
    "option_rule": "Option Rule",
    "option_side": "Option Side",
    "options_allowed": "Options Allowed",
    "observation_status": "Observation Status",
    "observed_time": "Observed Time",
    "observed_value": "Observed Value",
    "open_proof_items": "Open Proof Items",
    "open_blocker_count": "Open Blockers",
    "open_blocker_types": "Open Blocker Types",
    "outside_check_field_count": "Outside Checks",
    "outside_checks_needed": "Outside Checks Needed",
    "outside_proof_rows": "Outside Proof Rows",
    "outside_sources_to_check": "Outside Sources To Check",
    "outcome_label": "Outcome",
    "option_after_seed_approval": "Options After Seed Approval",
    "official_value": "Official Value",
    "participation_rate_pct": "Trading Volume Share",
    "paper_stop_pct": "Paper Stop",
    "pending_observations": "Pending Observations",
    "pending_count": "Waiting",
    "patch_status": "Patch Status",
    "patch_rows": "Patch Rows",
    "preferred_source": "Preferred Source",
    "price_reaction_checked": "Price Reaction Checked",
    "pit_quality_status": "Time-Stamp Quality",
    "plain_status": "Plain Status",
    "plain_problem": "Problem",
    "plain_answer": "Plain Answer",
    "plain_blocker": "Blocker",
    "plain_headline": "Plain Headline",
    "plain_rule": "Plain Rule",
    "pm_review_state": "Review State",
    "pm_review_status": "Review Status",
    "portfolio_v2_decision": "Portfolio Decision",
    "price_data_status": "Price Data",
    "price_rows": "Price Rows",
    "primary_route_now": "Choice Now",
    "price_status": "Price Status",
    "can_validate_forward": "Can Grade Later",
    "proof_required": "Proof Still Needed",
    "proof_needed": "Proof Still Needed",
    "proof_id": "Proof ID",
    "proof_note": "Proof Note",
    "proof_row_count": "Proof Rows",
    "ready_proof_rows": "Ready Proof Rows",
    "proof_score_0_100": "Proof Score",
    "proof_status": "Proof Status",
    "protected_loss_count": "Protected Loss Count",
    "published": "Published",
    "readthrough_decision": "Related-Stock Decision",
    "reason": "Reason",
    "recommended_action": "Recommended Action",
    "ready_observations": "Ready Observations",
    "ready_count": "Ready",
    "ready_for_accept_count": "Ready For Acceptance",
    "repair_status": "Fix Status",
    "repair_type": "Fix Type",
    "review_order": "Review Order",
    "review_priority": "Review Priority",
    "review_priority_score": "Review Priority Score",
    "required_next_action": "Next Step",
    "required_question": "Question To Answer",
    "review_state": "Review State",
    "review_status": "Review Status",
    "reviewer": "Reviewer",
    "review_date": "Review Date",
    "review_rows": "Review Rows",
    "rejected_count": "Rejected",
    "rejected_or_unavailable_count": "Rejected Or Unavailable",
    "risk_allows_pct": "Safety Allows",
    "risk_level": "Safety Level",
    "risk_seed_status": "Safety Review Status",
    "robust_weight_v2_pct": "Robust Weight",
    "route_before_risk": "Choice Before Safety Check",
    "sample_windows": "Sample Windows",
    "score_0_100": "Score",
    "score_component": "Check",
    "seed_cap_after_manual_approval_pct": "Seed Cap If Approved",
    "seed_ticker_count": "Seed Tickers",
    "sector_or_theme": "Sector Or Theme",
    "severity": "Severity",
    "short_term": "Short Term",
    "signal": "Model Ingredient",
    "sleeve": "Sleeve",
    "source_file": "Source File",
    "source_files": "Source Files",
    "source_check_type": "Source Check Type",
    "source_name": "Source Name",
    "source_proof_state": "Source Proof State",
    "source_url": "Source URL",
    "source_quality": "Source Quality",
    "suggested_value": "Suggested Value",
    "suggestion_id": "Suggestion ID",
    "suggestion_count": "Suggestions",
    "suggestion_status": "Suggestion Status",
    "still_blocks_after_seed_approval": "Still Blocks After Seed Approval",
    "still_forbidden": "Still Forbidden",
    "starter_cap_if_approved_pct": "Starter Cap If Approved",
    "source_news_ticker": "News Source",
    "spread_bps": "Trading Spread",
    "stage_order": "Step",
    "step204_action_hint": "Step204 Action Hint",
    "step204_rows": "Step204 Rows",
    "step204_suggested_value": "Step204 Suggested Value",
    "step204_suggestion_id": "Step204 Suggestion ID",
    "status": "Status",
    "system_seed_cap_pct": "System Seed Cap",
    "system_stop_pct": "System Stop",
    "stale_price_count": "Stale Prices",
    "stock_or_etf": "Stock Or ETF",
    "stress_cost_bps": "Stress Cost",
    "target_reason": "Why Linked",
    "target_relation": "Relationship",
    "target_role": "Role",
    "target_ticker": "Target",
    "ticker": "Ticker",
    "ticker_count": "Tickers",
    "top_answer": "Top Answer",
    "top_review_ticker": "Top Review Ticker",
    "top_missing_proof_ticker": "Top Missing Proof Ticker",
    "proposed_step204_decision": "Proposed Step204 Decision",
    "top_decision": "Top Decision",
    "top_headline": "Top Headline",
    "top_required_proof": "Proof Still Needed",
    "top_target_role": "Top Role",
    "top_tone": "Top Tone",
    "trade_notional_dollars": "Paper Notional",
    "unlock_status": "Unlock Status",
    "validation_note": "Validation Note",
    "volume_reaction_checked": "Volume Reaction Checked",
    "if_seed_approved_next_state": "If Seed Approved",
    "what_it_means": "What It Means",
    "what_to_check_first": "Check First",
    "what_to_collect": "What To Collect",
    "what_to_fix": "What To Fix",
    "what_to_do": "What To Do",
    "what_to_do_next": "What To Do Next",
    "what_is_needed": "What Is Needed",
    "will_fill_draft": "Will Fill Draft",
    "undecided_count": "Undecided",
    "workstream": "Workstream",
    "what_would_unlock": "What Would Unlock",
    "where_to_click": "Where To Click",
    "why": "Why",
    "why_this_ticker_first": "Why This Ticker First",
    "why_it_matters": "Why It Matters",
    "why_review": "Why Review",
    "why_this_lane": "Why This Lane",
    "why_this_target": "Why This Target",
    "worst_horizon": "Worst Holding Time",
    "worst_mean_ic": "Worst Signal Skill",
}

_FRIENDLY_COLUMNS.update({
    "active_files": "Active Files",
    "app_page": "Where To Click",
    "app_tab": "App Tab",
    "average_maturity_now_pct": "Average Build Level",
    "blocking_failure": "If This Fails",
    "current_bottleneck": "Current Bottleneck",
    "daily_runbook_steps": "Runbook Steps",
    "desk_name": "Desk",
    "do_not_do": "Do Not Do",
    "from_stage": "From",
    "gap_to_top_fund": "Gap To Top Fund",
    "gate_condition": "Gate Condition",
    "handoff_payload": "Handoff",
    "hard_gate": "Hard Gate",
    "hard_gate_count": "Hard Gates",
    "human_action": "Human Action",
    "if_gate_fails": "If Gate Fails",
    "input_files": "Input Files",
    "maturity_now_pct": "Build Level",
    "no_broker_connection": "No Broker Connection",
    "no_live_orders": "No Live Orders",
    "output_files": "Output Files",
    "output_to_read": "Output To Read",
    "owner_role": "Owner",
    "page_to_open": "Page To Open",
    "panel": "Panel",
    "primary_inputs": "Inputs",
    "primary_outputs": "Outputs",
    "plain_goal": "Goal",
    "related_steps": "Related Steps",
    "research_only": "Research Only",
    "run_order": "Order",
    "software_panels": "Software Panels",
    "stage_count": "Stages",
    "stage_name": "Stage",
    "stage_order": "Step",
    "system_action": "System Action",
    "to_stage": "To",
    "top_gap": "Top Gap",
    "weakest_stages": "Weakest Stages",
    "what_user_should_look_for": "What To Look For",
    "when_to_use": "When To Use",
})

_FRIENDLY_COLUMNS.update({
    "blocker_count": "Blockers",
    "blocker": "Blocker",
    "blocker_type": "Blocker Type",
    "can_take_new_risk": "Can Take New Risk",
    "current_state": "Current State",
    "done_when": "Done When",
    "execution_proof_count": "Execution Proof",
    "execution_read": "Execution Read",
    "fail_condition": "If It Fails",
    "first_action": "First Action",
    "first_page": "First Page",
    "input_contract": "Input Contract",
    "long_term_route": "Long Term",
    "medium_term_route": "Medium Term",
    "news_read": "News Read",
    "next_click": "Next Click",
    "next_click_count": "Next Clicks",
    "operating_mode": "Operating Mode",
    "option_route": "Option Route",
    "option_side": "Option Side",
    "output_contract": "Output Contract",
    "pass_condition": "Pass Condition",
    "proof_first_count": "Proof First",
    "proof_needed": "Proof Needed",
    "risk_blocked_count": "Risk Blocked",
    "risk_read": "Risk Read",
    "short_term_route": "Short Term",
    "source_trail": "Source Trail",
    "stage_contract_count": "Stage Contracts",
    "stock_or_etf_route": "Stock / ETF Route",
    "today_mode": "Today Mode",
    "what_to_read": "What To Read",
})

_FRIENDLY_COLUMNS.update({
    "allowed_now": "Allowed Now",
    "card_count": "Cards",
    "card_priority": "Priority",
    "first_answer": "First Answer",
    "forbidden_now": "Forbidden Now",
    "front_answer": "Answer",
    "plain_meaning": "Meaning",
    "proof_to_collect": "Proof To Collect",
    "quality_review_count": "QA Review",
    "source_summary": "Source Summary",
    "state_count": "States",
    "top_state": "Top State",
    "top_ticker": "Top Ticker",
    "unlock_condition": "Unlock Condition",
    "why_this_step": "Why This Step",
})

_FRIENDLY_COLUMNS.update({
    "acceptable_source": "Acceptable Source",
    "after_done": "After Done",
    "after_you_fill": "After You Fill",
    "current_ticker_answer": "Current Ticker Answer",
    "editable_file": "Editable File",
    "exact_file_to_edit": "File To Edit",
    "fields_to_fill": "Fields To Fill",
    "first_fields_to_fill": "First Fields To Fill",
    "first_proof_type": "First Proof Type",
    "first_question": "First Question",
    "first_source": "First Source",
    "good_sources": "Good Sources",
    "must_fill": "Must Fill",
    "open_proof_tasks": "Open Proof Tasks",
    "proof_task_count": "Proof Tasks",
    "proof_type": "Proof Type",
    "question_to_answer": "Question To Answer",
    "queue_rank": "Queue Rank",
    "task_rank": "Task Rank",
    "ticker_queue_count": "Ticker Queue",
    "what_counts_as_done": "What Counts As Done",
    "what_does_not_count": "What Does Not Count",
    "why_this_ticker_first": "Why This Ticker First",
})

_FRIENDLY_COLUMNS.update({
    "can_send_to_acceptance_bridge": "Can Send To Bridge",
    "counts_as": "Counts As",
    "does_not_count_as": "Does Not Count As",
    "first_fix": "First Fix",
    "first_state": "First State",
    "how_to_fill": "How To Fill",
    "missing_field": "Missing Field",
    "missing_field_rows": "Missing Fields",
    "proof_rows": "Proof Rows",
    "quality_score": "Quality Score",
    "quality_state": "Quality State",
    "ready_rows": "Ready Rows",
    "score_band": "Score Band",
    "source_examples": "Source Examples",
    "source_quality": "Source Quality",
    "weak_source_rows": "Weak Sources",
})

_FRIENDLY_COLUMNS.update({
    "after_filling": "After Filling",
    "bad_example": "Bad Example",
    "copy_sheet_rows": "Copy Sheet Rows",
    "estimated_minutes": "Estimated Minutes",
    "field_to_fill": "Field To Fill",
    "field_to_fill_count": "Fields To Fill",
    "fill_card_count": "Fill Cards",
    "fields_to_fill_now": "Fields To Fill Now",
    "fill_order": "Fill Order",
    "first_fields_to_fill": "First Fields To Fill",
    "first_source_to_open": "First Source To Open",
    "first_task": "First Task",
    "first_source_to_open": "First Source To Open",
    "good_example": "Good Example",
    "local_source_files": "Local Source Files",
    "missing_field_rows": "Missing Fields",
    "open_proof_rows": "Open Proof Rows",
    "plain_label": "Plain Label",
    "plain_task": "Plain Task",
    "source_to_open": "Source To Open",
    "ticker_rank": "Ticker Rank",
    "what_it_means": "What It Means",
    "what_to_type": "What To Type",
    "where_to_find_it": "Where To Find It",
    "why_this_blocks_progress": "Why This Blocks Progress",
})

_FRIENDLY_COLUMNS.update({
    "after_apply": "After Apply",
    "apply_decision": "Apply Decision",
    "apply_request_count": "Apply Requests",
    "applied_count": "Applied",
    "audit_event": "Audit Event",
    "backup_file": "Backup File",
    "missing_or_problem": "Missing Or Problem",
    "rejected_count": "Rejected",
    "source_url_or_file": "Source URL Or File",
    "template_rows": "Template Rows",
    "updated_fields": "Updated Fields",
    "user_entry_file": "User Entry File",
    "user_entry_rows": "User Entry Rows",
    "validation_state": "Validation State",
    "waiting_count": "Waiting",
    "will_apply": "Will Apply",
})

_FRIENDLY_COLUMNS.update({
    "accepted_evidence_ticker_count": "Tickers With Accepted Evidence",
    "bridge_conflict_count": "Bridge Conflicts",
    "bridge_conflicts": "Bridge Conflicts",
    "bridge_patch_rows": "Bridge Patch Rows",
    "closure_state": "Closure State",
    "fill_first_count": "Need Proof First",
    "intake_applied_rows": "Intake Applied Rows",
    "intake_apply_requests": "Intake Apply Requests",
    "missing_proof_rows": "Missing Proof Rows",
    "patch_ready_count": "Patch Ready",
    "plain_status": "Plain Status",
    "proof_progress_score": "Proof Progress Score",
    "quality_ready_rows": "Quality Ready Rows",
    "remaining_blocker": "Remaining Blocker",
    "stage_name": "Stage Name",
    "step204_accepted_rows": "Step204 Accepted Rows",
    "top_action": "Top Action",
    "top_action_ticker": "Top Action Ticker",
    "unblock_state": "Unblock State",
    "verified_source_rows": "Verified Source Rows",
})

_FRIENDLY_COLUMNS.update({
    "avg_causal_confidence": "Story Link Trust",
    "blocking_gates": "Blocking Checks",
    "bridge_decision": "Handoff Decision",
    "causal_chain_status": "Story Link Status",
    "causal_confidence_score": "Story Link Trust",
    "causal_confidence_0_100": "Story Link Trust",
    "causal_permission": "Can Use News For",
    "causal_thesis": "Why The Story Links",
    "current_final_gate": "Current Final Check",
    "directional_route": "Choice",
    "execution_permission": "Trading-Cost Permission",
    "execution_proof_count": "Trading-Cost Proof",
    "execution_proof_open": "Trading-Cost Proof Open",
    "execution_read": "Trading-Cost Read",
    "execution_status": "Trading-Cost Status",
    "first_blocking_gate": "First Blocking Check",
    "gate": "Check",
    "gate_condition": "Check Condition",
    "gate_or_blocker": "Check Or Blocker",
    "gate_state": "Check State",
    "hard_gate": "Hard Check",
    "hard_gate_count": "Hard Checks",
    "if_gate_fails": "If This Check Fails",
    "long_term_route": "Long-Term Choice",
    "medium_term_route": "Medium-Term Choice",
    "nav": "Account Value",
    "option_after_seed_approval": "Option Choice After Review",
    "option_route": "Option Choice",
    "output_contract": "Output Rule",
    "participation_rate_pct": "Trading Volume Share",
    "pit_quality_status": "Time-Stamp Quality",
    "pm_review_state": "Review State",
    "pm_review_status": "Review Status",
    "primary_route_now": "Choice Now",
    "readthrough_decision": "Related-Stock Decision",
    "repair_status": "Fix Status",
    "repair_type": "Fix Type",
    "risk_allows_pct": "Safety Allows",
    "risk_blocked_count": "Safety Blocked",
    "risk_level": "Safety Level",
    "risk_read": "Safety Read",
    "risk_seed_status": "Safety Review Status",
    "route_after_all_gates_clear": "Choice After All Checks Clear",
    "route_before_risk": "Choice Before Safety Check",
    "short_term_route": "Short-Term Choice",
    "signal": "Model Ingredient",
    "source_trail": "Data Used",
    "spread_bps": "Trading Spread",
    "stock_or_etf_route": "Stock / ETF Choice",
    "unlock_condition": "Can Move Forward When",
    "unlock_status": "Can Move Forward",
    "vehicle": "Stock / Option Choice",
    "where_to_click": "Where To Click",
})


def _friendly_value_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "No data"
    for raw, friendly in _FRIENDLY_PHRASES.items():
        text = text.replace(raw, friendly)
    upper = text.upper().strip()
    if upper == "DATA_GAP":
        return "missing data"
    if upper == "SIZE_DOWN":
        return "use smaller size"
    if upper == "NO_GO":
        return "not allowed"
    if upper == "BLOCKED":
        return "not ready yet"
    if "_" in text and upper == text:
        return text.replace("_", " ").title()
    return text.replace("_", " ")


def _humanize_key_value_text(text: str) -> str:
    raw = str(text or "").strip()
    if "=" not in raw:
        return raw

    parts = [part.strip() for part in re.split(r"[;|]", raw) if part.strip()]
    if len(parts) < 2 and raw.count("=") < 2:
        return raw

    lines = []
    for part in parts:
        if "=" not in part:
            if part:
                lines.append(part)
            continue
        key, value = part.split("=", 1)
        key_clean = key.strip().lower().replace(" ", "_")
        key_label = _FRIENDLY_KEYS.get(key_clean, key_clean.replace("_", " ").title())
        value_text = _friendly_value_text(value)
        if key_clean == "option_no_go_checks":
            lines.append(f"{key_label}: {value_text} checks still block options.")
        elif str(value).strip().upper() == "DATA_GAP":
            lines.append(f"{key_label}: missing data.")
        elif str(value).strip().upper() == "SIZE_DOWN":
            lines.append(f"{key_label}: use smaller size.")
        elif str(value).strip().upper() in {"NO_GO", "BLOCKED"}:
            lines.append(f"{key_label}: not allowed yet.")
        else:
            lines.append(f"{key_label}: {value_text}.")
    return " ".join(lines)


def _human_text(value, max_len: int | None = 220) -> str:
    if value is None:
        return "No data"
    try:
        if pd.isna(value):
            return "No data"
    except Exception:
        pass
    if isinstance(value, bool):
        return "Yes" if value else "No"

    text = str(value).strip()
    if not text:
        return "No data"

    upper = text.upper().strip()
    if upper in {"NAN", "NONE", "NULL"}:
        return "No data"
    if upper in {"TRUE", "FALSE"}:
        return "Yes" if upper == "TRUE" else "No"

    if "=" in text:
        text = _humanize_key_value_text(text)

    for raw, friendly in _FRIENDLY_PHRASES.items():
        text = text.replace(raw, friendly)

    readable_replacements = {
        "Annual volatility target": "The portfolio is moving more than the target",
        "Crisis-correlation volatility budget": "In a crisis, stocks may fall together",
        "Earnings gap-loss budget": "Earnings jump risk",
        "Macro scenario loss budget": "A bad macro scenario could lose too much",
        "Single-name tail-risk budget": "One stock could lose too much",
        "Backtest Credibility Center": "Old-data test quality",
        "Backtest credibility": "Old-data test quality",
        "Backtest": "Old-data test",
        "backtest": "old-data test",
        "Causal Confidence": "Story link trust",
        "Causal confidence": "Story link trust",
        "causal confidence": "story link trust",
        "Causal Status": "Story link status",
        "Causal status": "Story link status",
        "causal status": "story link status",
        "causal chain": "story link",
        "Causal chain": "Story link",
        "causal": "story link",
        "Causal": "Story link",
        "Correlation": "Moving together",
        "correlation": "moving together",
        "Execution": "Trading cost",
        "execution": "trading cost",
        "Factor": "Driver",
        "factor": "driver",
        "Final PM Gate": "Final review check",
        "final PM gate": "final review check",
        "gate": "check",
        "Gate": "Check",
        "Gross exposure": "Total account size used",
        "gross exposure": "total account size used",
        "gross": "account size",
        "IC decay": "model skill fading",
        "IC": "model follow-up score",
        "Kelly": "formula sizing",
        "NAV": "account value",
        "P&L": "paper gain/loss",
        "PM": "desk",
        "Proof Sharpe": "Trust score",
        "proof Sharpe": "trust score",
        "proof-adjusted Sharpe": "stricter score",
        "Proof-adjusted Sharpe": "Stricter score",
        "proof-adjusted": "trust score",
        "Sharpe": "performance score",
        "Signal": "Model ingredient",
        "signal": "model ingredient",
        "Source health": "Data source status",
        "source health": "data source status",
        "TCA": "trading cost",
        "VaR": "bad-day estimate",
        "CVaR": "very bad-day estimate",
        "Volatility": "Movement",
        "volatility": "movement",
        "alpha": "edge",
        "Alpha": "Edge",
        "directional route": "choice",
        "Directional route": "Choice",
        "drawdown": "drop from high",
        "Drawdown": "Drop from high",
        "horizon": "time frame",
        "Horizon": "Time frame",
        "liquidity": "trading volume",
        "Liquidity": "Trading volume",
        "look-ahead": "future-data mistake",
        "Look-Ahead": "Future-data mistake",
        "option route": "option choice",
        "Option route": "Option choice",
        "read-through": "related-stock effect",
        "Read-through": "Related-stock effect",
        "readthrough": "related-stock effect",
        "Readthrough": "Related-stock effect",
        "regime": "market mode",
        "Regime": "Market mode",
        "repair path": "fix path",
        "Repair path": "Fix path",
        "route": "choice",
        "Route": "Choice",
        "veto": "stop",
        "Veto": "Stop",
        "vehicle": "stock / option choice",
        "Vehicle": "Stock / option choice",
        "tail-risk": "large-loss risk",
        "risk gate": "risk check",
        "Risk gate": "Risk check",
        "paper sizing": "paper position size",
        "Paper sizing": "Paper position size",
        "event-time": "news-time",
        "Event-time": "News-time",
        "read-through": "linked-stock read",
        "Read-through": "Linked-stock read",
    }
    for raw, friendly in readable_replacements.items():
        text = text.replace(raw, friendly)

    if "_" in text and not text.startswith("http"):
        words = []
        for token in text.split():
            if "_" in token and token.upper() == token:
                words.append(token.replace("_", " ").title())
            elif "_" in token:
                words.append(token.replace("_", " "))
            else:
                words.append(token)
        text = " ".join(words)

    text = text.replace(" / ", " / ").replace(";", "; ")
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def _friendly_col(col) -> str:
    raw = str(col)
    if raw in _FRIENDLY_COLUMNS:
        return _FRIENDLY_COLUMNS[raw]
    return raw.replace("_", " ").strip().title()


def _humanize_df(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    work = work[[c for c in work.columns if not str(c).startswith("_")]]
    if max_rows is not None:
        work = work.head(max_rows)

    for col in work.columns:
        if work[col].dtype == "bool":
            work[col] = work[col].map(lambda x: "Yes" if bool(x) else "No")
        elif work[col].dtype == "object":
            work[col] = work[col].map(lambda x: _human_text(x, max_len=260))

    work = work.rename(columns={c: _friendly_col(c) for c in work.columns})
    return work


def _plain_status(value, default: str = "No data") -> str:
    text = _human_text(value)
    if text == "No data" and default:
        return default
    return text


def _clean_display(value, fallback: str = "No data") -> str:
    text = _human_text(value)
    return fallback if text == "No data" else text


def _show_status_table(df: pd.DataFrame, status_cols=None, height: int = 420):
    if df is None or df.empty:
        st.info("No rows to show yet.")
        return

    display = _humanize_df(df)
    st.dataframe(display, width="stretch", hide_index=True, height=height)


def _render_html(markup: str):
    cleaned = "\n".join(line.strip() for line in str(markup).strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def _simple_card(title: str, value: str, note: str = "", accent: str = "#111827"):
    _render_html(
        f"""
        <div style="
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-top: 3px solid {accent};
            border-radius: 10px;
            padding: 14px 16px 12px 16px;
            min-height: 108px;
            box-shadow: 0 1px 4px rgba(0,0,0,.06);
            transition: box-shadow .2s;
        ">
          <div style="font-size:10.5px; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">{_esc(title)}</div>
          <div style="font-size:22px; color:#0f172a; font-weight:800; line-height:1.2; margin-top:7px; letter-spacing:-0.3px;">{_esc(value)}</div>
          <div style="font-size:11.5px; color:#64748b; line-height:1.4; margin-top:8px;">{_esc(note)}</div>
        </div>
        """
    )


_SECTION_DEPTH_CONFIG = {
    "Today": {
        "title": "What should I do first today?",
        "question": "Which page should I open first, and what should I avoid?",
        "why": "This page gives the daily order. It stops you from jumping into a stock or option before the basic checks are done.",
        "rules": [
            "Risk and missing data come before new ideas.",
            "A news shock is only useful after source, timing, and price reaction are checked.",
            "A price move matters more when volume and spread are clean.",
        ],
        "changes": "The answer changes when a blocker clears, a new alert arrives, or the daily runner creates fresher files.",
        "not_allowed": "Do not use this page as a trade list. It is the order of research work.",
        "sources": [
            ("pm_morning_brief_state.json", "Daily answer"),
            ("quant_fund_flow_next_clicks.csv", "Next-click path"),
            ("daily_workflow_steps.csv", "Daily steps"),
            ("desk_monitor_events.csv", "Monitor alerts"),
            ("daily_workflow_queue.csv", "Ticker work queue"),
            ("action_readiness_monitor.csv", "Ticker ready checks"),
            ("proof_queue_daily_plan.csv", "Missing proof and fixes"),
        ],
    },
    "Ideas": {
        "title": "What can I do with each ticker?",
        "question": "Should this ticker be watched, researched as stock, researched as call, researched as put, or skipped?",
        "why": "This page separates short, medium, and long-term ideas first. Then it explains whether stock or options make sense.",
        "rules": [
            "Choose short, medium, or long-term first.",
            "Safety can stop a high score.",
            "Calls and puts need safety, event proof, price level, and trading-cost proof.",
        ],
        "changes": "The choice changes when price hits the watch level, safety improves, event proof improves, or option cost becomes acceptable.",
        "not_allowed": "Do not treat an option label as permission to trade. Options are research-only unless every check clears.",
        "sources": [
            ("horizon_vehicle_summary.csv", "Time frame and stock/option choice"),
            ("options_playbook.csv", "Call and put playbook"),
            ("options_execution_route_matrix.csv", "Option cost and no-go checks"),
            ("daily_workflow_queue.csv", "Daily ticker priority"),
        ],
    },
    "News": {
        "title": "Which news matters?",
        "question": "Who may benefit, who may get hurt, and is the story proven?",
        "why": "A headline alone is not enough. This page turns news into a clear story: affected stock, related stocks, industry chain, and proof needed.",
        "rules": [
            "Direct ticker impact comes before related-stock effects.",
            "Good news can hurt weak or expensive competitors.",
            "Price and volume reaction must confirm the story.",
        ],
        "changes": "The news view changes when the source is verified, timing is proven, price reacts, or the industry chain map adds better links.",
        "not_allowed": "Do not buy from a headline alone. No source, timing, and price proof means watch only.",
        "sources": [
            ("event_readthrough_event_summary.csv", "Headline summary"),
            ("news_impact_targets.csv", "Help and hurt targets"),
            ("news_supply_chain_readthrough.csv", "Industry-chain map"),
            ("event_causal_validation_queue.csv", "Proof list"),
        ],
    },
    "Risk": {
        "title": "Is it safe to add anything?",
        "question": "Can the account add a new idea, or should it stay smaller first?",
        "why": "This page is the safety check. If this says no, every new stock or option idea must wait.",
        "rules": [
            "If the account is already too risky, new ideas wait.",
            "A single stock can be not ready yet even when the market looks fine.",
            "Options cannot override safety limits.",
        ],
        "changes": "The answer changes when the account gets smaller, movement calms, earnings risk passes, or a bad-market test improves.",
        "not_allowed": "Do not search for calls, puts, or bigger size while risk says reduce or use smaller size.",
        "sources": [
            ("risk_desk_overview.json", "Portfolio risk answer"),
            ("portfolio_var_cvar_summary.csv", "Loss estimate"),
            ("single_name_risk_budget.csv", "Single-stock risk"),
            ("crisis_correlation_stress.csv", "Crisis correlation check"),
            ("depth5_portfolio_optimizer_v2.csv", "Allowed portfolio size"),
            ("depth5_execution_liquidity_desk.csv", "Trading cost and liquidity"),
        ],
    },
    "Performance": {
        "title": "Can I trust the results?",
        "question": "Is the model actually good, or does it only look good on old data?",
        "why": "This page tells you whether the result is strong enough to trust. A pretty score is not enough.",
        "rules": [
            "Old results are not enough by themselves.",
            "The model must prove it works on newer data too.",
            "Trading cost and failed fills must be counted.",
        ],
        "changes": "The answer changes when more real follow-up data arrives and the model still works after costs.",
        "not_allowed": "Do not size from old test results until the model proves itself on newer data.",
        "sources": [
            ("sharpe_target4_state.json", "Sharpe target state"),
            ("depth5_signal_ic_decay_failure_lab.csv", "Signal skill and decay"),
            ("backtest_credibility_scorecard.csv", "Backtest credibility"),
            ("execution_cost_model.csv", "Trading-cost model"),
        ],
    },
    "Live / Paper": {
        "title": "How is the paper account doing?",
        "question": "Are the paper ideas helping or hurting?",
        "why": "This page tracks the paper account and gives feedback. It does not connect to a broker.",
        "rules": [
            "Paper gain/loss is feedback, not proof by itself.",
            "Manual account value must be separated from real broker trading.",
            "Critical monitor events stop new risk until reviewed.",
        ],
        "changes": "The answer changes when account value updates, a position hits stop or target, a monitor event appears, or more follow-up evidence arrives.",
        "not_allowed": "Do not treat paper tracking as real trading. There is no broker connection and no live order path.",
        "sources": [
            ("paper_sim_summary.csv", "Paper account summary"),
            ("paper_sim_positions.csv", "Open paper positions"),
            ("live_nav_curve.csv", "Manual account value, if filled"),
            ("desk_monitor_events.csv", "Live-style monitor alerts"),
            ("decision_memory_state.json", "Decision feedback memory"),
            ("live_ic_observation_state.json", "Live follow-up checks"),
        ],
    },
}


def _section_source_health(files: list[tuple[str, str]]) -> str:
    parts = []
    for fname, label in files[:6]:
        path = ROOT / fname
        if not path.exists() or path.stat().st_size <= 10:
            parts.append(f"{label}: missing")
            continue
        rows = ""
        try:
            if fname.endswith(".csv"):
                df = safe_csv(path)
                rows = f", {len(df)} rows" if not df.empty else ", empty"
            elif fname.endswith(".json"):
                data = safe_json(path)
                rows = f", {len(data)} fields" if data else ", empty"
        except Exception:
            rows = ""
        parts.append(f"{label}: {file_age_str(path)}{rows}")
    return " | ".join(parts) if parts else "No source files listed."


def _render_section_depth(section: str):
    cfg = _SECTION_DEPTH_CONFIG.get(section)
    if not cfg:
        return

    rules = cfg.get("rules", [])
    rule_items = "".join(
        f"<li style='margin-bottom:5px;'>{_esc(rule)}</li>"
        for rule in rules[:4]
    )
    source_line = _section_source_health(cfg.get("sources", []))
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-radius:10px; padding:16px 18px; margin:10px 0 18px 0;">
          <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start;">
            <div>
              <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">How to read this page</div>
              <div style="font-size:22px; color:#111827; font-weight:950; line-height:1.2; margin-top:5px;">{_esc(cfg.get("title"))}</div>
              <div style="font-size:14px; color:#111827; line-height:1.45; margin-top:8px;"><b>The question:</b> {_esc(cfg.get("question"))}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:7px;">{_esc(cfg.get("why"))}</div>
            </div>
          </div>
          <div style="display:grid; grid-template-columns:1.1fr 1fr 1fr; gap:14px; margin-top:14px;">
            <div style="border-top:3px solid #111827; background:#f9fafb; border-radius:8px; padding:12px 13px;">
              <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">Simple rules</div>
              <ul style="font-size:13px; color:#111827; line-height:1.42; padding-left:18px; margin:8px 0 0 0;">{rule_items}</ul>
            </div>
            <div style="border-top:3px solid #334155; background:#f9fafb; border-radius:8px; padding:12px 13px;">
              <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">When this can change</div>
              <div style="font-size:13px; color:#111827; line-height:1.45; margin-top:8px;">{_esc(cfg.get("changes"))}</div>
            </div>
            <div style="border-top:3px solid #991b1b; background:#f9fafb; border-radius:8px; padding:12px 13px;">
              <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">Do not do this</div>
              <div style="font-size:13px; color:#111827; line-height:1.45; margin-top:8px;">{_esc(cfg.get("not_allowed"))}</div>
            </div>
          </div>
        </div>
        """
    )
    with st.expander("Show where this page gets its data", expanded=False):
        st.caption(source_line)


_SUBTAB_DEPTH_CONFIG = {
    ("Today", "Daily steps"): {
        "purpose": "Shows the order of work for the day.",
        "read": "Read the station, then the action, then the next page.",
        "decision": "If a step says review or fix, finish that before moving to ticker ideas.",
        "stop": "Do not treat the queue as a buy list.",
        "sources": [("daily_workflow_steps.csv", "Daily workflow"), ("daily_workflow_queue.csv", "Ticker queue")],
    },
    ("Today", "Blockers"): {
        "purpose": "Explains what is stopping action.",
        "read": "Start with highest priority blockers and the where-to-click field.",
        "decision": "A blocker must be cleared before the ticker can move forward.",
        "stop": "Do not skip a blocker because the ticker looks exciting.",
        "sources": [("quant_fund_flow_blocker_queue.csv", "Blocker queue"), ("proof_queue_daily_plan.csv", "Proof plan")],
    },
    ("Today", "Ticker states"): {
        "purpose": "Shows where each ticker sits in the workflow.",
        "read": "Read current state, first blocker, next click, then trigger.",
        "decision": "Only tickers with clear next action and clean risk can move to Ideas.",
        "stop": "Do not use a ticker with missing source or risk proof.",
        "sources": [("quant_fund_flow_current_state.csv", "Ticker state"), ("action_readiness_monitor.csv", "Readiness")],
    },
    ("Today", "Alerts"): {
        "purpose": "Shows what changed or broke today.",
        "read": "Read severity first, then ticker, then action.",
        "decision": "Critical alerts move the workflow back to Risk or System.",
        "stop": "Do not ignore critical alerts when reading Ideas.",
        "sources": [("desk_monitor_events.csv", "Monitor events"), ("daily_alerts.json", "Alert list")],
    },
    ("Today", "Source files"): {
        "purpose": "Shows the exact files behind the Today page.",
        "read": "Check missing, stale, or empty files first.",
        "decision": "If source files are stale, rerun the daily update before trusting the page.",
        "stop": "Do not over-read stale data.",
        "sources": [("pm_morning_brief_state.json", "Daily answer"), ("quant_fund_flow_next_clicks.csv", "Next clicks")],
    },
    ("Ideas", "Time-frame table"): {
        "purpose": "Separates short, medium, and long-term use cases.",
        "read": "Read best horizon, action, trigger, blocker, and unlock checklist.",
        "decision": "A ticker must first have a valid time frame before stock/call/put matters.",
        "stop": "Do not turn a long-term thesis into a short-term trade just because price moved.",
        "sources": [("horizon_vehicle_summary.csv", "Horizon summary"), ("timeframe_decision_matrix.csv", "Time-frame matrix")],
    },
    ("Ideas", "Option choices"): {
        "purpose": "Explains whether call, put, spread, hedge, or no option makes sense.",
        "read": "Read permission, side, blocker, trigger, and trading-cost proof.",
        "decision": "Options need safety, event proof, price trigger, and execution proof.",
        "stop": "No option label is permission to trade.",
        "sources": [("options_playbook.csv", "Call/put playbook"), ("options_execution_route_matrix.csv", "Option route")],
    },
    ("Ideas", "Daily queue"): {
        "purpose": "Shows which tickers deserve attention today.",
        "read": "Read priority, best horizon, risk action, what to watch, and what would change.",
        "decision": "Only top-priority tickers with a clean next condition should get time first.",
        "stop": "Do not read low-priority tickers before risk and proof items.",
        "sources": [("daily_workflow_queue.csv", "Daily queue")],
    },
    ("Ideas", "Final review"): {
        "purpose": "Final permission check before paper research.",
        "read": "Read final permission, primary route, first blocker, and max paper weight.",
        "decision": "If final permission is not clear, the ticker stays research-only.",
        "stop": "Do not use max paper weight as a target when blockers remain.",
        "sources": [("institutional_promotion_gate.csv", "Final permission")],
    },
    ("Ideas", "Older playbook"): {
        "purpose": "Keeps older logic available for comparison.",
        "read": "Use it to understand history, not as the main decision.",
        "decision": "If old and new views disagree, the newer risk-aware view wins.",
        "stop": "Do not let legacy outputs override the current risk gate.",
        "sources": [("strategy_route_playbook.csv", "Older strategy route"), ("timeframe_decision_matrix.csv", "Older time frame")],
    },
    ("Ideas", "Source files"): {
        "purpose": "Shows the files behind Ideas.",
        "read": "Check if the option, horizon, and final permission files are fresh.",
        "decision": "Missing option files mean no option conclusion.",
        "stop": "Do not guess calls or puts from missing option data.",
        "sources": [("horizon_vehicle_summary.csv", "Horizon"), ("options_playbook.csv", "Options"), ("institutional_promotion_gate.csv", "Permission")],
    },
    ("News", "Industry chain"): {
        "purpose": "Maps a headline into related winners, losers, suppliers, customers, and peers.",
        "read": "Read source ticker, target ticker, relationship, theme, and proof.",
        "decision": "A related stock only matters if the chain link and price reaction are proven.",
        "stop": "Do not buy a peer just because it is in the same theme.",
        "sources": [("event_causal_chain_map.csv", "Chain map"), ("news_supply_chain_readthrough.csv", "Supply chain")],
    },
    ("News", "Proof queue"): {
        "purpose": "Lists what must be proven before a headline becomes usable.",
        "read": "Read source, timestamp, affected target, price reaction, and missing proof.",
        "decision": "No proof means watch-only.",
        "stop": "Do not turn unverified news into a trade idea.",
        "sources": [("event_causal_validation_queue.csv", "Proof queue"), ("event_time_truth_ledger.csv", "Timing ledger")],
    },
    ("News", "Ticker lookup"): {
        "purpose": "Answers: what news is tied to this ticker?",
        "read": "Pick the ticker, then read headline direction, role, why, and proof.",
        "decision": "Direct impact matters more than weak related-stock mapping.",
        "stop": "Do not assume every headline attached to a ticker is important.",
        "sources": [("ticker_decision_room_news.csv", "Ticker news")],
    },
    ("News", "More story cards"): {
        "purpose": "Shows more headline stories in human form.",
        "read": "Read help, hurt, desk decision, and proof needed.",
        "decision": "Only stories with source, time, target, and price proof move forward.",
        "stop": "Do not chase headlines without proof.",
        "sources": [("event_readthrough_event_summary.csv", "Event summary"), ("news_impact_targets.csv", "Impact targets")],
    },
    ("News", "Source files"): {
        "purpose": "Shows the files behind News.",
        "read": "Check event, target, chain, and proof files.",
        "decision": "If these are missing or stale, news logic becomes hypothesis only.",
        "stop": "Do not trust stale headlines.",
        "sources": [("event_readthrough_event_summary.csv", "Events"), ("event_causal_validation_queue.csv", "Proof")],
    },
    ("Risk", "Can I add?"): {
        "purpose": "Answers whether the account can add any new risk.",
        "read": "Read the front answer, breach list, repair path, and unlock ladder.",
        "decision": "If the answer says no, Ideas and Options stay research-only.",
        "stop": "Do not look for bigger size while this page says reduce or wait.",
        "sources": [("risk_desk_overview.json", "Risk answer"), ("risk_desk_breach_table.csv", "Breaches")],
    },
    ("Risk", "Why no?"): {
        "purpose": "Explains the evidence gap behind a blocked idea.",
        "read": "Read proof quality, source trail, and missing risk-book fields.",
        "decision": "A blocked idea moves only after proof is filled and accepted.",
        "stop": "Do not override missing proof manually without recording why.",
        "sources": [("proof_queue_daily_plan.csv", "Proof plan"), ("risk_repair_recommendation_board.csv", "Risk repair")],
    },
    ("Risk", "What to fix"): {
        "purpose": "Turns risk problems into repair work.",
        "read": "Read the first repair ticker, required proof, and acceptance gate.",
        "decision": "Fix data and risk-book fields before changing idea status.",
        "stop": "Do not add new research layers before the first risk repair is clear.",
        "sources": [("data_repair_priority_board.csv", "Data repair"), ("risk_seed_pm_review_intake.csv", "Risk intake")],
    },
    ("Risk", "Limits"): {
        "purpose": "Shows portfolio-level and single-rule limits.",
        "read": "Read used percent, status, action if breached, and loss estimate.",
        "decision": "Breached limits force smaller size or no new exposure.",
        "stop": "Do not treat an OK ticker as OK if portfolio limits are breached.",
        "sources": [("institutional_risk_budget_summary.csv", "Risk budgets"), ("portfolio_var_cvar_summary.csv", "Loss estimate")],
    },
    ("Risk", "Stocks"): {
        "purpose": "Shows ticker-level safety.",
        "read": "Read weight, earnings gap, liquidity, sector, and final risk action.",
        "decision": "Ticker risk can block a ticker even if the portfolio is acceptable.",
        "stop": "Do not size all stocks equally.",
        "sources": [("final_risk_gate.csv", "Final stock risk"), ("single_name_risk_budget.csv", "Single-name budget")],
    },
    ("Risk", "Exposure"): {
        "purpose": "Shows where the account is concentrated.",
        "read": "Read sector, benchmark difference, factor beta, and correlation.",
        "decision": "Too much one-sector or one-factor exposure blocks more names from that group.",
        "stop": "Do not call ten semiconductor stocks diversified.",
        "sources": [("sector_active_exposure.csv", "Sector exposure"), ("factor_exposure_decomposition.csv", "Factor exposure")],
    },
    ("Risk", "Stress"): {
        "purpose": "Shows what happens in bad markets.",
        "read": "Read scenario impact, drawdown circuit, movement target, and crisis crowding.",
        "decision": "A bad stress result means shrink, hedge, or wait.",
        "stop": "Do not rely on normal correlation during crisis tests.",
        "sources": [("macro_scenario_stress.csv", "Scenario stress"), ("crisis_correlation_stress.csv", "Crisis correlation")],
    },
    ("Risk", "Portfolio Builder"): {
        "purpose": "Shows the target portfolio after safety rules.",
        "read": "Read current weight, target weight, final risk action, and binding constraints.",
        "decision": "The builder cannot add what the risk gate rejects.",
        "stop": "Do not use optimizer weight before checking constraints.",
        "sources": [("institutional_portfolio_construction_plan.csv", "Construction"), ("institutional_optimizer_bridge.csv", "Optimizer")],
    },
    ("Risk", "Source Files"): {
        "purpose": "Shows all files behind Risk.",
        "read": "Check freshness and missing files before trusting advanced safety tables.",
        "decision": "Missing risk files mean risk answer is incomplete.",
        "stop": "Do not repair ideas before repairing missing risk data.",
        "sources": [("risk_desk_overview.json", "Risk answer"), ("portfolio_constraint_matrix.csv", "Constraints")],
    },
    ("Live / Paper", "Account monitor"): {
        "purpose": "Tracks paper account and manual account feedback.",
        "read": "Read account value, positions, alert events, and feedback memory.",
        "decision": "Paper results are feedback; they do not prove the strategy by themselves.",
        "stop": "Do not confuse paper monitoring with live broker trading.",
        "sources": [("paper_sim_summary.csv", "Paper summary"), ("live_nav_curve.csv", "Manual account value")],
    },
    ("Live / Paper", "Trade notes"): {
        "purpose": "Stores notes, journal entries, and lessons.",
        "read": "Read thesis, reason for entry, reason for exit, and lesson.",
        "decision": "A lesson matters only after it is tied back to a signal or risk condition.",
        "stop": "Do not change model weights from one anecdote.",
        "sources": [("paper_portfolio_ledger.csv", "Paper ledger"), ("learning_attribution_summary.csv", "Learning")],
    },
    ("System", "Update / Fix"): {
        "purpose": "Runs updates and shows what is broken.",
        "read": "Read last run, failed steps, stale files, then fix list.",
        "decision": "Failed or stale system files must be fixed before trusting every panel.",
        "stop": "Do not diagnose strategy quality from stale outputs.",
        "sources": [("run_daily_all_log.csv", "Run log"), ("canyon_file_manifest.csv", "File manifest")],
    },
    ("System", "Where Things Are"): {
        "purpose": "Shows where old tools now live.",
        "read": "Read the current page, earlier tools inside, and what question it answers.",
        "decision": "Use main pages first; open deep tools only when needed.",
        "stop": "Do not put every old tab back into the top navigation.",
        "sources": [("HANDOFF_CANYON_V9.md", "Architecture notes")],
    },
    ("System", "Tools"): {
        "purpose": "Holds helper tools and command utilities.",
        "read": "Use only when you need a specific tool.",
        "decision": "Tools support the workflow; they are not the workflow.",
        "stop": "Do not start here unless something is broken.",
        "sources": [("system_health_check.csv", "Health check")],
    },
    ("System", "Proof To Fill"): {
        "purpose": "Shows missing proof that blocks decisions.",
        "read": "Read proof item, ticker, source needed, and acceptance rule.",
        "decision": "A blocked idea clears only after proof is filled and accepted.",
        "stop": "Do not clear proof by changing text only.",
        "sources": [("proof_queue_daily_plan.csv", "Proof plan"), ("pm_evidence_review_triage.csv", "Evidence review")],
    },
    ("System", "Data Fix"): {
        "purpose": "Shows stale, missing, or unreliable data to repair.",
        "read": "Read data reliability, repair center, risk-book seed, and review gates.",
        "decision": "Data repair comes before better signals.",
        "stop": "Do not deepen strategy logic on bad inputs.",
        "sources": [("data_reliability_state.json", "Reliability"), ("data_repair_priority_board.csv", "Repair board")],
    },
    ("System", "All Files"): {
        "purpose": "Shows the full file inventory.",
        "read": "Check status, age, rows, tickers, and empty files.",
        "decision": "Use this when a page looks missing or stale.",
        "stop": "Do not read raw files as the main product experience.",
        "sources": [("canyon_file_manifest.csv", "File manifest")],
    },
}


def _render_subtab_depth(page: str, tab: str):
    cfg = _SUBTAB_DEPTH_CONFIG.get((page, tab))
    if not cfg:
        return
    source_line = _section_source_health(cfg.get("sources", []))
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #111827; border-radius:9px; padding:14px 16px; margin:10px 0 16px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">This small tab's logic</div>
          <div style="font-size:20px; color:#111827; font-weight:950; line-height:1.2; margin-top:5px;">{_esc(cfg.get("purpose"))}</div>
          <div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin-top:12px;">
            <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:10px 11px;">
              <div style="font-size:11px; color:#6b7280; font-weight:900; text-transform:uppercase;">Read first</div>
              <div style="font-size:13px; color:#111827; line-height:1.4; margin-top:6px;">{_esc(cfg.get("read"))}</div>
            </div>
            <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:10px 11px;">
              <div style="font-size:11px; color:#6b7280; font-weight:900; text-transform:uppercase;">Decision rule</div>
              <div style="font-size:13px; color:#111827; line-height:1.4; margin-top:6px;">{_esc(cfg.get("decision"))}</div>
            </div>
            <div style="background:#fff7f7; border:1px solid #e5e7eb; border-radius:8px; padding:10px 11px;">
              <div style="font-size:11px; color:#991b1b; font-weight:900; text-transform:uppercase;">Stop here if</div>
              <div style="font-size:13px; color:#111827; line-height:1.4; margin-top:6px;">{_esc(cfg.get("stop"))}</div>
            </div>
          </div>
        </div>
        """
    )
    with st.expander(f"Data behind {page} / {tab}", expanded=False):
        st.caption(source_line)


def _latest_nav_row(*frames):
    for df in frames:
        if df is not None and not df.empty:
            work = df.copy()
            if "date" in work.columns:
                work["_date_sort"] = pd.to_datetime(work["date"], errors="coerce")
                work = work.sort_values("_date_sort")
            return work.iloc[-1], df
    return pd.Series(dtype=object), pd.DataFrame()


def _render_nav_chart(nav_df, source_label: str):
    if nav_df.empty:
        st.info("No account-value curve yet. Run the daily system or fill the manual account-value file.")
        return

    show = nav_df.copy()
    if "date" in show.columns:
        show["date"] = pd.to_datetime(show["date"], errors="coerce")
    if "nav" in show.columns and _PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=show["date"] if "date" in show.columns else show.index,
                y=pd.to_numeric(show["nav"], errors="coerce"),
                mode="lines+markers",
                name=source_label,
                line=dict(color="#111827", width=2.4),
            )
        )
        if "hwm" in show.columns:
            fig.add_trace(
                go.Scatter(
                    x=show["date"] if "date" in show.columns else show.index,
                    y=pd.to_numeric(show["hwm"], errors="coerce"),
                    mode="lines",
                    name="High-water mark",
                    line=dict(color="#64748b", width=1.4, dash="dash"),
                )
            )
        fig.update_layout(
            height=310,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=24, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eef2f7")
        fig.update_yaxes(showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig, width="stretch")
    else:
        nav_cols = [c for c in ["date", "nav", "account_equity", "daily_return", "hwm", "drawdown_pct", "cumulative_return_pct", "source"] if c in show.columns]
        _show_status_table(show[nav_cols].tail(20) if nav_cols else show.tail(20), [], height=320)


def _render_position_cards(positions: pd.DataFrame):
    if positions.empty:
        st.info("No open paper positions yet.")
        return

    work = positions.copy()
    if "unrealised_pnl" in work.columns:
        work["_pnl_sort"] = pd.to_numeric(work["unrealised_pnl"], errors="coerce").fillna(0)
        work = work.sort_values("_pnl_sort")

    st.markdown("##### Open paper positions")
    parts = ['<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; margin:8px 0 18px 0;">']
    for _, row in work.head(8).iterrows():
        pnl = _to_float(row.get("unrealised_pnl"), 0.0) or 0.0
        pnl_pct = _pct_display(row.get("unrealised_pct"))
        accent = "#166534" if pnl > 0 else "#991b1b" if pnl < 0 else "#334155"
        status = "Winning" if pnl > 0 else "Losing" if pnl < 0 else "Flat"
        parts.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid {accent}; border-radius:8px; padding:14px 15px; min-height:205px;">
              <div style="display:flex; justify-content:space-between; gap:10px;">
                <div style="font-size:22px; font-weight:850; color:#111827;">{_esc(row.get("ticker"), "")}</div>
                <div style="font-size:12px; font-weight:850; color:{accent};">{status}</div>
              </div>
              <div style="font-size:12px; color:#6b7280; margin-top:3px;">{_esc(row.get("sector"), "No sector")}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; display:grid; grid-template-columns:1fr 1fr; gap:9px;">
                <div><div style="font-size:11px; color:#6b7280;">P&L</div><div style="font-size:17px; font-weight:850; color:{accent};">{_money(pnl)}</div></div>
                <div><div style="font-size:11px; color:#6b7280;">Return</div><div style="font-size:17px; font-weight:850; color:{accent};">{_esc(pnl_pct)}</div></div>
                <div><div style="font-size:11px; color:#6b7280;">Current</div><div style="font-size:14px; color:#111827;">{_money(row.get("current_price"))}</div></div>
                <div><div style="font-size:11px; color:#6b7280;">Entry</div><div style="font-size:14px; color:#111827;">{_money(row.get("entry_price"))}</div></div>
              </div>
              <div style="border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px; font-size:12px; color:#4b5563; line-height:1.35;">
                Stop: {_money(row.get("stop_price"))} · Target: {_money(row.get("target_price"))}
              </div>
            </div>
            """
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

    cols = [c for c in [
        "ticker", "sector", "entry_date", "entry_price", "current_price", "shares",
        "market_value", "unrealised_pnl", "unrealised_pct", "stop_price",
        "target_price", "last_updated",
    ] if c in work.columns]
    with st.expander("Open full position table", expanded=False):
        _show_status_table(work[cols] if cols else work, [], height=480)


def _live_plain(value, max_len: int | None = 220) -> str:
    text = _human_text(value, max_len=None)
    replacements = {
        "REDUCE_ONLY": "risk reduction only",
        "Reduce Only": "risk reduction only",
        "SIZE_DOWN": "use smaller size",
        "Size Down": "use smaller size",
        "APPLY_VOL_MULTIPLIER": "use the lower risk budget",
        "Apply Vol Multiplier": "use the lower risk budget",
        "SIZE_DOWN_OR_REDUCE_ONLY": "use smaller size or reduce risk",
        "Size Down Or Reduce Only": "use smaller size or reduce risk",
        "DATA_GAP": "missing data",
        "Data Gap": "missing data",
        "LIVE_IC_ACTIVE": "live follow-up tracking is active",
        "Live Ic Active": "live follow-up tracking is active",
        "PENDING_FORWARD_RETURNS": "waiting for future price data",
        "Pending Forward Returns": "waiting for future price data",
        "EXECUTION_BLOCKED_OR_REPAIR_FIRST": "trading-cost check needs repair first",
        "Execution Blocked Or Repair First": "trading-cost check needs repair first",
        "RISK_LIMIT_BREACH": "risk limit warning",
        "Risk Limit Breach": "risk limit warning",
        "NEWS_SHOCK": "news shock",
        "News Shock": "news shock",
        "PRICE_BREAK": "price break",
        "Price Break": "price break",
        "CRITICAL": "critical",
        "Critical": "critical",
        "WARNING": "warning",
        "Warning": "warning",
        "INFO": "info",
        "Info": "info",
        "IC": "model follow-up score",
        "master:SIZE_DOWN": "portfolio size: use smaller size",
        "single:REDUCE_ONLY": "single-stock risk: reduce only",
        "earnings_gap:REDUCE_ONLY": "earnings gap risk: reduce only",
        "kelly:SIZE_DOWN": "signal-based size: smaller size",
        "sector:SIZE_DOWN": "sector crowding: smaller size",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = text.replace("master:", "portfolio size: ")
    text = text.replace("single:", "single-stock risk: ")
    text = text.replace("earnings gap:", "earnings risk: ")
    text = text.replace("earnings_gap:", "earnings risk: ")
    text = text.replace("kelly:", "signal size: ")
    text = text.replace("sector:", "sector crowding: ")
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text or "No data"


def _live_accent(value) -> str:
    text = str(value or "").upper()
    if any(x in text for x in ["CRITICAL", "REDUCE", "BLOCK", "DATA_GAP", "LOSING"]):
        return "#991b1b"
    if any(x in text for x in ["WARNING", "WAIT", "PENDING", "REVIEW", "PAPER"]):
        return "#334155"
    if any(x in text for x in ["READY", "WINNING", "CLEAR", "ACTIVE", "OK"]):
        return "#166534"
    return "#111827"


def _render_live_paper_workflow_board(
    live_connected: bool,
    critical_events: int,
    total_events: int,
    open_positions: int,
    paper_row: pd.Series,
    latest_nav: pd.Series,
    live_ic_state: dict,
    execution_state: dict,
    decision_state: dict,
):
    st.markdown("#### How to use this page today")
    if critical_events:
        first_move = "Read the critical monitor events before looking at any new idea."
        accent = "#991b1b"
    elif open_positions:
        first_move = "Check open paper positions first, then review signal feedback."
        accent = "#334155"
    else:
        first_move = "No open paper book pressure. Use this page as feedback, not as a trade list."
        accent = "#166534"

    live_line = (
        "Manual account-value tracking is active, but it is still manual tracking only."
        if live_connected
        else "Real account value is not connected. This page is paper-mode unless you fill the manual account-value file."
    )
    pnl_line = f"Paper gain/loss is {_money(paper_row.get('total_pnl'))}, return {_pct_display(paper_row.get('total_return_pct'))}."
    nav_line = f"Latest drop from high is {_pct_display(latest_nav.get('drawdown_pct'))}; daily return is {_pct_display(latest_nav.get('daily_return'))}."
    proof_line = (
        f"Decision memory has {decision_state.get('ready_forward_observations', 0)} ready grades and "
        f"{decision_state.get('pending_forward_observations', 0)} still waiting."
    )

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:9px; padding:16px 18px; margin:8px 0 15px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">Simple account answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(first_move)}</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.55; margin-top:9px;">
            {_esc(live_line)} {_esc(pnl_line)} {_esc(nav_line)} {_esc(proof_line)}
          </div>
        </div>
        """
    )

    workflow = [
        ("1. Account mode", "Confirm whether this is paper-only or manual account-value tracking.", live_line, "#111827"),
        ("2. Monitor warnings", f"{critical_events} critical / {total_events} total", "If critical > 0, protect first and do not add exposure.", "#991b1b" if critical_events else "#166534"),
        ("3. Open positions", str(open_positions), "Review worst losers, stops, and target distance before new research.", "#334155" if open_positions else "#166534"),
        ("4. Model feedback", str(live_ic_state.get("live_ic_windows", 0)), "The model only proves itself after future prices arrive.", "#334155"),
        ("5. Trading-cost check", _live_plain(execution_state.get("status"), 90), "Trading cost and volume can block paper ideas.", _live_accent(execution_state.get("status"))),
    ]
    cols = st.columns(5)
    for col, (title, value, note, card_accent) in zip(cols, workflow):
        with col:
            _simple_card(title, value, note, card_accent)


def _render_live_position_triage(positions: pd.DataFrame):
    if positions.empty:
        st.success("No open paper positions to triage.")
        return

    work = positions.copy()
    _pct_col = work["unrealised_pct"] if "unrealised_pct" in work.columns else pd.Series(0.0, index=work.index)
    work["_pnl_pct"] = pd.to_numeric(_pct_col, errors="coerce").fillna(0)
    work = work.sort_values("_pnl_pct", ascending=True)
    st.markdown("#### Open paper position triage")
    st.markdown("Read the weakest positions first. This is paper feedback, not a live order ticket.")

    for start in range(0, min(len(work), 8), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, work.iloc[start:start + 4].iterrows()):
            pnl_pct = _to_float(row.get("unrealised_pct"), 0) or 0
            pnl = _to_float(row.get("unrealised_pnl"), 0) or 0
            current = _to_float(row.get("current_price"))
            stop = _to_float(row.get("stop_price"))
            target = _to_float(row.get("target_price"))
            stop_line = "No stop data"
            if current is not None and stop is not None and current:
                distance = (current - stop) / current * 100
                stop_line = "At or below stop" if distance <= 0 else f"Stop is {distance:.1f}% below current"
            target_line = "No target data"
            if current is not None and target is not None and current:
                target_distance = (target - current) / current * 100
                target_line = f"Target is {target_distance:.1f}% away" if target_distance >= 0 else "Target is already below current"
            if pnl_pct <= -5:
                action = "Review first"
            elif pnl_pct < 0:
                action = "Watch closely"
            else:
                action = "Still okay"
            accent = "#991b1b" if pnl_pct <= -5 else "#334155" if pnl_pct < 0 else "#166534"
            with col:
                _render_html(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:13px 14px; min-height:250px; margin-bottom:12px;">
                      <div style="display:flex; justify-content:space-between; gap:8px;">
                        <div style="font-size:21px; color:#111827; font-weight:900;">{_esc(row.get("ticker"), "")}</div>
                        <div style="font-size:12px; color:{accent}; font-weight:850;">{_esc(action)}</div>
                      </div>
                      <div style="font-size:12px; color:#6b7280; margin-top:3px;">{_esc(row.get("sector"), "No sector")}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                        <div><div style="font-size:11px; color:#6b7280;">Paper gain/loss</div><div style="font-size:16px; font-weight:850; color:{accent};">{_esc(_money(pnl))}</div></div>
                        <div><div style="font-size:11px; color:#6b7280;">Return</div><div style="font-size:16px; font-weight:850; color:{accent};">{pnl_pct:.1f}%</div></div>
                      </div>
                      <div style="font-size:12px; color:#374151; line-height:1.38; margin-top:9px;"><b>Risk line:</b> {_esc(stop_line)}</div>
                      <div style="font-size:12px; color:#374151; line-height:1.38; margin-top:7px;"><b>Upside line:</b> {_esc(target_line)}</div>
                      <div style="font-size:11px; color:#6b7280; line-height:1.35; margin-top:8px;">Entry {_esc(_money(row.get("entry_price")))} · Current {_esc(_money(row.get("current_price")))}</div>
                    </div>
                    """
                )


def _render_live_feedback_board(live_ic_state: dict, live_ic_summary: pd.DataFrame, decision_state: dict, forward: pd.DataFrame, false_lab: pd.DataFrame):
    st.markdown("#### What the market has taught us so far")
    ready = int(_to_float(decision_state.get("ready_forward_observations"), 0) or 0)
    pending = int(_to_float(decision_state.get("pending_forward_observations"), 0) or 0)
    no_price = int(_to_float(decision_state.get("no_price_observations"), 0) or 0)
    live_windows = int(_to_float(live_ic_state.get("live_ic_windows"), 0) or 0)
    completed = int(_to_float(live_ic_state.get("complete_forward_return_rows"), 0) or 0)

    cols = st.columns(5)
    cards = [
        ("Ready grades", str(ready), "Past decisions that can now be judged.", "#166534" if ready else "#334155"),
        ("Still waiting", str(pending), "Future return windows have not arrived yet.", "#334155"),
        ("Missing price", str(no_price), "Needs better price history before grading.", "#991b1b" if no_price else "#166534"),
        ("Follow-up windows", str(live_windows), "The model follow-up check is running.", "#334155"),
        ("Completed rows", str(completed), "Rows with enough future price data.", "#166534" if completed else "#334155"),
    ]
    for col, (title, value, note, accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, accent)

    if not false_lab.empty:
        ready_rows = false_lab[false_lab.get("observation_status", "").astype(str).str.lower().eq("ready")] if "observation_status" in false_lab.columns else false_lab
        if not ready_rows.empty:
            st.markdown("##### Latest decision feedback")
            show = ready_rows.tail(8).copy()
            for start in range(0, len(show), 4):
                cols = st.columns(4)
                for col, (_, row) in zip(cols, show.iloc[start:start + 4].iterrows()):
                    outcome = _live_plain(row.get("outcome_label"), 90)
                    accent = "#166534" if "protected" in outcome.lower() else "#991b1b" if "missed" in outcome.lower() else "#334155"
                    with col:
                        _render_html(
                            f"""
                            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:12px 13px; min-height:180px; margin-bottom:10px;">
                              <div style="font-size:19px; font-weight:900; color:#111827;">{_esc(row.get("ticker"), "")}</div>
                              <div style="font-size:12px; color:{accent}; font-weight:850; margin-top:5px;">{_esc(outcome)}</div>
                              <div style="font-size:13px; color:#374151; line-height:1.4; margin-top:8px;">{_esc(_live_plain(row.get("plain_read"), 150))}</div>
                              <div style="font-size:11px; color:#6b7280; margin-top:8px;">{_esc(row.get("horizon_days"), "")}d return: {_esc(_pct_display(row.get("forward_return_pct")))}</div>
                            </div>
                            """
                        )

    if not live_ic_summary.empty:
        pending_only = live_ic_summary[live_ic_summary.get("status", "").astype(str).str.upper().str.contains("PENDING", na=False)] if "status" in live_ic_summary.columns else pd.DataFrame()
        if len(pending_only) == len(live_ic_summary):
            st.info("The live follow-up check is running, but it is still waiting for future prices. Do not call the model proven yet.")


def _render_event_cards(events: pd.DataFrame):
    if events.empty:
        st.success("No live-style monitor events are active.")
        return

    work = events.copy()
    rank = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "DATA_GAP": 3}
    if "severity" in work.columns:
        work["_rank"] = work["severity"].astype(str).str.upper().map(rank).fillna(9)
        work = work.sort_values(["_rank"] + [c for c in ["monitor", "ticker"] if c in work.columns])

    st.markdown("##### Events to read first")
    for _, row in work.head(8).iterrows():
        sev = str(row.get("severity", "INFO")).upper()
        accent = "#991b1b" if sev == "CRITICAL" else "#334155" if sev == "WARNING" else "#0f766e"
        ticker = _clean_display(row.get("ticker"), "Portfolio")
        if ticker.lower() in {"nan", "none", ""}:
            ticker = "Portfolio"
        st.markdown(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:13px 15px; margin:0 0 10px 0;">
              <div style="display:flex; justify-content:space-between; gap:12px;">
                <div style="font-size:15px; font-weight:850; color:#111827;">{_esc(ticker)} · {_esc(_live_plain(row.get("title"), 180))}</div>
                <div style="font-size:12px; font-weight:850; color:{accent};">{_esc(_live_plain(sev, 40))}</div>
              </div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:6px;">{_esc(_live_plain(row.get("detail"), 280))}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:8px; padding-top:7px; font-size:12px; color:#6b7280;">Next: {_esc(_live_plain(row.get("action"), 140), "Review manually")} · Source: {_esc(row.get("source_provider"), "Local monitor")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Open all monitor events", expanded=False):
        cols = [c for c in [
            "severity", "monitor", "ticker", "title", "detail", "action",
            "metric_1_name", "metric_1_value", "metric_2_name", "metric_2_value",
            "source_layer", "source_provider", "source_file",
        ] if c in work.columns]
        _show_status_table(work[cols] if cols else work, ["severity"], height=560)


def tab_live_paper_monitor():
    _render_section_depth("Live / Paper")

    paper_summary = safe_csv(ROOT / "paper_sim_summary.csv")
    positions = safe_csv(ROOT / "paper_sim_positions.csv")
    paper_nav = safe_csv(ROOT / "paper_nav_curve.csv")
    portfolio_nav = safe_csv(ROOT / "portfolio_nav.csv")
    live_nav = safe_csv(ROOT / "live_nav_curve.csv")
    live_template = safe_csv(ROOT / "live_nav_manual_template.csv")
    desk_summary = safe_json(ROOT / "desk_monitor_summary.json")
    events = safe_csv(ROOT / "desk_monitor_events.csv")
    ticker_state = safe_csv(ROOT / "desk_monitor_ticker_state.csv")
    live_ic_state = safe_json(ROOT / "live_ic_observation_state.json")
    live_ic_summary = safe_csv(ROOT / "live_ic_realized_summary.csv")
    execution_state = safe_json(ROOT / "execution_tca_state.json")
    execution_cards = safe_csv(ROOT / "execution_tca_ticker_cards.csv")
    decision_state = safe_json(ROOT / "decision_memory_state.json")
    decision_forward = safe_csv(ROOT / "decision_forward_return_check.csv")
    false_lab = safe_csv(ROOT / "decision_false_positive_negative_lab.csv")

    live_connected = not live_nav.empty
    mode = "Manual live tracking" if live_connected else "Paper mode"
    nav_source = "manual account value" if live_connected else "paper account value"
    latest_nav, nav_df = _latest_nav_row(live_nav, portfolio_nav, paper_nav)
    paper_row = paper_summary.iloc[0] if not paper_summary.empty else pd.Series(dtype=object)
    critical_events = int(_to_float(desk_summary.get("critical_count"), 0) or 0)
    total_events = int(_to_float(desk_summary.get("total_events"), len(events) if not events.empty else 0) or 0)
    open_positions = int(_to_float(paper_row.get("n_open"), len(positions) if not positions.empty else 0) or 0)
    account_size = _money(paper_row.get("account_size"))
    total_pnl = _money(paper_row.get("total_pnl"))
    total_return = _pct_display(paper_row.get("total_return_pct"))
    latest_nav_value = _first_non_empty(latest_nav.get("nav"), latest_nav.get("account_equity"), fallback="No data")
    nav_value = f"{_to_float(latest_nav_value):,.2f}" if _to_float(latest_nav_value) is not None else str(latest_nav_value)
    drawdown = _pct_display(latest_nav.get("drawdown_pct"))
    daily_return = _pct_display(latest_nav.get("daily_return"))
    live_ic_status = _plain_status(_first_non_empty(live_ic_state.get("overall_status"), fallback="No data"))
    execution_status = _plain_status(_first_non_empty(execution_state.get("status"), fallback="No data"))

    if critical_events > 0:
        answer = "Critical monitor events are active. Treat the book as protect-first; do not add exposure before reading the event list."
        accent = "#991b1b"
    elif not live_connected:
        answer = "This is paper-mode tracking. Real account value is not connected; use paper gain/loss and manual checks only."
        accent = "#334155"
    else:
        answer = "Manual account-value tracking is active. Still research-only: no broker link and no live orders."
        accent = "#166534"

    st.markdown(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:9px; padding:17px 19px; margin:8px 0 16px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">Simple account answer</div>
          <div style="font-size:24px; color:#111827; font-weight:850; line-height:1.25; margin-top:5px;">{_esc(answer)}</div>
          <div style="font-size:12px; color:#6b7280; margin-top:10px;">Mode: {_esc(mode)} · Source: {_esc(nav_source)} · No broker connection · No live order path</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _simple_card("Account mode", mode, "Manual live only appears if live_nav_curve.csv has real rows.", accent)
    with k2:
        _simple_card("Open paper positions", str(open_positions), f"Paper account size: {account_size}", "#111827")
    with k3:
        _simple_card("Paper gain/loss", total_pnl, f"Return: {total_return}", "#166534" if (_to_float(paper_row.get("total_pnl"), 0) or 0) >= 0 else "#991b1b")
    with k4:
        _simple_card("Critical events", str(critical_events), f"Total monitor events: {total_events}", "#991b1b" if critical_events else "#166534")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        _simple_card("Account value", nav_value, f"Daily return: {daily_return}", "#111827")
    with k6:
        _simple_card("Drop from high", drawdown, "Distance from the highest account value.", "#991b1b" if (_to_float(latest_nav.get("drawdown_pct"), 0) or 0) < 0 else "#166534")
    with k7:
        _simple_card("Model follow-up", live_ic_status, "Only real after future price windows finish.", "#334155")
    with k8:
        _simple_card("Trading-cost status", execution_status, "Spread, volume, fill, and cost checks.", "#334155")

    _render_live_paper_workflow_board(
        live_connected,
        critical_events,
        total_events,
        open_positions,
        paper_row,
        latest_nav,
        live_ic_state,
        execution_state,
        decision_state,
    )
    _render_live_position_triage(positions)
    _render_event_cards(events)
    _render_live_feedback_board(live_ic_state, live_ic_summary, decision_state, decision_forward, false_lab)

    show_detail = st.checkbox("Show technical paper-account tables", value=False, key="live_paper_show_detail")
    if not show_detail:
        st.caption("Detailed tables are hidden by default so this page reads like a monitor, not a log file.")
        return

    st.markdown("---")
    view_account, view_positions, view_monitor, view_signal, view_raw = st.tabs(
        ["Account value", "Open paper positions", "Alerts", "Model follow-up", "Source files"]
    )

    with view_account:
        st.markdown("#### Account value and account state")
        _render_nav_chart(nav_df, nav_source)
        nav_cols = [c for c in ["date", "nav", "account_equity", "daily_return", "hwm", "drawdown_pct", "cumulative_return_pct", "cash", "gross_exposure", "net_exposure", "source"] if c in nav_df.columns]
        if not nav_df.empty:
            _show_status_table(nav_df[nav_cols].tail(12) if nav_cols else nav_df.tail(12), [], height=260)
        if not live_connected:
            st.info(
                "Manual account-value tracking is not active yet. To track a real account manually, fill live_nav_manual.csv from the template, then run the daily system again. This still will not connect to a broker."
            )
        if not live_template.empty:
            with st.expander("Manual account-value template", expanded=False):
                _show_status_table(live_template, [], height=180)

    with view_positions:
        _render_position_cards(positions)
        if not execution_cards.empty:
            st.markdown("##### Trading-cost feasibility cards")
            show_cols = [c for c in ["ticker", "card_status", "score", "headline", "cost_line", "route_line", "blocker_line", "manual_check", "trigger"] if c in execution_cards.columns]
            _show_status_table(execution_cards[show_cols].head(30) if show_cols else execution_cards.head(30), ["card_status"], height=520)

    with view_monitor:
        _render_event_cards(events)
        if not ticker_state.empty:
            st.markdown("##### Ticker monitor state")
            cols = [c for c in [
                "ticker", "max_monitor_severity", "event_count", "price_break_state",
                "volume_spike_state", "volatility_regime_state", "spread_status",
                "latest_close", "daily_return", "volume_ratio",
            ] if c in ticker_state.columns]
            _show_status_table(
                ticker_state[cols] if cols else ticker_state,
                [c for c in ["max_monitor_severity", "price_break_state", "volume_spike_state", "volatility_regime_state", "spread_status"] if c in cols],
                height=420,
            )

    with view_signal:
        if live_ic_state:
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Live IC status", live_ic_status)
            with s2:
                st.metric("Observation rows", live_ic_state.get("observation_rows", "No data"))
            with s3:
                st.metric("Completed rows", live_ic_state.get("complete_forward_return_rows", "No data"))
            with s4:
                st.metric("Pending rows", live_ic_state.get("pending_forward_return_rows", "No data"))
            st.caption("Pending means the model has made observations, but future returns are not mature yet. Do not call it proven.")
        else:
            st.info("No live signal validation state yet.")
        if not live_ic_summary.empty:
            cols = [c for c in [
                "signal", "horizon_days", "live_observation_windows",
                "mean_live_ic", "positive_ic_pct", "status", "required_next_action",
            ] if c in live_ic_summary.columns]
            _show_status_table(live_ic_summary[cols] if cols else live_ic_summary, ["status"], height=520)

    with view_raw:
        st.markdown("#### Source files used by this page")
        rows = []
        for fname, label in [
            ("paper_sim_summary.csv", "Paper account summary"),
            ("paper_sim_positions.csv", "Open paper positions"),
            ("paper_nav_curve.csv", "Paper NAV curve"),
            ("portfolio_nav.csv", "Portfolio NAV fallback"),
            ("live_nav_curve.csv", "Manual live NAV curve"),
            ("live_nav_manual_template.csv", "Manual live NAV template"),
            ("desk_monitor_events.csv", "Monitor events"),
            ("desk_monitor_ticker_state.csv", "Ticker monitor state"),
            ("live_ic_observation_state.json", "Live IC state"),
            ("live_ic_realized_summary.csv", "Live IC summary"),
            ("execution_tca_state.json", "Execution status"),
            ("execution_tca_ticker_cards.csv", "Execution cards"),
        ]:
            path = ROOT / fname
            rows.append({
                "File": fname,
                "Meaning": label,
                "Exists": "Yes" if path.exists() and path.stat().st_size > 10 else "No",
                "Updated": file_age_str(path),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if callable(_ORIGINAL_TAB_LIVE_PAPER_MONITOR):
            with st.expander("Original detailed Live / Paper page", expanded=False):
                _run_with_markdown_replacements(
                    _ORIGINAL_TAB_LIVE_PAPER_MONITOR,
                    {'<p class="section-title">Portfolio</p>': '<p class="section-title">Original Live / Paper Details</p>'},
                )


def _render_system_health_bar():
    """
    Compact system health strip — last run date, pass/fail, and key file freshness.
    Always shown at the top of the System page regardless of sub-tab.
    """
    log = safe_csv(ROOT / "run_daily_all_log.csv")

    # Parse last run
    last_date = last_time = "—"
    n_ok = n_fail = n_total = 0
    if not log.empty and "date" in log.columns and "status" in log.columns:
        last_date_val = log["date"].max()
        last_date = str(last_date_val)[:10]
        last_time = str(last_date_val)[11:16] if len(str(last_date_val)) > 10 else ""
        last_rows = log[log["date"] == last_date_val]
        n_ok    = int((last_rows["status"] == "OK").sum())
        n_fail  = int((last_rows["status"].isin(["FAILED","TIMEOUT"])).sum())
        n_total = int(len(last_rows))

    health_color = "#16a34a" if n_fail == 0 else "#dc2626" if n_fail > 3 else "#d97706"
    health_label = "All passing" if n_fail == 0 else f"{n_fail} failed"

    # Key file freshness
    KEY_FILES = [
        ("alpha_scores.csv",       "Alpha scores"),
        ("momentum_scores.csv",    "Momentum"),
        ("risk_desk_overview.json","Risk overview"),
        ("macro_signals.json",     "Macro signals"),
        ("news_impact_targets.csv","News impact"),
    ]
    file_items = []
    for fname, label in KEY_FILES:
        fpath = ROOT / fname
        if fpath.exists():
            age_h = (time.time() - fpath.stat().st_mtime) / 3600
            age_str = f"{age_h:.0f}h" if age_h < 24 else f"{age_h/24:.0f}d"
            ok = age_h < 26   # fresh if < 26h
            file_items.append((label, age_str, ok))
        else:
            file_items.append((label, "missing", False))

    files_html = "".join(
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px;">'
        f'<div style="width:8px;height:8px;border-radius:50%;background:{"#16a34a" if ok else "#dc2626"};"></div>'
        f'<div style="font-size:9px;color:{"#166534" if ok else "#991b1b"};font-weight:700;">{_esc(age)}</div>'
        f'<div style="font-size:8px;color:#94a3b8;">{_esc(lbl)}</div>'
        f'</div>'
        for lbl, age, ok in file_items
    )

    _render_html(
        f"""
        <div style="
            background:#fff;
            border:1px solid #e2e8f0;
            border-radius:10px;
            padding:11px 16px;
            margin:0 0 14px 0;
            display:flex;
            flex-wrap:wrap;
            gap:16px;
            align-items:center;
            box-shadow:0 1px 3px rgba(0,0,0,.05);
        ">
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Last run</div>
            <div style="font-size:14px;font-weight:800;color:#0f172a;margin-top:2px;">{last_date} <span style="font-size:11px;color:#64748b;">{last_time}</span></div>
          </div>
          <div style="width:1px;height:28px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Steps</div>
            <div style="font-size:14px;font-weight:800;color:{health_color};margin-top:2px;">{n_ok}/{n_total} <span style="font-size:11px;font-weight:600;">{health_label}</span></div>
          </div>
          <div style="width:1px;height:28px;background:#e2e8f0;"></div>
          <div style="display:flex;gap:14px;align-items:flex-end;">{files_html}</div>
          <div style="margin-left:auto;font-size:10px;color:#94a3b8;">Canyon v9 · Research only</div>
        </div>
        """
    )


def tab_system_status():
    if not callable(_ORIGINAL_TAB_SYSTEM_STATUS):
        st.error("All Outputs page is not available in the cached dashboard snapshot.")
        return
    return _run_with_markdown_replacements(
        _ORIGINAL_TAB_SYSTEM_STATUS,
        {
            '<p class="section-title">System</p>': '<p class="section-title">All Outputs</p>',
            '<p class="section-sub">Run health, file freshness, data health, and system logs.</p>': (
                '<p class="section-sub">All generated files, reports, run logs, data freshness, and missing-output checks.</p>'
            ),
        },
    )


def _fmt_target_number(value, digits: int = 2, suffix: str = "") -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    return f"{num:.{digits}f}{suffix}"


def _render_sharpe4_p0_repair_pack():
    state = safe_json(ROOT / "sharpe4_p0_repair_state.json")
    blockers = safe_csv(ROOT / "sharpe4_p0_blocker_summary.csv")
    tickers = safe_csv(ROOT / "sharpe4_p0_ticker_repair_plan.csv")
    signals = safe_csv(ROOT / "sharpe4_p0_signal_policy_enforced.csv")
    execution = safe_csv(ROOT / "sharpe4_p0_execution_budget.csv")

    if not state and blockers.empty:
        return

    st.markdown("##### P0 Repair Pack")
    st.markdown(
        "This is the repair checklist before any Sharpe 4 chase. If alpha allowed is zero, the system is still in cleanup mode.",
    )

    status = _plain_status(state.get("p0_repair_status"), "No status")
    alpha_allowed = _to_float(state.get("sharpe4_alpha_gross_allowed_pct"), 0) or 0
    accent = "#166534" if status.startswith("P0 Clear") or alpha_allowed > 0 else "#991b1b"

    cols = st.columns(6)
    cards = [
        ("P0 status", status, "Repair gate before alpha review."),
        ("Current gross", _fmt_target_number(state.get("current_gross_pct"), suffix="%"), "Current research book."),
        ("Clean gross", _fmt_target_number(state.get("p0_clean_gross_pct"), suffix="%"), "After risk repair weights."),
        ("Alpha gross now", _fmt_target_number(state.get("sharpe4_alpha_gross_allowed_pct"), suffix="%"), "Can count toward Sharpe 4."),
        ("Blocked signals", str(state.get("blocked_signal_count", "No data")), "Active model must exclude these."),
        ("Turnover", _fmt_target_number(state.get("median_monthly_turnover_pct"), suffix="%"), "Target is 45% monthly."),
    ]
    for col, (title, value, note) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, accent if title in {"P0 status", "Alpha gross now"} else "#334155")

    st.markdown(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 16px; margin:14px 0;">
          <div style="font-size:15px; font-weight:850; color:#111827;">What this means</div>
          <div style="font-size:13px; color:#374151; line-height:1.55; margin-top:6px;">
            Right now the repair engine says the model should cut the research book from {_esc(_fmt_target_number(state.get("current_gross_pct"), suffix="%"))}
            to about {_esc(_fmt_target_number(state.get("p0_clean_gross_pct"), suffix="%"))} before new alpha review.
            No current name is allowed to count as Sharpe 4 alpha yet because risk, signal, and execution checks are still blocking.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.92, 1.08])
    with left:
        st.markdown("###### Main blockers")
        if blockers.empty:
            st.info("No P0 blocker summary yet.")
        else:
            cols = [c for c in ["blocker", "count", "status", "plain_english"] if c in blockers.columns]
            _show_status_table(blockers[cols], ["status"], height=300)
    with right:
        st.markdown("###### First ticker repairs")
        if tickers.empty:
            st.info("No P0 ticker plan yet.")
        else:
            cols = [c for c in [
                "ticker", "current_weight_pct", "p0_clean_weight_pct",
                "required_weight_cut_pct", "p0_permission", "new_exposure_allowed",
                "risk_action", "execution_verdict",
            ] if c in tickers.columns]
            _show_status_table(tickers[cols].head(8), ["risk_action", "execution_verdict"], height=300)

    with st.expander("Open signal and execution repair files", expanded=False):
        st.markdown("###### Signal permissions")
        if not signals.empty:
            cols = [c for c in ["signal", "original_signal_action", "sharpe4_active_multiplier", "active_use", "repair_action"] if c in signals.columns]
            _show_status_table(signals[cols], ["original_signal_action", "active_use"], height=360)
        st.markdown("###### Execution budget")
        if not execution.empty:
            cols = [c for c in ["budget_area", "current_value", "target_value", "status", "repair_instruction"] if c in execution.columns]
            _show_status_table(execution[cols], ["status"], height=360)


def _render_sharpe4_recovery_roadmap():
    state = safe_json(ROOT / "sharpe4_recovery_state.json")
    stages = safe_csv(ROOT / "sharpe4_recovery_stage_plan.csv")
    pool = safe_csv(ROOT / "sharpe4_recovery_candidate_pool.csv")
    actions = safe_csv(ROOT / "sharpe4_recovery_top_actions.csv")

    if not state and stages.empty and pool.empty:
        return

    st.markdown("##### Recovery Map")
    st.markdown(
        "This answers what to do because the gap is large: repair the current book, rebuild candidates outside it, then retest honestly.",
    )

    cols = st.columns(5)
    cards = [
        ("Mode", _plain_status(state.get("recovery_status"), "No data"), "The system is not in alpha-chase mode."),
        ("Candidate pool", str(state.get("candidate_pool_rows", "No data")), "Broader local universe rows."),
        ("Risk-entry first", str(state.get("risk_entry_first_count", "No data")), "Interesting but not tradeable yet."),
        ("Bad-book repairs", str(state.get("current_book_repair_only_count", "No data")), "Current names to repair first."),
        ("Watchlist now", str(state.get("recovery_watchlist_count", "No data")), "Clean enough for watch only."),
    ]
    for col, (title, value, note) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, "#334155")

    st.markdown(
        """
        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #334155; border-radius:8px; padding:14px 16px; margin:14px 0;">
          <div style="font-size:15px; font-weight:850; color:#111827;">Plain answer</div>
          <div style="font-size:13px; color:#374151; line-height:1.55; margin-top:6px;">
            The model is far from Sharpe 4 because the current book is blocked and the outside ideas are not risk-book-ready yet.
            The next useful work is not to buy more; it is to convert the best outside research names into audited risk-book candidates.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not actions.empty:
        st.markdown("###### Next actions")
        cols = [c for c in ["priority", "action", "plain_english", "where_to_click"] if c in actions.columns]
        _show_status_table(actions[cols], ["priority"], height=230)

    left, right = st.columns([0.92, 1.08])
    with left:
        st.markdown("###### Recovery stages")
        if stages.empty:
            st.info("No recovery stage plan yet.")
        else:
            cols = [c for c in ["stage", "status", "goal", "what_to_do"] if c in stages.columns]
            _show_status_table(stages[cols], ["status"], height=360)
    with right:
        st.markdown("###### Best outside research candidates")
        if pool.empty:
            st.info("No recovery candidate pool yet.")
        else:
            _rl = pool["recovery_lane"].astype(str) if "recovery_lane" in pool.columns else pd.Series("", index=pool.index)
            display = pool[~_rl.str.contains("Repair current book", na=False)].copy()
            cols = [c for c in [
                "recovery_rank", "ticker", "recovery_rank_score", "recovery_lane",
                "current_permission", "risk_status", "signal_status", "cycle_read",
            ] if c in display.columns]
            _show_status_table(display[cols].head(12), ["current_permission", "risk_status", "signal_status"], height=360)

    with st.expander("Open full recovery candidate pool", expanded=False):
        if not pool.empty:
            cols = [c for c in [
                "recovery_rank", "ticker", "recovery_rank_score", "recovery_lane",
                "next_action", "event_status", "event_score", "theme_attention_score",
                "liquidity_status", "risk_status", "signal_status", "source_files",
            ] if c in pool.columns]
            _show_status_table(pool[cols], ["recovery_lane", "risk_status", "signal_status"], height=520)


def _render_sharpe4_risk_book_intake():
    state = safe_json(ROOT / "sharpe4_risk_book_intake_state.json")
    cards = safe_csv(ROOT / "sharpe4_risk_book_candidate_cards.csv")
    var_liq = safe_csv(ROOT / "sharpe4_risk_book_var_liquidity.csv")
    event_route = safe_csv(ROOT / "sharpe4_risk_book_event_route.csv")
    corr = safe_csv(ROOT / "sharpe4_risk_book_correlation_proxy.csv")
    promo_state = safe_json(ROOT / "sharpe4_risk_book_promotion_state.json")
    promo_gate = safe_csv(ROOT / "sharpe4_risk_book_promotion_gate.csv")
    proof_queue = safe_csv(ROOT / "sharpe4_risk_book_manual_proof_queue.csv")

    if not state and cards.empty:
        return

    st.markdown("##### Names waiting for evidence")
    st.markdown(
        "These are research names only. The system is saying what must be checked before any name can move forward."
    )

    cols = st.columns(5)
    status = _plain_status(state.get("status"), "No data")
    summary_cards = [
        ("Status", status, "The intake gate is still open."),
        ("Names checked", str(state.get("candidate_count", "No data")), "Research names reviewed."),
        ("High-risk names", str(state.get("high_or_very_high_risk_count", "No data")), "Tail risk is too large for normal size."),
        ("Missing earnings check", str(state.get("earnings_calendar_missing_count", "No data")), "Calendar or gap risk must be filled."),
        ("Can paper trade now", str(state.get("paper_sizing_allowed_now_count", "0")), "Must stay zero until all checks clear."),
    ]
    for col, (title, value, note) in zip(cols, summary_cards):
        with col:
            accent = "#991b1b" if title == "Can paper trade now" and str(value) != "0" else "#334155"
            _simple_card(title, value, note, accent)

    st.markdown(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #334155; border-radius:8px; padding:14px 16px; margin:14px 0;">
          <div style="font-size:15px; font-weight:850; color:#111827;">Plain answer</div>
          <div style="font-size:13px; color:#374151; line-height:1.58; margin-top:6px;">
            {_esc(state.get("plain_english"), "These names are research only.")}
            The next useful step is to check earnings dates, live spread/liquidity, event source, and crowding before any name can move from "read only" to "study next."
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not cards.empty:
        st.markdown("###### Quick status table")
        top = cards.head(8).copy()
        display_cols = [c for c in [
            "ticker", "current_answer", "risk_level", "liquidity", "correlation",
            "earnings", "options_now", "main_blockers",
        ] if c in top.columns]
        _show_status_table(top[display_cols], ["risk_level", "liquidity", "correlation", "earnings"], height=360)

        st.markdown("###### Short / medium / long preview")
        route_cols = [c for c in [
            "ticker", "short_term", "medium_term", "long_term", "proof_needed",
        ] if c in top.columns]
        _show_status_table(top[route_cols], [], height=360)

    if promo_state or not promo_gate.empty:
        st.markdown("###### Can these names move forward?")
        st.markdown(
            "This says which names are still read-only, which can be studied next, and what remains forbidden."
        )
        promo_cols = st.columns(4)
        promo_cards = [
            ("Still read-only", str(promo_state.get("blocked_from_paper_review_count", "No data")), "Do not touch yet."),
            ("Can study after evidence", str(promo_state.get("can_become_watch_only_after_proof_count", "No data")), "Still not a trade."),
            ("Paper trades now", str(promo_state.get("paper_sizing_allowed_now_count", "0")), "Must remain zero."),
            ("Calls / puts now", str(promo_state.get("options_allowed_now_count", "0")), "Must remain zero."),
        ]
        for col, (title, value, note) in zip(promo_cols, promo_cards):
            with col:
                _simple_card(title, value, note, "#334155")

        if not promo_gate.empty:
            cols = [c for c in [
                "ticker", "promotion_status", "first_proof_to_collect",
                "current_permission", "option_gate", "why_this_is_first",
            ] if c in promo_gate.columns]
            _show_status_table(promo_gate[cols].head(18), ["promotion_status"], height=420)

        with st.expander("Open evidence queue", expanded=False):
            if proof_queue.empty:
                st.info("No manual proof queue yet.")
            else:
                cols = [c for c in [
                    "priority", "ticker", "task", "how_to_do_it",
                    "done_when", "still_forbidden",
                ] if c in proof_queue.columns]
                _show_status_table(proof_queue[cols], ["priority"], height=520)

    with st.expander("Open why each candidate is blocked", expanded=False):
        if cards.empty:
            st.info("No risk-book candidate cards yet.")
        else:
            for _, row in cards.head(12).iterrows():
                st.markdown(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid #64748b; border-radius:8px; padding:12px 14px; margin:9px 0;">
                      <div style="font-size:16px; font-weight:850; color:#111827;">{_esc(row.get("ticker"), "Ticker")}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.55; margin-top:5px;">{_esc(row.get("plain_thesis"), "")}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.55; margin-top:8px;"><b>Now:</b> {_esc(row.get("current_answer"), "")}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.55;"><b>Options:</b> {_esc(row.get("options_now"), "")}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.55;"><b>Blockers:</b> {_esc(row.get("main_blockers"), "")}</div>
                      <div style="font-size:12px; color:#6b7280; line-height:1.5; margin-top:8px;"><b>Proof needed:</b> {_esc(row.get("proof_needed"), "")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("Open source detail: VaR, events, correlation", expanded=False):
        st.markdown("###### VaR and liquidity")
        if not var_liq.empty:
            cols = [c for c in [
                "ticker", "annual_vol_pct", "daily_cvar_95_pct", "five_day_cvar_95_pct",
                "price_risk", "avg_dollar_volume_20d", "liquidity_status", "estimated_tca_bps",
            ] if c in var_liq.columns]
            _show_status_table(var_liq[cols].head(24), ["price_risk", "liquidity_status"], height=360)

        st.markdown("###### News / event / option route")
        if not event_route.empty:
            cols = [c for c in [
                "ticker", "event_score", "event_role", "event_headline",
                "earnings_status", "iv_rank", "event_route", "option_answer",
            ] if c in event_route.columns]
            _show_status_table(event_route[cols].head(24), ["earnings_status", "event_route"], height=360)

        st.markdown("###### Correlation and crowding")
        if not corr.empty:
            cols = [c for c in [
                "ticker", "corr_to_spy", "corr_to_qqq", "corr_to_smh",
                "highest_peer", "highest_peer_corr", "correlation_risk", "correlation_note",
            ] if c in corr.columns]
            _show_status_table(corr[cols].head(24), ["correlation_risk"], height=360)


def _render_sharpe4_proof_workbench():
    state = safe_json(ROOT / "sharpe4_proof_workbench_state.json")
    groups = safe_csv(ROOT / "sharpe4_proof_workbench_task_groups.csv")
    tasks = safe_csv(ROOT / "sharpe4_proof_workbench_ticker_tasks.csv")
    template = safe_csv(ROOT / "sharpe4_manual_proof_input_template.csv")

    if not state and groups.empty and tasks.empty and template.empty:
        st.info("Proof Workbench has not run yet. Run Step191 to create the fillable workflow.")
        return

    bucket_titles = {
        "Earnings and gap proof": "Check earnings date and jump risk",
        "Tail-risk stop proof": "Write the loss limit",
        "Crowding and overlap proof": "Check if this is the same bet twice",
        "Spread and fill proof": "Check if trading would be too expensive",
        "Event reaction proof": "Check if the news actually moved the stock",
        "Source proof": "Attach the source and explain the link",
    }

    st.markdown("##### What must be checked first")
    st.markdown(
        "These are not trade ideas. They are the missing checks before any name is allowed to move forward."
    )

    cols = st.columns(4)
    summary_cards = [
        ("Check groups", str(state.get("work_bucket_count", len(groups) if not groups.empty else 0)), "The work is grouped so it is less messy."),
        ("Names to check", str(state.get("total_tasks", len(tasks) if not tasks.empty else 0)), "These are names to verify, not buy."),
        ("First job", bucket_titles.get(_plain_status(state.get("first_bucket"), ""), _plain_status(state.get("first_bucket"), "No data")), "Start here."),
        (
            "Paper / calls-puts now",
            f"{state.get('paper_sizing_allowed_now_count', 0)} / {state.get('options_allowed_now_count', 0)}",
            "Both should stay 0 until the checks are filled.",
        ),
    ]
    for col, (title, value, note) in zip(cols, summary_cards):
        with col:
            _simple_card(title, value, note, "#334155")

    if not groups.empty:
        work = groups.copy()
        if "group_rank" in work.columns:
            work["_rank"] = pd.to_numeric(work["group_rank"], errors="coerce")
            work = work.sort_values("_rank")

        for start in range(0, len(work), 2):
            row_cols = st.columns(2)
            for col, (_, row) in zip(row_cols, work.iloc[start:start + 2].iterrows()):
                tickers = _clean_display(row.get("tickers"), "No tickers")
                purpose = _clean_display(row.get("purpose"), "No purpose recorded.")
                collect = _clean_display(row.get("what_to_collect"), "No proof instruction recorded.")
                done_when = _clean_display(row.get("done_when"), "No completion rule recorded.")
                forbidden = _clean_display(row.get("still_forbidden"), "No paper size. No calls or puts.")
                bucket = _clean_display(row.get("work_bucket"), "Proof bucket")
                plain_bucket = bucket_titles.get(bucket, bucket)
                with col:
                    st.markdown(
                        f"""
                        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #334155; border-radius:8px; padding:15px 16px; min-height:330px; margin:10px 0;">
                          <div style="font-size:12px; font-weight:850; color:#6b7280; text-transform:uppercase;">Check { _esc(row.get("group_rank"), "") } - { _esc(row.get("ticker_count"), "0") } names</div>
                          <div style="font-size:20px; font-weight:900; color:#111827; line-height:1.25; margin-top:6px;">{_esc(plain_bucket)}</div>
                          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;"><b>Names:</b> {_esc(tickers)}</div>
                          <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:13px; color:#374151; line-height:1.45;"><b>Why it matters:</b> {_esc(purpose)}</div>
                          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;"><b>Find this:</b> {_esc(collect)}</div>
                          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;"><b>Done when:</b> {_esc(done_when)}</div>
                          <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:8px 9px; margin-top:10px; font-size:12px; color:#6b7280; line-height:1.35;"><b>Do not do yet:</b> {_esc(forbidden)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with st.expander("Open the evidence sheet to fill", expanded=False):
        st.markdown(
            "Fill this file before asking the system to move a name forward: "
            "`sharpe4_manual_proof_input_template.csv`."
        )
        if template.empty:
            st.info("The proof input template is not available yet.")
        else:
            cols = [c for c in [
                "ticker", "work_bucket", "proof_to_collect", "source_name",
                "source_url_or_file", "source_date_or_timestamp", "key_numbers",
                "evidence_summary", "pass_fail_review", "next_gate_request",
            ] if c in template.columns]
            _show_status_table(template[cols].head(24), ["pass_fail_review"], height=420)

    with st.expander("Open every name and what to check", expanded=False):
        if tasks.empty:
            st.info("No ticker proof tasks yet.")
        else:
            cols = [c for c in [
                "ticker", "work_bucket", "proof_to_collect", "exact_next_step",
                "risk_snapshot", "option_gate", "source_headline", "still_forbidden",
            ] if c in tasks.columns]
            _show_status_table(tasks[cols].head(60), ["work_bucket", "option_gate"], height=560)


def _render_sharpe4_manual_proof_review_gate():
    state = safe_json(ROOT / "sharpe4_manual_proof_review_state.json")
    gate = safe_csv(ROOT / "sharpe4_manual_proof_review_gate.csv")
    watch = safe_csv(ROOT / "sharpe4_watch_only_review_queue.csv")
    missing = safe_csv(ROOT / "sharpe4_manual_proof_missing_fields.csv")

    if not state and gate.empty and watch.empty:
        st.info("Manual Proof Review has not run yet. Run Step192 after filling the proof template.")
        return

    st.markdown("##### Can any name move forward?")
    st.markdown(
        "Short answer first: a name can move forward only after the evidence sheet is filled. Moving forward means study only, not paper trading and not calls or puts."
    )

    cols = st.columns(5)
    review_cards = [
        ("Evidence filled", str(state.get("reviewed_rows", 0)), "Names where you wrote real evidence."),
        ("Can study next", str(state.get("ready_for_watch_only_review_count", 0)), "Study only. Not a buy list."),
        ("Still unsafe", str(state.get("accepted_but_still_blocked_count", 0)), "One issue fixed, another remains."),
        ("Still blank", str(state.get("not_reviewed_count", 0)), "Nothing has been written yet."),
        (
            "Paper / calls-puts",
            f"{state.get('paper_sizing_allowed_now_count', 0)} / {state.get('options_allowed_now_count', 0)}",
            "Both stay 0 on this page.",
        ),
    ]
    for col, (title, value, note) in zip(cols, review_cards):
        with col:
            _simple_card(title, value, note, "#334155")

    if watch.empty:
        st.markdown(
            """
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #111827; border-radius:8px; padding:14px 15px; margin:12px 0;">
              <div style="font-size:16px; font-weight:900; color:#111827;">Current review answer</div>
              <div style="font-size:14px; color:#374151; line-height:1.5; margin-top:6px;">
                No name can move forward yet. The evidence sheet is still blank. Start by filling one ticker with a source, date, key numbers, short evidence summary, and a clear pass/fail decision.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("##### Names you may study next")
        for start in range(0, min(len(watch), 8), 4):
            cols = st.columns(4)
            for col, (_, row) in zip(cols, watch.iloc[start:start + 4].iterrows()):
                with col:
                    st.markdown(
                        f"""
                        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #166534; border-radius:8px; padding:13px 14px; min-height:250px; margin-bottom:12px;">
                          <div style="font-size:22px; font-weight:900; color:#111827;">{_esc(row.get("ticker"), "")}</div>
                          <div style="font-size:12px; color:#166534; font-weight:850; text-transform:uppercase; margin-top:2px;">Study only</div>
                          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;"><b>Why:</b> {_esc(row.get("why_in_plain_english"), "")}</div>
                          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;"><b>Next:</b> {_esc(row.get("next_step_plain"), "")}</div>
                          <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:12px; color:#6b7280; line-height:1.35;">{_esc(row.get("what_not_to_do"), "Do not paper trade yet. Do not use calls or puts.")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with st.expander("Open the plain-English decision table", expanded=False):
        if gate.empty:
            st.info("No evidence review rows yet.")
        else:
            cols = [c for c in [
                "ticker", "plain_status", "can_i_touch_it", "why_in_plain_english",
                "next_step_plain", "what_not_to_do", "source_name",
                "source_date_or_timestamp", "key_numbers",
            ] if c in gate.columns]
            _show_status_table(gate[cols].head(60), ["plain_status"], height=560)

    if not missing.empty:
        with st.expander("Open missing evidence fields", expanded=False):
            cols = [c for c in ["ticker", "work_bucket", "missing_field", "why_it_matters"] if c in missing.columns]
            _show_status_table(missing[cols].head(80), ["missing_field"], height=420)


def _render_sharpe4_simple_command_center():
    state = safe_json(ROOT / "sharpe4_simple_command_state.json")
    cards = safe_csv(ROOT / "sharpe4_simple_today_cards.csv")
    queue = safe_csv(ROOT / "sharpe4_simple_candidate_queue.csv")

    if not state and cards.empty and queue.empty:
        st.info("Simple Command Center has not run yet. Run Step190 to create the clean view.")
        return

    st.markdown('<p class="section-title">Simple Command Center</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">One clean answer first. Detailed risk math is still available, but it stays folded unless you need it.</p>',
        unsafe_allow_html=True,
    )

    answer = _esc(state.get("answer"), "No new paper size and no options today.")
    first_job = _esc(state.get("first_job"), "Fix proof and risk book first.")
    mode = _plain_status(state.get("mode"), "Proof First")
    st.markdown(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #111827; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; font-weight:850; color:#6b7280; text-transform:uppercase; letter-spacing:.02em;">Mode · {_esc(mode)}</div>
          <div style="font-size:28px; font-weight:900; color:#111827; line-height:1.2; margin-top:6px;">{answer}</div>
          <div style="font-size:15px; color:#374151; line-height:1.55; margin-top:9px;">First job: <b>{first_job}</b></div>
          <div style="font-size:13px; color:#6b7280; line-height:1.5; margin-top:8px;">{_esc(state.get("main_warning"), "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    simple_cards = [
        ("Headline Sharpe", _fmt_target_number(state.get("current_headline_sharpe")), "Current visible backtest."),
        ("Proof-adjusted", _fmt_target_number(state.get("proof_adjusted_sharpe")), "Honest planning number."),
        ("Paper sizing", str(state.get("paper_sizing_allowed_now_count", 0)), "0 means none are allowed."),
        ("Options", str(state.get("options_allowed_now_count", 0)), "0 means none are allowed."),
        ("Proof queue", str(state.get("risk_book_candidates", "No data")), "Names to prove, not buy."),
    ]
    for col, (title, value, note) in zip(cols, simple_cards):
        with col:
            _simple_card(title, value, note, "#334155")

    if not cards.empty:
        st.markdown("##### Today's only checklist")
        cols = st.columns(3)
        for i, (_, row) in enumerate(cards.head(3).iterrows()):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:14px 15px; min-height:190px; margin-bottom:10px;">
                      <div style="font-size:12px; font-weight:850; color:#6b7280; text-transform:uppercase;">{_esc(row.get("card"), "Task")}</div>
                      <div style="font-size:17px; font-weight:850; color:#111827; line-height:1.3; margin-top:7px;">{_esc(row.get("value"), "")}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:13px; color:#4b5563; line-height:1.45;">
                        {_esc(row.get("why_it_matters"), "")}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    _render_sharpe4_proof_workbench()
    _render_sharpe4_manual_proof_review_gate()

    st.markdown("##### First names to understand")
    st.markdown("This is not a buy list. Each card says the first missing proof.")
    if queue.empty:
        st.info("No simple candidate queue yet.")
    else:
        top = queue.head(8)
        for start in range(0, len(top), 4):
            cols = st.columns(4)
            for col, (_, row) in zip(cols, top.iloc[start:start + 4].iterrows()):
                status = _plain_status(row.get("simple_status"), "Needs Proof")
                border = "#991b1b" if "Not Ready" in status else "#334155"
                with col:
                    st.markdown(
                        f"""
                        <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid {border}; border-radius:8px; padding:13px 14px; min-height:310px; margin-bottom:12px;">
                          <div style="font-size:22px; font-weight:900; color:#111827;">{_esc(row.get("ticker"), "")}</div>
                          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; margin-top:2px;">{_esc(status)}</div>
                          <div style="font-size:13px; color:#111827; font-weight:850; line-height:1.35; margin-top:10px;">First proof: {_esc(row.get("first_proof"), "")}</div>
                          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;">{_esc(row.get("why"), "")}</div>
                          <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:12px; color:#6b7280; line-height:1.42;">
                            Do not: size, use options, or count as Sharpe 4 alpha yet.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with st.expander("Open full simple candidate queue", expanded=False):
            cols = [c for c in [
                "ticker", "simple_status", "first_proof", "why",
                "what_to_do_next", "do_not_do", "short_term_read", "medium_term_read",
            ] if c in queue.columns]
            _show_status_table(queue[cols], ["simple_status"], height=560)


def _depth5_accent(score, status: str = "") -> str:
    score_num = _to_float(score, 0) or 0
    status_text = str(status or "").upper()
    if score_num < 45 or any(x in status_text for x in ["REPAIR", "NOT RELIABLE", "BLOCK"]):
        return "#991b1b"
    if score_num < 70 or "PROTOTYPE" in status_text:
        return "#334155"
    return "#166534"


def _prepare_source_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for col in ["source_file", "source_files"]:
        if col in work.columns:
            work[col] = work[col].map(_friendly_source_label)
    return work


def _render_depth5_module_cards(modules: pd.DataFrame):
    if modules.empty:
        st.info("The five-module scorecard is missing. Run Step194.")
        return

    for start in range(0, min(len(modules), 5), 5):
        cols = st.columns(5)
        for col, (_, row) in zip(cols, modules.iloc[start:start + 5].iterrows()):
            score = _to_float(row.get("score_0_100"), 0) or 0
            accent = _depth5_accent(score, row.get("status"))
            with col:
                _render_html(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {accent}; border-radius:8px; padding:13px 14px; min-height:245px; margin-bottom:12px;">
                      <div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">{_esc(row.get("module"), "Module")}</div>
                      <div style="font-size:26px; color:#111827; font-weight:900; line-height:1.1; margin-top:8px;">{score:.1f}<span style="font-size:13px; color:#6b7280;"> / 100</span></div>
                      <div style="font-size:13px; color:{accent}; font-weight:850; margin-top:6px;">{_esc(_human_text(row.get("status"), 80))}</div>
                      <div style="font-size:12px; color:#374151; line-height:1.38; margin-top:10px;"><b>Can size from this?</b> {_esc(_human_text(row.get("can_use_for_sizing"), 140))}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:12px; color:#4b5563; line-height:1.38;"><b>Fix next:</b> {_esc(_human_text(row.get("next_action"), 180))}</div>
                    </div>
                    """
                )


def _render_institutional_depth5_workbench():
    state = safe_json(ROOT / "institutional_depth5_state.json")
    modules = safe_csv(ROOT / "institutional_depth5_module_scorecard.csv")
    queue = safe_csv(ROOT / "institutional_depth5_priority_queue.csv")
    backtest = safe_csv(ROOT / "depth5_backtest_credibility_center.csv")
    signal = safe_csv(ROOT / "depth5_signal_ic_decay_failure_lab.csv")
    portfolio = safe_csv(ROOT / "depth5_portfolio_optimizer_v2.csv")
    execution = safe_csv(ROOT / "depth5_execution_liquidity_desk.csv")
    news = safe_csv(ROOT / "depth5_news_causal_proof_system.csv")

    if not state and modules.empty and queue.empty:
        st.info("Institutional Depth Workbench has not run yet. Run Step194 or the daily system.")
        return

    st.markdown('<p class="section-title">Institutional Depth Workbench</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Five deeper checks: backtest trust, signal proof, portfolio weights, trading cost, and news-to-industry proof. Research only; no live orders.</p>',
        unsafe_allow_html=True,
    )

    score = _to_float(state.get("overall_score_0_100"), 0) or 0
    status = _human_text(state.get("overall_status"), 80)
    accent = _depth5_accent(score, status)
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Depth status · {_esc(status)}</div>
          <div style="font-size:28px; color:#111827; font-weight:900; line-height:1.2; margin-top:6px;">{score:.1f} / 100</div>
          <div style="font-size:15px; color:#374151; line-height:1.55; margin-top:9px;">{_esc(_human_text(state.get("plain_answer"), 260))}</div>
          <div style="font-size:13px; color:#6b7280; line-height:1.45; margin-top:8px;">Sizing-ready modules: {_esc(state.get("sizing_ready_modules", 0))}. Repair/prototype modules: {_esc(state.get("repair_or_prototype_modules", 0))}. First queue items: {_esc(state.get("priority_queue_rows", 0))}.</div>
        </div>
        """
    )

    _render_depth5_module_cards(modules)

    st.markdown("##### Fix these first")
    st.markdown("This is the practical repair queue. If this list is long, the dashboard should stay in research mode.")
    if queue.empty:
        st.success("No depth repair queue is active.")
    else:
        display = queue.copy()
        if "priority" in display.columns:
            display = display.sort_values("priority").head(12)
        cols = [c for c in ["priority", "module", "item", "why_it_matters", "next_action", "source_files"] if c in display.columns]
        _show_status_table(_prepare_source_display(display[cols]), ["priority"], height=430)

    detail_tabs = st.tabs(["Backtest Trust", "Signal Lab", "Optimizer", "Execution Cost", "News Chain"])

    with detail_tabs[0]:
        st.markdown("##### Backtest Credibility Center")
        st.markdown("Question: can we trust the historical results enough to size from them? If the answer is weak, the backtest is only a research clue.")
        cols = [c for c in ["check", "score_0_100", "plain_status", "can_use_for_sizing", "what_it_means", "missing_proof", "source_files"] if c in backtest.columns]
        _show_status_table(_prepare_source_display(backtest[cols] if cols else backtest), ["plain_status", "can_use_for_sizing"], height=520)

    with detail_tabs[1]:
        st.markdown("##### Signal IC / Decay / Failure Lab")
        st.markdown("Question: which signals actually worked, how long they worked for, and which ones should be blocked or down-weighted?")
        cols = [c for c in ["signal", "best_horizon", "best_mean_ic", "worst_horizon", "worst_mean_ic", "sample_windows", "recommended_action", "live_vs_backtest_status", "failure_mode"] if c in signal.columns]
        _show_status_table(signal[cols] if cols else signal, ["recommended_action", "live_vs_backtest_status"], height=520)

    with detail_tabs[2]:
        st.markdown("##### Portfolio Optimizer 2.0")
        st.markdown("Question: after risk, signal confidence, sector budget, correlation, and trading cost, what weight is still allowed?")
        cols = [c for c in ["ticker", "sector", "sleeve", "current_weight_pct", "math_optimizer_wants_pct", "risk_allows_pct", "robust_weight_v2_pct", "portfolio_v2_decision", "confidence_0_100", "why", "what_would_unlock"] if c in portfolio.columns]
        _show_status_table(portfolio[cols] if cols else portfolio, ["portfolio_v2_decision"], height=560)

    with detail_tabs[3]:
        st.markdown("##### Execution Cost / Liquidity Desk")
        st.markdown("Question: would spread, volume, failed fills, or auction timing eat the idea before it becomes real?")
        cols = [c for c in ["ticker", "direction", "execution_permission", "execution_status", "spread_bps", "base_cost_bps", "stress_cost_bps", "expected_fill_rate_pct", "liquidity_read", "monitor_status", "what_to_do"] if c in execution.columns]
        _show_status_table(execution[cols] if cols else execution, ["execution_permission", "execution_status", "monitor_status"], height=560)

    with detail_tabs[4]:
        st.markdown("##### News-to-industry story proof")
        st.markdown("Question: if news helps one company, which suppliers, customers, peers, or vulnerable names should also be checked, and what proof is still missing?")
        cols = [c for c in ["ticker", "tone", "best_event_score", "causal_confidence_0_100", "causal_permission", "industry_chain_read", "route_before_risk", "risk_gate", "top_headline", "proof_needed"] if c in news.columns]
        _show_status_table(news[cols] if cols else news, ["tone", "causal_permission", "risk_gate"], height=620)


def _promotion_accent(permission: str) -> str:
    text = str(permission or "").lower()
    if "do not add" in text or "no new" in text:
        return "#991b1b"
    if "tiny" in text:
        return "#334155"
    if "study" in text or "research" in text:
        return "#111827"
    return "#166534"


def _render_promotion_ticker_cards(drilldown: pd.DataFrame, gate: pd.DataFrame, max_cards: int = 8):
    if drilldown.empty:
        st.info("No ticker cards from the final gate yet.")
        return

    order = []
    if not gate.empty and "ticker" in gate.columns:
        order = gate["ticker"].astype(str).head(max_cards).tolist()
    work = drilldown.copy()
    if order:
        work["_order"] = work["ticker"].astype(str).map({ticker: i for i, ticker in enumerate(order)}).fillna(999)
        work = work.sort_values("_order").drop(columns=["_order"])
    work = work.head(max_cards)

    for start in range(0, len(work), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, work.iloc[start:start + 4].iterrows()):
            answer = _human_text(row.get("top_answer"), 150)
            accent = _promotion_accent(answer)
            with col:
                _render_html(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:13px 14px; min-height:330px; margin-bottom:12px;">
                      <div style="font-size:22px; color:#111827; font-weight:900; line-height:1.1;">{_esc(row.get("ticker"), "")}</div>
                      <div style="font-size:14px; color:#111827; font-weight:850; line-height:1.3; margin-top:8px;">{_esc(answer)}</div>
                      <div style="font-size:12px; color:#374151; line-height:1.4; margin-top:9px;"><b>Why:</b> {_esc(_human_text(row.get("why"), 170))}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px; font-size:12px; color:#374151; line-height:1.38;"><b>Short:</b> {_esc(_human_text(row.get("short_term"), 80))}<br><b>Medium:</b> {_esc(_human_text(row.get("medium_term"), 80))}<br><b>Long:</b> {_esc(_human_text(row.get("long_term"), 80))}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.38;"><b>Next:</b> {_esc(_human_text(row.get("proof_needed"), 150))}<br><b>Click:</b> {_esc(row.get("where_to_click"), "Risk")}</div>
                    </div>
                    """
                )


def _render_promotion_gate_panel(compact: bool = False, label: str = "Home"):
    state = safe_json(ROOT / "institutional_promotion_gate_state.json")
    gate = safe_csv(ROOT / "institutional_promotion_gate.csv")
    drilldown = safe_csv(ROOT / "institutional_ticker_drilldown_cards.csv")
    horizon = safe_csv(ROOT / "institutional_horizon_route_matrix.csv")
    vehicle = safe_csv(ROOT / "institutional_vehicle_permission_matrix.csv")
    queue = safe_csv(ROOT / "institutional_promotion_queue.csv")

    if not state and gate.empty:
        st.info("Final PM Gate has not run yet. Run Step195 or the daily system.")
        return

    st.markdown('<p class="section-title">Final PM Gate</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">The final decision layer: data, risk, signal proof, news proof, portfolio weight, execution cost, and options route all have to agree before a ticker can move forward.</p>',
        unsafe_allow_html=True,
    )

    answer = _human_text(state.get("plain_answer"), 280)
    do_not = int(_to_float(state.get("do_not_add_count"), 0) or 0)
    study = int(_to_float(state.get("study_only_count"), 0) or 0)
    tiny = int(_to_float(state.get("tiny_paper_review_count"), 0) or 0)
    paper = int(_to_float(state.get("paper_allowed_now_count"), 0) or 0)
    options = int(_to_float(state.get("options_allowed_now_count"), 0) or 0)
    accent = "#991b1b" if do_not else "#334155" if study else "#166534"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Final answer</div>
          <div style="font-size:26px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(answer)}</div>
          <div style="font-size:13px; color:#6b7280; line-height:1.45; margin-top:8px;">No broker connection. No live orders. Options allowed now: {_esc(options)}.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Do Not Add", str(do_not), "Reduce or fix risk first.", "#991b1b"),
        ("Study Only", str(study), "Read and prove. No size.", "#111827"),
        ("Tiny Paper Review", str(tiny), "Only after final manual proof.", "#334155"),
        ("Paper Allowed Now", str(paper), "Research paper path only.", "#334155"),
        ("Options Allowed Now", str(options), "Should stay 0 until all gates clear.", "#991b1b" if options == 0 else "#166534"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    st.markdown("##### What to fix first")
    if queue.empty:
        st.success("No promotion queue is active.")
    else:
        q_cols = [c for c in ["priority", "ticker", "work", "why_it_matters", "next_step", "where_to_click"] if c in queue.columns]
        _show_status_table(queue[q_cols].head(10), ["priority"], height=360 if compact else 430)

    if compact:
        if not gate.empty:
            st.markdown("##### Final permission before ideas")
            g_cols = [c for c in ["ticker", "final_permission", "primary_route_now", "first_blocker", "where_to_click", "max_paper_weight_pct"] if c in gate.columns]
            _show_status_table(gate[g_cols].head(16), ["final_permission"], height=430)
        with st.expander(f"Open short / medium / long and vehicle routes ({label})", expanded=False):
            h_cols = [c for c in ["ticker", "horizon", "time_window", "plain_view", "allowed_vehicle", "option_side", "trigger_to_watch"] if c in horizon.columns]
            _show_status_table(horizon[h_cols].head(60), ["allowed_vehicle"], height=520)
            v_cols = [c for c in ["ticker", "stock_or_etf", "call", "put", "hedge", "max_paper_weight_pct"] if c in vehicle.columns]
            _show_status_table(vehicle[v_cols].head(40), ["call", "put", "hedge"], height=480)
        return

    st.markdown("##### One-page ticker cards")
    st.markdown("These are the names at the top of the final gate. They are written as decisions, not raw code.")
    _render_promotion_ticker_cards(drilldown, gate, max_cards=8)

    with st.expander(f"Open final permission table ({label})", expanded=False):
        g_cols = [c for c in ["ticker", "sector_or_theme", "final_permission", "primary_route_now", "confidence_0_100", "max_paper_weight_pct", "first_blocker", "next_step", "where_to_click"] if c in gate.columns]
        _show_status_table(gate[g_cols], ["final_permission"], height=620)

    with st.expander(f"Open one-page ticker drilldown table ({label})", expanded=False):
        d_cols = [c for c in ["ticker", "top_answer", "why", "short_term", "medium_term", "long_term", "stock_or_etf", "call", "put", "hedge", "proof_needed", "where_to_click"] if c in drilldown.columns]
        _show_status_table(drilldown[d_cols], ["top_answer"], height=620)

    with st.expander(f"Open short / medium / long route matrix ({label})", expanded=False):
        h_cols = [c for c in ["ticker", "horizon", "time_window", "plain_view", "allowed_vehicle", "option_side", "why_this_horizon", "trigger_to_watch", "invalidation"] if c in horizon.columns]
        _show_status_table(horizon[h_cols], ["allowed_vehicle"], height=620)

    with st.expander(f"Open stock / call / put / hedge permission matrix ({label})", expanded=False):
        v_cols = [c for c in ["ticker", "stock_or_etf", "call", "put", "hedge", "option_reason", "max_paper_weight_pct"] if c in vehicle.columns]
        _show_status_table(vehicle[v_cols], ["stock_or_etf", "call", "put", "hedge"], height=620)


def _render_decision_memory_center(compact: bool = False):
    state = safe_json(ROOT / "decision_memory_state.json")
    cards = safe_csv(ROOT / "decision_memory_review_cards.csv")
    ledger = safe_csv(ROOT / "decision_history_ledger.csv")
    forward = safe_csv(ROOT / "decision_forward_return_check.csv")
    false_lab = safe_csv(ROOT / "decision_false_positive_negative_lab.csv")
    calibration = safe_csv(ROOT / "decision_gate_calibration.csv")

    if not state and cards.empty:
        st.info("Decision Memory has not run yet. Run Step196 or the daily system.")
        return

    st.markdown('<p class="section-title">Decision Memory</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This checks whether yesterday’s gate was actually right later. Pending means the future price window has not arrived yet.</p>',
        unsafe_allow_html=True,
    )

    ready = int(_to_float(state.get("ready_forward_observations"), 0) or 0)
    pending = int(_to_float(state.get("pending_forward_observations"), 0) or 0)
    no_price = int(_to_float(state.get("no_price_observations"), 0) or 0)
    latest_price = _human_text(state.get("latest_price_date"), 80)
    accent = "#166534" if ready else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Memory answer</div>
          <div style="font-size:25px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 260))}</div>
          <div style="font-size:13px; color:#6b7280; line-height:1.45; margin-top:8px;">Latest local price date: {_esc(latest_price)}. Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    cols = st.columns(5)
    summary_cards = [
        ("Stored Decisions", str(state.get("ledger_decision_count", 0)), "Final Gate calls saved.", "#111827"),
        ("Ready Grades", str(ready), "Can be judged now.", "#166534" if ready else "#334155"),
        ("Waiting", str(pending), "Future price window not here yet.", "#334155"),
        ("No Price", str(no_price), "Need better price history.", "#991b1b" if no_price else "#166534"),
        ("Calibration Rows", str(state.get("calibration_rows", 0)), "Gate-level review groups.", "#111827"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, summary_cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not cards.empty:
        st.markdown("##### Plain-English review")
        _show_status_table(cards[[c for c in ["card", "answer", "why_it_matters", "next_step"] if c in cards.columns]], height=300 if compact else 380)

    if compact:
        with st.expander("Open memory details", expanded=False):
            f_cols = [c for c in ["decision_date", "ticker", "final_permission", "first_blocker", "horizon_days", "forward_return_pct", "observation_status", "status_note"] if c in forward.columns]
            _show_status_table(forward[f_cols].head(80), ["observation_status"], height=520)
        return

    detail_tabs = st.tabs(["Decision Ledger", "Forward Returns", "False Pos / Neg", "Gate Calibration"])
    with detail_tabs[0]:
        st.markdown("##### Decision History Ledger")
        st.markdown("Every Final PM Gate call is stored here so the system cannot quietly forget what it said.")
        l_cols = [c for c in ["decision_date", "ticker", "final_permission", "primary_route_now", "confidence_0_100", "first_blocker", "entry_price_date", "entry_price", "price_status", "next_step"] if c in ledger.columns]
        _show_status_table(ledger[l_cols].tail(120), ["final_permission", "price_status"], height=620)

    with detail_tabs[1]:
        st.markdown("##### Forward Return Check")
        st.markdown("1d, 5d, 21d, and 63d checks. Pending is honest: the future data does not exist yet.")
        f_cols = [c for c in ["decision_date", "ticker", "final_permission", "first_blocker", "horizon_days", "entry_price_date", "entry_price", "eval_price_date", "eval_price", "forward_return_pct", "observation_status", "status_note"] if c in forward.columns]
        _show_status_table(forward[f_cols].tail(240), ["observation_status"], height=680)

    with detail_tabs[2]:
        st.markdown("##### False Positive / False Negative Lab")
        st.markdown("This asks whether the system blocked winners, avoided losers, or stayed inconclusive.")
        fp_cols = [c for c in ["decision_date", "ticker", "final_permission", "first_blocker", "horizon_days", "forward_return_pct", "outcome_label", "plain_read", "observation_status"] if c in false_lab.columns]
        _show_status_table(false_lab[fp_cols].tail(240), ["outcome_label", "observation_status"], height=680)

    with detail_tabs[3]:
        st.markdown("##### Gate Calibration")
        st.markdown("Do not change gates from one anecdote. This table only becomes actionable after multiple matured observations.")
        cal_cols = [c for c in ["gate_or_blocker", "permission", "ready_observations", "pending_observations", "avg_forward_return_pct", "missed_winner_count", "protected_loss_count", "calibration_action", "plain_reason"] if c in calibration.columns]
        _show_status_table(calibration[cal_cols], ["calibration_action"], height=560)


def _render_data_reliability_center(compact: bool = False):
    state = safe_json(ROOT / "data_reliability_state.json")
    price_desk = safe_csv(ROOT / "price_refresh_desk.csv")
    forward_unlocker = safe_csv(ROOT / "forward_validation_unlocker.csv")
    repair_queue = safe_csv(ROOT / "data_gap_repair_queue.csv")
    scorecard = safe_csv(ROOT / "data_reliability_scorecard.csv")

    if not state and price_desk.empty:
        st.info("Data Reliability has not run yet. Run Step197 or the daily system.")
        return

    st.markdown('<p class="section-title">Data Reliability</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This checks whether local prices and proof files are fresh enough to read the dashboard honestly.</p>',
        unsafe_allow_html=True,
    )

    score = _to_float(state.get("overall_score_0_100"), 0) or 0
    status = _human_text(state.get("status"), 120)
    if score >= 80 and "repair" not in status.lower():
        accent = "#166534"
    elif score >= 55:
        accent = "#334155"
    else:
        accent = "#991b1b"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Can we trust today's local data?</div>
          <div style="font-size:25px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(status)} · {_esc(score)}/100</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.45; margin-top:8px;">{_esc(_human_text(state.get("plain_answer"), 340))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    cols = st.columns(5)
    stale_count = int(_to_float(state.get("stale_price_count"), 0) or 0)
    missing_count = int(_to_float(state.get("missing_price_count"), 0) or 0)
    ready_count = int(_to_float(state.get("ready_forward_observations"), 0) or 0)
    summary_cards = [
        ("Latest Price Date", _human_text(state.get("latest_price_date"), 80), "Newest local price in the system.", "#111827"),
        ("Fresh Prices", str(state.get("fresh_price_count", 0)), "Fresh enough for current research.", "#166534"),
        ("Stale Prices", str(stale_count), "Refresh before trusting today.", "#991b1b" if stale_count else "#334155"),
        ("Missing Prices", str(missing_count), "Cannot grade or size reliably.", "#991b1b" if missing_count else "#334155"),
        ("Forward Ready", str(ready_count), "Checks that can be judged now.", "#166534" if ready_count else "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, summary_cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not repair_queue.empty:
        st.markdown("##### Fix these first")
        repair_cols = [c for c in ["severity", "ticker", "repair_type", "plain_problem", "why_it_matters", "next_step", "owner_page"] if c in repair_queue.columns]
        _show_status_table(repair_queue[repair_cols].head(12), ["severity"], height=360 if compact else 440)
    else:
        st.success("No data repair queue is active.")

    if compact:
        with st.expander("Open Data Reliability details", expanded=False):
            if not scorecard.empty:
                s_cols = [c for c in ["score_component", "score_0_100", "plain_status", "next_step"] if c in scorecard.columns]
                _show_status_table(scorecard[s_cols], ["plain_status"], height=280)
            if not price_desk.empty:
                p_cols = [c for c in ["ticker", "latest_price_date", "days_stale_vs_today", "price_status", "can_validate_forward", "next_step"] if c in price_desk.columns]
                _show_status_table(price_desk[p_cols].head(80), ["price_status"], height=520)
        return

    detail_tabs = st.tabs(["Scorecard", "Price Desk", "Forward Unlocker", "Repair Queue"])
    with detail_tabs[0]:
        st.markdown("##### Scorecard")
        st.markdown("This is a data-quality score, not a trading signal.")
        s_cols = [c for c in ["score_component", "score_0_100", "plain_status", "why_it_matters", "next_step", "source_files"] if c in scorecard.columns]
        _show_status_table(scorecard[s_cols] if s_cols else scorecard, ["plain_status"], height=420)

    with detail_tabs[1]:
        st.markdown("##### Price Desk")
        st.markdown("A ticker with missing or stale price data cannot be trusted for fresh trigger distance, forward validation, or sizing.")
        p_cols = [c for c in ["ticker", "latest_price_date", "latest_price", "days_stale_vs_today", "price_status", "can_validate_forward", "source_quality", "market_snapshot_confidence", "next_step"] if c in price_desk.columns]
        _show_status_table(price_desk[p_cols] if p_cols else price_desk, ["price_status"], height=620)

    with detail_tabs[2]:
        st.markdown("##### Forward Unlocker")
        st.markdown("This tells you whether Decision Memory is blocked by missing data or simply waiting for future days to arrive.")
        f_cols = [c for c in ["ticker", "horizon_days", "observations", "ready_count", "pending_count", "no_price_count", "unlock_status", "what_is_needed"] if c in forward_unlocker.columns]
        _show_status_table(forward_unlocker[f_cols] if f_cols else forward_unlocker, ["unlock_status"], height=620)

    with detail_tabs[3]:
        st.markdown("##### Repair Queue")
        st.markdown("These are the data/proof repairs that should happen before the system becomes more confident.")
        r_cols = [c for c in ["severity", "ticker", "repair_type", "plain_problem", "why_it_matters", "next_step", "owner_page", "source_files"] if c in repair_queue.columns]
        _show_status_table(repair_queue[r_cols] if r_cols else repair_queue, ["severity"], height=680)


def _render_data_repair_center(compact: bool = False):
    state = safe_json(ROOT / "data_repair_state.json")
    priority = safe_csv(ROOT / "data_repair_priority_board.csv")
    attempts = safe_csv(ROOT / "price_repair_attempts.csv")
    risk_book = safe_csv(ROOT / "risk_book_repair_intake_queue.csv")
    news = safe_csv(ROOT / "news_proof_repair_queue.csv")
    execution = safe_csv(ROOT / "execution_spread_repair_queue.csv")

    if not state and priority.empty:
        st.info("Data Repair has not run yet. Run Step198 or the daily system.")
        return

    st.markdown('<p class="section-title">Data Repair</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This turns data gaps into a repair plan: price refresh, risk-book intake, news proof, and execution proof.</p>',
        unsafe_allow_html=True,
    )

    downloaded = int(_to_float(state.get("price_repair_downloaded_count"), 0) or 0)
    unresolved = int(_to_float(state.get("price_repair_unresolved_count"), 0) or 0)
    risk_count = int(_to_float(state.get("risk_book_repair_count"), 0) or 0)
    news_count = int(_to_float(state.get("news_proof_repair_count"), 0) or 0)
    execution_count = int(_to_float(state.get("execution_spread_repair_count"), 0) or 0)
    accent = "#991b1b" if unresolved or risk_count else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">What got repaired, and what is still blocked?</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 360))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Prices Downloaded", str(downloaded), "Supplemental public price rows.", "#166534" if downloaded else "#334155"),
        ("Still No Price", str(unresolved), "Needs manual price proof.", "#991b1b" if unresolved else "#334155"),
        ("Risk Book", str(risk_count), "Names needing risk intake.", "#991b1b" if risk_count else "#334155"),
        ("News Proof", str(news_count), "Causal links to verify.", "#334155"),
        ("Execution Proof", str(execution_count), "Spread/liquidity rows to fix.", "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not priority.empty:
        st.markdown("##### Repair order")
        p_cols = [c for c in ["priority", "workstream", "plain_answer", "why_it_matters", "next_step"] if c in priority.columns]
        _show_status_table(priority[p_cols], ["priority"], height=300 if compact else 360)

    if compact:
        with st.expander("Open repair details", expanded=False):
            a_cols = [c for c in ["ticker", "repair_status", "latest_download_date", "latest_download_close", "note"] if c in attempts.columns]
            _show_status_table(attempts[a_cols].head(80), ["repair_status"], height=500)
        return

    detail_tabs = st.tabs(["Price Repair", "Risk Book Intake", "News Proof", "Execution Proof"])
    with detail_tabs[0]:
        st.markdown("##### Price Repair")
        st.markdown("Downloaded rows are stored in a supplemental cache. Original historical files are not overwritten.")
        a_cols = [c for c in ["ticker", "repair_status", "latest_download_date", "latest_download_close", "note"] if c in attempts.columns]
        _show_status_table(attempts[a_cols] if a_cols else attempts, ["repair_status"], height=620)

    with detail_tabs[1]:
        st.markdown("##### Risk Book Intake")
        st.markdown("These names cannot move to paper, calls, or puts until their risk-book entry is filled.")
        r_cols = [c for c in ["priority", "ticker", "sector_or_theme", "main_blocker", "risk_level", "liquidity", "daily_cvar_95_pct", "starter_cap_after_all_gates_clear_pct", "fields_to_fill", "done_when", "still_forbidden"] if c in risk_book.columns]
        _show_status_table(risk_book[r_cols] if r_cols else risk_book, ["priority", "risk_level"], height=680)

    with detail_tabs[2]:
        st.markdown("##### News Proof")
        st.markdown("A headline is only useful after target, timing, and post-news price reaction are proven.")
        n_cols = [c for c in ["priority", "ticker", "headline", "tone", "why_blocked", "proof_to_collect", "done_when", "source_link"] if c in news.columns]
        _show_status_table(news[n_cols] if n_cols else news, ["priority"], height=680)

    with detail_tabs[3]:
        st.markdown("##### Execution Proof")
        st.markdown("This is where spread, liquidity, and option-route proof must be fixed before paper action.")
        e_cols = [c for c in ["priority", "ticker", "repair_type", "current_block", "cost_read", "proof_to_collect", "done_when"] if c in execution.columns]
        _show_status_table(execution[e_cols] if e_cols else execution, ["priority", "repair_type"], height=680)


def _render_risk_book_seed_center(compact: bool = False):
    state = safe_json(ROOT / "risk_book_seed_state.json")
    approval = safe_csv(ROOT / "risk_book_seed_manual_approval_queue.csv")
    metrics = safe_csv(ROOT / "risk_book_seed_metric_detail.csv")
    entries = safe_csv(ROOT / "risk_book_seed_entries.csv")
    sector = safe_csv(ROOT / "risk_book_seed_sector_exposure_preview.csv")

    if not state and approval.empty:
        st.info("Risk Book Seed has not run yet. Run Step199 or the daily system.")
        return

    st.markdown('<p class="section-title">Risk Book Seed</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This converts blank risk-book gaps into review-only seed entries. A seed is not permission to buy, size, call, or put.</p>',
        unsafe_allow_html=True,
    )

    seed_count = int(_to_float(state.get("seed_entry_count"), 0) or 0)
    approval_count = int(_to_float(state.get("manual_approval_count"), 0) or 0)
    p1_count = int(_to_float(state.get("high_priority_approval_count"), 0) or 0)
    priced_count = int(_to_float(state.get("seed_with_price_metrics_count"), 0) or 0)
    very_high = int(_to_float(state.get("very_high_risk_count"), 0) or 0)
    accent = "#991b1b" if p1_count else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">What changed?</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 360))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Still forbidden: no paper size, no calls, no puts, no live orders from seed entries alone.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Seed Entries", str(seed_count), "Blank risk gaps now have first-pass facts.", "#166534" if seed_count else "#334155"),
        ("Need Approval", str(approval_count), "Still not tradeable.", "#991b1b" if approval_count else "#334155"),
        ("P1 Reviews", str(p1_count), "Highest-risk manual checks.", "#991b1b" if p1_count else "#334155"),
        ("With Price Metrics", str(priced_count), "VaR / CVaR calculated.", "#166534" if priced_count else "#334155"),
        ("Very High Risk", str(very_high), "Tiny or no seed cap.", "#991b1b" if very_high else "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not approval.empty:
        st.markdown("##### Approve these before any promotion")
        a_cols = [c for c in ["priority", "ticker", "sector_or_theme", "risk_level", "starter_cap_if_approved_pct", "paper_stop_if_ever_tested_pct", "manual_items_open", "still_forbidden"] if c in approval.columns]
        _show_status_table(approval[a_cols].head(12), ["priority", "risk_level"], height=360 if compact else 460)

    if compact:
        with st.expander("Open risk seed details", expanded=False):
            m_cols = [c for c in ["ticker", "risk_level", "price_data_status", "daily_cvar_95_pct", "corr_spy", "corr_qqq", "corr_smh", "seed_cap_after_manual_approval_pct", "paper_stop_if_ever_tested_pct"] if c in metrics.columns]
            _show_status_table(metrics[m_cols].head(80), ["risk_level"], height=520)
        return

    detail_tabs = st.tabs(["Approval Queue", "Risk Metrics", "Seed Entries", "Sector Preview"])
    with detail_tabs[0]:
        st.markdown("##### Manual Approval Queue")
        st.markdown("These are review packets, not an action list.")
        a_cols = [c for c in ["priority", "ticker", "sector_or_theme", "risk_seed_status", "risk_level", "starter_cap_if_approved_pct", "paper_stop_if_ever_tested_pct", "manual_items_open", "done_when", "still_forbidden"] if c in approval.columns]
        _show_status_table(approval[a_cols] if a_cols else approval, ["priority", "risk_level"], height=680)

    with detail_tabs[1]:
        st.markdown("##### Risk Metrics")
        st.markdown("First-pass VaR/CVaR, volatility, drawdown, liquidity, and factor correlation from local/public prices.")
        m_cols = [c for c in ["ticker", "sector_or_theme", "price_data_status", "latest_price_date", "annual_vol_pct", "daily_cvar_95_pct", "max_drawdown_1y_pct", "liquidity_status", "corr_spy", "corr_qqq", "corr_smh", "beta_spy", "risk_level", "seed_cap_after_manual_approval_pct", "paper_stop_if_ever_tested_pct"] if c in metrics.columns]
        _show_status_table(metrics[m_cols] if m_cols else metrics, ["risk_level", "liquidity_status"], height=680)

    with detail_tabs[2]:
        st.markdown("##### Seed Entries")
        st.markdown("These rows can help the Final PM Gate stop saying blank risk book, but they do not approve sizing.")
        e_cols = [c for c in ["ticker", "sector", "final_risk_action", "risk_level", "liquidity_status", "seed_cap_after_manual_approval_pct", "paper_stop_if_ever_tested_pct", "reason_stack"] if c in entries.columns]
        _show_status_table(entries[e_cols] if e_cols else entries, ["final_risk_action", "risk_level"], height=680)

    with detail_tabs[3]:
        st.markdown("##### Sector Preview")
        st.markdown("This shows where the new review-only risk seeds are clustered.")
        s_cols = [c for c in ["sector_or_theme", "seed_ticker_count", "avg_seed_cap_after_manual_approval_pct", "high_or_very_high_count", "missing_liquidity_count"] if c in sector.columns]
        _show_status_table(sector[s_cols] if s_cols else sector, [], height=460)


def _render_risk_seed_approval_workbench(compact: bool = False):
    state = safe_json(ROOT / "risk_seed_approval_state.json")
    rank = safe_csv(ROOT / "risk_seed_approval_rank.csv")
    packets = safe_csv(ROOT / "risk_seed_approval_packets.csv")
    blockers = safe_csv(ROOT / "risk_seed_blocker_matrix.csv")
    sim = safe_csv(ROOT / "risk_seed_promotion_simulation.csv")

    if not state and rank.empty:
        st.info("Risk Seed Approval has not run yet. Run Step200 or the daily system.")
        return

    st.markdown('<p class="section-title">Risk Seed Approval</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This ranks review-only risk seeds. It tells you what can be reviewed first and what proof is still missing. It does not approve buying, sizing, calls, or puts.</p>',
        unsafe_allow_html=True,
    )

    seed_count = int(_to_float(state.get("seed_count"), 0) or 0)
    ready_count = int(_to_float(state.get("ready_for_pm_review_count"), 0) or 0)
    news_first = int(_to_float(state.get("news_proof_first_count"), 0) or 0)
    execution_first = int(_to_float(state.get("execution_proof_first_count"), 0) or 0)
    sandbox_count = int(_to_float(state.get("high_risk_sandbox_count"), 0) or 0)
    blocker_rows = int(_to_float(state.get("blocker_rows"), 0) or 0)
    accent = "#166534" if ready_count else "#991b1b"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Approval answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 360))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Simple rule: a risk seed is only a review packet. It is not permission to trade.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Seeds Ranked", str(seed_count), "Review-only names sorted by usefulness.", "#334155"),
        ("Closest To Review", str(ready_count), "Human review can start here.", "#166534" if ready_count else "#334155"),
        ("News Proof First", str(news_first), "Need source and price reaction.", "#991b1b" if news_first else "#334155"),
        ("Execution Proof First", str(execution_first), "Need spread and liquidity proof.", "#991b1b" if execution_first else "#334155"),
        ("High-Risk Sandbox", str(sandbox_count), "Only tiny review, or skip.", "#991b1b" if sandbox_count else "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not rank.empty:
        st.markdown("##### What to review first")
        st.markdown("Read this from left to right: ticker, review lane, score, risk, missing proof, then next step.")
        r_cols = [
            c for c in [
                "ticker",
                "sector_or_theme",
                "approval_lane",
                "approval_score_0_100",
                "risk_level",
                "starter_cap_if_approved_pct",
                "news_proof_open",
                "execution_proof_open",
                "open_blocker_count",
                "first_blocker",
                "next_step",
                "still_forbidden",
            ]
            if c in rank.columns
        ]
        _show_status_table(rank[r_cols].head(16 if compact else 40), ["approval_lane", "risk_level"], height=430 if compact else 620)

    if compact:
        with st.expander("Open approval packets and blocker examples", expanded=False):
            if not packets.empty:
                st.markdown("##### Review packets")
                p_cols = [c for c in ["ticker", "plain_answer", "why_review", "what_to_check_first", "news_and_execution", "option_rule"] if c in packets.columns]
                _show_status_table(packets[p_cols].head(24), [], height=420)
            if not blockers.empty:
                st.markdown("##### Blockers")
                b_cols = [c for c in ["ticker", "blocker_type", "severity", "plain_blocker", "what_to_collect", "source_files"] if c in blockers.columns]
                _show_status_table(blockers[b_cols].head(40), ["severity"], height=520)
        return

    detail_tabs = st.tabs(["Approval Rank", "Review Packets", "Blocker Matrix", "Promotion Simulation"])
    with detail_tabs[0]:
        st.markdown("##### Approval Rank")
        st.markdown("This is the human review queue. A high score means fewer missing facts, not permission to trade.")
        r_cols = [c for c in ["ticker", "sector_or_theme", "approval_lane", "approval_score_0_100", "why_this_lane", "risk_level", "daily_cvar_95_pct", "starter_cap_if_approved_pct", "paper_stop_if_ever_tested_pct", "liquidity_status", "news_proof_open", "execution_proof_open", "open_blocker_count", "first_blocker", "next_step", "source_files"] if c in rank.columns]
        _show_status_table(rank[r_cols] if r_cols else rank, ["approval_lane", "risk_level"], height=720)

    with detail_tabs[1]:
        st.markdown("##### Review Packets")
        st.markdown("Each row explains why the ticker exists in the review queue and what to check first.")
        p_cols = [c for c in ["ticker", "plain_answer", "why_review", "what_to_check_first", "news_and_execution", "current_final_gate", "current_first_blocker", "option_rule", "source_files"] if c in packets.columns]
        _show_status_table(packets[p_cols] if p_cols else packets, [], height=720)

    with detail_tabs[2]:
        st.markdown("##### Blocker Matrix")
        st.markdown("This is the proof checklist. Clear the blocker before asking whether a ticker deserves size or options.")
        b_cols = [c for c in ["ticker", "blocker_type", "severity", "plain_blocker", "what_to_collect", "source_files"] if c in blockers.columns]
        _show_status_table(blockers[b_cols] if b_cols else blockers, ["severity"], height=720)

    with detail_tabs[3]:
        st.markdown("##### Promotion Simulation")
        st.markdown("This shows what would still block the ticker even if a seed were manually approved.")
        s_cols = [c for c in ["ticker", "current_final_gate", "approval_lane", "if_seed_approved_next_state", "max_seed_cap_if_all_manual_gates_clear_pct", "still_blocks_after_seed_approval", "option_after_seed_approval", "source_files"] if c in sim.columns]
        _show_status_table(sim[s_cols] if s_cols else sim, ["approval_lane"], height=720)


def _render_risk_seed_pm_review_intake(compact: bool = False):
    state = safe_json(ROOT / "risk_seed_pm_review_state.json")
    status = safe_csv(ROOT / "risk_seed_pm_review_status.csv")
    todo = safe_csv(ROOT / "risk_seed_pm_review_todo.csv")
    template = safe_csv(ROOT / "risk_seed_pm_review_input.csv")
    audit = safe_csv(ROOT / "risk_seed_pm_review_audit.csv")

    if not state and status.empty:
        st.info("Risk Seed PM Review has not run yet. Run Step201 or the daily system.")
        return

    st.markdown('<p class="section-title">PM Review Intake</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This is the human evidence desk for risk seeds. Fill the review template one ticker at a time; the system checks whether the evidence is complete enough for the next gate.</p>',
        unsafe_allow_html=True,
    )

    rows = int(_to_float(state.get("template_rows"), 0) or 0)
    ready = int(_to_float(state.get("ready_for_final_gate_check_count"), 0) or 0)
    incomplete = int(_to_float(state.get("incomplete_approval_count"), 0) or 0)
    blocked = int(_to_float(state.get("approval_blocked_count"), 0) or 0)
    not_started = int(_to_float(state.get("not_started_count"), 0) or 0)
    todo_count = int(_to_float(state.get("todo_count"), 0) or 0)
    accent = "#166534" if ready else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Human review status</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 360))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Editable evidence file: risk_seed_pm_review_input.csv. A completed review still does not create an order.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Review Rows", str(rows), "One row per risk seed.", "#334155"),
        ("Next Gate Ready", str(ready), "Complete enough for another gate.", "#166534" if ready else "#334155"),
        ("Incomplete Approvals", str(incomplete), "A reviewer tried to approve, but proof is missing.", "#991b1b" if incomplete else "#334155"),
        ("Hard Stops", str(blocked), "A hard rule stops promotion.", "#991b1b" if blocked else "#334155"),
        ("Not Reviewed", str(not_started), f"{todo_count} proof tasks waiting.", "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not status.empty:
        st.markdown("##### Current review state")
        st.markdown("This tells you whether the human review file is empty, incomplete, blocked, or ready for the next gate.")
        s_cols = [
            c for c in [
                "ticker",
                "review_status",
                "review_state",
                "allowed_next_state",
                "proof_score_0_100",
                "approval_lane",
                "risk_level",
                "system_seed_cap_pct",
                "approved_cap_pct",
                "missing_fields_plain",
                "option_rule",
            ]
            if c in status.columns
        ]
        _show_status_table(status[s_cols].head(16 if compact else 60), ["review_state", "risk_level"], height=430 if compact else 660)

    if compact:
        with st.expander("Open PM review to-do list", expanded=False):
            t_cols = [c for c in ["ticker", "priority", "what_to_fix", "why_it_matters", "where_to_fill"] if c in todo.columns]
            _show_status_table(todo[t_cols].head(60) if t_cols else todo.head(60), ["priority"], height=520)
        return

    detail_tabs = st.tabs(["Review Status", "What To Fill", "Editable Template", "Field Audit"])
    with detail_tabs[0]:
        st.markdown("##### Review Status")
        st.markdown("A ticker can only move forward after the PM evidence packet is complete and conservative.")
        s_cols = [c for c in ["ticker", "review_status", "review_state", "allowed_next_state", "proof_score_0_100", "approval_lane", "risk_level", "system_seed_cap_pct", "approved_cap_pct", "paper_stop_pct", "open_blocker_types", "missing_fields_plain", "option_rule", "source_files"] if c in status.columns]
        _show_status_table(status[s_cols] if s_cols else status, ["review_state", "risk_level"], height=720)

    with detail_tabs[1]:
        st.markdown("##### What To Fill")
        st.markdown("Start here if a ticker looks interesting. The row explains exactly what proof is missing.")
        t_cols = [c for c in ["ticker", "priority", "what_to_fix", "why_it_matters", "where_to_fill"] if c in todo.columns]
        _show_status_table(todo[t_cols] if t_cols else todo, ["priority"], height=720)

    with detail_tabs[2]:
        st.markdown("##### Editable Template")
        st.markdown("This is the file a human fills. The dashboard shows it for visibility; the script preserves existing entries when rerun.")
        temp_cols = [c for c in ["ticker", "approval_lane", "risk_level", "system_seed_cap_pct", "system_stop_pct", "review_status", "reviewer", "review_date", "approved_cap_pct", "paper_stop_pct", "thesis_plain", "earnings_date", "expected_event_move_pct", "event_size_policy", "liquidity_snapshot_date", "bid_ask_spread_bps", "sector_confirmed", "crowding_check", "option_route_requested", "decision_note"] if c in template.columns]
        _show_status_table(template[temp_cols] if temp_cols else template, ["review_status", "risk_level"], height=720)

    with detail_tabs[3]:
        st.markdown("##### Field Audit")
        st.markdown("This is a simple audit of which cells are still blank.")
        a_cols = [c for c in ["ticker", "field", "filled", "value_preview"] if c in audit.columns]
        _show_status_table(audit[a_cols] if a_cols else audit, ["filled"], height=720)


def _render_pm_review_evidence_autofill(compact: bool = False):
    state = safe_json(ROOT / "pm_review_evidence_autofill_state.json")
    suggestions = safe_csv(ROOT / "pm_review_evidence_autofill_suggestions.csv")
    draft = safe_csv(ROOT / "pm_review_evidence_autofill_draft.csv")
    coverage = safe_csv(ROOT / "pm_review_evidence_autofill_coverage.csv")

    if not state and suggestions.empty:
        st.info("Evidence Autofill has not run yet. Run Step203 or the daily system.")
        return

    st.markdown('<p class="section-title">Evidence Autofill</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This pre-fills review evidence from local files. It is a draft assistant only: it does not approve a ticker, does not change review status, and does not allow options.</p>',
        unsafe_allow_html=True,
    )

    review_rows = int(_to_float(state.get("review_rows"), 0) or 0)
    suggestion_count = int(_to_float(state.get("suggestion_count"), 0) or 0)
    draft_cells = int(_to_float(state.get("draft_filled_cells"), 0) or 0)
    high_conf = int(_to_float(state.get("high_confidence_suggestions"), 0) or 0)
    low_conf = int(_to_float(state.get("low_confidence_suggestions"), 0) or 0)
    accent = "#166534" if suggestion_count else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Autofill answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 380))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Draft file: pm_review_evidence_autofill_draft.csv. Official review input remains unchanged until a human accepts evidence.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Review Rows", str(review_rows), "Rows scanned for blank evidence.", "#334155"),
        ("Suggestions", str(suggestion_count), "Field-level evidence suggestions.", "#166534" if suggestion_count else "#334155"),
        ("Draft Cells", str(draft_cells), "Blank cells filled in draft only.", "#166534" if draft_cells else "#334155"),
        ("High Confidence", str(high_conf), "Policy/date/default suggestions.", "#334155"),
        ("Low Confidence", str(low_conf), "Needs manual source check.", "#991b1b" if low_conf else "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not suggestions.empty:
        st.markdown("##### Evidence suggestions")
        st.markdown("Every suggestion shows the field, source, confidence, and whether it would fill the draft. Human confirmation is still required.")
        s_cols = [
            c for c in [
                "ticker",
                "field_name",
                "suggested_value",
                "confidence",
                "will_fill_draft",
                "human_confirmation_needed",
                "rationale",
                "source_files",
            ]
            if c in suggestions.columns
        ]
        _show_status_table(suggestions[s_cols].head(20 if compact else 120), ["confidence", "will_fill_draft"], height=460 if compact else 720)

    if compact:
        with st.expander("Open draft and coverage", expanded=False):
            c_cols = [c for c in ["field_name", "suggestion_count", "draft_fill_count", "high_confidence_count", "medium_confidence_count", "low_confidence_count", "human_confirmation_needed"] if c in coverage.columns]
            _show_status_table(coverage[c_cols] if c_cols else coverage, [], height=360)
            d_cols = [c for c in ["ticker", "review_status", "thesis_plain", "earnings_date", "expected_event_move_pct", "event_size_policy", "liquidity_snapshot_date", "avg_daily_dollar_volume_check", "sector_confirmed", "crowding_check", "news_proof_note", "execution_proof_note", "paper_stop_pct", "option_route_requested", "decision_note"] if c in draft.columns]
            _show_status_table(draft[d_cols].head(30) if d_cols else draft.head(30), ["review_status"], height=560)
        return

    detail_tabs = st.tabs(["Suggestions", "Draft Review File", "Coverage"])
    with detail_tabs[0]:
        st.markdown("##### Suggestions")
        s_cols = [c for c in ["ticker", "field_name", "suggested_value", "confidence", "existing_value", "will_fill_draft", "human_confirmation_needed", "rationale", "source_files"] if c in suggestions.columns]
        _show_status_table(suggestions[s_cols] if s_cols else suggestions, ["confidence", "will_fill_draft"], height=720)

    with detail_tabs[1]:
        st.markdown("##### Draft Review File")
        st.markdown("This is not the official PM review input. It is a draft created so a human can inspect possible evidence quickly.")
        d_cols = [c for c in ["ticker", "approval_lane", "risk_level", "review_status", "thesis_plain", "earnings_date", "expected_event_move_pct", "event_size_policy", "liquidity_snapshot_date", "bid_ask_spread_bps", "avg_daily_dollar_volume_check", "sector_confirmed", "crowding_check", "news_proof_note", "execution_proof_note", "paper_stop_pct", "option_route_requested", "decision_note", "last_updated"] if c in draft.columns]
        _show_status_table(draft[d_cols] if d_cols else draft, ["review_status", "risk_level"], height=720)

    with detail_tabs[2]:
        st.markdown("##### Coverage")
        st.markdown("This shows which PM review fields the system can help with and which still need outside/manual sourcing.")
        c_cols = [c for c in ["field_name", "suggestion_count", "draft_fill_count", "high_confidence_count", "medium_confidence_count", "low_confidence_count", "human_confirmation_needed"] if c in coverage.columns]
        _show_status_table(coverage[c_cols] if c_cols else coverage, [], height=520)


def _render_pm_evidence_acceptance_gate(compact: bool = False):
    state = safe_json(ROOT / "pm_review_evidence_acceptance_state.json")
    acceptance = safe_csv(ROOT / "pm_review_evidence_acceptance_input.csv")
    status = safe_csv(ROOT / "pm_review_evidence_acceptance_status.csv")
    ready_patch = safe_csv(ROOT / "pm_review_evidence_acceptance_ready_patch.csv")
    conflicts = safe_csv(ROOT / "pm_review_evidence_acceptance_conflicts.csv")

    if not state and acceptance.empty:
        st.info("Evidence Acceptance has not run yet. Run Step204 or the daily system.")
        return

    st.markdown('<p class="section-title">Evidence Acceptance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This is the human checkpoint after Evidence Autofill. It asks: do we accept this evidence, reject it, or still need an outside source? It does not change the official PM review file.</p>',
        unsafe_allow_html=True,
    )

    suggestions = int(_to_float(state.get("suggestion_count"), 0) or 0)
    accepted = int(_to_float(state.get("accepted_count"), 0) or 0)
    outside = int(_to_float(state.get("needs_external_confirmation_count"), 0) or 0)
    undecided = int(_to_float(state.get("undecided_count"), 0) or 0)
    patch_rows = int(_to_float(state.get("ready_patch_rows"), 0) or 0)
    conflict_count = int(_to_float(state.get("conflict_count"), 0) or 0)
    accent = "#166534" if accepted and not conflict_count else "#991b1b" if conflict_count else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Human evidence gate</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 400))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Editable decision file: pm_review_evidence_acceptance_input.csv. Accepted evidence becomes a copy-ready draft only; the official PM review file is still unchanged.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Suggestions", str(suggestions), "Evidence rows waiting for a human decision.", "#334155"),
        ("Accepted", str(accepted), "Human-approved evidence rows.", "#166534" if accepted else "#334155"),
        ("Outside Check", str(outside), "Rows that still need another source.", "#0f766e" if outside else "#334155"),
        ("Undecided", str(undecided), "Rows not reviewed yet.", "#991b1b" if undecided else "#334155"),
        ("Conflicts", str(conflict_count), f"{patch_rows} ticker patch rows ready.", "#991b1b" if conflict_count else "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not acceptance.empty:
        st.markdown("##### Evidence decision queue")
        st.markdown("Start here. Change Evidence Decision in the CSV to Accept, Reject, Needs outside confirmation, or Ignore. Add reviewer and date before accepting.")
        a_cols = [
            c for c in [
                "ticker",
                "field_name",
                "suggested_value",
                "confidence",
                "acceptance_status",
                "official_value",
                "how_to_decide",
                "reviewer",
                "review_date",
                "source_files",
            ]
            if c in acceptance.columns
        ]
        _show_status_table(acceptance[a_cols].head(20 if compact else 100), ["confidence", "acceptance_status"], height=470 if compact else 720)

    if compact:
        with st.expander("Open accepted evidence, patch draft, and conflicts", expanded=False):
            s_cols = [c for c in ["ticker", "suggestion_count", "accepted_count", "needs_external_confirmation_count", "undecided_count", "conflict_count", "next_step"] if c in status.columns]
            _show_status_table(status[s_cols].head(50) if s_cols else status.head(50), ["conflict_count"], height=420)
            p_cols = [c for c in ["ticker", "patch_status", "accepted_field_count", "blocked_field_count", "thesis_plain", "news_proof_note", "execution_proof_note", "option_route_requested", "source_files"] if c in ready_patch.columns]
            _show_status_table(ready_patch[p_cols].head(30) if p_cols else ready_patch.head(30), ["patch_status"], height=520)
            if conflicts.empty:
                st.success("No evidence conflicts found yet.")
            else:
                c_cols = [c for c in ["ticker", "field_name", "acceptance_status", "conflict_reason", "suggested_value", "official_value", "source_files"] if c in conflicts.columns]
                _show_status_table(conflicts[c_cols].head(60) if c_cols else conflicts.head(60), ["acceptance_status"], height=520)
        return

    detail_tabs = st.tabs(["Decision Queue", "Ticker Status", "Ready Patch Draft", "Conflicts"])
    with detail_tabs[0]:
        st.markdown("##### Decision Queue")
        st.markdown("This is the editable review queue. It is where a human says whether each autofill suggestion can be trusted.")
        a_cols = [c for c in ["ticker", "field_name", "suggested_value", "confidence", "acceptance_status", "official_value", "will_fill_draft", "how_to_decide", "rationale", "reviewer", "review_date", "human_note", "source_files"] if c in acceptance.columns]
        _show_status_table(acceptance[a_cols] if a_cols else acceptance, ["confidence", "acceptance_status"], height=760)

    with detail_tabs[1]:
        st.markdown("##### Ticker Status")
        st.markdown("This groups the evidence decisions by ticker so you can see whether a name is actually ready or still waiting.")
        s_cols = [c for c in ["ticker", "suggestion_count", "accepted_count", "rejected_count", "needs_external_confirmation_count", "ignored_count", "undecided_count", "conflict_count", "accepted_fields", "next_step"] if c in status.columns]
        _show_status_table(status[s_cols] if s_cols else status, ["conflict_count"], height=640)

    with detail_tabs[2]:
        st.markdown("##### Ready Patch Draft")
        st.markdown("This is only a copy-ready draft for accepted evidence. It does not write into risk_seed_pm_review_input.csv by itself.")
        if ready_patch.empty:
            st.info("No accepted evidence yet, so there is no patch draft.")
        else:
            p_cols = [c for c in ["ticker", "patch_status", "accepted_field_count", "blocked_field_count", "thesis_plain", "earnings_date", "expected_event_move_pct", "event_size_policy", "liquidity_snapshot_date", "bid_ask_spread_bps", "avg_daily_dollar_volume_check", "sector_confirmed", "crowding_check", "news_proof_note", "execution_proof_note", "paper_stop_pct", "option_route_requested", "decision_note", "last_updated", "source_files"] if c in ready_patch.columns]
            _show_status_table(ready_patch[p_cols] if p_cols else ready_patch, ["patch_status"], height=720)

    with detail_tabs[3]:
        st.markdown("##### Conflicts")
        st.markdown("A conflict means the evidence is not clean enough to copy. Fix these first.")
        if conflicts.empty:
            st.success("No evidence conflicts found yet.")
        else:
            c_cols = [c for c in ["ticker", "field_name", "acceptance_status", "conflict_reason", "suggested_value", "official_value", "source_files"] if c in conflicts.columns]
            _show_status_table(conflicts[c_cols] if c_cols else conflicts, ["acceptance_status"], height=720)


def _render_pm_evidence_review_triage(compact: bool = False):
    state = safe_json(ROOT / "pm_evidence_review_triage_state.json")
    priority = safe_csv(ROOT / "pm_evidence_review_priority_queue.csv")
    field_plan = safe_csv(ROOT / "pm_evidence_review_field_plan.csv")
    packets = safe_csv(ROOT / "pm_evidence_review_packet_cards.csv")
    source_ladder = safe_csv(ROOT / "pm_evidence_review_source_ladder.csv")

    if not state and priority.empty:
        st.info("Evidence Review Triage has not run yet. Run Step205 or the daily system.")
        return

    st.markdown('<p class="section-title">Evidence Review Triage</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This turns the long evidence queue into a short PM review route: which ticker first, which evidence fields first, and which items require an outside source.</p>',
        unsafe_allow_html=True,
    )

    tickers = int(_to_float(state.get("ticker_count"), 0) or 0)
    fields = int(_to_float(state.get("field_review_count"), 0) or 0)
    high = int(_to_float(state.get("high_priority_field_count"), 0) or 0)
    outside = int(_to_float(state.get("outside_check_field_count"), 0) or 0)
    top = _plain_status(state.get("top_review_ticker"), "No ticker")

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #334155; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Review route</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 420))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Use this to decide what evidence to review first. It is not a trade list and it does not unlock size or options.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Tickers", str(tickers), "Names with evidence to review.", "#334155"),
        ("Evidence Fields", str(fields), "Total field checks in the queue.", "#334155"),
        ("High Priority", str(high), "Fields to review before the rest.", "#991b1b" if high else "#334155"),
        ("Outside Checks", str(outside), "Needs source outside the model.", "#0f766e" if outside else "#334155"),
        ("Start With", top, "Top ticker for a focused review pass.", "#166534" if top != "No ticker" else "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not priority.empty:
        st.markdown("##### Start here")
        st.markdown("This is the order for reviewing evidence. It narrows the queue so you do not have to read hundreds of rows at once.")
        p_cols = [
            c for c in [
                "review_order",
                "ticker",
                "triage_rank_score",
                "approval_lane",
                "risk_level",
                "open_blocker_count",
                "high_priority_evidence_fields",
                "outside_checks_needed",
                "undecided_count",
                "first_checks_plain",
                "outside_sources_to_check",
                "next_step",
            ]
            if c in priority.columns
        ]
        _show_status_table(priority[p_cols].head(8 if compact else 30), ["risk_level"], height=520 if compact else 720)

    if compact:
        with st.expander("Open ticker packets and source ladder", expanded=False):
            card_cols = [c for c in ["ticker", "plain_headline", "why_this_ticker_first", "first_5_checks", "outside_sources_to_check", "what_not_to_do", "editable_file"] if c in packets.columns]
            _show_status_table(packets[card_cols].head(12) if card_cols else packets.head(12), [], height=560)
            ladder_cols = [c for c in ["field_group", "source_check_type", "field_count", "high_priority_count", "example_tickers", "plain_rule"] if c in source_ladder.columns]
            _show_status_table(source_ladder[ladder_cols] if ladder_cols else source_ladder, [], height=420)
        return

    detail_tabs = st.tabs(["Review Order", "Ticker Packets", "Field Plan", "Source Ladder"])
    with detail_tabs[0]:
        st.markdown("##### Review Order")
        st.markdown("A higher rank means the ticker is closer to a useful PM evidence pass, not that it is approved.")
        p_cols = [c for c in ["review_order", "ticker", "triage_rank_score", "approval_lane", "risk_level", "approval_score_0_100", "open_blocker_count", "high_priority_evidence_fields", "outside_checks_needed", "undecided_count", "accepted_count", "first_checks_plain", "outside_sources_to_check", "next_step"] if c in priority.columns]
        _show_status_table(priority[p_cols] if p_cols else priority, ["risk_level"], height=760)

    with detail_tabs[1]:
        st.markdown("##### Ticker Packets")
        st.markdown("One row is one human-readable packet. It explains what to check first and what not to do.")
        card_cols = [c for c in ["ticker", "plain_headline", "why_this_ticker_first", "first_5_checks", "outside_sources_to_check", "what_not_to_do", "editable_file", "source_files"] if c in packets.columns]
        _show_status_table(packets[card_cols] if card_cols else packets, [], height=760)

    with detail_tabs[2]:
        st.markdown("##### Field Plan")
        st.markdown("This is the detailed evidence checklist after sorting. Use it when a ticker packet needs more detail.")
        f_cols = [c for c in ["ticker", "field_group", "field_name", "review_priority", "review_priority_score", "source_check_type", "acceptance_status", "confidence", "suggested_value", "what_to_do", "why_it_matters", "source_files"] if c in field_plan.columns]
        _show_status_table(field_plan[f_cols] if f_cols else field_plan, ["review_priority", "source_check_type"], height=760)

    with detail_tabs[3]:
        st.markdown("##### Source Ladder")
        st.markdown("This separates evidence that can be reviewed inside the system from evidence that needs an outside source.")
        ladder_cols = [c for c in ["field_group", "source_check_type", "field_count", "high_priority_count", "example_tickers", "plain_rule"] if c in source_ladder.columns]
        _show_status_table(source_ladder[ladder_cols] if ladder_cols else source_ladder, [], height=520)


def _render_pm_evidence_source_proof_desk(compact: bool = False):
    state = safe_json(ROOT / "pm_evidence_source_proof_state.json")
    proof_input = safe_csv(ROOT / "pm_evidence_source_proof_input.csv")
    status = safe_csv(ROOT / "pm_evidence_source_proof_status.csv")
    ready = safe_csv(ROOT / "pm_evidence_source_proof_ready_for_acceptance.csv")
    gaps = safe_csv(ROOT / "pm_evidence_source_proof_gap_queue.csv")

    if not state and proof_input.empty:
        st.info("Source Proof Desk has not run yet. Run Step206 or the daily system.")
        return

    st.markdown('<p class="section-title">Source Proof Desk</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This is where outside checks become concrete proof tasks: source, observed value, reviewer, date, and for news, price/volume reaction.</p>',
        unsafe_allow_html=True,
    )

    proof_rows = int(_to_float(state.get("proof_row_count"), 0) or 0)
    ready_count = int(_to_float(state.get("ready_for_accept_count"), 0) or 0)
    needs = int(_to_float(state.get("needs_proof_count"), 0) or 0)
    high_needs = int(_to_float(state.get("high_priority_needs_proof_count"), 0) or 0)
    top = _plain_status(state.get("top_missing_proof_ticker"), "No ticker")
    accent = "#166534" if ready_count and not needs else "#991b1b" if high_needs else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Outside proof answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 420))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Editable proof file: pm_evidence_source_proof_input.csv. This does not update Step204 automatically and does not approve size or options.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Proof Rows", str(proof_rows), "Outside checks converted to proof tasks.", "#334155"),
        ("Ready", str(ready_count), "Can be manually accepted in Step204.", "#166534" if ready_count else "#334155"),
        ("Needs Proof", str(needs), "Still missing source or reviewer facts.", "#991b1b" if needs else "#334155"),
        ("High Priority Missing", str(high_needs), "Most important proof gaps.", "#991b1b" if high_needs else "#334155"),
        ("Start With", top, "Ticker with the largest missing proof burden.", "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not gaps.empty:
        st.markdown("##### Missing proof first")
        st.markdown("These are the outside-source checks to fill before accepting the matching Step204 evidence row.")
        g_cols = [
            c for c in [
                "ticker",
                "field_group",
                "field_name",
                "review_priority",
                "missing_proof",
                "required_question",
                "preferred_source",
                "acceptable_proof",
                "next_step",
            ]
            if c in gaps.columns
        ]
        _show_status_table(gaps[g_cols].head(12 if compact else 80), ["review_priority"], height=520 if compact else 720)

    if compact:
        with st.expander("Open proof file, ticker status, and ready rows", expanded=False):
            s_cols = [c for c in ["ticker", "outside_proof_rows", "ready_for_accept_count", "needs_proof_count", "high_priority_needs_proof_count", "first_missing_proof", "next_step"] if c in status.columns]
            _show_status_table(status[s_cols].head(40) if s_cols else status.head(40), [], height=420)
            r_cols = [c for c in ["ticker", "field_group", "field_name", "observed_value", "source_name", "reviewer", "review_date", "step204_action_hint"] if c in ready.columns]
            _show_status_table(ready[r_cols].head(40) if r_cols else ready.head(40), [], height=420)
        return

    detail_tabs = st.tabs(["Missing Proof", "Editable Proof File", "Ticker Status", "Ready For Step204"])
    with detail_tabs[0]:
        st.markdown("##### Missing Proof")
        g_cols = [c for c in ["ticker", "field_group", "field_name", "review_priority", "missing_proof", "required_question", "preferred_source", "acceptable_proof", "next_step", "source_files"] if c in gaps.columns]
        _show_status_table(gaps[g_cols] if g_cols else gaps, ["review_priority"], height=760)

    with detail_tabs[1]:
        st.markdown("##### Editable Proof File")
        st.markdown("A human fills this. Set Proof Status to Verified only after source name, observed value, reviewer, and date are filled.")
        p_cols = [c for c in ["ticker", "field_group", "field_name", "review_priority", "proof_status", "suggested_value", "required_question", "preferred_source", "acceptable_proof", "source_name", "source_url", "observed_value", "observed_time", "price_reaction_checked", "volume_reaction_checked", "reviewer", "review_date", "proof_note", "step204_action_hint"] if c in proof_input.columns]
        _show_status_table(proof_input[p_cols] if p_cols else proof_input, ["proof_status", "review_priority"], height=760)

    with detail_tabs[2]:
        st.markdown("##### Ticker Status")
        s_cols = [c for c in ["ticker", "outside_proof_rows", "ready_for_accept_count", "needs_proof_count", "high_priority_needs_proof_count", "rejected_or_unavailable_count", "first_missing_proof", "next_step"] if c in status.columns]
        _show_status_table(status[s_cols] if s_cols else status, [], height=640)

    with detail_tabs[3]:
        st.markdown("##### Ready For Step204")
        st.markdown("These rows have enough source proof to be manually accepted in Step204. This still does not update Step204 automatically.")
        if ready.empty:
            st.info("No outside-source proof is ready for Step204 acceptance yet.")
        else:
            r_cols = [c for c in ["ticker", "field_group", "field_name", "observed_value", "source_name", "source_url", "reviewer", "review_date", "step204_action_hint", "proof_note"] if c in ready.columns]
            _show_status_table(ready[r_cols] if r_cols else ready, [], height=620)


def _render_pm_evidence_proof_acceptance_bridge(compact: bool = False):
    state = safe_json(ROOT / "pm_evidence_proof_acceptance_bridge_state.json")
    bridge = safe_csv(ROOT / "pm_evidence_proof_acceptance_bridge.csv")
    patch = safe_csv(ROOT / "pm_evidence_proof_acceptance_patch.csv")
    conflicts = safe_csv(ROOT / "pm_evidence_proof_acceptance_conflicts.csv")

    if not state and bridge.empty and patch.empty:
        st.info("Proof-to-Acceptance Bridge has not run yet. Run Step207 or the daily system.")
        return

    st.markdown('<p class="section-title">Proof-to-Acceptance Bridge</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This checks whether verified outside proof can become a manual Step204 acceptance patch. It still does not edit Step204 automatically.</p>',
        unsafe_allow_html=True,
    )

    ready_rows = int(_to_float(state.get("ready_proof_rows"), 0) or 0)
    bridge_rows = int(_to_float(state.get("bridge_rows"), 0) or 0)
    patch_rows = int(_to_float(state.get("patch_rows"), 0) or 0)
    conflict_count = int(_to_float(state.get("conflict_count"), 0) or 0)
    step204_rows = int(_to_float(state.get("step204_rows"), 0) or 0)
    accent = "#166534" if patch_rows and not conflict_count else "#991b1b" if conflict_count else "#334155"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Acceptance bridge answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 420))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">This only prepares a manual copy patch. It does not approve a ticker, add size, or unlock options.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Ready Proof", str(ready_rows), "Verified proof rows from Step206.", "#166534" if ready_rows else "#334155"),
        ("Bridge Rows", str(bridge_rows), "Verified proof checked against Step204.", "#334155"),
        ("Patch Rows", str(patch_rows), "Manual Step204 accept rows ready.", "#166534" if patch_rows else "#334155"),
        ("Conflicts", str(conflict_count), "Must be fixed before copying.", "#991b1b" if conflict_count else "#334155"),
        ("Step204 Rows", str(step204_rows), "Evidence rows currently in Step204.", "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not patch.empty:
        st.markdown("##### Manual Step204 patch")
        st.markdown("Copy these rows manually into the matching Step204 evidence acceptance row, then rerun Step204.")
        p_cols = [c for c in ["ticker", "field_name", "step204_suggestion_id", "acceptance_status", "reviewer", "review_date", "human_note", "observed_value", "source_name", "source_url", "copy_instruction"] if c in patch.columns]
        _show_status_table(patch[p_cols].head(12 if compact else 80), ["acceptance_status"], height=520 if compact else 720)
    elif ready_rows == 0:
        st.info("No verified outside-source proof is ready yet. Fill Step206 proof rows first.")

    if compact:
        with st.expander("Open bridge rows and conflicts", expanded=False):
            b_cols = [c for c in ["ticker", "field_name", "bridge_state", "matching_step204_row_found", "current_step204_decision", "proposed_step204_decision", "observed_value", "step204_suggested_value", "next_step"] if c in bridge.columns]
            _show_status_table(bridge[b_cols].head(50) if b_cols else bridge.head(50), ["bridge_state"], height=420)
            if conflicts.empty:
                st.success("No proof-to-acceptance bridge conflicts.")
            else:
                c_cols = [c for c in ["ticker", "field_name", "conflict_reason", "current_step204_decision", "observed_value", "step204_suggested_value", "source_name"] if c in conflicts.columns]
                _show_status_table(conflicts[c_cols].head(80) if c_cols else conflicts.head(80), [], height=520)
        return

    detail_tabs = st.tabs(["Bridge Rows", "Manual Patch", "Conflicts"])
    with detail_tabs[0]:
        st.markdown("##### Bridge Rows")
        b_cols = [c for c in ["ticker", "field_name", "bridge_state", "matching_step204_row_found", "current_step204_decision", "proposed_step204_decision", "step204_suggestion_id", "observed_value", "step204_suggested_value", "source_name", "reviewer", "review_date", "next_step"] if c in bridge.columns]
        _show_status_table(bridge[b_cols] if b_cols else bridge, ["bridge_state"], height=720)

    with detail_tabs[1]:
        st.markdown("##### Manual Patch")
        if patch.empty:
            st.info("No manual Step204 patch rows are ready yet.")
        else:
            p_cols = [c for c in ["ticker", "field_name", "step204_suggestion_id", "acceptance_status", "reviewer", "review_date", "human_note", "observed_value", "source_name", "source_url", "copy_instruction"] if c in patch.columns]
            _show_status_table(patch[p_cols] if p_cols else patch, ["acceptance_status"], height=720)

    with detail_tabs[2]:
        st.markdown("##### Conflicts")
        if conflicts.empty:
            st.success("No proof-to-acceptance bridge conflicts.")
        else:
            c_cols = [c for c in ["ticker", "field_name", "conflict_reason", "current_step204_decision", "observed_value", "step204_suggested_value", "source_name", "reviewer", "review_date"] if c in conflicts.columns]
            _show_status_table(conflicts[c_cols] if c_cols else conflicts, [], height=720)


def _render_quant_fund_flow_navigator(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_flow_navigator_state.json")
    command = safe_json(ROOT / "quant_fund_flow_pm_command_center.json")
    current = safe_csv(ROOT / "quant_fund_flow_current_state.csv")
    blockers = safe_csv(ROOT / "quant_fund_flow_blocker_queue.csv")
    next_clicks = safe_csv(ROOT / "quant_fund_flow_next_clicks.csv")
    contracts = safe_csv(ROOT / "quant_fund_flow_stage_contracts.csv")

    if not state and current.empty:
        st.info("Today Flow Navigator has not run yet. Run Step209 or the daily system.")
        return

    st.markdown('<p class="section-title">Today Flow Navigator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This is the control tower: what mode the system is in, what blocks progress, and where to click next.</p>',
        unsafe_allow_html=True,
    )

    mode = _plain_status(command.get("today_mode") or state.get("today_mode"), "Research review")
    can_risk = _plain_status(command.get("can_take_new_risk") or state.get("can_take_new_risk"), "Only after final gate confirms")
    first_page = _plain_status(command.get("first_page") or state.get("first_page"), "Home")
    first_ticker = _plain_status(command.get("first_ticker") or state.get("first_ticker"), "No ticker")
    first_action = _plain_status(command.get("first_action"), state.get("plain_answer", "Read the first blocker."))

    accent = "#991b1b" if "No new risk" in can_risk else "#0f766e"
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">PM command center</div>
          <div style="font-size:25px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(command.get("plain_answer") or state.get("plain_answer"), 520))}</div>
          <div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; margin-top:14px;">
            <div><div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase;">Mode</div><div style="font-size:17px; font-weight:850; color:#111827;">{_esc(mode)}</div></div>
            <div><div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase;">New Risk</div><div style="font-size:17px; font-weight:850; color:#111827;">{_esc(can_risk)}</div></div>
            <div><div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase;">First Page</div><div style="font-size:17px; font-weight:850; color:#111827;">{_esc(first_page)}</div></div>
            <div><div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase;">First Ticker</div><div style="font-size:17px; font-weight:850; color:#111827;">{_esc(first_ticker)}</div></div>
          </div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:12px;"><b>First action:</b> {_esc(_human_text(first_action, 340))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Tickers", str(int(_to_float(state.get("ticker_count"), len(current) if not current.empty else 0) or 0)), "Tickers with a current state.", "#334155"),
        ("Blockers", str(int(_to_float(state.get("blocker_count"), len(blockers) if not blockers.empty else 0) or 0)), "Items that stop progress.", "#991b1b" if not blockers.empty else "#334155"),
        ("Proof First", str(int(_to_float(command.get("proof_first_count"), 0) or 0)), "Need outside-source proof.", "#991b1b"),
        ("Risk Blocked", str(int(_to_float(command.get("risk_blocked_count"), 0) or 0)), "Risk veto before ideas.", "#991b1b"),
        ("Execution Proof", str(int(_to_float(command.get("execution_proof_count"), 0) or 0)), "Need spread/liquidity proof.", "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not next_clicks.empty:
        st.markdown("##### Next clicks")
        st.markdown("Follow this order before looking for new trades.")
        click_cols = [c for c in ["order", "page", "panel", "what_to_read", "why_now", "done_when", "do_not_do"] if c in next_clicks.columns]
        _show_status_table(next_clicks[click_cols].head(5) if click_cols else next_clicks.head(5), [], height=280)

    if not current.empty:
        st.markdown("##### Current ticker states")
        view_cols = [c for c in ["ticker", "current_state", "operating_mode", "can_take_new_risk", "first_blocker", "next_click", "next_action", "stock_or_etf_route", "option_route", "option_side"] if c in current.columns]
        _show_status_table(current[view_cols].head(10 if compact else 35) if view_cols else current.head(35), [], height=430 if compact else 680)

    if compact:
        with st.expander("Open blocker queue and stage contracts", expanded=False):
            detail_tabs = st.tabs(["Blockers", "Next Clicks", "Stage Contracts"])
            with detail_tabs[0]:
                b_cols = [c for c in ["priority", "ticker", "blocker_type", "blocker", "what_to_do", "where_to_click", "why_it_matters"] if c in blockers.columns]
                _show_status_table(blockers[b_cols].head(80) if b_cols else blockers.head(80), [], height=580)
            with detail_tabs[1]:
                c_cols = [c for c in ["order", "page", "panel", "what_to_read", "why_now", "done_when", "do_not_do"] if c in next_clicks.columns]
                _show_status_table(next_clicks[c_cols] if c_cols else next_clicks, [], height=420)
            with detail_tabs[2]:
                s_cols = [c for c in ["stage", "input_contract", "output_contract", "pass_condition", "fail_condition", "dashboard_page"] if c in contracts.columns]
                _show_status_table(contracts[s_cols] if s_cols else contracts, [], height=620)
        return

    detail_tabs = st.tabs(["Current States", "Blocker Queue", "Next Clicks", "Stage Contracts"])
    with detail_tabs[0]:
        view_cols = [c for c in ["ticker", "current_state", "operating_mode", "can_take_new_risk", "first_blocker", "next_click", "next_action", "stock_or_etf_route", "option_route", "option_side", "why", "trigger_to_watch"] if c in current.columns]
        _show_status_table(current[view_cols] if view_cols else current, [], height=760)
    with detail_tabs[1]:
        b_cols = [c for c in ["priority", "ticker", "blocker_type", "blocker", "what_to_do", "where_to_click", "why_it_matters", "source_files"] if c in blockers.columns]
        _show_status_table(blockers[b_cols] if b_cols else blockers, [], height=760)
    with detail_tabs[2]:
        c_cols = [c for c in ["order", "page", "panel", "what_to_read", "why_now", "done_when", "do_not_do", "source_files"] if c in next_clicks.columns]
        _show_status_table(next_clicks[c_cols] if c_cols else next_clicks, [], height=520)
    with detail_tabs[3]:
        s_cols = [c for c in ["stage", "input_contract", "output_contract", "pass_condition", "fail_condition", "owner", "dashboard_page", "active_files"] if c in contracts.columns]
        _show_status_table(contracts[s_cols] if s_cols else contracts, [], height=760)


def _render_ticker_flow_cards(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_ticker_flow_cards_state.json")
    cards = safe_csv(ROOT / "quant_fund_ticker_flow_cards.csv")
    summary = safe_csv(ROOT / "quant_fund_state_machine_summary.csv")
    path_cards = safe_csv(ROOT / "quant_fund_user_path_cards.csv")
    qa = safe_csv(ROOT / "quant_fund_flow_card_quality_check.csv")

    if not state and cards.empty:
        st.info("Ticker Flow Cards have not run yet. Run Step210 or the daily system.")
        return

    st.markdown('<p class="section-title">Ticker Flow Cards</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Plain-English cards for each ticker: answer, why, next action, what not to do, and route.</p>',
        unsafe_allow_html=True,
    )

    card_count = int(_to_float(state.get("card_count"), len(cards) if not cards.empty else 0) or 0)
    top_ticker = _plain_status(state.get("top_ticker"), "No ticker")
    top_state = _plain_status(state.get("top_state"), "No state")
    qa_review = int(_to_float(state.get("quality_review_count"), 0) or 0)
    needs_proof = int(_to_float(state.get("needs_outside_proof_count"), 0) or 0)
    risk_blocked = int(_to_float(state.get("risk_blocked_count"), 0) or 0)

    accent = "#991b1b" if needs_proof or risk_blocked else "#111827"
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Ticker card answer</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 520))}</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.45; margin-top:9px;"><b>Start with:</b> {_esc(top_ticker)} / {_esc(top_state)}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    cols = st.columns(5)
    metrics = [
        ("Cards", str(card_count), "Readable ticker cards.", "#334155"),
        ("Start Ticker", top_ticker, "First card in the queue.", "#334155"),
        ("Need Proof", str(needs_proof), "Need source proof first.", "#991b1b" if needs_proof else "#334155"),
        ("Risk Blocked", str(risk_blocked), "Risk vetoes before ideas.", "#991b1b" if risk_blocked else "#334155"),
        ("QA Review", str(qa_review), "Card quality checks needing review.", "#991b1b" if qa_review else "#0f766e"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, metrics):
        with col:
            _simple_card(title, value, note, card_accent)

    if not cards.empty:
        top_cards = cards.head(8 if compact else 16)
        html = [
            '<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; margin:10px 0 18px 0;">'
        ]
        for _, row in top_cards.iterrows():
            ticker = _plain_status(row.get("ticker"), "")
            answer = _human_text(row.get("front_answer"), 170)
            state_text = _plain_status(row.get("state"), "")
            why = _human_text(row.get("why_now"), 240)
            do_now = _human_text(row.get("do_now"), 190)
            do_not = _human_text(row.get("do_not_do"), 170)
            where = _plain_status(row.get("where_to_click"), "Home")
            stock = _human_text(row.get("stock_or_etf_route"), 100)
            option = _human_text(row.get("option_route"), 115)
            border = "#991b1b" if state_text in {"Needs outside proof", "Risk blocked"} else "#334155"
            html.append(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {border}; border-radius:8px; padding:13px 14px; min-height:330px;">
                  <div style="display:flex; justify-content:space-between; gap:8px; align-items:flex-start;">
                    <div style="font-size:18px; color:#111827; font-weight:900;">{_esc(ticker)}</div>
                    <div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase; text-align:right;">{_esc(state_text)}</div>
                  </div>
                  <div style="font-size:15px; color:#111827; font-weight:850; line-height:1.3; margin-top:8px;">{_esc(answer)}</div>
                  <div style="font-size:12px; color:#4b5563; line-height:1.4; margin-top:8px;"><b>Why:</b> {_esc(why)}</div>
                  <div style="font-size:12px; color:#111827; line-height:1.4; margin-top:8px;"><b>Do now:</b> {_esc(do_now)}</div>
                  <div style="font-size:12px; color:#6b7280; line-height:1.35; margin-top:8px;"><b>Do not:</b> {_esc(do_not)}</div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:11px; color:#374151; line-height:1.35;">
                    <b>Click:</b> {_esc(where)}<br>
                    <b>Stock/ETF:</b> {_esc(stock)}<br>
                    <b>Options:</b> {_esc(option)}
                  </div>
                </div>
                """
            )
        html.append("</div>")
        _render_html("".join(html))

    if compact:
        with st.expander("Open full card tables, state machine, and QA", expanded=False):
            detail_tabs = st.tabs(["All Cards", "State Machine", "User Path", "QA"])
            with detail_tabs[0]:
                cols_view = [c for c in ["ticker", "front_answer", "state", "why_now", "do_now", "do_not_do", "where_to_click", "stock_or_etf_route", "option_route", "proof_to_collect"] if c in cards.columns]
                _show_status_table(cards[cols_view] if cols_view else cards, [], height=680)
            with detail_tabs[1]:
                s_cols = [c for c in ["state", "ticker_count", "plain_meaning", "allowed_now", "forbidden_now", "unlock_condition"] if c in summary.columns]
                _show_status_table(summary[s_cols] if s_cols else summary, [], height=520)
            with detail_tabs[2]:
                p_cols = [c for c in ["step", "title", "plain_instruction", "why_this_step", "done_when", "do_not_do", "page", "panel"] if c in path_cards.columns]
                _show_status_table(path_cards[p_cols] if p_cols else path_cards, [], height=480)
            with detail_tabs[3]:
                q_cols = [c for c in ["check", "status", "bad_rows", "what_it_checked", "fix_hint"] if c in qa.columns]
                _show_status_table(qa[q_cols] if q_cols else qa, ["status"], height=380)
        return

    detail_tabs = st.tabs(["All Cards", "State Machine", "User Path", "QA"])
    with detail_tabs[0]:
        cols_view = [c for c in ["ticker", "front_answer", "state", "why_now", "do_now", "do_not_do", "where_to_click", "stock_or_etf_route", "option_route", "proof_to_collect", "source_summary"] if c in cards.columns]
        _show_status_table(cards[cols_view] if cols_view else cards, [], height=760)
    with detail_tabs[1]:
        s_cols = [c for c in ["state", "ticker_count", "plain_meaning", "allowed_now", "forbidden_now", "unlock_condition"] if c in summary.columns]
        _show_status_table(summary[s_cols] if s_cols else summary, [], height=620)
    with detail_tabs[2]:
        p_cols = [c for c in ["step", "title", "plain_instruction", "why_this_step", "done_when", "do_not_do", "page", "panel"] if c in path_cards.columns]
        _show_status_table(path_cards[p_cols] if p_cols else path_cards, [], height=520)
    with detail_tabs[3]:
        q_cols = [c for c in ["check", "status", "bad_rows", "what_it_checked", "fix_hint"] if c in qa.columns]
        _show_status_table(qa[q_cols] if q_cols else qa, ["status"], height=420)


def _render_proof_collection_workbench(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_proof_workbench_state.json")
    tasks = safe_csv(ROOT / "quant_fund_proof_task_cards.csv")
    queue = safe_csv(ROOT / "quant_fund_proof_ticker_queue.csv")
    guide = safe_csv(ROOT / "quant_fund_proof_field_guide.csv")
    instructions = safe_csv(ROOT / "quant_fund_proof_user_instructions.csv")
    qa = safe_csv(ROOT / "quant_fund_proof_workbench_quality_check.csv")

    if not state and tasks.empty:
        st.info("Proof Collection Workbench has not run yet. Run Step211 or the daily system.")
        return

    st.markdown('<p class="section-title">Proof Collection Workbench</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">The proof-first workbench: exact question, acceptable source, fields to fill, and what to rerun after verification.</p>',
        unsafe_allow_html=True,
    )

    task_count = int(_to_float(state.get("proof_task_count"), len(tasks) if not tasks.empty else 0) or 0)
    ticker_count = int(_to_float(state.get("ticker_queue_count"), len(queue) if not queue.empty else 0) or 0)
    qa_review = int(_to_float(state.get("quality_review_count"), 0) or 0)
    first_ticker = _plain_status(state.get("first_ticker"), "No ticker")
    first_type = _plain_status(state.get("first_proof_type"), "No proof type")
    first_question = _human_text(state.get("first_question"), 320)

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #991b1b; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Proof desk answer</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 520))}</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.45; margin-top:9px;"><b>First proof:</b> {_esc(first_ticker)} / {_esc(first_type)} / {_esc(first_question)}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">This does not fetch sources and does not approve evidence. It tells you what a human must verify.</div>
        </div>
        """
    )

    cols = st.columns(5)
    metrics = [
        ("Proof Tasks", str(task_count), "Open proof questions.", "#991b1b" if task_count else "#334155"),
        ("Ticker Queue", str(ticker_count), "Tickers with proof work.", "#334155"),
        ("First Ticker", first_ticker, "Start here.", "#334155"),
        ("Proof Type", first_type, "First proof category.", "#334155"),
        ("QA Review", str(qa_review), "Workbench checks needing review.", "#991b1b" if qa_review else "#0f766e"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, metrics):
        with col:
            _simple_card(title, value, note, card_accent)

    if not tasks.empty:
        top_tasks = tasks.head(6 if compact else 12)
        html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:10px 0 18px 0;">']
        for _, row in top_tasks.iterrows():
            ticker = _plain_status(row.get("ticker"), "")
            proof_type = _plain_status(row.get("proof_type"), "")
            question = _human_text(row.get("question_to_answer"), 190)
            source = _human_text(row.get("acceptable_source"), 210)
            fields = _human_text(row.get("fields_to_fill"), 160)
            after = _human_text(row.get("after_you_fill"), 150)
            do_not = _human_text(row.get("do_not_do"), 140)
            html.append(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid #991b1b; border-radius:8px; padding:13px 14px; min-height:315px;">
                  <div style="display:flex; justify-content:space-between; gap:8px; align-items:flex-start;">
                    <div style="font-size:18px; color:#111827; font-weight:900;">{_esc(ticker)}</div>
                    <div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase; text-align:right;">{_esc(proof_type)}</div>
                  </div>
                  <div style="font-size:15px; color:#111827; font-weight:850; line-height:1.32; margin-top:8px;">{_esc(question)}</div>
                  <div style="font-size:12px; color:#4b5563; line-height:1.4; margin-top:8px;"><b>Acceptable source:</b> {_esc(source)}</div>
                  <div style="font-size:12px; color:#111827; line-height:1.4; margin-top:8px;"><b>Fill:</b> {_esc(fields)}</div>
                  <div style="font-size:12px; color:#374151; line-height:1.35; margin-top:8px;"><b>After:</b> {_esc(after)}</div>
                  <div style="font-size:11px; color:#6b7280; line-height:1.35; margin-top:8px;"><b>Do not:</b> {_esc(do_not)}</div>
                </div>
                """
            )
        html.append("</div>")
        _render_html("".join(html))

    if not instructions.empty:
        st.markdown("##### How to use this proof desk")
        i_cols = [c for c in ["step", "instruction", "why", "done_when", "do_not_do", "file_or_page"] if c in instructions.columns]
        _show_status_table(instructions[i_cols] if i_cols else instructions, [], height=310)

    if compact:
        with st.expander("Open proof task table, ticker queue, field guide, and QA", expanded=False):
            tabs = st.tabs(["Proof Tasks", "Ticker Queue", "Field Guide", "QA"])
            with tabs[0]:
                cols_view = [c for c in ["task_rank", "ticker", "proof_type", "question_to_answer", "acceptable_source", "fields_to_fill", "editable_file", "proof_id", "after_you_fill"] if c in tasks.columns]
                _show_status_table(tasks[cols_view] if cols_view else tasks, [], height=680)
            with tabs[1]:
                q_cols = [c for c in ["queue_rank", "ticker", "open_proof_tasks", "first_proof_type", "first_question", "first_source", "first_fields_to_fill", "after_done"] if c in queue.columns]
                _show_status_table(queue[q_cols] if q_cols else queue, [], height=520)
            with tabs[2]:
                g_cols = [c for c in ["proof_type", "plain_goal", "good_sources", "must_fill", "what_counts_as_done", "what_does_not_count", "next_step_after_verified"] if c in guide.columns]
                _show_status_table(guide[g_cols] if g_cols else guide, [], height=520)
            with tabs[3]:
                qa_cols = [c for c in ["check", "status", "bad_rows", "what_it_checked", "fix_hint"] if c in qa.columns]
                _show_status_table(qa[qa_cols] if qa_cols else qa, ["status"], height=360)
        return

    detail_tabs = st.tabs(["Proof Tasks", "Ticker Queue", "Field Guide", "Instructions", "QA"])
    with detail_tabs[0]:
        cols_view = [c for c in ["task_rank", "ticker", "proof_type", "question_to_answer", "why_this_matters", "acceptable_source", "fields_to_fill", "editable_file", "proof_id", "after_you_fill", "do_not_do"] if c in tasks.columns]
        _show_status_table(tasks[cols_view] if cols_view else tasks, [], height=760)
    with detail_tabs[1]:
        q_cols = [c for c in ["queue_rank", "ticker", "open_proof_tasks", "first_proof_type", "first_question", "first_source", "first_fields_to_fill", "why_this_ticker_first", "after_done"] if c in queue.columns]
        _show_status_table(queue[q_cols] if q_cols else queue, [], height=640)
    with detail_tabs[2]:
        g_cols = [c for c in ["proof_type", "plain_goal", "good_sources", "must_fill", "what_counts_as_done", "what_does_not_count", "next_step_after_verified"] if c in guide.columns]
        _show_status_table(guide[g_cols] if g_cols else guide, [], height=560)
    with detail_tabs[3]:
        i_cols = [c for c in ["step", "instruction", "why", "done_when", "do_not_do", "file_or_page"] if c in instructions.columns]
        _show_status_table(instructions[i_cols] if i_cols else instructions, [], height=520)
    with detail_tabs[4]:
        qa_cols = [c for c in ["check", "status", "bad_rows", "what_it_checked", "fix_hint"] if c in qa.columns]
        _show_status_table(qa[qa_cols] if qa_cols else qa, ["status"], height=420)


def _render_proof_quality_gate(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_proof_quality_gate_state.json")
    gate = safe_csv(ROOT / "quant_fund_proof_quality_gate.csv")
    missing = safe_csv(ROOT / "quant_fund_proof_missing_fields.csv")
    ready = safe_csv(ROOT / "quant_fund_proof_ready_review.csv")
    rules = safe_csv(ROOT / "quant_fund_source_quality_rules.csv")

    if not state and gate.empty:
        st.info("Proof Quality Gate has not run yet. Run Step212 or the daily system.")
        return

    st.markdown('<p class="section-title">Proof Quality Gate</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Checks whether a filled proof row is complete, source-backed, and strong enough for the acceptance bridge.</p>',
        unsafe_allow_html=True,
    )

    proof_rows = int(_to_float(state.get("proof_rows"), len(gate) if not gate.empty else 0) or 0)
    ready_rows = int(_to_float(state.get("ready_rows"), len(ready) if not ready.empty else 0) or 0)
    missing_rows = int(_to_float(state.get("missing_field_rows"), len(missing) if not missing.empty else 0) or 0)
    needs_fill = int(_to_float(state.get("needs_fill_rows"), 0) or 0)
    weak = int(_to_float(state.get("weak_source_rows"), 0) or 0)
    first_ticker = _plain_status(state.get("first_ticker"), "No ticker")
    first_state = _plain_status(state.get("first_state"), "No state")
    first_fix = _human_text(state.get("first_fix"), 320)

    accent = "#0f766e" if ready_rows else "#991b1b"
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Proof quality answer</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 520))}</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.45; margin-top:9px;"><b>First row:</b> {_esc(first_ticker)} / {_esc(first_state)} / {_esc(first_fix)}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">This gate does not approve evidence. It only tells whether proof is strong enough to send to the bridge.</div>
        </div>
        """
    )

    cols = st.columns(5)
    metrics = [
        ("Proof Rows", str(proof_rows), "Rows checked.", "#334155"),
        ("Ready", str(ready_rows), "Can go to bridge.", "#0f766e" if ready_rows else "#334155"),
        ("Missing Fields", str(missing_rows), "Fields still blank.", "#991b1b" if missing_rows else "#334155"),
        ("Needs Fill", str(needs_fill), "Proof rows not filled yet.", "#991b1b" if needs_fill else "#334155"),
        ("Weak Sources", str(weak), "Source quality is too weak.", "#991b1b" if weak else "#334155"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, metrics):
        with col:
            _simple_card(title, value, note, card_accent)

    if not gate.empty:
        st.markdown("##### Proof quality queue")
        g_cols = [c for c in ["ticker", "proof_type", "quality_state", "quality_score", "source_quality", "missing_fields", "what_to_fix", "can_send_to_acceptance_bridge"] if c in gate.columns]
        _show_status_table(gate[g_cols].head(10 if compact else 40) if g_cols else gate.head(40), ["quality_state"], height=420 if compact else 680)

    if compact:
        with st.expander("Open missing fields, ready rows, and source quality rules", expanded=False):
            tabs = st.tabs(["Missing Fields", "Ready Rows", "Source Rules", "All Gate Rows"])
            with tabs[0]:
                m_cols = [c for c in ["ticker", "proof_id", "proof_type", "missing_field", "why_needed", "how_to_fill", "editable_file"] if c in missing.columns]
                _show_status_table(missing[m_cols].head(100) if m_cols else missing.head(100), [], height=580)
            with tabs[1]:
                r_cols = [c for c in ["ticker", "proof_id", "proof_type", "source_name", "observed_value", "reviewer", "review_date", "next_step"] if c in ready.columns]
                _show_status_table(ready[r_cols] if r_cols else ready, [], height=420)
            with tabs[2]:
                s_cols = [c for c in ["rule", "source_examples", "score_band", "counts_as", "does_not_count_as"] if c in rules.columns]
                _show_status_table(rules[s_cols] if s_cols else rules, [], height=420)
            with tabs[3]:
                all_cols = [c for c in ["ticker", "proof_type", "quality_state", "quality_score", "source_quality", "missing_fields", "required_fields", "what_to_fix", "why"] if c in gate.columns]
                _show_status_table(gate[all_cols] if all_cols else gate, ["quality_state"], height=680)
        return

    detail_tabs = st.tabs(["Quality Gate", "Missing Fields", "Ready Rows", "Source Rules"])
    with detail_tabs[0]:
        g_cols = [c for c in ["ticker", "proof_type", "quality_state", "quality_score", "source_quality", "missing_fields", "required_fields", "source_name", "observed_value", "what_to_fix", "why"] if c in gate.columns]
        _show_status_table(gate[g_cols] if g_cols else gate, ["quality_state"], height=760)
    with detail_tabs[1]:
        m_cols = [c for c in ["ticker", "proof_id", "proof_type", "missing_field", "why_needed", "how_to_fill", "editable_file"] if c in missing.columns]
        _show_status_table(missing[m_cols] if m_cols else missing, [], height=760)
    with detail_tabs[2]:
        r_cols = [c for c in ["ticker", "proof_id", "proof_type", "source_name", "observed_value", "reviewer", "review_date", "next_step"] if c in ready.columns]
        _show_status_table(ready[r_cols] if r_cols else ready, [], height=520)
    with detail_tabs[3]:
        s_cols = [c for c in ["rule", "source_examples", "score_band", "counts_as", "does_not_count_as"] if c in rules.columns]
        _show_status_table(rules[s_cols] if s_cols else rules, [], height=520)


def _render_proof_fill_desk(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_proof_fill_desk_state.json")
    cards = safe_csv(ROOT / "quant_fund_proof_fill_cards.csv")
    ticker_plan = safe_csv(ROOT / "quant_fund_proof_fill_ticker_plan.csv")
    recipes = safe_csv(ROOT / "quant_fund_proof_fill_field_recipes.csv")
    copy_sheet = safe_csv(ROOT / "quant_fund_proof_fill_copy_sheet.csv")
    qa = safe_csv(ROOT / "quant_fund_proof_fill_quality_check.csv")

    if not state and cards.empty:
        st.info("Proof Fill Desk has not run yet. Run Step213 or the daily system.")
        return

    st.markdown('<p class="section-title">Proof Fill Desk</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Turns missing proof fields into a clear worklist: who first, what to open, what to type, and what to rerun.</p>',
        unsafe_allow_html=True,
    )

    fill_cards = int(_to_float(state.get("fill_card_count"), len(cards) if not cards.empty else 0) or 0)
    ticker_count = int(_to_float(state.get("ticker_count"), len(ticker_plan) if not ticker_plan.empty else 0) or 0)
    field_count = int(_to_float(state.get("field_to_fill_count"), len(copy_sheet) if not copy_sheet.empty else 0) or 0)
    qa_review = int(_to_float(state.get("qa_review_count"), 0) or 0)
    first_ticker = _plain_status(state.get("first_ticker"), "No ticker")
    first_type = _plain_status(state.get("first_proof_type"), "No proof type")
    first_fields = _human_text(state.get("first_fields_to_fill"), 260)
    first_source = _human_text(state.get("first_source_to_open"), 360)

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #111827; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Proof fill answer</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 520))}</div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;"><b>Start here:</b> {_esc(first_ticker)} / {_esc(first_type)} / fill {_esc(first_fields)}.</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;"><b>Open this source:</b> {_esc(first_source)}</div>
        </div>
        """
    )

    cols = st.columns(5)
    metrics = [
        ("Fill Cards", str(fill_cards), "Proof rows grouped into tasks.", "#111827"),
        ("Tickers", str(ticker_count), "Names with proof work.", "#334155"),
        ("Fields To Fill", str(field_count), "Individual blank fields.", "#991b1b" if field_count else "#0f766e"),
        ("First Ticker", first_ticker, "Start here.", "#334155"),
        ("QA Review", str(qa_review), "Desk checks needing review.", "#991b1b" if qa_review else "#0f766e"),
    ]
    for col, (title, value, note, accent) in zip(cols, metrics):
        with col:
            _simple_card(title, value, note, accent)

    if not cards.empty:
        st.markdown("##### Fill these proof cards first")
        top_cards = cards.head(8 if compact else 12)
        html = ['<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; margin:10px 0 18px 0;">']
        for _, row in top_cards.iterrows():
            ticker = _plain_status(row.get("ticker"), "")
            proof_type = _plain_status(row.get("proof_type"), "")
            task = _human_text(row.get("plain_task"), 150)
            question = _human_text(row.get("question_to_answer"), 170)
            source = _human_text(row.get("source_to_open"), 180)
            fields = _human_text(row.get("fields_to_fill_now"), 150)
            good = _human_text(row.get("good_example"), 130)
            after = _human_text(row.get("after_filling"), 150)
            score = _plain_status(row.get("priority_score"), "")
            html.append(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid #111827; border-radius:8px; padding:13px 14px; min-height:360px;">
                  <div style="display:flex; justify-content:space-between; gap:8px; align-items:flex-start;">
                    <div style="font-size:18px; color:#111827; font-weight:900;">{_esc(ticker)}</div>
                    <div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase; text-align:right;">{_esc(proof_type)} / {_esc(score)}</div>
                  </div>
                  <div style="font-size:14px; color:#111827; font-weight:850; line-height:1.3; margin-top:8px;">{_esc(task)}</div>
                  <div style="font-size:12px; color:#374151; line-height:1.38; margin-top:8px;"><b>Question:</b> {_esc(question)}</div>
                  <div style="font-size:12px; color:#4b5563; line-height:1.38; margin-top:8px;"><b>Open:</b> {_esc(source)}</div>
                  <div style="font-size:12px; color:#111827; line-height:1.38; margin-top:8px;"><b>Fill:</b> {_esc(fields)}</div>
                  <div style="font-size:12px; color:#374151; line-height:1.35; margin-top:8px;"><b>Good value:</b> {_esc(good)}</div>
                  <div style="font-size:11px; color:#6b7280; line-height:1.35; margin-top:8px;"><b>After:</b> {_esc(after)}</div>
                </div>
                """
            )
        html.append("</div>")
        _render_html("".join(html))

    if not ticker_plan.empty:
        st.markdown("##### Ticker fill plan")
        t_cols = [c for c in ["ticker_rank", "ticker", "open_proof_rows", "missing_field_rows", "first_proof_type", "first_question", "first_source_to_open", "first_fields_to_fill", "estimated_minutes", "after_done"] if c in ticker_plan.columns]
        _show_status_table(ticker_plan[t_cols].head(10 if compact else 30) if t_cols else ticker_plan.head(30), [], height=420 if compact else 680)

    if compact:
        with st.expander("Open fill cards, field recipes, copy sheet, and QA", expanded=False):
            tabs = st.tabs(["Fill Cards", "Ticker Plan", "Field Recipes", "Copy Sheet", "QA"])
            with tabs[0]:
                c_cols = [c for c in ["card_rank", "ticker", "proof_type", "plain_task", "source_to_open", "fields_to_fill_now", "good_example", "after_filling", "proof_id"] if c in cards.columns]
                _show_status_table(cards[c_cols] if c_cols else cards, [], height=760)
            with tabs[1]:
                t_cols = [c for c in ["ticker_rank", "ticker", "open_proof_rows", "missing_field_rows", "first_proof_type", "first_question", "first_source_to_open", "first_fields_to_fill", "estimated_minutes"] if c in ticker_plan.columns]
                _show_status_table(ticker_plan[t_cols] if t_cols else ticker_plan, [], height=620)
            with tabs[2]:
                r_cols = [c for c in ["proof_type", "plain_label", "what_it_means", "where_to_find_it", "what_to_type", "good_example", "bad_example"] if c in recipes.columns]
                _show_status_table(recipes[r_cols] if r_cols else recipes, [], height=620)
            with tabs[3]:
                s_cols = [c for c in ["ticker", "proof_type", "plain_label", "what_to_type", "where_to_find_it", "editable_file", "proof_id"] if c in copy_sheet.columns]
                _show_status_table(copy_sheet[s_cols].head(180) if s_cols else copy_sheet.head(180), [], height=720)
            with tabs[4]:
                q_cols = [c for c in ["check", "status", "bad_rows", "what_it_checked", "fix_hint"] if c in qa.columns]
                _show_status_table(qa[q_cols] if q_cols else qa, ["status"], height=420)
        return

    detail_tabs = st.tabs(["Fill Cards", "Ticker Plan", "Field Recipes", "Copy Sheet", "QA"])
    with detail_tabs[0]:
        c_cols = [c for c in ["card_rank", "ticker", "proof_type", "priority_score", "plain_task", "question_to_answer", "source_to_open", "fields_to_fill_now", "good_example", "bad_example", "why_this_blocks_progress", "proof_id", "after_filling"] if c in cards.columns]
        _show_status_table(cards[c_cols] if c_cols else cards, [], height=820)
    with detail_tabs[1]:
        t_cols = [c for c in ["ticker_rank", "ticker", "open_proof_rows", "missing_field_rows", "first_proof_type", "first_question", "first_source_to_open", "first_fields_to_fill", "why_this_ticker_first", "estimated_minutes", "after_done"] if c in ticker_plan.columns]
        _show_status_table(ticker_plan[t_cols] if t_cols else ticker_plan, [], height=760)
    with detail_tabs[2]:
        r_cols = [c for c in ["proof_type", "field_to_fill", "plain_label", "what_it_means", "where_to_find_it", "what_to_type", "good_example", "bad_example"] if c in recipes.columns]
        _show_status_table(recipes[r_cols] if r_cols else recipes, [], height=620)
    with detail_tabs[3]:
        s_cols = [c for c in ["ticker", "proof_type", "proof_id", "field_to_fill", "plain_label", "what_to_type", "where_to_find_it", "editable_file", "source_to_open", "question_to_answer"] if c in copy_sheet.columns]
        _show_status_table(copy_sheet[s_cols].head(260) if s_cols else copy_sheet.head(260), [], height=820)
    with detail_tabs[4]:
        q_cols = [c for c in ["check", "status", "bad_rows", "what_it_checked", "fix_hint"] if c in qa.columns]
        _show_status_table(qa[q_cols] if q_cols else qa, ["status"], height=420)


def _render_proof_intake_safe_apply(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_proof_intake_state.json")
    user_entry = safe_csv(ROOT / "quant_fund_proof_intake_user_entry.csv")
    preview = safe_csv(ROOT / "quant_fund_proof_intake_apply_preview.csv")
    applied = safe_csv(ROOT / "quant_fund_proof_intake_applied_rows.csv")
    rejected = safe_csv(ROOT / "quant_fund_proof_intake_rejected_rows.csv")
    audit = safe_csv(ROOT / "quant_fund_proof_intake_audit.csv")

    if not state and user_entry.empty:
        st.info("Proof Intake Safe Apply has not run yet. Run Step214 or the daily system.")
        return

    st.markdown('<p class="section-title">Proof Intake Safe Apply</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">A safe bridge from human proof entry back to the proof input file. It only writes rows marked APPLY.</p>',
        unsafe_allow_html=True,
    )

    user_rows = int(_to_float(state.get("user_entry_rows"), len(user_entry) if not user_entry.empty else 0) or 0)
    apply_requests = int(_to_float(state.get("apply_request_count"), 0) or 0)
    applied_count = int(_to_float(state.get("applied_count"), len(applied) if not applied.empty else 0) or 0)
    rejected_count = int(_to_float(state.get("rejected_count"), len(rejected) if not rejected.empty else 0) or 0)
    waiting_count = int(_to_float(state.get("waiting_count"), 0) or 0)
    first_ticker = _plain_status(state.get("first_ticker"), "No ticker")
    first_task = _human_text(state.get("first_task"), 360)
    user_file = _plain_status(state.get("user_entry_file"), "quant_fund_proof_intake_user_entry.csv")
    backup = _plain_status(state.get("backup_file"), "No backup because nothing was applied")
    accent = "#0f766e" if applied_count else "#111827"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Proof intake answer</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 560))}</div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;"><b>File to fill:</b> {_esc(user_file)}. <b>First row:</b> {_esc(first_ticker)} / {_esc(first_task)}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;"><b>Safety rule:</b> WAIT rows do nothing. Only APPLY rows that pass validation are written back. <b>Backup:</b> {_esc(backup)}</div>
        </div>
        """
    )

    cols = st.columns(5)
    metrics = [
        ("User Entry Rows", str(user_rows), "Rows available to fill.", "#334155"),
        ("Apply Requests", str(apply_requests), "Rows marked APPLY.", "#111827" if apply_requests else "#334155"),
        ("Applied", str(applied_count), "Rows written back.", "#0f766e" if applied_count else "#334155"),
        ("Rejected", str(rejected_count), "APPLY rows missing fields.", "#991b1b" if rejected_count else "#334155"),
        ("Waiting", str(waiting_count), "Rows left untouched.", "#334155"),
    ]
    for col, (title, value, note, accent_color) in zip(cols, metrics):
        with col:
            _simple_card(title, value, note, accent_color)

    st.markdown("##### How to use this")
    how_rows = pd.DataFrame([
        {"step": 1, "what_to_do": "Open the user entry CSV.", "detail": "Use quant_fund_proof_intake_user_entry.csv. It is the clean sheet, not the raw proof input."},
        {"step": 2, "what_to_do": "Fill the human proof fields.", "detail": "Source name, source URL or file, observed value, reviewer, and review date. News proof also needs source time plus price and volume checks."},
        {"step": 3, "what_to_do": "Set Apply Decision only when ready.", "detail": "Leave WAIT for unfinished rows. Type APPLY only for rows you want written back."},
        {"step": 4, "what_to_do": "Rerun the proof chain.", "detail": "After APPLY, rerun Steps 206, 207, 204, 212, 213, and 214."},
    ])
    _show_status_table(how_rows, [], height=220)

    if not preview.empty:
        st.markdown("##### Apply preview")
        p_cols = [c for c in ["ticker", "proof_type", "apply_decision", "validation_state", "will_apply", "missing_or_problem", "updated_fields", "next_step"] if c in preview.columns]
        _show_status_table(preview[p_cols].head(12 if compact else 60) if p_cols else preview.head(60), ["validation_state", "will_apply"], height=420 if compact else 720)

    if compact:
        with st.expander("Open user entry sheet, applied rows, rejected rows, and audit", expanded=False):
            tabs = st.tabs(["User Entry", "Applied", "Rejected", "Audit"])
            with tabs[0]:
                u_cols = [c for c in ["entry_rank", "ticker", "proof_type", "plain_task", "source_to_open", "apply_decision", "proof_status", "source_name", "source_url_or_file", "observed_value", "reviewer", "review_date", "proof_id"] if c in user_entry.columns]
                _show_status_table(user_entry[u_cols].head(120) if u_cols else user_entry.head(120), ["apply_decision"], height=760)
            with tabs[1]:
                _show_status_table(applied, ["validation_state"], height=420)
            with tabs[2]:
                _show_status_table(rejected, ["validation_state"], height=420)
            with tabs[3]:
                a_cols = [c for c in ["timestamp", "proof_id", "ticker", "audit_event", "detail", "backup_file"] if c in audit.columns]
                _show_status_table(audit[a_cols] if a_cols else audit, [], height=420)
        return

    detail_tabs = st.tabs(["User Entry", "Apply Preview", "Applied", "Rejected", "Audit"])
    with detail_tabs[0]:
        u_cols = [c for c in ["entry_rank", "ticker", "proof_type", "plain_task", "question_to_answer", "source_to_open", "apply_decision", "proof_status", "source_name", "source_url_or_file", "observed_value", "observed_time", "price_reaction_checked", "volume_reaction_checked", "reviewer", "review_date", "proof_note", "proof_id"] if c in user_entry.columns]
        _show_status_table(user_entry[u_cols] if u_cols else user_entry, ["apply_decision"], height=820)
    with detail_tabs[1]:
        p_cols = [c for c in ["proof_id", "ticker", "proof_type", "apply_decision", "validation_state", "will_apply", "missing_or_problem", "updated_fields", "next_step"] if c in preview.columns]
        _show_status_table(preview[p_cols] if p_cols else preview, ["validation_state", "will_apply"], height=760)
    with detail_tabs[2]:
        _show_status_table(applied, ["validation_state"], height=520)
    with detail_tabs[3]:
        _show_status_table(rejected, ["validation_state"], height=520)
    with detail_tabs[4]:
        a_cols = [c for c in ["timestamp", "proof_id", "ticker", "audit_event", "detail", "backup_file"] if c in audit.columns]
        _show_status_table(audit[a_cols] if a_cols else audit, [], height=520)


def _render_proof_closure_tracker(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_proof_closure_state.json")
    tickers = safe_csv(ROOT / "quant_fund_proof_closure_ticker_status.csv")
    counts = safe_csv(ROOT / "quant_fund_proof_closure_stage_counts.csv")
    actions = safe_csv(ROOT / "quant_fund_proof_closure_next_actions.csv")
    unblock = safe_csv(ROOT / "quant_fund_proof_closure_unblock_candidates.csv")

    if not state and tickers.empty:
        st.info("Proof Closure Tracker has not run yet. Run Step215 or the daily system.")
        return

    st.markdown('<p class="section-title">Proof Closure Tracker</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Tracks whether proof has actually closed the loop: intake, quality gate, bridge, and Step204 acceptance.</p>',
        unsafe_allow_html=True,
    )

    ticker_count = int(_to_float(state.get("ticker_count"), len(tickers) if not tickers.empty else 0) or 0)
    fill_first = int(_to_float(state.get("fill_first_count"), 0) or 0)
    patch_ready = int(_to_float(state.get("patch_ready_count"), 0) or 0)
    bridge_conflict = int(_to_float(state.get("bridge_conflict_count"), 0) or 0)
    accepted = int(_to_float(state.get("accepted_evidence_ticker_count"), 0) or 0)
    top_ticker = _plain_status(state.get("top_action_ticker"), "No ticker")
    top_action = _human_text(state.get("top_action"), 360)
    accent = "#0f766e" if patch_ready or accepted else "#111827"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Proof closure answer</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 560))}</div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;"><b>Top action:</b> {_esc(top_ticker)} / {_esc(top_action)}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">This tracker does not approve trades. It only tells whether proof work has reached the next gate.</div>
        </div>
        """
    )

    cols = st.columns(5)
    metrics = [
        ("Tickers", str(ticker_count), "Names tracked.", "#334155"),
        ("Need Proof First", str(fill_first), "Still need source proof.", "#991b1b" if fill_first else "#334155"),
        ("Patch Ready", str(patch_ready), "Manual Step204 patch ready.", "#0f766e" if patch_ready else "#334155"),
        ("Bridge Conflicts", str(bridge_conflict), "Need bridge review.", "#991b1b" if bridge_conflict else "#334155"),
        ("Accepted", str(accepted), "Tickers with accepted evidence.", "#0f766e" if accepted else "#334155"),
    ]
    for col, (title, value, note, accent_color) in zip(cols, metrics):
        with col:
            _simple_card(title, value, note, accent_color)

    if not counts.empty:
        st.markdown("##### Proof pipeline counts")
        c_cols = [c for c in ["stage_order", "stage_name", "row_count", "plain_meaning", "next_if_zero"] if c in counts.columns]
        _show_status_table(counts[c_cols] if c_cols else counts, [], height=340)

    if not actions.empty:
        st.markdown("##### Next proof actions")
        a_cols = [c for c in ["action_rank", "ticker", "action", "why", "page_or_file", "done_when", "do_not_do"] if c in actions.columns]
        _show_status_table(actions[a_cols].head(8 if compact else 20) if a_cols else actions.head(20), [], height=420 if compact else 680)

    if compact:
        with st.expander("Open ticker closure status and unblock candidates", expanded=False):
            tabs = st.tabs(["Ticker Status", "Unblock Candidates", "Stage Counts"])
            with tabs[0]:
                t_cols = [c for c in ["ticker", "closure_state", "plain_status", "next_action", "where_to_go", "proof_rows", "missing_proof_rows", "quality_ready_rows", "bridge_patch_rows", "step204_accepted_rows", "first_question"] if c in tickers.columns]
                _show_status_table(tickers[t_cols].head(120) if t_cols else tickers.head(120), ["closure_state"], height=760)
            with tabs[1]:
                u_cols = [c for c in ["ticker", "unblock_state", "what_would_unlock", "remaining_blocker", "proof_progress_score"] if c in unblock.columns]
                _show_status_table(unblock[u_cols].head(120) if u_cols else unblock.head(120), ["unblock_state"], height=620)
            with tabs[2]:
                c_cols = [c for c in ["stage_order", "stage_name", "row_count", "plain_meaning", "next_if_zero"] if c in counts.columns]
                _show_status_table(counts[c_cols] if c_cols else counts, [], height=420)
        return

    detail_tabs = st.tabs(["Ticker Status", "Next Actions", "Stage Counts", "Unblock Candidates"])
    with detail_tabs[0]:
        t_cols = [c for c in ["ticker", "closure_state", "plain_status", "next_action", "where_to_go", "proof_rows", "missing_proof_rows", "quality_ready_rows", "intake_apply_requests", "intake_applied_rows", "verified_source_rows", "bridge_patch_rows", "bridge_conflicts", "step204_accepted_rows", "first_question", "first_source"] if c in tickers.columns]
        _show_status_table(tickers[t_cols] if t_cols else tickers, ["closure_state"], height=820)
    with detail_tabs[1]:
        a_cols = [c for c in ["action_rank", "ticker", "action", "why", "page_or_file", "done_when", "do_not_do"] if c in actions.columns]
        _show_status_table(actions[a_cols] if a_cols else actions, [], height=680)
    with detail_tabs[2]:
        c_cols = [c for c in ["stage_order", "stage_name", "row_count", "plain_meaning", "next_if_zero"] if c in counts.columns]
        _show_status_table(counts[c_cols] if c_cols else counts, [], height=520)
    with detail_tabs[3]:
        u_cols = [c for c in ["ticker", "unblock_state", "what_would_unlock", "remaining_blocker", "proof_progress_score"] if c in unblock.columns]
        _show_status_table(unblock[u_cols] if u_cols else unblock, ["unblock_state"], height=680)


def _render_quant_fund_operating_flow(compact: bool = False):
    state = safe_json(ROOT / "quant_fund_operating_flow_state.json")
    stages = safe_csv(ROOT / "quant_fund_operating_flow_stages.csv")
    edges = safe_csv(ROOT / "quant_fund_operating_flow_edges.csv")
    runbook = safe_csv(ROOT / "quant_fund_operating_flow_daily_runbook.csv")
    software_map = safe_csv(ROOT / "quant_fund_operating_flow_software_map.csv")

    if not state and stages.empty:
        st.info("Quant Fund Operating Flow has not run yet. Run Step208 or the daily system.")
        return

    st.markdown('<p class="section-title">Quant Fund Operating Flow</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">The full research software path: what enters, what gets blocked, what reaches the PM gate, and where to click next.</p>',
        unsafe_allow_html=True,
    )

    stage_count = int(_to_float(state.get("stage_count"), len(stages) if not stages.empty else 0) or 0)
    hard_gates = int(_to_float(state.get("hard_gate_count"), 0) or 0)
    runbook_steps = int(_to_float(state.get("daily_runbook_steps"), len(runbook) if not runbook.empty else 0) or 0)
    software_panels = int(_to_float(state.get("software_panels"), len(software_map) if not software_map.empty else 0) or 0)
    avg_maturity = _plain_status(state.get("average_maturity_now_pct"), "No build level")
    bottleneck = _plain_status(state.get("current_bottleneck"), "Start with data, risk, and proof gates.")

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #111827; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Operating answer</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 520))}</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.45; margin-top:9px;"><b>Current bottleneck:</b> {_esc(_human_text(bottleneck, 320))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Flow Stages", str(stage_count), "Data to learning and governance.", "#111827"),
        ("Hard Gates", str(hard_gates), "Stops that can block an idea.", "#991b1b" if hard_gates else "#334155"),
        ("Runbook Steps", str(runbook_steps), "Daily order of operations.", "#334155"),
        ("Software Panels", str(software_panels), "Where the flow appears in the app.", "#334155"),
        ("Build Level", f"{avg_maturity}%", "Average current maturity across stages.", "#0f766e"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if compact:
        if not stages.empty:
            st.markdown("##### First eight stages")
            st.markdown("Read left to right: each stage must produce clean evidence before the next stage uses it.")
            s_cols = [c for c in ["stage_order", "stage_name", "plain_goal", "hard_gate", "if_gate_fails", "app_page"] if c in stages.columns]
            _show_status_table(stages[s_cols].head(8) if s_cols else stages.head(8), [], height=450)
        with st.expander("Open the full operating flow, daily runbook, and software map", expanded=False):
            detail_tabs = st.tabs(["All Stages", "Daily Runbook", "Software Map", "Gate Handoffs"])
            with detail_tabs[0]:
                s_cols = [c for c in ["stage_order", "stage_name", "desk_name", "plain_goal", "hard_gate", "if_gate_fails", "app_page", "maturity_now_pct", "gap_to_top_fund"] if c in stages.columns]
                _show_status_table(stages[s_cols] if s_cols else stages, [], height=680)
            with detail_tabs[1]:
                r_cols = [c for c in ["run_order", "when_to_use", "human_action", "system_action", "output_to_read", "do_not_do", "page_to_open"] if c in runbook.columns]
                _show_status_table(runbook[r_cols] if r_cols else runbook, [], height=560)
            with detail_tabs[2]:
                m_cols = [c for c in ["app_tab", "panel", "what_user_should_look_for", "related_steps", "maturity_now_pct", "next_upgrade"] if c in software_map.columns]
                _show_status_table(software_map[m_cols] if m_cols else software_map, [], height=560)
            with detail_tabs[3]:
                e_cols = [c for c in ["from_stage", "to_stage", "handoff_payload", "gate_condition", "blocking_failure", "why_it_matters"] if c in edges.columns]
                _show_status_table(edges[e_cols] if e_cols else edges, [], height=640)
        return

    detail_tabs = st.tabs(["Flow Stages", "Gate Handoffs", "Daily Runbook", "Software Map"])
    with detail_tabs[0]:
        st.markdown("##### Flow Stages")
        st.markdown("This is the institutional operating path. An idea should move forward only when the prior stage has clean evidence.")
        s_cols = [c for c in ["stage_order", "stage_name", "desk_name", "plain_goal", "primary_inputs", "primary_outputs", "hard_gate", "if_gate_fails", "app_page", "maturity_now_pct", "gap_to_top_fund"] if c in stages.columns]
        _show_status_table(stages[s_cols] if s_cols else stages, [], height=760)

    with detail_tabs[1]:
        st.markdown("##### Gate Handoffs")
        st.markdown("This shows what each desk hands to the next desk, and what stops the handoff.")
        e_cols = [c for c in ["from_stage", "to_stage", "handoff_payload", "gate_condition", "blocking_failure", "why_it_matters"] if c in edges.columns]
        _show_status_table(edges[e_cols] if e_cols else edges, [], height=720)

    with detail_tabs[2]:
        st.markdown("##### Daily Runbook")
        st.markdown("Use this as the daily click order. It is intentionally simple: check freshness, risk, news, ideas, proof, then paper monitor.")
        r_cols = [c for c in ["run_order", "when_to_use", "human_action", "system_action", "output_to_read", "do_not_do", "page_to_open"] if c in runbook.columns]
        _show_status_table(runbook[r_cols] if r_cols else runbook, [], height=650)

    with detail_tabs[3]:
        st.markdown("##### Software Map")
        st.markdown("This maps the operating flow to the website tabs, so you know where each part lives.")
        m_cols = [c for c in ["app_tab", "panel", "what_user_should_look_for", "input_files", "output_files", "related_steps", "maturity_now_pct", "next_upgrade"] if c in software_map.columns]
        _show_status_table(software_map[m_cols] if m_cols else software_map, [], height=650)


def _render_pm_review_final_gate_bridge(compact: bool = False):
    state = safe_json(ROOT / "pm_review_final_gate_bridge_state.json")
    bridge = safe_csv(ROOT / "pm_review_final_gate_bridge.csv")
    veto = safe_csv(ROOT / "pm_review_final_gate_veto_matrix.csv")
    next_actions = safe_csv(ROOT / "pm_review_final_gate_next_actions.csv")

    if not state and bridge.empty:
        st.info("PM Review Final Gate Bridge has not run yet. Run Step202 or the daily system.")
        return

    st.markdown('<p class="section-title">Final Gate Bridge</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This checks whether a completed PM review can actually move toward the final gate. PM review can clear one blocker, but risk, news, execution, liquidity, and options each keep veto power.</p>',
        unsafe_allow_html=True,
    )

    checked = int(_to_float(state.get("ticker_count"), 0) or 0)
    blocked = int(_to_float(state.get("blocked_before_final_gate_count"), 0) or 0)
    tiny_stock = int(_to_float(state.get("tiny_stock_etf_candidate_count"), 0) or 0)
    tiny_paper = int(_to_float(state.get("tiny_paper_candidate_count"), 0) or 0)
    options = int(_to_float(state.get("options_allowed_count"), 0) or 0)
    veto_rows = int(_to_float(state.get("veto_rows"), 0) or 0)
    accent = "#166534" if tiny_stock or tiny_paper else "#991b1b"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Bridge answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(_human_text(state.get("plain_answer"), 380))}</div>
          <div style="font-size:12px; color:#6b7280; line-height:1.4; margin-top:7px;">No broker connection. No live orders. Options must remain blocked unless the separate option gate passes.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Tickers Checked", str(checked), "PM review rows checked against final gates.", "#334155"),
        ("Blocked First", str(blocked), "Something still blocks before final gate.", "#991b1b" if blocked else "#334155"),
        ("Tiny Stock/ETF Candidates", str(tiny_stock), "Non-option tiny review candidates.", "#166534" if tiny_stock else "#334155"),
        ("Tiny Paper Candidates", str(tiny_paper), "All bridge checks pass.", "#166534" if tiny_paper else "#334155"),
        ("Options Allowed", str(options), f"{veto_rows} veto rows checked.", "#991b1b" if options == 0 else "#166534"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, card_accent)

    if not bridge.empty:
        st.markdown("##### Bridge decision")
        st.markdown("Read this as the path from PM review into the final gate. If the first blocking gate is not fixed, do not move to size, calls, or puts.")
        b_cols = [
            c for c in [
                "ticker",
                "bridge_decision",
                "first_blocking_gate",
                "next_step",
                "max_tiny_paper_review_pct",
                "non_option_gates_passed",
                "options_allowed",
                "current_final_permission",
                "pm_review_state",
                "proof_score_0_100",
            ]
            if c in bridge.columns
        ]
        _show_status_table(bridge[b_cols].head(16 if compact else 75), ["bridge_decision", "first_blocking_gate"], height=430 if compact else 720)

    if compact:
        with st.expander("Open veto details", expanded=False):
            v_cols = [c for c in ["ticker", "gate", "gate_state", "reason", "source_files"] if c in veto.columns]
            _show_status_table(veto[v_cols].head(80) if v_cols else veto.head(80), ["gate_state"], height=560)
        return

    detail_tabs = st.tabs(["Bridge Table", "Veto Matrix", "Next Actions"])
    with detail_tabs[0]:
        st.markdown("##### Bridge Table")
        b_cols = [c for c in ["ticker", "bridge_decision", "next_step", "max_tiny_paper_review_pct", "non_option_gates_passed", "options_allowed", "first_blocking_gate", "blocking_gates", "current_final_permission", "current_first_blocker", "pm_review_state", "pm_review_status", "proof_score_0_100", "source_files"] if c in bridge.columns]
        _show_status_table(bridge[b_cols] if b_cols else bridge, ["bridge_decision", "first_blocking_gate"], height=720)

    with detail_tabs[1]:
        st.markdown("##### Veto Matrix")
        st.markdown("Each ticker is checked through PM Review, Risk, News/Event, Execution/Liquidity, and Options.")
        v_cols = [c for c in ["ticker", "gate", "gate_state", "reason", "source_files"] if c in veto.columns]
        _show_status_table(veto[v_cols] if v_cols else veto, ["gate_state", "gate"], height=720)

    with detail_tabs[2]:
        st.markdown("##### Next Actions")
        st.markdown("This is the shortest route to unblock the bridge, not a trade list.")
        n_cols = [c for c in ["priority", "ticker", "bridge_decision", "what_to_do_next", "where_to_click", "why_it_matters", "source_files"] if c in next_actions.columns]
        _show_status_table(next_actions[n_cols] if n_cols else next_actions, ["priority", "bridge_decision"], height=720)


def _perf_plain(value, max_len: int | None = 220) -> str:
    text = _human_text(value, max_len=None)
    replacements = {
        "PROTOTYPE ONLY": "prototype evidence only",
        "Prototype Only": "prototype evidence only",
        "REPAIR FIRST": "repair first",
        "Repair First": "repair first",
        "EARLY STAGE": "early stage",
        "Early Stage": "early stage",
        "REPAIR REQUIRED": "needs repair",
        "Repair Required": "needs repair",
        "REVIEW REQUIRED": "needs review",
        "Review Required": "needs review",
        "NEEDS DEEPER TEST": "needs deeper testing",
        "Needs Deeper Test": "needs deeper testing",
        "NOT A SOLO PATH": "return alone cannot solve this",
        "Not A Solo Path": "return alone cannot solve this",
        "REQUIRES RISK AND SIGNAL REPAIR": "needs lower risk and better signal proof",
        "Requires Risk And Signal Repair": "needs lower risk and better signal proof",
        "WAIT FOR BASE REPAIR": "wait until the base is repaired",
        "Wait For Base Repair": "wait until the base is repaired",
        "EXECUTION REVIEW REQUIRED": "trading cost needs review",
        "Execution Review Required": "trading cost needs review",
        "REDUCE ONLY": "risk-reduction only",
        "Reduce Only": "risk-reduction only",
        "Reduce only": "risk-reduction only",
        "SIZE DOWN": "use smaller size",
        "Size Down": "use smaller size",
        "Size down": "use smaller size",
        "NO NEW PAPER SIZE": "no new paper sizing",
        "No New Paper Size": "no new paper sizing",
        "LIVE IC ACTIVE": "live signal tracking is active",
        "Live Ic Active": "live signal tracking is active",
        "PENDING FORWARD RETURNS": "waiting for future price data",
        "Pending Forward Returns": "waiting for future price data",
        "REGIME FRAGILE": "works only in some market conditions",
        "Regime Fragile": "works only in some market conditions",
        "KEEP CORE": "keep as a core research signal",
        "Keep Core": "keep as a core research signal",
        "DOWNWEIGHT": "use less weight",
        "Downweight": "use less weight",
        "BLOCK SIGNAL": "do not use this signal",
        "Block Signal": "do not use this signal",
        "USE ONLY AT SHORT HORIZON": "short-term use only",
        "Use Only At Short Horizon": "short-term use only",
        "NO DATA": "no data yet",
        "No Data": "no data yet",
        "DATA GAP": "missing data",
        "Data Gap": "missing data",
        "WEAK": "weak",
        "Weak": "weak",
        "REVIEW": "needs review",
        "Review": "needs review",
        "ACTIVE": "active",
        "Active": "active",
        "PASS": "passes",
        "Pass": "passes",
        "CLEAR": "clear",
        "Clear": "clear",
        "BLOCKER": "blocked",
        "Blocker": "blocked",
        "TCA": "trading cost",
        "IC": "predictive skill",
        "Ic": "predictive skill",
        "PIT": "time-accurate data",
        "OOS": "fresh test windows",
        "P0": "Top priority",
        "P1": "Next priority",
        "P2": "Later priority",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = text.replace("risk-reduction only and Use smaller size names", "names that risk says to reduce or keep small")
    text = text.replace("risk-reduction only and use smaller size names", "names that risk says to reduce or keep small")
    text = text.replace("Risk-reduction only and Use smaller size names", "Names that risk says to reduce or keep small")
    text = text.replace("risk-reduction only and use smaller size names", "names that risk says to reduce or keep small")
    text = text.replace("risk-reduction only names", "names that risk says to reduce")
    text = text.replace("risk-reduction only", "risk-reduction")
    text = text.replace("use smaller size names", "names that need smaller size")
    text = text.replace("Use smaller size names", "names that need smaller size")
    text = text.replace("not ready yet model ingredients", "weak model ingredients")
    text = text.replace("active scoring", "the main score")
    text = text.replace("new calls or size", "new option ideas or bigger size")
    text = text.replace("Sharpe 4", "the high performance goal")
    text = text.replace("Sharpe", "performance score")
    text = text.replace("backtest", "old-data test")
    text = text.replace("Backtest", "Old-data test")
    text = text.replace("signal", "model ingredient")
    text = text.replace("Signal", "Model ingredient")
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text or "No data"


def _perf_signal_label(value) -> str:
    raw = str(value or "").strip()
    labels = {
        "mom_12m_skip1m": "12-month momentum",
        "trend_200": "200-day trend",
        "mom_6m": "6-month momentum",
        "mom_3m": "3-month momentum",
        "mom_1m": "1-month momentum",
        "mom_accel": "momentum acceleration",
        "new_high_52w": "52-week high",
        "rsi_rev": "oversold rebound",
        "inv_vol": "low-volatility tilt",
        "eps_growth_yoy": "earnings growth",
    }
    return labels.get(raw, raw.replace("_", " ").strip().title() or "Signal")


def _perf_accent(score=None, status: str = "") -> str:
    score_num = _to_float(score)
    status_text = str(status or "").upper()
    if any(x in status_text for x in ["BLOCK", "WEAK", "REPAIR", "NOT RELIABLE"]):
        return "#991b1b"
    if score_num is not None:
        if score_num < 45:
            return "#991b1b"
        if score_num < 72:
            return "#334155"
        return "#166534"
    if any(x in status_text for x in ["REVIEW", "PROTOTYPE", "WAIT"]):
        return "#334155"
    return "#166534"


def _perf_module_score(modules: pd.DataFrame, module_name: str, fallback=None):
    if modules is None or modules.empty or "module" not in modules.columns:
        return fallback, ""
    mask = modules["module"].astype(str).str.lower().str.contains(module_name.lower(), na=False)
    if not mask.any():
        return fallback, ""
    row = modules[mask].iloc[0]
    return row.get("score_0_100", fallback), _perf_plain(row.get("status"), 120)


def _perf_number(value, digits: int = 2) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    return f"{num:.{digits}f}"


def _perf_score_text(value) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    return f"{num:.1f} / 100"


def _render_performance_command_center(
    simple_state: dict,
    target_state: dict,
    depth_state: dict,
    backtest_state: dict,
    live_state: dict,
    exec_state: dict,
    modules: pd.DataFrame,
):
    current = target_state.get("current_headline_sharpe", simple_state.get("current_headline_sharpe"))
    adjusted = target_state.get("credibility_adjusted_planning_sharpe", simple_state.get("proof_adjusted_sharpe"))
    target = target_state.get("target_sharpe", simple_state.get("target_sharpe", 4.0))
    headline_gap = None
    proof_gap = None
    current_num = _to_float(current)
    adjusted_num = _to_float(adjusted)
    target_num = _to_float(target)
    if current_num is not None and target_num is not None:
        headline_gap = max(target_num - current_num, 0)
    if adjusted_num is not None and target_num is not None:
        proof_gap = max(target_num - adjusted_num, 0)

    signal_score, signal_status = _perf_module_score(modules, "Signal", 0)
    backtest_score = backtest_state.get("overall_credibility_score")
    depth_score = depth_state.get("overall_score_0_100")
    exec_score = exec_state.get("execution_cost_model_score")
    live_rows = live_state.get("complete_forward_return_rows", live_state.get("live_ic_windows", 0))
    pending_rows = live_state.get("pending_forward_return_rows", 0)
    claim_allowed = bool(target_state.get("claim_allowed", False))
    accent = "#166534" if claim_allowed else "#991b1b"

    headline = "Do not trust the high target yet." if not claim_allowed else "The performance target is cleared for research reporting."
    explanation = (
        f"The old-data score looks like {_perf_number(current)}, but the stricter trust score is only {_perf_number(adjusted)}. "
        "Use the stricter score when deciding whether the model is actually good."
    )

    st.markdown("#### Can I trust the results?")
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #cbd5e1; border-left:7px solid {accent}; border-radius:10px; padding:18px 20px; margin:8px 0 15px 0;">
          <div style="font-size:12px; color:#64748b; font-weight:900; text-transform:uppercase;">Simple answer</div>
          <div style="font-size:28px; color:#111827; font-weight:950; line-height:1.18; margin-top:6px;">{_esc(headline)}</div>
          <div style="font-size:15px; color:#374151; line-height:1.5; margin-top:10px;">{_esc(explanation)}</div>
          <div style="font-size:12px; color:#6b7280; margin-top:10px;">Translation: this page is saying "not ready for bigger size yet."</div>
        </div>
        """
    )

    cards = [
        ("Can I use it for size?", "No", "The model is still in research mode.", "#991b1b"),
        ("Old-data score", _perf_number(current), "This is the nice-looking historical score.", "#334155"),
        ("Trust score", _perf_number(adjusted), "This is the stricter score after penalties.", "#991b1b" if adjusted_num is not None and adjusted_num < 1 else "#334155"),
        ("Goal score", _perf_number(target), "Your long-term target. It is not proven yet.", "#111827"),
        ("Main problem", "Model ingredients", "Some ingredients do not prove they work yet.", _perf_accent(signal_score, signal_status)),
        ("New-data checks", str(live_rows), f"{pending_rows} checks still wait for future prices.", "#334155"),
        ("Trading friction", _perf_score_text(exec_score), "Trades may cost too much if stress gets worse.", _perf_accent(exec_score, exec_state.get("overall_status"))),
        ("Overall trust", _perf_score_text(depth_score), "Good enough for research, not enough for automatic sizing.", _perf_accent(depth_score, depth_state.get("overall_status"))),
    ]

    html = ['<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:13px; margin:8px 0 20px 0;">']
    for title, value, note, color in cards:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {color}; border-radius:8px; padding:14px 15px; min-height:150px;">
              <div style="font-size:12px; color:#64748b; font-weight:900; text-transform:uppercase;">{_esc(title)}</div>
              <div style="font-size:24px; color:#111827; font-weight:950; line-height:1.15; margin-top:7px;">{_esc(value)}</div>
              <div style="font-size:13px; color:#4b5563; line-height:1.38; margin-top:9px;">{_esc(note)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_perf_next_actions_preview(queue: pd.DataFrame):
    st.markdown("#### What to fix next")
    st.caption("This is the shortest work order before performance numbers can be trusted more.")
    if queue is None or queue.empty:
        st.info("No performance repair queue is available yet.")
        return

    q = queue.copy()
    if "priority" in q.columns:
        priority_rank = {"P0": 0, "P1": 1, "P2": 2}
        q["_rank"] = q["priority"].astype(str).str.upper().map(lambda x: priority_rank.get(x, 9))
        q = q.sort_values(["_rank"], kind="stable")

    parts = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin:8px 0 20px 0;">']
    for _, row in q.head(3).iterrows():
        status = _perf_plain(row.get("status"), 90)
        accent = _perf_accent(status=status)
        workstream_raw = str(row.get("workstream", ""))
        workstream_key = workstream_raw.lower()
        title = _perf_plain(row.get("workstream"), 120)
        action_text = _perf_plain(row.get("action"), 230)
        effect_text = _perf_plain(row.get("expected_sharpe_effect"), 170)
        done_text = _perf_plain(row.get("done_when"), 160)
        if "risk repair" in workstream_key:
            title = "Make risky names smaller first"
            action_text = "Do not let names that risk wants smaller drive new upside ideas. Fix the risk page before reviewing new options or bigger size."
            effect_text = "This lowers the chance that one bad move ruins the whole model."
            done_text = "The risky names are no longer driving the idea queue."
        elif "signal repair" in workstream_key:
            title = "Remove weak model ingredients"
            action_text = "Take weak ingredients out of the main score until they prove they work on newer data."
            effect_text = "This makes the score less noisy and less likely to chase random moves."
            done_text = "Weak ingredients are no longer used as main drivers."
        elif "execution" in workstream_key or "cost" in workstream_key:
            title = "Count real trading friction"
            action_text = "Reduce turnover and require proof that the trade is not too expensive or hard to fill."
            effect_text = "This stops paper profits from disappearing after trading costs."
            done_text = "Cost and turnover are low enough to trust the result more."
        parts.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:9px; padding:15px 16px; min-height:260px;">
              <div style="font-size:12px; color:#64748b; font-weight:900; text-transform:uppercase;">{_esc(_perf_plain(row.get("priority"), 40))} · {_esc(status)}</div>
              <div style="font-size:19px; color:#111827; font-weight:950; line-height:1.18; margin-top:8px;">{_esc(title)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:10px;"><b>Do this:</b> {_esc(action_text)}</div>
              <div style="font-size:12px; color:#4b5563; line-height:1.42; border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px;"><b>Why it helps:</b> {_esc(effect_text)}</div>
              <div style="font-size:11px; color:#64748b; line-height:1.35; margin-top:8px;"><b>Done when:</b> {_esc(done_text)}</div>
            </div>
            """
        )
    parts.append("</div>")
    _render_html("".join(parts))


def _perf_metric_cards(state: dict, target_state: dict, depth_state: dict):
    current = target_state.get("current_headline_sharpe", state.get("current_headline_sharpe"))
    adjusted = target_state.get("credibility_adjusted_planning_sharpe", state.get("proof_adjusted_sharpe"))
    target = target_state.get("target_sharpe", state.get("target_sharpe", 4.0))
    depth_score = depth_state.get("overall_score_0_100")
    options_now = state.get("options_allowed_now_count", 0)
    paper_now = state.get("paper_sizing_allowed_now_count", 0)
    cards = [
        ("Headline Sharpe", _fmt_target_number(current), "The visible local backtest number.", "#334155"),
        ("Proof-adjusted Sharpe", _fmt_target_number(adjusted), "The more honest planning number after evidence haircuts.", "#991b1b"),
        ("Target", _fmt_target_number(target), "The long-term goal, not a current claim.", "#111827"),
        ("Deep Proof Score", _fmt_target_number(depth_score, 1) + " / 100", "Overall depth of the five proof modules.", _perf_accent(depth_score)),
        ("Paper / Options Now", f"{paper_now} / {options_now}", "Allowed today by the clean command center.", "#991b1b" if not paper_now and not options_now else "#334155"),
    ]
    cols = st.columns(5)
    for col, (title, value, note, accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, accent)


def _render_perf_reason_cards(
    backtest_state: dict,
    scorecard: pd.DataFrame,
    live_state: dict,
    exec_state: dict,
    depth_state: dict,
    modules: pd.DataFrame,
    decision_state: dict,
    bias_state: dict,
):
    signal_score, signal_status = _perf_module_score(modules, "Signal", 0)
    backtest_score = backtest_state.get("overall_credibility_score")
    exec_score = exec_state.get("execution_cost_model_score")
    depth_score = depth_state.get("overall_score_0_100")
    bias_score = bias_state.get("score", bias_state.get("backtest_bias_guard_score"))
    weak_rows = 0
    if scorecard is not None and not scorecard.empty and "status" in scorecard.columns:
        weak_rows = int(scorecard["status"].astype(str).str.upper().isin(["WEAK", "BLOCKER"]).sum())
    signal_note = signal_status or "Signals still need forward proof."
    if signal_note and signal_note[-1] not in ".!?":
        signal_note += "."

    reasons = [
        (
            "Old test quality",
            f"{_fmt_target_number(backtest_score, 1)} / 100",
            f"Old results are useful, but not enough. {weak_rows} weak areas still need work.",
            _perf_accent(backtest_score, backtest_state.get("overall_status")),
        ),
        (
            "Do the ingredients work?",
            f"{_fmt_target_number(signal_score, 1)} / 100",
            "Some model ingredients are weak or unstable. Weak ingredients cannot drive bigger size.",
            _perf_accent(signal_score, signal_status),
        ),
        (
            "Trading cost",
            f"{_fmt_target_number(exec_score, 1)} / 100",
            "The idea can look good on paper but shrink after trading costs.",
            _perf_accent(exec_score, exec_state.get("overall_status")),
        ),
        (
            "New-data check",
            str(live_state.get("live_ic_windows", live_state.get("complete_forward_return_rows", 0))),
            f"{live_state.get('pending_forward_return_rows', 0)} checks still wait for future prices. Until then, do not call it proven.",
            "#334155",
        ),
        (
            "Did it cheat by accident?",
            f"{_fmt_target_number(bias_score, 1)} / 100",
            "The test must prove it did not use information that would not have been known at the time.",
            _perf_accent(bias_score, bias_state.get("overall_status")),
        ),
        (
            "Learning log",
            str(decision_state.get("ready_forward_observations", 0)),
            f"{decision_state.get('pending_forward_observations', 0)} past decisions are still waiting for follow-up results.",
            "#166534" if _to_float(decision_state.get("ready_forward_observations"), 0) else "#334155",
        ),
    ]
    st.markdown("#### Why I should not trust it yet")
    for start in range(0, len(reasons), 3):
        cols = st.columns(3)
        for col, (title, value, note, accent) in zip(cols, reasons[start:start + 3]):
            with col:
                _simple_card(title, value, note, accent)


def _render_perf_signal_cards(signal_lab: pd.DataFrame, failure: pd.DataFrame):
    source = failure if failure is not None and not failure.empty else signal_lab
    if source is None or source.empty:
        st.info("No signal proof rows are available yet.")
        return

    work = source.copy()
    action_col = "recommended_signal_action" if "recommended_signal_action" in work.columns else "recommended_action"
    if action_col in work.columns:
        order = {"BLOCK_SIGNAL": 0, "block this signal": 0, "DOWNWEIGHT": 1, "down-weight this signal": 1, "USE_ONLY_AT_SHORT_HORIZON": 2, "KEEP_CORE": 3, "Keep under review": 3}
        work["_order"] = work[action_col].map(lambda x: order.get(str(x), 2))
        work = work.sort_values(["_order"], kind="stable")

    st.markdown("##### Signal scoreboard")
    st.markdown("Read this as: which ingredients are trusted, which should be used less, and which should be blocked for now.")
    if "_order" in work.columns:
        pieces = []
        for bucket, limit in [(0, 3), (1, 2), (2, 1), (3, 2)]:
            part = work[work["_order"].eq(bucket)].head(limit)
            if not part.empty:
                pieces.append(part)
        top = pd.concat(pieces, ignore_index=False) if pieces else work.head(8)
    else:
        top = work.head(8)
    for start in range(0, len(top), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, top.iloc[start:start + 4].iterrows()):
            action = _perf_plain(row.get(action_col), 120)
            status = _perf_plain(row.get("failure_status", row.get("baseline_status", "")), 80)
            accent = _perf_accent(row.get("baseline_mean_ic", row.get("best_mean_ic")), f"{action} {status}")
            best_h = _perf_plain(row.get("best_horizon"), 40)
            worst_h = _perf_plain(row.get("worst_horizon"), 40)
            best_ic = _to_float(row.get("best_horizon_ic", row.get("best_mean_ic")))
            worst_ic = _to_float(row.get("worst_horizon_ic", row.get("worst_mean_ic")))
            best_text = "No best window yet" if best_ic is None else f"Best: {best_h} at {best_ic:.3f}"
            worst_text = "No worst window yet" if worst_ic is None else f"Worst: {worst_h} at {worst_ic:.3f}"
            reason = _perf_plain(row.get("reason", row.get("failure_mode", "")), 150)
            next_action = _perf_plain(row.get("required_next_action", row.get("recommended_action", "")), 190)
            with col:
                _render_html(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 15px; min-height:285px; margin-bottom:12px;">
                      <div style="font-size:19px; color:#111827; font-weight:900; line-height:1.15;">{_esc(_perf_signal_label(row.get("signal")))}</div>
                      <div style="font-size:12px; color:{accent}; font-weight:850; margin-top:6px;">{_esc(action)}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:10px;">{_esc(best_text)}<br>{_esc(worst_text)}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;"><b>Why:</b> {_esc(reason)}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:12px; color:#6b7280; line-height:1.42;"><b>Next:</b> {_esc(next_action)}</div>
                    </div>
                    """
                )


def _render_perf_repair_queue(queue: pd.DataFrame, drivers: pd.DataFrame, blockers: pd.DataFrame):
    st.markdown("##### Repair queue")
    st.markdown("This is the work order before the dashboard can honestly claim a stronger performance target.")
    if queue is None or queue.empty:
        st.info("No Sharpe repair queue is available yet.")
    else:
        q = queue.copy()
        if "priority" in q.columns:
            q = q.sort_values("priority", kind="stable")
        cols = st.columns(3)
        for i, (_, row) in enumerate(q.head(6).iterrows()):
            with cols[i % 3]:
                priority = _perf_plain(row.get("priority"), 40)
                status = _perf_plain(row.get("status"), 90)
                action = _perf_plain(row.get("action"), 260)
                effect = _perf_plain(row.get("expected_sharpe_effect"), 180)
                accent = _perf_accent(status=status)
                _render_html(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {accent}; border-radius:8px; padding:14px 15px; min-height:235px; margin-bottom:12px;">
                      <div style="font-size:12px; color:#6b7280; font-weight:850;">{_esc(priority)} · {_esc(status)}</div>
                      <div style="font-size:16px; color:#111827; font-weight:850; line-height:1.25; margin-top:7px;">{_esc(_perf_plain(row.get("workstream"), 120))}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;">{_esc(action)}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:12px; color:#6b7280; line-height:1.4;">{_esc(effect)}</div>
                    </div>
                    """
                )

    if drivers is not None and not drivers.empty:
        with st.expander("Open the plain-English gap drivers", expanded=False):
            d = drivers.copy()
            for col in ["status", "plain_english", "driver"]:
                if col in d.columns:
                    d[col] = d[col].map(lambda x: _perf_plain(x, 260))
            d_cols = [c for c in ["driver", "current_value", "target_value", "gap", "plain_english", "status"] if c in d.columns]
            _show_status_table(d[d_cols], ["status"], height=340)

    if blockers is not None and not blockers.empty:
        with st.expander("Open backtest blockers", expanded=False):
            b = blockers.copy()
            for col in ["category", "status", "evidence", "next_required_action"]:
                if col in b.columns:
                    b[col] = b[col].map(lambda x: _perf_plain(x, 260))
            b_cols = [c for c in ["priority", "category", "status", "score_0_100", "evidence", "next_required_action"] if c in b.columns]
            _show_status_table(b[b_cols], ["status"], height=380)


def _render_signal_trust_ladder():
    """
    Signal Trust Ladder — classifies each signal by validation level.

    Levels (from weakest to strongest):
      1. Hypothesis   — academic or theoretical basis only, no live test
      2. Proxy        — correlated to known factor but not directly measured
      3. Limited live — some live forward observation (< 6 months)
      4. Validated    — 6+ months live IC track record with documented source

    Reads from:  signal_regime_ic_matrix.csv, live_ic_observation_log.csv,
                 depth5_signal_ic_decay_failure_lab.csv, alpha_scores.csv
    """
    ic_lab  = safe_csv(ROOT / "depth5_signal_ic_decay_failure_lab.csv")
    live_ic = safe_csv(ROOT / "live_ic_observation_log.csv")
    alpha   = safe_csv(ROOT / "alpha_scores.csv")

    # Signal metadata: name → (description, level, basis)
    # Level: 1=Hypothesis, 2=Proxy, 3=LimitedLive, 4=Validated
    SIGNAL_META = {
        "regime_ml":   ("Regime ML", "The model that reads market conditions (bull, bear, sideways). Trained on price, vol, and macro data.", 3, "ML trained on sp500_price_cache; some forward IC observed"),
        "quality":     ("Quality Score", "Company strength: low debt, high ROE, stable earnings. Based on Novy-Marx (2013).", 2, "Fundamental proxy from yfinance; no direct IC validation yet"),
        "momentum":    ("Momentum", "Price momentum: 12-1 month return + 52-week high + vol-scaled. Academic basis Jegadeesh-Titman (1993).", 3, "4-component composite; SPY residual now live; regime dampening built in"),
        "revision":    ("Earnings Revision", "Did analyst estimates go up or down recently? Earnings revision momentum.", 2, "From yfinance estimates; small coverage; no separate IC validation"),
        "surprise":    ("Earnings Surprise", "Did the company beat or miss last earnings? Historical surprise pattern.", 2, "From yfinance; point-in-time not guaranteed; coverage ~60-70%"),
        "sentiment":   ("Sentiment", "News and SEC filing tone (positive/negative). FinBERT model applied to headlines.", 2, "Proxy sentiment; FinBERT applied to headline text; no live IC yet"),
        "squeeze":     ("Short Squeeze", "High short interest + price momentum = squeeze risk or opportunity.", 2, "From yfinance short float; approximate; coverage varies"),
        "insider":     ("Insider Signal", "Are company insiders buying or selling? SEC Form 4 filings.", 2, "SEC EDGAR filings; delayed 2-4 days; coverage ~60%"),
        "options":     ("Options Signal", "Call/put ratio and implied vol changes. Institutional options flow proxy.", 2, "From yfinance options chain; only ~100-200 liquid tickers"),
        "ml_ensemble": ("ML Ensemble", "Combined ML model that integrates multiple factors. Random Forest on sp500 features.", 3, "Trained on historical data; subject to overfitting; no walk-forward yet"),
    }

    LEVEL_LABELS = {
        1: ("Hypothesis", "#6b7280", "#f9fafb"),
        2: ("Proxy evidence", "#854d0e", "#fffbeb"),
        3: ("Limited live", "#1e40af", "#eff6ff"),
        4: ("Validated", "#166534", "#f0fdf4"),
    }

    # Try to get live IC observations to upgrade levels
    observed_sigs = set()
    if not live_ic.empty and "signal" in live_ic.columns:
        obs_sig_col = live_ic["signal"].dropna().unique()
        observed_sigs = {str(s).lower() for s in obs_sig_col}

    with st.expander("Signal Trust Ladder — how validated is each ingredient?", expanded=False):
        st.caption(
            "Each signal in the model is classified by how well it has been tested with real, forward-looking data. "
            "A signal at level 1 (hypothesis) is theoretically sound but untested here. "
            "A signal at level 4 (validated) has a documented live track record. "
            "Most signals here are level 2–3, which is honest for a research system at this stage."
        )

        cols = st.columns(2)
        for i, (sig_key, (sig_name, sig_desc, default_level, basis)) in enumerate(SIGNAL_META.items()):
            # Check live IC for possible upgrade
            level = default_level
            if sig_key in observed_sigs and default_level < 3:
                level = 3
            if not ic_lab.empty and "signal" in ic_lab.columns:
                sig_rows = ic_lab[ic_lab["signal"].astype(str).str.lower() == sig_key]
                if not sig_rows.empty:
                    mean_ic = _to_float(sig_rows["mean_ic"].iloc[0]) if "mean_ic" in sig_rows.columns else None
                    if mean_ic is not None and abs(mean_ic) > 0.04:
                        level = max(level, 3)

            level_label, level_color, level_bg = LEVEL_LABELS.get(level, LEVEL_LABELS[1])
            dots = "●" * level + "○" * (4 - level)

            with cols[i % 2]:
                _render_html(
                    f"""
                    <div style="background:{level_bg}; border:1px solid #d1d5db; border-left:5px solid {level_color}; border-radius:8px; padding:12px 14px; margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:14px; font-weight:900; color:#111827;">{_esc(sig_name)}</div>
                        <div style="font-size:13px; font-weight:900; color:{level_color};">{_esc(dots)}  {_esc(level_label)}</div>
                      </div>
                      <div style="font-size:12px; color:#374151; line-height:1.45; margin-top:6px;">{_esc(sig_desc)}</div>
                      <div style="font-size:11px; color:#6b7280; margin-top:5px; font-style:italic;">{_esc(basis)}</div>
                    </div>
                    """
                )

        _render_html(
            """
            <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px 16px; margin-top:4px;">
              <div style="font-size:12px; color:#6b7280; line-height:1.6;">
                <b>How to read the dots:</b>
                ● Hypothesis — academic basis only &nbsp;|&nbsp;
                ●● Proxy — correlated to known factor &nbsp;|&nbsp;
                ●●● Limited live — some live IC data &nbsp;|&nbsp;
                ●●●● Validated — 6+ months live track record with source proof<br>
                <b>Target:</b> get the top 3 signals (regime_ml, quality, momentum) to level 4 before trusting model sizing.
              </div>
            </div>
            """
        )


def _render_performance_proof_board():
    simple_state = safe_json(ROOT / "sharpe4_simple_command_state.json")
    target_state = safe_json(ROOT / "sharpe_target4_state.json")
    depth_state = safe_json(ROOT / "institutional_depth5_state.json")
    backtest_state = safe_json(ROOT / "backtest_credibility_state.json")
    live_state = safe_json(ROOT / "live_ic_observation_state.json")
    exec_state = safe_json(ROOT / "execution_cost_model_state.json")
    decision_state = safe_json(ROOT / "decision_memory_state.json")
    bias_state = safe_json(ROOT / "backtest_bias_state.json")
    modules = safe_csv(ROOT / "institutional_depth5_module_scorecard.csv")
    scorecard = safe_csv(ROOT / "backtest_credibility_scorecard.csv")
    blockers = safe_csv(ROOT / "backtest_credibility_blockers.csv")
    drivers = safe_csv(ROOT / "sharpe_target4_driver_attribution.csv")
    queue = safe_csv(ROOT / "sharpe_target4_action_queue.csv")
    signal_lab = safe_csv(ROOT / "depth5_signal_ic_decay_failure_lab.csv")
    failure = safe_csv(ROOT / "signal_failure_deep_dive.csv")

    if not any([simple_state, target_state, depth_state, backtest_state]) and modules.empty and signal_lab.empty:
        st.info("Performance proof files are missing. Run the main daily runner before reading this page.")
        return

    current = target_state.get("current_headline_sharpe", simple_state.get("current_headline_sharpe"))
    adjusted = target_state.get("credibility_adjusted_planning_sharpe", simple_state.get("proof_adjusted_sharpe"))
    target = target_state.get("target_sharpe", simple_state.get("target_sharpe", 4.0))
    claim_allowed = bool(target_state.get("claim_allowed", False))
    accent = "#166534" if claim_allowed else "#991b1b"
    answer = "Sharpe 4 is not proven yet." if not claim_allowed else "Sharpe 4 claim is cleared for research reporting."
    sub_answer = (
        f"Headline Sharpe is {_fmt_target_number(current)}, but the proof-adjusted number is {_fmt_target_number(adjusted)}. "
        "That gap means the system must repair signal proof, bias control, trading cost realism, and live forward checks before it can size more."
    )

    st.markdown('<p class="section-title">Can I Trust The Results?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Start here. This page answers whether the model result is strong enough to trust, or only useful for research.</p>',
        unsafe_allow_html=True,
    )

    # ── Survivorship Bias Warning ─────────────────────────────────────────────
    _render_html(
        """
        <div style="background:#fffbeb; border:1px solid #d97706; border-left:6px solid #d97706; border-radius:9px; padding:16px 20px; margin:10px 0 16px 0;">
          <div style="font-size:13px; font-weight:900; color:#92400e; text-transform:uppercase; letter-spacing:.04em;">⚠ Data limitation — read before trusting any backtest number</div>
          <div style="font-size:14px; color:#78350f; line-height:1.6; margin-top:8px;">
            <b>Survivorship bias is present in all backtests on this system.</b>
            The price history uses today's S&P 500 list going back to 2000.
            Companies that were removed from the index (delisted, merged, failed) are <em>not</em> in the data.
            This makes every backtest look better than it would have in real life.
          </div>
          <div style="font-size:13px; color:#92400e; line-height:1.55; margin-top:9px;">
            <b>What this means:</b> A headline Sharpe of 1.37 on this system is probably 0.6–0.9 on clean, point-in-time data.
            Treat all backtest numbers as <em>hypothesis-grade</em>, not production-grade.
            The proof-adjusted Sharpe below is a better (but still not perfect) estimate.
          </div>
          <div style="font-size:12px; color:#b45309; margin-top:8px;">
            Fix: add a survivorship-free universe (e.g., CRSP or Compustat) before relying on backtest results for sizing decisions.
          </div>
        </div>
        """
    )

    _render_performance_command_center(simple_state, target_state, depth_state, backtest_state, live_state, exec_state, modules)
    _render_perf_reason_cards(backtest_state, scorecard, live_state, exec_state, depth_state, modules, decision_state, bias_state)
    _render_perf_next_actions_preview(queue)

    # ── Signal Trust Ladder ───────────────────────────────────────────────────
    _render_signal_trust_ladder()

    with st.expander("Open detailed model-ingredient scoreboard", expanded=False):
        _render_perf_signal_cards(signal_lab, failure)

    with st.expander("Open detailed repair queue and blockers", expanded=False):
        _render_perf_repair_queue(queue, drivers, blockers)


def _render_sharpe_target4_panel():
    state = safe_json(ROOT / "sharpe_target4_state.json")
    drivers = safe_csv(ROOT / "sharpe_target4_driver_attribution.csv")
    queue = safe_csv(ROOT / "sharpe_target4_action_queue.csv")
    tickers = safe_csv(ROOT / "sharpe_target4_ticker_gate.csv")
    policy = safe_csv(ROOT / "sharpe_target4_policy.csv")

    if not state and drivers.empty and queue.empty:
        st.info("Sharpe 4 Target Desk has not run yet. Run Step185 to generate the target gap and repair queue.")
        return

    target = state.get("target_sharpe", 4.0)
    current = state.get("current_headline_sharpe")
    gap = state.get("gap_to_target")
    planning = state.get("credibility_adjusted_planning_sharpe")
    claim_allowed = bool(state.get("claim_allowed", False))
    status = _plain_status(state.get("target_status"), "No status")
    accent = "#166534" if claim_allowed else "#991b1b" if _to_float(gap, 0) and _to_float(gap, 0) > 2 else "#334155"

    st.markdown('<p class="section-title">Sharpe 4 Target</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">A plain-English control panel for the Sharpe 4 goal. This does not claim the target is reached; it shows what must be repaired first.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cards = [
        ("Target", _fmt_target_number(target), "Goal level requested by user."),
        ("Current headline Sharpe", _fmt_target_number(current), "Visible backtest number."),
        ("Gap to target", _fmt_target_number(gap), "How much the headline number is short."),
        ("Proof-adjusted Sharpe", _fmt_target_number(planning), "Haircut for data, signal, execution, and optimizer proof."),
        ("Can claim Sharpe 4?", "Yes" if claim_allowed else "No", status),
    ]
    for col, (title, value, note) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, accent if title == "Can claim Sharpe 4?" else "#334155")

    st.markdown(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:15px 16px; margin:14px 0;">
          <div style="font-size:16px; font-weight:850; color:#111827;">Current answer</div>
          <div style="font-size:14px; line-height:1.55; color:#374151; margin-top:6px;">
            The system is not at Sharpe 4 yet. The headline Sharpe is {_esc(_fmt_target_number(current))}, while the proof-adjusted planning number is {_esc(_fmt_target_number(planning))}.
            Before the model can credibly target 4, it must repair signal proof, execution cost realism, point-in-time data, and risk scaling.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_sharpe4_p0_repair_pack()
    _render_sharpe4_recovery_roadmap()
    _render_sharpe4_risk_book_intake()

    if not queue.empty:
        p0 = queue[queue.get("priority", "").astype(str).str.upper().eq("P0")] if "priority" in queue.columns else queue.head(3)
        st.markdown("##### Repair first")
        action_cols = st.columns(3)
        for i, (_, row) in enumerate(p0.head(3).iterrows()):
            status = _plain_status(row.get("status"), "Review")
            with action_cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid #334155; border-radius:8px; padding:14px 15px; min-height:260px; margin-bottom:10px;">
                      <div style="font-size:12px; font-weight:850; color:#6b7280; text-transform:uppercase;">{_esc(row.get("priority"), "P0")} · {_esc(status)}</div>
                      <div style="font-size:16px; font-weight:850; color:#111827; margin-top:7px; line-height:1.25;">{_esc(row.get("workstream"), "Repair work")}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.48; margin-top:9px;">{_esc(row.get("action"), "")}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:12px; color:#6b7280; line-height:1.42;">
                        Why it matters: {_esc(row.get("expected_sharpe_effect"), "")}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with st.expander("Open detailed repair checklist", expanded=False):
            _show_status_table(
                p0[[c for c in ["priority", "workstream", "action", "expected_sharpe_effect", "done_when", "status"] if c in p0.columns]],
                ["status"],
                height=260,
            )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("##### Why the target is blocked")
        if drivers.empty:
            st.info("No driver attribution file yet.")
        else:
            cols = [c for c in ["driver", "current_value", "target_value", "gap", "plain_english", "status"] if c in drivers.columns]
            _show_status_table(drivers[cols], ["status"], height=360)

    with right:
        st.markdown("##### Which names cannot help yet")
        if tickers.empty:
            st.info("No ticker target-gate file yet.")
        else:
            cols = [c for c in ["ticker", "sector", "sharpe4_role", "can_help_target4", "current_blocker", "plain_next_step"] if c in tickers.columns]
            _show_status_table(tickers[cols].head(8), ["can_help_target4"], height=360)

    with st.expander("Claim rules and source files", expanded=False):
        if not policy.empty:
            _show_status_table(policy, ["current_gate"], height=300)
        st.markdown(
            "Source files: `sharpe_target4_state.json`, `sharpe_target4_driver_attribution.csv`, "
            "`sharpe_target4_action_queue.csv`, `sharpe_target4_ticker_gate.csv`, and current backtest / risk / TCA files."
        )


def _render_factor_attribution_panel():
    """
    Fama-French 5-Factor attribution panel.
    Shows Jensen's α, factor betas, and verdict from step221.
    """
    fa = safe_csv(ROOT / "factor_attribution.csv")
    if fa.empty:
        return

    with st.expander("Factor attribution — how much of the return is real alpha vs. factor exposure?", expanded=True):
        st.caption(
            "Regression of portfolio returns on 5 Fama-French factors (Market, Size, Value, Profitability, Investment). "
            "Jensen's α = return NOT explained by systematic factors. "
            "t-stat > 2.0 required for 95% statistical significance. "
            "Source: Ken French data library (free daily factors)."
        )
        for _, row in fa.iterrows():
            window   = str(row.get("window",""))
            alpha_a  = _to_float(row.get("alpha_ann"))
            t_stat   = _to_float(row.get("alpha_t"))
            r2       = _to_float(row.get("r_squared"))
            ir       = _to_float(row.get("info_ratio"))
            n_obs    = int(_to_float(row.get("n_obs"),0) or 0)
            if alpha_a is None or t_stat is None:
                continue
            is_sig   = abs(t_stat) >= 2.0
            accent   = "#16a34a" if is_sig and alpha_a > 0 else "#dc2626" if not is_sig else "#d97706"
            verdict  = ("✅ Statistically significant" if is_sig and alpha_a > 0
                        else "⚠ Not yet significant — likely factor exposure" if not is_sig
                        else "Significant but negative")
            _render_html(
                f"""
                <div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid {accent};
                    border-radius:9px;padding:14px 18px;margin-bottom:10px;
                    box-shadow:0 1px 4px rgba(0,0,0,.05);">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                    <div>
                      <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">{_esc(window)}</div>
                      <div style="font-size:24px;font-weight:900;color:{accent};margin-top:3px;">
                        α = {f'{alpha_a*100:+.2f}%' if alpha_a else '—'}/yr
                      </div>
                      <div style="font-size:13px;color:#475569;margin-top:4px;">{verdict}</div>
                    </div>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;">
                      <div style="text-align:center;">
                        <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;">t-stat</div>
                        <div style="font-size:18px;font-weight:800;color:{'#16a34a' if abs(t_stat or 0)>=2 else '#dc2626'};">{f'{t_stat:.2f}' if t_stat else '—'}</div>
                      </div>
                      <div style="text-align:center;">
                        <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;">R²</div>
                        <div style="font-size:18px;font-weight:800;color:#475569;">{f'{r2:.3f}' if r2 else '—'}</div>
                      </div>
                      <div style="text-align:center;">
                        <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;">Info Ratio</div>
                        <div style="font-size:18px;font-weight:800;color:#475569;">{f'{ir:.2f}' if ir else '—'}</div>
                      </div>
                      <div style="text-align:center;">
                        <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;">N obs</div>
                        <div style="font-size:16px;font-weight:700;color:#94a3b8;">{n_obs}d</div>
                      </div>
                    </div>
                  </div>
                  <div style="margin-top:10px;font-size:11px;color:#94a3b8;">
                    Factor betas:
                    Mkt {_to_float(row.get('beta_Mkt_RF'), 0):.3f} ·
                    SMB {_to_float(row.get('beta_SMB'), 0):.3f} ·
                    HML {_to_float(row.get('beta_HML'), 0):.3f} ·
                    RMW {_to_float(row.get('beta_RMW'), 0):.3f} ·
                    CMA {_to_float(row.get('beta_CMA'), 0):.3f}
                  </div>
                </div>
                """
            )
        st.caption(
            "⚠ Survivorship bias warning: returns computed from current S&P 500 constituents. "
            "α estimates are inflated by 2–5% due to data limitations. "
            "Treat as hypothesis-grade, not production-grade."
        )


def _render_ic_tracker_panel():
    """
    Rolling IC tracker panel — signal validation scorecard.
    Source: ic_summary.csv (from step222).
    """
    ic_sum  = safe_csv(ROOT / "ic_summary.csv")
    ic_log  = safe_csv(ROOT / "ic_daily_log.csv")
    n_total = len(ic_log) if not ic_log.empty else 0
    n_dates = int(ic_log["date"].nunique()) if not ic_log.empty and "date" in ic_log.columns else 0

    with st.expander(f"Signal IC tracker — {n_total} observations across {n_dates} days so far", expanded=False):
        st.caption(
            "IC = Spearman correlation between today's signal score and actual forward return N days later. "
            "IC > 0.05 is strong · IC-IR > 0.40 is production-grade · t-stat > 2.0 = statistically proven. "
            "This accumulates daily — run the daily update every day to build the evidence base."
        )
        if ic_sum.empty or n_total == 0:
            _render_html(
                """
                <div style="background:#fffbeb;border:1px solid #d97706;border-radius:9px;padding:16px 20px;">
                  <div style="font-size:14px;font-weight:700;color:#92400e;">IC tracker just started</div>
                  <div style="font-size:13px;color:#78350f;margin-top:6px;">
                    Run the daily update every day. After 20 observations, preliminary verdicts appear.
                    After 100 observations per signal, t-stats become meaningful.
                    Estimated time to first real verdict: <b>3–4 months</b> of daily runs.
                  </div>
                  <div style="font-size:12px;color:#b45309;margin-top:8px;">
                    What's happening: each day, the tracker records how well yesterday's signal scores
                    predicted today's actual stock returns. Over time, this reveals which signals have
                    genuine predictive power vs. which ones are noise.
                  </div>
                </div>
                """
            )
            return

        for horizon in sorted(ic_sum["horizon_days"].unique()):
            sub = ic_sum[ic_sum["horizon_days"] == horizon]
            st.markdown(f"**{int(horizon)}-day forward IC** ({int(horizon)} trading days forward)")
            for _, r in sub.iterrows():
                n       = int(_to_float(r.get("n_obs"),0) or 0)
                mic     = _to_float(r.get("mean_ic"))
                icir    = _to_float(r.get("ic_ir"))
                tstat   = _to_float(r.get("t_stat"))
                verdict = str(r.get("verdict",""))
                pct_pos = _to_float(r.get("pct_positive"))
                is_proven = tstat and abs(tstat) >= 2.0
                bar_color = "#16a34a" if is_proven and mic and mic > 0 else "#dc2626" if mic and mic < 0 else "#e2e8f0"
                bar_w = max(0, min(100, int((mic or 0) * 500 + 50)))
                _render_html(
                    f"""
                    <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f1f5f9;">
                      <div style="min-width:200px;font-size:12px;font-weight:600;color:#334155;">{_esc(str(r.get('signal_name','')))}</div>
                      <div style="font-size:10px;color:#94a3b8;min-width:40px;text-align:right;">{n} obs</div>
                      <div style="flex:1;background:#f1f5f9;border-radius:3px;height:8px;overflow:hidden;">
                        <div style="width:{bar_w}%;background:{bar_color};height:8px;border-radius:3px;"></div>
                      </div>
                      <div style="font-size:11px;font-weight:700;color:{bar_color};min-width:60px;">{f'IC {mic:+.4f}' if mic else '—'}</div>
                      <div style="font-size:10px;color:#64748b;min-width:120px;">{_esc(verdict)}</div>
                    </div>
                    """
                )


def tab_performance():
    if not callable(_ORIGINAL_TAB_PERFORMANCE):
        st.error("Performance page is not available in the cached dashboard snapshot.")
        return

    _render_section_depth("Performance")
    _render_performance_proof_board()
    _render_factor_attribution_panel()
    _render_ic_tracker_panel()

    show_details = st.checkbox("Show technical tables", value=False, key="performance_show_deep_tables")
    if not show_details:
        st.caption("Technical tables are hidden by default. Open them only when you want the raw files.")
        return

    _render_sharpe4_simple_command_center()
    _render_institutional_depth5_workbench()
    _render_decision_memory_center(compact=False)

    with st.expander("Open detailed Sharpe 4 diagnostics", expanded=False):
        _render_sharpe_target4_panel()

    original_risk_desk = _cached.get("_show_institutional_risk_desk")
    _cached["_show_institutional_risk_desk"] = lambda: None
    try:
        with st.expander("Open original performance report", expanded=False):
            return _ORIGINAL_TAB_PERFORMANCE()
    finally:
        if original_risk_desk is not None:
            _cached["_show_institutional_risk_desk"] = original_risk_desk


def _risk_accent(status: str) -> str:
    s = str(status or "").upper()
    if any(x in s for x in ["REDUCE", "BLOCK", "HARD", "CRITICAL"]):
        return "#991b1b"
    if any(x in s for x in ["SIZE_DOWN", "REVIEW", "WARNING"]):
        return "#334155"
    if any(x in s for x in ["CLEAR", "OK"]):
        return "#166534"
    return "#111827"


def _risk_human_action(status: str) -> str:
    s = str(status or "").upper()
    if "REDUCE" in s:
        return "Reduce risk first"
    if "BLOCK" in s:
        return "Do not add"
    if "SIZE_DOWN" in s:
        return "Use smaller size"
    if "REVIEW" in s:
        return "Review before action"
    if "CLEAR" in s or "OK" in s:
        return "Clear"
    return _plain_status(status)


def _risk_front_answer(master_action: str, recommended_gross, normal_gross) -> str:
    text = str(master_action or "").upper()
    if any(x in text for x in ["REDUCE", "SIZE_DOWN", "BLOCK", "HARD", "CRITICAL"]):
        if recommended_gross is not None and normal_gross is not None:
            return (
                f"No new risk yet. The portfolio should be around {recommended_gross * 100:.0f}% gross exposure, "
                f"not the normal {normal_gross * 100:.0f}%."
            )
        return "No new risk yet. Fix the risk blockers before looking for new ideas."
    if any(x in text for x in ["CLEAR", "OK"]):
        return "Risk is not blocking new research, but every ticker still needs its own checks."
    return "Risk needs review. Treat new ideas as watch-only until the risk files are clear."


def _risk_first_fix_text(breaches: pd.DataFrame, queue: pd.DataFrame) -> str:
    if breaches is not None and not breaches.empty:
        row = breaches.iloc[0]
        item = _clean_display(row.get("budget_item"), "the largest risk limit")
        action = _clean_display(row.get("required_next_action"), "reduce or review before adding risk")
        return f"{item}: {_human_text(action, max_len=170)}"
    if queue is not None and not queue.empty:
        row = queue.iloc[0]
        ticker = _clean_display(row.get("ticker"), "the first ticker")
        action = _risk_human_action(row.get("final_risk_action"))
        return f"Start with {ticker}: {action.lower()} before any new position size."
    return "No urgent risk fix is visible. Still check exposure, drawdown, and event risk before adding size."


def _risk_plain_reason(row) -> str:
    raw = _clean_display(row.get("reason_stack"), "")
    if raw and raw != "No data":
        return _human_text(raw, max_len=150)
    pieces = []
    for col, label in [
        ("single_name_action", "single-stock risk"),
        ("earnings_gap_action", "earnings jump risk"),
        ("kelly_status", "signal-size proof"),
        ("sector_status", "sector concentration"),
        ("liquidity_crisis_status", "liquidity stress"),
    ]:
        val = _clean_display(row.get(col), "")
        if val and val != "No data":
            pieces.append(f"{label}: {val}")
    return _human_text("; ".join(pieces) or "Risk gate is limiting this name.", max_len=150)


def _render_risk_plain_summary(overview: dict, breaches: pd.DataFrame, queue: pd.DataFrame):
    master_action = str(overview.get("master_risk_action", "No data"))
    recommended_gross = _to_float(overview.get("recommended_gross_exposure"))
    normal_gross = _to_float(overview.get("normal_gross_exposure"))
    accent = _risk_accent(master_action)
    front_answer = _risk_front_answer(master_action, recommended_gross, normal_gross)
    first_fix = _risk_first_fix_text(breaches, queue)

    st.markdown("#### Risk in plain English")
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:8px; padding:16px 18px; margin:8px 0 14px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">Can the portfolio add risk?</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.25; margin-top:5px;">{_esc(front_answer)}</div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:10px;"><b>First fix:</b> {_esc(first_fix)}</div>
        </div>
        """
    )

    cards = [
        (
            "1. Add new risk?",
            "No" if any(x in str(master_action).upper() for x in ["REDUCE", "SIZE_DOWN", "BLOCK", "HARD", "CRITICAL"]) else "Maybe",
            "If this says no, do not look for calls, puts, or bigger size yet.",
            accent,
        ),
        (
            "2. What blocks it?",
            f"{len(breaches):,}" if breaches is not None and not breaches.empty else "0",
            "Risk limits that need attention before the portfolio can grow.",
            "#991b1b" if breaches is not None and not breaches.empty else "#166534",
        ),
        (
            "3. Which stocks?",
            f"{len(queue):,}" if queue is not None and not queue.empty else "0",
            "Ticker-level items to reduce, review, or repair.",
            "#334155",
        ),
        (
            "4. Rule",
            "Risk first",
            "A good idea cannot override a bad risk setup.",
            "#111827",
        ),
    ]
    c1, c2, c3, c4 = st.columns(4)
    for col, (title, value, note, color) in zip([c1, c2, c3, c4], cards):
        with col:
            _simple_card(title, value, note, color)


def _risk_percent(value, already_pct: bool = False) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    if not already_pct:
        num *= 100
    return f"{num:.2f}%"


def _risk_limit_plain_name(value) -> str:
    text = str(value or "").upper()
    if "SINGLE-NAME" in text or "SINGLE NAME" in text:
        return "One stock can lose too much"
    if "MACRO SCENARIO" in text:
        return "A bad market could lose too much"
    if "CRISIS-CORRELATION" in text or "CRISIS CORRELATION" in text:
        return "In a crisis, positions may fall together"
    if "ANNUAL VOLATILITY" in text:
        return "The portfolio is moving too much"
    if "TOTAL GROSS" in text:
        return "The portfolio is already near full size"
    if "FACTOR BETA" in text:
        return "Too much exposure to one market driver"
    if "SECTOR CONCENTRATION" in text:
        return "Too much exposure to one sector"
    if "VAR" in text or "CVAR" in text:
        return "One bad day loss estimate is high"
    return _human_text(value, max_len=90)


def _risk_status_plain(status) -> str:
    text = str(status or "").upper()
    if "REDUCE_ONLY" in text or "REDUCE" in text:
        return "No new buying"
    if "SIZE_DOWN" in text:
        return "Use smaller size"
    if "REVIEW" in text:
        return "Needs review"
    if "CLEAR" in text or "OK" in text:
        return "Looks okay"
    if "BLOCK" in text:
        return "Do not add"
    return _human_text(status, max_len=80)


def _risk_limit_plain_next(row) -> str:
    status = str(row.get("status", "") or "").upper()
    item = str(row.get("budget_item", "") or "").upper()
    if "REDUCE" in status:
        return "Do not add to this risk. Reduce paper exposure or keep it watch-only."
    if "SIZE_DOWN" in status:
        return "Shrink paper size first, then rerun the daily system."
    if "REVIEW" in status:
        if "SECTOR" in item:
            return "Check whether too many names depend on the same sector move."
        if "FACTOR" in item:
            return "Check whether the book is mostly one market bet in disguise."
        return "Read the evidence before any new paper action."
    return "No urgent action from this limit."


def _risk_usage_plain(value) -> str:
    num = _to_float(value)
    if num is None:
        return "No usage data"
    if num >= 125:
        return f"{num:.0f}% of limit, too high"
    if num >= 100:
        return f"{num:.0f}% of limit, at the line"
    if num >= 85:
        return f"{num:.0f}% of limit, close to the line"
    return f"{num:.0f}% of limit"


def _risk_ticker_plain_reason(row) -> str:
    reasons = []
    checks = [
        ("single_name_action", "one-stock downside is too large"),
        ("earnings_gap_action", "earnings or gap risk is too high"),
        ("gap_down_action", "a sudden price drop would hurt too much"),
        ("kelly_status", "the signal does not justify this much size yet"),
        ("sector_status", "the sector exposure is already crowded"),
        ("liquidity_crisis_status", "selling could be harder in a stressed market"),
    ]
    for col, phrase in checks:
        val = str(row.get(col, "") or "").upper()
        if any(x in val for x in ["SIZE_DOWN", "REDUCE", "BLOCK", "REVIEW"]):
            reasons.append(phrase)
    if not reasons:
        return "Portfolio-level risk says this name should stay smaller until the book improves."
    return "Because " + ", ".join(reasons[:3]) + "."


def _risk_ticker_plain_action(row) -> str:
    current_w = _to_float(row.get("current_weight_pct"))
    target_w = _to_float(row.get("recommended_risk_weight_pct"))
    if current_w is not None and target_w is not None:
        if target_w < current_w:
            return f"Move paper weight from about {current_w:.1f}% toward {target_w:.1f}%."
        return f"Keep near {target_w:.1f}% unless risk improves."
    return "Keep this name watch-only until risk sizing is clear."


def _risk_main_problem(breaches: pd.DataFrame) -> str:
    if breaches is None or breaches.empty:
        return "No major limit breach"
    return _risk_limit_plain_name(breaches.iloc[0].get("budget_item"))


def _render_risk_human_limits(breaches: pd.DataFrame):
    if breaches is None or breaches.empty:
        st.success("No major risk limit is currently over the line.")
        return

    work = breaches.copy()
    if "used_pct_display" in work.columns:
        work["_sort"] = pd.to_numeric(work["used_pct_display"], errors="coerce").fillna(0)
        work = work.sort_values("_sort", ascending=False)

    st.markdown("#### Fix these first")
    st.caption("These are the reasons the system is saying no to new risk right now.")

    html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin:8px 0 20px 0;">']
    for _, row in work.head(6).iterrows():
        status = _risk_status_plain(row.get("status"))
        accent = _risk_accent(row.get("status"))
        item = _risk_limit_plain_name(row.get("budget_item"))
        usage = _risk_usage_plain(row.get("used_pct_display"))
        next_step = _risk_limit_plain_next(row)
        source = _friendly_source_label(row.get("source_file"))
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:15px 16px; min-height:210px;">
              <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">{_esc(status)}</div>
              <div style="font-size:20px; color:#111827; font-weight:900; line-height:1.22; margin-top:7px;">{_esc(item)}</div>
              <div style="font-size:15px; color:#374151; line-height:1.35; margin-top:9px;">{_esc(usage)}</div>
              <div style="font-size:13px; color:#111827; line-height:1.45; margin-top:12px;"><b>Do this:</b> {_esc(next_step)}</div>
              <div style="font-size:11px; color:#6b7280; line-height:1.35; margin-top:10px;">Source: {_esc(source)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_risk_human_tickers(queue: pd.DataFrame):
    if queue is None or queue.empty:
        st.info("No stock-level risk queue is available yet.")
        return

    work = queue.copy()
    if "risk_reduction_pct_of_current" in work.columns:
        work["_reduction"] = pd.to_numeric(work["risk_reduction_pct_of_current"], errors="coerce").fillna(0)
        work = work.sort_values("_reduction", ascending=False)

    st.markdown("#### Stocks to review first")
    st.caption("These are not trade ideas. They are the names risk wants checked or resized first.")

    html = ['<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; margin:8px 0 20px 0;">']
    for _, row in work.head(8).iterrows():
        ticker = _clean_display(row.get("ticker"), "")
        sector = _clean_display(row.get("sector"), "No sector")
        status = _risk_status_plain(row.get("final_risk_action"))
        accent = _risk_accent(row.get("final_risk_action"))
        action = _risk_ticker_plain_action(row)
        reason = _risk_ticker_plain_reason(row)
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {accent}; border-radius:8px; padding:14px 15px; min-height:255px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div style="font-size:24px; color:#111827; font-weight:900;">{_esc(ticker)}</div>
                <div style="font-size:11px; color:{accent}; font-weight:850; text-transform:uppercase; text-align:right;">{_esc(status)}</div>
              </div>
              <div style="font-size:12px; color:#6b7280; margin-top:2px;">{_esc(sector)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:10px; font-size:14px; color:#111827; line-height:1.4;"><b>What to do:</b> {_esc(action)}</div>
              <div style="font-size:12px; color:#4b5563; line-height:1.4; margin-top:10px;"><b>Why:</b> {_esc(reason)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _risk_weight_text(value) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    return f"{num:.1f}%"


def _risk_blocker_plain(value, max_len: int = 170) -> str:
    text = _risk_code_plain(value, max_len=None)
    replacements = {
        "monitor": "alert check",
        "live monitor": "alert check",
        "spread/trading cost": "trading-cost proof",
        "spread/TCA": "trading-cost proof",
        "event proof": "news or event proof",
        "no call thesis": "no proven call thesis",
        "IV/Greeks/Gamma": "option risk details",
        "option volatility, Greeks, and gamma": "option risk details",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = text.replace(";", ",")
    text = " ".join(text.split())
    if not text:
        text = "No extra blocker listed."
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _risk_repair_action_plain(value) -> str:
    text = str(value or "").upper()
    if "MANDATORY" in text or "REPAIR_TO_RISK_TARGET" in text:
        return "Must get smaller"
    if "SIZE_DOWN" in text:
        return "Make smaller"
    if "REVIEW" in text:
        return "Needs a human check"
    return _risk_code_plain(value, 90)


def _risk_route_plain(value) -> str:
    text = str(value or "").upper()
    if "NO_NEW_EXPOSURE" in text or "NO NEW" in text:
        return "No new exposure yet"
    if "WATCH" in text:
        return "Watch only"
    if "CALL" in text:
        return "Call research only"
    if "PUT" in text or "HEDGE" in text:
        return "Put or hedge research only"
    return _risk_code_plain(value, 120)


def _risk_reopen_plain(value) -> str:
    text = _risk_code_plain(value, max_len=None)
    replacements = {
        "Reduce or hold the ticker until the recommended scenario leaves it <= individual safe size, then rerun Steps 176-178.": "Make this ticker small enough, then run the daily update again.",
        "recommended scenario": "safer account setup",
        "individual safe size": "safe size",
        "rerun Steps 176-178": "run the daily update again",
        "<=": "at or below",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = re.sub(r"Steps?\\s+\\d+(?:-\\d+)?", "the daily update", text)
    text = " ".join(text.split())
    if len(text) > 165:
        text = text[:162].rstrip() + "..."
    return text or "Make this ticker safer, then run the daily update again."


def _risk_first_repair_ticker(repair_board: pd.DataFrame, queue: pd.DataFrame) -> str:
    source = repair_board if repair_board is not None and not repair_board.empty else queue
    if source is None or source.empty or "ticker" not in source.columns:
        return "No ticker"
    return _clean_display(source.iloc[0].get("ticker"), "No ticker")


def _render_risk_command_center(
    overview: dict,
    breaches: pd.DataFrame,
    queue: pd.DataFrame,
    budget: pd.DataFrame,
    var_cvar: pd.DataFrame,
    repair_board: pd.DataFrame,
):
    master_action = str(overview.get("master_risk_action", "No data"))
    recommended_gross = _to_float(overview.get("recommended_gross_exposure"))
    normal_gross = _to_float(overview.get("normal_gross_exposure"))
    annual_vol = _to_float(overview.get("annual_vol_pct"))
    target_vol = _to_float(overview.get("target_vol_pct"))
    var_pct = _to_float(overview.get("var_95_1d_pct"))
    cvar_pct = _to_float(overview.get("cvar_95_1d_pct"))
    cvar_dollars = _to_float(overview.get("cvar_95_1d_dollars"))
    hard_breaches = int(_to_float(overview.get("budget_hard_breach_count"), 0) or 0)
    size_downs = int(_to_float(overview.get("budget_size_down_count"), 0) or 0)
    first_ticker = _risk_first_repair_ticker(repair_board, queue)

    blocked = any(x in master_action.upper() for x in ["REDUCE", "SIZE_DOWN", "BLOCK", "HARD", "CRITICAL"])
    accent = _risk_accent(master_action)
    if blocked:
        headline = "Do not add anything new yet."
        short_reason = "The account is already moving too much. First make the riskiest names smaller."
    else:
        headline = "It may be okay to research a new idea."
        short_reason = "Still check the ticker, news, trading cost, and option route before any paper action."

    gross_line = "No account-size target is available yet."
    if recommended_gross is not None and normal_gross is not None:
        gross_line = f"Use about {recommended_gross * 100:.0f}% of the normal account size for now."
    vol_line = "No movement target is available yet."
    if annual_vol is not None and target_vol is not None:
        vol_line = f"The account is moving {annual_vol:.1f}% versus a {target_vol:.1f}% comfort line."
    loss_line = _risk_loss_estimate_line(overview, var_cvar)

    cards = [
        ("Can I add a new idea?", "No" if blocked else "Maybe", "If this says no, keep ideas watch-only.", accent),
        ("Account size today", f"{recommended_gross * 100:.0f}%" if recommended_gross is not None else "No data", gross_line, accent),
        ("Is it moving too much?", f"{annual_vol:.1f}%" if annual_vol is not None else "No data", vol_line, "#991b1b" if annual_vol and target_vol and annual_vol > target_vol else "#166534"),
        ("First name to check", first_ticker, "Start here before searching for new trades.", "#334155"),
        ("Problems to fix", str(hard_breaches + size_downs), "These are the reasons new ideas must wait.", "#991b1b" if hard_breaches or size_downs else "#166534"),
        ("Bad day estimate", f"{var_pct:.1f}% / {cvar_pct:.1f}%" if var_pct is not None and cvar_pct is not None else "No data", loss_line, "#334155"),
    ]

    st.markdown("#### Is it safe to add anything?")
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #cbd5e1; border-left:7px solid {accent}; border-radius:10px; padding:18px 20px; margin:8px 0 15px 0;">
          <div style="font-size:12px; color:#64748b; font-weight:900; text-transform:uppercase;">Simple answer</div>
          <div style="font-size:28px; color:#111827; font-weight:950; line-height:1.18; margin-top:6px;">{_esc(headline)}</div>
          <div style="font-size:15px; color:#374151; line-height:1.48; margin-top:10px;">{_esc(short_reason)}</div>
          <div style="font-size:12px; color:#6b7280; margin-top:10px;">Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:13px; margin:8px 0 20px 0;">']
    for title, value, note, color in cards:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {color}; border-radius:8px; padding:14px 15px; min-height:158px;">
              <div style="font-size:12px; color:#64748b; font-weight:900; text-transform:uppercase;">{_esc(title)}</div>
              <div style="font-size:24px; color:#111827; font-weight:950; line-height:1.15; margin-top:7px;">{_esc(value)}</div>
              <div style="font-size:13px; color:#4b5563; line-height:1.38; margin-top:9px;">{_esc(note)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))

    if budget is not None and not budget.empty:
        source_count = len(budget)
        st.caption(f"This answer uses {source_count} safety checks. Technical source details are hidden unless you open them.")


def _render_risk_repair_path(repair_board: pd.DataFrame, readiness: pd.DataFrame):
    if repair_board is None or repair_board.empty:
        st.info("No first-fix list is available yet.")
        return

    work = repair_board.copy()
    if "repair_rank" in work.columns:
        work["_rank"] = pd.to_numeric(work["repair_rank"], errors="coerce").fillna(999)
        work = work.sort_values("_rank", kind="stable")

    st.markdown("#### First names to make smaller or watch only")
    st.caption("These are not buy ideas. They are the first tickers the safety page wants checked.")

    readiness_map = {}
    if readiness is not None and not readiness.empty and "ticker" in readiness.columns:
        for _, row in readiness.iterrows():
            readiness_map[str(row.get("ticker", "")).upper()] = row

    parts = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin:8px 0 20px 0;">']
    for _, row in work.head(6).iterrows():
        ticker = _clean_display(row.get("ticker"), "")
        accent = _risk_accent(row.get("original_risk_unlock_status", row.get("primary_repair_action")))
        current_w = _risk_weight_text(row.get("current_weight_pct"))
        repair_w = _risk_weight_text(row.get("recommended_repair_weight_pct"))
        target_w = _risk_weight_text(row.get("risk_target_weight_pct"))
        repair_action = _risk_repair_action_plain(row.get("primary_repair_action"))
        route = _risk_route_plain(row.get("route_after_risk_repair"))
        option_route = _risk_route_plain(row.get("option_permission_after_repair"))
        blockers = _risk_blocker_plain(row.get("remaining_non_risk_blockers"), 150)
        trigger = _risk_code_plain(row.get("trigger_to_watch"), 130)
        ready_row = readiness_map.get(ticker.upper())
        clear_condition = ""
        if ready_row is not None:
            clear_condition = _risk_reopen_plain(ready_row.get("nearest_clear_condition"))
        if not clear_condition:
            clear_condition = "Hold the ticker small, then rerun the daily system."
        parts.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:9px; padding:15px 16px; min-height:330px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div>
                  <div style="font-size:25px; color:#111827; font-weight:950; line-height:1;">{_esc(ticker)}</div>
                  <div style="font-size:12px; color:#64748b; margin-top:4px;">{_esc(_clean_display(row.get("sector"), "No sector"))}</div>
                </div>
                <div style="font-size:12px; color:{accent}; font-weight:900; text-align:right;">{_esc(repair_action)}</div>
              </div>
              <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; border-top:1px solid #e5e7eb; margin-top:11px; padding-top:10px;">
                <div><div style="font-size:11px; color:#64748b;">Now</div><div style="font-size:16px; color:#111827; font-weight:900;">{_esc(current_w)}</div></div>
                <div><div style="font-size:11px; color:#64748b;">Safer size</div><div style="font-size:16px; color:#334155; font-weight:900;">{_esc(repair_w)}</div></div>
                <div><div style="font-size:11px; color:#64748b;">Max safe</div><div style="font-size:16px; color:{accent}; font-weight:900;">{_esc(target_w)}</div></div>
              </div>
              <div style="font-size:13px; color:#111827; line-height:1.42; margin-top:10px;"><b>After it is safer:</b> {_esc(route)}</div>
              <div style="font-size:13px; color:#111827; line-height:1.42; margin-top:7px;"><b>Options now:</b> {_esc(option_route)}</div>
              <div style="font-size:12px; color:#4b5563; line-height:1.42; margin-top:8px;"><b>Still missing:</b> {_esc(blockers)}</div>
              <div style="font-size:12px; color:#4b5563; line-height:1.42; margin-top:8px;"><b>Price level to watch:</b> {_esc(trigger)}</div>
              <div style="font-size:11px; color:#64748b; line-height:1.35; margin-top:8px;"><b>Can reopen when:</b> {_esc(clear_condition)}</div>
            </div>
            """
        )
    parts.append("</div>")
    _render_html("".join(parts))


def _render_risk_unlock_ladder(
    overview: dict,
    breaches: pd.DataFrame,
    queue: pd.DataFrame,
    repair_board: pd.DataFrame,
    readiness: pd.DataFrame,
):
    recommended_gross = _to_float(overview.get("recommended_gross_exposure"))
    normal_gross = _to_float(overview.get("normal_gross_exposure"))
    annual_vol = _to_float(overview.get("annual_vol_pct"))
    target_vol = _to_float(overview.get("target_vol_pct"))
    first_limit = _risk_main_problem(breaches)
    first_ticker = _risk_first_repair_ticker(repair_board, queue)

    steps = []
    if recommended_gross is not None and normal_gross is not None and recommended_gross < normal_gross:
        steps.append(("Step 1", "Use a smaller account size first", f"Stay near {recommended_gross * 100:.0f}% of normal size before hunting for new ideas."))
    if annual_vol is not None and target_vol is not None:
        steps.append(("Step 2", "Calm down the account movement", f"Current movement is {annual_vol:.1f}%; comfort line is {target_vol:.1f}%."))
    steps.append(("Step 3", f"Check {first_ticker} first", "Start with the first ticker above, then rerun the daily system."))
    steps.append(("Step 4", "Fix the biggest warning", f"First warning to explain: {first_limit}."))
    steps.append(("Step 5", "Only then reopen new ideas", "After this improves, go to Today, News, then Ideas for stock / call / put choices."))

    st.markdown("#### How this becomes open again")
    html = ['<div style="background:#f8fafc; border:1px solid #d1d5db; border-radius:9px; padding:14px 15px; margin:8px 0 18px 0;">']
    for idx, (kicker, title, note) in enumerate(steps):
        border = "border-top:1px solid #e5e7eb;" if idx else ""
        html.append(
            f"""
            <div style="{border} padding:11px 4px; display:grid; grid-template-columns:82px 1fr; gap:12px; align-items:start;">
              <div style="font-size:12px; color:#64748b; font-weight:900; text-transform:uppercase;">{_esc(kicker)}</div>
              <div>
                <div style="font-size:16px; color:#111827; font-weight:900;">{_esc(title)}</div>
                <div style="font-size:13px; color:#4b5563; line-height:1.45; margin-top:3px;">{_esc(note)}</div>
              </div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _risk_code_plain(value, max_len: int | None = 180) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "Total gross exposure": "Total portfolio size",
        "gross exposure": "account size",
        "gross": "account size",
        "Portfolio 1d VaR 95%": "Normal one-day loss estimate",
        "Portfolio 1d CVaR 95%": "Worse one-day loss estimate",
        "Portfolio 1d VaR": "Normal one-day loss estimate",
        "Portfolio 1d CVaR": "Worse one-day loss estimate",
        "VaR": "loss estimate",
        "CVaR": "worse-loss estimate",
        "1d": "one-day",
        "master:SIZE_DOWN": "portfolio says smaller size",
        "master:REDUCE_ONLY": "portfolio says reduce only",
        "single:SIZE_DOWN": "single-stock downside is too high",
        "single:REDUCE_ONLY": "single-stock downside blocks new buying",
        "earnings_gap:SIZE_DOWN": "earnings or gap risk is too high",
        "gap_down:SIZE_DOWN": "sudden drop risk is too high",
        "kelly:SIZE_DOWN": "signal proof does not justify this much size",
        "sector:SIZE_DOWN": "sector exposure is too crowded",
        "sector:REVIEW": "sector exposure needs review",
        "liquidity:SIZE_DOWN": "liquidity stress says use smaller size",
        "SIZE_DOWN_OR_REDUCE_ONLY": "use smaller size or reduce exposure",
        "RISK_REPAIR_REQUIRED": "risk repair needed",
        "MANDATORY_REPAIR_TO_RISK_TARGET": "must repair toward risk target",
        "SIZE_DOWN_TO_REPAIR_PATH": "use the repair size path",
        "MASTER_GROSS_70": "smaller-account safety scenario",
        "REDUCE_ONLY_LOCKED": "no new buying locked",
        "SIZE_DOWN_LOCKED": "smaller-size locked",
        "STILL_ABOVE_TICKER_RISK_TARGET": "still too large for this ticker",
        "NO_BULLISH_OPTION_RISK_NOT_REPAIRED": "no bullish option while risk is unrepaired",
        "NO_NEW_OPTION": "no option idea yet",
        "RISK_REPAIRED_FOR_MANUAL_REVIEW": "safe enough for human review",
        "WATCH_ONLY_RISK_STILL_LOCKED": "watch only while risk is locked",
        "RISK_REDUCTION_ONLY": "risk reduction only",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "put or hedge research only",
        "option_no_go_checks": "option not-ready checks",
        "IV/Greeks/Gamma": "option volatility, Greeks, and gamma",
        "spread/TCA": "trading-cost proof",
        "TCA": "trading cost",
        "REDUCE_ONLY": "no new buying",
        "SIZE_DOWN": "use smaller size",
        "NOT_IN_RISK_BOOK_REVIEW": "not in the risk book yet",
        "MISSING_DATA_REVIEW": "missing data review",
        "DATA_GAP": "missing data",
        "CLEAR": "clear",
        "REVIEW": "review",
        "NAN": "no data",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = _human_text(text, max_len=None)
    text = text.replace("risk target", "safe size")
    text = text.replace("repair", "make safer")
    text = text.replace("repaired", "made safer")
    text = text.replace(";", "; ").replace("  ", " ")
    text = re.sub(r"\bmaster\b", "portfolio", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsingle\b", "single-stock risk", text, flags=re.IGNORECASE)
    text = re.sub(r"\bkelly\b", "signal-size proof", text, flags=re.IGNORECASE)
    text = re.sub(r"\bearnings gap\b", "earnings or gap risk", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _risk_pct_any(value) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    if abs(num) <= 1.5:
        num *= 100
    return f"{num:.1f}%"


def _risk_loss_estimate_line(overview: dict, var_cvar: pd.DataFrame) -> str:
    var_pct = _to_float(overview.get("var_95_1d_pct"))
    cvar_pct = _to_float(overview.get("cvar_95_1d_pct"))
    var_dollars = _to_float(overview.get("var_95_1d_dollars"))
    cvar_dollars = _to_float(overview.get("cvar_95_1d_dollars"))
    if (var_pct is None or cvar_pct is None) and var_cvar is not None and not var_cvar.empty:
        row = var_cvar.iloc[0]
        var_pct = _to_float(row.get("var_95_1d"))
        cvar_pct = _to_float(row.get("cvar_95_1d"))
        var_dollars = _to_float(row.get("var_95_1d_dollars"))
        cvar_dollars = _to_float(row.get("cvar_95_1d_dollars"))
    if var_pct is None and cvar_pct is None:
        return "No one-day loss estimate is available yet."
    var_text = _risk_pct_any(var_pct)
    cvar_text = _risk_pct_any(cvar_pct)
    dollars = ""
    if var_dollars is not None and cvar_dollars is not None:
        dollars = f" About ${var_dollars:,.0f} to ${cvar_dollars:,.0f} on the model account."
    return f"Normal bad-day loss is around {var_text}; worse bad-day loss is around {cvar_text}.{dollars}"


def _risk_worst_macro_line(overview: dict, macro: pd.DataFrame) -> str:
    scenario = _clean_display(overview.get("worst_macro_scenario"), "")
    impact = _to_float(overview.get("worst_macro_impact_pct"))
    if (not scenario or scenario == "No data" or impact is None) and macro is not None and not macro.empty:
        work = macro.copy()
        if "conservative_portfolio_impact" in work.columns:
            work["_impact"] = pd.to_numeric(work["conservative_portfolio_impact"], errors="coerce")
            work = work.sort_values("_impact")
            row = work.iloc[0]
            scenario = _clean_display(row.get("scenario"), "worst macro case")
            impact = _to_float(row.get("conservative_portfolio_impact"))
    if impact is None:
        return "No macro stress estimate is available yet."
    return f"Worst macro case is {scenario}: estimated portfolio hit { _risk_pct_any(impact) }."


def _risk_crisis_line(overview: dict, crisis: pd.DataFrame) -> str:
    ratio = _to_float(overview.get("crisis_vol_increase_ratio"))
    action = _risk_status_plain(overview.get("crisis_action"))
    if ratio is None and crisis is not None and not crisis.empty:
        row = crisis.iloc[0]
        ratio = _to_float(row.get("vol_increase_ratio"))
        action = _risk_status_plain(row.get("stress_action"))
    if ratio is None:
        return "No crisis-correlation estimate is available yet."
    return f"If markets move together, portfolio movement could rise about {ratio:.1f}x. Current call: {action.lower()}."


def _risk_sector_line(sector: pd.DataFrame) -> str:
    if sector is None or sector.empty:
        return "No sector concentration file is available yet."
    work = sector.copy()
    sort_col = "cap_used_pct" if "cap_used_pct" in work.columns else "portfolio_weight_pct"
    work["_sector_sort"] = pd.to_numeric(work[sort_col], errors="coerce").fillna(0)
    row = work.sort_values("_sector_sort", ascending=False).iloc[0]
    sector_name = _clean_display(row.get("sector"), "top sector")
    weight = _to_float(row.get("portfolio_weight_pct"))
    cap_used = _to_float(row.get("cap_used_pct"))
    status = _risk_status_plain(row.get("cap_status"))
    weight_text = f"{weight:.1f}%" if weight is not None else "No data"
    cap_text = f"{cap_used:.0f}% of its cap" if cap_used is not None else "cap usage unknown"
    return f"{sector_name} is the biggest sector risk at {weight_text}, using {cap_text}. Status: {status.lower()}."


def _risk_factor_line(factor: pd.DataFrame, beta: pd.DataFrame) -> str:
    source = factor if factor is not None and not factor.empty else beta
    if source is None or source.empty:
        return "No factor exposure file is available yet."
    work = source.copy()
    if "portfolio_beta" in work.columns:
        work["_beta_abs"] = pd.to_numeric(work["portfolio_beta"], errors="coerce").abs().fillna(0)
        row = work.sort_values("_beta_abs", ascending=False).iloc[0]
        factor_name = _clean_display(_first_non_empty(row.get("factor"), row.get("proxy")), "top factor")
        beta_value = _to_float(row.get("portfolio_beta"))
        status = _risk_status_plain(row.get("status"))
        return f"Biggest factor bet is {factor_name}, beta about {beta_value:.2f}. Status: {status.lower()}." if beta_value is not None else f"Biggest factor bet is {factor_name}."
    return "Factor file exists, but beta columns are not available yet."


def _risk_unlock_steps(overview: dict, breaches: pd.DataFrame, queue: pd.DataFrame) -> list[str]:
    steps: list[str] = []
    recommended = _to_float(overview.get("recommended_gross_exposure"))
    normal = _to_float(overview.get("normal_gross_exposure"))
    if recommended is not None and normal is not None and recommended < normal:
        steps.append(f"Bring total paper exposure toward {recommended * 100:.0f}% instead of the normal {normal * 100:.0f}%.")
    if queue is not None and not queue.empty:
        work = queue.copy()
        if "risk_reduction_pct_of_current" in work.columns:
            work["_reduction"] = pd.to_numeric(work["risk_reduction_pct_of_current"], errors="coerce").fillna(0)
            work = work.sort_values("_reduction", ascending=False)
        names = ", ".join(work["ticker"].dropna().astype(str).head(4).tolist()) if "ticker" in work.columns else ""
        if names:
            steps.append(f"Resize the first risk names: {names}.")
    if breaches is not None and not breaches.empty:
        first = _risk_limit_plain_name(breaches.iloc[0].get("budget_item"))
        steps.append(f"Clear the first blocking limit: {first}.")
    steps.append("Rerun the daily system after size, news proof, or data repairs change.")
    return steps[:4]


def _render_risk_verdict_board(overview: dict, breaches: pd.DataFrame, queue: pd.DataFrame, var_cvar: pd.DataFrame, macro: pd.DataFrame, crisis: pd.DataFrame, sector: pd.DataFrame, factor: pd.DataFrame, beta: pd.DataFrame):
    master = str(overview.get("master_risk_action", "No data"))
    accent = _risk_accent(master)
    can_add = "No new risk" if any(x in master.upper() for x in ["REDUCE", "SIZE_DOWN", "BLOCK", "HARD", "CRITICAL"]) else "Risk not blocking"
    drivers = overview.get("top_risk_drivers", [])
    if isinstance(drivers, list):
        driver_text = "; ".join(_risk_code_plain(x, 80) for x in drivers[:4])
    else:
        driver_text = _risk_code_plain(drivers, 220)
    if not driver_text:
        driver_text = "No top risk driver list is available yet."

    unlock_items = "".join(
        f"<li style='margin-bottom:5px;'>{_esc(step)}</li>"
        for step in _risk_unlock_steps(overview, breaches, queue)
    )
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:7px solid {accent}; border-radius:10px; padding:18px 20px; margin:8px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">Risk Verdict Board</div>
          <div style="font-size:26px; color:#111827; font-weight:950; line-height:1.18; margin-top:6px;">{_esc(can_add)}: risk decides before ideas.</div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:10px;"><b>Why:</b> {_esc(driver_text)}</div>
          <div style="border-top:1px solid #e5e7eb; margin-top:12px; padding-top:10px;">
            <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">Unlock checklist</div>
            <ol style="font-size:13px; color:#111827; line-height:1.45; padding-left:18px; margin:8px 0 0 0;">{unlock_items}</ol>
          </div>
        </div>
        """
    )

    rows = [
        ("Bad-day loss", _risk_loss_estimate_line(overview, var_cvar), "#334155"),
        ("Macro shock", _risk_worst_macro_line(overview, macro), "#991b1b"),
        ("Crisis crowding", _risk_crisis_line(overview, crisis), "#991b1b"),
        ("Sector crowding", _risk_sector_line(sector), "#334155"),
        ("Factor exposure", _risk_factor_line(factor, beta), "#334155"),
        ("Single-stock risk", f"{len(queue):,} name(s) need smaller size or review." if queue is not None and not queue.empty else "No ticker risk queue is available.", "#991b1b" if queue is not None and not queue.empty else "#166534"),
    ]
    html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:8px 0 20px 0;">']
    for title, line, color in rows:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {color}; border-radius:8px; padding:13px 14px; min-height:150px;">
              <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">{_esc(title)}</div>
              <div style="font-size:14px; color:#111827; line-height:1.45; margin-top:8px;">{_esc(line)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_breach_cards(breaches: pd.DataFrame):
    if breaches.empty:
        st.success("No hard risk budget breach is active.")
        return

    work = breaches.copy()
    if "used_pct_display" in work.columns:
        work["_sort"] = pd.to_numeric(work["used_pct_display"], errors="coerce").fillna(0)
        work = work.sort_values("_sort", ascending=False)

    st.markdown("##### Risk limits to fix first")
    card_cols = st.columns(2)
    for idx, (_, row) in enumerate(work.head(6).iterrows()):
        status = _risk_human_action(row.get("status"))
        accent = _risk_accent(row.get("status"))
        used = _to_float(row.get("used_pct_display"))
        used_text = f"{used:.1f}% used" if used is not None else "No usage data"
        with card_cols[idx % 2]:
            _render_html(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 15px; min-height:178px; margin:0 0 14px 0;">
                  <div style="display:flex; justify-content:space-between; gap:12px;">
                    <div style="font-size:17px; font-weight:850; color:#111827; line-height:1.25;">{_esc(_clean_display(row.get("budget_item"), "Risk limit"))}</div>
                    <div style="font-size:12px; font-weight:850; color:{accent}; white-space:nowrap;">{_esc(status)}</div>
                  </div>
                  <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;">{_esc(used_text)} of the allowed risk budget.</div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:12px; color:#6b7280; line-height:1.4;">Next: {_esc(_clean_display(row.get("required_next_action"), "Reduce or review before adding risk."))}</div>
                  <div style="font-size:11px; color:#9ca3af; margin-top:7px;">Source: {_esc(_friendly_source_label(row.get("source_file")))}</div>
                </div>
                """
            )


def _render_ticker_risk_cards(queue: pd.DataFrame):
    if queue.empty:
        st.info("No ticker risk queue is available.")
        return

    work = queue.copy()
    if "risk_reduction_pct_of_current" in work.columns:
        work["_reduction"] = pd.to_numeric(work["risk_reduction_pct_of_current"], errors="coerce").fillna(0)
        work = work.sort_values("_reduction", ascending=False)

    st.markdown("##### Tickers to repair first")
    parts = ['<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; margin:8px 0 18px 0;">']
    for _, row in work.head(8).iterrows():
        action = _risk_human_action(row.get("final_risk_action"))
        accent = _risk_accent(row.get("final_risk_action"))
        current_w = _to_float(row.get("current_weight_pct"))
        target_w = _to_float(row.get("recommended_risk_weight_pct"))
        reduction = _to_float(row.get("risk_reduction_pct_of_current"))
        current_w_text = f"{current_w:.2f}%" if current_w is not None else "No data"
        target_w_text = f"{target_w:.2f}%" if target_w is not None else "No data"
        reduction_text = f"Cut about {reduction * 100:.0f}% of current risk" if reduction is not None else "Reduce to target risk"
        reason = _risk_plain_reason(row)
        parts.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {accent}; border-radius:8px; padding:13px 14px; min-height:232px;">
              <div style="display:flex; justify-content:space-between; gap:10px;">
                <div style="font-size:21px; font-weight:850; color:#111827;">{_esc(row.get("ticker"), "")}</div>
                <div style="font-size:12px; font-weight:850; color:{accent};">{_esc(action)}</div>
              </div>
              <div style="font-size:12px; color:#6b7280; margin-top:3px;">{_esc(row.get("sector"), "No sector")}</div>
              <div style="border-top:1px solid #e5e7eb; padding-top:8px; margin-top:9px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div><div style="font-size:11px; color:#6b7280;">Current</div><div style="font-size:17px; font-weight:850; color:#111827;">{_esc(current_w_text)}</div></div>
                <div><div style="font-size:11px; color:#6b7280;">Risk target</div><div style="font-size:17px; font-weight:850; color:{accent};">{_esc(target_w_text)}</div></div>
              </div>
              <div style="font-size:12px; color:#374151; line-height:1.38; margin-top:9px;">{_esc(reduction_text)}</div>
              <div style="font-size:11px; color:#6b7280; line-height:1.35; margin-top:8px;">{_esc(reason)}</div>
            </div>
            """
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

    cols = [c for c in [
        "ticker", "sector", "current_weight_pct", "recommended_risk_weight_pct",
        "risk_reduction_pct_of_current", "final_risk_action", "single_name_action",
        "earnings_gap_action", "kelly_status", "sector_status",
        "required_next_action", "reason_stack",
    ] if c in work.columns]
    with st.expander("Open full ticker risk queue", expanded=False):
        _show_status_table(work[cols] if cols else work, ["final_risk_action", "single_name_action", "earnings_gap_action", "kelly_status", "sector_status"], height=560)


def _render_risk_source_inventory(files: list[tuple[str, str]]):
    rows = []
    for fname, meaning in files:
        path = ROOT / fname
        rows.append({
            "File": fname,
            "Meaning": meaning,
            "Exists": "Yes" if path.exists() and path.stat().st_size > 10 else "No",
            "Updated": file_age_str(path),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _risk_desk_plain(value, max_len: int | None = 220) -> str:
    text = _human_text(value, max_len=None)
    replacements = {
        "RISK_REPAIR_REQUIRED": "risk repair needed",
        "MANDATORY_REPAIR_TO_RISK_TARGET": "must repair toward risk target",
        "SIZE_DOWN_TO_REPAIR_PATH": "use the repair size path",
        "MASTER_GROSS_70": "70% gross repair scenario",
        "REDUCE_ONLY_LOCKED": "no new buying locked",
        "SIZE_DOWN_LOCKED": "smaller-size locked",
        "STILL_ABOVE_TICKER_RISK_TARGET": "still above ticker risk target",
        "NO_BULLISH_OPTION_RISK_NOT_REPAIRED": "no bullish option while risk is unrepaired",
        "NO_NEW_OPTION": "no option idea yet",
        "RISK_REPAIRED_FOR_MANUAL_REVIEW": "risk repaired enough for manual review",
        "WATCH_ONLY_RISK_STILL_LOCKED": "watch only while risk is locked",
        "RISK_REDUCTION_ONLY": "risk reduction only",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "put or hedge research only",
        "option_no_go_checks": "option not-ready checks",
        "IV/Greeks/Gamma": "option volatility, Greeks, and gamma",
        "spread/TCA": "trading-cost proof",
        "REDUCE ONLY": "risk reduction only",
        "Reduce Only": "risk reduction only",
        "SIZE DOWN": "use smaller size",
        "Size Down": "use smaller size",
        "RISK REDUCTION ONLY": "risk reduction only",
        "Risk Reduction Only": "risk reduction only",
        "MANUAL SPREAD LIQUIDITY CHECK": "manual quote and liquidity check",
        "Manual Spread Liquidity Check": "manual quote and liquidity check",
        "NO NEW EXPOSURE": "no new exposure",
        "No New Exposure": "no new exposure",
        "TINY RESEARCH ONLY": "tiny research only",
        "Tiny Research Only": "tiny research only",
        "REVIEW REQUIRED": "needs review",
        "Review Required": "needs review",
        "REVIEW": "needs review",
        "Review": "needs review",
        "BLOCKER": "blocked",
        "Blocker": "blocked",
        "BLOCKED": "blocked",
        "Blocked": "blocked",
        "CLEAR": "clear",
        "Clear": "clear",
        "WARNING": "warning",
        "Warning": "warning",
        "CRITICAL": "critical",
        "Critical": "critical",
        "DATA GAP": "missing data",
        "Data Gap": "missing data",
        "MISSING DATA REVIEW": "missing data review",
        "Missing Data Review": "missing data review",
        "NO DATA": "no data yet",
        "No Data": "no data yet",
        "TCA": "trading cost",
        "ADV": "average daily trading value",
        "VWAP": "mid-day average-price",
        "P0": "Top priority",
        "P1": "Next priority",
        "P2": "Later priority",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text or "No data"


def _risk_desk_accent(value) -> str:
    text = str(value or "").upper()
    if any(x in text for x in ["REDUCE", "NO NEW", "BLOCK", "CRITICAL", "DATA_GAP"]):
        return "#991b1b"
    if any(x in text for x in ["TINY", "MANUAL", "REVIEW", "WARNING", "SIZE_DOWN"]):
        return "#334155"
    if any(x in text for x in ["CLEAR", "GOOD", "HIGH"]):
        return "#166534"
    return "#111827"


def _pct_text(value, digits: int = 1) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    return f"{num:.{digits}f}%"


def _bps_text(value, digits: int = 1) -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    return f"{num:.{digits}f} bps"


def _render_portfolio_optimizer_human_panel():
    state = safe_json(ROOT / "institutional_optimizer_state.json")
    construction_state = safe_json(ROOT / "portfolio_construction_state.json")
    opt = safe_csv(ROOT / "depth5_portfolio_optimizer_v2.csv")
    bridge = safe_csv(ROOT / "institutional_optimizer_bridge.csv")
    why_not_more = safe_csv(ROOT / "institutional_optimizer_why_not_more.csv")
    active_risk = safe_csv(ROOT / "institutional_optimizer_active_risk_budget.csv")
    constraints = safe_csv(ROOT / "institutional_optimizer_constraint_audit.csv")

    if not state and opt.empty and bridge.empty:
        return

    final_gross = state.get("final_gross_pct", construction_state.get("target_gross_pct"))
    cash = state.get("cash_reserve_pct", construction_state.get("cash_reserve_pct"))
    risk_gate_count = int(_to_float(state.get("risk_gate_dominates_count"), 0) or 0)
    constraint_flags = int(_to_float(state.get("constraint_flags", construction_state.get("constraint_flags", 0)), 0) or 0)
    score = _to_float(state.get("institutional_optimizer_score"), 0) or 0

    no_new = 0
    tiny = 0
    if not opt.empty and "portfolio_v2_decision" in opt.columns:
        decisions = opt["portfolio_v2_decision"].astype(str).str.lower()
        no_new = int(decisions.str.contains("no new", na=False).sum())
        tiny = int(decisions.str.contains("tiny", na=False).sum())

    if final_gross is not None and _to_float(final_gross, 0) < 15:
        answer = f"Optimizer says stay very small: about {_pct_text(final_gross)} research gross, with about {_pct_text(cash)} kept in cash."
    elif risk_gate_count:
        answer = "Optimizer is still mostly controlled by risk gates. Do not let math weights override the risk page."
    else:
        answer = "Optimizer has research weights, but every ticker still needs risk and execution checks."

    st.markdown("#### Portfolio Optimizer: allowed size")
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #111827; border-radius:9px; padding:16px 18px; margin:8px 0 15px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">Portfolio answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(answer)}</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.5; margin-top:9px;">This is not a buy list. It shows the maximum research size after risk, correlation, sector, signal, and execution checks.</div>
        </div>
        """
    )

    cols = st.columns(5)
    cards = [
        ("Final research gross", _pct_text(final_gross), "Total paper exposure allowed by optimizer.", _risk_desk_accent(state.get("overall_status"))),
        ("Cash reserve", _pct_text(cash), "High cash means risk is still defensive.", "#334155"),
        ("No-new-exposure names", str(no_new), "Names that should not be increased.", "#991b1b" if no_new else "#166534"),
        ("Tiny research names", str(tiny), "Only small research sizing, not conviction sizing.", "#334155"),
        ("Optimizer score", f"{score:.1f} / 100", f"{constraint_flags} constraint flags; {risk_gate_count} risk-gated names.", "#334155" if score < 80 else "#166534"),
    ]
    for col, (title, value, note, accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, accent)

    if not opt.empty:
        st.markdown("##### What the optimizer allows ticker by ticker")
        work = opt.copy()
        if "portfolio_v2_decision" in work.columns:
            work["_order"] = work["portfolio_v2_decision"].astype(str).str.lower().map(
                lambda x: 0 if "no new" in x else 1 if "tiny" in x else 2
            )
            work = work.sort_values(["_order", "confidence_0_100"], ascending=[True, False], kind="stable")
        for start in range(0, min(len(work), 8), 4):
            cols = st.columns(4)
            for col, (_, row) in zip(cols, work.iloc[start:start + 4].iterrows()):
                decision = _risk_desk_plain(row.get("portfolio_v2_decision"), 80)
                accent = _risk_desk_accent(row.get("portfolio_v2_decision"))
                with col:
                    _render_html(
                        f"""
                        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:13px 14px; min-height:265px; margin-bottom:12px;">
                          <div style="display:flex; justify-content:space-between; gap:8px;">
                            <div style="font-size:21px; color:#111827; font-weight:900;">{_esc(row.get("ticker"), "")}</div>
                            <div style="font-size:12px; color:{accent}; font-weight:850;">{_esc(decision)}</div>
                          </div>
                          <div style="font-size:12px; color:#6b7280; margin-top:3px;">{_esc(_risk_desk_plain(row.get("sector"), 80))} · {_esc(_risk_desk_plain(row.get("sleeve"), 80))}</div>
                          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px;">
                            <div><div style="font-size:11px; color:#6b7280;">Current</div><div style="font-size:16px; font-weight:850;">{_esc(_pct_text(row.get("current_weight_pct")))}</div></div>
                            <div><div style="font-size:11px; color:#6b7280;">Allowed</div><div style="font-size:16px; font-weight:850; color:{accent};">{_esc(_pct_text(row.get("final_weight_pct")))}</div></div>
                          </div>
                          <div style="font-size:12px; color:#374151; line-height:1.38; margin-top:9px;"><b>Why:</b> {_esc(_risk_desk_plain(row.get("why"), 175))}</div>
                          <div style="font-size:11px; color:#6b7280; line-height:1.35; margin-top:8px;"><b>Unlock:</b> {_esc(_risk_desk_plain(row.get("what_would_unlock"), 160))}</div>
                        </div>
                        """
                    )

    with st.expander("Open optimizer constraints and why-not-more detail", expanded=False):
        if not why_not_more.empty:
            cols = [c for c in ["ticker", "requested_weight_pct", "max_feasible_weight_pct", "final_weight_pct", "primary_reason_not_more", "what_would_allow_more"] if c in why_not_more.columns]
            _show_status_table(why_not_more[cols].head(40) if cols else why_not_more.head(40), [], height=360)
        if not active_risk.empty:
            st.markdown("Risk budget by bucket")
            cols = [c for c in ["budget_bucket", "current_pct", "limit_pct", "remaining_pct", "status", "note"] if c in active_risk.columns]
            _show_status_table(active_risk[cols] if cols else active_risk, ["status"], height=300)
        if not constraints.empty:
            st.markdown("Hard optimizer rules")
            cols = [c for c in ["constraint", "current_value", "limit_value", "status", "note"] if c in constraints.columns]
            _show_status_table(constraints[cols] if cols else constraints, ["status"], height=300)


def _render_execution_liquidity_human_panel():
    cost_state = safe_json(ROOT / "execution_cost_model_state.json")
    tca_state = safe_json(ROOT / "execution_tca_state.json")
    playbook_state = safe_json(ROOT / "execution_playbook_state.json")
    desk = safe_csv(ROOT / "depth5_execution_liquidity_desk.csv")
    board = safe_csv(ROOT / "execution_tca_decision_board.csv")
    cards = safe_csv(ROOT / "execution_tca_ticker_cards.csv")
    audit = safe_csv(ROOT / "execution_cost_constraint_audit.csv")
    repair = safe_csv(ROOT / "execution_spread_repair_queue.csv")

    if not cost_state and not tca_state and desk.empty and board.empty:
        return

    ready = int(_to_float(tca_state.get("execution_research_ready_count"), 0) or 0)
    manual = int(_to_float(tca_state.get("manual_spread_liquidity_check_count"), 0) or 0)
    reduce_only = int(_to_float(tca_state.get("risk_reduction_only_count"), 0) or 0)
    score = _to_float(tca_state.get("overall_execution_tca_score", cost_state.get("execution_cost_model_score")), 0) or 0
    base_cost = cost_state.get("weighted_base_cost_bps")
    stress_cost = cost_state.get("weighted_stress_cost_bps")
    avg_fill = playbook_state.get("average_expected_fill_rate_pct")

    if ready == 0:
        answer = "Execution is not ready for new exposure. Use risk-reduction or manual quote checks only."
    elif manual or reduce_only:
        answer = "Some routes may be researched, but manual spread/liquidity proof is still required first."
    else:
        answer = "Execution cost is not blocking the research route, but no live order path exists."

    st.markdown("#### Execution & Liquidity: can this be modeled?")
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {_risk_desk_accent(tca_state.get("status"))}; border-radius:9px; padding:16px 18px; margin:8px 0 15px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">Execution answer</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(answer)}</div>
          <div style="font-size:13px; color:#4b5563; line-height:1.5; margin-top:9px;">This estimates paper feasibility: spread, liquidity, likely cost, fill quality, and whether the route is still blocked. It never sends orders.</div>
        </div>
        """
    )

    cols = st.columns(5)
    summary_cards = [
        ("Ready routes", str(ready), "Routes that can pass execution research now.", "#166534" if ready else "#991b1b"),
        ("Manual quote checks", str(manual), "Need current spread and liquidity proof.", "#334155" if manual else "#166534"),
        ("Risk-reduction only", str(reduce_only), "Names that cannot be new buys.", "#991b1b" if reduce_only else "#166534"),
        ("Base / stress cost", f"{_bps_text(base_cost)} / {_bps_text(stress_cost)}", "Expected cost can jump in stress.", "#334155"),
        ("Execution score", f"{score:.1f} / 100", f"Average fill estimate: {_pct_text(avg_fill)}", "#991b1b" if score < 25 else "#334155"),
    ]
    for col, (title, value, note, accent) in zip(cols, summary_cards):
        with col:
            _simple_card(title, value, note, accent)

    source = cards if not cards.empty else board
    if not source.empty:
        st.markdown("##### Execution cards to check first")
        work = source.copy()
        if "score" in work.columns:
            work["_score"] = pd.to_numeric(work["score"], errors="coerce").fillna(100)
            work = work.sort_values("_score", ascending=True)
        elif "execution_score_0_100" in work.columns:
            work["_score"] = pd.to_numeric(work["execution_score_0_100"], errors="coerce").fillna(100)
            work = work.sort_values("_score", ascending=True)
        for start in range(0, min(len(work), 8), 4):
            cols = st.columns(4)
            for col, (_, row) in zip(cols, work.iloc[start:start + 4].iterrows()):
                status = _risk_desk_plain(row.get("card_status", row.get("execution_verdict")), 90)
                accent = _risk_desk_accent(row.get("card_status", row.get("execution_verdict")))
                headline = _risk_desk_plain(row.get("headline", row.get("plain_reason")), 210)
                cost_line = _risk_desk_plain(row.get("cost_line", f"Base {_bps_text(row.get('base_cost_bps'))}; stress {_bps_text(row.get('stress_cost_bps'))}"), 170)
                blocker = _risk_desk_plain(row.get("blocker_line", row.get("primary_blockers")), 190)
                manual_check = _risk_desk_plain(row.get("manual_check", row.get("next_manual_check")), 170)
                with col:
                    _render_html(
                        f"""
                        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:13px 14px; min-height:280px; margin-bottom:12px;">
                          <div style="display:flex; justify-content:space-between; gap:8px;">
                            <div style="font-size:21px; color:#111827; font-weight:900;">{_esc(row.get("ticker"), "")}</div>
                            <div style="font-size:12px; color:{accent}; font-weight:850;">{_esc(status)}</div>
                          </div>
                          <div style="font-size:13px; color:#111827; font-weight:850; line-height:1.35; margin-top:8px;">{_esc(headline)}</div>
                          <div style="border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px; font-size:12px; color:#374151; line-height:1.38;"><b>Cost:</b> {_esc(cost_line)}</div>
                          <div style="font-size:12px; color:#374151; line-height:1.38; margin-top:8px;"><b>Blocker:</b> {_esc(blocker)}</div>
                          <div style="font-size:11px; color:#6b7280; line-height:1.35; margin-top:8px;"><b>Proof needed:</b> {_esc(manual_check)}</div>
                        </div>
                        """
                    )

    with st.expander("Open execution constraints and repair queue", expanded=False):
        if not audit.empty:
            st.markdown("Execution controls")
            cols = [c for c in ["control", "current_value", "target_or_limit", "status", "evidence"] if c in audit.columns]
            _show_status_table(audit[cols] if cols else audit, ["status"], height=300)
        if not repair.empty:
            st.markdown("Repair queue")
            cols = [c for c in ["priority", "ticker", "repair_type", "current_block", "cost_read", "proof_to_collect", "done_when"] if c in repair.columns]
            _show_status_table(repair[cols].head(40) if cols else repair.head(40), ["priority"], height=360)
        if not desk.empty:
            st.markdown("Full execution/liquidity desk")
            cols = [c for c in ["ticker", "execution_permission", "base_cost_bps", "stress_cost_bps", "expected_fill_rate_pct", "liquidity_read", "monitor_status", "what_to_do"] if c in desk.columns]
            _show_status_table(desk[cols] if cols else desk, ["execution_permission", "monitor_status"], height=420)


def tab_risk_portfolio():
    overview = safe_json(ROOT / "risk_desk_overview.json")
    breaches = safe_csv(ROOT / "risk_desk_breach_table.csv")
    queue = safe_csv(ROOT / "risk_desk_ticker_action_queue.csv")
    budget = safe_csv(ROOT / "institutional_risk_budget_summary.csv")
    var_cvar = safe_csv(ROOT / "portfolio_var_cvar_summary.csv")
    final_gate = safe_csv(ROOT / "final_risk_gate.csv")
    sector = safe_csv(ROOT / "sector_active_exposure.csv")
    factor = safe_csv(ROOT / "factor_exposure_decomposition.csv")
    beta = safe_csv(ROOT / "portfolio_beta_report.csv")
    macro = safe_csv(ROOT / "macro_scenario_stress.csv")
    crisis = safe_csv(ROOT / "crisis_correlation_stress.csv")
    corr = safe_csv(ROOT / "holdings_correlation_matrix.csv")
    dd = safe_json(ROOT / "drawdown_control_state.json")
    vol = safe_json(ROOT / "vol_target_state.json")
    kelly = safe_csv(ROOT / "kelly_position_sizing.csv")
    optimizer = safe_csv(ROOT / "institutional_optimizer_bridge.csv")
    active_risk = safe_csv(ROOT / "institutional_optimizer_active_risk_budget.csv")
    constraints = safe_csv(ROOT / "portfolio_constraint_matrix.csv")
    construction = safe_csv(ROOT / "institutional_portfolio_construction_plan.csv")
    repair_board = safe_csv(ROOT / "risk_repair_recommendation_board.csv")
    readiness = safe_csv(ROOT / "action_readiness_monitor.csv")

    master_action = str(overview.get("master_risk_action", "No data"))

    # ── Risk status hero banner ────────────────────────────────────────────────
    _r_action   = str(overview.get("master_risk_action","—"))
    _r_mult     = _to_float(overview.get("master_exposure_multiplier"))
    _r_rec_exp  = _to_float(overview.get("recommended_gross_exposure"))
    _r_norm_exp = _to_float(overview.get("normal_gross_exposure"))
    _r_hard     = int(_to_float(overview.get("budget_hard_breach_count"),0) or 0)
    _r_size     = int(_to_float(overview.get("budget_size_down_count"),0) or 0)
    _r_cvar     = _to_float(overview.get("cvar_95_1d_pct"))
    _r_dd       = _to_float(overview.get("drawdown_pct"))
    _r_answer   = _risk_front_answer(_r_action, _r_rec_exp, _r_norm_exp)

    _is_red     = any(x in _r_action.upper() for x in ("SIZE_DOWN","REDUCE","BLOCK","HARD"))
    _risk_bg    = "linear-gradient(135deg,#450a0a 0%,#7f1d1d 100%)" if _is_red else "linear-gradient(135deg,#052e16 0%,#14532d 100%)"
    _rec_pct    = f"{_r_rec_exp*100:.0f}%" if _r_rec_exp else "—"
    _norm_pct   = f"{_r_norm_exp*100:.0f}%" if _r_norm_exp else "—"
    _cvar_str   = f"{_r_cvar:.2f}%" if _r_cvar else "—"
    _dd_str     = f"{_r_dd:.1f}%" if _r_dd else "—"
    _dd_color   = "#fca5a5" if _r_dd and _r_dd < -5 else "#86efac" if _r_dd and _r_dd > -2 else "#fde68a"
    _action_label = {"SIZE_DOWN":"Use smaller size","REDUCE_ONLY":"Reduce only","CLEAR":"OK to add","BLOCK":"Blocked"}.get(_r_action, _r_action)

    _render_html(
        f"""
        <div style="
            background:{_risk_bg};
            border-radius:14px;
            padding:22px 26px 18px 26px;
            margin:10px 0 20px 0;
            box-shadow:0 4px 20px rgba(0,0,0,.2);
            position:relative;
            overflow:hidden;
        ">
          <div style="position:absolute;top:0;right:0;width:180px;height:180px;background:radial-gradient(circle at 100% 0%,rgba(255,255,255,.06) 0%,transparent 70%);pointer-events:none;"></div>
          <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">Risk answer</div>
          <div style="font-size:1.75rem;color:#f8fafc;font-weight:900;line-height:1.2;margin-top:8px;letter-spacing:-0.3px;">{_esc(_r_answer)}</div>
          <div style="height:1px;background:rgba(255,255,255,.08);margin:14px 0 12px 0;"></div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
            <div>
              <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Action</div>
              <div style="font-size:14px;font-weight:800;color:#fca5a5;margin-top:3px;">{_esc(_action_label)}</div>
            </div>
            <div>
              <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Allowed size</div>
              <div style="font-size:14px;font-weight:800;color:#fde68a;margin-top:3px;">{_rec_pct} <span style="font-size:10px;font-weight:600;opacity:.7;">of {_norm_pct} normal</span></div>
            </div>
            <div>
              <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Hard limits hit</div>
              <div style="font-size:18px;font-weight:900;color:{'#fca5a5' if _r_hard else '#86efac'};margin-top:2px;">{_r_hard}</div>
            </div>
            <div>
              <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Size-down items</div>
              <div style="font-size:18px;font-weight:900;color:{'#fde68a' if _r_size else '#86efac'};margin-top:2px;">{_r_size}</div>
            </div>
            <div>
              <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">1-day bad-day loss</div>
              <div style="font-size:14px;font-weight:800;color:#e2e8f0;margin-top:3px;">{_cvar_str}</div>
            </div>
            <div>
              <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Drawdown from peak</div>
              <div style="font-size:14px;font-weight:800;color:{_dd_color};margin-top:3px;">{_dd_str}</div>
            </div>
          </div>
          <div style="margin-top:12px;font-size:11px;color:#334155;">No broker connection · No live orders · Risk overrides options</div>
        </div>
        """
    )

    _render_risk_command_center(overview, breaches, queue, budget, var_cvar, repair_board)
    _render_risk_repair_path(repair_board, readiness)
    _render_risk_unlock_ladder(overview, breaches, queue, repair_board, readiness)

    with st.expander("Open extra safety explanation cards", expanded=False):
        _render_risk_verdict_board(overview, breaches, queue, var_cvar, macro, crisis, sector, factor, beta)
        _render_risk_human_limits(breaches)
        _render_risk_human_tickers(queue)

    with st.expander("Open sizing and trading-cost details", expanded=False):
        _render_portfolio_optimizer_human_panel()
        _render_execution_liquidity_human_panel()

    st.markdown("#### After this")
    _render_html(
        """
        <div style="background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:15px 16px; margin:8px 0 18px 0;">
          <div style="font-size:15px; color:#111827; line-height:1.55;">
            When the main risk answer no longer says "No," go to <b>Today</b> for the workflow, then <b>News</b> for source proof, then <b>Ideas</b> for the stock / call / put / wait route.
          </div>
        </div>
        """
    )

    show_detail = st.checkbox("Show technical safety tables", value=False)
    if not show_detail:
        return

    st.markdown("---")
    detail_view = st.radio(
        "Technical detail to open",
        ["Limits", "Stocks", "Exposure", "Stress", "Portfolio Builder", "Source Files"],
        horizontal=True,
        label_visibility="collapsed",
    )
    _render_subtab_depth("Risk", detail_view)

    if detail_view == "Limits":
        _render_breach_cards(breaches)
        if not budget.empty:
            st.markdown("##### All risk budgets")
            cols = [c for c in ["scope", "budget_item", "current_value", "limit_value", "used_pct", "status", "action_if_breached", "source_file"] if c in budget.columns]
            _show_status_table(budget[cols] if cols else budget, ["status"], height=420)
        if not var_cvar.empty:
            st.markdown("##### Portfolio loss estimate")
            cols = [c for c in [
                "date", "gross_exposure", "annual_vol", "var_95_1d", "cvar_95_1d",
                "var_99_1d", "cvar_99_1d", "var_95_20d", "cvar_95_20d",
                "var_95_1d_dollars", "cvar_95_1d_dollars", "status",
            ] if c in var_cvar.columns]
            _show_status_table(var_cvar[cols] if cols else var_cvar, ["status"], height=180)

    elif detail_view == "Stocks":
        _render_ticker_risk_cards(queue)
        if not final_gate.empty:
            st.markdown("##### Final risk check by stock")
            cols = [c for c in [
                "ticker", "sector", "current_weight_pct", "recommended_risk_weight_pct",
                "master_risk_action", "single_name_action", "earnings_gap_action",
                "kelly_status", "liquidity_crisis_status", "sector_status",
                "final_risk_action", "reason_stack",
            ] if c in final_gate.columns]
            _show_status_table(final_gate[cols] if cols else final_gate, [
                "master_risk_action", "single_name_action", "earnings_gap_action", "kelly_status", "sector_status", "final_risk_action",
            ], height=520)

    elif detail_view == "Exposure":
        st.markdown("##### Sector exposure")
        sector_cols = [c for c in ["sector", "portfolio_weight_pct", "benchmark_weight_pct", "active_weight_pct", "cap_used_pct", "cap_status", "top_tickers"] if c in sector.columns]
        _show_status_table(sector[sector_cols] if sector_cols else sector, ["cap_status"], height=340)
        st.markdown("##### Market driver exposure")
        left, right = st.columns(2)
        with left:
            factor_cols = [c for c in ["exposure_type", "factor", "proxy", "portfolio_beta", "estimated_20d_impact", "status", "source_file"] if c in factor.columns]
            _show_status_table(factor[factor_cols] if factor_cols else factor, ["status"], height=360)
        with right:
            beta_cols = [c for c in ["factor", "proxy", "portfolio_beta", "abs_beta", "beta_status", "source_file"] if c in beta.columns]
            _show_status_table(beta[beta_cols] if beta_cols else beta, ["beta_status"], height=360)
        if not corr.empty:
            st.markdown("##### How holdings move together")
            st.dataframe(_humanize_df(corr), width="stretch", hide_index=True)

    elif detail_view == "Stress":
        st.markdown("##### Bad-market tests")
        macro_cols = [c for c in ["scenario", "description", "conservative_portfolio_impact", "scenario_action", "missing_factor_betas", "source_file"] if c in macro.columns]
        _show_status_table(macro[macro_cols] if macro_cols else macro, ["scenario_action"], height=380)
        st.markdown("##### Drawdown, movement target, and crisis crowding")
        c1, c2, c3 = st.columns(3)
        with c1:
            _simple_card("Drawdown circuit", str(dd.get("circuit_level", "No data")), str(dd.get("reason", "No data")), _risk_accent(dd.get("drawdown_action")))
        with c2:
            _simple_card("Movement target", _risk_human_action(vol.get("vol_action", "No data")), str(vol.get("reason", "No data")), _risk_accent(vol.get("vol_action")))
        with c3:
            if not crisis.empty:
                cr = crisis.iloc[0]
                _simple_card("Crisis crowding", _risk_human_action(cr.get("stress_action")), f"Movement x{_to_float(cr.get('vol_increase_ratio'), 0):.2f}", _risk_accent(cr.get("stress_action")))
            else:
                _simple_card("Crisis crowding", "No data", "No crisis stress file.", "#334155")
        if not kelly.empty:
            st.markdown("##### Signal-based size")
            cols = [c for c in ["ticker", "current_weight_pct", "recommended_kelly_weight_pct", "kelly_status", "ic_periods", "ic_sample_confidence", "source_file"] if c in kelly.columns]
            _show_status_table(kelly[cols].head(40) if cols else kelly, ["kelly_status"], height=360)

    elif detail_view == "Portfolio Builder":
        st.markdown("##### Portfolio construction plan")
        if not construction.empty:
            cols = [c for c in [
                "ticker", "sector", "sleeve", "alpha_score", "current_weight_pct",
                "target_weight_pct", "master_action", "final_risk_action",
                "event_gate", "execution_status", "target_status", "reason",
            ] if c in construction.columns]
            _show_status_table(construction[cols] if cols else construction, ["master_action", "final_risk_action", "event_gate", "execution_status", "target_status"], height=520)
        if not optimizer.empty:
            st.markdown("##### Why the builder does not want more")
            cols = [c for c in [
                "ticker", "sector", "subsector_cycle_phase", "sleeve", "current_weight_pct",
                "math_optimizer_weight_pct", "risk_gated_target_pct", "final_optimizer_weight_pct",
                "final_optimizer_status", "binding_constraints", "why_not_more",
            ] if c in optimizer.columns]
            _show_status_table(optimizer[cols].head(60) if cols else optimizer, ["final_optimizer_status"], height=520)
        if not active_risk.empty or not constraints.empty:
            left, right = st.columns(2)
            with left:
                st.markdown("Risk budget by bucket")
                ar_cols = [c for c in ["budget_bucket", "current_pct", "limit_pct", "remaining_pct", "status", "note"] if c in active_risk.columns]
                _show_status_table(active_risk[ar_cols] if ar_cols else active_risk, ["status"], height=300)
            with right:
                st.markdown("Hard portfolio rules")
                co_cols = [c for c in ["constraint", "current_value", "limit_value", "status", "note"] if c in constraints.columns]
                _show_status_table(constraints[co_cols] if co_cols else constraints, ["status"], height=300)

    elif detail_view == "Source Files":
        _render_risk_source_inventory([
            ("risk_desk_overview.json", "Top risk answer"),
            ("risk_desk_breach_table.csv", "Risk limits to fix first"),
            ("risk_desk_ticker_action_queue.csv", "Ticker repair queue"),
            ("institutional_risk_budget_summary.csv", "Risk budget summary"),
            ("portfolio_var_cvar_summary.csv", "Portfolio loss estimates"),
            ("final_risk_gate.csv", "Stock-level final risk check"),
            ("sector_active_exposure.csv", "Sector exposure"),
            ("factor_exposure_decomposition.csv", "Market driver exposure"),
            ("portfolio_beta_report.csv", "Portfolio market sensitivity"),
            ("macro_scenario_stress.csv", "Bad-market tests"),
            ("holdings_correlation_matrix.csv", "How holdings move together"),
            ("crisis_correlation_stress.csv", "Crisis crowding stress"),
            ("drawdown_control_state.json", "Drawdown circuit state"),
            ("vol_target_state.json", "Movement target"),
            ("kelly_position_sizing.csv", "Signal-based size"),
            ("institutional_portfolio_construction_plan.csv", "Portfolio construction"),
            ("institutional_optimizer_bridge.csv", "Optimizer bridge"),
            ("portfolio_constraint_matrix.csv", "Portfolio rules"),
            ("risk_repair_recommendation_board.csv", "Plain-English repair path"),
            ("risk_repair_ticker_plan.csv", "Ticker repair scenarios"),
            ("action_readiness_monitor.csv", "What unlocks each ticker"),
            ("execution_tca_decision_board.csv", "Trading-cost decision board"),
            ("depth5_execution_liquidity_desk.csv", "Execution and liquidity desk"),
        ])


def _news_accent(tone: str = "", decision: str = "") -> str:
    text = f"{tone} {decision}".upper()
    if any(x in text for x in ["NEGATIVE", "DOWNSIDE", "HEDGE", "HURT", "RISK"]):
        return "#991b1b"
    if any(x in text for x in ["POSITIVE", "BENEFICIARY", "HELP", "CONFIRMATION"]):
        return "#0f766e"
    if any(x in text for x in ["MIXED", "WATCH", "REVIEW", "PROOF"]):
        return "#334155"
    return "#111827"


def _news_tone_label(tone: str) -> str:
    text = str(tone or "").upper()
    if "POSITIVE" in text:
        return "Good news"
    if "NEGATIVE" in text:
        return "Bad news / risk"
    if "MIXED" in text:
        return "Mixed"
    return "Unknown"


def _news_decision_label(decision: str) -> str:
    text = str(decision or "").upper()
    mapping = {
        "WATCH_FOR_CONFIRMATION": "Watch, but prove it first",
        "DOWNSIDE_WATCH_OR_HEDGE_RESEARCH": "Watch downside or hedge research",
        "RESEARCH_READY": "Research ready",
        "CONTEXT_ONLY": "Context only",
        "NO_ACTION": "No action",
        "RISK_BLOCKED": "Risk blocks action",
    }
    for key, label in mapping.items():
        if key in text:
            return label
    if not text or text in {"NAN", "NONE"}:
        return "No decision yet"
    return _plain_status(decision)


def _news_route_label(route: str, side: str = "", permission: str = "") -> str:
    text = f"{route} {side} {permission}".upper()
    if "NO_GO" in text or "NO_DIRECTIONAL" in text:
        return "No option idea yet"
    if "PUT" in text:
        return "Put research only"
    if "CALL" in text:
        return "Call research only"
    if "HEDGE" in text:
        return "Hedge research only"
    if "STOCK" in text or "PAPER" in text:
        return "Stock paper research"
    return "No option idea yet"


def _split_news_names(value, limit: int = 8) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    parts = [p.strip() for p in str(value).replace(";", ",").split(",")]
    names = [p for p in parts if p and p.lower() not in {"nan", "none"}]
    return names[:limit]


def _news_link_html(url: str) -> str:
    if not url or str(url).lower() in {"nan", "none"}:
        return ""
    safe_url = _esc(url)
    return f'<a href="{safe_url}" target="_blank" style="color:#111827; font-weight:800; text-decoration:underline;">Open original article</a>'


def _news_metric_value(value, suffix: str = "") -> str:
    num = _to_float(value)
    if num is None:
        return "No data"
    if float(num).is_integer():
        return f"{int(num):,}{suffix}"
    return f"{num:,.1f}{suffix}"


def _news_plain(value, max_len: int = 180) -> str:
    raw = "" if value is None else str(value)
    raw_replacements = {
        "WATCH_FOR_CONFIRMATION": "watch, but prove it first",
        "DOWNSIDE_WATCH_OR_HEDGE_RESEARCH": "watch downside or hedge research only",
        "CALL_RESEARCH_ONLY": "call research only",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "put or hedge research only",
        "PUT_REVIEW": "put review only",
        "CALL_REVIEW": "call review only",
        "THEME_PUT_OR_REDUCE_REVIEW": "theme risk review, no new buying",
        "THEME_WATCHLIST_PUT_REVIEW_AFTER_DATA": "watchlist; put review only after data is fixed",
        "THEME_WATCHLIST_CALL_REVIEW_AFTER_DATA": "watchlist; call review only after data is fixed",
        "THEME_RISK_WATCH": "theme risk watch",
        "AVOID_OR_REDUCE": "avoid new buying or reduce exposure",
        "WATCH_NEGATIVE_CONFIRMATION": "watch for downside confirmation",
        "NOT_IN_RISK_BOOK_REVIEW": "not in the risk book yet",
        "UNKNOWN_NEEDS_DATA": "unknown because data is missing",
        "EXTERNAL_THEME_TARGET_NEEDS_DATA": "outside theme target; needs data",
        "NEEDS_DATA": "needs data",
        "HAS_CONTRADICTIONS": "has price-reaction contradictions",
        "HAS_VALIDATED_EDGES": "has some validated links",
        "CONTRADICTED_REVIEW_REQUIRED": "contradiction; review required",
        "PRICE_DISAGREES": "price reaction disagrees",
        "P1_REVIEW_CONTRADICTION": "urgent contradiction review",
        "P2_VALIDATE": "validate before use",
        "UPSTREAM_SUPPLIER": "upstream supplier",
        "THEME_PEER": "theme peer",
        "VULNERABLE_TARGET": "vulnerable target",
        "BENEFICIARY": "possible winner",
        "REDUCE_ONLY": "no new buying",
        "SIZE_DOWN": "use smaller size",
        "DATA_GAP": "missing data",
        "NO_DIRECTIONAL": "no clear direction",
        "CONTEXT_ONLY": "context only",
        "event-time": "news-time",
        "price/volume": "price and volume",
        "spread/liquidity": "trading cost and volume",
        "risk-book": "risk book",
        "model-seen": "seen by the model",
        "read-through": "related-stock effect",
        "Read-through": "Related-stock effect",
        "causal": "story link",
        "Causal": "Story link",
    }
    for raw_text, friendly in raw_replacements.items():
        raw = raw.replace(raw_text, friendly)

    text = _human_text(raw, max_len=None)
    readable = {
        "Watch For Confirmation": "watch, but prove it first",
        "Downside Watch Or Hedge Research": "watch downside or hedge research only",
        "Call Research Only": "call research only",
        "Put Or Hedge Research Only": "put or hedge research only",
        "Not In Risk Book Review": "not in the risk book yet",
        "Unknown Needs Data": "unknown because data is missing",
        "External Theme Target Needs Data": "outside theme target; needs data",
        "Has Contradictions": "has price-reaction contradictions",
        "Has Validated Edges": "has some validated links",
        "Contradicted Review Required": "contradiction; review required",
        "Price Disagrees": "price reaction disagrees",
        "Theme Put Or Reduce Review": "theme risk review, no new buying",
        "Theme Watchlist Put Review After Data": "watchlist; put review only after data is fixed",
        "Theme Watchlist Call Review After Data": "watchlist; call review only after data is fixed",
        "Avoid Or Reduce": "avoid new buying or reduce exposure",
        "Watch Negative Confirmation": "watch for downside confirmation",
        "Reduce Only": "no new buying",
        "Size Down": "use smaller size",
        "Data Gap": "missing data",
        "No Directional": "no clear direction",
        "Context Only": "context only",
        "No data / No data": "No data",
    }
    for raw_text, friendly in readable.items():
        text = text.replace(raw_text, friendly)
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _news_event_rows(row, decision_board: pd.DataFrame) -> pd.DataFrame:
    event_id = row.get("event_id")
    if event_id is None or decision_board is None or decision_board.empty or "event_id" not in decision_board.columns:
        return pd.DataFrame()

    event_rows = decision_board[decision_board["event_id"].astype(str) == str(event_id)].copy()
    if not event_rows.empty and "event_score" in event_rows.columns:
        event_rows["_score"] = pd.to_numeric(event_rows["event_score"], errors="coerce").fillna(0)
        event_rows = event_rows.sort_values("_score", ascending=False)
    return event_rows


def _infer_news_theme(row) -> str:
    headline = str(row.get("headline", "") or "").lower()
    tickers = " ".join(
        str(row.get(k, "") or "")
        for k in ["source_news_ticker", "top_beneficiaries", "top_vulnerable_targets"]
    ).upper()
    blob = f"{headline} {tickers}"

    if any(x in blob for x in ["stablecoin", "wallet", "payment", "payments", "fintech", "CPAY", "FICO", "ADP", "BR"]):
        return "payments / financial services"
    if any(x in blob for x in ["data center", "ai ", "artificial intelligence", "networking", "gpu", "chip", "semiconductor", "NVDA", "AMD", "AVGO", "ASML", "KLAC", "LRCX", "AMAT", "SMCI"]):
        return "AI / Data Center"
    if any(x in blob for x in ["ssd", "storage", "hard drive", "controller", "STX", "WDC"]):
        return "storage hardware"
    if any(x in blob for x in ["software", "cloud", "saas", "autodesk", "ADSK", "DDOG", "NOW", "CRM"]):
        return "software"
    if any(x in blob for x in ["lawsuit", "settlement", "regulatory", "probe"]):
        return "legal / event risk"
    if any(x in blob for x in ["energy", "nuclear", "power", "utility", "utilities", "AEP", "CCJ", "URA"]):
        return "energy / utilities"
    if any(x in blob for x in ["space", "launch", "rocket", "satellite", "SPACEX", "RKLB"]):
        return "space / launch"
    if any(x in blob for x in ["defense", "security", "cyber", "aerospace"]):
        return "defense / security"
    return "the same industry"


def _news_theme_from_rows(event_rows: pd.DataFrame, row=None) -> str:
    if event_rows is None or event_rows.empty or "theme" not in event_rows.columns:
        return _infer_news_theme(row) if row is not None else "the same industry"
    values = [
        _clean_display(v, "")
        for v in event_rows["theme"].dropna().astype(str).tolist()
        if str(v).strip() and str(v).strip().lower() not in {"nan", "none"}
    ]
    if not values:
        return _infer_news_theme(row) if row is not None else "the same industry"
    try:
        return pd.Series(values).mode().iloc[0]
    except Exception:
        return values[0]


def _news_role_label(role: str) -> str:
    text = str(role or "").upper()
    if "VULNERABLE" in text:
        return "possible loser if the story pressures the group"
    if "UPSTREAM" in text:
        return "supplier or upstream company"
    if "DOWNSTREAM" in text:
        return "customer or downstream company"
    if "PEER" in text:
        return "related peer"
    if "BENEFICIARY" in text:
        return "direct possible winner"
    if "HEDGE" in text:
        return "possible hedge"
    return "related stock"


def _news_target_action_phrase(row) -> str:
    route = str(row.get("directional_route", "") or "").upper()
    decision = str(row.get("readthrough_decision", "") or "").upper()
    risk = str(row.get("final_risk_action", "") or "").upper()

    if "NO_DIRECTIONAL" in route or "CONTEXT_ONLY" in decision:
        action = "Use as context only, not a trade."
    elif "PUT" in route or "HEDGE" in route:
        action = "Only research a put or hedge after downside confirmation."
    elif "CALL" in route:
        action = "Only research a call after price and volume confirm."
    elif "WATCH" in decision:
        action = "Watch first; do not act from the headline alone."
    else:
        action = "Research only until the checks are complete."

    if any(x in risk for x in ["NOT_IN_RISK_BOOK", "UNKNOWN", "DATA", "BLOCK", "REVIEW"]):
        action += " Risk data is not complete yet."
    return action


def _news_stock_plain_line(row) -> str:
    ticker = _clean_display(row.get("target_ticker"), "Target")
    role = _news_role_label(_first_non_empty(row.get("target_role"), row.get("target_relation")))
    theme = _clean_display(row.get("theme"), "")
    action = _news_target_action_phrase(row)
    if theme and theme != "No data":
        return f"{ticker}: {role} in {theme}. {action}"
    return f"{ticker}: {role}. {action}"


def _news_proof_items(value) -> list[str]:
    text = _clean_display(value, "")
    lower = text.lower()
    items: list[str] = []
    if "causal" in lower or "event-time" in lower or "price reaction" in lower:
        items.append("Prove the headline really moved the stock after it was published.")
    if "model-seen" in lower or "samples" in lower:
        items.append("Collect more examples so the model is not trusting one headline too much.")
    if "risk-book" in lower or "risk book" in lower or "risk data" in lower:
        items.append("Fill the risk book before any paper position size is allowed.")
    if "liquidity" in lower or "spread" in lower:
        items.append("Check trading volume and bid-ask spread before trusting the idea.")
    if not items:
        items.append(_human_text(text or "Check source timing, price reaction, risk, and liquidity before trusting it.", max_len=170))
    return items[:4]


def _news_event_plain_read(row, event_rows: pd.DataFrame) -> str:
    source = _clean_display(row.get("source_news_ticker"), "the source company")
    theme = _news_theme_from_rows(event_rows, row)
    tone = str(row.get("market_tone", "") or "").upper()
    if "NEGATIVE" in tone:
        return (
            f"This is a risk headline around {source}. The system is checking whether {source} "
            f"or weaker related stocks in {theme} could fall further."
        )
    if "POSITIVE" in tone:
        return (
            f"This is a positive headline around {source}. The system is checking whether it only helps {source}, "
            f"or also lifts related stocks in {theme}."
        )
    return f"This headline may matter for {source} and related stocks in {theme}, but the direction is not clear yet."


def _news_event_next_step(row, event_rows: pd.DataFrame) -> str:
    decision = str(row.get("top_decision", "") or "").upper()
    proof = str(row.get("top_required_proof", "") or "").upper()
    risk_rows = ""
    if event_rows is not None and not event_rows.empty and "final_risk_action" in event_rows.columns:
        risk_rows = " ".join(event_rows["final_risk_action"].fillna("").astype(str).str.upper().head(12).tolist())

    if "DOWNSIDE" in decision or "HEDGE" in decision:
        answer = "Do not chase a rebound. First watch for downside confirmation; hedge or put research only."
    elif "WATCH" in decision or "CONFIRMATION" in decision:
        answer = "Do not buy from the headline alone. First check price, volume, and whether the move holds."
    elif "CONTEXT_ONLY" in decision:
        answer = "Use it as background context. No new stock or option idea is ready."
    else:
        answer = "Treat it as research only until the missing checks are finished."

    if any(x in f"{proof} {risk_rows}" for x in ["RISK", "LIQUIDITY", "SPREAD", "NOT_IN_RISK_BOOK", "UNKNOWN", "DATA"]):
        answer += " No paper size until risk, liquidity, and proof files are complete."
    return answer


def _news_help_hurt_lines(row, decision_board: pd.DataFrame) -> tuple[list[str], list[str]]:
    event_rows = _news_event_rows(row, decision_board)
    help_lines: list[str] = []
    hurt_lines: list[str] = []

    if not event_rows.empty:
        for _, item in event_rows.head(18).iterrows():
            tone = str(item.get("market_tone", "")).upper()
            role = str(_first_non_empty(item.get("target_role"), item.get("target_relation")) or "").upper()
            line = _news_stock_plain_line(item)
            if "NEGATIVE" in tone or "VULNERABLE" in role or "DOWNSIDE" in str(item.get("readthrough_decision", "")).upper():
                if len(hurt_lines) < 5:
                    hurt_lines.append(line)
            else:
                if len(help_lines) < 5:
                    help_lines.append(line)

    if not help_lines:
        for name in _split_news_names(row.get("top_beneficiaries"), 5):
            help_lines.append(f"{name}: possible winner. Still needs price, volume, risk, and liquidity proof.")

    if not hurt_lines:
        for name in _split_news_names(row.get("top_vulnerable_targets"), 5):
            hurt_lines.append(f"{name}: possible loser. Check valuation, weak trend, risk, and liquidity first.")

    if not help_lines:
        help_lines = ["No clear winner has been proven yet."]
    if not hurt_lines:
        hurt_lines = ["No clear loser has been proven yet."]

    return help_lines[:5], hurt_lines[:5]


def _news_card_story(row, decision_board: pd.DataFrame) -> dict:
    event_rows = _news_event_rows(row, decision_board)
    help_lines, hurt_lines = _news_help_hurt_lines(row, decision_board)
    return {
        "plain_read": _news_event_plain_read(row, event_rows),
        "next_step": _news_event_next_step(row, event_rows),
        "help_lines": [_human_text(item, max_len=165) for item in help_lines[:4]],
        "hurt_lines": [_human_text(item, max_len=165) for item in hurt_lines[:4]],
        "proof_items": _news_proof_items(row.get("top_required_proof", "")),
    }


def _news_bullet_html(items: list[str], empty_text: str) -> str:
    clean = [x for x in items if x and x != "No data"]
    if not clean:
        clean = [empty_text]
    return "<ul style='margin:6px 0 0 18px; padding:0;'>" + "".join(
        f"<li style='margin:4px 0;'>{_esc(item)}</li>" for item in clean[:4]
    ) + "</ul>"


def _news_cycle_plain(value) -> str:
    text = _clean_display(value, "No cycle read yet")
    raw = str(value or "").upper()
    if "LATE" in raw or "CHASE" in raw:
        return "Strong theme, but late-cycle chase risk. Do not chase without confirmation."
    if "EARLY" in raw or "IMPROVEMENT" in raw:
        return "Early improvement. Watch for follow-through before trusting it."
    if "HANDOFF" in raw or "CATCH" in raw:
        return "Possible leadership handoff. Compare it against the old leaders."
    if text == "No cycle read yet":
        return text
    return _human_text(text, max_len=150)


def _news_option_plain(route: str, decision: str = "") -> str:
    text = f"{route} {decision}".upper()
    if "PUT" in text or "HEDGE" in text:
        return "Put or hedge research only after downside is confirmed."
    if "CALL" in text:
        return "Call research only after price and volume confirm."
    if "NO_DIRECTIONAL" in text or "CONTEXT" in text:
        return "No stock or option idea yet; use it as context."
    if "WATCH" in text:
        return "Watch first. Do not turn the headline into a trade yet."
    return "Research only until the missing checks clear."


def _news_target_plain_read(row) -> str:
    ticker = _clean_display(row.get("target_ticker"), "This stock")
    role = _news_role_label(row.get("top_target_role"))
    pos = int(_to_float(row.get("positive_event_count"), 0) or 0)
    neg = int(_to_float(row.get("negative_event_count"), 0) or 0)
    tone = str(row.get("top_tone", "") or "").upper()

    if "NEGATIVE" in tone or neg > pos:
        return f"{ticker} is on the risk list because recent news may pressure it or its peer group."
    if pos > 0:
        return f"{ticker} is on the watchlist because {pos} positive news link(s) map it as a {role}."
    return f"{ticker} is mapped to news, but the direction is not strong enough yet."


def _news_target_now(row) -> str:
    ticker = _clean_display(row.get("target_ticker"), "This stock")
    decision = str(row.get("top_decision", "") or "").upper()
    risk = str(row.get("final_risk_action", "") or "").upper()
    action = _news_option_plain(row.get("directional_route", ""), decision)

    if any(x in risk for x in ["NOT_IN_RISK_BOOK", "UNKNOWN", "DATA", "BLOCK", "REVIEW"]):
        return f"{ticker} is not ready for sizing. {action} Risk data or liquidity proof is still missing."
    if "DOWNSIDE" in decision or "HEDGE" in decision:
        return f"Watch downside first. {action}"
    if "WATCH" in decision:
        return f"Watch confirmation first. {action}"
    return action


def _news_chain_plain_read(row) -> str:
    source = _clean_display(row.get("source_news_ticker"), "the source company")
    tone = str(row.get("market_tone", "") or "").upper()
    theme = _clean_display(row.get("themes"), _infer_news_theme(row))
    targets = _clean_display(row.get("top_targets"), "No mapped stocks yet")
    if "NEGATIVE" in tone:
        return f"Bad news around {source} may pressure related names in {theme}. Check these first: {targets}."
    return f"Good news around {source} may spread through {theme}. Check whether these names react too: {targets}."


def _news_chain_proof_line(row) -> str:
    validated = int(_to_float(row.get("validated_edge_count"), 0) or 0)
    contradicted = int(_to_float(row.get("contradicted_edge_count"), 0) or 0)
    confidence = _news_metric_value(row.get("avg_causal_confidence"))
    if contradicted > 0:
        return f"Confidence {confidence}/100. {validated} link(s) look supported, but {contradicted} disagree with price action."
    if validated > 0:
        return f"Confidence {confidence}/100. {validated} link(s) have supporting evidence so far."
    return f"Confidence {confidence}/100. This is still a hypothesis and needs evidence."


def _news_validation_plain(row) -> str:
    ticker = _clean_display(row.get("target_ticker"), "Target")
    issue = str(row.get("issue", "") or "").upper()
    note = _clean_display(row.get("validation_note"), "")
    if "PRICE_DISAGREES" in issue or "CONTRADICTION" in issue:
        return f"{ticker}: the story and price reaction disagree. Do not trust the news link until this is explained."
    if "PRICE_VALIDATION" in issue or "PRICE" in issue:
        return f"{ticker}: check whether price and volume actually moved after the headline."
    if "EVENT_TIME" in issue or "LOOKAHEAD" in issue:
        return f"{ticker}: verify the news timestamp so the model is not using information from the future."
    if note and note != "No data":
        return f"{ticker}: {_human_text(note, max_len=180)}"
    return f"{ticker}: this news link needs manual proof before it can affect any idea."


def _news_ticker_plain_read(row) -> str:
    ticker = _clean_display(row.get("ticker"), "This ticker")
    direction = _clean_display(row.get("news_direction"), "Mapped news").lower()
    source = _clean_display(row.get("source_news_ticker"), "the source company")
    theme = _clean_display(row.get("theme"), _infer_news_theme(row))
    if "bear" in direction or "down" in direction:
        return f"{ticker} may be hurt by news around {source} or weaker names in {theme}."
    if "bull" in direction or "up" in direction:
        return f"{ticker} may benefit from news around {source} or related names in {theme}."
    return f"{ticker} is linked to this headline, but the direction still needs proof."


def _news_ticker_next_step(row) -> str:
    action = _clean_display(row.get("action_hint"), "Confirm price, volume, risk, and position size before any paper idea.")
    vulnerability = _clean_display(row.get("negative_vulnerability_summary"), "")
    out = _human_text(action, max_len=175)
    if vulnerability and vulnerability != "No data":
        out += f" Weakness check: {_human_text(vulnerability, max_len=110)}."
    return out


def _news_names_sentence(value, empty: str, limit: int = 6) -> str:
    names = _split_news_names(value, limit)
    return ", ".join(names) if names else empty


def _news_use_permission_text(row) -> str:
    decision = str(row.get("top_decision", "") or row.get("readthrough_decision", "") or "").upper()
    proof = str(row.get("top_required_proof", "") or row.get("proof_required", "") or "").upper()
    if any(x in decision for x in ["RISK_BLOCKED", "NO_ACTION", "CONTEXT_ONLY"]):
        return "Use as context only. Do not turn this into a trade idea yet."
    if any(x in decision for x in ["DOWNSIDE", "HEDGE"]):
        return "Downside or hedge research only, and only after price confirms."
    if any(x in decision for x in ["WATCH", "CONFIRMATION", "REVIEW"]):
        return "Watch first. It needs price, volume, risk, and source proof."
    if any(x in proof for x in ["RISK", "LIQUIDITY", "SPREAD", "EVENT", "CAUSAL", "PRICE"]):
        return "Research only. The proof checklist is not finished."
    return "Research-ready context, but still not a live order."


def _news_related_rows(row, table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    event_id = row.get("event_id")
    headline = str(row.get("headline", "") or "").strip()
    work = table.copy()
    if event_id is not None and "event_id" in work.columns:
        out = work[work["event_id"].astype(str) == str(event_id)].copy()
        if not out.empty:
            return out
    if headline and "headline" in work.columns:
        return work[work["headline"].astype(str).str.strip() == headline].copy()
    return pd.DataFrame()


def _news_sort_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    work = rows.copy()
    score_cols = [
        "event_score",
        "impact_score",
        "causal_confidence_score",
        "best_event_score",
        "total_vulnerability",
    ]
    found = [c for c in score_cols if c in work.columns]
    if found:
        work["_news_sort_score"] = 0.0
        for col in found:
            work["_news_sort_score"] += pd.to_numeric(work[col], errors="coerce").fillna(0).abs()
        work = work.sort_values("_news_sort_score", ascending=False)
    return work


def _news_target_line(row) -> str:
    ticker = _clean_display(_first_non_empty(row.get("target_ticker"), row.get("ticker")), "Target")
    role = _news_role_label(_first_non_empty(row.get("target_role"), row.get("target_relation"), row.get("chain_role")))
    theme = _clean_display(row.get("theme"), "")
    reason = _first_non_empty(row.get("why_this_target"), row.get("target_reason"), row.get("causal_thesis"), row.get("news_logic"))
    reason_text = _news_plain(reason, max_len=118) if reason is not None else "Needs proof before trust."
    if theme and theme != "No data":
        return f"{ticker}: {role} in {_news_plain(theme, 55)}. {reason_text}"
    return f"{ticker}: {role}. {reason_text}"


def _news_target_list_html(rows: pd.DataFrame, empty_text: str, kind: str, limit: int = 5) -> str:
    if rows is None or rows.empty:
        return f"<div style='font-size:12px; color:#6b7280; line-height:1.4;'>{_esc(empty_text)}</div>"
    work = _news_sort_rows(rows).head(limit)
    color = "#0f766e" if kind == "help" else "#991b1b" if kind == "hurt" else "#334155"
    items = []
    for _, item in work.iterrows():
        items.append(
            f"""
            <div style="border-left:3px solid {color}; padding-left:8px; margin:7px 0; font-size:12px; color:#374151; line-height:1.38;">
              {_esc(_news_target_line(item))}
            </div>
            """
        )
    return "".join(items)


def _news_good_bad_rows(event_row, board: pd.DataFrame, impact_targets: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    board_rows = _news_related_rows(event_row, board)
    impact_rows = _news_related_rows(event_row, impact_targets)
    edge_rows = _news_related_rows(event_row, edges)
    all_rows = pd.concat([x for x in [board_rows, impact_rows, edge_rows] if x is not None and not x.empty], ignore_index=True) if any(
        x is not None and not x.empty for x in [board_rows, impact_rows, edge_rows]
    ) else pd.DataFrame()
    if all_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    row_text = all_rows.apply(
        lambda r: " ".join(
            str(r.get(c, "") or "").upper()
            for c in ["market_tone", "target_role", "target_relation", "readthrough_decision", "directional_route", "suggested_research_route", "news_logic"]
        ),
        axis=1,
    )
    hurt_mask = row_text.str.contains("NEGATIVE|VULNERABLE|DOWNSIDE|HEDGE|AVOID|REDUCE|PRESSURE|RISK", regex=True, na=False)
    help_mask = row_text.str.contains("POSITIVE|BENEFICIARY|UPSTREAM|DOWNSTREAM|PEER|CALL|STOCK_OR_CALL|LIFT|BULLISH", regex=True, na=False) & ~hurt_mask
    good = all_rows[help_mask].copy()
    bad = all_rows[hurt_mask].copy()

    if good.empty and "top_beneficiaries" in event_row.index:
        rows = [{"target_ticker": name, "target_role": "BENEFICIARY", "theme": _infer_news_theme(event_row), "target_reason": "Possible winner from the headline; still needs proof."} for name in _split_news_names(event_row.get("top_beneficiaries"), 6)]
        good = pd.DataFrame(rows)
    if bad.empty and "top_vulnerable_targets" in event_row.index:
        rows = [{"target_ticker": name, "target_role": "VULNERABLE_TARGET", "theme": _infer_news_theme(event_row), "target_reason": "Possible loser from the headline; still needs proof."} for name in _split_news_names(event_row.get("top_vulnerable_targets"), 6)]
        bad = pd.DataFrame(rows)
    return good, bad


def _render_news_command_center(
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    validation_queue: pd.DataFrame,
    chain_map: pd.DataFrame,
    impact_state: dict,
    chain_state: dict,
):
    event_count = len(summary) if summary is not None and not summary.empty else 0
    target_count = len(ranking) if ranking is not None and not ranking.empty else 0
    proof_count = len(validation_queue) if validation_queue is not None and not validation_queue.empty else 0
    contradicted_edges = int(_to_float(chain_state.get("contradicted_edge_count"), 0) or 0)
    validated_edges = int(_to_float(chain_state.get("validated_edge_count"), 0) or 0)

    top_headline = "Run the daily system first."
    top_source = "No source yet"
    top_theme = "No theme yet"
    if summary is not None and not summary.empty:
        work = summary.copy()
        if "best_event_score" in work.columns:
            work["_score"] = pd.to_numeric(work["best_event_score"], errors="coerce").fillna(0)
            work = work.sort_values("_score", ascending=False)
        top_row = work.iloc[0]
        top_headline = _news_plain(top_row.get("headline"), 165)
        top_source = f"{_news_plain(top_row.get('source_news_ticker'), 45)} · {_news_plain(top_row.get('publisher'), 70)}"

    if chain_map is not None and not chain_map.empty:
        chain_work = chain_map.copy()
        if "avg_causal_confidence" in chain_work.columns:
            chain_work["_confidence"] = pd.to_numeric(chain_work["avg_causal_confidence"], errors="coerce").fillna(0)
            chain_work = chain_work.sort_values("_confidence", ascending=False)
        top_theme = _news_plain(chain_work.iloc[0].get("themes"), 90)
    themes = impact_state.get("themes_triggered", [])
    if isinstance(themes, list) and themes:
        top_theme = _news_plain(", ".join(str(x) for x in themes[:4]), 110)

    if proof_count or contradicted_edges:
        answer = "Use news as a map, not a signal. Prove the source, timing, price reaction, risk, and trading cost first."
        accent = "#334155"
    elif target_count:
        answer = "News has mapped affected stocks. Read the top story, then verify the linked names before studying any route."
        accent = "#0f766e"
    else:
        answer = "No usable news map is ready yet. Run the daily system before using this page."
        accent = "#991b1b"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:7px solid {accent}; border-radius:10px; padding:21px 23px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">Simple news answer</div>
          <div style="font-size:30px; color:#111827; font-weight:950; line-height:1.16; margin-top:7px;">{_esc(answer)}</div>
          <div style="font-size:15px; color:#374151; line-height:1.48; margin-top:12px;"><b>Read first:</b> {_esc(top_headline)}</div>
          <div style="font-size:13px; color:#6b7280; line-height:1.4; margin-top:7px;">Source: {_esc(top_source)}. Main chain: {_esc(top_theme)}. Research-only; no broker connection; no live orders.</div>
        </div>
        """
    )

    cards = [
        ("Headlines", f"{event_count:,}", "Stories collected and ranked.", "#111827"),
        ("Affected stocks", f"{target_count:,}", "Possible winners, losers, peers, suppliers, and customers.", "#334155"),
        ("Missing proof", f"{proof_count:,}", "Need source time, price reaction, or story-link proof.", "#991b1b" if proof_count else "#166534"),
        ("Price disagreements", f"{contradicted_edges:,}", f"{validated_edges:,} links have some support.", "#991b1b" if contradicted_edges else "#0f766e"),
    ]
    cols = st.columns(4)
    for col, (title, value, note, color) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, color)


def _news_chain_sentence(event_row, edges: pd.DataFrame, chain_map: pd.DataFrame) -> str:
    source = _clean_display(event_row.get("source_news_ticker"), "the source company")
    edge_rows = _news_related_rows(event_row, edges)
    map_rows = _news_related_rows(event_row, chain_map)
    theme = _infer_news_theme(event_row)
    roles: list[str] = []
    targets = ""
    if not edge_rows.empty:
        if "theme" in edge_rows.columns:
            theme_values = [x for x in edge_rows["theme"].dropna().astype(str).tolist() if x and x.lower() not in {"nan", "none"}]
            if theme_values:
                theme = pd.Series(theme_values).mode().iloc[0]
        if "chain_role" in edge_rows.columns:
            roles = [x for x in edge_rows["chain_role"].dropna().astype(str).tolist() if x and x.lower() not in {"nan", "none"}]
        if "target_ticker" in edge_rows.columns:
            targets = ", ".join(list(dict.fromkeys(edge_rows["target_ticker"].dropna().astype(str).tolist()))[:8])
    elif not map_rows.empty:
        theme = _clean_display(map_rows.iloc[0].get("themes"), theme)
        targets = _clean_display(map_rows.iloc[0].get("top_targets"), "")
        roles = _split_news_names(map_rows.iloc[0].get("chain_roles"), 5)

    role_text = ", ".join(list(dict.fromkeys([_human_text(x, 40).lower() for x in roles if x]))[:4])
    if not role_text:
        role_text = "direct and peer links"
    if targets:
        return f"{source} is the source ticker. The story is mapped to {theme} through {role_text}. Names to verify: {targets}."
    return f"{source} is the source ticker. The story may spread through {theme}, but the linked-stock list still needs proof."


def _news_proof_status_sentence(event_row, validation_queue: pd.DataFrame, edges: pd.DataFrame) -> str:
    val_rows = _news_related_rows(event_row, validation_queue)
    edge_rows = _news_related_rows(event_row, edges)
    contradiction_count = 0
    validated_count = 0
    confidence = None
    if not edge_rows.empty:
        text = " ".join(edge_rows.get("causal_chain_status", pd.Series(dtype=str)).fillna("").astype(str).str.upper().tolist())
        contradiction_count = text.count("CONTRADICTED")
        validated_count = text.count("VALIDATED") + text.count("SUPPORTED")
        if "causal_confidence_score" in edge_rows.columns:
            confidence = pd.to_numeric(edge_rows["causal_confidence_score"], errors="coerce").dropna()
            confidence = float(confidence.mean()) if not confidence.empty else None
    if not val_rows.empty:
        first_issue = _news_validation_plain(val_rows.iloc[0])
        return f"Still needs proof: {len(val_rows)} link(s) need review. First issue: {first_issue}"
    if contradiction_count:
        return f"Careful: {contradiction_count} link(s) disagree with price action. Do not size from this headline."
    if validated_count:
        conf = f" Average confidence {confidence:.0f}/100." if confidence is not None else ""
        return f"Some links have support: {validated_count} edge(s) look validated.{conf} Still verify risk and liquidity."
    proof = _news_proof_items(event_row.get("top_required_proof", ""))
    return "Still research-only. " + (proof[0] if proof else "Check source timing and price reaction.")


def _render_news_industry_proof_preview(validation_queue: pd.DataFrame, edges: pd.DataFrame, max_cards: int = 4):
    if validation_queue is None or validation_queue.empty:
        return

    st.markdown("#### Why this story could spread")
    st.caption("This checks whether a headline may affect suppliers, peers, customers, or weak stocks. If price disagrees, the link stays research-only.")
    work = validation_queue.copy()
    priority_order = {"P1_REVIEW_CONTRADICTION": 0, "P2_VALIDATE": 1, "P3_CONTEXT": 2}
    if "priority" in work.columns:
        work["_priority"] = work["priority"].astype(str).map(priority_order).fillna(9)
    if "causal_confidence_score" in work.columns:
        work["_confidence"] = pd.to_numeric(work["causal_confidence_score"], errors="coerce").fillna(0)
    sort_cols = [c for c in ["_priority", "_confidence"] if c in work.columns]
    if sort_cols:
        ascending = [True if c == "_priority" else False for c in sort_cols]
        work = work.sort_values(sort_cols, ascending=ascending)

    grouped = work.groupby("headline", dropna=False) if "headline" in work.columns else [(None, work)]
    cards = []
    for headline, rows in grouped:
        rows = rows.copy()
        if "causal_confidence_score" in rows.columns:
            rows = rows.sort_values("causal_confidence_score", ascending=False)
        cards.append((headline, rows))
        if len(cards) >= max_cards:
            break

    html = ['<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin:8px 0 18px 0;">']
    for headline, rows in cards:
        first = rows.iloc[0]
        targets = ", ".join(list(dict.fromkeys(rows.get("target_ticker", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:8])
        theme = _news_plain(first.get("theme"), 85)
        tone = _news_tone_label(first.get("market_tone"))
        confidence = _news_metric_value(rows.get("causal_confidence_score", pd.Series(dtype=float)).mean() if "causal_confidence_score" in rows.columns else None)
        issue_text = " ".join(rows.get("issue", pd.Series(dtype=str)).fillna("").astype(str).str.upper().tolist())
        accent = "#991b1b" if "CONTRADICT" in issue_text or "DISAGREE" in issue_text else "#334155"
        source = f"{_news_plain(first.get('publisher'), 70)} · {_news_plain(first.get('published'), 45)}"
        action = _news_plain(first.get("required_next_action"), 170)
        note = _news_plain(first.get("validation_note"), 155)
        link = _news_link_html(first.get("link", ""))
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:15px 16px; min-height:295px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div style="font-size:12px; color:{accent}; font-weight:900; text-transform:uppercase;">{_esc(tone)} · confidence {_esc(confidence)}/100</div>
                <div style="font-size:12px; color:#111827; font-weight:900;">{_esc(theme)}</div>
              </div>
              <div style="font-size:18px; color:#111827; font-weight:950; line-height:1.25; margin-top:7px;">{_esc(_news_plain(headline, 145))}</div>
              <div style="font-size:12px; color:#6b7280; margin-top:7px;">{_esc(source)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:13px; color:#111827; line-height:1.42;"><b>Linked names:</b> {_esc(targets or 'No linked names yet')}</div>
              <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:7px;"><b>Proof problem:</b> {_esc(note)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:7px;"><b>Do next:</b> {_esc(action)}</div>
              <details style="margin-top:9px; border-top:1px solid #e5e7eb; padding-top:8px;">
                <summary style="cursor:pointer; font-size:12px; color:#111827; font-weight:850;">Why this is not automatic</summary>
                <div style="font-size:12px; color:#6b7280; line-height:1.45; margin-top:7px;">A headline can name one company but move suppliers, peers, customers, or weak competitors. The system keeps it research-only until timing and price reaction agree.</div>
              </details>
              <div style="font-size:12px; margin-top:8px;">{link}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_news_story_board(events: pd.DataFrame, board: pd.DataFrame, impact_targets: pd.DataFrame, edges: pd.DataFrame, chain_map: pd.DataFrame, validation_queue: pd.DataFrame, max_cards: int = 3):
    if events is None or events.empty:
        return
    work = events.copy()
    if "best_event_score" in work.columns:
        work["_score"] = pd.to_numeric(work["best_event_score"], errors="coerce").fillna(0)
        work = work.sort_values("_score", ascending=False)

    st.markdown("#### News to read first")
    st.caption("Read this like a plain memo: what happened, who it may help, who it may hurt, how the story spreads, and what proof is missing.")

    for _, row in work.head(max_cards).iterrows():
        tone = _news_tone_label(row.get("market_tone", ""))
        decision = _news_decision_label(row.get("top_decision", ""))
        accent = _news_accent(tone, decision)
        score = _news_metric_value(row.get("best_event_score"))
        published = _clean_display(row.get("published"), "No date")
        publisher = _clean_display(row.get("publisher"), "No publisher")
        source_ticker = _clean_display(row.get("source_news_ticker"), "Market")
        good_rows, bad_rows = _news_good_bad_rows(row, board, impact_targets, edges)
        good_html = _news_target_list_html(good_rows, "No clear winner has been proven yet.", "help")
        bad_html = _news_target_list_html(bad_rows, "No clear loser has been proven yet.", "hurt")
        chain = _news_chain_sentence(row, edges, chain_map)
        proof = _news_proof_status_sentence(row, validation_queue, edges)
        permission = _news_use_permission_text(row)
        source_link = _news_link_html(row.get("link", ""))
        _render_html(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:10px; padding:17px 19px; margin:0 0 16px 0;">
              <div style="display:flex; justify-content:space-between; gap:14px; align-items:flex-start;">
                <div style="min-width:0;">
                  <div style="font-size:12px; color:{accent}; font-weight:900; text-transform:uppercase;">{_esc(tone)} · Strength {_esc(score)}/100</div>
                  <div style="font-size:22px; color:#111827; font-weight:950; line-height:1.22; margin-top:6px;">{_esc(row.get("headline"), "Untitled headline")}</div>
                  <div style="font-size:12px; color:#6b7280; margin-top:7px;">{_esc(published)} · {_esc(publisher)} · Source ticker: {_esc(source_ticker)}</div>
                </div>
                <div style="font-size:12px; color:#111827; font-weight:850; text-align:right; min-width:130px;">{_esc(decision)}</div>
              </div>
              <div style="border-top:1px solid #e5e7eb; margin-top:12px; padding-top:10px; font-size:14px; color:#111827; line-height:1.45;">
                <b>Desk read:</b> {_esc(_news_event_plain_read(row, _news_related_rows(row, board)))}
              </div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:12px;">
                <div style="background:#f9fafb; border:1px solid #e5e7eb; border-top:3px solid #0f766e; border-radius:8px; padding:12px 13px;">
                  <div style="font-size:12px; color:#0f766e; font-weight:900; text-transform:uppercase;">May help</div>
                  {good_html}
                </div>
                <div style="background:#f9fafb; border:1px solid #e5e7eb; border-top:3px solid #991b1b; border-radius:8px; padding:12px 13px;">
                  <div style="font-size:12px; color:#991b1b; font-weight:900; text-transform:uppercase;">May hurt</div>
                  {bad_html}
                </div>
              </div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:12px;">
                <div style="background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px 13px; font-size:13px; color:#374151; line-height:1.45;">
                  <b>Industry chain:</b> {_esc(chain)}
                </div>
                <div style="background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px 13px; font-size:13px; color:#374151; line-height:1.45;">
                  <b>Proof status:</b> {_esc(proof)}
                </div>
              </div>
              <div style="border-top:1px solid #e5e7eb; margin-top:12px; padding-top:9px; font-size:13px; color:#111827; line-height:1.45;">
                <b>Allowed use:</b> {_esc(permission)}
              </div>
              <details style="margin-top:8px;">
                <summary style="cursor:pointer; font-size:12px; color:#111827; font-weight:850;">Click for why the model linked these names</summary>
                <div style="font-size:12px; color:#6b7280; line-height:1.45; margin-top:8px;">
                  The link comes from related-stock mapping, industry-chain mapping, target scoring, safety checks, and price-reaction checks. Use it as a hypothesis until the proof status clears.
                </div>
              </details>
              <div style="font-size:12px; margin-top:9px;">{source_link}</div>
            </div>
            """
        )


def _render_news_plain_reading_order(summary: pd.DataFrame, ranking: pd.DataFrame, validation_queue: pd.DataFrame):
    top_headline = "No headline file yet"
    if summary is not None and not summary.empty:
        work = summary.copy()
        if "best_event_score" in work.columns:
            work["_score"] = pd.to_numeric(work["best_event_score"], errors="coerce").fillna(0)
            work = work.sort_values("_score", ascending=False)
        top_headline = _human_text(work.iloc[0].get("headline"), max_len=120)

    top_target = "No stock mapped yet"
    if ranking is not None and not ranking.empty:
        work = ranking.copy()
        if "best_event_score" in work.columns:
            work["_score"] = pd.to_numeric(work["best_event_score"], errors="coerce").fillna(0)
            work = work.sort_values("_score", ascending=False)
        top_target = _clean_display(work.iloc[0].get("target_ticker"), "No stock mapped yet")

    proof_count = len(validation_queue) if validation_queue is not None and not validation_queue.empty else 0
    st.markdown("#### How to read the news page")
    c1, c2, c3 = st.columns(3)
    with c1:
        _simple_card("1. Read first", top_headline, "Start with the highest-impact headline.", "#111827")
    with c2:
        _simple_card("2. Check affected stock", top_target, "Ask: direct winner, peer, supplier, customer, or possible loser?", "#334155")
    with c3:
        _simple_card("3. Prove it", f"{proof_count:,} proof item(s)", "Do not trust a headline until timing and price reaction are checked.", "#991b1b" if proof_count else "#166534")


def _render_news_event_cards(events: pd.DataFrame, decision_board: pd.DataFrame, impact_targets: pd.DataFrame):
    if events.empty:
        st.info("No event summary file is available yet. Run the daily system, then come back to News.")
        return

    work = events.copy()
    if "best_event_score" in work.columns:
        work["_score"] = pd.to_numeric(work["best_event_score"], errors="coerce").fillna(0)
        work = work.sort_values("_score", ascending=False)

    st.markdown("#### News to read first")
    st.caption("Each card is written as a plain-English research note. It is not a trade order.")
    card_cols = st.columns(2)
    for idx, (_, row) in enumerate(work.head(6).iterrows()):
        tone = _news_tone_label(row.get("market_tone", ""))
        decision = _news_decision_label(row.get("top_decision", ""))
        accent = _news_accent(tone, decision)
        score = _news_metric_value(row.get("best_event_score"))
        helps_text = _news_names_sentence(row.get("top_beneficiaries"), "No clear winner mapped yet", 6)
        hurts_text = _news_names_sentence(row.get("top_vulnerable_targets"), "No clear loser mapped yet", 6)
        story = _news_card_story(row, decision_board)
        help_bullets = _news_bullet_html(story["help_lines"], "No clear winner has been proven yet.")
        hurt_bullets = _news_bullet_html(story["hurt_lines"], "No clear loser has been proven yet.")
        proof_bullets = _news_bullet_html(story["proof_items"], "Check source timing, price reaction, risk, and liquidity.")
        source_link = _news_link_html(row.get("link", ""))
        published = _clean_display(row.get("published"), "No date")
        publisher = _clean_display(row.get("publisher"), "No publisher")
        source_ticker = _clean_display(row.get("source_news_ticker"), "Market")
        permission = _news_use_permission_text(row)

        with card_cols[idx % 2]:
            st.markdown(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:16px 17px; min-height:410px; margin:0 0 15px 0;">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
                    <div style="font-size:12px; color:{accent}; font-weight:850;">{_esc(tone)}</div>
                    <div style="font-size:12px; color:#6b7280; font-weight:800;">Strength {_esc(score)}/100</div>
                  </div>
                  <div style="font-size:20px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(row.get("headline"), "Untitled headline")}</div>
                  <div style="font-size:12px; color:#6b7280; margin-top:8px;">{_esc(published)} · {_esc(publisher)} · Source ticker: {_esc(source_ticker)}</div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:12px; padding-top:10px; font-size:13px; color:#111827; line-height:1.45;">
                    <b>What happened:</b> {_esc(story["plain_read"])}
                  </div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:13px; color:#374151; line-height:1.42;">
                    <div><b>May help</b><br>{_esc(helps_text)}</div>
                    <div><b>May hurt</b><br>{_esc(hurts_text)}</div>
                  </div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:13px; color:#111827; line-height:1.45;">
                    <b>Use it how?</b> {_esc(permission)}
                  </div>
                  <details style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px;">
                    <summary style="cursor:pointer; font-size:13px; color:#111827; font-weight:850;">Why may it help or hurt?</summary>
                    <div style="font-size:12px; color:#374151; line-height:1.45; margin-top:8px;">
                      <b>Why it may help</b>
                      {help_bullets}
                      <div style="height:6px;"></div>
                      <b>Why it may hurt</b>
                      {hurt_bullets}
                    </div>
                  </details>
                  <details style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px;">
                    <summary style="cursor:pointer; font-size:13px; color:#111827; font-weight:850;">What proof is still missing?</summary>
                    <div style="font-size:12px; color:#6b7280; line-height:1.45; margin-top:8px;">{proof_bullets}</div>
                  </details>
                  <div style="font-size:12px; margin-top:9px;">{source_link}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_news_target_cards(targets: pd.DataFrame, show_table: bool = False, limit: int = 8):
    if targets.empty:
        st.info("No target ranking file is available yet.")
        return

    # Load alpha + momentum for signal context
    alpha_df = safe_csv(ROOT / "alpha_scores.csv")
    alpha_map: dict = {}
    if not alpha_df.empty and "ticker" in alpha_df.columns:
        alpha_map = {str(r["ticker"]): r.to_dict() for _, r in alpha_df.iterrows()}
    mom_df = safe_csv(ROOT / "momentum_scores.csv")
    mom_map: dict = {}
    if not mom_df.empty and "ticker" in mom_df.columns:
        mom_map = {str(r["ticker"]): _to_float(r.get("momentum_score")) for _, r in mom_df.iterrows()}

    work = targets.copy()
    if "best_event_score" in work.columns:
        work["_score"] = pd.to_numeric(work["best_event_score"], errors="coerce").fillna(0)
        work = work.sort_values("_score", ascending=False)

    st.markdown("#### News watchlist — affected tickers")
    st.caption(
        "Stocks the news engine thinks may be affected. α = combined alpha score · Mom = momentum rank. "
        "Green pill = positive signal. Red = caution. Research only — no trade instruction."
    )
    card_cols = st.columns(4)
    for idx, (_, row) in enumerate(work.head(limit).iterrows()):
        decision = _news_decision_label(row.get("top_decision", ""))
        tone = _news_tone_label(row.get("top_tone", ""))
        accent = _news_accent(tone, decision)
        cycle = _news_cycle_plain(row.get("subsector_cycle_phase"))
        headline = _clean_display(row.get("top_headline"), "No headline")
        proof = _news_bullet_html(_news_proof_items(row.get("proof_required", "")), "Check source timing, price reaction, risk, and liquidity.")
        plain_read = _news_target_plain_read(row)
        now = _news_target_now(row)
        score = _news_metric_value(row.get("best_event_score"))
        tkr = str(row.get("target_ticker") or "")

        # Alpha + momentum context pills
        a_row = alpha_map.get(tkr)
        a_score = _to_float(a_row.get("alpha_score")) if a_row else None
        a_rank  = a_row.get("alpha_rank") if a_row else None
        m_score = mom_map.get(tkr)
        a_color = "#16a34a" if a_score and a_score >= 65 else "#dc2626" if a_score and a_score < 45 else "#64748b"
        m_color = "#16a34a" if m_score and m_score >= 65 else "#dc2626" if m_score and m_score < 35 else "#64748b"
        rank_str = f"#{int(a_rank)}" if a_rank and str(a_rank) not in {"nan","None",""} else ""
        quant_html = ""
        if a_score is not None:
            quant_html += f'<span style="background:#f8fafc;border:1px solid #e2e8f0;color:{a_color};font-size:10px;font-weight:700;padding:2px 7px;border-radius:99px;">α {a_score:.0f} {rank_str}</span>  '
        if m_score is not None:
            quant_html += f'<span style="background:#f8fafc;border:1px solid #e2e8f0;color:{m_color};font-size:10px;font-weight:700;padding:2px 7px;border-radius:99px;">Mom {m_score:.0f}</span>'

        tone_bg  = "#fef2f2" if "negative" in tone.lower() or "bearish" in tone.lower() else "#f0fdf4" if "positive" in tone.lower() or "bullish" in tone.lower() else "#f8fafc"
        tone_col = "#dc2626" if "negative" in tone.lower() else "#16a34a" if "positive" in tone.lower() else "#64748b"

        with card_cols[idx % 4]:
            st.markdown(
                f"""
                <div style="
                    background:#fff;
                    border:1px solid #e2e8f0;
                    border-top:3px solid {accent};
                    border-radius:12px;
                    padding:14px 16px;
                    min-height:340px;
                    margin:0 0 14px 0;
                    box-shadow:0 2px 6px rgba(0,0,0,.06);
                ">
                  <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:6px;">
                    <div style="font-size:26px; font-weight:900; color:#0f172a; letter-spacing:-0.3px;">{_esc(tkr)}</div>
                    <span style="font-size:10px;color:{tone_col};font-weight:700;background:{tone_bg};padding:3px 8px;border-radius:99px;text-transform:uppercase;letter-spacing:.04em;">{_esc(tone)}</span>
                  </div>
                  <div style="font-size:11px; color:#64748b; margin-bottom:7px;">{_esc(_news_role_label(row.get("top_target_role")))}</div>
                  <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px;">{quant_html}</div>
                  <div style="font-size:13px; color:#0f172a; line-height:1.45; font-weight:600;">{_esc(now)}</div>
                  <div style="height:1px;background:#f1f5f9;margin:9px 0;"></div>
                  <div style="font-size:12px; color:#475569; line-height:1.45;"><b>Why:</b> {_esc(plain_read)}</div>
                  <div style="font-size:11.5px; color:#64748b; line-height:1.38; margin-top:7px;"><b>News:</b> {_esc(headline[:100])}{'…' if len(str(headline)) > 100 else ''}</div>
                  <div style="font-size:11px; color:#94a3b8; margin-top:6px;">{_esc(cycle)}</div>
                  <details style="margin-top:9px; border-top:1px solid #f1f5f9; padding-top:8px;">
                    <summary style="cursor:pointer; font-size:11px; color:#64748b; font-weight:600;">Proof still needed</summary>
                    <div style="font-size:11px; color:#94a3b8; line-height:1.45; margin-top:6px;">{proof}</div>
                  </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if show_table:
        cols = [c for c in [
            "target_ticker", "best_event_score", "positive_event_count", "negative_event_count",
            "top_decision", "top_target_role", "top_tone", "directional_route",
            "final_risk_action", "subsector_cycle_phase", "top_headline", "proof_required",
        ] if c in work.columns]
        with st.expander("Open detailed target table", expanded=False):
            _show_status_table(work[cols] if cols else work, ["top_decision", "top_tone", "final_risk_action"], height=560)


def _render_news_chain(events: pd.DataFrame, chain_map: pd.DataFrame, ladder: pd.DataFrame, edges: pd.DataFrame):
    st.markdown("#### Industry chain logic")
    st.caption("This is where a headline becomes an upstream, downstream, peer, or vulnerable-target hypothesis.")

    if chain_map.empty:
        st.info("No causal chain map is available yet.")
        return

    work = chain_map.copy()
    if "avg_causal_confidence" in work.columns:
        work["_confidence"] = pd.to_numeric(work["avg_causal_confidence"], errors="coerce").fillna(0)
        work = work.sort_values("_confidence", ascending=False)

    card_cols = st.columns(2)
    for idx, (_, row) in enumerate(work.head(8).iterrows()):
        tone = _news_tone_label(row.get("market_tone", ""))
        accent = _news_accent(tone, row.get("map_status", ""))
        targets = _clean_display(row.get("top_targets"), "No targets mapped")
        themes = _clean_display(row.get("themes"), _infer_news_theme(row))
        chain_read = _news_chain_plain_read(row)
        proof_line = _news_chain_proof_line(row)
        confidence = _news_metric_value(row.get("avg_causal_confidence"))
        with card_cols[idx % 2]:
            st.markdown(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 15px; min-height:280px; margin:0 0 14px 0;">
                  <div style="font-size:12px; color:{accent}; font-weight:850;">{_esc(tone)} · Story-link trust {_esc(confidence)}/100</div>
                  <div style="font-size:18px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(row.get("headline"), "Untitled headline")}</div>
                  <div style="font-size:12px; color:#6b7280; margin-top:7px;">Theme: {_esc(themes)}</div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:13px; color:#111827; line-height:1.45;"><b>Plain English:</b> {_esc(chain_read)}</div>
                  <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;"><b>Stocks to check:</b> {_esc(targets)}</div>
                  <details style="margin-top:9px;">
                    <summary style="cursor:pointer; font-size:12px; color:#111827; font-weight:850;">Click to see evidence quality</summary>
                    <div style="font-size:12px; color:#6b7280; line-height:1.45; margin-top:7px;">
                      {_esc(proof_line)} Source file: {_esc(row.get("source_file"), "local files")}.
                    </div>
                  </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Open detailed industry-chain tables", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("##### Chain ladder")
            ladder_cols = [c for c in ["event_id", "target_ticker", "stage_order", "stage", "primary_evidence", "secondary_evidence", "decision", "source_files"] if c in ladder.columns]
            _show_status_table(ladder[ladder_cols].head(80) if ladder_cols else ladder.head(80), ["decision"], height=520)
        with right:
            st.markdown("##### Story links")
            edge_cols = [c for c in [
                "source_news_ticker", "target_ticker", "target_relation", "theme", "chain_role",
                "market_tone", "causal_chain_status", "causal_confidence_score",
                "event_to_latest_return_pct", "validation_note", "causal_thesis",
            ] if c in edges.columns]
            _show_status_table(edges[edge_cols].head(80) if edge_cols else edges.head(80), ["market_tone", "causal_chain_status"], height=520)


def _render_news_proof_queue(validation_queue: pd.DataFrame, truth_ledger: pd.DataFrame, dossier: pd.DataFrame):
    st.markdown("#### What still needs proof")
    st.caption("These are the news links that need evidence before they can influence any idea.")

    if not validation_queue.empty:
        work = validation_queue.copy()
        priority_order = {"P1_REVIEW_CONTRADICTION": 0, "P2_VALIDATE": 1, "P3_CONTEXT": 2}
        if "priority" in work.columns:
            work["_priority"] = work["priority"].astype(str).map(priority_order).fillna(9)
            work = work.sort_values("_priority")
        st.markdown("##### Proof gaps to fix first")
        card_cols = st.columns(2)
        for idx, (_, row) in enumerate(work.head(8).iterrows()):
            accent = "#991b1b" if "P1" in str(row.get("priority", "")) else "#334155"
            plain_issue = _news_validation_plain(row)
            next_step = _human_text(row.get("required_next_action", "Validate manually"), max_len=220)
            confidence = _news_metric_value(row.get("causal_confidence_score"))
            with card_cols[idx % 2]:
                st.markdown(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 15px; min-height:270px; margin:0 0 14px 0;">
                      <div style="font-size:12px; color:{accent}; font-weight:850;">Needs proof · Story-link trust {_esc(confidence)}/100</div>
                      <div style="font-size:18px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(row.get("target_ticker"), "")} · {_esc(_clean_display(row.get("theme"), _infer_news_theme(row)))}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.4; margin-top:8px;">{_esc(row.get("headline"), "No headline")}</div>
                      <div style="border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px; font-size:13px; color:#111827; line-height:1.45;">
                        <b>Why blocked:</b> {_esc(plain_issue)}
                      </div>
                      <div style="font-size:12px; color:#6b7280; line-height:1.45; margin-top:8px;">
                        <b>Next proof step:</b> {_esc(next_step)}
                      </div>
                      <div style="font-size:12px; margin-top:8px;">{_news_link_html(row.get("link", ""))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("Open event timing and dossier tables", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("##### Event time truth ledger")
            cols = [c for c in [
                "ticker", "related_ticker", "headline", "event_date", "first_seen_time",
                "publisher", "pit_quality_status", "event_time_status", "lookahead_risk",
                "limitation",
            ] if c in truth_ledger.columns]
            _show_status_table(truth_ledger[cols].head(80) if cols else truth_ledger.head(80), ["pit_quality_status", "event_time_status", "lookahead_risk"], height=520)
        with right:
            st.markdown("##### Event research dossier")
            cols = [c for c in [
                "ticker", "event_research_score", "event_source_coverage_pct",
                "event_gate", "status", "earnings_risk_flag", "latest_news_title",
                "catalysts", "risks", "required_next_action",
            ] if c in dossier.columns]
            _show_status_table(dossier[cols] if cols else dossier, ["event_gate", "status", "earnings_risk_flag"], height=520)


def _render_ticker_news_cards(rows: pd.DataFrame):
    if rows.empty:
        st.info("No mapped news for this ticker yet.")
        return

    work = rows.copy()
    if "news_rank" in work.columns:
        work["_rank"] = pd.to_numeric(work["news_rank"], errors="coerce").fillna(999)
        work = work.sort_values("_rank")
    elif "impact_score" in work.columns:
        work["_score"] = pd.to_numeric(work["impact_score"], errors="coerce").fillna(0)
        work = work.sort_values("_score", ascending=False)

    st.markdown("##### Why this ticker is in the news map")
    card_cols = st.columns(2)
    for idx, (_, row) in enumerate(work.head(8).iterrows()):
        direction = _clean_display(row.get("news_direction"), "Mapped news")
        tone_hint = "NEGATIVE" if "bear" in direction.lower() or "down" in direction.lower() else "POSITIVE" if "bull" in direction.lower() or "up" in direction.lower() else "MIXED"
        accent = _news_accent(tone_hint, row.get("calibrated_research_action", ""))
        reliability = _plain_status(row.get("calibrated_reliability_status", "No reliability tag"))
        action = _plain_status(row.get("calibrated_research_action", "Research only"))
        target_reason = _clean_display(row.get("target_reason"), "The system mapped this headline to the ticker, but the reason file is incomplete.")
        logic = _clean_display(row.get("news_logic"), "No news logic text yet.")
        vulnerability = _clean_display(row.get("negative_vulnerability_summary"), "No negative vulnerability flag.")
        plain_read = _news_ticker_plain_read(row)
        next_step = _news_ticker_next_step(row)
        link = _news_link_html(row.get("link", ""))
        with card_cols[idx % 2]:
            st.markdown(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 15px; min-height:340px; margin:0 0 14px 0;">
                  <div style="display:flex; justify-content:space-between; gap:10px;">
                    <div style="font-size:12px; color:{accent}; font-weight:850;">{_esc(_human_text(direction, max_len=80))}</div>
                    <div style="font-size:12px; color:#6b7280; font-weight:800;">Rank {_esc(row.get("news_rank"), "-")}</div>
                  </div>
                  <div style="font-size:18px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(row.get("headline"), "Untitled headline")}</div>
                  <div style="font-size:12px; color:#6b7280; margin-top:7px;">{_esc(row.get("published"), "No date")} · {_esc(row.get("publisher"), "No publisher")} · Source: {_esc(row.get("source_news_ticker"), "Market")}</div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:9px; font-size:13px; color:#111827; line-height:1.45;">
                    <b>Plain English:</b> {_esc(plain_read)}
                  </div>
                  <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;">
                    <b>What to check next:</b> {_esc(next_step)}
                  </div>
                  <div style="font-size:12px; color:#4b5563; line-height:1.4; margin-top:8px;">
                    <b>Why linked:</b> {_esc(_human_text(target_reason, max_len=190))}
                  </div>
                  <details style="margin-top:9px;">
                    <summary style="cursor:pointer; font-size:12px; color:#111827; font-weight:850;">Click for model notes and weakness check</summary>
                    <div style="font-size:12px; color:#6b7280; line-height:1.45; margin-top:8px;">
                      <b>Model note:</b> {_esc(_human_text(logic, max_len=220))}<br>
                      <b>Weakness check:</b> {_esc(_human_text(vulnerability, max_len=170))}<br>
                      <b>Reliability:</b> {_esc(reliability)}<br>
                      <b>Research action:</b> {_esc(action)}
                    </div>
                  </details>
                  <div style="font-size:12px; margin-top:9px;">{link}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def tab_news_room():
    st.markdown('<p class="section-title">News: what matters today?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Read the headline, see who may benefit or get hurt, then check what proof is still missing. Research-only. No broker connection. No live orders.</p>',
        unsafe_allow_html=True,
    )
    _render_section_depth("News")

    summary = safe_csv(ROOT / "event_readthrough_event_summary.csv")
    board = safe_csv(ROOT / "event_readthrough_decision_board.csv")
    ranking = safe_csv(ROOT / "event_readthrough_target_ranking.csv")
    chain_map = safe_csv(ROOT / "event_causal_chain_map.csv")
    chain_edges = safe_csv(ROOT / "event_causal_chain_edges.csv")
    chain_ladder = safe_csv(ROOT / "event_readthrough_chain_ladder.csv")
    validation_queue = safe_csv(ROOT / "event_causal_validation_queue.csv")
    impact_targets = safe_csv(ROOT / "news_impact_targets.csv")
    supply_chain = safe_csv(ROOT / "news_supply_chain_readthrough.csv")
    ticker_news = safe_csv(ROOT / "ticker_decision_room_news.csv")
    dossier = safe_csv(ROOT / "event_research_dossier.csv")
    truth_ledger = safe_csv(ROOT / "event_time_truth_ledger.csv")
    state = safe_json(ROOT / "event_readthrough_state.json")
    impact_state = safe_json(ROOT / "news_impact_targeting_state.json")
    chain_state = safe_json(ROOT / "event_causal_chain_state.json")
    time_state = safe_json(ROOT / "event_time_truth_state.json")
    reliability_state = safe_json(ROOT / "event_signal_reliability_state.json")

    _render_news_command_center(summary, ranking, validation_queue, chain_map, impact_state, chain_state)

    themes = impact_state.get("themes_triggered", [])
    if isinstance(themes, list) and themes:
        st.markdown(
            "<div style='font-size:13px; color:#374151; margin:4px 0 14px 0;'><b>Active themes:</b> "
            + _esc(_news_plain(", ".join(str(x) for x in themes), 220))
            + "</div>",
            unsafe_allow_html=True,
        )

    _render_news_plain_reading_order(summary, ranking, validation_queue)

    _render_news_story_board(summary, board, impact_targets, chain_edges, chain_map, validation_queue)
    _render_news_industry_proof_preview(validation_queue, chain_edges)
    _render_news_target_cards(ranking)

    show_detail = st.checkbox("Show deeper news evidence", value=False)
    if not show_detail:
        return

    st.markdown("---")
    detail_view = st.radio(
        "News detail to open",
        ["Industry chain", "Proof queue", "Ticker lookup", "More story cards", "Source files"],
        horizontal=True,
        label_visibility="collapsed",
    )
    _render_subtab_depth("News", detail_view)

    if detail_view == "Industry chain":
        _render_news_chain(summary, chain_map, chain_ladder, chain_edges)
        if not supply_chain.empty:
            with st.expander("Open detailed supply-chain table", expanded=False):
                cols = [c for c in [
                    "source_news_ticker", "target_ticker", "target_relation", "theme",
                    "chain_role", "market_tone", "impact_score", "news_logic",
                    "suggested_research_route", "option_side", "target_reason",
                    "final_risk_action", "headline",
                ] if c in supply_chain.columns]
                _show_status_table(supply_chain[cols].head(100) if cols else supply_chain.head(100), ["market_tone", "suggested_research_route", "final_risk_action"], height=560)

    elif detail_view == "Proof queue":
        _render_news_proof_queue(validation_queue, truth_ledger, dossier)
        with st.expander("Open detailed target-level decision table", expanded=False):
            cols = [c for c in [
                "published", "source_news_ticker", "target_ticker", "target_role",
                "market_tone", "theme", "chain_role", "event_score",
                "readthrough_decision", "directional_route", "option_side",
                "option_permission", "subsector_cycle_phase", "final_risk_action",
                "why_this_target", "proof_required", "headline",
            ] if c in board.columns]
            _show_status_table(board[cols].head(120) if cols else board.head(120), ["market_tone", "readthrough_decision", "directional_route", "option_permission", "final_risk_action"], height=620)

    elif detail_view == "Ticker lookup":
        st.markdown("#### News by ticker")
        st.caption("Use this when you already care about a ticker and want to know which headlines the system mapped to it.")
        if ticker_news.empty:
            st.info("No ticker news room file is available yet.")
        else:
            tickers = sorted([str(x) for x in ticker_news["ticker"].dropna().unique()]) if "ticker" in ticker_news.columns else []
            selected = st.selectbox("Ticker", tickers, index=0 if tickers else None)
            show = ticker_news[ticker_news["ticker"].astype(str) == str(selected)] if selected and "ticker" in ticker_news.columns else ticker_news
            _render_ticker_news_cards(show)
            cols = [c for c in [
                "ticker", "news_rank", "news_direction", "headline", "published",
                "publisher", "source_news_ticker", "target_relation", "theme",
                "chain_role", "impact_score", "news_logic", "calibrated_event_score",
                "calibrated_reliability_status", "calibrated_research_action",
                "negative_vulnerability_summary", "target_reason", "link",
            ] if c in show.columns]
            with st.expander("Open detailed ticker news table", expanded=False):
                _show_status_table(show[cols] if cols else show, ["news_direction", "calibrated_reliability_status", "calibrated_research_action"], height=640)

    elif detail_view == "More story cards":
        _render_news_event_cards(summary, board, impact_targets)

    elif detail_view == "Source files":
        st.markdown("#### Source files used by this page")
        _render_risk_source_inventory([
            ("event_readthrough_state.json", "Top news read-through state"),
            ("event_readthrough_event_summary.csv", "Headline summary and top help/hurt names"),
            ("event_readthrough_decision_board.csv", "Every headline-to-target decision row"),
            ("event_readthrough_target_ranking.csv", "Target stock ranking from event mapping"),
            ("news_impact_targeting_state.json", "News targeting counts and active themes"),
            ("news_impact_targets.csv", "News impact targets"),
            ("news_supply_chain_readthrough.csv", "Supply-chain read-through table"),
            ("event_causal_chain_state.json", "Causal chain state"),
            ("event_causal_chain_map.csv", "Headline-to-chain map"),
            ("event_causal_chain_edges.csv", "Target causal edge details"),
            ("event_readthrough_chain_ladder.csv", "Step-by-step event chain ladder"),
            ("event_causal_validation_queue.csv", "Manual proof queue"),
            ("event_time_truth_state.json", "Event timing state"),
            ("event_time_truth_ledger.csv", "Point-in-time event timing ledger"),
            ("event_research_dossier.csv", "Ticker event research dossier"),
            ("ticker_decision_room_news.csv", "Ticker-specific news room"),
            ("event_signal_reliability_state.json", "Event signal reliability state"),
            ("depth5_news_causal_proof_system.csv", "News-to-industry proof system"),
        ])
        if reliability_state:
            st.caption(
                "Reliability note: "
                + _plain_status(reliability_state.get("overall_status", "No data"))
                + ". This means the system can explain hypotheses, but still needs forward live evidence before institutional-grade trust."
            )
        if callable(_ORIGINAL_TAB_NEWS_ROOM):
            with st.expander("Original detailed News page", expanded=False):
                _ORIGINAL_TAB_NEWS_ROOM()


def _render_beginner_click_guide():
    st.markdown("#### How To Use This Site")
    st.caption("Follow this path. Do not start from a stock idea or an exciting headline.")

    _render_html(
        """
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #111827; border-radius:8px; padding:16px 18px; margin:8px 0 16px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">One rule</div>
          <div style="font-size:24px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">
            Risk first, then news proof, then ideas, then calls / puts.
          </div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;">
            If the first answer says risk is blocking, the rest of the site is for reading and proof only. It is not a trade list.
          </div>
        </div>
        """
    )

    steps = [
        (
            "1",
            "Start",
            "Stay on Home",
            "Read Today's answer and the first work queue.",
            "If it says Risk first, stop looking for new trades.",
        ),
        (
            "2",
            "What changed?",
            "Click Today",
            "Read alerts: price break, volume spike, news shock, earnings surprise, or risk breach.",
            "If the alert is red, treat the ticker as protect-first.",
        ),
        (
            "3",
            "Why does news matter?",
            "Click News",
            "Read who may benefit, who may get hurt, and what proof is missing.",
            "If proof is missing, the headline is not enough.",
        ),
        (
            "4",
            "Which names are worth studying?",
            "Click Ideas",
            "Separate short-term, medium-term, and long-term reads.",
            "One score is not enough. Horizon matters.",
        ),
        (
            "5",
            "Can I size it?",
            "Click Risk",
            "Check position size, sector crowding, drawdown, volatility, and stress.",
            "Risk can block. Options cannot override risk.",
        ),
        (
            "6",
            "Where did it go?",
            "Click Live / Paper",
            "Check paper book, manual live NAV, monitor events, and notes.",
            "Still no broker connection and no live order path.",
        ),
    ]

    html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:10px 0 18px 0;">']
    for number, title, click, read, stop in steps:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:14px 15px; min-height:210px;">
              <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:28px; height:28px; border-radius:50%; background:#111827; color:#fff; font-size:14px; font-weight:900; display:flex; align-items:center; justify-content:center;">{_esc(number)}</div>
                <div style="font-size:17px; color:#111827; font-weight:900;">{_esc(title)}</div>
              </div>
              <div style="font-size:13px; color:#111827; font-weight:850; margin-top:12px;">Go to: {_esc(click.replace("Click ", ""))}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:7px;">{_esc(read)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.35;">{_esc(stop)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_deep_logic_chain():
    pm_state = safe_json(ROOT / "pm_morning_brief_state.json")
    risk_state = safe_json(ROOT / "risk_desk_overview.json")
    risk_breaches = safe_csv(ROOT / "risk_desk_breach_table.csv")
    proof_state = safe_json(ROOT / "sharpe4_manual_proof_review_state.json")
    news_validation = safe_csv(ROOT / "event_causal_validation_queue.csv")
    readiness = safe_csv(ROOT / "action_readiness_detail_cards.csv")
    horizon = safe_csv(ROOT / "horizon_vehicle_summary.csv")
    route_matrix = safe_csv(ROOT / "options_execution_route_matrix.csv")
    sharpe_state = safe_json(ROOT / "sharpe4_simple_command_state.json")

    hard_breaches = int(_to_float(pm_state.get("hard_breaches", risk_state.get("budget_hard_breach_count")), 0) or 0)
    size_down = int(_to_float(pm_state.get("size_down_breaches", risk_state.get("budget_size_down_count")), 0) or 0)
    reviewed = int(_to_float(proof_state.get("reviewed_rows", pm_state.get("reviewed_rows")), 0) or 0)
    not_reviewed = int(_to_float(proof_state.get("not_reviewed_count", pm_state.get("not_reviewed_count")), 0) or 0)
    validation_rows = len(news_validation) if not news_validation.empty else int(_to_float(pm_state.get("news_validation_rows"), 0) or 0)
    blocked_cards = 0
    if not readiness.empty and "card_status" in readiness.columns:
        blocked_cards = int(readiness["card_status"].astype(str).str.lower().str.contains("blocked|conflict", regex=True).sum())
    paper_allowed = int(_to_float(sharpe_state.get("paper_sizing_allowed_now_count"), 0) or 0)
    options_allowed = int(_to_float(sharpe_state.get("options_allowed_now_count"), 0) or 0)
    headline_sharpe = _to_float(sharpe_state.get("current_headline_sharpe"), None)
    proof_sharpe = _to_float(sharpe_state.get("proof_adjusted_sharpe"), None)

    rows = [
        {
            "step": "1. Can we add risk?",
            "answer": "No. Protect first." if hard_breaches or size_down else "Maybe. Risk is not blocking.",
            "why": f"{hard_breaches} hard risk breach(es), {size_down} size-down warning(s).",
            "click": "Risk",
            "accent": "#991b1b" if hard_breaches or size_down else "#166534",
        },
        {
            "step": "2. Is the evidence filled?",
            "answer": "No. Evidence is still mostly blank." if not reviewed else "Some evidence is filled.",
            "why": f"{reviewed} evidence row(s) filled, {not_reviewed} still blank.",
            "click": "Performance > What must be checked first",
            "accent": "#334155" if not reviewed else "#166534",
        },
        {
            "step": "3. Did news prove the stock link?",
            "answer": "Not yet. Headlines still need proof." if validation_rows else "No major proof gaps found.",
            "why": f"{validation_rows} news link(s) still need timing, source, and price-reaction checks.",
            "click": "News",
            "accent": "#334155" if validation_rows else "#166534",
        },
        {
            "step": "4. Is the time horizon clear?",
            "answer": "Use horizon pages before any vehicle choice." if not horizon.empty else "Horizon route needs another run.",
            "why": "Short-term, medium-term, and long-term can disagree. A single score is not enough.",
            "click": "Ideas",
            "accent": "#111827",
        },
        {
            "step": "5. Stock, call, put, or wait?",
            "answer": "Wait / research only." if paper_allowed == 0 and options_allowed == 0 else "Some routes may be reviewed.",
            "why": f"Paper allowed now: {paper_allowed}. Options allowed now: {options_allowed}.",
            "click": "Ideas, then Risk",
            "accent": "#991b1b" if paper_allowed == 0 and options_allowed == 0 else "#166534",
        },
        {
            "step": "6. Is performance believable?",
            "answer": "Not enough yet." if proof_sharpe is not None and proof_sharpe < 1 else "Improving, but still needs live proof.",
            "why": (
                f"Headline Sharpe {headline_sharpe:.2f}, proof-adjusted {proof_sharpe:.2f}."
                if headline_sharpe is not None and proof_sharpe is not None
                else "Performance files are missing."
            ),
            "click": "Performance",
            "accent": "#334155",
        },
    ]

    st.markdown("#### Decision Logic")
    st.caption("This is the professional chain underneath the site, written in plain English.")

    html = ['<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin:10px 0 18px 0;">']
    for row in rows:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {row['accent']}; border-radius:8px; padding:14px 15px; min-height:175px;">
              <div style="font-size:16px; color:#111827; font-weight:900; line-height:1.25;">{_esc(row['step'])}</div>
              <div style="font-size:18px; color:#111827; font-weight:900; line-height:1.25; margin-top:8px;">{_esc(row['answer'])}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;">{_esc(row['why'])}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280;">Where to click: {_esc(row['click'])}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))

    with st.expander("Advanced proof trail", expanded=False):
        st.caption("These are the underlying files. You do not need this unless you are debugging the system.")
        _render_risk_source_inventory([
            ("pm_morning_brief_state.json", "PM morning answer"),
            ("risk_desk_overview.json", "Risk desk summary"),
            ("risk_desk_breach_table.csv", "Risk limit details"),
            ("sharpe4_manual_proof_review_state.json", "Evidence review state"),
            ("event_causal_validation_queue.csv", "News proof queue"),
            ("action_readiness_detail_cards.csv", "Ticker readiness cards"),
            ("horizon_vehicle_summary.csv", "Horizon and vehicle summary"),
            ("options_execution_route_matrix.csv", "Options route matrix"),
        ])


def _render_everything_map():
    st.markdown("#### Where Everything Is")
    st.caption("The top bar is now grouped into eight rooms. Use this map when you are not sure where to go.")

    cards = [
        (
            "Home",
            "Refresh the system, read the one-page answer, then choose the next room.",
            "Start here every time. It is the morning desk page.",
        ),
        (
            "Today",
            "Daily answer, alerts, price breaks, volume spikes, news shocks, earnings surprises, and risk breaches.",
            "Use this when you want to know what changed today and what needs attention first.",
        ),
        (
            "Ideas",
            "Research candidates plus short-term, medium-term, and long-term route.",
            "Use this only after Risk and News do not block the name.",
        ),
        (
            "News",
            "Plain-English headline impact, who may benefit, who may get hurt, and industry-chain links.",
            "Use this to understand why a headline matters. Do not trade from headline tone alone.",
        ),
        (
            "Risk",
            "Portfolio risk, single-name risk, sector concentration, VaR/CVaR, drawdown, correlation, and trust checks.",
            "Use this before sizing anything. If Risk says wait, the idea waits.",
        ),
        (
            "Performance",
            "Sharpe target, proof-adjusted performance, signal quality, and what must be fixed before trusting the model.",
            "Use this to ask whether the system is actually getting better, not just producing more files.",
        ),
        (
            "Live / Paper",
            "Paper book, manual live NAV, monitor events, trade notes, and learning journal.",
            "Use this to see the book. It is still research-only; there is no broker connection.",
        ),
        (
            "System",
            "Run tools, output inventory, stale files, logs, and source reports.",
            "Use this when something looks missing, old, or broken.",
        ),
    ]

    html_parts = [
        '<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; margin:10px 0 18px 0;">'
    ]
    for title, what, use in cards:
        html_parts.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:15px 16px; min-height:150px;">
              <div style="font-size:18px; color:#111827; font-weight:850; line-height:1.2;">{_esc(title)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;">{_esc(what)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:11px; padding-top:9px; font-size:12px; color:#6b7280; line-height:1.4;">{_esc(use)}</div>
            </div>
            """
        )
    html_parts.append("</div>")
    _render_html("".join(html_parts))


def _render_pm_operating_order():
    st.markdown("#### PM Daily Order")
    st.caption("Read the dashboard in this order. This keeps the process disciplined and prevents jumping from a headline straight to calls or puts.")

    steps = [
        (
            "1. Refresh",
            "Run the daily system or confirm the latest run is fresh.",
            "If files are stale, do not interpret signals yet.",
        ),
        (
            "2. Risk first",
            "Check Risk before reading exciting ideas.",
            "If portfolio risk is red, new ideas are study-only.",
        ),
        (
            "3. News proof",
            "Read the top news and the help/hurt chain.",
            "A headline must have source, timing, price reaction, and affected tickers.",
        ),
        (
            "4. Ideas by horizon",
            "Separate short-term, medium-term, and long-term reads.",
            "One score is not enough. The same ticker can be bad short-term but useful long-term.",
        ),
        (
            "5. Vehicle choice",
            "Only after risk and evidence clear, decide stock, wait, call research, or put research.",
            "Options never override risk. Weekly chase stays blocked unless the route is explicitly clean.",
        ),
        (
            "6. Record learning",
            "After paper work, record why it was entered, what invalidates it, and what happened.",
            "No journal means no learning. No learning means Sharpe claims are not trusted.",
        ),
    ]

    html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:10px 0 18px 0;">']
    for title, action, guard in steps:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid #111827; border-radius:8px; padding:14px 15px; min-height:154px;">
              <div style="font-size:16px; color:#111827; font-weight:900; line-height:1.2;">{_esc(title)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;">{_esc(action)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.4;">{_esc(guard)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _pm_brief_accent(color: str) -> str:
    tone = str(color or "").strip().lower()
    return {
        "red": "#991b1b",
        "green": "#166534",
        "blue": "#1d4ed8",
        "cyan": "#0f766e",
        "purple": "#6d28d9",
        "gray": "#334155",
        "yellow": "#0f766e",
    }.get(tone, "#111827")


def _render_pm_morning_brief():
    state = safe_json(ROOT / "pm_morning_brief_state.json")
    cards = safe_csv(ROOT / "pm_morning_brief_cards.csv")
    queue = safe_csv(ROOT / "pm_morning_brief_focus_queue.csv")
    news = safe_csv(ROOT / "pm_morning_brief_news_to_verify.csv")

    st.markdown("#### PM Morning Brief")
    st.caption("Read this first. It turns the system files into one plain answer, one work queue, and the news that still needs proof.")

    if not state and cards.empty and queue.empty and news.empty:
        st.info("PM Morning Brief has not run yet. Run Step193 or the daily system.")
        return

    answer = _plain_status(state.get("desk_answer"), "No desk answer yet.")
    risk_mode = _plain_status(state.get("risk_mode"), "No risk status yet.")
    risk_answer = _plain_status(state.get("risk_answer"), "Risk file has not been summarized yet.")
    accent = _pm_brief_accent(state.get("risk_color", "gray"))

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:8px; padding:18px 20px; margin:8px 0 16px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Today's answer</div>
          <div style="font-size:28px; color:#111827; font-weight:900; line-height:1.2; margin-top:7px;">{_esc(answer)}</div>
          <div style="font-size:14px; color:#374151; line-height:1.45; margin-top:10px;"><b>{_esc(risk_mode)}:</b> {_esc(risk_answer)}</div>
          <div style="border-top:1px solid #e5e7eb; margin-top:12px; padding-top:9px; font-size:12px; color:#6b7280;">Research only. No broker connection. No live orders.</div>
        </div>
        """
    )

    if not cards.empty:
        show_cards = cards.head(5).copy()
        cols = st.columns(min(5, len(show_cards)))
        for idx, (_, row) in enumerate(show_cards.iterrows()):
            with cols[idx % len(cols)]:
                _simple_card(
                    _plain_status(row.get("card"), "Status"),
                    _plain_status(row.get("value"), "No data"),
                    _plain_status(row.get("why_it_matters"), ""),
                    _pm_brief_accent(row.get("color", "gray")),
                )

    if not queue.empty:
        st.markdown("##### Today's first work queue")
        st.caption("This is ordered. Do these before looking for new calls, puts, or paper size.")
        work = queue.head(6).copy()
        html = ['<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin:8px 0 14px 0;">']
        for _, row in work.iterrows():
            bucket = _plain_status(row.get("bucket"), "Task")
            ticker = _plain_status(row.get("ticker"), "Portfolio")
            if ticker.lower() in {"nan", "none", "no data", ""}:
                ticker = "Portfolio"
            task = _plain_status(row.get("plain_task"), "Review this item first.")
            why = _plain_status(row.get("why"), "The system flagged this as important.")
            avoid = _plain_status(row.get("do_not_do"), "Do not trade before review.")
            source = _friendly_source_label(row.get("source"))
            item_accent = "#991b1b" if "risk" in bucket.lower() else "#334155" if "evidence" in bucket.lower() else "#0f766e"
            html.append(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid {item_accent}; border-radius:8px; padding:14px 15px; min-height:205px;">
                  <div style="display:flex; justify-content:space-between; gap:10px;">
                    <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">{_esc(bucket)}</div>
                    <div style="font-size:13px; color:#111827; font-weight:900;">{_esc(ticker)}</div>
                  </div>
                  <div style="font-size:16px; color:#111827; font-weight:850; line-height:1.3; margin-top:8px;">{_esc(task)}</div>
                  <div style="font-size:13px; color:#374151; line-height:1.4; margin-top:8px;">{_esc(why)}</div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.35;"><b>Do not:</b> {_esc(avoid)}</div>
                  <div style="font-size:11px; color:#9ca3af; margin-top:6px;">Proof trail: {_esc(source)}</div>
                </div>
                """
            )
        html.append("</div>")
        _render_html("".join(html))

        with st.expander("Open the full work queue", expanded=False):
            cols = [c for c in ["rank", "bucket", "ticker", "plain_task", "why", "do_not_do", "source"] if c in queue.columns]
            _show_status_table(queue[cols] if cols else queue, [], height=520)

    if not news.empty:
        st.markdown("##### News that still needs proof")
        st.caption("These headlines may matter, but the source, timing, and price reaction still need to be checked.")
        for _, row in news.head(3).iterrows():
            ticker = _plain_status(row.get("ticker"), "No ticker")
            headline = _plain_status(row.get("headline"), "No headline")
            why = _plain_status(row.get("why_to_check"), "Check whether the news actually moved the stock.")
            next_step = _plain_status(row.get("next_step"), "Verify the source and post-news price reaction.")
            source = _plain_status(row.get("source"), "News source")
            _render_html(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid #0f766e; border-radius:8px; padding:13px 15px; margin:0 0 10px 0;">
                  <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">{_esc(ticker)}</div>
                  <div style="font-size:17px; color:#111827; font-weight:850; line-height:1.35; margin-top:5px;">{_esc(headline)}</div>
                  <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:8px;"><b>Why check:</b> {_esc(why)}</div>
                  <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:5px;"><b>Next:</b> {_esc(next_step)}</div>
                  <div style="font-size:11px; color:#9ca3af; margin-top:6px;">{_esc(source)}</div>
                </div>
                """
            )
        with st.expander("Open all news proof items", expanded=False):
            cols = [c for c in ["ticker", "headline", "why_to_check", "next_step", "source", "link", "research_only"] if c in news.columns]
            _show_status_table(news[cols] if cols else news, [], height=560)


def _today_plain(value, max_len: int = 190) -> str:
    raw = "" if value is None else str(value)
    raw_replacements = {
        "SQUEEZE_WATCH": "Short-interest watch",
        "HIGH_CONVICTION": "Strong score watch",
        "NEW_BUY_LIST": "New research list",
        "REGIME_CHANGE": "Market regime update",
        "PRICE_BREAK": "Price level break",
        "VOLUME_SPIKE": "Volume spike",
        "VOLATILITY_REGIME_SHIFT": "Volatility change",
        "SPREAD_WIDENING": "Spread widening",
        "CORRELATION_BREAK": "Correlation break",
        "NEWS_SHOCK": "News shock",
        "EARNINGS_SURPRISE": "Earnings surprise",
        "RISK_LIMIT_BREACH": "Risk limit breach",
        "SQUEEZE_BUY": "short-interest watch",
        "SIZE_DOWN_OR_REDUCE_ONLY": "use smaller size or reduce exposure",
        "RISK_REPAIR_REQUIRED": "risk repair needed",
        "STILL_ABOVE_TICKER_RISK_TARGET": "still above its ticker risk limit",
        "WATCH_ONLY_RISK_STILL_LOCKED": "watch only because risk is still locked",
        "WATCH_ONLY_REQUIRE_PRICE_VOLUME_CONFIRMATION": "watch only until price and volume confirm",
        "CONTEXT_ONLY_NO_DIRECTIONAL_ACTION": "context only; no action from the headline",
        "UNPROVEN_LOCAL_CONTEXT": "not proven by local evidence yet",
        "LOW_SAMPLE_REVIEW": "small sample, review only",
        "MISSING_DATA_REVIEW": "missing data review",
        "CLEAR": "clear",
        "BLOCKED": "not allowed yet",
        "NO_GO": "not allowed yet",
        "REDUCE_ONLY": "no new buying",
        "SIZE_DOWN": "use smaller size",
        "NO_NEW_OPTION": "no option idea yet",
        "NO_DATA": "no data",
        "DATA_GAP": "missing data",
        "PM_BRIEF_ACTIVE": "morning brief active",
        "VaR": "loss limit",
        "CVaR": "bad-day loss limit",
        "spread/liquidity": "trading cost and volume",
        "execution-proof": "trading-cost proof",
        "proof/risk/execution": "proof, risk, and trading-cost",
        "weekly calls or puts": "short-term options",
        "event gap risk": "bad news or earnings jump risk",
        "model-generated text": "AI-generated text",
        "outside proof": "outside evidence",
        "Outside proof": "Outside evidence",
        "PM acceptance": "research approval",
    }
    for raw_text, friendly in raw_replacements.items():
        raw = raw.replace(raw_text, friendly)

    text = _human_text(raw, max_len=None)
    readable_replacements = {
        "Reduce Only": "no new buying",
        "Size Down": "use smaller size",
        "No New Option": "no option idea yet",
        "Missing Data Review": "missing data review",
        "Unproven Local Context": "not proven by local evidence yet",
        "Context Only No Directional Action": "context only; no action from the headline",
        "Watch Only Require Price Volume Confirmation": "watch only until price and volume confirm",
        "Risk Repair Required": "risk repair needed",
        "Still Above Ticker Risk Target": "still above its ticker risk limit",
        "Watch Only Risk Still Locked": "watch only because risk is still locked",
        "Low Sample Review": "small sample, review only",
        "Data Gap": "missing data",
        "No Go": "not allowed yet",
        "risk-not ready yet": "risk does not allow yet",
        "Risk-not ready yet": "Risk does not allow yet",
        "watch-only": "watch only",
        "Watch-only": "Watch only",
        "source/earnings/news": "source, earnings, and news",
        "price/volume": "price and volume",
        "IV/Greeks/Gamma": "option volatility, Greeks, and gamma",
        "spread/TCA": "trading cost",
        "TCA": "trading cost",
    }
    for raw_text, friendly in readable_replacements.items():
        text = text.replace(raw_text, friendly)
    text = re.sub(r"rank_squeeze\s*=\s*", "short-interest rank ", text, flags=re.IGNORECASE)
    text = re.sub(r"signal\s*=\s*[^.;]+", "signal is a watch item", text, flags=re.IGNORECASE)
    text = re.sub(r"used\s*=\s*[\d.]+", "limit is over the line", text, flags=re.IGNORECASE)
    text = re.sub(r"current\s*=\s*[\d.]+", "current risk is high", text, flags=re.IGNORECASE)
    text = re.sub(r"limit\s*=\s*[\d.]+", "limit is set", text, flags=re.IGNORECASE)
    text = text.replace(" ;", ";")
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _today_accent(severity: str = "", kind: str = "") -> str:
    text = f"{severity} {kind}".upper()
    if any(x in text for x in ["CRITICAL", "RISK_LIMIT", "PRICE_BREAK", "BREACH"]):
        return "#991b1b"
    if any(x in text for x in ["WARNING", "NEWS_SHOCK", "SQUEEZE", "VOLUME"]):
        return "#334155"
    if any(x in text for x in ["INFO", "CLEAR", "OK"]):
        return "#166534"
    return "#111827"


def _today_severity_label(value) -> str:
    text = str(value or "").upper()
    if "CRITICAL" in text:
        return "Needs attention now"
    if "WARNING" in text:
        return "Review today"
    if "INFO" in text:
        return "FYI"
    if "CLEAR" in text:
        return "Clear"
    return _today_plain(value, 70)


def _today_alert_kind(value) -> str:
    text = str(value or "").upper()
    mapping = {
        "SQUEEZE_WATCH": "Short-interest watch",
        "HIGH_CONVICTION": "Strong score watch",
        "NEW_BUY_LIST": "New research list",
        "REGIME_CHANGE": "Market regime update",
        "PRICE_BREAK": "Price level break",
        "VOLUME_SPIKE": "Volume spike",
        "VOLATILITY_REGIME_SHIFT": "Volatility change",
        "SPREAD_WIDENING": "Spread widening",
        "CORRELATION_BREAK": "Correlation break",
        "NEWS_SHOCK": "News shock",
        "EARNINGS_SURPRISE": "Earnings surprise",
        "RISK_LIMIT_BREACH": "Risk limit breach",
    }
    for raw, friendly in mapping.items():
        if raw in text:
            return friendly
    return _today_plain(value, 80)


def _today_action_from_monitor(row) -> str:
    kind = str(row.get("monitor", "") or "").upper()
    action = str(row.get("action", "") or "")
    if "RISK_LIMIT" in kind:
        return "Do not add exposure. Open Risk and make the book safer first."
    if "PRICE_BREAK" in kind:
        return "Check the price level and risk rule before any paper action."
    if "NEWS_SHOCK" in kind:
        return "Open News and prove the headline, timing, and price reaction."
    if "VOLUME_SPIKE" in kind:
        return "Check whether volume confirms the move or just marks noise."
    if "SPREAD" in kind:
        return "Check trading cost before any paper size."
    return _today_plain(action or "Review this alert before any action.", 160)


def _today_first_answer(brief: dict, command: dict, risk_state: dict) -> tuple[str, str]:
    answer = _plain_status(
        brief.get("desk_answer") or command.get("plain_answer"),
        "Run the daily system, then read this page first.",
    )
    risk_answer = _plain_status(
        brief.get("risk_answer") or risk_state.get("logic"),
        "Risk must be checked before any idea.",
    )
    return _today_plain(answer, 260), _today_plain(risk_answer, 260)


def _render_today_next_clicks(next_clicks: pd.DataFrame):
    if next_clicks is None or next_clicks.empty:
        st.info("No daily workflow order is available. Run the daily system first.")
        return

    st.markdown("#### Today's workflow")
    st.caption("Do these in order. This is the daily operating path, not a trade list.")

    for idx, (_, row) in enumerate(next_clicks.head(5).iterrows(), start=1):
        page = _plain_status(row.get("page"), "Home")
        panel = _home_panel_label(row.get("panel"))
        task = _today_plain(row.get("what_to_read"), 185)
        why = _today_plain(row.get("why_now"), 185)
        done = _today_plain(row.get("done_when"), 185)
        avoid = _today_plain(row.get("do_not_do"), 185)
        accent = "#991b1b" if page == "Risk" else "#334155" if page in {"Home", "Ideas"} else "#0f766e"
        open_label = "Stay here" if page == "Home" else f"Open {page}"
        if panel == "Risk check":
            task = "Check the tickers risk does not allow yet. Decide whether each is watch-only, tiny paper, or not ready."
            why = "Safety is the stop sign. A score, headline, or option idea cannot override it."
            done = "You know which names must stay smaller or blocked."
            avoid = "Do not let a good headline override loss limits, concentration, or jump risk."
        elif panel == "Trading cost check":
            task = "Check whether trading would be too expensive or hard to fill."
            why = "A good signal can disappear if spread, volume, or fill risk is bad."
            done = "A manual quote or better intraday source confirms trading cost and volume."
            avoid = "Do not size a ticker with missing trading-cost proof."
        _render_html(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:15px 17px; margin:0 0 12px 0;">
              <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
                <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">Step {idx} / {page}</div>
                <a href="{_page_href(page)}" target="_self" style="font-size:12px; color:#111827; font-weight:900; text-decoration:underline;">{_esc(open_label)}</a>
              </div>
              <div style="font-size:19px; color:#111827; font-weight:900; line-height:1.25; margin-top:7px;">{_esc(panel)}</div>
              <div style="font-size:14px; color:#111827; line-height:1.45; margin-top:8px;"><b>Do this:</b> {_esc(task)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:6px;"><b>Why:</b> {_esc(why)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:6px;"><b>Done when:</b> {_esc(done)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.4;"><b>Do not:</b> {_esc(avoid)}</div>
            </div>
            """
        )


def _render_today_monitor_cards(events: pd.DataFrame, alerts: list[dict]):
    st.markdown("#### What changed today")
    st.caption("These are alerts to investigate. They are not automatic trade signals.")

    cards: list[dict] = []
    if events is not None and not events.empty:
        work = events.copy()
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        if "severity" in work.columns:
            work["_sev"] = work["severity"].astype(str).str.upper().map(severity_order).fillna(9)
            work = work.sort_values(["_sev", "ticker"], na_position="last")
        for _, row in work.head(8).iterrows():
            cards.append({
                "severity": row.get("severity"),
                "kind": row.get("monitor"),
                "title": row.get("title"),
                "detail": row.get("detail"),
                "action": _today_action_from_monitor(row),
                "ticker": row.get("ticker"),
                "source": row.get("source_provider") or row.get("source_file"),
            })
    elif alerts:
        for item in alerts[:8]:
            cards.append({
                "severity": item.get("priority"),
                "kind": item.get("type"),
                "title": item.get("title"),
                "detail": item.get("detail"),
                "action": item.get("action"),
                "ticker": ", ".join(item.get("tickers") or []),
                "source": "daily alerts",
            })

    if not cards:
        st.info("No daily alerts are available yet.")
        return

    html = ['<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin:8px 0 18px 0;">']
    for card in cards[:6]:
        severity = _today_severity_label(card.get("severity"))
        kind = _today_alert_kind(card.get("kind"))
        accent = _today_accent(card.get("severity"), card.get("kind"))
        ticker = _clean_display(card.get("ticker"), "Portfolio")
        if ticker.lower() in {"nan", "none", "no data", ""}:
            ticker = "Portfolio"
        title = _today_plain(card.get("title"), 145)
        detail = _today_plain(card.get("detail"), 170)
        action = _today_plain(card.get("action"), 160)
        source = _today_plain(card.get("source"), 90)
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 15px; min-height:245px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div style="font-size:12px; color:{accent}; font-weight:900; text-transform:uppercase;">{_esc(severity)}</div>
                <div style="font-size:12px; color:#111827; font-weight:900; text-align:right;">{_esc(ticker)}</div>
              </div>
              <div style="font-size:17px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(kind)}: {_esc(title)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:8px;">{_esc(detail)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:13px; color:#111827; line-height:1.42;"><b>Do this:</b> {_esc(action)}</div>
              <div style="font-size:11px; color:#9ca3af; margin-top:7px;">Source: {_esc(source)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_today_focus_cards(queue: pd.DataFrame):
    if queue is None or queue.empty:
        return

    st.markdown("#### First blockers")
    st.caption("These are the first things that stop the system from trusting new ideas.")
    work = queue.head(6).copy()
    html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:8px 0 18px 0;">']
    for _, row in work.iterrows():
        bucket = _today_plain(row.get("bucket"), 70)
        ticker = _plain_status(row.get("ticker"), "Portfolio")
        if ticker.lower() in {"nan", "none", "no data", ""}:
            ticker = "Portfolio"
        task = _today_plain(row.get("plain_task"), 160)
        why = _today_plain(row.get("why"), 160)
        avoid = _today_plain(row.get("do_not_do"), 130)
        source = _friendly_source_label(row.get("source"))
        accent = "#991b1b" if "risk" in bucket.lower() else "#334155"
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {accent}; border-radius:8px; padding:13px 14px; min-height:235px;">
              <div style="display:flex; justify-content:space-between; gap:10px;">
                <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">{_esc(bucket)}</div>
                <div style="font-size:13px; color:#111827; font-weight:900;">{_esc(ticker)}</div>
              </div>
              <div style="font-size:15px; color:#111827; font-weight:900; line-height:1.3; margin-top:8px;">{_esc(task)}</div>
              <div style="font-size:12px; color:#374151; line-height:1.4; margin-top:8px;">{_esc(why)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.35;"><b>Do not:</b> {_esc(avoid)}</div>
              <div style="font-size:11px; color:#9ca3af; margin-top:6px;">Source: {_esc(source)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _today_gate_label(value) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return "No clear status yet"
    if any(x in text for x in ["REDUCE_ONLY", "SIZE_DOWN", "RISK_REPAIR", "BREACH"]):
        return "Risk says smaller or no new buying"
    if any(x in text for x in ["MISSING", "DATA_GAP"]):
        return "Needs missing data fixed"
    if any(x in text for x in ["UNPROVEN", "PROOF", "REVIEW", "LOW_SAMPLE"]):
        return "Needs proof before action"
    if any(x in text for x in ["NO_NEW_OPTION", "NO OPTION"]):
        return "No option idea yet"
    if any(x in text for x in ["CLEAR", "OK"]):
        return "Clear"
    if "BLOCKED" in text or "NO_GO" in text:
        return "Not allowed yet"
    return _today_plain(value, 95)


def _today_status_accent(*values) -> str:
    text = " ".join(str(v or "") for v in values).upper()
    if any(x in text for x in ["REDUCE", "SIZE_DOWN", "BLOCK", "BREACH", "RISK_REPAIR", "HIGH"]):
        return "#991b1b"
    if any(x in text for x in ["REVIEW", "WATCH", "PROOF", "MISSING", "DATA_GAP"]):
        return "#334155"
    if any(x in text for x in ["CLEAR", "OK", "DONE"]):
        return "#166534"
    return "#111827"


def _today_action_sentence(row) -> str:
    ticker = _plain_status(row.get("ticker"), "This ticker")
    bucket = _today_plain(row.get("workflow_bucket"), 90).lower()
    task = _today_plain(row.get("what_to_do"), 170)
    risk = _today_gate_label(row.get("risk_action"))
    option = _today_plain(row.get("option_route"), 120)
    trigger = _today_plain(row.get("what_to_watch"), 130)

    if "risk first" in bucket:
        return f"Open {ticker} only to check risk repair. Do not add size, calls, or puts yet."
    if "tiny research" in bucket:
        return f"{ticker} is research-only. Keep any paper idea tiny until risk, source proof, and price trigger line up."
    if "watch" in bucket:
        return f"Watch {ticker}. Wait for price and volume confirmation before studying stock, call, or put."
    if task and task != "No data":
        return task
    return f"Review {ticker}. Current check: {risk}. Option choice: {option}. Watch: {trigger}."


def _render_today_command_board(
    brief: dict,
    command: dict,
    risk_state: dict,
    flow_state: dict,
    workflow_state: dict,
    monitor_summary: dict,
    alerts: list[dict],
):
    answer, risk_answer = _today_first_answer(brief, command, risk_state)
    mode = _today_plain(command.get("today_mode") or flow_state.get("today_mode") or brief.get("risk_mode"), 90)
    first_page = _today_plain(command.get("first_page") or flow_state.get("first_page"), 60)
    first_ticker = _today_plain(command.get("first_ticker") or flow_state.get("first_ticker"), 60)
    first_action = _today_plain(command.get("first_action") or flow_state.get("plain_answer"), 280)
    can_risk = _today_plain(command.get("can_take_new_risk") or flow_state.get("can_take_new_risk"), 80)

    proof_first = int(_to_float(command.get("proof_first_count"), 0) or 0)
    risk_blocked = int(_to_float(command.get("risk_blocked_count"), 0) or 0)
    hard = int(_to_float(brief.get("hard_breaches"), risk_state.get("budget_hard_breach_count", 0)) or 0)
    size_down = int(_to_float(brief.get("size_down_breaches"), workflow_state.get("risk_first_rows", 0)) or 0)
    workflow_steps = int(_to_float(workflow_state.get("workflow_steps"), 0) or 0)
    queue_rows = int(_to_float(workflow_state.get("queue_rows"), 0) or 0)
    total_events = int(_to_float(monitor_summary.get("total_events"), len(alerts)) or 0)
    critical = int(_to_float(monitor_summary.get("critical_count"), 0) or 0)
    score_now = _to_float(brief.get("proof_adjusted_sharpe"))
    score_text = f"{score_now:.2f}" if score_now is not None else "No data"

    _is_risk_mode = "risk" in answer.lower() or "no new" in can_risk.lower()
    _hero_bg = "linear-gradient(135deg,#450a0a 0%,#7f1d1d 100%)" if _is_risk_mode else "linear-gradient(135deg,#0f172a 0%,#1e293b 100%)"
    _render_html(
        f"""
        <div style="
            background:{_hero_bg};
            border-radius:14px;
            padding:24px 28px 20px 28px;
            margin:10px 0 20px 0;
            box-shadow:0 4px 20px rgba(15,23,42,.2);
            position:relative;
            overflow:hidden;
        ">
          <div style="position:absolute;top:0;right:0;width:200px;height:200px;background:radial-gradient(circle at 100% 0%,rgba(255,255,255,.05) 0%,transparent 65%);pointer-events:none;"></div>
          <div style="font-size:10px; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:.1em;">Today · {_esc(mode)}</div>
          <div style="font-size:1.9rem; color:#f8fafc; font-weight:900; line-height:1.15; margin-top:8px; letter-spacing:-0.4px;">{_esc(answer)}</div>
          <div style="height:1px; background:rgba(255,255,255,.08); margin:14px 0 12px 0;"></div>
          <div style="display:flex; gap:28px; flex-wrap:wrap; align-items:flex-start;">
            <div>
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">First click</div>
              <div style="font-size:13px; color:#93c5fd; font-weight:700; margin-top:3px;">{_esc(first_page)} → {_esc(first_ticker)}</div>
            </div>
            <div style="flex:1; min-width:160px;">
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Risk says</div>
              <div style="font-size:13px; color:#fca5a5; font-weight:600; margin-top:3px; line-height:1.4;">{_esc(risk_answer)}</div>
            </div>
            <div>
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Hard breaks</div>
              <div style="font-size:16px; color:{'#fca5a5' if hard else '#86efac'}; font-weight:800; margin-top:2px;">{hard}</div>
            </div>
            <div>
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Trust</div>
              <div style="font-size:16px; color:#e2e8f0; font-weight:800; margin-top:2px;">{score_text}</div>
            </div>
          </div>
          <div style="margin-top:12px; font-size:11px; color:#334155;">Research only · No broker · No live orders</div>
        </div>
        """
    )

    cards = [
        ("Can I add now?", "No" if "no" in can_risk.lower() or "risk" in answer.lower() else "Maybe", "If this says No, skip calls, puts, and new size.", "#dc2626" if _is_risk_mode else "#16a34a"),
        ("Missing proof", str(proof_first), "Source checks needed before trusting model text.", "#dc2626" if proof_first else "#16a34a"),
        ("Safety blockers", f"{risk_blocked} blocked / {size_down} smaller", f"{hard} hard limits need attention.", "#dc2626" if risk_blocked or hard else "#16a34a"),
        ("Work queue", f"{workflow_steps} steps / {queue_rows} tickers", "Step through in order.", "#475569"),
        ("Live alerts", f"{total_events} total", f"{critical} critical.", "#dc2626" if critical else "#475569"),
        ("Trust score", score_text, "Quality-adjusted. Not a return forecast.", "#475569"),
    ]
    cols = st.columns(3)
    for idx, (title, value, note, color) in enumerate(cards):
        with cols[idx % 3]:
            _simple_card(title, value, note, color)


def _render_momentum_signal_panel():
    """Show top / bottom momentum tickers from step127 output — plain English."""
    mom = safe_csv(ROOT / "momentum_scores.csv")
    if mom.empty or "momentum_score" not in mom.columns or "ticker" not in mom.columns:
        return

    regime = str(mom["regime"].iloc[0]) if "regime" in mom.columns else "Unknown"
    vix_val = _to_float(mom["vix"].iloc[0]) if "vix" in mom.columns else None
    damp = bool(mom["regime_dampened"].iloc[0]) if "regime_dampened" in mom.columns else False
    mult = _to_float(mom["total_damp_mult"].iloc[0]) if "total_damp_mult" in mom.columns else 1.0

    regime_colors = {"BULL": "#166534", "LATE_BULL": "#854d0e", "SIDEWAYS": "#334155", "BEAR": "#991b1b"}
    regime_labels = {"BULL": "Bull — momentum works well", "LATE_BULL": "Late Bull — fade slowly",
                     "SIDEWAYS": "Sideways — cross-sectional still OK", "BEAR": "Bear — momentum danger zone"}
    accent = regime_colors.get(regime, "#334155")
    regime_label = regime_labels.get(regime, regime)
    vix_str = f"VIX {vix_val:.1f}" if vix_val is not None else "VIX unknown"
    damp_note = f"  ·  Crash protection ON (×{mult:.2f})" if damp else ""

    with st.expander(f"Momentum radar — {regime_label}  ·  {vix_str}{damp_note}", expanded=False):
        st.caption(
            "Price momentum snapshot (Step 127). "
            "Scores are cross-sectional ranks 0–100 across the S&P 500. "
            "Based on Jegadeesh & Titman (12-1 month), 52-week high, and vol-scaled return. "
            "Research only — not a buy or sell instruction."
        )
        top10 = mom.nlargest(10, "momentum_score")[["ticker", "momentum_score", "sub_cs_mom", "sub_high52", "sub_vol_scaled"]].reset_index(drop=True)
        bot5  = mom.nsmallest(5, "momentum_score")[["ticker", "momentum_score"]].reset_index(drop=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("**Top 10 — strongest price momentum**")
            for _, row in top10.iterrows():
                score = _to_float(row.get("momentum_score"), 50)
                bar_w = max(4, int(score))
                cs    = _to_float(row.get("sub_cs_mom"))
                h52   = _to_float(row.get("sub_high52"))
                vs    = _to_float(row.get("sub_vol_scaled"))
                detail = "  ·  ".join([
                    f"12m {cs:.0f}" if cs is not None else "",
                    f"52wH {h52:.0f}" if h52 is not None else "",
                    f"vol-adj {vs:.0f}" if vs is not None else "",
                ]).strip("  ·  ")
                _render_html(
                    f"""
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                      <div style="width:52px; font-size:13px; font-weight:900; color:#111827;">{_esc(str(row['ticker']))}</div>
                      <div style="flex:1; background:#f3f4f6; border-radius:4px; height:14px; overflow:hidden;">
                        <div style="width:{bar_w}%; background:{accent}; height:14px; border-radius:4px;"></div>
                      </div>
                      <div style="width:38px; font-size:12px; color:#374151; text-align:right;">{score:.0f}</div>
                      <div style="font-size:11px; color:#6b7280; min-width:120px;">{_esc(detail)}</div>
                    </div>
                    """
                )
        with c2:
            st.markdown("**Avoid list (lowest momentum)**")
            for _, row in bot5.iterrows():
                score = _to_float(row.get("momentum_score"), 0)
                _render_html(
                    f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; padding:4px 8px; background:#fef2f2; border-radius:5px;">
                      <span style="font-size:13px; font-weight:900; color:#991b1b;">{_esc(str(row['ticker']))}</span>
                      <span style="font-size:12px; color:#991b1b;">{score:.0f}</span>
                    </div>
                    """
                )
            st.caption("Low scores = weak or falling momentum. Avoid adding size here unless there's another strong reason.")

        # data freshness note
        mom_path = ROOT / "momentum_scores.csv"
        if mom_path.exists():
            age_h = (time.time() - mom_path.stat().st_mtime) / 3600
            age_label = f"{age_h:.1f}h ago" if age_h < 24 else f"{age_h/24:.0f}d ago"
            st.caption(f"Data: momentum_scores.csv  ·  last updated {age_label}  ·  {len(mom)} tickers scored")


def _render_today_station_flow(workflow_steps: pd.DataFrame):
    if workflow_steps is None or workflow_steps.empty:
        st.info("No daily steps are available yet. Run the daily system first.")
        return

    st.markdown("#### Today's steps")
    st.caption("Do these in order. More items can appear; the layout is not limited to four.")
    work = workflow_steps.copy()
    if "step_order" in work.columns:
        work["_order"] = pd.to_numeric(work["step_order"], errors="coerce").fillna(999)
        work = work.sort_values("_order")

    html = ['<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; margin:8px 0 18px 0;">']
    for _, row in work.iterrows():
        station = _today_plain(row.get("station"), 70)
        status = _today_gate_label(row.get("status"))
        items = _today_plain(row.get("items"), 45)
        task = _today_plain(row.get("what_to_do"), 150)
        why = _today_plain(row.get("why_this_exists"), 145)
        page = _today_plain(row.get("next_dashboard_section"), 70)
        accent = _today_status_accent(row.get("status"), row.get("station"), row.get("what_to_do"))
        status_bg = "#fef2f2" if accent == "#dc2626" else "#f0fdf4" if accent in {"#0f766e", "#166534"} else "#f8fafc"
        status_color = accent
        html.append(
            f"""
            <div style="
                background:#ffffff;
                border:1px solid #e2e8f0;
                border-top:3px solid {accent};
                border-radius:10px;
                padding:15px 16px;
                min-height:230px;
                box-shadow:0 1px 4px rgba(0,0,0,.06);
            ">
              <div style="display:flex; justify-content:space-between; gap:8px; align-items:center; margin-bottom:8px;">
                <span style="font-size:10px; color:{status_color}; font-weight:700; text-transform:uppercase; letter-spacing:.06em; background:{status_bg}; padding:2px 8px; border-radius:99px;">{_esc(status)}</span>
                <span style="font-size:11px; color:#94a3b8; font-weight:600;">{_esc(items)}</span>
              </div>
              <div style="font-size:17px; color:#0f172a; font-weight:800; line-height:1.25;">{_esc(station)}</div>
              <div style="font-size:12.5px; color:#1e293b; line-height:1.45; margin-top:9px;">{_esc(task)}</div>
              <div style="font-size:11.5px; color:#64748b; line-height:1.4; margin-top:6px;">{_esc(why)}</div>
              <div style="border-top:1px solid #f1f5f9; margin-top:10px; padding-top:7px; font-size:11px; color:#94a3b8;">Open: {_esc(page)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_today_ticker_queue_cards(workflow_queue: pd.DataFrame, readiness: pd.DataFrame):
    if workflow_queue is None or workflow_queue.empty:
        st.info("No ticker queue is available yet. Run the daily system first.")
        return

    st.markdown("#### Tickers to open first")
    st.caption("These are not buy recommendations. They are the tickers whose blockers should be understood first.")
    work = workflow_queue.copy()
    if "priority_rank" in work.columns:
        work["_rank"] = pd.to_numeric(work["priority_rank"], errors="coerce").fillna(999)
        work = work.sort_values(["_rank", "ticker"])

    readiness_map = {}
    if readiness is not None and not readiness.empty and "ticker" in readiness.columns:
        readiness_map = {str(row.get("ticker")): row for _, row in readiness.iterrows()}

    for start in range(0, min(len(work), 12), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, work.iloc[start:start + 4].iterrows()):
            ticker = _plain_status(row.get("ticker"), "Ticker")
            sector = _today_plain(row.get("sector"), 60)
            bucket = _today_plain(row.get("workflow_bucket"), 80)
            horizon = _today_plain(row.get("best_horizon"), 60)
            action = _today_action_sentence(row)
            risk = _today_gate_label(row.get("risk_action"))
            event = _today_gate_label(row.get("event_gate"))
            option = _today_plain(row.get("option_route"), 105)
            watch = _today_plain(row.get("what_to_watch"), 115)
            change = _today_plain(row.get("what_would_change"), 120)
            source = _friendly_source_label(row.get("source_files"))
            ready_row = readiness_map.get(str(ticker))
            readiness_score = _fmt_target_number(ready_row.get("readiness_score"), 1) if ready_row is not None else "No data"
            first_gate = _today_plain(ready_row.get("first_blocking_gate"), 80) if ready_row is not None else "No data"
            accent = _today_status_accent(row.get("risk_action"), row.get("workflow_bucket"), row.get("priority"))
            status_bg = "#fef2f2" if accent == "#dc2626" else "#f0fdf4" if accent in {"#0f766e","#166534"} else "#f8fafc"
            with col:
                _render_html(
                    f"""
                    <div style="
                        background:#ffffff;
                        border:1px solid #e2e8f0;
                        border-radius:12px;
                        padding:16px 16px 14px 16px;
                        min-height:400px;
                        margin-bottom:12px;
                        box-shadow:0 2px 8px rgba(0,0,0,.07);
                    ">
                      <div style="display:flex; justify-content:space-between; gap:8px; align-items:flex-start; margin-bottom:10px;">
                        <div style="font-size:28px; color:#0f172a; font-weight:900; line-height:1; letter-spacing:-0.5px;">{_esc(ticker)}</div>
                        <div style="text-align:right;">
                          <div style="font-size:11px; color:#64748b; font-weight:600;">{_esc(sector)}</div>
                          <div style="font-size:10px; color:#94a3b8; margin-top:2px;">{_esc(horizon)}</div>
                        </div>
                      </div>
                      <div style="margin-bottom:10px;">
                        <span style="font-size:10px; color:{accent}; font-weight:700; text-transform:uppercase; letter-spacing:.06em; background:{status_bg}; padding:3px 9px; border-radius:99px;">{_esc(bucket)}</span>
                      </div>
                      <div style="font-size:14px; color:#0f172a; font-weight:700; line-height:1.4;">{_esc(action)}</div>
                      <div style="height:1px; background:#f1f5f9; margin:11px 0;"></div>
                      <div style="font-size:12px; color:#475569; line-height:1.45; margin-top:4px;"><span style="font-weight:700; color:#334155;">Risk:</span> {_esc(risk)}</div>
                      <div style="font-size:12px; color:#475569; line-height:1.45; margin-top:5px;"><span style="font-weight:700; color:#334155;">Event:</span> {_esc(event)}</div>
                      <div style="font-size:12px; color:#475569; line-height:1.45; margin-top:5px;"><span style="font-weight:700; color:#334155;">Options:</span> {_esc(option)}</div>
                      <div style="font-size:12px; color:#1e293b; line-height:1.45; margin-top:7px;"><span style="font-weight:700;">Watch:</span> {_esc(watch)}</div>
                      <div style="font-size:11.5px; color:#64748b; line-height:1.4; margin-top:5px;">{_esc(change)}</div>
                      <div style="height:1px; background:#f1f5f9; margin:10px 0 8px 0;"></div>
                      <div style="font-size:11px; color:#94a3b8; line-height:1.4;">Ready: {_esc(readiness_score)} · Blocked by: {_esc(first_gate)}</div>
                      <div style="font-size:10px; color:#cbd5e1; margin-top:4px;">{_esc(source)}</div>
                    </div>
                    """
                )


def _render_today_proof_plan(plan: pd.DataFrame):
    if plan is None or plan.empty:
        return

    st.markdown("#### Missing proof and fixes")
    st.caption("This tells you what must be fixed before the dashboard can promote a ticker to a real research candidate.")
    html = ['<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin:8px 0 18px 0;">']
    work = plan.copy()
    if "work_order" in work.columns:
        work["_order"] = pd.to_numeric(work["work_order"], errors="coerce").fillna(999)
        work = work.sort_values("_order")
    for _, row in work.head(4).iterrows():
        name = _today_plain(row.get("station_name"), 90)
        tickers = _today_plain(row.get("tickers_to_check"), 130)
        do_this = _today_plain(row.get("do_this"), 170)
        success = _today_plain(row.get("success_condition"), 175)
        why = _today_plain(row.get("why_it_matters"), 175)
        rerun = _today_plain(row.get("rerun_after"), 130)
        avoid = _today_plain(row.get("do_not_do"), 150)
        source = _friendly_source_label(row.get("open_source"))
        accent = _today_status_accent(name, do_this)
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:15px 16px; min-height:280px;">
              <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">{_esc(name)}</div>
              <div style="font-size:15px; color:#111827; font-weight:900; line-height:1.35; margin-top:8px;">{_esc(do_this)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:8px;"><b>Tickers:</b> {_esc(tickers)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:6px;"><b>Success means:</b> {_esc(success)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:6px;"><b>Why:</b> {_esc(why)}</div>
              <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.35;"><b>After fixing:</b> {_esc(rerun)}</div>
              <div style="font-size:12px; color:#6b7280; line-height:1.35; margin-top:6px;"><b>Do not:</b> {_esc(avoid)}</div>
              <div style="font-size:10px; color:#9ca3af; margin-top:6px;">Source: {_esc(source)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def _render_market_context_strip():
    """
    Compact market pulse bar — VIX · Regime · Yield curve · Top 3 momentum · Bottom 3 avoid.
    Reads: macro_signals.json, momentum_scores.csv
    """
    macro = safe_json(ROOT / "macro_signals.json")
    mom   = safe_csv(ROOT / "momentum_scores.csv")

    vix       = _to_float(macro.get("vix"))
    regime_j  = safe_json(ROOT / "regime_current.json")
    regime    = str(regime_j.get("regime") or macro.get("regime") or "—")
    yc_signal = str(macro.get("yield_curve_signal") or "—")
    credit    = str(macro.get("credit_signal") or "—")
    yield_10y = _to_float(macro.get("yield_10y"))

    # VIX styling
    if vix is not None:
        if vix > 30:   vix_color, vix_label = "#dc2626", "HIGH"
        elif vix > 20: vix_color, vix_label = "#d97706", "ELEVATED"
        else:          vix_color, vix_label = "#16a34a", "LOW"
        vix_str = f"{vix:.1f}"
    else:
        vix_color, vix_label, vix_str = "#64748b", "—", "—"

    regime_colors = {"BULL":"#16a34a","LATE_BULL":"#d97706","SIDEWAYS":"#475569","BEAR":"#dc2626"}
    regime_color  = regime_colors.get(regime, "#475569")

    # Top 3 momentum / bottom 3 avoid
    top3_html = bot3_html = ""
    if not mom.empty and "momentum_score" in mom.columns:
        top3 = mom.nlargest(3, "momentum_score")[["ticker","momentum_score"]]
        bot3 = mom.nsmallest(3, "momentum_score")[["ticker","momentum_score"]]
        top3_html = "  ".join(
            f'<span style="background:#dcfce7;color:#166534;font-weight:700;font-size:11px;padding:2px 7px;border-radius:99px;">{r.ticker} {r.momentum_score:.0f}</span>'
            for r in top3.itertuples()
        )
        bot3_html = "  ".join(
            f'<span style="background:#fef2f2;color:#991b1b;font-weight:700;font-size:11px;padding:2px 7px;border-radius:99px;">{r.ticker} {r.momentum_score:.0f}</span>'
            for r in bot3.itertuples()
        )

    _render_html(
        f"""
        <div style="
            background:#fff;
            border:1px solid #e2e8f0;
            border-radius:10px;
            padding:12px 18px;
            margin:0 0 16px 0;
            display:flex;
            flex-wrap:wrap;
            gap:20px;
            align-items:center;
            box-shadow:0 1px 3px rgba(0,0,0,.05);
        ">
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">VIX</div>
            <div style="font-size:18px;font-weight:800;color:{vix_color};margin-top:2px;">{vix_str} <span style="font-size:10px;font-weight:700;vertical-align:middle;">{vix_label}</span></div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Regime</div>
            <div style="font-size:15px;font-weight:800;color:{regime_color};margin-top:2px;">{_esc(regime)}</div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Yield curve</div>
            <div style="font-size:13px;font-weight:700;color:#475569;margin-top:3px;">{_esc(yc_signal)}  {f'{yield_10y:.2f}%' if yield_10y else ''}</div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Credit</div>
            <div style="font-size:13px;font-weight:700;color:#475569;margin-top:3px;">{_esc(credit)}</div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#16a34a;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Momentum leaders</div>
            <div style="margin-top:4px;">{top3_html}</div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#991b1b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Avoid (weak momentum)</div>
            <div style="margin-top:4px;">{bot3_html}</div>
          </div>
        </div>
        """
    )


def _signal_dot_html(score: float | None, label: str) -> str:
    """
    Single coloured dot + label for the alpha signal mini-bar.
    Green >65 · Gray 40-65 · Red <40 · White = missing
    """
    if score is None or (isinstance(score, float) and score != score):
        return (
            f'<div style="text-align:center;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:#e2e8f0;margin:0 auto 2px auto;"></div>'
            f'<div style="font-size:9px;color:#cbd5e1;">{_esc(label)}</div></div>'
        )
    if score >= 65:   color = "#16a34a"
    elif score >= 40: color = "#94a3b8"
    else:             color = "#dc2626"
    return (
        f'<div style="text-align:center;">'
        f'<div style="width:10px;height:10px;border-radius:50%;background:{color};margin:0 auto 2px auto;"></div>'
        f'<div style="font-size:9px;color:{color};font-weight:700;">{score:.0f}</div>'
        f'<div style="font-size:8px;color:#94a3b8;margin-top:1px;">{_esc(label)}</div></div>'
    )


def _alpha_signal_mini_bar(ticker: str, alpha_map: dict) -> str:
    """
    Compact row of 5 signal dots + alpha rank pill for a ticker.
    alpha_map: { ticker → row dict from alpha_scores.csv }
    """
    row = alpha_map.get(str(ticker))
    if not row:
        return ""
    alpha_score = _to_float(row.get("alpha_score"))
    alpha_rank  = row.get("alpha_rank")
    sigs = {
        "ML":   _to_float(row.get("sig_regime_ml")),
        "Mom":  _to_float(row.get("sig_momentum")),
        "Qual": _to_float(row.get("sig_quality")),
        "Sent": _to_float(row.get("sig_sentiment")),
        "Ins":  _to_float(row.get("sig_insider")),
    }
    dots_html = "".join(_signal_dot_html(v, k) for k, v in sigs.items())
    rank_str = f"#{int(alpha_rank)}" if alpha_rank and str(alpha_rank) not in {"nan","None",""} else ""
    score_color = "#16a34a" if alpha_score and alpha_score >= 65 else "#dc2626" if alpha_score and alpha_score < 45 else "#475569"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;padding:7px 10px;background:#f8fafc;border-radius:8px;">'
        f'<div style="font-size:11px;font-weight:800;color:{score_color};min-width:36px;">α {alpha_score:.0f}</div>'
        f'<div style="font-size:10px;color:#94a3b8;min-width:24px;">{_esc(rank_str)}</div>'
        f'<div style="flex:1;display:flex;gap:10px;justify-content:flex-end;">{dots_html}</div>'
        f'</div>'
    )


def tab_today_workflow():
    brief = safe_json(ROOT / "pm_morning_brief_state.json")
    command = safe_json(ROOT / "quant_fund_flow_pm_command_center.json")
    flow_state = safe_json(ROOT / "quant_fund_flow_navigator_state.json")
    workflow_state = safe_json(ROOT / "daily_workflow_state.json")
    risk_state = safe_json(ROOT / "risk_desk_overview.json")
    next_clicks = safe_csv(ROOT / "quant_fund_flow_next_clicks.csv")
    blockers = safe_csv(ROOT / "quant_fund_flow_blocker_queue.csv")
    current = safe_csv(ROOT / "quant_fund_flow_current_state.csv")
    focus_queue = safe_csv(ROOT / "pm_morning_brief_focus_queue.csv")
    monitor_summary = safe_json(ROOT / "desk_monitor_summary.json")
    monitor_events = safe_csv(ROOT / "desk_monitor_events.csv")
    alert_state = safe_json(ROOT / "daily_alerts.json")
    workflow_steps = safe_csv(ROOT / "daily_workflow_steps.csv")
    workflow_queue = safe_csv(ROOT / "daily_workflow_queue.csv")
    workflow_explain = safe_csv(ROOT / "daily_workflow_ticker_explain.csv")
    readiness = safe_csv(ROOT / "action_readiness_monitor.csv")
    proof_plan = safe_csv(ROOT / "proof_queue_daily_plan.csv")

    alerts = alert_state.get("alerts", []) if isinstance(alert_state.get("alerts"), list) else []

    st.markdown('<p class="section-title">Today: what should I do first?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This is the daily operating page. Read it from top to bottom. It tells you what changed, what blocks action, and where to click next.</p>',
        unsafe_allow_html=True,
    )
    _render_section_depth("Today")
    _render_market_context_strip()

    _render_today_command_board(brief, command, risk_state, flow_state, workflow_state, monitor_summary, alerts)
    _render_today_next_clicks(next_clicks)
    _render_momentum_signal_panel()
    _render_today_station_flow(workflow_steps)
    _render_today_ticker_queue_cards(workflow_queue, readiness)
    _render_today_proof_plan(proof_plan)
    _render_today_monitor_cards(monitor_events, alerts)

    show_detail = st.checkbox("Show technical daily files", value=False)
    if not show_detail:
        return

    st.markdown("---")
    detail_view = st.radio(
        "Daily detail to open",
        ["Daily steps", "Blockers", "Ticker states", "Alerts", "Source files"],
        horizontal=True,
        label_visibility="collapsed",
    )
    _render_subtab_depth("Today", detail_view)
    if detail_view == "Daily steps":
        st.markdown("#### Daily workflow steps")
        s_cols = [c for c in ["step_order", "status", "station", "items", "what_to_do", "why_this_exists", "next_dashboard_section"] if c in workflow_steps.columns]
        _show_status_table(workflow_steps[s_cols] if s_cols else workflow_steps, ["status"], height=520)
        st.markdown("#### Daily ticker queue")
        q_cols = [c for c in ["workflow_rank", "priority", "ticker", "sector", "master_action", "why", "what_to_watch", "what_would_change", "next_dashboard_section"] if c in workflow_queue.columns]
        _show_status_table(workflow_queue[q_cols].head(60) if q_cols else workflow_queue.head(60), ["priority", "master_action"], height=650)
    elif detail_view == "Blockers":
        b_cols = [c for c in ["priority", "ticker", "blocker_type", "blocker", "what_to_do", "where_to_click", "why_it_matters", "source_files"] if c in blockers.columns]
        _show_status_table(blockers[b_cols].head(120) if b_cols else blockers.head(120), [], height=760)
    elif detail_view == "Ticker states":
        state_cols = [c for c in ["ticker", "current_state", "operating_mode", "can_take_new_risk", "first_blocker", "next_click", "next_action", "stock_or_etf_route", "option_route", "short_term_route", "medium_term_route", "long_term_route", "why", "trigger_to_watch"] if c in current.columns]
        _show_status_table(current[state_cols].head(120) if state_cols else current.head(120), [], height=760)
        if not workflow_explain.empty:
            st.markdown("#### Plain ticker explanations")
            e_cols = [c for c in ["ticker", "plain_english_summary", "risk_evidence", "event_evidence", "sector_evidence", "option_evidence"] if c in workflow_explain.columns]
            _show_status_table(workflow_explain[e_cols].head(60) if e_cols else workflow_explain.head(60), [], height=620)
        if not readiness.empty:
            st.markdown("#### Readiness monitor")
            r_cols = [c for c in ["ticker", "current_stage", "readiness_score", "first_blocking_gate", "nearest_clear_condition", "route_after_all_gates_clear", "option_permission_after_repair", "trigger_to_watch"] if c in readiness.columns]
            _show_status_table(readiness[r_cols].head(60) if r_cols else readiness.head(60), [], height=620)
    elif detail_view == "Alerts":
        if not monitor_events.empty:
            cols = [c for c in ["date", "monitor", "ticker", "severity", "title", "detail", "action", "source_layer", "source_provider", "source_file"] if c in monitor_events.columns]
            _show_status_table(monitor_events[cols] if cols else monitor_events, ["severity"], height=760)
        elif alerts:
            _show_status_table(pd.DataFrame(alerts), ["priority"], height=760)
        else:
            st.info("No alert table is available.")
    elif detail_view == "Source files":
        _render_risk_source_inventory([
            ("momentum_scores.csv", "Momentum signal — top/bottom tickers (Step 127)"),
            ("spy_price_cache.csv", "SPY benchmark prices used for residual momentum calculation"),
            ("pm_morning_brief_state.json", "Daily PM answer"),
            ("pm_morning_brief_focus_queue.csv", "First blockers"),
            ("pm_morning_brief_news_to_verify.csv", "News proof queue"),
            ("quant_fund_flow_pm_command_center.json", "Next click command center"),
            ("quant_fund_flow_next_clicks.csv", "Daily click path"),
            ("quant_fund_flow_blocker_queue.csv", "Proof and risk blockers"),
            ("quant_fund_flow_current_state.csv", "Ticker state map"),
            ("desk_monitor_summary.json", "Alert summary"),
            ("desk_monitor_events.csv", "Desk monitor events"),
            ("daily_alerts.json", "Daily alert list"),
            ("daily_workflow_steps.csv", "Dynamic workflow steps"),
            ("daily_workflow_queue.csv", "Dynamic ticker queue"),
            ("action_readiness_monitor.csv", "Ticker readiness monitor"),
            ("proof_queue_daily_plan.csv", "Proof and repair plan"),
        ])


def _ideas_plain(value, max_len: int = 190) -> str:
    raw = "" if value is None else str(value)
    raw_replacements = {
        "CALL_BLOCKED_BY_RISK": "No call now because safety risk is too high",
        "HEDGE_ONLY": "Hedge idea only",
        "TINY_STOCK_OR_ETF_PAPER_ONLY": "Tiny stock or ETF paper only",
        "WATCH_EVENT_PROOF_FIRST": "Wait until the news or event is proven",
        "WAIT_EXECUTION_OR_MONITOR_REVIEW": "Wait until trading cost and live alerts calm down",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "Put or hedge research only",
        "DEFINED_RISK_DEBIT_SPREAD_OK_IF_GATES_CLEAR": "Defined-risk debit spread only if all checks clear",
        "DEFINED_RISK_SPREAD_ONLY": "Defined-risk spread only",
        "SPREAD_OR_NO_OPTION": "Use a spread or no option",
        "NO_NEW_EXPOSURE": "Do not add",
        "NO_NEW_OPTION": "No option idea yet",
        "NO_OPTION_BACKTEST": "No useful option test yet",
        "NO_DIRECTIONAL_EVENT": "No clear bullish or bearish news yet",
        "DOWNSIDE_OR_HEDGE": "Downside or hedge idea",
        "UNPROVEN_LOCAL_CONTEXT": "News link is not proven yet",
        "WATCH_ONLY_REQUIRE_PRICE_VOLUME_CONFIRMATION": "Watch only until price and volume confirm",
        "CONTEXT_ONLY_NO_DIRECTIONAL_ACTION": "Context only; no action from the headline",
        "MISSING_DATA_REVIEW": "Missing data review",
        "REVIEW_PROXY_BACKTEST": "Treat the old-data test as a warning, not proof",
        "DEFINED_RISK_DEBIT_SPREAD_OK_IF_GATES_CLEAR": "Defined-risk spread only if all checks clear",
        "DEFINED_RISK_SPREAD_ONLY": "Defined-risk spread only",
        "SIZE_DOWN": "use smaller size",
        "REDUCE_ONLY": "reduce only",
        "CALL_REVIEW": "call review only",
        "PUT_REVIEW": "put review only",
        "NO_GO": "not ready",
        "CRITICAL": "critical",
        "WARNING": "warning",
        "OK": "ok",
        "DATA_GAP": "missing data",
        "NO_DATA": "no data",
        "NONE": "none",
    }
    for key, label in raw_replacements.items():
        raw = raw.replace(key, label)

    text = _human_text(raw, max_len=None)
    readable_replacements = {
        "No New Option": "No option idea yet",
        "Call Blocked By Risk": "No call now because safety risk is too high",
        "Hedge Only": "Hedge idea only",
        "Tiny Paper Only": "Tiny paper only",
        "Tiny Stock Or Etf Paper Only": "Tiny stock or ETF paper only",
        "Watch Event Proof First": "Wait until the news or event is proven",
        "Wait Execution Or Monitor Review": "Wait until trading cost and live alerts calm down",
        "Unproven Local Context": "News link is not proven yet",
        "Context Only No Directional Action": "Context only; no action from the headline",
        "Watch Only Require Price Volume Confirmation": "Watch only until price and volume confirm",
        "No Directional Event": "No clear bullish or bearish news yet",
        "Review Proxy Backtest": "Treat the old-data test as a warning, not proof",
        "Defined Risk Debit Spread Ok If Gates Clear": "Defined-risk spread only if all checks clear",
        "Defined Risk Spread Only": "Defined-risk spread only",
        "Put Or Hedge Research Only": "Put or hedge research only",
        "Spread Or No Option": "Use a spread or no option",
        "No New Exposure": "Do not add",
        "Call Review": "call review only",
        "Put Review": "put review only",
        "stock/ETF": "stock or ETF",
        "Stock/ETF": "stock or ETF",
        "spread, IV, liquidity": "trading cost, option price, and volume",
        "no-go": "not-ready",
        "No-Go": "Not-ready",
        "source/earnings/news": "source, earnings, and news",
        "price/volume": "price and volume",
        "size-down": "smaller size",
        "Size-Down": "Smaller size",
        "exposure": "size",
        "Exposure": "Size",
        "risk gate": "safety check",
        "Risk gate": "Safety check",
        "gates": "checks",
        "Gates": "Checks",
        "Theme-readthrough check": "related-theme check",
        "theme-readthrough check": "related-theme check",
    }
    for raw_text, friendly in readable_replacements.items():
        text = text.replace(raw_text, friendly)
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _ideas_count_contains(df: pd.DataFrame, col: str, needle: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].astype(str).str.contains(needle, case=False, na=False).sum())


def _ideas_horizon_summary(row) -> str:
    short = _ideas_plain(row.get("short_action"), 55)
    medium = _ideas_plain(row.get("medium_action"), 55)
    long = _ideas_plain(row.get("long_action"), 55)
    return f"Short: {short} | Medium: {medium} | Long: {long}"


def _ideas_vehicle_now(row) -> str:
    gate = _ideas_plain(row.get("gate_status"), 80)
    current = _ideas_plain(row.get("current_desk_action"), 120)
    if "tiny paper" in gate.lower():
        return "Tiny stock or ETF paper only. No option yet."
    if "wait" in current.lower():
        return current
    return current if current != "No data" else gate


def _ideas_call_line(row) -> str:
    text = _ideas_plain(row.get("call_status"), 130)
    if text == "No data":
        text = _ideas_plain(row.get("option_route"), 130)
    if "no call" in text.lower() or "risk" in text.lower():
        return "No call now. Study it only after risk, event proof, and trigger clear."
    if "call" in text.lower():
        return text
    return "No call setup is ready."


def _ideas_put_line(row) -> str:
    text = _ideas_plain(row.get("put_status"), 130)
    route = _ideas_plain(row.get("option_route"), 130)
    if "put" in text.lower() or "hedge" in text.lower():
        return text
    if "put" in route.lower() or "hedge" in route.lower():
        return "Put or hedge research only. Use it as protection, not a new bet."
    return "No put setup is ready."


def _ideas_main_blocker(row) -> str:
    blocker = _ideas_plain(row.get("main_blocker"), 150)
    if blocker != "No data":
        return blocker
    parts = []
    for label, col in [("Risk", "risk_action"), ("Event", "event_gate"), ("News", "news_reliability_status")]:
        val = _ideas_plain(row.get(col), 70)
        if val != "No data":
            parts.append(f"{label}: {val}")
    return "; ".join(parts) if parts else "No blocker found."


def _ideas_unlock_line(row) -> str:
    unlock = _ideas_plain(row.get("unlock_checklist"), 190)
    if unlock != "No data":
        return unlock
    trigger = _ideas_plain(row.get("trigger_to_watch"), 120)
    if trigger != "No data":
        return f"Wait for this trigger first: {trigger}"
    return "Wait for cleaner risk, cleaner news proof, and a confirmed price trigger."


def _ideas_accent(row) -> str:
    text = " ".join(
        _ideas_plain(row.get(col), 120).lower()
        for col in ["gate_status", "risk_action", "current_desk_action", "main_blocker", "option_route"]
    )
    if any(x in text for x in ["risk", "smaller size", "blocked", "tiny paper"]):
        return "#991b1b"
    if any(x in text for x in ["put", "hedge", "downside"]):
        return "#334155"
    if any(x in text for x in ["wait", "review", "prove"]):
        return "#0f766e"
    return "#111827"


def _ideas_status_accent(*values) -> str:
    text = " ".join(str(v or "") for v in values).upper()
    if any(x in text for x in ["REDUCE", "NO NEW", "BLOCK", "CRITICAL", "NO EXPOSURE"]):
        return "#991b1b"
    if any(x in text for x in ["PUT", "HEDGE", "DOWNSIDE"]):
        return "#334155"
    if any(x in text for x in ["WATCH", "REVIEW", "WAIT", "SIZE_DOWN", "DATA", "PROOF"]):
        return "#0f766e"
    if any(x in text for x in ["CALL", "STOCK", "ETF", "CLEAR"]):
        return "#111827"
    return "#334155"


def _ideas_ticker_list(df: pd.DataFrame, col: str = "ticker", limit: int = 7) -> str:
    if df is None or df.empty or col not in df.columns:
        return "None ready"
    names = []
    for item in df[col].dropna().astype(str).tolist():
        item = item.strip()
        if item and item.lower() not in {"nan", "none"} and item not in names:
            names.append(item)
        if len(names) >= limit:
            break
    return ", ".join(names) if names else "None ready"


def _ideas_route_permission(row) -> str:
    gate = _ideas_plain(row.get("gate_status"), 90).lower()
    risk = _ideas_plain(row.get("risk_action"), 90).lower()
    option = _ideas_plain(row.get("option_route"), 120).lower()
    call = _ideas_call_line(row).lower()
    put = _ideas_put_line(row).lower()
    if "no new exposure" in gate or "reduce" in risk:
        return "No new exposure. Risk repair comes first."
    if "tiny paper" in gate or "smaller size" in risk:
        return "Tiny stock or ETF research only. Options wait."
    if "call" in option or "call" in call:
        return "Call research only after risk, proof, trigger, and trading cost clear."
    if "put" in option or "hedge" in option or "put" in put or "hedge" in put:
        return "Put or hedge research only. Use as protection, not a new bullish bet."
    return "Watch only until price, volume, news proof, and risk line up."


def _render_alpha_movers():
    """
    Alpha movers panel — top 5 risers and top 5 fallers since last daily run.
    Source: alpha_score_history.csv (8 days of history).
    Shown above the leaderboard table.
    """
    hist_path = ROOT / "alpha_score_history.csv"
    if not hist_path.exists():
        return
    try:
        hist = pd.read_csv(hist_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.dropna(subset=["date"])
        dates = sorted(hist["date"].unique())
    except Exception:
        return
    if len(dates) < 2:
        return

    latest = hist[hist["date"] == dates[-1]][["ticker","alpha_score"]].rename(columns={"alpha_score":"now"})
    prev   = hist[hist["date"] == dates[-2]][["ticker","alpha_score"]].rename(columns={"alpha_score":"prev"})
    delta  = latest.merge(prev, on="ticker", how="inner")
    delta["delta"] = (delta["now"] - delta["prev"]).round(1)

    # Load sector for context
    alpha_df = safe_csv(ROOT / "alpha_scores.csv")
    sect_map = {}
    if not alpha_df.empty and "ticker" in alpha_df.columns and "sector" in alpha_df.columns:
        sect_map = {r["ticker"]: str(r["sector"]).split()[0] for _, r in alpha_df.iterrows()}

    top5 = delta.nlargest(5, "delta")[["ticker","prev","now","delta"]].reset_index(drop=True)
    bot5 = delta.nsmallest(5, "delta")[["ticker","prev","now","delta"]].reset_index(drop=True)

    prev_date = dates[-2].strftime("%b %d") if hasattr(dates[-2], "strftime") else str(dates[-2])[:10]
    today_date = dates[-1].strftime("%b %d") if hasattr(dates[-1], "strftime") else str(dates[-1])[:10]

    def _mover_row(row, up: bool) -> str:
        color  = "#16a34a" if up else "#dc2626"
        bg     = "#f0fdf4" if up else "#fef2f2"
        sign   = "↑" if up else "↓"
        ticker = str(row.get("ticker",""))
        sect   = sect_map.get(ticker, "")
        delta_val = _to_float(row.get("delta"))
        now_val   = _to_float(row.get("now"))
        delta_str = f"{sign}{abs(delta_val):.1f}" if delta_val is not None else "—"
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:7px 10px;background:{bg};border-radius:7px;margin-bottom:5px;">'
            f'<div>'
            f'<span style="font-size:15px;font-weight:900;color:#0f172a;">{_esc(ticker)}</span>'
            f'<span style="font-size:10px;color:#94a3b8;margin-left:7px;">{_esc(sect)}</span>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<span style="font-size:13px;font-weight:800;color:{color};">{delta_str}</span>'
            f'<span style="font-size:10px;color:#64748b;margin-left:6px;">{f"{now_val:.1f}" if now_val else "—"}</span>'
            f'</div>'
            f'</div>'
        )

    risers_html = "".join(_mover_row(r, True)  for _, r in top5.iterrows())
    fallers_html= "".join(_mover_row(r, False) for _, r in bot5.iterrows())

    c1, c2 = st.columns(2)
    with c1:
        _render_html(
            f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-top:3px solid #16a34a;
                border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.05);">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div style="font-size:12px;font-weight:800;color:#166534;text-transform:uppercase;letter-spacing:.04em;">↑ Rising alpha</div>
                <div style="font-size:10px;color:#94a3b8;">{prev_date} → {today_date}</div>
              </div>
              {risers_html}
            </div>
            """
        )
    with c2:
        _render_html(
            f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-top:3px solid #dc2626;
                border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.05);">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div style="font-size:12px;font-weight:800;color:#991b1b;text-transform:uppercase;letter-spacing:.04em;">↓ Falling alpha</div>
                <div style="font-size:10px;color:#94a3b8;">{prev_date} → {today_date}</div>
              </div>
              {fallers_html}
            </div>
            """
        )


def _render_sector_alpha_strip():
    """
    Compact sector alpha heatmap — average alpha per sector, sorted descending.
    Helps spot sector rotation without scrolling the full leaderboard.
    """
    df = safe_csv(ROOT / "alpha_scores.csv")
    if df.empty or "sector" not in df.columns or "alpha_score" not in df.columns:
        return

    sect = (
        df.groupby("sector")
        .agg(avg=("alpha_score","mean"), n=("alpha_score","count"),
             top=("ticker", lambda x: df.loc[x.index].nlargest(1,"alpha_score")["ticker"].values[0]),
             top_score=("alpha_score","max"))
        .round(1)
        .sort_values("avg", ascending=False)
        .reset_index()
    )

    SECT_SHORT = {
        "Technology": "Tech", "Health Care": "Health", "Energy": "Energy",
        "Industrials": "Indus", "Consumer Discretionary": "C.Disc",
        "Financials": "Finance", "Materials": "Mater",
        "Communication Services": "Comm", "Consumer Staples": "C.Stap",
        "Real Estate": "RE", "Utilities": "Utils",
    }

    pills_html = ""
    for _, row in sect.iterrows():
        avg = _to_float(row.get("avg"))
        if avg is None:
            continue
        short = SECT_SHORT.get(str(row["sector"]), str(row["sector"])[:6])
        top_t = str(row.get("top",""))
        top_s = _to_float(row.get("top_score"))
        # Green > 55, yellow 48-55, light gray < 48
        if avg >= 55:   bg, col = "#dcfce7", "#166534"
        elif avg >= 48: bg, col = "#fef9c3", "#854d0e"
        else:           bg, col = "#f1f5f9", "#64748b"
        pills_html += (
            f'<div style="background:{bg};border-radius:8px;padding:8px 12px;'
            f'text-align:center;min-width:72px;flex-shrink:0;">'
            f'<div style="font-size:11px;font-weight:800;color:{col};">{_esc(short)}</div>'
            f'<div style="font-size:16px;font-weight:900;color:#0f172a;margin-top:2px;">{avg:.0f}</div>'
            f'<div style="font-size:9px;color:#94a3b8;margin-top:1px;">{_esc(top_t)} {f"{top_s:.0f}" if top_s else ""}</div>'
            f'</div>'
        )

    _render_html(
        f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
            padding:12px 16px;margin:0 0 14px 0;box-shadow:0 1px 3px rgba(0,0,0,.05);">
          <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">
            Sector alpha — average score · green ≥ 55 · yellow ≥ 48
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">{pills_html}</div>
        </div>
        """
    )


def _alpha_conflict_badge(row) -> str:
    """
    Return a short HTML badge if this ticker has a notable signal conflict.
    Conflicts worth flagging:
      • News < 25 but α ≥ 65           → "Bad news"
      • Options < 25 but Regime > 75   → "Options disagree"
      • Quality < 35 but Mom★ > 80     → "Momentum, no qual"
      • Insider < 35 but α ≥ 65        → "Insider selling"
      • Crowding = WATCH               → "Crowded"
    Returns empty string if no conflict.
    """
    def _v(col): return _to_float(row.get(col))

    flags = []
    news    = _v("sig_sentiment")
    opts    = _v("sig_options")
    regime  = _v("sig_regime_ml")
    quality = _v("sig_quality")
    insider = _v("sig_insider")
    alpha   = _v("alpha_score")
    crowd   = str(row.get("crowding_level","")).upper()

    if news is not None    and news < 25    and alpha and alpha >= 65:
        flags.append(("Bad news", "#dc2626"))
    if opts is not None    and opts < 25    and regime and regime > 75:
        flags.append(("Opts ↓", "#d97706"))
    if quality is not None and quality < 35 and alpha and alpha >= 65:
        flags.append(("No qual", "#7c3aed"))
    if insider is not None and insider < 35 and alpha and alpha >= 65:
        flags.append(("Ins ↓", "#dc2626"))
    if crowd == "WATCH":
        flags.append(("Crowded", "#94a3b8"))

    if not flags:
        return ""
    # Return first two flags max
    return " ".join(
        f'<span style="background:{c}22;color:{c};font-size:8.5px;font-weight:700;'
        f'padding:1px 5px;border-radius:3px;white-space:nowrap;">{lbl}</span>'
        for lbl, c in flags[:2]
    )


def _render_alpha_screener():
    """
    Top-20 alpha leaderboard — color-coded signal cells.
    Source: alpha_scores.csv (495 tickers, updated daily by step87).
    """
    df = safe_csv(ROOT / "alpha_scores.csv")
    if df.empty or "alpha_score" not in df.columns:
        return

    mom_df = safe_csv(ROOT / "momentum_scores.csv")
    mom_map = {}
    if not mom_df.empty and "ticker" in mom_df.columns:
        mom_map = {str(r["ticker"]): _to_float(r.get("momentum_score")) for _, r in mom_df.iterrows()}

    work = df.nlargest(20, "alpha_score").reset_index(drop=True)

    SIG_LABELS = [
        ("sig_regime_ml",  "Regime"),
        ("sig_momentum",   "sMom"),    # signal-layer momentum (inside ML model)
        ("sig_quality",    "Quality"),
        ("sig_options",    "Options"),
        ("sig_sentiment",  "News"),
        ("sig_insider",    "Insider"),
    ]

    def _cell(val, w="48px"):
        v = _to_float(val)
        if v is None:
            return f'<td style="padding:5px 6px;text-align:center;font-size:11px;color:#cbd5e1;width:{w};">—</td>'
        bg = "#dcfce7" if v >= 65 else "#fef9c3" if v >= 40 else "#fee2e2"
        col= "#166534" if v >= 65 else "#854d0e" if v >= 40 else "#991b1b"
        return f'<td style="padding:5px 6px;text-align:center;font-size:11px;font-weight:700;color:{col};background:{bg};width:{w};border-radius:4px;">{v:.0f}</td>'

    # Header
    sig_headers = "".join(
        f'<th style="padding:5px 6px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.05em;min-width:44px;">{lbl}</th>'
        for _, lbl in SIG_LABELS
    )
    # Alpha delta map from history
    delta_map: dict = {}
    hist_path = ROOT / "alpha_score_history.csv"
    try:
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist = hist.dropna(subset=["date"])
            hdates = sorted(hist["date"].unique())
            if len(hdates) >= 2:
                lat = hist[hist["date"] == hdates[-1]][["ticker","alpha_score"]].rename(columns={"alpha_score":"now"})
                prv = hist[hist["date"] == hdates[-2]][["ticker","alpha_score"]].rename(columns={"alpha_score":"prev"})
                dm  = lat.merge(prv, on="ticker", how="inner")
                dm["delta"] = (dm["now"] - dm["prev"]).round(1)
                delta_map = {r["ticker"]: r["delta"] for _, r in dm.iterrows()}
    except Exception:
        pass

    rows_html = ""
    for i, row in work.iterrows():
        tkr    = str(row.get("ticker",""))
        score  = _to_float(row.get("alpha_score"))
        sector = str(row.get("sector","")).split()[0] if row.get("sector") else "—"
        regime = str(row.get("regime",""))
        signal = str(row.get("signal",""))
        signal_color = "#16a34a" if signal == "BUY" else "#dc2626" if signal in ("SELL","SHORT") else "#64748b"
        bar_w  = max(4, int(score or 0))
        sig_cells = "".join(_cell(row.get(col)) for col, _ in SIG_LABELS)
        mom    = mom_map.get(tkr)
        mom_cell = _cell(mom, "44px") if mom is not None else '<td style="padding:5px 6px;text-align:center;font-size:11px;color:#cbd5e1;width:44px;">—</td>'
        # Delta cell
        d = delta_map.get(tkr)
        if d is not None:
            d_col = "#16a34a" if d > 0.5 else "#dc2626" if d < -0.5 else "#94a3b8"
            d_sign = f"↑{d:.1f}" if d > 0.5 else f"↓{abs(d):.1f}" if d < -0.5 else f"~{d:.1f}"
            delta_cell = f'<td style="padding:5px 6px;text-align:center;font-size:10px;font-weight:700;color:{d_col};width:38px;">{d_sign}</td>'
        else:
            delta_cell = '<td style="padding:5px 6px;text-align:center;font-size:10px;color:#cbd5e1;width:38px;">—</td>'
        row_bg = "#f8fafc" if i % 2 == 0 else "#fff"
        rows_html += f"""
        <tr style="background:{row_bg};">
          <td style="padding:6px 8px;font-size:12px;font-weight:700;color:#64748b;width:28px;text-align:right;">{i+1}</td>
          <td style="padding:6px 10px;font-size:14px;font-weight:900;color:#0f172a;min-width:58px;letter-spacing:-0.2px;">{_esc(tkr)}</td>
          {delta_cell}
          <td style="padding:6px 6px;font-size:10px;color:#64748b;min-width:90px;">{_esc(sector)}</td>
          <td style="padding:6px 10px;min-width:110px;">
            <div style="display:flex;align-items:center;gap:6px;">
              <div style="flex:1;background:#e2e8f0;border-radius:3px;height:8px;overflow:hidden;">
                <div style="width:{bar_w}%;background:#0f172a;height:8px;border-radius:3px;"></div>
              </div>
              <span style="font-size:12px;font-weight:800;color:#0f172a;min-width:30px;">{score:.1f}</span>
            </div>
          </td>
          {sig_cells}{mom_cell}
          <td style="padding:5px 8px;font-size:10px;font-weight:700;color:{signal_color};">{_esc(signal)}</td>
          <td style="padding:5px 6px;font-size:9px;">{_alpha_conflict_badge(row)}</td>
        </tr>"""

    age_note = ""
    alpha_path = ROOT / "alpha_scores.csv"
    if alpha_path.exists():
        age_h = (time.time() - alpha_path.stat().st_mtime) / 3600
        age_note = f" · data {age_h:.1f}h old"

    with st.expander(f"Alpha leaderboard — top 20 of {len(df)} S&P 500 tickers{age_note}", expanded=True):
        st.caption(
            "Combined score (0–100) from regime model, momentum, quality, options, sentiment, and insider signals. "
            "Green cell ≥ 65 · Yellow 40–65 · Red < 40 · Mom = standalone momentum rank. "
            "Risk mode is SIZE_DOWN — these are research ideas, not trade instructions."
        )
        _render_html(
            f"""
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:separate;border-spacing:0 2px;font-family:inherit;">
              <thead>
                <tr style="background:#f1f5f9;">
                  <th style="padding:6px 8px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;width:28px;">#</th>
                  <th style="padding:6px 10px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;min-width:58px;">Ticker</th>
                  <th style="padding:5px 6px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;width:38px;" title="Change in alpha score since yesterday">Δ 1d</th>
                  <th style="padding:6px 6px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;min-width:90px;">Sector</th>
                  <th style="padding:6px 10px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;min-width:110px;">α Score</th>
                  {sig_headers}
                  <th style="padding:5px 6px;font-size:9px;color:#0f766e;font-weight:700;text-transform:uppercase;min-width:44px;" title="Full 4-component momentum (Step 127): J-T 12-1m + 52wH + vol-scaled + residual">Mom★</th>
                  <th style="padding:5px 8px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;">Signal</th>
                  <th style="padding:5px 6px;font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;" title="Signal conflict flags">⚑</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            </div>
            """
        )
        st.caption(
            "Regime = ML model reading market conditions · Mom = 12-month price momentum · "
            "Quality = company fundamentals (Novy-Marx) · Options = call/put flow signal · "
            "News = headline sentiment · Insider = SEC Form 4 filings · "
            "Stand-alone Mom column = full 4-component momentum rank (Step 127)"
        )


def _render_weekly_pulse():
    """
    Small weekly performance summary — 7-day return, regime, picks count.
    Source: weekly_summary.json + alpha_scores.csv
    """
    w = safe_json(ROOT / "weekly_summary.json")
    if not w:
        return

    ret      = _to_float(w.get("portfolio_return"))
    spy_ret  = _to_float(w.get("spy_return"))
    n_picks  = int(_to_float(w.get("n_picks"), 0) or 0)
    n_alpha  = int(_to_float(w.get("n_alpha_tracked"), 0) or 0)
    regime   = str(w.get("regime") or "—")
    rep_date = str(w.get("report_date") or "—")

    ret_color = "#16a34a" if ret and ret > 0 else "#dc2626" if ret and ret < 0 else "#64748b"
    ret_str   = f"+{ret:.1f}%" if ret and ret > 0 else f"{ret:.1f}%" if ret is not None else "—"
    spy_str   = f"+{spy_ret:.1f}%" if spy_ret and spy_ret > 0 else f"{spy_ret:.1f}%" if spy_ret is not None else "No data"

    _render_html(
        f"""
        <div style="
            background:#fff;
            border:1px solid #e2e8f0;
            border-left:4px solid #0f172a;
            border-radius:10px;
            padding:14px 18px;
            margin:0 0 14px 0;
            box-shadow:0 1px 4px rgba(0,0,0,.05);
            display:flex;
            flex-wrap:wrap;
            gap:20px;
            align-items:center;
        ">
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">7-Day Portfolio</div>
            <div style="font-size:20px;font-weight:800;color:{ret_color};margin-top:3px;">{ret_str}</div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">S&P 500</div>
            <div style="font-size:14px;font-weight:700;color:#64748b;margin-top:3px;">{spy_str}</div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Regime</div>
            <div style="font-size:14px;font-weight:700;color:#475569;margin-top:3px;">{_esc(regime)}</div>
          </div>
          <div style="width:1px;height:32px;background:#e2e8f0;"></div>
          <div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Picks tracked</div>
            <div style="font-size:14px;font-weight:700;color:#475569;margin-top:3px;">{n_picks} picks · {n_alpha} alpha universe</div>
          </div>
          <div style="margin-left:auto;font-size:10px;color:#94a3b8;">Week of {_esc(rep_date)}</div>
        </div>
        """
    )


def _render_ideas_command_center(horizon: pd.DataFrame, options_route: pd.DataFrame, playbook: pd.DataFrame, timeframe: pd.DataFrame):
    total = len(horizon) if horizon is not None and not horizon.empty else 0
    no_new = _ideas_count_contains(horizon, "gate_status", "No new exposure")
    tiny = _ideas_count_contains(horizon, "gate_status", "Tiny")
    put_routes = _ideas_count_contains(options_route, "final_option_side", "PUT") + _ideas_count_contains(playbook, "option_side", "PUT")
    call_watch = _ideas_count_contains(playbook, "option_side", "CALL")
    call_blocked = _ideas_count_contains(playbook, "option_permission", "CALL_BLOCKED")
    short_count = _ideas_count_contains(timeframe, "timeframe", "Short")
    medium_count = _ideas_count_contains(timeframe, "timeframe", "Medium")
    long_count = _ideas_count_contains(timeframe, "timeframe", "Long")

    if no_new >= max(1, total // 2):
        answer = "Safety is still the first decision. Study ideas, but do not add size, calls, or puts yet."
        accent = "#991b1b"
    elif put_routes:
        answer = "The idea board leans defensive. Put or hedge research comes before bullish calls."
        accent = "#334155"
    elif tiny:
        answer = "Only tiny stock or ETF research is open. Options still need stronger proof."
        accent = "#0f766e"
    else:
        answer = "Some ideas can be studied, but every ticker still needs time-frame, safety, news, and trading-cost checks."
        accent = "#111827"

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:7px solid {accent}; border-radius:10px; padding:21px 23px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">Simple idea answer</div>
          <div style="font-size:30px; color:#111827; font-weight:950; line-height:1.16; margin-top:7px;">{_esc(answer)}</div>
          <div style="font-size:15px; color:#374151; line-height:1.48; margin-top:12px;">Read this page left to right: first how long, then stock/call/put/wait, then the price level to watch. A ticker is not an idea until all three agree.</div>
          <div style="font-size:13px; color:#6b7280; line-height:1.4; margin-top:7px;">Research-only. No broker connection. No live orders.</div>
        </div>
        """
    )

    cards = [
        ("Short-term", str(short_count), "1-5 trading days; price level must be close.", "#334155"),
        ("Medium-term", str(medium_count), "2-8 weeks; trend and event proof matter.", "#334155"),
        ("Long-term", str(long_count), "3-12 months; thesis and portfolio safety matter.", "#334155"),
        ("Do not add", f"{no_new}/{total}", "Safety fix before any new idea.", "#991b1b" if no_new else "#111827"),
        ("Tiny stock / ETF", str(tiny), "Stock or ETF only; keep size tiny.", "#0f766e" if tiny else "#111827"),
        ("Calls to study", str(call_watch), f"{call_blocked} blocked by safety.", "#334155"),
        ("Put / protection", str(put_routes), "Protection research, not automatic buys.", "#334155" if put_routes else "#111827"),
    ]
    cols = st.columns(4)
    for idx, (title, value, note, color) in enumerate(cards):
        with cols[idx % 4]:
            _simple_card(title, value, note, color)


def _render_ideas_horizon_lanes(timeframe: pd.DataFrame):
    if timeframe is None or timeframe.empty:
        return

    st.markdown("#### How long is the idea?")
    st.caption("Pick the time frame first. The same ticker can be a no-go short-term and still be a long-term research question.")
    work = timeframe.copy()
    if "score" in work.columns:
        work["_score"] = pd.to_numeric(work["score"], errors="coerce").fillna(-999)

    lane_defs = [
        ("Short-term", "1-5 trading days", "Needs a near trigger. Do not force a thesis into a day trade."),
        ("Medium-term", "2-8 weeks", "Needs trend, event proof, and a clean safety check."),
        ("Long-term", "3-12 months", "Needs business thesis, valuation, and portfolio fit."),
    ]
    cols = st.columns(3)
    for col, (lane, window, rule) in zip(cols, lane_defs):
        _tf_mask = work["timeframe"].astype(str).str.contains(lane.split("-")[0], case=False, na=False) if "timeframe" in work.columns else pd.Series(False, index=work.index)
        lane_rows = work[_tf_mask].copy()
        if "_score" in lane_rows.columns:
            lane_rows = lane_rows.sort_values("_score", ascending=False)
        with col:
            items = []
            for _, row in lane_rows.head(4).iterrows():
                ticker = _plain_status(row.get("ticker"), "Ticker")
                action = _ideas_plain(row.get("action"), 70)
                vehicle = _ideas_plain(row.get("vehicle"), 65)
                option = _ideas_plain(row.get("option_side"), 45)
                trigger = _ideas_plain(row.get("trigger_to_watch"), 105)
                accent = _ideas_status_accent(row.get("action"), row.get("vehicle"), row.get("option_side"))
                items.append(
                    f"""
                    <div style="border-left:4px solid {accent}; padding:8px 0 8px 10px; margin:8px 0; border-top:1px solid #e5e7eb;">
                      <div style="display:flex; justify-content:space-between; gap:8px;">
                        <div style="font-size:17px; color:#111827; font-weight:950;">{_esc(ticker)}</div>
                        <div style="font-size:11px; color:{accent}; font-weight:900;">{_esc(option)}</div>
                      </div>
                      <div style="font-size:12px; color:#111827; line-height:1.35; margin-top:4px;"><b>{_esc(action)}</b> · {_esc(vehicle)}</div>
                      <div style="font-size:12px; color:#6b7280; line-height:1.35; margin-top:4px;">{_esc(trigger)}</div>
                    </div>
                    """
                )
            if not items:
                items.append("<div style='font-size:13px; color:#6b7280; margin-top:10px;'>No rows for this lane yet.</div>")
            _render_html(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-top:5px solid #111827; border-radius:8px; padding:15px 16px; min-height:420px; margin-bottom:14px;">
                  <div style="font-size:21px; color:#111827; font-weight:950; line-height:1.15;">{_esc(lane)}</div>
                  <div style="font-size:12px; color:#6b7280; font-weight:850; margin-top:4px;">{_esc(window)}</div>
                  <div style="font-size:12px; color:#374151; line-height:1.4; margin-top:8px;">{_esc(rule)}</div>
                  {''.join(items)}
                </div>
                """
            )


def _render_ideas_vehicle_board(horizon: pd.DataFrame, options_route: pd.DataFrame, playbook: pd.DataFrame):
    st.markdown("#### Stock, call, put, or wait")
    st.caption("This separates stock/ETF, call, put/protection, and wait. Options are research-only until every safety check clears.")

    stock_rows = pd.DataFrame()
    no_rows = pd.DataFrame()
    if horizon is not None and not horizon.empty:
        _gs = horizon["gate_status"].astype(str) if "gate_status" in horizon.columns else pd.Series("", index=horizon.index)
        stock_mask = _gs.str.contains("tiny|stock|ETF", case=False, na=False)
        stock_rows = horizon[stock_mask].copy()
        no_rows = horizon[_gs.str.contains("No new exposure|reduce", case=False, na=False)].copy()
    call_rows = pd.DataFrame()
    put_rows = pd.DataFrame()
    if playbook is not None and not playbook.empty:
        _os = playbook["option_side"].astype(str) if "option_side" in playbook.columns else pd.Series("", index=playbook.index)
        call_rows = playbook[_os.str.contains("CALL", case=False, na=False)].copy()
        put_rows  = playbook[_os.str.contains("PUT",  case=False, na=False)].copy()
    if options_route is not None and not options_route.empty:
        _fos = options_route["final_option_side"].astype(str) if "final_option_side" in options_route.columns else pd.Series("", index=options_route.index)
        route_put = options_route[_fos.str.contains("PUT", case=False, na=False)].copy()
        put_rows = pd.concat([put_rows, route_put], ignore_index=True) if not route_put.empty else put_rows

    cards = [
        (
            "Stock / ETF",
            _ideas_ticker_list(stock_rows, limit=8),
            "Only tiny paper research if risk allows. This is not approval to add real exposure.",
            "#0f766e" if not stock_rows.empty else "#111827",
        ),
        (
            "Call",
            _ideas_ticker_list(call_rows, limit=8),
            "Mostly blocked today. A call needs safety clear, price level, news proof, and trading-cost proof.",
            "#334155",
        ),
        (
            "Put / Hedge",
            _ideas_ticker_list(put_rows, limit=8),
            "Protection research only. Use for downside confirmation or risk reduction, not excitement.",
            "#334155" if not put_rows.empty else "#111827",
        ),
        (
            "Wait / No Add",
            _ideas_ticker_list(no_rows, limit=8),
            "Do not force an idea. Risk repair and proof collection come first.",
            "#991b1b" if not no_rows.empty else "#111827",
        ),
    ]
    cols = st.columns(4)
    for col, (title, tickers, note, color) in zip(cols, cards):
        with col:
            _render_html(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {color}; border-radius:8px; padding:15px 16px; min-height:190px; margin-bottom:14px;">
                  <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">{_esc(title)}</div>
                  <div style="font-size:20px; color:#111827; font-weight:950; line-height:1.2; margin-top:8px;">{_esc(tickers)}</div>
                  <div style="font-size:12px; color:#374151; line-height:1.4; margin-top:9px;">{_esc(note)}</div>
                </div>
                """
            )


def _render_ideas_route_guide():
    st.markdown("#### How to choose an idea")
    st.caption("Read this before looking at any ticker. Choose the time frame first, then stock/call/put/wait.")
    cards = [
        ("1. Choose time", "Short, medium, or long", "Do not mix a 2-day trade with a 12-month thesis."),
        ("2. Check safety", "Safety can stop everything", "If safety says smaller size, calls and new size wait."),
        ("3. Pick type", "Stock first, options last", "Calls or puts need safety, event proof, price level, and trading-cost checks."),
        ("4. Wait for price", "Price and volume must confirm", "A headline alone is not enough proof."),
    ]
    cols = st.columns(4)
    for col, (title, value, note) in zip(cols, cards):
        with col:
            _simple_card(title, value, note, "#111827")


def _render_ideas_cards(horizon: pd.DataFrame, max_cards: int = 12):
    if horizon is None or horizon.empty:
        st.info("No idea route file is available yet. Run the daily system first.")
        return

    # Load alpha scores for signal mini-bars
    alpha_df = safe_csv(ROOT / "alpha_scores.csv")
    alpha_map: dict = {}
    if not alpha_df.empty and "ticker" in alpha_df.columns:
        alpha_map = {str(r["ticker"]): r.to_dict() for _, r in alpha_df.iterrows()}

    work = horizon.copy()
    if "decision_depth_score" in work.columns:
        work["_score"] = pd.to_numeric(work["decision_depth_score"], errors="coerce").fillna(-999)
        work = work.sort_values(["_score", "ticker"], ascending=[False, True])
    work = work.head(max_cards)

    st.markdown("#### Ideas to read first")
    st.caption(
        "Each card shows the holding time, stock/option choice, what to watch, and a signal bar "
        "( α = combined score · ML · Mom = momentum · Qual = quality · Sent = sentiment · Ins = insider ). "
        "Green dot > 65 · Gray 40–65 · Red < 40."
    )

    for start in range(0, len(work), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, work.iloc[start:start + 4].iterrows()):
            ticker = _plain_status(row.get("ticker"), "Ticker")
            sector = _ideas_plain(row.get("sector"), 55)
            best_horizon = _ideas_plain(row.get("best_horizon"), 55)
            decision = _ideas_vehicle_now(row)
            horizon_line = _ideas_horizon_summary(row)
            stock_line = _ideas_plain(row.get("short_vehicle") or row.get("medium_vehicle") or row.get("long_vehicle"), 90)
            if stock_line == "No data" or "stock" in stock_line.lower():
                stock_line = "Stock or ETF is the only choice to study now."
            call_line = _ideas_call_line(row)
            put_line = _ideas_put_line(row)
            trigger = _ideas_plain(row.get("trigger_to_watch"), 115)
            blocker = _ideas_main_blocker(row)
            unlock = _ideas_unlock_line(row)
            news = _ideas_plain(row.get("top_news_headline"), 105)
            accent = _ideas_accent(row)
            signal_bar = _alpha_signal_mini_bar(ticker, alpha_map)
            gate_status = _plain_status(row.get("gate_status"), "")
            gate_bg = "#fef2f2" if "no" in gate_status.lower() or "block" in gate_status.lower() else "#f0fdf4" if "ready" in gate_status.lower() else "#f8fafc"
            gate_color = "#dc2626" if "no" in gate_status.lower() or "block" in gate_status.lower() else "#16a34a" if "ready" in gate_status.lower() else "#475569"
            with col:
                _render_html(
                    f"""
                    <div style="
                        background:#ffffff;
                        border:1px solid #e2e8f0;
                        border-top:3px solid {accent};
                        border-radius:12px;
                        padding:15px 16px;
                        min-height:500px;
                        margin-bottom:13px;
                        box-shadow:0 2px 8px rgba(0,0,0,.06);
                    ">
                      <div style="display:flex; justify-content:space-between; gap:8px; align-items:flex-start; margin-bottom:4px;">
                        <div style="font-size:28px; color:#0f172a; font-weight:900; line-height:1; letter-spacing:-0.5px;">{_esc(ticker)}</div>
                        <div style="text-align:right;">
                          <div style="font-size:11px;color:#64748b;font-weight:600;">{_esc(sector)}</div>
                          <div style="font-size:10px;color:#94a3b8;margin-top:2px;">{_esc(best_horizon)}</div>
                        </div>
                      </div>
                      <div style="margin-bottom:8px;">
                        <span style="font-size:10px;color:{gate_color};font-weight:700;text-transform:uppercase;letter-spacing:.06em;background:{gate_bg};padding:2px 8px;border-radius:99px;">{_esc(gate_status) or 'Review'}</span>
                      </div>
                      {signal_bar}
                      <div style="font-size:14px; color:#0f172a; font-weight:700; line-height:1.35; margin-top:8px;">{_esc(decision)}</div>
                      <div style="height:1px;background:#f1f5f9;margin:10px 0;"></div>
                      <div style="font-size:12px; color:#475569; line-height:1.45;"><b>Time frame:</b> {_esc(horizon_line)}</div>
                      <div style="font-size:12px; color:#475569; line-height:1.45; margin-top:6px;"><b>Stock / ETF:</b> {_esc(stock_line)}</div>
                      <div style="font-size:12px; color:#475569; line-height:1.45; margin-top:6px;"><b>Call:</b> {_esc(call_line)}</div>
                      <div style="font-size:12px; color:#475569; line-height:1.45; margin-top:6px;"><b>Put / hedge:</b> {_esc(put_line)}</div>
                      <div style="height:1px;background:#f1f5f9;margin:10px 0;"></div>
                      <div style="font-size:12px; color:#1e293b; line-height:1.4;"><b>Watch:</b> {_esc(trigger)}</div>
                      <div style="font-size:12px; color:#64748b; line-height:1.4; margin-top:6px;"><b>Latest news:</b> {_esc(news)}</div>
                      <div style="font-size:11.5px; color:#94a3b8; line-height:1.38; margin-top:7px;">Blocked: {_esc(blocker)}</div>
                      <div style="font-size:11.5px; color:#94a3b8; line-height:1.38; margin-top:5px;">Unlocks: {_esc(unlock)}</div>
                    </div>
                    """
                )


def _render_ideas_option_summary(horizon: pd.DataFrame, options_route: pd.DataFrame, playbook: pd.DataFrame):
    total = len(horizon) if horizon is not None else 0
    tiny = _ideas_count_contains(horizon, "gate_status", "tiny")
    puts = _ideas_count_contains(horizon, "put_status", "put") + _ideas_count_contains(horizon, "option_route", "hedge")
    calls_blocked = _ideas_count_contains(horizon, "call_status", "no call")
    ready_options = 0
    if options_route is not None and not options_route.empty and "final_option_side" in options_route.columns:
        final_side = options_route["final_option_side"].astype(str).str.upper()
        _fvd = options_route["final_vehicle_decision"].astype(str) if "final_vehicle_decision" in options_route.columns else pd.Series("", index=options_route.index)
        final_decision = _fvd.str.upper()
        ready_options = int(((final_side.isin(["CALL", "PUT"])) & (~final_decision.str.contains("WAIT|NO OPTION|NONE", na=False))).sum())
    call_watch = _ideas_count_contains(playbook, "option_permission", "CALL") if playbook is not None else 0

    cols = st.columns(4)
    with cols[0]:
        _simple_card("Stock / ETF choice", f"{tiny}/{total}", "Mostly tiny paper only while safety is tight.", "#991b1b" if tiny else "#111827")
    with cols[1]:
        _simple_card("Calls ready now", str(ready_options if ready_options else 0), "A call needs risk, trigger, event proof, and trading-cost proof.", "#111827")
    with cols[2]:
        _simple_card("Call ideas to study", str(call_watch), f"{calls_blocked} are blocked by risk today.", "#334155")
    with cols[3]:
        _simple_card("Put / hedge ideas", str(puts), "These are protection research, not automatic buys.", "#0f766e" if puts else "#111827")


def _render_ideas_horizon_counts(horizon: pd.DataFrame):
    if horizon is None or horizon.empty or "best_horizon" not in horizon.columns:
        return
    counts = horizon["best_horizon"].astype(str).replace({"nan": "Unknown"}).value_counts().to_dict()
    html = ['<div style="display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 18px 0;">']
    for label, count in counts.items():
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-radius:999px; padding:8px 12px; font-size:13px; color:#111827; font-weight:850;">
              {_esc(_ideas_plain(label, 60))}: {_esc(count)}
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))


def tab_ideas_workflow():
    horizon = safe_csv(ROOT / "horizon_vehicle_summary.csv")
    options_route = safe_csv(ROOT / "options_execution_route_matrix.csv")
    playbook = safe_csv(ROOT / "options_playbook.csv")
    daily_queue = safe_csv(ROOT / "daily_workflow_queue.csv")
    promotion_gate = safe_csv(ROOT / "institutional_promotion_gate.csv")
    timeframe = safe_csv(ROOT / "timeframe_decision_matrix.csv")
    strategy = safe_csv(ROOT / "strategy_route_playbook.csv")

    st.markdown('<p class="section-title">Ideas: stock, call, put, or wait?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This page separates short-term, medium-term, and long-term first. Then it tells you whether to study stock, call, put, hedge, or wait. Research only. No broker. No live orders.</p>',
        unsafe_allow_html=True,
    )
    _render_section_depth("Ideas")

    if horizon.empty:
        st.info("No idea route file is available yet. Run the daily system first.")
        return

    # 1. Sector heatmap + movers — market context before picking
    _render_sector_alpha_strip()
    _render_alpha_movers()

    # 2. Alpha leaderboard — always visible first
    _render_alpha_screener()

    # 2. Watched ticker cards — the main daily view
    _render_ideas_cards(horizon, max_cards=12)

    # 3. Momentum radar — collapsible
    _render_momentum_signal_panel()

    # 4. Deeper research context — collapsed by default
    with st.expander("Idea routing guide — how to pick short / medium / long and stock vs option", expanded=False):
        _render_ideas_command_center(horizon, options_route, playbook, timeframe)
        _render_ideas_route_guide()
        _render_ideas_horizon_lanes(timeframe)
        _render_ideas_vehicle_board(horizon, options_route, playbook)

    show_detail = st.checkbox("Show technical idea files", value=False)
    if not show_detail:
        return

    st.markdown("---")
    detail_view = st.radio(
        "Idea detail to open",
        ["Time-frame table", "Option choices", "Daily queue", "Final review", "Older playbook", "Source files"],
        horizontal=True,
        label_visibility="collapsed",
    )
    _render_subtab_depth("Ideas", detail_view)
    if detail_view == "Time-frame table":
        cols = [c for c in ["ticker", "sector", "gate_status", "best_horizon", "short_action", "medium_action", "long_action", "current_desk_action", "trigger_to_watch", "main_blocker", "unlock_checklist"] if c in horizon.columns]
        _show_status_table(horizon[cols].head(80) if cols else horizon.head(80), ["gate_status", "current_desk_action"], height=720)
    elif detail_view == "Option choices":
        if not playbook.empty:
            st.markdown("#### Call / put playbook")
            p_cols = [c for c in ["ticker", "option_permission", "option_side", "option_answer", "call_answer", "put_answer", "primary_blocker", "what_would_change", "call_trigger", "put_trigger"] if c in playbook.columns]
            _show_status_table(playbook[p_cols].head(80) if p_cols else playbook.head(80), ["option_permission", "option_side"], height=560)
        if not options_route.empty:
            st.markdown("#### Trading-cost-aware option choice")
            o_cols = [c for c in ["ticker", "horizon", "gate_status", "final_vehicle_decision", "final_option_side", "final_option_structure", "no_go_count", "required_confirmation", "why_this_route", "source_files"] if c in options_route.columns]
            _show_status_table(options_route[o_cols].head(100) if o_cols else options_route.head(100), ["gate_status", "final_vehicle_decision", "final_option_side"], height=640)
    elif detail_view == "Daily queue":
        cols = [c for c in ["priority_rank", "priority", "ticker", "sector", "workflow_bucket", "best_horizon", "option_route", "risk_action", "what_to_watch", "what_would_change", "why"] if c in daily_queue.columns]
        _show_status_table(daily_queue[cols].head(80) if cols else daily_queue.head(80), ["priority", "risk_action"], height=720)
    elif detail_view == "Final review":
        cols = [c for c in ["ticker", "final_permission", "primary_route_now", "first_blocker", "where_to_click", "max_paper_weight_pct"] if c in promotion_gate.columns]
        _show_status_table(promotion_gate[cols].head(80) if cols else promotion_gate.head(80), ["final_permission"], height=640)
    elif detail_view == "Older playbook":
        if not timeframe.empty:
            st.markdown("#### Time-frame matrix")
            _show_status_table(timeframe.head(120), [], height=560)
        if not strategy.empty:
            st.markdown("#### Strategy route playbook")
            _show_status_table(strategy.head(120), [], height=560)
        if st.checkbox("Open the older detailed ideas page", value=False):
            _run_with_plain_streamlit_text(tab_timeframe_playbook)
    elif detail_view == "Source files":
        _render_risk_source_inventory([
            ("horizon_vehicle_summary.csv", "Main plain-English stock, call, put, and horizon summary"),
            ("options_execution_route_matrix.csv", "Option choice with safety, trading cost, and alerts"),
            ("options_playbook.csv", "Call and put playbook"),
            ("daily_workflow_queue.csv", "Daily ticker queue"),
            ("institutional_promotion_gate.csv", "Final permission gate"),
            ("timeframe_decision_matrix.csv", "Older time-frame matrix"),
            ("strategy_route_playbook.csv", "Older strategy route playbook"),
        ])


def _page_href(page: str) -> str:
    return f"?page={quote(str(page), safe='')}"


def _render_page_button(label: str, page: str, note: str = ""):
    # Colour-code by destination
    _page_colors = {
        "Risk":        ("#7f1d1d", "#fef2f2", "#991b1b"),   # dark red bg, light red fill, border
        "News":        ("#1e3a5f", "#eff6ff", "#1d4ed8"),
        "Ideas":       ("#14532d", "#f0fdf4", "#15803d"),
        "Live / Paper":("#1e293b", "#f8fafc", "#334155"),
    }
    bg, fill, border = _page_colors.get(page, ("#0f172a", "#f8fafc", "#1e293b"))
    _render_html(
        f"""
        <a href="{_page_href(page)}" target="_self" style="
            display:block;
            text-decoration:none;
            background:{bg};
            color:#fff;
            border:1px solid {border};
            border-radius:10px;
            padding:14px 16px;
            font-size:14px;
            font-weight:700;
            text-align:center;
            box-shadow:0 2px 6px rgba(0,0,0,.15);
            letter-spacing:-0.1px;
        ">{_esc(label)}</a>
        <div style="font-size:11.5px; color:#64748b; line-height:1.35; margin-top:6px; text-align:center;">{_esc(note)}</div>
        """
    )


def _home_plain(value, max_len: int = 180) -> str:
    text = _human_text(value, max_len=None)
    replacements = {
        "VaR": "loss limit",
        "CVaR": "bad-day loss limit",
        "size-down": "smaller size",
        "Size-down": "Smaller size",
        "execution-proof": "trading-cost proof",
        "Execution-proof": "Trading-cost proof",
        "spread/liquidity": "trading cost and volume",
        "Spread/liquidity": "Trading cost and volume",
        "proof/risk/execution gates": "proof, risk, and trading-cost checks",
        "Proof/risk/execution gates": "Proof, risk, and trading-cost checks",
        "watch-only option routes": "option ideas waiting for confirmation",
        "weekly calls or puts": "short-term options",
        "no option, call research, put research, spread only, or wait for trigger": "no option, call idea, put idea, spread idea, or wait",
        "PM acceptance": "research approval",
        "model-generated text": "AI-generated text",
        "risk-not ready yet tickers": "tickers risk does not allow yet",
        "Risk blocks action tickers": "tickers risk does not allow yet",
        "risk blocks action tickers": "tickers risk does not allow yet",
        "spread and liquidity": "trading cost and volume",
        "Spread and liquidity": "Trading cost and volume",
        "market impact": "hard-to-fill trading cost",
        "Market impact": "Hard-to-fill trading cost",
        "event gap risk": "bad news or earnings jump risk",
        "Event gap risk": "Bad news or earnings jump risk",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = re.sub(r"risk\W*not ready yet tickers", "tickers risk does not allow yet", text, flags=re.IGNORECASE)
    text = text.replace("risk-not ready yet", "risk does not allow yet")
    text = text.replace("after spread, hard-to-fill trading cost, and fill risk", "because trading can be expensive or hard to fill")
    text = text.replace("missing spread or liquidity proof", "missing trading-cost proof")
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _home_panel_label(value) -> str:
    text = _plain_status(value, "Next step")
    mapping = {
        "Today Flow Navigator": "Daily map",
        "Source Proof Desk": "Proof to fill",
        "Risk Desk": "Risk check",
        "Execution Cost / Liquidity": "Trading cost check",
        "Options Route": "Options check",
    }
    return mapping.get(text, _home_plain(text, 80))


def _render_human_start_here():
    brief = safe_json(ROOT / "pm_morning_brief_state.json")
    command = safe_json(ROOT / "quant_fund_flow_pm_command_center.json")
    flow_state = safe_json(ROOT / "quant_fund_operating_flow_state.json")
    proof_state = safe_json(ROOT / "quant_fund_proof_closure_state.json")
    risk_state = safe_json(ROOT / "risk_desk_overview.json")
    next_clicks = safe_csv(ROOT / "quant_fund_flow_next_clicks.csv")
    news = safe_csv(ROOT / "pm_morning_brief_news_to_verify.csv")

    answer = _plain_status(
        brief.get("desk_answer")
        or command.get("plain_answer")
        or flow_state.get("plain_answer"),
        "Run the daily system, then read this page first.",
    )
    risk_answer = _plain_status(
        brief.get("risk_answer")
        or risk_state.get("logic"),
        "Risk must be checked before any idea.",
    )
    first_page = _plain_status(command.get("first_page"), "Risk")
    first_ticker = _plain_status(command.get("first_ticker") or proof_state.get("top_action_ticker"), "AAPL")
    first_action = _human_text(
        command.get("first_action")
        or proof_state.get("top_action")
        or "Fill the missing proof fields before accepting any idea.",
        240,
    )
    proof_count = int(_to_float(proof_state.get("fill_first_count"), 0) or 0)
    risk_blocked = int(_to_float(command.get("risk_blocked_count"), 0) or 0)
    hard_breaches = int(_to_float(risk_state.get("budget_hard_breach_count"), 0) or 0)
    trust_now = _to_float(brief.get("proof_adjusted_sharpe"))
    trust_text = f"{trust_now:.2f}" if trust_now is not None else "No data"
    gross = _to_float(risk_state.get("recommended_gross_exposure"))
    gross_text = f"{gross * 100:.0f}%" if gross is not None else "No data"

    st.markdown('<p class="section-title">Start Here</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">This is the simple daily path. Read this first, then click one page. Everything is research-only and paper-only.</p>',
        unsafe_allow_html=True,
    )

    _can_add = "No" if "no" in answer.lower() or "risk first" in answer.lower() else "Maybe"
    _hero_accent = "#dc2626" if _can_add == "No" else "#0f172a"
    _render_html(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 14px;
            padding: 26px 28px 22px 28px;
            margin: 10px 0 20px 0;
            box-shadow: 0 4px 20px rgba(15,23,42,.18);
            position: relative;
            overflow: hidden;
        ">
          <div style="position:absolute;top:0;right:0;width:180px;height:180px;background:radial-gradient(circle at 100% 0%,rgba(99,102,241,.15) 0%,transparent 70%);pointer-events:none;"></div>
          <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.1em;">Today's answer</div>
          <div style="font-size:2rem; color:#f8fafc; font-weight:900; line-height:1.15; margin-top:8px; letter-spacing:-0.5px;">{_esc(answer)}</div>
          <div style="height:1px; background:rgba(255,255,255,.08); margin:16px 0 14px 0;"></div>
          <div style="display:flex; gap:32px; flex-wrap:wrap;">
            <div>
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Risk</div>
              <div style="font-size:13px; color:#fca5a5; font-weight:600; margin-top:3px; line-height:1.4;">{_esc(risk_answer)}</div>
            </div>
            <div>
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Size cap</div>
              <div style="font-size:13px; color:#86efac; font-weight:700; margin-top:3px;">{_esc(gross_text)}</div>
            </div>
            <div>
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Trust score</div>
              <div style="font-size:13px; color:#93c5fd; font-weight:700; margin-top:3px;">{_esc(trust_text)}</div>
            </div>
            <div>
              <div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Hard breaks</div>
              <div style="font-size:13px; color:{'#fca5a5' if hard_breaches else '#86efac'}; font-weight:700; margin-top:3px;">{hard_breaches}</div>
            </div>
          </div>
          <div style="margin-top:14px; font-size:11px; color:#475569;">No broker connection · No live orders · Research and paper only</div>
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _simple_card("Can I add anything?", _can_add, "If this is No, do not look for calls, puts, or new size.", "#dc2626" if _can_add == "No" else "#16a34a")
    with c2:
        _simple_card("Start with", f"{first_page}: {first_ticker}", first_action, "#0f172a")
    with c3:
        _simple_card("Why blocked?", f"{proof_count} proof / {risk_blocked} risk", "Proof or risk must clear before ideas can move forward.", "#475569")
    with c4:
        _simple_card("First fix", first_action[:60] + "…" if len(first_action) > 60 else first_action, f"Trust: {trust_text}  ·  Breaks: {hard_breaches}", "#475569")

    _render_weekly_pulse()

    st.markdown("#### Quick navigation")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        _render_page_button("🔒  Check Risk first", "Risk", "Always the first page.")
    with b2:
        _render_page_button("📰  Read News", "News", "Only after risk context.")
    with b3:
        _render_page_button("💡  Browse Ideas", "Ideas", "Only after blockers clear.")
    with b4:
        _render_page_button("📒  Paper account", "Live / Paper", "No broker. Paper tracking only.")

    st.markdown("#### What to do now — in order")
    if next_clicks.empty:
        st.info("No work queue is available. Run the daily system first.")
    else:
        for idx, (_, row) in enumerate(next_clicks.head(5).iterrows(), start=1):
            page = _plain_status(row.get("page"), "Home")
            panel = _home_panel_label(row.get("panel"))
            task = _home_plain(row.get("what_to_read"), 170)
            why = _home_plain(row.get("why_now"), 170)
            done = _home_plain(row.get("done_when"), 170)
            avoid = _home_plain(row.get("do_not_do"), 170)
            accent = "#dc2626" if page == "Risk" else "#334155" if page in {"Home", "Ideas"} else "#0f766e"
            open_label = "Stay here" if page == "Home" else f"Open {page} →"
            if panel == "Risk check":
                task = "Check the 9 tickers that risk does not allow yet. Some may need smaller size or no exposure."
                why = "Safety is the stop sign. A score, headline, or option idea cannot override it."
                done = "You know whether each ticker is tiny paper, watch-only, or not ready yet."
                avoid = "Do not let a good headline override loss limits, concentration, or bad news risk."
            elif panel == "Trading cost check":
                task = "Check whether trading this ticker would be too expensive or too hard to fill."
                why = "A good signal can disappear if the trade is expensive or hard to fill."
                done = "A manual quote or better intraday source confirms trading cost and volume."
                avoid = "Do not size a ticker with missing trading-cost proof."
            _render_html(
                f"""
                <div style="
                    background:#fff;
                    border:1px solid #e2e8f0;
                    border-left:4px solid {accent};
                    border-radius:10px;
                    padding:16px 18px;
                    margin:0 0 10px 0;
                    box-shadow:0 1px 4px rgba(0,0,0,.05);
                ">
                  <div style="display:flex; justify-content:space-between; gap:12px; align-items:center;">
                    <div style="font-size:10px; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Step {idx} &nbsp;·&nbsp; {_esc(page)}</div>
                    <a href="{_page_href(page)}" target="_self" style="font-size:12px; color:{accent}; font-weight:700; text-decoration:none; background:{'#fef2f2' if accent=='#dc2626' else '#f0fdf4' if accent=='#0f766e' else '#f8fafc'}; padding:3px 10px; border-radius:99px;">{_esc(open_label)}</a>
                  </div>
                  <div style="font-size:17px; color:#0f172a; font-weight:800; line-height:1.25; margin-top:8px;">{_esc(panel)}</div>
                  <div style="font-size:13.5px; color:#1e293b; line-height:1.5; margin-top:7px;"><b>Do this:</b> {_esc(task)}</div>
                  <div style="font-size:12.5px; color:#475569; line-height:1.45; margin-top:5px;"><b>Why:</b> {_esc(why)}</div>
                  <div style="font-size:12.5px; color:#475569; line-height:1.45; margin-top:5px;"><b>Done when:</b> {_esc(done)}</div>
                  <div style="border-top:1px solid #e5e7eb; margin-top:10px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.4;"><b>Do not:</b> {_esc(avoid)}</div>
                </div>
                """
            )

    st.markdown("#### Plain glossary")
    g1, g2, g3 = st.columns(3)
    with g1:
        _simple_card("Blocked", "Not ready yet", "It does not mean forever. It means proof, risk, or liquidity is missing.", "#111827")
    with g2:
        _simple_card("Proof", "Outside evidence", "A source name, observed value, reviewer, and date. Model text alone does not count.", "#334155")
    with g3:
        _simple_card("Options", "Last step", "Calls and puts come after risk, proof, news timing, and trading cost checks.", "#991b1b")

    if not news.empty:
        if st.toggle("Show the top news items that still need proof", value=False):
            for _, row in news.head(3).iterrows():
                _render_html(
                    f"""
                    <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid #334155; border-radius:8px; padding:13px 15px; margin:0 0 10px 0;">
                      <div style="font-size:12px; color:#6b7280; font-weight:900;">{_esc(_plain_status(row.get("ticker"), "Ticker"))}</div>
                      <div style="font-size:17px; color:#111827; font-weight:900; line-height:1.3; margin-top:5px;">{_esc(_human_text(row.get("headline"), 140))}</div>
                      <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:7px;"><b>Check:</b> {_esc(_human_text(row.get("why_to_check"), 170))}</div>
                    </div>
                    """
                )


def _render_light_proof_summary():
    closure_state = safe_json(ROOT / "quant_fund_proof_closure_state.json")
    intake_state = safe_json(ROOT / "quant_fund_proof_intake_state.json")
    quality_state = safe_json(ROOT / "quant_fund_proof_quality_gate_state.json")
    stage_counts = safe_csv(ROOT / "quant_fund_proof_closure_stage_counts.csv")
    next_actions = safe_csv(ROOT / "quant_fund_proof_closure_next_actions.csv")

    if not closure_state and not intake_state and not quality_state:
        st.info("Proof-to-fill check has not run yet. Run the daily system first.")
        return

    st.markdown('<p class="section-title">Proof To Fill Summary</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">The short version: what is blocking proof closure and what to do next.</p>',
        unsafe_allow_html=True,
    )

    answer = _human_text(
        closure_state.get("plain_answer")
        or intake_state.get("plain_answer")
        or quality_state.get("plain_answer"),
        560,
    )
    top_ticker = _plain_status(closure_state.get("top_action_ticker") or intake_state.get("first_ticker"), "No ticker")
    top_action = _human_text(closure_state.get("top_action") or intake_state.get("first_task"), 360)
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #111827; border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase; letter-spacing:.02em;">Proof status</div>
          <div style="font-size:23px; color:#111827; font-weight:900; line-height:1.28; margin-top:7px;">{_esc(answer)}</div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:9px;"><b>Next:</b> {_esc(top_ticker)} / {_esc(top_action)}</div>
        </div>
        """
    )

    metrics = [
        ("Proof rows", str(int(_to_float(quality_state.get("proof_rows"), 0) or 0)), "Rows checked for evidence quality.", "#334155"),
        ("Ready", str(int(_to_float(quality_state.get("ready_rows"), 0) or 0)), "Rows ready for bridge.", "#0f766e" if int(_to_float(quality_state.get("ready_rows"), 0) or 0) else "#334155"),
        ("Fields To Fill", str(int(_to_float(intake_state.get("user_entry_rows"), 0) or 0)), "Rows in the user entry sheet.", "#334155"),
        ("Apply Requests", str(int(_to_float(intake_state.get("apply_request_count"), 0) or 0)), "Rows marked APPLY.", "#111827"),
        ("Need Proof First", str(int(_to_float(closure_state.get("fill_first_count"), 0) or 0)), "Tickers still blocked by proof.", "#991b1b"),
    ]
    cols = st.columns(5)
    for col, (title, value, note, accent) in zip(cols, metrics):
        with col:
            _simple_card(title, value, note, accent)

    if not stage_counts.empty:
        s_cols = [c for c in ["stage_order", "stage_name", "row_count", "plain_meaning"] if c in stage_counts.columns]
        _show_status_table(stage_counts[s_cols] if s_cols else stage_counts, [], height=300)
    if not next_actions.empty:
        st.markdown("##### Next proof actions")
        a_cols = [c for c in ["action_rank", "ticker", "action", "page_or_file", "done_when"] if c in next_actions.columns]
        _show_status_table(next_actions[a_cols].head(5) if a_cols else next_actions.head(5), [], height=280)


def tab_run_system():
    st.markdown('<p class="section-title">Home</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">A simple front page for the daily answer, the first click, and what not to do.</p>', unsafe_allow_html=True)

    _render_human_start_here()

    st.markdown("---")
    if st.toggle("Refresh or rebuild data", value=False, help="Open this only when you want to rerun the local research files."):
        _render_daily_run_panel()

    if st.toggle("Show detailed dashboard panels", value=False, help="Open this when you want the full research system view."):
        _render_quant_fund_flow_navigator(compact=True)
        st.markdown("---")
        _render_light_proof_summary()
        st.markdown("---")
        _render_promotion_gate_panel(compact=True, label="Home")
        st.markdown("---")
        _render_pm_morning_brief()
        st.markdown("---")
        _render_beginner_click_guide()
        st.markdown("---")
        _render_ticker_flow_cards(compact=True)
        st.markdown("---")
        _render_proof_collection_workbench(compact=True)
        st.markdown("---")
        _render_proof_quality_gate(compact=True)
        st.markdown("---")
        _render_proof_fill_desk(compact=True)
        st.markdown("---")
        _render_proof_intake_safe_apply(compact=True)
        st.markdown("---")
        _render_proof_closure_tracker(compact=True)
        st.markdown("---")
        _render_quant_fund_operating_flow(compact=True)

    if st.toggle("Show advanced evidence and system files", value=False, help="Debug view. This can be long and technical."):
        _render_decision_memory_center(compact=True)
        st.markdown("---")
        _render_data_reliability_center(compact=True)
        st.markdown("---")
        _render_data_repair_center(compact=True)
        st.markdown("---")
        _render_risk_book_seed_center(compact=True)
        st.markdown("---")
        _render_risk_seed_approval_workbench(compact=True)
        st.markdown("---")
        _render_risk_seed_pm_review_intake(compact=True)
        st.markdown("---")
        _render_pm_review_evidence_autofill(compact=True)
        st.markdown("---")
        _render_pm_evidence_acceptance_gate(compact=True)
        st.markdown("---")
        _render_pm_evidence_review_triage(compact=True)
        st.markdown("---")
        _render_pm_evidence_source_proof_desk(compact=True)
        st.markdown("---")
        _render_pm_evidence_proof_acceptance_bridge(compact=True)
        st.markdown("---")
        _render_pm_review_final_gate_bridge(compact=True)
        st.markdown("---")
        _render_deep_logic_chain()
        st.markdown("---")
        _render_pm_operating_order()
        st.markdown("---")
        _render_sector_theme_depth()
        _show_deep_decision_desk()
        st.markdown("---")
        run_log = load_run_log()
        if not run_log.empty and "date" in run_log.columns and "status" in run_log.columns:
            last_date = str(run_log["date"].max())
            last_rows = run_log[run_log["date"] == run_log["date"].max()]
            passed = int((last_rows["status"] == "OK").sum())
            total = int(len(last_rows))
            failed = total - passed
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Last refresh", last_date[:16])
            with c2:
                st.metric("Checks passed", f"{passed} / {total}")
            with c3:
                st.metric("Needs attention", failed)
        else:
            st.info("No run log found yet. Use Run Daily System Now at the top of this page.")
        _render_everything_map()
        _render_active_spine()
        _show_dynamic_daily_workflow()


def _system_plain(value, max_len: int | None = 220) -> str:
    text = _human_text(value, max_len=None)
    lower = text.lower()
    if "tabulate" in lower:
        text = "Missing tabulate package. Install tabulate, then rerun the daily research run."
    elif "killed after 180" in lower:
        text = "This step timed out after 180 seconds. Rerun later or keep it in the repair list."
    elif "killed after 60" in lower:
        text = "This step timed out after 60 seconds. Rerun later or keep it in the repair list."
    elif "traceback" in lower or "site-packages" in lower or "file \"/" in lower:
        text = "Python error in this step. Open detailed system tables only if you need the raw traceback."
    replacements = {
        "OK": "passed",
        "FAILED": "failed",
        "Failed": "failed",
        "TIMEOUT": "timed out",
        "Timeout": "timed out",
        "WARN": "warning",
        "Warn": "warning",
        "DATA_REPAIR_ACTIVE": "data repair is active",
        "Data Repair Active": "data repair is active",
        "PM_BRIEF_ACTIVE": "morning brief is active",
        "PM Brief Active": "morning brief is active",
        "Proof workflow": "proof to fill",
        "Proof Workflow": "Proof to fill",
        "REPAIR_REQUIRED": "repair required",
        "Repair Required": "repair required",
        "Run Center": "Update / Fix",
        "RISK_FIRST": "risk first",
        "Risk First": "risk first",
        "Proof First": "proof first",
        "Proof first": "proof first",
        "No Live Orders": "no live orders",
        "no live order": "no live order",
        "ImportError: `Import tabulate` failed": "Missing tabulate package",
        "Import tabulate failed": "Missing tabulate package",
        "killed after 180s": "timed out after 180 seconds",
        "killed after 60s": "timed out after 60 seconds",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = " ".join(text.split())
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text or "No data"


def _system_accent(value) -> str:
    text = str(value or "").upper()
    if any(x in text for x in ["FAILED", "TIMEOUT", "REPAIR", "CRITICAL", "MISSING", "WARN"]):
        return "#991b1b"
    if any(x in text for x in ["PROOF", "REVIEW", "WAIT", "CAUTIOUS"]):
        return "#334155"
    if any(x in text for x in ["OK", "PASS", "ACTIVE", "CLEAR", "FRESH"]):
        return "#166534"
    return "#111827"


def _latest_run_rows(run_log: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if run_log is None or run_log.empty or "date" not in run_log.columns:
        return "No run log", pd.DataFrame()
    work = run_log.copy()
    work["_run_time"] = pd.to_datetime(work["date"], errors="coerce")
    if work["_run_time"].notna().any():
        latest = work["_run_time"].max()
        rows = work[work["_run_time"].eq(latest)].copy()
        return str(latest)[:16], rows
    latest_raw = str(work["date"].iloc[-1])
    return latest_raw[:16], work[work["date"].astype(str).eq(latest_raw)].copy()


def _system_command_card(title: str, why: str, command: str, accent: str = "#111827"):
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid {accent}; border-radius:8px; padding:14px 15px; min-height:145px; margin-bottom:10px;">
          <div style="font-size:16px; color:#111827; font-weight:900;">{_esc(title)}</div>
          <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:7px;">{_esc(why)}</div>
        </div>
        """
    )
    st.code(command, language="bash")


def _browser_run_state() -> dict:
    try:
        if BROWSER_DAILY_RUN_STATE.exists():
            return json.loads(BROWSER_DAILY_RUN_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _write_browser_run_state(state: dict):
    try:
        BROWSER_DAILY_RUN_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def _pid_is_running(pid) -> bool:
    try:
        pid_int = int(pid)
        if pid_int <= 0:
            return False
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def _tail_text(path: Path, max_lines: int = 18) -> str:
    try:
        if not path.exists():
            return "No log yet."
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:]) if lines else "Log file is empty so far."
    except Exception as exc:
        return f"Could not read log: {exc}"


def _start_browser_daily_update() -> dict:
    state = _browser_run_state()
    pid = state.get("pid")
    if _pid_is_running(pid):
        state["status"] = "running"
        return state

    runner = ROOT / "canyon_final_v9_step70_daily_runner_all.py"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    command = [sys.executable, "-u", str(runner)]
    with BROWSER_DAILY_RUN_LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n\n=== Browser daily update started {now} ===\n")
        log.write("Command: " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    state = {
        "status": "running",
        "pid": proc.pid,
        "started_at": now,
        "command": " ".join(command),
        "log_file": str(BROWSER_DAILY_RUN_LOG),
        "note": "Research-only data update. No broker connection. No live orders.",
    }
    _write_browser_run_state(state)
    return state


def _render_browser_daily_update_button():
    state = _browser_run_state()
    running = _pid_is_running(state.get("pid"))
    status = "Running" if running else "Ready"
    status_note = (
        f"Started {state.get('started_at', 'recently')} · PID {state.get('pid')}"
        if running
        else "Updates prices, news, risk files, and dashboard outputs."
    )
    accent = "#0f766e" if running else "#111827"
    button_note = (
        "Update is already running. Wait a few minutes, then use Refresh."
        if running
        else "Click once. It runs in the background."
    )

    st.markdown(
        """
        <style>
        button[data-testid="stBaseButton-primary"] {
            background: #111827 !important;
            border-color: #111827 !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            box-shadow: none !important;
        }
        button[data-testid="stBaseButton-primary"] p {
            color: #ffffff !important;
            font-weight: 850 !important;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background: #1f2937 !important;
            border-color: #1f2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        c1, c2 = st.columns([4, 1.35])
        with c1:
            st.markdown("**Quick action**")
            st.markdown("### Update research data")
            st.caption(status_note)
            st.caption(button_note)
        with c2:
            _render_html(
                f"""
                <div style="border:1px solid #d1d5db; border-left:4px solid {accent}; border-radius:999px; padding:7px 12px; color:{accent}; font-size:12px; font-weight:900; text-align:center; margin-bottom:10px;">{_esc(status)}</div>
                """
            )
            if st.button("Update Now", type="primary", disabled=running, use_container_width=True, help="Starts the local research update in the background."):
                state = _start_browser_daily_update()
                st.success("Update started. Wait a few minutes, then click Refresh.")
                running = True
        if running:
            st.info("Running in the background. You can keep using the website, then refresh when it finishes.")

    with st.expander("What happened in the update?", expanded=False):
        st.code(_tail_text(BROWSER_DAILY_RUN_LOG), language="text")
        st.caption(f"Log file: {BROWSER_DAILY_RUN_LOG}")


def _render_where_everything_lives():
    st.markdown("#### Where Everything Lives")
    _render_html(
        """
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #111827; border-radius:9px; padding:17px 19px; margin:8px 0 16px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">Simple answer</div>
          <div style="font-size:25px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">The old tools are still part of the system. They are grouped under fewer pages so the website is readable.</div>
          <div style="font-size:13px; color:#374151; line-height:1.55; margin-top:9px;">Use the top pages first. Open the deeper tools only when you need proof, raw files, or a repair path.</div>
        </div>
        """
    )

    groups = [
        {
            "page": "Home",
            "title": "Daily answer and map",
            "now": "Start here when you want the plain answer and the whole system map.",
            "includes": "Overview, master decision, layer map, strategy scorecard, beginner guide.",
            "why": "This should answer: where do I start?",
        },
        {
            "page": "Today",
            "title": "What to do first",
            "now": "Daily workflow, first page, first ticker, focus list, alerts, and proof plan.",
            "includes": "Daily brief, daily desk, focus list, trigger board, decision playbook.",
            "why": "This should answer: what is today's order of work?",
        },
        {
            "page": "Ideas",
            "title": "Ticker decisions",
            "now": "Short / medium / long view, stock vs call vs put, and why an idea is blocked.",
            "includes": "Action board, pre-trade check, options lab, route guide, ticker dossier.",
            "why": "This should answer: what can I do with this ticker?",
        },
        {
            "page": "News",
            "title": "News and industry chain",
            "now": "Headlines, who may benefit, who may get hurt, and what proof is still missing.",
            "includes": "Research stack, research lab, news proof queue, supply-chain read-through.",
            "why": "This should answer: does this news actually matter?",
        },
        {
            "page": "Risk",
            "title": "Can the account add risk?",
            "now": "Portfolio safety, bad-day loss, correlation, liquidity, earnings, and repair path.",
            "includes": "Risk control, portfolio map, stress test, optimizer, factor exposure.",
            "why": "This should answer: are we allowed to add anything?",
        },
        {
            "page": "Performance",
            "title": "Can we trust the model?",
            "now": "Backtest trust, signal follow-up, decay, failure cases, and trading-cost proof.",
            "includes": "Backtest, model signals, SHAP-style explanation, signal lab, proof board.",
            "why": "This should answer: is this model really working?",
        },
        {
            "page": "Live / Paper",
            "title": "Paper account and feedback",
            "now": "Paper ledger, manual account value, alerts, position triage, and notes.",
            "includes": "Paper ledger, paper simulation, live monitor, trade journal.",
            "why": "This should answer: what happened after we watched or paper traded?",
        },
        {
            "page": "System",
            "title": "Fixes, files, and reports",
            "now": "Run status, update button, file health, report archive, data repair, and QA.",
            "includes": "Command center, report archive, output vault, data health, QA, weekly report.",
            "why": "This should answer: what is broken or stale?",
        },
    ]

    html = ['<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin:8px 0 18px 0;">']
    for item in groups:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid #111827; border-radius:8px; padding:14px 15px; min-height:230px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div>
                  <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">{_esc(item['page'])}</div>
                  <div style="font-size:20px; color:#111827; font-weight:900; line-height:1.2; margin-top:5px;">{_esc(item['title'])}</div>
                </div>
                <a href="{_page_href(item['page'])}" target="_self" style="white-space:nowrap; text-decoration:none; background:#111827; color:#fff; border:1px solid #111827; border-radius:8px; padding:8px 11px; font-size:12px; font-weight:850;">Open</a>
              </div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:11px;"><b>Use it for:</b> {_esc(item['now'])}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;"><b>Earlier tools inside:</b> {_esc(item['includes'])}</div>
              <div style="border-top:1px solid #e5e7eb; padding-top:8px; margin-top:10px; font-size:12px; color:#6b7280; line-height:1.35;">{_esc(item['why'])}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))

    st.markdown("##### How to use the website without getting lost")
    steps = [
        ("1", "Start with Today", "It tells you the first page and first ticker."),
        ("2", "Check Risk before Ideas", "If Risk says wait, every idea stays research-only."),
        ("3", "Use News for stories", "News must show who may benefit, who may get hurt, and what proof is missing."),
        ("4", "Use Performance for trust", "A model score only matters if newer data and trading cost still support it."),
        ("5", "Use System only to fix", "System is for stale files, failed runs, reports, and repair work."),
    ]
    html = ['<div style="display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:10px; margin:8px 0 14px 0;">']
    for num, title, text in steps:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:12px; min-height:125px;">
              <div style="font-size:12px; color:#6b7280; font-weight:900;">{_esc(num)}</div>
              <div style="font-size:15px; color:#111827; font-weight:900; line-height:1.25; margin-top:5px;">{_esc(title)}</div>
              <div style="font-size:12px; color:#4b5563; line-height:1.38; margin-top:7px;">{_esc(text)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))

    with st.expander("Original technical names, grouped by current page", expanded=False):
        mapping = pd.DataFrame(
            [
                {"Current page": "Home", "Earlier names": "tab_overview, tab_master, tab_strategy_scorecard, tab_layers, tab_l2_l6, tab_l7_l10"},
                {"Current page": "Today", "Earlier names": "tab_daily_brief, tab_daily_desk, tab_focus_list, tab_trigger_board, tab_decision_playbook"},
                {"Current page": "Ideas", "Earlier names": "tab_action, tab_pre_trade_gate, tab_options_chain, tab_options_lab, tab_ticker_dossier"},
                {"Current page": "News", "Earlier names": "tab_research_stack, tab_research_lab, tab_news_room, news proof and industry-chain modules"},
                {"Current page": "Risk", "Earlier names": "tab_risk_control, tab_portfolio_map, tab_risk_stress, tab_advanced_risk, tab_portfolio_optimizer, tab_factor_attribution"},
                {"Current page": "Performance", "Earlier names": "tab_backtest, tab_ml_signals, tab_shap_explainer, signal IC / decay / failure lab"},
                {"Current page": "Live / Paper", "Earlier names": "tab_paper_ledger, tab_paper_sim, paper monitor, trade journal"},
                {"Current page": "System", "Earlier names": "tab_command_center, tab_report_archive, tab_output_vault, tab_data_source_health, tab_system_qa, tab_daily_runner, tab_alerts, tab_weekly_report"},
            ]
        )
        st.dataframe(mapping, use_container_width=True, hide_index=True, height=330)


def _render_system_run_center():
    pm_state = safe_json(ROOT / "pm_morning_brief_state.json")
    data_state = safe_json(ROOT / "data_reliability_state.json")
    repair_state = safe_json(ROOT / "data_repair_state.json")
    flow_state = safe_json(ROOT / "quant_fund_flow_navigator_state.json")
    desk_state = safe_json(ROOT / "desk_monitor_summary.json")
    run_log = safe_csv(ROOT / "run_daily_all_log.csv")
    health = safe_csv(ROOT / "system_health_check.csv")
    manifest = safe_csv(ROOT / "canyon_file_manifest.csv")
    repair_board = safe_csv(ROOT / "data_repair_priority_board.csv")
    quality_flags = safe_csv(ROOT / "data_quality_flags.csv")

    latest_run, latest_rows = _latest_run_rows(run_log)
    passed = int(latest_rows.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("OK").sum()) if not latest_rows.empty else 0
    total = int(len(latest_rows)) if not latest_rows.empty else 0
    failed = max(total - passed, 0)
    critical = int(_to_float(desk_state.get("critical_count"), pm_state.get("critical_events", 0)) or 0)
    repair_rows = int(_to_float(data_state.get("repair_queue_rows", 0), 0) or 0)
    freshness = _to_float(data_state.get("overall_score_0_100"), None)
    stale_files = 0
    empty_files = 0
    if not manifest.empty and "status" in manifest.columns:
        status_text = manifest["status"].astype(str).str.upper()
        stale_files = int(status_text.str.contains("STALE", na=False).sum())
        empty_files = int(status_text.str.contains("EMPTY", na=False).sum())

    daily_answer = _system_plain(pm_state.get("desk_answer", "Run the daily system, then start with Risk."), 260)
    first_page = _system_plain(flow_state.get("first_page", "Risk"), 80)
    first_ticker = _system_plain(flow_state.get("first_ticker", "No ticker"), 80)
    first_action = _system_plain(flow_state.get("plain_answer", ""), 300)
    if failed:
        run_answer = f"Last full run passed {passed}/{total}. Fix {failed} failed or timed-out steps before trusting every panel."
        accent = "#991b1b"
    elif total:
        run_answer = f"Last full run passed {passed}/{total}. Still read Safety and proof checks before ideas."
        accent = "#166534"
    else:
        run_answer = "No full run log found. Use the daily runner command below, then refresh this page."
        accent = "#334155"

    st.markdown("#### Update / Fix The Site")
    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {accent}; border-radius:9px; padding:17px 19px; margin:8px 0 16px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:850; text-transform:uppercase;">Simple system answer</div>
          <div style="font-size:25px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(run_answer)}</div>
          <div style="font-size:13px; color:#374151; line-height:1.55; margin-top:9px;">Today's answer: <b>{_esc(daily_answer)}</b></div>
          <div style="font-size:12px; color:#6b7280; line-height:1.45; margin-top:8px;">First page: {_esc(first_page)} · First ticker: {_esc(first_ticker)} · Research-only · No broker connection · No live orders.</div>
        </div>
        """
    )

    _render_browser_daily_update_button()

    cols = st.columns(6)
    cards = [
        ("Last full run", latest_run, f"{passed}/{total} passed.", accent),
        ("Data reliability", f"{freshness:.1f}/100" if freshness is not None else "No data", data_state.get("status", "No data"), _system_accent(data_state.get("status"))),
        ("Critical events", str(critical), "Protect first if this is above zero.", "#991b1b" if critical else "#166534"),
        ("Fix list", str(repair_rows), "Files or proof items that still need work.", "#991b1b" if repair_rows else "#166534"),
        ("Stale files", str(stale_files), f"Empty files: {empty_files}", "#991b1b" if stale_files or empty_files else "#166534"),
        ("Data warnings", str(len(quality_flags)), "Data quality flags from local files.", "#991b1b" if len(quality_flags) else "#166534"),
    ]
    for col, (title, value, note, card_accent) in zip(cols, cards):
        with col:
            _simple_card(title, value, _system_plain(note, 120), card_accent)

    st.markdown("##### Commands to run")
    c1, c2, c3 = st.columns(3)
    with c1:
        _system_command_card(
            "Daily research run",
            "Use this when prices, news, risk, or proof files look stale. It rebuilds the local research outputs.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 -u canyon_final_v9_step70_daily_runner_all.py",
            "#111827",
        )
    with c2:
        _system_command_card(
            "Restart this website",
            "Use this if the page is stale or the server was stopped. Keep port 8512 for the current browser URL.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\nstreamlit run canyon_final_v9_step86_dashboard_v3.py --server.port 8512",
            "#334155",
        )
    with c3:
        _system_command_card(
            "Fast code check",
            "Use this after code edits. It catches Python syntax errors before you open the site.",
            "cd ~/Desktop/canyon_quant\npython3 -m py_compile canyon_final_v9_step86_dashboard_v3.py",
            "#166534",
        )

    st.markdown("##### After running, read in this order")
    order = [
        ("1. Today", "Get the daily path and first click.", "Do not start with a ticker story."),
        ("2. Risk", "Check if the portfolio can add anything.", "If Risk says no, ideas stay watch-only."),
        ("3. News", "Read headline impact and source proof.", "No proof means no action."),
        ("4. Ideas", "Choose short / medium / long, then stock/call/put/wait.", "Options cannot override safety."),
        ("5. Performance", "Check if the model is actually proving itself.", "Do not trust a high score without proof."),
        ("6. Live / Paper", "Track paper book and market feedback.", "Paper is feedback, not real trading."),
    ]
    html = ['<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:8px 0 18px 0;">']
    for title, read, stop in order:
        html.append(
            f"""
            <div style="background:#fff; border:1px solid #d1d5db; border-left:4px solid #111827; border-radius:8px; padding:13px 14px; min-height:150px;">
              <div style="font-size:16px; color:#111827; font-weight:900;">{_esc(title)}</div>
              <div style="font-size:13px; color:#374151; line-height:1.45; margin-top:8px;">{_esc(read)}</div>
              <div style="border-top:1px solid #e5e7eb; padding-top:8px; margin-top:9px; font-size:12px; color:#6b7280; line-height:1.35;">{_esc(stop)}</div>
            </div>
            """
        )
    html.append("</div>")
    _render_html("".join(html))

    if not latest_rows.empty:
        failed_rows = latest_rows[~latest_rows["status"].astype(str).str.upper().eq("OK")].copy() if "status" in latest_rows.columns else pd.DataFrame()
        if not failed_rows.empty:
            st.markdown("##### Fix these run failures first")
            for start in range(0, min(len(failed_rows), 6), 3):
                cols = st.columns(3)
                for col, (_, row) in zip(cols, failed_rows.iloc[start:start + 3].iterrows()):
                    status = _system_plain(row.get("status"), 80)
                    notes = _system_plain(row.get("notes"), 210)
                    name = _system_plain(row.get("name"), 120)
                    step = _system_plain(row.get("step"), 40)
                    with col:
                        _render_html(
                            f"""
                            <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #991b1b; border-radius:8px; padding:13px 14px; min-height:180px; margin-bottom:10px;">
                              <div style="font-size:12px; color:#991b1b; font-weight:850;">{_esc(status)}</div>
                              <div style="font-size:17px; color:#111827; font-weight:900; line-height:1.25; margin-top:6px;">{_esc(step)} · {_esc(name)}</div>
                              <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:8px;">{_esc(notes)}</div>
                            </div>
                            """
                        )
            if failed_rows.get("notes", pd.Series(dtype=str)).astype(str).str.contains("tabulate", case=False, na=False).any():
                st.info("Several failed report steps mention tabulate. The likely fix is: `python3 -m pip install tabulate`, then rerun the daily research run.")

    if not repair_board.empty:
        st.markdown("##### Current fix list")
        for start in range(0, min(len(repair_board), 4), 4):
            cols = st.columns(4)
            for col, (_, row) in zip(cols, repair_board.iloc[start:start + 4].iterrows()):
                priority = _system_plain(row.get("priority"), 40)
                workstream = _system_plain(row.get("workstream"), 80)
                answer = _system_plain(row.get("plain_answer"), 170)
                next_step = _system_plain(row.get("next_step"), 150)
                with col:
                    _render_html(
                        f"""
                        <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {_system_accent(priority)}; border-radius:8px; padding:13px 14px; min-height:220px; margin-bottom:12px;">
                          <div style="font-size:12px; color:#6b7280; font-weight:850;">{_esc(priority)}</div>
                          <div style="font-size:17px; color:#111827; font-weight:900; margin-top:6px;">{_esc(workstream)}</div>
                          <div style="font-size:13px; color:#374151; line-height:1.42; margin-top:8px;">{_esc(answer)}</div>
                          <div style="border-top:1px solid #e5e7eb; margin-top:9px; padding-top:8px; font-size:12px; color:#6b7280; line-height:1.35;">{_esc(next_step)}</div>
                        </div>
                        """
                    )

    with st.expander("Open detailed system tables", expanded=False):
        if not latest_rows.empty:
            cols = [c for c in ["date", "step", "name", "status", "duration_s", "returncode", "notes"] if c in latest_rows.columns]
            _show_status_table(latest_rows[cols] if cols else latest_rows, ["status"], height=520)
        if not health.empty:
            st.markdown("Health check")
            cols = [c for c in ["category", "item", "status", "detail", "action"] if c in health.columns]
            _show_status_table(health[cols] if cols else health, ["status"], height=420)
        if not manifest.empty:
            st.markdown("Important output files")
            cols = [c for c in ["file_name", "extension", "modified_at", "age_hours", "row_count", "ticker_count", "status"] if c in manifest.columns]
            _show_status_table(manifest[cols].head(120) if cols else manifest.head(120), ["status"], height=560)


def _render_walk_forward_oos_panel():
    summary  = safe_csv(ROOT / "wf_oos_summary.csv")
    ic_df    = safe_csv(ROOT / "wf_oos_ic_by_period.csv")
    perf_df  = safe_csv(ROOT / "wf_oos_backtest_perf.csv")
    equity   = safe_csv(ROOT / "wf_oos_equity_curve.csv")

    st.markdown('<p class="section-title">Walk-Forward Out-of-Sample Backtest</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Data before 2020 = in-sample (IS). Data from 2020 onward = locked test set (OOS). '
        'The model never trained on OOS data. OOS IC and return are the honest performance numbers.</p>',
        unsafe_allow_html=True,
    )

    if summary.empty and ic_df.empty:
        _render_html(
            """
            <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid #334155;
                        border-radius:9px; padding:18px 20px; margin:10px 0 18px 0;">
              <div style="font-size:22px; color:#111827; font-weight:900; line-height:1.3;">
                Walk-Forward OOS results not yet generated
              </div>
              <div style="font-size:14px; color:#374151; line-height:1.55; margin-top:10px;">
                Run the backtest first to see results here.
              </div>
              <div style="background:#f8fafc; border-radius:6px; padding:12px 14px; margin-top:12px;
                          font-family:monospace; font-size:13px; color:#334155;">
                cd ~/Desktop/canyon_quant<br>
                source .venv/bin/activate<br>
                python3 canyon_final_v9_step100_walk_forward_oos.py
              </div>
            </div>
            """
        )
        return

    # ── Survivorship bias warning ──────────────────────────────────────────────
    _render_html(
        """
        <div style="background:#fffbeb; border:1px solid #d97706; border-left:6px solid #d97706;
                    border-radius:9px; padding:14px 18px; margin:8px 0 16px 0;">
          <div style="font-size:12px; font-weight:900; color:#92400e; text-transform:uppercase;">
            Data limitation — survivorship bias present
          </div>
          <div style="font-size:13px; color:#78350f; line-height:1.55; margin-top:7px;">
            Universe uses current S&P 500 tickers back to 2000. Delisted or failed companies are missing.
            All IC and return numbers are optimistic vs. a point-in-time universe.
            The OOS period below is still the most realistic test available.
          </div>
        </div>
        """
    )

    # ── Hero card: IS IC vs OOS IC ─────────────────────────────────────────────
    is_row  = ic_df[(ic_df["period"] == "IS")  & (ic_df["signal"] == "ensemble_score")] if not ic_df.empty else pd.DataFrame()
    oos_row = ic_df[(ic_df["period"] == "OOS") & (ic_df["signal"] == "ensemble_score")] if not ic_df.empty else pd.DataFrame()
    is_ic   = float(is_row["mean_ic"].iloc[0])  if not is_row.empty  else None
    oos_ic  = float(oos_row["mean_ic"].iloc[0]) if not oos_row.empty else None
    is_t    = float(is_row["t_stat"].iloc[0])   if not is_row.empty  else None
    oos_t   = float(oos_row["t_stat"].iloc[0])  if not oos_row.empty else None
    oos_status = str(oos_row["status"].iloc[0]) if not oos_row.empty else "NO DATA"

    if oos_ic is not None and oos_ic > 0.05:
        verdict_color = "#166534"
        verdict = "OOS IC is STRONG — model generalises to unseen data."
    elif oos_ic is not None and oos_ic > 0.03:
        verdict_color = "#0f766e"
        verdict = "OOS IC is USABLE — positive but watch for decay over time."
    elif oos_ic is not None and oos_ic > 0:
        verdict_color = "#334155"
        verdict = "OOS IC is WEAK — positive signal but not statistically reliable."
    elif oos_ic is not None:
        verdict_color = "#991b1b"
        verdict = "OOS IC is NEGATIVE — model does not hold on unseen data."
    else:
        verdict_color = "#334155"
        verdict = "OOS results not available yet."

    is_ic_str  = f"{is_ic:+.4f}"  if is_ic  is not None else "—"
    oos_ic_str = f"{oos_ic:+.4f}" if oos_ic is not None else "—"
    is_t_str   = f"t={is_t:+.2f}"  if is_t  is not None else ""
    oos_t_str  = f"t={oos_t:+.2f}" if oos_t is not None else ""
    decay_str  = (f"{(is_ic - oos_ic):+.4f}" if (is_ic is not None and oos_ic is not None) else "—")

    _render_html(
        f"""
        <div style="background:#fff; border:1px solid #d1d5db; border-left:6px solid {verdict_color};
                    border-radius:10px; padding:18px 20px; margin:10px 0 18px 0;">
          <div style="font-size:12px; color:#6b7280; font-weight:900; text-transform:uppercase;">
            Does the model work on data it has never seen?
          </div>
          <div style="font-size:26px; color:#111827; font-weight:900; line-height:1.2; margin-top:6px;">
            {_esc(verdict)}
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:16px;">
            <div style="border-top:3px solid #334155; background:#f9fafb; border-radius:8px; padding:12px 14px;">
              <div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase;">In-Sample IC (pre-2020)</div>
              <div style="font-size:28px; font-weight:900; color:#334155; margin-top:4px;">{_esc(is_ic_str)}</div>
              <div style="font-size:12px; color:#6b7280; margin-top:3px;">{_esc(is_t_str)}</div>
            </div>
            <div style="border-top:3px solid {verdict_color}; background:#f9fafb; border-radius:8px; padding:12px 14px;">
              <div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase;">Out-of-Sample IC (2020+)</div>
              <div style="font-size:28px; font-weight:900; color:{verdict_color}; margin-top:4px;">{_esc(oos_ic_str)}</div>
              <div style="font-size:12px; color:#6b7280; margin-top:3px;">{_esc(oos_t_str)} · {_esc(oos_status)}</div>
            </div>
            <div style="border-top:3px solid #64748b; background:#f9fafb; border-radius:8px; padding:12px 14px;">
              <div style="font-size:11px; color:#6b7280; font-weight:850; text-transform:uppercase;">IC Decay (IS → OOS)</div>
              <div style="font-size:28px; font-weight:900; color:#64748b; margin-top:4px;">{_esc(decay_str)}</div>
              <div style="font-size:12px; color:#6b7280; margin-top:3px;">Positive = degraded on new data</div>
            </div>
          </div>
        </div>
        """
    )

    # ── IS vs OOS summary table ────────────────────────────────────────────────
    if not summary.empty:
        st.markdown("##### In-Sample vs Out-of-Sample Metrics")
        display_cols = [c for c in ["metric", "in_sample", "out_of_sample"] if c in summary.columns]
        if display_cols:
            _show_status_table(summary[display_cols], [], height=360)

    # ── Equity curve ──────────────────────────────────────────────────────────
    if not equity.empty:
        st.markdown("##### Equity Curve: In-Sample (gray) vs Out-of-Sample (dark)")
        chart_rows: list[dict] = []
        for _, row in equity.iterrows():
            spy_n = _to_float(row.get("spy_nav"))
            entry: dict = {"Date": row.get("rebalance_date")}
            ml_is  = _to_float(row.get("ml_nav_is"))
            ml_oos = _to_float(row.get("ml_nav_oos"))
            if ml_is  is not None: entry["ML In-Sample"]      = ml_is
            if ml_oos is not None: entry["ML Out-of-Sample"]   = ml_oos
            if spy_n  is not None: entry["SPY Benchmark"]      = spy_n
            chart_rows.append(entry)
        if chart_rows:
            chart_df = pd.DataFrame(chart_rows).set_index("Date")
            chart_df = chart_df.apply(pd.to_numeric, errors="coerce")
            st.line_chart(chart_df, use_container_width=True, height=280)
        st.caption(
            "Gray line = in-sample (pre-2020), dark line = out-of-sample (2020+). "
            "Both start at 1.0 for their respective periods. SPY shown for reference."
        )

    # ── IC breakdown by period ─────────────────────────────────────────────────
    if not ic_df.empty:
        with st.expander("Open IC breakdown by signal and period", expanded=False):
            st.markdown("Spearman IC of each signal vs 21-day forward returns. IS = pre-2020, OOS = 2020 onward.")
            display_ic_cols = [c for c in ["signal", "period", "mean_ic", "t_stat", "ic_positive_pct", "status"] if c in ic_df.columns]
            _show_status_table(ic_df[display_ic_cols] if display_ic_cols else ic_df, ["status"], height=480)

    # ── Monthly performance detail ─────────────────────────────────────────────
    if not perf_df.empty:
        with st.expander("Open monthly portfolio performance", expanded=False):
            st.markdown("Monthly ML portfolio returns vs SPY, by period.")
            perf_cols = [c for c in ["rebalance_date", "period", "ml_ret", "spy_ret", "alpha", "n_held", "turnover_pct", "tickers"] if c in perf_df.columns]
            _show_status_table(perf_df[perf_cols] if perf_cols else perf_df, ["period"], height=520)

    # ── Run command ───────────────────────────────────────────────────────────
    with st.expander("Run or refresh the Walk-Forward OOS backtest", expanded=False):
        st.code(
            "cd ~/Desktop/canyon_quant\n"
            "source .venv/bin/activate\n"
            "python3 canyon_final_v9_step100_walk_forward_oos.py",
            language="bash",
        )
        st.caption(
            "Takes 5–15 minutes depending on cache. "
            "Outputs: wf_oos_predictions.csv, wf_oos_ic_by_period.csv, "
            "wf_oos_backtest_perf.csv, wf_oos_equity_curve.csv, wf_oos_summary.csv, wf_oos_report.md"
        )


_GLOBAL_CSS = """
<style>
/* ════════════════════════════════════════════════════════════════════════════
   CANYON QUANT v9 — Design System
   Source of truth: canyon_v9_research.html
   All classes below are copied verbatim from the HTML file so that
   st.markdown() HTML fragments render identically to the static page.
   ════════════════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ── Reset + base ─────────────────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box}
body,html,[data-testid],[class*="st-"]{
    font-family:'Source Sans 3','Helvetica Neue',Arial,sans-serif !important;
    background:#FAFAF8 !important;
    color:#1A1A1A !important;
    line-height:1.65 !important;
}
h1,h2,h3,h4{font-family:'Playfair Display',Georgia,serif !important}

/* ── Streamlit chrome ─────────────────────────────────────────────────────── */
.stApp,[data-testid="stAppViewContainer"],section[data-testid="stMain"]{background:#FAFAF8 !important}
[data-testid="stHeader"]{display:none !important;height:0 !important}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{visibility:hidden !important;height:0 !important}
.block-container{max-width:1080px !important;padding:0 48px 80px 48px !important;background:#FAFAF8 !important}
[data-testid="stAppViewBlockContainer"]{padding-top:0 !important}

/* ── TYPOGRAPHY (verbatim from HTML) ─────────────────────────────────────── */
.eyebrow{font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:#B8943F;font-weight:600;margin-bottom:10px}
.section-head{font-family:'Playfair Display',Georgia,serif;font-size:32px;color:#1A1A1A;line-height:1.15;font-weight:700;margin-bottom:8px}
.rule{width:40px;height:2px;background:#B8943F;margin:14px 0 26px}
.lead{font-size:15px;color:#666;max-width:640px;margin-bottom:30px;font-weight:300;line-height:1.8}
.prose{font-size:15px;line-height:1.85;color:#2D2D2D}
.prose p{margin-bottom:18px}

/* section-title / section-sub = aliases used throughout the dashboard */
p.section-title{font-family:'Playfair Display',Georgia,serif !important;font-size:32px !important;color:#1A1A1A !important;line-height:1.15 !important;font-weight:700 !important;margin:0 0 8px 0 !important}
p.section-sub{font-size:15px !important;color:#666 !important;max-width:640px !important;margin:0 0 30px 0 !important;font-weight:300 !important;line-height:1.8 !important}

/* ── HEADINGS ─────────────────────────────────────────────────────────────── */
h1{font-family:'Playfair Display',Georgia,serif !important;font-size:32px !important;font-weight:700 !important;color:#1A1A1A !important;line-height:1.15 !important}
h2{font-family:'Playfair Display',Georgia,serif !important;font-size:26px !important;font-weight:700 !important;color:#1A1A1A !important}
h3{font-family:'Playfair Display',Georgia,serif !important;font-size:20px !important;font-weight:600 !important;color:#1A1A1A !important}
h4{font-size:11px !important;letter-spacing:2.5px !important;text-transform:uppercase !important;color:#B8943F !important;font-weight:600 !important;margin-bottom:10px !important;font-family:'Source Sans 3',sans-serif !important}
h5{font-size:10px !important;letter-spacing:2px !important;text-transform:uppercase !important;color:#B8943F !important;font-weight:600 !important;font-family:'Source Sans 3',sans-serif !important}

/* ── NAV (sticky, full-width — injected as HTML at top of main()) ─────────── */
.cq-nav{background:#1B2A4A;border-bottom:2px solid #B8943F;margin:0 -48px;padding:0 48px;display:flex;align-items:stretch;justify-content:space-between;height:54px;position:sticky;top:0;z-index:200}
.cq-nav-brand{display:flex;align-items:center;color:#fff;font-family:'Playfair Display',serif;font-size:16px;font-weight:700;letter-spacing:1px;text-decoration:none;gap:4px;flex-shrink:0}
.cq-nav-brand span{color:#B8943F}
.cq-nav-right{display:flex;align-items:center;gap:20px;flex-shrink:0}
.cq-nav-status{font-size:11px;color:rgba(255,255,255,.55);font-weight:600;letter-spacing:.04em}
.cq-nav-mode{font-size:11px;color:rgba(255,255,255,.38)}
.cq-nav-sep{width:1px;height:24px;background:rgba(255,255,255,.15)}

/* ── HERO STRIP (verbatim from HTML) ─────────────────────────────────────── */
.hero{background:#1B2A4A;padding:56px 0 48px;margin:0 -48px}
.hero .container{max-width:none;padding:0 48px}
.hero-eye{font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:#B8943F;font-weight:600;margin-bottom:14px}
.hero h1{font-family:'Playfair Display',Georgia,serif;font-size:52px;color:#fff;line-height:1.1;font-weight:700}
.hero-sub{font-size:52px;color:#B8943F;font-style:italic;display:block;line-height:1.1;margin-bottom:16px}
.hero-desc{color:rgba(255,255,255,.55);font-size:16px;max-width:560px;margin-bottom:44px;font-weight:300;line-height:1.7}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:rgba(255,255,255,.10)}
.kpi{background:#1B2A4A;padding:24px;border-left:1px solid rgba(255,255,255,.08)}
.kpi:first-child{border-left:none}
.kpi-label{font-size:10px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.45);margin-bottom:8px;font-weight:600}
.kpi-val{font-family:'Playfair Display',serif;font-size:38px;color:#fff;line-height:1;font-weight:700}
.kpi-val.g{color:#6BCCA0}
.kpi-note{font-size:11px;color:rgba(255,255,255,.32);margin-top:6px}

/* ── TODAY CARDS (verbatim from HTML) ────────────────────────────────────── */
.today-hero{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;margin-bottom:36px}
.today-card{background:#fff;border:1px solid #E2E0DC;padding:20px 22px}
.today-card-label{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600;margin-bottom:6px}
.today-card-val{font-family:'Playfair Display',serif;font-size:28px;font-weight:700;line-height:1}
.today-card-note{font-size:11px;color:#AAA;margin-top:5px}
.two-col-65{display:grid;grid-template-columns:1.3fr 1fr;gap:32px;align-items:start}
.two-col-even{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}

/* ── TABLES (verbatim from HTML) ─────────────────────────────────────────── */
.tbl-wrap{margin-top:8px}
.tbl-title{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600;margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
thead th{text-align:left;padding:7px 12px;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:#999;font-weight:600;border-bottom:2px solid #1B2A4A;white-space:nowrap}
thead th.r{text-align:right}
tbody tr{border-bottom:1px solid #ECEAE6}
tbody tr:hover{background:#F7F6F3}
tbody tr:last-child{border-bottom:2px solid #1B2A4A}
tbody td{padding:9px 12px}
tbody td.r{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:#1B6F4A;font-weight:600}
.neg{color:#B83232;font-weight:600}
tr.tr-strong{background:#F7FCF9}
.td-ticker{font-weight:700;color:#1B2A4A;font-family:'Playfair Display',serif;font-size:15px}
.td-rank{color:#BBB;font-size:11px;width:30px}
.td-score{display:flex;align-items:center;gap:8px}
.score-bar-wrap{width:60px;height:5px;background:#F0EFEC;border-radius:2px;flex-shrink:0}
.score-bar{height:100%;background:#1B2A4A;border-radius:2px}
.tbl-note{font-size:11px;color:#BBB;margin-top:8px;line-height:1.6}

/* ── CHART BOX (verbatim from HTML) ──────────────────────────────────────── */
.chart-box{background:#fff;border:1px solid #E2E0DC;padding:24px 24px 16px;margin-top:28px}
.chart-title{font-size:14px;font-weight:600;color:#1A1A1A;margin-bottom:2px}
.chart-sub{font-size:12px;color:#AAA;margin-bottom:18px}
.chart-inner{position:relative;height:300px}

/* ── IC STACK (verbatim from HTML) ───────────────────────────────────────── */
.ic-stack{display:flex;flex-direction:column;gap:8px}
.ic-row{display:flex;align-items:center;gap:12px;padding:9px 12px;background:#fff;border:1px solid #E2E0DC}
.ic-name{font-size:12.5px;font-weight:600;color:#1A1A1A;width:200px;flex-shrink:0}
.ic-step{font-size:10px;color:#CCC;width:48px;flex-shrink:0}
.ic-bar-wrap{flex:1;height:6px;background:#F0EFEC;border-radius:2px;overflow:hidden}
.ic-bar{height:100%;border-radius:2px}
.ic-bar.s{background:#1B6F4A}.ic-bar.m{background:#B8943F}.ic-bar.w{background:#CCC}
.ic-val{font-family:'Playfair Display',serif;font-size:15px;font-weight:700;color:#1B2A4A;width:52px;text-align:right;flex-shrink:0}
.ic-badge{font-size:10px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;padding:2px 6px;border-radius:2px;flex-shrink:0}
.b-s{background:#EAF5EE;color:#1B6F4A}.b-m{background:#FEF5E7;color:#B8943F}.b-w{background:#F3F3F3;color:#999}

/* ── RISK LADDER (verbatim from HTML) ────────────────────────────────────── */
.risk-ladder{display:flex;flex-direction:column;gap:0}
.rl-row{display:flex;align-items:stretch;border:1px solid #E2E0DC;border-bottom:none;background:#fff}
.rl-row:last-child{border-bottom:1px solid #E2E0DC}
.rl-row:hover{background:#F9F8F6}
.rl-num{width:48px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-family:'Playfair Display',serif;font-size:20px;font-weight:700;border-right:1px solid #E2E0DC}
.l1{color:#1B6F4A;background:#EAF5EE}.l2{color:#2A6F5A;background:#E5F3EE}.l3{color:#3A6F4A;background:#EBF5EC}
.l4{color:#B8943F;background:#FEF8EC}.l5{color:#B8943F;background:#FDF5E4}.l6{color:#8B6914;background:#FDF0D0}
.l7{color:#2563EB;background:#EFF6FF}.l8{color:#9333EA;background:#F5F3FF}
.l9{color:#DC2626;background:#FEF2F2}.l10{color:#1B2A4A;background:#EFF2F8}
.rl-body{flex:1;padding:12px 16px}
.rl-name{font-size:13px;font-weight:700;color:#1A1A1A;margin-bottom:1px}
.rl-step{font-size:10px;color:#BBB;margin-bottom:3px}
.rl-desc{font-size:11.5px;color:#666;line-height:1.5}
.rl-rule{flex-shrink:0;width:170px;padding:12px 14px;border-left:1px solid #E2E0DC;display:flex;flex-direction:column;justify-content:center}
.rl-rule-label{font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:#BBB;font-weight:600;margin-bottom:2px}
.rl-rule-val{font-size:12px;color:#334155;font-weight:600;line-height:1.4}

/* ── FACTOR CARDS (verbatim from HTML) ───────────────────────────────────── */
.fac-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.fac{background:#fff;border:1px solid #E2E0DC;padding:18px 20px}
.fac-name{font-size:13px;font-weight:600;color:#1A1A1A;margin-bottom:4px}
.fac-ic{font-family:'Playfair Display',serif;font-size:26px;font-weight:700;line-height:1;margin-bottom:5px}
.fac-sub{font-size:11.5px;color:#999;line-height:1.45}
.fac-regimes{display:flex;gap:14px;margin-top:10px;padding-top:8px;border-top:1px solid #F0EFEC}
.fac-reg{font-size:11px;text-align:center}
.fac-reg-label{color:#BBB;letter-spacing:.5px;text-transform:uppercase;font-size:9px}
.fac-reg-val{font-weight:700;margin-top:2px;font-size:12px}
.bull{color:#1B6F4A}.bear{color:#B83232}

/* ── MACRO CARDS (verbatim from HTML) ────────────────────────────────────── */
.macro-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:16px}
.mac-card{background:#fff;border:1px solid #E2E0DC;padding:18px 20px}
.mac-ticker{font-size:17px;font-weight:700;color:#1B2A4A;font-family:'Playfair Display',serif}
.mac-name{font-size:11px;color:#999;margin-top:2px;margin-bottom:10px}
.mac-role{font-size:11.5px;color:#555;line-height:1.45}
.mac-status{margin-top:10px;padding-top:8px;border-top:1px solid #F0EFEC;font-size:11px;font-weight:700;letter-spacing:.8px;text-transform:uppercase}
.risk-on{color:#1B6F4A}.neutral{color:#B8943F}.risk-off{color:#B83232}

/* ── REGIME CARDS (verbatim from HTML) ───────────────────────────────────── */
.reg-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}
.reg-card{background:#fff;border:1px solid #E2E0DC;padding:26px}
.reg-name{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600;margin-bottom:5px}
.reg-pct{font-family:'Playfair Display',serif;font-size:44px;font-weight:700;line-height:1;margin:6px 0}
.reg-info{margin-top:12px;padding-top:10px;border-top:1px solid #ECEAE6;font-size:12.5px;color:#555;line-height:1.7}

/* ── METHOD CARDS (verbatim from HTML) ───────────────────────────────────── */
.method-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.method-card{background:#fff;border:1px solid #E2E0DC;padding:18px 20px;border-top:3px solid #1B2A4A}
.method-card.acc{border-top-color:#B8943F}
.method-title{font-size:13px;font-weight:700;color:#1A1A1A;margin-bottom:5px}
.method-body{font-size:11.5px;color:#666;line-height:1.6}
.method-hl{font-size:11px;font-weight:700;color:#1B2A4A;margin-top:8px;padding-top:6px;border-top:1px solid #F0EFEC}

/* ── OOS BANNER (verbatim from HTML) ─────────────────────────────────────── */
.oos-banner{background:#F0F4F9;border:1px solid #C7D2E0;border-left:4px solid #1B2A4A;padding:16px 22px;margin-bottom:28px}
.oos-banner-title{font-size:12px;font-weight:700;color:#1B2A4A;margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase}
.oos-banner-body{font-size:13px;color:#374151;line-height:1.6}
.oos-kpi-row{display:flex;gap:32px;flex-wrap:wrap;margin-top:12px}
.oos-kpi label{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:#6B7280;font-weight:600;display:block;margin-bottom:2px}
.oos-kpi-val{font-family:'Playfair Display',serif;font-size:22px;font-weight:700;color:#1B2A4A}
.oos-kpi-val.g{color:#1B6F4A}

/* ── BUDGET GRID (verbatim from HTML) ────────────────────────────────────── */
.budget-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.bud{background:#fff;border:1px solid #E2E0DC;padding:18px 20px}
.bud-label{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600;margin-bottom:6px}
.bud-val{font-family:'Playfair Display',serif;font-size:28px;font-weight:700;color:#1B2A4A;line-height:1;margin-bottom:4px}
.bud-note{font-size:11px;color:#888;line-height:1.4}
.bud-trigger{font-size:11px;color:#B83232;font-weight:600;margin-top:6px}

/* ── STRESS CARDS (verbatim from HTML) ───────────────────────────────────── */
.stress-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.stress-card{background:#fff;border:1px solid #E2E0DC;padding:18px 20px;border-left:4px solid #E2E0DC}
.stress-card.bad{border-left-color:#B83232}.stress-card.ok{border-left-color:#B8943F}.stress-card.good{border-left-color:#1B6F4A}
.stress-name{font-size:13px;font-weight:700;color:#1A1A1A;margin-bottom:3px}
.stress-period{font-size:11px;color:#999;margin-bottom:10px}
.stress-metrics{display:flex;gap:18px}
.sm-item label{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:#BBB;display:block;margin-bottom:2px}
.sm-val{font-family:'Playfair Display',serif;font-size:18px;font-weight:700}
.sm-val.neg{color:#B83232}.sm-val.pos{color:#1B6F4A}.sm-val.neu{color:#B8943F}

/* ── UTILS (verbatim from HTML) ──────────────────────────────────────────── */
.mt16{margin-top:16px}.mt24{margin-top:24px}.mt36{margin-top:36px}.mt48{margin-top:48px}

/* ── STREAMLIT NATIVE COMPONENTS — matched to HTML design ────────────────── */

/* Segmented control = HTML nav tabs */
[data-testid="stSegmentedControl"]{background:#1B2A4A !important;border-radius:0 !important;padding:0 !important;border-bottom:2px solid #B8943F !important;width:100% !important;margin:0 -48px !important;padding:0 48px !important}
[data-testid="stSegmentedControl"]>div{width:100% !important;border-radius:0 !important}
[data-testid="stSegmentedControl"] button{border-radius:0 !important;font-weight:600 !important;font-size:11px !important;letter-spacing:1.2px !important;text-transform:uppercase !important;padding:0 16px !important;color:rgba(255,255,255,.55) !important;background:transparent !important;border:none !important;border-left:1px solid rgba(255,255,255,.07) !important;border-bottom:3px solid transparent !important;height:54px !important;transition:color .15s,background .15s !important}
[data-testid="stSegmentedControl"] button:hover{color:#fff !important;background:rgba(255,255,255,.05) !important}
[data-testid="stSegmentedControl"] button[aria-checked="true"]{background:transparent !important;color:#fff !important;font-weight:700 !important;border-bottom:3px solid #B8943F !important}

/* Metric = .kpi card */
[data-testid="stMetric"]{background:#1B2A4A !important;padding:24px !important;border-left:1px solid rgba(255,255,255,.08) !important;border-radius:0 !important;border:none !important}
[data-testid="stMetricLabel"]{font-size:10px !important;letter-spacing:1.8px !important;text-transform:uppercase !important;color:rgba(255,255,255,.45) !important;font-weight:600 !important}
[data-testid="stMetricValue"]{font-family:'Playfair Display',Georgia,serif !important;font-size:38px !important;color:#fff !important;line-height:1 !important;font-weight:700 !important}
[data-testid="stMetricDelta"]{font-size:12px !important;font-weight:600 !important}

/* Expander = .method-card */
[data-testid="stExpander"]{border:1px solid #E2E0DC !important;border-top:3px solid #1B2A4A !important;border-radius:0 !important;background:#fff !important;margin-bottom:8px !important;overflow:hidden !important}
[data-testid="stExpander"] summary{font-weight:700 !important;font-size:13px !important;color:#1A1A1A !important;padding:12px 16px !important;background:#fff !important;border-bottom:1px solid #E2E0DC !important}
[data-testid="stExpander"] summary:hover{background:#F7F6F3 !important}
[data-testid="stExpander"]>div>div{padding:12px 16px !important}

/* Buttons */
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:#1B2A4A !important;border:1px solid #1B2A4A !important;color:#fff !important;border-radius:2px !important;font-weight:600 !important;font-size:13px !important}
button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{background:#243860 !important}
button[kind="secondary"],[data-testid="stBaseButton-secondary"]{border-radius:2px !important;font-weight:600 !important;font-size:13px !important;border-color:#E2E0DC !important;color:#1B2A4A !important;background:#fff !important}
button[kind="secondary"]:hover,[data-testid="stBaseButton-secondary"]:hover{border-color:#1B2A4A !important}

/* Radio (sub-nav within pages) */
[data-testid="stRadio"] label,[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p{font-size:11px !important;font-weight:600 !important;color:#666 !important;letter-spacing:1px !important;text-transform:uppercase !important}

/* Toggle / Checkbox */
[data-testid="stToggle"] label,[data-testid="stCheckbox"] label{font-weight:600 !important;font-size:13px !important;color:#1A1A1A !important}

/* Alert */
[data-testid="stAlert"]{border-radius:0 !important;font-size:13.5px !important}

/* Caption */
[data-testid="stCaptionContainer"] p,.stCaption{font-size:11px !important;color:#BBB !important;line-height:1.6 !important}

/* Dataframe */
[data-testid="stDataFrame"]{border-radius:0 !important;border:1px solid #E2E0DC !important;overflow:hidden !important}

/* Container with border */
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid #E2E0DC !important;border-top:3px solid #1B2A4A !important;border-radius:0 !important;background:#fff !important;padding:18px 20px !important}

/* Divider */
hr{border:none !important;border-top:1px solid #ECEAE6 !important;margin:20px 0 !important}

/* Code */
code,pre{font-size:12px !important;background:#F0EFEC !important;border-radius:2px !important;color:#1B2A4A !important}
</style>
"""


def tab_research_proof_gate():
    st.markdown('<p class="section-title">Why Is This Blocked?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Each ticker listed here has at least one gate that must clear before ideas can move forward. Read the first blocking gate. Fix it, then re-check.</p>',
        unsafe_allow_html=True,
    )

    drilldown = safe_csv(ROOT / "action_readiness_ticker_drilldown.csv")
    explain   = safe_csv(ROOT / "daily_workflow_ticker_explain.csv")
    risk_gate = safe_csv(ROOT / "final_risk_gate.csv")

    if drilldown.empty:
        st.info("No ticker drilldown data found. Run the daily system, then return here.")
        return

    blocked = drilldown[drilldown.get("first_gate_status", drilldown.columns[0] if not drilldown.empty else "").isin(["BLOCKED", "DATA_GAP", "MISSING"])] if "first_gate_status" in drilldown.columns else drilldown
    if blocked.empty:
        blocked = drilldown

    st.markdown(f"**{len(blocked)} ticker(s) with active gate blocks**")

    for _, row in blocked.head(20).iterrows():
        ticker = str(row.get("ticker", "—"))
        stage  = _human_text(str(row.get("current_stage", "—")), 60)
        score  = _to_float(row.get("readiness_score"), None)
        score_text = f"{score:.1f}/100" if score is not None else "—"
        why    = _human_text(str(row.get("why_blocked_plain_english", row.get("plain_english_summary", "No explanation available."))), 280)
        gate   = _human_text(str(row.get("first_blocking_gate", "—")), 80)
        fix    = _human_text(str(row.get("first_clear_condition", "Check the source files listed in the drilldown report.")), 200)
        route  = _human_text(str(row.get("route_after_all_gates_clear", "—")), 120)
        donot  = _human_text(str(row.get("do_not_do", "")), 200)
        accent = "#991b1b" if score is not None and score < 30 else ("#b45309" if score is not None and score < 60 else "#1e3a5f")

        with st.container(border=True):
            c1, c2 = st.columns([1.1, 5])
            with c1:
                _render_html(
                    f"""
                    <div style="text-align:center; padding:12px 0 6px 0;">
                      <div style="font-size:22px; font-weight:900; color:{accent}; font-family:'Georgia',serif; letter-spacing:-0.5px;">{_esc(ticker)}</div>
                      <div style="font-size:10px; color:#6b7280; font-weight:700; text-transform:uppercase; margin-top:3px; letter-spacing:.06em;">Readiness</div>
                      <div style="font-size:18px; font-weight:900; color:{accent}; margin-top:2px;">{_esc(score_text)}</div>
                      <div style="font-size:10px; color:#9ca3af; margin-top:4px;">{_esc(stage)}</div>
                    </div>
                    """
                )
            with c2:
                st.markdown(f"**Why blocked:** {why}")
                st.markdown(f"**First gate to fix:** {gate}")
                st.markdown(f"**How to clear it:** {fix}")
                if route and route != "—":
                    st.caption(f"Route once all gates clear: {route}")
                if donot:
                    st.caption(f"Do not do: {donot}")

    with st.expander("All ticker blockers — table view", expanded=False):
        cols_to_show = [c for c in ["ticker", "readiness_score", "current_stage", "first_blocking_gate", "first_gate_status", "first_clear_condition"] if c in drilldown.columns]
        if cols_to_show:
            _show_status_table(drilldown[cols_to_show], height=400)
        else:
            _show_status_table(drilldown, height=400)

    if not explain.empty:
        with st.expander("Plain-English ticker summaries", expanded=False):
            for _, r in explain.head(15).iterrows():
                tk = str(r.get("ticker", "—"))
                summary = _human_text(str(r.get("plain_english_summary", "")), 300)
                if summary:
                    st.markdown(f"**{tk}** — {summary}")


def tab_journal():
    st.markdown('<p class="section-title">Trade Notes</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Paper and manual trade log. All entries are research-only. No broker connection. No live orders.</p>',
        unsafe_allow_html=True,
    )

    journal  = safe_csv(ROOT / "trade_journal.csv")
    ptlog    = safe_csv(ROOT / "paper_trading_log.csv")

    if not journal.empty:
        closed = journal[journal.get("status", journal.columns[0]).astype(str).str.upper() == "CLOSED"] if "status" in journal.columns else journal
        open_j = journal[journal.get("status", journal.columns[0]).astype(str).str.upper() != "CLOSED"] if "status" in journal.columns else pd.DataFrame()

        total_closed = len(closed)
        won = int((closed["pnl"] > 0).sum()) if "pnl" in closed.columns and not closed.empty else 0
        total_pnl = round(float(closed["pnl"].sum()), 2) if "pnl" in closed.columns and not closed.empty else 0
        win_rate = f"{won / total_closed * 100:.0f}%" if total_closed else "—"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _simple_card("Closed trades", str(total_closed), "All exits recorded in journal", "#0f172a")
        with c2:
            _simple_card("Win rate", win_rate, f"{won} winners", "#166534" if won >= total_closed * 0.5 else "#991b1b")
        with c3:
            pnl_accent = "#166534" if total_pnl >= 0 else "#991b1b"
            _simple_card("Total P&L", f"${total_pnl:,.2f}", "Paper only — no real money", pnl_accent)
        with c4:
            _simple_card("Open ideas", str(len(open_j)), "Not yet closed", "#0f172a")

        if not closed.empty:
            st.markdown("#### Closed trades")
            show_cols = [c for c in ["ticker", "entry_date", "exit_date", "direction", "entry_price", "exit_price", "pnl", "pnl_pct", "holding_days", "notes", "entry_reason", "exit_reason", "signal_grade"] if c in closed.columns]
            _show_status_table(closed[show_cols] if show_cols else closed, height=380)

        if not open_j.empty:
            st.markdown("#### Open ideas (not exited)")
            show_cols2 = [c for c in ["ticker", "entry_date", "direction", "entry_price", "ml_score", "notes", "entry_reason", "strategy_type", "regime_at_entry"] if c in open_j.columns]
            _show_status_table(open_j[show_cols2] if show_cols2 else open_j, height=280)
    else:
        st.info("No trade journal found. Entries are created when you exit a paper position.")

    if not ptlog.empty:
        with st.expander("Paper trading log — raw entries", expanded=False):
            _show_status_table(ptlog, height=320)
            st.caption("paper_trading_log.csv — one row per session, generated by step500.")

    st.markdown("---")
    _render_html(
        """
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #0f172a; border-radius:8px; padding:14px 18px; margin-top:8px;">
          <div style="font-size:11px; color:#6b7280; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Reminder</div>
          <div style="font-size:14px; color:#374151; margin-top:5px; line-height:1.6;">This is a research-only paper account. No broker connection. Sizing shown here does not represent real capital. All decisions require Risk and Before-Action check first.</div>
        </div>
        """
    )


def tab_tools():
    st.markdown('<p class="section-title">Tools</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Commands to refresh data, restart the site, run specific engines, or check files. Copy any command into Terminal and run from the project folder.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        _system_command_card(
            "Run daily pipeline (full)",
            "Refreshes all signals, risk, news, and outputs. Run this once per day before starting research.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 -u canyon_final_v9_step70_daily_runner_all.py",
            "#111827",
        )
        _system_command_card(
            "Run fast pipeline (skip slow steps)",
            "Skips step56 and step68. Use when you just want a quick signal refresh.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 canyon_final_v9_step70_daily_runner_all.py --fast",
            "#334155",
        )
        _system_command_card(
            "Run daily paper pipeline",
            "Records today's signals and paper positions. Run after market close.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 canyon_final_v9_step500_daily_pipeline.py",
            "#0f172a",
        )
    with c2:
        _system_command_card(
            "Restart the Streamlit dashboard",
            "Use this when the page is stale, frozen, or the server was stopped.",
            "pkill -f 'streamlit run' 2>/dev/null; sleep 2\ncd ~/Desktop/canyon_quant\nsource .venv/bin/activate\nstreamlit run canyon_final_v9_step86_dashboard_v3.py --server.port 8512",
            "#1e40af",
        )
        _system_command_card(
            "Compile-check dashboard",
            "Confirms the dashboard has no syntax errors before restarting.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 -m py_compile canyon_final_v9_step86_dashboard_v3.py && echo 'OK'",
            "#166534",
        )
        _system_command_card(
            "Generate research HTML page",
            "Rebuilds canyon_v9_research.html from local CSV/JSON outputs.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 update_research_html.py",
            "#334155",
        )

    with st.expander("More run commands", expanded=False):
        st.markdown("""
**Dry-run (test only, no writes):**
```
python3 canyon_final_v9_step70_daily_runner_all.py --dry-run
```
**Paper sim — mark to market:**
```
python3 canyon_final_v9_step69_paper_sim.py --mark-to-market
```
**Paper sim — status:**
```
python3 canyon_final_v9_step69_paper_sim.py --status
```
**Paper sim — rebalance:**
```
python3 canyon_final_v9_step69_paper_sim.py --rebalance
```
**Check runner output log:**
```
tail -60 autorun_stdout.log
```
        """)

    run_log = safe_csv(ROOT / "run_daily_all_log.csv")
    if not run_log.empty:
        st.markdown("#### Last run results")
        latest, latest_rows = _latest_run_rows(run_log)
        if not latest_rows.empty:
            passed = int((latest_rows.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "OK").sum())
            total  = len(latest_rows)
            failed = total - passed
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                _simple_card("Last run date", latest[:16] if latest else "—", "", "#0f172a")
            with cc2:
                _simple_card("Passed", f"{passed}/{total}", "steps completed OK", "#166534" if not failed else "#991b1b")
            with cc3:
                _simple_card("Failed", str(failed), "steps need attention" if failed else "all steps OK", "#991b1b" if failed else "#166534")
            with st.expander("Step-by-step results", expanded=False):
                show = [c for c in ["step", "status", "duration_s", "note"] if c in latest_rows.columns]
                _show_status_table(latest_rows[show] if show else latest_rows, height=380)


def _render_daily_run_panel():
    st.markdown("#### Daily Data Refresh")
    _render_browser_daily_update_button()
    st.markdown("##### Manual commands (Terminal)")
    c1, c2 = st.columns(2)
    with c1:
        _system_command_card(
            "Full daily run",
            "Refreshes all signals, news, risk, and proof files.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 -u canyon_final_v9_step70_daily_runner_all.py",
            "#111827",
        )
    with c2:
        _system_command_card(
            "Paper pipeline",
            "Records today's signals and paper positions.",
            "cd ~/Desktop/canyon_quant\nsource .venv/bin/activate\npython3 canyon_final_v9_step500_daily_pipeline.py",
            "#0f172a",
        )
    run_log = safe_csv(ROOT / "run_daily_all_log.csv")
    if not run_log.empty:
        latest, latest_rows = _latest_run_rows(run_log)
        if not latest_rows.empty:
            passed = int((latest_rows.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "OK").sum())
            total  = len(latest_rows)
            st.caption(f"Last run: {latest[:16]} — {passed}/{total} steps passed.")
    with st.expander("Run log tail", expanded=False):
        st.code(_tail_text(ROOT / "autorun_stdout.log"), language="text")


def _render_sector_theme_depth():
    st.markdown("#### Sector & Theme Depth")
    sector_df = safe_csv(ROOT / "sector_cycle_state.csv")
    if sector_df.empty:
        st.caption("No sector data found. Run daily pipeline.")
        return
    show_cols = [c for c in ["etf", "sector", "cycle_state", "rotation_label", "top_theme", "top_headline", "cycle_note", "top_alpha_names", "cap_status"] if c in sector_df.columns]
    for _, row in sector_df.head(12).iterrows():
        label    = str(row.get("sector", row.get("etf", "—")))
        cycle    = _human_text(str(row.get("cycle_state", "—")), 60)
        rotation = str(row.get("rotation_label", "—"))
        theme    = _human_text(str(row.get("top_theme", "")), 80)
        note     = _human_text(str(row.get("cycle_note", "")), 200)
        accent   = "#166534" if "leader" in cycle.lower() else ("#991b1b" if "down" in cycle.lower() or "bear" in cycle.lower() else "#374151")
        _render_html(
            f"""
            <div style="display:flex; align-items:baseline; gap:12px; padding:10px 14px; border:1px solid #e2e8f0; border-left:4px solid {accent}; background:#fff; border-radius:7px; margin-bottom:8px;">
              <span style="font-size:14px; font-weight:900; color:#0f172a; min-width:130px;">{_esc(label)}</span>
              <span style="font-size:12px; color:{accent}; font-weight:700;">{_esc(cycle)}</span>
              <span style="font-size:11px; color:#6b7280;">· {_esc(rotation)}</span>
              <span style="font-size:11px; color:#9ca3af;">{_esc(theme)}</span>
            </div>
            """
        )
        if note and note != "—":
            st.caption(f"  {note}")
    with st.expander("Full sector table", expanded=False):
        _show_status_table(sector_df[show_cols] if show_cols else sector_df, height=380)


def _show_deep_decision_desk():
    st.markdown("#### Deep Decision Desk")
    drilldown = safe_csv(ROOT / "action_readiness_ticker_drilldown.csv")
    decision_state = safe_json(ROOT / "action_readiness_card_deck_state.json")
    blocker_ex = safe_csv(ROOT / "action_readiness_blocker_explainer.csv")

    if drilldown.empty and not decision_state:
        st.caption("No decision desk data. Run daily pipeline.")
        return

    if decision_state:
        summary = _human_text(str(decision_state.get("plain_summary", decision_state.get("summary", ""))), 300)
        if summary:
            _render_html(
                f"""
                <div style="background:#fff; border:1px solid #d1d5db; border-left:5px solid #0f172a; border-radius:8px; padding:14px 18px; margin-bottom:14px;">
                  <div style="font-size:12px; color:#6b7280; font-weight:700; text-transform:uppercase; letter-spacing:.06em;">Deck summary</div>
                  <div style="font-size:15px; color:#111827; margin-top:6px; line-height:1.55;">{_esc(summary)}</div>
                </div>
                """
            )

    if not drilldown.empty:
        show_cols = [c for c in ["ticker", "readiness_score", "current_stage", "first_blocking_gate", "route_after_all_gates_clear", "decision_room_summary"] if c in drilldown.columns]
        with st.expander(f"Action readiness — {len(drilldown)} tickers", expanded=True):
            _show_status_table(drilldown[show_cols] if show_cols else drilldown, height=380)

    if not blocker_ex.empty:
        with st.expander("Blocker explainer", expanded=False):
            show_b = [c for c in ["ticker", "gate_name", "gate_status", "plain_english_blocker", "fix_action"] if c in blocker_ex.columns]
            _show_status_table(blocker_ex[show_b] if show_b else blocker_ex, height=380)


def _render_active_spine():
    st.markdown("#### Active Daily Spine")
    steps = safe_csv(ROOT / "daily_workflow_steps.csv")
    if steps.empty:
        st.caption("No workflow steps found. Run daily pipeline.")
        return
    show_cols = [c for c in ["step_order", "status", "station", "what_to_do", "why_this_exists", "next_dashboard_section"] if c in steps.columns]
    for _, row in steps.iterrows():
        order    = str(row.get("step_order", "—"))
        status   = str(row.get("status", "—")).upper()
        station  = str(row.get("station", "—"))
        what     = _human_text(str(row.get("what_to_do", "")), 180)
        accent   = "#166534" if status == "DONE" else ("#b45309" if status == "REVIEW" else "#374151")
        _render_html(
            f"""
            <div style="display:flex; align-items:flex-start; gap:10px; padding:9px 13px; border:1px solid #e2e8f0; border-radius:7px; background:#fff; margin-bottom:6px;">
              <span style="font-size:12px; font-weight:900; color:#0f172a; min-width:22px;">{_esc(order)}</span>
              <span style="font-size:10px; font-weight:700; padding:2px 7px; border-radius:12px; background:{'#dcfce7' if status == 'DONE' else '#fef3c7'}; color:{accent}; margin-top:1px; flex-shrink:0;">{_esc(status)}</span>
              <span style="font-size:13px; font-weight:700; color:#1e3a5f; min-width:120px; flex-shrink:0;">{_esc(station)}</span>
              <span style="font-size:12px; color:#374151; line-height:1.5;">{_esc(what)}</span>
            </div>
            """
        )


def _show_dynamic_daily_workflow():
    st.markdown("#### Daily Workflow Queue")
    queue = safe_csv(ROOT / "daily_workflow_queue.csv")
    explain = safe_csv(ROOT / "daily_workflow_ticker_explain.csv")
    if queue.empty:
        st.caption("No workflow queue. Run daily pipeline.")
        return
    show_cols = [c for c in ["priority_rank", "priority", "ticker", "sector", "workflow_bucket", "what_to_do", "sector_adjusted_action", "best_horizon", "option_route", "risk_action"] if c in queue.columns]
    st.markdown(f"**{len(queue)} items in today's research queue**")
    for _, row in queue.head(15).iterrows():
        ticker  = str(row.get("ticker", "—"))
        bucket  = _human_text(str(row.get("workflow_bucket", "—")), 60)
        what    = _human_text(str(row.get("what_to_do", "")), 200)
        action  = _human_text(str(row.get("sector_adjusted_action", row.get("risk_action", ""))), 80)
        priority = str(row.get("priority", "Normal"))
        accent   = "#991b1b" if "risk" in bucket.lower() else ("#166534" if "ready" in bucket.lower() else "#0f172a")
        _render_html(
            f"""
            <div style="display:flex; align-items:baseline; gap:10px; padding:9px 13px; border:1px solid #e2e8f0; border-left:4px solid {accent}; border-radius:7px; background:#fff; margin-bottom:6px;">
              <span style="font-size:14px; font-weight:900; color:#0f172a; min-width:52px;">{_esc(ticker)}</span>
              <span style="font-size:11px; font-weight:700; color:{accent};">{_esc(bucket)}</span>
              <span style="font-size:12px; color:#374151; flex:1; line-height:1.5;">{_esc(what)}</span>
              <span style="font-size:11px; color:#6b7280;">{_esc(action)}</span>
            </div>
            """
        )
    if not explain.empty:
        with st.expander("Plain-English summaries for each ticker", expanded=False):
            for _, r in explain.head(12).iterrows():
                tk = str(r.get("ticker", "—"))
                summary = _human_text(str(r.get("plain_english_summary", "")), 280)
                if summary:
                    st.markdown(f"**{tk}** — {summary}")


def main():
    # ── Inject global design system CSS ───────────────────────────────────────
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # ── Header bar ─────────────────────────────────────────────────────────
    brief_q = safe_json(ROOT / "pm_morning_brief_state.json")
    risk_q  = safe_json(ROOT / "risk_desk_overview.json")
    _risk_color_q = "#dc2626" if str(brief_q.get("risk_color", "")).lower() == "red" else "#16a34a"
    _risk_label_q = str(brief_q.get("risk_answer") or risk_q.get("master_risk_action") or "—")[:60]
    _esc_q = _esc(_risk_label_q)
    _render_html(
        f"""
        <div class="cq-nav">
          <a class="cq-nav-brand">CANYON <span>QUANT</span></a>
          <div class="cq-nav-right">
            <div class="cq-nav-status">{_esc_q}</div>
            <div class="cq-nav-sep"></div>
            <div class="cq-nav-mode">Research only &middot; No broker</div>
          </div>
        </div>
        """
    )

    _rc1, _rc2 = st.columns([8, 1])
    with _rc2:
        if st.button("⟳", help="Reload all data from disk", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    page_options = [
        "Home",
        "Today",
        "Ideas",
        "News",
        "Risk",
        "Performance",
        "Live / Paper",
        "System",
    ]
    query_page = "Home"
    try:
        raw_query_page = st.query_params.get("page", "Home")
        if isinstance(raw_query_page, list):
            raw_query_page = raw_query_page[0] if raw_query_page else "Home"
        query_page = str(raw_query_page)
    except Exception:
        query_page = "Home"
    default_page = query_page if query_page in page_options else "Home"

    selector_key = "main_page_selector_" + re.sub(r"[^A-Za-z0-9]+", "_", default_page)
    if hasattr(st, "segmented_control"):
        page = st.segmented_control("Page", page_options, default=default_page, label_visibility="collapsed", key=selector_key)
    else:
        page = st.radio("Page", page_options, index=page_options.index(default_page), horizontal=True, label_visibility="collapsed", key=selector_key)

    st.markdown(
        "<div style='height:32px;'></div>",
        unsafe_allow_html=True,
    )

    if page == "Home":
        tab_run_system()

    elif page == "Today":
        tab_today_workflow()

    elif page == "Ideas":
        tab_ideas_workflow()

    elif page == "News":
        tab_news_room()

    elif page == "Risk":
        st.markdown('<p class="section-title">Safety Check</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Start here before trusting any idea. If this page says wait, the idea waits.</p>', unsafe_allow_html=True)
        risk_view = st.radio("Safety page", ["Can I add?", "Why no?", "What to fix"], horizontal=True, label_visibility="collapsed", key="risk_page_selector")
        _render_subtab_depth("Risk", risk_view)
        if risk_view == "Can I add?":
            _run_with_plain_streamlit_text(tab_risk_portfolio)
        elif risk_view == "Why no?":
            _run_with_plain_streamlit_text(tab_research_proof_gate)
        else:
            _run_with_plain_streamlit_text(lambda: _render_risk_book_seed_center(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_risk_seed_approval_workbench(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_risk_seed_pm_review_intake(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_review_evidence_autofill(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_acceptance_gate(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_review_triage(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_source_proof_desk(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_proof_acceptance_bridge(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_review_final_gate_bridge(compact=False))

    elif page == "Performance":
        perf_view = st.radio(
            "Performance view",
            ["Summary", "Walk-Forward OOS"],
            horizontal=True,
            label_visibility="collapsed",
            key="performance_view_selector",
        )
        if perf_view == "Walk-Forward OOS":
            _render_walk_forward_oos_panel()
        else:
            _run_with_plain_streamlit_text(tab_performance)

    elif page == "Live / Paper":
        st.markdown('<p class="section-title">Paper / Manual Account</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Paper account, manual account value, alerts, and notes. No broker connection.</p>', unsafe_allow_html=True)
        live_view = st.radio("Live page", ["Account monitor", "Trade notes"], horizontal=True, label_visibility="collapsed", key="live_page_selector")
        _render_subtab_depth("Live / Paper", live_view)
        if live_view == "Account monitor":
            _run_with_plain_streamlit_text(tab_live_paper_monitor)
        else:
            _run_with_plain_streamlit_text(tab_journal)

    elif page == "System":
        st.markdown('<p class="section-title">System</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Use this when something looks missing, stale, or broken.</p>', unsafe_allow_html=True)
        system_view = st.radio("System page", ["Update / Fix", "Where Things Are", "Tools", "Proof To Fill", "Data Fix", "All Files"], horizontal=True, label_visibility="collapsed", key="system_page_selector")
        _render_subtab_depth("System", system_view)
        _render_system_health_bar()
        if system_view == "Update / Fix":
            _render_system_run_center()
        elif system_view == "Where Things Are":
            _render_where_everything_lives()
        elif system_view == "Tools":
            _run_with_plain_streamlit_text(tab_tools)
        elif system_view == "Proof To Fill":
            _run_with_plain_streamlit_text(lambda: _render_quant_fund_flow_navigator(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_ticker_flow_cards(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_proof_collection_workbench(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_proof_quality_gate(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_proof_fill_desk(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_proof_intake_safe_apply(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_proof_closure_tracker(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_quant_fund_operating_flow(compact=False))
        elif system_view == "Data Fix":
            _run_with_plain_streamlit_text(lambda: _render_data_reliability_center(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_data_repair_center(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_risk_book_seed_center(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_risk_seed_approval_workbench(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_risk_seed_pm_review_intake(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_review_evidence_autofill(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_acceptance_gate(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_review_triage(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_source_proof_desk(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_evidence_proof_acceptance_bridge(compact=False))
            st.markdown("---")
            _run_with_plain_streamlit_text(lambda: _render_pm_review_final_gate_bridge(compact=False))
        else:
            _run_with_plain_streamlit_text(tab_system_status)


if __name__ == "__main__":
    main()
