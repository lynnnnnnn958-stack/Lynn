#!/usr/bin/env python3
"""
Canyon v9 Step 208 - Quant Fund Operating Flow.

Research-only. No broker connection. No live orders.

This step turns the growing Canyon v9 system into a clear operating map:
what enters the process, what can block it, what evidence is required, and
which dashboard page should be used next. It is a software-flow blueprint, not
a trading engine and not an execution path.

Outputs:
  quant_fund_operating_flow_state.json
  quant_fund_operating_flow_stages.csv
  quant_fund_operating_flow_edges.csv
  quant_fund_operating_flow_daily_runbook.csv
  quant_fund_operating_flow_software_map.csv
  quant_fund_operating_flow_report.md
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "quant_fund_operating_flow_state.json"
OUT_STAGES = ROOT / "quant_fund_operating_flow_stages.csv"
OUT_EDGES = ROOT / "quant_fund_operating_flow_edges.csv"
OUT_RUNBOOK = ROOT / "quant_fund_operating_flow_daily_runbook.csv"
OUT_SOFTWARE_MAP = ROOT / "quant_fund_operating_flow_software_map.csv"
OUT_REPORT = ROOT / "quant_fund_operating_flow_report.md"


STAGE_COLUMNS = [
    "stage_order",
    "stage_name",
    "desk_name",
    "plain_goal",
    "primary_inputs",
    "primary_outputs",
    "hard_gate",
    "if_gate_fails",
    "app_page",
    "active_files",
    "owner_role",
    "maturity_now_pct",
    "gap_to_top_fund",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

EDGE_COLUMNS = [
    "from_stage",
    "to_stage",
    "handoff_payload",
    "gate_condition",
    "blocking_failure",
    "why_it_matters",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

RUNBOOK_COLUMNS = [
    "run_order",
    "when_to_use",
    "human_action",
    "system_action",
    "output_to_read",
    "do_not_do",
    "page_to_open",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

SOFTWARE_COLUMNS = [
    "app_tab",
    "panel",
    "what_user_should_look_for",
    "input_files",
    "output_files",
    "related_steps",
    "maturity_now_pct",
    "next_upgrade",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]


def guard_flags(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["research_only"] = True
    out["no_broker_connection"] = True
    out["no_live_orders"] = True
    return out


def build_stages() -> pd.DataFrame:
    rows = [
        {
            "stage_order": 1,
            "stage_name": "Data Intake",
            "desk_name": "Data Desk",
            "plain_goal": "Bring in prices, news, options, fundamentals, and local paper-book files.",
            "primary_inputs": "Yahoo/local CSV files, paper book, news files, options snapshots.",
            "primary_outputs": "Fresh raw files that later checks can read.",
            "hard_gate": "The file must exist, have rows, and have a usable date.",
            "if_gate_fails": "Stop the idea and repair the data first.",
            "app_page": "System -> Data Reliability",
            "active_files": "price_data_status.csv; data_truth_ledger.csv; run_log.csv",
            "owner_role": "Data engineer",
            "maturity_now_pct": 45,
            "gap_to_top_fund": "Needs licensed point-in-time market data and vendor timestamps.",
        },
        {
            "stage_order": 2,
            "stage_name": "Point-in-Time Truth",
            "desk_name": "Truth Ledger",
            "plain_goal": "Record what the model really knew at the time, so backtests do not cheat.",
            "primary_inputs": "Event-time ledger, historical snapshots, run logs.",
            "primary_outputs": "Truth-readiness flags and bias warnings.",
            "hard_gate": "No future information can enter a past decision.",
            "if_gate_fails": "Mark the backtest or signal as not trustworthy.",
            "app_page": "System -> Trust check",
            "active_files": "pit_truth_readiness.csv; event_time_truth_ledger.csv; backtest_bias_guard.csv",
            "owner_role": "Research QA",
            "maturity_now_pct": 35,
            "gap_to_top_fund": "Needs full point-in-time fundamentals, constituents, delistings, and news timestamps.",
        },
        {
            "stage_order": 3,
            "stage_name": "Universe And Exposure Map",
            "desk_name": "Universe Desk",
            "plain_goal": "Know what tickers are allowed and what each ticker really represents.",
            "primary_inputs": "Ticker universe, sector map, ETF/theme map.",
            "primary_outputs": "Clean ticker list, sector/theme labels, overlap warnings.",
            "hard_gate": "No unknown ticker, duplicated exposure, or stale sector label should pass silently.",
            "if_gate_fails": "Treat the ticker as research-only until mapped.",
            "app_page": "Ideas -> Research list",
            "active_files": "universe_master.csv; sector_map.csv; theme_candidate_enrichment.csv",
            "owner_role": "Research engineer",
            "maturity_now_pct": 55,
            "gap_to_top_fund": "Needs historical index membership and deeper company-to-theme mapping.",
        },
        {
            "stage_order": 4,
            "stage_name": "Market Weather",
            "desk_name": "Macro Desk",
            "plain_goal": "Decide whether the broad market is friendly, mixed, or hostile.",
            "primary_inputs": "SPY, QQQ, IWM, VIX, rates, credit, dollar, commodity proxies.",
            "primary_outputs": "Market tone, macro pressure, rate and credit warnings.",
            "hard_gate": "A hostile market can reduce or block new risk.",
            "if_gate_fails": "Shrink size and demand stronger proof.",
            "app_page": "Today -> Daily answer",
            "active_files": "macro_regime_signals.csv; macro_risk_sensitivity.csv; stress_test_results.csv",
            "owner_role": "Macro PM",
            "maturity_now_pct": 55,
            "gap_to_top_fund": "Needs calibrated macro factor betas and scenario loss history.",
        },
        {
            "stage_order": 5,
            "stage_name": "Sector And Theme Cycle",
            "desk_name": "Sector Desk",
            "plain_goal": "Separate early-cycle, mid-cycle, late-cycle, and fading sectors or themes.",
            "primary_inputs": "Sector ETFs, subsector baskets, theme linkage files.",
            "primary_outputs": "Sector cycle read, preferred sectors, weak sectors, linked tickers.",
            "hard_gate": "Do not buy a single stock story if its whole sector is rolling over.",
            "if_gate_fails": "Move the idea to watch-only or hedge-first review.",
            "app_page": "Ideas -> Sector and themes",
            "active_files": "sector_cycle_linkage.csv; institutional_subsector_cycle.csv; sector_timeframe_router.csv",
            "owner_role": "Sector strategist",
            "maturity_now_pct": 58,
            "gap_to_top_fund": "Needs richer subsector baskets, supply-chain maps, and relative earnings revisions.",
        },
        {
            "stage_order": 6,
            "stage_name": "Business And Event Risk",
            "desk_name": "Fundamental Desk",
            "plain_goal": "Check whether the business, valuation, earnings date, and known events support the idea.",
            "primary_inputs": "Fundamental files, earnings calendar, event research dossier.",
            "primary_outputs": "Business-quality read, valuation risk, earnings jump-risk warning.",
            "hard_gate": "Earnings or event gap risk must be known before sizing.",
            "if_gate_fails": "Reduce size, wait until after event, or require options-only insurance research.",
            "app_page": "News -> Events",
            "active_files": "fundamental_quality_valuation.csv; earnings_gap_down_risk.csv; event_research_dossier.csv",
            "owner_role": "Fundamental analyst",
            "maturity_now_pct": 45,
            "gap_to_top_fund": "Needs estimate revisions, call transcripts, guidance changes, and insider-quality scoring.",
        },
        {
            "stage_order": 7,
            "stage_name": "News To Industry Chain",
            "desk_name": "News Desk",
            "plain_goal": "Turn a headline into a clear map of who may benefit, who may be hurt, and why.",
            "primary_inputs": "Headlines, source links, target maps, causal-chain files.",
            "primary_outputs": "Beneficiary list, vulnerable list, proof request, price-reaction check.",
            "hard_gate": "A headline cannot create size until the causal link and timing are verified.",
            "if_gate_fails": "Keep it as a news read, not a trade idea.",
            "app_page": "News -> News to read first",
            "active_files": "news_impact_targeting.csv; event_causal_chain.csv; pm_evidence_source_proof_input.csv",
            "owner_role": "Event researcher",
            "maturity_now_pct": 48,
            "gap_to_top_fund": "Needs entity extraction, supply-chain graph, and verified event-time price reaction.",
        },
        {
            "stage_order": 8,
            "stage_name": "Price And Volume Confirmation",
            "desk_name": "Market Microstructure Desk",
            "plain_goal": "Check whether price, volume, volatility, and spreads confirm or reject the story.",
            "primary_inputs": "Price breaks, volume spikes, volatility shifts, spread warnings.",
            "primary_outputs": "Breakout, breakdown, wait, or liquidity warning.",
            "hard_gate": "If price/volume does not confirm, the idea stays on watch.",
            "if_gate_fails": "Wait for confirmation or remove it from the queue.",
            "app_page": "Today -> Alerts",
            "active_files": "daily_alerts.csv; technical_signal_matrix.csv; execution_cost_stress.csv",
            "owner_role": "Trading research",
            "maturity_now_pct": 55,
            "gap_to_top_fund": "Needs intraday quotes, order-book depth, and calibrated spread stress.",
        },
        {
            "stage_order": 9,
            "stage_name": "Options Route",
            "desk_name": "Options Desk",
            "plain_goal": "Decide whether options are useful, dangerous, or blocked.",
            "primary_inputs": "Options chain, gamma proxy, call/put route, event gap risk.",
            "primary_outputs": "Call research, put research, no-options, or wait.",
            "hard_gate": "Options cannot override risk, event, liquidity, or proof gates.",
            "if_gate_fails": "Do not look for calls or puts yet.",
            "app_page": "Ideas -> Options route",
            "active_files": "options_decision_matrix.csv; options_gamma_report.md; timeframe_options_playbook.csv",
            "owner_role": "Options strategist",
            "maturity_now_pct": 42,
            "gap_to_top_fund": "Needs full Greeks, IV surface, term structure, skew, and dealer-position validation.",
        },
        {
            "stage_order": 10,
            "stage_name": "Signal Skill Lab",
            "desk_name": "Research Lab",
            "plain_goal": "Measure whether signals actually worked, how fast they decay, and when they fail.",
            "primary_inputs": "Signal history, forward returns, failure labels, live IC observations.",
            "primary_outputs": "IC, decay, failure patterns, and signal reliability scores.",
            "hard_gate": "Weak or unproven signals cannot drive large size.",
            "if_gate_fails": "Use only tiny paper review or keep as research.",
            "app_page": "Performance -> Signal lab",
            "active_files": "signal_ic_decay_failure.csv; live_ic_observation_ledger.csv; signal_reliability_calibrator.csv",
            "owner_role": "Quant researcher",
            "maturity_now_pct": 38,
            "gap_to_top_fund": "Needs longer true live history, walk-forward tests, and regime-specific IC.",
        },
        {
            "stage_order": 11,
            "stage_name": "Candidate Promotion",
            "desk_name": "Idea Gate",
            "plain_goal": "Promote only the few ideas with enough signal, proof, and risk permission.",
            "primary_inputs": "All layer outputs, proof workbench, final gate bridge.",
            "primary_outputs": "Promote, wait, repair data, prove event, or reject.",
            "hard_gate": "Blocked ideas cannot move to sizing.",
            "if_gate_fails": "Fix the first blocker instead of searching for trades.",
            "app_page": "Home -> Final Gate Bridge",
            "active_files": "institutional_promotion_gate.csv; pm_review_final_gate_bridge.csv; pm_review_final_gate_next_actions.csv",
            "owner_role": "PM",
            "maturity_now_pct": 60,
            "gap_to_top_fund": "Needs clearer PM override policy and stronger evidence acceptance workflow.",
        },
        {
            "stage_order": 12,
            "stage_name": "Portfolio Builder",
            "desk_name": "Portfolio Construction",
            "plain_goal": "Turn approved ideas into a portfolio that respects risk, factor, and turnover limits.",
            "primary_inputs": "Approved candidates, expected returns, covariance, sector/factor budgets.",
            "primary_outputs": "Suggested research weights and rejected overweight ideas.",
            "hard_gate": "No idea can exceed single-name, sector, factor, or liquidity limits.",
            "if_gate_fails": "Reduce, diversify, or reject the idea.",
            "app_page": "Risk -> Portfolio builder",
            "active_files": "institutional_portfolio_optimizer.csv; portfolio_optimized_weights.csv; sector_factor_budget.csv",
            "owner_role": "Portfolio engineer",
            "maturity_now_pct": 48,
            "gap_to_top_fund": "Needs robust optimizer, uncertainty bands, sleeve budgets, and turnover-aware constraints.",
        },
        {
            "stage_order": 13,
            "stage_name": "Risk Budget",
            "desk_name": "Risk Desk",
            "plain_goal": "Estimate how much can be lost by ticker, sector, factor, macro shock, and crisis correlation.",
            "primary_inputs": "Weights, returns, sector map, factor proxies, stress scenarios.",
            "primary_outputs": "VaR, CVaR, drawdown, concentration, correlation, and stop warnings.",
            "hard_gate": "Risk can veto every idea, including high-scoring ideas.",
            "if_gate_fails": "Cut size, hedge, wait, or repair the book.",
            "app_page": "Risk -> Risk desk",
            "active_files": "portfolio_var_cvar_summary.csv; single_name_risk_budget.csv; crisis_correlation_stress.csv; institutional_risk_master_gate.csv",
            "owner_role": "Risk manager",
            "maturity_now_pct": 58,
            "gap_to_top_fund": "Needs calibrated factor risk model, intraday risk, margin, borrow, and crowding risk.",
        },
        {
            "stage_order": 14,
            "stage_name": "Execution Cost And Liquidity",
            "desk_name": "Execution Research",
            "plain_goal": "Ask whether the paper idea would survive real spreads, volume limits, and market impact.",
            "primary_inputs": "Bid/ask proxy, liquidity, participation rate, stress cost.",
            "primary_outputs": "Cost estimate, liquidity block, participation cap, execution warning.",
            "hard_gate": "If spread or liquidity data is missing, do not size the idea.",
            "if_gate_fails": "Get a quote snapshot or mark the idea as data repair first.",
            "app_page": "Risk -> Execution cost",
            "active_files": "execution_cost_stress.csv; execution_playbook.csv; conditional_action_tickets.csv",
            "owner_role": "Execution analyst",
            "maturity_now_pct": 35,
            "gap_to_top_fund": "Needs real bid/ask, TCA, market impact, auction risk, and fill-failure assumptions.",
        },
        {
            "stage_order": 15,
            "stage_name": "PM Evidence Review",
            "desk_name": "PM Gate",
            "plain_goal": "Make a human-readable proof packet before any idea gets even tiny paper size.",
            "primary_inputs": "Autofill suggestions, proof tasks, source checks, bridge patches.",
            "primary_outputs": "Accepted, rejected, needs outside proof, or still blocked.",
            "hard_gate": "Human acceptance is required for uncertain evidence.",
            "if_gate_fails": "Return to proof desk and do not promote.",
            "app_page": "Risk -> Source Proof Desk",
            "active_files": "pm_review_evidence_acceptance_input.csv; pm_evidence_source_proof_input.csv; pm_evidence_proof_acceptance_patch.csv",
            "owner_role": "PM / reviewer",
            "maturity_now_pct": 52,
            "gap_to_top_fund": "Needs automated outside-source ingestion and reviewer workflow UI.",
        },
        {
            "stage_order": 16,
            "stage_name": "Paper Book And Live Monitor",
            "desk_name": "Monitoring Desk",
            "plain_goal": "Track paper positions, manual NAV, alerts, and whether the thesis is still valid.",
            "primary_inputs": "Paper trades, manual NAV, alerts, watchlist, risk limits.",
            "primary_outputs": "Paper PnL, live NAV, breach alerts, thesis updates.",
            "hard_gate": "Live orders stay disabled; this is monitoring only.",
            "if_gate_fails": "Use manual paper update and do not connect a broker.",
            "app_page": "Live / Paper -> Book monitor",
            "active_files": "paper_portfolio_ledger.csv; live_nav_manual.csv; desk_monitor.csv; watchlist_tracker.csv",
            "owner_role": "PM assistant",
            "maturity_now_pct": 45,
            "gap_to_top_fund": "Needs cleaner NAV workflow, alert routing, and intraday monitoring.",
        },
        {
            "stage_order": 17,
            "stage_name": "Attribution And Learning",
            "desk_name": "Learning Desk",
            "plain_goal": "After outcomes happen, learn what worked, what failed, and which rules need changing.",
            "primary_inputs": "Closed paper trades, forward returns, signal logs, decision memory.",
            "primary_outputs": "PnL attribution, missed-winner review, protected-loss review, rule-change candidates.",
            "hard_gate": "Do not change model weights until enough verified samples exist.",
            "if_gate_fails": "Record the lesson but do not overfit.",
            "app_page": "Performance -> Attribution",
            "active_files": "pnl_attribution.csv; decision_memory_center.csv; learning_attribution_summary.csv",
            "owner_role": "Quant PM",
            "maturity_now_pct": 40,
            "gap_to_top_fund": "Needs larger live sample, regime-conditioned attribution, and formal model-change control.",
        },
        {
            "stage_order": 18,
            "stage_name": "Governance And Handoff",
            "desk_name": "Control Room",
            "plain_goal": "Keep the system understandable, auditable, and safe for the next developer or PM.",
            "primary_inputs": "Run logs, handoff docs, QA results, source files.",
            "primary_outputs": "QA checklist, file map, active spine, next priorities.",
            "hard_gate": "If the system cannot explain itself, do not add more complexity.",
            "if_gate_fails": "Simplify the page, document the active files, and archive confusing output.",
            "app_page": "System -> All outputs",
            "active_files": "HANDOFF_CANYON_V9.md; daily_runner_log.csv; active_spine.csv",
            "owner_role": "Engineering lead",
            "maturity_now_pct": 50,
            "gap_to_top_fund": "Needs formal release process, test suite, data contracts, and cleaner product navigation.",
        },
    ]
    return pd.DataFrame([guard_flags(r) for r in rows], columns=STAGE_COLUMNS)


def build_edges(stages: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(len(stages) - 1):
        current = stages.iloc[i]
        nxt = stages.iloc[i + 1]
        rows.append({
            "from_stage": current["stage_name"],
            "to_stage": nxt["stage_name"],
            "handoff_payload": current["primary_outputs"],
            "gate_condition": current["hard_gate"],
            "blocking_failure": current["if_gate_fails"],
            "why_it_matters": (
                f"{current['stage_name']} must be trustworthy before {nxt['stage_name']} can use it."
            ),
        })
    rows.extend([
        {
            "from_stage": "Risk Budget",
            "to_stage": "Options Route",
            "handoff_payload": "Risk veto, size cap, event gap warning, liquidity warning.",
            "gate_condition": "Options cannot be considered until risk and event gates allow research.",
            "blocking_failure": "Do not look for calls, puts, or new size yet.",
            "why_it_matters": "A hot option chain is not enough if the portfolio cannot afford the loss.",
        },
        {
            "from_stage": "News To Industry Chain",
            "to_stage": "PM Evidence Review",
            "handoff_payload": "Headline, affected tickers, causal thesis, source link, proof needed.",
            "gate_condition": "The event must have a verifiable source and event-time price reaction.",
            "blocking_failure": "Keep the headline in read-only mode.",
            "why_it_matters": "Read-through ideas are easy to overfit unless the causal chain is proven.",
        },
        {
            "from_stage": "Execution Cost And Liquidity",
            "to_stage": "Portfolio Builder",
            "handoff_payload": "Spread cost, stress cost, participation cap, liquidity block.",
            "gate_condition": "A paper weight must survive realistic cost and liquidity assumptions.",
            "blocking_failure": "Reduce the paper weight or block the idea.",
            "why_it_matters": "A signal can look profitable before trading costs and fail after costs.",
        },
    ])
    return pd.DataFrame([guard_flags(r) for r in rows], columns=EDGE_COLUMNS)


def build_daily_runbook() -> pd.DataFrame:
    rows = [
        {
            "run_order": 1,
            "when_to_use": "Start of day",
            "human_action": "Open Home and run the daily system.",
            "system_action": "Refresh all active engines and logs.",
            "output_to_read": "Home top answer and run status.",
            "do_not_do": "Do not read old files before checking freshness.",
            "page_to_open": "Home",
        },
        {
            "run_order": 2,
            "when_to_use": "Before looking at ideas",
            "human_action": "Read Data Reliability and repair any missing price or stale file.",
            "system_action": "Shows missing or stale inputs.",
            "output_to_read": "Data reliability answer.",
            "do_not_do": "Do not promote a ticker with broken data.",
            "page_to_open": "System",
        },
        {
            "run_order": 3,
            "when_to_use": "Before sizing",
            "human_action": "Read Market Weather and Risk Desk.",
            "system_action": "Checks macro tone, stress, VaR, correlation, and event risk.",
            "output_to_read": "Risk desk answer.",
            "do_not_do": "Do not let a strong story jump ahead of a risk veto.",
            "page_to_open": "Risk",
        },
        {
            "run_order": 4,
            "when_to_use": "When a headline appears",
            "human_action": "Open News and read why it may help or hurt each ticker.",
            "system_action": "Maps direct beneficiaries, linked peers, and vulnerable names.",
            "output_to_read": "News to read first.",
            "do_not_do": "Do not size from a headline until source, timing, and reaction are checked.",
            "page_to_open": "News",
        },
        {
            "run_order": 5,
            "when_to_use": "When a ticker looks interesting",
            "human_action": "Open Ideas and check short, medium, and long route.",
            "system_action": "Shows whether the ticker is wait, repair, prove, or tiny paper review.",
            "output_to_read": "Final gate and idea route.",
            "do_not_do": "Do not turn wait into buy.",
            "page_to_open": "Ideas",
        },
        {
            "run_order": 6,
            "when_to_use": "Before any option research",
            "human_action": "Check whether options are allowed at all.",
            "system_action": "Combines event risk, gamma, liquidity, and risk gate.",
            "output_to_read": "Option route.",
            "do_not_do": "Do not search for calls or puts when the idea is blocked.",
            "page_to_open": "Ideas",
        },
        {
            "run_order": 7,
            "when_to_use": "When the system asks for proof",
            "human_action": "Open Source Proof Desk and fill verified source, observed value, reviewer, and date.",
            "system_action": "Builds proof tasks and acceptance patch rows.",
            "output_to_read": "Missing proof first.",
            "do_not_do": "Do not accept a proof row from model text alone.",
            "page_to_open": "Risk",
        },
        {
            "run_order": 8,
            "when_to_use": "When proof is filled",
            "human_action": "Review the proof-to-acceptance bridge and copy only clean patch rows.",
            "system_action": "Checks conflicts against Step204.",
            "output_to_read": "Manual Step204 patch.",
            "do_not_do": "Do not override conflicts without a note.",
            "page_to_open": "Risk",
        },
        {
            "run_order": 9,
            "when_to_use": "Before paper tracking",
            "human_action": "Open Live / Paper and check paper book and manual NAV.",
            "system_action": "Shows paper PnL, NAV, and monitor alerts.",
            "output_to_read": "Book monitor.",
            "do_not_do": "Do not treat paper PnL as live brokerage PnL.",
            "page_to_open": "Live / Paper",
        },
        {
            "run_order": 10,
            "when_to_use": "End of day",
            "human_action": "Record what happened and what rule should be reviewed later.",
            "system_action": "Feeds decision memory and attribution.",
            "output_to_read": "Performance and decision memory.",
            "do_not_do": "Do not change model weights from one anecdote.",
            "page_to_open": "Performance",
        },
    ]
    return pd.DataFrame([guard_flags(r) for r in rows], columns=RUNBOOK_COLUMNS)


def build_software_map() -> pd.DataFrame:
    rows = [
        {
            "app_tab": "Home",
            "panel": "Operating Flow",
            "what_user_should_look_for": "The whole system path and the first bottleneck.",
            "input_files": "quant_fund_operating_flow_state.json",
            "output_files": "quant_fund_operating_flow_stages.csv; quant_fund_operating_flow_daily_runbook.csv",
            "related_steps": "208",
            "maturity_now_pct": 55,
            "next_upgrade": "Turn the flow into clickable navigation anchors.",
        },
        {
            "app_tab": "Today",
            "panel": "Daily Answer",
            "what_user_should_look_for": "What changed today, what to watch, and what to avoid.",
            "input_files": "daily_picks.csv; daily_alerts.csv; dynamic_daily_workflow.csv",
            "output_files": "today_action_queue.csv",
            "related_steps": "144, 150, 152",
            "maturity_now_pct": 58,
            "next_upgrade": "Make a cleaner one-screen PM answer.",
        },
        {
            "app_tab": "Ideas",
            "panel": "Final Gate / Route",
            "what_user_should_look_for": "Whether a ticker is blocked, waiting, proof-needed, or tiny paper review.",
            "input_files": "institutional_promotion_gate.csv; pm_review_final_gate_bridge.csv",
            "output_files": "pm_review_final_gate_next_actions.csv",
            "related_steps": "189, 195, 202",
            "maturity_now_pct": 62,
            "next_upgrade": "Separate short, medium, long, call, put, and stock route in one card.",
        },
        {
            "app_tab": "News",
            "panel": "News To Read First",
            "what_user_should_look_for": "Who may benefit, who may be hurt, why, and what proof is missing.",
            "input_files": "news_impact_targeting.csv; event_causal_chain.csv",
            "output_files": "pm_evidence_source_proof_input.csv",
            "related_steps": "99, 129, 160, 206",
            "maturity_now_pct": 50,
            "next_upgrade": "Add better source links, supply-chain graph, and cleaner why-help/why-hurt language.",
        },
        {
            "app_tab": "Risk",
            "panel": "Risk Desk",
            "what_user_should_look_for": "Can the portfolio afford this idea if it is wrong?",
            "input_files": "single_name_risk_budget.csv; portfolio_var_cvar_summary.csv; crisis_correlation_stress.csv",
            "output_files": "institutional_risk_master_gate.csv",
            "related_steps": "111-118, 131-141",
            "maturity_now_pct": 58,
            "next_upgrade": "Calibrate thresholds with longer history and add factor-model quality checks.",
        },
        {
            "app_tab": "Performance",
            "panel": "Signal Skill / Backtest Trust",
            "what_user_should_look_for": "Which signals worked, how fast they decay, and which backtests are credible.",
            "input_files": "signal_ic_decay_failure.csv; backtest_credibility_system.csv",
            "output_files": "signal_reliability_calibrator.csv",
            "related_steps": "84, 94, 155, 156, 162",
            "maturity_now_pct": 42,
            "next_upgrade": "Add real live sample history and walk-forward validation summaries.",
        },
        {
            "app_tab": "Live / Paper",
            "panel": "Book Monitor",
            "what_user_should_look_for": "Paper position, manual NAV, alerts, and thesis drift.",
            "input_files": "paper_portfolio_ledger.csv; live_nav_manual.csv; desk_monitor.csv",
            "output_files": "paper_nav_attribution.csv",
            "related_steps": "69, 89, 115, 119",
            "maturity_now_pct": 45,
            "next_upgrade": "Make manual NAV entry and monitor alerts clearer.",
        },
        {
            "app_tab": "System",
            "panel": "Data, Repair, Proof, Outputs",
            "what_user_should_look_for": "What is stale, broken, missing proof, or ready for review.",
            "input_files": "run_log.csv; data_repair_queue.csv; pm_evidence_source_proof_input.csv",
            "output_files": "system QA and output map",
            "related_steps": "70, 197, 198, 203-208",
            "maturity_now_pct": 55,
            "next_upgrade": "Create fewer panels with stronger summaries and less table noise.",
        },
    ]
    return pd.DataFrame([guard_flags(r) for r in rows], columns=SOFTWARE_COLUMNS)


def find_current_bottleneck() -> str:
    proof_state = read_json_safe(ROOT / "pm_evidence_source_proof_state.json", {})
    bridge_state = read_json_safe(ROOT / "pm_evidence_proof_acceptance_bridge_state.json", {})
    needs_proof = int(proof_state.get("needs_proof_count", 0) or 0)
    ready_proof = int(proof_state.get("ready_for_accept_count", 0) or 0)
    patch_rows = int(bridge_state.get("patch_rows", 0) or 0)
    if needs_proof and ready_proof == 0:
        return "Outside-source proof is the main bottleneck: fill source, observed value, reviewer, and date before accepting evidence."
    if ready_proof and patch_rows == 0:
        return "Proof exists, but the bridge still needs conflict review before Step204 can be updated."
    if patch_rows:
        return "Manual acceptance patch rows are ready; a human must copy them into Step204 and rerun."
    return "No single bottleneck is dominant. Start with data freshness, risk, and proof gates."


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stages = build_stages()
    edges = build_edges(stages)
    runbook = build_daily_runbook()
    software_map = build_software_map()

    hard_gates = int(stages["hard_gate"].astype(str).str.len().gt(0).sum())
    avg_maturity = round(float(pd.to_numeric(stages["maturity_now_pct"], errors="coerce").mean()), 1)
    weakest = stages.sort_values("maturity_now_pct").head(3)["stage_name"].tolist()
    bottleneck = find_current_bottleneck()
    state = {
        "date": today_str(),
        "status": "Active",
        "stage_count": int(len(stages)),
        "hard_gate_count": hard_gates,
        "daily_runbook_steps": int(len(runbook)),
        "software_panels": int(len(software_map)),
        "average_maturity_now_pct": avg_maturity,
        "weakest_stages": ", ".join(weakest),
        "current_bottleneck": bottleneck,
        "top_gap": "Point-in-time data, signal validation, optimizer quality, execution cost, and source-proof automation.",
        "plain_answer": (
            f"Quant fund operating flow is active. Canyon v9 now has {len(stages)} software stages "
            f"from data intake to learning and governance. Current bottleneck: {bottleneck}"
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return stages, edges, runbook, software_map, state


def main() -> None:
    stages, edges, runbook, software_map, state = build_outputs()
    stages.to_csv(OUT_STAGES, index=False)
    edges.to_csv(OUT_EDGES, index=False)
    runbook.to_csv(OUT_RUNBOOK, index=False)
    software_map.to_csv(OUT_SOFTWARE_MAP, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Operating Flow Stages\n\n" + df_to_markdown(stages, max_rows=40),
        "## Gate Handoffs\n\n" + df_to_markdown(edges, max_rows=60),
        "## Daily Runbook\n\n" + df_to_markdown(runbook, max_rows=30),
        "## Software Map\n\n" + df_to_markdown(software_map, max_rows=30),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Quant Fund Operating Flow", sections)
    print(
        "Step208 complete: "
        f"{len(stages)} stages, {len(edges)} handoffs, {len(runbook)} runbook steps, "
        f"{len(software_map)} software panels."
    )


if __name__ == "__main__":
    main()
