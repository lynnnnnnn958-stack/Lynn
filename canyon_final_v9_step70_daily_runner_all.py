#!/usr/bin/env python3
"""
canyon_final_v9_step70_daily_runner_all.py
One-click daily batch runner for Canyon v9 quantitative research system.
Chains all engine scripts in dependency order, captures output, logs results.
"""

import argparse
import csv
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

ENGINES = [
    # ── Base layer (price, backtest, ML, fundamentals) ────────────────────────
    (56,  "10-Layer Runner",      "canyon_final_v9_step56_full_10_layer_daily_runner_v2.py", []),
    (61,  "Data Source Health",   "canyon_final_v9_step61_data_source_health.py",             []),
    (62,  "Backtest Engine",      "canyon_final_v9_step62_backtest_engine.py",                []),
    (65,  "Earnings NLP",         "canyon_final_v9_step65_earnings_nlp.py",                   []),
    (66,  "ML Signals",           "canyon_final_v9_step66_ml_signals.py",                     []),
    (67,  "SHAP Explainer",       "canyon_final_v9_step67_shap_explainer.py",                 []),
    (68,  "Fundamental Features", "canyon_final_v9_step68_fundamental_features.py",           []),
    (69,  "Paper Sim Rebalance",  "canyon_final_v9_step69_paper_sim.py",                      ["--rebalance"]),
    (691, "Paper Sim MTM",        "canyon_final_v9_step69_paper_sim.py",                      ["--mark-to-market"]),
    # ── Alpha signal layer (must run BEFORE step 87 aggregator) ──────────────
    (76,  "Regime Detector",      "canyon_final_v9_step76_regime_detector.py",                []),
    # Daily run uses score-only mode. Full walk-forward validation is slow and belongs
    # in weekly/manual QA; otherwise Step77 often times out before the dashboard updates.
    (77,  "Regime ML Signals",    "canyon_final_v9_step77_regime_ml.py",                      ["--score"]),
    (79,  "FinBERT Sentiment",    "canyon_final_v9_step79_finbert_sentiment.py",               ["--top", "200"]),
    (80,  "Earnings Revision",    "canyon_final_v9_step80_earnings_revision.py",              ["--top", "200"]),
    (81,  "Earnings Surprise",    "canyon_final_v9_step81_earnings_surprise.py",              ["--top", "100", "--fast"]),
    (82,  "Options Signals",      "canyon_final_v9_step82_options_signals.py",                ["--top", "100"]),
    (83,  "Short Squeeze",        "canyon_final_v9_step83_short_squeeze.py",                  ["--top", "200"]),
    (85,  "Insider Signal",       "canyon_final_v9_step85_insider_signal.py",                 ["--top", "200"]),
    # Step 127 runs BEFORE step84/step87 so momentum_scores.csv is ready for the aggregator.
    # Institutional 4-component momentum: J-T 12-1m (35%) + 52-week-high (30%)
    # + vol-scaled/AQR (25%) + residual/idiosyncratic (10%).  Crash-protection built in.
    (127, "Momentum Signal",     "canyon_final_v9_step127_momentum_signal.py",               []),
    (84,  "Live IC Tracker",      "canyon_final_v9_step84_live_ic_tracker.py",                []),
    # ── Aggregation + optimization (depend on ALL signals above) ─────────────
    # Step 88: incremental (only fetches tickers missing a sector — near-instant
    # after the initial build).  Run weekly with --weekly --refresh to catch
    # quarterly S&P 500 constituent changes.
    (88,  "Sector Map",           "canyon_final_v9_step88_sector_map.py",                     []),
    (87,  "Alpha Aggregator",     "canyon_final_v9_step87_alpha_aggregator.py",               []),
    (63,  "Portfolio Optimizer",  "canyon_final_v9_step63_portfolio_optimizer.py",            []),
    # ── P1: cvxpy MV Optimizer — replaces rule-based weight truncation ────────
    # Reads alpha_scores.csv + price covariance → cvxpy_weights.csv
    (220, "cvxpy MV Optimizer",  "canyon_final_v9_step220_cvxpy_optimizer.py",               []),
    # ── P2: Fama-French 5-Factor Attribution ──────────────────────────────────
    # Downloads FF5 daily factors, runs OLS, outputs factor_attribution.csv
    (221, "FF5 Factor Attribution", "canyon_final_v9_step221_factor_attribution.py",         []),
    # ── P3: Rolling IC Tracker ─────────────────────────────────────────────────
    # Saves daily signal snapshot, computes forward IC → ic_daily_log.csv
    (222, "Rolling IC Tracker",  "canyon_final_v9_step222_rolling_ic_tracker.py",            []),
    # ── Phase 6 — Review, backtest, journaling (depend on alpha + signals above) ──
    (89,  "Trade Journal",        "canyon_final_v9_step89_trade_journal.py",                  []),
    (90,  "Options Backtest",     "canyon_final_v9_step90_options_backtest.py",               []),
    (91,  "Event Backtest",       "canyon_final_v9_step91_event_backtest.py",                 []),
    (92,  "Stock Backtest",       "canyon_final_v9_step92_stock_strategy_backtest.py",        []),
    # ── Phase 7 — Risk, calibration, macro, attribution, alerts ──────────────
    (95,  "Macro Signals",        "canyon_final_v9_step95_macro_signals.py",                  []),
    (93,  "Risk Manager",         "canyon_final_v9_step93_risk_manager.py",                   []),
    (94,  "Signal Calibrator",    "canyon_final_v9_step94_signal_calibrator.py",              []),
    (96,  "P&L Attribution",      "canyon_final_v9_step96_pnl_attribution.py",               []),
    (98,  "Daily Alerts",         "canyon_final_v9_step98_daily_alerts.py",                   []),
    (99,  "News Aggregator",      "canyon_final_v9_step99_news_aggregator.py",                ["--top", "50"]),
    (100, "Sector Calendar",      "canyon_final_v9_step100_sector_calendar.py",               []),
    (101, "Short Signals",        "canyon_final_v9_step101_short_signals.py",                  []),
    (102, "Earnings Calendar",   "canyon_final_v9_step102_earnings_calendar.py",              []),
    (103, "Position Rules",      "canyon_final_v9_step103_position_rules.py",                 []),
    (104, "Signal Correlation",  "canyon_final_v9_step104_signal_correlation.py",             []),
    (105, "Stress Test",         "canyon_final_v9_step105_stress_test.py",                    []),
    (107, "Watchlist Tracker",   "canyon_final_v9_step107_watchlist.py",                      []),
    (108, "Weekly Report",       "canyon_final_v9_step108_weekly_report.py",                  []),
    (109, "Factor Decomp",       "canyon_final_v9_step109_factor_decomp.py",                  []),
    (110, "Portfolio Risk Filter", "canyon_final_v9_step110_portfolio_risk_filter.py",         []),
    (111, "Single-name Risk Budget", "canyon_final_v9_step111_single_name_risk_budget.py",      []),
    (112, "Sector and Factor Budget", "canyon_final_v9_step112_sector_factor_budget.py",        []),
    (113, "Macro Risk Sensitivity", "canyon_final_v9_step113_macro_risk_sensitivity.py",        []),
    (114, "Drawdown Vol Kelly",   "canyon_final_v9_step114_drawdown_vol_kelly.py",             []),
    (115, "Paper NAV Attribution", "canyon_final_v9_step115_live_nav_attribution_slippage.py",  []),
    (116, "Correlation Beta Constraints", "canyon_final_v9_step116_correlation_beta_constraints.py", []),
    (117, "Extreme Market Protection", "canyon_final_v9_step117_extreme_market_protection.py",  []),
    (118, "Institutional Risk Master Gate", "canyon_final_v9_step118_institutional_risk_master_gate.py", []),
    (131, "Risk Desk Summary", "canyon_final_v9_step131_risk_desk_summary.py", []),
    (132, "Risk Threshold Calibration", "canyon_final_v9_step132_risk_threshold_calibration.py", []),
    (133, "Risk Policy Review", "canyon_final_v9_step133_risk_policy_review.py", []),
    (134, "Risk Policy Decision Log", "canyon_final_v9_step134_risk_policy_decision_log.py", []),
    (136, "Historical Risk Evidence", "canyon_final_v9_step136_historical_risk_evidence.py", []),
    (137, "Historical Threshold Review", "canyon_final_v9_step137_historical_threshold_review.py", []),
    (138, "Threshold Impact Simulator", "canyon_final_v9_step138_threshold_impact_simulator.py", []),
    (139, "Risk Policy Change Control", "canyon_final_v9_step139_risk_policy_change_control.py", []),
    (140, "Risk Policy Approval QA", "canyon_final_v9_step140_risk_policy_approval_qa.py", []),
    (141, "Risk Policy Dry-Run Guard", "canyon_final_v9_step141_risk_policy_dry_run_guard.py", []),
    (135, "Risk Policy Evidence Pack", "canyon_final_v9_step135_risk_policy_evidence_pack.py", []),
    (119, "Desk Monitor", "canyon_final_v9_step119_desk_monitor.py", []),
    (163, "Event Time Truth Ledger", "canyon_final_v9_step163_event_time_truth_ledger.py", []),
    (162, "Live IC Observation Ledger", "canyon_final_v9_step162_live_ic_observation_ledger.py", []),
    (161, "PIT Seed Store Builder", "canyon_final_v9_step161_pit_seed_store_builder.py", []),
    (121, "Data Truth Ledger", "canyon_final_v9_step121_data_truth_ledger.py", []),
    (159, "Point-in-Time Truth Readiness", "canyon_final_v9_step159_pit_truth_readiness.py", []),
    (122, "Backtest Bias Guard", "canyon_final_v9_step122_backtest_bias_guard.py", []),
    (155, "Backtest Credibility System", "canyon_final_v9_step155_backtest_credibility_system.py", []),
    (156, "Signal IC Decay Failure", "canyon_final_v9_step156_signal_ic_decay_failure.py", []),
    (124, "Event Research Dossier", "canyon_final_v9_step124_event_research_dossier.py", []),
    (123, "Institutional Portfolio Builder", "canyon_final_v9_step123_institutional_portfolio_builder.py", []),
    (125, "Execution Playbook", "canyon_final_v9_step125_execution_playbook.py", []),
    (126, "Institutional Risk Overlays", "canyon_final_v9_step126_institutional_risk_overlays.py", []),
    (128, "Timeframe Options Playbook", "canyon_final_v9_step128_timeframe_options_playbook.py", []),
    (129, "News Impact Targeting", "canyon_final_v9_step129_news_impact_targeting.py", []),
    (130, "Theme Candidate Enrichment", "canyon_final_v9_step130_theme_candidate_enrichment.py", []),
    (160, "Event Causal Chain", "canyon_final_v9_step160_event_causal_chain.py", []),
    (164, "Event Backtest Admissibility", "canyon_final_v9_step164_event_backtest_admissibility.py", []),
    (165, "Event Signal Local Audit", "canyon_final_v9_step165_event_signal_local_audit.py", []),
    (166, "Event Signal Reliability Calibrator", "canyon_final_v9_step166_event_signal_reliability_calibrator.py", []),
    (142, "Sector Cycle Linkage", "canyon_final_v9_step142_sector_cycle_linkage.py", []),
    (170, "Institutional Subsector Cycle", "canyon_final_v9_step170_institutional_subsector_cycle.py", []),
    (157, "Institutional Portfolio Optimizer", "canyon_final_v9_step157_institutional_portfolio_optimizer.py", []),
    (158, "Execution Cost Stress Model", "canyon_final_v9_step158_execution_cost_model.py", []),
    (143, "Sector Timeframe Router", "canyon_final_v9_step143_sector_timeframe_strategy_router.py", []),
    (144, "Dynamic Daily Workflow", "canyon_final_v9_step144_dynamic_daily_workflow.py", []),
    (145, "Ticker Evidence Binder", "canyon_final_v9_step145_ticker_evidence_binder.py", []),
    (146, "Decision Conflict Resolver", "canyon_final_v9_step146_decision_conflict_resolver.py", []),
    (147, "Conflict Resolution Playbook", "canyon_final_v9_step147_conflict_resolution_playbook.py", []),
    (148, "Gate Upgrade Simulator", "canyon_final_v9_step148_gate_upgrade_simulator.py", []),
    (149, "Gate-Clear Candidate Ranking", "canyon_final_v9_step149_gate_clear_candidate_ranking.py", []),
    (150, "Conditional Action Tickets", "canyon_final_v9_step150_conditional_action_tickets.py", []),
    (151, "Ticker Decision Room", "canyon_final_v9_step151_ticker_decision_room.py", []),
    (152, "Ticker Decision Cards", "canyon_final_v9_step152_ticker_decision_cards.py", []),
    (167, "Horizon Vehicle Router", "canyon_final_v9_step167_horizon_vehicle_router.py", []),
    (171, "Event Read-Through Decision Engine", "canyon_final_v9_step171_event_readthrough_decision_engine.py", []),
    (168, "Institutional Strategy Thesis", "canyon_final_v9_step168_institutional_strategy_thesis.py", []),
    (169, "Research Promotion Gate", "canyon_final_v9_step169_research_promotion_gate.py", []),
    (120, "Institutional Upgrade Master", "canyon_final_v9_step120_institutional_upgrade_master.py", []),
    (173, "Options Execution Route Engine", "canyon_final_v9_step173_options_execution_route_engine.py", []),
    (174, "Options Unlock Board", "canyon_final_v9_step174_options_unlock_board.py", []),
    (175, "Risk Unlock Sequencer", "canyon_final_v9_step175_risk_unlock_sequencer.py", []),
    (176, "Risk Repair Simulator", "canyon_final_v9_step176_risk_repair_simulator.py", []),
    (177, "Risk Repair Recommendation Board", "canyon_final_v9_step177_risk_repair_recommendation_board.py", []),
    (178, "Action Readiness Monitor", "canyon_final_v9_step178_action_readiness_monitor.py", []),
    (179, "Action Readiness Drilldown", "canyon_final_v9_step179_action_readiness_drilldown.py", []),
    (180, "Action Readiness Detail Cards", "canyon_final_v9_step180_action_readiness_detail_cards.py", []),
    (172, "Institutional Depth Upgrade Engine", "canyon_final_v9_step172_institutional_depth_upgrade_engine.py", []),
    (181, "Deep Decision Desk", "canyon_final_v9_step181_deep_decision_desk.py", []),
    (182, "Ticker Research Memo Desk", "canyon_final_v9_step182_ticker_research_memo.py", []),
    (183, "Ticker Reviewability Progress", "canyon_final_v9_step183_reviewability_progress.py", []),
    (184, "Proof Queue Workbench", "canyon_final_v9_step184_proof_queue_workbench.py", []),
    (185, "Data Truth Decision Desk", "canyon_pit_truth_depth.py", []),
    (186, "Execution / TCA Decision Desk", "canyon_execution_tca_depth.py", []),
    (1851, "Sharpe 4 Target Upgrade", "canyon_final_v9_step185_sharpe_target4_upgrade.py", []),
    (1861, "Sharpe 4 P0 Repair Engine", "canyon_final_v9_step186_sharpe4_p0_repair_engine.py", []),
    (187, "Sharpe 4 Recovery Roadmap", "canyon_final_v9_step187_sharpe4_recovery_roadmap.py", []),
    (188, "Sharpe 4 Risk-Book Intake", "canyon_final_v9_step188_sharpe4_risk_book_intake.py", []),
    (189, "Sharpe 4 Promotion Gate", "canyon_final_v9_step189_sharpe4_risk_book_promotion_gate.py", []),
    (190, "Simple Sharpe 4 Command Center", "canyon_final_v9_step190_simple_sharpe4_command_center.py", []),
    (191, "Sharpe 4 Proof Workbench", "canyon_final_v9_step191_proof_workbench.py", []),
    (192, "Manual Proof Review Gate", "canyon_final_v9_step192_manual_proof_review_gate.py", []),
    (193, "PM Morning Brief", "canyon_final_v9_step193_pm_morning_brief.py", []),
    (194, "Institutional Depth 5 Workbench", "canyon_final_v9_step194_institutional_depth5_workbench.py", []),
    (195, "Institutional Promotion Gate", "canyon_final_v9_step195_institutional_promotion_gate.py", []),
    (196, "Decision Memory Center", "canyon_final_v9_step196_decision_memory_center.py", []),
    (197, "Price / Data Reliability Center", "canyon_final_v9_step197_price_data_reliability_center.py", []),
    (198, "Data Repair Engine", "canyon_final_v9_step198_data_repair_engine.py", []),
    (199, "Risk Book Seed Engine", "canyon_final_v9_step199_risk_book_seed_engine.py", []),
    (200, "Risk Seed Approval Workbench", "canyon_final_v9_step200_risk_seed_approval_workbench.py", []),
    (201, "Risk Seed PM Review Intake", "canyon_final_v9_step201_risk_seed_pm_review_intake.py", []),
    (202, "PM Review Final Gate Bridge", "canyon_final_v9_step202_pm_review_final_gate_bridge.py", []),
    (203, "PM Review Evidence Autofill", "canyon_final_v9_step203_pm_review_evidence_autofill.py", []),
    (204, "PM Evidence Acceptance Gate", "canyon_final_v9_step204_pm_evidence_acceptance_gate.py", []),
    (205, "PM Evidence Review Triage", "canyon_final_v9_step205_pm_evidence_review_triage.py", []),
    (206, "PM Evidence Source Proof Desk", "canyon_final_v9_step206_pm_evidence_source_proof_desk.py", []),
    (207, "PM Proof-to-Acceptance Bridge", "canyon_final_v9_step207_pm_evidence_proof_acceptance_bridge.py", []),
    (208, "Quant Fund Operating Flow", "canyon_final_v9_step208_quant_fund_operating_flow.py", []),
    (209, "Quant Fund Flow Navigator", "canyon_final_v9_step209_quant_fund_flow_navigator.py", []),
    (210, "Ticker Flow Cards", "canyon_final_v9_step210_ticker_flow_cards.py", []),
    (211, "Proof Collection Workbench", "canyon_final_v9_step211_proof_collection_workbench.py", []),
    (212, "Proof Quality Gate", "canyon_final_v9_step212_proof_quality_gate.py", []),
    (213, "Proof Fill Desk", "canyon_final_v9_step213_proof_fill_desk.py", []),
    (214, "Proof Intake Safe Apply", "canyon_final_v9_step214_proof_intake_safe_apply.py", []),
    (215, "Proof Closure Tracker", "canyon_final_v9_step215_proof_closure_tracker.py", []),
    # Steps 75 & 78 are slow (full price download / financials) — weekly only
]

# Steps skipped with --fast flag
FAST_SKIP = {56, 65, 67, 68}   # slow: 10-layer runner, earnings NLP, SHAP, fundamentals

# Steps only run with --weekly flag (slow: full price cache, financials)
WEEKLY_ONLY = {75, 78}

WEEKLY_ENGINES = [
    (75,  "Universe Expansion",   "canyon_final_v9_step75_universe_expansion.py",             []),
    (78,  "Deep Fundamentals",    "canyon_final_v9_step78_deep_fundamentals.py",               ["--top", "200"]),
]

LOG_CSV    = ROOT / "run_daily_all_log.csv"
REPORT_MD  = ROOT / "run_daily_all_report.md"

CSV_FIELDNAMES = ["date", "step", "name", "status", "duration_s", "returncode", "notes"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canyon v9 one-click daily batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run",   action="store_true",
                   help="Print what would run, don't execute")
    p.add_argument("--fast",      action="store_true",
                   help="Skip slow engines (step56, step68)")
    p.add_argument("--only",      nargs="+", type=str, metavar="STEP",
                   help="Run only these step numbers, e.g. --only 66 67 69")
    p.add_argument("--skip",      nargs="+", type=str, metavar="STEP",
                   help="Skip these step numbers, e.g. --skip 62 68")
    p.add_argument("--timeout",   type=int, default=180, metavar="N",
                   help="Per-engine timeout in seconds (default 180)")
    p.add_argument("--no-notify", action="store_true",
                   help="Suppress macOS notifications")
    p.add_argument("--weekly",    action="store_true",
                   help="Also run weekly-only slow engines (step75 universe, step78 fundamentals)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Engine filtering
# ---------------------------------------------------------------------------

def build_run_list(args: argparse.Namespace):
    """Return the subset of ENGINES to actually run, in order."""
    skip_ids: set[int] = set()

    if args.fast:
        skip_ids |= FAST_SKIP

    if args.skip:
        for s in args.skip:
            skip_ids.add(int(s))

    only_ids: set[int] | None = None
    if args.only:
        only_ids = {int(s) for s in args.only}

    result = []
    all_engines = ENGINES + (WEEKLY_ENGINES if args.weekly else [])
    for engine in all_engines:
        step_id = engine[0]
        if only_ids is not None and step_id not in only_ids:
            continue
        if step_id in skip_ids:
            continue
        result.append(engine)

    return result


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _tail(text: str, n: int = 3) -> str:
    """Return the last n non-empty lines of text, joined."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines[-n:]) if lines else ""


def run_engine(
    step_id: int,
    name: str,
    script: str,
    extra_args: list[str],
    timeout: int,
    dry_run: bool,
) -> dict:
    """Execute a single engine and return a result dict."""
    script_path = ROOT / script
    cmd = [sys.executable, str(script_path)] + extra_args

    result = {
        "step":       step_id,
        "name":       name,
        "script":     script,
        "cmd":        cmd,
        "status":     None,
        "returncode": None,
        "duration_s": None,
        "stdout_tail": "",
        "stderr_excerpt": "",
        "notes":      "",
    }

    if dry_run:
        result.update(status="DRY-RUN", returncode=0, duration_s=0,
                      notes="would run: " + " ".join(cmd))
        return result

    t0 = datetime.now()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
        duration = (datetime.now() - t0).total_seconds()
        ok = proc.returncode == 0

        result.update(
            status="OK" if ok else "FAILED",
            returncode=proc.returncode,
            duration_s=round(duration, 1),
            stdout_tail=_tail(proc.stdout, 3),
            stderr_excerpt=_tail(proc.stderr, 2),
            notes=_tail(proc.stderr, 2) if not ok else "",
        )

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - t0).total_seconds()
        result.update(
            status="TIMEOUT",
            returncode=-1,
            duration_s=round(duration, 1),
            notes=f"killed after {timeout}s",
        )

    except FileNotFoundError:
        result.update(
            status="NOT FOUND",
            returncode=-2,
            duration_s=0.0,
            notes=f"script not found: {script}",
        )

    return result


# ---------------------------------------------------------------------------
# Progress + Summary printing
# ---------------------------------------------------------------------------

STATUS_ICON = {
    "OK":        "✓ OK",
    "FAILED":    "FAILED",
    "TIMEOUT":   "TIMEOUT",
    "NOT FOUND": "NOT FOUND",
    "DRY-RUN":   "DRY-RUN",
    "SKIPPED":   "SKIPPED",
}


def print_summary(results: list[dict], run_dt: datetime, total_s: float) -> None:
    """Print the final box-style summary table."""
    now_str = run_dt.strftime("%Y-%m-%d %H:%M")
    header  = f"Canyon Daily Run — {now_str}"

    col_w_name   = max(len(r["name"]) for r in results) + 2
    col_w_step   = 7
    col_w_status = 10
    col_w_dur    = 10
    col_w_notes  = 28

    row_w = col_w_step + col_w_name + col_w_status + col_w_dur + col_w_notes + 4
    box_w = max(row_w, len(header) + 6)

    def pad(s, w):
        return s[:w].ljust(w)

    top   = "╔" + "═" * box_w + "╗"
    bot   = "╚" + "═" * box_w + "╝"
    hline = "║" + " " + pad(header, box_w - 2) + " " + "║"
    sep   = "║" + "─" * box_w + "║"

    print(top)
    print(hline)
    print(sep)

    # Column header
    ch = (
        pad("step", col_w_step)
        + pad("Engine", col_w_name)
        + pad("Status", col_w_status)
        + pad("Duration", col_w_dur)
        + pad("Notes", col_w_notes)
    )
    print("║ " + ch[:box_w - 2] + " ║")
    print(sep)

    passed = failed = 0
    for r in results:
        icon = STATUS_ICON.get(r["status"], r["status"])
        dur  = f"{r['duration_s']}s" if r["duration_s"] is not None else "-"
        note = (r.get("notes") or "")[:col_w_notes - 1]

        if r["status"] == "OK":
            passed += 1
        elif r["status"] not in ("SKIPPED", "DRY-RUN"):
            failed += 1

        row = (
            pad(f"step{r['step']}", col_w_step)
            + pad(r["name"], col_w_name)
            + pad(icon, col_w_status)
            + pad(dur, col_w_dur)
            + pad(note, col_w_notes)
        )
        print("║ " + row[:box_w - 2] + " ║")

    print(sep)

    total_line = (
        f"TOTAL: {passed}/{len(results)} passed   "
        f"{failed} failed   total {round(total_s)}s"
    )
    print("║ " + pad(total_line, box_w - 2) + " ║")
    print(bot)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def append_csv(results: list[dict], run_dt: datetime) -> None:
    date_str = run_dt.strftime("%Y-%m-%d %H:%M:%S")
    write_header = not LOG_CSV.exists()

    with open(LOG_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "date":       date_str,
                "step":       r["step"],
                "name":       r["name"],
                "status":     r["status"],
                "duration_s": r["duration_s"],
                "returncode": r["returncode"],
                "notes":      (r.get("notes") or "")[:120],
            })


def write_report(results: list[dict], run_dt: datetime, total_s: float) -> None:
    passed = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] not in ("OK", "SKIPPED", "DRY-RUN"))
    now_str = run_dt.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Canyon Daily Run Report",
        f"",
        f"**Run time:** {now_str}  ",
        f"**Total:** {passed}/{len(results)} passed, {failed} failed, {round(total_s)}s  ",
        f"",
        f"| Step | Engine | Status | Duration | Notes |",
        f"|------|--------|--------|----------|-------|",
    ]
    for r in results:
        dur  = f"{r['duration_s']}s" if r["duration_s"] is not None else "-"
        note = (r.get("notes") or "").replace("|", "/")[:80]
        lines.append(
            f"| step{r['step']} | {r['name']} | {r['status']} | {dur} | {note} |"
        )

    lines += ["", "---", f"*Generated by canyon_final_v9_step70_daily_runner_all.py*", ""]

    with open(REPORT_MD, "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# macOS notification
# ---------------------------------------------------------------------------

def send_notification(passed: int, total: int, failed: int) -> None:
    if platform.system() != "Darwin":
        return
    msg = f"{passed}/{total} OK" + (f", {failed} failed" if failed else "")
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{msg}" with title "Canyon Daily Run"'],
            timeout=5,
        )
    except Exception:
        pass  # notifications are optional


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args   = parse_args()
    run_dt = datetime.now()
    run_list = build_run_list(args)

    if not run_list:
        print("No engines to run after applying --only / --skip / --fast filters.")
        return 0

    total_engines = len(run_list)
    if args.dry_run:
        print(f"[DRY-RUN] Would run {total_engines} engine(s):\n")

    results: list[dict] = []
    t_start = datetime.now()

    for idx, (step_id, name, script, extra) in enumerate(run_list, start=1):
        label = f"step{step_id}  {name}"
        print(f"[{idx}/{total_engines}] {label}... ", end="", flush=True)

        result = run_engine(
            step_id=step_id,
            name=name,
            script=script,
            extra_args=extra,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        results.append(result)

        dur_str = f"{result['duration_s']}s" if result["duration_s"] is not None else "-"
        if result["status"] in ("OK", "DRY-RUN"):
            print(f"done ({dur_str})")
        else:
            rc = result["returncode"]
            print(f"{result['status']} (exit {rc}, {dur_str})")
            if result.get("notes"):
                print(f"      ! {result['notes']}")

    total_s = (datetime.now() - t_start).total_seconds()

    print()
    print_summary(results, run_dt, total_s)
    print()

    if not args.dry_run:
        append_csv(results, run_dt)
        write_report(results, run_dt, total_s)
        print(f"Log appended  : {LOG_CSV}")
        print(f"Report written: {REPORT_MD}")

    passed = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] not in ("OK", "SKIPPED", "DRY-RUN"))

    if not args.no_notify and not args.dry_run:
        send_notification(passed, total_engines, failed)

    # ── Risk gate: hard-check all current positions after risk steps run ──────
    if not args.dry_run:
        try:
            import risk_gate as _rg
            _rg.run_gate_report()
        except Exception as _e:
            print(f"[risk_gate] warning: {_e}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
