#!/usr/bin/env python3
"""
Canyon v9 Step 155 - Backtest Credibility System.

Research-only. No broker connection. No live orders.

This step does not run a new backtest. It audits whether the existing backtest
can be trusted for sizing decisions. The output is deliberately conservative:
missing point-in-time proof, missing out-of-sample validation, or missing
execution realism can only reduce credibility.

Outputs:
  backtest_credibility_scorecard.csv
  backtest_credibility_evidence.csv
  backtest_credibility_blockers.csv
  backtest_credibility_state.json
  backtest_credibility_report.md
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    now_str,
    read_csv_safe,
    read_json_safe,
    source_age,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_SCORECARD = ROOT / "backtest_credibility_scorecard.csv"
OUT_EVIDENCE = ROOT / "backtest_credibility_evidence.csv"
OUT_BLOCKERS = ROOT / "backtest_credibility_blockers.csv"
OUT_STATE = ROOT / "backtest_credibility_state.json"
OUT_REPORT = ROOT / "backtest_credibility_report.md"


CORE_FILES = [
    "backtest_summary.csv",
    "backtest_monthly_perf.csv",
    "backtest_signal_ic.csv",
    "backtest_bias_guard.csv",
    "backtest_walk_forward_proxy.csv",
    "backtest_execution_reality_check.csv",
    "backtest_signal_failure_modes.csv",
    "institutional_backtest_integrity_audit.csv",
    "point_in_time_evidence_ledger.csv",
    "data_truth_state.json",
    "pit_truth_state.json",
    "pit_backtest_readiness_gates.csv",
]


def file_ready(name: str, min_bytes: int = 10) -> bool:
    path = ROOT / name
    return path.exists() and path.stat().st_size > min_bytes


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def score_status(score: float, force_review: bool = False) -> str:
    if not np.isfinite(score):
        return "DATA_GAP"
    if score >= 80 and not force_review:
        return "PASS"
    if score >= 60:
        return "REVIEW"
    if score >= 40:
        return "WEAK"
    return "BLOCKER"


def status_penalty(status: str) -> int:
    order = {
        "PASS": 0,
        "CLEAR": 0,
        "OK": 0,
        "REVIEW": 1,
        "WEAK": 2,
        "DATA_GAP": 3,
        "BLOCKER": 4,
        "BLOCKED": 4,
    }
    return order.get(str(status).upper(), 2)


def parse_pct(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).replace("%", "").strip()
    try:
        return float(text)
    except Exception:
        return np.nan


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def date_span_months(df: pd.DataFrame, *cols: str) -> tuple[int, str, str, list[int]]:
    dates = pd.Series(dtype="datetime64[ns]")
    for col in cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            dates = pd.concat([dates, parsed.dropna()], ignore_index=True)
    if dates.empty:
        return 0, "", "", []
    start = dates.min()
    end = dates.max()
    months = max(1, (end.year - start.year) * 12 + end.month - start.month + 1)
    years = sorted(set(dates.dt.year.astype(int).tolist()))
    return int(months), str(start.date()), str(end.date()), years


def add_row(
    rows: list[dict[str, Any]],
    category: str,
    score: float,
    evidence: str,
    why: str,
    next_action: str,
    source_files: str,
    force_review: bool = False,
    hard_cap: float | None = None,
) -> None:
    adjusted = float(score) if np.isfinite(score) else np.nan
    cap_note = ""
    if hard_cap is not None and np.isfinite(adjusted) and adjusted > hard_cap:
        adjusted = float(hard_cap)
        cap_note = f" Score capped at {hard_cap:.0f} because required proof is missing."
    rows.append({
        "category": category,
        "score_0_100": round(adjusted, 1) if np.isfinite(adjusted) else np.nan,
        "status": score_status(adjusted, force_review=force_review),
        "evidence": f"{evidence}{cap_note}",
        "why_it_matters": why,
        "next_required_action": next_action,
        "source_files": source_files,
        "research_only": True,
    })


def build_evidence_catalog() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in CORE_FILES:
        path = ROOT / name
        exists = path.exists()
        rows.append({
            "source_file": name,
            "exists": exists,
            "size_kb": round(path.stat().st_size / 1024, 2) if exists else 0.0,
            "age": source_age(path) if exists else "missing",
            "role": {
                "backtest_summary.csv": "headline performance",
                "backtest_monthly_perf.csv": "return path / turnover / cost columns",
                "backtest_signal_ic.csv": "signal IC sample depth",
                "backtest_bias_guard.csv": "bias controls",
                "backtest_walk_forward_proxy.csv": "walk-forward proxy",
                "backtest_execution_reality_check.csv": "transaction cost realism",
                "backtest_signal_failure_modes.csv": "signal failure modes",
                "institutional_backtest_integrity_audit.csv": "institutional audit summary",
                "point_in_time_evidence_ledger.csv": "source lineage / PIT proof",
                "data_truth_state.json": "data truth state",
                "pit_truth_state.json": "PIT backtest readiness state",
                "pit_backtest_readiness_gates.csv": "PIT hard readiness gates",
            }.get(name, "supporting evidence"),
        })
    return pd.DataFrame(rows)


def audit_data_lineage(rows: list[dict[str, Any]]) -> None:
    ledger = read_csv_safe(ROOT / "point_in_time_evidence_ledger.csv")
    data_truth = read_json_safe(ROOT / "data_truth_state.json", {})
    pit_truth = read_json_safe(ROOT / "pit_truth_state.json", {})
    pit_gates = read_csv_safe(ROOT / "pit_backtest_readiness_gates.csv")
    if ledger.empty:
        add_row(
            rows,
            "Point-in-time data truth",
            20,
            "No point_in_time_evidence_ledger.csv found.",
            "A backtest is not credible if the system cannot prove what data was visible at the decision time.",
            "Run Step121, then add as_of_time, source_publish_time, and model_read_time to every source table.",
            "point_in_time_evidence_ledger.csv / data_truth_state.json",
            hard_cap=35,
        )
        return

    score = float(pd.to_numeric(ledger.get("lineage_score", pd.Series(dtype=float)), errors="coerce").dropna().mean())
    if not np.isfinite(score):
        score = 40.0
    statuses = ledger.get("pit_lineage_status", pd.Series(dtype=str)).astype(str).str.upper()
    weak = int(statuses.str.contains("WEAK|PARTIAL|MISSING", regex=True, na=False).sum())
    local_proxy = int(ledger.get("source_type", pd.Series(dtype=str)).astype(str).str.contains("yfinance|proxy|local", case=False, na=False).sum())
    vendor_ready = int(ledger.get("source_type", pd.Series(dtype=str)).astype(str).str.contains("vendor|point-in-time|paid", case=False, na=False).sum())
    score = min(score, 75.0 if vendor_ready == 0 else 90.0)
    pit_status = str(pit_truth.get("overall_status", "")).upper()
    hard_gates = int(pit_truth.get("hard_gate_count", 0) or 0)
    if pit_status in {"PIT_BACKTEST_BLOCKED", "PIT_RESEARCH_ONLY"}:
        score = min(score, float(pit_truth.get("pit_truth_score", score) or score), 55.0 if pit_status == "PIT_BACKTEST_BLOCKED" else 68.0)
    if not pit_gates.empty and "status" in pit_gates.columns:
        hard_gates = max(hard_gates, int(pit_gates["status"].astype(str).str.upper().eq("BLOCKER").sum()))
    add_row(
        rows,
        "Point-in-time data truth",
        score,
        f"{len(ledger)} sources audited; weak/partial rows {weak}; local/proxy rows {local_proxy}; vendor-grade PIT rows {vendor_ready}; PIT status {pit_truth.get('overall_status', 'not recorded')}; hard PIT gates {hard_gates}. Data truth state: {data_truth.get('overall_status', 'not recorded')}.",
        "This separates local/proxy research from a true event-time institutional backtest.",
        "Close Step159 hard gates for prices, universe membership, delisted names, corporate actions, fundamentals, and event timestamps before using backtest metrics for sizing.",
        "point_in_time_evidence_ledger.csv / data_truth_state.json / pit_truth_state.json / pit_backtest_readiness_gates.csv",
        force_review=vendor_ready == 0 or hard_gates > 0,
    )


def audit_bias_control(rows: list[dict[str, Any]]) -> None:
    guard = read_csv_safe(ROOT / "backtest_bias_guard.csv")
    state = read_json_safe(ROOT / "backtest_bias_state.json", {})
    institutional = read_csv_safe(ROOT / "institutional_backtest_integrity_audit.csv")
    if guard.empty:
        add_row(
            rows,
            "Bias control",
            20,
            "No Step122 bias guard found.",
            "Look-ahead, survivorship, and delisted-name bias can make a weak strategy look excellent.",
            "Run Step122 and require explicit as-of joins plus historical universe membership.",
            "backtest_bias_guard.csv / backtest_bias_state.json",
            hard_cap=40,
        )
        return
    score = float(pd.to_numeric(guard.get("score", pd.Series(dtype=float)), errors="coerce").dropna().mean())
    if not np.isfinite(score):
        score = 45.0
    weak_rows = int(guard.get("status", pd.Series(dtype=str)).astype(str).str.upper().isin({"WEAK", "BLOCKER", "DATA_GAP"}).sum())
    inst_weak = 0
    if not institutional.empty and "status" in institutional.columns:
        inst_weak = int(institutional["status"].astype(str).str.upper().isin({"WEAK", "BLOCKER", "DATA_GAP"}).sum())
    add_row(
        rows,
        "Bias control",
        min(score, 78.0),
        f"Step122 score {state.get('backtest_bias_guard_score', round(score, 1))}; weak/blocker controls {weak_rows}; institutional audit weak rows {inst_weak}.",
        "The model must not learn from information unavailable at the rebalance timestamp.",
        "Close the weak controls before promoting the backtest from prototype to production evidence.",
        "backtest_bias_guard.csv / backtest_bias_state.json / institutional_backtest_integrity_audit.csv",
        force_review=True,
    )


def audit_oos(rows: list[dict[str, Any]]) -> None:
    walk = read_csv_safe(ROOT / "backtest_walk_forward_proxy.csv")
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    if walk.empty:
        months, start, end, years = date_span_months(monthly, "rebalance_date", "period_end")
        score = 30 if months == 0 else min(55, 25 + months * 0.5)
        add_row(
            rows,
            "Walk-forward / out-of-sample validation",
            score,
            f"No walk-forward table. Monthly history covers {months} months from {start or 'NA'} to {end or 'NA'} across years {years or 'NA'}.",
            "A single in-sample backtest cannot prove the strategy survives new periods.",
            "Add explicit train/test windows, walk-forward folds, frozen parameters, and post-fit out-of-sample results.",
            "backtest_walk_forward_proxy.csv / backtest_monthly_perf.csv",
            hard_cap=60,
        )
        return
    statuses = walk.get("status", pd.Series(dtype=str)).astype(str).str.upper()
    bad = int(statuses.isin({"WEAK", "BLOCKER", "DATA_GAP"}).sum())
    review = int(statuses.isin({"REVIEW"}).sum())
    sharpe = numeric_series(walk, "sharpe_proxy")
    mdd = numeric_series(walk, "max_drawdown")
    score = 65.0
    if len(walk) >= 3:
        score += 8.0
    if not sharpe.empty and float(sharpe.median()) > 0.5:
        score += 8.0
    if not mdd.empty and float(mdd.min()) > -0.25:
        score += 4.0
    score -= 10.0 * bad + 4.0 * review
    sharpe_label = f"{float(sharpe.median()):.2f}" if not sharpe.empty else "NA"
    add_row(
        rows,
        "Walk-forward / out-of-sample validation",
        max(20.0, min(score, 82.0)),
        f"{len(walk)} walk-forward rows; weak/blocker {bad}; review {review}; median Sharpe {sharpe_label}.",
        "A credible model should be stable across non-overlapping historical windows.",
        "Replace proxy walk-forward with frozen-signal train/test folds and include a true holdout period.",
        "backtest_walk_forward_proxy.csv",
        force_review=True,
    )


def audit_execution_realism(rows: list[dict[str, Any]]) -> None:
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    exec_check = read_csv_safe(ROOT / "backtest_execution_reality_check.csv")
    tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    has_cost = not monthly.empty and any(c in monthly.columns for c in ["tc_cost_bps", "transaction_cost_bps", "cost_bps"])
    has_turnover = not monthly.empty and "turnover_pct" in monthly.columns
    score = 20.0
    parts = []
    if has_cost:
        score += 25.0
        parts.append("monthly cost column present")
    if has_turnover:
        turnover = numeric_series(monthly, "turnover_pct")
        med_turnover = float(turnover.median()) if not turnover.empty else np.nan
        score += 15.0
        parts.append(f"median turnover {med_turnover:.1f}%" if np.isfinite(med_turnover) else "turnover column present")
    if not exec_check.empty:
        weak = int(exec_check.get("status", pd.Series(dtype=str)).astype(str).str.upper().isin({"WEAK", "BLOCKER", "DATA_GAP"}).sum())
        score += max(0.0, 20.0 - 6.0 * weak)
        parts.append(f"execution reality rows {len(exec_check)}; weak {weak}")
    if not tca.empty:
        score += 10.0
        parts.append(f"current TCA rows {len(tca)}")
    add_row(
        rows,
        "Execution and transaction cost realism",
        min(score, 78.0),
        "; ".join(parts) if parts else "No explicit cost, turnover, or TCA evidence found.",
        "A strategy can have good paper alpha and still lose after spread, slippage, missed fills, and capacity limits.",
        "Add bid/ask spread history, volume participation, open/close auction assumptions, missed-fill logic, and delayed-entry simulation.",
        "backtest_monthly_perf.csv / backtest_execution_reality_check.csv / institutional_tca_cost_estimates.csv",
        force_review=True,
    )


def audit_signal_validation(rows: list[dict[str, Any]]) -> None:
    signal_ic = read_csv_safe(ROOT / "backtest_signal_ic.csv")
    failure = read_csv_safe(ROOT / "backtest_signal_failure_modes.csv")
    live_ic = read_csv_safe(ROOT / "live_ic_history.csv")
    if signal_ic.empty:
        add_row(
            rows,
            "Signal IC / decay / failure analysis",
            20,
            "No backtest_signal_ic.csv found.",
            "Without IC and decay, the system cannot prove which signals actually predict returns.",
            "Track IC by date, horizon, signal, sector, and regime; then add decay curves and failure tags.",
            "backtest_signal_ic.csv / backtest_signal_failure_modes.csv / live_ic_history.csv",
            hard_cap=40,
        )
        return
    n_obs = pd.to_numeric(signal_ic.get("n_obs", pd.Series(dtype=float)), errors="coerce").fillna(0)
    min_obs = int(n_obs.min()) if len(n_obs) else 0
    weak_signal_count = int(signal_ic.get("status", pd.Series(dtype=str)).astype(str).str.upper().isin({"WEAK", "BLOCKER", "DATA_GAP"}).sum())
    mean_ic = pd.to_numeric(signal_ic.get("mean_ic", pd.Series(dtype=float)), errors="coerce").dropna()
    score = 35.0
    if min_obs >= 30:
        score += 12.0
    if min_obs >= 100:
        score += 10.0
    if not mean_ic.empty and float(mean_ic.median()) > 0.03:
        score += 10.0
    if not failure.empty:
        score += 12.0
    if not live_ic.empty:
        score += 10.0
    score -= weak_signal_count * 3.0
    add_row(
        rows,
        "Signal IC / decay / failure analysis",
        max(20.0, min(score, 82.0)),
        f"{len(signal_ic)} signals; minimum observations {min_obs}; weak signals {weak_signal_count}; failure rows {len(failure)}; live IC rows {len(live_ic)}.",
        "The system needs to know which signals decay, which fail by regime, and which should be down-weighted.",
        "Build Step156 with signal horizon decay, post-signal return buckets, regime failure tags, and live-vs-backtest drift.",
        "backtest_signal_ic.csv / backtest_signal_failure_modes.csv / live_ic_history.csv",
        force_review=True,
    )


def audit_regime_coverage(rows: list[dict[str, Any]]) -> None:
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    if monthly.empty:
        add_row(
            rows,
            "Regime and stress-period coverage",
            20,
            "No backtest_monthly_perf.csv found.",
            "A model must be tested across bull, bear, crash, high-rate, and sideways markets.",
            "Run Step62 and tag every month with market regime before trusting the backtest.",
            "backtest_monthly_perf.csv / regime_price_cache.csv",
            hard_cap=40,
        )
        return
    months, start, end, years = date_span_months(monthly, "rebalance_date", "period_end")
    strategy = numeric_series(monthly, "strategy_ret")
    spy = numeric_series(monthly, "spy_ret")
    neg = int((strategy < 0).sum()) if not strategy.empty else 0
    crash_like = int((spy < -0.07).sum()) if not spy.empty else 0
    score = min(72.0, 25.0 + months * 0.8 + min(len(years), 8) * 2.0 + min(crash_like, 3) * 4.0)
    add_row(
        rows,
        "Regime and stress-period coverage",
        score,
        f"{months} months from {start} to {end}; years {years}; negative strategy months {neg}; SPY crash-like months {crash_like}.",
        "Backtests that miss stress periods usually overstate Sharpe and understate drawdown.",
        "Add explicit 2020, 2022, 2018 Q4, inflation shock, liquidity shock, and sideways regime slices.",
        "backtest_monthly_perf.csv / regime_price_cache.csv",
        force_review=True,
    )


def audit_performance_path(rows: list[dict[str, Any]]) -> None:
    summary = read_csv_safe(ROOT / "backtest_summary.csv")
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    if summary.empty and monthly.empty:
        add_row(
            rows,
            "Performance path reproducibility",
            20,
            "No summary or monthly performance table found.",
            "Headline CAGR/Sharpe are not useful without the path that produced them.",
            "Run Step62 and store date-level holdings, returns, benchmark returns, turnover, and cost assumptions.",
            "backtest_summary.csv / backtest_monthly_perf.csv / backtest_holdings.csv",
            hard_cap=40,
        )
        return
    summary_metrics = []
    if {"metric", "value"}.issubset(summary.columns):
        for _, row in summary.iterrows():
            metric = str(row.get("metric", ""))
            if metric in {"Annualised Sharpe", "Max Drawdown", "Total Return (Strategy)", "Total Alpha vs SPY"}:
                summary_metrics.append(f"{metric}: {row.get('value', '')}")
    holdings_ready = file_ready("backtest_holdings.csv")
    monthly_cols = set(monthly.columns) if not monthly.empty else set()
    required = {"rebalance_date", "strategy_ret", "spy_ret", "tickers"}
    missing = sorted(required - monthly_cols)
    score = 45.0 + (20.0 if not missing else 5.0) + (15.0 if holdings_ready else 0.0)
    add_row(
        rows,
        "Performance path reproducibility",
        min(score, 80.0),
        f"Metrics: {'; '.join(summary_metrics) or 'not parsed'}; monthly missing columns {missing or 'none'}; holdings file {'present' if holdings_ready else 'missing'}.",
        "A trusted backtest must be rerunnable from holdings, weights, returns, and cost assumptions.",
        "Store per-date target weights, actual fills, realized costs, and benchmark membership used at that date.",
        "backtest_summary.csv / backtest_monthly_perf.csv / backtest_holdings.csv",
        force_review=not holdings_ready,
    )


def build_scorecard() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    audit_data_lineage(rows)
    audit_bias_control(rows)
    audit_oos(rows)
    audit_execution_realism(rows)
    audit_signal_validation(rows)
    audit_regime_coverage(rows)
    audit_performance_path(rows)
    return pd.DataFrame(rows)


def build_blockers(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    work = scorecard.copy()
    work["status_rank"] = work["status"].apply(status_penalty)
    blockers = work[(work["status_rank"] >= 1) | (pd.to_numeric(work["score_0_100"], errors="coerce") < 80)].copy()
    if blockers.empty:
        return blockers
    blockers["priority"] = np.where(
        blockers["status"].isin(["BLOCKER", "DATA_GAP"]),
        "P1",
        np.where(blockers["status"].eq("WEAK"), "P2", "P3"),
    )
    blockers = blockers.sort_values(["priority", "score_0_100", "category"]).reset_index(drop=True)
    keep = [
        "priority",
        "category",
        "status",
        "score_0_100",
        "evidence",
        "next_required_action",
        "source_files",
    ]
    return blockers[keep]


def build_state(scorecard: pd.DataFrame, blockers: pd.DataFrame, evidence: pd.DataFrame) -> dict[str, Any]:
    scores = pd.to_numeric(scorecard.get("score_0_100", pd.Series(dtype=float)), errors="coerce").dropna()
    overall = float(scores.mean()) if not scores.empty else 0.0
    status_counts = scorecard.get("status", pd.Series(dtype=str)).astype(str).str.upper().value_counts().to_dict()
    hard_blockers = int(scorecard.get("status", pd.Series(dtype=str)).astype(str).str.upper().isin({"BLOCKER", "DATA_GAP"}).sum())
    weak = int(scorecard.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("WEAK").sum())
    if hard_blockers:
        overall_status = "NOT_CREDIBLE_FOR_SIZING"
    elif overall < 70 or weak:
        overall_status = "PROTOTYPE_ONLY"
    elif overall < 85:
        overall_status = "REVIEW_REQUIRED"
    else:
        overall_status = "RESEARCH_CREDIBLE_NOT_INSTITUTIONAL"
    missing_files = evidence[evidence["exists"] == False]["source_file"].tolist() if not evidence.empty else []
    return {
        "date": today_str(),
        "generated_at": now_str(),
        "overall_credibility_score": round(overall, 1),
        "overall_status": overall_status,
        "category_count": int(len(scorecard)),
        "blocker_rows": int(len(blockers)),
        "hard_blocker_count": hard_blockers,
        "weak_count": weak,
        "status_counts": status_counts,
        "missing_core_files": missing_files,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "This audits the credibility of local/proxy backtests. It does not make the backtest institutional-grade without real point-in-time data, survivorship correction, execution history, and out-of-sample validation.",
    }


def write_report(scorecard: pd.DataFrame, blockers: pd.DataFrame, evidence: pd.DataFrame, state: dict[str, Any]) -> None:
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Overall credibility score: **{state['overall_credibility_score']}/100**",
        f"- Blocker/review rows: **{state['blocker_rows']}**",
        "",
        "This is a credibility audit, not a new backtest. A good Sharpe cannot override missing point-in-time proof, missing survivorship controls, or missing execution realism.",
        "",
        "## Scorecard",
        "",
        df_to_markdown(scorecard, max_rows=30),
        "",
        "## Blockers And Next Actions",
        "",
        df_to_markdown(blockers, max_rows=40),
        "",
        "## Evidence Files",
        "",
        df_to_markdown(evidence, max_rows=40),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 155 - Backtest Credibility System", sections)


def main() -> None:
    scorecard = build_scorecard()
    evidence = build_evidence_catalog()
    blockers = build_blockers(scorecard)
    state = build_state(scorecard, blockers, evidence)

    scorecard.to_csv(OUT_SCORECARD, index=False)
    evidence.to_csv(OUT_EVIDENCE, index=False)
    blockers.to_csv(OUT_BLOCKERS, index=False)
    write_json(OUT_STATE, state)
    write_report(scorecard, blockers, evidence, state)

    print("Canyon v9 Step155 backtest credibility system complete.")
    print(f"Overall: {state['overall_status']} ({state['overall_credibility_score']}/100)")
    print(f"Outputs: {OUT_SCORECARD.name}, {OUT_BLOCKERS.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
