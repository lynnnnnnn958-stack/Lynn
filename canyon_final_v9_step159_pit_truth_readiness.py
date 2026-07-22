#!/usr/bin/env python3
"""
Canyon v9 Step 159 - Point-in-Time Truth Readiness.

Research-only. No broker connection. No live orders.

Step121 records local file lineage. Step159 turns that ledger into a stricter
backtest-readiness gate: which sources are allowed for current research, which
are only local/proxy evidence, and which missing point-in-time controls block
institutional historical claims.

Outputs:
  pit_truth_scorecard.csv
  pit_source_risk_register.csv
  pit_backtest_readiness_gates.csv
  pit_truth_state.json
  pit_truth_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_SCORECARD = ROOT / "pit_truth_scorecard.csv"
OUT_REGISTER = ROOT / "pit_source_risk_register.csv"
OUT_GATES = ROOT / "pit_backtest_readiness_gates.csv"
OUT_STATE = ROOT / "pit_truth_state.json"
OUT_REPORT = ROOT / "pit_truth_report.md"


RAW_SOURCE_HINTS = {
    "price_history",
    "event_cache",
    "event_calendar",
    "event_signal",
    "options_signal",
    "risk_output",
    "validation_output",
}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def has_file(name: str, min_bytes: int = 10) -> bool:
    path = ROOT / name
    return path.exists() and path.stat().st_size >= min_bytes


def status_score(status: str) -> float:
    return {
        "PASS": 95.0,
        "CLEAR": 90.0,
        "REVIEW": 65.0,
        "WEAK": 45.0,
        "BLOCKER": 15.0,
        "MISSING": 5.0,
        "DATA_GAP": 25.0,
    }.get(str(status).upper(), 45.0)


def score_to_status(score: float, blocker: bool = False) -> str:
    if blocker:
        return "BLOCKER"
    if score >= 85:
        return "PASS"
    if score >= 70:
        return "REVIEW"
    if score >= 45:
        return "WEAK"
    return "BLOCKER"


def _lower(value: Any) -> str:
    return str(value or "").lower()


def source_flags(row: pd.Series, requirement_map: dict[str, str]) -> dict[str, Any]:
    file_name = str(row.get("source_file", ""))
    role = str(row.get("role", ""))
    source_type = str(row.get("source_type", ""))
    ts_cols = str(row.get("timestamp_columns", ""))
    lineage_status = str(row.get("pit_lineage_status", "MISSING")).upper()
    timestamp_quality = safe_float(row.get("timestamp_quality_pct"), 0.0)
    has_content_hash = bool(str(row.get("content_hash", "")).strip())
    is_local_proxy = any(x in _lower(source_type) for x in ["local", "proxy", "yfinance"])
    is_model_output = any(x in _lower(role) for x in ["model_output", "current_model_output", "backtest_output", "audit_output"])
    has_asof = any(x in _lower(ts_cols) for x in ["asof", "as_of", "model_read", "observed_at", "source_publish", "published", "filing"])
    has_immutable_hash = has_content_hash and str(row.get("file_mtime", "")).strip() != ""
    missing_vendor_id = is_local_proxy or not any(x in _lower(source_type) for x in ["vendor", "paid", "point-in-time"])

    controls: list[str] = []
    if not bool(row.get("exists", False)):
        controls.append("missing_source_file")
    if not has_asof:
        controls.append("missing_asof_or_publish_time")
    if timestamp_quality < 80:
        controls.append("weak_timestamp_quality")
    if missing_vendor_id:
        controls.append("missing_vendor_or_original_source_id")
    if is_model_output:
        controls.append("model_output_not_raw_point_in_time_input")
    if "price" in _lower(role) and requirement_map.get("point_in_time_prices.csv", "MISSING") not in {"PRESENT", "LOCAL_SEED"}:
        controls.append("no_point_in_time_price_store")
    if "price" in _lower(role) and requirement_map.get("universe_membership_history.csv", "MISSING") not in {"PRESENT", "LOCAL_SEED"}:
        controls.append("survivorship_bias_membership_missing")
    if "price" in _lower(role) and requirement_map.get("delisted_tickers.csv", "MISSING") not in {"PRESENT", "LOCAL_SEED"}:
        controls.append("survivorship_bias_delisted_missing")
    if "price" in _lower(role) and requirement_map.get("corporate_actions.csv", "MISSING") not in {"PRESENT", "LOCAL_SEED"}:
        controls.append("corporate_action_adjustment_trace_missing")
    if any(x in _lower(role) for x in ["event", "news", "earnings"]) and not has_asof:
        controls.append("event_time_lookahead_risk")
    if not has_immutable_hash:
        controls.append("no_immutable_hash_manifest")

    hard = {
        "missing_source_file",
        "no_point_in_time_price_store",
        "survivorship_bias_membership_missing",
        "survivorship_bias_delisted_missing",
        "corporate_action_adjustment_trace_missing",
        "event_time_lookahead_risk",
    }
    hard_count = len([c for c in controls if c in hard])
    if hard_count >= 2 or lineage_status == "MISSING":
        permission = "BACKTEST_BLOCKED_CURRENT_RESEARCH_ONLY"
    elif hard_count == 1 or is_model_output or timestamp_quality < 80 or missing_vendor_id:
        permission = "BACKTEST_REVIEW_CURRENT_RESEARCH_OK"
    else:
        permission = "BACKTEST_INPUT_ALLOWED_LOCAL_AUDIT"

    risk_score = min(100.0, 10.0 * len(controls) + 18.0 * hard_count + (10.0 if missing_vendor_id else 0.0) + (8.0 if is_model_output else 0.0))
    return {
        "has_asof_or_publish_time": has_asof,
        "has_immutable_hash": has_immutable_hash,
        "is_local_or_proxy_source": is_local_proxy,
        "is_model_output": is_model_output,
        "control_gaps": "; ".join(controls) if controls else "none",
        "hard_gap_count": hard_count,
        "pit_risk_score": round(risk_score, 1),
        "pit_use_permission": permission,
    }


def build_register(ledger: pd.DataFrame, requirements: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    req_map = {}
    if not requirements.empty and {"required_file", "current_status"}.issubset(requirements.columns):
        req_map = dict(zip(requirements["required_file"].astype(str), requirements["current_status"].astype(str)))

    rows: list[dict[str, Any]] = []
    for _, row in ledger.iterrows():
        flags = source_flags(row, req_map)
        rows.append({
            "source_file": row.get("source_file", ""),
            "layer": row.get("layer", ""),
            "role": row.get("role", ""),
            "source_type": row.get("source_type", ""),
            "pit_lineage_status": row.get("pit_lineage_status", ""),
            "lineage_score": safe_float(row.get("lineage_score"), 0.0),
            "timestamp_columns": row.get("timestamp_columns", ""),
            "timestamp_quality_pct": safe_float(row.get("timestamp_quality_pct"), 0.0),
            "freshness_hours": safe_float(row.get("freshness_hours"), np.nan),
            **flags,
            "required_next_action": row.get("required_next_action", ""),
            "research_only": True,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["hard_gap_count", "pit_risk_score", "lineage_score"], ascending=[False, False, True]).reset_index(drop=True)


def gate_row(control: str, required_files: list[str], status: str, evidence: str, next_action: str) -> dict[str, Any]:
    return {
        "control": control,
        "required_files": ", ".join(required_files),
        "status": status,
        "score_0_100": status_score(status),
        "evidence": evidence,
        "next_required_action": next_action,
        "research_only": True,
    }


def pit_file_quality(file_name: str) -> dict[str, Any]:
    path = ROOT / file_name
    out: dict[str, Any] = {
        "file": file_name,
        "exists": path.exists() and path.stat().st_size > 10,
        "row_count": 0,
        "vendor_validated": False,
        "local_seed": False,
        "quality_values": "",
    }
    if not out["exists"]:
        return out

    df = read_csv_safe(path)
    if df.empty:
        return out

    out["row_count"] = int(len(df))
    quality_cols = [
        c for c in [
            "pit_quality_status",
            "quality_status",
            "point_in_time_status",
            "source_vendor",
        ] if c in df.columns
    ]
    values: list[str] = []
    for col in quality_cols:
        values.extend(df[col].dropna().astype(str).str.upper().unique().tolist()[:8])
    quality_text = " | ".join(sorted(set(values)))
    out["quality_values"] = quality_text

    ready_col = next((c for c in ["can_support_institutional_backtest", "vendor_pit_validated"] if c in df.columns), None)
    if ready_col:
        ready = df[ready_col].astype(str).str.upper().isin({"TRUE", "1", "YES", "Y"})
        out["vendor_validated"] = bool(ready.any())
    if any(token in quality_text for token in ["VENDOR_PIT_VALIDATED", "VENDOR_GRADE", "POINT_IN_TIME_VALIDATED", "PAID_VENDOR"]):
        out["vendor_validated"] = True
    if any(token in quality_text for token in ["LOCAL_SEED", "LOCAL_", "PROXY", "NOT_VENDOR", "NOT_HISTORICAL"]):
        out["local_seed"] = True

    return out


def gate_status_for_files(files: list[str]) -> tuple[str, str]:
    qualities = [pit_file_quality(file_name) for file_name in files]
    missing = [q["file"] for q in qualities if not q["exists"]]
    if missing:
        return "BLOCKER", f"Missing required files: {', '.join(missing)}."

    detail = "; ".join(
        f"{q['file']} rows={q['row_count']} quality={q['quality_values'] or 'unknown'}"
        for q in qualities
    )
    if all(q["vendor_validated"] for q in qualities):
        return "PASS", f"Vendor/PIT validated files available. {detail}"
    if any(q["local_seed"] for q in qualities):
        return "REVIEW", f"Local seed files exist, but are not vendor-grade PIT proof. {detail}"
    return "REVIEW", f"Files exist, but PIT validation quality is unclear. {detail}"


def build_gates(register: pd.DataFrame, requirements: pd.DataFrame) -> pd.DataFrame:
    blocked_sources = 0
    review_sources = 0
    if not register.empty and "pit_use_permission" in register.columns:
        blocked_sources = int(register["pit_use_permission"].astype(str).str.contains("BLOCKED", na=False).sum())
        review_sources = int(register["pit_use_permission"].astype(str).str.contains("REVIEW", na=False).sum())

    price_status, price_evidence = gate_status_for_files(["point_in_time_prices.csv", "corporate_actions.csv"])
    universe_status, universe_evidence = gate_status_for_files(["universe_membership_history.csv", "delisted_tickers.csv"])
    fundamental_status, fundamental_evidence = gate_status_for_files(["pit_fundamentals.csv"])
    event_status, event_evidence = gate_status_for_files([
        "event_time_truth_ledger.csv",
        "event_first_seen_registry.csv",
        "event_backtest_admissibility.csv",
        "pit_safe_event_signal_panel.csv",
    ])

    gates = [
        gate_row(
            "Historical price truth",
            ["point_in_time_prices.csv", "corporate_actions.csv"],
            price_status,
            f"Backtests need prices as they were known then, plus adjustment trace for splits/dividends. {price_evidence}",
            "Replace local seed files with immutable vendor PIT price store, raw close, adjusted close, source vendor ID, as_of_time, and model_read_time.",
        ),
        gate_row(
            "Universe and survivorship truth",
            ["universe_membership_history.csv", "delisted_tickers.csv"],
            universe_status,
            f"Current S&P/universe membership cannot represent historical membership. {universe_evidence}",
            "Replace local seed files with official membership by date and delisted/dead tickers before trusting long historical performance.",
        ),
        gate_row(
            "Fundamental report-time truth",
            ["pit_fundamentals.csv"],
            fundamental_status,
            f"Fundamental values need report timestamp and restatement/revision timestamp. {fundamental_evidence}",
            "Replace local snapshot with fundamentals as originally seen, including restatement/revision timestamps.",
        ),
        gate_row(
            "News and event timing truth",
            ["event_time_truth_ledger.csv", "event_first_seen_registry.csv", "event_backtest_admissibility.csv", "pit_safe_event_signal_panel.csv", "stock_news.json", "earnings_calendar.csv"],
            "BLOCKER" if event_status == "BLOCKER" or blocked_sources > 0 else ("PASS" if review_sources == 0 and event_status == "PASS" else "REVIEW"),
            f"Event ledger: {event_evidence} Source register has {blocked_sources} blocked sources and {review_sources} review sources.",
            "Replace local event ledger with vendor event tape: exact publish time, first-seen time, timezone, source URL/vendor ID, and immutable raw snapshot for every event/news row.",
        ),
        gate_row(
            "Execution quote truth",
            ["execution_cost_model.csv", "institutional_tca_cost_estimates.csv"],
            "REVIEW" if has_file("execution_cost_model.csv") else "BLOCKER",
            "Execution cost is scenario-modeled, but not built from historical quote/order-book tape.",
            "Add bid/ask snapshots, auction participation assumptions, failed-fill history, and impact calibration.",
        ),
        gate_row(
            "Immutable run reproduction",
            ["point_in_time_evidence_ledger.csv"],
            "PASS" if not register.empty and bool(register.get("has_immutable_hash", pd.Series(dtype=bool)).all()) else "REVIEW",
            "Every run must hash input files so later results can be reproduced.",
            "Persist one run manifest per run, not only the latest overwritten ledger.",
        ),
    ]
    return pd.DataFrame(gates)


def build_scorecard(register: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_count = int(len(register))
    blocked = int(register.get("pit_use_permission", pd.Series(dtype=str)).astype(str).str.contains("BLOCKED", na=False).sum()) if not register.empty else source_count
    review = int(register.get("pit_use_permission", pd.Series(dtype=str)).astype(str).str.contains("REVIEW", na=False).sum()) if not register.empty else 0
    avg_lineage = float(pd.to_numeric(register.get("lineage_score", pd.Series(dtype=float)), errors="coerce").mean()) if not register.empty else 0.0
    timestamp_quality = float(pd.to_numeric(register.get("timestamp_quality_pct", pd.Series(dtype=float)), errors="coerce").mean()) if not register.empty else 0.0
    gate_score = float(pd.to_numeric(gates.get("score_0_100", pd.Series(dtype=float)), errors="coerce").mean()) if not gates.empty else 0.0

    rows.append({
        "category": "Source lineage quality",
        "score_0_100": round(avg_lineage, 1),
        "status": score_to_status(avg_lineage, blocker=source_count == 0),
        "evidence": f"{source_count} files in ledger; {blocked} blocked; {review} review.",
        "next_required_action": "Upgrade blocked source files before using them as historical backtest inputs.",
    })
    rows.append({
        "category": "Event timestamp quality",
        "score_0_100": round(timestamp_quality, 1),
        "status": score_to_status(timestamp_quality),
        "evidence": f"Average parseable timestamp quality {timestamp_quality:.1f}%.",
        "next_required_action": "Add source_publish_time, first_seen_time, and model_read_time to event/news/earnings tables.",
    })
    rows.append({
        "category": "Backtest readiness gates",
        "score_0_100": round(gate_score, 1),
        "status": score_to_status(gate_score, blocker=(gates.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "BLOCKER").any() if not gates.empty else True),
        "evidence": f"{int((gates.get('status', pd.Series(dtype=str)).astype(str).str.upper() == 'BLOCKER').sum()) if not gates.empty else 0} hard blocker gates.",
        "next_required_action": "Do not promote backtest results to sizing evidence until hard blocker gates are closed.",
    })
    rows.append({
        "category": "Current research permission",
        "score_0_100": 80.0 if blocked < source_count else 55.0,
        "status": "REVIEW" if blocked else "PASS",
        "evidence": "The dashboard may still be used for current research because every output is paper/research-only.",
        "next_required_action": "Keep labels explicit: prototype/local/proxy vs point-in-time validated.",
    })
    return pd.DataFrame(rows)


def write_outputs(scorecard: pd.DataFrame, register: pd.DataFrame, gates: pd.DataFrame) -> None:
    scorecard.to_csv(OUT_SCORECARD, index=False)
    register.to_csv(OUT_REGISTER, index=False)
    gates.to_csv(OUT_GATES, index=False)
    score = float(pd.to_numeric(scorecard.get("score_0_100", pd.Series(dtype=float)), errors="coerce").mean()) if not scorecard.empty else 0.0
    hard_gate_count = int(gates.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("BLOCKER").sum()) if not gates.empty else 0
    blocked_sources = int(register.get("pit_use_permission", pd.Series(dtype=str)).astype(str).str.contains("BLOCKED", na=False).sum()) if not register.empty else 0
    if hard_gate_count >= 3 or blocked_sources >= 5:
        overall = "PIT_BACKTEST_BLOCKED"
    elif hard_gate_count > 0 or blocked_sources > 0:
        overall = "PIT_RESEARCH_ONLY"
    elif score >= 80:
        overall = "PIT_RESEARCH_READY"
    else:
        overall = "PIT_REVIEW_REQUIRED"
    state = {
        "date": today_str(),
        "pit_truth_score": round(score, 1),
        "overall_status": overall,
        "hard_gate_count": hard_gate_count,
        "blocked_source_count": blocked_sources,
        "review_source_count": int(register.get("pit_use_permission", pd.Series(dtype=str)).astype(str).str.contains("REVIEW", na=False).sum()) if not register.empty else 0,
        "truth": "This gate separates current research from credible historical backtest evidence. It does not create vendor-grade point-in-time data.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- PIT truth score: **{state['pit_truth_score']}/100**",
        f"- Hard readiness gates: **{state['hard_gate_count']}**",
        f"- Blocked source rows: **{state['blocked_source_count']}**",
        "",
        state["truth"],
        "",
        "## Backtest Readiness Gates",
        "",
        df_to_markdown(gates, max_rows=40),
        "",
        "## Source Risk Register",
        "",
        df_to_markdown(register, max_rows=120),
        "",
        "## Scorecard",
        "",
        df_to_markdown(scorecard, max_rows=40),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 159 - Point-in-Time Truth Readiness", sections)


def main() -> None:
    ledger = read_csv_safe(ROOT / "point_in_time_evidence_ledger.csv")
    requirements = read_csv_safe(ROOT / "source_lineage_requirements.csv")
    register = build_register(ledger, requirements)
    gates = build_gates(register, requirements)
    scorecard = build_scorecard(register, gates)
    write_outputs(scorecard, register, gates)
    state = read_json_safe(OUT_STATE, {})
    print("Canyon v9 Step159 point-in-time truth readiness complete.")
    print(f"Overall: {state.get('overall_status')} ({state.get('pit_truth_score')}/100)")
    print(f"Hard gates: {state.get('hard_gate_count')} | blocked sources: {state.get('blocked_source_count')}")
    print(f"Outputs: {OUT_SCORECARD.name}, {OUT_REGISTER.name}, {OUT_GATES.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
