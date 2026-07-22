#!/usr/bin/env python3
"""
Canyon v9 Step 121 - Data Truth Ledger.

Research-only. No broker connection. No live orders.

This step starts the point-in-time discipline that the institutional audit
requires. It does not create true vendor-grade historical point-in-time data.
It builds an evidence ledger: which local file was observed, when it was
observed, what timestamps it carries, whether it has usable as-of columns, and
what still blocks institutional-quality backtests.

Outputs:
  point_in_time_evidence_ledger.csv
  source_lineage_requirements.csv
  universe_membership_snapshot.csv
  data_truth_state.json
  data_truth_report.md
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    load_current_book,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_LEDGER = ROOT / "point_in_time_evidence_ledger.csv"
OUT_REQUIREMENTS = ROOT / "source_lineage_requirements.csv"
OUT_UNIVERSE = ROOT / "universe_membership_snapshot.csv"
OUT_STATE = ROOT / "data_truth_state.json"
OUT_REPORT = ROOT / "data_truth_report.md"

IMPORTANT_FILES = [
    ("sp500_price_cache.csv", "L1 Data / Price", "price_history"),
    ("backtest_price_cache.csv", "L1 Data / Price", "price_history"),
    ("regime_price_cache.csv", "L2 Macro / Regime", "price_history"),
    ("daily_picks_filtered.csv", "Portfolio / Current Book", "current_model_output"),
    ("alpha_scores.csv", "Alpha / Signal", "model_output"),
    ("backtest_monthly_perf.csv", "Backtest", "backtest_output"),
    ("backtest_signal_ic.csv", "Backtest / IC", "backtest_output"),
    ("score_history.csv", "Live Validation", "validation_input"),
    ("live_ic_observation_ledger.csv", "Live Validation", "validation_input"),
    ("live_ic_history.csv", "Live Validation", "validation_output"),
    ("live_ic_realized_summary.csv", "Live Validation", "validation_output"),
    ("live_ic_observation_state.json", "Live Validation", "validation_output"),
    ("stock_news.json", "L5 News", "event_cache"),
    ("event_time_truth_ledger.csv", "L5 Event Time Truth", "event_time_truth"),
    ("event_first_seen_registry.csv", "L5 Event Time Truth", "event_time_truth"),
    ("event_time_quality_audit.csv", "L5 Event Time Truth", "audit_output"),
    ("event_time_truth_state.json", "L5 Event Time Truth", "audit_output"),
    ("event_backtest_admissibility.csv", "L5 Event Backtest Gate", "event_backtest_gate"),
    ("pit_safe_event_signal_panel.csv", "L5 Event Backtest Gate", "event_backtest_gate"),
    ("event_time_repair_queue.csv", "L5 Event Backtest Gate", "audit_output"),
    ("event_backtest_gate_state.json", "L5 Event Backtest Gate", "audit_output"),
    ("event_signal_local_audit_returns.csv", "L5 Event Signal Audit", "local_audit_output"),
    ("event_signal_local_audit_summary.csv", "L5 Event Signal Audit", "local_audit_output"),
    ("event_signal_failure_modes.csv", "L5 Event Signal Audit", "local_audit_output"),
    ("event_signal_audit_state.json", "L5 Event Signal Audit", "audit_output"),
    ("event_signal_reliability_by_bucket.csv", "L5 Event Signal Calibration", "local_calibration_output"),
    ("event_signal_reliability_by_ticker.csv", "L5 Event Signal Calibration", "local_calibration_output"),
    ("event_signal_reliability_adjusted_panel.csv", "L5 Event Signal Calibration", "local_calibration_output"),
    ("event_signal_reliability_watchlist.csv", "L5 Event Signal Calibration", "local_calibration_output"),
    ("event_signal_reliability_state.json", "L5 Event Signal Calibration", "audit_output"),
    ("earnings_calendar.csv", "L5 Earnings", "event_calendar"),
    ("earnings_surprise_scores.csv", "L5 Earnings", "event_signal"),
    ("earnings_revision_scores.csv", "L5 Earnings", "event_signal"),
    ("sec_event_layer.csv", "L5 SEC", "event_signal"),
    ("insider_signal_scores.csv", "L5 Insider", "event_signal"),
    ("options_signals.csv", "L7 Options", "options_signal"),
    ("single_name_risk_budget.csv", "L8 Risk", "risk_output"),
    ("final_risk_gate.csv", "L8 Risk", "risk_output"),
    ("desk_monitor_events.csv", "Desk Monitor", "alert_output"),
    ("horizon_vehicle_matrix.csv", "Decision Router", "derived_research_output"),
    ("horizon_vehicle_summary.csv", "Decision Router", "derived_research_output"),
    ("option_route_clarity_board.csv", "Decision Router", "derived_research_output"),
    ("horizon_vehicle_state.json", "Decision Router", "audit_output"),
    ("institutional_strategy_thesis_board.csv", "Strategy Thesis", "derived_research_output"),
    ("strategy_path_decision_tree.csv", "Strategy Thesis", "derived_research_output"),
    ("strategy_risk_budget_bridge.csv", "Strategy Thesis", "derived_research_output"),
    ("institutional_strategy_sleeve_book.csv", "Strategy Thesis", "derived_research_output"),
    ("institutional_strategy_action_playbook.csv", "Strategy Thesis", "derived_research_output"),
    ("strategy_exposure_overlap.csv", "Strategy Thesis", "derived_research_output"),
    ("institutional_strategy_state.json", "Strategy Thesis", "audit_output"),
    ("institutional_gap_master_scorecard.csv", "Institutional Audit", "audit_output"),
    ("point_in_time_prices.csv", "PIT Store / Price", "pit_seed_or_vendor_input"),
    ("corporate_actions.csv", "PIT Store / Corporate Actions", "pit_seed_or_vendor_input"),
    ("universe_membership_history.csv", "PIT Store / Universe", "pit_seed_or_vendor_input"),
    ("delisted_tickers.csv", "PIT Store / Survivorship", "pit_seed_or_vendor_input"),
    ("pit_fundamentals.csv", "PIT Store / Fundamentals", "pit_seed_or_vendor_input"),
    ("pit_store_build_audit.csv", "PIT Store / Audit", "audit_output"),
    ("pit_store_state.json", "PIT Store / Audit", "audit_output"),
]

TIMESTAMP_HINTS = [
    "date",
    "time",
    "timestamp",
    "asof",
    "as_of",
    "run_time",
    "generated",
    "published",
    "received",
    "filing",
    "earnings",
    "rebalance",
    "period_end",
]


def file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_age_hours(path: Path) -> float:
    if not path.exists():
        return np.inf
    return max(0.0, (time.time() - path.stat().st_mtime) / 3600.0)


def classify_source(file_name: str) -> str:
    low = file_name.lower()
    if "yfinance" in low or "price_cache" in low:
        return "local/yfinance/proxy"
    if "news" in low:
        return "news cache"
    if low in {
        "event_time_truth_ledger.csv",
        "event_first_seen_registry.csv",
        "event_time_quality_audit.csv",
        "event_time_truth_state.json",
        "event_backtest_admissibility.csv",
        "pit_safe_event_signal_panel.csv",
        "event_time_repair_queue.csv",
        "event_backtest_gate_state.json",
        "event_signal_local_audit_returns.csv",
        "event_signal_local_audit_summary.csv",
        "event_signal_failure_modes.csv",
        "event_signal_audit_state.json",
        "event_signal_reliability_by_bucket.csv",
        "event_signal_reliability_by_ticker.csv",
        "event_signal_reliability_adjusted_panel.csv",
        "event_signal_reliability_watchlist.csv",
        "event_signal_reliability_state.json",
    }:
        return "local event time ledger"
    if low in {
        "point_in_time_prices.csv",
        "corporate_actions.csv",
        "universe_membership_history.csv",
        "delisted_tickers.csv",
        "pit_fundamentals.csv",
    }:
        return "local point-in-time seed"
    if "sec" in low or "edgar" in low:
        return "public filing cache"
    if "manual" in low or "ledger" in low:
        return "manual/local ledger"
    if "backtest" in low:
        return "backtest output"
    return "local model output"


def inspect_csv(path: Path) -> dict[str, Any]:
    out = {
        "row_count": 0,
        "column_count": 0,
        "columns": "",
        "timestamp_columns": "",
        "timestamp_quality_pct": 0.0,
        "min_timestamp": "",
        "max_timestamp": "",
    }
    df = read_csv_safe(path)
    if df.empty:
        return out
    out["row_count"] = int(len(df))
    out["column_count"] = int(len(df.columns))
    out["columns"] = ", ".join([str(c) for c in df.columns[:80]])
    ts_cols = [
        c for c in df.columns
        if any(h in str(c).lower().replace("-", "_") for h in TIMESTAMP_HINTS)
    ]
    out["timestamp_columns"] = ", ".join([str(c) for c in ts_cols])
    parsed_all = []
    valid_total = 0
    possible_total = 0
    for c in ts_cols:
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        possible_total += int(df[c].notna().sum())
        valid_total += int(parsed.notna().sum())
        if parsed.notna().any():
            parsed_all.append(parsed.dropna())
    out["timestamp_quality_pct"] = round(100.0 * valid_total / max(possible_total, 1), 1)
    if parsed_all:
        joined = pd.concat(parsed_all)
        out["min_timestamp"] = str(joined.min())
        out["max_timestamp"] = str(joined.max())
    return out


def inspect_json(path: Path) -> dict[str, Any]:
    data = read_json_safe(path, {})
    row_count = 0
    timestamp_hits = 0

    def walk(value: Any) -> None:
        nonlocal row_count, timestamp_hits
        if isinstance(value, dict):
            row_count += 1
            for k, v in value.items():
                if any(h in str(k).lower() for h in TIMESTAMP_HINTS):
                    if str(v).strip() and str(v).strip() not in {"0", "1970-01-01"}:
                        timestamp_hits += 1
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return {
        "row_count": row_count,
        "column_count": np.nan,
        "columns": "json",
        "timestamp_columns": "json timestamp-like keys",
        "timestamp_quality_pct": round(100.0 * timestamp_hits / max(row_count, 1), 1),
        "min_timestamp": "",
        "max_timestamp": "",
    }


def pit_status(row: dict[str, Any]) -> tuple[str, float, str]:
    if not row["exists"]:
        return "MISSING", 0.0, "File missing."
    score = 20.0
    if row["content_hash"]:
        score += 15.0
    if row["timestamp_columns"]:
        score += 25.0
    if row["timestamp_quality_pct"] >= 80:
        score += 20.0
    elif row["timestamp_quality_pct"] >= 30:
        score += 10.0
    if row["source_type"] not in {"local/yfinance/proxy", "local model output"}:
        score += 10.0
    if "asof" in row["timestamp_columns"].lower() or "as_of" in row["timestamp_columns"].lower():
        score += 10.0
    score = min(score, 85.0)
    if score >= 75:
        return "AUDITABLE_LOCAL", score, "Has hash and usable timestamps, but still needs vendor-grade as-of validation."
    if score >= 45:
        return "PARTIAL_LINEAGE", score, "Can be audited locally, but point-in-time proof is incomplete."
    return "WEAK_LINEAGE", score, "Insufficient timestamps or source proof."


def build_ledger() -> pd.DataFrame:
    rows = []
    observed_at = datetime.now().replace(microsecond=0).isoformat()
    for file_name, layer, role in IMPORTANT_FILES:
        path = ROOT / file_name
        exists = path.exists() and path.stat().st_size > 0
        row = {
            "observed_at": observed_at,
            "snapshot_date": today_str(),
            "source_file": file_name,
            "layer": layer,
            "role": role,
            "exists": exists,
            "size_kb": round(path.stat().st_size / 1024.0, 2) if exists else 0.0,
            "file_mtime": datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat() if exists else "",
            "freshness_hours": round(file_age_hours(path), 2) if exists else np.inf,
            "content_hash": file_hash(path) if exists else "",
            "source_type": classify_source(file_name),
            "row_count": 0,
            "column_count": 0,
            "columns": "",
            "timestamp_columns": "",
            "timestamp_quality_pct": 0.0,
            "min_timestamp": "",
            "max_timestamp": "",
        }
        if exists and path.suffix.lower() == ".csv":
            row.update(inspect_csv(path))
        elif exists and path.suffix.lower() == ".json":
            row.update(inspect_json(path))
        status, score, note = pit_status(row)
        row["pit_lineage_status"] = status
        row["lineage_score"] = round(score, 1)
        row["audit_note"] = note
        row["required_next_action"] = required_action_for(row)
        rows.append(row)
    return pd.DataFrame(rows)


def required_action_for(row: dict[str, Any]) -> str:
    if not row["exists"]:
        return "Create this source or mark the dependent module as unavailable."
    if row["pit_lineage_status"] == "AUDITABLE_LOCAL":
        return "Keep hash/version history and add vendor/source ID."
    if not row["timestamp_columns"]:
        return "Add as_of_time, observed_at, source_publish_time, and model_read_time columns."
    if row["timestamp_quality_pct"] < 80:
        return "Fix timestamp parsing and remove blank/1970/default timestamps."
    return "Add point-in-time vendor validation and immutable snapshot storage."


def build_requirements(ledger: pd.DataFrame) -> pd.DataFrame:
    requirements = [
        ("Point-in-time price store", "point_in_time_prices.csv", "BLOCKER", "Daily adjusted/raw prices with as_of_time, vendor_id, adjustment flag, and model_read_time."),
        ("Point-in-time fundamentals", "pit_fundamentals.csv", "BLOCKER", "Fundamental values as originally reported plus restatement/revision timestamps."),
        ("Historical universe membership", "universe_membership_history.csv", "BLOCKER", "Ticker eligibility by date, including index membership, additions, removals, and inactive symbols."),
        ("Delisted ticker table", "delisted_tickers.csv", "BLOCKER", "Dead tickers, delist dates, final price handling, and mapping to replacement symbols when relevant."),
        ("Corporate action trace", "corporate_actions.csv", "BLOCKER", "Splits, dividends, special distributions, ticker changes, and adjustment factors."),
        ("Unified event time ledger", "event_time_truth_ledger.csv", "REVIEW", "Normalized source_publish_time, first_seen_time, model_read_time, source URL, vendor/source ID, and lookahead-risk status for every event."),
        ("Event first-seen registry", "event_first_seen_registry.csv", "REVIEW", "Stable event_id registry that preserves when the local system first saw each event."),
        ("Event backtest admissibility gate", "event_backtest_admissibility.csv", "REVIEW", "Per-event permission table: exclude, current-research-only, local-audit-only, or vendor-ready."),
        ("PIT-safe local event signal panel", "pit_safe_event_signal_panel.csv", "REVIEW", "Event signal panel allowed for local audit only after first_seen_time; not institutional-grade evidence."),
        ("Event signal local audit", "event_signal_local_audit_returns.csv", "REVIEW", "Local-only event reaction audit with source-publish and model-first-seen timing separated."),
        ("Event signal reliability calibration", "event_signal_reliability_by_bucket.csv", "REVIEW", "Local-only reliability scores by event tone, option route, causal link, ticker, and composite bucket."),
        ("News timestamp validation", "stock_news.json", "WEAK", "Vendor news ID, exact publish timestamp, first-seen timestamp, and link/source."),
        ("Earnings exact timing", "earnings_calendar.csv", "REVIEW", "Before/after market flag, exact release time, update time, source, and confirmed/reported flag."),
        ("Immutable run manifest", "point_in_time_evidence_ledger.csv", "REVIEW", "Hash every source consumed by the model for each run."),
    ]
    rows = []
    existing = set(ledger["source_file"].astype(str)) if not ledger.empty and "source_file" in ledger.columns else set()
    for control, file_name, status, requirement in requirements:
        p = ROOT / file_name
        ready = p.exists() and p.stat().st_size > 10
        rows.append({
            "control": control,
            "required_file": file_name,
            "current_status": "LOCAL_SEED" if file_name in existing and ready else ("PRESENT" if ready else status),
            "ready_for_institutional_backtest": bool(ready and status not in {"BLOCKER"}),
            "requirement": requirement,
            "source_file": "point_in_time_evidence_ledger.csv",
        })
    return pd.DataFrame(rows)


def build_universe_snapshot() -> pd.DataFrame:
    book = load_current_book(prefer_filtered=True)
    if book.empty:
        return pd.DataFrame(columns=[
            "snapshot_date", "observed_at", "ticker", "sector", "current_weight_pct",
            "source_file", "membership_type", "institutional_status",
        ])
    work = book.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work["snapshot_date"] = today_str()
    work["observed_at"] = datetime.now().replace(microsecond=0).isoformat()
    work["current_weight_pct"] = pd.to_numeric(work.get("weight", 0), errors="coerce").fillna(0.0) * 100.0
    work["membership_type"] = "current_snapshot_only"
    work["institutional_status"] = "NOT_HISTORICAL_MEMBERSHIP"
    cols = [
        "snapshot_date", "observed_at", "ticker", "sector", "theme",
        "current_weight_pct", "action", "alpha_score", "source_file",
        "membership_type", "institutional_status",
    ]
    return work[[c for c in cols if c in work.columns]].drop_duplicates("ticker")


def write_outputs(ledger: pd.DataFrame, requirements: pd.DataFrame, universe: pd.DataFrame) -> None:
    ledger.to_csv(OUT_LEDGER, index=False)
    requirements.to_csv(OUT_REQUIREMENTS, index=False)
    universe.to_csv(OUT_UNIVERSE, index=False)
    avg_score = float(pd.to_numeric(ledger.get("lineage_score", pd.Series(dtype=float)), errors="coerce").mean()) if not ledger.empty else 0.0
    blocker_count = int(requirements["current_status"].astype(str).str.upper().isin(["BLOCKER"]).sum()) if not requirements.empty else 0
    state = {
        "date": today_str(),
        "lineage_average_score": round(avg_score, 1),
        "source_files_observed": int(len(ledger)),
        "auditable_local_sources": int((ledger.get("pit_lineage_status", pd.Series(dtype=str)) == "AUDITABLE_LOCAL").sum()) if not ledger.empty else 0,
        "institutional_blockers": blocker_count,
        "truth": "This is a local point-in-time evidence ledger, not a paid vendor-grade PIT database.",
        "research_only": True,
        "no_broker_connection": True,
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Product Truth",
        "",
        state["truth"],
        "",
        f"- Average lineage score: {state['lineage_average_score']}",
        f"- Source files observed: {state['source_files_observed']}",
        f"- Auditable local sources: {state['auditable_local_sources']}",
        f"- Institutional blockers: {state['institutional_blockers']}",
        "",
        "## Point-in-Time Evidence Ledger",
        "",
        df_to_markdown(ledger, max_rows=80),
        "",
        "## Source Lineage Requirements",
        "",
        df_to_markdown(requirements, max_rows=40),
        "",
        "## Universe Membership Snapshot",
        "",
        df_to_markdown(universe, max_rows=80),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 121 - Data Truth Ledger", sections)


def main() -> None:
    ledger = build_ledger()
    requirements = build_requirements(ledger)
    universe = build_universe_snapshot()
    write_outputs(ledger, requirements, universe)
    state = read_json_safe(OUT_STATE, {})
    print(f"[step121] wrote {OUT_LEDGER.name}: {len(ledger)} sources")
    print(f"[step121] avg_lineage_score={state.get('lineage_average_score')} blockers={state.get('institutional_blockers')}")
    print(f"[step121] wrote {OUT_REQUIREMENTS.name}, {OUT_UNIVERSE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
