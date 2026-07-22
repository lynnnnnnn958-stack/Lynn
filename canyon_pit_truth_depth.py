#!/usr/bin/env python3
"""
Canyon v9 Point-in-Time / Data Truth Depth Desk.

This module is a readable truth layer, not another alpha signal.

Purpose:
  - combine PIT evidence, source lineage, event timing, live IC observations,
    bias guards, and backtest credibility into one PM-readable desk
  - separate "usable for today's research" from "credible enough for an
    institutional historical backtest"
  - keep all claims research-only: no broker connection, no live order path,
    no claim of vendor-grade point-in-time data
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


OUT_BOARD = ROOT / "pit_truth_decision_board.csv"
OUT_CARDS = ROOT / "pit_truth_source_cards.csv"
OUT_PERMISSION = ROOT / "pit_truth_backtest_permission.csv"
OUT_SOURCE_GUIDE = ROOT / "pit_truth_source_guide.csv"
OUT_STATE = ROOT / "pit_truth_depth_state.json"
OUT_REPORT = ROOT / "pit_truth_depth_report.md"


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def boolish(value: Any) -> bool:
    raw = clean_text(value).upper()
    return raw in {"TRUE", "1", "YES", "Y", "PASS", "OK"}


def pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 1) if total else 0.0


def score_from(df: pd.DataFrame, label: str, default: float = np.nan) -> float:
    if df.empty or "score_0_100" not in df.columns:
        return default
    category_col = "category" if "category" in df.columns else "control" if "control" in df.columns else ""
    if not category_col:
        return default
    mask = df[category_col].astype(str).str.lower().str.contains(label.lower(), na=False)
    if not mask.any():
        return default
    return safe_float(df.loc[mask, "score_0_100"].iloc[0], default)


def status_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].astype(str).value_counts(dropna=False).to_dict().items()}


def join_counts(counts: dict[str, int], max_items: int = 5) -> str:
    if not counts:
        return "No rows."
    parts = [f"{k}={v}" for k, v in list(counts.items())[:max_items]]
    return "; ".join(parts)


def build_context() -> dict[str, Any]:
    pit_score = read_csv_safe(ROOT / "pit_truth_scorecard.csv")
    source_risk = read_csv_safe(ROOT / "pit_source_risk_register.csv")
    readiness = read_csv_safe(ROOT / "pit_backtest_readiness_gates.csv")
    evidence = read_csv_safe(ROOT / "point_in_time_evidence_ledger.csv")
    event_truth = read_csv_safe(ROOT / "event_time_truth_ledger.csv")
    event_adm = read_csv_safe(ROOT / "event_backtest_admissibility.csv")
    live_ic = read_csv_safe(ROOT / "live_ic_observation_ledger.csv")
    bias = read_csv_safe(ROOT / "backtest_bias_guard.csv")
    seed = read_csv_safe(ROOT / "pit_store_build_audit.csv")
    credibility = read_csv_safe(ROOT / "backtest_credibility_scorecard.csv")
    pit_state = read_json_safe(ROOT / "pit_truth_state.json", {})
    seed_state = read_json_safe(ROOT / "pit_store_state.json", {})
    cred_state = read_json_safe(ROOT / "backtest_credibility_state.json", {})
    data_state = read_json_safe(ROOT / "data_truth_state.json", {})

    event_current_ok = int(event_truth["can_support_current_research"].apply(boolish).sum()) if "can_support_current_research" in event_truth.columns else 0
    event_inst_ok = int(event_truth["can_support_institutional_backtest"].apply(boolish).sum()) if "can_support_institutional_backtest" in event_truth.columns else 0
    event_rows = int(len(event_truth))
    event_adm_inst_ok = int(event_adm["can_support_institutional_backtest"].apply(boolish).sum()) if "can_support_institutional_backtest" in event_adm.columns else 0
    event_adm_rows = int(len(event_adm))
    event_excluded = int(event_adm["event_admissibility"].astype(str).str.upper().eq("EXCLUDE_FROM_HISTORICAL_BACKTEST").sum()) if "event_admissibility" in event_adm.columns else 0
    event_local_audit = int(event_adm["research_permission"].astype(str).str.upper().str.contains("LOCAL_AUDIT", na=False).sum()) if "research_permission" in event_adm.columns else 0

    live_current_ok = int(live_ic["can_support_current_research"].apply(boolish).sum()) if "can_support_current_research" in live_ic.columns else 0
    live_inst_ok = int(live_ic["can_support_institutional_backtest"].apply(boolish).sum()) if "can_support_institutional_backtest" in live_ic.columns else 0
    live_rows = int(len(live_ic))

    source_rows = int(len(source_risk))
    source_review = int(source_risk["pit_use_permission"].astype(str).str.upper().str.contains("REVIEW", na=False).sum()) if "pit_use_permission" in source_risk.columns else 0
    weak_lineage = int(source_risk["pit_lineage_status"].astype(str).str.upper().str.contains("WEAK", na=False).sum()) if "pit_lineage_status" in source_risk.columns else 0
    partial_lineage = int(source_risk["pit_lineage_status"].astype(str).str.upper().str.contains("PARTIAL", na=False).sum()) if "pit_lineage_status" in source_risk.columns else 0
    auditable_local = int(source_risk["pit_lineage_status"].astype(str).str.upper().str.contains("AUDITABLE_LOCAL", na=False).sum()) if "pit_lineage_status" in source_risk.columns else 0
    local_proxy = int(source_risk["is_local_or_proxy_source"].apply(boolish).sum()) if "is_local_or_proxy_source" in source_risk.columns else 0
    hard_gaps = int(pd.to_numeric(source_risk.get("hard_gap_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not source_risk.empty else 0

    seed_inst_ok = int(seed["can_support_institutional_backtest"].apply(boolish).sum()) if "can_support_institutional_backtest" in seed.columns else 0
    seed_rows = int(len(seed))
    bias_block_or_weak = int(bias["status"].astype(str).str.upper().str.contains("BLOCKER|WEAK", na=False).sum()) if "status" in bias.columns else 0
    readiness_review = int(readiness["status"].astype(str).str.upper().str.contains("REVIEW|WEAK|BLOCK", na=False).sum()) if "status" in readiness.columns else 0

    avg_scores = []
    for df, col in [(pit_score, "score_0_100"), (readiness, "score_0_100"), (credibility, "score_0_100")]:
        if not df.empty and col in df.columns:
            avg_scores.append(float(pd.to_numeric(df[col], errors="coerce").mean()))
    if not bias.empty and "score" in bias.columns:
        avg_scores.append(float(pd.to_numeric(bias["score"], errors="coerce").mean()))
    overall_score = round(float(np.nanmean(avg_scores)), 1) if avg_scores else 0.0

    return {
        "pit_score": pit_score,
        "source_risk": source_risk,
        "readiness": readiness,
        "evidence": evidence,
        "event_truth": event_truth,
        "event_adm": event_adm,
        "live_ic": live_ic,
        "bias": bias,
        "seed": seed,
        "credibility": credibility,
        "pit_state": pit_state,
        "seed_state": seed_state,
        "cred_state": cred_state,
        "data_state": data_state,
        "event_current_ok": event_current_ok,
        "event_inst_ok": event_inst_ok,
        "event_rows": event_rows,
        "event_adm_inst_ok": event_adm_inst_ok,
        "event_adm_rows": event_adm_rows,
        "event_excluded": event_excluded,
        "event_local_audit": event_local_audit,
        "live_current_ok": live_current_ok,
        "live_inst_ok": live_inst_ok,
        "live_rows": live_rows,
        "source_rows": source_rows,
        "source_review": source_review,
        "weak_lineage": weak_lineage,
        "partial_lineage": partial_lineage,
        "auditable_local": auditable_local,
        "local_proxy": local_proxy,
        "hard_gaps": hard_gaps,
        "seed_inst_ok": seed_inst_ok,
        "seed_rows": seed_rows,
        "bias_block_or_weak": bias_block_or_weak,
        "readiness_review": readiness_review,
        "overall_score": overall_score,
    }


def build_decision_board(ctx: dict[str, Any]) -> pd.DataFrame:
    pit_score = ctx["pit_score"]
    readiness = ctx["readiness"]
    credibility = ctx["credibility"]
    bias = ctx["bias"]

    pit_truth_score = safe_float(ctx["pit_state"].get("pit_truth_score"), score_from(pit_score, "Source", 0.0))
    backtest_score = safe_float(ctx["cred_state"].get("overall_credibility_score"), score_from(credibility, "Point-in-time", 0.0))
    bias_score = float(pd.to_numeric(bias.get("score", pd.Series(dtype=float)), errors="coerce").mean()) if not bias.empty else np.nan

    rows = [
        {
            "control": "Use for today's research",
            "plain_verdict": "CURRENT_RESEARCH_OK",
            "permission": "Allowed for research notes, watchlists, paper-only review, and source-traced decision pages.",
            "score_0_100": round(max(70.0, pit_truth_score), 1),
            "can_use_today": True,
            "can_use_for_institutional_backtest": False,
            "strongest_evidence": f"{ctx['source_rows']} source files audited; {ctx['event_current_ok']}/{ctx['event_rows']} event rows support current research; {ctx['live_current_ok']}/{ctx['live_rows']} live IC rows support current research.",
            "main_blocker": "Current research is allowed, but this is still local/proxy evidence, not a paid immutable vendor PIT database.",
            "first_repair": "Keep showing source files, observed_at, model_read_time, and source_publish_time on every signal page.",
            "source_files": "point_in_time_evidence_ledger.csv; event_time_truth_ledger.csv; live_ic_observation_ledger.csv",
        },
        {
            "control": "Use for historical backtest sizing",
            "plain_verdict": "PROTOTYPE_ONLY_NOT_INSTITUTIONAL",
            "permission": "Do not use these backtest numbers as production sizing evidence yet.",
            "score_0_100": round(min(pit_truth_score, backtest_score), 1),
            "can_use_today": True,
            "can_use_for_institutional_backtest": False,
            "strongest_evidence": f"PIT status={ctx['pit_state'].get('overall_status', 'NO_DATA')}; backtest status={ctx['cred_state'].get('overall_status', 'NO_DATA')}; {ctx['readiness_review']} readiness gates still review/weak.",
            "main_blocker": "Local/yfinance/proxy data and local PIT seeds do not prove what was knowable at each historical rebalance timestamp.",
            "first_repair": "Replace local seed stores with immutable vendor PIT prices, universe membership, delisted names, corporate actions, fundamentals, and quote history.",
            "source_files": "pit_truth_state.json; pit_backtest_readiness_gates.csv; backtest_credibility_scorecard.csv",
        },
        {
            "control": "Use event/news logic in backtests",
            "plain_verdict": "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL",
            "permission": "Use for local event audit and current research only; do not claim institutional event backtest validity.",
            "score_0_100": score_from(pit_score, "Event", 62.8),
            "can_use_today": True,
            "can_use_for_institutional_backtest": False,
            "strongest_evidence": f"{ctx['event_local_audit']} local-audit event rows; {ctx['event_excluded']} excluded from historical backtest; {ctx['event_adm_inst_ok']}/{ctx['event_adm_rows']} institutional-ready event rows.",
            "main_blocker": "Many event read-through rows still lack source_publish_time, first_seen_time, or model_read_time evidence strong enough for historical claims.",
            "first_repair": "Attach original publisher timestamp, first local seen time, URL/vendor id, and model read time to every event-to-ticker edge.",
            "source_files": "event_time_truth_ledger.csv; event_backtest_admissibility.csv; event_causal_chain_edges.csv",
        },
        {
            "control": "Use live IC observations",
            "plain_verdict": "LOCAL_LIVE_OBSERVATION_ONLY",
            "permission": "Useful for live validation and signal decay; not a vendor-grade historical database.",
            "score_0_100": 70.0 if ctx["live_rows"] else 0.0,
            "can_use_today": True,
            "can_use_for_institutional_backtest": False,
            "strongest_evidence": f"{ctx['live_rows']} local live IC observations; {ctx['live_current_ok']} support current research; {ctx['live_inst_ok']} support institutional backtest.",
            "main_blocker": "Live IC is real local forward observation, but it begins only after this system started recording it.",
            "first_repair": "Keep accumulating forward observations and compare live IC against historical proxy IC by signal, horizon, and regime.",
            "source_files": "live_ic_observation_ledger.csv; live_ic_realized_summary.csv; signal_decay_analysis.csv",
        },
        {
            "control": "Source lineage",
            "plain_verdict": "SOURCE_LINEAGE_REVIEW",
            "permission": "Usable with labels and caveats; not clean enough to silently feed institutional backtests.",
            "score_0_100": score_from(pit_score, "Source lineage", 64.5),
            "can_use_today": True,
            "can_use_for_institutional_backtest": False,
            "strongest_evidence": f"{ctx['auditable_local']} auditable-local files; {ctx['partial_lineage']} partial-lineage files; {ctx['weak_lineage']} weak-lineage files; {ctx['hard_gaps']} hard source gaps.",
            "main_blocker": "Several signal files still miss as-of/publish timestamps or original vendor/source identifiers.",
            "first_repair": "Prioritize highest pit_risk_score rows in pit_source_risk_register.csv.",
            "source_files": "pit_source_risk_register.csv; source_lineage_requirements.csv",
        },
        {
            "control": "Bias and look-ahead guard",
            "plain_verdict": "BIAS_GUARD_REVIEW",
            "permission": "Do not promote historical results without closing weak/blocker controls.",
            "score_0_100": round(safe_float(bias_score, 0.0), 1),
            "can_use_today": True,
            "can_use_for_institutional_backtest": False,
            "strongest_evidence": f"{ctx['bias_block_or_weak']} bias controls are WEAK/BLOCKER; status counts: {join_counts(status_counts(bias, 'status'))}.",
            "main_blocker": "Feature time must be strictly earlier than rebalance time for every historical join.",
            "first_repair": "Add automated feature_time < rebalance_time assertions to every backtest join.",
            "source_files": "backtest_bias_guard.csv; backtest_bias_report.md",
        },
        {
            "control": "PIT seed store",
            "plain_verdict": "LOCAL_SEED_READY_NOT_VENDOR_VALIDATED",
            "permission": "Good schema seed; not vendor-grade truth.",
            "score_0_100": 60.0 if ctx["seed_rows"] else 0.0,
            "can_use_today": True,
            "can_use_for_institutional_backtest": False,
            "strongest_evidence": f"{ctx['seed_rows']} local seed files; {ctx['seed_inst_ok']} institutional-ready files; seed status={ctx['seed_state'].get('overall_status', 'NO_DATA')}.",
            "main_blocker": "Seed files are local cache/model proxy, not immutable vendor point-in-time records.",
            "first_repair": "Swap seed content to vendor PIT feeds while preserving this schema and hash audit.",
            "source_files": "pit_store_build_audit.csv; pit_store_state.json",
        },
    ]
    board = pd.DataFrame(rows)
    order = {
        "PROTOTYPE_ONLY_NOT_INSTITUTIONAL": 0,
        "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL": 1,
        "BIAS_GUARD_REVIEW": 2,
        "SOURCE_LINEAGE_REVIEW": 3,
        "LOCAL_SEED_READY_NOT_VENDOR_VALIDATED": 4,
        "LOCAL_LIVE_OBSERVATION_ONLY": 5,
        "CURRENT_RESEARCH_OK": 6,
    }
    board["_rank"] = board["plain_verdict"].map(order).fillna(9)
    return board.sort_values(["_rank", "control"]).drop(columns=["_rank"])


def build_source_cards(ctx: dict[str, Any]) -> pd.DataFrame:
    source_risk = ctx["source_risk"]
    if source_risk.empty:
        return pd.DataFrame()

    work = source_risk.copy()
    if "pit_risk_score" in work.columns:
        work["pit_risk_score"] = pd.to_numeric(work["pit_risk_score"], errors="coerce").fillna(0.0)
        work = work.sort_values(["pit_risk_score", "source_file"], ascending=[False, True])

    rows: list[dict[str, Any]] = []
    for _, row in work.head(40).iterrows():
        source_file = clean_text(row.get("source_file"))
        lineage = clean_text(row.get("pit_lineage_status"), "NO_LINEAGE")
        permission = clean_text(row.get("pit_use_permission"), "REVIEW")
        risk_score = safe_float(row.get("pit_risk_score"), 0.0)
        timestamp_quality = safe_float(row.get("timestamp_quality_pct"), np.nan)
        can_today = "CURRENT_RESEARCH_OK" in permission.upper() or "CURRENT_RESEARCH" in permission.upper()
        rows.append({
            "source_file": source_file,
            "layer": clean_text(row.get("layer"), "Unknown"),
            "role": clean_text(row.get("role"), "Unknown"),
            "source_type": clean_text(row.get("source_type"), "Unknown"),
            "truth_status": lineage,
            "pit_risk_score": round(risk_score, 1),
            "timestamp_quality_pct": round(timestamp_quality, 1) if np.isfinite(timestamp_quality) else np.nan,
            "permission": permission,
            "can_use_today": can_today,
            "can_use_for_institutional_backtest": False,
            "why_it_matters": plain_source_risk(source_file, lineage, risk_score),
            "required_next_action": clean_text(row.get("required_next_action"), "Add source, timestamp, and lineage proof."),
            "control_gaps": clean_text(row.get("control_gaps"), "No listed gaps."),
        })
    return pd.DataFrame(rows)


def plain_source_risk(source_file: str, lineage: str, risk_score: float) -> str:
    if "news" in source_file.lower() or "event" in source_file.lower():
        return "Event/news timing can create look-ahead bias if publish time and first-seen time are missing."
    if "earnings" in source_file.lower():
        return "Earnings data must prove what was known before the trade date, not after the report."
    if "insider" in source_file.lower():
        return "Insider data needs filing timestamp and source id before it can support a historical test."
    if "price" in source_file.lower():
        return "Price history needs raw/adjusted trace, corporate actions, and as-of snapshots."
    if risk_score >= 60 or "WEAK" in lineage.upper():
        return "This source can support current review, but it is weak historical evidence until timestamps and original source proof improve."
    return "This source is locally auditable, but still not a paid vendor-grade PIT source."


def build_permission_table(ctx: dict[str, Any], board: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in board.iterrows():
        rows.append({
            "area": clean_text(row.get("control")),
            "today_research_permission": "YES" if boolish(row.get("can_use_today")) else "NO",
            "historical_backtest_permission": "NO_VENDOR_GRADE_CLAIM",
            "institutional_backtest_permission": "NO",
            "reason": clean_text(row.get("main_blocker")),
            "first_repair": clean_text(row.get("first_repair")),
            "source_files": clean_text(row.get("source_files")),
        })

    for _, row in ctx["readiness"].iterrows():
        rows.append({
            "area": f"Backtest gate: {clean_text(row.get('control'))}",
            "today_research_permission": "YES",
            "historical_backtest_permission": clean_text(row.get("status"), "REVIEW"),
            "institutional_backtest_permission": "NO" if clean_text(row.get("status")).upper() != "PASS" else "SCHEMA_ONLY_REVIEW",
            "reason": clean_text(row.get("evidence")),
            "first_repair": clean_text(row.get("next_required_action")),
            "source_files": clean_text(row.get("required_files")),
        })
    return pd.DataFrame(rows)


def build_source_guide() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source_area": "PIT evidence ledger",
            "file": "point_in_time_evidence_ledger.csv",
            "what_it_answers": "Which files exist, when they were observed, their content hash, timestamp columns, and lineage score.",
        },
        {
            "source_area": "PIT readiness",
            "file": "pit_truth_scorecard.csv; pit_source_risk_register.csv; pit_backtest_readiness_gates.csv",
            "what_it_answers": "Which data sources can be used today, which require review, and what blocks institutional backtest claims.",
        },
        {
            "source_area": "Event timing truth",
            "file": "event_time_truth_ledger.csv; event_backtest_admissibility.csv",
            "what_it_answers": "Whether news/event rows have publish time, first seen time, model read time, and causal-chain admissibility.",
        },
        {
            "source_area": "Live IC truth",
            "file": "live_ic_observation_ledger.csv; live_ic_realized_summary.csv",
            "what_it_answers": "Real local forward observations after signals are recorded; useful for live validation, not historical vendor PIT.",
        },
        {
            "source_area": "Bias guard",
            "file": "backtest_bias_guard.csv",
            "what_it_answers": "Look-ahead, survivorship, event timing, execution, and signal validity controls.",
        },
        {
            "source_area": "Local seed store",
            "file": "pit_store_build_audit.csv; pit_store_state.json",
            "what_it_answers": "Local schema seed for PIT prices, universe membership, delisted tickers, corporate actions, and fundamentals.",
        },
        {
            "source_area": "Backtest credibility",
            "file": "backtest_credibility_scorecard.csv; backtest_credibility_state.json",
            "what_it_answers": "Whether local/proxy backtests are production evidence or prototype-only evidence.",
        },
    ])


def write_outputs(ctx: dict[str, Any], board: pd.DataFrame, cards: pd.DataFrame, permission: pd.DataFrame, source_guide: pd.DataFrame) -> None:
    board.to_csv(OUT_BOARD, index=False)
    cards.to_csv(OUT_CARDS, index=False)
    permission.to_csv(OUT_PERMISSION, index=False)
    source_guide.to_csv(OUT_SOURCE_GUIDE, index=False)

    vendor_grade_ready = (
        ctx["seed_inst_ok"] > 0
        and ctx["event_inst_ok"] > 0
        and ctx["live_inst_ok"] > 0
        and ctx["readiness_review"] == 0
    )
    status = "DATA_TRUTH_RESEARCH_OK_BACKTEST_PROTOTYPE_ONLY"
    if not ctx["source_rows"] and not ctx["event_rows"] and not ctx["live_rows"]:
        status = "NO_DATA_TRUTH_INPUTS"
    elif vendor_grade_ready:
        status = "DATA_TRUTH_VENDOR_REVIEW_READY"

    state = {
        "date": today_str(),
        "status": status,
        "data_truth_depth_score": ctx["overall_score"],
        "current_research_permission": "CURRENT_RESEARCH_OK",
        "historical_backtest_permission": "PROTOTYPE_ONLY_NOT_INSTITUTIONAL",
        "event_backtest_permission": "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL",
        "live_ic_permission": "LOCAL_LIVE_OBSERVATION_ONLY",
        "vendor_grade_status": "NOT_VENDOR_GRADE_PIT" if not vendor_grade_ready else "VENDOR_REVIEW_READY",
        "source_files_reviewed": ctx["source_rows"],
        "source_files_in_review": ctx["source_review"],
        "weak_lineage_sources": ctx["weak_lineage"],
        "partial_lineage_sources": ctx["partial_lineage"],
        "auditable_local_sources": ctx["auditable_local"],
        "local_or_proxy_sources": ctx["local_proxy"],
        "hard_source_gap_count": ctx["hard_gaps"],
        "event_truth_rows": ctx["event_rows"],
        "event_current_research_rows": ctx["event_current_ok"],
        "event_institutional_backtest_rows": ctx["event_inst_ok"],
        "event_admissibility_rows": ctx["event_adm_rows"],
        "event_admissibility_institutional_rows": ctx["event_adm_inst_ok"],
        "event_historical_backtest_excluded_rows": ctx["event_excluded"],
        "event_local_audit_rows": ctx["event_local_audit"],
        "live_ic_rows": ctx["live_rows"],
        "live_ic_current_research_rows": ctx["live_current_ok"],
        "live_ic_institutional_backtest_rows": ctx["live_inst_ok"],
        "pit_seed_files": ctx["seed_rows"],
        "pit_seed_institutional_files": ctx["seed_inst_ok"],
        "bias_weak_or_blocker_controls": ctx["bias_block_or_weak"],
        "backtest_readiness_review_gates": ctx["readiness_review"],
        "top_source_risk": clean_text(cards.iloc[0].get("source_file")) if not cards.empty else "NO_SOURCE_CARD",
        "top_source_risk_score": safe_float(cards.iloc[0].get("pit_risk_score"), 0.0) if not cards.empty else 0.0,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": (
            "Current research can use local evidence with labels. Historical backtests remain prototype-only "
            "until vendor-grade point-in-time prices, universe membership, delisted names, corporate actions, "
            "fundamentals, event timestamps, and execution quote history are installed."
        ),
        "outputs": {
            "board": OUT_BOARD.name,
            "source_cards": OUT_CARDS.name,
            "permission": OUT_PERMISSION.name,
            "source_guide": OUT_SOURCE_GUIDE.name,
            "report": OUT_REPORT.name,
        },
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Verdict",
        "",
        f"- Status: **{state['status']}**",
        f"- Data truth depth score: **{state['data_truth_depth_score']}/100**",
        f"- Current research permission: **{state['current_research_permission']}**",
        f"- Historical backtest permission: **{state['historical_backtest_permission']}**",
        f"- Vendor-grade status: **{state['vendor_grade_status']}**",
        "",
        "Research-only. No broker connection. No live orders.",
        "",
        "## Permission Board",
        "",
        df_to_markdown(board, max_rows=30),
        "",
        "## Source Risk Cards",
        "",
        df_to_markdown(cards, max_rows=40),
        "",
        "## Backtest Permission",
        "",
        df_to_markdown(permission, max_rows=80),
        "",
        "## Source Guide",
        "",
        df_to_markdown(source_guide, max_rows=20),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Data Truth / Point-in-Time Depth Desk", sections)


def main() -> None:
    ctx = build_context()
    board = build_decision_board(ctx)
    cards = build_source_cards(ctx)
    permission = build_permission_table(ctx, board)
    source_guide = build_source_guide()
    write_outputs(ctx, board, cards, permission, source_guide)
    print("Canyon PIT/data truth depth desk complete.")
    print(f"Status: {read_json_safe(OUT_STATE, {}).get('status')}")
    print(f"Board rows: {len(board)} | source cards: {len(cards)}")
    print(f"Outputs: {OUT_BOARD.name}, {OUT_CARDS.name}, {OUT_PERMISSION.name}, {OUT_STATE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
