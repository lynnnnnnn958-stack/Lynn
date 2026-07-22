#!/usr/bin/env python3
"""
Canyon v9 Step 164 - Event Backtest Admissibility Gate.

Research-only. No broker connection. No live orders.

Step163 records event timing. Step164 turns that timing evidence into a hard
gate for event/news research: which event-derived signals may be used in a
local audit, which are current-research-only, and which are excluded from any
historical claim because the event time is too weak.

Outputs:
  event_backtest_admissibility.csv
  pit_safe_event_signal_panel.csv
  event_time_repair_queue.csv
  event_backtest_gate_state.json
  event_backtest_admissibility_report.md
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_ADMISSIBILITY = ROOT / "event_backtest_admissibility.csv"
OUT_SAFE_PANEL = ROOT / "pit_safe_event_signal_panel.csv"
OUT_REPAIR_QUEUE = ROOT / "event_time_repair_queue.csv"
OUT_STATE = ROOT / "event_backtest_gate_state.json"
OUT_REPORT = ROOT / "event_backtest_admissibility_report.md"

LOCAL_AUDIT_LABEL = "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL"


def stable_hash(*parts: Any) -> str:
    text = "|".join(str(p or "").strip().lower() for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_url(value: Any) -> str:
    text = str(value or "").strip()
    return text.split("?utm_")[0].split("&utm_")[0]


def row_key(ticker: Any, link: Any, headline: Any) -> str:
    return stable_hash(clean_ticker(ticker), clean_url(link), str(headline or "")[:240])


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def load_event_time_by_key() -> pd.DataFrame:
    ledger = read_csv_safe(ROOT / "event_time_truth_ledger.csv")
    if ledger.empty:
        return pd.DataFrame()
    work = ledger.copy()
    work["ticker"] = work.get("ticker", pd.Series(dtype=str)).map(clean_ticker)
    work["source_url_clean"] = work.get("source_url", pd.Series(dtype=str)).map(clean_url)
    work["event_join_key"] = work.apply(lambda r: row_key(r.get("ticker"), r.get("source_url_clean"), r.get("headline")), axis=1)
    cols = [
        "event_join_key", "event_id", "event_type", "source_file", "ticker",
        "source_publish_time", "first_seen_time", "model_read_time",
        "timestamp_precision", "event_time_score", "event_time_status",
        "lookahead_risk", "vendor_or_source_id", "source_url",
        "pit_quality_status", "can_support_institutional_backtest",
    ]
    work = work[[c for c in cols if c in work.columns]].copy()
    work = work.sort_values(["event_time_score"], ascending=False).drop_duplicates("event_join_key", keep="first")
    return work


def build_edges_panel(event_time: pd.DataFrame) -> pd.DataFrame:
    edges = read_csv_safe(ROOT / "event_causal_chain_edges.csv")
    targets = read_csv_safe(ROOT / "news_impact_targets.csv")

    frames: list[pd.DataFrame] = []
    if not edges.empty:
        e = edges.copy()
        e["target_ticker"] = e.get("target_ticker", pd.Series(dtype=str)).map(clean_ticker)
        e["event_join_key"] = e.apply(lambda r: row_key(r.get("target_ticker"), r.get("link"), r.get("headline")), axis=1)
        e["panel_source"] = "event_causal_chain_edges.csv"
        frames.append(e)
    if not targets.empty:
        t = targets.copy()
        t["target_ticker"] = t.get("target_ticker", pd.Series(dtype=str)).map(clean_ticker)
        t["event_join_key"] = t.apply(lambda r: row_key(r.get("target_ticker"), r.get("link"), r.get("headline")), axis=1)
        t["panel_source"] = "news_impact_targets.csv"
        if "causal_chain_status" not in t.columns:
            t["causal_chain_status"] = "NOT_YET_VALIDATED"
        if "causal_validation_status" not in t.columns:
            t["causal_validation_status"] = "NOT_YET_VALIDATED"
        frames.append(t)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel["_source_rank"] = panel["panel_source"].map({
        "event_causal_chain_edges.csv": 2,
        "news_impact_targets.csv": 1,
    }).fillna(0)
    panel["_has_causal_confidence"] = pd.to_numeric(panel.get("causal_confidence_score", np.nan), errors="coerce").notna().astype(int)
    panel["_has_relation_layer"] = panel.get("relation_layer", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).astype(int)
    panel = (
        panel.sort_values(["_source_rank", "_has_causal_confidence", "_has_relation_layer"], ascending=False)
             .drop_duplicates(["event_join_key", "target_ticker"], keep="first")
             .drop(columns=["_source_rank", "_has_causal_confidence", "_has_relation_layer"], errors="ignore")
             .reset_index(drop=True)
    )
    if not event_time.empty:
        panel = panel.merge(event_time, on="event_join_key", how="left", suffixes=("", "_time"))
    else:
        panel["event_time_status"] = "MISSING_EVENT_TIME_LEDGER"
        panel["event_time_score"] = 0
        panel["lookahead_risk"] = "BLOCKED_LOOKAHEAD_RISK"
    return panel


def admissibility_for(row: pd.Series) -> tuple[str, str, int, str]:
    time_status = str(row.get("event_time_status", "MISSING_EVENT_TIME_LEDGER"))
    lookahead = str(row.get("lookahead_risk", ""))
    causal_status = str(row.get("causal_chain_status", ""))
    validation_status = str(row.get("causal_validation_status", ""))
    time_score = safe_float(row.get("event_time_score"), 0.0)
    has_publish = bool(str(row.get("source_publish_time", "")).strip()) and str(row.get("source_publish_time", "")).lower() not in {"nan", "nat"}
    has_first = bool(str(row.get("first_seen_time", "")).strip()) and str(row.get("first_seen_time", "")).lower() not in {"nan", "nat"}
    has_url = bool(str(row.get("link", row.get("source_url", ""))).strip())
    has_vendor = bool(str(row.get("vendor_or_source_id", "")).strip()) and str(row.get("vendor_or_source_id", "")).lower() not in {"nan", "none"}

    reasons: list[str] = []
    if not has_publish:
        reasons.append("missing source_publish_time")
    if not has_first:
        reasons.append("missing first_seen_time")
    if not has_url:
        reasons.append("missing source URL")
    if time_score < 60:
        reasons.append(f"weak event time score {time_score:.0f}")
    if time_status in {"WEAK_EVENT_TIME", "BLOCKED_EVENT_TIME", "MISSING_EVENT_TIME_LEDGER"}:
        reasons.append(f"time status {time_status}")

    if reasons:
        return (
            "EXCLUDE_FROM_HISTORICAL_BACKTEST",
            "CURRENT_RESEARCH_ONLY",
            20,
            "; ".join(reasons),
        )

    if "CONTRADICTED" in causal_status or "DISAGREES" in validation_status:
        return (
            "LOCAL_AUDIT_REVIEW_ONLY",
            "RESEARCH_REVIEW_ONLY",
            58,
            "event time is usable locally, but causal/price validation disagrees",
        )

    if time_status == "AUDITABLE_LOCAL_EVENT_TIME" and has_vendor and has_url:
        return (
            "LOCAL_EVENT_BACKTEST_OK_VENDOR_REVIEW",
            LOCAL_AUDIT_LABEL,
            75,
            "usable for local audit with explicit local/proxy label; still not institutional vendor PIT",
        )

    return (
        "LOCAL_EVENT_BACKTEST_REVIEW",
        "CURRENT_RESEARCH_OK_LOCAL_AUDIT_REVIEW",
        65,
        "event timing exists but source/vendor proof is incomplete",
    )


def build_admissibility(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=[
            "target_ticker", "headline", "event_admissibility", "research_permission",
            "gate_score", "gate_reason",
        ])
    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
        admissibility, permission, score, reason = admissibility_for(row)
        event_id = row.get("event_id", "")
        if not event_id:
            event_id = stable_hash(row.get("panel_source"), row.get("target_ticker"), row.get("link"), row.get("headline"))
        rows.append({
            "event_id": event_id,
            "edge_id": row.get("edge_id", ""),
            "target_ticker": clean_ticker(row.get("target_ticker")),
            "source_news_ticker": clean_ticker(row.get("source_news_ticker")),
            "target_relation": row.get("target_relation", ""),
            "relation_layer": row.get("relation_layer", ""),
            "theme": row.get("theme", ""),
            "headline": row.get("headline", ""),
            "published": row.get("published", row.get("source_publish_time", "")),
            "source_publish_time": row.get("source_publish_time", ""),
            "first_seen_time": row.get("first_seen_time", ""),
            "model_read_time": row.get("model_read_time", ""),
            "publisher": row.get("publisher", ""),
            "link": row.get("link", row.get("source_url", "")),
            "market_tone": row.get("market_tone", ""),
            "impact_score": row.get("impact_score", np.nan),
            "news_logic": row.get("news_logic", ""),
            "suggested_research_route": row.get("suggested_research_route", ""),
            "option_side": row.get("option_side", ""),
            "event_time_status": row.get("event_time_status", "MISSING_EVENT_TIME_LEDGER"),
            "event_time_score": row.get("event_time_score", 0),
            "lookahead_risk": row.get("lookahead_risk", "BLOCKED_LOOKAHEAD_RISK"),
            "causal_chain_status": row.get("causal_chain_status", "NOT_YET_VALIDATED"),
            "causal_validation_status": row.get("causal_validation_status", "NOT_YET_VALIDATED"),
            "causal_confidence_score": row.get("causal_confidence_score", np.nan),
            "event_admissibility": admissibility,
            "research_permission": permission,
            "gate_score": score,
            "gate_reason": reason,
            "can_enter_local_event_backtest": admissibility in {
                "LOCAL_EVENT_BACKTEST_OK_VENDOR_REVIEW",
                "LOCAL_EVENT_BACKTEST_REVIEW",
                "LOCAL_AUDIT_REVIEW_ONLY",
            },
            "can_support_institutional_backtest": False,
            "pit_quality_status": "LOCAL_EVENT_BACKTEST_GATE_NOT_VENDOR_PIT",
            "panel_source": row.get("panel_source", ""),
            "source_file": "event_time_truth_ledger.csv / event_causal_chain_edges.csv / news_impact_targets.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    rank = {
        "EXCLUDE_FROM_HISTORICAL_BACKTEST": 0,
        "LOCAL_AUDIT_REVIEW_ONLY": 1,
        "LOCAL_EVENT_BACKTEST_REVIEW": 2,
        "LOCAL_EVENT_BACKTEST_OK_VENDOR_REVIEW": 3,
    }
    out["_rank"] = out["event_admissibility"].map(rank).fillna(0)
    return out.sort_values(["_rank", "gate_score", "target_ticker"], ascending=[True, True, True]).drop(columns=["_rank"]).reset_index(drop=True)


def build_safe_panel(admissibility: pd.DataFrame) -> pd.DataFrame:
    if admissibility.empty:
        return pd.DataFrame()
    ok = admissibility[admissibility["can_enter_local_event_backtest"]].copy()
    if ok.empty:
        return ok
    keep = [
        "event_id", "target_ticker", "source_news_ticker", "target_relation",
        "relation_layer", "theme", "headline", "source_publish_time",
        "first_seen_time", "market_tone", "impact_score", "suggested_research_route",
        "option_side", "event_time_status", "event_admissibility",
        "research_permission", "causal_chain_status", "causal_validation_status",
        "causal_confidence_score", "gate_score", "gate_reason", "link",
        "source_file", "research_only",
    ]
    out = ok[[c for c in keep if c in ok.columns]].copy()
    out["trade_allowed_after_time"] = out["first_seen_time"]
    out["truth_label"] = LOCAL_AUDIT_LABEL
    out["pit_quality_status"] = "LOCAL_EVENT_BACKTEST_GATE_NOT_VENDOR_PIT"
    out["can_support_institutional_backtest"] = False
    return out.sort_values(["gate_score", "target_ticker"], ascending=[False, True]).reset_index(drop=True)


def build_repair_queue(admissibility: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not audit.empty:
        for _, row in audit.iterrows():
            weak = int(safe_float(row.get("weak_rows"), 0))
            blocked = int(safe_float(row.get("blocked_rows"), 0))
            missing_publish = int(safe_float(row.get("missing_publish_time_rows"), 0))
            if weak or blocked or missing_publish:
                rows.append({
                    "priority": "P1" if blocked or missing_publish >= 50 else "P2",
                    "scope": "source_file",
                    "source_file": row.get("source_file", ""),
                    "ticker": "",
                    "headline": "",
                    "issue": f"weak={weak}; blocked={blocked}; missing_publish={missing_publish}",
                    "required_next_action": row.get("required_next_action", "Add exact event time and vendor/source ID."),
                })
    if not admissibility.empty:
        bad = admissibility[admissibility["event_admissibility"] == "EXCLUDE_FROM_HISTORICAL_BACKTEST"].head(200)
        for _, row in bad.iterrows():
            rows.append({
                "priority": "P1",
                "scope": "event_row",
                "source_file": row.get("panel_source", ""),
                "ticker": row.get("target_ticker", ""),
                "headline": row.get("headline", ""),
                "issue": row.get("gate_reason", ""),
                "required_next_action": "Add source_publish_time, first_seen_time, source URL, and vendor/source ID before this event can enter historical testing.",
            })
    if not rows:
        return pd.DataFrame(columns=["priority", "scope", "source_file", "ticker", "headline", "issue", "required_next_action"])
    return pd.DataFrame(rows).drop_duplicates(["scope", "source_file", "ticker", "headline", "issue"]).reset_index(drop=True)


def build_state(admissibility: pd.DataFrame, safe_panel: pd.DataFrame, repair: pd.DataFrame) -> dict[str, Any]:
    total = int(len(admissibility))
    local_ok = int(admissibility.get("can_enter_local_event_backtest", pd.Series(dtype=bool)).sum()) if not admissibility.empty else 0
    excluded = int((admissibility.get("event_admissibility", pd.Series(dtype=str)) == "EXCLUDE_FROM_HISTORICAL_BACKTEST").sum()) if not admissibility.empty else 0
    review_only = int(admissibility.get("event_admissibility", pd.Series(dtype=str)).astype(str).str.contains("REVIEW", na=False).sum()) if not admissibility.empty else 0
    if total == 0:
        status = "NO_EVENT_ADMISSIBILITY_DATA"
    elif excluded > total * 0.35:
        status = "EVENT_BACKTEST_REPAIR_REQUIRED"
    elif excluded > 0 or review_only > 0:
        status = "EVENT_BACKTEST_REVIEW_REQUIRED"
    else:
        status = "EVENT_BACKTEST_LOCAL_READY_NOT_INSTITUTIONAL"
    return {
        "date": today_str(),
        "overall_status": status,
        "admissibility_rows": total,
        "local_event_backtest_rows": local_ok,
        "excluded_from_historical_backtest": excluded,
        "review_rows": review_only,
        "repair_queue_rows": int(len(repair)),
        "institutional_ready_rows": 0,
        "truth": "This gate can allow local event-signal audits, but every output remains research-only and not institutional-grade without vendor PIT event tape.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def write_outputs(admissibility: pd.DataFrame, safe_panel: pd.DataFrame, repair: pd.DataFrame, state: dict[str, Any]) -> None:
    admissibility.to_csv(OUT_ADMISSIBILITY, index=False)
    safe_panel.to_csv(OUT_SAFE_PANEL, index=False)
    repair.to_csv(OUT_REPAIR_QUEUE, index=False)
    write_json(OUT_STATE, state)
    summary = admissibility.get("event_admissibility", pd.Series(dtype=str)).value_counts().rename_axis("event_admissibility").reset_index(name="rows") if not admissibility.empty else pd.DataFrame()
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Admissibility rows: **{state['admissibility_rows']}**",
        f"- Local event backtest rows: **{state['local_event_backtest_rows']}**",
        f"- Excluded from historical backtest: **{state['excluded_from_historical_backtest']}**",
        f"- Repair queue rows: **{state['repair_queue_rows']}**",
        "",
        state["truth"],
        "",
        "## Admissibility Summary",
        "",
        df_to_markdown(summary, max_rows=20),
        "",
        "## Repair Queue",
        "",
        df_to_markdown(repair.head(80), max_rows=80),
        "",
        "## Local Audit Panel Preview",
        "",
        df_to_markdown(safe_panel.head(80), max_rows=80),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 164 - Event Backtest Admissibility", sections)


def main() -> None:
    event_time = load_event_time_by_key()
    panel = build_edges_panel(event_time)
    admissibility = build_admissibility(panel)
    safe_panel = build_safe_panel(admissibility)
    audit = read_csv_safe(ROOT / "event_time_quality_audit.csv")
    repair = build_repair_queue(admissibility, audit)
    state = build_state(admissibility, safe_panel, repair)
    write_outputs(admissibility, safe_panel, repair, state)
    print("Canyon v9 Step164 event backtest admissibility gate complete.")
    print(f"Overall: {state.get('overall_status')} | local rows: {state.get('local_event_backtest_rows')} | excluded: {state.get('excluded_from_historical_backtest')}")
    print(f"Outputs: {OUT_ADMISSIBILITY.name}, {OUT_SAFE_PANEL.name}, {OUT_REPAIR_QUEUE.name}")


if __name__ == "__main__":
    main()
