#!/usr/bin/env python3
"""
Canyon v9 Step 171 - Event Read-Through Decision Engine.

Research-only. No broker connection. No live orders.

Step129 identifies possible headline targets, Step160 validates causal links,
Step166 calibrates event reliability, and Step170/157 add sector-cycle and
portfolio constraints. Step171 combines those into an event-to-target decision
board:

  headline -> causal role -> beneficiary/vulnerable target -> horizon route
  -> option permission -> risk override -> required proof.

Outputs:
  event_readthrough_decision_board.csv
  event_readthrough_chain_ladder.csv
  event_readthrough_target_ranking.csv
  event_readthrough_event_summary.csv
  event_readthrough_state.json
  event_readthrough_report.md
"""
from __future__ import annotations

import hashlib
from datetime import datetime
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


OUT_BOARD = ROOT / "event_readthrough_decision_board.csv"
OUT_LADDER = ROOT / "event_readthrough_chain_ladder.csv"
OUT_RANKING = ROOT / "event_readthrough_target_ranking.csv"
OUT_EVENT_SUMMARY = ROOT / "event_readthrough_event_summary.csv"
OUT_STATE = ROOT / "event_readthrough_state.json"
OUT_REPORT = ROOT / "event_readthrough_report.md"


TONE_SIGN = {
    "POSITIVE": 1.0,
    "NEGATIVE": -1.0,
    "MIXED": 0.0,
    "NEUTRAL": 0.0,
}

RELATION_STRENGTH = {
    "DIRECT_ENTITY": 1.00,
    "DIRECT_MENTION": 0.92,
    "UPSTREAM_SUPPLIER": 0.78,
    "DOWNSTREAM_BENEFICIARY": 0.72,
    "THEME_PEER": 0.68,
    "SECTOR_PEER": 0.58,
    "CONTEXT_LINK": 0.35,
}

STATUS_MULTIPLIER = {
    "VALIDATED_RESEARCH_LINK": 1.00,
    "SOURCE_SUPPORTED_NEEDS_PRICE_CONFIRMATION": 0.78,
    "HYPOTHESIS_NEEDS_VALIDATION": 0.55,
    "CONTRADICTED_REVIEW_REQUIRED": 0.20,
}

RELIABILITY_MULTIPLIER = {
    "ENOUGH_HISTORY_RELIABLE": 1.00,
    "RELIABLE": 1.00,
    "WATCH_LOCAL_CONTEXT": 0.82,
    "WATCH": 0.82,
    "LOW_SAMPLE_REVIEW": 0.68,
    "UNPROVEN_LOCAL_CONTEXT": 0.62,
    "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL": 0.58,
    "NO_BUCKET_DATA": 0.45,
}

RISK_MULTIPLIER = {
    "CLEAR": 1.00,
    "OK": 0.92,
    "SIZE_DOWN": 0.55,
    "REDUCE_ONLY": 0.25,
    "BLOCK": 0.05,
    "NO_NEW_EXPOSURE": 0.05,
    "UNKNOWN": 0.82,
    "UNKNOWN_NEEDS_DATA": 0.70,
    "NOT_IN_RISK_BOOK_REVIEW": 0.78,
}


def clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def make_event_id(*parts: Any, n: int = 12) -> str:
    text = "|".join(str(p or "") for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def as_upper(value: Any, default: str = "") -> str:
    out = str(value or "").strip().upper()
    return out if out and out != "NAN" else default


def row_index(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty or key not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[key] = out[key].map(clean_ticker)
    out = out[out[key] != ""].copy()
    if out.empty:
        return out
    return out.drop_duplicates(key).set_index(key, drop=False)


def reliability_lookup() -> pd.DataFrame:
    rel = read_csv_safe(ROOT / "event_signal_reliability_adjusted_panel.csv")
    if rel.empty:
        return pd.DataFrame()
    rel = rel.copy()
    for c in ["target_ticker", "source_news_ticker"]:
        if c in rel.columns:
            rel[c] = rel[c].map(clean_ticker)
    if "headline" not in rel.columns or "target_ticker" not in rel.columns:
        return pd.DataFrame()
    rel["_key"] = rel.apply(lambda r: make_event_id(r.get("headline"), r.get("target_ticker"), r.get("source_news_ticker")), axis=1)
    keep = [
        "_key",
        "calibrated_event_score",
        "reliability_multiplier",
        "calibrated_reliability_score",
        "calibrated_reliability_status",
        "ticker_reliability_score",
        "ticker_reliability_status",
        "event_admissibility",
        "research_permission",
        "calibrated_research_action",
        "calibration_note",
    ]
    keep = [c for c in keep if c in rel.columns]
    return rel[keep].drop_duplicates("_key").set_index("_key", drop=False)


def load_base_edges() -> pd.DataFrame:
    edges = read_csv_safe(ROOT / "event_causal_chain_edges.csv")
    if edges.empty:
        supply = read_csv_safe(ROOT / "news_supply_chain_readthrough.csv")
        targets = read_csv_safe(ROOT / "news_impact_targets.csv")
        edges = pd.concat([targets, supply], ignore_index=True, sort=False)
    if edges.empty:
        return pd.DataFrame()
    edges = edges.copy()
    if "target_ticker" not in edges.columns:
        return pd.DataFrame()
    edges["target_ticker"] = edges["target_ticker"].map(clean_ticker)
    edges = edges[edges["target_ticker"] != ""].copy()
    if "source_news_ticker" in edges.columns:
        edges["source_news_ticker"] = edges["source_news_ticker"].map(clean_ticker)
    else:
        edges["source_news_ticker"] = ""
    if "headline" not in edges.columns:
        edges["headline"] = ""
    if "published" not in edges.columns:
        edges["published"] = ""
    if "publisher" not in edges.columns:
        edges["publisher"] = ""
    if "market_tone" not in edges.columns:
        edges["market_tone"] = "NEUTRAL"
    if "relation_layer" not in edges.columns:
        edges["relation_layer"] = edges.get("target_relation", pd.Series(dtype=str)).astype(str).str.upper()
    if "causal_chain_status" not in edges.columns:
        edges["causal_chain_status"] = "HYPOTHESIS_NEEDS_VALIDATION"
    if "causal_confidence_score" not in edges.columns:
        edges["causal_confidence_score"] = 45.0
    edges["_key"] = edges.apply(lambda r: make_event_id(r.get("headline"), r.get("target_ticker"), r.get("source_news_ticker")), axis=1)
    dedupe_cols = [c for c in ["headline", "published", "source_news_ticker", "target_ticker", "relation_layer", "theme", "chain_role"] if c in edges.columns]
    return edges.drop_duplicates(dedupe_cols, keep="first").reset_index(drop=True)


def route_from_timeframe(ticker: str, timeframe: str, tf_idx: pd.DataFrame) -> pd.Series:
    if tf_idx.empty:
        return pd.Series(dtype=object)
    key = f"{ticker}|{timeframe}"
    if key in tf_idx.index:
        return tf_idx.loc[key]
    return pd.Series(dtype=object)


def build_timeframe_index() -> pd.DataFrame:
    tf = read_csv_safe(ROOT / "timeframe_decision_matrix.csv")
    if tf.empty or not {"ticker", "timeframe"}.issubset(tf.columns):
        return pd.DataFrame()
    tf = tf.copy()
    tf["ticker"] = tf["ticker"].map(clean_ticker)
    tf["timeframe"] = tf["timeframe"].astype(str)
    tf["_tf_key"] = tf["ticker"] + "|" + tf["timeframe"]
    return tf.drop_duplicates("_tf_key").set_index("_tf_key", drop=False)


def build_decision_board() -> pd.DataFrame:
    edges = load_base_edges()
    if edges.empty:
        return pd.DataFrame()

    rel_idx = reliability_lookup()
    theme_idx = row_index(read_csv_safe(ROOT / "theme_candidate_enrichment.csv"), "ticker")
    risk_idx = row_index(read_csv_safe(ROOT / "final_risk_gate.csv"), "ticker")
    monitor_idx = row_index(read_csv_safe(ROOT / "desk_monitor_ticker_state.csv"), "ticker")
    subsector_idx = row_index(read_csv_safe(ROOT / "subsector_ticker_cycle_map.csv"), "ticker")
    optimizer_idx = row_index(read_csv_safe(ROOT / "institutional_optimizer_bridge.csv"), "ticker")
    option_idx = row_index(read_csv_safe(ROOT / "option_route_clarity_board.csv"), "ticker")
    tf_idx = build_timeframe_index()

    rows: list[dict[str, Any]] = []
    for _, row in edges.iterrows():
        ticker = clean_ticker(row.get("target_ticker"))
        tone = as_upper(row.get("market_tone"), "NEUTRAL")
        relation = as_upper(row.get("relation_layer"), "CONTEXT_LINK")
        chain_status = as_upper(row.get("causal_chain_status"), "HYPOTHESIS_NEEDS_VALIDATION")
        confidence = safe_float(row.get("causal_confidence_score"), 45.0)
        impact = safe_float(row.get("impact_score"), 0.0)
        vulnerability = safe_float(row.get("total_vulnerability"), 50.0)

        rel = rel_idx.loc[row["_key"]] if not rel_idx.empty and row["_key"] in rel_idx.index else pd.Series(dtype=object)
        theme = theme_idx.loc[ticker] if not theme_idx.empty and ticker in theme_idx.index else pd.Series(dtype=object)
        risk = risk_idx.loc[ticker] if not risk_idx.empty and ticker in risk_idx.index else pd.Series(dtype=object)
        monitor = monitor_idx.loc[ticker] if not monitor_idx.empty and ticker in monitor_idx.index else pd.Series(dtype=object)
        subsector = subsector_idx.loc[ticker] if not subsector_idx.empty and ticker in subsector_idx.index else pd.Series(dtype=object)
        optimizer = optimizer_idx.loc[ticker] if not optimizer_idx.empty and ticker in optimizer_idx.index else pd.Series(dtype=object)
        option = option_idx.loc[ticker] if not option_idx.empty and ticker in option_idx.index else pd.Series(dtype=object)

        raw_risk = as_upper(risk.get("final_risk_action", row.get("final_risk_action", "")), "UNKNOWN")
        final_risk = raw_risk
        if raw_risk == "UNKNOWN" and ticker not in risk_idx.index:
            final_risk = "NOT_IN_RISK_BOOK_REVIEW"
        risk_mult = RISK_MULTIPLIER.get(final_risk, 0.65)
        status_mult = STATUS_MULTIPLIER.get(chain_status, 0.55)
        relation_mult = RELATION_STRENGTH.get(relation, 0.35)
        rel_status = as_upper(rel.get("calibrated_reliability_status"), "NO_RELIABILITY_DATA")
        reliability_mult = safe_float(rel.get("reliability_multiplier"), RELIABILITY_MULTIPLIER.get(rel_status, 0.58))
        if not np.isfinite(reliability_mult):
            reliability_mult = RELIABILITY_MULTIPLIER.get(rel_status, 0.58)

        theme_status = as_upper(theme.get("theme_candidate_status"), "NO_THEME_ENRICHMENT")
        trend_state = as_upper(theme.get("trend_state"), as_upper(monitor.get("price_break_state"), "NO_TREND_DATA"))
        liquidity_status = as_upper(theme.get("liquidity_status"), "NO_LIQUIDITY_DATA")
        monitor_severity = as_upper(monitor.get("max_monitor_severity", monitor.get("max_severity", "")), "NO_MONITOR_DATA")
        sector_phase = str(subsector.get("subsector_cycle_phase", "") or "")
        handoff = str(subsector.get("leadership_handoff_signal", "") or "")

        validation_bonus = {
            "VALIDATED_RESEARCH_LINK": 18.0,
            "SOURCE_SUPPORTED_NEEDS_PRICE_CONFIRMATION": 10.0,
            "HYPOTHESIS_NEEDS_VALIDATION": 2.0,
            "CONTRADICTED_REVIEW_REQUIRED": -28.0,
        }.get(chain_status, 0.0)
        reliability_bonus = {
            "ENOUGH_HISTORY_RELIABLE": 12.0,
            "RELIABLE": 12.0,
            "WATCH_LOCAL_CONTEXT": 8.0,
            "WATCH": 8.0,
            "LOW_SAMPLE_REVIEW": 4.0,
            "UNPROVEN_LOCAL_CONTEXT": 1.0,
            "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL": 0.0,
            "NO_RELIABILITY_DATA": -4.0,
        }.get(rel_status, 0.0)

        signal_score = (
            confidence * 0.42
            + relation_mult * 22.0
            + abs(impact) * 5.0
            + validation_bonus
            + reliability_bonus
        )
        if tone == "POSITIVE":
            signal_score += max(0.0, safe_float(theme.get("attention_score"), 0.0)) * 0.05
            if "UPTREND" in trend_state:
                signal_score += 8.0
            if "handoff" in handoff.lower() or "catch-up" in sector_phase.lower():
                signal_score += 8.0
            if "late-cycle" in sector_phase.lower():
                signal_score -= 9.0
        elif tone == "NEGATIVE":
            signal_score += max(0.0, vulnerability - 45.0) * 0.35
            if final_risk in {"SIZE_DOWN", "REDUCE_ONLY", "BLOCK", "NO_NEW_EXPOSURE"}:
                signal_score += 10.0
            if monitor_severity in {"WARNING", "HIGH", "CRITICAL"}:
                signal_score += 6.0
        else:
            signal_score *= 0.55

        proof_score = signal_score
        allowed_score = proof_score * risk_mult
        if status_mult < 0.35:
            allowed_score = min(allowed_score, 20.0)
        if chain_status == "CONTRADICTED_REVIEW_REQUIRED":
            allowed_score = min(allowed_score, 18.0)
        if final_risk in {"BLOCK", "NO_NEW_EXPOSURE", "REDUCE_ONLY"} and tone == "POSITIVE":
            allowed_score = min(allowed_score, 22.0)
        event_score = round(float(np.clip(allowed_score, 0, 100)), 1)

        if tone == "POSITIVE":
            target_role = "BENEFICIARY"
            if relation == "UPSTREAM_SUPPLIER":
                target_role = "UPSTREAM_BENEFICIARY"
            elif relation == "DOWNSTREAM_BENEFICIARY":
                target_role = "DOWNSTREAM_BENEFICIARY"
            elif relation in {"THEME_PEER", "SECTOR_PEER"}:
                target_role = "PEER_READ_THROUGH"
        elif tone == "NEGATIVE":
            target_role = "VULNERABLE_TARGET"
            if relation in {"THEME_PEER", "SECTOR_PEER"}:
                target_role = "VULNERABLE_PEER"
        else:
            target_role = "CONTEXT_TARGET"

        if event_score >= 72 and tone == "POSITIVE" and final_risk in {"CLEAR", "OK"}:
            decision = "RESEARCH_READY_BENEFICIARY"
        elif event_score >= 55 and tone == "POSITIVE" and chain_status != "CONTRADICTED_REVIEW_REQUIRED":
            decision = "WATCH_FOR_CONFIRMATION"
        elif event_score >= 55 and tone == "NEGATIVE" and chain_status != "CONTRADICTED_REVIEW_REQUIRED":
            decision = "DOWNSIDE_WATCH_OR_HEDGE_RESEARCH"
        elif chain_status == "CONTRADICTED_REVIEW_REQUIRED":
            decision = "DO_NOT_USE_CONTRADICTED_EVENT"
        elif final_risk in {"REDUCE_ONLY", "BLOCK", "NO_NEW_EXPOSURE"}:
            decision = "RISK_BLOCKED_CONTEXT_ONLY"
        else:
            decision = "CONTEXT_ONLY"

        short_tf = route_from_timeframe(ticker, "Short-term", tf_idx)
        medium_tf = route_from_timeframe(ticker, "Medium-term", tf_idx)
        long_tf = route_from_timeframe(ticker, "Long-term", tf_idx)

        option_side = str(option.get("option_side", row.get("option_side", "NONE")) or "NONE")
        option_permission = str(option.get("option_permission", "") or "")
        if tone == "POSITIVE" and decision in {"RESEARCH_READY_BENEFICIARY", "WATCH_FOR_CONFIRMATION"}:
            directional_route = "CALL_RESEARCH_ONLY" if "CALL" in option_side.upper() and "No" not in option_permission else "STOCK_OR_ETF_RESEARCH_ONLY"
        elif tone == "NEGATIVE" and decision in {"DOWNSIDE_WATCH_OR_HEDGE_RESEARCH"}:
            directional_route = "PUT_OR_HEDGE_RESEARCH_ONLY"
        else:
            directional_route = "NO_DIRECTIONAL_TRADE"

        proof_required = []
        if chain_status != "VALIDATED_RESEARCH_LINK":
            proof_required.append("validate causal link and event-time price reaction")
        if rel_status in {"LOW_SAMPLE_REVIEW", "NO_RELIABILITY_DATA"} or "LOCAL_AUDIT" in rel_status:
            proof_required.append("collect more model-seen event samples")
        if final_risk not in {"CLEAR", "OK"}:
            if final_risk == "NOT_IN_RISK_BOOK_REVIEW":
                proof_required.append("create risk-book entry before any paper sizing")
            elif final_risk == "UNKNOWN_NEEDS_DATA":
                proof_required.append("fill missing risk data before any paper sizing")
            else:
                proof_required.append("risk gate must improve or size must stay tiny")
        if liquidity_status not in {"LIQUID", "OK"}:
            proof_required.append("verify liquidity and spread manually")
        if monitor_severity in {"WARNING", "HIGH", "CRITICAL"}:
            proof_required.append("explain active monitor alert")
        if not proof_required:
            proof_required.append("confirm price trigger and news source before paper research")

        rows.append({
            "event_id": row.get("event_id", make_event_id(row.get("headline"), row.get("published"), row.get("publisher"), row.get("source_news_ticker"))),
            "headline": row.get("headline", ""),
            "published": row.get("published", ""),
            "publisher": row.get("publisher", ""),
            "link": row.get("link", ""),
            "source_news_ticker": row.get("source_news_ticker", ""),
            "target_ticker": ticker,
            "target_role": target_role,
            "market_tone": tone,
            "relation_layer": relation,
            "theme": row.get("theme", ""),
            "chain_role": row.get("chain_role", ""),
            "causal_chain_status": chain_status,
            "causal_confidence_score": round(confidence, 1),
            "calibrated_reliability_status": rel_status,
            "event_score": event_score,
            "readthrough_decision": decision,
            "directional_route": directional_route,
            "option_side": option_side,
            "option_permission": option_permission,
            "subsector_cycle_phase": sector_phase,
            "leadership_handoff_signal": handoff,
            "theme_candidate_status": theme_status,
            "trend_state": trend_state,
            "liquidity_status": liquidity_status,
            "final_risk_action": final_risk,
            "monitor_severity": monitor_severity,
            "short_term_action": short_tf.get("action", ""),
            "medium_term_action": medium_tf.get("action", ""),
            "long_term_action": long_tf.get("action", ""),
            "short_term_trigger": short_tf.get("trigger_to_watch", ""),
            "medium_term_trigger": medium_tf.get("trigger_to_watch", ""),
            "long_term_trigger": long_tf.get("trigger_to_watch", ""),
            "optimizer_final_weight_pct": safe_float(optimizer.get("final_optimizer_weight_pct"), np.nan),
            "optimizer_why_not_more": optimizer.get("why_not_more", ""),
            "proof_required": "; ".join(dict.fromkeys(proof_required)),
            "why_this_target": row.get("causal_thesis", row.get("target_reason", "")),
            "source_files": "event_causal_chain_edges.csv; event_signal_reliability_adjusted_panel.csv; theme_candidate_enrichment.csv; final_risk_gate.csv; subsector_ticker_cycle_map.csv; option_route_clarity_board.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["readthrough_decision", "event_score", "causal_confidence_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_chain_ladder(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    stage_map = [
        ("1 Source event", "headline", "market_tone"),
        ("2 Direct entity", "source_news_ticker", "relation_layer"),
        ("3 Industry role", "theme", "chain_role"),
        ("4 Target candidate", "target_ticker", "target_role"),
        ("5 Causal validation", "causal_chain_status", "causal_confidence_score"),
        ("6 Reliability calibration", "calibrated_reliability_status", "event_score"),
        ("7 Risk and vehicle gate", "final_risk_action", "directional_route"),
        ("8 Required proof", "proof_required", "readthrough_decision"),
    ]
    for _, r in board.head(250).iterrows():
        for order, (stage, primary_col, secondary_col) in enumerate(stage_map, 1):
            rows.append({
                "event_id": r.get("event_id", ""),
                "target_ticker": r.get("target_ticker", ""),
                "stage_order": order,
                "stage": stage,
                "primary_evidence": r.get(primary_col, ""),
                "secondary_evidence": r.get(secondary_col, ""),
                "decision": r.get("readthrough_decision", ""),
                "source_files": r.get("source_files", ""),
                "research_only": True,
            })
    return pd.DataFrame(rows)


def build_ranking(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    g = board.copy()
    g["is_positive"] = g["market_tone"].astype(str).eq("POSITIVE")
    g["is_negative"] = g["market_tone"].astype(str).eq("NEGATIVE")
    g["ready_rows"] = g["readthrough_decision"].astype(str).isin(["RESEARCH_READY_BENEFICIARY", "DOWNSIDE_WATCH_OR_HEDGE_RESEARCH"])
    rows: list[dict[str, Any]] = []
    for ticker, grp in g.groupby("target_ticker"):
        top = grp.sort_values("event_score", ascending=False).iloc[0]
        rows.append({
            "target_ticker": ticker,
            "best_event_score": round(float(pd.to_numeric(grp["event_score"], errors="coerce").max()), 1),
            "avg_event_score": round(float(pd.to_numeric(grp["event_score"], errors="coerce").mean()), 1),
            "positive_event_count": int(grp["is_positive"].sum()),
            "negative_event_count": int(grp["is_negative"].sum()),
            "ready_or_downside_rows": int(grp["ready_rows"].sum()),
            "top_decision": top.get("readthrough_decision", ""),
            "top_target_role": top.get("target_role", ""),
            "top_tone": top.get("market_tone", ""),
            "directional_route": top.get("directional_route", ""),
            "final_risk_action": top.get("final_risk_action", ""),
            "subsector_cycle_phase": top.get("subsector_cycle_phase", ""),
            "top_headline": top.get("headline", ""),
            "proof_required": top.get("proof_required", ""),
            "research_only": True,
        })
    return pd.DataFrame(rows).sort_values(["best_event_score", "ready_or_downside_rows"], ascending=[False, False]).reset_index(drop=True)


def build_event_summary(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for event_id, grp in board.groupby("event_id"):
        top = grp.sort_values("event_score", ascending=False).iloc[0]
        pos = grp[grp["market_tone"].astype(str).eq("POSITIVE")]
        neg = grp[grp["market_tone"].astype(str).eq("NEGATIVE")]
        rows.append({
            "event_id": event_id,
            "headline": top.get("headline", ""),
            "published": top.get("published", ""),
            "publisher": top.get("publisher", ""),
            "source_news_ticker": top.get("source_news_ticker", ""),
            "market_tone": top.get("market_tone", ""),
            "target_count": int(grp["target_ticker"].nunique()),
            "best_event_score": top.get("event_score", 0.0),
            "top_beneficiaries": ", ".join(pos.sort_values("event_score", ascending=False)["target_ticker"].head(6).astype(str).tolist()),
            "top_vulnerable_targets": ", ".join(neg.sort_values("event_score", ascending=False)["target_ticker"].head(6).astype(str).tolist()),
            "top_decision": top.get("readthrough_decision", ""),
            "top_required_proof": top.get("proof_required", ""),
            "link": top.get("link", ""),
            "research_only": True,
        })
    return pd.DataFrame(rows).sort_values(["best_event_score", "target_count"], ascending=[False, False]).reset_index(drop=True)


def write_outputs(board: pd.DataFrame, ladder: pd.DataFrame, ranking: pd.DataFrame, summary: pd.DataFrame) -> None:
    board.to_csv(OUT_BOARD, index=False)
    ladder.to_csv(OUT_LADDER, index=False)
    ranking.to_csv(OUT_RANKING, index=False)
    summary.to_csv(OUT_EVENT_SUMMARY, index=False)

    if board.empty:
        overall = "NO_EVENT_READTHROUGH_DATA"
    else:
        usable = int(board["readthrough_decision"].astype(str).isin(["RESEARCH_READY_BENEFICIARY", "DOWNSIDE_WATCH_OR_HEDGE_RESEARCH"]).sum())
        contradicted = int(board["readthrough_decision"].astype(str).eq("DO_NOT_USE_CONTRADICTED_EVENT").sum())
        blocked = int(board["readthrough_decision"].astype(str).str.contains("RISK_BLOCKED", na=False).sum())
        if usable >= 8 and contradicted < len(board) * 0.20:
            overall = "READTHROUGH_RESEARCH_BOARD_ACTIVE"
        elif usable > 0:
            overall = "READTHROUGH_WATCHLIST_ACTIVE"
        elif blocked > 0 or contradicted > 0:
            overall = "READTHROUGH_RISK_REVIEW_REQUIRED"
        else:
            overall = "READTHROUGH_CONTEXT_ONLY"

    state = {
        "date": today_str(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall,
        "decision_rows": int(len(board)),
        "event_count": int(summary["event_id"].nunique()) if not summary.empty and "event_id" in summary.columns else 0,
        "ranked_targets": int(len(ranking)),
        "research_ready_beneficiaries": int((board.get("readthrough_decision", pd.Series(dtype=str)).astype(str) == "RESEARCH_READY_BENEFICIARY").sum()) if not board.empty else 0,
        "watch_for_confirmation_rows": int((board.get("readthrough_decision", pd.Series(dtype=str)).astype(str) == "WATCH_FOR_CONFIRMATION").sum()) if not board.empty else 0,
        "downside_or_hedge_watch": int((board.get("readthrough_decision", pd.Series(dtype=str)).astype(str) == "DOWNSIDE_WATCH_OR_HEDGE_RESEARCH").sum()) if not board.empty else 0,
        "risk_blocked_context_rows": int(board.get("readthrough_decision", pd.Series(dtype=str)).astype(str).str.contains("RISK_BLOCKED", na=False).sum()) if not board.empty else 0,
        "contradicted_event_rows": int((board.get("readthrough_decision", pd.Series(dtype=str)).astype(str) == "DO_NOT_USE_CONTRADICTED_EVENT").sum()) if not board.empty else 0,
        "truth": "This is an event-to-target research router. It turns news into auditable hypotheses, not trades. Risk, source timing, price confirmation, and execution gates still dominate.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Decision rows: **{state['decision_rows']}**",
        f"- Events summarized: **{state['event_count']}**",
        f"- Ranked targets: **{state['ranked_targets']}**",
        f"- Research-ready beneficiaries: **{state['research_ready_beneficiaries']}**",
        f"- Watch-for-confirmation rows: **{state['watch_for_confirmation_rows']}**",
        f"- Downside / hedge watch rows: **{state['downside_or_hedge_watch']}**",
        "",
        state["truth"],
        "",
        "## Event Read-Through Decision Board",
        "",
        df_to_markdown(board, max_rows=80),
        "",
        "## Target Ranking",
        "",
        df_to_markdown(ranking, max_rows=60),
        "",
        "## Event Summary",
        "",
        df_to_markdown(summary, max_rows=60),
        "",
        "## Chain Ladder",
        "",
        df_to_markdown(ladder, max_rows=120),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 171 - Event Read-Through Decision Engine", sections)


def main() -> None:
    board = build_decision_board()
    ladder = build_chain_ladder(board)
    ranking = build_ranking(board)
    summary = build_event_summary(board)
    write_outputs(board, ladder, ranking, summary)
    state = read_json_safe(OUT_STATE, {})
    print("Canyon v9 Step171 event read-through decision engine complete.")
    print(f"Overall: {state.get('overall_status')} | rows={state.get('decision_rows')} targets={state.get('ranked_targets')}")
    print(f"Outputs: {OUT_BOARD.name}, {OUT_RANKING.name}, {OUT_EVENT_SUMMARY.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
