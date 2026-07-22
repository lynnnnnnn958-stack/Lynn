#!/usr/bin/env python3
"""
Canyon v9 Step 205 - PM Evidence Review Triage.

Research-only. No broker connection. No live orders.

Step204 gives the human a full evidence acceptance queue. This step makes that
queue usable: it ranks tickers, identifies the first few evidence fields to
review, separates internal evidence from outside-source checks, and writes
plain-English review packets.

Outputs:
  pm_evidence_review_triage_state.json
  pm_evidence_review_priority_queue.csv
  pm_evidence_review_field_plan.csv
  pm_evidence_review_packet_cards.csv
  pm_evidence_review_source_ladder.csv
  pm_evidence_review_triage_report.md
"""
from __future__ import annotations

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


OUT_STATE = ROOT / "pm_evidence_review_triage_state.json"
OUT_PRIORITY = ROOT / "pm_evidence_review_priority_queue.csv"
OUT_FIELD_PLAN = ROOT / "pm_evidence_review_field_plan.csv"
OUT_PACKET_CARDS = ROOT / "pm_evidence_review_packet_cards.csv"
OUT_SOURCE_LADDER = ROOT / "pm_evidence_review_source_ladder.csv"
OUT_REPORT = ROOT / "pm_evidence_review_triage_report.md"


FIELD_GROUPS = {
    "thesis_plain": "Story",
    "earnings_date": "Event risk",
    "expected_event_move_pct": "Event risk",
    "event_size_policy": "Event risk",
    "liquidity_snapshot_date": "Trading cost",
    "bid_ask_spread_bps": "Trading cost",
    "avg_daily_dollar_volume_check": "Trading cost",
    "sector_confirmed": "Crowding",
    "crowding_check": "Crowding",
    "news_proof_note": "News proof",
    "execution_proof_note": "Trading cost",
    "paper_stop_pct": "Risk control",
    "option_route_requested": "Options guardrail",
    "decision_note": "PM note",
    "last_updated": "Freshness",
}

OUTSIDE_CHECK_FIELDS = {
    "earnings_date",
    "expected_event_move_pct",
    "liquidity_snapshot_date",
    "bid_ask_spread_bps",
    "news_proof_note",
    "execution_proof_note",
}

FIRST_PASS_FIELDS = {
    "thesis_plain",
    "event_size_policy",
    "avg_daily_dollar_volume_check",
    "sector_confirmed",
    "crowding_check",
    "paper_stop_pct",
    "option_route_requested",
    "decision_note",
    "last_updated",
}


def as_text(value: Any, default: str = "") -> str:
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
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def short(value: Any, limit: int = 320) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def one_by_ticker(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "ticker" not in df.columns:
        return {}
    work = df.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby("ticker", sort=False)}


def field_group(field: Any) -> str:
    return FIELD_GROUPS.get(as_text(field, ""), "Other")


def source_kind(field: Any, source_files: Any) -> str:
    field = as_text(field, "")
    source = as_text(source_files, "").lower()
    if field in OUTSIDE_CHECK_FIELDS:
        return "Needs outside check"
    if "canyon risk policy" in source or "step203 policy" in source:
        return "Internal policy"
    if "risk_book_seed_metric_detail" in source or "sector_map" in source:
        return "Local model file"
    if "event_readthrough" in source or "event_causal" in source:
        return "Local news map"
    return "Local system file"


def field_priority_score(row: pd.Series, review_row: pd.Series, rank_row: pd.Series) -> float:
    field = as_text(row.get("field_name"), "")
    score = 0.0
    group = field_group(field)
    if group in {"Event risk", "Trading cost", "News proof"}:
        score += 30
    elif group in {"Risk control", "Crowding"}:
        score += 22
    elif group == "Story":
        score += 16
    elif group in {"Options guardrail", "PM note"}:
        score += 12

    confidence = as_text(row.get("confidence"), "").lower()
    if confidence == "low":
        score += 18
    elif confidence == "medium":
        score += 10
    elif confidence == "high":
        score += 3

    if field in OUTSIDE_CHECK_FIELDS:
        score += 15
    if as_text(row.get("acceptance_status"), "") == "Needs human decision":
        score += 12

    missing = as_text(review_row.get("missing_fields_plain"), "").lower()
    friendly_needles = {
        "earnings_date": ["next earnings date"],
        "expected_event_move_pct": ["expected earnings/event move"],
        "event_size_policy": ["event size policy"],
        "liquidity_snapshot_date": ["liquidity snapshot date"],
        "bid_ask_spread_bps": ["bid/ask spread"],
        "avg_daily_dollar_volume_check": ["dollar-volume check"],
        "sector_confirmed": ["sector/theme confirmation"],
        "crowding_check": ["crowding check"],
        "news_proof_note": ["news proof note"],
        "execution_proof_note": ["execution proof note"],
    }
    if any(needle in missing for needle in friendly_needles.get(field, [])):
        score += 20

    risk_level = as_text(rank_row.get("risk_level"), as_text(review_row.get("risk_level"), "")).lower()
    if risk_level == "high":
        score += 8
    elif risk_level == "medium":
        score += 4
    return round(score, 1)


def review_action(field: str, source: str, confidence: str) -> str:
    kind = source_kind(field, source)
    if kind == "Needs outside check":
        if field == "earnings_date":
            return "Check the next earnings date from a calendar before accepting."
        if field == "expected_event_move_pct":
            return "Check the option-implied event move or write why the fallback is acceptable."
        if field == "bid_ask_spread_bps":
            return "Check a current quote or spread source before accepting."
        if field == "liquidity_snapshot_date":
            return "Confirm the liquidity snapshot is current enough for review."
        if field == "news_proof_note":
            return "Open the headline source and check whether price/volume reacted after the news."
        if field == "execution_proof_note":
            return "Confirm the cost note with spread, volume, or execution desk data."
    if field == "option_route_requested":
        return "Accept only if the value is NO. This gate must not unlock calls or puts."
    if field == "paper_stop_pct":
        return "Check the stop is no wider than the system risk seed stop."
    if field in FIRST_PASS_FIELDS and confidence in {"High", "Medium"}:
        return "Can be first-pass accepted if the sentence is true and clear."
    return "Read the source and decide whether to accept, reject, or require outside proof."


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    acceptance = read_csv_safe(ROOT / "pm_review_evidence_acceptance_input.csv")
    acceptance_status = read_csv_safe(ROOT / "pm_review_evidence_acceptance_status.csv")
    review_status = read_csv_safe(ROOT / "risk_seed_pm_review_status.csv")
    rank = read_csv_safe(ROOT / "risk_seed_approval_rank.csv")
    blockers = read_csv_safe(ROOT / "risk_seed_blocker_matrix.csv")

    if acceptance.empty:
        empty = pd.DataFrame()
        state = {
            "date": today_str(),
            "status": "NO_EVIDENCE_ACCEPTANCE_QUEUE",
            "plain_answer": "Step205 needs pm_review_evidence_acceptance_input.csv from Step204 first.",
            "ticker_count": 0,
            "field_review_count": 0,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return empty, empty, empty, empty, state

    review_map = one_by_ticker(review_status)
    rank_map = one_by_ticker(rank)
    accept_map = one_by_ticker(acceptance_status)
    blocker_map: dict[str, list[str]] = {}
    if not blockers.empty and "ticker" in blockers.columns:
        b = blockers.copy()
        b["ticker"] = b["ticker"].apply(clean_ticker)
        for ticker, grp in b.groupby("ticker", sort=False):
            blocker_map[ticker] = [as_text(x, "") for x in grp.get("blocker_type", pd.Series(dtype=str)) if as_text(x, "")]

    field_rows: list[dict[str, Any]] = []
    work = acceptance.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()

    for _, row in work.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        review_row = review_map.get(ticker, pd.Series(dtype=object))
        rank_row = rank_map.get(ticker, pd.Series(dtype=object))
        field = as_text(row.get("field_name"), "")
        source = as_text(row.get("source_files"), "")
        confidence = as_text(row.get("confidence"), "")
        source_label = source_kind(field, source)
        priority_score = field_priority_score(row, review_row, rank_row)
        field_rows.append({
            "ticker": ticker,
            "field_name": field,
            "field_group": field_group(field),
            "review_priority_score": priority_score,
            "review_priority": "High" if priority_score >= 65 else "Medium" if priority_score >= 40 else "Low",
            "source_check_type": source_label,
            "acceptance_status": as_text(row.get("acceptance_status"), "Needs human decision"),
            "confidence": confidence,
            "suggested_value": short(row.get("suggested_value"), 260),
            "what_to_do": review_action(field, source, confidence),
            "why_it_matters": short(row.get("how_to_decide"), 260),
            "source_files": source,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    field_plan = pd.DataFrame(field_rows)
    if not field_plan.empty:
        field_plan = field_plan.sort_values(["review_priority_score", "ticker", "field_name"], ascending=[False, True, True]).reset_index(drop=True)

    priority_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    for ticker, grp in field_plan.groupby("ticker", sort=False):
        review_row = review_map.get(ticker, pd.Series(dtype=object))
        rank_row = rank_map.get(ticker, pd.Series(dtype=object))
        accept_row = accept_map.get(ticker, pd.Series(dtype=object))
        approval_score = safe_float(rank_row.get("approval_score_0_100"), 0.0)
        blocker_count = safe_float(rank_row.get("open_blocker_count"), len(blocker_map.get(ticker, [])))
        risk_level = as_text(rank_row.get("risk_level"), as_text(review_row.get("risk_level"), "Unknown"))
        lane = as_text(rank_row.get("approval_lane"), as_text(review_row.get("approval_lane"), "Carry-forward review"))

        high_fields = int((grp["review_priority"] == "High").sum())
        outside_fields = int((grp["source_check_type"] == "Needs outside check").sum())
        undecided = int((grp["acceptance_status"] == "Needs human decision").sum())
        accepted = int((grp["acceptance_status"] == "Accept").sum())
        score = approval_score + high_fields * 4 + outside_fields * 2 - blocker_count * 3
        if "ready" in lane.lower():
            score += 12
        if risk_level.lower() == "high":
            score += 6

        top_fields = grp.head(5)
        first_checks = "; ".join(f"{r.field_group}: {r.what_to_do}" for r in top_fields.itertuples(index=False))
        outside_needed = "; ".join(sorted(set(grp.loc[grp["source_check_type"] == "Needs outside check", "field_group"].astype(str).tolist()))) or "No outside checks in first pass"
        next_action = (
            "Review the top fields, then update pm_review_evidence_acceptance_input.csv."
            if accepted == 0
            else "Accepted evidence exists; check conflicts before copying anything to PM review."
        )

        priority_rows.append({
            "ticker": ticker,
            "triage_rank_score": round(score, 1),
            "approval_lane": lane,
            "risk_level": risk_level,
            "approval_score_0_100": approval_score,
            "open_blocker_count": int(blocker_count),
            "high_priority_evidence_fields": high_fields,
            "outside_checks_needed": outside_fields,
            "undecided_count": undecided,
            "accepted_count": accepted,
            "first_checks_plain": first_checks,
            "outside_sources_to_check": outside_needed,
            "next_step": next_action,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        headline = f"{ticker}: review evidence before any paper decision."
        if "ready" in lane.lower():
            headline = f"{ticker}: best candidate for a focused PM evidence pass."
        packet_rows.append({
            "ticker": ticker,
            "plain_headline": headline,
            "why_this_ticker_first": (
                f"Lane: {lane}. Risk: {risk_level}. Approval score: {approval_score:.0f}. "
                f"Open blockers: {int(blocker_count)}. Undecided evidence rows: {undecided}."
            ),
            "first_5_checks": first_checks,
            "outside_sources_to_check": outside_needed,
            "what_not_to_do": "Do not add size, calls, or puts from this packet. It only guides evidence review.",
            "editable_file": "pm_review_evidence_acceptance_input.csv",
            "source_files": "pm_review_evidence_acceptance_input.csv; risk_seed_approval_rank.csv; risk_seed_pm_review_status.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    priority = pd.DataFrame(priority_rows)
    if not priority.empty:
        priority = priority.sort_values(["triage_rank_score", "approval_score_0_100"], ascending=[False, False]).reset_index(drop=True)
        priority.insert(0, "review_order", range(1, len(priority) + 1))

    packet_cards = pd.DataFrame(packet_rows)
    if not packet_cards.empty and not priority.empty:
        order = dict(zip(priority["ticker"], priority["review_order"]))
        packet_cards["_order"] = packet_cards["ticker"].map(order).fillna(9999)
        packet_cards = packet_cards.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    source_rows: list[dict[str, Any]] = []
    if not field_plan.empty:
        for (group, source_type), grp in field_plan.groupby(["field_group", "source_check_type"], sort=False):
            source_rows.append({
                "field_group": group,
                "source_check_type": source_type,
                "field_count": int(len(grp)),
                "high_priority_count": int((grp["review_priority"] == "High").sum()),
                "example_tickers": ", ".join(grp["ticker"].drop_duplicates().head(8).tolist()),
                "plain_rule": (
                    "Needs a source outside the model before accepting."
                    if source_type == "Needs outside check"
                    else "Can be reviewed against local files and policy, but still needs human acceptance."
                ),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    source_ladder = pd.DataFrame(source_rows)

    top_ticker = as_text(priority.iloc[0]["ticker"], "") if not priority.empty else "No ticker"
    state = {
        "date": today_str(),
        "status": "PM_EVIDENCE_REVIEW_TRIAGE_ACTIVE",
        "ticker_count": int(priority["ticker"].nunique()) if not priority.empty else 0,
        "field_review_count": int(len(field_plan)),
        "high_priority_field_count": int((field_plan.get("review_priority", pd.Series(dtype=str)) == "High").sum()) if not field_plan.empty else 0,
        "outside_check_field_count": int((field_plan.get("source_check_type", pd.Series(dtype=str)) == "Needs outside check").sum()) if not field_plan.empty else 0,
        "top_review_ticker": top_ticker,
        "plain_answer": (
            f"Evidence review triage is active. {len(field_plan)} evidence fields were sorted across "
            f"{priority['ticker'].nunique() if not priority.empty else 0} tickers. Start with {top_ticker}, then review only the first few high-priority fields instead of reading the whole queue."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return priority, field_plan, packet_cards, source_ladder, state


def main() -> None:
    priority, field_plan, packet_cards, source_ladder, state = build_outputs()
    priority.to_csv(OUT_PRIORITY, index=False)
    field_plan.to_csv(OUT_FIELD_PLAN, index=False)
    packet_cards.to_csv(OUT_PACKET_CARDS, index=False)
    source_ladder.to_csv(OUT_SOURCE_LADDER, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Review Priority Queue\n\n" + df_to_markdown(priority.head(40)),
        "## First Field Checks\n\n" + df_to_markdown(field_plan.head(120)),
        "## Packet Cards\n\n" + df_to_markdown(packet_cards.head(40)),
        "## Source Ladder\n\n" + df_to_markdown(source_ladder),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 205 - PM Evidence Review Triage", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Tickers ranked: {state['ticker_count']}")
    print(f"[OK] Evidence fields sorted: {state['field_review_count']}")
    print(f"[OK] High-priority fields: {state['high_priority_field_count']}")
    print(f"[OK] Outside-check fields: {state['outside_check_field_count']}")
    print(f"[OK] Top review ticker: {state['top_review_ticker']}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
