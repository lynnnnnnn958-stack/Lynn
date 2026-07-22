#!/usr/bin/env python3
"""
Canyon v9 Step 206 - PM Evidence Source Proof Desk.

Research-only. No broker connection. No live orders.

Step205 says which evidence needs an outside check. This step converts those
outside checks into a human proof-capture desk: what question to answer, what
source to use, what value to record, and what is still missing before an
evidence row can be accepted in Step204.

Outputs:
  pm_evidence_source_proof_state.json
  pm_evidence_source_proof_input.csv
  pm_evidence_source_proof_status.csv
  pm_evidence_source_proof_ready_for_acceptance.csv
  pm_evidence_source_proof_gap_queue.csv
  pm_evidence_source_proof_report.md
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


OUT_STATE = ROOT / "pm_evidence_source_proof_state.json"
OUT_INPUT = ROOT / "pm_evidence_source_proof_input.csv"
OUT_STATUS = ROOT / "pm_evidence_source_proof_status.csv"
OUT_READY = ROOT / "pm_evidence_source_proof_ready_for_acceptance.csv"
OUT_GAPS = ROOT / "pm_evidence_source_proof_gap_queue.csv"
OUT_REPORT = ROOT / "pm_evidence_source_proof_report.md"


HUMAN_COLUMNS = [
    "proof_status",
    "source_name",
    "source_url",
    "observed_value",
    "observed_time",
    "price_reaction_checked",
    "volume_reaction_checked",
    "reviewer",
    "review_date",
    "proof_note",
]

INPUT_COLUMNS = [
    "proof_id",
    "proof_status",
    "ticker",
    "field_group",
    "field_name",
    "review_priority",
    "review_priority_score",
    "suggested_value",
    "required_question",
    "preferred_source",
    "acceptable_proof",
    "source_name",
    "source_url",
    "observed_value",
    "observed_time",
    "price_reaction_checked",
    "volume_reaction_checked",
    "reviewer",
    "review_date",
    "proof_note",
    "step204_action_hint",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

READY_COLUMNS = [
    "ticker",
    "field_name",
    "field_group",
    "observed_value",
    "source_name",
    "source_url",
    "reviewer",
    "review_date",
    "step204_action_hint",
    "proof_note",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

GAP_COLUMNS = [
    "ticker",
    "field_name",
    "field_group",
    "review_priority",
    "missing_proof",
    "required_question",
    "preferred_source",
    "acceptable_proof",
    "next_step",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]


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


def is_filled(value: Any) -> bool:
    return bool(as_text(value, ""))


def short(value: Any, limit: int = 300) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def proof_id(row: pd.Series) -> str:
    ticker = clean_ticker(row.get("ticker"))
    field = as_text(row.get("field_name"), "").lower()
    value = as_text(row.get("suggested_value"), "")
    raw = "|".join([ticker, field, value])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{ticker}-{field}-{digest}"


def norm_status(value: Any) -> str:
    text = as_text(value, "Needs proof")
    key = text.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "NEEDS_PROOF": "Needs proof",
        "PENDING": "Needs proof",
        "VERIFY": "Verified",
        "VERIFIED": "Verified",
        "ACCEPT": "Verified",
        "REJECT": "Rejected",
        "REJECTED": "Rejected",
        "SOURCE_UNAVAILABLE": "Source unavailable",
        "UNAVAILABLE": "Source unavailable",
        "NOT_NEEDED": "Not needed",
        "IGNORE": "Not needed",
    }
    return aliases.get(key, text if text in {"Needs proof", "Verified", "Rejected", "Source unavailable", "Not needed"} else "Needs proof")


def existing_by_id(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "proof_id" not in df.columns:
        return {}
    work = df.copy()
    work["proof_id"] = work["proof_id"].map(lambda x: as_text(x, ""))
    work = work[work["proof_id"] != ""].copy()
    return {pid: grp.iloc[0] for pid, grp in work.groupby("proof_id", sort=False)}


def proof_question(field: str, ticker: str) -> str:
    if field == "earnings_date":
        return f"What is the next confirmed earnings date for {ticker}?"
    if field == "expected_event_move_pct":
        return f"What move is the options market or event-risk source implying for {ticker}?"
    if field == "liquidity_snapshot_date":
        return f"Is the liquidity snapshot for {ticker} current enough to trust today?"
    if field == "bid_ask_spread_bps":
        return f"What is the current bid/ask spread for {ticker}, in basis points if possible?"
    if field == "news_proof_note":
        return f"Did the mapped headline for {ticker} have a source, timestamp, and post-news price/volume reaction?"
    if field == "execution_proof_note":
        return f"Is the trading-cost note for {ticker} supported by spread, dollar volume, or execution data?"
    return f"What outside evidence proves this field for {ticker}?"


def preferred_source(field: str) -> str:
    if field == "earnings_date":
        return "Company IR, Nasdaq earnings calendar, Yahoo Finance, or broker-grade earnings calendar."
    if field == "expected_event_move_pct":
        return "Options chain implied move, earnings-event desk, or documented fallback event move."
    if field == "liquidity_snapshot_date":
        return "Fresh quote snapshot, Yahoo Finance quote data, or local price/liquidity cache date."
    if field == "bid_ask_spread_bps":
        return "Current quote page, broker quote snapshot, or execution-cost file with timestamp."
    if field == "news_proof_note":
        return "Original news article, timestamp, linked-stock map, and price/volume reaction after the headline."
    if field == "execution_proof_note":
        return "Execution-cost model, spread snapshot, volume screen, or manual quote note."
    return "Reliable outside source or audited local file."


def acceptable_proof(field: str) -> str:
    if field == "earnings_date":
        return "Record source name, source URL or file, observed earnings date, reviewer, and review date."
    if field == "expected_event_move_pct":
        return "Record observed expected move %, source, whether it is implied or fallback, reviewer, and review date."
    if field == "liquidity_snapshot_date":
        return "Record quote/date observed and confirm it is not stale for today's review."
    if field == "bid_ask_spread_bps":
        return "Record bid, ask, or spread bps with source and timestamp."
    if field == "news_proof_note":
        return "Record source, headline timestamp, price reaction check, volume reaction check, and why the ticker link is direct or indirect."
    if field == "execution_proof_note":
        return "Record source, spread/liquidity evidence, and whether the cost note is usable."
    return "Record source, observed value, reviewer, and date."


def missing_for_row(row: pd.Series) -> list[str]:
    status = norm_status(row.get("proof_status"))
    if status in {"Rejected", "Source unavailable", "Not needed"}:
        return []
    missing: list[str] = []
    for col, label in [
        ("source_name", "source name"),
        ("observed_value", "observed value"),
        ("reviewer", "reviewer"),
        ("review_date", "review date"),
    ]:
        if not is_filled(row.get(col)):
            missing.append(label)
    field = as_text(row.get("field_name"), "")
    if field in {"news_proof_note", "execution_proof_note"}:
        if not is_filled(row.get("observed_time")):
            missing.append("source or observation time")
    if field == "news_proof_note":
        if as_text(row.get("price_reaction_checked"), "").lower() not in {"yes", "no"}:
            missing.append("price reaction checked")
        if as_text(row.get("volume_reaction_checked"), "").lower() not in {"yes", "no"}:
            missing.append("volume reaction checked")
    return missing


def build_input(field_plan: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    existing_map = existing_by_id(existing)
    if field_plan.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)

    work = field_plan.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()
    if "source_check_type" in work.columns:
        work = work[work["source_check_type"].astype(str).str.lower().eq("needs outside check")].copy()
    if work.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)
    work = work.sort_values(["review_priority_score", "ticker", "field_name"], ascending=[False, True, True])

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, src in work.iterrows():
        pid = proof_id(src)
        seen.add(pid)
        old = existing_map.get(pid)
        ticker = clean_ticker(src.get("ticker"))
        field = as_text(src.get("field_name"), "")
        row = {
            "proof_id": pid,
            "proof_status": "Needs proof",
            "ticker": ticker,
            "field_group": as_text(src.get("field_group"), ""),
            "field_name": field,
            "review_priority": as_text(src.get("review_priority"), ""),
            "review_priority_score": safe_float(src.get("review_priority_score"), 0.0),
            "suggested_value": as_text(src.get("suggested_value"), ""),
            "required_question": proof_question(field, ticker),
            "preferred_source": preferred_source(field),
            "acceptable_proof": acceptable_proof(field),
            "source_name": "",
            "source_url": "",
            "observed_value": "",
            "observed_time": "",
            "price_reaction_checked": "",
            "volume_reaction_checked": "",
            "reviewer": "",
            "review_date": "",
            "proof_note": "",
            "step204_action_hint": "If verified, manually set the matching Step204 evidence row to Accept.",
            "source_files": as_text(src.get("source_files"), ""),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        if old is not None:
            for col in HUMAN_COLUMNS:
                if col in old.index and is_filled(old.get(col)):
                    row[col] = old.get(col)
        row["proof_status"] = norm_status(row["proof_status"])
        rows.append(row)

    if not existing.empty:
        for _, old in existing.iterrows():
            pid = as_text(old.get("proof_id"), "")
            if not pid or pid in seen:
                continue
            row = {col: as_text(old.get(col), "") for col in INPUT_COLUMNS}
            row["proof_id"] = pid
            row["proof_status"] = norm_status(row.get("proof_status"))
            row["research_only"] = True
            row["no_broker_connection"] = True
            row["no_live_orders"] = True
            rows.append(row)

    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    field_plan = read_csv_safe(ROOT / "pm_evidence_review_field_plan.csv")
    existing = read_csv_safe(OUT_INPUT)
    proof_input = build_input(field_plan, existing)

    if proof_input.empty:
        empty = pd.DataFrame()
        state = {
            "date": today_str(),
            "status": "NO_OUTSIDE_SOURCE_CHECKS",
            "plain_answer": "No outside-source proof checks are waiting right now.",
            "proof_row_count": 0,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return proof_input, empty, empty, empty, state

    work = proof_input.copy()
    work["proof_status"] = work["proof_status"].apply(norm_status)
    work["_missing_list"] = work.apply(missing_for_row, axis=1)
    work["missing_proof"] = work["_missing_list"].map(lambda items: "; ".join(items) if items else "No missing proof.")
    work["source_proof_state"] = np.where(
        (work["proof_status"] == "Verified") & (work["_missing_list"].map(len) == 0),
        "Ready for Step204 accept",
        np.where(work["proof_status"].isin(["Rejected", "Source unavailable", "Not needed"]), work["proof_status"], "Needs proof"),
    )

    ready = work[work["source_proof_state"] == "Ready for Step204 accept"].copy()
    if ready.empty:
        ready_out = pd.DataFrame(columns=READY_COLUMNS)
    else:
        ready_out = ready[[c for c in READY_COLUMNS if c in ready.columns]].copy()

    gaps = work[work["source_proof_state"] == "Needs proof"].copy()
    if gaps.empty:
        gaps_out = pd.DataFrame(columns=GAP_COLUMNS)
    else:
        gaps["next_step"] = gaps.apply(
            lambda r: f"Fill {r['missing_proof']} for {clean_ticker(r.get('ticker'))} {as_text(r.get('field_group'))}.",
            axis=1,
        )
        gaps = gaps.sort_values(["review_priority_score", "ticker", "field_name"], ascending=[False, True, True])
        gaps_out = gaps[[c for c in GAP_COLUMNS if c in gaps.columns]].copy()

    status_rows: list[dict[str, Any]] = []
    for ticker, grp in work.groupby("ticker", sort=False):
        needed = int((grp["source_proof_state"] == "Needs proof").sum())
        ready_count = int((grp["source_proof_state"] == "Ready for Step204 accept").sum())
        rejected = int((grp["source_proof_state"].isin(["Rejected", "Source unavailable"])).sum())
        high_needed = int(((grp["source_proof_state"] == "Needs proof") & (grp["review_priority"].astype(str) == "High")).sum())
        first_gap = "No missing outside proof."
        if needed:
            top = grp[grp["source_proof_state"] == "Needs proof"].sort_values("review_priority_score", ascending=False).iloc[0]
            first_gap = f"{as_text(top.get('field_group'))}: {as_text(top.get('required_question'))}"
        status_rows.append({
            "ticker": ticker,
            "outside_proof_rows": int(len(grp)),
            "ready_for_accept_count": ready_count,
            "needs_proof_count": needed,
            "high_priority_needs_proof_count": high_needed,
            "rejected_or_unavailable_count": rejected,
            "first_missing_proof": first_gap,
            "next_step": "Finish the first missing proof, then manually accept the matching Step204 row." if needed else "Ready proofs can be reviewed in Step204.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    status = pd.DataFrame(status_rows)
    if not status.empty:
        status = status.sort_values(["high_priority_needs_proof_count", "needs_proof_count", "ticker"], ascending=[False, False, True]).reset_index(drop=True)

    proof_input = work.drop(columns=["_missing_list", "missing_proof", "source_proof_state"], errors="ignore")
    proof_count = int(len(work))
    ready_count = int((work["source_proof_state"] == "Ready for Step204 accept").sum())
    needs_count = int((work["source_proof_state"] == "Needs proof").sum())
    high_needs = int(((work["source_proof_state"] == "Needs proof") & (work["review_priority"].astype(str) == "High")).sum())
    top_ticker = as_text(status.iloc[0]["ticker"], "") if not status.empty else "No ticker"
    state = {
        "date": today_str(),
        "status": "PM_EVIDENCE_SOURCE_PROOF_DESK_ACTIVE",
        "proof_row_count": proof_count,
        "ready_for_accept_count": ready_count,
        "needs_proof_count": needs_count,
        "high_priority_needs_proof_count": high_needs,
        "ticker_count": int(status["ticker"].nunique()) if not status.empty else 0,
        "top_missing_proof_ticker": top_ticker,
        "plain_answer": (
            f"Outside-source proof desk is active. {proof_count} proof rows need a human source check. "
            f"{ready_count} are ready for Step204 acceptance, {needs_count} still need proof, and {high_needs} are high priority."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return proof_input, status, ready_out, gaps_out, state


def main() -> None:
    proof_input, status, ready, gaps, state = build_outputs()
    proof_input.to_csv(OUT_INPUT, index=False)
    status.to_csv(OUT_STATUS, index=False)
    ready.to_csv(OUT_READY, index=False)
    gaps.to_csv(OUT_GAPS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## How To Use\n\nOpen `pm_evidence_source_proof_input.csv`. Fill source name, observed value, reviewer, and review date. For news proof, also record whether price and volume reaction were checked. Then rerun Step206.",
        "## Source Proof Input\n\n" + df_to_markdown(proof_input.head(120)),
        "## Ticker Status\n\n" + df_to_markdown(status.head(80)),
        "## Ready For Step204 Acceptance\n\n" + df_to_markdown(ready.head(80)),
        "## Gaps\n\n" + df_to_markdown(gaps.head(120)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 206 - PM Evidence Source Proof Desk", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Proof rows: {state['proof_row_count']}")
    print(f"[OK] Ready for Step204 accept: {state['ready_for_accept_count']}")
    print(f"[OK] Still needs proof: {state['needs_proof_count']}")
    print(f"[OK] High-priority missing proof: {state['high_priority_needs_proof_count']}")
    print("[OK] Official PM review input changed: False")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
