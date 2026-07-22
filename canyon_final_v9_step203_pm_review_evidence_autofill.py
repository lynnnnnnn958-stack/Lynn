#!/usr/bin/env python3
"""
Canyon v9 Step 203 - PM Review Evidence Autofill Assistant.

Research-only. No broker connection. No live orders.

This step pre-fills evidence suggestions for the PM review template without
approving anything and without overwriting human edits. It creates a draft file
and a field-by-field source audit so a human can see where every suggestion came
from before deciding what belongs in the official review input.

Outputs:
  pm_review_evidence_autofill_state.json
  pm_review_evidence_autofill_suggestions.csv
  pm_review_evidence_autofill_draft.csv
  pm_review_evidence_autofill_coverage.csv
  pm_review_evidence_autofill_report.md
"""
from __future__ import annotations

from datetime import datetime, timedelta
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


OUT_STATE = ROOT / "pm_review_evidence_autofill_state.json"
OUT_SUGGESTIONS = ROOT / "pm_review_evidence_autofill_suggestions.csv"
OUT_DRAFT = ROOT / "pm_review_evidence_autofill_draft.csv"
OUT_COVERAGE = ROOT / "pm_review_evidence_autofill_coverage.csv"
OUT_REPORT = ROOT / "pm_review_evidence_autofill_report.md"


EVIDENCE_FIELDS = [
    "thesis_plain",
    "earnings_date",
    "expected_event_move_pct",
    "event_size_policy",
    "liquidity_snapshot_date",
    "bid_ask_spread_bps",
    "avg_daily_dollar_volume_check",
    "sector_confirmed",
    "crowding_check",
    "news_proof_note",
    "execution_proof_note",
    "paper_stop_pct",
    "option_route_requested",
    "decision_note",
    "last_updated",
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


def short(value: Any, limit: int = 240) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def first_row_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""].copy()
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby(ticker_col, sort=False)}


def best_queue_by_ticker(df: pd.DataFrame, ticker_col: str = "target_ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""].copy()
    if "causal_confidence_score" in work.columns:
        work["_score"] = pd.to_numeric(work["causal_confidence_score"], errors="coerce").fillna(-999)
        work = work.sort_values([ticker_col, "_score"], ascending=[True, False])
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby(ticker_col, sort=False)}


def parse_future_date(value: Any) -> str:
    text = as_text(value, "")
    if not text:
        return ""
    try:
        dt = pd.to_datetime(text, errors="coerce")
    except Exception:
        return ""
    if pd.isna(dt):
        return ""
    if dt.date() < datetime.strptime(today_str(), "%Y-%m-%d").date():
        return ""
    return dt.strftime("%Y-%m-%d")


def pct_text(value: Any) -> str:
    x = safe_float(value, np.nan)
    if not np.isfinite(x):
        return ""
    if abs(x) <= 1.5:
        x *= 100.0
    return f"{x:.1f}"


def money_text(value: Any) -> str:
    x = safe_float(value, np.nan)
    if not np.isfinite(x):
        return ""
    return f"${x:,.0f}"


def add_suggestion(rows: list[dict[str, Any]], ticker: str, field: str, value: Any, confidence: str, source: str, rationale: str, existing_value: Any = "") -> None:
    value_text = as_text(value, "")
    if not value_text:
        return
    rows.append({
        "ticker": ticker,
        "field_name": field,
        "suggested_value": value_text,
        "confidence": confidence,
        "existing_value": as_text(existing_value, ""),
        "will_fill_draft": "No" if is_filled(existing_value) else "Yes",
        "human_confirmation_needed": "Yes",
        "rationale": short(rationale, 320),
        "source_files": source,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    })


def future_date_from_days(days: Any) -> str:
    d = safe_float(days, np.nan)
    if not np.isfinite(d):
        return ""
    target = datetime.strptime(today_str(), "%Y-%m-%d") + timedelta(days=int(round(d)))
    if target.date() < datetime.strptime(today_str(), "%Y-%m-%d").date():
        return ""
    return target.strftime("%Y-%m-%d")


def event_policy_from_risk(label: str, action: str, days: Any) -> str:
    text = f"{label} {action}".lower()
    d = safe_float(days, np.nan)
    if "reduce" in text or "size" in text or "high" in text:
        return "Reduce or stay flat through the event window; no new size until the event gap is reviewed."
    if np.isfinite(d) and d <= 10:
        return "Event is near; keep review-only until the event passes or expected move is sourced."
    return "Tiny review only after earnings date, expected move, and event window are confirmed."


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    review_input = read_csv_safe(ROOT / "risk_seed_pm_review_input.csv")
    review_status = read_csv_safe(ROOT / "risk_seed_pm_review_status.csv")
    metrics = read_csv_safe(ROOT / "risk_book_seed_metric_detail.csv")
    earnings = read_csv_safe(ROOT / "earnings_calendar.csv")
    earnings_gap = read_csv_safe(ROOT / "earnings_gap_down_risk.csv")
    sector_map = read_csv_safe(ROOT / "sector_map.csv")
    event_rank = read_csv_safe(ROOT / "event_readthrough_target_ranking.csv")
    event_queue = read_csv_safe(ROOT / "event_causal_validation_queue.csv")
    event_summary = read_csv_safe(ROOT / "event_readthrough_event_summary.csv")
    execution_cost = read_csv_safe(ROOT / "execution_cost_model.csv")
    execution_cards = read_csv_safe(ROOT / "execution_tca_ticker_cards.csv")

    if review_input.empty:
        empty = pd.DataFrame()
        state = {
            "date": today_str(),
            "status": "NO_PM_REVIEW_INPUT",
            "plain_answer": "Step203 needs risk_seed_pm_review_input.csv from Step201 first.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return empty, empty, empty, state

    status_map = first_row_by_ticker(review_status)
    metric_map = first_row_by_ticker(metrics)
    earnings_map = first_row_by_ticker(earnings)
    gap_map = first_row_by_ticker(earnings_gap)
    sector_map_rows = first_row_by_ticker(sector_map)
    event_rank_map = first_row_by_ticker(event_rank, "target_ticker") if "target_ticker" in event_rank.columns else first_row_by_ticker(event_rank)
    event_queue_map = best_queue_by_ticker(event_queue)
    execution_cost_map = first_row_by_ticker(execution_cost)
    execution_card_map = first_row_by_ticker(execution_cards)

    suggestion_rows: list[dict[str, Any]] = []
    draft = review_input.copy()
    for col in EVIDENCE_FIELDS:
        if col in draft.columns:
            draft[col] = draft[col].astype("object")

    for idx, row in draft.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        metric = metric_map.get(ticker, pd.Series(dtype=object))
        status = status_map.get(ticker, pd.Series(dtype=object))
        earn = earnings_map.get(ticker, pd.Series(dtype=object))
        gap = gap_map.get(ticker, pd.Series(dtype=object))
        sector = sector_map_rows.get(ticker, pd.Series(dtype=object))
        event = event_rank_map.get(ticker, pd.Series(dtype=object))
        queue = event_queue_map.get(ticker, pd.Series(dtype=object))
        exec_cost = execution_cost_map.get(ticker, pd.Series(dtype=object))
        exec_card = execution_card_map.get(ticker, pd.Series(dtype=object))

        sector_value = as_text(sector.get("sector"), "")
        theme_value = as_text(metric.get("sector_or_theme"), "")
        if not sector_value or sector_value.lower() == "unknown":
            sector_value = theme_value
        if not sector_value or sector_value.lower() == "unknown":
            sector_value = as_text(event.get("subsector_cycle_phase"), "Needs manual sector/theme confirmation")

        headline = as_text(event.get("top_headline"), as_text(queue.get("headline"), as_text(metric.get("event_hook"), "")))
        role = as_text(event.get("top_target_role"), as_text(queue.get("chain_role"), ""))
        tone = as_text(event.get("top_tone"), as_text(queue.get("market_tone"), ""))
        proof_required = as_text(event.get("proof_required"), as_text(queue.get("required_next_action"), ""))
        event_score = safe_float(event.get("best_event_score"), safe_float(queue.get("causal_confidence_score"), np.nan))

        thesis_parts = []
        if sector_value:
            thesis_parts.append(f"{ticker} belongs to {sector_value}.")
        if headline:
            thesis_parts.append(f"Current review hook: {headline}.")
        if role or tone:
            thesis_parts.append(f"News read-through: {tone or 'unknown tone'} / {role or 'unknown role'}.")
        thesis_parts.append("This is a review note only, not an approval.")
        add_suggestion(
            suggestion_rows,
            ticker,
            "thesis_plain",
            " ".join(thesis_parts),
            "Medium" if headline or sector_value else "Low",
            "risk_book_seed_metric_detail.csv; event_readthrough_target_ranking.csv; event_causal_validation_queue.csv",
            "Combines sector/theme and latest mapped news hook into plain English.",
            row.get("thesis_plain"),
        )

        future_earnings = parse_future_date(earn.get("earnings_date"))
        if not future_earnings:
            future_earnings = future_date_from_days(gap.get("earnings_days_to_event"))
        add_suggestion(
            suggestion_rows,
            ticker,
            "earnings_date",
            future_earnings,
            "Medium" if future_earnings else "Low",
            "earnings_calendar.csv; earnings_gap_down_risk.csv",
            "Uses future earnings date when available; falls back to days-to-event estimate.",
            row.get("earnings_date"),
        )

        implied_move = pct_text(gap.get("implied_move_or_fallback"))
        add_suggestion(
            suggestion_rows,
            ticker,
            "expected_event_move_pct",
            implied_move,
            "Low" if "MISSING" in as_text(gap.get("data_status"), "").upper() else "Medium",
            "earnings_gap_down_risk.csv",
            "Uses implied move when available; otherwise fallback move must be manually confirmed.",
            row.get("expected_event_move_pct"),
        )
        add_suggestion(
            suggestion_rows,
            ticker,
            "event_size_policy",
            event_policy_from_risk(as_text(gap.get("earnings_risk_label")), as_text(gap.get("gap_down_action")), gap.get("earnings_days_to_event")),
            "Medium" if not gap.empty else "Low",
            "earnings_gap_down_risk.csv",
            "Conservative event policy based on earnings risk and gap-down action.",
            row.get("event_size_policy"),
        )

        latest_date = as_text(metric.get("latest_price_date"), "")
        add_suggestion(
            suggestion_rows,
            ticker,
            "liquidity_snapshot_date",
            latest_date,
            "Medium" if latest_date else "Low",
            "risk_book_seed_metric_detail.csv",
            "Uses latest local price date as the review snapshot date.",
            row.get("liquidity_snapshot_date"),
        )
        spread_bps = safe_float(exec_cost.get("spread_bps"), np.nan)
        add_suggestion(
            suggestion_rows,
            ticker,
            "bid_ask_spread_bps",
            f"{spread_bps:.1f}" if np.isfinite(spread_bps) else "",
            "Medium",
            "execution_cost_model.csv",
            "Uses local execution cost model spread proxy; still requires manual quote confirmation.",
            row.get("bid_ask_spread_bps"),
        )
        adv = money_text(metric.get("avg_dollar_volume_20d"))
        add_suggestion(
            suggestion_rows,
            ticker,
            "avg_daily_dollar_volume_check",
            adv,
            "Medium" if adv else "Low",
            "risk_book_seed_metric_detail.csv",
            "Uses 20-day average dollar volume when available.",
            row.get("avg_daily_dollar_volume_check"),
        )
        add_suggestion(
            suggestion_rows,
            ticker,
            "sector_confirmed",
            sector_value,
            "Medium" if sector_value and "Needs manual" not in sector_value else "Low",
            "sector_map.csv; risk_book_seed_metric_detail.csv; event_readthrough_target_ranking.csv",
            "Uses sector map first, then risk seed theme, then event subsector cycle.",
            row.get("sector_confirmed"),
        )

        corr_parts = []
        for label, col in [("SPY", "corr_spy"), ("QQQ", "corr_qqq"), ("SMH", "corr_smh")]:
            x = safe_float(metric.get(col), np.nan)
            if np.isfinite(x):
                corr_parts.append(f"{label} corr {x:.2f}")
        beta = safe_float(metric.get("beta_spy"), np.nan)
        if np.isfinite(beta):
            corr_parts.append(f"SPY beta {beta:.2f}")
        if sector_value:
            corr_parts.append(f"sector/theme {sector_value}")
        add_suggestion(
            suggestion_rows,
            ticker,
            "crowding_check",
            "; ".join(corr_parts),
            "Medium" if corr_parts else "Low",
            "risk_book_seed_metric_detail.csv; sector_map.csv",
            "Summarizes correlation, beta, and sector/theme crowding context.",
            row.get("crowding_check"),
        )

        news_note = []
        if headline:
            news_note.append(f"Headline: {headline}.")
        if np.isfinite(event_score):
            news_note.append(f"Event/causal score {event_score:.1f}.")
        if role or tone:
            news_note.append(f"Role/tone: {role or 'unknown'} / {tone or 'unknown'}.")
        if proof_required:
            news_note.append(f"Still needs proof: {proof_required}.")
        add_suggestion(
            suggestion_rows,
            ticker,
            "news_proof_note",
            " ".join(news_note),
            "Medium" if headline else "Low",
            "event_readthrough_target_ranking.csv; event_causal_validation_queue.csv",
            "Pulls mapped news headline, causal score, target role, and remaining proof.",
            row.get("news_proof_note"),
        )

        execution_note = []
        if not exec_card.empty:
            execution_note.append(as_text(exec_card.get("cost_line"), ""))
            execution_note.append(as_text(exec_card.get("manual_check"), ""))
        elif not exec_cost.empty:
            execution_note.append(as_text(exec_cost.get("execution_instruction"), ""))
            if np.isfinite(spread_bps):
                execution_note.append(f"Spread proxy {spread_bps:.1f} bps.")
        else:
            liquidity = as_text(metric.get("liquidity_status"), "")
            if liquidity:
                execution_note.append(f"Liquidity status: {liquidity}. Manual bid/ask quote still required.")
        add_suggestion(
            suggestion_rows,
            ticker,
            "execution_proof_note",
            " ".join(x for x in execution_note if x),
            "Medium" if execution_note else "Low",
            "execution_tca_ticker_cards.csv; execution_cost_model.csv; risk_book_seed_metric_detail.csv",
            "Pulls execution desk note or falls back to risk liquidity status.",
            row.get("execution_proof_note"),
        )

        stop = safe_float(row.get("system_stop_pct"), safe_float(metric.get("paper_stop_if_ever_tested_pct"), np.nan))
        add_suggestion(
            suggestion_rows,
            ticker,
            "paper_stop_pct",
            f"{stop:.1f}" if np.isfinite(stop) else "",
            "Medium",
            "risk_seed_pm_review_input.csv; risk_book_seed_metric_detail.csv",
            "Uses system stop as the draft paper stop. Human still decides whether it is acceptable.",
            row.get("paper_stop_pct"),
        )
        add_suggestion(
            suggestion_rows,
            ticker,
            "option_route_requested",
            "NO",
            "High",
            "Canyon risk policy",
            "Options remain blocked during PM review autofill.",
            row.get("option_route_requested"),
        )
        add_suggestion(
            suggestion_rows,
            ticker,
            "decision_note",
            "Autofill draft only. Human PM review is still required before any status change.",
            "High",
            "Step203 policy",
            "Prevents evidence suggestions from being mistaken for approval.",
            row.get("decision_note"),
        )
        add_suggestion(
            suggestion_rows,
            ticker,
            "last_updated",
            today_str(),
            "High",
            "Step203 run date",
            "Marks when this draft evidence was generated.",
            row.get("last_updated"),
        )

    suggestions = pd.DataFrame(suggestion_rows)
    if not suggestions.empty:
        for _, srow in suggestions.iterrows():
            if srow["will_fill_draft"] != "Yes":
                continue
            mask = draft["ticker"].apply(clean_ticker) == clean_ticker(srow["ticker"])
            field = srow["field_name"]
            if field in draft.columns:
                draft.loc[mask, field] = srow["suggested_value"]
        if "review_status" in draft.columns:
            draft["review_status"] = draft["review_status"].fillna("NEEDS_REVIEW")
        if "reviewer" in draft.columns:
            draft["reviewer"] = draft["reviewer"].fillna("")
        if "approved_cap_pct" in draft.columns:
            draft["approved_cap_pct"] = draft["approved_cap_pct"].where(draft["approved_cap_pct"].notna(), "")

    coverage_rows: list[dict[str, Any]] = []
    for field in EVIDENCE_FIELDS:
        field_rows = suggestions[suggestions["field_name"] == field] if not suggestions.empty else pd.DataFrame()
        coverage_rows.append({
            "field_name": field,
            "suggestion_count": int(len(field_rows)),
            "draft_fill_count": int((field_rows.get("will_fill_draft", pd.Series(dtype=str)) == "Yes").sum()) if not field_rows.empty else 0,
            "high_confidence_count": int((field_rows.get("confidence", pd.Series(dtype=str)) == "High").sum()) if not field_rows.empty else 0,
            "medium_confidence_count": int((field_rows.get("confidence", pd.Series(dtype=str)) == "Medium").sum()) if not field_rows.empty else 0,
            "low_confidence_count": int((field_rows.get("confidence", pd.Series(dtype=str)) == "Low").sum()) if not field_rows.empty else 0,
            "human_confirmation_needed": "Yes",
        })
    coverage = pd.DataFrame(coverage_rows)

    state = {
        "date": today_str(),
        "status": "PM_REVIEW_EVIDENCE_AUTOFILL_ACTIVE",
        "review_rows": int(len(review_input)),
        "suggestion_count": int(len(suggestions)),
        "draft_rows": int(len(draft)),
        "draft_filled_cells": int((suggestions.get("will_fill_draft", pd.Series(dtype=str)) == "Yes").sum()) if not suggestions.empty else 0,
        "high_confidence_suggestions": int((suggestions.get("confidence", pd.Series(dtype=str)) == "High").sum()) if not suggestions.empty else 0,
        "low_confidence_suggestions": int((suggestions.get("confidence", pd.Series(dtype=str)) == "Low").sum()) if not suggestions.empty else 0,
        "plain_answer": (
            f"Evidence autofill is active. {len(suggestions)} field-level suggestions were generated for {len(review_input)} PM review rows. "
            "They are drafts only; human confirmation is still required before Step201 can treat anything as reviewed."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return suggestions, draft, coverage, state


def main() -> None:
    suggestions, draft, coverage, state = build_outputs()
    suggestions.to_csv(OUT_SUGGESTIONS, index=False)
    draft.to_csv(OUT_DRAFT, index=False)
    coverage.to_csv(OUT_COVERAGE, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Suggestions\n\n" + df_to_markdown(suggestions.head(180)),
        "## Draft Review Input\n\n" + df_to_markdown(draft.head(80)),
        "## Coverage\n\n" + df_to_markdown(coverage),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 203 - PM Review Evidence Autofill Assistant", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Suggestions: {state['suggestion_count']}")
    print(f"[OK] Draft filled cells: {state['draft_filled_cells']}")
    print(f"[OK] Human confirmation required: True")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
