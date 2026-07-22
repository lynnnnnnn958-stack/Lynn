#!/usr/bin/env python3
"""
Canyon v9 Step 202 - PM Review Final Gate Bridge.

Research-only. No broker connection. No live orders.

Step201 validates the human PM review intake. Step202 bridges that evidence into
the final decision logic as a veto matrix. It never treats PM review as a trade
ticket. PM review can only remove the "manual review missing" blocker; risk,
news/event, execution, liquidity, and options still keep independent veto power.

Outputs:
  pm_review_final_gate_bridge_state.json
  pm_review_final_gate_bridge.csv
  pm_review_final_gate_veto_matrix.csv
  pm_review_final_gate_next_actions.csv
  pm_review_final_gate_report.md
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


OUT_STATE = ROOT / "pm_review_final_gate_bridge_state.json"
OUT_BRIDGE = ROOT / "pm_review_final_gate_bridge.csv"
OUT_VETO = ROOT / "pm_review_final_gate_veto_matrix.csv"
OUT_NEXT = ROOT / "pm_review_final_gate_next_actions.csv"
OUT_REPORT = ROOT / "pm_review_final_gate_report.md"


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


def first_row_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""].copy()
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby(ticker_col, sort=False)}


def option_rows_by_ticker(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty or "ticker" not in df.columns:
        return {}
    work = df.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()
    return {ticker: grp.copy() for ticker, grp in work.groupby("ticker", sort=False)}


def blocker_types_by_ticker(blockers: pd.DataFrame) -> dict[str, set[str]]:
    if blockers.empty or "ticker" not in blockers.columns:
        return {}
    work = blockers.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    out: dict[str, set[str]] = {}
    for ticker, grp in work.groupby("ticker", sort=False):
        out[ticker] = {as_text(x, "") for x in grp.get("blocker_type", pd.Series(dtype=str)) if as_text(x, "")}
    return out


def plain(value: Any, default: str = "No data") -> str:
    text = as_text(value, default)
    replacements = {
        "APPROVE_TINY_PAPER_REVIEW": "approve tiny paper review",
        "DATA_GAP": "missing data",
        "NEEDS_REVIEW": "needs review",
        "NO_GO": "not allowed",
        "REDUCE_ONLY": "reduce only",
        "RESEARCH_ONLY": "research only",
        "RISK_REDUCTION_FIRST": "risk reduction first",
        "SEED_REVIEW_ONLY": "risk seed needs review",
        "SIZE_DOWN": "use smaller size",
        "WATCH_ONLY": "watch only",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    return " ".join(text.replace("_", " ").split())


def pass_fail(condition: bool, pass_msg: str, fail_msg: str) -> tuple[str, str]:
    return ("Pass", pass_msg) if condition else ("Block", fail_msg)


def review_veto(status_row: pd.Series) -> tuple[str, str]:
    review_state = as_text(status_row.get("review_state"), "Not reviewed yet")
    review_status = as_text(status_row.get("review_status"), "NEEDS_REVIEW")
    if review_state == "Ready for final gate check":
        return "Pass", "PM review packet is complete enough for the next gate."
    if review_state == "Watch only" or review_status == "WATCH_ONLY":
        return "Block", "Reviewer marked this watch only. No paper size."
    if review_state == "Rejected by reviewer" or review_status == "REJECT":
        return "Block", "Reviewer rejected this seed."
    if review_state == "Approval blocked":
        return "Block", plain(status_row.get("allowed_next_state"), "A hard review rule blocks promotion.")
    if review_state == "Incomplete approval":
        return "Block", plain(status_row.get("allowed_next_state"), "Approval request is missing evidence.")
    return "Block", "PM review has not been filled yet."


def risk_veto(status_row: pd.Series, gate_row: pd.Series | None) -> tuple[str, str]:
    review_state = as_text(status_row.get("review_state"), "")
    risk_level = as_text(status_row.get("risk_level"), "Unknown").lower()
    final_permission = plain(gate_row.get("final_permission") if gate_row is not None else "", "")
    risk_status = plain(gate_row.get("risk_status") if gate_row is not None else "", "")
    first_blocker = plain(gate_row.get("first_blocker") if gate_row is not None else "", "")

    if "do not add" in final_permission.lower() or "risk reduction" in risk_status.lower():
        return "Block", "Final gate says risk reduction first. PM review cannot override this."
    if "very high" in risk_level:
        return "Block", "Risk level is very high. Keep this as sandbox research only."
    if review_state != "Ready for final gate check":
        return "Block", "Risk seed approval is not complete."
    if "risk needs manual review" in first_blocker.lower() or "risk" in first_blocker.lower():
        return "Pass", "PM review can clear the manual risk-seed blocker for the next gate, but only for tiny paper review."
    return "Pass", "No independent risk veto found after PM review."


def event_veto(status_row: pd.Series, input_row: pd.Series | None, blocker_types: set[str], news_row: pd.Series | None) -> tuple[str, str]:
    missing = plain(status_row.get("missing_fields_plain"), "")
    if "News proof" in blocker_types and "news proof" in missing.lower():
        return "Block", "News proof is still missing."
    if "Earnings gap" in blocker_types and any(term in missing.lower() for term in ["earnings", "event move", "event size"]):
        return "Block", "Earnings or event gap policy is still missing."
    event_policy = plain(input_row.get("event_size_policy") if input_row is not None else "", "")
    if event_policy and any(term in event_policy.lower() for term in ["flat", "no size", "avoid", "zero"]):
        return "Block", "Reviewer event policy says stay flat or avoid new size near the event."
    proof_required = plain(news_row.get("proof_required") if news_row is not None else "", "")
    if proof_required and any(term in proof_required.lower() for term in ["collect more", "validate", "fill missing", "create risk-book"]):
        return "Block", "News/event system still asks for causal proof."
    return "Pass", "No open news or earnings veto after review evidence."


def execution_veto(status_row: pd.Series, input_row: pd.Series | None, exec_row: pd.Series | None, blocker_types: set[str]) -> tuple[str, str]:
    missing = plain(status_row.get("missing_fields_plain"), "")
    if "Execution proof" in blocker_types and "execution proof" in missing.lower():
        return "Block", "Execution proof is still missing."
    if "Liquidity" in blocker_types and any(term in missing.lower() for term in ["liquidity", "spread", "dollar-volume"]):
        return "Block", "Liquidity or spread proof is still missing."

    spread = safe_float(input_row.get("bid_ask_spread_bps") if input_row is not None else np.nan, np.nan)
    if np.isfinite(spread) and spread > 35:
        return "Block", "Manual spread is too wide for clean tiny paper review."

    card_status = plain(exec_row.get("card_status") if exec_row is not None else "", "")
    blocker_line = plain(exec_row.get("blocker_line") if exec_row is not None else "", "")
    if any(term in card_status.lower() for term in ["risk reduction", "do not add"]):
        return "Block", "Execution desk says repair or reduce risk first."
    if "missing spread" in blocker_line.lower() or "liquidity proof" in blocker_line.lower():
        return "Block", "Execution desk still reports missing spread or liquidity proof."
    return "Pass", "No independent execution/liquidity veto found after review evidence."


def options_veto(option_rows: pd.DataFrame) -> tuple[str, str]:
    if option_rows.empty:
        return "Block", "No option route evidence. Options stay blocked."
    no_go = pd.to_numeric(option_rows.get("no_go_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
    final_decisions = " ".join(option_rows.get("final_vehicle_decision", pd.Series(dtype=str)).astype(str).tolist()).upper()
    if int(no_go.max()) > 0:
        return "Block", "Option no-go checks are still open. Do not look for calls or puts."
    if "OPTION" not in final_decisions and "CALL" not in final_decisions and "PUT" not in final_decisions:
        return "Block", "Options are not the approved vehicle for this ticker."
    return "Pass", "Option route evidence is clean, but final manual option review is still required."


def bridge_decision(vetos: dict[str, tuple[str, str]], status_row: pd.Series) -> tuple[str, str, float]:
    review_state = as_text(status_row.get("review_state"), "")
    cap = safe_float(status_row.get("approved_cap_pct"), np.nan)
    system_cap = safe_float(status_row.get("system_seed_cap_pct"), 0.0)
    allowed_cap = min(x for x in [cap if np.isfinite(cap) else system_cap, system_cap] if np.isfinite(x))
    hard_blocks = [name for name, (state, _) in vetos.items() if state != "Pass" and name != "Options"]
    option_state = vetos.get("Options", ("Block", ""))[0]

    if review_state == "Rejected by reviewer":
        return "Rejected", "Do not promote. Reopen review only if the thesis changes.", 0.0
    if review_state == "Watch only":
        return "Watch only", "Keep on watchlist. No paper size and no options.", 0.0
    if hard_blocks:
        first = hard_blocks[0]
        return "Blocked before final gate", f"Fix {first.lower()} first: {vetos[first][1]}", 0.0
    if option_state != "Pass":
        return (
            "Tiny stock/ETF review candidate",
            "PM review and non-option gates are clean enough for the next daily final-gate refresh. Options remain blocked.",
            max(0.0, round(min(allowed_cap, 1.0), 4)),
        )
    return (
        "Tiny paper review candidate",
        "All bridge checks passed. Next final gate may consider tiny paper review only; still no live order.",
        max(0.0, round(min(allowed_cap, 1.0), 4)),
    )


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    review = read_csv_safe(ROOT / "risk_seed_pm_review_status.csv")
    review_input = read_csv_safe(ROOT / "risk_seed_pm_review_input.csv")
    bridge_gate = read_csv_safe(ROOT / "institutional_promotion_gate.csv")
    blockers = read_csv_safe(ROOT / "risk_seed_blocker_matrix.csv")
    news = read_csv_safe(ROOT / "event_readthrough_target_ranking.csv")
    execution = read_csv_safe(ROOT / "execution_tca_ticker_cards.csv")
    options = read_csv_safe(ROOT / "options_execution_route_matrix.csv")

    if review.empty:
        empty = pd.DataFrame()
        state = {
            "date": today_str(),
            "status": "NO_PM_REVIEW_STATUS",
            "plain_answer": "Step202 needs Step201 PM review status first.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return empty, empty, empty, state

    input_map = first_row_by_ticker(review_input)
    gate_map = first_row_by_ticker(bridge_gate)
    blocker_map = blocker_types_by_ticker(blockers)
    news_map = first_row_by_ticker(news, "target_ticker") if "target_ticker" in news.columns else first_row_by_ticker(news)
    exec_map = first_row_by_ticker(execution)
    option_map = option_rows_by_ticker(options)

    bridge_rows: list[dict[str, Any]] = []
    veto_rows: list[dict[str, Any]] = []
    next_rows: list[dict[str, Any]] = []

    for _, row in review.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        input_row = input_map.get(ticker)
        gate_row = gate_map.get(ticker)
        blocker_types = blocker_map.get(ticker, set())
        news_row = news_map.get(ticker)
        exec_row = exec_map.get(ticker)
        opt_rows = option_map.get(ticker, pd.DataFrame())

        vetos = {
            "PM Review": review_veto(row),
            "Risk": risk_veto(row, gate_row),
            "News / Event": event_veto(row, input_row, blocker_types, news_row),
            "Execution / Liquidity": execution_veto(row, input_row, exec_row, blocker_types),
            "Options": options_veto(opt_rows),
        }
        decision, next_step, cap = bridge_decision(vetos, row)
        blocking = [name for name, (state, _) in vetos.items() if state != "Pass"]
        non_option_blocking = [name for name in blocking if name != "Options"]

        for gate_name, (gate_state, reason) in vetos.items():
            veto_rows.append({
                "ticker": ticker,
                "gate": gate_name,
                "gate_state": gate_state,
                "reason": reason,
                "source_files": "risk_seed_pm_review_status.csv; institutional_promotion_gate.csv; event/execution/options desks",
                "research_only": True,
            })

        bridge_rows.append({
            "ticker": ticker,
            "bridge_decision": decision,
            "next_step": next_step,
            "max_tiny_paper_review_pct": cap,
            "non_option_gates_passed": "Yes" if not non_option_blocking else "No",
            "options_allowed": "No" if "Options" in blocking else "Review required",
            "first_blocking_gate": non_option_blocking[0] if non_option_blocking else ("Options" if "Options" in blocking else "Final manual check"),
            "blocking_gates": "; ".join(blocking) if blocking else "None",
            "current_final_permission": as_text(gate_row.get("final_permission") if gate_row is not None else "", "No current final gate row"),
            "current_first_blocker": as_text(gate_row.get("first_blocker") if gate_row is not None else "", "No current final gate row"),
            "pm_review_state": as_text(row.get("review_state"), ""),
            "pm_review_status": as_text(row.get("review_status"), ""),
            "proof_score_0_100": safe_float(row.get("proof_score_0_100"), 0.0),
            "source_files": "risk_seed_pm_review_status.csv; institutional_promotion_gate.csv; pm_review_final_gate_veto_matrix.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        priority = "P1" if decision in {"Blocked before final gate", "Rejected"} else "P2" if decision == "Watch only" else "P3"
        next_rows.append({
            "priority": priority,
            "ticker": ticker,
            "bridge_decision": decision,
            "what_to_do_next": next_step,
            "where_to_click": "Risk" if non_option_blocking[:1] in [["PM Review"], ["Risk"], ["Execution / Liquidity"]] else "News" if non_option_blocking[:1] == ["News / Event"] else "Risk",
            "why_it_matters": "PM review cannot override independent veto gates.",
            "source_files": "pm_review_final_gate_bridge.csv; pm_review_final_gate_veto_matrix.csv",
            "research_only": True,
        })

    bridge = pd.DataFrame(bridge_rows)
    veto = pd.DataFrame(veto_rows)
    next_actions = pd.DataFrame(next_rows)
    if not bridge.empty:
        order = {
            "Blocked before final gate": 0,
            "Rejected": 1,
            "Watch only": 2,
            "Tiny stock/ETF review candidate": 3,
            "Tiny paper review candidate": 4,
        }
        bridge["_rank"] = bridge["bridge_decision"].map(order).fillna(9)
        bridge = bridge.sort_values(["_rank", "proof_score_0_100", "ticker"], ascending=[True, False, True]).drop(columns=["_rank"]).reset_index(drop=True)
    if not next_actions.empty:
        next_actions["_rank"] = next_actions["priority"].map({"P1": 0, "P2": 1, "P3": 2}).fillna(9)
        next_actions = next_actions.sort_values(["_rank", "ticker"]).drop(columns=["_rank"]).reset_index(drop=True)

    state = {
        "date": today_str(),
        "status": "PM_REVIEW_FINAL_GATE_BRIDGE_ACTIVE",
        "ticker_count": int(len(bridge)),
        "blocked_before_final_gate_count": int((bridge.get("bridge_decision", pd.Series(dtype=str)) == "Blocked before final gate").sum()) if not bridge.empty else 0,
        "tiny_stock_etf_candidate_count": int((bridge.get("bridge_decision", pd.Series(dtype=str)) == "Tiny stock/ETF review candidate").sum()) if not bridge.empty else 0,
        "tiny_paper_candidate_count": int((bridge.get("bridge_decision", pd.Series(dtype=str)) == "Tiny paper review candidate").sum()) if not bridge.empty else 0,
        "options_allowed_count": int((bridge.get("options_allowed", pd.Series(dtype=str)) != "No").sum()) if not bridge.empty else 0,
        "veto_rows": int(len(veto)),
        "plain_answer": (
            f"PM review bridge is active. {len(bridge)} tickers were checked against independent veto gates. "
            f"{int((bridge.get('bridge_decision', pd.Series(dtype=str)) == 'Tiny stock/ETF review candidate').sum()) if not bridge.empty else 0} are non-option tiny review candidates, "
            f"and options allowed now remains {int((bridge.get('options_allowed', pd.Series(dtype=str)) != 'No').sum()) if not bridge.empty else 0}."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return bridge, veto, next_actions, state


def main() -> None:
    bridge, veto, next_actions, state = build_outputs()
    bridge.to_csv(OUT_BRIDGE, index=False)
    veto.to_csv(OUT_VETO, index=False)
    next_actions.to_csv(OUT_NEXT, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Bridge Decisions\n\n" + df_to_markdown(bridge.head(100)),
        "## Veto Matrix\n\n" + df_to_markdown(veto.head(200)),
        "## Next Actions\n\n" + df_to_markdown(next_actions.head(120)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 202 - PM Review Final Gate Bridge", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Tickers checked: {state['ticker_count']}")
    print(f"[OK] Blocked before final gate: {state['blocked_before_final_gate_count']}")
    print(f"[OK] Tiny stock/ETF candidates: {state['tiny_stock_etf_candidate_count']}")
    print(f"[OK] Options allowed now: {state['options_allowed_count']}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
