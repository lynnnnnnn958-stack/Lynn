#!/usr/bin/env python3
"""
Canyon v9 Step 200 - Risk Seed Approval Workbench.

Research-only. No broker connection. No live orders.

Step199 created provisional risk-book seed entries. Step200 ranks those seed
entries for human review and explains exactly why a ticker is still blocked.
It does not approve anything by itself.

Outputs:
  risk_seed_approval_state.json
  risk_seed_approval_rank.csv
  risk_seed_approval_packets.csv
  risk_seed_blocker_matrix.csv
  risk_seed_promotion_simulation.csv
  risk_seed_approval_report.md
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


OUT_STATE = ROOT / "risk_seed_approval_state.json"
OUT_RANK = ROOT / "risk_seed_approval_rank.csv"
OUT_PACKETS = ROOT / "risk_seed_approval_packets.csv"
OUT_BLOCKERS = ROOT / "risk_seed_blocker_matrix.csv"
OUT_SIM = ROOT / "risk_seed_promotion_simulation.csv"
OUT_REPORT = ROOT / "risk_seed_approval_report.md"


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


def short(value: Any, limit: int = 260) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def plain(value: Any, default: str = "No data") -> str:
    text = as_text(value, default)
    replacements = {
        "DATA_GAP": "missing data",
        "REDUCE_ONLY": "reduce only",
        "SIZE_DOWN": "use smaller size",
        "SEED_REVIEW_ONLY": "seed review only",
        "NO_KELLY_UNTIL_LIVE_IC": "no Kelly sizing until live signal skill is proven",
        "MANUAL_REVIEW": "manual review",
        "CALL_RESEARCH_ONLY": "call research only",
        "STOCK_OR_ETF_RESEARCH_ONLY": "stock or ETF research only",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    return " ".join(text.replace("_", " ").split())


def one_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""].copy()
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby(ticker_col, sort=False)}


def counts_by_ticker(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if df.empty or "ticker" not in df.columns:
        return {}
    work = df.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()
    out: dict[str, dict[str, int]] = {}
    for ticker, grp in work.groupby("ticker", sort=False):
        priorities = grp.get("priority", pd.Series("", index=grp.index)).astype(str).str.upper()
        out[ticker] = {
            "total": int(len(grp)),
            "p1": int((priorities == "P1").sum()),
            "p2": int((priorities == "P2").sum()),
        }
    return out


def risk_penalty(risk_level: str) -> int:
    level = risk_level.lower()
    if "very high" in level:
        return 35
    if "high" in level:
        return 22
    if "medium" in level:
        return 12
    if "lower" in level:
        return 5
    return 28


def liquidity_penalty(liquidity: str) -> int:
    text = liquidity.lower()
    if "good" in text:
        return 0
    if "usable" in text:
        return 5
    if "manual" in text or "needs" in text:
        return 15
    return 10


def approval_lane(score: float, risk_level: str, news_p1: int, execution_p1: int, sector: str) -> tuple[str, str]:
    risk = risk_level.lower()
    if "very high" in risk:
        return "High-risk sandbox only", "Review only after downside scenario and stop rule are explicit."
    if news_p1 > 0:
        return "News proof first", "A P1 news or causal proof item blocks approval."
    if execution_p1 > 0:
        return "Execution proof first", "A P1 execution or spread item blocks approval."
    if sector.lower() == "unknown":
        return "Classify sector first", "Sector/theme is unknown, so crowding cannot be judged."
    if score >= 70:
        return "Ready for PM review", "This seed has enough first-pass risk facts for human review."
    if score >= 50:
        return "Review after proof", "Useful candidate, but proof gaps still matter."
    return "Backlog", "Too many proof or risk gaps for near-term approval."


def make_blockers(ticker: str, row: pd.Series, metric: pd.Series | None, news_count: dict[str, int], exec_count: dict[str, int]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    sector = as_text(row.get("sector_or_theme"), "Unknown")
    risk_level = as_text(row.get("risk_level"), "Unknown")
    liquidity = as_text(metric.get("liquidity_status") if metric is not None else row.get("liquidity_status"), "Needs manual liquidity proof")
    manual_items = as_text(row.get("manual_items_open"))

    blockers.append({
        "ticker": ticker,
        "blocker_type": "PM approval",
        "severity": "High",
        "plain_blocker": "A human has not approved this risk seed.",
        "what_to_collect": "PM approval note that accepts max seed cap, stop rule, and forbidden option route.",
        "source_files": "risk_book_seed_manual_approval_queue.csv",
    })

    if "Earnings date" in manual_items:
        blockers.append({
            "ticker": ticker,
            "blocker_type": "Earnings gap",
            "severity": "High",
            "plain_blocker": "Earnings date and expected gap risk are not sourced.",
            "what_to_collect": "Next earnings date, expected move, and whether size must be flat/reduced before the event.",
            "source_files": "risk_book_seed_manual_approval_queue.csv",
        })

    if sector.lower() == "unknown":
        blockers.append({
            "ticker": ticker,
            "blocker_type": "Sector classification",
            "severity": "Medium",
            "plain_blocker": "Sector or theme is unknown.",
            "what_to_collect": "Map the ticker to sector, subindustry, theme, and peer group.",
            "source_files": "risk_book_seed_metric_detail.csv",
        })

    if "very high" in risk_level.lower():
        blockers.append({
            "ticker": ticker,
            "blocker_type": "Downside risk",
            "severity": "High",
            "plain_blocker": "Tail risk is very high.",
            "what_to_collect": "Downside scenario, hard paper stop, and reason why this is worth reviewing despite high volatility.",
            "source_files": "risk_book_seed_metric_detail.csv",
        })

    if "manual" in liquidity.lower() or "needs" in liquidity.lower():
        blockers.append({
            "ticker": ticker,
            "blocker_type": "Liquidity",
            "severity": "Medium",
            "plain_blocker": "Liquidity or spread still needs manual proof.",
            "what_to_collect": "Current bid/ask spread, average dollar volume, and realistic fill assumption.",
            "source_files": "risk_book_seed_metric_detail.csv",
        })

    if news_count.get("p1", 0) > 0 or news_count.get("total", 0) > 0:
        blockers.append({
            "ticker": ticker,
            "blocker_type": "News proof",
            "severity": "High" if news_count.get("p1", 0) > 0 else "Medium",
            "plain_blocker": f"{news_count.get('total', 0)} news proof item(s) remain open.",
            "what_to_collect": "Source timestamp, affected ticker, causal link, and post-news price/volume reaction.",
            "source_files": "news_proof_repair_queue.csv",
        })

    if exec_count.get("p1", 0) > 0 or exec_count.get("total", 0) > 0:
        blockers.append({
            "ticker": ticker,
            "blocker_type": "Execution proof",
            "severity": "High" if exec_count.get("p1", 0) > 0 else "Medium",
            "plain_blocker": f"{exec_count.get('total', 0)} execution/spread proof item(s) remain open.",
            "what_to_collect": "Spread, liquidity, expected cost, stress cost, and option route proof if relevant.",
            "source_files": "execution_spread_repair_queue.csv",
        })

    return blockers


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    approval = read_csv_safe(ROOT / "risk_book_seed_manual_approval_queue.csv")
    metrics = read_csv_safe(ROOT / "risk_book_seed_metric_detail.csv")
    gate = read_csv_safe(ROOT / "institutional_promotion_gate.csv")
    news = read_csv_safe(ROOT / "news_proof_repair_queue.csv")
    execution = read_csv_safe(ROOT / "execution_spread_repair_queue.csv")

    if approval.empty:
        empty = pd.DataFrame()
        state = {
            "date": today_str(),
            "status": "NO_RISK_SEED_APPROVAL_QUEUE",
            "plain_answer": "No risk seed approval queue exists yet.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return empty, empty, empty, empty, state

    metric_map = one_by_ticker(metrics)
    gate_map = one_by_ticker(gate)
    news_counts = counts_by_ticker(news)
    exec_counts = counts_by_ticker(execution)

    rank_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []

    for _, row in approval.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        metric = metric_map.get(ticker)
        gate_row = gate_map.get(ticker)
        news_count = news_counts.get(ticker, {"total": 0, "p1": 0, "p2": 0})
        exec_count = exec_counts.get(ticker, {"total": 0, "p1": 0, "p2": 0})

        sector = as_text(row.get("sector_or_theme"), "Unknown")
        risk_level = as_text(row.get("risk_level"), "Unknown")
        liquidity = as_text(metric.get("liquidity_status") if metric is not None else "", "Needs manual liquidity proof")
        cap = safe_float(row.get("starter_cap_if_approved_pct"), 0.0)
        stop = safe_float(row.get("paper_stop_if_ever_tested_pct"), np.nan)
        cvar = safe_float(metric.get("daily_cvar_95_pct") if metric is not None else np.nan, np.nan)
        corr_spy = safe_float(metric.get("corr_spy") if metric is not None else np.nan, np.nan)
        corr_qqq = safe_float(metric.get("corr_qqq") if metric is not None else np.nan, np.nan)
        corr_smh = safe_float(metric.get("corr_smh") if metric is not None else np.nan, np.nan)
        max_corr = np.nanmax([abs(x) for x in [corr_spy, corr_qqq, corr_smh] if np.isfinite(x)]) if any(np.isfinite(x) for x in [corr_spy, corr_qqq, corr_smh]) else np.nan

        score = 100
        score -= risk_penalty(risk_level)
        score -= liquidity_penalty(liquidity)
        score -= 25 if news_count["p1"] else 12 if news_count["total"] else 0
        score -= 20 if exec_count["p1"] else 10 if exec_count["total"] else 0
        score -= 8 if sector.lower() == "unknown" else 0
        score -= 7 if np.isfinite(max_corr) and max_corr >= 0.75 else 0
        score -= 5 if np.isfinite(stop) and stop >= 12 else 0
        score = round(float(np.clip(score, 0, 100)), 1)

        lane, lane_reason = approval_lane(score, risk_level, news_count["p1"], exec_count["p1"], sector)
        blockers = make_blockers(ticker, row, metric, news_count, exec_count)
        blocker_rows.extend(blockers)

        first_blocker = blockers[0]["blocker_type"] if blockers else "Final PM check"
        open_blockers = len(blockers)
        if lane == "Ready for PM review":
            next_step = "Review the seed cap, stop rule, earnings date, spread, and sector crowding in one PM pass."
        elif lane == "News proof first":
            next_step = "Clear P1 news proof before spending time on sizing."
        elif lane == "Execution proof first":
            next_step = "Clear execution and spread proof before sizing discussion."
        elif lane == "Classify sector first":
            next_step = "Map sector/theme/peer group before judging crowding."
        elif lane == "High-risk sandbox only":
            next_step = "Write downside scenario first; keep any future test tiny or skip."
        else:
            next_step = "Keep in backlog until proof gaps shrink."

        rank_rows.append({
            "ticker": ticker,
            "sector_or_theme": sector,
            "approval_lane": lane,
            "approval_score_0_100": score,
            "why_this_lane": lane_reason,
            "risk_level": risk_level,
            "daily_cvar_95_pct": cvar,
            "starter_cap_if_approved_pct": cap,
            "paper_stop_if_ever_tested_pct": stop,
            "liquidity_status": liquidity,
            "news_proof_open": news_count["total"],
            "execution_proof_open": exec_count["total"],
            "open_blocker_count": open_blockers,
            "first_blocker": first_blocker,
            "next_step": next_step,
            "still_forbidden": "No paper size, no calls, no puts, no live orders until all proof and PM approval clear.",
            "source_files": "risk_book_seed_manual_approval_queue.csv; risk_book_seed_metric_detail.csv; news/execution repair queues",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        final_gate_now = as_text(gate_row.get("final_permission") if gate_row is not None else "", "Study only")
        packet_rows.append({
            "ticker": ticker,
            "plain_answer": f"{ticker}: {lane}. Not approved for paper or options.",
            "why_review": f"Risk level {risk_level}; seed cap if ever approved {cap}%; stop {stop if np.isfinite(stop) else 'not set'}%.",
            "what_to_check_first": next_step,
            "news_and_execution": f"News proof open: {news_count['total']}; execution/spread proof open: {exec_count['total']}.",
            "current_final_gate": final_gate_now,
            "current_first_blocker": as_text(gate_row.get("first_blocker") if gate_row is not None else "", "Risk seed approval"),
            "option_rule": "Options remain blocked. Do not look for calls or puts until PM approval, spread proof, IV check, and Final PM Gate clear.",
            "source_files": "risk_seed_approval_rank.csv; institutional_promotion_gate.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        if lane == "Ready for PM review":
            simulated = "Would still be study-only until earnings, spread, news, and execution proof are attached."
        elif lane in {"News proof first", "Execution proof first", "Classify sector first"}:
            simulated = f"Seed approval alone would not help much; {lane.lower()} remains the real blocker."
        elif lane == "High-risk sandbox only":
            simulated = "If ever approved, only tiny defined-risk paper review after downside scenario. Options still blocked."
        else:
            simulated = "Backlog. Approval is not the next useful action."
        sim_rows.append({
            "ticker": ticker,
            "current_final_gate": final_gate_now,
            "approval_lane": lane,
            "if_seed_approved_next_state": simulated,
            "max_seed_cap_if_all_manual_gates_clear_pct": cap,
            "still_blocks_after_seed_approval": "; ".join(sorted({b["blocker_type"] for b in blockers if b["blocker_type"] != "PM approval"})) or "Final manual check",
            "option_after_seed_approval": "Still blocked until option spread, IV, liquidity, risk, and event proof all clear.",
            "source_files": "risk_seed_approval_rank.csv; risk_seed_blocker_matrix.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    rank = pd.DataFrame(rank_rows)
    if not rank.empty:
        lane_order = {
            "Ready for PM review": 0,
            "Review after proof": 1,
            "News proof first": 2,
            "Execution proof first": 3,
            "Classify sector first": 4,
            "High-risk sandbox only": 5,
            "Backlog": 6,
        }
        rank["_lane_rank"] = rank["approval_lane"].map(lane_order).fillna(9)
        rank = rank.sort_values(["_lane_rank", "approval_score_0_100", "ticker"], ascending=[True, False, True]).drop(columns=["_lane_rank"]).reset_index(drop=True)

    packets = pd.DataFrame(packet_rows)
    blockers = pd.DataFrame(blocker_rows)
    sim = pd.DataFrame(sim_rows)
    if not blockers.empty:
        severity_rank = {"High": 0, "Medium": 1, "Low": 2}
        blockers["_rank"] = blockers["severity"].map(severity_rank).fillna(9)
        blockers = blockers.sort_values(["_rank", "ticker", "blocker_type"]).drop(columns=["_rank"]).reset_index(drop=True)

    lane_counts = rank.get("approval_lane", pd.Series(dtype=str)).value_counts().to_dict() if not rank.empty else {}
    state = {
        "date": today_str(),
        "status": "RISK_SEED_APPROVAL_WORKBENCH_ACTIVE",
        "seed_count": len(rank),
        "ready_for_pm_review_count": int(lane_counts.get("Ready for PM review", 0)),
        "news_proof_first_count": int(lane_counts.get("News proof first", 0)),
        "execution_proof_first_count": int(lane_counts.get("Execution proof first", 0)),
        "high_risk_sandbox_count": int(lane_counts.get("High-risk sandbox only", 0)),
        "backlog_count": int(lane_counts.get("Backlog", 0)),
        "blocker_rows": len(blockers),
        "plain_answer": (
            f"Risk seed approval workbench is active. {len(rank)} seeds were ranked. "
            f"{int(lane_counts.get('Ready for PM review', 0))} are closest to PM review, but none are approved for paper or options."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return rank, packets, blockers, sim, state


def main() -> None:
    rank, packets, blockers, sim, state = build_outputs()
    rank.to_csv(OUT_RANK, index=False)
    packets.to_csv(OUT_PACKETS, index=False)
    blockers.to_csv(OUT_BLOCKERS, index=False)
    sim.to_csv(OUT_SIM, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Approval Rank\n\n" + df_to_markdown(rank.head(80)),
        "## Approval Packets\n\n" + df_to_markdown(packets.head(80)),
        "## Blocker Matrix\n\n" + df_to_markdown(blockers.head(160)),
        "## Promotion Simulation\n\n" + df_to_markdown(sim.head(80)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 200 - Risk Seed Approval Workbench", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Seeds ranked: {state['seed_count']}")
    print(f"[OK] Ready for PM review: {state['ready_for_pm_review_count']}")
    print(f"[OK] Blocker rows: {state['blocker_rows']}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
