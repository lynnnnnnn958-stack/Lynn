#!/usr/bin/env python3
"""
Canyon v9 - Step 168: Institutional Strategy Thesis Engine
===========================================================

Research-only. No broker connection. No live orders.

This is not another shallow score table. Step168 turns the existing decision
router, risk budget, optimizer, evidence binder, events, conflicts, and option
clarity into an investment-committee style strategy thesis:

  - Which sleeve does the idea belong to?
  - What is the current strategy posture?
  - What is the base / bull / bear path?
  - What budget is allowed by risk and optimizer constraints?
  - Is the option expression call, put/hedge, stock/ETF, or no vehicle?
  - What has to happen before the idea can be upgraded?

Outputs:
  institutional_strategy_thesis_board.csv
  strategy_path_decision_tree.csv
  strategy_risk_budget_bridge.csv
  institutional_strategy_sleeve_book.csv
  institutional_strategy_action_playbook.csv
  strategy_exposure_overlap.csv
  institutional_strategy_state.json
  institutional_strategy_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    df_to_markdown,
    now_str,
    read_csv_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_HORIZON = ROOT / "horizon_vehicle_summary.csv"
IN_MATRIX = ROOT / "horizon_vehicle_matrix.csv"
IN_OPTION_CLARITY = ROOT / "option_route_clarity_board.csv"
IN_TARGET_WEIGHTS = ROOT / "institutional_target_weights.csv"
IN_OPTIMIZER = ROOT / "institutional_optimizer_bridge.csv"
IN_RISK_QUEUE = ROOT / "risk_desk_ticker_action_queue.csv"
IN_EVIDENCE = ROOT / "ticker_evidence_summary.csv"
IN_CONFLICT = ROOT / "decision_conflict_summary.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_SIGNAL_DOWNGRADE = ROOT / "signal_downgrade_queue.csv"
IN_WORKFLOW = ROOT / "daily_workflow_queue.csv"
IN_CORRELATION = ROOT / "holdings_correlation_matrix.csv"
IN_SECTOR_ACTIVE = ROOT / "sector_active_exposure.csv"
IN_OPTIMIZER_SECTOR = ROOT / "institutional_optimizer_sector_allocations.csv"

OUT_BOARD = ROOT / "institutional_strategy_thesis_board.csv"
OUT_TREE = ROOT / "strategy_path_decision_tree.csv"
OUT_RISK_BRIDGE = ROOT / "strategy_risk_budget_bridge.csv"
OUT_SLEEVE_BOOK = ROOT / "institutional_strategy_sleeve_book.csv"
OUT_ACTION_PLAYBOOK = ROOT / "institutional_strategy_action_playbook.csv"
OUT_OVERLAP = ROOT / "strategy_exposure_overlap.csv"
OUT_STATE = ROOT / "institutional_strategy_state.json"
OUT_REPORT = ROOT / "institutional_strategy_report.md"


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    raw = str(value).strip()
    return "" if raw.lower() == "nan" else raw


def upper(value: Any) -> str:
    return text(value).upper()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def pct(value: Any) -> str:
    out = safe_float(value, np.nan)
    return "N/A" if not np.isfinite(out) else f"{out:.2f}%"


def shorten(value: Any, limit: int = 760) -> str:
    raw = text(value)
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "..."


def first_nonempty(*values: Any) -> str:
    for value in values:
        raw = text(value)
        if raw:
            return raw
    return ""


def normalize_ticker(df: pd.DataFrame, column: str = "ticker") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[column] = out[column].astype(str).str.upper().str.strip()
    out = out[out[column] != ""]
    return out


def one_by_ticker(df: pd.DataFrame, column: str = "ticker") -> pd.DataFrame:
    out = normalize_ticker(df, column)
    if out.empty:
        return pd.DataFrame()
    return out.drop_duplicates(column, keep="first").set_index(column)


def row_at(indexed: pd.DataFrame, ticker: str) -> pd.Series:
    if indexed.empty or ticker not in indexed.index:
        return pd.Series(dtype=object)
    row = indexed.loc[ticker]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def signal_validation_map(signal_down: pd.DataFrame) -> dict[str, str]:
    if signal_down.empty or "signal" not in signal_down.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in signal_down.iterrows():
        signal = text(row.get("signal"))
        if signal:
            out[signal.upper()] = text(row.get("recommended_signal_action"))
    return out


def join_parts(parts: list[Any], sep: str = " | ", limit: int = 900) -> str:
    seen: list[str] = []
    for part in parts:
        raw = text(part)
        if raw and raw not in seen:
            seen.append(raw)
    return shorten(sep.join(seen), limit)


def choose_sleeve(horizon: pd.Series, optimizer: pd.Series, target: pd.Series) -> str:
    gate = upper(horizon.get("gate_status"))
    option_use = upper(horizon.get("option_use_case"))
    existing = first_nonempty(optimizer.get("sleeve"), target.get("sleeve"))
    if "NO NEW EXPOSURE" in gate:
        return "Risk Control"
    if "HEDGE" in option_use or "RISK REDUCTION" in option_use:
        return "Hedge / Risk Control"
    if existing:
        return existing
    if "LONG-TERM" in upper(horizon.get("best_horizon")):
        return "Core Watch"
    return "Tactical Watch"


def strategy_posture(horizon: pd.Series, option: pd.Series) -> str:
    gate = upper(horizon.get("gate_status"))
    use_case = upper(horizon.get("option_use_case"))
    if "NO NEW EXPOSURE" in gate:
        return "Capital preservation / de-risk"
    if "TINY PAPER" in gate:
        if "HEDGE" in use_case:
            return "Tiny paper plus hedge research"
        return "Tiny paper research only"
    if "NEWS" in gate or "SOURCE" in gate:
        return "Event confirmation watch"
    if "DEFINED-RISK CALL" in upper(option.get("call_status")):
        return "Tactical defined-risk call watch"
    if "PUT" in use_case or "HEDGE" in use_case:
        return "Protective hedge research"
    return "Research watch"


def confidence_tier(score: float, conflicts: float, risk_action: Any, signal_action: Any) -> tuple[str, float]:
    out = safe_float(score, 0.0)
    out -= min(18.0, safe_float(conflicts, 0.0) * 4.0)
    if "REDUCE_ONLY" in upper(risk_action):
        out = min(out, 20.0)
    elif "SIZE_DOWN" in upper(risk_action):
        out = min(out, 42.0)
    if upper(signal_action) in {"BLOCK_SIGNAL", "DOWNWEIGHT"}:
        out -= 10.0
    out = max(0.0, min(100.0, out))
    if out >= 70:
        return "High research conviction", round(out, 1)
    if out >= 50:
        return "Medium research conviction", round(out, 1)
    if out >= 30:
        return "Low conviction / monitor", round(out, 1)
    return "Observation or de-risk only", round(out, 1)


def budget_text(current: Any, risk_target: Any, optimizer_target: Any, gate: Any) -> str:
    gate_u = upper(gate)
    current_f = safe_float(current, np.nan)
    risk_f = safe_float(risk_target, np.nan)
    opt_f = safe_float(optimizer_target, np.nan)
    candidates = [x for x in [risk_f, opt_f] if np.isfinite(x)]
    cap = min(candidates) if candidates else np.nan
    if "NO NEW EXPOSURE" in gate_u:
        return f"No new exposure. Reduce from {pct(current_f)} toward risk/optimizer budget {pct(cap)}."
    if "TINY PAPER" in gate_u:
        return f"Tiny paper cap: use the lower of risk budget and optimizer target, about {pct(cap)}."
    return f"Research cap before manual approval: current {pct(current_f)}, risk {pct(risk_f)}, optimizer {pct(opt_f)}."


def base_case(horizon: pd.Series, budget: str) -> str:
    return shorten(
        f"{text(horizon.get('gate_status'))}. "
        f"{text(horizon.get('next_best_action'))}. "
        f"{budget}"
    )


def bull_case(horizon: pd.Series, option: pd.Series) -> str:
    return shorten(
        "Upgrade only if the unlock checklist clears, price confirms the trigger, "
        f"and the route moves beyond {text(horizon.get('gate_status'))}. "
        f"Call path: {text(option.get('call_status'))}. "
        f"Trigger: {text(horizon.get('trigger_to_watch'))}."
    )


def bear_case(horizon: pd.Series, option: pd.Series) -> str:
    return shorten(
        f"If invalidation or risk stress appears, keep de-risk/hedge first. "
        f"Put path: {text(option.get('put_status'))}. "
        f"Main blocker: {text(horizon.get('main_blocker'))}."
    )


def no_trade_case(horizon: pd.Series, option: pd.Series) -> str:
    return shorten(
        f"No-trade if: {text(horizon.get('unlock_checklist'))}. "
        f"No-go: {text(option.get('no_go_conditions'))}."
    )


def numeric_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def playbook_bucket(row: pd.Series) -> tuple[str, str, str, str]:
    posture = upper(row.get("strategy_posture"))
    gate = upper(row.get("gate_status"))
    option_expression = upper(row.get("option_expression"))
    veto = upper(row.get("veto_stack"))
    if "CAPITAL PRESERVATION" in posture or "NO NEW EXPOSURE" in gate:
        return (
            "1 Reduce / protect capital",
            "Reduce exposure toward the risk and optimizer budget before researching upside.",
            "New bullish exposure and standalone calls.",
            "L8 Risk",
        )
    if "TINY PAPER" in posture:
        if "HEDGE" in option_expression:
            return (
                "2 Tiny paper plus hedge research",
                "Keep only tiny stock/ETF paper sizing; research hedge only if it offsets book risk.",
                "Naked calls, weekly premium chase, and size increase.",
                "L8 Risk + L7 Options",
            )
        return (
            "3 Tiny paper watch",
            "Keep the idea on watch at tiny paper size only; require risk gate improvement before any upgrade.",
            "Full-size stock, call options, or thesis upgrade from score alone.",
            "L8 Risk",
        )
    if "EVENT" in posture or "L5" in veto:
        return (
            "4 Event/news confirmation",
            "Confirm event source, news reliability, price reaction, and volume before upgrading.",
            "Trading off headline direction alone.",
            "L5 Event / News",
        )
    if "CALL" in option_expression:
        return (
            "5 Defined-risk call research",
            "Research defined-risk call spread only after all manual gates clear.",
            "Naked long premium, oversized calls, and pre-confirmation entry.",
            "L7 Options + L9 Execution",
        )
    return (
        "6 Thesis build / monitor",
        "Build the thesis and wait for trigger, evidence, and risk budget confirmation.",
        "Action without source trail.",
        "Research Desk",
    )


def build_action_playbook(board: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if board.empty:
        return pd.DataFrame()
    for _, row in board.iterrows():
        bucket, first_move, prohibited, owner = playbook_bucket(row)
        score = safe_float(row.get("thesis_quality_score"), 0.0)
        current = safe_float(row.get("current_weight_pct"), 0.0)
        risk_target = safe_float(row.get("risk_target_weight_pct"), 0.0)
        reduction_need = max(0.0, current - risk_target)
        priority = (
            (0 if bucket.startswith("1") else 10 if bucket.startswith("2") else 20 if bucket.startswith("3") else 30)
            - min(8.0, reduction_need)
            - min(5.0, score / 20.0)
        )
        rows.append({
            "priority_rank": round(priority, 3),
            "action_bucket": bucket,
            "ticker": row.get("ticker", ""),
            "strategy_sleeve": row.get("strategy_sleeve", ""),
            "strategy_posture": row.get("strategy_posture", ""),
            "first_move": first_move,
            "why_now": row.get("current_strategy_thesis", ""),
            "prohibited_action": prohibited,
            "owner_layer": owner,
            "risk_budget_gap_pct": round(reduction_need, 3),
            "trigger_to_watch": row.get("trigger_to_watch", ""),
            "unlock_checklist": row.get("unlock_checklist", ""),
            "veto_stack": row.get("veto_stack", ""),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["priority_rank", "ticker"], ascending=[True, True]).reset_index(drop=True)
        out["priority"] = np.where(out.index < 5, "High", np.where(out.index < 12, "Medium", "Normal"))
    return out


def build_sleeve_book(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    work = board.copy()
    for col in ["current_weight_pct", "risk_target_weight_pct", "optimizer_target_weight_pct", "thesis_quality_score"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for (sleeve, posture), sub in work.groupby(["strategy_sleeve", "strategy_posture"], dropna=False):
        current_sum = numeric_sum(sub["current_weight_pct"])
        risk_sum = numeric_sum(sub["risk_target_weight_pct"])
        opt_sum = numeric_sum(sub["optimizer_target_weight_pct"])
        reduction_need = max(0.0, current_sum - min(risk_sum, opt_sum) if min(risk_sum, opt_sum) > 0 else current_sum - risk_sum)
        tickers = ", ".join(sub["ticker"].astype(str).head(12).tolist())
        avg_score = float(sub["thesis_quality_score"].mean()) if len(sub) else 0.0
        top_veto = first_nonempty(sub["veto_stack"].dropna().astype(str).head(1).iloc[0] if "veto_stack" in sub.columns and not sub["veto_stack"].dropna().empty else "")
        if "Risk Control" in text(sleeve) or "de-risk" in text(posture).lower():
            sleeve_action = "Reduce first; protect attention from upside stories."
        elif reduction_need > 1.0:
            sleeve_action = "Size down toward risk/optimizer budget before adding research."
        elif "hedge" in text(posture).lower():
            sleeve_action = "Keep hedge research tied to portfolio risk, not standalone bearish bets."
        else:
            sleeve_action = "Monitor thesis quality and wait for explicit unlocks."
        rows.append({
            "strategy_sleeve": sleeve,
            "strategy_posture": posture,
            "ticker_count": int(len(sub)),
            "tickers": tickers,
            "current_weight_sum_pct": round(current_sum, 3),
            "risk_target_sum_pct": round(risk_sum, 3),
            "optimizer_target_sum_pct": round(opt_sum, 3),
            "reduction_needed_pct": round(reduction_need, 3),
            "avg_thesis_quality_score": round(avg_score, 1),
            "call_now_count": int(sub.get("call_allowed_now", pd.Series(dtype=str)).astype(str).str.upper().eq("YES").sum()),
            "hedge_now_count": int(sub.get("put_or_hedge_allowed_now", pd.Series(dtype=str)).astype(str).str.upper().eq("YES").sum()),
            "sleeve_action": sleeve_action,
            "top_veto_stack": shorten(top_veto, 900),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["reduction_needed_pct", "ticker_count"], ascending=[False, False]).reset_index(drop=True)
    return out


def corr_pair_summary(corr: pd.DataFrame, tickers: list[str]) -> tuple[float, str]:
    if corr.empty or len(tickers) < 2:
        return np.nan, ""
    c = corr.copy()
    if "Unnamed: 0" in c.columns:
        c = c.rename(columns={"Unnamed: 0": "ticker"}).set_index("ticker")
    c.index = c.index.astype(str).str.upper()
    c.columns = [str(x).upper() for x in c.columns]
    vals: list[tuple[float, str]] = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            if a in c.index and b in c.columns:
                val = safe_float(c.loc[a, b], np.nan)
                if np.isfinite(val):
                    vals.append((abs(val), f"{a}-{b}:{val:.2f}"))
    if not vals:
        return np.nan, ""
    vals.sort(reverse=True)
    return vals[0][0], "; ".join([x[1] for x in vals[:5]])


def build_overlap_book(board: pd.DataFrame, risk_bridge: pd.DataFrame, corr: pd.DataFrame, sector_active: pd.DataFrame, optimizer_sector: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    merged = board.merge(
        risk_bridge[["ticker", "current_weight_pct", "risk_target_weight_pct", "optimizer_target_weight_pct"]],
        on="ticker",
        how="left",
        suffixes=("", "_risk_bridge"),
    )
    for col in ["current_weight_pct", "risk_target_weight_pct", "optimizer_target_weight_pct"]:
        merged[col] = pd.to_numeric(merged.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    sector_map = {}
    if not sector_active.empty and "sector" in sector_active.columns:
        sector_map = sector_active.set_index("sector").to_dict(orient="index")
    opt_sector_map = {}
    if not optimizer_sector.empty and "sector" in optimizer_sector.columns:
        opt_sector_map = optimizer_sector.set_index("sector").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for sector, sub in merged.groupby("sector", dropna=False):
        tickers = sub["ticker"].dropna().astype(str).str.upper().tolist()
        max_corr, corr_pairs = corr_pair_summary(corr, tickers)
        current_sum = numeric_sum(sub["current_weight_pct"])
        risk_sum = numeric_sum(sub["risk_target_weight_pct"])
        opt_sum = numeric_sum(sub["optimizer_target_weight_pct"])
        sector_info = sector_map.get(sector, {})
        opt_info = opt_sector_map.get(sector, {})
        duplicate_flag = "HIGH_OVERLAP" if len(tickers) >= 4 or (np.isfinite(max_corr) and max_corr >= 0.65) else "NORMAL"
        rows.append({
            "overlap_type": "Sector cluster",
            "group_key": sector,
            "tickers": ", ".join(tickers),
            "ticker_count": int(len(tickers)),
            "current_weight_sum_pct": round(current_sum, 3),
            "risk_target_sum_pct": round(risk_sum, 3),
            "optimizer_target_sum_pct": round(opt_sum, 3),
            "active_weight_pct": sector_info.get("active_weight_pct", ""),
            "sector_cap_used_pct": sector_info.get("cap_used_pct", ""),
            "optimizer_sector_weight_pct": opt_info.get("final_sector_weight_pct", ""),
            "max_pair_abs_corr": round(max_corr, 3) if np.isfinite(max_corr) else np.nan,
            "top_corr_pairs": corr_pairs,
            "overlap_flag": duplicate_flag,
            "book_action": "Treat as one risk expression; do not let multiple tickers bypass the same sector/correlation budget." if duplicate_flag == "HIGH_OVERLAP" else "Monitor normal cluster exposure.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).sort_values(["overlap_flag", "current_weight_sum_pct"], ascending=[True, False]).reset_index(drop=True)


def build_strategy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizon = one_by_ticker(read_csv_safe(IN_HORIZON))
    option = one_by_ticker(read_csv_safe(IN_OPTION_CLARITY))
    target = one_by_ticker(read_csv_safe(IN_TARGET_WEIGHTS))
    optimizer = one_by_ticker(read_csv_safe(IN_OPTIMIZER))
    risk = one_by_ticker(read_csv_safe(IN_RISK_QUEUE))
    evidence = one_by_ticker(read_csv_safe(IN_EVIDENCE))
    conflict = one_by_ticker(read_csv_safe(IN_CONFLICT))
    event = one_by_ticker(read_csv_safe(IN_EVENT))
    workflow = one_by_ticker(read_csv_safe(IN_WORKFLOW))
    matrix = normalize_ticker(read_csv_safe(IN_MATRIX))
    corr = read_csv_safe(IN_CORRELATION)
    sector_active = read_csv_safe(IN_SECTOR_ACTIVE)
    optimizer_sector = read_csv_safe(IN_OPTIMIZER_SECTOR)
    signal_actions = signal_validation_map(read_csv_safe(IN_SIGNAL_DOWNGRADE))

    tickers = list(horizon.index if not horizon.empty else [])
    board_rows: list[dict[str, Any]] = []
    tree_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for ticker in tickers:
        h = row_at(horizon, ticker)
        o = row_at(option, ticker)
        t = row_at(target, ticker)
        opt = row_at(optimizer, ticker)
        r = row_at(risk, ticker)
        ev = row_at(evidence, ticker)
        cf = row_at(conflict, ticker)
        event_row = row_at(event, ticker)
        wf = row_at(workflow, ticker)
        signal = first_nonempty(opt.get("top_signal"), event_row.get("top_signal"))
        signal_action = signal_actions.get(upper(signal), "")
        sleeve = choose_sleeve(h, opt, t)
        posture = strategy_posture(h, o)
        tier, thesis_score = confidence_tier(
            safe_float(h.get("decision_depth_score"), 0.0),
            safe_float(cf.get("high_conflicts"), safe_float(cf.get("conflict_count"), 0.0)),
            first_nonempty(h.get("risk_action"), r.get("final_risk_action")),
            signal_action,
        )

        current_weight = first_nonempty(r.get("current_weight_pct"), t.get("current_weight_pct"), opt.get("current_weight_pct"))
        risk_target = first_nonempty(r.get("recommended_risk_weight_pct"), t.get("target_weight_pct"), h.get("recommended_weight_pct"))
        optimizer_target = first_nonempty(opt.get("final_optimizer_weight_pct"), t.get("target_weight_pct"))
        budget = budget_text(current_weight, risk_target, optimizer_target, h.get("gate_status"))
        event_catalysts = first_nonempty(event_row.get("catalysts"))
        news_headline = first_nonempty(h.get("top_news_headline"))
        optimizer_status = first_nonempty(opt.get("final_optimizer_status"))
        option_expression = first_nonempty(h.get("option_use_case"))
        edge_stack = join_parts([
            f"Signal: {signal}" if signal else "",
            f"Signal validation: {signal_action}" if signal_action else "",
            f"Best horizon: {h.get('best_horizon')}",
            f"Sector: {h.get('sector')}",
            f"Event catalysts: {event_catalysts}" if event_catalysts else "",
            f"News: {news_headline}" if news_headline else "",
            f"Optimizer: {optimizer_status}" if optimizer_status else "",
            f"Option: {option_expression}" if option_expression else "",
        ])
        veto_stack = join_parts([
            h.get("override_stack"),
            cf.get("top_conflict"),
            cf.get("first_resolution"),
            r.get("reason_stack"),
            opt.get("binding_constraints"),
            event_row.get("risks"),
        ], sep=" > ")
        thesis = shorten(
            f"{ticker} belongs in {sleeve}. Current posture: {posture}. "
            f"Primary question: {h.get('primary_research_question')}. "
            f"Budget: {budget}"
        )

        board_rows.append({
            "ticker": ticker,
            "sector": h.get("sector", ""),
            "strategy_sleeve": sleeve,
            "strategy_posture": posture,
            "conviction_tier": tier,
            "thesis_quality_score": thesis_score,
            "best_horizon": h.get("best_horizon", ""),
            "horizon_consensus": h.get("horizon_consensus", ""),
            "gate_status": h.get("gate_status", ""),
            "current_strategy_thesis": thesis,
            "base_case": base_case(h, budget),
            "bull_case": bull_case(h, o),
            "bear_case": bear_case(h, o),
            "no_trade_case": no_trade_case(h, o),
            "position_budget": budget,
            "current_weight_pct": current_weight,
            "risk_target_weight_pct": risk_target,
            "optimizer_target_weight_pct": optimizer_target,
            "option_expression": h.get("option_use_case", ""),
            "call_allowed_now": o.get("call_allowed_now", ""),
            "put_or_hedge_allowed_now": o.get("put_or_hedge_allowed_now", ""),
            "call_status": o.get("call_status", h.get("call_status", "")),
            "put_status": o.get("put_status", h.get("put_status", "")),
            "trigger_to_watch": h.get("trigger_to_watch", ""),
            "unlock_checklist": h.get("unlock_checklist", ""),
            "edge_stack": edge_stack,
            "veto_stack": veto_stack,
            "evidence_rows": ev.get("evidence_rows", ""),
            "high_conflicts": cf.get("high_conflicts", ""),
            "event_gate": h.get("event_gate", ""),
            "news_reliability_status": h.get("news_reliability_status", ""),
            "source_files": join_parts([
                "horizon_vehicle_summary.csv",
                "option_route_clarity_board.csv",
                "institutional_optimizer_bridge.csv",
                "risk_desk_ticker_action_queue.csv",
                "ticker_evidence_summary.csv",
                "decision_conflict_summary.csv",
                "event_research_dossier.csv",
            ]),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        scenarios = [
            ("Now / Base", base_case(h, budget), "Keep current gate discipline", h.get("short_vehicle", "")),
            ("Bull Unlock", bull_case(h, o), h.get("unlock_checklist", ""), "Stock/ETF or defined-risk call after gates clear"),
            ("Bear / Hedge", bear_case(h, o), h.get("main_blocker", ""), "Put/hedge only if tied to portfolio risk"),
            ("No-Trade / Kill", no_trade_case(h, o), o.get("no_go_conditions", ""), "No vehicle"),
        ]
        for order, (scenario, path, condition, vehicle) in enumerate(scenarios, start=1):
            tree_rows.append({
                "ticker": ticker,
                "scenario_order": order,
                "scenario": scenario,
                "strategy_sleeve": sleeve,
                "path_action": path,
                "conditions_required": condition,
                "vehicle_expression": vehicle,
                "budget_rule": budget,
                "risk_gate": h.get("gate_status", ""),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })

        ticker_matrix = matrix[matrix["ticker"].astype(str).str.upper() == ticker] if not matrix.empty else pd.DataFrame()
        risk_rows.append({
            "ticker": ticker,
            "sector": h.get("sector", ""),
            "strategy_sleeve": sleeve,
            "current_weight_pct": current_weight,
            "risk_target_weight_pct": risk_target,
            "optimizer_target_weight_pct": optimizer_target,
            "final_risk_action": first_nonempty(r.get("final_risk_action"), h.get("risk_action")),
            "risk_reduction_pct_of_current": r.get("risk_reduction_pct_of_current", ""),
            "var_95_1d": r.get("var_95_1d", ""),
            "cvar_95_1d": r.get("cvar_95_1d", ""),
            "max_abs_corr_to_book": opt.get("max_abs_corr_to_book", ""),
            "binding_constraints": opt.get("binding_constraints", ""),
            "optimizer_status": opt.get("final_optimizer_status", ""),
            "horizon_confidence_min": pd.to_numeric(ticker_matrix.get("route_confidence_score", pd.Series(dtype=float)), errors="coerce").min() if not ticker_matrix.empty else np.nan,
            "horizon_confidence_max": pd.to_numeric(ticker_matrix.get("route_confidence_score", pd.Series(dtype=float)), errors="coerce").max() if not ticker_matrix.empty else np.nan,
            "budget_interpretation": budget,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    board = pd.DataFrame(board_rows)
    tree = pd.DataFrame(tree_rows)
    risk_bridge = pd.DataFrame(risk_rows)
    if not board.empty:
        posture_order = {
            "Capital preservation / de-risk": 0,
            "Tiny paper plus hedge research": 1,
            "Tiny paper research only": 2,
            "Event confirmation watch": 3,
            "Protective hedge research": 4,
            "Tactical defined-risk call watch": 5,
            "Research watch": 6,
        }
        board["_posture_rank"] = board["strategy_posture"].map(posture_order).fillna(9)
        board["_score"] = pd.to_numeric(board["thesis_quality_score"], errors="coerce").fillna(0)
        board = board.sort_values(["_posture_rank", "_score", "ticker"], ascending=[True, False, True]).drop(columns=["_posture_rank", "_score"]).reset_index(drop=True)
    sleeve_book = build_sleeve_book(board)
    action_playbook = build_action_playbook(board)
    overlap = build_overlap_book(board, risk_bridge, corr, sector_active, optimizer_sector)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "status": "READY" if len(board) else "NO_ROWS",
        "strategy_rows": int(len(board)),
        "decision_tree_rows": int(len(tree)),
        "risk_bridge_rows": int(len(risk_bridge)),
        "sleeve_book_rows": int(len(sleeve_book)),
        "action_playbook_rows": int(len(action_playbook)),
        "overlap_rows": int(len(overlap)),
        "avg_thesis_quality_score": round(float(pd.to_numeric(board.get("thesis_quality_score", pd.Series(dtype=float)), errors="coerce").mean()), 1) if not board.empty else 0.0,
        "de_risk_count": int(board.get("strategy_posture", pd.Series(dtype=str)).astype(str).str.contains("de-risk|Capital preservation", case=False, na=False).sum()) if not board.empty else 0,
        "tiny_research_count": int(board.get("strategy_posture", pd.Series(dtype=str)).astype(str).str.contains("Tiny paper", case=False, na=False).sum()) if not board.empty else 0,
        "hedge_research_count": int(board.get("strategy_posture", pd.Series(dtype=str)).astype(str).str.contains("hedge", case=False, na=False).sum()) if not board.empty else 0,
        "call_now_count": int(board.get("call_allowed_now", pd.Series(dtype=str)).astype(str).str.upper().eq("YES").sum()) if not board.empty else 0,
        "high_overlap_count": int(overlap.get("overlap_flag", pd.Series(dtype=str)).astype(str).str.upper().eq("HIGH_OVERLAP").sum()) if not overlap.empty else 0,
        "outputs": {
            "board": OUT_BOARD.name,
            "tree": OUT_TREE.name,
            "risk_bridge": OUT_RISK_BRIDGE.name,
            "sleeve_book": OUT_SLEEVE_BOOK.name,
            "action_playbook": OUT_ACTION_PLAYBOOK.name,
            "overlap": OUT_OVERLAP.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return board, tree, risk_bridge, sleeve_book, action_playbook, overlap, state


def main() -> int:
    board, tree, risk_bridge, sleeve_book, action_playbook, overlap, state = build_strategy()
    board.to_csv(OUT_BOARD, index=False)
    tree.to_csv(OUT_TREE, index=False)
    risk_bridge.to_csv(OUT_RISK_BRIDGE, index=False)
    sleeve_book.to_csv(OUT_SLEEVE_BOOK, index=False)
    action_playbook.to_csv(OUT_ACTION_PLAYBOOK, index=False)
    overlap.to_csv(OUT_OVERLAP, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Strategy rows: {state.get('strategy_rows', 0)}",
        f"- Decision tree rows: {state.get('decision_tree_rows', 0)}",
        f"- Risk bridge rows: {state.get('risk_bridge_rows', 0)}",
        f"- Sleeve book rows: {state.get('sleeve_book_rows', 0)}",
        f"- Action playbook rows: {state.get('action_playbook_rows', 0)}",
        f"- Overlap rows: {state.get('overlap_rows', 0)}",
        f"- Avg thesis quality score: {state.get('avg_thesis_quality_score', 0)}",
        f"- De-risk count: {state.get('de_risk_count', 0)}",
        f"- Tiny research count: {state.get('tiny_research_count', 0)}",
        f"- Hedge research count: {state.get('hedge_research_count', 0)}",
        f"- Call allowed now: {state.get('call_now_count', 0)}",
        f"- High overlap count: {state.get('high_overlap_count', 0)}",
        "",
        "## Strategy Sleeve Book",
        "",
        df_to_markdown(sleeve_book, max_rows=80),
        "",
        "## Strategy Action Playbook",
        "",
        df_to_markdown(action_playbook, max_rows=120),
        "",
        "## Strategy Exposure Overlap",
        "",
        df_to_markdown(overlap, max_rows=80),
        "",
        "## Strategy Thesis Board",
        "",
        df_to_markdown(board, max_rows=80),
        "",
        "## Strategy Path Decision Tree",
        "",
        df_to_markdown(tree, max_rows=160),
        "",
        "## Strategy Risk Budget Bridge",
        "",
        df_to_markdown(risk_bridge, max_rows=80),
        "",
        "## Product Truth",
        "",
        "These are research theses and scenario paths only. No broker connection, no live orders, no trade approval.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 168 - Institutional Strategy Thesis", sections)

    print(f"wrote {OUT_BOARD.name} rows={len(board)}")
    print(f"wrote {OUT_TREE.name} rows={len(tree)}")
    print(f"wrote {OUT_RISK_BRIDGE.name} rows={len(risk_bridge)}")
    print(f"wrote {OUT_SLEEVE_BOOK.name} rows={len(sleeve_book)}")
    print(f"wrote {OUT_ACTION_PLAYBOOK.name} rows={len(action_playbook)}")
    print(f"wrote {OUT_OVERLAP.name} rows={len(overlap)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
