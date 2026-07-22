#!/usr/bin/env python3
"""
Canyon v9 Sector And Theme Depth Desk.

This is a workstream-deepening module, not another numbered trading step.

Purpose:
  - turn broad sector labels into a PM-style thesis
  - separate semiconductor leadership from late-cycle chase risk
  - separate software catch-up/handoff watch from actual permission
  - connect theme/news read-through targets to current risk gates

Research-only. No broker connection. No live orders.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_THESIS = ROOT / "sector_theme_depth_thesis.csv"
OUT_HANDOFF = ROOT / "sector_theme_handoff_matrix.csv"
OUT_TICKER_MAP = ROOT / "sector_theme_depth_ticker_map.csv"
OUT_SOURCE_GUIDE = ROOT / "sector_theme_depth_source_guide.csv"
OUT_STATE = ROOT / "sector_theme_depth_state.json"
OUT_REPORT = ROOT / "sector_theme_depth_report.md"


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
        out = float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def compact(value: Any, limit: int = 360) -> str:
    text = " ".join(as_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def mode_text(series: pd.Series, default: str = "No data") -> str:
    values = [as_text(x) for x in series.dropna().tolist()]
    values = [v for v in values if v]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def ticker_list(series: pd.Series, limit: int = 12) -> str:
    values = []
    for item in series.dropna().astype(str).tolist():
        for token in item.replace(";", ",").split(","):
            token = token.strip().upper()
            if token and token not in values:
                values.append(token)
    shown = values[:limit]
    if len(values) > limit:
        shown.append(f"+{len(values) - limit} more")
    return ", ".join(shown) if shown else "NO_TICKERS"


def numeric_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def numeric_sum(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.sum()) if not vals.empty else np.nan


def stance_from_phase(phase: str, risk_actions: list[str], event_gates: list[str]) -> tuple[str, str, str]:
    phase_l = phase.lower()
    risk_join = " ".join(risk_actions).upper()
    event_join = " ".join(event_gates).upper()
    if "semiconductor" in phase_l:
        pass
    risk_block = any(x in risk_join for x in ["REDUCE_ONLY", "SIZE_DOWN", "BLOCK", "NO_NEW"])
    event_review = any(x in event_join for x in ["REVIEW", "MISSING"])
    if "late-cycle" in phase_l or "chase risk" in phase_l:
        return (
            "Leader but do not chase",
            "Semis still show price leadership, but the local evidence treats fresh chase as late-cycle risk.",
            "Only de-risk, wait for pullback, or hedge research after risk/source/spread gates clear.",
        )
    if "catch-up handoff" in phase_l:
        return (
            "Constructive handoff watch",
            "Software is not automatic buy permission; it is a handoff candidate that needs proof.",
            "Watch software relative strength, event proof, and risk repair before route promotion.",
        )
    if "early improvement" in phase_l:
        return (
            "Early improvement watch",
            "The group may be improving, but the evidence is not yet a full-cycle leadership claim.",
            "Require breadth, price confirmation, and source validation.",
        )
    if "downcycle" in phase_l or "laggard" in phase_l:
        return (
            "Avoid bullish chase",
            "The group is lagging or downcycle in the local evidence stack.",
            "Only defensive or tiny research review after risk gates clear.",
        )
    if risk_block:
        return (
            "Risk controls the thesis",
            "The sector story is secondary because risk gates are still blocking or size-down.",
            "Repair risk before promoting the group.",
        )
    if event_review:
        return (
            "Event proof needed",
            "The group needs cleaner event/source evidence before upgrade.",
            "Validate event timing, ticker mapping, and price reaction.",
        )
    return (
        "Neutral research watch",
        "The group has context but not enough local proof for a strong thesis.",
        "Keep it as watchlist evidence until a clearer catalyst appears.",
    )


def thesis_score(phase: str, ret20: float, ret63: float, positive: float, negative: float, risk_actions: list[str]) -> float:
    score = 50.0
    if np.isfinite(ret20):
        score += np.clip(ret20, -20, 20) * 0.5
    if np.isfinite(ret63):
        score += np.clip(ret63, -40, 40) * 0.25
    score += min(positive, 20) * 0.8
    score -= min(negative, 20) * 1.5
    phase_l = phase.lower()
    if "catch-up handoff" in phase_l:
        score += 8
    if "early improvement" in phase_l:
        score += 6
    if "late-cycle" in phase_l or "chase risk" in phase_l:
        score -= 10
    if "downcycle" in phase_l or "laggard" in phase_l:
        score -= 12
    risk_join = " ".join(risk_actions).upper()
    if "REDUCE_ONLY" in risk_join:
        score -= 12
    if "SIZE_DOWN" in risk_join:
        score -= 8
    return round(float(np.clip(score, 0, 100)), 1)


def build_current_book_thesis(subsector: pd.DataFrame, route: pd.DataFrame, optimizer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if subsector.empty or "subsector" not in subsector.columns:
        return pd.DataFrame(rows)
    for group, grp in subsector.groupby("subsector", dropna=False):
        group_name = as_text(group, "Unknown")
        phase = mode_text(grp.get("subsector_cycle_phase", pd.Series(dtype=str)))
        risk_actions = grp.get("risk_action", pd.Series(dtype=str)).dropna().astype(str).tolist()
        event_gates = grp.get("event_gate", pd.Series(dtype=str)).dropna().astype(str).tolist()
        stance, evidence, route_policy = stance_from_phase(phase, risk_actions, event_gates)
        current_weight = numeric_sum(grp, "current_weight_pct")
        ret20 = numeric_mean(grp, "ret_20d_pct")
        ret63 = numeric_mean(grp, "ret_63d_pct")
        positive = 0.0
        negative = 0.0
        if "top_headline" in grp.columns:
            positive = float(grp["top_headline"].dropna().astype(str).ne("").sum())
        if "subsector_why" in grp.columns:
            negative = float(grp["subsector_why"].astype(str).str.contains("weak|late-cycle|chase", case=False, na=False).sum())
        score = thesis_score(phase, ret20, ret63, positive, negative, risk_actions)
        opt_rows = route[route.get("subsector", pd.Series(dtype=str)).astype(str).eq(group_name)] if not route.empty and "subsector" in route.columns else pd.DataFrame()
        opt_summary = mode_text(opt_rows.get("option_permission_overlay", pd.Series(dtype=str)), "Risk gate controls option route")
        top_headline = mode_text(grp.get("top_headline", pd.Series(dtype=str)), "No mapped headline")
        opt_bridge = optimizer[optimizer.get("subsector", pd.Series(dtype=str)).astype(str).eq(group_name)] if not optimizer.empty and "subsector" in optimizer.columns else pd.DataFrame()
        why_not_more = mode_text(opt_bridge.get("why_not_more", pd.Series(dtype=str)), "No optimizer bridge row")
        rows.append({
            "thesis_rank": 0,
            "theme_or_subsector": group_name,
            "source_type": "current_risk_book",
            "cycle_thesis": phase,
            "stance": stance,
            "thesis_score_0_100": score,
            "current_book_tickers": ticker_list(grp.get("ticker", pd.Series(dtype=str))),
            "external_watchlist_tickers": "",
            "avg_ret_20d_pct": round(ret20, 2) if np.isfinite(ret20) else np.nan,
            "avg_ret_63d_pct": round(ret63, 2) if np.isfinite(ret63) else np.nan,
            "current_weight_pct": round(current_weight, 2) if np.isfinite(current_weight) else np.nan,
            "supporting_evidence": compact(f"{evidence} Avg 20d={ret20:.2f}% and avg 63d={ret63:.2f}% from current book rows. Top headline: {top_headline}", 650),
            "contradiction_or_risk": compact(f"{why_not_more}. Risk actions: {', '.join(sorted(set(risk_actions))) or 'NO_DATA'}.", 520),
            "route_policy": route_policy,
            "options_policy": compact(opt_summary, 420),
            "first_source_to_open": "subsector_ticker_cycle_map.csv",
            "next_research_question": next_question(group_name, phase),
            "source_files": "subsector_ticker_cycle_map.csv; sector_timeframe_option_route.csv; institutional_optimizer_bridge.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    return out


def build_external_theme_thesis(theme: pd.DataFrame, event_rank: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not theme.empty and "theme" in theme.columns:
        for group, grp in theme.groupby("theme", dropna=False):
            group_name = as_text(group, "Unknown Theme")
            pos = numeric_sum(grp, "positive_catalysts")
            neg = numeric_sum(grp, "negative_catalysts")
            attention = numeric_mean(grp, "attention_score")
            ret20 = numeric_mean(grp, "ret_20d_pct")
            ret63 = numeric_mean(grp, "ret_63d_pct")
            status = mode_text(grp.get("theme_candidate_status", pd.Series(dtype=str)), "Theme watch")
            role = mode_text(grp.get("chain_role", pd.Series(dtype=str)), "mixed")
            score = thesis_score(status, ret20, ret63, pos if np.isfinite(pos) else 0, neg if np.isfinite(neg) else 0, [])
            if "ACTIVE_RESEARCH_READY" in status:
                stance = "External theme research ready"
            elif "WATCH" in status.upper():
                stance = "External context watch"
            else:
                stance = "External theme evidence"
            rows.append({
                "thesis_rank": 0,
                "theme_or_subsector": group_name,
                "source_type": "external_theme_chain",
                "cycle_thesis": status,
                "stance": stance,
                "thesis_score_0_100": round(max(score, min(100.0, attention / 2.0 if np.isfinite(attention) else score)), 1),
                "current_book_tickers": "",
                "external_watchlist_tickers": ticker_list(grp.get("ticker", pd.Series(dtype=str))),
                "avg_ret_20d_pct": round(ret20, 2) if np.isfinite(ret20) else np.nan,
                "avg_ret_63d_pct": round(ret63, 2) if np.isfinite(ret63) else np.nan,
                "current_weight_pct": np.nan,
                "supporting_evidence": compact(f"{group_name} has {pos:.0f} positive and {neg:.0f} negative catalysts; dominant chain role={role}. Top headline: {mode_text(grp.get('top_headline', pd.Series(dtype=str)), 'No headline')}", 650),
                "contradiction_or_risk": "External theme targets are not automatically in the risk book. They need data truth, risk-book entry, liquidity, spread/TCA, event proof, and portfolio room.",
                "route_policy": "External watch only until risk-book entry and proof gates exist.",
                "options_policy": mode_text(grp.get("option_research_side", pd.Series(dtype=str)), "No option route"),
                "first_source_to_open": "theme_candidate_enrichment.csv",
                "next_research_question": "Is this theme a real causal chain, or just headline adjacency?",
                "source_files": "theme_candidate_enrichment.csv; news_supply_chain_readthrough.csv",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    if not event_rank.empty and "target_ticker" in event_rank.columns:
        event_watch = event_rank.head(12).copy()
        rows.append({
            "thesis_rank": 0,
            "theme_or_subsector": "Event read-through candidates",
            "source_type": "event_readthrough",
            "cycle_thesis": "Watch for confirmation",
            "stance": "External event watch",
            "thesis_score_0_100": round(numeric_mean(event_watch, "best_event_score"), 1),
            "current_book_tickers": "",
            "external_watchlist_tickers": ticker_list(event_watch.get("target_ticker", pd.Series(dtype=str))),
            "avg_ret_20d_pct": np.nan,
            "avg_ret_63d_pct": np.nan,
            "current_weight_pct": np.nan,
            "supporting_evidence": compact(f"Top event candidates include {ticker_list(event_watch.get('target_ticker', pd.Series(dtype=str)))}. Top headline: {mode_text(event_watch.get('top_headline', pd.Series(dtype=str)), 'No headline')}", 650),
            "contradiction_or_risk": "Most candidates are NOT_IN_RISK_BOOK_REVIEW, so they are research targets only.",
            "route_policy": "Create risk-book entry before any paper sizing.",
            "options_policy": "No option route before risk-book entry, event proof, liquidity, and spread/TCA checks.",
            "first_source_to_open": "event_readthrough_target_ranking.csv",
            "next_research_question": "Which targets have direct causal evidence instead of broad sympathy movement?",
            "source_files": "event_readthrough_target_ranking.csv; event_readthrough_decision_board.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def next_question(group_name: str, phase: str) -> str:
    g = group_name.lower()
    p = phase.lower()
    if "semi" in g:
        return "Is semiconductor leadership broadening safely, or is it late-cycle concentration that should be de-risked?"
    if "software" in g:
        return "Is software starting a real handoff with breadth and event proof, or only following mega-tech beta?"
    if "power" in g or "infrastructure" in g:
        return "Is AI infrastructure demand translating into durable orders, margins, and confirmed beneficiaries?"
    if "downcycle" in p or "laggard" in p:
        return "Is there a defensive reason to study this group, or should bullish chase stay blocked?"
    return "What evidence would move this group from context to actionable research?"


def build_handoff_matrix(thesis: pd.DataFrame, ticker_map: pd.DataFrame) -> pd.DataFrame:
    def get_row(name: str) -> pd.Series | None:
        if thesis.empty:
            return None
        sub = thesis[thesis["theme_or_subsector"].astype(str).str.lower().eq(name.lower())]
        return sub.iloc[0] if not sub.empty else None

    semi = get_row("Semiconductors")
    software = get_row("Software / Cloud")
    rows = []
    if semi is not None and software is not None:
        rows.append({
            "handoff_rank": 1,
            "source_group": "Semiconductors",
            "target_group": "Software / Cloud",
            "handoff_case": "Plausible handoff watch, not confirmed permission",
            "evidence_for": compact(f"Semis: {semi.get('cycle_thesis')} / {semi.get('supporting_evidence')} Software: {software.get('cycle_thesis')} / {software.get('supporting_evidence')}", 760),
            "evidence_against": compact(f"Semis still lead on price; software still needs risk/source/monitor proof. Semi risk: {semi.get('contradiction_or_risk')} Software risk: {software.get('contradiction_or_risk')}", 760),
            "what_would_confirm": "Software breadth and relative strength improve while semis stop making clean new highs; event proof maps directly to software tickers; risk gates move from reduce/size-down to review or clear.",
            "what_would_disprove": "Semis keep broad leadership with clean risk gates, or software fails price/event confirmation and remains only beta catch-up.",
            "first_source_to_open": "subsector_ticker_cycle_map.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    if not ticker_map.empty:
        ai_rows = ticker_map[ticker_map["theme_or_subsector"].astype(str).str.contains("AI / Data Center|AI Infrastructure|Data Center", case=False, na=False)]
        if not ai_rows.empty:
            rows.append({
                "handoff_rank": 2,
                "source_group": "AI / Data Center",
                "target_group": "Upstream / downstream beneficiaries",
                "handoff_case": "Supply-chain read-through watch",
                "evidence_for": compact(f"Theme rows: {ticker_list(ai_rows.get('ticker', pd.Series(dtype=str)), 14)}", 520),
                "evidence_against": "Theme adjacency is not enough. Each ticker still needs direct event proof, risk-book entry, liquidity, and spread/TCA.",
                "what_would_confirm": "Verified catalyst timestamp, price/volume confirmation, and consistent beneficiary mapping across multiple sources.",
                "what_would_disprove": "Headline link is broad context only, or beneficiary fails price/liquidity/risk gates.",
                "first_source_to_open": "theme_candidate_enrichment.csv",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return pd.DataFrame(rows)


def build_ticker_map(subsector: pd.DataFrame, route: pd.DataFrame, theme: pd.DataFrame, event_rank: pd.DataFrame, optimizer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = set(subsector.get("ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper()) if not subsector.empty else set()
    all_tickers = set(current)
    for df, col in [(theme, "ticker"), (event_rank, "target_ticker")]:
        if not df.empty and col in df.columns:
            all_tickers |= set(df[col].dropna().astype(str).str.upper())
    for ticker in sorted(all_tickers):
        sub = subsector[subsector.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)] if not subsector.empty else pd.DataFrame()
        rt = route[route.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)] if not route.empty else pd.DataFrame()
        th = theme[theme.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)] if not theme.empty else pd.DataFrame()
        ev = event_rank[event_rank.get("target_ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)] if not event_rank.empty else pd.DataFrame()
        opt = optimizer[optimizer.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)] if not optimizer.empty else pd.DataFrame()
        group = mode_text(sub.get("subsector", pd.Series(dtype=str)), mode_text(th.get("theme", pd.Series(dtype=str)), mode_text(ev.get("subsector_cycle_phase", pd.Series(dtype=str)), "Unknown")))
        source_type = "current_risk_book" if ticker in current else "external_watchlist"
        rows.append({
            "ticker": ticker,
            "source_type": source_type,
            "theme_or_subsector": group,
            "sector": mode_text(sub.get("sector", pd.Series(dtype=str)), mode_text(opt.get("sector", pd.Series(dtype=str)), "")),
            "chain_role": mode_text(th.get("chain_role", pd.Series(dtype=str)), mode_text(ev.get("top_target_role", pd.Series(dtype=str)), "")),
            "cycle_phase": mode_text(sub.get("subsector_cycle_phase", pd.Series(dtype=str)), mode_text(ev.get("subsector_cycle_phase", pd.Series(dtype=str)), "")),
            "handoff_signal": mode_text(sub.get("leadership_handoff_signal", pd.Series(dtype=str)), mode_text(opt.get("leadership_handoff_signal", pd.Series(dtype=str)), "")),
            "risk_action": mode_text(sub.get("risk_action", pd.Series(dtype=str)), mode_text(opt.get("final_risk_action", pd.Series(dtype=str)), mode_text(ev.get("final_risk_action", pd.Series(dtype=str)), ""))),
            "event_or_theme_decision": mode_text(ev.get("top_decision", pd.Series(dtype=str)), mode_text(th.get("theme_candidate_status", pd.Series(dtype=str)), "")),
            "option_route": mode_text(rt.get("option_route", pd.Series(dtype=str)), mode_text(th.get("option_research_side", pd.Series(dtype=str)), mode_text(ev.get("directional_route", pd.Series(dtype=str)), ""))),
            "top_headline": mode_text(sub.get("top_headline", pd.Series(dtype=str)), mode_text(th.get("top_headline", pd.Series(dtype=str)), mode_text(ev.get("top_headline", pd.Series(dtype=str)), ""))),
            "proof_required": mode_text(ev.get("proof_required", pd.Series(dtype=str)), "Risk/source/spread/trigger gates still apply."),
            "why_not_more": mode_text(opt.get("why_not_more", pd.Series(dtype=str)), ""),
            "first_source_to_open": "subsector_ticker_cycle_map.csv" if ticker in current else ("theme_candidate_enrichment.csv" if not th.empty else "event_readthrough_target_ranking.csv"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def build_source_guide() -> pd.DataFrame:
    rows = [
        ("Current subsector cycle", "subsector_ticker_cycle_map.csv", "Ticker-level sector/subsector phase, risk action, event gate, and top headline."),
        ("Sector route", "sector_timeframe_option_route.csv", "How sector cycle changes short/medium/long and option route language."),
        ("Optimizer bridge", "institutional_optimizer_bridge.csv", "Why risk/optimizer refuses or caps size even when a group looks interesting."),
        ("External theme chain", "theme_candidate_enrichment.csv", "Supply-chain and peer read-through candidates outside the current risk book."),
        ("Event targets", "event_readthrough_target_ranking.csv", "Headline-to-target mapping and proof required before a target becomes usable."),
    ]
    return pd.DataFrame([{
        "source_area": area,
        "source_file": file,
        "why_it_matters": why,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    } for area, file, why in rows])


def main() -> None:
    subsector = read_csv_safe(ROOT / "subsector_ticker_cycle_map.csv")
    route = read_csv_safe(ROOT / "sector_timeframe_option_route.csv")
    theme = read_csv_safe(ROOT / "theme_candidate_enrichment.csv")
    event_rank = read_csv_safe(ROOT / "event_readthrough_target_ranking.csv")
    optimizer = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")

    current_thesis = build_current_book_thesis(subsector, route, optimizer)
    external_thesis = build_external_theme_thesis(theme, event_rank)
    thesis = pd.concat([current_thesis, external_thesis], ignore_index=True)
    if not thesis.empty:
        thesis = thesis.sort_values(["thesis_score_0_100", "theme_or_subsector"], ascending=[False, True]).reset_index(drop=True)
        thesis["thesis_rank"] = range(1, len(thesis) + 1)

    ticker_map = build_ticker_map(subsector, route, theme, event_rank, optimizer)
    handoff = build_handoff_matrix(thesis, ticker_map)
    source_guide = build_source_guide()

    thesis.to_csv(OUT_THESIS, index=False)
    handoff.to_csv(OUT_HANDOFF, index=False)
    ticker_map.to_csv(OUT_TICKER_MAP, index=False)
    source_guide.to_csv(OUT_SOURCE_GUIDE, index=False)

    def thesis_for(name: str) -> str:
        if thesis.empty:
            return "NO_DATA"
        sub = thesis[thesis["theme_or_subsector"].astype(str).str.lower().eq(name.lower())]
        if sub.empty:
            return "NO_DATA"
        row = sub.iloc[0]
        return f"{row['stance']} / {row['cycle_thesis']}"

    state = {
        "status": "SECTOR_THEME_DEPTH_ACTIVE",
        "date": today_str(),
        "thesis_rows": int(len(thesis)),
        "handoff_rows": int(len(handoff)),
        "ticker_map_rows": int(len(ticker_map)),
        "semiconductor_thesis": thesis_for("Semiconductors"),
        "software_thesis": thesis_for("Software / Cloud"),
        "top_theme_or_subsector": as_text(thesis.iloc[0]["theme_or_subsector"]) if not thesis.empty else "NO_DATA",
        "top_stance": as_text(thesis.iloc[0]["stance"]) if not thesis.empty else "NO_DATA",
        "product_note": "Sector depth is a thesis map, not a permission layer. Risk/source/spread gates still control action.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Product note",
        state["product_note"],
        "",
        "## Key theses",
        f"- Semiconductors: {state['semiconductor_thesis']}",
        f"- Software / Cloud: {state['software_thesis']}",
        "",
        "## Thesis board",
        df_to_markdown(thesis, max_rows=30),
        "",
        "## Handoff matrix",
        df_to_markdown(handoff, max_rows=20),
        "",
        "## Ticker map",
        df_to_markdown(ticker_map.head(60), max_rows=60),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Sector And Theme Depth Desk", sections)
    print(f"Sector/theme depth complete: {len(thesis)} theses, {len(handoff)} handoff rows, {len(ticker_map)} tickers")


if __name__ == "__main__":
    main()
