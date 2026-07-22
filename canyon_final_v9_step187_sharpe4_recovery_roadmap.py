#!/usr/bin/env python3
"""
Canyon v9 Step 187 - Sharpe 4 Recovery Roadmap.

Research-only. No broker connection. No live orders.

Step185 measured the Sharpe 4 gap. Step186 made the P0 repair pack. Step187
answers the natural question: "If the target is so far away, what do we do
next?"

It builds a recovery roadmap and a clean-candidate research pool from existing
local outputs. The important distinction:

  - current broken book = repair only
  - outside candidates = research only until risk book, signal proof, event
    proof, execution/TCA, and price trigger are clean

Outputs:
  sharpe4_recovery_state.json
  sharpe4_recovery_stage_plan.csv
  sharpe4_recovery_candidate_pool.csv
  sharpe4_recovery_top_actions.csv
  sharpe4_recovery_report.md
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "sharpe4_recovery_state.json"
OUT_STAGES = ROOT / "sharpe4_recovery_stage_plan.csv"
OUT_POOL = ROOT / "sharpe4_recovery_candidate_pool.csv"
OUT_ACTIONS = ROOT / "sharpe4_recovery_top_actions.csv"
OUT_REPORT = ROOT / "sharpe4_recovery_report.md"


TOP_SIGNAL_TO_VALIDATION_SIGNAL = {
    "momentum": "mom_12m_skip1m",
    "quality": "quality_hist",
    "revision": "rev_growth_yoy",
    "surprise": "eps_growth_yoy",
    "regime_ml": "",
    "ml_ensemble": "",
    "sentiment": "",
    "squeeze": "",
    "insider": "",
    "options": "",
}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if not np.isfinite(value):
        return low
    return float(min(max(value, low), high))


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def one_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker", sort_col: str | None = None) -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""]
    if sort_col and sort_col in work.columns:
        work["_sort"] = pd.to_numeric(work[sort_col], errors="coerce").fillna(-1)
        work = work.sort_values([ticker_col, "_sort"], ascending=[True, False])
    out: dict[str, pd.Series] = {}
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get(ticker_col))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def best_rows(df: pd.DataFrame, ticker_col: str, score_col: str) -> dict[str, pd.Series]:
    return one_by_ticker(df, ticker_col=ticker_col, sort_col=score_col)


def add_source(sources: dict[str, set[str]], ticker: str, source: str) -> None:
    ticker = clean_ticker(ticker)
    if ticker:
        sources[ticker].add(source)


def signal_permission(top_signal: str, signal_policy: dict[str, pd.Series]) -> tuple[str, str, float, str]:
    top = as_text(top_signal).lower()
    validation = TOP_SIGNAL_TO_VALIDATION_SIGNAL.get(top, "")
    if not top:
        return "", "NO_SIGNAL", 0.0, "No mapped top signal."
    if not validation:
        return "", "UNVALIDATED_SIGNAL_FAMILY", 0.0, "Top signal family is not yet validated by Step156 live/IC policy."
    row = signal_policy.get(clean_ticker(validation))
    if row is None:
        return validation, "NO_POLICY_ROW", 0.0, "Mapped validation signal has no policy row."
    active_mult = safe_float(row.get("sharpe4_active_multiplier"), 0.0)
    action = as_text(row.get("original_signal_action"), "NO_DATA")
    active_use = as_text(row.get("active_use"), "")
    if active_mult <= 0:
        return validation, action, 0.0, active_use or "Not allowed in active Sharpe 4 model."
    return validation, action, active_mult, active_use or "Allowed with monitor."


def liquidity_score(label: str, adv: float) -> tuple[float, str, str]:
    raw = as_text(label).upper()
    if raw in {"HIGH", "LIQUID"}:
        return 88.0, "LIQUID", "Large enough for research sizing, still needs live bid/ask check."
    if raw == "GOOD":
        return 76.0, "GOOD", "Usable for research, but spread/TCA must be checked."
    if raw == "FAIR":
        return 58.0, "FAIR_REVIEW", "Manual spread and fill proof required before paper route."
    if raw in {"THIN", "LOW"}:
        return 25.0, "THIN_BLOCK", "Too fragile for Sharpe 4 active model without much better execution proof."
    if np.isfinite(adv):
        if adv >= 1_000_000_000:
            return 78.0, "ADV_GOOD_PROXY", "ADV is strong, but live spread is missing."
        if adv >= 300_000_000:
            return 58.0, "ADV_FAIR_PROXY", "ADV is fair; manual spread proof required."
    return 35.0, "LIQUIDITY_DATA_GAP", "Missing liquidity proof."


def sector_score(text: str) -> tuple[float, str]:
    raw = as_text(text).lower()
    if "early improvement" in raw:
        return 78.0, "Early improvement: interesting, but still needs trigger proof."
    if "software" in raw:
        return 74.0, "Software-style improvement gets a constructive research bias."
    if "leadership expansion" in raw:
        return 66.0, "Leadership is strong but must be checked for crowding."
    if "late-cycle" in raw or "chase risk" in raw:
        return 38.0, "Late-cycle leader: do not chase size; require stricter proof."
    if "downcycle" in raw or "laggard" in raw:
        return 25.0, "Downcycle or laggard: not a Sharpe 4 long candidate without reversal proof."
    if "neutral" in raw or "base" in raw:
        return 50.0, "Neutral cycle: research only."
    return 45.0, "No cycle evidence."


def event_score(row: pd.Series | None) -> tuple[float, str, str]:
    if row is None:
        return 0.0, "NO_EVENT_EDGE", "No mapped event read-through."
    best = safe_float(row.get("best_event_score"), 0.0)
    pos = safe_float(row.get("positive_event_count"), 0.0)
    neg = safe_float(row.get("negative_event_count"), 0.0)
    route = as_text(row.get("directional_route"), "")
    headline = as_text(row.get("top_headline"), "")
    score = clamp(best + max(0.0, pos - neg) * 1.5, 0.0, 100.0)
    if neg > pos:
        label = "NEGATIVE_EVENT_RISK"
    elif score >= 65:
        label = "EVENT_RESEARCH_CANDIDATE"
    elif score > 0:
        label = "EVENT_CONTEXT_ONLY"
    else:
        label = "NO_EVENT_EDGE"
    note = f"{route}; {headline}" if headline else route
    return score, label, note


def collect_candidate_tickers() -> tuple[set[str], dict[str, set[str]]]:
    sources: dict[str, set[str]] = defaultdict(set)
    tickers: set[str] = set()
    file_specs = [
        ("daily_picks.csv", "ticker", "daily alpha picks"),
        ("master_10_layer_decision_matrix_v2.csv", "ticker", "10-layer decision"),
        ("theme_candidate_enrichment.csv", "ticker", "theme/event supply chain"),
        ("event_readthrough_target_ranking.csv", "target_ticker", "news read-through"),
        ("long_term_hold_candidates.csv", "ticker", "long-term quality candidates"),
        ("short_candidates.csv", "ticker", "downside candidates"),
        ("intraday_liquidity_proxy.csv", "ticker", "liquidity coverage"),
    ]
    for fname, col, label in file_specs:
        df = read_csv_safe(ROOT / fname)
        if df.empty or col not in df.columns:
            continue
        for value in df[col].dropna().astype(str):
            ticker = clean_ticker(value)
            if ticker:
                tickers.add(ticker)
                add_source(sources, ticker, label)
    return tickers, sources


def build_candidate_pool() -> pd.DataFrame:
    tickers, sources = collect_candidate_tickers()

    daily = one_by_ticker(read_csv_safe(ROOT / "daily_picks.csv"), "ticker", "alpha_score")
    master = one_by_ticker(read_csv_safe(ROOT / "master_10_layer_decision_matrix_v2.csv"), "ticker", "stack_score_avg")
    p0 = one_by_ticker(read_csv_safe(ROOT / "sharpe4_p0_ticker_repair_plan.csv"), "ticker", "current_weight_pct")
    theme = one_by_ticker(read_csv_safe(ROOT / "theme_candidate_enrichment.csv"), "ticker", "attention_score")
    event = best_rows(read_csv_safe(ROOT / "event_readthrough_target_ranking.csv"), "target_ticker", "best_event_score")
    longterm = one_by_ticker(read_csv_safe(ROOT / "long_term_hold_candidates.csv"), "ticker", "quality_score")
    shorts = one_by_ticker(read_csv_safe(ROOT / "short_candidates.csv"), "ticker", "short_score")
    liquidity = one_by_ticker(read_csv_safe(ROOT / "intraday_liquidity_proxy.csv"), "ticker", "avg_20d_dollar_volume")
    signal_policy = one_by_ticker(read_csv_safe(ROOT / "sharpe4_p0_signal_policy_enforced.csv"), "signal")

    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        d = daily.get(ticker)
        m = master.get(ticker)
        p = p0.get(ticker)
        th = theme.get(ticker)
        ev = event.get(ticker)
        lt = longterm.get(ticker)
        sh = shorts.get(ticker)
        liq = liquidity.get(ticker)

        alpha = safe_float(d.get("alpha_score"), np.nan) if d is not None else np.nan
        alpha_rank = safe_float(d.get("alpha_rank"), np.nan) if d is not None else np.nan
        top_signal = as_text(d.get("top_signal"), "") if d is not None else ""
        validation_signal, signal_action, signal_mult, signal_note = signal_permission(top_signal, signal_policy)

        master_score = safe_float(m.get("stack_score_avg"), np.nan) if m is not None else np.nan
        master_action = as_text(m.get("master_action"), "") if m is not None else ""
        l8_state = as_text(m.get("L8_state"), "") if m is not None else ""
        sector = as_text(d.get("sector"), "") if d is not None else ""
        if not sector and m is not None:
            sector = as_text(m.get("L3_note"), "")
        if not sector and lt is not None:
            sector = as_text(lt.get("sector"), "")
        if not sector and sh is not None:
            sector = as_text(sh.get("sector"), "")

        cycle_text = ""
        if th is not None:
            cycle_text = as_text(th.get("theme"), "")
        if ev is not None and as_text(ev.get("subsector_cycle_phase"), ""):
            cycle_text = as_text(ev.get("subsector_cycle_phase"), "")
        if p is not None and as_text(p.get("cycle_phase"), ""):
            cycle_text = as_text(p.get("cycle_phase"), "")
        cycle_points, cycle_note = sector_score(cycle_text)

        event_points, event_label, event_note = event_score(ev)
        theme_attention = safe_float(th.get("attention_score"), 0.0) if th is not None else 0.0
        theme_points = clamp(theme_attention / 3.0, 0.0, 100.0)
        quality_points = safe_float(lt.get("quality_score"), np.nan) if lt is not None else np.nan
        short_points = safe_float(sh.get("short_score"), np.nan) if sh is not None else np.nan

        adv = safe_float(liq.get("avg_20d_dollar_volume"), np.nan) if liq is not None else safe_float(th.get("avg_dollar_volume_20d"), np.nan) if th is not None else np.nan
        liq_label = as_text(liq.get("liquidity_label"), "") if liq is not None else as_text(th.get("liquidity_status"), "") if th is not None else ""
        liq_points, liq_status, liq_note = liquidity_score(liq_label, adv)

        in_current_bad_book = p is not None
        if in_current_bad_book:
            risk_status = "CURRENT_BOOK_REPAIR_ONLY"
            risk_note = as_text(p.get("plain_next_step"), "Repair current risk first.")
            risk_points = 0.0
        elif ev is not None and as_text(ev.get("final_risk_action"), "").upper() in {"NOT_IN_RISK_BOOK_REVIEW", "UNKNOWN_NEEDS_DATA"}:
            risk_status = "NEEDS_RISK_BOOK_ENTRY"
            risk_note = "Event/theme candidate is not yet in the risk book; create risk entry before sizing."
            risk_points = 35.0
        elif m is not None and "RED" in l8_state.upper():
            risk_status = "PORTFOLIO_RISK_RED"
            risk_note = "10-layer view says portfolio risk is red; no new exposure."
            risk_points = 20.0
        else:
            risk_status = "RISK_COVERAGE_NEEDED"
            risk_note = "Needs explicit single-name risk, earnings, liquidity, correlation, and TCA checks."
            risk_points = 45.0

        if signal_action == "BLOCK_SIGNAL":
            signal_points = 0.0
        elif signal_action in {"DOWNWEIGHT", "RESEARCH_ONLY_DOWNWEIGHTED"}:
            signal_points = 35.0
        elif signal_action == "UNVALIDATED_SIGNAL_FAMILY":
            signal_points = 25.0
        elif signal_action in {"NO_SIGNAL", "NO_POLICY_ROW"}:
            signal_points = 30.0
        else:
            signal_points = clamp(signal_mult * 100.0, 0.0, 100.0)

        alpha_points = alpha if np.isfinite(alpha) else np.nanmean([event_points, theme_points, quality_points if np.isfinite(quality_points) else np.nan])
        if not np.isfinite(alpha_points):
            alpha_points = 35.0

        base = (
            0.20 * alpha_points
            + 0.16 * event_points
            + 0.12 * theme_points
            + 0.12 * cycle_points
            + 0.14 * liq_points
            + 0.13 * signal_points
            + 0.13 * risk_points
        )
        if np.isfinite(master_score):
            base = 0.80 * base + 0.20 * master_score
        if np.isfinite(quality_points):
            base += min(8.0, quality_points / 15.0)
        if np.isfinite(short_points):
            base += min(10.0, short_points * 1.5)
        if in_current_bad_book:
            base = min(base, 20.0)
        if risk_status == "NEEDS_RISK_BOOK_ENTRY":
            base = min(base, 68.0)
        if signal_points == 0.0 and not event_label.startswith("EVENT"):
            base = min(base, 45.0)

        if in_current_bad_book:
            lane = "Repair current book first"
            next_action = "Cut or repair current risk weight; do not count this toward Sharpe 4 alpha."
            permission = "NO_NEW_EXPOSURE"
        elif risk_status == "NEEDS_RISK_BOOK_ENTRY":
            lane = "Research candidate - risk entry first"
            next_action = "Create risk-book entry, then run single-name VaR, earnings gap, liquidity, correlation, and TCA checks."
            permission = "RESEARCH_ONLY"
        elif event_label == "NEGATIVE_EVENT_RISK" or (np.isfinite(short_points) and short_points >= 4.5):
            lane = "Downside research"
            next_action = "Only review defined-risk downside or hedge thesis after event proof and spread/TCA checks."
            permission = "DOWNSIDE_RESEARCH_ONLY"
        elif base >= 65 and liq_points >= 70:
            lane = "Recovery watchlist"
            next_action = "Watch source proof, price confirmation, and execution cost before tiny paper review."
            permission = "WATCH_ONLY"
        else:
            lane = "Research backlog"
            next_action = "Keep as context; not urgent for Sharpe 4 recovery."
            permission = "CONTEXT_ONLY"

        rows.append({
            "ticker": ticker,
            "recovery_rank_score": round(clamp(base), 2),
            "recovery_lane": lane,
            "current_permission": permission,
            "next_action": next_action,
            "sector": sector,
            "cycle_evidence": cycle_text,
            "cycle_read": cycle_note,
            "alpha_score": round(alpha, 2) if np.isfinite(alpha) else np.nan,
            "alpha_rank": int(alpha_rank) if np.isfinite(alpha_rank) else np.nan,
            "top_signal": top_signal,
            "validation_signal": validation_signal,
            "signal_status": signal_action,
            "signal_note": signal_note,
            "event_status": event_label,
            "event_score": round(event_points, 2),
            "event_note": event_note,
            "theme_attention_score": round(theme_attention, 2),
            "theme_status": as_text(th.get("theme_candidate_status"), "") if th is not None else "",
            "quality_score": round(quality_points, 2) if np.isfinite(quality_points) else np.nan,
            "short_score": round(short_points, 2) if np.isfinite(short_points) else np.nan,
            "liquidity_status": liq_status,
            "liquidity_note": liq_note,
            "avg_dollar_volume_20d": round(adv, 2) if np.isfinite(adv) else np.nan,
            "risk_status": risk_status,
            "risk_note": risk_note,
            "master_action": master_action,
            "master_score": round(master_score, 2) if np.isfinite(master_score) else np.nan,
            "source_count": len(sources.get(ticker, set())),
            "source_files": "; ".join(sorted(sources.get(ticker, set()))),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        lane_order = {
            "Recovery watchlist": 0,
            "Research candidate - risk entry first": 1,
            "Downside research": 2,
            "Repair current book first": 3,
            "Research backlog": 4,
        }
        out["_lane_order"] = out["recovery_lane"].map(lane_order).fillna(9)
        out = out.sort_values(["_lane_order", "recovery_rank_score"], ascending=[True, False]).drop(columns=["_lane_order"])
        out.insert(0, "recovery_rank", range(1, len(out) + 1))
    return out


def build_stage_plan(pool: pd.DataFrame) -> pd.DataFrame:
    p0 = read_json_safe(ROOT / "sharpe4_p0_repair_state.json", {})
    s185 = read_json_safe(ROOT / "sharpe_target4_state.json", {})

    current_sharpe = safe_float(s185.get("current_headline_sharpe"), np.nan)
    planning_sharpe = safe_float(s185.get("credibility_adjusted_planning_sharpe"), np.nan)
    p0_clean = safe_float(p0.get("p0_clean_gross_pct"), np.nan)
    alpha_gross = safe_float(p0.get("sharpe4_alpha_gross_allowed_pct"), np.nan)
    risk_entry_candidates = int(pool["recovery_lane"].eq("Research candidate - risk entry first").sum()) if not pool.empty else 0
    watchlist_candidates = int(pool["recovery_lane"].eq("Recovery watchlist").sum()) if not pool.empty else 0

    rows = [
        {
            "stage": "Stage 0 - Stop the leak",
            "status": "ACTIVE_NOW",
            "goal": "Do not let the current blocked book pretend to be alpha.",
            "what_to_do": f"Use P0 repair weights first. Current clean gross is {p0_clean:.2f}% and alpha gross allowed now is {alpha_gross:.2f}%.",
            "done_when": "Hard-risk names are no longer driving new ideas; alpha gross allowed is above zero only for clean names.",
            "source_files": "sharpe4_p0_repair_state.json / sharpe4_p0_ticker_repair_plan.csv",
        },
        {
            "stage": "Stage 1 - Rebuild the candidate pool",
            "status": "ACTIVE_NOW",
            "goal": "Find candidates outside the broken 8-name book.",
            "what_to_do": f"Review {risk_entry_candidates} risk-entry-first candidates and {watchlist_candidates} recovery-watchlist candidates from the broader local universe.",
            "done_when": "Top candidates have risk-book entries, source proof, liquidity/TCA proof, and a clear horizon route.",
            "source_files": "daily_picks.csv / theme_candidate_enrichment.csv / event_readthrough_target_ranking.csv",
        },
        {
            "stage": "Stage 2 - Repair signal proof",
            "status": "BLOCKING_TARGET_4",
            "goal": "Make sure active alpha is from proven signals, not noisy labels.",
            "what_to_do": "Keep 7 blocked signals at zero; keep downweighted signals research-only until live IC observations exist.",
            "done_when": "Active signal count is positive and no blocked signal contributes to the Sharpe 4 model.",
            "source_files": "sharpe4_p0_signal_policy_enforced.csv / signal_validation_state.json",
        },
        {
            "stage": "Stage 3 - Make costs real",
            "status": "BLOCKING_TARGET_4",
            "goal": "Stop paper Sharpe from disappearing after turnover, spread, and failed-fill assumptions.",
            "what_to_do": "Lower monthly turnover toward 45%, use current/stress TCA in backtest, and require spread proof before paper sizing.",
            "done_when": "Median turnover <=45%, average current TCA <=10 bps, and failed-fill assumptions are modeled.",
            "source_files": "sharpe4_p0_execution_budget.csv / backtest_execution_reality_check.csv",
        },
        {
            "stage": "Stage 4 - Retest honestly",
            "status": "NEXT_AFTER_P0",
            "goal": "Improve the headline Sharpe without overfitting.",
            "what_to_do": f"Current headline Sharpe is {current_sharpe:.2f}; proof-adjusted planning Sharpe is {planning_sharpe:.2f}. Re-run cost-aware OOS tests after P0 repairs.",
            "done_when": "Frozen-signal OOS Sharpe improves, no look-ahead/PIT gate is open, and drawdown attribution is clean.",
            "source_files": "backtest_walk_forward_proxy.csv / pit_truth_state.json / backtest_credibility_state.json",
        },
        {
            "stage": "Stage 5 - Only then add alpha sleeves",
            "status": "WAIT",
            "goal": "Add return after the system is cleaner.",
            "what_to_do": "Test event-confirmed momentum, software catch-up vs late semis, defensive hedge overlays, and quality long-term sleeve separately.",
            "done_when": "Each sleeve has its own IC, decay, risk, TCA, and drawdown attribution.",
            "source_files": "event_readthrough_target_ranking.csv / sector_theme_depth_ticker_map.csv / options_execution_route_matrix.csv",
        },
    ]
    return pd.DataFrame(rows)


def build_top_actions(pool: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p0 = read_json_safe(ROOT / "sharpe4_p0_repair_state.json", {})
    rows.append({
        "priority": "P0",
        "action": "Accept repair mode first",
        "plain_english": f"The book cannot chase Sharpe 4 while alpha gross allowed is {safe_float(p0.get('sharpe4_alpha_gross_allowed_pct'), 0):.2f}%.",
        "where_to_click": "Performance > Sharpe 4 Target > P0 Repair Pack",
        "source_files": "sharpe4_p0_repair_state.json",
    })
    if not pool.empty:
        risk_first = pool[pool["recovery_lane"].eq("Research candidate - risk entry first")].head(5)
        if not risk_first.empty:
            rows.append({
                "priority": "P1",
                "action": "Create risk-book entries for the best outside candidates",
                "plain_english": "Best research names are outside the current risk book. They are not tradeable until single-name risk, earnings gap, liquidity, correlation, and TCA are created.",
                "where_to_click": "Performance > Recovery Map, then News / Ideas for evidence",
                "source_files": "sharpe4_recovery_candidate_pool.csv",
            })
        downside = pool[pool["recovery_lane"].eq("Downside research")].head(3)
        if not downside.empty:
            rows.append({
                "priority": "P1",
                "action": "Separate downside research from long alpha",
                "plain_english": "Negative event or weak-quality names can be hedge/put-spread research only. Do not mix them with long Sharpe 4 alpha.",
                "where_to_click": "News and Time Horizon",
                "source_files": "short_candidates.csv / event_readthrough_target_ranking.csv",
            })
    rows.append({
        "priority": "P1",
        "action": "Retest after P0 changes",
        "plain_english": "A better Sharpe must come from cleaner evidence and lower drag, not from relabeling the same backtest.",
        "where_to_click": "Performance > Sharpe 4 Target",
        "source_files": "backtest_walk_forward_proxy.csv / sharpe4_p0_execution_budget.csv",
    })
    return pd.DataFrame(rows)


def write_report(state: dict[str, Any], stages: pd.DataFrame, pool: pd.DataFrame, actions: pd.DataFrame) -> None:
    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Current Answer",
            "",
            f"- Recovery status: **{state['recovery_status']}**",
            f"- Candidate pool rows: **{state['candidate_pool_rows']}**",
            f"- Current-book repair-only rows: **{state['current_book_repair_only_count']}**",
            f"- Risk-entry-first candidates: **{state['risk_entry_first_count']}**",
            f"- Recovery watchlist candidates: **{state['recovery_watchlist_count']}**",
            "",
            "Plain English: the system is far from a credible Sharpe 4 claim, but the next move is clear: repair the current book, rebuild candidates outside the broken book, then retest with real signal and execution proof.",
        ]),
        "## Stage Plan\n\n" + df_to_markdown(stages),
        "## Top Actions\n\n" + df_to_markdown(actions),
        "## Candidate Pool\n\n" + df_to_markdown(pool.head(30)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 187 - Sharpe 4 Recovery Roadmap", sections)


def main() -> None:
    pool = build_candidate_pool()
    stages = build_stage_plan(pool)
    actions = build_top_actions(pool, stages)

    pool.to_csv(OUT_POOL, index=False)
    stages.to_csv(OUT_STAGES, index=False)
    actions.to_csv(OUT_ACTIONS, index=False)

    state = {
        "date": today_str(),
        "recovery_status": "REPAIR_AND_REBUILD_MODE",
        "candidate_pool_rows": int(len(pool)),
        "current_book_repair_only_count": int(pool["recovery_lane"].eq("Repair current book first").sum()) if not pool.empty else 0,
        "risk_entry_first_count": int(pool["recovery_lane"].eq("Research candidate - risk entry first").sum()) if not pool.empty else 0,
        "recovery_watchlist_count": int(pool["recovery_lane"].eq("Recovery watchlist").sum()) if not pool.empty else 0,
        "downside_research_count": int(pool["recovery_lane"].eq("Downside research").sum()) if not pool.empty else 0,
        "top_candidate_tickers": pool.head(8)["ticker"].tolist() if not pool.empty else [],
        "claim_allowed": False,
        "next_required_work": "Create risk-book entries for outside candidates, repair signal proof, lower turnover/TCA, then rerun cost-aware OOS tests.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)
    write_report(state, stages, pool, actions)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Candidate pool rows: {state['candidate_pool_rows']}")
    print(f"[OK] Risk-entry-first candidates: {state['risk_entry_first_count']}")
    print(f"[OK] Recovery watchlist candidates: {state['recovery_watchlist_count']}")
    print(f"[OK] Current-book repair-only rows: {state['current_book_repair_only_count']}")


if __name__ == "__main__":
    main()
