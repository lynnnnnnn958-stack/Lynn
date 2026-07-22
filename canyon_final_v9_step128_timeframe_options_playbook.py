#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 128: Timeframe and Options Playbook
====================================================

Splits the decision system into three practical horizons:

  Short-term: 1-5 trading days
  Medium-term: 2-8 weeks
  Long-term: 3-12 months

It also builds a separate options playbook so the dashboard can clearly show
which tickers are stock-only, wait-only, hedge-only, call-spread candidates, or
put-spread candidates.

Research only. No broker connection. No live orders.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

OUT_MATRIX = ROOT / "timeframe_decision_matrix.csv"
OUT_SUMMARY = ROOT / "ticker_timeframe_summary.csv"
OUT_OPTIONS = ROOT / "options_playbook.csv"
OUT_STRATEGY = ROOT / "strategy_route_playbook.csv"
OUT_STATE = ROOT / "timeframe_options_playbook_state.json"
OUT_REPORT = ROOT / "timeframe_options_playbook_report.md"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 10:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json_safe(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists() or path.stat().st_size <= 2:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clean_ticker(value) -> str:
    return str(value).strip().upper()


def as_float(value, default=np.nan) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if not np.isfinite(value):
        return low
    return max(low, min(high, float(value)))


def merge_on_ticker(base: pd.DataFrame, other: pd.DataFrame, cols: list[str], suffix: str = "") -> pd.DataFrame:
    if base.empty or other.empty or "ticker" not in other.columns:
        return base
    tmp = other.copy()
    tmp["ticker"] = tmp["ticker"].apply(clean_ticker)
    keep = [c for c in cols if c in tmp.columns]
    if "ticker" not in keep:
        keep = ["ticker"] + keep
    return base.merge(tmp[keep].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", suffix))


def build_base() -> pd.DataFrame:
    picks = read_csv_safe(ROOT / "daily_picks_filtered.csv")
    if picks.empty:
        picks = read_csv_safe(ROOT / "daily_picks.csv")
    if picks.empty or "ticker" not in picks.columns:
        return pd.DataFrame()
    base = picks.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)

    target = read_csv_safe(ROOT / "institutional_target_weights.csv")
    risk = read_csv_safe(ROOT / "final_risk_gate.csv")
    event = read_csv_safe(ROOT / "event_research_dossier.csv")
    options = read_csv_safe(ROOT / "options_signals.csv")
    monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    quality = read_csv_safe(ROOT / "fundamental_quality_rank.csv")
    execution = read_csv_safe(ROOT / "execution_trade_plan.csv")
    overlay = read_csv_safe(ROOT / "options_greeks_book_risk.csv")

    base = merge_on_ticker(base, target, [
        "ticker", "target_weight_pct", "target_status", "master_action", "event_gate", "execution_status", "sleeve",
    ])
    base = merge_on_ticker(base, risk, [
        "ticker", "master_risk_action", "final_risk_action", "recommended_risk_weight_pct", "reason_stack",
    ])
    base = merge_on_ticker(base, event, [
        "ticker", "event_research_score", "event_source_coverage_pct", "event_risk_score", "event_gate",
        "earnings_date", "days_until_earnings", "earnings_risk_flag", "surprise_signal",
        "revision_signal", "catalysts", "risks", "missing_research_sources",
    ], "_event")
    base = merge_on_ticker(base, options, [
        "ticker", "rank_options", "options_strategy", "conviction", "iv_rank", "atm_iv", "pcr_vol",
        "net_call_premium", "uoa_flag", "uoa_bear_flag", "gex_sign", "gex_net", "squeeze_risk",
        "flow_score", "iv_score", "skew_score", "gamma_score", "alpha_options", "expiry",
    ])
    base = merge_on_ticker(base, monitor, [
        "ticker", "latest_close", "daily_return", "prior_20d_high", "prior_20d_low",
        "price_break_state", "volume_ratio", "volume_spike_state", "realized_vol_20d",
        "volatility_regime_state", "max_monitor_severity", "spread_status",
    ])
    base = merge_on_ticker(base, quality, [
        "ticker", "quality_score", "quality_label", "fcf_yield", "gross_margin", "debt_ebitda",
        "revenue_growth", "roe",
    ])
    base = merge_on_ticker(base, execution, [
        "ticker", "execution_playbook_status", "total_tca_cost_bps", "expected_fill_rate_pct",
        "auction_policy", "estimated_days_to_complete",
    ])
    base = merge_on_ticker(base, overlay, [
        "ticker", "delta_proxy", "gamma_proxy", "vega_proxy", "options_heat_score", "greeks_status",
    ])
    return base


def risk_penalty(row: pd.Series, horizon: str) -> tuple[float, list[str]]:
    final_risk = str(row.get("final_risk_action", row.get("target_status", "REVIEW"))).upper()
    master = str(row.get("master_risk_action", row.get("master_action", "REVIEW"))).upper()
    event_gate = str(row.get("event_gate", row.get("event_gate_event", "REVIEW"))).upper()
    exec_status = str(row.get("execution_playbook_status", row.get("execution_status", "REVIEW"))).upper()
    reasons: list[str] = []
    penalty = 0.0

    if final_risk in {"BLOCKED", "BLOCK_NEW"}:
        penalty -= 50
        reasons.append("risk gate blocks new exposure")
    elif final_risk == "REDUCE_ONLY":
        penalty -= 42
        reasons.append("risk gate says reduce only")
    elif final_risk == "SIZE_DOWN":
        penalty -= 18 if horizon == "Short-term" else 14
        reasons.append("risk gate says size down")

    if master in {"BLOCKED", "BLOCK_NEW", "REDUCE_ONLY"}:
        penalty -= 18
        reasons.append("portfolio risk state is restrictive")
    elif master == "SIZE_DOWN":
        penalty -= 10
        reasons.append("portfolio risk state requires smaller size")

    if event_gate in {"BLOCKED", "BLOCK_NEW"}:
        penalty -= 30
        reasons.append("event gate blocks new exposure")
    elif event_gate == "REVIEW":
        penalty -= 12 if horizon == "Short-term" else 8
        reasons.append("event source needs review")

    if exec_status in {"BLOCK_NEW", "DATA_GAP"}:
        penalty -= 18
        reasons.append("execution data gap or block")
    elif exec_status in {"SIZE_DOWN", "REVIEW", "REDUCE_ONLY"}:
        penalty -= 8
        reasons.append("execution playbook needs care")

    return penalty, reasons


def monitor_adjustment(row: pd.Series, horizon: str) -> tuple[float, list[str]]:
    if horizon != "Short-term":
        return 0.0, []
    reasons: list[str] = []
    adj = 0.0
    daily_ret = as_float(row.get("daily_return"), 0.0)
    volume_state = str(row.get("volume_spike_state", "OK")).upper()
    price_state = str(row.get("price_break_state", "OK")).upper()
    severity = str(row.get("max_monitor_severity", "OK")).upper()
    vol_state = str(row.get("volatility_regime_state", "OK")).upper()

    if price_state in {"WARNING", "CRITICAL"}:
        adj += 7 if daily_ret >= 0 else -7
        reasons.append("price break monitor is active")
    if volume_state in {"WARNING", "CRITICAL"}:
        adj += 8 if daily_ret >= 0 else -8
        reasons.append("volume spike monitor is active")
    if severity == "CRITICAL":
        adj -= 4
        reasons.append("desk monitor has critical flag")
    elif severity == "WARNING":
        adj -= 2
        reasons.append("desk monitor has warning flag")
    if vol_state in {"WARNING", "CRITICAL"}:
        adj -= 5
        reasons.append("volatility regime is unstable")
    return adj, reasons


def horizon_scores(row: pd.Series, macro: dict, regime: dict) -> dict[str, float]:
    alpha = as_float(row.get("alpha_score"), 50.0)
    sig_regime = as_float(row.get("sig_regime_ml"), 50.0)
    sig_quality = as_float(row.get("sig_quality"), as_float(row.get("quality_score"), 50.0))
    sig_revision = as_float(row.get("sig_revision"), 50.0)
    sig_surprise = as_float(row.get("sig_surprise"), 50.0)
    sig_sentiment = as_float(row.get("sig_sentiment"), 50.0)
    sig_squeeze = as_float(row.get("sig_squeeze"), 50.0)
    sig_options = as_float(row.get("sig_options"), as_float(row.get("rank_options"), 50.0))
    quality = as_float(row.get("quality_score"), sig_quality)
    macro_score = as_float(macro.get("macro_score", 50.0), 50.0)
    regime_name = str(regime.get("regime", row.get("regime", "UNKNOWN"))).upper()
    regime_adj = 4.0 if regime_name in {"BULL", "LATE_BULL"} else (-8.0 if regime_name == "BEAR" else 0.0)

    scores: dict[str, float] = {}
    for horizon in ["Short-term", "Medium-term", "Long-term"]:
        penalty, _ = risk_penalty(row, horizon)
        monitor_adj, _ = monitor_adjustment(row, horizon)
        if horizon == "Short-term":
            raw = (
                alpha * 0.26
                + sig_options * 0.19
                + sig_squeeze * 0.16
                + sig_sentiment * 0.12
                + sig_surprise * 0.12
                + sig_regime * 0.08
                + macro_score * 0.07
                + monitor_adj
                + penalty
            )
        elif horizon == "Medium-term":
            raw = (
                alpha * 0.28
                + sig_revision * 0.20
                + sig_surprise * 0.14
                + sig_regime * 0.14
                + quality * 0.12
                + sig_sentiment * 0.06
                + macro_score * 0.06
                + regime_adj
                + penalty
            )
        else:
            event_penalty = -6 if str(row.get("event_gate", "")).upper() == "REVIEW" else 0
            raw = (
                quality * 0.34
                + alpha * 0.20
                + sig_quality * 0.14
                + sig_regime * 0.12
                + sig_revision * 0.10
                + macro_score * 0.06
                + sig_sentiment * 0.04
                + regime_adj
                + penalty * 0.80
                + event_penalty
            )
        scores[horizon] = round(clamp(raw), 1)
    return scores


def trigger_text(row: pd.Series, side: str) -> str:
    close = as_float(row.get("latest_close"), np.nan)
    high = as_float(row.get("prior_20d_high"), np.nan)
    low = as_float(row.get("prior_20d_low"), np.nan)
    if side == "CALL" and np.isfinite(high):
        return f"Watch close above prior 20d high near {high:.2f}"
    if side == "PUT" and np.isfinite(low):
        return f"Watch close below prior 20d low near {low:.2f}"
    if np.isfinite(close):
        return f"Current reference price {close:.2f}; wait for confirmation"
    return "No price trigger available"


def invalidation_text(row: pd.Series, side: str) -> str:
    low = as_float(row.get("prior_20d_low"), np.nan)
    high = as_float(row.get("prior_20d_high"), np.nan)
    if side == "CALL" and np.isfinite(low):
        return f"Invalid if price loses prior 20d low near {low:.2f}"
    if side == "PUT" and np.isfinite(high):
        return f"Invalid if price reclaims prior 20d high near {high:.2f}"
    return "Invalidation requires manual chart check"


def build_options_play(row: pd.Series, scores: dict[str, float]) -> dict:
    final_risk = str(row.get("final_risk_action", row.get("target_status", "REVIEW"))).upper()
    event_gate = str(row.get("event_gate", row.get("event_gate_event", "REVIEW"))).upper()
    exec_status = str(row.get("execution_playbook_status", row.get("execution_status", "REVIEW"))).upper()
    options_strategy = str(row.get("options_strategy", "NO_OPTION_SIGNAL")).upper()
    rank_options = as_float(row.get("rank_options"), as_float(row.get("sig_options"), 50.0))
    flow_score = as_float(row.get("flow_score"), 50.0)
    gamma_score = as_float(row.get("gamma_score"), 50.0)
    iv_rank = as_float(row.get("iv_rank"), np.nan)
    pcr_vol = as_float(row.get("pcr_vol"), np.nan)
    uoa_flag = bool(row.get("uoa_flag", False))
    uoa_bear = bool(row.get("uoa_bear_flag", False))
    squeeze = bool(row.get("squeeze_risk", False))
    short_score = scores.get("Short-term", 0.0)
    medium_score = scores.get("Medium-term", 0.0)

    strategy_is_bullish = any(x in options_strategy for x in ["BULL", "LONG_CALL", "CALL"])
    strategy_is_protective = any(x in options_strategy for x in ["PROTECTIVE", "PUT", "HEDGE"])
    option_score = clamp(rank_options * 0.35 + flow_score * 0.25 + gamma_score * 0.20 + short_score * 0.20)
    call_score = clamp(
        rank_options * 0.32
        + flow_score * 0.28
        + gamma_score * 0.16
        + short_score * 0.16
        + (8 if strategy_is_bullish else 0)
        + (5 if uoa_flag else 0)
        - (15 if uoa_bear else 0)
    )
    put_score = clamp(
        (100 - short_score) * 0.18
        + gamma_score * 0.16
        + (22 if strategy_is_protective else 0)
        + (18 if final_risk in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"} else 0)
        + (10 if final_risk == "SIZE_DOWN" else 0)
        + (12 if uoa_bear else 0)
        + (10 if np.isfinite(pcr_vol) and pcr_vol < 0.35 else 0)
    )
    no_go = []
    call_blockers = []
    if event_gate == "REVIEW":
        no_go.append("event gate requires manual review")
        call_blockers.append("event gate is not clear")
    if exec_status in {"DATA_GAP", "BLOCK_NEW"}:
        no_go.append("execution/spread data needs manual check")
        call_blockers.append("execution or spread data is not clean")
    if np.isfinite(iv_rank) and iv_rank >= 75:
        no_go.append("IV rank is high; avoid naked long premium")
        call_blockers.append("IV is too high for naked long premium")
    if final_risk in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        no_go.append("risk gate forbids bullish new exposure")
        call_blockers.append("risk gate forbids bullish new exposure")
    elif final_risk == "SIZE_DOWN":
        no_go.append("risk gate allows only reduced/tiny research size")
        call_blockers.append("risk gate is size-down")
    if uoa_bear:
        call_blockers.append("bearish unusual options flow is present")

    side = "NONE"
    structure = "No option"
    permission = "WAIT_ONLY"
    expiry_bucket = "None"
    reason = []
    call_edge = (
        call_score >= 58
        or rank_options >= 68
        or flow_score >= 75
        or strategy_is_bullish
    ) and not uoa_bear
    put_edge = (
        put_score >= 42
        or strategy_is_protective
        or uoa_bear
        or (np.isfinite(pcr_vol) and pcr_vol < 0.35)
    )

    bearish_hedge = (
        final_risk in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}
        or "PROTECTIVE_PUT" in options_strategy
        or uoa_bear
        or (np.isfinite(pcr_vol) and pcr_vol < 0.30)
    )
    bullish_call = (
        final_risk not in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED", "SIZE_DOWN"}
        and event_gate != "BLOCK_NEW"
        and call_edge
        and call_score >= 58
        and not uoa_bear
    )

    if bearish_hedge and put_edge:
        side = "PUT"
        permission = "HEDGE_ONLY"
        expiry_bucket = "2-8 weeks"
        structure = "Put debit spread or protective put research"
        reason.append("risk/hedge conditions point to put protection, not bullish calls")
    elif final_risk == "SIZE_DOWN" and put_edge and not call_edge:
        side = "PUT"
        permission = "HEDGE_ONLY"
        expiry_bucket = "2-8 weeks"
        structure = "Put debit spread or protective put research"
        reason.append("size-down risk state supports hedge review, not upside chase")
    elif final_risk == "SIZE_DOWN" and call_edge:
        side = "CALL"
        permission = "CALL_BLOCKED_BY_RISK"
        expiry_bucket = "2-6 weeks only after risk gate clears"
        structure = "No call now; defined-risk call spread watch only if risk gate clears"
        reason.append("bullish call edge exists, but current risk gate blocks new bullish options")
    elif bullish_call and event_gate == "CLEAR" and exec_status not in {"BLOCK_NEW", "DATA_GAP"}:
        side = "CALL"
        permission = "CALL_SPREAD_RESEARCH"
        expiry_bucket = "2-6 weeks"
        structure = "Call debit spread research"
        if np.isfinite(iv_rank) and iv_rank >= 65:
            structure = "Defined-risk call spread only"
        reason.append("options flow and short-term score support a defined-risk bullish structure")
    elif bullish_call:
        side = "CALL"
        permission = "CALL_WATCH_ONLY"
        expiry_bucket = "2-6 weeks after manual gates"
        structure = "Call spread watch only"
        reason.append("bullish options signal exists, but manual gates are not clear")
    elif medium_score >= 65 and final_risk not in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        permission = "STOCK_ONLY"
        structure = "Stock or ETF paper only"
        reason.append("medium-term thesis is better expressed through underlying paper, not options")
    else:
        permission = "NO_NEW_OPTION"
        structure = "No option trade"
        reason.append("options evidence is not strong enough")

    if final_risk == "SIZE_DOWN" and permission in {"CALL_SPREAD_RESEARCH", "STOCK_ONLY"}:
        permission = "TINY_RESEARCH_ONLY"
        structure = "Tiny underlying paper only; no short-dated calls"
        side = "NONE"
        reason.append("risk gate size-down overrides bullish option structure")

    if no_go and permission in {"CALL_SPREAD_RESEARCH", "CALL_WATCH_ONLY"}:
        permission = "WAIT_ONLY"
        structure = "Wait; no option until no-go items clear"
        side = "NONE"
        reason.append("no-go conditions block option expression")

    trigger_side = "PUT" if side == "PUT" else "CALL"
    if call_edge and put_edge:
        pre_risk_edge = "Call edge + hedge edge"
    elif call_edge:
        pre_risk_edge = "Call edge"
    elif put_edge:
        pre_risk_edge = "Put / hedge edge"
    else:
        pre_risk_edge = "No clean option edge"

    if final_risk in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        primary_blocker = "Risk gate blocks bullish exposure"
        what_would_change = "Risk gate must improve before any bullish option can be researched."
    elif final_risk == "SIZE_DOWN" and call_edge:
        primary_blocker = "Risk gate is size-down"
        what_would_change = "Risk gate must move from SIZE_DOWN to CLEAR before call research can be allowed."
    elif event_gate == "REVIEW":
        primary_blocker = "Event gate needs review"
        what_would_change = "Event source review must clear before upgrading the route."
    elif exec_status in {"DATA_GAP", "BLOCK_NEW"}:
        primary_blocker = "Execution/spread data gap"
        what_would_change = "Spread, fill-rate, and execution assumptions must be checked manually."
    elif np.isfinite(iv_rank) and iv_rank >= 75:
        primary_blocker = "IV is expensive"
        what_would_change = "Use defined-risk spreads only, or wait for IV to cool."
    elif not call_edge and not put_edge:
        primary_blocker = "No strong option edge"
        what_would_change = "Need stronger flow, gamma, trigger, or event evidence."
    else:
        primary_blocker = "Manual confirmation"
        what_would_change = "Price trigger, source review, and risk gate must align."

    call_answer = "No call edge"
    if call_edge and permission in {"CALL_SPREAD_RESEARCH", "CALL_WATCH_ONLY"}:
        call_answer = "Call spread research allowed"
    elif call_edge:
        call_answer = "Call edge blocked: " + "; ".join(call_blockers or ["manual gates not clear"])

    put_answer = "No put edge"
    if permission == "HEDGE_ONLY":
        put_answer = "Put hedge review allowed"
    elif put_edge:
        put_answer = "Put watch only; hedge need not confirmed"

    if permission == "CALL_BLOCKED_BY_RISK":
        option_answer = "Call edge exists, but no call trade now"
    elif permission == "HEDGE_ONLY":
        option_answer = "Put hedge only"
    elif permission in {"CALL_SPREAD_RESEARCH", "CALL_WATCH_ONLY"}:
        option_answer = "Defined-risk call spread research"
    elif permission in {"STOCK_ONLY", "TINY_RESEARCH_ONLY"}:
        option_answer = "Underlying only; no options"
    else:
        option_answer = "No new option"

    return {
        "option_score": round(option_score, 1),
        "call_score": round(call_score, 1),
        "put_score": round(put_score, 1),
        "option_permission": permission,
        "option_side": side,
        "option_structure": structure,
        "option_expiry_bucket": expiry_bucket,
        "option_answer": option_answer,
        "pre_risk_edge": pre_risk_edge,
        "primary_blocker": primary_blocker,
        "what_would_change": what_would_change,
        "call_answer": call_answer,
        "put_answer": put_answer,
        "call_blockers": "; ".join(call_blockers) if call_blockers else "None",
        "call_trigger": trigger_text(row, "CALL"),
        "put_trigger": trigger_text(row, "PUT"),
        "option_invalidation": invalidation_text(row, trigger_side),
        "no_go_conditions": "; ".join(no_go) if no_go else "None",
        "option_reason": "; ".join(reason),
    }


def action_for_horizon(row: pd.Series, horizon: str, score: float, option_play: dict) -> tuple[str, str, str, str]:
    final_risk = str(row.get("final_risk_action", row.get("target_status", "REVIEW"))).upper()
    event_gate = str(row.get("event_gate", row.get("event_gate_event", "REVIEW"))).upper()
    quality = as_float(row.get("quality_score"), as_float(row.get("sig_quality"), 50.0))

    if final_risk in {"BLOCKED", "BLOCK_NEW"}:
        return "Blocked", "No new exposure", "NONE", "Risk gate blocks new exposure."
    if final_risk == "REDUCE_ONLY":
        return "Reduce / no new trade", "Risk reduction only", "PUT" if horizon == "Short-term" else "NONE", "Risk gate says reduce only."

    if horizon == "Short-term":
        if option_play["option_permission"] in {"CALL_SPREAD_RESEARCH", "CALL_WATCH_ONLY"}:
            return "Call setup watch", option_play["option_structure"], "CALL", option_play["option_reason"]
        if option_play["option_permission"] == "HEDGE_ONLY":
            return "Put hedge review", option_play["option_structure"], "PUT", option_play["option_reason"]
        if score >= 70 and final_risk == "SIZE_DOWN":
            return "Trigger watch only", "Tiny underlying paper only", "NONE", "Short-term score is strong, but risk gate requires size down."
        if score >= 60:
            return "Wait for trigger", "Underlying paper only", "NONE", "Needs price/volume confirmation."
        return "Skip short-term", "No trade", "NONE", "Short-term score is not strong enough."

    if horizon == "Medium-term":
        if score >= 72 and final_risk == "CLEAR" and event_gate == "CLEAR":
            return "Medium-term paper candidate", "Underlying paper", "NONE", "Signal stack supports a multi-week thesis."
        if score >= 62:
            return "Medium-term watch", "Underlying paper only after gates", "NONE", "Good enough to track, not enough to force action."
        return "No medium-term setup", "No trade", "NONE", "Medium-term score is too weak."

    if score >= 75 and quality >= 70 and final_risk == "CLEAR":
        return "Long-term research candidate", "Underlying only", "NONE", "Quality and signal stack support a long-term research thesis."
    if score >= 62 and quality >= 55:
        return "Long-term watch", "Underlying only", "NONE", "Some long-term support, but quality/risk is not fully confirmed."
    return "No long-term thesis yet", "No trade", "NONE", "Long-term quality or risk support is not sufficient."


def build_strategy_row(
    row: pd.Series,
    scores: dict[str, float],
    horizon_actions: dict[str, str],
    option_play: dict,
) -> dict:
    ticker = clean_ticker(row.get("ticker"))
    final_risk = str(row.get("final_risk_action", row.get("target_status", "REVIEW"))).upper()
    event_gate = str(row.get("event_gate", row.get("event_gate_event", "REVIEW"))).upper()
    exec_status = str(row.get("execution_playbook_status", row.get("execution_status", "REVIEW"))).upper()
    best_horizon = max(scores, key=scores.get)
    best_score = scores.get(best_horizon, 0.0)
    permission = str(option_play.get("option_permission", "NO_NEW_OPTION"))

    if final_risk in {"BLOCKED", "BLOCK_NEW", "REDUCE_ONLY"}:
        desk_action = "Risk reduction first"
        sleeve = "Risk control"
        allowed_tool = "No new bullish exposure"
    elif permission == "HEDGE_ONLY":
        desk_action = "Hedge review"
        sleeve = "Short-term risk hedge"
        allowed_tool = "Put debit spread or protective put research"
    elif permission == "CALL_BLOCKED_BY_RISK":
        desk_action = "Call edge blocked by risk"
        sleeve = "Short-term watchlist"
        allowed_tool = "No call now; revisit only after risk gate clears"
    elif permission in {"CALL_SPREAD_RESEARCH", "CALL_WATCH_ONLY"}:
        desk_action = "Call spread research"
        sleeve = "Short-term tactical"
        allowed_tool = option_play.get("option_structure", "Defined-risk call spread")
    elif permission in {"STOCK_ONLY", "TINY_RESEARCH_ONLY"}:
        desk_action = "Underlying only"
        sleeve = "Medium-term paper"
        allowed_tool = option_play.get("option_structure", "Underlying paper only")
    elif best_horizon == "Long-term" and best_score >= 62:
        desk_action = "Long-term watch"
        sleeve = "Long-term research"
        allowed_tool = "Underlying only after source and risk gates improve"
    elif best_horizon == "Medium-term" and best_score >= 58:
        desk_action = "Medium-term watch"
        sleeve = "Medium-term watchlist"
        allowed_tool = "Underlying paper only after confirmation"
    else:
        desk_action = "Wait / skip"
        sleeve = "Cash or backlog"
        allowed_tool = "No trade"

    if final_risk == "SIZE_DOWN":
        sizing_note = "Tiny paper/research size only; no short-dated premium chase."
    elif final_risk in {"REDUCE_ONLY", "BLOCKED", "BLOCK_NEW"}:
        sizing_note = "Reduce or hold cash first; no new bullish exposure."
    else:
        sizing_note = "Use normal paper sizing only after manual gates clear."

    if permission == "CALL_BLOCKED_BY_RISK":
        entry_trigger = option_play.get("call_trigger", "Wait for price confirmation")
        invalidation = "Do not act unless risk gate improves; " + option_play.get("option_invalidation", "")
    elif permission == "HEDGE_ONLY":
        entry_trigger = option_play.get("put_trigger", "Wait for downside confirmation")
        invalidation = option_play.get("option_invalidation", "Manual invalidation required")
    else:
        entry_trigger = option_play.get("call_trigger", "Wait for confirmation")
        invalidation = option_play.get("option_invalidation", "Manual invalidation required")

    source_summary = (
        "daily_picks_filtered.csv + options_signals.csv + final_risk_gate.csv + "
        "event_research_dossier.csv + execution_trade_plan.csv + desk_monitor_ticker_state.csv"
    )
    return {
        "ticker": ticker,
        "sector": row.get("sector", "Unknown"),
        "desk_action": desk_action,
        "sleeve": sleeve,
        "best_horizon": best_horizon,
        "best_score": round(best_score, 1),
        "allowed_tool": allowed_tool,
        "option_answer": option_play.get("option_answer", ""),
        "pre_risk_edge": option_play.get("pre_risk_edge", ""),
        "primary_blocker": option_play.get("primary_blocker", ""),
        "what_would_change": option_play.get("what_would_change", ""),
        "call_answer": option_play.get("call_answer", ""),
        "put_answer": option_play.get("put_answer", ""),
        "short_plan": horizon_actions.get("Short-term", ""),
        "medium_plan": horizon_actions.get("Medium-term", ""),
        "long_plan": horizon_actions.get("Long-term", ""),
        "entry_trigger": entry_trigger,
        "invalidation": invalidation,
        "sizing_note": sizing_note,
        "risk_action": final_risk,
        "event_gate": event_gate,
        "execution_status": exec_status,
        "no_go_conditions": option_play.get("no_go_conditions", "None"),
        "source_summary": source_summary,
    }


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    base = build_base()
    macro = read_json_safe(ROOT / "macro_signals.json", {})
    regime = read_json_safe(ROOT / "regime_current.json", {})
    if base.empty:
        state = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "overall_status": "NO_DATA",
            "truth": "No daily picks found.",
            "research_only": True,
            "no_broker_connection": True,
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), state

    matrix_rows = []
    summary_rows = []
    option_rows = []
    strategy_rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        scores = horizon_scores(row, macro, regime)
        option_play = build_options_play(row, scores)
        option_rows.append({
            "ticker": ticker,
            "sector": row.get("sector", "Unknown"),
            "alpha_score": as_float(row.get("alpha_score"), np.nan),
            "rank_options": as_float(row.get("rank_options"), np.nan),
            "iv_rank": as_float(row.get("iv_rank"), np.nan),
            "event_gate": row.get("event_gate", row.get("event_gate_event", "")),
            "final_risk_action": row.get("final_risk_action", row.get("target_status", "")),
            **option_play,
            "source_file": "daily_picks_filtered.csv / options_signals.csv / final_risk_gate.csv / event_research_dossier.csv",
            "research_only": True,
        })

        horizon_actions: dict[str, str] = {}
        horizon_scores_out: dict[str, float] = {}
        for horizon, days in [
            ("Short-term", "1-5 trading days"),
            ("Medium-term", "2-8 weeks"),
            ("Long-term", "3-12 months"),
        ]:
            score = scores[horizon]
            action, vehicle, side, why = action_for_horizon(row, horizon, score, option_play)
            penalty, penalty_reasons = risk_penalty(row, horizon)
            monitor_adj, monitor_reasons = monitor_adjustment(row, horizon)
            if horizon == "Short-term":
                trigger = option_play["call_trigger"] if side != "PUT" else option_play["put_trigger"]
                invalid = option_play["option_invalidation"]
            elif horizon == "Medium-term":
                trigger = "Wait for risk gate/event review to clear, then confirm trend holds."
                invalid = "Invalid if risk gate worsens or event source becomes blocked."
            else:
                trigger = "Only upgrade after quality, source coverage, and risk state support the thesis."
                invalid = "Invalid if quality score weakens or thesis relies only on short-term flow."
            reasons = [why] + penalty_reasons + monitor_reasons
            matrix_rows.append({
                "ticker": ticker,
                "sector": row.get("sector", "Unknown"),
                "timeframe": horizon,
                "horizon": days,
                "score": score,
                "action": action,
                "vehicle": vehicle,
                "option_side": side,
                "option_structure": option_play["option_structure"] if horizon == "Short-term" else "No option for this horizon",
                "trigger_to_watch": trigger,
                "invalidation": invalid,
                "size_rule": "Research only; risk gate controls size; no broker and no live orders.",
                "why": "; ".join([r for r in reasons if r]),
                "source_file": "daily_picks_filtered.csv / final_risk_gate.csv / options_signals.csv / event_research_dossier.csv / desk_monitor_ticker_state.csv",
            })
            horizon_actions[horizon] = action
            horizon_scores_out[horizon] = score

        summary_rows.append({
            "ticker": ticker,
            "sector": row.get("sector", "Unknown"),
            "alpha_score": as_float(row.get("alpha_score"), np.nan),
            "short_score": horizon_scores_out["Short-term"],
            "short_action": horizon_actions["Short-term"],
            "medium_score": horizon_scores_out["Medium-term"],
            "medium_action": horizon_actions["Medium-term"],
            "long_score": horizon_scores_out["Long-term"],
            "long_action": horizon_actions["Long-term"],
            "option_permission": option_play["option_permission"],
            "option_side": option_play["option_side"],
            "option_structure": option_play["option_structure"],
            "risk_action": row.get("final_risk_action", row.get("target_status", "")),
            "event_gate": row.get("event_gate", row.get("event_gate_event", "")),
            "top_signal": row.get("top_signal", ""),
        })
        strategy_rows.append(build_strategy_row(row, scores, horizon_actions, option_play))

    matrix = pd.DataFrame(matrix_rows)
    summary = pd.DataFrame(summary_rows).sort_values(["short_score", "medium_score", "long_score"], ascending=False)
    options = pd.DataFrame(option_rows).sort_values("option_score", ascending=False)
    strategy = pd.DataFrame(strategy_rows).sort_values(["desk_action", "best_score"], ascending=[True, False])
    permission_counts = options["option_permission"].astype(str).value_counts().to_dict() if not options.empty else {}
    permissions = options["option_permission"].astype(str) if not options.empty else pd.Series(dtype=str)
    state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "overall_status": "ACTIVE",
        "tickers": int(len(summary)),
        "timeframe_rows": int(len(matrix)),
        "call_research_or_watch": int(permissions.isin(["CALL_SPREAD_RESEARCH", "CALL_WATCH_ONLY"]).sum()),
        "call_blocked_by_risk": int(permissions.str.contains("CALL_BLOCKED", na=False).sum()),
        "put_or_hedge_only": int(permissions.str.contains("HEDGE|PUT", na=False).sum()),
        "stock_only_or_tiny": int(permissions.str.contains("STOCK|TINY", na=False).sum()),
        "wait_or_no_option": int(permissions.str.contains("WAIT|NO_NEW", na=False).sum()),
        "permission_counts": permission_counts,
        "truth": "Timeframe and options playbook only. Research dashboard, no broker connection, no live orders.",
        "research_only": True,
        "no_broker_connection": True,
    }
    return matrix, summary, options, strategy, state


def write_report(matrix: pd.DataFrame, summary: pd.DataFrame, options: pd.DataFrame, strategy: pd.DataFrame, state: dict) -> None:
    lines = [
        "# Canyon v9 Step 128 - Timeframe and Options Playbook",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Research-only. No broker connection. No live orders.",
        "",
        f"- Tickers: {state.get('tickers', 0)}",
        f"- Timeframe rows: {state.get('timeframe_rows', 0)}",
        f"- Call research/watch: {state.get('call_research_or_watch', 0)}",
        f"- Call edge blocked by risk: {state.get('call_blocked_by_risk', 0)}",
        f"- Put or hedge-only: {state.get('put_or_hedge_only', 0)}",
        f"- Stock-only or tiny: {state.get('stock_only_or_tiny', 0)}",
        f"- Wait/no option: {state.get('wait_or_no_option', 0)}",
        "",
        "## Output files",
        "",
        "- `timeframe_decision_matrix.csv`",
        "- `ticker_timeframe_summary.csv`",
        "- `options_playbook.csv`",
        "- `strategy_route_playbook.csv`",
        "- `timeframe_options_playbook_state.json`",
    ]
    if not options.empty:
        lines.extend(["", "## Option permission counts", ""])
        for key, val in options["option_permission"].astype(str).value_counts().to_dict().items():
            lines.append(f"- {key}: {val}")
    if not strategy.empty:
        lines.extend(["", "## Desk action counts", ""])
        for key, val in strategy["desk_action"].astype(str).value_counts().to_dict().items():
            lines.append(f"- {key}: {val}")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    matrix, summary, options, strategy, state = build_outputs()
    matrix.to_csv(OUT_MATRIX, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    options.to_csv(OUT_OPTIONS, index=False)
    strategy.to_csv(OUT_STRATEGY, index=False)
    write_json(OUT_STATE, state)
    write_report(matrix, summary, options, strategy, state)
    print(f"[step128] wrote {OUT_MATRIX.name}: {len(matrix)} rows")
    print(f"[step128] tickers={state.get('tickers')} call_watch={state.get('call_research_or_watch')} hedge={state.get('put_or_hedge_only')}")


if __name__ == "__main__":
    main()
