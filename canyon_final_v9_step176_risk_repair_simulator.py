#!/usr/bin/env python3
"""
Canyon v9 Step 176 - Risk Repair Simulator.

Research-only. No broker connection. No live orders.

Step175 says what must be reduced. Step176 simulates several repair paths and
estimates how the risk desk would change:
  - portfolio gross exposure
  - annualized volatility
  - 1d VaR / CVaR
  - sector concentration
  - option unlock projection after risk repair

The simulator uses local price-cache returns when available. If local return
coverage is thin, it falls back to calibrated scaling from the active risk desk.
It is a research planning tool only; it does not send orders or change the
paper ledger.

Outputs:
  risk_repair_scenario_summary.csv
  risk_repair_ticker_plan.csv
  risk_repair_metric_impact.csv
  risk_repair_sector_impact.csv
  risk_repair_option_projection.csv
  risk_repair_state.json
  risk_repair_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    annualized_vol,
    df_to_markdown,
    get_returns,
    read_csv_safe,
    read_json_safe,
    today_str,
    var_cvar,
    write_json,
    write_markdown_report,
)


OUT_SUMMARY = ROOT / "risk_repair_scenario_summary.csv"
OUT_TICKER_PLAN = ROOT / "risk_repair_ticker_plan.csv"
OUT_METRICS = ROOT / "risk_repair_metric_impact.csv"
OUT_SECTOR = ROOT / "risk_repair_sector_impact.csv"
OUT_OPTIONS = ROOT / "risk_repair_option_projection.csv"
OUT_STATE = ROOT / "risk_repair_state.json"
OUT_REPORT = ROOT / "risk_repair_report.md"


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def as_upper(value: Any, default: str = "") -> str:
    text = as_text(value, default)
    return text.upper() if text else default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def lookup_limit_pct(budget: pd.DataFrame, item: str, default_pct: float) -> float:
    if budget.empty or "budget_item" not in budget.columns:
        return default_pct
    mask = budget["budget_item"].astype(str).str.lower().eq(item.lower())
    if not mask.any() or "limit_value" not in budget.columns:
        return default_pct
    value = safe_float(budget.loc[mask, "limit_value"].iloc[0], default_pct / 100.0)
    return value * 100.0 if abs(value) <= 2.0 else value


def scenario_targets(action: pd.DataFrame, gross_target_pct: float) -> dict[str, dict[str, Any]]:
    current = action.set_index("ticker")["current_weight_pct"].astype(float)
    risk_target = action.set_index("ticker")["sequencer_target_weight_pct"].astype(float)
    hard_mask = action.set_index("ticker")["risk_unlock_status"].astype(str).str.upper().eq("REDUCE_ONLY_LOCKED")

    gross_target_pct = max(0.0, min(100.0, gross_target_pct))
    current_sum = float(current.sum())
    gross_scale = gross_target_pct / current_sum if current_sum > 0 else 0.0

    master_gross = current * min(1.0, gross_scale)

    hard_repair = current.copy()
    hard_repair.loc[hard_mask] = risk_target.loc[hard_mask]
    hard_total = float(hard_repair.sum())
    if hard_total > gross_target_pct and gross_target_pct > 0:
        nonhard = ~hard_mask
        hard_fixed = float(hard_repair.loc[hard_mask].sum())
        remaining_budget = max(0.0, gross_target_pct - hard_fixed)
        nonhard_sum = float(hard_repair.loc[nonhard].sum())
        if nonhard_sum > 0:
            hard_repair.loc[nonhard] = hard_repair.loc[nonhard] * (remaining_budget / nonhard_sum)
        else:
            hard_repair = hard_repair * (gross_target_pct / hard_total)

    return {
        "CURRENT_BOOK": {
            "target": current,
            "description": "No repair. Shows why the current book remains size-down.",
            "repair_intent": "Observe current risk only",
        },
        "MASTER_GROSS_70": {
            "target": master_gross,
            "description": "Scale the whole risk queue to the portfolio gross target.",
            "repair_intent": "Portfolio-level vol/gross repair",
        },
        "HARD_RISK_REPAIR_70": {
            "target": hard_repair,
            "description": "Cut REDUCE_ONLY tickers to risk targets, then keep total gross near the master target.",
            "repair_intent": "Repair hard single-name locks first",
        },
        "TICKER_RISK_TARGET": {
            "target": risk_target,
            "description": "Cut every ticker to its Step175 risk target. Leaves large cash buffer.",
            "repair_intent": "Ticker-level risk repair",
        },
        "STRICT_DEFENSIVE": {
            "target": risk_target * 0.5,
            "description": "Half of risk target for monitor/event/execution stress.",
            "repair_intent": "Defensive watch-only state",
        },
    }


def official_current_metrics(overview: dict[str, Any], portfolio_var: pd.DataFrame) -> dict[str, float]:
    row = portfolio_var.iloc[0].to_dict() if not portfolio_var.empty else {}
    annual_vol_pct = safe_float(overview.get("annual_vol_pct"), safe_float(row.get("annual_vol"), 0.0) * 100.0)
    var_pct = safe_float(overview.get("var_95_1d_pct"), safe_float(row.get("var_95_1d"), 0.0) * 100.0)
    cvar_pct = safe_float(overview.get("cvar_95_1d_pct"), safe_float(row.get("cvar_95_1d"), 0.0) * 100.0)
    var20_pct = safe_float(overview.get("var_95_20d_pct"), safe_float(row.get("var_95_20d"), 0.0) * 100.0)
    cvar20_pct = safe_float(overview.get("cvar_95_20d_pct"), safe_float(row.get("cvar_95_20d"), 0.0) * 100.0)
    return {
        "annual_vol_pct": annual_vol_pct,
        "var_95_1d_pct": var_pct,
        "cvar_95_1d_pct": cvar_pct,
        "var_95_20d_pct": var20_pct,
        "cvar_95_20d_pct": cvar20_pct,
    }


def scenario_return_metrics(
    tickers: list[str],
    current_weights_pct: pd.Series,
    target_weights_pct: pd.Series,
    official: dict[str, float],
) -> dict[str, Any]:
    rets = get_returns(tickers, lookback=252)
    gross_current = float(current_weights_pct.sum())
    gross_target = float(target_weights_pct.sum())
    gross_scale = gross_target / gross_current if gross_current > 0 else 0.0

    if rets.empty:
        scale = gross_scale
        return {
            "annual_vol_pct": official["annual_vol_pct"] * scale,
            "var_95_1d_pct": official["var_95_1d_pct"] * scale,
            "cvar_95_1d_pct": official["cvar_95_1d_pct"] * scale,
            "var_95_20d_pct": official["var_95_20d_pct"] * scale,
            "cvar_95_20d_pct": official["cvar_95_20d_pct"] * scale,
            "risk_estimation_source": "gross_scaled_fallback",
            "return_coverage_count": 0,
            "return_sample_days": 0,
        }

    available = [t for t in tickers if t in rets.columns]
    if not available:
        scale = gross_scale
        return {
            "annual_vol_pct": official["annual_vol_pct"] * scale,
            "var_95_1d_pct": official["var_95_1d_pct"] * scale,
            "cvar_95_1d_pct": official["cvar_95_1d_pct"] * scale,
            "var_95_20d_pct": official["var_95_20d_pct"] * scale,
            "cvar_95_20d_pct": official["cvar_95_20d_pct"] * scale,
            "risk_estimation_source": "gross_scaled_no_ticker_coverage",
            "return_coverage_count": 0,
            "return_sample_days": int(len(rets)),
        }

    r = rets[available].dropna(how="all")
    current_w = (current_weights_pct.reindex(available).fillna(0.0) / 100.0).astype(float)
    target_w = (target_weights_pct.reindex(available).fillna(0.0) / 100.0).astype(float)
    current_port = (r * current_w).sum(axis=1)
    target_port = (r * target_w).sum(axis=1)

    raw_current_vol = annualized_vol(current_port)
    raw_target_vol = annualized_vol(target_port)
    raw_current_var, raw_current_cvar = var_cvar(current_port, 0.95)
    raw_target_var, raw_target_cvar = var_cvar(target_port, 0.95)

    def scaled(official_value: float, raw_target: float, raw_current: float) -> float:
        if raw_current and np.isfinite(raw_current) and raw_current > 0 and np.isfinite(raw_target):
            return official_value * (raw_target / raw_current)
        return official_value * gross_scale

    var1 = scaled(official["var_95_1d_pct"], raw_target_var, raw_current_var)
    cvar1 = scaled(official["cvar_95_1d_pct"], raw_target_cvar, raw_current_cvar)
    vol = scaled(official["annual_vol_pct"], raw_target_vol, raw_current_vol)
    return {
        "annual_vol_pct": vol,
        "var_95_1d_pct": var1,
        "cvar_95_1d_pct": cvar1,
        "var_95_20d_pct": var1 * np.sqrt(20.0),
        "cvar_95_20d_pct": cvar1 * np.sqrt(20.0),
        "risk_estimation_source": "price_cache_scaled_to_active_risk_desk",
        "return_coverage_count": int(len(available)),
        "return_sample_days": int(len(r)),
    }


def metric_status(value_pct: float, limit_pct: float, soft_buffer: float = 1.1) -> str:
    if value_pct <= limit_pct:
        return "CLEAR"
    if value_pct <= limit_pct * soft_buffer:
        return "REVIEW"
    return "SIZE_DOWN"


def overall_status(row: dict[str, Any], limits: dict[str, float]) -> str:
    gross_ok = row["gross_exposure_pct"] <= limits["gross_pct"] + 1e-6
    vol_ok = row["annual_vol_pct"] <= limits["annual_vol_pct"] + 1e-6
    var_ok = row["var_95_1d_pct"] <= limits["var_95_1d_pct"] + 1e-6
    cvar_ok = row["cvar_95_1d_pct"] <= limits["cvar_95_1d_pct"] + 1e-6
    if gross_ok and vol_ok and var_ok and cvar_ok:
        return "RISK_REPAIRED_FOR_MANUAL_REVIEW"
    if gross_ok and row["annual_vol_pct"] <= limits["annual_vol_pct"] * 1.1:
        return "PARTIAL_REPAIR_REVIEW"
    if gross_ok:
        return "GROSS_REPAIRED_BUT_RISK_HIGH"
    return "STILL_SIZE_DOWN"


def build_summary_and_metrics(
    action: pd.DataFrame,
    scenarios: dict[str, dict[str, Any]],
    overview: dict[str, Any],
    portfolio_var: pd.DataFrame,
    budget: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = action["ticker"].astype(str).str.upper().tolist()
    current_weights = action.set_index("ticker")["current_weight_pct"].astype(float)
    official = official_current_metrics(overview, portfolio_var)
    limits = {
        "gross_pct": safe_float(overview.get("recommended_gross_exposure"), 0.7) * 100.0,
        "annual_vol_pct": safe_float(overview.get("target_vol_pct"), 15.0),
        "var_95_1d_pct": lookup_limit_pct(budget, "Portfolio 1d VaR 95%", 2.0),
        "cvar_95_1d_pct": lookup_limit_pct(budget, "Portfolio 1d CVaR 95%", 3.5),
    }

    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for rank, (scenario, spec) in enumerate(scenarios.items(), start=1):
        target = spec["target"].astype(float)
        metrics = scenario_return_metrics(tickers, current_weights, target, official)
        gross = float(target.sum())
        reduction = max(0.0, float(current_weights.sum()) - gross)
        row = {
            "scenario_rank": rank,
            "scenario": scenario,
            "repair_intent": spec["repair_intent"],
            "description": spec["description"],
            "gross_exposure_pct": round(gross, 4),
            "cash_buffer_pct": round(max(0.0, 100.0 - gross), 4),
            "reduction_vs_current_pct_points": round(reduction, 4),
            "annual_vol_pct": round(float(metrics["annual_vol_pct"]), 4),
            "annual_vol_limit_pct": round(limits["annual_vol_pct"], 4),
            "var_95_1d_pct": round(float(metrics["var_95_1d_pct"]), 4),
            "var_95_1d_limit_pct": round(limits["var_95_1d_pct"], 4),
            "cvar_95_1d_pct": round(float(metrics["cvar_95_1d_pct"]), 4),
            "cvar_95_1d_limit_pct": round(limits["cvar_95_1d_pct"], 4),
            "var_95_20d_pct": round(float(metrics["var_95_20d_pct"]), 4),
            "cvar_95_20d_pct": round(float(metrics["cvar_95_20d_pct"]), 4),
            "return_coverage_count": metrics["return_coverage_count"],
            "return_sample_days": metrics["return_sample_days"],
            "risk_estimation_source": metrics["risk_estimation_source"],
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        row["gross_status"] = metric_status(row["gross_exposure_pct"], limits["gross_pct"])
        row["vol_status"] = metric_status(row["annual_vol_pct"], limits["annual_vol_pct"])
        row["var_status"] = metric_status(row["var_95_1d_pct"], limits["var_95_1d_pct"])
        row["cvar_status"] = metric_status(row["cvar_95_1d_pct"], limits["cvar_95_1d_pct"])
        row["overall_repair_status"] = overall_status(row, limits)
        summary_rows.append(row)

        for metric, value_key, limit_key, status_key in [
            ("Gross exposure", "gross_exposure_pct", "gross_pct", "gross_status"),
            ("Annual volatility", "annual_vol_pct", "annual_vol_pct", "vol_status"),
            ("1d VaR 95", "var_95_1d_pct", "var_95_1d_pct", "var_status"),
            ("1d CVaR 95", "cvar_95_1d_pct", "cvar_95_1d_pct", "cvar_status"),
        ]:
            metric_rows.append({
                "scenario": scenario,
                "metric": metric,
                "current_value_pct": round(100.0 if metric == "Gross exposure" else official[value_key], 4),
                "simulated_value_pct": row[value_key],
                "limit_pct": round(limits[limit_key], 4),
                "improvement_pct_points": round((100.0 if metric == "Gross exposure" else official[value_key]) - row[value_key], 4),
                "status": row[status_key],
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(metric_rows)


def build_ticker_plan(action: pd.DataFrame, scenarios: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    action_idx = action.set_index("ticker")
    for scenario, spec in scenarios.items():
        target = spec["target"].astype(float)
        for ticker, target_weight in target.items():
            row = action_idx.loc[ticker]
            current = safe_float(row.get("current_weight_pct"))
            risk_target = safe_float(row.get("sequencer_target_weight_pct"))
            reduction = max(0.0, current - float(target_weight))
            repaired = float(target_weight) <= risk_target + 0.01
            status = "TICKER_RISK_REPAIRED" if repaired else "STILL_ABOVE_TICKER_RISK_TARGET"
            if scenario == "CURRENT_BOOK":
                status = "CURRENT_LOCKED_STATE"
            elif scenario == "STRICT_DEFENSIVE":
                status = "DEFENSIVE_WATCH_ONLY"
            rows.append({
                "scenario": scenario,
                "ticker": ticker,
                "sector": row.get("sector"),
                "current_weight_pct": round(current, 4),
                "simulated_weight_pct": round(float(target_weight), 4),
                "risk_target_weight_pct": round(risk_target, 4),
                "reduction_pct_points": round(reduction, 4),
                "reduction_pct_of_current": round(reduction / current, 4) if current > 0 else 0.0,
                "ticker_repair_status": status,
                "original_risk_unlock_status": row.get("risk_unlock_status"),
                "first_risk_lock": row.get("first_risk_lock"),
                "what_clears": "Single-name size pressure clears only if simulated weight is at/below risk target.",
                "what_remains": "Monitor, spread/TCA, event proof, and option route checks still require source-backed review.",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return pd.DataFrame(rows)


def build_sector_impact(action: pd.DataFrame, scenarios: dict[str, dict[str, Any]], sector_active: pd.DataFrame) -> pd.DataFrame:
    cap_map: dict[str, float] = {}
    if not sector_active.empty and {"sector", "sector_cap"}.issubset(sector_active.columns):
        cap_map = {
            as_text(row["sector"]): safe_float(row["sector_cap"]) * 100.0
            for _, row in sector_active.iterrows()
        }
    current_sector = action.groupby("sector")["current_weight_pct"].sum().to_dict()
    rows: list[dict[str, Any]] = []
    sector_series = action.set_index("ticker")["sector"]
    for scenario, spec in scenarios.items():
        target = spec["target"].astype(float)
        joined = pd.DataFrame({"target_weight_pct": target, "sector": sector_series})
        for sector, sub in joined.groupby("sector"):
            current = float(current_sector.get(sector, 0.0))
            simulated = float(sub["target_weight_pct"].sum())
            cap = cap_map.get(as_text(sector), 35.0)
            used = simulated / cap if cap > 0 else np.nan
            if simulated <= cap * 0.75:
                status = "CLEAR"
            elif simulated <= cap:
                status = "REVIEW"
            else:
                status = "SIZE_DOWN"
            rows.append({
                "scenario": scenario,
                "sector": sector,
                "current_sector_weight_pct": round(current, 4),
                "simulated_sector_weight_pct": round(simulated, 4),
                "sector_cap_pct": round(cap, 4),
                "cap_used_pct": round(used * 100.0, 2) if np.isfinite(used) else np.nan,
                "sector_repair_status": status,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return pd.DataFrame(rows)


def build_option_projection(
    action: pd.DataFrame,
    scenarios: dict[str, dict[str, Any]],
    option_bridge: pd.DataFrame,
    option_blockers: pd.DataFrame,
) -> pd.DataFrame:
    action_idx = action.set_index("ticker")
    bridge_idx = option_bridge.set_index("ticker") if not option_bridge.empty and "ticker" in option_bridge.columns else pd.DataFrame()
    blocker_flags: dict[str, dict[str, bool]] = {}
    if not option_blockers.empty and "ticker" in option_blockers.columns:
        for ticker, sub in option_blockers.groupby(option_blockers["ticker"].astype(str).str.upper()):
            blocker_flags[ticker] = {
                "monitor": bool(sub.get("monitor_blocker", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
                "execution": bool(sub.get("execution_blocker", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
                "event": bool(sub.get("event_proof_blocker", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
                "spread": bool(sub.get("spread_data_blocker", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
                "greeks": bool(sub.get("greeks_iv_gamma_blocker", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
            }

    rows: list[dict[str, Any]] = []
    for scenario, spec in scenarios.items():
        target = spec["target"].astype(float)
        for ticker, target_weight in target.items():
            row = action_idx.loc[ticker]
            risk_target = safe_float(row.get("sequencer_target_weight_pct"))
            risk_repaired = float(target_weight) <= risk_target + 0.01
            bridge = bridge_idx.loc[ticker] if not bridge_idx.empty and ticker in bridge_idx.index else pd.Series(dtype=object)
            call_status = as_upper(bridge.get("call_unlock_status"), as_upper(row.get("call_unlock_status"), "NO_CALL_DATA"))
            put_status = as_upper(bridge.get("put_hedge_unlock_status"), as_upper(row.get("put_hedge_unlock_status"), "NO_PUT_DATA"))
            flags = blocker_flags.get(ticker, {})
            remaining: list[str] = []
            if flags.get("monitor") or as_upper(row.get("monitor_status")) in {"CRITICAL", "WARNING"}:
                remaining.append("monitor")
            if flags.get("execution") or flags.get("spread") or as_upper(row.get("spread_status")) == "DATA_GAP":
                remaining.append("spread/TCA")
            if flags.get("event"):
                remaining.append("event proof")
            if flags.get("greeks"):
                remaining.append("IV/Greeks/Gamma")
            if "NO_CALL" in call_status:
                remaining.append("no call thesis")

            if not risk_repaired:
                projection = "NO_BULLISH_OPTION_RISK_NOT_REPAIRED"
                permission = "NO_NEW_OPTION"
            elif "NO_CALL" not in call_status and not remaining:
                projection = "DEFINED_RISK_CALL_REVIEW_POSSIBLE"
                permission = "DEFINED_RISK_OPTION_RESEARCH_ONLY"
            elif "PUT" in put_status or "HEDGE" in put_status:
                projection = "HEDGE_RESEARCH_ONLY_AFTER_RISK_REPAIR"
                permission = "PUT_OR_HEDGE_RESEARCH_ONLY"
            else:
                projection = "RISK_REPAIRED_BUT_NON_RISK_BLOCKERS_REMAIN"
                permission = "UNDERLYING_REVIEW_ONLY"

            rows.append({
                "scenario": scenario,
                "ticker": ticker,
                "sector": row.get("sector"),
                "simulated_weight_pct": round(float(target_weight), 4),
                "risk_target_weight_pct": round(risk_target, 4),
                "risk_repaired": bool(risk_repaired),
                "call_unlock_status_before_repair": call_status,
                "put_hedge_status_before_repair": put_status,
                "option_projection": projection,
                "option_permission_projection": permission,
                "remaining_non_risk_blockers": "; ".join(dict.fromkeys(remaining)) if remaining else "none",
                "required_next_proof": as_text(bridge.get("next_required_proof"), row.get("unlock_sequence")),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return pd.DataFrame(rows)


def build_state(summary: pd.DataFrame, options: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_RISK_REPAIR_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    repaired = summary[summary["overall_repair_status"].astype(str).eq("RISK_REPAIRED_FOR_MANUAL_REVIEW")]
    if repaired.empty:
        recommended = summary.sort_values(["gross_status", "vol_status", "scenario_rank"]).iloc[0]
    else:
        recommended = repaired.sort_values("scenario_rank").iloc[0]
    lowest_risk = summary.sort_values(["annual_vol_pct", "gross_exposure_pct"]).iloc[0]
    possible_option_review = 0
    if not options.empty and "option_permission_projection" in options.columns:
        possible_option_review = int(options["option_permission_projection"].astype(str).str.contains("OPTION_RESEARCH|HEDGE", regex=True).sum())
    return {
        "date": today_str(),
        "overall_status": "RISK_REPAIR_SIMULATOR_ACTIVE",
        "scenario_count": int(len(summary)),
        "manual_review_scenario_count": int(len(repaired)),
        "recommended_repair_scenario": as_text(recommended.get("scenario")),
        "recommended_repair_status": as_text(recommended.get("overall_repair_status")),
        "recommended_scenario_annual_vol_pct": round(safe_float(recommended.get("annual_vol_pct")), 4),
        "recommended_scenario_gross_pct": round(safe_float(recommended.get("gross_exposure_pct")), 4),
        "lowest_risk_scenario": as_text(lowest_risk.get("scenario")),
        "lowest_risk_scenario_status": as_text(lowest_risk.get("overall_repair_status")),
        "lowest_risk_annual_vol_pct": round(safe_float(lowest_risk.get("annual_vol_pct")), 4),
        "lowest_risk_gross_pct": round(safe_float(lowest_risk.get("gross_exposure_pct")), 4),
        "option_research_projection_rows": possible_option_review,
        "truth": "Scenario output is research planning only. It cannot change positions, send orders, or override the risk gate.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "summary": OUT_SUMMARY.name,
            "ticker_plan": OUT_TICKER_PLAN.name,
            "metric_impact": OUT_METRICS.name,
            "sector_impact": OUT_SECTOR.name,
            "option_projection": OUT_OPTIONS.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    action = read_csv_safe(ROOT / "risk_unlock_action_board.csv")
    if action.empty:
        empty = pd.DataFrame()
        for path in [OUT_SUMMARY, OUT_TICKER_PLAN, OUT_METRICS, OUT_SECTOR, OUT_OPTIONS]:
            empty.to_csv(path, index=False)
        state = build_state(empty, empty)
        write_json(OUT_STATE, state)
        write_markdown_report(OUT_REPORT, "Canyon v9 Step 176 - Risk Repair Simulator", ["## No data\nRun Step175 first."])
        return state

    action = action.copy()
    action["ticker"] = action["ticker"].astype(str).str.upper()
    for col in ["current_weight_pct", "sequencer_target_weight_pct"]:
        action[col] = pd.to_numeric(action[col], errors="coerce").fillna(0.0)

    overview = read_json_safe(ROOT / "risk_desk_overview.json")
    portfolio_var = read_csv_safe(ROOT / "portfolio_var_cvar_summary.csv")
    budget = read_csv_safe(ROOT / "institutional_risk_budget_summary.csv")
    sector_active = read_csv_safe(ROOT / "sector_active_exposure.csv")
    option_bridge = read_csv_safe(ROOT / "risk_unlock_option_bridge.csv")
    option_blockers = read_csv_safe(ROOT / "option_unlock_blocker_attribution.csv")

    gross_target_pct = safe_float(overview.get("recommended_gross_exposure"), 0.7) * 100.0
    scenarios = scenario_targets(action, gross_target_pct)
    summary, metrics = build_summary_and_metrics(action, scenarios, overview, portfolio_var, budget)
    ticker_plan = build_ticker_plan(action, scenarios)
    sector_impact = build_sector_impact(action, scenarios, sector_active)
    option_projection = build_option_projection(action, scenarios, option_bridge, option_blockers)
    state = build_state(summary, option_projection)

    summary.to_csv(OUT_SUMMARY, index=False)
    ticker_plan.to_csv(OUT_TICKER_PLAN, index=False)
    metrics.to_csv(OUT_METRICS, index=False)
    sector_impact.to_csv(OUT_SECTOR, index=False)
    option_projection.to_csv(OUT_OPTIONS, index=False)
    write_json(OUT_STATE, state)

    summary_cols = [c for c in [
        "scenario", "gross_exposure_pct", "cash_buffer_pct",
        "annual_vol_pct", "annual_vol_limit_pct", "var_95_1d_pct",
        "var_95_1d_limit_pct", "cvar_95_1d_pct", "cvar_95_1d_limit_pct",
        "overall_repair_status", "risk_estimation_source",
    ] if c in summary.columns]
    ticker_cols = [c for c in [
        "scenario", "ticker", "current_weight_pct", "simulated_weight_pct",
        "risk_target_weight_pct", "reduction_pct_points",
        "ticker_repair_status", "first_risk_lock",
    ] if c in ticker_plan.columns]
    option_cols = [c for c in [
        "scenario", "ticker", "risk_repaired", "call_unlock_status_before_repair",
        "option_projection", "option_permission_projection",
        "remaining_non_risk_blockers",
    ] if c in option_projection.columns]

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Scenarios: {state.get('scenario_count')}\n"
        f"- Manual-review-ready scenarios: {state.get('manual_review_scenario_count')}\n"
        f"- Recommended repair scenario: {state.get('recommended_repair_scenario')} ({state.get('recommended_repair_status')})\n"
        f"- Recommended scenario vol/gross: {state.get('recommended_scenario_annual_vol_pct')}% / {state.get('recommended_scenario_gross_pct')}%\n"
        f"- Lowest-risk scenario: {state.get('lowest_risk_scenario')} ({state.get('lowest_risk_scenario_status')})\n",
        "## Scenario summary\n" + df_to_markdown(summary[summary_cols] if summary_cols else summary, 20),
        "## Ticker repair plan\n" + df_to_markdown(ticker_plan[ticker_cols] if ticker_cols else ticker_plan, 80),
        "## Sector impact\n" + df_to_markdown(sector_impact, 80),
        "## Option projection\n" + df_to_markdown(option_projection[option_cols] if option_cols else option_projection, 80),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- Scenario output does not change the paper ledger.\n"
        "- Options still require spread/TCA, IV/Greeks, event proof, and monitor checks after risk repair.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 176 - Risk Repair Simulator", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 176 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Scenarios: {state.get('scenario_count')}")
    print(f"Recommended scenario: {state.get('recommended_repair_scenario')} ({state.get('recommended_repair_status')})")
    print("Outputs:")
    for path in [OUT_SUMMARY, OUT_TICKER_PLAN, OUT_METRICS, OUT_SECTOR, OUT_OPTIONS, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
