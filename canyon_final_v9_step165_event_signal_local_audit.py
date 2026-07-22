#!/usr/bin/env python3
"""
Canyon v9 Step 165 - Event Signal Local Audit.

Research-only. No broker connection. No live orders.

Step164 decides which event-derived rows are allowed into a local audit. Step165
checks whether those event signals were followed by the expected price move.
It keeps two timing lenses separate:

1. source_publish lens: event reaction from the headline publish date.
2. model_first_seen lens: forward return only after this local system first saw
   the event. If the price window has not elapsed, it stays pending.

This is intentionally local-audit-only. Local yfinance/proxy prices do not make
the event backtest institution-grade point-in-time evidence.

Outputs:
  event_signal_local_audit_returns.csv
  event_signal_local_audit_summary.csv
  event_signal_failure_modes.csv
  event_signal_audit_state.json
  event_signal_local_audit_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    load_price_cache,
    pct,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


SAFE_PANEL = ROOT / "pit_safe_event_signal_panel.csv"
ADMISSIBILITY = ROOT / "event_backtest_admissibility.csv"

OUT_RETURNS = ROOT / "event_signal_local_audit_returns.csv"
OUT_SUMMARY = ROOT / "event_signal_local_audit_summary.csv"
OUT_FAILURES = ROOT / "event_signal_failure_modes.csv"
OUT_STATE = ROOT / "event_signal_audit_state.json"
OUT_REPORT = ROOT / "event_signal_local_audit_report.md"

HORIZONS = [1, 3, 5, 10]
MIN_DIRECTIONAL_MOVE = 0.0025
LOCAL_TRUTH_LABEL = "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL"


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_time(value: Any) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return pd.NaT
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(None)
    except Exception:
        try:
            ts = ts.tz_localize(None)
        except Exception:
            pass
    return pd.Timestamp(ts)


def load_event_panel() -> pd.DataFrame:
    panel = read_csv_safe(SAFE_PANEL)
    if panel.empty:
        panel = read_csv_safe(ADMISSIBILITY)
        if not panel.empty and "can_enter_local_event_backtest" in panel.columns:
            panel = panel[panel["can_enter_local_event_backtest"].astype(bool)].copy()
    if panel.empty:
        return pd.DataFrame()

    out = panel.copy()
    ticker_col = "target_ticker" if "target_ticker" in out.columns else "ticker"
    out["target_ticker"] = out[ticker_col].map(clean_ticker)
    out = out[out["target_ticker"] != ""].copy()
    out["market_tone"] = out.get("market_tone", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    out["option_side"] = out.get("option_side", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    out["impact_score"] = pd.to_numeric(out.get("impact_score", np.nan), errors="coerce")
    out["source_publish_time"] = out.get("source_publish_time", out.get("published", "")).map(safe_time)
    out["first_seen_time"] = out.get("first_seen_time", out.get("trade_allowed_after_time", "")).map(safe_time)
    out["event_admissibility"] = out.get("event_admissibility", "LOCAL_EVENT_BACKTEST_REVIEW").fillna("LOCAL_EVENT_BACKTEST_REVIEW").astype(str)
    out["research_permission"] = out.get("research_permission", LOCAL_TRUTH_LABEL).fillna(LOCAL_TRUTH_LABEL).astype(str)
    out["truth_label"] = out.get("truth_label", LOCAL_TRUTH_LABEL).fillna(LOCAL_TRUTH_LABEL).astype(str)
    out["_has_event_id"] = out.get("event_id", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).astype(int)
    out["_has_relation_layer"] = out.get("relation_layer", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).astype(int)
    out["_has_causal_confidence"] = pd.to_numeric(out.get("causal_confidence_score", np.nan), errors="coerce").notna().astype(int)
    out["_dedupe_key"] = (
        out["target_ticker"].astype(str) + "|" +
        out.get("link", pd.Series(dtype=str)).fillna("").astype(str) + "|" +
        out.get("headline", pd.Series(dtype=str)).fillna("").astype(str).str.slice(0, 240) + "|" +
        out["event_admissibility"].astype(str)
    )
    out = (
        out.sort_values(["_has_event_id", "_has_relation_layer", "_has_causal_confidence"], ascending=False)
           .drop_duplicates("_dedupe_key", keep="first")
           .drop(columns=["_has_event_id", "_has_relation_layer", "_has_causal_confidence", "_dedupe_key"], errors="ignore")
           .reset_index(drop=True)
    )
    return out


def expected_direction(row: pd.Series) -> tuple[int, str]:
    tone = str(row.get("market_tone", "")).upper()
    route = str(row.get("suggested_research_route", "")).upper()
    option_side = str(row.get("option_side", "")).upper()
    impact = safe_float(row.get("impact_score"), 0.0)
    relation = str(row.get("target_relation", "")).lower()

    positive_clues = ["POSITIVE", "CALL", "STOCK_OR_CALL", "THEME_STOCK_OR_CALL"]
    negative_clues = ["NEGATIVE", "PUT", "HEDGE", "WATCH_NEGATIVE", "RISK"]

    if tone == "POSITIVE" or impact >= 1.5 or any(x in route for x in positive_clues) or "CALL" in option_side:
        return 1, "bullish event/read-through expects positive forward return"
    if tone == "NEGATIVE" or impact <= -1.5 or any(x in route for x in negative_clues) or any(x in option_side for x in ["PUT", "HEDGE"]):
        return -1, "bearish or vulnerable-event read-through expects negative forward return"
    if "vulnerable" in relation and impact < 0:
        return -1, "vulnerable peer mapped from bad news"
    return 0, "mixed or neutral event; no directional hit-rate claim"


def prepare_prices() -> pd.DataFrame:
    prices = load_price_cache()
    if prices.empty:
        return prices
    px = prices.copy()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index()
    px.columns = [clean_ticker(c) for c in px.columns]
    px = px.loc[:, ~pd.Index(px.columns).duplicated()]
    return px


def first_trade_idx(dates: pd.DatetimeIndex, anchor: pd.Timestamp | pd.NaT) -> int | None:
    if pd.isna(anchor):
        return None
    anchor_date = pd.Timestamp(anchor).normalize()
    pos = int(dates.searchsorted(anchor_date, side="left"))
    if pos >= len(dates):
        return None
    return pos


def forward_return(prices: pd.Series, dates: pd.DatetimeIndex, anchor: pd.Timestamp | pd.NaT, horizon: int) -> tuple[float, str, str, str]:
    idx = first_trade_idx(dates, anchor)
    if idx is None:
        return np.nan, "", "", "PENDING_OR_NO_ANCHOR"
    end_idx = idx + int(horizon)
    if end_idx >= len(dates):
        return np.nan, str(dates[idx].date()), "", "PENDING_PRICE_WINDOW"
    start_px = safe_float(prices.iloc[idx])
    end_px = safe_float(prices.iloc[end_idx])
    if not np.isfinite(start_px) or not np.isfinite(end_px) or start_px <= 0:
        return np.nan, str(dates[idx].date()), str(dates[end_idx].date()), "MISSING_PRICE"
    return (end_px / start_px - 1.0), str(dates[idx].date()), str(dates[end_idx].date()), "OK"


def directional_result(ret: float, direction: int) -> str:
    if direction == 0:
        return "NO_DIRECTIONAL_CLAIM"
    if not np.isfinite(ret):
        return "PENDING_OR_MISSING"
    if abs(ret) < MIN_DIRECTIONAL_MOVE:
        return "NOISY_FLAT"
    if direction > 0 and ret > 0:
        return "CONFIRMED"
    if direction < 0 and ret < 0:
        return "CONFIRMED"
    return "CONTRADICTED"


def build_returns(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    if prices.empty:
        out = panel.copy()
        out["audit_status"] = "NO_PRICE_CACHE"
        return out

    dates = pd.DatetimeIndex(prices.index)
    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
        ticker = clean_ticker(row.get("target_ticker"))
        direction, direction_reason = expected_direction(row)
        base = {
            "event_id": row.get("event_id", ""),
            "target_ticker": ticker,
            "source_news_ticker": clean_ticker(row.get("source_news_ticker", "")),
            "target_relation": row.get("target_relation", ""),
            "relation_layer": row.get("relation_layer", ""),
            "theme": row.get("theme", ""),
            "headline": row.get("headline", ""),
            "source_publish_time": row.get("source_publish_time", pd.NaT),
            "first_seen_time": row.get("first_seen_time", pd.NaT),
            "market_tone": row.get("market_tone", ""),
            "impact_score": safe_float(row.get("impact_score"), np.nan),
            "suggested_research_route": row.get("suggested_research_route", ""),
            "option_side": row.get("option_side", ""),
            "event_admissibility": row.get("event_admissibility", ""),
            "research_permission": row.get("research_permission", LOCAL_TRUTH_LABEL),
            "truth_label": row.get("truth_label", LOCAL_TRUTH_LABEL),
            "expected_direction": direction,
            "expected_direction_reason": direction_reason,
            "link": row.get("link", ""),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
            "can_support_institutional_backtest": False,
        }

        if ticker not in prices.columns:
            rec = dict(base)
            rec["audit_status"] = "MISSING_TICKER_PRICE"
            rows.append(rec)
            continue

        series = pd.to_numeric(prices[ticker], errors="coerce").ffill()
        source_statuses: list[str] = []
        model_statuses: list[str] = []
        rec = dict(base)
        for horizon in HORIZONS:
            ret, start, end, status = forward_return(series, dates, row.get("source_publish_time"), horizon)
            rec[f"source_start_date_{horizon}d"] = start
            rec[f"source_end_date_{horizon}d"] = end
            rec[f"source_return_{horizon}d"] = ret
            rec[f"source_return_{horizon}d_pct"] = ret * 100 if np.isfinite(ret) else np.nan
            rec[f"source_result_{horizon}d"] = directional_result(ret, direction)
            source_statuses.append(status)

            mret, mstart, mend, mstatus = forward_return(series, dates, row.get("first_seen_time"), horizon)
            rec[f"model_seen_start_date_{horizon}d"] = mstart
            rec[f"model_seen_end_date_{horizon}d"] = mend
            rec[f"model_seen_return_{horizon}d"] = mret
            rec[f"model_seen_return_{horizon}d_pct"] = mret * 100 if np.isfinite(mret) else np.nan
            rec[f"model_seen_result_{horizon}d"] = directional_result(mret, direction)
            model_statuses.append(mstatus)

        if all(s == "OK" for s in source_statuses):
            rec["source_audit_status"] = "SOURCE_WINDOW_COMPLETE"
        elif any(s == "OK" for s in source_statuses):
            rec["source_audit_status"] = "SOURCE_PARTIAL_WINDOW"
        else:
            rec["source_audit_status"] = source_statuses[0] if source_statuses else "NO_SOURCE_WINDOW"

        if all(s == "OK" for s in model_statuses):
            rec["model_seen_audit_status"] = "MODEL_SEEN_WINDOW_COMPLETE"
        elif any(s == "OK" for s in model_statuses):
            rec["model_seen_audit_status"] = "MODEL_SEEN_PARTIAL_WINDOW"
        else:
            rec["model_seen_audit_status"] = model_statuses[0] if model_statuses else "NO_MODEL_WINDOW"

        rec["primary_audit_status"] = rec["model_seen_audit_status"]
        if rec["model_seen_audit_status"] == "PENDING_OR_NO_ANCHOR":
            rec["primary_audit_status"] = "MODEL_FIRST_SEEN_AFTER_PRICE_CACHE_OR_MISSING"
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = [c for c in ["target_ticker", "source_publish_time", "headline"] if c in out.columns]
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    work = df.copy()
    for keys, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: key for col, key in zip(group_cols, keys)}
        rec["rows"] = int(len(sub))
        rec["directional_rows"] = int((sub.get("expected_direction", 0) != 0).sum())
        rec["source_1d_coverage_pct"] = float(sub["source_return_1d"].notna().mean() * 100) if "source_return_1d" in sub.columns else 0.0
        rec["source_5d_coverage_pct"] = float(sub["source_return_5d"].notna().mean() * 100) if "source_return_5d" in sub.columns else 0.0
        rec["model_seen_1d_coverage_pct"] = float(sub["model_seen_return_1d"].notna().mean() * 100) if "model_seen_return_1d" in sub.columns else 0.0
        rec["model_seen_5d_coverage_pct"] = float(sub["model_seen_return_5d"].notna().mean() * 100) if "model_seen_return_5d" in sub.columns else 0.0
        for horizon in HORIZONS:
            col = f"source_return_{horizon}d"
            res_col = f"source_result_{horizon}d"
            if col in sub.columns:
                rec[f"avg_source_return_{horizon}d_pct"] = float(pd.to_numeric(sub[col], errors="coerce").mean() * 100)
            if res_col in sub.columns:
                directional = sub[sub["expected_direction"] != 0]
                observed = directional[directional[res_col].isin(["CONFIRMED", "CONTRADICTED", "NOISY_FLAT"])]
                if not observed.empty:
                    rec[f"source_hit_rate_{horizon}d"] = float((observed[res_col] == "CONFIRMED").mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["rows"], ascending=False).reset_index(drop=True)


def build_summary(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for label, cols in [
        ("overall", []),
        ("by_tone", ["market_tone"]),
        ("by_option_side", ["option_side"]),
        ("by_relation", ["relation_layer"]),
        ("by_route", ["suggested_research_route"]),
        ("by_admissibility", ["event_admissibility"]),
    ]:
        if cols:
            use_cols = [c for c in cols if c in returns.columns]
            if not use_cols:
                continue
            part = summarize_group(returns, use_cols)
        else:
            temp = returns.copy()
            temp["all_events"] = "ALL_LOCAL_AUDIT_EVENTS"
            part = summarize_group(temp, ["all_events"])
        if not part.empty:
            part.insert(0, "summary_scope", label)
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for c in out.columns:
        if c.endswith("_pct") or c.startswith("avg_source_return"):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
        if c.startswith("source_hit_rate"):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    return out


def build_failures(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for horizon in [1, 3, 5, 10]:
        result_col = f"source_result_{horizon}d"
        ret_col = f"source_return_{horizon}d_pct"
        if result_col not in returns.columns:
            continue
        bad = returns[returns[result_col].isin(["CONTRADICTED", "PENDING_OR_MISSING"])].copy()
        if bad.empty:
            continue
        bad["failure_horizon"] = f"{horizon}d"
        bad["failure_type"] = np.where(
            bad[result_col] == "CONTRADICTED",
            "DIRECTION_CONTRADICTED",
            "PRICE_WINDOW_PENDING_OR_MISSING",
        )
        bad["observed_return_pct"] = pd.to_numeric(bad.get(ret_col, np.nan), errors="coerce")
        cols = [
            "failure_type", "failure_horizon", "target_ticker", "market_tone",
            "option_side", "expected_direction_reason", "observed_return_pct",
            "source_audit_status", "model_seen_audit_status", "event_admissibility",
            "headline", "link",
        ]
        rows.append(bad[[c for c in cols if c in bad.columns]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    priority = {"DIRECTION_CONTRADICTED": 0, "PRICE_WINDOW_PENDING_OR_MISSING": 1}
    out["_rank"] = out["failure_type"].map(priority).fillna(9)
    out = out.sort_values(["_rank", "failure_horizon", "target_ticker"]).drop(columns=["_rank"]).reset_index(drop=True)
    return out


def build_state(returns: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame, prices: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(returns))
    source_1d = float(returns["source_return_1d"].notna().mean()) if rows and "source_return_1d" in returns.columns else 0.0
    source_5d = float(returns["source_return_5d"].notna().mean()) if rows and "source_return_5d" in returns.columns else 0.0
    model_1d = float(returns["model_seen_return_1d"].notna().mean()) if rows and "model_seen_return_1d" in returns.columns else 0.0
    directional = returns[returns.get("expected_direction", pd.Series(dtype=int)) != 0] if rows else pd.DataFrame()
    observed_1d = directional[directional.get("source_result_1d", pd.Series(dtype=str)).isin(["CONFIRMED", "CONTRADICTED", "NOISY_FLAT"])] if not directional.empty else pd.DataFrame()
    observed_5d = directional[directional.get("source_result_5d", pd.Series(dtype=str)).isin(["CONFIRMED", "CONTRADICTED", "NOISY_FLAT"])] if not directional.empty else pd.DataFrame()
    hit_1d = float((observed_1d.get("source_result_1d", pd.Series(dtype=str)) == "CONFIRMED").mean()) if not observed_1d.empty else np.nan
    hit_5d = float((observed_5d.get("source_result_5d", pd.Series(dtype=str)) == "CONFIRMED").mean()) if not observed_5d.empty else np.nan

    if rows == 0:
        status = "NO_EVENT_AUDIT_ROWS"
    elif model_1d == 0:
        status = "SOURCE_EVENT_REACTION_ONLY_MODEL_FORWARD_PENDING"
    elif source_5d < 0.25:
        status = "EVENT_AUDIT_EARLY_PRICE_WINDOW"
    else:
        status = "LOCAL_EVENT_AUDIT_USABLE_NOT_INSTITUTIONAL"

    return {
        "generated": today_str(),
        "overall_status": status,
        "event_rows": rows,
        "directional_event_rows": int(len(directional)),
        "observed_directional_1d_rows": int(len(observed_1d)),
        "observed_directional_5d_rows": int(len(observed_5d)),
        "summary_rows": int(len(summary)),
        "failure_rows": int(len(failures)),
        "source_1d_coverage_pct": round(source_1d * 100, 2),
        "source_5d_coverage_pct": round(source_5d * 100, 2),
        "model_seen_1d_coverage_pct": round(model_1d * 100, 2),
        "source_1d_hit_rate": None if not np.isfinite(hit_1d) else round(hit_1d, 3),
        "source_5d_hit_rate": None if not np.isfinite(hit_5d) else round(hit_5d, 3),
        "price_cache_start": str(prices.index.min().date()) if not prices.empty else "",
        "price_cache_end": str(prices.index.max().date()) if not prices.empty else "",
        "truth_label": LOCAL_TRUTH_LABEL,
        "can_support_institutional_backtest": False,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def write_outputs(returns: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame, state: dict[str, Any]) -> None:
    returns.to_csv(OUT_RETURNS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    failures.to_csv(OUT_FAILURES, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## State",
        f"- Overall status: **{state['overall_status']}**",
        f"- Event rows: **{state['event_rows']}**",
        f"- Directional rows: **{state['directional_event_rows']}**",
        f"- Source 1d coverage: **{state['source_1d_coverage_pct']}%**",
        f"- Source 5d coverage: **{state['source_5d_coverage_pct']}%**",
        f"- Model-seen 1d coverage: **{state['model_seen_1d_coverage_pct']}%**",
        f"- Source 1d hit rate: **{state['source_1d_hit_rate']}**",
        f"- Source 5d hit rate: **{state['source_5d_hit_rate']}**",
        f"- Price cache: **{state['price_cache_start']} -> {state['price_cache_end']}**",
        "",
        "## Interpretation",
        "This is a local audit of whether event-derived research ideas were followed by the expected price reaction. It is not an institutional point-in-time backtest because the price cache and event tape are local/proxy sources.",
        "",
        "## Summary",
        df_to_markdown(summary.head(40), max_rows=40),
        "",
        "## Failure / Pending Modes",
        df_to_markdown(failures.head(60), max_rows=60),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 165 - Event Signal Local Audit", sections)


def main() -> None:
    panel = load_event_panel()
    prices = prepare_prices()
    returns = build_returns(panel, prices)
    summary = build_summary(returns)
    failures = build_failures(returns)
    state = build_state(returns, summary, failures, prices)
    write_outputs(returns, summary, failures, state)
    print("Canyon v9 Step165 event signal local audit complete.")
    print(f"Overall: {state.get('overall_status')} | rows: {state.get('event_rows')} | source 1d coverage: {state.get('source_1d_coverage_pct')}%")
    print("Reminder: local audit only; no broker connection; no live orders.")


if __name__ == "__main__":
    main()
