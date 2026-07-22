#!/usr/bin/env python3
"""
Canyon v9 Step 166 - Event Signal Reliability Calibrator.

Research-only. No broker connection. No live orders.

Step165 audits whether event-derived signals were followed by the expected
price reaction. Step166 turns that audit into local reliability calibration:
which tones, option routes, causal links, and tickers have enough observed
evidence to be trusted as research context, and which should be faded or
repaired.

This is not institutional-grade validation. It only calibrates local/proxy
event signals and explicitly waits for model-first-seen windows before making
stronger live validation claims.

Outputs:
  event_signal_reliability_by_bucket.csv
  event_signal_reliability_by_ticker.csv
  event_signal_reliability_adjusted_panel.csv
  event_signal_reliability_watchlist.csv
  event_signal_reliability_state.json
  event_signal_reliability_report.md
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
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


AUDIT_RETURNS = ROOT / "event_signal_local_audit_returns.csv"
AUDIT_STATE = ROOT / "event_signal_audit_state.json"
SAFE_PANEL = ROOT / "pit_safe_event_signal_panel.csv"

OUT_BUCKET = ROOT / "event_signal_reliability_by_bucket.csv"
OUT_TICKER = ROOT / "event_signal_reliability_by_ticker.csv"
OUT_PANEL = ROOT / "event_signal_reliability_adjusted_panel.csv"
OUT_WATCHLIST = ROOT / "event_signal_reliability_watchlist.csv"
OUT_STATE = ROOT / "event_signal_reliability_state.json"
OUT_REPORT = ROOT / "event_signal_reliability_report.md"

LOCAL_TRUTH_LABEL = "LOCAL_AUDIT_ONLY_NOT_INSTITUTIONAL"
HORIZONS = [1, 3, 5]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clean_label(value: Any, default: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    return text.upper()


def load_audit_returns() -> pd.DataFrame:
    df = read_csv_safe(AUDIT_RETURNS)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["target_ticker"] = out.get("target_ticker", pd.Series(dtype=str)).map(clean_ticker)
    for col in [
        "market_tone", "option_side", "relation_layer", "target_relation",
        "suggested_research_route", "theme", "event_admissibility",
    ]:
        out[col] = out.get(col, pd.Series(dtype=str)).map(clean_label)
    out["composite_bucket"] = (
        out["market_tone"] + " | " +
        out["option_side"] + " | " +
        out["relation_layer"] + " | " +
        out["suggested_research_route"]
    )
    out["expected_direction"] = pd.to_numeric(out.get("expected_direction", 0), errors="coerce").fillna(0).astype(int)
    for horizon in [1, 3, 5, 10]:
        ret_col = f"source_return_{horizon}d"
        if ret_col in out.columns:
            out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")
        result_col = f"source_result_{horizon}d"
        if result_col in out.columns:
            out[result_col] = out[result_col].map(clean_label)
    return out.reset_index(drop=True)


def observed_mask(df: pd.DataFrame, horizon: int) -> pd.Series:
    result_col = f"source_result_{horizon}d"
    if result_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[result_col].isin(["CONFIRMED", "CONTRADICTED", "NOISY_FLAT"])


def bucket_stats(df: pd.DataFrame, scope: str, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    if not group_cols:
        work["all_events"] = "ALL_LOCAL_EVENT_SIGNALS"
        group_cols = ["all_events"]

    rows: list[dict[str, Any]] = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: key for col, key in zip(group_cols, keys)}
        rec["calibration_scope"] = scope
        rec["rows"] = int(len(sub))
        directional = sub[sub["expected_direction"] != 0].copy()
        rec["directional_rows"] = int(len(directional))
        observed_counts: list[int] = []
        effective_hit_rates: list[float] = []
        payoff_scores: list[float] = []
        weights: list[float] = []
        horizon_weight = {1: 0.50, 3: 0.30, 5: 0.20}
        for horizon in HORIZONS:
            ret_col = f"source_return_{horizon}d"
            result_col = f"source_result_{horizon}d"
            if result_col not in sub.columns:
                rec[f"observed_{horizon}d"] = 0
                continue
            obs = directional[directional[result_col].isin(["CONFIRMED", "CONTRADICTED", "NOISY_FLAT"])]
            rec[f"observed_{horizon}d"] = int(len(obs))
            rec[f"confirmed_{horizon}d"] = int((obs[result_col] == "CONFIRMED").sum()) if not obs.empty else 0
            rec[f"contradicted_{horizon}d"] = int((obs[result_col] == "CONTRADICTED").sum()) if not obs.empty else 0
            rec[f"noisy_flat_{horizon}d"] = int((obs[result_col] == "NOISY_FLAT").sum()) if not obs.empty else 0
            if not obs.empty:
                strict_hit = float((obs[result_col] == "CONFIRMED").mean())
                effective_hit = float(((obs[result_col] == "CONFIRMED").astype(float) + 0.5 * (obs[result_col] == "NOISY_FLAT").astype(float)).mean())
                rec[f"strict_hit_rate_{horizon}d"] = strict_hit
                rec[f"effective_hit_rate_{horizon}d"] = effective_hit
                rec[f"hit_rate_{horizon}d"] = effective_hit
                if ret_col in obs.columns:
                    rets = pd.to_numeric(obs[ret_col], errors="coerce")
                    rec[f"avg_return_{horizon}d_pct"] = float(rets.mean() * 100)
                    rec[f"median_return_{horizon}d_pct"] = float(rets.median() * 100)
                    signed = rets * obs["expected_direction"].astype(float)
                    rec[f"avg_directional_payoff_{horizon}d_pct"] = float(signed.mean() * 100)
                    payoff_scores.append(float(np.clip(signed.mean() * 100, -8, 8)))
                observed_counts.append(int(len(obs)))
                effective_hit_rates.append(effective_hit)
                weights.append(horizon_weight.get(horizon, 0.1) * np.sqrt(max(len(obs), 1)))
            else:
                rec[f"hit_rate_{horizon}d"] = np.nan
                rec[f"strict_hit_rate_{horizon}d"] = np.nan
                rec[f"effective_hit_rate_{horizon}d"] = np.nan
                rec[f"avg_return_{horizon}d_pct"] = np.nan
                rec[f"median_return_{horizon}d_pct"] = np.nan
                rec[f"avg_directional_payoff_{horizon}d_pct"] = np.nan

        total_observed = int(max(observed_counts) if observed_counts else 0)
        rec["max_observed_directional_rows"] = total_observed
        if effective_hit_rates and sum(weights) > 0:
            weighted_hit = float(np.average(effective_hit_rates, weights=weights))
        else:
            weighted_hit = np.nan
        rec["weighted_hit_rate"] = weighted_hit
        sample_confidence = float(min(1.0, np.sqrt(max(total_observed, 0) / 30.0))) if total_observed > 0 else 0.0
        rec["sample_confidence"] = sample_confidence
        payoff_boost = float(np.nanmean(payoff_scores)) if payoff_scores else 0.0

        if rec["directional_rows"] == 0:
            raw_score = 50.0
            status = "CONTEXT_ONLY_NO_DIRECTIONAL_CLAIM"
            reason = "bucket has no directional event claim"
        elif total_observed == 0:
            raw_score = 45.0
            status = "PENDING_PRICE_WINDOW"
            reason = "directional rows exist but source price windows are not observed yet"
        else:
            raw_score = 50.0 + (weighted_hit - 0.50) * 70.0 + payoff_boost * 1.5
            score = 50.0 * (1.0 - sample_confidence) + raw_score * sample_confidence
            score = float(np.clip(score, 5.0, 95.0))
            rec["reliability_score"] = round(score, 1)
            if total_observed < 8:
                status = "LOW_SAMPLE_REVIEW"
                reason = "observed sample is too small for strong calibration"
            elif score >= 67:
                status = "RELIABLE_LOCAL_CONTEXT"
                reason = "local observed hit rate is supportive, but still research-only"
            elif score >= 55:
                status = "WATCH_LOCAL_CONTEXT"
                reason = "local signal is usable as context, not as standalone action"
            elif score >= 45:
                status = "UNPROVEN_LOCAL_CONTEXT"
                reason = "local signal is mixed; require price/volume/risk confirmation"
            else:
                status = "FADE_OR_REPAIR_SIGNAL"
                reason = "local observed windows contradict this event signal type"
            rec["reliability_status"] = status
            rec["reliability_reason"] = reason
            rows.append(rec)
            continue

        rec["reliability_score"] = round(raw_score, 1)
        rec["reliability_status"] = status
        rec["reliability_reason"] = reason
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for c in out.columns:
        if c.endswith("_pct") or c in {"weighted_hit_rate", "sample_confidence"}:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    out["reliability_score"] = pd.to_numeric(out["reliability_score"], errors="coerce").round(1)
    return out.sort_values(["reliability_score", "max_observed_directional_rows", "rows"], ascending=[True, False, False]).reset_index(drop=True)


def build_bucket_reliability(audit: pd.DataFrame) -> pd.DataFrame:
    scopes = [
        ("overall", []),
        ("by_tone", ["market_tone"]),
        ("by_option_side", ["option_side"]),
        ("by_relation_layer", ["relation_layer"]),
        ("by_route", ["suggested_research_route"]),
        ("by_target_relation", ["target_relation"]),
        ("by_theme", ["theme"]),
        ("by_composite", ["composite_bucket"]),
    ]
    frames = [bucket_stats(audit, label, cols) for label, cols in scopes]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_ticker_reliability(audit: pd.DataFrame) -> pd.DataFrame:
    out = bucket_stats(audit, "by_ticker", ["target_ticker"])
    if out.empty:
        return out
    cols = [
        "target_ticker", "rows", "directional_rows", "max_observed_directional_rows",
        "weighted_hit_rate", "sample_confidence", "reliability_score",
        "reliability_status", "reliability_reason", "hit_rate_1d",
        "avg_directional_payoff_1d_pct", "hit_rate_3d",
        "avg_directional_payoff_3d_pct",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values(["reliability_score", "max_observed_directional_rows"], ascending=[True, False]).reset_index(drop=True)


def choose_bucket_lookup(bucket: pd.DataFrame, scope: str, key_col: str) -> dict[str, dict[str, Any]]:
    if bucket.empty or key_col not in bucket.columns:
        return {}
    sub = bucket[bucket["calibration_scope"] == scope].copy()
    if sub.empty:
        return {}
    return {
        str(row[key_col]): row.to_dict()
        for _, row in sub.iterrows()
    }


def reliability_multiplier(score: float, status: str) -> float:
    status = str(status or "")
    if status == "FADE_OR_REPAIR_SIGNAL":
        return 0.35
    if status == "UNPROVEN_LOCAL_CONTEXT":
        return 0.65
    if status == "LOW_SAMPLE_REVIEW":
        return 0.75
    if status == "WATCH_LOCAL_CONTEXT":
        return 0.95
    if status == "RELIABLE_LOCAL_CONTEXT":
        return min(1.25, 0.90 + max(0.0, score - 60.0) / 100.0)
    if status == "CONTEXT_ONLY_NO_DIRECTIONAL_CLAIM":
        return 0.50
    return 0.60


def action_from_reliability(row: pd.Series) -> str:
    status = str(row.get("calibrated_reliability_status", ""))
    option_side = str(row.get("option_side", "")).upper()
    direction = int(safe_float(row.get("expected_direction", 0), 0))
    if direction == 0:
        return "CONTEXT_ONLY_NO_DIRECTIONAL_ACTION"
    if status == "FADE_OR_REPAIR_SIGNAL":
        return "DO_NOT_UPGRADE_FROM_THIS_EVENT_REQUIRE_CONFIRMATION"
    if status == "UNPROVEN_LOCAL_CONTEXT":
        return "WATCH_ONLY_REQUIRE_PRICE_VOLUME_CONFIRMATION"
    if status == "LOW_SAMPLE_REVIEW":
        return "RESEARCH_REVIEW_SMALL_SAMPLE"
    if status == "RELIABLE_LOCAL_CONTEXT" and "CALL" in option_side:
        return "CALL_RESEARCH_CONTEXT_ONLY_AFTER_RISK_GATES"
    if status == "RELIABLE_LOCAL_CONTEXT" and any(x in option_side for x in ["PUT", "HEDGE"]):
        return "PUT_OR_HEDGE_RESEARCH_CONTEXT_ONLY_AFTER_RISK_GATES"
    if status in {"RELIABLE_LOCAL_CONTEXT", "WATCH_LOCAL_CONTEXT"}:
        return "STOCK_OR_ETF_RESEARCH_CONTEXT_ONLY_AFTER_RISK_GATES"
    return "CURRENT_RESEARCH_ONLY"


def build_adjusted_panel(audit: pd.DataFrame, bucket: pd.DataFrame, ticker_rel: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    composite_lookup = choose_bucket_lookup(bucket, "by_composite", "composite_bucket")
    relation_lookup = choose_bucket_lookup(bucket, "by_relation_layer", "relation_layer")
    route_lookup = choose_bucket_lookup(bucket, "by_route", "suggested_research_route")
    tone_lookup = choose_bucket_lookup(bucket, "by_tone", "market_tone")
    ticker_lookup = {
        str(row["target_ticker"]): row.to_dict()
        for _, row in ticker_rel.iterrows()
    } if not ticker_rel.empty and "target_ticker" in ticker_rel.columns else {}

    rows: list[dict[str, Any]] = []
    for _, row in audit.iterrows():
        candidates = [
            ("COMPOSITE", composite_lookup.get(str(row.get("composite_bucket", "")), {})),
            ("RELATION_LAYER", relation_lookup.get(str(row.get("relation_layer", "")), {})),
            ("ROUTE", route_lookup.get(str(row.get("suggested_research_route", "")), {})),
            ("TONE", tone_lookup.get(str(row.get("market_tone", "")), {})),
        ]
        chosen_name = "NONE"
        chosen = {}
        for name, candidate in candidates:
            observed = int(safe_float(candidate.get("max_observed_directional_rows", 0), 0))
            if candidate and observed >= 5:
                chosen_name = name
                chosen = candidate
                break
        if not chosen:
            chosen_name = "FALLBACK_TONE_OR_PENDING"
            chosen = tone_lookup.get(str(row.get("market_tone", "")), {})

        score = safe_float(chosen.get("reliability_score"), 45.0)
        status = str(chosen.get("reliability_status", "PENDING_PRICE_WINDOW"))
        multiplier = reliability_multiplier(score, status)
        impact = safe_float(row.get("impact_score"), 0.0)
        ticker_info = ticker_lookup.get(str(row.get("target_ticker", "")), {})
        ticker_score = safe_float(ticker_info.get("reliability_score"), np.nan)
        calibrated = impact * multiplier
        if np.isfinite(ticker_score) and ticker_score < 45:
            calibrated *= 0.75

        rec = {
            "target_ticker": row.get("target_ticker", ""),
            "source_news_ticker": row.get("source_news_ticker", ""),
            "headline": row.get("headline", ""),
            "link": row.get("link", ""),
            "market_tone": row.get("market_tone", ""),
            "option_side": row.get("option_side", ""),
            "relation_layer": row.get("relation_layer", ""),
            "target_relation": row.get("target_relation", ""),
            "theme": row.get("theme", ""),
            "suggested_research_route": row.get("suggested_research_route", ""),
            "impact_score": round(impact, 3),
            "calibrated_event_score": round(calibrated, 3),
            "reliability_multiplier": round(multiplier, 3),
            "calibrated_reliability_score": round(score, 1),
            "calibrated_reliability_status": status,
            "calibration_source": chosen_name,
            "calibration_observed_rows": int(safe_float(chosen.get("max_observed_directional_rows", 0), 0)),
            "calibration_hit_rate": safe_float(chosen.get("weighted_hit_rate"), np.nan),
            "ticker_reliability_score": ticker_score,
            "ticker_reliability_status": ticker_info.get("reliability_status", "NO_TICKER_CALIBRATION"),
            "expected_direction": int(safe_float(row.get("expected_direction", 0), 0)),
            "source_result_1d": row.get("source_result_1d", ""),
            "source_return_1d_pct": safe_float(row.get("source_return_1d_pct"), np.nan),
            "model_seen_result_1d": row.get("model_seen_result_1d", ""),
            "model_seen_audit_status": row.get("model_seen_audit_status", ""),
            "event_admissibility": row.get("event_admissibility", ""),
            "research_permission": row.get("research_permission", LOCAL_TRUTH_LABEL),
            "truth_label": LOCAL_TRUTH_LABEL,
            "can_support_institutional_backtest": False,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        rec["calibrated_research_action"] = action_from_reliability(pd.Series(rec))
        if status == "FADE_OR_REPAIR_SIGNAL":
            rec["calibration_note"] = "Local observed windows contradict this signal family; do not let headline upgrade the ticker."
        elif status == "RELIABLE_LOCAL_CONTEXT":
            rec["calibration_note"] = "Locally supportive signal family; still must pass L1/L6/L7/L8/L9 gates."
        elif status == "PENDING_PRICE_WINDOW":
            rec["calibration_note"] = "Price window pending; use only as current research context."
        else:
            rec["calibration_note"] = "Use as context; require confirmation from price, volume, risk, and event gates."
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = ["calibrated_reliability_score", "calibration_observed_rows", "target_ticker"]
    return out.sort_values(sort_cols, ascending=[True, False, True]).reset_index(drop=True)


def build_watchlist(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    watch = panel.copy()
    watch["abs_calibrated_event_score"] = pd.to_numeric(watch["calibrated_event_score"], errors="coerce").abs()
    priority_rank = {
        "DO_NOT_UPGRADE_FROM_THIS_EVENT_REQUIRE_CONFIRMATION": 0,
        "WATCH_ONLY_REQUIRE_PRICE_VOLUME_CONFIRMATION": 1,
        "CALL_RESEARCH_CONTEXT_ONLY_AFTER_RISK_GATES": 2,
        "PUT_OR_HEDGE_RESEARCH_CONTEXT_ONLY_AFTER_RISK_GATES": 2,
        "STOCK_OR_ETF_RESEARCH_CONTEXT_ONLY_AFTER_RISK_GATES": 3,
        "RESEARCH_REVIEW_SMALL_SAMPLE": 4,
        "CONTEXT_ONLY_NO_DIRECTIONAL_ACTION": 5,
    }
    watch["_action_rank"] = watch["calibrated_research_action"].map(priority_rank).fillna(6)
    cols = [
        "calibrated_research_action", "target_ticker", "market_tone",
        "option_side", "relation_layer", "calibrated_event_score",
        "calibrated_reliability_score", "calibrated_reliability_status",
        "calibration_source", "calibration_observed_rows", "calibration_hit_rate",
        "source_result_1d", "source_return_1d_pct", "model_seen_audit_status",
        "headline", "calibration_note", "link",
    ]
    cols = [c for c in cols if c in watch.columns]
    return (
        watch.sort_values(["_action_rank", "abs_calibrated_event_score", "calibration_observed_rows"], ascending=[True, False, False])
             .drop(columns=["_action_rank", "abs_calibrated_event_score"], errors="ignore")[cols]
             .head(250)
             .reset_index(drop=True)
    )


def build_state(bucket: pd.DataFrame, ticker: pd.DataFrame, panel: pd.DataFrame, audit_state: dict[str, Any]) -> dict[str, Any]:
    reliable = int((bucket.get("reliability_status", pd.Series(dtype=str)) == "RELIABLE_LOCAL_CONTEXT").sum()) if not bucket.empty else 0
    repair = int((bucket.get("reliability_status", pd.Series(dtype=str)) == "FADE_OR_REPAIR_SIGNAL").sum()) if not bucket.empty else 0
    pending = int((bucket.get("reliability_status", pd.Series(dtype=str)) == "PENDING_PRICE_WINDOW").sum()) if not bucket.empty else 0
    low_sample = int((bucket.get("reliability_status", pd.Series(dtype=str)) == "LOW_SAMPLE_REVIEW").sum()) if not bucket.empty else 0
    model_seen_cov = float(audit_state.get("model_seen_1d_coverage_pct", 0) or 0)
    if panel.empty:
        status = "NO_EVENT_RELIABILITY_PANEL"
    elif model_seen_cov == 0:
        status = "SOURCE_REACTION_CALIBRATION_ONLY_MODEL_FORWARD_PENDING"
    elif repair > reliable:
        status = "EVENT_SIGNAL_REPAIR_REQUIRED"
    else:
        status = "LOCAL_EVENT_RELIABILITY_USABLE_NOT_INSTITUTIONAL"
    return {
        "generated": today_str(),
        "overall_status": status,
        "bucket_rows": int(len(bucket)),
        "ticker_rows": int(len(ticker)),
        "adjusted_panel_rows": int(len(panel)),
        "reliable_bucket_count": reliable,
        "fade_or_repair_bucket_count": repair,
        "pending_bucket_count": pending,
        "low_sample_bucket_count": low_sample,
        "model_seen_1d_coverage_pct": model_seen_cov,
        "source_audit_event_rows": int(audit_state.get("event_rows", 0) or 0),
        "truth_label": LOCAL_TRUTH_LABEL,
        "can_support_institutional_backtest": False,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def write_outputs(bucket: pd.DataFrame, ticker: pd.DataFrame, panel: pd.DataFrame, watch: pd.DataFrame, state: dict[str, Any]) -> None:
    bucket.to_csv(OUT_BUCKET, index=False)
    ticker.to_csv(OUT_TICKER, index=False)
    panel.to_csv(OUT_PANEL, index=False)
    watch.to_csv(OUT_WATCHLIST, index=False)
    write_json(OUT_STATE, state)

    weak = bucket[bucket.get("reliability_status", pd.Series(dtype=str)).isin(["FADE_OR_REPAIR_SIGNAL", "UNPROVEN_LOCAL_CONTEXT", "LOW_SAMPLE_REVIEW"])] if not bucket.empty else pd.DataFrame()
    strong = bucket[bucket.get("reliability_status", pd.Series(dtype=str)) == "RELIABLE_LOCAL_CONTEXT"] if not bucket.empty else pd.DataFrame()
    sections = [
        "## State",
        f"- Overall status: **{state['overall_status']}**",
        f"- Buckets: **{state['bucket_rows']}**",
        f"- Adjusted panel rows: **{state['adjusted_panel_rows']}**",
        f"- Reliable local buckets: **{state['reliable_bucket_count']}**",
        f"- Fade/repair buckets: **{state['fade_or_repair_bucket_count']}**",
        f"- Low-sample buckets: **{state['low_sample_bucket_count']}**",
        f"- Model-seen 1d coverage: **{state['model_seen_1d_coverage_pct']}%**",
        "",
        "## Strong Local Buckets",
        df_to_markdown(strong.head(30), max_rows=30),
        "",
        "## Weak / Repair Buckets",
        df_to_markdown(weak.head(40), max_rows=40),
        "",
        "## Calibrated Watchlist",
        df_to_markdown(watch.head(60), max_rows=60),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 166 - Event Signal Reliability Calibrator", sections)


def main() -> None:
    audit = load_audit_returns()
    audit_state = read_json_safe(AUDIT_STATE, {})
    bucket = build_bucket_reliability(audit)
    ticker = build_ticker_reliability(audit)
    panel = build_adjusted_panel(audit, bucket, ticker)
    watch = build_watchlist(panel)
    state = build_state(bucket, ticker, panel, audit_state)
    write_outputs(bucket, ticker, panel, watch, state)
    print("Canyon v9 Step166 event signal reliability calibration complete.")
    print(f"Overall: {state.get('overall_status')} | buckets: {state.get('bucket_rows')} | panel rows: {state.get('adjusted_panel_rows')}")
    print("Reminder: local calibration only; no broker connection; no live orders.")


if __name__ == "__main__":
    main()
