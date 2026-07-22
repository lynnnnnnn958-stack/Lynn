#!/usr/bin/env python3
"""
Canyon v9 Step 163 - Event Time Truth Ledger.

Research-only. No broker connection. No live orders.

This step normalizes news, earnings, SEC, insider, and derived event rows into
a single timing ledger. It preserves first-seen time by stable event_id so the
research system can later ask: was this signal actually observable at the time?

Outputs:
  event_time_truth_ledger.csv
  event_first_seen_registry.csv
  event_time_quality_audit.csv
  event_time_truth_state.json
  event_time_truth_report.md
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
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


OUT_LEDGER = ROOT / "event_time_truth_ledger.csv"
OUT_REGISTRY = ROOT / "event_first_seen_registry.csv"
OUT_AUDIT = ROOT / "event_time_quality_audit.csv"
OUT_STATE = ROOT / "event_time_truth_state.json"
OUT_REPORT = ROOT / "event_time_truth_report.md"

MODEL_READ_TIME = datetime.now().replace(microsecond=0).isoformat()
LOCAL_QUALITY = "LOCAL_EVENT_TIME_LEDGER_NOT_VENDOR_PIT"
LOCAL_VENDOR = "LOCAL_YFINANCE_PUBLIC_CACHE_OR_MODEL_PROXY"


def file_seen_time(path: Path) -> str:
    if not path.exists():
        return MODEL_READ_TIME
    return datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat()


def stable_hash(*parts: Any) -> str:
    text = "|".join(str(p or "").strip() for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_time(value: Any, numeric_unit: str | None = None) -> tuple[str, str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "", "MISSING"
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "nat", "0", "1970-01-01"}:
        return "", "MISSING"
    try:
        if numeric_unit and raw.replace(".", "", 1).isdigit():
            ts = pd.to_datetime(float(raw), unit=numeric_unit, errors="coerce", utc=True)
        elif raw.replace(".", "", 1).isdigit() and len(raw.split(".")[0]) >= 9:
            ts = pd.to_datetime(float(raw), unit="s", errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(raw, errors="coerce", utc=False)
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        return "", "UNPARSEABLE"
    if getattr(ts, "tzinfo", None) is not None:
        out = ts.tz_convert(None).replace(microsecond=0).isoformat()
    else:
        out = ts.replace(microsecond=0).isoformat()
    precision = "INTRADAY" if ("T" in out and not out.endswith("00:00:00")) else "DATE_ONLY"
    return out, precision


def load_registry() -> dict[str, str]:
    old = read_csv_safe(OUT_REGISTRY)
    if old.empty or not {"event_id", "first_seen_time"}.issubset(old.columns):
        return {}
    return dict(zip(old["event_id"].astype(str), old["first_seen_time"].astype(str)))


def first_seen_for(event_id: str, fallback: str, registry: dict[str, str]) -> str:
    old = registry.get(event_id, "")
    if old and old.lower() not in {"nan", "none", "nat"}:
        return old
    return fallback or MODEL_READ_TIME


def quality_for(row: dict[str, Any]) -> tuple[int, str, str]:
    has_publish = bool(row.get("source_publish_time"))
    has_first = bool(row.get("first_seen_time"))
    has_model = bool(row.get("model_read_time"))
    has_url = bool(row.get("source_url"))
    has_vendor = bool(row.get("vendor_or_source_id"))
    precision = str(row.get("timestamp_precision", "MISSING"))
    score = 0
    score += 25 if has_publish else 0
    score += 20 if has_first else 0
    score += 15 if has_model else 0
    score += 15 if has_url else 0
    score += 15 if has_vendor else 0
    score += 10 if precision == "INTRADAY" else (4 if precision == "DATE_ONLY" else 0)
    if score >= 85:
        status = "AUDITABLE_LOCAL_EVENT_TIME"
        risk = "LOW_LOOKAHEAD_RISK_LOCAL"
    elif score >= 60:
        status = "REVIEW_EVENT_TIME"
        risk = "MEDIUM_LOOKAHEAD_RISK"
    elif score >= 35:
        status = "WEAK_EVENT_TIME"
        risk = "HIGH_LOOKAHEAD_RISK"
    else:
        status = "BLOCKED_EVENT_TIME"
        risk = "BLOCKED_LOOKAHEAD_RISK"
    return int(score), status, risk


def make_row(
    *,
    event_type: str,
    source_file: str,
    ticker: str = "",
    related_ticker: str = "",
    headline: str = "",
    event_date_raw: Any = "",
    publish_raw: Any = "",
    first_seen_fallback: str = "",
    publisher: str = "",
    source_url: str = "",
    vendor_or_source_id: str = "",
    market_tone: str = "",
    impact_score: Any = np.nan,
    source_detail: str = "",
    registry: dict[str, str],
) -> dict[str, Any]:
    source_publish_time, precision = parse_time(publish_raw)
    event_date, event_precision = parse_time(event_date_raw)
    if not source_publish_time and event_date:
        source_publish_time = event_date
        precision = event_precision
    event_id = stable_hash(event_type, source_file, ticker, related_ticker, headline, source_url, vendor_or_source_id, source_publish_time, event_date)
    row: dict[str, Any] = {
        "event_id": event_id,
        "event_type": event_type,
        "source_file": source_file,
        "ticker": clean_ticker(ticker),
        "related_ticker": clean_ticker(related_ticker),
        "headline": str(headline or "")[:500],
        "event_date": event_date[:10] if event_date else (source_publish_time[:10] if source_publish_time else ""),
        "source_publish_time": source_publish_time,
        "timestamp_precision": precision if source_publish_time else "MISSING",
        "first_seen_time": first_seen_for(event_id, first_seen_fallback or MODEL_READ_TIME, registry),
        "model_read_time": MODEL_READ_TIME,
        "publisher": str(publisher or ""),
        "source_url": str(source_url or ""),
        "vendor_or_source_id": str(vendor_or_source_id or ""),
        "market_tone": str(market_tone or ""),
        "impact_score": impact_score,
        "source_detail": str(source_detail or ""),
        "source_vendor": LOCAL_VENDOR,
        "pit_quality_status": LOCAL_QUALITY,
        "can_support_current_research": True,
        "can_support_institutional_backtest": False,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    score, status, lookahead = quality_for(row)
    row["event_time_score"] = score
    row["event_time_status"] = status
    row["lookahead_risk"] = lookahead
    row["limitation"] = (
        "Local/public/model event timing ledger. Needs paid vendor event IDs, exact first-seen timestamps, "
        "timezone-normalized release times, and immutable raw snapshots before institutional backtest use."
    )
    return row


def build_stock_news_rows(registry: dict[str, str]) -> list[dict[str, Any]]:
    path = ROOT / "stock_news.json"
    data = read_json_safe(path, {})
    updated = data.get("updated") if isinstance(data, dict) else ""
    first_seen, _ = parse_time(updated)
    first_seen = first_seen or file_seen_time(path)
    news = data.get("news", {}) if isinstance(data, dict) else {}
    rows: list[dict[str, Any]] = []
    if not isinstance(news, dict):
        return rows
    for ticker, articles in news.items():
        if not isinstance(articles, list):
            continue
        for article in articles:
            if not isinstance(article, dict):
                continue
            publish_raw = article.get("published_ts") or article.get("published")
            rows.append(make_row(
                event_type="raw_news",
                source_file="stock_news.json",
                ticker=ticker,
                headline=article.get("title", ""),
                publish_raw=publish_raw,
                first_seen_fallback=first_seen,
                publisher=article.get("publisher", ""),
                source_url=article.get("link", ""),
                vendor_or_source_id=article.get("raw_id", ""),
                market_tone=article.get("market_tone", ""),
                impact_score=article.get("impact_score", np.nan),
                source_detail=article.get("summary", ""),
                registry=registry,
            ))
    return rows


def build_csv_event_rows(registry: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    news_targets = read_csv_safe(ROOT / "news_impact_targets.csv")
    first_seen_news_targets = file_seen_time(ROOT / "news_impact_targets.csv")
    if not news_targets.empty:
        for _, r in news_targets.head(3000).iterrows():
            rows.append(make_row(
                event_type="news_readthrough_target",
                source_file="news_impact_targets.csv",
                ticker=r.get("target_ticker", ""),
                related_ticker=r.get("source_news_ticker", ""),
                headline=r.get("headline", ""),
                publish_raw=r.get("published", ""),
                first_seen_fallback=first_seen_news_targets,
                publisher=r.get("publisher", ""),
                source_url=r.get("link", ""),
                vendor_or_source_id=stable_hash(r.get("link", ""), r.get("headline", "")),
                market_tone=r.get("market_tone", ""),
                impact_score=r.get("impact_score", np.nan),
                source_detail=r.get("target_reason", ""),
                registry=registry,
            ))

    earnings = read_csv_safe(ROOT / "earnings_calendar.csv")
    first_seen_earnings = file_seen_time(ROOT / "earnings_calendar.csv")
    if not earnings.empty:
        for _, r in earnings.iterrows():
            rows.append(make_row(
                event_type="scheduled_earnings",
                source_file="earnings_calendar.csv",
                ticker=r.get("ticker", ""),
                headline=f"Earnings date: {r.get('ticker', '')}",
                event_date_raw=r.get("earnings_date", ""),
                publish_raw=r.get("earnings_date", ""),
                first_seen_fallback=first_seen_earnings,
                vendor_or_source_id="",
                market_tone=r.get("risk_flag", ""),
                impact_score=r.get("iv_rank", np.nan),
                source_detail=r.get("recommended_action", ""),
                registry=registry,
            ))

    surprise = read_csv_safe(ROOT / "earnings_surprise_scores.csv")
    first_seen_surprise = file_seen_time(ROOT / "earnings_surprise_scores.csv")
    if not surprise.empty:
        for _, r in surprise.head(1000).iterrows():
            rows.append(make_row(
                event_type="earnings_surprise",
                source_file="earnings_surprise_scores.csv",
                ticker=r.get("ticker", ""),
                headline=f"Earnings surprise: {r.get('ticker', '')} {r.get('signal', '')}",
                event_date_raw=r.get("earnings_date", ""),
                publish_raw=r.get("earnings_date", ""),
                first_seen_fallback=first_seen_surprise,
                market_tone=r.get("signal", ""),
                impact_score=r.get("surprise_pct", np.nan),
                source_detail=f"eps_actual={r.get('eps_actual', '')}; eps_estimate={r.get('eps_estimate', '')}",
                registry=registry,
            ))

    revisions = read_csv_safe(ROOT / "earnings_revision_scores.csv")
    first_seen_revision = file_seen_time(ROOT / "earnings_revision_scores.csv")
    if not revisions.empty:
        for _, r in revisions.head(1000).iterrows():
            rows.append(make_row(
                event_type="analyst_revision_snapshot",
                source_file="earnings_revision_scores.csv",
                ticker=r.get("ticker", ""),
                headline=f"Analyst revision snapshot: {r.get('ticker', '')} {r.get('signal', '')}",
                first_seen_fallback=first_seen_revision,
                market_tone=r.get("signal", ""),
                impact_score=r.get("revision_score", np.nan),
                source_detail=f"bull_chg={r.get('bull_chg', '')}; bear_chg={r.get('bear_chg', '')}; n_analysts={r.get('n_analysts', '')}",
                registry=registry,
            ))

    sec = read_csv_safe(ROOT / "sec_event_layer.csv")
    first_seen_sec = file_seen_time(ROOT / "sec_event_layer.csv")
    if not sec.empty:
        for _, r in sec.iterrows():
            rows.append(make_row(
                event_type="sec_filing_snapshot",
                source_file="sec_event_layer.csv",
                ticker=r.get("ticker", ""),
                headline=f"SEC filing: {r.get('ticker', '')} {r.get('latest_filing_type', '')}",
                event_date_raw=r.get("latest_filing_date", ""),
                publish_raw=r.get("latest_filing_date", ""),
                first_seen_fallback=first_seen_sec,
                vendor_or_source_id=str(r.get("latest_filing_type", "")),
                market_tone=r.get("filing_status", ""),
                impact_score=r.get("filing_count", np.nan),
                source_detail=r.get("notes", ""),
                registry=registry,
            ))

    insider = read_csv_safe(ROOT / "insider_signal_scores.csv")
    first_seen_insider = file_seen_time(ROOT / "insider_signal_scores.csv")
    if not insider.empty:
        for _, r in insider.head(1000).iterrows():
            rows.append(make_row(
                event_type="insider_activity_snapshot",
                source_file="insider_signal_scores.csv",
                ticker=r.get("ticker", ""),
                headline=f"Insider activity snapshot: {r.get('ticker', '')} {r.get('insider_signal', '')}",
                first_seen_fallback=first_seen_insider,
                market_tone=r.get("insider_signal", ""),
                impact_score=r.get("insider_raw", np.nan),
                source_detail=f"buy_count={r.get('buy_count', '')}; sell_count={r.get('sell_count', '')}; net_direction={r.get('net_direction', '')}",
                registry=registry,
            ))

    return rows


def build_ledger() -> pd.DataFrame:
    registry = load_registry()
    rows = build_stock_news_rows(registry) + build_csv_event_rows(registry)
    if not rows:
        return pd.DataFrame(columns=[
            "event_id", "event_type", "source_file", "ticker", "source_publish_time",
            "first_seen_time", "model_read_time", "event_time_status",
        ])
    out = pd.DataFrame(rows)
    out = out.drop_duplicates("event_id", keep="last").reset_index(drop=True)
    out = out.sort_values(["event_time_score", "event_type", "ticker"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def build_registry(ledger: pd.DataFrame) -> pd.DataFrame:
    old = read_csv_safe(OUT_REGISTRY)
    cols = ["event_id", "first_seen_time", "first_seen_source_file", "first_seen_model_read_time"]
    if old.empty:
        old = pd.DataFrame(columns=cols)
    rows = []
    old_map = dict(zip(old.get("event_id", pd.Series(dtype=str)).astype(str), old.get("first_seen_time", pd.Series(dtype=str)).astype(str)))
    for _, row in ledger.iterrows():
        eid = str(row.get("event_id", ""))
        rows.append({
            "event_id": eid,
            "first_seen_time": old_map.get(eid, row.get("first_seen_time", MODEL_READ_TIME)),
            "first_seen_source_file": row.get("source_file", ""),
            "first_seen_model_read_time": MODEL_READ_TIME,
        })
    reg = pd.DataFrame(rows)
    if not old.empty:
        reg = pd.concat([old[cols], reg], ignore_index=True, sort=False)
    return reg.drop_duplicates("event_id", keep="first").sort_values("first_seen_time").reset_index(drop=True)


def build_audit(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    rows = []
    for source, grp in ledger.groupby("source_file", dropna=False):
        statuses = grp["event_time_status"].astype(str).value_counts().to_dict()
        score = float(pd.to_numeric(grp["event_time_score"], errors="coerce").mean())
        missing_publish = int(grp["source_publish_time"].astype(str).isin(["", "nan", "NaT"]).sum())
        missing_first = int(grp["first_seen_time"].astype(str).isin(["", "nan", "NaT"]).sum())
        rows.append({
            "source_file": source,
            "event_rows": int(len(grp)),
            "average_event_time_score": round(score, 1),
            "auditable_rows": int((grp["event_time_status"] == "AUDITABLE_LOCAL_EVENT_TIME").sum()),
            "review_rows": int((grp["event_time_status"] == "REVIEW_EVENT_TIME").sum()),
            "weak_rows": int((grp["event_time_status"] == "WEAK_EVENT_TIME").sum()),
            "blocked_rows": int((grp["event_time_status"] == "BLOCKED_EVENT_TIME").sum()),
            "missing_publish_time_rows": missing_publish,
            "missing_first_seen_rows": missing_first,
            "status_counts": "; ".join(f"{k}:{v}" for k, v in statuses.items()),
            "required_next_action": "Add vendor event ID, exact release time, timezone, and immutable raw snapshot for every event.",
        })
    return pd.DataFrame(rows).sort_values(["blocked_rows", "weak_rows", "average_event_time_score"], ascending=[False, False, True]).reset_index(drop=True)


def write_outputs(ledger: pd.DataFrame, registry: pd.DataFrame, audit: pd.DataFrame) -> dict[str, Any]:
    ledger.to_csv(OUT_LEDGER, index=False)
    registry.to_csv(OUT_REGISTRY, index=False)
    audit.to_csv(OUT_AUDIT, index=False)

    avg_score = float(pd.to_numeric(ledger.get("event_time_score", pd.Series(dtype=float)), errors="coerce").mean()) if not ledger.empty else 0.0
    weak_or_blocked = int(ledger.get("event_time_status", pd.Series(dtype=str)).astype(str).isin(["WEAK_EVENT_TIME", "BLOCKED_EVENT_TIME"]).sum()) if not ledger.empty else 0
    auditable = int((ledger.get("event_time_status", pd.Series(dtype=str)) == "AUDITABLE_LOCAL_EVENT_TIME").sum()) if not ledger.empty else 0
    if ledger.empty:
        overall = "NO_EVENT_TIME_DATA"
    elif weak_or_blocked > len(ledger) * 0.35:
        overall = "EVENT_TIME_REPAIR_REQUIRED"
    elif weak_or_blocked > 0:
        overall = "EVENT_TIME_REVIEW_REQUIRED"
    elif avg_score >= 75:
        overall = "EVENT_TIME_LOCAL_AUDITABLE"
    else:
        overall = "EVENT_TIME_REVIEW_REQUIRED"
    state = {
        "date": today_str(),
        "generated_at": MODEL_READ_TIME,
        "overall_status": overall,
        "event_rows": int(len(ledger)),
        "sources": int(ledger["source_file"].nunique()) if not ledger.empty and "source_file" in ledger.columns else 0,
        "average_event_time_score": round(avg_score, 1),
        "auditable_local_rows": auditable,
        "weak_or_blocked_rows": weak_or_blocked,
        "registry_rows": int(len(registry)),
        "truth": "This ledger normalizes event timing and first-seen evidence. It is local/public/proxy evidence, not vendor-grade point-in-time event tape.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Event rows: **{state['event_rows']}**",
        f"- Sources: **{state['sources']}**",
        f"- Average event-time score: **{state['average_event_time_score']}/100**",
        f"- Auditable local rows: **{state['auditable_local_rows']}**",
        f"- Weak or blocked rows: **{state['weak_or_blocked_rows']}**",
        "",
        state["truth"],
        "",
        "## Source Audit",
        "",
        df_to_markdown(audit, max_rows=40),
        "",
        "## Sample Event Ledger",
        "",
        df_to_markdown(ledger.head(80), max_rows=80),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 163 - Event Time Truth Ledger", sections)
    return state


def main() -> None:
    ledger = build_ledger()
    registry = build_registry(ledger)
    audit = build_audit(ledger)
    state = write_outputs(ledger, registry, audit)
    print("Canyon v9 Step163 event time truth ledger complete.")
    print(f"Overall: {state.get('overall_status')} | rows: {state.get('event_rows')} | sources: {state.get('sources')}")
    print(f"Outputs: {OUT_LEDGER.name}, {OUT_REGISTRY.name}, {OUT_AUDIT.name}")


if __name__ == "__main__":
    main()
