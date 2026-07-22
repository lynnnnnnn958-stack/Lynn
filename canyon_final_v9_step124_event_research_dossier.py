#!/usr/bin/env python3
"""
Canyon v9 Step 124 - Event Research Dossier.

Research-only. No broker connection. No live orders.

This step upgrades event and fundamental depth from scattered files into a
per-ticker research dossier:
  - earnings timing and surprise
  - analyst revisions
  - earnings call NLP / guidance tone
  - insider signal
  - SEC event status
  - news risk
  - source coverage and missing research gaps

The output can reduce or block research ideas. It cannot upgrade a ticker by
itself.

Outputs:
  event_research_dossier.csv
  event_research_gate.csv
  event_source_coverage.csv
  event_missing_research_queue.csv
  event_research_state.json
  event_research_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    load_current_book,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_DOSSIER = ROOT / "event_research_dossier.csv"
OUT_GATE = ROOT / "event_research_gate.csv"
OUT_COVERAGE = ROOT / "event_source_coverage.csv"
OUT_QUEUE = ROOT / "event_missing_research_queue.csv"
OUT_STATE = ROOT / "event_research_state.json"
OUT_REPORT = ROOT / "event_research_report.md"

REQUIRED_SOURCES = [
    "earnings_calendar",
    "earnings_surprise",
    "earnings_revision",
    "earnings_call_nlp",
    "insider_form4",
    "sec_event",
    "news_event",
    "raw_news",
]


def num(value: Any, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else default


def status_from_score(score: float) -> str:
    if score >= 85:
        return "PASS"
    if score >= 65:
        return "REVIEW"
    if score >= 40:
        return "WEAK"
    return "BLOCKER"


def gate_from_risk(score: float, coverage_score: float) -> str:
    if coverage_score < 45:
        return "MISSING_DATA_REVIEW"
    if score >= 75:
        return "BLOCK_NEW"
    if score >= 55:
        return "SIZE_DOWN"
    if score >= 35:
        return "REVIEW"
    return "CLEAR"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    out = df.copy()
    out["ticker"] = out["ticker"].apply(clean_ticker)
    return out


def first_row(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty or "ticker" not in df.columns:
        return pd.Series(dtype=object)
    sub = df[df["ticker"].astype(str) == ticker]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def load_sources() -> dict[str, Any]:
    news_json = read_json_safe(ROOT / "stock_news.json", {})
    return {
        "earnings_calendar": prep(read_csv_safe(ROOT / "earnings_calendar.csv")),
        "earnings_surprise": prep(read_csv_safe(ROOT / "earnings_surprise_scores.csv")),
        "earnings_revision": prep(read_csv_safe(ROOT / "earnings_revision_scores.csv")),
        "earnings_call_nlp": prep(read_csv_safe(ROOT / "earnings_nlp_scores.csv")),
        "insider_form4": prep(read_csv_safe(ROOT / "insider_signal_scores.csv")),
        "sec_event": prep(read_csv_safe(ROOT / "sec_event_layer.csv")),
        "news_event": prep(read_csv_safe(ROOT / "news_event_risk.csv")),
        "raw_news": news_json.get("news", {}) if isinstance(news_json, dict) else {},
    }


def latest_news_summary(news_map: dict[str, Any], ticker: str) -> tuple[bool, str, str]:
    items = news_map.get(ticker, []) if isinstance(news_map, dict) else []
    if not items:
        return False, "", ""
    valid = []
    for item in items:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        published = item.get("published") or item.get("providerPublishTime") or item.get("published_ts") or ""
        valid.append((title, str(published), str(item.get("publisher", ""))))
    if not valid:
        return False, "", ""
    title, published, publisher = valid[0]
    return True, title[:220], f"{publisher} {published}".strip()


def build_dossier() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    book = load_current_book(prefer_filtered=True)
    if book.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sources = load_sources()
    rows = []
    coverage_rows = []
    gate_rows = []
    queue_rows = []

    for _, b in book.iterrows():
        ticker = clean_ticker(b.get("ticker"))
        sector = str(b.get("sector", "Unknown"))
        top_signal = str(b.get("top_signal", ""))
        ec = first_row(sources["earnings_calendar"], ticker)
        es = first_row(sources["earnings_surprise"], ticker)
        er = first_row(sources["earnings_revision"], ticker)
        nlp = first_row(sources["earnings_call_nlp"], ticker)
        insider = first_row(sources["insider_form4"], ticker)
        sec = first_row(sources["sec_event"], ticker)
        news_event = first_row(sources["news_event"], ticker)
        has_raw_news, news_title, news_meta = latest_news_summary(sources["raw_news"], ticker)

        checks = {
            "earnings_calendar": not ec.empty,
            "earnings_surprise": not es.empty,
            "earnings_revision": not er.empty,
            "earnings_call_nlp": not nlp.empty,
            "insider_form4": not insider.empty,
            "sec_event": not sec.empty,
            "news_event": not news_event.empty,
            "raw_news": has_raw_news,
        }
        coverage_score = 100.0 * sum(bool(v) for v in checks.values()) / len(REQUIRED_SOURCES)

        risk_score = 0.0
        catalysts = []
        risks = []

        days_until = num(ec.get("days_until")) if not ec.empty else np.nan
        ec_risk = str(ec.get("risk_flag", "")).upper() if not ec.empty else ""
        if np.isfinite(days_until) and -2 <= days_until <= 7:
            risk_score += 25
            risks.append("near earnings window")
        if ec_risk == "HIGH":
            risk_score += 18
            risks.append("earnings calendar high risk")

        surprise_pct = num(es.get("surprise_pct")) if not es.empty else np.nan
        surprise_signal = str(es.get("signal", "")).upper() if not es.empty else ""
        days_since = num(es.get("days_since")) if not es.empty else np.nan
        if surprise_signal == "MISS" or (np.isfinite(surprise_pct) and surprise_pct <= -5):
            risk_score += 28
            risks.append("negative earnings surprise")
        elif surprise_signal == "BEAT" or (np.isfinite(surprise_pct) and surprise_pct >= 5):
            catalysts.append("positive earnings surprise")
            if np.isfinite(days_since) and days_since <= 5:
                risk_score += 8

        revision_score = num(er.get("revision_score")) if not er.empty else np.nan
        revision_signal = str(er.get("signal", "")).upper() if not er.empty else ""
        if revision_signal in {"DOWNGRADE", "CUT"} or (np.isfinite(revision_score) and revision_score < 35):
            risk_score += 18
            risks.append("weak analyst revision")
        elif revision_signal in {"UPGRADE", "RAISE"} or (np.isfinite(revision_score) and revision_score > 70):
            catalysts.append("positive analyst revision")

        nlp_sentiment = str(nlp.get("sentiment", "")).upper() if not nlp.empty else ""
        guidance = str(nlp.get("guidance_tone", "")).upper() if not nlp.empty else ""
        forward_score = num(nlp.get("forward_score")) if not nlp.empty else np.nan
        if "BEAR" in nlp_sentiment or guidance in {"LOWERED", "CUT"} or (np.isfinite(forward_score) and forward_score < -0.25):
            risk_score += 18
            risks.append("negative call/guidance tone")
        elif "BULL" in nlp_sentiment or guidance in {"RAISED"} or (np.isfinite(forward_score) and forward_score > 0.25):
            catalysts.append("positive call/guidance tone")

        insider_signal = str(insider.get("insider_signal", "")).upper() if not insider.empty else ""
        buy_pressure = num(insider.get("buy_pressure")) if not insider.empty else np.nan
        if "SELL" in insider_signal:
            risk_score += 10
            risks.append("insider selling pressure")
        elif "BUY" in insider_signal or (np.isfinite(buy_pressure) and buy_pressure > 0.5):
            catalysts.append("insider buy support")

        filing_status = str(sec.get("filing_status", "")).upper() if not sec.empty else ""
        filing_type = str(sec.get("latest_filing_type", "")) if not sec.empty else ""
        if filing_status and filing_status not in {"OK", "ETF_NO_SEC_COMPANY_FILINGS"}:
            risk_score += 12
            risks.append("SEC filing status requires manual review")
        elif filing_type:
            catalysts.append(f"latest SEC filing {filing_type}")

        news_risk = str(news_event.get("risk_label", "")).upper() if not news_event.empty else ""
        if news_risk in {"HIGH", "RED"}:
            risk_score += 20
            risks.append("high news risk")
        elif news_risk in {"LOW"}:
            catalysts.append("low headline risk")
        if has_raw_news and news_title:
            catalysts.append("raw news available")

        missing = [k for k, v in checks.items() if not v]
        if missing:
            risk_score += min(20, len(missing) * 3)

        risk_score = min(100.0, risk_score)
        gate = gate_from_risk(risk_score, coverage_score)
        research_score = max(0.0, min(100.0, coverage_score * 0.65 + (100.0 - risk_score) * 0.35))
        status = status_from_score(research_score)

        rows.append({
            "ticker": ticker,
            "sector": sector,
            "top_signal": top_signal,
            "event_research_score": round(research_score, 1),
            "event_source_coverage_pct": round(coverage_score, 1),
            "event_risk_score": round(risk_score, 1),
            "event_gate": gate,
            "status": status,
            "earnings_date": ec.get("earnings_date", "") if not ec.empty else "",
            "days_until_earnings": days_until,
            "earnings_risk_flag": ec_risk,
            "surprise_pct": surprise_pct,
            "surprise_signal": surprise_signal,
            "days_since_earnings": days_since,
            "revision_score": revision_score,
            "revision_signal": revision_signal,
            "call_sentiment": nlp_sentiment,
            "guidance_tone": guidance,
            "insider_signal": insider_signal,
            "sec_filing_status": filing_status,
            "latest_filing_type": filing_type,
            "news_risk_label": news_risk,
            "latest_news_title": news_title,
            "latest_news_meta": news_meta,
            "catalysts": "; ".join(catalysts) if catalysts else "none found",
            "risks": "; ".join(risks) if risks else "none found in current sources",
            "missing_research_sources": ", ".join(missing),
            "required_next_action": "Manually verify missing event sources before increasing size." if missing else "Review event gate before paper action.",
            "source_file": "earnings_calendar.csv / earnings_surprise_scores.csv / earnings_revision_scores.csv / earnings_nlp_scores.csv / insider_signal_scores.csv / sec_event_layer.csv / news_event_risk.csv / stock_news.json",
        })

        gate_rows.append({
            "ticker": ticker,
            "event_gate": gate,
            "event_risk_score": round(risk_score, 1),
            "event_research_score": round(research_score, 1),
            "event_source_coverage_pct": round(coverage_score, 1),
            "reason": "; ".join(risks) if risks else "no major event risk in current sources",
            "source_file": "event_research_dossier.csv",
        })
        coverage_rows.append({
            "ticker": ticker,
            **{f"has_{k}": bool(v) for k, v in checks.items()},
            "coverage_pct": round(coverage_score, 1),
            "missing_count": len(missing),
            "missing_sources": ", ".join(missing),
            "source_file": "event_research_dossier.csv",
        })
        for missing_source in missing:
            queue_rows.append({
                "ticker": ticker,
                "missing_source": missing_source,
                "priority": "HIGH" if missing_source in {"earnings_call_nlp", "sec_event", "news_event"} else "MEDIUM",
                "why_it_matters": {
                    "earnings_call_nlp": "Guidance tone can change the thesis even when EPS beats.",
                    "sec_event": "Fresh filings can reveal risks that price data misses.",
                    "news_event": "Headline risk needs source and timestamp validation.",
                    "raw_news": "Raw headline source is needed for manual verification.",
                    "insider_form4": "Insider quality helps separate noise from signal.",
                    "earnings_revision": "Analyst estimate direction changes forward expectations.",
                    "earnings_surprise": "Post-earnings drift and gap risk need recent surprise context.",
                    "earnings_calendar": "Position risk changes around exact earnings time.",
                }.get(missing_source, "Missing source weakens research confidence."),
                "source_file": "event_research_dossier.csv",
            })

    return pd.DataFrame(rows), pd.DataFrame(gate_rows), pd.DataFrame(coverage_rows), pd.DataFrame(queue_rows)


def write_outputs(dossier: pd.DataFrame, gate: pd.DataFrame, coverage: pd.DataFrame, queue: pd.DataFrame) -> None:
    dossier.to_csv(OUT_DOSSIER, index=False)
    gate.to_csv(OUT_GATE, index=False)
    coverage.to_csv(OUT_COVERAGE, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    avg_score = float(pd.to_numeric(dossier.get("event_research_score", pd.Series(dtype=float)), errors="coerce").mean()) if not dossier.empty else 0.0
    avg_coverage = float(pd.to_numeric(dossier.get("event_source_coverage_pct", pd.Series(dtype=float)), errors="coerce").mean()) if not dossier.empty else 0.0
    gate_flags = int(gate.get("event_gate", pd.Series(dtype=str)).astype(str).str.upper().isin(["SIZE_DOWN", "BLOCK_NEW", "MISSING_DATA_REVIEW"]).sum()) if not gate.empty else 0
    state = {
        "date": today_str(),
        "event_research_score": round(avg_score, 1),
        "event_source_coverage_pct": round(avg_coverage, 1),
        "overall_status": status_from_score(avg_score),
        "event_gate_flags": gate_flags,
        "missing_research_items": int(len(queue)),
        "research_only": True,
        "no_broker_connection": True,
        "truth": "Structured event dossier only. It does not replace analyst-grade manual research or paid event data.",
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Product Truth",
        "",
        state["truth"],
        "",
        f"- Event research score: {state['event_research_score']}",
        f"- Average source coverage: {state['event_source_coverage_pct']}%",
        f"- Event gate flags: {state['event_gate_flags']}",
        f"- Missing research items: {state['missing_research_items']}",
        "",
        "## Event Gate",
        "",
        df_to_markdown(gate, max_rows=80),
        "",
        "## Dossier",
        "",
        df_to_markdown(dossier, max_rows=80),
        "",
        "## Source Coverage",
        "",
        df_to_markdown(coverage, max_rows=80),
        "",
        "## Missing Research Queue",
        "",
        df_to_markdown(queue, max_rows=120),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 124 - Event Research Dossier", sections)


def main() -> None:
    dossier, gate, coverage, queue = build_dossier()
    write_outputs(dossier, gate, coverage, queue)
    state = read_json_safe(OUT_STATE, {})
    print(f"[step124] wrote {OUT_DOSSIER.name}: {len(dossier)} dossiers")
    print(f"[step124] score={state.get('event_research_score')} coverage={state.get('event_source_coverage_pct')}% flags={state.get('event_gate_flags')}")
    print(f"[step124] wrote {OUT_GATE.name}, {OUT_COVERAGE.name}, {OUT_QUEUE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
