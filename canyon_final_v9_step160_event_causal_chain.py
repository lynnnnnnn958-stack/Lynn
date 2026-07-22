#!/usr/bin/env python3
"""
Canyon v9 Step 160 - Event Causal Chain Validator.

Research-only. No broker connection. No live orders.

Step129 maps headlines to target tickers. Step130 adds price/liquidity checks
for theme candidates. Step160 turns those mappings into an auditable causal
chain: event -> direct entity -> sector/theme -> upstream/peer/downstream target
-> validation evidence.

The output is deliberately conservative. It labels most links as research
hypotheses unless they have source timestamp, link, chain logic, and price
confirmation.

Outputs:
  event_causal_chain_map.csv
  event_causal_chain_edges.csv
  event_causal_validation_queue.csv
  event_causal_chain_state.json
  event_causal_chain_report.md
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_MAP = ROOT / "event_causal_chain_map.csv"
OUT_EDGES = ROOT / "event_causal_chain_edges.csv"
OUT_QUEUE = ROOT / "event_causal_validation_queue.csv"
OUT_STATE = ROOT / "event_causal_chain_state.json"
OUT_REPORT = ROOT / "event_causal_chain_report.md"


def clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def make_id(*parts: Any, n: int = 12) -> str:
    text = "|".join(str(p or "") for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def parse_date(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    try:
        return pd.Timestamp(ts).tz_localize(None).normalize()
    except Exception:
        return pd.Timestamp(ts).normalize()


def load_price_cache() -> pd.DataFrame:
    for name in ["sp500_price_cache.csv", "backtest_price_cache.csv"]:
        df = read_csv_safe(ROOT / name)
        if df.empty:
            continue
        date_col = None
        for c in ["Date", "date", "Unnamed: 0"]:
            if c in df.columns:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > max(5, len(df) * 0.5):
                    date_col = c
                    break
        if date_col is None:
            continue
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        df.columns = [clean_ticker(c) for c in df.columns]
        return df.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame()


def event_window_returns(price: pd.DataFrame, ticker: str, published: Any) -> dict[str, Any]:
    ticker = clean_ticker(ticker)
    out = {
        "price_validation_source": "sp500_price_cache.csv/backtest_price_cache.csv",
        "event_price_status": "NO_PRICE_DATA",
        "event_trade_date": "",
        "available_days_after_event": 0,
        "event_to_latest_return_pct": np.nan,
        "post_1d_return_pct": np.nan,
        "post_5d_return_pct": np.nan,
        "post_20d_return_pct": np.nan,
    }
    if price.empty or ticker not in price.columns:
        return out
    event_date = parse_date(published)
    if event_date is None:
        out["event_price_status"] = "NO_PUBLISH_DATE"
        return out
    px = price[ticker].dropna()
    if px.empty:
        return out
    future_idx = px.index[px.index >= event_date]
    if len(future_idx) == 0:
        out["event_price_status"] = "EVENT_AFTER_PRICE_CACHE"
        return out
    event_idx = future_idx[0]
    loc = px.index.get_loc(event_idx)
    if isinstance(loc, slice):
        loc = loc.start
    event_price = float(px.iloc[int(loc)])
    if not np.isfinite(event_price) or event_price <= 0:
        return out
    out["event_trade_date"] = str(pd.Timestamp(event_idx).date())
    out["available_days_after_event"] = int(len(px.iloc[int(loc):]) - 1)
    latest = float(px.iloc[-1])
    out["event_to_latest_return_pct"] = round((latest / event_price - 1.0) * 100.0, 2)
    for horizon in [1, 5, 20]:
        if int(loc) + horizon < len(px):
            hpx = float(px.iloc[int(loc) + horizon])
            out[f"post_{horizon}d_return_pct"] = round((hpx / event_price - 1.0) * 100.0, 2)
    if out["available_days_after_event"] <= 0:
        out["event_price_status"] = "PENDING_PRICE_WINDOW"
    else:
        out["event_price_status"] = "PRICE_WINDOW_AVAILABLE"
    return out


def relation_layer(relation: Any) -> str:
    text = str(relation or "").lower()
    if text == "direct":
        return "DIRECT_ENTITY"
    if "mentioned" in text:
        return "DIRECT_MENTION"
    if "upstream" in text:
        return "UPSTREAM_SUPPLIER"
    if "downstream" in text:
        return "DOWNSTREAM_BENEFICIARY"
    if "theme" in text:
        return "THEME_PEER"
    if "peer" in text:
        return "SECTOR_PEER"
    return "CONTEXT_LINK"


def causal_thesis(row: pd.Series) -> str:
    tone = str(row.get("market_tone", "NEUTRAL")).upper()
    layer = relation_layer(row.get("target_relation", ""))
    theme = str(row.get("theme", "") or "")
    target = clean_ticker(row.get("target_ticker"))
    if tone == "NEGATIVE":
        if layer in {"SECTOR_PEER", "THEME_PEER"}:
            return f"Bad news can reprice weak or high-valuation peers; {target} is mapped as a vulnerable read-through target."
        return f"Bad news directly affects or mentions {target}; validate downside with price/volume and risk gates."
    if tone == "POSITIVE":
        if layer in {"UPSTREAM_SUPPLIER", "THEME_PEER", "DOWNSTREAM_BENEFICIARY"}:
            return f"Positive catalyst in {theme or 'the theme'} may lift related suppliers, peers, or downstream beneficiaries; {target} needs price confirmation."
        return f"Positive news directly affects or mentions {target}; validate upside with price/volume and risk gates."
    if tone == "MIXED":
        return f"Mixed event. Treat the {target} link as manual-review context until price or follow-up news clarifies direction."
    return f"Context headline. The {target} link is a research hypothesis, not a trade signal."


def price_confirmation(row: pd.Series) -> tuple[str, str]:
    tone = str(row.get("market_tone", "NEUTRAL")).upper()
    status = str(row.get("event_price_status", "NO_PRICE_DATA"))
    if status in {"NO_PRICE_DATA", "EVENT_AFTER_PRICE_CACHE", "NO_PUBLISH_DATE"}:
        return "NEED_PRICE_VALIDATION", "No usable event-time price window."
    if status == "PENDING_PRICE_WINDOW":
        return "PENDING_PRICE_WINDOW", "Headline is too fresh or price cache has no next observation yet."
    ret_candidates = [
        safe_float(row.get("post_5d_return_pct")),
        safe_float(row.get("post_1d_return_pct")),
        safe_float(row.get("event_to_latest_return_pct")),
    ]
    ret = next((x for x in ret_candidates if np.isfinite(x)), np.nan)
    if not np.isfinite(ret):
        return "NEED_MORE_CONFIRMATION", "Price window exists but confirmation return is not available."
    if tone == "POSITIVE":
        if ret >= 1.0:
            return "PRICE_CONFIRMING", f"Post-event return is positive ({ret:.2f}%)."
        if ret <= -1.0:
            return "PRICE_DISAGREES", f"Post-event return is negative ({ret:.2f}%)."
    if tone == "NEGATIVE":
        if ret <= -1.0:
            return "PRICE_CONFIRMING", f"Post-event return is negative ({ret:.2f}%)."
        if ret >= 1.0:
            return "PRICE_DISAGREES", f"Post-event return is positive ({ret:.2f}%)."
    return "NEED_MORE_CONFIRMATION", f"Post-event return is not decisive ({ret:.2f}%)."


def edge_confidence(row: pd.Series, validation_status: str) -> float:
    score = 25.0
    if str(row.get("link", "")).startswith("http"):
        score += 12.0
    if str(row.get("published", "")).strip():
        score += 10.0
    if str(row.get("publisher", "")).strip():
        score += 5.0
    layer = relation_layer(row.get("target_relation", ""))
    score += {
        "DIRECT_ENTITY": 25.0,
        "DIRECT_MENTION": 20.0,
        "SECTOR_PEER": 12.0,
        "THEME_PEER": 14.0,
        "UPSTREAM_SUPPLIER": 16.0,
        "DOWNSTREAM_BENEFICIARY": 14.0,
    }.get(layer, 5.0)
    if str(row.get("matched_theme_terms", "")).strip() or str(row.get("matched_terms", "")).strip():
        score += 8.0
    if validation_status == "PRICE_CONFIRMING":
        score += 18.0
    elif validation_status == "PRICE_DISAGREES":
        score -= 12.0
    elif validation_status in {"NEED_PRICE_VALIDATION", "PENDING_PRICE_WINDOW"}:
        score -= 5.0
    if str(row.get("data_status", "")).upper() != "IN_UNIVERSE":
        score -= 8.0
    return round(float(np.clip(score, 0, 100)), 1)


def build_edges() -> pd.DataFrame:
    targets = read_csv_safe(ROOT / "news_impact_targets.csv")
    supply = read_csv_safe(ROOT / "news_supply_chain_readthrough.csv")
    if targets.empty and supply.empty:
        return pd.DataFrame()
    source = pd.concat([targets, supply], ignore_index=True, sort=False).drop_duplicates(
        [c for c in ["target_ticker", "target_relation", "headline", "published", "theme", "chain_role"] if c in targets.columns or c in supply.columns],
        keep="first",
    )
    if source.empty:
        return source
    price = load_price_cache()
    price_metrics = read_csv_safe(ROOT / "theme_candidate_price_metrics.csv")
    metric_idx = pd.DataFrame()
    if not price_metrics.empty and "ticker" in price_metrics.columns:
        metric_idx = price_metrics.copy()
        metric_idx["ticker"] = metric_idx["ticker"].map(clean_ticker)
        metric_idx = metric_idx.drop_duplicates("ticker").set_index("ticker", drop=False)

    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        target = clean_ticker(row.get("target_ticker"))
        event_id = make_id(row.get("headline"), row.get("published"), row.get("publisher"), row.get("source_news_ticker"))
        edge_id = make_id(event_id, target, row.get("target_relation"), row.get("theme"), row.get("chain_role"))
        price_data = event_window_returns(price, target, row.get("published"))
        if price_data["event_price_status"] == "NO_PRICE_DATA" and target in metric_idx.index:
            m = metric_idx.loc[target]
            price_data.update({
                "price_validation_source": "theme_candidate_price_metrics.csv",
                "event_price_status": str(m.get("price_data_status", "THEME_CANDIDATE_PRICE_PROXY")),
                "post_5d_return_pct": safe_float(m.get("ret_5d_pct")),
                "post_20d_return_pct": safe_float(m.get("ret_20d_pct")),
                "event_to_latest_return_pct": safe_float(m.get("ret_20d_pct")),
            })
        base = {
            "event_id": event_id,
            "edge_id": edge_id,
            "headline": row.get("headline", ""),
            "published": row.get("published", ""),
            "publisher": row.get("publisher", ""),
            "link": row.get("link", ""),
            "source_news_ticker": clean_ticker(row.get("source_news_ticker")),
            "target_ticker": target,
            "target_relation": row.get("target_relation", ""),
            "relation_layer": relation_layer(row.get("target_relation", "")),
            "theme": row.get("theme", ""),
            "chain_role": row.get("chain_role", ""),
            "matched_theme_terms": row.get("matched_theme_terms", ""),
            "market_tone": str(row.get("market_tone", "NEUTRAL")).upper(),
            "impact_score": safe_float(row.get("impact_score"), 0.0),
            "news_logic": row.get("news_logic", ""),
            "suggested_research_route": row.get("suggested_research_route", ""),
            "option_side": row.get("option_side", ""),
            "data_status": row.get("data_status", ""),
            "total_vulnerability": safe_float(row.get("total_vulnerability"), np.nan),
            "final_risk_action": row.get("final_risk_action", ""),
            "source_file": row.get("source_file", "news_impact_targets.csv / news_supply_chain_readthrough.csv"),
            "causal_thesis": causal_thesis(row),
        }
        base.update(price_data)
        validation_status, validation_note = price_confirmation(pd.Series(base))
        base["causal_validation_status"] = validation_status
        base["validation_note"] = validation_note
        base["causal_confidence_score"] = edge_confidence(pd.Series(base), validation_status)
        if validation_status == "PRICE_CONFIRMING" and base["causal_confidence_score"] >= 75:
            base["causal_chain_status"] = "VALIDATED_RESEARCH_LINK"
        elif validation_status == "PRICE_DISAGREES":
            base["causal_chain_status"] = "CONTRADICTED_REVIEW_REQUIRED"
        elif base["relation_layer"] in {"DIRECT_ENTITY", "DIRECT_MENTION"} and base["causal_confidence_score"] >= 65:
            base["causal_chain_status"] = "SOURCE_SUPPORTED_NEEDS_PRICE_CONFIRMATION"
        else:
            base["causal_chain_status"] = "HYPOTHESIS_NEEDS_VALIDATION"
        base["research_only"] = True
        base["no_broker_connection"] = True
        rows.append(base)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["causal_chain_status", "causal_confidence_score", "impact_score"], ascending=[True, False, False]).reset_index(drop=True)


def build_map(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for event_id, grp in edges.groupby("event_id"):
        direct = grp[grp["relation_layer"].isin(["DIRECT_ENTITY", "DIRECT_MENTION"])]
        theme = ", ".join(sorted([x for x in grp.get("theme", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x and x.lower() != "nan"]))
        roles = ", ".join(sorted([x for x in grp.get("chain_role", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x and x.lower() != "nan"]))
        validated = int((grp["causal_chain_status"] == "VALIDATED_RESEARCH_LINK").sum())
        contradicted = int((grp["causal_chain_status"] == "CONTRADICTED_REVIEW_REQUIRED").sum())
        top = grp.sort_values("causal_confidence_score", ascending=False).iloc[0]
        rows.append({
            "event_id": event_id,
            "headline": top.get("headline", ""),
            "published": top.get("published", ""),
            "publisher": top.get("publisher", ""),
            "link": top.get("link", ""),
            "source_news_ticker": top.get("source_news_ticker", ""),
            "market_tone": top.get("market_tone", ""),
            "impact_score": top.get("impact_score", 0.0),
            "themes": theme,
            "chain_roles": roles,
            "direct_tickers": ", ".join(sorted(direct["target_ticker"].dropna().astype(str).unique().tolist())),
            "chain_target_count": int(grp["target_ticker"].nunique()),
            "validated_edge_count": validated,
            "contradicted_edge_count": contradicted,
            "avg_causal_confidence": round(float(pd.to_numeric(grp["causal_confidence_score"], errors="coerce").mean()), 1),
            "top_targets": ", ".join(grp.sort_values("causal_confidence_score", ascending=False)["target_ticker"].head(8).astype(str).tolist()),
            "map_status": "HAS_VALIDATED_EDGES" if validated else ("HAS_CONTRADICTIONS" if contradicted else "RESEARCH_HYPOTHESIS"),
            "source_file": "event_causal_chain_edges.csv",
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["map_status", "avg_causal_confidence", "chain_target_count"], ascending=[True, False, False]).reset_index(drop=True)


def build_queue(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    needs = edges[edges["causal_chain_status"] != "VALIDATED_RESEARCH_LINK"].copy()
    if needs.empty:
        return pd.DataFrame(columns=["priority", "target_ticker", "headline", "issue", "required_next_action"])
    def priority(row: pd.Series) -> str:
        if row.get("causal_chain_status") == "CONTRADICTED_REVIEW_REQUIRED":
            return "P1_REVIEW_CONTRADICTION"
        if row.get("causal_validation_status") in {"NEED_PRICE_VALIDATION", "PENDING_PRICE_WINDOW"}:
            return "P2_PRICE_VALIDATION"
        if str(row.get("link", "")).strip() == "" or str(row.get("published", "")).strip() == "":
            return "P1_SOURCE_TIMESTAMP"
        return "P3_CHAIN_EVIDENCE"

    needs["priority"] = needs.apply(priority, axis=1)
    needs["issue"] = needs["causal_chain_status"].astype(str) + " / " + needs["causal_validation_status"].astype(str)
    needs["required_next_action"] = needs.apply(
        lambda r: (
            "Check source link and event timestamp, then verify the target's post-event price/volume reaction."
            if r["priority"] in {"P1_SOURCE_TIMESTAMP", "P2_PRICE_VALIDATION"} else
            "Document why this target belongs in the supply chain and whether the link is direct, upstream, peer, or downstream."
        ),
        axis=1,
    )
    cols = [
        "priority", "target_ticker", "relation_layer", "theme", "chain_role",
        "market_tone", "headline", "published", "publisher", "causal_confidence_score",
        "issue", "validation_note", "required_next_action", "link",
    ]
    return needs[[c for c in cols if c in needs.columns]].sort_values(["priority", "causal_confidence_score"], ascending=[True, False]).reset_index(drop=True)


def write_outputs(edges: pd.DataFrame, event_map: pd.DataFrame, queue: pd.DataFrame) -> None:
    edges.to_csv(OUT_EDGES, index=False)
    event_map.to_csv(OUT_MAP, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    edge_count = int(len(edges))
    validated = int((edges.get("causal_chain_status", pd.Series(dtype=str)) == "VALIDATED_RESEARCH_LINK").sum()) if not edges.empty else 0
    contradicted = int((edges.get("causal_chain_status", pd.Series(dtype=str)) == "CONTRADICTED_REVIEW_REQUIRED").sum()) if not edges.empty else 0
    avg_conf = float(pd.to_numeric(edges.get("causal_confidence_score", pd.Series(dtype=float)), errors="coerce").mean()) if not edges.empty else 0.0
    if edge_count == 0:
        overall = "NO_EVENT_CHAIN_DATA"
    elif contradicted > 0 or len(queue) > edge_count * 0.75:
        overall = "CAUSAL_REVIEW_REQUIRED"
    elif validated / max(edge_count, 1) >= 0.35:
        overall = "CAUSAL_CHAIN_PARTLY_VALIDATED"
    else:
        overall = "CAUSAL_HYPOTHESIS_BOARD"
    state = {
        "date": today_str(),
        "overall_status": overall,
        "event_count": int(len(event_map)),
        "edge_count": edge_count,
        "validated_edge_count": validated,
        "contradicted_edge_count": contradicted,
        "validation_queue_rows": int(len(queue)),
        "average_causal_confidence": round(avg_conf, 1),
        "truth": "These are auditable event-to-industry hypotheses. They are not proof of causality unless source timing and post-event market response validate the link.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Events mapped: **{state['event_count']}**",
        f"- Causal edges: **{state['edge_count']}**",
        f"- Validated edges: **{state['validated_edge_count']}**",
        f"- Validation queue rows: **{state['validation_queue_rows']}**",
        "",
        state["truth"],
        "",
        "## Event Chain Map",
        "",
        df_to_markdown(event_map, max_rows=80),
        "",
        "## Causal Edges",
        "",
        df_to_markdown(edges, max_rows=120),
        "",
        "## Validation Queue",
        "",
        df_to_markdown(queue, max_rows=80),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 160 - Event Causal Chain Validator", sections)


def main() -> None:
    edges = build_edges()
    event_map = build_map(edges)
    queue = build_queue(edges)
    write_outputs(edges, event_map, queue)
    state = read_json_safe(OUT_STATE, {})
    print("Canyon v9 Step160 event causal chain complete.")
    print(f"Overall: {state.get('overall_status')} | edges={state.get('edge_count')} validated={state.get('validated_edge_count')}")
    print(f"Outputs: {OUT_MAP.name}, {OUT_EDGES.name}, {OUT_QUEUE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
