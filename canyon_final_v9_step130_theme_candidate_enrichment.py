#!/usr/bin/env python3
"""
Canyon v9 - Step 130: Theme Candidate Enrichment
================================================

Research-only. No broker connection. No live orders.

Step129 maps news into supply-chain/theme targets. This step turns those
targets into a real research queue by adding price, trend, volatility, and
liquidity checks. It is designed for external theme tickers such as RKLB,
LUNR, ASTS, RDW, PL, and other names that may not yet be in the Canyon
core universe.

Outputs:
  theme_candidate_enrichment.csv
  theme_candidate_price_metrics.csv
  theme_candidate_enrichment_state.json
  theme_candidate_enrichment_report.md
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

SUPPLY_CHAIN_CSV = ROOT / "news_supply_chain_readthrough.csv"
OUT_ENRICHED = ROOT / "theme_candidate_enrichment.csv"
OUT_PRICE_METRICS = ROOT / "theme_candidate_price_metrics.csv"
OUT_STATE = ROOT / "theme_candidate_enrichment_state.json"
OUT_REPORT = ROOT / "theme_candidate_enrichment_report.md"


def clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich theme read-through tickers with price/liquidity checks.")
    p.add_argument("--top", type=int, default=120, help="Max unique tickers to enrich.")
    p.add_argument("--period", default="1y", help="yfinance history period, e.g. 6mo, 1y, 2y.")
    p.add_argument("--refresh", action="store_true", help="Reserved for future cache refresh control.")
    return p.parse_args()


def summarize_supply_chain(sc: pd.DataFrame, top: int) -> pd.DataFrame:
    if sc.empty or "target_ticker" not in sc.columns:
        return pd.DataFrame()

    sc = sc.copy()
    sc["target_ticker"] = sc["target_ticker"].map(clean_ticker)
    sc["market_tone"] = sc.get("market_tone", "").astype(str).str.upper()
    sc["impact_score"] = pd.to_numeric(sc.get("impact_score", 0), errors="coerce").fillna(0)
    sc["is_external"] = sc.get("data_status", "").astype(str).eq("EXTERNAL_THEME_TARGET_NEEDS_DATA")

    rows: list[dict[str, Any]] = []
    for ticker, grp in sc.groupby("target_ticker"):
        pos = int((grp["market_tone"] == "POSITIVE").sum())
        neg = int((grp["market_tone"] == "NEGATIVE").sum())
        mixed = int((grp["market_tone"] == "MIXED").sum())
        neutral = int((grp["market_tone"] == "NEUTRAL").sum())
        external = bool(grp["is_external"].any())
        themes = ", ".join(sorted(grp.get("theme", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))
        roles = ", ".join(sorted(grp.get("chain_role", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))
        providers = ", ".join(sorted(grp.get("publisher", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:5])
        top_row = grp.sort_values(["market_tone", "impact_score"], ascending=[True, False]).iloc[0]
        attention = (
            len(grp) * 2
            + pos * 9
            + neg * 8
            + mixed * 4
            + min(20.0, float(grp["impact_score"].abs().sum()))
            + (5 if external else 0)
        )
        rows.append({
            "ticker": ticker,
            "theme": themes,
            "chain_role": roles,
            "catalyst_count": int(len(grp)),
            "positive_catalysts": pos,
            "negative_catalysts": neg,
            "mixed_catalysts": mixed,
            "context_catalysts": neutral,
            "attention_score": round(float(attention), 2),
            "is_external_theme_target": external,
            "top_headline": str(top_row.get("headline", "")),
            "source_news_ticker": str(top_row.get("source_news_ticker", "")),
            "publisher_sample": providers,
            "source_file": "news_supply_chain_readthrough.csv",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["attention_score", "positive_catalysts", "negative_catalysts", "catalyst_count"],
        ascending=[False, False, False, False],
    ).head(top).reset_index(drop=True)


def extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)
        if field in lvl0:
            out = raw[field].copy()
        elif field in lvl1:
            out = raw.xs(field, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if field not in raw.columns:
            return pd.DataFrame()
        out = raw[[field]].copy()
        if len(tickers) == 1:
            out.columns = tickers
    out.columns = [clean_ticker(c) for c in out.columns]
    return out.loc[:, ~pd.Index(out.columns).duplicated()]


def download_price_data(tickers: list[str], period: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), "NO_TICKERS"
    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"YFINANCE_IMPORT_FAILED: {exc}"

    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"YFINANCE_DOWNLOAD_FAILED: {exc}"

    close = extract_field(raw, "Close", tickers)
    volume = extract_field(raw, "Volume", tickers)
    return close, volume, "OK"


def safe_pct(close: pd.DataFrame, days: int) -> pd.Series:
    if close.empty or len(close) <= days:
        return pd.Series(dtype=float)
    return close.ffill().iloc[-1] / close.ffill().iloc[-days - 1] - 1.0


def build_price_metrics(close: pd.DataFrame, volume: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if close.empty:
        for ticker in requested:
            rows.append({"ticker": ticker, "price_data_status": "NO_PRICE_DATA"})
        return pd.DataFrame(rows)

    close = close.ffill()
    volume = volume.reindex_like(close).fillna(0) if not volume.empty else pd.DataFrame(0, index=close.index, columns=close.columns)
    rets = close.pct_change()
    ret_5 = safe_pct(close, 5)
    ret_20 = safe_pct(close, 20)
    ret_63 = safe_pct(close, 63)
    vol_20 = rets.tail(20).std() * np.sqrt(252)
    avg_dv_20 = (close * volume).tail(20).mean()
    ma20 = close.tail(20).mean()
    ma50 = close.tail(50).mean()
    high_252 = close.tail(min(252, len(close))).max()
    last = close.iloc[-1]

    for ticker in requested:
        if ticker not in close.columns:
            rows.append({"ticker": ticker, "price_data_status": "NO_PRICE_DATA"})
            continue
        series = close[ticker].dropna()
        data_days = int(series.shape[0])
        if data_days < 20:
            rows.append({"ticker": ticker, "price_data_status": "LIMITED_OR_NO_PRICE_DATA", "data_days": data_days})
            continue

        last_close = float(last.get(ticker, np.nan))
        rv20 = float(vol_20.get(ticker, np.nan) * 100) if ticker in vol_20.index else np.nan
        adv = float(avg_dv_20.get(ticker, np.nan)) if ticker in avg_dv_20.index else np.nan
        r5 = float(ret_5.get(ticker, np.nan) * 100) if ticker in ret_5.index else np.nan
        r20 = float(ret_20.get(ticker, np.nan) * 100) if ticker in ret_20.index else np.nan
        r63 = float(ret_63.get(ticker, np.nan) * 100) if ticker in ret_63.index else np.nan
        ma20_v = float(ma20.get(ticker, np.nan)) if ticker in ma20.index else np.nan
        ma50_v = float(ma50.get(ticker, np.nan)) if ticker in ma50.index else np.nan
        high = float(high_252.get(ticker, np.nan)) if ticker in high_252.index else np.nan

        if not np.isfinite(adv):
            liquidity_status = "UNKNOWN_LIQUIDITY"
        elif adv >= 50_000_000:
            liquidity_status = "LIQUID"
        elif adv >= 10_000_000:
            liquidity_status = "TRADEABLE_RESEARCH"
        elif adv >= 2_000_000:
            liquidity_status = "THIN"
        else:
            liquidity_status = "TOO_THIN"

        if np.isfinite(last_close) and np.isfinite(ma20_v) and np.isfinite(ma50_v):
            if last_close > ma20_v > ma50_v and r20 > 0:
                trend_state = "UPTREND_CONFIRMED"
            elif last_close > ma50_v and r20 >= 0:
                trend_state = "MIXED_UP"
            elif last_close < ma20_v < ma50_v and r20 < 0:
                trend_state = "DOWNTREND_CONFIRMED"
            else:
                trend_state = "MIXED"
        else:
            trend_state = "UNKNOWN_TREND"

        if not np.isfinite(rv20):
            vol_state = "UNKNOWN_VOL"
        elif rv20 >= 90:
            vol_state = "EXTREME_VOL"
        elif rv20 >= 55:
            vol_state = "HIGH_VOL"
        elif rv20 >= 30:
            vol_state = "ELEVATED_VOL"
        else:
            vol_state = "NORMAL_VOL"

        rows.append({
            "ticker": ticker,
            "price_data_status": "OK" if data_days >= 60 else "LIMITED_HISTORY",
            "data_days": data_days,
            "last_close": round(last_close, 4) if np.isfinite(last_close) else np.nan,
            "ret_5d_pct": round(r5, 2) if np.isfinite(r5) else np.nan,
            "ret_20d_pct": round(r20, 2) if np.isfinite(r20) else np.nan,
            "ret_63d_pct": round(r63, 2) if np.isfinite(r63) else np.nan,
            "realized_vol_20d_pct": round(rv20, 2) if np.isfinite(rv20) else np.nan,
            "avg_dollar_volume_20d": round(adv, 0) if np.isfinite(adv) else np.nan,
            "liquidity_status": liquidity_status,
            "trend_state": trend_state,
            "volatility_state": vol_state,
            "distance_to_52w_high_pct": round((last_close / high - 1.0) * 100, 2) if np.isfinite(last_close) and np.isfinite(high) and high else np.nan,
            "price_source": "yfinance",
        })
    return pd.DataFrame(rows)


def route_candidate(row: pd.Series) -> tuple[str, str, str]:
    price_status = str(row.get("price_data_status", "NO_PRICE_DATA"))
    liquidity = str(row.get("liquidity_status", "UNKNOWN"))
    trend = str(row.get("trend_state", "UNKNOWN"))
    vol_state = str(row.get("volatility_state", "UNKNOWN"))
    pos = int(row.get("positive_catalysts", 0) or 0)
    neg = int(row.get("negative_catalysts", 0) or 0)

    if "NO_PRICE" in price_status or "LIMITED_OR_NO" in price_status:
        return "NEED_PRICE_DATA", "NONE", "No reliable price history yet; keep it in research queue only."
    if liquidity in {"TOO_THIN", "UNKNOWN_LIQUIDITY"}:
        return "WATCH_ONLY_LIQUIDITY_BLOCK", "NONE", "Liquidity is too thin or unknown; no option idea until liquidity is checked."
    if vol_state == "EXTREME_VOL":
        return "WATCH_ONLY_EXTREME_VOL", "NONE", "Realized volatility is extreme; wait for calmer confirmation."
    if neg > pos and "DOWNTREND" in trend:
        return "DOWNSIDE_RISK_REVIEW", "PUT_REVIEW", "Negative catalyst plus confirmed downtrend; review downside risk only."
    if pos > 0 and "UPTREND" in trend and liquidity in {"LIQUID", "TRADEABLE_RESEARCH"}:
        return "ACTIVE_RESEARCH_READY", "CALL_OR_STOCK_REVIEW", "Positive catalyst has price confirmation and enough liquidity for research."
    if pos > 0:
        return "WAIT_FOR_PRICE_CONFIRMATION", "STOCK_FIRST_NO_OPTIONS", "Positive catalyst exists but price/liquidity confirmation is incomplete."
    return "CONTEXT_WATCHLIST", "NONE", "Theme link exists, but the current headline is context rather than a clear catalyst."


def build_enrichment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sc = read_csv_safe(SUPPLY_CHAIN_CSV)
    candidates = summarize_supply_chain(sc, args.top)
    if candidates.empty:
        state = {"status": "NO_SUPPLY_CHAIN_DATA", "run_time": datetime.now().replace(microsecond=0).isoformat()}
        return pd.DataFrame(), pd.DataFrame(), state

    tickers = candidates["ticker"].dropna().astype(str).map(clean_ticker).drop_duplicates().tolist()
    close, volume, download_status = download_price_data(tickers, args.period)
    price_metrics = build_price_metrics(close, volume, tickers)
    enriched = candidates.merge(price_metrics, on="ticker", how="left")

    routes = enriched.apply(route_candidate, axis=1, result_type="expand")
    routes.columns = ["theme_candidate_status", "option_research_side", "status_reason"]
    enriched = pd.concat([enriched, routes], axis=1)
    enriched["research_only"] = True
    enriched["no_broker_connection"] = True
    enriched["as_of"] = datetime.now().replace(microsecond=0).isoformat()
    enriched = enriched.sort_values(
        ["theme_candidate_status", "attention_score", "positive_catalysts"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    state = {
        "status": "OK" if download_status == "OK" else download_status,
        "run_time": datetime.now().replace(microsecond=0).isoformat(),
        "candidate_rows": int(len(enriched)),
        "price_metric_rows": int(len(price_metrics)),
        "external_candidate_rows": int(enriched.get("is_external_theme_target", pd.Series(dtype=bool)).fillna(False).sum()),
        "active_research_ready": int((enriched["theme_candidate_status"] == "ACTIVE_RESEARCH_READY").sum()),
        "needs_price_data": int(enriched["theme_candidate_status"].astype(str).str.contains("NEED_PRICE_DATA", na=False).sum()),
        "liquidity_blocked": int(enriched["theme_candidate_status"].astype(str).str.contains("LIQUIDITY_BLOCK", na=False).sum()),
        "themes": sorted([str(x) for x in sc.get("theme", pd.Series(dtype=str)).dropna().unique().tolist()]) if not sc.empty else [],
        "price_source": "yfinance",
        "research_only": True,
        "no_broker_connection": True,
    }
    return enriched, price_metrics, state


def write_report(enriched: pd.DataFrame, state: dict[str, Any]) -> None:
    lines = [
        "# Canyon v9 - Step 130 Theme Candidate Enrichment",
        f"Generated: {state.get('run_time', datetime.now().isoformat())}",
        "",
        "Research-only. No broker connection. No live orders.",
        "",
        "## What This Step Adds",
        "- Takes Step129 supply-chain read-through targets and adds price, liquidity, trend, and volatility checks.",
        "- Keeps external theme tickers such as RKLB in a research queue instead of pretending they are fully covered.",
        "- Produces a clear status: ACTIVE_RESEARCH_READY, WAIT_FOR_PRICE_CONFIRMATION, WATCH_ONLY_LIQUIDITY_BLOCK, or NEED_PRICE_DATA.",
        "",
        "## State",
        f"- Candidate rows: {state.get('candidate_rows', 0)}",
        f"- External candidate rows: {state.get('external_candidate_rows', 0)}",
        f"- Active research ready: {state.get('active_research_ready', 0)}",
        f"- Needs price data: {state.get('needs_price_data', 0)}",
        f"- Liquidity blocked: {state.get('liquidity_blocked', 0)}",
        f"- Price source: {state.get('price_source', 'unknown')}",
        "",
    ]
    if not enriched.empty:
        show_cols = [
            "ticker", "theme", "chain_role", "theme_candidate_status",
            "option_research_side", "attention_score", "positive_catalysts",
            "negative_catalysts", "price_data_status", "last_close",
            "ret_5d_pct", "ret_20d_pct", "realized_vol_20d_pct",
            "liquidity_status", "trend_state", "status_reason", "top_headline",
        ]
        show_cols = [c for c in show_cols if c in enriched.columns]
        lines += [
            "## Top Theme Candidate Queue",
            enriched[show_cols].head(60).to_markdown(index=False),
            "",
        ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start = time.time()
    print("[step130] enriching theme candidates with price/liquidity checks")
    enriched, price_metrics, state = build_enrichment(args)
    enriched.to_csv(OUT_ENRICHED, index=False)
    price_metrics.to_csv(OUT_PRICE_METRICS, index=False)
    OUT_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    write_report(enriched, state)
    print(f"[step130] wrote {OUT_ENRICHED.name}: {len(enriched)} rows")
    print(f"[step130] wrote {OUT_PRICE_METRICS.name}: {len(price_metrics)} rows")
    print(
        f"[step130] status={state.get('status')} active={state.get('active_research_ready', 0)} "
        f"needs_price={state.get('needs_price_data', 0)} elapsed={time.time() - start:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
