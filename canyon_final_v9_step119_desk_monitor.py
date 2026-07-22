#!/usr/bin/env python3
"""
Canyon v9 Step 119 - Desk Monitor.

Research-only. No broker connection. No live orders.

This step watches for desk-level changes that should get attention before any
new paper idea is considered:
  price break, volume spike, volatility regime shift, spread widening,
  correlation break, news shock, earnings surprise, and risk limit breach.

Outputs:
  desk_monitor_events.csv
  desk_monitor_ticker_state.csv
  desk_monitor_summary.json
  desk_monitor_report.md
  desk_monitor_price_volume_cache.csv
"""
from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    get_returns,
    load_current_book,
    load_price_cache,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_EVENTS = ROOT / "desk_monitor_events.csv"
OUT_STATE = ROOT / "desk_monitor_ticker_state.csv"
OUT_SUMMARY = ROOT / "desk_monitor_summary.json"
OUT_REPORT = ROOT / "desk_monitor_report.md"
OUT_PRICE_VOLUME = ROOT / "desk_monitor_price_volume_cache.csv"

SEVERITY_RANK = {
    "OK": 0,
    "INFO": 1,
    "DATA_GAP": 1,
    "WATCH": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}

NEWS_NEGATIVE = {
    "downgrade", "cuts guidance", "cut guidance", "guidance cut", "misses",
    "missed", "miss", "lawsuit", "sued", "investigation", "probe", "sec",
    "doj", "ftc", "fraud", "accounting", "short seller", "short-seller",
    "breach", "hack", "recall", "bankruptcy", "layoffs", "layoff",
    "resigns", "slumps", "plunges", "falls", "warning",
}
NEWS_POSITIVE = {
    "upgrade", "raises guidance", "raised guidance", "beat", "beats",
    "buyback", "repurchase", "approval", "approved", "partnership",
    "acquisition", "acquires", "record revenue", "surges", "jumps",
}


def keyword_hits(text: str, keywords: set[str]) -> list[str]:
    """Boundary-aware headline matching to avoid false hits like 'miss' in 'Mission'."""
    low = str(text).lower()
    hits = []
    for keyword in keywords:
        pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
        if re.search(pattern, low):
            hits.append(keyword)
    return sorted(hits)


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def severity_max(labels: list[str]) -> str:
    if not labels:
        return "OK"
    return max(labels, key=lambda x: SEVERITY_RANK.get(str(x), 0))


def numeric(value: Any, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else default


def source_layer_for(monitor: str) -> str:
    mapping = {
        "PRICE_BREAK": "L6 Price / Technical",
        "VOLUME_SPIKE": "L6 Price / Technical",
        "VOLATILITY_REGIME_SHIFT": "L6 Price / Technical",
        "SPREAD_WIDENING": "L9 Execution / Liquidity",
        "CORRELATION_BREAK": "L8 Portfolio Risk",
        "NEWS_SHOCK": "L5 Event / News",
        "EARNINGS_SURPRISE": "L5 Event / Earnings",
        "RISK_LIMIT_BREACH": "L8 Portfolio Risk",
        "DATA_GAP": "L1 Data Integrity",
    }
    return mapping.get(str(monitor).upper(), "Cross-layer Monitor")


def source_provider_for(source_file: str) -> str:
    source = str(source_file)
    if "yfinance" in source.lower() or "desk_monitor_price_volume_cache" in source:
        return "Yahoo Finance via yfinance, cached locally"
    if source in {"sp500_price_cache.csv/backtest_price_cache.csv", "local price cache fallback"}:
        return "Local historical price cache"
    if source == "stock_news.json":
        return "Step99 news cache"
    if source == "earnings_surprise_scores.csv":
        return "Step81 earnings surprise output"
    if source == "institutional_risk_budget_summary.csv":
        return "Step118 institutional risk budget"
    if source == "final_risk_gate.csv":
        return "Step118 final risk gate"
    return source or "Unknown local source"


def event(
    monitor: str,
    ticker: str,
    severity: str,
    title: str,
    detail: str,
    action: str,
    source_file: str,
    metric_1_name: str = "",
    metric_1_value: float | str | None = None,
    metric_2_name: str = "",
    metric_2_value: float | str | None = None,
) -> dict[str, Any]:
    return {
        "date": today_str(),
        "run_time": now_iso(),
        "monitor": monitor,
        "ticker": clean_ticker(ticker),
        "severity": severity,
        "title": title,
        "detail": detail,
        "action": action,
        "metric_1_name": metric_1_name,
        "metric_1_value": metric_1_value,
        "metric_2_name": metric_2_name,
        "metric_2_value": metric_2_value,
        "source_layer": source_layer_for(monitor),
        "source_provider": source_provider_for(source_file),
        "source_file": source_file,
        "research_only": True,
        "no_broker_connection": True,
    }


def load_universe(limit: int = 40) -> pd.DataFrame:
    book = load_current_book(prefer_filtered=True)
    if book.empty:
        picks = read_csv_safe(ROOT / "daily_picks_filtered.csv")
        if picks.empty:
            picks = read_csv_safe(ROOT / "daily_picks.csv")
        if picks.empty or "ticker" not in picks.columns:
            return pd.DataFrame(columns=["ticker", "weight", "alpha_score", "sector", "action"])
        out = picks.copy()
        out["ticker"] = out["ticker"].apply(clean_ticker)
        out["weight"] = pd.to_numeric(out.get("weight_pct", 0), errors="coerce").fillna(0.0) / 100.0
        return out.head(limit)
    return book.head(limit)


def fetch_price_volume(tickers: list[str]) -> pd.DataFrame:
    tickers = sorted({clean_ticker(t) for t in tickers if clean_ticker(t)})
    if not tickers:
        return pd.DataFrame()
    try:
        import yfinance as yf
        data = yf.download(
            tickers,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        data = pd.DataFrame()

    rows = []
    if data is not None and not data.empty:
        multi = isinstance(data.columns, pd.MultiIndex)
        for ticker in tickers:
            try:
                sub = data[ticker].copy() if multi else data.copy()
            except Exception:
                continue
            if sub.empty:
                continue
            sub = sub.reset_index()
            date_col = "Date" if "Date" in sub.columns else sub.columns[0]
            for _, row in sub.iterrows():
                rows.append({
                    "date": pd.to_datetime(row.get(date_col), errors="coerce"),
                    "ticker": ticker,
                    "open": numeric(row.get("Open")),
                    "high": numeric(row.get("High")),
                    "low": numeric(row.get("Low")),
                    "close": numeric(row.get("Close")),
                    "adj_close": numeric(row.get("Adj Close", row.get("Close"))),
                    "volume": numeric(row.get("Volume")),
                    "source_file": "yfinance daily OHLCV",
                })

    out = pd.DataFrame(rows)
    if out.empty:
        prices = load_price_cache()
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            s = prices[ticker].dropna().tail(140)
            for dt, px in s.items():
                out = pd.concat([out, pd.DataFrame([{
                    "date": pd.to_datetime(dt, errors="coerce"),
                    "ticker": ticker,
                    "open": np.nan,
                    "high": np.nan,
                    "low": np.nan,
                    "close": float(px),
                    "adj_close": float(px),
                    "volume": np.nan,
                    "source_file": "local price cache fallback",
                }])], ignore_index=True)
    if not out.empty:
        out = out.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])
        out.to_csv(OUT_PRICE_VOLUME, index=False)
    return out


def fetch_spread_snapshot(tickers: list[str]) -> pd.DataFrame:
    tickers = sorted({clean_ticker(t) for t in tickers if clean_ticker(t)})
    if not tickers:
        return pd.DataFrame()

    def one(ticker: str) -> dict[str, Any]:
        row = {
            "ticker": ticker,
            "bid": np.nan,
            "ask": np.nan,
            "mid": np.nan,
            "spread_bps": np.nan,
            "spread_status": "DATA_GAP",
            "source_file": "yfinance fast_info",
        }
        try:
            import yfinance as yf
            info = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
            bid = numeric(info.get("bid"))
            ask = numeric(info.get("ask"))
            last = numeric(info.get("last_price", info.get("lastPrice")))
            if np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0:
                mid = (bid + ask) / 2.0
                spread_bps = (ask - bid) / mid * 10000.0
                row.update({"bid": bid, "ask": ask, "mid": mid, "spread_bps": spread_bps})
                if spread_bps >= 100:
                    row["spread_status"] = "CRITICAL"
                elif spread_bps >= 50:
                    row["spread_status"] = "WARNING"
                elif spread_bps >= 20:
                    row["spread_status"] = "WATCH"
                else:
                    row["spread_status"] = "OK"
            elif np.isfinite(last):
                row.update({"mid": last, "spread_status": "DATA_GAP"})
        except Exception:
            pass
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(one, t): t for t in tickers}
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows)


def monitor_price_volume_vol(universe: pd.DataFrame, pv: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    events = []
    state_rows = []
    if pv.empty:
        for ticker in universe["ticker"].apply(clean_ticker).tolist():
            events.append(event(
                "DATA_GAP", ticker, "DATA_GAP",
                f"{ticker}: no price/volume data",
                "Could not build daily OHLCV monitor state.",
                "Fix price/volume source before acting on this ticker.",
                "desk_monitor_price_volume_cache.csv",
            ))
        return events, pd.DataFrame()

    for ticker, grp in pv.groupby("ticker"):
        g = grp.dropna(subset=["close"]).sort_values("date").tail(90).copy()
        if len(g) < 30:
            state_rows.append({"ticker": ticker, "max_severity": "DATA_GAP", "price_break_state": "DATA_GAP"})
            continue
        latest = g.iloc[-1]
        prior = g.iloc[:-1]
        close = numeric(latest.get("close"))
        prev_close = numeric(prior.iloc[-1].get("close")) if not prior.empty else np.nan
        daily_ret = close / prev_close - 1.0 if np.isfinite(close) and np.isfinite(prev_close) and prev_close > 0 else np.nan

        high_series = prior["high"] if "high" in prior.columns and prior["high"].notna().any() else prior["close"]
        low_series = prior["low"] if "low" in prior.columns and prior["low"].notna().any() else prior["close"]
        high_20 = numeric(high_series.tail(20).max())
        low_20 = numeric(low_series.tail(20).min())

        price_state = "OK"
        if np.isfinite(close) and np.isfinite(low_20) and close < low_20:
            price_state = "CRITICAL"
            events.append(event(
                "PRICE_BREAK", ticker, "CRITICAL",
                f"{ticker}: broke below 20-day support",
                f"Latest close {close:.2f} is below the prior 20-day low {low_20:.2f}.",
                "Do not add. Review stop/risk gate before any paper action.",
                "desk_monitor_price_volume_cache.csv",
                "close", close, "prior_20d_low", low_20,
            ))
        elif np.isfinite(close) and np.isfinite(high_20) and close > high_20:
            price_state = "WARNING"
            events.append(event(
                "PRICE_BREAK", ticker, "WARNING",
                f"{ticker}: broke above 20-day resistance",
                f"Latest close {close:.2f} is above the prior 20-day high {high_20:.2f}.",
                "Watch for confirmation. This is not an automatic buy.",
                "desk_monitor_price_volume_cache.csv",
                "close", close, "prior_20d_high", high_20,
            ))

        vol_latest = numeric(latest.get("volume"))
        vol_med = numeric(prior["volume"].dropna().tail(20).median()) if "volume" in prior.columns else np.nan
        vol_ratio = vol_latest / vol_med if np.isfinite(vol_latest) and np.isfinite(vol_med) and vol_med > 0 else np.nan
        volume_state = "OK"
        if np.isfinite(vol_ratio):
            if vol_ratio >= 4.0 and np.isfinite(daily_ret) and daily_ret < 0:
                volume_state = "CRITICAL"
            elif vol_ratio >= 2.5:
                volume_state = "WARNING"
            if volume_state != "OK":
                events.append(event(
                    "VOLUME_SPIKE", ticker, volume_state,
                    f"{ticker}: volume spike {vol_ratio:.1f}x normal",
                    f"Latest volume {vol_latest:,.0f}; 20-day median {vol_med:,.0f}; daily return {daily_ret:.2%}.",
                    "Check news, earnings, and price break context before acting.",
                    "desk_monitor_price_volume_cache.csv",
                    "volume_ratio", vol_ratio, "daily_return", daily_ret,
                ))

        rets = g["close"].pct_change(fill_method=None).dropna()
        vol20 = float(rets.tail(20).std(ddof=1) * math.sqrt(252)) if len(rets.tail(20)) >= 15 else np.nan
        vol60 = float(rets.tail(60).std(ddof=1) * math.sqrt(252)) if len(rets.tail(60)) >= 40 else np.nan
        vol_ratio_regime = vol20 / vol60 if np.isfinite(vol20) and np.isfinite(vol60) and vol60 > 0 else np.nan
        vol_state = "OK"
        if np.isfinite(vol_ratio_regime):
            if vol_ratio_regime >= 2.25:
                vol_state = "CRITICAL"
            elif vol_ratio_regime >= 1.50:
                vol_state = "WARNING"
            if vol_state != "OK":
                events.append(event(
                    "VOLATILITY_REGIME_SHIFT", ticker, vol_state,
                    f"{ticker}: realized volatility regime changed",
                    f"20-day vol {vol20:.1%}; 60-day vol {vol60:.1%}; ratio {vol_ratio_regime:.2f}x.",
                    "Reduce position confidence until volatility stabilizes.",
                    "desk_monitor_price_volume_cache.csv",
                    "vol20", vol20, "vol20_to_vol60", vol_ratio_regime,
                ))

        state_rows.append({
            "ticker": ticker,
            "latest_close": close,
            "daily_return": daily_ret,
            "prior_20d_high": high_20,
            "prior_20d_low": low_20,
            "price_break_state": price_state,
            "volume_ratio": vol_ratio,
            "volume_spike_state": volume_state,
            "realized_vol_20d": vol20,
            "realized_vol_60d": vol60,
            "vol20_to_vol60": vol_ratio_regime,
            "volatility_regime_state": vol_state,
            "max_severity": severity_max([price_state, volume_state, vol_state]),
        })
    return events, pd.DataFrame(state_rows)


def monitor_spread(spread: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    events = []
    if spread.empty:
        return events, pd.DataFrame()
    for _, row in spread.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        status = str(row.get("spread_status", "DATA_GAP"))
        bps = numeric(row.get("spread_bps"))
        if status in {"WARNING", "CRITICAL"}:
            events.append(event(
                "SPREAD_WIDENING", ticker, status,
                f"{ticker}: quoted spread widened",
                f"Current bid/ask spread is {bps:.1f} bps from yfinance fast_info.",
                "Manual liquidity check required. Do not assume close-price fills.",
                "yfinance fast_info",
                "spread_bps", bps,
            ))
    return events, spread


def monitor_correlation_break(universe: pd.DataFrame) -> list[dict[str, Any]]:
    events = []
    tickers = universe["ticker"].apply(clean_ticker).tolist() if not universe.empty else []
    rets = get_returns(tickers, lookback=90)
    if rets.empty or rets.shape[1] < 3:
        events.append(event(
            "CORRELATION_BREAK", "", "DATA_GAP",
            "Correlation monitor has insufficient data",
            "Need at least three holdings with return history.",
            "Fix price cache before relying on correlation monitor.",
            "sp500_price_cache.csv/backtest_price_cache.csv",
        ))
        return events
    corr20 = rets.tail(20).corr()
    corr60 = rets.tail(60).corr()

    def avg_abs_corr(corr: pd.DataFrame) -> float:
        vals = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                v = numeric(corr.loc[a, b])
                if np.isfinite(v):
                    vals.append(abs(v))
        return float(np.mean(vals)) if vals else np.nan

    avg20 = avg_abs_corr(corr20)
    avg60 = avg_abs_corr(corr60)
    max_pair = ("", "", np.nan)
    for i, a in enumerate(corr20.columns):
        for b in corr20.columns[i + 1:]:
            v = numeric(corr20.loc[a, b])
            if np.isfinite(v) and (not np.isfinite(max_pair[2]) or abs(v) > abs(max_pair[2])):
                max_pair = (a, b, v)

    status = "OK"
    if np.isfinite(avg20) and np.isfinite(avg60):
        jump = avg20 - avg60
        if avg20 >= 0.70 or jump >= 0.25:
            status = "CRITICAL"
        elif avg20 >= 0.55 or jump >= 0.15:
            status = "WARNING"
        if status != "OK":
            events.append(event(
                "CORRELATION_BREAK", "", status,
                "Holding correlations are rising",
                f"Average absolute correlation: 20d={avg20:.2f}, 60d={avg60:.2f}, change={jump:+.2f}. Max pair {max_pair[0]}/{max_pair[1]}={max_pair[2]:.2f}.",
                "Check whether diversification still works; reduce duplicate exposure if needed.",
                "sp500_price_cache.csv/backtest_price_cache.csv",
                "avg_abs_corr_20d", avg20, "avg_abs_corr_change", jump,
            ))
    return events


def parse_news_timestamp(item: dict[str, Any]) -> pd.Timestamp | None:
    ts = item.get("published_ts", 0)
    try:
        ts_int = int(ts)
        if ts_int > 0:
            return pd.Timestamp(datetime.fromtimestamp(ts_int, tz=timezone.utc)).tz_localize(None)
    except Exception:
        pass
    published = str(item.get("published", "")).strip()
    if not published or published == "1970-01-01":
        return None
    dt = pd.to_datetime(published, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).tz_localize(None) if getattr(dt, "tzinfo", None) else pd.Timestamp(dt)


def monitor_news_shock(universe: pd.DataFrame) -> list[dict[str, Any]]:
    events = []
    news = read_json_safe(ROOT / "stock_news.json", {})
    news_map = news.get("news", {}) if isinstance(news, dict) else {}
    if not news_map:
        return events
    now = pd.Timestamp(datetime.now())
    tickers = set(universe["ticker"].apply(clean_ticker).tolist()) if not universe.empty else set(news_map)
    for ticker in tickers:
        items = news_map.get(ticker, [])
        for item in items[:8]:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            dt = parse_news_timestamp(item)
            if dt is None or (now - dt).days > 3:
                continue
            low = title.lower()
            neg_hits = keyword_hits(low, NEWS_NEGATIVE)
            pos_hits = keyword_hits(low, NEWS_POSITIVE)
            if neg_hits:
                events.append(event(
                    "NEWS_SHOCK", ticker, "WARNING",
                    f"{ticker}: negative news shock",
                    f"{title} | keywords: {', '.join(sorted(neg_hits)[:4])}",
                    "Open the news link and confirm before any new paper action.",
                    "stock_news.json",
                    "published", str(dt.date()), "publisher", item.get("publisher", ""),
                ))
                break
            if pos_hits:
                events.append(event(
                    "NEWS_SHOCK", ticker, "INFO",
                    f"{ticker}: positive news catalyst",
                    f"{title} | keywords: {', '.join(sorted(pos_hits)[:4])}",
                    "Use as context only; do not upgrade without price/risk confirmation.",
                    "stock_news.json",
                    "published", str(dt.date()), "publisher", item.get("publisher", ""),
                ))
                break
    return events


def monitor_earnings_surprise(universe: pd.DataFrame) -> list[dict[str, Any]]:
    events = []
    surprise = read_csv_safe(ROOT / "earnings_surprise_scores.csv")
    if surprise.empty or "ticker" not in surprise.columns:
        return events
    surprise = surprise.copy()
    surprise["ticker"] = surprise["ticker"].apply(clean_ticker)
    tickers = set(universe["ticker"].apply(clean_ticker).tolist()) if not universe.empty else set(surprise["ticker"])
    sub = surprise[surprise["ticker"].isin(tickers)].copy()
    for col in ["surprise_pct", "days_since", "rank_sue"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    recent = sub[sub.get("days_since", 999).fillna(999) <= 5].copy() if "days_since" in sub.columns else pd.DataFrame()
    if recent.empty:
        return events
    for _, row in recent.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        pct = numeric(row.get("surprise_pct"))
        signal = str(row.get("signal", "")).upper()
        if not np.isfinite(pct):
            continue
        abs_pct = abs(pct)
        if pct <= -5 or signal == "MISS":
            severity = "CRITICAL" if abs_pct >= 10 else "WARNING"
            title = f"{ticker}: earnings miss / negative surprise"
            action = "Do not add. Review gap risk and thesis."
        elif pct >= 5 or signal == "BEAT":
            severity = "INFO" if abs_pct < 10 else "WARNING"
            title = f"{ticker}: earnings beat / positive surprise"
            action = "Watch post-earnings drift; risk gate still controls sizing."
        else:
            continue
        events.append(event(
            "EARNINGS_SURPRISE", ticker, severity,
            title,
            f"Surprise {pct:+.2f}%, days since earnings {row.get('days_since', 'N/A')}, signal {signal}.",
            action,
            "earnings_surprise_scores.csv",
            "surprise_pct", pct, "days_since", row.get("days_since", np.nan),
        ))
    return events


def monitor_risk_limit_breach() -> list[dict[str, Any]]:
    events = []
    budget = read_csv_safe(ROOT / "institutional_risk_budget_summary.csv")
    if not budget.empty:
        for _, row in budget.iterrows():
            status = str(row.get("status", "")).upper()
            if status in {"SIZE_DOWN", "REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
                sev = "CRITICAL" if status in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"} else "WARNING"
                events.append(event(
                    "RISK_LIMIT_BREACH", "", sev,
                    f"Risk budget breach: {row.get('budget_item')}",
                    f"Status {status}; current={row.get('current_value')}; limit={row.get('limit_value')}; used={row.get('used_pct')}.",
                    str(row.get("action_if_breached", "Review risk budget.")),
                    "institutional_risk_budget_summary.csv",
                    "used_pct", row.get("used_pct"), "scope", row.get("scope", ""),
                ))

    gate = read_csv_safe(ROOT / "final_risk_gate.csv")
    if not gate.empty and "final_risk_action" in gate.columns:
        flagged = gate[gate["final_risk_action"].astype(str).str.upper().isin(["REDUCE_ONLY", "BLOCK_NEW", "BLOCKED", "SIZE_DOWN"])].copy()
        action_rank = {"REDUCE_ONLY": 4, "BLOCK_NEW": 4, "BLOCKED": 4, "SIZE_DOWN": 3}
        flagged["_rank"] = flagged["final_risk_action"].astype(str).str.upper().map(action_rank).fillna(0)
        flagged = flagged.sort_values(["_rank", "current_weight_pct"], ascending=[False, False]).head(12)
        for _, row in flagged.iterrows():
            action = str(row.get("final_risk_action", "")).upper()
            sev = "CRITICAL" if action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"} else "WARNING"
            ticker = clean_ticker(row.get("ticker"))
            events.append(event(
                "RISK_LIMIT_BREACH", ticker, sev,
                f"{ticker}: final risk gate says {action}",
                str(row.get("reason_stack", "")),
                f"Use recommended risk weight {numeric(row.get('recommended_risk_weight_pct')):.2f}% for research sizing.",
                "final_risk_gate.csv",
                "current_weight_pct", row.get("current_weight_pct"), "recommended_weight_pct", row.get("recommended_risk_weight_pct"),
            ))
    return events


def build_state(universe: pd.DataFrame, states: list[pd.DataFrame], spread_state: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    base = universe.copy()
    if "ticker" not in base.columns:
        return pd.DataFrame()
    base["ticker"] = base["ticker"].apply(clean_ticker)
    keep = [c for c in ["ticker", "weight", "alpha_score", "action", "sector", "source_file"] if c in base.columns]
    out = base[keep].drop_duplicates("ticker").copy()
    for st in states:
        if not st.empty and "ticker" in st.columns:
            out = out.merge(st, on="ticker", how="left", suffixes=("", "_state"))
    if not spread_state.empty and "ticker" in spread_state.columns:
        cols = [c for c in ["ticker", "spread_bps", "spread_status", "bid", "ask"] if c in spread_state.columns]
        out = out.merge(spread_state[cols], on="ticker", how="left")

    per_ticker = {}
    for ev in events:
        ticker = clean_ticker(ev.get("ticker", ""))
        if not ticker:
            continue
        per_ticker.setdefault(ticker, []).append(str(ev.get("severity", "INFO")))
    out["max_monitor_severity"] = [
        severity_max(per_ticker.get(t, [])) for t in out["ticker"].astype(str)
    ]
    out["event_count"] = [len(per_ticker.get(t, [])) for t in out["ticker"].astype(str)]
    return out


def write_outputs(events: list[dict[str, Any]], state: pd.DataFrame) -> None:
    ev_df = pd.DataFrame(events)
    if ev_df.empty:
        ev_df = pd.DataFrame(columns=[
            "date", "run_time", "monitor", "ticker", "severity", "title",
            "detail", "action", "metric_1_name", "metric_1_value",
            "metric_2_name", "metric_2_value", "source_layer",
            "source_provider", "source_file",
            "research_only", "no_broker_connection",
        ])
    ev_df["_rank"] = ev_df["severity"].map(SEVERITY_RANK).fillna(0)
    ev_df = ev_df.sort_values(["_rank", "monitor", "ticker"], ascending=[False, True, True]).drop(columns=["_rank"])
    ev_df.to_csv(OUT_EVENTS, index=False)
    state.to_csv(OUT_STATE, index=False)

    summary = {
        "date": today_str(),
        "run_time": now_iso(),
        "total_events": int(len(ev_df)),
        "critical_count": int((ev_df["severity"] == "CRITICAL").sum()) if "severity" in ev_df.columns else 0,
        "warning_count": int((ev_df["severity"] == "WARNING").sum()) if "severity" in ev_df.columns else 0,
        "info_count": int((ev_df["severity"] == "INFO").sum()) if "severity" in ev_df.columns else 0,
        "data_gap_count": int((ev_df["severity"] == "DATA_GAP").sum()) if "severity" in ev_df.columns else 0,
        "monitor_counts": ev_df["monitor"].value_counts().to_dict() if "monitor" in ev_df.columns else {},
        "logic": "Desk monitor only. It warns and explains. It does not place orders or upgrade tickers.",
        "research_only": True,
        "no_broker_connection": True,
    }
    write_json(OUT_SUMMARY, summary)

    sections = [
        "## Summary",
        "",
        f"- Total events: {summary['total_events']}",
        f"- Critical: {summary['critical_count']}",
        f"- Warning: {summary['warning_count']}",
        f"- Info: {summary['info_count']}",
        f"- Data gaps: {summary['data_gap_count']}",
        "",
        "## Logic",
        "",
        "- Monitors watch for changes; they do not create buy signals.",
        "- Risk limit breach comes from Step118 and can only reduce or block.",
        "- Spread widening uses yfinance bid/ask snapshots when available; missing quotes are not treated as a risk breach.",
        "",
        "## Events",
        "",
        df_to_markdown(ev_df, max_rows=80),
        "",
        "## Ticker State",
        "",
        df_to_markdown(state, max_rows=80),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 119 - Desk Monitor", sections)


def main() -> None:
    universe = load_universe()
    tickers = universe["ticker"].apply(clean_ticker).tolist() if not universe.empty else []
    pv = fetch_price_volume(tickers)
    price_events, price_state = monitor_price_volume_vol(universe, pv)
    spread_events, spread_state = monitor_spread(fetch_spread_snapshot(tickers[:30]))
    events = []
    events.extend(price_events)
    events.extend(spread_events)
    events.extend(monitor_correlation_break(universe))
    events.extend(monitor_news_shock(universe))
    events.extend(monitor_earnings_surprise(universe))
    events.extend(monitor_risk_limit_breach())
    state = build_state(universe, [price_state], spread_state, events)
    write_outputs(events, state)

    summary = read_json_safe(OUT_SUMMARY, {})
    print(f"[step119] wrote {OUT_EVENTS.name}: {summary.get('total_events', 0)} events")
    print(f"[step119] wrote {OUT_STATE.name}: {len(state)} rows")
    print(f"[step119] wrote {OUT_SUMMARY.name}")
    print(f"[step119] wrote {OUT_REPORT.name}")
    print(
        "[step119] counts "
        f"critical={summary.get('critical_count', 0)} "
        f"warning={summary.get('warning_count', 0)} "
        f"info={summary.get('info_count', 0)} "
        f"data_gap={summary.get('data_gap_count', 0)}"
    )


if __name__ == "__main__":
    main()
