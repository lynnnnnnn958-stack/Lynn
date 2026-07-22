#!/usr/bin/env python3
"""
Canyon v9 - Step 136: Historical Risk Evidence Collector
=========================================================

Research-only. No broker connection. No live orders.

Step132/133/134 can identify policy tickets that still need history. Step136
collects local proxy history for the main missing-history controls:

  - crisis correlation stress
  - drawdown control
  - liquidity crisis exit risk

This is intentionally labeled proxy history. It improves the review evidence,
but it is not a substitute for paid point-in-time market data, historical
constituents, true bid/ask, or broker execution records.

Outputs:
  risk_historical_evidence_summary.csv
  risk_historical_correlation_windows.csv
  risk_historical_drawdown_windows.csv
  risk_historical_liquidity_ticker_stress.csv
  risk_historical_policy_bridge.csv
  risk_historical_evidence_state.json
  risk_historical_evidence_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    TRADING_DAYS,
    clean_ticker,
    df_to_markdown,
    load_current_book,
    portfolio_return_series,
    read_csv_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_TICKETS = ROOT / "risk_policy_open_tickets.csv"
IN_CRISIS = ROOT / "crisis_correlation_stress.csv"
IN_LIQUIDITY = ROOT / "liquidity_crisis_simulation.csv"
IN_DRAWDOWN_STATE = ROOT / "drawdown_control_state.json"

OUT_SUMMARY = ROOT / "risk_historical_evidence_summary.csv"
OUT_CORR = ROOT / "risk_historical_correlation_windows.csv"
OUT_DD = ROOT / "risk_historical_drawdown_windows.csv"
OUT_LIQ = ROOT / "risk_historical_liquidity_ticker_stress.csv"
OUT_BRIDGE = ROOT / "risk_historical_policy_bridge.csv"
OUT_STATE = ROOT / "risk_historical_evidence_state.json"
OUT_REPORT = ROOT / "risk_historical_evidence_report.md"


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def pct(value: Any, digits: int = 2) -> str:
    x = safe_float(value)
    if not np.isfinite(x):
        return "NA"
    return f"{x * 100:.{digits}f}%"


def load_price_cache() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fname in ["sp500_price_cache.csv", "backtest_price_cache.csv", "regime_price_cache.csv"]:
        path = ROOT / fname
        df = read_csv_safe(path)
        if df.empty:
            continue
        first = df.columns[0]
        if str(first).lower() in {"date", "unnamed: 0", "index"}:
            df[first] = pd.to_datetime(df[first], errors="coerce")
            df = df.dropna(subset=[first]).set_index(first)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.columns = [clean_ticker(c) for c in df.columns]
        frames.append(df.sort_index())
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.combine_first(df)
        for col in df.columns:
            if col not in out.columns:
                out[col] = df[col]
    out = out.loc[:, ~pd.Index(out.columns).duplicated()]
    return out.sort_index()


def current_book_and_returns(max_lookback: int = 1260) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    book = load_current_book(prefer_filtered=True)
    if book.empty:
        return book, pd.DataFrame(), pd.Series(dtype=float)
    prices = load_price_cache()
    if prices.empty:
        return book, pd.DataFrame(), pd.Series(dtype=float)
    tickers = [clean_ticker(t) for t in book["ticker"].tolist()]
    available = [t for t in tickers if t in prices.columns]
    if not available:
        return book, pd.DataFrame(), pd.Series(dtype=float)
    px = prices[available].ffill().bfill().tail(max_lookback + 5)
    returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    sub = book[book["ticker"].isin(returns.columns)].copy()
    if sub.empty:
        return book, returns, pd.Series(dtype=float)
    weights = sub.set_index("ticker")["weight"].astype(float)
    weights = weights / max(float(weights.sum()), 1e-12)
    common = [t for t in weights.index if t in returns.columns]
    portfolio_returns = (returns[common] * weights.loc[common]).sum(axis=1).dropna()
    return sub, returns[common], portfolio_returns


def offdiag_values(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty or matrix.shape[0] < 2:
        return np.array([], dtype=float)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    vals = matrix.to_numpy(dtype=float)[mask]
    return vals[np.isfinite(vals)]


def portfolio_vol_from_corr(corr: pd.DataFrame, vols: pd.Series, weights: pd.Series) -> float:
    common = [c for c in corr.columns if c in vols.index and c in weights.index]
    if len(common) < 2:
        return np.nan
    c = corr.loc[common, common].astype(float).to_numpy()
    v = vols.loc[common].astype(float).to_numpy()
    w = weights.loc[common].astype(float).to_numpy()
    cov = np.outer(v, v) * c
    var = float(w @ cov @ w)
    if var <= 0 or not np.isfinite(var):
        return np.nan
    return float(np.sqrt(var * TRADING_DAYS))


def build_correlation_history(book: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if book.empty or returns.empty or returns.shape[1] < 3:
        return pd.DataFrame(), {
            "control": "Crisis-correlation volatility budget",
            "historical_status": "NO_DATA",
            "reason": "Need at least three tickers with local return history.",
        }

    weights = book.set_index("ticker")["weight"].astype(float)
    weights = weights / max(float(weights.sum()), 1e-12)
    window = 63
    min_obs = 50
    rows: list[dict[str, Any]] = []
    work = returns.dropna(how="all").tail(1260)
    for end in range(min_obs, len(work) + 1):
        sub = work.iloc[max(0, end - window):end].dropna(axis=1, thresh=min_obs // 2)
        common = [c for c in sub.columns if c in weights.index]
        if len(common) < 3:
            continue
        sub = sub[common].dropna(how="all")
        corr = sub.corr().replace([np.inf, -np.inf], np.nan)
        vals = offdiag_values(corr)
        if len(vals) < 3:
            continue
        vols = sub.std(ddof=1)
        base_vol = portfolio_vol_from_corr(corr, vols, weights)
        stress_corr = corr.copy()
        for i in stress_corr.index:
            for j in stress_corr.columns:
                if i != j:
                    stress_corr.loc[i, j] = max(float(stress_corr.loc[i, j]), 0.85)
        stress_vol = portfolio_vol_from_corr(stress_corr, vols, weights)
        ratio = stress_vol / base_vol if np.isfinite(stress_vol) and np.isfinite(base_vol) and base_vol > 0 else np.nan
        date = work.index[end - 1]
        rows.append({
            "window_end": date,
            "window_days": int(len(sub)),
            "ticker_count": int(len(common)),
            "avg_pair_corr": float(np.nanmean(vals)),
            "max_pair_corr": float(np.nanmax(vals)),
            "high_corr_pair_count": int(np.sum(vals >= 0.75) / 2),
            "base_annual_vol": base_vol,
            "crisis_floor_annual_vol": stress_vol,
            "stress_vol_ratio": ratio,
            "source_file": "sp500_price_cache.csv; backtest_price_cache.csv",
            "research_only": True,
        })
    hist = pd.DataFrame(rows)
    if hist.empty:
        return hist, {
            "control": "Crisis-correlation volatility budget",
            "historical_status": "NO_DATA",
            "reason": "Rolling correlation windows could not be built from local history.",
        }

    ratio = pd.to_numeric(hist["stress_vol_ratio"], errors="coerce").dropna()
    current_ratio = safe_float(read_csv_safe(IN_CRISIS).get("vol_increase_ratio", pd.Series([np.nan])).iloc[0] if IN_CRISIS.exists() else np.nan)
    if not np.isfinite(current_ratio):
        current_ratio = safe_float(ratio.iloc[-1])
    p80 = float(ratio.quantile(0.80)) if not ratio.empty else np.nan
    p95 = float(ratio.quantile(0.95)) if not ratio.empty else np.nan
    percentile = float((ratio <= current_ratio).mean() * 100.0) if not ratio.empty and np.isfinite(current_ratio) else np.nan
    status = "PROXY_HISTORY_USABLE" if len(ratio) >= 180 else "PROXY_HISTORY_THIN"
    summary = {
        "risk_area": "Correlation stress",
        "control": "Crisis-correlation volatility budget",
        "historical_status": status,
        "sample_n": int(len(ratio)),
        "lookback_start": str(pd.to_datetime(hist["window_end"].iloc[0]).date()),
        "lookback_end": str(pd.to_datetime(hist["window_end"].iloc[-1]).date()),
        "current_value": current_ratio,
        "historical_median": float(ratio.median()) if not ratio.empty else np.nan,
        "historical_p80": p80,
        "historical_p95": p95,
        "worst_observed": float(ratio.max()) if not ratio.empty else np.nan,
        "historical_percentile": percentile,
        "proposed_warning": p80,
        "proposed_hard": p95,
        "evidence_note": "Rolling 63d current-book correlation stress using local price history; proxy history, not a paid Barra/Axioma risk model.",
        "source_file": "risk_historical_correlation_windows.csv",
        "research_only": True,
    }
    return hist.sort_values("window_end").reset_index(drop=True), summary


def build_drawdown_history(portfolio_returns: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    if len(p) < 120:
        return pd.DataFrame(), {
            "control": "Drawdown budget",
            "historical_status": "NO_DATA",
            "reason": "Need at least 120 daily portfolio-return observations.",
        }
    nav = (1.0 + p).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    rows = pd.DataFrame({
        "date": drawdown.index,
        "proxy_nav": nav.values,
        "proxy_high_water_mark": peak.values,
        "drawdown": drawdown.values,
        "drawdown_loss": -drawdown.values,
        "daily_return": p.reindex(drawdown.index).values,
        "source_file": "sp500_price_cache.csv; backtest_price_cache.csv",
        "research_only": True,
    })
    loss = pd.to_numeric(rows["drawdown_loss"], errors="coerce").dropna()
    current_loss = safe_float(loss.iloc[-1])
    p80 = float(loss.quantile(0.80))
    p95 = float(loss.quantile(0.95))
    percentile = float((loss <= current_loss).mean() * 100.0) if np.isfinite(current_loss) else np.nan
    status = "PROXY_HISTORY_USABLE" if len(loss) >= 180 else "PROXY_HISTORY_THIN"
    summary = {
        "risk_area": "Drawdown control",
        "control": "Drawdown budget",
        "historical_status": status,
        "sample_n": int(len(loss)),
        "lookback_start": str(pd.to_datetime(rows["date"].iloc[0]).date()),
        "lookback_end": str(pd.to_datetime(rows["date"].iloc[-1]).date()),
        "current_value": current_loss,
        "historical_median": float(loss.median()),
        "historical_p80": p80,
        "historical_p95": p95,
        "worst_observed": float(loss.max()),
        "historical_percentile": percentile,
        "proposed_warning": p80,
        "proposed_hard": p95,
        "evidence_note": "Current-book proxy NAV drawdown from local price history. This is not the live paper-account NAV history.",
        "source_file": "risk_historical_drawdown_windows.csv",
        "research_only": True,
    }
    return rows.sort_values("date").reset_index(drop=True), summary


def liquidity_shock_from_days(days_to_exit: float, adv_stress_ratio: float) -> float:
    if not np.isfinite(days_to_exit):
        return -0.12
    if days_to_exit >= 5:
        base = -0.12
    elif days_to_exit >= 2:
        base = -0.08
    elif days_to_exit >= 0.5:
        base = -0.04
    else:
        base = -0.015
    if np.isfinite(adv_stress_ratio) and adv_stress_ratio < 0.35:
        base *= 1.5
    return float(max(base, -0.20))


def build_liquidity_history(book: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_csv_safe(ROOT / "desk_monitor_price_volume_cache.csv")
    if raw.empty or book.empty or not {"date", "ticker", "close", "volume"}.issubset(raw.columns):
        return pd.DataFrame(), {
            "control": "Liquidity crisis liquidation budget",
            "historical_status": "NO_DATA",
            "reason": "Need desk_monitor_price_volume_cache.csv with date, ticker, close, and volume.",
        }
    raw = raw.copy()
    raw["ticker"] = raw["ticker"].apply(clean_ticker)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw = raw.dropna(subset=["date", "ticker", "close", "volume"])
    raw["dollar_volume"] = raw["close"] * raw["volume"]
    weights = book.set_index("ticker")["weight"].astype(float)
    rows: list[dict[str, Any]] = []
    for ticker, weight in weights.items():
        sub = raw[raw["ticker"] == ticker].sort_values("date").copy()
        if sub.empty:
            continue
        sub["adv20"] = sub["dollar_volume"].rolling(20, min_periods=10).mean()
        adv = pd.to_numeric(sub["adv20"], errors="coerce").dropna()
        if adv.empty:
            continue
        current_adv = float(adv.iloc[-1])
        p05_adv = float(adv.quantile(0.05))
        p10_adv = float(adv.quantile(0.10))
        p25_adv = float(adv.quantile(0.25))
        notional = float(weight) * MODEL_ACCOUNT_VALUE
        stressed_capacity = max(p05_adv * 0.10, 1e-9)
        days_to_exit_p05 = notional / stressed_capacity
        current_days_to_exit = notional / max(current_adv * 0.10, 1e-9)
        stress_ratio = p05_adv / current_adv if current_adv > 0 else np.nan
        shock = liquidity_shock_from_days(days_to_exit_p05, stress_ratio)
        loss = abs(shock) * notional
        rows.append({
            "ticker": ticker,
            "weight": float(weight),
            "weight_pct": float(weight) * 100.0,
            "notional_model_account": notional,
            "history_days": int(len(sub)),
            "adv20_current": current_adv,
            "adv20_p05": p05_adv,
            "adv20_p10": p10_adv,
            "adv20_p25": p25_adv,
            "adv_stress_ratio_p05_to_current": stress_ratio,
            "current_days_to_exit_at_10pct_adv": current_days_to_exit,
            "historical_p05_days_to_exit_at_10pct_adv": days_to_exit_p05,
            "historical_liquidity_shock": shock,
            "historical_stress_loss_model_account": loss,
            "historical_stress_loss_pct_nav": loss / MODEL_ACCOUNT_VALUE,
            "history_start": str(pd.to_datetime(sub["date"].min()).date()),
            "history_end": str(pd.to_datetime(sub["date"].max()).date()),
            "source_file": "desk_monitor_price_volume_cache.csv",
            "research_only": True,
        })
    hist = pd.DataFrame(rows)
    if hist.empty:
        return hist, {
            "control": "Liquidity crisis liquidation budget",
            "historical_status": "NO_DATA",
            "reason": "No current-book tickers had usable historical volume rows.",
        }
    total_loss = float(hist["historical_stress_loss_model_account"].sum())
    sample_n = int(hist["history_days"].sum())
    min_days = int(hist["history_days"].min())
    status = "PROXY_HISTORY_USABLE" if min_days >= 120 and sample_n >= 1000 else "PROXY_HISTORY_THIN"
    summary = {
        "risk_area": "Liquidity stress",
        "control": "Liquidity crisis liquidation budget",
        "historical_status": status,
        "sample_n": sample_n,
        "lookback_start": str(hist["history_start"].min()),
        "lookback_end": str(hist["history_end"].max()),
        "current_value": total_loss / MODEL_ACCOUNT_VALUE,
        "historical_median": float(hist["historical_stress_loss_pct_nav"].median()),
        "historical_p80": float(hist["historical_stress_loss_pct_nav"].quantile(0.80)),
        "historical_p95": float(hist["historical_stress_loss_pct_nav"].quantile(0.95)),
        "worst_observed": float(hist["historical_stress_loss_pct_nav"].max()),
        "historical_percentile": np.nan,
        "proposed_warning": min(0.03, max(0.01, total_loss / MODEL_ACCOUNT_VALUE * 1.25)),
        "proposed_hard": min(0.05, max(0.015, total_loss / MODEL_ACCOUNT_VALUE * 1.75)),
        "evidence_note": "Historical 20d dollar-volume stress using desk-monitor OHLCV cache. It is a proxy for liquidity crisis capacity, not true bid/ask depth.",
        "source_file": "risk_historical_liquidity_ticker_stress.csv",
        "research_only": True,
    }
    return hist.sort_values("historical_stress_loss_model_account", ascending=False).reset_index(drop=True), summary


def build_policy_bridge(tickets: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    summary_by_control = {}
    if not summary.empty and "control" in summary.columns:
        summary_by_control = {str(r["control"]): r for _, r in summary.iterrows()}
    for _, ticket in tickets.iterrows():
        control = str(ticket.get("control", ""))
        evidence = summary_by_control.get(control)
        if evidence is None:
            continue
        hist_status = str(evidence.get("historical_status", ""))
        if "USABLE" in hist_status:
            action = "Use proxy history in policy review; keep research-only caveat."
        elif "THIN" in hist_status:
            action = "Use as first-pass evidence, but keep missing-history warning."
        else:
            action = "Do not rely on this control until more history is collected."
        rows.append({
            "ticket_id": ticket.get("ticket_id", ""),
            "review_priority": ticket.get("review_priority", ""),
            "risk_area": ticket.get("risk_area", ""),
            "control": control,
            "previous_calibration_status": ticket.get("calibration_status", ""),
            "previous_calibration_mode": ticket.get("calibration_mode", ""),
            "historical_status": hist_status,
            "historical_sample_n": evidence.get("sample_n", np.nan),
            "current_value": evidence.get("current_value", np.nan),
            "historical_p95": evidence.get("historical_p95", np.nan),
            "worst_observed": evidence.get("worst_observed", np.nan),
            "bridge_action": action,
            "source_file": evidence.get("source_file", ""),
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_state(summary: pd.DataFrame, bridge: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    statuses = summary["historical_status"].astype(str).str.upper()
    usable = int(statuses.str.contains("USABLE").sum())
    thin = int(statuses.str.contains("THIN").sum())
    missing = int(statuses.isin(["NO_DATA", "MISSING"]).sum())
    if missing:
        overall = "HISTORY_GAPS"
    elif thin:
        overall = "PROXY_HISTORY_PARTIAL"
    else:
        overall = "PROXY_HISTORY_READY"
    return {
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Collect local proxy history for missing-history risk controls. This supports review but does not auto-approve policy changes.",
        "controls_checked": int(len(summary)),
        "usable_proxy_controls": usable,
        "thin_proxy_controls": thin,
        "missing_controls": missing,
        "policy_tickets_backfilled": int(len(bridge)),
        "outputs": {
            "summary": OUT_SUMMARY.name,
            "correlation_windows": OUT_CORR.name,
            "drawdown_windows": OUT_DD.name,
            "liquidity_ticker_stress": OUT_LIQ.name,
            "policy_bridge": OUT_BRIDGE.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    tickets = read_csv_safe(IN_TICKETS)
    book, returns, p_returns = current_book_and_returns(max_lookback=1260)

    corr_hist, corr_summary = build_correlation_history(book, returns)
    dd_hist, dd_summary = build_drawdown_history(p_returns)
    liq_hist, liq_summary = build_liquidity_history(book)

    summary_rows = [corr_summary, dd_summary, liq_summary]
    summary = pd.DataFrame(summary_rows)
    bridge = build_policy_bridge(tickets, summary)
    state = build_state(summary, bridge)

    summary.to_csv(OUT_SUMMARY, index=False)
    corr_hist.to_csv(OUT_CORR, index=False)
    dd_hist.to_csv(OUT_DD, index=False)
    liq_hist.to_csv(OUT_LIQ, index=False)
    bridge.to_csv(OUT_BRIDGE, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Controls checked: {state.get('controls_checked', 0)}",
        f"- Usable proxy controls: {state.get('usable_proxy_controls', 0)}",
        f"- Thin proxy controls: {state.get('thin_proxy_controls', 0)}",
        f"- Missing controls: {state.get('missing_controls', 0)}",
        f"- Policy tickets backfilled: {state.get('policy_tickets_backfilled', 0)}",
        "",
        "## Historical Evidence Summary",
        "",
        df_to_markdown(summary, max_rows=20),
        "",
        "## Policy Bridge",
        "",
        df_to_markdown(bridge, max_rows=20),
        "",
        "## Worst Correlation Windows",
        "",
        df_to_markdown(corr_hist.sort_values("stress_vol_ratio", ascending=False).head(15) if not corr_hist.empty else corr_hist, max_rows=15),
        "",
        "## Worst Drawdown Dates",
        "",
        df_to_markdown(dd_hist.sort_values("drawdown_loss", ascending=False).head(15) if not dd_hist.empty else dd_hist, max_rows=15),
        "",
        "## Worst Liquidity Names",
        "",
        df_to_markdown(liq_hist.head(20), max_rows=20),
        "",
        "## Product Truth",
        "",
        "This is proxy history from local files. It helps the review desk, but it is not point-in-time institutional data and does not permit live orders.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 136 - Historical Risk Evidence", sections)

    print(f"wrote {OUT_SUMMARY.name} rows={len(summary)}")
    print(f"wrote {OUT_BRIDGE.name} rows={len(bridge)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
