#!/usr/bin/env python3
"""
Canyon v9 Step 188 - Sharpe 4 Risk-Book Intake.

Research-only. No broker connection. No live orders.

Step187 found outside candidates, but they are not tradeable. Step188 turns
the best outside research names into a risk-book intake queue. A name can only
move forward after it has:

  - single-name VaR / CVaR
  - liquidity and TCA proxy
  - earnings / event proof
  - correlation and factor-crowding read
  - short / medium / long route
  - option route that is explicitly blocked until risk and evidence clear

Outputs:
  sharpe4_risk_book_intake_state.json
  sharpe4_risk_book_candidate_cards.csv
  sharpe4_risk_book_var_liquidity.csv
  sharpe4_risk_book_event_route.csv
  sharpe4_risk_book_correlation_proxy.csv
  sharpe4_risk_book_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    ROOT,
    annualized_vol,
    beta_to_factor,
    clean_ticker,
    df_to_markdown,
    get_returns,
    read_csv_safe,
    read_json_safe,
    today_str,
    var_cvar,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "sharpe4_risk_book_intake_state.json"
OUT_CARDS = ROOT / "sharpe4_risk_book_candidate_cards.csv"
OUT_VAR_LIQ = ROOT / "sharpe4_risk_book_var_liquidity.csv"
OUT_EVENT_ROUTE = ROOT / "sharpe4_risk_book_event_route.csv"
OUT_CORR = ROOT / "sharpe4_risk_book_correlation_proxy.csv"
OUT_REPORT = ROOT / "sharpe4_risk_book_report.md"

FACTOR_PROXIES = ["SPY", "QQQ", "XLK", "SMH", "SOXX", "TLT", "UUP", "XLE"]
MAX_INTAKE_ROWS = 24


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def one_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker", sort_col: str | None = None) -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""].copy()
    if sort_col and sort_col in work.columns:
        work["_sort"] = pd.to_numeric(work[sort_col], errors="coerce").fillna(-1e9)
        work = work.sort_values([ticker_col, "_sort"], ascending=[True, False])
    out: dict[str, pd.Series] = {}
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get(ticker_col))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def first_existing(row: pd.Series | None, names: list[str], default: Any = "") -> Any:
    if row is None:
        return default
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            value = row.get(name)
            if as_text(value, ""):
                return value
    return default


def short_text(text: str, max_len: int = 210) -> str:
    text = " ".join(as_text(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def pct(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return round(float(value) * 100.0, 2)


def choose_intake_pool() -> pd.DataFrame:
    pool = read_csv_safe(ROOT / "sharpe4_recovery_candidate_pool.csv")
    if pool.empty or "ticker" not in pool.columns:
        return pd.DataFrame()
    pool = pool.copy()
    pool["ticker"] = pool["ticker"].apply(clean_ticker)
    pool = pool[pool["ticker"] != ""].copy()

    lane = pool.get("recovery_lane", pd.Series("", index=pool.index)).astype(str)
    outside = pool[~lane.str.contains("Repair current book", case=False, na=False)].copy()
    outside_lane = outside.get("recovery_lane", pd.Series("", index=outside.index)).astype(str)
    preferred = outside[
        outside_lane.str.contains("Research candidate - risk entry first|Recovery watchlist|Downside research", case=False, na=False)
    ].copy()
    if preferred.empty:
        preferred = outside.copy()
    if "recovery_rank_score" in preferred.columns:
        preferred["_score"] = pd.to_numeric(preferred["recovery_rank_score"], errors="coerce").fillna(0)
        preferred = preferred.sort_values("_score", ascending=False)
    elif "recovery_rank" in preferred.columns:
        preferred["_rank"] = pd.to_numeric(preferred["recovery_rank"], errors="coerce").fillna(9999)
        preferred = preferred.sort_values("_rank")
    return preferred.drop_duplicates("ticker", keep="first").head(MAX_INTAKE_ROWS).drop(
        columns=[c for c in ["_score", "_rank"] if c in preferred.columns],
        errors="ignore",
    )


def price_risk_label(annual_vol: float, cvar95: float, data_status: str) -> tuple[str, str, float]:
    if "NO_PRICE" in data_status:
        return "Data missing", "Cannot size without real price history.", 0.0
    if annual_vol >= 0.80 or cvar95 >= 0.075:
        return "Very high", "Tail risk is large; no paper size until pullback, event proof, and stop plan are explicit.", 0.25
    if annual_vol >= 0.55 or cvar95 >= 0.055:
        return "High", "This can move too much for a normal starter; research only until risk budget is written.", 0.50
    if annual_vol >= 0.35 or cvar95 >= 0.040:
        return "Medium", "Risk is usable only with small defined budget and a stop rule.", 0.75
    return "Lower", "Risk is calmer, but still needs evidence and execution proof before paper sizing.", 1.00


def liquidity_read(adv: float, label: str) -> tuple[str, float, float, str]:
    label = as_text(label).upper()
    if not np.isfinite(adv) or adv <= 0:
        return "Data missing", np.nan, np.nan, "No ADV proof. Fill a live quote / spread snapshot first."
    if label in {"HIGH", "LIQUID"} or adv >= 2_000_000_000:
        spread = 2.0
        liq_label = "Good"
        note = "Daily liquidity is strong, but live bid/ask still must be checked."
    elif label == "GOOD" or adv >= 500_000_000:
        spread = 4.0
        liq_label = "Usable"
        note = "Liquidity is usable for research, but spread/TCA proof is still required."
    elif label == "FAIR" or adv >= 100_000_000:
        spread = 8.0
        liq_label = "Review"
        note = "Manual spread proof is required before any paper route."
    else:
        spread = 20.0
        liq_label = "Thin"
        note = "Too fragile for Sharpe 4 active use without better execution proof."
    tiny_trade = MODEL_ACCOUNT_VALUE * 0.005
    participation = tiny_trade / adv * 100.0
    return liq_label, spread, participation, note


def estimated_tca_bps(spread_bps: float, participation_pct: float, vol_annual: float) -> float:
    if not np.isfinite(spread_bps):
        return np.nan
    participation = participation_pct if np.isfinite(participation_pct) else 0.02
    vol_penalty = 3.0 if vol_annual >= 0.80 else 1.5 if vol_annual >= 0.55 else 0.75
    impact = min(12.0, np.sqrt(max(participation, 0.001)) * 1.8)
    return round(spread_bps * 0.5 + impact + 3.0 + vol_penalty, 2)


def calc_price_metrics(tickers: list[str], price_metrics: dict[str, pd.Series]) -> dict[str, dict[str, Any]]:
    returns = get_returns(sorted(set(tickers + FACTOR_PROXIES)), lookback=252)
    out: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        r = returns[ticker].dropna() if ticker in returns.columns else pd.Series(dtype=float)
        pm = price_metrics.get(ticker)
        fallback_ann_vol = safe_float(first_existing(pm, ["realized_vol_20d_pct"]), np.nan) / 100.0

        if len(r) >= 120:
            annual_vol = annualized_vol(r)
            var95, cvar95 = var_cvar(r, 0.95)
            var99, cvar99 = var_cvar(r, 0.99)
            data_status = f"OK: {len(r)} daily returns"
            data_source = "sp500_price_cache.csv / backtest_price_cache.csv"
        elif np.isfinite(fallback_ann_vol) and fallback_ann_vol > 0:
            annual_vol = fallback_ann_vol
            daily = annual_vol / np.sqrt(252)
            var95 = daily * 1.65
            cvar95 = daily * 2.25
            var99 = daily * 2.33
            cvar99 = daily * 3.00
            data_status = "Fallback: 20d realized vol only"
            data_source = "theme_candidate_price_metrics.csv"
        else:
            annual_vol = var95 = cvar95 = var99 = cvar99 = np.nan
            data_status = "NO_PRICE_HISTORY"
            data_source = "missing"

        if len(r) >= 80:
            wealth = (1.0 + r).cumprod()
            max_dd = float((wealth / wealth.cummax() - 1.0).min())
        else:
            max_dd = np.nan

        label, note, cap = price_risk_label(annual_vol, cvar95, data_status)
        stop_pct = np.nan
        if np.isfinite(cvar95):
            stop_pct = min(18.0, max(6.0, cvar95 * 100.0 * 1.6))

        out[ticker] = {
            "price_data_status": data_status,
            "price_source": data_source,
            "annual_vol_pct": pct(annual_vol),
            "daily_var_95_pct": pct(var95),
            "daily_cvar_95_pct": pct(cvar95),
            "daily_var_99_pct": pct(var99),
            "daily_cvar_99_pct": pct(cvar99),
            "five_day_cvar_95_pct": pct(cvar95 * np.sqrt(5)) if np.isfinite(cvar95) else np.nan,
            "max_drawdown_1y_pct": pct(max_dd),
            "price_risk": label,
            "price_risk_note": note,
            "starter_cap_after_all_gates_clear_pct": cap,
            "paper_stop_if_ever_tested_pct": round(stop_pct, 2) if np.isfinite(stop_pct) else np.nan,
            "returns": r,
        }
    return out


def calc_correlation(tickers: list[str], price_stats: dict[str, dict[str, Any]]) -> pd.DataFrame:
    returns = get_returns(sorted(set(tickers + FACTOR_PROXIES)), lookback=252)
    rows = []
    candidate_returns = [t for t in tickers if t in returns.columns]
    for ticker in tickers:
        if ticker not in returns.columns:
            rows.append({
                "ticker": ticker,
                "corr_to_spy": np.nan,
                "corr_to_qqq": np.nan,
                "corr_to_smh": np.nan,
                "beta_to_spy": np.nan,
                "beta_to_qqq": np.nan,
                "beta_to_smh": np.nan,
                "highest_peer_corr": np.nan,
                "highest_peer": "",
                "correlation_risk": "Data missing",
                "correlation_note": "No enough local return history to measure factor crowding.",
                "source_file": "sp500_price_cache.csv / backtest_price_cache.csv",
            })
            continue
        row: dict[str, Any] = {"ticker": ticker}
        for proxy in ["SPY", "QQQ", "SMH"]:
            if proxy in returns.columns:
                pair = returns[[ticker, proxy]].dropna()
                row[f"corr_to_{proxy.lower()}"] = round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 3) if len(pair) >= 40 else np.nan
                row[f"beta_to_{proxy.lower()}"] = round(beta_to_factor(pair.iloc[:, 0], pair.iloc[:, 1]), 3) if len(pair) >= 40 else np.nan
            else:
                row[f"corr_to_{proxy.lower()}"] = np.nan
                row[f"beta_to_{proxy.lower()}"] = np.nan
        peer_corrs: list[tuple[str, float]] = []
        for peer in candidate_returns:
            if peer == ticker:
                continue
            pair = returns[[ticker, peer]].dropna()
            if len(pair) >= 40:
                peer_corrs.append((peer, float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))))
        if peer_corrs:
            peer, corr = max(peer_corrs, key=lambda x: abs(x[1]))
            row["highest_peer"] = peer
            row["highest_peer_corr"] = round(corr, 3)
        else:
            row["highest_peer"] = ""
            row["highest_peer_corr"] = np.nan

        max_factor = np.nanmax([safe_float(row.get("corr_to_qqq")), safe_float(row.get("corr_to_smh")), safe_float(row.get("corr_to_spy"))])
        peer_corr = safe_float(row.get("highest_peer_corr"), np.nan)
        if np.isfinite(peer_corr) and abs(peer_corr) >= 0.82:
            risk = "Crowded"
            note = f"Moves too much like {row['highest_peer']}; do not treat as separate risk."
        elif np.isfinite(max_factor) and max_factor >= 0.78:
            risk = "Factor heavy"
            note = "Strong market/tech/semiconductor beta. Size must count against the same exposure bucket."
        else:
            risk = "Normal"
            note = "No extreme local correlation flag, but stress correlation can still jump."
        row["correlation_risk"] = risk
        row["correlation_note"] = note
        row["source_file"] = "sp500_price_cache.csv / backtest_price_cache.csv"
        rows.append(row)
    return pd.DataFrame(rows)


def earnings_read(ticker: str, earnings: dict[str, pd.Series], options: dict[str, pd.Series]) -> tuple[str, str, str, float, float]:
    er = earnings.get(ticker)
    opt = options.get(ticker)
    days = safe_float(first_existing(er, ["days_until", "days_until_earnings", "days_to_earnings"], np.nan), np.nan)
    if not np.isfinite(days):
        days = safe_float(first_existing(opt, ["days_to_earnings"], np.nan), np.nan)
    iv_rank = safe_float(first_existing(opt, ["iv_rank"], np.nan), np.nan)
    implied = safe_float(first_existing(er, ["implied_move"], np.nan), np.nan)
    if not np.isfinite(implied):
        atm_iv = safe_float(first_existing(opt, ["atm_iv"], np.nan), np.nan)
        implied = atm_iv if np.isfinite(atm_iv) and atm_iv > 0.02 else np.nan

    if np.isfinite(days) and -2 <= days <= 10:
        status = "Block until earnings clears"
        note = "Near earnings window. Do not add exposure before the gap risk is reviewed."
    elif np.isfinite(days):
        status = "Calendar known"
        note = f"Next earnings is about {days:.0f} days away; still verify date and implied move."
    else:
        status = "Calendar proof missing"
        note = "Earnings date is not proven in the local files. Manual calendar check required."

    date = as_text(first_existing(er, ["earnings_date"], ""))
    return status, note, date, days, iv_rank if np.isfinite(iv_rank) else np.nan


def route_read(ticker: str, pool_row: pd.Series, event_row: pd.Series | None, theme_row: pd.Series | None, price_row: pd.Series | None) -> tuple[str, str, str, str, str]:
    trend = as_text(first_existing(price_row, ["trend_state"], ""))
    vol_state = as_text(first_existing(price_row, ["volatility_state"], ""))
    ret20 = safe_float(first_existing(price_row, ["ret_20d_pct"], np.nan), np.nan)
    ret63 = safe_float(first_existing(price_row, ["ret_63d_pct"], np.nan), np.nan)
    event_route = as_text(first_existing(event_row, ["directional_route"], ""))
    cycle = as_text(first_existing(event_row, ["subsector_cycle_phase"], ""))
    theme_status = as_text(first_existing(theme_row, ["theme_candidate_status"], ""))

    if "EXTREME" in vol_state or ret20 >= 35:
        short_route = "Short term: no chase. Wait for a pullback, calmer volatility, and volume confirmation."
    elif "UPTREND" in trend and ret20 > 0:
        short_route = "Short term: watch for price confirmation; no size until risk entry is complete."
    elif "DOWNTREND" in trend:
        short_route = "Short term: avoid bullish route until trend repair is visible."
    else:
        short_route = "Short term: observe only; price is not clean enough yet."

    if "late-cycle" in cycle.lower():
        medium_route = "Medium term: late-cycle leader risk. Treat as crowded; demand stronger evidence before any call idea."
    elif "early improvement" in cycle.lower() or theme_status == "ACTIVE_RESEARCH_READY":
        medium_route = "Medium term: research candidate if event proof, liquidity, and correlation checks pass."
    elif ret63 > 15 and ret20 < 0:
        medium_route = "Medium term: possible pullback after strong run; wait for stabilization."
    else:
        medium_route = "Medium term: backlog until a cleaner thesis appears."

    if ticker in {"MSFT", "GOOGL", "AMZN", "META"}:
        long_route = "Long term: business-quality research sleeve only; needs valuation and earnings proof."
    elif ticker in {"ASML", "KLAC", "NVDA", "AMD", "SMCI", "ANET", "DELL"}:
        long_route = "Long term: AI infrastructure theme, but size must respect semiconductor / hardware cycle risk."
    else:
        long_route = "Long term: no durable thesis until fundamental proof is added."

    if "PUT" in event_route.upper():
        option_answer = "Option route: put / hedge research only after spread and event proof. No live option action."
    elif "CALL" in event_route.upper():
        option_answer = "Option route: call research only after risk-book entry, IV check, spread proof, and price trigger. No weekly chase."
    else:
        option_answer = "Option route: stock research first. Options are blocked until the risk book is complete."

    current_answer = "Research only. Build the risk entry first; no paper size, no live order, no option trade."
    return current_answer, short_route, medium_route, long_route, option_answer


def build_intake() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    intake_pool = choose_intake_pool()
    if intake_pool.empty:
        empty_state = {
            "date": today_str(),
            "status": "NO_INTAKE_POOL",
            "candidate_count": 0,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_state

    tickers = intake_pool["ticker"].dropna().astype(str).map(clean_ticker).drop_duplicates().tolist()
    price_metrics = one_by_ticker(read_csv_safe(ROOT / "theme_candidate_price_metrics.csv"), "ticker")
    theme = one_by_ticker(read_csv_safe(ROOT / "theme_candidate_enrichment.csv"), "ticker", "attention_score")
    event = one_by_ticker(read_csv_safe(ROOT / "event_readthrough_target_ranking.csv"), "target_ticker", "best_event_score")
    liquidity = one_by_ticker(read_csv_safe(ROOT / "intraday_liquidity_proxy.csv"), "ticker", "avg_20d_dollar_volume")
    earnings = one_by_ticker(read_csv_safe(ROOT / "earnings_calendar.csv"), "ticker")
    options = one_by_ticker(read_csv_safe(ROOT / "options_signals.csv"), "ticker", "rank_options")

    price_stats = calc_price_metrics(tickers, price_metrics)
    corr = calc_correlation(tickers, price_stats)
    corr_map = one_by_ticker(corr, "ticker")

    cards: list[dict[str, Any]] = []
    var_liq_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    pool_map = one_by_ticker(intake_pool, "ticker", "recovery_rank_score")

    for ticker in tickers:
        p = pool_map.get(ticker)
        th = theme.get(ticker)
        ev = event.get(ticker)
        pm = price_metrics.get(ticker)
        liq = liquidity.get(ticker)
        c = corr_map.get(ticker)
        ps = price_stats.get(ticker, {})

        adv = safe_float(first_existing(liq, ["avg_20d_dollar_volume"], np.nan), np.nan)
        if not np.isfinite(adv):
            adv = safe_float(first_existing(pm, ["avg_dollar_volume_20d"], np.nan), np.nan)
        liq_label_raw = as_text(first_existing(liq, ["liquidity_label"], ""))
        if not liq_label_raw:
            liq_label_raw = as_text(first_existing(pm, ["liquidity_status"], ""))
        liq_status, spread_bps, participation_pct, liq_note = liquidity_read(adv, liq_label_raw)
        tca_bps = estimated_tca_bps(spread_bps, participation_pct, safe_float(ps.get("annual_vol_pct"), np.nan) / 100.0)
        earnings_status, earnings_note, earnings_date, days_to_earnings, iv_rank = earnings_read(ticker, earnings, options)

        current_answer, short_route, medium_route, long_route, option_answer = route_read(ticker, p, ev, th, pm)

        event_headline = short_text(as_text(first_existing(ev, ["top_headline"], "")) or as_text(first_existing(th, ["top_headline"], "")), 170)
        event_role = as_text(first_existing(ev, ["top_target_role"], "")) or as_text(first_existing(th, ["chain_role"], ""))
        event_score = safe_float(first_existing(ev, ["best_event_score"], np.nan), np.nan)
        proof_required = as_text(first_existing(ev, ["proof_required"], "")) or "Validate causal link, event-time price reaction, liquidity, spread, and risk-book entry."
        source_ticker = as_text(first_existing(th, ["source_news_ticker"], ""))
        publishers = short_text(as_text(first_existing(th, ["publisher_sample"], "")), 160)

        corr_risk = as_text(first_existing(c, ["correlation_risk"], "Data missing"))
        corr_note = as_text(first_existing(c, ["correlation_note"], "Need local return history."))

        blocker_parts = []
        if "missing" in earnings_status.lower():
            blocker_parts.append("earnings calendar proof")
        if safe_float(ps.get("annual_vol_pct"), 0) >= 55:
            blocker_parts.append("high volatility")
        if corr_risk in {"Crowded", "Factor heavy"}:
            blocker_parts.append("factor crowding")
        if liq_status in {"Data missing", "Review", "Thin"}:
            blocker_parts.append("liquidity / spread proof")
        blocker_parts.append("risk-book entry not complete")
        blockers = "; ".join(dict.fromkeys(blocker_parts))

        plain_thesis = (
            f"{ticker} is on the research list because the local files show "
            f"{as_text(first_existing(p, ['event_status'], 'event/theme interest')).lower()} and "
            f"{as_text(first_existing(p, ['liquidity_status'], 'some liquidity coverage')).lower()}."
        )
        if event_headline:
            plain_thesis += f" Main news hook: {event_headline}"

        risk_level = as_text(ps.get("price_risk"), "Data missing")
        option_clean = option_answer.replace("Option route: ", "")
        cards.append({
            "intake_rank": len(cards) + 1,
            "ticker": ticker,
            "current_answer": current_answer,
            "plain_thesis": plain_thesis,
            "risk_level": risk_level,
            "liquidity": liq_status,
            "correlation": corr_risk,
            "earnings": earnings_status,
            "short_term": short_route,
            "medium_term": medium_route,
            "long_term": long_route,
            "options_now": option_clean,
            "proof_needed": proof_required,
            "main_blockers": blockers,
            "starter_cap_after_all_gates_clear_pct": ps.get("starter_cap_after_all_gates_clear_pct", 0.0),
            "paper_stop_if_ever_tested_pct": ps.get("paper_stop_if_ever_tested_pct", np.nan),
            "source_files": "sharpe4_recovery_candidate_pool.csv / theme_candidate_enrichment.csv / event_readthrough_target_ranking.csv / price cache / liquidity proxy",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        var_liq_rows.append({
            "ticker": ticker,
            "price_data_status": ps.get("price_data_status", "NO_PRICE_HISTORY"),
            "annual_vol_pct": ps.get("annual_vol_pct"),
            "daily_var_95_pct": ps.get("daily_var_95_pct"),
            "daily_cvar_95_pct": ps.get("daily_cvar_95_pct"),
            "daily_var_99_pct": ps.get("daily_var_99_pct"),
            "daily_cvar_99_pct": ps.get("daily_cvar_99_pct"),
            "five_day_cvar_95_pct": ps.get("five_day_cvar_95_pct"),
            "max_drawdown_1y_pct": ps.get("max_drawdown_1y_pct"),
            "price_risk": risk_level,
            "price_risk_note": ps.get("price_risk_note", ""),
            "avg_dollar_volume_20d": round(adv, 2) if np.isfinite(adv) else np.nan,
            "liquidity_status": liq_status,
            "spread_proxy_bps": spread_bps,
            "tiny_0_5pct_trade_participation_pct": round(participation_pct, 5) if np.isfinite(participation_pct) else np.nan,
            "estimated_tca_bps": tca_bps,
            "liquidity_note": liq_note,
            "starter_cap_after_all_gates_clear_pct": ps.get("starter_cap_after_all_gates_clear_pct", 0.0),
            "source_file": f"{ps.get('price_source', 'missing')} / intraday_liquidity_proxy.csv",
            "research_only": True,
        })

        event_rows.append({
            "ticker": ticker,
            "event_score": round(event_score, 2) if np.isfinite(event_score) else np.nan,
            "event_role": event_role,
            "event_headline": event_headline,
            "source_news_ticker": source_ticker,
            "publisher_sample": publishers,
            "earnings_status": earnings_status,
            "earnings_date": earnings_date,
            "days_to_earnings": round(days_to_earnings, 1) if np.isfinite(days_to_earnings) else np.nan,
            "iv_rank": round(iv_rank, 1) if np.isfinite(iv_rank) else np.nan,
            "event_route": as_text(first_existing(ev, ["directional_route"], "")),
            "option_answer": option_answer,
            "proof_required": proof_required,
            "source_file": "event_readthrough_target_ranking.csv / theme_candidate_enrichment.csv / earnings_calendar.csv / options_signals.csv",
            "research_only": True,
        })

    cards_df = pd.DataFrame(cards)
    var_liq_df = pd.DataFrame(var_liq_rows)
    event_df = pd.DataFrame(event_rows)
    corr_df = corr.copy()

    high_risk = int(cards_df["risk_level"].isin(["Very high", "High"]).sum()) if not cards_df.empty else 0
    missing_earnings = int(cards_df["earnings"].str.contains("missing", case=False, na=False).sum()) if not cards_df.empty else 0
    crowded = int(cards_df["correlation"].isin(["Crowded", "Factor heavy"]).sum()) if not cards_df.empty else 0
    option_blocked = int(cards_df["options_now"].str.contains("No weekly chase|blocked|research only", case=False, na=False).sum()) if not cards_df.empty else 0

    state = {
        "date": today_str(),
        "status": "RISK_BOOK_INTAKE_REQUIRED",
        "candidate_count": int(len(cards_df)),
        "high_or_very_high_risk_count": high_risk,
        "earnings_calendar_missing_count": missing_earnings,
        "crowded_or_factor_heavy_count": crowded,
        "options_blocked_or_research_only_count": option_blocked,
        "paper_sizing_allowed_now_count": 0,
        "plain_english": "These names are research candidates only. None can be used for Sharpe 4 paper sizing until the risk-book entry, event proof, liquidity/spread proof, and correlation check are complete.",
        "next_required_work": "Open candidate cards, fill missing earnings and spread proof, then promote only names that clear risk, event, correlation, and execution checks.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return cards_df, var_liq_df, event_df, corr_df, state


def write_report(cards: pd.DataFrame, var_liq: pd.DataFrame, event: pd.DataFrame, corr: pd.DataFrame, state: dict[str, Any]) -> None:
    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Current Answer",
            "",
            f"- Status: **{state['status']}**",
            f"- Candidates checked: **{state['candidate_count']}**",
            f"- High / very high single-name risk: **{state['high_or_very_high_risk_count']}**",
            f"- Missing earnings calendar proof: **{state['earnings_calendar_missing_count']}**",
            f"- Crowded / factor-heavy names: **{state['crowded_or_factor_heavy_count']}**",
            f"- Paper sizing allowed now: **{state['paper_sizing_allowed_now_count']}**",
            "",
            state["plain_english"],
        ]),
        "## Candidate Cards\n\n" + df_to_markdown(cards.head(24)),
        "## VaR / Liquidity\n\n" + df_to_markdown(var_liq.head(24)),
        "## Event / Route\n\n" + df_to_markdown(event.head(24)),
        "## Correlation Proxy\n\n" + df_to_markdown(corr.head(24)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 188 - Sharpe 4 Risk-Book Intake", sections)


def main() -> None:
    cards, var_liq, event, corr, state = build_intake()
    cards.to_csv(OUT_CARDS, index=False)
    var_liq.to_csv(OUT_VAR_LIQ, index=False)
    event.to_csv(OUT_EVENT_ROUTE, index=False)
    corr.to_csv(OUT_CORR, index=False)
    write_json(OUT_STATE, state)
    write_report(cards, var_liq, event, corr, state)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Risk-book intake candidates: {state['candidate_count']}")
    print(f"[OK] High/very-high risk names: {state['high_or_very_high_risk_count']}")
    print(f"[OK] Paper sizing allowed now: {state['paper_sizing_allowed_now_count']}")


if __name__ == "__main__":
    main()
