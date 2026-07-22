#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 126: Institutional Risk Overlays
=================================================

Research-only overlay audits for institutional gaps that are not captured by a
basic long-equity target book:

  1. Options Greeks book risk proxy
  2. Financing / borrow cost proxy
  3. Crowding risk matrix
  4. Tail hedge cost-benefit proxy

This is not a paid risk model, not a Barra/Axioma replacement, and not an
options portfolio system. It makes the missing institutional controls visible
and blocks overconfidence when sources are only local/yfinance/proxy.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PORTFOLIO_DOLLARS = 100_000.0

OUT_GREEKS = ROOT / "options_greeks_book_risk.csv"
OUT_FINANCING = ROOT / "financing_borrow_cost_model.csv"
OUT_CROWDING = ROOT / "crowding_risk_matrix.csv"
OUT_TAIL_CB = ROOT / "tail_hedge_cost_benefit.csv"
OUT_STATE = ROOT / "institutional_risk_overlay_state.json"
OUT_REPORT = ROOT / "institutional_risk_overlay_report.md"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 10:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json_safe(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists() or path.stat().st_size <= 2:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clean_ticker(value) -> str:
    return str(value).strip().upper()


def as_float(value, default=np.nan) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def status_from_score(score: float) -> str:
    if not np.isfinite(score):
        return "DATA_GAP"
    if score >= 80:
        return "PASS"
    if score >= 65:
        return "REVIEW"
    return "WEAK"


def status_from_risk(score: float) -> str:
    if not np.isfinite(score):
        return "DATA_GAP"
    if score >= 80:
        return "HIGH"
    if score >= 60:
        return "REVIEW"
    return "CLEAR"


def target_book() -> pd.DataFrame:
    target = read_csv_safe(ROOT / "institutional_target_weights.csv")
    if target.empty:
        target = read_csv_safe(ROOT / "daily_picks_filtered.csv")
        if not target.empty and "weight_pct" in target.columns:
            target["target_weight_pct"] = pd.to_numeric(target["weight_pct"], errors="coerce")
    if not target.empty and "ticker" in target.columns:
        target = target.copy()
        target["ticker"] = target["ticker"].apply(clean_ticker)
    return target


def build_options_greeks_proxy(book: pd.DataFrame) -> pd.DataFrame:
    options = read_csv_safe(ROOT / "options_signals.csv")
    if book.empty:
        return pd.DataFrame()
    base = book.copy()
    if not options.empty and "ticker" in options.columns:
        options = options.copy()
        options["ticker"] = options["ticker"].apply(clean_ticker)
        keep = [c for c in [
            "ticker", "options_strategy", "iv_rank", "atm_iv", "iv_skew", "pcr_vol",
            "pcr_oi", "gex_sign", "gex_net", "squeeze_risk", "gamma_score",
            "flow_score", "days_to_earnings", "term_structure",
        ] if c in options.columns]
        base = base.merge(options[keep], on="ticker", how="left")

    rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        weight = as_float(row.get("target_weight_pct"), as_float(row.get("weight_pct"), 0.0)) / 100.0
        strategy = str(row.get("options_strategy", "NO_OPTION_BOOK")).upper()
        iv_rank = as_float(row.get("iv_rank"), np.nan)
        atm_iv = as_float(row.get("atm_iv"), np.nan)
        gex_net = as_float(row.get("gex_net"), 0.0)
        gamma_score = as_float(row.get("gamma_score"), 50.0)
        pcr_vol = as_float(row.get("pcr_vol"), np.nan)
        squeeze = bool(row.get("squeeze_risk", False))

        direction_mult = 0.0
        if any(x in strategy for x in ["CALL", "BULL", "LONG"]):
            direction_mult = 1.0
        elif any(x in strategy for x in ["PUT", "PROTECTIVE", "BEAR"]):
            direction_mult = -0.35
        elif strategy == "MONITOR":
            direction_mult = 0.15

        delta_proxy = weight * direction_mult
        gamma_proxy = abs(gex_net) * abs(weight) / 1_000_000.0
        vega_proxy = abs(weight) * max(atm_iv if np.isfinite(atm_iv) else 0.35, 0.0)
        option_heat = 0.0
        if np.isfinite(iv_rank):
            option_heat += min(max(iv_rank, 0.0), 100.0) * 0.30
        option_heat += min(max(gamma_score, 0.0), 100.0) * 0.35
        if squeeze:
            option_heat += 20.0
        if np.isfinite(pcr_vol) and pcr_vol < 0.35:
            option_heat += 15.0
        option_heat = min(option_heat, 100.0)

        if strategy == "NO_OPTION_BOOK" or pd.isna(row.get("options_strategy", np.nan)):
            status = "DATA_GAP"
            required = "Run Step82 options signals or confirm there is no option overlay."
        elif option_heat >= 80:
            status = "HIGH"
            required = "Do not allow options to upgrade action; manual Greeks and IV term-structure check required."
        elif option_heat >= 60:
            status = "REVIEW"
            required = "Manual options desk review before any option-themed paper action."
        else:
            status = "CLEAR"
            required = "No option overlay action; monitor only."

        rows.append({
            "ticker": ticker,
            "target_weight_pct": weight * 100.0,
            "options_strategy": strategy,
            "delta_proxy": delta_proxy,
            "gamma_proxy": gamma_proxy,
            "vega_proxy": vega_proxy,
            "iv_rank": iv_rank,
            "atm_iv": atm_iv,
            "gex_net": gex_net,
            "gamma_score": gamma_score,
            "pcr_vol": pcr_vol,
            "squeeze_risk": squeeze,
            "options_heat_score": round(option_heat, 1),
            "greeks_status": status,
            "required_next_action": required,
            "source_file": "options_signals.csv / institutional_target_weights.csv",
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_financing_borrow_model(book: pd.DataFrame) -> pd.DataFrame:
    short = read_csv_safe(ROOT / "short_interest_scores.csv")
    if book.empty:
        return pd.DataFrame()
    base = book.copy()
    if not short.empty and "ticker" in short.columns:
        short = short.copy()
        short["ticker"] = short["ticker"].apply(clean_ticker)
        keep = [c for c in ["ticker", "short_pct_float", "short_ratio", "short_change_pct", "short_pressure", "signal"] if c in short.columns]
        base = base.merge(short[keep], on="ticker", how="left")
    rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        weight_pct = as_float(row.get("target_weight_pct"), as_float(row.get("weight_pct"), 0.0))
        short_pct = as_float(row.get("short_pct_float"), np.nan)
        short_ratio = as_float(row.get("short_ratio"), np.nan)
        short_pressure = as_float(row.get("short_pressure"), np.nan)
        borrow_bps = 25.0
        if np.isfinite(short_pct):
            borrow_bps += short_pct * 6_000.0
        if np.isfinite(short_ratio):
            borrow_bps += short_ratio * 15.0
        hard_to_borrow = bool((np.isfinite(short_pct) and short_pct >= 0.12) or (np.isfinite(short_ratio) and short_ratio >= 7.0))
        financing_dollars = abs(weight_pct) / 100.0 * MODEL_PORTFOLIO_DOLLARS * borrow_bps / 10000.0
        status = "REVIEW" if hard_to_borrow else "CLEAR"
        if not np.isfinite(short_pct) and not np.isfinite(short_ratio):
            status = "DATA_GAP"
        rows.append({
            "ticker": ticker,
            "target_weight_pct": weight_pct,
            "short_pct_float": short_pct,
            "short_ratio_days_to_cover": short_ratio,
            "short_pressure": short_pressure,
            "borrow_cost_bps_annual_proxy": round(borrow_bps, 1),
            "financing_cost_dollars_proxy": round(financing_dollars, 2),
            "hard_to_borrow_proxy": hard_to_borrow,
            "margin_requirement_pct_proxy": 100.0,
            "financing_status": status,
            "required_next_action": "Add broker-independent borrow/financing feed before short or leverage research." if status != "CLEAR" else "Long-only paper context; monitor borrow crowding only.",
            "source_file": "short_interest_scores.csv / institutional_target_weights.csv",
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_crowding_matrix(book: pd.DataFrame, greeks: pd.DataFrame, financing: pd.DataFrame) -> pd.DataFrame:
    corr = read_csv_safe(ROOT / "holdings_correlation_matrix.csv")
    monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    if book.empty:
        return pd.DataFrame()
    base = book.copy()
    for df, keep in [
        (greeks, ["ticker", "options_heat_score", "greeks_status"]),
        (financing, ["ticker", "short_pct_float", "short_ratio_days_to_cover", "financing_status"]),
        (monitor, ["ticker", "volume_spike_state", "price_break_state", "max_monitor_severity"]),
    ]:
        if not df.empty and "ticker" in df.columns:
            tmp = df.copy()
            tmp["ticker"] = tmp["ticker"].apply(clean_ticker)
            base = base.merge(tmp[[c for c in keep if c in tmp.columns]], on="ticker", how="left")

    corr_map: dict[str, float] = {}
    if not corr.empty and "Unnamed: 0" in corr.columns:
        cmat = corr.copy().rename(columns={"Unnamed: 0": "ticker"})
        cmat["ticker"] = cmat["ticker"].apply(clean_ticker)
        tickers = [clean_ticker(t) for t in base["ticker"].tolist()]
        for _, row in cmat.iterrows():
            ticker = clean_ticker(row.get("ticker"))
            vals = []
            for other in tickers:
                if other != ticker and other in row.index:
                    val = as_float(row.get(other), np.nan)
                    if np.isfinite(val):
                        vals.append(abs(val))
            corr_map[ticker] = float(np.nanmean(vals)) if vals else np.nan

    sector_weights = {}
    if "sector" in base.columns:
        temp = base.copy()
        temp["target_weight_pct"] = pd.to_numeric(temp.get("target_weight_pct", temp.get("weight_pct", 0.0)), errors="coerce").fillna(0.0)
        sector_weights = temp.groupby("sector")["target_weight_pct"].sum().to_dict()

    rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        sector = row.get("sector", "Unknown")
        sector_weight = as_float(sector_weights.get(sector, 0.0), 0.0)
        avg_corr = corr_map.get(ticker, np.nan)
        option_heat = as_float(row.get("options_heat_score"), 0.0)
        short_pct = as_float(row.get("short_pct_float"), 0.0)
        short_ratio = as_float(row.get("short_ratio_days_to_cover"), 0.0)
        monitor_sev = str(row.get("max_monitor_severity", "OK")).upper()
        crowding = 0.0
        crowding += min(sector_weight / 35.0 * 25.0, 25.0)
        if np.isfinite(avg_corr):
            crowding += min(avg_corr / 0.65 * 25.0, 25.0)
        crowding += min(option_heat / 100.0 * 20.0, 20.0)
        crowding += min((short_pct * 100.0) / 15.0 * 20.0, 20.0)
        if short_ratio >= 7:
            crowding += 5.0
        if monitor_sev == "CRITICAL":
            crowding += 10.0
        elif monitor_sev == "WARNING":
            crowding += 5.0
        crowding = min(crowding, 100.0)
        rows.append({
            "ticker": ticker,
            "sector": sector,
            "sector_target_weight_pct": sector_weight,
            "avg_abs_correlation_to_book": avg_corr,
            "options_heat_score": option_heat,
            "short_pct_float": short_pct,
            "short_ratio_days_to_cover": short_ratio,
            "monitor_severity": monitor_sev,
            "crowding_score": round(crowding, 1),
            "crowding_status": status_from_risk(crowding),
            "required_next_action": "Reduce/add no exposure until crowding source is manually reviewed." if crowding >= 80 else ("Monitor concentration and correlation before adding." if crowding >= 60 else "No immediate crowding block."),
            "source_file": "holdings_correlation_matrix.csv / options_signals.csv / short_interest_scores.csv / desk_monitor_ticker_state.csv",
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_tail_hedge_cost_benefit() -> pd.DataFrame:
    tail = read_csv_safe(ROOT / "tail_hedge_budget.csv")
    macro = read_csv_safe(ROOT / "macro_scenario_stress.csv")
    state = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})
    if tail.empty:
        return pd.DataFrame()
    worst_shock = 0.10
    if not macro.empty and "conservative_portfolio_impact" in macro.columns:
        vals = pd.to_numeric(macro["conservative_portfolio_impact"], errors="coerce").dropna()
        if not vals.empty:
            worst_shock = abs(float(vals.min()))
    exposure_mult = as_float(state.get("master_exposure_multiplier", 0.70), 0.70)
    rows = []
    for _, row in tail.iterrows():
        budget_dollars = as_float(row.get("research_budget_dollars"), 0.0)
        proxy = str(row.get("proxy", "hedge proxy"))
        gross = as_float(row.get("gross_research_exposure"), exposure_mult)
        notional_at_risk = MODEL_PORTFOLIO_DOLLARS * min(max(gross, 0.0), 1.5)
        expected_benefit = notional_at_risk * worst_shock * 0.35
        cost_to_benefit = budget_dollars / expected_benefit if expected_benefit > 0 else np.nan
        if not np.isfinite(cost_to_benefit):
            status = "DATA_GAP"
        elif cost_to_benefit <= 0.25:
            status = "PASS"
        elif cost_to_benefit <= 0.55:
            status = "REVIEW"
        else:
            status = "WEAK"
        rows.append({
            "hedge_sleeve": row.get("hedge_sleeve"),
            "proxy": proxy,
            "research_budget_dollars": budget_dollars,
            "worst_scenario_shock_abs": worst_shock,
            "estimated_loss_absorption_pct": 35.0,
            "estimated_benefit_dollars": round(expected_benefit, 2),
            "cost_to_benefit_ratio": cost_to_benefit,
            "tail_hedge_status": status,
            "required_next_action": "Get real option quotes and run payoff table before trusting hedge budget." if status != "PASS" else "Budget is reasonable as a planning proxy; still needs live quote verification.",
            "source_file": "tail_hedge_budget.csv / macro_scenario_stress.csv / institutional_risk_gate_state.json",
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_state(greeks: pd.DataFrame, financing: pd.DataFrame, crowding: pd.DataFrame, tail_cb: pd.DataFrame) -> dict:
    score_parts = []
    flags = 0
    for df, status_col in [
        (greeks, "greeks_status"),
        (financing, "financing_status"),
        (crowding, "crowding_status"),
        (tail_cb, "tail_hedge_status"),
    ]:
        if df.empty or status_col not in df.columns:
            score_parts.append(25.0)
            flags += 1
            continue
        status_scores = {
            "PASS": 85, "CLEAR": 82, "REVIEW": 62, "WEAK": 45,
            "HIGH": 35, "SIZE_DOWN": 40, "BLOCK_NEW": 20, "DATA_GAP": 25,
        }
        vals = df[status_col].astype(str).str.upper().map(status_scores).dropna()
        score_parts.append(float(vals.mean()) if not vals.empty else 25.0)
        flags += int(df[status_col].astype(str).str.upper().isin(["REVIEW", "WEAK", "HIGH", "DATA_GAP", "BLOCK_NEW"]).sum())
    raw_score = float(np.mean(score_parts)) if score_parts else 0.0
    # These overlays are intentionally proxy-based. Until the project has a
    # true options position book, borrow/financing feed, crowding dataset, and
    # live hedge quotes, the overlay score must not claim institutional PASS.
    proxy_quality_cap = 68.0
    score = min(raw_score, proxy_quality_cap)
    flags += 1
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "raw_overlay_score_before_proxy_cap": round(raw_score, 1),
        "proxy_quality_cap": proxy_quality_cap,
        "institutional_risk_overlay_score": round(score, 1),
        "overall_status": status_from_score(score),
        "overlay_flags": int(flags),
        "options_rows": int(len(greeks)),
        "financing_rows": int(len(financing)),
        "crowding_rows": int(len(crowding)),
        "tail_hedge_rows": int(len(tail_cb)),
        "research_only": True,
        "no_broker_connection": True,
        "truth": "Overlay risk proxies only. Not a paid Greeks book, margin system, borrow feed, crowding dataset, or hedge execution tool.",
    }


def write_report(state: dict, greeks: pd.DataFrame, financing: pd.DataFrame, crowding: pd.DataFrame, tail_cb: pd.DataFrame) -> None:
    lines = [
        "# Canyon v9 Step 126 — Institutional Risk Overlays",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Research-only overlay audit. No broker connection. No live orders.",
        "",
        f"- Overall status: {state.get('overall_status')}",
        f"- Overlay score: {state.get('institutional_risk_overlay_score')}",
        f"- Overlay flags: {state.get('overlay_flags')}",
        "",
        "## Outputs",
        "",
        "- `options_greeks_book_risk.csv`",
        "- `financing_borrow_cost_model.csv`",
        "- `crowding_risk_matrix.csv`",
        "- `tail_hedge_cost_benefit.csv`",
        "- `institutional_risk_overlay_state.json`",
        "",
        "## Product truth",
        "",
        "This makes institutional risk overlays visible, but it remains proxy-based until real options positions, borrow feeds, quote history, and hedge payoff tables are added.",
    ]
    for name, df, col in [
        ("Options Greeks proxy", greeks, "greeks_status"),
        ("Financing / borrow", financing, "financing_status"),
        ("Crowding", crowding, "crowding_status"),
        ("Tail hedge cost-benefit", tail_cb, "tail_hedge_status"),
    ]:
        lines.extend(["", f"## {name}", ""])
        if df.empty or col not in df.columns:
            lines.append("- No data.")
        else:
            for key, val in df[col].astype(str).value_counts().to_dict().items():
                lines.append(f"- {key}: {val}")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    book = target_book()
    greeks = build_options_greeks_proxy(book)
    financing = build_financing_borrow_model(book)
    crowding = build_crowding_matrix(book, greeks, financing)
    tail_cb = build_tail_hedge_cost_benefit()
    state = build_state(greeks, financing, crowding, tail_cb)

    greeks.to_csv(OUT_GREEKS, index=False)
    financing.to_csv(OUT_FINANCING, index=False)
    crowding.to_csv(OUT_CROWDING, index=False)
    tail_cb.to_csv(OUT_TAIL_CB, index=False)
    write_json(OUT_STATE, state)
    write_report(state, greeks, financing, crowding, tail_cb)

    print(f"[step126] wrote overlay files; score={state.get('institutional_risk_overlay_score')} status={state.get('overall_status')}")


if __name__ == "__main__":
    main()
