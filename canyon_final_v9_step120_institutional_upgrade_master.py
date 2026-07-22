#!/usr/bin/env python3
"""
Canyon v9 Step 120 - Institutional Upgrade Master.

Research-only. No broker connection. No live orders.

This step operationalizes the institutional gaps the user called out:
point-in-time data quality, strict backtest hygiene, risk calibration,
execution/TCA, portfolio construction, event/fundamental depth, and real-time
alert coverage.

The step does not pretend local/yfinance/proxy data is institutional data.
Instead it creates explicit controls, scores, source lineage, and hard upgrade
requirements so weak evidence cannot silently pass as production quality.

Outputs:
  institutional_data_quality_audit.csv
  institutional_backtest_integrity_audit.csv
  institutional_risk_calibration.csv
  institutional_tca_cost_estimates.csv
  institutional_execution_capacity_limits.csv
  institutional_portfolio_construction_plan.csv
  institutional_sleeve_budget_plan.csv
  institutional_event_research_depth.csv
  institutional_realtime_alert_matrix.csv
  institutional_gap_master_scorecard.csv
  institutional_upgrade_state.json
  institutional_upgrade_report.md
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    FACTOR_PROXIES,
    clean_ticker,
    df_to_markdown,
    get_returns,
    load_current_book,
    load_liquidity_proxy,
    load_price_cache,
    normalize_weight,
    portfolio_return_series,
    read_csv_safe,
    read_json_safe,
    today_str,
    var_cvar,
    write_json,
    write_markdown_report,
)


OUT_DATA = ROOT / "institutional_data_quality_audit.csv"
OUT_BACKTEST = ROOT / "institutional_backtest_integrity_audit.csv"
OUT_RISK = ROOT / "institutional_risk_calibration.csv"
OUT_TCA = ROOT / "institutional_tca_cost_estimates.csv"
OUT_CAPACITY = ROOT / "institutional_execution_capacity_limits.csv"
OUT_PORTFOLIO = ROOT / "institutional_portfolio_construction_plan.csv"
OUT_SLEEVES = ROOT / "institutional_sleeve_budget_plan.csv"
OUT_EVENT = ROOT / "institutional_event_research_depth.csv"
OUT_ALERT = ROOT / "institutional_realtime_alert_matrix.csv"
OUT_SCORECARD = ROOT / "institutional_gap_master_scorecard.csv"
OUT_STATE = ROOT / "institutional_upgrade_state.json"
OUT_REPORT = ROOT / "institutional_upgrade_report.md"

MODEL_ACCOUNT_VALUE = 100000.0


def file_age_hours(path: Path) -> float:
    if not path.exists():
        return np.inf
    return max(0.0, (time.time() - path.stat().st_mtime) / 3600.0)


def exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10


def status_from_score(score: float) -> str:
    if score >= 85:
        return "PASS"
    if score >= 65:
        return "REVIEW"
    if score >= 40:
        return "WEAK"
    return "BLOCKER"


def severity_status(score: float) -> str:
    if score >= 75:
        return "CLEAR"
    if score >= 55:
        return "REVIEW"
    if score >= 35:
        return "SIZE_DOWN"
    return "BLOCK_NEW"


def pct(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(x):
        return np.nan
    if abs(x) > 1.5:
        x = x / 100.0
    return x


def parse_percent_text(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).replace("%", "").replace(",", "").strip()
    try:
        return float(text) / 100.0
    except Exception:
        return np.nan


def source_type(path_name: str) -> str:
    low = path_name.lower()
    if "yfinance" in low or "yf" in low:
        return "yfinance/proxy"
    if "cache" in low:
        return "local cache"
    if "manual" in low or "template" in low:
        return "manual"
    if "sec" in low or "edgar" in low:
        return "public filing cache"
    if "news" in low:
        return "news cache"
    return "local output"


def audit_data_quality() -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(control: str, source_file: str, score: float, evidence: str, required: str, layer: str = "L1 Data Integrity") -> None:
        checks.append({
            "category": "Data quality and true history",
            "control": control,
            "layer": layer,
            "score": round(float(score), 1),
            "status": status_from_score(score),
            "source_file": source_file,
            "source_type": source_type(source_file),
            "freshness_hours": round(file_age_hours(ROOT / source_file), 2) if source_file else np.nan,
            "evidence": evidence,
            "required_next_action": required,
            "research_only": True,
        })

    price = load_price_cache()
    ledger = read_csv_safe(ROOT / "point_in_time_evidence_ledger.csv")
    requirements = read_csv_safe(ROOT / "source_lineage_requirements.csv")
    source_files = ["sp500_price_cache.csv", "backtest_price_cache.csv", "regime_price_cache.csv"]
    present_price = [f for f in source_files if exists_nonempty(ROOT / f)]
    n_prices = int(price.shape[1]) if not price.empty else 0
    n_days = int(price.shape[0]) if not price.empty else 0
    add(
        "Historical price coverage",
        "sp500_price_cache.csv/backtest_price_cache.csv/regime_price_cache.csv",
        min(80.0, 25 + n_prices * 1.5 + min(n_days, 750) / 15),
        f"{n_prices} tickers and {n_days} rows from {len(present_price)} local price files.",
        "Keep a versioned price store with adjustment flags, source timestamp, and vendor identifier.",
    )

    has_pit = exists_nonempty(ROOT / "point_in_time_prices.csv") or exists_nonempty(ROOT / "pit_fundamentals.csv")
    has_local_ledger = not ledger.empty
    add(
        "Point-in-time data availability",
        "point_in_time_prices.csv / pit_fundamentals.csv",
        85 if has_pit else (35 if has_local_ledger else 15),
        "Dedicated point-in-time files found." if has_pit else (
            f"Local evidence ledger has {len(ledger)} observed sources, but no dedicated vendor-grade point-in-time store."
            if has_local_ledger else "No dedicated point-in-time price/fundamental store found."
        ),
        "Add timestamped point-in-time snapshots: what was known, when it was known, and when the model consumed it.",
    )

    has_constituents = exists_nonempty(ROOT / "sp500_constituents_history.csv") or exists_nonempty(ROOT / "universe_membership_history.csv")
    has_snapshot = exists_nonempty(ROOT / "universe_membership_snapshot.csv")
    add(
        "Historical universe membership",
        "sp500_constituents_history.csv / universe_membership_history.csv",
        80 if has_constituents else (35 if has_snapshot else 20),
        "Historical constituent membership file found." if has_constituents else (
            "Current universe snapshot exists, but no historical membership table was found."
            if has_snapshot else "Current universe files exist, but no historical membership table was found."
        ),
        "Store historical index membership and ticker eligibility by date to reduce survivorship bias.",
    )

    has_delisted = exists_nonempty(ROOT / "delisted_tickers.csv")
    add(
        "Delisted and dead ticker coverage",
        "delisted_tickers.csv",
        80 if has_delisted else 10,
        "Delisted ticker table found." if has_delisted else "No delisted/dead ticker table found.",
        "Add delisted tickers and stale symbols to backtests so losers do not disappear.",
    )

    has_corp_actions = any(exists_nonempty(ROOT / f) for f in ["corporate_actions.csv", "splits_dividends.csv"])
    add(
        "Corporate action trace",
        "corporate_actions.csv / splits_dividends.csv",
        80 if has_corp_actions else 25,
        "Corporate action table found." if has_corp_actions else "Prices may be adjusted, but no explicit split/dividend trace file is present.",
        "Persist split/dividend adjustment details and verify adjusted vs raw close behavior.",
    )

    news = read_json_safe(ROOT / "stock_news.json", {})
    news_items = 0
    news_with_ts = 0
    if isinstance(news, dict):
        for items in news.get("news", {}).values():
            if isinstance(items, list):
                news_items += len(items)
                for item in items:
                    title = str(item.get("title", "")).strip()
                    ts = item.get("published_ts", 0)
                    published = str(item.get("published", "")).strip()
                    if title and ((isinstance(ts, (int, float)) and ts > 0) or (published and published != "1970-01-01")):
                        news_with_ts += 1
    news_score = 20 if news_items == 0 else min(85.0, 20 + 65 * news_with_ts / max(news_items, 1))
    add(
        "News timestamp quality",
        "stock_news.json",
        news_score,
        f"{news_with_ts}/{news_items} cached news items have usable titles and timestamps.",
        "Use vendor news IDs and exact publish timestamps; discard 1970/blank records from signal logic.",
        "L5 Event / News",
    )

    ec = read_csv_safe(ROOT / "earnings_calendar.csv")
    earnings_cols = {"ticker", "earnings_date", "days_until"}
    ec_score = 20
    evidence = "No earnings calendar output found."
    if not ec.empty:
        hit = len(earnings_cols.intersection(ec.columns))
        ec_score = 30 + hit / len(earnings_cols) * 45
        evidence = f"Earnings calendar has {len(ec)} rows and {hit}/{len(earnings_cols)} required columns."
    add(
        "Earnings event timestamp coverage",
        "earnings_calendar.csv",
        ec_score,
        evidence,
        "Add before/after-market flag, exact release time, source vendor, and revision timestamp.",
        "L5 Event / Earnings",
    )

    manifest = read_csv_safe(ROOT / "canyon_file_manifest.csv")
    add(
        "Source manifest and lineage",
        "canyon_file_manifest.csv",
        85 if (not manifest.empty and not ledger.empty) else (75 if not manifest.empty or not ledger.empty else 30),
        (
            f"Manifest has {len(manifest)} rows; point-in-time evidence ledger has {len(ledger)} rows."
            if (not manifest.empty and not ledger.empty)
            else (f"Manifest has {len(manifest)} rows." if not manifest.empty else (f"Evidence ledger has {len(ledger)} rows." if not ledger.empty else "No current file manifest found."))
        ),
        "Add producer step, input dependencies, run time, vendor/source, and freshness SLA for every output.",
    )
    if not requirements.empty:
        blockers = int(requirements["current_status"].astype(str).str.upper().isin(["BLOCKER"]).sum()) if "current_status" in requirements.columns else 0
        add(
            "Institutional data requirements tracker",
            "source_lineage_requirements.csv",
            max(35, 70 - blockers * 5),
            f"{len(requirements)} source-lineage requirements tracked; {blockers} blockers remain.",
            "Close blockers by adding vendor-grade PIT price/fundamental, membership, delisted, and corporate action stores.",
        )
    return pd.DataFrame(checks)


def audit_backtest_integrity() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    summary = read_csv_safe(ROOT / "backtest_summary.csv")
    signal_ic = read_csv_safe(ROOT / "backtest_signal_ic.csv")
    live_ic = read_csv_safe(ROOT / "live_ic_history.csv")
    bias_guard = read_csv_safe(ROOT / "backtest_bias_guard.csv")
    bias_state = read_json_safe(ROOT / "backtest_bias_state.json", {})

    def add(control: str, score: float, evidence: str, required: str, source_file: str) -> None:
        rows.append({
            "category": "Strict backtest system",
            "control": control,
            "score": round(float(score), 1),
            "status": status_from_score(score),
            "source_file": source_file,
            "evidence": evidence,
            "required_next_action": required,
            "research_only": True,
        })

    if monthly.empty:
        add("Monthly backtest history", 15, "No monthly backtest file found.", "Run Step62 and persist monthly returns.", "backtest_monthly_perf.csv")
    else:
        months = len(monthly)
        start = str(monthly.iloc[0].get("rebalance_date", monthly.iloc[0].get("date", "")))
        end = str(monthly.iloc[-1].get("period_end", monthly.iloc[-1].get("date", "")))
        add(
            "Monthly backtest history",
            min(80, 20 + months * 1.3),
            f"{months} monthly rows from {start} to {end}.",
            "Expand to multiple regimes and include delisted tickers before treating as institutional.",
            "backtest_monthly_perf.csv",
        )

    if not monthly.empty and "strategy_ret" in monthly.columns:
        ret = pd.to_numeric(monthly["strategy_ret"], errors="coerce").dropna()
        if len(ret) >= 24:
            midpoint = len(ret) // 2
            first = float(ret.iloc[:midpoint].mean() * 12)
            second = float(ret.iloc[midpoint:].mean() * 12)
            decay = second - first
            score = 65 if abs(decay) < 0.15 else 45
            add(
                "Walk-forward / out-of-sample proxy",
                score,
                f"First-half annualized mean {first:.2%}; second-half {second:.2%}; decay {decay:+.2%}.",
                "Replace this proxy with a true expanding-window walk-forward engine and frozen model snapshots.",
                "backtest_monthly_perf.csv",
            )
        else:
            add("Walk-forward / out-of-sample proxy", 35, f"Only {len(ret)} return periods.", "Need at least several market regimes.", "backtest_monthly_perf.csv")

    has_signal_dates = False
    if not signal_ic.empty:
        has_signal_dates = any(c in signal_ic.columns for c in ["signal_date", "asof_date", "prediction_date"])
    add(
        "Look-ahead bias defense",
        70 if has_signal_dates else (45 if not bias_guard.empty else 25),
        "Signal date columns found." if has_signal_dates else (
            "Backtest bias guard exists, but signal IC still lacks explicit signal/as-of dates."
            if not bias_guard.empty else "Backtest IC has no explicit signal/as-of date column."
        ),
        "Every feature must carry as-of time and every backtest join must enforce feature_time < trade_time.",
        "backtest_signal_ic.csv",
    )

    has_membership = exists_nonempty(ROOT / "universe_membership_history.csv") or exists_nonempty(ROOT / "sp500_constituents_history.csv")
    has_snapshot = exists_nonempty(ROOT / "universe_membership_snapshot.csv")
    add(
        "Survivorship-bias correction",
        75 if has_membership else (35 if has_snapshot else 20),
        "Historical membership table found." if has_membership else (
            "Current universe snapshot exists, but historical membership is still missing."
            if has_snapshot else "No historical membership table found."
        ),
        "Backtest universe must be reconstructed per date, including delisted names.",
        "universe_membership_history.csv",
    )

    if not monthly.empty and {"turnover_pct", "tc_cost_bps"}.issubset(monthly.columns):
        tc = pd.to_numeric(monthly["tc_cost_bps"], errors="coerce").dropna()
        turn = pd.to_numeric(monthly["turnover_pct"], errors="coerce").dropna()
        add(
            "Transaction cost and turnover model",
            65,
            f"Backtest has turnover and cost fields; median turnover {turn.median():.1f}%, median cost {tc.median():.1f} bps.",
            "Replace static cost assumptions with Step120 TCA: spread, volume, market impact, auction risk, and failed-fill assumptions.",
            "backtest_monthly_perf.csv",
        )
    else:
        add("Transaction cost and turnover model", 25, "Backtest lacks turnover/cost columns.", "Add turnover and cost model columns.", "backtest_monthly_perf.csv")

    if not signal_ic.empty and {"signal", "n_obs", "mean_ic"}.issubset(signal_ic.columns):
        n = pd.to_numeric(signal_ic["n_obs"], errors="coerce").fillna(0)
        mean_ic = pd.to_numeric(signal_ic["mean_ic"], errors="coerce").dropna()
        min_obs = int(n.min()) if not n.empty else 0
        avg_ic = float(mean_ic.mean()) if not mean_ic.empty else np.nan
        add(
            "Signal decay and IC depth",
            50 if min_obs < 30 else 75,
            f"{len(signal_ic)} signals; minimum observations {min_obs}; average IC {avg_ic:.3f}.",
            "Track IC by regime, horizon, sector, decay bucket, and failure mode before increasing sizing.",
            "backtest_signal_ic.csv",
        )
    else:
        add("Signal decay and IC depth", 20, "No signal IC table with required columns.", "Run signal IC tracking with enough observations.", "backtest_signal_ic.csv")

    if not live_ic.empty:
        add(
            "Paper/live validation link",
            min(75, 35 + len(live_ic)),
            f"Live IC file has {len(live_ic)} rows.",
            "Require live/paper IC confirmation before moving a signal from prototype to active.",
            "live_ic_history.csv",
        )
    else:
        add(
            "Paper/live validation link",
            25,
            "No live IC history found.",
            "Persist daily live IC predictions, realized outcomes, and rejected ideas.",
            "live_ic_history.csv",
        )
    if summary.empty:
        add("Backtest summary report", 20, "No summary found.", "Run Step62/92 and store key metrics.", "backtest_summary.csv")
    else:
        add("Backtest summary report", 70, f"Backtest summary has {len(summary)} metric rows.", "Audit whether reported results use actual signal history or proxy signals.", "backtest_summary.csv")
    if bias_state:
        score = float(bias_state.get("backtest_bias_guard_score", 0.0) or 0.0)
        add(
            "Backtest bias guard infrastructure",
            min(75, score + 15),
            f"Step122 bias guard status {bias_state.get('overall_status')}; score {score:.1f}; weak/blocker controls {bias_state.get('blocker_or_weak_controls')}.",
            "Replace proxy guardrails with a true event-time walk-forward backtester.",
            "backtest_bias_guard.csv / backtest_bias_state.json",
        )
    return pd.DataFrame(rows)


def audit_risk_calibration(book: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    single = read_csv_safe(ROOT / "single_name_risk_budget.csv")
    var_summary = read_csv_safe(ROOT / "portfolio_var_cvar_summary.csv")
    vol_state = read_json_safe(ROOT / "vol_target_state.json", {})
    beta = read_csv_safe(ROOT / "portfolio_beta_report.csv")
    options = read_csv_safe(ROOT / "options_signals.csv")
    final_gate = read_csv_safe(ROOT / "final_risk_gate.csv")
    p = portfolio_return_series(book, lookback=756)

    def add(control: str, score: float, status: str, evidence: str, required: str, source_file: str) -> None:
        rows.append({
            "category": "Risk calibration",
            "control": control,
            "score": round(float(score), 1),
            "status": status,
            "source_file": source_file,
            "evidence": evidence,
            "required_next_action": required,
            "research_only": True,
        })

    if p.empty:
        add("VaR/CVaR backtest calibration", 25, "BLOCK_NEW", "No portfolio return series for breach testing.", "Build return history for current book.", "sp500_price_cache.csv/backtest_price_cache.csv")
    else:
        var95, cvar95 = var_cvar(p, 0.95)
        breaches = int((p < -var95).sum()) if np.isfinite(var95) else 0
        expected = len(p) * 0.05
        ratio = breaches / max(expected, 1e-9)
        score = 75 if 0.5 <= ratio <= 1.8 else 50
        add(
            "VaR/CVaR backtest calibration",
            score,
            severity_status(score),
            f"{len(p)} daily portfolio returns; VaR95 {var95:.2%}; breaches {breaches}; expected {expected:.1f}; ratio {ratio:.2f}.",
            "Calibrate VaR windows and confidence levels on historical breach rates by regime.",
            "portfolio_var_cvar_summary.csv",
        )

    if single.empty:
        add("Single-name tail and earnings risk", 30, "BLOCK_NEW", "No single-name risk budget file found.", "Run Step111.", "single_name_risk_budget.csv")
    else:
        missing_earn = int(single.get("earnings_days_to_event", pd.Series(dtype=float)).isna().sum()) if "earnings_days_to_event" in single.columns else len(single)
        risk_flags = int(single.get("single_name_action", pd.Series(dtype=str)).astype(str).str.upper().isin(["SIZE_DOWN", "REDUCE_ONLY", "BLOCK_NEW", "MISSING_DATA_REVIEW"]).sum()) if "single_name_action" in single.columns else 0
        score = max(45, 80 - missing_earn * 3)
        add(
            "Single-name tail and earnings risk",
            score,
            severity_status(score),
            f"{len(single)} tickers; {risk_flags} flags; {missing_earn} missing earnings fields.",
            "Calibrate ticker VaR and earnings gap thresholds using historical pre/post-earnings gaps.",
            "single_name_risk_budget.csv",
        )

    if vol_state:
        est = pct(vol_state.get("estimated_annual_vol"))
        target = pct(vol_state.get("target_annual_vol", 0.15))
        mult = float(vol_state.get("vol_exposure_multiplier", 1.0) or 1.0)
        score = 75 if np.isfinite(est) else 45
        add(
            "Volatility target management",
            score,
            severity_status(score),
            f"Estimated annual vol {est:.2%}; target {target:.2%}; exposure multiplier {mult:.2f}.",
            "Backtest the vol-target rule across regimes and compare realized vs target volatility.",
            "vol_target_state.json",
        )
    else:
        add("Volatility target management", 30, "BLOCK_NEW", "No vol target state found.", "Run Step114.", "vol_target_state.json")

    if beta.empty:
        add("Factor risk model coverage", 30, "BLOCK_NEW", "No beta report found.", "Run Step116 and add Barra/Axioma-like factors.", "portfolio_beta_report.csv")
    else:
        factor_count = len(beta)
        score = min(70, 25 + factor_count * 6)
        add(
            "Factor risk model coverage",
            score,
            severity_status(score),
            f"{factor_count} factor/proxy betas found; this is proxy-based, not a full commercial risk model.",
            "Add style factors: value, momentum, quality, size, volatility, liquidity, leverage, and industry factors.",
            "portfolio_beta_report.csv",
        )

    if options.empty:
        add("Options Greeks book risk", 25, "BLOCK_NEW", "No options signals file found.", "Add option position book with delta/gamma/vega/theta by expiry.", "options_signals.csv")
    else:
        greek_cols = {"delta", "gamma", "vega", "theta"}
        has_greeks = greek_cols.issubset(set(options.columns))
        score = 70 if has_greeks else 35
        add(
            "Options Greeks book risk",
            score,
            severity_status(score),
            "Full Greeks columns found." if has_greeks else "Options signals exist, but no full portfolio Greeks book is present.",
            "Build options book risk: delta, gamma, vega, theta, expiry bucket, IV shock, and dealer-gamma stress.",
            "options_signals.csv",
        )

    if not final_gate.empty:
        flags = int(final_gate.get("final_risk_action", pd.Series(dtype=str)).astype(str).str.upper().isin(["SIZE_DOWN", "REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"]).sum())
        add(
            "Risk gate integration",
            75,
            "REVIEW",
            f"Final risk gate has {len(final_gate)} rows and {flags} active flags.",
            "Calibrate thresholds with historical false positives/false negatives.",
            "final_risk_gate.csv",
        )
    else:
        add("Risk gate integration", 35, "BLOCK_NEW", "No final risk gate file found.", "Run Step118.", "final_risk_gate.csv")
    return pd.DataFrame(rows)


def build_tca(book: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if book.empty:
        empty = pd.DataFrame()
        return empty, empty
    tickers = book["ticker"].apply(clean_ticker).tolist()
    liq = load_liquidity_proxy(tickers=tickers, refresh_missing=True)
    single = read_csv_safe(ROOT / "single_name_risk_budget.csv")
    desk_state = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    gate = read_csv_safe(ROOT / "final_risk_gate.csv")

    base = book.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)
    if "weight" not in base.columns:
        base["weight"] = 1.0 / max(len(base), 1)
    base["weight"] = base["weight"].apply(normalize_weight)
    base["target_trade_dollars"] = base["weight"] * MODEL_ACCOUNT_VALUE

    if not liq.empty and "ticker" in liq.columns:
        base = base.merge(liq[["ticker", "avg_20d_dollar_volume", "median_20d_volume", "liquidity_label"]], on="ticker", how="left")
    if not single.empty and "ticker" in single.columns:
        keep = [c for c in ["ticker", "days_to_liquidate", "max_liquidity_weight", "liquidity_label"] if c in single.columns]
        base = base.merge(single[keep].rename(columns={"liquidity_label": "risk_liquidity_label"}), on="ticker", how="left")
    if not desk_state.empty and "ticker" in desk_state.columns:
        keep = [c for c in ["ticker", "spread_bps", "spread_status", "volume_ratio"] if c in desk_state.columns]
        base = base.merge(desk_state[keep], on="ticker", how="left")
    if not gate.empty and "ticker" in gate.columns:
        keep = [c for c in ["ticker", "final_risk_action", "recommended_risk_weight_pct"] if c in gate.columns]
        base = base.merge(gate[keep], on="ticker", how="left")

    rows = []
    cap_rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        weight = normalize_weight(row.get("weight"))
        adv = pd.to_numeric(pd.Series([row.get("avg_20d_dollar_volume")]), errors="coerce").iloc[0]
        if not np.isfinite(adv) or adv <= 0:
            adv = pd.to_numeric(pd.Series([row.get("adv_dollar")]), errors="coerce").iloc[0] if "adv_dollar" in row.index else np.nan
        spread = pd.to_numeric(pd.Series([row.get("spread_bps")]), errors="coerce").iloc[0]
        if not np.isfinite(spread):
            label = str(row.get("liquidity_label", row.get("risk_liquidity_label", "MISSING"))).upper()
            spread = {"HIGH": 2.0, "GOOD": 5.0, "FAIR": 12.0, "THIN": 30.0, "LOW": 75.0}.get(label, 40.0)
        trade_dollars = float(row.get("target_trade_dollars", 0.0) or 0.0)
        participation = trade_dollars / adv if np.isfinite(adv) and adv > 0 else np.nan
        half_spread_bps = max(spread / 2.0, 0.0)
        impact_bps = 0.0 if not np.isfinite(participation) else 35.0 * math.sqrt(max(participation, 0.0))
        auction_risk_bps = 8.0 if weight > 0.05 else 4.0
        failed_fill_bps = 15.0 if not np.isfinite(participation) or participation > 0.05 else 3.0
        total_cost_bps = half_spread_bps + impact_bps + auction_risk_bps + failed_fill_bps
        status = "CLEAR"
        if total_cost_bps >= 80 or (np.isfinite(participation) and participation > 0.10):
            status = "BLOCK_NEW"
        elif total_cost_bps >= 40 or (np.isfinite(participation) and participation > 0.05):
            status = "SIZE_DOWN"
        elif total_cost_bps >= 20:
            status = "REVIEW"
        max_daily_trade_dollars = adv * 0.05 if np.isfinite(adv) else np.nan
        max_weight_by_capacity = max_daily_trade_dollars / MODEL_ACCOUNT_VALUE if np.isfinite(max_daily_trade_dollars) else np.nan
        rows.append({
            "ticker": ticker,
            "target_weight_pct": weight * 100,
            "target_trade_dollars": trade_dollars,
            "avg_20d_dollar_volume": adv,
            "participation_rate_pct": participation * 100 if np.isfinite(participation) else np.nan,
            "spread_bps_est": spread,
            "half_spread_cost_bps": half_spread_bps,
            "market_impact_bps": impact_bps,
            "auction_risk_bps": auction_risk_bps,
            "failed_fill_buffer_bps": failed_fill_bps,
            "total_tca_cost_bps": total_cost_bps,
            "total_cost_dollars": trade_dollars * total_cost_bps / 10000.0,
            "execution_status": status,
            "execution_assumption": "Research TCA only; no broker connection and no live order path.",
            "source_file": "intraday_liquidity_proxy.csv / desk_monitor_ticker_state.csv",
        })
        cap_rows.append({
            "ticker": ticker,
            "avg_20d_dollar_volume": adv,
            "max_daily_participation_pct": 5.0,
            "max_daily_trade_dollars": max_daily_trade_dollars,
            "max_weight_by_capacity_pct": max_weight_by_capacity * 100 if np.isfinite(max_weight_by_capacity) else np.nan,
            "current_weight_pct": weight * 100,
            "capacity_status": "BLOCK_NEW" if not np.isfinite(max_weight_by_capacity) else ("SIZE_DOWN" if weight > max_weight_by_capacity else "CLEAR"),
            "source_file": "intraday_liquidity_proxy.csv",
        })
    return pd.DataFrame(rows), pd.DataFrame(cap_rows)


def build_portfolio_construction(book: pd.DataFrame, tca: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if book.empty:
        return pd.DataFrame(), pd.DataFrame()
    external_plan = read_csv_safe(ROOT / "institutional_target_weights.csv")
    external_sleeve = read_csv_safe(ROOT / "institutional_sleeve_allocations.csv")
    if not external_plan.empty:
        plan = external_plan.copy()
        if "target_weight_pct" in plan.columns and "recommended_institutional_weight_pct" not in plan.columns:
            plan["recommended_institutional_weight_pct"] = plan["target_weight_pct"]
        if "target_status" in plan.columns and "construction_status" not in plan.columns:
            plan["construction_status"] = plan["target_status"]
        if "target_weight_pct" in plan.columns and "recommended_risk_weight_pct" not in plan.columns:
            plan["recommended_risk_weight_pct"] = plan["target_weight_pct"]
        if external_sleeve.empty:
            external_sleeve = pd.DataFrame()
        return plan, external_sleeve
    gate = read_csv_safe(ROOT / "final_risk_gate.csv")
    sector = read_csv_safe(ROOT / "sector_active_exposure.csv")
    beta = read_csv_safe(ROOT / "portfolio_beta_report.csv")
    master_state = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})
    master_action = str(master_state.get("master_risk_action", "REVIEW"))
    master_mult = float(master_state.get("master_exposure_multiplier", 0.7) or 0.7)

    base = book.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)
    base["weight"] = base["weight"].apply(normalize_weight)
    if not gate.empty and "ticker" in gate.columns:
        keep = [c for c in ["ticker", "final_risk_action", "recommended_risk_weight_pct"] if c in gate.columns]
        base = base.merge(gate[keep], on="ticker", how="left")
    if not tca.empty and "ticker" in tca.columns:
        keep = [c for c in ["ticker", "total_tca_cost_bps", "execution_status"] if c in tca.columns]
        base = base.merge(tca[keep], on="ticker", how="left")

    sector_flags = set()
    if not sector.empty and {"sector", "cap_status"}.issubset(sector.columns):
        sector_flags = set(sector.loc[sector["cap_status"].astype(str).str.upper().isin(["SIZE_DOWN", "BLOCK_NEW"]), "sector"].astype(str))

    rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        sector_name = str(row.get("sector", "Unknown"))
        alpha = pd.to_numeric(pd.Series([row.get("alpha_score")]), errors="coerce").iloc[0]
        action = str(row.get("final_risk_action", row.get("action", "REVIEW"))).upper()
        exec_status = str(row.get("execution_status", "REVIEW")).upper()
        tca_bps = pd.to_numeric(pd.Series([row.get("total_tca_cost_bps")]), errors="coerce").iloc[0]
        old_w = normalize_weight(row.get("weight"))
        risk_w = pct(row.get("recommended_risk_weight_pct"))
        if np.isfinite(risk_w):
            risk_w = risk_w
        else:
            risk_w = old_w
        sleeve = "Tactical"
        if ticker in {"SPY", "QQQ", "XLK", "XLF", "XLV", "XLE", "IYR", "TLT", "GLD"}:
            sleeve = "Core / Hedge"
        elif "EARN" in str(row.get("top_signal", "")).upper() or "SURPRISE" in str(row.get("top_signal", "")).upper():
            sleeve = "Event"
        elif action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
            sleeve = "Cash / Risk Control"
        elif np.isfinite(alpha) and alpha >= 80:
            sleeve = "Core"
        if exec_status in {"SIZE_DOWN", "BLOCK_NEW"} and sleeve not in {"Cash / Risk Control"}:
            sleeve = "Tactical"

        construction_status = "CLEAR"
        reasons = []
        adjusted = min(old_w * master_mult, risk_w)
        if sector_name in sector_flags:
            adjusted *= 0.75
            reasons.append("sector cap pressure")
        if exec_status == "SIZE_DOWN":
            adjusted *= 0.75
            reasons.append("execution cost pressure")
        if exec_status == "BLOCK_NEW":
            adjusted = min(adjusted, old_w * 0.25)
            construction_status = "BLOCK_NEW"
            reasons.append("execution capacity block")
        if action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
            adjusted = min(adjusted, old_w * 0.50)
            construction_status = "REDUCE_ONLY"
            reasons.append("final risk gate")
        elif action == "SIZE_DOWN" or master_action == "SIZE_DOWN":
            construction_status = "SIZE_DOWN"
            reasons.append("risk gate size down")
        elif not reasons:
            reasons.append("within prototype constraints")

        rows.append({
            "ticker": ticker,
            "sector": sector_name,
            "sleeve": sleeve,
            "current_weight_pct": old_w * 100,
            "recommended_institutional_weight_pct": adjusted * 100,
            "master_action": master_action,
            "final_risk_action": action,
            "execution_status": exec_status,
            "total_tca_cost_bps": tca_bps,
            "construction_status": construction_status,
            "reason": "; ".join(reasons),
            "source_file": "daily_picks_filtered.csv / final_risk_gate.csv / institutional_tca_cost_estimates.csv",
        })
    plan = pd.DataFrame(rows)

    risk_mult = master_mult
    if master_action in {"BLOCK_NEW", "REDUCE_ONLY", "BLOCKED"}:
        budget = {"Core": 0.30, "Tactical": 0.05, "Event": 0.00, "Core / Hedge": 0.15, "Cash / Risk Control": 0.50}
    elif master_action == "SIZE_DOWN":
        budget = {"Core": 0.35, "Tactical": 0.10, "Event": 0.05, "Core / Hedge": 0.15, "Cash / Risk Control": 0.35}
    else:
        budget = {"Core": 0.45, "Tactical": 0.20, "Event": 0.10, "Core / Hedge": 0.10, "Cash / Risk Control": 0.15}
    sleeve_rows = []
    for sleeve, target in budget.items():
        current = float(plan.loc[plan["sleeve"] == sleeve, "recommended_institutional_weight_pct"].sum() / 100.0) if not plan.empty else 0.0
        sleeve_rows.append({
            "sleeve": sleeve,
            "target_budget_pct": target * 100,
            "current_recommended_pct": current * 100,
            "budget_gap_pct": (current - target * risk_mult) * 100,
            "status": "REVIEW" if current > target * 1.25 else "CLEAR",
            "purpose": {
                "Core": "Higher-conviction slower-turnover ideas.",
                "Tactical": "Shorter-horizon ideas with tighter risk controls.",
                "Event": "Earnings/news/SEC-driven ideas with explicit event risk.",
                "Core / Hedge": "ETF, duration, gold, or hedge sleeve context.",
                "Cash / Risk Control": "Unused risk budget and forced de-risking sleeve.",
            }.get(sleeve, ""),
            "source_file": "institutional_risk_gate_state.json",
        })
    return plan, pd.DataFrame(sleeve_rows)


def audit_event_research_depth(book: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()
    external_dossier = read_csv_safe(ROOT / "event_research_dossier.csv")
    if not external_dossier.empty:
        return external_dossier
    datasets = {
        "earnings_calendar": read_csv_safe(ROOT / "earnings_calendar.csv"),
        "earnings_surprise": read_csv_safe(ROOT / "earnings_surprise_scores.csv"),
        "earnings_revision": read_csv_safe(ROOT / "earnings_revision_scores.csv"),
        "earnings_call_nlp": read_csv_safe(ROOT / "earnings_nlp_scores.csv"),
        "insider_form4": read_csv_safe(ROOT / "insider_signal_scores.csv"),
        "sec_event": read_csv_safe(ROOT / "sec_event_layer.csv"),
        "news_event": read_csv_safe(ROOT / "news_event_risk.csv"),
        "sentiment": read_csv_safe(ROOT / "finbert_sentiment_scores.csv"),
    }
    for key, df in list(datasets.items()):
        if not df.empty and "ticker" in df.columns:
            df = df.copy()
            df["ticker"] = df["ticker"].apply(clean_ticker)
            datasets[key] = df
    news_json = read_json_safe(ROOT / "stock_news.json", {})
    news_map = news_json.get("news", {}) if isinstance(news_json, dict) else {}

    rows = []
    for _, row in book.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        checks = {}
        for key, df in datasets.items():
            checks[key] = bool(not df.empty and "ticker" in df.columns and ticker in set(df["ticker"].astype(str)))
        checks["raw_news"] = bool(news_map.get(ticker))
        score = 10 + sum(1 for v in checks.values() if v) / max(len(checks), 1) * 80
        missing = [k for k, v in checks.items() if not v]
        status = status_from_score(score)
        rows.append({
            "ticker": ticker,
            "sector": row.get("sector", "Unknown"),
            "event_research_score": round(score, 1),
            "status": status,
            "has_earnings_calendar": checks.get("earnings_calendar"),
            "has_earnings_surprise": checks.get("earnings_surprise"),
            "has_earnings_revision": checks.get("earnings_revision"),
            "has_earnings_call_nlp": checks.get("earnings_call_nlp"),
            "has_insider_form4": checks.get("insider_form4"),
            "has_sec_event": checks.get("sec_event"),
            "has_news_event": checks.get("news_event"),
            "has_raw_news": checks.get("raw_news"),
            "missing_research_sources": ", ".join(missing),
            "required_next_action": "Add conference-call NLP, guidance revisions, 10-Q/10-K risk-factor diffs, Form 4 quality, estimate revisions, supplier/customer exposure, and litigation/regulatory risk.",
            "source_file": "earnings_calendar.csv / stock_news.json / sec_event_layer.csv / insider_signal_scores.csv",
        })
    return pd.DataFrame(rows)


def build_realtime_alert_matrix() -> pd.DataFrame:
    events = read_csv_safe(ROOT / "desk_monitor_events.csv")
    state = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    required = [
        ("PRICE_BREAK", "price break", "desk_monitor_price_volume_cache.csv"),
        ("VOLUME_SPIKE", "volume spike", "desk_monitor_price_volume_cache.csv"),
        ("VOLATILITY_REGIME_SHIFT", "volatility regime shift", "desk_monitor_price_volume_cache.csv"),
        ("SPREAD_WIDENING", "spread widening", "yfinance fast_info"),
        ("CORRELATION_BREAK", "correlation break", "sp500_price_cache.csv/backtest_price_cache.csv"),
        ("NEWS_SHOCK", "news shock", "stock_news.json"),
        ("EARNINGS_SURPRISE", "earnings surprise", "earnings_surprise_scores.csv"),
        ("RISK_LIMIT_BREACH", "risk limit breach", "institutional_risk_budget_summary.csv/final_risk_gate.csv"),
    ]
    rows = []
    for key, label, source in required:
        count = int((events["monitor"].astype(str) == key).sum()) if not events.empty and "monitor" in events.columns else 0
        crit = int(((events["monitor"].astype(str) == key) & (events["severity"].astype(str) == "CRITICAL")).sum()) if not events.empty and {"monitor", "severity"}.issubset(events.columns) else 0
        source_ready = True
        if key == "SPREAD_WIDENING":
            source_ready = not state.empty and "spread_status" in state.columns
        elif "/" in source:
            source_ready = any(exists_nonempty(ROOT / part.strip()) for part in source.split("/") if part.strip().endswith(".csv"))
        elif source.endswith(".csv"):
            source_ready = exists_nonempty(ROOT / source)
        elif source == "stock_news.json":
            source_ready = exists_nonempty(ROOT / "stock_news.json")
        rows.append({
            "monitor": key,
            "plain_english_name": label,
            "source_file": source,
            "source_ready": source_ready,
            "event_count": count,
            "critical_count": crit,
            "status": "CLEAR" if source_ready else "DATA_GAP",
            "required_next_action": "Add alert routing and persistent acknowledgement log." if source_ready else "Fix source before relying on this monitor.",
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_master_scorecard(*frames: pd.DataFrame) -> pd.DataFrame:
    categories = []
    named = [
        ("Data quality and true history", frames[0], 0.18),
        ("Strict backtest system", frames[1], 0.18),
        ("Risk calibration", frames[2], 0.18),
        ("Execution / TCA", frames[3], 0.14),
        ("Portfolio construction", frames[4], 0.14),
        ("Event and fundamental depth", frames[5], 0.10),
        ("Real-time monitoring", frames[6], 0.08),
    ]
    for name, df, weight in named:
        score_col = None
        for c in ["score", "event_research_score"]:
            if c in df.columns:
                score_col = c
                break
        if name == "Risk calibration" and score_col:
            vals = pd.to_numeric(df[score_col], errors="coerce").dropna()
            base_score = float(vals.mean()) if not vals.empty else 25.0
            overlay_state = read_json_safe(ROOT / "institutional_risk_overlay_state.json", {})
            if overlay_state:
                overlay_score = float(overlay_state.get("institutional_risk_overlay_score", 40.0) or 40.0)
                score = base_score * 0.70 + overlay_score * 0.30
                flags = int(overlay_state.get("overlay_flags", 0) or 0)
                if "status" in df.columns:
                    flags += int(df["status"].astype(str).str.upper().isin(["BLOCKER", "WEAK", "REVIEW"]).sum())
            else:
                score = base_score
                flags = int(df["status"].astype(str).str.upper().isin(["BLOCKER", "WEAK", "REVIEW"]).sum()) if "status" in df.columns else 0
        elif name == "Execution / TCA" and "execution_status" in df.columns:
            execution_state = read_json_safe(ROOT / "execution_playbook_state.json", {})
            if execution_state:
                score = float(execution_state.get("execution_readiness_score", 30.0) or 30.0)
                flags = int(execution_state.get("blocked_or_data_gap_trades", 0) or 0)
                flags += int(execution_state.get("review_or_size_down_trades", 0) or 0)
            else:
                status_scores = {"CLEAR": 80, "REVIEW": 60, "SIZE_DOWN": 40, "BLOCK_NEW": 20}
                vals = df["execution_status"].astype(str).str.upper().map(status_scores).dropna()
                score = float(vals.mean()) if not vals.empty else 30.0
                flags = int((df["execution_status"].astype(str).str.upper() != "CLEAR").sum())
        elif name == "Portfolio construction" and "construction_status" in df.columns:
            builder_state = read_json_safe(ROOT / "portfolio_construction_state.json", {})
            if builder_state:
                score = float(builder_state.get("portfolio_construction_score", 30.0) or 30.0)
                flags = int(builder_state.get("constraint_flags", 0) or 0)
                if "construction_status" in df.columns:
                    flags += int(df["construction_status"].astype(str).str.upper().isin(["SIZE_DOWN", "REDUCE_ONLY", "BLOCK_NEW"]).sum())
            else:
                status_scores = {"CLEAR": 80, "REVIEW": 60, "SIZE_DOWN": 45, "REDUCE_ONLY": 25, "BLOCK_NEW": 15}
                vals = df["construction_status"].astype(str).str.upper().map(status_scores).dropna()
                score = float(vals.mean()) if not vals.empty else 30.0
                flags = int((df["construction_status"].astype(str).str.upper() != "CLEAR").sum())
        elif name == "Real-time monitoring" and "source_ready" in df.columns:
            ready = int(df["source_ready"].fillna(False).astype(bool).sum())
            score = 20 + 80 * ready / max(len(df), 1)
            flags = int((~df["source_ready"].fillna(False).astype(bool)).sum())
        elif score_col:
            event_state = read_json_safe(ROOT / "event_research_state.json", {}) if name == "Event and fundamental depth" else {}
            if event_state:
                score = float(event_state.get("event_research_score", 25.0) or 25.0)
            else:
                vals = pd.to_numeric(df[score_col], errors="coerce").dropna()
                score = float(vals.mean()) if not vals.empty else 25.0
            status_cols = [c for c in ["status", "execution_status", "construction_status"] if c in df.columns]
            flags = 0
            if status_cols:
                s = df[status_cols[0]].astype(str).str.upper()
                flags = int(s.isin(["BLOCKER", "BLOCK_NEW", "SIZE_DOWN", "WEAK", "REDUCE_ONLY", "DATA_GAP"]).sum())
        else:
            score = 25.0
            flags = 0
        categories.append({
            "capability": name,
            "weight": weight,
            "score": round(score, 1),
            "weighted_score": round(score * weight, 2),
            "status": status_from_score(score),
            "flag_count": flags,
            "product_truth": "Active institutional prototype, not production-grade yet.",
        })
    out = pd.DataFrame(categories)
    return out


def write_outputs(
    data_audit: pd.DataFrame,
    backtest_audit: pd.DataFrame,
    risk_audit: pd.DataFrame,
    tca: pd.DataFrame,
    capacity: pd.DataFrame,
    portfolio_plan: pd.DataFrame,
    sleeve_plan: pd.DataFrame,
    event_depth: pd.DataFrame,
    alert_matrix: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> None:
    data_audit.to_csv(OUT_DATA, index=False)
    backtest_audit.to_csv(OUT_BACKTEST, index=False)
    risk_audit.to_csv(OUT_RISK, index=False)
    tca.to_csv(OUT_TCA, index=False)
    capacity.to_csv(OUT_CAPACITY, index=False)
    portfolio_plan.to_csv(OUT_PORTFOLIO, index=False)
    sleeve_plan.to_csv(OUT_SLEEVES, index=False)
    event_depth.to_csv(OUT_EVENT, index=False)
    alert_matrix.to_csv(OUT_ALERT, index=False)
    scorecard.to_csv(OUT_SCORECARD, index=False)

    readiness = float(scorecard["weighted_score"].sum()) if not scorecard.empty else 0.0
    blocker_count = int(scorecard["status"].astype(str).str.upper().isin(["BLOCKER", "WEAK"]).sum()) if not scorecard.empty else 0
    state = {
        "date": today_str(),
        "institutional_readiness_pct": round(readiness, 1),
        "overall_status": status_from_score(readiness),
        "blocker_or_weak_capabilities": blocker_count,
        "logic": "This is a research-only institutional upgrade control tower. It does not connect to a broker and cannot place orders.",
        "no_broker_connection": True,
        "research_only": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Product Truth",
        "",
        "This step turns institutional gaps into explicit controls. It does not claim that local/yfinance/proxy data equals a paid point-in-time institutional data stack.",
        "",
        f"- Institutional readiness: {state['institutional_readiness_pct']:.1f}%",
        f"- Overall status: {state['overall_status']}",
        f"- Weak or blocker capabilities: {state['blocker_or_weak_capabilities']}",
        "",
        "## Master Scorecard",
        "",
        df_to_markdown(scorecard),
        "",
        "## Data Quality and True History",
        "",
        df_to_markdown(data_audit, max_rows=80),
        "",
        "## Strict Backtest System",
        "",
        df_to_markdown(backtest_audit, max_rows=80),
        "",
        "## Risk Calibration",
        "",
        df_to_markdown(risk_audit, max_rows=80),
        "",
        "## Execution / TCA",
        "",
        df_to_markdown(tca, max_rows=60),
        "",
        "## Execution Capacity",
        "",
        df_to_markdown(capacity, max_rows=60),
        "",
        "## Portfolio Construction",
        "",
        df_to_markdown(portfolio_plan, max_rows=60),
        "",
        "## Sleeve Budget Plan",
        "",
        df_to_markdown(sleeve_plan, max_rows=20),
        "",
        "## Event and Fundamental Research Depth",
        "",
        df_to_markdown(event_depth, max_rows=60),
        "",
        "## Real-Time Alert Matrix",
        "",
        df_to_markdown(alert_matrix, max_rows=20),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 120 - Institutional Upgrade Master", sections)


def main() -> None:
    book = load_current_book(prefer_filtered=True)
    data_audit = audit_data_quality()
    backtest_audit = audit_backtest_integrity()
    risk_audit = audit_risk_calibration(book)
    tca, capacity = build_tca(book)
    portfolio_plan, sleeve_plan = build_portfolio_construction(book, tca)
    event_depth = audit_event_research_depth(book)
    alert_matrix = build_realtime_alert_matrix()
    scorecard = build_master_scorecard(
        data_audit,
        backtest_audit,
        risk_audit,
        tca,
        portfolio_plan,
        event_depth,
        alert_matrix,
    )
    write_outputs(
        data_audit,
        backtest_audit,
        risk_audit,
        tca,
        capacity,
        portfolio_plan,
        sleeve_plan,
        event_depth,
        alert_matrix,
        scorecard,
    )
    state = read_json_safe(OUT_STATE, {})
    print(f"[step120] wrote {OUT_SCORECARD.name}: {len(scorecard)} capabilities")
    print(f"[step120] readiness={state.get('institutional_readiness_pct')}% status={state.get('overall_status')}")
    print(f"[step120] wrote {OUT_DATA.name}, {OUT_BACKTEST.name}, {OUT_RISK.name}")
    print(f"[step120] wrote {OUT_TCA.name}, {OUT_PORTFOLIO.name}, {OUT_EVENT.name}, {OUT_ALERT.name}")
    print(f"[step120] wrote {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
