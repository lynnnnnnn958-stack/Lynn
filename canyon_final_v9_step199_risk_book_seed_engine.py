#!/usr/bin/env python3
"""
Canyon v9 Step 199 - Risk Book Seed Engine.

Research-only. No broker connection. No live orders.

Step198 says which tickers are outside the risk book. Step199 creates a
provisional risk-book seed for those tickers. A seed is not approval. It only
means the system now has first-pass risk facts: price history, VaR/CVaR,
volatility, drawdown, liquidity, factor correlation, starter cap, and stop rule.

The Final PM Gate can read these seed rows, but it must keep them review-only
until manual approval, event proof, and execution proof are complete.

Outputs:
  risk_book_seed_state.json
  risk_book_seed_entries.csv
  risk_book_seed_metric_detail.csv
  risk_book_seed_manual_approval_queue.csv
  risk_book_seed_sector_exposure_preview.csv
  risk_book_seed_report.md
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
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "risk_book_seed_state.json"
OUT_ENTRIES = ROOT / "risk_book_seed_entries.csv"
OUT_METRICS = ROOT / "risk_book_seed_metric_detail.csv"
OUT_APPROVAL = ROOT / "risk_book_seed_manual_approval_queue.csv"
OUT_SECTOR = ROOT / "risk_book_seed_sector_exposure_preview.csv"
OUT_REPORT = ROOT / "risk_book_seed_report.md"
OUT_ENTRIES_HISTORY = ROOT / "risk_book_seed_entries_history.csv"
OUT_METRICS_HISTORY = ROOT / "risk_book_seed_metric_history.csv"

FACTOR_TICKERS = ["SPY", "QQQ", "SMH", "SOXX", "XLK", "XLE", "TLT", "UUP"]


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def short(value: Any, limit: int = 240) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def pct(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return round(value * 100.0, 2)


def load_price_panel() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    repair = read_csv_safe(ROOT / "price_repair_download_cache.csv")
    if not repair.empty and {"date", "ticker", "close"}.issubset(repair.columns):
        cols = [c for c in ["date", "ticker", "close", "volume"] if c in repair.columns]
        work = repair[cols].copy()
        work["source_file"] = "price_repair_download_cache.csv"
        frames.append(work)

    monitor = read_csv_safe(ROOT / "desk_monitor_price_volume_cache.csv")
    if not monitor.empty and {"date", "ticker", "close"}.issubset(monitor.columns):
        cols = [c for c in ["date", "ticker", "close", "volume"] if c in monitor.columns]
        work = monitor[cols].copy()
        work["source_file"] = "desk_monitor_price_volume_cache.csv"
        frames.append(work)

    pit = read_csv_safe(ROOT / "point_in_time_prices.csv")
    if not pit.empty and {"price_date", "ticker", "adjusted_close"}.issubset(pit.columns):
        work = pit[["price_date", "ticker", "adjusted_close"]].copy()
        work = work.rename(columns={"price_date": "date", "adjusted_close": "close"})
        work["volume"] = np.nan
        work["source_file"] = "point_in_time_prices.csv"
        frames.append(work)

    for filename in ["backtest_price_cache.csv", "sp500_price_cache.csv"]:
        wide = read_csv_safe(ROOT / filename)
        if wide.empty:
            continue
        date_col = "Unnamed: 0" if "Unnamed: 0" in wide.columns else wide.columns[0]
        wide = wide.rename(columns={date_col: "date"})
        long = wide.melt(id_vars=["date"], var_name="ticker", value_name="close")
        long["volume"] = np.nan
        long["source_file"] = filename
        frames.append(long)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume", "source_file"])

    prices = pd.concat(frames, ignore_index=True)
    prices["ticker"] = prices["ticker"].apply(clean_ticker)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date.astype("string")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    if "volume" not in prices.columns:
        prices["volume"] = np.nan
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    prices["source_file"] = prices["source_file"].map(lambda x: as_text(x))
    prices = prices.dropna(subset=["date", "ticker", "close"])
    prices = prices[prices["ticker"] != ""].copy()
    prices = prices.sort_values(["ticker", "date", "source_file"]).drop_duplicates(["ticker", "date"], keep="last")
    return prices.reset_index(drop=True)


def build_candidate_queue() -> pd.DataFrame:
    queue = read_csv_safe(ROOT / "risk_book_repair_intake_queue.csv")
    if queue.empty or "ticker" not in queue.columns:
        gate = read_csv_safe(ROOT / "institutional_promotion_gate.csv")
        if gate.empty or "ticker" not in gate.columns:
            return pd.DataFrame()
        mask = gate.get("first_blocker", pd.Series("", index=gate.index)).astype(str).str.contains("risk book", case=False, na=False)
        queue = gate.loc[mask, ["ticker", "sector_or_theme", "first_blocker", "news_headline"]].copy()
        queue = queue.rename(columns={
            "first_blocker": "main_blocker",
            "news_headline": "news_or_event_hook",
        })
    queue = queue.copy()
    queue["ticker"] = queue["ticker"].apply(clean_ticker)
    queue = queue[queue["ticker"] != ""].drop_duplicates("ticker", keep="first")
    return queue.reset_index(drop=True)


def price_metrics(ticker: str, prices: pd.DataFrame, factor_returns: pd.DataFrame) -> dict[str, Any]:
    work = prices[prices["ticker"] == ticker].sort_values("date").copy()
    if work.empty:
        return {
            "price_data_status": "Missing price history",
            "price_rows": 0,
            "latest_price_date": "",
            "latest_close": np.nan,
            "annual_vol_pct": np.nan,
            "daily_var_95_pct": np.nan,
            "daily_cvar_95_pct": np.nan,
            "daily_var_99_pct": np.nan,
            "daily_cvar_99_pct": np.nan,
            "max_drawdown_1y_pct": np.nan,
            "avg_dollar_volume_20d": np.nan,
            "liquidity_status": "Needs manual liquidity proof",
            "corr_spy": np.nan,
            "corr_qqq": np.nan,
            "corr_smh": np.nan,
            "beta_spy": np.nan,
            "risk_level": "Data missing",
            "seed_cap_after_manual_approval_pct": 0.0,
            "paper_stop_if_ever_tested_pct": np.nan,
            "source_files": "price_repair_download_cache.csv; local price caches",
        }

    returns = work["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    tail_returns = returns.tail(252)
    latest = work.iloc[-1]
    if len(tail_returns) >= 40:
        annual_vol = float(tail_returns.std(ddof=0) * np.sqrt(252))
        losses = -tail_returns
        var95 = float(losses.quantile(0.95))
        cvar95 = float(losses[losses >= var95].mean()) if (losses >= var95).any() else var95
        var99 = float(losses.quantile(0.99))
        cvar99 = float(losses[losses >= var99].mean()) if (losses >= var99).any() else var99
        wealth = (1.0 + tail_returns).cumprod()
        max_dd = float((wealth / wealth.cummax() - 1.0).min())
        price_status = f"OK: {len(tail_returns)} daily returns"
    else:
        annual_vol = var95 = cvar95 = var99 = cvar99 = max_dd = np.nan
        price_status = f"Insufficient history: {len(tail_returns)} daily returns"

    adv = np.nan
    if "volume" in work.columns:
        vol_work = work.dropna(subset=["volume"]).tail(20).copy()
        if not vol_work.empty:
            adv = float((vol_work["close"] * vol_work["volume"]).mean())
    if np.isfinite(adv) and adv >= 2_000_000_000:
        liquidity = "Good"
    elif np.isfinite(adv) and adv >= 300_000_000:
        liquidity = "Usable"
    elif np.isfinite(adv) and adv >= 75_000_000:
        liquidity = "Manual review"
    else:
        liquidity = "Needs manual liquidity proof"

    ret_frame = work[["date", "close"]].copy()
    ret_frame["ticker_return"] = ret_frame["close"].pct_change()
    ret_frame = ret_frame[["date", "ticker_return"]].dropna()
    aligned = ret_frame.merge(factor_returns, on="date", how="inner").tail(252)

    def corr_col(col: str) -> float:
        if col not in aligned.columns or len(aligned) < 40:
            return np.nan
        return float(aligned["ticker_return"].corr(aligned[col]))

    corr_spy = corr_col("SPY")
    corr_qqq = corr_col("QQQ")
    corr_smh = corr_col("SMH")
    beta_spy = np.nan
    if "SPY" in aligned.columns and len(aligned) >= 40:
        spy_var = float(aligned["SPY"].var(ddof=0))
        if spy_var > 0:
            beta_spy = float(aligned["ticker_return"].cov(aligned["SPY"]) / spy_var)

    if not np.isfinite(cvar95):
        risk_level = "Data missing"
        cap = 0.0
    elif annual_vol >= 0.80 or cvar95 >= 0.075:
        risk_level = "Very high"
        cap = 0.25
    elif annual_vol >= 0.55 or cvar95 >= 0.055:
        risk_level = "High"
        cap = 0.50
    elif annual_vol >= 0.35 or cvar95 >= 0.040:
        risk_level = "Medium"
        cap = 0.75
    else:
        risk_level = "Lower"
        cap = 1.00

    stop = np.nan
    if np.isfinite(cvar95):
        stop = min(18.0, max(6.0, cvar95 * 100.0 * 1.6))

    return {
        "price_data_status": price_status,
        "price_rows": len(work),
        "latest_price_date": as_text(latest.get("date")),
        "latest_close": round(safe_float(latest.get("close"), np.nan), 4),
        "annual_vol_pct": pct(annual_vol),
        "daily_var_95_pct": pct(var95),
        "daily_cvar_95_pct": pct(cvar95),
        "daily_var_99_pct": pct(var99),
        "daily_cvar_99_pct": pct(cvar99),
        "max_drawdown_1y_pct": pct(max_dd),
        "avg_dollar_volume_20d": round(adv, 2) if np.isfinite(adv) else np.nan,
        "liquidity_status": liquidity,
        "corr_spy": round(corr_spy, 3) if np.isfinite(corr_spy) else np.nan,
        "corr_qqq": round(corr_qqq, 3) if np.isfinite(corr_qqq) else np.nan,
        "corr_smh": round(corr_smh, 3) if np.isfinite(corr_smh) else np.nan,
        "beta_spy": round(beta_spy, 3) if np.isfinite(beta_spy) else np.nan,
        "risk_level": risk_level,
        "seed_cap_after_manual_approval_pct": cap,
        "paper_stop_if_ever_tested_pct": round(stop, 2) if np.isfinite(stop) else np.nan,
        "source_files": "; ".join(sorted(set(work["source_file"].dropna().astype(str).tolist()))[:4]),
    }


def build_factor_returns(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker in FACTOR_TICKERS:
        work = prices[prices["ticker"] == ticker].sort_values("date").copy()
        if work.empty:
            continue
        work[ticker] = work["close"].pct_change()
        frames.append(work[["date", ticker]].dropna())
    if not frames:
        return pd.DataFrame(columns=["date"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = build_candidate_queue()
    prices = load_price_panel()
    factors = build_factor_returns(prices)
    news_proof = read_csv_safe(ROOT / "news_proof_repair_queue.csv")
    execution_proof = read_csv_safe(ROOT / "execution_spread_repair_queue.csv")

    news_tickers = set(news_proof.get("ticker", pd.Series(dtype=str)).dropna().map(clean_ticker).tolist()) if not news_proof.empty else set()
    execution_tickers = set(execution_proof.get("ticker", pd.Series(dtype=str)).dropna().map(clean_ticker).tolist()) if not execution_proof.empty else set()

    metric_rows: list[dict[str, Any]] = []
    entry_rows: list[dict[str, Any]] = []
    approval_rows: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        metrics = price_metrics(ticker, prices, factors)
        sector = as_text(row.get("sector_or_theme"), "Unknown")
        event_hook = short(row.get("news_or_event_hook"), 240)
        manual_items = [
            "Human PM approval of this seed",
            "Earnings date and gap-risk source",
            "Current spread / liquidity snapshot",
            "Sector and factor crowding check",
        ]
        if ticker in news_tickers:
            manual_items.append("News causal proof")
        if ticker in execution_tickers or metrics["liquidity_status"] != "Good":
            manual_items.append("Execution / spread proof")
        if metrics["risk_level"] in {"Data missing", "Very high"}:
            manual_items.append("Extra downside scenario note")

        cap = safe_float(metrics["seed_cap_after_manual_approval_pct"], 0.0)
        cvar95 = safe_float(metrics["daily_cvar_95_pct"], np.nan)
        if metrics["risk_level"] == "Very high":
            single_name_action = "SIZE_DOWN"
        elif metrics["risk_level"] == "Data missing":
            single_name_action = "REVIEW"
        else:
            single_name_action = "REVIEW"

        entry_rows.append({
            "ticker": ticker,
            "sector": sector,
            "current_action": "RESEARCH",
            "current_weight": 0.0,
            "current_weight_pct": 0.0,
            "master_risk_action": "SEED_REVIEW_ONLY",
            "single_name_action": single_name_action,
            "earnings_gap_action": "MANUAL_REVIEW",
            "kelly_status": "NO_KELLY_UNTIL_LIVE_IC",
            "liquidity_crisis_status": "MANUAL_REVIEW",
            "sector_status": "MANUAL_REVIEW",
            "final_risk_action": "SEED_REVIEW_ONLY",
            "max_allowed_weight": 0.0,
            "recommended_risk_weight": 0.0,
            "recommended_risk_weight_pct": 0.0,
            "risk_reduction_pct_of_current": 0.0,
            "seed_cap_after_manual_approval_pct": cap,
            "paper_stop_if_ever_tested_pct": metrics["paper_stop_if_ever_tested_pct"],
            "daily_cvar_95_pct": cvar95,
            "risk_level": metrics["risk_level"],
            "liquidity_status": metrics["liquidity_status"],
            "reason_stack": (
                "Provisional risk-book seed exists; manual PM approval, event proof, "
                "earnings gap proof, and execution proof still required."
            ),
            "source_file": "risk_book_seed_entries.csv; risk_book_seed_metric_detail.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        metric_rows.append({
            "ticker": ticker,
            "sector_or_theme": sector,
            "event_hook": event_hook,
            **metrics,
            "manual_items_open": "; ".join(manual_items),
            "source_files": metrics["source_files"],
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        approval_rows.append({
            "priority": "P1" if metrics["risk_level"] in {"Very high", "Data missing"} or ticker in news_tickers else "P2",
            "ticker": ticker,
            "sector_or_theme": sector,
            "risk_seed_status": "Seed created; manual approval required",
            "risk_level": metrics["risk_level"],
            "starter_cap_if_approved_pct": cap,
            "paper_stop_if_ever_tested_pct": metrics["paper_stop_if_ever_tested_pct"],
            "manual_items_open": "; ".join(manual_items),
            "done_when": "PM approves the seed, earnings gap proof is sourced, spread/liquidity is checked, and Final PM Gate no longer shows risk review as first blocker.",
            "still_forbidden": "No paper size, no calls, no puts, and no live orders from this seed alone.",
            "source_files": "risk_book_seed_entries.csv; risk_book_seed_metric_detail.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    entries = pd.DataFrame(entry_rows)
    metrics_df = pd.DataFrame(metric_rows)
    approval = pd.DataFrame(approval_rows)

    sector_preview = pd.DataFrame()
    if not metrics_df.empty:
        sector_preview = (
            metrics_df.groupby("sector_or_theme", dropna=False)
            .agg(
                seed_ticker_count=("ticker", "count"),
                avg_seed_cap_after_manual_approval_pct=("seed_cap_after_manual_approval_pct", "mean"),
                high_or_very_high_count=("risk_level", lambda s: int(s.astype(str).isin(["High", "Very high"]).sum())),
                missing_liquidity_count=("liquidity_status", lambda s: int(s.astype(str).str.contains("Needs manual", case=False, na=False).sum())),
            )
            .reset_index()
        )
        sector_preview["avg_seed_cap_after_manual_approval_pct"] = sector_preview["avg_seed_cap_after_manual_approval_pct"].round(3)
        sector_preview = sector_preview.sort_values(["seed_ticker_count", "sector_or_theme"], ascending=[False, True])

    state = {
        "date": today_str(),
        "status": "RISK_BOOK_SEED_ACTIVE",
        "seed_entry_count": len(entries),
        "manual_approval_count": len(approval),
        "high_priority_approval_count": int((approval.get("priority", pd.Series(dtype=str)).astype(str) == "P1").sum()) if not approval.empty else 0,
        "seed_with_price_metrics_count": int(metrics_df.get("price_data_status", pd.Series(dtype=str)).astype(str).str.startswith("OK").sum()) if not metrics_df.empty else 0,
        "very_high_risk_count": int((metrics_df.get("risk_level", pd.Series(dtype=str)).astype(str) == "Very high").sum()) if not metrics_df.empty else 0,
        "plain_answer": (
            f"Risk-book seed is active. {len(entries)} provisional risk entries were created. "
            "They are not approvals; they only replace a blank risk-book gap with a review-only risk seed."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return entries, metrics_df, approval, sector_preview, state


def maybe_rerun_gate_and_reliability() -> str:
    notes: list[str] = []
    try:
        import canyon_final_v9_step195_institutional_promotion_gate as step195
        step195.main()
        notes.append("Step195 rerun after risk-book seed.")
    except Exception as exc:
        notes.append(f"Step195 rerun skipped: {type(exc).__name__}: {short(exc, 180)}")
    try:
        import canyon_final_v9_step196_decision_memory_center as step196
        step196.main()
        notes.append("Step196 rerun after risk-book seed.")
    except Exception as exc:
        notes.append(f"Step196 rerun skipped: {type(exc).__name__}: {short(exc, 180)}")
    try:
        import canyon_final_v9_step197_price_data_reliability_center as step197
        step197.main()
        notes.append("Step197 rerun after risk-book seed.")
    except Exception as exc:
        notes.append(f"Step197 rerun skipped: {type(exc).__name__}: {short(exc, 180)}")
    return " ".join(notes)


def write_history(current: pd.DataFrame, path, date_col: str = "last_seen_date") -> pd.DataFrame:
    if current.empty or "ticker" not in current.columns:
        return current.copy()
    current_work = current.copy()
    current_work["ticker"] = current_work["ticker"].apply(clean_ticker)
    current_work[date_col] = today_str()
    current_work["active_in_current_seed"] = "Yes"

    existing = read_csv_safe(path)
    if not existing.empty and "ticker" in existing.columns:
        existing = existing.copy()
        existing["ticker"] = existing["ticker"].apply(clean_ticker)
        existing["active_in_current_seed"] = "No"
        combined = pd.concat([existing, current_work], ignore_index=True, sort=False)
    else:
        combined = current_work
    combined = combined[combined["ticker"] != ""].drop_duplicates("ticker", keep="last").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def main() -> None:
    entries, metrics_df, approval, sector_preview, state = build_outputs()
    entries.to_csv(OUT_ENTRIES, index=False)
    metrics_df.to_csv(OUT_METRICS, index=False)
    approval.to_csv(OUT_APPROVAL, index=False)
    sector_preview.to_csv(OUT_SECTOR, index=False)
    entries_history = write_history(entries, OUT_ENTRIES_HISTORY)
    metrics_history = write_history(metrics_df, OUT_METRICS_HISTORY)

    rerun_note = maybe_rerun_gate_and_reliability()
    state["rerun_note"] = rerun_note
    state["seed_entry_history_count"] = len(entries_history)
    state["seed_metric_history_count"] = len(metrics_history)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Seed Entries\n\n" + df_to_markdown(entries.head(120)),
        "## Risk Metrics\n\n" + df_to_markdown(metrics_df.head(120)),
        "## Manual Approval Queue\n\n" + df_to_markdown(approval.head(120)),
        "## Sector Exposure Preview\n\n" + df_to_markdown(sector_preview),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 199 - Risk Book Seed Engine", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Provisional risk-book seed entries: {state['seed_entry_count']}")
    print(f"[OK] Manual approvals still required: {state['manual_approval_count']}")
    print(f"[OK] {rerun_note}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
