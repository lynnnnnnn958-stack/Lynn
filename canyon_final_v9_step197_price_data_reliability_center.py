#!/usr/bin/env python3
"""
Canyon v9 Step 197 - Price / Data Reliability Center.

Research-only. No broker connection. No live orders.

This step answers a practical PM question:
"Can I trust the local data enough to read today's dashboard, and what must be
fixed before the system can grade itself?"

It does not upgrade any ticker. Missing or stale data can only create repair
work.

Outputs:
  data_reliability_state.json
  price_refresh_desk.csv
  forward_validation_unlocker.csv
  data_gap_repair_queue.csv
  data_reliability_scorecard.csv
  data_reliability_report.md
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


OUT_STATE = ROOT / "data_reliability_state.json"
OUT_PRICE_DESK = ROOT / "price_refresh_desk.csv"
OUT_FORWARD_UNLOCKER = ROOT / "forward_validation_unlocker.csv"
OUT_REPAIR_QUEUE = ROOT / "data_gap_repair_queue.csv"
OUT_SCORECARD = ROOT / "data_reliability_scorecard.csv"
OUT_REPORT = ROOT / "data_reliability_report.md"

FRESH_ENOUGH_DAYS = 2


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


def safe_int(value: Any, default: int = 0) -> int:
    out = safe_float(value, np.nan)
    if not np.isfinite(out):
        return default
    return int(out)


def shorten(value: Any, limit: int = 220) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def plain(value: Any, default: str = "No data") -> str:
    text = as_text(value, default)
    replacements = {
        "DATA_GAP": "missing data",
        "REDUCE_ONLY": "reduce only",
        "SIZE_DOWN": "use smaller size",
        "NO_GO": "not allowed",
        "P1_REVIEW_CONTRADICTION": "urgent contradiction review",
        "CONTRADICTED_REVIEW_REQUIRED": "contradiction review required",
        "PRICE_DISAGREES": "price reaction disagrees",
        "LOCAL_PRICE_PROXY_OK": "local price proxy found",
        "LOCAL_SEED_NOT_VENDOR_PIT": "local seed; not vendor point-in-time data",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    return " ".join(text.replace("_", " ").split())


def parse_date(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def days_between(left: Any, right: Any) -> float:
    left_dt = parse_date(left)
    right_dt = parse_date(right)
    if pd.isna(left_dt) or pd.isna(right_dt):
        return np.nan
    return float((right_dt.normalize() - left_dt.normalize()).days)


def load_price_panel() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    pit = read_csv_safe(ROOT / "point_in_time_prices.csv")
    if not pit.empty and {"price_date", "ticker", "adjusted_close"}.issubset(pit.columns):
        work = pit[["price_date", "ticker", "adjusted_close", "source_file", "pit_quality_status"]].copy()
        work = work.rename(columns={
            "price_date": "date",
            "adjusted_close": "close",
            "source_file": "price_source_file",
            "pit_quality_status": "source_quality",
        })
        work["source_table"] = "point_in_time_prices.csv"
        frames.append(work)

    monitor = read_csv_safe(ROOT / "desk_monitor_price_volume_cache.csv")
    if not monitor.empty and {"date", "ticker", "close"}.issubset(monitor.columns):
        cols = [c for c in ["date", "ticker", "close", "source_file"] if c in monitor.columns]
        work = monitor[cols].copy()
        work = work.rename(columns={"source_file": "price_source_file"})
        work["source_quality"] = "monitor cache"
        work["source_table"] = "desk_monitor_price_volume_cache.csv"
        frames.append(work)

    repair = read_csv_safe(ROOT / "price_repair_download_cache.csv")
    if not repair.empty and {"date", "ticker", "close"}.issubset(repair.columns):
        cols = [c for c in ["date", "ticker", "close", "source_file"] if c in repair.columns]
        work = repair[cols].copy()
        work = work.rename(columns={"source_file": "price_source_file"})
        work["source_quality"] = "public yfinance repair cache"
        work["source_table"] = "price_repair_download_cache.csv"
        frames.append(work)

    for filename in ["backtest_price_cache.csv", "sp500_price_cache.csv"]:
        wide = read_csv_safe(ROOT / filename)
        if wide.empty:
            continue
        date_col = "Unnamed: 0" if "Unnamed: 0" in wide.columns else wide.columns[0]
        wide = wide.rename(columns={date_col: "date"})
        long = wide.melt(id_vars=["date"], var_name="ticker", value_name="close")
        long["price_source_file"] = filename
        long["source_quality"] = "local adjusted close cache"
        long["source_table"] = filename
        frames.append(long)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "price_source_file", "source_quality", "source_table"])

    prices = pd.concat(frames, ignore_index=True)
    prices["ticker"] = prices["ticker"].apply(clean_ticker)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date.astype("string")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    for col in ["price_source_file", "source_quality", "source_table"]:
        if col not in prices.columns:
            prices[col] = ""
        prices[col] = prices[col].map(lambda x: as_text(x))
    prices = prices.dropna(subset=["date", "ticker", "close"])
    prices = prices[prices["ticker"] != ""].copy()
    prices = prices.sort_values(["ticker", "date", "source_table"]).drop_duplicates(["ticker", "date"], keep="last")
    return prices.reset_index(drop=True)


def build_universe() -> list[str]:
    tickers: set[str] = set()
    sources = [
        ("institutional_promotion_gate.csv", "ticker"),
        ("decision_history_ledger.csv", "ticker"),
        ("final_risk_gate.csv", "ticker"),
        ("event_causal_validation_queue.csv", "target_ticker"),
    ]
    for filename, col in sources:
        df = read_csv_safe(ROOT / filename)
        if df.empty or col not in df.columns:
            continue
        for value in df[col].dropna().tolist():
            ticker = clean_ticker(value)
            if ticker:
                tickers.add(ticker)
    return sorted(tickers)


def latest_price_rows(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=[
            "ticker", "latest_price_date", "latest_price", "price_source_file",
            "source_quality", "source_table",
        ])
    work = prices.sort_values(["ticker", "date"]).drop_duplicates("ticker", keep="last")
    return work.rename(columns={"date": "latest_price_date", "close": "latest_price"}).reset_index(drop=True)


def build_price_refresh_desk(tickers: list[str], prices: pd.DataFrame) -> pd.DataFrame:
    latest = latest_price_rows(prices)
    latest_map = latest.set_index("ticker").to_dict("index") if not latest.empty else {}
    snapshot = read_csv_safe(ROOT / "market_data_snapshot.csv")
    snapshot_map = snapshot.set_index("ticker").to_dict("index") if not snapshot.empty and "ticker" in snapshot.columns else {}
    today = today_str()

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        price_row = latest_map.get(ticker, {})
        snap = snapshot_map.get(ticker, {})
        latest_date = as_text(price_row.get("latest_price_date"))
        latest_price = safe_float(price_row.get("latest_price"), np.nan)
        days_stale = days_between(latest_date, today)
        if not latest_date or not np.isfinite(latest_price):
            price_status = "Missing"
            can_validate = "No"
            next_step = "Add a usable price history row before judging any signal for this ticker."
        elif np.isfinite(days_stale) and days_stale <= FRESH_ENOUGH_DAYS:
            price_status = "Fresh enough"
            can_validate = "Yes"
            next_step = "No price repair needed for current research. Future windows still need time to mature."
        else:
            price_status = "Stale"
            can_validate = "No"
            next_step = "Refresh local price cache before treating today's read as current."

        source_files = "; ".join(
            [part for part in [
                as_text(price_row.get("source_table")),
                as_text(price_row.get("price_source_file")),
                as_text(snap.get("price_source_file")),
            ] if part]
        )
        rows.append({
            "ticker": ticker,
            "latest_price_date": latest_date,
            "latest_price": round(latest_price, 4) if np.isfinite(latest_price) else np.nan,
            "days_stale_vs_today": round(days_stale, 1) if np.isfinite(days_stale) else np.nan,
            "price_status": price_status,
            "can_validate_forward": can_validate,
            "source_quality": plain(price_row.get("source_quality"), "No price source"),
            "market_snapshot_confidence": plain(snap.get("data_confidence"), "No local market snapshot"),
            "next_step": next_step,
            "source_files": source_files if source_files else "No price file found",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).sort_values(["price_status", "ticker"]).reset_index(drop=True)


def build_forward_unlocker(price_desk: pd.DataFrame) -> pd.DataFrame:
    forward = read_csv_safe(ROOT / "decision_forward_return_check.csv")
    if forward.empty:
        return pd.DataFrame(columns=[
            "ticker", "horizon_days", "observations", "ready_count", "pending_count",
            "no_price_count", "unlock_status", "what_is_needed", "source_files",
        ])

    rows: list[dict[str, Any]] = []
    price_status_map = {}
    latest_date_map = {}
    if not price_desk.empty and "ticker" in price_desk.columns:
        price_status_map = dict(zip(price_desk["ticker"], price_desk["price_status"]))
        latest_date_map = dict(zip(price_desk["ticker"], price_desk["latest_price_date"]))

    for (ticker, horizon), grp in forward.groupby(["ticker", "horizon_days"], dropna=False):
        ticker = clean_ticker(ticker)
        status = grp.get("observation_status", pd.Series(dtype=str)).astype(str)
        ready = int((status == "Ready").sum())
        pending = int((status == "Pending").sum())
        no_price = int((status == "No price").sum())
        if ready == len(grp) and len(grp) > 0:
            unlock = "Ready to grade"
            needed = "This horizon has enough future price rows to judge."
        elif no_price > 0 or price_status_map.get(ticker) == "Missing":
            unlock = "Needs price history"
            needed = "Add missing historical prices for this ticker."
        else:
            unlock = "Waiting for future window"
            needed = f"Latest local price is {as_text(latest_date_map.get(ticker), 'unknown')}; wait until the {safe_int(horizon)} trading-day window exists."

        rows.append({
            "ticker": ticker,
            "horizon_days": safe_int(horizon),
            "observations": len(grp),
            "ready_count": ready,
            "pending_count": pending,
            "no_price_count": no_price,
            "unlock_status": unlock,
            "what_is_needed": needed,
            "source_files": "decision_forward_return_check.csv; price_refresh_desk.csv",
        })

    return pd.DataFrame(rows).sort_values(["unlock_status", "ticker", "horizon_days"]).reset_index(drop=True)


def add_repair(rows: list[dict[str, Any]], severity: str, ticker: str, repair_type: str, problem: str, why: str, next_step: str, owner_page: str, source_file: str) -> None:
    rows.append({
        "severity": severity,
        "ticker": clean_ticker(ticker) or "Portfolio",
        "repair_type": repair_type,
        "plain_problem": plain(problem),
        "why_it_matters": why,
        "next_step": next_step,
        "owner_page": owner_page,
        "source_files": source_file,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    })


def build_repair_queue(price_desk: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not price_desk.empty:
        for _, row in price_desk.iterrows():
            status = as_text(row.get("price_status"))
            if status == "Missing":
                add_repair(
                    rows,
                    "High",
                    row.get("ticker"),
                    "Price history",
                    "No usable local price history.",
                    "Forward grading, risk checks, and trigger distance are unreliable without a price.",
                    "Refresh or seed the local price cache for this ticker.",
                    "System",
                    as_text(row.get("source_files"), "price_refresh_desk.csv"),
                )
            elif status == "Stale":
                add_repair(
                    rows,
                    "Medium",
                    row.get("ticker"),
                    "Price freshness",
                    f"Latest price is {row.get('days_stale_vs_today')} days old.",
                    "A stale price can make the dashboard look calm when the market already moved.",
                    "Refresh local prices before using today's dashboard as current.",
                    "System",
                    as_text(row.get("source_files"), "price_refresh_desk.csv"),
                )

    gate = read_csv_safe(ROOT / "institutional_promotion_gate.csv")
    if not gate.empty and {"ticker", "first_blocker"}.issubset(gate.columns):
        mask = gate["first_blocker"].astype(str).str.contains("risk book", case=False, na=False)
        for _, row in gate[mask].iterrows():
            add_repair(
                rows,
                "High",
                row.get("ticker"),
                "Risk book coverage",
                row.get("first_blocker"),
                "A ticker outside the risk book cannot be promoted safely.",
                "Open Risk / Ideas and add a risk-book entry before any paper sizing.",
                "Risk",
                "institutional_promotion_gate.csv",
            )

    seed_approval = read_csv_safe(ROOT / "risk_book_seed_manual_approval_queue.csv")
    if not seed_approval.empty and "ticker" in seed_approval.columns:
        for _, row in seed_approval.head(120).iterrows():
            add_repair(
                rows,
                "High" if as_text(row.get("priority")).upper() == "P1" else "Medium",
                row.get("ticker"),
                "Risk seed approval",
                row.get("risk_seed_status"),
                "A provisional risk seed is better than a blank risk book, but it is not permission to size.",
                shorten(row.get("manual_items_open"), 260) or "Approve the seed, earnings gap, liquidity, sector/factor crowding, and stop rule.",
                "Risk",
                "risk_book_seed_manual_approval_queue.csv",
            )

    event_queue = read_csv_safe(ROOT / "event_causal_validation_queue.csv")
    if not event_queue.empty and "target_ticker" in event_queue.columns:
        for _, row in event_queue.head(80).iterrows():
            severity = "High" if str(row.get("priority", "")).upper().startswith("P1") else "Medium"
            add_repair(
                rows,
                severity,
                row.get("target_ticker"),
                "News proof",
                row.get("issue"),
                "News can only affect sizing after source, timing, target, and price reaction are checked.",
                shorten(row.get("required_next_action"), 240) or "Validate source, timing, affected ticker, and post-news price reaction.",
                "News",
                "event_causal_validation_queue.csv",
            )

    execution = read_csv_safe(ROOT / "depth5_execution_liquidity_desk.csv")
    if not execution.empty and "ticker" in execution.columns:
        permission = execution.get("execution_permission", pd.Series("", index=execution.index)).astype(str)
        status = execution.get("execution_status", pd.Series("", index=execution.index)).astype(str)
        mask = permission.str.contains("manual|no new exposure|risk reduction", case=False, na=False) | status.str.contains("manual|spread|liquidity|risk", case=False, na=False)
        for _, row in execution[mask].head(80).iterrows():
            add_repair(
                rows,
                "High" if "no new exposure" in str(row.get("execution_permission", "")).lower() else "Medium",
                row.get("ticker"),
                "Execution and liquidity",
                f"{row.get('execution_permission')} / {row.get('execution_status')}",
                "A good signal can disappear if spread, liquidity, or fill quality is bad.",
                shorten(row.get("what_to_do"), 240) or "Check spread, volume, and realistic fill assumptions manually.",
                "Risk",
                "depth5_execution_liquidity_desk.csv",
            )

    option_route = read_csv_safe(ROOT / "options_execution_route_matrix.csv")
    if not option_route.empty and {"ticker", "spread_status"}.issubset(option_route.columns):
        mask = option_route["spread_status"].astype(str).str.contains("DATA_GAP|missing", case=False, na=False)
        for _, row in option_route[mask].head(80).iterrows():
            add_repair(
                rows,
                "Medium",
                row.get("ticker"),
                "Option spread data",
                row.get("spread_status"),
                "Option route cannot be trusted without spread and liquidity proof.",
                "Manually check option spread, IV, liquidity, and no-go conditions before any option research route.",
                "Ideas",
                "options_execution_route_matrix.csv",
            )

    if not rows:
        return pd.DataFrame(columns=[
            "severity", "ticker", "repair_type", "plain_problem", "why_it_matters",
            "next_step", "owner_page", "source_files", "research_only",
            "no_broker_connection", "no_live_orders",
        ])

    severity_rank = {"High": 0, "Medium": 1, "Low": 2}
    out = pd.DataFrame(rows)
    out["_rank"] = out["severity"].map(severity_rank).fillna(9)
    out = out.sort_values(["_rank", "repair_type", "ticker"]).drop(columns=["_rank"])
    out = out.drop_duplicates(["ticker", "repair_type", "plain_problem"], keep="first")
    return out.reset_index(drop=True)


def score_component(component: str, score: float, status: str, why: str, next_step: str, source_files: str) -> dict[str, Any]:
    score = max(0.0, min(100.0, float(score))) if np.isfinite(score) else 0.0
    return {
        "score_component": component,
        "score_0_100": round(score, 1),
        "plain_status": status,
        "why_it_matters": why,
        "next_step": next_step,
        "source_files": source_files,
    }


def build_scorecard(price_desk: pd.DataFrame, forward_unlocker: pd.DataFrame, repair_queue: pd.DataFrame) -> pd.DataFrame:
    ticker_count = max(len(price_desk), 1)
    price_status = price_desk.get("price_status", pd.Series(dtype=str)).astype(str) if not price_desk.empty else pd.Series(dtype=str)
    fresh_count = int((price_status == "Fresh enough").sum())
    missing_count = int((price_status == "Missing").sum())
    stale_count = int((price_status == "Stale").sum())
    price_score = fresh_count / ticker_count * 100.0

    if forward_unlocker.empty:
        ready_count = 0
        observation_count = 0
    else:
        ready_count = int(pd.to_numeric(forward_unlocker.get("ready_count", 0), errors="coerce").fillna(0).sum())
        observation_count = int(pd.to_numeric(forward_unlocker.get("observations", 0), errors="coerce").fillna(0).sum())
    forward_score = ready_count / observation_count * 100.0 if observation_count else 0.0

    repair_type = repair_queue.get("repair_type", pd.Series(dtype=str)).astype(str) if not repair_queue.empty else pd.Series(dtype=str)
    risk_gaps = int((repair_type == "Risk book coverage").sum())
    event_gaps = int((repair_type == "News proof").sum())
    execution_gaps = int(repair_type.isin(["Execution and liquidity", "Option spread data"]).sum())
    risk_score = max(0.0, 100.0 - (risk_gaps / ticker_count * 100.0))
    event_score = max(0.0, 100.0 - (event_gaps / ticker_count * 100.0))
    execution_score = max(0.0, 100.0 - (execution_gaps / ticker_count * 100.0))

    rows = [
        score_component(
            "Price freshness",
            price_score,
            f"{fresh_count} fresh, {stale_count} stale, {missing_count} missing.",
            "The dashboard can only be current when local prices are recent.",
            "Refresh stale/missing tickers first.",
            "price_refresh_desk.csv",
        ),
        score_component(
            "Forward validation readiness",
            forward_score,
            f"{ready_count} of {observation_count} forward observations are ready.",
            "The system cannot prove whether a decision worked until future prices exist.",
            "Let future windows mature; refresh prices before rerunning Decision Memory.",
            "forward_validation_unlocker.csv",
        ),
        score_component(
            "Risk book coverage",
            risk_score,
            f"{risk_gaps} ticker(s) still need risk-book coverage.",
            "A ticker outside the risk book cannot get a serious position decision.",
            "Add missing risk-book entries before promotion.",
            "data_gap_repair_queue.csv",
        ),
        score_component(
            "News proof coverage",
            event_score,
            f"{event_gaps} event proof item(s) still need validation.",
            "News is not alpha until source, timing, target, and price reaction are proven.",
            "Clear the highest-priority News proof items.",
            "event_causal_validation_queue.csv; data_gap_repair_queue.csv",
        ),
        score_component(
            "Execution and spread coverage",
            execution_score,
            f"{execution_gaps} execution/spread item(s) need repair or manual proof.",
            "A strategy can look profitable before costs and liquidity are applied.",
            "Fix spread, liquidity, and option route data before any paper action.",
            "depth5_execution_liquidity_desk.csv; options_execution_route_matrix.csv",
        ),
    ]
    return pd.DataFrame(rows)


def build_state(price_desk: pd.DataFrame, forward_unlocker: pd.DataFrame, repair_queue: pd.DataFrame, scorecard: pd.DataFrame) -> dict[str, Any]:
    ticker_count = len(price_desk)
    price_status = price_desk.get("price_status", pd.Series(dtype=str)).astype(str) if not price_desk.empty else pd.Series(dtype=str)
    stale_count = int((price_status == "Stale").sum())
    missing_count = int((price_status == "Missing").sum())
    fresh_count = int((price_status == "Fresh enough").sum())
    latest_price_date = ""
    if not price_desk.empty and "latest_price_date" in price_desk.columns:
        latest_price_date = as_text(price_desk["latest_price_date"].dropna().astype(str).max())

    ready_forward = int(pd.to_numeric(forward_unlocker.get("ready_count", 0), errors="coerce").fillna(0).sum()) if not forward_unlocker.empty else 0
    pending_forward = int(pd.to_numeric(forward_unlocker.get("pending_count", 0), errors="coerce").fillna(0).sum()) if not forward_unlocker.empty else 0
    no_price_forward = int(pd.to_numeric(forward_unlocker.get("no_price_count", 0), errors="coerce").fillna(0).sum()) if not forward_unlocker.empty else 0
    high_repairs = int((repair_queue.get("severity", pd.Series(dtype=str)).astype(str) == "High").sum()) if not repair_queue.empty else 0

    weights = {
        "Price freshness": 0.30,
        "Forward validation readiness": 0.15,
        "Risk book coverage": 0.20,
        "News proof coverage": 0.20,
        "Execution and spread coverage": 0.15,
    }
    overall = 0.0
    if not scorecard.empty:
        for _, row in scorecard.iterrows():
            overall += weights.get(as_text(row.get("score_component")), 0.0) * safe_float(row.get("score_0_100"), 0.0)
    overall = round(overall, 1)

    if missing_count or high_repairs:
        status = "Repair required"
    elif stale_count:
        status = "Refresh recommended"
    elif ready_forward == 0 and pending_forward:
        status = "Research-usable; waiting for future proof"
    else:
        status = "Usable for research"

    plain_answer = (
        f"Data reliability is active. {fresh_count} of {ticker_count} tickers have fresh-enough local prices; "
        f"{stale_count} are stale and {missing_count} are missing. "
        f"{ready_forward} forward checks are ready, {pending_forward} are waiting for future prices, "
        f"and {no_price_forward} need price history."
    )

    return {
        "date": today_str(),
        "status": status,
        "overall_score_0_100": overall,
        "ticker_count": ticker_count,
        "fresh_price_count": fresh_count,
        "stale_price_count": stale_count,
        "missing_price_count": missing_count,
        "latest_price_date": latest_price_date,
        "ready_forward_observations": ready_forward,
        "pending_forward_observations": pending_forward,
        "no_price_observations": no_price_forward,
        "repair_queue_rows": len(repair_queue),
        "high_priority_repairs": high_repairs,
        "plain_answer": plain_answer,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def main() -> None:
    tickers = build_universe()
    prices = load_price_panel()
    price_desk = build_price_refresh_desk(tickers, prices)
    forward_unlocker = build_forward_unlocker(price_desk)
    repair_queue = build_repair_queue(price_desk)
    scorecard = build_scorecard(price_desk, forward_unlocker, repair_queue)
    state = build_state(price_desk, forward_unlocker, repair_queue, scorecard)

    price_desk.to_csv(OUT_PRICE_DESK, index=False)
    forward_unlocker.to_csv(OUT_FORWARD_UNLOCKER, index=False)
    repair_queue.to_csv(OUT_REPAIR_QUEUE, index=False)
    scorecard.to_csv(OUT_SCORECARD, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Scorecard\n\n" + df_to_markdown(scorecard),
        "## Price Refresh Desk\n\n" + df_to_markdown(price_desk.head(80)),
        "## Forward Validation Unlocker\n\n" + df_to_markdown(forward_unlocker.head(120)),
        "## Data Gap Repair Queue\n\n" + df_to_markdown(repair_queue.head(120)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 197 - Price / Data Reliability Center", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Overall reliability score: {state['overall_score_0_100']}/100")
    print(f"[OK] Fresh prices: {state['fresh_price_count']} | stale: {state['stale_price_count']} | missing: {state['missing_price_count']}")
    print(f"[OK] Forward ready: {state['ready_forward_observations']} | pending: {state['pending_forward_observations']} | no price: {state['no_price_observations']}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
