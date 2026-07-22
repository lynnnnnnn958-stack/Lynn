#!/usr/bin/env python3
"""
Canyon v9 Step 196 - Decision Memory / Forward Validation Center.

Research-only. No broker connection. No live orders.

This step makes the system remember its own calls and later grade them.
It does not pretend a decision was right before future prices exist.

It implements:
1. Decision History Ledger
2. Forward Return Check
3. False Positive / False Negative Lab
4. Gate Calibration
5. Plain-English Review Cards

Outputs:
  decision_memory_state.json
  decision_history_ledger.csv
  decision_forward_return_check.csv
  decision_false_positive_negative_lab.csv
  decision_gate_calibration.csv
  decision_memory_review_cards.csv
  decision_memory_report.md
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


OUT_STATE = ROOT / "decision_memory_state.json"
OUT_LEDGER = ROOT / "decision_history_ledger.csv"
OUT_FORWARD = ROOT / "decision_forward_return_check.csv"
OUT_FALSE_LAB = ROOT / "decision_false_positive_negative_lab.csv"
OUT_CALIBRATION = ROOT / "decision_gate_calibration.csv"
OUT_CARDS = ROOT / "decision_memory_review_cards.csv"
OUT_REPORT = ROOT / "decision_memory_report.md"

HORIZONS = [1, 5, 21, 63]


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


def shorten(value: Any, limit: int = 260) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def plain(value: Any) -> str:
    text = as_text(value, "No data")
    replacements = {
        "DATA_GAP": "missing data",
        "REDUCE_ONLY": "reduce only",
        "SIZE_DOWN": "use smaller size",
        "NOT_IN_RISK_BOOK_REVIEW": "not in risk book; needs review",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "put or hedge research only",
        "CALL_RESEARCH_ONLY": "call research only",
        "TINY_STOCK_OR_ETF_PAPER_ONLY": "tiny stock or ETF paper only",
        "NO_GO": "not allowed",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    return " ".join(text.replace("_", " ").split())


def load_price_panel() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    pit = read_csv_safe(ROOT / "point_in_time_prices.csv")
    if not pit.empty and {"price_date", "ticker", "adjusted_close"}.issubset(pit.columns):
        work = pit[["price_date", "ticker", "adjusted_close"]].copy()
        work = work.rename(columns={"price_date": "date", "adjusted_close": "close"})
        frames.append(work)

    monitor = read_csv_safe(ROOT / "desk_monitor_price_volume_cache.csv")
    if not monitor.empty and {"date", "ticker", "close"}.issubset(monitor.columns):
        frames.append(monitor[["date", "ticker", "close"]].copy())

    repair = read_csv_safe(ROOT / "price_repair_download_cache.csv")
    if not repair.empty and {"date", "ticker", "close"}.issubset(repair.columns):
        frames.append(repair[["date", "ticker", "close"]].copy())

    for filename in ["backtest_price_cache.csv", "sp500_price_cache.csv"]:
        wide = read_csv_safe(ROOT / filename)
        if wide.empty:
            continue
        date_col = "Unnamed: 0" if "Unnamed: 0" in wide.columns else wide.columns[0]
        wide = wide.rename(columns={date_col: "date"})
        long = wide.melt(id_vars=["date"], var_name="ticker", value_name="close")
        frames.append(long)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    prices = pd.concat(frames, ignore_index=True)
    prices["ticker"] = prices["ticker"].apply(clean_ticker)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date.astype("string")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["date", "ticker", "close"])
    prices = prices[prices["ticker"] != ""].copy()
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return prices.reset_index(drop=True)


def price_lookup_maps(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if prices.empty:
        return {}
    return {ticker: grp.sort_values("date").reset_index(drop=True) for ticker, grp in prices.groupby("ticker", sort=False)}


def price_at_or_before(price_map: dict[str, pd.DataFrame], ticker: str, date_text: str) -> tuple[str, float]:
    ticker = clean_ticker(ticker)
    grp = price_map.get(ticker)
    if grp is None or grp.empty:
        return "", np.nan
    date_text = str(date_text)
    eligible = grp[grp["date"].astype(str) <= date_text]
    if eligible.empty:
        return "", np.nan
    row = eligible.iloc[-1]
    return as_text(row.get("date")), safe_float(row.get("close"))


def forward_price(price_map: dict[str, pd.DataFrame], ticker: str, entry_price_date: str, trading_days: int) -> tuple[str, float, str]:
    ticker = clean_ticker(ticker)
    grp = price_map.get(ticker)
    if grp is None or grp.empty or not entry_price_date:
        return "", np.nan, "No price history for ticker"
    dates = grp["date"].astype(str).tolist()
    if entry_price_date not in dates:
        return "", np.nan, "Entry price date not in price history"
    entry_idx = dates.index(entry_price_date)
    target_idx = entry_idx + int(trading_days)
    if target_idx >= len(grp):
        return "", np.nan, "Waiting for future price"
    row = grp.iloc[target_idx]
    return as_text(row.get("date")), safe_float(row.get("close")), "Ready"


def build_today_snapshot(gate: pd.DataFrame, price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if gate.empty:
        return pd.DataFrame()

    today = today_str()
    rows: list[dict[str, Any]] = []
    for _, row in gate.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        entry_date, entry_price = price_at_or_before(price_map, ticker, today)
        decision_id = f"{today}|{ticker}"
        rows.append({
            "decision_id": decision_id,
            "decision_date": today,
            "ticker": ticker,
            "final_permission": as_text(row.get("final_permission"), "Study only"),
            "primary_route_now": as_text(row.get("primary_route_now"), "Research only"),
            "confidence_0_100": safe_float(row.get("confidence_0_100"), np.nan),
            "max_paper_weight_pct": safe_float(row.get("max_paper_weight_pct"), 0.0),
            "first_blocker": as_text(row.get("first_blocker"), "No blocker recorded"),
            "why_now": shorten(row.get("why_now"), 360),
            "next_step": shorten(row.get("next_step"), 320),
            "where_to_click": as_text(row.get("where_to_click"), "Home"),
            "risk_status": as_text(row.get("risk_status"), "No data"),
            "news_headline": shorten(row.get("news_headline"), 260),
            "trigger_to_watch": shorten(row.get("trigger_to_watch"), 220),
            "entry_price_date": entry_date,
            "entry_price": round(entry_price, 4) if np.isfinite(entry_price) else np.nan,
            "price_status": "Entry price found" if np.isfinite(entry_price) else "Entry price missing",
            "source_files": as_text(row.get("source_files"), "institutional_promotion_gate.csv"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def upsert_ledger(today_snapshot: pd.DataFrame) -> pd.DataFrame:
    old = read_csv_safe(OUT_LEDGER)
    if old.empty:
        ledger = today_snapshot.copy()
    else:
        ledger = pd.concat([old, today_snapshot], ignore_index=True)
        ledger = ledger.drop_duplicates(["decision_id"], keep="last")
    if not ledger.empty:
        ledger = ledger.sort_values(["decision_date", "ticker"]).reset_index(drop=True)
    return ledger


def build_forward_check(ledger: pd.DataFrame, price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if ledger.empty:
        return pd.DataFrame()
    for _, row in ledger.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        entry_price = safe_float(row.get("entry_price"), np.nan)
        entry_date = as_text(row.get("entry_price_date"))
        for horizon in HORIZONS:
            eval_date, eval_price, status_note = forward_price(price_map, ticker, entry_date, horizon)
            if np.isfinite(entry_price) and np.isfinite(eval_price) and entry_price > 0:
                ret = (eval_price / entry_price) - 1.0
                status = "Ready"
            else:
                ret = np.nan
                status = "Pending" if "Waiting" in status_note else "No price"
            rows.append({
                "decision_id": as_text(row.get("decision_id")),
                "decision_date": as_text(row.get("decision_date")),
                "ticker": ticker,
                "final_permission": as_text(row.get("final_permission")),
                "first_blocker": as_text(row.get("first_blocker")),
                "horizon_days": horizon,
                "entry_price_date": entry_date,
                "entry_price": entry_price,
                "eval_price_date": eval_date,
                "eval_price": eval_price,
                "forward_return_pct": round(ret * 100.0, 3) if np.isfinite(ret) else np.nan,
                "observation_status": status,
                "status_note": status_note,
            })
    return pd.DataFrame(rows)


def classify_outcome(permission: str, ret_pct: float, horizon_days: int) -> tuple[str, str]:
    permission_lower = str(permission or "").lower()
    if not np.isfinite(ret_pct):
        return "Waiting", "Future price is not available yet."

    threshold_up = 2.0 if horizon_days <= 5 else 5.0
    threshold_down = -2.0 if horizon_days <= 5 else -5.0

    if "do not add" in permission_lower:
        if ret_pct <= threshold_down:
            return "Good block", "The gate avoided a later loss."
        if ret_pct >= threshold_up:
            return "Possible missed winner", "The gate blocked a name that later rose."
        return "Neutral block", "The later move was not large enough to judge."

    if "tiny" in permission_lower:
        if ret_pct <= threshold_down:
            return "Good size control", "Tiny sizing would have reduced damage."
        if ret_pct >= threshold_up:
            return "Maybe too conservative", "The idea rose after being kept tiny."
        return "Neutral tiny review", "The later move was not decisive."

    if "study" in permission_lower:
        if ret_pct <= threshold_down:
            return "Good caution", "Study-only avoided a weak forward move."
        if ret_pct >= threshold_up:
            return "Possible under-promotion", "Study-only may have missed a strong move."
        return "Still inconclusive", "The later move was small."

    if ret_pct >= threshold_up:
        return "Helpful watch", "The watched name improved."
    if ret_pct <= threshold_down:
        return "Bad watch", "The watched name fell."
    return "Neutral watch", "The later move was small."


def build_false_lab(forward: pd.DataFrame) -> pd.DataFrame:
    if forward.empty:
        return pd.DataFrame()
    ready = forward.copy()
    rows: list[dict[str, Any]] = []
    for _, row in ready.iterrows():
        ret = safe_float(row.get("forward_return_pct"), np.nan)
        label, note = classify_outcome(row.get("final_permission"), ret, safe_int(row.get("horizon_days"), 0))
        rows.append({
            "decision_id": as_text(row.get("decision_id")),
            "ticker": clean_ticker(row.get("ticker")),
            "decision_date": as_text(row.get("decision_date")),
            "final_permission": as_text(row.get("final_permission")),
            "first_blocker": as_text(row.get("first_blocker")),
            "horizon_days": safe_int(row.get("horizon_days")),
            "forward_return_pct": ret,
            "outcome_label": label,
            "plain_read": note,
            "observation_status": as_text(row.get("observation_status")),
        })
    return pd.DataFrame(rows)


def build_calibration(false_lab: pd.DataFrame) -> pd.DataFrame:
    if false_lab.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = false_lab.groupby(["first_blocker", "final_permission"], dropna=False)
    for (blocker, permission), grp in grouped:
        ready = grp[grp["observation_status"] == "Ready"].copy()
        pending = grp[grp["observation_status"] != "Ready"].copy()
        if ready.empty:
            rows.append({
                "gate_or_blocker": as_text(blocker, "Unknown"),
                "permission": as_text(permission, "Unknown"),
                "ready_observations": 0,
                "pending_observations": len(pending),
                "avg_forward_return_pct": np.nan,
                "missed_winner_count": 0,
                "protected_loss_count": 0,
                "calibration_action": "Collect more forward observations",
                "plain_reason": "No forward price window has matured yet. Do not change this gate from zero evidence.",
            })
            continue
        missed = int(ready["outcome_label"].astype(str).str.contains("missed|under-promotion|too conservative", case=False, regex=True).sum())
        protected = int(ready["outcome_label"].astype(str).str.contains("Good block|Good caution|Good size", case=False, regex=True).sum())
        avg_ret = safe_float(ready["forward_return_pct"].mean(), np.nan)
        if missed >= max(3, protected * 2):
            action = "Review if this gate is too strict"
            reason = "Several blocked or under-promoted names later rose. Review thresholds, but do not loosen from one case."
        elif protected >= max(3, missed * 2):
            action = "Keep or strengthen this gate"
            reason = "This gate repeatedly avoided later losses."
        else:
            action = "Keep collecting evidence"
            reason = "The matured observations are mixed or too small."
        rows.append({
            "gate_or_blocker": as_text(blocker, "Unknown"),
            "permission": as_text(permission, "Unknown"),
            "ready_observations": len(ready),
            "pending_observations": len(pending),
            "avg_forward_return_pct": round(avg_ret, 3) if np.isfinite(avg_ret) else np.nan,
            "missed_winner_count": missed,
            "protected_loss_count": protected,
            "calibration_action": action,
            "plain_reason": reason,
        })
    return pd.DataFrame(rows).sort_values(["ready_observations", "pending_observations"], ascending=[False, False]).reset_index(drop=True)


def build_review_cards(ledger: pd.DataFrame, forward: pd.DataFrame, false_lab: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    total_decisions = len(ledger)
    obs_status = forward.get("observation_status", pd.Series(dtype=str)).astype(str) if not forward.empty else pd.Series(dtype=str)
    ready_obs = int((obs_status == "Ready").sum()) if not forward.empty else 0
    pending_obs = int((obs_status == "Pending").sum()) if not forward.empty else 0
    no_price_obs = int((obs_status == "No price").sum()) if not forward.empty else 0
    possible_missed = int(false_lab.get("outcome_label", pd.Series(dtype=str)).astype(str).str.contains("missed|under-promotion|too conservative", case=False, regex=True).sum()) if not false_lab.empty else 0
    good_blocks = int(false_lab.get("outcome_label", pd.Series(dtype=str)).astype(str).str.contains("Good block|Good caution|Good size", case=False, regex=True).sum()) if not false_lab.empty else 0

    rows = [
        {
            "card": "What did the system say?",
            "answer": f"{total_decisions} ticker decisions are stored in the memory ledger.",
            "why_it_matters": "Without a ledger, the system can keep changing its mind without accountability.",
            "next_step": "Keep appending one daily snapshot after every Final PM Gate run.",
            "source_files": "decision_history_ledger.csv",
        },
        {
            "card": "What happened after?",
            "answer": f"{ready_obs} forward observations are ready; {pending_obs} are waiting; {no_price_obs} lack price history.",
            "why_it_matters": "A decision cannot be graded until the future window actually exists.",
            "next_step": "Refresh prices over time, then rerun Step196.",
            "source_files": "decision_forward_return_check.csv",
        },
        {
            "card": "Was the system too strict?",
            "answer": f"{possible_missed} possible missed-winner observation(s) so far.",
            "why_it_matters": "If blocked names repeatedly rally, the blocker may be too conservative.",
            "next_step": "Review only after multiple matured examples, not from one anecdote.",
            "source_files": "decision_false_positive_negative_lab.csv",
        },
        {
            "card": "Did risk protect us?",
            "answer": f"{good_blocks} good block / caution observation(s) so far.",
            "why_it_matters": "If blocked names later fall, the risk gate is doing its job.",
            "next_step": "Keep gates that repeatedly protect downside.",
            "source_files": "decision_false_positive_negative_lab.csv",
        },
        {
            "card": "What should change?",
            "answer": "Do not recalibrate yet." if calibration.empty or int(calibration.get("ready_observations", pd.Series([0])).sum()) == 0 else "Use matured calibration rows.",
            "why_it_matters": "Institutions do not tune gates from pending or missing data.",
            "next_step": "Wait for 1d/5d/21d/63d windows to mature, then review gate-level evidence.",
            "source_files": "decision_gate_calibration.csv",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    gate = read_csv_safe(ROOT / "institutional_promotion_gate.csv")
    prices = load_price_panel()
    price_map = price_lookup_maps(prices)
    today_snapshot = build_today_snapshot(gate, price_map)
    ledger = upsert_ledger(today_snapshot)
    forward = build_forward_check(ledger, price_map)
    false_lab = build_false_lab(forward)
    calibration = build_calibration(false_lab)
    cards = build_review_cards(ledger, forward, false_lab, calibration)

    ledger.to_csv(OUT_LEDGER, index=False)
    forward.to_csv(OUT_FORWARD, index=False)
    false_lab.to_csv(OUT_FALSE_LAB, index=False)
    calibration.to_csv(OUT_CALIBRATION, index=False)
    cards.to_csv(OUT_CARDS, index=False)

    obs_status = forward.get("observation_status", pd.Series(dtype=str)).astype(str) if not forward.empty else pd.Series(dtype=str)
    ready_obs = int((obs_status == "Ready").sum()) if not forward.empty else 0
    pending_obs = int((obs_status == "Pending").sum()) if not forward.empty else 0
    no_price_obs = int((obs_status == "No price").sum()) if not forward.empty else 0
    state = {
        "date": today_str(),
        "status": "DECISION_MEMORY_ACTIVE",
        "ledger_decision_count": len(ledger),
        "today_snapshot_count": len(today_snapshot),
        "forward_observation_rows": len(forward),
        "ready_forward_observations": ready_obs,
        "pending_forward_observations": pending_obs,
        "no_price_observations": no_price_obs,
        "calibration_rows": len(calibration),
        "latest_price_date": str(prices["date"].max()) if not prices.empty else "",
        "plain_answer": (
            "Decision memory is active. Today's calls are stored, but most forward grades are pending until future prices exist."
            if ready_obs == 0
            else "Decision memory is active and some forward grades are ready."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Review Cards\n\n" + df_to_markdown(cards),
        "## Decision History Ledger\n\n" + df_to_markdown(ledger.tail(30)),
        "## Forward Return Check\n\n" + df_to_markdown(forward.tail(80)),
        "## False Positive / False Negative Lab\n\n" + df_to_markdown(false_lab.tail(80)),
        "## Gate Calibration\n\n" + df_to_markdown(calibration.head(40)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 196 - Decision Memory Center", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Ledger decisions: {len(ledger)}")
    print(f"[OK] Ready forward observations: {ready_obs} | Pending: {pending_obs}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
