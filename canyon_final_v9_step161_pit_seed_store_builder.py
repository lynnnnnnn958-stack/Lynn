#!/usr/bin/env python3
"""
Canyon v9 Step 161 - PIT Seed Store Builder.

Research-only. No broker connection. No live orders.

This step creates the missing point-in-time store skeletons that Step159 asks
for. The output is intentionally labeled LOCAL_SEED, not vendor-grade PIT data.
It gives downstream modules a stable schema while preserving the product truth:
these files support research plumbing, not institutional historical proof yet.

Outputs:
  point_in_time_prices.csv
  corporate_actions.csv
  universe_membership_history.csv
  delisted_tickers.csv
  pit_fundamentals.csv
  pit_store_build_audit.csv
  pit_store_state.json
  pit_store_report.md
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
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


OUT_PRICES = ROOT / "point_in_time_prices.csv"
OUT_ACTIONS = ROOT / "corporate_actions.csv"
OUT_MEMBERSHIP = ROOT / "universe_membership_history.csv"
OUT_DELISTED = ROOT / "delisted_tickers.csv"
OUT_FUNDAMENTALS = ROOT / "pit_fundamentals.csv"
OUT_AUDIT = ROOT / "pit_store_build_audit.csv"
OUT_STATE = ROOT / "pit_store_state.json"
OUT_REPORT = ROOT / "pit_store_report.md"

LOCAL_QUALITY = "LOCAL_SEED_NOT_VENDOR_PIT"
LOCAL_VENDOR = "LOCAL_YFINANCE_OR_MODEL_PROXY"
MODEL_READ_TIME = datetime.now().replace(microsecond=0).isoformat()


def file_sha(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_date_col(df: pd.DataFrame) -> str | None:
    for col in ["Date", "date", "as_of", "Unnamed: 0", "index"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > max(2, len(df) * 0.5):
                return col
    if len(df.columns) > 0:
        col = df.columns[0]
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > max(2, len(df) * 0.5):
            return col
    return None


def load_price_matrix() -> tuple[pd.DataFrame, str]:
    for name in ["backtest_price_cache.csv", "sp500_price_cache.csv", "regime_price_cache.csv"]:
        df = read_csv_safe(ROOT / name)
        if df.empty:
            continue
        date_col = detect_date_col(df)
        if date_col is None:
            continue
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).set_index(date_col).sort_index()
        work.columns = [clean_ticker(c) for c in work.columns]
        work = work.loc[:, [c for c in work.columns if c and c != clean_ticker(date_col)]]
        work = work.apply(pd.to_numeric, errors="coerce")
        return work, name
    return pd.DataFrame(), ""


def build_price_store() -> pd.DataFrame:
    prices, source = load_price_matrix()
    if prices.empty:
        return pd.DataFrame(columns=[
            "price_date", "as_of_time", "model_read_time", "ticker", "raw_close",
            "adjusted_close", "adjustment_status", "source_vendor", "source_file",
            "pit_quality_status", "can_support_current_research",
            "can_support_institutional_backtest", "limitation",
        ])
    long = (
        prices.reset_index()
        .melt(id_vars=[prices.index.name or "index"], var_name="ticker", value_name="adjusted_close")
        .dropna(subset=["adjusted_close"])
    )
    date_col = long.columns[0]
    long = long.rename(columns={date_col: "price_date"})
    long["price_date"] = pd.to_datetime(long["price_date"], errors="coerce").dt.date.astype(str)
    long["as_of_time"] = pd.to_datetime(long["price_date"], errors="coerce").dt.strftime("%Y-%m-%dT23:59:59")
    long["model_read_time"] = MODEL_READ_TIME
    long["ticker"] = long["ticker"].map(clean_ticker)
    long["raw_close"] = np.nan
    long["adjusted_close"] = pd.to_numeric(long["adjusted_close"], errors="coerce")
    long["adjustment_status"] = "ADJUSTED_CLOSE_FROM_LOCAL_CACHE_RAW_CLOSE_NOT_STORED"
    long["source_vendor"] = LOCAL_VENDOR
    long["source_file"] = source
    long["pit_quality_status"] = LOCAL_QUALITY
    long["can_support_current_research"] = True
    long["can_support_institutional_backtest"] = False
    long["limitation"] = "Local adjusted-close cache. No vendor as-of proof, raw quote, bid/ask, or complete corporate-action trace."
    cols = [
        "price_date", "as_of_time", "model_read_time", "ticker", "raw_close",
        "adjusted_close", "adjustment_status", "source_vendor", "source_file",
        "pit_quality_status", "can_support_current_research",
        "can_support_institutional_backtest", "limitation",
    ]
    return long[cols].sort_values(["price_date", "ticker"]).reset_index(drop=True)


def build_universe_membership(prices: pd.DataFrame) -> pd.DataFrame:
    sources: list[pd.DataFrame] = []
    sector_map = read_csv_safe(ROOT / "sector_map.csv")
    if not sector_map.empty and "ticker" in sector_map.columns:
        tmp = sector_map.copy()
        tmp["ticker"] = tmp["ticker"].map(clean_ticker)
        sources.append(tmp[[c for c in ["ticker", "sector", "source"] if c in tmp.columns]])
    alpha = read_csv_safe(ROOT / "alpha_scores.csv")
    if not alpha.empty and "ticker" in alpha.columns:
        tmp = alpha[["ticker"] + (["sector"] if "sector" in alpha.columns else [])].copy()
        tmp["ticker"] = tmp["ticker"].map(clean_ticker)
        tmp["source"] = "alpha_scores.csv"
        sources.append(tmp)

    tickers = pd.DataFrame(columns=["ticker", "sector", "source"])
    if sources:
        tickers = pd.concat(sources, ignore_index=True, sort=False)
    if not prices.empty:
        price_dates = pd.to_datetime(prices["price_date"], errors="coerce") if "price_date" in prices.columns else pd.Series(dtype="datetime64[ns]")
        coverage = (
            prices.groupby("ticker", dropna=False)
            .agg(first_price_date=("price_date", "min"), last_price_date=("price_date", "max"), price_observations=("adjusted_close", "count"))
            .reset_index()
        )
    else:
        coverage = pd.DataFrame(columns=["ticker", "first_price_date", "last_price_date", "price_observations"])

    if tickers.empty and not coverage.empty:
        tickers = coverage[["ticker"]].copy()
    tickers["ticker"] = tickers["ticker"].map(clean_ticker)
    tickers = tickers[tickers["ticker"] != ""].drop_duplicates("ticker", keep="first")
    out = tickers.merge(coverage, on="ticker", how="outer")
    if "sector" not in out.columns:
        out["sector"] = ""
    out["membership_start"] = out.get("first_price_date", "").fillna("")
    out["membership_end"] = out.get("last_price_date", "").fillna("")
    out["membership_source"] = out.get("source", "").fillna("local_price_or_model_cache")
    out["membership_type"] = "LOCAL_COVERAGE_PROXY_NOT_TRUE_INDEX_MEMBERSHIP"
    out["model_read_time"] = MODEL_READ_TIME
    out["source_vendor"] = LOCAL_VENDOR
    out["source_file"] = "sector_map.csv / alpha_scores.csv / point_in_time_prices.csv"
    out["pit_quality_status"] = LOCAL_QUALITY
    out["can_support_current_research"] = True
    out["can_support_institutional_backtest"] = False
    out["limitation"] = "This is local coverage history, not official historical index membership or survivorship-corrected universe history."
    cols = [
        "ticker", "sector", "membership_start", "membership_end", "price_observations",
        "membership_type", "membership_source", "model_read_time", "source_vendor",
        "source_file", "pit_quality_status", "can_support_current_research",
        "can_support_institutional_backtest", "limitation",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values("ticker").reset_index(drop=True)


def build_corporate_actions(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in sorted(set(tickers)):
        rows.append({
            "ticker": ticker,
            "action_date": "",
            "as_of_time": "",
            "model_read_time": MODEL_READ_TIME,
            "action_type": "UNKNOWN_NO_VENDOR_TRACE",
            "split_ratio": np.nan,
            "cash_dividend": np.nan,
            "adjustment_factor": np.nan,
            "source_vendor": LOCAL_VENDOR,
            "source_file": "local price cache only",
            "pit_quality_status": LOCAL_QUALITY,
            "has_complete_action_trace": False,
            "can_support_current_research": True,
            "can_support_institutional_backtest": False,
            "limitation": "No official split/dividend/ticker-change trace. Adjusted prices may already include actions, but the action chain is not auditable.",
        })
    return pd.DataFrame(rows)


def build_delisted(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in sorted(set(tickers)):
        rows.append({
            "ticker": ticker,
            "delisted_flag": "UNKNOWN_NOT_VENDOR_CHECKED",
            "delist_date": "",
            "final_trade_date": "",
            "final_price": np.nan,
            "successor_ticker": "",
            "model_read_time": MODEL_READ_TIME,
            "source_vendor": LOCAL_VENDOR,
            "source_file": "local coverage universe",
            "pit_quality_status": LOCAL_QUALITY,
            "can_support_current_research": True,
            "can_support_institutional_backtest": False,
            "limitation": "No vendor delisting/dead-ticker table. Current local universe may still have survivorship bias.",
        })
    return pd.DataFrame(rows)


def build_fundamentals() -> pd.DataFrame:
    src = read_csv_safe(ROOT / "fundamental_features.csv")
    if src.empty:
        return pd.DataFrame(columns=[
            "ticker", "as_of_time", "report_period_end", "model_read_time",
            "source_vendor", "source_file", "pit_quality_status",
            "can_support_current_research", "can_support_institutional_backtest",
            "limitation",
        ])
    out = src.copy()
    out["ticker"] = out["ticker"].map(clean_ticker) if "ticker" in out.columns else ""
    if "as_of" in out.columns:
        out["as_of_time"] = pd.to_datetime(out["as_of"], errors="coerce").dt.strftime("%Y-%m-%dT23:59:59")
    else:
        out["as_of_time"] = MODEL_READ_TIME
    out["report_period_end"] = ""
    out["source_vendor"] = LOCAL_VENDOR
    out["source_file"] = "fundamental_features.csv"
    out["model_read_time"] = MODEL_READ_TIME
    out["pit_quality_status"] = LOCAL_QUALITY
    out["can_support_current_research"] = True
    out["can_support_institutional_backtest"] = False
    out["limitation"] = "Current/local fundamental snapshot. No original report timestamp, restatement history, or vendor as-of validation."
    front = [
        "ticker", "as_of_time", "report_period_end", "model_read_time",
        "source_vendor", "source_file", "pit_quality_status",
        "can_support_current_research", "can_support_institutional_backtest",
        "limitation",
    ]
    rest = [c for c in out.columns if c not in front and c != "as_of"]
    return out[front + rest].sort_values("ticker").reset_index(drop=True)


def audit_file(path: Path, label: str, source: str, rows: int, institutional_ready: bool) -> dict[str, Any]:
    return {
        "file": path.name,
        "control": label,
        "rows": int(rows),
        "exists": path.exists() and path.stat().st_size > 0,
        "size_kb": round(path.stat().st_size / 1024.0, 2) if path.exists() else 0.0,
        "source": source,
        "pit_quality_status": LOCAL_QUALITY,
        "can_support_current_research": True,
        "can_support_institutional_backtest": bool(institutional_ready),
        "content_hash": file_sha(path),
        "required_next_action": "Replace local seed with vendor-grade PIT feed and keep immutable daily snapshots.",
    }


def write_outputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    prices = build_price_store()
    prices.to_csv(OUT_PRICES, index=False)

    membership = build_universe_membership(prices)
    membership.to_csv(OUT_MEMBERSHIP, index=False)

    ticker_set = set(membership.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
    if not prices.empty and "ticker" in prices.columns:
        ticker_set.update(prices["ticker"].dropna().astype(str).tolist())

    actions = build_corporate_actions(list(ticker_set))
    actions.to_csv(OUT_ACTIONS, index=False)

    delisted = build_delisted(list(ticker_set))
    delisted.to_csv(OUT_DELISTED, index=False)

    fundamentals = build_fundamentals()
    fundamentals.to_csv(OUT_FUNDAMENTALS, index=False)

    audit = pd.DataFrame([
        audit_file(OUT_PRICES, "PIT price seed", "backtest_price_cache.csv / local cache", len(prices), False),
        audit_file(OUT_ACTIONS, "Corporate action seed", "local ticker list", len(actions), False),
        audit_file(OUT_MEMBERSHIP, "Universe membership seed", "sector_map.csv / alpha_scores.csv / local coverage", len(membership), False),
        audit_file(OUT_DELISTED, "Delisted ticker seed", "local ticker list", len(delisted), False),
        audit_file(OUT_FUNDAMENTALS, "PIT fundamentals seed", "fundamental_features.csv", len(fundamentals), False),
    ])
    audit.to_csv(OUT_AUDIT, index=False)

    state = {
        "date": today_str(),
        "generated_at": MODEL_READ_TIME,
        "overall_status": "LOCAL_PIT_SEED_READY_NOT_VENDOR_VALIDATED",
        "seed_files_created": int(len(audit)),
        "price_rows": int(len(prices)),
        "membership_rows": int(len(membership)),
        "fundamental_rows": int(len(fundamentals)),
        "institutional_ready_files": 0,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "PIT seed stores create schema and local evidence only. They are not vendor-grade point-in-time databases and should not be used to claim institutional backtest validity.",
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Seed files created: **{state['seed_files_created']}**",
        f"- Price rows: **{state['price_rows']}**",
        f"- Membership rows: **{state['membership_rows']}**",
        f"- Fundamental rows: **{state['fundamental_rows']}**",
        "",
        state["truth"],
        "",
        "## Build Audit",
        "",
        df_to_markdown(audit, max_rows=20),
        "",
        "## Required Replacement",
        "",
        "- Replace `point_in_time_prices.csv` with vendor historical raw/adjusted prices and model-read timestamps.",
        "- Replace `corporate_actions.csv` with official split/dividend/ticker-change adjustment trace.",
        "- Replace `universe_membership_history.csv` with historical membership by date, including additions/removals.",
        "- Replace `delisted_tickers.csv` with dead ticker and final trade handling.",
        "- Replace `pit_fundamentals.csv` with report-time fundamentals, restatements, and revisions.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 161 - PIT Seed Store Builder", sections)
    return audit, state


def main() -> None:
    audit, state = write_outputs()
    print("Canyon v9 Step161 PIT seed store builder complete.")
    print(f"Overall: {state.get('overall_status')}")
    print(f"Seed files: {state.get('seed_files_created')} | price rows: {state.get('price_rows')} | fundamentals: {state.get('fundamental_rows')}")
    print(f"Outputs: {OUT_PRICES.name}, {OUT_MEMBERSHIP.name}, {OUT_ACTIONS.name}, {OUT_DELISTED.name}, {OUT_FUNDAMENTALS.name}")


if __name__ == "__main__":
    main()
