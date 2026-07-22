#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 47 — L1 Data & Universe Integrity Layer

Purpose:
Make L1 real. This layer audits local data files and creates:
- canyon_file_manifest.csv
- universe_master.csv
- market_data_snapshot.csv
- data_quality_flags.csv
- data_quality_report.md

It does NOT fetch market data.
It does NOT trade.
It does NOT connect to broker.

L1 answers:
- Which files exist?
- Which tickers are covered?
- Which source produced each ticker?
- Are files stale?
- Are important fields missing?
- Is the universe traceable?
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import os

ROOT = Path.cwd()
NOW = datetime.now()

OUT_MANIFEST = ROOT / "canyon_file_manifest.csv"
OUT_UNIVERSE = ROOT / "universe_master.csv"
OUT_MARKET = ROOT / "market_data_snapshot.csv"
OUT_FLAGS = ROOT / "data_quality_flags.csv"
OUT_REPORT = ROOT / "data_quality_report.md"

IMPORTANT_FILES = [
    "technical_signal_matrix.csv",
    "sector_rotation_scores.csv",
    "fundamental_quality_valuation.csv",
    "options_decision_matrix.csv",
    "watch_triggers.csv",
    "action_cards.csv",
    "gamma_squeeze_candidates.csv",
    "option_kill_zone_risk.csv",
    "pre_trade_checklist.csv",
    "execution_gate_review.csv",
    "paper_portfolio_ledger.csv",
    "position_sizing_recommendations.csv",
    "exposure_warnings.csv",
    "scenario_stress_results.csv",
    "learning_attribution_summary.csv",
    "learning_weight_suggestions.csv",
]

TICKER_COL_CANDIDATES = ["ticker", "symbol", "asset", "underlying"]
PRICE_COL_CANDIDATES = [
    "spot", "close", "last", "last_price", "current_price", "price",
    "entry_price", "exit_price", "call_wall_breakout_trigger",
]
DATE_COL_CANDIDATES = [
    "generated_at", "created_at", "updated_at", "date", "entry_date",
    "exit_date", "timestamp", "asof", "as_of",
]

ETF_SET = {
    "SPY", "QQQ", "IWM", "DIA", "XLK", "SMH", "SOXX", "XLE", "XLF", "XLV",
    "XLI", "XLY", "XLP", "XLU", "IYR", "TLT", "GLD", "SLV", "UUP", "HYG",
    "LQD", "VNQ", "KRE", "ARKK", "XBI", "KWEB",
}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def safe_num(x):
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def file_age_hours(path: Path) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return (NOW - mtime).total_seconds() / 3600
    except Exception:
        return np.nan


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def list_relevant_files() -> list[Path]:
    files = []
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix.lower() in {".csv", ".md", ".txt", ".py"}:
            files.append(p)
    return sorted(files, key=lambda x: x.name)


def build_manifest() -> pd.DataFrame:
    rows = []
    for p in list_relevant_files():
        ext = p.suffix.lower()
        size = p.stat().st_size if p.exists() else 0
        age = file_age_hours(p)
        row_count = ""
        col_count = ""
        columns = ""
        tickers = ""
        ticker_count = ""
        status = "OK"

        if ext == ".csv":
            df = read_csv(p)
            row_count = len(df)
            col_count = len(df.columns)
            columns = ", ".join(df.columns[:30])
            tcol = find_col(df, TICKER_COL_CANDIDATES)
            if tcol:
                ts = sorted(set(df[tcol].astype(str).str.upper().str.strip()) - {""})
                tickers = ", ".join(ts[:80])
                ticker_count = len(ts)

            if df.empty:
                status = "WARN_EMPTY_CSV"
            elif tcol is None and p.name in IMPORTANT_FILES:
                status = "WARN_NO_TICKER_COL"

        if age > 48 and p.name in IMPORTANT_FILES:
            status = "WARN_STALE_IMPORTANT_FILE"

        rows.append({
            "file_name": p.name,
            "extension": ext,
            "size_bytes": size,
            "modified_at": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "age_hours": round(age, 2) if np.isfinite(age) else "",
            "row_count": row_count,
            "column_count": col_count,
            "ticker_count": ticker_count,
            "tickers_sample": tickers,
            "columns_sample": columns,
            "status": status,
        })

    return pd.DataFrame(rows)


def collect_universe() -> pd.DataFrame:
    rows = []
    for fname in IMPORTANT_FILES:
        p = ROOT / fname
        if not p.exists() or p.suffix.lower() != ".csv":
            continue

        df = read_csv(p)
        if df.empty:
            continue

        tcol = find_col(df, TICKER_COL_CANDIDATES)
        if not tcol:
            continue

        for ticker in sorted(set(df[tcol].astype(str).str.upper().str.strip()) - {""}):
            sub = df[df[tcol].astype(str).str.upper().str.strip() == ticker]
            rows.append({
                "ticker": ticker,
                "asset_type_guess": "ETF" if ticker in ETF_SET else "STOCK_OR_OTHER",
                "source_file": fname,
                "source_rows": len(sub),
                "file_modified_at": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "file_age_hours": round(file_age_hours(p), 2),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "asset_type_guess", "source_files", "source_count",
            "total_rows", "freshest_source_age_hours", "status"
        ])

    raw = pd.DataFrame(rows)
    grouped = raw.groupby(["ticker", "asset_type_guess"], as_index=False).agg(
        source_files=("source_file", lambda x: ", ".join(sorted(set(x)))),
        source_count=("source_file", lambda x: len(set(x))),
        total_rows=("source_rows", "sum"),
        freshest_source_age_hours=("file_age_hours", "min"),
    )

    def status(row):
        if row["source_count"] >= 4:
            return "GOOD_MULTI_SOURCE_LOCAL"
        if row["source_count"] >= 2:
            return "OK_MULTI_SOURCE_LOCAL"
        return "WARN_SINGLE_SOURCE_LOCAL"

    grouped["status"] = grouped.apply(status, axis=1)
    return grouped.sort_values(["status", "ticker"])


def build_market_snapshot(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(columns=[
            "ticker", "best_price_proxy", "price_source_file", "price_column",
            "source_age_hours", "data_confidence", "notes"
        ])

    rows = []
    price_sources = [
        "technical_signal_matrix.csv",
        "sector_rotation_scores.csv",
        "fundamental_quality_valuation.csv",
        "options_decision_matrix.csv",
        "watch_triggers.csv",
        "action_cards.csv",
        "gamma_squeeze_candidates.csv",
        "option_kill_zone_risk.csv",
        "pre_trade_checklist.csv",
        "paper_portfolio_ledger.csv",
        "position_sizing_recommendations.csv",
    ]

    for ticker in universe["ticker"].tolist():
        best = np.nan
        best_file = ""
        best_col = ""
        best_age = np.nan
        notes = []

        for fname in price_sources:
            p = ROOT / fname
            if not p.exists():
                continue
            df = read_csv(p)
            if df.empty:
                continue
            tcol = find_col(df, TICKER_COL_CANDIDATES)
            if not tcol:
                continue

            sub = df[df[tcol].astype(str).str.upper().str.strip() == ticker]
            if sub.empty:
                continue

            for pc in PRICE_COL_CANDIDATES:
                if pc in sub.columns:
                    vals = sub[pc].map(safe_num)
                    vals = vals[vals.notna()]
                    vals = vals[vals > 0]
                    if not vals.empty:
                        best = float(vals.iloc[0])
                        best_file = fname
                        best_col = pc
                        best_age = file_age_hours(p)
                        break
            if np.isfinite(best):
                break

        if not np.isfinite(best):
            confidence = "NO_PRICE_PROXY"
            notes.append("No usable local price/spot/close field found.")
        elif best_age > 48:
            confidence = "STALE_PRICE_PROXY"
            notes.append("Price proxy source older than 48h.")
        else:
            confidence = "LOCAL_PRICE_PROXY_OK"
            notes.append("Local price proxy found; still not live broker data.")

        rows.append({
            "ticker": ticker,
            "best_price_proxy": round(best, 4) if np.isfinite(best) else "",
            "price_source_file": best_file,
            "price_column": best_col,
            "source_age_hours": round(best_age, 2) if np.isfinite(best_age) else "",
            "data_confidence": confidence,
            "notes": " ".join(notes),
        })

    return pd.DataFrame(rows).sort_values(["data_confidence", "ticker"])


def build_flags(manifest: pd.DataFrame, universe: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    flags = []

    for fname in IMPORTANT_FILES:
        p = ROOT / fname
        if not p.exists():
            flags.append({
                "level": "WARN",
                "area": "missing_file",
                "item": fname,
                "message": f"Important file missing: {fname}",
            })

    for _, r in manifest.iterrows():
        if str(r.get("status", "")).startswith("WARN"):
            flags.append({
                "level": "WARN",
                "area": "file_manifest",
                "item": r.get("file_name", ""),
                "message": r.get("status", ""),
            })

    if not universe.empty:
        singles = universe[universe["status"] == "WARN_SINGLE_SOURCE_LOCAL"]
        for _, r in singles.iterrows():
            flags.append({
                "level": "INFO",
                "area": "universe",
                "item": r["ticker"],
                "message": "Ticker appears in only one local source file.",
            })

    if not market.empty:
        bad = market[market["data_confidence"].isin(["NO_PRICE_PROXY", "STALE_PRICE_PROXY"])]
        for _, r in bad.iterrows():
            flags.append({
                "level": "WARN",
                "area": "market_snapshot",
                "item": r["ticker"],
                "message": f"{r['data_confidence']}: {r['notes']}",
            })

    return pd.DataFrame(flags, columns=["level", "area", "item", "message"])


def md_table(df: pd.DataFrame, max_rows=80) -> str:
    if df.empty:
        return "_No data._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def build_report(manifest, universe, market, flags) -> str:
    md = []
    md.append("# Canyon v9 Step 47 — L1 Data & Universe Integrity Report")
    md.append("")
    md.append(f"Generated: {NOW.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## What L1 means")
    md.append("")
    md.append("L1 does not try to predict price. It checks whether the system is using traceable local data.")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Files audited: **{len(manifest)}**")
    md.append(f"- Universe tickers: **{len(universe)}**")
    md.append(f"- Market snapshot rows: **{len(market)}**")
    md.append(f"- Data flags: **{len(flags)}**")
    md.append("")
    md.append("## Data Quality Flags")
    md.append("")
    md.append(md_table(flags, max_rows=100))
    md.append("")
    md.append("## Universe Master")
    md.append("")
    md.append(md_table(universe, max_rows=100))
    md.append("")
    md.append("## Market Data Snapshot")
    md.append("")
    md.append(md_table(market, max_rows=100))
    md.append("")
    md.append("## Important file manifest")
    md.append("")
    important_manifest = manifest[manifest["file_name"].isin(IMPORTANT_FILES)].copy()
    md.append(md_table(important_manifest, max_rows=100))
    md.append("")
    md.append("## L1 rules")
    md.append("")
    md.append("- If a ticker has `NO_PRICE_PROXY`, downstream action should stay research-only.")
    md.append("- If an important file is stale, do not trust the dashboard until the runner is refreshed.")
    md.append("- Local price proxy is not broker/live data.")
    md.append("- This layer exists to stop the system from silently using missing or stale inputs.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 47")
    print("L1 Data & Universe Integrity")
    print("=" * 88)

    manifest = build_manifest()
    universe = collect_universe()
    market = build_market_snapshot(universe)
    flags = build_flags(manifest, universe, market)

    manifest.to_csv(OUT_MANIFEST, index=False)
    universe.to_csv(OUT_UNIVERSE, index=False)
    market.to_csv(OUT_MARKET, index=False)
    flags.to_csv(OUT_FLAGS, index=False)
    OUT_REPORT.write_text(build_report(manifest, universe, market, flags), encoding="utf-8")

    print(f"Files audited: {len(manifest)}")
    print(f"Universe tickers: {len(universe)}")
    print(f"Market snapshot rows: {len(market)}")
    print(f"Flags: {len(flags)}")
    print()
    if not flags.empty:
        print(flags.head(20).to_string(index=False))
    print()
    print("Files generated:")
    print(f"  {OUT_MANIFEST}")
    print(f"  {OUT_UNIVERSE}")
    print(f"  {OUT_MARKET}")
    print(f"  {OUT_FLAGS}")
    print(f"  {OUT_REPORT}")
    print()
    print("Next:")
    print("  python3 -u canyon_final_v9_step44_layer_architecture_registry.py")
    print("  python3 -u canyon_final_v9_step45_master_10_layer_decision.py")
    print("  streamlit run canyon_final_v9_step46_10_layer_dashboard.py")


if __name__ == "__main__":
    main()
