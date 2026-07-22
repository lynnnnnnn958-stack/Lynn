#!/usr/bin/env python3
"""
Canyon v9 — Unified Risk Gate

Single entry point called before any position change is recorded.
Reads live outputs of steps 111, 112, 114, 116 and returns a hard
GO / NO-GO decision. No broker connection. No live orders.

Usage:
    from risk_gate import check_position_allowed, run_gate_report

    ok, reason = check_position_allowed("NVDA", "BUY", 0.08, portfolio_weights)
    if not ok:
        print(f"BLOCKED: {reason}")

    # Full gate report (prints table of all current positions)
    run_gate_report()
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ── Hard limits ───────────────────────────────────────────────────────────────
MAX_SINGLE_NAME      = 0.08    # 8% max per position
MAX_SECTOR           = 0.35    # 35% max per sector
MAX_LIQUIDITY_WEIGHT = 0.05    # 5% max if liquidity flags SIZE_DOWN
DD_YELLOW_PCT        = 0.05    # 5% drawdown from HWM → reduce new size by 50%
DD_ORANGE_PCT        = 0.10    # 10% drawdown → freeze new longs, trim worst
DD_RED_PCT           = 0.15    # 15% drawdown → all positions to 50% target
MAX_CORR_TO_EXISTING = 0.85    # block new position if >85% corr to an existing one
MAX_PORTFOLIO_BETA   = 1.20    # block if adding this would push beta above 1.2


# ── Loaders ───────────────────────────────────────────────────────────────────

def _safe_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(ROOT / path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _safe_json(path: str | Path) -> dict:
    try:
        with open(ROOT / path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_all() -> dict:
    return {
        "s111": _safe_csv("single_name_risk_budget.csv"),
        "s112": _safe_csv("sector_active_exposure.csv"),
        "s114": _safe_json("drawdown_control_state.json"),
        "s116_corr": _safe_csv("holdings_correlation_matrix.csv", index_col=0),
        "s116_beta": _safe_csv("portfolio_beta_report.csv"),
    }


# ── Individual checks (each returns (ok: bool, reason: str)) ─────────────────

def _check_drawdown(dd: dict, side: str) -> tuple[bool, str]:
    dd_pct = abs(float(dd.get("drawdown_pct", 0.0)))
    if side != "BUY":
        return True, ""
    if dd_pct >= DD_RED_PCT:
        return False, (f"RED circuit: drawdown {dd_pct*100:.1f}% ≥ {DD_RED_PCT*100:.0f}% — "
                       "all positions to 50%, no new longs")
    if dd_pct >= DD_ORANGE_PCT:
        return False, (f"ORANGE circuit: drawdown {dd_pct*100:.1f}% ≥ {DD_ORANGE_PCT*100:.0f}% — "
                       "new longs frozen until recovery")
    return True, ""


def _check_single_name(ticker: str, size: float, s111: pd.DataFrame) -> tuple[bool, str]:
    if size > MAX_SINGLE_NAME:
        return False, (f"{ticker}: requested size {size*100:.1f}% > "
                       f"hard limit {MAX_SINGLE_NAME*100:.0f}%")
    if s111.empty or "ticker" not in s111.columns:
        return True, ""
    row = s111[s111["ticker"].str.upper() == ticker.upper()]
    if row.empty:
        return True, ""
    action = str(row.iloc[0].get("single_name_action", "")).upper()
    if action == "STOP":
        stop_lvl = row.iloc[0].get("single_name_stop_level", "?")
        return False, f"{ticker}: step111 STOP — price below stop level {stop_lvl}"
    liq = str(row.iloc[0].get("liquidity_label", "")).upper()
    if liq == "SIZE_DOWN" and size > MAX_LIQUIDITY_WEIGHT:
        return False, (f"{ticker}: liquidity SIZE_DOWN — max allowed "
                       f"{MAX_LIQUIDITY_WEIGHT*100:.0f}%, requested {size*100:.1f}%")
    return True, ""


def _check_sector(ticker: str, size: float, s111: pd.DataFrame,
                  s112: pd.DataFrame, portfolio: dict[str, float]) -> tuple[bool, str]:
    if s111.empty or s112.empty:
        return True, ""
    row111 = s111[s111["ticker"].str.upper() == ticker.upper()]
    if row111.empty:
        return True, ""
    sector = str(row111.iloc[0].get("sector", ""))
    if not sector:
        return True, ""

    # Current sector weight from live portfolio
    sector_tickers = (
        s111[s111["sector"] == sector]["ticker"].str.upper().tolist()
        if "sector" in s111.columns else []
    )
    current_sector_w = sum(
        portfolio.get(t, 0.0) for t in sector_tickers
    )
    projected = current_sector_w + size

    row112 = s112[s112["sector"].str.strip() == sector]
    cap = float(row112.iloc[0]["sector_cap"]) if not row112.empty and "sector_cap" in row112.columns else MAX_SECTOR

    if projected > cap:
        return False, (f"{ticker}: adding {size*100:.1f}% would put {sector} sector "
                       f"at {projected*100:.1f}% > cap {cap*100:.0f}%")
    return True, ""


def _check_correlation(ticker: str, s116_corr: pd.DataFrame,
                       portfolio: dict[str, float]) -> tuple[bool, str]:
    if s116_corr.empty or ticker.upper() not in s116_corr.index:
        return True, ""
    row = s116_corr.loc[ticker.upper()]
    for held_ticker, held_w in portfolio.items():
        if held_w <= 0 or held_ticker.upper() == ticker.upper():
            continue
        if held_ticker.upper() not in row.index:
            continue
        corr = float(row[held_ticker.upper()])
        if corr > MAX_CORR_TO_EXISTING:
            return False, (f"{ticker}: correlation {corr:.2f} with existing "
                           f"position {held_ticker} > limit {MAX_CORR_TO_EXISTING}")
    return True, ""


def _check_beta(ticker: str, size: float,
                s116_beta: pd.DataFrame) -> tuple[bool, str]:
    if s116_beta.empty or "portfolio_beta" not in s116_beta.columns:
        return True, ""
    try:
        current_beta = float(s116_beta["portfolio_beta"].iloc[0])
    except Exception:
        return True, ""
    if current_beta > MAX_PORTFOLIO_BETA:
        return False, (f"Portfolio beta {current_beta:.2f} already above "
                       f"limit {MAX_PORTFOLIO_BETA} — no new longs")
    return True, ""


# ── Public API ────────────────────────────────────────────────────────────────

def check_position_allowed(
    ticker: str,
    side: str,                      # "BUY" or "SELL"
    size: float,                    # weight as fraction (e.g. 0.08 = 8%)
    portfolio: dict[str, float],    # current weights {ticker: fraction}
) -> tuple[bool, str]:
    """
    Returns (approved: bool, reason: str).
    A False return is a hard NO — do not add or increase the position.
    """
    data = _load_all()
    checks = [
        ("drawdown_circuit",  lambda: _check_drawdown(data["s114"], side)),
        ("single_name_limit", lambda: _check_single_name(ticker, size, data["s111"])),
        ("sector_budget",     lambda: _check_sector(ticker, size, data["s111"], data["s112"], portfolio)),
        ("correlation",       lambda: _check_correlation(ticker, data["s116_corr"], portfolio)),
        ("portfolio_beta",    lambda: _check_beta(ticker, size, data["s116_beta"])),
    ]
    for name, fn in checks:
        ok, reason = fn()
        if not ok:
            return False, f"[{name}] {reason}"
    return True, "APPROVED"


def run_gate_report(portfolio: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Print a gate-check table for all current positions.
    Returns a DataFrame with columns: ticker, size, approved, reason.
    """
    data   = _load_all()
    s111   = data["s111"]
    dd     = data["s114"]

    if portfolio is None and not s111.empty and "ticker" in s111.columns:
        portfolio = {
            str(r["ticker"]).upper(): float(r.get("weight", 0.0))
            for _, r in s111.iterrows()
            if pd.notna(r.get("ticker"))
        }

    portfolio = portfolio or {}

    rows = []
    for ticker, size in portfolio.items():
        ok, reason = check_position_allowed(ticker, "BUY", size, portfolio)
        rows.append({"ticker": ticker, "size_pct": round(size * 100, 2),
                     "approved": "YES" if ok else "NO",
                     "reason": reason})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No positions found.")
        return df

    print()
    print("=" * 72)
    print(f"  Canyon v9 — Risk Gate Report   {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Drawdown: {abs(float(dd.get('drawdown_pct', 0)))*100:.2f}%  "
          f"Circuit: {dd.get('circuit_level', 'NONE')}")
    print("=" * 72)
    for _, r in df.iterrows():
        flag = "✓" if r["approved"] == "YES" else "✗"
        print(f"  {flag}  {r['ticker']:<8} {r['size_pct']:>5.1f}%   {r['reason']}")
    print("=" * 72)
    blocked = (df["approved"] == "NO").sum()
    print(f"  {len(df)} positions checked — {blocked} blocked")
    print()

    out_path = ROOT / "risk_gate_report.csv"
    df.to_csv(out_path, index=False)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_gate_report()
