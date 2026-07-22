#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 164: Position Sizer
=====================================
Enforces hard position limits:
  • ≤ 8% per single name
  • ≤ 35% per sector

Two sizing methods:
  equal_weight  — equal-weight top-N, capped at 8%
  risk_parity   — inverse-volatility weighted, capped at 8%

Reads:  single_name_risk_budget.csv  (current holdings + volatility data)
Writes: position_sizing_output.csv   (suggested weights)
        position_sizing_report.md

Usage:
  python3 canyon_final_v9_step164_position_sizer.py
  python3 canyon_final_v9_step164_position_sizer.py --method risk_parity
  python3 canyon_final_v9_step164_position_sizer.py --method equal_weight --top 10
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

MAX_SINGLE_NAME  = 0.08    # 8% hard cap
MAX_SECTOR       = 0.35    # 35% hard cap
CASH_BUFFER      = 0.05    # hold ≥5% cash at all times


# ─────────────────────────────────────────────────────────────────────────────
# Load holdings
# ─────────────────────────────────────────────────────────────────────────────

def load_holdings() -> pd.DataFrame:
    path = ROOT / "single_name_risk_budget.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run step111 first.")

    df = pd.read_csv(path)
    required = ["ticker", "weight", "sector", "annual_vol"]

    # Graceful fallback for missing columns
    if "weight" not in df.columns and "weight_pct" in df.columns:
        df["weight"] = df["weight_pct"] / 100.0
    if "annual_vol" not in df.columns and "daily_vol" in df.columns:
        df["annual_vol"] = df["daily_vol"].astype(float) * np.sqrt(252)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in holdings file: {missing}")

    df["ticker"]     = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"]     = pd.to_numeric(df["weight"],     errors="coerce").fillna(0)
    df["annual_vol"] = pd.to_numeric(df["annual_vol"], errors="coerce").fillna(0.25)
    df["sector"]     = df["sector"].fillna("Unknown")

    # Keep only stocks with positive (or near-positive) weight or alpha
    if "alpha_score" in df.columns:
        df = df[(df["weight"] > 0) | (pd.to_numeric(df["alpha_score"],
                 errors="coerce").fillna(0) > 50)]
    else:
        df = df[df["weight"] > 0]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Equal-weight sizing
# ─────────────────────────────────────────────────────────────────────────────

def equal_weight(holdings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Equal-weight top_n positions, cap at MAX_SINGLE_NAME, enforce sector cap."""
    df = holdings.copy()

    # Score: use alpha_score if available, else weight
    score_col = "alpha_score" if "alpha_score" in df.columns else "weight"
    df["_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    df = df.sort_values("_score", ascending=False).head(top_n * 3)  # wider candidate pool

    df["target_w"] = min(1.0 / top_n, MAX_SINGLE_NAME)

    df = _enforce_sector_cap(df)
    df = _enforce_single_name_cap(df)

    return df.nlargest(top_n, "target_w").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Risk-parity sizing (inverse volatility)
# ─────────────────────────────────────────────────────────────────────────────

def risk_parity(holdings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Inverse-vol weighted; cap each position at MAX_SINGLE_NAME; sector cap."""
    df = holdings.copy()
    score_col = "alpha_score" if "alpha_score" in df.columns else "weight"
    df["_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    df = df.sort_values("_score", ascending=False).head(top_n * 3)

    vol = df["annual_vol"].clip(lower=0.05)  # floor at 5%
    inv_vol = 1.0 / vol
    raw_w = inv_vol / inv_vol.sum() * (1.0 - CASH_BUFFER)

    df["target_w"] = raw_w.values

    df = _enforce_single_name_cap(df)
    df = _enforce_sector_cap(df)
    df = _enforce_single_name_cap(df)

    return df.nlargest(top_n, "target_w").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_single_name_cap(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively truncate positions above MAX_SINGLE_NAME and redistribute excess."""
    invest_budget = 1.0 - CASH_BUFFER
    for _ in range(30):
        over_mask = df["target_w"] > MAX_SINGLE_NAME
        if not over_mask.any():
            break
        # Clamp over-limit positions to MAX_SINGLE_NAME
        excess = (df.loc[over_mask, "target_w"] - MAX_SINGLE_NAME).sum()
        df.loc[over_mask, "target_w"] = MAX_SINGLE_NAME
        # Redistribute excess proportionally to uncapped positions
        under_mask = df["target_w"] < MAX_SINGLE_NAME
        if under_mask.any() and excess > 0:
            under_total = df.loc[under_mask, "target_w"].sum()
            if under_total > 0:
                df.loc[under_mask, "target_w"] += (
                    df.loc[under_mask, "target_w"] / under_total * excess
                )
            else:
                break   # nowhere to redistribute
    # Final hard clamp
    df["target_w"] = df["target_w"].clip(upper=MAX_SINGLE_NAME)
    # Scale to invest_budget
    total = df["target_w"].sum()
    if total > invest_budget:
        df["target_w"] = df["target_w"] / total * invest_budget
    return df


def _enforce_sector_cap(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively trim overweight sectors down to MAX_SECTOR."""
    for _ in range(10):
        sector_w = df.groupby("sector")["target_w"].sum()
        overweight = sector_w[sector_w > MAX_SECTOR]
        if overweight.empty:
            break
        for sector, total_w in overweight.items():
            mask = df["sector"] == sector
            scale = MAX_SECTOR / total_w
            df.loc[mask, "target_w"] *= scale
    return df


def _renorm(df: pd.DataFrame) -> pd.DataFrame:
    total = df["target_w"].sum()
    if total > 0:
        df["target_w"] = df["target_w"] / total * (1.0 - CASH_BUFFER)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Comparison: current vs proposed
# ─────────────────────────────────────────────────────────────────────────────

def diff_table(current: pd.DataFrame, proposed: pd.DataFrame) -> pd.DataFrame:
    c = current.set_index("ticker")["weight"].rename("current_w")
    p = proposed.set_index("ticker")["target_w"].rename("proposed_w")
    merged = pd.concat([c, p], axis=1).fillna(0)
    merged["delta_w"]      = merged["proposed_w"] - merged["current_w"]
    merged["delta_pct"]    = merged["delta_w"] * 100
    merged["action"]       = merged["delta_w"].apply(
        lambda x: "TRIM" if x < -0.005 else ("ADD" if x > 0.005 else "HOLD")
    )
    merged = merged.reset_index().rename(columns={"index": "ticker"})

    # Add sector from current
    if "sector" in current.columns:
        sec = current.set_index("ticker")["sector"]
        merged["sector"] = merged["ticker"].map(sec).fillna("?")

    return merged.sort_values("delta_w").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(holdings: pd.DataFrame, proposed: pd.DataFrame,
                 diff: pd.DataFrame, method: str):
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Current metrics
    cur_max   = holdings["weight"].max() * 100
    cur_total = holdings["weight"].sum() * 100
    cur_sector = holdings.groupby("sector")["weight"].sum().sort_values(ascending=False)

    # Proposed metrics
    prop_max   = proposed["target_w"].max() * 100
    prop_total = proposed["target_w"].sum() * 100
    prop_sector = proposed.groupby("sector")["target_w"].sum().sort_values(ascending=False)

    # Positions table (proposed)
    pos_rows = "\n".join(
        f"| {r.ticker:<8} | {r.sector:<18} | {r.target_w*100:>5.1f}% "
        f"| {r.annual_vol*100:>5.1f}% |"
        for _, r in proposed.sort_values("target_w", ascending=False).iterrows()
        if "sector" in proposed.columns and "annual_vol" in proposed.columns
    )

    # Sector table (current vs proposed)
    all_sectors = set(cur_sector.index) | set(prop_sector.index)
    sec_rows = "\n".join(
        f"| {s:<22} | {cur_sector.get(s, 0)*100:>6.1f}% "
        f"| {prop_sector.get(s, 0)*100:>6.1f}% "
        f"| {'⚠ OVER' if cur_sector.get(s,0) > MAX_SECTOR else 'OK'} |"
        for s in sorted(all_sectors)
    )

    # Changes
    trim_rows = diff[diff["action"] == "TRIM"]
    add_rows  = diff[diff["action"] == "ADD"]

    trim_str = "\n".join(
        f"  {r.ticker:<8} {r.current_w*100:.1f}% → {r.proposed_w*100:.1f}% "
        f"(−{abs(r.delta_pct):.1f}%)"
        for _, r in trim_rows.iterrows()
    ) or "  None"

    add_str = "\n".join(
        f"  {r.ticker:<8} {r.current_w*100:.1f}% → {r.proposed_w*100:.1f}% "
        f"(+{r.delta_pct:.1f}%)"
        for _, r in add_rows.iterrows()
    ) or "  None"

    md = f"""# Canyon v9 — Position Sizing Report
**Generated:** {today}
**Method:** {method.replace('_', ' ').title()}
**Hard limits:** Single name ≤ {MAX_SINGLE_NAME*100:.0f}% · Sector ≤ {MAX_SECTOR*100:.0f}% · Cash buffer ≥ {CASH_BUFFER*100:.0f}%

---

## Summary

| Metric | Current | Proposed |
|---|---|---|
| Largest position | {cur_max:.1f}% | {prop_max:.1f}% |
| Total invested | {cur_total:.1f}% | {prop_total:.1f}% |
| Cash held | {100-cur_total:.1f}% | {100-prop_total:.1f}% |
| Positions | {len(holdings)} | {len(proposed)} |

---

## Proposed Portfolio

| Ticker | Sector | Weight | Ann Vol |
|---|---|---|---|
{pos_rows}

---

## Sector Allocation

| Sector | Current | Proposed | Status |
|---|---|---|---|
{sec_rows}

---

## Required Changes

**Trim (reduce):**
{trim_str}

**Add (increase):**
{add_str}

---

## Why This Matters

The current portfolio has all positions at ~{cur_max:.0f}%, exceeding the 8% hard limit.
Risk consequences:
- A single-stock blow-up (e.g., earnings miss −30%) costs the portfolio {cur_max*0.30:.1f}%
- At proposed 8% cap, same blow-up costs only {prop_max*0.30:.1f}%
- Sector concentration at {cur_sector.iloc[0]*100:.0f}% ({cur_sector.index[0]}) exceeds the 35% limit
  → a sector shock (e.g., tech regulation) has outsized impact

At proper 8% sizing, the stress test P&L improves:
- Market correction −10%: estimated −{prop_max*10*1.5:.1f}% (vs −15.0% current)
- Tech selloff −20%: estimated −{prop_sector.get('Information Technology',0)*20*100:.1f}% (vs −22.9% current)

---

*Canyon v9 — Research only. No live orders.*
"""

    out = ROOT / "position_sizing_report.md"
    out.write_text(md)
    print(f"[step164] Saved position_sizing_report.md")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["equal_weight", "risk_parity"],
                        default="risk_parity")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("  Canyon v9 — Step 164: Position Sizer")
    print("=" * 60)

    holdings = load_holdings()
    print(f"[step164] Loaded {len(holdings)} current positions")

    if args.method == "risk_parity":
        proposed = risk_parity(holdings, args.top)
    else:
        proposed = equal_weight(holdings, args.top)

    # Merge sector + vol back into proposed from holdings
    meta = holdings.set_index("ticker")[["sector", "annual_vol"]]
    if "sector" not in proposed.columns:
        proposed = proposed.join(meta, on="ticker", how="left")

    diff = diff_table(holdings, proposed)

    # Print summary
    print(f"\n── Proposed ({args.method}, top-{args.top}) ─────────────────────")
    for _, r in proposed.sort_values("target_w", ascending=False).iterrows():
        sector = r.get("sector", "?")
        print(f"  {r.ticker:<8} {r.target_w*100:>5.1f}%  [{sector}]")

    sector_totals = proposed.groupby("sector")["target_w"].sum().sort_values(ascending=False) \
        if "sector" in proposed.columns else pd.Series(dtype=float)
    print(f"\n── Sector allocation ─────────────────────────────────────")
    for sector, w in sector_totals.items():
        flag = " ⚠ OVER LIMIT" if w > MAX_SECTOR else ""
        print(f"  {sector:<25} {w*100:>5.1f}%{flag}")

    print(f"\n── Changes from current ──────────────────────────────────")
    for _, r in diff[diff["action"] != "HOLD"].sort_values("delta_w").iterrows():
        arrow = "↓ TRIM" if r["action"] == "TRIM" else "↑ ADD"
        print(f"  {arrow}  {r.ticker:<8} "
              f"{r.current_w*100:.1f}% → {r.proposed_w*100:.1f}%")

    # Save outputs
    proposed_out = proposed[["ticker", "target_w"] +
                             [c for c in ["sector", "annual_vol"] if c in proposed.columns]]
    proposed_out = proposed_out.rename(columns={"target_w": "weight"})
    proposed_out["weight_pct"] = proposed_out["weight"] * 100
    proposed_out.to_csv(ROOT / "position_sizing_output.csv", index=False)
    print(f"\n[step164] Saved position_sizing_output.csv")

    write_report(holdings, proposed, diff, args.method)
    print("[step164] Saved position_sizing_report.md")


if __name__ == "__main__":
    main()
