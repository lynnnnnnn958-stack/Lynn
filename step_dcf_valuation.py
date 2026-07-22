#!/usr/bin/env python3
"""
Canyon DCF Valuation Engine — Damodaran 3-Stage Model
======================================================
Research tool answering the five Damodaran valuation questions per stock:

  1. Value of existing assets?      → ROIC, NOPAT, EVA
  2. Value of growth assets?        → PVGO, growth value decomposition
  3. Cash flow risk?                → WACC, beta, cost of capital
  4. When does it mature?           → Lifecycle stage, fade timeline
  5. Equity value per share?        → Intrinsic value vs. market price

Data: yfinance only (free). No paid sources.
Output: dcf_valuation.csv, dcf_valuation_report.md
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Global assumptions ────────────────────────────────────────────────────────
ERP            = 0.045   # Equity Risk Premium: Damodaran implied ~4.5%
STABLE_GROWTH  = 0.025   # Terminal growth = nominal GDP
STABLE_ROIC_PREMIUM = 0.02  # Terminal ROIC = WACC + 2% (modest moat)
RF_FALLBACK    = 0.044   # 10Y treasury fallback if FRED unavailable
TAX_FALLBACK   = 0.21    # US statutory rate fallback
MAX_TICKERS    = 200     # Cap per run to avoid rate limits
SLEEP_BETWEEN  = 1.5     # seconds between yfinance calls
STAGE1_YEARS   = 5
STAGE2_YEARS   = 5
STAGE3_TERMINAL = True


# ── Risk-free rate ─────────────────────────────────────────────────────────────
def _get_rf() -> float:
    """10Y treasury yield from yfinance ^TNX."""
    try:
        hist = yf.Ticker("^TNX").history(period="5d", auto_adjust=False)
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100
    except Exception:
        pass
    # fallback: try macro_signals.json
    try:
        ms = json.loads((ROOT / "macro_signals.json").read_text())
        r = ms.get("bond_yields", {}).get("tnx")
        if r:
            return float(r) / 100
    except Exception:
        pass
    return RF_FALLBACK


# ── Safe field extractor from yfinance DataFrames ─────────────────────────────
def _safe(df: pd.DataFrame, row_candidates: list[str], col_idx: int = 0) -> float | None:
    for row in row_candidates:
        try:
            val = df.loc[row].iloc[col_idx]
            if pd.notna(val):
                return float(val)
        except Exception:
            continue
    return None


# ── Fetch and parse financials for one ticker ─────────────────────────────────
def fetch_financials(tkr: str) -> dict | None:
    try:
        t    = yf.Ticker(tkr)
        info = t.info or {}
        if not info.get("marketCap"):
            return None

        inc  = t.income_stmt      # columns = annual dates, most recent first
        bs   = t.balance_sheet
        cf   = t.cashflow

        if inc is None or inc.empty:
            return None

        # ── Income statement ──────────────────────────────────────────────────
        revenue   = _safe(inc, ["Total Revenue"])
        ebit      = _safe(inc, ["EBIT", "Operating Income"])
        int_exp   = _safe(inc, ["Interest Expense"])
        tax_rate  = _safe(inc, ["Tax Rate For Calcs"])
        if tax_rate is None or tax_rate <= 0 or tax_rate > 0.5:
            try:
                tax_prov   = _safe(inc, ["Tax Provision"])
                pretax     = _safe(inc, ["Pretax Income"])
                tax_rate   = (tax_prov / pretax) if (pretax and pretax > 0 and tax_prov) else TAX_FALLBACK
            except Exception:
                tax_rate   = TAX_FALLBACK
        tax_rate = max(0.05, min(0.40, tax_rate))

        # ── Balance sheet ─────────────────────────────────────────────────────
        invested_cap   = _safe(bs, ["Invested Capital"])
        total_debt     = _safe(bs, ["Total Debt"]) or info.get("totalDebt", 0) or 0
        cash           = _safe(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]) or info.get("totalCash", 0) or 0
        net_debt       = _safe(bs, ["Net Debt"]) or (total_debt - cash)

        # ── Cash flows ────────────────────────────────────────────────────────
        capex          = _safe(cf, ["Capital Expenditure"]) or 0  # negative in yf
        fcf_yf         = _safe(cf, ["Free Cash Flow"])
        if fcf_yf is None:
            fcf_yf     = info.get("freeCashflow")

        # ── Market data from info ─────────────────────────────────────────────
        market_cap     = info.get("marketCap", 0)
        price          = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        shares         = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding", 0)
        beta           = info.get("beta") or 1.0
        beta           = max(0.3, min(3.0, float(beta)))   # clamp extreme betas

        rev_growth_1y  = info.get("revenueGrowth") or 0.0
        op_margin      = info.get("operatingMargins") or (ebit / revenue if revenue and ebit else None)
        sector         = info.get("sector", "Unknown")
        name           = info.get("shortName") or info.get("longName") or tkr

        # ── Revenue history for CAGR ──────────────────────────────────────────
        rev_hist = []
        for i in range(min(4, inc.shape[1])):
            r = _safe(inc, ["Total Revenue"], col_idx=i)
            if r:
                rev_hist.append(r)
        if len(rev_hist) >= 3:
            rev_cagr_3y = (rev_hist[0] / rev_hist[2]) ** (1/2) - 1
        elif len(rev_hist) >= 2:
            rev_cagr_3y = rev_hist[0] / rev_hist[1] - 1
        else:
            rev_cagr_3y = rev_growth_1y

        return {
            "tkr": tkr,
            "name": name,
            "sector": sector,
            "price": price,
            "market_cap": market_cap,
            "shares": shares,
            "beta": beta,
            "revenue": revenue,
            "ebit": ebit,
            "tax_rate": tax_rate,
            "invested_cap": invested_cap,
            "total_debt": total_debt,
            "cash": cash,
            "net_debt": net_debt,
            "fcf": fcf_yf,
            "capex": capex,
            "rev_growth_1y": rev_growth_1y,
            "rev_cagr_3y": rev_cagr_3y,
            "op_margin": op_margin,
            "int_exp": int_exp,
        }
    except Exception as e:
        return None


# ── WACC calculation ───────────────────────────────────────────────────────────
def calc_wacc(fin: dict, rf: float) -> dict:
    beta          = fin["beta"]
    ke            = rf + beta * ERP   # CAPM cost of equity

    total_debt    = fin["total_debt"] or 0
    market_cap    = fin["market_cap"] or 1
    int_exp       = fin["int_exp"]
    tax_rate      = fin["tax_rate"]

    if total_debt > 0 and int_exp and int_exp < 0:
        int_exp = abs(int_exp)
    kd_pretax     = (abs(int_exp) / total_debt) if (int_exp and total_debt > 0) else rf * 1.5
    kd_pretax     = max(rf, min(0.15, kd_pretax))   # clamp
    kd_aftertax   = kd_pretax * (1 - tax_rate)

    e_val  = market_cap
    d_val  = total_debt
    total  = e_val + d_val

    we     = e_val / total if total > 0 else 1.0
    wd     = d_val / total if total > 0 else 0.0

    wacc   = ke * we + kd_aftertax * wd

    return {
        "ke": ke,
        "kd_pretax": kd_pretax,
        "kd_aftertax": kd_aftertax,
        "we": we,
        "wd": wd,
        "wacc": wacc,
        "rf": rf,
        "erp": ERP,
    }


# ── ROIC & EVA ────────────────────────────────────────────────────────────────
def calc_roic_eva(fin: dict, wacc_r: dict) -> dict:
    ebit          = fin["ebit"]
    tax_rate      = fin["tax_rate"]
    invested_cap  = fin["invested_cap"]

    if not ebit or not invested_cap or invested_cap <= 0:
        return {"roic": None, "nopat": None, "eva": None, "eva_margin": None}

    nopat  = ebit * (1 - tax_rate)
    roic   = nopat / invested_cap
    wacc   = wacc_r["wacc"]
    eva    = (roic - wacc) * invested_cap   # excess return × capital = dollar EVA

    return {
        "nopat": nopat,
        "roic": roic,
        "wacc": wacc,
        "eva": eva,
        "roic_wacc_spread": roic - wacc,
        "invested_cap": invested_cap,
    }


# ── Lifecycle stage ────────────────────────────────────────────────────────────
def lifecycle_stage(rev_growth: float, roic: float | None, wacc: float) -> tuple[str, str]:
    if roic is None:
        return "Unknown", "Cannot determine without ROIC data"
    spread = roic - wacc
    if rev_growth > 0.15 and roic > wacc:
        stage = "High Growth"
        desc  = f"Growing fast (+{rev_growth:.0%}/yr), generating excess returns — PVGO dominates value"
    elif rev_growth > 0.07 and spread > 0.02:
        stage = "Maturing"
        desc  = f"Growth decelerating ({rev_growth:.0%}/yr), ROIC still above WACC — transition phase"
    elif rev_growth <= 0.07 and spread > 0:
        stage = "Mature"
        desc  = f"Slow growth ({rev_growth:.0%}/yr), ROIC barely exceeds WACC — existing asset value matters most"
    elif spread <= 0:
        stage = "Value Trap"
        desc  = f"ROIC ({roic:.1%}) below WACC ({wacc:.1%}) — destroying capital; growth makes it worse"
    else:
        stage = "Mature"
        desc  = f"Slow growth with adequate returns"
    return stage, desc


# ── 3-Stage DCF ───────────────────────────────────────────────────────────────
def calc_dcf_3stage(fin: dict, roic_r: dict, wacc_r: dict) -> dict:
    """
    3-stage Damodaran DCF:
      Stage 1 (yr 1-5):  current growth + current ROIC
      Stage 2 (yr 6-10): linear fade to stable
      Stage 3:           terminal value at stable growth

    Returns enterprise value, equity value, intrinsic value per share.
    """
    wacc   = wacc_r["wacc"]
    nopat  = roic_r.get("nopat")
    roic   = roic_r.get("roic")

    if not nopat or not roic or wacc <= 0:
        return {}

    revenue = fin["revenue"] or 0
    net_debt = fin["net_debt"] or 0
    shares   = fin["shares"] or 0
    price    = fin["price"] or 0

    # Stage 1 growth: blend of 1Y and 3Y CAGR, cap at 30%
    g1 = float(np.clip(
        0.6 * fin["rev_growth_1y"] + 0.4 * fin["rev_cagr_3y"],
        -0.10, 0.30
    ))
    g_stable = STABLE_GROWTH

    # Reinvestment rate = g / ROIC (Damodaran reinvestment formula)
    # Caps: reinvestment rate between 0 and 1 (can't reinvest more than NOPAT)
    rr1 = float(np.clip(g1 / roic if roic > 0 else 0.5, 0, 0.95))

    stable_roic    = wacc + STABLE_ROIC_PREMIUM
    rr_stable      = g_stable / stable_roic

    # ── Stage 1: years 1-5 ────────────────────────────────────────────────────
    ev = 0.0
    nopat_t = nopat
    for yr in range(1, STAGE1_YEARS + 1):
        nopat_t  *= (1 + g1)
        fcff_t    = nopat_t * (1 - rr1)
        ev       += fcff_t / (1 + wacc) ** yr

    nopat_end_s1 = nopat_t  # NOPAT at end of Stage 1

    # ── Stage 2: years 6-10, linear fade ─────────────────────────────────────
    for yr_local in range(1, STAGE2_YEARS + 1):
        yr_total = STAGE1_YEARS + yr_local
        frac     = yr_local / STAGE2_YEARS
        g_t      = g1 * (1 - frac) + g_stable * frac
        roic_t   = roic * (1 - frac) + stable_roic * frac
        rr_t     = float(np.clip(g_t / roic_t if roic_t > 0 else rr_stable, 0, 0.95))
        nopat_t  *= (1 + g_t)
        fcff_t    = nopat_t * (1 - rr_t)
        ev       += fcff_t / (1 + wacc) ** yr_total

    nopat_terminal = nopat_t * (1 + g_stable)

    # ── Terminal value ─────────────────────────────────────────────────────────
    fcff_terminal  = nopat_terminal * (1 - rr_stable)
    if wacc <= g_stable:
        wacc = g_stable + 0.01  # prevent division by zero
    tv             = fcff_terminal / (wacc - g_stable)
    pv_tv          = tv / (1 + wacc) ** (STAGE1_YEARS + STAGE2_YEARS)
    ev            += pv_tv

    # ── Equity value ──────────────────────────────────────────────────────────
    equity_val     = ev - net_debt
    iv_per_share   = equity_val / shares if shares > 0 else None
    upside         = (iv_per_share / price - 1) if (iv_per_share and price > 0) else None

    # ── PVGO decomposition ────────────────────────────────────────────────────
    # Value if zero growth (all NOPAT paid as dividend, no reinvestment)
    ev_no_growth   = nopat / wacc
    pvgo           = ev - ev_no_growth
    pvgo_pct       = pvgo / ev if ev > 0 else 0

    return {
        "ev": ev,
        "pv_tv": pv_tv,
        "tv_pct": pv_tv / ev if ev > 0 else 0,
        "equity_val": equity_val,
        "iv_per_share": iv_per_share,
        "upside_pct": upside,
        "pvgo": pvgo,
        "pvgo_pct": pvgo_pct,
        "g1_assumed": g1,
        "rr1_assumed": rr1,
        "g_stable": g_stable,
    }


# ── Full valuation for one ticker ─────────────────────────────────────────────
def value_ticker(tkr: str, rf: float) -> dict | None:
    fin = fetch_financials(tkr)
    if not fin:
        return None

    wacc_r  = calc_wacc(fin, rf)
    roic_r  = calc_roic_eva(fin, wacc_r)
    dcf_r   = calc_dcf_3stage(fin, roic_r, wacc_r)

    rev_growth = fin["rev_growth_1y"] or fin["rev_cagr_3y"] or 0
    stage, stage_desc = lifecycle_stage(rev_growth, roic_r.get("roic"), wacc_r["wacc"])

    result = {
        "ticker":          tkr,
        "name":            fin["name"],
        "sector":          fin["sector"],
        "price":           fin["price"],
        "market_cap_b":    round((fin["market_cap"] or 0) / 1e9, 1),
        # Risk / Cost of capital
        "beta":            round(fin["beta"], 2),
        "ke":              round(wacc_r["ke"], 4),
        "kd_aftertax":     round(wacc_r["kd_aftertax"], 4),
        "wacc":            round(wacc_r["wacc"], 4),
        "debt_pct":        round(wacc_r["wd"] * 100, 1),
        # Existing asset value
        "roic":            round(roic_r["roic"], 4) if roic_r.get("roic") is not None else None,
        "roic_wacc_spread":round(roic_r.get("roic_wacc_spread", 0), 4) if roic_r.get("roic") else None,
        "nopat_m":         round((roic_r.get("nopat") or 0) / 1e6, 1),
        "eva_m":           round((roic_r.get("eva") or 0) / 1e6, 1),
        # Growth
        "rev_growth_1y":   round(rev_growth, 4),
        "rev_cagr_3y":     round(fin["rev_cagr_3y"], 4),
        "stage":           stage,
        "stage_desc":      stage_desc,
        # DCF output
        "iv_per_share":    round(dcf_r["iv_per_share"], 2) if dcf_r.get("iv_per_share") else None,
        "upside_pct":      round(dcf_r["upside_pct"] * 100, 1) if dcf_r.get("upside_pct") is not None else None,
        "pvgo_pct":        round(dcf_r.get("pvgo_pct", 0) * 100, 1),
        "tv_pct":          round(dcf_r.get("tv_pct", 0) * 100, 1),
        "g1_assumed":      round(dcf_r.get("g1_assumed", 0) * 100, 1) if dcf_r else None,
        # Assumptions used
        "rf":              round(rf * 100, 2),
        "erp":             round(ERP * 100, 1),
        "as_of":           date.today().isoformat(),
    }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Canyon DCF Valuation Engine — {date.today()}")
    print("=" * 60)

    # ── Load ticker list ──────────────────────────────────────────────────────
    universe = os.environ.get("CANYON_UNIVERSE", "")
    universe_path = ROOT / f"universe_{universe}.csv" if universe else None
    scores_path   = ROOT / "alpha_scores.csv"

    if universe_path and universe_path.exists():
        src = pd.read_csv(universe_path)
        ticker_col = "ticker" if "ticker" in src.columns else src.columns[0]
        tickers = src[ticker_col].dropna().tolist()[:MAX_TICKERS]
        print(f"  Universe: {universe} ({len(tickers)} tickers from {universe_path.name})")
    elif scores_path.exists():
        src = pd.read_csv(scores_path)
        tickers = src["ticker"].tolist()[:MAX_TICKERS]
        print(f"  Universe: S&P 500 alpha_scores ({len(tickers)} tickers — top by alpha rank)")
        print(f"  Tip: run 'python step_get_universe.py russell1000' then set CANYON_UNIVERSE=russell1000")
    else:
        print("ERROR: alpha_scores.csv not found")
        return

    # ── Get risk-free rate ────────────────────────────────────────────────────
    rf = _get_rf()
    print(f"  Risk-free rate:  {rf:.2%}  |  ERP: {ERP:.1%}  |  Stable growth: {STABLE_GROWTH:.1%}")
    print()

    # ── Cache: skip tickers already valued today ──────────────────────────────
    cache_path = ROOT / "dcf_valuation.csv"
    done_today = set()
    if cache_path.exists():
        try:
            old = pd.read_csv(cache_path)
            if "as_of" in old.columns:
                today_old = old[old["as_of"] == date.today().isoformat()]
                done_today = set(today_old["ticker"].tolist())
        except Exception:
            pass

    # ── Run valuations ────────────────────────────────────────────────────────
    results = []
    # Load existing non-today rows to preserve them
    if cache_path.exists():
        try:
            existing = pd.read_csv(cache_path)
            old_rows = existing[existing["as_of"] != date.today().isoformat()]
            results.extend(old_rows.to_dict("records"))
        except Exception:
            pass

    todo = [t for t in tickers if t not in done_today]
    print(f"  {len(done_today)} already done today, {len(todo)} to fetch")

    ok, fail = 0, 0
    for i, tkr in enumerate(todo, 1):
        try:
            r = value_ticker(tkr, rf)
            if r:
                results.append(r)
                stage_icon = {"High Growth": "🚀", "Maturing": "📈", "Mature": "📊",
                              "Value Trap": "⚠", "Unknown": "?"}.get(r["stage"], "?")
                upside_str = f"{r['upside_pct']:+.0f}%" if r["upside_pct"] is not None else "n/a"
                print(f"  [{i:3d}/{len(todo)}] {tkr:6s}  {stage_icon} {r['stage']:12s}  "
                      f"ROIC {(r['roic']*100 if r['roic'] else 0):.1f}%  "
                      f"WACC {r['wacc']*100:.1f}%  "
                      f"IV ${r['iv_per_share'] or 0:.0f}  {upside_str}")
                ok += 1
            else:
                print(f"  [{i:3d}/{len(todo)}] {tkr:6s}  — no data")
                fail += 1
        except Exception as e:
            print(f"  [{i:3d}/{len(todo)}] {tkr:6s}  ERROR: {e}")
            fail += 1
        time.sleep(SLEEP_BETWEEN)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if results:
        df = pd.DataFrame(results)
        # Keep only today's data for each ticker (latest run wins)
        df = df.sort_values("as_of", ascending=False).drop_duplicates("ticker")
        # Sort: most undervalued first
        df = df.sort_values("upside_pct", ascending=False, na_position="last")
        df.to_csv(cache_path, index=False)
        print(f"\n  Saved {len(df)} records → {cache_path.name}")

    # ── Generate markdown report ──────────────────────────────────────────────
    _write_report(df if results else pd.DataFrame(), rf)
    print(f"  Report → dcf_valuation_report.md")
    print(f"\n  {ok} OK  |  {fail} failed  |  rf={rf:.2%}  ERP={ERP:.1%}")


def _write_report(df: pd.DataFrame, rf: float):
    today = date.today().isoformat()
    today_df = df[df["as_of"] == today] if "as_of" in df.columns else df

    lines = [
        f"# Canyon DCF Valuation Report",
        f"*{today} · 3-Stage Damodaran Model · rf={rf:.2%} · ERP={ERP:.1%} · Stable g={STABLE_GROWTH:.1%}*",
        "",
        "## Most Undervalued (DCF upside > 20%)",
        "",
        "| Ticker | Name | Price | IV | Upside | ROIC | WACC | Spread | Stage | PVGO% |",
        "|--------|------|-------|----|--------|------|------|--------|-------|-------|",
    ]
    for _, row in today_df[today_df["upside_pct"].notna() & (today_df["upside_pct"] > 20)].head(20).iterrows():
        lines.append(
            f"| **{row['ticker']}** | {row['name'][:22]} | ${row['price']:.0f} "
            f"| ${row['iv_per_share']:.0f} | **{row['upside_pct']:+.0f}%** "
            f"| {(row['roic']*100 if row['roic'] else 0):.1f}% | {row['wacc']*100:.1f}% "
            f"| {(row['roic_wacc_spread']*100 if row['roic_wacc_spread'] else 0):+.1f}% "
            f"| {row['stage']} | {row['pvgo_pct']:.0f}% |"
        )

    lines += [
        "",
        "## Most Overvalued (DCF upside < -20%)",
        "",
        "| Ticker | Name | Price | IV | Upside | ROIC | WACC | Stage |",
        "|--------|------|-------|----|--------|------|------|-------|",
    ]
    for _, row in today_df[today_df["upside_pct"].notna() & (today_df["upside_pct"] < -20)].tail(20).iterrows():
        lines.append(
            f"| {row['ticker']} | {row['name'][:22]} | ${row['price']:.0f} "
            f"| ${row['iv_per_share']:.0f} | {row['upside_pct']:+.0f}% "
            f"| {(row['roic']*100 if row['roic'] else 0):.1f}% | {row['wacc']*100:.1f}% "
            f"| {row['stage']} |"
        )

    lines += [
        "",
        "## Capital Destroyers (ROIC < WACC)",
        "",
        "| Ticker | Name | ROIC | WACC | Spread | EVA ($M) | Stage |",
        "|--------|------|------|------|--------|----------|-------|",
    ]
    destroyers = today_df[today_df["roic_wacc_spread"].notna() & (today_df["roic_wacc_spread"] < 0)].sort_values("roic_wacc_spread")
    for _, row in destroyers.head(15).iterrows():
        lines.append(
            f"| {row['ticker']} | {row['name'][:22]} "
            f"| {(row['roic']*100 if row['roic'] else 0):.1f}% | {row['wacc']*100:.1f}% "
            f"| **{(row['roic_wacc_spread']*100 if row['roic_wacc_spread'] else 0):+.1f}%** "
            f"| {row['eva_m']:+,.0f} | {row['stage']} |"
        )

    lines += [
        "",
        "## Lifecycle Distribution",
        "",
    ]
    if not today_df.empty and "stage" in today_df.columns:
        for stage, cnt in today_df["stage"].value_counts().items():
            pct = cnt / len(today_df) * 100
            lines.append(f"- **{stage}**: {cnt} stocks ({pct:.0f}%)")

    lines += [
        "",
        "---",
        "## Methodology",
        "",
        "**ROIC** = NOPAT / Invested Capital  where NOPAT = EBIT × (1 − tax rate)",
        "",
        "**WACC** = Ke × (E/V) + Kd×(1−t) × (D/V)  where Ke = rf + β × ERP",
        "",
        "**EVA** = (ROIC − WACC) × Invested Capital",
        "",
        "**DCF**: 3-stage model.  Stage 1 (yr 1-5): current ROIC × blended growth.  "
        "Stage 2 (yr 6-10): linear fade to stable.  Terminal: g=2.5%, ROIC=WACC+2%",
        "",
        "**PVGO** = Enterprise Value − (NOPAT/WACC).  The share of value from future growth.",
        "",
        f"*Assumptions: rf={rf:.2%}, ERP={ERP:.1%}, stable g={STABLE_GROWTH:.1%}, "
        f"terminal ROIC premium=+{STABLE_ROIC_PREMIUM:.0%}*",
    ]

    (ROOT / "dcf_valuation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
