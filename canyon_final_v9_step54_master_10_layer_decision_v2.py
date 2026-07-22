#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 54 — Master 10-Layer Decision Matrix v2

This replaces the old Step 45 logic.

Key difference:
Old Step 45 still marked L2/L4 as NO_DATA because those layers were not built yet.
Step 54 reads the newly generated L2-L6 files and produces a real 10-layer ticker matrix.

Inputs:
L1  market_data_snapshot.csv, universe_master.csv, data_quality_flags.csv
L2  macro_regime_signals.csv, index_breadth_dashboard.csv, volatility_regime.csv
L3  sector_rotation_scores.csv
L4  fundamental_quality_valuation.csv
L5  evidence_cards.csv, earnings_calendar_check.csv, news_event_risk.csv
L6  technical_signal_matrix.csv, intraday_liquidity_proxy.csv
L7  options_decision_matrix.csv
L8  exposure_warnings.csv, scenario_stress_results.csv, position_sizing_recommendations.csv
L9  pre_trade_checklist.csv, execution_gate_review.csv, action_cards.csv
L10 learning_attribution_summary.csv, learning_weight_suggestions.csv, paper_portfolio_ledger.csv

Outputs:
- master_10_layer_decision_matrix_v2.csv
- master_10_layer_decision_report_v2.md
- master_10_layer_scorecard.csv

No broker connection. No live order.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()

FILES = {
    "universe": ROOT / "universe_master.csv",
    "market": ROOT / "market_data_snapshot.csv",
    "dq_flags": ROOT / "data_quality_flags.csv",

    "macro": ROOT / "macro_regime_signals.csv",
    "breadth": ROOT / "index_breadth_dashboard.csv",
    "vol": ROOT / "volatility_regime.csv",

    "sector": ROOT / "sector_rotation_scores.csv",

    "fund": ROOT / "fundamental_quality_valuation.csv",

    "evidence": ROOT / "evidence_cards.csv",
    "earnings": ROOT / "earnings_calendar_check.csv",
    "news": ROOT / "news_event_risk.csv",

    "technical": ROOT / "technical_signal_matrix.csv",
    "liquidity": ROOT / "intraday_liquidity_proxy.csv",

    "options": ROOT / "options_decision_matrix.csv",
    "v8_options": ROOT / "v8_synthetic_options_overlay.csv",

    "risk": ROOT / "exposure_warnings.csv",
    "stress": ROOT / "scenario_stress_results.csv",
    "sizing": ROOT / "position_sizing_recommendations.csv",

    "pre": ROOT / "pre_trade_checklist.csv",
    "v8_l9": ROOT / "v8_l9_execution_gate.csv",
    "gate": ROOT / "execution_gate_review.csv",
    "cards": ROOT / "action_cards.csv",

    "learning": ROOT / "learning_attribution_summary.csv",
    "learn_weights": ROOT / "learning_weight_suggestions.csv",
    "ledger": ROOT / "paper_portfolio_ledger.csv",
}

OUT_MATRIX = ROOT / "master_10_layer_decision_matrix_v2.csv"
OUT_REPORT = ROOT / "master_10_layer_decision_report_v2.md"
OUT_SCORECARD = ROOT / "master_10_layer_scorecard.csv"

ETF_TO_THEME = {
    "SPY": "Broad Market",
    "QQQ": "Technology",
    "XLK": "Technology",
    "SMH": "Semiconductor",
    "SOXX": "Semiconductor",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "IYR": "Real Estate",
    "TLT": "Rates / Long Bond",
    "GLD": "Gold",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(x, default=np.nan):
    try:
        s = str(x).replace("%", "").replace(",", "").strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def get_row(df: pd.DataFrame, ticker: str) -> dict:
    if df.empty or "ticker" not in df.columns:
        return {}
    m = df[df["ticker"].astype(str).str.upper().str.strip() == ticker]
    if m.empty:
        return {}
    return m.iloc[0].to_dict()


def universe() -> list[str]:
    tickers = set()
    for key in ["universe", "market", "sector", "fund", "evidence", "technical", "options", "sizing", "pre", "cards", "ledger"]:
        df = read_csv(FILES[key])
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].astype(str).str.upper().str.strip().tolist())
    return sorted([t for t in tickers if t and t not in {"CASH", "TACTICAL_CASH"}])


def macro_regime() -> tuple[str, str, float]:
    macro = read_csv(FILES["macro"])
    breadth = read_csv(FILES["breadth"])
    vol = read_csv(FILES["vol"])

    if macro.empty:
        return "NO_DATA", "Macro layer missing.", 0

    def ret20(ticker):
        row = get_row(macro, ticker)
        return fnum(row.get("ret_20d", np.nan))

    spy20 = ret20("SPY")
    qqq20 = ret20("QQQ")
    iwm20 = ret20("IWM")
    tlt20 = ret20("TLT")
    uup20 = ret20("UUP")
    vix20 = ret20("^VIX")

    score = 0
    reasons = []

    if np.isfinite(spy20) and spy20 > 0:
        score += 20
        reasons.append("SPY 20d positive")
    elif np.isfinite(spy20):
        score -= 15
        reasons.append("SPY 20d negative")

    if np.isfinite(qqq20) and np.isfinite(spy20) and qqq20 > spy20:
        score += 10
        reasons.append("QQQ leads SPY")
    if np.isfinite(iwm20) and iwm20 > 0:
        score += 10
        reasons.append("IWM positive breadth")
    if np.isfinite(vix20) and vix20 < 0:
        score += 10
        reasons.append("VIX falling")
    elif np.isfinite(vix20) and vix20 > 0.15:
        score -= 15
        reasons.append("VIX rising")

    if np.isfinite(tlt20) and tlt20 < -0.03:
        score -= 10
        reasons.append("TLT weak / rates pressure")
    if np.isfinite(uup20) and uup20 > 0.02:
        score -= 10
        reasons.append("USD strength headwind")

    if not breadth.empty and "above_50dma" in breadth.columns:
        vals = breadth["above_50dma"].astype(str).str.upper()
        total = int(vals.isin(["TRUE", "FALSE"]).sum())
        above = int((vals == "TRUE").sum())
        if total > 0:
            pct = above / total
            if pct > 0.60:
                score += 15
                reasons.append(f"breadth above 50dma {pct:.0%}")
            elif pct < 0.40:
                score -= 15
                reasons.append(f"breadth weak above 50dma {pct:.0%}")

    score = max(0, min(100, score + 50))

    if score >= 70:
        label = "RISK_ON"
    elif score >= 45:
        label = "MIXED_RISK"
    else:
        label = "RISK_OFF_OR_CHOPPY"

    return label, "; ".join(reasons), score


def portfolio_risk() -> tuple[str, str]:
    risk = read_csv(FILES["risk"])
    stress = read_csv(FILES["stress"])

    high = med = 0
    if not risk.empty and "level" in risk.columns:
        levels = risk["level"].astype(str).str.upper()
        high = int((levels == "HIGH").sum())
        med = int((levels == "MEDIUM").sum())

    worst = np.nan
    if not stress.empty and "estimated_pnl" in stress.columns:
        vals = pd.to_numeric(stress["estimated_pnl"], errors="coerce")
        if vals.notna().any():
            worst = float(vals.min())

    if high > 0 or (np.isfinite(worst) and worst <= -0.02):
        state = "RED"
    elif med >= 3 or (np.isfinite(worst) and worst <= -0.01):
        state = "AMBER"
    else:
        state = "GREEN"

    detail = f"HIGH warnings={high}, MEDIUM warnings={med}"
    if np.isfinite(worst):
        detail += f", worst={worst:.2%}"
    return state, detail


def layer_l1(ticker: str) -> tuple[str, int, str]:
    market = read_csv(FILES["market"])
    uni = read_csv(FILES["universe"])

    m = get_row(market, ticker)
    u = get_row(uni, ticker)

    confidence = str(m.get("data_confidence", "NO_DATA")).upper()
    source_count = fnum(u.get("source_count", np.nan))

    if confidence == "LOCAL_PRICE_PROXY_OK" and np.isfinite(source_count) and source_count >= 2:
        return "OK", 90, f"{confidence}; sources={int(source_count)}"
    if confidence == "LOCAL_PRICE_PROXY_OK":
        return "PARTIAL", 70, f"{confidence}; sources={source_count if np.isfinite(source_count) else 'unknown'}"
    if confidence == "STALE_PRICE_PROXY":
        return "STALE", 35, "stale local price proxy"
    return "PRICE_DATA_UNAVAILABLE", 10, confidence or "No usable local price proxy"


def layer_l2() -> tuple[str, int, str]:
    label, reasons, score = macro_regime()
    return label, int(score), reasons


def layer_l3(ticker: str) -> tuple[str, int, str]:
    sector = read_csv(FILES["sector"])
    r = get_row(sector, ticker)

    if not r:
        if ticker in {"SPY", "QQQ"}:
            theme = ETF_TO_THEME.get(ticker, "Macro Benchmark")
            return "MACRO_BENCHMARK_CONTEXT", 50, f"{ticker} is a {theme} benchmark; L3 rotation is contextual, not missing."
        if ticker in {"GLD", "TLT"}:
            theme = ETF_TO_THEME.get(ticker, "Hedge")
            return "HEDGE_CONTEXT", 45, f"{ticker} is a {theme} hedge/rates context asset; L3 sector rotation is not applicable."
        # for non-sector stocks, map through fundamental sector if possible
        fund = get_row(read_csv(FILES["fund"]), ticker)
        sec = str(fund.get("sector", ""))
        if sec:
            return "SECTOR_CONTEXT_ONLY", 45, f"company sector={sec}; no direct ETF rotation row"
        return "SECTOR_DATA_UNAVAILABLE", 10, "No sector/theme rotation row; keep research-only until data refresh."

    label = str(r.get("rotation_label", "NO_DATA")).upper()
    if label == "NO_DATA":
        if ticker in {"SPY", "QQQ"}:
            theme = ETF_TO_THEME.get(ticker, "Macro Benchmark")
            return "MACRO_BENCHMARK_CONTEXT", 50, f"{ticker} is a {theme} benchmark; sector feed has no fresh rotation data."
        if ticker in {"GLD", "TLT"}:
            theme = ETF_TO_THEME.get(ticker, "Hedge")
            return "HEDGE_CONTEXT", 45, f"{ticker} is a {theme} context asset; sector rotation is not the right primary lens."
        if ticker in ETF_TO_THEME:
            theme = ETF_TO_THEME.get(ticker, "ETF")
            return "ETF_SECTOR_CONTEXT", 35, f"{ticker} maps to {theme}; no fresh relative-strength data, so L3 is context-only."

    score = fnum(r.get("rotation_score", np.nan))
    if not np.isfinite(score):
        score_norm = 0
    else:
        # convert raw rotation score roughly into 0-100
        score_norm = int(max(0, min(100, 50 + score)))

    note = f"theme={r.get('theme','')}; raw_score={r.get('rotation_score','')}; rel20={r.get('relative_20d_vs_spy','')}"
    return label, score_norm, note


def layer_l4(ticker: str) -> tuple[str, int, str]:
    r = get_row(read_csv(FILES["fund"]), ticker)
    if not r:
        if ticker in ETF_TO_THEME:
            return "ETF_NOT_FUNDAMENTAL", 50, "ETF judged by L2/L3/L8, not single-company fundamentals."
        return "FUNDAMENTAL_DATA_UNAVAILABLE", 10, "No fundamental row; keep research-only until data refresh."

    asset = str(r.get("asset_type", ""))
    label = str(r.get("fundamental_label", "NO_DATA"))
    score = fnum(r.get("quality_score", 0), 0)

    if asset == "ETF" or ticker in ETF_TO_THEME:
        return "ETF_NOT_FUNDAMENTAL", 50, "ETF judged by L2/L3/L8, not single-company fundamentals."

    if label.upper() == "NO_DATA":
        return "FUNDAMENTAL_DATA_UNAVAILABLE", 10, "No usable company fundamentals; keep research-only until data refresh."

    return label[:80], int(max(0, min(100, score))), f"quality_score={score}; sector={r.get('sector','')}; fwdPE={r.get('forward_pe','')}"


def layer_l5(ticker: str) -> tuple[str, int, str]:
    r = get_row(read_csv(FILES["evidence"]), ticker)
    if not r:
        if ticker in ETF_TO_THEME:
            return "ETF_EVENT_CONTEXT", 50, "ETF has no single-company earnings/insider row; use macro, sector, and risk context."
        return "NO_DATA", 0, "No event/evidence row"

    label = str(r.get("event_label", "NO_DATA")).upper()
    score = fnum(r.get("event_score", 0), 0)
    score_norm = int(max(0, min(100, 50 + score)))
    return label, score_norm, r.get("reasons", "")


def layer_l6(ticker: str) -> tuple[str, int, str]:
    tech = get_row(read_csv(FILES["technical"]), ticker)
    liq = get_row(read_csv(FILES["liquidity"]), ticker)

    if not tech:
        sector = get_row(read_csv(FILES["sector"]), ticker)
        if sector:
            ret20 = fnum(sector.get("ret_20d", 0), 0)
            ret63 = fnum(sector.get("ret_63d", 0), 0)
            trend_score = fnum(sector.get("trend_score", 0), 0)
            rotation = str(sector.get("rotation_label", "NO_DATA")).upper()
            score = int(max(0, min(100, 45 + trend_score * 8 + ret20 * 100)))
            if ret20 > 0 and trend_score >= 1:
                label = "WATCH"
            elif ret20 <= 0 and rotation == "LAGGARD":
                label = "NO_TECH_EDGE"
            else:
                label = "WATCH" if score >= 50 else "NO_TECH_EDGE"
            note = (
                f"sector technical fallback; ret20={sector.get('ret_20d','')}; "
                f"ret63={sector.get('ret_63d','')}; trend_score={sector.get('trend_score','')}; "
                f"rotation={rotation}"
            )
            return label, score, note
        return "NO_DATA", 0, "No technical row"

    label = str(tech.get("technical_label", "NO_DATA")).upper()
    score = fnum(tech.get("technical_score", 0), 0)
    liq_label = liq.get("liquidity_label", "NO_DATA") if liq else "NO_DATA"
    note = f"tech_score={score}; RSI={tech.get('rsi14','')}; ret20={tech.get('ret_20d','')}; vol_z={tech.get('volume_z60','')}; liquidity={liq_label}"
    return label, int(max(0, min(100, score))), note


def layer_l7(ticker: str) -> tuple[str, int, str]:
    r = get_row(read_csv(FILES["options"]), ticker)
    if not r:
        v8 = get_row(read_csv(FILES["v8_options"]), ticker)
        if v8:
            risk = str(v8.get("squeeze_risk", "LOW")).upper()
            score_raw = fnum(v8.get("squeeze_score", 0), 0)
            score = int(max(0, min(100, score_raw * 0.5)))
            note = (
                f"synthetic v8 overlay only; squeeze={risk}; "
                f"score={v8.get('squeeze_score','')}; gamma_flip={v8.get('gamma_flip','')}; "
                f"max_pain={v8.get('max_pain','')}; does not override real L7/L8/L9"
            )
            return "SYNTHETIC_OPTIONS_CONTEXT", score, note
        if ticker in ETF_TO_THEME:
            return "OPTIONS_DATA_UNAVAILABLE", 10, "No fresh options decision row; do not infer gamma or dealer pressure."
        return "NO_LISTED_OPTIONS_CONTEXT", 10, "No options decision row; options layer cannot confirm anything."

    label = str(r.get("final_options_decision", "NO_DATA")).upper()
    if label in {"", "NO_DATA"}:
        return "OPTIONS_DATA_UNAVAILABLE", 10, "Options decision row exists but has no usable decision; do not infer gamma or dealer pressure."
    gamma_score = fnum(r.get("gamma_squeeze_score", 0), 0)
    kill_score = fnum(r.get("option_kill_zone_score", 0), 0)
    score = int(max(0, min(100, gamma_score - max(0, kill_score - 50) * 0.5)))
    note = f"gamma={r.get('gamma_squeeze_label','')}; kill={r.get('option_kill_zone_label','')}; rule={r.get('rule','')}"
    return label, score, note


def layer_l8() -> tuple[str, int, str]:
    state, detail = portfolio_risk()
    score = {"GREEN": 85, "AMBER": 55, "RED": 20}.get(state, 0)
    return state, score, detail


def layer_l9(ticker: str) -> tuple[str, int, str]:
    r = get_row(read_csv(FILES["pre"]), ticker)
    v8 = get_row(read_csv(FILES["v8_l9"]), ticker)
    c = get_row(read_csv(FILES["cards"]), ticker)

    if not r and v8:
        r = v8

    if not r and not c:
        return "NO_DATA", 0, "No execution/pre-trade row"

    status = str(r.get("final_status", c.get("decision", "NO_DATA"))).upper()
    live = str(r.get("live_allowed", c.get("live_allowed", "NO"))).upper()
    paper = str(r.get("paper_allowed", "")).upper()

    if "BLOCKED" in status:
        score = 10
    elif "PENDING" in status:
        score = 45
    elif "RESEARCH_ONLY" in status or "RISK_REDUCTION" in status:
        score = 55
    elif "ALLOW" in status:
        score = 70
    else:
        score = 50

    return status, score, f"paper_allowed={paper}; live_allowed={live}; {c.get('one_liner','')}"


def layer_l10(ticker: str) -> tuple[str, int, str]:
    ledger = read_csv(FILES["ledger"])
    if ledger.empty or "ticker" not in ledger.columns:
        return "LEARNING_SAMPLE_PENDING", 20, "No paper ledger found yet"

    sub = ledger[ledger["ticker"].astype(str).str.upper().str.strip() == ticker]
    if sub.empty:
        return "LEARNING_SAMPLE_PENDING", 25, "No ticker-specific closed paper sample"

    statuses = ", ".join(sorted(set(sub["status"].astype(str)))) if "status" in sub.columns else "HAS_LEDGER"
    closed = int(sub["status"].astype(str).str.contains("CLOSED", case=False, na=False).sum()) if "status" in sub.columns else 0
    score = 60 if closed > 0 else 40
    return "HAS_SAMPLE" if closed > 0 else "OPEN_OR_WATCH_SAMPLE", score, f"ledger_rows={len(sub)}; closed={closed}; statuses={statuses}"


def master_decision(row: dict) -> tuple[str, str]:
    L1, L2, L3, L4, L5, L6, L7, L8, L9, L10 = [str(row.get(f"L{i}_state", "")).upper() for i in range(1, 11)]

    if "NO_PRICE" in L1 or "STALE" in L1:
        return "RESEARCH_ONLY", "L1 data integrity is weak; refresh/verify data before action."

    if "BLOCKED" in L9 or "SKIP" in L7:
        return "SKIP", "Blocked by execution gate or options decision."

    if "ALREADY_CLOSED" in L9:
        return "DO_NOT_REPEAT", "Already closed in paper ledger; do not manufacture repeated samples."

    if L8 == "RED":
        if L7 == "WAIT":
            return "WAIT_ONLY", "Risk is RED and options says WAIT; observe trigger only."
        if L7 == "PAPER_ONLY":
            return "TINY_PAPER_ONLY", "Risk is RED; at most tiny stock/ETF paper, no options."
        return "RISK_REDUCTION_FIRST", "Portfolio risk is RED; reduce concentration before adding new ideas."

    if "RISK_OFF" in L2 and L7 in {"WAIT", "PAPER_ONLY"}:
        return "WAIT_OR_TINY_PAPER", "Macro is hostile/choppy; no aggressive tactical action."

    if L7 == "WAIT":
        return "WAIT_TRIGGER", "Wait for breakout/breakdown confirmation."

    if L7 == "PAPER_ONLY":
        return "PAPER_STOCK_ETF_ONLY", "Use stock/ETF paper only; no short-dated options."

    if "TACTICAL_CANDIDATE" in L6 and L5 in {"EVENT_SUPPORT", "NEUTRAL_OR_NO_EVENT"} and L8 in {"GREEN", "AMBER"}:
        return "TACTICAL_REVIEW", "Technical setup exists; check event and risk before paper."

    if "QUALITY_HOLD_CANDIDATE" in L4 and L8 in {"GREEN", "AMBER"}:
        return "LONG_TERM_REVIEW", "Fundamental layer supports long-term review."

    return "RESEARCH_ONLY", "No full-stack confirmation yet."


def build_matrix() -> pd.DataFrame:
    rows = []
    for t in universe():
        l1 = layer_l1(t)
        l2 = layer_l2()
        l3 = layer_l3(t)
        l4 = layer_l4(t)
        l5 = layer_l5(t)
        l6 = layer_l6(t)
        l7 = layer_l7(t)
        l8 = layer_l8()
        l9 = layer_l9(t)
        l10 = layer_l10(t)

        vals = {
            "ticker": t,
            "L1_state": l1[0], "L1_score": l1[1], "L1_note": l1[2],
            "L2_state": l2[0], "L2_score": l2[1], "L2_note": l2[2],
            "L3_state": l3[0], "L3_score": l3[1], "L3_note": l3[2],
            "L4_state": l4[0], "L4_score": l4[1], "L4_note": l4[2],
            "L5_state": l5[0], "L5_score": l5[1], "L5_note": l5[2],
            "L6_state": l6[0], "L6_score": l6[1], "L6_note": l6[2],
            "L7_state": l7[0], "L7_score": l7[1], "L7_note": l7[2],
            "L8_state": l8[0], "L8_score": l8[1], "L8_note": l8[2],
            "L9_state": l9[0], "L9_score": l9[1], "L9_note": l9[2],
            "L10_state": l10[0], "L10_score": l10[1], "L10_note": l10[2],
        }

        scores = [vals[f"L{i}_score"] for i in range(1, 11)]
        vals["stack_score_avg"] = round(float(np.mean(scores)), 2)
        vals["stack_score_min"] = round(float(np.min(scores)), 2)

        action, reason = master_decision(vals)
        vals["master_action"] = action
        vals["master_reason"] = reason
        rows.append(vals)

    df = pd.DataFrame(rows)
    if not df.empty:
        order = {
            "TACTICAL_REVIEW": 0,
            "LONG_TERM_REVIEW": 1,
            "TINY_PAPER_ONLY": 2,
            "PAPER_STOCK_ETF_ONLY": 3,
            "WAIT_TRIGGER": 4,
            "WAIT_ONLY": 5,
            "WAIT_OR_TINY_PAPER": 6,
            "RESEARCH_ONLY": 7,
            "RISK_REDUCTION_FIRST": 8,
            "DO_NOT_REPEAT": 9,
            "SKIP": 10,
        }
        df["_sort"] = df["master_action"].map(order).fillna(99)
        df = df.sort_values(["_sort", "stack_score_avg", "ticker"], ascending=[True, False, True]).drop(columns=["_sort"])
    return df


def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(1, 11):
        col = f"L{i}_score"
        state = f"L{i}_state"
        if col in df.columns:
            rows.append({
                "layer": f"L{i}",
                "avg_score": round(pd.to_numeric(df[col], errors="coerce").mean(), 2),
                "min_score": round(pd.to_numeric(df[col], errors="coerce").min(), 2),
                "states": ", ".join(sorted(set(df[state].astype(str)))) if state in df.columns else "",
            })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols=None, max_rows=80) -> str:
    if df.empty:
        return "_No data._"
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    try:
        return d.head(max_rows).to_markdown(index=False)
    except Exception:
        return d.head(max_rows).to_string(index=False)


def build_report(df: pd.DataFrame, sc: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 54 — Master 10-Layer Decision Report v2")
    md.append("")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append("")
    md.append("## What changed")
    md.append("")
    md.append("This v2 report actually reads L2-L6 outputs. Options is now only one layer, not the whole system.")
    md.append("")
    md.append("## Master action summary")
    md.append("")
    if df.empty:
        md.append("_No rows._")
    else:
        s = df["master_action"].value_counts().reset_index()
        s.columns = ["master_action", "count"]
        md.append(s.to_markdown(index=False))
    md.append("")
    md.append("## Layer scorecard")
    md.append("")
    md.append(sc.to_markdown(index=False) if not sc.empty else "_No scorecard._")
    md.append("")
    md.append("## Compact matrix")
    md.append("")
    compact = [
        "ticker", "master_action", "master_reason", "stack_score_avg",
        "L1_state", "L2_state", "L3_state", "L4_state", "L5_state",
        "L6_state", "L7_state", "L8_state", "L9_state", "L10_state"
    ]
    md.append(md_table(df, compact, max_rows=100))
    md.append("")
    md.append("## Full matrix")
    md.append("")
    md.append(md_table(df, max_rows=100))
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- L1 data problems block action.")
    md.append("- L8 RED overrides attractive L7 options signals.")
    md.append("- L7 WAIT means no early weekly OTM chase.")
    md.append("- L4 is required before treating a ticker as a long-term hold.")
    md.append("- L5 event risk can block a trade even when technical/options look good.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 54")
    print("Master 10-Layer Decision Matrix v2")
    print("=" * 88)

    df = build_matrix()
    sc = scorecard(df)

    df.to_csv(OUT_MATRIX, index=False)
    sc.to_csv(OUT_SCORECARD, index=False)
    OUT_REPORT.write_text(build_report(df, sc), encoding="utf-8")

    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df[["ticker", "master_action", "stack_score_avg", "L2_state", "L3_state", "L4_state", "L5_state", "L6_state", "L7_state", "L8_state"]].to_string(index=False))
    print()
    print("Files generated:")
    print(f"  {OUT_MATRIX}")
    print(f"  {OUT_REPORT}")
    print(f"  {OUT_SCORECARD}")
    print()
    print("Next: open master_10_layer_decision_report_v2.md")


if __name__ == "__main__":
    main()
