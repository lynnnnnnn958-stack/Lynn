#!/usr/bin/env python3
"""
Canyon v9 Step 114 - Drawdown control, volatility target, and Kelly sizing.

Research-only. No broker connection. No live orders.

Outputs:
  portfolio_nav.csv
  drawdown_control_state.json
  vol_target_state.json
  kelly_position_sizing.csv
  drawdown_vol_kelly_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    annualized_vol,
    clean_ticker,
    df_to_markdown,
    get_returns,
    load_current_book,
    portfolio_vol,
    read_csv_safe,
    read_json_safe,
    today_str,
    var_cvar,
    write_json,
    write_markdown_report,
)


NAV_CSV = ROOT / "portfolio_nav.csv"
OUT_DD = ROOT / "drawdown_control_state.json"
OUT_VOL = ROOT / "vol_target_state.json"
OUT_KELLY = ROOT / "kelly_position_sizing.csv"
OUT_MD = ROOT / "drawdown_vol_kelly_report.md"

TARGET_VOL = 0.15
DD_SOFT = 0.10
DD_HARD = 0.15
KELLY_CAP = 0.15
FRACTIONAL_KELLY = 0.25


def load_or_build_nav() -> pd.DataFrame:
    frames = []
    manual = read_csv_safe(ROOT / "live_nav_manual.csv")
    if not manual.empty and "date" in manual.columns:
        tmp = manual.copy()
        if "nav" not in tmp.columns and "account_equity" in tmp.columns:
            equity = pd.to_numeric(tmp["account_equity"], errors="coerce")
            first = equity.dropna().iloc[0] if not equity.dropna().empty else np.nan
            tmp["nav"] = equity / first * 100.0 if np.isfinite(first) and first > 0 else np.nan
        if {"date", "nav"}.issubset(tmp.columns):
            tmp = tmp[["date", "nav"]].copy()
            tmp["source"] = "live_nav_manual.csv"
            frames.append(tmp)

    live_curve = read_csv_safe(ROOT / "live_nav_curve.csv")
    if not live_curve.empty and {"date", "nav"}.issubset(live_curve.columns):
        live_curve = live_curve[["date", "nav"]].copy()
        live_curve["source"] = "live_nav_curve.csv"
        frames.append(live_curve)

    existing = read_csv_safe(NAV_CSV)
    if not existing.empty and {"date", "nav"}.issubset(existing.columns):
        existing = existing[["date", "nav"]].copy()
        existing["source"] = "portfolio_nav.csv"
        frames.append(existing)

    sim = read_csv_safe(ROOT / "paper_sim_nav.csv")
    if not sim.empty and {"date", "nav"}.issubset(sim.columns):
        sim = sim[["date", "nav"]].copy()
        sim["source"] = "paper_sim_nav.csv"
        frames.append(sim)

    if not frames:
        nav = pd.DataFrame([{"date": today_str(), "nav": 100.0, "source": "seed"}])
    else:
        nav = pd.concat(frames, ignore_index=True)

    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["date", "nav"]).sort_values("date")
    if nav.empty:
        nav = pd.DataFrame([{"date": pd.Timestamp(today_str()), "nav": 100.0, "source": "seed"}])
    nav = nav.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    nav["hwm"] = nav["nav"].cummax()
    nav["drawdown_pct"] = nav["nav"] / nav["hwm"] - 1.0
    nav["daily_return"] = nav["nav"].pct_change(fill_method=None).fillna(0.0)
    return nav


def build_drawdown_state(nav: pd.DataFrame) -> dict:
    last = nav.tail(1).iloc[0]
    dd = abs(float(last["drawdown_pct"]))
    if dd >= DD_HARD:
        circuit = "HARD"
        multiplier = 0.20
        action = "REDUCE_ONLY"
        reason = "Drawdown is above 15%; emergency exposure reduction."
    elif dd >= DD_SOFT:
        circuit = "SOFT"
        multiplier = 0.50
        action = "SIZE_DOWN"
        reason = "Drawdown is above 10%; defensive exposure reduction."
    else:
        circuit = "NONE"
        multiplier = 1.00
        action = "CLEAR"
        reason = "Drawdown below circuit thresholds."
    return {
        "date": str(last["date"].date()),
        "nav": round(float(last["nav"]), 6),
        "high_water_mark": round(float(last["hwm"]), 6),
        "drawdown_pct": round(float(last["drawdown_pct"]), 6),
        "drawdown_abs_pct": round(dd, 6),
        "circuit_level": circuit,
        "drawdown_exposure_multiplier": multiplier,
        "drawdown_action": action,
        "reason": reason,
        "source_file": "portfolio_nav.csv; paper_sim_nav.csv",
    }


def build_vol_state(book: pd.DataFrame) -> dict:
    vol = portfolio_vol(book, lookback=252)
    if not np.isfinite(vol) or vol <= 0:
        multiplier = 0.50
        action = "MISSING_DATA_REVIEW"
        reason = "Portfolio volatility unavailable; use defensive scaler until price data is fixed."
    else:
        multiplier = min(1.0, TARGET_VOL / vol)
        if multiplier < 0.50:
            action = "SIZE_DOWN"
        elif multiplier < 1.00:
            action = "REVIEW"
        else:
            action = "CLEAR"
        reason = f"Annualized vol {vol:.2%} vs target {TARGET_VOL:.2%}."
    return {
        "date": today_str(),
        "target_annual_vol": TARGET_VOL,
        "estimated_annual_vol": None if not np.isfinite(vol) else round(float(vol), 6),
        "vol_exposure_multiplier": round(float(multiplier), 6),
        "vol_action": action,
        "reason": reason,
        "source_file": "sp500_price_cache.csv/backtest_price_cache.csv",
    }


def build_kelly_sizing(book: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()
    weights_meta = read_json_safe(ROOT / "signal_weights.json", {})
    n_ic = int(weights_meta.get("n_ic_periods", 0) or 0)
    sample_conf = min(1.0, max(0.15, n_ic / 20.0))
    tickers = book["ticker"].apply(clean_ticker).tolist()
    rets = get_returns(tickers, lookback=504)
    rows = []
    for _, row in book.iterrows():
        ticker = clean_ticker(row["ticker"])
        weight = float(row.get("weight", 0.0))
        alpha = pd.to_numeric(pd.Series([row.get("alpha_score", np.nan)]), errors="coerce").iloc[0]
        series = rets[ticker].dropna() if ticker in rets.columns else pd.Series(dtype=float)
        vol = annualized_vol(series)
        var95, cvar95 = var_cvar(series, alpha=0.95)
        if np.isfinite(alpha):
            edge = max(0.0, (float(alpha) - 50.0) / 50.0) * 0.08
        else:
            edge = 0.0
        if np.isfinite(vol) and vol > 0:
            raw_kelly = edge / (vol ** 2)
            fractional = raw_kelly * FRACTIONAL_KELLY * sample_conf
            recommended = min(KELLY_CAP, max(0.0, fractional))
            if np.isfinite(cvar95) and cvar95 > 0.06:
                recommended = min(recommended, 0.05)
            status = "CLEAR" if recommended >= weight * 0.80 else "SIZE_DOWN"
        else:
            raw_kelly = np.nan
            fractional = np.nan
            recommended = 0.01
            status = "MISSING_DATA_REVIEW"
        rows.append({
            "ticker": ticker,
            "current_weight": weight,
            "current_weight_pct": weight * 100.0,
            "alpha_score": alpha,
            "expected_edge_annual": edge,
            "annual_vol": vol,
            "var_95_1d": var95,
            "cvar_95_1d": cvar95,
            "raw_kelly_weight": raw_kelly,
            "fractional_kelly_weight": fractional,
            "recommended_kelly_weight": recommended,
            "recommended_kelly_weight_pct": recommended * 100.0,
            "kelly_status": status,
            "ic_periods": n_ic,
            "ic_sample_confidence": sample_conf,
            "source_file": str(row.get("source_file", "")),
        })
    return pd.DataFrame(rows).sort_values("recommended_kelly_weight", ascending=False).reset_index(drop=True)


def write_report(nav: pd.DataFrame, dd: dict, vol: dict, kelly: pd.DataFrame) -> None:
    sections = [
        "## Summary",
        "",
        f"- Current NAV: {dd.get('nav')}",
        f"- High-water mark: {dd.get('high_water_mark')}",
        f"- Drawdown: {dd.get('drawdown_abs_pct', 0.0) * 100:.2f}%",
        f"- Drawdown circuit: {dd.get('circuit_level')}",
        f"- Vol target: {vol.get('target_annual_vol', 0.0) * 100:.2f}%",
        f"- Estimated vol: {vol.get('estimated_annual_vol')}",
        f"- Vol scaler: {vol.get('vol_exposure_multiplier')}",
        "",
        "## Logic",
        "",
        "- Drawdown above 10% cuts exposure to 50%; above 15% cuts to 20%.",
        "- Vol above target scales exposure down; missing vol is defensive.",
        "- Kelly sizing is fractional and sample-shrunk because live IC history is young.",
        "",
        "## NAV curve tail",
        "",
        df_to_markdown(nav.tail(10)),
        "",
        "## Kelly sizing",
        "",
        df_to_markdown(kelly, max_rows=20) if not kelly.empty else "No Kelly rows.",
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 114 - Drawdown, Vol Target, Kelly", sections)


def main() -> None:
    book = load_current_book(prefer_filtered=True)
    nav = load_or_build_nav()
    nav.to_csv(NAV_CSV, index=False)
    dd_state = build_drawdown_state(nav)
    vol_state = build_vol_state(book)
    kelly = build_kelly_sizing(book)
    kelly.to_csv(OUT_KELLY, index=False)
    write_json(OUT_DD, dd_state)
    write_json(OUT_VOL, vol_state)
    write_report(nav, dd_state, vol_state, kelly)
    print(f"[step114] wrote {NAV_CSV.name}: {len(nav)} rows")
    print(f"[step114] wrote {OUT_DD.name}")
    print(f"[step114] wrote {OUT_VOL.name}")
    print(f"[step114] wrote {OUT_KELLY.name}: {len(kelly)} rows")
    print(f"[step114] wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
