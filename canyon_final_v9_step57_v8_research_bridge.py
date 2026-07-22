#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 57 - v8 Research Bridge

Safely imports Lynn's Canyon v8 legacy source and exposes only research,
diagnostic, and paper-only analytics to Canyon v9.

Hard guardrails:
- No broker connection.
- No live orders.
- AlpacaExecution / LiveTrader / execution algorithms are inventoried only.
- Synthetic options output is a model overlay, not real dealer positioning.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import importlib.util
import math

import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE_CANDIDATES = [
    ROOT / "canyon_final_v8_latest_source.py",
    ROOT / "canyon_final_v8_legacy_source.py",
]
SOURCE = next((path for path in SOURCE_CANDIDATES if path.exists()), SOURCE_CANDIDATES[-1])

OUT_INVENTORY = ROOT / "v8_research_module_inventory.csv"
OUT_BSM = ROOT / "v8_bsm_greeks_overlay.csv"
OUT_OPTIONS = ROOT / "v8_synthetic_options_overlay.csv"
OUT_REPORT = ROOT / "v8_research_bridge_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(value, default=np.nan) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except Exception:
        return default


def load_v8_module():
    if not SOURCE.exists():
        candidates = ", ".join(str(path) for path in SOURCE_CANDIDATES)
        raise FileNotFoundError(f"Missing v8 source. Checked: {candidates}")
    spec = importlib.util.spec_from_file_location("canyon_v8_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("Could not load canyon v8 legacy source")
    spec.loader.exec_module(module)
    return module


def build_inventory(module) -> pd.DataFrame:
    rows = [
        ("AlphaICEngine", "L10 Learning / alpha validation", "AVAILABLE_RESEARCH", "IC, ICIR, t-stat factor validation; not auto-weighted yet."),
        ("AdvancedStatTests", "L10 Learning / statistics", "AVAILABLE_RESEARCH", "PSR, DSR, stationary bootstrap diagnostics."),
        ("FullStatSuite", "L8/L10 risk and validation", "AVAILABLE_RESEARCH", "Comprehensive statistical report wrapper."),
        ("GBMModel", "L8 Portfolio risk", "AVAILABLE_RESEARCH", "Monte Carlo VaR/ES candidate for future stress layer."),
        ("GARCHModel", "L8 Volatility regime", "AVAILABLE_RESEARCH", "Forecast volatility candidate for future risk scaling."),
        ("PCAFactorModel", "L8 Crowding / factor exposure", "AVAILABLE_RESEARCH", "Hidden factor exposure and crowding diagnostics."),
        ("CopulaRiskModel", "L8 Tail risk", "AVAILABLE_RESEARCH", "Joint tail risk candidate for future stress layer."),
        ("BSMGreeks", "L7 Options math", "INTEGRATED_STEP57", "ATM Greeks overlay generated locally."),
        ("GEXEngine", "L7 Options / dealer gamma", "INTEGRATED_STEP57", "Synthetic-chain GEX diagnostics only."),
        ("OptionsSignalEngine", "L7 Options overlay", "INTEGRATED_STEP57", "Synthetic GEX, PCR, IVP, Max Pain overlay. Not real dealer data."),
        ("SectorRotationEngine", "L3 Sector rotation", "AVAILABLE_RESEARCH", "Can inform future L3 scoring improvements."),
        ("MasterRiskLayer", "L8 Master risk", "AVAILABLE_RESEARCH", "Can inform future global scaling and cooldown rules."),
        ("SleeveManager", "Portfolio construction", "AVAILABLE_RESEARCH", "Can inform future sleeve dashboard."),
        ("BookCh5_TrendFollowing", "L6 Technical", "AVAILABLE_RESEARCH", "Book-style trend diagnostics available."),
        ("BookCh6_CrossSectionalMomentum", "L3/L6 Momentum", "AVAILABLE_RESEARCH", "Book-style cross-sectional momentum available."),
        ("BookCh7_BacktestMetrics", "L10 Validation", "AVAILABLE_RESEARCH", "Backtest metric formulas available."),
        ("BookCh8_StatisticalArbitrage", "Research backlog", "AVAILABLE_RESEARCH", "Pairs research only; no execution."),
        ("BookCh9_BayesianOptimizer", "Research backlog", "AVAILABLE_RESEARCH", "Parameter search research only."),
        ("BookV2_OperationManual", "L9 Pre-trade process", "AVAILABLE_RESEARCH", "Checklist logic can inform L9."),
        ("AlpacaExecution", "Broker execution", "BLOCKED_NO_LIVE", "Kept in source archive only. Not imported into workflow."),
        ("LiveTrader", "Live trading loop", "BLOCKED_NO_LIVE", "Explicitly excluded by Canyon v9 no-live-order rule."),
        ("TWAPExecution", "Execution algorithm", "BLOCKED_NO_LIVE", "Not used; future paper-only simulator possible."),
        ("VWAPExecution", "Execution algorithm", "BLOCKED_NO_LIVE", "Not used; future paper-only simulator possible."),
        ("POVExecution", "Execution algorithm", "BLOCKED_NO_LIVE", "Not used; future paper-only simulator possible."),
    ]
    out = pd.DataFrame(rows, columns=["module", "target_layer", "integration_status", "note"])
    out["present_in_source"] = out["module"].map(lambda name: hasattr(module, name))
    return out


def hist_vol_proxy(ticker: str, technical: pd.DataFrame, sector: pd.DataFrame) -> float:
    if not technical.empty and "ticker" in technical.columns:
        row = technical[technical["ticker"].astype(str).str.upper().eq(ticker)]
        if not row.empty:
            atr_pct = fnum(row.iloc[0].get("atr14_pct", np.nan))
            if np.isfinite(atr_pct) and atr_pct > 0:
                return float(np.clip(atr_pct * math.sqrt(252), 0.10, 1.20))

    if not sector.empty and "ticker" in sector.columns:
        row = sector[sector["ticker"].astype(str).str.upper().eq(ticker)]
        if not row.empty:
            ret20 = abs(fnum(row.iloc[0].get("ret_20d", np.nan)))
            if np.isfinite(ret20) and ret20 > 0:
                return float(np.clip(ret20 * math.sqrt(252 / 20), 0.10, 1.20))

    return 0.30


def market_price_table() -> pd.DataFrame:
    market = read_csv(ROOT / "market_data_snapshot.csv")
    master = read_csv(ROOT / "master_10_layer_decision_matrix_v2.csv")
    technical = read_csv(ROOT / "technical_signal_matrix.csv")
    sector = read_csv(ROOT / "sector_rotation_scores.csv")

    rows = []
    if not market.empty and "ticker" in market.columns:
        tickers = market["ticker"].astype(str).str.upper().tolist()
    elif not master.empty and "ticker" in master.columns:
        tickers = master["ticker"].astype(str).str.upper().tolist()
    else:
        tickers = []

    for ticker in tickers:
        row = market[market["ticker"].astype(str).str.upper().eq(ticker)] if not market.empty else pd.DataFrame()
        price = fnum(row.iloc[0].get("best_price_proxy", np.nan)) if not row.empty else np.nan
        if not np.isfinite(price) or price <= 0:
            continue
        rows.append({
            "ticker": ticker,
            "spot": price,
            "hist_vol_proxy": hist_vol_proxy(ticker, technical, sector),
            "price_source": row.iloc[0].get("price_source_file", "") if not row.empty else "",
            "data_confidence": row.iloc[0].get("data_confidence", "") if not row.empty else "",
        })

    return pd.DataFrame(rows)


def build_bsm_overlay(module, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in prices.iterrows():
        ticker = row["ticker"]
        spot = float(row["spot"])
        sigma = float(row["hist_vol_proxy"])
        t_exp = 21 / 365
        strike = round(spot)
        rows.append({
            "ticker": ticker,
            "spot": round(spot, 4),
            "atm_strike": strike,
            "days_to_expiry": 21,
            "hist_vol_proxy": round(sigma, 4),
            "call_delta": round(module.BSMGreeks.delta(spot, strike, 0.05, sigma, t_exp, "call"), 4),
            "put_delta": round(module.BSMGreeks.delta(spot, strike, 0.05, sigma, t_exp, "put"), 4),
            "gamma": round(module.BSMGreeks.gamma(spot, strike, 0.05, sigma, t_exp), 8),
            "vega": round(module.BSMGreeks.vega(spot, strike, 0.05, sigma, t_exp), 4),
            "theta_call_per_day": round(module.BSMGreeks.theta(spot, strike, 0.05, sigma, t_exp, "call"), 4),
            "atm_call_price": round(module.BSMGreeks.call_price(spot, strike, 0.05, sigma, t_exp), 4),
            "source": "v8_BSMGreeks_research_overlay",
        })
    return pd.DataFrame(rows)


def build_synthetic_options_overlay(module, prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    engine = module.OptionsSignalEngine(r=0.05)
    price_frame = pd.DataFrame(
        {row["ticker"]: [float(row["spot"])] for _, row in prices.iterrows()},
        index=[pd.Timestamp(datetime.now().date())],
    )
    hist_vols = pd.Series(
        {row["ticker"]: float(row["hist_vol_proxy"]) for _, row in prices.iterrows()}
    )
    raw = engine.portfolio_options_signals(price_frame, hist_vols)
    if raw.empty:
        return pd.DataFrame()

    out = raw.reset_index().rename(columns={"index": "ticker"})
    out["overlay_status"] = "SYNTHETIC_RESEARCH_ONLY"
    out["decision_use"] = "Do not override real L7, L8 risk, or L9 execution gate."
    keep = [
        "ticker", "overlay_status", "decision_use", "spot", "combined_signal",
        "signal_direction", "squeeze_score", "squeeze_risk", "total_gex",
        "gamma_flip", "gex_regime", "pcr_volume", "pcr_oi", "atm_iv",
        "iv_percentile", "max_pain", "pain_distance", "dealer_flow",
    ]
    return out[[c for c in keep if c in out.columns]]


def write_report(inventory: pd.DataFrame, bsm: pd.DataFrame, options: pd.DataFrame):
    blocked = inventory[inventory["integration_status"].eq("BLOCKED_NO_LIVE")]
    integrated = inventory[inventory["integration_status"].eq("INTEGRATED_STEP57")]

    high_squeeze = pd.DataFrame()
    if not options.empty and "squeeze_risk" in options.columns:
        high_squeeze = options[options["squeeze_risk"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])]

    md = [
        "# Canyon v9 Step 57 - v8 Research Bridge Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Guardrails",
        "- Research dashboard only.",
        "- No broker connection.",
        "- No live orders.",
        "- v8 live trading classes are archived but not called.",
        "- Synthetic options overlay is not real dealer positioning and does not override L7/L8/L9.",
        "",
        "## What Was Integrated",
        f"- v8 source: `{SOURCE.name}`",
        f"- Integrated research modules: {', '.join(integrated['module'].tolist())}",
        f"- Blocked live/execution modules: {', '.join(blocked['module'].tolist())}",
        "",
        "## Outputs",
        f"- `{OUT_INVENTORY.name}` rows: {len(inventory)}",
        f"- `{OUT_BSM.name}` rows: {len(bsm)}",
        f"- `{OUT_OPTIONS.name}` rows: {len(options)}",
        "",
        "## Synthetic Options Watch",
        f"- Medium/High synthetic squeeze rows: {len(high_squeeze)}",
        "",
    ]

    if not high_squeeze.empty:
        cols = [c for c in ["ticker", "squeeze_risk", "squeeze_score", "combined_signal", "gamma_flip", "max_pain"] if c in high_squeeze.columns]
        md.append(high_squeeze[cols].to_markdown(index=False))
        md.append("")

    md.extend([
        "## Next Integration Candidates",
        "- Add GBM / Copula / PCA diagnostics to L8 after a stable historical return table exists.",
        "- Add PSR / DSR summaries to L10 after enough closed paper samples exist.",
        "- Add BookV2 checklist rules to L9 as paper-only pre-trade gates.",
    ])
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")


def main():
    print("=" * 88)
    print("CANYON v9 Step 57")
    print("v8 Research Bridge")
    print("=" * 88)

    module = load_v8_module()
    inventory = build_inventory(module)
    prices = market_price_table()
    bsm = build_bsm_overlay(module, prices)
    options = build_synthetic_options_overlay(module, prices)

    inventory.to_csv(OUT_INVENTORY, index=False)
    bsm.to_csv(OUT_BSM, index=False)
    options.to_csv(OUT_OPTIONS, index=False)
    write_report(inventory, bsm, options)

    print(f"Inventory rows: {len(inventory)}")
    print(f"BSM rows: {len(bsm)}")
    print(f"Synthetic options rows: {len(options)}")
    print()
    print("Files generated:")
    for path in [OUT_INVENTORY, OUT_BSM, OUT_OPTIONS, OUT_REPORT]:
        print(f"  {path}")
    print()
    print("No broker connection. No live order.")


if __name__ == "__main__":
    main()
