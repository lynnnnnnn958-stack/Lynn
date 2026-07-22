#!/usr/bin/env python3
"""
Canyon v9 — Step 129: News Impact Targeting
============================================

Research-only. No broker connection. No live orders.

This step converts headline tone into ticker-specific research targets:
  1. Which ticker is the headline actually about?
  2. Is it direct, related, or a broad sector read-through?
  3. If the news is bad, which high-valuation / weak / risky names are most
     vulnerable to being repriced?
  4. If the news is good, which names are reasonable beneficiaries, subject to
     risk gates?

Outputs:
  news_impact_targets.csv
  news_target_watchlist.csv
  news_supply_chain_readthrough.csv
  news_impact_targeting_state.json
  news_impact_targeting_report.md
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

OUT_TARGETS = ROOT / "news_impact_targets.csv"
OUT_WATCHLIST = ROOT / "news_target_watchlist.csv"
OUT_SUPPLY_CHAIN = ROOT / "news_supply_chain_readthrough.csv"
OUT_STATE = ROOT / "news_impact_targeting_state.json"
OUT_REPORT = ROOT / "news_impact_targeting_report.md"


COMPANY_ALIASES: dict[str, list[str]] = {
    "AAPL": ["Apple"],
    "ASTS": ["AST SpaceMobile", "ASTS"],
    "ADBE": ["Adobe"],
    "ADI": ["Analog Devices"],
    "ADM": ["Archer-Daniels", "ADM"],
    "ADP": ["Automatic Data Processing", "ADP"],
    "ADSK": ["Autodesk"],
    "AMD": ["Advanced Micro Devices", "AMD"],
    "AMZN": ["Amazon", "AWS"],
    "ANET": ["Arista", "Arista Networks"],
    "APP": ["AppLovin"],
    "APD": ["Air Products"],
    "AVGO": ["Broadcom"],
    "BA": ["Boeing"],
    "BBY": ["Best Buy"],
    "CHRW": ["C.H. Robinson", "CH Robinson"],
    "DELL": ["Dell"],
    "DXCM": ["DexCom", "Dexcom"],
    "ELV": ["Elevance", "Elevance Health"],
    "EXPE": ["Expedia"],
    "FFIV": ["F5", "F5 Networks"],
    "FIX": ["Comfort Systems"],
    "GOOG": ["Google", "Alphabet"],
    "GOOGL": ["Google", "Alphabet"],
    "GSAT": ["Globalstar"],
    "IRDM": ["Iridium"],
    "INTC": ["Intel"],
    "LHX": ["L3Harris", "L3 Harris"],
    "LMT": ["Lockheed", "Lockheed Martin"],
    "LUNR": ["Intuitive Machines"],
    "META": ["Meta", "Facebook"],
    "MSFT": ["Microsoft"],
    "NOC": ["Northrop", "Northrop Grumman"],
    "NVDA": ["Nvidia", "NVIDIA"],
    "PL": ["Planet Labs"],
    "RDW": ["Redwire"],
    "RKLB": ["Rocket Lab", "RocketLab"],
    "RTX": ["RTX", "Raytheon"],
    "SPCE": ["Virgin Galactic"],
    "TSLA": ["Tesla"],
    "VSAT": ["Viasat"],
}

COMMON_WORD_TICKERS = {
    "A", "ALL", "ARE", "BE", "C", "D", "F", "HAS", "IT", "L", "ON", "O", "T", "V",
}


SUPPLY_CHAIN_THEMES: dict[str, dict[str, Any]] = {
    "Space / Launch": {
        "keywords": [
            "spacex", "space x", "rocket", "rocket launch", "space launch",
            "launch vehicle", "orbital launch", "satellite launch", "starship",
            "satellite", "orbital", "spacecraft", "space economy", "lunar", "mars",
        ],
        "sector": "Aerospace / Space",
        "upstream": [
            ("BA", "Large aerospace manufacturer; supplier and program read-through."),
            ("LMT", "Defense and aerospace prime contractor with space systems exposure."),
            ("NOC", "Defense and space systems prime contractor."),
            ("RTX", "Aerospace and defense supplier with propulsion/systems exposure."),
            ("LHX", "Space, defense electronics, and communications supplier."),
            ("TDY", "High-end aerospace and sensing component supplier."),
            ("HEI", "Aerospace components supplier."),
            ("HON", "Aerospace systems and components supplier."),
        ],
        "peer": [
            ("RKLB", "Rocket Lab is the cleanest public launch peer read-through."),
            ("LUNR", "Intuitive Machines is a high-beta public space services/lunar peer."),
            ("ASTS", "AST SpaceMobile is a satellite-to-phone space connectivity peer."),
            ("RDW", "Redwire is a public space infrastructure peer."),
            ("PL", "Planet Labs is a public satellite data and imagery peer."),
            ("SPCE", "Virgin Galactic is a speculative space sympathy ticker."),
            ("IRDM", "Iridium is a satellite communications peer."),
            ("GSAT", "Globalstar is a satellite communications peer."),
            ("VSAT", "Viasat is a satellite and network infrastructure peer."),
            ("SATS", "EchoStar is a satellite and network infrastructure peer."),
        ],
        "downstream": [
            ("TMUS", "Potential downstream beneficiary of satellite-to-phone distribution."),
            ("VZ", "Potential downstream beneficiary of satellite connectivity distribution."),
            ("T", "Potential downstream beneficiary of satellite connectivity distribution."),
            ("GOOGL", "Cloud, mapping, AI, and data infrastructure adjacency."),
        ],
    },
    "AI / Data Center": {
        "keywords": [
            "artificial intelligence", " ai ", "generative ai", "gpu", "accelerator",
            "data center", "datacenter", "cloud capex", "training cluster",
        ],
        "sector": "AI Infrastructure",
        "upstream": [
            ("ASML", "Semiconductor equipment bottleneck for advanced chips."),
            ("AMAT", "Semiconductor equipment supplier."),
            ("LRCX", "Semiconductor equipment supplier."),
            ("KLAC", "Semiconductor inspection and process control supplier."),
        ],
        "peer": [
            ("NVDA", "GPU leader and AI compute bellwether."),
            ("AMD", "AI accelerator peer."),
            ("AVGO", "Networking/custom silicon AI infrastructure read-through."),
            ("SMCI", "High-beta AI server hardware read-through."),
            ("ANET", "Data-center networking read-through."),
            ("DELL", "AI server and enterprise hardware read-through."),
        ],
        "downstream": [
            ("MSFT", "Cloud and AI platform beneficiary."),
            ("GOOGL", "Cloud and AI platform beneficiary."),
            ("AMZN", "AWS cloud and AI platform beneficiary."),
            ("META", "AI infrastructure and advertising platform beneficiary."),
        ],
    },
    "Defense / Security": {
        "keywords": [
            "defense", "missile", "pentagon", "geopolitical", "drone",
            "national security", "military contract", "defence",
        ],
        "sector": "Defense",
        "upstream": [
            ("RTX", "Defense systems and aerospace supplier."),
            ("LHX", "Defense electronics and communications supplier."),
            ("TDY", "Sensing and aerospace components supplier."),
        ],
        "peer": [
            ("LMT", "Defense prime contractor."),
            ("NOC", "Defense and aerospace prime contractor."),
            ("GD", "Defense prime contractor."),
            ("BA", "Defense and aerospace prime contractor."),
        ],
        "downstream": [
            ("ITA", "Aerospace and defense ETF proxy."),
            ("XAR", "Aerospace and defense ETF proxy."),
        ],
    },
    "Energy / Nuclear": {
        "keywords": [
            "nuclear", "uranium", "power demand", "electricity demand",
            "grid", "power plant", "natural gas", "lng",
        ],
        "sector": "Power / Energy",
        "upstream": [
            ("CCJ", "Uranium supply read-through."),
            ("URA", "Uranium ETF proxy."),
            ("XLE", "Energy sector proxy."),
        ],
        "peer": [
            ("CEG", "Nuclear and power generation read-through."),
            ("VST", "Power generation read-through."),
            ("NRG", "Power generation read-through."),
        ],
        "downstream": [
            ("NVDA", "AI/data-center power demand adjacency."),
            ("MSFT", "Cloud/data-center power demand adjacency."),
            ("AMZN", "Cloud/data-center power demand adjacency."),
            ("GOOGL", "Cloud/data-center power demand adjacency."),
        ],
    },
}


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json_safe(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def percentile(series: pd.Series, higher_is_risk: bool = True, neutral: float = 50.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() < 3:
        return pd.Series(neutral, index=series.index)
    ranks = vals.rank(pct=True) * 100
    if not higher_is_risk:
        ranks = 100 - ranks
    return ranks.fillna(neutral).clip(0, 100)


def norm_score(value: Any, default: float = 50.0) -> float:
    try:
        v = float(value)
        if np.isnan(v):
            return default
        if 0 <= v <= 1:
            return v * 100
        return v
    except Exception:
        return default


def risk_gate_score(action: Any) -> float:
    text = str(action).upper()
    if "REDUCE_ONLY" in text or "BLOCK" in text:
        return 100.0
    if "SIZE_DOWN" in text:
        return 78.0
    if "REVIEW" in text:
        return 60.0
    if "CLEAR" in text:
        return 20.0
    return 45.0


def severity_score(value: Any) -> float:
    text = str(value).upper()
    if "CRITICAL" in text:
        return 100.0
    if "WARNING" in text or "URGENT" in text:
        return 75.0
    if "INFO" in text or "WATCH" in text:
        return 45.0
    return 20.0


def clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def ticker_pattern(term: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)


def keyword_hit(text: str, keyword: str) -> bool:
    text_l = f" {text.lower()} "
    key = keyword.lower()
    if key.strip() != key:
        return key in text_l
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text_l))


def detect_supply_chain_themes(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for theme, spec in SUPPLY_CHAIN_THEMES.items():
        matched = [kw for kw in spec.get("keywords", []) if keyword_hit(text, kw)]
        if theme == "Space / Launch" and matched:
            headline_zone = text[:220].lower()
            hard_space_terms = {
                "rocket", "rocket launch", "space launch", "launch vehicle",
                "orbital launch", "satellite launch", "starship", "satellite",
                "orbital", "spacecraft", "space economy", "lunar", "mars",
            }
            title_has_spacex = "spacex" in headline_zone or "space x" in headline_zone
            if not title_has_spacex and not any(term in hard_space_terms for term in matched):
                continue
        if matched:
            hits.append({
                "theme": theme,
                "sector": spec.get("sector", theme),
                "matched_theme_terms": ", ".join(matched[:8]),
                "spec": spec,
            })
    return hits


def find_mentioned_tickers(text: str, universe: set[str]) -> list[str]:
    found: list[str] = []
    for ticker in sorted(universe):
        aliases: list[str] = []
        # Short tickers such as IT, ON, A, O, T create false positives in prose.
        # Only match them when the headline uses ticker notation like "(T)".
        if len(ticker) >= 3 and ticker not in COMMON_WORD_TICKERS:
            aliases.append(ticker)
        elif re.search(rf"\({re.escape(ticker)}\)", text):
            found.append(ticker)
            continue
        aliases.extend([a for a in COMPANY_ALIASES.get(ticker, []) if len(a) >= 4])
        for alias in aliases:
            if ticker_pattern(alias).search(text):
                found.append(ticker)
                break
    return list(dict.fromkeys(found))


def is_external_theme_target(target: str, base_idx: pd.DataFrame) -> bool:
    return clean_ticker(target) not in base_idx.index


def base_or_external_row(base_idx: pd.DataFrame, target: str, theme_sector: str) -> tuple[pd.Series, str]:
    target = clean_ticker(target)
    if target in base_idx.index:
        return base_idx.loc[target], "IN_UNIVERSE"
    return pd.Series({
        "ticker": target,
        "sector": theme_sector or "External Theme",
        "valuation_vulnerability": 50.0,
        "weakness_vulnerability": 55.0,
        "risk_vulnerability": 55.0,
        "volatility_vulnerability": 65.0,
        "option_vulnerability": 50.0,
        "total_vulnerability": 56.0,
        "predicted_score": np.nan,
        "alpha_score": np.nan,
        "final_risk_action": "UNKNOWN_NEEDS_DATA",
        "max_monitor_severity": "NEEDS_DATA",
        "options_strategy": "",
    }), "EXTERNAL_THEME_TARGET_NEEDS_DATA"


def theme_route_for(direction: str, data_status: str, vulnerability: float, risk_action: Any) -> str:
    risk_text = str(risk_action).upper()
    if direction == "POSITIVE":
        if data_status != "IN_UNIVERSE":
            return "THEME_WATCHLIST_CALL_REVIEW_AFTER_DATA"
        if "REDUCE" in risk_text or "BLOCK" in risk_text:
            return "WATCH_ONLY_RISK_BLOCKED"
        if vulnerability <= 65:
            return "THEME_STOCK_OR_CALL_REVIEW"
        return "THEME_STOCK_ONLY_HIGH_VALUATION"
    if direction == "NEGATIVE":
        if data_status != "IN_UNIVERSE":
            return "THEME_WATCHLIST_PUT_REVIEW_AFTER_DATA"
        if vulnerability >= 60:
            return "THEME_PUT_OR_REDUCE_REVIEW"
        return "THEME_RISK_WATCH"
    if direction == "MIXED":
        return "THEME_MANUAL_REVIEW"
    return "THEME_CONTEXT_REVIEW"


def option_side_for_route(route: str) -> str:
    text = str(route).upper()
    if "PUT" in text:
        return "PUT_REVIEW"
    if "CALL" in text:
        return "CALL_REVIEW"
    return "NONE"


def readthrough_reason(theme: str, role: str, note: str, direction: str) -> str:
    direction_text = {
        "POSITIVE": "Positive catalyst",
        "NEGATIVE": "Negative catalyst",
        "MIXED": "Mixed catalyst",
        "NEUTRAL": "Context headline",
    }.get(direction, "Context headline")
    return f"{direction_text} in {theme}; {role} read-through. {note}"


def load_base_universe() -> pd.DataFrame:
    regime = read_csv_safe(ROOT / "regime_ml_scores.csv")
    alpha = read_csv_safe(ROOT / "alpha_scores.csv")
    sector_map = read_csv_safe(ROOT / "sector_map.csv")
    fundamentals = read_csv_safe(ROOT / "fundamental_features.csv")
    risk_gate = read_csv_safe(ROOT / "final_risk_gate.csv")
    monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    options = read_csv_safe(ROOT / "options_signals.csv")

    if not regime.empty and "ticker" in regime.columns:
        base = regime.copy()
    elif not alpha.empty and "ticker" in alpha.columns:
        base = alpha.copy()
    else:
        base = pd.DataFrame(columns=["ticker"])

    base["ticker"] = base["ticker"].map(clean_ticker)
    base = base.drop_duplicates("ticker").reset_index(drop=True)

    if not alpha.empty and "ticker" in alpha.columns:
        alpha = alpha.copy()
        alpha["ticker"] = alpha["ticker"].map(clean_ticker)
        keep = [c for c in ["ticker", "alpha_score", "alpha_rank", "top_signal"] if c in alpha.columns]
        base = base.merge(alpha[keep], on="ticker", how="left", suffixes=("", "_alpha"))

    if not sector_map.empty and "ticker" in sector_map.columns:
        sector_map = sector_map.copy()
        sector_map["ticker"] = sector_map["ticker"].map(clean_ticker)
        keep = [c for c in ["ticker", "sector"] if c in sector_map.columns]
        base = base.merge(sector_map[keep].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_map"))
        if "sector_map" in base.columns:
            current_sector = base.get("sector", pd.Series(index=base.index)).astype(str)
            mapped_sector = base["sector_map"]
            base["sector"] = current_sector.where(
                current_sector.notna() & ~current_sector.isin(["", "nan", "None", "Other"]),
                mapped_sector,
            )
            base = base.drop(columns=["sector_map"])

    if not fundamentals.empty and "ticker" in fundamentals.columns:
        fundamentals = fundamentals.copy()
        fundamentals["ticker"] = fundamentals["ticker"].map(clean_ticker)
        keep = [c for c in ["ticker", "pe_ratio", "pb_ratio", "ps_ratio", "market_cap", "analyst_upside"] if c in fundamentals.columns]
        base = base.merge(fundamentals[keep].drop_duplicates("ticker"), on="ticker", how="left")

    if not risk_gate.empty and "ticker" in risk_gate.columns:
        risk_gate = risk_gate.copy()
        risk_gate["ticker"] = risk_gate["ticker"].map(clean_ticker)
        keep = [c for c in ["ticker", "final_risk_action", "recommended_risk_weight_pct", "reason_stack"] if c in risk_gate.columns]
        base = base.merge(risk_gate[keep].drop_duplicates("ticker"), on="ticker", how="left")

    if not monitor.empty and "ticker" in monitor.columns:
        monitor = monitor.copy()
        monitor["ticker"] = monitor["ticker"].map(clean_ticker)
        keep = [c for c in ["ticker", "max_monitor_severity", "event_count", "price_break_state", "volume_spike_state", "volatility_regime_state"] if c in monitor.columns]
        base = base.merge(monitor[keep].drop_duplicates("ticker"), on="ticker", how="left")

    if not options.empty and "ticker" in options.columns:
        options = options.copy()
        options["ticker"] = options["ticker"].map(clean_ticker)
        keep = [c for c in ["ticker", "iv_rank", "rank_options", "uoa_bear_flag", "pcr_vol", "options_strategy"] if c in options.columns]
        base = base.merge(options[keep].drop_duplicates("ticker"), on="ticker", how="left")

    if "sector" not in base.columns:
        base["sector"] = "Other"
    base["sector"] = base["sector"].fillna("Other")

    # Valuation vulnerability: high multiples get hit harder when risk appetite falls.
    pe = percentile(base.get("pe_ratio", pd.Series(index=base.index)), True)
    pb = percentile(base.get("pb_ratio", pd.Series(index=base.index)), True)
    ps = percentile(base.get("ps_ratio", pd.Series(index=base.index)), True)
    base["valuation_vulnerability"] = (0.45 * pe + 0.25 * pb + 0.30 * ps).round(2)

    # Weakness vulnerability: weak ML score, bad trend, and negative momentum.
    pred = pd.to_numeric(base.get("predicted_score", pd.Series(index=base.index)), errors="coerce")
    pred_scaled = pred.where(pred > 1, pred * 100)
    pred_risk = 100 - pred_scaled.fillna(50).clip(0, 100)
    trend = pd.to_numeric(base.get("trend_200", pd.Series(index=base.index)), errors="coerce")
    mom1 = pd.to_numeric(base.get("mom_1m", pd.Series(index=base.index)), errors="coerce")
    mom3 = pd.to_numeric(base.get("mom_3m", pd.Series(index=base.index)), errors="coerce")
    base["weakness_vulnerability"] = (
        0.50 * pred_risk
        + 20 * (trend < 0).astype(float)
        + 15 * (mom1 < 0).astype(float)
        + 15 * (mom3 < 0).astype(float)
    ).clip(0, 100).round(2)

    inv_vol = pd.to_numeric(base.get("inv_vol", pd.Series(index=base.index)), errors="coerce")
    base["volatility_vulnerability"] = percentile(inv_vol, higher_is_risk=False).round(2)

    gate_risk = base.get("final_risk_action", pd.Series("", index=base.index)).map(risk_gate_score)
    monitor_risk = base.get("max_monitor_severity", pd.Series("", index=base.index)).map(severity_score)
    base["risk_vulnerability"] = (0.60 * gate_risk + 0.40 * monitor_risk).round(2)

    iv_risk = percentile(base.get("iv_rank", pd.Series(index=base.index)), True)
    uoa_bear = pd.to_numeric(base.get("uoa_bear_flag", pd.Series(0, index=base.index)), errors="coerce").fillna(0) * 25
    base["option_vulnerability"] = (iv_risk + uoa_bear).clip(0, 100).round(2)

    base["total_vulnerability"] = (
        0.30 * base["valuation_vulnerability"]
        + 0.30 * base["weakness_vulnerability"]
        + 0.20 * base["risk_vulnerability"]
        + 0.10 * base["volatility_vulnerability"]
        + 0.10 * base["option_vulnerability"]
    ).round(2)

    return base


def route_for(direction: str, vulnerability: float, risk_action: Any, options_strategy: Any) -> str:
    risk_text = str(risk_action).upper()
    opt_text = str(options_strategy or "").upper()
    if direction == "NEGATIVE":
        if vulnerability >= 75:
            return "PUT_HEDGE_REVIEW" if opt_text and opt_text != "NAN" else "AVOID_OR_REDUCE"
        if vulnerability >= 60 or "REDUCE" in risk_text or "SIZE_DOWN" in risk_text:
            return "AVOID_OR_REDUCE"
        return "WATCH_NEGATIVE_CONFIRMATION"
    if direction == "POSITIVE":
        if "REDUCE" in risk_text or "BLOCK" in risk_text:
            return "WATCH_ONLY_RISK_BLOCKED"
        if vulnerability <= 55:
            return "STOCK_OR_CALL_REVIEW" if opt_text and opt_text != "NAN" else "STOCK_REVIEW"
        return "WATCH_ONLY_HIGH_VALUATION"
    if direction == "MIXED":
        return "MANUAL_REVIEW"
    return "CONTEXT_ONLY"


def target_reason(row: pd.Series, direction: str) -> str:
    bits = [
        f"valuation risk {row.get('valuation_vulnerability', 50):.0f}",
        f"weakness risk {row.get('weakness_vulnerability', 50):.0f}",
        f"risk gate {row.get('final_risk_action', 'unknown')}",
        f"monitor {row.get('max_monitor_severity', 'none')}",
    ]
    if direction == "NEGATIVE":
        return "Bad news tends to hit high-multiple, weak, crowded, or risk-blocked names first; " + "; ".join(bits) + "."
    if direction == "POSITIVE":
        return "Good news is useful only if price/risk confirms; " + "; ".join(bits) + "."
    return "Headline has no clear standalone direction; " + "; ".join(bits) + "."


def flatten_news(news: dict[str, Any]) -> pd.DataFrame:
    rows = []
    news_map = news.get("news", {}) if isinstance(news, dict) else {}
    for source_ticker, items in news_map.items():
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "").strip()
            link = str(item.get("link", "") or "").strip()
            if not title and not link:
                continue
            rows.append({
                "source_news_ticker": clean_ticker(source_ticker),
                "headline": title,
                "summary": str(item.get("summary", "") or "").strip(),
                "market_tone": str(item.get("market_tone", "NEUTRAL")).upper(),
                "impact_score": norm_score(item.get("impact_score", 0), 0),
                "news_logic": str(item.get("news_logic", "") or ""),
                "action_hint": str(item.get("action_hint", "") or ""),
                "published": str(item.get("published", "") or ""),
                "publisher": str(item.get("publisher", "") or ""),
                "link": link,
                "matched_terms": ", ".join(item.get("matched_terms", []) or []),
                "affected_layers": ", ".join(item.get("affected_layers", []) or []),
            })
    return pd.DataFrame(rows)


def make_target_row(
    target: str,
    relation: str,
    headline: pd.Series,
    mentioned: list[str],
    source_ticker: str,
    direction: str,
    tone: str,
    trow: pd.Series,
    data_status: str,
    theme: str = "",
    chain_role: str = "",
    matched_theme_terms: str = "",
    reason_override: str = "",
    source_file: str = "stock_news.json / regime_ml_scores.csv / fundamental_features.csv / final_risk_gate.csv / desk_monitor_ticker_state.csv / options_signals.csv",
) -> dict[str, Any]:
    vulnerability = float(trow.get("total_vulnerability", 50))
    if relation.startswith("theme"):
        route = theme_route_for(direction, data_status, vulnerability, trow.get("final_risk_action", ""))
    else:
        route = route_for(direction, vulnerability, trow.get("final_risk_action", ""), trow.get("options_strategy", ""))
    option_side = option_side_for_route(route)
    return {
        "target_ticker": clean_ticker(target),
        "target_relation": relation,
        "source_news_ticker": source_ticker,
        "mentioned_tickers": ", ".join(mentioned),
        "headline": headline["headline"],
        "published": headline.get("published", ""),
        "publisher": headline.get("publisher", ""),
        "market_tone": tone,
        "impact_score": headline.get("impact_score", 0),
        "news_logic": headline.get("news_logic", ""),
        "action_hint": headline.get("action_hint", ""),
        "matched_terms": headline.get("matched_terms", ""),
        "affected_layers": headline.get("affected_layers", ""),
        "theme": theme,
        "chain_role": chain_role,
        "matched_theme_terms": matched_theme_terms,
        "target_sector": trow.get("sector", "Other"),
        "valuation_vulnerability": trow.get("valuation_vulnerability", 50),
        "weakness_vulnerability": trow.get("weakness_vulnerability", 50),
        "risk_vulnerability": trow.get("risk_vulnerability", 50),
        "volatility_vulnerability": trow.get("volatility_vulnerability", 50),
        "option_vulnerability": trow.get("option_vulnerability", 50),
        "total_vulnerability": vulnerability,
        "predicted_score": trow.get("predicted_score", np.nan),
        "alpha_score": trow.get("alpha_score", np.nan),
        "final_risk_action": trow.get("final_risk_action", "UNKNOWN"),
        "monitor_severity": trow.get("max_monitor_severity", "OK"),
        "options_strategy": trow.get("options_strategy", ""),
        "suggested_research_route": route,
        "option_side": option_side,
        "target_reason": reason_override or target_reason(trow, direction),
        "data_status": data_status,
        "link": headline.get("link", ""),
        "research_only": True,
        "no_broker_connection": True,
        "source_file": source_file,
    }


def build_targets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = load_base_universe()
    news = flatten_news(read_json_safe(ROOT / "stock_news.json"))
    if base.empty or news.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "NO_DATA"}

    universe = set(base["ticker"].dropna().astype(str))
    base_idx = base.set_index("ticker", drop=False)
    rows: list[dict[str, Any]] = []

    for _, headline in news.iterrows():
        text = f"{headline.get('headline','')} {headline.get('summary','')}"
        mentioned = find_mentioned_tickers(text, universe)
        source_ticker = clean_ticker(headline["source_news_ticker"])
        direct_ticker = mentioned[0] if mentioned else source_ticker
        if direct_ticker not in base_idx.index and source_ticker in base_idx.index:
            direct_ticker = source_ticker

        tone = str(headline.get("market_tone", "NEUTRAL")).upper()
        direction = "POSITIVE" if tone == "POSITIVE" else ("NEGATIVE" if tone == "NEGATIVE" else ("MIXED" if tone == "MIXED" else "NEUTRAL"))
        impact_abs = abs(float(headline.get("impact_score", 0) or 0))

        target_candidates: list[tuple[str, str]] = []
        if direct_ticker in base_idx.index:
            relation = "direct" if direct_ticker == source_ticker else "mentioned"
            target_candidates.append((direct_ticker, relation))

        if direction == "NEGATIVE" and impact_abs >= 3:
            sector = str(base_idx.loc[direct_ticker, "sector"]) if direct_ticker in base_idx.index else ""
            peer_pool = base[base["sector"].astype(str) == sector].copy() if sector else pd.DataFrame()
            if peer_pool.empty:
                peer_pool = base.copy()
            peer_pool = peer_pool[peer_pool["ticker"] != direct_ticker].sort_values("total_vulnerability", ascending=False).head(8)
            target_candidates.extend([(str(t), "vulnerable peer") for t in peer_pool["ticker"].tolist()])

        if direction == "POSITIVE" and impact_abs >= 3:
            sector = str(base_idx.loc[direct_ticker, "sector"]) if direct_ticker in base_idx.index else ""
            peer_pool = base[base["sector"].astype(str) == sector].copy() if sector else pd.DataFrame()
            if not peer_pool.empty:
                score_col = "predicted_score" if "predicted_score" in peer_pool.columns else "alpha_score"
                peer_pool[score_col] = pd.to_numeric(peer_pool[score_col], errors="coerce")
                peer_pool = peer_pool[peer_pool["ticker"] != direct_ticker].sort_values(score_col, ascending=False).head(5)
                target_candidates.extend([(str(t), "beneficiary peer") for t in peer_pool["ticker"].tolist()])

        seen = set()
        for target, relation in target_candidates:
            if target in seen or target not in base_idx.index:
                continue
            seen.add(target)
            trow = base_idx.loc[target]
            rows.append(make_target_row(
                target=target,
                relation=relation,
                headline=headline,
                mentioned=mentioned,
                source_ticker=source_ticker,
                direction=direction,
                tone=tone,
                trow=trow,
                data_status="IN_UNIVERSE",
            ))

        theme_hits = detect_supply_chain_themes(text)
        for hit in theme_hits:
            spec = hit["spec"]
            theme = str(hit["theme"])
            theme_sector = str(hit.get("sector", theme))
            matched_theme_terms = str(hit.get("matched_theme_terms", ""))
            theme_seen = set()
            for role in ["upstream", "peer", "downstream"]:
                for target, note in spec.get(role, []):
                    target = clean_ticker(target)
                    if target in theme_seen or target == direct_ticker:
                        continue
                    theme_seen.add(target)
                    trow, data_status = base_or_external_row(base_idx, target, theme_sector)
                    reason = readthrough_reason(theme, role, note, direction)
                    rows.append(make_target_row(
                        target=target,
                        relation=f"theme {role}",
                        headline=headline,
                        mentioned=mentioned,
                        source_ticker=source_ticker,
                        direction=direction,
                        tone=tone,
                        trow=trow,
                        data_status=data_status,
                        theme=theme,
                        chain_role=role,
                        matched_theme_terms=matched_theme_terms,
                        reason_override=reason,
                        source_file="stock_news.json / theme_readthrough_map / regime_ml_scores.csv / final_risk_gate.csv",
                    ))

    targets = pd.DataFrame(rows)
    if targets.empty:
        return targets, pd.DataFrame(), pd.DataFrame(), {"status": "NO_TARGETS"}

    dedupe_cols = [
        c for c in ["target_ticker", "target_relation", "headline", "theme", "chain_role", "market_tone"]
        if c in targets.columns
    ]
    targets = targets.drop_duplicates(dedupe_cols, keep="first").reset_index(drop=True)

    targets = targets.sort_values(
        ["market_tone", "impact_score", "total_vulnerability"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    watch_rows = []
    for ticker, grp in targets.groupby("target_ticker"):
        neg = grp[grp["market_tone"] == "NEGATIVE"]
        pos = grp[grp["market_tone"] == "POSITIVE"]
        mixed = grp[grp["market_tone"] == "MIXED"]
        worst = neg.sort_values(["impact_score", "total_vulnerability"], ascending=[True, False]).head(1)
        best = pos.sort_values(["impact_score", "total_vulnerability"], ascending=[False, True]).head(1)
        top_row = worst.iloc[0] if not worst.empty else (best.iloc[0] if not best.empty else grp.iloc[0])
        watch_rows.append({
            "target_ticker": ticker,
            "sector": top_row.get("target_sector", "Other"),
            "theme": top_row.get("theme", ""),
            "chain_role": top_row.get("chain_role", ""),
            "data_status": top_row.get("data_status", "IN_UNIVERSE"),
            "negative_headline_count": len(neg),
            "positive_headline_count": len(pos),
            "mixed_headline_count": len(mixed),
            "max_negative_vulnerability": float(neg["total_vulnerability"].max()) if not neg.empty else 0.0,
            "max_positive_score": float(pos["impact_score"].max()) if not pos.empty else 0.0,
            "suggested_research_route": top_row.get("suggested_research_route", "CONTEXT_ONLY"),
            "option_side": top_row.get("option_side", "NONE"),
            "final_risk_action": top_row.get("final_risk_action", "UNKNOWN"),
            "monitor_severity": top_row.get("monitor_severity", "OK"),
            "top_headline": top_row.get("headline", ""),
            "why_this_ticker": top_row.get("target_reason", ""),
        })
    watchlist = pd.DataFrame(watch_rows).sort_values(
        ["negative_headline_count", "max_negative_vulnerability", "positive_headline_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    supply_chain = targets[
        targets["target_relation"].astype(str).str.startswith("theme", na=False)
    ].copy()

    state = {
        "status": "OK",
        "run_time": datetime.now().replace(microsecond=0).isoformat(),
        "headline_count": int(len(news)),
        "target_rows": int(len(targets)),
        "watchlist_tickers": int(len(watchlist)),
        "negative_target_rows": int((targets["market_tone"] == "NEGATIVE").sum()),
        "positive_target_rows": int((targets["market_tone"] == "POSITIVE").sum()),
        "high_vulnerability_negative_rows": int(((targets["market_tone"] == "NEGATIVE") & (targets["total_vulnerability"] >= 75)).sum()),
        "supply_chain_rows": int(len(supply_chain)),
        "external_theme_targets": int((targets.get("data_status", pd.Series(dtype=str)).astype(str) == "EXTERNAL_THEME_TARGET_NEEDS_DATA").sum()),
        "themes_triggered": sorted([str(x) for x in supply_chain.get("theme", pd.Series(dtype=str)).dropna().unique().tolist()]),
        "research_only": True,
        "no_broker_connection": True,
    }
    return targets, watchlist, supply_chain, state


def write_report(targets: pd.DataFrame, watchlist: pd.DataFrame, supply_chain: pd.DataFrame, state: dict[str, Any]) -> None:
    lines = [
        "# Canyon v9 — Step 129 News Impact Targeting",
        f"Generated: {state.get('run_time', datetime.now().isoformat())}",
        "",
        "Research-only. No broker connection. No live orders.",
        "",
        "## What This Step Adds",
        "- Connects every headline to a target ticker.",
        "- Separates direct ticker news from related / peer read-throughs.",
        "- Adds supply-chain / theme read-throughs for major catalysts, including SpaceX -> RKLB-style links.",
        "- For bad news, ranks which high-valuation, weak, risky names are most vulnerable.",
        "- Suggests research routes such as PUT_HEDGE_REVIEW, AVOID_OR_REDUCE, or WATCH_ONLY.",
        "",
        "## State",
        f"- Headlines scanned: {state.get('headline_count', 0)}",
        f"- Target rows: {state.get('target_rows', 0)}",
        f"- Watchlist tickers: {state.get('watchlist_tickers', 0)}",
        f"- Negative target rows: {state.get('negative_target_rows', 0)}",
        f"- High-vulnerability negative rows: {state.get('high_vulnerability_negative_rows', 0)}",
        f"- Supply-chain read-through rows: {state.get('supply_chain_rows', 0)}",
        f"- External theme targets needing data: {state.get('external_theme_targets', 0)}",
        f"- Themes triggered: {', '.join(state.get('themes_triggered', []) or []) or 'none'}",
        "",
    ]
    if not watchlist.empty:
        lines += [
            "## Top News Target Watchlist",
            watchlist.head(25).to_markdown(index=False),
            "",
        ]
    if not targets.empty:
        lines += [
            "## Highest Negative Vulnerability Rows",
            targets[targets["market_tone"] == "NEGATIVE"].sort_values("total_vulnerability", ascending=False).head(25).to_markdown(index=False),
            "",
        ]
    if not supply_chain.empty:
        show_cols = [
            "theme", "chain_role", "target_ticker", "target_relation", "market_tone",
            "source_news_ticker", "headline", "suggested_research_route",
            "option_side", "data_status", "target_reason",
        ]
        show_cols = [c for c in show_cols if c in supply_chain.columns]
        lines += [
            "## Supply Chain / Theme Read-Through",
            supply_chain[show_cols].head(60).to_markdown(index=False),
            "",
        ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("[step129] building news impact target map")
    targets, watchlist, supply_chain, state = build_targets()
    targets.to_csv(OUT_TARGETS, index=False)
    watchlist.to_csv(OUT_WATCHLIST, index=False)
    supply_chain.to_csv(OUT_SUPPLY_CHAIN, index=False)
    OUT_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    write_report(targets, watchlist, supply_chain, state)
    print(f"[step129] wrote {OUT_TARGETS.name}: {len(targets)} rows")
    print(f"[step129] wrote {OUT_WATCHLIST.name}: {len(watchlist)} tickers")
    print(f"[step129] wrote {OUT_SUPPLY_CHAIN.name}: {len(supply_chain)} rows")
    print(f"[step129] status={state.get('status')} high_neg={state.get('high_vulnerability_negative_rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
