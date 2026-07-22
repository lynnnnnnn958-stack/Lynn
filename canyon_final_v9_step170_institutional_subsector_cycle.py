#!/usr/bin/env python3
"""
Canyon v9 - Step 170: Institutional Subsector Cycle and Leadership Handoff
==========================================================================

Research-only. No broker connection. No live orders.

This step upgrades the broad sector cycle layer into a PM-style subsector view.
The important distinction is that a group can still be the price leader while
also being late-cycle, crowded, volatile, and poor to chase with fresh calls.

Outputs:
  institutional_subsector_cycle_board.csv
  sector_leadership_handoff_matrix.csv
  subsector_ticker_cycle_map.csv
  sector_cycle_playbook.csv
  subsector_rotation_validation.csv
  subsector_rotation_observation_history.csv
  institutional_subsector_cycle_state.json
  institutional_subsector_cycle_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    df_to_markdown,
    now_str,
    read_csv_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_SECTOR_CYCLE = ROOT / "sector_cycle_state.csv"
IN_ROTATION = ROOT / "sector_rotation_scores.csv"
IN_SP500_PRICE = ROOT / "sp500_price_cache.csv"
IN_BACKTEST_PRICE = ROOT / "backtest_price_cache.csv"
IN_TECHNICAL = ROOT / "technical_signal_matrix.csv"
IN_PICKS = ROOT / "daily_picks_filtered.csv"
IN_SECTOR_MAP = ROOT / "sector_map.csv"
IN_FUNDAMENTAL = ROOT / "fundamental_quality_valuation.csv"
IN_FUND_FEATURES = ROOT / "fundamental_features.csv"
IN_RISK_QUEUE = ROOT / "risk_desk_ticker_action_queue.csv"
IN_PROMOTION = ROOT / "research_promotion_gate.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_NEWS_TARGETS = ROOT / "news_impact_targets.csv"
IN_SUPPLY_CHAIN = ROOT / "news_supply_chain_readthrough.csv"
IN_THEME_ENRICH = ROOT / "theme_candidate_enrichment.csv"
IN_MONITOR = ROOT / "desk_monitor_ticker_state.csv"

OUT_BOARD = ROOT / "institutional_subsector_cycle_board.csv"
OUT_HANDOFF = ROOT / "sector_leadership_handoff_matrix.csv"
OUT_TICKER_MAP = ROOT / "subsector_ticker_cycle_map.csv"
OUT_PLAYBOOK = ROOT / "sector_cycle_playbook.csv"
OUT_VALIDATION = ROOT / "subsector_rotation_validation.csv"
OUT_HISTORY = ROOT / "subsector_rotation_observation_history.csv"
OUT_STATE = ROOT / "institutional_subsector_cycle_state.json"
OUT_REPORT = ROOT / "institutional_subsector_cycle_report.md"

SOURCE_STACK = (
    "sector_cycle_state.csv; sector_rotation_scores.csv; sp500_price_cache.csv; "
    "backtest_price_cache.csv; technical_signal_matrix.csv; daily_picks_filtered.csv; "
    "risk_desk_ticker_action_queue.csv; research_promotion_gate.csv; "
    "news_impact_targets.csv; news_supply_chain_readthrough.csv"
)


SEMI_TICKERS = {
    "SMH", "SOXX", "NVDA", "AMD", "AVGO", "MU", "QCOM", "TXN", "ADI", "ARM",
    "INTC", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "MRVL", "MCHP", "ON",
    "NXPI", "MPWR", "TER", "SWKS", "QRVO", "LSCC",
}

SOFTWARE_TICKERS = {
    "MSFT", "ORCL", "CRM", "NOW", "ADBE", "INTU", "ADSK", "SNOW", "DDOG",
    "CRWD", "PANW", "NET", "PLTR", "MDB", "ZS", "OKTA", "WDAY", "TEAM",
    "SHOP", "APP", "AKAM", "FTNT", "ANSS", "CDNS", "SNPS", "ROP", "FICO",
}

AI_INFRA_TICKERS = {
    "DELL", "HPE", "SMCI", "ANET", "CSCO", "CIEN", "STX", "WDC", "NTAP",
    "APH", "TEL", "GLW", "JNPR", "KEYS", "TDY", "VRT",
}

INTERNET_PLATFORM_TICKERS = {
    "GOOGL", "GOOG", "META", "AMZN", "NFLX", "UBER", "ABNB", "BKNG",
}

DATA_CENTER_POWER_TICKERS = {
    "CEG", "NRG", "VST", "ETN", "PWR", "EME", "GEV", "AEP", "NEE", "SO",
    "DUK", "EXC", "VRT", "ABB",
}

FINTECH_TICKERS = {
    "CPAY", "MA", "V", "PYPL", "FIS", "FI", "GPN", "ADP", "PAYX",
}

DEFENSIVE_TICKERS = {
    "XLV", "XLP", "XLU", "JNJ", "LLY", "UNH", "MRK", "KO", "PEP", "WMT",
}


SUBSECTOR_ORDER = [
    "Semiconductors",
    "Software / Cloud",
    "AI Infrastructure / Hardware",
    "Internet Platforms",
    "Data Center / Power",
    "FinTech / Payments",
    "Defensives",
    "Industrials",
    "Financials",
    "Consumer",
    "Energy / Materials",
    "Rates / Real Estate",
    "Broad Technology",
    "Other",
]


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def upper(value: Any) -> str:
    return text(value).upper()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def pct_from_any(value: Any) -> float:
    x = safe_float(value)
    if not np.isfinite(x):
        return np.nan
    return x * 100.0 if abs(x) <= 1.5 else x


def mean_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def max_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.max()) if not vals.empty else np.nan


def load_wide_prices(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        raw = read_csv_safe(path)
        if raw.empty:
            continue
        df = raw.copy()
        first = df.columns[0]
        if first.lower() in {"date", "unnamed: 0", "index"}:
            df[first] = pd.to_datetime(df[first], errors="coerce")
            df = df.dropna(subset=[first]).set_index(first)
        df.columns = [str(c).upper() for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    out = out.sort_index().ffill().dropna(how="all")
    return out


def return_pct(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return np.nan
    base = float(s.iloc[-periods - 1])
    last = float(s.iloc[-1])
    if base <= 0:
        return np.nan
    return (last / base - 1.0) * 100.0


def price_metrics_from_cache(prices: pd.DataFrame, ticker: str) -> dict[str, float | str]:
    t = upper(ticker)
    if prices.empty or t not in prices.columns:
        return {
            "ticker": t,
            "price_data_status": "NO_PRICE_CACHE",
            "ret_5d_pct": np.nan,
            "ret_20d_pct": np.nan,
            "ret_63d_pct": np.nan,
            "distance_to_20d_high_pct": np.nan,
        }
    s = pd.to_numeric(prices[t], errors="coerce").dropna()
    if len(s) < 22:
        status = "SHORT_PRICE_HISTORY"
    else:
        status = "OK"
    last = float(s.iloc[-1]) if not s.empty else np.nan
    high20 = float(s.tail(20).max()) if len(s) >= 20 else np.nan
    dist_high = (last / high20 - 1.0) * 100.0 if np.isfinite(last) and np.isfinite(high20) and high20 > 0 else np.nan
    return {
        "ticker": t,
        "price_data_status": status,
        "ret_5d_pct": return_pct(s, 5),
        "ret_20d_pct": return_pct(s, 20),
        "ret_63d_pct": return_pct(s, 63),
        "distance_to_20d_high_pct": dist_high,
    }


def classify_subsector(ticker: Any, sector: Any = "", industry: Any = "") -> str:
    t = upper(ticker)
    sec = upper(sector)
    ind = upper(industry)
    blob = f"{sec} {ind}"
    if t in {"XLK", "QQQ"}:
        return "Broad Technology"
    if t in SEMI_TICKERS or "SEMICONDUCT" in blob or "CHIP" in blob:
        return "Semiconductors"
    if t in SOFTWARE_TICKERS or any(x in blob for x in ["SOFTWARE", "SAAS", "CLOUD", "CYBER", "APPLICATION"]):
        return "Software / Cloud"
    if t in AI_INFRA_TICKERS or any(x in blob for x in ["HARDWARE", "NETWORK", "STORAGE", "COMMUNICATION EQUIPMENT"]):
        return "AI Infrastructure / Hardware"
    if t in INTERNET_PLATFORM_TICKERS or "INTERNET" in blob or "INTERACTIVE MEDIA" in blob:
        return "Internet Platforms"
    if t in DATA_CENTER_POWER_TICKERS or sec in {"UTILITIES"}:
        return "Data Center / Power"
    if t in FINTECH_TICKERS or "PAYMENT" in blob or "DATA PROCESSING" in blob:
        return "FinTech / Payments"
    if t in DEFENSIVE_TICKERS or sec in {"HEALTH CARE", "HEALTHCARE", "CONSUMER STAPLES"}:
        return "Defensives"
    if sec == "INDUSTRIALS":
        return "Industrials"
    if sec == "FINANCIALS":
        return "Financials"
    if sec in {"CONSUMER DISCRETIONARY"}:
        return "Consumer"
    if sec in {"ENERGY", "MATERIALS"}:
        return "Energy / Materials"
    if sec in {"REAL ESTATE"} or t in {"IYR", "XLRE", "TLT"}:
        return "Rates / Real Estate"
    if sec == "TECHNOLOGY":
        return "Broad Technology"
    return "Other"


def load_ticker_sources(prices: pd.DataFrame) -> pd.DataFrame:
    picks = read_csv_safe(IN_PICKS)
    sector_map = read_csv_safe(IN_SECTOR_MAP)
    fundamental = read_csv_safe(IN_FUNDAMENTAL)
    fund_features = read_csv_safe(IN_FUND_FEATURES)
    technical = read_csv_safe(IN_TECHNICAL)
    risk_queue = read_csv_safe(IN_RISK_QUEUE)
    promotion = read_csv_safe(IN_PROMOTION)
    options = read_csv_safe(IN_OPTIONS)
    event = read_csv_safe(IN_EVENT)
    monitor = read_csv_safe(IN_MONITOR)
    theme = read_csv_safe(IN_THEME_ENRICH)

    tickers: set[str] = set()
    for df, col in [
        (picks, "ticker"),
        (sector_map, "ticker"),
        (fundamental, "ticker"),
        (fund_features, "ticker"),
        (technical, "ticker"),
        (risk_queue, "ticker"),
        (promotion, "ticker"),
        (options, "ticker"),
        (event, "ticker"),
        (monitor, "ticker"),
        (theme, "ticker"),
    ]:
        if not df.empty and col in df.columns:
            tickers |= set(df[col].dropna().astype(str).str.upper())
    tickers |= SEMI_TICKERS | SOFTWARE_TICKERS | AI_INFRA_TICKERS | INTERNET_PLATFORM_TICKERS | DATA_CENTER_POWER_TICKERS | FINTECH_TICKERS
    tickers = {t for t in tickers if t and t != "NAN"}

    def first(df: pd.DataFrame, ticker: str) -> pd.Series:
        if df.empty or "ticker" not in df.columns:
            return pd.Series(dtype=object)
        rows = df[df["ticker"].astype(str).str.upper() == ticker].head(1)
        return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)

    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        prow = first(picks, ticker)
        smap = first(sector_map, ticker)
        frow = first(fundamental, ticker)
        ffrow = first(fund_features, ticker)
        trow = first(technical, ticker)
        rrow = first(risk_queue, ticker)
        grow = first(promotion, ticker)
        orow = first(options, ticker)
        erow = first(event, ticker)
        mrow = first(monitor, ticker)
        throw = first(theme, ticker)
        pmet = price_metrics_from_cache(prices, ticker)

        sector = (
            text(prow.get("sector")) or text(frow.get("sector")) or text(ffrow.get("sector"))
            or text(smap.get("sector")) or text(rrow.get("sector")) or text(erow.get("sector"))
        )
        industry = text(frow.get("industry"))
        subsector = classify_subsector(ticker, sector, industry)

        ret_5d = pct_from_any(trow.get("ret_5d")) if not trow.empty else np.nan
        ret_20d = pct_from_any(trow.get("ret_20d")) if not trow.empty else np.nan
        ret_63d = pct_from_any(trow.get("ret_63d")) if not trow.empty else np.nan
        if not np.isfinite(ret_5d):
            ret_5d = safe_float(pmet["ret_5d_pct"])
        if not np.isfinite(ret_20d):
            ret_20d = safe_float(pmet["ret_20d_pct"])
        if not np.isfinite(ret_63d):
            ret_63d = safe_float(pmet["ret_63d_pct"])

        distance_to_high = pct_from_any(trow.get("distance_to_20d_high")) if "distance_to_20d_high" in trow.index else np.nan
        if not np.isfinite(distance_to_high):
            distance_to_high = safe_float(pmet["distance_to_20d_high_pct"])

        rows.append({
            "ticker": ticker,
            "sector": sector or "Unknown",
            "industry": industry,
            "subsector": subsector,
            "is_current_book": bool(not prow.empty),
            "current_action": text(prow.get("action")),
            "current_weight_pct": safe_float(prow.get("weight_pct"), safe_float(rrow.get("current_weight_pct"), 0.0)),
            "alpha_score": safe_float(prow.get("alpha_score"), safe_float(grow.get("promotion_score"), np.nan)),
            "top_signal": text(prow.get("top_signal")),
            "ret_5d_pct": ret_5d,
            "ret_20d_pct": ret_20d,
            "ret_63d_pct": ret_63d,
            "technical_score": safe_float(trow.get("technical_score"), np.nan),
            "rsi14": safe_float(trow.get("rsi14"), np.nan),
            "atr14_pct": pct_from_any(trow.get("atr14_pct")),
            "distance_to_20d_high_pct": distance_to_high,
            "price_data_status": text(trow.get("data_status")) or text(pmet["price_data_status"]),
            "quality_score": safe_float(frow.get("quality_score"), safe_float(ffrow.get("quality_score"), np.nan)),
            "pe_ratio": safe_float(ffrow.get("pe_ratio"), safe_float(frow.get("forward_pe"), np.nan)),
            "beta": safe_float(frow.get("beta"), np.nan),
            "risk_action": text(rrow.get("final_risk_action")) or text(grow.get("master_risk_action")),
            "risk_status_bucket": text(rrow.get("status_bucket")),
            "risk_target_weight_pct": safe_float(rrow.get("recommended_risk_weight_pct"), np.nan),
            "promotion_status": text(grow.get("promotion_status")),
            "promotion_score": safe_float(grow.get("promotion_score"), np.nan),
            "max_paper_weight_pct": safe_float(grow.get("max_paper_weight_pct"), np.nan),
            "event_gate": text(erow.get("event_gate")),
            "earnings_gap_action": text(rrow.get("earnings_gap_action")),
            "option_side": text(orow.get("option_side")),
            "option_permission": text(orow.get("option_permission")),
            "monitor_severity": text(mrow.get("max_monitor_severity")),
            "price_break_state": text(mrow.get("price_break_state")),
            "volume_spike_state": text(mrow.get("volume_spike_state")),
            "volatility_regime_state": text(mrow.get("volatility_regime_state")),
            "theme": text(throw.get("theme")),
            "theme_candidate_status": text(throw.get("theme_candidate_status")),
            "theme_attention_score": safe_float(throw.get("attention_score"), np.nan),
            "source_file": SOURCE_STACK,
            "research_only": True,
        })
    return pd.DataFrame(rows)


def aggregate_news() -> pd.DataFrame:
    frames = []
    for path in [IN_NEWS_TARGETS, IN_SUPPLY_CHAIN]:
        df = read_csv_safe(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "positive_news", "negative_news", "mixed_news", "total_vulnerability", "top_headline", "top_theme"])
    news = pd.concat(frames, ignore_index=True)
    if "target_ticker" not in news.columns:
        return pd.DataFrame()
    news["ticker"] = news["target_ticker"].astype(str).str.upper()
    news["market_tone"] = news.get("market_tone", "").astype(str).str.upper()
    rows = []
    for ticker, grp in news.groupby("ticker"):
        top = grp.sort_values("impact_score", key=lambda x: pd.to_numeric(x, errors="coerce").abs(), ascending=False).head(1)
        top_row = top.iloc[0] if not top.empty else pd.Series(dtype=object)
        rows.append({
            "ticker": ticker,
            "positive_news": int((grp["market_tone"] == "POSITIVE").sum()),
            "negative_news": int((grp["market_tone"] == "NEGATIVE").sum()),
            "mixed_news": int((grp["market_tone"] == "MIXED").sum()),
            "context_news": int((grp["market_tone"].isin(["NEUTRAL", ""])).sum()),
            "total_vulnerability": max_numeric(grp.get("total_vulnerability", pd.Series(dtype=float))),
            "valuation_vulnerability": max_numeric(grp.get("valuation_vulnerability", pd.Series(dtype=float))),
            "volatility_vulnerability": max_numeric(grp.get("volatility_vulnerability", pd.Series(dtype=float))),
            "top_headline": text(top_row.get("headline"))[:220],
            "top_theme": text(top_row.get("theme")),
            "top_news_source": text(top_row.get("publisher")),
            "source_news_ticker": text(top_row.get("source_news_ticker")),
        })
    return pd.DataFrame(rows)


def merge_news(ticker_df: pd.DataFrame) -> pd.DataFrame:
    news = aggregate_news()
    if news.empty:
        for col in ["positive_news", "negative_news", "mixed_news", "total_vulnerability", "top_headline", "top_theme"]:
            ticker_df[col] = 0 if col.endswith("_news") else ""
        return ticker_df
    out = ticker_df.merge(news, on="ticker", how="left")
    for col in ["positive_news", "negative_news", "mixed_news", "context_news"]:
        out[col] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0).astype(int)
    for col in ["total_vulnerability", "valuation_vulnerability", "volatility_vulnerability"]:
        out[col] = pd.to_numeric(out.get(col, np.nan), errors="coerce")
    return out


def proxy_rotation_rows() -> pd.DataFrame:
    rotation = read_csv_safe(IN_ROTATION)
    cycle = read_csv_safe(IN_SECTOR_CYCLE)
    rows = []
    if not rotation.empty:
        for _, row in rotation.iterrows():
            ticker = upper(row.get("ticker"))
            theme = text(row.get("theme"))
            if ticker in {"SMH", "SOXX"} or "SEMICONDUCTOR" in theme.upper():
                subsector = "Semiconductors"
            elif ticker == "XLK":
                subsector = "Broad Technology"
            else:
                continue
            rows.append({
                "ticker": ticker,
                "subsector": subsector,
                "ret_5d_pct": pct_from_any(row.get("ret_5d")),
                "ret_20d_pct": pct_from_any(row.get("ret_20d")),
                "ret_63d_pct": pct_from_any(row.get("ret_63d")),
                "rotation_score": safe_float(row.get("rotation_score"), np.nan),
                "proxy_source": "sector_rotation_scores.csv",
            })
    if not cycle.empty:
        for _, row in cycle.iterrows():
            ticker = upper(row.get("etf"))
            sector = text(row.get("sector"))
            if ticker in {"SMH", "SOXX"} or sector == "Semiconductors":
                subsector = "Semiconductors"
            elif ticker == "XLK" or sector == "Technology":
                subsector = "Broad Technology"
            else:
                continue
            rows.append({
                "ticker": ticker,
                "subsector": subsector,
                "ret_5d_pct": np.nan,
                "ret_20d_pct": safe_float(row.get("ret_20d_pct"), np.nan),
                "ret_63d_pct": safe_float(row.get("ret_63d_pct"), np.nan),
                "rotation_score": safe_float(row.get("rotation_score"), np.nan),
                "proxy_source": "sector_cycle_state.csv",
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["ticker", "subsector"], keep="first")


def score_subsector(group: pd.DataFrame, spy_ret20: float, spy_ret63: float) -> dict[str, Any]:
    n = len(group)
    current = group[group["is_current_book"] == True].copy()
    ret5 = mean_numeric(group["ret_5d_pct"])
    ret20 = mean_numeric(group["ret_20d_pct"])
    ret63 = mean_numeric(group["ret_63d_pct"])
    rel20 = ret20 - spy_ret20 if np.isfinite(ret20) and np.isfinite(spy_ret20) else np.nan
    rel63 = ret63 - spy_ret63 if np.isfinite(ret63) and np.isfinite(spy_ret63) else np.nan
    alpha = mean_numeric(group["alpha_score"])
    tech = mean_numeric(group["technical_score"])
    atr = mean_numeric(group["atr14_pct"])
    rsi = mean_numeric(group["rsi14"])
    dist_high = mean_numeric(group["distance_to_20d_high_pct"])
    quality = mean_numeric(group["quality_score"])
    pe = mean_numeric(group["pe_ratio"])
    weight = mean_numeric(current["current_weight_pct"]) * len(current) if not current.empty else 0.0

    risk_text = group["risk_action"].astype(str).str.upper()
    event_text = group["event_gate"].astype(str).str.upper()
    option_text = group["option_permission"].astype(str).str.upper()
    monitor_text = group["monitor_severity"].astype(str).str.upper()

    size_down_ratio = float(risk_text.str.contains("SIZE_DOWN|REDUCE|BLOCK|NO_NEW", na=False).mean()) if n else 0.0
    event_review_ratio = float(event_text.str.contains("REVIEW|MISSING", na=False).mean()) if n else 0.0
    call_blocked_ratio = float(option_text.str.contains("BLOCK|NO_NEW", na=False).mean()) if n else 0.0
    monitor_warning_ratio = float(monitor_text.str.contains("WARNING|CRITICAL", na=False).mean()) if n else 0.0

    positive_news = int(pd.to_numeric(group.get("positive_news", 0), errors="coerce").fillna(0).sum())
    negative_news = int(pd.to_numeric(group.get("negative_news", 0), errors="coerce").fillna(0).sum())
    mixed_news = int(pd.to_numeric(group.get("mixed_news", 0), errors="coerce").fillna(0).sum())
    vulnerability = max_numeric(group.get("total_vulnerability", pd.Series(dtype=float)))
    valuation_vuln = max_numeric(group.get("valuation_vulnerability", pd.Series(dtype=float)))
    vol_vuln = max_numeric(group.get("volatility_vulnerability", pd.Series(dtype=float)))
    catalyst_balance = positive_news - negative_news
    catalyst_score = clip01((catalyst_balance + 5.0) / 30.0)

    leadership_strength = 100.0 * (
        0.22 * clip01((rel20 if np.isfinite(rel20) else 0.0) / 18.0)
        + 0.22 * clip01((rel63 if np.isfinite(rel63) else 0.0) / 40.0)
        + 0.16 * clip01(((alpha if np.isfinite(alpha) else 50.0) - 50.0) / 35.0)
        + 0.14 * clip01((tech if np.isfinite(tech) else 50.0) / 100.0)
        + 0.14 * catalyst_score
        + 0.12 * clip01(((ret5 if np.isfinite(ret5) else 0.0) + 2.0) / 10.0)
    )

    overextension = max(0.0, ret63 if np.isfinite(ret63) else 0.0)
    short_heat = max(0.0, ret20 if np.isfinite(ret20) else 0.0)
    gate_heat = (size_down_ratio + event_review_ratio + call_blocked_ratio + monitor_warning_ratio) / 4.0
    catalyst_heat = clip01(positive_news / 75.0)
    exhaustion_risk = 100.0 * (
        0.18 * clip01(overextension / 60.0)
        + 0.16 * clip01(short_heat / 22.0)
        + 0.12 * clip01((atr if np.isfinite(atr) else 0.0) / 5.0)
        + 0.10 * clip01((vulnerability if np.isfinite(vulnerability) else 50.0) / 100.0)
        + 0.08 * clip01((valuation_vuln if np.isfinite(valuation_vuln) else 50.0) / 100.0)
        + 0.08 * clip01((vol_vuln if np.isfinite(vol_vuln) else 50.0) / 100.0)
        + 0.10 * catalyst_heat
        + 0.10 * clip01(leadership_strength / 100.0)
        + 0.08 * gate_heat
    )

    trend_turn = 0.0
    if np.isfinite(ret5) and np.isfinite(ret20):
        trend_turn = ret5 - ret20 / 4.0
    catch_up_score = 100.0 * (
        0.18 * clip01(((rel20 if np.isfinite(rel20) else 0.0) + 4.0) / 18.0)
        + 0.18 * clip01((trend_turn + 2.0) / 8.0)
        + 0.16 * clip01(((alpha if np.isfinite(alpha) else 50.0) - 45.0) / 35.0)
        + 0.16 * catalyst_score
        + 0.16 * clip01((100.0 - exhaustion_risk) / 100.0)
        + 0.16 * clip01((quality if np.isfinite(quality) else 50.0) / 100.0)
    )

    if group["ret_20d_pct"].notna().sum() < 2:
        phase = "Needs price basket"
        action_bias = "Do not upgrade without a real price basket."
        institutional_view = "Subsector data coverage is too thin for a cycle call."
    elif leadership_strength >= 62 and exhaustion_risk >= 58:
        phase = "Leader but late-cycle chase risk"
        action_bias = "Respect strength, but do not chase fresh calls."
        institutional_view = "Price leadership is real, but heat, risk gates, and volatility say the trade may be late."
    elif leadership_strength >= 64:
        phase = "Leadership expansion"
        action_bias = "Research longs only after risk and event gates clear."
        institutional_view = "The group has broad price support and still has room before late-cycle risk dominates."
    elif catch_up_score >= 52 and leadership_strength >= 38:
        phase = "Catch-up handoff candidate"
        action_bias = "Watch for medium-term handoff confirmation."
        institutional_view = "The group is not the loudest leader, but it has improving evidence and less chase risk."
    elif np.isfinite(rel20) and rel20 < -5 and np.isfinite(rel63) and rel63 < -8:
        phase = "Downcycle / laggard"
        action_bias = "Avoid new longs unless thesis is defensive or mean-reversion only."
        institutional_view = "Relative returns are weak across horizons."
    elif np.isfinite(rel20) and rel20 > 0:
        phase = "Early improvement"
        action_bias = "Research watchlist, require source and price confirmation."
        institutional_view = "Near-term relative strength is improving but not yet a clean leadership call."
    else:
        phase = "Neutral / base"
        action_bias = "Keep on watch, no automatic upgrade."
        institutional_view = "Evidence is mixed or not strong enough."

    top_names = ", ".join(
        group.sort_values("alpha_score", ascending=False)["ticker"].dropna().astype(str).head(8).tolist()
    )
    top_current = ", ".join(
        current.sort_values("alpha_score", ascending=False)["ticker"].dropna().astype(str).head(8).tolist()
    )
    top_headlines = group.get("top_headline", pd.Series(dtype=str)).dropna().astype(str)
    top_headline = top_headlines[top_headlines.ne("")].head(1).iloc[0] if not top_headlines[top_headlines.ne("")].empty else ""
    top_themes = group.get("top_theme", pd.Series(dtype=str)).dropna().astype(str)
    top_theme = top_themes[top_themes.ne("")].mode().iloc[0] if not top_themes[top_themes.ne("")].empty else ""

    return {
        "subsector": text(group["subsector"].iloc[0]),
        "cycle_phase": phase,
        "institutional_view": institutional_view,
        "action_bias": action_bias,
        "leadership_strength": round(leadership_strength, 2),
        "exhaustion_risk": round(exhaustion_risk, 2),
        "catch_up_score": round(catch_up_score, 2),
        "ret_5d_pct": round(ret5, 2) if np.isfinite(ret5) else np.nan,
        "ret_20d_pct": round(ret20, 2) if np.isfinite(ret20) else np.nan,
        "ret_63d_pct": round(ret63, 2) if np.isfinite(ret63) else np.nan,
        "relative_20d_vs_spy_pct": round(rel20, 2) if np.isfinite(rel20) else np.nan,
        "relative_63d_vs_spy_pct": round(rel63, 2) if np.isfinite(rel63) else np.nan,
        "avg_alpha_score": round(alpha, 2) if np.isfinite(alpha) else np.nan,
        "avg_technical_score": round(tech, 2) if np.isfinite(tech) else np.nan,
        "avg_rsi14": round(rsi, 2) if np.isfinite(rsi) else np.nan,
        "avg_atr14_pct": round(atr, 2) if np.isfinite(atr) else np.nan,
        "avg_distance_to_20d_high_pct": round(dist_high, 2) if np.isfinite(dist_high) else np.nan,
        "avg_quality_score": round(quality, 2) if np.isfinite(quality) else np.nan,
        "avg_pe_proxy": round(pe, 2) if np.isfinite(pe) else np.nan,
        "positive_news": positive_news,
        "negative_news": negative_news,
        "mixed_news": mixed_news,
        "catalyst_balance": catalyst_balance,
        "max_total_vulnerability": round(vulnerability, 2) if np.isfinite(vulnerability) else np.nan,
        "max_valuation_vulnerability": round(valuation_vuln, 2) if np.isfinite(valuation_vuln) else np.nan,
        "max_volatility_vulnerability": round(vol_vuln, 2) if np.isfinite(vol_vuln) else np.nan,
        "risk_size_down_ratio": round(size_down_ratio, 3),
        "event_review_ratio": round(event_review_ratio, 3),
        "call_blocked_ratio": round(call_blocked_ratio, 3),
        "monitor_warning_ratio": round(monitor_warning_ratio, 3),
        "current_book_weight_pct": round(weight, 2) if np.isfinite(weight) else 0.0,
        "current_book_tickers": top_current,
        "top_names": top_names,
        "top_theme": top_theme,
        "top_headline": top_headline,
        "members_count": int(n),
        "price_rows_used": int(group["ret_20d_pct"].notna().sum()),
        "source_file": SOURCE_STACK,
        "research_only": True,
    }


def subsector_adjustments(phase: str, subsector: str) -> tuple[float, float, float, str, str]:
    p = upper(phase)
    s = upper(subsector)
    if "LATE-CYCLE" in p or "CHASE RISK" in p:
        return (
            -6.0,
            -8.0,
            -4.0,
            "Late-cycle leader: strength is acknowledged, but new chase risk is penalized.",
            "No fresh bullish calls into strength; wait for pullback, evidence reset, or use hedge-only research.",
        )
    if "CATCH-UP" in p or "HANDOFF" in p:
        return (
            3.0,
            8.0,
            6.0,
            "Handoff watch: improve medium-term research priority but still require gates.",
            "Defined-risk call spread research only after risk, event, and spread gates clear.",
        )
    if "LEADERSHIP EXPANSION" in p:
        return (
            4.0,
            6.0,
            4.0,
            "Leadership expansion: supportive but not allowed to override risk.",
            "Stock or defined-risk option research only after gates clear.",
        )
    if "DOWNCYCLE" in p or "LAGGARD" in p:
        return (
            -8.0,
            -6.0,
            -4.0,
            "Weak subsector: avoid bullish research unless a defensive thesis is explicit.",
            "No bullish option route; put or hedge research only when risk logic supports it.",
        )
    if "NEEDS PRICE" in p:
        return (
            0.0,
            0.0,
            0.0,
            "No upgrade: price basket is not strong enough.",
            "No option upgrade from subsector cycle.",
        )
    if s == "SOFTWARE / CLOUD":
        return (
            1.0,
            3.0,
            2.0,
            "Software watch: keep it visible for possible handoff, but require proof.",
            "No automatic call upgrade; wait for confirmation.",
        )
    return (
        0.0,
        0.0,
        0.0,
        "Neutral subsector overlay.",
        "No option change from subsector cycle.",
    )


def build_handoff(board: pd.DataFrame) -> pd.DataFrame:
    def row_for(name: str) -> pd.Series:
        rows = board[board["subsector"].astype(str) == name].head(1)
        return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)

    pairs = [
        ("Semiconductors", "Software / Cloud", "AI spending may rotate from chips into apps, cloud, and productivity software."),
        ("Semiconductors", "AI Infrastructure / Hardware", "Chip strength can spill into servers, networking, storage, and optical equipment."),
        ("Semiconductors", "Data Center / Power", "AI compute demand can pull power, electrical equipment, and data-center capacity."),
        ("Broad Technology", "Software / Cloud", "Broad tech leadership needs internal confirmation from software, not only chips."),
        ("Software / Cloud", "Internet Platforms", "Software and platform data can reinforce each other when AI monetization broadens."),
    ]
    rows = []
    for source, target, thesis in pairs:
        srow = row_for(source)
        trow = row_for(target)
        s_phase = text(srow.get("cycle_phase", "NO_DATA"))
        t_phase = text(trow.get("cycle_phase", "NO_DATA"))
        s_exh = safe_float(srow.get("exhaustion_risk"), 0.0)
        t_catch = safe_float(trow.get("catch_up_score"), 0.0)
        t_lead = safe_float(trow.get("leadership_strength"), 0.0)
        s_lead = safe_float(srow.get("leadership_strength"), 0.0)

        if "late-cycle" in s_phase.lower() and ("catch-up" in t_phase.lower() or t_catch >= 52):
            signal = "Handoff watch"
            desk_note = "The source group is hot/late and the target group has enough improvement to monitor for rotation."
        elif "late-cycle" in s_phase.lower():
            signal = "Late leader, target not confirmed"
            desk_note = "Do not chase the hot source group, but do not call a handoff until target price proof improves."
        elif t_lead > s_lead + 8:
            signal = "Target already stronger"
            desk_note = "Target leadership is stronger than the old source group."
        elif s_lead >= 62 and t_catch < 50:
            signal = "No handoff yet"
            desk_note = "Source group remains the cleaner leader; target needs proof."
        else:
            signal = "Monitor"
            desk_note = "The pair matters, but the current evidence is not decisive."

        rows.append({
            "source_subsector": source,
            "target_subsector": target,
            "handoff_signal": signal,
            "source_phase": s_phase,
            "target_phase": t_phase,
            "source_leadership_strength": round(s_lead, 2),
            "source_exhaustion_risk": round(s_exh, 2),
            "target_leadership_strength": round(t_lead, 2),
            "target_catch_up_score": round(t_catch, 2),
            "thesis_to_test": thesis,
            "desk_note": desk_note,
            "confirmation_needed": "Target relative 20d strength, clean event/risk gates, and source heat cooling without broad-market damage.",
            "source_file": SOURCE_STACK,
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_ticker_map(ticker_df: pd.DataFrame, board: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    phase_by_sub = board.set_index("subsector").to_dict(orient="index") if not board.empty else {}
    handoff_text = ""
    sw = handoff[
        (handoff.get("source_subsector", pd.Series(dtype=str)).astype(str) == "Semiconductors")
        & (handoff.get("target_subsector", pd.Series(dtype=str)).astype(str) == "Software / Cloud")
    ]
    if not sw.empty:
        handoff_text = text(sw.iloc[0].get("handoff_signal"))

    rows = []
    for _, row in ticker_df.iterrows():
        ticker = upper(row.get("ticker"))
        current = bool(row.get("is_current_book"))
        subsector = text(row.get("subsector"))
        if not current and subsector not in {"Semiconductors", "Software / Cloud", "AI Infrastructure / Hardware", "Data Center / Power"}:
            continue
        b = phase_by_sub.get(subsector, {})
        phase = text(b.get("cycle_phase", "NO_SUBSECTOR_CYCLE"))
        short_adj, med_adj, long_adj, label, opt_overlay = subsector_adjustments(phase, subsector)
        risk_action = text(row.get("risk_action"))
        event_gate = text(row.get("event_gate"))
        if any(x in upper(risk_action) for x in ["REDUCE", "BLOCK", "SIZE_DOWN", "NO_NEW"]):
            opt_overlay = "Risk gate still controls the name; no new bullish option route from subsector logic."
        if "REVIEW" in upper(event_gate) or "MISSING" in upper(event_gate):
            opt_overlay = "Event/source review is required before any option upgrade."

        if subsector == "Semiconductors" and "late-cycle" in phase.lower():
            action = "De-risk or wait for pullback"
            why = "Semis still lead on price, but late-cycle heat means fresh chase risk is high."
        elif subsector == "Software / Cloud" and handoff_text in {"Handoff watch", "Late leader, target not confirmed"}:
            action = "Build software handoff watchlist"
            why = "Software is the clean place to test whether AI leadership is broadening beyond chips."
        elif "catch-up" in phase.lower():
            action = "Medium-term research watch"
            why = "The subsector is improving but still needs confirmation."
        elif "leadership expansion" in phase.lower():
            action = "Research leader, respect risk gate"
            why = "The subsector has real leadership but position size still comes from risk."
        elif "downcycle" in phase.lower():
            action = "Avoid bullish chase"
            why = "Subsector returns are weak versus the market."
        else:
            action = "Watch only"
            why = "Subsector evidence is not decisive."

        rows.append({
            "ticker": ticker,
            "sector": text(row.get("sector")),
            "subsector": subsector,
            "subsector_cycle_phase": phase,
            "leadership_handoff_signal": handoff_text if subsector in {"Semiconductors", "Software / Cloud"} else "",
            "subsector_action_bias": action,
            "subsector_adjustment_label": label,
            "subsector_short_adjustment": short_adj,
            "subsector_medium_adjustment": med_adj,
            "subsector_long_adjustment": long_adj,
            "option_permission_overlay": opt_overlay,
            "current_weight_pct": safe_float(row.get("current_weight_pct"), 0.0),
            "alpha_score": safe_float(row.get("alpha_score"), np.nan),
            "ret_20d_pct": safe_float(row.get("ret_20d_pct"), np.nan),
            "ret_63d_pct": safe_float(row.get("ret_63d_pct"), np.nan),
            "risk_action": risk_action,
            "event_gate": event_gate,
            "option_side": text(row.get("option_side")),
            "top_signal": text(row.get("top_signal")),
            "top_headline": text(row.get("top_headline"))[:220],
            "subsector_why": why,
            "source_file": SOURCE_STACK,
            "research_only": True,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_current"] = out["current_weight_pct"].fillna(0).gt(0).astype(int)
        out = out.sort_values(["_current", "subsector", "alpha_score"], ascending=[False, True, False]).drop(columns=["_current"])
    return out


def build_playbook(board: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in board.iterrows():
        subsector = text(row.get("subsector"))
        phase = text(row.get("cycle_phase"))
        _, _, _, label, opt = subsector_adjustments(phase, subsector)
        if subsector == "Semiconductors":
            what_to_do = "Treat the group as a hot leader first, then ask whether the trade is already late."
            what_not = "Do not buy fresh weekly calls simply because SMH/SOXX are strong."
            confirmation = "Need cooling volatility, cleaner risk gates, and continued breadth before adding."
        elif subsector == "Software / Cloud":
            what_to_do = "Build the handoff watchlist and compare software returns to semis and QQQ."
            what_not = "Do not call software leadership until price and event proof improve."
            confirmation = "Need relative 20d strength, improving 5d trend, and clean event/source rows."
        elif subsector == "AI Infrastructure / Hardware":
            what_to_do = "Separate durable AI infrastructure demand from one-day hardware squeezes."
            what_not = "Do not let a hardware gap-up become a full sector upgrade without liquidity and event checks."
            confirmation = "Need stable volume, lower gap risk, and read-through from data center demand."
        elif subsector == "Data Center / Power":
            what_to_do = "Track downstream AI power and grid beneficiaries as a separate sleeve."
            what_not = "Do not treat utilities/power as the same risk as software or chips."
            confirmation = "Need source-linked catalyst, price confirmation, and drawdown-safe sizing."
        else:
            what_to_do = "Use as context only unless ticker-level evidence is strong."
            what_not = "Do not upgrade a ticker from a broad label alone."
            confirmation = "Need ticker-level price, event, and risk evidence."

        rows.append({
            "subsector": subsector,
            "cycle_phase": phase,
            "what_to_do_now": what_to_do,
            "what_not_to_do": what_not,
            "confirmation_needed": confirmation,
            "option_permission": opt,
            "risk_overlay": label,
            "current_view": text(row.get("institutional_view")),
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_rotation_validation(board: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    def get_row(df: pd.DataFrame, col: str, value: str) -> pd.Series:
        if df.empty or col not in df.columns:
            return pd.Series(dtype=object)
        rows = df[df[col].astype(str) == value].head(1)
        return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)

    semis = get_row(board, "subsector", "Semiconductors")
    software = get_row(board, "subsector", "Software / Cloud")
    ai_hw = get_row(board, "subsector", "AI Infrastructure / Hardware")
    power = get_row(board, "subsector", "Data Center / Power")
    internet = get_row(board, "subsector", "Internet Platforms")
    semi_to_sw = handoff[
        (handoff.get("source_subsector", pd.Series(dtype=str)).astype(str) == "Semiconductors")
        & (handoff.get("target_subsector", pd.Series(dtype=str)).astype(str) == "Software / Cloud")
    ].head(1)
    semi_to_sw_row = semi_to_sw.iloc[0] if not semi_to_sw.empty else pd.Series(dtype=object)

    semis_phase = text(semis.get("cycle_phase"))
    sw_phase = text(software.get("cycle_phase"))
    hw_phase = text(ai_hw.get("cycle_phase"))
    power_phase = text(power.get("cycle_phase"))
    handoff_signal = text(semi_to_sw_row.get("handoff_signal"))

    semis_heat = safe_float(semis.get("exhaustion_risk"), 0.0)
    semis_lead = safe_float(semis.get("leadership_strength"), 0.0)
    sw_catch = safe_float(software.get("catch_up_score"), 0.0)
    sw_rel20 = safe_float(software.get("relative_20d_vs_spy_pct"), np.nan)
    hw_heat = safe_float(ai_hw.get("exhaustion_risk"), 0.0)
    power_rel20 = safe_float(power.get("relative_20d_vs_spy_pct"), np.nan)
    internet_rel20 = safe_float(internet.get("relative_20d_vs_spy_pct"), np.nan)

    rows = []

    def add(thesis_id: str, thesis: str, status: str, score: float, evidence: str, action: str, confirmation: str, kill_switch: str) -> None:
        rows.append({
            "thesis_id": thesis_id,
            "thesis": thesis,
            "validation_status": status,
            "validation_score": round(float(np.clip(score, 0, 100)), 2),
            "evidence": evidence,
            "pm_action": action,
            "confirmation_needed": confirmation,
            "kill_switch_or_refutation": kill_switch,
            "source_file": "institutional_subsector_cycle_board.csv; sector_leadership_handoff_matrix.csv",
            "research_only": True,
        })

    add(
        "SEMIS_LATE_CYCLE_RISK",
        "Semiconductors can remain price leaders while being too hot to chase.",
        "CONFIRMED" if "late-cycle" in semis_phase.lower() and semis_heat >= 65 else "WATCH" if semis_lead >= 60 else "NOT_CONFIRMED",
        0.55 * semis_heat + 0.45 * semis_lead,
        f"phase={semis_phase}; leadership={semis_lead:.1f}; heat={semis_heat:.1f}; ret63={safe_float(semis.get('ret_63d_pct'), np.nan):.1f}%; risk_size_down={safe_float(semis.get('risk_size_down_ratio'), 0.0):.2f}",
        "Respect the tape, but block fresh chase calls; prefer pullback/retest or de-risk research.",
        "Semis heat below 55, volatility cools, and breadth stays positive without gap risk.",
        "Semis relative 20d turns negative while software/power fail to improve: do not call it healthy rotation.",
    )
    add(
        "SOFTWARE_HANDOFF_WATCH",
        "Software/cloud may take the next leadership baton if AI spend broadens from chips into apps/cloud.",
        "CONFIRMED" if "leadership" in sw_phase.lower() and handoff_signal in {"Handoff watch", "Target already stronger"} else "WATCH" if "handoff" in sw_phase.lower() or handoff_signal == "Handoff watch" else "NOT_CONFIRMED",
        0.55 * sw_catch + 0.45 * max(0.0, sw_rel20 if np.isfinite(sw_rel20) else 0.0) * 4,
        f"phase={sw_phase}; catch_up={sw_catch:.1f}; rel20={sw_rel20:.1f}%; semis_to_software={handoff_signal}",
        "Build software watchlist, but require ticker-level event/risk proof before sizing.",
        "Software relative 20d stays positive, 5d trend improves, event gates clear, and semis heat cools.",
        "Software relative 20d drops below SPY or news/event rows fail reliability checks.",
    )
    add(
        "AI_HARDWARE_OVERHEAT",
        "AI infrastructure/hardware can benefit from chip demand but may also be a late-cycle squeeze.",
        "CONFIRMED" if "late-cycle" in hw_phase.lower() and hw_heat >= 55 else "WATCH" if safe_float(ai_hw.get("leadership_strength"), 0.0) >= 55 else "NOT_CONFIRMED",
        hw_heat,
        f"phase={hw_phase}; heat={hw_heat:.1f}; ret63={safe_float(ai_hw.get('ret_63d_pct'), np.nan):.1f}%; current names={text(ai_hw.get('current_book_tickers'))}",
        "Separate durable infrastructure demand from hardware squeeze risk; avoid gap-chasing.",
        "Lower volatility, cleaner event gates, and sustained relative strength after pullbacks.",
        "High volume gap reversals or event/source review should block bullish options.",
    )
    add(
        "DATA_CENTER_POWER_LAG",
        "Power/data-center beneficiaries are a separate sleeve and should not be assumed to follow semis automatically.",
        "CONFIRMED" if "downcycle" in power_phase.lower() else "WATCH" if np.isfinite(power_rel20) and power_rel20 < 0 else "NOT_CONFIRMED",
        max(0.0, -power_rel20 * 8) if np.isfinite(power_rel20) else 30.0,
        f"phase={power_phase}; rel20={power_rel20:.1f}%; top names={text(power.get('top_names'))}",
        "Keep downstream AI-power thesis on watch, but do not fund it from semis thesis without price proof.",
        "Power relative 20d turns positive with source-linked catalysts and risk gates clear.",
        "If power remains a laggard while semis cool, this is not a broad AI rotation yet.",
    )
    broadening_score = np.nanmean([
        sw_rel20 if np.isfinite(sw_rel20) else np.nan,
        internet_rel20 if np.isfinite(internet_rel20) else np.nan,
        power_rel20 if np.isfinite(power_rel20) else np.nan,
    ])
    add(
        "AI_BREADTH_BROADENING",
        "AI leadership is healthier if software, internet, hardware, and power confirm instead of one chip-only move.",
        "CONFIRMED" if np.isfinite(broadening_score) and broadening_score > 5 else "WATCH" if np.isfinite(broadening_score) and broadening_score > -2 else "NOT_CONFIRMED",
        50.0 + (broadening_score * 5.0 if np.isfinite(broadening_score) else 0.0),
        f"software_rel20={sw_rel20:.1f}%; internet_rel20={internet_rel20:.1f}%; power_rel20={power_rel20:.1f}%; average={broadening_score:.1f}%",
        "Use breadth to decide whether AI is a durable rotation or a narrow late-cycle leadership spike.",
        "At least two downstream buckets show positive relative 20d and clean risk/event gates.",
        "If breadth stays negative, cut confidence in broad AI theme and keep semis as crowded leader only.",
    )
    out = pd.DataFrame(rows)
    if not out.empty:
        status_rank = {"CONFIRMED": 0, "WATCH": 1, "NOT_CONFIRMED": 2}
        out["_rank"] = out["validation_status"].map(status_rank).fillna(9)
        out = out.sort_values(["_rank", "validation_score"], ascending=[True, False]).drop(columns=["_rank"]).reset_index(drop=True)
    return out


def update_rotation_history(board: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    run_time = now_str()
    rows = []
    for _, row in board.iterrows():
        rows.append({
            "run_time": run_time,
            "run_date": run_time[:10],
            "record_type": "subsector",
            "name": text(row.get("subsector")),
            "phase_or_status": text(row.get("cycle_phase")),
            "leadership_strength": safe_float(row.get("leadership_strength"), np.nan),
            "exhaustion_risk": safe_float(row.get("exhaustion_risk"), np.nan),
            "catch_up_score": safe_float(row.get("catch_up_score"), np.nan),
            "relative_20d_vs_spy_pct": safe_float(row.get("relative_20d_vs_spy_pct"), np.nan),
            "evidence": text(row.get("institutional_view")),
            "research_only": True,
        })
    for _, row in validation.iterrows():
        rows.append({
            "run_time": run_time,
            "run_date": run_time[:10],
            "record_type": "thesis",
            "name": text(row.get("thesis_id")),
            "phase_or_status": text(row.get("validation_status")),
            "leadership_strength": np.nan,
            "exhaustion_risk": np.nan,
            "catch_up_score": safe_float(row.get("validation_score"), np.nan),
            "relative_20d_vs_spy_pct": np.nan,
            "evidence": text(row.get("evidence")),
            "research_only": True,
        })
    today = pd.DataFrame(rows)
    old = read_csv_safe(OUT_HISTORY)
    if old.empty:
        hist = today
    else:
        hist = pd.concat([old, today], ignore_index=True)
        hist = hist.drop_duplicates(subset=["run_date", "record_type", "name"], keep="last")
    hist = hist.sort_values(["run_date", "record_type", "name"]).reset_index(drop=True)
    return hist


def build_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prices = load_wide_prices([IN_SP500_PRICE, IN_BACKTEST_PRICE])
    ticker_df = load_ticker_sources(prices)
    ticker_df = merge_news(ticker_df)

    proxy = proxy_rotation_rows()
    if not proxy.empty:
        proxy_rows = []
        for _, row in proxy.iterrows():
            proxy_rows.append({
                "ticker": upper(row.get("ticker")),
                "sector": "Technology" if text(row.get("subsector")) in {"Semiconductors", "Broad Technology"} else "Unknown",
                "industry": "",
                "subsector": text(row.get("subsector")),
                "is_current_book": False,
                "current_action": "",
                "current_weight_pct": 0.0,
                "alpha_score": np.nan,
                "top_signal": "",
                "ret_5d_pct": safe_float(row.get("ret_5d_pct"), np.nan),
                "ret_20d_pct": safe_float(row.get("ret_20d_pct"), np.nan),
                "ret_63d_pct": safe_float(row.get("ret_63d_pct"), np.nan),
                "technical_score": np.nan,
                "rsi14": np.nan,
                "atr14_pct": np.nan,
                "distance_to_20d_high_pct": np.nan,
                "price_data_status": "OK",
                "quality_score": np.nan,
                "pe_ratio": np.nan,
                "beta": np.nan,
                "risk_action": "",
                "risk_status_bucket": "",
                "risk_target_weight_pct": np.nan,
                "promotion_status": "",
                "promotion_score": np.nan,
                "max_paper_weight_pct": np.nan,
                "event_gate": "",
                "earnings_gap_action": "",
                "option_side": "",
                "option_permission": "",
                "monitor_severity": "",
                "price_break_state": "",
                "volume_spike_state": "",
                "volatility_regime_state": "",
                "theme": "",
                "theme_candidate_status": "",
                "theme_attention_score": np.nan,
                "positive_news": 0,
                "negative_news": 0,
                "mixed_news": 0,
                "context_news": 0,
                "total_vulnerability": np.nan,
                "valuation_vulnerability": np.nan,
                "volatility_vulnerability": np.nan,
                "top_headline": "",
                "top_theme": "",
                "top_news_source": "",
                "source_news_ticker": "",
                "source_file": f"{SOURCE_STACK}; {text(row.get('proxy_source'))}",
                "research_only": True,
            })
        ticker_df = pd.concat([ticker_df, pd.DataFrame(proxy_rows)], ignore_index=True)

    spy = price_metrics_from_cache(prices, "SPY")
    spy_ret20 = safe_float(spy.get("ret_20d_pct"), np.nan)
    spy_ret63 = safe_float(spy.get("ret_63d_pct"), np.nan)
    if not np.isfinite(spy_ret20):
        sector_cycle = read_csv_safe(IN_SECTOR_CYCLE)
        spy_row = sector_cycle[sector_cycle.get("etf", pd.Series(dtype=str)).astype(str).str.upper() == "SPY"] if not sector_cycle.empty else pd.DataFrame()
        spy_ret20 = safe_float(spy_row.iloc[0].get("ret_20d_pct"), 0.0) if not spy_row.empty else 0.0
        spy_ret63 = safe_float(spy_row.iloc[0].get("ret_63d_pct"), 0.0) if not spy_row.empty else 0.0

    board_rows = []
    for subsector in SUBSECTOR_ORDER:
        grp = ticker_df[ticker_df["subsector"].astype(str) == subsector].copy()
        if grp.empty:
            continue
        board_rows.append(score_subsector(grp, spy_ret20, spy_ret63))
    board = pd.DataFrame(board_rows)
    if not board.empty:
        board["_order"] = board["subsector"].map({name: i for i, name in enumerate(SUBSECTOR_ORDER)}).fillna(999)
        board = board.sort_values(["_order", "leadership_strength"], ascending=[True, False]).drop(columns=["_order"])

    handoff = build_handoff(board)
    ticker_map = build_ticker_map(ticker_df, board, handoff)
    playbook = build_playbook(board, handoff)
    validation = build_rotation_validation(board, handoff)
    history = update_rotation_history(board, validation)

    semis = board[board["subsector"].astype(str) == "Semiconductors"].head(1)
    software = board[board["subsector"].astype(str) == "Software / Cloud"].head(1)
    handoff_row = handoff[
        (handoff.get("source_subsector", pd.Series(dtype=str)).astype(str) == "Semiconductors")
        & (handoff.get("target_subsector", pd.Series(dtype=str)).astype(str) == "Software / Cloud")
    ].head(1)
    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "subsectors": int(len(board)),
        "tickers_mapped": int(len(ticker_map)),
        "semiconductor_phase": text(semis.iloc[0].get("cycle_phase")) if not semis.empty else "NO_DATA",
        "semiconductor_exhaustion_risk": safe_float(semis.iloc[0].get("exhaustion_risk"), np.nan) if not semis.empty else np.nan,
        "software_phase": text(software.iloc[0].get("cycle_phase")) if not software.empty else "NO_DATA",
        "software_catch_up_score": safe_float(software.iloc[0].get("catch_up_score"), np.nan) if not software.empty else np.nan,
        "semis_to_software_handoff": text(handoff_row.iloc[0].get("handoff_signal")) if not handoff_row.empty else "NO_DATA",
        "late_cycle_count": int(board["cycle_phase"].astype(str).str.contains("late-cycle|chase risk", case=False, na=False).sum()) if not board.empty else 0,
        "catch_up_count": int(board["cycle_phase"].astype(str).str.contains("catch-up|handoff", case=False, na=False).sum()) if not board.empty else 0,
        "rotation_theses_confirmed": int(validation["validation_status"].astype(str).eq("CONFIRMED").sum()) if not validation.empty else 0,
        "rotation_theses_watch": int(validation["validation_status"].astype(str).eq("WATCH").sum()) if not validation.empty else 0,
        "rotation_history_rows": int(len(history)),
        "logic": (
            "A subsector can be a price leader and still receive a late-cycle chase-risk penalty. "
            "The model does not hard-code bearish semis or bullish software; it tests whether the data supports that handoff."
        ),
        "outputs": {
            "board": OUT_BOARD.name,
            "handoff": OUT_HANDOFF.name,
            "ticker_map": OUT_TICKER_MAP.name,
            "playbook": OUT_PLAYBOOK.name,
            "validation": OUT_VALIDATION.name,
            "history": OUT_HISTORY.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return board, handoff, ticker_map, playbook, validation, history, state


def main() -> int:
    board, handoff, ticker_map, playbook, validation, history, state = build_all()
    board.to_csv(OUT_BOARD, index=False)
    handoff.to_csv(OUT_HANDOFF, index=False)
    ticker_map.to_csv(OUT_TICKER_MAP, index=False)
    playbook.to_csv(OUT_PLAYBOOK, index=False)
    validation.to_csv(OUT_VALIDATION, index=False)
    history.to_csv(OUT_HISTORY, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Semiconductor phase: {state.get('semiconductor_phase')}",
        f"- Semiconductor exhaustion risk: {state.get('semiconductor_exhaustion_risk')}",
        f"- Software phase: {state.get('software_phase')}",
        f"- Software catch-up score: {state.get('software_catch_up_score')}",
        f"- Semis to software handoff: {state.get('semis_to_software_handoff')}",
        "",
        "## Why this step exists",
        "",
        (
            "The old sector cycle board could say Semiconductors are leading, but it could not say whether "
            "that leadership is early, healthy, crowded, or late. This step separates those ideas."
        ),
        "",
        "## Subsector Cycle Board",
        "",
        df_to_markdown(board, max_rows=40),
        "",
        "## Leadership Handoff Matrix",
        "",
        df_to_markdown(handoff, max_rows=40),
        "",
        "## Rotation Thesis Validation",
        "",
        df_to_markdown(validation, max_rows=40),
        "",
        "## Ticker Subsector Overlay",
        "",
        df_to_markdown(ticker_map, max_rows=80),
        "",
        "## Playbook",
        "",
        df_to_markdown(playbook, max_rows=40),
        "",
        "## Product Truth",
        "",
        (
            "This is a research-only overlay. It can penalize a chase, promote a watchlist, or block option enthusiasm, "
            "but it cannot place trades and cannot override risk, event, execution, or data-quality gates."
        ),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 170 - Institutional Subsector Cycle and Leadership Handoff", sections)

    print(f"wrote {OUT_BOARD.name} rows={len(board)}")
    print(f"wrote {OUT_HANDOFF.name} rows={len(handoff)}")
    print(f"wrote {OUT_TICKER_MAP.name} rows={len(ticker_map)}")
    print(f"wrote {OUT_VALIDATION.name} rows={len(validation)}")
    print(f"semiconductor_phase={state.get('semiconductor_phase')}")
    print(f"software_phase={state.get('software_phase')}")
    print(f"semis_to_software_handoff={state.get('semis_to_software_handoff')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
