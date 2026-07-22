#!/usr/bin/env python3
"""
Canyon v9 - Step 142: Sector Cycle and Linkage Map
==================================================

Research-only. No broker connection. No live orders.

This step turns the existing sector, theme, and news read-through files into a
clear sector-cycle board and a key sector-linkage map:

  - where each sector sits in the cycle
  - which sectors are leading, improving, crowded, fading, or lagging
  - which important sector pairs are moving together
  - which news/theme catalysts link one sector to another

Outputs:
  sector_cycle_state.csv
  sector_linkage_matrix.csv
  key_sector_linkage.csv
  sector_cycle_linkage_state.json
  sector_cycle_linkage_report.md
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
    read_json_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_ROTATION = ROOT / "sector_rotation_scores.csv"
IN_HEATMAP = ROOT / "theme_heatmap.csv"
IN_SECTOR_EXPOSURE = ROOT / "sector_active_exposure.csv"
IN_THEME_EXPOSURE = ROOT / "theme_factor_exposure.csv"
IN_PRICE_CACHE = ROOT / "sector_etf_price_cache.csv"
IN_SEASONAL = ROOT / "current_month_sector_bias.json"
IN_SUPPLY_CHAIN = ROOT / "news_supply_chain_readthrough.csv"
IN_THEME_CANDIDATES = ROOT / "theme_candidate_enrichment.csv"
IN_DAILY_PICKS = ROOT / "daily_picks_filtered.csv"
IN_RISK_DESK = ROOT / "risk_desk_overview.json"

OUT_CYCLE = ROOT / "sector_cycle_state.csv"
OUT_LINKAGE = ROOT / "sector_linkage_matrix.csv"
OUT_KEY = ROOT / "key_sector_linkage.csv"
OUT_STATE = ROOT / "sector_cycle_linkage_state.json"
OUT_REPORT = ROOT / "sector_cycle_linkage_report.md"


ETF_TO_SECTOR = {
    "XLK": "Technology",
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "IYR": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
    "XLP": "Consumer Staples",
}

SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Semiconductor": "SMH",
    "Semiconductors": "SMH",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP",
}

ETF_THEME_ALIAS = {
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors",
    "XLK": "Technology",
}


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def pct_value(value: Any) -> float:
    x = safe_float(value)
    if not np.isfinite(x):
        return np.nan
    if abs(x) <= 1.5:
        return x * 100.0
    return x


def norm_sector(value: Any) -> str:
    raw = text(value)
    aliases = {
        "Healthcare": "Health Care",
        "Health Care": "Health Care",
        "Tech": "Technology",
        "Consumer Disc": "Consumer Discretionary",
        "Communication": "Communication Services",
        "Semiconductor": "Semiconductors",
    }
    return aliases.get(raw, raw or "Unknown")


def load_price_cache() -> pd.DataFrame:
    df = read_csv_safe(IN_PRICE_CACHE)
    if df.empty:
        return pd.DataFrame()
    first = df.columns[0]
    if first.lower() in {"date", "unnamed: 0", "index"}:
        df[first] = pd.to_datetime(df[first], errors="coerce")
        df = df.dropna(subset=[first]).set_index(first)
    df = df.apply(pd.to_numeric, errors="coerce").ffill()
    keep = [c for c in df.columns if c in ETF_TO_SECTOR or c == "SPY"]
    return df[keep].dropna(how="all")


def seasonal_scores() -> dict[str, float]:
    data = read_json_safe(IN_SEASONAL, default={})
    out = {}
    for sector, score in (data.get("all_sectors", {}) or {}).items():
        out[norm_sector(sector)] = safe_float(score, 50.0)
    return out


def aggregate_news_by_sector(supply_chain: pd.DataFrame) -> pd.DataFrame:
    if supply_chain.empty:
        return pd.DataFrame(columns=[
            "sector", "positive_catalysts", "negative_catalysts", "mixed_catalysts",
            "theme_catalysts", "top_headline", "top_theme", "top_tickers",
        ])
    sc = supply_chain.copy()
    sector_col = "target_sector" if "target_sector" in sc.columns else "sector"
    if sector_col not in sc.columns:
        sc["target_sector"] = "Unknown"
        sector_col = "target_sector"
    sc["sector"] = sc[sector_col].map(norm_sector)
    sc["market_tone"] = sc.get("market_tone", "").astype(str).str.upper()
    rows = []
    for sector, grp in sc.groupby("sector", dropna=False):
        pos = int((grp["market_tone"] == "POSITIVE").sum())
        neg = int((grp["market_tone"] == "NEGATIVE").sum())
        mixed = int((grp["market_tone"] == "MIXED").sum())
        themes = grp.get("theme", pd.Series(dtype=str)).dropna().astype(str)
        tickers = grp.get("target_ticker", pd.Series(dtype=str)).dropna().astype(str)
        top_row = grp.head(1).iloc[0] if not grp.empty else {}
        rows.append({
            "sector": sector,
            "positive_catalysts": pos,
            "negative_catalysts": neg,
            "mixed_catalysts": mixed,
            "theme_catalysts": int(len(grp)),
            "top_headline": text(top_row.get("headline", ""))[:220],
            "top_theme": themes.mode().iloc[0] if not themes.empty else "",
            "top_tickers": ", ".join(tickers.value_counts().head(8).index.tolist()),
        })
    return pd.DataFrame(rows)


def exposure_by_sector() -> pd.DataFrame:
    df = read_csv_safe(IN_SECTOR_EXPOSURE)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "sector" in df.columns:
        df["sector"] = df["sector"].map(norm_sector)
    return df


def picks_by_sector() -> pd.DataFrame:
    df = read_csv_safe(IN_DAILY_PICKS)
    if df.empty or "sector" not in df.columns:
        return pd.DataFrame(columns=["sector", "top_alpha_names", "avg_alpha_score", "buy_count"])
    work = df.copy()
    work["sector"] = work["sector"].map(norm_sector)
    work["alpha_score"] = pd.to_numeric(work.get("alpha_score", np.nan), errors="coerce")
    rows = []
    for sector, grp in work.groupby("sector", dropna=False):
        rows.append({
            "sector": sector,
            "top_alpha_names": ", ".join(grp.sort_values("alpha_score", ascending=False).get("ticker", pd.Series(dtype=str)).head(6).astype(str).tolist()),
            "avg_alpha_score": round(float(grp["alpha_score"].mean()), 2) if grp["alpha_score"].notna().any() else np.nan,
            "buy_count": int(grp.get("action", pd.Series(dtype=str)).astype(str).str.contains("BUY", case=False, na=False).sum()),
        })
    return pd.DataFrame(rows)


def sector_from_rotation_row(row: pd.Series) -> str:
    ticker = text(row.get("ticker")).upper()
    theme = text(row.get("theme"))
    if ticker in ETF_THEME_ALIAS:
        return ETF_THEME_ALIAS[ticker]
    if ticker in ETF_TO_SECTOR:
        return ETF_TO_SECTOR[ticker]
    return norm_sector(theme)


def cycle_state(score: float, rel20: float, rel63: float, ret20: float, ret63: float, label: str, seasonal: float, pos: int, neg: int, cap_status: str) -> tuple[str, str]:
    label_u = text(label).upper()
    cap_u = text(cap_status).upper()
    catalyst_balance = pos - neg
    if label_u == "LEADER" and score >= 65 and rel20 > 0 and ret20 > 0:
        if cap_u in {"SIZE_DOWN", "BLOCK_NEW"}:
            return "Crowded leadership", "Strong sector tape, but portfolio/risk concentration is already high."
        if catalyst_balance >= 3:
            return "Leadership expansion", "Price leadership is supported by positive catalyst read-through."
        return "Leadership", "Sector is leading on price and relative strength."
    if label_u in {"LEADER", "WATCH"} and ret20 > 0 and (rel20 > 0 or seasonal >= 60):
        return "Early improvement", "Sector is improving; watch for confirmation before increasing attention."
    if ret20 < 0 and rel20 < 0 and ret63 > 0:
        return "Fading momentum", "Longer trend remains alive, but short-term relative strength is slipping."
    if label_u == "LAGGARD" or (rel20 < 0 and rel63 < 0 and ret20 < 0):
        return "Downcycle / laggard", "Sector is underperforming across current windows."
    if neg > pos + 3 and ret20 <= 0:
        return "Event pressure", "Negative news pressure is larger than positive read-through."
    return "Neutral / base", "No clean cycle edge; treat as context, not a reason to force a trade."


def build_cycle_board() -> pd.DataFrame:
    rotation = read_csv_safe(IN_ROTATION)
    if rotation.empty:
        rotation = read_csv_safe(IN_HEATMAP)
    exposure = exposure_by_sector()
    supply = read_csv_safe(IN_SUPPLY_CHAIN)
    news = aggregate_news_by_sector(supply)
    picks = picks_by_sector()
    seasonal = seasonal_scores()

    if rotation.empty:
        return pd.DataFrame()

    rows = []
    for _, row in rotation.iterrows():
        etf = text(row.get("ticker")).upper()
        sector = sector_from_rotation_row(row)
        ret20 = pct_value(row.get("ret_20d"))
        ret63 = pct_value(row.get("ret_63d"))
        rel20 = pct_value(row.get("relative_20d_vs_spy"))
        rel63 = pct_value(row.get("relative_63d_vs_spy"))
        score = safe_float(row.get("rotation_score"), 0.0)
        label = text(row.get("rotation_label")) or "NO_DATA"
        exp_row = exposure[exposure["sector"].astype(str).eq(sector)].head(1) if not exposure.empty and "sector" in exposure.columns else pd.DataFrame()
        news_row = news[news["sector"].astype(str).eq(sector)].head(1) if not news.empty else pd.DataFrame()
        pick_row = picks[picks["sector"].astype(str).eq(sector)].head(1) if not picks.empty else pd.DataFrame()
        pos = int(safe_float(news_row["positive_catalysts"].iloc[0], 0)) if not news_row.empty else 0
        neg = int(safe_float(news_row["negative_catalysts"].iloc[0], 0)) if not news_row.empty else 0
        mixed = int(safe_float(news_row["mixed_catalysts"].iloc[0], 0)) if not news_row.empty else 0
        season = float(seasonal.get(sector, 50.0))
        cap_status = text(exp_row["cap_status"].iloc[0]) if not exp_row.empty and "cap_status" in exp_row.columns else "NO_POSITION"
        state, note = cycle_state(score, rel20, rel63, ret20, ret63, label, season, pos, neg, cap_status)
        catalyst_balance = pos - neg
        cycle_score = (
            score * 0.50
            + np.nan_to_num(rel20) * 0.60
            + np.nan_to_num(rel63) * 0.30
            + (season - 50.0) * 0.20
            + catalyst_balance * 1.25
        )
        rows.append({
            "etf": etf,
            "sector": sector,
            "cycle_state": state,
            "rotation_label": label,
            "cycle_score": round(float(cycle_score), 2),
            "rotation_score": round(float(score), 2),
            "ret_20d_pct": round(ret20, 2) if np.isfinite(ret20) else np.nan,
            "ret_63d_pct": round(ret63, 2) if np.isfinite(ret63) else np.nan,
            "relative_20d_vs_spy_pct": round(rel20, 2) if np.isfinite(rel20) else np.nan,
            "relative_63d_vs_spy_pct": round(rel63, 2) if np.isfinite(rel63) else np.nan,
            "seasonal_score": round(season, 2),
            "positive_catalysts": pos,
            "negative_catalysts": neg,
            "mixed_catalysts": mixed,
            "catalyst_balance": catalyst_balance,
            "portfolio_weight_pct": round(safe_float(exp_row["portfolio_weight_pct"].iloc[0]), 2) if not exp_row.empty and "portfolio_weight_pct" in exp_row.columns else np.nan,
            "active_weight_pct": round(safe_float(exp_row["active_weight_pct"].iloc[0]), 2) if not exp_row.empty and "active_weight_pct" in exp_row.columns else np.nan,
            "cap_status": cap_status,
            "top_portfolio_tickers": text(exp_row["top_tickers"].iloc[0]) if not exp_row.empty and "top_tickers" in exp_row.columns else "",
            "top_alpha_names": text(pick_row["top_alpha_names"].iloc[0]) if not pick_row.empty else "",
            "top_theme": text(news_row["top_theme"].iloc[0]) if not news_row.empty else "",
            "top_news_tickers": text(news_row["top_tickers"].iloc[0]) if not news_row.empty else "",
            "top_headline": text(news_row["top_headline"].iloc[0]) if not news_row.empty else "",
            "cycle_note": note,
            "source_file": "sector_rotation_scores.csv; sector_active_exposure.csv; news_supply_chain_readthrough.csv; current_month_sector_bias.json",
            "research_only": True,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["cycle_score", "rotation_score"], ascending=False).reset_index(drop=True)


def build_linkage_matrix(cycle: pd.DataFrame) -> pd.DataFrame:
    prices = load_price_cache()
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna(how="all")
    etfs = [c for c in returns.columns if c in ETF_TO_SECTOR]
    cycle_by_etf = {r["etf"]: r for _, r in cycle.iterrows()} if not cycle.empty and "etf" in cycle.columns else {}
    rows = []
    for i, a in enumerate(etfs):
        for b in etfs[i + 1:]:
            ra = returns[a].dropna()
            rb = returns[b].dropna()
            pair = pd.concat([ra, rb], axis=1).dropna()
            if len(pair) < 40:
                continue
            corr60 = pair.tail(60).corr().iloc[0, 1] if len(pair) >= 60 else pair.corr().iloc[0, 1]
            corr120 = pair.tail(120).corr().iloc[0, 1] if len(pair) >= 120 else pair.corr().iloc[0, 1]
            a_score = safe_float(cycle_by_etf.get(a, {}).get("cycle_score"), 0.0)
            b_score = safe_float(cycle_by_etf.get(b, {}).get("cycle_score"), 0.0)
            a_sector = text(cycle_by_etf.get(a, {}).get("sector")) or ETF_TO_SECTOR.get(a, a)
            b_sector = text(cycle_by_etf.get(b, {}).get("sector")) or ETF_TO_SECTOR.get(b, b)
            score_gap = a_score - b_score
            if corr60 >= 0.78:
                link_type = "High co-movement"
            elif corr60 >= 0.55:
                link_type = "Moderate co-movement"
            elif corr60 <= 0.20:
                link_type = "Low linkage / possible diversifier"
            else:
                link_type = "Loose linkage"
            if abs(score_gap) >= 15:
                leader = a_sector if score_gap > 0 else b_sector
                follower = b_sector if score_gap > 0 else a_sector
                direction = f"{leader} leading {follower}"
            else:
                direction = "No clear leader"
            rows.append({
                "sector_a": a_sector,
                "etf_a": a,
                "sector_b": b_sector,
                "etf_b": b,
                "corr_60d": round(float(corr60), 3) if np.isfinite(corr60) else np.nan,
                "corr_120d": round(float(corr120), 3) if np.isfinite(corr120) else np.nan,
                "cycle_score_a": round(float(a_score), 2),
                "cycle_score_b": round(float(b_score), 2),
                "cycle_score_gap": round(float(score_gap), 2),
                "linkage_type": link_type,
                "leadership_direction": direction,
                "source_file": "sector_etf_price_cache.csv; sector_cycle_state.csv",
                "research_only": True,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_rank"] = out["corr_60d"].abs().fillna(0) + out["cycle_score_gap"].abs().fillna(0) / 100
    out = out.sort_values("_rank", ascending=False).drop(columns=["_rank"]).reset_index(drop=True)
    return out


def build_key_linkage(cycle: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not matrix.empty:
        high = matrix[
            (pd.to_numeric(matrix["corr_60d"], errors="coerce").abs() >= 0.65)
            | (pd.to_numeric(matrix["cycle_score_gap"], errors="coerce").abs() >= 20)
        ].head(25)
        for _, row in high.iterrows():
            corr = safe_float(row.get("corr_60d"))
            gap = safe_float(row.get("cycle_score_gap"))
            if abs(gap) >= 20 and corr >= 0.45:
                action = "Watch leader-to-follower rotation"
            elif corr >= 0.78:
                action = "Treat as linked exposure"
            elif corr <= 0.20:
                action = "Possible diversifier, verify stress correlation"
            else:
                action = "Monitor linkage"
            rows.append({
                "link_source": "price_correlation",
                "primary_sector": row.get("sector_a", ""),
                "linked_sector": row.get("sector_b", ""),
                "primary_etf": row.get("etf_a", ""),
                "linked_etf": row.get("etf_b", ""),
                "linkage_type": row.get("linkage_type", ""),
                "corr_60d": row.get("corr_60d", np.nan),
                "cycle_score_gap": row.get("cycle_score_gap", np.nan),
                "leadership_direction": row.get("leadership_direction", ""),
                "catalyst_theme": "",
                "representative_tickers": "",
                "top_headline": "",
                "desk_action": action,
                "evidence_note": "Derived from sector ETF return correlation and relative cycle score.",
                "source_file": "sector_linkage_matrix.csv",
                "research_only": True,
            })

    supply = read_csv_safe(IN_SUPPLY_CHAIN)
    if not supply.empty:
        sc = supply.copy()
        sc["theme"] = sc.get("theme", "").astype(str)
        sc["chain_role"] = sc.get("chain_role", "").astype(str)
        sc["market_tone"] = sc.get("market_tone", "").astype(str).str.upper()
        if "target_sector" not in sc.columns:
            sc["target_sector"] = "Unknown"
        sc["target_sector"] = sc["target_sector"].map(norm_sector)
        grouped = sc.groupby(["theme", "target_sector", "chain_role"], dropna=False)
        catalyst_rows = []
        for (theme, sector, role), grp in grouped:
            if not text(theme) or not text(sector):
                continue
            pos = int((grp["market_tone"] == "POSITIVE").sum())
            neg = int((grp["market_tone"] == "NEGATIVE").sum())
            count = int(len(grp))
            if count < 3 and pos + neg < 2:
                continue
            top = grp.head(1).iloc[0]
            tickers = ", ".join(grp.get("target_ticker", pd.Series(dtype=str)).dropna().astype(str).value_counts().head(8).index.tolist())
            catalyst_rows.append({
                "theme": theme,
                "sector": sector,
                "role": role,
                "count": count,
                "pos": pos,
                "neg": neg,
                "tickers": tickers,
                "headline": text(top.get("headline", ""))[:220],
            })
        for item in sorted(catalyst_rows, key=lambda x: (x["count"], x["pos"], -x["neg"]), reverse=True)[:25]:
            tone = "positive" if item["pos"] > item["neg"] else "negative" if item["neg"] > item["pos"] else "mixed"
            rows.append({
                "link_source": "news_theme_readthrough",
                "primary_sector": item["theme"],
                "linked_sector": item["sector"],
                "primary_etf": "",
                "linked_etf": SECTOR_TO_ETF.get(item["sector"], ""),
                "linkage_type": f"{item['role']} {tone} catalyst",
                "corr_60d": np.nan,
                "cycle_score_gap": np.nan,
                "leadership_direction": "Catalyst read-through",
                "catalyst_theme": item["theme"],
                "representative_tickers": item["tickers"],
                "top_headline": item["headline"],
                "desk_action": "Map headline into linked sector watchlist",
                "evidence_note": f"{item['count']} read-through rows; positive={item['pos']}; negative={item['neg']}.",
                "source_file": "news_supply_chain_readthrough.csv",
                "research_only": True,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.head(60).reset_index(drop=True)


def build_state(cycle: pd.DataFrame, matrix: pd.DataFrame, key: pd.DataFrame) -> dict[str, Any]:
    if cycle.empty:
        overall = "NO_SECTOR_CYCLE_DATA"
    else:
        states = cycle["cycle_state"].astype(str)
        leaders = int(states.str.contains("Leadership", case=False, na=False).sum())
        improving = int(states.str.contains("Early improvement", case=False, na=False).sum())
        crowded = int(states.str.contains("Crowded", case=False, na=False).sum())
        laggards = int(states.str.contains("Downcycle|Fading", case=False, na=False).sum())
        if leaders >= 3 and crowded == 0:
            overall = "BROAD_LEADERSHIP"
        elif leaders >= 1:
            overall = "SELECTIVE_LEADERSHIP"
        elif improving >= 2:
            overall = "EARLY_ROTATION"
        elif laggards >= max(3, len(cycle) // 3):
            overall = "WEAK_ROTATION"
        else:
            overall = "MIXED_ROTATION"
    states = cycle["cycle_state"].astype(str) if not cycle.empty else pd.Series(dtype=str)
    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Sector cycle and linkage context only. It can focus research attention, but risk gates still decide sizing.",
        "sectors_checked": int(len(cycle)),
        "leadership_count": int(states.str.contains("Leadership", case=False, na=False).sum()) if not cycle.empty else 0,
        "crowded_leadership_count": int(states.str.contains("Crowded", case=False, na=False).sum()) if not cycle.empty else 0,
        "early_improvement_count": int(states.str.contains("Early improvement", case=False, na=False).sum()) if not cycle.empty else 0,
        "laggard_or_fading_count": int(states.str.contains("Downcycle|Fading", case=False, na=False).sum()) if not cycle.empty else 0,
        "linkage_pairs": int(len(matrix)),
        "key_linkage_rows": int(len(key)),
        "top_sector": text(cycle.iloc[0].get("sector")) if not cycle.empty else "NO_DATA",
        "top_sector_cycle_state": text(cycle.iloc[0].get("cycle_state")) if not cycle.empty else "NO_DATA",
        "outputs": {
            "cycle": OUT_CYCLE.name,
            "linkage_matrix": OUT_LINKAGE.name,
            "key_linkage": OUT_KEY.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    cycle = build_cycle_board()
    linkage = build_linkage_matrix(cycle)
    key = build_key_linkage(cycle, linkage)
    state = build_state(cycle, linkage, key)

    cycle.to_csv(OUT_CYCLE, index=False)
    linkage.to_csv(OUT_LINKAGE, index=False)
    key.to_csv(OUT_KEY, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Sectors checked: {state.get('sectors_checked', 0)}",
        f"- Leadership sectors: {state.get('leadership_count', 0)}",
        f"- Crowded leadership: {state.get('crowded_leadership_count', 0)}",
        f"- Early improvement: {state.get('early_improvement_count', 0)}",
        f"- Laggard or fading: {state.get('laggard_or_fading_count', 0)}",
        f"- Linkage pairs: {state.get('linkage_pairs', 0)}",
        f"- Key linkage rows: {state.get('key_linkage_rows', 0)}",
        "",
        "## Sector Cycle Board",
        "",
        df_to_markdown(cycle, max_rows=40),
        "",
        "## Key Sector Linkage",
        "",
        df_to_markdown(key, max_rows=60),
        "",
        "## Full Linkage Matrix",
        "",
        df_to_markdown(linkage, max_rows=80),
        "",
        "## Product Truth",
        "",
        "Sector cycle and linkage are context layers. They can focus research, but they cannot override risk, event, liquidity, or execution gates.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 142 - Sector Cycle and Linkage Map", sections)

    print(f"wrote {OUT_CYCLE.name} rows={len(cycle)}")
    print(f"wrote {OUT_KEY.name} rows={len(key)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
