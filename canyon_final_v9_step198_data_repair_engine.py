#!/usr/bin/env python3
"""
Canyon v9 Step 198 - Data Repair Engine.

Research-only. No broker connection. No live orders.

Step197 identifies gaps. Step198 turns those gaps into repair packets:
1. price refresh attempts through public yfinance data
2. risk-book intake queue for tickers that cannot be promoted yet
3. news-proof and execution-proof work queues
4. a plain-English repair priority board

It writes a supplemental price cache instead of overwriting original data.

Outputs:
  data_repair_state.json
  price_repair_attempts.csv
  price_repair_download_cache.csv
  risk_book_repair_intake_queue.csv
  news_proof_repair_queue.csv
  execution_spread_repair_queue.csv
  data_repair_priority_board.csv
  data_repair_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "data_repair_state.json"
OUT_PRICE_ATTEMPTS = ROOT / "price_repair_attempts.csv"
OUT_PRICE_CACHE = ROOT / "price_repair_download_cache.csv"
OUT_RISK_BOOK = ROOT / "risk_book_repair_intake_queue.csv"
OUT_NEWS_PROOF = ROOT / "news_proof_repair_queue.csv"
OUT_EXECUTION = ROOT / "execution_spread_repair_queue.csv"
OUT_PRIORITY = ROOT / "data_repair_priority_board.csv"
OUT_REPORT = ROOT / "data_repair_report.md"

MAX_PRICE_DOWNLOAD_TICKERS = 120


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    out = safe_float(value, np.nan)
    if not np.isfinite(out):
        return default
    return int(out)


def short(value: Any, limit: int = 240) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def plain(value: Any, default: str = "No data") -> str:
    text = as_text(value, default)
    replacements = {
        "DATA_GAP": "missing data",
        "REDUCE_ONLY": "reduce only",
        "SIZE_DOWN": "use smaller size",
        "NO_GO": "not allowed",
        "P1_REVIEW_CONTRADICTION": "urgent contradiction review",
        "CONTRADICTED_REVIEW_REQUIRED": "contradiction review required",
        "PRICE_DISAGREES": "price reaction disagrees",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    return " ".join(text.replace("_", " ").split())


def one_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""].copy()
    out: dict[str, pd.Series] = {}
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get(ticker_col))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def choose_price_repair_tickers(price_desk: pd.DataFrame, repair_queue: pd.DataFrame) -> list[str]:
    tickers: list[str] = []
    if not price_desk.empty and {"ticker", "price_status"}.issubset(price_desk.columns):
        work = price_desk.copy()
        work["ticker"] = work["ticker"].apply(clean_ticker)
        work["_rank"] = work["price_status"].astype(str).map({"Missing": 0, "Stale": 1, "Fresh enough": 9}).fillna(5)
        need = work[work["_rank"] <= 1].sort_values(["_rank", "ticker"])
        tickers.extend(need["ticker"].dropna().tolist())

    if not repair_queue.empty and {"ticker", "repair_type"}.issubset(repair_queue.columns):
        mask = repair_queue["repair_type"].astype(str).str.contains("Price", case=False, na=False)
        tickers.extend(repair_queue.loc[mask, "ticker"].dropna().map(clean_ticker).tolist())

    out: list[str] = []
    for ticker in tickers:
        if ticker and ticker not in out and ticker != "PORTFOLIO":
            out.append(ticker)
    return out[:MAX_PRICE_DOWNLOAD_TICKERS]


def extract_download_rows(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw is None or raw.empty:
        return pd.DataFrame()

    def append_rows(ticker: str, sub: pd.DataFrame) -> None:
        if sub is None or sub.empty:
            return
        work = sub.copy()
        work = work.reset_index()
        date_col = "Date" if "Date" in work.columns else work.columns[0]
        for _, row in work.iterrows():
            close = safe_float(row.get("Close"), np.nan)
            adj_close = safe_float(row.get("Adj Close"), close)
            if not np.isfinite(close) and not np.isfinite(adj_close):
                continue
            rows.append({
                "date": pd.to_datetime(row.get(date_col), errors="coerce").date().isoformat()
                if pd.notna(pd.to_datetime(row.get(date_col), errors="coerce")) else "",
                "ticker": ticker,
                "open": safe_float(row.get("Open"), np.nan),
                "high": safe_float(row.get("High"), np.nan),
                "low": safe_float(row.get("Low"), np.nan),
                "close": close if np.isfinite(close) else adj_close,
                "adj_close": adj_close,
                "volume": safe_float(row.get("Volume"), np.nan),
                "source_file": "price_repair_download_cache.csv",
                "source_vendor": "YFINANCE_PUBLIC_DAILY",
                "download_status": "Downloaded",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(str(x).upper() for x in raw.columns.get_level_values(0))
        level1 = set(str(x).upper() for x in raw.columns.get_level_values(1))
        for ticker in tickers:
            ticker = clean_ticker(ticker)
            if ticker in level0:
                append_rows(ticker, raw[ticker])
            elif ticker in level1:
                append_rows(ticker, raw.xs(ticker, axis=1, level=1))
    elif len(tickers) == 1:
        append_rows(clean_ticker(tickers[0]), raw)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out[out["date"] != ""].drop_duplicates(["ticker", "date"], keep="last")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_price_repair_cache(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts = []
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(columns=[
            "ticker", "repair_status", "latest_download_date", "latest_download_close", "note",
            "research_only", "no_broker_connection", "no_live_orders",
        ])

    try:
        import yfinance as yf
        raw = yf.download(
            tickers=tickers,
            period="18mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        cache = extract_download_rows(raw, tickers)
    except Exception as exc:
        cache = pd.DataFrame()
        for ticker in tickers:
            attempts.append({
                "ticker": ticker,
                "repair_status": "Download failed",
                "latest_download_date": "",
                "latest_download_close": np.nan,
                "note": f"Public price refresh failed: {type(exc).__name__}: {short(exc, 160)}",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
        return cache, pd.DataFrame(attempts)

    cache_map = {}
    if not cache.empty:
        latest = cache.sort_values(["ticker", "date"]).drop_duplicates("ticker", keep="last")
        cache_map = latest.set_index("ticker").to_dict("index")

    for ticker in tickers:
        row = cache_map.get(ticker, {})
        if row:
            attempts.append({
                "ticker": ticker,
                "repair_status": "Downloaded public daily prices",
                "latest_download_date": as_text(row.get("date")),
                "latest_download_close": safe_float(row.get("close"), np.nan),
                "note": "Supplemental cache created. It is public market data, not broker/live execution data.",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
        else:
            attempts.append({
                "ticker": ticker,
                "repair_status": "No public price rows returned",
                "latest_download_date": "",
                "latest_download_close": np.nan,
                "note": "Ticker may be invalid, unavailable, delisted, or blocked by data source limits.",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return cache, pd.DataFrame(attempts)


def build_risk_book_repair_queue(repair_queue: pd.DataFrame) -> pd.DataFrame:
    gate = read_csv_safe(ROOT / "institutional_promotion_gate.csv")
    cards = read_csv_safe(ROOT / "sharpe4_risk_book_candidate_cards.csv")
    var_liq = read_csv_safe(ROOT / "sharpe4_risk_book_var_liquidity.csv")
    event_route = read_csv_safe(ROOT / "sharpe4_risk_book_event_route.csv")
    sector_map = read_csv_safe(ROOT / "sector_map.csv")

    gate_map = one_by_ticker(gate)
    card_map = one_by_ticker(cards)
    var_map = one_by_ticker(var_liq)
    event_map = one_by_ticker(event_route)
    sector_map_by_ticker = one_by_ticker(sector_map)

    tickers: list[str] = []
    if not repair_queue.empty and {"ticker", "repair_type"}.issubset(repair_queue.columns):
        mask = repair_queue["repair_type"].astype(str).str.contains("Risk book", case=False, na=False)
        tickers = repair_queue.loc[mask, "ticker"].dropna().map(clean_ticker).tolist()
    if not tickers and not gate.empty and {"ticker", "first_blocker"}.issubset(gate.columns):
        mask = gate["first_blocker"].astype(str).str.contains("risk book", case=False, na=False)
        tickers = gate.loc[mask, "ticker"].dropna().map(clean_ticker).tolist()

    rows = []
    seen = set()
    for ticker in tickers:
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        gate_row = gate_map.get(ticker)
        card_row = card_map.get(ticker)
        var_row = var_map.get(ticker)
        event_row = event_map.get(ticker)
        sector_row = sector_map_by_ticker.get(ticker)
        sector = as_text(
            (gate_row.get("sector_or_theme") if gate_row is not None and "sector_or_theme" in gate_row.index else ""),
            as_text(sector_row.get("sector") if sector_row is not None and "sector" in sector_row.index else "", "Unknown"),
        )
        rows.append({
            "priority": "P1",
            "ticker": ticker,
            "sector_or_theme": sector,
            "current_answer": short(card_row.get("current_answer") if card_row is not None else "Research only. Build risk entry first.", 220),
            "main_blocker": plain(gate_row.get("first_blocker") if gate_row is not None else "Ticker is not in the risk book yet."),
            "risk_level": as_text(var_row.get("price_risk") if var_row is not None and "price_risk" in var_row.index else card_row.get("risk_level") if card_row is not None and "risk_level" in card_row.index else "Needs risk calculation", "Needs risk calculation"),
            "liquidity": as_text(var_row.get("liquidity_status") if var_row is not None and "liquidity_status" in var_row.index else card_row.get("liquidity") if card_row is not None and "liquidity" in card_row.index else "Needs liquidity proof", "Needs liquidity proof"),
            "daily_cvar_95_pct": safe_float(var_row.get("daily_cvar_95_pct") if var_row is not None and "daily_cvar_95_pct" in var_row.index else np.nan, np.nan),
            "starter_cap_after_all_gates_clear_pct": safe_float(card_row.get("starter_cap_after_all_gates_clear_pct") if card_row is not None and "starter_cap_after_all_gates_clear_pct" in card_row.index else np.nan, np.nan),
            "paper_stop_if_ever_tested_pct": safe_float(card_row.get("paper_stop_if_ever_tested_pct") if card_row is not None and "paper_stop_if_ever_tested_pct" in card_row.index else np.nan, np.nan),
            "news_or_event_hook": short(event_row.get("event_route") if event_row is not None and "event_route" in event_row.index else gate_row.get("news_headline") if gate_row is not None and "news_headline" in gate_row.index else "No event hook proven yet.", 220),
            "fields_to_fill": "single-name VaR/CVaR; earnings date and gap risk; liquidity/spread proof; sector/factor crowding; stop rule; source timestamp",
            "done_when": "Risk-book entry has traceable source files and the Final PM Gate no longer says this ticker is outside the risk book.",
            "still_forbidden": "No paper size, no calls, no puts, and no live orders until this queue item is complete.",
            "source_files": "data_gap_repair_queue.csv; institutional_promotion_gate.csv; sharpe4_risk_book_* files",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def build_news_proof_queue(repair_queue: pd.DataFrame) -> pd.DataFrame:
    event_queue = read_csv_safe(ROOT / "event_causal_validation_queue.csv")
    if event_queue.empty:
        return pd.DataFrame(columns=[
            "priority", "ticker", "headline", "why_blocked", "proof_to_collect",
            "done_when", "source_link", "research_only",
        ])
    rows = []
    for _, row in event_queue.head(120).iterrows():
        ticker = clean_ticker(row.get("target_ticker"))
        if not ticker:
            continue
        rows.append({
            "priority": "P1" if str(row.get("priority", "")).upper().startswith("P1") else "P2",
            "ticker": ticker,
            "headline": short(row.get("headline"), 240),
            "tone": plain(row.get("market_tone"), "No tone"),
            "why_blocked": plain(row.get("issue"), "News proof incomplete"),
            "proof_to_collect": short(row.get("required_next_action"), 260) or "Prove source, timestamp, target link, and post-news price reaction.",
            "done_when": "The causal link is direct enough to explain which ticker may help or hurt, and the price reaction does not contradict it.",
            "source_link": as_text(row.get("link")),
            "source_files": "event_causal_validation_queue.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).drop_duplicates(["ticker", "headline"], keep="first")


def build_execution_queue(repair_queue: pd.DataFrame) -> pd.DataFrame:
    execution = read_csv_safe(ROOT / "depth5_execution_liquidity_desk.csv")
    options = read_csv_safe(ROOT / "options_execution_route_matrix.csv")
    rows = []
    if not execution.empty:
        for _, row in execution.head(120).iterrows():
            permission = as_text(row.get("execution_permission"))
            status = as_text(row.get("execution_status"))
            if not any(word in f"{permission} {status}".lower() for word in ["manual", "no new", "risk reduction", "spread", "liquidity"]):
                continue
            rows.append({
                "priority": "P1" if "no new" in permission.lower() or "risk reduction" in permission.lower() else "P2",
                "ticker": clean_ticker(row.get("ticker")),
                "repair_type": "Execution and liquidity",
                "current_block": plain(f"{permission} / {status}"),
                "cost_read": f"Base {safe_float(row.get('base_cost_bps'), np.nan):.1f} bps; stress {safe_float(row.get('stress_cost_bps'), np.nan):.1f} bps",
                "proof_to_collect": short(row.get("what_to_do"), 260) or "Collect spread, volume, and fill-quality proof.",
                "done_when": "Spread, volume, and realistic fill assumptions are recorded before any paper route.",
                "source_files": "depth5_execution_liquidity_desk.csv",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    if not options.empty and {"ticker", "spread_status"}.issubset(options.columns):
        mask = options["spread_status"].astype(str).str.contains("DATA_GAP|missing", case=False, na=False)
        for _, row in options[mask].head(120).iterrows():
            rows.append({
                "priority": "P2",
                "ticker": clean_ticker(row.get("ticker")),
                "repair_type": "Option spread data",
                "current_block": "Option spread or liquidity is missing.",
                "cost_read": f"Base {safe_float(row.get('base_cost_bps'), np.nan):.1f} bps; stress {safe_float(row.get('stress_cost_bps'), np.nan):.1f} bps",
                "proof_to_collect": "Manually check option spread, IV rank, volume/open interest, and no-go conditions.",
                "done_when": "Option route can explain call/put/hedge route without missing spread data.",
                "source_files": "options_execution_route_matrix.csv",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["ticker", "repair_type", "current_block"], keep="first")


def build_priority_board(price_attempts: pd.DataFrame, risk_book: pd.DataFrame, news: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    downloaded = int(price_attempts.get("repair_status", pd.Series(dtype=str)).astype(str).str.contains("Downloaded", case=False, na=False).sum()) if not price_attempts.empty else 0
    failed_price = int(len(price_attempts) - downloaded) if not price_attempts.empty else 0
    rows.append({
        "priority": "P0",
        "workstream": "Price repair",
        "plain_answer": f"{downloaded} tickers downloaded into the supplemental cache; {failed_price} still need manual price proof.",
        "why_it_matters": "Without price history, forward validation and risk checks cannot be trusted.",
        "next_step": "Rerun Data Reliability after refresh. Manually check tickers with no public rows returned.",
        "source_files": "price_repair_attempts.csv; price_repair_download_cache.csv",
    })
    rows.append({
        "priority": "P1",
        "workstream": "Risk-book intake",
        "plain_answer": f"{len(risk_book)} tickers need a risk-book entry before promotion.",
        "why_it_matters": "The final gate cannot allow size, calls, or puts for names outside the risk book.",
        "next_step": "Fill VaR/CVaR, earnings gap, liquidity, sector/factor crowding, and stop-rule fields.",
        "source_files": "risk_book_repair_intake_queue.csv",
    })
    rows.append({
        "priority": "P1",
        "workstream": "News proof",
        "plain_answer": f"{len(news)} news-to-ticker links still need causal proof.",
        "why_it_matters": "News is not a trade signal until the affected ticker, timing, and price reaction are proven.",
        "next_step": "Clear direct source, timestamp, affected target, and post-news reaction for P1 headlines.",
        "source_files": "news_proof_repair_queue.csv",
    })
    rows.append({
        "priority": "P2",
        "workstream": "Execution and spread",
        "plain_answer": f"{len(execution)} execution or option-spread rows still need manual proof.",
        "why_it_matters": "A high score can vanish if costs, spread, or fills are unrealistic.",
        "next_step": "Add spread, volume, fill, and option liquidity proof before any paper route.",
        "source_files": "execution_spread_repair_queue.csv",
    })
    return pd.DataFrame(rows)


def maybe_rerun_memory_and_reliability() -> str:
    notes: list[str] = []
    try:
        import canyon_final_v9_step196_decision_memory_center as step196
        step196.main()
        notes.append("Step196 rerun after price repair.")
    except Exception as exc:
        notes.append(f"Step196 rerun skipped: {type(exc).__name__}: {short(exc, 180)}")

    try:
        import canyon_final_v9_step197_price_data_reliability_center as step197
        step197.main()
        notes.append("Step197 rerun after price repair.")
    except Exception as exc:
        notes.append(f"Step197 rerun skipped: {type(exc).__name__}: {short(exc, 180)}")
    return " ".join(notes)


def main() -> None:
    price_desk = read_csv_safe(ROOT / "price_refresh_desk.csv")
    repair_queue = read_csv_safe(ROOT / "data_gap_repair_queue.csv")

    price_tickers = choose_price_repair_tickers(price_desk, repair_queue)
    if price_tickers:
        price_cache, price_attempts = download_price_repair_cache(price_tickers)
        if price_cache.empty and OUT_PRICE_CACHE.exists():
            old_cache = read_csv_safe(OUT_PRICE_CACHE)
            if not old_cache.empty:
                price_cache = old_cache
    else:
        price_cache = read_csv_safe(OUT_PRICE_CACHE)
        price_attempts = read_csv_safe(OUT_PRICE_ATTEMPTS)
        if price_attempts.empty:
            price_attempts = pd.DataFrame(columns=[
                "ticker", "repair_status", "latest_download_date", "latest_download_close", "note",
                "research_only", "no_broker_connection", "no_live_orders",
            ])
    risk_book = build_risk_book_repair_queue(repair_queue)
    news = build_news_proof_queue(repair_queue)
    execution = build_execution_queue(repair_queue)
    priority = build_priority_board(price_attempts, risk_book, news, execution)

    price_attempts.to_csv(OUT_PRICE_ATTEMPTS, index=False)
    price_cache.to_csv(OUT_PRICE_CACHE, index=False)
    risk_book.to_csv(OUT_RISK_BOOK, index=False)
    news.to_csv(OUT_NEWS_PROOF, index=False)
    execution.to_csv(OUT_EXECUTION, index=False)
    priority.to_csv(OUT_PRIORITY, index=False)

    reliability_rerun_note = maybe_rerun_memory_and_reliability()
    downloaded_count = int(price_attempts.get("repair_status", pd.Series(dtype=str)).astype(str).str.contains("Downloaded", case=False, na=False).sum()) if not price_attempts.empty else 0
    failed_price_count = int(len(price_attempts) - downloaded_count) if not price_attempts.empty else 0
    state = {
        "date": today_str(),
        "status": "DATA_REPAIR_ACTIVE",
        "price_repair_tickers_attempted": len(price_attempts),
        "price_repair_downloaded_count": downloaded_count,
        "price_repair_unresolved_count": failed_price_count,
        "price_repair_cache_rows": len(price_cache),
        "risk_book_repair_count": len(risk_book),
        "news_proof_repair_count": len(news),
        "execution_spread_repair_count": len(execution),
        "reliability_rerun_note": reliability_rerun_note,
        "plain_answer": (
            f"Data repair is active. {downloaded_count} tickers received supplemental public price rows; "
            f"{failed_price_count} still need manual price proof. {len(risk_book)} names need risk-book intake, "
            f"{len(news)} news links need proof, and {len(execution)} execution/spread rows need repair."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Priority Board\n\n" + df_to_markdown(priority),
        "## Price Repair Attempts\n\n" + df_to_markdown(price_attempts.head(120)),
        "## Risk-Book Intake Queue\n\n" + df_to_markdown(risk_book.head(120)),
        "## News Proof Queue\n\n" + df_to_markdown(news.head(120)),
        "## Execution / Spread Repair Queue\n\n" + df_to_markdown(execution.head(120)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 198 - Data Repair Engine", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Price repair downloaded: {downloaded_count} | unresolved: {failed_price_count}")
    print(f"[OK] Risk-book repair queue: {len(risk_book)}")
    print(f"[OK] News proof queue: {len(news)} | execution/spread queue: {len(execution)}")
    print(f"[OK] {reliability_rerun_note}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
