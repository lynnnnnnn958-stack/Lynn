#!/usr/bin/env python3
"""
canyon_final_v9_step99_news_aggregator.py
==========================================
Canyon v9 — Step 99: News Aggregator

Fetches recent news for the top N tickers in alpha_scores.csv / daily_picks.csv
using yfinance (free, no API key required) and writes stock_news.json.

The JSON cache is consumed by the Canyon dashboard for per-stock news drill-down.

Inputs (tried in order):
  daily_picks.csv       — sorted by alpha_score descending (most relevant)
  alpha_scores.csv      — sorted by alpha_rank ascending
  regime_ml_scores.csv  — sorted by predicted_score descending

Output:
  stock_news.json       — {updated, tickers_fetched, news: {TICKER: [items]}}

Usage:
  python canyon_final_v9_step99_news_aggregator.py
  python canyon_final_v9_step99_news_aggregator.py --top 30 --max-items 5
  python canyon_final_v9_step99_news_aggregator.py --force
  python canyon_final_v9_step99_news_aggregator.py --top 50 --workers 12 --force
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CACHE_TTL_HOURS = 6          # skip fetch if stock_news.json is younger than this

OUT_JSON = ROOT / "stock_news.json"

F_DAILY_PICKS   = ROOT / "daily_picks.csv"
F_ALPHA_SCORES  = ROOT / "alpha_scores.csv"
F_REGIME_SCORES = ROOT / "regime_ml_scores.csv"

# ─────────────────────────────────────────────────────────────────────────────
# [1/3] Universe loader
# ─────────────────────────────────────────────────────────────────────────────

def load_universe(top_n: int) -> list[str]:
    """
    Load top N tickers from the first available source:
      1. daily_picks.csv   — alpha_score descending
      2. alpha_scores.csv  — alpha_rank ascending
      3. regime_ml_scores.csv — predicted_score descending
    Returns a deduplicated list of uppercase ticker strings.
    """
    def _try_daily_picks() -> list[str] | None:
        if not F_DAILY_PICKS.exists():
            return None
        try:
            df = pd.read_csv(F_DAILY_PICKS)
            if df.empty or "ticker" not in df.columns:
                return None
            if "alpha_score" in df.columns:
                df = df.sort_values("alpha_score", ascending=False)
            tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
            return tickers[:top_n] if tickers else None
        except Exception as exc:
            print(f"  [WARN] Could not load daily_picks.csv: {exc}")
            return None

    def _try_alpha_scores() -> list[str] | None:
        if not F_ALPHA_SCORES.exists():
            return None
        try:
            df = pd.read_csv(F_ALPHA_SCORES)
            if df.empty or "ticker" not in df.columns:
                return None
            if "alpha_rank" in df.columns:
                df = df.sort_values("alpha_rank", ascending=True)
            elif "alpha_score" in df.columns:
                df = df.sort_values("alpha_score", ascending=False)
            tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
            return tickers[:top_n] if tickers else None
        except Exception as exc:
            print(f"  [WARN] Could not load alpha_scores.csv: {exc}")
            return None

    def _try_regime_scores() -> list[str] | None:
        if not F_REGIME_SCORES.exists():
            return None
        try:
            df = pd.read_csv(F_REGIME_SCORES)
            if df.empty or "ticker" not in df.columns:
                return None
            if "predicted_score" in df.columns:
                df = df.sort_values("predicted_score", ascending=False)
            tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
            return tickers[:top_n] if tickers else None
        except Exception as exc:
            print(f"  [WARN] Could not load regime_ml_scores.csv: {exc}")
            return None

    for loader, label in [
        (_try_daily_picks,   "daily_picks.csv"),
        (_try_alpha_scores,  "alpha_scores.csv"),
        (_try_regime_scores, "regime_ml_scores.csv"),
    ]:
        result = loader()
        if result:
            print(f"  [OK]  Universe loaded from {label}: {len(result)} tickers")
            return result

    print("  [ERROR] No universe source found. Returning empty list.")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# [2/3] News fetch
# ─────────────────────────────────────────────────────────────────────────────

def _published_from_epoch(value: Any) -> tuple[str, int]:
    """Return (YYYY-MM-DD, unix seconds) from a yfinance timestamp."""
    try:
        ts = int(float(value))
        if ts > 10_000_000_000:  # milliseconds
            ts = int(ts / 1000)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"), ts
    except Exception:
        return "", 0


def _published_from_iso(value: Any) -> tuple[str, int]:
    """Return (YYYY-MM-DD, unix seconds) from an ISO date string."""
    if not value:
        return "", 0
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d"), int(dt.timestamp())
    except Exception:
        return "", 0


def _read_nested_url(value: Any) -> str:
    """yfinance now returns URL payloads as {url: ...}; older schema used strings."""
    if isinstance(value, dict):
        return str(value.get("url", "") or "").strip()
    return str(value or "").strip()


NEWS_IMPACT_RULES: list[dict[str, Any]] = [
    {
        "tone": "positive",
        "weight": 4,
        "category": "Earnings beat or guidance raise",
        "terms": [
            "beats earnings", "beat earnings", "beats estimates", "beat estimates",
            "tops estimates", "better than expected", "raises guidance",
            "raised guidance", "guidance raise", "raises outlook", "upbeat outlook",
        ],
        "logic": "Results or forward guidance improved versus expectations.",
        "layers": ["L4 Fundamentals", "L5 Events"],
    },
    {
        "tone": "positive",
        "weight": 3,
        "category": "Analyst upgrade or higher target",
        "terms": [
            "upgrade", "upgraded", "outperform", "buy rating", "price target raised",
            "raises price target", "initiates buy", "upgrades", "strong buy",
            "target raised", "raises target", "higher target",
        ],
        "logic": "Outside analysts are improving the market narrative or target price.",
        "layers": ["L5 Events", "L6 Price"],
    },
    {
        "tone": "positive",
        "weight": 2,
        "category": "Strategic deal, product, or partnership",
        "terms": [
            "partnership", "partners with", "strategic partnership", "major partnership",
            "customer contract", "supply contract", "strategic deal", "product launch",
            "launches", "launch",
            "acquisition", "acquires", "merger", "approval", "approved",
        ],
        "logic": "The headline may add a catalyst, revenue path, or strategic optionality.",
        "layers": ["L4 Fundamentals", "L5 Events"],
    },
    {
        "tone": "positive",
        "weight": 2,
        "category": "Capital return or demand strength",
        "terms": [
            "buyback", "repurchase", "dividend increase", "raises dividend",
            "raised dividend", "raised dividends", "raises dividends",
            "buying back shares", "share buyback", "stock buyback",
            "share repurchase",
            "record revenue", "strong demand", "growth accelerates", "margin expands",
            "profit rises", "surges", "jumps", "rallies",
        ],
        "logic": "The headline points to stronger cash return, demand, margin, or price momentum.",
        "layers": ["L4 Fundamentals", "L6 Price"],
    },
    {
        "tone": "negative",
        "weight": 5,
        "category": "Earnings miss or guidance cut",
        "terms": [
            "misses earnings", "missed earnings", "misses estimates", "missed estimates",
            "below estimates", "worse than expected", "cuts guidance", "cut guidance",
            "guidance cut", "lowers outlook", "warning", "profit warning",
        ],
        "logic": "Results or guidance disappointed versus expectations.",
        "layers": ["L4 Fundamentals", "L5 Events", "L8 Risk"],
    },
    {
        "tone": "negative",
        "weight": 3,
        "category": "Analyst downgrade or weaker target",
        "terms": [
            "downgrade", "downgraded", "underperform", "sell rating",
            "price target cut", "cuts price target", "lowers price target",
            "target lower", "target lowered", "lowers target", "lower target",
            "revises target lower", "cuts target",
        ],
        "logic": "Outside analysts are reducing confidence or expected upside.",
        "layers": ["L5 Events", "L6 Price"],
    },
    {
        "tone": "negative",
        "weight": 4,
        "category": "Legal, regulatory, or trust risk",
        "terms": [
            "lawsuit", "sued", "investigation", "probe", "sec probe", "doj",
            "ftc", "fraud", "accounting issue", "short seller", "short-seller",
            "regulatory risk",
        ],
        "logic": "Legal or regulatory pressure can change valuation, timing, and position size.",
        "layers": ["L5 Events", "L8 Risk"],
    },
    {
        "tone": "negative",
        "weight": 4,
        "category": "Operational or balance-sheet stress",
        "terms": [
            "breach", "hack", "recall", "bankruptcy", "liquidity crisis",
            "debt pressure", "layoffs", "layoff", "resigns", "plunges",
            "slumps", "falls sharply", "margin pressure", "weak demand",
        ],
        "logic": "The headline may signal business stress, execution risk, or liquidity risk.",
        "layers": ["L4 Fundamentals", "L5 Events", "L8 Risk"],
    },
]


def _term_hits(text: str, terms: list[str]) -> list[str]:
    """Boundary-aware keyword matching to avoid false hits like 'miss' in 'Mission'."""
    low = text.lower()
    hits = []
    for term in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
        if re.search(pattern, low):
            hits.append(term)
    return hits


def classify_news_impact(title: str, summary: str = "") -> dict[str, Any]:
    """Explain whether one headline is likely positive, negative, mixed, or neutral."""
    text = f"{title} {summary}".strip()
    score = 0
    bull_points: list[str] = []
    bear_points: list[str] = []
    matched_terms: list[str] = []
    layers: list[str] = []

    for rule in NEWS_IMPACT_RULES:
        hits = _term_hits(text, list(rule["terms"]))
        if not hits:
            continue

        matched_terms.extend(hits[:4])
        layers.extend(list(rule.get("layers", [])))
        point = f"{rule['category']}: {rule['logic']}"
        if rule["tone"] == "positive":
            score += int(rule["weight"])
            bull_points.append(point)
        else:
            score -= int(rule["weight"])
            bear_points.append(point)

    if score >= 2 and not bear_points:
        tone = "POSITIVE"
    elif score <= -2 and not bull_points:
        tone = "NEGATIVE"
    elif bull_points and bear_points:
        tone = "MIXED"
    elif score >= 2:
        tone = "POSITIVE"
    elif score <= -2:
        tone = "NEGATIVE"
    else:
        tone = "NEUTRAL"

    if tone == "POSITIVE":
        logic = "Bullish read: " + " | ".join(bull_points[:2])
        action_hint = "Potential catalyst. Confirm price, volume, risk gate, and position size before any paper action."
    elif tone == "NEGATIVE":
        logic = "Bearish read: " + " | ".join(bear_points[:2])
        action_hint = "Potential risk. Check gap risk, stop level, exposure, and event calendar before adding."
    elif tone == "MIXED":
        logic = "Mixed read: " + " | ".join((bull_points + bear_points)[:3])
        action_hint = "Manual review required. Do not upgrade or downgrade from the headline alone."
    else:
        logic = "Neutral read: no clear directional impact from the headline. Treat as context only."
        action_hint = "Context only. Wait for price, volume, or risk confirmation."

    return {
        "market_tone": tone,
        "impact_score": score,
        "bullish_reasons": bull_points,
        "bearish_reasons": bear_points,
        "news_logic": logic,
        "action_hint": action_hint,
        "matched_terms": sorted(set(matched_terms)),
        "affected_layers": sorted(set(layers)) if layers else ["L5 Events"],
    }


def _clean_news_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    Normalize both old and current yfinance news schemas.

    Old schema:
      {type, title, publisher, link, providerPublishTime}

    Current schema:
      {id, content: {contentType, title, summary, pubDate, provider,
                     canonicalUrl, clickThroughUrl}}
    """
    if not isinstance(item, dict):
        return None

    content = item.get("content")
    if not isinstance(content, dict):
        content = item

    news_type = str(
        content.get("contentType")
        or item.get("type")
        or content.get("type")
        or "STORY"
    ).upper()
    if news_type != "STORY":
        return None

    title = str(content.get("title") or item.get("title") or "").strip()
    summary = str(
        content.get("summary")
        or content.get("description")
        or item.get("summary")
        or ""
    ).strip()

    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = str(
            provider.get("displayName")
            or provider.get("name")
            or provider.get("source")
            or ""
        ).strip()
    else:
        publisher = str(item.get("publisher") or "").strip()

    link = (
        _read_nested_url(content.get("canonicalUrl"))
        or _read_nested_url(content.get("clickThroughUrl"))
        or _read_nested_url(item.get("link"))
        or _read_nested_url(item.get("url"))
    )

    published, published_ts = _published_from_epoch(item.get("providerPublishTime"))
    if not published_ts:
        published, published_ts = _published_from_iso(
            content.get("pubDate") or content.get("displayTime") or item.get("pubDate")
        )

    if not title and not link:
        return None

    impact = classify_news_impact(title, summary)

    return {
        "title":        title,
        "publisher":    publisher,
        "link":         link,
        "published":    published,
        "published_ts": published_ts,
        "type":         news_type,
        "summary":      summary,
        "raw_id":       str(item.get("id") or content.get("id") or "").strip(),
        **impact,
    }


def _fetch_news_one(ticker: str, max_items: int = 8) -> list[dict[str, Any]]:
    """
    Fetch news for one ticker via yfinance.
    Returns a list of cleaned news dicts.
    Returns [] on any error.

    Fields kept per item:
      title, publisher, link, published (YYYY-MM-DD), published_ts, type, summary

    Only STORY type items are kept (VIDEO and other types are filtered out).
    """
    try:
        raw: list[dict] = yf.Ticker(ticker).news or []
    except Exception:
        return []

    cleaned: list[dict[str, Any]] = []
    for item in raw:
        try:
            news_item = _clean_news_item(item)
            if not news_item:
                continue

            cleaned.append(news_item)

            if len(cleaned) >= max_items:
                break

        except Exception:
            continue

    return cleaned


def fetch_all_news(
    tickers: list[str],
    max_workers: int = 8,
    max_items: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch news for all tickers in parallel using ThreadPoolExecutor.
    Prints progress every 10 tickers.
    Throttles submissions with a 0.1 s sleep to avoid rate limiting.
    Returns {ticker: [news_items]}.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    total = len(tickers)
    completed = 0

    future_to_ticker: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker in tickers:
            future = executor.submit(_fetch_news_one, ticker, max_items)
            future_to_ticker[future] = ticker
            time.sleep(0.1)   # light throttle between submissions

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                news_items = future.result()
            except Exception:
                news_items = []

            results[ticker] = news_items

            if completed % 10 == 0 or completed == total:
                pct = completed / total * 100
                print(f"  [FETCH] {completed}/{total} ({pct:.0f}%) — "
                      f"latest: {ticker} ({len(news_items)} items)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# [3/3] Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    top_n: int = 50,
    max_items: int = 8,
    force: bool = False,
    max_workers: int = 8,
) -> int:
    """
    Main entry:
      1. Check cache TTL — skip if stock_news.json is fresh and not --force
      2. Load universe top N tickers
      3. Fetch all news in parallel
      4. Build output dict
      5. Write stock_news.json
      6. Print summary
    Returns 0 on success (always — news is supplementary, never fails the pipeline).
    """
    print()
    print("=" * 60)
    print("  Canyon v9 — Step 99: News Aggregator")
    print("=" * 60)

    # ── 1. Cache TTL check ───────────────────────────────────────────────────
    print()
    print("[1/3] Cache check")
    if OUT_JSON.exists() and not force:
        try:
            mtime = datetime.fromtimestamp(OUT_JSON.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours < CACHE_TTL_HOURS:
                print(f"  [SKIP] {OUT_JSON.name} is {age_hours:.1f}h old "
                      f"(TTL={CACHE_TTL_HOURS}h). Use --force to override.")
                print()
                print("  Done. (cache hit, no fetch performed)")
                print("=" * 60)
                return 0
            else:
                print(f"  [OK]  Cache is {age_hours:.1f}h old — refreshing.")
        except Exception:
            print("  [OK]  Cache age check failed — will fetch.")
    elif force and OUT_JSON.exists():
        print("  [OK]  --force flag set — ignoring cache.")
    else:
        print("  [OK]  No cache found — will fetch.")

    # ── 2. Load universe ─────────────────────────────────────────────────────
    print()
    print("[2/3] Loading universe")
    tickers = load_universe(top_n)
    if not tickers:
        print("  [ERROR] No tickers to fetch. Exiting with 0 (non-fatal).")
        return 0
    print(f"  Fetching news for {len(tickers)} tickers "
          f"(top_n={top_n}, max_items={max_items}, workers={max_workers})")

    # ── 3. Fetch ─────────────────────────────────────────────────────────────
    print()
    print("[3/3] Fetching news")
    t0 = time.time()
    news_map = fetch_all_news(tickers, max_workers=max_workers, max_items=max_items)
    elapsed = time.time() - t0
    print(f"  Fetch complete in {elapsed:.1f}s")

    # ── 4 & 5. Build output and write ────────────────────────────────────────
    total_items = sum(len(v) for v in news_map.values())
    avg_items   = total_items / len(news_map) if news_map else 0.0
    tickers_with_news = sum(1 for v in news_map.values() if v)

    output: dict[str, Any] = {
        "updated":         datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "tickers_fetched": len(news_map),
        "news":            news_map,
    }

    try:
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        print(f"  [OK]  Written → {OUT_JSON.name}")
    except Exception as exc:
        print(f"  [ERROR] Could not write {OUT_JSON.name}: {exc}")
        return 0   # still exit 0 — supplementary step

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"  Tickers fetched : {len(news_map)}")
    print(f"  Tickers w/ news : {tickers_with_news}")
    print(f"  Total items     : {total_items}")
    print(f"  Avg per ticker  : {avg_items:.1f}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print(f"  Output          : {OUT_JSON.name}")
    print("─" * 60)
    print()

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 99 — News Aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--top", type=int, default=50, metavar="N",
        help="Fetch news for top N tickers by alpha score (default: 50)",
    )
    parser.add_argument(
        "--max-items", type=int, default=8, metavar="N",
        help="Max news items per ticker (default: 8)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=f"Ignore cache TTL ({CACHE_TTL_HOURS}h) and always refetch",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="N",
        help="ThreadPoolExecutor max_workers (default: 8)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(
        run(
            top_n=args.top,
            max_items=args.max_items,
            force=args.force,
            max_workers=args.workers,
        )
    )
