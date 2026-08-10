#!/usr/bin/env python3
"""
Step — Alpha Vantage market news sentiment (free-tier friendly).

WHY THIS ONE ENDPOINT: Alpha Vantage's free tier is rate-limited (~25 calls/day),
so it can't cover a 500-name universe. Its prices/FX/fundamentals are already
covered by yfinance + FRED + EDGAR. The genuinely non-redundant thing it offers
free is NEWS_SENTIMENT — a curated news feed with per-article relevance and a
market-wide sentiment score. We spend just 1–2 calls/day on that.

GRACEFUL: if ALPHAVANTAGE_API_KEY is not set, this writes an honest
"not_enabled" marker and exits 0 — it never fabricates data.

Output: alphavantage_news_sentiment.json
"""
from __future__ import annotations
import json
import os
import socket
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
OUT  = ROOT / "alphavantage_news_sentiment.json"
socket.setdefaulttimeout(30)  # never hang the pipeline


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v and k not in os.environ:
            os.environ[k] = v


def _write(payload: dict) -> None:
    OUT.write_text(json.dumps(payload, indent=2))


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "canyon-quant/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    _load_env()
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    if not key:
        _write({
            "enabled": False,
            "as_of": date.today().isoformat(),
            "reason": "ALPHAVANTAGE_API_KEY not set in .env — add it to activate. "
                      "No data is being shown or faked.",
        })
        print("[alphavantage] no key — wrote not_enabled marker, exiting 0")
        return 0

    base = "https://www.alphavantage.co/query"
    # 1 call: market-wide financial-markets news sentiment (most recent 50).
    url = (f"{base}?function=NEWS_SENTIMENT&topics=financial_markets"
           f"&sort=LATEST&limit=50&apikey={key}")
    try:
        data = _get(url)
    except Exception as e:  # network/timeout — honest failure, no fake data
        _write({"enabled": True, "ok": False, "as_of": date.today().isoformat(),
                "error": f"fetch failed: {e}"})
        print(f"[alphavantage] fetch failed: {e}")
        return 0

    # Rate-limit / error notes come back as plain fields, not a feed.
    if "feed" not in data:
        note = data.get("Note") or data.get("Information") or data.get("Error Message") or str(data)[:200]
        _write({"enabled": True, "ok": False, "as_of": date.today().isoformat(),
                "error": f"no feed returned: {note}"})
        print(f"[alphavantage] no feed: {note}")
        return 0

    feed = data.get("feed", [])
    # Aggregate a clean, honest summary.
    scores = [float(a["overall_sentiment_score"]) for a in feed
              if a.get("overall_sentiment_score") not in (None, "")]
    avg = round(sum(scores) / len(scores), 4) if scores else None
    label = ("Bullish" if (avg or 0) > 0.15 else
             "Somewhat-Bullish" if (avg or 0) > 0.05 else
             "Bearish" if (avg or 0) < -0.15 else
             "Somewhat-Bearish" if (avg or 0) < -0.05 else "Neutral")
    top = []
    for a in feed[:12]:
        top.append({
            "title": a.get("title", "")[:160],
            "source": a.get("source", ""),
            "time": a.get("time_published", ""),
            "url": a.get("url", ""),
            "sentiment": a.get("overall_sentiment_label", ""),
            "score": a.get("overall_sentiment_score", ""),
        })
    _write({
        "enabled": True, "ok": True,
        "as_of": date.today().isoformat(),
        "source": "Alpha Vantage NEWS_SENTIMENT (topics=financial_markets)",
        "n_articles": len(feed),
        "avg_sentiment": avg,
        "market_label": label,
        "top_articles": top,
    })
    print(f"[alphavantage] ok — {len(feed)} articles, avg={avg} ({label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
