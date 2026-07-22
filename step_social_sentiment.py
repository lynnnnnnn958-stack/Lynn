#!/usr/bin/env python3
"""
Canyon — Social Sentiment (StockTwits + Reddit)
=================================================
Fetches real-time social sentiment for top alpha tickers.

StockTwits: No authentication required for public symbol streams.
  API: https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json
  Returns last 30 messages with optional self-labeled bullish/bearish.

Reddit (optional — requires REDDIT_CLIENT_ID env var):
  Searches r/wallstreetbets and r/stocks for ticker mentions in last 24h.
  Register a free script app at: reddit.com/prefs/apps

Signals produced per ticker:
  st_bull_pct       — StockTwits % bullish (0-1)
  st_bear_pct       — StockTwits % bearish (0-1)
  st_volume         — message count (retail attention proxy)
  st_net_sentiment  — bull_pct - bear_pct  (-1 to +1)
  reddit_mentions   — post count in r/wsb + r/stocks (24h)
  reddit_sentiment  — avg post sentiment from title FinBERT-lite scoring

Output: social_sentiment.csv

Usage:
  .venv/bin/python step_social_sentiment.py
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent
OUT    = ROOT / "social_sentiment.csv"
SLEEP  = 0.8   # polite delay between StockTwits calls

UA = "CanyonQuant/1.0 (research; lynnnnnnn958@gmail.com)"

# Simple positive/negative word lists for Reddit title scoring (no ML needed)
POS_WORDS = {"bull", "bullish", "long", "calls", "buy", "moon", "squeeze",
             "beat", "strong", "growth", "breakout", "upgrade", "record"}
NEG_WORDS = {"bear", "bearish", "short", "puts", "sell", "crash", "dump",
             "miss", "weak", "decline", "breakdown", "downgrade", "warning"}


# ── StockTwits ────────────────────────────────────────────────────────────────

def _stocktwits(ticker: str) -> dict:
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())

        msgs   = data.get("messages", [])
        total  = len(msgs)
        bull   = sum(1 for m in msgs
                     if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish")
        bear   = sum(1 for m in msgs
                     if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish")
        labeled = bull + bear

        bull_pct = bull / labeled if labeled > 0 else 0.5
        bear_pct = bear / labeled if labeled > 0 else 0.5

        return {
            "st_bull_pct":      round(bull_pct, 4),
            "st_bear_pct":      round(bear_pct, 4),
            "st_volume":        total,
            "st_labeled":       labeled,
            "st_net_sentiment": round(bull_pct - bear_pct, 4),
            "st_ok":            True,
        }
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"    {ticker}: StockTwits rate-limited — sleeping 30s")
            time.sleep(30)
        return {"st_ok": False}
    except Exception as e:
        return {"st_ok": False}


# ── Reddit (optional) ─────────────────────────────────────────────────────────

_reddit_token: str | None = None
_token_expires: float = 0.0

def _reddit_auth() -> str | None:
    """Get Reddit OAuth token. Returns None if credentials not set."""
    global _reddit_token, _token_expires
    cid = os.environ.get("REDDIT_CLIENT_ID", "")
    sec = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not cid or not sec:
        return None
    if _reddit_token and time.time() < _token_expires - 60:
        return _reddit_token
    try:
        import base64
        creds = base64.b64encode(f"{cid}:{sec}".encode()).decode()
        req = urllib.request.Request(
            "https://www.reddit.com/api/v1/access_token",
            data=b"grant_type=client_credentials",
            headers={
                "Authorization": f"Basic {creds}",
                "User-Agent": UA,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        _reddit_token   = resp.get("access_token")
        _token_expires  = time.time() + resp.get("expires_in", 3600)
        return _reddit_token
    except Exception:
        return None


def _reddit_mentions(ticker: str) -> dict:
    token = _reddit_auth()
    if not token:
        return {"reddit_mentions": None, "reddit_sentiment": None}

    subreddits = ["wallstreetbets", "stocks", "investing"]
    total_mentions = 0
    sentiment_scores = []

    for sub in subreddits:
        try:
            url = (f"https://oauth.reddit.com/r/{sub}/search.json"
                   f"?q=${ticker}&sort=new&limit=25&t=day&restrict_sr=1")
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": UA,
            })
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            posts = data.get("data", {}).get("children", [])
            for p in posts:
                title = p["data"].get("title", "").lower()
                if ticker.lower() in title or f"${ticker.lower()}" in title:
                    total_mentions += 1
                    words = set(re.findall(r"\b\w+\b", title))
                    pos = len(words & POS_WORDS)
                    neg = len(words & NEG_WORDS)
                    if pos + neg > 0:
                        sentiment_scores.append((pos - neg) / (pos + neg))
        except Exception:
            pass

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    return {
        "reddit_mentions":  total_mentions,
        "reddit_sentiment": round(avg_sentiment, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  Canyon Social Sentiment — {date.today()}")
    has_reddit = bool(os.environ.get("REDDIT_CLIENT_ID", ""))
    print(f"  StockTwits: enabled (no key needed)")
    print(f"  Reddit:     {'enabled' if has_reddit else 'disabled (set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET)'}")
    print("=" * 60)

    # Load top alpha tickers
    scores_path = ROOT / "alpha_scores.csv"
    if not scores_path.exists():
        print("ERROR: alpha_scores.csv not found")
        return
    scores    = pd.read_csv(scores_path)
    score_col = next((c for c in ["alpha_score", "score"] if c in scores.columns), None)
    max_t = int(os.environ.get("ST_MAX_TICKERS", "40"))
    if score_col:
        tickers = scores.sort_values(score_col, ascending=False)["ticker"].tolist()[:max_t]
    else:
        tickers = scores["ticker"].tolist()[:max_t]

    # Fallback: also include watchlist
    wl_path = ROOT / "watchlist.json"
    if wl_path.exists():
        try:
            wl = json.loads(wl_path.read_text())
            extra = [t for t in wl.get("tickers", []) if t not in tickers]
            tickers = tickers + extra[:10]
        except Exception:
            pass

    results = []
    for i, tkr in enumerate(tickers, 1):
        print(f"  [{i:3d}/{len(tickers)}] {tkr} … ", end="", flush=True)
        row = {"ticker": tkr, "as_of": date.today().isoformat()}
        try:
            st = _stocktwits(tkr)
            row.update(st)
            if has_reddit:
                rd = _reddit_mentions(tkr)
                row.update(rd)
            net = row.get("st_net_sentiment", 0) or 0
            vol = row.get("st_volume", 0) or 0
            print(f"net={net:+.2f} vol={vol}")
        except Exception as e:
            print(f"error: {e}")
        results.append(row)
        time.sleep(SLEEP)

    if results:
        df = pd.DataFrame(results)
        # If previous file exists, replace today's rows
        if OUT.exists():
            old = pd.read_csv(OUT)
            today_str = date.today().isoformat()
            old = old[old.get("as_of", pd.Series()) != today_str] if "as_of" in old.columns else old
            old = old[~old["ticker"].isin(df["ticker"].tolist())]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(OUT, index=False)
        print(f"\n  {len(results)} tickers saved → {OUT.name}")

        # Print top 5 most bullish
        today_df = pd.read_csv(OUT)
        if "st_net_sentiment" in today_df.columns:
            top = today_df.nlargest(5, "st_net_sentiment")[["ticker", "st_net_sentiment", "st_volume"]]
            print("\n  Top 5 net-bullish (StockTwits):")
            for _, r in top.iterrows():
                print(f"    {r['ticker']:<8} net={r['st_net_sentiment']:+.2f}  vol={int(r['st_volume'])}")


if __name__ == "__main__":
    main()
