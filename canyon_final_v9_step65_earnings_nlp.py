"""
Canyon v9  Step 65 — Earnings NLP Scorer
==========================================
Three-tier earnings sentiment analysis — works offline, upgrades with API keys:

  Tier 1 (always)  — yfinance news headlines + FinBERT keyword scoring
  Tier 2 (optional) — OpenAI GPT-4o-mini structured analysis per ticker
  Tier 3 (optional) — Earnings calendar from yfinance + surprise analysis

Scoring model (offline, no key needed):
  POSITIVE_WORDS: beat, exceed, record, strong, growth, raised, guidance, outperform …
  NEGATIVE_WORDS: miss, disappoint, below, cut, reduced, guidance, loss, impairment …
  UNCERTAIN_WORDS: outlook, monitor, environment, uncertain, headwind, cautious …
  Score = (positive_hits − negative_hits × 1.5) / (total_hits + 1)  ∈ [−1, +1]

GPT scoring (if OPENAI_API_KEY set):
  Structured prompt → JSON with: sentiment, guidance_tone, key_themes, beat_miss,
  forward_score (−1 to +1), analyst_reaction

Outputs:
  earnings_nlp_scores.csv        — per-ticker sentiment, score, themes, tier used
  earnings_calendar.csv          — upcoming earnings dates (next 30 days)
  earnings_nlp_report.md         — full markdown report

Usage:
  python canyon_final_v9_step65_earnings_nlp.py
  python canyon_final_v9_step65_earnings_nlp.py --tickers AAPL MSFT NVDA
  python canyon_final_v9_step65_earnings_nlp.py --gpt         # enable GPT tier
  python canyon_final_v9_step65_earnings_nlp.py --days 14     # news lookback window
  OPENAI_API_KEY=sk-... python canyon_final_v9_step65_earnings_nlp.py --gpt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","MU","JPM",
    "XOM","CVX","JNJ","WMT","KO","PEP","MRK","ABBV","UNH","LLY",
    "SPY","QQQ",
]
NEWS_LOOKBACK_DAYS = 30   # how far back to look for news


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — FinBERT-inspired keyword lists
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    # revenue / earnings beats
    "beat", "beats", "exceeded", "surpassed", "topped", "outperformed",
    "record", "record-high", "all-time",
    # growth
    "growth", "grew", "accelerated", "accelerating", "momentum",
    "expansion", "expanding", "gained",
    # guidance
    "raised", "raises", "increased", "above", "raised guidance",
    "upgraded", "upgrade", "strong", "strength", "robust",
    # sentiment
    "positive", "optimistic", "confident", "encouraging", "favorable",
    "opportunity", "opportunities", "upside",
    # earnings
    "profit", "profitable", "margin improvement", "margin expansion",
    "cash flow", "free cash", "buyback", "dividend increase",
}

NEGATIVE_WORDS = {
    # misses
    "miss", "missed", "below", "fell short", "disappointed", "disappointing",
    "weakness", "weak", "declined",
    # guidance cuts
    "cut", "cuts", "reduced", "lowered", "lowering", "warned", "warning",
    "cautious", "concern", "concerns",
    # macro headwinds
    "headwind", "headwinds", "pressure", "pressures", "challenging",
    "uncertainty", "uncertain", "volatile", "volatility",
    # losses
    "loss", "losses", "impairment", "write-down", "writedown",
    "charges", "restructuring", "layoffs", "downturn",
    # negative outlook
    "negative", "pessimistic", "disappointed", "downgraded", "downgrade",
}

UNCERTAIN_WORDS = {
    "monitor", "watching", "await", "depends", "unclear",
    "mixed", "cautiously", "may", "could", "might",
    "if conditions", "subject to",
}


def keyword_score(text: str) -> dict:
    """Score a piece of text using FinBERT-inspired keyword lists."""
    if not text:
        return {"score": 0.0, "positive": 0, "negative": 0, "uncertain": 0,
                "pos_words": [], "neg_words": []}

    lower = text.lower()
    pos_hits = [w for w in POSITIVE_WORDS if w in lower]
    neg_hits = [w for w in NEGATIVE_WORDS if w in lower]
    unc_hits = [w for w in UNCERTAIN_WORDS if w in lower]

    n_pos = len(pos_hits)
    n_neg = len(neg_hits)
    n_unc = len(unc_hits)

    # Weighted score: negative words penalised 1.5×
    raw = (n_pos - n_neg * 1.5) / (n_pos + n_neg + n_unc + 1.0)
    score = float(np.clip(raw, -1.0, 1.0))

    return {
        "score":     round(score, 4),
        "positive":  n_pos,
        "negative":  n_neg,
        "uncertain": n_unc,
        "pos_words": pos_hits[:5],
        "neg_words": neg_hits[:5],
    }


def score_to_label(score: float) -> str:
    if score >  0.20: return "BULLISH"
    if score >  0.05: return "MILDLY_BULLISH"
    if score > -0.05: return "NEUTRAL"
    if score > -0.20: return "MILDLY_BEARISH"
    return "BEARISH"


def score_to_color(score: float) -> str:
    if score >  0.20: return "green"
    if score >  0.05: return "lightgreen"
    if score > -0.05: return "gray"
    if score > -0.20: return "orange"
    return "red"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — yfinance news loader
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_yfinance(ticker: str, max_items: int = 20) -> list[dict]:
    """Return list of news dicts {title, summary, publisher, published_at}.
    Handles both old yfinance (<0.2) and new nested format (>=0.2).
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news or []
        results = []
        for item in news[:max_items]:
            # New yfinance format: item = {"id": ..., "content": {...}}
            if isinstance(item, dict) and "content" in item:
                c = item["content"]
                title   = c.get("title", "")
                summary = c.get("summary", "") or c.get("description", "")
                pub_str = c.get("pubDate", "") or c.get("displayTime", "")
                try:
                    published = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    published = datetime.min
                publisher = c.get("provider", {}).get("displayName", "")
                link      = (c.get("canonicalUrl") or {}).get("url", "")
            else:
                # Old format
                title     = item.get("title", "")
                summary   = ""
                ts        = item.get("providerPublishTime", 0)
                published = datetime.fromtimestamp(ts) if ts else datetime.min
                publisher = item.get("publisher", "")
                link      = item.get("link", "")

            if title:
                results.append({
                    "title":        title,
                    "summary":      summary,
                    "publisher":    publisher,
                    "link":         link,
                    "published_at": published,
                })
        return results
    except Exception:
        return []


def fetch_earnings_calendar(ticker: str) -> dict:
    """Return next earnings date and EPS surprise if available."""
    try:
        import yfinance as yf
        t   = yf.Ticker(ticker)
        cal = t.calendar
        info = t.info or {}

        result = {
            "next_earnings_date": None,
            "eps_estimate":      None,
            "eps_actual":        None,
            "eps_surprise_pct":  None,
            "revenue_estimate":  None,
        }

        if cal is not None:
            if isinstance(cal, dict):
                eds = cal.get("Earnings Date", [])
                if hasattr(eds, '__iter__') and not isinstance(eds, str):
                    dates = [d for d in eds if d]
                    if dates:
                        result["next_earnings_date"] = str(dates[0])
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.columns:
                    dates = cal["Earnings Date"].dropna()
                    if not dates.empty:
                        result["next_earnings_date"] = str(dates.iloc[0])
                elif len(cal.index) > 0:
                    result["next_earnings_date"] = str(cal.index[0])

        result["eps_estimate"] = info.get("epsCurrentYear") or info.get("forwardEps")
        result["eps_actual"]   = info.get("trailingEps")

        if result["eps_estimate"] and result["eps_actual"]:
            est = float(result["eps_estimate"])
            act = float(result["eps_actual"])
            if abs(est) > 1e-6:
                result["eps_surprise_pct"] = round((act - est) / abs(est) * 100, 2)

        return result
    except Exception:
        return {"next_earnings_date": None, "eps_estimate": None,
                "eps_actual": None, "eps_surprise_pct": None, "revenue_estimate": None}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — OpenAI GPT analysis
# ─────────────────────────────────────────────────────────────────────────────

GPT_SYSTEM = """You are a senior equity research analyst at a top-tier investment bank.
Analyze the provided news headlines about a stock and return a structured JSON assessment.

Your JSON must have exactly these fields:
{
  "sentiment": "BULLISH|MILDLY_BULLISH|NEUTRAL|MILDLY_BEARISH|BEARISH",
  "guidance_tone": "RAISED|MAINTAINED|LOWERED|UNKNOWN",
  "beat_miss": "BEAT|IN_LINE|MISS|NO_DATA",
  "forward_score": <float from -1.0 to +1.0>,
  "key_themes": ["theme1", "theme2", "theme3"],
  "risks": ["risk1", "risk2"],
  "analyst_action": "BUY|HOLD|SELL|NEUTRAL",
  "reasoning": "<2-3 sentence summary>"
}

Be conservative. Favor NEUTRAL when evidence is mixed. Do not hallucinate specifics not in the headlines."""


def gpt_score(ticker: str, headlines: list[str]) -> Optional[dict]:
    """Call GPT-4o-mini to score earnings sentiment. Returns parsed dict or None."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    headlines_text = "\n".join(f"- {h}" for h in headlines[:15])
    if not headlines_text.strip():
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GPT_SYSTEM},
                {"role": "user",
                 "content": f"Ticker: {ticker}\n\nRecent headlines:\n{headlines_text}"},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
            timeout=20,
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
        result["tier"] = "GPT-4o-mini"
        return result
    except Exception as e:
        print(f"      [GPT] Error for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_ticker(ticker: str, days_lookback: int, use_gpt: bool) -> dict:
    """
    Full pipeline for one ticker. Returns a flat dict with all fields.
    """
    result = {
        "ticker":           ticker,
        "as_of":            datetime.now().strftime("%Y-%m-%d"),
        "tier":             "keyword",
        "sentiment":        "NEUTRAL",
        "forward_score":    0.0,
        "guidance_tone":    "UNKNOWN",
        "beat_miss":        "NO_DATA",
        "key_themes":       "",
        "risks":            "",
        "analyst_action":   "NEUTRAL",
        "news_count":       0,
        "reasoning":        "",
        "next_earnings":    None,
        "eps_surprise_pct": None,
        "pos_words":        "",
        "neg_words":        "",
    }

    # ── fetch news ────────────────────────────────────────────────────────────
    news_items = fetch_news_yfinance(ticker, max_items=20)

    # Filter to lookback window
    cutoff = datetime.now() - timedelta(days=days_lookback)
    recent = [n for n in news_items if n.get("published_at", datetime.min) >= cutoff]
    if not recent:
        recent = news_items  # fallback: use all available news

    result["news_count"] = len(recent)
    # Use title + summary for richer keyword coverage
    headlines = [
        " ".join(filter(None, [n.get("title", ""), n.get("summary", "")]))
        for n in recent
    ]
    headlines = [h for h in headlines if h.strip()]

    # ── earnings calendar ─────────────────────────────────────────────────────
    cal = fetch_earnings_calendar(ticker)
    result["next_earnings"]    = cal.get("next_earnings_date")
    result["eps_surprise_pct"] = cal.get("eps_surprise_pct")

    if not headlines:
        result["reasoning"] = "No recent news found"
        return result

    # ── Tier 2: GPT ───────────────────────────────────────────────────────────
    if use_gpt and os.environ.get("OPENAI_API_KEY"):
        gpt_result = gpt_score(ticker, headlines)
        if gpt_result:
            result.update({
                "tier":           "GPT-4o-mini",
                "sentiment":      gpt_result.get("sentiment", "NEUTRAL"),
                "forward_score":  float(gpt_result.get("forward_score", 0.0)),
                "guidance_tone":  gpt_result.get("guidance_tone", "UNKNOWN"),
                "beat_miss":      gpt_result.get("beat_miss", "NO_DATA"),
                "key_themes":     " | ".join(gpt_result.get("key_themes", [])),
                "risks":          " | ".join(gpt_result.get("risks", [])),
                "analyst_action": gpt_result.get("analyst_action", "NEUTRAL"),
                "reasoning":      gpt_result.get("reasoning", ""),
            })
            return result

    # ── Tier 1: keyword scoring ───────────────────────────────────────────────
    combined_text = " ".join(headlines)
    kw = keyword_score(combined_text)
    score = kw["score"]

    # Earnings surprise bonus
    surprise = cal.get("eps_surprise_pct")
    if surprise is not None:
        surprise_adj = float(np.clip(surprise / 50.0, -0.3, 0.3))
        score = float(np.clip(score + surprise_adj, -1.0, 1.0))

    label = score_to_label(score)

    # Heuristic guidance tone from keywords
    all_lower = combined_text.lower()
    if any(w in all_lower for w in ["raised guidance", "raised its", "above guidance",
                                     "raised outlook", "increased outlook"]):
        guidance = "RAISED"
    elif any(w in all_lower for w in ["cut guidance", "lowered guidance", "reduced outlook",
                                       "below guidance", "lowered its"]):
        guidance = "LOWERED"
    elif any(w in all_lower for w in ["maintained", "reiterated", "in line", "on track"]):
        guidance = "MAINTAINED"
    else:
        guidance = "UNKNOWN"

    beat_miss = "NO_DATA"
    if any(w in all_lower for w in ["beat", "topped", "exceeded", "surpassed"]):
        beat_miss = "BEAT"
    elif any(w in all_lower for w in ["missed", "fell short", "below expectations"]):
        beat_miss = "MISS"
    elif any(w in all_lower for w in ["in line", "met expectations", "as expected"]):
        beat_miss = "IN_LINE"

    # Extract analyst action
    analyst_action = "NEUTRAL"
    if any(w in all_lower for w in ["upgrade", "upgraded", "overweight", "buy rating"]):
        analyst_action = "BUY"
    elif any(w in all_lower for w in ["downgrade", "downgraded", "underweight", "sell rating"]):
        analyst_action = "SELL"

    top_themes = []
    if "ai" in all_lower or "artificial intelligence" in all_lower: top_themes.append("AI")
    if "cloud" in all_lower: top_themes.append("Cloud")
    if "margin" in all_lower: top_themes.append("Margins")
    if "revenue" in all_lower: top_themes.append("Revenue")
    if "guidance" in all_lower: top_themes.append("Guidance")
    if "buyback" in all_lower or "repurchase" in all_lower: top_themes.append("Buyback")
    if "dividend" in all_lower: top_themes.append("Dividend")
    if "macro" in all_lower or "economy" in all_lower: top_themes.append("Macro")

    result.update({
        "tier":           "keyword",
        "sentiment":      label,
        "forward_score":  round(score, 4),
        "guidance_tone":  guidance,
        "beat_miss":      beat_miss,
        "key_themes":     " | ".join(top_themes[:4]),
        "analyst_action": analyst_action,
        "reasoning":      (f"{kw['positive']} positive signals, "
                           f"{kw['negative']} negative signals across "
                           f"{len(headlines)} headlines. "
                           f"Score={score:.3f}. "
                           + (f"EPS surprise: {surprise:+.1f}%." if surprise is not None else "")),
        "pos_words":      " | ".join(kw["pos_words"]),
        "neg_words":      " | ".join(kw["neg_words"]),
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(scores_df: pd.DataFrame, calendar_df: pd.DataFrame,
                 use_gpt: bool, ts: str) -> None:
    tier = "GPT-4o-mini + keyword" if use_gpt else "keyword-only (offline)"
    lines = [
        "# Canyon v9 — Earnings NLP Report (Step 65)",
        f"Generated: {ts}  |  Tier: {tier}",
        "",
        "## Sentiment Summary",
        "",
    ]
    if not scores_df.empty and "sentiment" in scores_df.columns:
        sent_counts = scores_df["sentiment"].value_counts()
        lines.append("| Sentiment | Count |")
        lines.append("|---|---|")
        for sent, cnt in sent_counts.items():
            lines.append(f"| {sent} | {cnt} |")

    lines += [
        "",
        "## Top Bullish Signals",
        "",
        "| Ticker | Score | Sentiment | Beat/Miss | Guidance | Themes |",
        "|---|---|---|---|---|---|",
    ]
    if not scores_df.empty:
        bullish = scores_df.sort_values("forward_score", ascending=False).head(8)
        for _, row in bullish.iterrows():
            lines.append(
                f"| {row.get('ticker','')} | {row.get('forward_score',0):+.3f} | "
                f"{row.get('sentiment','')} | {row.get('beat_miss','')} | "
                f"{row.get('guidance_tone','')} | {str(row.get('key_themes',''))[:40]} |"
            )

    lines += [
        "",
        "## Top Bearish Signals",
        "",
        "| Ticker | Score | Sentiment | Beat/Miss | Risks |",
        "|---|---|---|---|---|",
    ]
    if not scores_df.empty:
        bearish = scores_df.sort_values("forward_score", ascending=True).head(5)
        for _, row in bearish.iterrows():
            lines.append(
                f"| {row.get('ticker','')} | {row.get('forward_score',0):+.3f} | "
                f"{row.get('sentiment','')} | {row.get('beat_miss','')} | "
                f"{str(row.get('risks',''))[:40]} |"
            )

    if not calendar_df.empty:
        lines += [
            "",
            "## Upcoming Earnings Calendar",
            "",
            "| Ticker | Next Earnings | EPS Surprise % |",
            "|---|---|---|",
        ]
        for _, row in calendar_df.iterrows():
            lines.append(
                f"| {row.get('ticker','')} | {row.get('next_earnings','TBD')} | "
                f"{row.get('eps_surprise_pct', '—')} |"
            )

    lines += [
        "",
        "## Methodology",
        f"- **Tier used**: {tier}",
        "- **Keyword model**: FinBERT-inspired word lists — "
        f"{len(POSITIVE_WORDS)} positive / {len(NEGATIVE_WORDS)} negative / "
        f"{len(UNCERTAIN_WORDS)} uncertain words",
        "- **Score formula**: (positive_hits − negative_hits×1.5) / (total_hits+1)",
        "- **EPS surprise bonus**: ±30% adjustment based on trailing EPS surprise %",
        "- **GPT tier**: Available when OPENAI_API_KEY is set, uses gpt-4o-mini",
        "",
        "## Upgrade Path",
        "- Set `OPENAI_API_KEY` env variable and pass `--gpt` flag for structured analysis",
        "- Upgrade to `gpt-4o` for deeper reasoning (higher cost)",
        "- Integrate earnings call transcript APIs (Seekingalpha, Refinitiv) for full text",
    ]

    p = ROOT / "earnings_nlp_report.md"
    p.write_text("\n".join(lines))
    print(f"  [report] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 65 — Earnings NLP")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--days",    type=int, default=NEWS_LOOKBACK_DAYS,
                        help="News lookback in days")
    parser.add_argument("--gpt",     action="store_true",
                        help="Enable GPT-4o-mini scoring (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    tickers  = args.tickers if args.tickers else DEFAULT_TICKERS
    use_gpt  = args.gpt and bool(os.environ.get("OPENAI_API_KEY"))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*60}")
    print(f"Canyon v9 Step 65 — Earnings NLP Scorer")
    tier_label = "GPT-4o-mini" if use_gpt else "keyword-only (offline)"
    print(f"Tickers: {len(tickers)}  Lookback: {args.days}d  Tier: {tier_label}")
    if args.gpt and not use_gpt:
        print("  ⚠  --gpt flag set but OPENAI_API_KEY not found — falling back to keyword tier")
    print(f"{'='*60}")

    scores = []
    t0_total = time.time()
    for i, ticker in enumerate(tickers):
        print(f"  [{i+1:2d}/{len(tickers)}] {ticker:8s} … ", end="", flush=True)
        t0 = time.time()
        result = score_ticker(ticker, days_lookback=args.days, use_gpt=use_gpt)
        elapsed = time.time() - t0
        scores.append(result)
        score = result.get("forward_score", 0.0)
        sentiment = result.get("sentiment", "NEUTRAL")
        tier = result.get("tier", "keyword")
        news_n = result.get("news_count", 0)
        bar_len = max(0, int((score + 1) * 10))
        bar = ("▌" * bar_len).ljust(20)
        print(f"{bar}  {score:+.3f}  {sentiment:18s}  {news_n:2d} news  [{tier}]  {elapsed:.1f}s")

    total_time = time.time() - t0_total
    print(f"\n  Total: {total_time:.1f}s for {len(tickers)} tickers")

    if not scores:
        print("No results.")
        return

    scores_df = pd.DataFrame(scores)
    calendar_df = scores_df[["ticker", "next_earnings", "eps_surprise_pct"]].copy()

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("EARNINGS NLP RESULTS SUMMARY")
    print(f"{'─'*60}")
    if "sentiment" in scores_df.columns:
        for sent in ["BULLISH","MILDLY_BULLISH","NEUTRAL","MILDLY_BEARISH","BEARISH"]:
            cnt = int((scores_df["sentiment"] == sent).sum())
            if cnt > 0:
                pct = cnt / len(scores_df) * 100
                bar = "█" * cnt
                print(f"  {sent:18s} {bar:<25s} {cnt:2d} ({pct:.0f}%)")

    top5 = scores_df.sort_values("forward_score", ascending=False).head(5)
    print(f"\n  Top 5 BULLISH:")
    for _, row in top5.iterrows():
        print(f"    {row['ticker']:8s}  {row['forward_score']:+.3f}  {row.get('key_themes','')[:40]}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    sp = ROOT / "earnings_nlp_scores.csv"
    scores_df.to_csv(sp, index=False)
    print(f"  [scores]   {sp}")

    cp_path = ROOT / "earnings_calendar.csv"
    calendar_df.to_csv(cp_path, index=False)
    print(f"  [calendar] {cp_path}")

    write_report(scores_df, calendar_df, use_gpt, ts)

    print(f"\n{'='*60}")
    print(f"Earnings NLP complete — {len(scores_df)} tickers scored.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
