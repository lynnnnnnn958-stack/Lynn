#!/usr/bin/env python3
"""
Canyon v9 — Step 79: FinBERT News Sentiment Scorer
====================================================
Pulls recent news headlines for S&P 500 tickers via yfinance, scores
them with FinBERT (ProsusAI/finbert from HuggingFace), and outputs a
per-ticker sentiment score that can be used as an ML feature.

Why FinBERT?
------------
  FinBERT is a BERT model fine-tuned on financial text (Reuters, Bloomberg,
  analyst reports). It natively classifies finance headlines as positive /
  negative / neutral — far more accurate than generic VADER/TextBlob on
  earnings, guidance, or macroeconomic language.

Sentiment score formula
-----------------------
  For each headline: map positive → +confidence, negative → -confidence,
  neutral → 0.  Average across all headlines for the ticker.
  Range: -1.0 (fully bearish) to +1.0 (fully bullish).

  BULLISH  if sentiment_score >  0.15
  BEARISH  if sentiment_score < -0.15
  NEUTRAL  otherwise

Output
------
  finbert_sentiment.csv       — one row per ticker, all sentiment fields
  finbert_sentiment_report.md — markdown summary with top/bottom 10
  finbert_model_cache/        — local HuggingFace model cache

Usage
-----
  python3 canyon_final_v9_step79_finbert_sentiment.py
  python3 canyon_final_v9_step79_finbert_sentiment.py --top 50
  python3 canyon_final_v9_step79_finbert_sentiment.py --ticker AAPL
  python3 canyon_final_v9_step79_finbert_sentiment.py --fast
"""

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent
OUT_SENTIMENT = ROOT / "finbert_sentiment.csv"
OUT_REPORT    = ROOT / "finbert_sentiment_report.md"
MODEL_CACHE   = ROOT / "finbert_model_cache"
TICKER_FILE   = ROOT / "sp500_tickers.json"

MODEL_NAME    = "ProsusAI/finbert"

# Thresholds
BULLISH_THRESH  =  0.15
BEARISH_THRESH  = -0.15
MAX_HEADLINES   = 10
NEWS_DAYS       = 30         # only news within last 30 days
SLEEP_BETWEEN   = 0.1        # seconds between tickers

# ─────────────────────────────────────────────────────────────
# 1. UNIVERSE LOADER
# ─────────────────────────────────────────────────────────────

def load_tickers(n: int = 100) -> list[str]:
    """Load first n tickers from sp500_tickers.json (key='tickers')."""
    if not TICKER_FILE.exists():
        raise FileNotFoundError(f"Ticker file not found: {TICKER_FILE}")
    data: dict = json.loads(TICKER_FILE.read_text())
    tickers: list[str] = data.get("tickers", [])
    if not tickers:
        raise ValueError("sp500_tickers.json has no 'tickers' key or it is empty.")
    return tickers[:n]


# ─────────────────────────────────────────────────────────────
# 2. HEADLINE FETCHER
# ─────────────────────────────────────────────────────────────

def fetch_news_headlines(ticker: str, max_headlines: int = MAX_HEADLINES) -> list[str]:
    """
    Fetch recent news headlines for a ticker via yfinance.

    Returns a list of title strings (last 30 days only).
    Returns [] on any error or if no news is available.
    """
    try:
        tk = yf.Ticker(ticker)
        news_items = tk.news  # list of dicts
        if not news_items:
            return []

        cutoff_ts = (
            datetime.now(tz=timezone.utc).timestamp() - NEWS_DAYS * 86400
        )

        headlines: list[str] = []
        for item in news_items:
            if not isinstance(item, dict):
                continue

            # yfinance ≥0.2.x: nested structure item["content"]["title"]
            # yfinance <0.2.x:  flat structure item["title"]
            content = item.get("content", {}) or {}
            title = (
                content.get("title", "")          # new nested format
                or item.get("title", "")           # old flat format
            ).strip()

            # Filter by publish time — try both locations
            pub_time = (
                item.get("providerPublishTime", 0)
                or item.get("pubDate", 0)
            )
            # pubDate may be ISO string "2026-05-26T14:22:49Z"
            if isinstance(pub_time, str):
                try:
                    from datetime import datetime as _dt
                    pub_time = _dt.fromisoformat(pub_time.replace("Z", "+00:00")).timestamp()
                except Exception:
                    pub_time = 0
            if pub_time and float(pub_time) < cutoff_ts:
                continue

            if title:
                headlines.append(title)
            if len(headlines) >= max_headlines:
                break

        return headlines

    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# 3. FINBERT PIPELINE LOADER
# ─────────────────────────────────────────────────────────────

_pipe: Any = None  # module-level singleton


def load_finbert():
    """
    Load the FinBERT sentiment pipeline once.
    Caches the model to MODEL_CACHE to avoid re-downloading.
    Returns the transformers pipeline object.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers library not found. Install with: pip install transformers torch"
        ) from exc

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"  Loading FinBERT model from HuggingFace (cache: {MODEL_CACHE}) …")
    _pipe = hf_pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        max_length=512,
        truncation=True,
        model_kwargs={"cache_dir": str(MODEL_CACHE)},
    )
    print("  FinBERT loaded.")
    return _pipe


# ─────────────────────────────────────────────────────────────
# 4. PER-TICKER SCORER
# ─────────────────────────────────────────────────────────────

_EMPTY_RESULT = {
    "ticker": "",
    "sentiment_score": 0.0,
    "n_headlines": 0,
    "label": "NEUTRAL",
    "positive": 0,
    "negative": 0,
    "neutral": 0,
    "sample_headline": "",
}


def score_ticker(ticker: str, pipe) -> dict:
    """
    Fetch headlines for ticker, score with FinBERT, return result dict.

    sentiment_score = mean of (+confidence if positive,
                                -confidence if negative,
                                0 if neutral)
    Range: -1.0 to +1.0.
    """
    result = {**_EMPTY_RESULT, "ticker": ticker}

    headlines = fetch_news_headlines(ticker)
    if not headlines:
        return result

    result["n_headlines"] = len(headlines)
    result["sample_headline"] = headlines[0]

    scores: list[float] = []
    pos_count = neg_count = neu_count = 0

    for headline in headlines:
        try:
            output = pipe(headline)          # returns [{"label": "positive", "score": 0.94}]
            if not output or not isinstance(output, list):
                continue
            item = output[0]
            label_raw: str = item.get("label", "neutral").lower()
            conf: float = float(item.get("score", 0.0))

            if label_raw == "positive":
                scores.append(conf)
                pos_count += 1
            elif label_raw == "negative":
                scores.append(-conf)
                neg_count += 1
            else:
                scores.append(0.0)
                neu_count += 1
        except Exception:
            continue

    if not scores:
        return result

    sent_score = float(np.mean(scores))
    result["sentiment_score"] = round(sent_score, 6)
    result["positive"] = pos_count
    result["negative"] = neg_count
    result["neutral"] = neu_count

    if sent_score > BULLISH_THRESH:
        result["label"] = "BULLISH"
    elif sent_score < BEARISH_THRESH:
        result["label"] = "BEARISH"
    else:
        result["label"] = "NEUTRAL"

    return result


# ─────────────────────────────────────────────────────────────
# 5. RUN (batch)
# ─────────────────────────────────────────────────────────────

def run(tickers: list[str]) -> pd.DataFrame:
    """
    Score all tickers, save CSV + markdown report, print top/bottom 10.
    Returns the results DataFrame.
    """
    pipe = load_finbert()

    rows: list[dict] = []
    total = len(tickers)
    print(f"\n  Scoring {total} tickers with FinBERT …\n")

    for i, ticker in enumerate(tickers, 1):
        try:
            result = score_ticker(ticker, pipe)
        except Exception as exc:
            print(f"  [WARN] {ticker}: unexpected error — {exc}")
            result = {**_EMPTY_RESULT, "ticker": ticker}

        rows.append(result)
        label_str = result["label"]
        sc = result["sentiment_score"]
        n = result["n_headlines"]
        print(f"  [{i:3d}/{total}] {ticker:<6}  score={sc:+.3f}  {label_str:<7}  n={n}")

        time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(rows)

    # Cross-sectional rank normalize sentiment_score → 0–100
    df["rank_sentiment"] = (
        df["sentiment_score"]
        .rank(method="average", na_option="keep")
        .pipe(lambda s: (s - 1) / (len(df) - 1) * 100 if len(df) > 1 else s * 0 + 50)
        .round(2)
    )

    # Canonical column order
    col_order = [
        "ticker", "sentiment_score", "rank_sentiment", "label",
        "n_headlines", "positive", "negative", "neutral", "sample_headline",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Save CSV
    df.to_csv(OUT_SENTIMENT, index=False)
    print(f"\n  Saved: {OUT_SENTIMENT}")

    # Save markdown report
    _write_report(df)
    print(f"  Saved: {OUT_REPORT}")

    # Print top / bottom
    _print_summary(df)

    # ── IC Validation: correlate sentiment vs recent price returns ──────────
    _validate_ic(df)

    return df


def _validate_ic(df: pd.DataFrame) -> None:
    """
    Validate FinBERT sentiment by correlating rank_sentiment against
    recent 1M and 3M price returns (contemporaneous check).
    Also checks alignment with ML predicted_score if available.
    Prints IC and t-stat so you can judge signal quality.
    """
    from scipy import stats as spstats

    price_cache = ROOT / "sp500_price_cache.csv"
    ml_scores   = ROOT / "regime_ml_scores.csv"

    print("\n  ── FinBERT IC Validation ──────────────────────────────")

    # 1. vs price returns
    if price_cache.exists():
        try:
            prices = pd.read_csv(price_cache, index_col=0, parse_dates=True)
            ret_1m = prices.pct_change(21).iloc[-1]   # last 1M return per ticker
            ret_3m = prices.pct_change(63).iloc[-1]
            val = df[["ticker","rank_sentiment"]].copy()
            val["ret_1m"] = val["ticker"].map(ret_1m.to_dict())
            val["ret_3m"] = val["ticker"].map(ret_3m.to_dict())
            val = val.dropna()
            if len(val) >= 10:
                ic1, p1 = spstats.spearmanr(val["rank_sentiment"], val["ret_1m"])
                ic3, p3 = spstats.spearmanr(val["rank_sentiment"], val["ret_3m"])
                n = len(val)
                t1 = ic1 * np.sqrt(n-2) / np.sqrt(1-ic1**2+1e-9)
                t3 = ic3 * np.sqrt(n-2) / np.sqrt(1-ic3**2+1e-9)
                flag1 = "✅" if abs(t1) > 1.5 else ("⚠️" if abs(t1) > 1.0 else "❌")
                flag3 = "✅" if abs(t3) > 1.5 else ("⚠️" if abs(t3) > 1.0 else "❌")
                print(f"  vs 1M ret  : IC={ic1:+.4f}  t={t1:+.2f}  n={n}  {flag1}")
                print(f"  vs 3M ret  : IC={ic3:+.4f}  t={t3:+.2f}  n={n}  {flag3}")
                if abs(t1) < 1.0 and abs(t3) < 1.0:
                    print("  ⚠️  Sentiment shows weak correlation with recent returns.")
                    print("     Consider: (a) expand news coverage, (b) use as filter only")
                else:
                    print("  ✅  Sentiment shows meaningful alignment with recent price moves.")
        except Exception as e:
            print(f"  Price IC check skipped: {e}")

    # 2. vs ML scores (alignment check)
    if ml_scores.exists():
        try:
            ml = pd.read_csv(ml_scores)[["ticker","predicted_score"]]
            ml["predicted_score"] = pd.to_numeric(ml["predicted_score"], errors="coerce")
            val2 = df[["ticker","rank_sentiment"]].merge(ml, on="ticker", how="inner").dropna()
            if len(val2) >= 10:
                ic_ml, _ = spstats.spearmanr(val2["rank_sentiment"], val2["predicted_score"])
                n2 = len(val2)
                t_ml = ic_ml * np.sqrt(n2-2) / np.sqrt(1-ic_ml**2+1e-9)
                flag = "✅" if ic_ml > 0.1 else ("⚠️" if ic_ml > 0 else "❌ opposite direction")
                print(f"  vs ML score: IC={ic_ml:+.4f}  t={t_ml:+.2f}  n={n2}  {flag}")
        except Exception as e:
            print(f"  ML alignment check skipped: {e}")

    print("  ────────────────────────────────────────────────────────")


def _write_report(df: pd.DataFrame) -> None:
    """Write a markdown summary report with top/bottom 10 tickers."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_total  = len(df)
    n_bull   = int((df["label"] == "BULLISH").sum())
    n_bear   = int((df["label"] == "BEARISH").sum())
    n_neu    = int((df["label"] == "NEUTRAL").sum())

    sorted_df = df.sort_values("sentiment_score", ascending=False)
    top10    = sorted_df.head(10)
    bottom10 = sorted_df.tail(10).sort_values("sentiment_score")

    def _table(sub: pd.DataFrame) -> str:
        lines = ["| Ticker | Score | Rank | Label | Headlines |",
                 "|--------|-------|------|-------|-----------|"]
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['ticker']} | {row['sentiment_score']:+.3f} "
                f"| {row['rank_sentiment']:.1f} | {row['label']} "
                f"| {int(row['n_headlines'])} |"
            )
        return "\n".join(lines)

    report = f"""# Canyon v9 — FinBERT Sentiment Report
_Generated: {now_str}_

## Summary
| Metric | Value |
|--------|-------|
| Tickers scored | {n_total} |
| BULLISH | {n_bull} ({100*n_bull//max(n_total,1)}%) |
| BEARISH | {n_bear} ({100*n_bear//max(n_total,1)}%) |
| NEUTRAL | {n_neu} ({100*n_neu//max(n_total,1)}%) |
| Model | ProsusAI/finbert |
| News window | Last 30 days |

## Top 10 Most Bullish
{_table(top10)}

## Top 10 Most Bearish
{_table(bottom10)}
"""

    OUT_REPORT.write_text(report)


def _print_summary(df: pd.DataFrame) -> None:
    """Print top 10 BULLISH and top 10 BEARISH to stdout."""
    bullish = (
        df[df["label"] == "BULLISH"]
        .sort_values("sentiment_score", ascending=False)
        .head(10)
    )
    bearish = (
        df[df["label"] == "BEARISH"]
        .sort_values("sentiment_score")
        .head(10)
    )

    print("\n  ── Top 10 BULLISH ──────────────────────────────")
    if bullish.empty:
        print("  (none)")
    else:
        for _, r in bullish.iterrows():
            print(
                f"  {r['ticker']:<6}  score={r['sentiment_score']:+.3f}  "
                f"rank={r['rank_sentiment']:5.1f}  n={int(r['n_headlines'])}"
            )

    print("\n  ── Top 10 BEARISH ──────────────────────────────")
    if bearish.empty:
        print("  (none)")
    else:
        for _, r in bearish.iterrows():
            print(
                f"  {r['ticker']:<6}  score={r['sentiment_score']:+.3f}  "
                f"rank={r['rank_sentiment']:5.1f}  n={int(r['n_headlines'])}"
            )
    print()


# ─────────────────────────────────────────────────────────────
# 6. STREAMLIT UI  (imported when run via `streamlit run`)
# ─────────────────────────────────────────────────────────────

def streamlit_ui() -> None:
    """White professional Streamlit UI for FinBERT sentiment explorer."""
    import streamlit as st

    st.set_page_config(
        page_title="Canyon v9 · FinBERT Sentiment",
        page_icon="📰",
        layout="wide",
    )

    # ── White UI overrides ──────────────────────────────────
    st.markdown(
        """
        <style>
        body, .main, .block-container {background-color:#ffffff; color:#1a1a1a;}
        h1,h2,h3,h4,h5 {color:#1a1a1a;}
        .stMetric label {color:#555555;}
        .stDataFrame thead tr th {background:#f5f5f5; color:#1a1a1a;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Canyon v9 · Step 79 — FinBERT News Sentiment")
    st.caption("ProsusAI/finbert • per-ticker headline sentiment • S&P 500")

    # Load existing results if available
    if OUT_SENTIMENT.exists():
        df = pd.read_csv(OUT_SENTIMENT)
        mtime = datetime.fromtimestamp(OUT_SENTIMENT.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
        st.info(f"Loaded cached results from `finbert_sentiment.csv`  (updated {mtime})")
    else:
        st.warning("No sentiment data yet — run the script first.")
        st.code(
            "python3 canyon_final_v9_step79_finbert_sentiment.py --fast",
            language="bash",
        )
        return

    n_total = len(df)
    n_bull  = int((df["label"] == "BULLISH").sum())
    n_bear  = int((df["label"] == "BEARISH").sum())
    n_neu   = int((df["label"] == "NEUTRAL").sum())

    # ── KPI row ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers", f"{n_total}")
    c2.metric("BULLISH", f"{n_bull}", delta=f"{100*n_bull//max(n_total,1)}%")
    c3.metric("BEARISH", f"{n_bear}", delta=f"-{100*n_bear//max(n_total,1)}%")
    c4.metric("NEUTRAL", f"{n_neu}")

    st.divider()

    # ── Filters ─────────────────────────────────────────────
    col_filt, col_sort = st.columns([2, 1])
    with col_filt:
        label_filter = st.multiselect(
            "Filter by label",
            options=["BULLISH", "NEUTRAL", "BEARISH"],
            default=["BULLISH", "NEUTRAL", "BEARISH"],
        )
    with col_sort:
        sort_col = st.selectbox(
            "Sort by",
            options=["sentiment_score", "rank_sentiment", "n_headlines"],
            index=0,
        )

    view = df[df["label"].isin(label_filter)].sort_values(
        sort_col, ascending=(sort_col != "rank_sentiment")
    )

    # Colour-code sentiment_score column
    def _colour_score(val: float) -> str:
        if val > BULLISH_THRESH:
            return "color: #00bcd4; font-weight:bold"   # cyan = bullish
        if val < BEARISH_THRESH:
            return "color: #e53935; font-weight:bold"   # red  = bearish
        return "color: #555555"

    styled = (
        view.style
        .applymap(_colour_score, subset=["sentiment_score"])
        .format({"sentiment_score": "{:+.4f}", "rank_sentiment": "{:.1f}"})
    )

    st.dataframe(styled, use_container_width=True, height=520)

    # ── Single-ticker drill-down ─────────────────────────────
    st.divider()
    st.subheader("Ticker Detail")
    sel_ticker = st.selectbox("Select ticker", options=sorted(df["ticker"].tolist()))
    if sel_ticker:
        row = df[df["ticker"] == sel_ticker].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Sentiment Score", f"{row['sentiment_score']:+.4f}")
        d2.metric("Rank (0–100)", f"{row['rank_sentiment']:.1f}")
        d3.metric("Label", str(row["label"]))
        d4.metric("Headlines used", int(row["n_headlines"]))
        st.markdown(
            f"**Sample headline:** _{row.get('sample_headline','—')}_"
        )
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Positive", int(row["positive"]))
        bc2.metric("Negative", int(row["negative"]))
        bc3.metric("Neutral",  int(row["neutral"]))

    # ── Distribution chart ───────────────────────────────────
    st.divider()
    st.subheader("Sentiment Score Distribution")
    try:
        import altair as alt

        hist_df = df[["sentiment_score"]].copy()
        chart = (
            alt.Chart(hist_df)
            .mark_bar(color="#00bcd4", opacity=0.8)
            .encode(
                alt.X("sentiment_score:Q", bin=alt.Bin(maxbins=40), title="Sentiment Score"),
                alt.Y("count()", title="Count"),
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        st.bar_chart(df["sentiment_score"].dropna())


# ─────────────────────────────────────────────────────────────
# 7. MAIN / ARGPARSE
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 79 — FinBERT News Sentiment Scorer"
    )
    parser.add_argument(
        "--top", type=int, default=100,
        help="Number of tickers to score (default: 100)",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Score a single ticker with debug output (e.g. --ticker AAPL)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: top 30 tickers only",
    )
    args = parser.parse_args()

    if args.ticker:
        # ── Single-ticker debug mode ─────────────────────────
        ticker = args.ticker.upper().strip()
        print(f"\n=== FinBERT debug: {ticker} ===")
        headlines = fetch_news_headlines(ticker)
        print(f"  Headlines fetched: {len(headlines)}")
        for i, h in enumerate(headlines, 1):
            print(f"  [{i}] {h}")

        if not headlines:
            print("  No headlines found — returning neutral score.")
            return

        pipe = load_finbert()
        result = score_ticker(ticker, pipe)
        print(f"\n  sentiment_score : {result['sentiment_score']:+.4f}")
        print(f"  label           : {result['label']}")
        print(f"  n_headlines     : {result['n_headlines']}")
        print(f"  positive        : {result['positive']}")
        print(f"  negative        : {result['negative']}")
        print(f"  neutral         : {result['neutral']}")
        print(f"  sample_headline : {result['sample_headline']}")
        return

    # ── Batch mode ───────────────────────────────────────────
    n = 30 if args.fast else args.top
    print(f"\n=== Canyon v9 Step 79: FinBERT Sentiment (n={n}) ===")
    tickers = load_tickers(n)
    print(f"  Loaded {len(tickers)} tickers from {TICKER_FILE.name}")
    run(tickers)
    print("=== Step 79 complete ===\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # When invoked via `streamlit run`, __name__ == "__main__" too,
    # but sys.argv[0] will contain "streamlit" context.
    import sys

    _is_streamlit = (
        len(sys.argv) > 0
        and "streamlit" in sys.argv[0].lower()
    ) or any("streamlit" in a for a in sys.argv)

    if _is_streamlit:
        streamlit_ui()
    else:
        main()
