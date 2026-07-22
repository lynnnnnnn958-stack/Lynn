#!/usr/bin/env python3
"""
canyon_final_v9_step250_alt_data.py  —  Week 5: Alternative Data Integration

Four alternative data signals evaluated with Spearman IC:
  1. news_sentiment  — yfinance headlines + VADER/keyword NLP (per-ticker)
  2. google_trends   — pytrends weekly US search interest (5-year history)
  3. sec_filing_lag  — EDGAR 10-K lag vs fiscal year end (full history)
  4. fear_gauge      — CBOE VIX + put/call ratio (market-wide regime)

Outputs:
  alt_data_signals_daily.csv   — daily signal panel (date × ticker)
  alt_data_ic_report.csv       — Spearman IC per signal per period
  alt_data_report.md           — narrative report
  cboe_fear_gauge.csv          — VIX / P/C cache
  google_trends_cache.csv      — Google Trends cache
  edgar_filing_lag.csv         — EDGAR filing lag cache
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings

# urllib3 ≥ 2.0 renamed method_whitelist → allowed_methods; pytrends still uses the old name
try:
    import urllib3.util.retry as _retry_mod
    _orig_init = _retry_mod.Retry.__init__
    def _compat_retry_init(self, *a, **kw):
        if "method_whitelist" in kw:
            kw.setdefault("allowed_methods", kw.pop("method_whitelist"))
        _orig_init(self, *a, **kw)
    _retry_mod.Retry.__init__ = _compat_retry_init  # type: ignore[method-assign]
except Exception:
    pass
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Universe (stock tickers only — no ETFs for alt-data signals) ─────────────
STOCK_UNIVERSE: list[str] = list(dict.fromkeys([
    "NVDA", "TSLA", "AMD", "MU", "GOOGL", "AMZN", "MSFT", "AAPL", "META", "JPM",
    "XOM", "CVX", "JNJ", "WMT", "KO", "PEP", "MRK", "ABBV", "UNH", "LLY",
    "TMO", "COST", "V", "MA", "HD", "PYPL", "NFLX", "INTC", "QCOM", "TXN",
    "AVGO", "CRM", "ADBE",
]))

# Full universe for price loading (includes ETFs)
FULL_UNIVERSE: list[str] = list(dict.fromkeys([
    "SPY", "QQQ", "SMH", "SOXX", "XLK", "XLE", "XLF", "XLV", "XLU", "XLP",
    "NVDA", "TSLA", "AMD", "MU", "GOOGL", "AMZN", "MSFT", "AAPL", "META", "JPM",
    "XOM", "CVX", "JNJ", "WMT", "KO", "PEP", "MRK", "ABBV", "UNH", "LLY",
    "TMO", "COST", "V", "MA", "HD", "PYPL", "NFLX", "INTC", "QCOM", "TXN",
    "AVGO", "CRM", "ADBE",
]))

# ── Paths ─────────────────────────────────────────────────────────────────────
PRICES_FILE   = ROOT / "backtest_price_cache.csv"
PIT_PRICES    = ROOT / "pit_price_cache.csv"
SIGNALS_OUT   = ROOT / "alt_data_signals_daily.csv"
IC_OUT        = ROOT / "alt_data_ic_report.csv"
REPORT_OUT    = ROOT / "alt_data_report.md"
FEAR_CACHE    = ROOT / "cboe_fear_gauge.csv"
TRENDS_CACHE  = ROOT / "google_trends_cache.csv"
EDGAR_CACHE   = ROOT / "edgar_filing_lag.csv"

# ── Config ────────────────────────────────────────────────────────────────────
IS_CUTOFF    = pd.Timestamp("2020-01-01")
FORWARD_DAYS = 21        # IC horizon
TODAY        = pd.Timestamp.today().normalize()
CBOE_TIMEOUT = 20
EDGAR_DELAY  = 0.15      # seconds between EDGAR requests (10 req/sec limit)

# ── EDGAR endpoints ───────────────────────────────────────────────────────────
EDGAR_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_HEADERS         = {"User-Agent": "CanyonQuant-Research research@canyonquant.com"}

# ── CBOE endpoints (tried in order) ──────────────────────────────────────────
CBOE_PC_URLS = [
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/PCE_History.csv",
    "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/equitypc.csv",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Sentiment scorer (VADER → nltk VADER → keyword fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_score_fn = None
_scorer_name = ""

def _setup_scorer() -> None:
    global _score_fn, _scorer_name
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        _va = SentimentIntensityAnalyzer()
        _score_fn = lambda t: _va.polarity_scores(t)["compound"]
        _scorer_name = "VADER (vaderSentiment)"
        return
    except ImportError:
        pass
    try:
        import nltk  # type: ignore
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        _nv = SentimentIntensityAnalyzer()
        _score_fn = lambda t: _nv.polarity_scores(t)["compound"]
        _scorer_name = "VADER (nltk)"
        return
    except Exception:
        pass
    # Keyword fallback
    _POS = {
        "beat", "beats", "record", "growth", "surges", "strong", "profit",
        "gains", "rises", "raised", "upgrade", "upgraded", "outperform",
        "exceeds", "soars", "boost", "rally", "bullish", "expand", "wins",
        "innovation", "breakthrough", "dividend", "buyback", "positive",
    }
    _NEG = {
        "miss", "misses", "loss", "losses", "decline", "cut", "lawsuit",
        "recall", "fraud", "falls", "slumps", "downgrade", "downgraded",
        "underperform", "disappoints", "plunges", "warning", "layoffs",
        "charges", "writedown", "default", "bankruptcy", "probe", "crash",
    }
    def _kw(text: str) -> float:
        words = set(re.sub(r"[^a-z ]", "", text.lower()).split())
        pos, neg = len(words & _POS), len(words & _NEG)
        return float(pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    _score_fn = _kw
    _scorer_name = "keyword fallback"


def score_text(text: str) -> float:
    if _score_fn is None:
        return 0.0
    return float(_score_fn(str(text)))


# ═══════════════════════════════════════════════════════════════════════════════
# Price loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [PRICES_FILE, PIT_PRICES]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            # Handle both wide (index=date, col=ticker) and long (date, ticker, adj_close)
            if "ticker" in df.columns and "adj_close" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                wide = df.pivot(index="date", columns="ticker", values="adj_close")
            else:
                df.index = pd.to_datetime(df.iloc[:, 0])
                wide = df.iloc[:, 1:]
            frames.append(wide)
        except Exception as e:
            print(f"  [prices] Warning loading {path.name}: {e}")

    if not frames:
        # Last resort: try wide format with date as index
        if PRICES_FILE.exists():
            try:
                df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
                frames.append(df)
            except Exception:
                pass

    if not frames:
        raise FileNotFoundError(
            "No price cache found. Run step100 first (canyon_final_v9_step100_walk_forward_oos.py)."
        )

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index = pd.to_datetime(combined.index)
    return combined.astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# Signal 1: News Sentiment
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_news_sentiment(tickers: list[str]) -> tuple[pd.DataFrame, str]:
    """
    Score recent yfinance headlines per ticker.
    Returns (daily_panel, note). Coverage: ~30 days (yfinance API limit).
    """
    print("\n[1/4] News Sentiment (yfinance + NLP)")
    print(f"  Scorer: {_scorer_name}")

    records = []
    for i, tk in enumerate(tickers):
        try:
            news = yf.Ticker(tk).news or []
            for item in news:
                # Handle both legacy format (flat keys) and new format (nested 'content')
                content = item.get("content", item)
                title   = content.get("title", item.get("title", ""))
                # New format: pubDate is ISO string; legacy: providerPublishTime is unix int
                pub_str = content.get("pubDate", "")
                if pub_str:
                    try:
                        pub = pd.Timestamp(pub_str).tz_convert(None).normalize()
                    except Exception:
                        try:
                            pub = pd.Timestamp(pub_str).tz_localize(None).normalize()
                        except Exception:
                            continue
                else:
                    ts = item.get("providerPublishTime", 0)
                    if not ts:
                        continue
                    pub = pd.Timestamp(ts, unit="s").normalize()
                if not title:
                    continue
                records.append({"date": pub, "ticker": tk, "sentiment": score_text(title)})
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(tickers)} tickers processed")

    if not records:
        return pd.DataFrame(), "No headlines retrieved from yfinance."

    df = pd.DataFrame(records)
    panel = (
        df.groupby(["date", "ticker"])["sentiment"]
        .mean()
        .unstack("ticker")
        .sort_index()
    )
    date_range = f"{panel.index.min().date()} → {panel.index.max().date()}"
    n_days = len(panel)
    n_tickers = panel.shape[1]
    note = (
        f"Retrieved {n_days} trading days ({date_range}), {n_tickers} tickers. "
        f"Coverage limited to yfinance recent news (~30 days); IC evaluation on most-recent period only. "
        f"For full historical IC, subscribe to Bloomberg/Refinitiv or use EDGAR MD&A text."
    )
    print(f"  Retrieved: {n_tickers} tickers, {n_days} days ({date_range})")
    return panel, note


# ═══════════════════════════════════════════════════════════════════════════════
# Signal 2: Google Trends
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_google_trends(
    tickers: list[str], use_cache: bool = True
) -> tuple[pd.DataFrame, str]:
    """
    Download 5-year weekly Google Trends for each ticker.
    Returns z-score-normalized daily panel (forward-filled from weekly).
    """
    print("\n[2/4] Google Trends (pytrends, 5-year weekly)")

    if use_cache and TRENDS_CACHE.exists():
        df = pd.read_csv(TRENDS_CACHE, index_col=0, parse_dates=True)
        print(f"  Loaded from cache: {df.shape[1]} tickers, {len(df)} days")
        note = (
            f"Loaded from cache ({TRENDS_CACHE.name}): {df.shape[1]} tickers, {len(df)} days. "
            f"Z-score normalized; high score = above-average retail attention. "
            f"Positive IC = search attention → short-term momentum; negative = mean-reversion."
        )
        return df, note

    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError:
        msg = "pytrends not installed. Run: .venv/bin/pip install pytrends"
        print(f"  {msg}")
        return pd.DataFrame(), msg

    pytrends = TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=1.0)
    # Clean up tickers for search terms
    term_map = {tk: tk.replace("-B", "").replace("-A", "") for tk in tickers}

    all_series: dict[str, pd.Series] = {}
    batch_size = 5

    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start : batch_start + batch_size]
        terms = [term_map[t] for t in batch]
        try:
            pytrends.build_payload(terms, cat=0, timeframe="today 5-y", geo="US")
            weekly = pytrends.interest_over_time()
            if weekly.empty:
                print(f"  Batch {batch_start // batch_size + 1}: empty response")
                time.sleep(5)
                continue
            weekly = weekly.drop(columns=["isPartial"], errors="ignore")
            for term, tk in zip(terms, batch):
                if term in weekly.columns:
                    all_series[tk] = weekly[term].astype(float)
            print(f"  Batch {batch_start // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}: {batch}")
            time.sleep(4)
        except Exception as e:
            print(f"  Batch {batch_start // batch_size + 1} failed: {e}")
            time.sleep(12)

    if not all_series:
        return pd.DataFrame(), "Google Trends: no data retrieved (rate limit or network error)."

    trends_raw = pd.DataFrame(all_series)
    trends_raw.index = pd.to_datetime(trends_raw.index)

    # Z-score normalize each ticker over time (cross-sectionally comparable)
    trends_z = (trends_raw - trends_raw.mean()) / (trends_raw.std() + 1e-8)

    # Weekly pytrends dates (often Sundays) don't align with business days.
    # Union the weekly index with business days first, then ffill to fill gaps.
    bdays = pd.bdate_range(
        trends_z.index.min() - pd.Timedelta(days=7), TODAY
    )
    combined_idx = trends_z.index.union(bdays).sort_values()
    trends_daily = trends_z.reindex(combined_idx).ffill().reindex(bdays)

    trends_daily.to_csv(TRENDS_CACHE)
    print(f"  Saved to {TRENDS_CACHE.name}: {trends_daily.shape[1]} tickers, {len(trends_daily)} days")

    note = (
        f"Downloaded {len(all_series)} tickers, 5-year weekly data → z-score → daily ffill. "
        f"Signal represents retail search attention vs historical average. "
        f"IC interpretation: positive = momentum effect, negative = contrarian (overbought)."
    )
    return trends_daily, note


# ═══════════════════════════════════════════════════════════════════════════════
# Signal 3: SEC EDGAR Filing Lag
# ═══════════════════════════════════════════════════════════════════════════════

def _get_cik_map() -> dict[str, str]:
    resp = requests.get(EDGAR_TICKERS_URL, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    return {
        str(v.get("ticker", "")).upper(): str(v.get("cik_str", "")).zfill(10)
        for v in raw.values()
    }


def _get_annual_filings(cik: str) -> pd.DataFrame:
    url  = EDGAR_SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    data   = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame({
        "form":   recent.get("form", []),
        "filed":  pd.to_datetime(recent.get("filingDate", []), errors="coerce"),
        "period": pd.to_datetime(recent.get("reportDate", []),  errors="coerce"),
    }).dropna()
    annual = df[df["form"].isin(["10-K", "10-K/A"])].copy()
    annual["lag_days"] = (annual["filed"] - annual["period"]).dt.days
    return annual[annual["lag_days"].between(30, 365)]


def fetch_sec_filing_lag(
    tickers: list[str], use_cache: bool = True
) -> tuple[pd.DataFrame, str]:
    """
    10-K filing lag = days between fiscal year end and SEC filing date.
    Fast filers signal health; slow filers signal potential distress.
    Signal = -(lag_z): higher score = faster filer = positive.
    """
    print("\n[3/4] SEC EDGAR Filing Lag")

    if use_cache and EDGAR_CACHE.exists():
        df = pd.read_csv(EDGAR_CACHE, index_col=0, parse_dates=True)
        print(f"  Loaded from cache: {df.shape[1]} tickers, {len(df)} days")
        note = (
            f"Loaded from cache ({EDGAR_CACHE.name}): {df.shape[1]} tickers. "
            f"Signal = -(10-K lag z-score): fast filers score higher. "
            f"Typical S&P 500 large-cap lag: 45–75 days. Late >90 days = potential distress."
        )
        return df, note

    try:
        print("  Fetching ticker→CIK map from EDGAR...")
        cik_map = _get_cik_map()
        print(f"  CIK map loaded: {len(cik_map)} tickers")
    except Exception as e:
        msg = f"EDGAR CIK map failed: {e}"
        print(f"  {msg}")
        return pd.DataFrame(), msg

    all_lag: dict[str, pd.Series] = {}

    for i, tk in enumerate(tickers):
        cik = cik_map.get(tk, cik_map.get(tk.replace("-B", ""), ""))
        if not cik or cik == "0000000000":
            continue
        try:
            filings = _get_annual_filings(cik)
            if filings.empty:
                continue
            filings = filings.sort_values("filed")
            # Point-in-time: lag is known only on the filing date
            lag_s = filings.set_index("filed")["lag_days"].astype(float)
            lag_s = lag_s[~lag_s.index.duplicated(keep="last")]
            all_lag[tk] = lag_s
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(tickers)} tickers")
            time.sleep(EDGAR_DELAY)
        except Exception:
            time.sleep(1)

    if not all_lag:
        return pd.DataFrame(), "EDGAR: no filing data retrieved."

    lag_df = pd.DataFrame(all_lag)
    lag_df.index = pd.to_datetime(lag_df.index)

    # Expand to daily business days via forward-fill (only after filing date = PIT)
    bdays = pd.bdate_range("2000-01-01", TODAY)
    lag_daily = lag_df.reindex(bdays).ffill()

    # Cross-sectional z-score, then invert (fast filer = high score)
    lag_z = (lag_daily - lag_daily.mean()) / (lag_daily.std() + 1e-8)
    lag_signal = -lag_z

    lag_signal.to_csv(EDGAR_CACHE)
    n_tickers = lag_signal.shape[1]
    date_range_str = f"{lag_signal.dropna(how='all').index.min().date()} → {lag_signal.dropna(how='all').index.max().date()}"
    print(f"  Saved to {EDGAR_CACHE.name}: {n_tickers} tickers")

    note = (
        f"EDGAR 10-K filings for {n_tickers} tickers ({date_range_str}). "
        f"Signal = -(lag z-score): positive = fast filer vs peers. "
        f"Academic basis: Richardson et al. (2002) — filing delays predict negative returns."
    )
    return lag_signal, note


# ═══════════════════════════════════════════════════════════════════════════════
# Signal 4: Fear Gauge (VIX + CBOE Put/Call)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fear_gauge(use_cache: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Market-wide fear indicators:
      - VIX  (^VIX, yfinance) — volatility regime
      - P/C  (^PCE, CBOE)    — options market sentiment
    Returns DataFrame with columns ['vix', 'put_call'] indexed by date.
    High VIX / high P/C = fear = contrarian bullish signal historically.
    """
    print("\n[4/4] Fear Gauge (VIX + Put/Call ratio)")

    if use_cache and FEAR_CACHE.exists():
        df = pd.read_csv(FEAR_CACHE, index_col=0, parse_dates=True)
        print(f"  Loaded from cache: {df.shape[1]} cols, {len(df)} days")
        note = (
            f"Loaded from cache ({FEAR_CACHE.name}): {', '.join(df.columns.tolist())}. "
            f"VIX >25 = elevated fear; P/C >1.0 = heavy put buying. "
            f"Contrarian: peak fear historically precedes market reversals."
        )
        return df, note

    result_frames: dict[str, pd.Series] = {}

    # VIX (reliable via yfinance)
    try:
        vix = yf.download("^VIX", start="2000-01-01", progress=False, auto_adjust=True)
        if not vix.empty:
            vix_s = vix["Close"].squeeze()
            vix_s.index = pd.to_datetime(vix_s.index)
            vix_s.name = "vix"
            result_frames["vix"] = vix_s
            print(f"  VIX: {len(vix_s)} days ({vix_s.index.min().date()} → {vix_s.index.max().date()})")
    except Exception as e:
        print(f"  VIX download failed: {e}")

    # CBOE equity put/call ratio (try CBOE CDN first, then yfinance ^PCE)
    pc_obtained = False
    for url in CBOE_PC_URLS:
        try:
            resp = requests.get(url, timeout=CBOE_TIMEOUT, headers={"User-Agent": "CanyonQuant"})
            resp.raise_for_status()
            raw_text = resp.text
            # CBOE CSV has header rows — find the actual data
            lines = [l for l in raw_text.splitlines() if l.strip()]
            # Find the header line
            header_idx = next(
                (i for i, l in enumerate(lines) if re.search(r"date|Date", l)), 0
            )
            csv_text = "\n".join(lines[header_idx:])
            df_pc = pd.read_csv(StringIO(csv_text))
            df_pc.columns = [c.strip().lower() for c in df_pc.columns]
            date_col = next((c for c in df_pc.columns if "date" in c), None)
            pc_col   = next((c for c in df_pc.columns if "p/c" in c or "ratio" in c or "call" in c.lower()), None)
            if date_col and pc_col:
                df_pc[date_col] = pd.to_datetime(df_pc[date_col], errors="coerce")
                pc_s = pd.to_numeric(df_pc.set_index(date_col)[pc_col], errors="coerce").dropna()
                pc_s.index = pd.to_datetime(pc_s.index)
                pc_s = pc_s.sort_index()
                pc_s.name = "put_call"
                result_frames["put_call"] = pc_s
                print(f"  P/C ratio (CBOE): {len(pc_s)} days ({pc_s.index.min().date()} → {pc_s.index.max().date()})")
                pc_obtained = True
                break
        except Exception as e:
            print(f"  CBOE URL {url} failed: {e}")

    if not pc_obtained:
        try:
            pc_yf = yf.download("^PCE", start="2000-01-01", progress=False, auto_adjust=True)
            if not pc_yf.empty:
                pc_s = pc_yf["Close"].squeeze()
                pc_s.index = pd.to_datetime(pc_s.index)
                pc_s.name = "put_call"
                result_frames["put_call"] = pc_s
                print(f"  P/C ratio (yfinance ^PCE): {len(pc_s)} days")
        except Exception as e:
            print(f"  yfinance ^PCE also failed: {e}")

    if not result_frames:
        return pd.DataFrame(), "Fear gauge: both VIX and P/C download failed."

    fear_df = pd.DataFrame(result_frames).sort_index()
    fear_df.index.name = "date"
    fear_df.to_csv(FEAR_CACHE)

    cols_got = ", ".join(fear_df.columns.tolist())
    note = (
        f"Fear gauge columns: {cols_got}. "
        f"VIX: implied vol index (CBOE); P/C: equity put/call ratio (CBOE PCE). "
        f"Used as market-timing signal: high fear (VIX>25 or P/C>1.0) historically contrarian bullish."
    )
    return fear_df, note


# ═══════════════════════════════════════════════════════════════════════════════
# IC Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_forward_returns(prices: pd.DataFrame, days: int = FORWARD_DAYS) -> pd.DataFrame:
    """prices[t+days]/prices[t] - 1, aligned so .loc[t] = forward return from t."""
    return prices.pct_change(days).shift(-days)


def _spearman_ic(sig: pd.Series, fwd: pd.Series) -> float:
    df = pd.DataFrame({"s": sig, "f": fwd}).dropna()
    if len(df) < 5:
        return np.nan
    return float(stats.spearmanr(df["s"], df["f"]).statistic)


def evaluate_cross_sectional_ic(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    signal_name: str,
) -> pd.DataFrame:
    """
    Monthly cross-sectional Spearman IC: signal at date t vs 21-day forward return.
    Only uses stocks (not ETFs) in the panel.
    """
    fwd_ret = compute_forward_returns(prices)
    common  = sorted(set(panel.columns) & set(fwd_ret.columns) & set(STOCK_UNIVERSE))

    if not common:
        print(f"  [{signal_name}] No common tickers with price data")
        return pd.DataFrame()

    sig = panel[common]
    fwd = fwd_ret[common]

    start = max(sig.index.min(), fwd.index.min())
    end   = min(sig.index.max(), fwd.index.max()) - pd.Timedelta(days=FORWARD_DAYS + 5)

    if start >= end:
        print(f"  [{signal_name}] Date range too short for IC")
        return pd.DataFrame()

    rebal = pd.date_range(start, end, freq="MS")
    records = []
    for dt in rebal:
        past = sig.index[sig.index <= dt]
        if len(past) == 0:
            continue
        sig_row = sig.loc[past[-1]].dropna()

        future = fwd.index[fwd.index > dt]
        if len(future) == 0:
            continue
        fwd_row = fwd.loc[future[0]].dropna()

        ic = _spearman_ic(sig_row, fwd_row)
        if not np.isnan(ic):
            records.append({
                "date":   dt,
                "signal": signal_name,
                "ic":     ic,
                "period": "OOS" if dt >= IS_CUTOFF else "IS",
                "n_stocks": int(len(sig_row.dropna())),
            })

    result = pd.DataFrame(records)
    if not result.empty:
        is_ic  = result[result["period"] == "IS"]["ic"].mean()
        oos_ic = result[result["period"] == "OOS"]["ic"].mean()
        print(f"  [{signal_name}] {len(result)} months — IS IC={is_ic:+.4f}, OOS IC={oos_ic:+.4f}")
    return result


def evaluate_fear_gauge_ic(
    fear_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Market-timing IC: fear indicator vs equal-weight portfolio 21-day forward return.
    Rolling 12-month Spearman correlation (daily, not cross-sectional).
    """
    # Equal-weight market return (stock universe only)
    stock_cols = [c for c in prices.columns if c in STOCK_UNIVERSE]
    if not stock_cols:
        stock_cols = prices.columns.tolist()
    mkt_fwd = prices[stock_cols].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS).mean(axis=1)

    records = []
    for col in fear_df.columns:
        series = fear_df[col].dropna()
        common = series.index.intersection(mkt_fwd.index)
        if len(common) < 60:
            continue

        s = series.reindex(common)
        m = mkt_fwd.reindex(common)

        monthly = pd.date_range(common.min() + pd.DateOffset(months=12), common.max(), freq="MS")
        monthly = monthly[monthly <= common.max() - pd.Timedelta(days=FORWARD_DAYS + 5)]

        for dt in monthly:
            w_start = dt - pd.DateOffset(months=12)
            w_s = s.loc[w_start:dt].dropna()
            w_m = m.reindex(w_s.index).dropna()
            idx  = w_s.index.intersection(w_m.index)
            if len(idx) < 20:
                continue
            ic = _spearman_ic(w_s.reindex(idx), w_m.reindex(idx))
            if not np.isnan(ic):
                records.append({
                    "date":   dt,
                    "signal": f"fear_{col}",
                    "ic":     ic,
                    "period": "OOS" if dt >= IS_CUTOFF else "IS",
                    "n_stocks": len(stock_cols),
                })

    result = pd.DataFrame(records)
    if not result.empty:
        for sig_name, grp in result.groupby("signal"):
            is_ic  = grp[grp["period"] == "IS"]["ic"].mean()
            oos_ic = grp[grp["period"] == "OOS"]["ic"].mean()
            print(f"  [{sig_name}] {len(grp)} months — IS IC={is_ic:+.4f}, OOS IC={oos_ic:+.4f}")
    return result


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sig, period), grp in ic_df.groupby(["signal", "period"]):
        vals = grp["ic"].dropna()
        if len(vals) == 0:
            continue
        mean_ic = vals.mean()
        std_ic  = vals.std()
        t_stat  = mean_ic / (std_ic / np.sqrt(len(vals))) if len(vals) > 1 and std_ic > 0 else np.nan
        rows.append({
            "signal":    sig,
            "period":    period,
            "mean_ic":   round(mean_ic, 4),
            "ic_std":    round(std_ic, 4),
            "t_stat":    round(t_stat, 3),
            "n_months":  int(len(vals)),
            "pct_pos":   round((vals > 0).mean() * 100, 1),
        })
    return pd.DataFrame(rows).sort_values(["signal", "period"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Report Writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(
    summary: pd.DataFrame,
    notes: dict[str, str],
    fear_df: pd.DataFrame,
    news_panel: pd.DataFrame,
) -> None:
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    lines: list[str] = [
        "# Alternative Data Signals — IC Report",
        f"**Generated:** {today_str}  |  **Canyon v9 — Week 5**",
        "",
        "## Overview",
        "",
        "Four alternative data signals evaluated against 21-day Spearman IC.",
        "",
        "| # | Signal | Source | Coverage | IC Method |",
        "|---|--------|--------|----------|-----------|",
        "| 1 | `news_sentiment` | yfinance headlines + NLP | ~30 days (recent) | Cross-sectional |",
        "| 2 | `google_trends`  | pytrends weekly US search | 5 years weekly | Cross-sectional |",
        "| 3 | `sec_filing_lag` | EDGAR 10-K filings | 2000–present | Cross-sectional |",
        "| 4 | `fear_vix` / `fear_put_call` | CBOE via yfinance | 2000–present | Market-timing |",
        "",
        "---",
        "",
        "## IC Results",
        "",
    ]

    if summary.empty:
        lines.append("*No IC data computed — check individual signal logs.*")
    else:
        lines += [
            "| Signal | Period | Mean IC | Std | t-stat | N Months | % Positive |",
            "|--------|--------|---------|-----|--------|----------|-----------|",
        ]
        for _, row in summary.iterrows():
            if pd.isna(row["t_stat"]):
                flag = "?"
            elif abs(row["t_stat"]) > 2.0:
                flag = "**✓**"
            elif abs(row["t_stat"]) > 1.0:
                flag = "~"
            else:
                flag = "✗"
            lines.append(
                f"| `{row['signal']}` | {row['period']} "
                f"| {row['mean_ic']:+.4f} | {row['ic_std']:.4f} "
                f"| {row['t_stat']:+.3f} {flag} | {int(row['n_months'])} "
                f"| {row['pct_pos']:.1f}% |"
            )
        lines += [
            "",
            "> **t-stat guide:** ✓ = |t|>2.0 (statistically significant), "
            "~ = |t|>1.0 (suggestive), ✗ = weak/noise",
        ]

    lines += ["", "---", "", "## Signal Details", ""]
    for sig_name, note in notes.items():
        lines.append(f"### {sig_name}")
        lines.append(note)
        lines.append("")

    # Current fear gauge snapshot
    if not fear_df.empty:
        lines += ["---", "", "## Current Market Regime (Fear Gauge)", ""]
        latest = fear_df.iloc[-1]
        latest_date = fear_df.index[-1].date()
        if "vix" in latest.index and not pd.isna(latest["vix"]):
            vix_val = latest["vix"]
            vix_regime = "HIGH FEAR" if vix_val > 30 else ("ELEVATED" if vix_val > 20 else "LOW")
            lines.append(f"- **VIX** ({latest_date}): {vix_val:.1f} → {vix_regime}")
        if "put_call" in latest.index and not pd.isna(latest["put_call"]):
            pc_val = latest["put_call"]
            pc_regime = "FEAR (contrarian bullish)" if pc_val > 1.0 else (
                "NEUTRAL" if pc_val > 0.7 else "GREED (contrarian bearish)"
            )
            lines.append(f"- **P/C Ratio** ({latest_date}): {pc_val:.3f} → {pc_regime}")
        lines.append("")

    # Current news sentiment snapshot
    if not news_panel.empty:
        lines += ["---", "", "## Current News Sentiment Snapshot", ""]
        recent = news_panel.iloc[-1].dropna().sort_values(ascending=False)
        if len(recent) > 0:
            lines.append("| Rank | Ticker | Sentiment | Signal |")
            lines.append("|------|--------|-----------|--------|")
            for rank, (tk, val) in enumerate(recent.head(5).items(), 1):
                bar = "+" * min(int(abs(val) * 20), 20)
                lines.append(f"| {rank} | {tk} | {val:+.3f} | `{bar}` |")
            lines.append("")

    lines += [
        "---",
        "",
        "## Integration Roadmap",
        "",
        "**Signals to integrate into step100 ensemble (Week 6+):**",
        "",
        "| Signal | Condition | Action |",
        "|--------|-----------|--------|",
        "| `google_trends` | OOS IC > 0.02 | Add to LightGBM feature set |",
        "| `sec_filing_lag` | OOS IC > 0.02 | Add to LightGBM feature set |",
        "| `fear_vix` / `fear_put_call` | Rolling IC significant | Market-regime filter |",
        "| `news_sentiment` | Requires historical data | Subscribe to premium API |",
        "",
        "**Market regime filter logic (example):**",
        "```python",
        "# Suppress long signals when extreme fear: contrarian, reduce position by 50%",
        "if vix > 40 or put_call > 1.5:",
        "    target_weights *= 0.5  # reduce exposure in crash regime",
        "# Or: VIX-scaled position sizing",
        "vix_scalar = np.clip(20 / vix, 0.5, 1.5)  # normalize to VIX=20 baseline",
        "target_weights *= vix_scalar",
        "```",
        "",
    ]

    REPORT_OUT.write_text("\n".join(lines))
    print(f"\nReport saved → {REPORT_OUT.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5: Alternative Data Integration")
    parser.add_argument("--no-cache",     action="store_true", help="Force re-download all data")
    parser.add_argument("--skip-trends",  action="store_true", help="Skip Google Trends (slow, rate-limited)")
    parser.add_argument("--skip-edgar",   action="store_true", help="Skip SEC EDGAR filing lag")
    parser.add_argument("--skip-news",    action="store_true", help="Skip news sentiment")
    args = parser.parse_args()
    use_cache = not args.no_cache

    print("=" * 70)
    print("Canyon v9 — Step 250: Alternative Data Integration (Week 5)")
    print("=" * 70)

    _setup_scorer()

    # ── Load prices ──────────────────────────────────────────────────────────
    print("\nLoading prices...")
    try:
        prices = load_prices()
        print(
            f"  {prices.shape[1]} tickers, {len(prices)} days "
            f"({prices.index.min().date()} → {prices.index.max().date()})"
        )
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    all_ic: list[pd.DataFrame] = []
    notes: dict[str, str]      = {}
    signal_panels: dict[str, pd.DataFrame] = {}

    # ── Signal 1: News Sentiment ─────────────────────────────────────────────
    if not args.skip_news:
        news_panel, news_note = fetch_news_sentiment(STOCK_UNIVERSE)
        notes["news_sentiment"] = news_note
        if not news_panel.empty:
            signal_panels["news_sentiment"] = news_panel
            ic_df = evaluate_cross_sectional_ic(news_panel, prices, "news_sentiment")
            if not ic_df.empty:
                all_ic.append(ic_df)
    else:
        print("\n[1/4] News Sentiment — SKIPPED")
        news_panel = pd.DataFrame()
        notes["news_sentiment"] = "Skipped via --skip-news flag."

    # ── Signal 2: Google Trends ──────────────────────────────────────────────
    if not args.skip_trends:
        trends_panel, trends_note = fetch_google_trends(STOCK_UNIVERSE, use_cache=use_cache)
        notes["google_trends"] = trends_note
        if not trends_panel.empty:
            signal_panels["google_trends"] = trends_panel
            ic_df = evaluate_cross_sectional_ic(trends_panel, prices, "google_trends")
            if not ic_df.empty:
                all_ic.append(ic_df)
    else:
        print("\n[2/4] Google Trends — SKIPPED")
        notes["google_trends"] = "Skipped via --skip-trends flag."

    # ── Signal 3: SEC EDGAR Filing Lag ───────────────────────────────────────
    if not args.skip_edgar:
        lag_panel, lag_note = fetch_sec_filing_lag(STOCK_UNIVERSE, use_cache=use_cache)
        notes["sec_filing_lag"] = lag_note
        if not lag_panel.empty:
            signal_panels["sec_filing_lag"] = lag_panel
            ic_df = evaluate_cross_sectional_ic(lag_panel, prices, "sec_filing_lag")
            if not ic_df.empty:
                all_ic.append(ic_df)
    else:
        print("\n[3/4] SEC Filing Lag — SKIPPED")
        notes["sec_filing_lag"] = "Skipped via --skip-edgar flag."

    # ── Signal 4: Fear Gauge ─────────────────────────────────────────────────
    fear_df, fear_note = fetch_fear_gauge(use_cache=use_cache)
    notes["fear_gauge"] = fear_note
    if not fear_df.empty:
        ic_df = evaluate_fear_gauge_ic(fear_df, prices)
        if not ic_df.empty:
            all_ic.append(ic_df)
        # Current regime snapshot
        latest = fear_df.dropna(how="all").iloc[-1]
        latest_date = fear_df.dropna(how="all").index[-1].date()
        if "vix" in latest.index and not pd.isna(latest.get("vix")):
            vix = latest["vix"]
            regime = "HIGH FEAR" if vix > 30 else ("ELEVATED" if vix > 20 else "CALM")
            print(f"\n  Current VIX ({latest_date}): {vix:.1f} → {regime}")
        if "put_call" in latest.index and not pd.isna(latest.get("put_call")):
            pc = latest["put_call"]
            pc_r = "FEAR (contrarian bullish)" if pc > 1.0 else ("NEUTRAL" if pc > 0.7 else "GREED")
            print(f"  Current P/C  ({latest_date}): {pc:.3f} → {pc_r}")

    # ── Save combined signal panel ────────────────────────────────────────────
    if signal_panels:
        frames = []
        for sig_name, panel in signal_panels.items():
            melted = panel.stack().reset_index()
            melted.columns = ["date", "ticker", sig_name]
            frames.append(melted.set_index(["date", "ticker"]))
        combined = pd.concat(frames, axis=1).reset_index()
        combined.to_csv(SIGNALS_OUT, index=False)
        print(f"\nSignal panel → {SIGNALS_OUT.name}: {combined.shape[0]} rows, {combined.shape[1]} cols")

    # ── IC Summary ────────────────────────────────────────────────────────────
    if all_ic:
        ic_all = pd.concat(all_ic, ignore_index=True)
        ic_all.to_csv(IC_OUT, index=False)
        summary = summarize_ic(ic_all)

        print("\n" + "=" * 70)
        print("IC SUMMARY (Spearman, 21-day forward return)")
        print("=" * 70)
        if not summary.empty:
            print(summary.to_string(index=False))
    else:
        summary = pd.DataFrame()
        print("\nNo IC data computed.")

    # ── News snapshot ─────────────────────────────────────────────────────────
    if not news_panel.empty:
        recent = news_panel.iloc[-1].dropna().sort_values(ascending=False)
        if len(recent) > 0:
            print(f"\nCurrent News Sentiment (top 5 bullish):")
            for tk, val in recent.head(5).items():
                bar = "+" * min(int(abs(val) * 15), 15)
                print(f"  {tk:<8} {val:+.3f}  {bar}")
            if len(recent) >= 3:
                print("Most bearish:")
                for tk, val in recent.tail(3).items():
                    bar = "-" * min(int(abs(val) * 15), 15)
                    print(f"  {tk:<8} {val:+.3f}  {bar}")

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(summary, notes, fear_df, news_panel)

    print("\n" + "=" * 70)
    print("Week 5 Complete — Alternative Data Integration")
    print("=" * 70)
    print("\nOutputs:")
    for f in [SIGNALS_OUT, IC_OUT, REPORT_OUT, FEAR_CACHE, TRENDS_CACHE, EDGAR_CACHE]:
        if f.exists():
            kb = f.stat().st_size / 1024
            print(f"  {f.name:<40}  {kb:.1f} KB")


if __name__ == "__main__":
    main()
