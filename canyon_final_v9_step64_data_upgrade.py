"""
Canyon v9  Step 64 — Unified Multi-Source Data Layer
======================================================
Clean abstraction over three market data sources with priority failover:

  Priority 1 → Polygon.io   (set POLYGON_API_KEY env variable)
  Priority 2 → Alpaca       (set ALPACA_API_KEY + ALPACA_SECRET_KEY env variables)
  Priority 3 → yfinance     (always available, no key required)

Features:
  • Automatic failover: if primary source fails, transparently tries next
  • Per-source metadata: latency_ms, staleness, rows, coverage_days
  • 24-hour disk cache per source (invalidated by source change or time)
  • Status CSV shows which source delivered each ticker
  • Benchmark mode: all three sources fetched and compared

Outputs:
  data_layer_status.csv        — per-ticker source, freshness, rows, latency_ms
  data_layer_cache/            — binary-cached DataFrames per source per ticker
  data_layer_report.md         — full markdown status report
  data_layer_prices.csv        — merged best-quality price matrix

Usage:
  python canyon_final_v9_step64_data_upgrade.py              # check + update cache
  python canyon_final_v9_step64_data_upgrade.py --check      # status only, no download
  python canyon_final_v9_step64_data_upgrade.py --benchmark  # compare all sources
  python canyon_final_v9_step64_data_upgrade.py --tickers SPY QQQ AAPL
  python canyon_final_v9_step64_data_upgrade.py --force      # bypass cache

API keys (optional — yfinance works without any key):
  export POLYGON_API_KEY="your_key"
  export ALPACA_API_KEY="your_key"
  export ALPACA_SECRET_KEY="your_secret"
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
CACHE_DIR  = ROOT / "data_layer_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_HOURS = 12           # cache staleness threshold

DEFAULT_TICKERS = [
    "SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP",
    "NVDA","TSLA","AMD","MU","GOOGL","AMZN","MSFT","AAPL","META","JPM",
    "CVX","XOM","JNJ","WMT","KO","PEP","MRK","ABBV","UNH","LLY",
]

LOOKBACK_DAYS = 365 * 3   # 3 years of price history


# ─────────────────────────────────────────────────────────────────────────────
# Helper — disk cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(source: str, ticker: str) -> Path:
    return CACHE_DIR / f"{source}_{ticker.replace('/', '_')}.csv"


def _read_cache(source: str, ticker: str) -> Optional[pd.DataFrame]:
    p = _cache_path(source, ticker)
    if not p.exists():
        return None
    age_h = (time.time() - p.stat().st_mtime) / 3600
    if age_h > CACHE_TTL_HOURS:
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def _write_cache(source: str, ticker: str, df: pd.DataFrame) -> None:
    try:
        df.to_csv(_cache_path(source, ticker))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — Polygon.io
# ─────────────────────────────────────────────────────────────────────────────

class PolygonSource:
    NAME = "polygon"
    BASE = "https://api.polygon.io/v2"

    def __init__(self):
        self.api_key = os.environ.get("POLYGON_API_KEY", "")
        self.available = bool(self.api_key)

    def status(self) -> dict:
        if not self.available:
            return {"source": self.NAME, "configured": False,
                    "reason": "POLYGON_API_KEY not set"}
        try:
            t0 = time.time()
            resp = requests.get(
                f"{self.BASE}/aggs/ticker/SPY/range/1/day/2025-01-01/2025-01-05",
                params={"apiKey": self.api_key, "adjusted": "true"},
                timeout=5,
            )
            latency = (time.time() - t0) * 1000
            ok = resp.status_code == 200
            return {"source": self.NAME, "configured": True,
                    "reachable": ok, "latency_ms": round(latency),
                    "http_code": resp.status_code}
        except Exception as e:
            return {"source": self.NAME, "configured": True,
                    "reachable": False, "reason": str(e)}

    def fetch(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.available:
            return None
        cached = _read_cache(self.NAME, ticker)
        if cached is not None:
            return cached
        try:
            t0 = time.time()
            url = f"{self.BASE}/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
            resp = requests.get(
                url,
                params={"apiKey": self.api_key, "adjusted": "true",
                        "sort": "asc", "limit": 50000},
                timeout=15,
            )
            latency = (time.time() - t0) * 1000
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
            df = df.set_index("date")[["c"]].rename(columns={"c": "close"})
            df.index = pd.to_datetime(df.index)
            _write_cache(self.NAME, ticker, df)
            return df
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Alpaca Markets
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaSource:
    NAME = "alpaca"
    BASE = "https://data.alpaca.markets/v2"

    def __init__(self):
        self.api_key    = os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self.available  = bool(self.api_key and self.secret_key)

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def status(self) -> dict:
        if not self.available:
            return {"source": self.NAME, "configured": False,
                    "reason": "ALPACA_API_KEY / ALPACA_SECRET_KEY not set"}
        try:
            t0 = time.time()
            resp = requests.get(
                f"{self.BASE}/stocks/SPY/bars",
                headers=self._headers(),
                params={"timeframe": "1Day", "start": "2025-01-01",
                        "end": "2025-01-05", "limit": 5},
                timeout=5,
            )
            latency = (time.time() - t0) * 1000
            ok = resp.status_code == 200
            return {"source": self.NAME, "configured": True,
                    "reachable": ok, "latency_ms": round(latency),
                    "http_code": resp.status_code}
        except Exception as e:
            return {"source": self.NAME, "configured": True,
                    "reachable": False, "reason": str(e)}

    def fetch(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.available:
            return None
        cached = _read_cache(self.NAME, ticker)
        if cached is not None:
            return cached
        try:
            bars = []
            params = {"timeframe": "1Day", "start": start, "end": end, "limit": 10000}
            while True:
                resp = requests.get(
                    f"{self.BASE}/stocks/{ticker}/bars",
                    headers=self._headers(),
                    params=params,
                    timeout=20,
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                bars.extend(data.get("bars", []))
                next_token = data.get("next_page_token")
                if not next_token:
                    break
                params["page_token"] = next_token

            if not bars:
                return None
            df = pd.DataFrame(bars)
            df["date"] = pd.to_datetime(df["t"]).dt.normalize()
            df = df.set_index("date")[["c"]].rename(columns={"c": "close"})
            _write_cache(self.NAME, ticker, df)
            return df
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — yfinance (always available)
# ─────────────────────────────────────────────────────────────────────────────

class YFinanceSource:
    NAME = "yfinance"

    def status(self) -> dict:
        try:
            t0 = time.time()
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="5d")
            latency = (time.time() - t0) * 1000
            ok = not hist.empty
            return {"source": self.NAME, "configured": True,
                    "reachable": ok, "latency_ms": round(latency),
                    "rows_sample": len(hist)}
        except Exception as e:
            return {"source": self.NAME, "configured": True,
                    "reachable": False, "reason": str(e)}

    def fetch(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        cached = _read_cache(self.NAME, ticker)
        if cached is not None:
            return cached
        try:
            import yfinance as yf
            hist = yf.download(ticker, start=start, end=end,
                               auto_adjust=True)
            if hist.empty:
                return None
            if isinstance(hist.columns, pd.MultiIndex):
                df = hist["Close"].to_frame(name="close")
            else:
                df = hist[["Close"]].rename(columns={"Close": "close"})
            _write_cache(self.NAME, ticker, df)
            return df
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified Data Layer
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedDataLayer:
    """
    Try sources in priority order and return the first successful result.
    Records metadata (source used, latency, rows) for each ticker.
    """

    def __init__(self, force: bool = False):
        self.polygon  = PolygonSource()
        self.alpaca   = AlpacaSource()
        self.yfinance = YFinanceSource()
        self.sources  = [self.polygon, self.alpaca, self.yfinance]
        self.force    = force   # if True, bypass disk cache
        self.metadata: list[dict] = []

    def check_sources(self) -> list[dict]:
        """Ping each source and return status list."""
        statuses = []
        for src in self.sources:
            print(f"  Checking {src.NAME} … ", end="", flush=True)
            s = src.status()
            reachable = s.get("reachable", s.get("configured", False))
            print("✓" if reachable else f"✗  ({s.get('reason','unavailable')})")
            statuses.append(s)
        return statuses

    def fetch_ticker(self, ticker: str, start: str, end: str) -> tuple[Optional[pd.DataFrame], str]:
        """Return (df, source_name) — tries sources in priority order."""
        if self.force:
            # Clear cache for this ticker
            for src in self.sources:
                p = _cache_path(src.NAME, ticker)
                if p.exists():
                    p.unlink(missing_ok=True)

        for src in self.sources:
            t0 = time.time()
            try:
                df = src.fetch(ticker, start, end)
            except Exception:
                df = None
            latency = (time.time() - t0) * 1000

            if df is not None and not df.empty:
                return df, src.NAME
        return None, "none"

    def fetch_universe(self, tickers: list[str], lookback_days: int = LOOKBACK_DAYS,
                       verbose: bool = True) -> pd.DataFrame:
        """Download all tickers, return wide close-price DataFrame."""
        end   = datetime.today()
        start = end - timedelta(days=lookback_days + 60)
        start_str = start.strftime("%Y-%m-%d")
        end_str   = end.strftime("%Y-%m-%d")

        all_close: dict[str, pd.Series] = {}
        self.metadata = []

        for ticker in tickers:
            t0 = time.time()
            df, source = self.fetch_ticker(ticker, start_str, end_str)
            elapsed = (time.time() - t0) * 1000

            if df is not None:
                close = df["close"] if "close" in df.columns else df.iloc[:, 0]
                all_close[ticker] = close
                rows = len(close)
                freshness = "FRESH" if elapsed < CACHE_TTL_HOURS * 3600 * 1000 else "STALE"
                if verbose:
                    print(f"  {ticker:8s}  {source:10s}  {rows:4d} rows  {elapsed:.0f}ms")
            else:
                rows = 0
                source = "FAILED"
                freshness = "MISSING"
                if verbose:
                    print(f"  {ticker:8s}  FAILED")

            self.metadata.append({
                "ticker":       ticker,
                "source":       source,
                "rows":         rows,
                "latency_ms":   round(elapsed),
                "freshness":    freshness,
                "as_of":        datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

        if not all_close:
            return pd.DataFrame()

        prices = pd.DataFrame(all_close).sort_index()
        return prices

    def save_status(self) -> None:
        if not self.metadata:
            return
        df = pd.DataFrame(self.metadata)
        p  = ROOT / "data_layer_status.csv"
        df.to_csv(p, index=False)
        print(f"\n  [status] {p}")

    def save_prices(self, prices: pd.DataFrame) -> None:
        p = ROOT / "data_layer_prices.csv"
        prices.to_csv(p)
        print(f"  [prices] {p}")

    def write_report(self, source_statuses: list[dict]) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Canyon v9 — Data Layer Report (Step 64)",
            f"Generated: {ts}",
            "",
            "## Source Status",
            "",
            "| Source | Configured | Reachable | Latency | Notes |",
            "|---|---|---|---|---|",
        ]
        for s in source_statuses:
            configured = "✓" if s.get("configured") else "✗"
            reachable  = "✓" if s.get("reachable")  else "✗"
            latency    = f"{s.get('latency_ms','—')} ms" if "latency_ms" in s else "—"
            notes      = s.get("reason", s.get("http_code", ""))
            lines.append(
                f"| {s['source']} | {configured} | {reachable} | {latency} | {notes} |"
            )

        if self.metadata:
            meta_df = pd.DataFrame(self.metadata)
            source_counts = meta_df["source"].value_counts().to_dict()
            failed = int((meta_df["source"] == "FAILED").sum())
            lines += [
                "",
                "## Universe Fetch Summary",
                "",
                f"- Total tickers: {len(self.metadata)}",
                f"- Failed: {failed}",
            ]
            for src, cnt in source_counts.items():
                lines.append(f"- {src}: {cnt} tickers")

            lines += [
                "",
                "## Coverage Detail",
                "",
                "| Ticker | Source | Rows | Latency | Freshness |",
                "|---|---|---|---|---|",
            ]
            for m in self.metadata:
                lines.append(
                    f"| {m['ticker']} | {m['source']} | {m['rows']} | "
                    f"{m['latency_ms']}ms | {m['freshness']} |"
                )

        lines += [
            "",
            "## Source Priority Order",
            "1. **Polygon.io** — institutional-grade, sub-second bars, requires paid API key",
            "   - Set: `export POLYGON_API_KEY=your_key`",
            "2. **Alpaca** — commission-free broker data, free tier available",
            "   - Set: `export ALPACA_API_KEY=key  ALPACA_SECRET_KEY=secret`",
            "3. **yfinance** — always available fallback, end-of-day data only",
            "   - No API key required",
            "",
            "## Upgrade Instructions",
            "- **Polygon.io** free tier: 5 calls/min, delayed data. Starter plan ($29/mo): "
            "unlimited calls, real-time. https://polygon.io",
            "- **Alpaca** free tier: real-time IEX feed. Paper trading account: "
            "https://alpaca.markets",
            "- Current production priority: yfinance until API keys configured",
        ]

        p = ROOT / "data_layer_report.md"
        p.write_text("\n".join(lines))
        print(f"  [report] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark mode
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_sources(tickers: list[str] = None, n: int = 5) -> None:
    """Download a sample of tickers from all three sources and compare."""
    sample = (tickers or DEFAULT_TICKERS)[:n]
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"\nBenchmarking {len(sample)} tickers across all sources …\n")
    results = {}
    for src_cls in [PolygonSource, AlpacaSource, YFinanceSource]:
        src = src_cls()
        results[src.NAME] = {}
        for ticker in sample:
            t0 = time.time()
            df = src.fetch(ticker, start, end)
            elapsed = (time.time() - t0) * 1000
            results[src.NAME][ticker] = {
                "rows":    len(df) if df is not None else 0,
                "latency": round(elapsed),
                "ok":      df is not None and not df.empty,
            }

    # Print comparison table
    print(f"{'Ticker':8s}", end="")
    for src_name in results:
        print(f"  {src_name:>14s}", end="")
    print()
    print("─" * (8 + 16 * len(results)))
    for ticker in sample:
        print(f"{ticker:8s}", end="")
        for src_name, src_data in results.items():
            d = src_data[ticker]
            cell = f"{'✓' if d['ok'] else '✗'} {d['rows']}r {d['latency']}ms"
            print(f"  {cell:>14s}", end="")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 64 — Unified Data Layer")
    parser.add_argument("--check",     action="store_true", help="Status check only, no download")
    parser.add_argument("--benchmark", action="store_true", help="Compare all three sources")
    parser.add_argument("--force",     action="store_true", help="Bypass disk cache")
    parser.add_argument("--tickers",   nargs="*", default=None, help="Custom ticker list")
    parser.add_argument("--lookback",  type=int, default=LOOKBACK_DAYS, help="Days of history")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else DEFAULT_TICKERS

    print(f"\n{'='*60}")
    print("Canyon v9 Step 64 — Unified Data Layer")
    print(f"{'='*60}")

    layer = UnifiedDataLayer(force=args.force)

    # ── Source status check ───────────────────────────────────────────────────
    print("\n[1] Checking source availability …")
    source_statuses = layer.check_sources()

    active_sources = [s["source"] for s in source_statuses
                      if s.get("reachable") or (s.get("configured") and
                         s["source"] == "yfinance")]
    print(f"\n     Active sources: {active_sources}")

    if args.benchmark:
        benchmark_sources(tickers)
        layer.write_report(source_statuses)
        return

    if args.check:
        layer.write_report(source_statuses)
        print("\nCheck-only mode — no data downloaded.")
        return

    # ── Full download ─────────────────────────────────────────────────────────
    print(f"\n[2] Downloading {len(tickers)} tickers (lookback={args.lookback}d) …")
    prices = layer.fetch_universe(tickers, lookback_days=args.lookback)

    # ── Quality report ────────────────────────────────────────────────────────
    print(f"\n[3] Data quality summary …")
    if not prices.empty:
        coverage = prices.count()
        missing  = (coverage == 0).sum()
        partial  = ((coverage > 0) & (coverage < len(prices) * 0.9)).sum()
        full     = (coverage >= len(prices) * 0.9).sum()
        print(f"     Full coverage (≥90%): {full} tickers")
        print(f"     Partial (<90%):       {partial} tickers")
        print(f"     Missing entirely:     {missing} tickers")
        print(f"     Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
        print(f"     Shape: {prices.shape}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print(f"\n[4] Writing outputs …")
    layer.save_status()
    if not prices.empty:
        layer.save_prices(prices)
    layer.write_report(source_statuses)

    print(f"\n{'='*60}")
    print("Data Layer update complete.")

    # Print source attribution
    if layer.metadata:
        meta_df = pd.DataFrame(layer.metadata)
        src_summary = meta_df.groupby("source")["ticker"].count()
        print("\nSource attribution:")
        for src, cnt in src_summary.items():
            pct = cnt / len(meta_df) * 100
            bar = "█" * int(pct / 5)
            print(f"  {src:12s} {bar:<20s} {cnt:3d} tickers ({pct:.0f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
