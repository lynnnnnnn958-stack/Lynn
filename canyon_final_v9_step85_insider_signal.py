#!/usr/bin/env python3
"""
Canyon v9 — Step 85: SEC Form 4 Insider Buying Signal
======================================================
Fetches insider purchase data from SEC EDGAR (completely free, no API key required).

Core logic (from Livermore):
  "Smart money knows the inside story. When CEO/CFO buys with their own money,
   not from options, not from incentive plans, but direct open-market purchases
   — this is the most direct bullish signal."

Data source:
  SEC EDGAR public API
  - company_tickers.json → Ticker → CIK mapping
  - submissions/CIK{}.json → recent Form 4 filing list
  - Archives/edgar/... → Form 4 XML parsing of buy/sell transactions

Scoring logic:
  buy_pressure = buy_count in past 60 days × (1 + exec_buy_count × 0.5)
  net_direction = (buy_value - sell_value×0.3) / (total_trade_value + 1)
  insider_score = buy_pressure × max(net_direction, 0)

Note: sells discounted to 0.3 weight (large sells may be tax or diversification;
      buys are the genuine bullish signal).

Output:
  insider_signal_scores.csv  — rank_insider (0-100), insider_signal
  insider_signal_report.md   — report

Usage:
  python3 canyon_final_v9_step85_insider_signal.py [--top N]
"""

import argparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
ML_SCORES   = ROOT / "regime_ml_scores.csv"
OUT_SCORES  = ROOT / "insider_signal_scores.csv"
OUT_REPORT  = ROOT / "insider_signal_report.md"
CIK_CACHE   = ROOT / "edgar_cik_cache.json"
INS_CACHE   = ROOT / "insider_raw_cache.csv"
CACHE_TTL_H = 24 * 3    # 3-day cache (SEC allows; no need to refresh daily)
DEFAULT_TOP = 200
MAX_WORKERS = 3          # EDGAR rate limit is 10 req/s; use conservative setting

# SEC EDGAR requires User-Agent header (returns 403 otherwise)
EDGAR_HEADERS = {
    "User-Agent": "Canyon Quant Research canyon-research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fetch ticker pool
# ─────────────────────────────────────────────────────────────────────────────

def get_universe(top_n: int) -> list[str]:
    if ML_SCORES.exists():
        df  = pd.read_csv(ML_SCORES)
        col = "predicted_score" if "predicted_score" in df.columns else df.columns[-1]
        return df.sort_values(col, ascending=False)["ticker"].dropna().tolist()[:top_n]
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AMD","TSLA","AVGO","COST"][:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ticker → CIK mapping
# ─────────────────────────────────────────────────────────────────────────────

def load_cik_map() -> dict[str, str]:
    """Load Ticker→CIK mapping from SEC, cache to local JSON."""
    import json, requests

    if CIK_CACHE.exists():
        age_h = (datetime.now().timestamp() - CIK_CACHE.stat().st_mtime) / 3600
        if age_h < 24 * 7:   # 1-week cache
            return json.loads(CIK_CACHE.read_text())

    print("  [EDGAR] Loading Ticker→CIK mapping ...")
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers={**EDGAR_HEADERS, "Host": "www.sec.gov"}, timeout=15)
        data = resp.json()
        mapping = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }
        CIK_CACHE.write_text(json.dumps(mapping))
        print(f"  [EDGAR] {len(mapping)} CIK mappings cached")
        return mapping
    except Exception as e:
        print(f"  [EDGAR] CIK mapping failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fetch insider data for a single ticker
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_insider(ticker: str, cik: str, cutoff_date: str) -> dict:
    """
    Fetch 60-day Form 4 buy/sell summary for one ticker from EDGAR.
    Returns dict with: buy_count, sell_count, buy_value, sell_value, exec_buy_count
    """
    import requests, xml.etree.ElementTree as ET

    base = {
        "ticker": ticker, "buy_count": 0, "sell_count": 0,
        "buy_value": 0.0, "sell_value": 0.0, "exec_buy_count": 0,
    }

    try:
        # Fetch filing history
        url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if resp.status_code != 200:
            return {**base, "error": f"HTTP {resp.status_code}"}

        data     = resp.json()
        recent   = data.get("filings", {}).get("recent", {})
        forms    = recent.get("form", [])
        dates    = recent.get("filingDate", [])
        accs     = recent.get("accessionNumber", [])
        docs     = recent.get("primaryDocument", [])

        # Filter recent Form 4 filings
        form4s = [
            {"date": d, "acc": a.replace("-", ""), "doc": doc, "cik_int": int(cik)}
            for frm, d, a, doc in zip(forms, dates, accs, docs)
            if frm == "4" and d >= cutoff_date
        ]

        if not form4s:
            return base

        # Parse each Form 4 XML (up to 10, to avoid timeout)
        for filing in form4s[:10]:
            acc     = filing["acc"]
            doc     = filing["doc"]
            cik_int = filing["cik_int"]
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"

            try:
                time.sleep(0.12)   # respect EDGAR rate limit
                xresp = requests.get(
                    xml_url,
                    headers={**EDGAR_HEADERS, "Host": "www.sec.gov"},
                    timeout=12,
                )
                if xresp.status_code != 200:
                    continue

                root = ET.fromstring(xresp.text)

                # Determine if filer is an executive (CEO/CFO/Director)
                title_elem = root.find(".//officerTitle")
                title      = (title_elem.text or "").lower() if title_elem is not None else ""
                is_exec    = any(k in title for k in
                                 ["chief", "ceo", "cfo", "coo", "president",
                                  "director", "chairman"])

                # Parse non-derivative transactions
                for trans in root.findall(".//nonDerivativeTransaction"):
                    code_elem   = trans.find(".//transactionCode")
                    shares_elem = trans.find(".//transactionShares/value")
                    price_elem  = trans.find(".//transactionPricePerShare/value")
                    if code_elem is None:
                        continue

                    code = code_elem.text
                    try:
                        shares = float(shares_elem.text) if shares_elem is not None else 0
                        price  = float(price_elem.text)  if price_elem  is not None else 0
                        value  = shares * price
                    except (ValueError, TypeError):
                        value = 0

                    if code == "P":        # Purchase
                        base["buy_count"]  += 1
                        base["buy_value"]  += value
                        if is_exec:
                            base["exec_buy_count"] += 1
                    elif code == "S":      # Sale
                        base["sell_count"] += 1
                        base["sell_value"] += value

            except Exception:
                continue

        base["buy_value"]  = round(base["buy_value"])
        base["sell_value"] = round(base["sell_value"])
        return base

    except Exception as e:
        return {**base, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch load (with cache)
# ─────────────────────────────────────────────────────────────────────────────

def load_insider_data(tickers: list[str]) -> pd.DataFrame:
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    # Cache check
    if INS_CACHE.exists():
        age_h = (datetime.now().timestamp() - INS_CACHE.stat().st_mtime) / 3600
        if age_h < CACHE_TTL_H:
            cached  = pd.read_csv(INS_CACHE)
            covered = set(cached["ticker"].tolist())
            missing = [t for t in tickers if t not in covered]
            if not missing:
                print(f"  Insider cache hit: {len(cached)} tickers ({age_h:.0f}h old)")
                return cached[cached["ticker"].isin(tickers)]
            tickers_to_fetch = missing
            print(f"  Cache: {len(covered)} tickers, fetching {len(missing)} new ...")
        else:
            tickers_to_fetch = tickers
            print(f"  Cache expired ({age_h:.0f}h), re-fetching ...")
    else:
        tickers_to_fetch = tickers
        print(f"  First fetch of insider data for {len(tickers)} tickers ...")

    cik_map = load_cik_map()
    if not cik_map:
        print("  WARNING: CIK mapping empty, insider signal unavailable")
        return pd.DataFrame()

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_fetch_insider, t, cik_map[t], cutoff): t
            for t in tickers_to_fetch
            if t in cik_map
        }
        skipped = [t for t in tickers_to_fetch if t not in cik_map]
        if skipped:
            print(f"  (skipped {len(skipped)} tickers with no CIK found)")

        for fut in as_completed(futures):
            res = fut.result()
            if "error" not in res:
                rows.append(res)
            done += 1
            if done % 20 == 0:
                print(f"  … {done}/{len(futures)}")

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Merge with cache
    if INS_CACHE.exists() and tickers_to_fetch != tickers:
        old_df   = pd.read_csv(INS_CACHE)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="ticker", keep="last")
    else:
        combined = new_df

    if not combined.empty:
        combined.to_csv(INS_CACHE, index=False)

    return combined[combined["ticker"].isin(tickers)] if not combined.empty else combined


# ─────────────────────────────────────────────────────────────────────────────
# 5. Signal computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    for c in ["buy_count", "sell_count", "buy_value", "sell_value", "exec_buy_count"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    # Buy pressure: count × executive multiplier
    df["buy_pressure"] = df["buy_count"] * (1 + df["exec_buy_count"] * 0.5)

    # Net direction: buys minus discounted sells (sells may be tax-driven, discounted 70%)
    total = df["buy_value"] + df["sell_value"] * 0.3 + 1
    df["net_direction"] = ((df["buy_value"] - df["sell_value"] * 0.3) / total).clip(-1, 1)

    # Combined insider score
    df["insider_raw"] = df["buy_pressure"] * df["net_direction"].clip(0, None)

    # Cross-sectional rank (0-100)
    valid = df["insider_raw"] > 0
    if valid.sum() >= 5:
        df.loc[valid, "rank_insider"] = (
            df.loc[valid, "insider_raw"].rank(pct=True) * 100
        )
    df["rank_insider"] = df["rank_insider"].fillna(0)

    # Signal label
    df["insider_signal"] = df["rank_insider"].apply(
        lambda r: "INSIDER_BUY"    if r >= 80
        else ("INSIDER_WATCH" if r >= 60
        else ("INSIDER_SELL"  if (df.loc[df["rank_insider"] == r, "sell_count"].values[0] > 2
                                  and df.loc[df["rank_insider"] == r, "buy_count"].values[0] == 0)
        else "NEUTRAL"))
        if len(df.loc[df["rank_insider"] == r]) > 0 else "NEUTRAL"
    )

    return df


def compute_signals_safe(raw: pd.DataFrame) -> pd.DataFrame:
    """Safe signal computation variant, avoids indexing issues in apply."""
    df = raw.copy()

    for c in ["buy_count", "sell_count", "buy_value", "sell_value", "exec_buy_count"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    df["buy_pressure"] = df["buy_count"] * (1 + df["exec_buy_count"] * 0.5)

    total = df["buy_value"] + df["sell_value"] * 0.3 + 1
    df["net_direction"] = ((df["buy_value"] - df["sell_value"] * 0.3) / total).clip(-1, 1)

    df["insider_raw"] = df["buy_pressure"] * df["net_direction"].clip(0, None)

    # Cross-sectional rank
    valid = df["insider_raw"].notna() & (df["insider_raw"] > 0)
    df["rank_insider"] = 0.0
    if valid.sum() >= 3:
        df.loc[valid, "rank_insider"] = (
            df.loc[valid, "insider_raw"].rank(pct=True) * 100
        ).clip(0, 100)

    # Signal label (vectorized)
    conds = [
        df["rank_insider"] >= 80,
        df["rank_insider"] >= 60,
        (df["sell_count"] > 2) & (df["buy_count"] == 0),
    ]
    labels = ["INSIDER_BUY", "INSIDER_WATCH", "INSIDER_SELL"]
    df["insider_signal"] = np.select(conds, labels, default="NEUTRAL")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame) -> None:
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    buys     = df[df["insider_signal"] == "INSIDER_BUY"].sort_values("rank_insider", ascending=False)
    sells    = df[df["insider_signal"] == "INSIDER_SELL"].sort_values("rank_insider")
    watchers = df[df["insider_signal"] == "INSIDER_WATCH"].sort_values("rank_insider", ascending=False)

    lines = [
        "# Canyon v9 — Step 85: SEC Form 4 Insider Purchase Signal",
        f"Generated: {now}  |  Coverage: {len(df)} tickers",
        "",
        "## Core Logic",
        "Insiders (CEO/CFO/Director) buying with their own money = the most direct bullish signal.",
        "Score = buy pressure × net direction (sells discounted 70%)",
        "",
        "## Top Insider Purchases",
        "| # | Ticker | Rank | Buy Count | Exec Buys | Buy Value ($) | Signal |",
        "|---|------|------|---------|---------|------------|------|",
    ]
    for i, (_, r) in enumerate(buys.head(15).iterrows(), 1):
        lines.append(
            f"| {i} | **{r['ticker']}** | {r['rank_insider']:.0f} "
            f"| {int(r['buy_count'])} | {int(r['exec_buy_count'])} "
            f"| ${r['buy_value']:,.0f} | {r['insider_signal']} |"
        )

    lines += [
        "",
        "## Summary Statistics",
        f"- INSIDER_BUY  (rank>=80): {len(buys)} tickers",
        f"- INSIDER_WATCH(rank>=60): {len(watchers)} tickers",
        f"- INSIDER_SELL (net sell): {len(sells)} tickers",
        "",
        "## Usage Guidelines",
        "- **Executive buy + high SUE rank** = strongest combination (inside and outside aligned bullish)",
        "- **Buy value > $1M** = high confidence (not a token small purchase)",
        "- **Multiple buyers simultaneously** = collective conviction (stronger than single buyer)",
        "- **Primarily selling** = contrarian signal (combine with BEAR mechanism for short candidates)",
    ]

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main(top_n: int = DEFAULT_TOP) -> None:
    t0 = datetime.now()
    print()
    print("=" * 60)
    print("Canyon v9 — Step 85: SEC Form 4 Insider Purchase Signal")
    print("=" * 60)

    print(f"\n[1/4] Loading ticker pool (top {top_n}) ...")
    tickers = get_universe(top_n)
    print(f"  {len(tickers)} tickers")

    print("\n[2/4] Fetching insider data (SEC EDGAR) ...")
    raw = load_insider_data(tickers)
    if raw.empty:
        print("  No data obtained, skipping")
        return
    active = (raw["buy_count"] > 0).sum()
    print(f"  {active}/{len(raw)} tickers have buy records")

    print("\n[3/4] Computing signals ...")
    result = compute_signals_safe(raw)

    buy_cnt  = (result["insider_signal"] == "INSIDER_BUY").sum()
    sell_cnt = (result["insider_signal"] == "INSIDER_SELL").sum()
    print(f"  INSIDER_BUY: {buy_cnt}  INSIDER_SELL: {sell_cnt}")

    # Save
    out_cols = ["ticker", "buy_count", "sell_count", "buy_value", "sell_value",
                "exec_buy_count", "buy_pressure", "net_direction",
                "insider_raw", "rank_insider", "insider_signal"]
    out = result[[c for c in out_cols if c in result.columns]]
    out.to_csv(OUT_SCORES, index=False)
    print(f"  Saved: {OUT_SCORES.name} ({len(out)} rows)")

    # Top candidates
    top = result[result["insider_signal"] == "INSIDER_BUY"].sort_values(
        "rank_insider", ascending=False).head(10)
    if not top.empty:
        print()
        print("  Top 10 insider buy candidates:")
        for _, r in top.iterrows():
            print(f"    {r['ticker']:6s}  buys={int(r['buy_count'])}  "
                  f"exec={int(r['exec_buy_count'])}  "
                  f"${r['buy_value']:>10,.0f}  rank={r['rank_insider']:.0f}")

    print("\n[4/4] Writing report ...")
    write_report(result)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nComplete, elapsed {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 insider purchase signal")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"number of tickers to analyze (default {DEFAULT_TOP})")
    args = parser.parse_args()
    main(top_n=args.top)
