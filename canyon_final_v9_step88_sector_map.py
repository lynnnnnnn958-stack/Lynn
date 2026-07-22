#!/usr/bin/env python3
"""
canyon_final_v9_step88_sector_map.py
=====================================
Build and maintain sector_map.csv — GICS sector mapping for every ticker
in the Canyon v9 universe (~495 S&P 500 names).

Sources (priority order):
  1. Existing sector_map.csv          (incremental: only fetch missing / "Other")
  2. regime_ml_scores.csv             (sector column from Step 77, ~211 stocks)
  3. fundamental_features_cache.json  (Step 68 sector field)
  4. yfinance .info.get("sector")     (live fetch for remaining unknowns)

Output:
  sector_map.csv   columns: ticker, sector, source

Usage:
  python canyon_final_v9_step88_sector_map.py              # incremental
  python canyon_final_v9_step88_sector_map.py --refresh    # re-fetch all from yfinance
  python canyon_final_v9_step88_sector_map.py --dry-run    # show plan, don't write
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# Known "bad" values that need replacing
BAD_SECTOR_VALUES = {"Other", "", "N/A", "nan", "None"}

# Canonical GICS sector names (11 top-level sectors)
GICS_SECTORS = {
    "Technology", "Health Care", "Financials", "Consumer Discretionary",
    "Communication Services", "Industrials", "Consumer Staples", "Energy",
    "Utilities", "Real Estate", "Materials",
}

# Normalisation map: every known alias → canonical GICS name.
# yfinance and regime_ml use different abbreviations / industry-level names.
SECTOR_NORM: dict[str, str] = {
    # Technology variants
    "Tech":                    "Technology",
    "Information Technology":  "Technology",
    "Semiconductors":          "Technology",   # GICS industry → parent sector
    # Health Care variants
    "Healthcare":              "Health Care",
    "Health":                  "Health Care",
    # Financials variants
    "Financial Services":      "Financials",
    "Financial":               "Financials",
    # Consumer Discretionary variants
    "Consumer Cyclical":       "Consumer Discretionary",
    "Consumer Disc":           "Consumer Discretionary",
    # Consumer Staples variants
    "Consumer Defensive":      "Consumer Staples",
    # Materials variants
    "Basic Materials":         "Materials",
    # Real Estate variants
    "REITs":                   "Real Estate",
    "REIT":                    "Real Estate",
}


def _normalise_sector(raw: str) -> str:
    """Map any sector alias to its canonical GICS name."""
    s = str(raw).strip()
    return SECTOR_NORM.get(s, s)   # return canonical if known, else unchanged

# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — regime_ml_scores.csv
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_regime_ml() -> dict[str, tuple[str, str]]:
    """
    Returns {ticker: (sector, "regime_ml")} for all tickers with a
    non-"Other" sector in regime_ml_scores.csv.
    """
    path = ROOT / "regime_ml_scores.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns or "sector" not in df.columns:
            return {}
        df = df.drop_duplicates("ticker")
        result = {}
        for _, row in df.iterrows():
            t = str(row["ticker"]).strip()
            s = str(row.get("sector", "")).strip()
            if t and s and s not in BAD_SECTOR_VALUES:
                result[t] = (s, "regime_ml")
        print(f"  [regime_ml] {len(result)} sectors loaded")
        return result
    except Exception as e:
        print(f"  [regime_ml] error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — fundamental_features_cache.json
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_fundamentals() -> dict[str, tuple[str, str]]:
    """Returns {ticker: (sector, "fundamentals")} from step68 cache."""
    path = ROOT / "fundamental_features_cache.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        data = raw.get("data", {}) if isinstance(raw, dict) else raw
        result = {}
        for t, rec in data.items():
            if not isinstance(rec, dict):
                continue
            s = str(rec.get("sector", "")).strip()
            if s and s not in BAD_SECTOR_VALUES:
                result[str(t).strip()] = (s, "fundamentals")
        print(f"  [fundamentals] {len(result)} sectors loaded")
        return result
    except Exception as e:
        print(f"  [fundamentals] error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — load universe
# ─────────────────────────────────────────────────────────────────────────────

def _load_universe() -> list[str]:
    """Return list of all tickers in the Canyon v9 universe."""
    regime_path = ROOT / "regime_ml_scores.csv"
    if regime_path.exists():
        try:
            df = pd.read_csv(regime_path)
            if "ticker" in df.columns:
                return df["ticker"].dropna().drop_duplicates().tolist()
        except Exception:
            pass

    alpha_path = ROOT / "alpha_scores.csv"
    if alpha_path.exists():
        try:
            df = pd.read_csv(alpha_path)
            if "ticker" in df.columns:
                return df["ticker"].dropna().drop_duplicates().tolist()
        except Exception:
            pass

    price_path = ROOT / "sp500_price_cache.csv"
    if price_path.exists():
        try:
            cols = pd.read_csv(price_path, nrows=0).columns.tolist()
            return [c for c in cols if c not in ("", "Date", "Unnamed: 0")]
        except Exception:
            pass

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — yfinance live fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_one_sector(ticker: str) -> tuple[str, str, str]:
    """
    Returns (ticker, sector, source).
    Uses yfinance .info.get("sector") — not fast_info (doesn't have sector).
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        s = str(info.get("sector", "")).strip()
        if s and s not in BAD_SECTOR_VALUES:
            return ticker, s, "yfinance"
    except Exception:
        pass
    return ticker, "Other", "unknown"


def _fetch_sectors_yf(
    tickers: list[str],
    max_workers: int = 10,
) -> dict[str, tuple[str, str]]:
    """
    Batch-fetch sectors from yfinance for a list of tickers.
    Returns {ticker: (sector, "yfinance")}.
    """
    results: dict[str, tuple[str, str]] = {}
    total = len(tickers)
    if total == 0:
        return results

    print(f"  [yfinance] Fetching {total} sectors (workers={max_workers}) …")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_fetch_one_sector, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            ticker_r, sector_r, src_r = fut.result()
            results[ticker_r] = (sector_r, src_r)
            done += 1
            if done % 50 == 0 or done == total:
                pct = done / total * 100
                elapsed = time.time() - t0
                print(f"    {done}/{total} ({pct:.0f}%)  {elapsed:.0f}s elapsed")

    known = sum(1 for _, (s, _) in results.items() if s not in BAD_SECTOR_VALUES)
    print(f"  [yfinance] Done — {known}/{total} sectors found "
          f"({time.time()-t0:.0f}s)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sector_map(refresh: bool = False, dry_run: bool = False) -> pd.DataFrame:
    """
    Build or update sector_map.csv.

    Parameters
    ----------
    refresh  : bool  Force re-fetch all tickers from yfinance (ignore existing).
    dry_run  : bool  Print plan but don't write files.
    """
    print("\n" + "=" * 60)
    print("Canyon v9  Step 88 — Sector Map Builder")
    print("=" * 60)

    # ── Step 1: Load universe ────────────────────────────────────────────────
    print("\n[1/4] Loading universe …")
    universe = _load_universe()
    print(f"      {len(universe)} tickers in universe")

    if not universe:
        print("  ERROR: empty universe — run Step 76/77 first")
        return pd.DataFrame()

    # ── Step 2: Collect sectors from static sources ──────────────────────────
    print("\n[2/4] Loading sectors from static sources …")
    sector_data: dict[str, tuple[str, str]] = {}   # ticker → (sector, source)

    # Load existing sector_map.csv (highest trust if not refreshing)
    existing_map_path = ROOT / "sector_map.csv"
    if existing_map_path.exists() and not refresh:
        try:
            existing = pd.read_csv(existing_map_path)
            if {"ticker", "sector"}.issubset(existing.columns):
                src_col = "source" if "source" in existing.columns else None
                for _, row in existing.iterrows():
                    t = str(row["ticker"]).strip()
                    s = str(row["sector"]).strip()
                    src = str(row[src_col]).strip() if src_col else "existing"
                    if t and s and s not in BAD_SECTOR_VALUES:
                        sector_data[t] = (s, src)
                print(f"  [existing] {len(sector_data)} sectors from sector_map.csv")
        except Exception as e:
            print(f"  [existing] load error: {e}")

    # regime_ml_scores.csv
    for t, val in _load_from_regime_ml().items():
        if refresh or t not in sector_data or sector_data[t][0] in BAD_SECTOR_VALUES:
            sector_data[t] = val

    # fundamental_features_cache.json
    for t, val in _load_from_fundamentals().items():
        if refresh or t not in sector_data or sector_data[t][0] in BAD_SECTOR_VALUES:
            sector_data[t] = val

    # Summary of coverage so far
    covered = {t for t in universe if t in sector_data
               and sector_data[t][0] not in BAD_SECTOR_VALUES}
    missing = [t for t in universe if t not in covered]
    print(f"\n  Coverage after static sources: {len(covered)}/{len(universe)} tickers")
    print(f"  Remaining unknowns: {len(missing)}")

    # ── Step 3: yfinance fetch for unknowns ──────────────────────────────────
    if missing:
        print(f"\n[3/4] Fetching {len(missing)} remaining tickers from yfinance …")
        if not dry_run:
            yf_data = _fetch_sectors_yf(missing, max_workers=12)
            for t, val in yf_data.items():
                sector_data[t] = val
        else:
            print(f"  [DRY-RUN] would fetch {len(missing)} tickers from yfinance")
    else:
        print("\n[3/4] All tickers covered — skipping yfinance fetch")

    # ── Step 4: Build output DataFrame (with canonical GICS normalisation) ────
    print("\n[4/4] Building sector_map.csv …")
    rows = []
    for t in universe:
        if t in sector_data:
            s, src = sector_data[t]
        else:
            s, src = "Other", "unknown"
        # Normalise to canonical 11-sector GICS naming
        s = _normalise_sector(s)
        rows.append({"ticker": t, "sector": s, "source": src})

    df = pd.DataFrame(rows)

    # Stats
    sector_counts = df[df["sector"] != "Other"]["sector"].value_counts()
    n_covered     = (df["sector"] != "Other").sum()
    print(f"\n  Coverage: {n_covered}/{len(df)} ({n_covered/len(df)*100:.0f}%)")
    print("  By sector:")
    for sec, cnt in sector_counts.items():
        print(f"    {sec:<30} {cnt:>3}")

    if not dry_run:
        out_path = ROOT / "sector_map.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  [written] {out_path}  ({len(df)} rows)")
    else:
        print("\n  [DRY-RUN] sector_map.csv not written")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 88 — Sector Map Builder"
    )
    parser.add_argument("--refresh",  action="store_true",
                        help="Re-fetch all tickers from yfinance (ignore existing cache)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show plan but don't write sector_map.csv")
    args = parser.parse_args()

    df = build_sector_map(refresh=args.refresh, dry_run=args.dry_run)

    n_ok  = (df["sector"] != "Other").sum() if not df.empty else 0
    total = len(df)
    print(f"\n✓ Sector map complete — {n_ok}/{total} tickers mapped "
          f"({n_ok/total*100:.0f}% coverage)\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
