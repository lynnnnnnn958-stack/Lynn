#!/usr/bin/env python3
"""
Canyon v9 — Step 107: Watchlist Tracker
========================================
Maintain a user-defined watchlist of tickers with alert conditions.
Checks current signal data against thresholds and surfaces triggered alerts.

Watchlist state  : watchlist.json        (user-editable, created if missing)
Daily snapshot   : watchlist_status.csv  (current status of all tickers)
Alert log        : watchlist_alerts.csv  (triggered alerts only)

Usage
-----
  python3 step107_watchlist.py                            # daily check (default)
  python3 step107_watchlist.py --add AAPL --note "Watching for breakout"
  python3 step107_watchlist.py --add AAPL --alpha-above 75 --price-below 200
  python3 step107_watchlist.py --remove AAPL
  python3 step107_watchlist.py --list
  python3 step107_watchlist.py --clear
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── File paths ────────────────────────────────────────────────────────────────
WATCHLIST_JSON   = ROOT / "watchlist.json"
STATUS_CSV       = ROOT / "watchlist_status.csv"
ALERTS_CSV       = ROOT / "watchlist_alerts.csv"

# ── Source data files ─────────────────────────────────────────────────────────
ALPHA_SCORES_CSV  = ROOT / "alpha_scores.csv"
PRICE_CACHE_CSV   = ROOT / "sp500_price_cache.csv"
OPTIONS_CSV       = ROOT / "options_signals.csv"
DAILY_PICKS_CSV   = ROOT / "daily_picks.csv"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ALERTS: dict = {
    "alpha_score_above": 70,   # alert when ticker enters buy territory
    "alpha_score_below": None,
    "price_above": None,
    "price_below": None,
    "regime_ml_above": None,
    "regime_ml_below": None,
    "now_in_buy_list": None,
    "iv_rank_below": None,
}

TODAY = datetime.now().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = np.nan) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_csv_safe(path: Path, label: str) -> pd.DataFrame | None:
    """Load CSV gracefully; return None if missing or unreadable."""
    if not path.exists():
        print(f"  [SKIP] {label} not found ({path.name})")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [SKIP] {label} is empty")
            return None
        # Normalise ticker column name to 'ticker'
        for col in df.columns:
            if col.lower() in ("ticker", "symbol"):
                if col != "ticker":
                    df = df.rename(columns={col: "ticker"})
                break
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        return df
    except Exception as exc:
        print(f"  [WARN] {label}: could not read ({exc})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# [1] Watchlist JSON I/O
# ─────────────────────────────────────────────────────────────────────────────

def _empty_watchlist() -> dict:
    return {"updated": TODAY, "tickers": {}}


def load_watchlist() -> dict:
    """Load watchlist.json. Create an empty one if not exists."""
    if not WATCHLIST_JSON.exists():
        wl = _empty_watchlist()
        _save_watchlist(wl)
        print(f"  [INFO] Created new watchlist at {WATCHLIST_JSON.name}")
        return wl
    try:
        with open(WATCHLIST_JSON, "r") as fh:
            wl = json.load(fh)
        if "tickers" not in wl:
            wl["tickers"] = {}
        return wl
    except Exception as exc:
        print(f"  [WARN] Could not read watchlist.json ({exc}). Using empty watchlist.")
        return _empty_watchlist()


def _save_watchlist(wl: dict) -> None:
    wl["updated"] = TODAY
    with open(WATCHLIST_JSON, "w") as fh:
        json.dump(wl, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# [2] Add / Remove / List / Clear
# ─────────────────────────────────────────────────────────────────────────────

def add_ticker(ticker: str, note: str = "", alerts: dict | None = None) -> None:
    """Add ticker to watchlist with optional note and alert conditions."""
    ticker = ticker.strip().upper()
    wl = load_watchlist()

    if ticker in wl["tickers"]:
        existing = wl["tickers"][ticker]
        print(f"  [INFO] {ticker} already on watchlist (added {existing.get('added_date', '?')}). Updating.")
        # Merge: update note and any new alert keys
        if note:
            existing["note"] = note
        if alerts:
            existing["alerts"].update(alerts)
    else:
        # Build alerts dict with defaults for any missing keys
        merged_alerts = dict(DEFAULT_ALERTS)
        if alerts:
            merged_alerts.update(alerts)
        wl["tickers"][ticker] = {
            "added_date": TODAY,
            "note": note,
            "alerts": merged_alerts,
        }
        print(f"  [OK] Added {ticker} to watchlist.")

    _save_watchlist(wl)


def remove_ticker(ticker: str) -> None:
    """Remove ticker from watchlist."""
    ticker = ticker.strip().upper()
    wl = load_watchlist()
    if ticker not in wl["tickers"]:
        print(f"  [WARN] {ticker} is not on the watchlist.")
        return
    del wl["tickers"][ticker]
    _save_watchlist(wl)
    print(f"  [OK] Removed {ticker} from watchlist.")


def list_watchlist() -> None:
    """Print the current watchlist in a readable table."""
    wl = load_watchlist()
    tickers = wl.get("tickers", {})
    if not tickers:
        print("  Watchlist is empty.")
        return

    print(f"\n  Watchlist — {len(tickers)} ticker(s)  (updated {wl.get('updated', '?')})")
    print(f"  {'Ticker':<8}  {'Added':<12}  {'Note':<35}  Alert conditions")
    print("  " + "-" * 90)
    for tkr, meta in sorted(tickers.items()):
        note = (meta.get("note") or "")[:33]
        added = meta.get("added_date", "?")
        alrt = meta.get("alerts", {})
        conds = [f"{k}={v}" for k, v in alrt.items() if v is not None]
        cond_str = ", ".join(conds) if conds else "(default: alpha_score_above=70)"
        print(f"  {tkr:<8}  {added:<12}  {note:<35}  {cond_str}")
    print()


def clear_watchlist() -> None:
    """Remove all tickers from the watchlist."""
    wl = load_watchlist()
    count = len(wl.get("tickers", {}))
    wl["tickers"] = {}
    _save_watchlist(wl)
    print(f"  [OK] Cleared {count} ticker(s) from watchlist.")


# ─────────────────────────────────────────────────────────────────────────────
# [3] Load current data for watchlist tickers
# ─────────────────────────────────────────────────────────────────────────────

def get_current_data(tickers: list[str]) -> pd.DataFrame:
    """
    Load current data for watchlist tickers from:
      - alpha_scores.csv  (alpha_score, sig_* columns, action)
      - sp500_price_cache.csv  (latest price)
      - options_signals.csv  (iv_rank / iv_pct, rank_options)
      - daily_picks.csv  (to check if now in buy list)

    Returns a DataFrame with one row per ticker and all available fields.
    """
    if not tickers:
        return pd.DataFrame()

    upper = [t.upper() for t in tickers]
    result = pd.DataFrame({"ticker": upper})

    # ── alpha_scores.csv ────────────────────────────────────────────────────
    alpha_df = _load_csv_safe(ALPHA_SCORES_CSV, "alpha_scores.csv")
    if alpha_df is not None and "ticker" in alpha_df.columns:
        alpha_keep = ["ticker", "alpha_score", "signal", "regime",
                      "sig_regime_ml", "sig_quality", "sig_revision",
                      "sig_surprise", "sig_sentiment", "sig_squeeze",
                      "sig_insider", "sig_options", "sig_ml_ensemble",
                      "sector", "crowding_level"]
        alpha_keep = [c for c in alpha_keep if c in alpha_df.columns]
        result = result.merge(alpha_df[alpha_keep], on="ticker", how="left")

    # ── sp500_price_cache.csv  (wide format: dates as rows, tickers as cols) ─
    price_df = _load_csv_safe(PRICE_CACHE_CSV, "sp500_price_cache.csv")
    if price_df is not None:
        # The price cache has dates as the index column (first col unnamed or 'date')
        # and tickers as column headers — extract the last row's values
        first_col = price_df.columns[0]
        price_df = price_df.set_index(first_col)
        price_df.index.name = "date"
        # Drop non-numeric columns that snuck through
        price_df = price_df.apply(pd.to_numeric, errors="coerce")
        last_prices: dict[str, float] = {}
        if not price_df.empty:
            last_row = price_df.iloc[-1]
            for t in upper:
                if t in last_row.index:
                    val = last_row[t]
                    if pd.notna(val):
                        last_prices[t] = float(val)
        price_rows = [{"ticker": t, "price": last_prices.get(t, np.nan)} for t in upper]
        price_ser = pd.DataFrame(price_rows)
        result = result.merge(price_ser, on="ticker", how="left")

    # ── options_signals.csv ─────────────────────────────────────────────────
    opt_df = _load_csv_safe(OPTIONS_CSV, "options_signals.csv")
    if opt_df is not None and "ticker" in opt_df.columns:
        # iv_pct is the IV rank (percentile 0-100) used for iv_rank_below alert
        opt_keep = ["ticker", "iv_pct", "rank_options", "options_score",
                    "options_signal", "pcr"]
        opt_keep = [c for c in opt_keep if c in opt_df.columns]
        result = result.merge(opt_df[opt_keep], on="ticker", how="left")
        # Alias iv_pct -> iv_rank for clarity
        if "iv_pct" in result.columns and "iv_rank" not in result.columns:
            result = result.rename(columns={"iv_pct": "iv_rank"})

    # ── daily_picks.csv ─────────────────────────────────────────────────────
    picks_df = _load_csv_safe(DAILY_PICKS_CSV, "daily_picks.csv")
    buy_set: set[str] = set()
    if picks_df is not None and "ticker" in picks_df.columns:
        buy_set = set(picks_df["ticker"].str.upper().unique())

    result["in_buy_list"] = result["ticker"].apply(lambda t: t in buy_set)

    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# [4] Alert checking
# ─────────────────────────────────────────────────────────────────────────────

def check_alerts(
    ticker: str,
    ticker_data: dict,
    alert_conditions: dict,
) -> list[dict]:
    """
    Check each alert condition against current ticker_data.
    Returns a list of triggered alert dicts:
      [{condition, current_value, threshold, message}]
    """
    triggered: list[dict] = []

    def _fire(condition: str, current_val, threshold, msg: str) -> None:
        triggered.append({
            "ticker": ticker,
            "condition": condition,
            "current_value": current_val,
            "threshold": threshold,
            "message": msg,
        })

    def _get(field: str) -> float:
        return _safe_float(ticker_data.get(field))

    # alpha_score_above
    thresh = alert_conditions.get("alpha_score_above")
    if thresh is not None:
        cur = _get("alpha_score")
        if pd.notna(cur) and cur >= float(thresh):
            _fire("alpha_score_above", round(cur, 1), thresh,
                  f"{ticker} alpha score {cur:.1f} is above threshold {thresh}")

    # alpha_score_below
    thresh = alert_conditions.get("alpha_score_below")
    if thresh is not None:
        cur = _get("alpha_score")
        if pd.notna(cur) and cur <= float(thresh):
            _fire("alpha_score_below", round(cur, 1), thresh,
                  f"{ticker} alpha score {cur:.1f} is below threshold {thresh}")

    # price_above
    thresh = alert_conditions.get("price_above")
    if thresh is not None:
        cur = _get("price")
        if pd.notna(cur) and cur >= float(thresh):
            _fire("price_above", round(cur, 2), thresh,
                  f"{ticker} price ${cur:.2f} is above threshold ${thresh:.2f}")

    # price_below
    thresh = alert_conditions.get("price_below")
    if thresh is not None:
        cur = _get("price")
        if pd.notna(cur) and cur <= float(thresh):
            _fire("price_below", round(cur, 2), thresh,
                  f"{ticker} price ${cur:.2f} is below threshold ${thresh:.2f}")

    # regime_ml_above
    thresh = alert_conditions.get("regime_ml_above")
    if thresh is not None:
        cur = _get("sig_regime_ml")
        if pd.notna(cur) and cur >= float(thresh):
            _fire("regime_ml_above", round(cur, 1), thresh,
                  f"{ticker} regime ML score {cur:.1f} is above threshold {thresh}")

    # regime_ml_below
    thresh = alert_conditions.get("regime_ml_below")
    if thresh is not None:
        cur = _get("sig_regime_ml")
        if pd.notna(cur) and cur <= float(thresh):
            _fire("regime_ml_below", round(cur, 1), thresh,
                  f"{ticker} regime ML score {cur:.1f} is below threshold {thresh}")

    # now_in_buy_list — alert when ticker appears in daily_picks
    flag = alert_conditions.get("now_in_buy_list")
    if flag is True or flag == "true" or flag == 1:
        in_buy = bool(ticker_data.get("in_buy_list", False))
        if in_buy:
            _fire("now_in_buy_list", True, True,
                  f"{ticker} now appears in daily buy picks list")

    # iv_rank_below — cheap options alert
    thresh = alert_conditions.get("iv_rank_below")
    if thresh is not None:
        cur = _get("iv_rank")
        if pd.notna(cur) and cur <= float(thresh):
            _fire("iv_rank_below", round(cur, 1), thresh,
                  f"{ticker} IV rank {cur:.1f}% is below threshold {thresh}% (cheap options)")

    return triggered


# ─────────────────────────────────────────────────────────────────────────────
# [5] Main check runner
# ─────────────────────────────────────────────────────────────────────────────

def run_watchlist_check() -> pd.DataFrame:
    """
    Main function:
      1. Load watchlist
      2. Get current data for all watchlist tickers
      3. Check all alert conditions
      4. Build status DataFrame
      5. Write watchlist_status.csv and watchlist_alerts.csv
      6. Print formatted status table
    Returns the status DataFrame.
    """
    print("\n" + "=" * 60)
    print("  Canyon v9 — Step 107: Watchlist Check")
    print(f"  Date: {TODAY}")
    print("=" * 60)

    wl = load_watchlist()
    ticker_meta = wl.get("tickers", {})

    if not ticker_meta:
        print("\n  Watchlist is empty. Use --add TICKER to add tickers.")
        print("=" * 60 + "\n")
        return pd.DataFrame()

    tickers_list = sorted(ticker_meta.keys())
    print(f"\n  Checking {len(tickers_list)} ticker(s): {', '.join(tickers_list)}")
    print()

    # Load current data
    data_df = get_current_data(tickers_list)

    # Build per-ticker lookup
    data_lookup: dict[str, dict] = {}
    for _, row in data_df.iterrows():
        t = str(row.get("ticker", "")).upper()
        data_lookup[t] = row.to_dict()

    # Check alerts and build status rows
    status_rows: list[dict] = []
    all_alerts: list[dict] = []

    for ticker in tickers_list:
        meta = ticker_meta[ticker]
        added_date = meta.get("added_date", TODAY)
        note = meta.get("note", "")
        alert_conds = meta.get("alerts", {})

        # Fill missing alert keys with defaults
        merged_conds: dict = dict(DEFAULT_ALERTS)
        merged_conds.update({k: v for k, v in alert_conds.items() if v is not None})
        # Re-apply explicit None overrides from the watchlist
        for k, v in alert_conds.items():
            if v is None:
                merged_conds[k] = None

        td = data_lookup.get(ticker, {})

        # days on watch
        try:
            added_dt = datetime.strptime(added_date, "%Y-%m-%d")
            today_dt = datetime.strptime(TODAY, "%Y-%m-%d")
            days_on_watch = (today_dt - added_dt).days
        except Exception:
            days_on_watch = 0

        # Check alerts
        triggered = check_alerts(ticker, td, merged_conds)
        all_alerts.extend(triggered)
        alerts_str = "; ".join(a["condition"] for a in triggered) if triggered else ""

        alpha = _safe_float(td.get("alpha_score"))
        price = _safe_float(td.get("price"))
        regime_ml = _safe_float(td.get("sig_regime_ml"))
        iv_rank = _safe_float(td.get("iv_rank"))
        in_buy = bool(td.get("in_buy_list", False))
        signal = str(td.get("signal", "")) or ""

        status_rows.append({
            "ticker": ticker,
            "in_buy_list": in_buy,
            "alpha_score": round(alpha, 1) if pd.notna(alpha) else np.nan,
            "signal": signal,
            "price": round(price, 2) if pd.notna(price) else np.nan,
            "regime_ml": round(regime_ml, 1) if pd.notna(regime_ml) else np.nan,
            "iv_rank": round(iv_rank, 1) if pd.notna(iv_rank) else np.nan,
            "days_on_watch": days_on_watch,
            "alerts_triggered": len(triggered),
            "alert_conditions": alerts_str,
            "note": note,
            "added_date": added_date,
        })

    status_df = pd.DataFrame(status_rows)

    # ── Print status table ─────────────────────────────────────────────────
    print(f"  {'Ticker':<7}  {'InBuy':<6}  {'Alpha':>6}  {'Signal':<12}  "
          f"{'Price':>8}  {'RegML':>6}  {'Days':>5}  {'Alerts':>6}  Note")
    print("  " + "-" * 100)

    for _, row in status_df.iterrows():
        buy_flag = "YES" if row["in_buy_list"] else "no"
        alpha_s  = f"{row['alpha_score']:>6.1f}" if pd.notna(row["alpha_score"]) else "   N/A"
        price_s  = f"{row['price']:>8.2f}" if pd.notna(row["price"]) else "     N/A"
        regml_s  = f"{row['regime_ml']:>6.1f}" if pd.notna(row["regime_ml"]) else "   N/A"
        alerts_n = int(row["alerts_triggered"])
        alrt_tag = f"[{alerts_n} ALERT{'S' if alerts_n != 1 else ''}]" if alerts_n else "      -"
        note_s   = str(row["note"])[:28]
        signal_s = str(row["signal"])[:12]
        print(f"  {row['ticker']:<7}  {buy_flag:<6}  {alpha_s}  {signal_s:<12}  "
              f"{price_s}  {regml_s}  {int(row['days_on_watch']):>5}  {alrt_tag:>8}  {note_s}")

    print()

    # ── Print triggered alerts ────────────────────────────────────────────
    if all_alerts:
        print(f"  *** {len(all_alerts)} ALERT(S) TRIGGERED ***")
        print()
        for a in all_alerts:
            print(f"    [{a['ticker']}] {a['condition'].upper()}: {a['message']}")
        print()
    else:
        print("  No alerts triggered.")
        print()

    # ── Write outputs ─────────────────────────────────────────────────────
    status_df["check_date"] = TODAY
    status_df.to_csv(STATUS_CSV, index=False)
    print(f"  Wrote: {STATUS_CSV.name}")

    alerts_df = pd.DataFrame(all_alerts) if all_alerts else pd.DataFrame(
        columns=["ticker", "condition", "current_value", "threshold", "message"]
    )
    alerts_df["check_date"] = TODAY
    alerts_df.to_csv(ALERTS_CSV, index=False)
    print(f"  Wrote: {ALERTS_CSV.name}")

    print("=" * 60 + "\n")
    return status_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="step107_watchlist",
        description="Canyon v9 Step 107 — Watchlist Tracker",
    )
    p.add_argument("--add",    metavar="TICKER", help="Add a ticker to the watchlist")
    p.add_argument("--remove", metavar="TICKER", help="Remove a ticker from the watchlist")
    p.add_argument("--list",   action="store_true", help="Show all watchlist tickers")
    p.add_argument("--clear",  action="store_true", help="Remove all tickers from watchlist")
    p.add_argument("--note",   default="", help="Note to attach when using --add")

    # Alert condition flags (used with --add)
    p.add_argument("--alpha-above",   type=float, metavar="N",
                   help="Alert when alpha score >= N")
    p.add_argument("--alpha-below",   type=float, metavar="N",
                   help="Alert when alpha score <= N")
    p.add_argument("--price-above",   type=float, metavar="N",
                   help="Alert when price >= N")
    p.add_argument("--price-below",   type=float, metavar="N",
                   help="Alert when price <= N")
    p.add_argument("--regime-ml-above", type=float, metavar="N",
                   help="Alert when regime ML score >= N")
    p.add_argument("--regime-ml-below", type=float, metavar="N",
                   help="Alert when regime ML score <= N")
    p.add_argument("--in-buy-list",   action="store_true",
                   help="Alert when ticker enters daily buy list")
    p.add_argument("--iv-rank-below", type=float, metavar="N",
                   help="Alert when IV rank <= N (cheap options)")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Subcommands ────────────────────────────────────────────────────────
    if args.list:
        list_watchlist()
        sys.exit(0)

    if args.clear:
        clear_watchlist()
        sys.exit(0)

    if args.remove:
        remove_ticker(args.remove)
        sys.exit(0)

    if args.add:
        # Build alerts dict from CLI flags
        alerts: dict = {}
        if args.alpha_above is not None:
            alerts["alpha_score_above"] = args.alpha_above
        if args.alpha_below is not None:
            alerts["alpha_score_below"] = args.alpha_below
        if args.price_above is not None:
            alerts["price_above"] = args.price_above
        if args.price_below is not None:
            alerts["price_below"] = args.price_below
        if args.regime_ml_above is not None:
            alerts["regime_ml_above"] = args.regime_ml_above
        if args.regime_ml_below is not None:
            alerts["regime_ml_below"] = args.regime_ml_below
        if args.in_buy_list:
            alerts["now_in_buy_list"] = True
        if args.iv_rank_below is not None:
            alerts["iv_rank_below"] = args.iv_rank_below

        add_ticker(args.add, note=args.note, alerts=alerts if alerts else None)
        sys.exit(0)

    # ── Default: run daily check ───────────────────────────────────────────
    run_watchlist_check()
    sys.exit(0)


if __name__ == "__main__":
    main()
