#!/usr/bin/env python3
"""
Canyon — step_alpaca_pnl.py
=============================
Fetch realized P&L from Alpaca paper account and compare against
Canyon's predicted mu_override (expected return per ticker).

Outputs:
  alpaca_pnl_attribution.csv   ticker | predicted_mu | realized_ret | alpha | book
  alpaca_pnl_summary.json      per-book stats + overall IC vs mu

Runs daily after market close.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg): print(f"  {msg}")
def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()


def get_alpaca_positions_pnl() -> list[dict]:
    """Fetch all closed orders + current positions P&L from Alpaca."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError:
        err("alpaca-py not installed")
        return []

    key    = os.environ.get("ALPACA_KEY_ID", "")
    secret = os.environ.get("ALPACA_KEY_SECRET", "")
    if not key or not secret:
        err("Alpaca keys not set")
        return []

    try:
        client = TradingClient(key, secret, paper=True)

        # Current open positions with unrealized P&L
        positions = client.get_all_positions()
        records = []
        for p in positions:
            records.append({
                "ticker":         p.symbol,
                "qty":            float(p.qty),
                "avg_cost":       float(p.avg_entry_price),
                "current_price":  float(p.current_price),
                "market_value":   float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc),
                "type":           "open",
            })

        # Closed orders in last 30 days for realized P&L
        since = (datetime.now() - timedelta(days=30)).isoformat() + "Z"
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=since, limit=500)
            orders = client.get_orders(req)
            for o in orders:
                if o.filled_at and o.filled_avg_price and o.side.value == "sell":
                    records.append({
                        "ticker":        o.symbol,
                        "qty":           float(o.filled_qty or 0),
                        "filled_price":  float(o.filled_avg_price),
                        "filled_at":     str(o.filled_at)[:10],
                        "type":          "closed",
                    })
        except Exception as e:
            warn(f"  Could not fetch order history: {e}")

        ok(f"Alpaca: {len([r for r in records if r['type']=='open'])} open positions, "
           f"{len([r for r in records if r['type']=='closed'])} closed orders")
        return records

    except Exception as e:
        err(f"Alpaca P&L fetch failed: {e}")
        return []


def load_predicted_mu() -> pd.DataFrame:
    """Load Canyon's predicted mu_override per ticker."""
    paths = [ROOT / "alpha_scores.csv", ROOT / "daily_picks.csv"]
    for path in paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "ticker" in df.columns and "mu_override" in df.columns:
                    return df[["ticker", "mu_override", "alpha_score"]].set_index("ticker")
                if "ticker" in df.columns and "alpha_score" in df.columns:
                    df["mu_override"] = (df["alpha_score"] - 50) / 50 * 0.25  # rough conversion
                    return df[["ticker", "mu_override", "alpha_score"]].set_index("ticker")
            except Exception:
                pass
    return pd.DataFrame()


def load_book_state() -> dict[str, list[str]]:
    """Returns {book_name: [ticker, ...]}"""
    state_path = ROOT / "alpaca_book_state.json"
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text())
        return {book: list(data.get("positions", {}).keys())
                for book, data in state.items()}
    except Exception:
        return {}


def main():
    print(f"\n{BOLD}Canyon — Alpaca P&L Attribution{RESET}  {TODAY}")
    load_env()

    records = get_alpaca_positions_pnl()
    predicted_mu = load_predicted_mu()
    book_state   = load_book_state()

    if not records:
        warn("No Alpaca data — exiting")
        return

    # Build ticker → book mapping
    ticker_to_book = {}
    for book, tickers in book_state.items():
        for tk in tickers:
            ticker_to_book[tk] = book

    # Build attribution table for open positions
    rows = []
    for r in records:
        if r["type"] != "open":
            continue
        tk = r["ticker"]
        pred = predicted_mu.loc[tk, "mu_override"] if tk in predicted_mu.index else None
        alpha_score = predicted_mu.loc[tk, "alpha_score"] if tk in predicted_mu.index else None
        rows.append({
            "ticker":          tk,
            "book":            ticker_to_book.get(tk, "UNKNOWN"),
            "market_value":    r["market_value"],
            "unrealized_ret":  r["unrealized_pnl_pct"],
            "unrealized_pnl":  r["unrealized_pnl"],
            "predicted_mu":    pred,
            "alpha_score":     alpha_score,
            "direction":       "long" if r["qty"] > 0 else "short",
        })

    if not rows:
        warn("No open position rows to attribute")
        return

    df = pd.DataFrame(rows)
    df["predicted_mu"] = df["predicted_mu"].astype(float)

    # Alpha vs predicted: unrealized_ret - predicted_mu (annualised approximation)
    df["alpha_vs_predicted"] = df["unrealized_ret"] - df["predicted_mu"]

    # Per-book summary
    summary = {}
    total_pnl = df["unrealized_pnl"].sum()
    for book in ["SHORT", "MEDIUM", "LONG"]:
        grp = df[df["book"] == book]
        if grp.empty:
            continue
        book_pnl  = grp["unrealized_pnl"].sum()
        book_ret  = grp["unrealized_ret"].mean()
        pred_mu   = grp["predicted_mu"].dropna().mean()
        summary[book] = {
            "n_positions":   len(grp),
            "total_pnl_$":   round(float(book_pnl), 2),
            "avg_return":    round(float(book_ret), 4),
            "avg_pred_mu":   round(float(pred_mu), 4) if not pd.isna(pred_mu) else None,
            "realized_alpha":round(float(book_ret - (pred_mu or 0)), 4),
        }

    # IC check: correlation(predicted_mu, unrealized_ret)
    ic_df = df[["predicted_mu", "unrealized_ret"]].dropna()
    if len(ic_df) > 5:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(ic_df["predicted_mu"], ic_df["unrealized_ret"])
        summary["ic_mu_vs_realized"] = round(float(rho), 4)
        summary["ic_pval"]           = round(float(pval), 4)
        ok(f"IC(predicted_mu vs realized_ret): {rho:.4f}  p={pval:.4f}")
    else:
        summary["ic_mu_vs_realized"] = None

    summary["total_pnl_$"] = round(float(total_pnl), 2)
    summary["as_of"]       = TODAY

    # Save
    out_csv = ROOT / "alpaca_pnl_attribution.csv"
    df.to_csv(out_csv, index=False)
    ok(f"alpaca_pnl_attribution.csv → {len(df)} rows")

    out_json = ROOT / "alpaca_pnl_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    ok(f"alpaca_pnl_summary.json saved")

    # Print summary
    print(f"\n  {'Book':<10} {'Positions':>10} {'Total P&L':>12} {'Avg Ret':>10} {'Pred μ':>10} {'Alpha':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for book in ["SHORT", "MEDIUM", "LONG"]:
        s = summary.get(book, {})
        if not s:
            continue
        alpha_c = GREEN if (s.get("realized_alpha") or 0) > 0 else RED
        print(f"  {book:<10} {s['n_positions']:>10} "
              f"  ${s['total_pnl_$']:>9,.0f}  "
              f"{s['avg_return']*100:>9.1f}%  "
              f"{(s['avg_pred_mu'] or 0)*100:>9.1f}%  "
              f"{alpha_c}{s['realized_alpha']*100:>9.1f}%{RESET}")
    print(f"\n  Total unrealized P&L: ${summary.get('total_pnl_$', 0):,.0f}")
    print(f"\n{GREEN}✓ P&L attribution complete{RESET}\n")


if __name__ == "__main__":
    main()
