#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 23B — Free yfinance Options Chain + Black-Scholes Gamma

No Polygon API key required; no manual options_chain_input.csv needed.
Fetches Yahoo Finance option chains via yfinance, approximates gamma using
Black-Scholes, and outputs options_chain_snapshot.csv for use by Step 23 / Step 24.

Limitations:
- yfinance is not an institutional real-time data source; suitable for research/prototyping only.
- Yahoo option chains include bid/ask/volume/openInterest/impliedVolatility but not dealer real positions.
- Gamma is approximated via Black-Scholes from IV, spot, strike, and DTE.
- No order submission; no broker connection.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
import math
import time
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing yfinance. Run: pip install yfinance")

ROOT = Path.cwd()

PRETRADE_FILE = ROOT / "pre_trade_checklist.csv"
SIZING_FILE = ROOT / "position_sizing_recommendations.csv"

OUT_CHAIN = ROOT / "options_chain_snapshot.csv"
OUT_INPUT = ROOT / "options_chain_input.csv"
OUT_REPORT = ROOT / "yfinance_options_fetch_report.md"

MAX_TICKERS = 8
MAX_EXPIRATIONS_PER_TICKER = 3
RISK_FREE_RATE = 0.04


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_gamma(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    try:
        S = float(S)
        K = float(K)
        T = float(T)
        sigma = float(sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        return norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except Exception:
        return np.nan


def bs_delta(S: float, K: float, T: float, sigma: float, option_type: str, r: float = RISK_FREE_RATE) -> float:
    try:
        S = float(S)
        K = float(K)
        T = float(T)
        sigma = float(sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        if option_type.lower() == "call":
            return norm_cdf(d1)
        return norm_cdf(d1) - 1.0
    except Exception:
        return np.nan


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def choose_tickers() -> list[str]:
    tickers = []

    pre = read_csv(PRETRADE_FILE)
    if not pre.empty and "ticker" in pre.columns:
        df = pre.copy()
        if "final_status" in df.columns:
            df = df[df["final_status"].astype(str).str.upper().isin([
                "PENDING_MANUAL_CHECKS",
                "ALREADY_OPEN_PAPER",
            ])]
        if "suggested_weight" in df.columns:
            df["_w"] = pd.to_numeric(df["suggested_weight"], errors="coerce").fillna(0)
            df = df.sort_values("_w", ascending=False)
        tickers = df["ticker"].astype(str).str.upper().str.strip().tolist()

    if not tickers:
        sizing = read_csv(SIZING_FILE)
        if not sizing.empty and "ticker" in sizing.columns:
            if "suggested_weight" in sizing.columns:
                sizing["_w"] = pd.to_numeric(sizing["suggested_weight"], errors="coerce").fillna(0)
                sizing = sizing.sort_values("_w", ascending=False)
            tickers = sizing["ticker"].astype(str).str.upper().str.strip().tolist()

    out = []
    for t in tickers:
        if t and t not in {"CASH", "TACTICAL_CASH"} and t not in out:
            out.append(t)
    return out[:MAX_TICKERS]


def get_spot(t: yf.Ticker, symbol: str) -> float:
    hist = t.history(period="5d", interval="1d", auto_adjust=False)
    if not hist.empty and "Close" in hist.columns:
        return float(hist["Close"].dropna().iloc[-1])

    try:
        info = t.info
        for key in ["regularMarketPrice", "currentPrice", "previousClose"]:
            v = info.get(key)
            if v:
                return float(v)
    except Exception:
        pass

    raise ValueError(f"Could not get spot price for {symbol}")


def fetch_one_ticker(symbol: str) -> tuple[pd.DataFrame, list[str]]:
    logs = []
    rows = []

    ticker = yf.Ticker(symbol)

    try:
        spot = get_spot(ticker, symbol)
    except Exception as e:
        return pd.DataFrame(), [f"{symbol}: spot failed: {e}"]

    try:
        expirations = list(ticker.options)
    except Exception as e:
        return pd.DataFrame(), [f"{symbol}: options expirations failed: {e}"]

    if not expirations:
        return pd.DataFrame(), [f"{symbol}: no option expirations found"]

    selected_exp = expirations[:MAX_EXPIRATIONS_PER_TICKER]
    logs.append(f"{symbol}: spot={spot:.2f}, expirations={selected_exp}")

    for exp in selected_exp:
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
        except Exception as e:
            logs.append(f"{symbol} {exp}: option_chain failed: {e}")
            continue

        for opt_type, df in [("call", calls), ("put", puts)]:
            if df is None or df.empty:
                continue

            df = df.copy()
            df["ticker"] = symbol
            df["expiration_date"] = exp
            df["option_type"] = opt_type
            df["spot"] = spot

            rename = {
                "contractSymbol": "contract_ticker",
                "openInterest": "open_interest",
                "impliedVolatility": "implied_volatility",
            }
            df = df.rename(columns=rename)

            needed = ["contract_ticker", "strike", "bid", "ask", "volume", "open_interest", "implied_volatility"]
            for col in needed:
                if col not in df.columns:
                    df[col] = np.nan

            exp_dt = pd.to_datetime(exp, errors="coerce")
            dte = (exp_dt - pd.Timestamp(date.today())).days if pd.notna(exp_dt) else np.nan
            T = max(float(dte) / 365.0, 1.0 / 365.0) if pd.notna(dte) else np.nan

            df["dte"] = dte
            df["gamma"] = df.apply(lambda r: bs_gamma(spot, r["strike"], T, r["implied_volatility"]), axis=1)
            df["delta"] = df.apply(lambda r: bs_delta(spot, r["strike"], T, r["implied_volatility"], opt_type), axis=1)

            bid = pd.to_numeric(df["bid"], errors="coerce")
            ask = pd.to_numeric(df["ask"], errors="coerce")
            df["mid"] = np.where(bid.notna() & ask.notna(), (bid + ask) / 2, np.nan)

            rows.append(df[[
                "ticker", "contract_ticker", "expiration_date", "option_type",
                "strike", "spot", "gamma", "delta", "implied_volatility",
                "open_interest", "volume", "bid", "ask", "mid", "dte"
            ]])

        time.sleep(0.25)

    if rows:
        return pd.concat(rows, ignore_index=True), logs

    return pd.DataFrame(), logs + [f"{symbol}: no chain rows collected"]


def summarize(df: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 23B — yfinance Options Fetch Report")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Status")
    md.append("")
    if df.empty:
        md.append("No option chain rows collected.")
        return "\n".join(md)

    md.append(f"- Rows: **{len(df)}**")
    md.append(f"- Tickers: **{', '.join(sorted(df['ticker'].unique()))}**")
    md.append("")
    md.append("## Coverage")
    md.append("")
    cov = df.groupby("ticker").agg(
        rows=("strike", "count"),
        expirations=("expiration_date", "nunique"),
        total_oi=("open_interest", "sum"),
        total_volume=("volume", "sum"),
    ).reset_index()

    try:
        md.append(cov.to_markdown(index=False))
    except Exception:
        md.append(cov.to_string(index=False))

    md.append("")
    md.append("## Important Limitations")
    md.append("")
    md.append("- yfinance/Yahoo options data is suitable for research/prototype use, not institutional execution.")
    md.append("- Gamma and delta here are calculated approximations using Black-Scholes and Yahoo implied volatility.")
    md.append("- Open interest does not reveal dealer true long/short gamma.")
    md.append("- Use Step 23 and Step 24 as risk filters, not as buy/sell signals.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 23B")
    print("Free yfinance Options Chain + BS Gamma")
    print("=" * 88)

    tickers = choose_tickers()
    if not tickers:
        tickers = ["SPY", "QQQ", "AAPL", "MSFT"]

    print("Tickers:", ", ".join(tickers))

    all_rows = []
    logs = []
    for symbol in tickers:
        df, lg = fetch_one_ticker(symbol)
        logs.extend(lg)
        if not df.empty:
            all_rows.append(df)

    if all_rows:
        chain = pd.concat(all_rows, ignore_index=True)
    else:
        chain = pd.DataFrame()

    if not chain.empty:
        for col in ["strike", "spot", "gamma", "delta", "implied_volatility", "open_interest", "volume", "bid", "ask", "mid", "dte"]:
            chain[col] = pd.to_numeric(chain[col], errors="coerce")
        chain = chain.dropna(subset=["ticker", "strike", "spot", "gamma", "implied_volatility"])
        chain.to_csv(OUT_CHAIN, index=False)
        chain.to_csv(OUT_INPUT, index=False)
    else:
        old_chain = read_csv(OUT_CHAIN)
        old_input = read_csv(OUT_INPUT)
        if old_chain.empty:
            pd.DataFrame().to_csv(OUT_CHAIN, index=False)
        else:
            logs.append("No fresh rows collected; preserved existing options_chain_snapshot.csv.")
        if old_input.empty:
            pd.DataFrame().to_csv(OUT_INPUT, index=False)
        else:
            logs.append("No fresh rows collected; preserved existing options_chain_input.csv.")

    OUT_REPORT.write_text(summarize(chain) + "\n\n## Logs\n\n" + "\n".join(f"- {x}" for x in logs), encoding="utf-8")

    print(f"Rows collected: {len(chain)}")
    if not chain.empty:
        print(chain.groupby("ticker").size().to_string())

    print("\nFiles generated:")
    print(f"  {OUT_CHAIN}")
    print(f"  {OUT_INPUT}")
    print(f"  {OUT_REPORT}")

    print("\nNext:")
    print("  python3 -u canyon_final_v9_step23_options_gamma_layer.py")
    print("  python3 -u canyon_final_v9_step24_option_kill_zone.py")


if __name__ == "__main__":
    main()
