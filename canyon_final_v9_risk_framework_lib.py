#!/usr/bin/env python3
"""
Canyon v9 institutional risk framework helpers.

This module is intentionally local-data-first. It does not connect to a broker,
does not send orders, and does not treat missing data as positive evidence.
All consumers should use conservative fallbacks: missing data can reduce or
block a decision, but it must never upgrade a ticker.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
MODEL_ACCOUNT_VALUE = 100000.0
TRADING_DAYS = 252


SECTOR_BENCHMARK_WEIGHTS = {
    "Technology": 0.315,
    "Information Technology": 0.315,
    "Financials": 0.140,
    "Healthcare": 0.115,
    "Health Care": 0.115,
    "Consumer Discretionary": 0.105,
    "Communication Services": 0.090,
    "Industrials": 0.085,
    "Consumer Staples": 0.060,
    "Energy": 0.035,
    "Utilities": 0.025,
    "Real Estate": 0.020,
    "Materials": 0.020,
    "Broad Market": 1.000,
    "Semiconductors": 0.060,
    "Tech": 0.315,
}


FACTOR_PROXIES = {
    "SPY_beta": "SPY",
    "QQQ_growth_beta": "QQQ",
    "rates_beta_TLT": "TLT",
    "usd_beta_UUP": "UUP",
    "oil_energy_beta_XLE": "XLE",
    "gold_beta_GLD": "GLD",
    "credit_beta_HYG": "HYG",
    "semiconductor_beta_SMH": "SMH",
}


AI_SEMI_TICKERS = {
    "NVDA", "AMD", "AVGO", "MU", "SMCI", "QCOM", "TXN", "ADI", "AMAT",
    "LRCX", "KLAC", "MRVL", "TSM", "ASML", "INTC", "ARM", "SMH", "SOXX",
}
MEGA_TECH_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA",
    "NFLX", "ADBE", "CRM", "ORCL", "QQQ", "XLK",
}
ETF_COMPONENT_MAP = {
    "SPY": "Broad Market",
    "QQQ": "Growth / Nasdaq",
    "XLK": "Technology",
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLI": "Industrials",
    "XLB": "Materials",
    "IYR": "Real Estate",
    "TLT": "Rates",
    "GLD": "Gold",
}


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def read_json_safe(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def clean_ticker(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().upper()


def normalize_weight(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(x):
        return 0.0
    if abs(x) > 1.5:
        x = x / 100.0
    return max(0.0, x)


def normalize_weight_series(series: pd.Series) -> pd.Series:
    w = series.apply(normalize_weight).astype(float)
    total = float(w.sum())
    if total > 1.5:
        w = w / 100.0
        total = float(w.sum())
    if total > 1.0:
        w = w / total
    return w


def load_sector_map() -> dict[str, str]:
    df = read_csv_safe(ROOT / "sector_map.csv")
    if {"ticker", "sector"}.issubset(df.columns):
        out = {}
        for _, row in df.iterrows():
            out[clean_ticker(row["ticker"])] = str(row["sector"])
        return out
    return {}


def infer_theme(ticker: str, sector: str = "") -> str:
    ticker = clean_ticker(ticker)
    if ticker in AI_SEMI_TICKERS:
        return "AI / Semiconductors"
    if ticker in MEGA_TECH_TICKERS:
        return "Mega-cap Technology"
    if ticker in ETF_COMPONENT_MAP:
        return ETF_COMPONENT_MAP[ticker]
    sec = str(sector or "").strip()
    return sec if sec else "Unknown"


def load_current_book(prefer_filtered: bool = True) -> pd.DataFrame:
    """
    Return the current research book with conservative weights.

    Priority:
    1. daily_picks_filtered.csv when available and requested.
    2. daily_picks.csv.
    3. paper_sim_positions.csv market values.
    4. portfolio_optimized_weights.csv turnover-aware weights.
    """
    candidates = []
    if prefer_filtered:
        candidates.append((ROOT / "daily_picks_filtered.csv", "daily_picks_filtered.csv"))
    candidates.extend([
        (ROOT / "daily_picks.csv", "daily_picks.csv"),
        (ROOT / "paper_sim_positions.csv", "paper_sim_positions.csv"),
        (ROOT / "portfolio_optimized_weights.csv", "portfolio_optimized_weights.csv"),
    ])

    sector_map = load_sector_map()
    frames: list[pd.DataFrame] = []

    for path, source in candidates:
        df = read_csv_safe(path)
        if df.empty or "ticker" not in df.columns:
            continue
        out = df.copy()
        out["ticker"] = out["ticker"].apply(clean_ticker)
        out = out[out["ticker"] != ""].copy()
        if out.empty:
            continue

        if "weight_pct" in out.columns:
            out["weight"] = normalize_weight_series(out["weight_pct"])
        elif "effective_weight" in out.columns:
            out["weight"] = normalize_weight_series(out["effective_weight"])
        elif "turnover_aware" in out.columns:
            out["weight"] = normalize_weight_series(out["turnover_aware"])
        elif "market_value" in out.columns:
            mv = pd.to_numeric(out["market_value"], errors="coerce").fillna(0.0)
            total = float(mv.sum())
            out["weight"] = mv / total if total > 0 else 0.0
        else:
            out["weight"] = 1.0 / max(len(out), 1)

        if "alpha_score" not in out.columns:
            out["alpha_score"] = np.nan
        if "action" not in out.columns:
            out["action"] = "RESEARCH"
        if "sector" not in out.columns:
            out["sector"] = out["ticker"].map(sector_map).fillna("Unknown")
        else:
            out["sector"] = out["sector"].fillna(out["ticker"].map(sector_map)).fillna("Unknown")
        out["theme"] = [infer_theme(t, s) for t, s in zip(out["ticker"], out["sector"])]
        out["source_file"] = source
        keep = [
            "ticker", "weight", "alpha_score", "action", "sector", "theme",
            "source_file",
        ]
        for extra in ["top_signal", "score_trend", "status", "regime"]:
            if extra in out.columns:
                keep.append(extra)
        frames.append(out[keep].copy())
        break

    if not frames:
        return pd.DataFrame(columns=["ticker", "weight", "alpha_score", "action", "sector", "theme", "source_file"])

    book = frames[0].copy()
    book["weight"] = normalize_weight_series(book["weight"])
    book = book.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    return book


def _load_one_price_file(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    first = df.columns[0]
    if first.lower() in {"date", "unnamed: 0", "index"}:
        df[first] = pd.to_datetime(df[first], errors="coerce")
        df = df.dropna(subset=[first]).set_index(first)
    else:
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
            if idx.notna().sum() > len(idx) * 0.8:
                df.index = idx
        except Exception:
            pass
    df = df.sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def load_price_cache() -> pd.DataFrame:
    """
    Load and merge local price caches. sp500_price_cache has broad single-name
    coverage; backtest_price_cache has ETF/factor proxies.
    """
    frames = []
    for fname in ["sp500_price_cache.csv", "backtest_price_cache.csv", "regime_price_cache.csv"]:
        df = _load_one_price_file(ROOT / fname)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = frames[0].copy()
    for df in frames[1:]:
        out = out.combine_first(df)
        for col in df.columns:
            if col not in out.columns:
                out[col] = df[col]
    out.columns = [clean_ticker(c) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated()].sort_index()
    return out


def get_returns(tickers: list[str], lookback: int | None = 252) -> pd.DataFrame:
    prices = load_price_cache()
    if prices.empty:
        return pd.DataFrame()
    wanted = [clean_ticker(t) for t in tickers]
    available = [t for t in wanted if t in prices.columns]
    if not available:
        return pd.DataFrame()
    px = prices[available].ffill().bfill()
    if lookback is not None:
        px = px.tail(int(lookback) + 5)
    rets = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets.dropna(axis=1, how="all")


def get_latest_prices(tickers: list[str]) -> dict[str, float]:
    prices = load_price_cache()
    out: dict[str, float] = {}
    if prices.empty:
        return out
    for t in [clean_ticker(x) for x in tickers]:
        if t in prices.columns:
            s = prices[t].dropna()
            if not s.empty:
                out[t] = float(s.iloc[-1])
    return out


def var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 20:
        return np.nan, np.nan
    q = float(r.quantile(1.0 - alpha))
    tail = r[r <= q]
    var = max(0.0, -q)
    cvar = max(0.0, -float(tail.mean())) if not tail.empty else var
    return var, cvar


def annualized_vol(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 20:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def beta_to_factor(asset_returns: pd.Series, factor_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, factor_returns], axis=1).dropna()
    if df.shape[0] < 40:
        return np.nan
    x = df.iloc[:, 1].astype(float)
    y = df.iloc[:, 0].astype(float)
    var = float(np.var(x, ddof=1))
    if var <= 0 or not np.isfinite(var):
        return np.nan
    return float(np.cov(y, x, ddof=1)[0, 1] / var)


def load_earnings_calendar() -> pd.DataFrame:
    ec = read_csv_safe(ROOT / "earnings_calendar.csv")
    if ec.empty:
        return ec
    ec = ec.copy()
    if "ticker" in ec.columns:
        ec["ticker"] = ec["ticker"].apply(clean_ticker)
    if "earnings_date" in ec.columns:
        ec["earnings_date"] = pd.to_datetime(ec["earnings_date"], errors="coerce")
    for col in ["days_until", "days_until_earnings", "days_to_earnings"]:
        if col in ec.columns:
            ec[col] = pd.to_numeric(ec[col], errors="coerce")
    return ec


def load_options_signals() -> pd.DataFrame:
    opt = read_csv_safe(ROOT / "options_signals.csv")
    if opt.empty:
        return opt
    opt = opt.copy()
    if "ticker" in opt.columns:
        opt["ticker"] = opt["ticker"].apply(clean_ticker)
    for col in ["iv_rank", "atm_iv", "days_to_earnings", "rank_options", "max_pain_dist", "pcr_vol"]:
        if col in opt.columns:
            opt[col] = pd.to_numeric(opt[col], errors="coerce")
    return opt


def _liquidity_tier_from_adv(adv: float) -> str:
    if not np.isfinite(adv) or adv <= 0:
        return "MISSING"
    if adv >= 2_000_000_000:
        return "HIGH"
    if adv >= 500_000_000:
        return "GOOD"
    if adv >= 100_000_000:
        return "FAIR"
    if adv >= 25_000_000:
        return "THIN"
    return "LOW"


def _fetch_liquidity_proxy_yfinance(tickers: list[str]) -> pd.DataFrame:
    tickers = sorted({clean_ticker(t) for t in tickers if clean_ticker(t)})
    if not tickers:
        return pd.DataFrame()
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers,
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()

    rows = []
    multi = isinstance(data.columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            sub = data[ticker].copy() if multi else data.copy()
        except Exception:
            continue
        if sub.empty or "Volume" not in sub.columns:
            continue
        price_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
        if price_col not in sub.columns:
            continue
        px = pd.to_numeric(sub[price_col], errors="coerce")
        vol = pd.to_numeric(sub["Volume"], errors="coerce")
        dollar = (px * vol).replace([np.inf, -np.inf], np.nan).dropna().tail(20)
        med_vol = vol.replace([np.inf, -np.inf], np.nan).dropna().tail(20)
        if dollar.empty:
            continue
        adv = float(dollar.mean())
        rows.append({
            "ticker": ticker,
            "avg_20d_dollar_volume": adv,
            "median_20d_volume": float(med_vol.median()) if not med_vol.empty else np.nan,
            "liquidity_label": _liquidity_tier_from_adv(adv),
            "notes": "Fetched from yfinance daily data; still check live bid-ask manually before paper action.",
        })
    return pd.DataFrame(rows)


def load_liquidity_proxy(tickers: list[str] | None = None, refresh_missing: bool = True) -> pd.DataFrame:
    liq = read_csv_safe(ROOT / "intraday_liquidity_proxy.csv")
    if not liq.empty:
        liq = liq.copy()
        if "ticker" in liq.columns:
            liq["ticker"] = liq["ticker"].apply(clean_ticker)
    else:
        liq = pd.DataFrame(columns=["ticker", "avg_20d_dollar_volume", "median_20d_volume", "liquidity_label", "notes"])

    requested = sorted({clean_ticker(t) for t in (tickers or []) if clean_ticker(t)})
    if refresh_missing and requested:
        existing = set(liq["ticker"].dropna().astype(str)) if "ticker" in liq.columns else set()
        missing = [t for t in requested if t not in existing]
        fetched = _fetch_liquidity_proxy_yfinance(missing)
        if not fetched.empty:
            liq = pd.concat([liq, fetched], ignore_index=True)
            liq = liq.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)
            try:
                liq.to_csv(ROOT / "intraday_liquidity_proxy.csv", index=False)
            except Exception:
                pass

    for col in ["avg_20d_dollar_volume", "median_20d_volume"]:
        if col in liq.columns:
            liq[col] = pd.to_numeric(liq[col], errors="coerce")
    return liq


def portfolio_return_series(book: pd.DataFrame, lookback: int = 252) -> pd.Series:
    if book.empty or "ticker" not in book.columns:
        return pd.Series(dtype=float)
    tickers = book["ticker"].apply(clean_ticker).tolist()
    rets = get_returns(tickers, lookback=lookback)
    if rets.empty:
        return pd.Series(dtype=float)
    sub = book[book["ticker"].isin(rets.columns)].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    weights = sub.set_index("ticker")["weight"].astype(float)
    weights = weights / max(float(weights.sum()), 1e-12)
    common = [t for t in weights.index if t in rets.columns]
    return (rets[common] * weights.loc[common]).sum(axis=1).dropna()


def portfolio_vol(book: pd.DataFrame, lookback: int = 252) -> float:
    p = portfolio_return_series(book, lookback=lookback)
    if p.empty:
        return np.nan
    return annualized_vol(p)


def source_age(path: Path) -> str:
    if not path.exists():
        return "missing"
    seconds = max(0.0, datetime.now().timestamp() - path.stat().st_mtime)
    hours = seconds / 3600.0
    if hours < 1:
        return f"{hours * 60:.0f}m ago"
    if hours < 48:
        return f"{hours:.1f}h ago"
    return f"{hours / 24:.1f}d ago"


def write_markdown_report(path: Path, title: str, sections: list[str]) -> None:
    body = [f"# {title}", "", f"Generated: {now_str()}", ""]
    body.extend(sections)
    if not body[-1].endswith("\n"):
        body.append("")
    path.write_text("\n".join(body), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """
    Dependency-free markdown table writer.

    pandas.DataFrame.to_markdown requires the optional tabulate package, which
    is not guaranteed in this local environment. This helper keeps reports
    runnable without installing anything.
    """
    if df is None or df.empty:
        return "No rows."
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    work = work.replace([np.inf, -np.inf], np.nan)
    cols = [str(c) for c in work.columns]

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        text = str(value)
        return text.replace("|", "/").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)


def status_rank(label: str) -> int:
    order = {
        "CLEAR": 0,
        "OK": 0,
        "WATCH": 1,
        "REVIEW": 2,
        "SIZE_DOWN": 3,
        "REDUCE_ONLY": 4,
        "BLOCK_NEW": 5,
        "BLOCKED": 5,
        "MISSING_DATA_REVIEW": 3,
    }
    return order.get(str(label).upper(), 2)


def worst_status(labels: list[str]) -> str:
    labels = [str(x).upper() for x in labels if str(x)]
    if not labels:
        return "REVIEW"
    return max(labels, key=status_rank)


def pct(x: float | int | None, digits: int = 2) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "NA"
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "NA"
