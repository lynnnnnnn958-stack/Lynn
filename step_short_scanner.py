#!/usr/bin/env python3
"""
Canyon Short Technical Scanner
================================
Scans S&P 500 for technically strong short candidates:
  - RSI overbought / momentum exhaustion
  - Price extended above moving averages
  - MACD bearish crossover / histogram turning negative
  - Bollinger Band upper break + reversion setup
  - Volume divergence (up on low vol, down on high vol)

For each candidate: entry price range (next 1-5 days), stop loss, target.

Output: short_scanner.csv, short_scanner_report.md
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Parameters ─────────────────────────────────────────────────────────────────
SHORT_SCORE_THRESHOLD = 35   # minimum score to include as short candidate
MAX_TICKERS           = 200
SLEEP_BETWEEN         = 0.8  # seconds between yfinance calls
LOOKBACK              = "14mo"

# Score weights (sum to 100)
W_RSI         = 25
W_MACD        = 20
W_EXTENSION   = 20
W_BB          = 15
W_VOLUME_DIV  = 10
W_MOMENTUM    = 10


# ── Technical indicator calculations ──────────────────────────────────────────
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = close.rolling(n).mean()
    std   = close.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def analyze_ticker(tkr: str) -> dict | None:
    try:
        raw = yf.download(tkr, period=LOOKBACK, progress=False, auto_adjust=True)
        if raw is None or len(raw) < 60:
            return None

        close  = raw["Close"].squeeze().dropna()
        high   = raw["High"].squeeze().dropna()
        low    = raw["Low"].squeeze().dropna()
        volume = raw["Volume"].squeeze().dropna()

        if len(close) < 60:
            return None

        # ── Core indicators ───────────────────────────────────────────────────
        rsi_s  = _rsi(close)
        ma20   = close.rolling(20).mean()
        ma50   = close.rolling(50).mean()
        ma200  = close.rolling(200).mean()
        macd_l, macd_sig, macd_hist = _macd(close)
        bb_up, bb_mid, bb_lo = _bollinger(close)
        atr_s  = _atr(high, low, close)

        # ── Current values ────────────────────────────────────────────────────
        price   = float(close.iloc[-1])
        rsi     = float(rsi_s.iloc[-1])
        ma20v   = float(ma20.iloc[-1])   if not pd.isna(ma20.iloc[-1])   else price
        ma50v   = float(ma50.iloc[-1])   if not pd.isna(ma50.iloc[-1])   else price
        ma200v  = float(ma200.iloc[-1])  if not pd.isna(ma200.iloc[-1])  else price
        atr     = float(atr_s.iloc[-1])  if not pd.isna(atr_s.iloc[-1]) else price * 0.02
        bb_upv  = float(bb_up.iloc[-1])  if not pd.isna(bb_up.iloc[-1]) else price
        bb_midv = float(bb_mid.iloc[-1]) if not pd.isna(bb_mid.iloc[-1]) else price

        macd_hist_now  = float(macd_hist.iloc[-1])
        macd_hist_prev = float(macd_hist.iloc[-2])
        macd_cross     = macd_hist_prev > 0 and macd_hist_now < 0   # just crossed bearish
        macd_negative  = macd_hist_now < 0
        macd_weakening = macd_hist_prev > macd_hist_now              # histogram shrinking

        # ── Extension above MAs ───────────────────────────────────────────────
        ext_20  = (price - ma20v)  / ma20v  if ma20v  > 0 else 0
        ext_50  = (price - ma50v)  / ma50v  if ma50v  > 0 else 0
        ext_200 = (price - ma200v) / ma200v if ma200v > 0 else 0

        # ── BB position ───────────────────────────────────────────────────────
        bb_pct = (price - bb_midv) / (bb_upv - bb_midv) if (bb_upv - bb_midv) > 0 else 0  # 1.0 = at upper band

        # ── Volume divergence (5D) ────────────────────────────────────────────
        # Bearish: price rising but volume declining → exhaustion
        vol5  = volume.tail(5)
        ret5  = close.tail(6).pct_change().dropna()
        up_days_vol   = vol5[ret5.values > 0].mean() if len(vol5[ret5.values > 0]) > 0 else 0
        down_days_vol = vol5[ret5.values < 0].mean() if len(vol5[ret5.values < 0]) > 0 else 0
        vol_diverge   = (down_days_vol > up_days_vol * 1.2) if (up_days_vol > 0 and down_days_vol > 0) else False

        # ── 5-day momentum ────────────────────────────────────────────────────
        ret_5d  = float((close.iloc[-1] / close.iloc[-6] - 1)) if len(close) > 6  else 0
        ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1)) if len(close) > 21 else 0

        # ── 52-week high proximity ────────────────────────────────────────────
        high_52w = float(close.tail(252).max())
        pct_from_high = (high_52w - price) / high_52w  # 0 = at 52w high

        # ── Scoring ───────────────────────────────────────────────────────────
        score = 0.0

        # RSI component
        if rsi >= 80:
            score += W_RSI * 1.0
        elif rsi >= 70:
            score += W_RSI * 0.7
        elif rsi >= 65:
            score += W_RSI * 0.4
        elif rsi >= 60:
            score += W_RSI * 0.15

        # MACD component
        if macd_cross:
            score += W_MACD * 1.0        # fresh bearish cross = strongest signal
        elif macd_negative and macd_weakening:
            score += W_MACD * 0.5
        elif macd_negative:
            score += W_MACD * 0.25

        # Extension component
        max_ext = max(ext_20, ext_50 * 0.8, ext_200 * 0.6)
        if max_ext > 0.25:
            score += W_EXTENSION * 1.0
        elif max_ext > 0.15:
            score += W_EXTENSION * 0.7
        elif max_ext > 0.08:
            score += W_EXTENSION * 0.35

        # Bollinger component
        if bb_pct > 1.0:
            score += W_BB * 1.0          # above upper band
        elif bb_pct > 0.85:
            score += W_BB * 0.6
        elif bb_pct > 0.70:
            score += W_BB * 0.25

        # Volume divergence
        if vol_diverge:
            score += W_VOLUME_DIV

        # Momentum component (negative = already turning)
        if ret_5d < -0.02:
            score += W_MOMENTUM * 0.8    # already rolling over
        elif ret_5d < 0:
            score += W_MOMENTUM * 0.4
        elif ret_5d > 0.05:
            score += W_MOMENTUM * 0.2    # still running up = overbought

        score = min(100, score)

        if score < SHORT_SCORE_THRESHOLD:
            return None

        # ── Short setup: entry range, stop, targets ───────────────────────────
        # Entry: enter short at current or slight bounce (next 1-5 days)
        # Use ATR for volatility-adjusted levels
        entry_ideal = price                                    # current level
        entry_low   = price * 0.98                            # 2% lower (don't chase if already fallen)
        entry_high  = min(price * 1.015, bb_upv * 1.005)     # slight bounce = better entry

        # Stop loss: above recent 10D high + 0.5 ATR
        recent_high_10d = float(high.tail(10).max())
        stop_loss       = round(recent_high_10d + 0.5 * atr, 2)

        # Target 1: revert to 20MA (if price is above)
        target1 = round(ma20v, 2) if price > ma20v * 1.02 else round(price * 0.94, 2)
        # Target 2: revert to 50MA
        target2 = round(ma50v, 2) if price > ma50v * 1.03 else round(price * 0.90, 2)

        risk    = stop_loss - entry_ideal
        reward1 = entry_ideal - target1
        reward2 = entry_ideal - target2
        rr1     = round(reward1 / risk, 1) if risk > 0 else None
        rr2     = round(reward2 / risk, 1) if risk > 0 else None

        # ── Signal label ──────────────────────────────────────────────────────
        signals = []
        if rsi >= 70:
            signals.append(f"RSI={rsi:.0f} Overbought")
        if macd_cross:
            signals.append("MACD Bearish Cross")
        elif macd_negative and macd_weakening:
            signals.append("MACD Weakening")
        if ext_20 > 0.10:
            signals.append(f"+{ext_20:.0%} above 20MA")
        if ext_200 > 0.20:
            signals.append(f"+{ext_200:.0%} above 200MA")
        if bb_pct > 0.9:
            signals.append("Near BB Upper")
        if vol_diverge:
            signals.append("Vol Divergence")

        # Urgency: days until setup expires
        if macd_cross or rsi >= 75:
            urgency = "NOW (1-2d)"
        elif rsi >= 65 or bb_pct > 0.85:
            urgency = "THIS WEEK (3-5d)"
        else:
            urgency = "WATCH (5-10d)"

        return {
            "ticker":        tkr,
            "score":         round(score, 1),
            "price":         round(price, 2),
            # Entry zone
            "entry_low":     round(entry_low, 2),
            "entry_ideal":   round(entry_ideal, 2),
            "entry_high":    round(entry_high, 2),
            # Risk management
            "stop_loss":     stop_loss,
            "target_1":      target1,
            "target_2":      target2,
            "rr_1":          rr1,
            "rr_2":          rr2,
            "atr":           round(atr, 2),
            # Technical readings
            "rsi":           round(rsi, 1),
            "ext_20ma":      round(ext_20 * 100, 1),
            "ext_50ma":      round(ext_50 * 100, 1),
            "ext_200ma":     round(ext_200 * 100, 1),
            "bb_pct":        round(bb_pct * 100, 1),
            "macd_cross":    macd_cross,
            "vol_diverge":   vol_diverge,
            "ret_5d":        round(ret_5d * 100, 1),
            "pct_from_high": round(pct_from_high * 100, 1),
            "ma20":          round(ma20v, 2),
            "ma50":          round(ma50v, 2),
            "ma200":         round(ma200v, 2),
            # Setup description
            "signals":       " · ".join(signals[:4]),
            "urgency":       urgency,
            "as_of":         date.today().isoformat(),
        }

    except Exception as e:
        return None


# ── Report writer ──────────────────────────────────────────────────────────────
def _write_report(df: pd.DataFrame):
    today = date.today().isoformat()
    df = df[df["as_of"] == today] if "as_of" in df.columns else df

    lines = [
        "# Canyon Short Technical Scanner",
        f"*{today} · RSI + MACD + Extension + Bollinger + Volume · Score ≥ {SHORT_SCORE_THRESHOLD}*",
        "",
        f"**{len(df)} short candidates found** (score threshold: {SHORT_SCORE_THRESHOLD}/100)",
        "",
        "| # | Ticker | Score | Price | Entry Range | Stop | Target1 | Target2 | R/R | RSI | Signals | Urgency |",
        "|---|--------|-------|-------|-------------|------|---------|---------|-----|-----|---------|---------|",
    ]
    for i, (_, r) in enumerate(df.iterrows(), 1):
        lines.append(
            f"| {i} | **{r['ticker']}** | {r['score']:.0f} | ${r['price']:.2f} "
            f"| ${r['entry_low']:.2f}–${r['entry_high']:.2f} "
            f"| ${r['stop_loss']:.2f} | ${r['target_1']:.2f} | ${r['target_2']:.2f} "
            f"| {r['rr_1']}x | {r['rsi']:.0f} | {r['signals']} | {r['urgency']} |"
        )

    lines += [
        "",
        "---",
        "## How to Read This Table",
        "",
        "- **Entry Range**: short the stock anywhere in this range over the next 1-5 days",
        "- **Stop Loss**: close the short if price reaches this level (capital protection)",
        "- **Target 1**: conservative target (20MA reversion); **Target 2**: full target (50MA)",
        "- **R/R**: reward-to-risk ratio. ≥ 2.0 = favorable setup",
        "- **Score**: 0-100 composite of 5 technical signals. Higher = stronger short case",
        "- **Urgency**: how quickly to act before setup expires",
    ]

    (ROOT / "short_scanner_report.md").write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Canyon Short Scanner — {date.today()}")
    print(f"  Score threshold: {SHORT_SCORE_THRESHOLD}/100")
    print("=" * 60)

    # Universe: env var CANYON_UNIVERSE overrides default (sp500)
    universe = os.environ.get("CANYON_UNIVERSE", "")
    universe_path = ROOT / f"universe_{universe}.csv" if universe else None
    scores_path   = ROOT / "alpha_scores.csv"

    if universe_path and universe_path.exists():
        scores  = pd.read_csv(universe_path)
        ticker_col = "ticker" if "ticker" in scores.columns else scores.columns[0]
        tickers = scores[ticker_col].dropna().tolist()[:MAX_TICKERS]
        print(f"  Universe: {universe} ({len(tickers)} tickers from {universe_path.name})")
    elif scores_path.exists():
        scores  = pd.read_csv(scores_path)
        tickers = scores["ticker"].tolist()[:MAX_TICKERS]
        print(f"  Universe: S&P 500 alpha_scores ({len(tickers)} tickers)")
        print(f"  Tip: run 'python step_get_universe.py russell1000' then set CANYON_UNIVERSE=russell1000")
    else:
        print("ERROR: alpha_scores.csv not found"); return

    print(f"  Scanning {len(tickers)} tickers …\n")

    results = []
    ok = fail = 0
    for i, tkr in enumerate(tickers, 1):
        r = analyze_ticker(tkr)
        if r:
            results.append(r)
            rr_str = f"R/R={r['rr_1']}x" if r['rr_1'] else ""
            print(f"  [{i:3d}] ★ {tkr:6s}  score={r['score']:4.0f}  "
                  f"RSI={r['rsi']:.0f}  entry=${r['entry_low']:.2f}–${r['entry_high']:.2f}  "
                  f"stop=${r['stop_loss']:.2f}  t1=${r['target_1']:.2f}  {rr_str}  {r['urgency']}")
            ok += 1
        else:
            fail += 1
        time.sleep(SLEEP_BETWEEN)

    if results:
        df = pd.DataFrame(results).sort_values("score", ascending=False)
        df.to_csv(ROOT / "short_scanner.csv", index=False)
        _write_report(df)
        print(f"\n  {ok} candidates  |  {fail} filtered out")
        print(f"  Saved → short_scanner.csv, short_scanner_report.md")
    else:
        print("  No candidates above threshold today.")


if __name__ == "__main__":
    main()
