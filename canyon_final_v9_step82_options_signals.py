#!/usr/bin/env python3
"""
canyon_final_v9_step82_options_signals.py
=========================================
Institutional-grade options trading signal engine.
Canyon v9 — Step 82 (Wall Street desk rewrite).

ARCHITECTURE
  Layer 1 · Data    — near + next-expiry chains, earnings calendar
  Layer 2 · IV      — ATM IV, IV Rank (30d rolling history), term structure, 25Δ skew
  Layer 3 · Flow    — Net call premium ($), PCR vol+OI, Unusual Options Activity (UOA)
  Layer 4 · Gamma   — Dealer GEX (Black-Scholes approx), max-pain, squeeze risk
  Layer 5 · Alpha   — Weighted composite → rank_options 0-100
  Layer 6 · Edge    — Strategy recommendation + conviction (★ 1-5)

SIGNAL WEIGHTS
  Flow Score   35%  (where the money flows: premium + UOA)
  IV Regime    25%  (cheap/expensive vol — strategy selector)
  Skew Score   20%  (OTM put/call IV spread — institutional directional bias)
  Gamma Score  20%  (squeeze & pin dynamics)

STRATEGIES (actionable, not vague)
  LONG_CALLS           strong call flow + low IV rank
  BULL_CALL_SPREAD     directional call flow + moderate IV (defined risk)
  BULL_RISK_REVERSAL   high PCR (fear spike) + flow reversing → synthetic long
  OTM_CALL_BACKSPREAD  UOA on OTM calls + negative GEX = gamma squeeze play
  SHORT_PUT            high IV + PCR > 1.2 + bullish → income + discounted entry
  LONG_STRADDLE        IV rank < 30 + catalyst/earnings → buy cheap vol
  IRON_CONDOR          IV rank > 70 + neutral + positive GEX → harvest vol premium
  PROTECTIVE_PUT       PCR < 0.3 (complacency) + cheap puts → hedge crowded longs
  MONITOR              insufficient data or mixed signals

Outputs:
  options_signals.csv  — ticker, rank_options, options_strategy, conviction, iv_rank,
                         atm_iv, iv_skew, pcr_vol, pcr_oi, net_call_premium, uoa_flag,
                         uoa_detail, gex_sign, gex_net, squeeze_risk, max_pain,
                         max_pain_dist, flow_score, iv_score, skew_score, gamma_score,
                         alpha_options, days_to_earnings, term_structure, expiry
  options_report.md    — morning desk briefing (high-conviction setups, UOA, GEX alerts)
  iv_history.csv       — 30d rolling ATM IV per ticker (source for IV rank)

Usage:
  python3 canyon_final_v9_step82_options_signals.py [--top N] [--workers N] [--refresh]
"""

from __future__ import annotations

import argparse
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent
ML_SCORES    = ROOT / "regime_ml_scores.csv"
OUT_SCORES   = ROOT / "options_signals.csv"
OUT_REPORT   = ROOT / "options_report.md"
CACHE_FILE   = ROOT / "options_raw_cache.csv"
IV_HISTORY   = ROOT / "iv_history.csv"

CACHE_TTL_H  = 4        # hours before refreshing options chain cache
IV_HIST_DAYS = 30       # rolling window for IV rank
DEFAULT_TOP  = 100
MAX_WORKERS  = 4
MIN_OI       = 50       # minimum OI per strike (filter illiquid strikes)
UOA_VO_RATIO = 3.0      # unusual activity: single-strike vol / OI > 3×
UOA_MIN_VOL  = 200      # minimum contract volume for UOA flag
SKEW_LOW     = 0.05     # OTM band lower bound (5% from spot)
SKEW_HIGH    = 0.20     # OTM band upper bound (20% from spot)


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers (no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _gamma_bs(S: float, K: float, sigma: float, T: float) -> float:
    """
    Black-Scholes gamma: γ = N'(d₁) / (S · σ · √T)
    Used to approximate dealer gamma exposure without paid data.
    """
    if sigma < 0.01 or T <= 0.0 or K <= 0.0 or S <= 0.0:
        return 0.0
    try:
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────────────

def get_universe(top_n: int) -> list[str]:
    """Top N tickers by ML predicted score (Step 77)."""
    if ML_SCORES.exists():
        df = pd.read_csv(ML_SCORES)
        col = "predicted_score" if "predicted_score" in df.columns else df.columns[-1]
        return df.sort_values(col, ascending=False)["ticker"].dropna().tolist()[:top_n]
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","AMD","TSLA","AVGO","COST",
        "JPM","UNH","LLY","V","MA","NFLX","BKNG","CRWD","PANW","PLTR",
    ][:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Expiry selection
# ─────────────────────────────────────────────────────────────────────────────

def _pick_two_expiries(expiries: tuple) -> tuple[str | None, str | None]:
    """
    Return (near, next) expiries.
    near  ≥  7d from today, closest to 30d
    next  ≥ 35d from today, used for term-structure comparison
    """
    today = datetime.today().date()
    valid: list[tuple[int, str]] = []
    for e in expiries:
        try:
            d    = datetime.strptime(e, "%Y-%m-%d").date()
            days = (d - today).days
            if days >= 7:
                valid.append((days, e))
        except Exception:
            continue
    valid.sort()
    near = valid[0][1] if valid else None
    nxt  = next((e for d, e in valid if d >= 35), None)
    return near, nxt


# ─────────────────────────────────────────────────────────────────────────────
# Max pain
# ─────────────────────────────────────────────────────────────────────────────

def _max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """
    Strike price that maximises option-seller profit (minimises total intrinsic value).
    Stocks frequently drift toward max-pain near expiry as market makers hedge.
    """
    strikes = sorted(set(
        calls["strike"].dropna().tolist() + puts["strike"].dropna().tolist()
    ))
    if not strikes:
        return 0.0

    call_oi = calls.set_index("strike")["openInterest"].fillna(0)
    put_oi  = puts.set_index("strike")["openInterest"].fillna(0)

    best_price  = strikes[len(strikes) // 2]
    best_pain   = float("inf")
    for price in strikes:
        pain = sum(max(0.0, price - k) * oi for k, oi in call_oi.items())
        pain += sum(max(0.0, k - price) * oi for k, oi in put_oi.items())
        if pain < best_pain:
            best_pain  = pain
            best_price = price
    return float(best_price)


# ─────────────────────────────────────────────────────────────────────────────
# Dealer Gamma Exposure (GEX)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_gex(
    calls: pd.DataFrame,
    puts:  pd.DataFrame,
    spot:  float,
    T:     float,
) -> dict:
    """
    Approximate dealer Gamma Exposure (GEX).

    Assumption: market makers are net short options to directional traders.
    GEX = Σ(call_OI × γ) − Σ(put_OI × γ)   [units: $M per 1% spot move]

    Positive GEX  → dealers net long gamma → buy dips / sell rips → STABILISING
                    → mean-reversion regime, iron-condor friendly
    Negative GEX  → dealers net short gamma → buy rips / sell dips → DESTABILISING
                    → momentum/breakout regime, gamma-squeeze risk

    squeeze_risk: negative GEX + heavy ATM call OI = dealers need to buy stock
                  aggressively as price rises → explosive upside potential
    """
    call_gex = 0.0
    put_gex  = 0.0

    for _, row in calls.iterrows():
        K     = float(row.get("strike", 0) or 0)
        sigma = float(row.get("impliedVolatility", 0) or 0)
        oi    = float(row.get("openInterest", 0) or 0)
        if K > 0 and sigma > 0.01 and oi >= MIN_OI:
            call_gex += _gamma_bs(spot, K, sigma, T) * oi * spot ** 2 * 0.01

    for _, row in puts.iterrows():
        K     = float(row.get("strike", 0) or 0)
        sigma = float(row.get("impliedVolatility", 0) or 0)
        oi    = float(row.get("openInterest", 0) or 0)
        if K > 0 and sigma > 0.01 and oi >= MIN_OI:
            put_gex += _gamma_bs(spot, K, sigma, T) * oi * spot ** 2 * 0.01

    net_gex = call_gex - put_gex

    # Squeeze risk: heavy call OI near ATM + dealers net short gamma
    atm_calls = calls[
        (calls["strike"] >= spot * 0.97) & (calls["strike"] <= spot * 1.05)
    ]
    atm_call_oi = float(atm_calls["openInterest"].fillna(0).sum())
    squeeze_risk = bool(
        net_gex < 0
        and call_gex > 0
        and atm_call_oi >= 500
        and call_gex / max(abs(net_gex), 1e-9) > 0.6
    )

    return {
        "gex_net":      round(net_gex,   3),
        "gex_calls":    round(call_gex,  3),
        "gex_puts":     round(put_gex,   3),
        "gex_sign":     1 if net_gex > 0 else (-1 if net_gex < 0 else 0),
        "squeeze_risk": squeeze_risk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Advanced per-ticker fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_one(ticker: str) -> dict:
    """
    Fetch full options intelligence for a single ticker.

    Computes:
      atm_iv          — ATM implied volatility (near expiry)
      iv_skew         — OTM put IV − OTM call IV  (positive = fear)
      pcr_vol/pcr_oi  — put/call volume & OI ratios
      net_call_prem   — net $ premium flowing into calls vs puts
      uoa_flag        — unusual call activity (vol/OI > 3× at a strike)
      uoa_bear_flag   — unusual put activity
      uoa_detail      — human-readable description of top UOA
      gex_*           — dealer gamma exposure fields
      max_pain        — max-pain strike price
      max_pain_dist   — (max_pain − spot) / spot × 100  (% distance)
      term_structure  — near_ATM_IV / next_ATM_IV  (> 1 = backwardation)
      days_to_earnings — calendar days to next earnings event
    """
    import yfinance as yf

    res: dict = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)

        # ── Spot ──────────────────────────────────────────────────────────────
        try:
            spot = float(t.fast_info.last_price or t.fast_info.regularMarketPrice or 0)
        except Exception:
            spot = 0.0
        res["spot"] = round(spot, 2)

        if not t.options or spot <= 0:
            res["error"] = "no options or no price"
            return res

        # ── Expiries ─────────────────────────────────────────────────────────
        near_exp, next_exp = _pick_two_expiries(t.options)
        if near_exp is None:
            res["error"] = "no valid expiry"
            return res

        res["expiry"] = near_exp
        today   = datetime.today().date()
        T_near  = max((datetime.strptime(near_exp, "%Y-%m-%d").date() - today).days, 1) / 365.0

        # ── Near-expiry chain ─────────────────────────────────────────────────
        chain = t.option_chain(near_exp)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        for df_opt in [calls, puts]:
            for col in ["impliedVolatility", "volume", "openInterest", "lastPrice"]:
                df_opt[col] = pd.to_numeric(df_opt.get(col, np.nan), errors="coerce").fillna(0)

        # ── ATM IV ────────────────────────────────────────────────────────────
        atm_mask  = (calls["strike"] >= spot * 0.97) & (calls["strike"] <= spot * 1.03)
        atm_calls = calls[atm_mask] if atm_mask.any() else calls
        atm_iv    = float(atm_calls["impliedVolatility"].replace(0, np.nan).median() or 0.0)
        res["atm_iv"] = round(atm_iv, 4)

        # ── 25Δ Skew: OTM put IV − OTM call IV ───────────────────────────────
        otm_put  = puts[
            (puts["strike"]  >= spot * (1 - SKEW_HIGH)) &
            (puts["strike"]  <= spot * (1 - SKEW_LOW))  &
            (puts["openInterest"] >= MIN_OI)
        ]
        otm_call = calls[
            (calls["strike"] >= spot * (1 + SKEW_LOW))  &
            (calls["strike"] <= spot * (1 + SKEW_HIGH)) &
            (calls["openInterest"] >= MIN_OI)
        ]
        put_iv_otm  = float(otm_put["impliedVolatility"].replace(0, np.nan).median()  or atm_iv)
        call_iv_otm = float(otm_call["impliedVolatility"].replace(0, np.nan).median() or atm_iv)
        res["iv_skew"]     = round(put_iv_otm - call_iv_otm, 4)  # >0 = fear/put-heavy
        res["otm_put_iv"]  = round(put_iv_otm,  4)
        res["otm_call_iv"] = round(call_iv_otm, 4)

        # ── PCR (volume + OI) ─────────────────────────────────────────────────
        call_vol = float(calls["volume"].sum())
        put_vol  = float(puts["volume"].sum())
        call_oi  = float(calls["openInterest"].sum())
        put_oi   = float(puts["openInterest"].sum())
        res.update(
            call_vol  = int(call_vol),
            put_vol   = int(put_vol),
            call_oi   = int(call_oi),
            put_oi    = int(put_oi),
            total_oi  = int(call_oi + put_oi),
            pcr_vol   = round(put_vol / max(call_vol, 1), 4),
            pcr_oi    = round(put_oi  / max(call_oi,  1), 4),
        )

        # ── Net call premium ($) ──────────────────────────────────────────────
        # Dollar premium flowing into calls vs puts (vol × lastPrice × 100 shares)
        call_prem = float((calls["volume"] * calls["lastPrice"]).sum() * 100)
        put_prem  = float((puts["volume"]  * puts["lastPrice"]).sum()  * 100)
        res["net_call_premium"] = round(call_prem - put_prem, 0)

        # ── Unusual Options Activity (UOA) ────────────────────────────────────
        uoa_calls: list[dict] = []
        uoa_puts:  list[dict] = []

        for opt_df, opt_type, collector in [
            (calls, "CALL", uoa_calls),
            (puts,  "PUT",  uoa_puts),
        ]:
            for _, row in opt_df.iterrows():
                vol    = float(row.get("volume", 0) or 0)
                oi     = float(row.get("openInterest", 0) or 0)
                strike = float(row.get("strike", 0))
                iv     = float(row.get("impliedVolatility", 0) or 0)
                if vol < UOA_MIN_VOL or oi <= 0:
                    continue
                ratio = vol / oi
                if ratio >= UOA_VO_RATIO:
                    collector.append({
                        "type":      opt_type,
                        "strike":    strike,
                        "vol":       int(vol),
                        "oi":        int(oi),
                        "ratio":     round(ratio, 1),
                        "iv":        round(iv, 4),
                        "moneyness": round((strike - spot) / max(spot, 1) * 100, 1),
                    })

        uoa_calls.sort(key=lambda x: x["vol"], reverse=True)
        uoa_puts.sort( key=lambda x: x["vol"], reverse=True)

        res["uoa_flag"]      = 1 if uoa_calls else 0
        res["uoa_bear_flag"] = 1 if uoa_puts  else 0
        res["uoa_count"]     = len(uoa_calls) + len(uoa_puts)

        if uoa_calls:
            top = uoa_calls[0]
            res["uoa_detail"] = (
                f"CALL K={top['strike']:.0f}  "
                f"vol={top['vol']:,} / OI={top['oi']:,}  "
                f"ratio={top['ratio']:.1f}×  IV={top['iv']:.0%}"
            )
        elif uoa_puts:
            top = uoa_puts[0]
            res["uoa_detail"] = (
                f"PUT  K={top['strike']:.0f}  "
                f"vol={top['vol']:,} / OI={top['oi']:,}  "
                f"ratio={top['ratio']:.1f}×  IV={top['iv']:.0%}"
            )
        else:
            res["uoa_detail"] = ""

        # ── GEX ───────────────────────────────────────────────────────────────
        gex = _compute_gex(calls, puts, spot, T_near)
        res.update(gex)

        # ── Max pain ──────────────────────────────────────────────────────────
        mp = _max_pain(calls, puts)
        res["max_pain"]      = round(mp, 2)
        res["max_pain_dist"] = round((mp - spot) / max(spot, 1) * 100, 2)

        # ── Term structure (near vs next expiry IV) ───────────────────────────
        if next_exp:
            try:
                T_next = max(
                    (datetime.strptime(next_exp, "%Y-%m-%d").date() - today).days, 1
                ) / 365.0
                c2 = t.option_chain(next_exp).calls.copy()
                c2["impliedVolatility"] = pd.to_numeric(
                    c2.get("impliedVolatility", np.nan), errors="coerce"
                ).fillna(0)
                atm2 = c2[(c2["strike"] >= spot * 0.97) & (c2["strike"] <= spot * 1.03)]
                if atm2.empty:
                    atm2 = c2
                iv_next = float(atm2["impliedVolatility"].replace(0, np.nan).median() or 0)
                res["atm_iv_next"]  = round(iv_next, 4)
                # > 1 = inverted (near > far) = event-driven / fear spike
                res["term_structure"] = round(atm_iv / iv_next, 3) if iv_next > 0 else np.nan
            except Exception:
                res["atm_iv_next"]   = np.nan
                res["term_structure"] = np.nan

        # ── Earnings proximity ────────────────────────────────────────────────
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, pd.DataFrame):
                    raw_dates = (
                        cal.loc["Earnings Date"].tolist()
                        if "Earnings Date" in cal.index else []
                    )
                elif isinstance(cal, dict):
                    raw_dates = cal.get("Earnings Date", []) or []
                else:
                    raw_dates = []
                future = [
                    pd.Timestamp(d).date()
                    for d in raw_dates
                    if pd.notna(d) and pd.Timestamp(d).date() >= today
                ]
                if future:
                    res["days_to_earnings"] = int((min(future) - today).days)
        except Exception:
            pass

    except Exception as e:
        res["error"] = str(e)[:120]

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Batch fetch with 4-hour cache
# ─────────────────────────────────────────────────────────────────────────────

def load_options_data(
    tickers: list[str],
    max_workers: int = MAX_WORKERS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Batch fetch options data with 4h cache."""
    tickers_to_fetch = tickers

    if not force_refresh and CACHE_FILE.exists():
        age_h = (datetime.now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < CACHE_TTL_H:
            cached  = pd.read_csv(CACHE_FILE)
            covered = set(cached["ticker"].tolist())
            missing = [t for t in tickers if t not in covered]
            if not missing:
                print(f"  Cache hit: {len(cached)} tickers ({age_h:.1f}h old)")
                return cached[cached["ticker"].isin(tickers)].copy()
            print(f"  Cache partial: {len(covered)} cached, fetching {len(missing)} new …")
            tickers_to_fetch = missing
        else:
            print(f"  Cache stale ({age_h:.1f}h) — refreshing {len(tickers)} …")

    rows: list[dict] = []
    done  = 0
    total = len(tickers_to_fetch)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers_to_fetch}
        for fut in as_completed(futures):
            res = fut.result()
            if "error" not in res:
                rows.append(res)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  … {done}/{total}")

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if CACHE_FILE.exists() and not force_refresh and tickers_to_fetch != tickers:
        old_df  = pd.read_csv(CACHE_FILE)
        combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
            subset="ticker", keep="last"
        )
    else:
        combined = new_df

    if not combined.empty:
        combined.to_csv(CACHE_FILE, index=False)

    return combined[combined["ticker"].isin(tickers)].copy() if not combined.empty else combined


# ─────────────────────────────────────────────────────────────────────────────
# IV History & IV Rank
# ─────────────────────────────────────────────────────────────────────────────

def update_iv_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append today's ATM IV to iv_history.csv (rolling IV_HIST_DAYS window).
    Compute IV Rank = (current_IV − 30d_low) / (30d_high − 30d_low) × 100.
    Falls back to cross-sectional IV percentile where history is insufficient (< 5 days).
    """
    today_str = datetime.today().strftime("%Y-%m-%d")

    hist = pd.read_csv(IV_HISTORY) if IV_HISTORY.exists() else pd.DataFrame(
        columns=["date", "ticker", "atm_iv"]
    )

    # Replace today's entries
    hist = hist[hist["date"] != today_str]
    new_rows = df[["ticker", "atm_iv"]].dropna().assign(date=today_str)
    hist = pd.concat([hist, new_rows[["date", "ticker", "atm_iv"]]], ignore_index=True)

    # Trim to rolling window
    hist["date"] = pd.to_datetime(hist["date"])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=IV_HIST_DAYS)
    hist   = hist[hist["date"] >= cutoff]
    hist.to_csv(IV_HISTORY, index=False)

    # Compute IV rank per ticker
    curr_ivs = df.set_index("ticker")["atm_iv"].to_dict()
    iv_ranks: dict[str, float] = {}

    for ticker, grp in hist.groupby("ticker"):
        ivs = grp["atm_iv"].dropna()
        if len(ivs) < 5:
            continue
        iv_lo = ivs.min()
        iv_hi = ivs.max()
        iv_cur = curr_ivs.get(ticker, np.nan)  # type: ignore[arg-type]
        if pd.isna(iv_cur) or iv_hi <= iv_lo:
            continue
        iv_ranks[str(ticker)] = round((iv_cur - iv_lo) / (iv_hi - iv_lo) * 100, 1)

    df = df.copy()
    df["iv_rank"] = df["ticker"].map(iv_ranks)

    # Cross-sectional fallback for tickers with insufficient history
    valid = df["atm_iv"].notna()
    cross_pct = df.loc[valid, "atm_iv"].rank(pct=True) * 100
    needs_fill = df["iv_rank"].isna() & valid
    df.loc[needs_fill, "iv_rank"] = cross_pct[needs_fill].round(1)
    df["iv_rank"] = df["iv_rank"].fillna(50.0)

    n_hist = sum(1 for t in df["ticker"] if t in iv_ranks)
    print(f"  IV Rank: {n_hist}/{len(df)} from {IV_HIST_DAYS}d history, "
          f"{len(df) - n_hist} cross-sectional fallback")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sub-score functions
# ─────────────────────────────────────────────────────────────────────────────

def _iv_regime_score(iv_rank: float) -> float:
    """
    IV regime → buy-vol opportunity score (0–100).

    Low IV rank = cheap options = high leverage value for directional bets → high score
    High IV rank = expensive options = buy-vol disadvantage → low score
    (Premium sellers see it reversed: use iron condor when iv_rank > 70)
    """
    if   iv_rank <= 15: return 90.0
    elif iv_rank <= 25: return 78.0
    elif iv_rank <= 40: return 64.0
    elif iv_rank <= 55: return 50.0
    elif iv_rank <= 65: return 38.0
    elif iv_rank <= 80: return 24.0
    else:               return 12.0


def _flow_score(
    pcr_vol:           float,
    net_call_premium:  float,
    uoa_flag:          int,
    uoa_bear_flag:     int,
) -> float:
    """
    Options flow → directional alpha score (0–100).

    Combines:
      • PCR (volume): contrarian — very low PCR is dangerous (crowded calls),
                      very high PCR is opportunity (capitulation)
      • Net call premium: dollar vote — where institutional money is flowing
      • UOA: single-strike unusual activity = informed positioning
    """
    score = 50.0

    # PCR: nonlinear / contrarian
    if   pcr_vol < 0.30:  score -= 22.0   # extreme call-crowding = danger
    elif pcr_vol < 0.60:  score += 14.0   # call-dominant = bullish
    elif pcr_vol < 0.90:  score +=  5.0   # slightly bullish
    elif pcr_vol < 1.30:  score -=  5.0   # mild put bias
    elif pcr_vol < 2.00:  score += 10.0   # fear = contrarian buy
    else:                 score += 20.0   # capitulation = strong buy

    # Net call premium ($) — most direct institutional signal
    if   net_call_premium >  5_000_000:  score += 20.0
    elif net_call_premium >  1_000_000:  score += 12.0
    elif net_call_premium >          0:  score +=  5.0
    elif net_call_premium < -5_000_000:  score -= 20.0
    elif net_call_premium < -1_000_000:  score -= 12.0
    elif net_call_premium <          0:  score -=  5.0

    # UOA: institutional positioning signal
    if uoa_flag:      score += 15.0
    if uoa_bear_flag: score -= 12.0

    return float(np.clip(score, 0.0, 100.0))


def _skew_score(iv_skew: float, atm_iv: float) -> float:
    """
    25Δ skew → institutional directional bias score (0–100).

    iv_skew = OTM_put_IV − OTM_call_IV
      > 0  (put-heavy): institutions buying downside protection → fearful → lower score
      < 0  (call-heavy): institutions chasing upside → aggressive → higher score
      ≈ 0  (flat): neutral

    Normalised by ATM IV to handle varying overall vol regimes.
    """
    if pd.isna(iv_skew) or atm_iv <= 0:
        return 50.0

    rel = iv_skew / max(atm_iv, 0.01)   # relative skew

    if   rel < -0.20: return 92.0   # aggressive upside chase (calls vastly pricier)
    elif rel < -0.08: return 75.0   # mild upside bias
    elif rel <  0.05: return 58.0   # near-flat: neutral-to-slightly-bullish
    elif rel <  0.15: return 46.0   # normal fear skew
    elif rel <  0.30: return 34.0   # elevated fear
    elif rel <  0.50: return 22.0   # high fear
    else:             return 14.0   # extreme fear/crush (contrarian — near capitulation)


def _gamma_score(
    gex_sign:     int,
    squeeze_risk: bool,
    max_pain_dist: float,
    uoa_flag:     int,
) -> float:
    """
    Gamma dynamics → explosive upside potential score (0–100).

    Negative GEX + squeeze risk + UOA calls = highest gamma score
    Positive GEX (range-bound, dealers dampen moves) = lower score
    Max pain above spot = gravity pulling price upward
    """
    score = 50.0

    if gex_sign < 0:       # dealers short gamma = amplifying moves
        score += 18.0
        if squeeze_risk:
            score += 16.0  # gamma squeeze potential (GME-style)
    elif gex_sign > 0:     # dealers long gamma = dampen moves
        score -= 10.0

    # Max pain magnetism
    if   max_pain_dist >  5.0: score += 12.0   # strong upward pull
    elif max_pain_dist >  2.0: score +=  6.0
    elif max_pain_dist >  0.0: score +=  3.0
    elif max_pain_dist < -3.0: score -=  8.0   # downward pull

    # UOA + negative GEX amplifies squeeze
    if uoa_flag and gex_sign < 0:
        score += 8.0

    return float(np.clip(score, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────────────────
# Strategy recommendation engine
# ─────────────────────────────────────────────────────────────────────────────

def _recommend_strategy(row: pd.Series) -> tuple[str, int]:
    """
    Multi-factor strategy selector.  Returns (strategy, conviction_1_to_5).

    Decision logic mirrors a Wall Street options desk framework:
      1. Earnings straddle — buy cheap vol before a catalyst
      2. Gamma squeeze / OTM backspread — negative GEX + UOA calls
      3. Strong directional flow — UOA + net premium
      4. Fear reversal (risk reversal) — high PCR + flow turning
      5. Income / short put — high IV + bullish
      6. Range vol-sell (iron condor) — very high IV + positive GEX
      7. Cheap-vol directional — low IV rank + positive flow
      8. Crowding hedge (protective put) — extreme call imbalance
    """
    iv_rank    = float(row.get("iv_rank",        50) or 50)
    pcr_vol    = float(row.get("pcr_vol",         1) or  1)
    uoa        = int(  row.get("uoa_flag",         0) or  0)
    uoa_bear   = int(  row.get("uoa_bear_flag",    0) or  0)
    gex_sign   = int(  row.get("gex_sign",         0) or  0)
    squeeze    = bool( row.get("squeeze_risk",  False))
    net_prem   = float(row.get("net_call_premium", 0) or  0)
    mp_dist    = float(row.get("max_pain_dist",    0) or  0)
    flow_s     = float(row.get("flow_score",      50) or 50)

    days_earn_raw = row.get("days_to_earnings", 999)
    days_earn = int(days_earn_raw) if pd.notna(days_earn_raw) else 999

    # 1. Pre-earnings straddle (IV cheap, catalyst in ≤14 days)
    if days_earn <= 14 and iv_rank < 40:
        if days_earn <= 7 and iv_rank < 25:
            return "LONG_STRADDLE", 5
        if iv_rank < 30:
            return "LONG_STRADDLE", 4
        return "LONG_STRADDLE", 3

    # 2. Gamma squeeze / OTM backspread
    #    Sell 1× ATM call, buy 2× OTM calls — self-financing, unlimited upside
    if uoa and gex_sign < 0 and squeeze:
        c = 5 if (flow_s > 70 and iv_rank < 50) else 4
        return "OTM_CALL_BACKSPREAD", c

    # 3a. Strong directional call flow + cheap vol → outright calls
    if flow_s >= 72 and uoa and not uoa_bear and iv_rank < 35:
        c = 5 if net_prem > 2_000_000 else 4
        return "LONG_CALLS", c

    # 3b. Strong directional call flow + moderate IV → defined-risk spread
    if flow_s >= 68 and uoa and not uoa_bear and iv_rank < 60:
        c = 4 if net_prem > 1_000_000 else 3
        return "BULL_CALL_SPREAD", c

    # 4. Fear reversal: high PCR (capitulation) + flow improving
    #    Sell OTM put, buy OTM call — synthetic long funded by fear premium
    if pcr_vol >= 1.5 and flow_s >= 55:
        c = 4 if (pcr_vol >= 2.0 and gex_sign >= 0) else 3
        return "BULL_RISK_REVERSAL", c

    # 5. Short put: high IV + slight fear + bullish max-pain tilt + income
    if iv_rank >= 60 and pcr_vol >= 0.9 and mp_dist >= 0:
        c = 4 if iv_rank >= 75 else 3
        return "SHORT_PUT", c

    # 6. Iron condor: very high IV + dealers stabilising + no catalyst
    if iv_rank >= 70 and gex_sign > 0 and 0.5 <= pcr_vol <= 1.3 and days_earn > 21:
        return "IRON_CONDOR", 3

    # 7. Cheap vol + positive flow → outright calls
    if iv_rank <= 25 and flow_s >= 58:
        return "LONG_CALLS", 3

    # 8. Moderate positive flow + moderate IV → spread
    if flow_s >= 60 and iv_rank < 55:
        return "BULL_CALL_SPREAD", 2

    # 9. Crowded calls (complacency) + cheap puts → hedge
    if pcr_vol < 0.30 and iv_rank < 25:
        return "PROTECTIVE_PUT", 2

    return "MONITOR", 1


# ─────────────────────────────────────────────────────────────────────────────
# Signal assembly
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(raw: pd.DataFrame) -> pd.DataFrame:
    """Assemble all sub-scores → composite rank_options + strategy recommendation."""
    df = raw.copy()

    # ── Coerce numerics ───────────────────────────────────────────────────────
    for c in ["atm_iv", "iv_skew", "pcr_vol", "pcr_oi", "net_call_premium",
              "gex_net", "gex_sign", "max_pain_dist", "iv_rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["uoa_flag", "uoa_bear_flag", "uoa_count"]:
        df[c] = df.get(c, pd.Series(0, index=df.index)).fillna(0).astype(int)

    # ── Safe defaults ─────────────────────────────────────────────────────────
    df["pcr_vol"]           = df.get("pcr_vol",           pd.Series(1.0,  index=df.index)).fillna(1.0)
    df["pcr_oi"]            = df.get("pcr_oi",            pd.Series(1.0,  index=df.index)).fillna(1.0)
    df["atm_iv"]            = df.get("atm_iv",            pd.Series(0.30, index=df.index)).fillna(0.30)
    df["iv_skew"]           = df.get("iv_skew",           pd.Series(0.0,  index=df.index)).fillna(0.0)
    df["gex_sign"]          = df.get("gex_sign",          pd.Series(0,    index=df.index)).fillna(0).astype(int)
    df["max_pain_dist"]     = df.get("max_pain_dist",     pd.Series(0.0,  index=df.index)).fillna(0.0)
    df["net_call_premium"]  = df.get("net_call_premium",  pd.Series(0.0,  index=df.index)).fillna(0.0)
    df["squeeze_risk"]      = df.get("squeeze_risk",      pd.Series(False, index=df.index)).fillna(False)
    df["squeeze_risk"]      = df["squeeze_risk"].apply(
        lambda x: bool(x) if not isinstance(x, bool) else x
    )

    # ── Layer scores ─────────────────────────────────────────────────────────
    df["iv_score"] = df["iv_rank"].apply(
        lambda x: _iv_regime_score(float(x) if pd.notna(x) else 50.0)
    )
    df["flow_score"] = df.apply(
        lambda r: _flow_score(r["pcr_vol"], r["net_call_premium"],
                              r["uoa_flag"], r["uoa_bear_flag"]),
        axis=1,
    )
    df["skew_score"] = df.apply(
        lambda r: _skew_score(r["iv_skew"], r["atm_iv"]), axis=1
    )
    df["gamma_score"] = df.apply(
        lambda r: _gamma_score(
            int(r["gex_sign"]), bool(r["squeeze_risk"]),
            float(r["max_pain_dist"]), int(r["uoa_flag"])
        ),
        axis=1,
    )

    # ── Composite alpha ───────────────────────────────────────────────────────
    # Flow 35% | IV regime 25% | Skew 20% | Gamma 20%
    df["alpha_options"] = (
        df["flow_score"]  * 0.35
        + df["iv_score"]  * 0.25
        + df["skew_score"] * 0.20
        + df["gamma_score"] * 0.20
    )

    # ── Cross-sectional rank 0-100 ────────────────────────────────────────────
    valid = df["alpha_options"].notna()
    df.loc[valid, "rank_options"] = (
        df.loc[valid, "alpha_options"].rank(pct=True) * 100
    ).round(1)
    df["rank_options"] = df["rank_options"].fillna(50.0)

    # ── Strategy + conviction ─────────────────────────────────────────────────
    strat_conv = df.apply(_recommend_strategy, axis=1)
    df["options_strategy"]  = strat_conv.apply(lambda x: x[0])
    df["conviction"]        = strat_conv.apply(lambda x: x[1])
    df["conviction_stars"]  = df["conviction"].apply(lambda n: "★" * n + "☆" * (5 - n))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# IC Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_ic(df: pd.DataFrame) -> None:
    """Quick Spearman IC check vs 21-day forward returns (informational only)."""
    try:
        from scipy import stats as sp
    except ImportError:
        return
    price_cache = ROOT / "sp500_price_cache.csv"
    if not price_cache.exists():
        return
    try:
        prices = pd.read_csv(price_cache, index_col=0, parse_dates=True)
        fwd    = prices.pct_change(21).iloc[-1]
        m      = df[["ticker", "rank_options", "flow_score"]].copy()
        m["fwd"] = m["ticker"].map(fwd)
        m = m.dropna()
        if len(m) < 10:
            return
        ic_r,  pv_r  = sp.spearmanr(m["rank_options"], m["fwd"])
        ic_f,  pv_f  = sp.spearmanr(m["flow_score"],   m["fwd"])
        print(f"  rank_options IC vs 21d fwd: {ic_r:+.4f}  p={pv_r:.3f}  "
              f"{'✅' if abs(ic_r) > 0.04 else '⚠️'}")
        print(f"  flow_score   IC vs 21d fwd: {ic_f:+.4f}  p={pv_f:.3f}  "
              f"{'✅' if abs(ic_f) > 0.04 else '⚠️'}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Morning Desk Briefing Report
# ─────────────────────────────────────────────────────────────────────────────

_STRAT_EXPLAIN = {
    "LONG_CALLS":          "Buy ATM/near-OTM calls — strong flow, cheap vol",
    "BULL_CALL_SPREAD":    "Buy ATM call + sell OTM call — defined-risk bull",
    "BULL_RISK_REVERSAL":  "Sell OTM put + buy OTM call — synthetic long via fear premium",
    "OTM_CALL_BACKSPREAD": "Sell 1× ATM call + buy 2× OTM — gamma squeeze, self-financing",
    "SHORT_PUT":           "Sell 30Δ put — harvest fear premium, bullish at lower entry",
    "LONG_STRADDLE":       "Buy ATM call + put — cheap vol pre-earnings/catalyst",
    "IRON_CONDOR":         "Sell put spread + call spread — IV crush, range-bound",
    "PROTECTIVE_PUT":      "Buy OTM put — hedge complacency, crowded-call warning",
    "MONITOR":             "No high-conviction setup — await cleaner signal",
}
_GEX_LABEL = {1: "↑ LONG (stable)", -1: "↓ SHORT (volatile)", 0: "—"}


def write_report(df: pd.DataFrame) -> None:
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    top15 = df.sort_values("rank_options", ascending=False).head(15)
    hc    = df[df["conviction"] >= 4].sort_values("rank_options", ascending=False)
    uoa   = df[df["uoa_flag"] == 1].sort_values("rank_options", ascending=False).head(10)
    sqz   = df[df["squeeze_risk"] == True].sort_values("rank_options", ascending=False).head(8)
    crowd = df[df["pcr_vol"] < 0.30].sort_values("rank_options").head(8)
    sdist = df["options_strategy"].value_counts()

    L = [
        "# Canyon v9 — Step 82: Institutional Options Desk Briefing",
        f"**Generated:** {now}  |  **Universe:** {len(df)} tickers",
        "",
        "## Signal Architecture",
        "| Layer | Weight | What it measures |",
        "|-------|--------|-----------------|",
        "| Flow Score   | **35%** | Net call premium ($) + PCR + Unusual Options Activity |",
        "| IV Regime    | **25%** | IV Rank (30d history) — cheap vs expensive vol |",
        "| Skew Score   | **20%** | OTM put/call IV spread — institutional directional bias |",
        "| Gamma Score  | **20%** | Dealer GEX (Black-Scholes) + max-pain + squeeze risk |",
        "",
        f"## High-Conviction Setups — ★★★★+ ({len(hc)} found)",
        "",
        "| # | Ticker | Strategy | Conv | Rank | IV Rank | PCR | Net Prem | UOA | GEX |",
        "|---|--------|----------|------|------|---------|-----|----------|-----|-----|",
    ]
    for i, (_, r) in enumerate(hc.head(12).iterrows(), 1):
        uoa_s  = "✅CALL" if r.get("uoa_flag")      else ("⚠️PUT" if r.get("uoa_bear_flag") else "—")
        gex_s  = _GEX_LABEL.get(int(r.get("gex_sign", 0)), "—")
        nprem  = r.get("net_call_premium", 0) / 1e6
        L.append(
            f"| {i} | **{r['ticker']}** | {r['options_strategy']} "
            f"| {r['conviction_stars']} | {r['rank_options']:.0f} "
            f"| {r.get('iv_rank', 50):.0f} | {r.get('pcr_vol', 1):.2f} "
            f"| {nprem:+.1f}M | {uoa_s} | {gex_s} |"
        )

    L += [
        "",
        "## Top 15 Options Alpha Rankings",
        "",
        "| # | Ticker | Rank | IV Rank | PCR | Net Prem $M | Skew | Strategy | ★ |",
        "|---|--------|------|---------|-----|-------------|------|----------|---|",
    ]
    for i, (_, r) in enumerate(top15.iterrows(), 1):
        nprem = r.get("net_call_premium", 0) / 1e6
        L.append(
            f"| {i} | **{r['ticker']}** | {r['rank_options']:.0f} "
            f"| {r.get('iv_rank', 50):.0f} | {r.get('pcr_vol', 1):.2f} "
            f"| {nprem:+.1f}M | {r.get('iv_skew', 0):+.3f} "
            f"| {r['options_strategy']} | {r.get('conviction_stars', '—')} |"
        )

    if not uoa.empty:
        L += [
            "",
            "## 🎯 Unusual Options Activity — Smart Money Positioning",
            "*(call volume > 3× open interest at a single strike — institutional directive bet)*",
            "",
            "| Ticker | Rank | UOA Detail | IV Rank | Strategy | ★ |",
            "|--------|------|-----------|---------|----------|---|",
        ]
        for _, r in uoa.iterrows():
            L.append(
                f"| **{r['ticker']}** | {r['rank_options']:.0f} "
                f"| {r.get('uoa_detail', '—')} "
                f"| {r.get('iv_rank', 50):.0f} "
                f"| {r['options_strategy']} | {r.get('conviction_stars', '—')} |"
            )

    if not sqz.empty:
        L += [
            "",
            "## ⚡ Gamma Squeeze Candidates",
            "*(negative dealer GEX near ATM + call UOA = dealers forced to buy as price rises)*",
            "",
            "| Ticker | Rank | GEX ($M) | Max Pain Dist | Strategy | ★ |",
            "|--------|------|----------|--------------|----------|---|",
        ]
        for _, r in sqz.iterrows():
            L.append(
                f"| **{r['ticker']}** | {r['rank_options']:.0f} "
                f"| {r.get('gex_net', 0):.2f}M "
                f"| {r.get('max_pain_dist', 0):+.1f}% "
                f"| {r['options_strategy']} | {r.get('conviction_stars', '—')} |"
            )

    if not crowd.empty:
        L += [
            "",
            "## ⚠️ Crowded Call Side — Soros Consensus Warning",
            "*(PCR < 0.30: peak optimism / everyone bought calls → consider PROTECTIVE_PUT)*",
            "",
            "| Ticker | PCR | Rank | IV Rank | Action |",
            "|--------|-----|------|---------|--------|",
        ]
        for _, r in crowd.iterrows():
            L.append(
                f"| {r['ticker']} | {r.get('pcr_vol', 0):.2f} "
                f"| {r['rank_options']:.0f} "
                f"| {r.get('iv_rank', 50):.0f} "
                f"| {r['options_strategy']} |"
            )

    L += [
        "",
        "## Strategy Distribution",
        "",
        "| Strategy | Count | Rationale |",
        "|----------|-------|-----------|",
    ]
    for strat, cnt in sdist.items():
        L.append(f"| {strat} | {cnt} | {_STRAT_EXPLAIN.get(strat, '')} |")

    L += [
        "",
        "## Score Distribution",
        f"- rank_options > 70 (strong buy signal):  {(df['rank_options'] > 70).sum()}",
        f"- rank_options 40–70 (neutral):            {((df['rank_options'] >= 40) & (df['rank_options'] <= 70)).sum()}",
        f"- rank_options < 40 (weak/avoid):          {(df['rank_options'] < 40).sum()}",
        f"- High conviction (★★★★+):                {(df['conviction'] >= 4).sum()}",
        f"- UOA call alerts:                         {int(df['uoa_flag'].sum())}",
        f"- Gamma squeeze candidates:                {int(df.get('squeeze_risk', pd.Series([False]*len(df))).sum())}",
        "",
        "---",
        "*Canyon v9 Step 82 — Institutional Options Trading Algorithm*",
        f"*GEX via Black-Scholes dealer approx | Skew: {int(SKEW_LOW*100)}-{int(SKEW_HIGH*100)}% OTM "
        f"| IV Rank: {IV_HIST_DAYS}d rolling | UOA threshold: {UOA_VO_RATIO:.0f}× vol/OI*",
    ]

    OUT_REPORT.write_text("\n".join(L))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(top_n: int = DEFAULT_TOP, workers: int = MAX_WORKERS,
         force_refresh: bool = False) -> None:
    t0 = datetime.now()
    print()
    print("=" * 65)
    print("Canyon v9 — Step 82: Institutional Options Trading Algorithm")
    print("=" * 65)

    # ── Universe ──────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading universe (top {top_n}) …")
    tickers = get_universe(top_n)
    print(f"  {len(tickers)} tickers")

    # ── Fetch options chains ──────────────────────────────────────────────────
    print(f"\n[2/5] Fetching options chains (workers={workers}) …")
    print("  4h cache — first run ≈2-3 min, cached runs near-instant")
    raw_df = load_options_data(tickers, max_workers=workers, force_refresh=force_refresh)
    if raw_df.empty:
        print("  ERROR: No options data fetched")
        return
    print(f"  {len(raw_df)} tickers with valid options data")

    # ── IV Rank ───────────────────────────────────────────────────────────────
    print("\n[3/5] Computing IV Rank from rolling history …")
    raw_df = update_iv_history(raw_df)

    # ── Signal computation ────────────────────────────────────────────────────
    print("\n[4/5] Computing signals (flow / IV / skew / gamma) …")
    result = compute_signals(raw_df)
    validate_ic(result)

    # Console summary
    strat_counts = result["options_strategy"].value_counts()
    print("\n  Strategy distribution:")
    for s, n in strat_counts.items():
        print(f"    {s:<28}  {n:>3}")

    hc = result[result["conviction"] >= 4].sort_values("rank_options", ascending=False)
    if not hc.empty:
        print(f"\n  High-conviction setups (★★★★+):  {len(hc)}")
        for _, r in hc.head(6).iterrows():
            tags = ""
            if r.get("uoa_flag"):      tags += " ⚡UOA"
            if r.get("squeeze_risk"):  tags += " 🔥SQZ"
            print(f"    {r['ticker']:6s}  {r['options_strategy']:<25}"
                  f"  rank={r['rank_options']:.0f}"
                  f"  IV%={r.get('iv_rank', 50):.0f}"
                  f"  PCR={r.get('pcr_vol', 1):.2f}"
                  f"  {r['conviction_stars']}{tags}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_cols = [
        "ticker", "rank_options", "options_strategy", "conviction", "conviction_stars",
        "iv_rank", "atm_iv", "iv_skew", "pcr_vol", "pcr_oi",
        "net_call_premium", "uoa_flag", "uoa_bear_flag", "uoa_detail",
        "gex_sign", "gex_net", "squeeze_risk", "max_pain", "max_pain_dist",
        "flow_score", "iv_score", "skew_score", "gamma_score", "alpha_options",
        "days_to_earnings", "term_structure", "expiry",
    ]
    out = result[[c for c in out_cols if c in result.columns]].copy()
    out.to_csv(OUT_SCORES, index=False)
    print(f"\n  Saved: {OUT_SCORES.name}  ({len(out)} rows)")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[5/5] Writing desk briefing report …")
    write_report(result)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 82 — Institutional Options Trading Algorithm"
    )
    parser.add_argument("--top",     type=int,  default=DEFAULT_TOP,
                        help=f"Universe size (default {DEFAULT_TOP})")
    parser.add_argument("--workers", type=int,  default=MAX_WORKERS,
                        help=f"Fetch workers (default {MAX_WORKERS})")
    parser.add_argument("--refresh", action="store_true",
                        help="Force-refresh options cache (ignore 4h TTL)")
    args = parser.parse_args()
    main(top_n=args.top, workers=args.workers, force_refresh=args.refresh)
