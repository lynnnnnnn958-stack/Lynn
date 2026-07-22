"""
Canyon v9 — Step 74: Layer 7 Options Chain Context Engine
Pulls options chain data via yfinance and computes PCR, Max Pain,
Gamma Wall, IV Rank, and Kill Zone for each ticker.

Usage:
    python3 canyon_final_v9_step74_options_chain.py
    python3 canyon_final_v9_step74_options_chain.py --ticker NVDA
    python3 canyon_final_v9_step74_options_chain.py --expiry 2026-06-20
"""

import argparse
import json
import math
import os
import time
import warnings
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_FILE = os.path.join(os.path.dirname(__file__), "options_chain_cache.json")
VIX_CACHE_FILE = os.path.join(os.path.dirname(__file__), "vix_iv_rank_cache.json")
CACHE_TTL_HOURS = 2
OUTPUT_DIR = os.path.dirname(__file__)
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "options_chain_summary.csv")
DETAIL_CSV = os.path.join(OUTPUT_DIR, "options_chain_detail.csv")
REPORT_MD = os.path.join(OUTPUT_DIR, "options_chain_report.md")
ML_SCORES_CSV = os.path.join(OUTPUT_DIR, "ml_signal_scores.csv")
SLEEP_BETWEEN = 0.3  # seconds

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AMD", "NFLX", "CRM",
    "PANW", "CRWD", "SNOW", "PLTR", "MSTR",
    "JPM", "GS", "BAC", "XOM", "UNH",
]
ALWAYS_INCLUDE = ["SPY", "QQQ"]
TOP_N = 20
TOP_DETAIL_N = 5

PCR_BEARISH = 1.2
PCR_BULLISH = 0.7
IVR_RICH = 75
IVR_CHEAP = 25
IVR_EXPENSIVE = 70
IVR_CHEAP_SIGNAL = 60
MIN_OI_KILL = 500
KILL_BAND = 0.02   # ±2%
PIN_BAND = 0.01    # ±1%


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_monthly_expiry(ref_date: date) -> date:
    """Return the 3rd Friday of the nearest month whose 3rd Friday >= ref_date."""
    year, month = ref_date.year, ref_date.month
    for _ in range(6):  # search at most 6 months ahead
        # find 3rd Friday of (year, month)
        first_day = date(year, month, 1)
        # weekday(): Monday=0, Friday=4
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        if third_friday >= ref_date:
            return third_friday
        month += 1
        if month > 12:
            month = 1
            year += 1
    return ref_date + timedelta(days=30)


def _nearest_expiry_gte(expiry_list: List[str], min_days: int = 7) -> Optional[str]:
    """Pick the earliest expiry that is at least min_days from today."""
    today = date.today()
    cutoff = today + timedelta(days=min_days)
    for exp in sorted(expiry_list):
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d >= cutoff:
            return exp
    return None


def _load_cache() -> Dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            raw = json.load(f)
        return raw
    except Exception:
        return {}


def _save_cache(cache: Dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, default=str)
    except Exception:
        pass


def _cache_key(ticker: str, expiry: str) -> str:
    return f"{ticker}::{expiry}"


def _cache_valid(entry: Dict) -> bool:
    ts = entry.get("timestamp")
    if ts is None:
        return False
    try:
        saved_at = datetime.fromisoformat(ts)
        return (datetime.now() - saved_at).total_seconds() < CACHE_TTL_HOURS * 3600
    except Exception:
        return False


def _load_ml_tickers() -> List[str]:
    """Load tickers from ml_signal_scores.csv, sorted by ensemble_score desc."""
    try:
        df = pd.read_csv(ML_SCORES_CSV)
        if "ticker" not in df.columns or "ensemble_score" not in df.columns:
            raise ValueError("Missing columns")
        latest = df.sort_values("rebalance_date", ascending=False) if "rebalance_date" in df.columns else df
        # Take the latest snapshot per ticker
        if "rebalance_date" in latest.columns:
            latest = latest.groupby("ticker", as_index=False).last()
        latest = latest.sort_values("ensemble_score", ascending=False)
        return latest["ticker"].tolist()
    except Exception:
        return []


def _atm_delta(moneyness: float) -> float:
    """Approximate delta that decays from 0.5 as moneyness moves away from 1.0."""
    # moneyness = strike / price
    return 0.5 * math.exp(-4.0 * (moneyness - 1.0) ** 2)


# ---------------------------------------------------------------------------
# VIX Rank cache  (downloaded once per session, reused for all tickers)
# ---------------------------------------------------------------------------

_VIX_RANK_CACHE: Optional[Dict] = None   # module-level singleton

def _get_vix_rank() -> Dict:
    """
    Compute VIX 52-week percentile rank using daily ^VIX closes.
    Caches to disk for CACHE_TTL_HOURS; returns from memory within session.

    Returns dict with:
      vix_current  : today's VIX close
      vix_rank     : 0–100 percentile over past 252 trading days
      vix_1y_low   : 52-week VIX low
      vix_1y_high  : 52-week VIX high
    """
    global _VIX_RANK_CACHE

    # In-memory hit
    if _VIX_RANK_CACHE is not None:
        return _VIX_RANK_CACHE

    # Disk cache hit
    if os.path.exists(VIX_CACHE_FILE):
        try:
            with open(VIX_CACHE_FILE) as f:
                cached = json.load(f)
            ts = datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))
            if (datetime.now() - ts).total_seconds() < CACHE_TTL_HOURS * 3600:
                _VIX_RANK_CACHE = cached
                return cached
        except Exception:
            pass

    # Fresh download
    try:
        vix = yf.download("^VIX", period="1y", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        closes = vix["Close"].dropna()
        if len(closes) < 30:
            raise ValueError("Insufficient VIX history")
        current = float(closes.iloc[-1])
        lo      = float(closes.min())
        hi      = float(closes.max())
        rank    = float((closes < current).sum() / len(closes) * 100)
        result = {
            "vix_current": round(current, 2),
            "vix_rank":    round(rank, 1),
            "vix_1y_low":  round(lo, 2),
            "vix_1y_high": round(hi, 2),
            "timestamp":   datetime.now().isoformat(),
        }
    except Exception as exc:
        result = {
            "vix_current": float("nan"),
            "vix_rank":    float("nan"),
            "vix_1y_low":  float("nan"),
            "vix_1y_high": float("nan"),
            "timestamp":   datetime.now().isoformat(),
            "error":       str(exc),
        }

    _VIX_RANK_CACHE = result
    try:
        with open(VIX_CACHE_FILE, "w") as f:
            json.dump(result, f)
    except Exception:
        pass
    return result


def _compute_hv_30d(ticker: str) -> float:
    """
    30-day annualised historical (realised) volatility from daily closes.
    Returns float or nan on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 22:
            return float("nan")
        rets = hist["Close"].pct_change().dropna()
        hv = float(rets.tail(21).std() * math.sqrt(252))
        return hv
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class OptionsChainEngine:
    def __init__(self, force_expiry: Optional[str] = None):
        self.force_expiry = force_expiry
        self.cache = _load_cache()

    # ------------------------------------------------------------------
    def fetch_chain(
        self, ticker: str, expiry: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Return (calls_df, puts_df) for the given expiry. Uses cache if fresh."""
        key = _cache_key(ticker, expiry)
        entry = self.cache.get(key, {})
        if _cache_valid(entry):
            try:
                calls = pd.DataFrame(entry["calls"])
                puts = pd.DataFrame(entry["puts"])
                return calls, puts
            except Exception:
                pass

        try:
            tk = yf.Ticker(ticker)
            chain = tk.option_chain(expiry)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            # Normalise column names (yfinance can vary)
            for df in (calls, puts):
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            self.cache[key] = {
                "timestamp": datetime.now().isoformat(),
                "calls": calls.to_dict(orient="records"),
                "puts": puts.to_dict(orient="records"),
            }
            _save_cache(self.cache)
            return calls, puts
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    def compute_pcr(
        self, calls: pd.DataFrame, puts: pd.DataFrame
    ) -> Tuple[float, float]:
        """Return (volume_pcr, oi_pcr). NaN when denominator is zero."""
        def safe_div(a, b):
            return float(a) / float(b) if b and not math.isnan(float(b)) and float(b) > 0 else float("nan")

        call_vol = calls.get("volume", pd.Series(dtype=float)).fillna(0).sum()
        put_vol = puts.get("volume", pd.Series(dtype=float)).fillna(0).sum()
        call_oi = calls.get("openinterest", pd.Series(dtype=float)).fillna(0).sum()
        put_oi = puts.get("openinterest", pd.Series(dtype=float)).fillna(0).sum()

        vol_pcr = safe_div(put_vol, call_vol)
        oi_pcr = safe_div(put_oi, call_oi)
        return vol_pcr, oi_pcr

    # ------------------------------------------------------------------
    def compute_max_pain(
        self, calls: pd.DataFrame, puts: pd.DataFrame, price: float
    ) -> Optional[float]:
        """Return the max-pain strike that minimises total option buyer loss."""
        try:
            c = calls[["strike", "openinterest"]].copy().fillna(0)
            p = puts[["strike", "openinterest"]].copy().fillna(0)
            strikes = sorted(set(c["strike"].tolist() + p["strike"].tolist()))
            if len(strikes) < 3:
                return None

            min_pain = float("inf")
            max_pain_strike = strikes[0]

            for k in strikes:
                # Call pain: call buyers lose when K < strike
                call_pain = float(
                    c.apply(lambda r: max(0.0, r["strike"] - k) * r["openinterest"], axis=1).sum()
                )
                # Put pain: put buyers lose when K > strike
                put_pain = float(
                    p.apply(lambda r: max(0.0, k - r["strike"]) * r["openinterest"], axis=1).sum()
                )
                total = call_pain + put_pain
                if total < min_pain:
                    min_pain = total
                    max_pain_strike = k

            return float(max_pain_strike)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def compute_gamma_wall(
        self, calls: pd.DataFrame, puts: pd.DataFrame, price: float
    ) -> Tuple[Optional[float], str]:
        """Return (wall_strike, direction). direction is 'CALL' or 'PUT'."""
        try:
            rows = []
            for _, r in calls.iterrows():
                oi = float(r.get("openinterest", 0) or 0)
                strike = float(r.get("strike", price))
                moneyness = strike / price if price > 0 else 1.0
                delta = _atm_delta(moneyness)
                gex = oi * delta * 100
                rows.append({"strike": strike, "gex": gex, "side": "call"})

            for _, r in puts.iterrows():
                oi = float(r.get("openinterest", 0) or 0)
                strike = float(r.get("strike", price))
                moneyness = strike / price if price > 0 else 1.0
                delta = -_atm_delta(moneyness)  # negative delta for puts
                gex = oi * delta * 100
                rows.append({"strike": strike, "gex": gex, "side": "put"})

            if not rows:
                return None, "UNKNOWN"

            gex_df = pd.DataFrame(rows)
            by_strike = gex_df.groupby("strike")["gex"].sum().reset_index()
            by_strike["abs_gex"] = by_strike["gex"].abs()
            best = by_strike.sort_values("abs_gex", ascending=False).iloc[0]
            direction = "CALL" if float(best["gex"]) > 0 else "PUT"
            return float(best["strike"]), direction
        except Exception:
            return None, "UNKNOWN"

    # ------------------------------------------------------------------
    def compute_iv_context(self, ticker: str, atm_iv: float) -> Tuple[float, float, float]:
        """
        Compute three correct IV-related metrics.

        Returns (vix_rank, iv_hv_ratio, hv_30d) where:

        vix_rank (0–100):
            VIX 52-week percentile rank.
            This IS the canonical "IVR" definition — where does current market
            fear sit relative to the past 52 weeks?
            0 = historically cheap IV, 100 = historically expensive IV.
            Used as iv_rank in the summary (backward-compatible field name).

            Previously this field computed:
                (atm_iv - rolling_hv_low) / (rolling_hv_high - rolling_hv_low)
            That was WRONG: it compared implied vol (forward-looking) against
            the range of realised historical vol (backward-looking).  The two
            series are on different scales and the rank was meaningless.

        iv_hv_ratio:
            atm_iv / hv_30d — "options premium" metric.
            > 1.3 → options pricing in MORE vol than recently realised (rich)
            < 0.8 → options pricing in LESS vol (cheap, or low fear)
            This is stock-specific and actionable.

        hv_30d:
            30-day annualised realised vol — context only.
        """
        vix_info = _get_vix_rank()
        vix_rank = vix_info.get("vix_rank", float("nan"))

        hv_30d = _compute_hv_30d(ticker)

        if (not math.isnan(atm_iv) and atm_iv > 0
                and not math.isnan(hv_30d) and hv_30d > 0):
            iv_hv_ratio = round(atm_iv / hv_30d, 2)
        else:
            iv_hv_ratio = float("nan")

        return vix_rank, iv_hv_ratio, hv_30d

    # Legacy alias — returns only vix_rank for callers that expect a float
    def compute_iv_rank(self, ticker: str, atm_iv: float) -> float:
        vix_rank, _, _ = self.compute_iv_context(ticker, atm_iv)
        return vix_rank

    # ------------------------------------------------------------------
    def compute_kill_zone(
        self, calls: pd.DataFrame, puts: pd.DataFrame, price: float
    ) -> Tuple[List[float], bool]:
        """
        Return (kill_strikes, pin_risk).
        kill_strikes: strikes within ±2% of price with OI > 500.
        pin_risk: True if price is within 1% of the highest-OI kill strike.
        """
        try:
            combined = pd.concat([
                calls[["strike", "openinterest"]].copy(),
                puts[["strike", "openinterest"]].copy(),
            ]).fillna(0)
            combined["openinterest"] = combined["openinterest"].astype(float)
            combined = combined.groupby("strike")["openinterest"].sum().reset_index()

            band_lo = price * (1 - KILL_BAND)
            band_hi = price * (1 + KILL_BAND)
            zone = combined[
                (combined["strike"] >= band_lo)
                & (combined["strike"] <= band_hi)
                & (combined["openinterest"] > MIN_OI_KILL)
            ].sort_values("openinterest", ascending=False)

            kill_strikes = zone["strike"].tolist()
            if not kill_strikes:
                return [], False

            top_strike = float(zone.iloc[0]["strike"])
            pin_risk = abs(top_strike - price) / price <= PIN_BAND
            return [float(s) for s in kill_strikes], pin_risk
        except Exception:
            return [], False

    # ------------------------------------------------------------------
    def build_signal(self, m: Dict) -> str:
        """
        Derive the options signal from the metrics dict.

        iv_rank = vix_rank (0–100 VIX percentile).
        iv_hv_ratio = atm_iv / hv_30d (options premium relative to realised vol).
        """
        pcr_oi      = m.get("pcr_oi",      float("nan"))
        ivr         = m.get("iv_rank",     float("nan"))   # VIX rank
        iv_hv_ratio = m.get("iv_hv_ratio", float("nan"))
        pin         = m.get("pin_risk", False)

        def ok(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v))

        signals = []

        # PCR + VIX rank combos
        if ok(pcr_oi) and ok(ivr):
            if pcr_oi > PCR_BEARISH and ivr < 40:
                signals.append("BULLISH_OPTIONS")   # heavy puts but calm VIX → contrarian bullish
            elif pcr_oi < PCR_BULLISH and ivr > IVR_EXPENSIVE:
                signals.append("BEARISH_OPTIONS")   # light puts but elevated VIX → caution

        # VIX rank alone
        if ok(ivr):
            if ivr > IVR_RICH:
                signals.append("IV_RICH")     # sell-premium environment
            elif ivr < IVR_CHEAP:
                signals.append("IV_CHEAP")    # buy-premium environment

        # IV/HV ratio signals (stock-specific)
        if ok(iv_hv_ratio):
            if iv_hv_ratio > 1.5:
                signals.append("OPTIONS_OVERPRICED")   # market fears >50% more than recent moves
            elif iv_hv_ratio < 0.7:
                signals.append("OPTIONS_CHEAP")        # market pricing in less than recent moves

        if pin:
            signals.append("PINNED")

        return "|".join(signals) if signals else "NEUTRAL"

    # ------------------------------------------------------------------
    def _get_price(self, ticker: str) -> float:
        try:
            tk = yf.Ticker(ticker)
            info = tk.fast_info
            price = getattr(info, "last_price", None)
            if price is None or math.isnan(float(price)):
                hist = tk.history(period="2d", interval="1d", auto_adjust=True)
                price = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")
            return float(price)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    def _get_atm_iv(
        self, calls: pd.DataFrame, puts: pd.DataFrame, price: float
    ) -> float:
        """Average IV of the ATM call and ATM put."""
        try:
            def atm_iv(df: pd.DataFrame) -> float:
                col = "impliedvolatility" if "impliedvolatility" in df.columns else None
                if col is None:
                    return float("nan")
                df2 = df[["strike", col]].copy().dropna()
                if df2.empty:
                    return float("nan")
                idx = (df2["strike"] - price).abs().idxmin()
                return float(df2.loc[idx, col])

            c_iv = atm_iv(calls)
            p_iv = atm_iv(puts)
            vals = [v for v in (c_iv, p_iv) if not math.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    def _pick_expiry(self, ticker: str) -> Optional[str]:
        if self.force_expiry:
            return self.force_expiry
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                return None
            nearest = _nearest_expiry_gte(list(exps), min_days=7)
            return nearest
        except Exception:
            return None

    # ------------------------------------------------------------------
    def process_ticker(self, ticker: str) -> Dict:
        """Full pipeline for one ticker. Returns a metrics dict."""
        base = {
            "ticker": ticker,
            "price": float("nan"),
            "expiry": None,
            "pcr_volume": float("nan"),
            "pcr_oi": float("nan"),
            "max_pain": float("nan"),
            "gamma_wall_strike": float("nan"),
            "gex_direction": "UNKNOWN",
            "iv_rank": float("nan"),    # = vix_rank (see compute_iv_context)
            "atm_iv": float("nan"),
            "iv_hv_ratio": float("nan"),  # atm_iv / hv_30d
            "hv_30d": float("nan"),       # 30-day realised vol
            "vix_rank": float("nan"),     # VIX 52-week percentile (same as iv_rank)
            "kill_zone_strikes": [],
            "pin_risk": False,
            "signal": "NO_OPTIONS",
        }

        try:
            price = self._get_price(ticker)
            base["price"] = price

            expiry = self._pick_expiry(ticker)
            if expiry is None:
                return base

            base["expiry"] = expiry
            calls, puts = self.fetch_chain(ticker, expiry)
            if calls is None or puts is None or calls.empty or puts.empty:
                return base

            # PCR
            vol_pcr, oi_pcr = self.compute_pcr(calls, puts)
            base["pcr_volume"] = round(vol_pcr, 4) if not math.isnan(vol_pcr) else float("nan")
            base["pcr_oi"] = round(oi_pcr, 4) if not math.isnan(oi_pcr) else float("nan")

            # Max Pain
            mp = self.compute_max_pain(calls, puts, price)
            base["max_pain"] = round(mp, 2) if mp is not None else float("nan")

            # Gamma Wall
            gw, gdir = self.compute_gamma_wall(calls, puts, price)
            base["gamma_wall_strike"] = round(gw, 2) if gw is not None else float("nan")
            base["gex_direction"] = gdir

            # ATM IV
            atm_iv = self._get_atm_iv(calls, puts, price)
            base["atm_iv"] = round(atm_iv, 4) if not math.isnan(atm_iv) else float("nan")

            # IV context: VIX rank (= iv_rank), IV/HV ratio, 30d HV
            vix_rank, iv_hv_ratio, hv_30d = self.compute_iv_context(ticker, atm_iv)
            base["iv_rank"]    = round(vix_rank,    1) if not math.isnan(vix_rank)    else float("nan")
            base["vix_rank"]   = base["iv_rank"]
            base["iv_hv_ratio"]= round(iv_hv_ratio, 2) if not math.isnan(iv_hv_ratio) else float("nan")
            base["hv_30d"]     = round(hv_30d,      4) if not math.isnan(hv_30d)      else float("nan")

            # Kill Zone
            kz, pin = self.compute_kill_zone(calls, puts, price)
            base["kill_zone_strikes"] = [round(s, 2) for s in kz[:10]]
            base["pin_risk"] = bool(pin)

            # Signal
            base["signal"] = self.build_signal(base)

        except Exception as exc:
            base["signal"] = f"ERROR: {exc}"

        return base

    # ------------------------------------------------------------------
    def run(self, tickers: List[str]) -> List[Dict]:
        results = []
        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i:2d}/{total}] {ticker:<8s}", end=" ", flush=True)
            try:
                m = self.process_ticker(ticker)
                results.append(m)
                sig = m.get("signal", "?")
                price = m.get("price", float("nan"))
                vix_rank  = m.get("vix_rank",    float("nan"))
                iv_hv     = m.get("iv_hv_ratio", float("nan"))
                price_s   = f"${price:.2f}" if not math.isnan(price) else "N/A"
                vix_s     = f"VIXRk={vix_rank:.0f}" if not math.isnan(vix_rank) else "VIXRk=N/A"
                ivhv_s    = f"IV/HV={iv_hv:.2f}" if not math.isnan(iv_hv) else "IV/HV=N/A"
                print(f"price={price_s:>8s}  {vix_s:<12s}  {ivhv_s:<12s}  sig={sig}")
            except Exception as exc:
                print(f"FAILED: {exc}")
                results.append({"ticker": ticker, "signal": f"ERROR: {exc}"})
            time.sleep(SLEEP_BETWEEN)
        return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _build_summary_df(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for m in results:
        rows.append({
            "ticker":             m.get("ticker"),
            "price":              m.get("price"),
            "expiry":             m.get("expiry"),
            "pcr_volume":         m.get("pcr_volume"),
            "pcr_oi":             m.get("pcr_oi"),
            "max_pain":           m.get("max_pain"),
            "gamma_wall_strike":  m.get("gamma_wall_strike"),
            "gex_direction":      m.get("gex_direction"),
            "vix_rank":           m.get("vix_rank"),        # VIX 52-week pctile (0-100)
            "iv_rank":            m.get("iv_rank"),         # alias of vix_rank
            "atm_iv":             m.get("atm_iv"),          # true ATM implied vol
            "iv_hv_ratio":        m.get("iv_hv_ratio"),     # atm_iv / hv_30d
            "hv_30d":             m.get("hv_30d"),          # 30-day realised vol
            "kill_zone_strikes":  "|".join(str(s) for s in m.get("kill_zone_strikes", [])),
            "pin_risk":           m.get("pin_risk"),
            "signal":             m.get("signal"),
        })
    return pd.DataFrame(rows)


def _fetch_detail_chains(
    engine: OptionsChainEngine, tickers: List[str]
) -> pd.DataFrame:
    """Fetch raw chain data for the top-N tickers for detail CSV."""
    frames = []
    for ticker in tickers[:TOP_DETAIL_N]:
        expiry = engine._pick_expiry(ticker)
        if expiry is None:
            continue
        calls, puts = engine.fetch_chain(ticker, expiry)
        if calls is not None and not calls.empty:
            calls = calls.copy()
            calls["ticker"] = ticker
            calls["expiry"] = expiry
            calls["option_type"] = "call"
            frames.append(calls)
        if puts is not None and not puts.empty:
            puts = puts.copy()
            puts["ticker"] = ticker
            puts["expiry"] = expiry
            puts["option_type"] = "put"
            frames.append(puts)
        time.sleep(SLEEP_BETWEEN)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _write_report(df: pd.DataFrame, run_dt: str) -> None:
    lines = [
        f"# Canyon v9 — Options Chain Report",
        f"",
        f"**Generated:** {run_dt}",
        f"",
        f"## Summary",
        f"",
        f"| Ticker | Price | Expiry | PCR(OI) | VIX Rank | ATM IV | IV/HV | HV 30d | Max Pain | Gamma Wall | Signal |",
        f"|--------|-------|--------|---------|----------|--------|-------|--------|----------|------------|--------|",
    ]
    for _, r in df.iterrows():
        def fmt(v, fmt_str=".2f"):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:{fmt_str}}"

        lines.append(
            f"| {r.ticker} "
            f"| {fmt(r.price)} "
            f"| {r.expiry or '—'} "
            f"| {fmt(r.pcr_oi)} "
            f"| {fmt(r.get('vix_rank', r.iv_rank), '.1f')} "
            f"| {fmt(r.atm_iv, '.3f')} "
            f"| {fmt(r.get('iv_hv_ratio', float('nan')), '.2f')} "
            f"| {fmt(r.get('hv_30d', float('nan')), '.3f')} "
            f"| {fmt(r.max_pain)} "
            f"| {fmt(r.gamma_wall_strike)} "
            f"| {r.signal or '—'} |"
        )

    # Counts by signal
    lines += [
        f"",
        f"## Signal Breakdown",
        f"",
    ]
    sig_counts: Dict[str, int] = {}
    for sig in df["signal"].dropna():
        for s in str(sig).split("|"):
            sig_counts[s] = sig_counts.get(s, 0) + 1
    for sig, cnt in sorted(sig_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{sig}**: {cnt}")

    lines += [
        f"",
        f"## Metrics Explained",
        f"",
        f"**VIX Rank (0–100):** Where does the VIX sit in its 52-week range?",
        f"  - 0 = lowest VIX of the year (very cheap options, complacent market)",
        f"  - 100 = highest VIX of the year (very expensive options, fearful market)",
        f"  - This is the correct definition of IVR at the market level.",
        f"  - _Previous version incorrectly compared ATM implied vol against the range of",
        f"    historical realised vol — those are different quantities on different scales._",
        f"",
        f"**ATM IV:** True implied volatility of the at-the-money option (from yfinance).",
        f"  - This IS real implied vol — forward-looking market expectation of moves.",
        f"",
        f"**IV/HV Ratio (atm_iv ÷ hv_30d):** Are options expensive vs recent realised moves?",
        f"  - > 1.3 → OPTIONS_OVERPRICED: market prices in >30% more vol than recent history",
        f"  - 0.8–1.3 → Fair value",
        f"  - < 0.7 → OPTIONS_CHEAP: options priced below recent realised vol",
        f"",
        f"**HV 30d:** 30-day annualised realised volatility (context for IV/HV ratio).",
        f"",
        f"## Signal Rules",
        f"- PCR > {PCR_BEARISH} with VIX Rank < 40 → BULLISH_OPTIONS (heavy puts but calm market)",
        f"- PCR < {PCR_BULLISH} with VIX Rank > {IVR_EXPENSIVE} → BEARISH_OPTIONS",
        f"- VIX Rank > {IVR_RICH} → IV_RICH (sell-premium environment)",
        f"- VIX Rank < {IVR_CHEAP} → IV_CHEAP (buy-premium environment)",
        f"- IV/HV > 1.5 → OPTIONS_OVERPRICED (stock-specific)",
        f"- IV/HV < 0.7 → OPTIONS_CHEAP (stock-specific)",
        f"- PINNED → price within 1% of top kill-zone strike",
    ]

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_console_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(f"{'TICKER':<8} {'PRICE':>8} {'EXPIRY':<12} {'PCR(OI)':>8} "
          f"{'VIXRk':>6} {'ATM_IV':>8} {'IV/HV':>6} {'MAX_PAIN':>10} {'SIGNAL':<25}")
    print("-" * 100)

    def fv(v, fmt=".2f"):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        try:
            return f"{v:{fmt}}"
        except Exception:
            return str(v)

    for _, r in df.iterrows():
        vix_rank = r.get("vix_rank", r.get("iv_rank", float("nan"))) if hasattr(r, "get") else getattr(r, "vix_rank", float("nan"))
        iv_hv    = r.get("iv_hv_ratio", float("nan")) if hasattr(r, "get") else getattr(r, "iv_hv_ratio", float("nan"))
        print(
            f"{str(r.ticker):<8} "
            f"{fv(r.price):>8} "
            f"{str(r.expiry or 'N/A'):<12} "
            f"{fv(r.pcr_oi):>8} "
            f"{fv(vix_rank, '.1f'):>6} "
            f"{fv(r.atm_iv, '.4f'):>8} "
            f"{fv(iv_hv, '.2f'):>6} "
            f"{fv(r.max_pain):>10} "
            f"{str(r.signal):<25}"
        )
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Ticker selection
# ---------------------------------------------------------------------------

def _select_tickers(single: Optional[str] = None) -> List[str]:
    if single:
        return [single.upper()]

    ml_tickers = _load_ml_tickers()
    if ml_tickers:
        top_n = ml_tickers[:TOP_N]
    else:
        top_n = FALLBACK_TICKERS[:TOP_N]

    # Always include SPY and QQQ, deduplicate, preserve order
    ordered = list(dict.fromkeys(ALWAYS_INCLUDE + top_n))
    return ordered


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Canyon v9 Step 74: Options Chain Engine")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker to analyse")
    parser.add_argument("--expiry", type=str, default=None, help="Force specific expiry YYYY-MM-DD")
    args = parser.parse_args()

    run_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nCanyon v9 | Step 74 | Options Chain Engine")
    print(f"Run: {run_dt}")
    print("-" * 50)

    tickers = _select_tickers(single=args.ticker)
    print(f"Tickers to process ({len(tickers)}): {', '.join(tickers)}\n")

    engine = OptionsChainEngine(force_expiry=args.expiry)

    print("Fetching options data...")
    results = engine.run(tickers)

    # Summary CSV
    summary_df = _build_summary_df(results)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {SUMMARY_CSV}")

    # Detail CSV (top 5 tickers only)
    # Use tickers that had valid options, in original order
    valid_tickers = [
        r["ticker"] for r in results
        if r.get("expiry") is not None and r.get("signal") != "NO_OPTIONS"
    ]
    detail_df = _fetch_detail_chains(engine, valid_tickers)
    if not detail_df.empty:
        detail_df.to_csv(DETAIL_CSV, index=False)
        print(f"Saved: {DETAIL_CSV}")

    # Report
    _write_report(summary_df, run_dt)
    print(f"Saved: {REPORT_MD}")

    # Console table
    _print_console_table(summary_df)

    # Quick counts
    sig_counts: Dict[str, int] = {}
    for sig in summary_df["signal"].dropna():
        for s in str(sig).split("|"):
            sig_counts[s] = sig_counts.get(s, 0) + 1
    print("Signal counts:")
    for sig, cnt in sorted(sig_counts.items(), key=lambda x: -x[1]):
        print(f"  {sig:<25} {cnt}")
    print()


if __name__ == "__main__":
    main()
