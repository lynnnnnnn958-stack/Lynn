"""
Canyon Quant — Paper Trading Signal Generator
==============================================
Runs v20 signal on today's data and appends to paper_trading_signals.csv.
Run this at end of each month to generate a dated, auditable prediction record.

Prediction made TODAY is compared to realized returns NEXT MONTH.
This creates an independently verifiable track record for institutional review.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).parent
TODAY   = datetime.today().strftime("%Y-%m-%d")
HOLD_M  = 21
WARMUP  = 252
LOG_FILE = ROOT / "paper_trading_signals.csv"
HIST_FILE = ROOT / "paper_trading_history.csv"

print(f"Canyon Quant v20 — Paper Trading Signal Generator")
print(f"Signal date: {TODAY}\n")

# ── Load data ─────────────────────────────────────────────────────────────────
p25 = pd.read_csv(ROOT / "sp500_price_25yr.csv", index_col=0, parse_dates=True).sort_index()
spy_all = p25["SPY"].copy()
stocks  = p25.drop(columns=["SPY"], errors="ignore")
tickers = list(stocks.columns)
col_idx = {c: i for i, c in enumerate(tickers)}
N = len(tickers)
all_dates = stocks.index

sector_map = {}
sp = ROOT / "sp500_sectors.csv"
if sp.exists():
    sector_map = pd.read_csv(sp, index_col=0).squeeze().to_dict()
all_sectors = sorted(set(sector_map.values()))

# ── Historical constituents ────────────────────────────────────────────────────
sp500_intervals = {}
hist_path = ROOT / "sp500_hist_constituents.csv"
if hist_path.exists():
    hdf = pd.read_csv(hist_path)
    hdf["start_date"] = pd.to_datetime(hdf["start_date"], errors="coerce")
    hdf["end_date"]   = pd.to_datetime(hdf["end_date"], errors="coerce")
    for _, row in hdf.iterrows():
        tk = str(row["ticker"]).strip()
        s  = row["start_date"]
        e  = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp("2030-01-01")
        if pd.notna(s):
            sp500_intervals.setdefault(tk, []).append((s, e))

def in_sp500_at(ticker, date):
    intervals = sp500_intervals.get(ticker)
    if not intervals: return True
    for (s, e) in intervals:
        if s <= date <= e: return True
    return False

# ── SUE cache ─────────────────────────────────────────────────────────────────
sue_df = pd.read_csv(ROOT / "sue_data.csv")
sue_df["date"] = pd.to_datetime(sue_df["date"])
HAS_SUE = len(sue_df) > 1000

def sue_score(t_date):
    if not HAS_SUE: return np.full(N, 50.0)
    window  = t_date - pd.DateOffset(months=15)
    recent  = sue_df[(sue_df.date >= window) & (sue_df.date < t_date)]
    if len(recent) < 20: return np.full(N, 50.0)
    agg     = recent.groupby("ticker")["sue"].mean()
    arr     = np.full(N, np.nan)
    for tk, val in agg.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = val
    valid   = np.isfinite(arr)
    ranked  = np.full(N, 50.0)
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
    return ranked

# ── Analyst revision cache (if available) ────────────────────────────────────
ar_df = None
ar_path = ROOT / "analyst_revisions.csv"
if ar_path.exists():
    ar_df = pd.read_csv(ar_path)
    ar_df["date"] = pd.to_datetime(ar_df["date"])
    print(f"  Analyst revisions loaded: {len(ar_df)} rows, {ar_df.ticker.nunique()} tickers")

def analyst_rev_score(t_date):
    if ar_df is None: return np.full(N, 50.0)
    window = t_date - pd.DateOffset(days=60)
    recent = ar_df[(ar_df.date >= window) & (ar_df.date < t_date)]
    if len(recent) < 10: return np.full(N, 50.0)
    agg = recent.groupby("ticker")["rev_score"].sum()
    arr = np.full(N, np.nan)
    for tk, val in agg.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = float(val)
    valid  = np.isfinite(arr)
    ranked = np.full(N, 50.0)
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
    return ranked

# ── Core signal ───────────────────────────────────────────────────────────────
def sector_rank(arr):
    result = np.full(len(arr), np.nan)
    for sec in all_sectors:
        sec_idx = [col_idx[tk] for tk in tickers
                   if sector_map.get(tk) == sec and tk in col_idx]
        if len(sec_idx) < 2: continue
        vals = arr[sec_idx]; valid = np.isfinite(vals)
        if valid.sum() < 2: continue
        rnk = np.full(len(vals), np.nan)
        rnk[valid] = (vals[valid].argsort().argsort() + 1) / valid.sum() * 100
        for ii, ci in enumerate(sec_idx): result[ci] = rnk[ii]
    miss = ~np.isfinite(result)
    if miss.any():
        v = np.isfinite(arr)
        if v.sum() >= 2:
            g = np.full(len(arr), np.nan)
            g[v] = (arr[v].argsort().argsort() + 1) / v.sum() * 100
            result[miss] = g[miss]
    return result

MOM_LK = 9   # locked from train 2000-2017
SPY_WIN = 200

def generate_signal():
    price_arr = stocks.values.astype(float)
    t = len(price_arr) - 1
    t_date = all_dates[-1]

    spy_arr = spy_all.values.astype(float)
    spy_ma200 = spy_arr[t - SPY_WIN + 1 : t + 1].mean() if t >= SPY_WIN else spy_arr[:t+1].mean()
    bull = bool(spy_arr[t] > spy_ma200)

    p    = price_arr[:t + 1]
    now  = p[-1]
    p3m  = p[-64] if t >= 63 else p[0]
    n_lk = MOM_LK * 21 + 21
    p_lk = p[-n_lk] if t >= n_lk - 1 else p[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        mom_lt = np.where(p_lk > 0, now / p_lk - 1, np.nan).clip(-1, 1)
        mom3m  = np.where(p3m  > 0, now / p3m  - 1, np.nan).clip(-1, 1)

    log_p  = np.log(np.where(p[-22:] > 0, p[-22:], np.nan))
    vol20  = np.nanstd(np.diff(log_p, axis=0), axis=0)
    invvol = 1.0 / (vol20 + 0.005)
    sma200 = np.nanmean(p[-200:], axis=0) if t >= 199 else np.nanmean(p, axis=0)
    above  = (now > sma200).astype(float)

    ivr  = sector_rank(invvol)
    civr = (100.0 - ivr) if bull else ivr

    score = (sector_rank(mom_lt) * 0.40 +
             sector_rank(mom3m)  * 0.20 +
             sector_rank(above)  * 0.15 +
             civr                * 0.10)

    # SUE blend
    sue_arr = sue_score(t_date)
    score = score * 0.85 + sue_arr * 0.15

    # Analyst revision blend (if available)
    ar_arr = analyst_rev_score(t_date)
    if ar_df is not None:
        score = score * 0.90 + ar_arr * 0.10

    # Point-in-time eligibility
    for tk in tickers:
        ci = col_idx.get(tk)
        if ci is not None and not in_sp500_at(tk, t_date):
            score[ci] = np.nan

    # Quality filter
    delta   = np.diff(log_p, axis=0)
    gains   = np.nanmean(np.clip(delta[-14:], 0, None), axis=0)
    loss    = np.nanmean(np.clip(-delta[-14:], 0, None), axis=0)
    rsi14   = 100 - 100 / (1 + gains / (loss + 1e-9))
    dist    = np.where(sma200 > 0, now / sma200 - 1, 0)
    vol75   = np.nanpercentile(vol20, 75)
    quality = (rsi14 < 75) & (dist < 0.5) & (vol20 < vol75)

    valid = quality & np.isfinite(score)
    idx   = np.where(valid)[0]
    if len(idx) == 0: idx = np.where(np.isfinite(score))[0]

    top_n    = 25
    sorted_i = idx[np.argsort(score[idx])[::-1]]
    sel, sec_cnt = [], {}
    for ci in sorted_i:
        if len(sel) >= top_n: break
        sec = sector_map.get(tickers[ci], "Unknown")
        sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
        if sec_cnt[sec] <= max(1, int(top_n * 0.35)):
            sel.append(ci)
    if len(sel) < 5: sel = sorted_i[:top_n].tolist()

    sv  = vol20[sel].clip(min=0.001)
    raw = 1.0 / sv; raw /= raw.sum()
    raw = np.clip(raw, 0, 0.15); raw /= raw.sum()

    # TQQQ allocation
    letf_path = ROOT / "letf_prices.csv"
    tqqq_w = 0.0
    if letf_path.exists() and bull:
        letf_df = pd.read_csv(letf_path, index_col=0, parse_dates=True).sort_index()
        tqqq_p  = letf_df["TQQQ"].reindex(all_dates).ffill().values.astype(float)
        log_tqqq = np.log(np.where(tqqq_p > 0, tqqq_p, np.nan))
        tr_daily = np.diff(log_tqqq)[max(0, t - 20):t]
        tr_daily = tr_daily[np.isfinite(tr_daily)]
        tvol = float(np.std(tr_daily) * np.sqrt(252)) if len(tr_daily) >= 5 else 0.60
        tqqq_w = float(np.clip(0.18 / (tvol + 0.001), 0.0, 0.40))

        macro_path = ROOT / "macro_regime.csv"
        if macro_path.exists():
            macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True).sort_index()
            macro_now = macro_df.reindex(all_dates).ffill().iloc[t]
            vix_v = float(macro_now.get("VIX", np.nan))
            if np.isfinite(vix_v) and vix_v > 30: tqqq_w *= 0.60
            hyg   = float(macro_now.get("HYG", np.nan))
            if np.isfinite(hyg) and np.isfinite(float(macro_df["HYG"].iloc[max(0,t-20):t].mean())):
                if hyg < float(macro_df["HYG"].iloc[max(0,t-20):t].mean()):
                    tqqq_w *= 0.70

        tma50 = float(tqqq_p[t - 50 : t].mean()) if t >= 50 else 0.0
        if float(tqqq_p[t]) <= tma50: tqqq_w *= 0.50

    stock_w = 0.40 if bull else 0.20

    holdings = []
    for ii, ci in enumerate(sel):
        tk  = tickers[ci]
        wt  = float(raw[ii]) * stock_w
        px  = float(now[ci])
        sc  = float(score[ci])
        sec = sector_map.get(tk, "Unknown")
        holdings.append(dict(
            signal_date=TODAY, ticker=tk, weight=round(wt, 5),
            score=round(sc, 2), price=round(px, 2),
            sector=sec, regime="bull" if bull else "bear",
        ))

    return holdings, tqqq_w, bull

# ── Run and save ──────────────────────────────────────────────────────────────
holdings, tqqq_w, bull = generate_signal()
spy_price = float(spy_all.iloc[-1])
spy_ma    = float(spy_all.iloc[-200:].mean())

print(f"Market regime: {'BULL' if bull else 'BEAR'}  "
      f"(SPY={spy_price:.2f}, 200MA={spy_ma:.2f})")
print(f"TQQQ overlay weight: {tqqq_w:.1%}")
print(f"Stock allocation: {0.40 if bull else 0.20:.0%}  across {len(holdings)} positions\n")

# Append to signal log
new_rows = pd.DataFrame(holdings)
if LOG_FILE.exists():
    existing = pd.read_csv(LOG_FILE)
    # Remove any existing rows for today (allow re-run)
    existing = existing[existing["signal_date"] != TODAY]
    combined = pd.concat([existing, new_rows], ignore_index=True)
else:
    combined = new_rows
combined.to_csv(LOG_FILE, index=False)
print(f"Signal log saved: {LOG_FILE.name}  "
      f"({len(combined)} total rows, {combined['signal_date'].nunique()} months)")

# Also save a clean summary row
summary = dict(
    signal_date=TODAY, regime="bull" if bull else "bear",
    n_holdings=len(holdings), tqqq_weight=round(tqqq_w, 4),
    stock_weight=round(0.40 if bull else 0.20, 2),
    spy_price=round(spy_price, 2), spy_ma200=round(spy_ma, 2),
    top5="|".join([h["ticker"] for h in holdings[:5]]),
)
hist_rows = [summary]
if HIST_FILE.exists():
    hist_df = pd.read_csv(HIST_FILE)
    hist_df = hist_df[hist_df["signal_date"] != TODAY]
    hist_df = pd.concat([hist_df, pd.DataFrame(hist_rows)], ignore_index=True)
else:
    hist_df = pd.DataFrame(hist_rows)
hist_df.to_csv(HIST_FILE, index=False)

# Print current predictions
print("\nTOP 25 PREDICTIONS:")
print(f"{'#':>3} {'Ticker':<7} {'Weight':>7} {'Score':>7} {'Sector':<22} {'Price':>8}")
print("-" * 60)
for i, h in enumerate(holdings, 1):
    print(f"{i:>3} {h['ticker']:<7} {h['weight']:>7.3%} {h['score']:>7.1f} "
          f"{h['sector'][:22]:<22} {h['price']:>8.2f}")

print(f"\n  TQQQ overlay: {tqqq_w:.3%}")
print(f"  Total equity ~{sum(h['weight'] for h in holdings):.1%} stocks "
      f"+ {tqqq_w:.1%} TQQQ")
print(f"\nNext step: run again on {pd.Timestamp(TODAY) + pd.offsets.MonthEnd(1):%Y-%m-%d} "
      f"to build the 2nd month of paper record")
print(f"\nPaper trading record started: {TODAY}")
print(f"6-month track record complete: {pd.Timestamp(TODAY) + pd.DateOffset(months=6):%Y-%m-%d}")
