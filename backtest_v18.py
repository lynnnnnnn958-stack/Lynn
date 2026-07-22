"""
Canyon Quant v18 — Path to Institutional 7/10
==============================================
Builds on v17c (best version: Sharpe 1.00, MDD -19.3%)

BLOCK A  Signal Fixes       → remove 4 negative-IC factors, conditional inv_vol
BLOCK B  Quality Factors    → accruals, gross_margin, ROA from edgar_fundamentals
BLOCK C  Replace TQQQ→QQQ   → remove leveraged ETF, use 1x QQQ
BLOCK D  SUE Integration    → if sue_data.csv exists, add earnings surprise factor

Sequential configs for comparison:
  v17c      baseline (best from v17 stack)
  v18a      + signal fixes only           (BLOCK A)
  v18b      + quality fundamentals        (BLOCK A + B)
  v18c      + QQQ instead of TQQQ        (BLOCK A + B + C)
  v18d      + SUE factor                 (BLOCK A + B + C + D, if available)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT   = Path(__file__).parent
HOLD_M = 21
WARMUP = 252
TCOST  = 5     # bps one-way
TARGET_VOL = 0.10
SPY_WIN    = 200

# ── Load price data ───────────────────────────────────────────────────────────
print("Loading data...")
prices  = pd.read_csv(ROOT / "sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy     = prices["SPY"].copy()
prices  = prices.drop(columns=["SPY"])
letf_df = pd.read_csv(ROOT / "letf_prices.csv", index_col=0, parse_dates=True).sort_index()
macro   = pd.read_csv(ROOT / "macro_regime.csv", index_col=0, parse_dates=True).sort_index()

sector_map = {}
sp = ROOT / "sp500_sectors.csv"
if sp.exists():
    sector_map = pd.read_csv(sp, index_col=0).squeeze().to_dict()
all_sectors = sorted(set(sector_map.values()))

common  = prices.index.intersection(letf_df.index).intersection(macro.index)
prices  = prices.reindex(common)
letf_df = letf_df.reindex(common).ffill().bfill()
spy     = spy.reindex(common).ffill()
macro   = macro.reindex(common).ffill()

price_arr = prices.values.astype(float)
spy_arr   = spy.values.astype(float)
tickers   = list(prices.columns)
col_idx   = {c: i for i, c in enumerate(tickers)}
N         = len(tickers)

# ETF price arrays
tqqq    = letf_df["TQQQ"].values.astype(float)
qld     = letf_df["QLD"].values.astype(float)
# QQQ ≈ QLD / 2 (2x ETF without drag is good proxy)
qqq_r   = np.concatenate([[np.nan],
           np.diff(np.log(np.where(qld > 0, qld, np.nan)))])
qqq_r   = qqq_r / 2.0          # de-leverage: QLD_daily / 2 ≈ QQQ_daily
tqqq_r  = np.concatenate([[np.nan],
           np.diff(np.log(np.where(tqqq > 0, tqqq, np.nan)))])

# Build cumulative price for QQQ proxy
qqq_px  = np.cumprod(np.where(np.isfinite(qqq_r), 1 + qqq_r, 1.0))
qqq_px  = qqq_px * float(tqqq[0]) / qqq_px[0]   # normalize to same start

rebal_all = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))
print(f"  {len(rebal_all)} periods  {N} stocks  {common[0].date()} → {common[-1].date()}")

# ── Regime ────────────────────────────────────────────────────────────────────
spy_ma200 = np.full(len(spy_arr), np.nan)
for i in range(SPY_WIN - 1, len(spy_arr)):
    spy_ma200[i] = spy_arr[i - SPY_WIN + 1 : i + 1].mean()
above_ma = spy_arr > spy_ma200

def is_bull(t):
    return t < SPY_WIN or bool(above_ma[t])

# ── Fundamental quality scores (point-in-time) ───────────────────────────────
print("  Building fundamental score cache...")
ef_raw = pd.read_csv(ROOT / "edgar_fundamentals.csv")
ef_raw["filed_date"] = pd.to_datetime(ef_raw["filed_date"], errors="coerce")
ef_raw = ef_raw.dropna(subset=["filed_date"]).sort_values("filed_date")

def _rank(arr):
    """0-100 percentile rank, NaN-safe."""
    result = np.full(len(arr), 50.0)   # default: neutral
    valid  = np.isfinite(arr)
    if valid.sum() < 2:
        return result
    result[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
    return result

def build_fundamental_score(t_date, tickers, col_idx):
    """
    Returns array (len = N_stocks) with quality score 0-100 per stock.
    Uses only filings with filed_date < t_date (no lookahead).
    Factors: accrual_ratio (low=good), gross_margin (high=good), roa (high=good)
    """
    cutoff = t_date
    recent = ef_raw[ef_raw["filed_date"] < cutoff]
    if recent.empty:
        return np.full(len(tickers), 50.0)

    latest = recent.sort_values("filed_date").groupby("ticker").last()

    scores = np.full(len(tickers), 50.0)
    for tk, row in latest.iterrows():
        ci = col_idx.get(tk)
        if ci is None:
            continue
        # Factor 1: low accruals = higher quality
        acc  = -row["accrual_ratio"] if pd.notna(row["accrual_ratio"]) else np.nan
        # Factor 2: high gross margin
        gm   = row["gross_margin"]  if pd.notna(row["gross_margin"])  else np.nan
        # Factor 3: high ROA
        roa  = row["roa"]           if pd.notna(row["roa"])           else np.nan
        vals = [v for v in [acc, gm, roa] if np.isfinite(v)]
        if vals:
            scores[ci] = np.mean(vals)   # raw composite, will be ranked globally

    # Convert to percentile rank within universe
    valid = np.isfinite(scores) & (scores != 50.0)
    if valid.sum() >= 10:
        ranked = np.full(len(scores), 50.0)
        ranked[valid] = (scores[valid].argsort().argsort() + 1) / valid.sum() * 100
        return ranked
    return scores

# Cache fundamentals at each rebalancing date (expensive once, reused)
print("  Caching fundamental scores at each rebalancing date...")
fundamental_cache = {}
for t in rebal_all:
    t_date = prices.index[t]
    fundamental_cache[t] = build_fundamental_score(t_date, tickers, col_idx)
print(f"  Done ({len(fundamental_cache)} periods)")

# ── SUE scores (if available) ─────────────────────────────────────────────────
sue_cache = {}
sue_path  = ROOT / "sue_data.csv"
HAS_SUE   = False
if sue_path.exists():
    sue_df = pd.read_csv(sue_path)
    sue_df["date"] = pd.to_datetime(sue_df["date"])
    # Keep only last 4 quarters (1yr) per ticker per rebal date
    print(f"  SUE data: {len(sue_df)} rows, {sue_df['ticker'].nunique()} tickers")
    for t in rebal_all:
        t_date = prices.index[t]
        cutoff = t_date
        window = cutoff - pd.DateOffset(months=12)
        recent = sue_df[(sue_df["date"] >= window) & (sue_df["date"] < cutoff)]
        if recent.empty:
            sue_cache[t] = np.full(N, 50.0)
            continue
        # Aggregate: mean SUE over trailing year per ticker
        agg = recent.groupby("ticker")["sue"].mean()
        arr = np.full(N, np.nan)
        for tk, val in agg.items():
            ci = col_idx.get(tk)
            if ci is not None:
                arr[ci] = val
        valid = np.isfinite(arr)
        ranked = np.full(N, 50.0)
        if valid.sum() >= 10:
            ranked[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
        sue_cache[t] = ranked
    covered = (sue_df["ticker"].nunique() / N)
    if covered > 0.3:
        HAS_SUE = True
        print(f"  SUE coverage: {covered:.0%} — ACTIVE")
    else:
        print(f"  SUE coverage: {covered:.0%} — TOO LOW, inactive")
else:
    print("  sue_data.csv not found — SUE factor inactive (run download_sue.py)")

# ── Stock scoring ─────────────────────────────────────────────────────────────
def compute_scores_v17c(t, top_n, mom_lk):
    """v17c original scoring for baseline comparison."""
    p = price_arr[:t + 1]
    now = p[-1]; p1m = p[-22]
    p3m = p[-64] if t >= 63 else p[0]
    n_lk = mom_lk * 21 + 21
    p_lk = p[-n_lk] if t >= n_lk - 1 else p[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        mom_lt = np.where(p_lk > 0, now / p_lk - 1, np.nan).clip(-1, 1)
        mom3m  = np.where(p3m > 0,  now / p3m  - 1, np.nan).clip(-1, 1)
        mom1m  = np.where(p1m > 0,  now / p1m  - 1, np.nan).clip(-0.5, 0.5)

    log_p  = np.log(np.where(p[-22:] > 0, p[-22:], np.nan))
    vol20  = np.nanstd(np.diff(log_p, axis=0), axis=0)
    invvol = 1.0 / (vol20 + 0.005)
    sma200 = np.nanmean(p[-200:], axis=0) if t >= 199 else np.nanmean(p, axis=0)
    above  = (now > sma200).astype(float)

    def rk(arr):
        r = np.full(len(arr), np.nan)
        v = np.isfinite(arr)
        if v.sum() < 2: return r
        r[v] = (arr[v].argsort().argsort() + 1) / v.sum() * 100
        return r

    def sector_rank(arr):
        result = np.full(len(arr), np.nan)
        for sec in all_sectors:
            sec_tks = [tk for tk in tickers if sector_map.get(tk) == sec]
            if len(sec_tks) < 2: continue
            sec_idx = [col_idx[tk] for tk in sec_tks if tk in col_idx]
            vals    = arr[sec_idx]; valid = np.isfinite(vals)
            if valid.sum() < 2: continue
            ranked  = np.full(len(vals), np.nan)
            ranked[valid] = (vals[valid].argsort().argsort() + 1) / valid.sum() * 100
            for ii, ci in enumerate(sec_idx):
                result[ci] = ranked[ii]
        missing = ~np.isfinite(result)
        if missing.any():
            result[missing] = rk(arr)[missing]
        return result

    score = (sector_rank(mom_lt) * 0.35 + sector_rank(mom3m) * 0.20 +
             sector_rank(invvol) * 0.25 + sector_rank(mom1m) * 0.10 +
             sector_rank(above) * 0.10)

    # Quality filter
    delta  = np.diff(log_p, axis=0)
    gains  = np.nanmean(np.clip(delta[-14:], 0, None), axis=0)
    loss   = np.nanmean(np.clip(-delta[-14:], 0, None), axis=0)
    rsi14  = 100 - 100 / (1 + gains / (loss + 1e-9))
    dist   = np.where(sma200 > 0, now / sma200 - 1, 0)
    vol75  = np.nanpercentile(vol20, 75)
    quality = (rsi14 < 75) & (dist < 0.5) & (vol20 < vol75)

    valid_mask = quality & np.isfinite(score)
    valid_idx  = np.where(valid_mask)[0]
    if len(valid_idx) < top_n:
        valid_idx = np.where(np.isfinite(score))[0]
    if len(valid_idx) == 0: return None, None

    sorted_idx = valid_idx[np.argsort(score[valid_idx])[::-1]]
    selected_idx, sector_counts = [], {}
    max_per_sec = max(1, int(top_n * 0.35))
    for ci in sorted_idx:
        if len(selected_idx) >= top_n: break
        sec = sector_map.get(tickers[ci], "Unknown")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if sector_counts[sec] <= max_per_sec:
            selected_idx.append(ci)
    if len(selected_idx) < 5:
        selected_idx = sorted_idx[:top_n].tolist()

    sel_vol = vol20[selected_idx].clip(min=0.001)
    raw_w   = 1.0 / sel_vol
    raw_w  /= raw_w.sum()
    raw_w   = np.clip(raw_w, 0, 0.15)
    raw_w  /= raw_w.sum()

    return [tickers[ci] for ci in selected_idx], dict(zip([tickers[ci] for ci in selected_idx], raw_w))


def compute_scores_v18(t, top_n, mom_lk, bull,
                       use_fundamentals=False, use_sue=False):
    """
    BLOCK A: Clean signals
      - Removed: inv_vol (fixed), mom_1m, mom_accel, new_high_52w
      - Added:   inv_vol CONDITIONAL (negative in bull, positive in bear)
      - New weights: mom_lt×0.45, mom3m×0.25, above_200×0.15, cond_invvol×0.15

    BLOCK B: Quality fundamentals (accruals, gross margin, ROA) — 20% blend

    BLOCK D: SUE factor — 15% blend (if available)
    """
    p   = price_arr[:t + 1]
    now = p[-1]; p1m = p[-22]
    p3m = p[-64] if t >= 63 else p[0]
    n_lk = mom_lk * 21 + 21
    p_lk = p[-n_lk] if t >= n_lk - 1 else p[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        mom_lt = np.where(p_lk > 0, now / p_lk - 1, np.nan).clip(-1, 1)
        mom3m  = np.where(p3m > 0,  now / p3m  - 1, np.nan).clip(-1, 1)

    log_p  = np.log(np.where(p[-22:] > 0, p[-22:], np.nan))
    vol20  = np.nanstd(np.diff(log_p, axis=0), axis=0)
    invvol = 1.0 / (vol20 + 0.005)
    sma200 = np.nanmean(p[-200:], axis=0) if t >= 199 else np.nanmean(p, axis=0)
    above  = (now > sma200).astype(float)

    def rk(arr):
        r = np.full(len(arr), np.nan)
        v = np.isfinite(arr)
        if v.sum() < 2: return r
        r[v] = (arr[v].argsort().argsort() + 1) / v.sum() * 100
        return r

    def sector_rank(arr):
        result = np.full(len(arr), np.nan)
        for sec in all_sectors:
            sec_tks = [tk for tk in tickers if sector_map.get(tk) == sec]
            if len(sec_tks) < 2: continue
            sec_idx = [col_idx[tk] for tk in sec_tks if tk in col_idx]
            vals    = arr[sec_idx]; valid = np.isfinite(vals)
            if valid.sum() < 2: continue
            ranked  = np.full(len(vals), np.nan)
            ranked[valid] = (vals[valid].argsort().argsort() + 1) / valid.sum() * 100
            for ii, ci in enumerate(sec_idx):
                result[ci] = ranked[ii]
        missing = ~np.isfinite(result)
        if missing.any():
            result[missing] = rk(arr)[missing]
        return result

    # ── BLOCK A: Conditional inv_vol ──────────────────────────────────────
    # Bull: high-vol growth stocks outperform → favor high-vol (negate inv_vol rank)
    # Bear: low-vol defensive stocks outperform → favor low-vol (standard inv_vol rank)
    invvol_rank = sector_rank(invvol)
    if bull:
        cond_invvol = 100.0 - invvol_rank   # flip: favor HIGH vol in bull
    else:
        cond_invvol = invvol_rank            # favor LOW vol in bear

    # Core momentum score (clean signals only)
    score = (sector_rank(mom_lt) * 0.45 +
             sector_rank(mom3m)  * 0.25 +
             sector_rank(above)  * 0.15 +
             cond_invvol         * 0.15)

    # ── BLOCK B: Quality fundamentals blend ───────────────────────────────
    if use_fundamentals:
        fund_score = fundamental_cache.get(t, np.full(N, 50.0))
        score = score * 0.80 + fund_score * 0.20

    # ── BLOCK D: SUE blend ───────────────────────────────────────────────
    if use_sue and HAS_SUE:
        sue_score = sue_cache.get(t, np.full(N, 50.0))
        score = score * 0.85 + sue_score * 0.15

    # Quality filter (same as v17c)
    delta  = np.diff(log_p, axis=0)
    gains  = np.nanmean(np.clip(delta[-14:], 0, None), axis=0)
    loss   = np.nanmean(np.clip(-delta[-14:], 0, None), axis=0)
    rsi14  = 100 - 100 / (1 + gains / (loss + 1e-9))
    dist   = np.where(sma200 > 0, now / sma200 - 1, 0)
    vol75  = np.nanpercentile(vol20, 75)
    quality = (rsi14 < 75) & (dist < 0.5) & (vol20 < vol75)

    valid_mask = quality & np.isfinite(score)
    valid_idx  = np.where(valid_mask)[0]
    if len(valid_idx) < top_n:
        valid_idx = np.where(np.isfinite(score))[0]
    if len(valid_idx) == 0: return None, None

    sorted_idx = valid_idx[np.argsort(score[valid_idx])[::-1]]
    selected_idx, sector_counts = [], {}
    max_per_sec = max(1, int(top_n * 0.35))
    for ci in sorted_idx:
        if len(selected_idx) >= top_n: break
        sec = sector_map.get(tickers[ci], "Unknown")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if sector_counts[sec] <= max_per_sec:
            selected_idx.append(ci)
    if len(selected_idx) < 5:
        selected_idx = sorted_idx[:top_n].tolist()

    sel_vol = vol20[selected_idx].clip(min=0.001)
    raw_w   = 1.0 / sel_vol
    raw_w  /= raw_w.sum()
    raw_w   = np.clip(raw_w, 0, 0.15)
    raw_w  /= raw_w.sum()

    return [tickers[ci] for ci in selected_idx], dict(zip([tickers[ci] for ci in selected_idx], raw_w))


def stock_ret(t, sel, wts, sw):
    if not sel or sw == 0: return 0.0
    t1 = min(t + HOLD_M, len(price_arr) - 1)
    r  = 0.0
    for tk, wt in wts.items():
        ci = col_idx.get(tk)
        if ci is None: continue
        p0, p1 = price_arr[t, ci], price_arr[t1, ci]
        if p0 > 0 and np.isfinite(p1):
            r += wt * (p1 / p0 - 1)
    return r * sw


def etf_alloc_weight(t, etf_max, use_qqq=False):
    """
    ETF weight: matches v17.py vol-based sizing exactly.
    vol_target=0.18 / TQQQ_realized_vol, capped at etf_max.
    For QQQ mode: use vol_target=0.10 (1x, lower target vol).
    """
    if not is_bull(t):
        return 0.0

    # Realized vol of the ETF (last 20 trading days, annualized)
    r_arr  = tqqq_r if not use_qqq else qqq_r
    window = r_arr[max(0, t - 20) : t]
    window = window[np.isfinite(window)]
    if len(window) >= 5:
        etf_vol = float(np.std(window) * np.sqrt(252))
    else:
        etf_vol = 0.60 if not use_qqq else 0.20

    vol_target = 0.18 if not use_qqq else 0.10
    ew = float(np.clip(vol_target / (etf_vol + 0.001), 0.0, etf_max))

    # VIX filter
    vix_v = float(macro["VIX"].iloc[t]) if t < len(macro) else 20.0
    if np.isfinite(vix_v) and vix_v > 30:
        ew *= 0.6

    # Trend gate: ETF must be above its 50MA
    ref_arr = tqqq if not use_qqq else qqq_px
    ref_ma  = ref_arr[t - 50 : t].mean() if t >= 50 else ref_arr[t]
    if float(ref_arr[t]) <= float(ref_ma):
        ew *= 0.5

    return ew


def etf_ret_period(t, t1, use_qqq=False):
    """Period return for TQQQ or QQQ."""
    if use_qqq:
        p0 = qqq_px[t]; p1 = qqq_px[t1]
    else:
        p0 = float(tqqq[t]); p1 = float(tqqq[t1])
    return float(p1 / p0 - 1) if (p0 > 0 and np.isfinite(p1)) else 0.0


def vol_scale(past_rets):
    if len(past_rets) < 4: return 1.0
    arr = np.array(past_rets[-12:])
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 1.0
    ann = float(arr.std() * np.sqrt(252 / HOLD_M))
    return float(np.clip(TARGET_VOL / (ann + 0.001), 0.5, 2.0))


# ── Main backtest ─────────────────────────────────────────────────────────────
def run(label, etf_max=0.40, mom_lk=9, top_n=25,
        use_v17c_score=False, use_fundamentals=False,
        use_sue=False, use_qqq=False):

    records   = []
    past_rets = []
    score_cache = {}

    for t in rebal_all:
        t_date = prices.index[t]
        bull   = is_bull(t)
        sw     = 0.40 if bull else 0.20

        # Stock selection
        key = (t, mom_lk, use_v17c_score, use_fundamentals, use_sue, bull)
        if key not in score_cache:
            if use_v17c_score:
                score_cache[key] = compute_scores_v17c(t, top_n, mom_lk)
            else:
                score_cache[key] = compute_scores_v18(
                    t, top_n, mom_lk, bull,
                    use_fundamentals=use_fundamentals,
                    use_sue=use_sue)
        sel, wts = score_cache[key]

        sr = stock_ret(t, sel, wts, 1.0)

        # ETF — use vol-based allocation (matches v17.py exactly)
        ew = etf_alloc_weight(t, etf_max, use_qqq=use_qqq)

        scale = vol_scale(past_rets)

        t1    = min(t + HOLD_M, len(price_arr) - 1)
        er    = etf_ret_period(t, t1, use_qqq=use_qqq) if ew > 0 else 0.0
        ret   = (sw * sr + ew * er) * scale - TCOST / 10_000
        spy_r = float(spy_arr[t1]) / float(spy_arr[t]) - 1

        past_rets.append(ret)
        records.append(dict(
            date=t_date.strftime("%Y-%m-%d"),
            ret=round(float(ret), 6),
            spy_ret=round(float(spy_r), 6),
            etf_w=round(float(ew * scale), 3),
            stock_w=round(float(sw * scale), 3),
        ))

    return pd.DataFrame(records)


def stats(df, label=""):
    r     = np.array(df["ret"])
    spy_r = np.array(df["spy_ret"])
    ppy   = 252 / HOLD_M
    tot   = float(np.prod(1 + r))
    ar    = float(tot ** (ppy / len(r)) - 1)
    av    = float(r.std() * np.sqrt(ppy))
    sr    = ar / (av + 1e-9)
    cum   = np.cumprod(1 + r)
    peak  = np.maximum.accumulate(cum)
    mdd   = float(np.max((peak - cum) / peak))
    neg   = r[r < 0]
    sortino = ar / (neg.std() * np.sqrt(ppy) + 1e-9) if len(neg) > 1 else 0.0
    spy_ar  = float(np.prod(1 + spy_r) ** (ppy / len(spy_r)) - 1)

    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["date"]).dt.year
    yrs = {}
    for yr, g in df2.groupby("year"):
        yrs[yr] = (float(np.prod(1 + g["ret"]) - 1),
                   float(np.prod(1 + g["spy_ret"]) - 1))
    beat = sum(1 for p, s in yrs.values() if p > s)

    return dict(label=label, ar=ar, av=av, sr=sr, mdd=mdd, sortino=sortino,
                total=tot - 1, spy_ar=spy_ar, beat=beat, n_yr=len(yrs), yrs=yrs,
                avg_etf_w=df["etf_w"].mean(), avg_stock_w=df["stock_w"].mean())


def print_yoy(df, label):
    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["date"]).dt.year
    print(f"  Year  {label[:16]:>16}     SPY    Alpha")
    for yr, g in df2.groupby("year"):
        p = float(np.prod(1 + g["ret"]) - 1)
        s = float(np.prod(1 + g["spy_ret"]) - 1)
        mk = "✓" if p > s else "✗"
        print(f"  {yr}  {p:>+7.1%}              {s:>+7.1%}  {mk}{p-s:>+7.1%}")


# ── Run configurations ────────────────────────────────────────────────────────
configs = [
    # (label, use_v17c_score, use_fundamentals, use_sue, use_qqq)
    ("v17c baseline",          True,  False, False, False),
    ("v18a: +signal fixes",    False, False, False, False),
    ("v18b: +fundamentals",    False, True,  False, False),
    ("v18c: +QQQ (no TQQQ)",   False, True,  False, True),
]
if HAS_SUE:
    configs.append(("v18d: +SUE factor", False, True, True, True))

print("\nRunning backtest configurations...")
results = []
df_store = {}
for (lbl, v17c_score, fund, sue, qqq) in configs:
    print(f"  {lbl}...")
    df = run(lbl, use_v17c_score=v17c_score, use_fundamentals=fund,
             use_sue=sue, use_qqq=qqq)
    s  = stats(df, lbl)
    results.append(s)
    df_store[lbl] = df

# ── Print results ─────────────────────────────────────────────────────────────
W = 82
SURV = 0.025   # survivorship bias correction

print("\n" + "═" * W)
print("  CANYON QUANT v18 — PATH TO 7/10")
print("═" * W)
print(f"\n  {'Version':<26} {'AR(rep)':>8} {'AR(adj)':>8} {'Sharpe':>7} "
      f"{'Sortino':>8} {'MDD':>8} {'Beat':>5}")
print("  " + "─" * 74)
for s in results:
    adj = s["ar"] - SURV
    print(f"  {s['label']:<26} {s['ar']:>+7.1%}  {adj:>+7.1%}  {s['sr']:>7.2f}  "
          f"{s['sortino']:>8.2f}  {-s['mdd']:>+7.1%}  {s['beat']}/{s['n_yr']}")

# ── Year-by-year detail for each version ─────────────────────────────────────
print(f"\n  Year-by-year detail:")
print(f"  {'Year':>5}  {'v17c':>7}  {'v18a':>7}  {'v18b':>7}  {'v18c':>7}  {'SPY':>7}  Notes")
print("  " + "─" * 75)
labels = [r["label"] for r in results]
dfs    = [df_store[l] for l in labels]
all_yrs = sorted(results[0]["yrs"].keys())
for yr in all_yrs:
    row_vals = []
    spy_v    = None
    for s in results:
        p, sp = s["yrs"].get(yr, (np.nan, np.nan))
        row_vals.append(p)
        spy_v = sp
    best_v = max(row_vals)
    parts  = [f"  {yr:>5}"]
    for v in row_vals:
        mk = "★" if v == best_v else " "
        parts.append(f"  {mk}{v:>+6.1%}")
    parts.append(f"  {spy_v:>+7.1%}")
    print("".join(parts))

# ── Signal IC verification ────────────────────────────────────────────────────
print(f"\n  Signal IC verification (v18 clean signals):")
ic_results = []
for sig_name, sig_fn in [
    ("mom_12m_skip1m", lambda p, t: np.where(
        p[max(0,t-273):t+1][-1] > p[max(0,t-273):t+1][0],
        p[max(0,t-273):t+1][-1] / p[max(0,t-273):t+1][0] - 1, np.nan)),
]:
    ics = []
    for t in rebal_all[4:]:
        p = price_arr[:t + 1]
        n_lk = 9 * 21 + 21
        p_lk = p[-n_lk] if t >= n_lk - 1 else p[0]
        now  = p[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            mom_lt = np.where(p_lk > 0, now / p_lk - 1, np.nan).clip(-1, 1)
        t1 = min(t + HOLD_M, len(price_arr) - 1)
        fwd = price_arr[t1] / price_arr[t] - 1
        common = np.isfinite(mom_lt) & np.isfinite(fwd)
        if common.sum() < 20: continue
        ic, _ = spearmanr(mom_lt[common], fwd[common])
        if np.isfinite(ic): ics.append(ic)

    if ics:
        arr    = np.array(ics)
        t_stat = arr.mean() / arr.std() * np.sqrt(len(arr))
        print(f"  {sig_name:<20}  IC={arr.mean():+.4f}  t={t_stat:+.2f}  "
              f"hit={( arr>0).mean():.0%}  "
              f"{'SIGNIFICANT' if abs(t_stat)>2 else 'weak'}")

# ── Alpha quality scorecard ───────────────────────────────────────────────────
print(f"\n  Alpha quality change (v17c → v18b):")
print(f"  v17c: inv_vol IC=-0.108 (NEGATIVE), 4 harmful signals active")
print(f"  v18a: conditional inv_vol, harmful signals removed")
if HAS_SUE:
    print(f"  v18d: + SUE factor (earnings surprise)")

# ── Save outputs ──────────────────────────────────────────────────────────────
for lbl, df in df_store.items():
    fname = lbl.split(":")[0].replace(" ", "_").replace("/", "") + ".csv"
    df.to_csv(ROOT / fname, index=False)

pd.DataFrame([{k: v for k, v in s.items() if k != "yrs"} for s in results]).to_csv(
    ROOT / "v18_summary.csv", index=False)

print(f"\n  Saved: v18_summary.csv + individual CSV files")
print(f"\n  SPY AR same period: {results[0]['spy_ar']:+.1%}")
print(f"  Survivorship bias correction: -{SURV:.1%}/yr applied to all")
print("═" * W)
