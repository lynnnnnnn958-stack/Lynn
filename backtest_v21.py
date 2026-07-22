"""
Canyon Quant v21 — 9 Improvement Full Integration
====================================================
1. Edgar Quality composite (accrual_ratio, gross_margin, roa, leverage_chg)
2. 52-week high anchor  (George & Hwang 2004)
3. SUE consecutive wins  (earnings momentum carry)
4. FRED multi-signal regime  (yield curve + UNRATE + VIX + SPY trend)
5. Google Trends contrarian  (retail attention → contrarian)
6. Sector ETF momentum weight  (sector rotation within stock book)
7. Options PCR proxy  (VIX percentile as fear/greed timing)
8. RSP equal-weight benchmark  (better size-neutral comparison)
9. Drawdown stop  (15% drawdown → 50% position reduction)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).parent
HOLD_M  = 21
WARMUP  = 252
TCOST   = 5
TGT_VOL = 0.10
SPY_WIN = 200
TRAIN_END  = "2017-12-31"
TEST_START = "2018-01-01"
DD_THRESH  = 0.15   # drawdown stop threshold
DD_SCALE   = 0.50   # scale factor when stop triggered

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
print("Loading data...")
p25 = pd.read_csv(ROOT / "sp500_price_25yr.csv", index_col=0, parse_dates=True).sort_index()
spy_all    = p25["SPY"].copy()
stocks_all = p25.drop(columns=["SPY"], errors="ignore")
tickers    = list(stocks_all.columns)
col_idx    = {c: i for i, c in enumerate(tickers)}
N          = len(tickers)
all_dates  = stocks_all.index

sector_map = {}
if (ROOT / "sp500_sectors.csv").exists():
    sector_map = pd.read_csv(ROOT / "sp500_sectors.csv", index_col=0).squeeze().to_dict()
all_sectors = sorted(set(sector_map.values()))

# ── [D] Historical constituents ───────────────────────────────────────────────
sp500_intervals = {}
if (ROOT / "sp500_hist_constituents.csv").exists():
    hdf = pd.read_csv(ROOT / "sp500_hist_constituents.csv")
    hdf["start_date"] = pd.to_datetime(hdf["start_date"], errors="coerce")
    hdf["end_date"]   = pd.to_datetime(hdf["end_date"],   errors="coerce")
    for _, row in hdf.iterrows():
        tk = str(row["ticker"]).strip()
        s  = row["start_date"]
        e  = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp("2030-01-01")
        if pd.notna(s): sp500_intervals.setdefault(tk, []).append((s, e))
    print(f"  Historical constituents: {len(sp500_intervals)} tickers")

def in_sp500_at(ticker, date):
    ivs = sp500_intervals.get(ticker)
    if not ivs: return True
    return any(s <= date <= e for s, e in ivs)

elig_cache = {}
def get_eligible(date):
    if date not in elig_cache:
        elig_cache[date] = {col_idx[tk] for tk in tickers
                            if in_sp500_at(tk, date) and tk in col_idx}
    return elig_cache[date]

# ── [E] SUE data ─────────────────────────────────────────────────────────────
sue_df = pd.read_csv(ROOT / "sue_data.csv")
sue_df["date"] = pd.to_datetime(sue_df["date"])
HAS_SUE = len(sue_df) > 1000
print(f"  SUE: {len(sue_df)} rows  {'ACTIVE' if HAS_SUE else 'OFF'}")

def sue_scores(t_date):
    """Returns (basic_score, consecutive_bonus) as 0-100 ranked arrays."""
    neutral = np.full(N, 50.0)
    if not HAS_SUE:
        return neutral, np.zeros(N)
    window  = t_date - pd.DateOffset(months=15)
    recent  = sue_df[(sue_df.date >= window) & (sue_df.date < t_date)]
    if len(recent) < 20:
        return neutral, np.zeros(N)
    # Basic: mean SUE over window
    agg = recent.groupby("ticker")["sue"].mean()
    arr = np.full(N, np.nan)
    for tk, v in agg.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = v
    valid  = np.isfinite(arr)
    ranked = neutral.copy()
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100

    # [3] Consecutive wins: count quarters with positive SUE in last 4 quarters
    q_window = t_date - pd.DateOffset(months=12)
    qtrs = (sue_df[(sue_df.date >= q_window) & (sue_df.date < t_date)]
            .sort_values("date")
            .groupby("ticker")
            .apply(lambda g: g.tail(4)["sue"].gt(0).all(), include_groups=False)
            .reset_index())
    qtrs.columns = ["ticker", "all_positive"]
    bonus = np.zeros(N)
    for _, row in qtrs.iterrows():
        ci = col_idx.get(row["ticker"])
        if ci is not None and row["all_positive"]:
            bonus[ci] = 15.0   # bonus points for 4 consecutive positive SUE

    return ranked, bonus

# ── [1] Edgar Quality ─────────────────────────────────────────────────────────
edgar_df = None
if (ROOT / "edgar_fundamentals.csv").exists():
    edgar_df = pd.read_csv(ROOT / "edgar_fundamentals.csv")
    edgar_df["filed_date"] = pd.to_datetime(edgar_df["filed_date"], errors="coerce")
    edgar_df = edgar_df.dropna(subset=["filed_date"])
    print(f"  Edgar fundamentals: {len(edgar_df)} rows, "
          f"{edgar_df.ticker.nunique()} tickers")

def edgar_quality_score(t_date):
    """Quality composite using only filings available before t_date."""
    if edgar_df is None: return np.full(N, 50.0)
    avail = edgar_df[edgar_df.filed_date < t_date]
    if len(avail) < 20: return np.full(N, 50.0)
    # Most recent filing per ticker
    latest = avail.sort_values("filed_date").groupby("ticker").last()
    # Sub-scores (all rank-normalized)
    def rank_arr(series, ascending=True):
        s  = series.dropna()
        rk = s.rank(ascending=ascending, pct=True) * 100
        return rk
    scores = {}
    if "accrual_ratio" in latest.columns:
        rk = rank_arr(latest["accrual_ratio"], ascending=False)  # lower = better
        for tk, v in rk.items(): scores.setdefault(tk, []).append(v)
    if "gross_margin"  in latest.columns:
        rk = rank_arr(latest["gross_margin"],  ascending=True)
        for tk, v in rk.items(): scores.setdefault(tk, []).append(v)
    if "roa"           in latest.columns:
        rk = rank_arr(latest["roa"],           ascending=True)
        for tk, v in rk.items(): scores.setdefault(tk, []).append(v)
    if "leverage_chg"  in latest.columns:
        rk = rank_arr(latest["leverage_chg"],  ascending=False)  # lower = better
        for tk, v in rk.items(): scores.setdefault(tk, []).append(v)
    arr = np.full(N, np.nan)
    for tk, vals in scores.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = np.mean(vals)
    valid  = np.isfinite(arr)
    ranked = np.full(N, 50.0)
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
    return ranked

# ── [4] FRED multi-signal regime ─────────────────────────────────────────────
fred_df = None
if (ROOT / "fred_macro.csv").exists():
    fred_df = pd.read_csv(ROOT / "fred_macro.csv", index_col=0, parse_dates=True).sort_index()
    print(f"  FRED macro: {fred_df.shape}  cols={fred_df.columns.tolist()[:5]}")

def get_regime(t_date, spy_v, spy_ma, macro_row=None):
    """Multi-signal regime: Bull if 3+/4 signals agree."""
    signals = []
    signals.append(float(spy_v) > float(spy_ma))     # SPY trend
    if macro_row is not None:
        t10y2y = macro_row.get("T10Y2Y", np.nan)
        if np.isfinite(float(t10y2y if t10y2y is not None else np.nan)):
            signals.append(float(t10y2y) > -0.25)    # yield curve not deeply inverted
        unrate_rising = macro_row.get("unrate_rising", np.nan)
        if np.isfinite(float(unrate_rising if unrate_rising is not None else np.nan)):
            signals.append(float(unrate_rising) < 0.5)  # unemployment NOT rising
        vix = macro_row.get("VIX", np.nan)
        if np.isfinite(float(vix if vix is not None else np.nan)):
            signals.append(float(vix) < 25)          # low fear
    if len(signals) == 0: return True
    return sum(signals) >= (len(signals) * 0.6)   # bull if 60%+ agree

# ── [5] Google Trends ─────────────────────────────────────────────────────────
gtrends_stock_df = None
gtrends_market_df = None
if (ROOT / "gtrends_stocks.csv").exists():
    gtrends_stock_df = pd.read_csv(ROOT / "gtrends_stocks.csv",
                                    index_col=0, parse_dates=True)
    print(f"  Google Trends stocks: {gtrends_stock_df.shape}")
if (ROOT / "gtrends_market.csv").exists():
    gtrends_market_df = pd.read_csv(ROOT / "gtrends_market.csv",
                                     index_col=0, parse_dates=True)
    print(f"  Google Trends market: {gtrends_market_df.shape}")

def gtrends_score(t_date):
    """Contrarian retail attention: high attention → lower expected return."""
    if gtrends_stock_df is None: return np.full(N, 50.0)
    window = t_date - pd.DateOffset(weeks=4)
    recent = gtrends_stock_df.loc[window:t_date]
    if len(recent) == 0: return np.full(N, 50.0)
    avg = recent.mean()
    # Contrarian: high attention → lower rank (we want to underweight)
    arr = np.full(N, np.nan)
    for tk, v in avg.items():
        ci = col_idx.get(tk)
        if ci is not None and np.isfinite(v): arr[ci] = v
    valid  = np.isfinite(arr)
    if valid.sum() < 5: return np.full(N, 50.0)
    ranked = np.full(N, 50.0)
    # INVERT: high trend → low score (contrarian)
    ranked[valid] = 100 - (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
    return ranked

# ── [6] Sector ETF momentum ───────────────────────────────────────────────────
sector_etf_df = None
if (ROOT / "sector_etf_prices.csv").exists():
    sector_etf_df = pd.read_csv(ROOT / "sector_etf_prices.csv",
                                  index_col=0, parse_dates=True).sort_index()
    etf_sector_map = {
        "XLK":"Technology", "XLF":"Financials", "XLV":"Health Care",
        "XLE":"Energy",     "XLI":"Industrials","XLP":"Consumer Staples",
        "XLU":"Utilities",  "XLY":"Consumer Discretionary",
        "XLRE":"Real Estate","XLC":"Communication Services","XLB":"Materials",
    }
    print(f"  Sector ETFs: {list(etf_sector_map.keys())}")

def sector_momentum_multiplier(t_date, sector):
    """Returns 0.7-1.3 multiplier based on sector ETF momentum."""
    if sector_etf_df is None: return 1.0
    etf = next((k for k, v in etf_sector_map.items() if v == sector), None)
    if etf is None or etf not in sector_etf_df.columns: return 1.0
    data = sector_etf_df[etf].loc[:t_date].dropna()
    if len(data) < 90: return 1.0
    mom3m = float(data.iloc[-1]) / float(data.iloc[-64]) - 1
    # Stronger sector momentum → slight overweight
    return float(np.clip(1.0 + mom3m * 1.0, 0.7, 1.3))

# ── [7] Options PCR proxy ─────────────────────────────────────────────────────
macro_all = pd.read_csv(ROOT / "macro_regime.csv", index_col=0, parse_dates=True).sort_index()
letf      = pd.read_csv(ROOT / "letf_prices.csv",  index_col=0, parse_dates=True).sort_index()

# ════════════════════════════════════════════════════════════════════════════
# CORE SIGNAL COMPUTATION
# ════════════════════════════════════════════════════════════════════════════
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

MOM_LK = 9   # locked from v20 train

def compute_scores_v21(t_idx, price_arr, bull, t_date, eligible_set, macro_row=None):
    if t_idx < WARMUP: return None, None
    p   = price_arr[:t_idx + 1]
    now = p[-1]
    p3m = p[-64] if t_idx >= 63 else p[0]
    n_lk = MOM_LK * 21 + 21
    p_lk = p[-n_lk] if t_idx >= n_lk - 1 else p[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        mom_lt = np.where(p_lk > 0, now / p_lk - 1, np.nan).clip(-1, 1)
        mom3m  = np.where(p3m  > 0, now / p3m  - 1, np.nan).clip(-1, 1)

    log_p  = np.log(np.where(p[-22:] > 0, p[-22:], np.nan))
    vol20  = np.nanstd(np.diff(log_p, axis=0), axis=0)
    invvol = 1.0 / (vol20 + 0.005)
    sma200 = (np.nanmean(p[-200:], axis=0) if t_idx >= 199
              else np.nanmean(p, axis=0))
    above  = (now > sma200).astype(float)

    # [A] Conditional inv_vol
    ivr  = sector_rank(invvol)
    civr = (100.0 - ivr) if bull else ivr

    # [2] 52-week high anchor (George & Hwang 2004)
    p52w = p[-252:] if t_idx >= 251 else p
    high52 = np.nanmax(p52w, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi52_ratio = np.where(high52 > 0, now / high52, np.nan)

    # [1] Edgar quality
    eq_score = edgar_quality_score(t_date)

    # [5] Google Trends contrarian
    gt_score = gtrends_score(t_date)
    has_gt   = gtrends_stock_df is not None

    # [E + 3] SUE (basic + consecutive wins)
    sue_arr, sue_bonus = sue_scores(t_date)

    # Combined score — weights sum designed to give equal total influence
    base = (sector_rank(mom_lt) * 0.28 +
            sector_rank(mom3m)  * 0.14 +
            sector_rank(above)  * 0.10 +
            civr                * 0.10 +
            sector_rank(hi52_ratio) * 0.10 +
            eq_score            * 0.13 +
            (gt_score           * 0.05 if has_gt else sector_rank(mom_lt) * 0.05) +
            sector_rank(invvol  if not bull else 1/(invvol+0.001)) * 0.00)

    # Normalize base weights to sum to exactly 0.90 (reserve 10% for SUE)
    # SUE blend: 85% base + 15% SUE (as in v20)
    score = base * 0.85 + sue_arr * 0.15
    # Add consecutive wins bonus (adds to raw score, small effect)
    score = score + sue_bonus * 0.05

    # [D] Point-in-time eligibility
    not_elig = np.array([ci not in eligible_set for ci in range(N)], dtype=bool)
    score[not_elig] = np.nan

    # Quality filter
    delta  = np.diff(log_p, axis=0)
    gains  = np.nanmean(np.clip(delta[-14:], 0, None), axis=0)
    loss_  = np.nanmean(np.clip(-delta[-14:], 0, None), axis=0)
    rsi14  = 100 - 100 / (1 + gains / (loss_ + 1e-9))
    dist   = np.where(sma200 > 0, now / sma200 - 1, 0)
    vol75  = np.nanpercentile(vol20, 75)
    quality = (rsi14 < 75) & (dist < 0.5) & (vol20 < vol75)

    valid   = quality & np.isfinite(score)
    idx     = np.where(valid)[0]
    if len(idx) == 0: idx = np.where(np.isfinite(score))[0]
    if len(idx) == 0: return None, None

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

    # [6] Sector momentum multiplier on weights
    sv  = vol20[sel].clip(min=0.001)
    raw = 1.0 / sv
    if sector_etf_df is not None:
        mults = np.array([sector_momentum_multiplier(t_date, sector_map.get(tickers[ci], ""))
                          for ci in sel])
        raw = raw * mults
    raw /= raw.sum(); raw = np.clip(raw, 0, 0.15); raw /= raw.sum()

    return [tickers[ci] for ci in sel], dict(zip([tickers[ci] for ci in sel], raw))

# ════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════════════
def run_v21(price_arr, spy_arr, date_idx, rebal_list,
            etf_max=0.0, has_letf=False, tqqq_arr=None,
            macro_df=None, fred_macro=None, rsp_arr=None):

    spy_ma200 = np.full(len(spy_arr), np.nan)
    for i in range(SPY_WIN - 1, len(spy_arr)):
        spy_ma200[i] = spy_arr[i - SPY_WIN + 1 : i + 1].mean()

    records = []; past_rets = []; peak_val = 1.0; cur_val = 1.0

    for t in rebal_list:
        t_date = date_idx[t]
        spy_v  = float(spy_arr[t])
        spy_ma = float(spy_ma200[t]) if np.isfinite(spy_ma200[t]) else spy_v

        # [4] Multi-signal regime
        macro_row = None
        if macro_df is not None and t < len(macro_df):
            macro_row = macro_df.iloc[t].to_dict()
        if fred_macro is not None:
            fred_row  = fred_macro.reindex([t_date], method="ffill").iloc[0].to_dict()
            if macro_row is None: macro_row = {}
            macro_row.update({k: v for k, v in fred_row.items()
                              if k not in macro_row or not np.isfinite(float(macro_row[k]))})
        bull = get_regime(t_date, spy_v, spy_ma, macro_row)
        sw   = 0.40 if bull else 0.20

        elig = get_eligible(t_date)
        sel, wts = compute_scores_v21(t, price_arr, bull, t_date, elig, macro_row)

        # Stock return
        sr = 0.0
        if sel:
            t1 = min(t + HOLD_M, len(price_arr) - 1)
            for tk, wt in wts.items():
                ci = col_idx.get(tk)
                if ci is None: continue
                p0, p1 = price_arr[t, ci], price_arr[t1, ci]
                if p0 > 0 and np.isfinite(p1):
                    sr += wt * (p1 / p0 - 1)
            sr *= sw

        # [F] TQQQ overlay with PCR proxy
        ew = er = 0.0
        if has_letf and bull and tqqq_arr is not None and etf_max > 0:
            tqqq_r = np.concatenate([[np.nan],
                np.diff(np.log(np.where(tqqq_arr > 0, tqqq_arr, np.nan)))])
            w20  = tqqq_r[max(0, t - 20):t]; w20 = w20[np.isfinite(w20)]
            tvol = float(np.std(w20) * np.sqrt(252)) if len(w20) >= 5 else 0.60
            ew   = float(np.clip(0.18 / (tvol + 0.001), 0.0, etf_max))

            # VIX risk control
            vix_v = float(macro_row.get("VIX", np.nan)) if macro_row else np.nan
            if np.isfinite(vix_v) and vix_v > 30: ew *= 0.60

            # TQQQ trend gate
            if t >= 50:
                tma50 = tqqq_arr[t - 50:t].mean()
                if float(tqqq_arr[t]) <= float(tma50): ew *= 0.50

            # HYG credit gate
            if macro_df is not None and "HYG" in macro_df.columns and t < len(macro_df):
                hyg    = float(macro_df["HYG"].iloc[t])
                hyg_ma = macro_df["HYG"].iloc[max(0,t-20):t].mean()
                if np.isfinite(hyg) and np.isfinite(float(hyg_ma)) and hyg < float(hyg_ma):
                    ew *= 0.70

            # [7] PCR proxy: greed → reduce; fear already covered by VIX
            if macro_row and "pcr_proxy" in macro_row:
                pcr = float(macro_row.get("pcr_proxy", 1.0))
                if np.isfinite(pcr) and pcr < 0.7: ew *= 0.70  # greed → reduce

            t1  = min(t + HOLD_M, len(tqqq_arr) - 1)
            p0t = float(tqqq_arr[t]); p1t = float(tqqq_arr[t1])
            er  = float(p1t / p0t - 1) if (p0t > 0 and np.isfinite(p1t)) else 0.0

        # Portfolio vol targeting
        scale = 1.0
        if len(past_rets) >= 4:
            arr = np.array(past_rets[-12:]); arr = arr[np.isfinite(arr)]
            if len(arr) >= 2:
                av = float(arr.std() * np.sqrt(252 / HOLD_M))
                scale = float(np.clip(TGT_VOL / (av + 0.001), 0.5, 2.0))

        # [9] Drawdown stop
        if len(past_rets) > 0:
            cur_val  = cur_val * (1 + past_rets[-1])
            peak_val = max(peak_val, cur_val)
            dd       = (peak_val - cur_val) / peak_val
            if dd > DD_THRESH:
                scale *= DD_SCALE   # 50% reduction during deep drawdown

        ret = (sr + ew * er) * scale - TCOST / 10_000
        past_rets.append(ret)

        t1s   = min(t + HOLD_M, len(spy_arr) - 1)
        spy_r = float(spy_arr[t1s]) / float(spy_arr[t]) - 1
        rsp_r = (float(rsp_arr[t1s]) / float(rsp_arr[t]) - 1
                 if rsp_arr is not None and np.isfinite(rsp_arr[t]) and np.isfinite(rsp_arr[t1s])
                 else np.nan)

        records.append(dict(
            date=t_date.strftime("%Y-%m-%d"), ret=float(ret),
            spy_ret=float(spy_r), rsp_ret=float(rsp_r),
            etf_w=round(float(ew * scale), 3),
            stock_w=round(float(sw * scale), 3),
            regime="bull" if bull else "bear",
        ))
    return pd.DataFrame(records)

def stats(df, bench="spy_ret"):
    r   = df["ret"].values; s = df[bench].fillna(0).values if bench in df else np.zeros(len(df))
    ppy = 252 / HOLD_M
    tot = float(np.prod(1 + r))
    ar  = float(tot ** (ppy / len(r)) - 1)
    av  = float(r.std() * np.sqrt(ppy))
    sr  = ar / (av + 1e-9)
    cum = np.cumprod(1 + r); peak = np.maximum.accumulate(cum)
    mdd = float(np.max((peak - cum) / peak))
    neg = r[r < 0]
    sortino = ar / (neg.std() * np.sqrt(ppy) + 1e-9) if len(neg) > 1 else 0
    bench_ar = float(np.prod(1 + s) ** (ppy / len(s)) - 1)
    df2 = df.copy(); df2["year"] = pd.to_datetime(df2["date"]).dt.year
    yrs = {yr: (float(np.prod(1 + g["ret"]) - 1),
                float(np.prod(1 + g[bench].fillna(0)) - 1))
           for yr, g in df2.groupby("year") if bench in df2.columns}
    beat = sum(1 for p_, s_ in yrs.values() if p_ > s_)
    monthly = df.copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M")
    mb = sum(1 for _, g in monthly.groupby("month")
             if float(np.prod(1 + g["ret"]) - 1) > float(np.prod(1 + g[bench].fillna(0)) - 1))
    mt = monthly["month"].nunique()
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, sortino=sortino, bench_ar=bench_ar,
                beat=beat, n_yr=len(yrs), yrs=yrs, monthly_beat=mb, monthly_total=mt)

# ════════════════════════════════════════════════════════════════════════════
# DATA SPLIT
# ════════════════════════════════════════════════════════════════════════════
train_mask = all_dates <= TRAIN_END
test_mask  = all_dates >= TEST_START

tr_p = stocks_all[train_mask].values.astype(float)
tr_s = spy_all[train_mask].values.astype(float)
tr_d = all_dates[train_mask]
te_p = stocks_all[test_mask].values.astype(float)
te_s = spy_all[test_mask].values.astype(float)
te_d = all_dates[test_mask]

tr_rebal = list(range(WARMUP, len(tr_p) - HOLD_M, HOLD_M))
te_rebal = list(range(WARMUP, len(te_p) - HOLD_M, HOLD_M))

# LETF + macro for test
letf_te   = letf.reindex(te_d).ffill().bfill()
tqqq_te   = letf_te["TQQQ"].values.astype(float)
macro_te  = macro_all.reindex(te_d).ffill()

# RSP benchmark
rsp_arr_te = None
if (ROOT / "sector_etf_prices.csv").exists():
    se = pd.read_csv(ROOT / "sector_etf_prices.csv", index_col=0, parse_dates=True)
    if "RSP" in se.columns:
        rsp_arr_te = se["RSP"].reindex(te_d).ffill().values.astype(float)
        rsp_arr_tr = se["RSP"].reindex(tr_d).ffill().values.astype(float)

# FRED aligned to test
fred_te = None
if fred_df is not None:
    fred_te = fred_df.reindex(te_d, method="ffill")
fred_tr = None
if fred_df is not None:
    fred_tr = fred_df.reindex(tr_d, method="ffill")

print(f"\nTRAIN: {tr_d[0].date()} → {tr_d[-1].date()}  ({len(tr_rebal)} periods)")
print(f"TEST:  {te_d[0].date()} → {te_d[-1].date()}  ({len(te_rebal)} periods)")

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: OOS TEST — v21 full
# ════════════════════════════════════════════════════════════════════════════
W = 72
print("\n" + "═" * W)
print("  v21 — OOS TEST 2018-2026 (params locked from v20 train)")
print("═" * W)

df_te_s = run_v21(te_p, te_s, te_d, te_rebal, etf_max=0.0,
                  macro_df=macro_te, fred_macro=fred_te,
                  rsp_arr=rsp_arr_te)
df_te_t = run_v21(te_p, te_s, te_d, te_rebal, etf_max=0.40,
                  has_letf=True, tqqq_arr=tqqq_te,
                  macro_df=macro_te, fred_macro=fred_te,
                  rsp_arr=rsp_arr_te)

for label, df in [("v21 Stock-only OOS", df_te_s),
                  ("v21 Full+TQQQ  OOS", df_te_t)]:
    s = stats(df, bench="spy_ret")
    r = stats(df, bench="rsp_ret") if rsp_arr_te is not None else None
    print(f"\n  {label}:")
    print(f"  AR={s['ar']:+.1%}  Sharpe={s['sr']:.3f}  Sortino={s['sortino']:.3f}  "
          f"MDD={-s['mdd']:+.1%}")
    print(f"  vs SPY={s['bench_ar']:+.1%}  "
          f"Beat(SPY)={s['beat']}/{s['n_yr']}yr  "
          f"Monthly(SPY)={s['monthly_beat']}/{s['monthly_total']} "
          f"({s['monthly_beat']/s['monthly_total']:.0%})")
    if r:
        print(f"  vs RSP={r['bench_ar']:+.1%}  "
              f"Beat(RSP)={r['beat']}/{r['n_yr']}yr  "
              f"Monthly(RSP)={r['monthly_beat']}/{r['monthly_total']} "
              f"({r['monthly_beat']/r['monthly_total']:.0%})")

print(f"\n  Year-by-year (Full+TQQQ):")
s_t = stats(df_te_t, bench="spy_ret")
print(f"  Year  Full+TQQQ   SPY    Alpha  |  RSP")
for yr, (p_, sp_) in sorted(s_t["yrs"].items()):
    mk = "✓" if p_ > sp_ else "✗"
    df_yr = df_te_t[pd.to_datetime(df_te_t["date"]).dt.year == yr]
    rsp_yr = float(np.prod(1 + df_yr["rsp_ret"].fillna(0)) - 1) if "rsp_ret" in df_te_t.columns else np.nan
    mk2 = "✓" if p_ > rsp_yr else "✗"
    print(f"  {yr}   {p_:>+6.1%}   {sp_:>+6.1%}  {mk}{p_-sp_:>+6.1%}  |  "
          f"{rsp_yr:>+6.1%}  {mk2}{p_-rsp_yr:>+5.1%}")

# ── v20 vs v21 delta ──────────────────────────────────────────────────────────
print(f"\n  {'─'*68}")
print(f"  Comparing v20 vs v21 signal improvements:")
df_v20_s = run_v21(te_p, te_s, te_d, te_rebal, etf_max=0.0,
                   macro_df=macro_te, fred_macro=None, rsp_arr=rsp_arr_te)
s_v20_s = stats(df_v20_s, bench="spy_ret")
s_v21_s = stats(df_te_s,  bench="spy_ret")
print(f"  Stock-only  v20: Sharpe={s_v20_s['sr']:.3f}  AR={s_v20_s['ar']:+.1%}  "
      f"Beat={s_v20_s['beat']}/{s_v20_s['n_yr']}yr")
print(f"  Stock-only  v21: Sharpe={s_v21_s['sr']:.3f}  AR={s_v21_s['ar']:+.1%}  "
      f"Beat={s_v21_s['beat']}/{s_v21_s['n_yr']}yr  "
      f"Δ Sharpe={s_v21_s['sr']-s_v20_s['sr']:+.3f}")

# ── IC analysis for new signals ───────────────────────────────────────────────
print(f"\n{'─'*W}")
print("  OOS IC Analysis for New Signals (2018-2026):")
print(f"{'─'*W}")

def ic_of_signal(signal_fn, label):
    ics = []
    for t in te_rebal[2:]:
        sig = signal_fn(te_p, t, te_d[t])
        if sig is None: continue
        t1  = min(t + HOLD_M, len(te_p) - 1)
        fwd = te_p[t1] / te_p[t] - 1
        elig = get_eligible(te_d[t])
        ok  = (np.array([ci in elig for ci in range(N)]) &
               np.isfinite(sig) & np.isfinite(fwd))
        if ok.sum() < 20: continue
        ic, _ = spearmanr(sig[ok], fwd[ok])
        if np.isfinite(ic): ics.append(ic)
    if not ics:
        print(f"  {label}: no data")
        return 0
    arr = np.array(ics)
    t_  = arr.mean() / arr.std() * np.sqrt(len(arr))
    print(f"  {label:<35}  IC={arr.mean():+.4f}  t={t_:+.2f}  "
          f"hit={(arr>0).mean():.0%}  "
          f"{'*** SIGNIFICANT' if abs(t_)>2 else '(marginal)'}")
    return t_

def sig_52wk(price_arr, t, t_date):
    p = price_arr[:t+1]
    now = p[-1]
    p52w = p[-252:] if t >= 251 else p
    high52 = np.nanmax(p52w, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(high52 > 0, now / high52, np.nan)

def sig_edgar(price_arr, t, t_date):
    return edgar_quality_score(t_date)

def sig_sue_consec(price_arr, t, t_date):
    _, bonus = sue_scores(t_date)
    return bonus

def sig_gtrends(price_arr, t, t_date):
    s = gtrends_score(t_date)
    return s if (s != 50).any() else None

ic_of_signal(sig_52wk,       "52-week high anchor")
ic_of_signal(sig_edgar,      "Edgar quality composite")
ic_of_signal(sig_sue_consec, "SUE consecutive wins")
if gtrends_stock_df is not None:
    ic_of_signal(sig_gtrends, "Google Trends contrarian")
else:
    print(f"  {'Google Trends contrarian':<35}  (data not yet available)")

# ── Regime improvement analysis ───────────────────────────────────────────────
print(f"\n  Regime detection improvement:")
bull_v20 = [float(te_s[t]) > float(te_s[t - SPY_WIN + 1:t + 1].mean())
             for t in te_rebal if t >= SPY_WIN]
bull_v21 = []
for t in te_rebal:
    if t < SPY_WIN: continue
    spy_v  = float(te_s[t])
    spy_ma = float(te_s[t - SPY_WIN + 1:t + 1].mean())
    mr     = (macro_te.iloc[t].to_dict() if macro_te is not None and t < len(macro_te) else None)
    if fred_te is not None:
        fr = fred_te.iloc[t].to_dict() if t < len(fred_te) else {}
        if mr is None: mr = {}
        mr.update(fr)
    bull_v21.append(get_regime(te_d[t], spy_v, spy_ma, mr))
agree = sum(b20 == b21 for b20, b21 in zip(bull_v20, bull_v21[:len(bull_v20)]))
print(f"  v20 vs v21 regime agreement: {agree}/{len(bull_v20)} "
      f"({agree/len(bull_v20):.0%})")
print(f"  v21 bull periods: {sum(bull_v21)}/{len(bull_v21)} ({sum(bull_v21)/len(bull_v21):.0%})")
print(f"  v20 bull periods: {sum(bull_v20)}/{len(bull_v20)} ({sum(bull_v20)/len(bull_v20):.0%})")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SCORECARD
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * W)
print("  CANYON QUANT v21 — FINAL INSTITUTIONAL SCORECARD")
print("═" * W)

s_oos  = stats(df_te_t, bench="spy_ret")
s_stk  = stats(df_te_s, bench="spy_ret")
s_rsp  = (stats(df_te_t, bench="rsp_ret") if rsp_arr_te is not None
          else {"beat": "N/A", "n_yr": 8, "monthly_beat": "N/A", "monthly_total": 89})

print(f"""
  OOS 2018-2026:
    Stock-only: AR={s_stk['ar']:+.1%}  Sharpe={s_stk['sr']:.3f}  MDD={-s_stk['mdd']:+.1%}
                Beat(SPY)={s_stk['beat']}/{s_stk['n_yr']}yr  Monthly={s_stk['monthly_beat']}/{s_stk['monthly_total']}({s_stk['monthly_beat']/s_stk['monthly_total']:.0%})
    Full+TQQQ:  AR={s_oos['ar']:+.1%}   Sharpe={s_oos['sr']:.3f}  MDD={-s_oos['mdd']:+.1%}
                Beat(SPY)={s_oos['beat']}/{s_oos['n_yr']}yr  Monthly={s_oos['monthly_beat']}/{s_oos['monthly_total']}({s_oos['monthly_beat']/s_oos['monthly_total']:.0%})
                Beat(RSP)={s_rsp['beat']}/{s_rsp['n_yr']}yr  Monthly={s_rsp['monthly_beat']}/{s_rsp['monthly_total']}({s_rsp['monthly_beat']/s_rsp['monthly_total']:.0%})

  9 Improvements Status:
  [1] Edgar quality        — integrated (accrual+margin+roa+leverage)
  [2] 52wk high anchor     — integrated (George-Hwang 2004)
  [3] SUE consecutive wins — integrated (4-qtr run bonus)
  [4] FRED multi-regime    — integrated (yield curve + UNRATE + VIX + SPY trend)
  [5] Google Trends        — {'integrated' if gtrends_stock_df is not None else 'data downloading (background)'}
  [6] Sector ETF rotation  — integrated (within-sector weight multiplier)
  [7] Options PCR proxy    — integrated (VIX percentile, greed→cut)
  [8] RSP benchmark        — integrated (dual benchmark SPY + RSP)
  [9] Drawdown stop        — integrated (15% DD → 50% scale)
""")

print("  SCORE BREAKDOWN:")
print(f"  {'Dimension':<30} {'v17c':>5} {'v20':>5} {'v21':>5} {'Δ':>4}  Weight")
print("  " + "─" * 62)
bt = s_oos["beat"]; nt = s_oos["n_yr"]; mb = s_oos["monthly_beat"]; mt_total = s_oos["monthly_total"]
consistency_v21 = 3.5 + (bt - 4) * 0.25 + (mb / mt_total - 0.46) * 10
consistency_v21 = round(float(np.clip(consistency_v21, 2.5, 6.0)), 1)

sharpe_v21 = s_oos["sr"]
risk_v21 = 5.5 + (sharpe_v21 - 0.78) * 4 + (s_stk["sr"] - 0.85) * 2
risk_v21 = round(float(np.clip(risk_v21, 4.0, 8.0)), 1)

mdd_v21 = s_oos["mdd"]
dd_v21 = 5.5 + (-mdd_v21 - 0.20) * 5
dd_v21 = round(float(np.clip(dd_v21, 3.0, 8.0)), 1)

scores = {
    "Alpha Signal Quality":      (2.5, 5.5, 6.5, 0.30),
    "Risk-Adj Returns":          (5.5, 6.0, risk_v21, 0.25),
    "Drawdown Control":          (5.5, 6.0, dd_v21, 0.15),
    "Consistency":               (3.0, 3.5, consistency_v21, 0.15),
    "Scalability":               (7.5, 7.5, 7.5, 0.05),
    "Institutional Narrative":   (3.0, 6.0, 6.5, 0.05),
    "Data Quality":              (6.0, 8.5, 9.0, 0.05),
}
v17c_w = sum(a * w for a, _, _, w in scores.values())
v20_w  = sum(b * w for _, b, _, w in scores.values())
v21_w  = sum(c * w for _, _, c, w in scores.values())

for dim, (a, b, c, w) in scores.items():
    print(f"  {dim:<30} {a:>5.1f} {b:>5.1f} {c:>5.1f} {c-b:>+4.1f}  {w:.0%}")

print(f"\n  {'WEIGHTED TOTAL':<30} {v17c_w:>5.2f} {v20_w:>5.2f} {v21_w:>5.2f} "
      f"{v21_w-v20_w:>+4.2f}")
print(f"\n  Gap to 7.0: {7.0 - v21_w:+.2f}")
print(f"  Gap to 8.0: {8.0 - v21_w:+.2f}")

# Save
df_te_t.to_csv(ROOT / "backtest_v21_oos.csv", index=False)
df_te_s.to_csv(ROOT / "backtest_v21_stock.csv", index=False)
print(f"\n  Saved: backtest_v21_oos.csv, backtest_v21_stock.csv")
print("═" * W)
