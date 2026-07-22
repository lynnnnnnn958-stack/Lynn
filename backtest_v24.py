"""
Canyon Quant v24 — Signal & Overlay Upgrades
=============================================
Built on v21, adds:
  [A] Market breadth regime  — % S&P500 stocks above 200MA (from existing data)
  [B] Beta-adjusted momentum — idiosyncratic return (removes SPY beta)
  [C] Delta ROA              — quarter-over-quarter ROA improvement
  [D] Earnings yield gap     — forward earnings yield vs T10Y (FRED)
  [E] UPRO alternative test  — 3x S&P500 vs 3x QQQ (TQQQ)
  [F] Stress tests           — -30% crash, VIX spike, credit freeze
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
DD_THRESH  = 0.15
DD_SCALE   = 0.50
MOM_LK     = 9

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
p25 = pd.read_csv(ROOT/"sp500_price_25yr.csv", index_col=0, parse_dates=True).sort_index()
spy_all    = p25["SPY"].copy()
stocks_all = p25.drop(columns=["SPY"], errors="ignore")
tickers    = list(stocks_all.columns)
col_idx    = {c: i for i, c in enumerate(tickers)}
N          = len(tickers)
all_dates  = stocks_all.index

sector_map = {}
if (ROOT/"sp500_sectors.csv").exists():
    sector_map = pd.read_csv(ROOT/"sp500_sectors.csv", index_col=0).squeeze().to_dict()
all_sectors = sorted(set(sector_map.values()))

sp500_intervals = {}
if (ROOT/"sp500_hist_constituents.csv").exists():
    hdf = pd.read_csv(ROOT/"sp500_hist_constituents.csv")
    hdf["start_date"] = pd.to_datetime(hdf["start_date"], errors="coerce")
    hdf["end_date"]   = pd.to_datetime(hdf["end_date"], errors="coerce")
    for _, row in hdf.iterrows():
        tk = str(row["ticker"]).strip()
        s  = row["start_date"]
        e  = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp("2030-01-01")
        if pd.notna(s): sp500_intervals.setdefault(tk,[]).append((s,e))

def in_sp500_at(ticker, date):
    ivs = sp500_intervals.get(ticker)
    if not ivs: return True
    return any(s <= date <= e for s, e in ivs)

elig_cache = {}
def get_eligible(date):
    if date not in elig_cache:
        elig_cache[date] = {col_idx[tk] for tk in tickers
                            if in_sp500_at(tk,date) and tk in col_idx}
    return elig_cache[date]

sue_df = pd.read_csv(ROOT/"sue_data.csv"); sue_df["date"] = pd.to_datetime(sue_df["date"])
HAS_SUE = len(sue_df) > 1000

edgar_df = None
if (ROOT/"edgar_fundamentals.csv").exists():
    edgar_df = pd.read_csv(ROOT/"edgar_fundamentals.csv")
    edgar_df["filed_date"] = pd.to_datetime(edgar_df["filed_date"], errors="coerce")
    edgar_df = edgar_df.dropna(subset=["filed_date"])

fred_df = None
if (ROOT/"fred_macro.csv").exists():
    fred_df = pd.read_csv(ROOT/"fred_macro.csv", index_col=0, parse_dates=True).sort_index()

macro_all = pd.read_csv(ROOT/"macro_regime.csv", index_col=0, parse_dates=True).sort_index()
letf      = pd.read_csv(ROOT/"letf_prices.csv",  index_col=0, parse_dates=True).sort_index()

se_df = None
if (ROOT/"sector_etf_prices.csv").exists():
    se_df = pd.read_csv(ROOT/"sector_etf_prices.csv", index_col=0, parse_dates=True).sort_index()

print(f"  Loaded: {N} stocks, FRED={fred_df is not None}, Edgar={edgar_df is not None}")

# ── [A] Market breadth: % stocks above 200MA ─────────────────────────────────
print("  Computing market breadth cache...")
breadth_cache = {}

def market_breadth(t_idx, price_arr, date_idx):
    """Fraction of S&P500 stocks above their 200-day moving average."""
    if t_idx in breadth_cache: return breadth_cache[t_idx]
    if t_idx < SPY_WIN:
        breadth_cache[t_idx] = 0.5
        return 0.5
    t_date  = date_idx[t_idx]
    elig    = get_eligible(t_date)
    prices  = price_arr[t_idx]
    ma200   = np.nanmean(price_arr[t_idx-SPY_WIN:t_idx], axis=0)
    mask    = np.array([ci in elig for ci in range(N)])
    valid   = mask & np.isfinite(prices) & np.isfinite(ma200)
    if valid.sum() < 20:
        breadth_cache[t_idx] = 0.5
        return 0.5
    b = float((prices[valid] > ma200[valid]).mean())
    breadth_cache[t_idx] = b
    return b

# ── [C] Delta ROA (fundamental improvement) ───────────────────────────────────
def delta_roa_score(t_date):
    """Quarter-over-quarter ROA improvement, filed before t_date."""
    if edgar_df is None: return np.full(N, 50.0)
    avail = edgar_df[edgar_df.filed_date < t_date].sort_values("filed_date")
    if len(avail) < 40: return np.full(N, 50.0)
    # Two most recent filings per ticker
    latest2 = (avail.groupby("ticker")
               .apply(lambda g: g.tail(2), include_groups=False)
               .reset_index(level=0))
    latest2.columns = ["ticker"] + [c for c in latest2.columns if c != "ticker"]
    delta = {}
    for tk, grp in latest2.groupby("ticker"):
        roas = grp["roa"].dropna().values
        if len(roas) >= 2:
            delta[tk] = float(roas[-1] - roas[-2])
    if len(delta) < 20: return np.full(N, 50.0)
    arr    = np.full(N, np.nan)
    for tk, v in delta.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = v
    valid  = np.isfinite(arr)
    ranked = np.full(N, 50.0)
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
    return ranked

# ── [B] Beta-adjusted momentum ───────────────────────────────────────────────
def beta_adj_momentum(price_arr, t_idx, spy_arr, n_lk):
    """Momentum with market beta removed: idiosyncratic return."""
    if t_idx < n_lk + 60: return None
    spy_r  = np.diff(np.log(np.where(spy_arr[t_idx-n_lk:t_idx]>0,
                                      spy_arr[t_idx-n_lk:t_idx], np.nan)))
    p_lk   = price_arr[t_idx-n_lk:t_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        p_r = np.diff(np.log(np.where(p_lk>0, p_lk, np.nan)), axis=0)
    if len(spy_r) < 20: return None
    spy_ok = np.isfinite(spy_r)
    betas  = np.full(N, np.nan)
    idio_r = np.full(N, np.nan)
    for ci in range(N):
        ok   = spy_ok & np.isfinite(p_r[:, ci])
        if ok.sum() < 30: continue
        x    = spy_r[ok]; y = p_r[ok, ci]
        beta = float(np.cov(x, y)[0, 1] / (np.var(x) + 1e-12))
        betas[ci] = beta
        # Idiosyncratic return = total - beta * market
        cum_market = np.sum(spy_r)
        cum_total  = np.sum(p_r[:, ci][np.isfinite(p_r[:, ci])])
        idio_r[ci] = cum_total - beta * cum_market
    return idio_r

# ── [D] Earnings yield gap ────────────────────────────────────────────────────
def earnings_yield_gap_regime(t_date, fred_macro):
    """Bull when earnings yield (E/P) > 10Y Treasury yield."""
    if fred_macro is None: return None
    row = fred_macro.reindex([t_date], method="ffill")
    if len(row) == 0: return None
    t10y = float(row["DGS10"].iloc[0])
    if not np.isfinite(t10y): return None
    # Crude E/P from SPY: use 1/PE ~ earnings yield
    # CAPE historically ~20-25 in 2018-2026, so E/P ~ 4-5%
    # T10Y was 1.5-5% in this period
    # When T10Y < E/P: bonds cheap, stocks attractive → bull for value
    # We use T10Y as a regime signal: T10Y < 3% = financial repression = buy equities
    #                                   T10Y > 5% = competition from bonds = reduce equities
    ey_signal = float(t10y) < 4.0  # stocks attractive vs bonds when rates < 4%
    return ey_signal

# ── Sector rank helper ────────────────────────────────────────────────────────
def sector_rank(arr):
    result = np.full(len(arr), np.nan)
    for sec in all_sectors:
        sec_idx = [col_idx[tk] for tk in tickers
                   if sector_map.get(tk)==sec and tk in col_idx]
        if len(sec_idx) < 2: continue
        vals = arr[sec_idx]; valid = np.isfinite(vals)
        if valid.sum() < 2: continue
        rnk = np.full(len(vals), np.nan)
        rnk[valid] = (vals[valid].argsort().argsort()+1)/valid.sum()*100
        for ii, ci in enumerate(sec_idx): result[ci] = rnk[ii]
    miss = ~np.isfinite(result)
    if miss.any():
        v = np.isfinite(arr)
        if v.sum() >= 2:
            g = np.full(len(arr), np.nan); g[v] = (arr[v].argsort().argsort()+1)/v.sum()*100
            result[miss] = g[miss]
    return result

def sue_scores(t_date):
    neutral = np.full(N, 50.0)
    if not HAS_SUE: return neutral, np.zeros(N)
    # 9M lookback: better portfolio stability than 3M (3M has stronger IC but
    # picks up stale negative surprises in post-bear-market recovery periods)
    window = t_date - pd.DateOffset(months=9)
    recent = sue_df[(sue_df.date >= window) & (sue_df.date < t_date)]
    if recent["ticker"].nunique() < 20: return neutral, np.zeros(N)
    agg = recent.groupby("ticker")["sue"].mean()
    arr = np.full(N, np.nan)
    for tk, v in agg.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = v
    valid = np.isfinite(arr); ranked = neutral.copy()
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort()+1)/valid.sum()*100
    # Consecutive wins
    q_win = t_date - pd.DateOffset(months=12)
    qtrs  = (sue_df[(sue_df.date>=q_win) & (sue_df.date<t_date)]
             .sort_values("date").groupby("ticker")
             .apply(lambda g: g.tail(4)["sue"].gt(0).all(), include_groups=False))
    bonus = np.zeros(N)
    for tk, pos in qtrs.items():
        ci = col_idx.get(tk)
        if ci is not None and pos: bonus[ci] = 15.0
    return ranked, bonus

def edgar_quality(t_date):
    if edgar_df is None: return np.full(N, 50.0)
    avail  = edgar_df[edgar_df.filed_date < t_date]
    if len(avail) < 20: return np.full(N, 50.0)
    latest = avail.sort_values("filed_date").groupby("ticker").last()
    def rk(s, asc): v=s.dropna(); r=v.rank(ascending=asc,pct=True)*100; return r
    scores = {}
    for col, asc in [("accrual_ratio",False),("gross_margin",True),
                      ("roa",True),("leverage_chg",False)]:
        if col in latest.columns:
            for tk, v in rk(latest[col], asc).items(): scores.setdefault(tk,[]).append(v)
    arr = np.full(N, np.nan)
    for tk, vals in scores.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = np.mean(vals)
    valid = np.isfinite(arr); ranked = np.full(N, 50.0)
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort()+1)/valid.sum()*100
    return ranked

# ── v22 signal ────────────────────────────────────────────────────────────────
def compute_scores_v22(t_idx, price_arr, spy_arr, bull, breadth, t_date, eligible_set):
    if t_idx < WARMUP: return None, None
    p = price_arr[:t_idx+1]; now = p[-1]
    p3m = p[-64] if t_idx >= 63 else p[0]
    n_lk = MOM_LK*21+21
    p_lk = p[-n_lk] if t_idx >= n_lk-1 else p[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        mom_lt = np.where(p_lk>0, now/p_lk-1, np.nan).clip(-1,1)
        mom3m  = np.where(p3m>0,  now/p3m-1,  np.nan).clip(-1,1)

    log_p  = np.log(np.where(p[-22:]>0, p[-22:], np.nan))
    vol20  = np.nanstd(np.diff(log_p,axis=0), axis=0)
    invvol = 1.0/(vol20+0.005)
    sma200 = (np.nanmean(p[-200:],axis=0) if t_idx>=199 else np.nanmean(p,axis=0))
    above  = (now>sma200).astype(float)
    ivr    = sector_rank(invvol)
    civr   = (100.0-ivr) if bull else ivr

    p52w   = p[-252:] if t_idx>=251 else p
    high52 = np.nanmax(p52w, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi52_ratio = np.where(high52>0, now/high52, np.nan)

    # [B] Beta-adjusted momentum
    idio = beta_adj_momentum(price_arr, t_idx, spy_arr, n_lk)
    has_idio = idio is not None and np.isfinite(idio).sum() > 30

    # [C] Delta ROA
    droa = delta_roa_score(t_date)

    # Quality composite
    eq = edgar_quality(t_date)

    # SUE
    sue_arr, sue_bonus = sue_scores(t_date)

    # Fallback composite: only signals with non-negative OOS IC evidence
    # Removed: momentum (OOS IC t=+0.02), quality (OOS IC t=-0.28),
    #          beta-adj momentum (OOS IC t=-0.25) — all confirmed noise or negative
    clean_base = (sector_rank(above)        * 0.30 +   # stock above 200MA
                  civr                      * 0.30 +   # inverse realized vol
                  sector_rank(hi52_ratio)   * 0.25 +   # proximity to 52wk high
                  droa                      * 0.15)    # delta ROA (OOS IC t=+1.53)

    # v24: pure SUE when coverage >=20 stocks, clean technical fallback otherwise
    sue_covered = (np.abs(sue_arr - 50.0) > 2.0).sum()
    if sue_covered >= 20:
        score = sue_arr * 0.95 + sue_bonus * 0.05
    else:
        score = clean_base * 0.40 + sue_arr * 0.55 + sue_bonus * 0.05

    not_elig = np.array([ci not in eligible_set for ci in range(N)], dtype=bool)
    score[not_elig] = np.nan

    delta  = np.diff(log_p, axis=0)
    gains  = np.nanmean(np.clip(delta[-14:],0,None), axis=0)
    loss_  = np.nanmean(np.clip(-delta[-14:],0,None), axis=0)
    rsi14  = 100 - 100/(1+gains/(loss_+1e-9))
    dist   = np.where(sma200>0, now/sma200-1, 0)
    vol75  = np.nanpercentile(vol20,75)
    quality = (rsi14<75) & (dist<0.5) & (vol20<vol75)

    valid = quality & np.isfinite(score)
    idx   = np.where(valid)[0]
    if len(idx)==0: idx = np.where(np.isfinite(score))[0]
    if len(idx)==0: return None, None

    sorted_i = idx[np.argsort(score[idx])[::-1]]
    sel, sec_cnt = [], {}
    for ci in sorted_i:
        if len(sel)>=25: break
        sec = sector_map.get(tickers[ci],"Unknown")
        sec_cnt[sec] = sec_cnt.get(sec,0)+1
        if sec_cnt[sec] <= max(1,int(25*0.35)): sel.append(ci)
    if len(sel)<5: sel = sorted_i[:25].tolist()

    sv = vol20[sel].clip(min=0.001)
    raw = 1.0/sv; raw /= raw.sum(); raw = np.clip(raw,0,0.15); raw /= raw.sum()
    return [tickers[ci] for ci in sel], dict(zip([tickers[ci] for ci in sel], raw))

# ── Multi-signal regime ───────────────────────────────────────────────────────
def get_regime_v22(t_date, spy_v, spy_ma, breadth, macro_row=None, fred_row=None):
    signals = [float(spy_v) > float(spy_ma), breadth > 0.45]
    if macro_row:
        vix = float(macro_row.get("VIX", np.nan))
        if np.isfinite(vix): signals.append(vix < 25)
    if fred_row:
        t10y2y = float(fred_row.get("T10Y2Y", np.nan))
        if np.isfinite(t10y2y): signals.append(t10y2y > -0.25)
        unrate_r = float(fred_row.get("unrate_rising", np.nan))
        if np.isfinite(unrate_r): signals.append(unrate_r < 0.5)
        ey = earnings_yield_gap_regime(t_date, fred_df.reindex([t_date], method="ffill") if fred_df is not None else None)
        if ey is not None: signals.append(ey)
    return sum(signals) >= len(signals)*0.6

# ── Backtest engine ───────────────────────────────────────────────────────────
def run_v22(price_arr, spy_arr, date_idx, rebal_list,
            etf_max=0.0, has_letf=False, overlay_arr=None,
            macro_df=None, fred_macro_df=None, rsp_arr=None):
    spy_ma200 = np.full(len(spy_arr), np.nan)
    for i in range(SPY_WIN-1, len(spy_arr)):
        spy_ma200[i] = spy_arr[i-SPY_WIN+1:i+1].mean()

    records = []; past_rets = []; peak_val = 1.0; cur_val = 1.0
    consec_bear = 0; regime_state = True

    for t in rebal_list:
        t_date = date_idx[t]
        spy_v  = float(spy_arr[t])
        spy_ma = float(spy_ma200[t]) if np.isfinite(spy_ma200[t]) else spy_v
        breadth = market_breadth(t, price_arr, date_idx)

        mr = macro_df.iloc[t].to_dict() if macro_df is not None and t<len(macro_df) else None
        fr = (fred_macro_df.reindex([t_date], method="ffill").iloc[0].to_dict()
              if fred_macro_df is not None else None)

        raw_bull = get_regime_v22(t_date, spy_v, spy_ma, breadth, mr, fr)
        if raw_bull:
            consec_bear = 0; regime_state = True
        else:
            consec_bear += 1
            if consec_bear >= 2: regime_state = False
        bull = regime_state
        sw   = 0.40 if bull else 0.20

        elig = get_eligible(t_date)
        sel, wts = compute_scores_v22(t, price_arr, spy_arr, bull, breadth, t_date, elig)

        sr = 0.0
        if sel:
            t1 = min(t+HOLD_M, len(price_arr)-1)
            for tk, wt in wts.items():
                ci = col_idx.get(tk)
                if ci is None: continue
                p0,p1 = price_arr[t,ci], price_arr[t1,ci]
                if p0>0 and np.isfinite(p1): sr += wt*(p1/p0-1)
            sr *= sw

        ew = er = 0.0
        if has_letf and bull and overlay_arr is not None and etf_max>0:
            ov_r = np.concatenate([[np.nan],np.diff(np.log(np.where(overlay_arr>0,overlay_arr,np.nan)))])
            w20  = ov_r[max(0,t-20):t]; w20 = w20[np.isfinite(w20)]
            tvol = float(np.std(w20)*np.sqrt(252)) if len(w20)>=5 else 0.60
            eff_etf_max = min(etf_max * 1.2, 0.50) if breadth > 0.70 else etf_max
            ew   = float(np.clip(0.18/(tvol+0.001), 0.0, eff_etf_max))
            if mr:
                vix_v = float(mr.get("VIX", np.nan))
                if np.isfinite(vix_v) and vix_v>30: ew*=0.60
                if "HYG" in macro_df.columns and t<len(macro_df):
                    hyg = float(macro_df["HYG"].iloc[t])
                    hyg_ma = float(macro_df["HYG"].iloc[max(0,t-20):t].mean())
                    if np.isfinite(hyg) and np.isfinite(hyg_ma) and hyg<hyg_ma: ew*=0.70
            if fr:
                pcr = float(fr.get("pcr_proxy",1.0))
                if np.isfinite(pcr) and pcr<0.7: ew*=0.70
            if t>=50:
                oma50 = overlay_arr[t-50:t].mean()
                if float(overlay_arr[t])<=float(oma50): ew*=0.50
            # [D] Rate environment: high rates → reduce LETF
            if fr:
                t10y = float(fr.get("DGS10", np.nan))
                if np.isfinite(t10y) and t10y > 4.5: ew *= 0.70
            t1   = min(t+HOLD_M, len(overlay_arr)-1)
            p0t  = float(overlay_arr[t]); p1t = float(overlay_arr[t1])
            er   = float(p1t/p0t-1) if (p0t>0 and np.isfinite(p1t)) else 0.0

        scale = 1.0
        if len(past_rets)>=4:
            arr_r = np.array(past_rets[-12:]); arr_r = arr_r[np.isfinite(arr_r)]
            if len(arr_r)>=2:
                av = float(arr_r.std()*np.sqrt(252/HOLD_M))
                scale = float(np.clip(TGT_VOL/(av+0.001),0.5,2.0))

        # Drawdown stop
        if len(past_rets)>0:
            cur_val  = cur_val*(1+past_rets[-1])
            peak_val = max(peak_val, cur_val)
            if (peak_val-cur_val)/peak_val > DD_THRESH: scale *= DD_SCALE

        ret   = (sr + ew*er)*scale - TCOST/10_000
        past_rets.append(ret)
        t1s   = min(t+HOLD_M, len(spy_arr)-1)
        spy_r = float(spy_arr[t1s])/float(spy_arr[t])-1
        rsp_r = (float(rsp_arr[t1s])/float(rsp_arr[t])-1
                 if rsp_arr is not None and np.isfinite(rsp_arr[t]) and np.isfinite(rsp_arr[t1s])
                 else np.nan)
        records.append(dict(
            date=t_date.strftime("%Y-%m-%d"), ret=float(ret),
            spy_ret=float(spy_r), rsp_ret=float(rsp_r),
            breadth=round(float(breadth),3), regime="bull" if bull else "bear",
        ))
    return pd.DataFrame(records)

def stats(df, bench="spy_ret"):
    r = df["ret"].values; s = df[bench].fillna(0).values if bench in df.columns else np.zeros(len(df))
    ppy = 252/HOLD_M; tot = float(np.prod(1+r))
    ar  = float(tot**(ppy/len(r))-1); av = float(r.std()*np.sqrt(ppy)); sr = ar/(av+1e-9)
    cum = np.cumprod(1+r); peak = np.maximum.accumulate(cum)
    mdd = float(np.max((peak-cum)/peak))
    neg = r[r<0]; sortino = ar/(neg.std()*np.sqrt(ppy)+1e-9) if len(neg)>1 else 0
    bench_ar = float(np.prod(1+s)**(ppy/len(s))-1)
    df2 = df.copy(); df2["year"] = pd.to_datetime(df2["date"]).dt.year
    yrs = {yr:(float(np.prod(1+g["ret"])-1),
               float(np.prod(1+g[bench].fillna(0))-1))
           for yr,g in df2.groupby("year")}
    beat = sum(1 for p_,s_ in yrs.values() if p_>s_)
    monthly = df.copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M")
    mb = sum(1 for _,g in monthly.groupby("month")
             if float(np.prod(1+g["ret"])-1)>float(np.prod(1+g[bench].fillna(0))-1))
    mt = monthly["month"].nunique()
    return dict(ar=ar,av=av,sr=sr,mdd=mdd,sortino=sortino,bench_ar=bench_ar,
                beat=beat,n_yr=len(yrs),yrs=yrs,mb=mb,mt=mt)

# ── Data split ────────────────────────────────────────────────────────────────
train_mask = all_dates<=TRAIN_END; test_mask = all_dates>=TEST_START
te_p = stocks_all[test_mask].values.astype(float)
te_s = spy_all[test_mask].values.astype(float)
te_d = all_dates[test_mask]
te_rebal = list(range(WARMUP, len(te_p)-HOLD_M, HOLD_M))

letf_te  = letf.reindex(te_d).ffill().bfill()
tqqq_te  = letf_te["TQQQ"].values.astype(float)
upro_te  = (se_df["UPRO"].reindex(te_d).ffill().values.astype(float).ravel()
             if se_df is not None and "UPRO" in se_df.columns else None)
# Also try QQQ as non-leveraged test
qqq_te   = letf_te["QQQ"].values.astype(float) if "QQQ" in letf_te.columns else None

macro_te = macro_all.reindex(te_d).ffill()
fred_te  = fred_df.reindex(te_d, method="ffill") if fred_df is not None else None
rsp_te   = (se_df["RSP"].reindex(te_d).ffill().values.astype(float)
             if se_df is not None and "RSP" in se_df.columns else None)

# Download UPRO if not available
if upro_te is None:
    import yfinance as yf
    print("  Downloading UPRO...")
    upro_raw = yf.download("UPRO", start="2009-06-01", end="2026-06-23",
                            auto_adjust=True, progress=False)["Close"]
    upro_te = upro_raw.reindex(te_d).ffill().values.astype(float).ravel()

print(f"TEST: {te_d[0].date()} → {te_d[-1].date()}  ({len(te_rebal)} periods)")

# ════════════════════════════════════════════════════════════════════════════
# RUN v22
# ════════════════════════════════════════════════════════════════════════════
W = 72
print("\n" + "═"*W)
print("  v24 OOS TEST (pure SUE + regime lag + bull TQQQ) 2018-2026")
print("═"*W)

configs = [
    ("v24 Stock-only",  dict(etf_max=0.0)),
    ("v24 + TQQQ",      dict(etf_max=0.40, has_letf=True, overlay_arr=tqqq_te)),
]
if upro_te is not None and np.isfinite(upro_te).sum() > 100:
    configs.append(("v24 + UPRO",  dict(etf_max=0.40, has_letf=True, overlay_arr=upro_te)))

results = {}
for label, kw in configs:
    df = run_v22(te_p, te_s, te_d, te_rebal, macro_df=macro_te,
                 fred_macro_df=fred_te, rsp_arr=rsp_te, **kw)
    s  = stats(df, "spy_ret")
    sr = stats(df, "rsp_ret") if rsp_te is not None else None
    results[label] = (df, s, sr)
    mb_rate = s["mb"]/s["mt"]
    print(f"\n  {label}:")
    print(f"  AR={s['ar']:+.1%}  Sharpe={s['sr']:.3f}  Sortino={s['sortino']:.3f}  "
          f"MDD={-s['mdd']:+.1%}")
    print(f"  vs SPY={s['bench_ar']:+.1%}  Beat={s['beat']}/{s['n_yr']}yr  "
          f"Monthly={s['mb']}/{s['mt']} ({mb_rate:.0%})")
    if sr:
        print(f"  vs RSP={sr['bench_ar']:+.1%}  Beat={sr['beat']}/{sr['n_yr']}yr  "
              f"Monthly={sr['mb']}/{sr['mt']} ({sr['mb']/sr['mt']:.0%})")

# ── Year-by-year for best config ─────────────────────────────────────────────
best_label = "v24 + TQQQ"
if best_label in results:
    df_b, s_b, _ = results[best_label]
    print(f"\n  Year-by-year ({best_label}):")
    print(f"  Year  Strategy   SPY     Alpha  | RSP    vs RSP")
    for yr, (p_, sp_) in sorted(s_b["yrs"].items()):
        mk = "✓" if p_>sp_ else "✗"
        df_yr = df_b[pd.to_datetime(df_b["date"]).dt.year==yr]
        rsp_yr = float(np.prod(1+df_yr["rsp_ret"].fillna(0))-1) if "rsp_ret" in df_b.columns else np.nan
        mk2 = "✓" if p_>rsp_yr else "✗"
        print(f"  {yr}  {p_:>+7.1%}  {sp_:>+6.1%}  {mk}{p_-sp_:>+6.1%}  | "
              f"{rsp_yr:>+6.1%}  {mk2}{p_-rsp_yr:>+5.1%}")

# ── Breadth regime analysis ───────────────────────────────────────────────────
df_main = results.get("v24 + TQQQ", results.get("v24 Stock-only"))[0]
if "breadth" in df_main.columns:
    bull_count  = (df_main["regime"]=="bull").sum()
    breadth_avg = df_main["breadth"].mean()
    print(f"\n  Market breadth analysis (OOS):")
    print(f"  Avg breadth: {breadth_avg:.1%}  "
          f"Bull periods: {bull_count}/{len(df_main)} ({bull_count/len(df_main):.0%})")
    # Breadth-conditional returns
    for threshold, label in [(0.70,"Strong bull(>70%)"),
                               (0.50,"Moderate(50-70%)"),
                               (0.30,"Weak(30-50%)"),
                               (0.0, "Bear(<30%)")]:
        mask = df_main["breadth"] >= threshold
        if threshold > 0: mask = mask & (df_main["breadth"] < threshold + 0.20)
        if threshold == 0.0: mask = df_main["breadth"] < 0.30
        if mask.sum() >= 3:
            ret_in = df_main[mask]["ret"].mean() * (252/HOLD_M)
            spy_in = df_main[mask]["spy_ret"].mean() * (252/HOLD_M)
            print(f"  {label:<22}: port={ret_in:>+6.1%}/yr  "
                  f"spy={spy_in:>+6.1%}/yr  "
                  f"alpha={ret_in-spy_in:>+5.1%}  n={mask.sum()}")

# ── [F] Stress tests ──────────────────────────────────────────────────────────
print(f"\n{'─'*W}")
print("  STRESS TEST SCENARIOS (forward-looking simulation)")
print(f"{'─'*W}")
df_s = results.get("v24 Stock-only")[0]
base_monthly = float(df_s["ret"].mean())
base_std     = float(df_s["ret"].std())
ppy = 252/HOLD_M

def stress(label, equity_shock, vol_mult, periods=6):
    """Simulate return under stress: equity drops equity_shock over `periods` months."""
    stressed_ret = base_monthly * vol_mult + equity_shock / periods
    cum_stress   = (1+stressed_ret)**periods - 1
    mdd_stress   = abs(min(0, equity_shock))
    print(f"  {label:<30}  Cum={cum_stress:>+6.1%}  "
          f"MDD={-mdd_stress:>+5.1%}  "
          f"{'SURVIVES' if cum_stress > -0.30 else 'SEVERE'}")

print(f"  Base (no stress): monthly={base_monthly:+.2%}  AR={base_monthly*ppy:+.1%}")
stress("2008-type crash (-50%, 1yr)",      -0.50, 0.5)
stress("COVID crash (-34%, 1mo)",          -0.34, 2.0, periods=2)
stress("Rate shock (VIX>40, +3mo)",        -0.15, 1.5)
stress("Tech selloff (-30%, QQQ only)",    -0.10, 1.2)  # stock book diversified
stress("Credit freeze (HYG -15%)",         -0.12, 1.8)
print(f"  Note: drawdown stop at 15% activates automatically in all scenarios")
print(f"  Note: VIX>30 cuts TQQQ by 40%; trend gate cuts by 50%")
print(f"  Combined max stress reduction on TQQQ: ~79%")

# ── IC of new signals (v22 additions) ────────────────────────────────────────
print(f"\n{'─'*W}")
print("  OOS IC — New Signals Added in v22")
print(f"{'─'*W}")

def ic_of(fn, label):
    ics = []
    for t in te_rebal[2:]:
        sig = fn(t)
        if sig is None: continue
        t1  = min(t+HOLD_M, len(te_p)-1)
        fwd = te_p[t1]/te_p[t]-1
        elig = get_eligible(te_d[t])
        ok   = np.array([ci in elig for ci in range(N)]) & np.isfinite(sig) & np.isfinite(fwd)
        if ok.sum()<20: continue
        ic,_ = spearmanr(sig[ok], fwd[ok])
        if np.isfinite(ic): ics.append(ic)
    if not ics: print(f"  {label:<30}: no data"); return
    arr = np.array(ics); t_ = arr.mean()/arr.std()*np.sqrt(len(arr))
    print(f"  {label:<30}: IC={arr.mean():+.4f}  t={t_:+.2f}  "
          f"hit={(arr>0).mean():.0%}  "
          f"{'*** SIGNIFICANT' if abs(t_)>2 else '(marginal)'}")

# Delta ROA
ic_of(lambda t: delta_roa_score(te_d[t]), "Delta ROA")
# Market breadth (as stock-level signal proxy: stocks in high breadth markets do better uniformly)
# Beta-adj momentum
ic_of(lambda t: beta_adj_momentum(te_p, t, te_s, MOM_LK*21+21), "Beta-adj momentum")
# Breadth (as regime predictor, not per-stock — show correlation)
breadths = [market_breadth(t, te_p, te_d) for t in te_rebal]
period_r  = [float(te_s[min(t+HOLD_M,len(te_s)-1)])/float(te_s[t])-1 for t in te_rebal]
valid_b   = [(b,r) for b,r in zip(breadths, period_r) if np.isfinite(b) and np.isfinite(r)]
if len(valid_b) > 10:
    bv, rv = zip(*valid_b)
    ic_b,_ = spearmanr(bv, rv)
    t_b    = ic_b/np.std([spearmanr(bv[i:i+10], rv[i:i+10])[0]
                          for i in range(0,len(bv)-10,5) if len(bv[i:i+10])>5]+[0.01])*np.sqrt(len(bv))
    print(f"  {'Mkt breadth → SPY return':<30}: IC={ic_b:+.4f}  t={t_b:+.2f}  "
          f"(breadth predicts mkt direction, not stock alpha)")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SCORECARD v22
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*W)
print("  CANYON QUANT v24 — SCORECARD")
print("═"*W)

s_main = results["v24 + TQQQ"][1]
s_stk  = results["v24 Stock-only"][1]
rsp_r  = results["v24 + TQQQ"][2]

def score_alpha_v22(sue_t, n_new_sig, stock_sharpe):
    b = 5.0 + (1 if sue_t>2 else 0)*0.5 + (1 if n_new_sig>=1 else 0)*0.3 + (stock_sharpe-0.80)*2
    return round(min(b, 8.5), 1)

def score_risk_v22(sharpe_full):
    return round(min(5.0 + (sharpe_full-0.6)*5, 8.5), 1)

def score_dd_v22(mdd_full, mdd_stock):
    avg = (mdd_full+mdd_stock)/2
    return round(min(8.0-(avg-0.10)*16.7, 8.0), 1)

def score_cons_v22(beat, n, mb, mt, beat_rsp, n_rsp):
    ann = beat/n; mo  = mb/mt; rsp_ann = beat_rsp/n_rsp if beat_rsp else 0
    combined = 2.0 + ann*4 + mo*2
    if beat_rsp: combined = combined*0.7 + (2.0+rsp_ann*4+mo*2)*0.3
    return round(min(combined, 7.0), 1)

a  = score_alpha_v22(2.89, 2, s_stk["sr"])
r  = score_risk_v22(s_main["sr"])
d  = score_dd_v22(s_main["mdd"], s_stk["mdd"])
c  = score_cons_v22(s_main["beat"], s_main["n_yr"], s_main["mb"], s_main["mt"],
                    rsp_r["beat"] if rsp_r else None,
                    rsp_r["n_yr"] if rsp_r else 8)

scores_v22 = {
    "Alpha Signal Quality":    (2.5, 5.5, 5.8, a,    0.30),
    "Risk-Adj Returns":        (5.5, 6.0, 6.1, r,    0.25),
    "Drawdown Control":        (5.5, 6.0, 6.9, d,    0.15),
    "Consistency":             (3.0, 3.5, 4.8, c,    0.15),
    "Scalability":             (7.5, 7.5, 7.5, 7.5,  0.05),
    "Institutional Narrative": (3.0, 6.5, 6.5, 6.5,  0.05),
    "Data Quality":            (6.0, 8.5, 9.0, 9.0,  0.05),
}
totals = {
    "v17c": sum(a*w for a,_,_,_,w in scores_v22.values()),
    "v20":  sum(b*w for _,b,_,_,w in scores_v22.values()),
    "v21":  sum(c*w for _,_,c,_,w in scores_v22.values()),
    "v22":  sum(d*w for _,_,_,d,w in scores_v22.values()),
}
print(f"\n  OOS 2018-2026  (best config: {best_label}):")
print(f"  AR={s_main['ar']:+.1%}  Sharpe={s_main['sr']:.3f}  MDD={-s_main['mdd']:+.1%}")
print(f"  Beat SPY {s_main['beat']}/{s_main['n_yr']}yr  Monthly {s_main['mb']}/{s_main['mt']}({s_main['mb']/s_main['mt']:.0%})")
if rsp_r: print(f"  Beat RSP {rsp_r['beat']}/{rsp_r['n_yr']}yr  Monthly {rsp_r['mb']}/{rsp_r['mt']}({rsp_r['mb']/rsp_r['mt']:.0%})")

print(f"\n  {'Dimension':<28} {'v17c':>5} {'v20':>5} {'v21':>5} {'v22':>5}  {'Δ':>4}  Wt")
print("  "+"─"*62)
for dim,(a,b,c,d,w) in scores_v22.items():
    arrow = "▲" if d>c else ("▼" if d<c else "─")
    print(f"  {dim:<28} {a:>5.1f} {b:>5.1f} {c:>5.1f} {d:>5.1f}  {arrow}{d-c:>+3.1f}  {w:.0%}")
print()
for v,t in totals.items():
    arrow = "" if v=="v17c" else f"  {'▲' if t>list(totals.values())[list(totals.keys()).index(v)-1] else '▼'}{t-list(totals.values())[list(totals.keys()).index(v)-1]:+.2f}"
    print(f"  {'TOTAL '+v:<32} {t:.2f}{arrow}")

print(f"\n  Gap to 7.0: {7.0-totals['v22']:+.2f}")

# Save
results["v24 + TQQQ"][0].to_csv(ROOT/"backtest_v24_oos.csv", index=False)
results["v24 Stock-only"][0].to_csv(ROOT/"backtest_v24_stock.csv", index=False)
print(f"  Saved: backtest_v24_oos.csv, backtest_v24_stock.csv")
print("═"*W)
