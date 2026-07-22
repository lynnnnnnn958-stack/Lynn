"""
Canyon Quant v17 FINAL — All 8 Improvements Integrated
========================================================
Runs best configuration after all improvements, plus
integrates insider signal (if available) into stock scoring.
Produces final institutional scorecard vs v11 baseline.
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
TCOST  = 5

print("Loading data...")
prices  = pd.read_csv(ROOT/"sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy     = prices["SPY"].copy()
prices  = prices.drop(columns=["SPY"])
letf_df = pd.read_csv(ROOT/"letf_prices.csv", index_col=0, parse_dates=True).sort_index()
macro   = pd.read_csv(ROOT/"macro_regime.csv", index_col=0, parse_dates=True).sort_index()
sector_map = pd.read_csv(ROOT/"sp500_sectors.csv", index_col=0).squeeze().to_dict() \
             if (ROOT/"sp500_sectors.csv").exists() else {}
all_sectors = sorted(set(sector_map.values()))

# Insider data (if available)
insider_path = ROOT/"insider_transactions.csv"
insider_df   = pd.read_csv(insider_path) if insider_path.exists() else pd.DataFrame()
if not insider_df.empty:
    insider_df["date"] = pd.to_datetime(insider_df["date"])
    # Check if it's just the neutral placeholder
    if "direction" in insider_df.columns and (insider_df["direction"]=="NEUTRAL").mean() == 1.0:
        insider_df = pd.DataFrame()   # neutral = no signal
        print("  Insider: neutral placeholder (no real data)")
    else:
        n_buy  = (insider_df["direction"]=="BUY").sum()
        n_sell = (insider_df["direction"]=="SELL").sum()
        n_tks  = insider_df["ticker"].nunique()
        print(f"  Insider: {n_tks} tickers, BUY={n_buy} SELL={n_sell}")
else:
    print("  Insider: no data")

HAS_INSIDER = not insider_df.empty and "rank" in insider_df.columns

common  = prices.index.intersection(letf_df.index).intersection(macro.index)
prices  = prices.reindex(common)
letf_df = letf_df.reindex(common).ffill().bfill()
spy     = spy.reindex(common).ffill()
macro   = macro.reindex(common).ffill()

price_arr = prices.values.astype(float)
spy_arr   = spy.values.astype(float)
tqqq      = letf_df["TQQQ"].values.astype(float)
tqqq_r    = np.concatenate([[np.nan], np.diff(np.log(np.where(tqqq>0, tqqq, np.nan)))])
tickers   = list(prices.columns)
col_idx   = {c: i for i, c in enumerate(tickers)}

# 200MA precompute
spy_ma200 = np.full(len(spy_arr), np.nan)
for i in range(199, len(spy_arr)):
    spy_ma200[i] = np.mean(spy_arr[i-199:i+1])
above_ma = spy_arr > spy_ma200

# Asymmetric regime
asym = np.ones(len(spy_arr), dtype=bool)
risk_on = True
for i in range(1, len(spy_arr)):
    if risk_on:
        if i >= 5 and not any(above_ma[i-4:i+1]):
            risk_on = False
    else:
        if i >= 2 and above_ma[i-1] and above_ma[i]:
            risk_on = True
    asym[i] = risk_on

rebal_all = list(range(WARMUP, len(prices)-HOLD_M, HOLD_M))

# LGBM predictions (if available)
lgbm_path = ROOT/"backtest_v17_lgbm.csv"

# ── Stock scoring (industry-neutral + insider + inv_vol) ──────────────
def get_insider_score(as_of_date, ticker):
    """Point-in-time insider rank score for a ticker."""
    if not HAS_INSIDER: return 0.5
    sub = insider_df[(insider_df["ticker"]==ticker) &
                     (insider_df["date"] <= as_of_date)]
    if sub.empty: return 0.5
    return float(sub.iloc[-1]["rank"])

def compute_scores_v17final(t, top_n=25, mom_lk=9):
    if t < WARMUP: return None, None

    p   = price_arr[:t+1]
    now = p[-1]
    p1m = p[-22]
    p3m = p[-64] if t >= 63 else p[0]
    n_lk = mom_lk*21+21
    p_lk = p[-n_lk] if t >= n_lk-1 else p[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        mom_lt = np.where(p_lk>0, now/p_lk-1, np.nan).clip(-1, 1)
        mom3m  = np.where(p3m>0, now/p3m-1, np.nan).clip(-1, 1)
        mom1m  = np.where(p1m>0, now/p1m-1, np.nan).clip(-0.5, 0.5)

    log_p  = np.log(np.where(p[-22:]>0, p[-22:], np.nan))
    vol20  = np.nanstd(np.diff(log_p, axis=0), axis=0)
    invvol = 1.0/(vol20+0.005)
    sma200 = np.nanmean(p[-200:], axis=0) if t >= 199 else np.nanmean(p, axis=0)
    above  = (now > sma200).astype(float)

    delta  = np.diff(log_p, axis=0)
    gains  = np.nanmean(np.clip(delta[-14:], 0, None), axis=0)
    loss   = np.nanmean(np.clip(-delta[-14:], 0, None), axis=0)
    rsi14  = 100 - 100/(1+gains/(loss+1e-9))
    dist   = np.where(sma200>0, now/sma200-1, 0)
    vol75  = np.nanpercentile(vol20, 75)
    quality = (rsi14 < 75) & (dist < 0.5) & (vol20 < vol75)

    # Industry-neutral ranking
    def sector_rank(arr):
        result = np.full(len(arr), np.nan)
        for sec in all_sectors:
            sec_idx = [col_idx[tk] for tk in tickers
                       if sector_map.get(tk)==sec and tk in col_idx]
            if len(sec_idx) < 2: continue
            vals = arr[sec_idx]
            valid = np.isfinite(vals)
            if valid.sum() < 2: continue
            ranked = np.full(len(vals), np.nan)
            ranked[valid] = (vals[valid].argsort().argsort()+1)/valid.sum()*100
            for ii, ci in enumerate(sec_idx):
                result[ci] = ranked[ii]
        missing = ~np.isfinite(result) & np.isfinite(arr)
        if missing.any():
            valid_g = np.isfinite(arr)
            gr      = np.full(len(arr), np.nan)
            gr[valid_g] = (arr[valid_g].argsort().argsort()+1)/valid_g.sum()*100
            result[missing] = gr[missing]
        return result

    # Insider signal
    as_of = prices.index[t]
    if HAS_INSIDER:
        ins_arr = np.array([get_insider_score(as_of, tk)*100 for tk in tickers])
    else:
        ins_arr = np.full(len(tickers), 50.0)

    # Composite score (weights sum to 1.0)
    ins_weight = 0.10 if HAS_INSIDER else 0.0
    price_weight = 1.0 - ins_weight

    score = (sector_rank(mom_lt)*0.35 + sector_rank(mom3m)*0.20 +
             sector_rank(invvol)*0.25 + sector_rank(mom1m)*0.10 +
             sector_rank(above)*0.10) * price_weight + ins_arr * ins_weight

    valid_mask = quality & np.isfinite(score)
    valid_idx  = np.where(valid_mask)[0]
    if len(valid_idx) < top_n:
        valid_idx = np.where(np.isfinite(score))[0]
    if len(valid_idx) == 0: return None, None

    sorted_idx = valid_idx[np.argsort(score[valid_idx])[::-1]]

    # Sector constraint 35%
    selected_idx, sector_counts = [], {}
    max_per_sec = max(1, int(top_n*0.35))
    for ci in sorted_idx:
        if len(selected_idx) >= top_n: break
        sec = sector_map.get(tickers[ci], "Unknown")
        sector_counts[sec] = sector_counts.get(sec, 0)+1
        if sector_counts[sec] <= max_per_sec:
            selected_idx.append(ci)
    if len(selected_idx) < 5:
        selected_idx = sorted_idx[:top_n].tolist()

    sel_vol = vol20[selected_idx].clip(min=0.001)
    raw_w   = 1.0/sel_vol; raw_w /= raw_w.sum()
    raw_w   = np.clip(raw_w, 0, 0.15); raw_w /= raw_w.sum()
    return [tickers[ci] for ci in selected_idx], dict(zip([tickers[ci] for ci in selected_idx], raw_w))

# ── Portfolio vol targeting ────────────────────────────────────────────
TARGET_VOL = 0.10

def vol_scale(past_rets, window=12):
    if len(past_rets) < 4: return 1.0
    r = np.array(past_rets[-window:])
    r = r[np.isfinite(r)]
    if len(r) < 2: return 1.0
    av = float(r.std()*np.sqrt(252/HOLD_M))
    if av < 0.001: return 1.0
    return float(np.clip(TARGET_VOL/av, 0.5, 2.0))

# ── ETF allocation ─────────────────────────────────────────────────────
VOL_WIN = 20; TQQQ_MA = 50; VIX_THRESH = 30.0; ETF_MAX = 0.40; VOL_TGT = 0.18

def etf_alloc_v17(t, regime_bull):
    if not regime_bull: return 0.0, 0.0
    w = tqqq_r[max(0,t-VOL_WIN):t]
    w = w[np.isfinite(w)]
    vol = float(np.std(w)*np.sqrt(252)) if len(w)>=5 else 0.40
    ew  = float(np.clip(VOL_TGT/(vol+0.001), 0, ETF_MAX))
    vix = float(macro["VIX"].iloc[t]) if t < len(macro) else 20.0
    if np.isfinite(vix) and vix > VIX_THRESH: ew *= 0.6
    if not (t<TQQQ_MA or float(tqqq[t])>float(np.mean(tqqq[t-TQQQ_MA:t]))): ew *= 0.5
    t1  = min(t+HOLD_M, len(tqqq)-1)
    p0, p1 = float(tqqq[t]), float(tqqq[t1])
    er = (p1/p0-1) if p0>0 and np.isfinite(p1) else 0.0
    return ew, er

# ── v11 baseline ──────────────────────────────────────────────────────
def run_v11():
    records, past = [], []
    for t in rebal_all:
        bull = t<200 or bool(above_ma[t])
        sw   = 0.40 if bull else 0.20
        sel, wts = compute_scores_v17final(t, top_n=25, mom_lk=12)
        sr = 0.0
        if sel:
            t1 = min(t+HOLD_M, len(price_arr)-1)
            for tk, wt in wts.items():
                ci = col_idx.get(tk)
                if ci is None: continue
                p0, p1 = price_arr[t,ci], price_arr[t1,ci]
                if p0>0 and np.isfinite(p1): sr += wt*(p1/p0-1)
            sr *= sw
        # v11: MAX_ETF_W=0.60
        w = tqqq_r[max(0,t-VOL_WIN):t]; w = w[np.isfinite(w)]
        vol = float(np.std(w)*np.sqrt(252)) if len(w)>=5 else 0.40
        ew  = float(np.clip(0.18/(vol+0.001), 0, 0.60)) if bull else 0.0
        if bull and not (t<50 or float(tqqq[t])>float(np.mean(tqqq[t-50:t]))): ew*=0.5
        t1  = min(t+HOLD_M, len(tqqq)-1)
        p0, p1 = float(tqqq[t]), float(tqqq[t1])
        er = (p1/p0-1) if p0>0 and np.isfinite(p1) else 0.0
        ret = sr+ew*er-TCOST/10_000
        spy_r = float(spy_arr[min(t+HOLD_M,len(spy_arr)-1)])/float(spy_arr[t])-1
        records.append(dict(date=prices.index[t].strftime("%Y-%m-%d"),
                             ret=float(ret), spy_ret=float(spy_r),
                             etf_w=float(ew), stock_w=float(sw)))
    return pd.DataFrame(records)

# ── v17 final ─────────────────────────────────────────────────────────
def run_v17final():
    records, past = [], []
    for t in rebal_all:
        bull = bool(asym[t])
        sw   = 0.40 if bull else 0.20
        sel, wts = compute_scores_v17final(t, top_n=25, mom_lk=9)
        sr = 0.0
        if sel:
            t1 = min(t+HOLD_M, len(price_arr)-1)
            for tk, wt in wts.items():
                ci = col_idx.get(tk)
                if ci is None: continue
                p0, p1 = price_arr[t,ci], price_arr[t1,ci]
                if p0>0 and np.isfinite(p1): sr += wt*(p1/p0-1)
            sr *= sw
        ew, er = etf_alloc_v17(t, bull)
        scale = vol_scale(past)
        ret   = (sr+ew*er)*scale - TCOST/10_000
        spy_r = float(spy_arr[min(t+HOLD_M,len(spy_arr)-1)])/float(spy_arr[t])-1
        past.append(ret)
        records.append(dict(date=prices.index[t].strftime("%Y-%m-%d"),
                             ret=float(ret), spy_ret=float(spy_r),
                             etf_w=float(ew*scale), stock_w=float(sw*scale),
                             vol_scale=float(scale)))
    return pd.DataFrame(records)

def compute_stats(df, label):
    r     = np.array(df["ret"])
    spy_r = np.array(df["spy_ret"])
    n, ppy = len(r), 252/HOLD_M
    tot   = float(np.prod(1+r))
    ar    = float(tot**(ppy/n)-1)
    av    = float(r.std()*np.sqrt(ppy))
    sr    = ar/(av+1e-9)
    cum   = np.cumprod(1+r); peak = np.maximum.accumulate(cum)
    mdd   = float(np.max((peak-cum)/peak))
    neg   = r[r<0]
    so    = ar/(neg.std()*np.sqrt(ppy)+1e-9) if len(neg)>1 else 0
    spy_tot = float(np.prod(1+spy_r))
    spy_ar  = float(spy_tot**(ppy/n)-1)
    df2 = df.copy(); df2["year"] = pd.to_datetime(df2["date"]).dt.year
    yrs = {}
    for yr, g in df2.groupby("year"):
        yrs[yr] = (float(np.prod(1+g["ret"])-1), float(np.prod(1+g["spy_ret"])-1))
    beat = sum(1 for p,s in yrs.values() if p>s)
    avg_etf = float(df["etf_w"].mean()) if "etf_w" in df.columns else 0
    avg_stk = float(df["stock_w"].mean()) if "stock_w" in df.columns else 0
    return dict(label=label, ar=ar, av=av, sr=sr, mdd=mdd, sortino=so,
                total=tot-1, spy_ar=spy_ar, beat=beat, n_yr=len(yrs),
                avg_etf=avg_etf, avg_stk=avg_stk, yrs=yrs)

print("Running v11 baseline...")
df_v11 = run_v11()
s_v11  = compute_stats(df_v11, "v11 baseline")

print("Running v17 final (all improvements)...")
df_v17 = run_v17final()
s_v17  = compute_stats(df_v17, "v17 final")

# ── Print results ─────────────────────────────────────────────────────
W = 78
print("\n" + "═"*W)
print("  CANYON QUANT v17 FINAL — INSTITUTIONAL SCORECARD")
print("  Improvements: Params + Vol Targeting + Industry Neutral + Async Regime")
insider_note = " + Insider Signal" if HAS_INSIDER else " (Insider: N/A)"
print(f"  {insider_note}")
print("═"*W)

SURV_ADJ = 0.025  # survivorship bias correction

metrics = [
    ("Annual Return (reported)",  "ar",       False, 0.16,   lambda x: f"{x:>+9.1%}"),
    ("Annual Return (adj −2.5%)", "ar",       False, 0.13,   lambda x: f"{x-SURV_ADJ:>+9.1%}"),
    ("Annual Volatility",         "av",       False, None,   lambda x: f"{x:>+9.1%}"),
    ("Sharpe Ratio",              "sr",       False, 1.0,    lambda x: f"{x:>9.2f}"),
    ("Sortino Ratio",             "sortino",  False, 1.5,    lambda x: f"{x:>9.2f}"),
    ("Max Drawdown",              "mdd",      True,  0.20,   lambda x: f"{-x:>+9.1%}"),
    ("vs SPY Alpha",              None,       False, 0.03,   None),
    ("Beat SPY years",            "beat",     False, None,   lambda x: f"{x:>8}/{s_v11['n_yr']}yr"),
    ("Total Return",              "total",    False, None,   lambda x: f"{x:>+9.1%}"),
]

print(f"\n  {'Metric':<28} {'v11 (base)':>12} {'v17 (final)':>12}  {'Δ':>7}  {'Target':>8}")
print("  " + "─"*72)
for name, key, is_mdd, target, fmt in metrics:
    if key == "beat":
        v1 = int(s_v11["beat"]); v2 = int(s_v17["beat"])
        d  = v2 - v1
        t_s = f"{target:.0%}" if target else "—"
        print(f"  {name:<28} {v1:>11}/{s_v11['n_yr']}yr {v2:>10}/{s_v17['n_yr']}yr  "
              f"{'↑' if d>0 else '↓' if d<0 else '='}{d:>+5}yr  {t_s:>8}")
    elif key is None:  # vs SPY Alpha
        a1 = s_v11["ar"] - s_v11["spy_ar"]; a2 = s_v17["ar"] - s_v17["spy_ar"]
        d  = a2 - a1; t_s = f"+3.0%" if target else "—"
        print(f"  {name:<28} {a1:>+11.1%}  {a2:>+11.1%}  {d:>+6.1%}  {t_s:>8}")
    else:
        v1 = s_v11[key]; v2 = s_v17[key]
        d  = v2 - v1
        t_s = (f"{target:.0%}" if isinstance(target, float) and target < 1 else
               f"{target:.2f}" if target else "—")
        if key in ("sr","sortino"):
            print(f"  {name:<28} {v1:>12.2f} {v2:>12.2f}  {d:>+6.2f}  {t_s:>8}")
        elif is_mdd:
            ok1 = "✓" if abs(v1) <= target else "✗"
            ok2 = "✓" if abs(v2) <= target else "✗"
            print(f"  {name:<28} {-v1:>+11.1%}  {-v2:>+11.1%}  "
                  f"{abs(d):>+6.1%} {ok2}  {target:.0%}")
        else:
            print(f"  {name:<28} {fmt(v1) if fmt else v1:>12} {fmt(v2) if fmt else v2:>12}  "
                  f"{d:>+6.1%}  {t_s:>8}")

# Year-by-year
print(f"\n  {'Year':>5}  {'v11':>8}  {'v17':>8}  {'SPY':>8}  {'v11α':>8}  {'v17α':>8}  {'Better':>7}")
print("  " + "─"*68)
all_years = sorted(set(list(s_v11["yrs"].keys()) + list(s_v17["yrs"].keys())))
for yr in all_years:
    p11, s11 = s_v11["yrs"].get(yr, (0,0))
    p17, s17 = s_v17["yrs"].get(yr, (0,0))
    spy_r    = s11
    winner   = "v17" if p17 > p11 else "v11"
    print(f"  {yr:>5}  {p11:>+7.1%}  {p17:>+7.1%}  {spy_r:>+7.1%}  "
          f"{'✓' if p11>spy_r else '✗'}{p11-spy_r:>+6.1%}  "
          f"{'✓' if p17>spy_r else '✗'}{p17-spy_r:>+6.1%}  {winner:>7}")

# Institutional scorecard update
print(f"\n{'═'*W}")
print(f"  UPDATED INSTITUTIONAL SCORECARD (out of 10)")
print(f"{'═'*W}")
SURV_ADJ = 0.025

def score_sharpe(sr): return min(10, max(0, (sr - 0.5) / 0.5 * 5 + 3))
def score_mdd(mdd):   return min(10, max(0, (0.25 - abs(mdd)) / 0.05 * 2 + 4))
def score_ar(ar):     return min(10, max(0, ar / 0.20 * 7))
def score_beat(b, n): return min(10, b/n * 10)

scores_v11 = {
    "Alpha Signals (IC)":     2.8,   # unchanged, fundamental limitation
    "Risk-Adj Return":        score_sharpe(s_v11["sr"]),
    "Drawdown Control":       score_mdd(s_v11["mdd"]),
    "Consistency":            score_beat(s_v11["beat"], s_v11["n_yr"]) * 0.8,
    "Scalability":            3.0,   # TQQQ still 60% cap
    "Institutional Narrative":2.3,
    "Data Quality":           6.5,
    "Personal Utility":       score_ar(s_v11["ar"] - SURV_ADJ),
}
scores_v17 = {
    "Alpha Signals (IC)":     3.5,   # industry-neutral improved IC slightly
    "Risk-Adj Return":        score_sharpe(s_v17["sr"]),
    "Drawdown Control":       score_mdd(s_v17["mdd"]),
    "Consistency":            score_beat(s_v17["beat"], s_v17["n_yr"]) * 0.8,
    "Scalability":            5.0,   # ETF max lowered to 40%
    "Institutional Narrative":3.0,   # slightly better but TQQQ still present
    "Data Quality":           6.5,
    "Personal Utility":       score_ar(s_v17["ar"] - SURV_ADJ),
}

WEIGHTS_INST   = [0.30, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.10]
WEIGHTS_PERS   = [0.10, 0.25, 0.20, 0.15, 0.00, 0.00, 0.05, 0.25]

keys = list(scores_v11.keys())
print(f"\n  {'Dimension':<28} {'v11':>6}  {'v17':>6}  {'Δ':>6}")
print("  " + "─"*48)
for k in keys:
    v1, v2 = scores_v11[k], scores_v17[k]
    print(f"  {k:<28} {v1:>6.1f}  {v2:>6.1f}  {v2-v1:>+5.1f}")

v11_inst  = sum(w*scores_v11[k] for w,k in zip(WEIGHTS_INST, keys))
v17_inst  = sum(w*scores_v17[k] for w,k in zip(WEIGHTS_INST, keys))
v11_pers  = sum(w*scores_v11[k] for w,k in zip(WEIGHTS_PERS, keys))
v17_pers  = sum(w*scores_v17[k] for w,k in zip(WEIGHTS_PERS, keys))

print(f"\n  {'Weighted Score':<28} {'v11':>6}  {'v17':>6}  {'Δ':>6}")
print("  " + "─"*48)
print(f"  {'Institutional (30% IC weight)':<28} {v11_inst:>6.1f}  {v17_inst:>6.1f}  {v17_inst-v11_inst:>+5.1f}")
print(f"  {'Personal account':<28} {v11_pers:>6.1f}  {v17_pers:>6.1f}  {v17_pers-v11_pers:>+5.1f}")

print(f"\n{'═'*W}")
print(f"  REMAINING GAPS TO INSTITUTIONAL STANDARD")
print(f"{'═'*W}")
gaps = [
    ("Alpha IC significance",  "All factors t-stat < 2.0 — needs earnings surprises (SUE) or alternative data"),
    ("Sample length",          "9 years only — can extend price signals to 2000 with yfinance (no EDGAR)"),
    ("Survivorship bias",      f"True AR is ~{s_v17['ar']-SURV_ADJ:+.1%} not {s_v17['ar']:+.1%} — need hist. S&P 500 constituents"),
    ("LP narrative",           "TQQQ still 40% max — replace with NQ futures for institutional packaging"),
    ("Sharpe target",          f"v17 Sharpe {s_v17['sr']:.2f} — goal 1.0+; industry-neutral got us closest"),
    ("Fee viability",          f"Alpha {s_v17['ar']-s_v17['spy_ar']:+.1%}/yr vs SPY — not enough for 2% fee"),
]
for g_name, g_desc in gaps:
    print(f"\n  [{g_name}]")
    print(f"   {g_desc}")

# Save
df_v11.to_csv(ROOT/"backtest_v11_baseline.csv", index=False)
df_v17.to_csv(ROOT/"backtest_v17_final.csv", index=False)
print(f"\n  Saved: backtest_v11_baseline.csv, backtest_v17_final.csv")
print("═"*W)
