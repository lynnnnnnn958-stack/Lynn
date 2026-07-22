"""
Canyon Quant v25 — Three blindfold upgrades
============================================
vs v24 baseline (stock-only AR +6.4%, full strategy AR +16.5%, Sharpe 0.878)

Path 1: SUE + Analyst Revision composite  (analyst_revisions.csv, 482 tickers)
Path 2: TQQQ weight scan                  (find Sharpe-maximising etf_max)
Path 3: Long-Short overlay                (short bottom-N by same score, beta→0)

OOS window: 2019-01-01 → 2026-05-31
No lookahead: signals use data strictly before rebalance date
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT      = Path(__file__).parent
HOLD_M    = 21        # trading days per month
WARMUP    = 252
TCOST_BPS = 5         # one-way cost per stock per rebalance
BORROW    = 50 / 252  # annual borrow cost in bps per day → per period
SPY_WIN   = 200
LONG_N    = 25
SHORT_N   = 15
OOS_START = "2019-01-01"
DD_THRESH = 0.15
DD_SCALE  = 0.50

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
p25    = pd.read_csv(ROOT/"sp500_price_25yr.csv", index_col=0, parse_dates=True).sort_index()
spy_s  = p25["SPY"].copy()
stk    = p25.drop(columns=["SPY"], errors="ignore")
tickers= list(stk.columns)
col_idx= {c:i for i,c in enumerate(tickers)}
N      = len(tickers)
dates  = stk.index

price_arr= stk.values.astype(float)
spy_arr  = spy_s.values.astype(float)

sue_df = pd.read_csv(ROOT/"sue_data.csv"); sue_df["date"] = pd.to_datetime(sue_df["date"])

rev_df = pd.read_csv(ROOT/"analyst_revisions.csv")
rev_df["date"] = pd.to_datetime(rev_df["date"])
HAS_REV = len(rev_df) > 1000
print(f"  Stocks: {N}  SUE rows: {len(sue_df)}  Revision rows: {len(rev_df)}")

letf = pd.read_csv(ROOT/"letf_prices.csv", index_col=0, parse_dates=True).sort_index()
tqqq_arr = letf["TQQQ"].reindex(dates).ffill().bfill().values.astype(float)

# S&P 500 eligibility
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

def in_sp500(tk, d):
    ivs = sp500_intervals.get(tk)
    if not ivs: return True
    return any(s<=d<=e for s,e in ivs)

elig_cache = {}
def eligible(d):
    if d not in elig_cache:
        elig_cache[d] = {col_idx[tk] for tk in tickers if in_sp500(tk,d) and tk in col_idx}
    return elig_cache[d]

# ── Signals ───────────────────────────────────────────────────────────────────
def sue_signal(t_date):
    window = t_date - pd.DateOffset(months=9)
    rec    = sue_df[(sue_df.date>=window) & (sue_df.date<t_date)]
    if rec["ticker"].nunique() < 20: return np.full(N, 50.0)
    agg = rec.groupby("ticker")["sue"].mean()
    arr = np.full(N, np.nan)
    for tk,v in agg.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = v
    valid = np.isfinite(arr)
    ranked = np.full(N, 50.0)
    if valid.sum() >= 20:
        ranked[valid] = (arr[valid].argsort().argsort()+1)/valid.sum()*100
    return ranked

def revision_signal(t_date, lookback=60):
    """Net analyst revision score over past lookback days. No lookahead."""
    if not HAS_REV: return np.full(N, 50.0)
    ws  = t_date - pd.Timedelta(days=lookback)
    rec = rev_df[(rev_df.date>=ws) & (rev_df.date<t_date)]
    if rec["ticker"].nunique() < 10: return np.full(N, 50.0)
    agg = rec.groupby("ticker")["rev_score"].sum()
    arr = np.full(N, np.nan)
    for tk,v in agg.items():
        ci = col_idx.get(tk)
        if ci is not None: arr[ci] = v
    valid = np.isfinite(arr)
    ranked = np.full(N, 50.0)
    if valid.sum() >= 10:
        ranked[valid] = (arr[valid].argsort().argsort()+1)/valid.sum()*100
    return ranked

def regime_signal(t_idx):
    spy_v  = spy_arr[t_idx]
    spy_ma = spy_arr[max(0,t_idx-SPY_WIN):t_idx].mean() if t_idx>=SPY_WIN else spy_v
    above  = (price_arr[t_idx] > np.nanmean(price_arr[max(0,t_idx-SPY_WIN):t_idx], axis=0))
    breadth= float(above[np.isfinite(above)].mean())
    return (spy_v > spy_ma), breadth

# ── Core backtest ─────────────────────────────────────────────────────────────
def run_backtest(rev_weight=0.0, etf_max=0.40, long_short=False, label=""):
    """
    rev_weight : weight on analyst revision vs SUE (0=pure SUE, 0.20=20% revision)
    etf_max    : cap on TQQQ effective weight
    long_short : if True, also short bottom-SHORT_N stocks
    """
    oos_start = pd.Timestamp(OOS_START)
    rebal_dates= []
    for t in range(WARMUP, len(dates)-HOLD_M):
        d = dates[t]
        if d < oos_start: continue
        # First trading day of each month
        if t == WARMUP or dates[t-1].month != d.month:
            rebal_dates.append(t)

    records = []
    peak    = 1.0; nav = 1.0; dd_active = False
    consec_bear = 0; regime_state = True

    for t in rebal_dates:
        t_date = dates[t]
        t1     = min(t + HOLD_M, len(dates)-1)
        hold_days = t1 - t

        # Regime
        raw_bull, breadth = regime_signal(t)
        if raw_bull:
            consec_bear = 0; regime_state = True
        else:
            consec_bear += 1
            if consec_bear >= 2: regime_state = False
        bull = regime_state

        # Scores
        sue  = sue_signal(t_date)
        rev  = revision_signal(t_date) if rev_weight > 0 else np.full(N, 50.0)
        score= sue * (1-rev_weight) + rev * rev_weight

        elig = eligible(t_date)
        not_elig = np.array([ci not in elig for ci in range(N)])
        score[not_elig] = np.nan

        # Position filter: avoid overextended / too volatile
        if t >= 22:
            log_p = np.log(np.where(price_arr[t-22:t]>0, price_arr[t-22:t], np.nan))
            vol20 = np.nanstd(np.diff(log_p, axis=0), axis=0)
            sma200= np.nanmean(price_arr[max(0,t-SPY_WIN):t], axis=0)
            above = price_arr[t] > sma200
            p52w  = price_arr[max(0,t-252):t]
            high52= np.nanmax(p52w, axis=0)
            dist  = np.where(sma200>0, price_arr[t]/sma200-1, 0)
            gains = np.nanmean(np.clip(np.diff(log_p,axis=0)[-14:],0,None), axis=0)
            loss_ = np.nanmean(np.clip(-np.diff(log_p,axis=0)[-14:],0,None), axis=0)
            rsi14 = 100 - 100/(1+gains/(loss_+1e-9))
            vol75 = np.nanpercentile(vol20, 75)
            quality = (rsi14 < 75) & (dist < 0.5) & (vol20 < vol75)
        else:
            quality = np.ones(N, dtype=bool)
            vol20   = np.full(N, 0.20)

        valid = quality & np.isfinite(score)
        idx   = np.where(valid)[0]
        if len(idx) == 0: idx = np.where(np.isfinite(score))[0]
        if len(idx) == 0:
            records.append((t_date, 0.0, 0.0, 0.0, "bull" if bull else "bear"))
            continue

        sorted_i = idx[np.argsort(score[idx])[::-1]]

        # Long leg: top LONG_N
        sel = sorted_i[:LONG_N].tolist()
        if len(sel) == 0:
            records.append((t_date, 0.0, 0.0, 0.0, "bull" if bull else "bear"))
            continue

        sv  = vol20[sel].clip(min=0.001)
        raw = 1.0/sv; raw /= raw.sum()
        raw = np.clip(raw, 0, 0.15); raw /= raw.sum()
        stock_w = dict(zip(sel, raw))

        sw = 0.40 if bull else 0.20   # stock sleeve weight

        # Long leg return
        long_r = 0.0
        for ci, w in stock_w.items():
            p0 = price_arr[t, ci]; p1 = price_arr[t1, ci]
            if p0 > 0 and np.isfinite(p1):
                long_r += w * (p1/p0 - 1)
        long_r -= TCOST_BPS * 1e-4
        sr = long_r * sw

        # Short leg return (Path 3)
        short_r = 0.0
        if long_short and len(idx) > LONG_N + SHORT_N:
            short_cands = sorted_i[-(SHORT_N):]  # bottom SHORT_N by score
            sv_s = vol20[short_cands].clip(min=0.001)
            rw   = 1.0/sv_s; rw /= rw.sum()
            rw   = np.clip(rw, 0, 0.15); rw /= rw.sum()
            for ci, w in zip(short_cands, rw):
                p0 = price_arr[t, ci]; p1 = price_arr[t1, ci]
                if p0 > 0 and np.isfinite(p1):
                    short_r += w * (p1/p0 - 1)   # raw long return of short basket
            short_r_net = -short_r - TCOST_BPS*1e-4 - BORROW*hold_days*1e-4
            sr += short_r_net * sw   # short overlay equal-weight vs long sleeve

        # TQQQ overlay (Path 2)
        ew = er = 0.0
        if bull and etf_max > 0 and t >= 30:
            ov_r  = np.concatenate([[np.nan], np.diff(np.log(np.where(tqqq_arr>0, tqqq_arr, np.nan)))])
            w20   = ov_r[max(0,t-20):t]; w20 = w20[np.isfinite(w20)]
            tvol  = float(np.std(w20)*np.sqrt(252)) if len(w20)>=5 else 0.60
            eff_max = min(etf_max*1.2, 0.55) if breadth > 0.70 else etf_max
            ew    = float(np.clip(0.18/(tvol+0.001), 0.0, eff_max))
            p0t   = tqqq_arr[t]; p1t = tqqq_arr[t1]
            if p0t > 0 and np.isfinite(p1t):
                er = ew * (p1t/p0t - 1)

        total_r = sr + er

        # Drawdown stop
        nav *= (1 + total_r)
        peak = max(peak, nav)
        dd = (nav - peak) / peak
        if dd < -DD_THRESH: nav *= (1 + total_r * (DD_SCALE - 1))  # partial rescale

        records.append((t_date, total_r, long_r, float(ew), "bull" if bull else "bear"))

    df = pd.DataFrame(records, columns=["date","strat_ret","stock_ret","tqqq_w","regime"])
    df["date"] = pd.to_datetime(df["date"])
    return df

# ── Metrics ───────────────────────────────────────────────────────────────────
RF_M = 0.026/12

def metrics(df, label):
    r = df["strat_ret"].values
    n = len(r)
    if n == 0: return {}
    cum  = np.cumprod(1+r)
    yrs  = n/12
    ar   = float(cum[-1]**(1/yrs)-1) if yrs > 0 else 0
    vol  = float(np.std(r)*np.sqrt(12))
    rf   = RF_M
    sharpe = (ar - rf*12/12) / (vol+1e-9) if vol > 0 else 0
    # Sortino
    down = r[r<rf]; dsig = np.std(down)*np.sqrt(12) if len(down)>1 else vol
    sortino = (ar - rf*12/12)/(dsig+1e-9)
    # MDD
    cum_v = np.cumprod(1+r); peak_v = np.maximum.accumulate(cum_v)
    mdd   = float(np.min(cum_v/peak_v)-1)
    # SPY comparison
    spy_sub = spy_arr
    return dict(label=label, AR=ar, Sharpe=sharpe, Sortino=sortino, MDD=mdd, n=n, yrs=yrs)

# ── Run all configurations ────────────────────────────────────────────────────
print("\nRunning backtests (OOS 2019–2026)...")

configs = [
    # (label,               rev_w, etf_max, ls)
    ("v24 baseline (SUE)",  0.00,  0.40,    False),
    ("Path1: +15% Revision",0.15,  0.40,    False),
    ("Path1: +25% Revision",0.25,  0.40,    False),
    ("Path2: TQQQ 20% cap", 0.00,  0.20,    False),
    ("Path2: TQQQ 30% cap", 0.00,  0.30,    False),
    ("Path2: TQQQ 50% cap", 0.00,  0.50,    False),
    ("Path2: TQQQ 60% cap", 0.00,  0.60,    False),
    ("Path3: L/S overlay",  0.00,  0.40,    True),
    ("All3: Rev+TQQQ50+LS", 0.20,  0.50,    True),
]

results = []
for label, rev_w, etf_max, ls in configs:
    print(f"  {label}...", end="", flush=True)
    df = run_backtest(rev_weight=rev_w, etf_max=etf_max, long_short=ls, label=label)
    m  = metrics(df, label)
    results.append(m)
    print(f" AR={m['AR']:+.1%}  Sharpe={m['Sharpe']:.3f}  MDD={m['MDD']:.1%}")

# SPY OOS baseline
spy_oos = spy_arr[[i for i,d in enumerate(dates) if d >= pd.Timestamp(OOS_START)]]
spy_r   = np.diff(spy_oos)/spy_oos[:-1]
spy_n   = len(spy_r)//HOLD_M
spy_monthly = [float(np.prod(1+spy_r[i*HOLD_M:(i+1)*HOLD_M])-1) for i in range(spy_n)]
spy_cum = np.prod(1+np.array(spy_monthly))
spy_yrs = spy_n/12
spy_ar  = float(spy_cum**(1/spy_yrs)-1) if spy_yrs > 0 else 0
spy_vol = float(np.std(spy_monthly)*np.sqrt(12))
spy_sharpe = (spy_ar - 0.026)/(spy_vol+1e-9)
spy_peak   = np.maximum.accumulate(np.cumprod(1+np.array(spy_monthly)))
spy_mdd    = float(np.min(np.cumprod(1+np.array(spy_monthly))/spy_peak)-1)

# ── Results table ─────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"CANYON QUANT v25 — OOS COMPARISON  ({OOS_START} → 2026-05)")
print(f"{'='*80}")
print(f"{'Strategy':<35} {'AR':>8} {'Sharpe':>8} {'Sortino':>8} {'MDD':>8}")
print(f"{'-'*80}")
for m in results:
    marker = " ★" if m["Sharpe"] == max(r["Sharpe"] for r in results) else ""
    print(f"{m['label']:<35} {m['AR']:>+8.1%} {m['Sharpe']:>8.3f} {m['Sortino']:>8.3f} {m['MDD']:>8.1%}{marker}")
print(f"{'─'*80}")
print(f"{'SPY benchmark':<35} {spy_ar:>+8.1%} {spy_sharpe:>8.3f} {'—':>8} {spy_mdd:>8.1%}")
print(f"{'='*80}")

# Save
out = pd.DataFrame(results)
out.to_csv(ROOT/"backtest_v25_results.csv", index=False)

# ── Explain L/S result ────────────────────────────────────────────────────────
ls_result = next(r for r in results if "L/S" in r["label"])
base_result = results[0]
print(f"\nPath 3 (L/S) net vs baseline:")
print(f"  AR delta:     {ls_result['AR']-base_result['AR']:+.1%}")
print(f"  Sharpe delta: {ls_result['Sharpe']-base_result['Sharpe']:+.3f}")
print(f"  If L/S hurts: short basket is beating long basket → signal has no negative predictive power on short side")
print(f"\nBest config: {max(results, key=lambda x: x['Sharpe'])['label']}")
print(f"Saved: backtest_v25_results.csv")
