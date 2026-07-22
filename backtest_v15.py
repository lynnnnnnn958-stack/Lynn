"""
Canyon Quant v15 — FRED Macro-Enhanced TQQQ Timing
===================================================
Core insight from IC analysis: v11's alpha comes from TQQQ timing,
not stock selection (stock ICs all ≈ 0, not significant).

v15 upgrades the TQQQ regime detector with 3 FRED signals:
  1. Initial Jobless Claims (ICSA): weekly leading indicator
     - Rising claims 15%+ above 52-week trend → risk-off
     - Leads market peaks by 3-6 months historically
  2. Moody's BAA-AAA credit spread (DBAA - DAAA):
     - Real credit conditions, not ETF proxy
     - Spread > threshold or rapidly widening → risk-off
  3. T10Y2Y: 10Y-2Y yield curve slope
     - Extreme inversion + tightening → risk-off

Stock selection: unchanged from v11 (IC ≈ 0 on all signals,
no point changing a system that already works).
"""

import warnings, numpy as np, pandas as pd, requests
from io import StringIO
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(".")
HOLD_M     = 21
TOP_M      = 25
TCOST      = 5
WARMUP     = 252
SPY_WIN    = 200
TQQQ_WIN   = 50
VOL_WIN    = 20
VOL_TARGET = 0.18
MAX_ETF_W  = 0.60
TRAIL_STOP = 0.20
STOCK_BULL = 0.40
STOCK_BEAR = 0.20
MAX_SECTOR = 0.35
VIX_THRESH = 30.0
HYG_WIN    = 20

# ── FRED macro thresholds ─────────────────────────────────────────────────
ICSA_RATIO_WARN   = 1.15   # claims 15% above 52w trend = warning
ICSA_RATIO_BEAR   = 1.30   # claims 30% above 52w trend = bear
CREDIT_WARN       = 1.10   # BAA-AAA > 1.10% = elevated risk
CREDIT_BEAR       = 1.50   # BAA-AAA > 1.50% = credit stress
CREDIT_DELTA_WARN = 0.20   # 20-day rise in spread > 0.20% = warning
T2Y10_EXTREME     = -0.80  # 10Y-2Y < -0.80% = extreme inversion

# ── Load price/macro data ─────────────────────────────────────────────────

print("Loading data...")
prices  = pd.read_csv(ROOT/"sp500_price_8yr.csv",  index_col=0, parse_dates=True).sort_index()
spy     = prices["SPY"].copy()
prices  = prices.drop(columns=["SPY"])
letf_df = pd.read_csv(ROOT/"letf_prices.csv",      index_col=0, parse_dates=True).sort_index()
macro   = pd.read_csv(ROOT/"macro_regime.csv",     index_col=0, parse_dates=True).sort_index()
sector_map = (pd.read_csv(ROOT/"sp500_sectors.csv", index_col=0).squeeze().to_dict()
              if (ROOT/"sp500_sectors.csv").exists() else {})

common   = prices.index.intersection(letf_df.index).intersection(macro.index)
prices   = prices.reindex(common)
letf_df  = letf_df.reindex(common).ffill().bfill()
spy      = spy.reindex(common).ffill()
macro    = macro.reindex(common).ffill()
tqqq     = letf_df["TQQQ"].values
tqqq_logr = np.concatenate([[np.nan], np.diff(np.log(np.where(tqqq > 0, tqqq, np.nan)))])

# ── Download FRED macro signals ───────────────────────────────────────────

def fred_csv(sid: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    r = requests.get(url, headers={"User-Agent": "canyon-quant research@example.com"}, timeout=20)
    df = pd.read_csv(StringIO(r.text), na_values=[".", ""])
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    return df.set_index("observation_date").iloc[:, 0].rename(sid)

print("Downloading FRED macro signals...")
icsa  = fred_csv("ICSA")               # weekly initial jobless claims
baa   = fred_csv("DBAA")               # Moody's Baa corporate yield
aaa   = fred_csv("DAAA")               # Moody's Aaa corporate yield
t2y10 = fred_csv("T10Y2Y")             # 10Y minus 2Y yield spread

credit_spread = (baa - aaa).rename("CREDIT_SPREAD")

# Forward-fill FRED weekly to daily, align to price index
fred_daily = pd.DataFrame({
    "icsa":    icsa.reindex(common, method="ffill"),
    "credit":  credit_spread.reindex(common, method="ffill"),
    "t2y10":   t2y10.reindex(common, method="ffill"),
}).ffill().bfill()

print(f"  ICSA: {icsa.dropna().iloc[-1]:.0f}  "
      f"Credit: {credit_spread.dropna().iloc[-1]:.2f}%  "
      f"T10Y2Y: {t2y10.dropna().iloc[-1]:.2f}%")

# ── Regime helpers ────────────────────────────────────────────────────────

def spy_above_200ma(t): return t < SPY_WIN or float(spy.iloc[t]) > spy.iloc[t-SPY_WIN+1:t+1].mean()
def tqqq_above_50ma(t): return t < TQQQ_WIN or float(tqqq[t]) > np.mean(tqqq[t-TQQQ_WIN:t])

def tqqq_vol(t):
    w = tqqq_logr[max(0, t-VOL_WIN):t]; w = w[np.isfinite(w)]
    return float(np.std(w)*np.sqrt(252)) if len(w) >= 5 else 0.40


def macro_regime_v15(t: int) -> tuple[float, dict]:
    """
    Returns (etf_alloc, details_dict).
    Replaces HYG/IEF proxy with real FRED signals.
    """
    # ── Hard floor: SPY below 200MA ──
    if not spy_above_200ma(t):
        return 0.0, {"reason": "SPY<200MA"}

    details = {}

    # ── VIX signal (unchanged from v11) ──
    vix_v = float(macro["VIX"].iloc[t])
    sig_vix = np.isfinite(vix_v) and vix_v < VIX_THRESH
    details["vix"] = vix_v

    # ── ICSA signal ──
    if t >= 252:
        icsa_now  = float(fred_daily["icsa"].iloc[t])
        icsa_52w  = float(fred_daily["icsa"].iloc[t-252:t].mean())
        icsa_ratio = icsa_now / (icsa_52w + 1)
        sig_icsa = icsa_ratio < ICSA_RATIO_WARN
        icsa_severe = icsa_ratio > ICSA_RATIO_BEAR
        details["icsa_ratio"] = icsa_ratio
    else:
        sig_icsa, icsa_severe = True, False

    # ── Credit spread signal (Moody's BAA-AAA) ──
    if t >= HYG_WIN:
        cs_now  = float(fred_daily["credit"].iloc[t])
        cs_20d  = float(fred_daily["credit"].iloc[t-20])
        cs_delta = cs_now - cs_20d
        sig_credit = (cs_now < CREDIT_WARN) and (cs_delta < CREDIT_DELTA_WARN)
        credit_severe = (cs_now > CREDIT_BEAR) or (cs_delta > CREDIT_DELTA_WARN * 2)
        details["credit"] = cs_now
        details["credit_delta"] = cs_delta
    else:
        sig_credit, credit_severe = True, False

    # ── T10Y2Y: only flag extreme inversions ──
    t2y10_v = float(fred_daily["t2y10"].iloc[t])
    sig_curve = (t2y10_v > T2Y10_EXTREME)  # only trigger on extreme inversion
    details["t2y10"] = t2y10_v

    # ── Combine signals into ETF allocation ──
    # Count how many risk signals are on
    n_risk = sum([not sig_vix, not sig_icsa, not sig_credit, not sig_curve])
    severe = icsa_severe or credit_severe

    # Base allocation from vol targeting
    vol   = tqqq_vol(t)
    vc    = min(MAX_ETF_W, VOL_TARGET / (vol + 0.001))
    base  = vc if tqqq_above_50ma(t) else vc * 0.5

    # Scale down by risk signals (more aggressive than v11)
    scale_map = {0: 1.00, 1: 0.80, 2: 0.55, 3: 0.30, 4: 0.10}
    scale = scale_map.get(n_risk, 0.10)
    if severe:
        scale *= 0.5  # extra cut for severe conditions

    alloc = float(np.clip(base * scale, 0, MAX_ETF_W))
    details["n_risk"] = n_risk
    details["alloc"]  = alloc
    return alloc, details


def tqqq_period_return(t, ew):
    if ew <= 0: return 0.0
    t1 = min(t + HOLD_M, len(tqqq) - 1)
    e  = tqqq[t]
    if e <= 0 or not np.isfinite(e): return 0.0
    peak, ed = e, t1
    for d in range(t+1, t1+1):
        p = tqqq[d]
        if np.isfinite(p):
            peak = max(peak, p)
            if p < peak * (1 - TRAIL_STOP): ed = d; break
    ep = tqqq[ed]
    return float(ep/e - 1) if np.isfinite(ep) and ep > 0 else 0.0


def rk(s): return s.rank(pct=True, na_option="bottom") * 100


def stock_signal_v11(t):
    """Unchanged from v11 — stock ICs are all ~0, no point modifying."""
    if t < WARMUP: return {}
    p    = prices.iloc[:t+1]; now = p.iloc[-1]; p1m = p.iloc[-22]
    p3m  = p.iloc[-64] if t >= 63 else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])
    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    mom = (rk((p1m/p13m-1).clip(-1,1))*0.35 + rk((now/p3m-1).clip(-1,1))*0.20 +
           rk(invvol)*0.25 + rk((now/p1m-1).clip(-0.5,0.5))*0.10 +
           rk((now>sma200).astype(float))*0.10).dropna()
    vol20 = p.pct_change().iloc[-21:].std()
    gains = p.pct_change().iloc[-15:].clip(lower=0).iloc[-14:].mean()
    loss  = (-p.pct_change().iloc[-15:].clip(upper=0)).iloc[-14:].mean()
    rsi14 = 100 - 100/(1 + gains/(loss+1e-9))
    dist  = now/(sma200+1e-9)-1; vol75 = vol20.quantile(0.75)
    qmask = (rsi14<75) & (dist<0.50) & (vol20<vol75)
    valid = mom[mom.index.isin(qmask[qmask].index)]
    pool  = valid if len(valid) >= TOP_M else mom
    cands = pool.nlargest(TOP_M*2).index.tolist(); sel, sc = [], {}
    for tk in cands:
        sec = sector_map.get(tk, "Unknown")
        if sc.get(sec, 0) < MAX_SECTOR and len(sel) < TOP_M:
            sel.append(tk); sc[sec] = sc.get(sec, 0) + 1/TOP_M
    if len(sel) < 10: sel = pool.nlargest(TOP_M).index.tolist()
    sv  = vol20[sel].clip(lower=0.001); rw = 1.0/sv; rw = rw/rw.sum()
    rw  = rw.clip(upper=0.15)
    return (rw/rw.sum()).to_dict()


# ── Backtest engine ───────────────────────────────────────────────────────

def run_backtest(use_v15_regime=True):
    rebal   = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))
    records = []

    for t in rebal:
        t1   = min(t + HOLD_M, len(prices) - 1)
        bull = spy_above_200ma(t)
        sw   = STOCK_BULL if bull else STOCK_BEAR

        # Stock selection (v11 unchanged)
        wts   = stock_signal_v11(t)
        raw_r = sum(wts.get(tk, 1/len(wts)) * (float(prices.iloc[t1].get(tk, np.nan))
                    / float(prices.iloc[t].get(tk, np.nan)) - 1)
                    for tk in wts if float(prices.iloc[t].get(tk, np.nan)) > 0) if wts else 0.0
        sr2   = raw_r * sw

        # ETF timing
        if use_v15_regime:
            ew, dets = macro_regime_v15(t)
            n_risk = dets.get("n_risk", 0)
        else:
            # v11 regime
            if not spy_above_200ma(t):
                ew = 0.0
            else:
                vix_v = float(macro["VIX"].iloc[t])
                s2 = np.isfinite(vix_v) and vix_v < VIX_THRESH
                if t >= HYG_WIN and "HYG" in macro.columns:
                    hyg, ief = float(macro["HYG"].iloc[t]), float(macro["IEF"].iloc[t])
                    hyg0, ief0 = float(macro["HYG"].iloc[t-HYG_WIN]), float(macro["IEF"].iloc[t-HYG_WIN])
                    s3 = (hyg/ief >= hyg0/ief0*0.99) if hyg0>0 and ief0>0 else True
                else:
                    s3 = True
                vol = tqqq_vol(t); vc = min(MAX_ETF_W, VOL_TARGET/(vol+0.001))
                base = vc if tqqq_above_50ma(t) else vc*0.5
                fine = {0:1.0, 1:0.85, 2:0.65}[sum([not s2, not s3])]
                ew = float(np.clip(base*fine, 0, MAX_ETF_W))
            n_risk = 0

        er  = tqqq_period_return(t, ew)
        ret = sr2 + ew * er - TCOST / 10_000
        spy_r = float(spy.iloc[t1]) / float(spy.iloc[t]) - 1
        records.append(dict(date=prices.index[t], ret=ret, sr=sr2, etf=ew*er,
                            ew=ew, spy_ret=spy_r, n_risk=n_risk, bull=bull))

    return pd.DataFrame(records).set_index("date")


def stats(df, label):
    r   = df["ret"]; n = len(r); ppy = 252/HOLD_M
    tot = float((1+r).prod()); ar = float(tot**(ppy/n)-1)
    vol = float(r.std()*np.sqrt(ppy))
    sr  = ar / (vol+1e-9)
    cum = (1+r).cumprod(); mdd = float(-((cum-cum.cummax())/cum.cummax()).min())
    spy_tot = float((1+df["spy_ret"]).prod()); spy_ar = float(spy_tot**(ppy/n)-1)
    sort_r  = r.copy(); sort_r[sort_r > 0] = 0
    sortino = ar / (sort_r.std()*np.sqrt(ppy)+1e-9)

    print(f"\n{'═'*62}")
    print(f"  {label}")
    print(f"{'═'*62}")
    print(f"  Annual Return:  {ar:>+7.1%}    Sharpe: {sr:.2f}    Sortino: {sortino:.2f}")
    print(f"  Max Drawdown:   {-mdd:>+7.1%}    vs SPY: {ar-spy_ar:>+.1%}/yr")
    print(f"  Total Return:   {tot-1:>+7.1%}")

    df2 = df.copy(); df2["year"] = df2.index.year
    beats = 0
    print(f"\n  {'Year':<6} {'Strat':>8} {'SPY':>8} {'Alpha':>8} {'ETF_w':>7} {'Risk':>5}")
    for y in sorted(df2["year"].unique()):
        yd = df2[df2["year"]==y]
        yr = float((1+yd["ret"]).prod()-1)
        ys = float((1+yd["spy_ret"]).prod()-1)
        ew_avg = yd["ew"].mean()
        nr_avg = yd["n_risk"].mean()
        f = "✓" if yr>ys else "✗"
        beats += (yr>ys)
        print(f"  {y:<6} {yr:>+7.1%}  {ys:>+7.1%}  {f}{yr-ys:>+6.1%}  {ew_avg:>5.1%}  {nr_avg:.1f}")
    print(f"\n  Beat SPY: {beats}/{len(df2['year'].unique())} years")
    return ar, sr, mdd


# ── Run comparison ────────────────────────────────────────────────────────

print("\nRunning v11 (baseline)...")
df11 = run_backtest(use_v15_regime=False)
ar11, sr11, mdd11 = stats(df11, "v11 — Original Regime (VIX + HYG/IEF proxy)")

print("\nRunning v15 (FRED-enhanced regime)...")
df15 = run_backtest(use_v15_regime=True)
ar15, sr15, mdd15 = stats(df15, "v15 — FRED Regime (ICSA + Credit + Curve)")

print(f"\n{'═'*62}")
print("  Delta: v15 vs v11")
print(f"{'═'*62}")
print(f"  AR:     {ar15-ar11:>+.1%}")
print(f"  Sharpe: {sr15-sr11:>+.2f}")
print(f"  MaxDD:  {mdd11-mdd15:>+.1%}")
print(f"{'═'*62}")

# Regime signal breakdown for 2022
print("\n  2022 regime detail (worst year for TQQQ):")
df22 = df15[df15.index.year == 2022]
print(f"  Avg ETF weight: {df22['ew'].mean():.1%}  (v11 would be higher)")
print(f"  Avg risk flags: {df22['n_risk'].mean():.1f}")

df15.to_csv("backtest_v15.csv")
df11.to_csv("backtest_v11_rerun.csv")
print("\nSaved: backtest_v15.csv")
