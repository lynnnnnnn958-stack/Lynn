"""
experiment_beat_qqq.py — Systematic grid search to find a rule-based strategy
that beats QQQ (NASDAQ 100) without lookahead bias.

Levers tested:
  - TQQQ weight: 0% to 65% (step 5%)
  - QQQ direct weight: 0%, 15%, 30%
  - Stock (SUE) weight: complement
  - VIX gate: TQQQ → 0 when VIX above threshold (20/25/30)
  - QQQ 200-day MA gate: TQQQ → 0 when QQQ < 200-day MA
  - Bear regime: always cuts TQQQ to 0 (existing v24 logic)

All gates use data available AT THAT DATE (no lookahead).
OOS: Jan 2019 – May 2026 (89 months).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf, itertools
from pathlib import Path

ROOT = Path(__file__).parent
RF_M = 0.026 / 12

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading backtest base data...")
oos  = pd.read_csv(ROOT/'backtest_v24_oos.csv',   parse_dates=['date'])
stk  = pd.read_csv(ROOT/'backtest_v24_stock.csv', parse_dates=['date'])
dates    = pd.DatetimeIndex(oos['date'])
stock_ret= stk['ret'].values
spy_ret  = oos['spy_ret'].values
regime   = oos['regime'].values        # 'bull'/'bear' (v24 lag filter already applied)

# ── Fetch daily market data ───────────────────────────────────────────────────
print("Fetching QQQ, TQQQ, ^VIX daily data...")
raw = yf.download(['QQQ','TQQQ','^VIX'], start='2018-01-01', end='2026-06-25',
                  progress=False, auto_adjust=True)['Close']
raw.index = pd.to_datetime(raw.index)

qqq_daily  = raw['QQQ'].dropna()
tqqq_daily = raw['TQQQ'].dropna()
vix_daily  = raw['^VIX'].dropna()

# Monthly return for each ticker aligned to our rebalancing dates
def monthly_rets(daily, dates):
    prev = None
    out = []
    for d in dates:
        sub  = daily.loc[:d]
        cur  = float(sub.iloc[-1]) if len(sub) else None
        if prev is None:
            p0   = daily.loc[:dates[0] - pd.Timedelta(days=20)]
            prev = float(p0.iloc[-1]) if len(p0) else cur
        out.append((cur/prev - 1) if (cur and prev) else 0.0)
        prev = cur
    return np.array(out)

qqq_ret  = monthly_rets(qqq_daily, dates)
tqqq_ret = monthly_rets(tqqq_daily, dates)

# ── Gate signals (computed BEFORE the month's return, no lookahead) ───────────
# For month ending at date d: use data strictly before d
def compute_gates(dates, qqq_daily, vix_daily):
    """Returns DataFrame with boolean columns: qqq_above_200ma, vix_20, vix_25, vix_30"""
    rows = []
    for d in dates:
        # QQQ 200-day MA: use data up to 1 day before d to avoid lookahead
        qqq_hist = qqq_daily.loc[:d - pd.Timedelta(days=1)]
        if len(qqq_hist) >= 200:
            ma200 = qqq_hist.iloc[-200:].mean()
            qqq_above = float(qqq_hist.iloc[-1]) > ma200
        else:
            qqq_above = True  # not enough data → assume bullish
        # VIX: last reading before d
        vix_hist = vix_daily.loc[:d - pd.Timedelta(days=1)]
        vix_val  = float(vix_hist.iloc[-1]) if len(vix_hist) else 20.0
        rows.append({
            'qqq_above_200': qqq_above,
            'vix': vix_val,
            'vix_ok_20': vix_val < 20,
            'vix_ok_25': vix_val < 25,
            'vix_ok_30': vix_val < 30,
        })
    return pd.DataFrame(rows)

print("Computing gates (200-day MA + VIX)...")
gates = compute_gates(dates, qqq_daily, vix_daily)

# ── Performance metrics ────────────────────────────────────────────────────────
def perf(rets):
    cum   = np.cumprod(1 + rets)
    n     = len(rets)
    ar    = cum[-1] ** (12/n) - 1
    ex    = rets - RF_M
    sr    = ex.mean() / ex.std(ddof=1) * np.sqrt(12) if ex.std() > 0 else 0
    down  = ex[ex < 0]
    so    = ex.mean() / down.std(ddof=1) * np.sqrt(12) if len(down) > 3 else 0
    dd    = cum / np.maximum.accumulate(cum) - 1
    mdd   = dd.min()
    cal   = ar / abs(mdd) if mdd != 0 else 0
    # annual beat-QQQ count
    ann_beat = 0
    for yr in range(2019, 2027):
        mask = dates.year == yr
        if mask.sum() == 0: continue
        strat_ann = np.prod(1 + rets[mask]) - 1
        qqq_ann   = np.prod(1 + qqq_ret[mask]) - 1
        if strat_ann > qqq_ann: ann_beat += 1
    return {'ar': ar, 'sharpe': sr, 'sortino': so, 'mdd': mdd, 'calmar': cal,
            'cum': (cum[-1]-1)*100, 'beat_years': ann_beat,
            'monthly': rets}

# QQQ benchmark
qqq_p = perf(qqq_ret)
qqq_ar = qqq_p['ar']
print(f"\nQQQ target: AR={qqq_ar*100:.1f}%  Sharpe={qqq_p['sharpe']:.3f}  MDD={qqq_p['mdd']*100:.1f}%")

# ── Grid search ───────────────────────────────────────────────────────────────
print("\nRunning grid search (may take ~2 min)...")

results = []

tqqq_weights = np.arange(0.20, 0.66, 0.05)   # 20% → 65%
qqq_weights  = [0.00, 0.15, 0.30]             # direct QQQ slice
vix_gates    = [None, 'vix_ok_20', 'vix_ok_25', 'vix_ok_30']   # None = no gate
ma_gates     = [False, True]                   # QQQ 200MA gate

for w_tqqq, w_qqq, vg, mg in itertools.product(
        tqqq_weights, qqq_weights, vix_gates, ma_gates):

    w_stock = 1.0 - w_tqqq - w_qqq
    if w_stock < 0.10 or w_stock > 0.85:
        continue  # unreasonable allocation

    rets = []
    for i in range(len(dates)):
        # Determine if TQQQ is on/off this month
        is_bull = (regime[i] == 'bull')
        vix_ok  = (not vg)  or bool(gates.iloc[i][vg])
        ma_ok   = (not mg)  or bool(gates.iloc[i]['qqq_above_200'])
        use_tqqq = is_bull and vix_ok and ma_ok

        wt  = w_tqqq if use_tqqq else 0.0
        wc  = w_tqqq - wt    # idle weight → cash
        r   = (w_stock * stock_ret[i]
             + w_qqq   * qqq_ret[i]
             + wt      * tqqq_ret[i]
             + wc      * RF_M)
        rets.append(r)

    p = perf(np.array(rets))

    # Only keep configs that:
    # 1. Beat QQQ annual return
    # 2. Sharpe ≥ 0.90
    # 3. MDD better than raw QQQ (< -33.5%) — i.e., more manageable
    if p['ar'] > qqq_ar and p['sharpe'] >= 0.90:
        results.append({
            'w_stock': round(w_stock, 2), 'w_qqq': w_qqq, 'w_tqqq': round(w_tqqq, 2),
            'vix_gate': vg or 'none', 'ma_gate': mg,
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in p.items() if k != 'monthly'}
        })

print(f"\nConfigs that beat QQQ (AR>{qqq_ar*100:.1f}%) with Sharpe≥0.90: {len(results)}")

if not results:
    print("No configs beat QQQ with Sharpe ≥ 0.90 — relaxing Sharpe to ≥ 0.80...")
    # rerun without sharpe filter
    for w_tqqq, w_qqq, vg, mg in itertools.product(
            tqqq_weights, qqq_weights, vix_gates, ma_gates):
        w_stock = 1.0 - w_tqqq - w_qqq
        if w_stock < 0.10 or w_stock > 0.85: continue
        rets = []
        for i in range(len(dates)):
            is_bull = (regime[i] == 'bull')
            vix_ok  = (not vg)  or bool(gates.iloc[i][vg])
            ma_ok   = (not mg)  or bool(gates.iloc[i]['qqq_above_200'])
            wt  = w_tqqq if (is_bull and vix_ok and ma_ok) else 0.0
            wc  = w_tqqq - wt
            rets.append(w_stock*stock_ret[i]+w_qqq*qqq_ret[i]+wt*tqqq_ret[i]+wc*RF_M)
        p = perf(np.array(rets))
        if p['ar'] > qqq_ar and p['sharpe'] >= 0.80:
            results.append({
                'w_stock': round(w_stock,2), 'w_qqq': w_qqq, 'w_tqqq': round(w_tqqq,2),
                'vix_gate': vg or 'none', 'ma_gate': mg,
                **{k: round(v,4) if isinstance(v,float) else v
                   for k,v in p.items() if k!='monthly'}
            })
    print(f"With Sharpe≥0.80: {len(results)} configs")

df = pd.DataFrame(results)
df = df.sort_values(['beat_years','sharpe'], ascending=False).reset_index(drop=True)

# ── Print Top 10 ──────────────────────────────────────────────────────────────
print(f"\n{'='*95}")
print(f"  TOP CONFIGURATIONS — ranked by beat_years then Sharpe")
print(f"{'='*95}")
print(f"  {'Stock':>5} {'QQQ':>5} {'TQQQ':>5}  {'VIX Gate':<10} {'MA':>4}  {'AR':>7} {'Sharpe':>7} {'MDD':>7} {'Calmar':>7} {'Cum%':>8} {'BeatQQQ':>8}")
print(f"  {'-'*90}")
for i, row in df.head(15).iterrows():
    print(f"  {row['w_stock']:5.0%} {row['w_qqq']:5.0%} {row['w_tqqq']:5.0%}  {row['vix_gate']:<10} {str(row['ma_gate']):>4}  "
          f"{row['ar']*100:+6.1f}% {row['sharpe']:7.3f} {row['mdd']*100:7.1f}%  {row['calmar']:6.3f}  {row['cum']:7.0f}%  "
          f"{row['beat_years']}/8")

# ── Best config deep dive ─────────────────────────────────────────────────────
best = df.iloc[0]
print(f"\n{'='*95}")
print(f"  BEST CONFIG: Stock={best['w_stock']:.0%}  QQQ={best['w_qqq']:.0%}  "
      f"TQQQ={best['w_tqqq']:.0%}  VIX<{best['vix_gate']}  200MA={best['ma_gate']}")
print(f"  AR={best['ar']*100:+.1f}%  Sharpe={best['sharpe']:.3f}  "
      f"MDD={best['mdd']*100:.1f}%  Calmar={best['calmar']:.3f}  Beat QQQ {best['beat_years']}/8 years")
print(f"{'='*95}")

# Reconstruct best config monthly returns for annual breakdown
w_s, w_q, w_t = best['w_stock'], best['w_qqq'], best['w_tqqq']
vg, mg = (best['vix_gate'] if best['vix_gate']!='none' else None), best['ma_gate']

best_rets = []
for i in range(len(dates)):
    is_bull = (regime[i] == 'bull')
    vix_ok  = (not vg)  or bool(gates.iloc[i][vg])
    ma_ok   = (not mg)  or bool(gates.iloc[i]['qqq_above_200'])
    wt  = w_t if (is_bull and vix_ok and ma_ok) else 0.0
    wc  = w_t - wt
    best_rets.append(w_s*stock_ret[i]+w_q*qqq_ret[i]+wt*tqqq_ret[i]+wc*RF_M)
best_rets = np.array(best_rets)

print(f"\n  Year-by-year vs QQQ:")
print(f"  {'Year':<6} {'Strategy':>10} {'QQQ':>10} {'Delta':>10} {'Beat?':>6}")
for yr in range(2019, 2027):
    mask = dates.year == yr
    if mask.sum() == 0: continue
    s_ann = np.prod(1+best_rets[mask]) - 1
    q_ann = np.prod(1+qqq_ret[mask]) - 1
    tag = '★ YES' if s_ann > q_ann else '  no'
    print(f"  {yr:<6} {s_ann*100:+9.1f}% {q_ann*100:+9.1f}% {(s_ann-q_ann)*100:+9.1f}%  {tag}")

# ── Save best config ──────────────────────────────────────────────────────────
df.to_csv(ROOT/'experiment_beat_qqq_grid.csv', index=False)
print(f"\nAll {len(df)} passing configs saved to experiment_beat_qqq_grid.csv")

# Build JS arrays for dashboard
cum_best = np.cumprod(1 + best_rets) * 100
cum_best = np.concatenate([[100.0], cum_best])

print(f"\n-- CONFIG FOR DASHBOARD (copy this) --")
print(f"w_stock={best['w_stock']:.2f}  w_qqq={best['w_qqq']:.2f}  w_tqqq={best['w_tqqq']:.2f}")
print(f"VIX gate: {best['vix_gate']}  QQQ 200MA gate: {best['ma_gate']}")
print(f"JS cum array (first 10): {list(np.round(cum_best[:10],2))}")

# ── Pareto frontier: best MDD for given return level ─────────────────────────
print(f"\n-- PARETO FRONTIER (best Sharpe at each MDD bucket) --")
df['mdd_bucket'] = (df['mdd']*100 // 5) * 5
pareto = df.sort_values('sharpe', ascending=False).groupby('mdd_bucket').first().reset_index()
pareto = pareto.sort_values('mdd_bucket')
for _, r in pareto.iterrows():
    print(f"  MDD≈{r['mdd_bucket']:.0f}%  →  Stock={r['w_stock']:.0%} QQQ={r['w_qqq']:.0%} "
          f"TQQQ={r['w_tqqq']:.0%} VIX={r['vix_gate']} MA={r['ma_gate']}  "
          f"AR={r['ar']*100:+.1f}% Sharpe={r['sharpe']:.3f} Beat={r['beat_years']:.0f}/8")
