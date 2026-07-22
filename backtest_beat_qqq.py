"""
backtest_beat_qqq.py — Test 3 paths to beat QQQ (NASDAQ).
Baseline: Canyon v24 (stock SUE + 23% TQQQ overlay)
Path A: Raise TQQQ to 50% in bull regime
Path B: 50% QQQ index + 50% SUE stock selection (no TQQQ)
Path C: 30% QQQ + 40% SUE stocks + 30% TQQQ (blended)
Run: python backtest_beat_qqq.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).parent

# ── Load baseline data ────────────────────────────────────────────────────────
oos  = pd.read_csv(ROOT/'backtest_v24_oos.csv',   parse_dates=['date'])
stk  = pd.read_csv(ROOT/'backtest_v24_stock.csv', parse_dates=['date'])

dates     = oos['date'].tolist()
full_ret  = oos['ret'].values          # v24 full strategy (stock + TQQQ)
stock_ret = stk['ret'].values          # SUE stock-only portfolio
spy_ret   = oos['spy_ret'].values
regime    = oos['regime'].values        # 'bull' / 'bear'
RF_M      = 0.026 / 12                 # monthly risk-free rate

# ── Fetch live monthly QQQ and TQQQ ──────────────────────────────────────────
print("Fetching QQQ and TQQQ monthly data...")
for ticker in ['QQQ','TQQQ']:
    prices = yf.download(ticker, start='2018-12-15', end='2026-06-25',
                         progress=False, auto_adjust=True)['Close'].squeeze()
    prev = None
    rets = []
    for d in dates:
        sub = prices.loc[:d]
        cur = float(sub.iloc[-1]) if not sub.empty else None
        if prev is None:
            p0 = prices.loc[:dates[0] - pd.Timedelta(days=20)]
            prev = float(p0.iloc[-1]) if not p0.empty else cur
        rets.append((cur / prev - 1) if (cur and prev) else 0.0)
        prev = cur
    if ticker == 'QQQ':
        qqq_ret = np.array(rets)
    else:
        tqqq_ret = np.array(rets)

print(f"QQQ  2019-2026 cum: {(np.prod(1+qqq_ret)-1)*100:.0f}%")
print(f"TQQQ 2019-2026 cum: {(np.prod(1+tqqq_ret)-1)*100:.0f}%")

# ── Portfolio return builders ─────────────────────────────────────────────────
def build_returns(w_stock, w_qqq, w_tqqq, w_cash=0.0, bear_tqqq_off=True):
    """Build monthly portfolio returns.
    In bear: w_tqqq → 0, remaining goes to cash (RF_M).
    """
    rets = []
    for i, reg in enumerate(regime):
        wt = w_tqqq if (reg == 'bull' or not bear_tqqq_off) else 0.0
        wc = w_cash + (w_tqqq - wt)   # extra cash in bear
        r = (w_stock * stock_ret[i]
           + w_qqq   * qqq_ret[i]
           + wt       * tqqq_ret[i]
           + wc       * RF_M)
        rets.append(r)
    return np.array(rets)

# ── Performance metrics ────────────────────────────────────────────────────────
def metrics(rets, label):
    n = len(rets)
    cum  = np.cumprod(1 + rets)
    ar   = cum[-1] ** (12/n) - 1
    excess = rets - RF_M
    sr   = excess.mean() / excess.std(ddof=1) * np.sqrt(12)
    down = excess[excess < 0]
    sort = excess.mean() / down.std(ddof=1) * np.sqrt(12) if len(down) else np.nan
    dd   = (cum / np.maximum.accumulate(cum)) - 1
    mdd  = dd.min()
    calmar = ar / abs(mdd) if mdd != 0 else np.nan
    # Annual returns
    ann_rets = {}
    for yr in range(2019, 2027):
        mask = pd.DatetimeIndex(dates).year == yr
        if mask.sum() == 0: continue
        ann_rets[yr] = np.prod(1 + rets[mask]) - 1
    # Beat QQQ years
    beat_qqq = sum(1 for yr, r in ann_rets.items()
                   if yr in ann_qqq and r > ann_qqq[yr])
    return {
        'label': label, 'ar': ar, 'sharpe': sr, 'sortino': sort,
        'mdd': mdd, 'calmar': calmar,
        'cum': (cum[-1]-1)*100, 'cum_arr': (cum*100).tolist(),
        'ann_rets': ann_rets, 'beat_qqq_years': beat_qqq,
        'total_years': len(ann_rets),
    }

# Annual QQQ returns for beat-rate calculation
def ann_series(rets):
    out = {}
    for yr in range(2019, 2027):
        mask = pd.DatetimeIndex(dates).year == yr
        if mask.sum() == 0: continue
        out[yr] = np.prod(1 + rets[mask]) - 1
    return out
ann_qqq = ann_series(qqq_ret)

# ── Run all paths ─────────────────────────────────────────────────────────────
# Baseline: v24 as-is  (use actual backtest returns)
baseline_m = metrics(full_ret,   'Canyon v24 (baseline)')
qqq_m      = metrics(qqq_ret,    'QQQ / NASDAQ 100')
spy_m      = metrics(spy_ret,    'SPY S&P 500')

# Path A: TQQQ to 50% in bull, 50% stock
# (v24 has ~23% TQQQ; here we go to 50%)
pathA = build_returns(w_stock=0.50, w_qqq=0.00, w_tqqq=0.50, w_cash=0.00)
pathA_m = metrics(pathA, 'Path A: TQQQ 50% (max leverage)')

# Path B: 50% QQQ index + 50% SUE stock selection (no TQQQ)
pathB = build_returns(w_stock=0.50, w_qqq=0.50, w_tqqq=0.00, w_cash=0.00)
pathB_m = metrics(pathB, 'Path B: 50% QQQ + 50% SUE stocks')

# Path C: 30% QQQ + 40% SUE stocks + 30% TQQQ
pathC = build_returns(w_stock=0.40, w_qqq=0.30, w_tqqq=0.30, w_cash=0.00)
pathC_m = metrics(pathC, 'Path C: 30% QQQ + 40% SUE + 30% TQQQ')

# Bonus D: Pure QQQ — what we're trying to beat
# Already have qqq_m

# ── Print results ─────────────────────────────────────────────────────────────
all_m = [qqq_m, baseline_m, pathA_m, pathB_m, pathC_m, spy_m]

print(f"\n{'='*80}")
print(f"  BEAT QQQ BACKTEST — Out-of-Sample 2019–2026")
print(f"{'='*80}")
print(f"{'Strategy':<38} {'AR':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7} {'Cum%':>8} {'Beat QQQ':>9}")
print(f"{'-'*80}")
for m in all_m:
    beat = f"{m['beat_qqq_years']}/{m['total_years']}" if 'beat_qqq_years' in m else '—'
    print(f"  {m['label']:<36} {m['ar']*100:>+6.1f}% {m['sharpe']:>7.3f} {m['mdd']*100:>7.1f}%  {m['calmar']:>6.3f}  {m['cum']:>7.0f}%  {beat:>8}")

print(f"\n{'─'*80}")
print(f"  Annual Returns by Year")
print(f"{'─'*80}")
header = f"  {'Year':<6}" + "".join(f"  {m['label'][:14]:>14}" for m in all_m)
print(header)
for yr in range(2019, 2027):
    row = f"  {yr:<6}"
    for m in all_m:
        val = m['ann_rets'].get(yr)
        if val is None:
            row += f"  {'—':>14}"
        else:
            tag = '★' if val > ann_qqq.get(yr, 0) else ' '
            row += f"  {val*100:>+12.1f}%{tag}"
    print(row)

print(f"\n★ = beats QQQ that year")
print(f"\n{'─'*80}")
print(f"  KEY TAKEAWAYS")
print(f"{'─'*80}")
for m in [pathA_m, pathB_m, pathC_m]:
    gap = m['ar'] - qqq_m['ar']
    tag = 'BEATS QQQ ★' if gap > 0 else f'trails QQQ by {abs(gap)*100:.1f}%/yr'
    print(f"  {m['label']}: AR={m['ar']*100:+.1f}%  ({tag})")
    print(f"    Sharpe: {m['sharpe']:.3f}  MDD: {m['mdd']*100:.1f}%  Beat QQQ in {m['beat_qqq_years']}/{m['total_years']} years")
    print()

# ── Save results ──────────────────────────────────────────────────────────────
rows = []
for m in all_m:
    rows.append({
        'strategy': m['label'], 'ar_pct': round(m['ar']*100,2),
        'sharpe': round(m['sharpe'],3), 'sortino': round(m['sortino'],3),
        'mdd_pct': round(m['mdd']*100,2), 'calmar': round(m['calmar'],3),
        'cum_pct': round(m['cum'],1),
        **{str(yr): round(v*100,2) for yr,v in m['ann_rets'].items()}
    })
pd.DataFrame(rows).to_csv(ROOT/'backtest_beat_qqq_results.csv', index=False)
print(f"Results saved to backtest_beat_qqq_results.csv")

# ── Export JS arrays for dashboard update ─────────────────────────────────────
print(f"\nJS arrays for dashboard:")
for m in all_m:
    print(f"  {m['label'][:25]}: [{','.join(str(round(v,2)) for v in m['cum_arr'][:5])}...]")
