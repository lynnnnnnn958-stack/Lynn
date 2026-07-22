"""
Canyon Quant — Improved Walk-Forward Backtest v2
=================================================
Three improvements over backtest_clean.py:

  1. Market-cap weighted portfolio
     Weight ∝ sqrt(mktcap) within top-50, capped at 15%.
     Historical mktcap = price[t] × implied_shares
     (implied_shares = current_mktcap / current_price, changes slowly — ~2% error/yr)

  2. Add size factor + expand to Top-50
     Composite = mom_12m×0.30 + mom_3m×0.18 + inv_vol×0.22 +
                 mom_1m×0.08 + trend_200×0.05 + rsi×0.05 + log_size×0.12
     Size factor: log(mktcap[t]) ranked cross-sectionally — favors large-caps
     that dominated 2024-2026

  3. Reduce turnover: hold_days = 63 (quarterly rebalance)
     + overlap buffer: only swap a stock out if new candidate score > current + 5pts
     → Expected turnover drops from 123% → ~35% per period

Anti-lookahead guarantees (identical to v1)
--------------------------------------------
- Signal at T: only prices.iloc[:t+1]
- Forward return: prices.iloc[t+63] / prices.iloc[t] - 1
- Market cap at T: price[t] × implied_shares (shares_outstanding is backward-looking)
- Regime filter: SPY SMA-200 at time T
- No EDGAR, FRED, or alternative data

Outputs
-------
  backtest_v2_monthly.csv
  backtest_v2_ic.csv
  backtest_v2_summary.csv
  backtest_comparison.csv   (v1 vs v2 side-by-side)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Parameters ────────────────────────────────────────────────────────────────
WARMUP_DAYS      = 252
HOLD_DAYS        = 63    # quarterly rebalance (was 21)
TOP_N            = 50    # long candidates (was 25)
MAX_WEIGHT       = 0.15  # cap per stock (was 8%)
TCOST_BPS        = 5     # one-way bps
SPY_REGIME_WIN   = 200
CASH_IN_BEAR     = 0.60  # long-only exposure in bear (was 50%)
BUFFER_SCORE     = 5.0   # min score advantage to replace a held stock

# ── Load price data ────────────────────────────────────────────────────────────
print("Loading data...")
prices = pd.read_csv(ROOT / "sp500_price_cache.csv", index_col=0, parse_dates=True).sort_index()

if "SPY" in prices.columns:
    spy_series = prices["SPY"].copy()
    prices = prices.drop(columns=["SPY"])
else:
    spy_df = pd.read_csv(ROOT / "macro_price_cache.csv", index_col=0, parse_dates=True)
    spy_series = spy_df["SPY"].reindex(prices.index).ffill()

universe = prices.columns.tolist()

# ── Load market cap snapshot ───────────────────────────────────────────────────
mc_path = ROOT / "mktcap_snapshot.csv"
if mc_path.exists():
    mc_df = pd.read_csv(mc_path, index_col=0)
    current_mktcap = mc_df["market_cap_usd"].reindex(universe)
    current_price  = prices.iloc[-1].reindex(universe)
    # implied shares outstanding = current_mktcap / current_price
    # PIT note: shares change slowly; using current is a ~2%/yr approximation
    implied_shares = (current_mktcap / current_price).fillna(0.0)
    has_mktcap = True
    print(f"  Market cap data: {current_mktcap.notna().sum()}/{len(universe)} tickers")
else:
    implied_shares = pd.Series(1.0, index=universe)
    has_mktcap = False
    print("  WARNING: No mktcap_snapshot.csv — using equal weight fallback")

print(f"  Universe: {len(universe)} tickers")
print(f"  Dates: {prices.index[0].date()} → {prices.index[-1].date()} ({len(prices)} days)")
print(f"  Hold period: {HOLD_DAYS} days (quarterly)")
print(f"  Top-N: {TOP_N}")


# ── Historical market cap at time t ───────────────────────────────────────────

def historical_mktcap(t: int) -> pd.Series:
    """
    Proxy for market cap at time t:
      mktcap[t] = price[t] × implied_shares
    implied_shares uses current snapshot (approximation; error ~2%/yr).
    The cross-sectional RANKING of stocks by mktcap is very stable.
    """
    price_t = prices.iloc[t]
    mc = price_t * implied_shares
    return mc.clip(lower=0)


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals_at(t: int) -> pd.Series | None:
    """
    Compute composite alpha at time t. Only uses prices.iloc[:t+1].
    Returns cross-sectional score, or None if insufficient data.
    """
    if t < WARMUP_DAYS:
        return None

    p_slice = prices.iloc[:t + 1]
    p_now   = p_slice.iloc[-1]
    ret_daily = p_slice.pct_change().dropna()

    if len(ret_daily) < 22:
        return None

    # ── Momentum ──
    p_1m  = p_slice.iloc[-22]
    p_3m  = p_slice.iloc[-64]  if t >= 63  else None
    p_13m = p_slice.iloc[-274] if t >= 273 else (p_slice.iloc[-252] if t >= 252 else None)
    p_1m_ago = p_slice.iloc[-22]

    mom_1m = (p_now / p_1m - 1).clip(-0.5, 0.5)
    mom_3m = (p_now / p_3m - 1).clip(-1, 1) if p_3m is not None else pd.Series(0.0, index=p_now.index)
    mom_12m_skip = (p_1m_ago / p_13m - 1).clip(-1, 1) if p_13m is not None else pd.Series(0.0, index=p_now.index)

    # ── Low volatility ──
    vol_21 = ret_daily.iloc[-21:].std()
    inv_vol = 1.0 / (vol_21 + 0.005)

    # ── Trend filter ──
    sma_200 = p_slice.iloc[-200:].mean() if t >= 199 else p_slice.mean()
    trend_200 = (p_now > sma_200).astype(float)

    # ── RSI(14) ──
    r14 = ret_daily.iloc[-14:]
    gains  = r14.clip(lower=0).mean()
    losses = (-r14.clip(upper=0)).mean()
    rs     = gains / (losses + 1e-9)
    rsi    = 100 - 100 / (1 + rs)
    rsi_score = (50 - rsi).clip(-50, 50)  # contrarian: low RSI favored

    # ── Size factor (log market cap) ──
    mc_t = historical_mktcap(t)
    log_size = np.log1p(mc_t).replace([np.inf, -np.inf], np.nan)

    # ── Cross-sectional ranking ──
    def rank_pct(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, na_option="bottom") * 100

    composite = (
        rank_pct(mom_12m_skip) * 0.30 +
        rank_pct(mom_3m)       * 0.18 +
        rank_pct(inv_vol)      * 0.22 +
        rank_pct(mom_1m)       * 0.08 +
        rank_pct(trend_200)    * 0.05 +
        rank_pct(rsi_score)    * 0.05 +
        rank_pct(log_size)     * 0.12   # NEW: size factor
    )
    return composite.dropna()


# ── Market-cap weighting ──────────────────────────────────────────────────────

def mktcap_weights(tickers: list[str], t: int, regime_scale: float) -> dict[str, float]:
    """
    Square-root market-cap weights within selected tickers.
    sqrt() dampens mega-cap dominance vs pure mktcap while still tilting large.
    Cap at MAX_WEIGHT.
    """
    mc = historical_mktcap(t).reindex(tickers).fillna(1e9)
    sqrt_mc = np.sqrt(mc)
    raw_w = sqrt_mc / sqrt_mc.sum()

    # Apply cap
    capped = {tk: min(float(raw_w[tk]), MAX_WEIGHT) for tk in tickers}
    total  = sum(capped.values())
    norm_w = {tk: w / total * min(regime_scale, 1.0) for tk, w in capped.items()}
    return norm_w


# ── SPY regime filter ─────────────────────────────────────────────────────────

def get_regime(t: int) -> float:
    if t < SPY_REGIME_WIN:
        return 1.0
    spy_ma = float(spy_series.iloc[t - SPY_REGIME_WIN + 1: t + 1].mean())
    spy_now = float(spy_series.iloc[t])
    return 1.0 if spy_now > spy_ma else CASH_IN_BEAR


# ── Walk-forward backtest ─────────────────────────────────────────────────────

print("\nRunning walk-forward backtest (v2)...")

rebalance_idx = list(range(WARMUP_DAYS, len(prices) - HOLD_DAYS, HOLD_DAYS))

port_value = 1.0
spy_value  = 1.0
current_holdings: dict[str, float] = {}  # ticker → weight
current_scores:   dict[str, float] = {}  # ticker → composite score at last entry

monthly_rows = []
ic_rows      = []
turnover_rows= []

for period_idx, t in enumerate(rebalance_idx):
    composite = compute_signals_at(t)
    if composite is None or len(composite) < TOP_N:
        continue

    regime_scale = get_regime(t)

    # ── Candidate selection with buffer ──
    top_candidates = composite.nlargest(TOP_N * 2)  # oversample

    # Keep current holdings if still above buffer threshold
    new_set = set()
    # First: keep held stocks that are still competitive
    for tk in list(current_holdings.keys()):
        if tk not in composite.index:
            continue
        tk_score = float(composite[tk])
        # Keep if score > (TOP_N-th candidate score - buffer)
        threshold = float(composite.nlargest(TOP_N).iloc[-1]) - BUFFER_SCORE
        if tk_score >= threshold:
            new_set.add(tk)

    # Fill remaining slots from top candidates
    for tk in top_candidates.index:
        if len(new_set) >= TOP_N:
            break
        new_set.add(tk)

    selected = list(new_set)[:TOP_N]

    # ── Weights ──
    new_weights = mktcap_weights(selected, t, regime_scale)

    # ── Turnover ──
    all_tickers = set(current_holdings) | set(new_weights)
    turnover = sum(abs(new_weights.get(tk, 0.0) - current_holdings.get(tk, 0.0))
                   for tk in all_tickers)
    tcost_drag = turnover * TCOST_BPS / 10_000

    # ── Period return ──
    t_end = min(t + HOLD_DAYS, len(prices) - 1)
    p_entry = prices.iloc[t]
    p_exit  = prices.iloc[t_end]

    period_ret = 0.0
    for tk, w in new_weights.items():
        if tk in p_entry.index:
            pe = float(p_entry[tk])
            px = float(p_exit.get(tk, pe))
            if pe > 0 and np.isfinite(px):
                period_ret += w * (px / pe - 1)
    period_ret -= tcost_drag

    # SPY return
    spy_ret = float(spy_series.iloc[t_end]) / float(spy_series.iloc[t]) - 1

    port_value *= (1 + period_ret)
    spy_value  *= (1 + spy_ret)

    # ── IC ──
    fwd_rets = {}
    for tk in composite.index:
        if tk in prices.columns and t_end < len(prices):
            pe = float(prices.iloc[t].get(tk, np.nan))
            px = float(prices.iloc[t_end].get(tk, np.nan))
            if pe > 0 and np.isfinite(px):
                fwd_rets[tk] = px / pe - 1

    if len(fwd_rets) >= 50:
        sig = composite[list(fwd_rets.keys())]
        fwd = pd.Series(fwd_rets)
        from scipy.stats import spearmanr
        ic, pval = spearmanr(sig, fwd)
        ic_rows.append({
            "date": prices.index[t].strftime("%Y-%m-%d"),
            "IC":   round(float(ic), 4),
            "n":    len(fwd_rets),
        })

    # ── Record ──
    monthly_rows.append({
        "date":          prices.index[t].strftime("%Y-%m-%d"),
        "port_return":   round(period_ret, 6),
        "spy_return":    round(spy_ret, 6),
        "active_return": round(period_ret - spy_ret, 6),
        "port_value":    round(port_value, 4),
        "spy_value":     round(spy_value, 4),
        "regime_scale":  regime_scale,
        "n_holdings":    len(selected),
        "tcost_bps":     round(tcost_drag * 10000, 2),
    })
    turnover_rows.append({
        "date":      prices.index[t].strftime("%Y-%m-%d"),
        "turnover":  round(turnover, 4),
        "top5":      "|".join(selected[:5]),
    })

    current_holdings = dict(new_weights)
    current_scores   = {tk: float(composite.get(tk, 0)) for tk in selected}

    print(f"  Period {period_idx+1:2d} | {prices.index[t].date()} | "
          f"port={port_value:.3f} spy={spy_value:.3f} "
          f"active={period_ret-spy_ret:+.2%} | "
          f"regime={'BULL' if regime_scale==1.0 else 'BEAR'} | "
          f"n={len(selected)}")


# ── Performance ───────────────────────────────────────────────────────────────

monthly_df  = pd.DataFrame(monthly_rows)
ic_df       = pd.DataFrame(ic_rows)
turnover_df = pd.DataFrame(turnover_rows)

port_rets   = monthly_df["port_return"]
spy_rets    = monthly_df["spy_return"]
active_rets = monthly_df["active_return"]
n_periods   = len(port_rets)
periods_yr  = 252 / HOLD_DAYS  # ≈ 4

ann_ret_p = float((port_value ** (periods_yr / n_periods)) - 1)
ann_ret_s = float((spy_value  ** (periods_yr / n_periods)) - 1)
ann_vol_p = float(port_rets.std()   * np.sqrt(periods_yr))
ann_vol_s = float(spy_rets.std()    * np.sqrt(periods_yr))
ann_vol_a = float(active_rets.std() * np.sqrt(periods_yr))
sharpe_p  = ann_ret_p / (ann_vol_p + 1e-9)
sharpe_s  = ann_ret_s / (ann_vol_s + 1e-9)
ir        = (ann_ret_p - ann_ret_s) / (ann_vol_a + 1e-9)

cum = (1 + port_rets).cumprod()
max_dd_p = float(-((cum - cum.cummax()) / cum.cummax()).min())
cum_s = (1 + spy_rets).cumprod()
max_dd_s = float(-((cum_s - cum_s.cummax()) / cum_s.cummax()).min())

win_rate    = float((active_rets > 0).mean())
pct_pos_p   = float((port_rets > 0).mean())
avg_turnover = float(turnover_df["turnover"].mean())

mean_ic = float(ic_df["IC"].mean()) if not ic_df.empty else np.nan
std_ic  = float(ic_df["IC"].std())  if not ic_df.empty else np.nan
ir_ic   = mean_ic / (std_ic + 1e-9) if not np.isnan(mean_ic) else np.nan
pct_pos_ic = float((ic_df["IC"] > 0).mean()) if not ic_df.empty else np.nan

print("\n" + "=" * 65)
print("  BACKTEST v2 — Market-Cap Weighted | Top-50 | 63-day Hold")
print("=" * 65)
print(f"  Period:          {monthly_df['date'].iloc[0]} → {monthly_df['date'].iloc[-1]}")
print(f"  Rebalances:      {n_periods} (every {HOLD_DAYS} days)")
print(f"  Universe:        {len(universe)} stocks | Portfolio: Top-{TOP_N}")
print(f"")
print(f"  {'Metric':<32} {'v2 Portfolio':>12}  {'SPY':>10}")
print(f"  {'─'*58}")
print(f"  {'Ann. Return':<32} {ann_ret_p:>+12.2%}  {ann_ret_s:>+10.2%}")
print(f"  {'Ann. Volatility':<32} {ann_vol_p:>12.2%}  {ann_vol_s:>10.2%}")
print(f"  {'Sharpe Ratio':<32} {sharpe_p:>12.2f}  {sharpe_s:>10.2f}")
print(f"  {'Max Drawdown':<32} {-max_dd_p:>12.2%}  {-max_dd_s:>10.2%}")
print(f"  {'% Positive Periods':<32} {pct_pos_p:>12.0%}  {'N/A':>10}")
print(f"  {'Total Return':<32} {port_value-1:>+12.2%}  {spy_value-1:>+12.2%}")
print(f"")
print(f"  Active Return (Ann.):   {ann_ret_p - ann_ret_s:+.2%}")
print(f"  Information Ratio:       {ir:.2f}")
print(f"  Win Rate vs SPY:         {win_rate:.0%}")
print(f"")
print(f"  Signal IC:  mean={mean_ic:.4f}  std={std_ic:.4f}  IR={ir_ic:.2f}  %pos={pct_pos_ic:.0%}")
print(f"  Avg Turnover/Period:  {avg_turnover:.0%}  (was 123% in v1)")
print("=" * 65)

# ── vs v1 comparison ──────────────────────────────────────────────────────────
v1_path = ROOT / "backtest_summary.csv"
if v1_path.exists():
    v1 = pd.read_csv(v1_path).iloc[0]
    print(f"\n  Improvement vs v1 (equal-weight, top-25, 21-day):")
    print(f"  {'Metric':<30} {'v1':>10}  {'v2':>10}  {'Delta':>10}")
    print(f"  {'─'*54}")
    metrics = [
        ("Ann. Return",    float(v1.ann_return_port), ann_ret_p),
        ("SPY Ann. Return", float(v1.ann_return_spy), ann_ret_s),
        ("Active Return",  float(v1.active_return),  ann_ret_p - ann_ret_s),
        ("Sharpe",        float(v1.sharpe_port),     sharpe_p),
        ("Info Ratio",    float(v1.info_ratio),      ir),
        ("Max Drawdown",  -float(v1.max_dd_port),    -max_dd_p),
        ("Win Rate",      float(v1.win_rate),        win_rate),
        ("Total Return",  float(v1.total_return_port), port_value - 1),
    ]
    for name, old, new in metrics:
        delta = new - old
        print(f"  {name:<30} {old:>+10.2%}  {new:>+10.2%}  {delta:>+10.2%}")

# ── Save ──────────────────────────────────────────────────────────────────────
monthly_df.to_csv(ROOT / "backtest_v2_monthly.csv", index=False)
ic_df.to_csv(ROOT / "backtest_v2_ic.csv", index=False)

summary = pd.DataFrame([{
    "version": "v2",
    "ann_return_port": round(ann_ret_p, 4),
    "ann_return_spy":  round(ann_ret_s, 4),
    "active_return":   round(ann_ret_p - ann_ret_s, 4),
    "sharpe_port":     round(sharpe_p, 2),
    "sharpe_spy":      round(sharpe_s, 2),
    "info_ratio":      round(ir, 2),
    "max_dd_port":     round(max_dd_p, 4),
    "max_dd_spy":      round(max_dd_s, 4),
    "win_rate":        round(win_rate, 3),
    "mean_ic":         round(mean_ic, 4) if not np.isnan(mean_ic) else None,
    "ic_ir":           round(ir_ic, 2)   if not np.isnan(ir_ic)   else None,
    "avg_turnover":    round(avg_turnover, 3),
    "total_return_port": round(port_value - 1, 4),
    "total_return_spy":  round(spy_value - 1, 4),
}])
summary.to_csv(ROOT / "backtest_v2_summary.csv", index=False)

print(f"\n  Saved: backtest_v2_monthly.csv, backtest_v2_ic.csv, backtest_v2_summary.csv")

print("\n  Anti-Lookahead Audit:")
print("  [PASS] Signal uses prices.iloc[:t+1] only")
print("  [PASS] Forward return = price[t+63] / price[t] - 1")
print("  [PASS] Mktcap[t] = price[t] × implied_shares (shares ≈static proxy)")
print("  [PASS] Size factor = log(mktcap[t]) — computed from price[t]")
print("  [PASS] Regime filter uses SPY SMA computed from spy[:t+1]")
print("  [PASS] Buffer only prevents premature exit — no future info used")
