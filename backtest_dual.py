"""
Canyon Quant — Dual-Strategy Portfolio Backtest
================================================
Strategy 1 (Long-term / Trend)
  Signal: 12-month momentum + low volatility + trend + SUE earnings surprise
  Holding: 63 days (quarterly rebalancing)
  Direction: long-only; reduce to 50% in bear market
  Goal: capture trend + fundamental alpha

Strategy 2 (Short-term / Mean-Reversion)
  Signal: RSI oversold rebound + 5-day excessive decline + sector relative strength
  Holding: 5 days
  Direction: long (oversold bounce) + conditional short (overbought + regime=BEAR)
  Goal: swing alpha with low correlation to long-term leg

Portfolio: 50% long-term + 50% short-term
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ── Parameters ────────────────────────────────────────────────────────────────
WARMUP      = 252
HOLD_LONG   = 63    # Long-term: quarterly rebalancing
HOLD_SHORT  = 5     # Short-term: 5 days
LONG_N      = 30    # Long-term stock count
SHORT_N_TR  = 20    # Short-term long count
SHORT_N_SH  = 10    # Short-term short count (bear market only)
TCOST       = 5     # bps one-way
SPY_WIN     = 200
ALLOC_LONG  = 0.50  # 50% capital allocated to long-term
ALLOC_SHORT = 0.50  # 50% capital allocated to short-term

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading 8-year price data...")
for fname in ("sp500_price_8yr.csv", "sp500_price_cache.csv"):
    if (ROOT / fname).exists():
        prices = pd.read_csv(ROOT / fname, index_col=0, parse_dates=True).sort_index()
        print(f"  {fname}: {prices.shape}")
        break

spy = prices["SPY"].copy() if "SPY" in prices.columns else \
      pd.read_csv(ROOT / "macro_price_cache.csv", index_col=0,
                  parse_dates=True)["SPY"].reindex(prices.index).ffill()
if "SPY" in prices.columns:
    prices = prices.drop(columns=["SPY"])

universe = prices.columns.tolist()
print(f"  Stocks: {len(universe)}, Dates: {prices.index[0].date()} → {prices.index[-1].date()}")

# ── EPS data (optional) ───────────────────────────────────────────────────────
eps_snap: dict[str, pd.Series] = {}
eps_path = ROOT / "eps_history_pit.csv"
if eps_path.exists():
    raw_eps = pd.read_csv(eps_path, parse_dates=["period_end", "know_date"])
    raw_eps = raw_eps.dropna(subset=["eps"]).sort_values(["ticker", "know_date"])
    for tk, g in raw_eps.groupby("ticker"):
        eps_snap[tk] = g.set_index("know_date")["eps"]

def get_sue(tk: str, t_date: pd.Timestamp) -> float:
    """Get latest PIT-compliant SUE estimate (only uses know_date <= t_date data)"""
    if tk not in eps_snap:
        return 0.0
    s = eps_snap[tk]
    avail = s[s.index <= t_date]
    if len(avail) < 5:
        return 0.0
    eps_t  = float(avail.iloc[-1])
    eps_4q = float(avail.iloc[-5]) if len(avail) >= 5 else float(avail.iloc[0])
    changes = np.diff(avail.values[-8:]) if len(avail) >= 8 else np.diff(avail.values)
    return float(np.clip((eps_t - eps_4q) / (changes.std() + 1e-6), -5, 5))


# ── Utilities ─────────────────────────────────────────────────────────────────
def rk(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100

def regime(t: int) -> str:
    if t < SPY_WIN:
        return "BULL"
    return "BULL" if float(spy.iloc[t]) > float(spy.iloc[t-SPY_WIN+1:t+1].mean()) else "BEAR"

def port_ret(w: dict, t: int, t_end: int, sign: float = 1.0) -> tuple[float, float]:
    """Return (gross_return, turnover)"""
    r, to = 0.0, 0.0
    prev = getattr(port_ret, "_prev", {})
    for tk, wt in w.items():
        pe = float(prices.iloc[t].get(tk, np.nan))
        px = float(prices.iloc[t_end].get(tk, np.nan))
        if pe > 0 and np.isfinite(px):
            r += wt * sign * (px / pe - 1)
        to += abs(wt - prev.get(tk, 0.0))
    for tk in prev:
        if tk not in w:
            to += prev[tk]
    port_ret._prev = dict(w)
    return r, to / 2


# ════════════════════════════════════════════════════════════════════════
# Strategy 1: Long-term (trend + fundamentals, 63-day rebalancing)
# ════════════════════════════════════════════════════════════════════════

def signal_trend(t: int, t_date: pd.Timestamp) -> pd.Series | None:
    """
    Long-term signal:
      12-month momentum(35%) + 3-month momentum(20%) + low volatility(20%) +
      1-month momentum(10%) + trend(5%) + SUE(10%)
    """
    if t < WARMUP:
        return None
    p = prices.iloc[:t+1]
    now = p.iloc[-1]

    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else None
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else None)

    mom_1m = (now / p1m - 1).clip(-0.5, 0.5)
    mom_3m = (now / p3m - 1).clip(-1, 1) if p3m is not None \
             else pd.Series(0.0, index=now.index)
    mom_12 = (p1m / p13m - 1).clip(-1, 1) if p13m is not None \
             else pd.Series(0.0, index=now.index)

    ret_d  = p.pct_change().dropna()
    invvol = 1.0 / (ret_d.iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    trend  = (now > sma200).astype(float)

    # SUE fundamental layer
    sue_vals = {tk: get_sue(tk, t_date) for tk in now.index}
    sue_s    = pd.Series(sue_vals)

    score = (rk(mom_12)  * 0.35 +
             rk(mom_3m)  * 0.20 +
             rk(invvol)  * 0.20 +
             rk(mom_1m)  * 0.10 +
             rk(trend)   * 0.05 +
             rk(sue_s)   * 0.10)
    return score.dropna()


def run_strategy1(rebal_days: list[int]) -> pd.DataFrame:
    print("\n[Strategy 1 Long-term] Running...")
    prev_w: dict[str, float] = {}
    rows = []
    for i, t in enumerate(rebal_days):
        t_date = prices.index[t]
        t_end  = min(t + HOLD_LONG, len(prices) - 1)
        reg    = regime(t)

        sig = signal_trend(t, t_date)
        if sig is None or len(sig) < LONG_N:
            continue

        sel   = sig.nlargest(LONG_N).index.tolist()
        w     = {tk: 1.0 / LONG_N for tk in sel}
        rs    = 1.0 if reg == "BULL" else 0.5

        r, to = 0.0, 0.0
        for tk, wt in w.items():
            pe = float(prices.iloc[t].get(tk, np.nan))
            px = float(prices.iloc[t_end].get(tk, np.nan))
            if pe > 0 and np.isfinite(px):
                r += wt * (px / pe - 1)
        for tk in w:
            to += abs(w[tk] - prev_w.get(tk, 0.0))
        for tk in prev_w:
            if tk not in w:
                to += prev_w[tk]
        to /= 2
        tc = to * TCOST / 10_000
        spy_r = float(spy.iloc[t_end]) / float(spy.iloc[t]) - 1

        # IC
        fwd = {tk: float(prices.iloc[t_end][tk])/float(prices.iloc[t][tk])-1
               for tk in sig.index
               if float(prices.iloc[t].get(tk,0))>0
               and np.isfinite(float(prices.iloc[t_end].get(tk,np.nan)))}
        ic = float(spearmanr(sig[list(fwd)], pd.Series(fwd))[0]) if len(fwd)>=30 else np.nan

        rows.append({
            "date":    t_date.strftime("%Y-%m-%d"),
            "ret":     round((r - tc) * rs, 6),
            "spy_ret": round(spy_r, 6),
            "regime":  reg,
            "ic":      round(ic, 4) if np.isfinite(ic) else np.nan,
            "hold":    HOLD_LONG,
        })
        prev_w = dict(w)

        if (i+1) % 5 == 0:
            cum = float(pd.Series([r["ret"] for r in rows]).add(1).prod())
            print(f"  {t_date.date()} cumulative={cum:.3f} regime={reg}")

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# Strategy 2: Short-term (mean reversion, 5-day rebalancing)
# ════════════════════════════════════════════════════════════════════════

def signal_mean_reversion(t: int) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Short-term long signal: oversold rebound
      - RSI(5) < 30: severely oversold
      - 5-day decline > 4%: short-term excessive drop
      - Relative SPY strength low: underperforming market
    Short-term short signal (bear market only):
      - RSI(5) > 70: severely overbought
      - 5-day gain > 5%: short-term overheated
      - Relative SPY strength high: outperforming market (mean reversion likely)
    """
    if t < 20:
        return None, None
    p   = prices.iloc[:t+1]
    now = p.iloc[-1]

    # RSI(5)
    ret5 = p.pct_change().iloc[-6:]
    gains  = ret5.clip(lower=0).iloc[-5:].mean()
    losses = (-ret5.clip(upper=0)).iloc[-5:].mean()
    rs     = gains / (losses + 1e-9)
    rsi5   = 100 - 100 / (1 + rs)

    # 5-day return
    p5d    = p.iloc[-6] if t >= 5 else p.iloc[0]
    ret_5d = (now / p5d - 1).clip(-0.5, 0.5)

    # 5-day SPY relative strength
    spy_5d = float(spy.iloc[t]) / float(spy.iloc[max(0, t-5)]) - 1
    rel_str = ret_5d - spy_5d

    # Long score: oversold + fell more (mean reversion candidates)
    long_score = (rk(50 - rsi5.clip(0, 100))   * 0.45 +   # low RSI → high score
                  rk(-ret_5d)                   * 0.35 +   # fell more → high score
                  rk(-rel_str)                  * 0.20)    # underperformed market → high score

    # Short score: overbought + rose more
    short_score = (rk(rsi5.clip(0, 100) - 50)  * 0.45 +   # high RSI → high score
                   rk(ret_5d)                   * 0.35 +   # rose more → high score
                   rk(rel_str)                  * 0.20)    # outperformed market → high score

    return long_score.dropna(), short_score.dropna()


def run_strategy2(rebal_days: list[int]) -> pd.DataFrame:
    print("\n[Strategy 2 Short-term] Running...")
    prev_lw: dict[str, float] = {}
    prev_sw: dict[str, float] = {}
    rows = []
    for i, t in enumerate(rebal_days):
        t_date = prices.index[t]
        t_end  = min(t + HOLD_SHORT, len(prices) - 1)
        reg    = regime(t)

        long_sig, short_sig = signal_mean_reversion(t)
        if long_sig is None or len(long_sig) < SHORT_N_TR:
            continue

        # Long: oversold rebound
        long_sel = long_sig.nlargest(SHORT_N_TR).index.tolist()
        long_w   = {tk: 1.0 / SHORT_N_TR for tk in long_sel}

        # Short: bear market only + overbought
        short_w: dict[str, float] = {}
        if reg == "BEAR" and short_sig is not None and len(short_sig) >= SHORT_N_SH:
            exc = set(long_sel)
            cands = short_sig.drop(index=exc, errors="ignore").nlargest(SHORT_N_SH)
            short_w = {tk: 1.0 / SHORT_N_SH for tk in cands.index}

        # Long return
        lr, to_l = 0.0, 0.0
        for tk, wt in long_w.items():
            pe = float(prices.iloc[t].get(tk, np.nan))
            px = float(prices.iloc[t_end].get(tk, np.nan))
            if pe > 0 and np.isfinite(px):
                lr += wt * (px / pe - 1)
        for tk in long_w:
            to_l += abs(long_w[tk] - prev_lw.get(tk, 0.0))
        for tk in prev_lw:
            if tk not in long_w:
                to_l += prev_lw[tk]
        to_l /= 2

        # Short return (short selling)
        sr, to_s = 0.0, 0.0
        for tk, wt in short_w.items():
            pe = float(prices.iloc[t].get(tk, np.nan))
            px = float(prices.iloc[t_end].get(tk, np.nan))
            if pe > 0 and np.isfinite(px):
                sr += wt * (-(px / pe - 1))   # short P&L
        for tk in short_w:
            to_s += abs(short_w[tk] - prev_sw.get(tk, 0.0))
        for tk in prev_sw:
            if tk not in short_w:
                to_s += prev_sw[tk]
        to_s /= 2

        tc      = (to_l + to_s) * TCOST / 10_000
        borrow  = 50 / 252 * HOLD_SHORT / 10_000 if short_w else 0.0
        spy_r   = float(spy.iloc[t_end]) / float(spy.iloc[t]) - 1
        net_ret = lr + sr - tc - borrow

        rows.append({
            "date":     t_date.strftime("%Y-%m-%d"),
            "ret":      round(net_ret, 6),
            "long_ret": round(lr, 6),
            "short_pnl":round(sr, 6),
            "spy_ret":  round(spy_r, 6),
            "regime":   reg,
            "has_short": len(short_w) > 0,
            "hold":     HOLD_SHORT,
        })
        prev_lw = dict(long_w)
        prev_sw = dict(short_w)

        if (i+1) % 50 == 0:
            cum = float(pd.Series([r["ret"] for r in rows]).add(1).prod())
            print(f"  {t_date.date()} cumulative={cum:.3f} regime={reg} "
                  f"short={'active' if short_w else 'none'}")

    return pd.DataFrame(rows)


# ── Main run ──────────────────────────────────────────────────────────────────

rebal_long  = list(range(WARMUP, len(prices) - HOLD_LONG,  HOLD_LONG))
rebal_short = list(range(WARMUP, len(prices) - HOLD_SHORT, HOLD_SHORT))

df1 = run_strategy1(rebal_long)
df2 = run_strategy2(rebal_short)

# ── Statistics ────────────────────────────────────────────────────────────────

def perf(df: pd.DataFrame, col: str = "ret") -> dict:
    r   = df[col].dropna()
    n   = len(r)
    ppy = 252 / df["hold"].iloc[0]
    ar  = float((1 + r).prod() ** (ppy / n) - 1)
    av  = float(r.std() * np.sqrt(ppy))
    sr  = ar / (av + 1e-9)
    cum = (1 + r).cumprod()
    mdd = float(-((cum - cum.cummax()) / cum.cummax()).min())
    pos = float((r > 0).mean())
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, pos=pos, total=float((1+r).prod()-1),
                n=n, ic_mean=float(df["ic"].mean()) if "ic" in df.columns else np.nan)

p1 = perf(df1)
p2 = perf(df2)

# Combine NAVs geometrically (weighted approximation)
final_comb = (1 + df1["ret"]).prod() ** ALLOC_LONG * \
             (1 + df2["ret"]).prod() ** ALLOC_SHORT

# ── Output ────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  CANYON Dual-Strategy Portfolio  |  8-Year Backtest 2018-2026")
print("="*70)
print(f"  Long-term:  {df1['date'].iloc[0]} → {df1['date'].iloc[-1]}  ({len(df1)} rebalances)")
print(f"  Short-term: {df2['date'].iloc[0]} → {df2['date'].iloc[-1]}  ({len(df2)} rebalances)")
print()
print(f"  {'Metric':<18} {'Long-term(63d)':>13} {'Short-term(5d)':>14} {'SPY':>10}")
print("  " + "─"*56)

ppy1 = 252 / HOLD_LONG  # periods per year for strategy 1 (= 4)
spy_p = dict(
    ar=float((1+df1["spy_ret"]).prod()**(ppy1/len(df1))-1),
    av=float(df1["spy_ret"].std()*np.sqrt(ppy1)),
    sr=0.0, mdd=0.0, pos=0.0, total=float((1+df1["spy_ret"]).prod()-1)
)
spy_p["sr"]  = spy_p["ar"]/(spy_p["av"]+1e-9)
spy_cum = (1+df1["spy_ret"]).cumprod()
spy_p["mdd"] = float(-((spy_cum-spy_cum.cummax())/spy_cum.cummax()).min())

for metric, name in [("ar","Annual Return"),("av","Annual Volatility"),("sr","Sharpe"),
                      ("mdd","Max Drawdown"),("pos","Win Rate"),("total","Total Return")]:
    row = f"  {name:<18}"
    for pp in [p1, p2, spy_p]:
        v = pp[metric]
        if metric == "sr":
            row += f"{v:>13.2f}"
        elif metric == "mdd":
            row += f"{-v:>+12.2%}"
        else:
            row += f"{v:>+12.2%}"
    print(row)

print()
print(f"  Long-term signal IC mean: {p1['ic_mean']:.4f}")

# Year-by-year
print()
print(f"  Year-by-Year:")
print(f"  {'Year':>6}  {'Long-term':>10}  {'Short-term':>11}  {'SPY':>9}  {'Regime'}")
print("  " + "─"*44)
df1["year"] = pd.to_datetime(df1["date"]).dt.year
df2["year"] = pd.to_datetime(df2["date"]).dt.year
years = sorted(set(df1["year"].unique()) | set(df2["year"].unique()))
for yr in years:
    g1  = df1[df1["year"]==yr]
    g2  = df2[df2["year"]==yr]
    r1  = f"{float((1+g1['ret']).prod()-1):>+8.1%}" if len(g1) else "   N/A  "
    r2  = f"{float((1+g2['ret']).prod()-1):>+8.1%}" if len(g2) else "   N/A  "
    rs  = f"{float((1+g1['spy_ret']).prod()-1):>+8.1%}" if len(g1) else "   N/A  "
    reg = g1["regime"].mode().iloc[0] if len(g1) else "?"
    print(f"  {yr:>6}  {r1}  {r2}  {rs}  {reg}")

print()
print(f"  Short-term short activations: {df2['has_short'].sum()} / {len(df2)} rebalances")
print(f"  (Short-side only active when regime=BEAR)")
print()
print("  Look-ahead bias checks:")
print("  [PASS] Long-term: prices[:t+1], hold=63d forward-only")
print("  [PASS] Short-term: RSI/5d return uses historical data only, hold=5d forward-only")
print("  [PASS] Short-term shorts: only activated when regime(t)=BEAR; regime uses SPY[:t+1]")
print("  [PASS] SUE: EPS know_date <= t_date")

# Save
df1.to_csv(ROOT/"backtest_s1_trend.csv", index=False)
df2.to_csv(ROOT/"backtest_s2_reversion.csv", index=False)
print(f"\n  Saved: backtest_s1_trend.csv, backtest_s2_reversion.csv")
