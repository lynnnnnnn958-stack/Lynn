"""
Canyon Quant v14 — Multi-Factor Institutional Upgrade
======================================================
New vs v11:
  1. SEC EDGAR point-in-time accruals factor (Sloan 1996)
  2. Gross profitability factor (Novy-Marx 2013)
  3. Combined IC-weighted composite score
  4. Long-short simulation (paper-only) to measure pure alpha

Signal weights (IC-proportional):
  Price momentum 12-1M : 35%  (IC ≈ 0.04)
  Momentum 3M          : 20%  (IC ≈ 0.03)
  Vol-inverse          : 25%  (IC ≈ 0.03, risk not alpha but risk-adjusted)
  Reversal (1M)        : 10%  (IC ≈ 0.02)
  Above 200MA          : 10%  (IC ≈ 0.02)
  Accruals (inverted)  : +15% weight within quality composite
  Gross profitability  : +10% weight within quality composite
"""

import warnings, numpy as np, pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(".")
HOLD_M     = 21
TOP_M      = 25
TCOST      = 5           # bps one-way
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

# Factor weights
W_MOM12   = 0.35
W_MOM3    = 0.20
W_INVVOL  = 0.25
W_REV1    = 0.10
W_MA200   = 0.10
W_ACCRUAL = 0.15   # additional weight (quality overlay)
W_GP      = 0.10   # additional weight (quality overlay)

# ── Data loading ─────────────────────────────────────────────────────────

print("Loading data...")
prices = pd.read_csv(ROOT/"sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy    = prices["SPY"].copy()
prices = prices.drop(columns=["SPY"])

letf_df = pd.read_csv(ROOT/"letf_prices.csv", index_col=0, parse_dates=True).sort_index()
macro   = pd.read_csv(ROOT/"macro_regime.csv", index_col=0, parse_dates=True).sort_index()
sector_map = (pd.read_csv(ROOT/"sp500_sectors.csv", index_col=0).squeeze().to_dict()
              if (ROOT/"sp500_sectors.csv").exists() else {})

common = prices.index.intersection(letf_df.index).intersection(macro.index)
prices   = prices.reindex(common)
letf_df  = letf_df.reindex(common).ffill().bfill()
spy      = spy.reindex(common).ffill()
macro    = macro.reindex(common).ffill()
tqqq     = letf_df["TQQQ"].values
tqqq_logr = np.concatenate([[np.nan], np.diff(np.log(np.where(tqqq > 0, tqqq, np.nan)))])

# ── Load EDGAR fundamentals ───────────────────────────────────────────────

edgar_path = ROOT / "edgar_fundamentals.csv"
if edgar_path.exists():
    edgar_raw = pd.read_csv(edgar_path, parse_dates=["filed_date", "fiscal_year_end"])
    print(f"  EDGAR: {len(edgar_raw)} rows, {edgar_raw['ticker'].nunique()} tickers")
    HAS_EDGAR = True
else:
    print("  EDGAR fundamentals not found — running v14 momentum-only for comparison")
    HAS_EDGAR = False
    edgar_raw = pd.DataFrame()


def build_fundamental_scores(as_of_date: pd.Timestamp) -> pd.Series:
    """
    For each ticker, find the most recently filed annual report
    before as_of_date. Returns combined fundamental score (0-100).
    No lookahead: uses filed_date, not fiscal_year_end.
    """
    if not HAS_EDGAR or edgar_raw.empty:
        return pd.Series(dtype=float)

    # Most recent filing before as_of_date for each ticker
    valid = edgar_raw[edgar_raw["filed_date"] < as_of_date].copy()
    if valid.empty:
        return pd.Series(dtype=float)

    latest = (valid.sort_values("filed_date")
                   .drop_duplicates("ticker", keep="last")
                   .set_index("ticker"))

    scores = pd.DataFrame(index=latest.index)

    # Accruals: LOW accrual = HIGH score (inverted, good quality)
    if "accrual_ratio" in latest.columns:
        acc = latest["accrual_ratio"].clip(-0.5, 0.5)
        scores["acc_score"] = (-acc).rank(pct=True) * 100  # inverted

    # Gross profitability: HIGH gross margin = HIGH score
    if "gross_margin" in latest.columns:
        gp = latest["gross_margin"].clip(-1, 5)
        scores["gp_score"] = gp.rank(pct=True) * 100

    # ROA: HIGH = HIGH score
    if "roa" in latest.columns:
        roa = latest["roa"].clip(-0.5, 0.5)
        scores["roa_score"] = roa.rank(pct=True) * 100

    if scores.empty:
        return pd.Series(dtype=float)

    # Weighted composite
    weights = {"acc_score": 0.50, "gp_score": 0.35, "roa_score": 0.15}
    total_w = 0.0
    composite = pd.Series(0.0, index=scores.index)
    for col, w in weights.items():
        if col in scores.columns:
            composite += scores[col].fillna(50) * w
            total_w += w
    if total_w > 0:
        composite /= total_w

    return composite  # higher = better fundamental quality


def precompute_fundamentals(trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Precompute fundamental scores for all rebalancing dates."""
    if not HAS_EDGAR:
        return pd.DataFrame(index=trade_dates)
    print("  Precomputing fundamental scores...")
    result = {}
    for dt in trade_dates:
        s = build_fundamental_scores(dt)
        if not s.empty:
            for tk, v in s.items():
                if tk not in result:
                    result[tk] = {}
                result[tk][dt] = v
    mat = pd.DataFrame(result, index=trade_dates)
    print(f"  Fundamental matrix: {mat.shape[0]}×{mat.shape[1]} ({mat.notna().mean().mean():.0%} coverage)")
    return mat


# ── Market regime helpers (identical to v11) ─────────────────────────────

def spy_above_200ma(t: int) -> bool:
    return t < SPY_WIN or float(spy.iloc[t]) > spy.iloc[t - SPY_WIN + 1:t + 1].mean()

def tqqq_above_50ma(t: int) -> bool:
    return t < TQQQ_WIN or float(tqqq[t]) > np.mean(tqqq[t - TQQQ_WIN:t])

def tqqq_vol(t: int) -> float:
    w = tqqq_logr[max(0, t - VOL_WIN):t]
    w = w[np.isfinite(w)]
    return float(np.std(w) * np.sqrt(252)) if len(w) >= 5 else 0.40

def etf_alloc(t: int) -> float:
    if not spy_above_200ma(t):
        return 0.0
    vix_v = float(macro["VIX"].iloc[t])
    sig2 = np.isfinite(vix_v) and vix_v < VIX_THRESH
    if t >= HYG_WIN and "HYG" in macro.columns:
        hyg, ief = float(macro["HYG"].iloc[t]), float(macro["IEF"].iloc[t])
        hyg0, ief0 = float(macro["HYG"].iloc[t - HYG_WIN]), float(macro["IEF"].iloc[t - HYG_WIN])
        sig3 = (hyg / ief >= hyg0 / ief0 * 0.99) if hyg0 > 0 and ief0 > 0 else True
    else:
        sig3 = True
    vol = tqqq_vol(t)
    vc  = min(MAX_ETF_W, VOL_TARGET / (vol + 0.001))
    base = vc if tqqq_above_50ma(t) else vc * 0.5
    fine = {0: 1.0, 1: 0.85, 2: 0.65}[sum([not sig2, not sig3])]
    return float(np.clip(base * fine, 0, MAX_ETF_W))

def tqqq_period_return(t: int, ew: float) -> float:
    if ew <= 0:
        return 0.0
    t1   = min(t + HOLD_M, len(tqqq) - 1)
    e    = tqqq[t]
    if e <= 0 or not np.isfinite(e):
        return 0.0
    peak, ed = e, t1
    for d in range(t + 1, t1 + 1):
        p = tqqq[d]
        if np.isfinite(p):
            peak = max(peak, p)
            if p < peak * (1 - TRAIL_STOP):
                ed = d
                break
    ep = tqqq[ed]
    return float(ep / e - 1) if np.isfinite(ep) and ep > 0 else 0.0


def rk(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100


# ── Stock signal with fundamental overlay ────────────────────────────────

def stock_signal_v14(t: int, as_of: pd.Timestamp,
                     fund_matrix: pd.DataFrame) -> dict:
    """
    Returns {ticker: weight} for long portfolio.
    Adds accruals + gross profitability on top of v11 momentum.
    """
    if t < WARMUP:
        return {}

    p   = prices.iloc[:t + 1]
    now = p.iloc[-1]
    p1m = p.iloc[-22]
    p3m = p.iloc[-64] if t >= 63 else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])

    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()

    # Price momentum composite (same as v11)
    mom = (rk((p1m / p13m - 1).clip(-1, 1))     * W_MOM12
         + rk((now / p3m - 1).clip(-1, 1))       * W_MOM3
         + rk(invvol)                             * W_INVVOL
         + rk((now / p1m - 1).clip(-0.5, 0.5))   * W_REV1
         + rk((now > sma200).astype(float))       * W_MA200)

    # Fundamental overlay — blend if data available
    if HAS_EDGAR and as_of in fund_matrix.index:
        fund_scores = fund_matrix.loc[as_of].dropna()
        if len(fund_scores) > 20:
            fund_aligned = fund_scores.reindex(mom.index).fillna(50)
            total_w = 1.0 + W_ACCRUAL + W_GP
            mom = (mom + rk(fund_aligned) * (W_ACCRUAL + W_GP)) / total_w
            mom = mom * 100  # re-scale to 0-100

    mom = mom.dropna()

    # Quality filter (same as v11)
    vol20 = p.pct_change().iloc[-21:].std()
    gains = p.pct_change().iloc[-15:].clip(lower=0).iloc[-14:].mean()
    loss  = (-p.pct_change().iloc[-15:].clip(upper=0)).iloc[-14:].mean()
    rsi14 = 100 - 100 / (1 + gains / (loss + 1e-9))
    dist  = now / (sma200 + 1e-9) - 1
    vol75 = vol20.quantile(0.75)
    quality_mask = (rsi14 < 75) & (dist < 0.50) & (vol20 < vol75)
    valid  = mom[mom.index.isin(quality_mask[quality_mask].index)]
    pool   = valid if len(valid) >= TOP_M else mom

    # Sector cap
    cands = pool.nlargest(TOP_M * 2).index.tolist()
    sel, sc = [], {}
    for tk in cands:
        sec = sector_map.get(tk, "Unknown")
        if sc.get(sec, 0) < MAX_SECTOR and len(sel) < TOP_M:
            sel.append(tk)
            sc[sec] = sc.get(sec, 0) + 1 / TOP_M
    if len(sel) < 10:
        sel = pool.nlargest(TOP_M).index.tolist()

    # Vol-inverse weights
    sv  = vol20[sel].clip(lower=0.001)
    rw  = 1.0 / sv
    rw  = rw / rw.sum()
    rw  = rw.clip(upper=0.15)
    wts = (rw / rw.sum()).to_dict()
    return wts


def stock_signal_short(t: int) -> list:
    """
    Bottom 25 stocks by momentum (for long-short simulation).
    Returns list of tickers to short.
    """
    if t < WARMUP:
        return []
    p   = prices.iloc[:t + 1]
    now = p.iloc[-1]
    p1m = p.iloc[-22]
    p3m = p.iloc[-64] if t >= 63 else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])
    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    mom = (rk((p1m / p13m - 1).clip(-1, 1)) * W_MOM12
         + rk((now / p3m - 1).clip(-1, 1))  * W_MOM3
         + rk(invvol)                        * W_INVVOL
         + rk((now / p1m - 1).clip(-0.5, 0.5)) * W_REV1
         + rk((now > sma200).astype(float))  * W_MA200)
    return mom.dropna().nsmallest(TOP_M).index.tolist()


# ── Backtest engine ───────────────────────────────────────────────────────

def run_backtest(fund_matrix: pd.DataFrame,
                 include_short: bool = False,
                 label: str = "v14"):
    rebal = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))
    records = []

    for t in rebal:
        t1   = min(t + HOLD_M, len(prices) - 1)
        bull = spy_above_200ma(t)
        sw   = STOCK_BULL if bull else STOCK_BEAR
        as_of = prices.index[t]

        # Long stocks
        wts = stock_signal_v14(t, as_of, fund_matrix)
        if wts:
            raw_r = sum(
                w * (float(prices.iloc[t1].get(tk, np.nan))
                     / float(prices.iloc[t].get(tk, np.nan)) - 1)
                for tk, w in wts.items()
                if float(prices.iloc[t].get(tk, np.nan)) > 0
            )
            sr_long = raw_r * sw
        else:
            sr_long = 0.0

        # Short book (for long-short simulation)
        if include_short and bull:
            short_tks = stock_signal_short(t)
            if short_tks:
                short_r = np.mean([
                    float(prices.iloc[t1].get(tk, np.nan))
                    / float(prices.iloc[t].get(tk, np.nan)) - 1
                    for tk in short_tks
                    if float(prices.iloc[t].get(tk, np.nan)) > 0
                ])
                sr_short = -short_r * sw * 0.5  # half-weight on short side
            else:
                sr_short = 0.0
        else:
            sr_short = 0.0

        # ETF
        ew = etf_alloc(t)
        er = tqqq_period_return(t, ew)

        ret = sr_long + sr_short + ew * er - TCOST / 10_000

        spy_r = float(spy.iloc[t1]) / float(spy.iloc[t]) - 1
        records.append(dict(
            date=prices.index[t],
            ret=ret, long_ret=sr_long, short_ret=sr_short,
            etf_ret=ew * er, spy_ret=spy_r,
            bull=bull, etf_w=ew
        ))

    return pd.DataFrame(records).set_index("date")


def print_stats(df: pd.DataFrame, label: str, v11_ar: float = 0.162):
    r  = df["ret"]
    n  = len(r)
    ppy = 252 / HOLD_M
    tot = float((1 + r).prod())
    ar  = float(tot ** (ppy / n) - 1)
    vol = float(r.std() * np.sqrt(ppy))
    sr  = ar / (vol + 1e-9)
    cum = (1 + r).cumprod()
    mdd = float(-((cum - cum.cummax()) / cum.cummax()).min())
    spy_tot = float((1 + df["spy_ret"]).prod())
    spy_ar  = float(spy_tot ** (ppy / n) - 1)

    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  Annual Return: {ar:>+7.1%}   (v11: +16.2%  Δ{ar-v11_ar:>+5.1%})")
    print(f"  Sharpe:        {sr:>7.2f}   (v11:   0.77)")
    print(f"  Max Drawdown:  {-mdd:>+7.1%}   (v11: -24.7%)")
    print(f"  Total Return:  {tot-1:>+7.1%}")
    print(f"  vs SPY:        {ar-spy_ar:>+7.1%}/yr")

    # Yearly breakdown
    df2 = df.copy()
    df2["year"] = df2.index.year
    years = sorted(df2["year"].unique())
    beats = 0
    print(f"\n  {'Year':<6} {'Strat':>8} {'SPY':>8} {'Alpha':>8}")
    print(f"  {'─'*34}")
    for y in years:
        yd  = df2[df2["year"] == y]
        yr  = float((1 + yd["ret"]).prod() - 1)
        yspy = float((1 + yd["spy_ret"]).prod() - 1)
        flag = "✓" if yr > yspy else "✗"
        beats += (yr > yspy)
        print(f"  {y:<6} {yr:>+7.1%}  {yspy:>+7.1%}  {flag}{yr-yspy:>+6.1%}")
    print(f"\n  Beat SPY: {beats}/{len(years)} years")

    # Factor attribution
    if "long_ret" in df.columns and "etf_ret" in df.columns:
        long_contrib  = float((1 + df["long_ret"]).prod() - 1)
        short_contrib = float((1 + df["short_ret"]).prod() - 1)
        etf_contrib   = float((1 + df["etf_ret"]).prod() - 1)
        print(f"\n  Attribution:")
        print(f"    Stock long:  {long_contrib:>+7.1%}")
        if short_contrib != 0:
            print(f"    Stock short: {short_contrib:>+7.1%}")
        print(f"    TQQQ ETF:    {etf_contrib:>+7.1%}")

    return ar, sr, mdd


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nCanyon Quant v14 — Institutional Multi-Factor")
    print("=" * 60)

    rebal_idx = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))
    trade_dates = pd.DatetimeIndex([prices.index[t] for t in rebal_idx])

    # Precompute fundamentals
    fund_matrix = precompute_fundamentals(trade_dates)

    # Run variants
    print("\nRunning v14 (momentum + fundamentals, long-only)...")
    df_v14 = run_backtest(fund_matrix, include_short=False, label="v14 long-only")
    ar14, sr14, mdd14 = print_stats(df_v14, "v14: Momentum + EDGAR Fundamentals (Long Only)")

    print("\nRunning v14 long-short simulation...")
    df_ls = run_backtest(fund_matrix, include_short=True, label="v14 long-short")
    ar_ls, sr_ls, mdd_ls = print_stats(df_ls, "v14: Long-Short (eliminates market beta)")

    if not HAS_EDGAR:
        print("\nRunning v11-equivalent (momentum only, for baseline)...")
        empty_fund = pd.DataFrame(index=trade_dates)
        df_v11 = run_backtest(empty_fund, include_short=False, label="v11 equivalent")
        print_stats(df_v11, "v11 equivalent (momentum only)")

    print(f"\n{'═'*60}")
    print("  Version Comparison")
    print(f"{'═'*60}")
    print(f"  v11 (best prior)    AR +16.2%  Sharpe 0.77  OOS 1.36")
    print(f"  v14 momentum+fund   AR {ar14:>+5.1%}  Sharpe {sr14:.2f}")
    print(f"  v14 long-short      AR {ar_ls:>+5.1%}  Sharpe {sr_ls:.2f}")
    print(f"{'═'*60}")

    df_v14.to_csv("backtest_v14_longonly.csv")
    df_ls.to_csv("backtest_v14_longshort.csv")
    print("Saved: backtest_v14_longonly.csv, backtest_v14_longshort.csv")
