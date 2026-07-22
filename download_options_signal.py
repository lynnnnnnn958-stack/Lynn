"""
Options Signals — Improvement 7
Two-layer approach:
  A) Market-level: SPY put/call open interest ratio (macro timing)
  B) Per-stock: implied vol rank (IVR) as cross-sectional signal
     - Low IVR stocks: options market expects stability → momentum stocks
     - High IVR stocks: uncertainty priced in → avoid until resolution
Note: yfinance only has CURRENT options chains, not historical.
This script produces CURRENT signals for paper trading.
For historical backtest, we use VIX percentile as IVR proxy.
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

TODAY = datetime.today().strftime("%Y-%m-%d")
print(f"Options signal download — {TODAY}")

# ── A. Market-level put/call ratio from SPY ───────────────────────────────────
print("\nA. SPY Put/Call Ratio...")
try:
    spy = yf.Ticker("SPY")
    exps = spy.options
    if exps:
        # Use nearest expiration > 2 weeks out (monthly)
        exp_dates = [pd.Timestamp(e) for e in exps]
        min_date  = pd.Timestamp.today() + pd.DateOffset(days=14)
        valid_exp = [e for e in exp_dates if e >= min_date]
        exp       = valid_exp[0].strftime("%Y-%m-%d") if valid_exp else exps[-1]

        chain = spy.option_chain(exp)
        calls = chain.calls
        puts  = chain.puts
        # Focus on near-the-money options (strikes within 10% of spot)
        spot  = float(spy.history(period="1d")["Close"].iloc[-1])
        ntm   = lambda df: df[(df["strike"] >= spot * 0.90) &
                               (df["strike"] <= spot * 1.10)]
        call_oi = ntm(calls)["openInterest"].sum()
        put_oi  = ntm(puts)["openInterest"].sum()
        pcr     = float(put_oi) / (float(call_oi) + 1)
        print(f"  SPY spot: {spot:.2f}")
        print(f"  Expiry: {exp}  (NTM options ±10%)")
        print(f"  Call OI: {call_oi:,}   Put OI: {put_oi:,}")
        print(f"  Put/Call ratio: {pcr:.3f}")
        if pcr > 1.2:
            print(f"  Signal: FEAR (PCR>{1.2:.1f}) → contrarian BULLISH")
        elif pcr < 0.7:
            print(f"  Signal: GREED (PCR<{0.7:.1f}) → contrarian BEARISH")
        else:
            print(f"  Signal: NEUTRAL")

        # Save current market signal
        mkt_signal = pd.DataFrame([{
            "date": TODAY, "spy_spot": round(spot, 2),
            "exp": exp, "call_oi": call_oi, "put_oi": put_oi,
            "pcr": round(pcr, 4),
            "regime": "fear" if pcr > 1.2 else ("greed" if pcr < 0.7 else "neutral"),
        }])
        out = ROOT / "options_market_signal.csv"
        if out.exists():
            old = pd.read_csv(out)
            old = old[old["date"] != TODAY]
            mkt_signal = pd.concat([old, mkt_signal], ignore_index=True)
        mkt_signal.to_csv(out, index=False)
        print(f"  Saved: options_market_signal.csv")
except Exception as e:
    print(f"  SPY PCR failed: {e}")

# ── B. Per-stock Implied Vol Rank (IVR) ─────────────────────────────────────
print("\nB. Per-stock Implied Vol (current, for paper trading)...")
p25_path = ROOT / "sp500_price_25yr.csv"
if not p25_path.exists():
    print("  No price data found.")
else:
    tickers = [c for c in pd.read_csv(p25_path, nrows=0).columns if c != "SPY"]
    # Focus on top 50 for speed
    import time
    sample = tickers[:50]
    print(f"  Downloading IV for {len(sample)} tickers...")
    iv_records = []
    n_ok = 0
    for tk in sample:
        try:
            t    = yf.Ticker(tk)
            exps = t.options
            if not exps: continue
            # Nearest expiry > 2 weeks
            exp_dates = [pd.Timestamp(e) for e in exps]
            min_dt    = pd.Timestamp.today() + pd.DateOffset(days=14)
            valid     = [e for e in exp_dates if e >= min_dt]
            if not valid: continue
            exp = valid[0].strftime("%Y-%m-%d")
            chain = t.option_chain(exp)
            calls = chain.calls
            puts  = chain.puts
            # ATM IV = average of ATM call and put IV
            hist  = t.history(period="5d")
            if hist.empty: continue
            spot  = float(hist["Close"].iloc[-1])
            atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
            atm_p = puts.iloc[(puts["strike"]  - spot).abs().argsort()[:1]]
            iv_call = float(atm_c["impliedVolatility"].iloc[0]) if len(atm_c) else np.nan
            iv_put  = float(atm_p["impliedVolatility"].iloc[0]) if len(atm_p) else np.nan
            iv      = np.nanmean([iv_call, iv_put])
            if np.isfinite(iv):
                iv_records.append({"date": TODAY, "ticker": tk, "iv": round(iv, 4)})
                n_ok += 1
        except Exception:
            pass
        time.sleep(0.1)
        if n_ok % 10 == 0 and n_ok > 0:
            print(f"    [{n_ok}/{len(sample)}]")

    if iv_records:
        iv_df = pd.DataFrame(iv_records)
        iv_df.to_csv(ROOT / "options_stock_iv.csv", index=False)
        med_iv = iv_df["iv"].median()
        print(f"\n  Saved: options_stock_iv.csv  ({len(iv_df)} stocks)")
        print(f"  Median IV: {med_iv:.1%}  "
              f"(range: {iv_df['iv'].min():.1%} - {iv_df['iv'].max():.1%})")
        # Low IV = stable, suitable for momentum continuation
        low_iv = iv_df.nsmallest(10, "iv")["ticker"].tolist()
        high_iv = iv_df.nlargest(10, "iv")["ticker"].tolist()
        print(f"  Low IV (momentum friendly): {low_iv[:5]}")
        print(f"  High IV (avoid): {high_iv[:5]}")

# ── C. Historical PCR proxy via VIX percentile ────────────────────────────────
print("\nC. Building historical PCR proxy from VIX for backtest...")
macro_path = ROOT / "macro_regime.csv"
if macro_path.exists():
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    if "VIX" in macro.columns:
        vix       = macro["VIX"].dropna()
        vix_pctile = vix.expanding().rank(pct=True)
        # PCR ~ 1 when VIX pctile high (fear), PCR ~ 0.6 when low (greed)
        pcr_proxy  = 0.6 + 0.8 * vix_pctile
        macro["pcr_proxy"] = pcr_proxy
        macro["opt_signal"] = np.where(macro["pcr_proxy"] > 1.2, "fear",
                               np.where(macro["pcr_proxy"] < 0.7, "greed", "neutral"))
        macro.to_csv(macro_path)
        print(f"  pcr_proxy added to macro_regime.csv")
        print(f"  Fear regime: {(macro['pcr_proxy']>1.2).sum()} days "
              f"({(macro['pcr_proxy']>1.2).mean():.0%})")
        print(f"  Greed regime: {(macro['pcr_proxy']<0.7).sum()} days "
              f"({(macro['pcr_proxy']<0.7).mean():.0%})")

print("\nDone.")
