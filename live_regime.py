"""
live_regime.py — Canyon Quant v25.1 live regime checker
VIX-scaled TQQQ rule: 50% SUE stocks + VIX-tiered TQQQ
TQQQ weight: 50% (VIX<20) / 25% (VIX 20-25) / 0% (VIX>=25 or gates fail)
Gates: bull regime AND QQQ > 200-day MA AND QQQ 3M momentum positive
Run: python live_regime.py
"""
import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import re
from datetime import datetime, date
from io import StringIO
from pathlib import Path

ROOT  = Path(__file__).parent
UA    = {"User-Agent": "canyon-quant research lynnnnnnn958@gmail.com"}
TODAY = date.today().isoformat()
NOW   = datetime.now().strftime("%Y-%m-%d %H:%M")

# ── 1. FRED macro data ────────────────────────────────────────────────────────
def fetch_fred(series_id, lookback_days=400):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r  = requests.get(url, headers=UA, timeout=20)
        df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
        df = df.set_index("observation_date").rename(columns={series_id: series_id})
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df.dropna().sort_index().iloc[-lookback_days:]
    except Exception as e:
        print(f"  FRED {series_id} error: {e}")
        return pd.DataFrame()

print("Fetching live data...")

t10y2y_df = fetch_fred("T10Y2Y")
unrate_df  = fetch_fred("UNRATE")
dgs10_df   = fetch_fred("DGS10")
hyg_df     = fetch_fred("BAMLH0A0HYM2")

t10y2y      = float(t10y2y_df["T10Y2Y"].iloc[-1])             if not t10y2y_df.empty else np.nan
unrate      = float(unrate_df["UNRATE"].iloc[-1])              if not unrate_df.empty else np.nan
unrate_ma90 = float(unrate_df["UNRATE"].iloc[-90:].mean())     if len(unrate_df) >= 90 else unrate
dgs10       = float(dgs10_df["DGS10"].iloc[-1])                if not dgs10_df.empty else np.nan
hyg_oas     = float(hyg_df["BAMLH0A0HYM2"].iloc[-1])          if not hyg_df.empty else np.nan
hyg_ma20    = float(hyg_df["BAMLH0A0HYM2"].iloc[-20:].mean()) if len(hyg_df) >= 20 else hyg_oas

# ── 2. Market data via yfinance ───────────────────────────────────────────────
try:
    import yfinance as yf

    spy_hist  = yf.download("SPY",  period="300d", progress=False, auto_adjust=True)
    spy_close = spy_hist["Close"].values.flatten()
    spy_price = float(spy_close[-1])
    spy_ma200 = float(np.nanmean(spy_close[-200:]))

    qqq_hist  = yf.download("QQQ",  period="300d", progress=False, auto_adjust=True)
    qqq_close = qqq_hist["Close"].values.flatten()
    qqq_price = float(qqq_close[-1])
    qqq_ma200 = float(np.nanmean(qqq_close[-200:]))
    qqq_3m_price = float(qqq_close[-65]) if len(qqq_close) >= 65 else float(qqq_close[0])

    vix_hist  = yf.download("^VIX", period="5d",   progress=False, auto_adjust=True)
    vix       = float(vix_hist["Close"].values.flatten()[-1])

    tqqq_hist = yf.download("TQQQ", period="30d",  progress=False, auto_adjust=True)
    tqqq_close= tqqq_hist["Close"].values.flatten()
    tqqq_price= float(tqqq_close[-1])

except Exception as e:
    print(f"  yfinance error: {e}")
    spy_price = spy_ma200 = vix = qqq_price = qqq_ma200 = qqq_3m_price = np.nan
    tqqq_price = np.nan

# ── 3. Breadth (last known from backtest CSV) ─────────────────────────────────
try:
    bdf          = pd.read_csv(ROOT / "backtest_v24_stock.csv")
    last_breadth = float(bdf["breadth"].iloc[-1])
    breadth_date = bdf["date"].iloc[-1]
except Exception:
    last_breadth = 0.568
    breadth_date = "2026-05-12"

# ── 4. Evaluate v24 regime signals (5 macro signals, ≥60% → bull) ────────────
s1_bull = spy_price > spy_ma200          if np.isfinite(spy_price) else None
s2_bull = last_breadth > 0.45
s3_bull = vix < 25                       if np.isfinite(vix)       else None
s4_bull = t10y2y > -0.25                 if np.isfinite(t10y2y)    else None
s5_bull = unrate <= unrate_ma90          if np.isfinite(unrate)     else None

all_signals = [s1_bull, s2_bull, s3_bull, s4_bull, s5_bull]
available   = [s for s in all_signals if s is not None]
bull_count  = sum(available)
raw_regime  = (bull_count / len(available)) >= 0.60 if available else True
REGIME_STR  = "BULL" if raw_regime else "BEAR"

# ── 5. v25.1 VIX-Scaled TQQQ rule ────────────────────────────────────────────
# Gate A: QQQ above 200-day moving average
gate_qqq_ma  = np.isfinite(qqq_price) and np.isfinite(qqq_ma200) and qqq_price > qqq_ma200
# Gate B: QQQ 3-month momentum positive
gate_qqq_mom = np.isfinite(qqq_price) and np.isfinite(qqq_3m_price) and qqq_price > qqq_3m_price
# VIX tier (determines weight, not binary on/off)
vix_tier = "LOW" if vix < 20 else ("MID" if vix < 25 else "HIGH")

gates_pass = raw_regime and gate_qqq_ma and gate_qqq_mom
if gates_pass:
    if vix < 20:   tqqq_weight = 0.50   # full
    elif vix < 25: tqqq_weight = 0.25   # half
    else:          tqqq_weight = 0.00   # zero (VIX too high)
else:
    tqqq_weight = 0.00

tqqq_on      = tqqq_weight > 0
stock_weight = 0.50
cash_weight  = 1.0 - stock_weight - tqqq_weight

def fmt(v, d=2):
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"

# ── 6. Console report ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  CANYON QUANT v25.1 — QQQ HUNTER  {NOW}")
print(f"{'='*65}")
print(f"\n  v24 MACRO REGIME: {REGIME_STR}  ({bull_count}/{len(available)} signals positive)")
print(f"  (Note: 2-month lag filter applies before actual portfolio switch)\n")

signals_detail = [
    ("SPY trend",    f"SPY {fmt(spy_price)} vs MA {fmt(spy_ma200)}",  s1_bull,  "> 200MA"),
    ("Breadth",      f"{last_breadth:.1%} stocks above 200MA",         s2_bull,  "> 45%"),
    ("VIX",          f"^VIX = {fmt(vix,1)}",                          s3_bull,  "< 25"),
    ("Yield curve",  f"T10Y2Y = {fmt(t10y2y,3)}",                     s4_bull,  "> −0.25"),
    ("Employment",   f"UNRATE {fmt(unrate,1)}% vs MA {unrate_ma90:.1f}%", s5_bull, "UNRATE ≤ MA"),
]
print(f"  {'Signal':<16} {'Value':<30} {'Threshold':<14} Status")
print(f"  {'-'*66}")
for name, val, bull, thr in signals_detail:
    status = "BULL ✓" if bull else ("BEAR ✗" if bull is False else "N/A")
    print(f"  {name:<16} {val:<30} {thr:<14} {status}")

print(f"\n  v25.1 VIX-SCALED TQQQ CHECK:")
print(f"  {'Gate A: QQQ > 200-day MA':<30} QQQ={fmt(qqq_price)} vs MA={fmt(qqq_ma200)} → {'✓ PASS' if gate_qqq_ma else '✗ FAIL'}")
print(f"  {'Gate B: QQQ 3M momentum':<30} QQQ={fmt(qqq_price)} vs 3M-ago={fmt(qqq_3m_price)} → {'✓ PASS' if gate_qqq_mom else '✗ FAIL'}")
print(f"  {'Regime: BULL':<30}               → {'✓ PASS' if raw_regime else '✗ FAIL'}")
print(f"  {'VIX tier':<30} VIX={fmt(vix,1)} → {vix_tier} (<20 Full / 20-25 Half / >=25 Zero)")
print(f"\n  {'─'*40}")
if tqqq_on:
    tier_label = "FULL 50%" if tqqq_weight == 0.50 else "HALF 25%"
    print(f"  TQQQ STATUS: ON  {tier_label}  (VIX {vix_tier})")
    print(f"  v25.1 ALLOCATION: 50% SUE stocks  +  {tqqq_weight*100:.0f}% TQQQ  +  {cash_weight*100:.0f}% cash")
else:
    failed = []
    if not raw_regime:   failed.append("regime=BEAR")
    if not gate_qqq_ma:  failed.append("QQQ<200MA")
    if not gate_qqq_mom: failed.append("QQQ 3M mom negative")
    if vix >= 25:        failed.append(f"VIX={fmt(vix,1)}≥25")
    print(f"  TQQQ STATUS: OFF  ({', '.join(failed)})")
    print(f"  v25.1 ALLOCATION: 50% SUE stocks  +  50% CASH")
print(f"  {'─'*40}")

print(f"\n  FRED:  T10Y2Y={fmt(t10y2y,3)}  UNRATE={fmt(unrate,1)}%  DGS10={fmt(dgs10,2)}%  HYG={fmt(hyg_oas,2)}bps")
print(f"  Mkt:   SPY={fmt(spy_price)} (MA {fmt(spy_ma200)})  QQQ={fmt(qqq_price)} (MA {fmt(qqq_ma200)})  VIX={fmt(vix,1)}")

# ── 7. Patch HTML ─────────────────────────────────────────────────────────────
html_path = ROOT / "canyon_v24_research.html"
if not html_path.exists():
    print("\nHTML not found — skipping patch")
else:
    html = html_path.read_text(encoding="utf-8")

    regime_color = "#6BCCA0" if raw_regime else "#EF9090"
    tqqq_color   = "#6BCCA0" if tqqq_on    else "#EF9090"
    tqqq_label   = "ON ✓"    if tqqq_on    else "OFF ✗"

    # Patch hero KPI: Market Regime
    html = re.sub(
        r'(<p class="kpi-label">Market Regime</p>\s*<p class="kpi-val"[^>]*>)(?:BULL|BEAR)(</p>\s*<p class="kpi-note"[^>]*>)[^<]*(</p>)',
        (f'<p class="kpi-label">Market Regime</p>'
         f'<p class="kpi-val" style="color:{regime_color};font-size:28px">{REGIME_STR}</p>'
         f'<p class="kpi-note" style="color:{regime_color}">live · {NOW} · {bull_count}/{len(available)} signals</p>'),
        html, flags=re.DOTALL
    )

    # Patch Today card regime
    html = re.sub(
        r'(<div class="today-card-label">Regime[^<]*</div>\s*<div class="today-card-val [^"]*">)(?:BULL|BEAR)(</div>\s*<div class="today-card-note">)[^<]*(</div>)',
        (f'<div class="today-card-label">Regime <span style="color:#1B6F4A;font-size:9px;font-weight:700">LIVE</span></div>'
         f'<div class="today-card-val {"bull" if raw_regime else "bear-clr"}">{REGIME_STR}</div>'
         f'<div class="today-card-note">{bull_count}/{len(available)} signals · {TODAY}</div>'),
        html, flags=re.DOTALL
    )

    # Patch TQQQ card
    vix_label = f"VIX {fmt(vix,1)} ({vix_tier})"
    html = re.sub(
        r'(<div class="today-card-label">TQQQ Overlay</div>\s*<div class="today-card-val">)[^<]*(</div>\s*<div class="today-card-note">)[^<]*(</div>)',
        (f'<div class="today-card-label">TQQQ Overlay</div>'
         f'<div class="today-card-val" style="color:{tqqq_color}">{tqqq_weight*100:.0f}% {tqqq_label}</div>'
         f'<div class="today-card-note">{vix_label} · QQQ MA {"✓" if gate_qqq_ma else "✗"} · mom {"✓" if gate_qqq_mom else "✗"} · {TODAY}</div>'),
        html, flags=re.DOTALL
    )

    # Build new macro signal rows
    sig_rows_html = ""
    for name, val, bull, thr in signals_detail:
        icon  = "✓" if bull else ("✗" if bull is False else "—")
        color_cls = "pos" if bull else ("neg" if bull is False else "")
        sig_rows_html += f'<tr><td>{name}</td><td style="font-size:12px;color:#666">{val}</td><td class="r {color_cls}">{icon} {thr}</td></tr>\n            '

    # Patch macro signal table
    new_sig_table = (
        f'<div class="tbl-title mt24">Macro Regime Signals (FRED) '
        f'<span style="color:#1B6F4A;font-weight:700">— LIVE {TODAY}</span></div>\n'
        f'        <div style="font-size:11px;color:#666;margin-bottom:6px">Fetched {NOW} · breadth from {breadth_date}</div>\n'
        f'        <table>\n'
        f'          <thead><tr><th>Signal</th><th>Live Value</th><th class="r">Status</th></tr></thead>\n'
        f'          <tbody>\n'
        f'            {sig_rows_html}</tbody>\n'
        f'        </table>'
    )
    html = re.sub(
        r'<div class="tbl-title mt24">Macro Regime Signals \(FRED\).*?</table>',
        new_sig_table,
        html, flags=re.DOTALL, count=1
    )

    # Patch v25 two-gate status in Today tab (if the section exists)
    tier_display = "FULL 50%" if tqqq_weight == 0.50 else ("HALF 25%" if tqqq_weight == 0.25 else "ZERO")
    gate_status_new = (
        f'<div style="background:{"#EAF5EE" if tqqq_on else "#FDECEA"};border-radius:6px;padding:14px;margin-top:16px">'
        f'<div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;color:{"#1B6F4A" if tqqq_on else "#B83232"};margin-bottom:8px">'
        f'v25.1 VIX-Scaled TQQQ: {"TQQQ ON " + tier_display + " ✓" if tqqq_on else "TQQQ OFF ✗"}</div>'
        f'<div style="font-size:12px;line-height:1.8;color:{"#1B6F4A" if tqqq_on else "#B83232"}">'
        f'Regime: {"✓ " if raw_regime else "✗ "}{REGIME_STR}&nbsp;&nbsp;'
        f'QQQ&gt;200MA: {"✓ " if gate_qqq_ma else "✗ "}QQQ={fmt(qqq_price)} MA={fmt(qqq_ma200)}&nbsp;&nbsp;'
        f'3M Mom: {"✓ " if gate_qqq_mom else "✗ "}{fmt(qqq_price)} vs {fmt(qqq_3m_price)}&nbsp;&nbsp;'
        f'VIX: {fmt(vix,1)} → {vix_tier} tier ({tqqq_weight*100:.0f}% TQQQ)'
        f'</div></div>'
    )
    # Replace existing gate status div or insert after TQQQ card
    if 'v25' in html and ('Two-Gate TQQQ:' in html or 'VIX-Scaled TQQQ:' in html):
        html = re.sub(
            r'<div style="background:#(?:EAF5EE|FDECEA);border-radius:6px;padding:14px;margin-top:16px">.*?</div></div>',
            gate_status_new,
            html, flags=re.DOTALL, count=1
        )
    else:
        html = html.replace(
            '</div>  <!-- end today-cards -->',
            gate_status_new + '\n</div>  <!-- end today-cards -->'
        )

    html_path.write_text(html, encoding="utf-8")
    print(f"\n✓ Patched: {html_path.name}")
    print(f"  Regime: {REGIME_STR}  TQQQ: {tqqq_label}  ({NOW})")
    print(f"\nRefresh the browser to see live updates.")
