"""
expand_dashboard.py — Canyon Quant v24 dashboard expansion.
Adds: Full Signals Universe, Macro, Earnings, Insiders, Factor Lab, Guide tabs.
Run: python expand_dashboard.py
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

ROOT = Path('/Users/renjingru/Desktop/model/canyon_quant')
HTML = ROOT / 'canyon_v24_research.html'

# ── Load all data ─────────────────────────────────────────────────────────────
alpha   = pd.read_csv(ROOT / 'alpha_scores.csv')
earn    = pd.read_csv(ROOT / 'earnings_calendar.csv')
insiders= pd.read_csv(ROOT / 'insider_form4_signals.csv')
fic     = pd.read_csv(ROOT / 'factor_ic_history.csv', parse_dates=['date'])
macro   = pd.read_csv(ROOT / 'macro_signals_daily.csv', parse_dates=['date'])

sig_cols = ['sig_regime_ml','sig_quality','sig_revision','sig_surprise',
            'sig_sentiment','sig_squeeze','sig_insider','sig_options',
            'sig_ml_ensemble','sig_momentum']

alpha = alpha.sort_values('alpha_rank').reset_index(drop=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def score_color(v):
    if v >= 70: return '#6BCCA0'
    if v >= 50: return '#B8943F'
    return '#EF9090'

def badge(v, high=70, mid=50):
    if v >= high: return f'<span style="background:#EAF5EE;color:#1B6F4A;padding:1px 5px;border-radius:3px;font-size:10px">{v:.0f}</span>'
    if v >= mid:  return f'<span style="background:#FEF3D7;color:#7A5D1A;padding:1px 5px;border-radius:3px;font-size:10px">{v:.0f}</span>'
    return         f'<span style="background:#FDECEA;color:#B83232;padding:1px 5px;border-radius:3px;font-size:10px">{v:.0f}</span>'

# ── 1. Build JS universe data for full signals tab ────────────────────────────
def build_universe_js(df):
    rows = []
    for _, r in df.iterrows():
        sigs = [round(float(r.get(c, 50)), 1) for c in sig_cols]
        rows.append(
            f'[{r["alpha_rank"]},"{r["ticker"]}",{r["alpha_score"]:.1f},'
            f'"{r.get("signal","HOLD")}","{r.get("sector","—")}","{r.get("crowding_level","—")}",'
            f'{",".join(str(s) for s in sigs)}]'
        )
    return '[\n    ' + ',\n    '.join(rows) + '\n  ]'

universe_js = build_universe_js(alpha)

# ── 2. Macro section: 30d and 90d cumulative ─────────────────────────────────
macro = macro.sort_values('date').tail(252).reset_index(drop=True)
m30 = macro.tail(22)
m90 = macro.tail(66)

def cum_pct(s): return ((1 + s).prod() - 1) * 100
def trend_arrow(v): return '▲' if v > 0 else '▼'
def trend_color(v): return '#6BCCA0' if v > 0 else '#EF9090'
def fmt_pct(v): return f'{v:+.2f}%'

macro_cards = []
cols_labels = {
    'credit_spread': ('Credit Spread (HYG)', 'neg', 'Widening = bearish for risk assets'),
    'yield_curve':   ('Yield Curve Slope',   'pos', 'Steepening = economic expansion'),
    'dollar_strength':('Dollar Strength (UUP)','neg', 'Rising dollar = tighter global conditions'),
    'gold_flow':     ('Gold Flow (GLD)',      'pos', 'Rising gold = risk-off or inflation hedge'),
}

for col, (label, direction, note) in cols_labels.items():
    c30 = cum_pct(m30[col])
    c90 = cum_pct(m90[col])
    last = float(macro[col].iloc[-1])
    clr = '#6BCCA0' if (c30 > 0) == (direction == 'pos') else '#EF9090'
    macro_cards.append(f'''
      <div class="mac-card">
        <div class="mac-label">{label}</div>
        <div class="mac-val" style="color:{clr}">{trend_arrow(c30)} {fmt_pct(c30)}</div>
        <div class="mac-sub">30d return</div>
        <div class="mac-row">
          <span class="mac-tag">90d: <b>{fmt_pct(c90)}</b></span>
          <span class="mac-tag">Daily: <b>{last:+.4f}</b></span>
        </div>
        <div class="mac-note">{note}</div>
      </div>''')

# 30d sparkline data for Chart.js (4 series x 22 points)
spark_dates = [str(d.date()) for d in macro.tail(22)['date']]
spark_data = {}
for col in cols_labels:
    cumvals = []
    cum = 1.0
    for v in macro.tail(22)[col]:
        cum *= (1 + v)
        cumvals.append(round((cum - 1) * 100, 4))
    spark_data[col] = cumvals

macro_spark_js = f"""
  const SPARK_DATES = {spark_dates};
  const SPARK = {{
    credit_spread: {spark_data['credit_spread']},
    yield_curve:   {spark_data['yield_curve']},
    dollar:        {spark_data['dollar_strength']},
    gold:          {spark_data['gold_flow']}
  }};"""

# ── 3. Earnings section ───────────────────────────────────────────────────────
def risk_badge(flag):
    colors = {'HIGH': ('#B83232','#FDECEA'), 'MEDIUM': ('#7A5D1A','#FEF3D7'), 'LOW': ('#1B6F4A','#EAF5EE')}
    c,bg = colors.get(str(flag).upper(), ('#666','#F5F5F5'))
    return f'<span style="color:{c};background:{bg};padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700">{flag}</span>'

def action_badge(a):
    if 'STRONG' in str(a).upper(): return f'<span style="color:#1B6F4A;font-weight:700">{a}</span>'
    if 'BUY' in str(a).upper():    return f'<span style="color:#1B6F4A">{a}</span>'
    return f'<span style="color:#666">{a}</span>'

earn_rows = []
for _, r in earn.sort_values('days_until', ascending=False).iterrows():
    d = r.get('days_until')
    d_str = f'{int(d):+d}d' if pd.notna(d) else '—'
    ivr = f"{r.get('iv_rank',0):.0f}" if pd.notna(r.get('iv_rank')) else '—'
    sur = f"{r.get('surprise_pct_last',0):+.1f}%" if pd.notna(r.get('surprise_pct_last')) else '—'
    earn_rows.append(
        f'<tr>'
        f'<td class="td-ticker">{r["ticker"]}</td>'
        f'<td>{r.get("earnings_date","—")}</td>'
        f'<td style="text-align:center">{d_str}</td>'
        f'<td>{risk_badge(r.get("risk_flag","—"))}</td>'
        f'<td style="text-align:right">{r.get("alpha_score",0):.1f}</td>'
        f'<td style="text-align:right">{ivr}</td>'
        f'<td style="text-align:right">{sur}</td>'
        f'<td>{action_badge(r.get("recommended_action","—"))}</td>'
        f'</tr>'
    )
earn_table = '\n'.join(earn_rows)

high_cnt = (earn['risk_flag'] == 'HIGH').sum()
med_cnt  = (earn['risk_flag'] == 'MEDIUM').sum() if 'MEDIUM' in earn['risk_flag'].values else 0
low_cnt  = (earn['risk_flag'] == 'LOW').sum() if 'LOW' in earn['risk_flag'].values else 0

# ── 4. Insiders section ───────────────────────────────────────────────────────
def insider_badge(direction):
    if 'SELL' in str(direction).upper(): return '<span style="color:#B83232;font-weight:700">▼ NET SELL</span>'
    if 'BUY'  in str(direction).upper(): return '<span style="color:#1B6F4A;font-weight:700">▲ NET BUY</span>'
    return '<span style="color:#999">— ETF</span>'

ins_rows = []
for _, r in insiders.iterrows():
    txn = int(r.get('recent_transactions', 0))
    status = str(r.get('insider_status','—'))
    note = str(r.get('notes','')) if pd.notna(r.get('notes')) else '—'
    ins_rows.append(
        f'<tr>'
        f'<td class="td-ticker">{r["ticker"]}</td>'
        f'<td style="font-size:11px;color:#666">{status}</td>'
        f'<td style="text-align:center">{txn}</td>'
        f'<td>{insider_badge(r.get("net_direction",""))}</td>'
        f'<td style="font-size:11px;color:#888">{note}</td>'
        f'</tr>'
    )
ins_table = '\n'.join(ins_rows)

# ── 5. Factor Lab section ─────────────────────────────────────────────────────
factor_display = {
    'ic_mom_12_1':  ('Momentum 12M−1',  'b-d'),
    'ic_mom_3m':    ('Momentum 3M',      'b-d'),
    'ic_mom_1m':    ('Momentum 1M',      'b-d'),
    'ic_inv_vol':   ('Inverse Vol',      'b-m'),
    'ic_above_200': ('Above 200-day MA', 'b-m'),
    'ic_low_rsi':   ('Low RSI',          'b-w'),
}

def t_stat(series):
    n = len(series)
    if n < 5: return 0.0
    return float(series.mean() / (series.std(ddof=1) / np.sqrt(n)))

fac_rows = []
for col, (label, cls) in factor_display.items():
    if col not in fic.columns: continue
    vals = fic[col].dropna()
    mean_ic = vals.mean()
    t = t_stat(vals)
    pct_pos = (vals > 0).mean() * 100
    best = vals.max()
    worst = vals.min()
    ic_color = '#1B6F4A' if mean_ic > 0 else '#B83232'
    t_color  = '#1B6F4A' if abs(t) > 2 else ('#B8943F' if abs(t) > 1.65 else '#B83232')
    fac_rows.append(
        f'<tr>'
        f'<td>{label}</td>'
        f'<td style="color:{ic_color};font-weight:600;text-align:right">{mean_ic:+.3f}</td>'
        f'<td style="color:{t_color};font-weight:600;text-align:right">{t:+.2f}</td>'
        f'<td style="text-align:right">{pct_pos:.0f}%</td>'
        f'<td style="text-align:right;color:#1B6F4A">{best:+.3f}</td>'
        f'<td style="text-align:right;color:#B83232">{worst:+.3f}</td>'
        f'<td><div class="ic-bar-wrap" style="width:120px"><div class="ic-bar {"s" if abs(mean_ic)>0.05 else "w"}" style="width:{min(abs(mean_ic)*300,100):.0f}%"></div></div></td>'
        f'</tr>'
    )
fac_table = '\n'.join(fac_rows)

# Factor Lab: 12-month rolling IC for above_200 (v24 primary fallback)
ic200 = fic['ic_above_200'].dropna().values[-36:]
ic200_js = '[' + ','.join(f'{x:.4f}' for x in ic200) + ']'
ic200_dates = [str(d.date()) for d in fic['date'].dropna().iloc[-36:]]

# ── 6. Build alpha universe table note ───────────────────────────────────────
long_count = (alpha['signal'] == 'LONG').sum()
avoid_count = (alpha['signal'] == 'SHORT_AVOID').sum()
hold_count  = (alpha['signal'] == 'HOLD').sum()
sectors = alpha['sector'].dropna().unique()

# ── Assemble the complete new HTML sections ───────────────────────────────────

NEW_CSS = """
    /* ── EXPANDED DASHBOARD STYLES ─────────────────────────────────────────── */
    .mac-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    @media(max-width:900px){.mac-grid{grid-template-columns:repeat(2,1fr)}}
    .mac-card{background:#fff;border:1px solid #E8E2D6;border-radius:6px;padding:18px;border-top:3px solid #B8943F}
    .mac-label{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#888;margin-bottom:6px;font-weight:600}
    .mac-val{font-family:'Playfair Display',serif;font-size:26px;font-weight:700;line-height:1;margin-bottom:4px}
    .mac-sub{font-size:10px;color:#AAA;margin-bottom:8px}
    .mac-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}
    .mac-tag{background:#F5F3EE;padding:2px 7px;border-radius:3px;font-size:11px;color:#555}
    .mac-note{font-size:11px;color:#888;line-height:1.5}
    .mac-chart-wrap{background:#fff;border:1px solid #E8E2D6;border-radius:6px;padding:20px;margin-bottom:30px}
    .mac-chart-title{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#888;font-weight:600;margin-bottom:12px}

    /* FULL UNIVERSE */
    .uni-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
    .uni-search{padding:8px 12px;border:1px solid #D4CFC5;border-radius:4px;font-size:13px;width:200px;outline:none;font-family:inherit}
    .uni-search:focus{border-color:#B8943F}
    .uni-filter{padding:7px 10px;border:1px solid #D4CFC5;border-radius:4px;font-size:12px;background:#fff;font-family:inherit;cursor:pointer}
    .uni-count{font-size:12px;color:#888;margin-left:4px}
    .sig-bar{display:inline-block;height:8px;border-radius:2px;vertical-align:middle}
    .sig-bar-wrap{display:inline-flex;gap:2px;align-items:center}
    .td-ticker{font-weight:700;font-family:monospace;font-size:13px;color:#1B2A4A}
    table.uni-tbl{width:100%;font-size:12px}
    table.uni-tbl th{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#888;border-bottom:2px solid #E8E2D6;padding:6px 4px;white-space:nowrap;cursor:pointer}
    table.uni-tbl td{padding:5px 4px;border-bottom:1px solid #F0EDE6;vertical-align:middle}
    table.uni-tbl tr:hover td{background:#FAFAF8}
    .signal-long  {color:#1B6F4A;font-weight:700;font-size:11px}
    .signal-avoid {color:#B83232;font-weight:700;font-size:11px}
    .signal-hold  {color:#888;font-size:11px}
    .uni-pager{display:flex;align-items:center;gap:12px;margin-top:14px;font-size:12px;color:#666}
    .uni-pager button{padding:4px 12px;border:1px solid #D4CFC5;border-radius:4px;background:#fff;cursor:pointer;font-size:12px;font-family:inherit}
    .uni-pager button:hover{border-color:#B8943F;color:#B8943F}

    /* EARNINGS */
    .earn-kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}
    .earn-kpi{background:#fff;border:1px solid #E8E2D6;border-radius:6px;padding:18px;text-align:center}
    .earn-kpi-n{font-family:'Playfair Display',serif;font-size:36px;font-weight:700}
    .earn-kpi-l{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#888;margin-top:4px;font-weight:600}

    /* INSIDERS */
    .ins-note{background:#FEF3D7;border:1px solid #D4A92A;border-radius:4px;padding:12px 16px;font-size:12px;color:#7A5D1A;margin-bottom:20px;line-height:1.6}

    /* FACTOR LAB */
    .fac-note{background:#EAF5EE;border:1px solid #87CCB0;border-radius:4px;padding:12px 16px;font-size:12px;color:#1B6F4A;margin-bottom:20px;line-height:1.6}
    .fac-chart-wrap{background:#fff;border:1px solid #E8E2D6;border-radius:6px;padding:20px;margin-top:24px}

    /* GUIDE */
    .guide-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:32px}
    @media(max-width:800px){.guide-grid{grid-template-columns:1fr}}
    .guide-card{background:#fff;border:1px solid #E8E2D6;border-radius:6px;padding:22px;border-top:3px solid #1B2A4A}
    .guide-card.gold{border-top-color:#B8943F}
    .guide-card.green{border-top-color:#6BCCA0}
    .guide-head{font-weight:700;font-size:14px;margin-bottom:8px;color:#1A1A1A}
    .guide-body{font-size:13px;color:#555;line-height:1.7}
    .guide-step{display:flex;gap:10px;align-items:flex-start;margin-bottom:8px}
    .guide-step-n{background:#1B2A4A;color:#fff;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:2px}
    .checklist{list-style:none;padding:0}
    .checklist li{padding:7px 0;border-bottom:1px solid #F0EDE6;font-size:13px;color:#444;display:flex;align-items:center;gap:8px}
    .checklist li::before{content:'☐';color:#B8943F;font-size:16px;flex-shrink:0}
"""

SEC_FULL_SIGNALS = f"""
<!-- ═══════════════════════════════════════════ FULL UNIVERSE -->
<section id="sec-universe" class="tab-section">
  <div class="container">
    <p class="eyebrow">Complete Signal Universe · Jun 10, 2026</p>
    <h2 class="section-head">All {len(alpha)} Stocks — Signal Breakdown</h2>
    <div class="rule"></div>
    <p class="lead">Every S&P 500 stock ranked by composite alpha score. Ten signal components shown as color-coded bars. Search by ticker or sector; filter by signal status.</p>

    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px">
      <div style="background:#EAF5EE;border-radius:6px;padding:16px;text-align:center">
        <div style="font-size:28px;font-weight:700;color:#1B6F4A">{long_count}</div>
        <div style="font-size:11px;color:#1B6F4A;font-weight:600;text-transform:uppercase;letter-spacing:1px">LONG positions</div>
      </div>
      <div style="background:#F5F3EE;border-radius:6px;padding:16px;text-align:center">
        <div style="font-size:28px;font-weight:700;color:#1B2A4A">{hold_count}</div>
        <div style="font-size:11px;color:#666;font-weight:600;text-transform:uppercase;letter-spacing:1px">HOLD / monitor</div>
      </div>
      <div style="background:#FDECEA;border-radius:6px;padding:16px;text-align:center">
        <div style="font-size:28px;font-weight:700;color:#B83232">{avoid_count}</div>
        <div style="font-size:11px;color:#B83232;font-weight:600;text-transform:uppercase;letter-spacing:1px">SHORT AVOID</div>
      </div>
    </div>

    <div class="uni-controls">
      <input id="uni-search" class="uni-search" type="text" placeholder="Search ticker or sector…" oninput="filterUni()">
      <select id="uni-sig" class="uni-filter" onchange="filterUni()">
        <option value="all">All signals</option>
        <option value="LONG">LONG only</option>
        <option value="SHORT_AVOID">SHORT AVOID only</option>
        <option value="HOLD">HOLD only</option>
      </select>
      <select id="uni-sector" class="uni-filter" onchange="filterUni()">
        <option value="all">All sectors</option>
        {"".join(f'<option value="{s}">{s}</option>' for s in sorted(s for s in sectors if pd.notna(s)))}
      </select>
      <span id="uni-count" class="uni-count">Showing {len(alpha)} stocks</span>
    </div>

    <div style="overflow-x:auto">
      <table class="uni-tbl" id="uni-table">
        <thead>
          <tr>
            <th onclick="sortUni(0)">Rank</th>
            <th onclick="sortUni(1)">Ticker</th>
            <th onclick="sortUni(2)">Score</th>
            <th>Signal</th>
            <th>Sector</th>
            <th title="Regime ML">ML</th>
            <th title="Quality">Qual</th>
            <th title="Analyst Revision">Rev</th>
            <th title="Earnings Surprise">EPS</th>
            <th title="Sentiment">Sent</th>
            <th title="Squeeze">Sqz</th>
            <th title="Insider">Ins</th>
            <th title="Options Flow">Opt</th>
            <th title="ML Ensemble">Ens</th>
            <th title="Momentum">Mom</th>
          </tr>
        </thead>
        <tbody id="uni-tbody">
          <tr><td colspan="15" style="text-align:center;padding:20px;color:#888">Loading…</td></tr>
        </tbody>
      </table>
    </div>
    <div class="uni-pager">
      <button onclick="uniPage(-1)">← Prev</button>
      <span id="uni-page-info">Page 1</span>
      <button onclick="uniPage(1)">Next →</button>
      <span style="margin-left:8px;color:#AAA">50 per page</span>
    </div>
    <p style="font-size:11px;color:#AAA;margin-top:12px">Signal columns: ML=Regime ML · Qual=Quality · Rev=Analyst Revision · EPS=Earnings Surprise · Sent=Sentiment · Sqz=Squeeze · Ins=Insider · Opt=Options Flow · Ens=ML Ensemble · Mom=Momentum. Bar width = score (0–100), color: green ≥70 / gold ≥50 / red &lt;50.</p>
  </div>
</section>
"""

SEC_MACRO = f"""
<!-- ═══════════════════════════════════════════ MACRO -->
<section id="sec-macro" class="tab-section">
  <div class="container">
    <p class="eyebrow">Cross-Asset Macro · Live Market Signals</p>
    <h2 class="section-head">Macro Overlay Dashboard</h2>
    <div class="rule"></div>
    <p class="lead">Four cross-asset signals — credit spreads, yield curve slope, dollar strength, and gold flows — provide the macro context for position sizing and TQQQ overlay decisions.</p>

    <div class="mac-grid">
      {"".join(macro_cards)}
    </div>

    <div class="mac-chart-wrap">
      <div class="mac-chart-title">30-Day Cumulative Returns — Cross-Asset Signals</div>
      <canvas id="macro-chart" height="90"></canvas>
    </div>

    <div class="mt24">
      <div class="tbl-title">Signal Interpretation Guide</div>
      <table>
        <thead><tr><th>Signal</th><th>Source</th><th>Bull Reading</th><th>Bear Reading</th><th class="r">v24 Use</th></tr></thead>
        <tbody>
          <tr><td>Credit Spread</td><td>HYG ETF daily returns</td><td>Spread tightening (HYG rising)</td><td>Spread widening (HYG falling)</td><td class="r">TQQQ ×0.70 if elevated</td></tr>
          <tr><td>Yield Curve</td><td>TLT/IEI ratio</td><td>Steepening curve</td><td>Flattening / inverted</td><td class="r">Regime signal</td></tr>
          <tr><td>Dollar Strength</td><td>UUP ETF</td><td>Weak dollar (risk-on)</td><td>Strong dollar (tightening)</td><td class="r">Context only</td></tr>
          <tr><td>Gold Flow</td><td>GLD ETF</td><td>Gold rising (inflation hedge)</td><td>Gold falling (risk-on)</td><td class="r">Inflation monitor</td></tr>
        </tbody>
      </table>
    </div>

    <div class="disc-box mt24">
      <div class="disc-title">Macro Data Note</div>
      <div class="disc-body">Daily returns computed from ETF proxies (HYG, TLT/IEI, UUP, GLD). For the actual v24 regime signals (T10Y2Y, UNRATE, VIX, SPY/200MA), run <code>python live_regime.py</code> to fetch live FRED data. The cross-asset panel above is complementary context — not the formal regime inputs.</div>
    </div>
  </div>
</section>
"""

SEC_EARNINGS = f"""
<!-- ═══════════════════════════════════════════ EARNINGS -->
<section id="sec-earnings" class="tab-section">
  <div class="container">
    <p class="eyebrow">Earnings Calendar · Risk Assessment</p>
    <h2 class="section-head">Earnings Risk Monitor</h2>
    <div class="rule"></div>
    <p class="lead">Upcoming and recent earnings events for stocks in the Canyon universe. High-risk events require position review — consider partial exit or hedge before announcement.</p>

    <div class="earn-kpi-grid">
      <div class="earn-kpi">
        <div class="earn-kpi-n" style="color:#B83232">{high_cnt}</div>
        <div class="earn-kpi-l" style="color:#B83232">HIGH RISK events</div>
      </div>
      <div class="earn-kpi">
        <div class="earn-kpi-n" style="color:#B8943F">{med_cnt}</div>
        <div class="earn-kpi-l" style="color:#B8943F">MEDIUM RISK events</div>
      </div>
      <div class="earn-kpi">
        <div class="earn-kpi-n" style="color:#1B6F4A">{low_cnt if low_cnt else (len(earn)-high_cnt-med_cnt)}</div>
        <div class="earn-kpi-l" style="color:#1B6F4A">LOW RISK events</div>
      </div>
    </div>

    <div class="disc-box" style="margin-bottom:20px">
      <div class="disc-title">Risk Classification Logic</div>
      <div class="disc-body">HIGH risk = position held + earnings within window + IV rank &gt; 30 or high surprise history. Review position before announcement. Consider exiting or hedging. MEDIUM = earnings window but low IV or small position. LOW = no current position or earnings far out.</div>
    </div>

    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Earnings Date</th>
            <th style="text-align:center">Days</th>
            <th>Risk</th>
            <th style="text-align:right">Alpha Score</th>
            <th style="text-align:right">IV Rank</th>
            <th style="text-align:right">Last Surprise</th>
            <th>Recommendation</th>
          </tr>
        </thead>
        <tbody>
          {earn_table}
        </tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#AAA;margin-top:12px">Days = trading days from Jun 24, 2026 (negative = past). IV Rank = implied volatility percentile (0–100). Source: earnings_calendar.csv generated Jun 2026.</p>
  </div>
</section>
"""

SEC_INSIDERS = f"""
<!-- ═══════════════════════════════════════════ INSIDERS -->
<section id="sec-insiders" class="tab-section">
  <div class="container">
    <p class="eyebrow">Form 4 Insider Activity · SEC Filings</p>
    <h2 class="section-head">Insider Signal Monitor ⭐</h2>
    <div class="rule"></div>
    <p class="lead">Corporate insiders (officers, directors, 10%+ owners) must file Form 4 with the SEC within 2 business days of a transaction. Cluster insider buying is a historically significant signal of positive management outlook.</p>

    <div class="ins-note">
      <strong>⚠ Data Quality Note:</strong> Current insider data uses Yahoo Finance as a proxy (yfinance ticker.insider_transactions). This is a best-effort approximation — insider counts may include derivative exercises and open-market sales. For production use, cross-reference with SEC EDGAR Form 4 filings directly at edgar.sec.gov. ETF tickers (GLD, IYR) have no insiders by design.
    </div>

    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Data Source</th>
            <th style="text-align:center">Transaction Count</th>
            <th>Net Direction</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {ins_table}
        </tbody>
      </table>
    </div>

    <div class="mt36">
      <h3 style="font-family:'Playfair Display',serif;font-size:20px;margin-bottom:14px">Insider Signal Research Basis</h3>
      <div class="three-col">
        <div class="method-card">
          <div class="method-title">Academic Evidence</div>
          <div class="method-body">Seyhun (1986) showed insider purchases predict 3.5% abnormal returns over 12 months. Lakonishok &amp; Lee (2001) found cluster insider buying (≥3 insiders same month) is the strongest signal — single transactions have much weaker predictive power.</div>
        </div>
        <div class="method-card acc">
          <div class="method-title">v24 Status</div>
          <div class="method-body">Insider signal (sig_insider) is included in the alpha score composite but carries low weight. OOS IC was near zero in testing (t &lt; 1.0). It serves as a tiebreaker for stocks with similar SUE scores rather than a primary driver.</div>
        </div>
        <div class="method-card">
          <div class="method-title">Congressional Activity</div>
          <div class="method-body">Congressional trading (STOCK Act filings) is a public signal. Unmonitored here but worth checking periodically for stocks near legislative activity. SEC EDGAR: efts.sec.gov/LATEST/search-index?q=%22form+4%22</div>
        </div>
      </div>
    </div>
  </div>
</section>
"""

SEC_FACTORS = f"""
<!-- ═══════════════════════════════════════════ FACTOR LAB -->
<section id="sec-factors" class="tab-section">
  <div class="container">
    <p class="eyebrow">Factor Research Lab · IC Decay Analysis</p>
    <h2 class="section-head">Signal Quality &amp; Information Coefficient</h2>
    <div class="rule"></div>
    <p class="lead">Information Coefficient (IC) = Spearman rank correlation between signal scores and next-month returns. IC t-stat &gt; 2.0 = statistically significant. Used to select and weight signals in v24.</p>

    <div class="fac-note">
      <strong>v24 Design Decision:</strong> SUE (earnings surprise) is the primary signal — IC t=+1.70 in OOS (marginal). Momentum (t=+0.02) and quality (t=−0.28) were dropped after OOS IC testing. All IC values below are from the full sample including in-sample training period.
    </div>

    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Factor</th>
            <th style="text-align:right">Mean IC</th>
            <th style="text-align:right">t-stat</th>
            <th style="text-align:right">% Positive</th>
            <th style="text-align:right">Best Month</th>
            <th style="text-align:right">Worst Month</th>
            <th>IC Bar</th>
          </tr>
        </thead>
        <tbody>
          {fac_table}
        </tbody>
      </table>
    </div>

    <div class="fac-chart-wrap">
      <div class="mac-chart-title">Rolling IC — Above 200-Day MA Signal (36 Months)</div>
      <canvas id="fac-chart" height="90"></canvas>
    </div>

    <div class="mt24">
      <div class="tbl-title mt24">IC Interpretation Guide</div>
      <table>
        <thead><tr><th>IC Range</th><th>t-stat</th><th>Interpretation</th><th class="r">Signal Decision</th></tr></thead>
        <tbody>
          <tr><td>&gt; 0.05</td><td>&gt; 2.0</td><td>Statistically significant positive predictive power</td><td class="r pos">Include as primary</td></tr>
          <tr><td>0.02 – 0.05</td><td>1.65 – 2.0</td><td>Marginal — directionally positive but sub-threshold</td><td class="r" style="color:#B8943F">Include as secondary</td></tr>
          <tr><td>-0.02 – 0.02</td><td>&lt; 1.65</td><td>Statistically indistinguishable from noise</td><td class="r neg">Drop from model</td></tr>
          <tr><td>&lt; -0.02</td><td>— (negative)</td><td>Negative predictive power</td><td class="r neg">Drop immediately</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>
"""

SEC_GUIDE = """
<!-- ═══════════════════════════════════════════ GUIDE -->
<section id="sec-guide" class="tab-section">
  <div class="container">
    <p class="eyebrow">User Guide · Canyon Quant v24</p>
    <h2 class="section-head">Strategy Guide &amp; Daily Routine</h2>
    <div class="rule"></div>
    <p class="lead">A plain-English reference for running Canyon Quant v24: what the system does, when to act, and how to interpret each dashboard section.</p>

    <div class="guide-grid">
      <div class="guide-card">
        <div class="guide-head">What This System Does</div>
        <div class="guide-body">Canyon Quant is a rule-based long-only stock strategy. It ranks all S&P 500 stocks by earnings surprise (SUE), selects the top 25, and rebalances monthly. A macro filter adjusts position size in bear regimes. A TQQQ overlay adds 15–23% Nasdaq-100 leveraged exposure in bull markets. No discretionary trades — every decision follows the rulebook.</div>
      </div>
      <div class="guide-card gold">
        <div class="guide-head">When To Act</div>
        <div class="guide-body"><strong>First trading day of each month:</strong> Run data pipeline → check regime → rebalance to 25 new positions. <strong>During the month:</strong> No action unless drawdown stop triggers (&gt;15% from peak). <strong>Never:</strong> Trade intraday, react to single-day drops, add discretionary positions. The regime does not change from one bad day.</div>
      </div>
      <div class="guide-card green">
        <div class="guide-head">What Beta = 0.375 Means</div>
        <div class="guide-body">When SPY falls 10%, expect Canyon to fall ~3.75%. When SPY rises 10%, expect Canyon to rise ~5.7% (upside capture 57%). This is by design — low correlation means genuine portfolio diversification. Compare against the risk-matched benchmark (37.5% SPY + 62.5% cash), not raw SPY.</div>
      </div>
    </div>

    <h3 style="font-family:'Playfair Display',serif;font-size:22px;margin-bottom:16px">Monthly Rebalancing Checklist</h3>
    <ul class="checklist">
      <li>Run <code>python live_regime.py</code> — confirm current regime (BULL or BEAR)</li>
      <li>Check TQQQ effective weight output — apply any stress multipliers</li>
      <li>Run main pipeline → get updated alpha_scores.csv (top 25 by alpha_rank)</li>
      <li>Compare new top-25 to current holdings → identify adds / drops</li>
      <li>Check earnings_calendar.csv — any positions with HIGH risk earnings this month?</li>
      <li>Check factor_ic_history.csv — confirm SUE IC still positive</li>
      <li>Execute trades (paper or live) — target weights from position table</li>
      <li>Log actual fills vs targets in Live Track section</li>
      <li>Update action_cards.csv with new month's context</li>
    </ul>

    <div class="mt36">
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;margin-bottom:16px">Dashboard Tab Reference</h3>
      <table>
        <thead><tr><th>Tab</th><th>What It Shows</th><th class="r">Update Frequency</th></tr></thead>
        <tbody>
          <tr><td><strong>Today</strong></td><td>Current regime KPIs, top-5 positions, TQQQ overlay, action cards</td><td class="r">Run live_regime.py daily</td></tr>
          <tr><td><strong>Live Positions</strong></td><td>Full 25-stock portfolio with weights, avoid list</td><td class="r">Monthly rebalance</td></tr>
          <tr><td><strong>Performance</strong></td><td>OOS backtest returns vs SPY, monthly bar chart</td><td class="r">Static (Jun 2026)</td></tr>
          <tr><td><strong>Attribution</strong></td><td>Return sources: stock alpha vs TQQQ contribution</td><td class="r">Static (Jun 2026)</td></tr>
          <tr><td><strong>Risk</strong></td><td>MDD, drawdown chart, VaR, beta analysis</td><td class="r">Static (Jun 2026)</td></tr>
          <tr><td><strong>Signals</strong></td><td>IC quality of each signal, SUE construction</td><td class="r">Static (methodology)</td></tr>
          <tr><td><strong>Full Universe</strong></td><td>All 495 stocks searchable, 10 signal component bars</td><td class="r">Monthly rebalance</td></tr>
          <tr><td><strong>Macro</strong></td><td>Cross-asset trend cards: credit, yield curve, dollar, gold</td><td class="r">Run live_regime.py</td></tr>
          <tr><td><strong>Earnings</strong></td><td>Upcoming earnings calendar with risk flags</td><td class="r">Monthly update</td></tr>
          <tr><td><strong>Insiders</strong></td><td>Form 4 insider activity proxy (yfinance)</td><td class="r">Monthly update</td></tr>
          <tr><td><strong>Factor Lab</strong></td><td>IC decay curves, signal quality statistics</td><td class="r">Monthly update</td></tr>
          <tr><td><strong>Regime</strong></td><td>Bull/bear classification logic and history</td><td class="r">Run live_regime.py</td></tr>
          <tr><td><strong>Methodology</strong></td><td>Full strategy specification, parameter choices</td><td class="r">Static (reference)</td></tr>
          <tr><td><strong>Scorecard</strong></td><td>Institutional quality checklist (7.81/10)</td><td class="r">Static (Jun 2026)</td></tr>
        </tbody>
      </table>
    </div>

    <div class="disc-box mt24">
      <div class="disc-title">Research Disclaimer</div>
      <div class="disc-body">
        Canyon Quant v24 is a research and paper-trading system. It does not constitute investment advice, a solicitation, or an offer to buy or sell securities. Past performance is not indicative of future results. The out-of-sample backtest (2019–2026) shows OOS FF6 alpha of +6.80%/yr at t=+1.73 — this does not meet the conventional 5% significance threshold (t&gt;2.0 required). The combined 2001–2026 period reaches t=+2.18 (5% sig) but mixes in-sample and OOS data.<br><br>
        All data sourced from free public sources (Yahoo Finance, FRED, SEC EDGAR). No paid data subscriptions. No broker connections. System is for research purposes only.
      </div>
    </div>
  </div>
</section>
"""

# ── JS for new sections ────────────────────────────────────────────────────────
NEW_JS = f"""
// ── UNIVERSE DATA ──────────────────────────────────────────────────────────
const UNIVERSE = {universe_js};
// Columns: [rank, ticker, score, signal, sector, crowding, ml, qual, rev, eps, sent, sqz, ins, opt, ens, mom]

let uniFiltered = [...UNIVERSE];
let uniPageNum = 0;
const UNI_PER_PAGE = 50;

function sigBar(v) {{
  const w = Math.round(v);
  const c = v >= 70 ? '#6BCCA0' : v >= 50 ? '#B8943F' : '#EF9090';
  return `<span class="sig-bar" style="width:${{w/2}}px;max-width:50px;background:${{c}}"></span>`;
}}

function renderUniPage() {{
  const tbody = document.getElementById('uni-tbody');
  if (!tbody) return;
  const start = uniPageNum * UNI_PER_PAGE;
  const slice = uniFiltered.slice(start, start + UNI_PER_PAGE);
  if (slice.length === 0) {{
    tbody.innerHTML = '<tr><td colspan="15" style="text-align:center;padding:20px;color:#888">No stocks match the filter.</td></tr>';
    return;
  }}
  const sigClass = s => s === 'LONG' ? 'signal-long' : s === 'SHORT_AVOID' ? 'signal-avoid' : 'signal-hold';
  tbody.innerHTML = slice.map(r => {{
    const [rank,ticker,score,signal,sector,crowd,...sigs] = r;
    const bars = sigs.map(sigBar).join('');
    return `<tr>
      <td style="color:#999;font-size:11px">#${{rank}}</td>
      <td class="td-ticker">${{ticker}}</td>
      <td style="font-weight:700;color:${{score>=70?'#1B6F4A':score>=50?'#B8943F':'#B83232'}}">${{score.toFixed(1)}}</td>
      <td><span class="${{sigClass(signal)}}">${{signal}}</span></td>
      <td style="font-size:11px;color:#666">${{sector||'—'}}</td>
      ${{sigs.map(sigBar).map((b,i) => `<td style="padding:4px 2px">${{b}}</td>`).join('')}}
    </tr>`;
  }}).join('');
  const total = uniFiltered.length;
  const pages = Math.ceil(total / UNI_PER_PAGE);
  document.getElementById('uni-page-info').textContent = `Page ${{uniPageNum+1}} of ${{pages}}`;
  document.getElementById('uni-count').textContent = `Showing ${{Math.min(start+UNI_PER_PAGE,total)}} of ${{total}} stocks`;
}}

function filterUni() {{
  const q = (document.getElementById('uni-search').value||'').toLowerCase();
  const sig = document.getElementById('uni-sig').value;
  const sec = document.getElementById('uni-sector').value;
  uniFiltered = UNIVERSE.filter(r => {{
    const [rank,ticker,score,signal,sector] = r;
    if (q && !ticker.toLowerCase().includes(q) && !(sector||'').toLowerCase().includes(q)) return false;
    if (sig !== 'all' && signal !== sig) return false;
    if (sec !== 'all' && sector !== sec) return false;
    return true;
  }});
  uniPageNum = 0;
  renderUniPage();
  document.getElementById('uni-count').textContent = `Showing ${{uniFiltered.length}} stocks`;
}}

function uniPage(dir) {{
  const pages = Math.ceil(uniFiltered.length / UNI_PER_PAGE);
  uniPageNum = Math.max(0, Math.min(uniPageNum + dir, pages - 1));
  renderUniPage();
}}

function sortUni(col) {{
  const dir = (window._uniSortDir || 1) * -1;
  window._uniSortDir = dir;
  uniFiltered.sort((a,b) => {{
    const av = typeof a[col] === 'number' ? a[col] : (a[col]||'').toString();
    const bv = typeof b[col] === 'number' ? b[col] : (b[col]||'').toString();
    return av < bv ? -dir : av > bv ? dir : 0;
  }});
  uniPageNum = 0;
  renderUniPage();
}}

// ── MACRO CHART ─────────────────────────────────────────────────────────────
{macro_spark_js}

function initMacroChart() {{
  const ctx = document.getElementById('macro-chart');
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: SPARK_DATES,
      datasets: [
        {{label:'Credit Spread', data:SPARK.credit_spread, borderColor:'#EF9090', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3}},
        {{label:'Yield Curve',   data:SPARK.yield_curve,   borderColor:'#6BCCA0', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3}},
        {{label:'Dollar',        data:SPARK.dollar,        borderColor:'#B8943F', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3}},
        {{label:'Gold',          data:SPARK.gold,          borderColor:'#1B2A4A', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3}},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:true,
      plugins:{{legend:{{position:'top',labels:{{font:{{size:11}},boxWidth:12}}}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{
        x:{{ticks:{{maxTicksLimit:8,font:{{size:10}}}},grid:{{color:'rgba(0,0,0,.04)'}}}},
        y:{{ticks:{{font:{{size:10}},callback:v=>v.toFixed(2)+'%'}},grid:{{color:'rgba(0,0,0,.04)'}}}}
      }}
    }}
  }});
}}

// ── FACTOR LAB CHART ─────────────────────────────────────────────────────────
const IC200 = {ic200_js};
const IC200_DATES = {ic200_dates};

function initFacChart() {{
  const ctx = document.getElementById('fac-chart');
  if (!ctx) return;
  const colors = IC200.map(v => v >= 0 ? '#6BCCA0' : '#EF9090');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: IC200_DATES,
      datasets: [{{
        label: 'Monthly IC (Above 200MA)',
        data: IC200, backgroundColor: colors, borderWidth: 0
      }}]
    }},
    options: {{
      responsive:true, maintainAspectRatio:true,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.parsed.y.toFixed(3)}}}}}},
      scales:{{
        x:{{ticks:{{maxTicksLimit:12,font:{{size:10}}}},grid:{{display:false}}}},
        y:{{ticks:{{font:{{size:10}}}},grid:{{color:'rgba(0,0,0,.04)'}},title:{{display:true,text:'IC',font:{{size:10}}}}}}
      }}
    }}
  }});
}}

// ── INIT ON SHOW ──────────────────────────────────────────────────────────────
(function() {{
  // Render universe immediately on load
  renderUniPage();

  // Track tab changes to init charts lazily
  const origShow = window.showTab;
  window.showTab = function(name) {{
    origShow(name);
    if (name === 'macro')   {{ initMacroChart(); }}
    if (name === 'factors') {{ initFacChart();   }}
    if (name === 'universe') {{ renderUniPage(); }}
  }};
}})();
"""

# ── NEW NAV TABS ──────────────────────────────────────────────────────────────
NEW_NAV = """      <a onclick="showTab('universe')" id="tab-universe"      >Universe ◉</a>
      <a onclick="showTab('macro')"    id="tab-macro"          >Macro</a>
      <a onclick="showTab('earnings')" id="tab-earnings"       >Earnings</a>
      <a onclick="showTab('insiders')" id="tab-insiders"       >Insiders ⭐</a>
      <a onclick="showTab('factors')"  id="tab-factors"        >Factor Lab</a>
      <a onclick="showTab('guide')"    id="tab-guide"          >Guide</a>"""

# ── Patch HTML ────────────────────────────────────────────────────────────────
html = HTML.read_text(encoding='utf-8')

# 1. Inject CSS before closing </style>
if NEW_CSS.strip()[:20] not in html:
    html = html.replace('  </style>', NEW_CSS + '\n  </style>', 1)
    print('✓ CSS injected')

# 2. Add nav tabs after tab-score
insert_after = '      <a onclick="showTab(\'score\')"   id="tab-score"          >Scorecard ★</a>'
if 'tab-universe' not in html:
    html = html.replace(insert_after, insert_after + '\n' + NEW_NAV)
    print('✓ Nav tabs added (6 new tabs)')

# 3. Insert new sections before the SCRIPT block
script_marker = '<!-- ═══════════════════════════════════════════════════ SCRIPT -->'
new_sections = (
    SEC_FULL_SIGNALS + '\n' +
    SEC_MACRO + '\n' +
    SEC_EARNINGS + '\n' +
    SEC_INSIDERS + '\n' +
    SEC_FACTORS + '\n' +
    SEC_GUIDE + '\n'
)
if 'sec-universe' not in html:
    html = html.replace(script_marker, new_sections + script_marker)
    print('✓ 6 new sections inserted')

# 4. Inject JS before closing </script>
closing_script = '</script>'
last_script_pos = html.rfind(closing_script)
if last_script_pos != -1 and 'UNIVERSE =' not in html:
    html = html[:last_script_pos] + NEW_JS + '\n' + html[last_script_pos:]
    print('✓ JS injected (universe render + charts)')

HTML.write_text(html, encoding='utf-8')
line_count = html.count('\n')
print(f'\n✓ Dashboard saved: {HTML.name}')
print(f'  Lines: {line_count:,}  (was ~1,318)')
print(f'  New tabs: Universe ◉, Macro, Earnings, Insiders ⭐, Factor Lab, Guide')
print(f'  Universe: {len(alpha)} stocks with 10-signal breakdown')
print(f'  Macro: 4 cross-asset cards + 30d chart')
print(f'  Earnings: {len(earn)} events by risk tier')
print(f'  Insiders: {len(insiders)} tickers from Form 4 proxy')
print(f'  Factor Lab: {len(fic.columns)-2} IC series, 36-month IC chart')
print(f'\n  Refresh the browser to see the expanded dashboard.')
