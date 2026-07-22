#!/usr/bin/env python3
"""
Canyon v9 — Step 72: Weekly Research Report Generator
Generates a self-contained HTML report. No CDN, no external server.
Usage:
    python3 step72_weekly_report.py
    python3 step72_weekly_report.py --open
    python3 step72_weekly_report.py --date 2026-05-18
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# ── Key output files ──────────────────────────────────────────────────────────
KEY_FILES = {
    "enhanced_ml_scores":       ROOT / "enhanced_ml_scores.csv",
    "ml_signal_scores":         ROOT / "ml_signal_scores.csv",
    "ml_ic_comparison":         ROOT / "ml_ic_comparison.csv",
    "paper_sim_positions":      ROOT / "paper_sim_positions.csv",
    "paper_sim_summary":        ROOT / "paper_sim_summary.csv",
    "paper_sim_nav":            ROOT / "paper_sim_nav.csv",
    "portfolio_optimized_weights": ROOT / "portfolio_optimized_weights.csv",
    "shap_summary":             ROOT / "shap_summary.csv",
    "earnings_calendar":        ROOT / "earnings_calendar.csv",
    "alerts":                   ROOT / "alerts.csv",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def safe_read(path: Path, **kw) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            pass
    return pd.DataFrame()


def load_all_data() -> dict:
    d = {}
    # ML scores — prefer enhanced
    if KEY_FILES["enhanced_ml_scores"].exists():
        d["ml_scores"] = safe_read(KEY_FILES["enhanced_ml_scores"])
        d["ml_scores_source"] = "enhanced_ml_scores.csv"
    else:
        d["ml_scores"] = safe_read(KEY_FILES["ml_signal_scores"])
        d["ml_scores_source"] = "ml_signal_scores.csv (fallback)"

    d["ml_ic"] = safe_read(KEY_FILES["ml_ic_comparison"])
    d["positions"] = safe_read(KEY_FILES["paper_sim_positions"])
    d["summary"] = safe_read(KEY_FILES["paper_sim_summary"])
    d["nav"] = safe_read(KEY_FILES["paper_sim_nav"])
    d["weights"] = safe_read(KEY_FILES["portfolio_optimized_weights"])
    d["shap"] = safe_read(KEY_FILES["shap_summary"])
    d["earnings"] = safe_read(KEY_FILES["earnings_calendar"])
    d["alerts"] = safe_read(KEY_FILES["alerts"])
    return d


def get_market_regime() -> tuple[str, str]:
    """Return (label, badge_color) inferred from SPY vs 200d MA via yfinance."""
    try:
        import yfinance as yf  # optional dep
        spy = yf.download("SPY", period="300d", progress=False, auto_adjust=True)
        if spy.empty:
            raise ValueError("empty")
        close = spy["Close"].squeeze()
        ma200 = close.rolling(200).mean().iloc[-1]
        last = close.iloc[-1]
        if last > ma200 * 1.02:
            return "Bull", "#27ae60"
        elif last < ma200 * 0.98:
            return "Bear", "#c0392b"
        else:
            return "Neutral", "#e67e22"
    except Exception:
        return "N/A", "#7f8c8d"


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f7fa; color: #222; font-size: 14px; }
.page { max-width: 1200px; margin: 0 auto; padding: 20px; }
/* header */
.report-header { background: #1e3a5f; color: #fff; padding: 28px 32px;
                 border-radius: 8px; margin-bottom: 20px;
                 display: flex; align-items: center; justify-content: space-between; }
.report-header h1 { font-size: 24px; font-weight: 700; letter-spacing: .5px; }
.report-header .meta { font-size: 12px; color: #a8c4e0; margin-top: 4px; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 20px;
         font-weight: 700; font-size: 13px; color: #fff; }
/* sections */
.section { background: #fff; border-radius: 8px; padding: 20px 24px;
           margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.07); }
.section-title { font-size: 16px; font-weight: 700; color: #1e3a5f;
                 border-bottom: 2px solid #e8edf3; padding-bottom: 10px;
                 margin-bottom: 16px; }
/* tables */
table { width: 100%; border-collapse: collapse; }
th { background: #1e3a5f; color: #fff; padding: 8px 10px; text-align: left;
     font-size: 12px; font-weight: 600; }
td { padding: 7px 10px; border-bottom: 1px solid #eef1f5; font-size: 13px; }
tr:hover td { background: #f0f4fa; }
/* metric cards */
.cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
.card { flex: 1; min-width: 140px; background: #f0f4fa; border-radius: 6px;
        padding: 14px 16px; text-align: center; }
.card .label { font-size: 11px; color: #666; margin-bottom: 4px; }
.card .value { font-size: 22px; font-weight: 700; color: #1e3a5f; }
.card .value.pos { color: #27ae60; }
.card .value.neg { color: #c0392b; }
/* bar chart */
.bar-row { display: flex; align-items: center; margin-bottom: 6px; }
.bar-label { width: 130px; font-size: 12px; text-align: right;
             padding-right: 10px; color: #444; white-space: nowrap; overflow: hidden;
             text-overflow: ellipsis; }
.bar-track { flex: 1; background: #e8edf3; border-radius: 3px; height: 20px; }
.bar-fill  { height: 20px; border-radius: 3px; background: #1e3a5f;
             display: flex; align-items: center; padding-left: 6px; }
.bar-fill .bar-val { font-size: 11px; color: #fff; font-weight: 600; }
.bar-fill.green { background: #27ae60; }
/* status / alerts */
.stale { color: #c0392b; font-weight: 600; }
.ok    { color: #27ae60; }
.tag   { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 700; color: #fff; }
.tag-crit { background: #c0392b; }
.tag-warn { background: #e67e22; }
.tag-info { background: #2980b9; }
.ic-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; }
.ic-chip { background: #eaf1fb; border-radius: 6px; padding: 8px 14px;
           font-size: 12px; }
.ic-chip strong { color: #1e3a5f; font-size: 15px; }
.placeholder { color: #999; font-style: italic; font-size: 13px;
               padding: 20px; text-align: center; background: #f9f9f9;
               border-radius: 6px; }
"""


# ── Section renderers ─────────────────────────────────────────────────────────
def render_section_header(report_date: str, regime: str, regime_color: str) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""
<div class="report-header">
  <div>
    <h1>Canyon v9 &mdash; Weekly Research Report</h1>
    <div class="meta">Generated: {now_str} &nbsp;|&nbsp; Week ending: {report_date}</div>
  </div>
  <div>
    <span class="badge" style="background:{regime_color};">&#x25CF; {regime} Regime</span>
  </div>
</div>"""


def render_section_ml(data: dict) -> str:
    df = data.get("ml_scores", pd.DataFrame())
    source = data.get("ml_scores_source", "—")
    score_col = next((c for c in ["enhanced_score", "ml_score", "score"] if c in df.columns), None)

    top10_html = ""
    if not df.empty and score_col:
        top = df.nlargest(10, score_col)[["ticker", score_col] if "ticker" in df.columns else [score_col]].copy()
        top[score_col] = top[score_col].round(4)
        rows = "".join(
            f"<tr><td>{i+1}</td><td><strong>{r.get('ticker','—')}</strong></td>"
            f"<td>{r[score_col]:.4f}</td></tr>"
            for i, (_, r) in enumerate(top.iterrows())
        )
        top10_html = f"""
<table>
  <thead><tr><th>#</th><th>Ticker</th><th>Score</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="font-size:11px;color:#888;margin-top:6px;">Source: {source}</p>"""
    else:
        top10_html = '<p class="placeholder">ML score data unavailable.</p>'

    # IC comparison chips
    ic_df = data.get("ml_ic", pd.DataFrame())
    ic_html = ""
    if not ic_df.empty and "signal" in ic_df.columns and "mean_ic" in ic_df.columns:
        chips = "".join(
            f'<div class="ic-chip"><div>{r["signal"]}</div>'
            f'<strong>{float(r["mean_ic"]):.4f}</strong>'
            f'<div style="font-size:10px;color:#888;">{r.get("status","")}</div></div>'
            for _, r in ic_df.head(5).iterrows()
        )
        ic_html = f'<div style="margin-top:14px;font-weight:700;font-size:13px;color:#1e3a5f;">IC Comparison</div><div class="ic-row">{chips}</div>'

    return f"""
<div class="section">
  <div class="section-title">&#x1F9E0; ML Signal Summary</div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;">
    <div style="flex:1;min-width:260px;">{top10_html}</div>
    <div style="flex:1;min-width:240px;">{ic_html}</div>
  </div>
</div>"""


def _pnl_class(val) -> str:
    try:
        return "pos" if float(val) >= 0 else "neg"
    except Exception:
        return ""


def render_section_paper(data: dict) -> str:
    pos_df = data.get("positions", pd.DataFrame())
    summ_df = data.get("summary", pd.DataFrame())

    # Metric cards
    cards_html = ""
    if not summ_df.empty:
        row = summ_df.iloc[0]
        def cv(k, fmt=".2f"):
            try:
                return f"{float(row[k]):{fmt}}" if k in row else "—"
            except Exception:
                return "—"
        total_pnl = cv("total_pnl")
        pnl_cls = _pnl_class(row.get("total_pnl", 0))
        win_rate = f"{float(row['win_rate'])*100:.1f}%" if "win_rate" in row else "—"
        n_open = str(int(float(row["n_open"]))) if "n_open" in row else "—"
        ret_pct = f"{float(row['total_return_pct']):.2f}%" if "total_return_pct" in row else "—"
        cards_html = f"""
<div class="cards">
  <div class="card"><div class="label">Total P&amp;L ($)</div>
    <div class="value {pnl_cls}">{total_pnl}</div></div>
  <div class="card"><div class="label">Return %</div>
    <div class="value {pnl_cls}">{ret_pct}</div></div>
  <div class="card"><div class="label">Win Rate</div>
    <div class="value">{win_rate}</div></div>
  <div class="card"><div class="label">Open Positions</div>
    <div class="value">{n_open}</div></div>
</div>"""

    # Positions table
    pos_html = ""
    if not pos_df.empty:
        cols = ["ticker", "entry_date", "entry_price", "shares", "cost_basis",
                "stop_price", "target_price", "sector"]
        show = [c for c in cols if c in pos_df.columns]
        if "unrealised_pnl" in pos_df.columns:
            show.append("unrealised_pnl")
        header = "".join(f"<th>{c}</th>" for c in show)
        rows = ""
        for _, r in pos_df.iterrows():
            cells = ""
            for c in show:
                val = r.get(c, "—")
                if c == "unrealised_pnl":
                    cls = _pnl_class(val)
                    try:
                        val = f"{float(val):,.2f}"
                    except Exception:
                        pass
                    cells += f'<td style="font-weight:600;" class="{cls}">{val}</td>'
                else:
                    cells += f"<td>{val}</td>"
            rows += f"<tr>{cells}</tr>"
        pos_html = f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
    else:
        pos_html = '<p class="placeholder">No open positions.</p>'

    # NAV placeholder
    nav_df = data.get("nav", pd.DataFrame())
    nav_html = ""
    if nav_df.empty:
        nav_html = '<p class="placeholder">NAV curve data (paper_sim_nav.csv) not yet available.</p>'
    else:
        # Simple sparkline as inline SVG
        nav_col = next((c for c in ["nav", "portfolio_nav", "value"] if c in nav_df.columns), None)
        if nav_col:
            vals = nav_df[nav_col].dropna().tolist()
            if len(vals) >= 2:
                mn, mx = min(vals), max(vals)
                rng = mx - mn if mx != mn else 1
                W, H = 600, 80
                pts = " ".join(
                    f"{int(i/(len(vals)-1)*W)},{int(H - (v - mn)/rng*H)}"
                    for i, v in enumerate(vals)
                )
                nav_html = f"""
<div style="margin-top:14px;">
  <div style="font-size:12px;color:#666;margin-bottom:4px;">NAV Curve (portfolio)</div>
  <svg width="{W}" height="{H}" style="background:#f5f7fa;border-radius:4px;">
    <polyline points="{pts}" fill="none" stroke="#1e3a5f" stroke-width="2"/>
  </svg>
</div>"""

    return f"""
<div class="section">
  <div class="section-title">&#x1F4BC; Paper Portfolio Status</div>
  {cards_html}
  {pos_html}
  {nav_html}
</div>"""


def render_section_optimizer(data: dict) -> str:
    df = data.get("weights", pd.DataFrame())
    if df.empty or "ticker" not in df.columns or "max_sharpe" not in df.columns:
        return f"""
<div class="section">
  <div class="section-title">&#x2696;&#xFE0F; Portfolio Optimizer — Max-Sharpe Weights</div>
  <p class="placeholder">portfolio_optimized_weights.csv unavailable.</p>
</div>"""

    top = df.nlargest(12, "max_sharpe")[["ticker", "max_sharpe"]].copy()
    max_w = top["max_sharpe"].max()
    bars = ""
    for _, r in top.iterrows():
        pct = r["max_sharpe"] / max_w * 100 if max_w > 0 else 0
        bars += f"""
<div class="bar-row">
  <div class="bar-label">{r['ticker']}</div>
  <div class="bar-track">
    <div class="bar-fill" style="width:{pct:.1f}%;">
      <span class="bar-val">{r['max_sharpe']:.3f}</span>
    </div>
  </div>
</div>"""

    return f"""
<div class="section">
  <div class="section-title">&#x2696;&#xFE0F; Portfolio Optimizer — Max-Sharpe Weights</div>
  {bars}
</div>"""


def render_section_shap(data: dict) -> str:
    df = data.get("shap", pd.DataFrame())
    if df.empty or "feature" not in df.columns:
        return f"""
<div class="section">
  <div class="section-title">&#x1F52C; SHAP Top Drivers</div>
  <p class="placeholder">shap_summary.csv unavailable.</p>
</div>"""

    shap_col = next((c for c in ["avg_abs_shap", "rf_mean_abs_shap", "mean_abs_shap"] if c in df.columns), None)
    if not shap_col:
        return f"""
<div class="section">
  <div class="section-title">&#x1F52C; SHAP Top Drivers</div>
  <p class="placeholder">No SHAP value column found.</p>
</div>"""

    top = df.nlargest(10, shap_col)[["feature", shap_col]].copy()
    max_v = top[shap_col].max()
    bars = ""
    for _, r in top.iterrows():
        pct = r[shap_col] / max_v * 100 if max_v > 0 else 0
        bars += f"""
<div class="bar-row">
  <div class="bar-label">{r['feature']}</div>
  <div class="bar-track">
    <div class="bar-fill green" style="width:{pct:.1f}%;">
      <span class="bar-val">{r[shap_col]:.4f}</span>
    </div>
  </div>
</div>"""

    return f"""
<div class="section">
  <div class="section-title">&#x1F52C; SHAP Top Drivers (mean |SHAP|)</div>
  {bars}
</div>"""


def render_section_earnings(data: dict, report_date: str) -> str:
    df = data.get("earnings", pd.DataFrame())
    if df.empty or "ticker" not in df.columns:
        return f"""
<div class="section">
  <div class="section-title">&#x1F4C5; Earnings Watchlist</div>
  <p class="placeholder">earnings_calendar.csv unavailable.</p>
</div>"""

    date_col = next((c for c in ["next_earnings", "earnings_date", "date"] if c in df.columns), None)
    upcoming = df.copy()
    if date_col:
        try:
            ref = pd.to_datetime(report_date)
            upcoming[date_col] = pd.to_datetime(upcoming[date_col], errors="coerce")
            upcoming = upcoming[
                (upcoming[date_col] >= ref) & (upcoming[date_col] <= ref + timedelta(days=7))
            ].sort_values(date_col)
        except Exception:
            pass

    if upcoming.empty:
        body = '<p class="placeholder">No earnings in the next 7 days.</p>'
    else:
        show_cols = [c for c in [date_col, "ticker", "eps_surprise_pct"] if c and c in upcoming.columns]
        header = "".join(f"<th>{c}</th>" for c in show_cols)
        rows = ""
        for _, r in upcoming.iterrows():
            cells = "".join(f"<td>{r.get(c, '—')}</td>" for c in show_cols)
            rows += f"<tr>{cells}</tr>"
        body = f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"

    return f"""
<div class="section">
  <div class="section-title">&#x1F4C5; Earnings Watchlist (next 7 days)</div>
  {body}
</div>"""


def render_section_alerts(data: dict) -> str:
    df = data.get("alerts", pd.DataFrame())
    if df.empty:
        return f"""
<div class="section">
  <div class="section-title">&#x26A0;&#xFE0F; Alert Summary</div>
  <p class="placeholder">alerts.csv not found or empty — no active alerts.</p>
</div>"""

    level_col = next((c for c in ["level", "severity", "type"] if c in df.columns), None)
    n_crit = n_warn = 0
    if level_col:
        levels = df[level_col].astype(str).str.upper()
        n_crit = int((levels == "CRITICAL").sum())
        n_warn = int((levels == "WARNING").sum())

    counts = f"""
<div class="cards">
  <div class="card"><div class="label">CRITICAL</div>
    <div class="value {'neg' if n_crit else ''}">{n_crit}</div></div>
  <div class="card"><div class="label">WARNING</div>
    <div class="value {'neg' if n_warn else ''}">{n_warn}</div></div>
  <div class="card"><div class="label">Total Alerts</div>
    <div class="value">{len(df)}</div></div>
</div>"""

    # Top 5
    msg_col = next((c for c in ["message", "msg", "description", "alert"] if c in df.columns), None)
    top5_html = ""
    if msg_col:
        rows = ""
        for _, r in df.head(5).iterrows():
            lv = str(r.get(level_col, "INFO")).upper() if level_col else "INFO"
            tag_cls = "tag-crit" if lv == "CRITICAL" else ("tag-warn" if lv == "WARNING" else "tag-info")
            rows += f"<tr><td><span class='tag {tag_cls}'>{lv}</span></td><td>{r.get(msg_col,'—')}</td></tr>"
        top5_html = f"""<table>
  <thead><tr><th>Level</th><th>Message</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""

    return f"""
<div class="section">
  <div class="section-title">&#x26A0;&#xFE0F; Alert Summary</div>
  {counts}
  {top5_html}
</div>"""


def render_section_run_status() -> str:
    now = datetime.now()
    rows = ""
    for name, path in KEY_FILES.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age_h = (now - mtime).total_seconds() / 3600
            age_str = f"{age_h:.1f}h ago"
            stale = age_h > 24
            cls = "stale" if stale else "ok"
            flag = " &#x25CF; STALE" if stale else " &#x2714;"
            rows += f"<tr><td>{path.name}</td><td>{mtime.strftime('%Y-%m-%d %H:%M')}</td>" \
                    f"<td class='{cls}'>{age_str}{flag}</td></tr>"
        else:
            rows += f"<tr><td>{path.name}</td><td>—</td>" \
                    f"<td class='stale'>MISSING &#x25CF;</td></tr>"

    return f"""
<div class="section">
  <div class="section-title">&#x1F9FE; Run Status — Key Output Files</div>
  <table>
    <thead><tr><th>File</th><th>Last Modified</th><th>Status</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


# ── HTML builder ──────────────────────────────────────────────────────────────
def build_html(sections: list[str]) -> str:
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Canyon v9 — Weekly Research Report</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page">
{body}
<p style="text-align:center;color:#bbb;font-size:11px;margin-top:20px;padding-bottom:20px;">
  Canyon v9 Quantitative Research System &mdash; Internal Use Only
</p>
</div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Weekly Report Generator")
    parser.add_argument("--open", action="store_true", help="Open report in browser after generating")
    parser.add_argument("--date", type=str, default=None, help="Week-end date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    report_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(report_date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: Invalid date format '{report_date}'. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    print(f"[Canyon v9] Generating weekly report for week ending {report_date}...")

    print("  Loading data...", end=" ", flush=True)
    data = load_all_data()
    print("done.")

    print("  Fetching market regime...", end=" ", flush=True)
    regime, regime_color = get_market_regime()
    print(f"{regime}.")

    print("  Rendering sections...", end=" ", flush=True)
    sections = [
        render_section_header(report_date, regime, regime_color),
        render_section_ml(data),
        render_section_paper(data),
        render_section_optimizer(data),
        render_section_shap(data),
        render_section_earnings(data, report_date),
        render_section_alerts(data),
        render_section_run_status(),
    ]
    print("done.")

    html = build_html(sections)

    date_slug = report_date.replace("-", "")
    dated_path = ROOT / f"weekly_report_{date_slug}.html"
    latest_path = ROOT / "weekly_report_latest.html"

    dated_path.write_text(html, encoding="utf-8")
    latest_path.write_text(html, encoding="utf-8")

    print(f"  Saved: {dated_path}")
    print(f"  Saved: {latest_path}")

    if args.open:
        import webbrowser
        webbrowser.open(latest_path.as_uri())
        print(f"  Opened in browser: {latest_path}")


if __name__ == "__main__":
    main()
