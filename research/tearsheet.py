"""
W42: Factor Tearsheet HTML Generator
=====================================
Generates an institutional-grade HTML factor tearsheet with charts and tables.

Sections:
  1. Executive Summary — key performance metrics
  2. Cumulative Return Chart — strategy vs SPY vs equal-weight benchmark
  3. Rolling Sharpe — 12-month rolling Sharpe ratio
  4. Factor IC Decay — empirical half-lives chart
  5. Signal Correlation Heatmap — from signal_correlation_matrix.csv
  6. Factor Exposure Table — current Barra exposures
  7. Backtest Credibility Scorecard — from backtest_scorecard.csv
  8. Data Quality Summary — from data_quality_report.csv

Outputs:
  canyon_factor_tearsheet.html — standalone HTML (no external dependencies)

Usage:
    from research.tearsheet import generate_tearsheet
    generate_tearsheet()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


def _load_or_empty(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _perf_metrics(returns: pd.Series) -> dict:
    """Compute standard performance metrics from a return series."""
    if returns.empty or len(returns) < 3:
        return {}
    r = returns.dropna()
    ann_ret  = float(r.mean() * 12)
    ann_vol  = float(r.std() * np.sqrt(12))
    sharpe   = ann_ret / (ann_vol + 1e-9)
    cum      = (1 + r).cumprod()
    peak     = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd   = float(-drawdown.min())
    calmar   = ann_ret / (max_dd + 1e-9)
    hit_rate = float((r > 0).mean())
    return {
        "Annual Return":   f"{ann_ret:.1%}",
        "Annual Vol":      f"{ann_vol:.1%}",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Max Drawdown":    f"{-max_dd:.1%}",
        "Calmar Ratio":    f"{calmar:.2f}",
        "Hit Rate":        f"{hit_rate:.0%}",
        "Months of Data":  str(len(r)),
    }


def _df_to_html_table(df: pd.DataFrame, title: str, max_rows: int = 20) -> str:
    """Convert DataFrame to styled HTML table."""
    if df.empty:
        return f"<h3>{title}</h3><p><em>Data not available</em></p>"

    rows_html = ""
    for _, row in df.head(max_rows).iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row.values)
        rows_html += f"<tr>{cells}</tr>\n"

    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"""
<h3>{title}</h3>
<table class="data-table">
<thead><tr>{headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
"""


def _mini_line_chart(labels: list, values: list, title: str,
                     color: str = "#2563eb") -> str:
    """Generate a simple inline SVG line chart."""
    if not values or len(values) < 2:
        return f"<p><em>{title}: insufficient data</em></p>"

    w, h, pad = 600, 200, 40
    v_min, v_max = min(values), max(values)
    v_range = v_max - v_min or 1.0

    def px(i, v):
        x = pad + (i / (len(values) - 1)) * (w - 2 * pad)
        y = h - pad - ((v - v_min) / v_range) * (h - 2 * pad)
        return x, y

    points = " ".join(f"{px(i,v)[0]:.1f},{px(i,v)[1]:.1f}" for i, v in enumerate(values))
    zero_y = h - pad - ((0 - v_min) / v_range) * (h - 2 * pad)
    zero_line = f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{w-pad}" y2="{zero_y:.1f}" stroke="#e5e7eb" stroke-dasharray="4"/>' \
                if v_min <= 0 <= v_max else ""

    # Axis labels
    y_labels = ""
    for pct in [0, 0.25, 0.5, 0.75, 1.0]:
        v_label = v_min + pct * v_range
        y_pos   = h - pad - pct * (h - 2 * pad)
        y_labels += f'<text x="{pad-5}" y="{y_pos:.1f}" text-anchor="end" font-size="10" fill="#9ca3af">{v_label:.2f}</text>'

    return f"""
<h4>{title}</h4>
<svg width="{w}" height="{h}" style="background:#f9fafb;border-radius:8px">
  {zero_line}
  {y_labels}
  <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>
</svg>
"""


def generate_tearsheet(output_path: Optional[Path] = None) -> Path:
    """
    Generate full HTML factor tearsheet.

    Returns path to generated HTML file.
    """
    if output_path is None:
        output_path = ROOT / "canyon_factor_tearsheet.html"

    today_str = datetime.today().strftime("%Y-%m-%d %H:%M")

    # Load all data sources
    bt_monthly   = _load_or_empty(ROOT / "v11_backtest_monthly.csv")
    scorecard    = _load_or_empty(ROOT / "backtest_scorecard.csv")
    signal_ic    = _load_or_empty(ROOT / "v11_signal_ic.csv")
    halflife_df  = _load_or_empty(ROOT / "signal_halflife.csv")
    corr_matrix  = _load_or_empty(ROOT / "signal_correlation_matrix.csv", index_col=0)
    factor_exp   = _load_or_empty(ROOT / "factor_exposure_daily.csv")
    dq_report    = _load_or_empty(ROOT / "data_quality_report.csv")
    shap_df      = _load_or_empty(ROOT / "shap_feature_importance.csv")

    # Strategy returns
    ret_col = None
    for c in ["ls_return", "long_short_return", "oos_return"]:
        if c in bt_monthly.columns:
            ret_col = c
            break

    cum_chart_html  = ""
    rolling_sr_html = ""
    if ret_col and not bt_monthly.empty:
        r = bt_monthly[ret_col].dropna()
        cum_ret = ((1 + r).cumprod() - 1).tolist()
        labels  = list(range(len(cum_ret)))
        cum_chart_html = _mini_line_chart(
            labels, cum_ret, "Cumulative Long-Short Return (OOS)", "#059669"
        )
        # Rolling 12-month Sharpe
        roll_sharpe = r.rolling(12).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
        ).dropna().tolist()
        rolling_sr_html = _mini_line_chart(
            list(range(len(roll_sharpe))), roll_sharpe,
            "Rolling 12-Month Sharpe Ratio", "#7c3aed"
        )

    perf = _perf_metrics(bt_monthly[ret_col].dropna()) if ret_col and not bt_monthly.empty else {}

    # Scorecard summary
    overall_score = float(scorecard.iloc[0]["overall_score"]) \
                    if not scorecard.empty and "overall_score" in scorecard.columns else "N/A"
    scorecard_html = _df_to_html_table(
        scorecard[["dimension", "value", "score", "status"]].round(3) if not scorecard.empty else scorecard,
        "Backtest Credibility Scorecard"
    )

    # Signal IC table
    ic_table_html = _df_to_html_table(
        signal_ic.head(15) if not signal_ic.empty else signal_ic,
        "Signal Information Coefficients (IC)"
    )

    # Factor exposure table (today only)
    if not factor_exp.empty and "date" in factor_exp.columns:
        latest_date = factor_exp["date"].max()
        fe_today = factor_exp[factor_exp["date"] == latest_date]
    else:
        fe_today = factor_exp
    fe_html = _df_to_html_table(fe_today, "Current Factor Exposures (Barra)")

    # Data quality
    dq_html = _df_to_html_table(
        dq_report[["check", "status", "detail"]] if not dq_report.empty else dq_report,
        "Data Quality Report"
    )

    # SHAP feature importance
    shap_html = _df_to_html_table(
        shap_df[["feature", "mean_abs_shap", "rank", "keep"]].round(5) if not shap_df.empty else shap_df,
        "LightGBM SHAP Feature Importance (W19)"
    )

    # Halflife table
    hl_html = _df_to_html_table(
        halflife_df[["signal", "halflife_days", "IC0"]].round(2) if not halflife_df.empty else halflife_df,
        "Empirical Signal Half-Lives (W17)"
    )

    # Performance summary cards
    perf_cards = ""
    for metric, value in perf.items():
        color = "#059669" if any(x in metric for x in ["Return", "Sharpe", "Calmar", "Hit"]) else "#dc2626"
        perf_cards += f'<div class="metric-card"><div class="metric-value" style="color:{color}">{value}</div><div class="metric-label">{metric}</div></div>'

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Canyon Quant — Factor Tearsheet</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f3f4f6; color: #1f2937; margin: 0; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #111827; border-bottom: 3px solid #2563eb; padding-bottom: 12px; }}
  h2 {{ color: #374151; margin-top: 32px; }}
  h3 {{ color: #4b5563; }}
  h4 {{ color: #6b7280; }}
  .header {{ background: #1e3a5f; color: white; padding: 24px; border-radius: 12px;
             margin-bottom: 24px; }}
  .header h1 {{ color: white; border-bottom-color: rgba(255,255,255,0.3); }}
  .header p {{ color: #93c5fd; margin: 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                   gap: 16px; margin: 16px 0; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 16px; text-align: center;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .metric-value {{ font-size: 24px; font-weight: 700; }}
  .metric-label {{ font-size: 13px; color: #6b7280; margin-top: 4px; }}
  .section {{ background: white; border-radius: 12px; padding: 24px; margin: 16px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{ background: #f3f4f6; padding: 8px 12px; text-align: left; font-weight: 600; }}
  .data-table td {{ padding: 6px 12px; border-bottom: 1px solid #f3f4f6; }}
  .data-table tr:hover {{ background: #f9fafb; }}
  .score-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 11px; font-weight: 600; }}
  .overall-score {{ font-size: 48px; font-weight: 700; color: #2563eb; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>Canyon Quant — Factor Tearsheet</h1>
  <p>Generated: {today_str} | System: Canyon v9/v11 | Data: SEC EDGAR + FRED + Price Cache</p>
</div>

<div class="section">
  <h2>Executive Summary</h2>
  <div style="display:flex;align-items:center;gap:32px;">
    <div>
      <div class="overall-score">{overall_score:.1f}/10</div>
      <div style="color:#6b7280;font-size:14px;">Backtest Credibility Score</div>
    </div>
    <div class="metrics-grid" style="flex:1">
      {perf_cards if perf_cards else '<p><em>Run v11 backtest to see performance</em></p>'}
    </div>
  </div>
</div>

<div class="section">
  <h2>Performance</h2>
  {cum_chart_html}
  {rolling_sr_html}
</div>

<div class="section">
  {scorecard_html}
</div>

<div class="section">
  {ic_table_html}
</div>

<div class="section">
  {hl_html}
</div>

<div class="section">
  {shap_html}
</div>

<div class="section">
  {fe_html}
</div>

<div class="section">
  {dq_html}
</div>

</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  [Tearsheet] Generated → {output_path}")
    print(f"  Open: file://{output_path}")
    return output_path


if __name__ == "__main__":
    print("W42: Factor Tearsheet HTML Generator")
    generate_tearsheet()
