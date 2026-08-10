#!/usr/bin/env python3
"""
Canyon v9 — Daily Email Summary
Sends a concise morning briefing to configured recipient.

Setup (first time only):
    .venv/bin/python email_summary.py --setup

Manual send:
    .venv/bin/python email_summary.py
"""

from __future__ import annotations

import argparse
import json
import re
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT        = Path(__file__).parent
CONFIG_FILE = Path.home() / ".canyon_quant_config.json"
RECIPIENT   = "lynnnnnnn958@gmail.com"


# ── config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text())


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    CONFIG_FILE.chmod(0o600)  # owner-read-only — keeps credentials safe


def setup_interactive():
    print("\nCanyon v9 — Email Setup")
    print("─" * 50)
    print("You need a Gmail App Password (not your regular Gmail password).")
    print("Steps to get one:")
    print("  1. Go to https://myaccount.google.com/apppasswords")
    print("  2. Select 'Mail' + 'Mac' → Generate")
    print("  3. Copy the 16-character password shown")
    print()
    sender = input("Your Gmail address (the one sending emails): ").strip()
    if not sender:
        sender = RECIPIENT
    app_pw = input("Gmail App Password (16 chars, spaces OK): ").strip().replace(" ", "")
    cfg = load_config()
    cfg["email_sender"]   = sender
    cfg["email_app_pw"]   = app_pw
    cfg["email_recipient"] = RECIPIENT
    save_config(cfg)
    print(f"\n✓  Saved to {CONFIG_FILE}")
    print("   Run a test:  .venv/bin/python email_summary.py")


# ── data loaders ─────────────────────────────────────────────────────────────

def _read_daily_report(date_str: str) -> dict:
    """Parse key fields from the daily markdown report."""
    result = {
        "regime": "—", "macro": "—",
        "longs": [], "shorts": [],
        "new_long": [], "exit_long": [],
        "new_short": [], "exit_short": [],
    }
    p = ROOT / "daily_reports" / f"daily_{date_str}.md"
    if not p.exists():
        # Try most recent
        reports = sorted((ROOT / "daily_reports").glob("daily_*.md"))
        if reports:
            p = reports[-1]
        else:
            return result

    txt = p.read_text(encoding="utf-8")

    m = re.search(r"HMM State:\s+\*\*(.+?)\*\*", txt)
    if m: result["regime"] = m.group(1)

    m = re.search(r"Macro Overlay:\s+\*\*(.+?)\*\*", txt)
    if m: result["macro"] = m.group(1)

    # Long/short books: read from the clean ranked CSVs (daily_picks / daily_shorts),
    # NOT the markdown table (which can carry nan scores + alphabetical/garbage tickers).
    try:
        import pandas as _pd
        dp = _pd.read_csv(ROOT / "daily_picks.csv")
        dp = dp[dp["ticker"].astype(str).str.fullmatch(r"[A-Z][A-Z.\-]{0,6}")]
        for i, (_, row) in enumerate(dp.head(8).iterrows(), start=1):
            z = (float(row.get("alpha_score", 50)) - 50) / 25.0
            result["longs"].append({"rank": i, "ticker": str(row["ticker"]), "score": f"{z:+.2f}"})
    except Exception:
        pass
    try:
        import pandas as _pd
        ds = _pd.read_csv(ROOT / "daily_shorts.csv")
        ds = ds[ds["ticker"].astype(str).str.fullmatch(r"[A-Z][A-Z.\-]{0,6}")]
        for i, (_, row) in enumerate(ds.head(8).iterrows(), start=1):
            z = (float(row.get("alpha_score", 50)) - 50) / 25.0
            result["shorts"].append({"rank": i, "ticker": str(row["ticker"]), "score": f"{z:+.2f}"})
    except Exception:
        pass

    # Signal changes
    m = re.search(r"NEW LONG:\s+(.+)", txt)
    if m: result["new_long"] = [t.strip() for t in m.group(1).split(",") if t.strip()]
    m = re.search(r"EXIT LONG:\s+(.+)", txt)
    if m: result["exit_long"] = [t.strip() for t in m.group(1).split(",") if t.strip()]
    m = re.search(r"NEW SHORT:\s+(.+)", txt)
    if m: result["new_short"] = [t.strip() for t in m.group(1).split(",") if t.strip()]
    m = re.search(r"EXIT SHORT:\s+(.+)", txt)
    if m: result["exit_short"] = [t.strip() for t in m.group(1).split(",") if t.strip()]

    return result


def _read_alerts() -> dict:
    p = ROOT / "daily_alerts.json"
    if not p.exists():
        return {"total_alerts": 0, "critical_count": 0, "warning_count": 0, "alerts": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _read_nav() -> dict:
    import pandas as pd
    result = {"nav": 0.0, "gain": 0.0, "daily_ret": 0.0, "max_dd": 0.0}
    p = ROOT / "paper_nav_curve.csv"
    if not p.exists():
        return result
    df = pd.read_csv(p).dropna(subset=["nav"])
    if df.empty:
        return result
    last = df.iloc[-1]
    result["nav"]       = float(last.get("nav", 0))
    result["gain"]      = float(last.get("cumulative_return_pct", 0)) * 100
    result["daily_ret"] = float(last.get("daily_return", 0)) * 100
    result["max_dd"]    = float(df["drawdown_pct"].min()) * 100
    return result


def _read_log_summary() -> str:
    """Find the latest run log and extract the summary section."""
    logs = sorted((ROOT / "daily_runs").glob("*_daily_run.log"))
    if not logs:
        return ""
    lines = logs[-1].read_text(encoding="utf-8").splitlines()
    # Find lines after SUMMARY header
    idx = next((i for i, l in enumerate(lines) if "SUMMARY" in l), None)
    if idx is None:
        return ""
    return "\n".join(lines[idx:idx+15])


# ── HTML email builder ────────────────────────────────────────────────────────

_COLOR = {
    "bull":    "#1B6F4A",
    "bear":    "#B83232",
    "neutral": "#B8943F",
    "navy":    "#1B2A4A",
    "gold":    "#B8943F",
    "cream":   "#FAFAF8",
    "border":  "#E2E0DC",
    "grey":    "#666666",
}


def _regime_color(regime: str) -> str:
    r = regime.upper()
    return _COLOR["bull"] if "BULL" in r else (_COLOR["bear"] if "BEAR" in r else _COLOR["neutral"])


def build_html_email(date_str: str, report: dict, alerts: dict, nav: dict) -> str:
    regime      = report["regime"]
    macro       = report["macro"]
    reg_color   = _regime_color(regime)
    mac_color   = _COLOR["bull"] if "ON" in macro.upper() else (_COLOR["bear"] if "OFF" in macro.upper() else _COLOR["neutral"])

    nav_color   = _COLOR["bull"] if nav["gain"] >= 0 else _COLOR["bear"]
    day_color   = _COLOR["bull"] if nav["daily_ret"] >= 0 else _COLOR["bear"]

    crit  = alerts.get("critical_count", 0)
    warns = alerts.get("warning_count", 0)
    alert_color = _COLOR["bear"] if crit > 0 else (_COLOR["neutral"] if warns > 0 else _COLOR["bull"])
    alert_txt   = f"{crit} critical, {warns} warnings" if (crit + warns) > 0 else "All clear"

    # Longs table rows
    long_rows = ""
    for r in report["longs"][:5]:
        try:
            sc = float(r["score"])
        except (ValueError, TypeError):
            sc = 0.0
        sc_color = _COLOR["bull"] if sc > 0 else _COLOR["bear"]
        long_rows += f"""
        <tr>
          <td style="padding:7px 10px;font-weight:700;color:{_COLOR['navy']};font-size:14px">{r['ticker']}</td>
          <td style="padding:7px 10px;color:{sc_color};font-weight:700;text-align:right">{r['score']}</td>
        </tr>"""

    # Signal changes
    def change_row(label: str, tickers: list, color: str) -> str:
        if not tickers:
            return ""
        return f'<tr><td style="padding:4px 10px;color:#999;font-size:12px;white-space:nowrap">{label}</td><td style="padding:4px 10px;font-weight:700;color:{color};font-size:13px">{", ".join(tickers)}</td></tr>'

    changes = (
        change_row("New long ▲",   report["new_long"],   _COLOR["bull"])
      + change_row("Exit long ▼",  report["exit_long"],  _COLOR["bear"])
      + change_row("New short ▼",  report["new_short"],  _COLOR["bear"])
      + change_row("Exit short ▲", report["exit_short"], _COLOR["bull"])
    )
    changes_block = f"""
    <table style="width:100%;border-collapse:collapse;margin-top:8px">
      {changes}
    </table>""" if changes else '<p style="color:#AAA;font-size:13px">No signal changes today.</p>'

    # Top alerts (up to 3)
    top_alerts = ""
    for a in (alerts.get("alerts") or [])[:3]:
        pri = str(a.get("priority","")).upper()
        ac  = _COLOR["bear"] if pri == "CRITICAL" else _COLOR["neutral"]
        top_alerts += f"""
        <tr>
          <td style="padding:6px 10px;font-size:12.5px;color:{ac};font-weight:700;white-space:nowrap">{pri}</td>
          <td style="padding:6px 10px;font-size:12.5px;color:#333">{a.get('title','')}</td>
        </tr>"""

    alerts_block = f"""
    <table style="width:100%;border-collapse:collapse;margin-top:8px">
      {top_alerts}
    </table>""" if top_alerts else '<p style="color:#1B6F4A;font-size:13px">No alerts today.</p>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#F2F1EE;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#F2F1EE;padding:28px 0">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#FAFAF8;border:1px solid #E2E0DC;border-radius:4px;overflow:hidden">

  <!-- Header -->
  <tr>
    <td style="background:#1B2A4A;padding:22px 28px">
      <p style="margin:0;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.5)">Canyon Quant v9 · {date_str}</p>
      <p style="margin:6px 0 0;font-size:22px;font-weight:700;color:#fff">Daily Signal Report</p>
    </td>
  </tr>

  <!-- KPI row -->
  <tr>
    <td style="padding:20px 28px 0">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td width="33%" style="text-align:center;padding:14px;background:#fff;border:1px solid #E2E0DC;border-radius:3px">
            <p style="margin:0;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600">Regime</p>
            <p style="margin:4px 0 0;font-size:22px;font-weight:700;color:{reg_color}">{regime}</p>
          </td>
          <td width="4px"></td>
          <td width="33%" style="text-align:center;padding:14px;background:#fff;border:1px solid #E2E0DC;border-radius:3px">
            <p style="margin:0;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600">Paper NAV</p>
            <p style="margin:4px 0 0;font-size:20px;font-weight:700;color:{nav_color}">{nav['gain']:+.2f}%</p>
            <p style="margin:2px 0 0;font-size:11px;color:{day_color}">Today: {nav['daily_ret']:+.2f}%</p>
          </td>
          <td width="4px"></td>
          <td width="33%" style="text-align:center;padding:14px;background:#fff;border:1px solid #E2E0DC;border-radius:3px">
            <p style="margin:0;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600">Alerts</p>
            <p style="margin:4px 0 0;font-size:20px;font-weight:700;color:{alert_color}">{alerts.get('total_alerts',0)}</p>
            <p style="margin:2px 0 0;font-size:11px;color:#999">{alert_txt}</p>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Top longs -->
  <tr>
    <td style="padding:24px 28px 0">
      <p style="margin:0 0 2px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600">Top 5 long signals today</p>
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#fff;border:1px solid #E2E0DC;border-radius:3px">
        <tr style="background:#F7F6F3">
          <th style="padding:8px 10px;text-align:left;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:600">Ticker</th>
          <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:600">Score</th>
        </tr>
        {long_rows}
      </table>
    </td>
  </tr>

  <!-- Signal changes -->
  <tr>
    <td style="padding:20px 28px 0">
      <p style="margin:0 0 2px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600">Signal changes — action required</p>
      <div style="background:#fff;border:1px solid #E2E0DC;border-radius:3px;padding:4px 0">
        {changes_block}
      </div>
    </td>
  </tr>

  <!-- Alerts -->
  <tr>
    <td style="padding:20px 28px 0">
      <p style="margin:0 0 2px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:600">Desk alerts</p>
      <div style="background:#fff;border:1px solid #E2E0DC;border-radius:3px;padding:4px 0">
        {alerts_block}
      </div>
    </td>
  </tr>

  <!-- Macro bar -->
  <tr>
    <td style="padding:20px 28px 0">
      <div style="background:#fff;border:1px solid #E2E0DC;border-radius:3px;padding:14px 16px;display:flex">
        <span style="font-size:12px;color:#999">Macro overlay: </span>
        <strong style="font-size:12px;color:{mac_color};margin-left:6px">{macro}</strong>
        <span style="font-size:12px;color:#BBB;margin-left:16px"> · Max drawdown: </span>
        <strong style="font-size:12px;color:#B83232;margin-left:6px">{nav['max_dd']:.2f}%</strong>
      </div>
    </td>
  </tr>

  <!-- Footer -->
  <tr>
    <td style="padding:24px 28px;border-top:1px solid #E2E0DC;margin-top:24px">
      <p style="margin:0;font-size:11px;color:#BBB;line-height:1.6">
        Canyon v9 · Research only · No broker connection · No live orders<br>
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · Signals are paper-only
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def build_text_email(date_str: str, report: dict, alerts: dict, nav: dict) -> str:
    """Plain-text fallback."""
    longs_str  = "  ".join(f"{r['ticker']}({r['score']})" for r in report["longs"][:5])
    changes = []
    if report["new_long"]:   changes.append(f"New long:  {', '.join(report['new_long'])}")
    if report["exit_long"]:  changes.append(f"Exit long: {', '.join(report['exit_long'])}")
    if report["new_short"]:  changes.append(f"New short: {', '.join(report['new_short'])}")
    if report["exit_short"]: changes.append(f"Exit short:{', '.join(report['exit_short'])}")
    changes_str = "\n".join(changes) or "No changes today."
    alert_str = f"{alerts.get('total_alerts',0)} alerts ({alerts.get('critical_count',0)} critical, {alerts.get('warning_count',0)} warnings)"
    return f"""Canyon v9 Daily Report — {date_str}
{"=" * 50}
Regime:    {report['regime']}
Macro:     {report['macro']}
Paper NAV: {nav['gain']:+.2f}%  (today: {nav['daily_ret']:+.2f}%)
Max DD:    {nav['max_dd']:.2f}%
Alerts:    {alert_str}

Top 5 longs:
  {longs_str}

Signal changes:
{changes_str}

Research: file://{ROOT / 'canyon_v9_research.html'}
Server:   http://localhost:8513
---
Research only · No broker connection · No live orders
"""


# ── send ──────────────────────────────────────────────────────────────────────

def send(date_str: str | None = None, verbose: bool = True) -> bool:
    """Build and send the daily email. Returns True on success."""
    cfg = load_config()
    sender = cfg.get("email_sender", "")
    app_pw = cfg.get("email_app_pw", "")
    recipient = cfg.get("email_recipient", RECIPIENT)

    if not sender or not app_pw:
        if verbose:
            print("  Email not configured — run:  .venv/bin/python email_summary.py --setup")
        return False

    today = date_str or datetime.now().strftime("%Y-%m-%d")
    report = _read_daily_report(today)
    alerts = _read_alerts()
    nav    = _read_nav()

    regime  = report["regime"]
    n_alert = alerts.get("total_alerts", 0)
    top1    = report["longs"][0]["ticker"] if report["longs"] else "—"
    subject = f"Canyon v9 — {today} | {regime} | #{1} {top1} | {n_alert} alerts"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient

    text_part = MIMEText(build_text_email(today, report, alerts, nav), "plain", "utf-8")
    html_part = MIMEText(build_html_email(today, report, alerts, nav), "html",  "utf-8")
    msg.attach(text_part)
    msg.attach(html_part)

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(sender, app_pw)
            server.sendmail(sender, recipient, msg.as_string())
        if verbose:
            print(f"  Email sent → {recipient}")
            print(f"  Subject: {subject}")
        return True
    except smtplib.SMTPAuthenticationError:
        if verbose:
            print("  Email auth failed — check your App Password")
            print("  Re-run setup:  .venv/bin/python email_summary.py --setup")
        return False
    except Exception as e:
        if verbose:
            print(f"  Email error: {e}")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 email summary")
    parser.add_argument("--setup", action="store_true", help="Configure Gmail credentials")
    parser.add_argument("--date",  type=str, default=None, help="Date override YYYY-MM-DD")
    args = parser.parse_args()

    if args.setup:
        setup_interactive()
    else:
        ok = send(date_str=args.date)
        import sys; sys.exit(0 if ok else 1)
