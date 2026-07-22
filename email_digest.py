#!/usr/bin/env python3
"""
Canyon Quant — Daily Email Digest
Sends a plain-English morning summary to your email.

Setup (one time):
  1. Go to myaccount.google.com → Security → App passwords
  2. Create an app password for "Canyon Quant"
  3. Run:  python email_digest.py --setup
  4. Done. The digest sends itself every weekday at 8 AM via cron.
"""
import smtplib, json, csv, sys, argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text       import MIMEText
from pathlib import Path
from datetime import datetime

ROOT    = Path(__file__).parent
CONFIG  = ROOT / ".email_config.json"
TO_ADDR = "lynnnnnnn958@gmail.com"


# ── Config helpers ────────────────────────────────────────────────────────────

def setup():
    print("\nCanyon Quant — Email Digest Setup")
    print("─" * 40)
    print("You need a Gmail App Password (NOT your normal Gmail password).")
    print("Steps:")
    print("  1. Go to: https://myaccount.google.com/apppasswords")
    print("  2. Select app: Mail  |  device: Mac")
    print("  3. Copy the 16-character password shown")
    print()
    sender = input(f"Your Gmail address (press Enter for {TO_ADDR}): ").strip() or TO_ADDR
    pw     = input("Paste your App Password here: ").strip().replace(" ", "")
    if len(pw) not in (16, 19):
        print(f"  ⚠ Password looks unusual ({len(pw)} chars) — double-check it")
    cfg = {"sender": sender, "password": pw, "recipient": TO_ADDR}
    CONFIG.write_text(json.dumps(cfg, indent=2))
    CONFIG.chmod(0o600)
    print(f"\n  Saved to {CONFIG}")
    print("  Testing connection …")
    try:
        _send_test(cfg)
        print("  ✓ Test email sent! Check your inbox.")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Check your App Password and try again.")


def _send_test(cfg):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Canyon Quant — email digest is set up ✓"
    msg["From"]    = cfg["sender"]
    msg["To"]      = cfg["recipient"]
    msg.attach(MIMEText("Email digest is configured. You'll receive your first summary tomorrow at 8 AM.", "plain"))
    _smtp_send(cfg, msg)


def _smtp_send(cfg, msg):
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
        s.starttls()
        s.login(cfg["sender"], cfg["password"])
        s.sendmail(cfg["sender"], cfg["recipient"], msg.as_string())


# ── Data loaders ──────────────────────────────────────────────────────────────

def _read_csv(path, max_rows=200):
    p = ROOT / path
    if not p.exists():
        return []
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= max_rows:
                break
            rows.append(row)
    return rows


def _load_regime():
    rows = _read_csv("paper_trading_log.csv", max_rows=5)
    if rows:
        r = rows[-1]
        return r.get("hmm_regime", "—"), float(r.get("hmm_exposure", 0) or 0)
    return "—", 0.0


def _load_top_picks(n=5):
    rows = _read_csv("alpha_scores.csv", max_rows=100)
    longs  = [r for r in rows if r.get("signal","").upper() == "LONG"][:n]
    shorts = [r for r in rows if r.get("signal","").upper() == "SHORT"][:n]
    return longs, shorts


def _load_alerts():
    # Primary: step98 daily_alerts.json output
    p = ROOT / "daily_alerts.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            alerts_raw = data.get("alerts", [])
            # Normalise to {severity, message, ticker} shape
            out = []
            for a in alerts_raw[:5]:
                out.append({
                    "severity": a.get("level", a.get("severity", "INFO")),
                    "message":  a.get("message", a.get("msg", str(a))),
                    "ticker":   a.get("ticker", ""),
                })
            return out
        except Exception:
            pass
    # Fallback: desk_monitor_events.csv (legacy)
    rows = _read_csv("desk_monitor_events.csv", max_rows=50)
    today_str = datetime.now().strftime("%Y-%m-%d")
    recent = [r for r in rows if r.get("date","") >= today_str[:7]]
    critical = [r for r in recent if r.get("severity","").upper() in ("CRITICAL","HIGH")]
    return critical[:5]


def _load_paper_nav():
    rows = _read_csv("paper_trading_log.csv", max_rows=5)
    if not rows:
        return None, None, None
    r = rows[-1]
    pnl_vals = [float(r.get(k, 0) or 0) for k in r if k.startswith("price_")]
    start_row = _read_csv("paper_trading_log.csv", max_rows=1)
    try:
        start_date = rows[0].get("date","")[:10]
        end_date   = rows[-1].get("date","")[:10]
    except Exception:
        start_date = end_date = "—"
    return start_date, end_date, None


def _load_paper_summary():
    """Load paper NAV from the generated daily data if available."""
    p = ROOT / "alpha_scores_v26.csv"
    if not p.exists():
        p = ROOT / "alpha_scores.csv"
    return None


# ── Email builder ─────────────────────────────────────────────────────────────

_REGIME_EMOJI  = {"Bull": "🟢", "Bear": "🔴", "BULL": "🟢", "BEAR": "🔴"}
_REGIME_PLAIN  = {"Bull": "Bull — market is rising, full strategy active",
                  "Bear": "Bear — market is falling, strategy reduces size",
                  "BULL": "Bull — market is rising, full strategy active",
                  "BEAR": "Bear — market is falling, strategy reduces size"}
_SEV_EMOJI     = {"CRITICAL": "🚨", "HIGH": "⚠️", "WARNING": "⚠️"}

_GATE_PLAIN = {
    "master":       "overall position is too large",
    "single":       "exceeds max per-stock limit",
    "earnings_gap": "earnings report coming up",
    "kelly":        "too large for the signal strength",
    "sector":       "sector already at limit",
    "liquidity":    "stock not liquid enough",
    "crisis":       "crash risk too high",
}

_MONITOR_PLAIN = {
    "PRICE_BREAK":       "Price alert",
    "RISK_LIMIT_BREACH": "Risk limit",
    "NEWS_SHOCK":        "News alert",
    "SQUEEZE_WATCH":     "Squeeze setup",
    "OPTIONS_ALERT":     "Options signal",
}


def _clean_detail(detail: str, monitor: str) -> str:
    import re
    if monitor == "PRICE_BREAK":
        detail = re.sub(r"Latest close\s+([\d.]+) is (below|above) the prior 20-?day (low|high)\s+([\d.]+)\.",
                        lambda m: f"Price {'dropped to' if m.group(2)=='below' else 'rose to'} ${float(m.group(1)):,.2f} "
                                  f"({'below' if m.group(2)=='below' else 'above'} the 4-week {'low' if m.group(3)=='low' else 'high'} of ${float(m.group(4)):,.2f}).",
                        detail)
    elif monitor == "RISK_LIMIT_BREACH":
        gates = {}
        for part in detail.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                gates[k.strip()] = v.strip()
        failed = [_GATE_PLAIN.get(k, k) for k, v in gates.items()
                  if v.upper() not in ("CLEAR", "OK", "PASS")]
        if failed:
            return "Issues: " + "; ".join(failed) + "."
    return detail


def build_email_html(date_str: str) -> tuple[str, str]:
    """Return (subject, html_body)."""
    regime, exposure = _load_regime()
    longs, shorts    = _load_top_picks(5)
    alerts           = _load_alerts()

    reg_emoji = _REGIME_EMOJI.get(regime, "⚪")
    reg_plain = _REGIME_PLAIN.get(regime, regime)
    exp_pct   = f"{exposure*100:.0f}%" if exposure else "—"

    subject = f"Canyon Daily — {date_str} — {reg_emoji} {regime} Mode"

    # ── HTML email ────────────────────────────────────────────────────────────
    def row(label, val, color="#333"):
        return f'<tr><td style="padding:4px 12px 4px 0;color:#888;font-size:13px;white-space:nowrap">{label}</td><td style="padding:4px 0;color:{color};font-size:13px;font-weight:600">{val}</td></tr>'

    def pick_rows(picks, signal_color):
        if not picks:
            return '<tr><td colspan="3" style="color:#AAA;font-size:12px;padding:4px">No picks today</td></tr>'
        out = []
        for p in picks:
            score = float(p.get("alpha_score", 0) or 0)
            sector = p.get("sector", "")
            out.append(
                f'<tr>'
                f'<td style="padding:5px 14px 5px 0;font-weight:700;color:#1B2A4A;font-size:13px">{p["ticker"]}</td>'
                f'<td style="padding:5px 14px 5px 0;color:#666;font-size:12px">{sector}</td>'
                f'<td style="padding:5px 0;color:{signal_color};font-size:13px;font-weight:700">{score:.0f}/100</td>'
                f'</tr>'
            )
        return "".join(out)

    alert_html = ""
    if alerts:
        parts = []
        for a in alerts:
            sev     = a.get("severity", "").upper()
            mon     = a.get("monitor",  "").upper()
            ticker  = a.get("ticker",   "")
            detail  = _clean_detail(a.get("detail", ""), mon)
            action  = a.get("action",  "")
            emoji   = _SEV_EMOJI.get(sev, "•")
            m_plain = _MONITOR_PLAIN.get(mon, mon.replace("_", " ").title())
            color   = "#B83232" if sev == "CRITICAL" else "#B8943F"
            parts.append(
                f'<div style="border-left:4px solid {color};padding:10px 14px;margin-bottom:8px;background:#FAFAFA">'
                f'<p style="margin:0 0 2px;font-size:12px;color:{color};font-weight:700;text-transform:uppercase">'
                f'{emoji} {sev.title()} — {m_plain} — {ticker}</p>'
                f'<p style="margin:0 0 2px;font-size:13px;color:#333">{detail}</p>'
                f'<p style="margin:0;font-size:12px;color:#888">→ {action}</p>'
                f'</div>'
            )
        alert_html = f"""
        <h3 style="font-size:14px;color:#1B2A4A;font-weight:700;margin:28px 0 10px;border-bottom:1px solid #E2E0DC;padding-bottom:6px">
          Alerts today
        </h3>
        {"".join(parts)}"""
    else:
        alert_html = '<p style="color:#1B6F4A;font-size:13px;margin-top:28px">✓ No active alerts today.</p>'

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Georgia,serif;max-width:600px;margin:0 auto;padding:24px;color:#1A1A1A;background:#fff">

  <!-- Header -->
  <div style="border-bottom:3px solid #1B2A4A;padding-bottom:14px;margin-bottom:24px">
    <p style="margin:0;font-size:11px;color:#999;letter-spacing:2px;text-transform:uppercase">Canyon Quant · Daily Digest</p>
    <h1 style="margin:6px 0 0;font-size:22px;color:#1B2A4A">{date_str}</h1>
  </div>

  <!-- Market Mode -->
  <h3 style="font-size:14px;color:#1B2A4A;font-weight:700;margin:0 0 10px;border-bottom:1px solid #E2E0DC;padding-bottom:6px">
    Today's market mode
  </h3>
  <table style="border-collapse:collapse">
    {row("Mode", f"{reg_emoji}  {reg_plain}")}
    {row("Strategy exposure", exp_pct)}
  </table>

  <!-- Top Picks -->
  <h3 style="font-size:14px;color:#1B2A4A;font-weight:700;margin:28px 0 10px;border-bottom:1px solid #E2E0DC;padding-bottom:6px">
    Top buy candidates today
  </h3>
  <table style="border-collapse:collapse;width:100%">
    <thead>
      <tr>
        <th style="text-align:left;font-size:11px;color:#AAA;font-weight:600;padding-bottom:4px">Ticker</th>
        <th style="text-align:left;font-size:11px;color:#AAA;font-weight:600;padding-bottom:4px">Sector</th>
        <th style="text-align:left;font-size:11px;color:#AAA;font-weight:600;padding-bottom:4px">Model score</th>
      </tr>
    </thead>
    <tbody>
      {pick_rows(longs, "#1B6F4A")}
    </tbody>
  </table>

  <h3 style="font-size:14px;color:#1B2A4A;font-weight:700;margin:28px 0 10px;border-bottom:1px solid #E2E0DC;padding-bottom:6px">
    Top stocks to avoid / watch short
  </h3>
  <table style="border-collapse:collapse;width:100%">
    <thead>
      <tr>
        <th style="text-align:left;font-size:11px;color:#AAA;font-weight:600;padding-bottom:4px">Ticker</th>
        <th style="text-align:left;font-size:11px;color:#AAA;font-weight:600;padding-bottom:4px">Sector</th>
        <th style="text-align:left;font-size:11px;color:#AAA;font-weight:600;padding-bottom:4px">Model score</th>
      </tr>
    </thead>
    <tbody>
      {pick_rows(shorts, "#B83232")}
    </tbody>
  </table>

  {alert_html}

  <!-- Footer -->
  <div style="margin-top:36px;padding-top:14px;border-top:1px solid #E2E0DC">
    <a href="http://localhost:8888" style="display:inline-block;background:#1B2A4A;color:#fff;text-decoration:none;padding:10px 24px;font-size:13px;font-weight:600;border-radius:3px">
      Open full dashboard →
    </a>
    <p style="font-size:11px;color:#BBB;margin-top:12px">
      Canyon Quant · research only · no live trading
    </p>
  </div>

</body>
</html>"""

    return subject, html


# ── Send ──────────────────────────────────────────────────────────────────────

def send_digest():
    if not CONFIG.exists():
        print("Email not configured. Run:  python email_digest.py --setup")
        return

    cfg = json.loads(CONFIG.read_text())
    date_str = datetime.now().strftime("%B %d, %Y")
    subject, html_body = build_email_html(date_str)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = cfg["sender"]
    msg["To"]      = cfg["recipient"]
    msg.attach(MIMEText(html_body, "html"))

    try:
        _smtp_send(cfg, msg)
        print(f"  ✓ Digest sent to {cfg['recipient']}")
    except Exception as e:
        print(f"  ✗ Failed to send: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Configure email credentials")
    parser.add_argument("--test",  action="store_true", help="Send a test digest right now")
    args = parser.parse_args()

    if args.setup:
        setup()
    else:
        send_digest()
