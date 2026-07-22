#!/usr/bin/env python3
"""
Canyon Push Alerts — ntfy.sh
==============================
Sends free push notifications to your phone when key events occur:
  • HMM regime flips (BULL → BEAR or BEAR → BULL)
  • High-conviction short candidates (score ≥ 70, urgency NOW)
  • Top long signals today (alpha score ≥ 80th percentile)
  • Pipeline freshness failures (key files stale)
  • Bear probability spike (≥ 60%)

Setup (one-time, 2 minutes):
  1. Install the free "ntfy" app on your iPhone/Android
  2. In the app, subscribe to topic: canyon-quant-YOUR_NAME
     (e.g. canyon-quant-lynn)
  3. Edit NTFY_TOPIC below to match that topic
  4. This script runs automatically as part of run_daily.py (Step 374)

No account required. ntfy.sh is open-source and free.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parent

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Change this to your personal topic name (no spaces, keep it unique)
NTFY_TOPIC = os.environ.get("CANYON_NTFY_TOPIC", "canyon-quant-lynn")
NTFY_URL   = f"https://ntfy.sh/{NTFY_TOPIC}"

# Thresholds
SHORT_URGENT_SCORE  = 70    # minimum score to include in "NOW" short alert
BEAR_PROB_THRESHOLD = 0.60  # bear probability above this → alert
ALPHA_TOP_N         = 5     # how many top longs to include in daily summary

# State file — prevents duplicate alerts on the same day
STATE_FILE = ROOT / ".push_alert_state.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _send(title: str, body: str, priority: str = "default",
          tags: list[str] | None = None, click_url: str = "",
          actions: str = "") -> bool:
    """POST rich notification to ntfy.sh. Returns True on success."""
    # Detect local server URL for action buttons
    local_url = os.environ.get("CANYON_DASHBOARD_URL", "http://localhost:8513")

    headers = {
        "Title":    title.encode(),
        "Priority": priority.encode(),
    }
    if tags:
        headers["Tags"] = ",".join(tags).encode()
    if click_url:
        headers["Click"] = click_url.encode()
    elif local_url:
        headers["Click"] = local_url.encode()

    # Default action buttons: open dashboard + open relevant tab
    if actions:
        headers["Actions"] = actions.encode()
    elif local_url:
        headers["Actions"] = (
            f"view, Open Dashboard, {local_url};"
            f"view, Today Tab, {local_url}/#today"
        ).encode()

    data = body.encode()
    req  = urllib.request.Request(NTFY_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            ok = r.status == 200
            if ok:
                print(f"  ✓ Push sent: {title}")
            return ok
    except urllib.error.URLError as e:
        print(f"  ✗ Push failed ({e}): {title}")
        return False
    except Exception as e:
        print(f"  ✗ Push error ({e}): {title}")
        return False


# ── ALERT CHECKS ──────────────────────────────────────────────────────────────

def check_regime_flip(state: dict) -> list[dict]:
    """Alert if HMM regime flipped since yesterday."""
    alerts = []
    hmm_path = ROOT / "hmm_regime_daily.csv"
    if not hmm_path.exists():
        return alerts
    try:
        import pandas as pd
        df = pd.read_csv(hmm_path)
        if "regime" not in df.columns or len(df) < 2:
            return alerts
        # Use last 2 rows
        latest = str(df["regime"].iloc[-1]).upper()
        prev   = str(df["regime"].iloc[-2]).upper()
        date_col = "date" if "date" in df.columns else df.columns[0]
        latest_date = str(df[date_col].iloc[-1])

        prev_regime = state.get("last_regime", "")
        if latest != prev_regime and prev_regime:
            emoji = "🐻" if "BEAR" in latest else "🐂"
            is_bear = "BEAR" in latest
            local_url = os.environ.get("CANYON_DASHBOARD_URL", "http://localhost:8513")
            alerts.append({
                "title":    f"{emoji} Regime Flip: {prev_regime} → {latest}",
                "body":     (f"HMM regime changed as of {latest_date}.\n"
                             f"Previous: {prev_regime}  →  New: {latest}\n\n"
                             f"{'⚠️ Action: Reduce gross exposure to 50%' if is_bear else '✅ Action: Full exposure allowed'}\n"
                             f"Check Macro tab for bear probability details."),
                "priority": "urgent" if is_bear else "high",
                "tags":     ["rotating_light" if is_bear else "white_check_mark"],
                "actions":  (f"view, Open Dashboard, {local_url};"
                             f"view, Macro Tab, {local_url}/#macro"),
            })
        state["last_regime"] = latest
    except Exception as e:
        print(f"  regime check error: {e}")
    return alerts


def check_bear_probability(state: dict) -> list[dict]:
    """Alert if macro bear probability spikes."""
    alerts = []
    outlook_path = ROOT / "macro_regime_outlook.json"
    if not outlook_path.exists():
        return alerts
    try:
        data  = json.loads(outlook_path.read_text())
        prob  = float(data.get("bear_prob_4w", 0))
        today = date.today().isoformat()

        last_alert_date = state.get("bear_prob_alert_date", "")
        if prob >= BEAR_PROB_THRESHOLD and last_alert_date != today:
            alerts.append({
                "title":    f"⚠️ Bear Probability: {prob:.0%}",
                "body":     f"4-week bear probability hit {prob:.0%} (threshold: {BEAR_PROB_THRESHOLD:.0%}).\n"
                            f"Composite signal: {data.get('composite_signal', 'N/A')}\n"
                            f"Leading indicators: {data.get('n_bearish', '?')}/5 bearish\n\n"
                            f"Check Macro tab for full details.",
                "priority": "high",
                "tags":     ["warning"],
            })
            state["bear_prob_alert_date"] = today
    except Exception as e:
        print(f"  bear prob check error: {e}")
    return alerts


def check_short_urgents(state: dict) -> list[dict]:
    """Alert if any NOW-urgency shorts scored ≥ threshold today."""
    alerts = []
    path = ROOT / "short_scanner.csv"
    if not path.exists():
        return alerts
    try:
        import pandas as pd
        df = pd.read_csv(path)
        today = date.today().isoformat()

        if "as_of" in df.columns:
            df = df[df["as_of"] == today]
        if df.empty:
            return alerts

        urgent = df[(df["score"] >= SHORT_URGENT_SCORE) & (df["urgency"].str.contains("NOW", na=False))]
        if urgent.empty:
            return alerts

        last_alert_date = state.get("short_alert_date", "")
        if last_alert_date == today:
            return alerts

        lines = []
        for _, r in urgent.head(5).iterrows():
            lines.append(
                f"• {r['ticker']}: score={r['score']:.0f} RSI={r['rsi']:.0f} "
                f"entry ${r['entry_low']:.2f}–${r['entry_high']:.2f} "
                f"stop ${r['stop_loss']:.2f} R/R {r['rr_1']}x"
            )

        alerts.append({
            "title":    f"📉 {len(urgent)} High-Score Short(s) — Act NOW",
            "body":     "\n".join(lines) + f"\n\nCheck Short Scanner tab for full details.",
            "priority": "high",
            "tags":     ["chart_with_downwards_trend"],
        })
        state["short_alert_date"] = today
    except Exception as e:
        print(f"  short check error: {e}")
    return alerts


def check_top_longs(state: dict) -> list[dict]:
    """Daily morning summary of top long signals."""
    alerts = []
    path = ROOT / "alpha_scores.csv"
    if not path.exists():
        return alerts
    try:
        import pandas as pd
        today = date.today().isoformat()

        last_alert_date = state.get("long_alert_date", "")
        if last_alert_date == today:
            return alerts

        df = pd.read_csv(path)
        if "alpha_score" not in df.columns and "score" not in df.columns:
            return alerts

        score_col = "alpha_score" if "alpha_score" in df.columns else "score"
        df = df.sort_values(score_col, ascending=False).head(ALPHA_TOP_N)

        lines = []
        for _, r in df.iterrows():
            ticker = r.get("ticker", "?")
            score  = r.get(score_col, 0)
            sector = r.get("sector", "")
            lines.append(f"• {ticker} ({sector}): score={score:.1f}")

        # Also get regime
        regime_str = ""
        hmm_path = ROOT / "hmm_regime_daily.csv"
        if hmm_path.exists():
            hmm = pd.read_csv(hmm_path)
            if "regime" in hmm.columns:
                regime_str = f"\nRegime: {hmm['regime'].iloc[-1]}"

        # Enrich with current SPY price if available
        spy_str = ""
        try:
            import yfinance as yf
            spy_info = yf.Ticker("SPY").info or {}
            spy_px = spy_info.get("currentPrice") or spy_info.get("regularMarketPrice")
            spy_prev = spy_info.get("previousClose")
            if spy_px and spy_prev:
                chg = (spy_px - spy_prev) / spy_prev * 100
                spy_str = f"\nSPY ${spy_px:.2f} ({chg:+.1f}%)"
        except Exception:
            pass

        local_url = os.environ.get("CANYON_DASHBOARD_URL", "http://localhost:8513")
        alerts.append({
            "title":    f"📊 Canyon Daily — Top {ALPHA_TOP_N} Signals",
            "body":     "\n".join(lines) + regime_str + spy_str + "\n\nTap to open Canyon.",
            "priority": "default",
            "tags":     ["bar_chart"],
            "actions":  (f"view, Today, {local_url}/#today;"
                         f"view, Longs, {local_url}/#longs"),
        })
        state["long_alert_date"] = today
    except Exception as e:
        print(f"  long check error: {e}")
    return alerts


def check_book_drawdown(state: dict) -> list[dict]:
    """Alert if any book's NAV drops >5% from its peak (read from alpaca_book_state.json)."""
    alerts = []
    book_state_path = ROOT / "alpaca_book_state.json"
    pnl_path        = ROOT / "alpaca_pnl_summary.json"
    if not book_state_path.exists():
        return alerts
    try:
        import pandas as pd
        book_state = json.loads(book_state_path.read_text())
        today_str  = date.today().isoformat()

        # Also try to pull latest equity from P&L summary
        pnl_summary = {}
        if pnl_path.exists():
            try:
                pnl_summary = json.loads(pnl_path.read_text())
            except Exception:
                pass

        new_alerts: list[dict] = []
        for book, data in book_state.items():
            nav_history = data.get("nav_history", {})   # {date: nav} accumulated over time
            effective_cap = data.get("effective_capital", 0.0)
            if effective_cap <= 0:
                continue

            # Update NAV history with today's effective capital
            if nav_history:
                nav_history[today_str] = effective_cap
            else:
                nav_history = {today_str: effective_cap}

            # Also incorporate any total_pnl$ adjustment from pnl_summary
            book_pnl = (pnl_summary.get(book, {}) or {}).get("total_pnl_$", 0.0)
            current_nav = effective_cap + book_pnl

            # Compute peak NAV
            all_navs = list(nav_history.values())
            if not all_navs:
                continue
            peak_nav = max(all_navs)
            if peak_nav <= 0:
                continue

            drawdown = (current_nav - peak_nav) / peak_nav  # negative = drawdown

            # Write back updated nav_history to state (best-effort)
            try:
                book_state[book]["nav_history"] = nav_history
                book_state_path.write_text(json.dumps(book_state, indent=2, default=str))
            except Exception:
                pass

            dd_threshold = -0.05
            dd_key = f"dd_alert_{book}"
            last_dd_alert = state.get(dd_key, "")

            if drawdown <= dd_threshold and last_dd_alert != today_str:
                emoji = "🔴" if drawdown <= -0.10 else "🟡"
                priority = "urgent" if drawdown <= -0.10 else "high"
                local_url = os.environ.get("CANYON_DASHBOARD_URL", "http://localhost:8513")
                new_alerts.append({
                    "title":    f"{emoji} Drawdown Alert: {book} Book {drawdown*100:.1f}%",
                    "body":     (f"Canyon {book} book has fallen {abs(drawdown)*100:.1f}% "
                                 f"from peak.\n\n"
                                 f"Peak NAV: ${peak_nav:,.0f}\n"
                                 f"Current:  ${current_nav:,.0f}\n"
                                 f"Drawdown: {drawdown*100:.1f}%\n\n"
                                 f"{'⚠️ Consider reducing exposure.' if drawdown <= -0.10 else 'Monitor closely.'}\n"
                                 f"Check Performance tab for details."),
                    "priority": priority,
                    "tags":     ["rotating_light" if drawdown <= -0.10 else "warning"],
                    "actions":  (f"view, Dashboard, {local_url};"
                                 f"view, Performance, {local_url}/#perf"),
                })
                state[dd_key] = today_str
                print(f"  {emoji} {book} drawdown: {drawdown*100:.1f}% (peak ${peak_nav:,.0f} → ${current_nav:,.0f})")
            else:
                print(f"  ✓ {book} drawdown: {drawdown*100:.1f}% (OK)")

        alerts.extend(new_alerts)
    except Exception as e:
        print(f"  drawdown check error: {e}")
    return alerts


def check_pipeline_freshness(state: dict) -> list[dict]:
    """Alert if key output files are stale (not updated today)."""
    alerts = []
    today = date.today()
    stale = []

    required = [
        ("alpha_scores.csv",          "Price signals"),
        ("hmm_regime_daily.csv",      "HMM Regime"),
        ("macro_regime_outlook.json", "Macro Outlook"),
    ]
    for fname, label in required:
        p = ROOT / fname
        if not p.exists():
            stale.append(f"{label}: MISSING")
        else:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).date()
            if mtime < today:
                stale.append(f"{label}: {(today - mtime).days}d old")

    last_alert_date = state.get("freshness_alert_date", "")
    if stale and last_alert_date != today.isoformat():
        alerts.append({
            "title":    "⚠️ Canyon Pipeline: Stale Data",
            "body":     "Key outputs not updated today:\n" + "\n".join(f"• {s}" for s in stale)
                        + "\n\nCheck that launchd schedule ran successfully.",
            "priority": "high",
            "tags":     ["warning"],
        })
        state["freshness_alert_date"] = today.isoformat()
    return alerts


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  Canyon Push Alerts — {date.today()}")
    print(f"  Topic: {NTFY_TOPIC}")
    print("=" * 60)

    if NTFY_TOPIC == "canyon-quant-lynn":
        print("\n  ⚙️  Using default topic: canyon-quant-lynn")
        print("  To customize: set env var CANYON_NTFY_TOPIC=your-topic")
        print("  or edit NTFY_TOPIC in this file.\n")

    state = _load_state()
    all_alerts: list[dict] = []

    # Run each check
    checks = [
        ("Regime flip",         check_regime_flip),
        ("Bear probability",    check_bear_probability),
        ("Short urgents",       check_short_urgents),
        ("Top longs (daily)",   check_top_longs),
        ("Book drawdown",       check_book_drawdown),
        ("Pipeline freshness",  check_pipeline_freshness),
    ]
    for label, fn in checks:
        try:
            found = fn(state)
            if found:
                print(f"  [{label}] → {len(found)} alert(s)")
                all_alerts.extend(found)
            else:
                print(f"  [{label}] → nothing to send")
        except Exception as e:
            print(f"  [{label}] ERROR: {e}")

    # Send alerts
    sent = 0
    for a in all_alerts:
        ok = _send(
            title     = a["title"],
            body      = a["body"],
            priority  = a.get("priority", "default"),
            tags      = a.get("tags"),
            click_url = a.get("click_url", ""),
            actions   = a.get("actions", ""),
        )
        if ok:
            sent += 1

    _save_state(state)
    print(f"\n  {sent}/{len(all_alerts)} alerts sent → ntfy.sh/{NTFY_TOPIC}")
    if not all_alerts:
        print("  (No alerts today — all signals normal)")


if __name__ == "__main__":
    main()
