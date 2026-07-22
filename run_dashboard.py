#!/usr/bin/env python3
"""
run_dashboard.py — Canyon Quant real-time dashboard server
===========================================================
Usage:
  python3 run_dashboard.py              # price refresh every 3 min (default)
  python3 run_dashboard.py --interval 2 # refresh every 2 min

What it does:
  1. Initial price refresh + full HTML rebuild
  2. Starts local HTTP server on http://localhost:8080
  3. Every N minutes: re-fetches prices → rebuilds HTML → browser auto-reloads
  4. Opens browser automatically
  5. /api/chat  — Canyon AI chat backed by Claude (needs ANTHROPIC_API_KEY env var)
"""
import subprocess, threading, time, json, pathlib, webbrowser, sys, os
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT      = pathlib.Path(__file__).parent
PORT      = 8080
URL       = f"http://localhost:{PORT}/canyon_v24_research.html"
TS_FILE   = ROOT / "last_updated.json"

INTERVAL_MIN = 3
_args = sys.argv[1:]
if "--interval" in _args:
    try:
        INTERVAL_MIN = int(_args[_args.index("--interval") + 1])
    except (IndexError, ValueError):
        pass
INTERVAL_SEC = INTERVAL_MIN * 60


def _run(script: str) -> bool:
    r = subprocess.run(
        [sys.executable, str(ROOT / script)],
        capture_output=True, text=True, cwd=ROOT,
    )
    if r.returncode != 0 and r.stderr:
        tail = "\n  ".join(r.stderr.strip().splitlines()[-3:])
        print(f"  [warn] {script}:\n  {tail}")
    return r.returncode == 0


def _write_ts():
    with open(TS_FILE, "w", encoding="utf-8") as f:
        json.dump({"ts": round(time.time()), "hhmm": time.strftime("%H:%M")}, f)


_flow_tick = 0  # run options flow every 3 cycles (~9 min)
_daily_lock = threading.Lock()
_daily_running = False
_DAILY_STAMP = ROOT / ".daily_run_date"


def _daily_run_needed() -> bool:
    """Return True if run_daily.py hasn't been run today yet."""
    try:
        stamp = _DAILY_STAMP.read_text().strip()
        return stamp != time.strftime("%Y-%m-%d")
    except Exception:
        return True


def _run_daily_async():
    """Run the full daily pipeline in a background thread (non-blocking)."""
    global _daily_running
    with _daily_lock:
        if _daily_running:
            return
        _daily_running = True
    try:
        print(f"\n[{time.strftime('%H:%M:%S')}] ═══ Starting full daily pipeline (run_daily.py) …")
        r = subprocess.run(
            [sys.executable, str(ROOT / "run_daily.py")],
            cwd=ROOT, text=True, capture_output=True,
        )
        if r.returncode == 0:
            _DAILY_STAMP.write_text(time.strftime("%Y-%m-%d"))
            print(f"[{time.strftime('%H:%M:%S')}] ═══ Daily pipeline complete ✓")
        else:
            tail = "\n  ".join((r.stderr or "").strip().splitlines()[-5:])
            print(f"[{time.strftime('%H:%M:%S')}] ═══ Daily pipeline had errors:\n  {tail}")
            # Still stamp so we don't retry every 3 min if it fails
            _DAILY_STAMP.write_text(time.strftime("%Y-%m-%d"))
    finally:
        global _daily_running
        _daily_running = False


def refresh():
    global _flow_tick
    # ── Trigger full daily pipeline once per day (runs async, non-blocking) ──
    if _daily_run_needed() and not _daily_running:
        threading.Thread(target=_run_daily_async, daemon=True).start()

    print(f"\n[{time.strftime('%H:%M:%S')}] Refreshing prices …")
    ok = _run("step_price_refresh_rt.py")
    if not ok:
        print("  Price refresh failed — skipping HTML rebuild")
        return
    _run("step_etf_flow_rt.py")
    _flow_tick += 1
    if _flow_tick % 3 == 1:
        print(f"[{time.strftime('%H:%M:%S')}] Scanning options flow …")
        _run("step_options_flow.py")
    print(f"[{time.strftime('%H:%M:%S')}] Rebuilding HTML …")
    if _run("update_research_html.py"):
        _write_ts()
        print(f"[{time.strftime('%H:%M:%S')}] Done ✓")


def _refresh_loop():
    while True:
        time.sleep(INTERVAL_SEC)
        try:
            refresh()
        except Exception as e:
            print(f"  Loop error: {e}")


# ── Canyon AI chat context builder ────────────────────────────────────────────

def _build_context(ticker: str = "") -> str:
    """Load relevant CSV data into a compact text context for Claude."""
    import csv, io
    lines = ["=== CANYON QUANT SYSTEM CONTEXT ===\n"]

    def _read_csv(path, max_rows=20):
        try:
            import pandas as pd
            df = pd.read_csv(ROOT / path)
            return df.head(max_rows).to_string(index=False)
        except Exception:
            return ""

    # Market regime
    try:
        import pandas as pd
        r = pd.read_csv(ROOT / "hmm_regime_log.csv").iloc[-1]
        lines.append(f"Market Regime: {r.get('regime','?')}  Bear prob 4w: {r.get('bear_prob_4w','?')}")
    except Exception:
        pass

    # Top alpha scores
    try:
        import pandas as pd
        df = pd.read_csv(ROOT / "alpha_scores.csv").sort_values("alpha_rank").head(30)
        lines.append("\nTop 30 Alpha Scores (ranked):")
        lines.append(df[["ticker","alpha_score","alpha_rank","signal","sector"]].to_string(index=False))
    except Exception:
        pass

    # Ticker-specific deep dive
    if ticker:
        lines.append(f"\n=== TICKER FOCUS: {ticker} ===")
        try:
            import pandas as pd
            df = pd.read_csv(ROOT / "alpha_scores.csv")
            row = df[df["ticker"] == ticker]
            if not row.empty:
                lines.append(row.to_string(index=False))
        except Exception:
            pass
        try:
            import pandas as pd
            df = pd.read_csv(ROOT / "dcf_valuations.csv") if (ROOT/"dcf_valuations.csv").exists() else pd.DataFrame()
            if not df.empty:
                row = df[df["ticker"] == ticker]
                if not row.empty:
                    lines.append("DCF: " + row.to_string(index=False))
        except Exception:
            pass
        try:
            import pandas as pd
            df = pd.read_csv(ROOT / "short_scores.csv") if (ROOT/"short_scores.csv").exists() else pd.DataFrame()
            if not df.empty:
                row = df[df["ticker"] == ticker]
                if not row.empty:
                    lines.append("Short: " + row.to_string(index=False))
        except Exception:
            pass

    # Paper positions
    try:
        import pandas as pd
        df = pd.read_csv(ROOT / "paper_positions.csv")
        lines.append("\nPaper Positions:")
        lines.append(df.head(20).to_string(index=False))
    except Exception:
        pass

    # ETF sector rotation
    try:
        with open(ROOT / "etf_flow_daily.json") as f:
            etf = json.load(f)
        top3 = etf.get("sectors", [])[:3]
        lines.append("\nSector Rotation (top 3 by momentum):")
        for s in top3:
            lines.append(f"  {s['etf']} {s['name']}: 1D={s['ret_1d']:+.2f}%  5D={s['ret_5d']:+.2f}%  [{s['flow_signal']}]")
    except Exception:
        pass

    return "\n".join(lines)


def _handle_chat(body_bytes: bytes) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set. Add it to your shell: export ANTHROPIC_API_KEY=sk-ant-..."}

    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic package not installed. Run: pip3 install anthropic"}

    try:
        payload  = json.loads(body_bytes)
        question = payload.get("question", "").strip()
        ticker   = payload.get("ticker", "").strip().upper()
        history  = payload.get("history", [])   # list of {role, content}
    except Exception:
        return {"error": "Invalid request body"}

    if not question:
        return {"error": "Empty question"}

    context = _build_context(ticker)
    system_prompt = f"""You are Canyon AI, a quantitative research assistant embedded in Canyon Quant — a personal S&P 500 research dashboard.

You have access to the following live data from the user's system:
{context}

Guidelines:
- Be direct, specific, and quantitative. Reference actual scores, percentages, and rankings from the data above.
- When discussing signals, mention the alpha_score (0-100), signal direction (LONG/SHORT/NEUTRAL), and sector.
- For portfolio questions, reference the paper positions and regime.
- Keep answers concise — 3-6 sentences or a short bulleted list. No filler phrases.
- If you don't have data on something, say so clearly.
- This is paper trading only. Never suggest real trades."""

    messages = []
    for h in history[-8:]:  # keep last 8 turns for context
        role = h.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": h.get("content", "")})
    messages.append({"role": "user", "content": question})

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system_prompt,
            messages=messages,
        )
        answer = response.content[0].text
        return {"answer": answer}
    except Exception as e:
        return {"error": f"API error: {e}"}


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            result = _handle_chat(body)
            resp   = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_error(404)


if __name__ == "__main__":
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print("═" * 56)
    print("  Canyon Quant — Real-Time Dashboard")
    print(f"  Refresh interval : every {INTERVAL_MIN} min")
    print(f"  Dashboard URL    : {URL}")
    print(f"  AI Chat          : {'✓ API key found' if api_key_set else '✗ Set ANTHROPIC_API_KEY to enable'}")
    print("═" * 56)

    print("\n[startup] Initial price refresh + HTML build …")
    refresh()

    threading.Thread(target=_refresh_loop, daemon=True).start()

    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(URL)
    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n  Ctrl-C to stop\n")
    server = HTTPServer(("", PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
