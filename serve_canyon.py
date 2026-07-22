#!/usr/bin/env python3
"""
Canyon Quant — local dashboard server.
Run this once; browse to http://localhost:8888
Data refreshes automatically in the background whenever it's stale.
"""
import http.server, subprocess, threading, json, time, sys
from pathlib import Path

ROOT   = Path(__file__).parent
HTML   = ROOT / "canyon_v9_research.html"
PORT   = 8888
STALE_HOURS = 8   # re-run pipeline if data is older than this

_lock   = threading.Lock()
_state  = {"running": False, "started_at": None, "last_ok": None, "error": None}


def _data_age_hours() -> float:
    """Hours since alpha_scores.csv was last written."""
    p = ROOT / "alpha_scores.csv"
    if not p.exists():
        return 999
    return (time.time() - p.stat().st_mtime) / 3600


def _is_stale() -> bool:
    return _data_age_hours() > STALE_HOURS


def _run_pipeline():
    with _lock:
        if _state["running"]:
            return
        _state["running"]    = True
        _state["started_at"] = time.time()
        _state["error"]      = None
    try:
        result = subprocess.run(
            [sys.executable, "run_daily.py"],
            cwd=ROOT, timeout=2400,
            capture_output=False,
        )
        if result.returncode != 0:
            _state["error"] = f"Pipeline exited with code {result.returncode}"
        else:
            _state["last_ok"] = time.time()
    except subprocess.TimeoutExpired:
        _state["error"] = "Pipeline timed out after 40 minutes"
    except Exception as e:
        _state["error"] = str(e)
    finally:
        _state["running"] = False


def _trigger_if_stale():
    if _is_stale() and not _state["running"]:
        print(f"  Data is {_data_age_hours():.1f}h old — refreshing in background…")
        threading.Thread(target=_run_pipeline, daemon=True).start()


class Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.startswith("/api/status"):
            self._json({
                "running":    _state["running"],
                "stale":      _is_stale(),
                "age_hours":  round(_data_age_hours(), 1),
                "error":      _state["error"],
                "html_mtime": HTML.stat().st_mtime if HTML.exists() else 0,
                "started_at": _state["started_at"],
            })

        elif self.path.startswith("/refresh"):
            threading.Thread(target=_run_pipeline, daemon=True).start()
            self._json({"ok": True, "message": "Refresh started"})

        else:
            if HTML.exists():
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
                # Trigger background refresh if stale
                threading.Thread(target=_trigger_if_stale, daemon=True).start()
            else:
                msg = b"<h2>Dashboard not found. Run run_daily.py first.</h2>"
                self.send_response(404)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(msg)

    def _json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass   # stay quiet unless there's an error


if __name__ == "__main__":
    print()
    print("  Canyon Quant dashboard server")
    print(f"  Open your browser at:  http://localhost:{PORT}")
    print(f"  Data age: {_data_age_hours():.1f}h  (refreshes automatically when >{STALE_HOURS}h old)")
    print()

    # Kick off a refresh immediately if stale
    _trigger_if_stale()

    try:
        server = http.server.HTTPServer(("", PORT), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"  Port {PORT} already in use — another Canyon server may be running.")
            print(f"  Open http://localhost:{PORT} in your browser.")
        else:
            raise
