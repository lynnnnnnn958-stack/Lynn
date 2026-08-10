#!/usr/bin/env python3
"""Bootstrap runner for the daily pipeline.

WHY: the network steps (yfinance / SEC EDGAR / FRED / CFTC) had no socket
timeout, so a single stuck recv() would block until the per-step watchdog
killed the whole step — surfacing as "a different step FAILED each run"
(the 6-偶发失败 symptom). This runner installs a global default socket
timeout BEFORE the target script imports anything, then executes the target
as if it were run directly (__name__ == "__main__", correct __file__/argv),
so a hung download fails fast and run_daily's retry can recover.

Usage:  python3 _canyon_run.py <target_script.py> [args...]

The framework Python here does not auto-import a project-local sitecustomize,
so this explicit bootstrap is the reliable place to set the timeout for all
~70 steps without editing each script.
"""
import os
import runpy
import socket
import sys

try:
    _t = float(os.environ.get("CANYON_SOCKET_TIMEOUT", "45"))
except Exception:
    _t = 45.0

if _t > 0:
    # (1) urllib / raw-socket calls (FRED, CFTC, some SEC) respect the socket
    #     default timeout.
    try:
        socket.setdefaulttimeout(_t)
    except Exception:
        pass
    # (2) requests-based calls (yfinance downloads, EDGAR) do NOT respect the
    #     socket default — requests sets its own per-request timeout (None =
    #     wait forever) onto the socket, overriding it. That was the real cause
    #     of steps hanging until the watchdog killed them. Patch Session.request
    #     to inject a default timeout whenever the caller didn't pass one.
    try:
        import requests  # imported before the target script imports yfinance
        _orig_request = requests.Session.request

        def _request_with_timeout(self, method, url, **kw):
            if kw.get("timeout") is None:
                kw["timeout"] = _t
            return _orig_request(self, method, url, **kw)

        requests.Session.request = _request_with_timeout
    except Exception:
        pass
    # (3) yfinance 1.3+ does NOT use the standard requests lib — it uses
    #     `from curl_cffi import requests` with Session(impersonate="chrome").
    #     That is the backend for ALL price downloads, so it must be bounded too;
    #     this is the actual fix for the hanging price-signal step.
    try:
        from curl_cffi import requests as _cc
        _orig_cc = _cc.Session.request

        def _cc_request_with_timeout(self, method, url, *a, **kw):
            if kw.get("timeout") is None:
                kw["timeout"] = _t
            return _orig_cc(self, method, url, *a, **kw)

        _cc.Session.request = _cc_request_with_timeout
    except Exception:
        pass

if len(sys.argv) < 2:
    sys.stderr.write("usage: _canyon_run.py <script.py> [args...]\n")
    raise SystemExit(2)

_target = sys.argv[1]
# Make the target see a normal argv: argv[0] = its own path, then its args.
sys.argv = sys.argv[1:]
runpy.run_path(_target, run_name="__main__")
