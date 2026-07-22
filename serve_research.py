#!/usr/bin/env python3
"""
Canyon v9 — Dynamic Research Server
Serves canyon_v9_research.html with live data on every page load.
Run:  .venv/bin/python serve_research.py
Open: http://localhost:8513
"""
from __future__ import annotations
import json, os, sys, time, threading, subprocess, traceback, urllib.request, urllib.error
from pathlib import Path
from flask import Flask, Response, request

# Auto-load .env from project root
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() and _v.strip() and _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import update_research_html as ur

app = Flask(__name__)

# ── Alpaca price cache ────────────────────────────────────────────────────────
# When ALPACA_KEY_ID + ALPACA_KEY_SECRET are set, a background thread refreshes
# prices every 15 seconds from Alpaca's free REST snapshot API (IEX feed).
# /api/live reads from this cache first; falls back to yfinance if not populated.

_alpaca_cache: dict[str, dict] = {}  # {ticker: {price, chg_pct, open, time}}
_alpaca_cache_ts: float = 0.0

ALPACA_KEY_ID  = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_KEY_SEC = os.environ.get("ALPACA_KEY_SECRET", "")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/snapshots"


def _alpaca_snapshot(tickers: list[str]) -> dict[str, dict]:
    """Fetch latest quote + trade from Alpaca IEX snapshot endpoint."""
    if not (ALPACA_KEY_ID and ALPACA_KEY_SEC):
        return {}
    try:
        import requests as _req
        r = _req.get(
            ALPACA_DATA_URL,
            params={"symbols": ",".join(tickers), "feed": "iex"},
            headers={
                "APCA-API-KEY-ID":     ALPACA_KEY_ID,
                "APCA-API-SECRET-KEY": ALPACA_KEY_SEC,
            },
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        result = {}
        for tkr, snap in data.items():
            lat  = snap.get("latestTrade") or snap.get("dailyBar") or {}
            bar  = snap.get("dailyBar") or {}
            prev = snap.get("prevDailyBar") or {}
            price = lat.get("p") or bar.get("c")
            open_ = bar.get("o") or prev.get("c")
            if price is None:
                continue
            chg = (price - open_) / open_ if open_ and open_ > 0 else 0
            result[tkr] = {
                "price":   round(float(price), 2),
                "chg_pct": round(chg * 100, 2),
                "open":    round(float(open_), 2) if open_ else None,
            }
        return result
    except Exception:
        return {}


def _alpaca_refresh_loop():
    """Background thread: refresh price cache every 15 seconds using Alpaca."""
    global _alpaca_cache, _alpaca_cache_ts
    DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "VIX", "GLD", "TLT", "HYG"]
    while True:
        try:
            tickers = list(DEFAULT_TICKERS)
            wl_path = ROOT / "watchlist.json"
            if wl_path.exists():
                try:
                    wl = json.loads(wl_path.read_text())
                    tickers = list(dict.fromkeys(tickers + wl.get("tickers", [])[:20]))
                except Exception:
                    pass
            snap = _alpaca_snapshot(tickers)
            if snap:
                _alpaca_cache.update(snap)
                _alpaca_cache_ts = time.time()
        except Exception:
            pass
        time.sleep(15)


if ALPACA_KEY_ID and ALPACA_KEY_SEC:
    threading.Thread(target=_alpaca_refresh_loop, daemon=True).start()

STALE_HOURS = 8
_lock  = threading.Lock()
_state = {"running": False, "error": None, "last_ok": None}


def _data_age_hours() -> float:
    p = ROOT / "alpha_scores.csv"
    if not p.exists():
        return 999
    return (time.time() - p.stat().st_mtime) / 3600


def _run_pipeline():
    with _lock:
        if _state["running"]:
            return
        _state["running"] = True
        _state["error"]   = None
    try:
        result = subprocess.run(
            [sys.executable, "run_daily.py"],
            cwd=ROOT, timeout=2400, capture_output=False,
        )
        if result.returncode != 0:
            _state["error"] = f"Pipeline exited with code {result.returncode}"
        else:
            _state["last_ok"] = time.time()
    except Exception as e:
        _state["error"] = str(e)
    finally:
        _state["running"] = False


def _trigger_if_stale():
    if _data_age_hours() > STALE_HOURS and not _state["running"]:
        print(f"  Data is {_data_age_hours():.1f}h old — auto-refreshing in background…")
        threading.Thread(target=_run_pipeline, daemon=True).start()


def _build() -> str:
    daily      = ur.load_daily_report()
    chart      = ur.load_oos_chart()
    summ       = ur.load_oos_summary()
    accruals   = ur.load_accruals()
    squeeze    = ur.load_squeeze()
    live       = ur.load_live_data()
    bt_monthly = ur.load_backtest_monthly()
    paper_nav  = ur.load_paper_nav_chart()
    wf_steps    = ur.load_workflow_steps()
    wf_queue    = ur.load_workflow_queue()
    alpha_sc    = ur.load_alpha_scores()
    risk_gate   = ur.load_risk_gate()
    ticker_dd   = ur.load_ticker_drilldown()
    desk_mon    = ur.load_desk_monitor()
    sector_cy   = ur.load_sector_cycle()
    news_items  = ur.load_news()
    earn_cal    = ur.load_earnings_calendar()
    mac_breadth = ur.load_macro_breadth()
    roll_ic     = ur.load_rolling_ic()
    fac_attr    = ur.load_factor_attribution()
    mo_pnl      = ur.load_monthly_pnl()
    pos_pnl     = ur.load_position_pnl()
    crowd       = ur.load_crowding_monitor()
    macro_sigs  = ur.load_macro_signal_snapshot()
    v251_bt     = ur.load_v251_backtest()
    v251_regime = ur.load_v251_regime()
    deep        = ur.load_deep_analysis()
    sig_health  = ur.load_signal_health()
    barra_risk  = ur.load_barra_risk()
    hmm_data    = ur.load_hmm_regime()
    macro_out   = ur.load_macro_regime_outlook()
    econ_cal    = ur.load_economic_calendar()
    earn_ai     = ur.load_earnings_ai()
    watchlist   = ur.load_watchlist()
    return ur.build_html(daily, chart, summ, accruals, squeeze, live, bt_monthly, paper_nav,
                         wf_steps, wf_queue, alpha_sc, risk_gate, ticker_dd, desk_mon, sector_cy,
                         news_items, earn_cal, mac_breadth, roll_ic, fac_attr, mo_pnl,
                         pos_pnl, crowd, macro_sigs,
                         v251_bt=v251_bt, v251_regime=v251_regime, deep=deep,
                         signal_health=sig_health, barra_risk=barra_risk, hmm_data=hmm_data,
                         macro_outlook=macro_out, econ_cal=econ_cal,
                         earnings_ai=earn_ai, watchlist=watchlist)


# ── Canyon AI context builder ──────────────────────────────────────────────────

def _build_ai_context(ticker: str = "") -> str:
    """Build Canyon's data as a system prompt context for Claude."""
    lines = [
        "You are Canyon Quant's AI research assistant. You have access to the following live data:",
        "",
    ]

    # Regime
    try:
        hmm_path = ROOT / "hmm_regime_daily.csv"
        if hmm_path.exists():
            import pandas as pd
            hmm = pd.read_csv(hmm_path)
            regime = hmm["regime"].iloc[-1] if "regime" in hmm.columns else "UNKNOWN"
            lines.append(f"Current market regime (HMM): {regime}")
    except Exception:
        pass

    # Macro outlook
    try:
        outlook_path = ROOT / "macro_regime_outlook.json"
        if outlook_path.exists():
            out = json.loads(outlook_path.read_text())
            lines.append(f"4-week bear probability: {float(out.get('bear_prob_4w', 0)):.0%}")
            lines.append(f"Composite macro signal: {out.get('composite_signal', 'N/A')}")
    except Exception:
        pass

    # Top alpha scores
    try:
        import pandas as pd
        scores_path = ROOT / "alpha_scores.csv"
        if scores_path.exists():
            scores = pd.read_csv(scores_path)
            score_col = "alpha_score" if "alpha_score" in scores.columns else (
                "score" if "score" in scores.columns else None)
            if score_col:
                top = scores.sort_values(score_col, ascending=False).head(20)
                lines.append(f"\nTop 20 alpha-ranked tickers: {', '.join(top['ticker'].tolist())}")
    except Exception:
        pass

    # Ticker-specific data
    if ticker:
        ticker = ticker.upper().strip()
        lines.append(f"\n--- Ticker-specific data for {ticker} ---")
        try:
            import pandas as pd
            scores_path = ROOT / "alpha_scores.csv"
            if scores_path.exists():
                scores = pd.read_csv(scores_path)
                row = scores[scores["ticker"] == ticker]
                if not row.empty:
                    r = row.iloc[0]
                    for col in scores.columns:
                        val = r[col]
                        if str(val) not in ("nan", "None", ""):
                            lines.append(f"  {col}: {val}")
        except Exception:
            pass

        # DCF data
        try:
            import pandas as pd
            dcf_path = ROOT / "dcf_valuation.csv"
            if dcf_path.exists():
                dcf = pd.read_csv(dcf_path)
                row = dcf[dcf["ticker"] == ticker]
                if not row.empty:
                    r = row.iloc[0]
                    lines.append(f"\nDCF (Damodaran 3-stage):")
                    for col in ["wacc", "roic", "roic_wacc_spread", "eva_m", "iv_per_share",
                                "upside_pct", "pvgo_pct", "stage", "rev_growth_1y"]:
                        if col in r:
                            lines.append(f"  {col}: {r[col]}")
        except Exception:
            pass

        # Short scanner data
        try:
            import pandas as pd
            short_path = ROOT / "short_scanner.csv"
            if short_path.exists():
                shorts = pd.read_csv(short_path)
                row = shorts[shorts["ticker"] == ticker]
                if not row.empty:
                    r = row.iloc[0]
                    lines.append(f"\nShort Scanner:")
                    for col in ["score", "rsi", "entry_low", "entry_high", "stop_loss",
                                "target_1", "target_2", "rr_1", "signals", "urgency"]:
                        if col in r:
                            lines.append(f"  {col}: {r[col]}")
        except Exception:
            pass

        # Earnings AI summary
        try:
            import pandas as pd
            ai_path = ROOT / "earnings_ai_summaries.csv"
            if ai_path.exists():
                summaries = pd.read_csv(ai_path)
                row = summaries[summaries["ticker"] == ticker]
                if not row.empty:
                    r = row.iloc[0]
                    lines.append(f"\nEarnings AI Analysis:\n{r.get('summary', '')}")
        except Exception:
            pass

    lines += [
        "",
        "Answer concisely and accurately using this data. For questions about stocks not in Canyon's universe,",
        "draw on your general financial knowledge and clearly note when data comes from general knowledge vs. Canyon's system.",
        "Use numbers and be specific. Format dollar values with $. Avoid generic advice.",
    ]
    return "\n".join(lines)


def _claude_api(messages: list[dict], system: str) -> str:
    """Call Anthropic Claude API via HTTP. Returns answer text."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "To enable Canyon AI Chat, set your Anthropic API key:\n\n"
            "  export ANTHROPIC_API_KEY=your-key-here\n\n"
            "Then restart serve_research.py."
        )

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": system,
        "messages": messages,
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
            return resp["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return f"API error {e.code}: {body}"
    except Exception as e:
        return f"Error: {e}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    threading.Thread(target=_trigger_if_stale, daemon=True).start()
    try:
        html = _build()
        return Response(html, mimetype="text/html")
    except Exception:
        tb = traceback.format_exc()
        return Response(f"<pre>{tb}</pre>", status=500, mimetype="text/html")


@app.route("/refresh")
def refresh():
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return Response("Refresh started — come back in ~10 minutes.", mimetype="text/plain")


@app.route("/status")
def status():
    return Response(json.dumps({
        "data_age_hours":   round(_data_age_hours(), 1),
        "pipeline_running": _state["running"],
        "last_error":       _state["error"],
    }), mimetype="application/json")


@app.route("/save")
def save():
    try:
        html = _build()
        out  = ROOT / "canyon_v9_research.html"
        out.write_text(html, encoding="utf-8")
        return Response(f"Saved → {out}  ({len(html)//1024} KB)", mimetype="text/plain")
    except Exception:
        tb = traceback.format_exc()
        return Response(tb, status=500, mimetype="text/plain")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Natural language Q&A using Canyon data as context."""
    try:
        body     = json.loads(request.data)
        question = body.get("question", "").strip()
        ticker   = body.get("ticker", "").strip()
        history  = body.get("history", [])   # list of {role, content}

        if not question:
            return Response(json.dumps({"error": "empty question"}), mimetype="application/json")

        system   = _build_ai_context(ticker)
        messages = history[-8:] + [{"role": "user", "content": question}]
        answer   = _claude_api(messages, system)

        return Response(json.dumps({"answer": answer, "ticker": ticker}),
                        mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)


@app.route("/api/live")
def api_live():
    """Return intraday prices for portfolio / watchlist tickers.
    Source priority: Alpaca cache (15s freshness) → yfinance fallback.
    """
    try:
        import pandas as pd

        tickers_param = request.args.get("tickers", "")
        if tickers_param:
            tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
        else:
            tickers = []
            scores_path = ROOT / "alpha_scores.csv"
            if scores_path.exists():
                scores    = pd.read_csv(scores_path)
                score_col = next((c for c in ["alpha_score", "score"] if c in scores.columns), None)
                if score_col:
                    tickers = scores.sort_values(score_col, ascending=False)["ticker"].head(20).tolist()
            wl_path = ROOT / "watchlist.json"
            if wl_path.exists():
                wl = json.loads(wl_path.read_text())
                tickers = list(dict.fromkeys(tickers + wl.get("tickers", [])))
            if not tickers:
                tickers = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]

        result: dict[str, dict] = {}
        cache_age = time.time() - _alpaca_cache_ts

        # ── Primary: Alpaca cache (< 30s old) ────────────────────────────────
        if _alpaca_cache and cache_age < 30:
            for tkr in tickers:
                if tkr in _alpaca_cache:
                    result[tkr] = _alpaca_cache[tkr]
            missing = [t for t in tickers if t not in result]
        else:
            missing = tickers

        # ── Fallback: on-demand Alpaca REST for missing tickers ───────────────
        if missing and ALPACA_KEY_ID:
            snap = _alpaca_snapshot(missing)
            result.update(snap)
            _alpaca_cache.update(snap)
            missing = [t for t in missing if t not in result]

        # ── Final fallback: yfinance ──────────────────────────────────────────
        if missing:
            try:
                import yfinance as yf
                data = yf.download(missing, period="1d", interval="5m",
                                   progress=False, auto_adjust=True, group_by="ticker")
                for tkr in missing:
                    try:
                        if len(missing) == 1:
                            close = data["Close"].dropna()
                        else:
                            close = (data[tkr]["Close"].dropna()
                                     if tkr in data.columns.get_level_values(0) else pd.Series())
                        if close.empty:
                            continue
                        cur   = float(close.iloc[-1])
                        open_ = float(close.iloc[0])
                        chg   = (cur - open_) / open_ if open_ > 0 else 0
                        result[tkr] = {
                            "price":   round(cur, 2),
                            "chg_pct": round(chg * 100, 2),
                            "open":    round(open_, 2),
                        }
                    except Exception:
                        continue
            except Exception:
                pass

        source = ("alpaca" if (_alpaca_cache and cache_age < 30) else
                  "alpaca_rt" if ALPACA_KEY_ID else "yfinance")
        return Response(json.dumps({
            "prices": result,
            "as_of":  time.strftime("%H:%M"),
            "source": source,
        }), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)


@app.route("/api/watchlist", methods=["GET", "POST", "DELETE"])
def api_watchlist():
    """Read/write custom watchlist."""
    wl_path = ROOT / "watchlist.json"

    if request.method == "GET":
        if wl_path.exists():
            return Response(wl_path.read_text(), mimetype="application/json")
        return Response(json.dumps({"tickers": [], "notes": {}}), mimetype="application/json")

    elif request.method == "POST":
        body   = json.loads(request.data)
        ticker = body.get("ticker", "").upper().strip()
        note   = body.get("note", "")
        if not ticker:
            return Response(json.dumps({"error": "ticker required"}), mimetype="application/json")
        wl = json.loads(wl_path.read_text()) if wl_path.exists() else {"tickers": [], "notes": {}}
        if ticker not in wl["tickers"]:
            wl["tickers"].append(ticker)
        wl["notes"][ticker] = note
        wl_path.write_text(json.dumps(wl, indent=2))
        return Response(json.dumps({"ok": True, "ticker": ticker}), mimetype="application/json")

    elif request.method == "DELETE":
        body   = json.loads(request.data)
        ticker = body.get("ticker", "").upper().strip()
        if wl_path.exists():
            wl = json.loads(wl_path.read_text())
            wl["tickers"] = [t for t in wl["tickers"] if t != ticker]
            wl["notes"].pop(ticker, None)
            wl_path.write_text(json.dumps(wl, indent=2))
        return Response(json.dumps({"ok": True}), mimetype="application/json")


@app.route("/api/stockinfo/<ticker>")
def api_stockinfo(ticker):
    """Full stock profile: live price, Canyon signals, DCF, short scanner, earnings AI, analyst data."""
    import pandas as pd
    ticker = ticker.upper().strip()
    result = {"ticker": ticker, "as_of": time.strftime("%H:%M")}

    # ── Live price from yfinance ──────────────────────────────────────────────
    try:
        import yfinance as yf
        tkr  = yf.Ticker(ticker)
        info = tkr.info or {}
        hist = tkr.history(period="5d")
        price    = info.get("currentPrice") or info.get("regularMarketPrice")
        prev     = info.get("previousClose") or (float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None)
        chg_pct  = ((price - prev) / prev * 100) if (price and prev) else None
        day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
        day_low  = info.get("dayLow")  or info.get("regularMarketDayLow")
        w52_high = info.get("fiftyTwoWeekHigh")
        w52_low  = info.get("fiftyTwoWeekLow")
        # Intraday 5-min sparkline (last 1 day)
        try:
            intra = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=True)
            sparkline = [round(float(v), 2) for v in intra["Close"].dropna().tolist()[-48:]] if not intra.empty else []
        except Exception:
            sparkline = []

        result["live"] = {
            "price":       round(float(price), 2) if price else None,
            "chg_pct":     round(chg_pct, 2) if chg_pct is not None else None,
            "prev_close":  round(float(prev), 2) if prev else None,
            "day_high":    round(float(day_high), 2) if day_high else None,
            "day_low":     round(float(day_low), 2) if day_low else None,
            "w52_high":    round(float(w52_high), 2) if w52_high else None,
            "w52_low":     round(float(w52_low), 2) if w52_low else None,
            "market_cap_b": round(float(info.get("marketCap", 0)) / 1e9, 1),
            "sector":      info.get("sector", ""),
            "industry":    info.get("industry", ""),
            "name":        info.get("longName", ticker),
            "pe_ttm":      round(float(info.get("trailingPE", 0) or 0), 1) or None,
            "pe_fwd":      round(float(info.get("forwardPE", 0) or 0), 1) or None,
            "beta":        round(float(info.get("beta", 0) or 0), 2) or None,
            "short_float": round(float(info.get("shortPercentOfFloat", 0) or 0) * 100, 1),
            "analyst_rating": info.get("recommendationKey", ""),
            "analyst_target": round(float(info.get("targetMeanPrice", 0) or 0), 2) or None,
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
            "sparkline":   sparkline,
            "description": (info.get("longBusinessSummary") or "")[:300],
        }
    except Exception as e:
        result["live"] = {"error": str(e)}

    # ── Canyon alpha signal ──────────────────────────────────────────────────
    try:
        scores_path = ROOT / "alpha_scores.csv"
        if scores_path.exists():
            scores = pd.read_csv(scores_path)
            row = scores[scores["ticker"] == ticker]
            if not row.empty:
                r = row.iloc[0]
                score_col = next((c for c in ["alpha_score", "score"] if c in r.index), None)
                result["canyon_signal"] = {
                    "alpha_score":  float(r[score_col]) if score_col else None,
                    "rank":         int(scores[scores["ticker"] == ticker].index[0]) + 1,
                    "total_stocks": len(scores),
                    "signal_data":  {c: (None if str(r[c]) in ("nan","None","") else r[c])
                                     for c in scores.columns if c != "ticker"},
                }
    except Exception:
        pass

    # ── DCF valuation ────────────────────────────────────────────────────────
    try:
        dcf_path = ROOT / "dcf_valuation.csv"
        if dcf_path.exists():
            dcf = pd.read_csv(dcf_path)
            row = dcf[dcf["ticker"] == ticker]
            if not row.empty:
                r = row.iloc[0]
                result["dcf"] = {c: (None if str(r[c]) in ("nan","None","") else r[c])
                                 for c in dcf.columns}
    except Exception:
        pass

    # ── Short scanner ────────────────────────────────────────────────────────
    try:
        short_path = ROOT / "short_scanner.csv"
        if short_path.exists():
            shorts = pd.read_csv(short_path)
            row = shorts[shorts["ticker"] == ticker]
            if not row.empty:
                r = row.iloc[0]
                result["short"] = {c: (None if str(r[c]) in ("nan","None","") else r[c])
                                   for c in shorts.columns}
    except Exception:
        pass

    # ── Earnings AI summary ──────────────────────────────────────────────────
    try:
        ai_path = ROOT / "earnings_ai_summaries.csv"
        if ai_path.exists():
            ai_df = pd.read_csv(ai_path)
            row   = ai_df[ai_df["ticker"] == ticker]
            if not row.empty:
                r = row.iloc[0]
                result["earnings_ai"] = {
                    "summary":       str(r.get("summary", "")),
                    "has_ai":        bool(r.get("has_ai_summary", False)),
                    "pe_fwd":        r.get("pe_fwd"),
                    "rev_growth":    r.get("rev_growth_yoy"),
                    "beats_last_4q": r.get("beats_last_4q"),
                }
    except Exception:
        pass

    # ── Macro regime for context ─────────────────────────────────────────────
    try:
        hmm_path = ROOT / "hmm_regime_daily.csv"
        if hmm_path.exists():
            hmm = pd.read_csv(hmm_path)
            result["regime"] = str(hmm["regime"].iloc[-1]) if "regime" in hmm.columns else "UNKNOWN"
    except Exception:
        pass

    return Response(json.dumps(result, default=str), mimetype="application/json")


@app.route("/sw.js")
def service_worker():
    """PWA Service Worker — offline cache + stale-while-revalidate for stock info."""
    sw_js = r"""
const CACHE_NAME = 'canyon-v2';
const STATIC = ['/', '/manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Real-time endpoints: always network, no cache
  if (url.pathname.startsWith('/api/live') ||
      url.pathname.startsWith('/api/chat') ||
      url.pathname === '/refresh') {
    return;
  }

  // Stock info: stale-while-revalidate (show cached, update in background)
  if (url.pathname.startsWith('/api/stockinfo')) {
    e.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match(e.request);
        const fetchPromise = fetch(e.request).then(resp => {
          if (resp.ok) cache.put(e.request, resp.clone());
          return resp;
        }).catch(() => cached);
        return cached || fetchPromise;
      })
    );
    return;
  }

  // Dashboard HTML: cache-first, network fallback
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(resp => {
        if (resp.ok) {
          caches.open(CACHE_NAME).then(c => c.put(e.request, resp.clone()));
        }
        return resp;
      });
    })
  );
});
"""
    return Response(sw_js, mimetype="application/javascript",
                    headers={"Service-Worker-Allowed": "/"})


@app.route("/manifest.json")
def manifest():
    """PWA manifest for installing Canyon to iPhone home screen."""
    m = {
        "name":             "Canyon Quant",
        "short_name":       "Canyon",
        "description":      "Systematic quant research dashboard",
        "start_url":        "/",
        "display":          "standalone",
        "background_color": "#0D1B35",
        "theme_color":      "#0D1B35",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    }
    return Response(json.dumps(m), mimetype="application/manifest+json")


@app.route("/icon-<size>.png")
def icon(size):
    """Minimal navy square icon for PWA (SVG→PNG via data URI fallback)."""
    # Generate a simple SVG-based PNG placeholder
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size.split('x')[0] if 'x' in size else 192}" height="{size.split('x')[0] if 'x' in size else 192}">
  <rect width="100%" height="100%" fill="#0D1B35"/>
  <text x="50%" y="55%" font-family="serif" font-size="80" fill="#B8943F" text-anchor="middle">C</text>
</svg>'''
    return Response(svg, mimetype="image/svg+xml")


if __name__ == "__main__":
    print("Canyon v9 Research Server")
    print(f"  URL:    http://localhost:8513")
    print(f"  Data age: {_data_age_hours():.1f}h  (auto-refreshes when >{STALE_HOURS}h old)")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    print(f"  AI Chat: {'enabled' if has_key else 'no ANTHROPIC_API_KEY — chat will prompt for key'}")
    print()
    _trigger_if_stale()
    app.run(host="0.0.0.0", port=8513, debug=False, use_reloader=False)
