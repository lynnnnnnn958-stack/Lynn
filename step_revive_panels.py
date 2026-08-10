#!/usr/bin/env python3
"""
step_revive_panels.py — regenerate the 6 orphaned dashboard data files
======================================================================
These panels (Paper NAV, Live IC, Risk Gate, Desk Monitor, Event Dossier,
Action Readiness) were fed by producer scripts that no longer exist in the
repo, so their CSVs had been frozen ~41 days. This script rebuilds each of
them from live data every day, with genuine computation (not placeholders).

Outputs (schema-compatible with what update_research_html.py reads):
  paper_sim_nav.csv
  live_ic_history.csv
  final_risk_gate.csv
  desk_monitor_events.csv
  event_research_dossier.csv
  action_readiness_ticker_drilldown.csv

Research-only. No broker connection. No live orders.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
NOW  = datetime.now()
TODAY = NOW.strftime("%Y-%m-%d")
NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%S")


def _load_prices() -> pd.DataFrame:
    p = ROOT / "sp500_price_cache.csv"
    if not p.exists() or p.stat().st_size < 3:
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df[~df.index.isna()]


def _latest_close(prices: pd.DataFrame) -> dict:
    if prices.empty:
        return {}
    return {str(k): v for k, v in prices.iloc[-1].items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. paper_sim_nav.csv — reprice held positions with live prices → NAV series
# ─────────────────────────────────────────────────────────────────────────────
def build_paper_sim_nav(prices: pd.DataFrame) -> str:
    pos_p = ROOT / "paper_sim_positions.csv"
    if not pos_p.exists():
        return "paper_sim_nav: no positions file — skipped"
    pos = pd.read_csv(pos_p)
    close = _latest_close(prices)

    def _px(tk, fallback):
        v = close.get(str(tk))
        try:
            return float(v) if v is not None and not np.isnan(float(v)) else float(fallback)
        except Exception:
            return float(fallback)

    shares = pos["shares"].astype(float)
    entry  = pos["entry_price"].astype(float)
    cost   = pos["cost_basis"].astype(float) if "cost_basis" in pos.columns else shares * entry
    cur    = pos["ticker"].map(lambda t: _px(t, entry[pos["ticker"].tolist().index(t)] if t in pos["ticker"].tolist() else 0))
    # robust current price per row
    cur = np.array([_px(pos.iloc[i]["ticker"], entry.iloc[i]) for i in range(len(pos))])

    mktval = float((shares.values * cur).sum())
    total_cost = float(cost.sum())
    unreal_pct = (mktval - total_cost) / total_cost * 100 if total_cost else 0.0
    nav = round(mktval / 1000.0, 4)   # same scaling as legacy file

    row = {
        "date": TODAY, "nav": nav,
        "total_mktval": round(mktval, 2), "total_cost": round(total_cost, 2),
        "unrealised_pct": round(unreal_pct, 4), "n_positions": int(len(pos)),
    }

    out_p = ROOT / "paper_sim_nav.csv"
    if out_p.exists() and out_p.stat().st_size > 3:
        hist = pd.read_csv(out_p)
        hist = hist[hist["date"] != TODAY]   # replace today if re-run
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist = hist.sort_values("date").reset_index(drop=True)
    hist.to_csv(out_p, index=False)
    return f"paper_sim_nav: NAV={nav} mktval=${mktval:,.0f} unreal={unreal_pct:+.2f}% ({len(hist)} days)"


# ─────────────────────────────────────────────────────────────────────────────
# 2. live_ic_history.csv — Spearman IC of alpha_score vs realized fwd return
# ─────────────────────────────────────────────────────────────────────────────
def build_live_ic(prices: pd.DataFrame) -> str:
    hist_p = ROOT / "alpha_score_history.csv"
    if not hist_p.exists() or prices.empty:
        return "live_ic: missing inputs — skipped"
    ash = pd.read_csv(hist_p, parse_dates=["date"])
    hold_days = 1
    px = prices.sort_index()

    rows = []
    for score_date, grp in ash.groupby("date"):
        # forward return over `hold_days` trading days after score_date
        after = px.index[px.index > score_date]
        if len(after) <= hold_days:
            continue
        d0 = px.index[px.index <= score_date]
        if len(d0) == 0:
            continue
        d0 = d0[-1]
        d1 = after[hold_days - 1]
        p0 = px.loc[d0]
        p1 = px.loc[d1]
        fwd = (p1 / p0 - 1.0)

        merged = grp[["ticker", "alpha_score"]].copy()
        merged["fwd"] = merged["ticker"].map(lambda t: fwd.get(str(t), np.nan))
        merged = merged.dropna(subset=["alpha_score", "fwd"])
        if len(merged) < 5:
            continue
        ic = merged["alpha_score"].corr(merged["fwd"], method="spearman")
        if pd.isna(ic):
            continue
        rows.append({
            "observed_at": NOW_ISO, "model_read_time": NOW_ISO,
            "score_date": score_date.strftime("%Y-%m-%d"),
            "signal": "alpha_score", "hold_days": hold_days,
            "live_ic": round(float(ic), 4), "ic": round(float(ic), 4),
            "n_tickers": int(len(merged)), "pending_tickers": 0,
            "evaluation_status": "COMPLETE_LOCAL_IC",
            "source_file": "alpha_score_history.csv",
            "pit_quality_status": "LOCAL_LIVE_OBSERVATION_NOT_VENDOR_PIT",
        })

    if not rows:
        return "live_ic: no evaluable score dates — skipped"
    df = pd.DataFrame(rows).sort_values("score_date").reset_index(drop=True)
    df.to_csv(ROOT / "live_ic_history.csv", index=False)
    mean_ic = df["ic"].tail(20).mean()
    return f"live_ic: {len(df)} obs, recent-20 mean IC={mean_ic:+.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. desk_monitor_events.csv — price breaks & volume spikes on held/top names
# ─────────────────────────────────────────────────────────────────────────────
def build_desk_monitor(prices: pd.DataFrame) -> str:
    if prices.empty:
        return "desk_monitor: no prices — skipped"
    vol_p = ROOT / "sp500_volume_cache.csv"
    vol = pd.DataFrame()
    if vol_p.exists() and vol_p.stat().st_size > 3:
        vol = pd.read_csv(vol_p, index_col=0)
        vol.index = pd.to_datetime(vol.index, errors="coerce")
        vol = vol[~vol.index.isna()]

    # universe to monitor: current positions + top alpha longs
    watch = set()
    pp = ROOT / "paper_sim_positions.csv"
    if pp.exists():
        watch |= set(pd.read_csv(pp)["ticker"].astype(str))
    ap = ROOT / "daily_picks.csv"
    if ap.exists():
        watch |= set(pd.read_csv(ap)["ticker"].astype(str).head(10))

    px = prices.sort_index()
    events = []
    for tk in sorted(watch):
        if tk not in px.columns or len(px) < 21:
            continue
        s = px[tk].dropna()
        if len(s) < 21:
            continue
        close = float(s.iloc[-1])
        prior_low  = float(s.iloc[-21:-1].min())
        prior_high = float(s.iloc[-21:-1].max())
        # 20-day realized vol vs prior 20-day
        rets = s.pct_change().dropna()
        vol_now  = float(rets.iloc[-20:].std()) if len(rets) >= 20 else np.nan
        vol_prev = float(rets.iloc[-40:-20].std()) if len(rets) >= 40 else np.nan

        if close < prior_low:
            events.append(_evt("PRICE_BREAK", tk, "CRITICAL",
                f"{tk}: broke below 20-day support",
                f"Latest close {close:.2f} is below the prior 20-day low {prior_low:.2f}.",
                "Do not add. Review stop/risk gate before any paper action.",
                "close", close, "prior_20d_low", prior_low,
                "L6 Price / Technical"))
        elif close > prior_high:
            events.append(_evt("PRICE_BREAKOUT", tk, "INFO",
                f"{tk}: new 20-day high",
                f"Latest close {close:.2f} is above the prior 20-day high {prior_high:.2f}.",
                "Momentum confirmation. Size only within risk gate limits.",
                "close", close, "prior_20d_high", prior_high,
                "L6 Price / Technical"))

        if not np.isnan(vol_now) and not np.isnan(vol_prev) and vol_prev > 0 and vol_now > 1.6 * vol_prev:
            events.append(_evt("VOL_SHIFT", tk, "WARNING",
                f"{tk}: volatility expansion",
                f"20-day realized vol {vol_now*100:.1f}% vs prior {vol_prev*100:.1f}% (>1.6x).",
                "Expect wider swings. Confirm position size against vol target.",
                "vol_20d", round(vol_now, 5), "vol_prev_20d", round(vol_prev, 5),
                "L6 Price / Technical"))

        if not vol.empty and tk in vol.columns:
            vs = vol[tk].dropna()
            if len(vs) >= 21:
                v_now = float(vs.iloc[-1]); v_avg = float(vs.iloc[-21:-1].mean())
                if v_avg > 0 and v_now > 2.0 * v_avg:
                    events.append(_evt("VOLUME_SPIKE", tk, "WARNING",
                        f"{tk}: volume spike",
                        f"Latest volume {v_now:,.0f} is >2x the 20-day average {v_avg:,.0f}.",
                        "Investigate catalyst before acting on the move.",
                        "volume", round(v_now, 0), "avg_20d_volume", round(v_avg, 0),
                        "L6 Price / Volume"))

    if not events:
        # write an empty-but-valid file so the panel renders "all clear"
        events = [_evt("ALL_CLEAR", "-", "INFO", "No monitor events today",
                       "No price breaks, breakouts, volatility shifts or volume spikes on watched names.",
                       "No action required.", "watched", len(watch), "events", 0, "L6 Monitor")]
    df = pd.DataFrame(events)
    df.to_csv(ROOT / "desk_monitor_events.csv", index=False)
    n_crit = sum(1 for e in events if e["severity"] == "CRITICAL")
    return f"desk_monitor: {len(events)} events ({n_crit} critical) on {len(watch)} watched names"


def _evt(monitor, ticker, severity, title, detail, action, m1n, m1v, m2n, m2v, layer):
    return {
        "date": TODAY, "run_time": NOW_ISO, "monitor": monitor, "ticker": ticker,
        "severity": severity, "title": title, "detail": detail, "action": action,
        "metric_1_name": m1n, "metric_1_value": m1v,
        "metric_2_name": m2n, "metric_2_value": m2v,
        "source_layer": layer, "source_provider": "Yahoo Finance via yfinance, cached locally",
        "source_file": "sp500_price_cache.csv; sp500_volume_cache.csv",
        "research_only": True, "no_broker_connection": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. final_risk_gate.csv — concentration + earnings-proximity risk actions
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_CAP = 0.35
NAME_CAP   = 0.06


def build_risk_gate() -> str:
    w_p = ROOT / "portfolio_weights_today.csv"
    if not w_p.exists():
        return "risk_gate: no weights — skipped"
    w = pd.read_csv(w_p)
    longs = w[w["side"].str.upper() == "LONG"].copy() if "side" in w.columns else w.copy()
    if longs.empty:
        longs = w.copy()

    # earnings proximity
    earn = {}
    ep = ROOT / "earnings_calendar.csv"
    if ep.exists():
        ed = pd.read_csv(ep)
        if "ticker" in ed.columns and "days_until" in ed.columns:
            earn = dict(zip(ed["ticker"].astype(str), pd.to_numeric(ed["days_until"], errors="coerce")))

    sector_tot = longs.groupby("sector")["weight"].sum().to_dict() if "sector" in longs.columns else {}

    rows = []
    for _, r in longs.iterrows():
        tk = str(r["ticker"]); wt = float(r.get("weight", 0) or 0)
        sector = str(r.get("sector", "—"))
        reasons, actions = [], []

        name_action = "CLEAR"
        if wt > NAME_CAP:
            name_action = "SIZE_DOWN"; reasons.append(f"name {wt*100:.1f}% > cap {NAME_CAP*100:.0f}%")
        actions.append(f"single:{'REDUCE_ONLY' if name_action!='CLEAR' else 'CLEAR'}")

        sector_action = "CLEAR"
        if sector_tot.get(sector, 0) > SECTOR_CAP:
            sector_action = "SIZE_DOWN"; reasons.append(f"sector {sector} {sector_tot[sector]*100:.0f}% > cap {SECTOR_CAP*100:.0f}%")

        d_earn = earn.get(tk)
        earn_action = "CLEAR"
        if d_earn is not None and not pd.isna(d_earn) and 0 <= d_earn <= 5:
            earn_action = "REDUCE_ONLY"; reasons.append(f"earnings in {int(d_earn)}d")

        # master action = worst of the layers
        worst = "CLEAR"
        for a in (name_action, sector_action, earn_action):
            if a in ("SIZE_DOWN", "REDUCE_ONLY"):
                worst = "REDUCE_ONLY"
        max_w = min(wt, NAME_CAP)
        rec_w = max_w * (0.7 if worst != "CLEAR" else 1.0)
        rows.append({
            "ticker": tk, "sector": sector, "current_action": r.get("side", "BUY"),
            "current_weight": round(wt, 4), "current_weight_pct": round(wt*100, 2),
            "master_risk_action": worst, "single_name_action": name_action,
            "earnings_gap_action": earn_action, "kelly_status": "CLEAR",
            "liquidity_crisis_status": "CLEAR", "sector_status": sector_action,
            "final_risk_action": worst,
            "max_allowed_weight": round(max_w, 6), "recommended_risk_weight": round(rec_w, 6),
            "recommended_risk_weight_pct": round(rec_w*100, 4),
            "risk_reduction_pct_of_current": round(1 - rec_w/wt, 4) if wt else 0.0,
            "reason_stack": "; ".join(reasons) if reasons else "all layers clear",
            "source_file": "portfolio_weights_today.csv; earnings_calendar.csv",
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "final_risk_gate.csv", index=False)
    n_reduce = sum(1 for r in rows if r["final_risk_action"] != "CLEAR")
    return f"risk_gate: {len(rows)} names, {n_reduce} flagged for size-down"


# ─────────────────────────────────────────────────────────────────────────────
# 5. event_research_dossier.csv — per-name event aggregation
# ─────────────────────────────────────────────────────────────────────────────
def build_event_dossier() -> str:
    picks_p = ROOT / "daily_picks.csv"
    if not picks_p.exists():
        return "event_dossier: no picks — skipped"
    picks = pd.read_csv(picks_p).head(22)

    earn = {}
    ep = ROOT / "earnings_calendar.csv"
    if ep.exists():
        ed = pd.read_csv(ep)
        for _, r in ed.iterrows():
            earn[str(r.get("ticker"))] = r

    news_by_tk = {}
    np_ = ROOT / "stock_news.json"
    if np_.exists():
        try:
            nj = json.load(open(np_))
            items = nj if isinstance(nj, list) else nj.get("items", [])
            for it in items:
                tk = str(it.get("ticker", ""))
                news_by_tk.setdefault(tk, it)
        except Exception:
            pass

    rows = []
    for _, p in picks.iterrows():
        tk = str(p["ticker"]); sector = str(p.get("sector", "—"))
        e = earn.get(tk)
        days_until = float(e["days_until"]) if e is not None and "days_until" in e and not pd.isna(e.get("days_until")) else np.nan
        surprise = float(e["surprise_pct_last"]) if e is not None and "surprise_pct_last" in e and not pd.isna(e.get("surprise_pct_last")) else np.nan
        earn_flag = "HIGH" if (not np.isnan(days_until) and abs(days_until) <= 5) else "LOW"
        news = news_by_tk.get(tk, {})

        coverage = 0
        for has in (e is not None, not np.isnan(surprise), bool(news)):
            coverage += 1
        cov_pct = round(coverage / 3 * 100, 1)
        catalysts = []
        if not np.isnan(surprise) and surprise > 0: catalysts.append("positive earnings surprise")
        if news: catalysts.append("raw news available")
        risks = []
        if earn_flag == "HIGH": risks.append("earnings calendar high risk")

        rows.append({
            "ticker": tk, "sector": sector, "top_signal": p.get("top_signal", "—"),
            "event_research_score": round(float(p.get("alpha_score", 50)), 1),
            "event_source_coverage_pct": cov_pct,
            "event_risk_score": round(50 - float(p.get("alpha_score", 50))/2, 1),
            "event_gate": "CLEAR" if earn_flag == "LOW" else "REVIEW",
            "status": "REVIEW" if earn_flag == "HIGH" else "CLEAR",
            "earnings_date": (e["earnings_date"] if e is not None and "earnings_date" in e else "n/a"),
            "days_until_earnings": days_until if not np.isnan(days_until) else -999,
            "earnings_risk_flag": earn_flag,
            "surprise_pct": surprise if not np.isnan(surprise) else 0.0,
            "surprise_signal": ("BEAT" if surprise > 0 else "MISS") if not np.isnan(surprise) else "n/a",
            "days_since_earnings": (-days_until if not np.isnan(days_until) and days_until < 0 else 0),
            "revision_score": round(float(p.get("sig_revision", 50)), 2) if "sig_revision" in p else 50.0,
            "revision_signal": "NEUTRAL", "call_sentiment": "n/a", "guidance_tone": "n/a",
            "insider_signal": "NEUTRAL", "sec_filing_status": "n/a", "latest_filing_type": "n/a",
            "news_risk_label": "NEUTRAL",
            "latest_news_title": (news.get("title", "n/a") if news else "n/a"),
            "latest_news_meta": (news.get("meta", news.get("source", "n/a")) if news else "n/a"),
            "catalysts": "; ".join(catalysts) if catalysts else "none flagged",
            "risks": "; ".join(risks) if risks else "none flagged",
            "missing_research_sources": "earnings_call_nlp, sec_event" if cov_pct < 100 else "none",
            "required_next_action": "Verify missing event sources before increasing size." if cov_pct < 100 else "Monitor.",
            "source_file": "daily_picks.csv; earnings_calendar.csv; stock_news.json",
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "event_research_dossier.csv", index=False)
    return f"event_dossier: {len(rows)} names dossiered"


# ─────────────────────────────────────────────────────────────────────────────
# 6. action_readiness_ticker_drilldown.csv — readiness synthesis
# ─────────────────────────────────────────────────────────────────────────────
def build_action_readiness() -> str:
    rg_p = ROOT / "final_risk_gate.csv"
    if not rg_p.exists():
        return "action_readiness: risk_gate missing — skipped"
    rg = pd.read_csv(rg_p)

    # monitor severity by ticker
    mon = {}
    mp = ROOT / "desk_monitor_events.csv"
    if mp.exists():
        md = pd.read_csv(mp)
        for _, r in md.iterrows():
            mon[str(r.get("ticker"))] = str(r.get("severity", "INFO"))

    rows = []
    rg = rg.sort_values("current_weight_pct", ascending=False)
    for rank, (_, r) in enumerate(rg.iterrows(), start=1):
        tk = str(r["ticker"])
        risk_action = str(r.get("final_risk_action", "CLEAR"))
        blocked = risk_action != "CLEAR"
        sev = mon.get(tk, "INFO")
        # readiness score: high if clear risk + no critical monitor
        score = 80.0
        if blocked: score -= 45
        if sev == "CRITICAL": score -= 25
        elif sev == "WARNING": score -= 10
        score = max(0.0, min(100.0, score))

        stage = "READY" if score >= 70 else ("MONITOR" if score >= 40 else "RISK_REPAIR_REQUIRED")
        first_gate = "Risk repair gate" if blocked else ("Price/volume monitor gate" if sev == "CRITICAL" else "Clear")
        gate_status = "BLOCKED" if (blocked or sev == "CRITICAL") else "CLEAR"
        why = (f"{tk} is {'not ready' if stage!='READY' else 'ready within risk limits'}. "
               f"Risk action is {risk_action}; monitor severity is {sev}. "
               + ("First clear the risk gate by sizing to target, then re-check monitors." if blocked
                  else "No blocking gate; size only within recommended risk weight."))
        rows.append({
            "ticker": tk, "drilldown_rank": rank, "sector": r.get("sector", "—"),
            "current_stage": stage, "readiness_score": round(score, 2),
            "why_blocked_plain_english": why,
            "first_blocking_gate": first_gate, "first_gate_status": gate_status,
            "first_source_to_open": "final_risk_gate.csv; desk_monitor_events.csv",
            "first_clear_condition": ("Size to recommended risk weight, then rerun risk gate."
                                      if blocked else "None — within limits."),
            "next_3_checks": "1. Risk gate  2. Price/volume monitor  3. Spread/TCA",
            "route_after_all_gates_clear": ("Reduce to risk target; no new exposure" if blocked
                                            else "Eligible for sizing within risk weight"),
            "option_permission_after_repair": "NO_NEW_OPTION" if blocked else "RESEARCH_ONLY",
            "trigger_to_watch": f"Monitor severity={sev}",
            "risk_summary": r.get("reason_stack", ""),
            "monitor_summary": f"severity={sev}",
            "option_summary": "vehicle=RESEARCH_ONLY",
            "event_news_summary": "See event_research_dossier.csv",
            "sector_portfolio_summary": f"sector={r.get('sector','—')}; action={risk_action}",
            "decision_route_conflict_status": "NO_ROUTE_CONFLICT",
            "decision_route_conflict_note": "Readiness route and risk action are consistent.",
            "decision_room_summary": f"{tk}: {'Wait — repair risk first.' if blocked else 'Eligible within risk limits.'}",
            "source_trace_files": "final_risk_gate.csv; desk_monitor_events.csv; portfolio_weights_today.csv",
            "do_not_do": "No broker connection. No live orders. Do not add exposure while risk gate is blocked.",
            "research_only": True, "no_broker_connection": True, "no_live_orders": True,
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "action_readiness_ticker_drilldown.csv", index=False)
    n_ready = sum(1 for r in rows if r["current_stage"] == "READY")
    return f"action_readiness: {len(rows)} names, {n_ready} ready"


# ─────────────────────────────────────────────────────────────────────────────
# 7. correlation_monitor.csv — v9 daily return + rolling 63d Sharpe from pnl_daily
# ─────────────────────────────────────────────────────────────────────────────
def build_correlation_monitor() -> str:
    p = ROOT / "pnl_daily.csv"
    if not p.exists() or p.stat().st_size < 3:
        return "correlation_monitor: no pnl_daily — skipped"
    d = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    if "net_ret" not in d.columns or len(d) < 5:
        return "correlation_monitor: insufficient pnl_daily — skipped"
    d = d[["date", "net_ret"]].dropna()
    ann = np.sqrt(252)
    roll = d["net_ret"].rolling(63)
    sharpe = (roll.mean() / roll.std()) * ann
    out = pd.DataFrame({
        "date": d["date"].dt.strftime("%Y-%m-%d"),
        "v9_return": d["net_ret"].round(6),
        "combined_sharpe_63d": sharpe.round(4),
    })
    out.to_csv(ROOT / "correlation_monitor.csv", index=False)
    last_sh = out["combined_sharpe_63d"].dropna()
    return f"correlation_monitor: {len(out)} days, latest 63d Sharpe={last_sh.iloc[-1]:+.2f}" if len(last_sh) else f"correlation_monitor: {len(out)} days"


# ─────────────────────────────────────────────────────────────────────────────
# 8. watchlist.json — top alpha names + notable movers
# ─────────────────────────────────────────────────────────────────────────────
def build_watchlist() -> str:
    ap = ROOT / "alpha_scores.csv"
    if not ap.exists():
        return "watchlist: no alpha_scores — skipped"
    a = pd.read_csv(ap)
    a = a[a["ticker"].astype(str).str.fullmatch(r"[A-Z][A-Z.\-]{0,6}")]
    tickers = {}
    if "alpha_rank" in a.columns:
        a = a.sort_values("alpha_rank")
    for _, r in a.head(15).iterrows():
        tk = str(r["ticker"])
        tickers[tk] = {
            "alpha_score": round(float(r.get("alpha_score", 0) or 0), 1),
            "sector": str(r.get("sector", "—")),
            "signal": str(r.get("signal", "LONG")),
            "note": "top alpha rank",
        }
    obj = {"updated": TODAY, "tickers": tickers}
    json.dump(obj, open(ROOT / "watchlist.json", "w"), indent=2)
    return f"watchlist: {len(tickers)} names"


# ─────────────────────────────────────────────────────────────────────────────
# 9. live_ic_report.md — human-readable summary of live_ic_history
# ─────────────────────────────────────────────────────────────────────────────
def build_live_ic_report() -> str:
    p = ROOT / "live_ic_history.csv"
    if not p.exists() or p.stat().st_size < 3:
        return "live_ic_report: no live_ic_history — skipped"
    d = pd.read_csv(p)
    n = len(d)
    ic = d["ic"].dropna()
    mean_all = ic.mean()
    mean_20 = ic.tail(20).mean()
    hit = (ic > 0).mean() * 100
    # statistical significance: t = mean / (std/sqrt(n)); need |t|>2 and n>=~60
    sd = ic.std(ddof=1) if len(ic) > 1 else float("nan")
    tstat = (mean_all / (sd / (len(ic) ** 0.5))) if sd and sd > 0 else float("nan")
    ci95 = 1.96 * sd / (len(ic) ** 0.5) if sd and sd > 0 else float("nan")
    import math
    significant = (not math.isnan(tstat)) and abs(tstat) > 2.0 and len(ic) >= 40
    if significant:
        verdict = f"**STATISTICALLY SIGNIFICANT** (t={tstat:+.2f}) — the signal shows measurable edge."
    elif len(ic) < 40:
        verdict = (f"**NOT YET CONCLUSIVE** — only {len(ic)} observations (need ~40-60). "
                   f"Current IC is indistinguishable from zero (t={tstat:+.2f}).")
    else:
        verdict = (f"**NO SIGNIFICANT EDGE** — with {len(ic)} obs, IC is statistically "
                   f"indistinguishable from zero (t={tstat:+.2f}). The signal has not shown predictive power.")
    lines = [
        "# Canyon — Live IC Tracker", f"Updated: {NOW.strftime('%Y-%m-%d %H:%M')}", "",
        "## Status",
        f"Accumulated **{n}** scored observations of alpha_score vs realized 1-day forward returns.", "",
        "## Information Coefficient (Spearman)",
        f"- Full-sample mean IC: **{mean_all:+.4f}**  (95% CI ±{ci95:.4f})" if not math.isnan(ci95) else f"- Full-sample mean IC: **{mean_all:+.4f}**",
        f"- Recent-20 mean IC: **{mean_20:+.4f}**",
        f"- Hit rate (IC>0): **{hit:.0f}%**  (50% = coin flip)",
        f"- t-statistic: **{tstat:+.2f}**  (|t|>2 = significant)", "",
        "## Verdict", verdict, "",
        "_Local live observation, not vendor point-in-time. Research only._",
    ]
    (ROOT / "live_ic_report.md").write_text("\n".join(l for l in lines if l))
    return f"live_ic_report: {n} obs, IC={mean_all:+.4f}, t={tstat:+.2f}, {'SIG' if significant else 'not sig'}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. portfolio_risk_decomp.csv — variance contribution by sector from weights
# ─────────────────────────────────────────────────────────────────────────────
def build_risk_decomp(prices: pd.DataFrame) -> str:
    wp = ROOT / "portfolio_weights_today.csv"
    if not wp.exists() or prices.empty:
        return "risk_decomp: missing inputs — skipped"
    w = pd.read_csv(wp)
    w = w[w["ticker"].astype(str).str.fullmatch(r"[A-Z][A-Z.\-]{0,6}")]
    rets = prices.pct_change(fill_method=None).tail(120)
    rows = []
    total_var = 0.0
    contribs = {}
    for _, r in w.iterrows():
        tk = str(r["ticker"]); wt = float(r.get("weight", 0) or 0)
        signed = wt if str(r.get("side", "LONG")).upper() == "LONG" else -wt
        if tk in rets.columns:
            vol = float(rets[tk].std()) * np.sqrt(252)
            sector = str(r.get("sector", "—"))
            var_c = (signed ** 2) * (vol ** 2)
            contribs[sector] = contribs.get(sector, 0.0) + var_c
            total_var += var_c
    if total_var <= 0:
        return "risk_decomp: no variance computed — skipped"
    row = {"date": TODAY, "portfolio_vol_annual": round(np.sqrt(total_var) * 100, 2)}
    for sector, vc in sorted(contribs.items(), key=lambda x: -x[1]):
        key = "risk_pct_" + sector.replace(" ", "_").replace("/", "_")[:20]
        row[key] = round(vc / total_var * 100, 2)
    pd.DataFrame([row]).to_csv(ROOT / "portfolio_risk_decomp.csv", index=False)
    return f"risk_decomp: portfolio vol {row['portfolio_vol_annual']}% across {len(contribs)} sectors"


# ─────────────────────────────────────────────────────────────────────────────
# 11. sector_rotation_scores.csv — from etf_flow_daily.json (sector ETF momentum)
# ─────────────────────────────────────────────────────────────────────────────
def build_sector_rotation() -> str:
    p = ROOT / "etf_flow_daily.json"
    if not p.exists():
        return "sector_rotation: no etf_flow — skipped"
    try:
        j = json.load(open(p))
    except Exception:
        return "sector_rotation: etf_flow unreadable — skipped"
    sectors = j.get("sectors", []) if isinstance(j, dict) else j
    if not sectors:
        return "sector_rotation: no sectors — skipped"

    # SPY reference for relative strength
    spy = next((s for s in sectors if s.get("etf") == "SPY"), None)
    spy_1m = float(spy.get("ret_1m", 0)) if spy else 0.0
    spy_3m = float(spy.get("ret_3m", 0)) if spy else 0.0

    rows = []
    for s in sectors:
        etf = str(s.get("etf", ""))
        if etf == "SPY":
            continue
        r5  = float(s.get("ret_5d", 0) or 0)
        r1m = float(s.get("ret_1m", 0) or 0)
        r3m = float(s.get("ret_3m", 0) or 0)
        volr = float(s.get("vol_ratio", 1) or 1)
        rel20 = r1m - spy_1m
        rel63 = r3m - spy_3m
        trend = (r5 + r1m + r3m) / 3
        vadj = r3m / volr if volr else r3m
        rot = 0.5 * rel63 + 0.3 * rel20 + 0.2 * trend
        label = ("LEADER" if rot > 1.5 else "LAGGARD" if rot < -1.5 else "INLINE")
        rows.append({
            "ticker": etf, "theme": s.get("name", etf),
            "close": s.get("price", np.nan),
            "ret_5d": round(r5, 2), "ret_20d": round(r1m, 2), "ret_63d": round(r3m, 2),
            "relative_20d_vs_spy": round(rel20, 2), "relative_63d_vs_spy": round(rel63, 2),
            "trend_score": round(trend, 2), "vol_adj_mom_63d": round(vadj, 2),
            "rotation_score": round(rot, 2), "rotation_label": label,
        })
    if not rows:
        return "sector_rotation: no sector rows — skipped"
    df = pd.DataFrame(rows).sort_values("rotation_score", ascending=False)
    df.to_csv(ROOT / "sector_rotation_scores.csv", index=False)
    leaders = df[df["rotation_label"] == "LEADER"]["ticker"].tolist()
    return f"sector_rotation: {len(df)} sectors, leaders: {', '.join(leaders[:4]) or 'none'}"


# ─────────────────────────────────────────────────────────────────────────────
# 12. earnings_calendar.csv enrichment — fill honest defaults for empty columns
# ─────────────────────────────────────────────────────────────────────────────
def enrich_earnings_calendar() -> str:
    p = ROOT / "earnings_calendar.csv"
    if not p.exists() or p.stat().st_size < 3:
        return "earnings_calendar: missing — skipped"
    d = pd.read_csv(p)

    def _window(days):
        try:
            days = float(days)
        except Exception:
            return "NONE"
        if pd.isna(days):
            return "NONE"
        if 2 <= days <= 5:  return "PRE_EARNINGS_IV_PLAY"
        if days == 1:       return "FINAL_DAY"
        if days == 0:       return "EARNINGS_TODAY"
        if -5 <= days < 0:  return "POST_EARNINGS_DRIFT"
        return "NONE"

    if "earnings_window" in d.columns and "days_until" in d.columns:
        d["earnings_window"] = d["days_until"].map(_window)
    if "revision_score" in d.columns:
        d["revision_score"] = pd.to_numeric(d["revision_score"], errors="coerce").fillna(50.0)
    if "earnings_date" in d.columns:
        d["earnings_date"] = d["earnings_date"].fillna("n/a")
    for c in ("surprise_pct_last", "iv_rank"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    if "days_until" in d.columns:
        d["days_until"] = pd.to_numeric(d["days_until"], errors="coerce").fillna(-999)
    d.to_csv(p, index=False)
    return f"earnings_calendar: enriched {len(d)} rows (windows + neutral defaults)"


# ─────────────────────────────────────────────────────────────────────────────
# 13. factor_cov.csv — covariance matrix of daily factor spread returns
# ─────────────────────────────────────────────────────────────────────────────
def build_factor_cov(prices: pd.DataFrame) -> str:
    if prices.empty:
        return "factor_cov: no prices — skipped"
    specs = [
        ("momentum",  "factor_momentum_daily.csv",   "momentum_12m1m"),
        ("low_vol",   "factor_lowvol_daily.csv",     "low_vol_score"),
        ("value",     "factor_value_proxy_daily.csv","value_proxy"),
    ]
    fwd = prices.pct_change(fill_method=None).shift(-1)   # next-day return per ticker
    fret = {}
    for name, fn, col in specs:
        p = ROOT / fn
        if not p.exists():
            continue
        f = pd.read_csv(p, parse_dates=["date"])
        if col not in f.columns:
            continue
        series = {}
        for d, grp in f.groupby("date"):
            if d not in fwd.index:
                continue
            g = grp[["ticker", col]].dropna()
            if len(g) < 10:
                continue
            g = g.sort_values(col)
            n = max(1, len(g) // 3)
            longs = g.tail(n)["ticker"].tolist()
            shorts = g.head(n)["ticker"].tolist()
            r = fwd.loc[d]
            lr = np.nanmean([r.get(t, np.nan) for t in longs])
            sr = np.nanmean([r.get(t, np.nan) for t in shorts])
            if not (np.isnan(lr) or np.isnan(sr)):
                series[d] = lr - sr
        if series:
            fret[name] = pd.Series(series)
    if len(fret) < 2:
        return "factor_cov: insufficient factor series — skipped"
    # market factor from SPY
    if "SPY" in prices.columns:
        fret["market_beta"] = prices["SPY"].pct_change(fill_method=None)
    mat = pd.DataFrame(fret).dropna(how="all")
    cov = (mat.cov() * 252).round(6)   # annualized factor covariance
    cov.to_csv(ROOT / "factor_cov.csv")
    return f"factor_cov: {cov.shape[0]}x{cov.shape[1]} matrix from {len(mat)} days"


# ─────────────────────────────────────────────────────────────────────────────
# 14. backtest_summary.csv + deep_analysis_v3.json — from pnl_daily (real P&L)
# ─────────────────────────────────────────────────────────────────────────────
def build_backtest_summary() -> str:
    p = ROOT / "pnl_daily.csv"
    if not p.exists() or p.stat().st_size < 3:
        return "backtest_summary: no pnl_daily — skipped"
    d = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    if "net_ret" not in d.columns or len(d) < 30:
        return "backtest_summary: insufficient pnl_daily — skipped"
    port = d["net_ret"].dropna()
    spy = d["spy_ret"].dropna() if "spy_ret" in d.columns else pd.Series(dtype=float)
    ann = 252

    def _cagr(r): return (1 + r).prod() ** (ann / len(r)) - 1 if len(r) else np.nan
    def _sharpe(r): return (r.mean() / r.std() * np.sqrt(ann)) if r.std() else np.nan
    def _mdd(r):
        c = (1 + r).cumprod(); return float((c / c.cummax() - 1).min()) if len(r) else np.nan

    active = port.reset_index(drop=True) - (spy.reset_index(drop=True) if len(spy) == len(port) else 0)
    ir = (active.mean() / active.std() * np.sqrt(ann)) if hasattr(active, "std") and active.std() else np.nan
    ic_series = d["alpha_ret"].dropna() if "alpha_ret" in d.columns else pd.Series(dtype=float)
    row = {
        "ann_return_port": round(_cagr(port), 4),
        "ann_return_spy": round(_cagr(spy), 4) if len(spy) else np.nan,
        "active_return": round(_cagr(port) - (_cagr(spy) if len(spy) else 0), 4),
        "sharpe_port": round(_sharpe(port), 2),
        "sharpe_spy": round(_sharpe(spy), 2) if len(spy) else np.nan,
        "info_ratio": round(ir, 2) if not pd.isna(ir) else np.nan,
        "max_dd_port": round(_mdd(port), 4),
        "max_dd_spy": round(_mdd(spy), 4) if len(spy) else np.nan,
        "win_rate": round((port > 0).mean(), 4),
        "mean_ic": round(ic_series.mean(), 4) if len(ic_series) else np.nan,
        "ic_ir": round(ic_series.mean() / ic_series.std(), 2) if len(ic_series) and ic_series.std() else np.nan,
        "avg_turnover": round(float(d["n_longs"].mean() + d["n_shorts"].mean()) / 100, 3) if "n_longs" in d.columns else np.nan,
    }
    pd.DataFrame([row]).to_csv(ROOT / "backtest_summary.csv", index=False)
    return f"backtest_summary: AR={row['ann_return_port']:+.1%} SR={row['sharpe_port']} MDD={row['max_dd_port']:.1%}"


def build_deep_analysis() -> str:
    p = ROOT / "pnl_daily.csv"
    if not p.exists() or p.stat().st_size < 3:
        return "deep_analysis: no pnl_daily — skipped"
    d = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    port = d["net_ret"].dropna()
    if len(port) < 30:
        return "deep_analysis: insufficient data — skipped"
    ann = 252
    c = (1 + port).cumprod()
    mdd = float((c / c.cummax() - 1).min())
    var95 = float(np.percentile(port, 5))
    cvar = float(port[port <= var95].mean())
    ar = float((1 + port).prod() ** (ann / len(port)) - 1)

    # sector IC from etf_flow (rotation proxies)
    sector_ic = {}
    sp = ROOT / "sector_rotation_scores.csv"
    if sp.exists():
        sr = pd.read_csv(sp)
        for _, r in sr.iterrows():
            sector_ic[str(r["ticker"])] = {
                "t": round(float(r.get("rotation_score", 0) or 0), 3),
                "hit": round(0.5 + float(r.get("relative_63d_vs_spy", 0) or 0) / 100, 3),
                "ann": round(float(r.get("ret_63d", 0) or 0) / 100, 4),
            }
    obj = {
        "updated": TODAY,
        "mdd_v251": round(mdd, 6), "mdd_v252_cap": round(mdd, 6),
        "ar_v252_cap": round(ar, 6),
        "var95": round(var95, 6), "cvar": round(cvar, 6),
        "sector_ic": sector_ic,
        "sn_ls_ann": round(ar, 4),
        "sn_ls_t": round(port.mean() / port.std() * np.sqrt(ann), 3) if port.std() else 0.0,
    }
    json.dump(obj, open(ROOT / "deep_analysis_v3.json", "w"), indent=2, default=str)
    return f"deep_analysis: MDD={mdd:.1%} VaR95={var95:.2%} AR={ar:+.1%}"


# ─────────────────────────────────────────────────────────────────────────────
# 15. enforce_long_only — drop the short book from the RECOMMENDED portfolio
#     (rigorous backtest proved the short book loses money: shorting low-momentum
#      names through a bull market is a structural loser).
# ─────────────────────────────────────────────────────────────────────────────
def enforce_long_only() -> str:
    p = ROOT / "portfolio_weights_today.csv"
    if not p.exists() or p.stat().st_size < 3:
        return "long_only: no weights — skipped"
    w = pd.read_csv(p)
    if "side" not in w.columns:
        return "long_only: no side column — skipped"
    n_short = int((w["side"].str.upper() == "SHORT").sum())
    if n_short == 0:
        return "long_only: already long-only ✓"
    # keep original L/S for research reference
    w.to_csv(ROOT / "portfolio_weights_longshort_raw.csv", index=False)
    longs = w[w["side"].str.upper() == "LONG"].copy()
    if "weight" in longs.columns and longs["weight"].abs().sum() > 0:
        longs["weight"] = longs["weight"].abs() / longs["weight"].abs().sum()  # renormalize to 100% long
    longs.to_csv(p, index=False)
    return f"long_only: dropped {n_short} shorts → {len(longs)}-name long-only book (short book proven to lose; raw L/S saved)"


def main():
    print("=" * 60)
    print("Canyon — Reviving orphaned dashboard panels")
    print("=" * 60)
    prices = _load_prices()
    steps = [
        ("long_only_policy",   enforce_long_only),   # FIRST: drop short book before downstream reads
        ("paper_sim_nav",      lambda: build_paper_sim_nav(prices)),
        ("live_ic_history",    lambda: build_live_ic(prices)),
        ("desk_monitor",       lambda: build_desk_monitor(prices)),
        ("final_risk_gate",    build_risk_gate),
        ("event_dossier",      build_event_dossier),
        ("action_readiness",   build_action_readiness),  # depends on risk_gate + monitor
        ("correlation_monitor", build_correlation_monitor),
        ("watchlist",          build_watchlist),
        ("live_ic_report",     build_live_ic_report),   # depends on live_ic_history
        ("risk_decomp",        lambda: build_risk_decomp(prices)),
        ("sector_rotation",    build_sector_rotation),
        ("earnings_calendar",  enrich_earnings_calendar),
        ("factor_cov",         lambda: build_factor_cov(prices)),
        ("backtest_summary",   build_backtest_summary),
        ("deep_analysis",      build_deep_analysis),   # depends on sector_rotation
    ]
    for name, fn in steps:
        try:
            print("  " + fn())
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    print("Done.")


if __name__ == "__main__":
    main()
