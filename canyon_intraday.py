#!/usr/bin/env python3
"""
canyon_intraday.py — 日内感知层(盘中牛熊 + 盘中新闻 + 盘中入场时机)
====================================================================
用免费日内数据(yfinance 5/15分钟K线 + 分钟级新闻)。三件事:
  1. 盘中牛熊判读: QQQ 日内 vs VWAP/短均线 + 当日涨跌 + VIX → 进攻/中性/避险
  2. 盘中新闻轮询: 给集中清单标的扫近8小时新闻(分钟级时间戳), 抓刚发生的事件
  3. 盘中入场时机: 每只集中标的日内 vs VWAP/日内区间 → 可入场/等回调/避免

诚实边界: 免费日内只有近7-60天历史 → 能实时监控/择时, 但无法做深度回测验证。
是"合理且免费的盘中工具", 非"验证过必赚"。运行时联网, 收盘后数据为最后一个交易日。
输出: intraday_signals.json
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent


def _dl(ticker, period, interval):
    import yfinance as yf
    d = yf.download(ticker, period=period, interval=interval, progress=False,
                    auto_adjust=True, threads=False)
    if d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    return d


def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return float((tp * df["Volume"]).cumsum().iloc[-1] / max(df["Volume"].cumsum().iloc[-1], 1))


def intraday_regime():
    """盘中牛熊: QQQ 日内结构 + VIX。返回 dict。"""
    q = _dl("QQQ", "5d", "15m")
    if q.empty:
        return {"regime": "数据缺失", "detail": ""}
    today = q.index[-1].date()
    qd = q[q.index.date == today]
    if len(qd) < 3:
        qd = q.tail(26)
    px = float(qd["Close"].iloc[-1])
    vw = vwap(qd)
    ma20 = float(q["Close"].tail(20).mean())
    prev_close = float(q["Close"].iloc[-len(qd) - 1]) if len(q) > len(qd) else float(qd["Open"].iloc[0])
    day_chg = px / prev_close - 1
    # VIX
    vix = _dl("^VIX", "5d", "15m")
    vix_now = float(vix["Close"].iloc[-1]) if not vix.empty else np.nan
    # 判读
    score = 0
    reasons = []
    if px > vw: score += 1; reasons.append("站上日内VWAP")
    else: score -= 1; reasons.append("跌破日内VWAP")
    if px > ma20: score += 1; reasons.append("站上20根均线")
    else: score -= 1; reasons.append("跌破20根均线")
    if day_chg > 0.003: score += 1; reasons.append(f"当日+{day_chg:.1%}")
    elif day_chg < -0.003: score -= 1; reasons.append(f"当日{day_chg:.1%}")
    if not np.isnan(vix_now):
        if vix_now > 25: score -= 1; reasons.append(f"VIX高({vix_now:.0f})恐慌")
        elif vix_now < 15: score += 1; reasons.append(f"VIX低({vix_now:.0f})平静")
    regime = "进攻" if score >= 2 else "避险" if score <= -2 else "中性"
    return {"regime": regime, "score": score, "qqq_px": round(px, 2), "vwap": round(vw, 2),
            "day_chg_%": round(day_chg * 100, 2), "vix": round(vix_now, 1) if not np.isnan(vix_now) else None,
            "reasons": reasons, "asof": str(q.index[-1])}


def intraday_news(tickers, hours=8):
    """盘中新闻: 扫近 hours 小时的新闻(分钟级时间戳)。"""
    import yfinance as yf
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    hits = []
    for tk in tickers:
        try:
            for it in (yf.Ticker(tk).news or [])[:8]:
                c = it.get("content", it)
                ts = c.get("pubDate") or it.get("providerPublishTime")
                title = c.get("title") or it.get("title", "")
                dt = None
                if isinstance(ts, str):
                    try:
                        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    except Exception:
                        dt = None
                elif isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, timezone.utc)
                if dt and dt >= cutoff and title:
                    hits.append({"ticker": tk, "time": dt.strftime("%m-%d %H:%M"),
                                 "mins_ago": int((now - dt).total_seconds() / 60), "title": title[:80]})
        except Exception:
            continue
    hits.sort(key=lambda x: x["mins_ago"])
    return hits[:15]


def intraday_entry(tickers):
    """盘中入场时机: 每只 vs 日内VWAP/区间 → 可入场/等回调/避免。"""
    rows = []
    for tk in tickers:
        d = _dl(tk, "5d", "15m")
        if d.empty or len(d) < 5:
            continue
        today = d.index[-1].date()
        dd = d[d.index.date == today]
        if len(dd) < 3:
            dd = d.tail(26)
        px = float(dd["Close"].iloc[-1])
        vw = vwap(dd)
        hi = float(dd["High"].max()); lo = float(dd["Low"].min())
        rng_pos = (px - lo) / max(hi - lo, 1e-9)          # 0=日内低, 1=日内高
        if px > vw and rng_pos < 0.7:
            sig = "可入场(站VWAP未追高)"
        elif px > vw and rng_pos >= 0.7:
            sig = "偏强·等回调VWAP"
        elif px < vw and rng_pos > 0.3:
            sig = "弱·观望"
        else:
            sig = "日内破位·避免"
        rows.append({"ticker": tk, "px": round(px, 2), "vwap": round(vw, 2),
                     "日内位置": round(rng_pos, 2), "信号": sig})
    return rows


def run():
    conc = ROOT / "concentrated_portfolio.csv"
    tickers = []
    if conc.exists():
        tickers = pd.read_csv(conc)["ticker"].astype(str).tolist()[:10]
    reg = intraday_regime()
    news = intraday_news(tickers) if tickers else []
    entry = intraday_entry(tickers) if tickers else []
    out = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "regime": reg, "news": news, "entry": entry, "watchlist": tickers}
    json.dump(out, open(ROOT / "intraday_signals.json", "w"), ensure_ascii=False, indent=2)
    return out


def main():
    print("=" * 62)
    print("日内感知层 — 盘中牛熊 + 新闻 + 入场时机")
    print("=" * 62)
    r = run()
    reg = r["regime"]
    print(f"\n  【1·盘中牛熊】{reg.get('regime')} (分{reg.get('score')})  "
          f"QQQ {reg.get('qqq_px')} vs VWAP {reg.get('vwap')} · 当日{reg.get('day_chg_%')}% · VIX {reg.get('vix')}")
    print(f"     {' / '.join(reg.get('reasons', []))}")
    print(f"\n  【2·盘中新闻】近8小时 {len(r['news'])} 条:")
    for n in r["news"][:8]:
        print(f"     {n['time']}({n['mins_ago']}分前) {n['ticker']:5} {n['title']}")
    print(f"\n  【3·盘中入场时机】集中清单 {len(r['entry'])} 只:")
    for e in r["entry"]:
        print(f"     {e['ticker']:5} {e['px']:>8} vs VWAP{e['vwap']:>8} 位置{e['日内位置']}  → {e['信号']}")
    print("\n  → intraday_signals.json")
    print("  诚实: 免费日内仅近7-60天, 可实时监控/择时, 无法深度回测验证。收盘后=最后交易日数据。")


if __name__ == "__main__":
    main()
