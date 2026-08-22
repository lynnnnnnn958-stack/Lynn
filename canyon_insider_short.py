#!/usr/bin/env python3
"""
canyon_insider_short.py — 内部人卖出做空扫描器(组合的空头腿)
============================================================
验证(78只/6542笔真Form4卖出, 剔除反转后仍显著): 内部人 *开放市场卖出* 预示
小盘下跌。做空毛收益 +22.7%/t=5.3; 扣 10% 借券后 +12.7%/t=3.0 仍显著; 但借券
25% 就死 —— 所以 **能不能做全看借券**。

本模块 = 空头腿:
  1. 扫近 LOOKBACK 天的内部人卖出(INSIDER_SIDE=SELL 复用买入扫描器的抓取管线)
  2. 聚合成做空候选(近期卖得多/集中的名字)
  3. **借券闸门**: 用 Alpaca shortable + easy_to_borrow 过滤 —— 难借的直接跳过
     (难借 = 借券费高 = 回测里会死的那些, 现实防线)
  4. 输出 insider_short_today.csv

诚实边界: 做空亏损无上限、有逼空风险; 仅研究信号, 每单前仍需人工确认借券费。
用法: python3 canyon_insider_short.py [--quick N]
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

# 复用买入扫描器 / 抓取器的机件
from canyon_insider_scanner import (universe, cikmap, _recent_form4, merge_cache as _mc,
                                     _trading_days_ago, HOLD_TDAYS, CLUSTER_WIN, LARGE_USD)
from step_smallcap_form4 import parse_form4
import time
from step_smallcap_form4 import DELAY

ROOT = Path(__file__).parent
SIGNALS = ROOT / "insider_short_signals.csv"
TODAY_OUT = ROOT / "insider_short_today.csv"
LOOKBACK_DAYS = 20            # 卖出信号短周期(10交易日持仓 + 缓冲)


def _load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _borrow_gate(tickers):
    """{ticker: (shortable, easy_to_borrow)} via Alpaca. 难借=借券费高=跳过。"""
    _load_env()
    out = {}
    try:
        from alpaca.trading.client import TradingClient
        c = TradingClient(os.environ["ALPACA_KEY_ID"], os.environ["ALPACA_KEY_SECRET"], paper=True)
        for tk in tickers:
            try:
                a = c.get_asset(tk)
                out[tk] = (bool(a.shortable), bool(a.easy_to_borrow))
            except Exception:
                out[tk] = (False, False)
    except Exception as e:
        print(f"  borrow gate unavailable ({str(e)[:40]}) — leaving flags blank")
    return out


SCAN_BUDGET_S = int(os.environ.get("SHORT_BUDGET_S", "1200"))   # 时间预算: 到点收工


def scan_sells(tickers):
    os.environ["INSIDER_SIDE"] = "SELL"          # 让 parse_form4 抓卖出
    since = (pd.Timestamp.today() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    cm = cikmap(tickers)
    rows = []
    t0 = time.time()
    for k, tk in enumerate(tickers, 1):
        if time.time() - t0 > SCAN_BUDGET_S:
            print(f"  budget {SCAN_BUDGET_S}s reached at {k}/{len(tickers)} — building from partial", flush=True)
            break
        cik = cm.get(tk)
        if cik is None:
            continue
        try:
            for acc, prim in _recent_form4(cik, since):
                for b in parse_form4(cik, acc.replace("-", ""), prim):
                    b["ticker"] = tk
                    rows.append(b)
                time.sleep(DELAY)
        except Exception as e:
            print(f"  {tk}: skip ({str(e)[:30]})", flush=True)
        if k % 50 == 0:
            print(f"  scanned {k}/{len(tickers)} · {len(rows)} recent sells", flush=True)
    return pd.DataFrame(rows)


def build_short_watchlist(alls: pd.DataFrame) -> pd.DataFrame:
    if alls.empty:
        return alls
    alls = alls.copy()
    alls["dt"] = pd.to_datetime(alls["date"], errors="coerce")
    active = alls[alls["dt"] >= _trading_days_ago(HOLD_TDAYS)]
    if active.empty:
        return active
    out = []
    for tk, g in active.groupby("ticker"):
        g = g.sort_values("dt")
        sellers = g["owner"].nunique() if "owner" in g.columns else len(g)
        val = pd.to_numeric(g.get("value", 0), errors="coerce").fillna(0).sum()
        cxo = int(pd.to_numeric(g.get("role_cxo", 0), errors="coerce").fillna(0).max())
        latest = g["dt"].max()
        cl = False
        for _, r in g.iterrows():
            w = g[(g["dt"] > r["dt"] - pd.Timedelta(days=CLUSTER_WIN)) & (g["dt"] <= r["dt"])]
            if (w["owner"].nunique() if "owner" in w.columns else len(w)) >= 2:
                cl = True; break
        held_td = int((pd.Timestamp.today() - latest).days / 1.4)
        out.append({"ticker": tk, "latest_sell": latest.strftime("%Y-%m-%d"),
                    "sellers": int(sellers), "cluster": cl, "cxo_involved": bool(cxo),
                    "total_usd": int(val), "large": val >= LARGE_USD,
                    "approx_days_left": max(HOLD_TDAYS - held_td, 0)})
    w = pd.DataFrame(out)
    # 借券闸门
    gate = _borrow_gate(list(w["ticker"]))
    w["shortable"] = [gate.get(tk, (None, None))[0] for tk in w["ticker"]]
    w["easy_to_borrow"] = [gate.get(tk, (None, None))[1] for tk in w["ticker"]]
    w["tradable_short"] = w["shortable"].fillna(False) & w["easy_to_borrow"].fillna(False)
    # 强度: 卖得多/大额/CXO 参与 越强(注: cluster 卖出在验证中偏弱, 权重低)
    w["strength"] = (w["large"].astype(int) * 2 + (w["sellers"] >= 3).astype(int) * 2
                     + w["cxo_involved"].astype(int) + w["cluster"].astype(int))
    # 只把"可做空"的排前面
    w = w.sort_values(["tradable_short", "strength", "total_usd"],
                      ascending=[False, False, False]).reset_index(drop=True)
    w.to_csv(TODAY_OUT, index=False)
    return w


def main():
    tickers = universe()
    if "--quick" in sys.argv:
        tickers = tickers[:int(sys.argv[sys.argv.index("--quick") + 1])]
    print(f"Insider SHORT scan · {len(tickers)} small-caps · sells in last {LOOKBACK_DAYS}d", flush=True)
    new = scan_sells(tickers)
    print(f"  {len(new)} recent open-market sells", flush=True)
    # 复用买入扫描器的缓存合并(同schema)
    globals_signals = SIGNALS
    if globals_signals.exists():
        old = pd.read_csv(globals_signals)
        allb = pd.concat([old, new], ignore_index=True)
    else:
        allb = new
    if not allb.empty:
        allb["date"] = pd.to_datetime(allb["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        allb = allb.dropna(subset=["date"]).drop_duplicates(
            subset=[c for c in ["ticker", "date", "owner", "shares"] if c in allb.columns])
        allb.to_csv(globals_signals, index=False)
    w = build_short_watchlist(allb)
    if w.empty:
        print("\n  No active insider-sell short signals right now."); return
    tradable = w[w["tradable_short"] == True]
    print(f"\n=== INSIDER-SELL SHORT WATCHLIST ({len(w)} names, {len(tradable)} borrowable) ===")
    for _, r in w.head(25).iterrows():
        gate = "borrowable" if r["tradable_short"] else "HARD-TO-BORROW→skip"
        tags = "/".join([t for t, f in [("CLUSTER", r["cluster"]), ("LARGE", r["large"]),
                                        ("CEO/CFO", r["cxo_involved"])] if f])
        print(f"  {str(r['ticker']):6} {r['latest_sell']}  {r['sellers']} sellers  "
              f"${int(r['total_usd']):>10,}  [{gate}]  {tags}")
    print(f"\n  → {TODAY_OUT.name} · only 'borrowable' names are actually shortable "
          f"(hard-to-borrow = high fee = the backtest's death zone)")


if __name__ == "__main__":
    main()
