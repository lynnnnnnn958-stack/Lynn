#!/usr/bin/env python3
"""
canyon_insider_scanner.py — 每日小盘内部人买入扫描器 (validated edge → daily watchlist)
========================================================================================
把回测证实的 edge 变成每天能用的清单:扫 S&P600 小盘最近的 Form 4 *开放市场买入*
(code P / AD A), 按验证过的规则排出候选。

Validated backtest (58 deep tickers, 28yr, HAC t + Deflated Sharpe + costs + OOS):
  - 21天持仓, 市场中性 vs 小盘等权:
      all buys   t=3.0  DSR=0.97  +alpha  OOS+   ← 最实用 (持仓最分散)
      cluster    t=2.6  DSR=0.97
      large≥$100k t=2.4  DSR=0.88
  - 63天持仓 / 仅CEO-CFO 不显著 → 不用
边界: alpha 幅度实盘会缩水; 小盘容量小 (适合个人); 仅作研究信号, 非下单。

流程 (增量, 每天可跑):
  1. 每票拉 submissions.recent, 取近 LOOKBACK_DAYS 天的 Form 4, 解析 P/A 买入
  2. 合并进滚动缓存 insider_scan_signals.csv (按 ticker+date+owner 去重)
  3. 标"活跃"(买入在近 HOLD 交易日内 = 21天持仓时钟还在走)
  4. 每票聚合: 不同内部人数、总金额、is_cluster/is_large、进/出场日
  5. 排序输出 insider_scan_today.csv + 打印 watchlist

用法: python3 canyon_insider_scanner.py           # 全量 602 (~2-3min)
      python3 canyon_insider_scanner.py --quick 40 # 测试: 前 40 只
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

from step_smallcap_form4 import universe, cikmap, collect_all_form4, parse_form4, _get, DELAY
import json, time

ROOT = Path(__file__).parent
SIGNALS = ROOT / "insider_scan_signals.csv"
TODAY_OUT = ROOT / "insider_scan_today.csv"
LOOKBACK_DAYS = 35          # 只看近 35 天的新 Form 4 (覆盖 21 交易日窗口 + 缓冲)
HOLD_TDAYS = 21             # 验证过的持仓期 (交易日)
CLUSTER_WIN = 30            # cluster 判定窗口 (天)
LARGE_USD = 100_000


def _recent_form4(cik, since_iso):
    """只取 submissions.recent 里 filingDate >= since 的 Form 4 (acc, prim)。快。"""
    j = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").decode())
    time.sleep(DELAY)
    rec = j.get("filings", {}).get("recent", {})
    forms = rec.get("form", []); dates = rec.get("filingDate", [])
    accs = rec.get("accessionNumber", []); prims = rec.get("primaryDocument", [])
    out = []
    for i, f in enumerate(forms):
        if f == "4" and i < len(dates) and dates[i] >= since_iso:
            out.append((accs[i], prims[i]))
    return out


def scan(tickers):
    since = (pd.Timestamp.today() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    cm = cikmap(tickers)
    rows = []
    for k, tk in enumerate(tickers, 1):
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
            print(f"  scanned {k}/{len(tickers)} · {len(rows)} recent buys", flush=True)
    return pd.DataFrame(rows)


def merge_cache(new: pd.DataFrame) -> pd.DataFrame:
    if SIGNALS.exists():
        old = pd.read_csv(SIGNALS)
        allb = pd.concat([old, new], ignore_index=True)
    else:
        allb = new
    if allb.empty:
        return allb
    allb["date"] = pd.to_datetime(allb["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    allb = allb.dropna(subset=["date"])
    key = ["ticker", "date", "owner", "shares"]
    allb = allb.drop_duplicates(subset=[c for c in key if c in allb.columns])
    allb.to_csv(SIGNALS, index=False)
    return allb


def _trading_days_ago(n):
    """近 n 交易日的起始日历日期 (用小盘价格日历; 没有则按 n*1.4 日历日近似)。"""
    p = ROOT / "smallcap_price_history.csv"
    if p.exists():
        try:
            idx = pd.read_csv(p, usecols=[0]).iloc[:, 0]
            days = pd.to_datetime(idx, errors="coerce").dropna().sort_values()
            if len(days) > n:
                return days.iloc[-n]
        except Exception:
            pass
    return pd.Timestamp.today() - pd.Timedelta(days=int(n * 1.4))


def build_watchlist(allb: pd.DataFrame) -> pd.DataFrame:
    if allb.empty:
        return allb
    allb = allb.copy()
    allb["dt"] = pd.to_datetime(allb["date"], errors="coerce")
    cutoff = _trading_days_ago(HOLD_TDAYS)      # 仍在 21 交易日持仓窗口内 = 活跃
    active = allb[allb["dt"] >= cutoff]
    if active.empty:
        return active
    out = []
    for tk, g in active.groupby("ticker"):
        g = g.sort_values("dt")
        owners = g["owner"].nunique() if "owner" in g.columns else len(g)
        val = pd.to_numeric(g.get("value", 0), errors="coerce").fillna(0).sum()
        cxo = int(pd.to_numeric(g.get("role_cxo", 0), errors="coerce").fillna(0).max())
        latest = g["dt"].max()
        first = g["dt"].min()
        # cluster: 窗口内 ≥2 不同内部人
        cl = False
        for _, r in g.iterrows():
            w = g[(g["dt"] > r["dt"] - pd.Timedelta(days=CLUSTER_WIN)) & (g["dt"] <= r["dt"])]
            if (w["owner"].nunique() if "owner" in w.columns else len(w)) >= 2:
                cl = True; break
        held_td = int((pd.Timestamp.today() - latest).days / 1.4)   # 约略已持交易日
        out.append({
            "ticker": tk, "latest_buy": latest.strftime("%Y-%m-%d"),
            "insiders": int(owners), "cluster": cl, "cxo_involved": bool(cxo),
            "total_usd": int(val), "large": val >= LARGE_USD,
            "buys_in_window": len(g),
            "approx_days_held": max(held_td, 0),
            "approx_days_left": max(HOLD_TDAYS - held_td, 0),
        })
    w = pd.DataFrame(out)
    # 信号强度排序: cluster > large > 单人; 再按不同内部人数、金额、最新
    w["strength"] = (w["cluster"].astype(int) * 3 + w["large"].astype(int) * 2
                     + (w["insiders"] >= 2).astype(int) + w["cxo_involved"].astype(int))
    w = w.sort_values(["strength", "insiders", "total_usd", "latest_buy"],
                      ascending=[False, False, False, False]).reset_index(drop=True)
    w.to_csv(TODAY_OUT, index=False)
    return w


def main():
    tickers = universe()
    if "--quick" in sys.argv:
        n = int(sys.argv[sys.argv.index("--quick") + 1])
        tickers = tickers[:n]
    print(f"Insider scan · {len(tickers)} small-caps · Form 4 buys in last {LOOKBACK_DAYS}d", flush=True)
    new = scan(tickers)
    print(f"  {len(new)} recent open-market buys found", flush=True)
    allb = merge_cache(new)
    w = build_watchlist(allb)
    if w.empty:
        print("\n  No active insider-buy signals in the 21-trading-day window right now.")
        return
    print(f"\n=== ACTIVE INSIDER-BUY WATCHLIST ({len(w)} names, 21-day hold clock) ===")
    show = w.head(25)
    for _, r in show.iterrows():
        tag = []
        if r["cluster"]: tag.append("CLUSTER")
        if r["large"]: tag.append("LARGE")
        if r["cxo_involved"]: tag.append("CEO/CFO")
        tags = ("[" + "/".join(tag) + "]") if tag else ""
        print(f"  {r['ticker']:6} {r['latest_buy']}  {r['insiders']} insiders  "
              f"${r['total_usd']:>10,}  ~{r['approx_days_left']}d left  {tags}")
    print(f"\n  → {TODAY_OUT.name} ({len(w)} names) + {SIGNALS.name} (rolling cache)")


if __name__ == "__main__":
    main()
