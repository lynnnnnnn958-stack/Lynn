#!/usr/bin/env python3
"""
canyon_insider_paper.py — 内部人买入策略的独立纸面实盘验证
==========================================================
把扫描器的 watchlist 变成一个 **隔离的纸面账本**, 用 Alpaca 真实价格入场/出场,
跟踪实盘表现 vs 回测预期 —— 这是从"回测证实"到"实盘证实"的最后一关。

隔离设计: 单独的 insider_paper_book.json, 不碰你主策略的 alpaca_book_state.json,
所以这块 edge 的实盘归因是干净的。

规则 (与验证过的回测一致):
  - 入场: insider_scan_today.csv 里活跃(21交易日窗口内)且未持有的信号 → 建仓
  - 持仓: 21 交易日 (≈30 日历日) 后出场
  - 等权, 每仓 SLOT_USD; 用 Alpaca 最新成交价记账
  - 每日跑一次, 增量更新账本

模式:
  - 默认: 账本模拟 (Alpaca 真价, 不下单) —— 稳, 足够验 IC
  - INSIDER_PAPER_LIVE=1: 额外真的下 Alpaca 纸面市价单 (需盘中 + 标的可交易)

输出: insider_paper_book.json (持仓+已平仓) + insider_paper_summary.json + report
"""
from __future__ import annotations
import json, os
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent


def _load_env():
    """Zero-dependency .env loader — the repo exports .env via shell, but this makes
    the script self-sufficient when run directly."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
BOOK = ROOT / "insider_paper_book.json"
SUMMARY = ROOT / "insider_paper_summary.json"
WATCH = ROOT / "insider_scan_today.csv"
SLOT_USD = 10_000              # 每个信号等权名义金额
HOLD_CAL_DAYS = 14            # ≈10 交易日 (验证过的最优持仓期)
MAX_POSITIONS = 15


def _load_book():
    if BOOK.exists():
        try:
            return json.loads(BOOK.read_text())
        except Exception:
            pass
    return {"open": {}, "closed": [], "nav_start": 0.0, "created": datetime.now().isoformat()}


def _prices(tickers):
    """Alpaca 最新成交价; 失败则回退到 smallcap_price_history 最新收盘。"""
    px = {}
    tickers = [t for t in tickers if t]
    if not tickers:
        return px
    try:
        from step_alpaca_execution import get_client, get_latest_prices
        client = get_client()
        if client is not None:
            px = get_latest_prices(client, tickers) or {}
    except Exception as e:
        print(f"  Alpaca price feed unavailable ({str(e)[:40]}) — using local close")
    missing = [t for t in tickers if t not in px or not px[t]]
    if missing:
        p = ROOT / "smallcap_price_history.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0).tail(1)
                last = df.iloc[-1]
                for t in missing:
                    if t in last and pd.notna(last[t]):
                        px[t] = float(last[t])
            except Exception:
                pass
    return px


def _maybe_live_order(ticker, qty, side):
    if os.environ.get("INSIDER_PAPER_LIVE") != "1":
        return None
    try:
        from step_alpaca_execution import get_client, get_latest_prices
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = get_client()
        req = MarketOrderRequest(symbol=ticker, qty=round(qty, 4),
                                 side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY)
        o = client.submit_order(req)
        return getattr(o, "id", None)
    except Exception as e:
        print(f"  live order {side} {ticker} failed: {str(e)[:50]}")
        return None


def run():
    book = _load_book()
    today = pd.Timestamp.today().normalize()
    watch = pd.read_csv(WATCH) if WATCH.exists() and WATCH.stat().st_size > 20 else pd.DataFrame()

    open_pos = book["open"]
    # 需要报价的标的 = 当前持仓 + 今日新信号
    want = set(open_pos.keys()) | (set(watch["ticker"].astype(str)) if not watch.empty else set())
    px = _prices(sorted(want))

    # ---- 出场: 持仓满 HOLD_CAL_DAYS ----
    for tk in list(open_pos.keys()):
        pos = open_pos[tk]
        held_days = (today - pd.Timestamp(pos["entry_date"])).days
        if held_days >= HOLD_CAL_DAYS:
            exit_px = px.get(tk)
            if exit_px is None:
                continue                       # 没报价, 下次再平
            ret = exit_px / pos["entry_price"] - 1
            oid = _maybe_live_order(tk, pos["qty"], "SELL")
            book["closed"].append({**pos, "ticker": tk, "exit_date": today.strftime("%Y-%m-%d"),
                                    "exit_price": exit_px, "return": round(ret, 4),
                                    "held_days": held_days, "exit_order": oid})
            del open_pos[tk]

    # ---- 入场: 今日活跃且未持有的信号 ----
    entered = 0
    if not watch.empty:
        for _, r in watch.iterrows():
            tk = str(r["ticker"])
            if tk in open_pos or len(open_pos) >= MAX_POSITIONS:
                continue
            p0 = px.get(tk)
            if not p0 or p0 <= 0:
                continue
            qty = SLOT_USD / p0
            oid = _maybe_live_order(tk, qty, "BUY")
            open_pos[tk] = {"entry_date": today.strftime("%Y-%m-%d"), "entry_price": p0,
                            "qty": round(qty, 4), "notional": SLOT_USD,
                            "cluster": bool(r.get("cluster")), "large": bool(r.get("large")),
                            "insiders": int(r.get("insiders", 0)), "entry_order": oid}
            entered += 1

    book["open"] = open_pos
    BOOK.write_text(json.dumps(book, indent=2))
    return _summarize(book, px, entered)


def _summarize(book, px, entered):
    closed = pd.DataFrame(book["closed"])
    openp = book["open"]
    # 未实现
    unrl = []
    for tk, pos in openp.items():
        cur = px.get(tk, pos["entry_price"])
        unrl.append(cur / pos["entry_price"] - 1)
    s = {
        "as_of": datetime.now().isoformat(),
        "open_positions": len(openp),
        "entered_today": entered,
        "closed_trades": int(len(closed)),
    }
    if len(closed):
        r = closed["return"]
        s.update({
            "win_rate": round(float((r > 0).mean()), 3),
            "avg_return_per_trade": round(float(r.mean()), 4),
            "median_return": round(float(r.median()), 4),
            "best": round(float(r.max()), 4), "worst": round(float(r.min()), 4),
            "total_realized_pnl_usd": round(float((r * SLOT_USD).sum()), 2),
            "cluster_avg_return": round(float(closed[closed.get("cluster", False) == True]["return"].mean()), 4)
                if "cluster" in closed.columns and (closed["cluster"] == True).any() else None,
        })
    if unrl:
        s["open_unrealized_avg"] = round(float(pd.Series(unrl).mean()), 4)
    SUMMARY.write_text(json.dumps(s, indent=2))
    return s


def main():
    print("=" * 64)
    print("Insider paper validation — isolated ledger, Alpaca live prices")
    print("=" * 64)
    live = os.environ.get("INSIDER_PAPER_LIVE") == "1"
    print(f"  mode: {'LIVE paper orders' if live else 'ledger-only (Alpaca prices, no orders)'}")
    s = run()
    print(f"\n  open {s['open_positions']} · entered today {s['entered_today']} · "
          f"closed {s['closed_trades']}")
    if s.get("closed_trades"):
        print(f"  realized: win {s.get('win_rate')} · avg/trade {s.get('avg_return_per_trade'):+.2%} · "
              f"PnL ${s.get('total_realized_pnl_usd'):,.0f}")
        if s.get("cluster_avg_return") is not None:
            print(f"  cluster trades avg: {s['cluster_avg_return']:+.2%}")
    if s.get("open_unrealized_avg") is not None:
        print(f"  open unrealized avg: {s['open_unrealized_avg']:+.2%}")
    print(f"\n  → {BOOK.name} + {SUMMARY.name}")
    print("  (validates live IC vs backtest; run daily via run_daily.py)")


if __name__ == "__main__":
    main()
