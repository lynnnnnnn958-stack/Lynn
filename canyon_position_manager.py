#!/usr/bin/env python3
"""
canyon_position_manager.py — 持仓管理与退出引擎(止损 + 抽本金 + 移动止损)
==========================================================================
把两条铁律硬编码进系统, 让"暴雷=小伤"而不是重伤:
  ① 止损减仓: 单只从入场跌破 -STOP% → 清仓(不让一只拖垮组合, 防暴雷)
  ② 抽本金只留利润: 涨够 +TAKE% → 卖掉约"本金那部分", 剩下用"赢来的钱"博(house money)
  ③ 移动止损: 已盈利后从最高点回撤 -TRAIL% → 减/清, 锁住利润
对齐手册第10层分型退出。

维护一个纸面持仓账本 position_ledger.csv(首次运行按当前价建仓, 之后逐日跟踪)。
输出: position_ledger.csv(更新) + position_actions.csv(今日动作)
诚实: 纸面模拟, 无券商连接; 入场价=首次进入集中清单当日价(无法回填历史)。
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

STOP_LOSS = 0.15     # 从入场跌 15% → 止损清仓(防暴雷)
TRAIL = 0.15         # 盈利后从最高点回撤 15% → 移动止损减仓
TAKE = 0.50          # 涨 50% → 抽回本金, 只留利润
BIG_WIN = 1.00       # 翻倍 → 再减一档锁利


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            d = pd.read_csv(p, index_col=0, parse_dates=True)
            return d.iloc[-1]
    return pd.Series(dtype=float)


def load_ledger():
    p = ROOT / "position_ledger.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "peak_price",
                                 "principal_recovered", "status"])


def run():
    conc = ROOT / "concentrated_portfolio.csv"
    if not conc.exists():
        print("缺 concentrated_portfolio.csv"); return {}
    target = pd.read_csv(conc)
    px = prices()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    led = load_ledger()
    led_map = {str(r["ticker"]): dict(r) for _, r in led.iterrows()}

    target_tks = [str(t) for t in target["ticker"]]
    actions, new_rows = [], []

    for tk in target_tks:
        cur = float(px.get(tk, np.nan))
        if np.isnan(cur):
            continue
        if tk not in led_map:                         # 新进清单 → 建仓
            led_map[tk] = {"ticker": tk, "entry_date": today, "entry_price": cur,
                           "peak_price": cur, "principal_recovered": 0, "status": "持有"}
            actions.append({"ticker": tk, "action": "新建仓", "detail": f"入场价 {cur:.2f}",
                            "ret_%": 0.0})
            continue
        pos = led_map[tk]
        entry = float(pos["entry_price"]); peak = max(float(pos["peak_price"]), cur)
        pos["peak_price"] = peak
        ret = cur / entry - 1
        dd_peak = cur / peak - 1
        recovered = int(pos.get("principal_recovered", 0))
        act, detail = "持有", f"入场{entry:.2f} 现价{cur:.2f}"
        # ① 止损(最高优先, 防暴雷)
        if ret <= -STOP_LOSS:
            act, detail = "🔴止损清仓", f"跌破入场-{STOP_LOSS:.0%}({ret:+.1%}), 认错离场防暴雷"
            pos["status"] = "已止损"
        # ③ 移动止损(已盈利后回撤)
        elif ret > 0.10 and dd_peak <= -TRAIL:
            act, detail = "🟡移动止损·减半", f"从最高{peak:.2f}回撤{dd_peak:+.1%}, 锁利减仓"
        # ② 抽本金
        elif ret >= TAKE and not recovered:
            sell_frac = 1 / (1 + ret)                 # 卖这部分=收回本金
            act = "🟢抽回本金·只留利润"
            detail = f"涨{ret:+.0%}, 卖{sell_frac:.0%}收回本金, 剩{1-sell_frac:.0%}用赢来的钱博"
            pos["principal_recovered"] = 1
        # 翻倍再减
        elif ret >= BIG_WIN and recovered:
            act, detail = "🟢大赢·再减一档", f"涨{ret:+.0%}(已抽本金), 再落袋部分利润"
        elif recovered:
            act, detail = "持有(纯利润仓)", f"本金已收回, 现价{cur:.2f} 涨{ret:+.0%}, house money"
        actions.append({"ticker": tk, "action": act, "detail": detail, "ret_%": round(ret * 100, 1)})

    # 已不在清单、也没止损的 → 标记退出观察
    for tk, pos in led_map.items():
        if tk not in target_tks and pos.get("status") == "持有":
            cur = float(px.get(tk, np.nan))
            ret = (cur / float(pos["entry_price"]) - 1) if not np.isnan(cur) else 0
            actions.append({"ticker": tk, "action": "移出清单·了结", "detail": "已不在集中清单, 兑现",
                            "ret_%": round(ret * 100, 1)})
            pos["status"] = "已了结"

    # ── 资本再循环: 抽出/止损/了结腾出的钱 → 滚入下一批最高信念新标的 ──
    VALID = {"行业爆发型", "第二春重估型", "企业重大事故型"}
    wmap = {str(r["ticker"]): float(r.get("weight_pct", 0)) for _, r in target.iterrows()}
    freed, freed_src = 0.0, []
    for a in actions:
        tk = a["ticker"]; w = wmap.get(tk, 8.0)
        if "止损清仓" in a["action"] or "了结" in a["action"]:
            freed += w; freed_src.append(f"{tk}({'止损' if '止损' in a['action'] else '了结'} {w:.0f}%)")
        elif "抽回本金" in a["action"]:
            r = a["ret_%"] / 100
            sf = 1 / (1 + r) if r > -1 else 0
            freed += w * sf; freed_src.append(f"{tk}(抽本金 {w*sf:.0f}%)")
    redeploy = []
    if freed > 0.5:
        ecp = ROOT / "event_candidates.csv"
        if ecp.exists():
            ec = pd.read_csv(ecp)
            held_now = {t for t, p in led_map.items() if p.get("status") == "持有"}
            pool = ec[ec["event_type"].isin(VALID) & ~ec["ticker"].astype(str).isin(held_now)]
            pool = pool.sort_values("FinalEventScore", ascending=False).head(5)
            n = max(len(pool), 1)
            for _, r in pool.iterrows():
                redeploy.append({"ticker": str(r["ticker"]), "event_type": r.get("event_type", ""),
                                 "FES": round(float(r["FinalEventScore"]), 1),
                                 "target_weight_%": round(freed / n, 1)})

    ledger = pd.DataFrame(list(led_map.values()))
    ledger.to_csv(ROOT / "position_ledger.csv", index=False)
    adf = pd.DataFrame(actions)
    if not adf.empty:
        adf.to_csv(ROOT / "position_actions.csv", index=False)
    rdf = pd.DataFrame(redeploy)
    rdf.to_csv(ROOT / "position_redeploy.csv", index=False)
    json.dump({"freed_pct": round(freed, 1), "freed_from": freed_src,
               "redeploy_to": redeploy},
              open(ROOT / "position_redeploy.json", "w"), ensure_ascii=False, indent=2)
    return {"actions": adf, "ledger": ledger, "freed": freed, "freed_src": freed_src, "redeploy": redeploy}


def main():
    print("=" * 62)
    print("持仓管理与退出引擎 — 止损 + 抽本金 + 移动止损")
    print("=" * 62)
    print(f"  规则: 止损-{STOP_LOSS:.0%} · 抽本金+{TAKE:.0%} · 移动止损回撤-{TRAIL:.0%}")
    r = run()
    if not r:
        return
    adf = r["actions"]
    live = [a for a in adf.to_dict("records") if a["action"] != "持有"]
    print(f"\n  今日需动作 {len(live)} / {len(adf)} 只:")
    for a in adf.to_dict("records"):
        flag = "  " if a["action"] in ("持有", "持有(纯利润仓)") else "→ "
        print(f"  {flag}{a['ticker']:6} {a['ret_%']:+6.1f}%  {a['action']:<16} {a['detail']}")
    freed = r.get("freed", 0)
    if freed > 0.5:
        print(f"\n  ♻ 资本再循环: 腾出 {freed:.0f}% 本金 (来自 {', '.join(r['freed_src'])})")
        print(f"     → 建议滚入下一批高信念新标的:")
        for d in r["redeploy"]:
            print(f"       {d['ticker']:6} {d['event_type']:12} FES{d['FES']:.0f}  配 {d['target_weight_%']:.1f}%")
    else:
        print(f"\n  ♻ 资本再循环: 今日无腾出资金(无止损/抽本金)。有了会自动滚入下一批。")
    print("\n  → position_ledger.csv · position_actions.csv · position_redeploy.csv")
    print("  诚实: 纸面模拟; 入场价=首次进清单当日价。真用需连券商/手动记录成交价。")


if __name__ == "__main__":
    main()
