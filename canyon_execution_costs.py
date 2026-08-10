#!/usr/bin/env python3
"""
canyon_execution_costs.py — 执行成本建模 (事件交易 TCA)
=======================================================
把仓位计划变成"扣掉真实摩擦后还剩多少"。用行业标准平方根冲击模型, 按订单相对 ADV
的参与度算冲击, 加半价差, 往返(进+出)成本, 再按持有窗口年化, 判断事件 edge 能否覆盖摩擦。

成本构成 (单边):
  半价差   = spread_bps / 2                                   (立即成本)
  市场冲击 = η · σ_daily · sqrt(订单$ / ADV$) · 1e4  (bps)     (平方根冲击律, Almgren)
  滑点/佣金= slippage_bps + 固定佣金                            (execution_cost_estimates)
往返成本 = 2 × 单边   (进场 + 退场)
年化拖累 = 往返成本 / 持有年数   (持有越短, 摩擦年化越狠 → 高换手警报)

净 edge 判断: 事件级机会的经验目标收益(按事件类型持有窗口) 减 往返成本 → 是否还值得做。

输入: position_plan_event.csv + execution_cost_estimates.csv + event_candidates.csv(hold_window)
输出: execution_cost_plan.csv + execution_cost_summary.json
环境变量: CANYON_NAV (组合规模, 默认 1,000,000 美元) —— 订单越大冲击越高
"""
from __future__ import annotations
import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
NAV = float(os.environ.get("CANYON_NAV", "1000000"))   # 组合规模(美元)
ETA = 1.0                                              # 冲击系数(Almgren ~1)
COMMISSION_BPS = 1.0                                   # 固定佣金/费用(bps 单边)


def load(name):
    p = ROOT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def hold_years(hold_window: str) -> float:
    """把 '3-12个月' / '2-3个月' / '1-3个月(少数6)' 解析成中值年数。"""
    s = str(hold_window)
    nums = [float(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return 0.5
    if "个月" in s or "月" in s:
        mid = np.mean(nums[:2]) if len(nums) >= 2 else nums[0]
        return max(mid / 12.0, 1 / 12.0)
    return max(np.mean(nums[:2]) / 12.0, 1 / 12.0)


# 事件类型经验目标收益(持有窗口内, 手册量级的合理捕获目标)
EVENT_TARGET = {
    "行业爆发型": 0.35, "商品供需错配型": 0.22, "第二春重估型": 0.28,
    "战争/地缘冲击型": 0.20, "自然灾变型": 0.18, "企业重大事故型": 0.20,
}


def run():
    plan = load("position_plan_event.csv")
    if plan.empty:
        print("缺 position_plan_event.csv"); return {}
    cost = load("execution_cost_estimates.csv")
    cand = load("event_candidates.csv")
    dvol = load("sp500_price_history_deep.csv")

    cmap = {str(r["ticker"]): r for _, r in cost.iterrows()} if not cost.empty else {}
    hold_map = {str(r["ticker"]): r.get("hold_window", "3个月") for _, r in cand.iterrows()} if not cand.empty else {}
    # 每只日波动(年化→日)
    volmap = {}
    if not dvol.empty:
        dvol = dvol.set_index(dvol.columns[0]) if "date" in str(dvol.columns[0]).lower() else pd.read_csv(ROOT / "sp500_price_history_deep.csv", index_col=0)
    px = pd.read_csv(ROOT / "sp500_price_history_deep.csv", index_col=0, parse_dates=True) if (ROOT / "sp500_price_history_deep.csv").exists() else pd.DataFrame()

    rows = []
    for _, p in plan.iterrows():
        tk = str(p["ticker"])
        w = float(p["weight_pct"]) / 100.0
        order = w * NAV                                        # 订单美元
        c = cmap.get(tk)
        adv_m = float(c["adv_dollar_m"]) if c is not None else 300.0   # ADV(百万美元)
        adv = adv_m * 1e6
        spread_bps = float(c["spread_bps"]) if c is not None else 25.0
        base_slip = float(c["slippage_bps"]) if (c is not None and "slippage_bps" in c) else 3.0
        # 日波动
        sig_d = 0.02
        if tk in px.columns:
            s = px[tk].dropna()
            if len(s) > 40:
                sig_d = float(s.pct_change().dropna().tail(63).std())
        # 平方根冲击 (bps)
        participation = order / max(adv, 1e5)
        impact_bps = ETA * sig_d * np.sqrt(participation) * 1e4
        one_way_bps = spread_bps / 2 + impact_bps + base_slip + COMMISSION_BPS
        rt_bps = 2 * one_way_bps                               # 往返
        rt_pct = rt_bps / 1e4
        rt_usd = rt_pct * order
        hy = hold_years(hold_map.get(tk, "3个月"))
        ann_drag = rt_pct / hy
        et = str(p.get("event_type", ""))
        target = EVENT_TARGET.get(et, 0.25)
        net_edge = target - rt_pct                            # 目标收益 - 往返成本
        rows.append({
            "ticker": tk, "weight_pct": round(w * 100, 1), "order_usd": round(order),
            "adv_$m": round(adv_m, 1), "participation_%": round(participation * 100, 3),
            "spread_half_bps": round(spread_bps / 2, 1), "impact_bps": round(impact_bps, 1),
            "roundtrip_bps": round(rt_bps, 1), "roundtrip_usd": round(rt_usd),
            "hold_years": round(hy, 2), "ann_drag_%": round(ann_drag * 100, 2),
            "event_target_%": round(target * 100, 1), "net_edge_%": round(net_edge * 100, 1),
            "liquidity": str(c["liquidity_tier"]) if c is not None else "—",
        })
    df = pd.DataFrame(rows).sort_values("roundtrip_bps", ascending=False).reset_index(drop=True)
    df.to_csv(ROOT / "execution_cost_plan.csv", index=False)

    tot_order = df["order_usd"].sum()
    tot_cost = df["roundtrip_usd"].sum()
    summary = {
        "nav_usd": NAV,
        "total_deployed_usd": int(tot_order),
        "total_roundtrip_cost_usd": int(tot_cost),
        "blended_roundtrip_bps": round(tot_cost / max(tot_order, 1) * 1e4, 1),
        "avg_participation_%": round(float(df["participation_%"].mean()), 3),
        "worst_cost_name": df.iloc[0]["ticker"] if len(df) else "—",
        "worst_cost_bps": float(df.iloc[0]["roundtrip_bps"]) if len(df) else 0,
        "positions_edge_survives": int((df["net_edge_%"] > 0).sum()),
        "positions_total": len(df),
        "verdict": "",
    }
    bb = summary["blended_roundtrip_bps"]
    if bb < 40:
        summary["verdict"] = f"摩擦低({bb:.0f}bps往返) — 组合流动性好, 成本不构成障碍"
    elif bb < 90:
        summary["verdict"] = f"摩擦中等({bb:.0f}bps) — 可接受, 但避免频繁换手"
    else:
        summary["verdict"] = f"摩擦偏高({bb:.0f}bps) — 订单相对流动性偏大, 建议分批或降规模"
    json.dump(summary, open(ROOT / "execution_cost_summary.json", "w"), ensure_ascii=False, indent=2)
    return {"df": df, "summary": summary}


def main():
    print("=" * 66)
    print(f"执行成本建模 — 事件交易 TCA (组合规模 ${NAV:,.0f})")
    print("=" * 66)
    r = run()
    if not r:
        return
    s = r["summary"]; df = r["df"]
    print(f"  部署 ${s['total_deployed_usd']:,} · 往返摩擦 ${s['total_roundtrip_cost_usd']:,} "
          f"({s['blended_roundtrip_bps']:.0f}bps)")
    print(f"  平均参与度 {s['avg_participation_%']:.3f}% ADV · 最贵 {s['worst_cost_name']}({s['worst_cost_bps']:.0f}bps)")
    print(f"  净edge为正 {s['positions_edge_survives']}/{s['positions_total']} 个仓位")
    print(f"\n  {'标的':<7}{'仓位':>6}{'订单$':>10}{'参与%':>8}{'冲击bp':>8}{'往返bp':>8}{'年化拖累':>9}{'净edge':>8}")
    for _, p in df.iterrows():
        print(f"  {p['ticker']:<7}{p['weight_pct']:>5.1f}%{p['order_usd']:>10,.0f}"
              f"{p['participation_%']:>7.3f}%{p['impact_bps']:>8.1f}{p['roundtrip_bps']:>8.1f}"
              f"{p['ann_drag_%']:>8.2f}%{p['net_edge_%']:>7.1f}%")
    print(f"\n  判词: {s['verdict']}")
    print("  → execution_cost_plan.csv · execution_cost_summary.json")


if __name__ == "__main__":
    main()
