#!/usr/bin/env python3
"""
canyon_event_validate.py — 事件打分验证引擎 (IC / 分层收益)
============================================================
华尔街判断一个信号有没有用, 看的是 IC(信息系数): 打分与"未来收益"的排序相关性。
这里用 16 年历史价格, 对事件打分中**可从价格历史重建的骨架**做严格的走查验证:

  可重建骨架(不含前瞻偏差, 每个历史时点只用当时可得的价格):
    L_proxy 动量强度   ← 12月动量
    M_proxy 错价空间   ← 距52周高的回撤(反弹空间)
    结构     ← 站上 ma50/ma200
    赔率     ← 回撤深度
  合成 proto-FES → 每月横截面排名 → 测未来 3 个月收益的 IC 与分层价差。

诚实边界: 这只验证价格骨架。真正的事件层(新闻催化 N/映射 P/前瞻确认 C)需要
历史 PIT 新闻与基本面才能回测, 免费数据没有 → 那部分只能靠 review_history 向前累积。
本引擎回答的是: "这套打分的价格地基, 有没有预测力?"

输出: event_validation.json + 控制台报告
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()


def proto_fes(px: pd.DataFrame, asof_idx: int) -> pd.Series:
    """在 asof_idx 这个时点, 只用历史价格算每只的 proto-FES(0-1 合成)。"""
    win = px.iloc[max(0, asof_idx - 252):asof_idx + 1]
    if len(win) < 200:
        return pd.Series(dtype=float)
    last = win.iloc[-1]
    hi = win.max()
    ma50 = win.iloc[-50:].mean()
    ma200 = win.mean()
    mom = last / win.iloc[0] - 1                    # 12月动量 → L
    dd = last / hi                                  # ≤1, 越小回撤越深
    L = (mom.clip(-0.5, 1.5) + 0.5) / 2.0           # 0-1
    M = (1 - dd).clip(0, 1)                          # 回撤空间 0-1
    struct = 0.5 * (last > ma50).astype(float) + 0.5 * (last > ma200).astype(float)
    odds = (1.2 - dd).clip(0, 1)
    score = 0.35 * L + 0.30 * M + 0.20 * struct + 0.15 * odds
    return score.dropna()


def forward_ret(px: pd.DataFrame, asof_idx: int, horizon: int) -> pd.Series:
    if asof_idx + horizon >= len(px):
        return pd.Series(dtype=float)
    p0 = px.iloc[asof_idx]
    p1 = px.iloc[asof_idx + horizon]
    return (p1 / p0 - 1).dropna()


def run(horizon: int = 63, step: int = 21):
    px = prices()
    if px.empty:
        print("缺价格历史"); return {}
    px = px.sort_index()
    n = len(px)
    ics, spreads, dates = [], [], []
    i = 252
    while i + horizon < n:
        s = proto_fes(px, i)
        fr = forward_ret(px, i, horizon)
        common = s.index.intersection(fr.index)
        if len(common) >= 50:
            sc = s.loc[common]; rr = fr.loc[common]
            ic = sc.rank().corr(rr.rank())          # Spearman IC
            if pd.notna(ic):
                ics.append(float(ic))
                # 分层价差: top20% - bottom20% 平均未来收益
                q = sc.quantile([0.2, 0.8])
                top = rr[sc >= q[0.8]].mean(); bot = rr[sc <= q[0.2]].mean()
                if pd.notna(top) and pd.notna(bot):
                    spreads.append(float(top - bot))
                dates.append(str(px.index[i].date()))
        i += step
    if not ics:
        print("样本不足"); return {}
    ic_arr = np.array(ics); sp_arr = np.array(spreads)
    ic_mean = float(ic_arr.mean()); ic_std = float(ic_arr.std())
    ic_ir = ic_mean / ic_std * np.sqrt(len(ic_arr)) if ic_std > 0 else 0.0  # IC t-stat
    hit = float((ic_arr > 0).mean())
    out = {
        "horizon_days": horizon, "periods": len(ics),
        "span": f"{dates[0]} → {dates[-1]}",
        "IC_mean": round(ic_mean, 4),
        "IC_t_stat": round(ic_ir, 2),
        "IC_hit_rate": round(hit, 3),
        "decile_spread_mean": round(float(sp_arr.mean()), 4) if len(sp_arr) else None,
        "decile_spread_ann": round(float(sp_arr.mean()) * (252 / horizon), 4) if len(sp_arr) else None,
        "verdict": "",
    }
    # 诚实判词 (华尔街标准: |IC|≥0.03 且 t≥2 才算有微弱但真实的信号)
    if abs(ic_mean) >= 0.05 and abs(ic_ir) >= 3:
        out["verdict"] = "价格骨架有明确预测力(IC强、显著)"
    elif abs(ic_mean) >= 0.03 and abs(ic_ir) >= 2:
        out["verdict"] = "价格骨架有微弱但统计显著的预测力(华尔街可用量级)"
    elif abs(ic_ir) >= 2:
        out["verdict"] = "预测力弱但方向显著; 价格地基能用, edge 需靠事件层贡献"
    else:
        out["verdict"] = "价格骨架单独无显著预测力 — edge 必须来自事件/催化/前瞻层(需向前验证)"
    json.dump(out, open(ROOT / "event_validation.json", "w"), ensure_ascii=False, indent=2)
    return out


def main():
    print("=" * 62)
    print("事件打分验证引擎 — 价格骨架 IC 走查 (16年)")
    print("=" * 62)
    r = run()
    if not r:
        return
    print(f"  验证区间: {r['span']} · {r['periods']} 个重叠期 · 未来{r['horizon_days']}天")
    print(f"  IC 均值:      {r['IC_mean']:+.4f}   (华尔街可用门槛 |IC|≥0.03)")
    print(f"  IC t 统计量:  {r['IC_t_stat']:+.2f}     (显著门槛 |t|≥2)")
    print(f"  IC 胜率:      {r['IC_hit_rate']:.0%}      (>50% 说明方向稳定)")
    if r.get("decile_spread_ann") is not None:
        print(f"  头尾20%年化价差: {r['decile_spread_ann']:+.1%}")
    print(f"\n  判词: {r['verdict']}")
    print("\n  → event_validation.json")
    print("  注: 这只验证价格骨架; 事件层(新闻催化/前瞻确认)的 edge 需靠 review_history 向前累积。")


if __name__ == "__main__":
    main()
