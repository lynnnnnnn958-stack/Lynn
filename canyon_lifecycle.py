#!/usr/bin/env python3
"""
canyon_lifecycle.py — 第2层 生命周期与行为风格
===============================================
手册第2层: 每只标的的"生命周期阶段"和"行为风格"决定了它适合哪种打法、
该给多大波动容忍、以及第5层的"阶段匹配度"。全部从价格历史自动推断,标普500全体。

生命周期阶段 (lifecycle) — 由长期趋势斜率 + 距高点 + 动量持续性推断:
  萌芽期 (Emerging)  —— 长期底部起势, 刚突破, 高弹性高不确定
  成长期 (Growth)    —— 稳定上行趋势, 创新高, 趋势清晰   ← 事件爆发最佳载体
  成熟期 (Mature)    —— 高位震荡, 趋势走平, 波动收敛
  衰退期 (Decline)   —— 长期下行, 远离高点, 破位

行为风格 (style) — 由已实现波动 + 趋势拟合度(R²) + 回撤深度推断:
  趋势型 (Trend)     —— 高 R², 顺势而为, 适合利润发动机轨
  波动型 (Volatile)  —— 高波动低 R², 高赔率高风险, 需小仓位宽止损
  防御型 (Defensive) —— 低波动低回撤, 稳健, 适合核心储备

输出: lifecycle_style.csv (ticker, lifecycle, style, 波动容忍, 阶段匹配基准, trend_r2, ann_vol, dist_high, mom_12m)
其中"阶段匹配基准"喂回第5层的阶段匹配度; "波动容忍"喂回波动容忍匹配。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent


def universe():
    for f in ("alpha_scores.csv", "regime_ml_scores.csv"):
        p = ROOT / f
        if p.exists():
            d = pd.read_csv(p)
            if "ticker" in d.columns:
                return d["ticker"].astype(str).tolist()
    return []


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()


def classify_one(s: pd.Series):
    """s = 单只价格序列(已 dropna)。返回 (lifecycle, style, features)"""
    n = len(s)
    px = s.values.astype(float)
    ret = np.diff(np.log(px))
    ann_vol = float(np.std(ret[-252:]) * np.sqrt(252)) if len(ret) >= 60 else float(np.std(ret) * np.sqrt(252))
    # 12月动量 & 距高点
    look = min(252, n - 1)
    mom_12m = float(px[-1] / px[-look - 1] - 1) if n > look + 1 else 0.0
    hi_win = px[-min(504, n):]
    dist_high = float(px[-1] / np.max(hi_win) - 1)      # ≤0, 越接近0越靠近高点
    # 长期趋势拟合度 R² (log price vs time, 近2年)
    seg = np.log(px[-min(504, n):])
    t = np.arange(len(seg))
    if len(seg) > 30:
        b1, b0 = np.polyfit(t, seg, 1)
        fit = b0 + b1 * t
        ss_res = np.sum((seg - fit) ** 2); ss_tot = np.sum((seg - seg.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        slope = float(b1 * 252)                         # 年化对数斜率
    else:
        r2, slope = 0.0, 0.0

    # ---- 生命周期 ----
    if slope > 0.08 and dist_high > -0.12 and mom_12m > 0.05:
        lifecycle = "成长期"
    elif slope > 0.15 and mom_12m > 0.35 and dist_high > -0.05:
        lifecycle = "萌芽期"                             # 强突破加速
    elif slope < -0.05 or dist_high < -0.35:
        lifecycle = "衰退期"
    else:
        lifecycle = "成熟期"
    # 萌芽期需要高弹性佐证
    if lifecycle == "成长期" and mom_12m > 0.6 and ann_vol > 0.45:
        lifecycle = "萌芽期"

    # ---- 行为风格 ----
    if ann_vol >= 0.42:
        style = "波动型"
    elif r2 >= 0.55 and abs(slope) >= 0.06:
        style = "趋势型"
    elif ann_vol <= 0.25:
        style = "防御型"
    else:
        style = "趋势型" if r2 >= 0.4 else "波动型"

    return lifecycle, style, dict(trend_r2=round(r2, 2), ann_vol=round(ann_vol, 3),
                                  dist_high=round(dist_high, 3), mom_12m=round(mom_12m, 3),
                                  slope=round(slope, 3))


# 生命周期 → 阶段匹配基准 (0-1, 喂回第5层"阶段匹配度"): 成长期最适合承接事件
STAGE_MATCH = {"成长期": 1.0, "萌芽期": 0.85, "成熟期": 0.6, "衰退期": 0.35}
# 行为风格 → 波动容忍 (0-1, 喂回第5层"波动容忍匹配"): 趋势型容忍高, 波动型需谨慎
VOL_TOL = {"趋势型": 0.9, "防御型": 0.75, "波动型": 0.55}


def run():
    uni = universe()
    px = prices()
    if px.empty or not uni:
        print("缺价格历史或 universe"); return pd.DataFrame()
    rows = []
    for tk in uni:
        if tk not in px.columns:
            continue
        s = px[tk].dropna()
        if len(s) < 260:
            continue
        lc, st, f = classify_one(s)
        rows.append({"ticker": tk, "lifecycle": lc, "style": st,
                     "波动容忍": VOL_TOL[st], "阶段匹配基准": STAGE_MATCH[lc], **f})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(ROOT / "lifecycle_style.csv", index=False)
    return df


def main():
    print("=" * 56)
    print("第2层 生命周期与行为风格 — 标普500全体")
    print("=" * 56)
    df = run()
    if df.empty:
        print("  无输出"); return
    print(f"  分类 {len(df)} 只")
    print("  生命周期:", df["lifecycle"].value_counts().to_dict())
    print("  行为风格:", df["style"].value_counts().to_dict())
    print("\n  成长期+趋势型 (事件爆发最佳载体) 样例:")
    best = df[(df["lifecycle"] == "成长期") & (df["style"] == "趋势型")].sort_values("mom_12m", ascending=False)
    for _, r in best.head(8).iterrows():
        print(f"    {r.ticker:6} R²{r.trend_r2}  年化波动{r.ann_vol}  12月动量{r.mom_12m:+.0%}")
    print("\n  → lifecycle_style.csv (喂回第5层 阶段匹配度 + 波动容忍匹配)")


if __name__ == "__main__":
    main()
