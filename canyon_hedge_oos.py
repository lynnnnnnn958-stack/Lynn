#!/usr/bin/env python3
"""
canyon_hedge_oos.py — 择时对冲的样本外验证(16年 QQQ)
====================================================
那个"跌破50线才对冲/避险"的规则, 是在2年数据上拟合的。真检验: 放到16年历史、
尤其2022熊市, 它到底管不管用。若长期能护住回撤、夏普不输买入持有, 才算真信号;
若被反复假信号打脸、长期跑输, 那2年的48.8%就是过拟合。

测(全部 out-of-sample, 简单规则不回头看未来):
  · 买入持有 QQQ
  · 均线择时(50/100/200日): 站上均线满仓, 跌破转现金(=避险/减仓)
  · 200日多空: 站上做多, 跌破做空
分别看 16年 CAGR/夏普/最大回撤, 以及关键熊市(2022/2020/2018Q4)的表现。
诚实: 择时规则简单、无手续费; 但这是真样本外(规则不看未来)。
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent


def load_qqq():
    for f in ("qqq_price_long.csv", "qqq_price.csv"):
        p = ROOT / f
        if p.exists():
            d = pd.read_csv(p, index_col=0, parse_dates=True)
            return d[d.columns[0]].dropna()
    return pd.Series(dtype=float)


def metrics(nav):
    r = nav.pct_change().dropna()
    yrs = len(nav) / 252
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    dd = (nav / nav.cummax() - 1).min()
    return cagr, sharpe, dd


def timed(px, N, short=False):
    """站上N日线满仓; 跌破→现金(short=False)或做空(short=True)。次日执行, 无前视。"""
    ma = px.rolling(N).mean()
    sig = (px > ma).shift(1).fillna(False)          # 昨天的信号今天执行
    r = px.pct_change().fillna(0)
    pos = np.where(sig, 1.0, (-1.0 if short else 0.0))
    return (1 + r * pos).cumprod()


def sub(px, s, e):
    return px[(px.index >= s) & (px.index <= e)]


def main():
    px = load_qqq()
    if px.empty:
        print("缺 QQQ"); return
    print("=" * 68)
    print(f"择时对冲 样本外验证 — QQQ {px.index[0].date()} → {px.index[-1].date()}")
    print("=" * 68)
    bh = px / px.iloc[0]
    strats = {
        "买入持有 QQQ": bh,
        "50日择时(跌破转现金)": timed(px, 50),
        "100日择时": timed(px, 100),
        "200日择时": timed(px, 200),
        "200日多空(跌破做空)": timed(px, 200, short=True),
    }
    print(f"\n  {'策略':<22}{'年化':>8}{'夏普':>7}{'最大回撤':>10}")
    for name, nav in strats.items():
        c, s, d = metrics(nav)
        print(f"  {name:<22}{c*100:>7.1f}%{s:>7.2f}{d*100:>9.1f}%")

    print(f"\n  --- 关键熊市: 买入持有 vs 50日择时(看避险能不能护住) ---")
    bears = {"2022 科技熊市": ("2021-11-19", "2022-12-31"),
             "2020 COVID崩盘": ("2020-02-19", "2020-04-30"),
             "2018Q4 大跌": ("2018-09-30", "2018-12-31")}
    for label, (s, e) in bears.items():
        seg = sub(px, s, e)
        if len(seg) < 10:
            continue
        bh_ret = float(seg.iloc[-1] / seg.iloc[0] - 1)
        t = timed(px, 50); tseg = t[(t.index >= s) & (t.index <= e)]
        t_ret = float(tseg.iloc[-1] / tseg.iloc[0] - 1) if len(tseg) else float("nan")
        print(f"  {label:<16} 买入持有 {bh_ret*100:+6.1f}%  |  50日择时 {t_ret*100:+6.1f}%  "
              f"{'✅护住' if t_ret > bh_ret else '❌没护住'}")

    print("\n  诚实结论: 择时若在16年+熊市里夏普不输、回撤更小, 才是真信号;")
    print("  牛市里择时通常略降收益(假信号whipsaw), 换的是抗跌。看你要收益还是要睡得着。")


if __name__ == "__main__":
    main()
