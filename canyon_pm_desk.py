#!/usr/bin/env python3
"""
canyon_pm_desk.py — 投委会决策层(像基金经理一样,把全部板块整合成一个每日决定)
================================================================================
真正的 PM 不孤立看单个策略。他把 [验证过的 alpha 引擎] × [宏观 regime 决定敞口]
× [风控限额] 合成 **一个连贯的每日决定**: 今天进攻还是防守、上多少仓、净敞口、
具体买卖什么、预期表现如何。

整合的板块:
  · Alpha 引擎 = 内部人多空(唯一验证过的 edge): 多头抄底买入 + 空头集中卖出
    (Sharpe ~1.05, 纯 alpha) —— insider_sizing_today / insider_short_today / insider_ls_backtest
  · 宏观 regime = 决定 GROSS 敞口档位: HMM 牛熊 + 宏观前瞻熊概率 + VIX
    —— hmm_regime_daily / macro_regime_outlook / intraday_signals
  · 风控 = 右尾仓位引擎的限额(每只小注、按流动性封顶)

诚实立场(PM 的纪律): 只有内部人 edge 是验证过的 alpha; 宏观是"敞口旋钮"不是
alpha; 系统其余 20 个面板是 PM 的酌情参考, 不是机械输入。市场中性 → beta≈0,
但 regime 差时仍降 gross(右尾策略在危机里会放大)。

Output: pm_desk_decision.json + 打印决定
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SLEEVE = 100_000                 # 分配给这个策略的资金


def _regime():
    """三个 regime 输入 → 一个 risk posture + gross 敞口档位。"""
    bull = bear_prob = vix = None
    try:
        h = pd.read_csv(ROOT / "hmm_regime_daily.csv")
        bull = float(h.iloc[-1].get("prob_bull", 0.5))
    except Exception:
        pass
    try:
        m = json.loads((ROOT / "macro_regime_outlook.json").read_text())
        sig = m.get("signals", {})
        tot = sum(s.get("bear_score", 0) for s in sig.values())
        mx = sum(s.get("max_score", 0) for s in sig.values())
        bear_prob = (tot / mx) if mx else None
    except Exception:
        pass
    try:
        i = json.loads((ROOT / "intraday_signals.json").read_text())
        vix = float(i.get("regime", {}).get("vix") or 0) or None
    except Exception:
        pass
    # 综合评分 → gross 档位
    reasons = []
    gross = 1.0
    if bull is not None:
        reasons.append(f"HMM 牛概率 {bull:.0%}")
        if bull < 0.5:
            gross = min(gross, 0.5); reasons.append("HMM 转熊 → 降 gross")
    if bear_prob is not None:
        reasons.append(f"宏观熊概率 {bear_prob:.0%}")
        if bear_prob > 0.5:
            gross = min(gross, 0.4); reasons.append("宏观偏熊 → 降 gross")
        elif bear_prob > 0.3:
            gross = min(gross, 0.75)
    if vix is not None:
        reasons.append(f"VIX {vix:.0f}")
        if vix > 30:
            gross = min(gross, 0.4); reasons.append("VIX>30 恐慌 → 降 gross")
        elif vix > 22:
            gross = min(gross, 0.7)
    posture = ("AGGRESSIVE" if gross >= 0.9 else "NEUTRAL" if gross >= 0.6 else "DEFENSIVE")
    return {"posture": posture, "gross_pct": round(gross, 2),
            "bull_prob": bull, "macro_bear_prob": bear_prob, "vix": vix,
            "reasons": reasons}


def _book():
    """多头(已定注码) + 空头(可借 cluster) → 组合构成。"""
    longs = shorts = None
    try:
        s = pd.read_csv(ROOT / "insider_sizing_today.csv")
        longs = s[s["size_usd"] > 0] if "size_usd" in s.columns else s
    except Exception:
        longs = pd.DataFrame()
    try:
        sh = pd.read_csv(ROOT / "insider_short_today.csv")
        if "tradable_short" in sh.columns:
            sh = sh[sh["tradable_short"] == True]
        if "cluster" in sh.columns and sh["cluster"].any():
            sh = sh[sh["cluster"] == True]         # 只做集中卖出
        shorts = sh
    except Exception:
        shorts = pd.DataFrame()
    return longs, shorts


def _expected():
    try:
        m = json.loads((ROOT / "insider_ls_backtest.json").read_text())
        return m.get("combined_long_short", {})
    except Exception:
        return {}


# regime → 多空倾斜(净敞口方向)。市场中性是验证过的核心; 倾斜是"酌情叠加", 幅度小。
_TILT = {"AGGRESSIVE": 0.55, "NEUTRAL": 0.50, "DEFENSIVE": 0.42}   # long 权重
MAX_NAME_PCT = 0.08          # 单只上限
MAX_SECTOR_PCT = 0.35        # 单行业上限(有数据时)


def _risk_checks(longs, shorts, long_usd):
    """组合层风控: 名字重叠、集中度、单只上限。行业集中度待 SIC 缓存。"""
    lt = set(longs["ticker"].astype(str)) if longs is not None and len(longs) else set()
    st = set(shorts["ticker"].astype(str)) if shorts is not None and len(shorts) else set()
    checks, flags = {}, []
    overlap = lt & st
    checks["long_short_overlap"] = sorted(overlap)
    if overlap:
        flags.append(f"CONFLICT: {len(overlap)} name(s) both long & short")
    # 集中度: 前5多头占多头账本比例
    if longs is not None and "size_usd" in getattr(longs, "columns", []) and long_usd > 0:
        top5 = float(longs.nlargest(5, "size_usd")["size_usd"].sum())
        maxn = float(longs["size_usd"].max())
        checks["top5_long_pct"] = round(top5 / longs["size_usd"].sum(), 3)
        checks["max_name_pct"] = round(maxn / longs["size_usd"].sum(), 3)
        if maxn / longs["size_usd"].sum() > MAX_NAME_PCT * 1.5:
            flags.append(f"single name {maxn/longs['size_usd'].sum():.0%} > cap")
    # 行业集中度(有 SIC 缓存时)
    secp = ROOT / "smallcap_sectors.csv"
    if secp.exists() and longs is not None and len(longs) and "size_usd" in getattr(longs, "columns", []):
        try:
            sm = pd.read_csv(secp)
            smap = dict(zip(sm["ticker"].astype(str), sm["sector"].astype(str)))
            lg = longs.copy(); lg["sector"] = lg["ticker"].astype(str).map(smap).fillna("Unknown")
            by_sec = lg.groupby("sector")["size_usd"].sum() / lg["size_usd"].sum()
            top_sec = by_sec.idxmax(); top_pct = float(by_sec.max())
            checks["top_sector"] = f"{top_sec} {top_pct:.0%}"
            checks["sector_breakdown"] = {k: round(float(v), 3) for k, v in by_sec.sort_values(ascending=False).head(5).items()}
            if top_pct > MAX_SECTOR_PCT:
                flags.append(f"sector {top_sec} {top_pct:.0%} > {MAX_SECTOR_PCT:.0%} cap")
        except Exception:
            checks["top_sector"] = "n/a"
    else:
        checks["top_sector"] = "n/a — SIC cache pending"
    checks["breaches"] = flags
    checks["ok"] = len(flags) == 0
    return checks


def decide():
    reg = _regime()
    longs, shorts = _book()
    exp = _expected()
    # 冲突解决: 同时被买入和集中卖出的票 = 信号矛盾 → 两边都剔除, 回避
    conflicts = []
    if longs is not None and shorts is not None and len(longs) and len(shorts):
        lt = set(longs["ticker"].astype(str)); st = set(shorts["ticker"].astype(str))
        conflicts = sorted(lt & st)
        if conflicts:
            longs = longs[~longs["ticker"].astype(str).isin(conflicts)]
            shorts = shorts[~shorts["ticker"].astype(str).isin(conflicts)]
    gross = reg["gross_pct"]
    n_long = int(len(longs)) if longs is not None else 0
    n_short = int(len(shorts)) if shorts is not None else 0
    # regime 决定多空倾斜: 进攻略偏多、防守偏空(但主体仍中性)
    lw = _TILT.get(reg["posture"], 0.50)
    gross_usd_target = SLEEVE * gross
    long_usd = gross_usd_target * lw
    short_cap = n_short * 8000                          # 集中卖出稀少 → 空头容量有限
    short_usd = min(gross_usd_target * (1 - lw), short_cap)
    gross_usd = long_usd + short_usd
    net_usd = long_usd - short_usd
    risk = _risk_checks(longs, shorts, long_usd)
    return {
        "as_of": pd.Timestamp.now().isoformat(),
        "regime": reg,
        "decision": {
            "posture": reg["posture"],
            "gross_target_pct": gross,
            "long_tilt_pct": round(lw, 2),
            "sleeve_usd": SLEEVE,
            "gross_deployed_usd": round(gross_usd, 0),
            "cash_reserve_usd": round(SLEEVE - gross_usd, 0) if gross_usd < SLEEVE else 0,
            "long_names": min(n_long, 30), "long_usd": round(long_usd, 0),
            "short_names": min(n_short, 15), "short_usd": round(short_usd, 0),
            "net_exposure_usd": round(net_usd, 0),
            "net_pct_of_sleeve": round(net_usd / SLEEVE, 3),
            "market_neutral": bool(abs(net_usd) < 0.1 * SLEEVE),
        },
        "portfolio_risk": {**risk, "conflicts_resolved": conflicts},
        "expected_book": exp,
        "top_longs": (longs.head(5)["ticker"].tolist() if n_long else []),
        "cluster_shorts": (shorts.head(5)["ticker"].tolist() if n_short else []),
        "honesty": "Only the insider L/S is validated alpha. Macro = exposure dial. The L/S tilt is a "
                   "small discretionary regime overlay (not validated). Other panels inform discretion. "
                   "Paper-validating live; sector-concentration cap pending a SIC cache.",
    }


def main():
    print("=" * 70)
    print("PM DESK — integrated daily decision (alpha × regime × risk)")
    print("=" * 70)
    d = decide()
    json.dump(d, open(ROOT / "pm_desk_decision.json", "w"), indent=2, default=str)
    r = d["regime"]; dec = d["decision"]; e = d["expected_book"]
    print(f"\n  REGIME: {r['posture']}  (gross {r['gross_pct']:.0%})")
    print(f"    {' · '.join(r['reasons'])}")
    print(f"\n  DECISION:")
    print(f"    gross ${dec['gross_deployed_usd']:,.0f} of ${dec['sleeve_usd']:,} "
          f"({dec['gross_target_pct']:.0%}) · cash ${dec['cash_reserve_usd']:,.0f}")
    print(f"    LONG  {dec['long_names']} names  ${dec['long_usd']:,.0f}")
    print(f"    SHORT {dec['short_names']} cluster-sells  ${dec['short_usd']:,.0f}")
    _neu = "market-neutral" if dec['market_neutral'] else f"net long {dec['net_pct_of_sleeve']:+.0%}"
    print(f"    net exposure ${dec['net_exposure_usd']:,.0f} ({_neu}) · long-tilt {dec['long_tilt_pct']:.0%}")
    rk = d.get("portfolio_risk", {})
    conf = rk.get("conflicts_resolved", [])
    print(f"\n  RISK: {'✓ all clear' if rk.get('ok') else '⚠ ' + '; '.join(rk.get('breaches', []))}")
    if conf:
        print(f"    ⚑ 剔除 {len(conf)} 只信号冲突(既被买又被集中卖): {', '.join(conf)}")
    print(f"    top-5 长仓集中度 {rk.get('top5_long_pct','—')} · 单只上限 {rk.get('max_name_pct','—')}")
    if e:
        print(f"\n  EXPECTED (backtest): alpha {e.get('alpha_annual',0):+.1%}  "
              f"Sharpe {e.get('sharpe','—')}  MaxDD {e.get('max_dd',0):.1%}")
    print(f"\n  top longs: {', '.join(d['top_longs'])}")
    print(f"  cluster shorts: {', '.join(d['cluster_shorts']) or '(none today)'}")
    print(f"\n  → pm_desk_decision.json")


if __name__ == "__main__":
    main()
