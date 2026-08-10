#!/usr/bin/env python3
"""
canyon_build_pool.py — 自动建股票池 (底库永远是标普500)
======================================================
不用人工给名单: 用标普500全体当底库,自动给每只分事件类型 + 预填 L/N/M/P/C,
写入 event_pool.csv。有新闻事件信号的用侦测结果覆盖,其余按行业给默认事件类型。
使用者只需微调关注的少数标的。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# 行业 → 默认事件类型 (无新闻信号时的初判; 事件本质是动态的, 这只是起点)
SECTOR_EVENT = {
    "Technology": "行业爆发型",
    "Communication Services": "行业爆发型",
    "Energy": "商品供需错配型",
    "Materials": "商品供需错配型",
    "Industrials": "行业爆发型",          # 电气化/CapEx; 军工名单单独归战争型
    "Health Care": "第二春重估型",
    "Financials": "第二春重估型",
    "Consumer Discretionary": "第二春重估型",
    "Consumer Staples": "第二春重估型",
    "Utilities": "行业爆发型",             # 数据中心用电
    "Real Estate": "第二春重估型",
}
# 军工/国防名单 → 战争/地缘冲击型
DEFENSE = {"LMT", "RTX", "NOC", "GD", "LHX", "HII", "AXON", "LDOS", "TDG", "HWM", "BA"}
# "行业爆发型"严格门: 只有这些行业 + AI基建电力白名单才算真·产业爆发(AI/半导体/数据中心/电力)
BOOM_SECTORS = {"Technology", "Communication Services", "Utilities"}
BOOM_ALLOW = {"ETN", "EMR", "ROK", "PH", "AME", "HUBB", "PWR", "VRT", "GEV", "JCI",
              "CARR", "TT", "GE", "ANET", "CDW", "GLW", "APH", "TEL"}  # 电气/AI基建/连接
# 能源/油气/化肥 → 商品供需 (即使被归到别的sector)
COMMODITY = {"XOM", "CVX", "COP", "OXY", "SLB", "HAL", "VLO", "MPC", "PSX", "CF", "MOS", "FCX", "NUE"}


def universe_with_sector() -> pd.DataFrame:
    for f in ("alpha_scores.csv", "regime_ml_scores.csv"):
        p = ROOT / f
        if p.exists():
            d = pd.read_csv(p)
            if "ticker" in d.columns and "sector" in d.columns:
                d = d[d["ticker"].astype(str).str.fullmatch(r"[A-Z][A-Z.\-]{0,6}")]
                return d[["ticker", "sector"]].drop_duplicates("ticker")
    return pd.DataFrame()


def mispricing_valuation() -> dict:
    """M(错价程度) 的估值维度: 把 DCF上行 / 分析师目标上行 转成横截面百分位(0-1)再融合。
    绝对值不可信(DCF 的 WACC 假设偏狠), 但同一模型下的相对排序有信号。
    返回 {ticker: (val_pct 0-1, note)}; val_pct 越高=相对越便宜/上行空间越大。"""
    parts = {}   # ticker -> list of (percentile, label)
    # DCF 上行空间 (dcf_valuation.csv), 覆盖最广
    p = ROOT / "dcf_valuation.csv"
    if p.exists():
        d = pd.read_csv(p)
        if "ticker" in d.columns and "upside_pct" in d.columns:
            u = pd.to_numeric(d["upside_pct"], errors="coerce")
            u = u.clip(u.quantile(0.05), u.quantile(0.95))     # 掐掉离谱离群
            rk = u.rank(pct=True)
            for i, tk in enumerate(d["ticker"].astype(str)):
                if pd.notna(rk.iloc[i]):
                    parts.setdefault(tk, []).append((float(rk.iloc[i]), "DCF低估"))
    # 分析师目标上行 (earnings_ai_summaries.csv)
    p2 = ROOT / "earnings_ai_summaries.csv"
    if p2.exists():
        d = pd.read_csv(p2)
        if {"ticker", "analyst_target", "price"}.issubset(d.columns):
            up = pd.to_numeric(d["analyst_target"], errors="coerce") / pd.to_numeric(d["price"], errors="coerce") - 1
            rk = up.rank(pct=True)
            for i, tk in enumerate(d["ticker"].astype(str)):
                if pd.notna(rk.iloc[i]):
                    parts.setdefault(tk, []).append((float(rk.iloc[i]), "分析师上行"))
    out = {}
    for tk, lst in parts.items():
        pct = float(np.mean([x[0] for x in lst]))
        labels = "+".join(sorted(set(x[1] for x in lst)))
        out[tk] = (pct, labels)
    return out


def forward_confirmation() -> dict:
    """数据驱动的 C(前瞻确认, 0-4): 手册三大增强项 —— 分析师上调/盈利超预期/内部人买入。
    alpha_scores.csv 里 sig_revision/sig_surprise/sig_insider 是 0-100 百分位(50=中性)。
    映射: 50→中性≈1.5, 100→强≈4, <50→弱。返回 {ticker: (C值, 主导来源说明)}。"""
    p = ROOT / "alpha_scores.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    # 真实内部人活动 (SEC EDGAR Form 4) — 补 alpha_scores 里那个恒为0的 sig_insider
    edgar_ins = {}
    ep = ROOT / "edgar_events.csv"
    if ep.exists():
        try:
            e = pd.read_csv(ep)
            for _, r in e.iterrows():
                edgar_ins[str(r["ticker"])] = int(pd.to_numeric(r.get("insider_active", 0), errors="coerce") or 0)
        except Exception:
            pass
    out = {}
    for _, r in d.iterrows():
        tk = str(r.get("ticker", ""))
        rev = float(pd.to_numeric(r.get("sig_revision", 50), errors="coerce") or 50)
        sur = float(pd.to_numeric(r.get("sig_surprise", 50), errors="coerce") or 50)
        ins = float(pd.to_numeric(r.get("sig_insider", 50), errors="coerce") or 50)
        # 各项 (pctl-50)/50 ∈ [-1,1]; 加权(上调0.5/超预期0.3/内部人0.2)
        z = ((rev - 50) / 50) * 0.5 + ((sur - 50) / 50) * 0.3 + ((ins - 50) / 50) * 0.2
        insider_active = edgar_ins.get(tk, 0)
        z += 0.12 * insider_active                       # EDGAR 真实内部人活跃 → 前瞻确认加分
        C = float(np.clip(1.5 + z * 2.5, 0, 4))
        drivers = []
        if rev >= 65: drivers.append("分析师上调")
        if sur >= 65: drivers.append("盈利超预期")
        if ins >= 55 or insider_active: drivers.append("内部人买入")
        out[tk] = (round(C, 1), "+".join(drivers) if drivers else "中性")
    return out


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()


def main():
    uni = universe_with_sector()
    if uni.empty:
        print("找不到标普500 universe (alpha_scores.csv)"); return
    print(f"标普500 底库: {len(uni)} 只")

    auto = {}
    ap = ROOT / "auto_event_candidates.csv"
    if ap.exists():
        for _, r in pd.read_csv(ap).iterrows():
            auto[str(r["ticker"])] = r

    px = prices()
    fwd = forward_confirmation()                         # 数据驱动 C: {ticker: (C, drivers)}
    val = mispricing_valuation()                         # M 估值维度: {ticker: (val_pct, note)}
    cot = {}                                             # CFTC COT: 商品供需错配型的 L 调整
    cp = ROOT / "cot_ticker_signal.csv"
    if cp.exists():
        try:
            for _, r in pd.read_csv(cp).iterrows():
                cot[str(r["ticker"])] = (float(r.get("cot_boost", 0)), str(r.get("cot_note", "")))
        except Exception:
            pass
    try:
        from canyon_event_detect import catalyst_map     # N: 临近财报 + 财报后漂移
        catmap = catalyst_map()
    except Exception:
        catmap = {}

    def _M(tk):
        """M(错价): 价格回撤(反弹空间, 全覆盖) 与 估值百分位(DCF+分析师) 融合。返回 (M, note)。"""
        dd_norm = None
        if tk in px.columns:
            s = px[tk].dropna()
            if len(s) > 260:
                dd = float(s.iloc[-1] / s.tail(252).max())     # ≤1, 越小回撤越深
                dd_norm = float(np.clip(1 - dd, 0, 1))         # 回撤越深→反弹空间越大→越接近1
        vp = val.get(tk)
        if vp is not None and dd_norm is not None:
            combined = 0.45 * dd_norm + 0.55 * vp[0]
            note = f"错价:{vp[1]}+回撤"
        elif vp is not None:
            combined = vp[0]; note = f"错价:{vp[1]}"
        elif dd_norm is not None:
            combined = dd_norm; note = "错价:回撤反弹空间"
        else:
            return 2.0, ""
        return float(np.clip(1 + combined * 3, 0, 4)), note
    CONF_MIN = 0.38                                      # 侦测置信度门槛: 达不到就不信侦测,回退行业默认
    rows = []
    n_used = 0
    for _, u in uni.iterrows():
        tk, sec = str(u["ticker"]), str(u["sector"])
        r = auto.get(tk)
        conf = float(r.get("detect_confidence", 0)) if r is not None else 0.0
        tag = str(r.get("confidence_tag", "")) if r is not None else ""
        cdata, cdrv = fwd.get(tk, (1.5, "中性"))          # 前瞻确认(分析师/盈利/内部人)
        # 行业存疑(事件类型与所属行业不符, 如金融股判行业爆发型) → 一律不信, 回退行业默认
        trust = (r is not None and conf >= CONF_MIN and "行业存疑" not in tag)
        if trust:                                        # 侦测足够可信 → 用侦测的事件类型与打分
            # C = 新闻侦测C 与 数据前瞻确认C 各半融合; M 用严谨估值维度
            C = round(0.5 * float(r["C"]) + 0.5 * cdata, 1)
            Mval, mnote = _M(tk)
            M = round(0.5 * float(r["M"]) + 0.5 * Mval, 1)
            rows.append({"ticker": tk, "sector": sec, "event_type": r["event_type"],
                         "L": r["L"], "N": r["N"], "M": M, "P": r["P"], "C": C,
                         "催化清晰度": r.get("催化清晰度", 0.6), "逻辑确认度": r.get("逻辑确认度", 0.6),
                         "失效清晰度": r.get("失效清晰度", 0.6),
                         "note": f"侦测(置信{conf:.2f}):{r['event_type']}" + (f"·前瞻:{cdrv}" if cdrv != "中性" else "")})
            n_used += 1
            continue
        # 侦测缺失或低置信 → 行业默认事件类型 (军工/能源优先)
        if tk in DEFENSE:
            et = "战争/地缘冲击型"
        elif tk in COMMODITY:
            et = "商品供需错配型"
        else:
            et = SECTOR_EVENT.get(sec, "第二春重估型")
        ndens, _nnote = catmap.get(tk, (2, ""))            # N 用真实催化临近度
        Mval, _mnote = _M(tk)                               # M 用严谨估值维度(DCF+分析师+回撤)
        L, N, M, P, C = 2.3, float(ndens), Mval, 2.5, cdata  # C 数据驱动前瞻确认
        if tk in px.columns:
            s = px[tk].dropna()
            if len(s) > 260:
                mom = float(s.iloc[-1] / s.iloc[-252] - 1)
                L = float(np.clip(2.2 + mom, 1, 4))
        cnote = f"·前瞻:{cdrv}" if cdrv != "中性" else ""
        if r is not None and "行业存疑" in tag:
            note = f"底库默认({sec}→{et}),侦测判{r['event_type']}但行业不符已忽略{cnote}"
        elif r is not None:
            note = f"底库默认({sec}→{et}),侦测低置信{conf:.2f}已忽略{cnote}"
        else:
            note = f"底库默认({sec}→{et}){cnote or ',待精修'}"
        rows.append({"ticker": tk, "sector": sec, "event_type": et,
                     "L": round(L, 1), "N": N, "M": round(M, 1), "P": P, "C": C,
                     "催化清晰度": 0.5, "逻辑确认度": 0.5, "失效清晰度": 0.6, "note": note})
    df = pd.DataFrame(rows)
    # 行业爆发型严格门: 非科技/通信/公用 且 不在AI基建白名单 → 降级为第二春重估型
    # (铁路/保险/烟草等被误标"行业爆发"的传统股, 改回更贴切的重估型)
    n_demote = 0
    for i, row in df.iterrows():
        if row["event_type"] == "行业爆发型":
            tk, sec = str(row["ticker"]), str(row["sector"])
            if sec not in BOOM_SECTORS and tk not in BOOM_ALLOW:
                df.at[i, "event_type"] = "第二春重估型"
                df.at[i, "note"] = str(row["note"]) + "·行业爆发降级(非科技/AI基建)"
                n_demote += 1
    if n_demote:
        print(f"  行业爆发型严格门: {n_demote} 只非科技股降级为第二春重估型")
    # CFTC COT 调整: 商品供需错配型标的按持仓 setup 调 L
    if cot:
        for i, row in df.iterrows():
            tk = str(row["ticker"])
            if row["event_type"] == "商品供需错配型" and tk in cot:
                b, cnote = cot[tk]
                if b != 0:
                    df.at[i, "L"] = round(float(np.clip(float(row["L"]) + b, 0, 4)), 1)
                    df.at[i, "note"] = str(row["note"]) + f"·COT({cnote})"
    df.to_csv(ROOT / "event_pool.csv", index=False)
    n_auto = n_used
    print(f"✓ event_pool.csv: {len(df)} 只 (标普500全体; {n_auto} 只用可信侦测(置信≥{CONF_MIN}), 其余按行业默认)")
    print("  事件类型分布:", df["event_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
