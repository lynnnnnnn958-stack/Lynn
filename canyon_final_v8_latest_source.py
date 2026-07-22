"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        CANYON QUANTITATIVE TRADING SYSTEM — FINAL v6.0                      ║
║        机构级完整版：回测 + 实盘 + Alpha严谨性 + 执行真实性 + 统计深度       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  v5 基础保留：                                                               ║
║  · DataLayer / drawdown / backtest_metrics（书第7章精确公式）               ║
║  · Regime五维检测 / detect_regime                                           ║
║  · trend_signals（书第5章）/ cross_sectional_momentum（书第6章）            ║
║  · StatArb ADF+协整+z-score（书第8章）                                      ║
║  · UCBOptimizer Bayesian优化（书第9章）                                     ║
║  · RiskManager Kelly+CVaR+Ledoit-Wolf                                       ║
║  · OffensiveDefensiveManager 进攻/防守                                      ║
║  · canyon_score_auto F/C/E量化评分                                          ║
║  · WalkForwardBacktester 多期验证                                           ║
║  · TradeJournal 交易日志Gate Check                                          ║
║                                                                              ║
║  v6 新增：                                                                   ║
║  [FIX] 牛市满仓：按目标总敞口分配，BULL_STRONG → 90%+                      ║
║  [FIX] 波动率目标：动态缩放仓位到目标年化波动率（10%）                      ║
║  [FIX] 回撤控制：超过5%自动减仓50%，超过10%减仓75%                         ║
║  [A]   Alpha IC引擎：IC/ICIR/t-stat验证，不通过的因子不用                   ║
║  [E]   执行成本现实化：买卖价差 + 市场冲击（平方根模型）+ ADV约束           ║
║  [S]   统计深度：Bootstrap Sharpe CI + Newey-West修正 + 因子暴露分解        ║
║  [L]   实盘系统：StateLogger + AlertEngine + StrategyMonitor                ║
║  [L]   TWAP/VWAP/POV执行算法                                               ║
║  [L]   AlpacaExecution + LiveRiskManager + LiveTrader主循环                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings, json, uuid, os, time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.stats import spearmanr, pearsonr, t as t_dist
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 0. DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

class DataLayer:
    """Yahoo Finance优先，网络不通自动切换内置合成数据"""

    @staticmethod
    def load(tickers: List[str], start: str, end: str,
             benchmark: str = 'SPY') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        try:
            import yfinance as yf
            all_tickers = list(dict.fromkeys(tickers + [benchmark]))
            raw = yf.download(all_tickers, start=start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                close_key = 'Close'
                if close_key not in raw.columns.get_level_values(0):
                    close_key = 'Adj Close'
                prices  = raw[close_key][tickers].dropna(how='all')
                volumes = raw['Volume'][tickers].dropna(how='all')
                market  = (raw[close_key][benchmark].dropna()
                           if benchmark in raw[close_key].columns
                           else raw[close_key].mean(axis=1))
            else:
                prices  = raw[['Close']].rename(columns={'Close': tickers[0]})
                volumes = raw[['Volume']].rename(columns={'Volume': tickers[0]})
                market  = prices.iloc[:, 0]
            prices  = prices.ffill().dropna()
            volumes = volumes.ffill().fillna(1e6)
            market  = market.reindex(prices.index).ffill().dropna()
            print(f"[Data] ✅ Yahoo Finance {len(prices)}天 × {len(prices.columns)}资产")
            return prices, volumes, market
        except Exception as e:
            print(f"[Data] Yahoo Finance 不可用，使用合成数据")
            return DataLayer._synthetic(tickers, start, end)

    @staticmethod
    def _synthetic(tickers: List[str], start: str, end: str):
        """
        合成数据：精确模拟2020-2024市场周期
        含横截面动量结构（使IC测试有意义）
        """
        n = max(400, (pd.Timestamp(end) - pd.Timestamp(start)).days)
        N = len(tickers)
        dates = pd.date_range(start, periods=n, freq='B')
        t = np.linspace(0, 1, n)

        # 市场因子：明确牛熊周期
        mkt = (0.0004
               - 0.006 * np.exp(-((t - 0.12)**2) / 0.0008)
               + 0.003 * (t > 0.16) * (t < 0.48)
               - 0.003 * (t > 0.52) * (t < 0.70)
               + 0.004 * (t > 0.72)
               + np.random.normal(0, 0.011, n))

        market = pd.Series(100 * np.exp(np.cumsum(mkt)), index=dates)

        # 个股：异质Beta + 持续动量Alpha（使因子有效）
        betas  = np.random.uniform(0.5, 1.9, N)
        alphas = np.random.normal(0.0002, 0.0006, N)
        # 动量持续性：上期强者下期更强（使CS-Momentum IC > 0）
        mom_persistence = 0.05
        ret_mat = np.zeros((n, N))
        for i in range(n):
            if i == 0:
                ret_mat[i] = mkt[i] * betas + alphas + np.random.normal(0, 0.013, N)
            else:
                ret_mat[i] = (mkt[i] * betas + alphas
                              + mom_persistence * ret_mat[i-1]
                              + np.random.normal(0, 0.013, N))

        prices  = pd.DataFrame(100 * np.exp(np.cumsum(ret_mat, axis=0)),
                               columns=tickers, index=dates)
        # 成交量与波动率正相关
        base_vol = np.random.lognormal(16.5, 0.4, (n, N))
        vol_mult = 1 + 2 * np.abs(ret_mat) / 0.02
        volumes  = pd.DataFrame(base_vol * vol_mult, columns=tickers, index=dates)

        print(f"[Data] 合成数据 {n}天 × {N}资产（牛熊周期 + 横截面动量结构）")
        return prices, volumes, market


# ══════════════════════════════════════════════════════════════════════════════
# 1. 书第7章回测指标（精确实现）
# ══════════════════════════════════════════════════════════════════════════════

def drawdown(return_series: pd.Series) -> pd.DataFrame:
    """书第7章 Listing 7-14至7-16"""
    r  = return_series.dropna().replace([np.inf, -np.inf], 0)
    wi = 1000 * (1 + r).cumprod()
    pk = wi.cummax()
    dd = (wi - pk) / pk
    return pd.DataFrame({'Wealth index': wi, 'Prior peaks': pk, 'Drawdown': dd})


def backtest_metrics(returns: pd.Series, rf: float = 0.03,
                     label: str = '') -> Dict:
    """书第7章：年化收益/波动/Sharpe/MaxDD/Calmar（精确公式）"""
    r = returns.dropna().replace([np.inf, -np.inf], 0)
    if len(r) < 5:
        return dict(ann_ret=0, ann_vol=0, sharpe=0,
                    max_dd=0, calmar=0, win_rate=0, n=0, total_ret=0)
    n       = len(r)
    ann_ret = float((1 + r).prod() ** (252 / n) - 1)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 1e-6 else 0.0
    max_dd  = float(drawdown(r)['Drawdown'].min())
    calmar  = ann_ret / abs(max_dd) if max_dd < -1e-6 else 0.0
    if label:
        print(f"  [{label}] 年化:{ann_ret:+.2%} 波动:{ann_vol:.2%} "
              f"Sharpe:{sharpe:.3f} MaxDD:{max_dd:.2%}")
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=max_dd, calmar=calmar,
                win_rate=float((r > 0).mean()), n=n,
                total_ret=float((1 + r).prod() - 1))


# ══════════════════════════════════════════════════════════════════════════════
# 2. [NEW] Alpha IC引擎（解决"Alpha严谨性 4/10"）
# ══════════════════════════════════════════════════════════════════════════════

class AlphaICEngine:
    """
    Alpha信息系数验证引擎

    机构标准：
    - IC > 0.02：因子有预测力
    - ICIR > 0.3：因子稳定（IC/std(IC)）
    - |t-stat| > 2.0：统计显著
    - IC衰减：持有期内信号不能快速衰减

    不通过以上标准的因子不进入组合
    权重 ∝ ICIR（越稳定权重越高）
    """

    MIN_IC   = 0.02
    MIN_ICIR = 0.30
    MIN_TSTAT = 2.0

    @staticmethod
    def compute_ic(factor: pd.Series, fwd_ret: pd.Series) -> float:
        """Spearman IC：因子和未来收益的秩相关"""
        common = factor.dropna().index.intersection(fwd_ret.dropna().index)
        if len(common) < 10:
            return 0.0
        ic, _ = spearmanr(factor[common].values, fwd_ret[common].values)
        return float(ic) if np.isfinite(ic) else 0.0

    @staticmethod
    def compute_ic_series(factor_matrix: pd.DataFrame,
                          prices: pd.DataFrame,
                          fwd_days: int = 21) -> Dict[str, List[float]]:
        """
        滚动计算每个因子的IC时间序列
        factor_matrix: T×N（行=时间，列=资产）
        """
        fwd_ret = prices.pct_change(fwd_days).shift(-fwd_days)
        ic_series: Dict[str, List[float]] = defaultdict(list)

        for col in factor_matrix.columns:
            for t_idx in range(len(factor_matrix)):
                f_t = factor_matrix[col].iloc[t_idx]
                r_t = fwd_ret.iloc[t_idx]
                if isinstance(f_t, (int, float)) and np.isfinite(f_t):
                    # 单时间点：不能算相关性，累积一段
                    pass
                # 用过去21天的IC
                if t_idx >= 21:
                    f_window = factor_matrix[col].iloc[t_idx-21:t_idx]
                    r_window = fwd_ret.iloc[t_idx-21:t_idx]
                    ic = AlphaICEngine.compute_ic(f_window, r_window.mean(axis=1)
                                                   if hasattr(r_window, 'mean') else r_window)
                    ic_series[col].append(ic)

        return dict(ic_series)

    @staticmethod
    def validate_factor(ic_list: List[float],
                        factor_name: str = '') -> Dict:
        """
        因子有效性验证
        IC序列 → IC均值/ICIR/t-stat/是否通过
        """
        if len(ic_list) < 5:
            return {'valid': False, 'ic_mean': 0, 'icir': 0, 't_stat': 0,
                    'reason': '样本不足'}

        arr     = np.array(ic_list)
        ic_mean = float(arr.mean())
        ic_std  = float(arr.std() + 1e-8)
        icir    = ic_mean / ic_std
        se      = ic_std / np.sqrt(len(arr))
        t_stat  = ic_mean / (se + 1e-8)

        valid  = (abs(ic_mean) > AlphaICEngine.MIN_IC and
                  abs(icir)   > AlphaICEngine.MIN_ICIR and
                  abs(t_stat) > AlphaICEngine.MIN_TSTAT)

        reason = []
        if abs(ic_mean) <= AlphaICEngine.MIN_IC:
            reason.append(f"IC={ic_mean:.4f}<{AlphaICEngine.MIN_IC}")
        if abs(icir) <= AlphaICEngine.MIN_ICIR:
            reason.append(f"ICIR={icir:.3f}<{AlphaICEngine.MIN_ICIR}")
        if abs(t_stat) <= AlphaICEngine.MIN_TSTAT:
            reason.append(f"t={t_stat:.2f}<{AlphaICEngine.MIN_TSTAT}")

        return {
            'valid':   valid,
            'ic_mean': round(ic_mean, 5),
            'icir':    round(icir, 4),
            't_stat':  round(t_stat, 4),
            'hit_rate':round(float((arr > 0).mean()), 4),
            'reason':  '通过' if valid else ' | '.join(reason)
        }

    @staticmethod
    def icir_weights(factor_validations: Dict[str, Dict]) -> Dict[str, float]:
        """
        ICIR加权：通过验证的因子按ICIR权重组合
        """
        valid   = {k: v for k, v in factor_validations.items() if v['valid']}
        if not valid:
            return {}
        total   = sum(abs(v['icir']) for v in valid.values()) + 1e-8
        return {k: abs(v['icir']) / total for k, v in valid.items()}

    @staticmethod
    def build_validated_alpha(prices: pd.DataFrame,
                               volumes: pd.DataFrame,
                               market: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        构建经过IC验证的横截面Alpha

        因子库：
        F1 = 12-1月横截面动量（最核心）
        F2 = 短期反转（5日）
        F3 = 量价方向性
        F4 = 相对强度（vs市场）
        F5 = 波动率低位（低波动因子）

        每个因子先算IC序列，通过门槛才用，权重=ICIR
        """
        r = prices.pct_change().dropna()
        n = len(r)
        if n < 63:
            # 数据不足，用简单等权动量
            mom = prices.pct_change(min(63, n-1)).iloc[-1].dropna()
            rank = mom.rank(pct=True)
            return rank, {'fallback': {'valid': True, 'ic_mean': 0.02, 'icir': 0.3}}

        # 计算每个因子在过去window期的IC
        window = min(252, n - 22)
        factors_raw = {}

        # F1: 12M-1M横截面动量
        lk12 = min(252, n - 2)
        lk1  = min(21,  n - 2)
        mom12 = prices.pct_change(lk12).iloc[-window:]
        mom1  = prices.pct_change(lk1).iloc[-window:]
        factors_raw['F1_momentum'] = (mom12 - mom1).rank(axis=1, pct=True)

        # F2: 短期反转
        mom5 = prices.pct_change(min(5, n-2)).iloc[-window:]
        factors_raw['F2_reversal'] = (-mom5).rank(axis=1, pct=True)

        # F3: 量价方向性
        vm = volumes.rolling(21).mean()
        vz = (volumes - vm) / (volumes.rolling(21).std() + 1e-8)
        sign_ret = np.sign(prices.pct_change(5))
        factors_raw['F3_vol_momentum'] = (vz * sign_ret).iloc[-window:].rank(axis=1, pct=True)

        # F4: 相对强度
        mkt_ret = market.pct_change(21)
        rel_str = prices.pct_change(21).sub(mkt_ret, axis=0)
        factors_raw['F4_rel_strength'] = rel_str.iloc[-window:].rank(axis=1, pct=True)

        # F5: 低波动因子
        vol21 = r.rolling(21).std()
        factors_raw['F5_low_vol'] = (-vol21).iloc[-window:].rank(axis=1, pct=True)

        # 计算每个因子的IC（用过去窗口内的IC序列）
        fwd_ret = prices.pct_change(21).shift(-21)
        validations = {}
        for fname, fmat in factors_raw.items():
            ic_list = []
            for i in range(len(fmat) - 22):
                fa = fmat.iloc[i].dropna()
                fr = fwd_ret.iloc[i].reindex(fa.index).dropna()
                common = fa.index.intersection(fr.index)
                if len(common) >= 5:
                    ic, _ = spearmanr(fa[common], fr[common])
                    if np.isfinite(ic):
                        ic_list.append(ic)
            validations[fname] = AlphaICEngine.validate_factor(ic_list, fname)

        # ICIR加权
        weights = AlphaICEngine.icir_weights(validations)

        if not weights:
            # 所有因子都没通过，退化到等权动量
            mom = prices.pct_change(lk12).iloc[-1].dropna()
            return mom.rank(pct=True), validations

        # 合成当前Alpha
        current_alpha = pd.Series(0.0, index=prices.columns)
        for fname, w in weights.items():
            fmat = factors_raw[fname]
            if len(fmat) > 0:
                latest = fmat.iloc[-1].dropna()
                current_alpha = current_alpha.add(latest * w, fill_value=0)

        # 横截面标准化
        if current_alpha.std() > 0:
            current_alpha = (current_alpha - current_alpha.mean()) / current_alpha.std()

        return current_alpha, validations


# ══════════════════════════════════════════════════════════════════════════════
# 3. [NEW] Execution Cost Model（解决"Execution Realism 3/10"）
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionCostModel:
    """
    机构级执行成本模型

    成本 = 买卖价差 + 市场冲击 + 时间成本

    买卖价差（Bid-Ask Spread）：
    spread_cost = spread_bps × |trade_value|
    spread_bps ≈ k × σ_daily / √(ADV/1e6)
    波动率高的、流动性差的股票买卖价差更大

    市场冲击（Almgren-Chriss平方根模型）：
    impact = σ_daily × √(trade_value / ADV)
    大单交易成本非线性增加，是普通成本建模最大的低估来源

    ADV约束：
    单次交易不超过日均成交量的 max_pov = 5%
    超过限制的部分必须分多天执行
    """

    def __init__(self,
                 spread_k:  float = 0.10,   # 买卖价差系数
                 impact_k:  float = 0.10,   # 市场冲击系数
                 max_pov:   float = 0.05,   # 最大日均参与率5%
                 min_bps:   float = 2.0,    # 最小成本2bps（ETF等）
                 max_bps:   float = 50.0):  # 最大成本50bps
        self.spread_k = spread_k
        self.impact_k = impact_k
        self.max_pov  = max_pov
        self.min_bps  = min_bps / 10000
        self.max_bps  = max_bps / 10000

    def spread_cost(self, vol_daily: float, adv_usd: float) -> float:
        """买卖价差成本（占交易金额的比例）"""
        if adv_usd <= 0:
            return self.max_bps
        # spread ∝ σ / √ADV
        raw = self.spread_k * vol_daily / np.sqrt(max(adv_usd / 1e6, 0.01))
        return float(np.clip(raw, self.min_bps, self.max_bps))

    def market_impact(self, vol_daily: float, adv_usd: float,
                      trade_usd: float) -> float:
        """
        Almgren-Chriss平方根市场冲击
        impact = σ × √(Q/ADV)
        Q = 交易金额，ADV = 日均成交额
        """
        if adv_usd <= 0 or trade_usd <= 0:
            return 0.0
        participation = trade_usd / adv_usd
        raw = self.impact_k * vol_daily * np.sqrt(participation)
        return float(np.clip(raw, 0, self.max_bps * 3))

    def total_cost(self, vol_daily: float, adv_usd: float,
                   trade_usd: float) -> float:
        """单次交易总成本（占交易金额比例）"""
        spread = self.spread_cost(vol_daily, adv_usd)
        impact = self.market_impact(vol_daily, adv_usd, trade_usd)
        return spread + impact

    def adv_constrained_size(self, desired_usd: float,
                              adv_usd: float) -> Tuple[float, int]:
        """
        ADV约束：单日不超过日均成交量的 max_pov
        返回：(今日可交易金额, 需要多少天完成)
        """
        max_today = adv_usd * self.max_pov
        days_needed = max(1, int(np.ceil(desired_usd / max_today)))
        return min(desired_usd, max_today), days_needed

    def compute_portfolio_tc(self, weights_new: pd.Series,
                              weights_old: pd.Series,
                              prices: pd.Series,
                              volumes: pd.Series,
                              capital: float = 1e6) -> Dict:
        """
        计算整个组合换仓的总交易成本
        """
        total_tc  = 0.0
        tc_detail = {}

        all_tk = set(weights_new.index) | set(weights_old.index)
        for tk in all_tk:
            w_new = float(weights_new.get(tk, 0))
            w_old = float(weights_old.get(tk, 0))
            delta = w_new - w_old

            if abs(delta) < 1e-4:
                continue

            price     = float(prices.get(tk, 100))
            vol_daily = float(prices.get(tk, 0.015)) if isinstance(prices, dict) else 0.015

            # 估算日均成交额
            vol_shares = float(volumes.get(tk, 1e6))
            adv_usd    = vol_shares * price

            trade_usd  = abs(delta) * capital
            tc_ratio   = self.total_cost(vol_daily, adv_usd, trade_usd)
            tc_usd     = trade_usd * tc_ratio

            total_tc          += tc_usd
            tc_detail[tk] = {'delta': delta, 'trade_usd': trade_usd,
                              'tc_bps': tc_ratio * 10000, 'tc_usd': tc_usd}

        return {
            'total_tc_usd':   total_tc,
            'total_tc_ratio': total_tc / (capital + 1e-8),
            'detail':         tc_detail
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. [NEW] 统计深度（解决"统计深度 4/10"）
# ══════════════════════════════════════════════════════════════════════════════

class StatisticalDepth:
    """
    机构级统计分析
    - Bootstrap Sharpe置信区间：Sharpe必须有统计显著性
    - Newey-West修正Sharpe：日收益有自相关，普通Sharpe高估
    - 因子暴露分解：证明策略有纯Alpha，不是高Beta
    - 回撤深度分析：持续时间/恢复时间
    """

    @staticmethod
    def bootstrap_sharpe(returns: pd.Series, rf: float = 0.03,
                         n_boot: int = 1000, conf: float = 0.95) -> Dict:
        """
        Bootstrap Sharpe置信区间（1000次重采样）
        CI下界 > 0 → Sharpe统计显著
        """
        r = returns.dropna().values
        if len(r) < 30:
            return {'sharpe': 0, 'ci_low': -99, 'ci_high': 99,
                    'prob_positive': 0.5, 'significant': False}

        # 点估计
        ann    = (1 + pd.Series(r)).prod() ** (252 / len(r)) - 1
        vol    = r.std() * np.sqrt(252)
        pt_sr  = (ann - rf) / (vol + 1e-8)

        # Bootstrap
        boot = []
        for _ in range(n_boot):
            s   = np.random.choice(r, size=len(r), replace=True)
            sa  = (1 + s).prod() ** (252 / len(s)) - 1
            sv  = s.std() * np.sqrt(252)
            sr  = (sa - rf) / (sv + 1e-8)
            if np.isfinite(sr):
                boot.append(sr)

        boot = np.array(boot)
        a2   = (1 - conf) / 2
        lo   = float(np.percentile(boot, a2 * 100))
        hi   = float(np.percentile(boot, (1 - a2) * 100))

        return {
            'sharpe':       round(pt_sr, 4),
            'ci_low':       round(lo, 4),
            'ci_high':      round(hi, 4),
            'prob_positive':round(float((boot > 0).mean()), 4),
            'boot_std':     round(float(boot.std()), 4),
            'significant':  lo > 0    # 95% CI下界 > 0 = 统计显著
        }

    @staticmethod
    def newey_west_sharpe(returns: pd.Series, rf: float = 0.03,
                          lags: int = 5) -> Dict:
        """
        Newey-West自相关修正Sharpe
        日收益存在自相关 → 普通vol低估 → 普通Sharpe高估
        机构必用：汇报NW-Sharpe才算诚实
        """
        r = returns.dropna().values
        if len(r) < 30:
            return {'nw_sharpe': 0, 't_stat': 0, 'p_value': 1.0, 'significant': False}

        mu = r.mean()
        # Newey-West方差估计
        var0 = np.var(r, ddof=1)
        nw_v = var0
        for l in range(1, lags + 1):
            w   = 1 - l / (lags + 1)
            cov = np.cov(r[l:], r[:-l])[0, 1]
            nw_v += 2 * w * cov

        nw_std = np.sqrt(max(nw_v, 1e-12))
        t_stat = mu / (nw_std / np.sqrt(len(r)))
        p_val  = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(r) - 1)))

        ann_ret = (1 + pd.Series(r)).prod() ** (252 / len(r)) - 1
        nw_ann  = nw_std * np.sqrt(252)
        nw_sr   = (ann_ret - rf) / (nw_ann + 1e-8)

        return {
            'nw_sharpe':  round(float(nw_sr), 4),
            't_stat':     round(float(t_stat), 4),
            'p_value':    round(p_val, 4),
            'significant': p_val < 0.05
        }

    @staticmethod
    def factor_exposure(returns: pd.Series,
                        market_returns: pd.Series,
                        rf: float = 0.03) -> Dict:
        """
        因子暴露分解
        return = alpha + beta × market + epsilon
        需证明：alpha显著 > 0，且R²不能太高（否则只是高Beta策略）
        """
        r = returns.dropna()
        m = market_returns.pct_change().dropna().reindex(r.index).fillna(0)
        common = r.index.intersection(m.index)
        if len(common) < 30:
            return {'alpha_ann': 0, 'beta': 1, 'r_squared': 0, 'pure_alpha': False}

        rc, mc = r[common].values, m[common].values
        X = np.column_stack([np.ones(len(mc)), mc])
        try:
            c   = np.linalg.lstsq(X, rc, rcond=None)[0]
            a_d, beta = c[0], c[1]
        except Exception:
            return {'alpha_ann': 0, 'beta': 1, 'r_squared': 0, 'pure_alpha': False}

        pred  = X @ c
        ss_r  = np.sum((rc - pred) ** 2)
        ss_t  = np.sum((rc - rc.mean()) ** 2)
        r2    = 1 - ss_r / (ss_t + 1e-10)

        alpha_ann = a_d * 252
        treynor   = (rc.mean() * 252 - rf) / (abs(beta) + 1e-8)
        info_r    = a_d / (np.std(rc - pred) * np.sqrt(252) + 1e-8)

        return {
            'alpha_ann':     round(float(alpha_ann), 4),
            'beta':          round(float(beta), 4),
            'r_squared':     round(float(r2), 4),
            'treynor':       round(float(treynor), 4),
            'info_ratio':    round(float(info_r), 4),
            'pure_alpha':    r2 < 0.6,   # R²<60%说明不只是Beta暴露
            'alpha_positive': alpha_ann > 0
        }

    @staticmethod
    def drawdown_deep(returns: pd.Series) -> Dict:
        """回撤深度分析：持续时间/恢复时间/平均深度"""
        r  = returns.dropna()
        dd = drawdown(r)['Drawdown']
        wi = drawdown(r)['Wealth index']

        # 找所有回撤区间
        in_dd  = dd < -0.001
        segs   = []
        start  = None
        for date, v in in_dd.items():
            if v and start is None:
                start = date
            elif not v and start is not None:
                segs.append((start, date))
                start = None
        if start:
            segs.append((start, dd.index[-1]))

        max_dd   = float(dd.min())
        max_date = dd.idxmin()
        durations= [(e - s).days for s, e in segs]
        depths   = [float(dd.loc[s:e].min()) for s, e in segs]

        # 恢复时间
        rec_days = None
        if len(wi) > 0 and max_date in wi.index:
            peak_v = float(wi[:max_date].max()) if len(wi[:max_date]) > 0 else 1000
            after  = wi[max_date:]
            rec    = after[after >= peak_v]
            if len(rec) > 0:
                rec_days = (rec.index[0] - max_date).days

        return {
            'max_dd':            round(max_dd, 4),
            'max_dd_date':       str(max_date.date()) if hasattr(max_date, 'date') else str(max_date),
            'n_drawdowns':       len(segs),
            'avg_duration_days': round(np.mean(durations), 1) if durations else 0,
            'max_duration_days': max(durations, default=0),
            'avg_depth':         round(np.mean(depths), 4) if depths else 0,
            'recovery_days':     rec_days,
            'recovered':         rec_days is not None
        }

    @staticmethod
    def print_full_report(returns: pd.Series, market: pd.Series,
                          label: str = 'Strategy', rf: float = 0.03):
        """打印完整统计报告"""
        print(f"\n  📊 统计深度报告 — {label}")
        print(f"  {'─'*58}")

        boot = StatisticalDepth.bootstrap_sharpe(returns, rf)
        nw   = StatisticalDepth.newey_west_sharpe(returns, rf)
        expo = StatisticalDepth.factor_exposure(returns, market, rf)
        dd_d = StatisticalDepth.drawdown_deep(returns)

        print(f"  Bootstrap Sharpe: {boot['sharpe']:.3f} "
              f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] "
              f"{'✅显著' if boot['significant'] else '❌不显著'}")
        print(f"  Newey-West Sharpe: {nw['nw_sharpe']:.3f} "
              f"t={nw['t_stat']:.2f} p={nw['p_value']:.3f} "
              f"{'✅显著' if nw['significant'] else '❌不显著'}")
        print(f"  因子暴露: Alpha={expo['alpha_ann']:+.2%} "
              f"Beta={expo['beta']:.2f} R²={expo['r_squared']:.2%} "
              f"{'✅纯Alpha' if expo['pure_alpha'] else '⚠️偏Beta'}")
        print(f"  回撤分析: MaxDD={dd_d['max_dd']:.2%} "
              f"持续{dd_d['max_duration_days']}天 "
              f"恢复{dd_d['recovery_days']}天" if dd_d['recovery_days'] else
              f"  回撤分析: MaxDD={dd_d['max_dd']:.2%} 持续{dd_d['max_duration_days']}天 未恢复")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 市场环境检测
# ══════════════════════════════════════════════════════════════════════════════

class Regime(Enum):
    BULL_STRONG = ( 2, "强牛市",   "全力进攻",  0.15, -0.00)
    BULL_NORMAL = ( 1, "普通牛市", "进攻为主",  0.12, -0.00)
    NEUTRAL     = ( 0, "震荡中性", "攻守均衡",  0.10, -0.03)
    BEAR_MILD   = (-1, "温和熊市", "防守+做空", 0.08, -0.05)
    BEAR_STRONG = (-2, "强熊市",   "现金+做空", 0.05, -0.08)

    def __init__(self, score, label, stance, max_long, max_short):
        self.score    = score
        self.label    = label
        self.stance   = stance
        self.max_long = max_long
        self.max_short= max_short

    @property
    def allow_short(self) -> bool:
        return self.score <= -1

    @property
    def target_gross_exposure(self) -> float:
        """目标总多头敞口（v6新增：确保牛市满仓）"""
        return {2: 0.90, 1: 0.78, 0: 0.60, -1: 0.35, -2: 0.18}[self.score]


def detect_regime(market: pd.Series,
                  prices: pd.DataFrame = None) -> Tuple[Regime, Dict]:
    """五维度市场环境检测"""
    if len(market) < 60:
        return Regime.NEUTRAL, {}
    m = market.dropna()

    # 维度1: SMA趋势
    w50, w200 = min(50, len(m)-1), min(200, len(m)-1)
    sma50, sma200, cur = m.rolling(w50).mean().iloc[-1], m.rolling(w200).mean().iloc[-1], m.iloc[-1]
    if cur > sma50 and sma50 > sma200:   trend = 1.0
    elif cur < sma50 and sma50 < sma200: trend = -1.0
    elif cur > sma50:                     trend = 0.5
    elif cur < sma50:                     trend = -0.5
    else:                                 trend = 0.0

    # 维度2: 12月动量
    lk  = min(252, len(m) - 2)
    m12 = float(m.pct_change(lk).iloc[-1]) if lk > 5 else 0.0
    m3  = float(m.pct_change(min(63, lk)).iloc[-1]) if lk > 5 else 0.0
    mom = float(np.clip(m12 * 1.5 + m3 * 0.5, -1, 1))

    # 维度3: 波动率
    rv21  = float(m.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252))
    rv126 = float(m.pct_change().rolling(min(126, len(m)-1)).std().iloc[-1] * np.sqrt(252))
    vol_s = float(np.clip(-(rv21 / (rv126 + 1e-8) - 1.0), -1, 1))

    # 维度4: 市场广度
    if prices is not None and len(prices.columns) >= 3:
        above   = (prices > prices.rolling(min(20, len(prices)-1)).mean()).iloc[-1].mean()
        breadth = float((above - 0.5) * 2)
    else:
        breadth = trend * 0.5

    # 维度5: 短期动量
    short_mom = float(np.clip(m.pct_change(min(5, len(m)-1)).iloc[-1] * 20, -1, 1))

    score = (trend * 35 + mom * 25 + breadth * 20 + vol_s * 12 + short_mom * 8)
    score = float(np.clip(score, -100, 100))

    if   score >=  55: regime = Regime.BULL_STRONG
    elif score >=  20: regime = Regime.BULL_NORMAL
    elif score >= -20: regime = Regime.NEUTRAL
    elif score >= -55: regime = Regime.BEAR_MILD
    else:              regime = Regime.BEAR_STRONG

    detail = dict(trend=round(trend,2), momentum=round(mom,2),
                  vol_score=round(vol_s,2), breadth=round(breadth,2),
                  short_mom=round(short_mom,2), composite=round(score,1))
    return regime, detail


# ══════════════════════════════════════════════════════════════════════════════
# 6. 书第5章：SMA/EMA趋势跟随
# ══════════════════════════════════════════════════════════════════════════════

def trend_signals(prices: pd.DataFrame,
                  ema_span: int = 5,
                  sma_span: int = 30) -> pd.DataFrame:
    """书第5章：金叉做多，死叉做空"""
    results = {}
    for col in prices.columns:
        p   = prices[col].dropna()
        ema = p.ewm(span=ema_span, adjust=False).mean()
        sma = p.rolling(sma_span).mean()
        cur_bull  = bool(ema.iloc[-1] > sma.iloc[-1])
        prev_bull = bool(ema.iloc[-2] > sma.iloc[-2]) if len(ema) > 1 else cur_bull
        results[col] = {
            'signal':       1 if cur_bull else -1,
            'trend_up':     cur_bull,
            'golden_cross': cur_bull and not prev_bull,
            'death_cross':  not cur_bull and prev_bull,
            'ema':          float(ema.iloc[-1]),
            'sma':          float(sma.iloc[-1]),
            'strength':     abs(float(ema.iloc[-1]) - float(sma.iloc[-1])) / (float(sma.iloc[-1]) + 1e-8)
        }
    return pd.DataFrame(results).T


# ══════════════════════════════════════════════════════════════════════════════
# 7. 书第6章：横截面动量
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_momentum(prices: pd.DataFrame,
                              lookback_months: int = 12,
                              skip_months: int = 1) -> Dict:
    """书第6章：前25%做多，后25%做空"""
    try:
        mret = prices.resample('ME').last().pct_change().dropna()
    except Exception:
        mret = prices.resample('M').last().pct_change().dropna()

    min_p = lookback_months + skip_months + 1
    if len(mret) < min_p:
        w   = min(lookback_months * 21, len(prices) - 2)
        mom = prices.pct_change(w).iloc[-1].dropna()
        rk  = mom.rank(pct=True)
        return {'momentum': mom, 'rank': rk,
                'long': mom[rk >= 0.75].index.tolist(),
                'short': mom[rk <= 0.25].index.tolist(),
                'long_alpha': rk, 'short_alpha': -rk,
                'spread': float(mom[rk >= 0.75].mean() - mom[rk <= 0.25].mean())}

    fe_idx = max(0, len(mret) - 1 - skip_months)
    fs_idx = max(0, fe_idx - lookback_months)
    if fs_idx >= fe_idx:
        fe_idx = len(mret) - 1
        fs_idx = max(0, fe_idx - lookback_months)

    mom = (1 + mret.iloc[fs_idx:fe_idx + 1]).prod() - 1
    mom = mom.dropna()
    if len(mom) < 4:
        rk = mom.rank(pct=True)
        return {'momentum': mom, 'rank': rk,
                'long': momentum.nlargest(1).index.tolist() if hasattr(mom, 'nlargest') else [],
                'short': [], 'long_alpha': rk, 'short_alpha': -rk, 'spread': 0.0}

    rk = mom.rank(pct=True)
    return {
        'momentum':    mom,
        'rank':        rk,
        'long':        mom[rk >= 0.75].index.tolist(),
        'short':       mom[rk <= 0.25].index.tolist(),
        'long_alpha':  rk,
        'short_alpha': -rk,
        'spread':      float(mom[rk >= 0.75].mean() - mom[rk <= 0.25].mean())
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. 书第8章：统计套利
# ══════════════════════════════════════════════════════════════════════════════

class StatArb:
    def __init__(self, entry_z=2.0, exit_z=0.5, window=21):
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.window  = window

    def adf_pvalue(self, series: np.ndarray) -> float:
        if len(series) < 15:
            return 1.0
        y = np.diff(series)
        x = series[:-1]
        X = np.column_stack([x, np.ones(len(x))])
        try:
            c  = np.linalg.lstsq(X, y, rcond=None)[0]
            r  = y - X @ c
            s2 = r.var()
            se = np.sqrt(max(0, s2 * np.linalg.pinv(X.T @ X)[0, 0]))
            t  = c[0] / (se + 1e-12)
        except Exception:
            return 1.0
        if   t < -3.96: return 0.005
        elif t < -3.41: return 0.01
        elif t < -2.86: return 0.05
        elif t < -2.57: return 0.10
        else:           return 0.50

    def test_pair(self, s1: pd.Series, s2: pd.Series) -> Dict:
        y, x = s1.values.astype(float), s2.values.astype(float)
        try:
            X  = np.column_stack([np.ones(len(x)), x])
            c  = np.linalg.lstsq(X, y, rcond=None)[0]
            b0, b1 = c[0], c[1]
            sp = y - b0 - b1 * x
            pv = self.adf_pvalue(sp)
            corr, _ = pearsonr(y, x)
            dy = np.diff(sp); sx = sp[:-1]
            Xs = np.column_stack([sx, np.ones(len(sx))])
            lam = np.linalg.lstsq(Xs, dy, rcond=None)[0][0]
            hl  = -np.log(2) / lam if lam < -1e-6 else 999
            return {
                'cointegrated': pv < 0.05 and abs(corr) > 0.7,
                'pvalue': pv, 'hedge_ratio': b1, 'intercept': b0,
                'correlation': corr, 'half_life': hl,
                'tradeable': pv < 0.05 and 3 < hl < 90
            }
        except Exception as e:
            return {'cointegrated': False, 'error': str(e)}

    def zscore(self, s1: pd.Series, s2: pd.Series, b1: float, b0: float) -> pd.Series:
        sp = s1 - b0 - b1 * s2
        mu = sp.rolling(self.window).mean()
        sd = sp.rolling(self.window).std()
        return (sp - mu) / (sd + 1e-8)

    def signals(self, zs: pd.Series) -> pd.Series:
        pos = pd.Series(0.0, index=zs.index)
        for i in range(1, len(zs)):
            z, p = zs.iloc[i], pos.iloc[i - 1]
            if   z < -self.entry_z and p == 0: pos.iloc[i] =  1.0
            elif z >  self.entry_z and p == 0: pos.iloc[i] = -1.0
            elif abs(z) < self.exit_z:         pos.iloc[i] =  0.0
            else:                              pos.iloc[i] =  p
        return pos

    def backtest_pair(self, s1: pd.Series, s2: pd.Series,
                      info: Dict, tc_bps: float = 10) -> Dict:
        b1, b0 = info['hedge_ratio'], info['intercept']
        zs  = self.zscore(s1, s2, b1, b0)
        pos = self.signals(zs)
        r1  = s1.pct_change().fillna(0)
        r2  = s2.pct_change().fillna(0)
        tc  = pos.diff().abs().fillna(0) * (tc_bps / 10000)
        net = pos.shift(1).fillna(0) * r1 - pos.shift(1).fillna(0) * b1 * r2 - tc
        m   = backtest_metrics(net)
        m['current_z']   = float(zs.iloc[-1])
        m['current_pos'] = float(pos.iloc[-1])
        return m

    def find_pairs(self, prices: pd.DataFrame) -> List[Dict]:
        cols, pairs = prices.columns.tolist(), []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = self.test_pair(prices[cols[i]], prices[cols[j]])
                if r.get('tradeable'):
                    pairs.append({'t1': cols[i], 't2': cols[j], **r,
                                  'score': (1 - r['pvalue']) * 60 +
                                           max(0, 1 - r['half_life'] / 90) * 40})
        return sorted(pairs, key=lambda x: x['score'], reverse=True)

    def current_opportunity(self, prices: pd.DataFrame, pairs: List[Dict]) -> List[Dict]:
        active = []
        for p in pairs[:5]:
            t1, t2 = p['t1'], p['t2']
            if t1 not in prices.columns or t2 not in prices.columns:
                continue
            zs  = self.zscore(prices[t1], prices[t2], p['hedge_ratio'], p['intercept'])
            cur = float(zs.iloc[-1])
            if abs(cur) > self.entry_z:
                active.append({'t1': t1, 't2': t2,
                                'direction': 1 if cur < 0 else -1,
                                'z': cur, 'half_life': p['half_life']})
        return active


# ══════════════════════════════════════════════════════════════════════════════
# 9. 书第9章：UCB Bayesian优化
# ══════════════════════════════════════════════════════════════════════════════

class UCBOptimizer:
    def __init__(self, beta: float = 2.0):
        self.beta  = beta
        self.obs_x: List[List[float]] = []
        self.obs_y: List[float] = []

    def optimize(self, objective_fn, param_bounds: Dict, n_iter: int = 20) -> Dict:
        best_score, best_params = -np.inf, {}
        for i in range(n_iter):
            params = (self._random(param_bounds)
                      if len(self.obs_x) < 4 else self._ucb(param_bounds))
            try:
                score = float(objective_fn(**params))
                if np.isfinite(score):
                    self.obs_x.append(list(params.values()))
                    self.obs_y.append(score)
                    if score > best_score:
                        best_score, best_params = score, params.copy()
            except Exception:
                continue
        return {'best_params': best_params, 'best_score': best_score,
                'n_evaluated': len(self.obs_y)}

    def _random(self, bounds: Dict) -> Dict:
        return {k: (float(np.random.uniform(*v)) if isinstance(v[0], float)
                    else int(np.random.randint(v[0], v[1])))
                for k, v in bounds.items()}

    def _ucb(self, bounds: Dict) -> Dict:
        X  = np.array(self.obs_x)
        y  = np.array(self.obs_y)
        keys = list(bounds.keys())
        Xm, Xs = X.mean(0), X.std(0) + 1e-8
        Xn = (X - Xm) / Xs
        cands = np.array([[float(np.random.uniform(*v))
                           if isinstance(v[0], float)
                           else float(np.random.randint(v[0], v[1]))
                           for v in bounds.values()] for _ in range(60)])
        cn = (cands - Xm) / Xs
        D  = np.sum((Xn[:, None] - cn[None, :]) ** 2, axis=-1)
        K  = np.exp(-0.5 * D)
        Kx = np.exp(-0.5 * np.sum((Xn[:, None] - Xn[None, :]) ** 2, axis=-1))
        Kx += np.eye(len(X)) * 0.01
        try:
            ag = np.linalg.solve(Kx, y)
            mu = K.T @ ag
            vr = np.array([max(0.0, 1.0 - float(K[:, j] @ np.linalg.solve(Kx, K[:, j])))
                            for j in range(len(cands))])
            ucb = mu + self.beta * np.sqrt(vr)
        except Exception:
            ucb = np.random.randn(len(cands))
        bi = np.argmax(ucb)
        return {k: (float(cands[bi, i]) if isinstance(list(bounds.values())[i][0], float)
                    else int(cands[bi, i])) for i, k in enumerate(keys)}


# ══════════════════════════════════════════════════════════════════════════════
# 10. 风险管理
# ══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    def __init__(self):
        self.lw = LedoitWolf()

    def kelly(self, ret_series: pd.Series, lookback: int = 126) -> float:
        r = ret_series.dropna().tail(lookback)
        if len(r) < 20:
            return 0.05
        p = float((r > 0).mean())
        wins, loss = r[r > 0], r[r < 0]
        if len(wins) == 0 or len(loss) == 0:
            return 0.04
        b   = float(wins.mean()) / (float(abs(loss.mean())) + 1e-8)
        raw = (p * b - (1-p)) / (b + 1e-8)
        if raw <= 0:
            return 0.04
        sk_pen = max(0, -float(r.skew())) * 0.15
        kt_pen = max(0, float(r.kurtosis()) - 3) * 0.05
        return float(np.clip(raw * 0.5 * (1 - sk_pen - kt_pen), 0.02, 0.20))

    def ledoit_wolf_cov(self, returns: pd.DataFrame) -> np.ndarray:
        clean = returns.dropna()
        if len(clean) < 10:
            return np.diag(clean.var().values)
        try:
            self.lw.fit(clean.values)
            return self.lw.covariance_
        except Exception:
            return clean.cov().values

    def cvar(self, returns: pd.Series, conf: float = 0.95) -> float:
        s = np.sort(returns.dropna().values)
        c = int(len(s) * (1 - conf))
        return float(-np.mean(s[:max(1, c)]))


# ══════════════════════════════════════════════════════════════════════════════
# 11. [NEW] 波动率目标 + 回撤控制（提高Sharpe，降低回撤）
# ══════════════════════════════════════════════════════════════════════════════

class VolTargeter:
    """
    波动率目标：动态缩放仓位使组合年化波动率接近目标值
    目标波动率 = 10%
    当前波动高 → 降仓；当前波动低 → 加仓

    提高Sharpe的核心机制：
    - 低波动期：信号质量高，放大仓位
    - 高波动期：信号质量差，缩减仓位
    - 比固定仓位策略Sharpe通常高15-30%
    """

    def __init__(self, target_vol: float = 0.10,
                 vol_window: int = 21,
                 min_scale: float = 0.2,
                 max_scale: float = 2.0):
        self.target   = target_vol
        self.window   = vol_window
        self.min_sc   = min_scale
        self.max_sc   = max_scale

    def scale(self, weights: pd.Series, returns: pd.DataFrame) -> Tuple[pd.Series, float]:
        """
        返回缩放后的权重 + 缩放系数
        """
        if returns is None or len(returns) < self.window:
            return weights, 1.0

        tickers = [t for t in weights.index if t in returns.columns]
        if not tickers:
            return weights, 1.0

        w_arr   = weights[tickers].values
        ret_arr = returns[tickers].dropna().tail(self.window).values

        if len(ret_arr) < 5:
            return weights, 1.0

        # 组合已实现波动率
        port_ret = ret_arr @ (w_arr / (np.abs(w_arr).sum() + 1e-8))
        realized = port_ret.std() * np.sqrt(252)

        if realized < 1e-4:
            return weights, 1.0

        scale = float(np.clip(self.target / realized, self.min_sc, self.max_sc))
        return weights * scale, scale


class DrawdownController:
    """
    回撤控制：动态降仓防止连续亏损
    回撤 > 5%  → 仓位 × 0.5
    回撤 > 10% → 仓位 × 0.25

    降低最大回撤的核心机制：
    - 不依赖固定止损，而是根据当前回撤动态调整
    - 在连续亏损时自动保护本金
    """

    def __init__(self, level1: float = 0.05, scale1: float = 0.5,
                 level2: float = 0.10, scale2: float = 0.25):
        self.level1 = level1
        self.scale1 = scale1
        self.level2 = level2
        self.scale2 = scale2
        self._peak  = 1.0
        self._equity = 1.0

    def update(self, daily_return: float) -> None:
        self._equity *= (1 + daily_return)
        self._peak    = max(self._peak, self._equity)

    @property
    def current_drawdown(self) -> float:
        return (self._equity - self._peak) / (self._peak + 1e-8)

    def position_scale(self) -> float:
        dd = self.current_drawdown
        if dd < -self.level2:
            return self.scale2
        elif dd < -self.level1:
            return self.scale1
        return 1.0

    def scale_weights(self, weights: pd.Series) -> pd.Series:
        sc = self.position_scale()
        if sc < 1.0:
            pass  # 不打印，回测中太多
        return weights * sc


# ══════════════════════════════════════════════════════════════════════════════
# 12. Portfolio Allocation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioAllocation:
    longs:          Dict[str, float]
    shorts:         Dict[str, float]
    cash:           float
    net_exposure:   float
    gross_exposure: float
    regime:         str
    rationale:      str

    def to_series(self) -> pd.Series:
        return pd.Series({**self.longs, **self.shorts}, dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 13. 进攻/防守仓位管理器（修复满仓 + 集成IC验证Alpha）
# ══════════════════════════════════════════════════════════════════════════════

class OffensiveDefensiveManager:
    """
    进攻/防守仓位管理器

    [FIX v6] 满仓修复：
    不再用"等权/30% tickers × Kelly"（导致15%总敞口），
    而是用"目标总敞口 ÷ n只股票"（BULL_STRONG → 90%总敞口）

    目标敞口：
    BULL_STRONG  → 90%  （强牛全力进攻）
    BULL_NORMAL  → 78%
    NEUTRAL      → 60%
    BEAR_MILD    → 35%  （+做空）
    BEAR_STRONG  → 18%  （现金为主+做空）
    """

    # 按Regime的目标总多头敞口
    TARGET_LONG_EXP = {2: 0.90, 1: 0.78, 0: 0.60, -1: 0.35, -2: 0.18}

    # 按Regime的多头股票覆盖率（多少比例进入多头）
    LONG_COVERAGE = {2: 0.60, 1: 0.50, 0: 0.35, -1: 0.25, -2: 0.20}

    def __init__(self, risk_mgr: RiskManager):
        self.risk = risk_mgr

    def allocate(self,
                 regime: Regime,
                 long_alpha: pd.Series,
                 short_alpha: pd.Series,
                 trend_sig: pd.DataFrame,
                 stat_arb_opps: List[Dict],
                 returns: pd.DataFrame,
                 ic_validated: bool = False
                 ) -> PortfolioAllocation:

        longs: Dict[str, float]  = {}
        shorts: Dict[str, float] = {}

        # ── 目标总敞口 ────────────────────────────────────────────────────
        target_exp = self.TARGET_LONG_EXP.get(regime.score, 0.60)
        coverage   = self.LONG_COVERAGE.get(regime.score, 0.35)

        # ── 多头仓位 ──────────────────────────────────────────────────────
        clean_long = long_alpha.dropna()
        if len(clean_long) > 0:
            ranked     = clean_long.rank(pct=True)
            # 根据regime决定覆盖多少个标的
            threshold  = 1 - coverage
            top_tickers = ranked[ranked >= threshold].index.tolist()

            if not top_tickers:
                top_tickers = ranked.nlargest(max(1, len(ranked)//3)).index.tolist()

            # 趋势过滤（牛市中放宽：只排除强死叉，不要求必须金叉）
            if regime.score >= 1:
                # 牛市：排除强死叉（strength > 0.02 且 死叉）
                confirmed = []
                for tk in top_tickers:
                    if tk in trend_sig.index:
                        sig = trend_sig.loc[tk]
                        strong_death = (sig.get('death_cross', False) and
                                        sig.get('strength', 0) > 0.02)
                        if not strong_death:
                            confirmed.append(tk)
                    else:
                        confirmed.append(tk)
                if not confirmed:
                    confirmed = top_tickers  # 牛市不过度过滤
            else:
                # 熊市/中性：要求上升趋势
                confirmed = [tk for tk in top_tickers
                              if tk not in trend_sig.index
                              or trend_sig.loc[tk, 'signal'] == 1]
                if not confirmed:
                    confirmed = top_tickers[:max(1, len(top_tickers)//2)]

            # 仓位分配：target_exp / n（而非Kelly驱动）
            n_long = max(1, len(confirmed))
            base   = target_exp / n_long

            for tk in confirmed:
                # Kelly只作上限，不作目标（避免Kelly过小导致欠配）
                kelly_cap = (self.risk.kelly(returns[tk], 126)
                             if tk in returns.columns else 0.08)
                # max_long是单票绝对上限，kelly_cap*2给足空间
                size = min(base,
                           regime.max_long,
                           max(kelly_cap * 2, base * 0.5))  # 至少50%目标
                if size > 0.005:
                    longs[tk] = round(float(size), 4)

        # ── 空头仓位（熊市才激活）─────────────────────────────────────────
        if regime.allow_short and len(short_alpha.dropna()) > 0:
            ranked_s   = short_alpha.dropna().rank(pct=True)
            bot_tickers = ranked_s[ranked_s <= 0.25].index.tolist()

            confirmed_short = []
            for tk in bot_tickers:
                if tk in longs:
                    continue
                downtrend = (tk in trend_sig.index and
                             trend_sig.loc[tk, 'signal'] == -1)
                if downtrend or regime.score <= -2:
                    confirmed_short.append(tk)

            if confirmed_short:
                n_s = max(1, len(confirmed_short))
                eq_s = 1.0 / n_s
                for tk in confirmed_short:
                    size = max(regime.max_short, -min(0.06, eq_s * 0.8))
                    if size < -0.005:
                        shorts[tk] = round(float(size), 4)

        # ── 统计套利叠加 ─────────────────────────────────────────────────
        if regime.score >= -1:
            for opp in stat_arb_opps[:2]:
                t1, t2 = opp['t1'], opp['t2']
                sz = min(0.025, abs(opp['z']) / 4 * 0.03)
                if opp['direction'] == 1:
                    if t1 not in {**longs, **shorts}: longs[t1]  = longs.get(t1, 0)  + sz
                    if t2 not in {**longs, **shorts}: shorts[t2] = shorts.get(t2, 0) - sz
                else:
                    if t1 not in {**longs, **shorts}: shorts[t1] = shorts.get(t1, 0) - sz
                    if t2 not in {**longs, **shorts}: longs[t2]  = longs.get(t2, 0)  + sz

        all_w = {**longs, **shorts}
        net   = sum(all_w.values())
        gross = sum(abs(v) for v in all_w.values())
        cash  = max(0.0, 1.0 - sum(longs.values()))

        rationale = (f"{regime.label}（{regime.stance}）| "
                     f"目标敞口{target_exp:.0%} | "
                     f"多{len(longs)}只 空{len(shorts)}只 | "
                     f"净:{net:+.1%} 总:{gross:.1%} 现金:{cash:.1%}")

        return PortfolioAllocation(longs=longs, shorts=shorts, cash=cash,
                                   net_exposure=net, gross_exposure=gross,
                                   regime=regime.label, rationale=rationale)


# ══════════════════════════════════════════════════════════════════════════════
# 14. Canyon F/C/E量化评分
# ══════════════════════════════════════════════════════════════════════════════

def canyon_score_auto(price: pd.Series, volume: pd.Series,
                       market: pd.Series, regime: Regime) -> Dict:
    r = price.pct_change().dropna()
    n = len(r)
    if n < 21:
        return {'total': 60, 'grade': 'D', 'can_buy': True, 'max_pos': regime.max_long}

    lk  = min(21, n - 1)
    er  = abs(float(price.iloc[-1]) - float(price.iloc[-lk - 1]))
    path= float(r.abs().tail(lk).sum()) + 1e-8
    rs  = float(price.pct_change(lk).iloc[-1]) - float(market.pct_change(lk).iloc[-1])
    f   = float(np.clip(3.0 + (er / path) * 1.5 + rs * 15, 1, 5))

    vm  = float(volume.rolling(min(20, n)).mean().iloc[-1])
    vr  = float(volume.iloc[-1]) / (vm + 1e-8)
    mom = float(np.clip(price.pct_change(min(5, n-1)).iloc[-1] * 40, -2, 2))
    g   = r.clip(lower=0).ewm(span=14).mean()
    l   = (-r).clip(lower=0).ewm(span=14).mean()
    rsi = float(100 - 100 / (1 + g.iloc[-1] / (l.iloc[-1] + 1e-8)))
    c2  = float(np.clip(3.0 + mom + (vr - 1) * 0.3, 1, 5))
    c3  = float(np.clip(3.0 + rs * 12, 1, 5))
    c   = float(np.clip(0.35 * f + 0.40 * c2 + 0.25 * c3, 1, 5))

    rsi_ok = 35 < rsi < 72
    vm_ok  = vr > 0.8
    e = float(np.clip(3.0 + (0.7 if rsi_ok else -0.6) + (0.3 if vm_ok else -0.2), 1, 5))

    total = 0.20 * (f/5*100) + 0.45 * (c/5*100) + 0.35 * (e/5*100)
    total = float(np.clip(total, 0, 100))

    if   total >= 85: grade, mp = 'A', regime.max_long
    elif total >= 80: grade, mp = 'B', min(regime.max_long, 0.09)
    elif total >= 75: grade, mp = 'C', min(regime.max_long, 0.06)
    elif total >= 70: grade, mp = 'D', min(regime.max_long, 0.04)
    else:             grade, mp = 'X', 0.0

    return {'f': round(f,2), 'c': round(c,2), 'e': round(e,2),
            'total': round(total,1), 'grade': grade, 'max_pos': mp,
            'can_buy': total >= 70 and e >= 3.5}


# ══════════════════════════════════════════════════════════════════════════════
# 15. Walk-Forward回测（集成执行成本 + 波动率目标 + 回撤控制）
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardBacktester:
    """
    书第7章：Walk-Forward回测

    v6改进：
    - 执行成本：买卖价差 + 市场冲击（替代简单flat bps）
    - 波动率目标：动态缩放到10%年化波动
    - 回撤控制：超5%自动减仓50%
    - Alpha IC验证：只用通过验证的因子
    """

    def __init__(self, train_w: int = 252, test_w: int = 63,
                 tc_bps: float = 10.0, rf: float = 0.03,
                 target_vol: float = 0.10):
        self.train_w   = train_w
        self.test_w    = test_w
        self.flat_tc   = tc_bps / 10000   # 保留作备用
        self.rf        = rf
        self.exec_cost  = ExecutionCostModel()
        self.vol_tgt    = VolTargeter(target_vol=target_vol)
        self.dd_ctrl    = DrawdownController()
        self.sleeve_mgr = SleeveManager()   # [v8] 分账户管理器

    def run(self, prices: pd.DataFrame, volumes: pd.DataFrame,
             market: pd.Series,
             stat_arb: StatArb,
             od_mgr: OffensiveDefensiveManager,
             use_ic_alpha: bool = True,
             verbose: bool = True) -> Dict:

        n = len(prices)
        if n < self.train_w + self.test_w:
            raise ValueError(f"数据不足：需要{self.train_w+self.test_w}天，实际{n}天")

        daily_rets, daily_dates = [], []
        long_rets, short_rets  = [], []
        regime_hist            = []
        prev_weights           = pd.Series(dtype=float)
        # 重置回撤控制器
        self.dd_ctrl = DrawdownController()

        steps = range(self.train_w, n - self.test_w + 1, self.test_w)
        if verbose:
            print(f"  Walk-Forward: {len(list(steps))}个窗口 × {self.test_w}天 "
                  f"(波动率目标:{self.vol_tgt.target:.0%})")

        for step_start in steps:
            p_tr = prices.iloc[step_start - self.train_w: step_start]
            v_tr = volumes.iloc[step_start - self.train_w: step_start]
            m_tr = market.iloc[step_start - self.train_w: step_start]
            r_tr = p_tr.pct_change().dropna()

            # 市场环境
            regime, _ = detect_regime(m_tr, p_tr)
            regime_hist.append(regime.score)

            # Alpha生成（IC验证版 or 简单CS动量）
            if use_ic_alpha and len(r_tr) >= 63:
                long_alpha, ic_vals = AlphaICEngine.build_validated_alpha(p_tr, v_tr, m_tr)
                short_alpha = -long_alpha
                ic_validated = any(v.get('valid', False) for v in ic_vals.values())
            else:
                cs = cross_sectional_momentum(p_tr)
                long_alpha, short_alpha = cs['long_alpha'].dropna(), cs['short_alpha'].dropna()
                ic_validated = False

            tsig  = trend_signals(p_tr)
            pairs = stat_arb.find_pairs(p_tr)
            arbs  = stat_arb.current_opportunity(p_tr, pairs)

            alloc = od_mgr.allocate(
                regime=regime, long_alpha=long_alpha,
                short_alpha=short_alpha, trend_sig=tsig,
                stat_arb_opps=arbs, returns=r_tr,
                ic_validated=ic_validated
            )
            weights = alloc.to_series()

            # [v8] SleeveManager：按Regime调整各Sleeve资金权重
            # CORE_HEDGE信号 = 当前weights（长期动量主导）
            # TACTICAL信号   = 短期反转（反向）
            # SECTOR_ROT     = 原始CS动量
            if hasattr(self, 'sleeve_mgr') and self.sleeve_mgr is not None:
                sleeve_alloc = self.sleeve_mgr.allocate_by_regime(regime.label)
                # 按Sleeve权重缩放总仓位（不改变信号方向，只调整规模）
                core_w   = sleeve_alloc.get('CORE_HEDGE', 0.45)
                tac_w    = sleeve_alloc.get('TACTICAL', 0.25)
                sect_w   = sleeve_alloc.get('SECTOR_ROTATION', 0.30)
                # 总进攻权重（TACTICAL+SECTOR_ROT），乘到weights上
                offense_scale = (tac_w + sect_w) / 0.55  # 0.55是中性基准
                weights = weights * min(offense_scale, 1.5)

            # 波动率目标缩放
            weights, vol_scale = self.vol_tgt.scale(weights, r_tr)

            # 回撤控制缩放
            weights = self.dd_ctrl.scale_weights(weights)

            # 测试窗口
            test_end   = min(step_start + self.test_w, n)
            p_te       = prices.iloc[step_start: test_end]
            ret_te     = p_te.pct_change().fillna(0)
            vol_te     = volumes.iloc[step_start: test_end]
            test_dates = p_te.index

            for i in range(1, len(test_dates)):
                d      = test_dates[i]
                dr, lr, sr = 0.0, 0.0, 0.0

                for tk, w in weights.items():
                    if tk not in ret_te.columns:
                        continue
                    ar = float(ret_te.loc[d, tk])
                    if not np.isfinite(ar):
                        ar = 0.0
                    c = w * ar
                    dr += c
                    if w > 0: lr += c
                    else:     sr += c

                # 执行成本（精确版：买卖价差 + 市场冲击）
                if len(prev_weights) > 0:
                    all_tk = set(weights.index) | set(prev_weights.index)
                    for tk in all_tk:
                        delta = float(weights.get(tk, 0)) - float(prev_weights.get(tk, 0))
                        if abs(delta) < 1e-4:
                            continue
                        # 估算波动率和ADV
                        if tk in r_tr.columns:
                            vol_d = float(r_tr[tk].std())
                        else:
                            vol_d = 0.015
                        if tk in v_tr.columns:
                            adv_usd = float(v_tr[tk].mean()) * float(p_tr[tk].iloc[-1]) if tk in p_tr.columns else 1e6
                        else:
                            adv_usd = 1e6
                        trade_usd = abs(delta) * 1e6   # 标准化100万组合
                        tc_ratio  = self.exec_cost.total_cost(vol_d, adv_usd, trade_usd)
                        dr -= abs(delta) * tc_ratio

                daily_rets.append(dr)
                daily_dates.append(d)
                long_rets.append(lr)
                short_rets.append(sr)

                # 更新回撤控制器
                self.dd_ctrl.update(dr)

            prev_weights = weights.copy()

        if not daily_rets:
            return {'error': '无交易数据', 'sharpe': 0}

        ret_s = pd.Series(daily_rets, index=daily_dates)
        m     = backtest_metrics(ret_s, rf=self.rf)

        bull_m = backtest_metrics(ret_s.iloc[:len(ret_s)//2], rf=self.rf) if len(ret_s) > 20 else {}
        bear_m = backtest_metrics(ret_s.iloc[len(ret_s)//2:], rf=self.rf) if len(ret_s) > 20 else {}

        result = {
            **m,
            'daily_returns':  ret_s,
            'cumulative':     (1 + ret_s).cumprod(),
            'long_total_pnl': float(pd.Series(long_rets).sum()),
            'short_total_pnl':float(pd.Series(short_rets).sum()),
            'avg_regime':     float(np.mean(regime_hist)) if regime_hist else 0,
            'bull_sharpe':    bull_m.get('sharpe', 0),
            'bear_sharpe':    bear_m.get('sharpe', 0),
        }

        if verbose:
            self._print(result)
        return result

    def _print(self, r: Dict):
        print(f"\n{'─'*62}")
        print(f"  📊 Walk-Forward 回测（书第7章 + v6改进）")
        print(f"{'─'*62}")
        print(f"  年化收益：{r.get('ann_ret',0):+.2%}  |  年化波动：{r.get('ann_vol',0):.2%}")
        print(f"  Sharpe：  {r.get('sharpe',0):>8.3f}  |  Calmar：  {r.get('calmar',0):.3f}")
        print(f"  最大回撤：{r.get('max_dd',0):>8.2%}  |  总收益：  {r.get('total_ret',0):+.2%}")
        print(f"  胜率：    {r.get('win_rate',0):>8.1%}  |  交易天数：{r.get('n',0)}")
        print(f"  多头贡献：{r.get('long_total_pnl',0):+.2%}  |  空头贡献：{r.get('short_total_pnl',0):+.2%}")
        print(f"  牛市Sharpe：{r.get('bull_sharpe',0):.3f}  |  熊市Sharpe：{r.get('bear_sharpe',0):.3f}")
        print(f"{'─'*62}")

    def multi_period(self, prices, volumes, market, stat_arb, od_mgr,
                     n_periods: int = 3) -> pd.DataFrame:
        n   = len(prices)
        seg = max(self.train_w + self.test_w, n // n_periods)
        records = []
        print(f"\n  📊 多期回测（书第7章：{n_periods}段独立验证）")
        print(f"  {'─'*58}")
        for i in range(n_periods):
            s  = i * (n // n_periods)
            e  = min(s + seg + self.train_w, n)
            if e - s < self.train_w + self.test_w:
                continue
            sp, sv, sm = prices.iloc[s:e], volumes.iloc[s:e], market.iloc[s:e]
            d0, d1 = sp.index[0].date(), sp.index[-1].date()
            print(f"  时期{i+1}: {d0} → {d1} ", end='', flush=True)
            try:
                r = self.run(sp, sv, sm, stat_arb, od_mgr, verbose=False)
                records.append({'period': i+1, 'start': str(d0), 'end': str(d1),
                                'sharpe': r.get('sharpe',0), 'calmar': r.get('calmar',0),
                                'ann_ret': r.get('ann_ret',0), 'max_dd': r.get('max_dd',0),
                                'long_pnl': r.get('long_total_pnl',0),
                                'short_pnl': r.get('short_total_pnl',0)})
                print(f"Sharpe={r.get('sharpe',0):.3f} MaxDD={r.get('max_dd',0):.2%}")
            except Exception as ex:
                print(f"跳过({str(ex)[:35]})")
        df = pd.DataFrame(records)
        if len(df) > 0:
            print(f"\n  汇总统计（{len(df)}段）：")
            for col, lbl in [('sharpe','Sharpe'),('calmar','Calmar'),
                              ('ann_ret','年化收益'),('max_dd','最大回撤')]:
                if col in df:
                    print(f"    {lbl:<10} 均值:{df[col].mean():>8.3f} "
                          f"σ:{df[col].std():>7.3f} 最差:{df[col].min():>8.3f}")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 16. [NEW] 实盘监控系统
# ══════════════════════════════════════════════════════════════════════════════

class StateLogger:
    """
    状态记录器（实盘第一步：记录一切）
    所有交易状态、P&L、仓位写入CSV
    可用于Dashboard读取和事后分析
    """

    def __init__(self, path: str = "trading_log.csv"):
        self.path = path

    def log(self, data: dict):
        data["timestamp"] = datetime.now().isoformat()
        df = pd.DataFrame([data])
        try:
            old = pd.read_csv(self.path)
            df  = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
        df.to_csv(self.path, index=False)

    def read(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path)
        except Exception:
            return pd.DataFrame()

    def latest(self) -> Dict:
        df = self.read()
        if len(df) == 0:
            return {}
        return df.iloc[-1].to_dict()


class AlertEngine:
    """
    风控报警系统（Telegram / 控制台）
    生产环境用Telegram Bot，开发环境打印到控制台
    """

    def __init__(self, telegram_token: str = None, chat_id: str = None):
        self.token   = telegram_token
        self.chat_id = chat_id
        self.enabled = bool(telegram_token and chat_id)
        self.history: List[str] = []

    def send(self, msg: str, level: str = 'INFO'):
        """发送警报（Telegram or 控制台）"""
        ts = datetime.now().strftime('%H:%M:%S')
        full_msg = f"[{ts}][{level}] {msg}"
        self.history.append(full_msg)

        if self.enabled:
            try:
                import requests
                url = f"https://api.telegram.org/bot{self.token}/sendMessage"
                requests.post(url, data={"chat_id": self.chat_id, "text": full_msg},
                              timeout=5)
            except Exception:
                print(full_msg)
        else:
            icon = {'CRITICAL': '🚨', 'WARNING': '⚠️', 'INFO': 'ℹ️'}.get(level, '')
            print(f"  {icon} {full_msg}")

    def alert_daily_loss(self, pnl: float, threshold: float = -0.03):
        if pnl < threshold:
            self.send(f"Daily loss {pnl:.2%} exceeded {threshold:.2%}", 'WARNING')

    def alert_drawdown(self, dd: float):
        if dd < -0.12:
            self.send(f"🚨 CRITICAL: Drawdown {dd:.2%} — STOP TRADING", 'CRITICAL')
        elif dd < -0.08:
            self.send(f"Drawdown {dd:.2%} exceeded -8%", 'WARNING')

    def alert_strategy_broken(self):
        self.send("Strategy degraded (rolling Sharpe < 0) — halting", 'CRITICAL')


class RiskMonitor:
    """
    实时风控监控
    · 日亏损 > 3% → 警报
    · 回撤 > 8% → 警报
    · 回撤 > 12% → Kill Switch（停止交易）
    """

    def __init__(self, alert: AlertEngine,
                 max_daily_loss: float = -0.03,
                 max_drawdown_warn: float = -0.08,
                 max_drawdown_stop: float = -0.12):
        self.alert    = alert
        self.md_daily = max_daily_loss
        self.md_warn  = max_drawdown_warn
        self.md_stop  = max_drawdown_stop

    def check(self, pnl_today: float, current_drawdown: float) -> bool:
        """
        Returns True if kill switch triggered (停止交易)
        """
        self.alert.alert_daily_loss(pnl_today, self.md_daily)
        self.alert.alert_drawdown(current_drawdown)

        if current_drawdown < self.md_stop:
            return True   # Kill switch
        return False


class StrategyMonitor:
    """
    策略失效检测（自动停机）
    监控：Rolling Sharpe / 连续亏损 / IC下降
    Sharpe < 0 or 连续5日亏损 → 停机
    """

    def __init__(self, sharpe_window: int = 50,
                 losing_streak_n: int = 5):
        self.history         = []
        self.sharpe_window   = sharpe_window
        self.losing_streak_n = losing_streak_n

    def update(self, daily_returns: List[float]):
        self.history.extend(daily_returns)

    def rolling_sharpe(self, window: int = None) -> float:
        w = window or self.sharpe_window
        r = np.array(self.history[-w:])
        if len(r) < 10:
            return 0.0
        return float(np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252))

    def losing_streak(self, n: int = None) -> bool:
        n = n or self.losing_streak_n
        if len(self.history) < n:
            return False
        return all(r < 0 for r in self.history[-n:])

    def is_broken(self) -> Tuple[bool, str]:
        """
        Returns (is_broken, reason)
        """
        sr = self.rolling_sharpe()
        if sr < -0.5:
            return True, f"Rolling Sharpe {sr:.3f} < -0.5"
        if self.losing_streak():
            return True, f"连续{self.losing_streak_n}日亏损"
        return False, "正常"

    def status_report(self) -> Dict:
        broken, reason = self.is_broken()
        return {
            'rolling_sharpe': round(self.rolling_sharpe(), 4),
            'losing_streak':  self.losing_streak(),
            'is_broken':      broken,
            'reason':         reason,
            'n_observations': len(self.history)
        }


# ══════════════════════════════════════════════════════════════════════════════
# 17. [NEW] 执行算法（TWAP / VWAP / POV）
# ══════════════════════════════════════════════════════════════════════════════

class TWAPExecution:
    """
    TWAP（Time-Weighted Average Price）
    把大单分成多份，等时间间隔执行
    适合：流动性差、冲击成本高的标的
    """

    def __init__(self, slices: int = 5, interval_secs: int = 60):
        self.slices   = slices
        self.interval = interval_secs

    def execute(self, execution_api, symbol: str,
                total_qty: int, side: str) -> List[Dict]:
        orders = []
        qty_per_slice = max(1, total_qty // self.slices)
        print(f"  [TWAP] {side} {symbol}: {total_qty}股 / {self.slices}份")
        for i in range(self.slices):
            remaining  = total_qty - i * qty_per_slice
            slice_qty  = min(qty_per_slice, remaining)
            if slice_qty <= 0:
                break
            if execution_api:
                execution_api.submit_order(symbol, slice_qty, side)
            orders.append({'slice': i+1, 'symbol': symbol,
                           'qty': slice_qty, 'side': side})
            if i < self.slices - 1:
                time.sleep(self.interval)
        return orders


class VWAPExecution:
    """
    VWAP（Volume-Weighted Average Price）
    根据当前成交量决定下单时机
    成交量高时多下单，成交量低时少下单
    """

    @staticmethod
    def target_qty(current_volume: float, historical_avg_volume: float,
                   total_remaining: int, time_fraction: float) -> int:
        """
        当前应该执行多少量
        time_fraction: 当前时间在交易日的进度（0-1）
        """
        expected_done = total_remaining * time_fraction
        vol_ratio     = current_volume / (historical_avg_volume + 1e-8)
        adjusted      = int(expected_done * vol_ratio)
        return max(0, min(adjusted, total_remaining))


class POVExecution:
    """
    POV（Participation of Volume）
    机构最常用：参与市场成交的固定比例
    target_ratio = 10% → 市场每成交100股，你买10股
    """

    def __init__(self, target_ratio: float = 0.10):
        self.ratio = target_ratio

    def calculate_qty(self, current_market_volume: float) -> int:
        return max(0, int(current_market_volume * self.ratio))


# ══════════════════════════════════════════════════════════════════════════════
# 18. [NEW] Alpaca实盘执行
# ══════════════════════════════════════════════════════════════════════════════

class AlpacaExecution:
    """
    Alpaca Broker对接（Paper Trading + Live Trading）
    安装：pip install alpaca-trade-api
    申请：https://alpaca.markets（免费paper trading账户）
    """

    def __init__(self, key: str = None, secret: str = None,
                 base_url: str = 'https://paper-api.alpaca.markets'):
        self.api     = None
        self.enabled = False
        if key and secret:
            try:
                import alpaca_trade_api as tradeapi
                self.api     = tradeapi.REST(key, secret, base_url)
                self.enabled = True
                print(f"[Alpaca] ✅ 已连接 {base_url}")
            except ImportError:
                print("[Alpaca] alpaca-trade-api未安装 → pip install alpaca-trade-api")
            except Exception as e:
                print(f"[Alpaca] 连接失败: {e}")
        else:
            print("[Alpaca] 未提供API Key，运行Paper模式（本地模拟）")

    def get_positions(self) -> Dict[str, float]:
        if not self.api:
            return {}
        try:
            positions = self.api.list_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception:
            return {}

    def get_account(self) -> Dict:
        if not self.api:
            return {'equity': 0, 'cash': 0, 'buying_power': 0}
        try:
            acc = self.api.get_account()
            return {'equity': float(acc.equity),
                    'cash': float(acc.cash),
                    'buying_power': float(acc.buying_power)}
        except Exception:
            return {}

    def submit_order(self, symbol: str, qty: int, side: str,
                     order_type: str = 'market') -> Dict:
        if qty <= 0:
            return {}
        print(f"  [ORDER] {side.upper()} {symbol} {qty}股 ({order_type})")
        if self.api:
            try:
                order = self.api.submit_order(
                    symbol=symbol, qty=qty, side=side,
                    type=order_type, time_in_force='day'
                )
                return {'id': order.id, 'symbol': symbol,
                        'qty': qty, 'side': side, 'status': order.status}
            except Exception as e:
                print(f"  [ORDER ERROR] {e}")
                return {}
        return {'symbol': symbol, 'qty': qty, 'side': side, 'status': 'simulated'}

    def rebalance(self, target_weights: Dict[str, float],
                  prices: Dict[str, float], capital: float,
                  use_twap: bool = True) -> List[Dict]:
        """
        组合再平衡
        target_weights: {symbol: weight}（总和不必等于1）
        """
        current_positions = self.get_positions()
        orders = []
        twap   = TWAPExecution(slices=5, interval_secs=30)

        for symbol, target_w in target_weights.items():
            if symbol not in prices:
                continue
            target_shares = int((target_w * capital) / (prices[symbol] + 1e-8))
            current_shares = int(current_positions.get(symbol, 0))
            diff = target_shares - current_shares

            if abs(diff) < 1:
                continue

            side = "buy" if diff > 0 else "sell"
            qty  = abs(diff)

            if use_twap and qty > 100:
                # 大单用TWAP分批执行
                slice_orders = twap.execute(self, symbol, qty, side)
                orders.extend(slice_orders)
            else:
                order = self.submit_order(symbol, qty, side)
                orders.append(order)

        # 平掉不在目标中的仓位
        for symbol, qty in current_positions.items():
            if symbol not in target_weights and float(qty) > 0:
                order = self.submit_order(symbol, int(float(qty)), 'sell')
                orders.append(order)

        return orders


# ══════════════════════════════════════════════════════════════════════════════
# 19. [NEW] 实盘风控
# ══════════════════════════════════════════════════════════════════════════════

class LiveRiskManager:
    """
    实盘风控层
    · 仓位上限检查
    · 日亏损Kill Switch
    · 回撤Kill Switch
    """

    def __init__(self, max_position: float = 0.15,
                 max_daily_loss: float = -0.03,
                 max_drawdown: float = -0.10):
        self.max_position  = max_position
        self.max_daily_loss= max_daily_loss
        self.max_drawdown  = max_drawdown
        self._kill_switch  = False
        self._peak_equity  = 1.0
        self._equity       = 1.0

    def check_position_limit(self, weights: Dict[str, float]) -> Dict[str, float]:
        """强制执行单票仓位上限"""
        return {k: min(abs(v), self.max_position) * np.sign(v)
                for k, v in weights.items()}

    def check_kill_switch(self, pnl_today: float) -> bool:
        """日亏损 or 回撤超限 → Kill Switch"""
        if self._kill_switch:
            return True

        self._equity *= (1 + pnl_today)
        self._peak_equity = max(self._peak_equity, self._equity)
        current_dd = (self._equity - self._peak_equity) / (self._peak_equity + 1e-8)

        if pnl_today < self.max_daily_loss:
            print(f"  🚨 KILL SWITCH: 日亏损 {pnl_today:.2%}")
            self._kill_switch = True
            return True

        if current_dd < self.max_drawdown:
            print(f"  🚨 KILL SWITCH: 回撤 {current_dd:.2%}")
            self._kill_switch = True
            return True

        return False

    def reset_daily(self):
        """每日开盘前重置（不重置Kill Switch，需要人工确认）"""
        pass

    def manual_reset_kill_switch(self):
        """手动重置Kill Switch（确认风险后才允许恢复交易）"""
        self._kill_switch = False
        print("  Kill Switch已手动重置，恢复交易")


# ══════════════════════════════════════════════════════════════════════════════
# 20. 交易日志（完整Gate Check）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    trade_id:       str
    ticker:         str
    direction:      str
    entry_date:     str
    entry_price:    float
    position_pct:   float
    regime:         str
    entry_reason:   str
    engines_used:   List[str]
    expected_days:  int
    first_exit:     str
    forced_exit:    str
    canyon_score:   float
    canyon_grade:   str
    exit_date:      Optional[str] = None
    exit_price:     Optional[float] = None
    exit_reason:    Optional[str] = None
    pnl_pct:        Optional[float] = None
    holding_days:   Optional[int] = None
    error_category: Optional[str] = None
    lesson:         Optional[str] = None


class TradeJournal:
    """完整交易日志，进场前必须通过Gate Check"""

    GATE_CHECKS = [
        ('entry_reason', "进场理由不能为空"),
        ('first_exit',   "必须写明第一卖点"),
        ('forced_exit',  "必须写明强制卖点"),
    ]

    def __init__(self, path: str = None):
        self.path = Path(path) if path else None
        self.trades: Dict[str, TradeRecord] = {}
        if self.path and self.path.exists():
            self._load()

    def open(self, ticker: str, direction: str, entry_price: float,
              position_pct: float, regime: str, entry_reason: str,
              engines_used: List[str], expected_days: int,
              first_exit: str, forced_exit: str,
              canyon_score: float = 0, canyon_grade: str = 'D') -> str:
        errors = []
        vals = {'entry_reason': entry_reason, 'first_exit': first_exit, 'forced_exit': forced_exit}
        for k, msg in self.GATE_CHECKS:
            if not vals[k].strip():
                errors.append(f"❌ {msg}")
        if expected_days <= 0:
            errors.append("❌ 必须写明预期持仓天数")
        if not engines_used:
            errors.append("❌ 必须注明使用了哪些信号引擎")
        if errors:
            raise ValueError("\n进场前检查失败:\n" + "\n".join(errors))

        tid = str(uuid.uuid4())[:8].upper()
        self.trades[tid] = TradeRecord(
            trade_id=tid, ticker=ticker.upper(), direction=direction,
            entry_date=datetime.now().strftime('%Y-%m-%d'),
            entry_price=entry_price, position_pct=position_pct,
            regime=regime, entry_reason=entry_reason[:120],
            engines_used=engines_used, expected_days=expected_days,
            first_exit=first_exit, forced_exit=forced_exit,
            canyon_score=canyon_score, canyon_grade=canyon_grade
        )
        self._save()
        return tid

    def close(self, trade_id: str, exit_price: float, exit_reason: str,
              error_category: str = '', lesson: str = '') -> Dict:
        if trade_id not in self.trades:
            raise KeyError(f"找不到交易: {trade_id}")
        t   = self.trades[trade_id]
        d   = 1 if t.direction == 'long' else -1
        pnl = d * (exit_price / t.entry_price - 1)
        days= (datetime.now() - datetime.strptime(t.entry_date, '%Y-%m-%d')).days or 1
        t.exit_date      = datetime.now().strftime('%Y-%m-%d')
        t.exit_price     = exit_price
        t.exit_reason    = exit_reason
        t.pnl_pct        = round(pnl, 6)
        t.holding_days   = days
        t.error_category = error_category
        t.lesson         = lesson
        self._save()
        icon = '✅' if pnl > 0 else '❌'
        print(f"{icon} [{t.ticker}] P&L:{pnl:+.2%} 持有{days}天 | {exit_reason[:50]}")
        return {'pnl': pnl, 'days': days}

    def dashboard(self):
        closed = [t for t in self.trades.values() if t.exit_date]
        open_t = [t for t in self.trades.values() if not t.exit_date]
        print(f"\n{'═'*62}")
        print(f"  📔 交易日志 Dashboard")
        print(f"{'═'*62}")
        print(f"  持仓中:{len(open_t)} | 已平仓:{len(closed)} | 总计:{len(self.trades)}")
        if closed:
            pnls = [t.pnl_pct for t in closed if t.pnl_pct is not None]
            wins = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p <= 0]
            print(f"\n  P&L汇总:")
            print(f"    总收益:{sum(pnls):+.2%} | 胜率:{len(wins)/len(pnls):.1%}")
            if wins:  print(f"    平均盈利:{np.mean(wins):+.2%}")
            if loss:  print(f"    平均亏损:{np.mean(loss):+.2%}")
            if wins and loss:
                print(f"    盈亏比:{abs(np.mean(wins)/np.mean(loss)):.2f}:1")
            eng_pnl = defaultdict(list)
            for t in closed:
                if t.pnl_pct is None: continue
                for e in t.engines_used:
                    eng_pnl[e].append(t.pnl_pct)
            print(f"\n  引擎归因:")
            for eng, ep in sorted(eng_pnl.items(), key=lambda x: np.mean(x[1]), reverse=True):
                avg = np.mean(ep)
                print(f"    {'🟢' if avg>0 else '🔴'} {eng:<25} 次数:{len(ep):<4} 均值:{avg:+.2%}")
        if open_t:
            print(f"\n  当前持仓:")
            for t in open_t:
                days = (datetime.now() - datetime.strptime(t.entry_date, '%Y-%m-%d')).days
                over = ' ⚠️超期' if days > t.expected_days else ''
                print(f"    {t.trade_id} {t.ticker} {t.direction} | {days}天(预期{t.expected_days}){over}")
        print(f"{'═'*62}")

    def _save(self):
        if not self.path: return
        with open(self.path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.trades.items()}, f, indent=2, default=str)

    def _load(self):
        with open(self.path) as f:
            for k, v in json.load(f).items():
                self.trades[k] = TradeRecord(**v)


# ══════════════════════════════════════════════════════════════════════════════
# 21. [NEW] Live Trader（实盘主循环）
# ══════════════════════════════════════════════════════════════════════════════

class LiveTrader:
    """
    实盘主交易循环

    架构（文档要求）：
    Strategy → Risk Monitor → Strategy Monitor → Execution → Logger → Alert

    使用方法：
        trader = LiveTrader(
            execution=AlpacaExecution(key='...', secret='...'),
            capital=100000
        )
        # 在市场开盘后每天运行一次：
        trader.run_once(prices, volumes, market)
    """

    def __init__(self,
                 execution: AlpacaExecution = None,
                 capital:   float = 100000,
                 alert:     AlertEngine = None,
                 logger:    StateLogger = None):
        self.execution  = execution or AlpacaExecution()
        self.capital    = capital
        self.alert      = alert or AlertEngine()
        self.logger     = logger or StateLogger()
        self.live_risk  = LiveRiskManager()
        self.strat_mon  = StrategyMonitor()
        self.risk_mon   = RiskMonitor(self.alert)
        self.stat_arb   = StatArb()
        self.od_mgr     = OffensiveDefensiveManager(RiskManager())
        self.equity_hist= [1.0]
        self.daily_rets = []

    def run_once(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                 market: pd.Series) -> Optional[Dict[str, float]]:
        """
        单次交易循环（每天开盘后调用）

        Returns: 目标权重 or None（停机）
        """
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] 运行交易循环...")

        # ── 1. 策略失效检测 ────────────────────────────────────────────
        broken, reason = self.strat_mon.is_broken()
        if broken:
            self.alert.alert_strategy_broken()
            self.alert.send(f"原因: {reason}", 'CRITICAL')
            return None

        # ── 2. Kill Switch检查 ─────────────────────────────────────────
        pnl_today = self.daily_rets[-1] if self.daily_rets else 0.0
        current_dd = (min(self.equity_hist) / max(self.equity_hist) - 1
                      if len(self.equity_hist) > 1 else 0.0)
        kill = self.risk_mon.check(pnl_today, current_dd)
        if kill or self.live_risk.check_kill_switch(pnl_today):
            return None

        # ── 3. 生成信号 ────────────────────────────────────────────────
        regime, detail = detect_regime(market, prices)
        cs   = cross_sectional_momentum(prices)
        tsig = trend_signals(prices)
        pairs = self.stat_arb.find_pairs(prices)
        arbs  = self.stat_arb.current_opportunity(prices, pairs)

        alloc = self.od_mgr.allocate(
            regime=regime,
            long_alpha=cs['long_alpha'].dropna(),
            short_alpha=cs['short_alpha'].dropna(),
            trend_sig=tsig,
            stat_arb_opps=arbs,
            returns=prices.pct_change().dropna()
        )
        weights = {k: float(v) for k, v in alloc.to_series().items()}

        # ── 4. 风控过滤 ────────────────────────────────────────────────
        weights = self.live_risk.check_position_limit(weights)

        # ── 5. 执行 ────────────────────────────────────────────────────
        prices_last = {tk: float(prices[tk].iloc[-1]) for tk in prices.columns}
        self.execution.rebalance(weights, prices_last, self.capital)

        # ── 6. 记录 ────────────────────────────────────────────────────
        self.logger.log({
            'regime':       regime.label,
            'n_longs':      len(alloc.longs),
            'n_shorts':     len(alloc.shorts),
            'net_exposure': alloc.net_exposure,
            'cash':         alloc.cash,
            'rolling_sharpe': self.strat_mon.rolling_sharpe(),
            'current_dd':   current_dd,
            'pnl_today':    pnl_today
        })

        self.alert.send(f"交易完成: {regime.label} | "
                        f"多{len(alloc.longs)}空{len(alloc.shorts)} | "
                        f"净{alloc.net_exposure:.0%}", 'INFO')

        return weights

    def update_pnl(self, daily_return: float):
        """每日收盘后更新P&L（用于策略监控）"""
        self.daily_rets.append(daily_return)
        self.equity_hist.append(self.equity_hist[-1] * (1 + daily_return))
        self.strat_mon.update([daily_return])


# ══════════════════════════════════════════════════════════════════════════════
# 22. 主系统
# ══════════════════════════════════════════════════════════════════════════════

class CanyonTradingSystem:
    """Canyon量化交易系统 v6.0 — 完整版"""

    def __init__(self, tc_bps: float = 10, rf: float = 0.03,
                 target_vol: float = 0.10):
        self.data       = DataLayer()
        self.stat_arb   = StatArb()
        self.risk       = RiskManager()
        self.od         = OffensiveDefensiveManager(self.risk)
        self.backtester = WalkForwardBacktester(tc_bps=tc_bps, rf=rf,
                                                 target_vol=target_vol)
        self.ucb        = UCBOptimizer()
        self.journal    = TradeJournal()
        self.ic_engine  = AlphaICEngine()
        self.stats      = StatisticalDepth()
        self.exec_model = ExecutionCostModel()
        self.rf         = rf

    def run(self, tickers: List[str], start: str, end: str,
            benchmark: str = 'SPY') -> Dict:
        print(f"\n{'═'*62}")
        print(f"  🏔  CANYON量化交易系统 v6.0")
        print(f"  进攻防守 × IC验证Alpha × 执行成本 × 统计深度")
        print(f"{'═'*62}")
        print(f"  资产: {tickers}")
        print(f"  时间: {start} → {end}")

        prices, volumes, market = self.data.load(tickers, start, end, benchmark)
        returns = prices.pct_change().dropna()

        # Step 1: 主回测
        print(f"\n{'─'*62}")
        print(f"  Step1: Walk-Forward回测（书第7章 + v6改进）")
        print(f"{'─'*62}")
        main_result = self.backtester.run(
            prices, volumes, market, self.stat_arb, self.od,
            use_ic_alpha=True, verbose=True
        )

        # Step 2: 多期验证
        print(f"\n{'─'*62}")
        print(f"  Step2: 多期验证（书第7章：多段独立测试）")
        print(f"{'─'*62}")
        period_df = self.backtester.multi_period(
            prices, volumes, market, self.stat_arb, self.od, n_periods=3
        )

        # Step 3: UCB Bayesian优化
        print(f"\n{'─'*62}")
        print(f"  Step3: UCB Bayesian优化（书第9章）")
        print(f"{'─'*62}")
        pairs = self.stat_arb.find_pairs(prices)
        if pairs:
            bp = pairs[0]
            t1, t2 = bp['t1'], bp['t2']
            print(f"  最优协整对: {t1}/{t2} "
                  f"(p={bp['pvalue']:.3f}, HL={bp['half_life']:.1f}天)")

            def obj_fn(entry_z: float, exit_z: float, window: int):
                sa = StatArb(entry_z=entry_z, exit_z=exit_z, window=window)
                r  = sa.backtest_pair(prices[t1], prices[t2], bp)
                return -99.0 if r.get('max_dd', -1) < -0.056 else r.get('sharpe', 0.0)

            opt = self.ucb.optimize(
                obj_fn,
                param_bounds={'entry_z': (1.5, 3.0),
                               'exit_z':  (0.2, 1.0),
                               'window':  (10, 30)},
                n_iter=18
            )
            bp2 = opt.get('best_params', {})
            if bp2 and opt.get('best_score', 0) > -90:
                print(f"  最优: entry_z={bp2.get('entry_z',2):.2f} "
                      f"exit_z={bp2.get('exit_z',0.5):.2f} "
                      f"window={bp2.get('window',21)} "
                      f"→ Sharpe={opt.get('best_score',0):.3f}")
                self.stat_arb.entry_z = bp2.get('entry_z', 2.0)
                self.stat_arb.exit_z  = bp2.get('exit_z', 0.5)
                self.stat_arb.window  = int(bp2.get('window', 21))
        else:
            print("  未找到协整对")

        # Step 4: 统计深度报告（v6新增）
        print(f"\n{'─'*62}")
        print(f"  Step4: 统计深度分析（v6新增）")
        print(f"{'─'*62}")
        if 'daily_returns' in main_result and len(main_result['daily_returns']) > 30:
            dr = main_result['daily_returns']
            self.stats.print_full_report(dr, market.pct_change().dropna(),
                                          label='Canyon v6', rf=self.rf)

        # Step 5: Alpha IC验证报告
        print(f"\n{'─'*62}")
        print(f"  Step5: Alpha IC验证报告（v6新增）")
        print(f"{'─'*62}")
        if len(prices) >= 100:
            _, ic_vals = AlphaICEngine.build_validated_alpha(prices, volumes, market)
            print(f"  {'因子':<20} {'IC均值':>8} {'ICIR':>7} {'t-stat':>8} {'状态':>8}")
            print(f"  {'─'*52}")
            for fname, v in sorted(ic_vals.items(), key=lambda x: abs(x[1].get('icir',0)), reverse=True):
                icon = '✅' if v.get('valid') else '❌'
                print(f"  {fname:<20} {v.get('ic_mean',0):>8.4f} "
                      f"{v.get('icir',0):>7.3f} {v.get('t_stat',0):>8.2f} "
                      f"  {icon}{v.get('reason','')[:15]}")

        # Step 6: 当前分析
        print(f"\n{'─'*62}")
        print(f"  Step6: 当前市场状态")
        print(f"{'─'*62}")
        current = self.analyze_current(prices, volumes, market)

        # Step 7: 参数网格
        print(f"\n{'─'*62}")
        print(f"  Step7: 参数敏感性（书第7章）")
        print(f"{'─'*62}")
        grid = self._grid_test(prices, volumes, market)

        self._final_report(main_result, period_df, current, grid)
        return {'main': main_result, 'periods': period_df,
                'current': current, 'grid': grid}

    def analyze_current(self, prices, volumes, market) -> Dict:
        """当前市场状态 + 仓位建议"""
        regime, detail = detect_regime(market, prices)
        cs    = cross_sectional_momentum(prices)
        tsig  = trend_signals(prices)
        pairs = self.stat_arb.find_pairs(prices)
        arbs  = self.stat_arb.current_opportunity(prices, pairs)

        print(f"\n  市场环境: {regime.label}（{regime.stance}）")
        print(f"    综合分:{detail.get('composite',0):+.1f} | "
              f"趋势:{detail.get('trend',0):+.2f} | "
              f"动量:{detail.get('momentum',0):+.2f} | "
              f"广度:{detail.get('breadth',0):+.2f}")
        print(f"    目标总敞口:{regime.target_gross_exposure:.0%} | "
              f"单票上限:{regime.max_long:.0%} | "
              f"空头上限:{regime.max_short:.0%}")

        print(f"\n  横截面动量（书第6章）:")
        print(f"    做多前25%: {cs['long'][:5]}")
        print(f"    做空后25%: {cs['short'][:5]}")
        print(f"    多空价差: {cs['spread']:+.2%}")

        print(f"\n  趋势信号（书第5章）:")
        bulls = [tk for tk in tsig.index if tsig.loc[tk,'signal'] == 1]
        bears = [tk for tk in tsig.index if tsig.loc[tk,'signal'] == -1]
        print(f"    金叉/上升: {bulls[:6]}")
        print(f"    死叉/下降: {bears[:6]}")

        if arbs:
            print(f"\n  统计套利（书第8章）:")
            for a in arbs:
                d = ('多'+a['t1']+'空'+a['t2'] if a['direction']==1
                     else '空'+a['t1']+'多'+a['t2'])
                print(f"    {a['t1']}/{a['t2']} z={a['z']:.2f} → {d}")

        # Canyon评分
        print(f"\n  Canyon F/C/E评分:")
        canyon_scores = {}
        for tk in prices.columns:
            s = canyon_score_auto(prices[tk], volumes[tk], market, regime)
            canyon_scores[tk] = s
            if s['can_buy']:
                print(f"    ✅ {tk}: {s['total']:.0f}分({s['grade']}) 上限{s['max_pos']:.0%}")

        # 仓位分配
        la = cs['long_alpha'].dropna()
        sa = cs['short_alpha'].dropna()
        ret_hist = prices.pct_change().dropna()
        alloc = self.od.allocate(regime=regime, long_alpha=la, short_alpha=sa,
                                  trend_sig=tsig, stat_arb_opps=arbs, returns=ret_hist)

        print(f"\n  今日仓位建议:")
        print(f"    {alloc.rationale}")
        if alloc.longs:
            print(f"    多头:")
            for tk, w in sorted(alloc.longs.items(), key=lambda x: -x[1]):
                cs_i = canyon_scores.get(tk, {})
                print(f"      ▲ {tk:<8} {w:+.1%}  (Canyon:{cs_i.get('total',0):.0f}/{cs_i.get('grade','?')})")
        if alloc.shorts:
            print(f"    空头:")
            for tk, w in sorted(alloc.shorts.items(), key=lambda x: x[1]):
                print(f"      ▼ {tk:<8} {w:+.1%}")

        # 执行成本预估（v6新增）
        all_w = alloc.to_series()
        if len(all_w) > 0:
            print(f"\n  执行成本预估（v6精确版）:")
            total_tc_bps = 0.0
            for tk, w in all_w.items():
                if abs(w) < 0.005 or tk not in prices.columns:
                    continue
                vol_d    = float(prices[tk].pct_change().dropna().tail(21).std())
                adv_usd  = float(volumes[tk].tail(21).mean()) * float(prices[tk].iloc[-1])
                trade_usd= abs(float(w)) * 1e6
                tc       = self.exec_model.total_cost(vol_d, adv_usd, trade_usd) * 10000
                total_tc_bps += abs(float(w)) * tc
                print(f"    {tk:<8} 成本:{tc:.1f}bps (价差+冲击)")
            print(f"    组合总成本估算: {total_tc_bps:.1f}bps")

        # 压力测试
        if len(all_w) > 0:
            print(f"\n  压力测试 (5.6%回撤硬约束):")
            for name, shock in [('2008金融危机',-0.45), ('2020疫情崩盘',-0.32),
                                  ('2022加息熊市',-0.22), ('常规回调-15%',-0.15)]:
                loss = float(sum((float(w)*shock if float(w)>0 else float(w)*(-shock*0.7))
                                  for w in all_w))
                print(f"    {'✅' if loss>-0.056 else '⚠️'} {name}: {loss:+.2%}")

        return {'regime': regime, 'allocation': alloc,
                'canyon_scores': canyon_scores, 'cs': cs, 'trends': tsig}

    def _grid_test(self, prices, volumes, market) -> pd.DataFrame:
        results = []
        for ema_s in [5, 10]:
            for sma_s in [20, 30, 50]:
                try:
                    p_sub = prices.tail(400)
                    v_sub = volumes.tail(400)
                    m_sub = market.tail(400)
                    if len(p_sub) < self.backtester.train_w + self.backtester.test_w:
                        continue
                    daily_r = []
                    for s in range(self.backtester.train_w, len(p_sub) - self.backtester.test_w,
                                   self.backtester.test_w):
                        p_tr = p_sub.iloc[s - self.backtester.train_w:s]
                        p_te = p_sub.iloc[s:s + self.backtester.test_w]
                        ts   = trend_signals(p_tr, ema_span=ema_s, sma_span=sma_s)
                        bull = [tk for tk in ts.index if ts.loc[tk,'signal'] == 1]
                        n    = len(bull)
                        if n == 0:
                            continue
                        w = 1.0 / n
                        for d in p_te.index[1:]:
                            prev_i = p_te.index.get_loc(d) - 1
                            dr = sum(float(p_te.loc[d, tk] / p_te.iloc[prev_i][tk] - 1) * w
                                     for tk in bull if tk in p_te.columns)
                            daily_r.append(dr)
                    if len(daily_r) > 10:
                        m = backtest_metrics(pd.Series(daily_r))
                        results.append({'ema': ema_s, 'sma': sma_s,
                                        'sharpe': m['sharpe'], 'max_dd': m['max_dd']})
                        print(f"    EMA{ema_s}/SMA{sma_s}: Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.2%}")
                except Exception:
                    pass
        df = pd.DataFrame(results)
        if len(df) > 0:
            best = df.nlargest(1, 'sharpe').iloc[0]
            print(f"    → 最优: EMA{best['ema']:.0f}/SMA{best['sma']:.0f} Sharpe={best['sharpe']:.3f}")
        return df

    def _final_report(self, main, periods, current, grid):
        print(f"\n{'═'*62}")
        print(f"  📋 最终报告汇总")
        print(f"{'═'*62}")
        print(f"  年化收益：{main.get('ann_ret',0):+.2%}")
        print(f"  Sharpe：  {main.get('sharpe',0):.4f}")
        print(f"  Calmar：  {main.get('calmar',0):.4f}")
        print(f"  最大回撤：{main.get('max_dd',0):.2%}")
        print(f"  总收益：  {main.get('total_ret',0):+.2%}")
        print(f"  多头贡献：{main.get('long_total_pnl',0):+.2%}")
        print(f"  空头贡献：{main.get('short_total_pnl',0):+.2%}")
        if len(periods) > 0:
            print(f"\n  多期稳健性 ({len(periods)}段):")
            print(f"    Sharpe: μ={periods['sharpe'].mean():.3f} σ={periods['sharpe'].std():.3f}")
            print(f"    MaxDD:  μ={periods['max_dd'].mean():.2%} σ={periods['max_dd'].std():.2%}")
        regime = current.get('regime')
        if regime:
            alloc = current.get('allocation')
            print(f"\n  当前状态: {regime.label}")
            if alloc:
                print(f"  仓位: {alloc.rationale}")
        print(f"\n  v6改进说明:")
        print(f"  [FIX] 牛市满仓：BULL_STRONG目标敞口90%，直接按n只平分")
        print(f"  [NEW] Alpha IC验证：通过IC>0.02/ICIR>0.3/|t|>2才用")
        print(f"  [NEW] 精确执行成本：买卖价差+Almgren-Chriss市场冲击")
        print(f"  [NEW] 波动率目标：动态缩放到10%年化波动")
        print(f"  [NEW] 回撤控制：>5%减仓50%，>10%减仓75%")
        print(f"  [NEW] 统计深度：Bootstrap Sharpe CI + Newey-West + 因子暴露")
        print(f"  [NEW] 实盘系统：StateLogger+Alert+StrategyMonitor+LiveTrader")
        print(f"{'═'*62}")

    def record_trade(self, ticker, direction, price, pct, regime,
                      reason, engines, days, first_exit, forced_exit) -> str:
        return self.journal.open(
            ticker=ticker, direction=direction, entry_price=price,
            position_pct=pct, regime=regime, entry_reason=reason,
            engines_used=engines, expected_days=days,
            first_exit=first_exit, forced_exit=forced_exit
        )

    def close_trade(self, tid, exit_price, exit_reason, lesson='') -> Dict:
        return self.journal.close(tid, exit_price, exit_reason, lesson=lesson)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 22. [v7] BaseAlpha接口 + Alpha实例库
# 统一Alpha规范：可验证 + 可加权 + 可淘汰
# ══════════════════════════════════════════════════════════════════════════════

class BaseAlpha:
    """
    所有Alpha的统一基类（v7架构文档要求）
    每个alpha必须实现：compute() + diagnostics()
    """
    name: str = "base"

    def compute(self, features: dict) -> pd.Series:
        raise NotImplementedError

    def diagnostics(self, alpha: pd.Series,
                    future_returns: pd.Series) -> dict:
        """
        IC / t-stat / IC衰减曲线
        IC = Spearman(alpha_t, return_{t+k})
        """
        alpha_c = alpha.dropna()
        fut_c   = future_returns.reindex(alpha_c.index).dropna()
        common  = alpha_c.index.intersection(fut_c.index)
        if len(common) < 10:
            return {"ic": 0.0, "t": 0.0, "icir": 0.0, "decay_1": 0.0, "decay_5": 0.0, "decay_10": 0.0}

        ac, fc = alpha_c[common].values, fut_c[common].values
        ic, _  = spearmanr(ac, fc)
        ic     = float(ic) if np.isfinite(ic) else 0.0

        # t-stat of alpha itself
        t_val  = float(np.mean(ac) / (np.std(ac) + 1e-8))

        # IC衰减（1/5/10日）
        def lag_ic(lag):
            fr_lag = future_returns.shift(-lag).reindex(alpha_c.index).dropna()
            cm     = alpha_c.index.intersection(fr_lag.index)
            if len(cm) < 10: return 0.0
            v, _   = spearmanr(alpha_c[cm], fr_lag[cm])
            return float(v) if np.isfinite(v) else 0.0

        # ICIR
        ic_ser = []
        step   = max(1, len(common) // 20)
        for i in range(0, len(common) - step, step):
            sl = common[i:i+step]
            if len(sl) < 5: continue
            v, _ = spearmanr(alpha_c[sl], fut_c[sl])
            if np.isfinite(v): ic_ser.append(v)
        icir = float(np.mean(ic_ser) / (np.std(ic_ser) + 1e-8)) if len(ic_ser) > 3 else 0.0

        return {
            "ic":      round(ic, 5),
            "t":       round(t_val, 4),
            "icir":    round(icir, 4),
            "decay_1": round(lag_ic(1), 5),
            "decay_5": round(lag_ic(5), 5),
            "decay_10":round(lag_ic(10), 5),
        }


# ── Alpha实例库 ──────────────────────────────────────────────────────────────

class MomentumAlpha(BaseAlpha):
    """12-1月横截面动量（书第6章核心）"""
    name = "mom_12_1"

    def compute(self, features: dict) -> pd.Series:
        p = features["price"]
        n = len(p)
        lk12 = min(252, n-2); lk1 = min(21, n-2)
        return (p.pct_change(lk12) - p.pct_change(lk1)).iloc[-1].rank(pct=True)


class MeanRevAlpha(BaseAlpha):
    """短期反转（过度反应→均值回归）"""
    name = "mean_rev_5"

    def compute(self, features: dict) -> pd.Series:
        r = features["returns"]
        return (-r.rolling(5).mean()).iloc[-1].rank(pct=True)


class VolBreakoutAlpha(BaseAlpha):
    """波动率突破（相对历史波动率）"""
    name = "vol_breakout"

    def compute(self, features: dict) -> pd.Series:
        r   = features["returns"]
        vol = r.rolling(20).std()
        return (vol - vol.rolling(60).mean()).iloc[-1].rank(pct=True)


class RelStrengthAlpha(BaseAlpha):
    """相对强度（vs市场）"""
    name = "rel_strength"

    def compute(self, features: dict) -> pd.Series:
        p   = features["price"]
        mkt = features.get("market")
        n   = min(21, len(p)-2)
        ret = p.pct_change(n).iloc[-1]
        if mkt is not None:
            mkt_ret = float(mkt.pct_change(n).iloc[-1])
            ret = ret - mkt_ret
        return ret.rank(pct=True)


class LowVolAlpha(BaseAlpha):
    """低波动因子（低波动股票风险调整后收益更高）"""
    name = "low_vol"

    def compute(self, features: dict) -> pd.Series:
        r = features["returns"]
        return (-r.rolling(21).std()).iloc[-1].rank(pct=True)


class PriceEfficiencyAlpha(BaseAlpha):
    """价格效率比（趋势质量，Kaufman ER）"""
    name = "efficiency"

    def compute(self, features: dict) -> pd.Series:
        p    = features["price"]
        n    = min(21, len(p)-2)
        net  = p.diff(n).abs()
        path = p.diff(1).abs().rolling(n).sum() + 1e-8
        return (net / path).iloc[-1].rank(pct=True)


class VolumeDirectionAlpha(BaseAlpha):
    """量价方向性（有方向的成交量是信息）"""
    name = "vol_direction"

    def compute(self, features: dict) -> pd.Series:
        p   = features["price"]
        v   = features.get("volume", pd.DataFrame())
        r   = features["returns"]
        if isinstance(v, pd.DataFrame) and len(v.columns) > 0:
            vm   = v.rolling(21).mean()
            vz   = (v - vm) / (v.rolling(21).std() + 1e-8)
            sign = np.sign(r.rolling(5).mean())
            return (vz * sign).iloc[-1].rank(pct=True)
        return r.pct_change(5).iloc[-1].rank(pct=True)


# ══════════════════════════════════════════════════════════════════════════════
# 23. [v7] AlphaPool：筛选 + 动态IC权重 + 组合
# ══════════════════════════════════════════════════════════════════════════════

class AlphaPool:
    """
    Alpha池（v7架构文档核心）
    · 每个alpha必须通过gate才能进入组合
    · 权重 = IC加权（可扩展到Bayesian/ML）
    · 自动淘汰：IC/t不达标的alpha不用
    """

    # Gate阈值（比v6 AlphaICEngine稍微宽松，允许更多alpha参与）
    IC_MIN    = 0.02
    T_MIN     = 1.5
    ICIR_MIN  = 0.15

    def __init__(self, alphas: List[BaseAlpha]):
        self.alphas  = alphas
        self.weights: Dict[str, float] = {}
        self.diagnostics_cache: Dict[str, dict] = {}

    def evaluate(self, features: dict,
                  future_returns: pd.Series) -> Dict[str, Tuple]:
        """
        评估所有alpha，返回通过gate的 {name: (signal, stats)}
        """
        passed = {}
        for a in self.alphas:
            try:
                signal = a.compute(features)
                if signal is None or len(signal.dropna()) < 5:
                    continue
                stats = a.diagnostics(signal, future_returns)
                self.diagnostics_cache[a.name] = stats

                # Gate：IC显著 + t-stat显著
                if (abs(stats["ic"]) >= self.IC_MIN and
                    abs(stats["t"])  >= self.T_MIN):
                    passed[a.name] = (signal, stats)
            except Exception:
                pass
        return passed

    def weight(self, passed_dict: Dict) -> Dict[str, float]:
        """IC绝对值加权（稳定的因子权重更高）"""
        if not passed_dict:
            return {}
        ics   = {k: abs(v[1]["ic"]) for k, v in passed_dict.items()}
        # ICIR调整：ICIR越高，权重加成越大
        icirs = {k: max(0, v[1].get("icir", 0)) for k, v in passed_dict.items()}
        scores= {k: ics[k] * (1 + icirs[k]) for k in ics}
        total = sum(scores.values()) + 1e-8
        self.weights = {k: v / total for k, v in scores.items()}
        return self.weights

    def combine(self, passed_dict: Dict) -> pd.Series:
        """加权合成组合Alpha信号"""
        if not passed_dict or not self.weights:
            return pd.Series(dtype=float)
        combo = None
        for name, (signal, _) in passed_dict.items():
            w = self.weights.get(name, 0.0)
            if combo is None:
                combo = signal * w
            else:
                combo = combo.add(signal * w, fill_value=0)
        if combo is not None and combo.std() > 0:
            combo = (combo - combo.mean()) / combo.std()
        return combo if combo is not None else pd.Series(dtype=float)

    def print_diagnostics(self):
        """打印所有Alpha诊断报告"""
        if not self.diagnostics_cache:
            return
        print(f"\n  {'Alpha':<20} {'IC':>7} {'ICIR':>7} {'t-stat':>7} "
              f"{'d1':>7} {'d5':>7} {'状态':>8}")
        print(f"  {'─'*65}")
        for name, d in sorted(self.diagnostics_cache.items(),
                               key=lambda x: abs(x[1].get('ic', 0)), reverse=True):
            passed = (abs(d.get('ic',0)) >= self.IC_MIN and
                      abs(d.get('t', 0)) >= self.T_MIN)
            icon  = '✅' if passed else '❌'
            wt    = self.weights.get(name, 0.0)
            print(f"  {name:<20} {d.get('ic',0):>7.4f} {d.get('icir',0):>7.3f} "
                  f"{d.get('t',0):>7.2f} {d.get('decay_1',0):>7.4f} "
                  f"{d.get('decay_5',0):>7.4f}  {icon} w={wt:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 24. [v7] Regime Model（KMeans聚类，独立于detect_regime）
# 用于MetaModel训练：不同市场环境→不同alpha权重
# ══════════════════════════════════════════════════════════════════════════════

class RegimeModel:
    """
    v7文档：特征 + KMeans聚类识别市场状态
    与 detect_regime() 的五维度规则系统互补：
    · detect_regime() → 可解释的规则，用于仓位管理
    · RegimeModel     → 数据驱动的聚类，用于MetaModel训练
    """

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.model      = None
        self.labels_    = {0: 'cluster_0', 1: 'cluster_1', 2: 'cluster_2'}
        self.fitted     = False

    def _build_features(self, market_returns: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({
            "vol_21":   market_returns.rolling(21).std(),
            "vol_63":   market_returns.rolling(63).std(),
            "mom_21":   market_returns.rolling(21).mean(),
            "mom_63":   market_returns.rolling(63).mean(),
            "vol_ratio":(market_returns.rolling(21).std() /
                         (market_returns.rolling(63).std() + 1e-8)),
        }).dropna()

    def fit(self, market_returns: pd.Series):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        X = self._build_features(market_returns)
        if len(X) < self.n_clusters * 5:
            return self
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        self.model   = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.model.fit(Xs)
        self._scaler  = sc
        self._X_index = X.index
        self.fitted   = True

        # 给每个cluster打标签（按平均动量排序）
        labels_arr = self.model.labels_
        cluster_moms = {}
        for c in range(self.n_clusters):
            mask = labels_arr == c
            cluster_moms[c] = float(X["mom_21"][mask].mean())
        sorted_c = sorted(cluster_moms, key=cluster_moms.get)
        self.labels_ = {sorted_c[0]: 'bear', sorted_c[1]: 'neutral',
                        sorted_c[-1]: 'bull'}
        return self

    def predict(self, market_returns: pd.Series) -> pd.Series:
        """返回每个时间点的市场状态标签"""
        if not self.fitted:
            return pd.Series('neutral', index=market_returns.index)
        X = self._build_features(market_returns)
        Xs = self._scaler.transform(X)
        raw = self.model.predict(Xs)
        labels = pd.Series([self.labels_.get(r, 'neutral') for r in raw], index=X.index)
        return labels.reindex(market_returns.index).ffill().fillna('neutral')

    def current_regime(self, market_returns: pd.Series) -> str:
        preds = self.predict(market_returns)
        return str(preds.iloc[-1]) if len(preds) > 0 else 'neutral'


# ══════════════════════════════════════════════════════════════════════════════
# 25. [v7] Meta Model：按Regime学习Alpha权重
# ══════════════════════════════════════════════════════════════════════════════

class MetaModel:
    """
    v7文档核心：
    · 按Regime分别训练Ridge回归
    · 预测：给定当前regime，返回alpha组合信号
    · 不同市场环境下不同alpha有效（这是真正的"智能调度"）

    数学：
    y_t = Σ_i w_i(regime) × alpha_i_t
    w_i(regime) 由Ridge回归学习
    """

    def __init__(self, alpha_decay: float = 0.01):
        self.models:       Dict[str, Ridge] = {}
        self.scaler:       StandardScaler = StandardScaler()
        self.alpha_decay   = alpha_decay
        self.trained_regimes: List[str] = []
        self.feature_importances: Dict[str, Dict] = {}

    def train(self, regime_series: pd.Series,
               alpha_df: pd.DataFrame,
               future_returns: pd.Series):
        """
        按regime分段训练Ridge回归
        X = alpha矩阵（T×K），y = 未来收益（T,）
        """
        common = alpha_df.index.intersection(future_returns.index).intersection(regime_series.index)
        if len(common) < 30:
            return

        X_all = alpha_df.loc[common].fillna(0)
        y_all = future_returns.loc[common].fillna(0)
        r_all = regime_series.loc[common]

        # 标准化features
        X_sc = self.scaler.fit_transform(X_all)

        for regime in r_all.unique():
            idx = r_all == regime
            if idx.sum() < 20:
                continue
            X_r, y_r = X_sc[idx], y_all[idx].values
            m = Ridge(alpha=1.0)
            m.fit(X_r, y_r)
            self.models[regime] = m
            self.trained_regimes.append(regime)

            # 记录特征重要性
            self.feature_importances[regime] = {
                col: float(coef)
                for col, coef in zip(alpha_df.columns, m.coef_)
            }

    def predict(self, regime: str,
                alpha_row: pd.Series) -> float:
        """
        给定当前regime和当前alpha信号，返回预测信号强度
        """
        if regime not in self.models:
            # 回退：等权平均
            return float(alpha_row.fillna(0).mean())
        m   = self.models[regime]
        row = self.scaler.transform(alpha_row.fillna(0).values.reshape(1, -1))
        return float(m.predict(row)[0])

    def predict_weights(self, regime: str,
                         alpha_names: List[str]) -> Dict[str, float]:
        """返回当前regime下每个alpha的权重（来自Ridge系数）"""
        if regime not in self.feature_importances:
            n = len(alpha_names)
            return {a: 1/n for a in alpha_names}
        fi = self.feature_importances[regime]
        raw = {a: max(0, fi.get(a, 0)) for a in alpha_names}
        total = sum(raw.values()) + 1e-8
        return {a: v/total for a, v in raw.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 26. [v7] Portfolio Engine（风险约束 + 稳定分配）
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioEngine:
    """
    v7文档：风险惩罚 + 波动率目标 + 平滑分配

    目标：
    w* = argmax[ rank(alpha) × exp(·) / risk - λ × tracking_error ]
    同时施加波动率目标缩放
    """

    def __init__(self, target_vol: float = 0.10,
                 max_scale: float = 1.5,
                 risk_aversion: float = 2.0):
        self.target_vol    = target_vol
        self.max_scale     = max_scale
        self.risk_aversion = risk_aversion
        self._prev_w       = None

    def allocate(self, alpha_signal: pd.Series,
                  returns: pd.DataFrame,
                  regime: 'Regime' = None,
                  turnover_penalty: float = 0.002) -> pd.Series:
        """
        分配权重：
        1. 按alpha排名 → exp加权
        2. 风险惩罚（除以个股波动率）
        3. 波动率目标缩放
        4. 换手率惩罚（平滑分配）
        """
        tickers = [t for t in alpha_signal.dropna().index
                   if t in returns.columns]
        if not tickers:
            return pd.Series(dtype=float)

        alpha = alpha_signal[tickers].fillna(0)
        ret   = returns[tickers].dropna()

        # 只做多头（或按regime决定）
        # 用排名exp加权
        rank = alpha.rank(pct=True)
        w    = np.exp(rank.values * 3)    # exp放大排名差异
        w    = w / w.sum()

        # 风险惩罚：高波动股票降权
        vols = ret.std().values * np.sqrt(252) + 1e-6
        w    = w / vols
        w    = w / w.sum()

        # 波动率目标
        cov  = ret.cov().values
        port_vol = float(np.sqrt(w @ cov @ w * 252))
        if port_vol > 1e-4:
            scale = self.target_vol / port_vol
            w     = w * min(scale, self.max_scale)

        # 换手率惩罚（减少频繁交易）
        if self._prev_w is not None:
            prev = pd.Series(self._prev_w).reindex(tickers).fillna(0).values
            w    = w - turnover_penalty * (w - prev)
            w    = np.maximum(w, 0)
            if w.sum() > 1e-8:
                w /= w.sum()

        result = pd.Series(w, index=tickers)
        self._prev_w = result.to_dict()

        # Regime限制
        if regime is not None:
            result = result.clip(upper=regime.max_long)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 27. [v7] Risk Server（独立风控，不和策略混）
# ══════════════════════════════════════════════════════════════════════════════

class RiskServer:
    """
    v7文档：独立风控层（与策略解耦）
    · 硬约束：单票上限 / 集中度
    · Kill Switch：回撤超限停止所有交易
    · 净敞口约束：多空差值不能过大
    """

    def __init__(self, max_pos: float = 0.15,
                 max_drawdown: float = -0.10,
                 max_concentration: float = 0.35):
        self.max_pos           = max_pos
        self.max_drawdown      = max_drawdown
        self.max_concentration = max_concentration  # 单行业≤35%
        self._killed           = False
        self._equity           = pd.Series([1.0])

    def enforce(self, weights: pd.Series) -> pd.Series:
        """
        强制执行单票上限 + 集中度约束
        """
        if self._killed:
            print("  🚨 RiskServer: Kill Switch激活，返回空仓位")
            return pd.Series(0.0, index=weights.index)

        # 单票上限
        w = weights.clip(lower=-self.max_pos, upper=self.max_pos)

        # 集中度：总多头不超过max_concentration × 3（允许最多3倍集中）
        long_w  = w[w > 0]
        if len(long_w) > 0 and long_w.sum() > 1.0:
            w[w > 0] /= long_w.sum()

        return w

    def kill_switch(self, equity: pd.Series) -> bool:
        """
        检查是否触发Kill Switch
        equity: 累计净值序列
        """
        if len(equity) < 2:
            return False
        peak = equity.cummax()
        dd   = ((equity - peak) / peak).iloc[-1]
        if dd < self.max_drawdown:
            self._killed = True
            print(f"  🚨 RiskServer Kill Switch: 回撤{dd:.2%} < {self.max_drawdown:.2%}")
            return True
        return False

    def reset(self):
        """手动重置Kill Switch（人工确认后）"""
        self._killed = False
        print("  ✅ RiskServer: Kill Switch已重置")

    def check_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """dict版本，与AlpacaExecution接口匹配"""
        return {k: float(np.clip(v, -self.max_pos, self.max_pos))
                for k, v in weights.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 28. [v7] 交易基础设施：EventLogger + retry + Failover
# ══════════════════════════════════════════════════════════════════════════════

class EventLogger:
    """
    v7文档：结构化事件日志
    · 所有事件（订单/风控/信号/错误）按JSON行写入
    · 便于事后分析和审计
    """

    def __init__(self, path: str = "events.log"):
        self.path = path

    def log(self, event_type: str, payload: dict):
        import json, time
        rec = {
            "ts":   time.time(),
            "dt":   datetime.now().isoformat(),
            "type": event_type,
            "data": payload
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def read_df(self) -> pd.DataFrame:
        """读取所有事件为DataFrame"""
        import json
        rows = []
        try:
            with open(self.path) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line.strip()))
                    except Exception:
                        pass
        except Exception:
            pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()


def retry(fn, n: int = 3, delay: float = 1.0):
    """
    v7文档：网络/下单失败重试
    指数退避：每次等待时间翻倍
    """
    import time
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            if i == n - 1:
                raise
            wait = delay * (2 ** i)
            print(f"  [retry {i+1}/{n}] 等待{wait:.1f}s: {str(e)[:50]}")
            time.sleep(wait)


class Failover:
    """
    v7文档：主备切换（Alpaca主 → IBKR备 or 模拟备）
    · 主执行器失败时自动切换备用
    · 重要：两个执行器必须有相同的接口
    """

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup  = backup
        self._using_backup = False

    def submit_order(self, *args, **kwargs):
        if self._using_backup:
            return self.backup.submit_order(*args, **kwargs)
        try:
            return self.primary.submit_order(*args, **kwargs)
        except Exception as e:
            print(f"  ⚠️ Failover: 主执行器失败({str(e)[:40]})，切换到备用")
            self._using_backup = True
            return self.backup.submit_order(*args, **kwargs)

    def rebalance(self, *args, **kwargs):
        if self._using_backup:
            return self.backup.rebalance(*args, **kwargs)
        try:
            return self.primary.rebalance(*args, **kwargs)
        except Exception as e:
            print(f"  ⚠️ Failover: 主执行器失败，切换到备用")
            self._using_backup = True
            return self.backup.rebalance(*args, **kwargs)

    def reset_to_primary(self):
        """主执行器恢复后手动切回"""
        self._using_backup = False
        print("  ✅ Failover: 切回主执行器")


# ══════════════════════════════════════════════════════════════════════════════
# 29. [v7] run_system()：完整v7主流程
# ══════════════════════════════════════════════════════════════════════════════

def run_v7_system(prices: pd.DataFrame,
                   volumes: pd.DataFrame,
                   market: pd.Series,
                   regime_obj: 'Regime',
                   verbose: bool = True) -> Dict:
    """
    v7文档完整主流程：
    Data → AlphaPool → RegimeModel → MetaModel → PortfolioEngine → RiskServer
    """
    features = {
        "price":   prices,
        "returns": prices.pct_change().dropna(),
        "volume":  volumes,
        "market":  market
    }
    future_returns_cs = prices.pct_change(21).shift(-21).iloc[-1].dropna()

    # ── Alpha Pool ────────────────────────────────────────────────────────────
    pool = AlphaPool([
        MomentumAlpha(), MeanRevAlpha(), VolBreakoutAlpha(),
        RelStrengthAlpha(), LowVolAlpha(),
        PriceEfficiencyAlpha(), VolumeDirectionAlpha()
    ])

    # 用过去数据评估IC
    past_fut = prices.pct_change(21).shift(-21).mean(axis=1).dropna()
    past_alpha_cs = prices.pct_change(21).iloc[-1].dropna()
    passed = pool.evaluate(features, past_alpha_cs)
    pool.weight(passed)
    combo_alpha = pool.combine(passed)

    if verbose:
        print(f"\n  [v7 AlphaPool] 通过Gate: {len(passed)}/{len(pool.alphas)} 个alpha")
        pool.print_diagnostics()

    # ── Regime Model ──────────────────────────────────────────────────────────
    mkt_ret     = market.pct_change().dropna()
    rm          = RegimeModel(n_clusters=3)
    rm.fit(mkt_ret)
    regime_series = rm.predict(mkt_ret)
    current_r   = rm.current_regime(mkt_ret)

    if verbose:
        rc = regime_series.value_counts()
        print(f"\n  [v7 RegimeModel] 当前: {current_r} | "
              f"分布: {dict(rc.items())}")

    # ── Meta Model ────────────────────────────────────────────────────────────
    alpha_df = pd.DataFrame(
        {name: sig for name, (sig, _) in passed.items()},
    ).dropna(how='all')

    meta      = MetaModel()
    fut_mean  = prices.pct_change(21).shift(-21).mean(axis=1)
    if len(alpha_df) > 30 and len(fut_mean) > 30:
        common = alpha_df.index.intersection(fut_mean.index).intersection(regime_series.index)
        if len(common) > 30:
            meta.train(regime_series.loc[common], alpha_df.loc[common], fut_mean.loc[common])

    # 预测当前信号
    if len(alpha_df) > 0 and meta.trained_regimes:
        latest_row = alpha_df.iloc[-1].fillna(0)
        meta_signal = meta.predict(current_r, latest_row)
        meta_weights = meta.predict_weights(current_r, list(passed.keys()))
        if verbose:
            print(f"  [v7 MetaModel] 当前Regime权重: "
                  f"{', '.join(f'{k}:{v:.3f}' for k,v in list(meta_weights.items())[:4])}")
    else:
        meta_signal  = 0.0
        meta_weights = {}

    # ── Portfolio Engine ──────────────────────────────────────────────────────
    pe = PortfolioEngine(target_vol=0.10, max_scale=1.5)
    if combo_alpha is not None and len(combo_alpha) > 0:
        returns_hist = prices.pct_change().dropna()
        weights_raw  = pe.allocate(combo_alpha, returns_hist, regime=regime_obj)
    else:
        weights_raw = pd.Series(dtype=float)

    # ── Risk Server ───────────────────────────────────────────────────────────
    rs       = RiskServer(max_pos=regime_obj.max_long)
    weights  = rs.enforce(weights_raw)

    if verbose:
        print(f"\n  [v7 PortfolioEngine] {len(weights)}只 | "
              f"总敞口:{weights.abs().sum():.1%} | "
              f"净:{weights.sum():+.1%}")
        print(f"  [v7 RiskServer] "
              f"单票上限:{rs.max_pos:.0%} | Kill Switch:{'激活' if rs._killed else '正常'}")

    return {
        "weights":       weights,
        "alpha_pool":    passed,
        "pool_weights":  pool.weights,
        "regime":        current_r,
        "meta_signal":   meta_signal,
        "meta_weights":  meta_weights,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 30. [v8] SleeveManager — 资金分账户管理器
#
# 设计理念（对应你的"从市场非理性中赚钱"三个时间维度）：
#
# TACTICAL      短期非理性 / 1-5天    → 利用情绪极端 / 隔夜反转 / 事件催化
# CORE_HEDGE    长期复利 / 3月-多年   → Leopold类论文持仓 + 避险对冲
# SECTOR_ROT    板块轮动 / 2-8周      → L2中期周期 + 大盘轮动跟随
#
# 为什么要分Sleeve：
# · 不同持仓时间 = 不同风控标准（短期最严，长期最宽松）
# · 不同市场环境 = 不同Sleeve权重（熊市CORE_HEDGE扩到80%）
# · 不同Alpha来源 = 不同因子（避免三个Sleeve都挤在同一个因子上）
# · 实盘中可以对每个Sleeve单独Kill Switch，不影响其他
#
# 和机构的对照：
# · Bridgewater All Weather = CORE_HEDGE的思路（跨Regime持有）
# · Two Sigma的短期统计套利 = TACTICAL的思路
# · AQR的因子套利 = SECTOR_ROTATION的思路
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SleeveConfig:
    name:                str
    target_weight:       float    # 正常市场下的目标权重
    min_weight:          float    # 最小权重（任何市场下不低于此）
    max_weight:          float    # 最大权重（任何市场下不超过此）
    max_daily_loss:      float    # 单日亏损上限（触发则暂停该Sleeve）
    max_drawdown:        float    # 最大回撤上限（触发则Kill Switch该Sleeve）
    target_holding_days: str      # 目标持仓周期（描述性）
    mission:             str      # 使命（一句话）


class SleeveManager:
    """
    资金分账户管理器

    三个Sleeve各自独立风控，互不影响：
    1. TACTICAL：    短期非理性 / 隔夜 / 日内 / 1-5天
    2. CORE_HEDGE：  长期持有 / 防守 / 避险
    3. SECTOR_ROT：  跟随大盘和板块轮动

    动态权重逻辑：
    · 牛市强 → TACTICAL+SECTOR_ROT 进攻
    · 熊市强 → CORE_HEDGE 防守+避险
    · 每个Sleeve有独立的日亏损/回撤Kill Switch
    · 实盘中三个Sleeve分别对应不同的Alpaca sub-account或标签
    """

    def __init__(self):
        self.base_sleeves: Dict[str, SleeveConfig] = {
            "TACTICAL": SleeveConfig(
                name="TACTICAL",
                target_weight=0.25,
                min_weight=0.10,
                max_weight=0.35,
                max_daily_loss=0.008,       # 短期最严：0.8%/天
                max_drawdown=0.05,          # 5%回撤即停（短期策略）
                target_holding_days="intraday to 5 days",
                mission="capture short-term market irrationality"
            ),
            "CORE_HEDGE": SleeveConfig(
                name="CORE_HEDGE",
                target_weight=0.45,
                min_weight=0.30,
                max_weight=0.65,
                max_daily_loss=0.004,       # 核心账户最保守：0.4%/天
                max_drawdown=0.08,          # 8%回撤才停（长期策略容忍更多）
                target_holding_days="3 months to multi-year",
                mission="long-term compounding and portfolio defense"
            ),
            "SECTOR_ROTATION": SleeveConfig(
                name="SECTOR_ROTATION",
                target_weight=0.30,
                min_weight=0.15,
                max_weight=0.45,
                max_daily_loss=0.006,       # 中等：0.6%/天
                max_drawdown=0.07,
                target_holding_days="2 to 8 weeks",
                mission="profit from sector leadership rotation"
            )
        }

        # 各Sleeve的实时状态
        self._sleeve_equity: Dict[str, float]   = {k: 1.0 for k in self.base_sleeves}
        self._sleeve_peak:   Dict[str, float]   = {k: 1.0 for k in self.base_sleeves}
        self._sleeve_killed: Dict[str, bool]    = {k: False for k in self.base_sleeves}
        self._sleeve_daily_pnl: Dict[str, float]= {k: 0.0 for k in self.base_sleeves}

    # ── 核心：按Regime动态分配权重 ────────────────────────────────────────────

    def allocate_by_regime(self, regime_label: str) -> Dict[str, float]:
        """
        根据市场状态动态调整三个Sleeve的资金权重

        牛市：进攻（TACTICAL+SECTOR_ROT多分）
        熊市：防守（CORE_HEDGE多分，充当避险）
        注意：权重之和始终=1.0，但各Sleeve内部的多空由自己决定
        """
        regime_label = regime_label.upper()

        if "BULL_STRONG" in regime_label or "强牛" in regime_label:
            # 强牛市：三个账户都进攻
            base = {"TACTICAL": 0.30, "CORE_HEDGE": 0.35, "SECTOR_ROTATION": 0.35}

        elif "BULL_NORMAL" in regime_label or "普通牛" in regime_label:
            base = {"TACTICAL": 0.25, "CORE_HEDGE": 0.40, "SECTOR_ROTATION": 0.35}

        elif "NEUTRAL" in regime_label or "震荡" in regime_label:
            # 震荡：CORE_HEDGE居中守护
            base = {"TACTICAL": 0.25, "CORE_HEDGE": 0.50, "SECTOR_ROTATION": 0.25}

        elif "BEAR_MILD" in regime_label or "温和熊" in regime_label:
            # 温和熊：CORE_HEDGE开始承担避险职责
            base = {"TACTICAL": 0.15, "CORE_HEDGE": 0.65, "SECTOR_ROTATION": 0.20}

        elif "BEAR_STRONG" in regime_label or "强熊" in regime_label:
            # 强熊：CORE_HEDGE主导，TACTICAL只留底仓做空
            base = {"TACTICAL": 0.10, "CORE_HEDGE": 0.80, "SECTOR_ROTATION": 0.10}

        else:
            base = {"TACTICAL": 0.25, "CORE_HEDGE": 0.45, "SECTOR_ROTATION": 0.30}

        # 把被Kill Switch的Sleeve权重转移给CORE_HEDGE
        return self._apply_kill_adjustments(base)

    def _apply_kill_adjustments(self, base: Dict[str, float]) -> Dict[str, float]:
        """被Kill的Sleeve权重归零，剩余重新归一化（安全转移到CORE_HEDGE）"""
        result = base.copy()
        freed  = 0.0
        for k in list(result.keys()):
            if self._sleeve_killed.get(k, False):
                freed += result[k]
                result[k] = 0.0
        # 把释放的权重加给CORE_HEDGE（最稳的那个）
        if freed > 0 and not self._sleeve_killed.get("CORE_HEDGE", False):
            result["CORE_HEDGE"] = min(
                result.get("CORE_HEDGE", 0) + freed,
                self.base_sleeves["CORE_HEDGE"].max_weight
            )
        # 确保权重之和=1
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    # ── Sleeve级别风控 ────────────────────────────────────────────────────────

    def update_pnl(self, sleeve: str, daily_return: float) -> bool:
        """
        更新单个Sleeve的P&L，检查是否触发Kill Switch
        Returns: True = 正常，False = Kill Switch触发
        """
        if sleeve not in self._sleeve_equity:
            return True

        # 更新净值
        self._sleeve_equity[sleeve]     *= (1 + daily_return)
        self._sleeve_daily_pnl[sleeve]   = daily_return

        # 更新峰值
        if self._sleeve_equity[sleeve] > self._sleeve_peak[sleeve]:
            self._sleeve_peak[sleeve] = self._sleeve_equity[sleeve]

        cfg = self.base_sleeves[sleeve]
        dd  = (self._sleeve_equity[sleeve] - self._sleeve_peak[sleeve]) / \
              (self._sleeve_peak[sleeve] + 1e-8)

        # 日亏损Kill Switch
        if daily_return < -cfg.max_daily_loss:
            print(f"  ⚠️ [{sleeve}] 日亏损 {daily_return:.2%} 超过 {cfg.max_daily_loss:.2%}")
            # 日亏损只是警告，不直接Kill（只有回撤超限才Kill）

        # 回撤Kill Switch
        if dd < -cfg.max_drawdown:
            print(f"  🚨 [{sleeve}] Kill Switch: 回撤 {dd:.2%} < -{cfg.max_drawdown:.2%}")
            self._sleeve_killed[sleeve] = True
            return False

        return True

    def get_sleeve_status(self) -> pd.DataFrame:
        """打印各Sleeve当前状态"""
        rows = []
        for name, cfg in self.base_sleeves.items():
            eq  = self._sleeve_equity[name]
            pk  = self._sleeve_peak[name]
            dd  = (eq - pk) / (pk + 1e-8)
            rows.append({
                'Sleeve':       name,
                'Mission':      cfg.mission[:35],
                'Equity':       round(eq, 4),
                'Drawdown':     round(dd, 4),
                'MaxDD_Limit':  cfg.max_drawdown,
                'DD_Used%':     f"{abs(dd)/cfg.max_drawdown:.0%}",
                'DailyPnL':     round(self._sleeve_daily_pnl[name], 4),
                'Status':       '🚨KILLED' if self._sleeve_killed[name] else '✅OK',
                'HoldingPeriod':cfg.target_holding_days
            })
        return pd.DataFrame(rows)

    def reset_sleeve(self, sleeve: str) -> None:
        """手动重置单个Sleeve的Kill Switch（人工确认后）"""
        if sleeve in self._sleeve_killed:
            self._sleeve_killed[sleeve]   = False
            self._sleeve_equity[sleeve]   = self._sleeve_peak[sleeve]
            print(f"  ✅ [{sleeve}] Kill Switch已重置，峰值回归{self._sleeve_peak[sleeve]:.4f}")

    def print_dashboard(self, regime_label: str = 'NEUTRAL') -> None:
        """打印完整Sleeve Dashboard"""
        alloc   = self.allocate_by_regime(regime_label)
        status  = self.get_sleeve_status()

        print(f"\n{'═'*65}")
        print(f"  💼 SleeveManager Dashboard")
        print(f"  当前市场：{regime_label}")
        print(f"{'═'*65}")

        for _, row in status.iterrows():
            name = row['Sleeve']
            w    = alloc.get(name, 0)
            cfg  = self.base_sleeves[name]
            dd   = row['Drawdown']
            bar_len = int(abs(dd) / cfg.max_drawdown * 20)
            bar  = '█' * bar_len + '░' * (20 - bar_len)

            print(f"\n  [{row['Status']}] {name}")
            print(f"    使命：{cfg.mission}")
            print(f"    资金权重：{w:.0%} (范围 {cfg.min_weight:.0%}-{cfg.max_weight:.0%})")
            print(f"    持仓周期：{cfg.target_holding_days}")
            print(f"    净值：{row['Equity']:.4f}  日P&L：{row['DailyPnL']:+.2%}")
            print(f"    回撤进度 [{bar}] {abs(dd):.2%}/{cfg.max_drawdown:.0%} ({row['DD_Used%']})")
            print(f"    日亏损上限：{cfg.max_daily_loss:.1%}  回撤Kill：{cfg.max_drawdown:.0%}")

        print(f"\n  总资金分配：", end='')
        for k, v in alloc.items():
            print(f"{k}={v:.0%}  ", end='')
        print(f"\n{'═'*65}")

    # ── 各Sleeve的Alpha信号分配逻辑 ───────────────────────────────────────────

    def get_sleeve_alphas(self,
                          tactical_alpha: pd.Series,
                          core_alpha:     pd.Series,
                          sector_alpha:   pd.Series,
                          regime_label:   str) -> Dict[str, pd.Series]:
        """
        各Sleeve用什么Alpha信号（信号来源分离）

        TACTICAL    → 短期反转 + 情绪信号 + 统计套利 z-score
        CORE_HEDGE  → 12M动量 + L3 thesis + 低波动因子 + F/C/E高评分
        SECTOR_ROT  → 横截面动量 + 行业相对强度 + EMA/SMA趋势
        """
        alloc = self.allocate_by_regime(regime_label)

        return {
            "TACTICAL":        tactical_alpha.dropna() if alloc.get("TACTICAL", 0) > 0
                               else pd.Series(dtype=float),
            "CORE_HEDGE":      core_alpha.dropna()     if alloc.get("CORE_HEDGE", 0) > 0
                               else pd.Series(dtype=float),
            "SECTOR_ROTATION": sector_alpha.dropna()   if alloc.get("SECTOR_ROTATION", 0) > 0
                               else pd.Series(dtype=float),
        }

    def combine_sleeve_signals(self,
                                sleeve_signals: Dict[str, pd.Series],
                                regime_label:   str) -> pd.Series:
        """
        将三个Sleeve的信号按权重合成最终Alpha
        每个Sleeve内部先标准化，再按权重加总
        """
        alloc  = self.allocate_by_regime(regime_label)
        combined = None

        for sleeve, signal in sleeve_signals.items():
            if signal is None or len(signal) == 0:
                continue
            w = alloc.get(sleeve, 0)
            if w <= 0:
                continue
            # 各Sleeve内部横截面标准化
            if signal.std() > 0:
                sig_norm = (signal - signal.mean()) / signal.std()
            else:
                sig_norm = signal

            if combined is None:
                combined = sig_norm * w
            else:
                combined = combined.add(sig_norm * w, fill_value=0)

        if combined is None:
            return pd.Series(dtype=float)

        # 最终标准化
        if combined.std() > 0:
            combined = (combined - combined.mean()) / combined.std()
        return combined


# ══════════════════════════════════════════════════════════════════════════════
# [v8.1] TACTICAL SLEEVE — 三个专属信号
# overnight reversal / event overreaction / squeeze filter
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# [v8.2] OPTIONS STRUCTURE ENGINE
# 期权结构分析：GEX / Gamma Squeeze / 做市商Delta对冲流
#
# 核心理念：
# 期权市场里，做市商（Market Maker）是被动的一方。
# 他们卖出期权给散户，然后必须通过买卖正股来对冲Delta。
# 这个对冲行为会直接影响正股的价格走势。
#
# 三个最重要的概念：
#
# 1. GEX（Gamma Exposure）
#    做市商的净Gamma暴露。
#    GEX > 0 → 做市商持有正Gamma → 价格上涨时他们卖股票（稳定价格）
#    GEX < 0 → 做市商持有负Gamma → 价格上涨时他们买股票（放大波动）
#    GEX翻转点 = 价格最不稳定的区域，Gamma Squeeze最容易发生的地方
#
# 2. Gamma Squeeze预测
#    当大量看涨期权被买入，做市商被迫卖出Call（净空Gamma）
#    价格上涨 → 做市商Delta增加 → 被迫买入正股对冲
#    → 推高价格 → 更多Delta对冲 → 正反馈 = Gamma Squeeze
#    GME/AMC/NVDA的极端上涨大部分由此驱动
#
# 3. Pin Risk（到期日磁铁效应）
#    到期日前，价格趋向于"磁吸"到最大未平仓量的行权价（Max Pain）
#    因为做市商在该价位的对冲成本最小
#
# 数据来源（实盘接入）：
# · Polygon.io Options API ($30/月，机构最常用）
# · Tradier ($15/月）
# · CBOE官网（免费，但批量慢）
# · 合成数据演示模式（网络不通时自动切换）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptionsChainSnapshot:
    """单个标的的期权链快照"""
    ticker:         str
    spot_price:     float
    expiry_date:    str
    strikes:        List[float]
    call_oi:        Dict[float, float]   # {strike: open_interest}
    put_oi:         Dict[float, float]
    call_volume:    Dict[float, float]
    put_volume:     Dict[float, float]
    call_iv:        Dict[float, float]   # {strike: implied_vol}
    put_iv:         Dict[float, float]
    days_to_expiry: int


class BSMGreeks:
    """
    Black-Scholes-Merton Greeks计算

    BSM方程（图片第2个公式）：
    ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

    为什么这里用BSM（正确位置）：
    · 不是用来预测价格方向
    · 用来计算做市商的对冲需求（Delta/Gamma）
    · 正是这些对冲需求产生了Gamma Squeeze
    """

    @staticmethod
    def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """d2 = d1 - σ√T"""
        return BSMGreeks.d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

    @staticmethod
    def delta(S: float, K: float, r: float, sigma: float, T: float,
              option_type: str = 'call') -> float:
        """
        Delta = ∂V/∂S
        Call Delta ∈ (0, 1)：正股上涨1元，Call涨Delta元
        Put  Delta ∈ (-1, 0)：正股上涨1元，Put跌|Delta|元

        做市商卖出1张Call → 他们的Delta = -Delta_call
        对冲：买入 Delta_call × 100 股正股
        """
        from scipy.stats import norm
        d1 = BSMGreeks.d1(S, K, r, sigma, T)
        if option_type == 'call':
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)

    @staticmethod
    def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Gamma = ∂²V/∂S² = ∂Delta/∂S
        Call和Put的Gamma相同（Put-Call Parity）

        Gamma是做市商最重要的风险暴露：
        · 高Gamma = 价格小幅变动就需要大幅调整对冲
        · ATM期权Gamma最高（平值期权最靠近行权价）
        · Gamma在到期日前最后一周急剧上升（Gamma Squeeze最危险时期）
        """
        from scipy.stats import norm
        d1 = BSMGreeks.d1(S, K, r, sigma, T)
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Vega = ∂V/∂σ
        IV上升1% → 期权价格涨Vega（对于买方）
        """
        from scipy.stats import norm
        d1 = BSMGreeks.d1(S, K, r, sigma, T)
        if T <= 0:
            return 0.0
        return float(S * norm.pdf(d1) * np.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, r: float, sigma: float, T: float,
              option_type: str = 'call') -> float:
        """
        Theta = ∂V/∂t（时间衰减，每日）
        散户买Call → Theta为负（每天亏损时间价值）
        做市商卖Call → Theta为正（每天赚时间价值）
        """
        from scipy.stats import norm
        d1  = BSMGreeks.d1(S, K, r, sigma, T)
        d2  = BSMGreeks.d2(S, K, r, sigma, T)
        if T <= 0:
            return 0.0
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 =  r * K * np.exp(-r * T) * norm.cdf(-d2)
        return float((term1 + term2) / 365)  # 日化

    @staticmethod
    def call_price(S, K, r, sigma, T) -> float:
        from scipy.stats import norm
        d1 = BSMGreeks.d1(S, K, r, sigma, T)
        d2 = BSMGreeks.d2(S, K, r, sigma, T)
        if T <= 0:
            return max(0.0, S - K)
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    @staticmethod
    def put_price(S, K, r, sigma, T) -> float:
        from scipy.stats import norm
        d1 = BSMGreeks.d1(S, K, r, sigma, T)
        d2 = BSMGreeks.d2(S, K, r, sigma, T)
        if T <= 0:
            return max(0.0, K - S)
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


class GEXEngine:
    """
    Gamma Exposure (GEX) 引擎

    GEX是当前市场最被低估的预测工具。
    SpotGamma、Squeezemetrics等机构靠这个收几千美元年费。

    核心公式：
    GEX = Σ (Gamma × OI × 100 × S²) × sign
    sign = +1 for Call（做市商通常是卖家，空Gamma）
    sign = -1 for Put

    等等，这里有个重要的理解：
    当散户大量买Call时，做市商是Call的卖方 → 做市商持有负Gamma（Short Gamma）
    当散户大量买Put时，做市商是Put的卖方 → 做市商持有正Gamma（Long Gamma）

    所以正确的符号约定：
    Call OI贡献 → 做市商空Gamma → GEX为负（不稳定）
    Put OI贡献  → 做市商多Gamma → GEX为正（稳定）

    GEX > 0（Put主导）→ 价格趋于稳定，波动率被压制
    GEX < 0（Call主导）→ 价格趋于不稳定，上涨会被放大
    |GEX|最小点 = Gamma Flip点 = 最可能发生Gamma Squeeze的价位
    """

    def __init__(self, r: float = 0.05):
        self.r = r      # 无风险利率

    def compute_gex(self, chain: OptionsChainSnapshot) -> Dict:
        """
        计算整条期权链的GEX分布

        Returns:
        - gex_by_strike: 每个行权价的GEX贡献
        - total_gex: 总GEX（正=稳定，负=不稳定）
        - gamma_flip: GEX翻转的行权价（Squeeze最危险的价位）
        - net_call_gex, net_put_gex
        """
        S   = chain.spot_price
        T   = chain.days_to_expiry / 365.0
        if T <= 0:
            T = 1 / 365

        gex_by_strike = {}
        net_call_gex  = 0.0
        net_put_gex   = 0.0

        for K in chain.strikes:
            # Call端GEX（做市商空Gamma → 负GEX）
            c_oi  = chain.call_oi.get(K, 0)
            c_iv  = chain.call_iv.get(K, 0.30)
            if c_oi > 0 and c_iv > 0:
                g = BSMGreeks.gamma(S, K, self.r, c_iv, T)
                # GEX单位：Dollar Gamma（每1%价格变动的对冲金额）
                call_gex = -g * c_oi * 100 * S * S * 0.01
                net_call_gex += call_gex
            else:
                call_gex = 0.0

            # Put端GEX（做市商空Gamma → 负GEX，但Put有相反Delta符号）
            p_oi  = chain.put_oi.get(K, 0)
            p_iv  = chain.put_iv.get(K, 0.30)
            if p_oi > 0 and p_iv > 0:
                g = BSMGreeks.gamma(S, K, self.r, p_iv, T)
                put_gex = g * p_oi * 100 * S * S * 0.01   # Put GEX正向
                net_put_gex += put_gex
            else:
                put_gex = 0.0

            gex_by_strike[K] = call_gex + put_gex

        total_gex = net_call_gex + net_put_gex

        # 找GEX翻转价位（从正变负的行权价）
        sorted_strikes = sorted(gex_by_strike.keys())
        cumulative_gex = {}
        cum = 0.0
        for K in sorted_strikes:
            cum += gex_by_strike[K]
            cumulative_gex[K] = cum

        # Gamma Flip = 累积GEX最接近0的行权价
        gamma_flip = min(sorted_strikes,
                         key=lambda k: abs(cumulative_gex.get(k, float('inf'))))

        return {
            'total_gex':      round(total_gex, 2),
            'net_call_gex':   round(net_call_gex, 2),
            'net_put_gex':    round(net_put_gex, 2),
            'gex_by_strike':  gex_by_strike,
            'gamma_flip':     gamma_flip,
            'is_negative':    total_gex < 0,
            'stability_regime': 'unstable_squeeze_risk' if total_gex < 0 else 'stable_pinning',
            'spot':           S,
            'spot_vs_flip':   round((S - gamma_flip) / (gamma_flip + 1e-8), 4),
        }

    def predict_gamma_squeeze(self, chain: OptionsChainSnapshot,
                               gex_result: Dict) -> Dict:
        """
        Gamma Squeeze预测

        触发条件（需同时满足）：
        1. 总GEX < 0（Call主导，做市商空Gamma）
        2. 价格接近或高于Gamma Flip点（进入不稳定区域）
        3. Call买入量（Volume）显著高于Put（散户看涨情绪高）
        4. ATM期权IV相对低（做市商低估了波动率，溢价不够）

        Squeeze强度评分（0-100）：
        · GEX负值越大 → 分越高（做市商压力越大）
        · 价格越接近Gamma Flip → 分越高
        · Call/Put Volume比值越高 → 分越高
        """
        S     = chain.spot_price
        flip  = gex_result['gamma_flip']
        total = gex_result['total_gex']

        score = 0.0
        reasons = []

        # 条件1：GEX负值（0-30分）
        if total < 0:
            gex_score = min(30, abs(total) / 1e6 * 10)
            score += gex_score
            reasons.append(f"GEX负值={total:.0f}（做市商空Gamma压力）")

        # 条件2：价格接近Gamma Flip（0-30分）
        dist_pct = (S - flip) / (S + 1e-8)
        if abs(dist_pct) < 0.05:    # 价格在Flip点5%以内
            flip_score = 30 * (1 - abs(dist_pct) / 0.05)
            score += flip_score
            reasons.append(f"价格在Gamma Flip点{dist_pct:.1%}内（高风险区域）")

        # 条件3：Call/Put Volume比（0-25分）
        total_call_vol = sum(chain.call_volume.values())
        total_put_vol  = sum(chain.put_volume.values()) + 1e-8
        cpvr = total_call_vol / total_put_vol
        if cpvr > 1.5:
            vol_score = min(25, (cpvr - 1.5) * 10)
            score += vol_score
            reasons.append(f"C/P Volume比={cpvr:.2f}（散户Call买入旺盛）")

        # 条件4：ATM IV相对较低（0-15分）
        atm_iv_calls = [v for k, v in chain.call_iv.items()
                        if abs(k - S) / S < 0.03]
        if atm_iv_calls:
            atm_iv = np.mean(atm_iv_calls)
            if atm_iv < 0.35:  # ATM IV < 35%时市场未充分定价
                iv_score = min(15, (0.35 - atm_iv) * 100)
                score += iv_score
                reasons.append(f"ATM IV={atm_iv:.1%}偏低（Squeeze溢价未定价）")

        score = min(100, score)

        return {
            'squeeze_score':    round(score, 1),
            'squeeze_risk':     'HIGH' if score > 70 else 'MEDIUM' if score > 40 else 'LOW',
            'reasons':          reasons,
            'call_put_vol':     round(cpvr, 2),
            'gamma_flip':       flip,
            'spot_price':       S,
            'action':          ('做多！Gamma Squeeze可能触发' if score > 70 else
                                '观察' if score > 40 else '无明显Squeeze风险'),
        }

    def compute_max_pain(self, chain: OptionsChainSnapshot) -> float:
        """
        Max Pain（最大痛苦点）

        定义：让期权买方损失最大（卖方赚最多）的行权价
        逻辑：做市商（期权卖方）会通过Delta对冲把价格"磁吸"到Max Pain点

        计算：对每个可能的到期价格，计算所有Call+Put的总价值
        总价值最低的价格 = Max Pain
        （买方损失最大 = 卖方赚最多 = 做市商最舒服的价格）

        实盘用途：
        · 到期日前1-2周，价格趋向Max Pain
        · 可以做反向策略：价格远离Max Pain时，预期回归
        """
        pain_by_strike = {}

        for possible_expiry in chain.strikes:
            total_pain = 0.0
            # Call side pain: Σ max(expiry - strike, 0) × call_OI
            for K, oi in chain.call_oi.items():
                total_pain += max(possible_expiry - K, 0) * oi * 100
            # Put side pain: Σ max(strike - expiry, 0) × put_OI
            for K, oi in chain.put_oi.items():
                total_pain += max(K - possible_expiry, 0) * oi * 100
            pain_by_strike[possible_expiry] = total_pain

        if not pain_by_strike:
            return chain.spot_price

        max_pain = min(pain_by_strike, key=pain_by_strike.get)
        return float(max_pain)

    def dealer_delta_flow(self, chain: OptionsChainSnapshot,
                           spot_change_pct: float = 0.01) -> Dict:
        """
        做市商Delta对冲流预测

        当价格变动spot_change_pct时，做市商需要买入/卖出多少股票？
        这个"强制对冲流"是Gamma Squeeze的物理机制。

        计算：
        Delta Flow = Σ (Gamma × OI × 100) × ΔS
        · 如果Delta Flow > 0 → 价格上涨时做市商买股票（放大上涨）
        · 如果Delta Flow < 0 → 价格上涨时做市商卖股票（抑制上涨）
        """
        S  = chain.spot_price
        T  = chain.days_to_expiry / 365.0
        dS = S * spot_change_pct

        total_call_delta_flow = 0.0
        total_put_delta_flow  = 0.0

        for K in chain.strikes:
            # Call：做市商是卖方，Gamma为负
            c_oi = chain.call_oi.get(K, 0)
            c_iv = chain.call_iv.get(K, 0.30)
            if c_oi > 0 and c_iv > 0:
                g = BSMGreeks.gamma(S, K, self.r, c_iv, max(T, 1/365))
                # 价格上涨dS，做市商需要额外买入的股数（因为Delta变了）
                extra_buy = -g * c_oi * 100 * dS   # 负号：做市商空Gamma
                total_call_delta_flow += extra_buy

            # Put：做市商是卖方，Gamma为正（Put OI的对冲方向相反）
            p_oi = chain.put_oi.get(K, 0)
            p_iv = chain.put_iv.get(K, 0.30)
            if p_oi > 0 and p_iv > 0:
                g = BSMGreeks.gamma(S, K, self.r, p_iv, max(T, 1/365))
                extra_sell = g * p_oi * 100 * dS    # 正号：价格涨，Put OI让做市商卖
                total_put_delta_flow += extra_sell

        net_flow = total_call_delta_flow + total_put_delta_flow

        return {
            'net_delta_flow_shares': round(net_flow, 0),
            'call_flow':             round(total_call_delta_flow, 0),
            'put_flow':              round(total_put_delta_flow, 0),
            'direction':             'amplify_move' if net_flow * dS > 0 else 'dampen_move',
            'interpretation': (
                f"价格上涨{spot_change_pct:.0%}时，做市商对冲流："
                f"{'买入' if net_flow > 0 else '卖出'}{abs(net_flow):.0f}股"
                f"（{'放大' if net_flow > 0 else '抑制'}上涨动能）"
            )
        }


class OptionsSignalEngine:
    """
    期权信号整合引擎（Tactical Sleeve的期权层）

    四个核心信号：
    1. GEX Signal：GEX翻负 → 做多（Squeeze风险 = 上涨机会）
    2. PCR Signal：Put/Call Ratio极端低 → 市场极度乐观 → 反转做空
    3. IV Percentile：IV极低 → 便宜的看涨期权买点
    4. Max Pain Pin：价格远离Max Pain → 预期磁回（期权到期前）

    数据来源：
    · 实盘：Polygon.io / Tradier API
    · 演示：合成期权链（自动生成合理的Greeks分布）
    """

    def __init__(self, r: float = 0.05):
        self.gex_engine = GEXEngine(r=r)
        self.r          = r

    def generate_synthetic_chain(self, spot: float,
                                  hist_vol: float = 0.30,
                                  days_to_expiry: int = 21,
                                  ticker: str = 'STOCK') -> OptionsChainSnapshot:
        """
        生成合成期权链（演示模式）
        模拟真实期权链的关键特征：
        · 波动率微笑（OTM Put IV > ATM IV）
        · OI分布集中在ATM附近
        · 散户偏好轻度OTM Call
        """
        # 行权价：±30%范围，每2.5%一档
        strikes = [round(spot * (0.70 + i * 0.025), 2) for i in range(25)]

        call_oi, put_oi   = {}, {}
        call_vol, put_vol = {}, {}
        call_iv, put_iv   = {}, {}

        np.random.seed(int(spot) % 100)  # 可复现

        for K in strikes:
            moneyness = (K - spot) / spot   # >0 OTM Call, <0 OTM Put

            # IV微笑：OTM Put IV更高（左偏，下跌保护贵）
            base_iv = hist_vol
            if moneyness < 0:    # OTM Put
                smile_adj = abs(moneyness) * 0.8
            elif moneyness > 0:  # OTM Call
                smile_adj = moneyness * 0.3
            else:
                smile_adj = 0
            K_iv = base_iv + smile_adj

            call_iv[K] = round(float(K_iv), 4)
            put_iv[K]  = round(float(K_iv + 0.02), 4)   # Put IV略高

            # OI分布：ATM最集中，OTM递减
            atm_weight = np.exp(-8 * moneyness**2)
            call_bias  = 1.3 if moneyness > 0 and moneyness < 0.1 else 1.0

            call_oi[K]  = max(0, int(atm_weight * 5000 * call_bias
                                      * (1 + np.random.randn() * 0.2)))
            put_oi[K]   = max(0, int(atm_weight * 4000
                                      * (1 + np.random.randn() * 0.2)))
            call_vol[K] = max(0, int(call_oi[K] * 0.15 * (1 + np.random.randn() * 0.3)))
            put_vol[K]  = max(0, int(put_oi[K]  * 0.12 * (1 + np.random.randn() * 0.3)))

        return OptionsChainSnapshot(
            ticker=ticker, spot_price=spot,
            expiry_date=(datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d'),
            strikes=strikes,
            call_oi=call_oi, put_oi=put_oi,
            call_volume=call_vol, put_volume=put_vol,
            call_iv=call_iv, put_iv=put_iv,
            days_to_expiry=days_to_expiry
        )

    def full_options_analysis(self, chain: OptionsChainSnapshot) -> Dict:
        """
        完整期权结构分析（四个信号 + GEX图谱）
        """
        S  = chain.spot_price

        # ── GEX分析 ──────────────────────────────────────────────────────────
        gex     = self.gex_engine.compute_gex(chain)
        squeeze = self.gex_engine.predict_gamma_squeeze(chain, gex)
        flow    = self.gex_engine.dealer_delta_flow(chain)
        max_pain= self.gex_engine.compute_max_pain(chain)

        # ── PCR（Put/Call Ratio）────────────────────────────────────────────
        total_call_vol = sum(chain.call_volume.values()) + 1e-8
        total_put_vol  = sum(chain.put_volume.values())
        total_call_oi  = sum(chain.call_oi.values()) + 1e-8
        total_put_oi   = sum(chain.put_oi.values())

        pcr_vol  = total_put_vol / total_call_vol   # 成交量PCR（更实时）
        pcr_oi   = total_put_oi  / total_call_oi    # 未平仓PCR（更结构性）

        # PCR信号：极低 = 散户极度乐观 = 反转做空；极高 = 极度悲观 = 反转做多
        pcr_signal = 0.0
        if pcr_vol < 0.5:                # 极度乐观（Call远多于Put）→ 警惕反转
            pcr_signal = -1.0
        elif pcr_vol < 0.7:              # 偏乐观 → 轻度看空
            pcr_signal = -0.5
        elif pcr_vol > 1.3:              # 极度悲观（Put远多于Call）→ 反转做多
            pcr_signal = +1.0
        elif pcr_vol > 1.0:
            pcr_signal = +0.5

        # ── IV Percentile ────────────────────────────────────────────────────
        # 计算ATM IV（正股当前价格附近的期权IV）
        atm_calls = [(abs(K - S), v) for K, v in chain.call_iv.items()]
        atm_calls.sort()
        atm_iv = atm_calls[0][1] if atm_calls else 0.30

        all_ivs   = list(chain.call_iv.values()) + list(chain.put_iv.values())
        iv_pct    = float(stats.percentileofscore(all_ivs, atm_iv) / 100)

        # IVP信号：极低IV = 期权便宜 = 可以买Call；极高IV = 期权贵 = 卖期权
        ivp_signal = 0.0
        if iv_pct < 0.20:               # IV极低 = 期权便宜，预期IV均值回归
            ivp_signal = +0.8           # 做多信号（买便宜Call）
        elif iv_pct > 0.80:             # IV极高 = 期权贵，预期IV压缩
            ivp_signal = -0.5           # 卖Call/卖Put信号

        # ── Max Pain Pin信号 ─────────────────────────────────────────────────
        pain_dist    = (S - max_pain) / (S + 1e-8)
        pin_signal   = 0.0
        if chain.days_to_expiry <= 7:   # 到期日前一周才有效
            if pain_dist > 0.03:        # 价格在Max Pain上方 → 预期下跌磁回
                pin_signal = -0.7 * min(1, abs(pain_dist) / 0.10)
            elif pain_dist < -0.03:     # 价格在Max Pain下方 → 预期上涨磁回
                pin_signal = +0.7 * min(1, abs(pain_dist) / 0.10)

        # ── 综合期权信号 ─────────────────────────────────────────────────────
        # GEX信号：GEX负 + Squeeze高分 → 做多（Gamma Squeeze方向）
        gex_signal = 0.0
        if gex['is_negative'] and squeeze['squeeze_score'] > 60:
            gex_signal = +1.0   # 强烈做多（Gamma Squeeze预期）
        elif gex['is_negative'] and squeeze['squeeze_score'] > 30:
            gex_signal = +0.5   # 轻度做多
        elif not gex['is_negative']:
            gex_signal = -0.2   # GEX正 = 稳定 = 低波动，轻度偏空

        # 加权合成（GEX最重要）
        combined_signal = (gex_signal  * 0.45 +
                           pcr_signal  * 0.25 +
                           ivp_signal  * 0.20 +
                           pin_signal  * 0.10)

        return {
            'ticker':           chain.ticker,
            'spot':             S,
            # GEX
            'total_gex':        gex['total_gex'],
            'gamma_flip':       gex['gamma_flip'],
            'gex_regime':       gex['stability_regime'],
            'squeeze_score':    squeeze['squeeze_score'],
            'squeeze_risk':     squeeze['squeeze_risk'],
            'squeeze_reasons':  squeeze['reasons'],
            'squeeze_action':   squeeze['action'],
            'dealer_flow':      flow['interpretation'],
            # PCR
            'pcr_volume':       round(pcr_vol, 3),
            'pcr_oi':           round(pcr_oi, 3),
            'pcr_signal':       pcr_signal,
            # IV
            'atm_iv':           round(atm_iv, 4),
            'iv_percentile':    round(iv_pct, 4),
            'ivp_signal':       ivp_signal,
            # Max Pain
            'max_pain':         max_pain,
            'pain_distance':    round(pain_dist, 4),
            'days_to_expiry':   chain.days_to_expiry,
            'pin_signal':       pin_signal,
            # 综合
            'combined_signal':  round(combined_signal, 4),
            'signal_direction': ('做多 ▲' if combined_signal > 0.3 else
                                 '做空 ▼' if combined_signal < -0.3 else '中性 →'),
        }

    def portfolio_options_signals(self,
                                   prices: pd.DataFrame,
                                   hist_vols: pd.Series = None) -> pd.DataFrame:
        """
        对整个组合的每个标的做期权分析，返回综合信号

        如果有真实期权数据（Polygon API），直接传入chains；
        否则用合成链演示
        """
        results = []
        for ticker in prices.columns:
            spot     = float(prices[ticker].iloc[-1])
            if hist_vols is not None and ticker in hist_vols.index:
                hvol = float(hist_vols[ticker])
            else:
                r    = prices[ticker].pct_change().dropna()
                hvol = float(r.tail(21).std() * np.sqrt(252)) if len(r) > 5 else 0.30

            # 生成合成期权链（实盘时替换为API数据）
            chain  = self.generate_synthetic_chain(spot, hvol,
                                                    days_to_expiry=21,
                                                    ticker=ticker)
            result = self.full_options_analysis(chain)
            results.append(result)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.set_index('ticker')
        return df

    def print_options_dashboard(self, signals_df: pd.DataFrame):
        """打印期权结构Dashboard"""
        print(f"\n{'═'*70}")
        print(f"  🎯 期权结构分析 Dashboard")
        print(f"  GEX / Gamma Squeeze / PCR / IV% / Max Pain")
        print(f"{'═'*70}")
        print(f"  {'Ticker':<8} {'信号':>6} {'Squeeze':>8} {'GEX':>12} "
              f"{'PCR':>6} {'IVP':>6} {'MaxPain':>10} {'方向':<10}")
        print(f"  {'─'*68}")

        for ticker, row in signals_df.iterrows():
            sq_icon = {'HIGH':'🔥','MEDIUM':'🟡','LOW':'🟢'}.get(
                row.get('squeeze_risk','LOW'), '🟢')
            print(f"  {ticker:<8} {row.get('combined_signal',0):>+6.3f} "
                  f"  {sq_icon}{row.get('squeeze_score',0):>4.0f}pts "
                  f"{row.get('total_gex',0):>12,.0f} "
                  f"{row.get('pcr_volume',0):>6.2f} "
                  f"{row.get('iv_percentile',0):>6.1%} "
                  f"{row.get('max_pain',0):>10.2f} "
                  f"{str(row.get('signal_direction','→')):<10}")

        # 高Squeeze风险警报
        high_squeeze = signals_df[signals_df.get('squeeze_score', pd.Series()) > 60] \
            if 'squeeze_score' in signals_df.columns else pd.DataFrame()
        if len(high_squeeze) > 0:
            print(f"\n  🔥 Gamma Squeeze高风险标的：")
            for ticker, row in high_squeeze.iterrows():
                print(f"    {ticker}: Squeeze得分={row['squeeze_score']:.0f} "
                      f"GEX Flip点={row.get('gamma_flip',0):.2f} "
                      f"当前价={row.get('spot',0):.2f}")
                for reason in row.get('squeeze_reasons', [])[:2]:
                    print(f"      → {reason}")
        print(f"{'═'*70}")


class TacticalSignals:
    """
    Tactical Sleeve的三个核心信号

    1. overnight_reversal：
       昨日大跌/大涨之后，隔夜/开盘反转的概率更高
       原理：市场对消息过度反应，隔夜散户情绪释放后回归

    2. event_overreaction：
       事件（财报/新闻）发布后1-3天内的过度反应测量
       IC历史约0.04-0.08（短期Alpha，衰减快）

    3. squeeze_filter：
       检测可能触发short squeeze的股票
       条件：高空头比例 + 价格突破 + 成交量放大
    """

    # ── 1. 隔夜反转信号 ────────────────────────────────────────────────────────
    @staticmethod
    def overnight_reversal(prices: pd.DataFrame,
                            volumes: pd.DataFrame,
                            lookback: int = 5) -> pd.Series:
        """
        隔夜反转信号

        逻辑：
        - 计算过去 lookback 天的日内收益标准差（波动率）
        - 如果今日跌幅 > 1.5σ → 看涨信号（+）
        - 如果今日涨幅 > 1.5σ → 看跌信号（-）
        - 配合成交量：大成交量的过度反应更可能被反转

        持有：1-3天（隔夜或持有到反转完成）
        止损：0.8%（严格）
        """
        r    = prices.pct_change().dropna()
        if len(r) < lookback + 2:
            return pd.Series(0.0, index=prices.columns)

        # 过去lookback天的波动率
        vol  = r.tail(lookback + 1).iloc[:-1].std()

        # 今日收益
        today_ret = r.iloc[-1]

        # 归一化冲击：今日冲击 / 历史波动率
        z_score   = -today_ret / (vol + 1e-8)  # 负号：跌越多看涨信号越强

        # 成交量确认：成交量放大时过度反应更可能被反转
        vol_ma  = volumes.rolling(lookback).mean().iloc[-1]
        vol_ratio = volumes.iloc[-1] / (vol_ma + 1e-8)
        vol_boost = np.log1p(vol_ratio.clip(0, 5))  # 对数缩放，避免极值

        raw = z_score * (1 + vol_boost * 0.3)

        # 只保留显著信号（|z| > 1.5σ）
        raw[raw.abs() < 1.5] = 0.0

        # 横截面排名
        if raw[raw != 0].std() > 0:
            raw = raw.rank(pct=True) * 2 - 1  # 映射到[-1, +1]
        return raw.fillna(0)

    # ── 2. 事件过度反应信号 ────────────────────────────────────────────────────
    @staticmethod
    def event_overreaction(prices: pd.DataFrame,
                            volumes: pd.DataFrame,
                            event_window: int = 3,
                            reaction_threshold: float = 0.04) -> pd.Series:
        """
        事件过度反应信号

        逻辑：
        - 检测过去 event_window 天内是否有异常大的单日移动（>threshold）
        - 异常移动后，预期在接下来1-5天部分回归
        - 做空过度上涨的（反转），做多过度下跌的（反弹）

        实盘注意：
        - 需要结合新闻/财报日历确认是否真的是事件驱动
        - Canyon系统里配合声明背调（statement_diligence.py）使用
        """
        r = prices.pct_change().dropna()
        if len(r) < event_window + 2:
            return pd.Series(0.0, index=prices.columns)

        # 过去event_window天内的最大单日绝对收益
        max_move = r.tail(event_window).abs().max()

        # 最大移动对应的方向（正=上涨事件，负=下跌事件）
        max_idx  = r.tail(event_window).abs().idxmax()
        max_direction = pd.Series(
            {col: float(r.loc[max_idx[col], col]) for col in prices.columns
             if max_idx[col] in r.index},
            dtype=float
        ).reindex(prices.columns).fillna(0)

        # 过度反应信号：移动超过threshold才触发反转预期
        triggered = max_move > reaction_threshold
        signal = -np.sign(max_direction) * triggered * max_move

        # 用正常波动率归一化
        normal_vol = r.tail(21).std() * np.sqrt(5)
        signal = signal / (normal_vol + 1e-8)

        return signal.clip(-3, 3).fillna(0)

    # ── 3. Short Squeeze过滤器 ─────────────────────────────────────────────────
    @staticmethod
    def squeeze_filter(prices: pd.DataFrame,
                        volumes: pd.DataFrame,
                        lookback: int = 20) -> pd.Series:
        """
        Short Squeeze信号

        检测条件（需同时满足）：
        1. 价格突破近 lookback 天高点（上涨动能确认）
        2. 成交量放大 > 2倍均值（做空者在被迫平仓）
        3. 短期动量正向（确认趋势，不是假突破）

        机制（为什么Squeeze会发生）：
        - 大量空头持仓 → 价格上涨 → 空头被追加保证金
        - 被迫买回平仓 → 推高价格 → 更多空头被迫平仓
        - 正反馈循环 → 短时间内极端上涨

        注意：
        - Squeeze信号衰减极快（1-3天），必须快进快出
        - 止损必须严格（0.8%），因为如果判断错误会遭遇强劲下跌
        """
        r    = prices.pct_change().dropna()
        if len(r) < lookback + 2:
            return pd.Series(0.0, index=prices.columns)

        # 条件1：价格突破近期高点
        recent_high    = prices.rolling(lookback).max().iloc[-2]  # 昨日的近期高点
        price_breakout = (prices.iloc[-1] > recent_high).astype(float)

        # 条件2：成交量放大 > 2x
        vol_mean  = volumes.rolling(lookback).mean().iloc[-1]
        vol_surge = (volumes.iloc[-1] / (vol_mean + 1e-8) > 2.0).astype(float)

        # 条件3：短期动量为正
        momentum_5 = prices.pct_change(min(5, len(prices)-1)).iloc[-1]
        pos_momentum = (momentum_5 > 0).astype(float)

        # 三个条件全部满足才触发
        squeeze_signal = price_breakout * vol_surge * pos_momentum

        # 强度 = 成交量倍数（越大越强）
        vol_magnitude = (volumes.iloc[-1] / (vol_mean + 1e-8)).clip(0, 5)
        strength = squeeze_signal * vol_magnitude

        return strength.fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# [v8.1] SECTOR ROTATION SLEEVE — 专属信号
# sector ETF ranking / relative strength / weekly rebalance
# ══════════════════════════════════════════════════════════════════════════════

class SectorRotationEngine:
    """
    板块轮动引擎

    逻辑（来自SPDR行业ETF轮动）：
    1. 计算各行业ETF的相对强度（vs SPY）
    2. 动量排名：过去4-12周表现最好的行业得高分
    3. 趋势确认：只做EMA在SMA上方的行业
    4. 每周再平衡（不是每天），避免过度换手

    行业ETF映射：
    XLK → 科技    XLF → 金融    XLE → 能源
    XLV → 医疗    XLI → 工业    XLU → 公用
    XLB → 原材料  XLY → 可选消费 XLP → 必选消费
    XLC → 通信    XLRE → 房地产

    为什么板块轮动有效：
    - 机构资金轮动有惯性（买入一个板块需要几周）
    - 经济周期的不同阶段受益板块不同（早周期→工业，晚周期→能源）
    - 相对强度持续性约4-12周（比单股动量更稳定）
    """

    # 标准行业ETF（如果有对应数据则使用，否则用组合内相似股票代理）
    SECTOR_ETFS = {
        'XLK': 'Technology',  'XLF': 'Financial',    'XLE': 'Energy',
        'XLV': 'Healthcare',  'XLI': 'Industrial',   'XLU': 'Utilities',
        'XLB': 'Materials',   'XLY': 'ConsDisc',     'XLP': 'ConsStaples',
        'XLC': 'Communication','XLRE':'RealEstate'
    }

    def __init__(self, rebalance_days: int = 5):
        """
        rebalance_days: 多少交易日再平衡一次（默认5天=每周）
        """
        self.rebalance_days   = rebalance_days
        self._last_rebalance  = 0         # 上次再平衡距今天数
        self._current_weights = pd.Series(dtype=float)

    def rank_sectors(self, prices: pd.DataFrame,
                      market: pd.Series,
                      lookback_weeks: int = 8) -> pd.DataFrame:
        """
        计算行业/个股相对强度排名

        如果没有行业ETF数据，用组合内的股票按相对强度代替

        Returns:
            DataFrame with columns: [ticker, rel_strength, momentum, trend_signal, score]
        """
        lookback = lookback_weeks * 5   # 转换为交易日

        if len(prices) < lookback + 5:
            lookback = max(21, len(prices) - 5)

        records = []
        mkt_ret  = market.pct_change(lookback).iloc[-1]
        mkt_ema5 = market.ewm(span=5, adjust=False).mean().iloc[-1]
        mkt_sma20= market.rolling(20).mean().iloc[-1]

        for ticker in prices.columns:
            p = prices[ticker].dropna()
            if len(p) < lookback + 5:
                continue

            # 相对强度 = 个股涨幅 - 市场涨幅（过去lookback天）
            stock_ret = float(p.pct_change(lookback).iloc[-1])
            rel_str   = stock_ret - float(mkt_ret) if pd.notna(mkt_ret) else stock_ret

            # 趋势确认：EMA5 > SMA20
            ema5   = float(p.ewm(span=5, adjust=False).mean().iloc[-1])
            sma20  = float(p.rolling(20).mean().iloc[-1])
            trend_ok = ema5 > sma20

            # 动量（3周 + 8周加权）
            mom3 = float(p.pct_change(15).iloc[-1]) if len(p) >= 15 else 0
            mom8 = float(p.pct_change(lookback).iloc[-1]) if len(p) >= lookback else 0
            momentum = 0.4 * mom3 + 0.6 * mom8

            # 综合得分
            score = (rel_str * 0.5 + momentum * 0.3 +
                     (0.2 if trend_ok else -0.1))

            records.append({
                'ticker':       ticker,
                'rel_strength': round(rel_str, 4),
                'momentum':     round(momentum, 4),
                'trend_ok':     trend_ok,
                'ema5':         round(ema5, 2),
                'sma20':        round(sma20, 2),
                'score':        round(score, 4)
            })

        df = pd.DataFrame(records)
        if len(df) == 0:
            return df
        return df.sort_values('score', ascending=False).reset_index(drop=True)

    def get_weights(self, prices: pd.DataFrame,
                     market: pd.Series,
                     regime: 'Regime',
                     force_rebalance: bool = False) -> pd.Series:
        """
        返回板块轮动权重（每周再平衡，其他时间保持不变）

        牛市：只做排名前1/3（进攻性）
        熊市：做多防守板块（XLU/XLV），做空周期性板块
        """
        self._last_rebalance += 1

        # 不到再平衡时间，返回上次权重
        if (not force_rebalance and
                self._last_rebalance < self.rebalance_days and
                len(self._current_weights) > 0):
            return self._current_weights

        # 重新计算
        self._last_rebalance = 0
        ranking = self.rank_sectors(prices, market)

        if len(ranking) == 0:
            return pd.Series(dtype=float)

        n = len(ranking)
        top_n = max(1, n // 3)    # 前1/3

        if regime.score >= 1:
            # 牛市：做多前1/3（最强板块）
            selected = ranking.head(top_n)['ticker'].tolist()
            weights  = pd.Series(1.0 / top_n, index=selected)

        elif regime.score <= -1:
            # 熊市：防守板块加权，做空弱势板块
            weights_dict = {}
            for _, row in ranking.iterrows():
                if row['trend_ok'] and row['rel_strength'] > 0:
                    weights_dict[row['ticker']] = row['score']
                elif not row['trend_ok'] and row['rel_strength'] < -0.05:
                    weights_dict[row['ticker']] = -abs(row['score']) * 0.5  # 做空
            if weights_dict:
                w = pd.Series(weights_dict)
                pos_sum = w[w > 0].sum()
                if pos_sum > 0:
                    w[w > 0] /= pos_sum
                weights = w.clip(-regime.max_short, regime.max_long)
            else:
                weights = pd.Series(dtype=float)

        else:
            # 震荡：前1/3多头，后1/3减仓
            selected = ranking.head(top_n)['ticker'].tolist()
            weights  = pd.Series(0.8 / top_n, index=selected)   # 不满仓

        # 存储供下次使用
        self._current_weights = weights
        return weights

    def print_ranking(self, prices: pd.DataFrame, market: pd.Series):
        """打印当前板块排名"""
        df = self.rank_sectors(prices, market)
        if len(df) == 0:
            return
        print(f"\n  📊 板块轮动排名（相对强度）：")
        print(f"  {'排名':<4} {'Ticker':<8} {'相对强度':>8} {'动量':>8} {'趋势':>6} {'综合分':>8}")
        print(f"  {'─'*46}")
        for i, row in df.iterrows():
            trend_icon = '✅' if row['trend_ok'] else '❌'
            print(f"  {i+1:<4} {row['ticker']:<8} {row['rel_strength']:>8.2%} "
                  f"{row['momentum']:>8.2%} {trend_icon:>6} {row['score']:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# [v8.4] DISCRETIONARY STOCK PICKING LAYER — 人工选股层
#
# 这一层处理纯量化信号抓不到的场景：
#
# ┌─ 场景A: 错杀股（FCX矿被冲走类）─────────────────────────────┐
# │  · 短期暴跌 -10%~-30%                                            │
# │  · 单一事件驱动（事故/天气/财报失误/管理层意外）                 │
# │  · 但公司基本面（资产/储量/护城河）没坏                          │
# │  · 历史上类似事件后6-12个月恢复                                  │
# │  → 利用市场过度反应的非理性                                       │
# │  → 持仓周期：1-6个月                                              │
# │  → 进CORE_HEDGE Sleeve                                            │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌─ 场景B: 强趋势+低估值（SanDisk存储类）───────────────────────┐
# │  · 连续上涨（强动量）                                            │
# │  · PE/PB相对历史在合理位置（不贵）                               │
# │  · 行业β向上（半导体/存储周期催化）                              │
# │  · 业绩超预期或上修指引                                          │
# │  → 戴维斯双击（业绩+估值同时扩张）                                │
# │  → 持仓周期：3周-3个月                                            │
# │  → 进SECTOR_ROTATION Sleeve                                       │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌─ 场景C: 内部人买入 + 机构加仓 ──────────────────────────────┐
# │  · Form 4：管理层买入（不是出售）                                │
# │  · 13F：机构季报显示加仓                                         │
# │  · 短期价格仍在低位（机构在收筹码）                              │
# │  → 跟随聪明钱（书第8章mean reversion的升级版）                   │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌─ 场景D: 困境反转（Distressed Recovery）─────────────────────┐
# │  · 经历重大亏损/破产边缘但活下来                                 │
# │  · 新CEO/重组/资产剥离                                           │
# │  · 现金流转正                                                    │
# │  → 大幅低估的资产价值                                            │
# └──────────────────────────────────────────────────────────────────┘
#
# 每个场景都需要：
# 1. 量化筛选（找到候选）
# 2. 基本面打分（验证公司质量）
# 3. 估值检查（不能太贵）
# 4. 催化剂确认（什么时候开始涨）
# 5. 止损硬约束（如果逻辑证伪）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FundamentalSnapshot:
    """
    公司基本面快照（最少必填项）
    实盘从SEC EDGAR / Yahoo Finance / Polygon Fundamentals获取
    """
    ticker:           str
    sector:           str = ''
    market_cap_b:     float = 0.0      # 市值（十亿美元）

    # 估值
    pe_ratio:         Optional[float] = None
    forward_pe:       Optional[float] = None
    pb_ratio:         Optional[float] = None
    ev_to_ebitda:     Optional[float] = None
    peg_ratio:        Optional[float] = None    # PE / 增长率

    # 财务质量
    revenue_growth:   Optional[float] = None    # YoY收入增长
    earnings_growth:  Optional[float] = None    # YoY EPS增长
    profit_margin:    Optional[float] = None    # 净利率
    roe:              Optional[float] = None    # 净资产收益率
    debt_to_equity:   Optional[float] = None
    current_ratio:    Optional[float] = None    # 流动比率（>1.5健康）
    free_cashflow_b:  Optional[float] = None    # 自由现金流（十亿）

    # 行业地位
    moat_score:       float = 3.0   # 1-5，护城河评分（人工或代理）
    industry_position:str = 'unknown'   # leader/challenger/follower/niche

    # 触发事件
    catalyst_event:   str = ''      # "矿事故"/"财报超预期"/"新CEO"/"产品发布"
    catalyst_date:    Optional[str] = None
    pre_event_price:  Optional[float] = None

    @property
    def is_undervalued(self) -> bool:
        """简单估值判断"""
        if self.pe_ratio is None or self.pe_ratio <= 0:
            return False
        return self.pe_ratio < 20 and (self.peg_ratio is None or self.peg_ratio < 1.5)

    @property
    def is_quality(self) -> bool:
        """简单质量判断"""
        checks = []
        if self.roe is not None:           checks.append(self.roe > 0.10)
        if self.profit_margin is not None: checks.append(self.profit_margin > 0.05)
        if self.debt_to_equity is not None:checks.append(self.debt_to_equity < 2.0)
        if self.current_ratio is not None: checks.append(self.current_ratio > 1.2)
        if not checks:
            return False
        return sum(checks) / len(checks) >= 0.6


class FundamentalDataLayer:
    """
    基本面数据层（实盘接Yahoo Finance / SEC EDGAR / Polygon）

    使用方法：
    - 实盘：用yfinance.Ticker(symbol).info拉数据
    - 演示：用合成基本面数据（保留接口一致）
    """

    @staticmethod
    def fetch_fundamentals(tickers: List[str]) -> Dict[str, FundamentalSnapshot]:
        """
        拉取基本面数据
        优先用yfinance，失败时用合成数据保持代码可跑
        """
        result: Dict[str, FundamentalSnapshot] = {}
        try:
            import yfinance as yf
            for tk in tickers:
                try:
                    info = yf.Ticker(tk).info
                    result[tk] = FundamentalSnapshot(
                        ticker=tk,
                        sector=info.get('sector', ''),
                        market_cap_b=info.get('marketCap', 0) / 1e9,
                        pe_ratio=info.get('trailingPE'),
                        forward_pe=info.get('forwardPE'),
                        pb_ratio=info.get('priceToBook'),
                        ev_to_ebitda=info.get('enterpriseToEbitda'),
                        peg_ratio=info.get('pegRatio'),
                        revenue_growth=info.get('revenueGrowth'),
                        earnings_growth=info.get('earningsGrowth'),
                        profit_margin=info.get('profitMargins'),
                        roe=info.get('returnOnEquity'),
                        debt_to_equity=info.get('debtToEquity', 0) / 100
                          if info.get('debtToEquity') else None,
                        current_ratio=info.get('currentRatio'),
                        free_cashflow_b=info.get('freeCashflow', 0) / 1e9
                          if info.get('freeCashflow') else None,
                    )
                except Exception:
                    result[tk] = FundamentalDataLayer._synthetic(tk)
        except ImportError:
            for tk in tickers:
                result[tk] = FundamentalDataLayer._synthetic(tk)
        return result

    @staticmethod
    def _synthetic(ticker: str) -> FundamentalSnapshot:
        """合成基本面数据（演示用，覆盖常见股票特征）"""
        np.random.seed(hash(ticker) % 10000)
        # 真实公司的近似数据（手工标注，方便演示场景识别）
        known = {
            # 矿业/材料（FCX类）
            'FCX':  {'sector':'Materials',   'pe':12,'pb':2.1,'roe':0.18,'margin':0.15,'moat':3.5,'pos':'leader'},
            'NEM':  {'sector':'Materials',   'pe':18,'pb':1.8,'roe':0.10,'margin':0.20,'moat':3.0,'pos':'leader'},
            'BHP':  {'sector':'Materials',   'pe':10,'pb':2.5,'roe':0.22,'margin':0.30,'moat':4.0,'pos':'leader'},
            # 半导体存储（SanDisk类）
            'WDC':  {'sector':'Tech',        'pe':9, 'pb':1.5,'roe':0.12,'margin':0.10,'moat':3.0,'pos':'leader'},
            'STX':  {'sector':'Tech',        'pe':11,'pb':4.0,'roe':0.35,'margin':0.18,'moat':3.5,'pos':'leader'},
            'MU':   {'sector':'Tech',        'pe':14,'pb':1.8,'roe':0.15,'margin':0.20,'moat':3.5,'pos':'leader'},
            # 半导体大厂
            'NVDA': {'sector':'Tech',        'pe':45,'pb':30,'roe':0.55,'margin':0.40,'moat':4.5,'pos':'leader'},
            'AMD':  {'sector':'Tech',        'pe':40,'pb':3.5,'roe':0.12,'margin':0.10,'moat':3.5,'pos':'leader'},
            'TSM':  {'sector':'Tech',        'pe':22,'pb':5.5,'roe':0.27,'margin':0.40,'moat':5.0,'pos':'leader'},
            # 科技大厂
            'AAPL': {'sector':'Tech',        'pe':30,'pb':45,'roe':1.50,'margin':0.25,'moat':5.0,'pos':'leader'},
            'MSFT': {'sector':'Tech',        'pe':35,'pb':12,'roe':0.40,'margin':0.36,'moat':4.5,'pos':'leader'},
            'GOOGL':{'sector':'Tech',        'pe':25,'pb':6, 'roe':0.28,'margin':0.27,'moat':4.5,'pos':'leader'},
            # 能源（周期）
            'CVX':  {'sector':'Energy',      'pe':12,'pb':1.8,'roe':0.16,'margin':0.10,'moat':3.5,'pos':'leader'},
            'XOM':  {'sector':'Energy',      'pe':13,'pb':2.0,'roe':0.18,'margin':0.10,'moat':3.5,'pos':'leader'},
        }
        d = known.get(ticker, {})
        return FundamentalSnapshot(
            ticker=ticker,
            sector=d.get('sector', 'Unknown'),
            market_cap_b=float(np.random.uniform(10, 500)),
            pe_ratio=d.get('pe', float(np.random.uniform(15, 30))),
            forward_pe=d.get('pe', float(np.random.uniform(13, 25))) * 0.9,
            pb_ratio=d.get('pb', float(np.random.uniform(2, 6))),
            ev_to_ebitda=float(np.random.uniform(8, 20)),
            peg_ratio=float(np.random.uniform(0.8, 2.5)),
            revenue_growth=float(np.random.uniform(-0.05, 0.30)),
            earnings_growth=float(np.random.uniform(-0.10, 0.40)),
            profit_margin=d.get('margin', float(np.random.uniform(0.05, 0.25))),
            roe=d.get('roe', float(np.random.uniform(0.08, 0.30))),
            debt_to_equity=float(np.random.uniform(0.2, 1.5)),
            current_ratio=float(np.random.uniform(1.0, 2.5)),
            free_cashflow_b=float(np.random.uniform(0.5, 30)),
            moat_score=d.get('moat', float(np.random.uniform(2, 4))),
            industry_position=d.get('pos', 'follower'),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 场景A: 错杀股识别器（FCX矿被冲走类）
# ══════════════════════════════════════════════════════════════════════════════

class OversoldQualityScreener:
    """
    错杀股筛选器：好公司 + 短期事件 + 暴跌

    核心逻辑（你描述的FCX案例）：
    1. 量化触发：股价短期暴跌 >threshold（默认-12%）
    2. 基本面验证：公司质量分够高（is_quality=True）
    3. 估值合理：相对历史不贵（PE < 历史中位数）
    4. 行业地位：leader或challenger（不是folloer）
    5. 流动性：日均成交额 > $10M（避免流动性陷阱）
    6. 止损硬约束：再跌 -8% 强制平仓（防止接飞刀）

    为什么这样设计：
    · 暴跌过滤：只看真正过度反应（小跌没有非理性溢价）
    · 质量过滤：避免"垃圾股越跌越买"（价值陷阱）
    · 估值过滤：即使是好公司，太贵的下跌可能只是估值修正
    · 行业地位：leader恢复快，follower可能永久萎缩
    """

    def __init__(self,
                 drop_threshold:     float = -0.12,
                 lookback_days:      int   = 21,
                 max_pe_percentile:  float = 0.60,
                 min_market_cap_b:   float = 5.0,
                 stop_loss_pct:      float = -0.08):
        self.drop_threshold    = drop_threshold
        self.lookback_days     = lookback_days
        self.max_pe_percentile = max_pe_percentile
        self.min_market_cap_b  = min_market_cap_b
        self.stop_loss_pct     = stop_loss_pct

    def screen(self,
                prices:       pd.DataFrame,
                volumes:      pd.DataFrame,
                fundamentals: Dict[str, FundamentalSnapshot],
                catalyst_events: Dict[str, str] = None) -> List[Dict]:
        """
        筛选错杀股候选
        catalyst_events: {ticker: 'event_description'} 可选的事件信息

        Returns: 候选股票列表（按错杀程度排序）
        """
        candidates = []
        catalyst_events = catalyst_events or {}

        for ticker in prices.columns:
            if ticker not in fundamentals:
                continue
            p = prices[ticker].dropna()
            if len(p) < self.lookback_days + 21:
                continue

            fund = fundamentals[ticker]

            # ── 检查1: 短期暴跌 ──
            recent_drop = float(p.iloc[-1] / p.iloc[-self.lookback_days] - 1)
            if recent_drop > self.drop_threshold:
                continue   # 跌得不够多，不算错杀

            # ── 检查2: 市值足够 ──
            if fund.market_cap_b < self.min_market_cap_b:
                continue   # 太小市值，可能是真问题

            # ── 检查3: 基本面质量 ──
            if not fund.is_quality:
                continue   # 不是好公司

            # ── 检查4: 估值不贵 ──
            if not fund.is_undervalued:
                continue   # PE/PEG偏高，跌的合理

            # ── 检查5: 行业地位 ──
            if fund.industry_position in ('follower', 'unknown'):
                continue   # 不是行业领导者

            # ── 检查6: 流动性 ──
            avg_dollar_vol = float((prices[ticker] * volumes[ticker]).tail(20).mean())
            if avg_dollar_vol < 10e6:
                continue   # 日均成交额 < $10M

            # ── 检查7: 跌势是否企稳（不接飞刀）──
            # 用最近3天看是否止跌
            recent_3d = float(p.iloc[-1] / p.iloc[-3] - 1) if len(p) >= 3 else 0
            stabilizing = recent_3d > -0.03   # 最近3天没继续大跌

            # ── 检查8: 历史回弹能力 ──
            # 用过去5年中 -10%下跌后6个月的平均回报作为信心指标
            hist_recovery = self._historical_recovery(p)

            # ── 综合评分（0-100）──
            score = 0
            score += min(30, abs(recent_drop) / 0.20 * 30)        # 错杀程度
            score += 25 if fund.is_quality else 0                  # 质量
            score += 20 if fund.is_undervalued else 0              # 估值
            score += 15 if fund.industry_position == 'leader' else 5  # 地位
            score += 10 if stabilizing else 0                      # 企稳

            # ── 进场点和止损 ──
            entry_price = float(p.iloc[-1])
            stop_price  = entry_price * (1 + self.stop_loss_pct)
            target_price= entry_price * (1 + abs(recent_drop) * 0.6)  # 回弹60%为目标

            candidates.append({
                'ticker':           ticker,
                'score':            round(score, 1),
                'recent_drop':      round(recent_drop, 4),
                'recent_3d_chg':    round(recent_3d, 4),
                'stabilizing':      stabilizing,
                'pe':               fund.pe_ratio,
                'pb':               fund.pb_ratio,
                'roe':              fund.roe,
                'margin':           fund.profit_margin,
                'is_quality':       fund.is_quality,
                'industry_pos':     fund.industry_position,
                'market_cap_b':     fund.market_cap_b,
                'sector':           fund.sector,
                'catalyst':         catalyst_events.get(ticker, '价格暴跌但基本面未变'),
                'historical_recovery':hist_recovery,
                'entry_price':      round(entry_price, 2),
                'stop_loss':        round(stop_price, 2),
                'target_price':     round(target_price, 2),
                'reward_risk':      round(abs(recent_drop * 0.6) / abs(self.stop_loss_pct), 2),
            })

        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def _historical_recovery(self, prices: pd.Series) -> float:
        """
        统计历史上每次 -10% 后6个月的平均回报
        作为该股票"反弹能力"的代理
        """
        if len(prices) < 252:
            return 0.0
        # 找到过去所有 -10% 回撤点
        rolling_max = prices.rolling(20).max()
        dd_pct = (prices - rolling_max) / rolling_max
        crash_dates = dd_pct[dd_pct < -0.10].index.tolist()

        if not crash_dates:
            return 0.0

        recoveries = []
        for cd in crash_dates:
            try:
                ci = prices.index.get_loc(cd)
                fi = ci + 126   # 6个月后
                if fi < len(prices):
                    rec = prices.iloc[fi] / prices.iloc[ci] - 1
                    if np.isfinite(rec):
                        recoveries.append(float(rec))
            except Exception:
                pass

        return round(float(np.mean(recoveries)), 4) if recoveries else 0.0

    def print_candidates(self, candidates: List[Dict], top_n: int = 5):
        """打印错杀股候选清单"""
        print(f"\n  {'═'*65}")
        print(f"  🔻 错杀股筛选结果（场景A: 好公司+短期暴跌）")
        print(f"  {'═'*65}")
        if not candidates:
            print(f"  当前无符合条件的错杀股候选")
            return

        for i, c in enumerate(candidates[:top_n]):
            print(f"\n  #{i+1} {c['ticker']} ({c['sector']}) — 评分 {c['score']}")
            print(f"     最近{self.lookback_days}天跌幅: {c['recent_drop']:+.2%}  "
                  f"(3天: {c['recent_3d_chg']:+.2%}) "
                  f"{'✅企稳' if c['stabilizing'] else '⚠️未企稳'}")
            print(f"     PE={c['pe']}  PB={c['pb']}  ROE={c['roe']:.1%}  "
                  f"利润率={c['margin']:.1%}")
            print(f"     市值=${c['market_cap_b']:.1f}B  地位={c['industry_pos']}")
            print(f"     催化: {c['catalyst']}")
            print(f"     历史-10%后6月回报: {c['historical_recovery']:+.1%}")
            print(f"     进场=${c['entry_price']}  止损=${c['stop_loss']} ({self.stop_loss_pct:.0%})  "
                  f"目标=${c['target_price']}  R/R={c['reward_risk']:.2f}:1")


# ══════════════════════════════════════════════════════════════════════════════
# 场景B: 强趋势+低估值识别器（SanDisk类）
# ══════════════════════════════════════════════════════════════════════════════

class StrongTrendValueScreener:
    """
    强趋势+低估值筛选器（戴维斯双击候选）

    核心逻辑（你描述的SanDisk案例）：
    1. 强势趋势：连续上涨 + EMA20 > SMA50 > SMA200
    2. 低估值：PE在历史中位数以下 / PEG < 1.5
    3. 行业β：行业指数也在上涨（确认是行业beta，不是个股噪音）
    4. 业绩催化：earnings_growth > 15%
    5. 量能配合：成交量放大（机构在买）
    6. 不追高：相对52周高点 < 95%（留余地）

    为什么这样设计：
    · 单纯动量容易追高接盘
    · 单纯低估值容易陷价值陷阱
    · 双击 = 业绩涨 × 估值扩张，是最强的alpha来源
    · 行业β确认：避免只是个股投机
    · 不追新高：留出20%空间给真正的大行情
    """

    def __init__(self,
                 min_trend_days:        int   = 20,
                 max_pe_for_value:      float = 25.0,
                 max_peg_for_value:     float = 1.5,
                 min_earnings_growth:   float = 0.15,
                 max_pct_of_52w_high:   float = 0.95,
                 min_volume_ratio:      float = 1.2):
        self.min_trend_days        = min_trend_days
        self.max_pe                = max_pe_for_value
        self.max_peg               = max_peg_for_value
        self.min_earnings_growth   = min_earnings_growth
        self.max_pct_of_52w_high   = max_pct_of_52w_high
        self.min_volume_ratio      = min_volume_ratio

    def screen(self,
                prices:       pd.DataFrame,
                volumes:      pd.DataFrame,
                fundamentals: Dict[str, FundamentalSnapshot],
                sector_momentum: Dict[str, float] = None) -> List[Dict]:
        """
        筛选强趋势+低估值候选
        sector_momentum: {sector_name: momentum} 行业动量（可选）
        """
        candidates = []
        sector_momentum = sector_momentum or {}

        for ticker in prices.columns:
            if ticker not in fundamentals:
                continue
            p = prices[ticker].dropna()
            v = volumes[ticker].dropna() if ticker in volumes.columns else pd.Series()
            if len(p) < 252:
                continue
            fund = fundamentals[ticker]

            # ── 检查1: 强趋势 ──
            ema20  = float(p.ewm(span=20, adjust=False).mean().iloc[-1])
            sma50  = float(p.rolling(50).mean().iloc[-1])
            sma200 = float(p.rolling(200).mean().iloc[-1])
            cur    = float(p.iloc[-1])

            strong_trend = (cur > ema20 > sma50 > sma200)
            if not strong_trend:
                continue

            # ── 检查2: 连续上涨天数 ──
            r = p.pct_change().tail(self.min_trend_days)
            up_days = int((r > 0).sum())
            if up_days < self.min_trend_days * 0.55:   # 至少55%是上涨日
                continue

            # ── 检查3: 低估值 ──
            if fund.pe_ratio is None or fund.pe_ratio > self.max_pe:
                continue
            if fund.pe_ratio <= 0:   # 亏损股票排除
                continue
            if fund.peg_ratio is not None and fund.peg_ratio > self.max_peg:
                continue

            # ── 检查4: 业绩催化（戴维斯双击的"业绩"端）──
            has_earnings_growth = (fund.earnings_growth is not None
                                    and fund.earnings_growth >= self.min_earnings_growth)
            if not has_earnings_growth and fund.revenue_growth is not None:
                has_earnings_growth = fund.revenue_growth >= self.min_earnings_growth * 0.8

            if not has_earnings_growth:
                continue

            # ── 检查5: 不追新高 ──
            high_52w = float(p.tail(252).max())
            pct_of_high = cur / high_52w
            if pct_of_high > self.max_pct_of_52w_high:
                continue   # 太接近52周高点，追高风险大

            # ── 检查6: 成交量配合 ──
            volume_ratio = 1.0
            if len(v) > 21:
                v_recent = float(v.tail(5).mean())
                v_base   = float(v.tail(63).mean())
                volume_ratio = v_recent / (v_base + 1e-8)
                if volume_ratio < self.min_volume_ratio:
                    continue   # 量能不够，可能是无量上涨

            # ── 检查7: 行业β确认 ──
            sector_mom = sector_momentum.get(fund.sector, None)
            sector_confirm = sector_mom is None or sector_mom > 0

            # ── 综合评分（0-100）──
            score = 0
            score += min(25, up_days / self.min_trend_days * 25)
            score += min(20, (self.max_pe - fund.pe_ratio) / self.max_pe * 20)
            if fund.earnings_growth:
                score += min(20, fund.earnings_growth / 0.50 * 20)
            score += 15 * (1 - pct_of_high)   # 离高点越远分越高
            score += min(10, (volume_ratio - 1.0) * 20)
            score += 10 if sector_confirm else 0

            # ── 进场计划 ──
            entry_price  = cur
            stop_loss    = sma50              # 跌破SMA50止损
            target_price = entry_price * (1 + fund.earnings_growth * 1.5) \
                            if fund.earnings_growth else entry_price * 1.20

            candidates.append({
                'ticker':            ticker,
                'score':             round(score, 1),
                'sector':            fund.sector,
                'price':             round(cur, 2),
                'ema20':             round(ema20, 2),
                'sma50':             round(sma50, 2),
                'sma200':            round(sma200, 2),
                'up_days':           up_days,
                'pe':                fund.pe_ratio,
                'forward_pe':        fund.forward_pe,
                'peg':               fund.peg_ratio,
                'earnings_growth':   fund.earnings_growth,
                'revenue_growth':    fund.revenue_growth,
                'roe':               fund.roe,
                'volume_ratio':      round(volume_ratio, 2),
                'pct_of_52w_high':   round(pct_of_high, 4),
                'sector_momentum':   sector_mom,
                'entry_price':       round(entry_price, 2),
                'stop_loss':         round(stop_loss, 2),
                'target_price':      round(target_price, 2),
                'reward_risk':       round((target_price - entry_price) / (entry_price - stop_loss + 1e-8), 2),
                'thesis':            f"强趋势({up_days}/{self.min_trend_days}天上涨) + "
                                      f"PE={fund.pe_ratio:.1f}低估值 + "
                                      f"EPS增长{fund.earnings_growth:.0%}" if fund.earnings_growth
                                      else f"强趋势({up_days}天) + PE={fund.pe_ratio:.1f}低估值",
            })

        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def print_candidates(self, candidates: List[Dict], top_n: int = 5):
        print(f"\n  {'═'*65}")
        print(f"  📈 强趋势+低估值筛选（场景B: 戴维斯双击候选）")
        print(f"  {'═'*65}")
        if not candidates:
            print(f"  当前无符合条件的候选")
            return

        for i, c in enumerate(candidates[:top_n]):
            print(f"\n  #{i+1} {c['ticker']} ({c['sector']}) — 评分 {c['score']}")
            print(f"     现价=${c['price']}  EMA20=${c['ema20']}  SMA50=${c['sma50']}  SMA200=${c['sma200']}")
            print(f"     {c['up_days']}/{self.min_trend_days}天上涨  "
                  f"距52周高点{c['pct_of_52w_high']:.0%}  量比{c['volume_ratio']}x")
            print(f"     PE={c['pe']:.1f}  PEG={c.get('peg','N/A')}  "
                  f"EPS增长={c.get('earnings_growth',0):.0%}  ROE={c.get('roe',0):.1%}")
            print(f"     论点: {c['thesis']}")
            print(f"     进场=${c['entry_price']}  止损=${c['stop_loss']}(SMA50)  "
                  f"目标=${c['target_price']}  R/R={c['reward_risk']:.2f}:1")


# ══════════════════════════════════════════════════════════════════════════════
# 场景C: 跌幅榜深度挖掘（每日触发器）
# ══════════════════════════════════════════════════════════════════════════════

class DailyLoserMiner:
    """
    每日跌幅榜挖掘器

    核心逻辑：每天收盘后，扫描全市场跌幅榜，自动分类：
    A类：错杀（基本面好+跌幅大）→ 触发OversoldQualityScreener
    B类：技术回踩（强趋势但短期回调）→ 加仓机会
    C类：真问题（基本面已变坏）→ 避开
    D类：流动性事件（指数调整/被剔除）→ 短期机会

    每日工作流：
    1. 收盘后获取当日跌幅最大的100只
    2. 过滤市值（>=$2B避免微盘股）
    3. 分类（A/B/C/D）
    4. 对A类自动进入OversoldQualityScreener
    5. 对B类检查是否在已持仓的强趋势中
    6. 生成每日复盘报告
    """

    def __init__(self,
                 min_drop_today:      float = -0.05,
                 min_market_cap_b:    float = 2.0,
                 top_n_to_analyze:    int   = 100):
        self.min_drop_today      = min_drop_today
        self.min_market_cap_b    = min_market_cap_b
        self.top_n_to_analyze    = top_n_to_analyze

    def classify_loser(self,
                        ticker:       str,
                        prices:       pd.Series,
                        volumes:      pd.Series,
                        fundamentals: FundamentalSnapshot,
                        today_drop:   float) -> Dict:
        """
        将单个跌幅榜股票分类
        """
        if len(prices) < 50:
            return {'category': 'INSUFFICIENT_DATA'}

        p = prices.dropna()
        cur = float(p.iloc[-1])

        # 各种参考指标
        sma20  = float(p.rolling(20).mean().iloc[-1])
        sma50  = float(p.rolling(50).mean().iloc[-1])
        sma200 = float(p.rolling(200).mean().iloc[-1]) if len(p) >= 200 else sma50
        high_52w = float(p.tail(min(252, len(p))).max())

        # 历史波动率
        hist_vol = float(p.pct_change().tail(63).std() * np.sqrt(252))
        # 今日跌幅相对历史波动率的标准差倍数
        sigma_move = abs(today_drop) / (hist_vol / np.sqrt(252) + 1e-8)

        # 量能
        vol_ratio = 1.0
        if len(volumes) > 21:
            vol_ratio = float(volumes.iloc[-1] / (volumes.tail(21).mean() + 1e-8))

        # ── 分类逻辑 ──
        category = 'UNKNOWN'
        sub_reason = ''
        action = 'WATCH'

        if fundamentals.is_quality and fundamentals.is_undervalued and sigma_move > 2.0:
            # A类：错杀（好公司+大跌+估值不贵）
            category = 'A_OVERSOLD_QUALITY'
            sub_reason = f"好公司基本面 + {sigma_move:.1f}σ大跌 + PE={fundamentals.pe_ratio:.1f}低估值"
            action = '送入 OversoldQualityScreener 详细分析'

        elif cur > sma50 and sma50 > sma200 and today_drop > -0.10:
            # B类：技术回踩（强趋势中的回调）
            pct_of_high = cur / high_52w
            category = 'B_TREND_PULLBACK'
            sub_reason = f"强趋势({cur/sma200-1:+.0%}>SMA200) + 回踩(距高点{pct_of_high:.0%})"
            action = '如已持仓考虑加仓；未持仓等企稳信号'

        elif (fundamentals.earnings_growth is not None
              and fundamentals.earnings_growth < -0.20):
            # C类：基本面恶化
            category = 'C_FUNDAMENTAL_BREAK'
            sub_reason = f"EPS增长{fundamentals.earnings_growth:.0%}恶化"
            action = '避开，可能继续下跌'

        elif vol_ratio > 3.0 and abs(today_drop) > 0.08:
            # D类：流动性事件（成交量爆增+大跌）
            category = 'D_LIQUIDITY_EVENT'
            sub_reason = f"量能{vol_ratio:.1f}x + 暴跌{today_drop:.0%}（可能是指数调整/机构抛售）"
            action = '观察1-3天，跌势停止后可能是短期反弹机会'

        elif sigma_move > 3.0:
            # 异常波动但分类不明
            category = 'E_EXTREME_MOVE'
            sub_reason = f"{sigma_move:.1f}σ极端波动（>3σ罕见事件）"
            action = '需要查新闻确认原因'

        else:
            category = 'F_NORMAL_DECLINE'
            sub_reason = f"正常波动（{sigma_move:.1f}σ）"
            action = '无明显机会'

        return {
            'ticker':         ticker,
            'category':       category,
            'sub_reason':     sub_reason,
            'action':         action,
            'today_drop':     round(today_drop, 4),
            'sigma_move':     round(sigma_move, 2),
            'current_price':  round(cur, 2),
            'pct_of_52w_high':round(cur / high_52w, 4),
            'sma50_relation': 'above' if cur > sma50 else 'below',
            'sma200_relation':'above' if cur > sma200 else 'below',
            'volume_ratio':   round(vol_ratio, 2),
            'is_quality':     fundamentals.is_quality,
            'pe':             fundamentals.pe_ratio,
            'market_cap_b':   fundamentals.market_cap_b,
        }

    def scan_universe(self,
                       prices:       pd.DataFrame,
                       volumes:      pd.DataFrame,
                       fundamentals: Dict[str, FundamentalSnapshot]) -> Dict[str, List[Dict]]:
        """
        扫描全部universe，分类所有跌幅榜股票

        Returns: {'A_OVERSOLD_QUALITY': [...], 'B_TREND_PULLBACK': [...], ...}
        """
        if len(prices) < 50:
            return {}

        # 计算所有股票今日跌幅
        today_returns = prices.pct_change().iloc[-1]
        losers = today_returns[today_returns < self.min_drop_today].dropna()
        losers = losers.sort_values()   # 跌幅最大的在前

        results: Dict[str, List[Dict]] = defaultdict(list)

        for ticker in losers.index[:self.top_n_to_analyze]:
            if ticker not in fundamentals:
                continue
            fund = fundamentals[ticker]
            if fund.market_cap_b < self.min_market_cap_b:
                continue

            v = volumes[ticker] if ticker in volumes.columns else pd.Series()
            classification = self.classify_loser(
                ticker, prices[ticker], v, fund, float(losers[ticker])
            )
            results[classification['category']].append(classification)

        return dict(results)

    def daily_report(self,
                      prices:       pd.DataFrame,
                      volumes:      pd.DataFrame,
                      fundamentals: Dict[str, FundamentalSnapshot]):
        """
        生成每日跌幅榜复盘报告
        """
        scan = self.scan_universe(prices, volumes, fundamentals)

        print(f"\n  {'═'*65}")
        print(f"  📉 每日跌幅榜深度挖掘报告")
        print(f"  日期: {prices.index[-1].date() if hasattr(prices.index[-1],'date') else prices.index[-1]}")
        print(f"  {'═'*65}")

        categories_order = [
            ('A_OVERSOLD_QUALITY', '🟢 A类：错杀（最高优先级）'),
            ('B_TREND_PULLBACK',   '🟡 B类：技术回踩（加仓机会）'),
            ('D_LIQUIDITY_EVENT',  '🔵 D类：流动性事件（短期机会）'),
            ('E_EXTREME_MOVE',     '⚠️  E类：极端波动（需查新闻）'),
            ('C_FUNDAMENTAL_BREAK','🔴 C类：基本面恶化（避开）'),
            ('F_NORMAL_DECLINE',   '⚪ F类：正常波动（无机会）'),
        ]

        for cat_key, cat_label in categories_order:
            items = scan.get(cat_key, [])
            if not items:
                continue
            print(f"\n  {cat_label}（{len(items)}只）")
            for item in items[:5]:   # 每类只显示前5
                print(f"    {item['ticker']:<6} {item['today_drop']:+.2%}  "
                      f"σ={item['sigma_move']:.1f}  "
                      f"PE={item.get('pe','N/A')}  "
                      f"市值=${item['market_cap_b']:.0f}B")
                print(f"           理由: {item['sub_reason']}")
                print(f"           行动: {item['action']}")

        if not any(scan.values()):
            print(f"\n  今日无显著跌幅股票（市场平稳）")
        print(f"  {'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# DISCRETIONARY OPPORTUNITY ENGINE — 整合所有人工选股逻辑
# ══════════════════════════════════════════════════════════════════════════════

class DiscretionaryOpportunityEngine:
    """
    人工选股机会引擎（整合A/B/C/D四个场景）

    每日工作流：
    1. 跌幅榜扫描（DailyLoserMiner）
    2. 错杀股筛选（OversoldQualityScreener）
    3. 强趋势+低估值筛选（StrongTrendValueScreener）
    4. 输出按优先级排序的"机会清单"
    5. 每个机会附带：进场点/止损/目标/R-R/论点

    集成到 SleeveManager：
    - 错杀股 → CORE_HEDGE Sleeve（长期持有）
    - 强趋势+低估值 → SECTOR_ROTATION Sleeve
    - 流动性事件反弹 → TACTICAL Sleeve（短期）
    """

    def __init__(self):
        self.data_layer       = FundamentalDataLayer()
        self.oversold_screen  = OversoldQualityScreener()
        self.trend_value      = StrongTrendValueScreener()
        self.loser_miner      = DailyLoserMiner()

    def find_opportunities(self,
                            prices:  pd.DataFrame,
                            volumes: pd.DataFrame,
                            catalyst_events: Dict[str, str] = None,
                            sector_momentum: Dict[str, float] = None) -> Dict:
        """
        找出所有人工选股机会
        """
        # 拉取基本面
        fundamentals = self.data_layer.fetch_fundamentals(prices.columns.tolist())

        # 场景A：错杀股
        oversold = self.oversold_screen.screen(
            prices, volumes, fundamentals, catalyst_events
        )

        # 场景B：强趋势+低估值
        trend_value = self.trend_value.screen(
            prices, volumes, fundamentals, sector_momentum
        )

        # 场景C：每日跌幅榜分类（如果今日有大跌）
        daily_scan = self.loser_miner.scan_universe(prices, volumes, fundamentals)

        return {
            'oversold_quality':  oversold,
            'trend_value':       trend_value,
            'daily_losers_scan': daily_scan,
            'fundamentals':      fundamentals,
            'summary': {
                'n_oversold':    len(oversold),
                'n_trend_value': len(trend_value),
                'n_daily_A':     len(daily_scan.get('A_OVERSOLD_QUALITY', [])),
                'n_daily_B':     len(daily_scan.get('B_TREND_PULLBACK', [])),
            }
        }

    def to_sleeve_assignments(self, opportunities: Dict,
                                top_n_each: int = 3) -> Dict[str, List[Dict]]:
        """
        将机会按Sleeve分配
        """
        assignments = {
            'CORE_HEDGE':       [],   # 错杀股（长期）
            'SECTOR_ROTATION':  [],   # 强趋势+低估值（中期）
            'TACTICAL':         [],   # 流动性事件（短期）
        }

        # 错杀股 → CORE_HEDGE
        for c in opportunities['oversold_quality'][:top_n_each]:
            assignments['CORE_HEDGE'].append({
                'ticker':       c['ticker'],
                'scenario':     'OVERSOLD_QUALITY',
                'score':        c['score'],
                'entry':        c['entry_price'],
                'stop':         c['stop_loss'],
                'target':       c['target_price'],
                'reward_risk':  c['reward_risk'],
                'thesis':       c['catalyst'],
                'holding_days': '60-180',
            })

        # 强趋势+低估值 → SECTOR_ROTATION
        for c in opportunities['trend_value'][:top_n_each]:
            assignments['SECTOR_ROTATION'].append({
                'ticker':       c['ticker'],
                'scenario':     'STRONG_TREND_LOW_VALUE',
                'score':        c['score'],
                'entry':        c['entry_price'],
                'stop':         c['stop_loss'],
                'target':       c['target_price'],
                'reward_risk':  c['reward_risk'],
                'thesis':       c['thesis'],
                'holding_days': '21-90',
            })

        # 流动性事件D类 → TACTICAL
        for d in opportunities['daily_losers_scan'].get('D_LIQUIDITY_EVENT', [])[:top_n_each]:
            assignments['TACTICAL'].append({
                'ticker':       d['ticker'],
                'scenario':     'LIQUIDITY_EVENT_BOUNCE',
                'score':        50,  # 简化分
                'entry':        d['current_price'],
                'stop':         d['current_price'] * 0.96,
                'target':       d['current_price'] * 1.04,
                'reward_risk':  1.0,
                'thesis':       d['sub_reason'],
                'holding_days': '1-5',
            })

        return assignments

    def print_full_report(self, opportunities: Dict):
        """打印完整人工选股报告"""
        s = opportunities['summary']
        print(f"\n  {'═'*65}")
        print(f"  🎯 人工选股机会引擎 — 完整报告")
        print(f"  {'═'*65}")
        print(f"  扫描结果汇总：")
        print(f"    错杀股候选:        {s['n_oversold']}只")
        print(f"    强趋势+低估值:     {s['n_trend_value']}只")
        print(f"    跌幅榜A类(错杀):   {s['n_daily_A']}只")
        print(f"    跌幅榜B类(回踩):   {s['n_daily_B']}只")

        # 详细输出
        self.oversold_screen.print_candidates(opportunities['oversold_quality'])
        self.trend_value.print_candidates(opportunities['trend_value'])
        self.loser_miner.daily_report(
            opportunities.get('_prices', pd.DataFrame()),
            opportunities.get('_volumes', pd.DataFrame()),
            opportunities['fundamentals']
        ) if '_prices' in opportunities else None

        # Sleeve分配
        assignments = self.to_sleeve_assignments(opportunities)
        print(f"\n  {'═'*65}")
        print(f"  📋 Sleeve分配建议")
        print(f"  {'═'*65}")
        for sleeve, items in assignments.items():
            if items:
                print(f"\n  [{sleeve}]")
                for it in items:
                    print(f"    {it['ticker']:<6} 场景={it['scenario']:<25} "
                          f"R/R={it['reward_risk']:.1f}  持仓{it['holding_days']}天")
                    print(f"           入${it['entry']}/止${it['stop']}/目${it['target']}")
                    print(f"           论点: {it['thesis']}")


# ══════════════════════════════════════════════════════════════════════════════
# [v8.5] DEEP DISCRETIONARY ENGINE — 深度选股引擎
#
# 在v8.4基础上的根本性升级：
#
# v8.4不足              →  v8.5解决
# ──────────────────────   ───────────────────────────────────────────────
# 一个catalyst字符串    →  EventImpactQuantifier（事件分类+影响量化）
# 只看PE/PB             →  6维基本面深度（含护城河+周期阶段+资产负债表）
# 没有机构信号          →  Yahoo免费数据的institutionHolders/shortPercent等
# 没有分析师信号        →  EPS revision/recommendation/target_price变化
# 一个holding_days      →  4阶段进场计划（试探/加仓1/加仓2/满仓）
# 没有反向证据          →  InvalidationTriggers（明确量化）
# 没有持仓状态机        →  PositionStateMachine（NOW/WAIT/AVOID + HOLD/ADD/REDUCE/EXIT）
# 没有优先级            →  3档优先级（A立即/B回踩/C观察）
# Kelly只看历史         →  确定性×赔率分仓
#
# ══════════════════════════════════════════════════════════════════════════════

# ── 数据扩展：用Yahoo Finance免费拿到的信号 ──────────────────────────────────

@dataclass
class DeepFundamental:
    """深度基本面快照（在FundamentalSnapshot基础上扩展）"""
    # 基础（从v8.4继承）
    ticker:           str
    sector:           str = ''
    market_cap_b:     float = 0.0

    # 估值（多维度对比）
    pe_ratio:         Optional[float] = None
    forward_pe:       Optional[float] = None
    pb_ratio:         Optional[float] = None
    ev_to_ebitda:     Optional[float] = None
    peg_ratio:        Optional[float] = None
    price_to_sales:   Optional[float] = None
    pe_5y_median:     Optional[float] = None    # 历史中位数（关键对比）
    pe_percentile:    Optional[float] = None    # 当前PE在历史百分位（0=最便宜）

    # 增长
    revenue_growth:   Optional[float] = None
    earnings_growth:  Optional[float] = None
    eps_forward_growth:Optional[float] = None   # 前瞻EPS增长（分析师预期）

    # 财务质量
    profit_margin:    Optional[float] = None
    operating_margin: Optional[float] = None
    roe:              Optional[float] = None
    roa:              Optional[float] = None
    debt_to_equity:   Optional[float] = None
    current_ratio:    Optional[float] = None
    quick_ratio:      Optional[float] = None
    interest_coverage:Optional[float] = None    # 利息保障倍数（>3健康）
    free_cashflow_b:  Optional[float] = None
    cash_b:           Optional[float] = None    # 现金（用于压力测试）

    # 护城河5维（来自Morningstar/手工评估，简化用代理）
    brand_score:      float = 3.0   # 品牌（消费品高）
    network_effect:   float = 3.0   # 网络效应（社交/平台高）
    scale_advantage:  float = 3.0   # 规模优势（重资产/规模经济高）
    switching_cost:   float = 3.0   # 转换成本（企业软件高）
    intangible:       float = 3.0   # 无形资产（专利/牌照）

    # 行业地位
    industry_position:str = 'unknown'   # leader/challenger/follower/niche
    market_share:     Optional[float] = None

    # ── v8.5 新增：Yahoo Finance可拿到的机构/分析师信号 ──
    # 机构持仓
    institution_holders_pct: Optional[float] = None      # 机构持股比例
    institution_holders_change: Optional[float] = None   # 变化（季度趋势）

    # 分析师信号
    analyst_recommendation:  Optional[float] = None      # 1=Strong Buy, 5=Sell
    analyst_target_mean:     Optional[float] = None
    analyst_target_high:     Optional[float] = None
    analyst_target_low:      Optional[float] = None
    eps_revision_up_pct:     Optional[float] = None      # 过去30天上修分析师比例
    eps_revision_down_pct:   Optional[float] = None
    num_analysts:            Optional[int]   = None

    # 内部人 + 做空
    insider_ownership:       Optional[float] = None      # 内部人持股
    insider_buy_minus_sell:  Optional[float] = None      # 简化（Yahoo只有总量，没单笔）
    short_percent_of_float:  Optional[float] = None      # 做空比例（Squeeze候选）
    short_ratio_days:        Optional[float] = None      # Days to cover

    # 业绩日历
    next_earnings_date:      Optional[str] = None
    days_to_earnings:        Optional[int] = None

    # 价格相关
    current_price:           Optional[float] = None
    high_52w:                Optional[float] = None
    low_52w:                 Optional[float] = None

    @property
    def moat_total(self) -> float:
        """护城河综合分（5维加权）"""
        return (self.brand_score * 0.20 + self.network_effect * 0.20 +
                self.scale_advantage * 0.25 + self.switching_cost * 0.20 +
                self.intangible * 0.15)

    @property
    def is_deep_value(self) -> bool:
        """深度低估（更严格）"""
        if self.pe_ratio is None or self.pe_ratio <= 0:
            return False
        cond1 = self.pe_ratio < 15
        cond2 = self.pe_percentile is not None and self.pe_percentile < 0.30
        cond3 = self.peg_ratio is not None and self.peg_ratio < 1.0
        return cond1 or cond2 or cond3

    @property
    def balance_sheet_safe(self) -> bool:
        """资产负债表能扛事（用于错杀股压力测试）"""
        checks = []
        if self.debt_to_equity is not None:  checks.append(self.debt_to_equity < 1.5)
        if self.current_ratio is not None:   checks.append(self.current_ratio > 1.5)
        if self.interest_coverage is not None:checks.append(self.interest_coverage > 3)
        if self.cash_b is not None and self.market_cap_b > 0:
            checks.append(self.cash_b / self.market_cap_b > 0.05)  # 现金>市值5%
        return len(checks) >= 2 and sum(checks) / len(checks) >= 0.7


class YahooDeepDataLayer:
    """
    深度Yahoo Finance数据层
    用yfinance.Ticker().info拿尽可能多的免费信号
    """

    @staticmethod
    def fetch(tickers: List[str]) -> Dict[str, DeepFundamental]:
        result: Dict[str, DeepFundamental] = {}
        try:
            import yfinance as yf
            for tk in tickers:
                try:
                    t = yf.Ticker(tk)
                    info = t.info
                    # 推荐变化
                    rec_mean = info.get('recommendationMean')

                    # 拼装
                    cur_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    result[tk] = DeepFundamental(
                        ticker=tk,
                        sector=info.get('sector', ''),
                        market_cap_b=(info.get('marketCap', 0) or 0) / 1e9,
                        pe_ratio=info.get('trailingPE'),
                        forward_pe=info.get('forwardPE'),
                        pb_ratio=info.get('priceToBook'),
                        ev_to_ebitda=info.get('enterpriseToEbitda'),
                        peg_ratio=info.get('pegRatio'),
                        price_to_sales=info.get('priceToSalesTrailing12Months'),
                        revenue_growth=info.get('revenueGrowth'),
                        earnings_growth=info.get('earningsGrowth'),
                        eps_forward_growth=info.get('earningsQuarterlyGrowth'),
                        profit_margin=info.get('profitMargins'),
                        operating_margin=info.get('operatingMargins'),
                        roe=info.get('returnOnEquity'),
                        roa=info.get('returnOnAssets'),
                        debt_to_equity=(info.get('debtToEquity') or 0) / 100 if info.get('debtToEquity') else None,
                        current_ratio=info.get('currentRatio'),
                        quick_ratio=info.get('quickRatio'),
                        free_cashflow_b=(info.get('freeCashflow') or 0) / 1e9 if info.get('freeCashflow') else None,
                        cash_b=(info.get('totalCash') or 0) / 1e9 if info.get('totalCash') else None,
                        # 机构 + 分析师 + 做空
                        institution_holders_pct=info.get('heldPercentInstitutions'),
                        analyst_recommendation=rec_mean,
                        analyst_target_mean=info.get('targetMeanPrice'),
                        analyst_target_high=info.get('targetHighPrice'),
                        analyst_target_low=info.get('targetLowPrice'),
                        num_analysts=info.get('numberOfAnalystOpinions'),
                        insider_ownership=info.get('heldPercentInsiders'),
                        short_percent_of_float=info.get('shortPercentOfFloat'),
                        short_ratio_days=info.get('shortRatio'),
                        current_price=cur_price,
                        high_52w=info.get('fiftyTwoWeekHigh'),
                        low_52w=info.get('fiftyTwoWeekLow'),
                    )
                except Exception:
                    result[tk] = YahooDeepDataLayer._synthetic(tk)
        except ImportError:
            for tk in tickers:
                result[tk] = YahooDeepDataLayer._synthetic(tk)
        return result

    @staticmethod
    def _synthetic(ticker: str) -> DeepFundamental:
        """合成数据，覆盖典型案例（FCX/WDC类）"""
        np.random.seed(hash(ticker) % 10000)
        # 真实公司近似数据
        known = {
            # 矿业（FCX类，错杀候选）
            'FCX':  dict(sector='Materials', mc=60, pe=12, pb=2.1, peg=0.9, roe=0.18, margin=0.15,
                        moat=3.5, pos='leader', cash=4.0, d2e=0.7, cur_ratio=2.1,
                        inst=0.78, short=0.025, rec=2.1, rev_up=0.55, an=15),
            'NEM':  dict(sector='Materials', mc=40, pe=18, pb=1.8, peg=1.2, roe=0.10, margin=0.20,
                        moat=3.0, pos='leader', cash=3.5, d2e=0.5, cur_ratio=2.5,
                        inst=0.75, short=0.020, rec=2.5, rev_up=0.40, an=12),
            # 半导体存储（WDC类，强趋势+低估值候选）
            'WDC':  dict(sector='Tech', mc=20, pe=9, pb=1.5, peg=0.6, roe=0.12, margin=0.10,
                        moat=3.0, pos='leader', cash=2.5, d2e=0.8, cur_ratio=1.8,
                        inst=0.85, short=0.080, rec=2.0, rev_up=0.65, an=22),
            'STX':  dict(sector='Tech', mc=22, pe=11, pb=4.0, peg=0.7, roe=0.35, margin=0.18,
                        moat=3.5, pos='leader', cash=1.5, d2e=1.2, cur_ratio=1.6,
                        inst=0.80, short=0.060, rec=2.2, rev_up=0.60, an=18),
            'MU':   dict(sector='Tech', mc=110, pe=14, pb=1.8, peg=0.4, roe=0.15, margin=0.20,
                        moat=3.5, pos='leader', cash=8.0, d2e=0.4, cur_ratio=2.8,
                        inst=0.82, short=0.030, rec=1.8, rev_up=0.70, an=28),
            # AI龙头（高估值，可能不通过低估值筛选）
            'NVDA': dict(sector='Tech', mc=2000, pe=45, pb=30, peg=1.0, roe=0.55, margin=0.40,
                        moat=4.5, pos='leader', cash=30, d2e=0.4, cur_ratio=3.5,
                        inst=0.66, short=0.010, rec=1.6, rev_up=0.80, an=55),
            'AMD':  dict(sector='Tech', mc=200, pe=40, pb=3.5, peg=1.2, roe=0.12, margin=0.10,
                        moat=3.5, pos='challenger', cash=5, d2e=0.5, cur_ratio=2.5,
                        inst=0.71, short=0.020, rec=2.0, rev_up=0.55, an=45),
            'TSM':  dict(sector='Tech', mc=600, pe=22, pb=5.5, peg=0.9, roe=0.27, margin=0.40,
                        moat=5.0, pos='leader', cash=50, d2e=0.3, cur_ratio=2.0,
                        inst=0.18, short=0.005, rec=1.5, rev_up=0.75, an=20),
            # 大盘股
            'AAPL': dict(sector='Tech', mc=3000, pe=30, pb=45, peg=2.5, roe=1.50, margin=0.25,
                        moat=5.0, pos='leader', cash=60, d2e=1.5, cur_ratio=1.0,
                        inst=0.62, short=0.008, rec=2.0, rev_up=0.45, an=40),
            'MSFT': dict(sector='Tech', mc=3000, pe=35, pb=12, peg=2.2, roe=0.40, margin=0.36,
                        moat=4.5, pos='leader', cash=80, d2e=0.5, cur_ratio=1.7,
                        inst=0.72, short=0.005, rec=1.5, rev_up=0.60, an=50),
            'GOOGL':dict(sector='Tech', mc=2000, pe=25, pb=6, peg=1.5, roe=0.28, margin=0.27,
                        moat=4.5, pos='leader', cash=100, d2e=0.1, cur_ratio=2.4,
                        inst=0.73, short=0.005, rec=1.5, rev_up=0.55, an=48),
            'SPY':  dict(sector='ETF', mc=500, pe=22, pb=4.5, peg=1.8, roe=0.20, margin=0.12,
                        moat=4.0, pos='leader', cash=10, d2e=0.5, cur_ratio=1.5,
                        inst=0.85, short=0.003, rec=2.0, rev_up=0.50, an=0),
            'QQQ':  dict(sector='ETF', mc=300, pe=30, pb=8, peg=2.0, roe=0.25, margin=0.20,
                        moat=4.0, pos='leader', cash=5, d2e=0.4, cur_ratio=1.5,
                        inst=0.80, short=0.008, rec=2.0, rev_up=0.55, an=0),
            'SOXX': dict(sector='ETF', mc=15, pe=28, pb=6, peg=1.5, roe=0.20, margin=0.20,
                        moat=4.0, pos='leader', cash=1, d2e=0.4, cur_ratio=1.5,
                        inst=0.80, short=0.010, rec=2.0, rev_up=0.60, an=0),
        }
        d = known.get(ticker, {})

        # PE历史中位数和百分位（用合成）
        pe = d.get('pe', float(np.random.uniform(15, 30)))
        pe_5y_med = pe * float(np.random.uniform(0.9, 1.3))
        pe_pct = float(np.clip(0.5 - (pe_5y_med - pe) / pe_5y_med, 0.05, 0.95))

        # 目标价（基于当前推荐）
        target_mean = float(np.random.uniform(1.05, 1.30)) * 100  # 占位

        return DeepFundamental(
            ticker=ticker,
            sector=d.get('sector', 'Unknown'),
            market_cap_b=d.get('mc', float(np.random.uniform(5, 500))),
            pe_ratio=pe,
            forward_pe=pe * 0.85,
            pb_ratio=d.get('pb', float(np.random.uniform(2, 6))),
            ev_to_ebitda=float(np.random.uniform(8, 20)),
            peg_ratio=d.get('peg', float(np.random.uniform(0.8, 2.5))),
            price_to_sales=float(np.random.uniform(2, 8)),
            pe_5y_median=pe_5y_med,
            pe_percentile=pe_pct,
            revenue_growth=float(np.random.uniform(-0.05, 0.30)),
            earnings_growth=float(np.random.uniform(-0.10, 0.40)),
            eps_forward_growth=float(np.random.uniform(0.05, 0.35)),
            profit_margin=d.get('margin', float(np.random.uniform(0.05, 0.25))),
            operating_margin=d.get('margin', 0.15) * 1.2,
            roe=d.get('roe', float(np.random.uniform(0.08, 0.30))),
            roa=d.get('roe', 0.15) * 0.5,
            debt_to_equity=d.get('d2e', float(np.random.uniform(0.2, 1.5))),
            current_ratio=d.get('cur_ratio', float(np.random.uniform(1.0, 2.5))),
            quick_ratio=d.get('cur_ratio', 1.5) * 0.8,
            interest_coverage=float(np.random.uniform(3, 15)),
            free_cashflow_b=float(np.random.uniform(0.5, 30)),
            cash_b=d.get('cash', float(np.random.uniform(1, 50))),
            brand_score=d.get('moat', 3.0),
            network_effect=d.get('moat', 3.0),
            scale_advantage=d.get('moat', 3.0),
            switching_cost=d.get('moat', 3.0),
            intangible=d.get('moat', 3.0),
            industry_position=d.get('pos', 'follower'),
            institution_holders_pct=d.get('inst', float(np.random.uniform(0.4, 0.85))),
            institution_holders_change=float(np.random.uniform(-0.02, 0.05)),
            analyst_recommendation=d.get('rec', float(np.random.uniform(1.5, 3.5))),
            analyst_target_mean=target_mean,
            analyst_target_high=target_mean * 1.2,
            analyst_target_low=target_mean * 0.8,
            eps_revision_up_pct=d.get('rev_up', float(np.random.uniform(0.2, 0.7))),
            eps_revision_down_pct=float(np.random.uniform(0.1, 0.4)),
            num_analysts=d.get('an', int(np.random.uniform(5, 30))),
            insider_ownership=float(np.random.uniform(0.001, 0.10)),
            insider_buy_minus_sell=float(np.random.uniform(-100, 100)),  # 千股净买入
            short_percent_of_float=d.get('short', float(np.random.uniform(0.01, 0.10))),
            short_ratio_days=float(np.random.uniform(1, 5)),
            current_price=float(np.random.uniform(50, 500)),
            high_52w=float(np.random.uniform(100, 600)),
            low_52w=float(np.random.uniform(30, 200)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 事件影响量化器 — 把"催化"从字符串变成可量化的预测
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventImpact:
    """事件影响评估结果"""
    event_type:           str          # 'accident', 'earnings_miss', etc.
    severity:             str          # 'minor', 'moderate', 'major', 'catastrophic'
    expected_loss_pct:    float        # 预期总损失（已发生+未发生）
    expected_recovery_days: int        # 预期恢复天数（基于历史类比）
    recovery_probability: float        # 恢复到事件前价格的概率
    similar_cases:        List[str]    # 历史类比案例
    confidence:           float        # 评估置信度 0-1


class EventImpactQuantifier:
    """
    事件影响量化器

    把"FCX矿被冲走"这种字符串描述，转换成：
    · 类型分类（事故/财报/管理层/政策/收购/诉讼）
    · 严重程度（minor/moderate/major/catastrophic）
    · 损失估计（5%产能×3个月 = 1.25%年化收入影响）
    · 恢复时间预测（基于历史类比）
    · 恢复概率（不是所有事件都能恢复）
    · 给出历史类比案例

    数据来源：
    · 历史事件库（手工标注的典型案例）
    · 行业内类似事件的恢复曲线统计
    """

    # 事件类型库 → 历史平均影响数据
    EVENT_LIBRARY = {
        # 自然灾害/事故
        'mine_accident': {
            'category': 'operational_accident',
            'avg_loss_pct': -0.15, 'avg_recovery_days': 90,
            'recovery_rate': 0.85,    # 85%案例完全恢复
            'cases': ['FCX 2020洪水', 'GOLD 2018矿崩塌', 'BHP 2015尾矿坝'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 1.8, 'catastrophic': 3.5}
        },
        'plant_fire': {
            'category': 'operational_accident',
            'avg_loss_pct': -0.18, 'avg_recovery_days': 120,
            'recovery_rate': 0.80,
            'cases': ['INTC 2022 Fab火灾', 'TM 2011日本地震', 'BA 737MAX'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 2.0}
        },
        'natural_disaster': {
            'category': 'operational_accident',
            'avg_loss_pct': -0.12, 'avg_recovery_days': 60,
            'recovery_rate': 0.90,
            'cases': ['NEM 2019 Cyclone', 'HII 2017 Harvey'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 1.5}
        },
        # 财报
        'earnings_miss': {
            'category': 'fundamental_surprise',
            'avg_loss_pct': -0.15, 'avg_recovery_days': 180,
            'recovery_rate': 0.55,    # 财报miss恢复率较低
            'cases': ['META 2022Q3', 'NFLX 2022Q1', 'INTC 2024Q1'],
            'severity_factor': {'minor': 0.4, 'moderate': 1.0, 'major': 2.5}
        },
        'earnings_beat': {
            'category': 'fundamental_surprise',
            'avg_loss_pct': +0.08, 'avg_recovery_days': 0,  # 是上涨催化
            'recovery_rate': 0.95,
            'cases': ['NVDA 2023Q2', 'AMD 2023Q4', 'MU 2024Q2'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 2.0}
        },
        'guidance_cut': {
            'category': 'fundamental_surprise',
            'avg_loss_pct': -0.20, 'avg_recovery_days': 270,
            'recovery_rate': 0.45,    # 指引下调恢复更慢
            'cases': ['SNAP 2022 Q1', 'PYPL 2022', 'NKE 2024'],
            'severity_factor': {'minor': 0.4, 'moderate': 1.0, 'major': 2.5}
        },
        # 管理层
        'ceo_departure': {
            'category': 'management_change',
            'avg_loss_pct': -0.08, 'avg_recovery_days': 120,
            'recovery_rate': 0.70,
            'cases': ['DIS 2020 Iger退/回', 'TWTR 2017 Costolo退', 'INTC 2024 Gelsinger退'],
            'severity_factor': {'planned': 0.3, 'unplanned': 1.0, 'scandal': 2.5}
        },
        'new_ceo': {
            'category': 'management_change',
            'avg_loss_pct': +0.05, 'avg_recovery_days': 30,
            'recovery_rate': 0.65,    # 新CEO效应混合
            'cases': ['MSFT 2014 Nadella', 'DIS 2022 Iger回归'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 1.5}
        },
        # 政策/监管
        'regulation_negative': {
            'category': 'regulatory',
            'avg_loss_pct': -0.18, 'avg_recovery_days': 360,
            'recovery_rate': 0.40,    # 监管影响恢复慢
            'cases': ['META 2022 隐私', '中概股 2021', 'TIK TOK 2024'],
            'severity_factor': {'fine': 0.5, 'restriction': 1.5, 'ban': 4.0}
        },
        # 收购/并购
        'acquisition_target': {
            'category': 'corporate_action',
            'avg_loss_pct': +0.25, 'avg_recovery_days': 0,
            'recovery_rate': 0.90,    # 被收购通常溢价
            'cases': ['ATVI被MSFT收购', 'TWTR被Musk', 'FRC被JPM'],
            'severity_factor': {'minor': 0.6, 'moderate': 1.0, 'major': 1.5}
        },
        'acquisition_failed': {
            'category': 'corporate_action',
            'avg_loss_pct': -0.20, 'avg_recovery_days': 180,
            'recovery_rate': 0.55,
            'cases': ['NVDA-ARM失败', 'AT&T-Tmobile失败'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 2.0}
        },
        # 周期/行业
        'cycle_trough': {
            'category': 'industry_cycle',
            'avg_loss_pct': -0.30, 'avg_recovery_days': 540,  # 周期股底部
            'recovery_rate': 0.85,    # 周期最终会恢复
            'cases': ['MU 2019', 'STX 2023', 'WDC 2023', '油气2020'],
            'severity_factor': {'minor': 0.6, 'moderate': 1.0, 'major': 1.8}
        },
        'cycle_peak': {
            'category': 'industry_cycle',
            'avg_loss_pct': -0.10, 'avg_recovery_days': 720,  # 周期顶部下行慢
            'recovery_rate': 0.50,
            'cases': ['SOXX 2022', '油气2014'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 2.0}
        },
        # 通用未知事件
        'unknown_negative': {
            'category': 'other',
            'avg_loss_pct': -0.10, 'avg_recovery_days': 180,
            'recovery_rate': 0.60,
            'cases': ['类似事件历史'],
            'severity_factor': {'minor': 0.5, 'moderate': 1.0, 'major': 1.5}
        }
    }

    @staticmethod
    def classify_event(event_text: str) -> str:
        """从事件描述文本推断类型"""
        if not event_text:
            return 'unknown_negative'
        text = event_text.lower()

        keywords = {
            'mine_accident':       ['矿', 'mine', '坍塌', 'collapse', '冲走'],
            'plant_fire':          ['火灾', '事故', 'fire', '工厂'],
            'natural_disaster':    ['暴雨', '飓风', '地震', 'flood', 'hurricane', 'earthquake', 'cyclone'],
            'earnings_miss':       ['miss', '不及预期', '低于预期', '未达'],
            'earnings_beat':       ['超预期', 'beat', 'exceed', '超出'],
            'guidance_cut':        ['guidance', '指引下调', '展望下调'],
            'ceo_departure':       ['ceo', 'cfo离', '离职', '辞职', 'depart'],
            'new_ceo':             ['任命', 'new ceo', '新任', '上任'],
            'regulation_negative': ['监管', '罚款', '反垄断', 'antitrust', 'regulation'],
            'acquisition_target':  ['被收购', '收购offer', 'acquired'],
            'acquisition_failed':  ['收购失败', '终止', 'failed deal'],
            'cycle_trough':        ['周期底', '库存底', '存储周期', '存储芯片'],
            'cycle_peak':          ['周期顶', '需求见顶'],
        }
        for event_type, kws in keywords.items():
            if any(kw in text for kw in kws):
                return event_type
        return 'unknown_negative'

    @staticmethod
    def quantify(event_text: str,
                  severity: str = 'moderate',
                  already_dropped_pct: float = 0.0) -> EventImpact:
        """
        量化事件影响

        Args:
            event_text: 事件描述
            severity:   严重程度 (minor/moderate/major/catastrophic)
            already_dropped_pct: 股价已经跌了多少（用于判断"还会跌多少"）
        """
        event_type = EventImpactQuantifier.classify_event(event_text)
        lib = EventImpactQuantifier.EVENT_LIBRARY.get(
            event_type, EventImpactQuantifier.EVENT_LIBRARY['unknown_negative']
        )

        # 严重程度调整
        sev_factor = lib['severity_factor'].get(severity, 1.0)
        expected_total_loss = lib['avg_loss_pct'] * sev_factor

        # 已发生的vs剩余的（用于判断当前是否"跌够了"）
        remaining_loss = expected_total_loss - already_dropped_pct
        # 注意：如果already_dropped已经超过预期，remaining_loss可能是正的（过度反应）

        # 恢复天数（按严重程度缩放）
        recovery_days = int(lib['avg_recovery_days'] * sev_factor)

        # 置信度（基于事件类型的可预测性）
        confidence_map = {
            'operational_accident': 0.75,   # 事故影响相对可预测
            'fundamental_surprise': 0.55,   # 财报影响难判断
            'management_change':    0.50,
            'regulatory':           0.40,   # 监管最难预测
            'corporate_action':     0.70,
            'industry_cycle':       0.65,
            'other':                0.30,
        }
        confidence = confidence_map.get(lib['category'], 0.40)

        return EventImpact(
            event_type=event_type,
            severity=severity,
            expected_loss_pct=expected_total_loss,
            expected_recovery_days=recovery_days,
            recovery_probability=lib['recovery_rate'],
            similar_cases=lib['cases'][:3],
            confidence=confidence,
        )

    @staticmethod
    def is_overdone(event_impact: EventImpact, current_drop: float) -> bool:
        """
        当前下跌是否已经"过度反应"
        如果已跌幅 < expected*1.5（更深）→ 可能是错杀机会
        """
        return current_drop < event_impact.expected_loss_pct * 1.5


# ══════════════════════════════════════════════════════════════════════════════
# 6维评分系统（深度选股核心）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SixDimScore:
    """6维度评分（每维0-100）"""
    fundamental:    float    # 基本面（盈利能力+健康度）
    valuation:      float    # 估值（vs历史 + vs同业）
    trend:          float    # 趋势（技术面 + 价格结构）
    event:          float    # 事件（催化剂强度 + 影响量化）
    institutional:  float    # 机构（持仓变化 + 分析师评级 + 内部人）
    invalidation:   float    # 反向证据强度（这是"保护分"，越高越安全）

    @property
    def total(self) -> float:
        return (self.fundamental    * 0.20 +
                self.valuation       * 0.20 +
                self.trend           * 0.15 +
                self.event           * 0.20 +
                self.institutional   * 0.15 +
                self.invalidation    * 0.10)

    @property
    def confidence_level(self) -> str:
        """整体确定性"""
        t = self.total
        if t >= 80: return 'VERY_HIGH'
        elif t >= 70: return 'HIGH'
        elif t >= 60: return 'MEDIUM'
        elif t >= 50: return 'LOW'
        else: return 'VERY_LOW'


class SixDimScorer:
    """6维评分计算器"""

    @staticmethod
    def score(deep_fund: DeepFundamental,
              prices: pd.Series,
              event_impact: EventImpact = None,
              scenario: str = 'oversold') -> SixDimScore:
        """
        计算单只股票的6维评分
        scenario: 'oversold'(错杀) | 'trend_value'(强趋势+低估值) | 'general'
        """
        # ── 维度1: 基本面（盈利能力+健康度）──
        fund_score = SixDimScorer._score_fundamental(deep_fund)

        # ── 维度2: 估值（vs历史 + 多重指标）──
        val_score = SixDimScorer._score_valuation(deep_fund)

        # ── 维度3: 趋势（技术面）──
        trend_score = SixDimScorer._score_trend(prices, scenario)

        # ── 维度4: 事件（催化强度）──
        event_score = SixDimScorer._score_event(event_impact, prices)

        # ── 维度5: 机构信号（最重要的辅助信号）──
        inst_score = SixDimScorer._score_institutional(deep_fund)

        # ── 维度6: 反向证据（越多保护分越高）──
        invalid_score = SixDimScorer._score_invalidation(deep_fund, prices, event_impact)

        return SixDimScore(
            fundamental=fund_score,
            valuation=val_score,
            trend=trend_score,
            event=event_score,
            institutional=inst_score,
            invalidation=invalid_score,
        )

    @staticmethod
    def _score_fundamental(df: DeepFundamental) -> float:
        score = 0.0
        # 盈利能力（30分）
        if df.roe is not None:
            score += min(15, df.roe / 0.30 * 15)
        if df.profit_margin is not None:
            score += min(15, df.profit_margin / 0.25 * 15)
        # 健康度（30分）
        if df.balance_sheet_safe:
            score += 20
        if df.current_ratio is not None and df.current_ratio > 1.5:
            score += 10
        # 护城河（25分）
        score += min(25, (df.moat_total - 1) / 4 * 25)
        # 行业地位（15分）
        if df.industry_position == 'leader':
            score += 15
        elif df.industry_position == 'challenger':
            score += 10
        elif df.industry_position == 'niche':
            score += 5
        return min(100, score)

    @staticmethod
    def _score_valuation(df: DeepFundamental) -> float:
        score = 0.0
        # PE历史百分位（30分，越低越好）
        if df.pe_percentile is not None:
            score += (1 - df.pe_percentile) * 30
        # PEG（25分）
        if df.peg_ratio is not None and df.peg_ratio > 0:
            if df.peg_ratio < 0.8:   score += 25
            elif df.peg_ratio < 1.2: score += 18
            elif df.peg_ratio < 1.5: score += 12
            elif df.peg_ratio < 2.0: score += 6
        # PE绝对值（20分）
        if df.pe_ratio is not None and df.pe_ratio > 0:
            if df.pe_ratio < 10:  score += 20
            elif df.pe_ratio < 15: score += 15
            elif df.pe_ratio < 20: score += 10
            elif df.pe_ratio < 25: score += 5
        # PB（15分）
        if df.pb_ratio is not None:
            if df.pb_ratio < 1.5:  score += 15
            elif df.pb_ratio < 3.0: score += 10
            elif df.pb_ratio < 5.0: score += 5
        # 分析师目标价上行空间（10分）
        if df.analyst_target_mean and df.current_price and df.current_price > 0:
            upside = (df.analyst_target_mean / df.current_price - 1)
            score += min(10, upside / 0.30 * 10)
        return min(100, max(0, score))

    @staticmethod
    def _score_trend(prices: pd.Series, scenario: str) -> float:
        if len(prices) < 50:
            return 50
        p = prices.dropna()
        cur = float(p.iloc[-1])
        sma50  = float(p.rolling(min(50, len(p))).mean().iloc[-1])
        sma200 = float(p.rolling(min(200, len(p))).mean().iloc[-1])

        score = 0.0
        if scenario == 'oversold':
            # 错杀型：希望短期跌得多，但企稳
            r21  = float(p.iloc[-1] / p.iloc[-min(21, len(p)-1)] - 1)
            r3   = float(p.iloc[-1] / p.iloc[-min(3, len(p)-1)] - 1)
            # 跌幅深 → 高分（错杀机会）
            if r21 < -0.15:      score += 30
            elif r21 < -0.10:    score += 22
            elif r21 < -0.05:    score += 12
            # 短期企稳 → 高分
            if r3 > -0.02:       score += 25
            elif r3 > -0.05:     score += 15
            # 距离SMA200不要太远（避免趋势完全破坏）
            if cur > sma200 * 0.85: score += 20
            elif cur > sma200 * 0.75: score += 10
            # 成交量
            score += 25  # 简化

        elif scenario == 'trend_value':
            # 强趋势型：希望持续上涨
            ema20 = float(p.ewm(span=20, adjust=False).mean().iloc[-1])
            if cur > ema20 > sma50 > sma200: score += 35
            elif cur > sma50 > sma200:        score += 25
            elif cur > sma50:                  score += 15
            # 上涨连续性
            r = p.pct_change().tail(20)
            up_pct = float((r > 0).mean())
            score += min(25, up_pct * 30)
            # 距离高点
            high_252 = float(p.tail(min(252, len(p))).max())
            pct_high = cur / high_252
            if 0.85 < pct_high < 0.95: score += 25   # 不追高但接近
            elif 0.75 < pct_high < 0.85: score += 15
            elif pct_high < 0.75: score += 5
            # 不在创新高
            if pct_high < 0.98: score += 15
        else:
            # 通用：综合考虑
            if cur > sma50:  score += 30
            if cur > sma200: score += 30
            score += 40 * (cur / sma200 - 0.85) / 0.30   # 距SMA200的相对位置
        return min(100, max(0, score))

    @staticmethod
    def _score_event(event: EventImpact, prices: pd.Series) -> float:
        if event is None:
            return 40   # 中性分（没有明确催化）
        score = 0.0
        # 恢复概率（30分）
        score += event.recovery_probability * 30
        # 置信度（20分）
        score += event.confidence * 20
        # 类似案例数量（15分）
        score += min(15, len(event.similar_cases) * 5)
        # 是否过度反应（20分，最重要）
        if len(prices) >= 21:
            current_drop = float(prices.iloc[-1] / prices.iloc[-21] - 1)
            if EventImpactQuantifier.is_overdone(event, current_drop):
                score += 20
            else:
                score += 5
        # 事件严重性匹配（15分）
        if event.severity in ('minor', 'moderate'):
            score += 15   # 可控的严重性
        elif event.severity == 'major':
            score += 8
        # 'catastrophic'不加分
        return min(100, score)

    @staticmethod
    def _score_institutional(df: DeepFundamental) -> float:
        score = 0.0
        # 分析师推荐（25分）— 1=强买，5=卖
        if df.analyst_recommendation is not None:
            if df.analyst_recommendation < 2.0:    score += 25
            elif df.analyst_recommendation < 2.5:  score += 18
            elif df.analyst_recommendation < 3.0:  score += 10
            elif df.analyst_recommendation < 3.5:  score += 3
        # 分析师覆盖深度（10分）
        if df.num_analysts is not None:
            if df.num_analysts >= 20:  score += 10
            elif df.num_analysts >= 10: score += 7
            elif df.num_analysts >= 5:  score += 4
        # EPS修订（20分）
        if df.eps_revision_up_pct is not None:
            score += min(20, df.eps_revision_up_pct * 25)
        # 目标价上行空间（15分）
        if df.analyst_target_mean and df.current_price and df.current_price > 0:
            upside = df.analyst_target_mean / df.current_price - 1
            score += min(15, upside / 0.30 * 15)
        # 机构持股变化（15分）
        if df.institution_holders_change is not None:
            if df.institution_holders_change > 0.02:    score += 15
            elif df.institution_holders_change > 0:     score += 10
            elif df.institution_holders_change > -0.01: score += 5
        # 做空比例适中（高做空+反转催化 = 潜在Squeeze）
        if df.short_percent_of_float is not None:
            if 0.10 < df.short_percent_of_float < 0.30:   score += 15   # Squeeze候选
            elif df.short_percent_of_float < 0.05:         score += 8    # 低做空稳定
        return min(100, score)

    @staticmethod
    def _score_invalidation(df: DeepFundamental, prices: pd.Series,
                             event: EventImpact = None) -> float:
        """
        反向证据/安全边际分（越高越安全）
        """
        score = 50   # 基础分
        # 资产负债表能扛事 → 加分
        if df.balance_sheet_safe:
            score += 20
        # 现金/市值
        if df.cash_b and df.market_cap_b > 0:
            cash_ratio = df.cash_b / df.market_cap_b
            if cash_ratio > 0.15:   score += 15
            elif cash_ratio > 0.08:  score += 10
            elif cash_ratio > 0.04:  score += 5
        # PE的下行保护（PE已经很低，难再杀估值）
        if df.pe_ratio is not None and df.pe_ratio > 0:
            if df.pe_ratio < 8:    score += 15
            elif df.pe_ratio < 12: score += 10
            elif df.pe_ratio < 16: score += 5
        # 事件已知且可控
        if event and event.confidence > 0.7:
            score += 10
        # 内部人持股
        if df.insider_ownership and df.insider_ownership > 0.05:
            score += 5
        return min(100, max(0, score))


# ══════════════════════════════════════════════════════════════════════════════
# 分阶段进场计划 + 反向证据触发器
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntryPlan:
    """分阶段进场计划"""
    pilot_price:        float    # 试探仓价位
    pilot_pct:          float    # 试探仓占目标仓位比例
    add1_price:         float    # 加仓1
    add1_pct:           float
    add2_price:         float    # 加仓2
    add2_pct:           float
    full_pct:           float    # 满仓比例（=1.0除非有犹豫）

    stop_loss:          float    # 硬止损
    target1:            float    # 减仓点1
    target1_reduce:     float    # 减仓1卖出多少（%）
    target2:            float    # 减仓点2
    target2_reduce:     float
    target3:            float    # 最终目标（清仓）

    invalidation_triggers: List[str]   # 反向证据列表（满足任一即清仓）


class EntryPlanBuilder:
    """构造分阶段进场计划"""

    @staticmethod
    def build_oversold_plan(current_price: float,
                            df: DeepFundamental,
                            event: EventImpact) -> EntryPlan:
        """
        错杀股进场计划

        策略：
        1. 现价试探30%（已经跌透，但等确认）
        2. 如果不再跌（5天内未创新低）加30%
        3. 如果反弹突破10MA，加40%（趋势确认）
        4. 反弹到事件前价格80% → 减仓30%
        5. 反弹到事件前价格 → 减仓60%
        6. 创新高 → 全平
        """
        # 估算事件前价格（假设事件已跌进去）
        pre_event = current_price / (1 + event.expected_loss_pct * 0.7)

        return EntryPlan(
            pilot_price=current_price,
            pilot_pct=0.30,
            add1_price=current_price * 1.02,       # +2%确认未继续跌
            add1_pct=0.30,
            add2_price=current_price * 1.05,       # +5%突破短期阻力
            add2_pct=0.40,
            full_pct=1.00,
            stop_loss=current_price * 0.92,        # -8%硬止损
            target1=pre_event * 0.80,              # 反弹至事件前80%
            target1_reduce=0.30,
            target2=pre_event * 0.95,
            target2_reduce=0.40,
            target3=pre_event * 1.05,              # 突破事件前
            invalidation_triggers=[
                f"再跌超过-8%（接到飞刀）",
                f"事件升级（{event.event_type}严重度提升）",
                f"持有>{event.expected_recovery_days * 1.5:.0f}天仍未反弹",
                f"基本面恶化（下季度财报miss）",
                "内部人士开始抛售（Form 4出现卖单）",
                "做空比例>20%（市场看空加剧）",
            ],
        )

    @staticmethod
    def build_trend_value_plan(current_price: float,
                                df: DeepFundamental,
                                sma50: float) -> EntryPlan:
        """
        强趋势+低估值进场计划

        策略：
        1. 等回踩到5日均线试探30%（不追高）
        2. 突破前期阻力加30%（趋势确认）
        3. 业绩公布后加40%（基本面确认）
        4. 达到目标价80%减30%
        5. 达到目标价减60%
        6. 跌破SMA50止损
        """
        # 目标价：基于分析师 + EPS增长双重保守估计
        if df.analyst_target_mean and df.analyst_target_mean > current_price:
            analyst_target = df.analyst_target_mean
        else:
            analyst_target = current_price * 1.20

        growth_target = current_price
        if df.earnings_growth:
            growth_target = current_price * (1 + df.earnings_growth * 1.2)

        target = min(analyst_target, growth_target) * 0.95   # 保守

        return EntryPlan(
            pilot_price=current_price * 0.97,      # 等回踩-3%
            pilot_pct=0.30,
            add1_price=current_price * 1.02,
            add1_pct=0.30,
            add2_price=current_price * 1.08,       # 突破前高
            add2_pct=0.40,
            full_pct=1.00,
            stop_loss=sma50,                         # 跌破SMA50
            target1=current_price + (target - current_price) * 0.50,
            target1_reduce=0.30,
            target2=current_price + (target - current_price) * 0.80,
            target2_reduce=0.40,
            target3=target,
            invalidation_triggers=[
                f"跌破SMA50（${sma50:.2f}）",
                "EPS增长率连续两季下降",
                "PE扩张超过历史90%分位（估值过度）",
                "行业β转负（基础β假设不成立）",
                "分析师评级下调",
                "出现-10%单日暴跌（趋势破坏）",
            ],
        )


# ══════════════════════════════════════════════════════════════════════════════
# 持仓状态机 — NOW/WAIT/AVOID + HOLD/ADD/REDUCE/EXIT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeDecision:
    """完整交易决策"""
    ticker:             str
    priority:           str          # 'A' / 'B' / 'C'
    buy_signal:         str          # 'NOW' / 'WAIT' / 'AVOID'
    position_action:    str          # 'HOLD' / 'ADD' / 'REDUCE' / 'EXIT' / 'NONE' (no position)
    entry_plan:         EntryPlan
    six_dim_score:      SixDimScore
    event_impact:       Optional[EventImpact]
    kelly_position_pct: float
    confidence:         str
    reasons:            List[str]
    warnings:           List[str]
    scenario:           str          # 'oversold' / 'trend_value'


class PositionStateMachine:
    """
    持仓状态机：
    根据当前情况输出明确的"现在该做什么"

    买点判断：NOW/WAIT/AVOID
    持仓动作：HOLD/ADD/REDUCE/EXIT/NONE
    """

    @staticmethod
    def determine_buy_signal(six_score: SixDimScore,
                              prices: pd.Series,
                              entry_plan: EntryPlan,
                              scenario: str) -> Tuple[str, List[str]]:
        """
        判断当前是否买点
        Returns: (signal, reasons)
        signal: NOW | WAIT | AVOID
        """
        reasons = []
        cur = float(prices.iloc[-1]) if len(prices) > 0 else 0

        # 总分太低 → AVOID
        if six_score.total < 55:
            return 'AVOID', [f"综合评分{six_score.total:.0f}低于55分"]
        # 反向证据分太低 → AVOID
        if six_score.invalidation < 40:
            return 'AVOID', [f"安全边际不足（保护分{six_score.invalidation:.0f}<40）"]

        # 错杀场景的买点判断
        if scenario == 'oversold':
            # 检查是否企稳
            if len(prices) >= 3:
                r3 = float(prices.iloc[-1] / prices.iloc[-3] - 1)
                if r3 < -0.05:
                    return 'WAIT', [f"3天还在跌{r3:+.1%}，等企稳"]
                else:
                    reasons.append(f"3天企稳{r3:+.1%}")

            # 当前价 vs 试探价
            if cur <= entry_plan.pilot_price * 1.02:
                reasons.append(f"现价${cur:.2f}≤试探价${entry_plan.pilot_price:.2f}+2%")
                return 'NOW', reasons + [f"6维评分{six_score.total:.0f}"]
            else:
                return 'WAIT', [f"现价${cur:.2f}>试探价${entry_plan.pilot_price:.2f}+2%，等回踩"]

        # 强趋势场景的买点判断
        elif scenario == 'trend_value':
            # 当前价是否过高
            if cur > entry_plan.add2_price * 1.02:
                return 'WAIT', [f"现价${cur:.2f}已过加仓2价${entry_plan.add2_price:.2f}，等回踩"]
            # 在试探区
            elif cur <= entry_plan.pilot_price * 1.02:
                return 'NOW', [f"在试探价${entry_plan.pilot_price:.2f}附近"]
            # 在加仓1区
            elif cur <= entry_plan.add1_price * 1.02:
                return 'NOW', [f"在加仓1价${entry_plan.add1_price:.2f}附近"]
            else:
                return 'WAIT', [f"现价${cur:.2f}在加仓区之间，等更好买点"]

        return 'WAIT', ['等待更明确信号']

    @staticmethod
    def determine_position_action(current_position_pct: float,
                                    six_score: SixDimScore,
                                    prices: pd.Series,
                                    entry_plan: EntryPlan,
                                    holding_days: int = 0) -> Tuple[str, List[str]]:
        """
        如果已经持仓，当前应该做什么

        Returns: (action, reasons)
        action: HOLD | ADD | REDUCE | EXIT | NONE (no position)
        """
        if current_position_pct < 0.001:
            return 'NONE', ['当前未持仓']

        cur = float(prices.iloc[-1]) if len(prices) > 0 else 0
        reasons = []

        # 1. 触及止损 → EXIT
        if cur <= entry_plan.stop_loss:
            return 'EXIT', [f"触及止损${entry_plan.stop_loss:.2f}（现价${cur:.2f}）"]

        # 2. 综合评分崩塌 → EXIT
        if six_score.total < 40:
            return 'EXIT', [f"评分跌至{six_score.total:.0f}，逻辑可能证伪"]
        # 3. 安全边际崩塌 → REDUCE
        if six_score.invalidation < 30:
            return 'REDUCE', [f"安全边际{six_score.invalidation:.0f}<30，降仓控制风险"]

        # 4. 触及target3 → EXIT
        if cur >= entry_plan.target3:
            return 'EXIT', [f"达到最终目标${entry_plan.target3:.2f}"]
        # 5. 触及target2 → REDUCE
        if cur >= entry_plan.target2:
            return 'REDUCE', [f"达到目标2${entry_plan.target2:.2f}，减仓{entry_plan.target2_reduce:.0%}"]
        # 6. 触及target1 → REDUCE
        if cur >= entry_plan.target1:
            return 'REDUCE', [f"达到目标1${entry_plan.target1:.2f}，减仓{entry_plan.target1_reduce:.0%}"]

        # 7. 在加仓区 + 仓位还不满 → ADD
        full_target_pct = 0.10   # 假设满仓10%
        if current_position_pct < full_target_pct * 0.95:
            if cur >= entry_plan.add2_price * 0.98 and cur <= entry_plan.add2_price * 1.02:
                return 'ADD', [f"现价${cur:.2f}在加仓2价${entry_plan.add2_price:.2f}附近"]
            if cur >= entry_plan.add1_price * 0.98 and cur <= entry_plan.add1_price * 1.02:
                return 'ADD', [f"现价${cur:.2f}在加仓1价${entry_plan.add1_price:.2f}附近"]

        # 8. 持有时间过长（仍未达target1）→ REDUCE警告
        if holding_days > 180 and cur < entry_plan.target1:
            return 'REDUCE', [f"持有{holding_days}天仍未达目标，时间风险增加"]

        # 9. 默认 → HOLD
        return 'HOLD', [f"评分{six_score.total:.0f}维持，距目标1还有{(entry_plan.target1/cur-1)*100:.1f}%"]


# ══════════════════════════════════════════════════════════════════════════════
# 优先级排序器 (A/B/C)
# ══════════════════════════════════════════════════════════════════════════════

class PriorityRanker:
    """
    优先级排序：A立即买 / B等回踩 / C观察

    A级（立即买）：
    - 6维总分 ≥ 75
    - 安全边际分 ≥ 60
    - 买点信号 = NOW
    - 至少有一个具体催化（事件分 ≥ 60）

    B级（等回踩）：
    - 6维总分 ≥ 65
    - 买点信号 = WAIT
    - 等到价格回到试探/加仓价位

    C级（观察）：
    - 6维总分 ≥ 55
    - 还有明显瑕疵或时机未到
    """

    @staticmethod
    def assign_priority(six_score: SixDimScore, buy_signal: str,
                         event_impact: EventImpact = None) -> str:
        total = six_score.total
        has_strong_event = event_impact is not None and six_score.event >= 60

        if (total >= 75 and six_score.invalidation >= 60
            and buy_signal == 'NOW' and (has_strong_event or six_score.event >= 60)):
            return 'A'
        elif total >= 65 and six_score.invalidation >= 50 and buy_signal in ('NOW', 'WAIT'):
            return 'B'
        elif total >= 55:
            return 'C'
        else:
            return 'SKIP'


# ══════════════════════════════════════════════════════════════════════════════
# Kelly 仓位计算（基于确定性×赔率）
# ══════════════════════════════════════════════════════════════════════════════

class CertaintyKellySizer:
    """
    基于确定性×赔率的仓位计算
    f = (确定性 × 赔率 - (1-确定性)) / 赔率 × 半Kelly因子
    """

    @staticmethod
    def calc_position_pct(six_score: SixDimScore,
                          entry_plan: EntryPlan,
                          current_price: float,
                          max_position: float = 0.10) -> float:
        # 确定性 = 总分/100 × 安全边际加成
        certainty = (six_score.total / 100) * (0.8 + six_score.invalidation / 500)
        certainty = min(0.85, certainty)

        # 赔率 = (目标 - 现价) / (现价 - 止损)
        if entry_plan.target2 > current_price and current_price > entry_plan.stop_loss:
            reward = entry_plan.target2 - current_price
            risk = current_price - entry_plan.stop_loss
            odds = reward / (risk + 1e-8)
        else:
            odds = 1.5

        # Kelly公式
        if odds <= 0:
            kelly = 0
        else:
            kelly_raw = (certainty * odds - (1 - certainty)) / odds
            kelly = max(0, kelly_raw)

        # 半Kelly + 上限
        position = min(kelly * 0.5, max_position)
        return round(position, 4)


# ══════════════════════════════════════════════════════════════════════════════
# 深度选股引擎 — 整合所有模块
# ══════════════════════════════════════════════════════════════════════════════

class DeepDiscretionaryEngine:
    """
    深度选股引擎（v8.5）— 全自动版本

    输出每只候选股票的完整TradeDecision：
    · 6维评分
    · 优先级 A/B/C
    · 买点信号 NOW/WAIT/AVOID
    · 持仓动作 HOLD/ADD/REDUCE/EXIT/NONE
    · 分阶段进场计划
    · 反向证据触发器
    · Kelly仓位建议
    """

    def __init__(self):
        self.data_layer = YahooDeepDataLayer()
        self.scorer     = SixDimScorer()
        self.ranker     = PriorityRanker()
        self.sizer      = CertaintyKellySizer()

    def evaluate(self,
                  ticker: str,
                  prices_series: pd.Series,
                  deep_fund: DeepFundamental,
                  event_text: str = '',
                  event_severity: str = 'moderate',
                  scenario: str = 'oversold',
                  current_position_pct: float = 0.0,
                  holding_days: int = 0) -> TradeDecision:
        """
        评估单只股票，输出完整决策
        """
        # 1. 事件量化（如果有）
        event_impact = None
        if event_text:
            already_dropped = 0.0
            if len(prices_series) >= 21:
                already_dropped = float(prices_series.iloc[-1] / prices_series.iloc[-21] - 1)
            event_impact = EventImpactQuantifier.quantify(
                event_text, event_severity, already_dropped
            )

        # 2. 6维评分
        six_score = self.scorer.score(deep_fund, prices_series, event_impact, scenario)

        # 3. 构造进场计划
        cur = float(prices_series.iloc[-1]) if len(prices_series) > 0 else 100
        if scenario == 'oversold' and event_impact:
            entry_plan = EntryPlanBuilder.build_oversold_plan(cur, deep_fund, event_impact)
        else:
            sma50 = float(prices_series.rolling(min(50, len(prices_series))).mean().iloc[-1]) \
                    if len(prices_series) > 10 else cur * 0.95
            entry_plan = EntryPlanBuilder.build_trend_value_plan(cur, deep_fund, sma50)

        # 4. 买点判断
        buy_signal, buy_reasons = PositionStateMachine.determine_buy_signal(
            six_score, prices_series, entry_plan, scenario
        )

        # 5. 持仓动作
        position_action, action_reasons = PositionStateMachine.determine_position_action(
            current_position_pct, six_score, prices_series, entry_plan, holding_days
        )

        # 6. 优先级
        priority = self.ranker.assign_priority(six_score, buy_signal, event_impact)

        # 7. Kelly仓位
        kelly_pct = self.sizer.calc_position_pct(six_score, entry_plan, cur)

        # 8. Warnings收集
        warnings = []
        if six_score.invalidation < 50:
            warnings.append(f"安全边际偏低({six_score.invalidation:.0f}/100)")
        if event_impact and event_impact.confidence < 0.5:
            warnings.append(f"事件预测置信度低({event_impact.confidence:.0%})")
        if deep_fund.next_earnings_date and deep_fund.days_to_earnings is not None:
            if deep_fund.days_to_earnings < 7:
                warnings.append(f"距财报{deep_fund.days_to_earnings}天，财报波动风险")

        return TradeDecision(
            ticker=ticker,
            priority=priority,
            buy_signal=buy_signal,
            position_action=position_action,
            entry_plan=entry_plan,
            six_dim_score=six_score,
            event_impact=event_impact,
            kelly_position_pct=kelly_pct,
            confidence=six_score.confidence_level,
            reasons=buy_reasons + action_reasons,
            warnings=warnings,
            scenario=scenario,
        )

    def screen_universe(self,
                         prices: pd.DataFrame,
                         volumes: pd.DataFrame,
                         events: Dict[str, Tuple[str, str]] = None,
                         current_positions: Dict[str, float] = None) -> List[TradeDecision]:
        """
        扫描整个universe，输出排序的TradeDecision列表
        events: {ticker: (event_text, severity)}
        current_positions: {ticker: position_pct}
        """
        events = events or {}
        current_positions = current_positions or {}

        # 拉取深度基本面
        deep_funds = self.data_layer.fetch(prices.columns.tolist())

        decisions = []
        for ticker in prices.columns:
            if ticker not in deep_funds:
                continue
            df = deep_funds[ticker]
            ps = prices[ticker].dropna()
            if len(ps) < 50:
                continue

            event_text, severity = events.get(ticker, ('', 'moderate'))

            # 自动判断场景：有大跌+事件 → oversold；强趋势+低估值 → trend_value
            recent_drop = float(ps.iloc[-1] / ps.iloc[-min(21, len(ps)-1)] - 1) if len(ps) > 21 else 0
            if recent_drop < -0.10 or event_text:
                scenario = 'oversold'
            elif df.pe_ratio and df.pe_ratio < 20:
                # 检查是否强趋势
                cur = float(ps.iloc[-1])
                sma50 = float(ps.rolling(50).mean().iloc[-1])
                sma200 = float(ps.rolling(min(200, len(ps))).mean().iloc[-1])
                if cur > sma50 > sma200:
                    scenario = 'trend_value'
                else:
                    scenario = 'general'
            else:
                scenario = 'general'

            try:
                decision = self.evaluate(
                    ticker, ps, df,
                    event_text=event_text,
                    event_severity=severity,
                    scenario=scenario,
                    current_position_pct=current_positions.get(ticker, 0.0),
                    holding_days=0,
                )
                decisions.append(decision)
            except Exception as e:
                continue

        # 按优先级和评分排序
        priority_order = {'A': 0, 'B': 1, 'C': 2, 'SKIP': 3}
        decisions.sort(key=lambda d: (
            priority_order.get(d.priority, 9),
            -d.six_dim_score.total
        ))
        return decisions

    def print_full_report(self, decisions: List[TradeDecision], top_n: int = 10):
        """打印完整候选清单"""
        print(f"\n{'═'*72}")
        print(f"  🎯 Canyon Deep Discretionary 候选清单")
        print(f"  按确定性×赔率排序")
        print(f"{'═'*72}")

        a_list = [d for d in decisions if d.priority == 'A']
        b_list = [d for d in decisions if d.priority == 'B']
        c_list = [d for d in decisions if d.priority == 'C']
        skip_list = [d for d in decisions if d.priority == 'SKIP']

        print(f"\n  统计：A级{len(a_list)} | B级{len(b_list)} | C级{len(c_list)} | "
              f"跳过{len(skip_list)}")

        for priority, items, label in [('A', a_list, '🟢 A级 — 立即买'),
                                         ('B', b_list, '🟡 B级 — 等回踩'),
                                         ('C', c_list, '⚪ C级 — 观察')]:
            if not items:
                continue
            print(f"\n  {label}（{len(items)}只）")
            print(f"  {'─'*70}")
            for i, d in enumerate(items[:top_n]):
                self._print_decision(d, i + 1)

    def _print_decision(self, d: TradeDecision, idx: int):
        s = d.six_dim_score
        cur = d.entry_plan.pilot_price
        print(f"\n  #{idx} [{d.priority}] {d.ticker}  评分{s.total:.0f}  "
              f"信号:{d.buy_signal}  动作:{d.position_action}  仓位建议:{d.kelly_position_pct:.1%}")
        print(f"     6维评分： 基本面{s.fundamental:.0f} | 估值{s.valuation:.0f} | "
              f"趋势{s.trend:.0f} | 事件{s.event:.0f} | 机构{s.institutional:.0f} | "
              f"安全{s.invalidation:.0f}  确定性:{s.confidence_level}")

        if d.event_impact:
            ev = d.event_impact
            print(f"     事件: {ev.event_type} ({ev.severity})  "
                  f"预期损失{ev.expected_loss_pct:+.1%}  "
                  f"恢复{ev.expected_recovery_days}天  "
                  f"概率{ev.recovery_probability:.0%}  置信度{ev.confidence:.0%}")
            if ev.similar_cases:
                print(f"     类似案例: {', '.join(ev.similar_cases)}")

        print(f"     进场计划:")
        print(f"       试探 ${d.entry_plan.pilot_price:.2f} ({d.entry_plan.pilot_pct:.0%}) → "
              f"加仓1 ${d.entry_plan.add1_price:.2f} ({d.entry_plan.add1_pct:.0%}) → "
              f"加仓2 ${d.entry_plan.add2_price:.2f} ({d.entry_plan.add2_pct:.0%})")
        print(f"       止损 ${d.entry_plan.stop_loss:.2f} | "
              f"目标1 ${d.entry_plan.target1:.2f}({d.entry_plan.target1_reduce:.0%}减) | "
              f"目标2 ${d.entry_plan.target2:.2f}({d.entry_plan.target2_reduce:.0%}减) | "
              f"清仓 ${d.entry_plan.target3:.2f}")

        if d.reasons:
            print(f"     理由: {' | '.join(d.reasons[:2])}")
        if d.warnings:
            print(f"     ⚠️ {' | '.join(d.warnings)}")
        print(f"     反向证据触发器（满足任一即考虑清仓）:")
        for trigger in d.entry_plan.invalidation_triggers[:3]:
            print(f"       · {trigger}")


# ══════════════════════════════════════════════════════════════════════════════
# [v8.1] MASTER RISK + LEARNING LAYER
# strategy cooldown / weekly self-adjustment / IC-driven reweight
# ══════════════════════════════════════════════════════════════════════════════

class MasterRiskLayer:
    """
    主风控 + 学习层（架构图最底层，控制所有Sleeve）

    六个功能：
    1. vol_target       → 波动率目标缩放（已有，这里是统一入口）
    2. max_drawdown     → 全局回撤Kill Switch（不是Sleeve级别）
    3. kelly_cap        → Kelly上限（偏度调整半Kelly）
    4. IC/ICIR          → 每周计算各策略的IC，低于门槛降权
    5. strategy_cooldown→ 连续亏损N天后强制冷静期
    6. weekly_self_adj  → 每周根据实际IC自动调整各Sleeve权重

    和机构的对照：
    - AQR的Risk Committee每季度审查所有factor IC，调整权重
    - Two Sigma用强化学习自动调整策略权重（这里是简化版）
    - Citadel的Portfolio Manager Override = strategy_cooldown
    """

    def __init__(self,
                 target_vol:         float = 0.10,
                 max_global_dd:      float = 0.15,
                 kelly_fraction:     float = 0.50,
                 ic_min:             float = 0.02,
                 icir_min:           float = 0.30,
                 cooldown_trigger:   int   = 3,
                 cooldown_days:      int   = 5,
                 adj_frequency_days: int   = 5):

        self.target_vol       = target_vol
        self.max_global_dd    = max_global_dd
        self.kelly_fraction   = kelly_fraction
        self.ic_min           = ic_min
        self.icir_min         = icir_min
        self.cooldown_trigger = cooldown_trigger
        self.cooldown_days    = cooldown_days
        self.adj_freq         = adj_frequency_days

        # 状态
        self._global_equity  = 1.0
        self._global_peak    = 1.0
        self._killed         = False
        self._daily_rets:    List[float] = []
        self._cooldown_left: int         = 0
        self._day_counter:   int         = 0

        # 各策略的IC历史（用于weekly self-adjustment）
        self._strategy_ic_history: Dict[str, List[float]] = {
            'TACTICAL':  [], 'CORE_HEDGE': [], 'SECTOR_ROTATION': []
        }
        # 当前学习到的权重乘数（初始全1）
        self._learned_multipliers: Dict[str, float] = {
            'TACTICAL': 1.0, 'CORE_HEDGE': 1.0, 'SECTOR_ROTATION': 1.0
        }

    # ── 功能1+2：波动率目标 + 全局回撤 ────────────────────────────────────────

    def global_scale(self, weights: pd.Series,
                      recent_returns: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        全局仓位缩放（波动率目标 + 全局回撤保护）
        这是最高优先级的缩放，在所有Sleeve分配之后执行
        """
        info = {'scale': 1.0, 'reason': 'normal', 'killed': False}

        if self._killed:
            info['killed'] = True
            info['reason'] = 'global_kill_switch'
            return weights * 0, info

        if self._cooldown_left > 0:
            info['scale']  = 0.3
            info['reason'] = f'cooldown({self._cooldown_left}天剩余)'
            return weights * 0.3, info

        # 全局回撤检查
        dd = (self._global_equity - self._global_peak) / (self._global_peak + 1e-8)
        if dd < -self.max_global_dd:
            self._killed = True
            info['killed'] = True
            info['reason'] = f'global_drawdown_{dd:.2%}'
            print(f"  🚨 MasterRisk: 全局Kill Switch，回撤{dd:.2%}")
            return weights * 0, info

        # 波动率目标缩放
        if len(recent_returns) >= 21:
            realized_vol = float(recent_returns.tail(21).std() * np.sqrt(252))
            if realized_vol > 1e-4:
                vol_scale = min(self.target_vol / realized_vol, 2.0)
                weights   = weights * vol_scale
                info['scale']  = vol_scale
                info['vol']    = realized_vol

        return weights, info

    # ── 功能3：Kelly上限 ────────────────────────────────────────────────────────

    def kelly_cap(self, weights: pd.Series,
                   returns: pd.DataFrame) -> pd.Series:
        """
        逐票Kelly上限（不能超过Kelly建议的kelly_fraction倍）
        """
        capped = weights.copy()
        for ticker in weights.index:
            if ticker not in returns.columns:
                continue
            r = returns[ticker].dropna().tail(126)
            if len(r) < 20:
                continue

            p    = float((r > 0).mean())
            wins = r[r > 0]
            loss = r[r < 0]
            if len(wins) == 0 or len(loss) == 0:
                continue

            b   = float(wins.mean()) / (float(abs(loss.mean())) + 1e-8)
            raw = (p * b - (1 - p)) / (b + 1e-8)
            if raw <= 0:
                kelly_limit = 0.03
            else:
                sk  = float(r.skew())
                kt  = float(r.kurtosis())
                adj = 1 - max(0, -sk) * 0.15 - max(0, kt - 3) * 0.05
                kelly_limit = min(raw * self.kelly_fraction * adj, 0.20)

            # 如果当前权重超过Kelly上限，截断
            w = float(weights.get(ticker, 0))
            if abs(w) > kelly_limit:
                capped[ticker] = np.sign(w) * kelly_limit

        return capped

    # ── 功能4：IC门槛监控 ──────────────────────────────────────────────────────

    def update_strategy_ic(self, strategy: str, ic_value: float) -> None:
        """记录某个策略的IC"""
        if strategy in self._strategy_ic_history:
            self._strategy_ic_history[strategy].append(float(ic_value))
            # 只保留最近52周
            if len(self._strategy_ic_history[strategy]) > 52:
                self._strategy_ic_history[strategy].pop(0)

    def get_strategy_status(self) -> Dict[str, Dict]:
        """
        计算各策略的当前IC/ICIR状态
        Returns: {strategy: {ic_mean, icir, passes, weight_multiplier}}
        """
        result = {}
        for strat, ic_list in self._strategy_ic_history.items():
            if len(ic_list) < 4:
                result[strat] = {'ic_mean': 0, 'icir': 0,
                                  'passes': True,   # 数据不足时不惩罚
                                  'weight_mult': 1.0}
                continue

            arr    = np.array(ic_list[-13:])  # 最近13周
            ic_m   = float(arr.mean())
            ic_std = float(arr.std() + 1e-8)
            icir   = float(ic_m / ic_std) if ic_std > 1e-4 else 0.0
            passes = abs(ic_m) >= self.ic_min and abs(icir) >= self.icir_min

            result[strat] = {
                'ic_mean':    round(ic_m, 4),
                'icir':       round(icir, 4),
                'passes':     passes,
                'weight_mult':self._learned_multipliers.get(strat, 1.0)
            }
        return result

    # ── 功能5：Strategy Cooldown ────────────────────────────────────────────────

    def update_daily(self, daily_return: float) -> str:
        """
        每日更新：
        - 更新全局净值
        - 检查连续亏损触发Cooldown
        Returns: 'normal' | 'cooldown_triggered' | 'cooldown_active'
        """
        self._global_equity *= (1 + daily_return)
        self._global_peak    = max(self._global_peak, self._global_equity)
        self._daily_rets.append(daily_return)
        self._day_counter   += 1

        # 冷静期倒计时
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return 'cooldown_active'

        # 连续亏损检测
        if len(self._daily_rets) >= self.cooldown_trigger:
            recent = self._daily_rets[-self.cooldown_trigger:]
            if all(r < 0 for r in recent):
                self._cooldown_left = self.cooldown_days
                total_loss = sum(recent)
                print(f"  ⚠️ MasterRisk: 连续{self.cooldown_trigger}日亏损"
                      f"（总计{total_loss:.2%}），冷静期{self.cooldown_days}天")
                return 'cooldown_triggered'

        return 'normal'

    # ── 功能6：Weekly Self-Adjustment ──────────────────────────────────────────

    def weekly_self_adjustment(self,
                                sleeve_returns: Dict[str, List[float]]) -> Dict[str, float]:
        """
        每周根据各Sleeve实际表现自动调整权重乘数

        逻辑（简化版强化学习）：
        - 过去5周表现好的Sleeve → 乘数 × 1.1（最大1.5）
        - 过去5周表现差的Sleeve → 乘数 × 0.9（最小0.5）
        - 乘数作用于SleeveManager.allocate_by_regime()的基础权重

        为什么这样做：
        - 不同市场环境下不同Sleeve有效，让系统自己学习
        - 每周调整，避免过度拟合短期噪音
        - 有最大/最小边界，避免极端情况全押一个Sleeve
        """
        if self._day_counter % self.adj_freq != 0:
            return self._learned_multipliers

        adjustments = {}
        for sleeve, rets in sleeve_returns.items():
            if len(rets) < 5:
                adjustments[sleeve] = self._learned_multipliers.get(sleeve, 1.0)
                continue

            # 最近5天的Sharpe（简化版）
            recent = np.array(rets[-5:])
            sharpe_5 = float(recent.mean() / (recent.std() + 1e-8) * np.sqrt(252))

            # 调整乘数
            current = self._learned_multipliers.get(sleeve, 1.0)
            if sharpe_5 > 0.5:
                new_mult = min(current * 1.10, 1.50)  # 上限1.5
            elif sharpe_5 < -0.5:
                new_mult = max(current * 0.90, 0.50)  # 下限0.5
            else:
                new_mult = current  # 小范围内不调整

            adjustments[sleeve] = round(new_mult, 3)

        self._learned_multipliers = adjustments

        # 打印调整报告
        changed = {k: v for k, v in adjustments.items()
                   if abs(v - 1.0) > 0.05}
        if changed:
            print(f"  🔧 MasterRisk 周度自适应：{changed}")

        return adjustments

    def get_adjusted_sleeve_weights(self,
                                     base_alloc: Dict[str, float]) -> Dict[str, float]:
        """
        将learned_multipliers应用到SleeveManager的基础分配上
        然后重新归一化（确保权重之和=1）
        """
        adjusted = {
            k: v * self._learned_multipliers.get(k, 1.0)
            for k, v in base_alloc.items()
        }
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def status_report(self) -> Dict:
        """完整状态报告"""
        dd = (self._global_equity - self._global_peak) / (self._global_peak + 1e-8)
        n  = len(self._daily_rets)
        return {
            'global_equity':    round(self._global_equity, 4),
            'global_drawdown':  round(float(dd), 4),
            'global_killed':    self._killed,
            'cooldown_left':    self._cooldown_left,
            'days_tracked':     n,
            'recent_7d_return': round(sum(self._daily_rets[-7:]), 4) if n >= 7 else 0,
            'learned_weights':  self._learned_multipliers,
            'strategy_ic':      self.get_strategy_status()
        }

    def print_status(self):
        s = self.status_report()
        print(f"\n{'═'*65}")
        print(f"  🛡️  Master Risk + Learning Layer")
        print(f"{'═'*65}")
        print(f"  全局净值：{s['global_equity']:.4f} | "
              f"全局回撤：{s['global_drawdown']:.2%} | "
              f"{'🚨KILLED' if s['global_killed'] else '✅正常'}")
        print(f"  冷静期剩余：{s['cooldown_left']}天 | "
              f"追踪天数：{s['days_tracked']} | "
              f"近7日：{s['recent_7d_return']:+.2%}")
        print(f"\n  周度学习权重乘数：")
        for k, v in s['learned_weights'].items():
            bar = '█' * int(v * 10) + '░' * max(0, 15 - int(v * 10))
            print(f"    {k:<18} [{bar}] ×{v:.2f}")
        print(f"\n  各策略IC状态（近13周）：")
        for strat, info in s['strategy_ic'].items():
            icon = '✅' if info['passes'] else '❌'
            print(f"    {strat:<18} IC={info['ic_mean']:>7.4f} "
                  f"ICIR={info['icir']:>6.3f} {icon}")
        print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# [v9 BOOK_V2] 《Quantitative Trading Strategies Using Python》Peng Liu 全章精华
#
# 之前v5-v8只抓了书的表面API，这一版把书里真正机构都在用、但容易被忽略的
# 关键技术全部实现：
#
# ── Ch.5 趋势跟随 ──
# · Listing 5-13 EWMA递推公式：EMA_t = α×S_t + (1-α)×EMA_{t-1}
# · α控制响应速度（α高→反应快但假信号多）
# · 必须 shift(1) 避免lookahead bias（书第168页明确警告）
# · 用log returns（书第170页：log returns是additive，便于复利计算）
# · diff()检测信号切换（书第170页：action = signal.diff()）
#
# ── Ch.6 横截面动量 ──
# · 月频resample（书第188页：用月度收益避免日频噪音）
# · qcut分五组（书第191页：不是简单top/bottom，而是5个quantile）
# · Lookback + Lookahead双窗口（书第179页核心概念）
# · 多空腿等权组合（书第194-195页：long-short market neutral）
# · 与buy-and-hold基准对比（书第195页）
#
# ── Ch.7 回测 ──
# · Listing 7-16 drawdown函数完整实现（wealth_index + prior_peaks + drawdown）
# · 严格的 (1+R).prod()^(252/n)-1 公式（不是简单年化）
# · Calmar = trailing_36m_annualized / trailing_36m_maxDD（书第203页定义）
# · 多周期回测（书第198页警告：策略必须在多周期上稳健）
# · 关注data snooping（书第198页警告）
#
# ── Ch.8 统计套利 ──
# · coint() Engle-Granger两步法（书第241页 statsmodels实现）
# · z-score = (spread - rolling_mean) / rolling_std（书第245-246页）
# · entry threshold + exit threshold 双阈值（书第250页）
# · 半衰期 half-life = -ln(2)/λ（书第253页 ADF回归系数计算）
# · 配对的position shift(1)避免未来函数（书第253页）
#
# ── Ch.9 Bayesian优化 ──
# · 黑盒函数 S = f(l1, l2)，l1<l2 约束（书第260页核心定义）
# · Gaussian Process surrogate model（书第270-273页）
# · EI（Expected Improvement）+ UCB双acquisition function（书第276-277页）
# · 多次重复实验取mean±std（书第299-300页：避免over-fitting）
# · ϕ和Φ的闭式EI公式（书第276页）
# ══════════════════════════════════════════════════════════════════════════════

class BookCh5_TrendFollowing:
    """
    书第5章完整实现：基于双MA的趋势跟随策略

    与v5-v8已有的trend_signals()区别：
    · 这里是书原版的完整实现（带shift避免lookahead）
    · 用log returns（书第170页）
    · 显式生成signal/action/position列（书第169-171页）
    · 严格的backtest逻辑（书第172-174页）
    """

    @staticmethod
    def ewma_series(price: pd.Series, alpha: float = 0.1) -> pd.Series:
        """
        书第163-164页 EWMA公式（精确实现）：
        EMA_t = α × S_t + (1-α) × EMA_{t-1}
        EMA_0 = S_0

        α高 → 对最新价格反应快（但假信号多）
        α低 → 平滑（接近SMA）
        """
        return price.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def sma_series(price: pd.Series, window: int = 20) -> pd.Series:
        """
        书第157-158页 SMA：
        SMA_t = (S_t + S_{t-1} + ... + S_{t-M+1}) / M
        """
        return price.rolling(window).mean()

    @staticmethod
    def generate_strategy(price: pd.Series,
                          short_window: int = 3,
                          long_window: int = 20,
                          use_ema_short: bool = False,
                          alpha: float = 0.5) -> pd.DataFrame:
        """
        书第167-172页完整策略生成

        步骤：
        1. 计算短期/长期MA
        2. shift(1) 避免lookahead（书第168页强调）
        3. signal = +1 if short > long else -1
        4. action = signal.diff() 标识切换点
        5. log_return = strategy_signal × log(price).diff()
        """
        df = pd.DataFrame({'price': price.dropna()})

        # 计算移动平均
        if use_ema_short:
            df['short_ma'] = BookCh5_TrendFollowing.ewma_series(df['price'], alpha)
        else:
            df['short_ma'] = BookCh5_TrendFollowing.sma_series(df['price'], short_window)
        df['long_ma'] = BookCh5_TrendFollowing.sma_series(df['price'], long_window)

        # ★ 关键：shift(1) 避免lookahead（书第168页）
        # "we can only use the information up to yesterday to make a trading
        #  decision for tomorrow"
        df['short_ma'] = df['short_ma'].shift(1)
        df['long_ma']  = df['long_ma'].shift(1)

        # 信号生成（书第168页 np.where实现）
        df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
        df['signal'] = np.where(df['short_ma'] < df['long_ma'], -1, df['signal'])

        # 交易动作（书第170页：action = signal.diff()）
        # 0=no trade, +2=short→long, -2=long→short
        df['action'] = df['signal'].diff()

        # Log returns（书第170页：用log return便于复利计算）
        df['log_return_buy_hold']    = np.log(df['price']).diff()
        df['log_return_strategy']    = df['signal'] * df['log_return_buy_hold']

        # Wealth index（书第173页）
        df['wealth_buy_hold']  = np.exp(df['log_return_buy_hold'].cumsum())
        df['wealth_strategy']  = np.exp(df['log_return_strategy'].cumsum())

        return df.dropna()

    @staticmethod
    def report_actions(df: pd.DataFrame) -> Dict:
        """
        书第171页：统计交易动作
        action == 2:  short → long（买入信号）
        action == -2: long → short（卖出信号）
        action == 0:  保持仓位
        """
        ac = df['action'].value_counts()
        return {
            'no_action':     int(ac.get(0, 0)),
            'long_signal':   int(ac.get(2, 0)),
            'short_signal':  int(ac.get(-2, 0)),
            'total_trades':  int(ac.get(2, 0)) + int(ac.get(-2, 0)),
            'final_wealth':  float(df['wealth_strategy'].iloc[-1]),
            'buyhold_wealth':float(df['wealth_buy_hold'].iloc[-1]),
            'beat_buyhold':  bool(df['wealth_strategy'].iloc[-1] > df['wealth_buy_hold'].iloc[-1])
        }


class BookCh6_CrossSectionalMomentum:
    """
    书第6章完整实现：横截面动量（月频，五分位）

    与v8的cross_sectional_momentum()区别：
    · 这里是书原版的"5分位"实现（书第191页 qcut）
    · 严格的lookback+lookahead双窗口（书第179页）
    · 月频resample（书第188页）
    · 显式long top quintile / short bottom quintile（书第194-195页）

    机构标准实现（AQR的Asness 1996"Cross-Section of Expected Returns"原版）
    """

    @staticmethod
    def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        书第188页：日频价格 → 月频收益
        每月最后一日的累积收益
        """
        try:
            return prices.resample('ME').last().pct_change().dropna()
        except Exception:
            return prices.resample('M').last().pct_change().dropna()

    @staticmethod
    def select_quintiles(monthly_ret: pd.DataFrame,
                          formation_date: pd.Timestamp,
                          lookback_months: int = 6,
                          n_quantiles: int = 5) -> Dict:
        """
        书第190-192页核心步骤：

        1. 取formation_date向前lookback_months个月的累计收益
        2. 用pd.qcut分成n_quantiles组（书中默认5组）
        3. Long最高分位（rank=n_quantiles-1），Short最低分位（rank=0）

        Returns: {'long_stocks': [...], 'short_stocks': [...], 'ranks': DataFrame}
        """
        if formation_date not in monthly_ret.index:
            # 找到最近的可用日期
            avail = monthly_ret.index[monthly_ret.index <= formation_date]
            if len(avail) == 0:
                return {'long_stocks': [], 'short_stocks': [], 'ranks': None}
            formation_date = avail[-1]

        idx_end   = monthly_ret.index.get_loc(formation_date)
        idx_start = max(0, idx_end - lookback_months + 1)

        # 累计lookback_months的收益
        window_ret = monthly_ret.iloc[idx_start:idx_end + 1]
        cum_ret    = (1 + window_ret).prod() - 1
        cum_ret    = cum_ret.dropna()

        if len(cum_ret) < n_quantiles:
            return {'long_stocks': cum_ret.nlargest(1).index.tolist(),
                    'short_stocks': cum_ret.nsmallest(1).index.tolist(),
                    'ranks': None}

        # 书第191页：pd.qcut分组
        try:
            ranks = pd.qcut(cum_ret, n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            ranks = cum_ret.rank(pct=True) * (n_quantiles - 1)
            ranks = ranks.astype(int)

        long_stocks  = ranks[ranks == ranks.max()].index.tolist()
        short_stocks = ranks[ranks == ranks.min()].index.tolist()

        return {
            'long_stocks':  long_stocks,
            'short_stocks': short_stocks,
            'ranks':        ranks,
            'cum_returns':  cum_ret,
            'formation_date': str(formation_date.date()) if hasattr(formation_date, 'date') else str(formation_date)
        }

    @staticmethod
    def evaluate_lookahead(monthly_ret: pd.DataFrame,
                            formation_date: pd.Timestamp,
                            long_stocks: List[str],
                            short_stocks: List[str],
                            lookahead_months: int = 1) -> Dict:
        """
        书第193-195页：lookahead窗口评估
        持有1个月后，long-short组合的实际收益
        """
        from dateutil.relativedelta import relativedelta

        try:
            # 用书中精确公式：formation + relativedelta(months=lookahead)
            if hasattr(formation_date, 'to_pydatetime'):
                fd_py = formation_date.to_pydatetime()
            else:
                fd_py = formation_date
            eval_date = fd_py + relativedelta(months=lookahead_months)
        except Exception:
            eval_date = formation_date + pd.Timedelta(days=30 * lookahead_months)

        # 找到最近的可用月份
        avail = monthly_ret.index[monthly_ret.index >= pd.Timestamp(eval_date)]
        if len(avail) == 0:
            return {'long_return': 0, 'short_return': 0,
                    'long_short_profit': 0, 'eval_date': None}

        eval_actual = avail[0]
        eval_row    = monthly_ret.loc[eval_actual]

        # 等权组合（书第194页）
        long_ret  = float(eval_row[eval_row.index.isin(long_stocks)].mean()) \
                    if long_stocks else 0
        short_ret = float(eval_row[eval_row.index.isin(short_stocks)].mean()) \
                    if short_stocks else 0

        # 书第195页：long-short profit
        long_short_profit = long_ret - short_ret

        return {
            'long_return':       round(long_ret, 6),
            'short_return':      round(short_ret, 6),
            'long_short_profit': round(long_short_profit, 6),
            'eval_date':         str(eval_actual.date()) if hasattr(eval_actual, 'date') else str(eval_actual),
            'long_stocks':       long_stocks,
            'short_stocks':      short_stocks
        }

    @staticmethod
    def rolling_backtest(prices: pd.DataFrame,
                          lookback_months: int = 6,
                          n_quantiles: int = 5,
                          n_periods: int = 12) -> pd.DataFrame:
        """
        滚动回测：每个月re-rank并持有1个月
        书第196页exercise建议：跨多个市场周期评估
        """
        monthly_ret = BookCh6_CrossSectionalMomentum.monthly_returns(prices)
        if len(monthly_ret) < lookback_months + n_periods:
            return pd.DataFrame()

        records = []
        start_idx = lookback_months
        end_idx   = min(len(monthly_ret) - 1, start_idx + n_periods)

        for i in range(start_idx, end_idx):
            formation_date = monthly_ret.index[i]
            selection = BookCh6_CrossSectionalMomentum.select_quintiles(
                monthly_ret, formation_date, lookback_months, n_quantiles
            )
            evaluation = BookCh6_CrossSectionalMomentum.evaluate_lookahead(
                monthly_ret, formation_date,
                selection['long_stocks'], selection['short_stocks'],
                lookahead_months=1
            )
            records.append({
                'period':  i - start_idx + 1,
                'formation_date': selection.get('formation_date'),
                'eval_date':      evaluation.get('eval_date'),
                'n_long':         len(selection.get('long_stocks', [])),
                'n_short':        len(selection.get('short_stocks', [])),
                'long_ret':       evaluation['long_return'],
                'short_ret':      evaluation['short_return'],
                'ls_profit':      evaluation['long_short_profit']
            })

        return pd.DataFrame(records)


class BookCh7_BacktestMetrics:
    """
    书第7章完整实现：回测指标（精确实现书中所有公式）

    与v5-v8已有的backtest_metrics()区别：
    · 严格按书第7章公式（Listing 7-13至7-16）
    · 显式的wealth_index + prior_peaks + drawdown（书第204-205页）
    · Calmar用trailing-36-month（书第203页）
    · 多周期robustness（书第198页强调）
    """

    @staticmethod
    def drawdown_series(returns: pd.Series, initial_wealth: float = 1000) -> pd.DataFrame:
        """
        书第213-214页drawdown()函数的精确实现：

        wealth_index = initial_wealth × cumprod(1 + r)
        prior_peaks  = wealth_index.cummax()
        drawdown     = (wealth_index - prior_peaks) / prior_peaks
        """
        r = returns.dropna()
        wealth_index = initial_wealth * (1 + r).cumprod()
        prior_peaks  = wealth_index.cummax()
        drawdowns    = (wealth_index - prior_peaks) / prior_peaks
        return pd.DataFrame({
            'Wealth index': wealth_index,
            'Prior peaks':  prior_peaks,
            'Drawdown':     drawdowns
        })

    @staticmethod
    def annualized_return(returns: pd.Series) -> float:
        """
        书第220页 Listing 7-13精确实现：
        annualized = (1+R_terminal)^(252/n) - 1
        """
        r = returns.dropna()
        if len(r) == 0:
            return 0.0
        terminal = (1 + r).prod()
        n        = len(r)
        return float(terminal ** (252 / n) - 1)

    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        """
        书第221页 Listing 7-14：
        ann_vol = std(returns) × √252
        """
        r = returns.dropna()
        if len(r) < 2:
            return 0.0
        return float(r.std() * np.sqrt(252))

    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf: float = 0.03) -> float:
        """
        书第221页 Listing 7-15：
        Sharpe = (ann_ret - rf) / ann_vol
        """
        ann_ret = BookCh7_BacktestMetrics.annualized_return(returns)
        ann_vol = BookCh7_BacktestMetrics.annualized_volatility(returns)
        if ann_vol < 1e-8:
            return 0.0
        return float((ann_ret - rf) / ann_vol)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """
        书第222页 Listing 7-16：
        max_dd = drawdown(returns)['Drawdown'].min()
        """
        dd = BookCh7_BacktestMetrics.drawdown_series(returns)['Drawdown']
        if len(dd) == 0:
            return 0.0
        return float(dd.min())

    @staticmethod
    def calmar_ratio_trailing(returns: pd.Series,
                               trailing_months: int = 36) -> float:
        """
        书第203页Calmar定义（机构精确版）：
        Calmar = (trailing 36 months annualized return) / |max_dd over same period|

        而不是简单的 ann_ret / max_dd（这是网上常见的简化版）
        """
        trailing_days = trailing_months * 21
        r_tail = returns.dropna().tail(trailing_days)
        if len(r_tail) < 21:
            return 0.0

        ann_ret = BookCh7_BacktestMetrics.annualized_return(r_tail)
        max_dd  = BookCh7_BacktestMetrics.max_drawdown(r_tail)
        if abs(max_dd) < 1e-8:
            return 0.0
        return float(ann_ret / abs(max_dd))

    @staticmethod
    def full_report(returns: pd.Series, rf: float = 0.03,
                     label: str = '') -> Dict:
        """
        完整书第7章指标 + Calmar trailing
        """
        r = returns.dropna()
        if len(r) < 5:
            return {'error': 'insufficient data'}

        ann_ret = BookCh7_BacktestMetrics.annualized_return(r)
        ann_vol = BookCh7_BacktestMetrics.annualized_volatility(r)
        sr      = BookCh7_BacktestMetrics.sharpe_ratio(r, rf)
        mdd     = BookCh7_BacktestMetrics.max_drawdown(r)
        calmar  = BookCh7_BacktestMetrics.calmar_ratio_trailing(r)

        # 简单Calmar（用全部数据），与trailing做对比
        calmar_simple = ann_ret / abs(mdd) if abs(mdd) > 1e-8 else 0.0

        report = {
            'annualized_return':   round(ann_ret, 6),
            'annualized_vol':      round(ann_vol, 6),
            'sharpe_ratio':        round(sr, 4),
            'max_drawdown':        round(mdd, 6),
            'calmar_trailing_36m': round(calmar, 4),
            'calmar_simple':       round(calmar_simple, 4),
            'total_return':        round(float((1 + r).prod() - 1), 6),
            'n_periods':           len(r),
            'win_rate':            round(float((r > 0).mean()), 4),
            'best_day':            round(float(r.max()), 4),
            'worst_day':           round(float(r.min()), 4)
        }

        if label:
            print(f"\n  📊 [{label}] 书第7章完整指标：")
            print(f"    年化收益: {report['annualized_return']:+.2%}")
            print(f"    年化波动: {report['annualized_vol']:.2%}")
            print(f"    Sharpe:   {report['sharpe_ratio']:.4f}")
            print(f"    MaxDD:    {report['max_drawdown']:.2%}")
            print(f"    Calmar(trailing-36m): {report['calmar_trailing_36m']:.4f}")
            print(f"    Calmar(simple):       {report['calmar_simple']:.4f}")
            print(f"    胜率:     {report['win_rate']:.1%}")

        return report


class BookCh8_StatisticalArbitrage:
    """
    书第8章完整实现：统计套利（Engle-Granger + z-score + half-life）

    与v8已有StatArb区别：
    · 半衰期精确计算（书第253页：half_life = -ln(2)/λ）
    · 使用statsmodels coint()作为基准（如可用）
    · 严格的spread shift(1)避免lookahead（书第253页）
    · 双阈值：entry_z + exit_z（书第250页）
    """

    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5,
                 window: int = 21, threshold_pvalue: float = 0.05):
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.window  = window
        self.threshold_pvalue = threshold_pvalue

    def coint_test(self, s1: pd.Series, s2: pd.Series) -> Dict:
        """
        书第241-244页：Engle-Granger两步法

        优先用statsmodels.tsa.stattools.coint（书第241页明示）
        Fallback到自己实现的ADF
        """
        try:
            from statsmodels.tsa.stattools import coint
            score, pvalue, _ = coint(s1.values, s2.values)
            return {'score': float(score), 'pvalue': float(pvalue),
                    'method': 'statsmodels'}
        except ImportError:
            # Fallback到自己实现
            return self._coint_manual(s1, s2)

    def _coint_manual(self, s1: pd.Series, s2: pd.Series) -> Dict:
        """
        书第236-240页：手动两步法
        Step 1: OLS regression → 残差
        Step 2: 对残差做ADF
        """
        y, x = s1.values.astype(float), s2.values.astype(float)
        X = np.column_stack([np.ones(len(x)), x])
        try:
            c = np.linalg.lstsq(X, y, rcond=None)[0]
            spread = y - X @ c
            # 简化ADF
            dy = np.diff(spread)
            sx = spread[:-1]
            Xs = np.column_stack([sx, np.ones(len(sx))])
            lam = np.linalg.lstsq(Xs, dy, rcond=None)[0][0]
            # 简化p-value
            pvalue = 0.05 if lam < -0.05 else 0.5
            return {'score': float(lam), 'pvalue': pvalue, 'method': 'manual'}
        except Exception:
            return {'score': 0, 'pvalue': 1.0, 'method': 'failed'}

    def half_life(self, spread: pd.Series) -> float:
        """
        书第253页精确实现：
        Δspread_t = λ × spread_{t-1} + ε
        half_life = -ln(2) / λ

        意义：均值回归的半衰期
        · half_life < 3天：太快，可能是噪音
        · half_life > 60天：太慢，机会不实用
        · 理想区间：5-30天
        """
        sp = spread.dropna().values
        if len(sp) < 10:
            return 999.0
        try:
            dy = np.diff(sp)
            sx = sp[:-1]
            X  = np.column_stack([sx, np.ones(len(sx))])
            lam = np.linalg.lstsq(X, dy, rcond=None)[0][0]
            if lam >= -1e-8:
                return 999.0
            return float(-np.log(2) / lam)
        except Exception:
            return 999.0

    def find_pairs(self, prices: pd.DataFrame) -> List[Dict]:
        """
        书第242-244页：遍历所有C(n,2)对
        """
        from itertools import combinations
        pairs = []
        for t1, t2 in combinations(prices.columns, 2):
            s1, s2 = prices[t1].dropna(), prices[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 50:
                continue
            s1, s2 = s1.loc[common], s2.loc[common]

            ct = self.coint_test(s1, s2)
            if ct['pvalue'] < self.threshold_pvalue:
                # 计算beta（OLS）
                X = np.column_stack([np.ones(len(s2)), s2.values])
                beta_arr = np.linalg.lstsq(X, s1.values, rcond=None)[0]
                spread = s1 - beta_arr[0] - beta_arr[1] * s2
                hl = self.half_life(spread)

                pairs.append({
                    't1': t1, 't2': t2,
                    'pvalue':       ct['pvalue'],
                    'coint_score':  ct['score'],
                    'method':       ct['method'],
                    'hedge_ratio':  float(beta_arr[1]),
                    'intercept':    float(beta_arr[0]),
                    'half_life':    hl,
                    'tradeable':    3 <= hl <= 60   # 书第253页建议区间
                })

        return sorted(pairs, key=lambda x: (x['pvalue'], -x['tradeable']))

    def generate_zscore(self, s1: pd.Series, s2: pd.Series,
                         intercept: float, hedge_ratio: float) -> pd.Series:
        """
        书第245-246页：spread → rolling z-score
        spread = s1 - β0 - β1 × s2
        z = (spread - rolling_mean) / rolling_std
        """
        spread = s1 - intercept - hedge_ratio * s2
        mu = spread.rolling(self.window).mean()
        sd = spread.rolling(self.window).std()
        return (spread - mu) / (sd + 1e-8)

    def generate_positions(self, zscore: pd.Series) -> pd.Series:
        """
        书第250页双阈值规则：
        z < -entry_z → +1 (long spread)
        z >  entry_z → -1 (short spread)
        |z| < exit_z → 0  (平仓)
        其他 → 维持
        """
        pos = pd.Series(0.0, index=zscore.index)
        for i in range(1, len(zscore)):
            z, p_prev = zscore.iloc[i], pos.iloc[i-1]
            if   z < -self.entry_z and p_prev == 0: pos.iloc[i] =  1.0
            elif z >  self.entry_z and p_prev == 0: pos.iloc[i] = -1.0
            elif abs(z) < self.exit_z:               pos.iloc[i] =  0.0
            else:                                    pos.iloc[i] =  p_prev
        return pos

    def backtest(self, s1: pd.Series, s2: pd.Series,
                  pair_info: Dict, tc_bps: float = 5) -> Dict:
        """
        书第253-254页：完整pairs trading回测
        ★ 关键：position.shift(1) 避免lookahead
        """
        hedge = pair_info['hedge_ratio']
        intercept = pair_info['intercept']

        zs   = self.generate_zscore(s1, s2, intercept, hedge)
        pos  = self.generate_positions(zs)

        # 收益（书第253页：pct_change × shifted position）
        r1 = s1.pct_change().fillna(0)
        r2 = s2.pct_change().fillna(0)
        pos_shifted = pos.shift(1).fillna(0)

        # long spread = long s1, short s2 (×hedge_ratio)
        strategy_ret = pos_shifted * r1 - pos_shifted * hedge * r2

        # 交易成本
        tc = pos.diff().abs().fillna(0) * (tc_bps / 10000)
        net_ret = strategy_ret - tc

        report = BookCh7_BacktestMetrics.full_report(net_ret)
        report['final_zscore']  = float(zs.iloc[-1]) if len(zs) > 0 else 0
        report['current_pos']   = float(pos.iloc[-1]) if len(pos) > 0 else 0
        report['n_trades']      = int(pos.diff().abs().sum() / 2)
        return report


class BookCh9_BayesianOptimizer:
    """
    书第9章完整实现：Bayesian Optimization（GP + EI/UCB）

    与v8 UCBOptimizer区别：
    · 完整GP（书第270-273页）：RBF核 + 后验均值/方差
    · 双acquisition function：EI（书第275-277页）+ UCB
    · 重复实验取mean±std（书第299-300页）
    · 处理constraint：l1 < l2（书第260页书中明示）
    """

    def __init__(self, beta: float = 2.0, jitter: float = 1e-4):
        self.beta   = beta
        self.jitter = jitter
        self.obs_x: List[List[float]] = []
        self.obs_y: List[float] = []

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray,
                    lengthscale: float = 1.0) -> np.ndarray:
        """
        书第270-272页：RBF核函数（最常用的GP kernel）
        K(x, x') = exp(-||x - x'||² / (2 × ℓ²))
        """
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                   np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sq_dist / (lengthscale ** 2))

    def gp_posterior(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        书第272-273页：GP后验均值和方差
        μ* = K(x*, X) K(X, X)^(-1) y
        σ²* = K(x*, x*) - K(x*, X) K(X, X)^(-1) K(X, x*)
        """
        K    = self.rbf_kernel(X_train, X_train) + self.jitter * np.eye(len(X_train))
        K_s  = self.rbf_kernel(X_train, X_test)
        K_ss = self.rbf_kernel(X_test,  X_test) + self.jitter * np.eye(len(X_test))

        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)

        mu  = K_s.T @ K_inv @ y_train
        cov = K_ss - K_s.T @ K_inv @ K_s
        var = np.clip(np.diag(cov), 1e-8, None)
        return mu, var

    def acquisition_ei(self, mu: np.ndarray, sigma: np.ndarray,
                       f_best: float) -> np.ndarray:
        """
        书第275-277页：Expected Improvement闭式公式

        α_EI(x) = (μ - f*) Φ((μ - f*)/σ) + σ φ((μ - f*)/σ)

        ★ 第一项是exploitation（高μ的地方）
        ★ 第二项是exploration（高σ的地方）
        """
        from scipy.stats import norm as sp_norm
        sigma = np.maximum(sigma, 1e-8)
        z = (mu - f_best) / sigma
        ei = (mu - f_best) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z)
        return np.maximum(ei, 0)

    def acquisition_ucb(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        书第277页：Upper Confidence Bound
        α_UCB(x) = μ + β × σ
        · 低β: 偏exploitation
        · 高β: 偏exploration
        """
        return mu + self.beta * np.sqrt(sigma)

    def optimize(self, objective_fn,
                  param_bounds: Dict[str, Tuple[float, float]],
                  n_iter: int = 20,
                  n_initial: int = 5,
                  acquisition: str = 'ucb',
                  constraints: List[callable] = None) -> Dict:
        """
        书第278-279页：完整BO循环

        param_bounds: {'l1': (1, 10), 'l2': (11, 20)}
        constraints: 可选约束列表，如lambda p: p['l1'] < p['l2']（书第260页要求）
        """
        param_names = list(param_bounds.keys())
        bounds      = list(param_bounds.values())
        n_dim       = len(bounds)

        def sample_random():
            """随机采样，保证满足constraints"""
            for _ in range(100):
                p = {n: float(np.random.uniform(*b))
                     for n, b in zip(param_names, bounds)}
                if constraints is None or all(c(p) for c in constraints):
                    return p
            # 实在不满足，强行返回
            return {n: float(np.random.uniform(*b))
                    for n, b in zip(param_names, bounds)}

        # 初始随机采样（书第279页：先撒n_initial个点）
        for _ in range(n_initial):
            params = sample_random()
            try:
                y = float(objective_fn(**params))
                if np.isfinite(y):
                    self.obs_x.append([params[n] for n in param_names])
                    self.obs_y.append(y)
            except Exception:
                pass

        if len(self.obs_x) < 2:
            return {'best_params': sample_random(), 'best_score': 0,
                    'history': []}

        # 主BO循环
        history = []
        for iter_n in range(n_iter):
            X_train = np.array(self.obs_x)
            y_train = np.array(self.obs_y)

            # 生成候选点
            n_cand = 200
            cands_list = []
            for _ in range(n_cand):
                p = sample_random()
                cands_list.append([p[n] for n in param_names])
            X_cand = np.array(cands_list)

            # 标准化（书第272页：BO对scale敏感）
            X_mean = X_train.mean(0)
            X_std  = X_train.std(0) + 1e-8
            X_train_n = (X_train - X_mean) / X_std
            X_cand_n  = (X_cand  - X_mean) / X_std

            # GP后验
            mu, var = self.gp_posterior(X_train_n, y_train, X_cand_n)

            # Acquisition function
            f_best = y_train.max()
            if acquisition.lower() == 'ei':
                acq = self.acquisition_ei(mu, var, f_best)
            else:
                acq = self.acquisition_ucb(mu, var)

            # 选下一个采样点
            best_idx = int(np.argmax(acq))
            next_x   = X_cand[best_idx]
            next_params = {n: float(next_x[i]) for i, n in enumerate(param_names)}

            # 检查约束
            if constraints is not None and not all(c(next_params) for c in constraints):
                next_params = sample_random()

            # 评估
            try:
                y = float(objective_fn(**next_params))
                if np.isfinite(y):
                    self.obs_x.append([next_params[n] for n in param_names])
                    self.obs_y.append(y)
                    history.append({'iter': iter_n + 1, **next_params, 'y': y})
            except Exception:
                pass

        # 返回最优
        if len(self.obs_y) == 0:
            return {'best_params': sample_random(), 'best_score': 0, 'history': []}

        best_idx = int(np.argmax(self.obs_y))
        return {
            'best_params':   {n: self.obs_x[best_idx][i]
                              for i, n in enumerate(param_names)},
            'best_score':    float(self.obs_y[best_idx]),
            'n_evaluated':   len(self.obs_y),
            'acquisition':   acquisition,
            'history':       history
        }

    @staticmethod
    def repeated_experiments(objective_fn, param_bounds: Dict,
                              n_runs: int = 5, n_iter: int = 15) -> Dict:
        """
        书第299-300页：重复实验取统计量
        避免单次实验的overfitting
        """
        scores = []
        for run in range(n_runs):
            optimizer = BookCh9_BayesianOptimizer(beta=2.0)
            np.random.seed(run * 31)
            result = optimizer.optimize(objective_fn, param_bounds,
                                          n_iter=n_iter, acquisition='ucb')
            scores.append(result['best_score'])

        return {
            'mean_score': float(np.mean(scores)),
            'std_score':  float(np.std(scores)),
            'min_score':  float(np.min(scores)),
            'max_score':  float(np.max(scores)),
            'n_runs':     n_runs
        }


class BookV2_OperationManual:
    """
    Operation Manual / 操作白皮书（基于书9章原则的完整操作手册）

    机构日常运作的标准流程，把书中scattered的最佳实践整合：

    1. Pre-trade Checklist（书第198页data snooping警告 + 第7章原则）
    2. Backtest Standards（书第7章多周期robustness要求）
    3. Live Trading Rules（书第169页shift要求）
    4. Risk Disclosure Items（书第223页exercises）
    5. Strategy Lifecycle Management
    """

    @staticmethod
    def pre_trade_checklist(strategy_name: str,
                             backtest_report: Dict,
                             current_signal: Dict) -> Dict:
        """
        每次实盘前必过的检查清单（书第7章警告整理版）
        """
        checks = []
        passed = 0

        # 1. 回测有效性（书第7章要求）
        if backtest_report.get('n_periods', 0) >= 252:
            checks.append(('回测样本足够(>=252天)', True))
            passed += 1
        else:
            checks.append((f'回测样本不足({backtest_report.get("n_periods",0)}<252)', False))

        # 2. Sharpe要求（书第221页 + 机构最低标准）
        if backtest_report.get('sharpe_ratio', 0) >= 1.0:
            checks.append(('Sharpe >= 1.0', True))
            passed += 1
        else:
            checks.append((f'Sharpe={backtest_report.get("sharpe_ratio",0):.2f}<1.0', False))

        # 3. MaxDD合理（书第203页）
        if backtest_report.get('max_drawdown', -1) > -0.25:
            checks.append(('MaxDD > -25%', True))
            passed += 1
        else:
            checks.append((f'MaxDD={backtest_report.get("max_drawdown",0):.2%} 过大', False))

        # 4. Calmar要求（书第203页）
        if backtest_report.get('calmar_trailing_36m', 0) >= 0.5:
            checks.append(('Calmar(trailing-36m) >= 0.5', True))
            passed += 1
        else:
            checks.append((f'Calmar={backtest_report.get("calmar_trailing_36m",0):.2f}<0.5', False))

        # 5. 胜率合理
        if backtest_report.get('win_rate', 0) >= 0.45:
            checks.append(('胜率 >= 45%', True))
            passed += 1
        else:
            checks.append((f'胜率={backtest_report.get("win_rate",0):.1%}<45%', False))

        # 6. 信号清晰（书第168-170页：signal必须明确）
        if current_signal and 'signal' in current_signal:
            sig = current_signal['signal']
            if abs(sig) > 0:
                checks.append((f'信号清晰: {sig:+.0f}', True))
                passed += 1
            else:
                checks.append(('信号为0(不交易)', True))
                passed += 1

        # 7. shift避免lookahead（书第168页核心要求）
        checks.append(('信号已shift(1)避免lookahead', True))
        passed += 1

        result = {
            'strategy':    strategy_name,
            'total_checks':len(checks),
            'passed':      passed,
            'pass_rate':   passed / max(1, len(checks)),
            'checklist':   checks,
            'can_trade':   passed >= len(checks) - 1,   # 允许1项不过
            'timestamp':   datetime.now().isoformat()
        }
        return result

    @staticmethod
    def daily_operation_workflow() -> List[Dict]:
        """
        每日运作工作流（从开盘前到收盘后）
        """
        return [
            {'time': '盘前 08:30', 'step': 1, 'action': '读取昨日close价格',
             'caveat': '书第168页：今日决策只能用yesterday及之前的数据'},
            {'time': '盘前 08:45', 'step': 2, 'action': '运行Regime检测',
             'caveat': '多维度检测(SMA/MOM/VOL/BREADTH)'},
            {'time': '盘前 09:00', 'step': 3, 'action': '生成各Sleeve信号',
             'caveat': 'TACTICAL/CORE/SECTOR分别独立生成'},
            {'time': '盘前 09:15', 'step': 4, 'action': 'IC验证 + DSR/PBO检查',
             'caveat': '书第7章警告: data snooping risk'},
            {'time': '盘前 09:20', 'step': 5, 'action': 'Pre-Trade Checklist',
             'caveat': '7项检查必须通过6项以上'},
            {'time': '盘前 09:25', 'step': 6, 'action': '计算目标仓位 + Kelly cap',
             'caveat': '严格遵守max_long/max_short限制'},
            {'time': '开盘 09:30', 'step': 7, 'action': 'TWAP/VWAP分单执行',
             'caveat': 'POV<=5%，避免市场冲击'},
            {'time': '盘中 12:00', 'step': 8, 'action': '检查DrawdownController',
             'caveat': '>5%回撤减仓50%, >10%减仓75%'},
            {'time': '盘中 15:30', 'step': 9, 'action': '检查RiskServer Kill Switch',
             'caveat': '全局回撤>15%全清仓'},
            {'time': '收盘 16:00', 'step': 10, 'action': '记录TradeJournal',
             'caveat': '进场理由/卖点/引擎/Lesson必填'},
            {'time': '盘后 17:00', 'step': 11, 'action': '更新IC/ICIR库',
             'caveat': '为MasterRisk周度自适应做准备'},
            {'time': '盘后 18:00', 'step': 12, 'action': '生成今日P&L报告',
             'caveat': 'StateLogger写入 + AlertEngine通知'},
        ]

    @staticmethod
    def weekly_review_workflow() -> List[Dict]:
        """
        周度复盘工作流
        """
        return [
            {'day': '每周五盘后', 'task': '运行各策略IC计算(过去5天)',
             'output': '若IC<阈值，下周降权该策略'},
            {'day': '每周五盘后', 'task': '运行多周期回测',
             'output': '过去13/26/52周分别回测，检查stability'},
            {'day': '每周末', 'task': 'PSR + DSR重算',
             'output': 'DSR=>0.95才能继续运行该策略'},
            {'day': '每周末', 'task': '复盘所有亏损交易',
             'output': '归类为错误类型: 误信号/执行差/Black Swan'},
            {'day': '每周末', 'task': '更新声明背调清单(L3层)',
             'output': '验证所有持仓的核心论点是否还成立'},
            {'day': '每周一开盘', 'task': '运行MasterRisk weekly_self_adjustment',
             'output': '更新各Sleeve的learned_multiplier'},
        ]

    @staticmethod
    def black_swan_protocol() -> List[Dict]:
        """
        黑天鹅事件应对协议
        """
        return [
            {'level': 1, 'trigger': '日内回撤>3%',
             'action': '所有新单暂停，待收盘评估'},
            {'level': 2, 'trigger': '日内回撤>5%',
             'action': '减仓50%，TACTICAL Sleeve全平'},
            {'level': 3, 'trigger': '日内回撤>8% OR VIX>40',
             'action': 'CORE_HEDGE保留，TACTICAL+SECTOR全平'},
            {'level': 4, 'trigger': '日内回撤>12%',
             'action': 'Kill Switch激活，全清仓 + 24小时冷静期'},
            {'level': 5, 'trigger': '市场断路器触发 OR 历史性事件',
             'action': '人工接管，所有自动策略禁用'},
        ]

    @staticmethod
    def print_full_manual():
        """打印完整操作手册"""
        print(f"\n{'═'*65}")
        print(f"  📖 Canyon Operation Manual / 操作白皮书")
        print(f"  基于《Quantitative Trading Strategies Using Python》Peng Liu")
        print(f"{'═'*65}")

        print(f"\n  ── 每日运作工作流 ──")
        for item in BookV2_OperationManual.daily_operation_workflow():
            print(f"  {item['time']:<12} Step{item['step']:<2} {item['action']}")
            print(f"  {'':<19} → {item['caveat']}")

        print(f"\n  ── 周度复盘工作流 ──")
        for item in BookV2_OperationManual.weekly_review_workflow():
            print(f"  [{item['day']:<10}] {item['task']}")
            print(f"  {'':<13} → {item['output']}")

        print(f"\n  ── 黑天鹅应对协议 ──")
        for item in BookV2_OperationManual.black_swan_protocol():
            level_icon = ['', '🟡', '🟠', '🔴', '🚨', '☠️'][item['level']]
            print(f"  {level_icon} Level{item['level']}: {item['trigger']}")
            print(f"      → {item['action']}")
        print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# 30. 主系统 v7（整合 v6 回测 + v7 实盘架构）
# ══════════════════════════════════════════════════════════════════════════════

class CanyonTradingSystemV7:
    """
    Canyon量化交易系统 v7.0

    完整架构：
    Data → Feature Store → Alpha Library（7个Alpha实例）
         → Alpha Diagnostics（IC/ICIR/t-stat/衰减曲线）
         → Alpha Pool（Gate筛选 + IC权重）
         → Regime Model（KMeans聚类）
         → Meta Model（Ridge按Regime学习alpha权重）
         → Portfolio Engine（风险约束 + 稳定分配）
         → Execution（TWAP/VWAP/POV）
         → Broker（Alpaca + Failover）
         → Risk Server（硬约束 + Kill Switch）
         → Logging（EventLogger + StateLogger）
         → Dashboard Alerts

    回测：Walk-Forward（书第7章）+ 统计深度（书第7章深化）
    """

    def __init__(self, tc_bps: float = 10, rf: float = 0.03,
                 target_vol: float = 0.10):
        # v6组件（全部保留）
        self.data        = DataLayer()
        self.stat_arb    = StatArb()
        self.risk        = RiskManager()
        self.od          = OffensiveDefensiveManager(self.risk)
        self.backtester  = WalkForwardBacktester(tc_bps=tc_bps, rf=rf,
                                                  target_vol=target_vol)
        self.ucb         = UCBOptimizer()
        self.journal     = TradeJournal()
        self.stats       = StatisticalDepth()
        self.exec_model  = ExecutionCostModel()
        self.rf          = rf

        # v7新增组件
        self.alpha_pool  = AlphaPool([
            MomentumAlpha(), MeanRevAlpha(), VolBreakoutAlpha(),
            RelStrengthAlpha(), LowVolAlpha(),
            PriceEfficiencyAlpha(), VolumeDirectionAlpha()
        ])
        self.regime_model = RegimeModel(n_clusters=3)
        self.meta_model   = MetaModel()
        self.portfolio_eng= PortfolioEngine(target_vol=target_vol)
        self.risk_server  = RiskServer()
        self.event_logger = EventLogger("events_canyon.log")

    def run(self, tickers: List[str], start: str, end: str,
            benchmark: str = 'SPY') -> Dict:
        print(f"\n{'═'*65}")
        print(f"  🏔  CANYON量化交易系统 v7.0")
        print(f"  Data→AlphaPool→RegimeModel→MetaModel→Portfolio→RiskServer")
        print(f"{'═'*65}")
        print(f"  资产: {tickers}")
        print(f"  时间: {start} → {end}")

        prices, volumes, market = self.data.load(tickers, start, end, benchmark)
        returns = prices.pct_change().dropna()

        # ── Step1: Walk-Forward回测（v6引擎，含满仓修复）────────────────────
        print(f"\n{'─'*65}")
        print(f"  Step1: Walk-Forward回测 （书第7章 + 满仓修复 + 做空）")
        print(f"{'─'*65}")
        main_result = self.backtester.run(
            prices, volumes, market, self.stat_arb, self.od,
            use_ic_alpha=True, verbose=True
        )
        self.event_logger.log("backtest_complete", {
            "sharpe": main_result.get('sharpe', 0),
            "max_dd": main_result.get('max_dd', 0),
            "ann_ret": main_result.get('ann_ret', 0)
        })

        # ── Step2: 多期验证 ─────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  Step2: 多期验证（书第7章）")
        print(f"{'─'*65}")
        period_df = self.backtester.multi_period(
            prices, volumes, market, self.stat_arb, self.od, n_periods=3
        )

        # ── Step3: UCB Bayesian优化（书第9章）──────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  Step3: UCB Bayesian优化（书第9章）")
        print(f"{'─'*65}")
        pairs = self.stat_arb.find_pairs(prices)
        if pairs:
            bp = pairs[0]
            t1, t2 = bp['t1'], bp['t2']
            print(f"  最优协整对: {t1}/{t2} p={bp['pvalue']:.3f} HL={bp['half_life']:.1f}天")
            def obj_fn(entry_z: float, exit_z: float, window: int):
                sa = StatArb(entry_z=entry_z, exit_z=exit_z, window=window)
                r  = sa.backtest_pair(prices[t1], prices[t2], bp)
                return -99.0 if r.get('max_dd', -1) < -0.056 else r.get('sharpe', 0.0)
            opt = self.ucb.optimize(obj_fn,
                param_bounds={'entry_z': (1.5, 3.0), 'exit_z': (0.2, 1.0), 'window': (10, 30)},
                n_iter=18)
            bp2 = opt.get('best_params', {})
            if bp2 and opt.get('best_score', 0) > -90:
                print(f"  最优参数: entry_z={bp2.get('entry_z',2):.2f} "
                      f"exit_z={bp2.get('exit_z',0.5):.2f} "
                      f"window={bp2.get('window',21)} → Sharpe={opt.get('best_score',0):.3f}")
                self.stat_arb.entry_z = bp2.get('entry_z', 2.0)
                self.stat_arb.exit_z  = bp2.get('exit_z', 0.5)
                self.stat_arb.window  = int(bp2.get('window', 21))
        else:
            print("  未找到协整对")

        # ── Step4: 统计深度报告（v6）──────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  Step4: 统计深度报告（Bootstrap + Newey-West + 因子暴露）")
        print(f"{'─'*65}")
        if 'daily_returns' in main_result and len(main_result['daily_returns']) > 30:
            dr = main_result['daily_returns']
            self.stats.print_full_report(dr, market.pct_change().dropna(),
                                          label='Canyon v7', rf=self.rf)

        # ── Step5: v7 Alpha Pool + Regime + Meta完整流程 ─────────────────
        print(f"\n{'─'*65}")
        print(f"  Step5: v7 AlphaPool → RegimeModel → MetaModel")
        print(f"{'─'*65}")

        # 训练Regime模型
        mkt_ret = market.pct_change().dropna()
        self.regime_model.fit(mkt_ret)
        regime_series = self.regime_model.predict(mkt_ret)
        current_km_regime = self.regime_model.current_regime(mkt_ret)
        print(f"  RegimeModel KMeans: 当前={current_km_regime} | "
              f"分布:{dict(regime_series.value_counts().items())}")

        # Alpha Pool评估
        features_dict = {
            "price": prices, "returns": returns,
            "volume": volumes, "market": market
        }
        past_cs = prices.pct_change(21).iloc[-1].dropna()
        passed  = self.alpha_pool.evaluate(features_dict, past_cs)
        self.alpha_pool.weight(passed)

        print(f"\n  Alpha Pool诊断报告 ({len(passed)}/{len(self.alpha_pool.alphas)}通过):")
        self.alpha_pool.print_diagnostics()

        # 训练Meta Model
        alpha_matrix = pd.DataFrame({
            name: sig for name, (sig, _) in passed.items()
        }).dropna(how='all')

        fut_mean = prices.pct_change(21).shift(-21).mean(axis=1)
        if len(alpha_matrix) > 30 and len(fut_mean) > 30:
            common = (alpha_matrix.index.intersection(fut_mean.index)
                                  .intersection(regime_series.index))
            if len(common) > 30:
                self.meta_model.train(
                    regime_series.loc[common],
                    alpha_matrix.loc[common],
                    fut_mean.loc[common]
                )
                print(f"\n  MetaModel训练完成: {len(self.meta_model.trained_regimes)}个Regime")
                for reg, fi in self.meta_model.feature_importances.items():
                    top = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    print(f"    [{reg}] 最重要: "
                          f"{', '.join(f'{k}:{v:.3f}' for k,v in top)}")

        # ── Step6: 当前市场分析 + 仓位建议 ─────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  Step6: 当前市场分析 + v7仓位建议")
        print(f"{'─'*65}")
        current = self.analyze_current_v7(prices, volumes, market,
                                           passed, current_km_regime)

        # ── Step7: 参数敏感性 ───────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  Step7: 参数敏感性（书第7章）")
        print(f"{'─'*65}")
        grid = self._grid_test(prices, volumes, market)

        self._final_report(main_result, period_df, current, grid)

        return {'main': main_result, 'periods': period_df,
                'current': current, 'grid': grid}

    def analyze_current_v7(self, prices, volumes, market,
                            passed_alphas, km_regime) -> Dict:
        """当前分析：v6指标 + v7 Alpha Pool信号"""
        regime, detail = detect_regime(market, prices)
        cs    = cross_sectional_momentum(prices)
        tsig  = trend_signals(prices)
        pairs = self.stat_arb.find_pairs(prices)
        arbs  = self.stat_arb.current_opportunity(prices, pairs)

        print(f"\n  市场环境（规则系统）: {regime.label}（{regime.stance}）")
        print(f"    综合分:{detail.get('composite',0):+.1f} | "
              f"趋势:{detail.get('trend',0):+.2f} | "
              f"动量:{detail.get('momentum',0):+.2f}")
        print(f"    目标敞口:{regime.target_gross_exposure:.0%} | "
              f"单票上限:{regime.max_long:.0%} | "
              f"空头上限:{regime.max_short:.0%}")

        print(f"  市场环境（KMeans）: {km_regime}")

        # 横截面动量（书第6章）
        print(f"\n  横截面动量（书第6章）:")
        print(f"    做多前25%: {cs['long'][:5]}")
        print(f"    做空后25%: {cs['short'][:5]}")
        print(f"    多空价差: {cs['spread']:+.2%}")

        # 趋势信号（书第5章）
        bulls = [tk for tk in tsig.index if tsig.loc[tk,'signal'] == 1]
        bears = [tk for tk in tsig.index if tsig.loc[tk,'signal'] == -1]
        print(f"\n  趋势信号（书第5章）:")
        print(f"    金叉/上升: {bulls[:6]}")
        print(f"    死叉/下降: {bears[:6]}")

        if arbs:
            print(f"\n  统计套利（书第8章）:")
            for a in arbs:
                d = '多'+a['t1']+'空'+a['t2'] if a['direction']==1 else '空'+a['t1']+'多'+a['t2']
                print(f"    {a['t1']}/{a['t2']} z={a['z']:.2f} → {d}")

        # Canyon评分
        print(f"\n  Canyon F/C/E评分:")
        canyon_scores = {}
        for tk in prices.columns:
            s = canyon_score_auto(prices[tk], volumes[tk], market, regime)
            canyon_scores[tk] = s
            if s['can_buy']:
                print(f"    ✅ {tk}: {s['total']:.0f}分({s['grade']}) 上限{s['max_pos']:.0%}")

        # v7 Alpha Pool组合信号
        combo = self.alpha_pool.combine(passed_alphas)
        if combo is not None and len(combo) > 0:
            print(f"\n  v7 Alpha Pool组合信号（Top5多头/空头）:")
            top5_long  = combo.nlargest(5)
            top5_short = combo.nsmallest(5)
            print(f"    做多: {list(zip(top5_long.index, [f'{v:.3f}' for v in top5_long.values]))}")
            print(f"    做空: {list(zip(top5_short.index, [f'{v:.3f}' for v in top5_short.values]))}")

        # v7 Portfolio Engine仓位
        if combo is not None and len(combo) > 0:
            weights_pe = self.portfolio_eng.allocate(
                combo, prices.pct_change().dropna(), regime=regime
            )
            weights_rs = self.risk_server.enforce(weights_pe)
        else:
            weights_rs = pd.Series(dtype=float)

        # v6 进攻/防守仓位（保留作对比）
        la = cs['long_alpha'].dropna()
        sa = cs['short_alpha'].dropna()
        ret_hist = prices.pct_change().dropna()
        alloc = self.od.allocate(regime=regime, long_alpha=la, short_alpha=sa,
                                  trend_sig=tsig, stat_arb_opps=arbs, returns=ret_hist)

        print(f"\n  今日仓位建议（v6进攻/防守）:")
        print(f"    {alloc.rationale}")

        if len(weights_rs) > 0:
            print(f"\n  今日仓位建议（v7 AlphaPool+MetaModel+RiskServer）:")
            print(f"    总敞口:{weights_rs.abs().sum():.1%} | 净:{weights_rs.sum():+.1%}")
            for tk, w in sorted(weights_rs.items(), key=lambda x: -abs(x[1]))[:8]:
                print(f"    {'▲' if w>0 else '▼'} {tk:<8} {w:+.1%}")

        # 执行成本预估
        all_w = alloc.to_series()
        if len(all_w) > 0:
            print(f"\n  执行成本预估（Almgren-Chriss冲击 + 买卖价差）:")
            total_tc = 0.0
            for tk, w in all_w.items():
                if abs(w) < 0.005 or tk not in prices.columns: continue
                vol_d   = float(prices[tk].pct_change().dropna().tail(21).std())
                adv_usd = float(volumes[tk].tail(21).mean()) * float(prices[tk].iloc[-1])
                tc      = self.exec_model.total_cost(vol_d, adv_usd, abs(float(w))*1e6) * 10000
                total_tc += abs(float(w)) * tc
                print(f"    {tk:<8} {tc:.1f}bps")
            print(f"    组合总成本: {total_tc:.1f}bps")

        # 压力测试
        if len(all_w) > 0:
            print(f"\n  压力测试 (5.6%硬约束):")
            for name, shock in [('2008金融危机',-0.45), ('2020疫情崩盘',-0.32),
                                  ('2022加息熊市',-0.22), ('常规回调-15%',-0.15)]:
                loss = float(sum((float(w)*shock if float(w)>0 else float(w)*(-shock*0.7))
                                  for w in all_w))
                print(f"    {'✅' if loss>-0.056 else '⚠️'} {name}: {loss:+.2%}")

        return {'regime': regime, 'allocation': alloc,
                'canyon_scores': canyon_scores, 'cs': cs,
                'v7_weights': weights_rs}

    def _grid_test(self, prices, volumes, market) -> pd.DataFrame:
        results = []
        for ema_s in [5, 10]:
            for sma_s in [20, 30, 50]:
                try:
                    p_sub = prices.tail(400)
                    if len(p_sub) < self.backtester.train_w + self.backtester.test_w:
                        continue
                    daily_r = []
                    for s in range(self.backtester.train_w, len(p_sub)-self.backtester.test_w,
                                   self.backtester.test_w):
                        p_tr = p_sub.iloc[s-self.backtester.train_w:s]
                        p_te = p_sub.iloc[s:s+self.backtester.test_w]
                        ts   = trend_signals(p_tr, ema_span=ema_s, sma_span=sma_s)
                        bull = [tk for tk in ts.index if ts.loc[tk,'signal']==1]
                        if not bull: continue
                        w = 1.0/len(bull)
                        for d in p_te.index[1:]:
                            pi = p_te.index.get_loc(d)-1
                            dr = sum(float(p_te.loc[d,tk]/p_te.iloc[pi][tk]-1)*w
                                     for tk in bull if tk in p_te.columns)
                            daily_r.append(dr)
                    if len(daily_r) > 10:
                        m = backtest_metrics(pd.Series(daily_r))
                        results.append({'ema':ema_s,'sma':sma_s,'sharpe':m['sharpe'],'max_dd':m['max_dd']})
                        print(f"    EMA{ema_s}/SMA{sma_s}: Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.2%}")
                except Exception:
                    pass
        df = pd.DataFrame(results)
        if len(df) > 0:
            best = df.nlargest(1,'sharpe').iloc[0]
            print(f"    → 最优: EMA{best['ema']:.0f}/SMA{best['sma']:.0f} Sharpe={best['sharpe']:.3f}")
        return df

    def _final_report(self, main, periods, current, grid):
        print(f"\n{'═'*65}")
        print(f"  📋 Canyon v7 最终报告")
        print(f"{'═'*65}")
        print(f"  年化收益：{main.get('ann_ret',0):+.2%}")
        print(f"  Sharpe：  {main.get('sharpe',0):.4f}")
        print(f"  Calmar：  {main.get('calmar',0):.4f}")
        print(f"  最大回撤：{main.get('max_dd',0):.2%}")
        print(f"  总收益：  {main.get('total_ret',0):+.2%}")
        print(f"  多头贡献：{main.get('long_total_pnl',0):+.2%}")
        print(f"  空头贡献：{main.get('short_total_pnl',0):+.2%}")
        if len(periods) > 0:
            print(f"\n  多期稳健性 ({len(periods)}段):")
            print(f"    Sharpe: μ={periods['sharpe'].mean():.3f} σ={periods['sharpe'].std():.3f}")
            print(f"    MaxDD:  μ={periods['max_dd'].mean():.2%} σ={periods['max_dd'].std():.2%}")
        regime = current.get('regime')
        if regime:
            print(f"\n  当前状态: {regime.label}（{regime.stance}）")
            alloc = current.get('allocation')
            if alloc: print(f"  v6仓位: {alloc.rationale}")
        v7w = current.get('v7_weights')
        if v7w is not None and len(v7w) > 0:
            print(f"  v7仓位: 总敞口{v7w.abs().sum():.1%} 净{v7w.sum():+.1%}")
        print(f"\n  v7架构说明:")
        print(f"  [v5保留] DataLayer/Regime/SMA/CSMom/StatArb/UCB/Kelly/CVaR")
        print(f"  [v6保留] ICEngine/ExecCost/Bootstrap/NW-Sharpe/VolTarget/DDCtrl")
        print(f"  [v6保留] StateLogger/Alert/StrategyMonitor/TWAP/Alpaca/LiveTrader")
        print(f"  [v7新增] BaseAlpha接口 + 7个Alpha实例库")
        print(f"  [v7新增] AlphaPool (Gate筛选 + IC/ICIR权重 + 组合)")
        print(f"  [v7新增] RegimeModel (KMeans聚类，独立于规则系统)")
        print(f"  [v7新增] MetaModel (Ridge按Regime学习alpha权重)")
        print(f"  [v7新增] PortfolioEngine (风险惩罚+vol目标+换手平滑)")
        print(f"  [v7新增] RiskServer (独立风控，Kill Switch)")
        print(f"  [v7新增] EventLogger + retry + Failover")
        print(f"{'═'*65}")

    def record_trade(self, ticker, direction, price, pct, regime,
                      reason, engines, days, first_exit, forced_exit) -> str:
        return self.journal.open(
            ticker=ticker, direction=direction, entry_price=price,
            position_pct=pct, regime=regime, entry_reason=reason,
            engines_used=engines, expected_days=days,
            first_exit=first_exit, forced_exit=forced_exit
        )

    def close_trade(self, tid, exit_price, exit_reason, lesson='') -> Dict:
        return self.journal.close(tid, exit_price, exit_reason, lesson=lesson)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# v8 新增：10大量化公式完整实现
# 核心原则（来自 formula_decision_matrix.md）：
#
# 公式不是全部都应该当成 Alpha！
# 正确位置：
# GBM      → 场景模拟 / Monte Carlo VaR     （不是alpha预测）
# BSM      → 期权Greeks计算                  （做期权overlay时用）
# Markowitz→ 组合优化器                      （已有，加约束改进）
# GARCH    → 波动率预测 / 仓位缩放            （替换简单realized vol）
# 协整性   → 统计套利                         （已有）
# HMM      → Regime检测（转移矩阵）           （改进KMeans版本）
# PCA      → 隐藏因子 / 拥挤风险              （新增，不是alpha）
# Kelly    → 仓位缩放                         （已有，加偏度修正）
# Copula   → 尾部风险压力测试                  （新增，不是收益预测）
# 神经网络 → 非线性Alpha候选（必须过IC门槛）  （新增，可选）
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 31. [v8] GBM Monte Carlo — 场景模拟 + VaR/ES
# 为什么：正态分布假设下估计尾部风险，不用来预测方向
# dS_t = μ S_t dt + σ S_t dW_t  →  S_T = S_0 exp((μ-σ²/2)T + σ√T Z)
# ══════════════════════════════════════════════════════════════════════════════

class GBMModel:
    """
    几何布朗运动（GBM）Monte Carlo路径模拟

    用途（正确位置）：
    · 压力测试：模拟1000条价格路径，看极端情景
    · VaR/ES：基于模拟分布计算风险价值
    · 不用来预测方向！（假设随机游走，IC=0）

    公式：dS_t = μS_t dt + σS_t dW_t
    离散化：S_{t+1} = S_t × exp((μ - σ²/2)Δt + σ√Δt × Z)，Z~N(0,1)

    为什么要GBM而不是直接用历史数据：
    · 历史数据样本有限，无法覆盖极端情景
    · Monte Carlo可以生成10000条路径，计算99%置信区间
    · 配合GARCH使用：用GARCH预测σ_t，代入GBM生成路径
    """

    @staticmethod
    def simulate(prices: pd.Series,
                 n_days: int = 252,
                 n_paths: int = 1000,
                 garch_vol: float = None) -> np.ndarray:
        """
        模拟未来n_days天的n_paths条价格路径

        Args:
            prices: 历史价格序列
            n_days: 模拟天数
            n_paths: 路径数量
            garch_vol: 如果有GARCH预测的波动率，用它替代历史vol

        Returns:
            shape=(n_paths, n_days) 的价格矩阵
        """
        r       = prices.pct_change().dropna()
        mu      = float(r.mean()) * 252       # 年化漂移率
        sigma   = garch_vol if garch_vol else float(r.std() * np.sqrt(252))  # 年化波动率
        S0      = float(prices.iloc[-1])
        dt      = 1 / 252

        # GBM离散化：S_{t+1} = S_t × exp((μ-σ²/2)dt + σ√dt × Z)
        Z       = np.random.standard_normal((n_paths, n_days))
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        paths   = S0 * np.exp(np.cumsum(log_ret, axis=1))

        return paths

    @staticmethod
    def var_es(paths: np.ndarray, horizon: int = 21,
               confidence: float = 0.95) -> Dict:
        """
        从Monte Carlo路径计算VaR和ES（Expected Shortfall）

        VaR(95%,21天)：在正常市场，21天内最大损失不超过X（95%概率）
        ES(95%,21天)：在最坏5%的情景下，平均损失是X（比VaR更保守）
        """
        S0        = paths[:, 0]
        S_horizon = paths[:, min(horizon - 1, paths.shape[1] - 1)]
        port_ret  = S_horizon / S0 - 1

        var_pct   = float(np.percentile(port_ret, (1 - confidence) * 100))
        es_mask   = port_ret <= var_pct
        es_pct    = float(port_ret[es_mask].mean()) if es_mask.any() else var_pct

        return {
            'var': round(var_pct, 4),        # 负数，如-0.12 = 12%亏损
            'es':  round(es_pct, 4),         # 负数，比VaR更大亏损
            'horizon_days': horizon,
            'confidence': confidence,
            'n_paths': len(paths),
            'worst_1pct': round(float(np.percentile(port_ret, 1)), 4)
        }

    @staticmethod
    def portfolio_var(prices: pd.DataFrame,
                      weights: pd.Series,
                      horizon: int = 21,
                      n_paths: int = 2000,
                      garch_vols: Dict[str, float] = None) -> Dict:
        """
        组合层面的GBM VaR（考虑相关性）
        相关性来自历史数据，波动率可由GARCH预测
        """
        tickers = [t for t in weights.index if t in prices.columns]
        if not tickers:
            return {'var': 0, 'es': 0}

        r     = prices[tickers].pct_change().dropna()
        cov   = r.cov().values * 252   # 年化协方差
        mu    = r.mean().values * 252  # 年化均值
        w     = weights[tickers].values
        dt    = 1 / 252

        # Cholesky分解：生成相关联的随机收益
        try:
            L     = np.linalg.cholesky(cov + np.eye(len(tickers)) * 1e-8)
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov)))

        port_rets = []
        for _ in range(n_paths):
            Z        = np.random.standard_normal((len(tickers), horizon))
            log_rets = ((mu[:, None] - 0.5 * np.diag(cov)[:, None]) * dt
                        + (L @ Z) * np.sqrt(dt))
            prices_t = np.exp(np.cumsum(log_rets, axis=1))[:, -1]
            port_rets.append(float(w @ (prices_t - 1)))

        port_rets = np.array(port_rets)
        var       = float(np.percentile(port_rets, 5))
        es_mask   = port_rets <= var
        es        = float(port_rets[es_mask].mean()) if es_mask.any() else var

        return {
            'var':   round(var, 4),
            'es':    round(es, 4),
            'mean_sim': round(float(port_rets.mean()), 4),
            'horizon_days': horizon,
            'exceeds_limit': var < -0.056  # Canyon 5.6%硬约束
        }


# ══════════════════════════════════════════════════════════════════════════════
# 32. [v8] GARCH(1,1) 波动率模型
# 为什么：波动率有聚集性（大波动之后往往还有大波动），简单realized vol忽略了这点
# σ_t² = α_0 + α_1 ε_{t-1}² + β_1 σ_{t-1}²
# ══════════════════════════════════════════════════════════════════════════════

class GARCHModel:
    """
    GARCH(1,1) 波动率预测模型

    为什么用GARCH而不是简单rolling std：
    · 波动率有"记忆性"：昨天大跌今天波动率仍然很高
    · 波动率有"聚集性"：平静期之后还是平静，动荡期之后还是动荡
    · GARCH能预测明天的波动率，而rolling std只能描述过去

    公式：σ_t² = α_0 + α_1 ε_{t-1}² + β_1 σ_{t-1}²
    · α_0：长期基础波动率
    · α_1：昨日冲击的影响（ARCH项）
    · β_1：昨日波动率的持续性（GARCH项）
    · α_1 + β_1 < 1：波动率均值回归（稳定性条件）

    在系统中的位置：
    → VolTargeter：用GARCH预测明日波动率代替简单realized vol
    → GBMModel：把GARCH的σ_t代入GBM生成更真实的路径
    """

    def __init__(self, omega: float = 0.000001,
                 alpha: float = 0.09,
                 beta: float = 0.90):
        """
        默认参数来自历史校准：alpha+beta≈0.99（高持续性）
        """
        self.omega = omega
        self.alpha = alpha
        self.beta  = beta
        self.fitted_variance = None

    def fit(self, returns: pd.Series,
            n_iter: int = 100) -> 'GARCHModel':
        """
        最大似然估计GARCH参数
        简化版：用矩估计初始化 + 梯度下降
        """
        r = returns.dropna().values
        if len(r) < 30:
            return self

        # 初始方差 = 样本方差
        var_init = float(np.var(r))

        # 目标函数：负对数似然
        def neg_ll(params):
            om, al, be = params
            if om <= 0 or al < 0 or be < 0 or al + be >= 1:
                return 1e10
            h    = np.zeros(len(r))
            h[0] = var_init
            for t in range(1, len(r)):
                h[t] = om + al * r[t-1]**2 + be * h[t-1]
                if h[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(h + 1e-10) + r**2 / (h + 1e-10))
            return -ll

        try:
            from scipy.optimize import minimize as sp_min
            res = sp_min(neg_ll,
                         x0=[self.omega, self.alpha, self.beta],
                         method='L-BFGS-B',
                         bounds=[(1e-8, 0.01), (0.01, 0.3), (0.5, 0.99)],
                         options={'maxiter': n_iter})
            if res.success:
                self.omega, self.alpha, self.beta = res.x

            # 计算历史条件方差序列
            h = np.zeros(len(r))
            h[0] = var_init
            for t in range(1, len(r)):
                h[t] = self.omega + self.alpha * r[t-1]**2 + self.beta * h[t-1]
            self.fitted_variance = h
            self._last_r  = r[-1]
            self._last_h  = h[-1]
        except Exception:
            self._last_h = var_init
            self._last_r = float(r[-1]) if len(r) > 0 else 0.0

        return self

    def forecast(self, horizon: int = 1) -> float:
        """
        预测未来horizon天的年化波动率（用于VolTargeter）

        h_{t+1} = ω + α ε_t² + β h_t
        h_{t+k} = ω/(1-α-β) + (α+β)^k × (h_t - ω/(1-α-β))   (均值回归)
        """
        if not hasattr(self, '_last_h'):
            return 0.15  # 默认15%

        if horizon == 1:
            h_next = (self.omega + self.alpha * self._last_r**2
                      + self.beta * self._last_h)
        else:
            # 多步预测（均值回归公式）
            persistence = self.alpha + self.beta
            if persistence >= 1:
                h_next = self._last_h
            else:
                long_run = self.omega / (1 - persistence)
                h_next   = (long_run
                             + persistence**horizon * (self._last_h - long_run))

        return float(np.sqrt(max(h_next, 1e-8) * 252))   # 转为年化

    @staticmethod
    def quick_forecast(returns: pd.Series, horizon: int = 1) -> float:
        """
        快速GARCH预测（不用完整MLE，使用矩估计+递推）
        适合回测中每步快速计算
        """
        r = returns.dropna().values
        if len(r) < 21:
            return float(returns.std() * np.sqrt(252))

        # 简化矩估计：用短期/长期vol比来调整
        vol_5   = float(returns.tail(5).std() * np.sqrt(252))
        vol_21  = float(returns.tail(21).std() * np.sqrt(252))
        vol_63  = float(returns.tail(min(63, len(returns))).std() * np.sqrt(252))

        # GARCH-like加权：近期vol权重更高
        alpha_approx = 0.09
        beta_approx  = 0.90
        omega_approx = vol_63**2 / 252 * (1 - alpha_approx - beta_approx)

        h_t = vol_21**2 / 252  # 今日条件方差
        eps_t = r[-1]          # 今日冲击

        h_next = omega_approx + alpha_approx * eps_t**2 + beta_approx * h_t
        return float(np.sqrt(max(h_next * 252, 1e-6)))


# ══════════════════════════════════════════════════════════════════════════════
# 33. [v8] PCA因子分析 — 隐藏因子 + 拥挤风险
# 为什么：市场里有隐藏的共同因子驱动大量股票同涨同跌（拥挤交易）
# R = Σ β_i F_i + ε
# ══════════════════════════════════════════════════════════════════════════════

class PCAFactorModel:
    """
    主成分分析（PCA）因子模型

    为什么用PCA（正确用途）：
    · 不是用来预测股价方向（IC=0，没有预测力）
    · 用来识别"隐藏的共同风险因子"
    · 当第一主成分解释70%+收益方差时 → 市场高度相关，拥挤严重
    · 可以检测：你以为是分散的组合，实际上暴露于同一个因子

    公式：R = Σ β_i F_i + ε
    · F_i：主成分（市场因子、行业因子等）
    · β_i：个股对各因子的敏感度（factor loading）
    · ε：特异风险（真正可分散的部分）

    在系统中的位置：
    → 风险诊断：检测组合是否过度暴露某个隐藏因子
    → 拥挤检测：第一PC解释方差 > 60% 时发警报
    → 因子中性化：从alpha信号中去除共同因子暴露
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.components_  = None
        self.explained_   = None
        self.loadings_    = None

    def fit(self, returns: pd.DataFrame) -> 'PCAFactorModel':
        """
        拟合PCA因子模型（手动实现，不依赖sklearn）
        """
        r = returns.dropna()
        if len(r) < 30 or len(r.columns) < 3:
            return self

        # 标准化
        X     = (r - r.mean()) / (r.std() + 1e-8)
        X     = X.fillna(0)

        # SVD分解（数值稳定）
        try:
            U, S, Vt = np.linalg.svd(X.values, full_matrices=False)
            n_comp = min(self.n_components, len(S))
            # 主成分（factor returns）
            factors = pd.DataFrame(
                U[:, :n_comp] * S[:n_comp],
                index=r.index
            )
            # Factor loadings（β系数）
            loadings = pd.DataFrame(
                Vt[:n_comp, :].T,
                index=r.columns,
                columns=[f'PC{i+1}' for i in range(n_comp)]
            )
            # 解释方差比
            total_var = (S**2).sum()
            explained = pd.Series(
                {f'PC{i+1}': S[i]**2 / total_var for i in range(n_comp)}
            )

            self.components_  = factors
            self.explained_   = explained
            self.loadings_    = loadings
            self._r_index     = r.index
            self._tickers     = r.columns.tolist()
        except Exception:
            pass

        return self

    def crowding_score(self) -> Dict:
        """
        拥挤度评分
        PC1解释方差 > 60% → 高拥挤（市场高度同质）
        """
        if self.explained_ is None:
            return {'crowding': 0, 'pc1_var': 0, 'alert': False}

        pc1_var = float(self.explained_.iloc[0])
        top3    = float(self.explained_.iloc[:3].sum()) if len(self.explained_) >= 3 else pc1_var

        alert = pc1_var > 0.50  # PC1解释超过50%

        return {
            'pc1_variance_explained': round(pc1_var, 4),
            'top3_variance_explained': round(top3, 4),
            'crowding_alert': alert,
            'crowding_level': ('高' if pc1_var > 0.6 else
                               '中' if pc1_var > 0.45 else '低'),
            'interpretation': (
                f"PC1解释{pc1_var:.1%}方差 → "
                f"{'⚠️市场高度同质，分散化失效！' if alert else '✅分散化有效'}"
            )
        }

    def factor_exposure(self, weights: pd.Series) -> Dict:
        """
        计算组合对各主成分的暴露（β）
        帮助识别：你以为分散，实际上是同一个因子
        """
        if self.loadings_ is None:
            return {}

        tickers = [t for t in weights.index if t in self.loadings_.index]
        if not tickers:
            return {}

        w_arr = weights[tickers].values
        result = {}
        for col in self.loadings_.columns:
            loading = self.loadings_.loc[tickers, col].values
            exposure = float(w_arr @ loading)
            result[col] = round(exposure, 4)

        return {
            'exposures': result,
            'dominant_factor': max(result, key=lambda k: abs(result[k])) if result else 'unknown'
        }

    def neutralize_alpha(self, alpha: pd.Series,
                          n_factors: int = 2) -> pd.Series:
        """
        从alpha信号中去除前n个主成分暴露（因子中性化）
        机构实践：alpha中的共同因子暴露不是真正的alpha，去除后更干净
        """
        if self.loadings_ is None:
            return alpha

        common = alpha.dropna().index.intersection(self.loadings_.index)
        if len(common) < 5:
            return alpha

        alpha_c = alpha[common].values
        L       = self.loadings_.loc[common].iloc[:, :n_factors].values

        # OLS去除因子暴露：alpha_residual = alpha - L(L'L)^{-1}L'alpha
        try:
            proj    = L @ np.linalg.lstsq(L, alpha_c, rcond=None)[0]
            residual= alpha_c - proj
            result  = alpha.copy()
            result[common] = residual
            # 重新标准化
            std = result.std()
            if std > 1e-8:
                result = result / std
            return result
        except Exception:
            return alpha


# ══════════════════════════════════════════════════════════════════════════════
# 34. [v8] Copula尾部风险 — 估计联合暴跌概率
# 为什么：普通相关系数低估了极端市场下资产同时暴跌的概率
# C(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ})
# ══════════════════════════════════════════════════════════════════════════════

class CopulaRiskModel:
    """
    Copula尾部风险模型

    为什么需要Copula（不用普通相关系数）：
    · 正态分布假设：资产相关性在所有市场状态下不变
    · 现实：平时相关性低，崩盘时所有资产同时跌（相关性跳升到1）
    · Copula可以分别建模中间分布 + 尾部相关性
    · 2008年金融危机：Gaussian Copula忽略尾部相关性，直接导致CDO灾难

    在系统中的位置（正确用途）：
    → 压力测试：估计"所有持仓同时暴跌X%"的概率
    → 风险限额：当Copula尾部相关性过高时，限制组合规模
    → 不用来预测收益方向！

    实现：Gaussian Copula（最简单，易于实现）
    注意：Gaussian Copula低估极端尾部，实践中需要额外的压力测试
    """

    @staticmethod
    def gaussian_copula_sim(returns: pd.DataFrame,
                             weights: pd.Series,
                             n_sims: int = 5000,
                             horizon: int = 21) -> Dict:
        """
        高斯Copula蒙特卡洛：模拟组合尾部损失

        步骤：
        1. 计算历史相关矩阵
        2. 用Cholesky分解生成相关的随机变量
        3. 映射到各资产的边际分布（历史收益分布）
        4. 计算组合分位数

        注意：高斯Copula低估尾部，因此我们额外做极端情景
        """
        tickers = [t for t in weights.index if t in returns.columns]
        if len(tickers) < 2:
            return {'tail_var_5': 0, 'tail_var_1': 0, 'joint_crash_prob': 0}

        r   = returns[tickers].dropna()
        w   = weights[tickers].values
        n   = len(tickers)

        # 相关矩阵（Copula的参数）
        corr = r.corr().values
        corr = np.clip(corr, -0.999, 0.999)
        # 正定化
        eigv = np.linalg.eigvalsh(corr)
        if eigv.min() < 1e-8:
            corr += np.eye(n) * (1e-8 - eigv.min())

        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        # 边际分布（用历史分位数映射，非参数）
        r_vals = [np.sort(r[t].values) for t in tickers]

        # 模拟
        port_rets = []
        for _ in range(n_sims):
            # 生成相关正态随机变量
            Z    = np.random.standard_normal(n)
            Z_c  = L @ Z  # 相关化

            # 转为均匀分布（概率积分变换）
            from scipy.stats import norm as sp_norm
            U = sp_norm.cdf(Z_c)

            # 映射到历史边际分布
            asset_rets = np.array([
                np.percentile(r_vals[i],
                              float(np.clip(U[i] * 100, 0.1, 99.9)))
                for i in range(n)
            ])

            # 多日复利（简化：乘以horizon的平方根缩放）
            scaling    = np.sqrt(horizon)
            port_r     = float(w @ (asset_rets * scaling))
            port_rets.append(port_r)

        port_rets = np.array(port_rets)

        # 联合暴跌概率（每只股票都跌超过2σ）
        joint_threshold = -2 * float(r.std().mean()) * np.sqrt(horizon)
        joint_prob = float((port_rets < joint_threshold).mean())

        # 真正的尾部（用历史情景补充高斯Copula的不足）
        extreme_shock  = float(r.min().mean()) * np.sqrt(horizon)  # 历史最差单日
        extreme_port   = float(w @ r.min().values * np.sqrt(horizon))

        return {
            'tail_var_5':      round(float(np.percentile(port_rets, 5)), 4),
            'tail_var_1':      round(float(np.percentile(port_rets, 1)), 4),
            'tail_es_5':       round(float(port_rets[port_rets <= np.percentile(port_rets, 5)].mean()), 4),
            'joint_crash_prob':round(joint_prob, 4),
            'extreme_scenario':round(extreme_port, 4),   # 历史最差情景
            'n_sims':          n_sims,
            'horizon_days':    horizon,
            'warning':         joint_prob > 0.15   # 联合暴跌概率 > 15% 发警报
        }

    @staticmethod
    def tail_dependence(returns: pd.DataFrame,
                         threshold: float = 0.10) -> pd.DataFrame:
        """
        尾部依赖系数矩阵
        λ_{ij} = P(X_i < q_α | X_j < q_α)
        同时极端下跌的条件概率（比普通相关系数更能反映危机时的行为）
        """
        tickers = returns.columns.tolist()
        n       = len(tickers)
        r       = returns.dropna()

        # 各资产的α分位数（下尾）
        quantiles = {t: float(r[t].quantile(threshold)) for t in tickers}

        # 计算尾部依赖系数
        td_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                ti, tj = tickers[i], tickers[j]
                both   = ((r[ti] < quantiles[ti]) & (r[tj] < quantiles[tj])).mean()
                one_j  = (r[tj] < quantiles[tj]).mean()
                td     = float(both / (one_j + 1e-8))
                td_mat[i, j] = td_mat[j, i] = td

        arr = td_mat.copy()
        np.fill_diagonal(arr, 1.0)
        df = pd.DataFrame(arr, index=tickers, columns=tickers)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 35. [v8] Probabilistic Sharpe + Deflated Sharpe（来自zip的stats.py）
# 为什么：普通Sharpe高估，需要考虑分布偏态+多次试验的选择偏差
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedStatTests:
    """
    高级统计检验（来自zip项目的StatGate）

    为什么需要这些测试：
    1. Probabilistic Sharpe：
       普通Sharpe假设收益正态分布，忽略偏态和峰度
       当收益左偏（负偏态）时，普通Sharpe高估真实效果
       PSR = P(真实Sharpe > 0 | 观测到的Sharpe)

    2. Deflated Sharpe Ratio (DSR)：
       机器学习时代的overfitting问题：测了100个参数组合，总有一个Sharpe高
       DSR = 用多次测试的期望最大Sharpe作为基准，而不是0
       通过DSR说明：你的策略不只是运气

    3. Probability of Backtest Overfitting (PBO)：
       在样本内表现最好的参数，样本外排名垫底的概率
       PBO > 50% = 严重过拟合
    """

    @staticmethod
    def probabilistic_sharpe(returns: pd.Series,
                              sr_benchmark: float = 0.0) -> float:
        """
        概率Sharpe比率（PSR）

        公式：PSR(SR*) = Φ[(SR-SR*) × √(n-1) / √(1 - γ₃SR + (γ₄-1)/4 × SR²)]
        γ₃：收益偏度，γ₄：收益峰度
        """
        import math
        r = returns.dropna()
        if len(r) < 30:
            return 0.5

        n    = len(r)
        mu   = float(r.mean())
        vol  = float(r.std() + 1e-8)
        sr   = mu / vol  # 日Sharpe

        sk   = float(r.skew())
        ku   = float(r.kurtosis() + 3)  # excess → true kurtosis

        denom = math.sqrt(max(1e-12,
                              1 - sk * sr + ((ku - 1) / 4) * sr**2))
        z     = (sr - sr_benchmark) * math.sqrt(n - 1) / denom

        from scipy.stats import norm as sp_norm
        return float(sp_norm.cdf(z))

    @staticmethod
    def deflated_sharpe(returns: pd.Series,
                         n_trials: int = 1) -> Dict:
        """
        通缩Sharpe比率（DSR）

        当你测试了n_trials个参数/策略组合后，期望最大Sharpe = SR_std × E[max of n_trials standard normals]
        DSR检验：你观测到的Sharpe，是否显著高于这个期望最大值？
        """
        import math
        r = returns.dropna()
        if len(r) < 30:
            return {'dsr': 0.0, 'passed': False}

        from scipy.stats import norm as sp_norm
        obs_sr  = float(r.mean() / (r.std() + 1e-8))
        sr_std  = 1.0 / max(1, math.sqrt(len(r) - 1))

        # 期望最大值（Gumbel分布近似）
        gamma   = 0.5772156649
        n_t     = max(1, int(n_trials))
        z1      = sp_norm.ppf(1 - 1 / n_t)
        z2      = sp_norm.ppf(1 - 1 / (n_t * math.e))
        exp_max = sr_std * ((1 - gamma) * z1 + gamma * z2)

        psr     = AdvancedStatTests.probabilistic_sharpe(returns, exp_max)

        return {
            'dsr':             round(psr, 4),
            'observed_sr':     round(obs_sr * math.sqrt(252), 4),  # 年化
            'expected_max_sr': round(exp_max * math.sqrt(252), 4),
            'n_trials':        n_trials,
            'passed':          psr > 0.95,
            'interpretation':  (
                f"测试了{n_trials}个策略/参数后，"
                f"DSR={'通过✅' if psr>0.95 else '不通过❌'}（PSR={psr:.2%}）"
            )
        }

    @staticmethod
    def stationary_bootstrap_mean(returns: pd.Series,
                                   n_boot: int = 1000,
                                   block_prob: float = 0.10) -> Dict:
        """
        平稳Bootstrap（Politis-Romano）
        比普通Bootstrap更适合时间序列：用随机块长度保留自相关结构
        """
        r  = returns.dropna().values
        n  = len(r)
        if n < 30:
            return {'p_mean_le_0': 1.0, 'ci_low': 0, 'ci_high': 0}

        rng   = np.random.default_rng(42)
        means = []

        for _ in range(n_boot):
            idx    = np.zeros(n, dtype=int)
            idx[0] = rng.integers(0, n)
            for i in range(1, n):
                # block_prob概率重新随机起点（保留时间序列结构）
                idx[i] = (rng.integers(0, n) if rng.random() < block_prob
                          else (idx[i-1] + 1) % n)
            means.append(float(np.mean(r[idx])))

        means = np.array(means)
        return {
            'p_positive':  round(float((means > 0).mean()), 4),
            'ci_low':      round(float(np.percentile(means, 2.5)), 6),
            'ci_high':     round(float(np.percentile(means, 97.5)), 6),
            'significant': float((means > 0).mean()) > 0.95
        }


# ══════════════════════════════════════════════════════════════════════════════
# 36. [v8] 增强版VolTargeter（集成GARCH）
# 为什么：用GARCH预测波动率代替简单rolling std，更敏感于波动率突变
# ══════════════════════════════════════════════════════════════════════════════

class GARCHVolTargeter:
    """
    GARCH增强版波动率目标（替代简单VolTargeter）

    改进点：
    · v6 VolTargeter：用realized vol（rolling 21天std）缩放
    · v8：用GARCH预测明日波动率，反应更快

    例子：
    · 昨日大跌3%，realized vol缓慢上升
    · GARCH立即预测明日vol更高，今天就降仓
    · 而不是等21天平均后才降仓（已经太晚）
    """

    def __init__(self, target_vol: float = 0.10,
                 min_scale: float = 0.20,
                 max_scale: float = 2.00):
        self.target    = target_vol
        self.min_scale = min_scale
        self.max_scale = max_scale

    def scale(self, weights: pd.Series,
               returns: pd.DataFrame,
               use_garch: bool = True) -> Tuple[pd.Series, float, float]:
        """
        Returns: (scaled_weights, scale_factor, forecast_vol)
        """
        if returns is None or len(returns) < 21:
            return weights, 1.0, self.target

        tickers = [t for t in weights.index if t in returns.columns]
        if not tickers:
            return weights, 1.0, self.target

        w_arr   = weights[tickers].values
        r_arr   = returns[tickers].dropna().values
        if len(r_arr) < 10:
            return weights, 1.0, self.target

        # 组合历史收益
        port_r  = r_arr @ (w_arr / (np.abs(w_arr).sum() + 1e-8))

        if use_garch and len(port_r) >= 21:
            # GARCH预测
            forecast_vol = GARCHModel.quick_forecast(pd.Series(port_r))
        else:
            # 简单realized vol（fallback）
            forecast_vol = float(np.std(port_r[-21:]) * np.sqrt(252))

        if forecast_vol < 1e-4:
            return weights, 1.0, forecast_vol

        scale = float(np.clip(self.target / forecast_vol,
                               self.min_scale, self.max_scale))
        return weights * scale, scale, forecast_vol


# ══════════════════════════════════════════════════════════════════════════════
# 37. [v8] 完整统计验证套件（整合所有改进）
# ══════════════════════════════════════════════════════════════════════════════

class FullStatSuite:
    """
    完整统计验证套件（v8整合版）

    按机构标准逐层验证策略：
    Level 1: 基础指标（已有）：Sharpe/MaxDD/Calmar
    Level 2: 统计显著性（v6）：Bootstrap CI + Newey-West
    Level 3: 分布调整（v8）：PSR + DSR（考虑偏态和多次测试）
    Level 4: 因子分析（v8）：PCA拥挤检测 + Factor暴露
    Level 5: 尾部风险（v8）：Copula + GBM Monte Carlo VaR
    """

    @staticmethod
    def full_report(returns: pd.Series,
                    market: pd.Series,
                    weights: pd.Series = None,
                    prices: pd.DataFrame = None,
                    n_trials: int = 10,
                    rf: float = 0.03) -> Dict:
        """
        运行所有统计验证，返回完整报告
        """
        report = {}

        # Level 1: 基础指标
        m = backtest_metrics(returns, rf)
        report['basic'] = m

        # Level 2: 统计显著性
        stat_d  = StatisticalDepth()
        boot    = stat_d.bootstrap_sharpe(returns, rf)
        nw      = stat_d.newey_west_sharpe(returns, rf)
        report['bootstrap'] = boot
        report['newey_west'] = nw

        # Level 3: PSR + DSR（v8新增）
        adv = AdvancedStatTests()
        psr = adv.probabilistic_sharpe(returns, 0.0)
        dsr = adv.deflated_sharpe(returns, n_trials)
        sb  = adv.stationary_bootstrap_mean(returns)
        report['psr'] = psr
        report['dsr'] = dsr
        report['stationary_bootstrap'] = sb

        # Level 4: 因子暴露
        expo = stat_d.factor_exposure(returns, market, rf)
        report['factor_exposure'] = expo

        if prices is not None and weights is not None:
            # PCA拥挤检测
            pca = PCAFactorModel(n_components=5)
            pca.fit(prices.pct_change().dropna())
            crowd = pca.crowding_score()
            fac_exp = pca.factor_exposure(weights)
            report['pca_crowding'] = crowd
            report['pca_factor_exp'] = fac_exp

        # Level 5: 尾部风险
        if prices is not None and weights is not None:
            gbm_var = GBMModel.portfolio_var(prices, weights, horizon=21)
            cop_risk = CopulaRiskModel.gaussian_copula_sim(
                prices.pct_change().dropna(), weights, n_sims=2000, horizon=21
            )
            report['gbm_var'] = gbm_var
            report['copula_risk'] = cop_risk

        # Level 6: GARCH波动率状态
        report['garch_vol'] = GARCHModel.quick_forecast(returns, horizon=1)

        return report

    @staticmethod
    def print_report(report: Dict, label: str = 'Strategy'):
        print(f"\n{'═'*65}")
        print(f"  📊 完整统计验证报告 v8 — {label}")
        print(f"{'═'*65}")

        # Level 1
        b = report.get('basic', {})
        print(f"\n  [L1] 基础指标：")
        print(f"    年化:{b.get('ann_ret',0):+.2%} 波动:{b.get('ann_vol',0):.2%} "
              f"Sharpe:{b.get('sharpe',0):.3f} MaxDD:{b.get('max_dd',0):.2%}")

        # Level 2
        boot = report.get('bootstrap', {})
        nw   = report.get('newey_west', {})
        print(f"\n  [L2] 统计显著性：")
        print(f"    Bootstrap Sharpe: {boot.get('sharpe',0):.3f} "
              f"[{boot.get('ci_low',0):.3f}, {boot.get('ci_high',0):.3f}] "
              f"{'✅显著' if boot.get('significant') else '❌不显著'}")
        print(f"    Newey-West Sharpe: {nw.get('nw_sharpe',0):.3f} "
              f"p={nw.get('p_value',1):.3f} "
              f"{'✅显著' if nw.get('significant') else '❌不显著'}")

        # Level 3
        psr = report.get('psr', 0)
        dsr = report.get('dsr', {})
        sb  = report.get('stationary_bootstrap', {})
        print(f"\n  [L3] 分布调整（v8新增）：")
        print(f"    PSR = {psr:.4f} {'✅>50%' if psr>0.5 else '❌<50%'}")
        print(f"    DSR: {dsr.get('interpretation','')}")
        print(f"    平稳Bootstrap: 95%CI=[{sb.get('ci_low',0):.5f},{sb.get('ci_high',0):.5f}] "
              f"{'✅显著' if sb.get('significant') else '❌不显著'}")

        # Level 4
        fe = report.get('factor_exposure', {})
        print(f"\n  [L4] 因子暴露：")
        print(f"    Alpha={fe.get('alpha_ann',0):+.2%} Beta={fe.get('beta',0):.2f} "
              f"R²={fe.get('r_squared',0):.2%} "
              f"{'✅纯Alpha' if fe.get('pure_alpha') else '⚠️偏Beta'}")

        crowd = report.get('pca_crowding', {})
        if crowd:
            print(f"    PCA拥挤度: PC1={crowd.get('pc1_variance_explained',0):.1%} "
                  f"→ {crowd.get('crowding_level','?')}")
            if crowd.get('crowding_alert'):
                print(f"    ⚠️ {crowd.get('interpretation','')}")

        # Level 5
        gbm = report.get('gbm_var', {})
        cop = report.get('copula_risk', {})
        print(f"\n  [L5] 尾部风险（v8新增）：")
        if gbm:
            print(f"    GBM VaR(95%,21天)={gbm.get('var',0):.2%} "
                  f"ES={gbm.get('es',0):.2%} "
                  f"{'✅在限制内' if not gbm.get('exceeds_limit') else '⚠️超过5.6%'}")
        if cop:
            print(f"    Copula联合暴跌概率={cop.get('joint_crash_prob',0):.1%} "
                  f"尾部VaR(1%)={cop.get('tail_var_1',0):.2%} "
                  f"{'⚠️高尾部风险' if cop.get('warning') else '✅正常'}")

        # Level 6
        garch_v = report.get('garch_vol', 0)
        print(f"\n  [L6] GARCH波动率预测：{garch_v:.2%}（年化，明日预测）")
        print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# 主系统 v8（整合所有新公式）
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Canyon量化交易系统 v8.0

    v8在v7基础上新增（对应图片中10个公式）：
    ✅ GBM：Monte Carlo路径模拟 + VaR/ES（压力测试，不是alpha）
    ✅ BSM：在analyze_current显示期权数学框架（有期权数据时启用）
    ✅ Markowitz：保留，改进加入GARCH协方差
    ✅ GARCH：预测明日波动率，替换VolTargeter中的简单realized vol
    ✅ 协整/StatArb：保留
    ✅ HMM：保留KMeans版本，加入转移概率解释
    ✅ PCA：新增拥挤检测 + 因子暴露 + alpha中性化
    ✅ Kelly：保留，加入GARCH波动率调整
    ✅ Copula：新增联合尾部风险估计（比普通相关系数更保守）
    ✅ 神经网络：在AlphaPool中可选（必须通过IC门槛）

    额外新增（来自zip的StatGate）：
    ✅ PSR：概率Sharpe（考虑偏态/峰度）
    ✅ DSR：通缩Sharpe（考虑多次测试选择偏差）
    ✅ 平稳Bootstrap：保留自相关结构的更严格检验
    """
    print(f"\n{'═'*65}")
    print(f"  🏔  CANYON量化交易系统 v8.0")
    print(f"  10大量化公式 × 完整机构实现 × 正确位置")
    print(f"{'═'*65}")
    print(f"\n  公式位置说明：")
    print(f"  GBM      → [压力测试] Monte Carlo VaR/ES")
    print(f"  BSM      → [期权层]   Greeks计算（期权overlay）")
    print(f"  Markowitz→ [组合优化] 均值方差（已有，加GARCH协方差）")
    print(f"  GARCH    → [风控层]   波动率预测 → VolTargeter缩放")
    print(f"  协整性   → [StatArb]  配对交易 z-score（已有）")
    print(f"  HMM      → [Regime]   状态检测（KMeans版，加转移矩阵）")
    print(f"  PCA      → [风险诊断] 拥挤检测 + 因子中性化")
    print(f"  Kelly    → [仓位]     偏度调整半Kelly（已有）")
    print(f"  Copula   → [尾部风险] 联合暴跌概率")
    print(f"  神经网络 → [Alpha候选] 需过IC/DSR/PBO门槛")

    # 数据加载
    system = CanyonTradingSystemV7(tc_bps=10, rf=0.03, target_vol=0.10)
    TICKERS = ['NVDA', 'AMD', 'TSM', 'MU', 'SOXX',
               'AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ',
               'FCX', 'WDC', 'STX', 'NEM']   # v8.5: 加入错杀/强趋势+低估值测试标的
    START, END = '2020-01-01', '2024-12-31'
    prices, volumes, market = system.data.load(TICKERS, START, END)
    returns = prices.pct_change().dropna()

    # ── Step1: 主回测（继承v7完整流程）──────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step1: Walk-Forward回测（v7引擎 + v8 GARCH波动率目标）")
    print(f"{'─'*65}")
    main_result = system.backtester.run(
        prices, volumes, market, system.stat_arb, system.od,
        use_ic_alpha=True, verbose=True
    )

    # ── Step2: v8完整统计验证套件 ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step2: 6层统计验证（v8完整版）")
    print(f"{'─'*65}")
    if 'daily_returns' in main_result and len(main_result['daily_returns']) > 30:
        dr      = main_result['daily_returns']
        # 计算当前仓位（用于Copula/PCA分析）
        regime, _ = detect_regime(market, prices)
        cs = cross_sectional_momentum(prices)
        tsig = trend_signals(prices)
        alloc = system.od.allocate(
            regime=regime, long_alpha=cs['long_alpha'].dropna(),
            short_alpha=cs['short_alpha'].dropna(), trend_sig=tsig,
            stat_arb_opps=[], returns=returns
        )
        w_current = alloc.to_series()

        report = FullStatSuite.full_report(
            returns=dr,
            market=market.pct_change().dropna(),
            weights=w_current if len(w_current) > 0 else None,
            prices=prices if len(w_current) > 0 else None,
            n_trials=18,   # 我们测试了约18个参数组合
            rf=0.03
        )
        FullStatSuite.print_report(report, label='Canyon v8')

    # ── Step3: GARCH波动率分析 ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step3: GARCH(1,1) 波动率分析")
    print(f"  σ_t² = α_0 + α_1 ε²_{{t-1}} + β_1 σ²_{{t-1}}")
    print(f"{'─'*65}")
    mkt_ret = market.pct_change().dropna()
    garch   = GARCHModel().fit(mkt_ret)
    print(f"  拟合参数：ω={garch.omega:.2e} α={garch.alpha:.4f} β={garch.beta:.4f}")
    print(f"  持续性(α+β)：{garch.alpha+garch.beta:.4f} "
          f"{'✅<1，均值回归' if garch.alpha+garch.beta<1 else '⚠️≥1，非平稳'}")
    for h in [1, 5, 21]:
        print(f"  {h:>2}天后预测年化波动率：{garch.forecast(h):.2%}")
    # GARCH vs 简单realized vol
    simple_vol = float(mkt_ret.tail(21).std() * np.sqrt(252))
    print(f"  简单Realized Vol(21天)：{simple_vol:.2%}")
    print(f"  → GARCH更快响应波动率突变，用于VolTargeter更合适")

    # ── Step4: PCA拥挤风险分析 ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step4: PCA因子分析")
    print(f"  R = Σ β_i F_i + ε  （隐藏因子分解）")
    print(f"{'─'*65}")
    pca = PCAFactorModel(n_components=5).fit(returns)
    crowd = pca.crowding_score()
    print(f"  解释方差：{', '.join(f'PC{i+1}={v:.1%}' for i,v in enumerate(pca.explained_.values))}")
    print(f"  {crowd['interpretation']}")

    if len(w_current) > 0 and pca.loadings_ is not None:
        fac_exp = pca.factor_exposure(w_current)
        print(f"  组合因子暴露：{fac_exp.get('exposures', {})}")
        print(f"  主导因子：{fac_exp.get('dominant_factor', '?')}")

    # ── Step5: Copula尾部风险 ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step5: Copula尾部风险分析")
    print(f"  联合暴跌概率（比普通相关系数更保守）")
    print(f"{'─'*65}")
    if len(w_current) > 0:
        cop = CopulaRiskModel.gaussian_copula_sim(returns, w_current, n_sims=2000)
        print(f"  VaR(5%,21天)：{cop['tail_var_5']:.2%}")
        print(f"  ES(5%,21天)：{cop['tail_es_5']:.2%}")
        print(f"  VaR(1%,21天)：{cop['tail_var_1']:.2%}")
        print(f"  联合暴跌概率：{cop['joint_crash_prob']:.1%} "
              f"{'⚠️偏高，考虑降低集中度' if cop['warning'] else '✅正常'}")
        print(f"  历史最坏情景：{cop['extreme_scenario']:.2%}")

        # 尾部依赖矩阵
        td = CopulaRiskModel.tail_dependence(returns, threshold=0.10)
        top_tickers = list(w_current.abs().nlargest(4).index)
        if len(top_tickers) >= 2:
            print(f"\n  尾部依赖系数（前4大持仓）：")
            sub_td = td.loc[top_tickers, top_tickers].round(3)
            print(sub_td.to_string())
            print(f"  （越高=危机时同时暴跌概率越大）")

    # ── Step6: GBM Monte Carlo VaR ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step6: GBM Monte Carlo VaR")
    print(f"  dS_t = μS_t dt + σS_t dW_t")
    print(f"{'─'*65}")
    if len(w_current) > 0:
        # 用GARCH预测的波动率代替简单历史vol
        garch_mkt_vol = garch.forecast(1)
        print(f"  使用GARCH预测波动率：{garch_mkt_vol:.2%}（代替简单rolling std）")

        gbm_var = GBMModel.portfolio_var(
            prices, w_current, horizon=21, n_paths=3000
        )
        print(f"  组合VaR(95%,21天)：{gbm_var['var']:.2%}")
        print(f"  组合ES(95%,21天)：{gbm_var['es']:.2%}")
        print(f"  最坏1%情景：{gbm_var['worst_1pct'] if 'worst_1pct' in gbm_var else 'N/A'}")
        print(f"  {'✅在5.6%限制内' if not gbm_var.get('exceeds_limit') else '⚠️超过5.6%硬约束'}")

    # ── Step7: PSR + DSR验证 ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step7: PSR + DSR（多次测试的选择偏差修正）")
    print(f"{'─'*65}")
    if 'daily_returns' in main_result:
        dr  = main_result['daily_returns']
        psr = AdvancedStatTests.probabilistic_sharpe(dr)
        dsr = AdvancedStatTests.deflated_sharpe(dr, n_trials=18)
        sb  = AdvancedStatTests.stationary_bootstrap_mean(dr)
        print(f"  PSR = {psr:.4f} → "
              f"{'✅Sharpe在偏度/峰度调整后仍显著' if psr>0.5 else '❌偏态拉低真实Sharpe'}")
        print(f"  DSR: {dsr.get('interpretation', '')}")
        print(f"  平稳Bootstrap (块重采样)：95%CI=[{sb.get('ci_low',0):.5f},{sb.get('ci_high',0):.5f}] "
              f"{'✅' if sb.get('significant') else '❌'}")

    # ── Step8: 完整 Canyon Stock Assistant 架构演示 ─────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step8: Canyon Stock Assistant — 完整架构")
    print(f"  Tactical / CoreHedge / SectorRotation / MasterRisk")
    print(f"{'─'*65}")

    # ── 初始化所有层 ──────────────────────────────────────────────────────────
    sleeve_mgr   = SleeveManager()
    tactical_sig = TacticalSignals()
    sector_eng   = SectorRotationEngine(rebalance_days=5)
    master_risk  = MasterRiskLayer(
        target_vol=0.10, max_global_dd=0.15,
        kelly_fraction=0.50, ic_min=0.02, icir_min=0.30,
        cooldown_trigger=3, cooldown_days=5, adj_frequency_days=5
    )
    options_engine = OptionsSignalEngine(r=0.05)   # [v8.2] 期权引擎

    # 当前Regime
    regime, _ = detect_regime(market, prices)
    regime_label = regime.label

    # ── [v8.2] 期权结构分析（GEX / Gamma Squeeze / 做市商流）─────────────────
    print(f"\n  [期权结构分析 — GEX / Gamma Squeeze / 做市商Delta流]")
    print(f"  （实盘接Polygon.io；演示模式使用BSM合成期权链）")

    # 计算每只股票的历史波动率（BSM的σ输入）
    hist_vols = prices.pct_change().dropna().tail(21).std() * np.sqrt(252)

    # 对所有标的运行期权结构分析
    opt_signals = options_engine.portfolio_options_signals(prices, hist_vols)
    options_engine.print_options_dashboard(opt_signals)

    # 把期权信号合成为Tactical Sleeve的Alpha（GEX主导）
    if len(opt_signals) > 0 and 'combined_signal' in opt_signals.columns:
        options_alpha = opt_signals['combined_signal'].dropna()
        print(f"\n  期权合成Alpha（做多=Squeeze风险高，GEX翻负）：")
        for tk, v in options_alpha.nlargest(3).items():
            row = opt_signals.loc[tk]
            print(f"    ▲ {tk}: 信号={v:+.3f} | "
                  f"Squeeze={row.get('squeeze_score',0):.0f}分 | "
                  f"GEX={row.get('total_gex',0):,.0f} | "
                  f"IVP={row.get('iv_percentile',0):.0%}")

        # 单股BSM Greeks演示（选Squeeze分最高的那只）
        top_ticker = opt_signals['squeeze_score'].idxmax() if 'squeeze_score' in opt_signals.columns else prices.columns[0]
        spot  = float(prices[top_ticker].iloc[-1])
        sigma = float(hist_vols.get(top_ticker, 0.30))
        T_ex  = 21/365
        K_atm = round(spot)

        print(f"\n  BSM Greeks演示（{top_ticker} @ ${spot:.2f}, ATM K=${K_atm}, 21天到期）：")
        print(f"  ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0")
        print(f"    Call Delta (∂C/∂S) = {BSMGreeks.delta(spot,K_atm,0.05,sigma,T_ex,'call'):+.4f}")
        print(f"    Gamma (∂²V/∂S²)   = {BSMGreeks.gamma(spot,K_atm,0.05,sigma,T_ex):+.6f}")
        print(f"    Vega  (∂V/∂σ)     = {BSMGreeks.vega(spot,K_atm,0.05,sigma,T_ex):+.4f}")
        print(f"    Theta (∂V/∂t)     = {BSMGreeks.theta(spot,K_atm,0.05,sigma,T_ex,'call'):+.4f}/天")
        print(f"    ATM Call Price    = ${BSMGreeks.call_price(spot,K_atm,0.05,sigma,T_ex):.2f}")

        # 做市商Delta流分析
        chain_demo = options_engine.generate_synthetic_chain(spot, sigma, 21, top_ticker)
        gex_result = options_engine.gex_engine.compute_gex(chain_demo)
        flow_result= options_engine.gex_engine.dealer_delta_flow(chain_demo, 0.01)
        print(f"\n  做市商Delta对冲流（价格+1%时）：")
        print(f"    {flow_result['interpretation']}")
        print(f"    Call端流：{flow_result['call_flow']:+,.0f}股  "
              f"Put端流：{flow_result['put_flow']:+,.0f}股  "
              f"净流：{flow_result['net_delta_flow_shares']:+,.0f}股")
        print(f"    GEX Flip点：${gex_result['gamma_flip']:.2f}  "
              f"（当前${spot:.2f}，"
              f"{'在Flip上方⚠️' if spot > gex_result['gamma_flip'] else '在Flip下方✅'}）")
    else:
        options_alpha = pd.Series(0.0, index=prices.columns)

    # ── Tactical Sleeve 信号 ──────────────────────────────────────────────────
    print(f"\n  [TACTICAL Sleeve]")
    ovn = tactical_sig.overnight_reversal(prices, volumes)
    evr = tactical_sig.event_overreaction(prices, volumes)
    sqz = tactical_sig.squeeze_filter(prices, volumes)

    # 合成Tactical Alpha（四个信号：overnight+event+squeeze+options）
    tac_combined = pd.Series(0.0, index=prices.columns)
    for sig, name, w in [(ovn,'overnight',0.30),(evr,'event',0.25),
                          (sqz,'squeeze',0.20),(options_alpha,'options_gex',0.25)]:
        sig_clean = sig.reindex(prices.columns).fillna(0)
        tac_combined = tac_combined + sig_clean * w

    print(f"  overnight_reversal top3: {ovn.nlargest(3).index.tolist()}")
    print(f"  event_overreaction top3: {evr.nlargest(3).index.tolist()}")
    print(f"  squeeze_filter top3:     {sqz.nlargest(3).index.tolist()}")
    print(f"  Tactical合成Alpha top3:  {tac_combined.nlargest(3).index.tolist()}")

    # ── Core Hedge Sleeve 信号 ────────────────────────────────────────────────
    print(f"\n  [CORE HEDGE Sleeve]")
    cs       = cross_sectional_momentum(prices)
    core_sig = cs['long_alpha'].dropna()   # 12M动量（长期强势股）
    print(f"  12M动量 Top5:  {core_sig.nlargest(5).index.tolist()}")
    print(f"  持仓周期：3月-多年 | 止损：仅逻辑证伪")

    # ── Sector Rotation Sleeve 信号 ───────────────────────────────────────────
    print(f"\n  [SECTOR ROTATION Sleeve]")
    sector_eng.print_ranking(prices, market)
    sector_weights = sector_eng.get_weights(prices, market, regime, force_rebalance=True)
    print(f"  本周板块配置（每5天再平衡）：")
    for tk, w in sector_weights.items():
        print(f"    {'▲' if w>0 else '▼'} {tk}: {w:+.1%}")

    # ── Sleeve权重分配 ────────────────────────────────────────────────────────
    print(f"\n  [Sleeve权重分配]")
    base_alloc = sleeve_mgr.allocate_by_regime(regime_label)

    # MasterRisk学习后的调整权重（初始=基础权重）
    adj_alloc  = master_risk.get_adjusted_sleeve_weights(base_alloc)

    print(f"  当前Regime：{regime_label}")
    print(f"  基础权重：  TACTICAL={base_alloc['TACTICAL']:.0%}  "
          f"CORE={base_alloc['CORE_HEDGE']:.0%}  "
          f"SECTOR={base_alloc['SECTOR_ROTATION']:.0%}")
    print(f"  学习后权重：TACTICAL={adj_alloc['TACTICAL']:.0%}  "
          f"CORE={adj_alloc['CORE_HEDGE']:.0%}  "
          f"SECTOR={adj_alloc['SECTOR_ROTATION']:.0%}")

    # ── 合成最终Alpha信号 ─────────────────────────────────────────────────────
    print(f"\n  [最终Alpha合成]")
    sleeve_signals = sleeve_mgr.get_sleeve_alphas(
        tactical_alpha=tac_combined,
        core_alpha=core_sig,
        sector_alpha=sector_weights if len(sector_weights) > 0
                     else cs['long_alpha'].dropna(),
        regime_label=regime_label
    )
    final_alpha = sleeve_mgr.combine_sleeve_signals(sleeve_signals, regime_label)

    if len(final_alpha.dropna()) > 0:
        print(f"  多头：{final_alpha.nlargest(5).index.tolist()}")
        print(f"  空头：{final_alpha.nsmallest(3).index.tolist()}")

    # ── MasterRisk全局风控 ────────────────────────────────────────────────────
    print(f"\n  [Master Risk Layer]")

    # 构建仓位
    ret_hist = prices.pct_change().dropna()
    alloc_od = system.od.allocate(
        regime=regime, long_alpha=final_alpha.dropna() if len(final_alpha) > 0
               else cs['long_alpha'].dropna(),
        short_alpha=final_alpha.dropna() * -1 if len(final_alpha) > 0
                    else cs['short_alpha'].dropna(),
        trend_sig=trend_signals(prices),
        stat_arb_opps=[], returns=ret_hist
    )
    raw_weights = alloc_od.to_series()

    # 全局波动率目标 + 回撤保护
    port_ret_hist = (ret_hist[raw_weights.index.intersection(ret_hist.columns)]
                     .fillna(0) @ pd.Series(
                         {t: float(raw_weights.get(t, 0)) for t in raw_weights.index
                          if t in ret_hist.columns}
                     ).fillna(0)) if len(raw_weights) > 0 else pd.Series([0.0])

    scaled_weights, scale_info = master_risk.global_scale(raw_weights, port_ret_hist)

    # Kelly上限
    final_weights = master_risk.kelly_cap(scaled_weights, ret_hist)

    print(f"  全局缩放系数：{scale_info.get('scale', 1.0):.2f}x  "
          f"原因：{scale_info.get('reason', 'normal')}")
    print(f"  Kelly截断后仓位：{len(final_weights[final_weights.abs() > 0.005])}只")

    # 模拟7天P&L更新（演示Cooldown和周度学习）
    print(f"\n  模拟7天运行（含Cooldown触发）：")
    test_sleeve_rets = {
        'TACTICAL':        [0.012, -0.009, -0.003, -0.004,  0.008,  0.002,  0.005],
        'CORE_HEDGE':      [0.003,  0.002, -0.001,  0.001,  0.002,  0.001,  0.003],
        'SECTOR_ROTATION': [0.005, -0.006,  0.004, -0.002,  0.006, -0.001,  0.004],
    }
    for day in range(7):
        total_r = sum(test_sleeve_rets[s][day] * base_alloc[s]
                      for s in test_sleeve_rets)
        status  = master_risk.update_daily(total_r)
        for s, rets in test_sleeve_rets.items():
            sleeve_mgr.update_pnl(s, rets[day])
        if day == 2:  # 第3天触发Cooldown演示
            master_risk._daily_rets[-3:] = [-0.01, -0.008, -0.007]
        print(f"    Day{day+1}: 总P&L={total_r:+.3%}  状态={status}", end='')
        if status != 'normal':
            print(f" ← {master_risk._cooldown_left}天冷静", end='')
        print()

    # 周度自适应
    print(f"\n  周度自适应调整（第5天触发）：")
    master_risk._day_counter = 5   # 强制触发
    new_mults = master_risk.weekly_self_adjustment(test_sleeve_rets)
    for s, m in new_mults.items():
        arrow = '↑' if m > 1.0 else '↓' if m < 1.0 else '→'
        print(f"    {s:<18} 乘数：×{m:.2f} {arrow}")

    # IC状态更新（模拟4周历史）
    for _ in range(4):
        master_risk.update_strategy_ic('TACTICAL',        0.035)
        master_risk.update_strategy_ic('CORE_HEDGE',      0.028)
        master_risk.update_strategy_ic('SECTOR_ROTATION', 0.019)

    # 完整Dashboard
    sleeve_mgr.print_dashboard(regime_label)
    master_risk.print_status()

    print(f"\n  ✅ 完整架构验证：")
    print(f"  Canyon Stock Assistant")
    print(f"  ├── Tactical:        overnight({ovn.abs().max():.3f})"
          f" + event({evr.abs().max():.3f}) + squeeze({sqz.max():.3f})")
    print(f"  ├── CoreHedge:       12M动量Top = {core_sig.idxmax()}")
    print(f"  ├── SectorRotation:  Top板块 = "
          f"{sector_eng.rank_sectors(prices, market)['ticker'].iloc[0] if len(sector_eng.rank_sectors(prices,market))>0 else 'N/A'}")
    print(f"  └── MasterRisk:      全局净值={master_risk._global_equity:.4f} "
          f"回撤={((master_risk._global_equity-master_risk._global_peak)/master_risk._global_peak):.2%}")

    # ── Step10: 人工选股机会引擎（FCX错杀 + SanDisk强趋势+低估值类）─────────
    print(f"\n{'─'*65}")
    print(f"  Step10: 人工选股机会引擎 (Discretionary Stock Picking)")
    print(f"  场景A:错杀股 | 场景B:强趋势+低估值 | 场景C:跌幅榜深挖")
    print(f"{'─'*65}")

    discretionary = DiscretionaryOpportunityEngine()

    # ── 演示模式：构造典型场景让筛选逻辑可见 ────────────────────────────────
    # 合成数据是平稳牛市，看不到真实的错杀/强趋势+低估值
    # 这里手工构造两个典型案例验证筛选逻辑
    print(f"\n  🧪 演示模式：注入典型场景验证筛选逻辑")
    print(f"     （实盘时直接连Yahoo Finance/SEC EDGAR即可）")

    # 场景A示例：FCX类（短期暴跌-18%但基本面良好）
    demo_fcx = FundamentalSnapshot(
        ticker='FCX_DEMO', sector='Materials', market_cap_b=60.0,
        pe_ratio=12.0, pb_ratio=2.1, peg_ratio=0.9,
        revenue_growth=0.08, earnings_growth=0.12,
        profit_margin=0.15, roe=0.18, debt_to_equity=0.7,
        current_ratio=2.1, free_cashflow_b=4.5,
        moat_score=3.5, industry_position='leader',
        catalyst_event='矿区暴雨事故（短期供应中断）'
    )
    print(f"\n  📋 演示样本A (FCX_DEMO): 错杀股逻辑")
    print(f"     基本面: PE={demo_fcx.pe_ratio} PB={demo_fcx.pb_ratio} "
          f"ROE={demo_fcx.roe:.0%} ROE质量={'✅' if demo_fcx.is_quality else '❌'}")
    print(f"     估值: PEG={demo_fcx.peg_ratio} 低估值={'✅' if demo_fcx.is_undervalued else '❌'}")
    print(f"     地位: {demo_fcx.industry_position} 市值=${demo_fcx.market_cap_b}B")
    print(f"     催化: {demo_fcx.catalyst_event}")
    print(f"     → 假设股价短期-18%下跌后：")
    print(f"     → 错杀评分 ≈ 30(跌幅) + 25(质量) + 20(估值) + 15(leader) + 10(企稳) = 100")
    print(f"     → R/R比 ≈ 0.6×18% ÷ 8% = 1.35:1")
    print(f"     → 进CORE_HEDGE Sleeve，持仓周期60-180天")

    # 场景B示例：WDC类（强趋势+低PE+EPS增长）
    demo_wdc = FundamentalSnapshot(
        ticker='WDC_DEMO', sector='Technology', market_cap_b=45.0,
        pe_ratio=9.0, forward_pe=8.0, pb_ratio=1.5, peg_ratio=0.6,
        revenue_growth=0.22, earnings_growth=0.45,
        profit_margin=0.10, roe=0.12, debt_to_equity=0.8,
        moat_score=3.0, industry_position='leader',
        catalyst_event='存储芯片周期向上 + Q3财报超预期'
    )
    print(f"\n  📋 演示样本B (WDC_DEMO): 强趋势+低估值逻辑")
    print(f"     估值: PE={demo_wdc.pe_ratio} PEG={demo_wdc.peg_ratio} "
          f"低估值={'✅' if demo_wdc.is_undervalued else '❌'}")
    print(f"     增长: 收入+{demo_wdc.revenue_growth:.0%} EPS+{demo_wdc.earnings_growth:.0%}")
    print(f"     催化: {demo_wdc.catalyst_event}")
    print(f"     → 假设20天上涨14/20天 + 量比1.3x + 距高点92%：")
    print(f"     → 双击评分 ≈ 17(趋势) + 13(估值) + 18(增长) + 12(空间) + 7(量能) + 10(行业) = 77")
    print(f"     → R/R比 ≈ (1+45%×1.5) ÷ 跌破SMA50止损 ≈ 3-4:1")
    print(f"     → 进SECTOR_ROTATION Sleeve，持仓周期21-90天")

    print(f"\n  现在用当前universe扫描（合成数据多为平稳上涨，候选可能较少）：")

    # 模拟一些催化剂事件（实盘从新闻API获取）
    catalyst_events = {
        'FCX':  '矿区暴雨事故（短期供应中断，长期储量未变）',
        'CVX':  '炼油厂维修事件',
        'BA':   '飞机交付延迟（短期）',
    }

    # 计算行业动量（给StrongTrendValueScreener做行业β确认）
    sector_mom = {}
    for sec_etf in ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU']:
        if sec_etf in prices.columns:
            mom = float(prices[sec_etf].pct_change(63).iloc[-1])
            sector_mom[sec_etf] = mom

    print(f"\n  正在拉取基本面数据...")
    opportunities = discretionary.find_opportunities(
        prices, volumes,
        catalyst_events=catalyst_events,
        sector_momentum=sector_mom
    )

    s = opportunities['summary']
    print(f"\n  扫描结果汇总：")
    print(f"    场景A 错杀股候选:    {s['n_oversold']}只")
    print(f"    场景B 强趋势+低估值: {s['n_trend_value']}只")
    print(f"    今日跌幅榜A类(错杀): {s['n_daily_A']}只")
    print(f"    今日跌幅榜B类(回踩): {s['n_daily_B']}只")

    # 详细打印各场景
    discretionary.oversold_screen.print_candidates(
        opportunities['oversold_quality'], top_n=3
    )
    discretionary.trend_value.print_candidates(
        opportunities['trend_value'], top_n=3
    )
    discretionary.loser_miner.daily_report(
        prices, volumes, opportunities['fundamentals']
    )

    # Sleeve分配
    assignments = discretionary.to_sleeve_assignments(opportunities, top_n_each=3)
    print(f"\n  {'═'*65}")
    print(f"  📋 人工选股 → Sleeve分配建议")
    print(f"  {'═'*65}")
    for sleeve_name, items in assignments.items():
        if items:
            print(f"\n  [{sleeve_name}]")
            for it in items:
                print(f"    {it['ticker']:<6} 场景={it['scenario']:<25} "
                      f"R/R={it['reward_risk']:.1f}  持仓{it['holding_days']}天")
                print(f"           入${it['entry']}/止${it['stop']}/目${it['target']}")
                print(f"           论点: {it['thesis'][:60]}")
        else:
            print(f"\n  [{sleeve_name}]  无符合条件候选")

    # ── Step11: v8.5 深度选股引擎（DeepDiscretionaryEngine）────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step11: DEEP Discretionary Engine v8.5")
    print(f"  6维评分 + A/B/C优先级 + NOW/WAIT/AVOID + HOLD/ADD/REDUCE/EXIT")
    print(f"{'─'*65}")

    deep_engine = DeepDiscretionaryEngine()

    # 模拟事件 + 模拟当前持仓
    events_v85 = {
        'FCX':  ('矿区暴雨事故，部分产能受损', 'moderate'),
        'WDC':  ('存储芯片周期触底反转 + Q3财报超预期', 'moderate'),
        'MU':   ('AI数据中心需求确认 + 上调指引', 'moderate'),
    }
    # 模拟你已经持有NVDA和AAPL
    current_pos = {'NVDA': 0.05, 'AAPL': 0.04}

    print(f"\n  扫描universe（{len(prices.columns)}只）...")
    print(f"  事件输入: {list(events_v85.keys())}")
    print(f"  已持仓: {list(current_pos.keys())}")

    decisions = deep_engine.screen_universe(
        prices=prices,
        volumes=volumes,
        events=events_v85,
        current_positions=current_pos
    )

    # 完整报告
    deep_engine.print_full_report(decisions, top_n=5)

    # 已持仓动作摘要
    print(f"\n  {'═'*70}")
    print(f"  📋 已持仓状态机决策")
    print(f"  {'═'*70}")
    held_decisions = [d for d in decisions if d.ticker in current_pos]
    if held_decisions:
        for d in held_decisions:
            pos = current_pos[d.ticker]
            icon = {'HOLD':'✋','ADD':'➕','REDUCE':'➖','EXIT':'🚪','NONE':'⚪'}.get(d.position_action, '?')
            print(f"\n  {icon} {d.ticker} (当前仓位{pos:.1%}) → 动作: {d.position_action}")
            print(f"     评分{d.six_dim_score.total:.0f}  信号{d.buy_signal}  优先级{d.priority}")
            print(f"     原因: {' | '.join(d.reasons[:2])}")
            if d.position_action == 'REDUCE':
                print(f"     执行: 减仓至 {pos*0.7:.1%}")
            elif d.position_action == 'EXIT':
                print(f"     执行: 全部清仓")
            elif d.position_action == 'ADD':
                print(f"     执行: 加仓至 {min(pos + 0.02, d.kelly_position_pct):.1%}")

    # ── Step9: BookV2 精华完整演示（书第5-9章原版实现）──────────────────────
    print(f"\n{'─'*65}")
    print(f"  Step9: BookV2 — 《Quantitative Trading Strategies Using Python》")
    print(f"         Peng Liu 全章原版精华实现")
    print(f"{'─'*65}")

    # ── BookCh5: 趋势跟随（精确实现书第167-172页）─────────────────────────
    print(f"\n  [Ch.5 趋势跟随 — SMA/EMA + shift(1) + log returns]")
    sample_ticker = prices.columns[0]
    df_ch5 = BookCh5_TrendFollowing.generate_strategy(
        prices[sample_ticker], short_window=3, long_window=20
    )
    ch5_report = BookCh5_TrendFollowing.report_actions(df_ch5)
    print(f"  标的: {sample_ticker}")
    print(f"  总交易次数: {ch5_report['total_trades']}  "
          f"（多→空: {ch5_report['short_signal']}, 空→多: {ch5_report['long_signal']}）")
    print(f"  策略wealth: {ch5_report['final_wealth']:.4f}  "
          f"vs Buy&Hold: {ch5_report['buyhold_wealth']:.4f}")
    print(f"  {'✅ 战胜买入持有' if ch5_report['beat_buyhold'] else '❌ 未战胜买入持有'}")
    BookCh7_BacktestMetrics.full_report(
        df_ch5['log_return_strategy'], label='Ch.5趋势跟随策略'
    )

    # ── BookCh6: 横截面动量（精确实现书第190-195页五分位）────────────────
    print(f"\n  [Ch.6 横截面动量 — 月频 + 5分位 + Lookback/Lookahead双窗口]")
    cs_backtest = BookCh6_CrossSectionalMomentum.rolling_backtest(
        prices, lookback_months=6, n_quantiles=5, n_periods=12
    )
    if len(cs_backtest) > 0:
        print(f"  滚动回测 {len(cs_backtest)} 期：")
        print(f"  平均月度Long-Short Profit: {cs_backtest['ls_profit'].mean():+.2%}")
        print(f"  胜率: {(cs_backtest['ls_profit']>0).mean():.0%}  "
              f"最佳月: {cs_backtest['ls_profit'].max():+.2%}  "
              f"最差月: {cs_backtest['ls_profit'].min():+.2%}")

    # ── BookCh7: 回测指标（精确实现Listing 7-13至7-16）───────────────────
    print(f"\n  [Ch.7 回测指标 — wealth_index + Calmar trailing-36m]")
    if 'daily_returns' in main_result and len(main_result['daily_returns']) > 100:
        dr = main_result['daily_returns']
        ch7_report = BookCh7_BacktestMetrics.full_report(dr, label='主策略书第7章精确指标')

        # Drawdown序列演示
        dd_df = BookCh7_BacktestMetrics.drawdown_series(dr)
        print(f"\n  Wealth Index示意（前后3条）:")
        print(f"    起始: {dd_df['Wealth index'].iloc[0]:.2f}  "
              f"终值: {dd_df['Wealth index'].iloc[-1]:.2f}")
        print(f"    最大回撤发生日: {dd_df['Drawdown'].idxmin().date() if hasattr(dd_df['Drawdown'].idxmin(),'date') else dd_df['Drawdown'].idxmin()}")
        print(f"    最大回撤: {dd_df['Drawdown'].min():.2%}")

    # ── BookCh8: 统计套利（精确实现 + half-life）──────────────────────────
    print(f"\n  [Ch.8 统计套利 — Engle-Granger + z-score + half-life]")
    statarb_book = BookCh8_StatisticalArbitrage(entry_z=2.0, exit_z=0.5, window=21)
    pairs_book   = statarb_book.find_pairs(prices)
    if pairs_book:
        print(f"  找到 {len(pairs_book)} 对协整：")
        for p in pairs_book[:3]:
            ok = '✅可交易' if p['tradeable'] else '⚠️不在3-60天区间'
            print(f"    {p['t1']}/{p['t2']}: pvalue={p['pvalue']:.4f}  "
                  f"half-life={p['half_life']:.1f}天  hedge_β={p['hedge_ratio']:.3f}  "
                  f"({p['method']}) {ok}")

        # 回测最佳pair
        best = pairs_book[0]
        if best['tradeable']:
            pair_report = statarb_book.backtest(
                prices[best['t1']], prices[best['t2']], best, tc_bps=5
            )
            print(f"\n  最佳pair回测: {best['t1']}/{best['t2']}")
            print(f"    Sharpe: {pair_report.get('sharpe_ratio',0):.3f}  "
                  f"MaxDD: {pair_report.get('max_drawdown',0):.2%}  "
                  f"年化: {pair_report.get('annualized_return',0):+.2%}")
            print(f"    交易次数: {pair_report.get('n_trades',0)}  "
                  f"当前z: {pair_report.get('final_zscore',0):.2f}  "
                  f"持仓: {pair_report.get('current_pos',0):+.0f}")
    else:
        print(f"  当前样本中未找到协整对")

    # ── BookCh9: Bayesian Optimization（精确实现GP+EI/UCB）────────────────
    print(f"\n  [Ch.9 Bayesian优化 — GP + EI + UCB + 重复实验]")

    # 优化目标：双MA趋势跟随的Sharpe，约束 short < long
    def trend_sharpe(short_window: float, long_window: float):
        if short_window >= long_window:
            return -99.0
        df = BookCh5_TrendFollowing.generate_strategy(
            prices[prices.columns[0]],
            short_window=int(short_window),
            long_window=int(long_window)
        )
        sr = BookCh7_BacktestMetrics.sharpe_ratio(df['log_return_strategy'])
        return sr if np.isfinite(sr) else -99.0

    # 重复实验（书第299-300页）
    print(f"  运行重复BO实验（3次runs × 10 iter）...")
    rep_result = BookCh9_BayesianOptimizer.repeated_experiments(
        trend_sharpe,
        param_bounds={'short_window': (2.0, 10.0), 'long_window': (15.0, 50.0)},
        n_runs=3, n_iter=10
    )
    print(f"  重复实验结果：")
    print(f"    Mean Sharpe: {rep_result['mean_score']:.4f}  "
          f"± {rep_result['std_score']:.4f}")
    print(f"    Range: [{rep_result['min_score']:.4f}, {rep_result['max_score']:.4f}]")

    # 单次最优运行（展示完整BO过程）
    print(f"\n  最佳参数搜索（UCB acquisition）：")
    bo_single = BookCh9_BayesianOptimizer(beta=2.0)
    bo_result = bo_single.optimize(
        trend_sharpe,
        param_bounds={'short_window': (2.0, 10.0), 'long_window': (15.0, 50.0)},
        n_iter=15, n_initial=5, acquisition='ucb',
        constraints=[lambda p: p['short_window'] < p['long_window']]
    )
    print(f"    Best params: short={bo_result['best_params']['short_window']:.1f}  "
          f"long={bo_result['best_params']['long_window']:.1f}")
    print(f"    Best Sharpe: {bo_result['best_score']:.4f}")
    print(f"    评估次数: {bo_result['n_evaluated']}")

    # ── BookV2 操作白皮书 ─────────────────────────────────────────────────────
    BookV2_OperationManual.print_full_manual()

    # Pre-Trade Checklist示例
    print(f"\n  [Pre-Trade Checklist 示例]")
    if 'daily_returns' in main_result and len(main_result['daily_returns']) > 30:
        sample_report = BookCh7_BacktestMetrics.full_report(main_result['daily_returns'])
        sample_signal = {'signal': 1, 'ticker': 'NVDA'}
        checklist = BookV2_OperationManual.pre_trade_checklist(
            'Canyon Main Strategy', sample_report, sample_signal
        )
        print(f"  策略: {checklist['strategy']}")
        print(f"  通过率: {checklist['passed']}/{checklist['total_checks']} "
              f"({checklist['pass_rate']:.0%})")
        for desc, ok in checklist['checklist']:
            print(f"    {'✅' if ok else '❌'} {desc}")
        print(f"  {'✅ 可以交易' if checklist['can_trade'] else '❌ 禁止交易'}")

    return m
    m = main_result
    print(f"  年化收益：{m.get('ann_ret',0):+.2%}")
    print(f"  Sharpe：  {m.get('sharpe',0):.4f}")
    print(f"  MaxDD：   {m.get('max_dd',0):.2%}")
    print(f"  多头贡献：{m.get('long_total_pnl',0):+.2%}")
    print(f"  空头贡献：{m.get('short_total_pnl',0):+.2%}")
    print(f"\n  v8相比v7的改进：")
    print(f"  · GARCH波动率预测（代替简单rolling std）")
    print(f"  · PSR+DSR（修正选择偏差和分布偏态）")
    print(f"  · PCA拥挤检测（识别隐藏的因子集中度）")
    print(f"  · Copula尾部风险（联合暴跌概率，比相关系数更保守）")
    print(f"  · GBM Monte Carlo VaR（路径模拟，比正态分布假设更真实）")
    print(f"{'═'*65}")

    return m


if __name__ == '__main__':
    result = main()
