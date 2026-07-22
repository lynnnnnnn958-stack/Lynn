#!/usr/bin/env python3
"""
Canyon v9 Step 156 - Signal IC, Decay, and Failure Analysis.

Research-only. No broker connection. No live orders.

Step155 tells whether the backtest is credible. Step156 digs into the signals:
which signals work, which horizons they work on, where they decay, and which
signals should be down-weighted until more evidence arrives.

Outputs:
  signal_decay_analysis.csv
  signal_failure_deep_dive.csv
  signal_regime_ic_matrix.csv
  signal_failure_by_market_bucket.csv
  signal_horizon_regime_policy.csv
  signal_live_vs_backtest_drift.csv
  signal_downgrade_queue.csv
  signal_validation_state.json
  signal_validation_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    load_price_cache,
    now_str,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_DECAY = ROOT / "signal_decay_analysis.csv"
OUT_FAILURE = ROOT / "signal_failure_deep_dive.csv"
OUT_REGIME_IC = ROOT / "signal_regime_ic_matrix.csv"
OUT_BUCKET_FAILURE = ROOT / "signal_failure_by_market_bucket.csv"
OUT_POLICY = ROOT / "signal_horizon_regime_policy.csv"
OUT_DRIFT = ROOT / "signal_live_vs_backtest_drift.csv"
OUT_QUEUE = ROOT / "signal_downgrade_queue.csv"
OUT_STATE = ROOT / "signal_validation_state.json"
OUT_REPORT = ROOT / "signal_validation_report.md"

HORIZONS = [5, 10, 21, 63]
WARMUP_DAYS = 252
MIN_TICKERS_PER_DATE = 10
MIN_OBS_USABLE = 30


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def status_from_ic(mean_ic: float, t_stat: float, n_obs: int) -> str:
    if n_obs < MIN_OBS_USABLE:
        return "THIN_SAMPLE"
    if not np.isfinite(mean_ic):
        return "NO_DATA"
    if mean_ic > 0.05 and t_stat > 2.0:
        return "STRONG"
    if mean_ic > 0.02 and t_stat > 1.5:
        return "USABLE"
    if mean_ic > 0:
        return "WEAK"
    return "NEGATIVE"


def action_rank(action: str) -> int:
    order = {
        "KEEP_CORE": 0,
        "KEEP_WITH_MONITOR": 1,
        "USE_ONLY_AT_SHORT_HORIZON": 2,
        "REVIEW_SAMPLE_SIZE": 3,
        "DOWNWEIGHT": 4,
        "BLOCK_SIGNAL": 5,
        "NO_DATA": 6,
    }
    return order.get(str(action).upper(), 4)


def build_price_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = prices.ffill().dropna(how="all", axis=1)
    returns = p.pct_change(fill_method=None)
    signals: dict[str, pd.DataFrame] = {}
    signals["mom_1m"] = p.pct_change(21, fill_method=None).shift(1)
    signals["mom_3m"] = p.pct_change(63, fill_method=None).shift(1)
    signals["mom_6m"] = p.pct_change(126, fill_method=None).shift(1)
    signals["mom_12m_skip1m"] = (p.pct_change(252, fill_method=None) - p.pct_change(21, fill_method=None)).shift(1)
    signals["trend_200"] = ((p / (p.rolling(200).mean() + 1e-10)) - 1).shift(1)

    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    signals["rsi_rev"] = (100 - rsi).shift(1)

    vol21 = returns.rolling(21).std() * np.sqrt(252)
    signals["inv_vol"] = (1 / (vol21 + 1e-6)).shift(1)

    mom_3m_raw = p.pct_change(63, fill_method=None)
    mom_6m_raw = p.pct_change(126, fill_method=None)
    signals["mom_accel"] = (mom_3m_raw - mom_6m_raw).shift(1)
    signals["new_high_52w"] = (p / (p.rolling(252).max() + 1e-10)).shift(1)
    return signals


def spearman_ic(signal_row: pd.Series, forward_row: pd.Series) -> float:
    s = pd.to_numeric(signal_row, errors="coerce").dropna()
    f = pd.to_numeric(forward_row, errors="coerce").dropna()
    common = s.index.intersection(f.index)
    if len(common) < MIN_TICKERS_PER_DATE:
        return np.nan
    return float(s.loc[common].rank().corr(f.loc[common].rank()))


def compute_decay(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=[
            "signal", "horizon_days", "n_obs", "mean_ic", "std_ic",
            "t_stat", "ic_positive_pct", "status", "decay_from_5d",
            "sample_start", "sample_end", "source_file",
        ])

    prices = prices.sort_index().ffill().dropna(how="all", axis=1)
    liquid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= WARMUP_DAYS + 80]
    prices = prices[liquid_cols].tail(1500)
    signals = build_price_signals(prices)
    sample_dates = prices.index[WARMUP_DAYS:-max(HORIZONS):21]
    rows: list[dict[str, Any]] = []

    for sig_name, sig_df in signals.items():
        sig_df = sig_df.reindex(prices.index)
        for horizon in HORIZONS:
            fwd_ret = prices.pct_change(horizon, fill_method=None).shift(-horizon)
            ics: list[float] = []
            used_dates = []
            for dt in sample_dates:
                if dt not in sig_df.index or dt not in fwd_ret.index:
                    continue
                ic = spearman_ic(sig_df.loc[dt], fwd_ret.loc[dt])
                if np.isfinite(ic):
                    ics.append(ic)
                    used_dates.append(dt)
            arr = np.array(ics, dtype=float)
            if arr.size:
                mean_ic = float(arr.mean())
                std_ic = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
                t_stat = mean_ic / (std_ic / np.sqrt(arr.size) + 1e-12) if np.isfinite(std_ic) and std_ic > 0 else np.nan
                pos_pct = float((arr > 0).mean() * 100.0)
                status = status_from_ic(mean_ic, t_stat, int(arr.size))
                start = str(pd.Timestamp(min(used_dates)).date())
                end = str(pd.Timestamp(max(used_dates)).date())
            else:
                mean_ic = std_ic = t_stat = pos_pct = np.nan
                status = "NO_DATA"
                start = end = ""
            rows.append({
                "signal": sig_name,
                "horizon_days": horizon,
                "n_obs": int(arr.size),
                "mean_ic": round(mean_ic, 4) if np.isfinite(mean_ic) else np.nan,
                "std_ic": round(std_ic, 4) if np.isfinite(std_ic) else np.nan,
                "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else np.nan,
                "ic_positive_pct": round(pos_pct, 1) if np.isfinite(pos_pct) else np.nan,
                "status": status,
                "sample_start": start,
                "sample_end": end,
                "source_file": "local price cache; signals lagged 1 day",
            })

    decay = pd.DataFrame(rows)
    if not decay.empty:
        h5 = decay[decay["horizon_days"] == 5][["signal", "mean_ic"]].rename(columns={"mean_ic": "ic_5d"})
        decay = decay.merge(h5, on="signal", how="left")
        decay["decay_from_5d"] = pd.to_numeric(decay["mean_ic"], errors="coerce") - pd.to_numeric(decay["ic_5d"], errors="coerce")
        decay["decay_from_5d"] = decay["decay_from_5d"].round(4)
        decay = decay.drop(columns=["ic_5d"])
    return decay


def build_market_regimes(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["date", "market_regime", "vol_bucket", "trend_bucket", "breadth_bucket"])
    p = prices.sort_index().ffill().dropna(how="all", axis=1)
    proxy = "SPY" if "SPY" in p.columns else p.columns[0]
    spy = pd.to_numeric(p[proxy], errors="coerce").ffill()
    returns = p.pct_change(fill_method=None)
    spy_ret_63 = spy.pct_change(63, fill_method=None)
    spy_ret_21 = spy.pct_change(21, fill_method=None)
    vol21 = spy.pct_change(fill_method=None).rolling(21).std() * np.sqrt(252)
    vol_pct = vol21.rolling(252, min_periods=80).rank(pct=True)
    ma200 = spy.rolling(200, min_periods=120).mean()
    breadth = (returns.rolling(21).sum() > 0).mean(axis=1)
    rows = []
    for dt in p.index:
        r63 = safe_float(spy_ret_63.loc[dt])
        r21 = safe_float(spy_ret_21.loc[dt])
        vp = safe_float(vol_pct.loc[dt])
        above = safe_float(spy.loc[dt]) > safe_float(ma200.loc[dt], np.inf)
        br = safe_float(breadth.loc[dt])
        if not np.isfinite(r63) or not np.isfinite(vp):
            regime = "REGIME_DATA_GAP"
        elif r63 > 0 and above and vp < 0.70:
            regime = "RISK_ON_UPTREND"
        elif r63 > 0 and vp >= 0.70:
            regime = "RISK_ON_HIGH_VOL"
        elif r63 <= 0 and vp >= 0.65:
            regime = "RISK_OFF_HIGH_VOL"
        elif r63 <= 0:
            regime = "RISK_OFF_DOWNTREND"
        else:
            regime = "MIXED"
        if not np.isfinite(vp):
            vol_bucket = "VOL_DATA_GAP"
        elif vp >= 0.80:
            vol_bucket = "HIGH_VOL"
        elif vp <= 0.25:
            vol_bucket = "LOW_VOL"
        else:
            vol_bucket = "MID_VOL"
        if not np.isfinite(r21):
            trend_bucket = "TREND_DATA_GAP"
        elif r21 > 0.03:
            trend_bucket = "STRONG_1M_UP"
        elif r21 < -0.03:
            trend_bucket = "WEAK_1M_DOWN"
        else:
            trend_bucket = "FLAT_1M"
        if not np.isfinite(br):
            breadth_bucket = "BREADTH_DATA_GAP"
        elif br >= 0.65:
            breadth_bucket = "BREADTH_STRONG"
        elif br <= 0.40:
            breadth_bucket = "BREADTH_WEAK"
        else:
            breadth_bucket = "BREADTH_MIXED"
        rows.append({
            "date": dt,
            "market_regime": regime,
            "vol_bucket": vol_bucket,
            "trend_bucket": trend_bucket,
            "breadth_bucket": breadth_bucket,
            "spy_ret_21d": r21,
            "spy_ret_63d": r63,
            "vol_percentile": vp,
            "breadth_positive_21d_pct": br * 100 if np.isfinite(br) else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


def compute_regime_ic(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()
    prices = prices.sort_index().ffill().dropna(how="all", axis=1)
    liquid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= WARMUP_DAYS + 80]
    prices = prices[liquid_cols].tail(1500)
    signals = build_price_signals(prices)
    regimes = build_market_regimes(prices)
    sample_dates = prices.index[WARMUP_DAYS:-max(HORIZONS):21]
    obs_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for sig_name, sig_df in signals.items():
        sig_df = sig_df.reindex(prices.index)
        for horizon in HORIZONS:
            fwd_ret = prices.pct_change(horizon, fill_method=None).shift(-horizon)
            for dt in sample_dates:
                if dt not in sig_df.index or dt not in fwd_ret.index or dt not in regimes.index:
                    continue
                ic = spearman_ic(sig_df.loc[dt], fwd_ret.loc[dt])
                if not np.isfinite(ic):
                    continue
                reg = regimes.loc[dt]
                obs_rows.append({
                    "signal": sig_name,
                    "horizon_days": horizon,
                    "sample_date": str(pd.Timestamp(dt).date()),
                    "ic": ic,
                    "market_regime": str(reg.get("market_regime")),
                    "vol_bucket": str(reg.get("vol_bucket")),
                    "trend_bucket": str(reg.get("trend_bucket")),
                    "breadth_bucket": str(reg.get("breadth_bucket")),
                })
    obs = pd.DataFrame(obs_rows)
    if obs.empty:
        return pd.DataFrame(), pd.DataFrame()

    for group_col in ["market_regime", "vol_bucket", "trend_bucket", "breadth_bucket"]:
        for (sig, horizon, bucket), grp in obs.groupby(["signal", "horizon_days", group_col], dropna=False):
            vals = pd.to_numeric(grp["ic"], errors="coerce").dropna()
            n = len(vals)
            mean_ic = float(vals.mean()) if n else np.nan
            std_ic = float(vals.std(ddof=1)) if n > 1 else np.nan
            t_stat = mean_ic / (std_ic / np.sqrt(n) + 1e-12) if n > 1 and np.isfinite(std_ic) and std_ic > 0 else np.nan
            pos_pct = float((vals > 0).mean() * 100) if n else np.nan
            summary_rows.append({
                "signal": sig,
                "horizon_days": int(horizon),
                "bucket_type": group_col,
                "bucket": bucket,
                "n_obs": int(n),
                "mean_ic": round(mean_ic, 4) if np.isfinite(mean_ic) else np.nan,
                "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else np.nan,
                "ic_positive_pct": round(pos_pct, 1) if np.isfinite(pos_pct) else np.nan,
                "status": status_from_ic(mean_ic, t_stat, n),
                "source_file": "local price cache; regime buckets derived from SPY trend/vol/breadth",
            })
    summary = pd.DataFrame(summary_rows)
    return summary, obs


def build_bucket_failure(regime_ic: pd.DataFrame) -> pd.DataFrame:
    if regime_ic.empty:
        return pd.DataFrame()
    work = regime_ic.copy()
    work["mean_ic_num"] = pd.to_numeric(work["mean_ic"], errors="coerce")
    rows = []
    for sig, grp in work.groupby("signal", dropna=False):
        usable = grp.dropna(subset=["mean_ic_num"])
        if usable.empty:
            continue
        worst = usable.sort_values("mean_ic_num", ascending=True).iloc[0]
        best = usable.sort_values("mean_ic_num", ascending=False).iloc[0]
        neg = usable[usable["mean_ic_num"] <= 0]
        thin = usable[pd.to_numeric(usable["n_obs"], errors="coerce") < MIN_OBS_USABLE]
        if len(neg) >= max(2, len(usable) * 0.35):
            failure_status = "REGIME_FRAGILE"
            action = "Use only in proven buckets; downweight otherwise."
        elif len(thin) >= len(usable) * 0.50:
            failure_status = "THIN_BUCKET_EVIDENCE"
            action = "Collect more observations before regime-specific sizing."
        else:
            failure_status = "BUCKET_MONITOR"
            action = "Keep monitored; no bucket-specific upgrade yet."
        rows.append({
            "signal": sig,
            "failure_status": failure_status,
            "worst_bucket_type": worst.get("bucket_type"),
            "worst_bucket": worst.get("bucket"),
            "worst_bucket_horizon_days": int(worst.get("horizon_days")),
            "worst_bucket_ic": round(float(worst.get("mean_ic_num")), 4),
            "best_bucket_type": best.get("bucket_type"),
            "best_bucket": best.get("bucket"),
            "best_bucket_horizon_days": int(best.get("horizon_days")),
            "best_bucket_ic": round(float(best.get("mean_ic_num")), 4),
            "negative_bucket_count": int(len(neg)),
            "thin_bucket_count": int(len(thin)),
            "required_next_action": action,
            "source_file": "signal_regime_ic_matrix.csv",
        })
    return pd.DataFrame(rows).sort_values(["failure_status", "signal"]).reset_index(drop=True)


def build_horizon_regime_policy(failure: pd.DataFrame, regime_ic: pd.DataFrame, bucket_failure: pd.DataFrame) -> pd.DataFrame:
    signals = sorted(set(failure.get("signal", pd.Series(dtype=str)).dropna().astype(str).tolist()) |
                     set(regime_ic.get("signal", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    rows = []
    for sig in signals:
        f = failure[failure["signal"].astype(str) == sig].head(1) if not failure.empty and "signal" in failure.columns else pd.DataFrame()
        b = bucket_failure[bucket_failure["signal"].astype(str) == sig].head(1) if not bucket_failure.empty and "signal" in bucket_failure.columns else pd.DataFrame()
        r = regime_ic[regime_ic["signal"].astype(str) == sig].copy() if not regime_ic.empty and "signal" in regime_ic.columns else pd.DataFrame()
        base_action = str(f.iloc[0].get("recommended_signal_action")) if not f.empty else "NO_DATA"
        best_horizon = str(f.iloc[0].get("best_horizon")) if not f.empty else ""
        worst_horizon = str(f.iloc[0].get("worst_horizon")) if not f.empty else ""
        bucket_status = str(b.iloc[0].get("failure_status")) if not b.empty else "NO_BUCKET_DATA"
        allowed = "BLOCK"
        weight = 0.0
        why = "No signal validation evidence."
        if base_action == "KEEP_CORE":
            allowed, weight, why = "ALL_HORIZONS_WITH_MONITOR", 1.00, "Core IC evidence is usable across horizons."
        elif base_action == "KEEP_WITH_MONITOR":
            allowed, weight, why = f"BEST_HORIZON_{best_horizon}_ONLY_UNTIL_MORE_PROOF", 0.80, "Signal works, but horizon decay requires monitoring."
        elif base_action == "USE_ONLY_AT_SHORT_HORIZON":
            allowed, weight, why = "SHORT_HORIZON_ONLY", 0.55, "Signal works only at short horizon; block medium/long promotion."
        elif base_action == "DOWNWEIGHT":
            allowed, weight, why = "RESEARCH_ONLY_DOWNWEIGHTED", 0.35, "Signal IC is weak or unstable."
        elif base_action == "REVIEW_SAMPLE_SIZE":
            allowed, weight, why = "MANUAL_REVIEW_ONLY", 0.25, "Sample size is too shallow for automatic sizing."
        elif base_action == "BLOCK_SIGNAL":
            allowed, weight, why = "BLOCK", 0.0, "Negative or blocker signal evidence."
        if bucket_status == "REGIME_FRAGILE" and weight > 0:
            weight *= 0.70
            why += " Regime bucket failures reduce weight."
        proven = r[
            (r["status"].astype(str).isin(["STRONG", "USABLE", "WEAK"]))
            & (pd.to_numeric(r["mean_ic"], errors="coerce") > 0)
            & (r["bucket_type"].astype(str) == "market_regime")
        ]
        allowed_regimes = ", ".join(sorted(proven["bucket"].dropna().astype(str).unique().tolist())) if not proven.empty else "NONE_PROVEN"
        rows.append({
            "signal": sig,
            "allowed_use": allowed,
            "weight_multiplier": round(weight, 3),
            "base_action": base_action,
            "best_horizon": best_horizon,
            "worst_horizon": worst_horizon,
            "bucket_failure_status": bucket_status,
            "allowed_market_regimes": allowed_regimes,
            "why": why,
            "source_file": "signal_failure_deep_dive.csv; signal_regime_ic_matrix.csv; signal_failure_by_market_bucket.csv",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["weight_multiplier", "signal"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_failure_deep_dive(decay: pd.DataFrame) -> pd.DataFrame:
    base_ic = read_csv_safe(ROOT / "backtest_signal_ic.csv")
    failure_modes = read_csv_safe(ROOT / "backtest_signal_failure_modes.csv")

    signals = sorted(set(decay.get("signal", pd.Series(dtype=str)).dropna().astype(str).tolist()) |
                     set(base_ic.get("signal", pd.Series(dtype=str)).dropna().astype(str).tolist()) |
                     set(failure_modes.get("signal", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    signals = [s for s in signals if not s.lower().startswith("strategy:")]

    rows: list[dict[str, Any]] = []
    for sig in signals:
        sub = decay[decay["signal"] == sig].copy() if not decay.empty else pd.DataFrame()
        base = base_ic[base_ic["signal"].astype(str) == sig].copy() if not base_ic.empty and "signal" in base_ic.columns else pd.DataFrame()
        fail = failure_modes[failure_modes["signal"].astype(str) == sig].copy() if not failure_modes.empty and "signal" in failure_modes.columns else pd.DataFrame()

        best_horizon = ""
        worst_horizon = ""
        best_ic = np.nan
        worst_ic = np.nan
        neg_horizons = 0
        usable_horizons = 0
        if not sub.empty:
            sub["mean_ic_num"] = pd.to_numeric(sub["mean_ic"], errors="coerce")
            good = sub.dropna(subset=["mean_ic_num"])
            if not good.empty:
                best = good.sort_values("mean_ic_num", ascending=False).iloc[0]
                worst = good.sort_values("mean_ic_num", ascending=True).iloc[0]
                best_horizon = f"{int(best['horizon_days'])}d"
                worst_horizon = f"{int(worst['horizon_days'])}d"
                best_ic = float(best["mean_ic_num"])
                worst_ic = float(worst["mean_ic_num"])
                neg_horizons = int((good["mean_ic_num"] <= 0).sum())
                usable_horizons = int(good["status"].astype(str).isin(["STRONG", "USABLE"]).sum())

        base_mean = safe_float(base["mean_ic"].iloc[0]) if not base.empty and "mean_ic" in base.columns else np.nan
        base_n = int(safe_float(base["n_obs"].iloc[0], 0)) if not base.empty and "n_obs" in base.columns else 0
        base_status = str(base["status"].iloc[0]) if not base.empty and "status" in base.columns else "NO_BASELINE"
        failure_mode = str(fail["failure_mode"].iloc[0]) if not fail.empty and "failure_mode" in fail.columns else ""
        failure_status = str(fail["status"].iloc[0]) if not fail.empty and "status" in fail.columns else ""

        if base_n and base_n < MIN_OBS_USABLE:
            action = "REVIEW_SAMPLE_SIZE"
            reason = "baseline IC sample is shallow"
        elif failure_status in {"BLOCKER", "NEGATIVE"} or (np.isfinite(base_mean) and base_mean <= 0 and neg_horizons >= 2):
            action = "BLOCK_SIGNAL"
            reason = "negative IC or blocker failure mode"
        elif usable_horizons >= 2 and neg_horizons == 0:
            action = "KEEP_CORE"
            reason = "multiple usable horizons and no negative decay horizon"
        elif np.isfinite(best_ic) and best_ic > 0.03 and neg_horizons >= 1:
            action = "USE_ONLY_AT_SHORT_HORIZON" if best_horizon in {"5d", "10d"} else "KEEP_WITH_MONITOR"
            reason = "signal works only on selected horizons"
        elif np.isfinite(base_mean) and base_mean > 0:
            action = "DOWNWEIGHT"
            reason = "positive but weak or unstable"
        else:
            action = "NO_DATA"
            reason = "insufficient signal validation evidence"

        rows.append({
            "signal": sig,
            "recommended_signal_action": action,
            "reason": reason,
            "baseline_mean_ic": round(base_mean, 4) if np.isfinite(base_mean) else np.nan,
            "baseline_n_obs": base_n,
            "baseline_status": base_status,
            "best_horizon": best_horizon,
            "best_horizon_ic": round(best_ic, 4) if np.isfinite(best_ic) else np.nan,
            "worst_horizon": worst_horizon,
            "worst_horizon_ic": round(worst_ic, 4) if np.isfinite(worst_ic) else np.nan,
            "negative_horizon_count": neg_horizons,
            "usable_horizon_count": usable_horizons,
            "failure_mode": failure_mode,
            "failure_status": failure_status,
            "required_next_action": {
                "KEEP_CORE": "Keep monitored; require live IC drift check before increasing risk weight.",
                "KEEP_WITH_MONITOR": "Keep but monitor horizon-specific decay and regime failures.",
                "USE_ONLY_AT_SHORT_HORIZON": "Use only for short-horizon research; block medium/long-horizon promotion.",
                "REVIEW_SAMPLE_SIZE": "Collect more observations before sizing; no automatic upgrade.",
                "DOWNWEIGHT": "Reduce model weight until IC is stable by horizon and regime.",
                "BLOCK_SIGNAL": "Set signal weight to zero until logic is repaired and retested.",
                "NO_DATA": "Build missing history or remove from scoring.",
            }.get(action, "Review manually."),
            "source_files": "signal_decay_analysis.csv / backtest_signal_ic.csv / backtest_signal_failure_modes.csv",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["action_rank"] = out["recommended_signal_action"].apply(action_rank)
        out = out.sort_values(["action_rank", "signal"]).drop(columns=["action_rank"]).reset_index(drop=True)
    return out


def build_live_vs_backtest_drift(failure: pd.DataFrame) -> pd.DataFrame:
    live = read_csv_safe(ROOT / "live_ic_history.csv")
    base = read_csv_safe(ROOT / "backtest_signal_ic.csv")
    signals = sorted(set(failure.get("signal", pd.Series(dtype=str)).dropna().astype(str).tolist()) |
                     set(base.get("signal", pd.Series(dtype=str)).dropna().astype(str).tolist()))

    rows: list[dict[str, Any]] = []
    for sig in signals:
        base_row = base[base["signal"].astype(str) == sig] if not base.empty and "signal" in base.columns else pd.DataFrame()
        backtest_ic = safe_float(base_row["mean_ic"].iloc[0]) if not base_row.empty and "mean_ic" in base_row.columns else np.nan
        if live.empty:
            rows.append({
                "signal": sig,
                "backtest_mean_ic": round(backtest_ic, 4) if np.isfinite(backtest_ic) else np.nan,
                "live_mean_ic": np.nan,
                "ic_drift": np.nan,
                "live_observations": 0,
                "drift_status": "MISSING_LIVE_HISTORY",
                "required_next_action": "Run Step84 long enough to accumulate live forward-return observations before trusting live drift.",
                "source_files": "live_ic_history.csv / backtest_signal_ic.csv",
            })
            continue

        live_sub = live[live.astype(str).apply(lambda col: col.str.contains(sig, case=False, na=False)).any(axis=1)]
        possible_ic_cols = [c for c in live.columns if "ic" in str(c).lower() and c != "signal"]
        live_vals = pd.Series(dtype=float)
        for col in possible_ic_cols:
            live_vals = pd.concat([live_vals, pd.to_numeric(live_sub[col], errors="coerce")], ignore_index=True)
        live_vals = live_vals.replace([np.inf, -np.inf], np.nan).dropna()
        live_ic = float(live_vals.mean()) if not live_vals.empty else np.nan
        drift = live_ic - backtest_ic if np.isfinite(live_ic) and np.isfinite(backtest_ic) else np.nan
        if not np.isfinite(drift):
            drift_status = "DATA_GAP"
        elif drift < -0.05:
            drift_status = "DECAY_ALERT"
        elif drift < -0.02:
            drift_status = "WATCH"
        else:
            drift_status = "OK"
        rows.append({
            "signal": sig,
            "backtest_mean_ic": round(backtest_ic, 4) if np.isfinite(backtest_ic) else np.nan,
            "live_mean_ic": round(live_ic, 4) if np.isfinite(live_ic) else np.nan,
            "ic_drift": round(drift, 4) if np.isfinite(drift) else np.nan,
            "live_observations": int(len(live_vals)),
            "drift_status": drift_status,
            "required_next_action": "Down-weight if live drift persists for 30+ observations." if drift_status in {"DECAY_ALERT", "WATCH"} else "Keep accumulating live IC observations.",
            "source_files": "live_ic_history.csv / backtest_signal_ic.csv",
        })
    return pd.DataFrame(rows)


def build_queue(failure: pd.DataFrame, drift: pd.DataFrame, bucket_failure: pd.DataFrame | None = None) -> pd.DataFrame:
    if failure.empty:
        return pd.DataFrame()
    out = failure[failure["recommended_signal_action"].isin([
        "USE_ONLY_AT_SHORT_HORIZON", "REVIEW_SAMPLE_SIZE", "DOWNWEIGHT", "BLOCK_SIGNAL", "NO_DATA"
    ])].copy()
    if not drift.empty and "signal" in drift.columns:
        out = out.merge(drift[["signal", "drift_status", "live_observations"]], on="signal", how="left")
    if bucket_failure is not None and not bucket_failure.empty and "signal" in bucket_failure.columns:
        keep_bucket = [c for c in ["signal", "failure_status", "worst_bucket_type", "worst_bucket", "worst_bucket_ic"] if c in bucket_failure.columns]
        out = out.merge(bucket_failure[keep_bucket], on="signal", how="left", suffixes=("", "_bucket"))
        out["reason"] = out["reason"].astype(str) + np.where(
            out.get("failure_status", pd.Series(dtype=str)).astype(str).eq("REGIME_FRAGILE"),
            "; regime bucket fragility",
            "",
        )
    out["queue_priority"] = out["recommended_signal_action"].map({
        "BLOCK_SIGNAL": "P1",
        "NO_DATA": "P1",
        "DOWNWEIGHT": "P2",
        "REVIEW_SAMPLE_SIZE": "P2",
        "USE_ONLY_AT_SHORT_HORIZON": "P3",
    }).fillna("P3")
    keep = [
        "queue_priority", "signal", "recommended_signal_action", "reason",
        "baseline_mean_ic", "baseline_n_obs", "best_horizon", "best_horizon_ic",
        "worst_horizon", "worst_horizon_ic", "drift_status", "live_observations",
        "failure_status", "worst_bucket_type", "worst_bucket", "worst_bucket_ic",
        "required_next_action",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values(["queue_priority", "signal"]).reset_index(drop=True)


def build_state(decay: pd.DataFrame, failure: pd.DataFrame, drift: pd.DataFrame, queue: pd.DataFrame, regime_ic: pd.DataFrame | None = None, bucket_failure: pd.DataFrame | None = None, policy: pd.DataFrame | None = None) -> dict[str, Any]:
    action_counts = failure.get("recommended_signal_action", pd.Series(dtype=str)).value_counts().to_dict() if not failure.empty else {}
    decay_status_counts = decay.get("status", pd.Series(dtype=str)).value_counts().to_dict() if not decay.empty else {}
    p1 = int(queue.get("queue_priority", pd.Series(dtype=str)).eq("P1").sum()) if not queue.empty else 0
    p2 = int(queue.get("queue_priority", pd.Series(dtype=str)).eq("P2").sum()) if not queue.empty else 0
    live_observations = int(pd.to_numeric(drift.get("live_observations", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not drift.empty else 0
    live_ic_windows = int(pd.to_numeric(drift.get("live_mean_ic", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not drift.empty else 0
    regime_fragile = int(bucket_failure.get("failure_status", pd.Series(dtype=str)).astype(str).eq("REGIME_FRAGILE").sum()) if bucket_failure is not None and not bucket_failure.empty else 0
    blocked_policy = int(policy.get("allowed_use", pd.Series(dtype=str)).astype(str).eq("BLOCK").sum()) if policy is not None and not policy.empty else 0
    if p1:
        overall = "SIGNAL_REPAIR_REQUIRED"
    elif p2:
        overall = "SIGNAL_REVIEW_REQUIRED"
    else:
        overall = "SIGNAL_MONITOR"
    return {
        "date": today_str(),
        "generated_at": now_str(),
        "overall_status": overall,
        "signals_reviewed": int(failure["signal"].nunique()) if not failure.empty and "signal" in failure.columns else 0,
        "decay_rows": int(len(decay)),
        "downgrade_queue_rows": int(len(queue)),
        "p1_signal_repairs": p1,
        "p2_signal_reviews": p2,
        "action_counts": action_counts,
        "decay_status_counts": decay_status_counts,
        "live_ic_available": bool(live_observations > 0 and live_ic_windows > 0),
        "live_ic_observations": live_observations,
        "live_ic_windows": live_ic_windows,
        "regime_ic_rows": int(len(regime_ic)) if regime_ic is not None else 0,
        "regime_fragile_signals": regime_fragile,
        "policy_blocked_signals": blocked_policy,
        "research_only": True,
        "no_broker_connection": True,
        "truth": "This is a signal validation layer using local price/cache data and existing IC files. It supports research weights; it is not a paid vendor-grade point-in-time signal research database.",
    }


def write_report(decay: pd.DataFrame, failure: pd.DataFrame, regime_ic: pd.DataFrame, bucket_failure: pd.DataFrame, policy: pd.DataFrame, drift: pd.DataFrame, queue: pd.DataFrame, state: dict[str, Any]) -> None:
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Signals reviewed: **{state['signals_reviewed']}**",
        f"- Downgrade queue rows: **{state['downgrade_queue_rows']}**",
        f"- P1 signal repairs: **{state['p1_signal_repairs']}**",
        f"- Regime-fragile signals: **{state.get('regime_fragile_signals', 0)}**",
        f"- Policy-blocked signals: **{state.get('policy_blocked_signals', 0)}**",
        "",
        "Step156 is deliberately stricter than a single score. It separates horizon decay, regime/bucket fragility, baseline IC, live drift, and failure modes.",
        "",
        "## Signal Horizon/Regime Policy",
        "",
        df_to_markdown(policy, max_rows=40),
        "",
        "## Signal Downgrade Queue",
        "",
        df_to_markdown(queue, max_rows=40),
        "",
        "## Failure Deep Dive",
        "",
        df_to_markdown(failure, max_rows=40),
        "",
        "## IC Decay By Horizon",
        "",
        df_to_markdown(decay, max_rows=60),
        "",
        "## Regime IC Matrix",
        "",
        df_to_markdown(regime_ic, max_rows=80),
        "",
        "## Failure By Market Bucket",
        "",
        df_to_markdown(bucket_failure, max_rows=40),
        "",
        "## Live Versus Backtest Drift",
        "",
        df_to_markdown(drift, max_rows=40),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 156 - Signal IC, Decay, and Failure Analysis", sections)


def main() -> None:
    prices = load_price_cache()
    decay = compute_decay(prices)
    failure = build_failure_deep_dive(decay)
    regime_ic, _obs = compute_regime_ic(prices)
    bucket_failure = build_bucket_failure(regime_ic)
    policy = build_horizon_regime_policy(failure, regime_ic, bucket_failure)
    drift = build_live_vs_backtest_drift(failure)
    queue = build_queue(failure, drift, bucket_failure)
    state = build_state(decay, failure, drift, queue, regime_ic, bucket_failure, policy)

    decay.to_csv(OUT_DECAY, index=False)
    failure.to_csv(OUT_FAILURE, index=False)
    regime_ic.to_csv(OUT_REGIME_IC, index=False)
    bucket_failure.to_csv(OUT_BUCKET_FAILURE, index=False)
    policy.to_csv(OUT_POLICY, index=False)
    drift.to_csv(OUT_DRIFT, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    write_json(OUT_STATE, state)
    write_report(decay, failure, regime_ic, bucket_failure, policy, drift, queue, state)

    print("Canyon v9 Step156 signal IC/decay/failure analysis complete.")
    print(f"Overall: {state['overall_status']} | signals: {state['signals_reviewed']} | queue rows: {state['downgrade_queue_rows']}")
    print(f"Outputs: {OUT_DECAY.name}, {OUT_FAILURE.name}, {OUT_QUEUE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
