#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 59 - v8 Advanced Risk Bridge

Integrates v8 risk diagnostics into Canyon v9 as research-only outputs:
- PCA crowding and factor exposure
- GBM Monte Carlo VaR / ES
- Gaussian Copula tail risk
- GARCH-like volatility forecast

Guardrails:
- No broker connection.
- No live orders.
- Does not change portfolio weights.
- If online price history is unavailable, output is marked SYNTHETIC_FALLBACK.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import importlib.util

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SOURCE_CANDIDATES = [
    ROOT / "canyon_final_v8_latest_source.py",
    ROOT / "canyon_final_v8_legacy_source.py",
]
SOURCE = next(
    (path for path in SOURCE_CANDIDATES if path.exists()), SOURCE_CANDIDATES[-1]
)

OUT_SUMMARY = ROOT / "v8_advanced_risk_summary.csv"
OUT_PCA = ROOT / "v8_pca_factor_exposure.csv"
OUT_TAIL = ROOT / "v8_tail_dependence_matrix.csv"
OUT_REPORT = ROOT / "v8_advanced_risk_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(value, default=np.nan) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except Exception:
        return default


def load_v8_module():
    if not SOURCE.exists():
        candidates = ", ".join(str(path) for path in SOURCE_CANDIDATES)
        raise FileNotFoundError(f"Missing v8 source. Checked: {candidates}")
    spec = importlib.util.spec_from_file_location("canyon_v8_source", SOURCE)
    if spec is None:
        raise ImportError(f"Cannot create module spec from {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("spec.loader is None — cannot exec module")
    spec.loader.exec_module(module)
    return module


def portfolio_weights() -> pd.Series:
    exposure = read_csv(ROOT / "exposure_dashboard.csv")
    if not exposure.empty and {"ticker", "effective_weight"}.issubset(exposure.columns):
        rows = []
        for _, row in exposure.iterrows():
            ticker = str(row.get("ticker", "")).upper().strip()
            weight = fnum(row.get("effective_weight", np.nan))
            if ticker and np.isfinite(weight) and abs(weight) > 0:
                rows.append((ticker, weight))
        if rows:
            return pd.Series(dict(rows), dtype=float)

    sizing = read_csv(ROOT / "position_sizing_recommendations.csv")
    if not sizing.empty and {"ticker", "suggested_weight"}.issubset(sizing.columns):
        out = {}
        for _, row in sizing.iterrows():
            ticker = str(row.get("ticker", "")).upper().strip()
            weight = fnum(row.get("suggested_weight", np.nan))
            if ticker and np.isfinite(weight) and abs(weight) > 0:
                out[ticker] = out.get(ticker, 0.0) + weight
        if out:
            return pd.Series(out, dtype=float)

    return pd.Series(dtype=float)


def load_price_history(module, tickers: list[str]) -> tuple[pd.DataFrame, str]:
    end = datetime.now().date()
    start = end - timedelta(days=730)
    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=str(start),
            end=str(end + timedelta(days=1)),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise RuntimeError("empty yfinance response")
        if isinstance(raw.columns, pd.MultiIndex):
            level_1 = raw.columns.get_level_values(1)
            col_name = "Close" if "Close" in level_1 else "Adj Close"
            prices = raw.xs(col_name, axis=1, level=1)
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices.reindex(columns=tickers).ffill().dropna(how="all")
        prices = (
            prices.dropna(axis=1, thresh=max(30, int(len(prices) * 0.6)))
            .ffill()
            .dropna()
        )
        if len(prices) < 60 or len(prices.columns) < 3:
            raise RuntimeError("insufficient online history")
        return prices, "YFINANCE_HISTORY"
    except Exception:
        prices, _volumes, _market = module.DataLayer._synthetic(
            tickers, str(start), str(end)
        )
        return prices, "SYNTHETIC_FALLBACK"


def compute_risk(module, prices: pd.DataFrame, weights: pd.Series, data_source: str):
    common = [t for t in weights.index if t in prices.columns]
    weights = weights.reindex(common).dropna()
    prices = prices[common].dropna()  # pyright: ignore[reportAssignmentType]

    if len(common) < 3 or len(prices) < 60:
        summary = pd.DataFrame(
            [
                {
                    "status": "MISSING",
                    "data_source": data_source,
                    "metric": "advanced_risk",
                    "value": "",
                    "interpretation": "Insufficient price history or weights.",
                }
            ]
        )
        return summary, pd.DataFrame(), pd.DataFrame()

    returns = prices.pct_change().dropna()
    port_returns = (
        returns[weights.index] @ weights
    ).dropna()  # pyright: ignore[reportAttributeAccessIssue]

    pca_model = module.PCAFactorModel(n_components=min(5, len(common))).fit(returns)
    crowd = pca_model.crowding_score()
    exposure = pca_model.factor_exposure(weights)

    gbm = module.GBMModel.portfolio_var(prices, weights, horizon=21, n_paths=2000)
    copula = module.CopulaRiskModel.gaussian_copula_sim(
        returns, weights, n_sims=3000, horizon=21
    )
    garch_vol = module.GARCHModel.quick_forecast(port_returns, horizon=1)

    rows = [
        {
            "status": "RISK" if bool(crowd.get("crowding_alert", False)) else "OK",
            "data_source": data_source,
            "metric": "pca_crowding",
            "value": crowd.get("pc1_variance_explained", 0),
            "interpretation": crowd.get("interpretation", ""),
        },
        {
            "status": "RISK" if gbm.get("exceeds_limit", False) else "OK",
            "data_source": data_source,
            "metric": "gbm_var_21d_95",
            "value": gbm.get("var", 0),
            "interpretation": "GBM Monte Carlo 21d 95% VaR. Research-only stress input.",
        },
        {
            "status": "RISK" if gbm.get("exceeds_limit", False) else "OK",
            "data_source": data_source,
            "metric": "gbm_es_21d_95",
            "value": gbm.get("es", 0),
            "interpretation": "GBM Monte Carlo expected shortfall for worst 5%.",
        },
        {
            "status": "RISK" if copula.get("warning", False) else "OK",
            "data_source": data_source,
            "metric": "copula_joint_crash_prob",
            "value": copula.get("joint_crash_prob", 0),
            "interpretation": "Gaussian copula joint crash probability. Research-only tail-risk input.",
        },
        {
            "status": "RISK" if copula.get("tail_var_5", 0) < -0.056 else "OK",
            "data_source": data_source,
            "metric": "copula_tail_var_5",
            "value": copula.get("tail_var_5", 0),
            "interpretation": "Copula 5% tail VaR over 21 days.",
        },
        {
            "status": "OK",
            "data_source": data_source,
            "metric": "garch_portfolio_vol_1d",
            "value": round(float(garch_vol), 4),
            "interpretation": "GARCH-like annualized volatility forecast for current weighted portfolio.",
        },
        {
            "status": "OK",
            "data_source": data_source,
            "metric": "portfolio_weight_sum_abs",
            "value": round(float(weights.abs().sum()), 4),
            "interpretation": "Absolute gross weight used by advanced risk overlay.",
        },
    ]
    summary = pd.DataFrame(rows)

    pca_rows = []
    explained = getattr(pca_model, "explained_", None)
    if explained is not None:
        for pc, val in explained.items():
            pca_rows.append(
                {
                    "status": "RISK" if pc == "PC1" and float(val) > 0.50 else "OK",
                    "item": pc,
                    "value": round(float(val), 4),
                    "description": "variance_explained",
                }
            )
    for pc, val in exposure.get("exposures", {}).items():
        pca_rows.append(
            {
                "status": "OK",
                "item": pc,
                "value": val,
                "description": "portfolio_factor_exposure",
            }
        )
    pca_rows.append(
        {
            "status": "OK",
            "item": "dominant_factor",
            "value": exposure.get("dominant_factor", ""),
            "description": "largest absolute PCA exposure",
        }
    )
    pca_df = pd.DataFrame(pca_rows)

    tail = module.CopulaRiskModel.tail_dependence(returns, threshold=0.10)
    tail.index.name = "ticker"
    tail_df = tail.reset_index()

    return summary, pca_df, tail_df


def write_report(summary: pd.DataFrame, pca: pd.DataFrame, tail: pd.DataFrame):
    data_source = (
        summary["data_source"].iloc[0]
        if not summary.empty and "data_source" in summary.columns
        else "UNKNOWN"
    )
    risk_rows = (
        summary[summary["status"].astype(str).eq("RISK")]
        if not summary.empty
        else pd.DataFrame()
    )

    md = [
        "# Canyon v9 Step 59 - v8 Advanced Risk Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Guardrails",
        "- Research-only advanced risk overlay.",
        "- No broker connection.",
        "- No live orders.",
        "- Does not change portfolio weights.",
        "",
        "## Data Source",
        f"- v8 source: `{SOURCE.name}`",
        f"- `{data_source}`",
        "",
        "## Summary",
        f"- Risk flags: {len(risk_rows)}",
        f"- Summary rows: {len(summary)}",
        f"- PCA rows: {len(pca)}",
        f"- Tail-dependence columns: {len(tail.columns) if not tail.empty else 0}",
        "",
    ]
    if not summary.empty:
        md.append(summary.to_markdown(index=False))
        md.append("")
    md.extend(
        [
            "## Interpretation",
            "- PCA crowding checks whether the portfolio is secretly one common factor.",
            "- GBM VaR/ES is a Monte Carlo stress input, not a return forecast.",
            "- Copula tail risk estimates crisis co-movement; it can still underestimate real panic behavior.",
            "- If data source is `SYNTHETIC_FALLBACK`, treat numbers as plumbing validation only.",
        ]
    )
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")


def main():
    print("=" * 88)
    print("CANYON v9 Step 59")
    print("v8 Advanced Risk Bridge")
    print("=" * 88)

    module = load_v8_module()
    weights = portfolio_weights()
    if weights.empty:
        summary = pd.DataFrame(
            [
                {
                    "status": "MISSING",
                    "data_source": "NO_WEIGHTS",
                    "metric": "advanced_risk",
                    "value": "",
                    "interpretation": "No portfolio weights found.",
                }
            ]
        )
        pca = pd.DataFrame()
        tail = pd.DataFrame()
    else:
        prices, data_source = load_price_history(
            module, weights.index.tolist()
        )  # pyright: ignore[reportArgumentType]
        summary, pca, tail = compute_risk(module, prices, weights, data_source)

    summary.to_csv(OUT_SUMMARY, index=False)
    pca.to_csv(OUT_PCA, index=False)
    tail.to_csv(OUT_TAIL, index=False)
    write_report(summary, pca, tail)

    print(f"Summary rows: {len(summary)}")
    print(f"PCA rows: {len(pca)}")
    print(f"Tail matrix rows: {len(tail)}")
    print("Files generated:")
    for path in [OUT_SUMMARY, OUT_PCA, OUT_TAIL, OUT_REPORT]:
        print(f"  {path}")
    print("No broker connection. No live order.")


if __name__ == "__main__":
    main()
