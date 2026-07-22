#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 61 - Data Source Health Monitor

Audits external-data availability and local fallback usage after a full run.

This does not fetch market data or make decisions. It only explains whether the
dashboard is running from live/online inputs, preserved local files, or explicit
fallback states.

No broker connection. No live order.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import socket

import pandas as pd

ROOT = Path.cwd()
OUT_HEALTH = ROOT / "data_source_health.csv"
OUT_REPORT = ROOT / "data_source_health_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def file_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    if path.suffix.lower() != ".csv":
        return 0
    try:
        return int(len(pd.read_csv(path, dtype=str)))
    except Exception:
        return 0


def dns_check(host: str, timeout: float = 2.0) -> tuple[str, str]:
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        socket.getaddrinfo(host, 443)
        return "OK", "DNS resolved"
    except Exception as exc:
        return "RISK", f"DNS failed: {exc}"
    finally:
        socket.setdefaulttimeout(old_timeout)


def count_contains(df: pd.DataFrame, column: str, pattern: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].astype(str).str.upper().str.contains(pattern, regex=True, na=False).sum())


def build_health() -> pd.DataFrame:
    rows = []

    for host in ["guce.yahoo.com", "query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        status, detail = dns_check(host)
        rows.append({
            "status": status,
            "area": "External DNS",
            "source": host,
            "signal": "dns_resolution",
            "value": status,
            "detail": detail,
            "action": "If RISK, online yfinance pulls will probably fail; rely on local fallback / research-only states.",
        })

    try:
        import yfinance  # noqa: F401
        rows.append({
            "status": "OK",
            "area": "Python Package",
            "source": "yfinance",
            "signal": "import",
            "value": "AVAILABLE",
            "detail": "Package import succeeded.",
            "action": "Connectivity, not package installation, is the likely blocker if Yahoo fetches fail.",
        })
    except Exception as exc:
        rows.append({
            "status": "RISK",
            "area": "Python Package",
            "source": "yfinance",
            "signal": "import",
            "value": "MISSING",
            "detail": str(exc),
            "action": "Install yfinance in the venv before relying on online market/option pulls.",
        })

    files = [
        ("Options Chain", ROOT / "options_chain_snapshot.csv", "Rows should be >0 when fresh Yahoo options are available."),
        ("Options Decision", ROOT / "options_decision_matrix.csv", "Can remain populated even when chain is unavailable because decisions are gated research-only."),
        ("Technical Matrix", ROOT / "technical_signal_matrix.csv", "Rows can exist with NO_PRICE fallback when Yahoo prices are unavailable."),
        ("Macro Signals", ROOT / "macro_regime_signals.csv", "Rows can exist with fallback regime labels when macro pulls fail."),
        ("Master v2", ROOT / "master_10_layer_decision_matrix_v2.csv", "Master should stay populated even when data sources are degraded."),
        ("Output Vault", ROOT / "canyon_output_vault_index.csv", "Vault should be populated so old outputs are recoverable."),
    ]
    for area, path, note in files:
        rows_count = file_rows(path)
        status = "OK" if rows_count > 0 else "WARN"
        if area in {"Master v2", "Output Vault"} and rows_count == 0:
            status = "RISK"
        rows.append({
            "status": status,
            "area": area,
            "source": path.name,
            "signal": "row_count",
            "value": rows_count,
            "detail": note,
            "action": "Review the relevant dashboard page if rows are unexpectedly zero.",
        })

    master = read_csv(ROOT / "master_10_layer_decision_matrix_v2.csv")
    unavailable = 0
    research_only = 0
    risk_reduction = 0
    if not master.empty:
        layer_cols = [c for c in master.columns if c.startswith("L") and c.endswith("_state")]
        unavailable = int(sum(count_contains(master, c, "DATA_UNAVAILABLE|PRICE_DATA_UNAVAILABLE|SECTOR_DATA_UNAVAILABLE|FUNDAMENTAL_DATA_UNAVAILABLE|OPTIONS_DATA_UNAVAILABLE") for c in layer_cols))
        research_only = count_contains(master, "master_action", "RESEARCH_ONLY")
        risk_reduction = count_contains(master, "master_action", "RISK_REDUCTION_FIRST")
    rows.append({
        "status": "WARN" if unavailable else "OK",
        "area": "Master Fallback Usage",
        "source": "master_10_layer_decision_matrix_v2.csv",
        "signal": "data_unavailable_cells",
        "value": unavailable,
        "detail": f"research_only={research_only}; risk_reduction_first={risk_reduction}",
        "action": "Data-unavailable states are explicit conservative fallbacks, not missing layers.",
    })

    tech = read_csv(ROOT / "technical_signal_matrix.csv")
    no_price = count_contains(tech, "data_status", "NO_PRICE|NO_DATA")
    rows.append({
        "status": "WARN" if no_price else "OK",
        "area": "L6 Technical",
        "source": "technical_signal_matrix.csv",
        "signal": "no_price_rows",
        "value": no_price,
        "detail": "NO_PRICE rows mean L6 cannot confirm timing from fresh price history.",
        "action": "Treat L6 as no edge until online or local price data refreshes.",
    })

    adv = read_csv(ROOT / "v8_advanced_risk_summary.csv")
    source = adv["data_source"].iloc[0] if not adv.empty and "data_source" in adv.columns else "UNKNOWN"
    rows.append({
        "status": "WARN" if source == "SYNTHETIC_FALLBACK" else ("OK" if source == "YFINANCE_HISTORY" else "RISK"),
        "area": "V8 Advanced Risk",
        "source": "v8_advanced_risk_summary.csv",
        "signal": "data_source",
        "value": source,
        "detail": "Synthetic fallback validates plumbing but should not be treated as fresh market history.",
        "action": "Use as research-only diagnostics while online history is unavailable.",
    })

    alerts = read_csv(ROOT / "canyon_output_shrinkage_alerts.csv")
    high_or_medium = 0
    if not alerts.empty and "status" in alerts.columns:
        high_or_medium = int(alerts["status"].astype(str).str.upper().isin(["HIGH", "MEDIUM"]).sum())
    rows.append({
        "status": "RISK" if high_or_medium else "OK",
        "area": "Output Vault",
        "source": "canyon_output_shrinkage_alerts.csv",
        "signal": "shrinkage_alerts",
        "value": high_or_medium,
        "detail": "High/medium alerts mean at least one generated file shrank sharply versus the prior vault snapshot.",
        "action": "Review Output Vault before trusting a run with shrinkage alerts.",
    })

    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame) -> None:
    counts = df["status"].value_counts() if not df.empty and "status" in df.columns else pd.Series(dtype=int)
    md = [
        "# Canyon v9 Step 61 - Data Source Health Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Summary",
        f"- OK: `{int(counts.get('OK', 0))}`",
        f"- WARN: `{int(counts.get('WARN', 0))}`",
        f"- RISK: `{int(counts.get('RISK', 0))}`",
        "",
        "## Guardrails",
        "- This is a data-source diagnostic only.",
        "- No broker connection.",
        "- No live orders.",
        "- It does not change model decisions.",
        "",
        "## Health Table",
        "",
        df.to_markdown(index=False) if not df.empty else "_No health rows._",
        "",
    ]
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    print("=" * 88)
    print("CANYON v9 Step 61")
    print("Data Source Health Monitor")
    print("=" * 88)
    df = build_health()
    df.to_csv(OUT_HEALTH, index=False)
    write_report(df)
    counts = df["status"].value_counts() if "status" in df.columns else pd.Series(dtype=int)
    print(f"Rows: {len(df)}")
    print(f"OK={int(counts.get('OK', 0))}, WARN={int(counts.get('WARN', 0))}, RISK={int(counts.get('RISK', 0))}")
    print("Files generated:")
    print(f"  {OUT_HEALTH}")
    print(f"  {OUT_REPORT}")
    print("No broker connection. No live order.")


if __name__ == "__main__":
    main()
