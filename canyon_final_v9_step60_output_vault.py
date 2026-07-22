#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 60 - Output Vault & Shrinkage Audit

Creates a timestamped local snapshot of generated CSV/Markdown outputs and
checks whether current files shrank versus the previous snapshot.

Purpose:
- Preserve reports before a runner can overwrite them.
- Make "things disappeared" visible as explicit alerts.
- Keep this local only. No broker connection. No live order.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import shutil

import pandas as pd

ROOT = Path.cwd()
VAULT_DIR = ROOT / "canyon_output_vault"
GLOBAL_INDEX = ROOT / "canyon_output_vault_index.csv"
ALERTS = ROOT / "canyon_output_shrinkage_alerts.csv"
REPORT = ROOT / "canyon_output_vault_report.md"

EXCLUDE = {
    "canyon_output_vault_index.csv",
    "canyon_output_shrinkage_alerts.csv",
    "canyon_output_vault_report.md",
}


def file_rows(path: Path) -> int:
    if path.suffix.lower() != ".csv" or path.stat().st_size == 0:
        return 0
    try:
        return int(len(pd.read_csv(path, dtype=str)))
    except Exception:
        return 0


def discover_outputs() -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.csv", "*.md"):
        for path in ROOT.glob(suffix):
            if path.name in EXCLUDE:
                continue
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def read_index() -> pd.DataFrame:
    if not GLOBAL_INDEX.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(GLOBAL_INDEX, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def latest_previous_by_file(index: pd.DataFrame, snapshot_id: str) -> dict[str, dict]:
    if index.empty:
        return {}
    prior = index[index["snapshot_id"].astype(str) != snapshot_id].copy()
    if prior.empty:
        return {}
    prior["_ts"] = pd.to_datetime(prior["created_at"], errors="coerce")
    prior = prior.sort_values(["filename", "_ts"])
    out = {}
    for filename, group in prior.groupby("filename"):
        out[str(filename)] = group.iloc[-1].to_dict()
    return out


def make_snapshot(label: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    created = datetime.now()
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "run"
    snapshot_id = f"{created:%Y%m%d_%H%M%S}_{safe_label}"
    snapshot_dir = VAULT_DIR / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    old_index = read_index()
    previous = latest_previous_by_file(old_index, snapshot_id)

    rows = []
    alerts = []
    for src in discover_outputs():
        dest = snapshot_dir / src.name
        shutil.copy2(src, dest)
        size_bytes = int(src.stat().st_size)
        rows_count = file_rows(src)
        modified = datetime.fromtimestamp(src.stat().st_mtime)

        rows.append({
            "snapshot_id": snapshot_id,
            "label": label,
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": src.name,
            "suffix": src.suffix.lower(),
            "size_bytes": size_bytes,
            "rows": rows_count,
            "source_path": str(src),
            "snapshot_path": str(dest),
            "source_modified": modified.strftime("%Y-%m-%d %H:%M:%S"),
        })

        old = previous.get(src.name)
        if old:
            old_size = pd.to_numeric(old.get("size_bytes", 0), errors="coerce")
            old_rows = pd.to_numeric(old.get("rows", 0), errors="coerce")
            old_size = 0 if pd.isna(old_size) else int(old_size)
            old_rows = 0 if pd.isna(old_rows) else int(old_rows)
            size_drop = old_size - size_bytes
            row_drop = old_rows - rows_count
            size_drop_pct = size_drop / old_size if old_size > 0 else 0
            row_drop_pct = row_drop / old_rows if old_rows > 0 else 0
            shrank = size_drop_pct >= 0.50 or row_drop_pct >= 0.50 or (old_rows > 0 and rows_count == 0)
            if shrank:
                severity = "HIGH" if rows_count == 0 and old_rows > 0 else "MEDIUM"
                alerts.append({
                    "status": severity,
                    "filename": src.name,
                    "current_rows": rows_count,
                    "previous_rows": old_rows,
                    "current_size_bytes": size_bytes,
                    "previous_size_bytes": old_size,
                    "row_drop_pct": f"{row_drop_pct:.1%}",
                    "size_drop_pct": f"{size_drop_pct:.1%}",
                    "previous_snapshot": old.get("snapshot_id", ""),
                    "current_snapshot": snapshot_id,
                    "suggested_action": "Review before trusting this run; restore from vault if the shrinkage is accidental.",
                })

    current = pd.DataFrame(rows)
    combined = pd.concat([old_index, current], ignore_index=True) if not old_index.empty else current
    combined.to_csv(GLOBAL_INDEX, index=False)

    alert_df = pd.DataFrame(alerts)
    if alert_df.empty:
        alert_df = pd.DataFrame([{
            "status": "OK",
            "filename": "ALL",
            "current_rows": "",
            "previous_rows": "",
            "current_size_bytes": "",
            "previous_size_bytes": "",
            "row_drop_pct": "",
            "size_drop_pct": "",
            "previous_snapshot": "",
            "current_snapshot": snapshot_id,
            "suggested_action": "No shrinkage alerts versus the previous vault snapshot.",
        }])
    alert_df.to_csv(ALERTS, index=False)

    write_report(current, alert_df, snapshot_dir)
    return current, alert_df, snapshot_dir


def write_report(current: pd.DataFrame, alerts: pd.DataFrame, snapshot_dir: Path) -> None:
    high = int((alerts["status"].astype(str).str.upper() == "HIGH").sum()) if "status" in alerts.columns else 0
    medium = int((alerts["status"].astype(str).str.upper() == "MEDIUM").sum()) if "status" in alerts.columns else 0

    md = [
        "# Canyon v9 Step 60 - Output Vault Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Guardrails",
        "- Local output snapshot only.",
        "- No broker connection.",
        "- No live orders.",
        "- Does not modify trading decisions.",
        "",
        "## Snapshot",
        f"- Folder: `{snapshot_dir}`",
        f"- Files copied: `{len(current)}`",
        f"- Total bytes: `{int(pd.to_numeric(current.get('size_bytes', 0), errors='coerce').fillna(0).sum())}`",
        "",
        "## Shrinkage Alerts",
        f"- HIGH: `{high}`",
        f"- MEDIUM: `{medium}`",
        "",
        alerts.to_markdown(index=False),
        "",
        "## Latest Snapshot Manifest",
        "",
        current[["filename", "suffix", "size_bytes", "rows", "source_modified"]].to_markdown(index=False) if not current.empty else "_No files copied._",
        "",
    ]
    REPORT.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="manual", help="Snapshot label, such as pre-run or post-run.")
    args = parser.parse_args()

    print("=" * 88)
    print("CANYON v9 Step 60")
    print("Output Vault & Shrinkage Audit")
    print("=" * 88)

    current, alerts, snapshot_dir = make_snapshot(args.label)
    print(f"Snapshot: {snapshot_dir}")
    print(f"Files copied: {len(current)}")
    print(f"Alerts: {len(alerts[alerts['status'].astype(str).str.upper().isin(['HIGH', 'MEDIUM'])]) if not alerts.empty else 0}")
    print("Files generated:")
    print(f"  {GLOBAL_INDEX}")
    print(f"  {ALERTS}")
    print(f"  {REPORT}")
    print("No broker connection. No live order.")


if __name__ == "__main__":
    main()
