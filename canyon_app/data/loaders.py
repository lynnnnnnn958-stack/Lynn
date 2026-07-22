"""Safe data loaders for Streamlit pages and runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def safe_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file; return an empty DataFrame if missing or broken."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def safe_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object; return an empty dict if missing or broken."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def file_age_label(path: str | Path) -> str:
    """Small user-facing existence label. Full age logic can migrate later."""
    p = Path(path)
    if not p.exists():
        return "Missing"
    if p.stat().st_size == 0:
        return "Empty"
    return "Available"

