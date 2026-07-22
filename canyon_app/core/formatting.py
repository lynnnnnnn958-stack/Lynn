"""Plain-English formatting helpers.

These helpers intentionally avoid finance jargon where possible because the
dashboard's default pages should be readable before the user opens technical
details.
"""

from __future__ import annotations

import math
import re
from typing import Any


def to_float(value: Any, default: float | None = None) -> float | None:
    """Convert common CSV/JSON values into a float."""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "").replace("%", "")
            if not cleaned or cleaned.lower() in {"nan", "none", "null", "no data"}:
                return default
            value = cleaned
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def compact_text(value: Any, max_len: int | None = 220) -> str:
    """Return one clean line of display text."""
    if value is None:
        text = "No data"
    else:
        text = str(value)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        text = "No data"
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def pct(value: Any, digits: int = 1, already_pct: bool = True) -> str:
    """Format a percent safely."""
    num = to_float(value)
    if num is None:
        return "No data"
    if not already_pct:
        num *= 100
    return f"{num:.{digits}f}%"


def score(value: Any, digits: int = 1) -> str:
    """Format a 0-100 score."""
    num = to_float(value)
    if num is None:
        return "No data"
    return f"{num:.{digits}f} / 100"


def money(value: Any, digits: int = 0) -> str:
    """Format a dollar value."""
    num = to_float(value)
    if num is None:
        return "No data"
    return f"${num:,.{digits}f}"

