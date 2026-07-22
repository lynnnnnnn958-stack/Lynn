"""Risk label translation.

The user-facing dashboard should not show raw internal labels such as
``REDUCE_ONLY`` or ``DATA_GAP`` on default pages. Keep the translation here so
Risk, Ideas, Today, and News all speak the same language.
"""

from __future__ import annotations

from canyon_app.core.formatting import compact_text


RISK_REPLACEMENTS: dict[str, str] = {
    "RISK_REPAIR_REQUIRED": "risk repair needed",
    "MANDATORY_REPAIR_TO_RISK_TARGET": "must repair toward risk target",
    "SIZE_DOWN_TO_REPAIR_PATH": "use the repair size path",
    "MASTER_GROSS_70": "70% gross risk-repair scenario",
    "REDUCE_ONLY_LOCKED": "no new buying locked",
    "SIZE_DOWN_LOCKED": "smaller-size locked",
    "STILL_ABOVE_TICKER_RISK_TARGET": "still above ticker risk target",
    "NO_BULLISH_OPTION_RISK_NOT_REPAIRED": "no bullish option while risk is unrepaired",
    "NO_NEW_OPTION": "no option idea yet",
    "RISK_REPAIRED_FOR_MANUAL_REVIEW": "risk repaired enough for manual review",
    "WATCH_ONLY_RISK_STILL_LOCKED": "watch only while risk is locked",
    "RISK_REDUCTION_ONLY": "risk reduction only",
    "PUT_OR_HEDGE_RESEARCH_ONLY": "put or hedge research only",
    "REDUCE_ONLY": "no new buying",
    "SIZE_DOWN": "use smaller size",
    "DATA_GAP": "missing data",
    "MISSING_DATA_REVIEW": "missing data review",
    "NOT_IN_RISK_BOOK_REVIEW": "not in the risk book yet",
    "IV/Greeks/Gamma": "option volatility, Greeks, and gamma",
    "spread/TCA": "trading-cost proof",
    "TCA": "trading cost",
    "option_no_go_checks": "option not-ready checks",
}


def risk_plain(value, max_len: int | None = 180) -> str:
    """Translate a raw risk value into plain English."""
    text = "" if value is None else str(value)
    for raw, friendly in RISK_REPLACEMENTS.items():
        text = text.replace(raw, friendly)
    text = text.replace(";", "; ")
    return compact_text(text, max_len=max_len)


def risk_status_label(value) -> str:
    """Small label for status chips/cards."""
    text = str(value or "").upper()
    if "REDUCE" in text:
        return "No new buying"
    if "SIZE_DOWN" in text:
        return "Use smaller size"
    if "BLOCK" in text:
        return "Blocked"
    if "REVIEW" in text:
        return "Needs review"
    if "CLEAR" in text or "OK" in text:
        return "Looks okay"
    return risk_plain(value, 80)


def risk_accent(value) -> str:
    """Return the standard white-dashboard accent color for risk state."""
    text = str(value or "").upper()
    if any(token in text for token in ["REDUCE", "BLOCK", "HARD", "CRITICAL", "DATA_GAP"]):
        return "#991b1b"
    if any(token in text for token in ["SIZE_DOWN", "REVIEW", "WARNING"]):
        return "#334155"
    if any(token in text for token in ["CLEAR", "OK"]):
        return "#166534"
    return "#111827"

