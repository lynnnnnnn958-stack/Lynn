"""Performance and signal label translation."""

from __future__ import annotations

from canyon_app.core.formatting import compact_text, to_float


PERFORMANCE_REPLACEMENTS: dict[str, str] = {
    "PROTOTYPE_ONLY": "prototype evidence only",
    "PROTOTYPE ONLY": "prototype evidence only",
    "REPAIR_REQUIRED": "needs repair",
    "REPAIR REQUIRED": "needs repair",
    "EARLY_STAGE": "early stage",
    "EARLY STAGE": "early stage",
    "PENDING_FORWARD_RETURNS": "waiting for future price data",
    "LIVE_IC_ACTIVE": "live signal tracking is active",
    "BLOCK_SIGNAL": "do not use this signal",
    "KEEP_CORE": "keep as a core research signal",
    "USE_ONLY_AT_SHORT_HORIZON": "short-term use only",
    "WAIT_FOR_BASE_REPAIR": "wait until the base is repaired",
    "NEEDS_DEEPER_TEST": "needs deeper testing",
    "REDUCE_ONLY": "risk-reduction only",
    "SIZE_DOWN": "use smaller size",
    "TCA": "trading cost",
    "IC": "predictive skill",
    "PIT": "time-accurate data",
    "OOS": "fresh test windows",
}


SIGNAL_LABELS: dict[str, str] = {
    "mom_12m_skip1m": "12-month momentum",
    "trend_200": "200-day trend",
    "mom_6m": "6-month momentum",
    "mom_3m": "3-month momentum",
    "mom_1m": "1-month momentum",
    "mom_accel": "momentum acceleration",
    "new_high_52w": "52-week high",
    "rsi_rev": "oversold rebound",
    "inv_vol": "low-volatility tilt",
    "eps_growth_yoy": "earnings growth",
}


def performance_plain(value, max_len: int | None = 220) -> str:
    """Translate raw performance/backtest labels into display text."""
    text = "" if value is None else str(value)
    for raw, friendly in PERFORMANCE_REPLACEMENTS.items():
        text = text.replace(raw, friendly)
    return compact_text(text, max_len=max_len)


def signal_label(value) -> str:
    """Return a plain signal name."""
    raw = str(value or "").strip()
    return SIGNAL_LABELS.get(raw, raw.replace("_", " ").strip().title() or "Signal")


def performance_accent(score=None, status: str = "") -> str:
    """Return standard accent color for performance proof state."""
    score_num = to_float(score)
    status_text = str(status or "").upper()
    if any(token in status_text for token in ["BLOCK", "WEAK", "REPAIR", "NOT RELIABLE"]):
        return "#991b1b"
    if score_num is not None:
        if score_num < 45:
            return "#991b1b"
        if score_num < 72:
            return "#334155"
        return "#166534"
    if any(token in status_text for token in ["REVIEW", "PROTOTYPE", "WAIT"]):
        return "#334155"
    return "#166534"

