"""Reusable card models.

The current dashboard renders cards inline with HTML. During migration, page
functions should build ``CardSpec`` objects first, then render them through one
shared component so the whole site stays visually consistent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CardSpec:
    title: str
    value: str
    note: str = ""
    accent: str = "#111827"


def card_html(card: CardSpec) -> str:
    """Return a simple white dashboard card.

    Streamlit rendering remains in the page layer; this helper only standardizes
    markup.
    """
    return f"""
    <div style="background:#fff; border:1px solid #d1d5db; border-top:4px solid {card.accent}; border-radius:8px; padding:14px 15px; min-height:140px;">
      <div style="font-size:12px; color:#64748b; font-weight:900; text-transform:uppercase;">{card.title}</div>
      <div style="font-size:24px; color:#111827; font-weight:950; line-height:1.15; margin-top:7px;">{card.value}</div>
      <div style="font-size:13px; color:#4b5563; line-height:1.38; margin-top:9px;">{card.note}</div>
    </div>
    """

