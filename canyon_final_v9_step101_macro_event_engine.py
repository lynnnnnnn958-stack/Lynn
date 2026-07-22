#!/usr/bin/env python3
"""
Canyon v9 — Step 101: Macro Event Intelligence Engine
=======================================================
Understands WHY things happen, not just WHAT prices say.

Core principle: markets move on CAUSE→EFFECT chains.
  - Trump brings NVDA/GOOGL/AMZN to Saudi Arabia → signing deals → revenue ↑ → bullish
  - Trump announces 25% tariffs on China → AAPL/NKE supply chain → margin ↓ → bearish
  - Fed hawkish pivot → rate-sensitive tech multiples compressed → growth stocks bearish
  - OPEC cut → oil supply shock → XOM/CVX earnings ↑ → energy bullish

This engine:
  1. Fetches real news via yfinance (no API key needed)
  2. Classifies events into macro catalysts with direction and magnitude
  3. Maps catalysts to affected tickers via causal rule library
  4. Scores each ticker: catalyst_score 0-100 (>50 = tailwind, <50 = headwind)
  5. Writes macro_catalyst_scores.csv → consumed by Step 87 alpha aggregator

Outputs:
  macro_catalyst_scores.csv  — per-ticker catalyst score + reason
  macro_event_catalog.json   — structured event log for audit trail

Run: python3 canyon_final_v9_step101_macro_event_engine.py
"""

from __future__ import annotations

import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL RULE LIBRARY
# Each rule maps a pattern of keywords → {event_type, direction, magnitude,
# affected_tickers/sectors, cause_explanation}
#
# Design principle: encode the BUSINESS LOGIC, not just keyword matching.
# The system must know: state visit + tech company = deal flow = revenue.
# ─────────────────────────────────────────────────────────────────────────────

# Sector → tickers mapping (GICS-aligned)
SECTOR_TICKERS: dict[str, list[str]] = {
    "AI_CHIPS":         ["NVDA", "AMD", "AVGO", "MRVL", "QCOM", "MU", "INTC"],
    "CLOUD":            ["MSFT", "GOOGL", "AMZN", "ORCL", "CRM", "META", "SNOW"],
    "DEFENSE":          ["LMT", "RTX", "NOC", "BA", "GD", "HII", "PLTR", "SAIC", "LDOS"],
    "ENERGY_OIL":       ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
    "ENERGY_CLEAN":     ["FSLR", "ENPH", "NEE", "BEP", "RUN", "SEDG", "PLUG"],
    "FINANCIALS":       ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "C", "SCHW"],
    "CONSUMER_IMPORT":  ["AAPL", "NKE", "SBUX", "MCD", "COST", "TGT", "WMT", "DKNG"],
    "SEMIS_CAPITAL":    ["AMAT", "LRCX", "KLAC", "ASML", "TER", "ONTO", "COHU"],
    "AUTO":             ["TSLA", "GM", "F", "RIVN", "LCID", "STLA"],
    # ── PHARMA: large-cap biopharma with FDA-approved commercial pipelines ──
    "PHARMA":           ["LLY", "MRK", "PFE", "ABBV", "BMY", "AMGN", "GILD", "REGN"],
    # ── BIOTECH: clinical-stage / pipeline-dependent (high FDA sensitivity) ──
    "BIOTECH":          ["MRNA", "BNTX", "VRTX", "BIIB", "SGEN", "IONS", "ALNY"],
    # ── MEDTECH: devices, diagnostics, surgical systems ──────────────────────
    "MEDTECH":          ["MDT", "ABT", "SYK", "BSX", "EW", "ISRG", "ZBH", "HOLX"],
    # ── INDUSTRIALS: machinery, aerospace components, electrical equipment ───
    "INDUSTRIALS":      ["CAT", "DE", "HON", "MMM", "GE", "EMR", "ETN", "ITW",
                         "PH", "ROK", "ROP", "FAST"],
    # ── MATERIALS: chemicals, metals, mining, construction materials ─────────
    "MATERIALS":        ["LIN", "APD", "NEM", "FCX", "NUE", "VMC", "MLM", "IP",
                         "PKG", "CF", "MOS"],
    # ── INSURANCE: property/casualty, life, specialty lines ──────────────────
    "INSURANCE":        ["BRK-B", "CB", "AIG", "PGR", "TRV", "AFL", "MET", "PRU"],
    "RATES_SENSITIVE":  ["NVDA", "META", "AMZN", "TSLA", "CRM", "ADBE", "GOOGL",
                         "NFLX", "SHOP"],
    "BANKS_YIELD":      ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "TFC"],
    "REAL_ESTATE":      ["AMT", "PLD", "EQIX", "SPG", "EQR", "PSA", "O", "DLR"],
    "UTILITIES":        ["NEE", "DUK", "SO", "AEP", "XEL", "EXC", "SRE"],
    # ── RETAIL: domestic consumer spend (sensitive to jobs/consumer confidence) ──
    "RETAIL":           ["AMZN", "WMT", "COST", "TGT", "HD", "LOW", "LULU",
                         "TJX", "ROST"],
}

# CEO / Executive → Ticker mapping
# When a CEO appears in a state visit / deal signing context, their company
# is the DIRECT beneficiary.  This is the core insight: Trump brings Jensen Huang
# to Saudi Arabia → NVDA is getting deals, not just "the AI sector".
CEO_TICKER_MAP: dict[str, str] = {
    # ── AI / Chips ────────────────────────────────────────────────────────────
    "jensen huang":       "NVDA",
    "lisa su":            "AMD",
    "hock tan":           "AVGO",
    "pat gelsinger":      "INTC",
    "cristiano amon":     "QCOM",
    "sanjay mehrotra":    "MU",
    # ── Cloud / Software ─────────────────────────────────────────────────────
    "satya nadella":      "MSFT",
    "sundar pichai":      "GOOGL",
    "andy jassy":         "AMZN",
    "mark zuckerberg":    "META",
    "thomas kurian":      "GOOGL",   # Google Cloud CEO
    "adam selipsky":      "AMZN",    # AWS CEO
    "marc benioff":       "CRM",
    "larry ellison":      "ORCL",
    "frank slootman":     "SNOW",    # Snowflake (or successor)
    "sam altman":         "MSFT",    # OpenAI CEO → Microsoft partner
    # ── Defense / Government tech ─────────────────────────────────────────────
    "alex karp":          "PLTR",
    "kathy warden":       "NOC",
    "greg hayes":         "RTX",
    "james taiclet":      "LMT",
    "phebe novakovic":    "GD",
    "christopher mccord": "HII",
    # ── Energy ───────────────────────────────────────────────────────────────
    "darren woods":       "XOM",
    "mike wirth":         "CVX",
    "ryan lance":         "COP",
    # ── Finance / Banks ──────────────────────────────────────────────────────
    "jamie dimon":        "JPM",
    "david solomon":      "GS",
    "ted pick":           "MS",
    "brian moynihan":     "BAC",
    "charlie scharf":     "WFC",
    "larry fink":         "BLK",
    "warren buffett":     "BRK-B",
    # ── Consumer / Retail ────────────────────────────────────────────────────
    "tim cook":           "AAPL",
    "elon musk":          "TSLA",
    "brian niccol":       "SBUX",
    "doug mcmillon":      "WMT",
    "ron vachris":        "COST",
    "ted sarandos":       "NFLX",
    "reed hastings":      "NFLX",
    "bob iger":           "DIS",
    # ── Pharma / Biotech ─────────────────────────────────────────────────────
    "david ricks":        "LLY",
    "albert bourla":      "PFE",
    "rob davis":          "MRK",
    "richard gonzalez":   "ABBV",
    "stephane bancel":    "MRNA",
    "leonard schleifer":  "REGN",
    "reshma kewalramani": "VRTX",
    # ── Industrials / Auto ───────────────────────────────────────────────────
    "jim farley":         "F",
    "mary barra":         "GM",
    "vimal kapur":        "HON",
    "jim umpleby":        "CAT",
    "john may":           "DE",
}


# Company name → ticker mapping
# Catches headlines like "Nvidia raises guidance" without naming Jensen Huang.
# Keys are lowercase company names / common abbreviations.
COMPANY_TICKER_MAP: dict[str, str] = {
    # Mega-cap tech
    "nvidia":       "NVDA",
    "amd":          "AMD",
    "intel":        "INTC",
    "broadcom":     "AVGO",
    "qualcomm":     "QCOM",
    "micron":       "MU",
    "microsoft":    "MSFT",
    "google":       "GOOGL",
    "alphabet":     "GOOGL",
    "amazon":       "AMZN",
    "meta":         "META",
    "facebook":     "META",
    "oracle":       "ORCL",
    "salesforce":   "CRM",
    "snowflake":    "SNOW",
    # Consumer / retail
    "apple":        "AAPL",
    "tesla":        "TSLA",
    "netflix":      "NFLX",
    "starbucks":    "SBUX",
    "walmart":      "WMT",
    "costco":       "COST",
    "target":       "TGT",
    "home depot":   "HD",
    "nike":         "NKE",
    # Energy
    "exxon":        "XOM",
    "exxonmobil":   "XOM",
    "chevron":      "CVX",
    "conocophillips": "COP",
    # Defense
    "lockheed":     "LMT",
    "raytheon":     "RTX",
    "northrop":     "NOC",
    "boeing":       "BA",
    "palantir":     "PLTR",
    # Finance
    "jpmorgan":     "JPM",
    "jp morgan":    "JPM",
    "goldman":      "GS",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "wells fargo":  "WFC",
    "blackrock":    "BLK",
    # Pharma / Biotech
    "eli lilly":    "LLY",
    "lilly":        "LLY",
    "pfizer":       "PFE",
    "merck":        "MRK",
    "abbvie":       "ABBV",
    "moderna":      "MRNA",
    "regeneron":    "REGN",
    "vertex":       "VRTX",
    "gilead":       "GILD",
    # Industrials
    "caterpillar":  "CAT",
    "deere":        "DE",
    "john deere":   "DE",
    "honeywell":    "HON",
    # Auto
    "ford":         "F",
    "gm":           "GM",
    "general motors": "GM",
    "rivian":       "RIVN",
}


def _extract_company_tickers(text: str) -> list[str]:
    """
    Scan text for known company names.  Returns list of tickers.
    Complements _extract_ceo_tickers() — catches headlines that name the
    company directly rather than its CEO.

    Handles possessives ("Apple's", "Microsoft's") and trailing punctuation.
    """
    text_lower = text.lower()
    # Strip possessives so "Apple's" → "Apple " and "Nvidia's" → "Nvidia "
    text_clean = re.sub(r"'s?\b", " ", text_lower)
    padded     = f" {text_clean} "
    found = []
    for name, ticker in COMPANY_TICKER_MAP.items():
        if f" {name} " in padded:
            found.append(ticker)
    return list(set(found))


def _extract_ceo_tickers(text: str) -> list[str]:
    """
    Scan text for known CEO names. Returns list of tickers for CEOs mentioned.
    Used to identify SPECIFIC beneficiaries when a deal headline names executives.
    """
    text_lower = text.lower()
    found = []
    for name, ticker in CEO_TICKER_MAP.items():
        if name in text_lower:
            found.append(ticker)
    return list(set(found))


# Causal event rules
# Each rule: {
#   "keywords":  list of (word/phrase sets, ALL must appear) — e.g. [["trump","tariff"],["china"]]
#   "event_type": str
#   "direction":  "BULLISH" | "BEARISH" | "MIXED"
#   "magnitude":  "HIGH" | "MEDIUM" | "LOW"
#   "sectors":    list of sector keys from SECTOR_TICKERS
#   "tickers":    list of specific tickers (overrides/adds to sectors)
#   "duration_days": int — how long the catalyst stays relevant
#   "cause_chain": str — the causal logic in plain English
# }
CAUSAL_RULES: list[dict] = [

    # ── STATE VISITS / DELEGATIONS (deal flow) ────────────────────────────────
    {
        "keywords":     [["saudi", "deal"], ["nvidia", "nvda"]],
        "event_type":   "STATE_VISIT_DEAL",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      ["NVDA"],
        "sectors":      [],
        "duration_days": 90,
        "cause_chain":  "US-Saudi tech deal → NVDA AI chip orders → revenue growth",
    },
    {
        "keywords":     [["saudi"], ["trump", "president", "visit", "delegation"]],
        "event_type":   "STATE_VISIT_DEAL",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["AI_CHIPS", "CLOUD", "DEFENSE"],
        "duration_days": 60,
        "cause_chain":  "US state visit to Saudi → deal-signing delegation → "
                        "AI/tech/defense contract flow",
    },
    {
        "keywords":     [["trump", "president", "visit", "delegation", "summit"],
                         ["deal", "agreement", "contract", "signed", "billion"]],
        "event_type":   "STATE_VISIT_DEAL",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["DEFENSE", "AI_CHIPS", "CLOUD"],
        "duration_days": 45,
        "cause_chain":  "State visit with trade delegation → deal/contract signing "
                        "→ order flow for accompanying companies",
    },
    {
        "keywords":     [["uae", "emirates", "abu dhabi", "dubai"],
                         ["deal", "investment", "ai", "technology", "data center"]],
        "event_type":   "STATE_VISIT_DEAL",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["AI_CHIPS", "CLOUD"],
        "duration_days": 45,
        "cause_chain":  "UAE AI/tech investment → cloud + chip order flow",
    },
    {
        "keywords":     [["india", "modi"],
                         ["deal", "agreement", "investment", "semiconductor", "defense"]],
        "event_type":   "STATE_VISIT_DEAL",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["AI_CHIPS", "DEFENSE", "SEMIS_CAPITAL"],
        "duration_days": 60,
        "cause_chain":  "US-India strategic deal → semiconductor and defense contracts",
    },

    # ── TARIFFS / TRADE WAR ────────────────────────────────────────────────────
    {
        "keywords":     [["tariff", "tariffs", "trade war", "import tax", "duty"],
                         ["china", "chinese"]],
        "event_type":   "TARIFF_CHINA",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      ["AAPL", "NKE", "SBUX"],
        "sectors":      ["CONSUMER_IMPORT", "AUTO", "SEMIS_CAPITAL"],
        "duration_days": 180,
        "cause_chain":  "China tariffs → supply chain cost ↑ → margin compression "
                        "→ consumer/tech hardware earnings miss risk",
    },
    {
        "keywords":     [["tariff", "tariffs"],
                         ["global", "universal", "all countries", "everyone", "liberation day"]],
        "event_type":   "TARIFF_GLOBAL",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["CONSUMER_IMPORT", "AUTO", "SEMIS_CAPITAL"],
        "duration_days": 120,
        "cause_chain":  "Global tariffs → broad supply chain disruption → "
                        "import-dependent companies face margin/demand headwinds",
    },
    {
        "keywords":     [["tariff", "tariffs"],
                         ["pause", "suspended", "exemption", "waived", "delay"]],
        "event_type":   "TARIFF_PAUSE",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["CONSUMER_IMPORT", "AUTO", "SEMIS_CAPITAL"],
        "duration_days": 90,
        "cause_chain":  "Tariff pause/exemption → supply chain relief → "
                        "margin recovery → previously-tariffed companies rebound",
    },
    {
        "keywords":     [["tariff", "tariffs"],
                         ["semiconductor", "chip", "chips", "nvidia", "advanced"]],
        "event_type":   "TARIFF_CHIPS",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["AI_CHIPS", "SEMIS_CAPITAL"],
        "duration_days": 90,
        "cause_chain":  "Chip export tariffs/controls → US semiconductor companies "
                        "lose China revenue → earnings headwind",
    },

    # ── EXPORT CONTROLS / SANCTIONS ────────────────────────────────────────────
    {
        # Require an IMPOSING word (not "removes/lifts/allows") to avoid false positives
        # when headlines are about LIFTING restrictions
        "keywords":     [["export control", "export ban", "entity list", "blacklist",
                          "bis restriction", "export restriction imposed",
                          "new restrictions", "tighten", "crackdown"],
                         ["china", "semiconductor", "advanced chip", "nvidia a100",
                          "nvidia h100", "nvidia blackwell"]],
        "event_type":   "EXPORT_CONTROL",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      ["NVDA", "AMD", "AMAT", "LRCX", "KLAC"],
        "sectors":      ["AI_CHIPS", "SEMIS_CAPITAL"],
        "duration_days": 180,
        "cause_chain":  "Export controls on advanced chips/equipment → China revenue "
                        "cut off → direct earnings impact for chip companies",
    },
    {
        "keywords":     [["export control", "restriction", "ban"],
                         ["saudi", "uae", "middle east"]],
        "event_type":   "EXPORT_CONTROL_LIFTED",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      ["NVDA", "AMD"],
        "sectors":      ["AI_CHIPS"],
        "duration_days": 180,
        "cause_chain":  "Export restrictions lifted for Middle East → unlocks massive "
                        "AI infrastructure deal flow for chip companies",
    },

    # ── FEDERAL RESERVE / RATES ────────────────────────────────────────────────
    {
        "keywords":     [["fed", "federal reserve", "fomc"],
                         ["rate hike", "raise rates", "hawkish", "higher for longer",
                          "tightening"]],
        "event_type":   "FED_HAWKISH",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "REAL_ESTATE", "UTILITIES"],
        "duration_days": 60,
        "cause_chain":  "Fed hawkish → discount rate ↑ → long-duration growth stock "
                        "DCF multiples compressed → rate-sensitive names de-rate",
    },
    {
        "keywords":     [["fed", "federal reserve", "fomc"],
                         ["rate cut", "lower rates", "dovish", "pivot", "easing"]],
        "event_type":   "FED_DOVISH",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "REAL_ESTATE"],
        "duration_days": 60,
        "cause_chain":  "Fed dovish pivot → lower discount rate → growth stock "
                        "multiples expand → FANG/tech re-rates",
    },

    # ── ENERGY / OPEC ─────────────────────────────────────────────────────────
    {
        "keywords":     [["opec", "opec+", "saudi aramco"],
                         ["cut", "reduce", "production cut", "output cut"]],
        "event_type":   "OPEC_CUT",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["ENERGY_OIL"],
        "duration_days": 90,
        "cause_chain":  "OPEC production cut → oil supply ↓ → oil price ↑ → "
                        "integrated oil company earnings ↑",
    },
    {
        "keywords":     [["opec", "opec+"],
                         ["increase", "production increase", "flood", "market share"]],
        "event_type":   "OPEC_INCREASE",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["ENERGY_OIL"],
        "duration_days": 60,
        "cause_chain":  "OPEC production increase → oil oversupply → oil price ↓ → "
                        "oil company earnings headwind",
    },

    # ── GOVERNMENT CONTRACTS / DEFENSE SPENDING ───────────────────────────────
    {
        "keywords":     [["defense", "military", "pentagon", "dod"],
                         ["contract", "awarded", "billion", "budget increase",
                          "spending", "authorization"]],
        "event_type":   "DEFENSE_CONTRACT",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["DEFENSE"],
        "duration_days": 60,
        "cause_chain":  "Defense contract/budget increase → direct revenue for "
                        "defense primes (LMT/RTX/NOC/BA/PLTR)",
    },
    {
        "keywords":     [["ai", "artificial intelligence"],
                         ["government", "federal", "national security", "dod",
                          "pentagon", "cia", "intelligence"]],
        "event_type":   "GOV_AI_CONTRACT",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      ["PLTR", "MSFT", "GOOGL", "AMZN", "SAIC"],
        "sectors":      [],
        "duration_days": 90,
        "cause_chain":  "Government AI contract → direct revenue for AI/cloud "
                        "companies with federal relationships",
    },

    # ── TECH REGULATORY / ANTITRUST ────────────────────────────────────────────
    {
        "keywords":     [["antitrust", "doj", "ftc", "monopoly", "breakup",
                          "competition investigation"],
                         ["google", "alphabet", "meta", "amazon", "microsoft",
                          "apple", "nvidia"]],
        "event_type":   "ANTITRUST",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["CLOUD"],
        "duration_days": 120,
        "cause_chain":  "Antitrust investigation/ruling → regulatory overhang → "
                        "multiple compression + operational uncertainty",
    },

    # ── CHINA RELATIONS ────────────────────────────────────────────────────────
    {
        # Must contain BOTH a China reference AND a positive/relief keyword
        # to avoid false-positives on tariff-imposing headlines that also mention China
        "keywords":     [["china", "us-china"],
                         ["trade deal", "truce", "de-escalation", "trade talks",
                          "trade agreement", "tariff relief", "tariff removed",
                          "tariff lifted", "tariff suspended", "ceasefire"]],
        "event_type":   "CHINA_TRADE_RELIEF",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      ["AAPL", "NVDA"],
        "sectors":      ["CONSUMER_IMPORT", "AI_CHIPS", "SEMIS_CAPITAL"],
        "duration_days": 90,
        "cause_chain":  "US-China trade de-escalation → supply chain relief + "
                        "China market reopened → revenue recovery for exposed names",
    },
    {
        "keywords":     [["china", "taiwan", "strait"],
                         ["tension", "military", "blockade", "threat",
                          "exercise", "conflict"]],
        "event_type":   "TAIWAN_RISK",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      ["TSM", "NVDA", "AMD", "AVGO", "AAPL"],
        "sectors":      ["AI_CHIPS", "SEMIS_CAPITAL"],
        "duration_days": 60,
        "cause_chain":  "Taiwan tension → semiconductor supply chain risk → "
                        "chip stocks reprice geopolitical premium",
    },

    # ── FDA / DRUG APPROVALS ──────────────────────────────────────────────────
    {
        # Positive FDA decision: approval, clearance, breakthrough designation
        # CEO match handles specific company (e.g. "Eli Lilly" → LLY)
        # Note: "approve" matches both "approved" and "approves" (substring)
        "keywords":     [["fda", "food and drug", "fdca"],
                         ["approve", "approval", "cleared", "510k",
                          "breakthrough therapy", "fast track designation",
                          "accelerated approval", "priority review", "granted",
                          "green light", "greenlight", "positive opinion"]],
        "event_type":   "FDA_APPROVAL",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["PHARMA", "BIOTECH", "MEDTECH"],
        "duration_days": 90,
        "cause_chain":  "FDA approval/designation → drug/device can now be sold → "
                        "direct revenue unlock for the approving company; "
                        "sector re-rates on successful validation of pipeline",
    },
    {
        # Negative FDA decision: rejection, CRL, clinical hold
        "keywords":     [["fda", "food and drug"],
                         ["rejected", "rejection", "complete response letter", "crl",
                          "refuse to file", "clinical hold", "warning letter",
                          "not approved", "approvable issues"]],
        "event_type":   "FDA_REJECTION",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["PHARMA", "BIOTECH"],
        "duration_days": 90,
        "cause_chain":  "FDA rejection/CRL → pipeline asset loses value → "
                        "clinical stage companies de-rate sharply; "
                        "R&D spend on failed drug written down",
    },
    {
        # FDA advisory committee (adcom) vote — leading indicator, not final approval
        "keywords":     [["fda", "advisory committee", "adcom", "advisory panel"],
                         ["voted", "vote", "recommended", "recommendation",
                          "supported", "endorsed"]],
        "event_type":   "FDA_ADCOM_POSITIVE",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["PHARMA", "BIOTECH"],
        "duration_days": 30,
        "cause_chain":  "FDA adcom positive vote → increases probability of final approval "
                        "→ options market prices in ~80% chance → stock re-rates",
    },

    # ── CONGRESSIONAL LEGISLATION ─────────────────────────────────────────────
    {
        # Infrastructure / physical investment bill
        "keywords":     [["congress", "senate", "house", "bipartisan", "signed into law",
                          "passed", "legislation", "act"],
                         ["infrastructure", "roads", "bridges", "broadband",
                          "water infrastructure", "grid modernization",
                          "electric grid", "transit", "rail", "ports"]],
        "event_type":   "CONGRESS_INFRASTRUCTURE",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["INDUSTRIALS", "MATERIALS", "UTILITIES", "ENERGY_CLEAN"],
        "duration_days": 365,
        "cause_chain":  "Infrastructure bill → direct federal spend on construction "
                        "materials, equipment, utilities upgrades → "
                        "multi-year revenue tailwind for industrial/materials names",
    },
    {
        # Defense appropriations / NDAA — annual defense budget
        "keywords":     [["congress", "senate", "house", "ndaa", "appropriations"],
                         ["defense", "military", "pentagon",
                          "authorization act", "defense budget",
                          "defense spending", "armed forces"]],
        "event_type":   "CONGRESS_DEFENSE_BUDGET",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["DEFENSE"],
        "duration_days": 180,
        "cause_chain":  "NDAA/defense appropriations → multi-year contract backlog "
                        "for LMT/RTX/NOC/GD/PLTR → earnings visibility ↑",
    },
    {
        # Clean energy / IRA extensions / climate legislation
        "keywords":     [["congress", "senate", "house", "signed", "passed", "legislation"],
                         ["clean energy", "renewable energy", "solar", "wind",
                          "inflation reduction act", "ira", "climate", "ev credit",
                          "electric vehicle tax credit", "battery storage credit"]],
        "event_type":   "CONGRESS_CLEAN_ENERGY",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      ["TSLA"],
        "sectors":      ["ENERGY_CLEAN", "AUTO"],
        "duration_days": 365,
        "cause_chain":  "Clean energy legislation → tax credits for solar/wind/EV → "
                        "demand stimulus for FSLR/ENPH/TSLA/NEE; "
                        "ITC/PTC extension → project economics improve",
    },
    {
        # Clean energy rollback / IRA repeal (bearish for clean energy)
        "keywords":     [["congress", "senate", "house", "repeal", "eliminate",
                          "rollback", "cut", "remove"],
                         ["inflation reduction act", "ira", "clean energy credit",
                          "ev tax credit", "solar credit", "wind credit",
                          "renewable tax credit"]],
        "event_type":   "CONGRESS_CLEAN_ENERGY_ROLLBACK",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      ["TSLA", "FSLR", "ENPH"],
        "sectors":      ["ENERGY_CLEAN"],
        "duration_days": 180,
        "cause_chain":  "IRA credit rollback → clean energy project economics worsen → "
                        "installation demand drops → solar/wind/EV stocks de-rate",
    },
    {
        # AI policy — US government support / national AI strategy
        "keywords":     [["congress", "senate", "executive order", "signed",
                          "legislation", "act"],
                         ["artificial intelligence", "ai", "machine learning"],
                         ["fund", "invest", "national strategy", "initiative",
                          "billions", "boost", "accelerate", "develop"]],
        "event_type":   "GOV_AI_POLICY_BULLISH",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      ["MSFT", "GOOGL", "NVDA", "AMZN", "META"],
        "sectors":      ["AI_CHIPS", "CLOUD"],
        "duration_days": 120,
        "cause_chain":  "Government AI investment/strategy → demand for AI compute "
                        "and cloud services → hyperscalers and chip companies benefit",
    },
    {
        # AI regulation — restrictions, mandatory safety testing, liability
        "keywords":     [["congress", "senate", "eu", "european union",
                          "legislation", "regulation", "bill"],
                         ["artificial intelligence", "ai", "foundation model",
                          "large language model", "llm"],
                         ["regulate", "ban", "restrict", "liability", "mandatory",
                          "audit", "compliance", "safety testing"]],
        "event_type":   "AI_REGULATION_BEARISH",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      ["MSFT", "GOOGL", "META", "AMZN"],
        "sectors":      ["CLOUD"],
        "duration_days": 120,
        "cause_chain":  "AI regulation → compliance cost + product restrictions → "
                        "slows AI monetization timeline → multiple compression for AI plays",
    },
    {
        # CHIPS Act / semiconductor manufacturing subsidies
        "keywords":     [["chips act", "chips and science act", "semiconductor",
                          "manufacturing grant", "fab", "foundry"],
                         ["grant", "awarded", "billion", "subsidy", "funding",
                          "investment", "domestic manufacturing"]],
        "event_type":   "CHIPS_ACT_FUNDING",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      ["INTC", "TSMC", "MU", "TXN"],
        "sectors":      ["AI_CHIPS", "SEMIS_CAPITAL"],
        "duration_days": 365,
        "cause_chain":  "CHIPS Act grant → subsidizes US fab construction → "
                        "capex partially covered by government → margin improvement "
                        "for US semiconductor manufacturing",
    },

    # ── FEDERAL RESERVE MINUTES ───────────────────────────────────────────────
    {
        # FOMC minutes showing hawkish tone — markets fear more hikes or slower cuts
        "keywords":     [["fomc minutes", "fed minutes", "meeting minutes",
                          "federal reserve minutes"]],
        "event_type":   "FED_MINUTES_HAWKISH",
        "direction":    "BEARISH",
        "magnitude":    "LOW",    # minutes = already-priced meeting; smaller impact
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "REAL_ESTATE", "UTILITIES"],
        "duration_days": 14,
        "cause_chain":  "Fed minutes reveal hawkish committee discussion → "
                        "market re-prices fewer/later rate cuts → "
                        "duration assets and rate-sensitive growth stocks pressured",
    },
    {
        # FOMC minutes showing dovish tone — more cuts expected
        "keywords":     [["fomc minutes", "fed minutes", "meeting minutes",
                          "federal reserve minutes"]],
        "event_type":   "FED_MINUTES_DOVISH",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "REAL_ESTATE"],
        "duration_days": 14,
        "cause_chain":  "Fed minutes show dovish committee majority → "
                        "market prices in earlier/more rate cuts → "
                        "growth multiples expand, real estate unlocked",
    },

    # ── MACRO DATA RELEASES (CPI, JOBS, PMI) ─────────────────────────────────
    {
        # Hot CPI / inflation surprise — hawkish Fed fear
        "keywords":     [["cpi", "consumer price", "inflation", "pce", "ppi"],
                         ["hotter than expected", "above forecast", "beat",
                          "accelerated", "jumped", "rose more", "surprise",
                          "higher than anticipated", "above consensus"]],
        "event_type":   "CPI_HOT",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "REAL_ESTATE", "UTILITIES",
                         "CONSUMER_IMPORT"],
        "duration_days": 21,
        "cause_chain":  "Hot CPI print → Fed stays higher for longer → "
                        "discount rate ↑ → long-duration growth stock DCF compresses; "
                        "consumer discretionary hurt by spending power squeeze",
    },
    {
        # Cool CPI / inflation below expectations — dovish pivot hope
        "keywords":     [["cpi", "consumer price", "inflation", "pce", "ppi"],
                         ["cooled", "slowed", "below forecast", "missed",
                          "decelerated", "tame", "lower than expected",
                          "below consensus", "softer than", "eased"]],
        "event_type":   "CPI_COOL",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "REAL_ESTATE", "RETAIL"],
        "duration_days": 21,
        "cause_chain":  "Cool CPI print → Fed more likely to cut rates → "
                        "growth multiples expand; consumer real income ↑ "
                        "→ retail/discretionary spending recovers",
    },
    {
        # Strong jobs / payrolls beat — dual signal (good for economy, hawkish risk)
        "keywords":     [["nonfarm payrolls", "jobs report", "payrolls", "employment",
                          "jobless", "unemployment"],
                         ["beat", "blowout", "stronger than expected", "added",
                          "surged", "blew past", "above forecast",
                          "labor market strong", "jobs boom"]],
        "event_type":   "JOBS_STRONG",
        "direction":    "MIXED",     # good for economy, but hawkish fear
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["FINANCIALS", "RETAIL", "CONSUMER_IMPORT"],
        "duration_days": 14,
        "cause_chain":  "Strong payrolls → consumer spending power intact → "
                        "retail/consumer revenue supported; "
                        "BUT tight labor market → Fed delays cuts → growth stocks hedged",
    },
    {
        # Weak jobs / payrolls miss — recession risk or dovish pivot hope
        "keywords":     [["nonfarm payrolls", "jobs report", "payrolls", "employment",
                          "jobless claims"],
                         ["missed", "weaker than expected", "fell", "below forecast",
                          "slowdown", "layoffs", "rising unemployment",
                          "labor market cools", "job losses"]],
        "event_type":   "JOBS_WEAK",
        "direction":    "BEARISH",   # primary read = recession risk
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["FINANCIALS", "RETAIL", "CONSUMER_IMPORT", "INDUSTRIALS"],
        "duration_days": 14,
        "cause_chain":  "Weak payrolls → consumer income risk → retail/discretionary "
                        "revenue threatened; financial sector loan books at risk; "
                        "HOWEVER: may accelerate Fed cutting cycle",
    },
    {
        # Strong manufacturing PMI — industrial/commodity bullish
        "keywords":     [["pmi", "purchasing managers", "ism manufacturing",
                          "manufacturing index", "china pmi", "caixin pmi"],
                         ["expansion", "beat", "above 50", "rose", "surged",
                          "accelerated", "stronger than expected"]],
        "event_type":   "PMI_STRONG",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["INDUSTRIALS", "MATERIALS", "ENERGY_OIL"],
        "duration_days": 14,
        "cause_chain":  "Strong PMI → manufacturing activity expanding → "
                        "demand for industrial inputs (materials, energy, equipment) ↑; "
                        "China PMI recovery → commodity cycle + supply chain improvement",
    },
    {
        # Weak PMI — industrial contraction, commodity demand falls
        "keywords":     [["pmi", "purchasing managers", "ism manufacturing",
                          "manufacturing index", "china pmi", "caixin pmi"],
                         ["contraction", "below 50", "missed", "slowed",
                          "weaker than expected", "decelerated",
                          "manufacturing slump"]],
        "event_type":   "PMI_WEAK",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["INDUSTRIALS", "MATERIALS", "ENERGY_OIL"],
        "duration_days": 14,
        "cause_chain":  "Weak PMI → manufacturing contraction → "
                        "capex cuts → industrial equipment and materials demand falls; "
                        "commodity prices pressured",
    },

    # ── BANK STRESS TESTS (Fed DFAST) ─────────────────────────────────────────
    {
        # Banks pass Fed stress test → capital return (buybacks/dividends) approved
        "keywords":     [["stress test", "dfast", "fed stress", "ccar",
                          "capital plan"],
                         ["passed", "pass", "approved", "cleared", "adequate capital",
                          "buyback", "dividend increase", "capital return",
                          "no objection"]],
        "event_type":   "BANK_STRESS_PASS",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["BANKS_YIELD", "FINANCIALS"],
        "duration_days": 60,
        "cause_chain":  "Fed stress test pass → banks cleared for capital return → "
                        "buyback/dividend announcements follow → "
                        "bank stocks re-rate on improved shareholder return visibility",
    },
    {
        # Banks fail or weakly pass — capital return restricted
        "keywords":     [["stress test", "dfast", "fed stress", "ccar"],
                         ["failed", "fail", "objected", "restricted",
                          "capital deficiency", "remediation", "concern",
                          "below threshold"]],
        "event_type":   "BANK_STRESS_FAIL",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["BANKS_YIELD"],
        "duration_days": 60,
        "cause_chain":  "Fed stress test objection → bank restricted from buybacks/dividends → "
                        "regulatory overhang + capital uncertainty → banks de-rate",
    },

    # ── NATO / EUROPEAN DEFENSE SPENDING ─────────────────────────────────────
    {
        "keywords":     [["nato", "european", "europe", "germany", "uk", "france",
                          "defence", "defense spending", "rearmament"],
                         ["increase", "boost", "surge", "gdp target", "expand",
                          "ramp up", "billions", "largest", "historic"]],
        "event_type":   "NATO_DEFENSE_SURGE",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["DEFENSE"],
        "duration_days": 180,
        "cause_chain":  "NATO/European defense spending surge → export orders for "
                        "US defense primes (LMT/RTX/NOC/GD) and PLTR NATO contracts; "
                        "European rearmament = multi-year order backlog",
    },

    # ── CREDIT RATING / SOVEREIGN EVENTS ─────────────────────────────────────
    {
        # US or major sovereign downgrade
        "keywords":     [["moody", "s&p", "fitch", "credit rating", "downgrade",
                          "rating action"],
                         ["united states", "us debt", "treasury", "sovereign",
                          "aaa", "aa+", "downgraded", "outlook negative"]],
        "event_type":   "SOVEREIGN_DOWNGRADE",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["FINANCIALS", "BANKS_YIELD", "RATES_SENSITIVE",
                         "REAL_ESTATE"],
        "duration_days": 60,
        "cause_chain":  "US sovereign downgrade → Treasury yields ↑ (risk premium) → "
                        "financing costs rise for all rate-sensitive borrowers; "
                        "bank book values pressured on held-to-maturity losses",
    },
    {
        # Debt ceiling deal — fiscal risk resolved
        "keywords":     [["debt ceiling", "debt limit", "x date", "default"],
                         ["deal", "agreement", "raised", "suspended", "resolved",
                          "passed", "averted", "signed"]],
        "event_type":   "DEBT_CEILING_RESOLVED",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["FINANCIALS", "RATES_SENSITIVE"],
        "duration_days": 30,
        "cause_chain":  "Debt ceiling resolved → systemic default risk removed → "
                        "Treasury market normalizes → risk-off unwind → "
                        "equities rebound on reduced fiscal uncertainty",
    },
    {
        # Debt ceiling impasse — fiscal cliff risk
        "keywords":     [["debt ceiling", "debt limit"],
                         ["approaching", "impasse", "default risk", "brinkmanship",
                          "shutdown", "x date", "warning", "standoff",
                          "no deal", "stalled"]],
        "event_type":   "DEBT_CEILING_RISK",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["FINANCIALS", "RATES_SENSITIVE", "BANKS_YIELD"],
        "duration_days": 30,
        "cause_chain":  "Debt ceiling impasse → US default risk priced in → "
                        "Treasury yields spike + dollar weakens → "
                        "broad equity de-rating until resolved",
    },

    # ── CORPORATE ACTIONS / EARNINGS SIGNALS ─────────────────────────────────
    {
        # Positive earnings preannouncement / guidance raise
        "keywords":     [["preannounce", "pre-announce", "raised guidance",
                          "raises guidance", "raised outlook", "ahead of expectations",
                          "preliminary results"],
                         ["beat", "above", "raise", "increased", "stronger",
                          "exceeded", "outperformed", "upside"]],
        "event_type":   "EARNINGS_PREANNOUNCE_POSITIVE",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 21,
        "cause_chain":  "Positive earnings preannouncement → current quarter tracking "
                        "above consensus → earnings revision cycle starts → "
                        "stock re-rates before official earnings date",
    },
    {
        # Profit warning / guidance cut
        "keywords":     [["profit warning", "guidance cut", "lowered guidance",
                          "reduces guidance", "below expectations",
                          "preannounce", "pre-announce"],
                         ["miss", "shortfall", "below", "cut", "reduced",
                          "disappointed", "warns", "revision lower"]],
        "event_type":   "EARNINGS_PREANNOUNCE_NEGATIVE",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 30,
        "cause_chain":  "Profit warning → earnings revision cycle to downside → "
                        "consensus estimates cut → stock de-rates to new fair value; "
                        "sector contagion if macro-driven warning",
    },
    {
        # Large buyback announcement
        "keywords":     [["buyback", "share repurchase", "stock repurchase",
                          "repurchase program"],
                         ["billion", "authorized", "announced", "approved",
                          "expanded", "accelerated"]],
        "event_type":   "BUYBACK_ANNOUNCEMENT",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 30,
        "cause_chain":  "Large buyback authorization → EPS accretion + signal of "
                        "management confidence in intrinsic value → "
                        "float reduction over time supports share price",
    },

    # ── TECH-SPECIFIC ──────────────────────────────────────────────────────────
    {
        # Major AI model release / breakthrough (ChatGPT moment, GPT-5, Gemini Ultra, etc.)
        "keywords":     [["openai", "anthropic", "google deepmind", "meta ai",
                          "gpt-5", "gpt5", "gemini", "claude", "llama"],
                         ["released", "launch", "introduced", "unveiled",
                          "breakthrough", "surpassed", "new model",
                          "state of the art"]],
        "event_type":   "AI_MODEL_RELEASE",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      ["MSFT", "GOOGL", "META", "NVDA", "AMZN"],
        "sectors":      ["AI_CHIPS", "CLOUD"],
        "duration_days": 45,
        "cause_chain":  "Major AI model release → compute demand surge → "
                        "NVDA GPU orders ↑; cloud provider inference revenue ↑; "
                        "new capabilities drive enterprise AI adoption cycle",
    },
    {
        # Quantum computing breakthrough
        "keywords":     [["quantum", "quantum computing", "qubit"],
                         ["breakthrough", "milestone", "advantage", "error correction",
                          "fault tolerant", "commercial", "demonstration"]],
        "event_type":   "QUANTUM_BREAKTHROUGH",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      ["GOOGL", "IBM", "MSFT", "IONQ", "RGTI"],
        "sectors":      [],
        "duration_days": 60,
        "cause_chain":  "Quantum breakthrough → accelerates commercial timeline → "
                        "narrative-driven re-rate for quantum-adjacent names",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # M&A / CORPORATE TRANSACTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    {
        # M&A deal announced — target surges, sector consolidation signals pricing power
        # Company/CEO name detection identifies the specific target and acquirer
        "keywords":     [["acqui", "merger", "takeover", "buyout", "going private"],
                         ["billion", "agreed to buy", "agreed to acquire",
                          "deal to acquire", "purchase agreement",
                          "definitive agreement", "to be acquired",
                          "all-cash deal", "all-stock deal"]],
        "event_type":   "MA_DEAL_ANNOUNCED",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 90,
        "cause_chain":  "M&A announced → target trades to deal price (20-40% premium) → "
                        "sector consolidation signals pricing power → peers re-rate",
    },
    {
        # DOJ/FTC blocks deal — target drops back to standalone; acquirer mixed
        "keywords":     [["doj", "ftc", "justice department", "antitrust division",
                          "competition authority"],
                         ["block", "blocked", "suing to block", "challenge",
                          "oppose", "halt the merger", "injunction",
                          "prevent the deal", "illegal merger"]],
        "event_type":   "MA_DEAL_BLOCKED",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 30,
        "cause_chain":  "Antitrust block → deal dies → target drops 20-40% back to "
                        "standalone value; acquirer freed but M&A strategy questioned",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CLINICAL TRIALS / DRUG DEVELOPMENT
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["phase 3", "phase iii", "phase 2", "phase ii",
                          "pivotal trial", "clinical trial data",
                          "trial results", "clinical data readout"]],
        "event_type":   "CLINICAL_TRIAL_POSITIVE",  # matched alone, direction from outcome kws
        # Two-group design: phase keyword + positive outcome
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["BIOTECH", "PHARMA"],
        "duration_days": 60,
        "cause_chain":  "Phase 2/3 trial success → asset de-risked → FDA approval path "
                        "opened → company re-rates on new revenue potential",
    },
    {
        "keywords":     [["phase 3 failed", "phase iii failed", "phase 2 failed",
                          "trial failed", "failed phase",
                          "primary endpoint not met", "did not meet primary",
                          "negative trial", "trial halted"]],
        "event_type":   "CLINICAL_TRIAL_NEGATIVE",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["BIOTECH"],
        "duration_days": 30,
        "cause_chain":  "Phase 3 failure → pipeline asset written to zero → "
                        "clinical-stage biotech can fall 50-80% on single asset failure",
    },
    {
        # Phase 3 positive with explicit "met primary endpoint"
        "keywords":     [["phase 3", "phase iii", "clinical trial", "pivotal study"],
                         ["met primary endpoint", "statistically significant",
                          "demonstrated efficacy", "superior to placebo",
                          "significant reduction", "positive results confirmed"]],
        "event_type":   "CLINICAL_TRIAL_POSITIVE",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      ["BIOTECH", "PHARMA"],
        "duration_days": 60,
        "cause_chain":  "Phase 3 meets primary endpoint → drug approval near-certain → "
                        "company peak-sales potential now priced in → re-rate",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # GDP / ECONOMIC GROWTH
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["gdp", "gross domestic product", "economic growth",
                          "quarterly growth"]],
        "event_type":   "GDP_STRONG",
        "direction":    "MIXED",    # strong growth = good for earners, bad for rate-sensitive
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["FINANCIALS", "INDUSTRIALS", "RETAIL"],
        "duration_days": 21,
        "cause_chain":  "GDP beat → economic expansion intact → financial/industrial revenue ↑; "
                        "BUT may delay Fed cuts → rate-sensitive stocks net neutral",
    },
    {
        "keywords":     [["gdp", "gross domestic product"],
                         ["contracted", "recession", "shrank", "negative growth",
                          "below forecast", "gdp miss", "gdp slowed", "gdp declined",
                          "technical recession", "two quarters"]],
        "event_type":   "GDP_WEAK",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["FINANCIALS", "INDUSTRIALS", "RETAIL", "ENERGY_OIL"],
        "duration_days": 21,
        "cause_chain":  "GDP miss/contraction → recession risk elevated → "
                        "cyclical earnings at risk; financials face loan loss provisions; "
                        "industrials/energy face demand destruction",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # RETAIL SALES & HOUSING DATA
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["retail sales", "consumer spending", "personal spending",
                          "personal consumption"]],
        "event_type":   "RETAIL_SALES_STRONG",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["RETAIL", "CONSUMER_IMPORT"],
        "duration_days": 14,
        "cause_chain":  "Strong retail sales → consumer spending power intact → "
                        "direct revenue uplift for retailers; income/employment health confirmed",
    },
    {
        "keywords":     [["retail sales", "consumer spending"],
                         ["fell", "declined", "missed", "below forecast",
                          "unexpected drop", "weakened", "consumer pullback",
                          "spending slump"]],
        "event_type":   "RETAIL_SALES_WEAK",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["RETAIL", "CONSUMER_IMPORT"],
        "duration_days": 14,
        "cause_chain":  "Weak retail sales → consumer spending contracting → "
                        "retailer same-store sales at risk; inventory build → margin pressure",
    },
    {
        "keywords":     [["housing starts", "building permits", "existing home sales",
                          "new home sales", "pending home sales",
                          "home sales", "housing data"]],
        "event_type":   "HOUSING_STRONG",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["REAL_ESTATE", "MATERIALS", "INDUSTRIALS"],
        "duration_days": 14,
        "cause_chain":  "Strong housing data → construction activity ↑ → "
                        "demand for lumber/cement/copper rises; "
                        "homebuilders and materials suppliers benefit",
    },
    {
        "keywords":     [["housing starts", "home sales", "housing market",
                          "building permits"],
                         ["fell", "lowest since", "slumped", "dropped", "missed",
                          "weakest", "below forecast", "housing slowdown"]],
        "event_type":   "HOUSING_WEAK",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["REAL_ESTATE", "MATERIALS"],
        "duration_days": 14,
        "cause_chain":  "Weak housing data → construction pipeline shrinks → "
                        "materials demand falls; REITs face occupancy headwind",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # OIL & ENERGY SUPPLY
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["oil", "crude", "petroleum"],
                         ["supply disruption", "pipeline attack", "tanker seized",
                          "strait of hormuz", "houthi attack", "iran sanctions",
                          "russia oil ban", "production halt", "hurricane gulf",
                          "force majeure", "oil supply shock"]],
        "event_type":   "OIL_SUPPLY_DISRUPTION",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["ENERGY_OIL"],
        "duration_days": 30,
        "cause_chain":  "Oil supply disruption → crude price spike → "
                        "E&P and integrated oil margins expand; "
                        "airlines/transportation face cost headwind",
    },
    {
        "keywords":     [["oil", "crude", "brent", "wti", "petroleum price"],
                         ["crashed", "plunged", "demand destruction", "oil glut",
                          "storage overflow", "price war", "opec floods",
                          "oil price fell", "oil price slump"]],
        "event_type":   "OIL_PRICE_CRASH",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["ENERGY_OIL"],
        "duration_days": 30,
        "cause_chain":  "Oil price crash → upstream producer revenue drops → "
                        "E&P capex cuts → integrated oil earnings miss; "
                        "airlines/consumer benefit from lower fuel costs",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # REGULATORY / LEGAL
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["sec", "securities and exchange commission",
                          "securities fraud", "accounting fraud", "restatement",
                          "sec investigation", "sec charges"]],
        "event_type":   "SEC_INVESTIGATION",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 90,
        "cause_chain":  "SEC investigation/fraud charges → management credibility crisis → "
                        "earnings quality questioned → institutional selling; "
                        "fraud charges can drop a stock 20-50% immediately",
    },
    {
        "keywords":     [["hindenburg research", "muddy waters", "citron research",
                          "gotham city research", "short report", "short seller report",
                          "activist short"],
                         ["fraud", "misleading", "accounting manipulation",
                          "fabricated", "overvalued", "scheme",
                          "questions", "alleges", "exposes"]],
        "event_type":   "SHORT_SELLER_REPORT",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 30,
        "cause_chain":  "Short seller fraud report → immediate 20-50% drop on publication; "
                        "even if disproven, scrutiny + institutional redemptions cause damage",
    },
    {
        "keywords":     [["recall", "nhtsa recall", "product recall",
                          "voluntary recall"],
                         ["million units", "million vehicles", "safety defect",
                          "fire risk", "injury risk", "class action",
                          "recall investigation"]],
        "event_type":   "PRODUCT_RECALL",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["AUTO"],
        "duration_days": 30,
        "cause_chain":  "Product recall → direct cost + legal liability + brand damage → "
                        "earnings headwind; large auto recalls = $1-5B event",
    },
    {
        "keywords":     [["strike", "walkout", "work stoppage", "labor dispute"],
                         ["workers strike", "union strike", "uaw strike",
                          "production halted", "plant shut", "factory shut",
                          "employees walked"]],
        "event_type":   "LABOR_STRIKE",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["AUTO", "INDUSTRIALS"],
        "duration_days": 30,
        "cause_chain":  "Labor strike → production halt → direct revenue loss; "
                        "auto plant strike ≈ $500M-1B/week; "
                        "resolution risks higher wage settlement → structural cost ↑",
    },
    {
        "keywords":     [["ceo", "chief executive", "cfo", "chief financial officer"],
                         ["resign", "resigned", "stepping down", "departure announced",
                          "fired", "terminated without cause",
                          "abruptly", "effective immediately", "sudden exit"]],
        "event_type":   "EXECUTIVE_DEPARTURE",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 14,
        "cause_chain":  "Unexpected CEO/CFO departure → leadership uncertainty → "
                        "strategy continuity risk → institutional selling pressure",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # INDEX MEMBERSHIP CHANGES
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["s&p 500", "s&p500", "sp 500", "sp500",
                          "standard and poor"],
                         ["added", "inclusion", "joins the s&p",
                          "will join", "effective date", "new constituent",
                          "being added to"]],
        "event_type":   "SP500_INCLUSION",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 30,
        "cause_chain":  "S&P 500 inclusion → forced buying from ~$7T passive funds → "
                        "guaranteed demand; stock typically +3-5% from announcement to effective",
    },
    {
        "keywords":     [["s&p 500", "s&p500", "sp 500"],
                         ["removed", "deletion", "excluded", "will be removed",
                          "dropped from", "no longer in",
                          "replaced in the index"]],
        "event_type":   "SP500_EXCLUSION",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      [],
        "duration_days": 30,
        "cause_chain":  "S&P 500 exclusion → forced selling from passive funds → "
                        "guaranteed supply overhang; signals company below size threshold",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AI & TECH INFRASTRUCTURE
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["microsoft azure", "amazon aws", "google cloud",
                          "meta platforms", "hyperscaler", "cloud provider"],
                         ["capex", "capital expenditure", "data center investment",
                          "ai infrastructure", "accelerating investment",
                          "billion in data centers", "raised capex guidance",
                          "record spending", "expanding data center"]],
        "event_type":   "HYPERSCALER_CAPEX_RAISE",
        "direction":    "BULLISH",
        "magnitude":    "HIGH",
        "tickers":      ["NVDA", "AMD", "AVGO", "MRVL"],
        "sectors":      ["AI_CHIPS", "SEMIS_CAPITAL"],
        "duration_days": 90,
        "cause_chain":  "Hyperscaler raises AI capex → GPU demand surge (NVDA H100/B200) → "
                        "chip lead times extend → pricing power for NVDA/AMD; "
                        "networking semis (MRVL/AVGO) benefit from fabric buildout",
    },
    {
        "keywords":     [["microsoft azure", "amazon aws", "google cloud",
                          "hyperscaler"],
                         ["cut capex", "reduce spending", "pause data center",
                          "scale back", "cancel orders", "lower capital expenditure",
                          "capex concern", "ai spending pause"]],
        "event_type":   "HYPERSCALER_CAPEX_CUT",
        "direction":    "BEARISH",
        "magnitude":    "HIGH",
        "tickers":      ["NVDA", "AMD", "AVGO"],
        "sectors":      ["AI_CHIPS", "SEMIS_CAPITAL"],
        "duration_days": 60,
        "cause_chain":  "Hyperscaler cuts AI capex → GPU demand signal weakens → "
                        "NVDA/AMD forward order concern → chip stocks de-rate on uncertainty",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # HEALTHCARE POLICY
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["medicare", "drug pricing", "drug price negotiation",
                          "cms", "centers for medicare", "drug price reform",
                          "drug prices", "prescription drug"],
                         ["negotiate", "negotiation", "price control",
                          "price setting", "lower drug prices",
                          "selected for negotiation", "fair price"]],
        "event_type":   "DRUG_PRICE_NEGOTIATION",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["PHARMA"],
        "duration_days": 120,
        "cause_chain":  "Medicare drug price negotiation → direct revenue cut → "
                        "selected drugs face 40-80% price reduction → margin compression",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CRYPTOCURRENCY
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["bitcoin", "crypto", "cryptocurrency", "btc", "ethereum"],
                         ["spot etf approved", "spot bitcoin etf", "strategic reserve",
                          "legal tender", "clear regulatory framework",
                          "crypto friendly", "national bitcoin reserve"]],
        "event_type":   "CRYPTO_REGULATION_BULLISH",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      ["COIN", "MSTR", "MARA", "RIOT", "CLSK"],
        "sectors":      [],
        "duration_days": 45,
        "cause_chain":  "Crypto regulatory clarity → institutional adoption accelerates → "
                        "crypto-adjacent stocks (COIN/MSTR) benefit from narrative",
    },
    {
        "keywords":     [["crypto exchange", "cryptocurrency", "bitcoin exchange",
                          "crypto platform"],
                         ["banned", "criminal charges", "hacked", "collapsed",
                          "bankrupt", "fraud investigation", "seizure",
                          "shutdown", "exit scam"]],
        "event_type":   "CRYPTO_CRACKDOWN",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      ["COIN", "MSTR", "MARA", "RIOT"],
        "sectors":      [],
        "duration_days": 30,
        "cause_chain":  "Crypto crackdown/collapse → confidence shock → "
                        "contagion to crypto-adjacent stocks; "
                        "systemic collapse (FTX-style) triggers broader risk-off",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # GEOPOLITICAL
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["russia", "ukraine", "north korea", "iran", "middle east",
                          "israel", "hamas", "hezbollah", "south china sea"],
                         ["escalat", "attack launched", "invasion", "missile strike",
                          "troops cross", "crisis escalates", "nuclear threat",
                          "military offensive", "bombing"]],
        "event_type":   "GEOPOLITICAL_ESCALATION",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["ENERGY_OIL", "DEFENSE"],
        "duration_days": 30,
        "cause_chain":  "Geopolitical escalation → risk-off selling + oil supply risk → "
                        "energy stocks paradoxically benefit; defense stocks gain on "
                        "rearmament; broad equity de-rates on uncertainty premium",
    },
    {
        "keywords":     [["russia", "ukraine", "middle east", "israel", "iran",
                          "north korea"],
                         ["ceasefire", "peace deal", "truce announced",
                          "withdrawal begins", "de-escalation agreement",
                          "negotiations succeed", "talks resume", "peace talks"]],
        "event_type":   "GEOPOLITICAL_DE_ESCALATION",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "CONSUMER_IMPORT", "ENERGY_CLEAN"],
        "duration_days": 30,
        "cause_chain":  "Geopolitical de-escalation → risk premium unwinds → "
                        "growth stocks re-rate; supply chain uncertainty eases → "
                        "consumer goods recover; clean energy unlocked",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # US DOLLAR REGIME
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["dollar", "dxy", "dollar index", "usd"],
                         ["surged", "strengthened", "multi-year high", "strong dollar",
                          "dollar rally", "dollar jumped", "dollar climbed"]],
        "event_type":   "DOLLAR_STRENGTHENING",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "CONSUMER_IMPORT"],
        "duration_days": 21,
        "cause_chain":  "Strong dollar → overseas revenue translates at worse rate → "
                        "EPS headwind for multinationals with >50% international revenue "
                        "(AAPL, MSFT, GOOGL); USD-denominated commodities become pricier",
    },
    {
        "keywords":     [["dollar", "dxy", "dollar index", "usd"],
                         ["weakened", "falling dollar", "dollar dropped",
                          "dollar weakness", "dollar decline", "soft dollar",
                          "dollar near lows", "dollar plunged"]],
        "event_type":   "DOLLAR_WEAKENING",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["RATES_SENSITIVE", "MATERIALS", "ENERGY_OIL"],
        "duration_days": 21,
        "cause_chain":  "Weak dollar → overseas revenue repatriated at better rate → "
                        "EPS tailwind for multinational tech; "
                        "commodities (gold, oil, copper) see demand from non-USD buyers ↑",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # BANKING REGULATION
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["basel iii endgame", "basel iv", "capital requirements",
                          "bank capital rules", "tier 1 capital"],
                         ["finalized", "higher requirements", "tighter", "more capital",
                          "proposed rule", "final rule", "new capital standards"]],
        "event_type":   "BANK_REGULATION_TIGHTER",
        "direction":    "BEARISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["BANKS_YIELD", "FINANCIALS"],
        "duration_days": 90,
        "cause_chain":  "Tighter capital rules → banks hold more equity → ROE compresses → "
                        "buyback/dividend capacity reduced; smaller banks most impacted",
    },
    {
        "keywords":     [["bank regulation", "banking deregulation", "capital rules",
                          "banking rules", "bank capital"],
                         ["eased", "loosened", "reduced", "deregulation",
                          "relief", "relaxed rules", "less capital required",
                          "lower requirements"]],
        "event_type":   "BANK_REGULATION_EASED",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      [],
        "sectors":      ["BANKS_YIELD", "FINANCIALS"],
        "duration_days": 90,
        "cause_chain":  "Capital relief → freed equity → increased buyback/dividend capacity → "
                        "bank ROE improves → bank stocks re-rate",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # ENERGY TRANSITION
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["lng", "liquefied natural gas", "lng terminal",
                          "natural gas export"],
                         ["approved", "approval", "permit granted",
                          "export license", "construction permit",
                          "doe approved", "ferc approved"]],
        "event_type":   "LNG_EXPORT_APPROVED",
        "direction":    "BULLISH",
        "magnitude":    "MEDIUM",
        "tickers":      ["LNG", "AR", "EQT", "COP", "XOM"],
        "sectors":      ["ENERGY_OIL"],
        "duration_days": 180,
        "cause_chain":  "LNG export terminal approved → US gas producers gain export market → "
                        "domestic gas prices ↑ → E&P revenues improve; "
                        "Cheniere (LNG) volumes increase directly",
    },
    {
        "keywords":     [["nuclear", "nuclear power", "nuclear plant",
                          "smr", "small modular reactor", "nuclear reactor"],
                         ["restart", "approved", "license granted",
                          "comes online", "reopened", "new plant approved",
                          "construction begins"]],
        "event_type":   "NUCLEAR_APPROVED",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      ["CCJ", "NNE", "SMR", "BWX", "OKLO"],
        "sectors":      ["UTILITIES", "ENERGY_CLEAN"],
        "duration_days": 180,
        "cause_chain":  "Nuclear approval/restart → uranium demand ↑ (CCJ); "
                        "SMR developers re-rate on regulatory clarity; "
                        "utilities with nuclear benefit from clean energy premium",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CONSUMER CONFIDENCE
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "keywords":     [["consumer confidence", "consumer sentiment",
                          "university of michigan", "conference board consumer"],
                         ["beat", "rose", "highest", "above forecast",
                          "improved significantly", "surged", "beat expectations",
                          "multi-year high"]],
        "event_type":   "CONSUMER_CONFIDENCE_STRONG",
        "direction":    "BULLISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["RETAIL", "CONSUMER_IMPORT"],
        "duration_days": 14,
        "cause_chain":  "High consumer confidence → forward spending intentions ↑ → "
                        "retailers and discretionary names benefit from demand signal",
    },
    {
        "keywords":     [["consumer confidence", "consumer sentiment",
                          "university of michigan", "conference board consumer"],
                         ["fell", "lowest", "plunged", "weakest",
                          "below forecast", "multi-year low",
                          "collapsed", "sharply declined"]],
        "event_type":   "CONSUMER_CONFIDENCE_WEAK",
        "direction":    "BEARISH",
        "magnitude":    "LOW",
        "tickers":      [],
        "sectors":      ["RETAIL", "CONSUMER_IMPORT"],
        "duration_days": 14,
        "cause_chain":  "Weak consumer confidence → spending caution → "
                        "discretionary purchases deferred → retailer demand headwind",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# NEWS FETCHER — yfinance, no API key required
# ─────────────────────────────────────────────────────────────────────────────

def fetch_market_news(tickers: list[str],
                      max_headlines: int = 200) -> list[dict]:
    """
    Fetch recent news headlines for a list of tickers via yfinance.
    Returns list of {title, ticker, published, link}.
    """
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_news: list[dict] = []

    def _fetch_one(t: str) -> list[dict]:
        try:
            ticker_obj = yf.Ticker(t)
            news_items = ticker_obj.news or []
            results = []
            for item in news_items[:10]:
                # yfinance ≥0.2.50 nests data under item["content"]
                content = item.get("content", item)   # fallback to item itself
                title   = content.get("title", "") or item.get("title", "")
                if not title:
                    continue
                # pubDate is ISO string "2026-05-31T12:30:00Z" or Unix int
                pub_raw = content.get("pubDate", "") or item.get("providerPublishTime", 0)
                if isinstance(pub_raw, str) and pub_raw:
                    try:
                        from datetime import timezone
                        pub_ts = int(datetime.strptime(
                            pub_raw[:19], "%Y-%m-%dT%H:%M:%S"
                        ).replace(tzinfo=timezone.utc).timestamp())
                    except Exception:
                        pub_ts = 0
                else:
                    pub_ts = int(pub_raw) if pub_raw else 0
                results.append({
                    "ticker":    t,
                    "title":     title,
                    "published": pub_ts,
                    "publisher": content.get("provider", {}).get("displayName", "")
                                 or item.get("publisher", ""),
                    "link":      content.get("canonicalUrl", {}).get("url", "")
                                 or item.get("link", ""),
                })
            return results
        except Exception:
            return []

    # Also fetch general market ETF news (catches macro events)
    macro_tickers = ["SPY", "QQQ", "TLT", "GLD", "UUP"]
    all_fetch = list(set(tickers + macro_tickers))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in all_fetch[:60]}
        for fut in as_completed(futures):
            all_news.extend(fut.result())

    # Deduplicate by title
    seen  = set()
    dedup = []
    for item in all_news:
        key = item["title"].lower()[:80]
        if key not in seen:
            seen.add(key)
            dedup.append(item)

    # Sort by published (newest first)
    dedup.sort(key=lambda x: x.get("published", 0), reverse=True)
    return dedup[:max_headlines]


# ─────────────────────────────────────────────────────────────────────────────
# EVENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _text_matches_rule(text: str, keyword_groups: list[list[str]]) -> bool:
    """
    Returns True if the text matches ALL keyword groups.
    Within each group, ANY keyword can match (OR logic within group).
    Across groups: AND logic.
    """
    text_lower = text.lower()
    for group in keyword_groups:
        if not any(kw.lower() in text_lower for kw in group):
            return False
    return True


def classify_event(title: str, published_ts: int) -> list[dict]:
    """
    Run a headline through all causal rules.
    Returns list of matched events (may match multiple rules).

    Improvements over previous version:
      1. Dynamic magnitude scaling: extract deal size ($B) and scale magnitude
         (e.g. "$500B AI deal" upgrades magnitude to HIGH regardless of base)
      2. CEO intelligence: named executives identify SPECIFIC beneficiaries
      3. Deal-size metadata attached to event for audit trail
    """
    matches = []
    published_dt = datetime.fromtimestamp(published_ts) \
                   if published_ts > 0 else datetime.now()

    # Extract deal size once (reused across all rule matches for this headline)
    deal_size_bn = _extract_deal_size_bn(title)

    for rule in CAUSAL_RULES:
        if _text_matches_rule(title, rule["keywords"]):
            # Resolve affected tickers
            affected = list(rule.get("tickers", []))
            for sector in rule.get("sectors", []):
                affected.extend(SECTOR_TICKERS.get(sector, []))

            # Named-entity intelligence: CEO name OR company name in headline
            # → directly-affected ticker added with highest confidence
            ceo_tickers     = _extract_ceo_tickers(title)
            company_tickers = _extract_company_tickers(title)
            for t in ceo_tickers + company_tickers:
                if t not in affected:
                    affected.append(t)

            affected = sorted(set(affected))
            if not affected:
                continue

            # Dynamic magnitude: scale by extracted deal size
            base_mag     = rule["magnitude"]
            final_mag    = _deal_size_to_magnitude(deal_size_bn, base_mag)
            mag_upgraded = (final_mag != base_mag)

            matches.append({
                "event_type":    rule["event_type"],
                "direction":     rule["direction"],
                "magnitude":     final_mag,
                "magnitude_base": base_mag,         # original rule magnitude
                "deal_size_bn":   deal_size_bn,     # None if not found
                "mag_upgraded":   mag_upgraded,
                "affected":      affected,
                "duration_days": rule["duration_days"],
                "cause_chain":   rule["cause_chain"],
                "headline":      title,
                "published":     published_dt.strftime("%Y-%m-%d %H:%M"),
                "expires":       (published_dt + timedelta(days=rule["duration_days"]))
                                 .strftime("%Y-%m-%d"),
            })

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# CATALYST SCORER
# ─────────────────────────────────────────────────────────────────────────────

MAGNITUDE_SCORE = {"HIGH": 30, "MEDIUM": 18, "LOW": 10}
DIRECTION_SIGN  = {"BULLISH": +1, "BEARISH": -1, "MIXED": 0}

# ─────────────────────────────────────────────────────────────────────────────
# DEAL SIZE EXTRACTOR
# Parses dollar amounts from news headlines to scale catalyst magnitude.
# Example: "$500 billion AI deal" → deal_size_bn=500 → magnitude VERY_HIGH
# ─────────────────────────────────────────────────────────────────────────────

def _extract_deal_size_bn(text: str) -> float | None:
    """
    Extract largest dollar amount from text.  Returns amount in $billions, or None.

    Handles patterns:
      $500 billion / $500B / $500bn  → 500.0
      $2.5 trillion / $2.5T          → 2500.0
      $150 million / $150M / $150mn  → 0.15
      500 billion dollar deal        → 500.0
    """
    text = text.replace(",", "")  # remove thousands separator
    patterns = [
        # "$NNN trillion/billion/million" with optional decimal
        (r"\$\s*(\d+(?:\.\d+)?)\s*trillion",   1_000.0),
        (r"\$\s*(\d+(?:\.\d+)?)\s*[Tt]",       1_000.0),
        (r"\$\s*(\d+(?:\.\d+)?)\s*billion",     1.0),
        (r"\$\s*(\d+(?:\.\d+)?)\s*[Bb]n?\b",   1.0),
        (r"\$\s*(\d+(?:\.\d+)?)\s*[Bb]\b",     1.0),
        (r"\$\s*(\d+(?:\.\d+)?)\s*million",     0.001),
        (r"\$\s*(\d+(?:\.\d+)?)\s*[Mm]n?\b",   0.001),
        # "NNN billion dollar deal" (no $ sign)
        (r"(\d+(?:\.\d+)?)\s*trillion.{0,10}deal", 1_000.0),
        (r"(\d+(?:\.\d+)?)\s*billion.{0,10}deal",  1.0),
    ]
    amounts: list[float] = []
    for pattern, multiplier in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                amounts.append(float(m.group(1)) * multiplier)
            except (ValueError, IndexError):
                pass
    return max(amounts) if amounts else None


def _deal_size_to_magnitude(deal_bn: float | None, base_magnitude: str) -> str:
    """
    Scale rule magnitude based on extracted deal size.

    Thresholds (deal size in $billions):
      > 200B → HIGH     (mega deal: Saudi $500B AI → always HIGH)
      50-200B → HIGH    (large deal)
      10-50B  → MEDIUM  (significant deal)
      1-10B   → same as base magnitude
      < 1B    → LOW     (small deal, limited impact)
      None    → base magnitude (no dollar figure in headline)
    """
    if deal_bn is None:
        return base_magnitude
    if deal_bn >= 50.0:
        return "HIGH"
    elif deal_bn >= 10.0:
        return "MEDIUM" if base_magnitude == "LOW" else base_magnitude
    elif deal_bn < 1.0:
        return "LOW"
    return base_magnitude


# ─────────────────────────────────────────────────────────────────────────────
# CALENDAR-BASED PRE-POSITIONING
# Generates anticipation events for known scheduled macro events.
# Pre-positions BEFORE the event so the system captures the run-up, not just
# the post-announcement move.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULED EVENT CALENDARS
# Update quarterly when BLS / Fed publish their release schedules.
# Sources:
#   Fed meetings:  federalreserve.gov/monetarypolicy/fomccalendars.htm
#   CPI:           bls.gov/schedule/news_release/cpi.htm
#   Jobs report:   bls.gov/schedule/news_release/empsit.htm
#   FOMC minutes:  released ~3 weeks after each FOMC meeting
# ─────────────────────────────────────────────────────────────────────────────

# FOMC decision dates
FED_MEETING_DATES: list[str] = [
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# FOMC meeting minutes release dates (~3 weeks after decision)
FED_MINUTES_DATES: list[str] = [
    # 2025
    "2025-02-19", "2025-04-09", "2025-05-28", "2025-07-09",
    "2025-08-20", "2025-10-08", "2025-11-19", "2026-01-07",
    # 2026
    "2026-02-18", "2026-04-08", "2026-05-27", "2026-07-08",
    "2026-08-19", "2026-10-07", "2026-11-18", "2027-01-06",
]

# CPI release dates (approx 2nd Tuesday–Wednesday of each month, 8:30am ET)
# Source: bls.gov — verify exact dates each quarter
CPI_RELEASE_DATES: list[str] = [
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-10", "2025-10-15", "2025-11-13", "2025-12-11",
    # 2026
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-08",
    "2026-05-13", "2026-06-10", "2026-07-09", "2026-08-12",
    "2026-09-09", "2026-10-14", "2026-11-12", "2026-12-09",
]

# Non-Farm Payrolls release dates (first Friday of each month, 8:30am ET)
JOBS_REPORT_DATES: list[str] = [
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # 2026
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-10", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

# GDP advance estimate release dates (last Wednesday of Jan/Apr/Jul/Oct)
# Source: bea.gov advance estimate schedule
GDP_RELEASE_DATES: list[str] = [
    # 2025
    "2025-01-30", "2025-04-30", "2025-07-30", "2025-10-29",
    # 2026
    "2026-01-28", "2026-04-29", "2026-07-29", "2026-10-28",
]

# Quarterly earnings season kickoff dates
# Convention: first major bank (JPM) reports ~2nd Friday of Jan/Apr/Jul/Oct
EARNINGS_SEASON_DATES: list[str] = [
    # 2025
    "2025-01-15", "2025-04-11", "2025-07-11", "2025-10-10",
    # 2026
    "2026-01-14", "2026-04-08", "2026-07-08", "2026-10-07",
]

# Quarterly triple witching (options + futures expiration) — 3rd Friday of Mar/Jun/Sep/Dec
# High volume + potential for pinning and volatility spike
TRIPLE_WITCHING_DATES: list[str] = [
    "2025-03-21", "2025-06-20", "2025-09-19", "2025-12-19",
    "2026-03-20", "2026-06-19", "2026-09-18", "2026-12-18",
]

# Russell 1000/2000 annual reconstitution — last Friday of June
# ~$12T in index funds rebalance; small-cap additions surge, deletions collapse
RUSSELL_REBALANCE_DATES: list[str] = [
    "2025-06-27",
    "2026-06-26",
]

# Known major scheduled events that create pre-event alpha
# Format: {date: [(event_label, direction, magnitude, sectors, tickers, cause)]}
CALENDAR_EVENTS: dict[str, list[dict]] = {
    # Add one-off events here as they are announced, e.g.:
    # "2026-06-10": [{"label": "Apple WWDC", "direction": "BULLISH",
    #                 "magnitude": "LOW", "tickers": ["AAPL"],
    #                 "cause": "Apple WWDC keynote → new AI features announced → AAPL re-rates"}],
}


def _build_calendar_events(
    universe: list[str],
    lookahead_days: int = 5,
) -> list[dict]:
    """
    Generate anticipation events for upcoming scheduled macro releases.

    Covers:
      1. FOMC rate decisions      — FED_MEETING_DATES
      2. FOMC meeting minutes     — FED_MINUTES_DATES
      3. CPI / inflation report   — CPI_RELEASE_DATES
      4. Non-Farm Payrolls        — JOBS_REPORT_DATES
      5. One-off CALENDAR_EVENTS  — user-defined

    All use the same direction logic:
      - Read macro_signals.json for current VIX + macro_score
      - HAWKISH bias if  VIX < 16 AND macro_score > 60  (hot economy = tighten)
      - DOVISH  bias if  VIX > 25 OR  macro_score < 35  (stressed = ease)
      - For data releases: also consider recent CPI/jobs trend stored in
        macro_signals.json (fields: cpi_yoy_last, jobs_3m_avg)

    Returns list of event dicts compatible with build_catalyst_scores().
    """
    now   = datetime.now()
    events: list[dict] = []

    # ── Read macro context once ───────────────────────────────────────────────
    macro_score    = 50.0
    vix_level      = 18.0
    cpi_yoy_last   = 3.0    # last known CPI y/y%
    jobs_3m_avg    = 150    # 3-month average monthly job additions (thousands)
    macro_path     = ROOT / "macro_signals.json"
    if macro_path.exists():
        try:
            ms = json.loads(macro_path.read_text())
            macro_score  = float(ms.get("macro_score", 50))
            vix_level    = float(ms.get("vix") or ms.get("vix_spot") or 18)
            cpi_yoy_last = float(ms.get("cpi_yoy_last", 3.0))
            jobs_3m_avg  = float(ms.get("jobs_3m_avg", 150))
        except Exception:
            pass

    _hawkish_env = vix_level < 16 and macro_score > 60
    _dovish_env  = vix_level > 25 or macro_score < 35

    def _make_event(
        event_type: str,
        headline: str,
        direction: str,
        magnitude: str,
        cause: str,
        sectors: list[str],
        days_away: int,
    ) -> dict | None:
        affected: list[str] = []
        for sec in sectors:
            affected.extend(SECTOR_TICKERS.get(sec, []))
        affected = sorted(set(t for t in affected if t in universe))
        if not affected:
            return None
        return {
            "event_type":    event_type,
            "direction":     direction,
            "magnitude":     magnitude,
            "affected":      affected,
            "deal_size_bn":  None,
            "mag_upgraded":  False,
            "duration_days": days_away + 3,
            "cause_chain":   cause,
            "headline":      headline,
            "published":     now.strftime("%Y-%m-%d %H:%M"),
            "expires":       (now + timedelta(days=days_away + 3)).strftime("%Y-%m-%d"),
            "source_ticker": "CALENDAR",
        }

    # ── 1. FOMC rate decision ─────────────────────────────────────────────────
    for date_str in FED_MEETING_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (0 <= days_away <= lookahead_days):
            continue

        if _hawkish_env:
            direction = "BEARISH"
            cause = (f"FOMC in {days_away}d — strong economy + low VIX suggests "
                     "market pricing in hawkish hold or delayed cut")
        elif _dovish_env:
            direction = "BULLISH"
            cause = (f"FOMC in {days_away}d — stressed market suggests "
                     "rate cut / dovish pivot anticipated")
        else:
            continue   # neutral — no pre-position needed

        ev = _make_event("FED_CALENDAR_ANTICIPATION",
                         f"[CALENDAR] FOMC meeting {date_str} in {days_away}d",
                         direction, "LOW", cause,
                         ["RATES_SENSITIVE", "REAL_ESTATE"], days_away)
        if ev:
            events.append(ev)

    # ── 2. FOMC meeting minutes ───────────────────────────────────────────────
    for date_str in FED_MINUTES_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (0 <= days_away <= lookahead_days):
            continue

        # Minutes reveal the deliberation — same hawkish/dovish bias as meeting
        if _hawkish_env:
            direction = "BEARISH"
            cause = (f"Fed minutes in {days_away}d — hawks likely emphasized "
                     "patience; market may re-price fewer cuts")
        elif _dovish_env:
            direction = "BULLISH"
            cause = (f"Fed minutes in {days_away}d — doves likely discussed "
                     "cutting rates; markets anticipate dovish reveal")
        else:
            continue

        ev = _make_event("FED_MINUTES_CALENDAR",
                         f"[CALENDAR] FOMC minutes {date_str} in {days_away}d",
                         direction, "LOW", cause,
                         ["RATES_SENSITIVE", "REAL_ESTATE"], days_away)
        if ev:
            events.append(ev)

    # ── 3. CPI / inflation release ────────────────────────────────────────────
    for date_str in CPI_RELEASE_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (0 <= days_away <= lookahead_days):
            continue

        # Anticipation direction based on recent CPI trend
        # If CPI has been running hot (>3.5% y/y), fear of another hot print
        # If CPI has been cooling (<3%), hope for confirmation of easing
        if cpi_yoy_last > 3.5:
            direction = "BEARISH"
            cause = (f"CPI release in {days_away}d — recent inflation {cpi_yoy_last:.1f}% y/y "
                     "above target; market fears another hot print → Fed delays cuts")
        elif cpi_yoy_last < 2.8:
            direction = "BULLISH"
            cause = (f"CPI release in {days_away}d — inflation cooling to {cpi_yoy_last:.1f}% y/y; "
                     "market hopes for continued disinflation → rate cut path clears")
        else:
            continue   # near target, no strong directional anticipation

        ev = _make_event("CPI_CALENDAR_ANTICIPATION",
                         f"[CALENDAR] CPI data {date_str} in {days_away}d "
                         f"(recent: {cpi_yoy_last:.1f}% y/y)",
                         direction, "LOW", cause,
                         ["RATES_SENSITIVE", "REAL_ESTATE", "CONSUMER_IMPORT"],
                         days_away)
        if ev:
            events.append(ev)

    # ── 4. Non-Farm Payrolls ──────────────────────────────────────────────────
    for date_str in JOBS_REPORT_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (0 <= days_away <= lookahead_days):
            continue

        # Anticipation: strong job market → Fed stays higher → rate-sensitive bearish
        # Weak job market → recession risk + dovish pivot → consumer/financials mixed
        if jobs_3m_avg > 200:
            direction = "BEARISH"
            cause = (f"Jobs report in {days_away}d — 3-month avg {jobs_3m_avg:.0f}k/month "
                     "signals labor market too hot; market fears Fed tightening delay")
        elif jobs_3m_avg < 100:
            direction = "BEARISH"    # weak = recession fear dominates
            cause = (f"Jobs report in {days_away}d — 3-month avg {jobs_3m_avg:.0f}k/month "
                     "signals labor market softening; recession risk re-emerges")
        else:
            continue   # Goldilocks range — minimal directional anticipation

        ev = _make_event("JOBS_CALENDAR_ANTICIPATION",
                         f"[CALENDAR] NFP jobs report {date_str} in {days_away}d "
                         f"(3m avg: {jobs_3m_avg:.0f}k)",
                         direction, "LOW", cause,
                         ["FINANCIALS", "RATES_SENSITIVE", "RETAIL"], days_away)
        if ev:
            events.append(ev)

    # ── 5. GDP advance estimate ───────────────────────────────────────────────
    for date_str in GDP_RELEASE_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (0 <= days_away <= lookahead_days):
            continue

        # GDP anticipation: if economy was strong last quarter, good odds of beat
        # If macro_score < 35 (recessionary signals), fear of miss
        if macro_score > 55:
            direction = "BULLISH"
            cause = (f"GDP release in {days_away}d — macro_score={macro_score:.0f} suggests "
                     "strong growth print likely; cyclicals/financials benefit")
        elif macro_score < 35:
            direction = "BEARISH"
            cause = (f"GDP release in {days_away}d — macro_score={macro_score:.0f} signals "
                     "weak growth; recession risk re-prices cyclicals lower")
        else:
            continue

        ev = _make_event("GDP_CALENDAR_ANTICIPATION",
                         f"[CALENDAR] GDP advance estimate {date_str} in {days_away}d",
                         direction, "LOW", cause,
                         ["FINANCIALS", "INDUSTRIALS", "RETAIL"], days_away)
        if ev:
            events.append(ev)

    # ── 6. Earnings season kickoff ────────────────────────────────────────────
    # When big bank earnings start, it sets the tone for the whole quarter.
    # Banks are read-across indicators: loan growth, NIM, credit quality signal
    # the health of the economy and all borrower sectors.
    for date_str in EARNINGS_SEASON_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (0 <= days_away <= lookahead_days):
            continue

        # Tone depends on macro conditions
        if macro_score > 55:
            direction = "BULLISH"
            cause = (f"Earnings season starts {date_str} in {days_away}d — "
                     "strong macro_score={macro_score:.0f} supports beat expectations; "
                     "bank NIM/loan growth likely solid")
        elif macro_score < 40:
            direction = "BEARISH"
            cause = (f"Earnings season starts {date_str} in {days_away}d — "
                     f"macro_score={macro_score:.0f} flags credit quality risk; "
                     "loan loss provisions may disappoint")
        else:
            continue

        ev = _make_event("EARNINGS_SEASON_KICKOFF",
                         f"[CALENDAR] Earnings season opens {date_str} in {days_away}d",
                         direction, "LOW", cause,
                         ["BANKS_YIELD", "FINANCIALS"], days_away)
        if ev:
            events.append(ev)

    # ── 7. Triple witching (quarterly options + futures expiration) ───────────
    # 3rd Friday of Mar/Jun/Sep/Dec.  The week before: dealers hedge gamma exposure
    # → exaggerated intraday moves; stocks pin to high-OI strikes.
    # Net effect: slight increase in vol + potential for whipsaw.
    for date_str in TRIPLE_WITCHING_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (1 <= days_away <= lookahead_days):   # skip the day itself
            continue

        # Triple witching creates mechanical volatility but no directional edge
        # → only flag as anticipation if current IV is very low (complacency risk)
        if vix_level < 14:
            direction = "BEARISH"
            cause = (f"Triple witching in {days_away}d — VIX={vix_level:.1f} (complacent); "
                     "dealer gamma unwind + expiration flow may cause volatility spike")
            ev = _make_event("TRIPLE_WITCHING_ANTICIPATION",
                             f"[CALENDAR] Triple witching {date_str} in {days_away}d",
                             direction, "LOW", cause,
                             ["RATES_SENSITIVE"], days_away)
            if ev:
                events.append(ev)

    # ── 8. Russell rebalancing (end of June) ──────────────────────────────────
    # ~$12T in index funds rebalance at market close on reconstitution day.
    # Additions to Russell 1000/2000 surge; deletions collapse.
    # In the week before: known additions are bought aggressively by front-runners.
    for date_str in RUSSELL_REBALANCE_DATES:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (3 <= days_away <= lookahead_days):   # 3+ days out for front-running
            continue

        cause = (f"Russell index reconstitution in {days_away}d — "
                 "~$12T in index funds rebalance; additions see forced buying, "
                 "deletions face systematic selling; small-cap momentum amplified")
        ev = _make_event("RUSSELL_REBALANCING",
                         f"[CALENDAR] Russell reconstitution {date_str} in {days_away}d",
                         "BULLISH", "LOW", cause,
                         ["FINANCIALS"],  # broad market structure event
                         days_away)
        if ev:
            events.append(ev)

    # ── 9. One-off CALENDAR_EVENTS (user-defined) ─────────────────────────────
    for date_str, one_off_list in CALENDAR_EVENTS.items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        days_away = (dt - now).days
        if not (0 <= days_away <= lookahead_days):
            continue
        for oe in one_off_list:
            affected: list[str] = list(oe.get("tickers", []))
            for sec in oe.get("sectors", []):
                affected.extend(SECTOR_TICKERS.get(sec, []))
            affected = sorted(set(t for t in affected if t in universe))
            if not affected:
                continue
            events.append({
                "event_type":    f"CALENDAR_{oe['label'].upper().replace(' ','_')}",
                "direction":     oe["direction"],
                "magnitude":     oe["magnitude"],
                "affected":      affected,
                "deal_size_bn":  None,
                "mag_upgraded":  False,
                "duration_days": days_away + 3,
                "cause_chain":   oe.get("cause", f"Scheduled event: {oe['label']}"),
                "headline":      f"[CALENDAR] {oe['label']} on {date_str} in {days_away}d",
                "published":     now.strftime("%Y-%m-%d %H:%M"),
                "expires":       (now + timedelta(days=days_away + 3)).strftime("%Y-%m-%d"),
                "source_ticker": "CALENDAR",
            })

    return events

def build_catalyst_scores(events: list[dict],
                          universe: list[str]) -> pd.DataFrame:
    """
    Aggregate all active events into per-ticker catalyst scores (0-100).
    Neutral = 50. Above 50 = tailwind, below 50 = headwind.
    Multiple events stack (capped at ±40 total shift).
    """
    now = datetime.now()
    ticker_deltas: dict[str, float] = {t: 0.0 for t in universe}
    ticker_reasons: dict[str, list[str]] = {t: [] for t in universe}

    for ev in events:
        # Skip expired events
        try:
            expires = datetime.strptime(ev["expires"], "%Y-%m-%d")
            if expires < now:
                continue
        except Exception:
            pass

        delta = DIRECTION_SIGN[ev["direction"]] * MAGNITUDE_SCORE[ev["magnitude"]]

        for t in ev["affected"]:
            if t in ticker_deltas:
                ticker_deltas[t]  += delta
                ticker_reasons[t].append(
                    f"{ev['event_type']}({ev['direction'][:4]})"
                )

    rows = []
    for t in universe:
        raw_delta = float(np.clip(ticker_deltas[t], -40, 40))
        score     = round(50.0 + raw_delta, 1)
        reasons   = " | ".join(ticker_reasons[t]) if ticker_reasons[t] else "NO_CATALYST"
        rows.append({
            "ticker":          t,
            "catalyst_score":  score,
            "catalyst_delta":  round(raw_delta, 1),
            "catalyst_count":  len(ticker_reasons[t]),
            "catalyst_reason": reasons,
        })

    df = pd.DataFrame(rows).sort_values("catalyst_delta", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    for path in [ROOT / "alpha_scores.csv",
                 ROOT / "regime_ml_scores.csv",
                 ROOT / "backtest_price_cache.csv"]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "ticker" in df.columns:
                    return df["ticker"].dropna().unique().tolist()
                else:
                    return [c for c in pd.read_csv(path, nrows=0).columns
                            if c not in ("", "Date", "Unnamed: 0")]
            except Exception:
                pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 65)
        print("Canyon v9 — Step 101: Macro Event Intelligence Engine")
        print("=" * 65)

    # 1. Universe
    universe = load_universe()
    if not universe:
        print("  ERROR: no universe found")
        return pd.DataFrame()
    if verbose:
        print(f"\n[1/4] Universe: {len(universe)} tickers")

    # 2. Fetch news
    if verbose:
        print(f"\n[2/4] Fetching news headlines …")
    try:
        headlines = fetch_market_news(universe)
        if verbose:
            print(f"  {len(headlines)} unique headlines fetched")
    except Exception as e:
        if verbose:
            print(f"  news fetch failed: {e}  — running with empty headlines")
        headlines = []

    # 3. Classify events (news-based + calendar-based)
    if verbose:
        print(f"\n[3/4] Classifying macro catalysts …")
    all_events: list[dict] = []
    matched_headlines = 0
    n_mag_upgraded = 0
    for item in headlines:
        evs = classify_event(item["title"],
                             int(item.get("published", 0)))
        if evs:
            matched_headlines += 1
            for ev in evs:
                ev["source_ticker"] = item["ticker"]
                if ev.get("mag_upgraded"):
                    n_mag_upgraded += 1
            all_events.extend(evs)

    # Calendar-based pre-positioning (Fed meetings, etc.)
    calendar_evs = _build_calendar_events(universe, lookahead_days=5)
    if calendar_evs:
        all_events.extend(calendar_evs)
        if verbose:
            for cev in calendar_evs:
                icon = "🟢" if cev["direction"] == "BULLISH" else "🔴"
                print(f"  {icon} [CALENDAR] {cev['headline']}  → {cev['cause_chain']}")

    if verbose:
        print(f"  {matched_headlines}/{len(headlines)} headlines matched rules")
        if n_mag_upgraded:
            print(f"  {n_mag_upgraded} events had magnitude upgraded by deal-size extraction")
        print(f"  {len(all_events)} total events (news + calendar)")
        if all_events:
            news_evs = [e for e in all_events if e.get("source_ticker", "") != "CALENDAR"]
            print(f"\n  Active macro catalysts:")
            for ev in news_evs[:8]:
                icon = "🟢" if ev["direction"] == "BULLISH" else \
                       ("🔴" if ev["direction"] == "BEARISH" else "⚪")
                deal_tag = f"  [${ev['deal_size_bn']:.0f}B]" if ev.get("deal_size_bn") else ""
                upg_tag  = " ↑MAG" if ev.get("mag_upgraded") else ""
                print(f"    {icon} [{ev['event_type']:<22}] "
                      f"{ev['direction']:<8} {ev['magnitude']:<6}{upg_tag}{deal_tag}  "
                      f"→ {ev['cause_chain'][:60]}")

    # 4. Build catalyst scores
    if verbose:
        print(f"\n[4/4] Building per-ticker catalyst scores …")
    scores_df = build_catalyst_scores(all_events, universe)

    # Show top movers
    if verbose and not scores_df.empty:
        top_bull = scores_df[scores_df["catalyst_delta"] > 0].head(8)
        top_bear = scores_df[scores_df["catalyst_delta"] < 0].head(8)
        if not top_bull.empty:
            print("\n  Tailwind (macro support):")
            for _, r in top_bull.iterrows():
                print(f"    🟢 {r['ticker']:<6} score={r['catalyst_score']:.0f}  "
                      f"{r['catalyst_reason']}")
        if not top_bear.empty:
            print("\n  Headwind (macro risk):")
            for _, r in top_bear.iterrows():
                print(f"    🔴 {r['ticker']:<6} score={r['catalyst_score']:.0f}  "
                      f"{r['catalyst_reason']}")

    # Save
    out_scores   = ROOT / "macro_catalyst_scores.csv"
    out_catalog  = ROOT / "macro_event_catalog.json"
    scores_df.to_csv(out_scores, index=False)

    catalog = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_headlines": len(headlines),
        "n_events":    len(all_events),
        "events":      all_events[:50],   # store top 50
    }
    out_catalog.write_text(json.dumps(catalog, indent=2, default=str))

    if verbose:
        print(f"\n  Saved: {out_scores.name}  |  {out_catalog.name}")
        print("=" * 65)

    return scores_df


if __name__ == "__main__":
    run(verbose=True)
