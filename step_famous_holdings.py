#!/usr/bin/env python3
"""
Canyon — step_famous_holdings.py
==================================
Fetch and parse 13F-HR filings from SEC EDGAR for the world's top
hedge fund managers. Produces a comprehensive holdings analysis for
the Famous Holdings dashboard tab.

Funds tracked (hardcoded CIKs — these never change):
  Berkshire Hathaway  (Warren Buffett)    CIK: 0001067983
  Pershing Square     (Bill Ackman)       CIK: 0001336528
  Scion Asset Mgmt    (Michael Burry)     CIK: 0001694820
  Appaloosa Mgmt      (David Tepper)      CIK: 0000930547
  Duquesne Family     (Druckenmiller)     CIK: 0001056831
  Coatue Management   (Philippe Laffont)  CIK: 0001356093
  Viking Global       (Andreas Halvorsen) CIK: 0001103804
  Tiger Global        (Chase Coleman)     CIK: 0001167483
  Greenlight Capital  (David Einhorn)     CIK: 0001079114
  Third Point         (Dan Loeb)          CIK: 0001040273

Outputs:
  famous_holdings.json    per-fund holdings + consensus analysis
  famous_holdings.csv     flat table: fund | ticker | value_m | pct_portfolio | change
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg): print(f"  {msg}")
def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")

CACHE_DIR = ROOT / "sec_filings_cache" / "famous"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MAX_AGE_DAYS = 30   # re-fetch at most monthly

# ── Famous fund registry ───────────────────────────────────────────────────────
FAMOUS_FUNDS = {
    "Berkshire Hathaway":  {"cik": "0001067983", "manager": "Warren Buffett",         "style": "Value/Concentrated"},
    "Pershing Square":     {"cik": "0001336528", "manager": "Bill Ackman",            "style": "Activist/Concentrated"},
    "Scion Asset Mgmt":    {"cik": "0001649339", "manager": "Michael Burry",          "style": "Contrarian/Value"},
    "Duquesne Family":     {"cik": "0001056831", "manager": "Stanley Druckenmiller",  "style": "Macro/Momentum"},
    "Coatue Management":   {"cik": "0001135730", "manager": "Philippe Laffont",       "style": "Tech/Growth"},
    "Viking Global":       {"cik": "0001103804", "manager": "Andreas Halvorsen",      "style": "Long/Short Equity"},
    "Tiger Global":        {"cik": "0001167483", "manager": "Chase Coleman",          "style": "Tech/Growth/VC"},
    "Greenlight Capital":  {"cik": "0001079114", "manager": "David Einhorn",          "style": "Value/Short"},
    "Third Point":         {"cik": "0001040273", "manager": "Dan Loeb",               "style": "Activist/Event"},
    "Baupost Group":       {"cik": "0001061768", "manager": "Seth Klarman",           "style": "Value/Event-driven"},
}


# ── EDGAR HTTP helper ──────────────────────────────────────────────────────────

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

_HEADERS = {
    "User-Agent": "Canyon Quant Research canyon@research.com",
    "Accept":     "application/json, application/xml, text/plain, */*",
}


def _get(url: str, as_text: bool = True) -> str | bytes | None:
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as r:
            raw = r.read()
            if as_text:
                enc = r.headers.get_content_charset("utf-8") or "utf-8"
                try:
                    return raw.decode(enc)
                except Exception:
                    return raw.decode("latin-1")
            return raw
    except Exception as e:
        warn(f"  GET {url[:70]}… → {e}")
        return None
    finally:
        time.sleep(0.15)   # SEC rate limit: ≤10 req/sec


# ── Fetch latest 13F-HR accession for a fund ──────────────────────────────────

def get_latest_13f_accession(cik: str) -> tuple[str, str] | None:
    """Returns (accession_number, filing_date) or None."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    raw = _get(url)
    if not raw:
        return None
    try:
        d = json.loads(raw)
        filings = d.get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        accns   = filings.get("accessionNumber", [])
        dates   = filings.get("filingDate", [])
        for i, form in enumerate(forms):
            if "13F-HR" in str(form):
                return accns[i], dates[i]
    except Exception as e:
        warn(f"  CIK {cik} submissions parse error: {e}")
    return None


# ── Fetch and parse 13F-HR holdings XML ───────────────────────────────────────

import re as _re


def _get_filing_xml_urls(cik_int: str, accn_clean: str) -> list[str]:
    """Fetch the EDGAR filing directory and return all XML/TXT hrefs."""
    dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_clean}/"
    html    = _get(dir_url)
    if not html:
        return []
    return _re.findall(r'href="(/Archives/edgar/data/[^"]+\.[xX][mM][lL])"', html)


def _extract_infotable_from_txt(txt: str) -> str | None:
    """Extract the INFORMATION TABLE XML section from a full EDGAR .txt submission."""
    # The INFORMATION TABLE section starts with <TYPE>INFORMATION TABLE
    idx = txt.find("<TYPE>INFORMATION TABLE")
    if idx < 0:
        return None
    seg = txt[idx:]
    # The XML is inside <XML>…</XML> tags within the document
    xml_start = seg.find("<XML>")
    xml_end   = seg.find("</XML>")
    if xml_start >= 0 and xml_end > xml_start:
        return seg[xml_start + 5 : xml_end].strip()
    # Fallback: look for the informationTable or infoTable tag directly
    for marker in ("<informationTable", "<ns1:informationTable", "<InformationTable"):
        mi = seg.find(marker)
        if mi >= 0:
            return seg[mi:]
    return None


def fetch_holdings(cik: str, accession: str) -> list[dict]:
    """Discover and parse the 13F infotable XML from EDGAR filing directory."""
    accn_clean = accession.replace("-", "")
    cik_int    = str(int(cik))

    # Step 1: try separate XML files listed in the directory
    xml_paths = _get_filing_xml_urls(cik_int, accn_clean)
    log(f"  XML files found: {[p.split('/')[-1] for p in xml_paths]}")

    infotable_paths = [p for p in xml_paths if "primary_doc" not in p.lower()]
    for path in infotable_paths:
        xml_raw = _get(f"https://www.sec.gov{path}")
        if not xml_raw:
            continue
        holdings = _parse_infotable_xml(xml_raw)
        if holdings:
            log(f"  Parsed {len(holdings)} holdings from {path.split('/')[-1]}")
            return holdings

    # Step 2: fallback — parse embedded XML in the full .txt submission
    txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_clean}/{accession}.txt"
    log(f"  Trying full submission .txt …")
    txt_raw = _get(txt_url)
    if txt_raw:
        xml_sect = _extract_infotable_from_txt(txt_raw)
        if xml_sect:
            holdings = _parse_infotable_xml(xml_sect)
            if holdings:
                log(f"  Parsed {len(holdings)} holdings from embedded INFORMATION TABLE")
                return holdings

    return []


def _strip_ns(xml_text: str) -> str:
    """Remove all XML namespace declarations so ElementTree can parse with simple tag names."""
    # Remove xmlns= (default namespace) — covers both cases SEC uses
    xml_text = _re.sub(r'\sxmlns="[^"]+"', "", xml_text)
    xml_text = _re.sub(r'\sxmlns:\w+="[^"]+"', "", xml_text)
    # Also strip xsi: prefixed attributes which may cause parse errors
    xml_text = _re.sub(r'\sxsi:\w+="[^"]+"', "", xml_text)
    # Remove all remaining namespace prefixes on elements (n1:tag → tag)
    xml_text = _re.sub(r"<(/?)[\w]+:([\w]+)", r"<\1\2", xml_text)
    return xml_text


def _find_text(el: ET.Element, *tags: str) -> str:
    """Try several tag name variants (case-insensitive search)."""
    for tag in tags:
        found = el.find(tag)
        if found is None:
            found = el.find(tag.lower())
        if found is None:
            found = el.find(tag.upper())
        if found is not None and found.text:
            return found.text.strip()
    return ""


def _parse_infotable_xml(xml_text: str) -> list[dict]:
    """Parse 13F-HR infotable XML → list of holdings dicts."""
    xml_text = _strip_ns(xml_text)
    try:
        root = ET.fromstring(xml_text.strip().lstrip("﻿"))
    except ET.ParseError as exc:
        warn(f"  XML parse error: {exc}")
        return []

    # The entry tag is <infoTable> (lowercase 't') in SEC schema
    entries = root.findall(".//infoTable")
    if not entries:
        entries = root.findall(".//InfoTable")
    if not entries:
        # Try direct children in case root IS the table
        entries = list(root)

    holdings = []
    for entry in entries:
        name      = _find_text(entry, "nameOfIssuer", "nameofissuer")
        cusip     = _find_text(entry, "cusip")
        value_raw = _find_text(entry, "value")
        # sshPrnamt lives inside <shrsOrPrnAmt>
        shr_el   = entry.find("shrsOrPrnAmt") or entry.find("shrsorprnamt")
        if shr_el is not None:
            shares_raw = _find_text(shr_el, "sshPrnamt", "sshprnamt")
        else:
            shares_raw = _find_text(entry, "sshPrnamt", "sshprnamt")

        try:
            # SEC 13F XML value field is in actual US dollars
            value_usd = float(value_raw.replace(",", "")) if value_raw else 0
        except ValueError:
            value_usd = 0

        try:
            shares = float(shares_raw.replace(",", "")) if shares_raw else 0
        except ValueError:
            shares = 0

        if value_usd > 0 and name:
            holdings.append({
                "name":      name,
                "cusip":     cusip,
                "value_usd": value_usd,
                "shares":    shares,
            })

    return sorted(holdings, key=lambda x: -x["value_usd"])


# ── Match holdings to tickers via alpha_scores.csv ───────────────────────────

def build_name_to_ticker_map() -> dict[str, str]:
    """Build {company_name_upper → ticker} from alpha_scores.csv."""
    mapping: dict[str, str] = {}
    for fname in ["alpha_scores.csv", "daily_picks.csv"]:
        p = ROOT / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            name_col = next((c for c in ["company", "name", "company_name"] if c in df.columns), None)
            if "ticker" in df.columns and name_col:
                for _, r in df.iterrows():
                    key = str(r[name_col]).upper().strip()
                    mapping[key] = str(r["ticker"])
        except Exception:
            pass

    # Common partial-match overrides for well-known names
    OVERRIDES = {
        # FAANG + Mega-cap
        "APPLE INC": "AAPL", "APPLE": "AAPL",
        "MICROSOFT CORP": "MSFT", "MICROSOFT CORPORATION": "MSFT",
        "AMAZON COM INC": "AMZN", "AMAZON.COM INC": "AMZN", "AMAZON COM": "AMZN",
        "NVIDIA CORP": "NVDA", "NVIDIA CORPORATION": "NVDA",
        "ALPHABET INC": "GOOGL", "ALPHABET INC-CL A": "GOOGL", "ALPHABET INC-CL C": "GOOG",
        "GOOGLE INC": "GOOGL",
        "TESLA INC": "TSLA", "TESLA MOTORS INC": "TSLA",
        "META PLATFORMS INC": "META", "FACEBOOK INC": "META",
        "BERKSHIRE HATHAWAY INC": "BRK/B", "BERKSHIRE HATHAWAY": "BRK/B",
        # Finance
        "JPMORGAN CHASE CO": "JPM", "JPMORGAN CHASE & CO": "JPM",
        "VISA INC": "V", "VISA INC-CLASS A SHARES": "V",
        "MASTERCARD INC": "MA", "MASTERCARD INCORPORATED": "MA",
        "BANK OF AMERICA CORP": "BAC", "BANK OF AMERICA": "BAC",
        "AMERICAN EXPRESS CO": "AXP", "AMERICAN EXPRESS": "AXP",
        "CITIGROUP INC": "C", "WELLS FARGO CO": "WFC",
        "GOLDMAN SACHS GROUP INC": "GS", "MORGAN STANLEY": "MS",
        # Tech
        "BROADCOM INC": "AVGO", "MICRON TECHNOLOGY": "MU", "MICRON TECHNOLOGY INC": "MU",
        "ADVANCED MICRO DEVICES": "AMD", "ADVANCED MICRO DEVICES INC": "AMD",
        "TAIWAN SEMICONDUCTOR": "TSM", "TAIWAN SEMICONDUCTOR MFG": "TSM",
        "ORACLE CORP": "ORCL", "ORACLE CORPORATION": "ORCL",
        "SALESFORCE INC": "CRM", "SALESFORCE.COM INC": "CRM",
        "INTEL CORP": "INTC", "QUALCOMM INC": "QCOM",
        "SERVICENOW INC": "NOW", "SNOWFLAKE INC": "SNOW",
        "UBER TECHNOLOGIES INC": "UBER", "UBER TECHNOLOGIES": "UBER",
        "AIRBNB INC": "ABNB", "LYFT INC": "LYFT",
        # Consumer
        "EXXON MOBIL CORP": "XOM", "EXXON MOBIL": "XOM",
        "CHEVRON CORP": "CVX", "CHEVRON CORPORATION": "CVX",
        "WALMART INC": "WMT", "WALMART INC.": "WMT",
        "OCCIDENTAL PETROLEUM": "OXY", "OCCIDENTAL PETROLEUM CORP": "OXY",
        "KRAFT HEINZ CO": "KHC", "KRAFT HEINZ": "KHC",
        "ULTA BEAUTY INC": "ULTA", "CHIPOTLE MEXICAN GRILL": "CMG",
        "CHIPOTLE MEXICAN GRILL INC": "CMG",
        "HILTON WORLDWIDE": "HLT", "HILTON WORLDWIDE HOLDINGS INC": "HLT",
        "RESTAURANT BRANDS INTL INC": "QSR", "RESTAURANT BRANDS": "QSR",
        "HOWARD HUGHES HOLDINGS INC": "HHH", "HOWARD HUGHES": "HHH",
        "HERTZ GLOBAL HLDGS INC": "HTZ", "HERTZ GLOBAL HOLDINGS INC": "HTZ",
        "SEAPORT ENTMT GROUP INC": "SEG",
        "BROOKFIELD CORP": "BN", "BROOKFIELD ASSET MANAGEMENT": "BAM",
        "BROOKFIELD ASSET MGMT INC": "BAM",
        # Industrial/Other
        "ALLY FINL INC": "ALLY", "ALLY FINANCIAL INC": "ALLY",
        "LIBERTY MEDIA": "LLYVA", "CONSTELLATION BRANDS": "STZ",
        "DAVITA INC": "DVA", "FLOOR & DECOR": "FND",
        "SIRIUS XM HLDGS INC": "SIRI", "SIRIUS XM HOLDINGS INC": "SIRI",
        "T-MOBILE US INC": "TMUS", "CHARTER COMMUNICATIONS": "CHTR",
        "COMCAST CORP": "CMCSA", "DISNEY (WALT) CO": "DIS",
        # Scion / Burry favourites
        "PALANTIR TECHNOLOGIES INC": "PLTR",
        "PFIZER INC": "PFE", "HALLIBURTON CO": "HAL",
        "MOLINA HEALTHCARE INC": "MOH", "STIFEL FINANCIAL CORP": "SF",
        # Druckenmiller / Duquesne
        "ST JOE CO": "JOE", "ENTERPRISE PRODS PARTNERS L": "EPD",
        "ENTERPRISE PRODUCTS PARTNERS": "EPD",
        "BANK OZK": "OZK", "WR BERKLEY CORP": "WRB",
        # Baupost / Greenlight
        "WESCO INTL INC": "WCC", "UNION PAC CORP": "UNP",
        "ELEVANCE HEALTH INC FORMERLY": "ELV", "ELEVANCE HEALTH INC": "ELV",
        "ANTHOEM INC": "ELV", "HCA HEALTHCARE INC": "HCA",
        "CIGNA GROUP": "CI", "CVS HEALTH CORP": "CVS",
        "BAUSCH HEALTH": "BHC", "VIACOMCBS INC": "PARA",
        "AERCAP HOLDINGS": "AER", "CONSOL ENERGY INC": "CEIX",
        # Greenlight
        "ATLAS AIR WORLDWIDE": "AAWW", "BRIGHTHOUSE FINL INC": "BHF",
        "TENET HEALTHCARE CORP": "THC",
        # Coatue / Tiger Global
        "SNOWFLAKE INC": "SNOW", "CROWDSTRIKE HOLDINGS INC": "CRWD",
        "SHOPIFY INC": "SHOP", "BLOCK INC": "SQ",
        "DATADOG INC": "DDOG", "CLOUDFLARE INC": "NET",
        "ROBLOX CORP": "RBLX", "DOORDASH INC": "DASH",
        "INSTACART": "CART", "COUPANG INC": "CPNG",
        "GRAB HOLDINGS LTD": "GRAB", "DIDI GLOBAL INC": "DIDI",
        "TAIWAN SEMI MFG CO LTD": "TSM", "ASML HOLDING N V": "ASML",
        "SAMSUNG ELECTRONICS CO": "SSNLF", "TENCENT HOLDINGS": "TCEHY",
        "ALIBABA GROUP HOLDING": "BABA", "BAIDU INC": "BIDU",
        "NETEASE INC": "NTES", "JD COM INC": "JD",
        # Viking
        "VISA INC CL A": "V", "VISA INC": "V",
        "UNITEDHEALTH GROUP INC": "UNH", "ABBVIE INC": "ABBV",
        "INTUITIVE SURGICAL INC": "ISRG", "DANAHER CORP": "DHR",
        "S P GLOBAL INC": "SPGI", "MSCI INC": "MSCI",
        "FAIR ISAAC CORP": "FICO", "MOODYS CORP": "MCO",
        "AMERICAN TOWER CORP": "AMT", "PROLOGIS INC": "PLD",
        "ADAPTIVE BIOTECHNOLOGIES COR": "ACRS",
        "AIR PRODUCTS AND CHEMICALS I": "APD",
        "CANADIAN PACIFIC RAILWAY": "CP", "CSX CORP": "CSX",
        "NORFOLK SOUTHERN CORP": "NSC", "UNION PACIFIC CORP": "UNP",
        "CORTEVA INC": "CTVA", "DEERE CO": "DE",
        "PARKER HANNIFIN CORP": "PH", "CATERPILLAR INC": "CAT",
    }
    mapping.update(OVERRIDES)
    return mapping


def resolve_ticker(name: str, name_map: dict[str, str]) -> str | None:
    """Try to match a holding name to a known ticker."""
    key = name.upper().strip()
    if key in name_map:
        return name_map[key]
    # Partial match: check if any known name starts with the holding name
    for k, v in name_map.items():
        if key and (k.startswith(key[:8]) or key.startswith(k[:8])):
            return v
    return None


# ── Load previous holdings for change detection ───────────────────────────────

def load_previous_holdings(fund_name: str) -> dict[str, float]:
    """Returns {ticker: value_usd} from previous quarter if cached."""
    prev_path = CACHE_DIR / f"prev_{fund_name.replace(' ', '_')}.json"
    if prev_path.exists():
        try:
            return json.loads(prev_path.read_text())
        except Exception:
            pass
    return {}


def save_current_as_previous(fund_name: str, holdings: dict[str, float]):
    prev_path = CACHE_DIR / f"prev_{fund_name.replace(' ', '_')}.json"
    prev_path.write_text(json.dumps(holdings))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}Canyon — Famous Holdings (SEC 13F){RESET}  {TODAY}")
    print(f"  Fetching from SEC EDGAR for {len(FAMOUS_FUNDS)} top fund managers …\n")

    # Check cache age
    out_json = ROOT / "famous_holdings.json"
    if out_json.exists():
        age = (datetime.now() - datetime.fromtimestamp(out_json.stat().st_mtime)).days
        if age < CACHE_MAX_AGE_DAYS:
            ok(f"Cache is {age}d old (< {CACHE_MAX_AGE_DAYS}d) — skipping fetch")
            return

    name_map  = build_name_to_ticker_map()
    all_funds: dict[str, dict] = {}
    flat_rows: list[dict]      = []

    for fund_name, info in FAMOUS_FUNDS.items():
        cik = info["cik"]
        log(f"Fetching {fund_name} ({info['manager']}) …")

        result = get_latest_13f_accession(cik)
        if not result:
            warn(f"  No 13F-HR found for {fund_name} — skipping")
            continue
        accession, filing_date = result
        log(f"  Latest 13F-HR: {accession}  filed: {filing_date}")

        holdings_raw = fetch_holdings(cik, accession)
        if not holdings_raw:
            warn(f"  Could not parse holdings XML for {fund_name}")
            continue

        # Aggregate duplicate ticker entries (same stock, multiple accounts/classes)
        aggregated: dict[str, dict] = {}
        for h in holdings_raw:
            ticker = resolve_ticker(h["name"], name_map)
            key    = ticker or h["name"]
            if key in aggregated:
                aggregated[key]["value_usd"] += h["value_usd"]
                aggregated[key]["shares"]    += h["shares"]
            else:
                aggregated[key] = {**h, "ticker_resolved": ticker}

        holdings_agg = sorted(aggregated.values(), key=lambda x: -x["value_usd"])

        # Resolve tickers and compute portfolio stats
        total_value = sum(h["value_usd"] for h in holdings_agg)
        prev_h      = load_previous_holdings(fund_name)
        cur_values: dict[str, float] = {}

        processed = []
        for h in holdings_agg[:30]:   # top 30 unique positions
            ticker = h.get("ticker_resolved") or resolve_ticker(h["name"], name_map)
            pct    = h["value_usd"] / total_value * 100 if total_value > 0 else 0
            prev_v = prev_h.get(ticker or h["name"], 0)
            if prev_v > 0:
                change = (h["value_usd"] - prev_v) / prev_v
                change_flag = "NEW" if prev_v == 0 else ("ADD" if change > 0.1 else ("TRIM" if change < -0.1 else "HOLD"))
            elif ticker and ticker not in prev_h:
                change_flag = "NEW"
                change = None
            else:
                change_flag = "NEW"
                change = None

            if ticker:
                cur_values[ticker] = h["value_usd"]

            processed.append({
                "ticker":         ticker or "?",
                "name":           h["name"],
                "value_m":        round(h["value_usd"] / 1e6, 1),
                "pct_portfolio":  round(pct, 2),
                "shares":         int(h["shares"]),
                "change_flag":    change_flag,
                "change_pct":     round(change * 100, 1) if change is not None else None,
            })

            flat_rows.append({
                "fund":           fund_name,
                "manager":        info["manager"],
                "style":          info["style"],
                "filing_date":    filing_date,
                "ticker":         ticker or "?",
                "name":           h["name"],
                "value_m":        round(h["value_usd"] / 1e6, 1),
                "pct_portfolio":  round(pct, 2),
                "change_flag":    change_flag,
            })

        save_current_as_previous(fund_name, cur_values)

        # Sector breakdown (from alpha_scores.csv if available)
        sector_alloc: dict[str, float] = {}
        try:
            asc = pd.read_csv(ROOT / "alpha_scores.csv")
            if "sector" in asc.columns and "ticker" in asc.columns:
                sec_map = asc.set_index("ticker")["sector"].to_dict()
                for p in processed:
                    sec = sec_map.get(p["ticker"], "Other")
                    sector_alloc[sec] = sector_alloc.get(sec, 0) + p["pct_portfolio"]
        except Exception:
            pass

        # Top buys and sells
        new_buys = [p for p in processed if p["change_flag"] == "NEW"][:5]
        adds     = [p for p in processed if p["change_flag"] == "ADD"][:5]
        trims    = [p for p in processed if p["change_flag"] == "TRIM"][:5]

        all_funds[fund_name] = {
            "manager":       info["manager"],
            "style":         info["style"],
            "cik":           cik,
            "filing_date":   filing_date,
            "total_aum_m":   round(total_value / 1e6, 0),
            "n_positions":   len(holdings_agg),
            "top_holdings":  processed[:15],
            "new_buys":      new_buys,
            "adds":          adds,
            "trims":         trims,
            "sector_alloc":  sector_alloc,
        }

        ok(f"  {fund_name}: ${total_value/1e9:.1f}B AUM, {len(holdings_agg)} unique positions, "
           f"top={processed[0]['ticker'] if processed else '?'} ({processed[0]['pct_portfolio']:.1f}%)")

    if not all_funds:
        warn("No holdings data fetched — check EDGAR connectivity")
        return

    # ── Consensus analysis: stocks owned by multiple funds ────────────────────
    # Deduplicate per fund (a fund may have multiple entries for same ticker)
    ticker_funds: dict[str, set[str]] = {}
    ticker_total: dict[str, float]    = {}
    ticker_seen: dict[str, set[str]]  = {}   # track (fund, ticker) pairs already added
    for fund_name, fd in all_funds.items():
        for h in fd["top_holdings"]:
            tk = h["ticker"]
            if not tk or tk == "?":
                continue
            key = (fund_name, tk)
            if key in ticker_seen.get(tk, set()):
                continue
            ticker_seen.setdefault(tk, set()).add(key)
            ticker_funds.setdefault(tk, set()).add(fund_name)
            ticker_total[tk] = ticker_total.get(tk, 0) + h["value_m"]

    consensus = sorted(
        [{"ticker": tk, "n_funds": len(fds), "funds": sorted(fds), "total_value_m": round(ticker_total[tk], 1)}
         for tk, fds in ticker_funds.items() if len(fds) >= 2],
        key=lambda x: -x["n_funds"]
    )[:20]

    # ── Canyon overlap ────────────────────────────────────────────────────────
    try:
        canyon_picks = set(pd.read_csv(ROOT / "alpha_scores.csv").nlargest(30, "alpha_score")["ticker"].tolist())
    except Exception:
        canyon_picks = set()

    for c in consensus:
        c["canyon_owns"] = c["ticker"] in canyon_picks

    output = {
        "as_of":     TODAY,
        "n_funds":   len(all_funds),
        "funds":     all_funds,
        "consensus": consensus,
        "canyon_overlap": [c for c in consensus if c.get("canyon_owns")],
    }

    out_json.write_text(json.dumps(output, indent=2, default=str))
    ok(f"famous_holdings.json saved ({len(all_funds)} funds, {len(consensus)} consensus positions)")

    if flat_rows:
        df = pd.DataFrame(flat_rows)
        df.to_csv(ROOT / "famous_holdings.csv", index=False)
        ok(f"famous_holdings.csv → {len(df)} rows")

    # Print consensus table
    print(f"\n  {'Ticker':<8} {'#Funds':>6} {'AUM $':>12}  Funds")
    print(f"  {'─'*8} {'─'*6} {'─'*12}  {'─'*40}")
    for c in consensus[:10]:
        canyon_tag = f" {GREEN}★Canyon{RESET}" if c.get("canyon_owns") else ""
        print(f"  {c['ticker']:<8} {c['n_funds']:>6}  ${c['total_value_m']:>9,.0f}M  "
              f"{', '.join(c['funds'][:3])}{'...' if len(c['funds'])>3 else ''}{canyon_tag}")

    print(f"\n{GREEN}✓ Famous holdings complete{RESET}\n")


if __name__ == "__main__":
    main()
