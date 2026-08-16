#!/usr/bin/env python3
"""
step_smallcap_form4.py — 抓 S&P 600 小盘的真 Form 4 内部人 *买入*
================================================================
为什么: 内部人开放市场买入 (Form 4 交易代码 P) 是文献里最稳的异象之一
(Lakonishok-Lee; Cohen-Malloy-Pomorski "opportunistic insiders"), 且在
**小盘** 最强 —— 覆盖少、机构因容量做不了。之前仓库里的 insider 数据是大盘 +
yfinance 代理 (假), 这里抓的是 EDGAR 原始 Form 4 XML。

只保留: transactionCode == 'P' 且 AcquiredDisposed == 'A' (公开市场买入)。
记录角色 (CEO/CFO/Officer/Director/10%股东)、股数、价格、金额。

限速: SEC ~10 req/s, 用合规 User-Agent。可断点续传 (已抓的 ticker 跳过)。
输出: smallcap_form4_buys.csv
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).parent
OUT = ROOT / "smallcap_form4_buys.csv"
DELAY = 0.09                       # ~8-9 req/s (under SEC 10/s limit)
UA = "canyon-quant research lynnnnnnn958@gmail.com"
_sn = lambda x: re.sub(r"\{[^}]+\}", "", x)


MAX_FORM4_PER_TICKER = 100000      # effectively no cap — full history (deep mode)
OUT = ROOT / "smallcap_form4_buys_full.csv"   # deep-history output (2yr backup kept separate)


def _new_session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.4,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET"]))
    ad = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    return s


_SESSION = _new_session()


def _get(url, timeout=(6, 12)):
    """Resilient GET with HARD (connect, read) timeouts so it can never hang.
    On failure, rebuild the session once and retry; then give up fast (skip > stall)."""
    global _SESSION
    for _ in range(2):
        try:
            r = _SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception:
            _SESSION = _new_session()
            time.sleep(0.5)
    raise ConnectionError(f"failed after retries: {url}")


def _val(el):
    for s in el.iter():
        if _sn(s.tag) == "value":
            return (s.text or "").strip()
    return ""


def universe():
    df = pd.read_csv(ROOT / "sp600_smallcap_universe.csv")
    return sorted(df["ticker"].astype(str).str.upper().unique())


def cikmap(tks):
    j = json.loads(_get("https://www.sec.gov/files/company_tickers.json").decode())
    m = {e["ticker"].upper(): int(e["cik_str"]) for e in j.values()}
    return {t: m[t] for t in tks if t in m}


def collect_all_form4(cik):
    """All Form 4 (accession, primaryDocument) for a CIK across the FULL filing
    history — the recent block PLUS the older submission shards (filings.files[]),
    which reach back to ~2003. Returns [(accession, primaryDocument), ...]."""
    out = []

    def _harvest(block):
        forms = block.get("form", []); accs = block.get("accessionNumber", [])
        prims = block.get("primaryDocument", [])
        for i, f in enumerate(forms):
            if f == "4" and i < len(accs) and i < len(prims):
                out.append((accs[i], prims[i]))

    j = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").decode())
    time.sleep(DELAY)
    filings = j.get("filings", {})
    _harvest(filings.get("recent", {}))
    for shard in filings.get("files", []):        # older history shards
        name = shard.get("name")
        if not name:
            continue
        try:
            sj = json.loads(_get(f"https://data.sec.gov/submissions/{name}").decode())
            time.sleep(DELAY)
            _harvest(sj)
        except Exception:
            continue
    return out


def parse_form4(cik, acc, prim):
    """Return list of open-market BUY dicts from one Form 4 (code P, AD A)."""
    prim = re.sub(r"^xsl[^/]+/", "", prim)     # strip XSL wrapper → raw xml
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{prim}"
    try:
        body = _get(url).decode(errors="ignore")
    except Exception:
        return []
    if "<ownershipDocument" not in body:
        return []
    try:
        root = ET.fromstring(body)
    except Exception:
        return []
    owner = next((e for e in root.iter() if _sn(e.tag) == "rptOwnerName"), None)
    owner_name = (owner.text or "").strip() if owner is not None else ""
    rel = next((e for e in root.iter() if _sn(e.tag) == "reportingOwnerRelationship"), None)
    is_dir = is_off = is_10 = 0
    title = ""
    if rel is not None:
        for c in rel:
            tg = _sn(c.tag); tx = (c.text or "").strip().lower()
            if tg == "isDirector" and tx in ("1", "true"): is_dir = 1
            elif tg == "isOfficer" and tx in ("1", "true"): is_off = 1
            elif tg == "isTenPercentOwner" and tx in ("1", "true"): is_10 = 1
            elif tg == "officerTitle": title = (c.text or "").strip()
    is_cxo = 1 if re.search(r"chief|CEO|CFO|President", title, re.I) else 0
    out = []
    for tx in root.iter():
        if _sn(tx.tag) != "nonDerivativeTransaction":
            continue
        code = ad = tdate = shares = price = ""
        for el in tx.iter():
            tg = _sn(el.tag)
            if tg == "transactionCode": code = (el.text or "").strip()
            elif tg == "transactionDate": tdate = _val(el)
            elif tg == "transactionShares": shares = _val(el)
            elif tg == "transactionPricePerShare": price = _val(el)
            elif tg == "transactionAcquiredDisposedCode": ad = _val(el)
        if code == "P" and ad == "A":          # open-market purchase
            try:
                sh = float(shares); pr = float(price)
            except Exception:
                continue
            out.append({"date": tdate, "owner": owner_name, "role_dir": is_dir,
                        "role_off": is_off, "role_10pct": is_10, "role_cxo": is_cxo,
                        "title": title, "shares": sh, "price": pr, "value": sh * pr})
    return out


def main():
    tks = universe()
    cm = cikmap(tks)
    print(f"universe {len(tks)} · CIK-mapped {len(cm)}")
    done = set()
    rows = []
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(prev["ticker"].astype(str).str.upper().unique())
        print(f"resume: {len(done)} tickers already done, {len(rows)} buys cached")

    todo = [t for t in tks if t in cm and t not in done]
    print(f"todo: {len(todo)} tickers · DEEP mode (full history via submission shards)", flush=True)
    for k, tk in enumerate(todo, 1):
        cik = cm[tk]
        try:
            f4 = collect_all_form4(cik)          # [(acc, prim), ...] across recent + old shards
            n_buys = 0
            for acc, prim in f4[:MAX_FORM4_PER_TICKER]:
                for b in parse_form4(cik, acc.replace("-", ""), prim):
                    b["ticker"] = tk
                    rows.append(b); n_buys += 1
                time.sleep(DELAY)
            print(f"  [{k}/{len(todo)}] {tk}: {len(f4)} form4 → +{n_buys} buys", flush=True)
        except Exception as e:
            print(f"  [{k}/{len(todo)}] {tk}: ERR {str(e)[:40]}", flush=True)
            continue
        if k % 10 == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n✓ {len(df)} open-market insider BUYS across {df['ticker'].nunique()} small-caps → {OUT.name}")
    if not df.empty:
        d = pd.to_datetime(df["date"], errors="coerce")
        print(f"  date range: {d.min().date()} → {d.max().date()}")
        print(f"  CEO/CFO/Pres buys: {int(df['role_cxo'].sum())} · median buy value ${df['value'].median():,.0f}")


if __name__ == "__main__":
    main()
