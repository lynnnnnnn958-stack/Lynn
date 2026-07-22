"""
Improvement 5 — SEC Form 4 Insider Transaction Download (Fast)
================================================================
Approach:
  1. Download 34 quarterly form.idx index files (fast: ~34 requests)
  2. Build per-company, per-quarter filing list
  3. For each company × quarter, sample the FIRST Form 4 filing
  4. Download XML directly (1 request per filing, no index page needed)
  5. Parse buy vs sell direction

Limited to top-N companies by filing count to stay under 10 minutes.
Output: insider_transactions.csv
"""
import re, time, warnings, json, random
import numpy as np
import pandas as pd
import requests
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT  = Path(__file__).parent
DELAY = 0.12   # 8 req/sec

UA = {"User-Agent": "canyon-quant research lynnnnnnn958@gmail.com",
      "Accept-Encoding": "gzip, deflate"}

def get(url, as_json=False):
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code == 200:
            return r.json() if as_json else r.text
    except Exception:
        pass
    return None

# ── Load universe ─────────────────────────────────────────────────────
print("Loading ticker universe...")
edgar_df  = pd.read_csv(ROOT/"edgar_fundamentals.csv")
tickers   = sorted(edgar_df["ticker"].unique())

print("Fetching CIK map...")
cik_raw = get("https://www.sec.gov/files/company_tickers.json", as_json=True)
time.sleep(DELAY)
cik_map = {}
if cik_raw:
    for entry in cik_raw.values():
        tk = entry["ticker"].upper()
        cik_map[tk] = str(entry["cik_str"])   # integer string, no padding
universe_ciks = {tk: cik_map[tk] for tk in tickers if tk in cik_map}
cik_to_ticker = {v: k for k, v in universe_ciks.items()}
print(f"  {len(universe_ciks)} tickers mapped")

# ── Download form.idx files and build per-company index ──────────────
print("\nDownloading form.idx quarterly index files...")

def parse_form_idx_line(line):
    """Parse form.idx line using whitespace splitting (not fixed-width).
    Company names can contain spaces, so take last 3 tokens: CIK, date, path.
    """
    try:
        parts = line.split()
        if len(parts) < 4: return None
        cik    = parts[-3]   # integer string WITHOUT leading zeros
        date_s = parts[-2]
        fpath  = parts[-1]
        if not (cik.isdigit() and len(date_s) == 10 and fpath.startswith("edgar/")):
            return None
        return cik, date_s, fpath
    except Exception:
        return None

# company_filings: {cik_int_str: [(date, fpath), ...]}
company_filings = {}   # key = cik without padding

for year in range(2018, 2027):
    for qtr in range(1, 5):
        if year == 2026 and qtr > 2: break
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"
        raw = get(url)
        time.sleep(DELAY)
        if not raw:
            print(f"  {year} Q{qtr}: failed")
            continue
        count = 0
        for line in raw.splitlines()[10:]:
            if not (line[:2] == "4 " or line[:4] == "4/A "):
                continue
            res = parse_form_idx_line(line)
            if not res: continue
            cik_str, date_s, fpath = res
            # Match without padding
            cik_noz = cik_str   # already non-padded from form.idx
            if cik_noz not in cik_to_ticker: continue
            company_filings.setdefault(cik_noz, []).append((date_s, fpath))
            count += 1
        print(f"  {year} Q{qtr}: {count} Form 4s for our universe  "
              f"({len(company_filings)} unique companies)")

total_filings = sum(len(v) for v in company_filings.values())
print(f"\nTotal indexed: {total_filings} filings across {len(company_filings)} companies")

# ── Select top companies by filing count (most active insiders) ───────
sorted_companies = sorted(company_filings.items(), key=lambda x: len(x[1]), reverse=True)
TOP_N = 120
top_companies = sorted_companies[:TOP_N]
print(f"\nFocusing on top {TOP_N} most-active companies")
# Show top 5
for cik, filings in top_companies[:5]:
    tk = cik_to_ticker.get(cik, "?")
    print(f"  {tk}: {len(filings)} filings")

# ── Build quarterly sample: 1 filing per company × quarter ───────────
def quarter_key(date_s):
    """Returns (year, quarter) from 'YYYY-MM-DD'."""
    try:
        y, m = int(date_s[:4]), int(date_s[5:7])
        return (y, (m-1)//3+1)
    except Exception:
        return (0, 0)

def build_xml_url(fpath):
    """Construct Form 4 XML URL directly from form.idx file path.

    fpath: 'edgar/data/1084869/0001084869-18-000001.txt'
    The filename already has dashes. Nodash version = remove dashes.
    XML dir:  edgar/data/1084869/000108486918000001/
    XML file: edgar/data/1084869/000108486918000001/0001084869-18-000001.xml
    """
    try:
        parts      = fpath.replace(".txt","").split("/")
        cik_int    = parts[2]                    # "1084869"
        acc_dashed = parts[3]                    # "0001084869-18-000001"
        acc_nodash = acc_dashed.replace("-","")  # "000108486918000001"
        # All Form 4 XMLs are named form4.xml inside their accession directory
        return (f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{acc_nodash}/form4.xml")
    except Exception:
        return None

def parse_form4_xml(xml_text):
    """Returns list of (date_str, net_shares) for common stock transactions.

    Handles two XML schemas:
      v1: <transactionCode><value>S</value></transactionCode>
      v2: <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
    """
    results = []
    blocks  = re.findall(r'<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>',
                         xml_text, re.DOTALL)
    for blk in blocks:
        # Schema v1: <transactionCode><value>S</value></transactionCode>
        code_m = re.search(r'<transactionCode>\s*<value>([A-Z])</value>', blk, re.DOTALL)
        if not code_m:
            # Schema v2: <transactionCode>S</transactionCode> (direct, no value child)
            code_m = re.search(r'<transactionCode>\s*([A-Z])\s*</transactionCode>', blk, re.DOTALL)

        shrs_m = re.search(r'<transactionShares>.*?<value>([\d.,]+)</value>', blk, re.DOTALL)
        disp_m = re.search(r'<transactionAcquiredDisposedCode>.*?<value>([AD])</value>', blk, re.DOTALL)
        date_m = re.search(r'<transactionDate>.*?<value>(\d{4}-\d{2}-\d{2})</value>', blk, re.DOTALL)
        if not date_m:
            date_m = re.search(r'<transactionDate>\s*(\d{4}-\d{2}-\d{2})\s*</transactionDate>', blk, re.DOTALL)

        if not (code_m and shrs_m and date_m): continue
        code = code_m.group(1)
        # Keep open-market buys (P), open-market sales (S), and grants/awards (A)
        # Exclude: F=tax-withholding, M=option-exercise, G=gift, D=disposition-to-issuer
        if code not in ('P', 'S', 'A'): continue
        try:
            shares = float(shrs_m.group(1).replace(',',''))
        except ValueError:
            continue
        disp = disp_m.group(1) if disp_m else ('A' if code in ('P', 'A') else 'D')
        sign = 1 if disp == 'A' else -1
        # Open-market buys (P) weight 1.0; grants/awards (A) weight 0.5 (less informative)
        weight = 1.0 if code == 'P' else (0.5 if code == 'A' else 1.0)
        results.append((date_m.group(1), shares * sign * weight))
    return results

# ── Download and parse sampled filings ───────────────────────────────
print(f"\nDownloading Form 4 XMLs (1 per company × quarter)...")
records     = []
n_ok, n_fail = 0, 0

for rank, (cik_noz, filings) in enumerate(top_companies):
    ticker = cik_to_ticker.get(cik_noz, "?")

    # Group by quarter, pick first filing per quarter
    by_qtr = {}
    for date_s, fpath in sorted(filings):
        qk = quarter_key(date_s)
        if qk not in by_qtr:
            by_qtr[qk] = (date_s, fpath)

    for (yr, qtr_n), (date_s, fpath) in by_qtr.items():
        xml_url = build_xml_url(fpath)
        if not xml_url: continue

        xml_text = get(xml_url)
        time.sleep(DELAY)

        if not xml_text or "<ownershipDocument>" not in xml_text:
            # Try adding the issuer trading symbol prefix to filename
            n_fail += 1
            continue

        txns = parse_form4_xml(xml_text)
        for d, net_s in txns:
            records.append(dict(ticker=ticker, date=d, net_shares=net_s))
        n_ok += 1

    if (rank+1) % 20 == 0:
        print(f"  [{rank+1}/{TOP_N}] {ticker}: ok={n_ok} fail={n_fail} txns={len(records)}")

print(f"\nCompleted: {n_ok} filings parsed, {n_fail} failed, {len(records)} transactions")

# ── Aggregate and compute quarterly signal ────────────────────────────
if len(records) >= 50:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker","date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly = df.groupby(["ticker","month"])["net_shares"].sum().reset_index()
    monthly["date"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.sort_values(["ticker","date"])
    monthly["roll3m"] = (monthly.groupby("ticker")["net_shares"]
                         .transform(lambda x: x.rolling(3, min_periods=1).sum()))
    monthly["rank"] = monthly.groupby("date")["roll3m"].rank(pct=True)
    monthly["direction"] = pd.cut(monthly["rank"],
                                   bins=[0, 0.33, 0.67, 1.01],
                                   labels=["SELL","NEUTRAL","BUY"])
    out = monthly[["ticker","date","roll3m","rank","direction"]].copy()
    out.to_csv(ROOT/"insider_transactions.csv", index=False)
    print(f"Saved: insider_transactions.csv ({len(out)} rows, {out['ticker'].nunique()} tickers)")

    # IC Analysis
    print("\nInsider Signal IC Analysis:")
    # Load price data to compute forward returns
    prices_df = pd.read_csv(ROOT/"sp500_price_8yr.csv", index_col=0, parse_dates=True)
    prices_df = prices_df.drop(columns=["SPY"], errors="ignore")

    monthly_ic = []
    for m in sorted(out["date"].unique()):
        m_dt    = pd.Timestamp(m)
        m_next  = m_dt + pd.DateOffset(months=1)
        factors = out[out["date"]==m_dt].set_index("ticker")["rank"]

        # Forward 21-day return
        if m_dt in prices_df.index and m_next in prices_df.index:
            fwd_r = prices_df.loc[m_next] / prices_df.loc[m_dt] - 1
            common = factors.index.intersection(fwd_r.index)
            if len(common) >= 5:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factors[common].values, fwd_r[common].values)
                if np.isfinite(ic):
                    monthly_ic.append(ic)

    if monthly_ic:
        ic_arr = np.array(monthly_ic)
        print(f"  Mean IC:    {ic_arr.mean():+.4f}")
        print(f"  IC Std:     {ic_arr.std():.4f}")
        print(f"  t-stat:     {ic_arr.mean()/ic_arr.std()*np.sqrt(len(ic_arr)):+.2f}")
        print(f"  Hit rate:   {(ic_arr>0).mean():.0%}")
        verdict = "SIGNIFICANT" if abs(ic_arr.mean()/ic_arr.std()*np.sqrt(len(ic_arr))) > 2.0 else "NOT SIGNIFICANT"
        print(f"  Verdict:    {verdict}")
else:
    print("Insufficient data - creating neutral placeholder")
    dates = pd.date_range("2018-01-01", "2026-06-01", freq="MS")
    rows  = [dict(ticker=tk, date=d, roll3m=0.0, rank=0.5, direction="NEUTRAL")
             for tk in tickers for d in dates]
    pd.DataFrame(rows).to_csv(ROOT/"insider_transactions.csv", index=False)
    print(f"  Neutral placeholder saved")

print("\nDone.")
