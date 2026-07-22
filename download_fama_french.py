"""
Fama-French 5-Factor + Momentum Data
======================================
Source: Ken French Data Library (free, no registration)
Factors: Rm-Rf, SMB, HML, RMW, CMA, Mom
Used for: Jensen's alpha regression (institutional gold standard)
"""
import warnings, zipfile, io
import numpy as np
import pandas as pd
import requests
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
UA   = {"User-Agent": "canyon-quant research lynnnnnnn958@gmail.com"}

URLS = {
    "ff5_daily": ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                  "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                  "F-F_Research_Data_5_Factors_2x3_daily.csv"),
    "mom_daily": ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                  "F-F_Momentum_Factor_daily_CSV.zip",
                  "F-F_Momentum_Factor_daily.csv"),
    "ff5_monthly": ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                    "F-F_Research_Data_5_Factors_2x3_CSV.zip",
                    "F-F_Research_Data_5_Factors_2x3.csv"),
    "mom_monthly": ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                    "F-F_Momentum_Factor_CSV.zip",
                    "F-F_Momentum_Factor.csv"),
}

def parse_ff_csv(text, freq="daily"):
    """Parse Ken French CSV format: skip header rows, read until empty line."""
    lines  = text.split("\n")
    data_lines = []
    in_data = False
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        # Data rows start with a date (YYYYMMDD or YYYYMM)
        parts = stripped.split(",")
        if len(parts) >= 5:
            try:
                int(parts[0].strip())
                in_data = True
            except ValueError:
                if in_data: break  # end of data section
                continue
        if in_data:
            data_lines.append(stripped)

    if not data_lines:
        return None
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None)
    return df

frames = {}
print("Downloading Fama-French factors...")
for name, (url, csv_name) in URLS.items():
    try:
        resp = requests.get(url, headers=UA, timeout=60)
        if resp.status_code != 200:
            print(f"  {name:<15} FAILED  status={resp.status_code}")
            continue
        z   = zipfile.ZipFile(io.BytesIO(resp.content))
        txt = z.read(csv_name).decode("latin-1")
        freq = "monthly" if "monthly" in name else "daily"
        df  = parse_ff_csv(txt, freq)
        if df is None:
            print(f"  {name:<15} parse failed")
            continue
        # Assign column names
        if "ff5" in name:
            df.columns = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"][:len(df.columns)]
        else:  # momentum
            df.columns = ["date", "Mom", "RF"][:len(df.columns)]
        df = df.dropna()
        df["date"] = df["date"].astype(str).str.strip()
        if freq == "daily":
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"].str[:6], format="%Y%m", errors="coerce")
            df["date"] = df["date"] + pd.offsets.MonthEnd(0)
        df = df.dropna(subset=["date"]).set_index("date")
        # Convert from percent to decimal
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
        frames[name] = df
        print(f"  {name:<15} {len(df):>6} rows  "
              f"[{df.index.min().date()} → {df.index.max().date()}]")
    except Exception as e:
        print(f"  {name:<15} ERROR: {e}")

# Combine daily
if "ff5_daily" in frames and "mom_daily" in frames:
    daily = frames["ff5_daily"].join(frames["mom_daily"][["Mom"]], how="outer")
    daily = daily.sort_index().dropna(subset=["Mkt-RF"])
    daily.to_csv(ROOT / "ff_factors_daily.csv")
    print(f"\nSaved: ff_factors_daily.csv  {daily.shape}")
    print(f"  Columns: {daily.columns.tolist()}")

# Combine monthly
if "ff5_monthly" in frames and "mom_monthly" in frames:
    monthly = frames["ff5_monthly"].join(frames["mom_monthly"][["Mom"]], how="outer")
    monthly = monthly.sort_index().dropna(subset=["Mkt-RF"])
    monthly.to_csv(ROOT / "ff_factors_monthly.csv")
    print(f"Saved: ff_factors_monthly.csv  {monthly.shape}")

print("\nDone.")
