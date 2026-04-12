import requests
import pandas as pd
from io import StringIO
import time
import numpy as np

BASE_URL = "https://api.statistiken.bundesbank.de/rest/data/BBSIS"

def fetch_series(key, retries=3):
    url = f"{BASE_URL}/{key}"
    params = {"startPeriod": "1997-01-01", "format": "csv"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200 and r.text.strip():
                return r.text
            elif r.status_code == 404:
                return None
        except requests.exceptions.RequestException as e:
            print(f"  Retry {attempt+1} for {key}: {e}")
            time.sleep(2)
    return None

def parse_series(csv_text, col_name):
    """
    Bundesbank CSV format:
      - BOM character at start
      - 7 metadata header lines
      - Data lines: DATE;VALUE;FLAGS  (semicolon-separated)
      - Decimal separator is comma (e.g. 3,45)
      - Missing values are '.'
    """
    lines = csv_text.strip().split("\n")

    # Skip the 7 metadata lines, read the rest
    data_lines = lines[7:]
    if not data_lines:
        return pd.Series(dtype=float, name=col_name)

    records = []
    for line in data_lines:
        line = line.strip().strip('"')
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 2:
            continue
        date_str = parts[0].strip()
        val_str  = parts[1].strip().replace(",", ".")  # comma → dot decimal
        if val_str == "." or val_str == "":
            val = np.nan
        else:
            try:
                val = float(val_str)
            except ValueError:
                val = np.nan
        try:
            date = pd.to_datetime(date_str)
            records.append((date, val))
        except Exception:
            continue

    if not records:
        return pd.Series(dtype=float, name=col_name)

    dates, vals = zip(*records)
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name=col_name, dtype=float)


# ── 1. Svensson parameters ────────────────────────────────────────────────────
PARAM_KEYS = {
    "BETA0": "D.I.ZST.B0.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A",
    "BETA1": "D.I.ZST.B1.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A",
    "BETA2": "D.I.ZST.B2.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A",
    "BETA3": "D.I.ZST.B3.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A",
    "TAU1":  "D.I.ZST.T1.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A",
    "TAU2":  "D.I.ZST.T2.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A",
}

MATURITIES = range(1, 31)

def build_spot_key(mat):
    yy = f"{mat:02d}"
    return f"D.I.ZAR.ZI.EUR.S1311.B.A604.R{yy}XX.R.A.A._Z._Z.A"


# ── 2. Svensson formula ───────────────────────────────────────────────────────
def svensson_spot(m, b0, b1, b2, b3, t1, t2):
    e1 = np.exp(-m / t1)
    e2 = np.exp(-m / t2)
    return (b0
            + b1 * (1 - e1) / (m / t1)
            + b2 * ((1 - e1) / (m / t1) - e1)
            + b3 * ((1 - e2) / (m / t2) - e2))

def svensson_forward(m, b0, b1, b2, b3, t1, t2):
    e1 = np.exp(-m / t1)
    e2 = np.exp(-m / t2)
    return (b0
            + b1 * e1
            + b2 * (m / t1) * e1
            + b3 * (m / t2) * e2)

def spot_to_par(spot_dict, maturities):
    par = {}
    for n in maturities:
        try:
            discount_sum = sum(
                1 / (1 + spot_dict[i] / 100) ** i
                for i in range(1, n + 1)
                if not np.isnan(spot_dict.get(i, np.nan))
            )
            d_n = 1 / (1 + spot_dict[n] / 100) ** n
            par[n] = ((1 - d_n) / discount_sum) * 100 if discount_sum > 0 else np.nan
        except Exception:
            par[n] = np.nan
    return par


# ── 3. Fetch everything ───────────────────────────────────────────────────────
all_series = {}

print("Fetching Svensson parameters...")
params_found = True
for col, key in PARAM_KEYS.items():
    print(f"  {col} ...", end=" ", flush=True)
    raw = fetch_series(key)
    if raw:
        s = parse_series(raw, col)
        if len(s) > 0:
            all_series[col] = s
            print(f"OK ({len(s)} obs)")
        else:
            print("returned 0 obs — key may be wrong")
            params_found = False
    else:
        print("NOT FOUND")
        params_found = False
    time.sleep(0.3)

print("\nFetching zero-coupon spot rates (30 maturities)...")
for mat in MATURITIES:
    key = build_spot_key(mat)
    col = f"SVENY{mat:02d}"
    print(f"  {col} ...", end=" ", flush=True)
    raw = fetch_series(key)
    if raw:
        s = parse_series(raw, col)
        if len(s) > 0:
            all_series[col] = s
            print(f"OK ({len(s)} obs)")
        else:
            print("0 obs")
    else:
        print("missing")
    time.sleep(0.2)


# ── 4. Assemble base DataFrame ────────────────────────────────────────────────
print("\nAssembling DataFrame...")
df = pd.DataFrame(all_series)
df.index.name = "Date"
df = df.sort_index()


# ── 5. Derive SVENF and SVENPY from parameters ────────────────────────────────
if params_found and all(c in df.columns for c in ["BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]):
    print("Computing par yields and forward rates from Svensson parameters...")

    svenf_data  = {f"SVENF{m:02d}":  [] for m in MATURITIES}
    svenpy_data = {f"SVENPY{m:02d}": [] for m in MATURITIES}

    for date, row in df.iterrows():
        b0, b1, b2, b3 = row["BETA0"], row["BETA1"], row["BETA2"], row["BETA3"]
        t1, t2 = row["TAU1"], row["TAU2"]

        if any(pd.isna(v) for v in [b0, b1, b2, b3, t1, t2]) or t1 <= 0 or t2 <= 0:
            for m in MATURITIES:
                svenf_data[f"SVENF{m:02d}"].append(np.nan)
                svenpy_data[f"SVENPY{m:02d}"].append(np.nan)
            continue

        spot_dict = {m: svensson_spot(m, b0, b1, b2, b3, t1, t2) for m in MATURITIES}
        par_dict  = spot_to_par(spot_dict, list(MATURITIES))

        for m in MATURITIES:
            svenf_data[f"SVENF{m:02d}"].append(svensson_forward(m, b0, b1, b2, b3, t1, t2))
            svenpy_data[f"SVENPY{m:02d}"].append(par_dict[m])

    for col, vals in {**svenf_data, **svenpy_data}.items():
        df[col] = vals

else:
    print("WARNING: parameters not found — SVENF and SVENPY will be absent.")
    print("The spot rates (SVENY) are still saved.")


# ── 6. Reorder to match Fed CSV layout ───────────────────────────────────────
param_cols  = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
svenf_cols  = [f"SVENF{m:02d}"  for m in MATURITIES]
svenpy_cols = [f"SVENPY{m:02d}" for m in MATURITIES]
sveny_cols  = [f"SVENY{m:02d}"  for m in MATURITIES]

ordered_cols = [c for c in param_cols + svenf_cols + svenpy_cols + sveny_cols
                if c in df.columns]
df = df[ordered_cols].round(4)


# ── 7. Save ───────────────────────────────────────────────────────────────────
out_path = "bundesbank_yield_curve.csv"
df.to_csv(out_path, na_rep="NA")
print(f"\nDone. {len(df)} rows × {len(df.columns)} columns → {out_path}")
print(df.tail(5).to_string())