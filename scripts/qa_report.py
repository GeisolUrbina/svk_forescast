# Enkel QA: sammanfatta täckning / timmar per månad
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def _read_parquet_dir(base: str) -> pd.DataFrame:
    files = list(Path(base).rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esett", default="data/raw/esett")
    args = ap.parse_args()

    es = _read_parquet_dir(args.esett)
    if es.empty:
        print("No eSett data found"); return
    es["month"] = es["time_utc"].dt.to_period("M")
    cov = es.groupby(["area","month"]).size().rename("rows").reset_index()
    cov["hours_expected"] = cov["month"].dt.to_timestamp("M").dt.days_in_month * 24
    print(cov.head(20))

if __name__ == "__main__":
    main()