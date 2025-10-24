from __future__ import annotations
from pathlib import Path
import pandas as pd

RAW_CAL = Path("data/raw/calendar/calendar.parquet")
RAW_SMHI_DIR = Path("data/raw/smhi")
PROC_ESETT_DIR = Path("data/processed/esett")
OUT = Path("data/processed/training/training.parquet")

def read_smhi() -> pd.DataFrame:
    # Antag att fetch_area skrev per-feature-filer med kolumner: time_utc, area, <feature>
    feats = []
    for name in ("temp_c", "wind_ms", "precip_mm"):
        f = RAW_SMHI_DIR / f"{name}_2023-01-01_2025-10-20.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            if "time_utc" in df:
                df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
            feats.append(df.set_index(["time_utc","area"])[[name]])
    if not feats:
        return pd.DataFrame()
    smhi = pd.concat(feats, axis=1).reset_index()
    return smhi

def read_calendar() -> pd.DataFrame:
    cal = pd.read_parquet(RAW_CAL)
    if "time_utc" in cal:
        cal["time_utc"] = pd.to_datetime(cal["time_utc"], utc=True)
    return cal

def read_esett_processed() -> pd.DataFrame:
    parts = list(PROC_ESETT_DIR.rglob("part.parquet"))
    dfs = []
    for p in parts:
        df = pd.read_parquet(p)
        # säkerställ schema
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        dfs.append(df[["time_utc","area","value"]])
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def main():
    PROC_ESETT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    load = read_esett_processed()
    if load.empty:
        raise SystemExit("No processed eSett found.")

    smhi = read_smhi()
    cal = read_calendar()

    # Joina per area & time_utc (inner för att bara behålla kompletta rader)
    df = load.copy()
    if not smhi.empty:
        df = df.merge(smhi, on=["time_utc","area"], how="inner")
    if not cal.empty:
        df = df.merge(cal, on="time_utc", how="left")  # kalender är samma för alla areas

    # Sortera + enklare quality flaggar
    df = df.sort_values(["area","time_utc"]).reset_index(drop=True)
    df["is_na_any"] = df.isna().any(axis=1)

    df.to_parquet(OUT, index=False)
    print(f"Saved training frame: {OUT}  rows={len(df)}  cols={len(df.columns)}")
    print("Sample:")
    print(df.head())

if __name__ == "__main__":
    main()
