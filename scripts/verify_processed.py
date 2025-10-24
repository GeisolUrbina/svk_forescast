from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path("data/processed/esett")

def expected_hours_dynamic(df: pd.DataFrame) -> int:
    # Räkna förväntade timmar från månadens början till min(månadens slut, sista observation + 1h)
    tmin = pd.to_datetime(df["time_utc"], utc=True).min()
    tmax = pd.to_datetime(df["time_utc"], utc=True).max()
    month_start = pd.Timestamp(year=tmin.year, month=tmin.month, day=1, tz="UTC")
    month_end = (month_start + pd.offsets.MonthBegin(1))  # [start, month_end)
    # om filen slutar innan månadens slut – acceptera partiell månad
    effective_end = min(month_end, tmax + pd.Timedelta(hours=1))
    return int((effective_end - month_start) / pd.Timedelta(hours=1))

def check_file(p:Path) -> list[str]:
    issues = []
    df = pd.read_parquet(p)
    req = {"time_utc","value","area"}
    if not req.issubset(df.columns):
        issues.append("missing required columns")
        return issues

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)

    # timsteg?
    step = df["time_utc"].sort_values().diff().dropna().min()
    if step != pd.Timedelta(hours=1):
        issues.append(f"min step != 1h (got {step})")

    # NaN?
    if df["value"].isna().any():
        issues.append("value has NaN")

    # enhet + resolution (om finns)
    if "unit" in df and not df["unit"].astype(str).str.upper().eq("MWH").all():
        issues.append("unit not MWh")
    if "resolution" in df and not df["resolution"].astype(str).eq("PT60M").all():
        issues.append("resolution not PT60M")

    # dynamisk radräkning (acceptera partiell månad)
    exp = expected_hours_dynamic(df)
    if len(df) not in (exp, exp-1, exp-2):  # liten tolerans
        issues.append(f"rows={len(df)} expected≈{exp}")

    return issues

def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE
    files = sorted(base.rglob("part.parquet"))
    if not files:
        print(f"No files under {base}")
        sys.exit(1)

    bad = 0
    for p in files:
        issues = check_file(p)
        if issues:
            bad += 1
            print(f"[WARN] {p}: " + " | ".join(issues))

    if bad == 0:
        print("✅ All processed hourly files look good (including partial final month).")
    else:
        print(f"⚠️  {bad} file(s) with issues.")

if __name__ == "__main__":
    main()
