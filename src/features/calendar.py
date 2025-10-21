# Kalenderflaggor i UTC
from __future__ import annotations
import pandas as pd

def build_calendar(start_utc: str, end_utc: str, tz_local: str="Europe/Stockholm") -> pd.DataFrame:
    idx = pd.date_range(start_utc, end_utc, freq="1H", inclusive="left", tz="UTC")
    df = pd.DataFrame({"time_utc": idx})
    local = df["time_utc"].dt.tz_convert(tz_local)
    df["hour"], df["dow"], df["month"] = local.dt.hour, local.dt.dayofweek, local.dt.month
    df["is_weekend"] = df["dow"].isin([5,6])
    df["is_dst"] = (local.dt.dst().fillna(pd.Timedelta(0)) != pd.Timedelta(0))
    return df