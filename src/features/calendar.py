# Kalenderflaggor i UTC (Sprint 1)
from __future__ import annotations

import pandas as pd


def build_calendar(start_utc: str, end_utc: str, tz_local: str = "Europe/Stockholm") -> pd.DataFrame:
    """
    Bygger timvis kalender för intervallet [start_utc, end_utc) i UTC, med lokala flaggor.
    Kolumner:
      - time_utc (datetime64[ns, UTC])
      - date (lokal kalenderdag, YYYY-MM-DD)
      - hour (lokal timme 0-23)
      - dow (lokal veckodag, mån=0 ... sön=6)
      - month (lokal månad 1-12)
      - is_weekend (lokal helg, lör/sön)
      - is_dst (lokal sommartid)
    """
    # Validera intervall
    start = pd.Timestamp(start_utc, tz="UTC")
    end = pd.Timestamp(end_utc, tz="UTC")
    if end <= start:
        raise ValueError("end_utc måste vara strikt större än start_utc")

    # Timindex i UTC, vänster-inklusive intervall
    idx = pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC")
    if len(idx) == 0:
        # Returnera tom df med rätt schema
        return pd.DataFrame(
            {
                "time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "date": pd.Series(dtype="string"),
                "hour": pd.Series(dtype="int8"),
                "dow": pd.Series(dtype="int8"),
                "month": pd.Series(dtype="int8"),
                "is_weekend": pd.Series(dtype="bool"),
                "is_dst": pd.Series(dtype="bool"),
            }
        )

    df = pd.DataFrame({"time_utc": idx})

    # Konvertera till lokal tid för kalenderflaggor
    local = df["time_utc"].dt.tz_convert(tz_local)

    # Grundflaggor
    df["hour"] = local.dt.hour.astype("int8")
    df["dow"] = local.dt.dayofweek.astype("int8")  # mån=0 ... sön=6
    df["month"] = local.dt.month.astype("int8")
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("bool")  # lör/sön

    # Sommartid (DST) – True om offset != 0
    df["is_dst"] = local.apply(lambda t: (t.dst() or pd.Timedelta(0)) != pd.Timedelta(0)).astype("bool")


    # Praktisk dagskolumn (lokal kalenderdag)
    df["date"] = local.dt.date.astype("string")

    # Säkerställ sortering och monotoni
    df = df.sort_values("time_utc").reset_index(drop=True)

    return df
