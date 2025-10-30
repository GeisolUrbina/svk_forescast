# src/io_smhi.py
# SMHI: stations-ID per elområde + timaggregering (mean/sum) över stationer
from __future__ import annotations

from typing import Dict, List, Optional
from requests.adapters import HTTPAdapter
from requests import Session
from urllib3.util.retry import Retry
import io
import warnings
import pandas as pd
import requests


# Stationer per elområde
STATIONS = {
    "SE1": [159880, 168940, 162860, 176740, 176760],
    "SE2": [142940, 140480, 135300, 144840, 144860],
    "SE3": [98230, 97530, 98410, 97510, 97150],
    "SE4": [52350, 62410, 53430, 52230, 64810],
}


# API-kod -> SMHI MetObs parameter-id
# t = lufttemperatur, ws = medelvind, rr = nederbördsmängd
API2PID = {"t": 1, "ws": 4, "rr": 7}

# Tillåt både alias och kanoniska namn in/ut
FEATURE_MAP = {
    # temp
    "temp_c": {"api": "t", "out": "temp_c"},
    "t": {"api": "t", "out": "temp_c"},
    # wind
    "wind_ms": {"api": "ws", "out": "wind_ms"},
    "ws": {"api": "ws", "out": "wind_ms"},
    # precip
    "precip_mm": {"api": "rr", "out": "precip_mm"},
    "rr": {"api": "rr", "out": "precip_mm"},
}


class SMHI:
    def __init__(self):
        # HTTP-session med retry och user-agent
        s: Session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.headers.update({"Accept": "text/csv, */*", "User-Agent": "svk-forecast/ingest"})
        self.s = s

    def _csv(self, pid: int, sid: int) -> pd.DataFrame:
        """
        Hämta SMHI MetObs CSV för given parameter- och stations-id och returnera (time_utc, value).
        Robust mot varierande kolumnnamn, decimaltecken och kvalitetskolumner.
        """
        url = (
            "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/"
            f"{pid}/station/{sid}/period/corrected-archive/data.csv"
        )
        r = self.s.get(url, timeout=30)
        r.raise_for_status()

        # SMHI-CSV: semikolon, decimal = ',', '#' kommentarer
        df = pd.read_csv(
            io.StringIO(r.text),
            sep=";",
            decimal=",",
            comment="#",
            engine="python",
            on_bad_lines="skip",
            dtype=str,  # läs som str först så vi själva kan välja numerisk kolumn robust
        )

        if df.empty:
            return pd.DataFrame(columns=["time_utc", "value"])

        # hitta tidskolumn
        time_col: Optional[str] = None
        for c in df.columns:
            cl = c.lower()
            if ("tid" in cl) or ("time" in cl) or ("datum" in cl) or ("date" in cl):
                time_col = c
                break
        if time_col is None:
            time_col = df.columns[0]

        # välj bästa numeriska kolumn (exkludera kvalitet/flag/status/enhet)
        def _is_quality_col(cname: str) -> bool:
            cl = cname.lower()
            return ("kval" in cl) or ("qual" in cl) or ("flag" in cl) or ("status" in cl) or ("unit" in cl)

        candidates = [c for c in df.columns if c != time_col and not _is_quality_col(c)]
        best_col: Optional[str] = None
        best_nonnull = -1
        best_series: Optional[pd.Series] = None

        for c in candidates:
            s = df[c].str.replace(",", ".", regex=False)
            # ersätt ev. U+2212 (minustecken) med ASCII-minus
            s = s.str.replace("−", "-", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            nn = int(num.notna().sum())
            if nn > best_nonnull:
                best_nonnull = nn
                best_col = c
                best_series = num

        if best_col is None or best_series is None or best_nonnull == 0:
            # inget användbart
            return pd.DataFrame(columns=["time_utc", "value"])

        # parse tid – prova fast format först, sedan generiskt
        ts = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")
        if ts.isna().all():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")

        out = (
            pd.DataFrame({"time_utc": ts, "value": best_series})
            .dropna(subset=["time_utc"])
            .sort_values("time_utc")
            .reset_index(drop=True)
        )
        # deduplicera per tid
        out = out.groupby("time_utc", as_index=False)["value"].mean()
        return out

    def fetch_area(
        self,
        start_utc: str,
        end_utc: str,
        feature: str = "temp_c",
        stations_map: Dict[str, List[int]] | None = None,
    ) -> pd.DataFrame:
        """
        Hämta MetObs för valda stationer per område och aggregera till timserie per område.
         - temp_c, wind_ms: tim-MEAN över stationer
         - precip_mm:       tim-SUM  över stationer
        """
        # Normalisera/validera feature och mappa till API-kod + utkolumn
        f = FEATURE_MAP.get(str(feature).lower())
        if f is None:
            raise ValueError(f"Okänd feature: {feature}")
        api_code = f["api"]  # 't' | 'ws' | 'rr'
        out_name = f["out"]  # 'temp_c' | 'wind_ms' | 'precip_mm'
        pid = API2PID[api_code]  # 1 | 4 | 7

        start = pd.Timestamp(start_utc, tz="UTC")
        end = pd.Timestamp(end_utc, tz="UTC")
        if end <= start:
            raise ValueError("end_utc måste vara > start_utc")

        stations = stations_map or STATIONS

        frames: list[pd.DataFrame] = []
        for area, ids in stations.items():
            station_frames: list[pd.DataFrame] = []

            for sid in ids:
                try:
                    df = self._csv(pid, sid)
                except requests.RequestException:
                    # Hoppa station som felar
                    continue

                # Filtrera intervall
                df = df[(df["time_utc"] >= start) & (df["time_utc"] < end)]
                if df.empty:
                    continue

                # Indexera på tid och säkerställ unikt index
                sdf = df.set_index("time_utc").rename(columns={"value": f"v_{sid}"})
                sdf = sdf.groupby(level=0).mean()
                station_frames.append(sdf)

            if not station_frames:
                # Ingen station gav data i intervallet
                continue

            # Sammanfoga stationer på tidsaxeln
            wide = pd.concat(station_frames, axis=1)

            # Resampling per timme
            if out_name == "precip_mm":
                hourly = wide.resample("1h").sum(min_count=1)       # nederbörd = mängd
                agg_series = hourly.filter(like="v_").sum(axis=1)   # SUM över stationer
            else:
                hourly = wide.resample("1h").mean()                 # temp/vind = medel
                agg_series = hourly.filter(like="v_").mean(axis=1)  # MEAN över stationer

            out = pd.DataFrame({"time_utc": hourly.index, "area": area, out_name: agg_series.values})
            out["area"] = out["area"].astype("category")
            frames.append(out)

        if frames:
            return (
                pd.concat(frames, ignore_index=True)
                .sort_values(["area", "time_utc"])
                .reset_index(drop=True)
            )

        # Tomt resultat med rätt schema
        return pd.DataFrame(columns=["time_utc", "area", out_name])
