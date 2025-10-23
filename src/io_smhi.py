# SMHI: förifyllda stations-ID per elområde + timaggregering (mean/sum) över stationer
from __future__ import annotations
from typing import Dict, List
from requests.adapters import HTTPAdapter, Retry

import io
import pandas as pd
import requests

# Stationer per elområde 
STATIONS: Dict[str, List[int]] = {
    "SE1": [159880, 168940, 162860],
    "SE2": [142940, 140480, 135300],
    "SE3": [98230, 97530, 98410],
    "SE4": [52350, 62410, 53430],
}

# SMHI MetObs parameter-id
PARAM_ID = {"temp_c": 1, "wind_ms": 4, "precip_mm": 7}


class SMHI:
    def __init__(self):
        # Session med retry + User-Agent
        s = requests.Session()
        s.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(total=5, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
            ),
        )
        s.headers.update({"Accept": "text/csv, */*", "User-Agent": "svk-forecast/ingest"})
        self.s = s

    def _csv(self, pid: int, sid: int) -> pd.DataFrame:
        """Hämta en station som CSV och returnera (time_utc, value)."""
        url = (
            "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/"
            f"{pid}/station/{sid}/period/corrected-archive/data.csv"
        )
        r = self.s.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), comment="#")

        # Tidskolumn kan heta "DatumTid (UTC)" eller vara första kolumnen.
        ts = pd.to_datetime(
            df.get("DatumTid (UTC)", df.iloc[:, 0]), utc=True, errors="coerce"
        )
        # Värde-kolumn kan heta "Värde"/"Value" eller vara sista kolumnen.
        val_col = "Värde" if "Värde" in df.columns else (
            "Value" if "Value" in df.columns else df.columns[-1]
        )
        val = pd.to_numeric(df[val_col], errors="coerce")

        out = (
            pd.DataFrame({"time_utc": ts, "value": val})
            .dropna(subset=["time_utc"])
            .sort_values("time_utc")
            .reset_index(drop=True)
        )
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
        - precip_mm: tim-SUM över stationer
        """
        # --- Validera input ---
        if feature not in PARAM_ID:
            raise ValueError(f"Okänd feature: {feature}")
        pid = PARAM_ID[feature]

        start = pd.Timestamp(start_utc, tz="UTC")
        end = pd.Timestamp(end_utc, tz="UTC")
        if end <= start:
            raise ValueError("end_utc måste vara > start_utc")

        stations = stations_map or STATIONS

        # --- Hämta & aggregera ---
        frames: list[pd.DataFrame] = []
        for area, ids in stations.items():
            station_frames: list[pd.DataFrame] = []

            for sid in ids:
                try:
                    df = self._csv(pid, sid)
                except requests.RequestException:
                    # Hoppa station som felar
                    continue

                # Filtrera intervall och indexera på tid
                df = df[(df["time_utc"] >= start) & (df["time_utc"] < end)]
                if df.empty:
                    continue
                station_frames.append(
                    df.set_index("time_utc").rename(columns={"value": f"v_{sid}"})
                )

            if not station_frames:
                # Ingen station gav data i intervallet
                continue

            wide = pd.concat(station_frames, axis=1)

            # Resampling per timme: SUM för nederbörd, annars MEAN
            if feature == "precip_mm":
                hourly = wide.resample("1H").sum(min_count=1)
                agg_series = hourly.filter(like="v_").sum(axis=1)
            else:
                hourly = wide.resample("1H").mean()
                agg_series = hourly.filter(like="v_").mean(axis=1)

            out = pd.DataFrame(
                {"time_utc": hourly.index, "area": area, feature: agg_series.values}
            )
            out["area"] = out["area"].astype("category")
            frames.append(out)

        if frames:
            return (
                pd.concat(frames, ignore_index=True)
                .sort_values(["area", "time_utc"])
                .reset_index(drop=True)
            )

        # Tomt resultat med rätt kolumner om inget fanns
        return pd.DataFrame(columns=["time_utc", "area", feature])
