# SMHI: förifyllda stations-ID per elområde + timmedel över stationer
from __future__ import annotations
import io
from datetime import datetime, timezone
from typing import Dict, List
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# Exempelstationer (2–4 per område). Byt gärna till er preferens.
STATIONS: Dict[str, List[int]] = {
    "SE1": [159880, 168940, 162860],
    "SE2": [142940, 140480, 135300],
    "SE3": [98230, 97530, 98410],
    "SE4": [52350, 62410, 53430],
}

PARAM_ID = {"temp_c": 1, "wind_ms": 4, "precip_mm": 7}

class SMHI:
    def __init__(self):
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])) )
        s.headers.update({"Accept": "text/csv, */*"})
        self.s = s

    def _csv(self, pid: int, sid: int) -> pd.DataFrame:
        url = (
            "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/"
            f"{pid}/station/{sid}/period/corrected-archive/data.csv"
        )
        r = self.s.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), comment="#")
        ts = pd.to_datetime(df.get("DatumTid (UTC)", df.iloc[:,0]), utc=True, errors="coerce")
        val_col = "Värde" if "Värde" in df.columns else ("Value" if "Value" in df.columns else df.columns[-1])
        val = pd.to_numeric(df[val_col], errors="coerce")
        return pd.DataFrame({"time_utc": ts, "value": val}).dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)

    def fetch_area(self, start_utc: str, end_utc: str, feature: str = "temp_c", stations_map: Dict[str, List[int]] | None = None) -> pd.DataFrame:
        pid = PARAM_ID[feature]
        start = datetime.fromisoformat(start_utc.replace("Z","+00:00")).astimezone(timezone.utc)
        end = datetime.fromisoformat(end_utc.replace("Z","+00:00")).astimezone(timezone.utc)
        stations = stations_map or STATIONS
        frames = []
        for area, ids in stations.items():
            parts = []
            for sid in ids:
                df = self._csv(pid, sid)
                df = df[(df["time_utc"]>=start) & (df["time_utc"]<end)]
                parts.append(df.set_index("time_utc").rename(columns={"value": f"v_{sid}"}))
            if not parts: continue
            wide = pd.concat(parts, axis=1)
            hourly = wide.resample("1H").mean()  # timmedel över stationer
            hourly["area"] = area
            hourly = hourly.reset_index()
            hourly[feature] = hourly.filter(like="v_").mean(axis=1)
            frames.append(hourly[["time_utc","area",feature]])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time_utc","area",feature])