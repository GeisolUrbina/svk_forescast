# Strikt schema + validering + partitionerad skrivning
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from pydantic import BaseModel, ValidationError, field_validator

BASE = "https://api.opendata.esett.com"
DEFAULT_TIMEOUT = 30
RATE_LIMIT_SLEEP = 1.2

# --- Pydantic-schema för en rad ---
class ConsumptionRecord(BaseModel):
    time_utc: pd.Timestamp
    area: str
    value: float
    unit: str | None = None
    resolution: str | None = None

    @field_validator("time_utc", mode="before")
    def _to_ts(cls, v):
        return pd.to_datetime(v, utc=True, errors="raise")

# --- Hjälpare ---
def _to_utc_iso(ts: str) -> str:
    # Normalisera till UTC "Z"
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _new_session() -> requests.Session:
    # HTTP-session med retry/backoff
    s = requests.Session()
    retries = Retry(total=6, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=["GET"], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Accept": "application/json"})
    return s

# Explicit fältmappning (minimerar heuristik). Justera vid behov.
FIELD_CANDIDATES = {
    "time": ["startTime", "time", "timestamp", "start", "periodStart"],
    "area": ["mba", "MBA", "area"],
    "value": ["value", "quantity", "consumption"],
    "unit": ["unit", "Unit"],
    "resolution": ["resolution", "Resolution", "granularity"],
}

def _pick(d: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d: return d[k]
    return None

def _records_from_items(items: List[Dict[str, Any]], default_area: str | None = None, default_res: str | None = None) -> List[ConsumptionRecord]:
    recs: List[ConsumptionRecord] = []
    for it in items:
        row = {
            "time_utc": _pick(it, FIELD_CANDIDATES["time"]),
            "area": _pick(it, FIELD_CANDIDATES["area"]) or default_area,
            "value": _pick(it, FIELD_CANDIDATES["value"]),
            "unit": _pick(it, FIELD_CANDIDATES["unit"]),
            "resolution": _pick(it, FIELD_CANDIDATES["resolution"]) or default_res,
        }
        try:
            recs.append(ConsumptionRecord(**row))  # validerar typer och obligatoriska fält
        except ValidationError:
            continue  # hoppa rader som inte validerar
    return recs

def _iterate_pages(url: str, params: Dict[str, Any], session: requests.Session) -> Iterable[List[Dict[str, Any]]]:
    # Stöd både 'next'-URL och 'continuationToken'
    next_url = url
    token_key = "continuationToken"
    token = None
    while True:
        q = dict(params)
        if token: q[token_key] = token
        r = session.get(next_url, params=q, timeout=DEFAULT_TIMEOUT)
        if r.status_code == 429:
            time.sleep(RATE_LIMIT_SLEEP)
            r = session.get(next_url, params=q, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            yield data; break
        items = data.get("results") or data.get("items") or data.get("data") or []
        yield items
        next_url = data.get("next") or next_url
        token = data.get(token_key)
        if not data.get("next") and not token:
            break

def fetch_esett_consumption(start_utc: str, end_utc: str, area: str = "SE3", resolution: str = "PT60M", session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Hämta timvis konsumtion [start,end) för ett område och validera till strikt schema."""
    url = f"{BASE}/EXP15/Consumption"
    params = {"mba": area, "start": _to_utc_iso(start_utc), "end": _to_utc_iso(end_utc), "resolution": resolution}
    s = session or _new_session()
    recs: list[ConsumptionRecord] = []
    for items in _iterate_pages(url, params, s):
        recs.extend(_records_from_items(items, default_area=area, default_res=resolution))
    if not recs:
        return pd.DataFrame(columns=["time_utc","area","value","unit","resolution"]).astype({"time_utc":"datetime64[ns, UTC]"})
    df = pd.DataFrame([r.model_dump() for r in recs]).sort_values("time_utc").reset_index(drop=True)
    return df

# --- DQC (enkla kvalitetskontroller) ---
def dqc_summary(df: pd.DataFrame) -> dict:
    """Returnera enkla DQC-mått per område (täckning, NaN-andel, monotoni)."""
    out: dict[str, dict] = {}
    if df.empty: return out
    for a, d in df.groupby("area", dropna=False):
        out[str(a)] = {
            "rows": int(len(d)),
            "nan_share": float(d["value"].isna().mean()),
            "time_monotonic": bool(d["time_utc"].is_monotonic_increasing),
        }
    return out

# --- Partitionerad skrivning (area/year/month) ---
def save_partitioned(df: pd.DataFrame, base_dir: str) -> list[Tuple[str,int]]:
    """Skriv Parquet partitionerat: base/area=SE?/year=YYYY/month=MM/part.parquet."""
    if df.empty: return []
    df = df.copy()
    df["year"] = df["time_utc"].dt.year
    df["month"] = df["time_utc"].dt.month
    written: list[Tuple[str,int]] = []
    for (area, y, m), part in df.groupby(["area","year","month"], dropna=False):
        path = f"{base_dir}/area={area}/year={y:04d}/month={m:02d}/part.parquet"
        part.drop(columns=["year","month"]).to_parquet(path, index=False)
        written.append((path, len(part)))
    return written