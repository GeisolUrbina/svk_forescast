from __future__ import annotations

import io
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from pydantic import BaseModel, ValidationError, field_validator
from requests.adapters import HTTPAdapter, Retry



# ====== Konstanter & områdeskoder ========


BASE = os.getenv("ESETT_BASE", "https://api.opendata.esett.com")
DEFAULT_TIMEOUT = int(os.getenv("ESETT_TIMEOUT", "30"))
RATE_LIMIT_SLEEP = 1.2
USER_AGENT = os.getenv(
    "USER_AGENT",
    "svk-forecast/ingest (+https://github.com/GeisolUrbina)",
)

# Vissa endpoints använder underscore i områdeskoden (bra att ha men ej kritiskt)
AREA_ALIASES = {"SE1": "SE_1", "SE2": "SE_2", "SE3": "SE_3", "SE4": "SE_4"}

# MBA EIC-koder (SE1–SE4)
AREA_EIC = {
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
}



# ========= Felklass ===========


class EsettAPIError(RuntimeError):
    pass



# ======== Pydantic-schema (rad) ============

class ConsumptionRecord(BaseModel):
    time_utc: datetime  # tz-aware Python datetime
    area: str
    value: float
    unit: str | None = None
    resolution: str | None = None

    @field_validator("time_utc", mode="before")
    def _to_ts(cls, v):
        ts = pd.to_datetime(v, utc=True, errors="raise")
        return ts.to_pydatetime()


# ========== Hjälpfunktioner (HTTP/JSON) =========


def _to_utc_iso(ts: str) -> str:
    """Till UTC med millisekunder: 2023-01-01T00:00:00.000Z"""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _new_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Accept": "application/json", "User-Agent": USER_AGENT})
    return s


FIELD_CANDIDATES = {
    "time": ["timestampUTC", "startTime", "time", "timestamp", "start", "periodStart"],
    "area": ["mba", "MBA", "area", "mga", "MGA"],
    "value": ["total", "value", "quantity", "consumption", "metered", "profiled", "flex"],
    "unit": ["unit", "Unit"],
    "resolution": ["resolution", "Resolution", "granularity"],
}

def _pick(d: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _records_from_items(
    items: List[Dict[str, Any]],
    default_area: str | None = None,
    default_res: str | None = None,
) -> List[ConsumptionRecord]:
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
            recs.append(ConsumptionRecord(**row))
        except ValidationError:
            continue
    return recs


def _iterate_pages(url: str, params: Dict[str, Any], session: requests.Session) -> Iterable[List[Dict[str, Any]]]:
    """
    Paginering med stöd för:
      1) {'results': [...], 'next': 'https://...'}          (next-URL)
      2) {'results': [...], 'continuationToken': 'abc123'}  (token)
      3) [ {...}, {...} ]                                   (ren lista)
    Robust felinfo: inkluderar svarskroppen vid 4xx/5xx.
    """
    next_url = url
    token_key = "continuationToken"
    token = None

    while True:
        q = dict(params)
        if token:
            q[token_key] = token

        r = session.get(next_url, params=q, timeout=DEFAULT_TIMEOUT)

        if r.status_code == 429:
            time.sleep(RATE_LIMIT_SLEEP * (1.0 + 0.5 * random.random()))
            r = session.get(next_url, params=q, timeout=DEFAULT_TIMEOUT)

        if r.status_code >= 400:
            body = (r.text or "")[:500]
            raise EsettAPIError(f"HTTP {r.status_code} GET {next_url} params={q} body={body}")

        try:
            data = r.json()
        except ValueError:
            body = (r.text or "")[:200]
            raise EsettAPIError(f"Oväntat icke-JSON-svar från {next_url} params={q} body={body}")

        if isinstance(data, list):
            yield data
            break

        if not isinstance(data, dict):
            raise EsettAPIError(f"Oväntat svar: typ={type(data)} från {next_url}")

        items = data.get("results") or data.get("items") or data.get("data") or []
        yield items

        next_url = data.get("next") or next_url
        token = data.get(token_key)

        if not data.get("next") and not token:
            break


def _area_to_eic(area: str) -> str:
    a = (area or "").upper().strip()
    if a in AREA_EIC:
        return AREA_EIC[a]
    if a.startswith("10Y") and len(a) >= 16:
        return a
    raise ValueError(f"Okänt område/EIC: {area} (förväntar SE1–SE4 eller giltig EIC)")



# ========== Publika API-funktioner (JSON) ============


def fetch_esett_consumption(
    start_utc: str,
    end_utc: str,
    area: str = "SE3",
    resolution: str = "PT60M",
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Hämta timvis aggregerad konsumtion [start,end) för en MBA via Open Data JSON.
    Provar /EXP15/Consumption och /EXP15/Aggregate och skickar 'mbas' med MBA-EIC.
    Returnerar DataFrame med ['time_utc','area','value','unit','resolution'].
    """
    start_iso = _to_utc_iso(start_utc)
    end_iso = _to_utc_iso(end_utc)
    mba_eic = _area_to_eic(area)

    endpoints = (f"{BASE}/EXP15/Consumption", f"{BASE}/EXP15/Aggregate")
    s = session or _new_session()

    recs: list[ConsumptionRecord] = []
    param_variants: list[dict[str, str] | list[tuple[str, str]]] = [
        [("mbas", mba_eic), ("start", start_iso), ("end", end_iso)],  # list-of-tuples
        {"mbas": mba_eic, "start": start_iso, "end": end_iso},        # dict
        {"MBA":  mba_eic, "start": start_iso, "end": end_iso},        # fallback
    ]

    for url in endpoints:
        found_any = False
        for params in param_variants:
            try:
                q = params if isinstance(params, dict) else dict(params)
                got_items = False
                for items in _iterate_pages(url, q, s):
                    if items:
                        got_items = True
                        recs.extend(_records_from_items(items, default_area=area, default_res=resolution))
                if got_items:
                    found_any = True
                    break
            except EsettAPIError:
                continue
        if found_any:
            break

    if not recs:
        return pd.DataFrame(
            columns=["time_utc", "area", "value", "unit", "resolution"]
        ).astype({"time_utc": "datetime64[ns, UTC]"})

    df = pd.DataFrame([r.model_dump() for r in recs]).sort_values("time_utc").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time_utc", "value"])
    df["area"] = df["area"].astype("category")
    if "resolution" in df:
        df["resolution"] = df["resolution"].astype("category")
    return df


# ========== DQC + partitionerad skrivning ===========

def dqc_summary(df: pd.DataFrame) -> dict:
    """Enkla kvalitetsmått per område."""
    out: dict[str, dict] = {}
    if df.empty:
        return out
    for a, d in df.groupby("area", dropna=False, observed=False):
        out[str(a)] = {
            "rows": int(len(d)),
            "nan_share": float(d["value"].isna().mean()),
            "time_monotonic": bool(d["time_utc"].is_monotonic_increasing),
        }
    return out


def save_partitioned(df: pd.DataFrame, base_dir: str) -> list[Tuple[str, int]]:
    """
    Skriv Parquet partitionerat:
    base/area=SE?/year=YYYY/month=MM/part.parquet (atomic rename).
    """
    if df.empty:
        return []
    df = df.copy()
    df["year"] = df["time_utc"].dt.year
    df["month"] = df["time_utc"].dt.month
    written: list[Tuple[str, int]] = []
    for (area, y, m), part in df.groupby(["area", "year", "month"], dropna=False, observed=False):
        dir_ = f"{base_dir}/area={area}/year={y:04d}/month={m:02d}"
        tmp = f"{dir_}/part.tmp.parquet"
        dst = f"{dir_}/part.parquet"
        os.makedirs(dir_, exist_ok=True)
        part.drop(columns=["year", "month"]).to_parquet(tmp, index=False)
        os.replace(tmp, dst)
        written.append((dst, len(part)))
    return written


#  ============= Diagnostik (MBAOptions) ==========

def list_mba_options(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Lista MBAs (EIC-koder) som API exponeras för EXP15 (om endpointen svarar).
    Tål både list[str] och list[dict], samt loggar exempelnycklar i DEBUG.
    """
    import logging, json

    url = f"{BASE}/EXP15/MBAOptions"
    s = session or _new_session()
    frames: list[pd.DataFrame] = []

    for items in _iterate_pages(url, {}, s):
        if not items:
            continue

        try:
            logging.debug("MBAOptions example item: %s", json.dumps(items[0])[:300])
        except Exception:
            pass

        # Lista av strängar -> anta EIC-koder
        if items and isinstance(items[0], str):
            frames.append(pd.DataFrame({"mba_eic": items}))
            continue

        # Lista av dictar -> mappa vanliga nycklar + spara råpayload
        rows = []
        for it in items:
            if not isinstance(it, dict):
                continue
            rows.append({
                "mba_eic": it.get("eic") or it.get("EIC") or it.get("code") or it.get("Code")
                            or it.get("mba") or it.get("MBA"),
                "name": it.get("name") or it.get("Name") or it.get("description") or it.get("Description"),
                "label": it.get("label") or it.get("Label"),
                "_raw": json.dumps(it),
            })
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["mba_eic", "name", "label", "_raw"])

    out = pd.concat(frames, ignore_index=True)
    if {"mba_eic", "name", "label"}.issubset(out.columns):
        all_na = out[["mba_eic", "name", "label"]].isna().all(axis=1)
        out = out[~all_na].reset_index(drop=True)
    return out



# ============== CSV- & JSON-fallback (UI-export) ================

def parse_esett_csv(content: str, area_label: str) -> pd.DataFrame:
    """
    Parsar CSV (UI-export) och normaliserar till vårt schema.
    Kräver minst en tidskolumn och en värdekolumn (namn autodetekteras).
    """
    df = pd.read_csv(io.StringIO(content))

    time_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "start", "timestamp", "date"])]
    val_cols = [c for c in df.columns if any(k in c.lower() for k in ["value", "consumption", "quantity", "mwh", "kwh"])]

    time_col = time_cols[0] if time_cols else df.columns[0]
    val_col = val_cols[0] if val_cols else df.columns[-1]

    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    val = pd.to_numeric(df[val_col], errors="coerce")

    out = pd.DataFrame({"time_utc": ts, "area": area_label, "value": val})
    out = out.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    out["area"] = out["area"].astype("category")
    out["unit"] = pd.Series(["MWh"] * len(out), dtype="category")
    out["resolution"] = pd.Series(["PT60M"] * len(out), dtype="category")
    return out[["time_utc", "area", "value", "unit", "resolution"]]


def fetch_esett_consumption_from_csvfile(path: str, area: str) -> pd.DataFrame:
    """Läs redan exporterad CSV (från UI) och normalisera."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_esett_csv(content, area_label=area)


def fetch_esett_consumption_from_jsonfile(path: str, area: str) -> pd.DataFrame:
    """
    Läs en sparad JSON-respons (från eSett Open Data) och normalisera till vårt schema.
    Stödjer både dict/list och JSON Lines (en JSON per rad).
    """
    import json

    # Försök auto-detektera JSONL (en JSON per rad)
    with open(path, "r", encoding="utf-8") as f:
        first_chunk = f.read(2048)
        f.seek(0)
        items: list[dict] = []

        if "\n" in first_chunk and not first_chunk.strip().startswith("{") and not first_chunk.strip().startswith("["):
            # JSON Lines
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, list):
                        items.extend(obj)
                    elif isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    continue
        else:
            data = json.load(f)
            if isinstance(data, dict):
                items = data.get("results") or data.get("items") or data.get("data") or []
            elif isinstance(data, list):
                items = data
            else:
                items = []

    recs = _records_from_items(items, default_area=area, default_res="PT60M")
    if not recs:
        return pd.DataFrame(
            columns=["time_utc", "area", "value", "unit", "resolution"]
        ).astype({"time_utc": "datetime64[ns, UTC]"})

    df = pd.DataFrame([r.model_dump() for r in recs]).sort_values("time_utc").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time_utc", "value"])
    df["area"] = df["area"].astype("category")
    if "resolution" in df:
        df["resolution"] = df["resolution"].astype("category")
    return df
