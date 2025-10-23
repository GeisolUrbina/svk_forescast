from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import yaml

from src.io_esett import fetch_esett_consumption, dqc_summary, save_partitioned
from src.io_smhi import SMHI
from src.features.calendar import build_calendar


# ---------- Enhetshjälpare ----------

def normalize_unit(df: pd.DataFrame, target_unit: str = "MWh") -> pd.DataFrame:
    """Konvertera till önskad enhet och sätt unit-kolumnen konsekvent."""
    if df.empty:
        return df
    target = (target_unit or "MWh").upper()

    # Sätt default om saknas
    if "unit" not in df.columns or df["unit"].isna().all():
        df["unit"] = pd.Series(["MWh"] * len(df), dtype="category")
    else:
        # Om källan råkar vara kWh → konvertera till MWh
        is_kwh = df["unit"].astype(str).str.upper().eq("KWH")
        if is_kwh.any():
            df.loc[is_kwh, "value"] = df.loc[is_kwh, "value"] / 1000.0
            df.loc[is_kwh, "unit"] = "MWh"

    if target == "KWH":
        # Om man vill ha kWh ut
        df["value"] = df["value"] * 1000.0
        df["unit"] = pd.Series(["kWh"] * len(df), dtype="category")
    else:
        # Standard: MWh
        df["unit"] = pd.Series(["MWh"] * len(df), dtype="category")

    return df


# ---------- Hjälpare ----------

def month_spans(start_iso: str, end_iso: str) -> Iterable[Tuple[str, str, int, int]]:
    """Generera [start,end) per månad i UTC."""
    idx = pd.date_range(start_iso, end_iso, freq="MS", tz="UTC")
    if len(idx) == 0:
        # Om start och end ligger inom samma månad utan MS-gräns, hantera som en enda spann
        s = pd.Timestamp(start_iso, tz="UTC")
        e = pd.Timestamp(end_iso, tz="UTC")
        yield s.strftime("%Y-%m-%dT%H:%M:%SZ"), e.strftime("%Y-%m-%dT%H:%M:%SZ"), s.year, s.month
        return

    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else pd.Timestamp(end_iso, tz="UTC")
        yield s.strftime("%Y-%m-%dT%H:%M:%SZ"), e.strftime("%Y-%m-%dT%H:%M:%SZ"), s.year, s.month


def partition_exists(base_dir: str, area: str, year: int, month: int) -> bool:
    """Enkel idempotens: om part.parquet redan finns för area/year/month skippas hämtning."""
    p = Path(base_dir) / f"area={area}" / f"year={year:04d}" / f"month={month:02d}" / "part.parquet"
    return p.exists()


# ---------- Körsteg ----------

def run_esett(cfg: dict):
    required = {"areas", "start", "end", "outdir"}
    missing = required - set(cfg)
    if missing:
        raise SystemExit(f"[eSett] config saknar nycklar: {sorted(missing)}")

    areas = cfg["areas"]
    start, end = cfg["start"], cfg["end"]
    res = cfg.get("resolution", "PT60M")
    outdir = cfg["outdir"]
    target_unit = cfg.get("unit", "MWh")
    use_abs = bool(cfg.get("abs", False))

    Path(outdir).mkdir(parents=True, exist_ok=True)

    for a in areas:
        for s, e, y, m in month_spans(start, end):
            if partition_exists(outdir, a, y, m):
                logging.info("[eSett] SKIP %s %04d-%02d (partition finns)", a, y, m)
                continue

            logging.info("[eSett]  %s  %s → %s", a, s, e)
            try:
                df = fetch_esett_consumption(s, e, area=a, resolution=res)
            except Exception as ex:
                logging.exception("[eSett] FEL vid hämtning %s %s→%s: %s", a, s, e, ex)
                continue

            if df.empty:
                logging.warning("[eSett] Tomt svar: %s %s→%s (ingen skrivning)", a, s, e)
                continue

            # Normalisering: ev. absolutvärde + enhet
            if use_abs:
                df["value"] = df["value"].abs()
                logging.info("[eSett]  abs(value) tillämpad")

            df = normalize_unit(df, target_unit=target_unit)

            logging.debug("[eSett] DQC %s: %s", a, dqc_summary(df).get(a, {}))
            files = save_partitioned(df, outdir)
            logging.info("[eSett]  Skrivet: %d rader i %d filer",
                         sum(n for _, n in files), len(files))


def run_smhi(cfg: dict, esett_cfg: dict):
    """SMHI är valfri; kör bara om blocket finns i config."""
    if not cfg:
        logging.info("[SMHI] (ingen smhi-konfig hittades, hoppar över)")
        return

    features = cfg.get("features", ["temp_c"])
    stations = cfg.get("stations", {})
    outdir = cfg.get("outdir", "data/raw/smhi")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    start, end = esett_cfg["start"], esett_cfg["end"]
    client = SMHI()

    for feat in features:
        out = Path(outdir) / f"{feat}_{start[:10]}_{end[:10]}.parquet"
        if out.exists():
            logging.info("[SMHI] SKIP %s (fil finns: %s)", feat, out)
            continue

        logging.info("[SMHI]  %s  %s → %s", feat, start, end)
        try:
            df = client.fetch_area(start, end, feature=feat, stations_map=stations)
        except Exception as ex:
            logging.exception("[SMHI] FEL vid hämtning %s: %s", feat, ex)
            continue

        if df.empty:
            logging.warning("[SMHI] Tomt svar för %s – ingen skrivning", feat)
            continue

        df.to_parquet(out, index=False)
        logging.info("[SMHI]  Skrivet: %d rader -> %s", len(df), out)


def run_calendar(cfg: dict, esett_cfg: dict):
    """Bygg kalender-features i samma intervall som eSett."""
    out = cfg.get("out", "data/raw/calendar/calendar.parquet")
    if Path(out).exists():
        logging.info("[Calendar] SKIP (fil finns: %s)", out)
        return

    start, end = esett_cfg["start"], esett_cfg["end"]
    tz_local = cfg.get("tz_local", "Europe/Stockholm")
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)

    logging.info("[Calendar] %s → %s  (tz_local=%s)", start, end, tz_local)
    try:
        df = build_calendar(start, end, tz_local=tz_local)
    except Exception as ex:
        logging.exception("[Calendar] FEL: %s", ex)
        return

    df.to_parquet(out, index=False)
    logging.info("[Calendar] Skrivet: %d rader -> %s", len(df), out)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Kör ingestion enligt config.yaml")
    ap.add_argument("--config", default="config.yaml", help="Sökväg till konfigfil (YAML)")
    ap.add_argument(
        "--steps",
        default="esett,smhi,calendar",
        help="Kommaseparerat: esett, smhi, calendar (default: alla)"
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Loggnivå (default INFO)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Hittar inte konfigfil: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    if "esett" not in cfg:
        raise SystemExit("config.yaml måste innehålla blocket 'esett'.")

    steps = {s.strip().lower() for s in args.steps.split(",") if s.strip()}
    logging.info("Körsteg: %s", ",".join(sorted(steps)))

    if "esett" in steps:
        run_esett(cfg["esett"])
    if "smhi" in steps:
        run_smhi(cfg.get("smhi", {}), cfg["esett"])
    if "calendar" in steps:
        run_calendar(cfg.get("calendar", {}), cfg["esett"])

    logging.info("Klart.")


if __name__ == "__main__":
    main()
