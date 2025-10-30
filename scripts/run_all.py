from __future__ import annotations

import argparse
import logging
import os
from glob import glob
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
        df["value"] = df["value"] * 1000.0
        df["unit"] = pd.Series(["kWh"] * len(df), dtype="category")
    else:
        df["unit"] = pd.Series(["MWh"] * len(df), dtype="category")

    return df


# ---------- helpers för completeness + tim-aggregering ----------

def _infer_step_minutes(df: pd.DataFrame) -> int:
    """Gissa minsta tidssteg i minuter (t.ex. 15 eller 60)."""
    if df.empty or "time_utc" not in df:
        return 60
    d = df["time_utc"].sort_values().diff().dropna()
    if d.empty:
        return 60
    step = d.min()
    return int(step / pd.Timedelta(minutes=1))


def _expected_rows(span_start: pd.Timestamp, span_end: pd.Timestamp, step_min: int) -> int:
    """Beräkna förväntat antal punkter i [start, end) givet steg i minuter."""
    total_minutes = int((span_end - span_start) / pd.Timedelta(minutes=1))
    step_min = max(1, int(step_min))
    return total_minutes // step_min


def _log_completeness(df: pd.DataFrame, span_start: pd.Timestamp, span_end: pd.Timestamp, label: str):
    """Logga enkel completeness-status för en månad/area."""
    if df.empty:
        logging.info("[eSett] COMPLETENESS %s: 0/0 (empty)", label)
        return
    step = _infer_step_minutes(df)
    exp = _expected_rows(span_start, span_end, step)
    got = len(df)
    tol = 4 if step <= 15 else 1  # liten tolerans
    status = "ok" if got >= exp - tol else "incomplete"
    logging.info("[eSett] COMPLETENESS %s: %d / %d rows (step=%d min) -> %s", label, got, exp, step, status)


def _write_hourly_processed(df: pd.DataFrame, out_base: str) -> list[tuple[str, int]]:
    """
    Skriv tim-aggregerad version av df till processed-katalog med samma partitionering.
    15-min → tim-SUM. Om redan 60-min → resample('1h').sum() (ofarligt, 1 värde/timme).
    """
    if df.empty:
        return []
    df = df.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time_utc", "value"])

    written: list[tuple[str, int]] = []
    for area, g in df.groupby("area", dropna=False, observed=False):
        g = g.sort_values("time_utc").set_index("time_utc")
        hourly = g["value"].resample("1h").sum(min_count=1).to_frame("value").reset_index()
        hourly["area"] = area
        hourly["unit"] = pd.Series(["MWh"] * len(hourly), dtype="category")
        hourly["resolution"] = pd.Series(["PT60M"] * len(hourly), dtype="category")

        # partitionera på year/month
        hourly["year"] = hourly["time_utc"].dt.year
        hourly["month"] = hourly["time_utc"].dt.month
        for (y, m), part in hourly.groupby(["year", "month"], dropna=False):
            dir_ = Path(out_base) / f"area={area}" / f"year={y:04d}" / f"month={m:02d}"
            tmp = dir_ / "part.tmp.parquet"
            dst = dir_ / "part.parquet"
            dir_.mkdir(parents=True, exist_ok=True)
            part = part.drop(columns=["year", "month"])
            part.to_parquet(tmp, index=False)
            os.replace(tmp, dst)
            written.append((str(dst), len(part)))
    return written


def processed_partition_path(base_dir: str, area: str, year: int, month: int) -> Path:
    return Path(base_dir) / f"area={area}" / f"year={year:04d}" / f"month={month:02d}" / "part.parquet"


def processed_exists(base_dir: str, area: str, year: int, month: int) -> bool:
    return processed_partition_path(base_dir, area, year, month).exists()


# ---------- Hjälpare ----------

def month_spans(start_iso: str, end_iso: str) -> Iterable[Tuple[str, str, int, int]]:
    """Generera [start,end) per månad i UTC."""
    idx = pd.date_range(start_iso, end_iso, freq="MS", tz="UTC")
    if len(idx) == 0:
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
    outdir = cfg["outdir"]  # RAW
    processed_outdir = cfg.get("processed_outdir", "data/processed/esett")
    target_unit = cfg.get("unit", "MWh")
    use_abs = bool(cfg.get("abs", False))
    reprocess = bool(cfg.get("reprocess", False))

    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(processed_outdir).mkdir(parents=True, exist_ok=True)

    for a in areas:
        for s, e, y, m in month_spans(start, end):
            raw_part = Path(outdir) / f"area={a}" / f"year={y:04d}" / f"month={m:02d}" / "part.parquet"
            proc_part = processed_partition_path(processed_outdir, a, y, m)

            # --- Om RAW finns: hoppa hämtning men bygg processed vid behov ---
            if raw_part.exists():
                logging.info("[eSett] SKIP %s %04d-%02d (raw finns)", a, y, m)

                # Bygg processed om saknas eller om reprocess=True
                if reprocess or not proc_part.exists():
                    try:
                        df_raw = pd.read_parquet(raw_part)
                        if df_raw.empty:
                            continue
                        if use_abs:
                            df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce").abs()
                        df_raw = normalize_unit(df_raw, target_unit=target_unit)

                        # Completeness rapport
                        span_start = pd.Timestamp(s, tz="UTC")
                        span_end = pd.Timestamp(e, tz="UTC")
                        _log_completeness(df_raw, span_start, span_end, label=f"{a} {s[:7]}")

                        w = _write_hourly_processed(df_raw, processed_outdir)
                        if w:
                            logging.info("[eSett]  Processed hourly: %s %04d-%02d -> %s",
                                         a, y, m, proc_part)
                    except Exception as ex:
                        logging.exception("[eSett] FEL vid processed-bygg %s %04d-%02d: %s", a, y, m, ex)
                continue

            # --- Vanlig väg: hämta från API eftersom RAW saknas ---
            logging.info("[eSett]  %s  %s → %s", a, s, e)
            try:
                df = fetch_esett_consumption(s, e, area=a, resolution=res)
            except Exception as ex:
                logging.exception("[eSett] FEL vid hämtning %s %s→%s: %s", a, s, e, ex)
                continue

            if df.empty:
                logging.warning("[eSett] Tomt svar: %s %s→%s (ingen skrivning)", a, s, e)
                continue

            if use_abs:
                df["value"] = pd.to_numeric(df["value"], errors="coerce").abs()
                logging.info("[eSett]  abs(value) tillämpad")

            df = normalize_unit(df, target_unit=target_unit)

            logging.debug("[eSett] DQC %s: %s", a, dqc_summary(df).get(a, {}))
            files = save_partitioned(df, outdir)
            tot_raw = sum(n for _, n in files)
            logging.info("[eSett]  Skrivet: %d rader i %d filer", tot_raw, len(files))

            # Completeness-rapport för månaden
            span_start = pd.Timestamp(s, tz="UTC")
            span_end = pd.Timestamp(e, tz="UTC")
            _log_completeness(df, span_start, span_end, label=f"{a} {s[:7]}")

            # Tim-aggregerad "processed"-skrivning
            w = _write_hourly_processed(df, processed_outdir)
            if w:
                logging.info("[eSett]  Processed hourly: %d rader till %s",
                             sum(n for _, n in w), processed_outdir)


def run_smhi(cfg: dict, esett_cfg: dict):
    """SMHI är valfri; kör bara om blocket finns i config."""
    if not cfg:
        logging.info("[SMHI] (ingen smhi-konfig hittades, hoppar över)")
        return

    features = cfg.get("features", ["temp_c"])
    stations = cfg.get("stations", {})
    outdir = cfg.get("outdir", "data/raw/smhi")
    reprocess = bool(cfg.get("reprocess", False)) or bool(esett_cfg.get("reprocess", False))
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Alias → standardkolumn (så vi kan hoppa innan vi hämtar)
    alias_to_std = {
        "temp_c": "temp_c", "t": "temp_c",
        "wind_ms": "wind_ms", "ws": "wind_ms",
        "precip_mm": "precip_mm", "rr": "precip_mm",
    }

    start, end = esett_cfg["start"], esett_cfg["end"]
    client = SMHI()

    for feat in features:
        std_name = alias_to_std.get(feat, feat)
        expected_path = Path(outdir) / f"{std_name}_{start[:10]}_{end[:10]}.parquet"

        # Skip-logik: om fil med standardnamnet finns och vi inte reprocessar
        if expected_path.exists() and not reprocess:
            logging.info("[SMHI] SKIP %s (fil finns: %s)", feat, expected_path)
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

        # Hitta kolumnnamnet som faktiskt kom tillbaka
        smhi_cols = [c for c in df.columns if c not in ("time_utc", "area")]
        if len(smhi_cols) != 1:
            logging.error("[SMHI] oväntade kolumner: %s", df.columns.tolist())
            continue
        out_col = smhi_cols[0]

        # Spara alltid med standardnamnet (temp_c / wind_ms / precip_mm)
        if out_col != std_name:
            df = df.rename(columns={out_col: std_name})

        out = expected_path
        if reprocess and out.exists():
            try:
                out.unlink()
            except OSError:
                pass

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
