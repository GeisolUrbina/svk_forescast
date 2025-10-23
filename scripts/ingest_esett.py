# Ingest + DQC + partitionerad skrivning
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.io_esett import (
    fetch_esett_consumption,
    fetch_esett_consumption_from_csvfile,
    fetch_esett_consumption_from_jsonfile,
    dqc_summary,
    save_partitioned,
)


def _partition_exists(base: str, area: str, start_iso: str) -> bool:
    """Kollar om månadspartitionen redan finns (idempotens)."""
    ts = pd.Timestamp(start_iso, tz="UTC")
    p = Path(base) / f"area={area}" / f"year={ts.year:04d}" / f"month={ts.month:02d}" / "part.parquet"
    return p.exists()


def main():
    ap = argparse.ArgumentParser(description="Ingest eSett consumption (JSON API, CSV-export eller lokal JSON).")
    ap.add_argument("--start", required=True, help="Starttid i UTC, t.ex. 2023-01-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="Sluttid i UTC, t.ex. 2023-01-02T00:00:00Z")
    ap.add_argument("--area", default="SE3", help="SE1–SE4 (eller EIC).")
    ap.add_argument("--resolution", default="PT60M", help="Behålls i output (skickas ej i API-query).")
    ap.add_argument("--outdir", default="data/raw/esett", help="Basutdata (Parquet, partitionerat).")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--unit", default="MWh",
                    help="Tvinga/slå fast enhet i output (t.ex. MWh eller kWh). Konverterar till MWh om behövs.")
    ap.add_argument("--abs", dest="abs_value", action="store_true",
                    help="Ta absolutvärde på 'value' innan skrivning.")

    # Datakälla
    ap.add_argument(
        "--source",
        choices=["json", "csv", "jsonfile"],
        default="json",
        help="Välj källa: 'json' (API), 'csv' (UI-export) eller 'jsonfile' (sparad JSON).",
    )
    ap.add_argument("--csv-file", help="Sökväg till CSV-export (om --source=csv)")
    ap.add_argument("--csv-glob", help=r"Globmönster för CSV, t.ex. 'exports\*.csv'")
    ap.add_argument("--json-file", help="Sökväg till JSON-fil (om --source=jsonfile)")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if _partition_exists(args.outdir, args.area, args.start):
        logging.info("SKIP %s %s (partition finns redan)", args.area, args.start[:7])
        return

    logging.info("Hämtar %s  %s → %s", args.area, args.start, args.end)

    # --- Källa: JSON (API) / CSV (lokal) / JSON (lokal) ---
    if args.source == "csv":
        csv_path: Path | None = None

        if args.csv_file:
            csv_path = Path(args.csv_file)
        elif args.csv_glob:
            matches = sorted(Path(".").rglob(args.csv_glob))
            if not matches:
                logging.error("Ingen fil matchar --csv-glob: %s", args.csv_glob)
                logging.error("Tips: lista CSV-filer med:  Get-ChildItem -Recurse -Filter *.csv | Select FullName")
                raise SystemExit(1)
            csv_path = matches[-1]  # ta den 'senaste' i sorteringen
            logging.info("Hittade CSV via glob: %s", csv_path)
        else:
            raise SystemExit("Du valde --source=csv men angav varken --csv-file eller --csv-glob")

        if not csv_path.exists():
            logging.error("CSV-filen finns inte: %s", csv_path)
            logging.error("Tips: kontrollera sökvägen eller använd --csv-glob för att hitta filen.")
            raise SystemExit(1)

        df = fetch_esett_consumption_from_csvfile(str(csv_path), area=args.area)

    elif args.source == "jsonfile":
        if not args.json_file:
            raise SystemExit("Du valde --source=jsonfile men angav inte --json-file")
        json_path = Path(args.json_file)
        if not json_path.exists():
            raise SystemExit(f"JSON-filen finns inte: {json_path}")
        df = fetch_esett_consumption_from_jsonfile(str(json_path), area=args.area)

    else:
        # JSON via Open Data API (EXP15).
        df = fetch_esett_consumption(args.start, args.end, area=args.area, resolution=args.resolution)

    # --- Normalisering efter källa: filtrera intervall + ev. absolutvärde ---
    if df.empty:
        logging.warning("Tomt svar – ingen skrivning")
        return

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    before = len(df)
    df = df[(df["time_utc"] >= start) & (df["time_utc"] < end)].copy()
    logging.info("Filtrerade på intervallet [%s, %s): %d → %d rader", args.start, args.end, before, len(df))

    if args.abs_value:
        df["value"] = df["value"].abs()
        logging.info("Använde --abs: satte value = abs(value)")

    # --- Enhet: konvertera och sätt unit-kolumn ---
    target_unit = (args.unit or "MWh").upper()

    if "unit" in df.columns and df["unit"].notna().any():
        # Om källan säger kWh → konvertera till MWh
        if df["unit"].astype(str).str.upper().eq("KWH").any():
            df["value"] = df["value"] / 1000.0
            logging.info("Konverterade värden från kWh till MWh")
    else:
        # Sätt default MWh om saknas
        df["unit"] = pd.Series(["MWh"] * len(df), dtype="category")

    # Tvinga önskad slut-enhet
    if target_unit == "MWH":
        is_kwh = df["unit"].astype(str).str.upper().eq("KWH")
        if is_kwh.any():
            df.loc[is_kwh, "value"] = df.loc[is_kwh, "value"] / 1000.0
        df["unit"] = pd.Series(["MWh"] * len(df), dtype="category")

    elif target_unit == "KWH":
        df["value"] = df["value"] * 1000.0
        df["unit"] = pd.Series(["kWh"] * len(df), dtype="category")

    else:
        logging.warning("Okänd --unit=%s, lämnar värden som är", target_unit)

    # --- DQC + partitionerad skrivning ---
    logging.info("DQC: %s", dqc_summary(df).get(args.area, {}))
    files = save_partitioned(df, args.outdir)
    total = sum(n for _, n in files)
    logging.info("Saved %d rows into %d partitions under %s", total, len(files), args.outdir)


if __name__ == "__main__":
    main()
