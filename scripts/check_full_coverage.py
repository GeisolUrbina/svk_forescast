#helhetskontroll av processed (per area)
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

def load_from_config(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    e = cfg["esett"]
    areas = e.get("areas", ["SE1","SE2","SE3","SE4"])
    start = pd.Timestamp(e["start"], tz="UTC")
    end   = pd.Timestamp(e["end"], tz="UTC")
    base  = Path(e.get("processed_outdir", "data/processed/esett"))
    return areas, start, end, base

def main():
    ap = argparse.ArgumentParser(description="Kontrollera full timtäckning i processed/esett")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--areas", nargs="*", help="T.ex. SE3 SE4 (om du vill överskriva config)")
    ap.add_argument("--start", help="ISO-tid, t.ex. 2023-01-01T00:00:00Z")
    ap.add_argument("--end", help="ISO-tid, t.ex. 2025-10-20T00:00:00Z")
    ap.add_argument("--base", default=None, help="Basdir, default från config: data/processed/esett")
    args = ap.parse_args()

    # Hämta standarder från config.yaml
    areas, start, end, base = load_from_config(Path(args.config))

    # Manuella overrides
    if args.areas: areas = args.areas
    if args.start: start = pd.Timestamp(args.start, tz="UTC")
    if args.end:   end   = pd.Timestamp(args.end, tz="UTC")
    if args.base:  base  = Path(args.base)

    full_range = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    ok = True

    for a in areas:
        area_dir = base / f"area={a}"
        try:
            df = pd.read_parquet(area_dir, engine="pyarrow")
        except FileNotFoundError:
            print(f"[MISS] {a}: Hittar inte {area_dir}")
            ok = False
            continue

        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        df = df[(df["time_utc"] >= start) & (df["time_utc"] < end)]

        got = pd.Index(df["time_utc"].sort_values().unique())
        missing = full_range.difference(got)

        print(f"{a}: hours={len(got)}/{len(full_range)}, missing={len(missing)}")
        if len(missing):
            ok = False
            # visa några exempel om det saknas
            sample = list(missing[:10])
            if sample:
                print("  first missing examples:", sample)

    print("\n✅ FULLTÄCKT!" if ok else "\n⚠️ Det saknas timmar – se raderna ovan.")

if __name__ == "__main__":
    main()
