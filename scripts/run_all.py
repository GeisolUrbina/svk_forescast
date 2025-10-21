
# Kör eSett (månadsbatch), SMHI (timmedel) och kalender enligt config.yaml
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import pandas as pd
import yaml
from src.io_esett import fetch_esett_consumption, dqc_summary, save_partitioned
from src.io_smhi import SMHI
from src.features.calendar import build_calendar


def month_spans(start_iso: str, end_iso: str):
    # Generera [start,end) per månad i UTC
    idx = pd.date_range(start_iso, end_iso, freq="MS", tz="UTC")
    for i, s in enumerate(idx):
        e = idx[i+1] if i+1 < len(idx) else pd.Timestamp(end_iso, tz="UTC")
        yield s.strftime("%Y-%m-%dT%H:%M:%SZ"), e.strftime("%Y-%m-%dT%H:%M:%SZ")


def run_esett(cfg: dict):
    areas = cfg["areas"]; start = cfg["start"]; end = cfg["end"]; res = cfg.get("resolution","PT60M"); outdir = cfg["outdir"]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for a in areas:
        for s,e in month_spans(start, end):
            print(f"[eSett] {a} {s} → {e}")
            df = fetch_esett_consumption(s, e, area=a, resolution=res)
            print("  DQC:", dqc_summary(df).get(a, {}))
            save_partitioned(df, outdir)


def run_smhi(cfg: dict, esett_cfg: dict):
    features = cfg.get("features", ["temp_c"])  # valfritt
    stations = cfg.get("stations", {})
    outdir = cfg.get("outdir", "data/raw/smhi")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    start, end = esett_cfg["start"], esett_cfg["end"]
    client = SMHI()
    for feat in features:
        print(f"[SMHI] {feat} {start} → {end}")
        df = client.fetch_area(start, end, feature=feat, stations_map=stations)
        out = Path(outdir)/f"{feat}_{start[:10]}_{end[:10]}.parquet"
        df.to_parquet(out, index=False)
        print(f"  Saved {len(df)} rows -> {out}")


def run_calendar(cfg: dict, esett_cfg: dict):
    start, end = esett_cfg["start"], esett_cfg["end"]
    out = cfg.get("out", "data/raw/calendar/calendar.parquet")
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    df = build_calendar(start, end, tz_local=cfg.get("tz_local", "Europe/Stockholm"))
    df.to_parquet(out, index=False)
    print(f"[Calendar] Saved {len(df)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_esett(cfg["esett"])     # konsumtion
    run_smhi(cfg.get("smhi", {}), cfg["esett"])   # väder
    run_calendar(cfg.get("calendar", {}), cfg["esett"])  # kalender

if __name__ == "__main__":
    main()