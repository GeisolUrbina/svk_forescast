# Ingest + DQC + partitionerad skrivning
from __future__ import annotations
import argparse
from pathlib import Path
from src.io_esett import fetch_esett_consumption, dqc_summary, save_partitioned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--area", default="SE3")
    ap.add_argument("--resolution", default="PT60M")
    ap.add_argument("--outdir", default="data/raw/esett")
    args = ap.parse_args()

    df = fetch_esett_consumption(args.start, args.end, area=args.area, resolution=args.resolution)

    # DQC – skriv ut snabb summering
    summary = dqc_summary(df)
    print("DQC:", summary.get(args.area, {}))

    # Partitionerad skrivning
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    files = save_partitioned(df, args.outdir)
    total = sum(n for _, n in files)
    print(f"Saved {total} rows into {len(files)} partitions under {args.outdir}")

if __name__ == "__main__":
    main()