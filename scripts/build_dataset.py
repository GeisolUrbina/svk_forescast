from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _read_parquets(files: List[str], cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=cols or [])
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if cols:
            keep = [c for c in cols if c in df.columns]
            df = df[keep]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=cols or [])


def _ensure_utc(df: pd.DataFrame, col: str = "time_utc") -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _resample_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("time_utc").set_index("time_utc")
    out = df.resample("1h").sum(min_count=1) if how == "sum" else df.resample("1h").mean()
    return out.reset_index()


def _load_esett_processed(esett_dir: str, areas: List[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    files = []
    for a in areas:
        files.extend(glob(str(Path(esett_dir) / f"area={a}" / "year=*" / "month=*" / "part.parquet")))
    df = _read_parquets(files, cols=["time_utc", "area", "value", "unit", "resolution"])
    if df.empty:
        return df
    df = _ensure_utc(df)
    df["time_utc"] = df["time_utc"].dt.floor("h")
    if start:
        df = df[df["time_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["time_utc"] < pd.Timestamp(end, tz="UTC")]
    df = df.groupby(["area", "time_utc"], observed=True, as_index=False)["value"].sum()
    df = df.sort_values(["area", "time_utc"])
    return df


# ------- SMHI helpers -------

_ALIAS = {
    "temp_c":   ["temp_c", "t2m", "air_temperature", "temperature", "temp", "t"],
    "wind_ms":  ["wind_ms", "ws", "wind_speed", "ff", "ff_mean"],
    "precip_mm":["precip_mm", "precipitation", "rr", "rain_mm"],
}
_META = {"year", "month", "lat", "lon", "station_id", "station", "name", "geometry"}


def _pick_value_column(df: pd.DataFrame, feat: str) -> Optional[str]:
    for cand in _ALIAS.get(feat, []):
        if cand in df.columns:
            return cand
    for c in df.columns:
        if c in _META:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _candidate_files_for_feature(smhi_dir: str, feat: str) -> List[str]:
    files = glob(str(Path(smhi_dir) / f"{feat}_*.parquet"))
    if files:
        return files
    return glob(str(Path(smhi_dir) / "*.parquet"))


def _load_smhi_features(smhi_dir: str, features: List[str], areas: List[str],
                        start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out: Optional[pd.DataFrame] = None

    for feat in features:
        files = _candidate_files_for_feature(smhi_dir, feat)
        if not files:
            continue

        dfs_feat = []
        for f in files:
            df = pd.read_parquet(f)

            time_col = None
            for t in ["time_utc", "time", "date", "valid_time"]:
                if t in df.columns:
                    time_col = t
                    break
            if time_col is None:
                continue

            val_col = _pick_value_column(df, feat)
            if val_col is None:
                continue

            df = df.rename(columns={time_col: "time_utc", val_col: feat})
            df = _ensure_utc(df)
            df["time_utc"] = df["time_utc"].dt.floor("h")

            df[feat] = pd.to_numeric(df[feat], errors="coerce")

            keep = ["time_utc", feat] + (["area"] if "area" in df.columns else [])
            df = df[keep]

            if start:
                df = df[df["time_utc"] >= pd.Timestamp(start, tz="UTC")]
            if end:
                df = df[df["time_utc"] < pd.Timestamp(end, tz="UTC")]

            if df.empty:
                continue

            if "area" in df.columns:
                df = df.groupby(["area", "time_utc"], observed=True, as_index=False)[feat].mean(numeric_only=True)
            else:
                df = df.groupby("time_utc", as_index=False)[feat].mean(numeric_only=True)

            how = "sum" if "precip" in feat.lower() else "mean"
            if "area" in df.columns:
                resampled = []
                for a in df["area"].unique():
                    g = df[df["area"] == a][["time_utc", feat]]
                    g = _resample_hourly(g, how=how)
                    g["area"] = a
                    resampled.append(g)
                df = pd.concat(resampled, ignore_index=True)
                df = df[df["area"].isin(areas)]
            else:
                df = _resample_hourly(df[["time_utc", feat]], how=how)
                df = df.merge(pd.DataFrame({"area": areas}), how="cross")

            dfs_feat.append(df)

        if not dfs_feat:
            continue

        feat_df = pd.concat(dfs_feat, ignore_index=True)
        feat_df = feat_df.groupby(["time_utc", "area"], as_index=False)[feat].mean(numeric_only=True)

        out = feat_df if out is None else out.merge(feat_df, on=["time_utc", "area"], how="outer")

    return out if out is not None else pd.DataFrame(columns=["time_utc", "area"] + features)


# ------- Imputering av väder (korttidsluckor) -------

def _impute_weather_short_gaps(
    df: pd.DataFrame,
    areas: List[str],
    cols: List[str] = ["temp_c", "wind_ms"],
    max_gap_hours: int = 3,
) -> pd.DataFrame:
    """
    Fyll små luckor (upp till max_gap_hours) i angivna väderkolumner per area.
    - Interpolation över tid (method='time', limit=H)
    - Därefter kort ffill/bfill (limit=H)
    - 'precip_mm' berörs inte här.
    """
    if df.empty:
        return df
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    out = df.copy()
    out = out.sort_values(["area", "time_utc"])

    for a in areas:
        m = out["area"] == a
        if not m.any():
            continue

        g = out.loc[m, ["time_utc"] + cols].set_index("time_utc").sort_index()

        g_interp = g.copy()
        for c in cols:
            s = g_interp[c]
            s = s.interpolate(method="time", limit=max_gap_hours, limit_direction="both")
            s = s.ffill(limit=max_gap_hours).bfill(limit=max_gap_hours)
            g_interp[c] = s

        if "temp_c" in cols:
            g_interp["temp_c"] = g_interp["temp_c"].clip(lower=-50, upper=50)
        if "wind_ms" in cols:
            g_interp["wind_ms"] = g_interp["wind_ms"].clip(lower=0, upper=60)

        out.loc[m, cols] = g_interp[cols].values

    return out


# ------- Konservativ imputering av nederbörd -------

def _impute_precip_series(s: pd.Series, max_gap: int = 6, zero_gap: int = 3) -> pd.Series:
    """
    Imputera nederbörd (mm/h) konservativt.
    - Om båda gränser runt ett NaN-gap är 0 och gap-längden <= zero_gap -> fyll 0.
    - Annars: interpolera på kumulativ serie inuti luckor upp till max_gap och differensiera tillbaka.
    Skapar inte negativa värden och undviker att 'uppfinna' regn där båda sidor är torra.
    """
    s = s.astype(float)
    out = s.copy()

    # Brygga korta 'torr-luckor' till 0 om båda sidor = 0
    if out.isna().any() and zero_gap > 0:
        na = out.isna().to_numpy()
        i, n = 0, len(out)
        while i < n:
            if not na[i]:
                i += 1
                continue
            j = i
            while j < n and na[j]:
                j += 1
            gap_len = j - i
            left_val = out.iloc[i-1] if i-1 >= 0 else None
            right_val = out.iloc[j] if j < n else None
            if (gap_len <= zero_gap) and ((left_val == 0 or pd.isna(left_val)) and (right_val == 0 or pd.isna(right_val))):
                out.iloc[i:j] = 0.0
            i = j

    if not out.isna().any():
        return out

    #  Kumulativ metod för luckor upp till max_gap (inside only)
    p = out.clip(lower=0)
    c = p.fillna(0).cumsum()
    c_interp = c.interpolate(limit=max_gap, limit_area="inside")

    p_rec = c_interp.diff().clip(lower=0)
    if len(p_rec) > 0:
        p_rec.iloc[0] = p.iloc[0] if not pd.isna(p.iloc[0]) else 0.0

    need = out.isna()
    out[need] = p_rec[need]
    return out


def _load_calendar(calendar_file: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not Path(calendar_file).exists():
        return pd.DataFrame()
    cal = pd.read_parquet(calendar_file)
    time_col = "time_utc" if "time_utc" in cal.columns else ("time" if "time" in cal.columns else None)
    if time_col is None:
        return pd.DataFrame()
    cal = cal.rename(columns={time_col: "time_utc"})
    cal = _ensure_utc(cal)
    cal["time_utc"] = cal["time_utc"].dt.floor("h")
    if start:
        cal = cal[cal["time_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        cal = cal[cal["time_utc"] < pd.Timestamp(end, tz="UTC")]
    drop_cols = [c for c in ["area", "value", "unit", "resolution"] if c in cal.columns]
    cal = cal.drop(columns=drop_cols, errors="ignore")
    return cal


def main():
    ap = argparse.ArgumentParser(description="Bygg EDA-redo dataset (eSett processed + SMHI + Calendar).")
    ap.add_argument("--areas", default="SE1,SE2,SE3,SE4", help="Komma-separerade områden (default: alla)")
    ap.add_argument("--start", default=None, help="Start (ISO, UTC). Om None används minsta gemensamma.")
    ap.add_argument("--end", default=None, help="End [exklusiv] (ISO, UTC). Om None används största gemensamma.")
    ap.add_argument("--esett-dir", default="data/processed/esett", help="Katalog för processed eSett")
    ap.add_argument("--smhi-dir", default="data/raw/smhi", help="Katalog för SMHI-filer")
    ap.add_argument("--calendar-file", default="data/raw/calendar/calendar.parquet", help="Kalenderfil")
    ap.add_argument("--features", default="temp_c,wind_ms,precip_mm", help="SMHI-features att inkludera")
    ap.add_argument("--out", default="data/processed/dataset/hourly_eda.parquet", help="Målfil (parquet)")
    ap.add_argument("--drop-na", action="store_true", help="Droppa rader där någon feature saknas")
    ap.add_argument("--impute-weather", action="store_true", help="Fyll korta luckor i temp_c/wind_ms")
    ap.add_argument("--impute-max-gap", type=int, default=3, help="Max antal timmar att fylla per lucka (default 3)")
    ap.add_argument("--impute-precip", action="store_true", help="Imputera nederbörd konservativt")
    ap.add_argument("--impute-precip-max-gap", type=int, default=6, help="Max gap (timmar) för precip-imputering")
    ap.add_argument("--impute-precip-zero-gap", type=int, default=3, help="Max torr-gap (timmar) som fylls med 0")
    args = ap.parse_args()

    areas = [s.strip() for s in args.areas.split(",") if s.strip()]
    features = [s.strip() for s in args.features.split(",") if s.strip()]

    # eSett
    es = _load_esett_processed(args.esett_dir, areas, args.start, args.end)
    if es.empty:
        raise SystemExit("Hittade ingen processed eSett-data. Kör ingestion först.")

    # SMHI
    smhi = _load_smhi_features(args.smhi_dir, features, areas, args.start, args.end)

    # Kalender
    cal = _load_calendar(args.calendar_file, args.start, args.end)

    # Join
    df = es.copy()
    if not smhi.empty:
        df = df.merge(smhi, on=["time_utc", "area"], how="left")
    if not cal.empty:
        df = df.merge(cal, on="time_utc", how="left")

    # Imputera korta luckor i väder
    if args.impute_weather:
        before = {c: int(df[c].notna().sum()) for c in ["temp_c", "wind_ms"] if c in df.columns}
        df = _impute_weather_short_gaps(df, areas, cols=["temp_c", "wind_ms"], max_gap_hours=args.impute_max_gap)
        after = {c: int(df[c].notna().sum()) for c in ["temp_c", "wind_ms"] if c in df.columns}
        print(f"Imputation väder (<= {args.impute_max_gap}h):", {"före": before, "efter": after})

    # Imputera nederbörd konservativt
    if args.impute_precip and "precip_mm" in df.columns:
        before_p = int(df["precip_mm"].notna().sum())
        df["precip_mm"] = (
            df.groupby("area", observed=True)["precip_mm"]
              .transform(lambda s: _impute_precip_series(s,
                                                         max_gap=args.impute_precip_max_gap,
                                                         zero_gap=args.impute_precip_zero_gap))
        )
        after_p = int(df["precip_mm"].notna().sum())
        print(f"Imputation precip (<= {args.impute_precip_max_gap}h, zero_gap={args.impute_precip_zero_gap}): "
              f"före={before_p}, efter={after_p}")

    # Städning
    df = df.sort_values(["area", "time_utc"]).reset_index(drop=True)
    if args.drop_na:
        df = df.dropna()

    # Skriv
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Summering
    print(f"[OK] Skrev {len(df):,} rader -> {out_path}")
    print("Kolumner:", ", ".join(df.columns))
    print("Tidsintervall:", df['time_utc'].min(), "→", df['time_utc'].max())
    print("Areas:", ", ".join(sorted(df['area'].unique())))
    present = {c: int(df[c].notna().sum()) for c in ["temp_c", "wind_ms", "precip_mm"] if c in df.columns}
    if present:
        print("Icke-NaN per SMHI-kolumn:", present)


if __name__ == "__main__":
    main()
