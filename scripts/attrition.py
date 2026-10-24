"""Count how many sessions each cleaning rule removes, per site (results/tables/attrition.csv).

Replicates the rules of src/data.py step by step on the raw files.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import MAX_DWELL_H, MIN_DWELL_H, MIN_ENERGY, RAW  # noqa: E402


def raw_tables():
    pa = pd.read_csv(RAW / "paloalto.csv", low_memory=False)
    arr = pd.to_datetime(pa["Start Date"], format="%m/%d/%Y %H:%M", errors="coerce")
    yield "paloalto", len(pa), [
        ("before 2016 (network build-out)", arr >= "2016-01-01"),
    ], pd.DataFrame({"arrival": arr, "departure": arr + pd.to_timedelta(pa["Total Duration (hh:mm:ss)"], errors="coerce"),
                     "energy": pd.to_numeric(pa["Energy (kWh)"], errors="coerce")})

    du = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "dundee_*.csv")))])
    n0 = len(du)
    dedup = ~du.duplicated(subset=["SDR ID"])
    ac = du["Connector Type"].str.strip().str.lower() == "ac"
    yield "dundee", n0, [("duplicate records (overlapping 2024 files)", dedup),
                         ("DC rapid / ultra-rapid connectors", ac)], pd.DataFrame({
        "arrival": pd.to_datetime(du["Start"], format="%d/%m/%Y %H:%M", errors="coerce"),
        "departure": pd.to_datetime(du["End"], format="%d/%m/%Y %H:%M", errors="coerce"),
        "energy": pd.to_numeric(du["Consum(kWh)"], errors="coerce")})

    pe = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "perth_*.csv")))])
    n0 = len(pe)
    acm = pe["Model"].isin(["APT 22kW Dual Outlet", "APT 22kW Raption", "Siemens 7 kW Dual Outlet", "APT 7kW Dual Outlet"])
    dedup = ~pe.duplicated(subset=["CP ID", "Connector", "Start Date", "Start Time"])

    def ts(d, t):
        return pd.to_datetime(pe[d].str.slice(0, 10) + " " + pe[t], format="%Y-%m-%d %H:%M", errors="coerce")
    yield "perth", n0, [("rapid / semi-rapid charger models", acm), ("duplicate records", dedup)], pd.DataFrame({
        "arrival": ts("Start Date", "Start Time"), "departure": ts("End Date", "End Time"),
        "energy": pd.to_numeric(pe["Total kWh"], errors="coerce")})

    tr = pd.read_csv(RAW / "trondheim_sessions.csv", sep=";", decimal=",")
    yield "trondheim", len(tr), [], pd.DataFrame({
        "arrival": pd.to_datetime(tr.Start_plugin, format="%d.%m.%Y %H:%M", errors="coerce"),
        "departure": pd.to_datetime(tr.End_plugout, format="%d.%m.%Y %H:%M", errors="coerce"),
        "energy": pd.to_numeric(tr.El_kWh, errors="coerce")})


if __name__ == "__main__":
    rows = []
    for site, n0, rules, df in raw_tables():
        keep = np.ones(len(df), bool)
        rows.append(dict(site=site, step="raw records", removed=0, remaining=n0))
        for name, mask in rules:
            m = mask.values & keep
            rows.append(dict(site=site, step=name, removed=int(keep.sum() - m.sum()), remaining=int(m.sum())))
            keep = m
        ok = df[["arrival", "departure", "energy"]].notna().all(1).values & keep
        rows.append(dict(site=site, step="unparseable timestamps/energy", removed=int(keep.sum() - ok.sum()), remaining=int(ok.sum())))
        keep = ok
        dwell = ((df.departure - df.arrival).dt.total_seconds() / 3600).values
        m = keep & (dwell >= MIN_DWELL_H)
        rows.append(dict(site=site, step=f"dwell < {MIN_DWELL_H*60:.0f} min", removed=int(keep.sum() - m.sum()), remaining=int(m.sum())))
        keep = m
        m = keep & (df.energy.values >= MIN_ENERGY)
        rows.append(dict(site=site, step=f"energy < {MIN_ENERGY} kWh", removed=int(keep.sum() - m.sum()), remaining=int(m.sum())))
        keep = m
        rows.append(dict(site=site, step=f"dwell > {MAX_DWELL_H:.0f} h (truncated, kept)", removed=0,
                         remaining=int(keep.sum()), note=int((keep & (dwell > MAX_DWELL_H)).sum())))
    out = pd.DataFrame(rows)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "attrition.csv", index=False)
    print(out.to_string())
